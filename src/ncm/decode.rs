//! Decoding strategies for NCM level-wise generators.
//!
//! Plain NCM decoding makes one target call per level, coarse to fine: the
//! codes of level `k` are drawn from `target.level_logits(k, ẑ^(k), prompt)`,
//! independently per code, and the emitted tokens are the argmax of the
//! codec's decoder on the final reconstruction `ẑ^(K)`. Every strategy in this
//! module is an optional layer over that loop. Each is a field of
//! [`DecodeConfig`], they compose freely, and with all of them off
//! ([`DecodeConfig::default`]) the result *is* plain decoding.
//!
//! | strategy | field | what it does |
//! |---|---|---|
//! | plain sampling | [`DecodeConfig::sampler`] | greedy, temperature and top-k per code |
//! | level early exit | [`DecodeConfig::early_exit`] | stop after level `k` when the codec's decoder is confident at every position of `ẑ^(k)` |
//! | level speculation | [`DecodeConfig::speculation`] | a draft model proposes several levels, the target verifies them; output distribution is the target's exactly |
//! | immune repair | [`DecodeConfig::repair`] | re-encode the text so far, rewrite coarse codes the tokenizer disagrees with and the target was unsure of, regenerate what depended on them |
//! | particles | [`DecodeConfig::particles`] | `N` sequences per prompt, resampled between levels, final pick by minimum Bayes risk |
//! | outline cache | [`DecodeConfig::outline_cache`] | reuse the coarse levels decoded earlier for the same prompt and model, with a seeded bypass |
//!
//! # Order of operations
//!
//! Each step of the level loop is: (cache, once, before the loop) → one plain
//! level *or* one speculative span of levels → a repair round → an early-exit
//! probe → particle resampling. Early exit and resampling are therefore
//! decided at step boundaries; after a speculative span the exit is checked
//! once, at the end of the span.
//!
//! # Cost is reported, not promised
//!
//! "Never lose speed" is a design goal, not a guarantee. Every strategy draws
//! from one [`DecodeBudget`], and every output carries a [`DecodeReport`] that
//! counts what was actually spent: target forward passes, draft passes, codec
//! calls, probes, repairs. A multi-level verification counts as one target
//! pass only when the target says [`LevelModel::batches_levels`]; otherwise
//! the verification is made level by level and costs exactly what plain
//! decoding of those levels costs, so speculation against a non-batching
//! target saves no target calls and adds the draft's. The report never
//! contains a "speed-up" figure — compare `level_calls` with the number of
//! levels yourself, on your hardware.
//!
//! # Randomness
//!
//! All draws come from ChaCha8 streams derived from [`DecodeConfig::seed`],
//! one stream per purpose (codes, draft, acceptance, resampling, repair,
//! cache bypass). A strategy that is off draws nothing, so turning one on
//! never perturbs the draws of the others; this is what makes "disabled ⇒
//! identical output" an exact statement.
//!
//! # Honesty
//!
//! These strategies are certified against small test doubles (`dblocks
//! verify`, group `ncm-decode`): the certificates prove the *mechanics* —
//! exactness of speculative sampling, identities when a strategy is off,
//! budget enforcement, honest counting. None of them has been run on a
//! trained nina model. Whether early exit, speculation, repair, particles or
//! the outline cache improve speed or quality there is unmeasured; such a
//! claim needs a measurement in this crate first.

use std::collections::HashMap;

use anyhow::{anyhow, bail, ensure, Context, Result};
use burn::tensor::{backend::Backend, Int, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::{ConceptCodec, LevelModel};

mod certify;
mod doubles;

pub(crate) use certify::certificate_checks;

// ------------------------------------------------------------------ config --

/// How each code is drawn from a level's logits. Codes within a level are
/// drawn independently, as the generator's factorised output says they are.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CodeSampler {
    /// Softmax temperature. `0` is greedy: the argmax, ties to the lowest
    /// code index.
    pub temperature: f64,
    /// Keep only the `k` most likely codes (ties to the lower index) before
    /// sampling; `None` keeps all.
    pub top_k: Option<usize>,
}

impl Default for CodeSampler {
    fn default() -> Self {
        Self::greedy()
    }
}

impl CodeSampler {
    /// Deterministic argmax decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
        }
    }

    /// Ancestral sampling at `temperature`, no top-k.
    pub fn with_temperature(temperature: f64) -> Self {
        Self {
            temperature,
            top_k: None,
        }
    }

    /// Whether codes are the argmax rather than a draw.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "sampler.temperature must be finite and >= 0 (0 is greedy), got {}",
            self.temperature
        );
        if let Some(k) = self.top_k {
            ensure!(k >= 1, "sampler.top_k must be at least 1 when set, got {k}");
        }
        Ok(())
    }

    /// The normalised distribution a code is drawn from at one position,
    /// written into `out`: one-hot at the argmax when greedy, otherwise the
    /// temperature softmax over the top-k codes.
    fn distribution(&self, logits: &[f32], out: &mut Vec<f64>) {
        out.clear();
        out.resize(logits.len(), 0.0);
        if self.is_greedy() {
            if let Some(slot) = out.get_mut(argmax(logits)) {
                *slot = 1.0;
            }
            return;
        }
        let mut keep = vec![true; logits.len()];
        if let Some(k) = self.top_k.filter(|&k| k < logits.len()) {
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            keep.fill(false);
            for &index in order.iter().take(k) {
                keep[index] = true;
            }
        }
        let max = logits
            .iter()
            .zip(&keep)
            .filter(|(_, &kept)| kept)
            .map(|(&l, _)| f64::from(l))
            .fold(f64::NEG_INFINITY, f64::max);
        for ((slot, &l), &kept) in out.iter_mut().zip(logits).zip(&keep) {
            if kept {
                *slot = ((f64::from(l) - max) / self.temperature).exp();
            }
        }
        // The maximum kept term is exp(0) = 1, so the total is at least 1.
        let total: f64 = out.iter().sum();
        for slot in out.iter_mut() {
            *slot /= total;
        }
    }
}

/// Level early exit ("tuning fork": short answers skip the fine levels).
///
/// After a step leaves `k` levels decoded (`min_levels ≤ k < K`), the codec
/// decodes `ẑ^(k)`; if at every position of every sequence the most likely
/// token has probability at least `threshold`, decoding stops and emits those
/// tokens. NCM's tokenizer is trained with nested dropout precisely so that
/// `ẑ^(k)` is a usable partial reconstruction. A threshold above 1 can never
/// fire (a probability is at most 1), which reproduces plain decoding
/// exactly; each probe costs one codec decode, drawn from
/// [`DecodeBudget::max_probes`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EarlyExit {
    /// `θ`: the per-position confidence every position must reach.
    pub threshold: f64,
    /// Fewest levels decoded before an exit is allowed, at least 1.
    pub min_levels: usize,
}

/// Level speculation ("draft and verify").
///
/// The draft model proposes `draft_levels` levels in sequence (each drafted
/// level conditions the next, exactly as the target would see it); the
/// target then verifies them. Per code this is standard speculative sampling
/// — accept the drafted code `d` with probability `min(1, p(d)/q(d))`, else
/// draw from `normalise(max(0, p − q))` — so every code is distributed as the
/// target's `p`. Level `k+1` is conditioned on level `k`, so only the longest
/// prefix of fully accepted levels is kept: the first level with any
/// rejection is emitted with its resampled codes and every drafted level
/// after it is discarded. Greedy decoding is the special case of one-hot
/// `p` and `q`: a code is accepted iff the two argmaxes agree, and the output
/// is plain greedy target decoding bit for bit.
///
/// Several sequences (a batch, or particles) share one level loop, so the
/// kept prefix is the shortest over the sequences. This preserves the
/// target's distribution: whether a sequence's later levels are kept depends
/// only on the other sequences' independent draws.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Speculation {
    /// Levels the draft proposes per verification, at least 1.
    pub draft_levels: usize,
}

/// Immune repair ("rest day: declare good or redo that level").
///
/// After each step with at least two levels decoded, the text decoded from
/// the current reconstruction is re-encoded by the codec, and each strictly
/// coarser level's codes are compared with the generated ones. A position is
/// *flagged* when the re-encoded code differs **and** the target's
/// probability for the generated code is below `confidence_guard` — the
/// autoimmunity guard: an unusual code the target was confident in is never
/// attacked. The coarsest level with a flag is repaired: one target call on
/// its (unchanged) conditioning gives the target's probability for each
/// re-encoded alternative, a flagged code is replaced by its alternative when
/// that probability is at least `min_alternative_probability`, and every
/// finer level is regenerated for the sequences that changed (one call per
/// level). Codes taken from the outline cache count as fully confident.
///
/// Why re-encode rather than re-quantise against
/// [`super::SoftConceptCodec::level_codebook`]: re-quantising a coarse level
/// against the current residual needs the codec's down-sampling and
/// projection from the latent `ẑ` to a level's code space, which the codec
/// traits do not expose. Re-encoding uses only [`ConceptCodec::encode`], i.e.
/// the frozen tokenizer's own quantiser, so a disagreement means exactly "the
/// tokenizer would have chosen a different coarse code for the text the finer
/// levels now spell". Each check costs one codec decode (shared with an
/// early-exit probe at the same step) and one encode, drawn from
/// [`DecodeBudget::max_probes`]; a repair round runs only if the budget can
/// still pay for every remaining plain level afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Repair {
    /// Flag only codes whose target probability is below this. `0` flags
    /// nothing; above `1` flags every disagreement, even a certain code.
    pub confidence_guard: f64,
    /// Replace a flagged code only if the target gives the re-encoded
    /// alternative at least this probability, in `[0, 1]`.
    pub min_alternative_probability: f64,
}

/// Particles ("weather: hold a cloud of outlines, resolve late").
///
/// `count` sequences per prompt are decoded in parallel through the batch
/// dimension (one target call per level serves all of them). Each particle
/// carries a log-weight that grows by `tilt · Σ log p_target(code)` over the
/// codes it draws (`p_target` at temperature 1, no top-k). When a prompt's
/// effective sample size `1 / Σ w²` drops below `ess_threshold · count`, its
/// particles are resampled systematically (seeded) and their weights reset.
/// This is sequential Monte Carlo whose target is the *tilted* distribution
/// `∝ Π_k p(level k | coarser)^(1 + tilt)` — sharper than the model, not the
/// model itself; `tilt = 0` never resamples, and is `count` independent
/// samples. The final sequence per prompt is the minimum-Bayes-risk choice:
/// the particle minimising the weight-averaged token Hamming distance to the
/// others (ties to the lowest index). `count = 1` is plain sampling exactly.
/// A repair adjusts a particle's weight by the change in its log-probability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Particles {
    /// Particles per prompt, at least 1; clamped to
    /// [`DecodeBudget::max_particles`].
    pub count: usize,
    /// Weight exponent on the target's log-probability, finite and `>= 0`.
    pub tilt: f64,
    /// Resample when the effective sample size falls below this fraction of
    /// `count`, in `[0, 1]`.
    pub ess_threshold: f64,
}

/// Outline cache ("erosion: well-worn channels", with a "flash flood"
/// override).
///
/// The first `levels` levels decoded for a prompt are stored in an
/// [`OutlineCache`] under the exact prompt tokens, `model_id`, the level
/// schedule and the codebook size; a later decode of the same prompt with the
/// same model reuses them and skips those target calls (reported as
/// `cached_levels`). With probability `bypass_probability` (seeded) the cache
/// is ignored and the outline regenerated and stored afresh, so decoding is
/// not trapped in one groove. An outline is never shared across model ids.
/// All sequences of a batch must hit for any call to be skipped; with
/// particles, every particle of a prompt starts from that prompt's outline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlineCacheUse {
    /// Coarse levels to store and reuse, in `1..=K`.
    pub levels: usize,
    /// Probability of ignoring a hit and regenerating, in `[0, 1]`.
    pub bypass_probability: f64,
    /// Identity of the target model; outlines never cross ids.
    pub model_id: String,
}

/// Which strategies run, and with what settings. The default is plain greedy
/// decoding with every strategy off.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DecodeConfig {
    /// Per-code sampling rule.
    pub sampler: CodeSampler,
    /// Seed of every random stream.
    pub seed: u64,
    /// Level early exit; `None` decodes every level.
    pub early_exit: Option<EarlyExit>,
    /// Level speculation; needs a draft model ([`Decoder::with_draft`]).
    pub speculation: Option<Speculation>,
    /// Immune repair.
    pub repair: Option<Repair>,
    /// Particles.
    pub particles: Option<Particles>,
    /// Outline cache; needs an [`OutlineCache`] passed to [`Decoder::decode`].
    pub outline_cache: Option<OutlineCacheUse>,
}

impl DecodeConfig {
    /// Check every field against a schedule of `levels` levels, naming the
    /// field at fault.
    pub fn validate(&self, levels: usize) -> Result<()> {
        ensure!(
            levels >= 1,
            "decoding needs a schedule with at least one level"
        );
        self.sampler.validate()?;
        if let Some(exit) = &self.early_exit {
            ensure!(
                exit.threshold.is_finite() && exit.threshold >= 0.0,
                "early_exit.threshold must be finite and >= 0, got {}",
                exit.threshold
            );
            ensure!(
                (1..=levels).contains(&exit.min_levels),
                "early_exit.min_levels must be in 1..={levels}, got {}",
                exit.min_levels
            );
        }
        if let Some(spec) = &self.speculation {
            ensure!(
                spec.draft_levels >= 1,
                "speculation.draft_levels must be at least 1, got {}",
                spec.draft_levels
            );
        }
        if let Some(repair) = &self.repair {
            ensure!(
                repair.confidence_guard.is_finite() && repair.confidence_guard >= 0.0,
                "repair.confidence_guard must be finite and >= 0, got {}",
                repair.confidence_guard
            );
            ensure!(
                (0.0..=1.0).contains(&repair.min_alternative_probability),
                "repair.min_alternative_probability must be in [0, 1], got {}",
                repair.min_alternative_probability
            );
        }
        if let Some(particles) = &self.particles {
            ensure!(
                particles.count >= 1,
                "particles.count must be at least 1, got {}",
                particles.count
            );
            ensure!(
                particles.tilt.is_finite() && particles.tilt >= 0.0,
                "particles.tilt must be finite and >= 0, got {}",
                particles.tilt
            );
            ensure!(
                (0.0..=1.0).contains(&particles.ess_threshold),
                "particles.ess_threshold must be in [0, 1], got {}",
                particles.ess_threshold
            );
        }
        if let Some(cache) = &self.outline_cache {
            ensure!(
                (1..=levels).contains(&cache.levels),
                "outline_cache.levels must be in 1..={levels}, got {}",
                cache.levels
            );
            ensure!(
                (0.0..=1.0).contains(&cache.bypass_probability),
                "outline_cache.bypass_probability must be in [0, 1], got {}",
                cache.bypass_probability
            );
            ensure!(
                !cache.model_id.is_empty(),
                "outline_cache.model_id must not be empty"
            );
        }
        Ok(())
    }
}

/// What a decode may spend. Every strategy draws from it; when a counter is
/// exhausted the optional strategies stand down (speculation falls back to
/// plain levels, repair and probes are skipped), and when target calls run
/// out the decode stops and emits the partial reconstruction it has, with
/// [`DecodeReport::stopped_by_budget`] set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DecodeBudget {
    /// Target forward passes (a batched multi-level verification is one).
    pub max_level_calls: usize,
    /// Draft forward passes.
    pub max_draft_calls: usize,
    /// Repair rounds (each costs one target call plus one per regenerated level).
    pub max_repairs: usize,
    /// Particles per prompt; at least 1.
    pub max_particles: usize,
    /// Optional codec work: early-exit decodes and repair decode/encode pairs.
    pub max_probes: usize,
}

impl Default for DecodeBudget {
    fn default() -> Self {
        Self::unlimited()
    }
}

impl DecodeBudget {
    /// No limit on anything.
    pub fn unlimited() -> Self {
        Self {
            max_level_calls: usize::MAX,
            max_draft_calls: usize::MAX,
            max_repairs: usize::MAX,
            max_particles: usize::MAX,
            max_probes: usize::MAX,
        }
    }

    /// `Ok` iff every spending counter of `report` is within this budget; the
    /// error names the first counter that is not.
    pub fn admits(&self, report: &DecodeReport) -> Result<()> {
        for (name, spent, limit) in [
            ("level_calls", report.level_calls, self.max_level_calls),
            ("draft_calls", report.draft_calls, self.max_draft_calls),
            ("repairs", report.repairs, self.max_repairs),
            ("particles", report.particles, self.max_particles),
            ("probes", report.probes, self.max_probes),
        ] {
            ensure!(
                spent <= limit,
                "decode spent {spent} {name}, over the budget of {limit}"
            );
        }
        Ok(())
    }
}

/// What a decode actually did and spent. Counts, not claims.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeReport {
    /// Target forward passes. A multi-level verification counts as one only
    /// when the target [`LevelModel::batches_levels`].
    pub level_calls: usize,
    /// Of those, multi-level verifications made in one pass.
    pub batched_calls: usize,
    /// Levels the target evaluated in total (`>= level_calls`).
    pub level_evaluations: usize,
    /// Draft forward passes.
    pub draft_calls: usize,
    /// Codec reconstructions.
    pub reconstructs: usize,
    /// Codec token decodes (probes and the final decode; a probe reused as
    /// the final decode is counted once).
    pub decodes: usize,
    /// Codec encodes (repair checks).
    pub encodes: usize,
    /// Optional codec calls charged to [`DecodeBudget::max_probes`].
    pub probes: usize,
    /// Sequences in each call: batch × particles.
    pub rows: usize,
    /// Tokens emitted: batch × sequence length.
    pub tokens: usize,
    /// Levels in the emitted reconstruction.
    pub levels_used: usize,
    /// Levels decoded when early exit fired, if it did.
    pub early_exit_level: Option<usize>,
    /// Whether target calls ran out before the last level.
    pub stopped_by_budget: bool,
    /// Drafted levels accepted in full.
    pub accepted_drafts: usize,
    /// Drafted levels with at least one rejected code (emitted resampled).
    pub rejected_drafts: usize,
    /// Drafted levels thrown away after an earlier rejection or unverified.
    pub discarded_drafts: usize,
    /// Drafted codes accepted, over verified levels.
    pub accepted_draft_codes: usize,
    /// Drafted codes rejected and resampled.
    pub rejected_draft_codes: usize,
    /// Repair checks made (decode + re-encode + compare).
    pub repair_checks: usize,
    /// Repair rounds that spent a target call.
    pub repairs: usize,
    /// Coarse codes replaced by a repair.
    pub repaired_positions: usize,
    /// Finer levels regenerated after a repair.
    pub regenerated_levels: usize,
    /// Repairs that were flagged but not affordable.
    pub repairs_declined_for_budget: usize,
    /// Particles per prompt actually used.
    pub particles: usize,
    /// Resampling events (per prompt).
    pub resamples: usize,
    /// Prompts whose outline was found in the cache.
    pub cache_hits: usize,
    /// Prompts whose outline was not found (or bypassed).
    pub cache_misses: usize,
    /// Whether the seeded bypass ignored the cache this time.
    pub cache_bypassed: bool,
    /// Levels taken from the cache instead of the target.
    pub cached_levels: usize,
}

/// A decoded batch and what it cost.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeOutput {
    /// Token ids, `tokens[row][position]`, one row per prompt.
    pub tokens: Vec<Vec<usize>>,
    /// Codes of every level used, `codes[level][row][position]`.
    pub codes: Vec<Vec<Vec<usize>>>,
    /// Realised cost and strategy outcomes.
    pub report: DecodeReport,
}

impl DecodeOutput {
    /// The tokens as an `[batch, L]` tensor.
    pub fn token_tensor<B: Backend>(&self, device: &B::Device) -> Result<Tensor<B, 2, Int>> {
        int_matrix(&self.tokens, device)
    }

    /// The codes as one `[batch, ℓ_k]` tensor per level used.
    pub fn code_tensors<B: Backend>(&self, device: &B::Device) -> Result<Vec<Tensor<B, 2, Int>>> {
        self.codes
            .iter()
            .map(|level| int_matrix(level, device))
            .collect()
    }
}

fn int_matrix<B: Backend>(rows: &[Vec<usize>], device: &B::Device) -> Result<Tensor<B, 2, Int>> {
    let width = rows.first().map_or(0, Vec::len);
    ensure!(
        rows.iter().all(|r| r.len() == width),
        "rows of a code/token matrix differ in length"
    );
    let flat = rows
        .iter()
        .flatten()
        .map(|&v| i64::try_from(v).with_context(|| format!("id {v} does not fit an i64")))
        .collect::<Result<Vec<i64>>>()?;
    Ok(Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device).reshape([rows.len(), width]))
}

// ---------------------------------------------------------- outline cache --

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OutlineKey {
    model_id: String,
    prompt: Vec<i64>,
    lengths: Vec<usize>,
    codebook: usize,
    levels: usize,
}

/// An in-memory least-recently-used store of coarse outlines, keyed by the
/// exact prompt tokens (no hash collisions), model id, schedule, codebook
/// size and outline depth.
#[derive(Debug, Clone)]
pub struct OutlineCache {
    capacity: usize,
    clock: u64,
    entries: HashMap<OutlineKey, (u64, Vec<Vec<usize>>)>,
}

impl OutlineCache {
    /// An empty cache holding at most `capacity` outlines.
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(
            capacity >= 1,
            "an OutlineCache needs capacity of at least 1, got {capacity}"
        );
        Ok(Self {
            capacity,
            clock: 0,
            entries: HashMap::new(),
        })
    }

    /// Outlines stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Most outlines held at once.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn get(&mut self, key: &OutlineKey) -> Option<Vec<Vec<usize>>> {
        self.clock += 1;
        let clock = self.clock;
        self.entries.get_mut(key).map(|(used, outline)| {
            *used = clock;
            outline.clone()
        })
    }

    fn put(&mut self, key: OutlineKey, outline: Vec<Vec<usize>>) {
        self.clock += 1;
        if !self.entries.contains_key(&key) && self.entries.len() >= self.capacity {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, (used, _))| *used)
                .map(|(k, _)| k.clone());
            if let Some(oldest) = oldest {
                self.entries.remove(&oldest);
            }
        }
        self.entries.insert(key, (self.clock, outline));
    }
}

// ----------------------------------------------------------------- decoder --

/// Decodes with a frozen codec, a target level model and optionally a draft.
pub struct Decoder<'a, B: Backend> {
    codec: &'a dyn ConceptCodec<B>,
    target: &'a dyn LevelModel<B>,
    draft: Option<&'a dyn LevelModel<B>>,
    device: B::Device,
}

impl<'a, B: Backend> Decoder<'a, B> {
    /// A decoder for `target` over `codec`, building code tensors on `device`.
    pub fn new(
        codec: &'a dyn ConceptCodec<B>,
        target: &'a dyn LevelModel<B>,
        device: &B::Device,
    ) -> Self {
        Self {
            codec,
            target,
            draft: None,
            device: device.clone(),
        }
    }

    /// Attach the draft model level speculation proposes with.
    pub fn with_draft(mut self, draft: &'a dyn LevelModel<B>) -> Self {
        self.draft = Some(draft);
        self
    }

    /// Decode `batch` sequences, each conditioned on its row of `prompt`
    /// (`[batch, P]` token ids) if given. `cache` is required iff
    /// `config.outline_cache` is set.
    pub fn decode(
        &self,
        prompt: Option<Tensor<B, 2, Int>>,
        batch: usize,
        config: &DecodeConfig,
        budget: &DecodeBudget,
        mut cache: Option<&mut OutlineCache>,
    ) -> Result<DecodeOutput> {
        let schedule = self.codec.schedule();
        schedule.validate()?;
        let lengths = schedule.lengths().to_vec();
        config.validate(lengths.len())?;
        ensure!(
            batch >= 1,
            "decode needs a batch of at least one sequence, got {batch}"
        );
        ensure!(
            budget.max_particles >= 1,
            "budget.max_particles is 0, so not even one sequence per prompt can be decoded"
        );
        let codebook = self.codec.codebook_size();
        let vocab = self.codec.vocab_size();
        ensure!(codebook >= 1, "the codec reports codebook_size 0");
        ensure!(vocab >= 1, "the codec reports vocab_size 0");
        let draft = match (&config.speculation, self.draft) {
            (Some(_), None) => {
                bail!("config.speculation is set but the decoder has no draft model (use Decoder::with_draft)")
            }
            (Some(_), Some(draft)) => Some(draft),
            (None, _) => None,
        };
        ensure!(
            config.outline_cache.is_none() || cache.is_some(),
            "config.outline_cache is set but no OutlineCache was passed to decode"
        );
        let particles = config
            .particles
            .as_ref()
            .map_or(1, |p| p.count)
            .min(budget.max_particles);
        let rows = batch
            .checked_mul(particles)
            .ok_or_else(|| anyhow!("batch {batch} × particles {particles} overflows"))?;

        if let Some(p) = &prompt {
            let [prompt_rows, _] = p.dims();
            ensure!(
                prompt_rows == batch,
                "the prompt has {prompt_rows} rows but batch is {batch}"
            );
        }
        // The cache keys on the exact prompt tokens; only read them back to
        // the host when a cache is in use.
        let prompt_tokens: Vec<Vec<i64>> = match (&prompt, &config.outline_cache) {
            (Some(p), Some(_)) => {
                let [_, width] = p.dims();
                let flat: Vec<i64> = p
                    .clone()
                    .into_data()
                    .convert::<i64>()
                    .iter::<i64>()
                    .collect();
                (0..batch)
                    .map(|row| {
                        flat.get(row * width..(row + 1) * width)
                            .map(<[i64]>::to_vec)
                            .ok_or_else(|| anyhow!("prompt row {row} of width {width} is missing"))
                    })
                    .collect::<Result<_>>()?
            }
            _ => vec![Vec::new(); batch],
        };
        let prompt = prompt.map(|p| {
            if particles > 1 {
                p.repeat_dim(0, particles)
            } else {
                p
            }
        });

        let ctx = Ctx {
            codec: self.codec,
            target: self.target,
            device: &self.device,
            config,
            budget,
            prompt,
            seq_len: schedule.seq_len(),
            lengths,
            codebook,
            vocab,
            latent: self.codec.latent_dim(),
            groups: batch,
            particles,
            rows,
            tilt: config.particles.as_ref().map_or(0.0, |p| p.tilt),
        };
        let mut run = Run {
            ctx,
            draft,
            codes: Vec::new(),
            conf: Vec::new(),
            logp: Vec::new(),
            log_weight: vec![0.0; rows],
            version: 0,
            zhat: None,
            view: None,
            rng: Streams::new(config.seed),
            report: DecodeReport {
                rows,
                particles,
                ..DecodeReport::default()
            },
        };

        let keys = match (&config.outline_cache, cache.as_deref_mut()) {
            (Some(plan), Some(store)) => {
                let keys: Vec<OutlineKey> = prompt_tokens
                    .into_iter()
                    .map(|prompt| OutlineKey {
                        model_id: plan.model_id.clone(),
                        prompt,
                        lengths: run.ctx.lengths.clone(),
                        codebook,
                        levels: plan.levels,
                    })
                    .collect();
                run.consult_cache(store, plan, &keys)?;
                Some((plan.levels, keys))
            }
            _ => None,
        };

        run.decode_levels()?;
        let output = run.finish()?;

        if let (Some((levels, keys)), Some(store)) = (keys, cache) {
            if output.codes.len() >= levels {
                for (group, key) in keys.into_iter().enumerate() {
                    let outline = output
                        .codes
                        .iter()
                        .take(levels)
                        .map(|level| {
                            level.get(group).cloned().ok_or_else(|| {
                                anyhow!("decoded output has no row {group} to cache")
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    store.put(key, outline);
                }
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------- machinery --

/// One purpose, one stream: enabling a strategy never shifts another's draws.
struct Streams {
    code: ChaCha8Rng,
    draft: ChaCha8Rng,
    accept: ChaCha8Rng,
    resample: ChaCha8Rng,
    repair: ChaCha8Rng,
    cache: ChaCha8Rng,
}

impl Streams {
    fn new(seed: u64) -> Self {
        let stream = |id: u64| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            rng.set_stream(id);
            rng
        };
        Self {
            code: stream(1),
            draft: stream(2),
            accept: stream(3),
            resample: stream(4),
            repair: stream(5),
            cache: stream(6),
        }
    }
}

/// Index of the largest value, ties to the lowest index.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best = index;
            best_value = value;
        }
    }
    best
}

/// `ln Σ exp(l)` over one position's logits, computed stably.
fn log_normalizer(logits: &[f32]) -> f64 {
    let max = logits
        .iter()
        .map(|&l| f64::from(l))
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|&l| (f64::from(l) - max).exp()).sum();
    max + sum.ln()
}

/// The target's own log-probability (temperature 1, no top-k) of `code`.
fn log_prob(logits: &[f32], code: usize) -> f64 {
    logits.get(code).map_or(f64::NEG_INFINITY, |&l| {
        f64::from(l) - log_normalizer(logits)
    })
}

/// Inverse-CDF draw from non-negative weights with `u ∈ [0, 1)`.
fn draw_index(weights: &[f64], u: f64) -> usize {
    let total: f64 = weights.iter().sum();
    let target = u * total;
    let mut acc = 0.0;
    let mut last = 0usize;
    for (index, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            last = index;
            acc += w;
            if target < acc {
                return index;
            }
        }
    }
    last
}

/// Speculative sampling of one code. `p` and `q` are the target's and the
/// draft's sampling distributions, `drafted` was drawn from `q`, `u` is a
/// uniform draw in `[0, 1)`. Returns the emitted code and whether the draft
/// was accepted. The emitted code is distributed exactly as `p`:
/// `P(c) = q(c)·min(1, p(c)/q(c)) + (1 − Σ_d min(p(d), q(d)))·r(c)` with
/// `r = max(0, p − q)/Σ max(0, p − q)`, which is `p(c)`.
fn speculative_code<R: Rng>(
    p: &[f64],
    q: &[f64],
    drafted: usize,
    u: f64,
    rng: &mut R,
    residual: &mut Vec<f64>,
) -> (usize, bool) {
    let pd = p.get(drafted).copied().unwrap_or(0.0);
    let qd = q.get(drafted).copied().unwrap_or(0.0);
    if qd > 0.0 && u < (pd / qd).min(1.0) {
        return (drafted, true);
    }
    residual.clear();
    residual.extend(p.iter().zip(q).map(|(a, b)| (a - b).max(0.0)));
    let mass: f64 = residual.iter().sum();
    let resample = rng.random::<f64>();
    // `mass` is 0 only when p == q, where rejection has probability 0 in
    // exact arithmetic; rounding can still get here, and p is then the
    // residual's limit.
    let code = if mass > 0.0 {
        draw_index(residual, resample)
    } else {
        draw_index(p, resample)
    };
    (code, false)
}

/// Systematic resampling: `n` evenly spaced points offset by `u ∈ [0, 1)`
/// against the cumulative normalised `weights`.
fn systematic_resample(weights: &[f64], u: f64) -> Vec<usize> {
    let n = weights.len();
    let mut picks = Vec::with_capacity(n);
    let mut cumulative = 0.0;
    let mut index = 0usize;
    for slot in 0..n {
        let point = (u + slot as f64) / n as f64;
        while index + 1 < n && cumulative + weights.get(index).copied().unwrap_or(0.0) <= point {
            cumulative += weights.get(index).copied().unwrap_or(0.0);
            index += 1;
        }
        picks.push(index);
    }
    picks
}

/// `exp(l − max)` normalised to sum to one.
fn normalized_weights(log_weights: &[f64]) -> Vec<f64> {
    let max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let raw: Vec<f64> = log_weights.iter().map(|&l| (l - max).exp()).collect();
    let total: f64 = raw.iter().sum();
    raw.into_iter().map(|w| w / total).collect()
}

/// Minimum Bayes risk under token Hamming distance: the candidate with the
/// smallest weighted distance to the others, ties to the lowest index.
fn mbr_index(candidates: &[&[usize]], weights: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_risk = f64::INFINITY;
    for (index, a) in candidates.iter().enumerate() {
        let risk: f64 = candidates
            .iter()
            .zip(weights)
            .map(|(b, w)| w * a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as f64)
            .sum();
        if risk < best_risk {
            best = index;
            best_risk = risk;
        }
    }
    best
}

fn row_slice<T>(values: &[T], row: usize, len: usize) -> Result<&[T]> {
    values.get(row * len..(row + 1) * len).ok_or_else(|| {
        anyhow!(
            "row {row} of width {len} is outside a buffer of {} values",
            values.len()
        )
    })
}

/// One level's codes, the target's probability of each, and each row's
/// summed log-probability.
struct LevelDraw {
    codes: Vec<usize>,
    conf: Vec<f64>,
    logp: Vec<f64>,
}

impl LevelDraw {
    fn new(rows: usize, len: usize) -> Self {
        Self {
            codes: Vec::with_capacity(rows * len),
            conf: Vec::with_capacity(rows * len),
            logp: vec![0.0; rows],
        }
    }

    fn record(&mut self, row: usize, code: usize, log_p: f64) {
        self.codes.push(code);
        self.conf.push(log_p.exp());
        if let Some(total) = self.logp.get_mut(row) {
            *total += log_p;
        }
    }
}

/// Draw every code of a level from its `[rows, len, codebook]` logits.
fn sample_level(
    logits: &[f32],
    len: usize,
    codebook: usize,
    rows: usize,
    sampler: &CodeSampler,
    rng: &mut ChaCha8Rng,
) -> LevelDraw {
    let mut draw = LevelDraw::new(rows, len);
    let mut dist = Vec::with_capacity(codebook);
    for (index, position) in logits.chunks_exact(codebook).enumerate() {
        let code = if sampler.is_greedy() {
            argmax(position)
        } else {
            sampler.distribution(position, &mut dist);
            draw_index(&dist, rng.random::<f64>())
        };
        draw.record(index / len, code, log_prob(position, code));
    }
    draw
}

/// The argmax tokens of a reconstruction and the least confident position.
#[derive(Clone)]
struct TokenView {
    tokens: Vec<usize>,
    min_top_probability: f64,
}

/// Everything a decode reads but does not change.
struct Ctx<'r, B: Backend> {
    codec: &'r dyn ConceptCodec<B>,
    target: &'r dyn LevelModel<B>,
    device: &'r B::Device,
    config: &'r DecodeConfig,
    budget: &'r DecodeBudget,
    /// `[rows, P]`: each prompt row tiled once per particle.
    prompt: Option<Tensor<B, 2, Int>>,
    lengths: Vec<usize>,
    seq_len: usize,
    codebook: usize,
    vocab: usize,
    latent: usize,
    /// Prompts. Row `r` belongs to prompt `r % groups`, particle `r / groups`.
    groups: usize,
    particles: usize,
    rows: usize,
    tilt: f64,
}

impl<B: Backend> Ctx<'_, B> {
    fn len_of(&self, level: usize) -> Result<usize> {
        self.lengths.get(level).copied().ok_or_else(|| {
            anyhow!(
                "level {level} is outside a {}-level schedule",
                self.lengths.len()
            )
        })
    }

    fn reconstruct(&self, levels: &[&[usize]]) -> Result<Tensor<B, 3>> {
        let tensors = levels
            .iter()
            .zip(&self.lengths)
            .map(|(codes, &len)| {
                ensure!(
                    codes.len() == self.rows * len,
                    "a level of length {len} holds {} codes",
                    codes.len()
                );
                let ints = codes
                    .iter()
                    .map(|&c| {
                        i64::try_from(c).with_context(|| format!("code {c} does not fit an i64"))
                    })
                    .collect::<Result<Vec<i64>>>()?;
                Ok(Tensor::<B, 1, Int>::from_ints(ints.as_slice(), self.device)
                    .reshape([self.rows, len]))
            })
            .collect::<Result<Vec<_>>>()?;
        let zhat = self
            .codec
            .reconstruct(&tensors, self.rows)
            .with_context(|| format!("codec reconstruct from {} levels", levels.len()))?;
        let dims = zhat.dims();
        ensure!(
            dims == [self.rows, self.seq_len, self.latent],
            "codec reconstruct returned shape {dims:?}, expected [{}, {}, {}]",
            self.rows,
            self.seq_len,
            self.latent
        );
        Ok(zhat)
    }

    fn host_level(&self, logits: Tensor<B, 3>, level: usize, who: &str) -> Result<Vec<f32>> {
        let len = self.len_of(level)?;
        host_floats(
            logits,
            [self.rows, len, self.codebook],
            &format!("{who} logits for level {level}"),
        )
    }

    fn level_logits(
        &self,
        model: &dyn LevelModel<B>,
        level: usize,
        zhat: Tensor<B, 3>,
        who: &str,
    ) -> Result<Vec<f32>> {
        let logits = model
            .level_logits(level, zhat, self.prompt.clone())
            .with_context(|| format!("{who} level_logits for level {level}"))?;
        self.host_level(logits, level, who)
    }
}

fn host_floats<B: Backend>(
    tensor: Tensor<B, 3>,
    expect: [usize; 3],
    what: &str,
) -> Result<Vec<f32>> {
    let dims = tensor.dims();
    ensure!(
        dims == expect,
        "{what} has shape {dims:?}, expected {expect:?}"
    );
    let values: Vec<f32> = tensor.into_data().convert::<f32>().iter::<f32>().collect();
    if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
        bail!(
            "{what} holds the non-finite value {} at flat index {bad}",
            values.get(bad).copied().unwrap_or(f32::NAN)
        );
    }
    Ok(values)
}

fn host_codes<B: Backend>(
    tensor: Tensor<B, 2, Int>,
    expect: [usize; 2],
    codebook: usize,
    what: &str,
) -> Result<Vec<usize>> {
    let dims = tensor.dims();
    ensure!(
        dims == expect,
        "{what} has shape {dims:?}, expected {expect:?}"
    );
    tensor
        .into_data()
        .convert::<i64>()
        .iter::<i64>()
        .map(|v| {
            usize::try_from(v)
                .ok()
                .filter(|&c| c < codebook)
                .ok_or_else(|| anyhow!("{what} holds code {v}, outside 0..{codebook}"))
        })
        .collect()
}

/// One decode in progress.
struct Run<'r, B: Backend> {
    ctx: Ctx<'r, B>,
    draft: Option<&'r dyn LevelModel<B>>,
    /// `codes[level]` is `[rows × ℓ_level]`, row-major.
    codes: Vec<Vec<usize>>,
    /// The target's probability of each code (1 for cached codes).
    conf: Vec<Vec<f64>>,
    /// `logp[level][row]`: the row's summed target log-probability.
    logp: Vec<Vec<f64>>,
    log_weight: Vec<f64>,
    /// Bumped on every change to the codes; keys the two caches below.
    version: u64,
    zhat: Option<(u64, Tensor<B, 3>)>,
    view: Option<(u64, TokenView)>,
    rng: Streams,
    report: DecodeReport,
}

impl<B: Backend> Run<'_, B> {
    fn levels_done(&self) -> usize {
        self.codes.len()
    }

    fn calls_left(&self) -> usize {
        self.ctx
            .budget
            .max_level_calls
            .saturating_sub(self.report.level_calls)
    }

    fn draft_calls_left(&self) -> usize {
        self.ctx
            .budget
            .max_draft_calls
            .saturating_sub(self.report.draft_calls)
    }

    fn probes_left(&self) -> usize {
        self.ctx
            .budget
            .max_probes
            .saturating_sub(self.report.probes)
    }

    fn view_is_fresh(&self) -> bool {
        matches!(&self.view, Some((v, _)) if *v == self.version)
    }

    /// `ẑ` from the first `levels` levels of the current codes.
    fn reconstruct_prefix(&mut self, levels: usize) -> Result<Tensor<B, 3>> {
        let full = levels == self.levels_done();
        if full {
            if let Some((v, zhat)) = &self.zhat {
                if *v == self.version {
                    return Ok(zhat.clone());
                }
            }
        }
        let refs: Vec<&[usize]> = self.codes.iter().take(levels).map(Vec::as_slice).collect();
        let zhat = self.ctx.reconstruct(&refs)?;
        self.report.reconstructs += 1;
        if full {
            self.zhat = Some((self.version, zhat.clone()));
        }
        Ok(zhat)
    }

    /// The argmax tokens of the current reconstruction; `probe` charges the
    /// decode to the probe budget (the caller has checked it is affordable).
    fn token_view(&mut self, probe: bool) -> Result<TokenView> {
        if let Some((v, view)) = &self.view {
            if *v == self.version {
                return Ok(view.clone());
            }
        }
        let zhat = self.reconstruct_prefix(self.levels_done())?;
        let logits = self
            .ctx
            .codec
            .decode_logits(zhat)
            .context("codec decode_logits")?;
        self.report.decodes += 1;
        if probe {
            self.report.probes += 1;
        }
        let vocab = self.ctx.vocab;
        let values = host_floats(
            logits,
            [self.ctx.rows, self.ctx.seq_len, vocab],
            "codec token logits",
        )?;
        let mut tokens = Vec::with_capacity(self.ctx.rows * self.ctx.seq_len);
        let mut min_top_probability = f64::INFINITY;
        for position in values.chunks_exact(vocab) {
            let best = argmax(position);
            tokens.push(best);
            min_top_probability = min_top_probability.min(log_prob(position, best).exp());
        }
        let view = TokenView {
            tokens,
            min_top_probability,
        };
        self.view = Some((self.version, view.clone()));
        Ok(view)
    }

    fn target_call(&mut self, level: usize, zhat: Tensor<B, 3>) -> Result<Vec<f32>> {
        ensure!(
            self.calls_left() >= 1,
            "no target call left for level {level} (budget.max_level_calls = {})",
            self.ctx.budget.max_level_calls
        );
        let logits = self
            .ctx
            .level_logits(self.ctx.target, level, zhat, "target")?;
        self.report.level_calls += 1;
        self.report.level_evaluations += 1;
        Ok(logits)
    }

    fn push_level(&mut self, draw: LevelDraw) {
        let tilt = self.ctx.tilt;
        for (weight, lp) in self.log_weight.iter_mut().zip(&draw.logp) {
            *weight += tilt * lp;
        }
        self.codes.push(draw.codes);
        self.conf.push(draw.conf);
        self.logp.push(draw.logp);
        self.version += 1;
    }

    fn decode_levels(&mut self) -> Result<()> {
        let config = self.ctx.config;
        let levels = self.ctx.lengths.len();
        while self.levels_done() < levels {
            if self.calls_left() == 0 {
                self.report.stopped_by_budget = true;
                break;
            }
            let span = match (config.speculation.as_ref(), self.draft) {
                (Some(spec), Some(draft)) => {
                    let m = spec
                        .draft_levels
                        .min(levels - self.levels_done())
                        .min(self.draft_calls_left());
                    (m >= 1).then_some((draft, m))
                }
                _ => None,
            };
            match span {
                Some((draft, m)) => self.speculative_step(draft, m)?,
                None => self.plain_step()?,
            }
            if let Some(repair) = config.repair.as_ref() {
                self.repair_round(repair)?;
            }
            if self.levels_done() >= levels {
                break;
            }
            let exit = match config.early_exit.as_ref() {
                Some(exit) => self.try_early_exit(exit)?,
                None => false,
            };
            if exit {
                break;
            }
            if let Some(particles) = config.particles.as_ref() {
                self.maybe_resample(particles)?;
            }
        }
        Ok(())
    }

    fn plain_step(&mut self) -> Result<()> {
        let level = self.levels_done();
        let len = self.ctx.len_of(level)?;
        let zhat = self.reconstruct_prefix(level)?;
        let logits = self.target_call(level, zhat)?;
        let draw = sample_level(
            &logits,
            len,
            self.ctx.codebook,
            self.ctx.rows,
            &self.ctx.config.sampler,
            &mut self.rng.code,
        );
        self.push_level(draw);
        Ok(())
    }

    fn speculative_step(&mut self, draft: &dyn LevelModel<B>, m: usize) -> Result<()> {
        let start = self.levels_done();
        let sampler = self.ctx.config.sampler;
        let mut drafted: Vec<Vec<usize>> = Vec::with_capacity(m);
        let mut draft_logits: Vec<Vec<f32>> = Vec::with_capacity(m);
        let mut zhats: Vec<Tensor<B, 3>> = Vec::with_capacity(m);
        for j in 0..m {
            let level = start + j;
            let zhat = if j == 0 {
                self.reconstruct_prefix(start)?
            } else {
                let refs: Vec<&[usize]> = self
                    .codes
                    .iter()
                    .chain(drafted.iter())
                    .map(Vec::as_slice)
                    .collect();
                let zhat = self.ctx.reconstruct(&refs)?;
                self.report.reconstructs += 1;
                zhat
            };
            let q = self.ctx.level_logits(draft, level, zhat.clone(), "draft")?;
            self.report.draft_calls += 1;
            let len = self.ctx.len_of(level)?;
            let codes = sample_level(
                &q,
                len,
                self.ctx.codebook,
                self.ctx.rows,
                &sampler,
                &mut self.rng.draft,
            )
            .codes;
            drafted.push(codes);
            draft_logits.push(q);
            zhats.push(zhat);
        }

        let proposed = drafted.len();
        let pairs = draft_logits.iter().zip(&drafted);
        if self.ctx.target.batches_levels() && proposed > 1 {
            let requests: Vec<(usize, Tensor<B, 3>)> = zhats
                .into_iter()
                .enumerate()
                .map(|(j, zhat)| (start + j, zhat))
                .collect();
            let outputs = self
                .ctx
                .target
                .levels_logits(requests, self.ctx.prompt.clone())
                .with_context(|| {
                    format!(
                        "target levels_logits for levels {start}..{}",
                        start + proposed
                    )
                })?;
            self.report.level_calls += 1;
            self.report.batched_calls += 1;
            self.report.level_evaluations += proposed;
            ensure!(
                outputs.len() == proposed,
                "target levels_logits answered {} of {proposed} requests",
                outputs.len()
            );
            for (j, (output, (q, codes))) in outputs.into_iter().zip(pairs).enumerate() {
                let p = self.ctx.host_level(output, start + j, "target")?;
                if self.accept_level(start + j, &p, q, codes)? {
                    self.report.discarded_drafts += proposed - j - 1;
                    break;
                }
            }
        } else {
            for (j, (zhat, (q, codes))) in zhats.into_iter().zip(pairs).enumerate() {
                if self.calls_left() == 0 {
                    self.report.discarded_drafts += proposed - j;
                    break;
                }
                let p = self.target_call(start + j, zhat)?;
                if self.accept_level(start + j, &p, q, codes)? {
                    self.report.discarded_drafts += proposed - j - 1;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Verify one drafted level against the target's logits, push the
    /// accepted-or-resampled codes, and say whether any code was rejected.
    fn accept_level(
        &mut self,
        level: usize,
        p_logits: &[f32],
        q_logits: &[f32],
        drafted: &[usize],
    ) -> Result<bool> {
        let len = self.ctx.len_of(level)?;
        let codebook = self.ctx.codebook;
        let sampler = self.ctx.config.sampler;
        ensure!(
            p_logits.len() == q_logits.len() && drafted.len() * codebook == p_logits.len(),
            "level {level}: {} target logits, {} draft logits and {} drafted codes do not line up",
            p_logits.len(),
            q_logits.len(),
            drafted.len()
        );
        let mut p = Vec::with_capacity(codebook);
        let mut q = Vec::with_capacity(codebook);
        let mut residual = Vec::with_capacity(codebook);
        let mut draw = LevelDraw::new(self.ctx.rows, len);
        let mut rejected = 0usize;
        for (index, ((pl, ql), &d)) in p_logits
            .chunks_exact(codebook)
            .zip(q_logits.chunks_exact(codebook))
            .zip(drafted)
            .enumerate()
        {
            sampler.distribution(pl, &mut p);
            sampler.distribution(ql, &mut q);
            let u = self.rng.accept.random::<f64>();
            let (code, accepted) =
                speculative_code(&p, &q, d, u, &mut self.rng.accept, &mut residual);
            if !accepted {
                rejected += 1;
            }
            draw.record(index / len, code, log_prob(pl, code));
        }
        self.report.accepted_draft_codes += drafted.len() - rejected;
        self.report.rejected_draft_codes += rejected;
        if rejected == 0 {
            self.report.accepted_drafts += 1;
        } else {
            self.report.rejected_drafts += 1;
        }
        self.push_level(draw);
        Ok(rejected > 0)
    }

    fn try_early_exit(&mut self, exit: &EarlyExit) -> Result<bool> {
        let done = self.levels_done();
        if done < exit.min_levels || done >= self.ctx.lengths.len() {
            return Ok(false);
        }
        if !self.view_is_fresh() && self.probes_left() == 0 {
            return Ok(false);
        }
        let view = self.token_view(true)?;
        if view.min_top_probability >= exit.threshold {
            self.report.early_exit_level = Some(done);
            return Ok(true);
        }
        Ok(false)
    }

    fn repair_round(&mut self, repair: &Repair) -> Result<()> {
        let done = self.levels_done();
        let total_levels = self.ctx.lengths.len();
        if done < 2 || self.report.repairs >= self.ctx.budget.max_repairs {
            return Ok(());
        }
        let probes_needed = if self.view_is_fresh() { 1 } else { 2 };
        if self.probes_left() < probes_needed {
            return Ok(());
        }
        let (rows, codebook, seq_len) = (self.ctx.rows, self.ctx.codebook, self.ctx.seq_len);
        let view = self.token_view(true)?;
        let tokens = view
            .tokens
            .iter()
            .map(|&t| i64::try_from(t).with_context(|| format!("token {t} does not fit an i64")))
            .collect::<Result<Vec<i64>>>()?;
        let tokens = Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), self.ctx.device)
            .reshape([rows, seq_len]);
        let encoded = self
            .ctx
            .codec
            .encode(tokens)
            .context("codec encode during a repair check")?;
        self.report.encodes += 1;
        self.report.probes += 1;
        self.report.repair_checks += 1;
        ensure!(
            encoded.codes.len() == total_levels,
            "codec encode returned {} levels, the schedule has {total_levels}",
            encoded.codes.len()
        );

        // The coarsest level (strictly coarser than the newest) with a flag.
        let guard = repair.confidence_guard;
        let mut found: Option<(usize, Vec<(usize, usize)>)> = None;
        for (level, reencoded) in encoded.codes.into_iter().enumerate().take(done - 1) {
            let len = self.ctx.len_of(level)?;
            let reencoded = host_codes(
                reencoded,
                [rows, len],
                codebook,
                &format!("re-encoded level {level}"),
            )?;
            let generated = self
                .codes
                .get(level)
                .ok_or_else(|| anyhow!("level {level} has no codes"))?;
            let conf = self
                .conf
                .get(level)
                .ok_or_else(|| anyhow!("level {level} has no confidences"))?;
            let flags: Vec<(usize, usize)> = generated
                .iter()
                .zip(&reencoded)
                .zip(conf)
                .enumerate()
                .filter(|(_, ((&g, &r), &c))| g != r && c < guard)
                .map(|(index, ((_, &r), _))| (index, r))
                .collect();
            if !flags.is_empty() {
                found = Some((level, flags));
                break;
            }
        }
        let Some((level, flags)) = found else {
            return Ok(());
        };

        // One call to re-read the level plus one per finer level, while still
        // paying for every remaining plain level.
        let cost = done - level;
        let reserve = total_levels - done;
        if self.calls_left() < cost + reserve {
            self.report.repairs_declined_for_budget += 1;
            return Ok(());
        }
        let zhat = self.reconstruct_prefix(level)?;
        let logits = self.target_call(level, zhat)?;
        self.report.repairs += 1;

        let len = self.ctx.len_of(level)?;
        let tilt = self.ctx.tilt;
        let mut changed = vec![false; rows];
        for (index, alternative) in flags {
            let position = row_slice(&logits, index, codebook)?;
            let lp_alt = log_prob(position, alternative);
            if lp_alt.exp() < repair.min_alternative_probability {
                continue;
            }
            let row = index / len;
            let codes = self
                .codes
                .get_mut(level)
                .ok_or_else(|| anyhow!("level {level} vanished"))?;
            let slot = codes
                .get_mut(index)
                .ok_or_else(|| anyhow!("position {index} outside level {level}"))?;
            let lp_old = log_prob(position, *slot);
            *slot = alternative;
            if let Some(c) = self.conf.get_mut(level).and_then(|c| c.get_mut(index)) {
                *c = lp_alt.exp();
            }
            if let Some(total) = self.logp.get_mut(level).and_then(|l| l.get_mut(row)) {
                *total += lp_alt - lp_old;
            }
            if let Some(weight) = self.log_weight.get_mut(row) {
                *weight += tilt * (lp_alt - lp_old);
            }
            if let Some(flag) = changed.get_mut(row) {
                *flag = true;
            }
            self.report.repaired_positions += 1;
        }
        if !changed.iter().any(|&c| c) {
            return Ok(());
        }
        self.version += 1;

        // Everything finer was conditioned on the old codes: redraw it for
        // the sequences that changed.
        let sampler = self.ctx.config.sampler;
        for finer in (level + 1)..done {
            let len = self.ctx.len_of(finer)?;
            let zhat = self.reconstruct_prefix(finer)?;
            let logits = self.target_call(finer, zhat)?;
            let draw = sample_level(&logits, len, codebook, rows, &sampler, &mut self.rng.repair);
            for (row, _) in changed.iter().enumerate().filter(|(_, &c)| c) {
                let range = row * len..(row + 1) * len;
                let new_codes = row_slice(&draw.codes, row, len)?;
                let new_conf = row_slice(&draw.conf, row, len)?;
                let new_logp = draw.logp.get(row).copied().unwrap_or(0.0);
                self.codes
                    .get_mut(finer)
                    .and_then(|c| c.get_mut(range.clone()))
                    .ok_or_else(|| anyhow!("row {row} outside level {finer}"))?
                    .copy_from_slice(new_codes);
                self.conf
                    .get_mut(finer)
                    .and_then(|c| c.get_mut(range))
                    .ok_or_else(|| anyhow!("row {row} outside level {finer}"))?
                    .copy_from_slice(new_conf);
                if let Some(total) = self.logp.get_mut(finer).and_then(|l| l.get_mut(row)) {
                    if let Some(weight) = self.log_weight.get_mut(row) {
                        *weight += tilt * (new_logp - *total);
                    }
                    *total = new_logp;
                }
            }
            self.report.regenerated_levels += 1;
            self.version += 1;
        }
        Ok(())
    }

    fn maybe_resample(&mut self, plan: &Particles) -> Result<()> {
        let (groups, n, rows) = (self.ctx.groups, self.ctx.particles, self.ctx.rows);
        if n <= 1 {
            return Ok(());
        }
        let mut source: Vec<usize> = (0..rows).collect();
        let mut any = false;
        for group in 0..groups {
            let members: Vec<usize> = (0..n).map(|p| p * groups + group).collect();
            let log_weights: Vec<f64> = members
                .iter()
                .map(|&r| self.log_weight.get(r).copied().unwrap_or(0.0))
                .collect();
            let weights = normalized_weights(&log_weights);
            let ess = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
            if ess < plan.ess_threshold * n as f64 {
                let picks = systematic_resample(&weights, self.rng.resample.random::<f64>());
                for (&slot, &pick) in members.iter().zip(&picks) {
                    let from = members
                        .get(pick)
                        .copied()
                        .ok_or_else(|| anyhow!("resampled particle {pick} of {n}"))?;
                    if let Some(s) = source.get_mut(slot) {
                        *s = from;
                    }
                }
                for &r in &members {
                    if let Some(w) = self.log_weight.get_mut(r) {
                        *w = 0.0;
                    }
                }
                self.report.resamples += 1;
                any = true;
            }
        }
        if !any {
            return Ok(());
        }
        for (level, (codes, conf)) in self.codes.iter_mut().zip(self.conf.iter_mut()).enumerate() {
            let len = self.ctx.lengths.get(level).copied().unwrap_or(0);
            let mut new_codes = Vec::with_capacity(codes.len());
            let mut new_conf = Vec::with_capacity(conf.len());
            for &from in &source {
                new_codes.extend_from_slice(row_slice(codes.as_slice(), from, len)?);
                new_conf.extend_from_slice(row_slice(conf.as_slice(), from, len)?);
            }
            *codes = new_codes;
            *conf = new_conf;
        }
        for logp in &mut self.logp {
            let permuted = source
                .iter()
                .map(|&from| logp.get(from).copied().unwrap_or(0.0))
                .collect();
            *logp = permuted;
        }
        self.version += 1;
        Ok(())
    }

    fn consult_cache(
        &mut self,
        store: &mut OutlineCache,
        plan: &OutlineCacheUse,
        keys: &[OutlineKey],
    ) -> Result<()> {
        if self.rng.cache.random::<f64>() < plan.bypass_probability {
            self.report.cache_bypassed = true;
            self.report.cache_misses += keys.len();
            return Ok(());
        }
        let outlines: Vec<Option<Vec<Vec<usize>>>> =
            keys.iter().map(|key| store.get(key)).collect();
        let hits = outlines.iter().filter(|o| o.is_some()).count();
        self.report.cache_hits += hits;
        self.report.cache_misses += keys.len() - hits;
        if hits < keys.len() {
            return Ok(());
        }
        let outlines: Vec<Vec<Vec<usize>>> = outlines.into_iter().flatten().collect();
        for level in 0..plan.levels {
            let len = self.ctx.len_of(level)?;
            let mut draw = LevelDraw::new(self.ctx.rows, len);
            for row in 0..self.ctx.rows {
                let group = row % self.ctx.groups;
                let codes = outlines
                    .get(group)
                    .and_then(|outline| outline.get(level))
                    .filter(|codes| codes.len() == len)
                    .ok_or_else(|| {
                        anyhow!(
                            "cached outline of prompt {group} has no level {level} of length {len}"
                        )
                    })?;
                for &code in codes {
                    ensure!(
                        code < self.ctx.codebook,
                        "cached code {code} is outside 0..{}",
                        self.ctx.codebook
                    );
                    // Cached codes are trusted: log-probability 0, confidence 1.
                    draw.record(row, code, 0.0);
                }
            }
            self.push_level(draw);
        }
        self.report.cached_levels = plan.levels;
        Ok(())
    }

    fn finish(mut self) -> Result<DecodeOutput> {
        let used = self.levels_done();
        ensure!(
            used >= 1,
            "no level was decoded: budget.max_level_calls = {} allowed no target call and nothing was cached",
            self.ctx.budget.max_level_calls
        );
        let view = self.token_view(false)?;
        let (groups, n, seq_len) = (self.ctx.groups, self.ctx.particles, self.ctx.seq_len);
        let mut tokens = Vec::with_capacity(groups);
        let mut picks = Vec::with_capacity(groups);
        for group in 0..groups {
            let members: Vec<usize> = (0..n).map(|p| p * groups + group).collect();
            let log_weights: Vec<f64> = members
                .iter()
                .map(|&r| self.log_weight.get(r).copied().unwrap_or(0.0))
                .collect();
            let weights = normalized_weights(&log_weights);
            let candidates = members
                .iter()
                .map(|&r| row_slice(&view.tokens, r, seq_len))
                .collect::<Result<Vec<&[usize]>>>()?;
            let best = mbr_index(&candidates, &weights);
            let row = members
                .get(best)
                .copied()
                .ok_or_else(|| anyhow!("MBR picked particle {best} of {n}"))?;
            tokens.push(row_slice(&view.tokens, row, seq_len)?.to_vec());
            picks.push(row);
        }
        let codes = self
            .codes
            .iter()
            .zip(&self.ctx.lengths)
            .map(|(level, &len)| {
                picks
                    .iter()
                    .map(|&row| row_slice(level, row, len).map(<[usize]>::to_vec))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        self.report.levels_used = used;
        self.report.tokens = groups * seq_len;
        Ok(DecodeOutput {
            tokens,
            codes,
            report: self.report,
        })
    }
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it (see the `[lints]` table in
// `Cargo.toml` and the contract in the crate docs).
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::dbg_macro,
    clippy::let_underscore_must_use,
    clippy::redundant_pattern_matching,
    clippy::mem_forget,
    clippy::exit,
    clippy::print_stdout,
    clippy::print_stderr
)]
mod tests {
    use super::doubles::{CountingCodec, CountingModel, ToyCodec, ToyModel};
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn setup(seed: u64) -> (ToyCodec<B>, ToyModel<B>) {
        let device = Default::default();
        let codec = ToyCodec::<B>::new(vec![1, 2, 4, 8], 6, 4, 5, 3.0, seed, &device).unwrap();
        let model =
            ToyModel::<B>::random(vec![1, 2, 4, 8], 6, 4, 1.0, seed + 100, &device).unwrap();
        (codec, model)
    }

    fn sampling(seed: u64) -> DecodeConfig {
        DecodeConfig {
            sampler: CodeSampler::with_temperature(1.0),
            seed,
            ..DecodeConfig::default()
        }
    }

    #[test]
    fn test_sampler_greedy_is_one_hot_and_top_k_restricts_support() {
        let logits = [0.5f32, 2.0, 2.0, -1.0];
        let mut out = Vec::new();
        CodeSampler::greedy().distribution(&logits, &mut out);
        assert_eq!(out, vec![0.0, 1.0, 0.0, 0.0], "ties go to the lowest index");
        let sampler = CodeSampler {
            temperature: 1.0,
            top_k: Some(2),
        };
        sampler.distribution(&logits, &mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[3], 0.0);
        assert!((out[1] - 0.5).abs() < 1e-12 && (out[2] - 0.5).abs() < 1e-12);
        CodeSampler::with_temperature(2.0).distribution(&logits, &mut out);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(CodeSampler {
            temperature: -1.0,
            top_k: None
        }
        .validate()
        .is_err());
        assert!(CodeSampler {
            temperature: 1.0,
            top_k: Some(0)
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_systematic_resampling_and_mbr() {
        let picks = systematic_resample(&[0.0, 1.0, 0.0], 0.3);
        assert_eq!(picks, vec![1, 1, 1]);
        // Points (u + s)/3 against cumulative weights 0.5, 0.75, 1: the
        // pick is the first index whose cumulative weight exceeds the point.
        assert_eq!(systematic_resample(&[0.5, 0.25, 0.25], 0.2), vec![0, 0, 1]);
        assert_eq!(systematic_resample(&[0.5, 0.25, 0.25], 0.5), vec![0, 1, 2]);
        // Every index is picked floor(n·w) or ceil(n·w) times, for any u.
        let weights = [0.1, 0.45, 0.2, 0.25];
        for step in 0..20 {
            let picks = systematic_resample(&weights, f64::from(step) / 20.0);
            for (index, w) in weights.iter().enumerate() {
                let count = picks.iter().filter(|&&p| p == index).count() as f64;
                let expected = w * weights.len() as f64;
                assert!(
                    count >= expected.floor() - 1e-9 && count <= expected.ceil() + 1e-9,
                    "{picks:?}"
                );
            }
        }
        let a = [1usize, 2, 3];
        let b = [1usize, 2, 4];
        let c = [9usize, 9, 9];
        let candidates: Vec<&[usize]> = vec![&c, &a, &b];
        // `a` and `b` are close, `c` is far: the medoid is the first of a/b.
        assert_eq!(mbr_index(&candidates, &[1.0 / 3.0; 3]), 1);
    }

    #[test]
    fn test_config_validation_names_the_field() {
        let bad = DecodeConfig {
            early_exit: Some(EarlyExit {
                threshold: 0.5,
                min_levels: 0,
            }),
            ..DecodeConfig::default()
        };
        let err = bad.validate(4).unwrap_err().to_string();
        assert!(err.contains("early_exit.min_levels"), "{err}");
        let bad = DecodeConfig {
            speculation: Some(Speculation { draft_levels: 0 }),
            ..DecodeConfig::default()
        };
        assert!(bad
            .validate(4)
            .unwrap_err()
            .to_string()
            .contains("speculation.draft_levels"));
        let bad = DecodeConfig {
            particles: Some(Particles {
                count: 2,
                tilt: f64::NAN,
                ess_threshold: 0.5,
            }),
            ..DecodeConfig::default()
        };
        assert!(bad
            .validate(4)
            .unwrap_err()
            .to_string()
            .contains("particles.tilt"));
        // Speculation without a draft, cache without a store.
        let (codec, model) = setup(1);
        let device = Default::default();
        let decoder = Decoder::new(&codec, &model, &device);
        let spec = DecodeConfig {
            speculation: Some(Speculation { draft_levels: 2 }),
            ..DecodeConfig::default()
        };
        let err = decoder
            .decode(None, 1, &spec, &DecodeBudget::unlimited(), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("draft"), "{err}");
        let cached = DecodeConfig {
            outline_cache: Some(OutlineCacheUse {
                levels: 2,
                bypass_probability: 0.0,
                model_id: "m".into(),
            }),
            ..DecodeConfig::default()
        };
        assert!(decoder
            .decode(None, 1, &cached, &DecodeBudget::unlimited(), None)
            .is_err());
    }

    #[test]
    fn test_plain_decode_makes_k_calls_and_emits_codec_argmax() {
        let (codec, model) = setup(2);
        let device = Default::default();
        let out = Decoder::new(&codec, &model, &device)
            .decode(
                None,
                2,
                &DecodeConfig::default(),
                &DecodeBudget::unlimited(),
                None,
            )
            .unwrap();
        assert_eq!(out.report.level_calls, 4);
        assert_eq!(out.report.levels_used, 4);
        assert_eq!(out.report.decodes, 1);
        assert_eq!(out.tokens.len(), 2);
        assert_eq!(out.codes.len(), 4);
        // The tokens are the codec's argmax on the emitted codes.
        let zhat = codec
            .reconstruct(&out.code_tensors::<B>(&device).unwrap(), 2)
            .unwrap();
        let logits = codec.decode_logits(zhat).unwrap().argmax(2).reshape([2, 8]);
        assert_eq!(
            logits.into_data(),
            out.token_tensor::<B>(&device).unwrap().into_data()
        );
    }

    #[test]
    fn test_budget_truncation_emits_a_partial_reconstruction() {
        let (codec, model) = setup(3);
        let device = Default::default();
        let decoder = Decoder::new(&codec, &model, &device);
        let budget = DecodeBudget {
            max_level_calls: 2,
            ..DecodeBudget::unlimited()
        };
        let out = decoder
            .decode(None, 1, &DecodeConfig::default(), &budget, None)
            .unwrap();
        assert!(out.report.stopped_by_budget);
        assert_eq!(out.report.levels_used, 2);
        let none = DecodeBudget {
            max_level_calls: 0,
            ..DecodeBudget::unlimited()
        };
        let err = decoder
            .decode(None, 1, &DecodeConfig::default(), &none, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("max_level_calls"), "{err}");
    }

    #[test]
    fn test_batched_speculation_with_an_exact_draft_saves_calls() {
        let (codec, model) = setup(4);
        let model = model.batching(true);
        let device = Default::default();
        let config = DecodeConfig {
            speculation: Some(Speculation { draft_levels: 4 }),
            ..sampling(9)
        };
        let out = Decoder::new(&codec, &model, &device)
            .with_draft(&model)
            .decode(None, 2, &config, &DecodeBudget::unlimited(), None)
            .unwrap();
        assert_eq!(out.report.level_calls, 1, "{:?}", out.report);
        assert_eq!(out.report.level_evaluations, 4);
        assert_eq!(out.report.accepted_drafts, 4);
        assert_eq!(out.report.draft_calls, 4);
    }

    #[test]
    fn test_early_exit_emits_the_codec_argmax_of_the_partial_reconstruction() {
        let (codec, model) = setup(5);
        let device = Default::default();
        let config = DecodeConfig {
            early_exit: Some(EarlyExit {
                threshold: 0.0,
                min_levels: 2,
            }),
            ..sampling(3)
        };
        let out = Decoder::new(&codec, &model, &device)
            .decode(None, 1, &config, &DecodeBudget::unlimited(), None)
            .unwrap();
        assert_eq!(out.report.early_exit_level, Some(2));
        assert_eq!(out.report.levels_used, 2);
        assert_eq!(
            out.report.decodes, 1,
            "the probe's decode is reused for the output"
        );
        let zhat = codec
            .reconstruct(&out.code_tensors::<B>(&device).unwrap(), 1)
            .unwrap();
        let tokens = codec.decode_logits(zhat).unwrap().argmax(2).reshape([1, 8]);
        assert_eq!(
            tokens.into_data(),
            out.token_tensor::<B>(&device).unwrap().into_data()
        );
    }

    #[test]
    fn test_outline_cache_hits_skip_calls_and_never_cross_model_ids() {
        let (codec, model) = setup(6);
        let device = Default::default();
        let counting = CountingModel::new(&model);
        let decoder = Decoder::new(&codec, &counting, &device);
        let prompt = || {
            Tensor::<B, 1, Int>::from_ints([3i64, 1, 4, 1, 5, 9].as_slice(), &device)
                .reshape([2, 3])
        };
        let mut cache = OutlineCache::new(4).unwrap();
        let config = |id: &str, bypass: f64| DecodeConfig {
            outline_cache: Some(OutlineCacheUse {
                levels: 2,
                bypass_probability: bypass,
                model_id: id.into(),
            }),
            ..sampling(11)
        };
        let budget = DecodeBudget::unlimited();
        let first = decoder
            .decode(
                Some(prompt()),
                2,
                &config("a", 0.0),
                &budget,
                Some(&mut cache),
            )
            .unwrap();
        assert_eq!(first.report.cache_misses, 2);
        assert_eq!(first.report.level_calls, 4);
        assert_eq!(cache.len(), 2);
        let second = decoder
            .decode(
                Some(prompt()),
                2,
                &config("a", 0.0),
                &budget,
                Some(&mut cache),
            )
            .unwrap();
        assert_eq!(second.report.cache_hits, 2);
        assert_eq!(second.report.cached_levels, 2);
        assert_eq!(second.report.level_calls, 2);
        assert_eq!(&second.codes[..2], &first.codes[..2]);
        // A different model id never sees model a's outlines.
        let other = decoder
            .decode(
                Some(prompt()),
                2,
                &config("b", 0.0),
                &budget,
                Some(&mut cache),
            )
            .unwrap();
        assert_eq!(other.report.cache_hits, 0);
        assert_eq!(other.report.level_calls, 4);
        // Bypass forces fresh generation.
        let flood = decoder
            .decode(
                Some(prompt()),
                2,
                &config("a", 1.0),
                &budget,
                Some(&mut cache),
            )
            .unwrap();
        assert!(flood.report.cache_bypassed);
        assert_eq!(flood.report.level_calls, 4);
        assert_eq!(counting.passes(), 4 + 2 + 4 + 4);
    }

    #[test]
    fn test_outline_cache_evicts_the_least_recently_used() {
        let key = |p: i64| OutlineKey {
            model_id: "m".into(),
            prompt: vec![p],
            lengths: vec![1],
            codebook: 2,
            levels: 1,
        };
        let mut cache = OutlineCache::new(2).unwrap();
        cache.put(key(1), vec![vec![0]]);
        cache.put(key(2), vec![vec![1]]);
        assert!(cache.get(&key(1)).is_some());
        cache.put(key(3), vec![vec![1]]);
        assert!(cache.get(&key(2)).is_none(), "2 was least recently used");
        assert!(cache.get(&key(1)).is_some() && cache.get(&key(3)).is_some());
        assert!(OutlineCache::new(0).is_err());
    }

    #[test]
    fn test_counting_codec_sees_what_the_report_says() {
        let (codec, model) = setup(7);
        let device = Default::default();
        let counted = CountingCodec::new(&codec);
        let config = DecodeConfig {
            repair: Some(Repair {
                confidence_guard: 2.0,
                min_alternative_probability: 0.0,
            }),
            early_exit: Some(EarlyExit {
                threshold: 2.0,
                min_levels: 1,
            }),
            ..sampling(5)
        };
        let out = Decoder::new(&counted, &model, &device)
            .decode(None, 2, &config, &DecodeBudget::unlimited(), None)
            .unwrap();
        assert_eq!(
            counted.counts(),
            (
                out.report.reconstructs,
                out.report.decodes,
                out.report.encodes
            )
        );
        assert!(out.report.repair_checks > 0);
    }

    #[test]
    fn test_real_tokenizer_and_generator_decode_through_the_strategies() {
        // The untrained NCM tokenizer and generator (random weights), bound
        // as `ConceptCodec` and `LevelModel`: the strategies' identities hold
        // on the real implementations, not just on the doubles. Nothing here
        // says anything about quality — the models are untrained.
        use crate::ncm::generator::{CodecShape, ConceptFieldLm, NcmGenerator, NcmGeneratorConfig};
        use crate::ncm::tokenizer::{NcmTokenizer, NcmTokenizerConfig};
        use crate::ncm::LevelSchedule;

        let device = Default::default();
        let tokenizer_config = NcmTokenizerConfig {
            latent_dim: 8,
            codebook_size: 16,
            schedule: LevelSchedule::new(vec![1, 2, 4, 8]).unwrap(),
            heads: 2,
            ..NcmTokenizerConfig::tiny().unwrap()
        };
        let tokenizer = NcmTokenizer::<B>::new(&tokenizer_config, &device).unwrap();
        let generator_config = NcmGeneratorConfig {
            blocks: 1,
            hidden: 8,
            heads: 2,
            mlp_ratio: 2,
            prompt_len: 4,
            thinking_steps: 2,
            ..NcmGeneratorConfig::tiny()
        };
        let shape = CodecShape::of(&tokenizer).unwrap();
        let target_model = NcmGenerator::<B>::new(&generator_config, &shape, &device).unwrap();
        let draft_model = NcmGenerator::<B>::new(&generator_config, &shape, &device).unwrap();
        let target =
            ConceptFieldLm::new(&target_model, &tokenizer, generator_config.relax()).unwrap();
        let draft =
            ConceptFieldLm::new(&draft_model, &tokenizer, generator_config.relax()).unwrap();
        let prompt = || {
            Tensor::<B, 1, Int>::from_ints(
                [72i64, 101, 108, 112, 111, 33, 10, 65].as_slice(),
                &device,
            )
            .reshape([2, 4])
        };
        let unlimited = DecodeBudget::unlimited();

        // Plain greedy: K calls, counted independently.
        let counted = CountingModel::new(&target);
        let plain = Decoder::new(&tokenizer, &counted, &device)
            .decode(
                Some(prompt()),
                2,
                &DecodeConfig::default(),
                &unlimited,
                None,
            )
            .unwrap();
        assert_eq!(plain.report.level_calls, 4);
        assert_eq!(counted.passes(), 4);
        assert_eq!(plain.tokens.len(), 2);
        assert!(plain.tokens.iter().all(|row| row.len() == 8));

        // Greedy speculation with an independent draft is plain greedy.
        let spec = DecodeConfig {
            speculation: Some(Speculation { draft_levels: 2 }),
            ..DecodeConfig::default()
        };
        let speculated = Decoder::new(&tokenizer, &target, &device)
            .with_draft(&draft)
            .decode(Some(prompt()), 2, &spec, &unlimited, None)
            .unwrap();
        assert_eq!(speculated.codes, plain.codes);
        assert_eq!(speculated.tokens, plain.tokens);
        // The generator does not batch levels, so speculation saved no calls.
        assert!(
            speculated.report.level_calls >= 4,
            "{:?}",
            speculated.report
        );

        // Early exit that cannot fire, and a guard-0 repair, are plain.
        let inert = DecodeConfig {
            early_exit: Some(EarlyExit {
                threshold: 1.5,
                min_levels: 1,
            }),
            repair: Some(Repair {
                confidence_guard: 0.0,
                min_alternative_probability: 0.0,
            }),
            ..DecodeConfig::default()
        };
        let out = Decoder::new(&tokenizer, &target, &device)
            .decode(Some(prompt()), 2, &inert, &unlimited, None)
            .unwrap();
        assert_eq!(out.codes, plain.codes);
        assert_eq!(out.tokens, plain.tokens);
        assert!(out.report.repair_checks > 0 && out.report.probes > 0);

        // Everything at once under a tight budget stays within it.
        let everything = DecodeConfig {
            sampler: CodeSampler {
                temperature: 1.0,
                top_k: Some(4),
            },
            seed: 3,
            early_exit: Some(EarlyExit {
                threshold: 0.9,
                min_levels: 2,
            }),
            speculation: Some(Speculation { draft_levels: 2 }),
            repair: Some(Repair {
                confidence_guard: 1.5,
                min_alternative_probability: 0.0,
            }),
            particles: Some(Particles {
                count: 3,
                tilt: 0.5,
                ess_threshold: 0.8,
            }),
            outline_cache: Some(OutlineCacheUse {
                levels: 1,
                bypass_probability: 0.0,
                model_id: "untrained".into(),
            }),
        };
        let budget = DecodeBudget {
            max_level_calls: 5,
            max_draft_calls: 3,
            max_repairs: 1,
            max_particles: 2,
            max_probes: 3,
        };
        let mut cache = OutlineCache::new(4).unwrap();
        for _ in 0..2 {
            let out = Decoder::new(&tokenizer, &target, &device)
                .with_draft(&draft)
                .decode(Some(prompt()), 2, &everything, &budget, Some(&mut cache))
                .unwrap();
            budget.admits(&out.report).unwrap();
            assert_eq!(out.report.particles, 2);
        }
    }

    #[test]
    fn test_certificates_hold() {
        let certificates = certificate_checks().unwrap();
        assert_eq!(certificates.len(), 8);
        for c in &certificates {
            eprintln!(
                "{}::{} residual {:e} tolerance {:e}",
                c.group, c.name, c.residual, c.tolerance
            );
        }
        for c in &certificates {
            assert!(
                c.passed(),
                "{}::{} residual {} > {}",
                c.group,
                c.name,
                c.residual,
                c.tolerance
            );
        }
    }
}
