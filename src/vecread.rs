//! Vecread: reading the model's raw-vector thought.
//!
//! The models in this crate reason in raw vectors: the student's trunk hidden
//! states and its geometric stream's slot states in a learned Riemannian
//! metric ([`crate::geometry`], [`crate::student`]). This module is the
//! readout lens that makes that thinking legible *without constraining it*:
//! the model keeps its native math, and vecread translates intermediate
//! states into vocabulary space after the fact.
//!
//! # Which lens this is
//!
//! The method is the **logit lens** (nostalgebraist, 2021; refined as the
//! *tuned lens* by Belrose et al., 2023): an intermediate state is decoded by
//! passing it through the model's *own* readout — for the student's trunk
//! that is the final norm plus the tied embedding transpose, exactly
//! [`crate::lm::LanguageModel::logits_from_hidden`]. No separate probe is
//! trained, so the lens cannot drift from the model it reads. Applied across
//! a sweep of the geometric stream's certified refinement depth
//! ([`crate::student::GeometricStudent::forward_depth`]), it turns a
//! continuous-thought trajectory (COCONUT-style latent reasoning; here the
//! latent is read, not trained to be legible) into a sequence of decodes: how
//! the answer tokens crystallize across refinement depth.
//!
//! # Tuned-lens drift correction
//!
//! The raw lens is exact at the output layer but systematically
//! basis-mismatched at intermediate refine depths. [`train_translators`]
//! fits an optional per-depth affine translator `T_d(h) = A_d h + b_d`
//! (hidden-size square matrices) by minimizing
//! `KL(student final-output distribution || lens(T_d(intermediate state)))`
//! on prompts from the student's own byte-token domain — the tuned lens of
//! Belrose et al. (arXiv:2303.08112). A reader with `translators: None` is
//! bit-identical to the raw lens; the certificates pin that.
//!
//! # Readable is not causal
//!
//! The field's hard-won lesson (Lanham et al., arXiv:2307.13702; RepE
//! steering work): a legible decode is not evidence the decode reflects what
//! drives the answer. The `vecread` certificate group therefore includes
//! *causal validation*, not just exactness: `lens_decode_tracks_perturbed_trajectory`
//! perturbs the refinement and requires the decode to move, and
//! `answer_depends_on_refine_steps` requires the final logits to actually
//! depend on the refinement depth. A lens that reads a decoration fails
//! these.
//!
//! # Metric-native readout
//!
//! The geometry is the building; vocabulary is a flashlight. Every
//! [`DepthTrace`] therefore also carries a [`SlotGeometry`] section: per-slot
//! RMS and a pairwise distance summary computed under the *learned* metric
//! ([`crate::geometry::MetricField::sq_dist`]) — never Euclidean claims about
//! a space whose distances are learned.
//!
//! Two honesty properties are certificates, not prose (the `vecread` group of
//! [`crate::verify`]):
//!
//! - At the output layer the lens is **exact**: the decode of the final
//!   hidden state reproduces the model's own logits argmax at tolerance 0.0 —
//!   it *is* the model's readout, not an approximation of it.
//! - Every decode is cut from a normalized softmax row (tolerance 1e-6).
//!
//! Before the output layer the lens is an **approximation**, and the rendered
//! trace says so on its first line ([`LENS_DISCLAIMER`]): the model did not
//! think these words; this is the closest vocabulary projection of its slot
//! states.

use anyhow::Context;
use burn::module::{Module, Param};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{
    activation::{log_softmax, softmax},
    backend::Backend,
    Int, Tensor, TensorData,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::geometry::MetricField;
use crate::lm::LanguageModel;
use crate::student::GeometricStudent;
use crate::tokenizer::{Special, BYTE_TOKENS};

/// The honesty header every rendered trace opens with. The lens decodes slot
/// states into vocabulary space; the model itself never saw these words.
pub const LENS_DISCLAIMER: &str =
    "APPROXIMATION: the model did not think these words; this is the \
     closest vocabulary projection of its slot states (logit lens through the trunk's own final \
     norm and tied embedding).";

fn default_top_k() -> usize {
    5
}

fn default_per_step() -> bool {
    true
}

fn default_span_end() -> usize {
    usize::MAX
}

/// Which positions of the hidden state a decode covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PositionSelection {
    /// Only the last position — where the next-token answer comes from.
    #[default]
    Last,
    /// Every position.
    All,
    /// The half-open span `[start, end)`; the default `end` is the sequence
    /// length. Out-of-range spans are an error naming both bounds.
    Span {
        #[serde(default)]
        start: usize,
        #[serde(default = "default_span_end")]
        end: usize,
    },
}

/// A fitted per-depth affine translator `T_d(h) = A_d h + b_d`: the tuned
/// lens's drift correction (Belrose et al., arXiv:2303.08112). `matrix` is
/// row-major `hidden x hidden`; applied to `[b, n, hidden]` states as
/// `h.matmul(A^T) + b`. Fitted by [`train_translators`]; the two KL fields
/// are the measured decode divergence from the student's own final-output
/// distribution before and after fitting, so a translator reports what it
/// bought.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AffineTranslator {
    /// The refine depth this translator corrects.
    pub depth: usize,
    /// Row-major `hidden x hidden`.
    pub matrix: Vec<f32>,
    pub bias: Vec<f32>,
    /// Mean per-position KL(final || lens(translated)) on the fit set before
    /// fitting (the raw lens's drift) and after.
    pub initial_kl: f32,
    pub final_kl: f32,
}

/// Hyperparameters of [`train_translators`]. Every field carries
/// `#[serde(default)]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslatorFit {
    /// AdamW steps per translator.
    #[serde(default = "default_fit_steps")]
    pub steps: usize,
    #[serde(default = "default_fit_lr")]
    pub lr: f64,
}

fn default_fit_steps() -> usize {
    200
}

fn default_fit_lr() -> f64 {
    0.05
}

impl Default for TranslatorFit {
    fn default() -> Self {
        Self {
            steps: default_fit_steps(),
            lr: default_fit_lr(),
        }
    }
}

/// The vecread configuration. Every field carries `#[serde(default)]`, so a
/// sidecar written by an older build still parses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VecReader {
    /// How many top tokens each decode reports.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Which positions of the hidden state are decoded.
    #[serde(default)]
    pub positions: PositionSelection,
    /// `true` (default): decode at every requested refine depth — the
    /// crystallization trace. `false`: decode only at the deepest requested
    /// depth.
    #[serde(default = "default_per_step")]
    pub per_step: bool,
    /// Optional tuned-lens translators, one per depth, from
    /// [`train_translators`]. `None` (the default) is the raw logit lens, bit
    /// for bit. A requested depth with no translator decodes raw.
    #[serde(default)]
    pub translators: Option<Vec<AffineTranslator>>,
}

impl Default for VecReader {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
            positions: PositionSelection::default(),
            per_step: default_per_step(),
            translators: None,
        }
    }
}

impl VecReader {
    /// Cross-check the config, naming the field at fault.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.top_k >= 1,
            "top_k must be at least 1, got {}",
            self.top_k
        );
        if let Some(translators) = &self.translators {
            for (i, translator) in translators.iter().enumerate() {
                for later in translators.iter().skip(i + 1) {
                    anyhow::ensure!(
                        translator.depth != later.depth,
                        "two translators claim depth {}",
                        translator.depth
                    );
                }
            }
        }
        Ok(())
    }

    /// The positions this reader decodes for a sequence of `n` tokens.
    fn selected_positions(&self, n: usize) -> anyhow::Result<Vec<usize>> {
        match self.positions {
            PositionSelection::Last => Ok(vec![n - 1]),
            PositionSelection::All => Ok((0..n).collect()),
            PositionSelection::Span { start, end } => {
                let end = if end == usize::MAX { n } else { end };
                anyhow::ensure!(
                    start < end && end <= n,
                    "the decode span [{start}, {end}) is empty or out of range for a sequence of {n} tokens"
                );
                Ok((start..end).collect())
            }
        }
    }

    /// Read one student's thinking over `tokens`: run the student's geometric
    /// pass at each depth of `depth_sweep` and decode the resulting hidden
    /// states through the trunk's own readout (with the fitted translator for
    /// the depth, when one is installed).
    ///
    /// Deterministic: the forward pass is a pure function of the weights and
    /// the tokens, so two calls over the same model agree bit for bit. The
    /// sweep must be non-empty and every depth at least 1 — a depth of 0
    /// would skip the certified refinement entirely, which is not a thought
    /// trace.
    pub fn read_student_thinking<B: Backend<FloatElem = f32>>(
        &self,
        student: &GeometricStudent<B>,
        tokens: Tensor<B, 2, Int>,
        depth_sweep: &[usize],
    ) -> anyhow::Result<ThoughtTrace> {
        self.validate()?;
        anyhow::ensure!(
            !depth_sweep.is_empty(),
            "the depth sweep must request at least one refine depth"
        );
        for depth in depth_sweep {
            anyhow::ensure!(
                *depth >= 1,
                "a refine depth must be at least 1, got {depth} (0 would skip the refinement entirely)"
            );
        }
        let [b, n] = tokens.dims();
        anyhow::ensure!(
            b == 1,
            "the lens reads one prompt at a time, got a batch of {b}"
        );
        anyhow::ensure!(n >= 1, "the prompt must hold at least one token");
        let positions = self.selected_positions(n)?;

        let depths: Vec<usize> = if self.per_step {
            depth_sweep.to_vec()
        } else {
            let deepest = depth_sweep.iter().copied().max().ok_or_else(|| {
                anyhow::anyhow!("the depth sweep must request at least one refine depth")
            })?;
            vec![deepest]
        };
        let mut traces = Vec::with_capacity(depths.len());
        let mut translators_used = Vec::new();
        for depth in depths {
            let pass = student.slots_at_depth(tokens.clone(), Some(depth))?;
            let energy =
                pass.refine.energy.last().copied().ok_or_else(|| {
                    anyhow::anyhow!("depth {depth} produced an empty energy trace")
                })?;
            let displacement = pass.refine.displacement.last().copied().ok_or_else(|| {
                anyhow::anyhow!("depth {depth} produced an empty displacement trace")
            })?;
            let translator = self
                .translators
                .as_ref()
                .and_then(|ts| ts.iter().find(|t| t.depth == depth));
            if let Some(t) = translator {
                translators_used.push(TranslatorUse {
                    depth,
                    kl_before: t.initial_kl,
                    kl_after: t.final_kl,
                });
            }
            let decodes = decode_hidden(
                student.trunk(),
                &pass.hidden,
                &positions,
                self.top_k,
                DecodeSite {
                    depth,
                    energy,
                    displacement,
                    translator,
                },
            )?;
            let geometry = slot_geometry(student.geom_metric(), &pass.slots)?;
            traces.push(DepthTrace {
                depth,
                energy: pass.refine.energy,
                displacement: pass.refine.displacement,
                decodes,
                geometry,
            });
        }
        Ok(ThoughtTrace {
            n_tokens: n,
            positions,
            top_k: self.top_k,
            depths: traces,
            translators_used,
        })
    }
}

/// [`VecReader::read_student_thinking`] with the default reader (top-5, last
/// position, per-step, raw lens).
pub fn read_student_thinking<B: Backend<FloatElem = f32>>(
    student: &GeometricStudent<B>,
    tokens: Tensor<B, 2, Int>,
    depth_sweep: &[usize],
) -> anyhow::Result<ThoughtTrace> {
    VecReader::default().read_student_thinking(student, tokens, depth_sweep)
}

/// One decoded token candidate: the id, its printable text and its
/// probability under the model's own readout.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenCandidate {
    pub id: u16,
    pub text: String,
    /// Softmax probability at this id, from the full-vocabulary row the
    /// top-k was cut from.
    pub probability: f32,
}

/// One decode target: one position's hidden state read through the lens at
/// one refine depth.
#[derive(Debug, Clone, PartialEq)]
pub struct LensReport {
    /// The refine depth this state came from.
    pub depth: usize,
    /// The sequence position decoded.
    pub position: usize,
    /// RMS of the hidden state at this position (over the hidden dim).
    pub rms: f32,
    /// The certified energy after `depth` refine steps (the trace's last
    /// entry at this depth).
    pub energy: f32,
    /// Mean squared displacement of the `depth`-th refine step.
    pub displacement: f32,
    /// Sum of the full softmax row the top-k was cut from: 1.0 up to float
    /// error. Kept on the record so "the decode is a distribution" is a
    /// number the certificates can check, not a claim.
    pub prob_sum: f32,
    /// The top-k tokens at this position, best first.
    pub top: Vec<TokenCandidate>,
}

/// The slot-space half of a depth entry, computed under the *learned* metric
/// of the geometric stream — no Euclidean neighborhood claims about a space
/// whose distances are learned.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotGeometry {
    /// RMS of each relaxed slot state (over the geometric dim), in slot
    /// order.
    pub slot_rms: Vec<f32>,
    /// Min / median / max of the pairwise learned-metric distances between
    /// distinct slots (`sqrt(sq_dist)` of [`MetricField::sq_dist`]).
    pub dist_min: f32,
    pub dist_median: f32,
    pub dist_max: f32,
}

/// One depth of a thought trace: the certified per-step energy and
/// displacement, the slot geometry of the relaxed state, and the decodes of
/// the state the refinement landed on.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthTrace {
    /// How many certified refine steps this depth ran.
    pub depth: usize,
    /// Energy after each refine step (`len == depth`); descends by
    /// construction (the `geom` certificates).
    pub energy: Vec<f32>,
    /// Mean squared displacement of each refine step (`len == depth`).
    pub displacement: Vec<f32>,
    /// One [`LensReport`] per decoded position at this depth's final state.
    pub decodes: Vec<LensReport>,
    /// The relaxed slots' geometry under the learned metric.
    pub geometry: SlotGeometry,
}

/// Record of a translator that was applied while reading: which depth, and
/// the decode KL it was measured to correct when fitted.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslatorUse {
    pub depth: usize,
    pub kl_before: f32,
    pub kl_after: f32,
}

/// A thought trace: the sequence of per-depth decodes of one prompt — how
/// the answer tokens crystallize across refinement depth.
#[derive(Debug, Clone, PartialEq)]
pub struct ThoughtTrace {
    /// Tokens in the prompt (including the document markers).
    pub n_tokens: usize,
    /// The positions decoded at every depth.
    pub positions: Vec<usize>,
    pub top_k: usize,
    /// One entry per requested depth (per-step) or a single entry at the
    /// deepest depth (final-only).
    pub depths: Vec<DepthTrace>,
    /// Translators applied during the read (empty: the raw logit lens).
    pub translators_used: Vec<TranslatorUse>,
}

impl ThoughtTrace {
    /// The plain-text table a human reads in a terminal: the honesty header,
    /// then per depth the certified energy/displacement of every refine step,
    /// the slot geometry under the learned metric, and the top-k decode of
    /// each covered position. No emoji.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(LENS_DISCLAIMER);
        out.push('\n');
        out.push_str(&format!(
            "lens: logit lens through the trunk's final norm + tied embedding (exact at the \
             output layer) | prompt tokens: {} | positions decoded: {} | top-{}\n",
            self.n_tokens,
            self.positions.len(),
            self.top_k
        ));
        if !self.translators_used.is_empty() {
            let uses = self
                .translators_used
                .iter()
                .map(|u| {
                    format!(
                        "depth {} KL {:.4} -> {:.4}",
                        u.depth, u.kl_before, u.kl_after
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "tuned-lens drift correction applied (trained affine translators): {uses}\n"
            ));
        }
        for depth in &self.depths {
            out.push_str(&format!(
                "\ndepth {}: {} certified refine step(s)\n",
                depth.depth,
                depth.energy.len()
            ));
            out.push_str("  step      energy  displacement\n");
            for (i, (energy, disp)) in depth
                .energy
                .iter()
                .zip(depth.displacement.iter())
                .enumerate()
            {
                out.push_str(&format!("  {:>4}  {energy:>11.6}  {disp:>12.6}\n", i + 1));
            }
            let rms = depth
                .geometry
                .slot_rms
                .iter()
                .map(|r| format!("{r:.4}"))
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!(
                "  slot geometry (learned metric G): slot rms [{rms}] | pairwise dist min {:.4} \
                 median {:.4} max {:.4}\n",
                depth.geometry.dist_min, depth.geometry.dist_median, depth.geometry.dist_max
            ));
            for dec in &depth.decodes {
                let top = dec
                    .top
                    .iter()
                    .map(|c| format!("\"{}\" {:.4}", c.text, c.probability))
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push_str(&format!(
                    "  pos {:>4}  rms {:>9.4}  energy {:>11.6}  disp {:>10.6}  | {top}\n",
                    dec.position, dec.rms, dec.energy, dec.displacement
                ));
            }
        }
        out
    }
}

/// The printable form of one token id: the special's name, the ASCII byte
/// itself when printable, `\xNN` for the other bytes.
pub fn token_text(id: u16) -> String {
    if let Some(special) = Special::from_id(id) {
        return special.name().to_string();
    }
    if usize::from(id) < BYTE_TOKENS {
        // A byte id is a byte value by construction (tokenizer.rs: a raw
        // byte's id is exactly its value), so the try_from cannot fail.
        match u8::try_from(id) {
            Ok(b) if (0x20..=0x7e).contains(&b) => (b as char).to_string(),
            Ok(b) => format!("\\x{b:02x}"),
            Err(_) => format!("<id {id}>"),
        }
    } else {
        format!("<id {id}>")
    }
}

/// Deterministic synthetic prompts in the student's byte-token domain:
/// seeded ChaCha draws of printable ASCII, one id sequence per prompt. These
/// are fit data for [`train_translators`], not text the model has ever
/// claimed to understand.
pub fn synthetic_prompts(count: usize, len: usize, seed: u64) -> Vec<Vec<u16>> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..len).map(|_| rng.random_range(0x20u16..=0x7e)).collect())
        .collect()
}

/// The per-depth affine translator's parameters as a Burn module, so the
/// crate's standard optimizer machinery (`GradientsParams`, AdamW) applies.
#[derive(Module, Debug)]
struct TranslatorModule<B: Backend> {
    matrix: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
}

/// Mean per-position `KL(final || lens(T(h)))` of one translator over one
/// prompt's intermediate hidden states.
fn translated_kl<B: Backend<FloatElem = f32>>(
    trunk: &LanguageModel<B>,
    module: &TranslatorModule<B>,
    hidden: &Tensor<B, 3>,
    target_logp: &Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [b, n, h] = hidden.dims();
    let translated = hidden
        .clone()
        .reshape([b * n, h])
        .matmul(module.matrix.val().transpose())
        .reshape([b, n, h])
        + module.bias.val().reshape([1, 1, h]);
    let logits = trunk.logits_from_hidden(translated);
    let logq = log_softmax(logits, 2);
    let p = target_logp.clone().exp();
    (p * (target_logp.clone() - logq))
        .sum()
        .div_scalar(n as f32)
}

/// Fit one affine translator per requested depth: the tuned lens's drift
/// correction, trained by minimizing
/// `KL(student final-output distribution || lens(T_d(h_d)))` over `prompts`,
/// where the final distribution is the student's standard forward (its
/// configured refine depth) and `h_d` the hidden state after `d` refine
/// steps.
///
/// The student is converted to the autodiff backend through the crate's
/// checkpoint round trip (full-precision msgpack, bit-exact), so a trained
/// checkpoint fits as-is. Deterministic by construction: identity init, zero
/// bias, fixed prompt order, zero weight decay — no RNG participates. On a
/// fresh zero-init student the geometric stream is an exact identity, every
/// depth's hidden state equals the final one, and the initial KL is already
/// ~0: there is no drift to correct and the translator stays the identity.
/// That is the honest degenerate case, not a bug.
pub fn train_translators<B: Backend<FloatElem = f32>>(
    student: &GeometricStudent<B>,
    prompts: &[Vec<u16>],
    depths: &[usize],
    fit: &TranslatorFit,
) -> anyhow::Result<Vec<AffineTranslator>> {
    anyhow::ensure!(!prompts.is_empty(), "fitting needs at least one prompt");
    for (i, prompt) in prompts.iter().enumerate() {
        anyhow::ensure!(!prompt.is_empty(), "prompt {i} is empty");
    }
    anyhow::ensure!(
        !depths.is_empty(),
        "fitting needs at least one refine depth"
    );
    for depth in depths {
        anyhow::ensure!(
            *depth >= 1,
            "a refine depth must be at least 1, got {depth}"
        );
    }
    anyhow::ensure!(fit.steps >= 1, "fit steps must be at least 1");
    anyhow::ensure!(fit.lr > 0.0, "fit lr must be positive, got {}", fit.lr);

    type A<B> = burn::backend::Autodiff<B>;
    let device_owned: B::Device = Default::default();
    let device = &device_owned;

    // Burn initializes lazily: cloning a student with unforced parameters
    // redraws them (AGENTS.md), which would make two fits of the same student
    // differ. Force first; already-materialized parameters are untouched.
    crate::tensor_ext::force_initialization(student);

    // Backend conversion through the sanctioned checkpoint path: save to a
    // scratch file, load into the autodiff backend, remove the scratch. The
    // round trip is full-precision, so the autodiff student IS this student.
    let scratch = std::env::temp_dir().join(format!(
        "vecread-fit-{}-{}",
        std::process::id(),
        FIT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&scratch).with_context(|| format!("create {}", scratch.display()))?;
    let converted = (|| -> anyhow::Result<GeometricStudent<A<B>>> {
        let path =
            crate::checkpoint::save_content_addressed(student.clone(), &scratch, "vecread-fit")?;
        let fresh = GeometricStudent::<A<B>>::new(student.config(), device)?;
        crate::checkpoint::load::<A<B>, _>(fresh, &path, device)
    })();
    // Scratch cleanup is best-effort: a leftover temp dir is clutter, not
    // incorrectness, and the error (if any) from the conversion above is the
    // one worth reporting.
    drop(std::fs::remove_dir_all(&scratch));
    let student_ad = converted?;

    let hidden_size = student.config().hidden_size;
    let to_tokens = |ids: &[u16]| {
        let flat: Vec<i64> = ids.iter().map(|t| i64::from(*t)).collect();
        Tensor::<A<B>, 1, Int>::from_ints(flat.as_slice(), device).reshape([1, ids.len()])
    };
    // The target: the student's own final-output distribution per prompt,
    // detached. log_softmax of finite logits is finite, so p * logp has no
    // 0 * -inf hazard.
    let mut targets = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let logits = student_ad.forward(to_tokens(prompt))?.logits;
        targets.push(log_softmax(logits, 2).detach());
    }

    let mut translators = Vec::with_capacity(depths.len());
    for &depth in depths {
        let mut hiddens = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let pass = student_ad.slots_at_depth(to_tokens(prompt), Some(depth))?;
            hiddens.push(pass.hidden.detach());
        }
        let mut module = TranslatorModule {
            matrix: Param::from_tensor(Tensor::<A<B>, 2>::eye(hidden_size, device)),
            bias: Param::from_tensor(Tensor::<A<B>, 1>::zeros([hidden_size], device)),
        };
        let mean_kl = |module: &TranslatorModule<A<B>>| -> anyhow::Result<f32> {
            let mut total = Tensor::<A<B>, 1>::zeros([1], device);
            for (hidden, target_logp) in hiddens.iter().zip(targets.iter()) {
                total = total + translated_kl(student_ad.trunk(), module, hidden, target_logp);
            }
            Ok((total / prompts.len() as f32).into_scalar())
        };
        let initial_kl = mean_kl(&module)?;
        let mut optim = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<A<B>, TranslatorModule<A<B>>>();
        for _ in 0..fit.steps {
            let mut loss = Tensor::<A<B>, 1>::zeros([1], device);
            for (hidden, target_logp) in hiddens.iter().zip(targets.iter()) {
                loss = loss + translated_kl(student_ad.trunk(), &module, hidden, target_logp);
            }
            let loss = loss / prompts.len() as f32;
            let grads = GradientsParams::from_grads(loss.backward(), &module);
            module = optim.step(fit.lr, module, grads);
        }
        let final_kl = mean_kl(&module)?;
        let matrix: Vec<f32> = module
            .matrix
            .val()
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let bias: Vec<f32> = module
            .bias
            .val()
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        translators.push(AffineTranslator {
            depth,
            matrix,
            bias,
            initial_kl,
            final_kl,
        });
    }
    Ok(translators)
}

/// Disambiguates the scratch dir of concurrent translator fits in one
/// process.
static FIT_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Apply a fitted translator: `T(h) = A h + b` on `[b, n, hidden]`.
fn apply_translator<B: Backend<FloatElem = f32>>(
    hidden: &Tensor<B, 3>,
    translator: &AffineTranslator,
) -> anyhow::Result<Tensor<B, 3>> {
    let [b, n, h] = hidden.dims();
    anyhow::ensure!(
        translator.matrix.len() == h * h,
        "translator for depth {} holds {} matrix entries, the hidden size {h} needs {}",
        translator.depth,
        translator.matrix.len(),
        h * h
    );
    anyhow::ensure!(
        translator.bias.len() == h,
        "translator for depth {} holds a {}-wide bias for hidden size {h}",
        translator.depth,
        translator.bias.len()
    );
    let device = hidden.device();
    let matrix =
        Tensor::<B, 2>::from_data(TensorData::new(translator.matrix.clone(), [h, h]), &device);
    let bias =
        Tensor::<B, 3>::from_data(TensorData::new(translator.bias.clone(), [1, 1, h]), &device);
    Ok(hidden
        .clone()
        .reshape([b * n, h])
        .matmul(matrix.transpose())
        .reshape([b, n, h])
        + bias)
}

/// The slot-geometry section of one depth entry: per-slot RMS and the
/// pairwise distance summary under the learned metric.
fn slot_geometry<B: Backend<FloatElem = f32>>(
    metric: &MetricField<B>,
    slots: &Tensor<B, 3>,
) -> anyhow::Result<SlotGeometry> {
    let [b, k, d] = slots.dims();
    anyhow::ensure!(
        b == 1,
        "the lens reads one prompt at a time, got a batch of {b}"
    );
    anyhow::ensure!(k >= 2, "a slot geometry needs at least 2 slots, got {k}");
    let host: Vec<f32> = slots
        .clone()
        .reshape([k, d])
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let slot_rms: Vec<f32> = (0..k)
        .map(|s| {
            let row = &host[s * d..(s + 1) * d];
            (row.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt()
        })
        .collect();
    let d2 = metric.sq_dist(slots.clone(), slots.clone());
    let d2_host: Vec<f32> = d2
        .reshape([k, k])
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let mut dists: Vec<f32> = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in (i + 1)..k {
            // Clamp at 0: G is positive definite by construction, so a
            // negative squared distance can only be float dust.
            dists.push(d2_host[i * k + j].max(0.0).sqrt());
        }
    }
    dists.sort_by(f32::total_cmp);
    let min = *dists
        .first()
        .ok_or_else(|| anyhow::anyhow!("a {k}-slot state has no pairwise distances"))?;
    let max = *dists
        .last()
        .ok_or_else(|| anyhow::anyhow!("a {k}-slot state has no pairwise distances"))?;
    let mid = dists.len() / 2;
    let median = if dists.len() % 2 == 1 {
        dists[mid]
    } else {
        (dists[mid - 1] + dists[mid]) / 2.0
    };
    Ok(SlotGeometry {
        slot_rms,
        dist_min: min,
        dist_median: median,
        dist_max: max,
    })
}

/// The per-depth context a decode carries into its [`LensReport`]: which
/// refine depth produced the state, the certified energy and displacement at
/// that depth, and the tuned-lens translator to apply first (if any).
struct DecodeSite<'a> {
    depth: usize,
    energy: f32,
    displacement: f32,
    translator: Option<&'a AffineTranslator>,
}

/// Decode `positions` of `hidden` (`[1, n, hidden]`) through the trunk's own
/// readout: optional translator, final norm, tied embedding transpose,
/// softmax over the full vocabulary, top-k.
///
/// The ranking is done on the logits (softmax is monotone, so the order is
/// identical and tie-breaking stays in the logit domain); the probabilities
/// are read from the softmax row at the chosen indices.
fn decode_hidden<B: Backend<FloatElem = f32>>(
    trunk: &LanguageModel<B>,
    hidden: &Tensor<B, 3>,
    positions: &[usize],
    top_k: usize,
    site: DecodeSite<'_>,
) -> anyhow::Result<Vec<LensReport>> {
    let [b, n, h] = hidden.dims();
    anyhow::ensure!(
        b == 1,
        "the lens reads one prompt at a time, got a batch of {b}"
    );
    let hidden = match site.translator {
        Some(t) => apply_translator(hidden, t)?,
        None => hidden.clone(),
    };
    let logits = trunk.logits_from_hidden(hidden.clone());
    let vocab = logits.dims()[2];
    let probs = softmax(logits.clone(), 2);
    let logits_host: Vec<f32> = logits
        .reshape([n, vocab])
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let probs_host: Vec<f32> = probs
        .reshape([n, vocab])
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let hidden_host: Vec<f32> = hidden
        .reshape([n, h])
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();

    let mut reports = Vec::with_capacity(positions.len());
    for &position in positions {
        anyhow::ensure!(
            position < n,
            "position {position} is out of range for a sequence of {n} tokens"
        );
        let logits_row = &logits_host[position * vocab..(position + 1) * vocab];
        let probs_row = &probs_host[position * vocab..(position + 1) * vocab];
        let hidden_row = &hidden_host[position * h..(position + 1) * h];
        let rms = (hidden_row.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
        let prob_sum: f32 = probs_row.iter().sum();

        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|a, b| {
            logits_row[*b]
                .total_cmp(&logits_row[*a])
                .then_with(|| a.cmp(b))
        });
        let mut top = Vec::with_capacity(top_k.min(vocab));
        for idx in order.into_iter().take(top_k) {
            let id = u16::try_from(idx).with_context(|| {
                format!("vocabulary index {idx} does not fit the u16 token space")
            })?;
            top.push(TokenCandidate {
                id,
                text: token_text(id),
                probability: probs_row[idx],
            });
        }
        reports.push(LensReport {
            depth: site.depth,
            position,
            rms,
            energy: site.energy,
            displacement: site.displacement,
            prob_sum,
            top,
        });
    }
    Ok(reports)
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
    use super::*;
    use crate::student::StudentConfig;
    use burn::backend::NdArray;
    use burn::nn::LinearConfig;

    type B = NdArray<f32>;

    fn tiny_config() -> StudentConfig {
        StudentConfig {
            hidden_size: 64,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            context: 32,
            intermediate_size: 32,
            cond_hidden_size: 16,
            frequency_embedding_size: 16,
            experts_per_box: 2,
            geom_slots: 4,
            geom_dim: 8,
            geom_refine_steps: 2,
            geom_cond_hidden: 16,
            geom_frequency: 8,
            ..StudentConfig::default()
        }
    }

    fn tiny_student(
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> GeometricStudent<B> {
        let student = GeometricStudent::new(&tiny_config(), device).unwrap();
        crate::tensor_ext::force_initialization(&student);
        student
    }

    /// A student whose geometric stream actually writes: the zero-init
    /// readout replaced by a seeded random projection (the same mechanism
    /// the causal certificates use, and for the same reason).
    fn writing_student(
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> GeometricStudent<B> {
        let mut student = tiny_student(device);
        let readout = LinearConfig::new(8, 64).with_bias(true).init(device);
        student.set_geom_readout(readout);
        deterministic_redraw(student)
    }

    /// Redraw every non-constant float parameter from a host-side ChaCha at
    /// a fixed scale.
    ///
    /// `GeometricStudent::new` and `LinearConfig::init` draw from burn's
    /// backend RNG, which is process-global: under `cargo test --lib`'s
    /// default parallelism a neighbour test can reseed it mid-build, and
    /// the fit test then runs on a *different* student each run (the
    /// observed flake: an interleaved build whose depth-1 drift the 60-step
    /// fit did not reduce). Seeding harder cannot fix that — the race is
    /// between the seed and the draws — and neither can matching the built
    /// tensors' scale, because the built values are exactly what the race
    /// corrupts. So the random parameters are redrawn from a seeded host
    /// RNG no other test can touch, at one fixed standard deviation for
    /// every tensor: the fixture needs a valid writing student, not Burn's
    /// init distribution. Constant-initialized parameters (norm gains, zero
    /// biases, the geometry's zero-init metric and projections) are
    /// RNG-independent already and keep their values — including the
    /// structural zeros the stream's identity-at-init property rests on.
    fn deterministic_redraw(student: GeometricStudent<B>) -> GeometricStudent<B> {
        /// Standard deviation of the redrawn weights: mid-range among the
        /// init scales Burn produced for this config (0.016–0.145), small
        /// enough that no block dominates at init.
        const REDRAW_STD: f64 = 0.05;
        struct Redraw(ChaCha12Rng);
        impl Redraw {
            /// Standard normal via Box-Muller; `1.0 - u` keeps the log finite.
            fn normal(&mut self) -> f64 {
                let u1 = 1.0 - self.0.random::<f64>();
                let u2 = self.0.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
        impl burn::module::ModuleMapper<B> for Redraw {
            fn map_float<const D: usize>(
                &mut self,
                param: burn::module::Param<Tensor<B, D>>,
            ) -> burn::module::Param<Tensor<B, D>> {
                let values: Vec<f32> = param.val().to_data().iter::<f32>().collect();
                let constant = values.windows(2).all(|w| w[0] == w[1]);
                if constant {
                    return param;
                }
                let dims = param.val().dims();
                let device = param.val().device();
                let redrawn: Vec<f32> = (0..values.len())
                    .map(|_| (REDRAW_STD * self.normal()) as f32)
                    .collect();
                burn::module::Param::initialized(
                    param.id,
                    Tensor::<B, 1>::from_floats(redrawn.as_slice(), &device).reshape(dims),
                )
            }
        }
        student.map(&mut Redraw(ChaCha12Rng::seed_from_u64(7)))
    }

    fn prompt_tokens(
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        text: &str,
    ) -> Tensor<B, 2, Int> {
        let ids = crate::tokenizer::ByteTokenizer::new().encode_document(text);
        let n = ids.len();
        Tensor::<B, 1, Int>::from_ints(
            ids.iter()
                .map(|t| i64::from(*t))
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        )
        .reshape([1, n])
    }

    #[test]
    fn test_reader_config_serde_defaults() {
        let parsed: VecReader = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, VecReader::default());
        let partial: VecReader =
            serde_json::from_str(r#"{"positions": {"span": {"start": 2}}}"#).unwrap();
        assert_eq!(
            partial.positions,
            PositionSelection::Span {
                start: 2,
                end: usize::MAX
            }
        );
        assert_eq!(partial.translators, None);
        let text = serde_json::to_string(&VecReader::default()).unwrap();
        assert_eq!(
            serde_json::from_str::<VecReader>(&text).unwrap(),
            VecReader::default()
        );
        let err = VecReader {
            top_k: 0,
            ..VecReader::default()
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("top_k"), "{err}");
        // Duplicate translator depths are rejected.
        let dup = AffineTranslator {
            depth: 1,
            matrix: vec![0.0; 64 * 64],
            bias: vec![0.0; 64],
            initial_kl: 0.0,
            final_kl: 0.0,
        };
        let err = VecReader {
            translators: Some(vec![dup.clone(), dup]),
            ..VecReader::default()
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("depth 1"), "{err}");
    }

    #[test]
    fn test_lens_at_output_layer_matches_model_argmax() {
        // The lens at the final state IS the model's readout: top-1 of every
        // decoded position must equal the argmax of the trunk's own logits,
        // exactly (the certificate pins this at tolerance 0.0).
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "fn main() {}");
        let reader = VecReader {
            positions: PositionSelection::All,
            ..VecReader::default()
        };
        let trace = reader
            .read_student_thinking(&student, tokens.clone(), &[2])
            .unwrap();
        let out = student.forward_depth(tokens, Some(2)).unwrap();
        let [_, n, vocab] = out.logits.dims();
        let logits: Vec<f32> = out
            .logits
            .reshape([n, vocab])
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let depth = &trace.depths[0];
        assert_eq!(depth.decodes.len(), n);
        for dec in &depth.decodes {
            let row = &logits[dec.position * vocab..(dec.position + 1) * vocab];
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(
                usize::from(dec.top.first().unwrap().id),
                argmax,
                "position {}",
                dec.position
            );
        }
    }

    #[test]
    fn test_trace_has_one_entry_per_requested_depth() {
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "hello");
        let trace = read_student_thinking(&student, tokens, &[1, 2, 4]).unwrap();
        assert_eq!(trace.depths.len(), 3);
        for (entry, want) in trace.depths.iter().zip([1usize, 2, 4]) {
            assert_eq!(entry.depth, want);
            assert_eq!(entry.energy.len(), want);
            assert_eq!(entry.displacement.len(), want);
            assert_eq!(entry.decodes.len(), 1, "last position only");
        }
    }

    #[test]
    fn test_final_only_decodes_the_deepest_depth() {
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "hello");
        let reader = VecReader {
            per_step: false,
            ..VecReader::default()
        };
        let trace = reader
            .read_student_thinking(&student, tokens, &[1, 4, 2])
            .unwrap();
        assert_eq!(trace.depths.len(), 1);
        assert_eq!(trace.depths[0].depth, 4);
    }

    #[test]
    fn test_render_carries_the_honesty_header() {
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "x");
        let rendered = read_student_thinking(&student, tokens, &[1, 2])
            .unwrap()
            .render();
        assert!(rendered.starts_with("APPROXIMATION"), "{rendered}");
        assert!(rendered.contains("did not think these words"), "{rendered}");
        assert!(rendered.contains("depth 1"), "{rendered}");
        assert!(rendered.contains("energy"), "{rendered}");
        assert!(rendered.contains("displacement"), "{rendered}");
        assert!(
            rendered.contains("slot geometry (learned metric G)"),
            "{rendered}"
        );
    }

    #[test]
    fn test_read_is_deterministic() {
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "fn f() {}");
        let a = read_student_thinking(&student, tokens.clone(), &[1, 2]).unwrap();
        let b = read_student_thinking(&student, tokens, &[1, 2]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.render(), b.render());
    }

    #[test]
    fn test_decode_probabilities_are_finite_and_sum_to_one() {
        // Fresh weights: the end-to-end think run must still produce a
        // normalized softmax row per decode (the lens decodes, it does not
        // need a trained model to be well-formed).
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "let x = 1;");
        let trace = read_student_thinking(&student, tokens, &[1, 2]).unwrap();
        for depth in &trace.depths {
            for dec in &depth.decodes {
                assert!(
                    (dec.prob_sum - 1.0).abs() < 1e-6,
                    "softmax row sums to {}",
                    dec.prob_sum
                );
                assert_eq!(dec.top.len(), 5);
                for cand in &dec.top {
                    assert!(cand.probability.is_finite());
                    assert!(cand.probability > 0.0 && cand.probability <= 1.0);
                }
                assert!(dec.rms.is_finite() && dec.rms > 0.0);
            }
        }
    }

    #[test]
    fn test_bad_inputs_are_loud() {
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "hi");
        let err = read_student_thinking(&student, tokens.clone(), &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least one refine depth"), "{err}");
        let err = read_student_thinking(&student, tokens.clone(), &[0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least 1"), "{err}");
        let reader = VecReader {
            positions: PositionSelection::Span {
                start: 100,
                end: 200,
            },
            ..VecReader::default()
        };
        let err = reader
            .read_student_thinking(&student, tokens.clone(), &[1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("[100, 200)"), "{err}");
        // A translator shaped for a different hidden size names the depth.
        let bad = AffineTranslator {
            depth: 1,
            matrix: vec![0.0; 9],
            bias: vec![0.0; 3],
            initial_kl: 0.0,
            final_kl: 0.0,
        };
        let reader = VecReader {
            translators: Some(vec![bad]),
            ..VecReader::default()
        };
        let err = reader
            .read_student_thinking(&student, tokens, &[1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("depth 1"), "{err}");
    }

    #[test]
    fn test_token_text() {
        assert_eq!(token_text(Special::Bos.id()), "<bos>");
        assert_eq!(token_text(u16::from(b'a')), "a");
        assert_eq!(token_text(0x0a), "\\x0a");
        assert_eq!(token_text(0xff), "\\xff");
        assert_eq!(token_text(400), "<id 400>");
    }

    #[test]
    fn test_slot_geometry_is_metric_native_and_ordered() {
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "geometry");
        let trace = read_student_thinking(&student, tokens, &[2]).unwrap();
        let geometry = &trace.depths[0].geometry;
        assert_eq!(geometry.slot_rms.len(), 4, "one RMS per slot");
        assert!(geometry.slot_rms.iter().all(|r| r.is_finite() && *r >= 0.0));
        assert!(geometry.dist_min >= 0.0);
        assert!(geometry.dist_min <= geometry.dist_median);
        assert!(geometry.dist_median <= geometry.dist_max);
        // The distances really come from the learned metric, not from the
        // Euclidean default: the metric's G at init is the identity only by
        // construction detail, so cross-check against sq_dist directly.
        let pass = student
            .slots_at_depth(prompt_tokens(&device, "geometry"), Some(2))
            .unwrap();
        let d2 = student
            .geom_metric()
            .sq_dist(pass.slots.clone(), pass.slots.clone());
        let k = pass.slots.dims()[1];
        let d2_host: Vec<f32> = d2
            .reshape([k, k])
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let off_diag_max = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .map(|(i, j)| d2_host[i * k + j].max(0.0).sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            (off_diag_max - geometry.dist_max).abs() < 1e-6,
            "dist_max {} must be the metric's own max pairwise distance {off_diag_max}",
            geometry.dist_max
        );
    }

    #[test]
    fn test_no_translator_is_bit_identical_to_empty_translators() {
        // None and Some([]) are both the raw lens; the existing
        // determinism/exactness tests pin None.
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "fn f() {}");
        let raw = read_student_thinking(&student, tokens.clone(), &[1, 2]).unwrap();
        let empty = VecReader {
            translators: Some(Vec::new()),
            ..VecReader::default()
        }
        .read_student_thinking(&student, tokens, &[1, 2])
        .unwrap();
        assert_eq!(raw.depths, empty.depths);
        assert!(raw.translators_used.is_empty());
        assert!(empty.translators_used.is_empty());
    }

    #[test]
    fn test_applied_translator_changes_the_decode_and_is_recorded() {
        let device = Default::default();
        let student = tiny_student(&device);
        let tokens = prompt_tokens(&device, "translate me");
        // A translator that cannot be a no-op: 1.5x scale plus a fixed bias
        // pattern.
        let h = 64;
        let mut matrix = vec![0.0f32; h * h];
        for i in 0..h {
            matrix[i * h + i] = 1.5;
        }
        let bias: Vec<f32> = (0..h)
            .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        let translator = AffineTranslator {
            depth: 2,
            matrix,
            bias,
            initial_kl: 1.0,
            final_kl: 0.5,
        };
        let reader = VecReader {
            translators: Some(vec![translator]),
            ..VecReader::default()
        };
        let raw = read_student_thinking(&student, tokens.clone(), &[2]).unwrap();
        let tuned = reader
            .read_student_thinking(&student, tokens, &[2])
            .unwrap();
        assert_eq!(tuned.translators_used.len(), 1);
        assert_eq!(tuned.translators_used[0].depth, 2);
        assert_eq!(tuned.translators_used[0].kl_before, 1.0);
        assert_eq!(tuned.translators_used[0].kl_after, 0.5);
        // The decode moved (the translated state's distribution differs).
        let raw_top: Vec<u16> = raw.depths[0].decodes[0].top.iter().map(|c| c.id).collect();
        let tuned_top: Vec<u16> = tuned.depths[0].decodes[0]
            .top
            .iter()
            .map(|c| c.id)
            .collect();
        assert_ne!(raw_top, tuned_top);
        // The rendered trace reports the correction.
        let rendered = tuned.render();
        assert!(
            rendered.contains("tuned-lens drift correction"),
            "{rendered}"
        );
        assert!(rendered.contains("KL 1.0000 -> 0.5000"), "{rendered}");
    }

    #[test]
    fn test_train_translators_reduces_decode_kl_deterministically() {
        // On a student whose stream writes, the intermediate-depth lens has
        // real drift (initial KL > 0) and fitting reduces it. Two fits agree
        // bit for bit: identity init, fixed prompt order, and a student whose
        // weights come from a host-seeded redraw (see `deterministic_redraw`),
        // so the result does not depend on burn's process-global RNG — which
        // the parallel gate's neighbour tests reseed constantly — at all.
        let device = Default::default();
        let student = writing_student(&device);
        let prompts = synthetic_prompts(3, 12, 5);
        assert_eq!(prompts, synthetic_prompts(3, 12, 5), "prompts are seeded");
        let fit = TranslatorFit {
            steps: 60,
            lr: 0.05,
        };
        let first = train_translators(&student, &prompts, &[1], &fit).unwrap();
        let second = train_translators(&student, &prompts, &[1], &fit).unwrap();
        assert_eq!(first, second, "the fit is deterministic");
        let translator = &first[0];
        assert_eq!(translator.depth, 1);
        assert_eq!(translator.matrix.len(), 64 * 64);
        assert_eq!(translator.bias.len(), 64);
        println!(
            "translator depth 1: decode KL {:.6} -> {:.6}",
            translator.initial_kl, translator.final_kl
        );
        assert!(
            translator.initial_kl > 1e-6,
            "the writing stream must drift"
        );
        assert!(
            translator.final_kl < translator.initial_kl,
            "fitting must reduce the decode KL: {} -> {}",
            translator.initial_kl,
            translator.final_kl
        );
        // Re-measured through the lens path itself, not trusted from the fit
        // report: KL(final || lens(T(h_1))) against KL(final || lens(h_1)).
        let prompt = &prompts[0];
        let tokens = Tensor::<B, 1, Int>::from_ints(
            prompt
                .iter()
                .map(|t| i64::from(*t))
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([1, prompt.len()]);
        let pass = student.slots_at_depth(tokens.clone(), Some(1)).unwrap();
        let target = log_softmax(student.forward(tokens).unwrap().logits, 2);
        let decode_kl = |hidden: Tensor<B, 3>| -> f32 {
            let logq = log_softmax(student.trunk().logits_from_hidden(hidden), 2);
            let p = target.clone().exp();
            (p * (target.clone() - logq)).sum().into_scalar() / prompt.len() as f32
        };
        let raw_kl = decode_kl(pass.hidden.clone());
        let tuned_kl = decode_kl(apply_translator(&pass.hidden, translator).unwrap());
        println!("lens-path KL at depth 1: raw {raw_kl:.6}, tuned {tuned_kl:.6}");
        assert!(
            tuned_kl < raw_kl,
            "the fitted translator must reduce the lens-path KL: {raw_kl} -> {tuned_kl}"
        );
    }
}
