//! The TRISKELE concept-field generator: NCM's level-wise generator in which
//! vector thinking and geometric reasoning are not extra layers but faces of
//! **one operator**.
//!
//! # The factorization (NCM)
//!
//! `p(r_{1:K}) = Π_k p(r_k | ẑ^(k-1))`. The codes of one level are
//! conditionally independent given the level input, so a whole level is
//! predicted by one call and generation costs `K` calls (NFE = K), coarse to
//! fine (Kumbong et al., arXiv:2610.14162).
//!
//! # The concept field (one operator per level)
//!
//! For level `k` with `ℓ = ℓ_k` codes the state is `U ∈ R^{ℓ×h}`, one concept
//! vector per code position, initialised from the coarser levels:
//!
//! ```text
//! U_0 = W_in · resize_ℓ(ẑ^(k-1)) + level_embedding[k]
//! ```
//!
//! `resize_ℓ` is average pooling over the windows of [`resize_window`]: the
//! `L` positions are split into `ℓ` contiguous, non-empty, disjoint windows
//! `[⌊iL/ℓ⌋, ⌊(i+1)L/ℓ⌋)` and each window's vectors are averaged.
//!
//! The level's attractors are its frozen codebook mapped into the field:
//! `a_j = W_a e_{k,j}` (`W_a` linear, no bias). Distances use one learned
//! metric `G = L Lᵀ + εI` (the parametrisation of [`crate::geometry::MetricField`]:
//! `L` lower-triangular with a positive diagonal, so `G` is positive definite
//! for every parameter value), normalised per dimension:
//! `‖x‖²_G = xᵀ G x / h`. One energy drives the field,
//!
//! ```text
//! E(U) = Σ_i A_G(u_i) + E_int(U; C),   A_G(u) = −τ log Σ_j exp(−‖u − a_j‖²_G / 2τ)
//! ```
//!
//! and one shared update is repeated `T = thinking_steps` times:
//!
//! ```text
//! U_{t+1} = U_t − η [ (U_t − P_t A) + F_θ(U_t, C) ],   P_t = softmax_j(−‖u − a_j‖²_G / 2τ)
//! ```
//!
//! - **Vector thinking** (Soft Thinking, arXiv:2505.15778): `G⁻¹∇A_G(u) =
//!   u − P·A`, so the attractor half of the update pulls every position toward
//!   the probability-weighted mixture of the codebook — the "soft concept",
//!   a convex combination of code vectors. At `η = 1` with `F_θ = 0` the state
//!   *becomes* that mixture.
//! - **Geometric reasoning** (this crate's [`crate::geometry`]): the same `G`
//!   measures the attractor energy and the attention. `F_θ(U, C) = U −
//!   Block_θ([C; U])` where `Block_θ` is `blocks` pre-norm transformer layers
//!   whose attention scores are metric distances, `−‖q̃_i − k̃_j‖²_{G_hh} / √d_h`
//!   (`G_hh` is the principal block of `G` on head `h`'s coordinates, `q̃, k̃`
//!   carry rotary positions).
//! - **NCM**: the output is distance-to-attractor classification,
//!   `logits_j = −‖u_T − a_j‖²_G / 2τ`, trained with the paper's weighted
//!   cross-entropy. The committed code is the basin `u_T` settled into. The
//!   readout uses the same `2τ` as `P_t`, so the model's predictive
//!   distribution and the thought's mixture weights are one distribution, as
//!   in Soft Thinking, where the mixture weights are the model's own output
//!   probabilities. (Since `G` is learned, the constant factor is absorbed by
//!   its scale.)
//!
//! ## The certified step size
//!
//! In coordinates `v = G^{1/2} u` the attractor energy is `A(v) = −τ log Σ_j
//! exp(−‖v − f_j‖² / 2τ)` with Hessian `I − Cov_P(f)/τ ⪯ I`, and the update
//! with `F_θ = 0` is plain gradient descent `v' = v − η∇A(v)`. The descent
//! lemma with smoothness constant 1 gives `A(v') ≤ A(v) − η(1 − η/2)‖∇A‖²`, so
//! **the attractor energy is non-increasing for every `η ∈ (0, 2]`, for every
//! metric `G` and temperature `τ`**. The metric drops out because the update
//! is the natural-gradient step; this is why the bound is the closed form 2
//! rather than a power-iteration estimate as in
//! [`crate::geometry`]'s quadratic relaxation. `step_size` is validated
//! against it, and the `ncm/attractor_energy_descends` certificate checks it,
//! with a control at `η = 3` that does increase the energy. With `F_θ ≠ 0`
//! nothing is claimed: the interaction field is not the gradient of a known
//! energy.
//!
//! ## The prompt (prefix-LM)
//!
//! A prompt of `P` token ids enters as a prefix segment `C` (token embedding
//! plus a learned segment vector) in front of the level's `ℓ` positions.
//! Prompt rows attend only to (non-padding) prompt rows, and to themselves so
//! a padding row is never empty; target rows attend to the non-padding prompt
//! and to every target row. `C` is the same input at every relaxation step and
//! its outputs are discarded, so no target information can reach a prompt
//! position. Prompts are left-padded with `pad_id`; prompt slot `s` sits at
//! rotary position `s − P` and target code `i` at the centre of its window,
//! so the last prompt token is adjacent to the completion. Under a
//! non-identity `G` the rotated score is not a pure function of the offset
//! (the metric mixes the rotated coordinate pairs); positions still enter
//! every score.
//!
//! ## Crystal growth (adaptive per-position depth), off by default
//!
//! With [`AdaptiveThinking`] a position *crystallises* once its predictive
//! confidence `max_j P_t(j)` reaches `confidence`: its state is frozen
//! bit-for-bit and stays as context for the others. The criterion is the
//! confidence rather than the displacement because it is scale-free (in
//! `(1/V, 1]` whatever the learned metric's scale) and it is the quantity the
//! committed code is read from; a displacement threshold would have to be
//! retuned whenever `G` rescales. Each batch row has a hard budget of
//! `⌊budget_factor · ℓ · T⌋` position-steps that is never exceeded; budget
//! freed by early crystals goes to the least confident remaining positions,
//! each capped at `max_steps_per_position`. Honest cost: the budget bounds
//! *state updates*. The trunk still evaluates every row of the level each
//! round (frozen rows are keys and values for the others), so FLOPs are
//! `rounds × full level`, and the realised rounds are reported.
//!
//! ## Interpretability (§12 of the plan)
//!
//! Latent steps are less readable than tokens, so every intermediate state is
//! decodable: [`thought_readout`] quantizes `U_t` to its nearest attractors
//! under `G` and decodes them through the frozen codec, giving text the cheat
//! scanner can read. Any extension that adds latent passes must stay behind
//! the crate's cheat and certificate gate.
//!
//! ## Honesty about cost
//!
//! NFE (level calls, the paper's unit) is `K` per sample. Each level call runs
//! the interaction field `T` times (more or fewer under crystal growth), so
//! the inner trunk evaluations are reported separately and are about `K·T`.
//! The paper's 10–31× speed-up is a measurement of a different model on
//! different data; nothing here claims it for this generator.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, ensure, Context, Result};
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig,
    Linear, LinearConfig,
};
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, silu, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Bool, Distribution, Int, Tensor, TensorData};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use super::{ConceptCodec, LevelCodes, LevelModel, LevelSchedule, SoftConceptCodec};
use crate::checkpoint::{self, DatasetIdentity, TrainState};
use crate::corpus::TokenCorpus;
use crate::geometry::MetricField;
use crate::verify::Certificate;

/// The `kind` a generator checkpoint's training state carries.
pub const STATE_KIND: &str = "ncm-generator";

/// The certificate group these checks report under.
const CERT_GROUP: &str = "ncm";

/// The byte tokenizer's `<pad>` id: the default prompt padding.
fn default_pad_id() -> usize {
    usize::from(crate::tokenizer::Special::Pad.id())
}

// ------------------------------------------------------------- config --

/// How a corrupted code's per-level rate is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateDraw {
    /// Every example uses the level's mean rate.
    Fixed,
    /// Each example draws its rate uniformly from `[0, 2·mean]`, so the mean
    /// is the level's mean and some examples see almost clean conditioning.
    Uniform,
}

/// Conditioning corruption: before `ẑ^(k-1)` is reconstructed for training
/// level `k`, a random fraction of the codes of every coarser level `j < k`
/// is replaced, so the generator learns to work from its own imperfect
/// coarser levels. Labels are never touched.
///
/// The mean rate of conditioning level `j` (0-based) falls linearly from
/// `coarse_mean` at `j = 0` to `fine_mean` at `j = K − 2`, the finest level
/// that ever conditions another. A replaced code is drawn from the model's
/// own prediction for that level with probability `self_prob` ("vaccination",
/// scheduled sampling under no-grad) and uniformly from the codebook
/// otherwise. A uniform draw can return the clean code (probability `1/V`),
/// so the realised change rate is slightly below the drawn rate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CorruptionSchedule {
    pub coarse_mean: f64,
    pub fine_mean: f64,
    pub draw: RateDraw,
    pub self_prob: f64,
}

impl Default for CorruptionSchedule {
    fn default() -> Self {
        Self {
            coarse_mean: 0.5,
            fine_mean: 0.1,
            draw: RateDraw::Uniform,
            self_prob: 0.0,
        }
    }
}

impl CorruptionSchedule {
    /// No corruption at all: clean conditioning.
    pub fn off() -> Self {
        Self {
            coarse_mean: 0.0,
            fine_mean: 0.0,
            draw: RateDraw::Fixed,
            self_prob: 0.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let cap = match self.draw {
            RateDraw::Fixed => 1.0,
            RateDraw::Uniform => 0.5,
        };
        for (name, value) in [
            ("coarse_mean", self.coarse_mean),
            ("fine_mean", self.fine_mean),
        ] {
            ensure!(
                value.is_finite() && (0.0..=cap).contains(&value),
                "corruption {name} must be in [0, {cap}] for a {:?} draw, got {value}",
                self.draw
            );
        }
        ensure!(
            self.fine_mean <= self.coarse_mean,
            "corruption must not increase with level: fine_mean {} > coarse_mean {}",
            self.fine_mean,
            self.coarse_mean
        );
        ensure!(
            self.self_prob.is_finite() && (0.0..=1.0).contains(&self.self_prob),
            "corruption self_prob must be in [0, 1], got {}",
            self.self_prob
        );
        Ok(())
    }

    /// Mean replacement rate of conditioning level `level` (0-based) in a
    /// schedule of `levels` levels.
    pub fn mean_rate(&self, level: usize, levels: usize) -> f64 {
        if levels <= 2 {
            return self.coarse_mean;
        }
        let span = (levels - 2) as f64;
        let position = (level as f64 / span).min(1.0);
        self.coarse_mean + (self.fine_mean - self.coarse_mean) * position
    }

    fn draw_rate<R: Rng>(&self, level: usize, levels: usize, rng: &mut R) -> f64 {
        let mean = self.mean_rate(level, levels);
        match self.draw {
            RateDraw::Fixed => mean,
            RateDraw::Uniform => rng.random::<f64>() * 2.0 * mean,
        }
    }
}

/// Crystal growth: per-position adaptive relaxation depth under a hard
/// budget. See the module docs for the criterion and the cost caveat.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveThinking {
    /// A position crystallises once `max_j P(j) ≥ confidence`. A value above
    /// 1 never triggers (every position runs until the budget or cap).
    pub confidence: f64,
    /// Budget per batch row and level, as a fraction of the fixed-depth cost
    /// `ℓ · T`; in `(0, 1]`, so the budget never exceeds the fixed cost.
    pub budget_factor: f64,
    /// Per-position cap on relaxation steps (≥ 1). Above `T`, budget freed by
    /// early crystals lets the hardest positions think longer.
    pub max_steps_per_position: usize,
}

impl AdaptiveThinking {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.confidence.is_finite() && self.confidence > 0.0,
            "adaptive confidence must be positive, got {}",
            self.confidence
        );
        ensure!(
            self.budget_factor.is_finite() && self.budget_factor > 0.0 && self.budget_factor <= 1.0,
            "adaptive budget_factor must be in (0, 1], got {}",
            self.budget_factor
        );
        ensure!(
            self.max_steps_per_position >= 1,
            "adaptive max_steps_per_position must be at least 1"
        );
        Ok(())
    }

    /// The hard position-step budget for a level of `len` codes at fixed
    /// depth `steps`.
    pub fn budget(&self, len: usize, steps: usize) -> usize {
        ((self.budget_factor * (len * steps) as f64).floor() as usize).min(len * steps)
    }
}

/// The generator's architecture and training knobs. Codebook size, latent
/// width and the level schedule are not here: they come from the codec
/// ([`CodecShape`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NcmGeneratorConfig {
    /// Layers in the interaction field `Block_θ` (one shared operator,
    /// applied once per relaxation step).
    pub blocks: usize,
    /// Field width `h`.
    pub hidden: usize,
    pub heads: usize,
    /// MLP width as a multiple of `hidden`.
    pub mlp_ratio: usize,
    pub rope_base: f64,
    pub dropout: f64,
    /// Level-weight exponent of the loss, `w_k = ℓ_k^{-α} / Σ_j ℓ_j^{1-α}`.
    pub alpha: f64,
    /// Longest prompt prefix, in tokens; 0 trains an unconditional generator.
    pub prompt_len: usize,
    /// Prompt padding id (masked out as a key).
    #[serde(default = "default_pad_id")]
    pub pad_id: usize,
    /// Relaxation steps `T` per level call (at least 1).
    pub thinking_steps: usize,
    /// `η`, in `(0, 2]` (the certified bound; see the module docs).
    pub step_size: f64,
    /// Whether the attractor (Soft Thinking) term is in the update.
    pub attractor: bool,
    /// `τ`.
    pub attractor_temperature: f64,
    /// `ε` in `G = L Lᵀ + εI`.
    pub metric_epsilon: f64,
    /// Weight of the cross-entropy on intermediate states `U_1..U_{T-1}`
    /// (deep supervision, the EBRM anti-drift lesson). Off (0) by default.
    pub deep_supervision: f64,
    pub corruption: CorruptionSchedule,
    /// Crystal growth; `None` runs every position for exactly `T` steps.
    pub adaptive: Option<AdaptiveThinking>,
    pub layer_norm_eps: f64,
}

impl Default for NcmGeneratorConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

impl NcmGeneratorConfig {
    /// For CPU smoke runs and the certificates.
    pub fn tiny() -> Self {
        Self {
            blocks: 2,
            hidden: 64,
            heads: 4,
            mlp_ratio: 4,
            rope_base: 10_000.0,
            dropout: 0.0,
            alpha: 0.0,
            prompt_len: 32,
            pad_id: default_pad_id(),
            thinking_steps: 2,
            step_size: 1.0,
            attractor: true,
            attractor_temperature: 1.0,
            metric_epsilon: 1e-3,
            deep_supervision: 0.0,
            corruption: CorruptionSchedule::default(),
            adaptive: None,
            layer_norm_eps: 1e-5,
        }
    }

    /// The paper's Sudoku-sized generator: 8 blocks, width 512, 8 heads.
    pub fn sudoku() -> Self {
        Self {
            blocks: 8,
            hidden: 512,
            heads: 8,
            prompt_len: 64,
            rope_base: 10_000.0,
            ..Self::tiny()
        }
    }

    /// The paper's TinyStories/OWT-sized generator: 12 blocks, width 768,
    /// 12 heads.
    pub fn paper() -> Self {
        Self {
            blocks: 12,
            hidden: 768,
            heads: 12,
            prompt_len: 128,
            rope_base: 100_000.0,
            ..Self::tiny()
        }
    }

    pub fn preset(name: &str) -> Result<Self> {
        match name {
            "tiny" => Ok(Self::tiny()),
            "sudoku" => Ok(Self::sudoku()),
            "paper" => Ok(Self::paper()),
            other => bail!("unknown generator preset {other:?} (tiny, sudoku, paper)"),
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads.max(1)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.blocks >= 1, "blocks must be at least 1");
        ensure!(self.hidden >= 2, "hidden must be at least 2");
        ensure!(
            self.heads >= 1 && self.hidden % self.heads == 0,
            "hidden {} must be a multiple of heads {}",
            self.hidden,
            self.heads
        );
        ensure!(
            self.head_dim() % 2 == 0,
            "rotary positions need an even head dimension, got {}",
            self.head_dim()
        );
        ensure!(self.mlp_ratio >= 1, "mlp_ratio must be at least 1");
        ensure!(
            self.rope_base.is_finite() && self.rope_base > 1.0,
            "rope_base must be finite and above 1, got {}",
            self.rope_base
        );
        ensure!(
            self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),
            "dropout must be in [0, 1), got {}",
            self.dropout
        );
        ensure!(
            self.alpha.is_finite() && (0.0..=1.0).contains(&self.alpha),
            "alpha must be in [0, 1], got {}",
            self.alpha
        );
        ensure!(
            self.thinking_steps >= 1,
            "thinking_steps must be at least 1"
        );
        ensure!(
            self.step_size.is_finite() && self.step_size > 0.0 && self.step_size <= 2.0,
            "step_size must be in (0, 2] (the certified descent bound), got {}",
            self.step_size
        );
        ensure!(
            self.attractor_temperature.is_finite() && self.attractor_temperature > 0.0,
            "attractor_temperature must be positive, got {}",
            self.attractor_temperature
        );
        ensure!(
            self.metric_epsilon.is_finite() && self.metric_epsilon >= 0.0,
            "metric_epsilon must be non-negative, got {}",
            self.metric_epsilon
        );
        ensure!(
            self.deep_supervision.is_finite() && self.deep_supervision >= 0.0,
            "deep_supervision must be non-negative, got {}",
            self.deep_supervision
        );
        ensure!(
            self.layer_norm_eps.is_finite() && self.layer_norm_eps > 0.0,
            "layer_norm_eps must be positive"
        );
        self.corruption.validate()?;
        if let Some(adaptive) = &self.adaptive {
            adaptive.validate()?;
        }
        Ok(())
    }

    /// The default relaxation for this configuration (no recording).
    pub fn relax(&self) -> RelaxOptions {
        RelaxOptions {
            steps: self.thinking_steps,
            step_size: self.step_size,
            attractor: self.attractor,
            adaptive: self.adaptive,
            record_states: false,
            record_prompt: false,
            record_step_logits: false,
        }
    }
}

/// What the generator takes from the codec, recorded with every checkpoint so
/// a generator is never paired with a codec of a different shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodecShape {
    pub lengths: Vec<usize>,
    pub codebook_size: usize,
    pub latent_dim: usize,
    /// Width of the codebook rows (`level_codebook(k).dims()[1]`).
    pub code_dim: usize,
    pub vocab_size: usize,
}

impl CodecShape {
    /// Read the shape off a codec, checking every level's codebook agrees.
    pub fn of<B: Backend, C: SoftConceptCodec<B>>(codec: &C) -> Result<Self> {
        let schedule = codec.schedule();
        schedule.validate()?;
        let mut code_dim = None;
        for level in 0..schedule.levels() {
            let [rows, width] = codec.level_codebook(level)?.dims();
            ensure!(
                rows == codec.codebook_size(),
                "level {level} codebook has {rows} rows, the codec declares {}",
                codec.codebook_size()
            );
            match code_dim {
                None => code_dim = Some(width),
                Some(seen) => ensure!(
                    seen == width,
                    "level {level} codebook is {width} wide, level 0's is {seen}"
                ),
            }
        }
        Ok(Self {
            lengths: schedule.lengths().to_vec(),
            codebook_size: codec.codebook_size(),
            latent_dim: codec.latent_dim(),
            code_dim: code_dim.ok_or_else(|| anyhow!("the codec has no levels"))?,
            vocab_size: codec.vocab_size(),
        })
    }

    pub fn schedule(&self) -> Result<LevelSchedule> {
        LevelSchedule::new(self.lengths.clone())
    }

    pub fn levels(&self) -> usize {
        self.lengths.len()
    }

    pub fn seq_len(&self) -> usize {
        self.lengths.last().copied().unwrap_or(0)
    }
}

/// Everything needed to rebuild a generator before loading its weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratorSpec {
    pub config: NcmGeneratorConfig,
    pub codec: CodecShape,
}

/// Per-call relaxation knobs (the on-device depth flags).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelaxOptions {
    /// `T`, the fixed depth (and the depth the adaptive budget is scaled by).
    pub steps: usize,
    pub step_size: f64,
    pub attractor: bool,
    pub adaptive: Option<AdaptiveThinking>,
    /// Keep `U_0..U_rounds`.
    pub record_states: bool,
    /// Keep the prompt rows of the field's output at every trunk evaluation.
    pub record_prompt: bool,
    /// Keep the logits of every pre-update state `U_0..U_{rounds-1}`.
    pub record_step_logits: bool,
}

impl RelaxOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.steps >= 1, "relaxation needs at least one step");
        ensure!(
            self.step_size.is_finite() && self.step_size > 0.0 && self.step_size <= 2.0,
            "step_size must be in (0, 2], got {}",
            self.step_size
        );
        if let Some(adaptive) = &self.adaptive {
            adaptive.validate()?;
        }
        Ok(())
    }

    /// Upper bound on trunk evaluations per level call.
    pub fn max_rounds(&self) -> usize {
        match &self.adaptive {
            Some(adaptive) => adaptive.max_steps_per_position,
            None => self.steps,
        }
    }
}

// ------------------------------------------------------------ helpers --

fn host_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| anyhow!("read float tensor: {err:?}"))
}

fn host_i64<B: Backend, const D: usize>(tensor: Tensor<B, D, Int>) -> Result<Vec<i64>> {
    tensor
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .map_err(|err| anyhow!("read int tensor: {err:?}"))
}

fn int_tensor<B: Backend>(
    values: &[i64],
    rows: usize,
    cols: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::<B, 1, Int>::from_ints(values, device).reshape([rows, cols])
}

fn float_tensor<B: Backend, const D: usize>(
    values: &[f32],
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, 1>::from_floats(values, device).reshape(shape)
}

/// Window `i` of `to` windows over `from` positions:
/// `[⌊i·from/to⌋, max(start + 1, ⌊(i+1)·from/to⌋))`. For `to ≤ from` the
/// windows are non-empty, disjoint and cover `0..from`; for `to > from` each
/// window is the single position at or below its proportional place.
pub fn resize_window(i: usize, from: usize, to: usize) -> (usize, usize) {
    let to = to.max(1);
    let start = (i * from / to).min(from.saturating_sub(1));
    let end = ((i + 1) * from / to).max(start + 1).min(from.max(1));
    (start, end)
}

/// The averaging matrix `[to, from]` of [`resize_window`].
fn pool_matrix(from: usize, to: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; to * from];
    for i in 0..to {
        let (start, end) = resize_window(i, from, to);
        let weight = 1.0 / (end - start) as f32;
        for j in start..end {
            m[i * from + j] = weight;
        }
    }
    m
}

/// Average-pool `x [b, from, d]` to `[b, to, d]` over the windows of
/// [`resize_window`]. The identity when `to == from`.
pub fn resize_positions<B: Backend>(x: Tensor<B, 3>, to: usize) -> Result<Tensor<B, 3>> {
    let [batch, from, _] = x.dims();
    ensure!(
        to >= 1 && from >= 1,
        "cannot resize {from} positions to {to}"
    );
    if to == from {
        return Ok(x);
    }
    let device = x.device();
    let pool = float_tensor::<B, 3>(&pool_matrix(from, to), [1, to, from], &device);
    Ok(pool.expand([batch, to, from]).matmul(x))
}

/// The centre of window `i` (in `0..from` coordinates).
fn window_centre(i: usize, from: usize, to: usize) -> f64 {
    let (start, end) = resize_window(i, from, to);
    (start + end - 1) as f64 / 2.0
}

/// Rotary tables `(cos, sin)`, `[1, 1, n, dim]`, at arbitrary positions, in
/// the rotate-half layout of [`crate::hybrid::apply_rotary`].
fn rope_tables<B: Backend>(
    positions: &[f64],
    dim: usize,
    base: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = dim / 2;
    let n = positions.len();
    let mut cos = Vec::with_capacity(n * dim);
    let mut sin = Vec::with_capacity(n * dim);
    for &position in positions {
        let mut row_cos = Vec::with_capacity(half);
        let mut row_sin = Vec::with_capacity(half);
        for i in 0..half {
            let angle = position * base.powf(-(2.0 * i as f64) / dim as f64);
            row_cos.push(angle.cos() as f32);
            row_sin.push(angle.sin() as f32);
        }
        cos.extend_from_slice(&row_cos);
        cos.extend_from_slice(&row_cos);
        sin.extend_from_slice(&row_sin);
        sin.extend_from_slice(&row_sin);
    }
    (
        float_tensor::<B, 4>(&cos, [1, 1, n, dim], device),
        float_tensor::<B, 4>(&sin, [1, 1, n, dim], device),
    )
}

fn apply_rope<B: Backend>(x: Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, _, dim] = x.dims();
    let half = dim / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let rotated = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos.clone() + rotated * sin.clone()
}

/// `‖u_i − a_j‖²_G = (u−a)ᵀ G (u−a) / h` for `u [b, n, h]`, `a [V, h]`,
/// `G [h, h]` symmetric: `[b, n, V]`, by the expansion
/// `uᵀGu − 2 uᵀGa + aᵀGa` (no `[b, n, V, h]` difference tensor).
pub fn metric_sq_dist<B: Backend>(
    u: &Tensor<B, 3>,
    a: &Tensor<B, 2>,
    g: &Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [batch, n, hidden] = u.dims();
    let [codes, _] = a.dims();
    let ug = u.clone().reshape([batch * n, hidden]).matmul(g.clone());
    let ugu = ug
        .clone()
        .mul(u.clone().reshape([batch * n, hidden]))
        .sum_dim(1); // [bn, 1]
    let ag = a.clone().matmul(g.clone());
    let aga = ag.mul(a.clone()).sum_dim(1).reshape([1, codes]); // [1, V]
    let cross = ug.matmul(a.clone().transpose()); // [bn, V]
    cross
        .mul_scalar(-2.0)
        .add(ugu)
        .add(aga)
        .div_scalar(hidden as f64)
        .reshape([batch, n, codes])
}

/// Distance-to-attractor logits `−‖u − a_j‖²_G / 2τ`.
pub fn distance_logits<B: Backend>(
    u: &Tensor<B, 3>,
    a: &Tensor<B, 2>,
    g: &Tensor<B, 2>,
    tau: f64,
) -> Tensor<B, 3> {
    metric_sq_dist(u, a, g).div_scalar(-2.0 * tau)
}

/// The soft concept `P · A` for weights `p [b, n, V]` over attractors
/// `a [V, h]`: `[b, n, h]`.
pub fn soft_concept<B: Backend>(p: Tensor<B, 3>, a: &Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, _, _] = p.dims();
    let [codes, hidden] = a.dims();
    p.matmul(
        a.clone()
            .reshape([1, codes, hidden])
            .expand([batch, codes, hidden]),
    )
}

/// The attractor energy `A_G(u) = −τ log Σ_j exp(−‖u − a_j‖²_G / 2τ)` per
/// position, `[b, n]`.
pub fn attractor_energy<B: Backend>(
    u: &Tensor<B, 3>,
    a: &Tensor<B, 2>,
    g: &Tensor<B, 2>,
    tau: f64,
) -> Tensor<B, 2> {
    let logits = distance_logits(u, a, g, tau);
    let [batch, n, _] = logits.dims();
    let max = logits.clone().max_dim(2);
    let lse = logits.sub(max.clone()).exp().sum_dim(2).log().add(max);
    lse.mul_scalar(-tau).reshape([batch, n])
}

/// One attractor-only update `u − η (u − P·A)` (the `F_θ = 0` dynamics).
pub fn attractor_step<B: Backend>(
    u: &Tensor<B, 3>,
    a: &Tensor<B, 2>,
    g: &Tensor<B, 2>,
    tau: f64,
    eta: f64,
) -> Tensor<B, 3> {
    let p = softmax(distance_logits(u, a, g, tau), 2);
    let pull = u.clone().sub(soft_concept(p, a));
    u.clone().sub(pull.mul_scalar(eta))
}

/// `max_j softmax(logits)_j` per position, on the host.
fn max_probability<B: Backend>(logits: &Tensor<B, 3>) -> Result<Vec<f32>> {
    let max = logits.clone().max_dim(2);
    host_f32(logits.clone().sub(max).exp().sum_dim(2).recip())
}

/// Argmax over the last axis, on the host (ties to the lowest index).
fn argmax_rows(values: &[f32], classes: usize) -> Vec<i64> {
    values
        .chunks(classes.max(1))
        .map(|row| {
            let mut best = 0usize;
            for (j, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = j;
                }
            }
            best as i64
        })
        .collect()
}

// -------------------------------------------------------------- layer --

/// One interaction-field layer: pre-norm metric attention, pre-norm SiLU MLP.
#[derive(Module, Debug)]
pub struct FieldLayer<B: Backend> {
    norm_attn: LayerNorm<B>,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    out: Linear<B>,
    norm_mlp: LayerNorm<B>,
    up: Linear<B>,
    down: Linear<B>,
    dropout: Dropout,
    heads: usize,
}

/// Per-level-call constants shared by every layer and step.
struct Frame<B: Backend> {
    /// Additive attention bias `[b, 1, S, S]`: 0 or −∞. `None` without a
    /// prompt (every target row sees every target row).
    bias: Option<Tensor<B, 4>>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    /// `[1, H, d_h, d_h]`, the principal blocks of `G` per head.
    head_metric: Tensor<B, 4>,
    prompt_len: usize,
}

impl<B: Backend> FieldLayer<B> {
    fn new(config: &NcmGeneratorConfig, device: &B::Device) -> Self {
        let h = config.hidden;
        let norm = |d: usize| {
            LayerNormConfig::new(d)
                .with_epsilon(config.layer_norm_eps)
                .init(device)
        };
        Self {
            norm_attn: norm(h),
            query: LinearConfig::new(h, h).init(device),
            key: LinearConfig::new(h, h).init(device),
            value: LinearConfig::new(h, h).init(device),
            out: LinearConfig::new(h, h).init(device),
            norm_mlp: norm(h),
            up: LinearConfig::new(h, h * config.mlp_ratio).init(device),
            down: LinearConfig::new(h * config.mlp_ratio, h).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
            heads: config.heads,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, frame: &Frame<B>) -> Tensor<B, 3> {
        let [batch, seq, hidden] = x.dims();
        let heads = self.heads;
        let head_dim = hidden / heads;
        let split = |t: Tensor<B, 3>| t.reshape([batch, seq, heads, head_dim]).swap_dims(1, 2);
        let a = self.norm_attn.forward(x.clone());
        let q = apply_rope(split(self.query.forward(a.clone())), &frame.cos, &frame.sin);
        let k = apply_rope(split(self.key.forward(a.clone())), &frame.cos, &frame.sin);
        let v = split(self.value.forward(a));
        let mut scores = metric_attention_scores(q, k, &frame.head_metric);
        if let Some(bias) = &frame.bias {
            scores = scores.add(bias.clone());
        }
        let weights = softmax(scores, 3);
        let context = weights
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq, hidden]);
        let x = x.add(self.dropout.forward(self.out.forward(context)));
        let m = self
            .down
            .forward(silu(self.up.forward(self.norm_mlp.forward(x.clone()))));
        x.add(self.dropout.forward(m))
    }
}

/// `−(q_i − k_j)ᵀ G_hh (q_i − k_j) / √d_h` for `q, k [b, H, S, d_h]` and
/// per-head metric blocks `[1, H, d_h, d_h]`.
fn metric_attention_scores<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    head_metric: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, heads, _, head_dim] = q.dims();
    let g = head_metric
        .clone()
        .expand([batch, heads, head_dim, head_dim]);
    let qg = q.clone().matmul(g.clone());
    let kg = k.clone().matmul(g);
    let qq = qg.clone().mul(q).sum_dim(3); // [b, H, S, 1]
    let kk = kg.mul(k.clone()).sum_dim(3).swap_dims(2, 3); // [b, H, 1, S]
    let cross = qg.matmul(k.swap_dims(2, 3)); // [b, H, S, S]
    cross
        .mul_scalar(-2.0)
        .add(qq)
        .add(kk)
        .mul_scalar(-1.0 / (head_dim as f64).sqrt())
}

// -------------------------------------------------------------- model --

/// The concept-field generator. See the module docs.
#[derive(Module, Debug)]
pub struct NcmGenerator<B: Backend> {
    cond_in: Linear<B>,
    level_embed: Param<Tensor<B, 2>>,
    prompt_embed: Embedding<B>,
    prompt_segment: Param<Tensor<B, 2>>,
    attractor_in: Linear<B>,
    layers: Vec<FieldLayer<B>>,
    metric: MetricField<B>,
    heads: usize,
    hidden: usize,
    levels: usize,
    latent_dim: usize,
    code_dim: usize,
    codebook_size: usize,
    vocab_size: usize,
    pad_id: usize,
    rope_base: f64,
    temperature: f64,
    epsilon: f64,
}

/// What one level call computed.
#[derive(Debug, Clone)]
pub struct LevelForward<B: Backend> {
    /// Code logits of the final state, `[b, ℓ, V]`.
    pub logits: Tensor<B, 3>,
    /// Logits of the pre-update states `U_0..U_{rounds-1}` (when recorded).
    pub step_logits: Vec<Tensor<B, 3>>,
    /// `U_0..U_rounds` (when recorded).
    pub states: Vec<Tensor<B, 3>>,
    /// Which positions were updated in each round, `[b·ℓ]` per round (when
    /// states are recorded).
    pub active: Vec<Vec<bool>>,
    /// Prompt rows of the field output, one per trunk evaluation (when
    /// recorded).
    pub prompt_states: Vec<Tensor<B, 3>>,
    pub usage: ThinkingUsage,
}

/// Realised relaxation cost of one level call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingUsage {
    /// Trunk evaluations (each is the full field over the level).
    pub rounds: usize,
    /// State updates per batch row.
    pub position_steps: Vec<usize>,
    /// The hard budget per batch row.
    pub budget: Vec<usize>,
    /// Positions per batch row that crystallised by confidence.
    pub crystallised: Vec<usize>,
}

/// The prompt as the field sees it.
struct PromptContext<B: Backend> {
    embedded: Tensor<B, 3>,
    /// `[b·P]`, true for a real (non-padding) token.
    valid: Vec<bool>,
}

impl<B: Backend<FloatElem = f32>> NcmGenerator<B> {
    pub fn new(
        config: &NcmGeneratorConfig,
        shape: &CodecShape,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(shape.levels() >= 1, "the codec has no levels");
        ensure!(
            shape.codebook_size >= 2,
            "a codebook needs at least two codes"
        );
        ensure!(
            shape.latent_dim >= 1 && shape.code_dim >= 1,
            "latent and code widths must be positive"
        );
        ensure!(
            config.pad_id < shape.vocab_size,
            "pad_id {} is outside the codec's vocabulary of {}",
            config.pad_id,
            shape.vocab_size
        );
        let h = config.hidden;
        let small = Distribution::Normal(0.0, 0.02);
        let model = Self {
            cond_in: LinearConfig::new(shape.latent_dim, h).init(device),
            level_embed: Param::from_tensor(Tensor::random([shape.levels(), h], small, device)),
            prompt_embed: EmbeddingConfig::new(shape.vocab_size, h)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            prompt_segment: Param::from_tensor(Tensor::random([1, h], small, device)),
            attractor_in: LinearConfig::new(shape.code_dim, h)
                .with_bias(false)
                .init(device),
            layers: (0..config.blocks)
                .map(|_| FieldLayer::new(config, device))
                .collect(),
            metric: MetricField::new(h, device),
            heads: config.heads,
            hidden: h,
            levels: shape.levels(),
            latent_dim: shape.latent_dim,
            code_dim: shape.code_dim,
            codebook_size: shape.codebook_size,
            vocab_size: shape.vocab_size,
            pad_id: config.pad_id,
            rope_base: config.rope_base,
            temperature: config.attractor_temperature,
            epsilon: config.metric_epsilon,
        };
        // Burn initialises lazily, and cloning an unforced parameter draws a
        // different value under the same id: `valid()` and checkpoint clones
        // would then disagree with the live model.
        crate::tensor_ext::force_initialization(&model);
        Ok(model)
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn levels(&self) -> usize {
        self.levels
    }

    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Number of parameters.
    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Refuse a codec whose shape is not the one this generator was built for.
    pub fn check_codec<C: SoftConceptCodec<B>>(&self, codec: &C) -> Result<()> {
        ensure!(
            codec.schedule().levels() == self.levels
                && codec.codebook_size() == self.codebook_size
                && codec.latent_dim() == self.latent_dim
                && codec.vocab_size() == self.vocab_size,
            "codec (levels {}, codebook {}, latent {}, vocab {}) does not match the generator (levels {}, codebook {}, latent {}, vocab {})",
            codec.schedule().levels(),
            codec.codebook_size(),
            codec.latent_dim(),
            codec.vocab_size(),
            self.levels,
            self.codebook_size,
            self.latent_dim,
            self.vocab_size
        );
        Ok(())
    }

    /// The metric `G = L Lᵀ + εI`, `[h, h]`.
    pub fn full_metric(&self) -> Tensor<B, 2> {
        let device = self.level_embed.val().device();
        let g = self.metric.metric();
        if self.epsilon > 0.0 {
            g.add(Tensor::<B, 2>::eye(self.hidden, &device).mul_scalar(self.epsilon))
        } else {
            g
        }
    }

    /// Level `level`'s attractors `W_a e_{k,j}`, `[V, h]`. The codebook is
    /// detached: no gradient reaches the frozen codec.
    pub fn attractors<C: SoftConceptCodec<B>>(
        &self,
        codec: &C,
        level: usize,
    ) -> Result<Tensor<B, 2>> {
        let codebook = codec.level_codebook(level)?;
        let [rows, width] = codebook.dims();
        ensure!(
            rows == self.codebook_size && width == self.code_dim,
            "level {level} codebook is [{rows}, {width}], the generator expects [{}, {}]",
            self.codebook_size,
            self.code_dim
        );
        Ok(self.attractor_in.forward(codebook.detach()))
    }

    /// `U_0 = W_in · resize_ℓ(ẑ) + level_embedding[level]`.
    pub fn initial_state(
        &self,
        level: usize,
        len: usize,
        zhat_coarser: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        ensure!(
            level < self.levels,
            "level {level} is outside the {} levels",
            self.levels
        );
        let pooled = resize_positions(zhat_coarser, len)?;
        let embed = self
            .level_embed
            .val()
            .narrow(0, level, 1)
            .reshape([1, 1, self.hidden]);
        Ok(self.cond_in.forward(pooled).add(embed))
    }

    fn prompt_context(&self, prompt: &Tensor<B, 2, Int>, batch: usize) -> Result<PromptContext<B>> {
        let [rows, _] = prompt.dims();
        ensure!(
            rows == batch,
            "prompt has {rows} rows, the level input has {batch}"
        );
        let ids = host_i64(prompt.clone())?;
        let mut valid = Vec::with_capacity(ids.len());
        for &id in &ids {
            ensure!(
                id >= 0 && (id as usize) < self.vocab_size,
                "prompt token {id} is outside the vocabulary of {}",
                self.vocab_size
            );
            valid.push(id as usize != self.pad_id);
        }
        let embedded = self
            .prompt_embed
            .forward(prompt.clone())
            .add(self.prompt_segment.val().reshape([1, 1, self.hidden]));
        Ok(PromptContext { embedded, valid })
    }

    fn frame(
        &self,
        batch: usize,
        len: usize,
        seq_len: usize,
        prompt_valid: Option<(&[bool], usize)>,
        metric: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Frame<B> {
        let prompt_len = prompt_valid.map_or(0, |(_, p)| p);
        let total = prompt_len + len;
        let mut positions = Vec::with_capacity(total);
        for s in 0..prompt_len {
            positions.push(s as f64 - prompt_len as f64);
        }
        for i in 0..len {
            positions.push(window_centre(i, seq_len, len));
        }
        let head_dim = self.hidden / self.heads;
        let (cos, sin) = rope_tables::<B>(&positions, head_dim, self.rope_base, device);
        let blocks: Vec<Tensor<B, 2>> = (0..self.heads)
            .map(|h| {
                metric
                    .clone()
                    .narrow(0, h * head_dim, head_dim)
                    .narrow(1, h * head_dim, head_dim)
            })
            .collect();
        let head_metric =
            Tensor::stack::<3>(blocks, 0).reshape([1, self.heads, head_dim, head_dim]);
        let bias = prompt_valid.map(|(valid, p)| {
            let mut values = vec![0.0f32; batch * total * total];
            for b in 0..batch {
                let base = b * total * total;
                for q in 0..total {
                    for k in 0..total {
                        let blocked = if q < p {
                            k >= p || !(valid[b * p + k] || k == q)
                        } else {
                            k < p && !valid[b * p + k]
                        };
                        if blocked {
                            values[base + q * total + k] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            float_tensor::<B, 4>(&values, [batch, 1, total, total], device)
        });
        Frame {
            bias,
            cos,
            sin,
            head_metric,
            prompt_len,
        }
    }

    /// One level call: code logits for every position of `level`, from the
    /// cumulative reconstruction of the coarser levels and an optional
    /// left-padded prompt `[b, P]`.
    pub fn forward_level<C: SoftConceptCodec<B>>(
        &self,
        codec: &C,
        level: usize,
        zhat_coarser: Tensor<B, 3>,
        prompt: Option<&Tensor<B, 2, Int>>,
        relax: &RelaxOptions,
    ) -> Result<LevelForward<B>> {
        relax.validate()?;
        self.check_codec(codec)?;
        let schedule = codec.schedule();
        let len = schedule
            .len_of(level)
            .ok_or_else(|| anyhow!("level {level} is outside the {} levels", schedule.levels()))?;
        let [batch, seq_len, latent] = zhat_coarser.dims();
        ensure!(
            seq_len == schedule.seq_len() && latent == self.latent_dim,
            "level input is [{batch}, {seq_len}, {latent}], expected [b, {}, {}]",
            schedule.seq_len(),
            self.latent_dim
        );
        let device = zhat_coarser.device();
        let attractors = self.attractors(codec, level)?;
        let metric = self.full_metric();
        let u0 = self.initial_state(level, len, zhat_coarser)?;
        let context = prompt.map(|p| self.prompt_context(p, batch)).transpose()?;
        let prompt_len = prompt.map_or(0, |p| p.dims()[1]);
        let frame = self.frame(
            batch,
            len,
            seq_len,
            context.as_ref().map(|c| (c.valid.as_slice(), prompt_len)),
            &metric,
            &device,
        );
        self.relax(
            u0,
            context.map(|c| c.embedded),
            &frame,
            &attractors,
            &metric,
            relax,
        )
    }

    /// The relaxation loop (see the module docs for the update).
    fn relax(
        &self,
        u0: Tensor<B, 3>,
        context: Option<Tensor<B, 3>>,
        frame: &Frame<B>,
        attractors: &Tensor<B, 2>,
        metric: &Tensor<B, 2>,
        relax: &RelaxOptions,
    ) -> Result<LevelForward<B>> {
        let [batch, len, hidden] = u0.dims();
        let device = u0.device();
        let prompt_len = frame.prompt_len;
        let (budget_per_row, cap) = match &relax.adaptive {
            Some(adaptive) => (
                adaptive.budget(len, relax.steps),
                adaptive.max_steps_per_position,
            ),
            None => (len * relax.steps, relax.steps),
        };
        let mut usage = ThinkingUsage {
            rounds: 0,
            position_steps: vec![0; batch],
            budget: vec![budget_per_row; batch],
            crystallised: vec![0; batch],
        };
        let mut step_logits = Vec::new();
        let mut states = Vec::new();
        let mut active_record = Vec::new();
        let mut prompt_states = Vec::new();
        let mut frozen = vec![false; batch * len];
        let mut steps_taken = vec![0usize; batch * len];
        let mut state = u0;
        if relax.record_states {
            states.push(state.clone());
        }
        for _ in 0..relax.max_rounds() {
            let need_logits =
                relax.attractor || relax.adaptive.is_some() || relax.record_step_logits;
            let logits =
                need_logits.then(|| distance_logits(&state, attractors, metric, self.temperature));
            let active_mask = match (&relax.adaptive, &logits) {
                (Some(adaptive), Some(logits)) => {
                    let confidence = max_probability(logits)?;
                    let mut active = vec![false; batch * len];
                    let mut any = false;
                    for b in 0..batch {
                        let mut eligible = Vec::new();
                        for i in 0..len {
                            let idx = b * len + i;
                            if frozen[idx] {
                                continue;
                            }
                            if f64::from(confidence[idx]) >= adaptive.confidence {
                                frozen[idx] = true;
                                usage.crystallised[b] += 1;
                            } else if steps_taken[idx] >= cap {
                                frozen[idx] = true;
                            } else {
                                eligible.push(idx);
                            }
                        }
                        let allowed = usage.budget[b].saturating_sub(usage.position_steps[b]);
                        eligible.sort_by(|&x, &y| {
                            confidence[x].total_cmp(&confidence[y]).then(x.cmp(&y))
                        });
                        for &idx in eligible.iter().take(allowed) {
                            active[idx] = true;
                            any = true;
                        }
                    }
                    if !any {
                        break;
                    }
                    Some(active)
                }
                (Some(_), None) => bail!("adaptive relaxation needs the state logits"),
                (None, _) => None,
            };
            if relax.record_step_logits {
                if let Some(logits) = &logits {
                    step_logits.push(logits.clone());
                }
            }
            let sequence = match &context {
                Some(c) => Tensor::cat(vec![c.clone(), state.clone()], 1),
                None => state.clone(),
            };
            let mut x = sequence;
            for layer in &self.layers {
                x = layer.forward(x, frame);
            }
            if relax.record_prompt && prompt_len > 0 {
                prompt_states.push(x.clone().narrow(1, 0, prompt_len));
            }
            let block = if prompt_len > 0 {
                x.narrow(1, prompt_len, len)
            } else {
                x
            };
            let delta = block.sub(state.clone());
            // `update = (U − P·A) + F_θ` with `F_θ = U − Block = −Δ`.
            let mut update = delta.neg();
            if relax.attractor {
                let logits =
                    logits.ok_or_else(|| anyhow!("attractor term needs the state logits"))?;
                let soft = soft_concept(softmax(logits, 2), attractors);
                update = update.add(state.clone().sub(soft));
            }
            let candidate = state.clone().sub(update.mul_scalar(relax.step_size));
            state = match &active_mask {
                Some(active) => {
                    let mask = Tensor::<B, 1, Bool>::from_data(
                        TensorData::new(active.clone(), [batch * len]),
                        &device,
                    )
                    .reshape([batch, len, 1])
                    .expand([batch, len, hidden]);
                    for (idx, &on) in active.iter().enumerate() {
                        if on {
                            steps_taken[idx] += 1;
                            usage.position_steps[idx / len] += 1;
                        }
                    }
                    state.mask_where(mask, candidate)
                }
                None => {
                    for steps in usage.position_steps.iter_mut() {
                        *steps += len;
                    }
                    candidate
                }
            };
            usage.rounds += 1;
            if relax.record_states {
                states.push(state.clone());
                active_record.push(active_mask.unwrap_or_else(|| vec![true; batch * len]));
            }
        }
        let logits = distance_logits(&state, attractors, metric, self.temperature);
        Ok(LevelForward {
            logits,
            step_logits,
            states,
            active: active_record,
            prompt_states,
            usage,
        })
    }
}

// -------------------------------------------------------- the codecs --

/// A deterministic residual-VQ codec with *fixed random* codebooks: token
/// embedding table, average pooling down, nearest-code quantization of the
/// residual, upsampling by window repetition back. It is not trained and
/// reconstructs poorly; it exists so the generator, its certificates and its
/// smoke runs have a codec that satisfies the [`SoftConceptCodec`] contract
/// exactly without depending on the trained tokenizer.
#[derive(Debug, Clone)]
pub struct ReferenceCodec<B: Backend> {
    schedule: LevelSchedule,
    codebooks: Vec<Tensor<B, 2>>,
    token_table: Tensor<B, 2>,
    vocab: usize,
    latent: usize,
    codes: usize,
}

impl<B: Backend> ReferenceCodec<B> {
    /// Codebook rows of level `k` are `N(0, (1/(k+1))²)` per coordinate (the
    /// residual shrinks with depth), token embeddings `N(0, 1)`; all drawn
    /// from a host RNG seeded by `seed`, so the codec is the same on every
    /// backend and in every process.
    pub fn new(
        schedule: LevelSchedule,
        codebook_size: usize,
        latent_dim: usize,
        vocab: usize,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self> {
        schedule.validate()?;
        ensure!(
            codebook_size >= 2 && latent_dim >= 1 && vocab >= 1,
            "degenerate reference codec shape"
        );
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let mut normal = |n: usize, std: f64| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    // Box-Muller from two uniforms in (0, 1].
                    let u1 = 1.0 - rng.random::<f64>();
                    let u2 = rng.random::<f64>();
                    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos() * std) as f32
                })
                .collect()
        };
        let token_table = float_tensor::<B, 2>(
            &normal(vocab * latent_dim, 1.0),
            [vocab, latent_dim],
            device,
        );
        let codebooks = (0..schedule.levels())
            .map(|k| {
                let std = 1.0 / (k + 1) as f64;
                float_tensor::<B, 2>(
                    &normal(codebook_size * latent_dim, std),
                    [codebook_size, latent_dim],
                    device,
                )
            })
            .collect();
        Ok(Self {
            schedule,
            codebooks,
            token_table,
            vocab,
            latent: latent_dim,
            codes: codebook_size,
        })
    }

    /// Level `level`'s code vectors `[b, ℓ, d]` repeated over their windows
    /// to `[b, L, d]`.
    fn upsample(&self, vectors: Tensor<B, 3>, level: usize) -> Result<Tensor<B, 3>> {
        let [batch, len, _] = vectors.dims();
        let seq = self.schedule.seq_len();
        let expected = self
            .schedule
            .len_of(level)
            .ok_or_else(|| anyhow!("no level {level}"))?;
        ensure!(
            len == expected,
            "level {level} has {expected} codes, got {len}"
        );
        let mut m = vec![0.0f32; seq * len];
        for i in 0..len {
            let (start, end) = resize_window(i, seq, len);
            for t in start..end {
                m[t * len + i] = 1.0;
            }
        }
        let device = vectors.device();
        let up = float_tensor::<B, 3>(&m, [1, seq, len], &device).expand([batch, seq, len]);
        Ok(up.matmul(vectors))
    }

    fn lookup(&self, level: usize, codes: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>> {
        let [batch, len] = codes.dims();
        let book = self
            .codebooks
            .get(level)
            .ok_or_else(|| anyhow!("no level {level}"))?;
        Ok(book
            .clone()
            .select(0, codes.reshape([batch * len]))
            .reshape([batch, len, self.latent]))
    }
}

impl<B: Backend> ConceptCodec<B> for ReferenceCodec<B> {
    fn schedule(&self) -> &LevelSchedule {
        &self.schedule
    }

    fn codebook_size(&self) -> usize {
        self.codes
    }

    fn latent_dim(&self) -> usize {
        self.latent
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn encode(&self, tokens: Tensor<B, 2, Int>) -> Result<LevelCodes<B>> {
        let [batch, seq] = tokens.dims();
        ensure!(
            seq == self.schedule.seq_len(),
            "encode expects length {}, got {seq}",
            self.schedule.seq_len()
        );
        let ids = host_i64(tokens.clone())?;
        ensure!(
            ids.iter().all(|&t| t >= 0 && (t as usize) < self.vocab),
            "token outside the vocabulary of {}",
            self.vocab
        );
        let mut residual = self
            .token_table
            .clone()
            .select(0, tokens.reshape([batch * seq]))
            .reshape([batch, seq, self.latent]);
        let mut codes = Vec::with_capacity(self.schedule.levels());
        for (level, &len) in self.schedule.lengths().iter().enumerate() {
            let pooled = resize_positions(residual.clone(), len)?;
            let book = self
                .codebooks
                .get(level)
                .ok_or_else(|| anyhow!("no level {level}"))?;
            let ident = Tensor::<B, 2>::eye(self.latent, &pooled.device());
            let dist = metric_sq_dist(&pooled, book, &ident);
            let chosen = dist.argmin(2).reshape([batch, len]);
            let contribution = self.upsample(self.lookup(level, chosen.clone())?, level)?;
            residual = residual.sub(contribution);
            codes.push(chosen);
        }
        Ok(LevelCodes { codes })
    }

    fn reconstruct(&self, codes: &[Tensor<B, 2, Int>], batch: usize) -> Result<Tensor<B, 3>> {
        ensure!(
            codes.len() <= self.schedule.levels(),
            "{} levels of codes for a {}-level codec",
            codes.len(),
            self.schedule.levels()
        );
        let device = self.token_table.device();
        let mut zhat =
            Tensor::<B, 3>::zeros([batch, self.schedule.seq_len(), self.latent], &device);
        for (level, c) in codes.iter().enumerate() {
            ensure!(
                c.dims()[0] == batch,
                "level {level} codes have {} rows, expected {batch}",
                c.dims()[0]
            );
            zhat = zhat.add(self.upsample(self.lookup(level, c.clone())?, level)?);
        }
        Ok(zhat)
    }

    fn decode_logits(&self, zhat: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, seq, latent] = zhat.dims();
        ensure!(
            latent == self.latent,
            "decode expects width {}, got {latent}",
            self.latent
        );
        Ok(zhat
            .reshape([batch * seq, latent])
            .matmul(self.token_table.clone().transpose())
            .reshape([batch, seq, self.vocab]))
    }
}

impl<B: Backend> SoftConceptCodec<B> for ReferenceCodec<B> {
    fn level_codebook(&self, level: usize) -> Result<Tensor<B, 2>> {
        self.codebooks
            .get(level)
            .cloned()
            .ok_or_else(|| anyhow!("no level {level} in a {}-level codec", self.codebooks.len()))
    }
}

/// A codec on an autodiff backend seen from its inner backend: every tensor
/// crosses with `inner`/`from_inner`, so evaluation and no-grad sampling run
/// without building a graph.
pub struct InnerCodec<'a, B: AutodiffBackend, C> {
    codec: &'a C,
    _backend: PhantomData<B>,
}

impl<'a, B: AutodiffBackend, C: SoftConceptCodec<B>> InnerCodec<'a, B, C> {
    pub fn new(codec: &'a C) -> Self {
        Self {
            codec,
            _backend: PhantomData,
        }
    }
}

impl<B: AutodiffBackend, C: SoftConceptCodec<B>> ConceptCodec<B::InnerBackend>
    for InnerCodec<'_, B, C>
{
    fn schedule(&self) -> &LevelSchedule {
        self.codec.schedule()
    }

    fn codebook_size(&self) -> usize {
        self.codec.codebook_size()
    }

    fn latent_dim(&self) -> usize {
        self.codec.latent_dim()
    }

    fn vocab_size(&self) -> usize {
        self.codec.vocab_size()
    }

    fn encode(
        &self,
        tokens: Tensor<B::InnerBackend, 2, Int>,
    ) -> Result<LevelCodes<B::InnerBackend>> {
        let codes = self.codec.encode(Tensor::from_inner(tokens))?;
        Ok(LevelCodes {
            codes: codes.codes.into_iter().map(Tensor::inner).collect(),
        })
    }

    fn reconstruct(
        &self,
        codes: &[Tensor<B::InnerBackend, 2, Int>],
        batch: usize,
    ) -> Result<Tensor<B::InnerBackend, 3>> {
        let outer: Vec<Tensor<B, 2, Int>> = codes.iter().cloned().map(Tensor::from_inner).collect();
        Ok(self.codec.reconstruct(&outer, batch)?.inner())
    }

    fn decode_logits(
        &self,
        zhat: Tensor<B::InnerBackend, 3>,
    ) -> Result<Tensor<B::InnerBackend, 3>> {
        Ok(self.codec.decode_logits(Tensor::from_inner(zhat))?.inner())
    }
}

impl<B: AutodiffBackend, C: SoftConceptCodec<B>> SoftConceptCodec<B::InnerBackend>
    for InnerCodec<'_, B, C>
{
    fn level_codebook(&self, level: usize) -> Result<Tensor<B::InnerBackend, 2>> {
        Ok(self.codec.level_codebook(level)?.inner())
    }
}

/// A codec wrapper that counts calls (the cost accounting of generation).
pub struct CountingCodec<'a, C> {
    inner: &'a C,
    reconstruct_calls: Cell<usize>,
    decode_calls: Cell<usize>,
}

impl<'a, C> CountingCodec<'a, C> {
    pub fn new(inner: &'a C) -> Self {
        Self {
            inner,
            reconstruct_calls: Cell::new(0),
            decode_calls: Cell::new(0),
        }
    }

    pub fn reconstruct_calls(&self) -> usize {
        self.reconstruct_calls.get()
    }

    pub fn decode_calls(&self) -> usize {
        self.decode_calls.get()
    }
}

impl<B: Backend, C: ConceptCodec<B>> ConceptCodec<B> for CountingCodec<'_, C> {
    fn schedule(&self) -> &LevelSchedule {
        self.inner.schedule()
    }

    fn codebook_size(&self) -> usize {
        self.inner.codebook_size()
    }

    fn latent_dim(&self) -> usize {
        self.inner.latent_dim()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn encode(&self, tokens: Tensor<B, 2, Int>) -> Result<LevelCodes<B>> {
        self.inner.encode(tokens)
    }

    fn reconstruct(&self, codes: &[Tensor<B, 2, Int>], batch: usize) -> Result<Tensor<B, 3>> {
        self.reconstruct_calls.set(self.reconstruct_calls.get() + 1);
        self.inner.reconstruct(codes, batch)
    }

    fn decode_logits(&self, zhat: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        self.decode_calls.set(self.decode_calls.get() + 1);
        self.inner.decode_logits(zhat)
    }
}

impl<B: Backend, C: SoftConceptCodec<B>> SoftConceptCodec<B> for CountingCodec<'_, C> {
    fn level_codebook(&self, level: usize) -> Result<Tensor<B, 2>> {
        self.inner.level_codebook(level)
    }
}

/// A [`LevelModel`] wrapper that counts level calls: NFE, measured rather
/// than assumed.
pub struct CountingLevelModel<'a, M> {
    inner: &'a M,
    calls: Cell<usize>,
}

impl<'a, M> CountingLevelModel<'a, M> {
    pub fn new(inner: &'a M) -> Self {
        Self {
            inner,
            calls: Cell::new(0),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.get()
    }
}

impl<B: Backend, M: LevelModel<B>> LevelModel<B> for CountingLevelModel<'_, M> {
    fn level_logits(
        &self,
        level: usize,
        zhat_coarser: Tensor<B, 3>,
        prompt: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 3>> {
        self.calls.set(self.calls.get() + 1);
        self.inner.level_logits(level, zhat_coarser, prompt)
    }
}

// ------------------------------------------------- the LevelModel impl --

/// What one level call of a [`ConceptFieldLm`] did, for reports and readout.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LevelTrace {
    pub level: usize,
    pub usage: ThinkingUsage,
    /// Nearest-attractor codes of every state `U_0..U_rounds` (`[b·ℓ]` each),
    /// when thoughts are recorded.
    pub state_codes: Vec<Vec<i64>>,
}

/// The generator bound to its frozen codec and relaxation options: the
/// [`LevelModel`] the decoding strategies drive. Levels are evaluated one
/// call at a time ([`LevelModel::batches_levels`] stays `false`: each level
/// has its own attractor set, and no batched multi-level pass is
/// implemented).
pub struct ConceptFieldLm<'a, B: Backend, C> {
    model: &'a NcmGenerator<B>,
    codec: &'a C,
    relax: RelaxOptions,
    traces: RefCell<Vec<LevelTrace>>,
}

impl<'a, B: Backend<FloatElem = f32>, C: SoftConceptCodec<B>> ConceptFieldLm<'a, B, C> {
    pub fn new(model: &'a NcmGenerator<B>, codec: &'a C, relax: RelaxOptions) -> Result<Self> {
        relax.validate()?;
        model.check_codec(codec)?;
        Ok(Self {
            model,
            codec,
            relax,
            traces: RefCell::new(Vec::new()),
        })
    }

    /// The traces recorded so far, emptying the record.
    pub fn take_traces(&self) -> Result<Vec<LevelTrace>> {
        let mut traces = self
            .traces
            .try_borrow_mut()
            .map_err(|err| anyhow!("level traces are borrowed: {err}"))?;
        Ok(std::mem::take(&mut *traces))
    }
}

impl<B: Backend<FloatElem = f32>, C: SoftConceptCodec<B>> LevelModel<B>
    for ConceptFieldLm<'_, B, C>
{
    fn level_logits(
        &self,
        level: usize,
        zhat_coarser: Tensor<B, 3>,
        prompt: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 3>> {
        let forward = self.model.forward_level(
            self.codec,
            level,
            zhat_coarser,
            prompt.as_ref(),
            &self.relax,
        )?;
        let mut state_codes = Vec::new();
        if self.relax.record_step_logits {
            let classes = self.model.codebook_size();
            for logits in forward
                .step_logits
                .iter()
                .chain(std::iter::once(&forward.logits))
            {
                state_codes.push(argmax_rows(&host_f32(logits.clone())?, classes));
            }
        }
        self.traces
            .try_borrow_mut()
            .map_err(|err| anyhow!("level traces are borrowed: {err}"))?
            .push(LevelTrace {
                level,
                usage: forward.usage,
                state_codes,
            });
        Ok(forward.logits)
    }
}

// ------------------------------------------------------------- losses --

/// Mean cross-entropy over every position and batch row, `[1]`.
pub fn level_cross_entropy<B: Backend>(
    logits: Tensor<B, 3>,
    labels: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let picked = log_softmax(logits, 2).gather(2, labels.unsqueeze_dim::<3>(2));
    picked.mean().neg()
}

/// `L_NCM = −Σ_k w_k Σ_i log p(r_k^i | z̃^(k-1))`, averaged over the batch,
/// from one logits tensor per level.
pub fn weighted_ncm_loss<B: Backend>(
    logits: &[Tensor<B, 3>],
    labels: &[Tensor<B, 2, Int>],
    weights: &[f64],
) -> Result<Tensor<B, 1>> {
    ensure!(
        logits.len() == labels.len() && labels.len() == weights.len() && !logits.is_empty(),
        "one logits tensor, label tensor and weight per level"
    );
    let mut total: Option<Tensor<B, 1>> = None;
    for ((l, y), &w) in logits.iter().zip(labels).zip(weights) {
        let [batch, len, _] = l.dims();
        let term = level_cross_entropy(l.clone(), y.clone()).mul_scalar(w * len as f64);
        ensure!(batch >= 1, "empty batch");
        total = Some(match total {
            Some(t) => t.add(term),
            None => term,
        });
    }
    total.ok_or_else(|| anyhow!("no levels"))
}

/// The level-sampling distribution `q_k = w_k ℓ_k` (it sums to 1 for any
/// `α`). Sampling `k ~ q` and taking the *mean* cross-entropy of level `k`
/// is an unbiased estimator of `L_NCM`:
/// `E_q[S_k / ℓ_k] = Σ_k w_k ℓ_k S_k / ℓ_k = Σ_k w_k S_k`, with every sample
/// of size O(1) (a per-code mean), which keeps the variance low.
pub fn level_sampling_distribution(schedule: &LevelSchedule, alpha: f64) -> Result<Vec<f64>> {
    let weights = schedule.loss_weights(alpha)?;
    Ok(weights
        .iter()
        .zip(schedule.lengths())
        .map(|(w, &l)| w * l as f64)
        .collect())
}

/// How training estimates `L_NCM` per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossEstimator {
    /// One level per example, `k ~ q_k = w_k ℓ_k`, mean CE of that level
    /// (unbiased; see [`level_sampling_distribution`]). Costs one level call
    /// per example.
    SampledLevel,
    /// Every level for every example, weighted exactly. Costs `K` level calls
    /// per example.
    AllLevels,
}

// --------------------------------------------------------- corruption --

/// The training example of one level for a group of rows: the (possibly
/// corrupted) conditioning codes of the coarser levels, and the labels.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelExample {
    /// One `[rows·ℓ_j]` vector per conditioning level `j < level`.
    pub conditioning: Vec<Vec<i64>>,
    /// `[rows·ℓ_level]`, always the clean codes.
    pub labels: Vec<i64>,
    pub replaced_uniform: usize,
    pub replaced_self: usize,
}

/// What corruption needs to know about the codes it corrupts.
#[derive(Debug, Clone, Copy)]
pub struct CorruptionContext<'a> {
    pub schedule: &'a LevelSchedule,
    pub corruption: &'a CorruptionSchedule,
    pub codebook_size: usize,
}

/// Build one level's training example from clean codes (`clean[j]` is
/// `[rows·ℓ_j]`). Only `clean[..level]` is ever corrupted; the labels are a
/// copy of `clean[level]`. `self_codes`, when given, holds the model's own
/// sampled codes for each conditioning level (same layout as `clean`).
pub fn build_level_example<R: Rng>(
    clean: &[Vec<i64>],
    level: usize,
    rows: usize,
    context: CorruptionContext<'_>,
    self_codes: Option<&[Vec<i64>]>,
    rng: &mut R,
) -> Result<LevelExample> {
    let CorruptionContext {
        schedule,
        corruption,
        codebook_size,
    } = context;
    ensure!(
        level < schedule.levels() && clean.len() == schedule.levels(),
        "level {level} of a {}-level schedule",
        schedule.levels()
    );
    let labels = clean[level].clone();
    let mut conditioning = Vec::with_capacity(level);
    let mut replaced_uniform = 0;
    let mut replaced_self = 0;
    for (j, clean_j) in clean.iter().enumerate().take(level) {
        let len = schedule.len_of(j).ok_or_else(|| anyhow!("no level {j}"))?;
        let mut codes = clean_j.clone();
        ensure!(
            codes.len() == rows * len,
            "level {j} has {} codes for {rows} rows of {len}",
            codes.len()
        );
        let own = match self_codes {
            Some(own) => {
                let level_codes = own
                    .get(j)
                    .ok_or_else(|| anyhow!("no self codes for level {j}"))?;
                ensure!(
                    level_codes.len() == codes.len(),
                    "self codes of level {j} have the wrong length"
                );
                Some(level_codes)
            }
            None => None,
        };
        for r in 0..rows {
            let rate = corruption.draw_rate(j, schedule.levels(), rng);
            for i in 0..len {
                if rng.random::<f64>() >= rate {
                    continue;
                }
                let idx = r * len + i;
                match own {
                    Some(own) if rng.random::<f64>() < corruption.self_prob => {
                        codes[idx] = own[idx];
                        replaced_self += 1;
                    }
                    _ => {
                        codes[idx] = rng.random_range(0..codebook_size) as i64;
                        replaced_uniform += 1;
                    }
                }
            }
        }
        conditioning.push(codes);
    }
    Ok(LevelExample {
        conditioning,
        labels,
        replaced_uniform,
        replaced_self,
    })
}

// ----------------------------------------------------------- sampling --

/// How codes are drawn from level logits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CodeSampling {
    /// 0 is greedy (argmax).
    pub temperature: f64,
    /// 0 keeps every class.
    pub top_k: usize,
}

impl CodeSampling {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
        }
    }
}

/// Draw one class per row of `logits` (`classes` per row) on the host.
pub fn sample_rows<R: Rng>(
    logits: &[f32],
    classes: usize,
    sampling: &CodeSampling,
    rng: &mut R,
) -> Result<Vec<i64>> {
    ensure!(
        classes >= 1 && logits.len() % classes == 0,
        "logits are not whole rows of {classes}"
    );
    ensure!(
        sampling.temperature.is_finite() && sampling.temperature >= 0.0,
        "temperature must be non-negative, got {}",
        sampling.temperature
    );
    if sampling.temperature == 0.0 {
        return Ok(argmax_rows(logits, classes));
    }
    let mut out = Vec::with_capacity(logits.len() / classes);
    let mut order: Vec<usize> = (0..classes).collect();
    for row in logits.chunks(classes) {
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        let keep = if sampling.top_k == 0 {
            classes
        } else {
            sampling.top_k.min(classes)
        };
        let best = f64::from(row[order[0]]);
        let weights: Vec<f64> = order[..keep]
            .iter()
            .map(|&j| ((f64::from(row[j]) - best) / sampling.temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        ensure!(total.is_finite() && total > 0.0, "non-finite logits row");
        let mut target = rng.random::<f64>() * total;
        let mut chosen = order[keep - 1];
        for (slot, &w) in weights.iter().enumerate() {
            if target < w {
                chosen = order[slot];
                break;
            }
            target -= w;
        }
        out.push(chosen as i64);
    }
    Ok(out)
}

/// The codes of a generation, coarse to fine, plus (when recorded) what each
/// level saw and predicted.
#[derive(Debug, Clone)]
pub struct LevelRun<B: Backend> {
    pub codes: Vec<Tensor<B, 2, Int>>,
    /// The `ẑ^(k-1)` each level call received (when recorded).
    pub inputs: Vec<Tensor<B, 3>>,
    /// The logits each level call returned (when recorded).
    pub logits: Vec<Tensor<B, 3>>,
}

/// NCM decoding for any [`LevelModel`]: for each level, one model call on the
/// reconstruction of the committed coarser codes, then every code of the
/// level drawn in parallel. Exactly one model call per level.
pub fn generate_levels<B: Backend, M: LevelModel<B>, C: ConceptCodec<B>>(
    model: &M,
    codec: &C,
    prompt: Option<Tensor<B, 2, Int>>,
    batch: usize,
    sampling: &CodeSampling,
    record: bool,
    rng: &mut ChaCha12Rng,
) -> Result<LevelRun<B>> {
    let schedule = codec.schedule().clone();
    let classes = codec.codebook_size();
    let mut run = LevelRun {
        codes: Vec::with_capacity(schedule.levels()),
        inputs: Vec::new(),
        logits: Vec::new(),
    };
    for (level, &len) in schedule.lengths().iter().enumerate() {
        let zhat = codec.reconstruct(&run.codes, batch)?;
        let logits = model.level_logits(level, zhat.clone(), prompt.clone())?;
        let dims = logits.dims();
        ensure!(
            dims == [batch, len, classes],
            "level {level} logits are {dims:?}, expected [{batch}, {len}, {classes}]"
        );
        let device = logits.device();
        let drawn = sample_rows(&host_f32(logits.clone())?, classes, sampling, rng)?;
        run.codes.push(int_tensor::<B>(&drawn, batch, len, &device));
        if record {
            run.inputs.push(zhat);
            run.logits.push(logits);
        }
    }
    Ok(run)
}

/// Tokens from a full reconstruction: greedy (`temperature == 0`) or
/// sampled.
pub fn decode_tokens<B: Backend, C: ConceptCodec<B>, R: Rng>(
    codec: &C,
    codes: &[Tensor<B, 2, Int>],
    batch: usize,
    temperature: f64,
    rng: &mut R,
) -> Result<Vec<Vec<u32>>> {
    let zhat = codec.reconstruct(codes, batch)?;
    let logits = codec.decode_logits(zhat)?;
    let [_, seq, vocab] = logits.dims();
    let drawn = sample_rows(
        &host_f32(logits)?,
        vocab,
        &CodeSampling {
            temperature,
            top_k: 0,
        },
        rng,
    )?;
    Ok(drawn
        .chunks(seq.max(1))
        .map(|row| row.iter().map(|&t| t as u32).collect())
        .collect())
}

/// Read an intermediate thought: the coarser committed codes plus the
/// nearest-attractor codes of a relaxation state for `level`, reconstructed
/// and greedily decoded by the frozen codec. The text a cheat scanner reads.
pub fn thought_readout<B: Backend, C: ConceptCodec<B>>(
    codec: &C,
    committed_coarser: &[Vec<i64>],
    thought: &[i64],
    batch: usize,
    device: &B::Device,
) -> Result<Vec<Vec<u32>>> {
    let schedule = codec.schedule();
    let level = committed_coarser.len();
    let mut codes = Vec::with_capacity(level + 1);
    for (j, c) in committed_coarser
        .iter()
        .chain(std::iter::once(&thought.to_vec()))
        .enumerate()
    {
        let len = schedule.len_of(j).ok_or_else(|| anyhow!("no level {j}"))?;
        ensure!(
            c.len() == batch * len,
            "level {j} codes have length {}, expected {}",
            c.len(),
            batch * len
        );
        codes.push(int_tensor::<B>(c, batch, len, device));
    }
    let mut rng = ChaCha12Rng::seed_from_u64(0);
    decode_tokens(codec, &codes, batch, 0.0, &mut rng)
}

/// Generation knobs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GenerateOptions {
    pub codes: CodeSampling,
    /// Final token decode: 0 is greedy.
    pub token_temperature: f64,
    /// Samples drawn for minimum-Bayes-risk selection; 1 disables it.
    pub mbr_samples: usize,
    pub seed: u64,
    pub relax: RelaxOptions,
    /// Record each level's per-step nearest-attractor codes for
    /// [`thought_readout`] (single-sample runs only).
    pub record_thoughts: bool,
}

/// One level's recorded thoughts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelThoughts {
    pub level: usize,
    /// Nearest-attractor codes of `U_0..U_rounds`, `[b·ℓ]` each.
    pub state_codes: Vec<Vec<i64>>,
}

/// The minimum-Bayes-risk choice: per batch row, the sample with the least
/// total normalised Hamming distance to the other samples' decoded tokens
/// (ties to the lowest index).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MbrChoice {
    pub samples: usize,
    pub chosen: Vec<usize>,
    /// Mean distance of the chosen sample to the others, per row.
    pub mean_distance: Vec<f64>,
}

/// What a generation did, measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Generation {
    /// `[batch][L]` token ids.
    pub tokens: Vec<Vec<u32>>,
    /// Committed codes per level, `[b·ℓ_k]`.
    pub codes: Vec<Vec<i64>>,
    /// Level calls per sample, counted (the paper's NFE; `K` by
    /// construction, measured by a counting wrapper).
    pub nfe_per_sample: usize,
    /// Level calls over all samples.
    pub nfe_total: usize,
    pub samples: usize,
    /// Inner trunk evaluations over all samples (about `K·T` per sample).
    pub trunk_evals: usize,
    pub position_steps: usize,
    pub position_budget: usize,
    pub crystallised: usize,
    pub reconstruct_calls: usize,
    pub decode_calls: usize,
    pub thoughts: Vec<LevelThoughts>,
    pub mbr: Option<MbrChoice>,
}

/// Generate `batch` sequences, optionally conditioned on a left-padded prompt
/// `[batch, P]`.
pub fn generate<B: Backend<FloatElem = f32>, C: SoftConceptCodec<B>>(
    model: &NcmGenerator<B>,
    codec: &C,
    prompt: Option<Tensor<B, 2, Int>>,
    batch: usize,
    options: &GenerateOptions,
) -> Result<Generation> {
    ensure!(batch >= 1, "generate needs at least one sequence");
    ensure!(options.mbr_samples >= 1, "mbr_samples must be at least 1");
    let mut relax = options.relax;
    relax.record_step_logits = options.record_thoughts && options.mbr_samples == 1;
    relax.record_states = false;
    relax.record_prompt = false;
    let counting_codec = CountingCodec::new(codec);
    let field = ConceptFieldLm::new(model, &counting_codec, relax)?;
    let counted = CountingLevelModel::new(&field);
    let mut sample_tokens = Vec::with_capacity(options.mbr_samples);
    let mut sample_codes = Vec::with_capacity(options.mbr_samples);
    let mut usage = ThinkingUsage::default();
    let mut thoughts = Vec::new();
    let mut first_sample_calls = 0;
    for s in 0..options.mbr_samples {
        let mut rng = ChaCha12Rng::seed_from_u64(crate::train::step_seed(options.seed, s));
        let run = generate_levels(
            &counted,
            &counting_codec,
            prompt.clone(),
            batch,
            &options.codes,
            false,
            &mut rng,
        )?;
        if s == 0 {
            first_sample_calls = counted.calls();
        }
        let tokens = decode_tokens(
            &counting_codec,
            &run.codes,
            batch,
            options.token_temperature,
            &mut rng,
        )?;
        let codes = run
            .codes
            .into_iter()
            .map(host_i64)
            .collect::<Result<Vec<_>>>()?;
        for trace in field.take_traces()? {
            usage.rounds += trace.usage.rounds;
            usage
                .position_steps
                .push(trace.usage.position_steps.iter().sum());
            usage.budget.push(trace.usage.budget.iter().sum());
            usage
                .crystallised
                .push(trace.usage.crystallised.iter().sum());
            if relax.record_step_logits {
                thoughts.push(LevelThoughts {
                    level: trace.level,
                    state_codes: trace.state_codes,
                });
            }
        }
        sample_tokens.push(tokens);
        sample_codes.push(codes);
    }
    let (chosen, mbr) = if options.mbr_samples > 1 {
        let choice = mbr_select(&sample_tokens, batch)?;
        (choice.chosen.clone(), Some(choice))
    } else {
        (vec![0; batch], None)
    };
    let schedule = codec.schedule();
    let mut tokens = Vec::with_capacity(batch);
    let mut codes: Vec<Vec<i64>> = schedule
        .lengths()
        .iter()
        .map(|&l| Vec::with_capacity(batch * l))
        .collect();
    for (row, &s) in chosen.iter().enumerate() {
        let sample = sample_tokens
            .get(s)
            .ok_or_else(|| anyhow!("no sample {s}"))?;
        tokens.push(
            sample
                .get(row)
                .cloned()
                .ok_or_else(|| anyhow!("no row {row}"))?,
        );
        for (level, &len) in schedule.lengths().iter().enumerate() {
            let level_codes = &sample_codes[s][level];
            codes[level].extend_from_slice(&level_codes[row * len..(row + 1) * len]);
        }
    }
    Ok(Generation {
        tokens,
        codes,
        nfe_per_sample: first_sample_calls,
        nfe_total: counted.calls(),
        samples: options.mbr_samples,
        trunk_evals: usage.rounds,
        position_steps: usage.position_steps.iter().sum(),
        position_budget: usage.budget.iter().sum(),
        crystallised: usage.crystallised.iter().sum(),
        reconstruct_calls: counting_codec.reconstruct_calls(),
        decode_calls: counting_codec.decode_calls(),
        thoughts,
        mbr,
    })
}

/// Normalised Hamming distance between two token sequences (length
/// difference counts as mismatches).
fn hamming(a: &[u32], b: &[u32]) -> f64 {
    let n = a.len().max(b.len()).max(1);
    let same = a.iter().zip(b).filter(|(x, y)| x == y).count();
    (n - same) as f64 / n as f64
}

fn mbr_select(samples: &[Vec<Vec<u32>>], batch: usize) -> Result<MbrChoice> {
    let n = samples.len();
    let mut chosen = Vec::with_capacity(batch);
    let mut mean_distance = Vec::with_capacity(batch);
    for row in 0..batch {
        let candidates: Vec<&[u32]> = samples
            .iter()
            .map(|sample| {
                sample
                    .get(row)
                    .map(Vec::as_slice)
                    .ok_or_else(|| anyhow!("sample has no row {row}"))
            })
            .collect::<Result<_>>()?;
        let mut best = (f64::INFINITY, 0usize);
        for (i, a) in candidates.iter().enumerate() {
            let total: f64 = candidates
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, b)| hamming(a, b))
                .sum();
            if total < best.0 {
                best = (total, i);
            }
        }
        chosen.push(best.1);
        mean_distance.push(best.0 / (n.saturating_sub(1)).max(1) as f64);
    }
    Ok(MbrChoice {
        samples: n,
        chosen,
        mean_distance,
    })
}

/// A prompt `[rows, P]`: the last `P` tokens of `tokens`, left-padded with
/// `pad`, repeated for every row.
pub fn prompt_tensor<B: Backend>(
    tokens: &[u16],
    prompt_len: usize,
    pad: usize,
    rows: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let keep = tokens.len().min(prompt_len);
    let mut row = vec![pad as i64; prompt_len - keep];
    row.extend(tokens[tokens.len() - keep..].iter().map(|&t| i64::from(t)));
    let all: Vec<i64> = (0..rows).flat_map(|_| row.iter().copied()).collect();
    int_tensor::<B>(&all, rows, prompt_len, device)
}

// ----------------------------------------------------------- training --

/// The training run's configuration (the model's is inside it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NcmTrainConfig {
    pub model: NcmGeneratorConfig,
    pub steps: usize,
    pub batch_size: usize,
    pub peak_lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub seed: u64,
    pub estimator: LossEstimator,
    /// Input is prearranged in prompt + target records of a fixed width.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub aligned_windows: bool,
    /// Validate every this many steps (0: only at start and end).
    pub val_every: usize,
    /// Fixed validation windows (evenly spaced over the validation corpus).
    pub val_windows: usize,
    pub checkpoint_every: usize,
    pub out_dir: Option<PathBuf>,
}

impl Default for NcmTrainConfig {
    fn default() -> Self {
        Self {
            model: NcmGeneratorConfig::tiny(),
            steps: 200,
            batch_size: 8,
            peak_lr: 3e-4,
            min_lr: 3e-5,
            warmup_steps: 20,
            weight_decay: 0.01,
            grad_clip: 1.0,
            seed: 0,
            estimator: LossEstimator::SampledLevel,
            aligned_windows: false,
            val_every: 0,
            val_windows: 16,
            checkpoint_every: 0,
            out_dir: None,
        }
    }
}

impl NcmTrainConfig {
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        ensure!(
            self.steps >= 1 && self.batch_size >= 1,
            "steps and batch_size must be positive"
        );
        ensure!(
            self.peak_lr.is_finite()
                && self.peak_lr > 0.0
                && self.min_lr >= 0.0
                && self.min_lr <= self.peak_lr,
            "need 0 <= min_lr <= peak_lr, got {} and {}",
            self.min_lr,
            self.peak_lr
        );
        ensure!(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "weight_decay must be non-negative"
        );
        ensure!(
            self.grad_clip.is_finite() && self.grad_clip >= 0.0,
            "grad_clip must be non-negative"
        );
        Ok(())
    }

    pub fn lr_schedule(&self) -> crate::schedule::LrSchedule {
        crate::schedule::LrSchedule::WarmupCosine {
            peak: self.peak_lr,
            min_lr: self.min_lr,
            warmup_steps: self.warmup_steps,
            total_steps: self.steps,
        }
    }
}

/// The generator's optimizer: AdamW with `β = (0.9, 0.95)`.
pub type GeneratorOptim<B> = burn::optim::adaptor::OptimizerAdaptor<AdamW, NcmGenerator<B>, B>;
/// Its record.
pub type GeneratorOptimRecord<B> = <GeneratorOptim<B> as Optimizer<NcmGenerator<B>, B>>::Record;

fn new_optimizer<B: AutodiffBackend<FloatElem = f32>>(
    config: &NcmTrainConfig,
) -> GeneratorOptim<B> {
    AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_weight_decay(config.weight_decay as f32)
        .init()
}

/// Per-level validation with clean conditioning and (if the model takes one)
/// the full prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelEval {
    /// Mean cross-entropy per code, per level (nats).
    pub cross_entropy: Vec<f64>,
    /// Argmax accuracy per level.
    pub accuracy: Vec<f64>,
    /// `Σ_k w_k ℓ_k CE_k` at `α = 0`: the mean cross-entropy over every code
    /// of every level (`L_NCM` per code), whatever `α` training used, so runs
    /// with different `α` stay comparable.
    pub weighted: f64,
    /// `ln V`: a uniform guess.
    pub uniform_baseline: f64,
    /// Empirical entropy of each level's codes over the validation windows:
    /// what a context-free predictor fitted *to these very codes* would score
    /// (an optimistic baseline).
    pub unigram_entropy: Vec<f64>,
    pub windows: usize,
}

/// The relaxed final states' distance to the codebook (EBRM's drift check):
/// `‖u_T − a_nearest‖_G` and `‖u_T − P·A‖_G`, each divided by the level's
/// median nearest-neighbour attractor spacing, pooled over levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HullReport {
    /// min, p10, median, p90, max of distance-to-nearest / spacing.
    pub to_nearest: [f64; 5],
    /// The same for distance to the soft mixture `P·A` (always inside the
    /// hull).
    pub to_mixture: [f64; 5],
    pub positions: usize,
}

fn quantiles(mut values: Vec<f64>) -> [f64; 5] {
    if values.is_empty() {
        return [f64::NAN; 5];
    }
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    [at(0.0), at(0.1), at(0.5), at(0.9), at(1.0)]
}

/// Evenly spaced windows of `span` tokens over a corpus.
pub fn fixed_windows(corpus: &mut TokenCorpus, count: usize, span: usize) -> Result<Vec<Vec<u16>>> {
    spaced_windows(corpus, count, span, 1)
}

fn spaced_windows(
    corpus: &mut TokenCorpus,
    count: usize,
    span: usize,
    stride: usize,
) -> Result<Vec<Vec<u16>>> {
    ensure!(
        corpus.len() >= span,
        "{} tokens cannot hold a window of {span}",
        corpus.len()
    );
    let room = (corpus.len() - span) / stride;
    (0..count)
        .map(|i| {
            let start = if count <= 1 {
                0
            } else {
                (i * room / (count - 1)) * stride
            };
            corpus.window(start, span)
        })
        .collect()
}

/// Split windows of `prompt_len + L` into a left-padded prompt (the last
/// `prompt_lens[r]` tokens of the prefix) and the target.
fn split_windows(
    windows: &[Vec<u16>],
    prompt_len: usize,
    prompt_lens: &[usize],
    pad: usize,
    vocab: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let mut prompts = Vec::with_capacity(windows.len() * prompt_len);
    let mut targets = Vec::new();
    for (window, &p) in windows.iter().zip(prompt_lens) {
        ensure!(window.len() > prompt_len, "window too short");
        ensure!(
            window.iter().all(|&t| usize::from(t) < vocab),
            "corpus token outside the codec vocabulary of {vocab}"
        );
        let p = p.min(prompt_len);
        prompts.extend(std::iter::repeat_n(pad as i64, prompt_len - p));
        prompts.extend(
            window[prompt_len - p..prompt_len]
                .iter()
                .map(|&t| i64::from(t)),
        );
        targets.extend(window[prompt_len..].iter().map(|&t| i64::from(t)));
    }
    Ok((prompts, targets))
}

fn encode_host<B: Backend, C: ConceptCodec<B>>(
    codec: &C,
    targets: &[i64],
    rows: usize,
    device: &B::Device,
) -> Result<Vec<Vec<i64>>> {
    let seq = codec.schedule().seq_len();
    let tokens = int_tensor::<B>(targets, rows, seq, device);
    codec
        .encode(tokens)?
        .codes
        .into_iter()
        .map(host_i64)
        .collect()
}

fn rows_of(level_codes: &[i64], len: usize, rows: &[usize]) -> Vec<i64> {
    rows.iter()
        .flat_map(|&r| level_codes[r * len..(r + 1) * len].iter().copied())
        .collect()
}

/// Validate per level on fixed windows of `prompt_len + L` tokens.
pub fn evaluate_levels<B: Backend<FloatElem = f32>, C: SoftConceptCodec<B>>(
    model: &NcmGenerator<B>,
    codec: &C,
    windows: &[Vec<u16>],
    prompt_len: usize,
    pad: usize,
    relax: &RelaxOptions,
    device: &B::Device,
) -> Result<LevelEval> {
    let schedule = codec.schedule().clone();
    let classes = codec.codebook_size();
    let levels = schedule.levels();
    let mut ce_sum = vec![0.0f64; levels];
    let mut hits = vec![0usize; levels];
    let mut counts = vec![0usize; levels];
    let mut histograms = vec![vec![0usize; classes]; levels];
    for chunk in windows.chunks(8) {
        let rows = chunk.len();
        let lens = vec![prompt_len; rows];
        let (prompts, targets) = split_windows(chunk, prompt_len, &lens, pad, codec.vocab_size())?;
        let codes = encode_host(codec, &targets, rows, device)?;
        let prompt = (prompt_len > 0).then(|| int_tensor::<B>(&prompts, rows, prompt_len, device));
        let mut tensors = Vec::with_capacity(levels);
        for (k, &len) in schedule.lengths().iter().enumerate() {
            let zhat = codec.reconstruct(&tensors, rows)?;
            let logits = model
                .forward_level(codec, k, zhat, prompt.as_ref(), relax)?
                .logits;
            let labels = int_tensor::<B>(&codes[k], rows, len, device);
            let picked =
                log_softmax(logits.clone(), 2).gather(2, labels.clone().unsqueeze_dim::<3>(2));
            ce_sum[k] -= host_f32(picked.sum())?
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>();
            let predicted = argmax_rows(&host_f32(logits)?, classes);
            hits[k] += predicted
                .iter()
                .zip(&codes[k])
                .filter(|(a, b)| a == b)
                .count();
            counts[k] += rows * len;
            for &c in &codes[k] {
                if let Some(slot) = histograms[k].get_mut(c as usize) {
                    *slot += 1;
                }
            }
            tensors.push(labels);
        }
    }
    let cross_entropy: Vec<f64> = ce_sum
        .iter()
        .zip(&counts)
        .map(|(s, &n)| s / n.max(1) as f64)
        .collect();
    let accuracy = hits
        .iter()
        .zip(&counts)
        .map(|(&h, &n)| h as f64 / n.max(1) as f64)
        .collect();
    let unigram_entropy = histograms
        .iter()
        .zip(&counts)
        .map(|(hist, &n)| {
            hist.iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / n.max(1) as f64;
                    -p * p.ln()
                })
                .sum()
        })
        .collect();
    let q = level_sampling_distribution(&schedule, 0.0)?;
    let weighted = cross_entropy.iter().zip(&q).map(|(c, w)| c * w).sum();
    Ok(LevelEval {
        cross_entropy,
        accuracy,
        weighted,
        uniform_baseline: (classes as f64).ln(),
        unigram_entropy,
        windows: windows.len(),
    })
}

/// The EBRM drift diagnostic of [`HullReport`], with clean conditioning.
pub fn hull_report<B: Backend<FloatElem = f32>, C: SoftConceptCodec<B>>(
    model: &NcmGenerator<B>,
    codec: &C,
    windows: &[Vec<u16>],
    prompt_len: usize,
    pad: usize,
    relax: &RelaxOptions,
    device: &B::Device,
) -> Result<HullReport> {
    let schedule = codec.schedule().clone();
    let metric = model.full_metric();
    let mut near = Vec::new();
    let mut mix = Vec::new();
    let rows = windows.len().min(8);
    ensure!(rows >= 1, "hull report needs at least one window");
    let lens = vec![prompt_len; rows];
    let (prompts, targets) =
        split_windows(&windows[..rows], prompt_len, &lens, pad, codec.vocab_size())?;
    let codes = encode_host(codec, &targets, rows, device)?;
    let prompt = (prompt_len > 0).then(|| int_tensor::<B>(&prompts, rows, prompt_len, device));
    let mut tensors = Vec::new();
    let mut options = *relax;
    options.record_states = true;
    for (k, &len) in schedule.lengths().iter().enumerate() {
        let zhat = codec.reconstruct(&tensors, rows)?;
        let forward = model.forward_level(codec, k, zhat, prompt.as_ref(), &options)?;
        let state = forward
            .states
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("no recorded state"))?;
        let attractors = model.attractors(codec, k)?;
        // Spacing: median nearest-neighbour distance among (up to 256)
        // attractors.
        let sample = attractors
            .clone()
            .narrow(0, 0, attractors.dims()[0].min(256));
        let [s, h] = sample.dims();
        let pairwise = host_f32(metric_sq_dist(
            &sample.clone().reshape([1, s, h]),
            &attractors,
            &metric,
        ))?;
        let classes = attractors.dims()[0];
        let mut nn = Vec::with_capacity(s);
        for (i, row) in pairwise.chunks(classes).enumerate() {
            let best = row
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &d)| f64::from(d.max(0.0)))
                .fold(f64::INFINITY, f64::min);
            nn.push(best.sqrt());
        }
        let spacing = quantiles(nn)[2].max(1e-12);
        let dist = host_f32(metric_sq_dist(&state, &attractors, &metric))?;
        for row in dist.chunks(classes) {
            let best = row
                .iter()
                .map(|&d| f64::from(d.max(0.0)))
                .fold(f64::INFINITY, f64::min);
            near.push(best.sqrt() / spacing);
        }
        let p = softmax(
            distance_logits(&state, &attractors, &metric, model.temperature()),
            2,
        );
        let soft = soft_concept(p, &attractors);
        let diff = state.sub(soft);
        let [b, n, hdim] = diff.dims();
        let d2 = diff
            .clone()
            .reshape([b * n, hdim])
            .matmul(metric.clone())
            .mul(diff.reshape([b * n, hdim]))
            .sum_dim(1)
            .div_scalar(hdim as f64);
        for v in host_f32(d2)? {
            mix.push(f64::from(v.max(0.0)).sqrt() / spacing);
        }
        tensors.push(int_tensor::<B>(&codes[k], rows, len, device));
    }
    Ok(HullReport {
        positions: near.len(),
        to_nearest: quantiles(near),
        to_mixture: quantiles(mix),
    })
}

/// One optimizer step, for the caller's progress display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NcmStepLog {
    pub step: usize,
    pub loss: f64,
    pub lr: f64,
    pub grad_norm: f64,
    pub clipped: bool,
    /// Levels trained this step and the rows each got.
    pub level_rows: Vec<(usize, usize)>,
    pub val: Option<LevelEval>,
}

/// What a training run did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NcmTrainReport {
    pub start_step: usize,
    pub steps_taken: usize,
    pub steps_skipped: usize,
    pub steps_clipped: usize,
    pub first_loss: f64,
    pub last_loss: f64,
    pub mean_loss: f64,
    pub val_start: Option<LevelEval>,
    pub val_end: Option<LevelEval>,
    pub hull_end: Option<HullReport>,
    /// Examples trained per level.
    pub level_visits: Vec<usize>,
    pub replaced_uniform: usize,
    pub replaced_self: usize,
    pub trunk_evals: usize,
    pub params: usize,
    pub elapsed_secs: f64,
    pub checkpoint: Option<PathBuf>,
}

/// The corpora a run reads.
pub struct NcmTrainData<'a> {
    pub train: &'a mut TokenCorpus,
    /// Held-out corpus for the per-level validation (fixed windows).
    pub validation: Option<&'a mut TokenCorpus>,
}

/// What ties a run to its frozen tokenizer, and where it resumes from.
pub struct TrainSession<B: AutodiffBackend> {
    /// sha256 of the tokenizer checkpoint (recorded; a resume with a
    /// different one is refused).
    pub tokenizer_sha256: String,
    pub resume: Option<ResumePoint<B>>,
}

/// Where a resumed run continues from.
pub struct ResumePoint<B: AutodiffBackend> {
    pub step: usize,
    pub rng: ChaCha12Rng,
    pub optimizer: Option<GeneratorOptimRecord<B>>,
}

/// The training sidecar's generator-specific part.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeneratorExtras {
    tokenizer_sha256: String,
    spec: GeneratorSpec,
}

/// Train the generator against a frozen codec.
///
/// Per step: windows of `prompt_len + L` tokens (each row's prompt is the
/// last `p ~ U{0..P}` tokens before its target, so the model learns both
/// unconditional and prompted continuation), frozen `encode`, then the
/// [`LossEstimator`]. AdamW `β = (0.9, 0.95)`, warmup + cosine from
/// `peak_lr` to `min_lr`, global grad-norm clipping, the backend reseeded
/// from `(seed, step)` each step and the host RNG checkpointed, so a resumed
/// run draws what the uninterrupted one would have.
pub fn train_generator<B, C>(
    mut model: NcmGenerator<B>,
    codec: &C,
    data: NcmTrainData<'_>,
    config: &NcmTrainConfig,
    session: TrainSession<B>,
    device: &B::Device,
    progress: &mut dyn FnMut(&NcmStepLog),
) -> Result<(NcmGenerator<B>, NcmTrainReport)>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
    C: SoftConceptCodec<B>,
{
    let NcmTrainData { train, validation } = data;
    let TrainSession {
        tokenizer_sha256,
        resume,
    } = session;
    config.validate()?;
    model.check_codec(codec)?;
    let spec = GeneratorSpec {
        config: config.model.clone(),
        codec: CodecShape::of(codec)?,
    };
    let schedule = codec.schedule().clone();
    let levels = schedule.levels();
    let seq_len = schedule.seq_len();
    let prompt_len = config.model.prompt_len;
    let span = prompt_len + seq_len;
    ensure!(
        train.len() > span,
        "the training corpus ({} tokens) is shorter than one window of {span}",
        train.len()
    );
    if config.aligned_windows {
        ensure!(
            train.len().is_multiple_of(span)
                && validation
                    .as_ref()
                    .is_none_or(|corpus| corpus.len().is_multiple_of(span)),
            "aligned train and validation corpora must contain whole records of {span} tokens"
        );
    }
    let classes = codec.codebook_size();
    let pad = config.model.pad_id;
    let relax = config.model.relax();
    let mut relax_train = relax;
    relax_train.record_step_logits = config.model.deep_supervision > 0.0;
    let q = level_sampling_distribution(&schedule, config.model.alpha)?;
    let lr = config.lr_schedule();
    let inner_codec = InnerCodec::<B, C>::new(codec);
    let inner_device = device.clone();
    let corruption = CorruptionContext {
        schedule: &schedule,
        corruption: &config.model.corruption,
        codebook_size: classes,
    };

    let (start_step, mut rng, mut optim) = match resume {
        Some(point) => {
            let mut optim = new_optimizer::<B>(config);
            if let Some(record) = point.optimizer {
                optim = optim.load_record(record);
            }
            (point.step, point.rng, optim)
        }
        None => (
            0,
            ChaCha12Rng::seed_from_u64(config.seed),
            new_optimizer::<B>(config),
        ),
    };
    let val_windows = match validation {
        Some(corpus) => {
            let stride = if config.aligned_windows { span } else { 1 };
            Some(spaced_windows(corpus, config.val_windows.max(1), span, stride)?)
        }
        None => None,
    };
    let evaluate = |model: &NcmGenerator<B>| -> Result<Option<LevelEval>> {
        match &val_windows {
            Some(windows) => {
                let inner = model.valid();
                evaluate_levels(
                    &inner,
                    &inner_codec,
                    windows,
                    prompt_len,
                    pad,
                    &relax,
                    &inner_device,
                )
                .map(Some)
            }
            None => Ok(None),
        }
    };
    let mut report = NcmTrainReport {
        start_step,
        steps_taken: 0,
        steps_skipped: 0,
        steps_clipped: 0,
        first_loss: f64::NAN,
        last_loss: f64::NAN,
        mean_loss: f64::NAN,
        val_start: evaluate(&model)?,
        val_end: None,
        hull_end: None,
        level_visits: vec![0; levels],
        replaced_uniform: 0,
        replaced_self: 0,
        trunk_evals: 0,
        params: model.num_params(),
        elapsed_secs: 0.0,
        checkpoint: None,
    };
    let datasets = vec![DatasetIdentity::of_path(
        train.path(),
        "ncm generator training corpus",
    )?];
    let meta = CheckpointMeta {
        config,
        spec: &spec,
        tokenizer_sha256: &tokenizer_sha256,
        datasets: &datasets,
    };
    let started = std::time::Instant::now();
    let mut loss_sum = 0.0;
    for step in start_step..config.steps {
        B::seed(device, crate::train::step_seed(config.seed, step));
        let batch = config.batch_size;
        let (windows, prompt_lens) = if config.aligned_windows {
            (
                crate::geometry::sample_aligned_batch(train, span, batch, &mut rng)?,
                vec![prompt_len; batch],
            )
        } else {
            let mut windows = Vec::with_capacity(batch);
            let mut prompt_lens = Vec::with_capacity(batch);
            for _ in 0..batch {
                let start = rng.random_range(0..=train.len() - span);
                windows.push(train.window(start, span)?);
                prompt_lens.push(if prompt_len > 0 {
                    rng.random_range(0..=prompt_len)
                } else {
                    0
                });
            }
            (windows, prompt_lens)
        };
        let (prompts, targets) =
            split_windows(&windows, prompt_len, &prompt_lens, pad, codec.vocab_size())?;
        let clean = encode_host(codec, &targets, batch, device)?;
        let groups: Vec<(usize, Vec<usize>, f64)> = match config.estimator {
            LossEstimator::SampledLevel => {
                let mut by_level: Vec<Vec<usize>> = vec![Vec::new(); levels];
                for r in 0..batch {
                    let u = rng.random::<f64>();
                    let mut acc = 0.0;
                    let mut chosen = levels - 1;
                    for (k, &p) in q.iter().enumerate() {
                        acc += p;
                        if u < acc {
                            chosen = k;
                            break;
                        }
                    }
                    by_level[chosen].push(r);
                }
                by_level
                    .into_iter()
                    .enumerate()
                    .filter(|(_, rows)| !rows.is_empty())
                    .map(|(k, rows)| {
                        let share = rows.len() as f64 / batch as f64;
                        (k, rows, share)
                    })
                    .collect()
            }
            LossEstimator::AllLevels => (0..levels)
                .map(|k| (k, (0..batch).collect(), q[k]))
                .collect(),
        };
        let mut total: Option<Tensor<B, 1>> = None;
        let mut level_rows = Vec::with_capacity(groups.len());
        for (level, rows, share) in &groups {
            let (level, share) = (*level, *share);
            let n = rows.len();
            level_rows.push((level, n));
            report.level_visits[level] += n;
            let clean_rows: Vec<Vec<i64>> = schedule
                .lengths()
                .iter()
                .enumerate()
                .map(|(j, &len)| rows_of(&clean[j], len, rows))
                .collect();
            let prompt_rows = (prompt_len > 0).then(|| {
                let flat = rows_of(&prompts, prompt_len, rows);
                int_tensor::<B>(&flat, n, prompt_len, device)
            });
            let own = if config.model.corruption.self_prob > 0.0 && level > 0 {
                let request = SelfConditioning {
                    clean_rows: &clean_rows,
                    level,
                    rows: n,
                    prompt: prompt_rows.as_ref(),
                };
                Some(self_conditioning_codes(
                    &model,
                    &inner_codec,
                    request,
                    &relax,
                    &mut rng,
                    &inner_device,
                )?)
            } else {
                None
            };
            let example =
                build_level_example(&clean_rows, level, n, corruption, own.as_deref(), &mut rng)?;
            report.replaced_uniform += example.replaced_uniform;
            report.replaced_self += example.replaced_self;
            let conditioning: Vec<Tensor<B, 2, Int>> = example
                .conditioning
                .iter()
                .enumerate()
                .map(|(j, c)| int_tensor::<B>(c, n, schedule.lengths()[j], device))
                .collect();
            let zhat = codec.reconstruct(&conditioning, n)?.detach();
            let len = schedule.lengths()[level];
            let labels = int_tensor::<B>(&example.labels, n, len, device);
            let forward =
                model.forward_level(codec, level, zhat, prompt_rows.as_ref(), &relax_train)?;
            report.trunk_evals += forward.usage.rounds;
            let mut loss = level_cross_entropy(forward.logits, labels.clone());
            if config.model.deep_supervision > 0.0 && forward.step_logits.len() > 1 {
                let inner_steps = &forward.step_logits[1..];
                let mut ds: Option<Tensor<B, 1>> = None;
                for logits in inner_steps {
                    let term = level_cross_entropy(logits.clone(), labels.clone());
                    ds = Some(match ds {
                        Some(d) => d.add(term),
                        None => term,
                    });
                }
                if let Some(ds) = ds {
                    loss = loss.add(
                        ds.mul_scalar(config.model.deep_supervision / inner_steps.len() as f64),
                    );
                }
            }
            let weighted = loss.mul_scalar(share);
            total = Some(match total {
                Some(t) => t.add(weighted),
                None => weighted,
            });
        }
        let loss = total.ok_or_else(|| anyhow!("step {step} trained no level"))?;
        let loss_value = f64::from(
            host_f32(loss.clone().detach())?
                .first()
                .copied()
                .unwrap_or(f32::NAN),
        );
        if !loss_value.is_finite() {
            report.steps_skipped += 1;
            continue;
        }
        let mut grads = GradientsParams::from_grads(loss.backward(), &model);
        let grad_norm = crate::quality::global_grad_norm(&model, &grads);
        let clipped = config.grad_clip > 0.0 && grad_norm > config.grad_clip as f32;
        if clipped {
            crate::schedule::clip_gradients(&mut grads, &model, grad_norm, config.grad_clip as f32);
            report.steps_clipped += 1;
        }
        let rate = lr.at(step);
        model = optim.step(rate, model, grads);
        if report.steps_taken == 0 {
            report.first_loss = loss_value;
        }
        report.last_loss = loss_value;
        loss_sum += loss_value;
        report.steps_taken += 1;
        let val = if config.val_every > 0
            && (step + 1) % config.val_every == 0
            && step + 1 < config.steps
        {
            evaluate(&model)?
        } else {
            None
        };
        progress(&NcmStepLog {
            step,
            loss: loss_value,
            lr: rate,
            grad_norm: f64::from(grad_norm),
            clipped,
            level_rows,
            val,
        });
        if config.checkpoint_every > 0 && (step + 1) % config.checkpoint_every == 0 {
            if let Some(dir) = &config.out_dir {
                save_generator_checkpoint(&model, &optim, dir, step + 1, &rng, &meta)?;
            }
        }
    }
    report.elapsed_secs = started.elapsed().as_secs_f64();
    report.mean_loss = loss_sum / report.steps_taken.max(1) as f64;
    report.val_end = evaluate(&model)?;
    if let Some(windows) = &val_windows {
        let inner = model.valid();
        report.hull_end = Some(hull_report(
            &inner,
            &inner_codec,
            windows,
            prompt_len,
            pad,
            &relax,
            &inner_device,
        )?);
    }
    if let Some(dir) = &config.out_dir {
        report.checkpoint = Some(save_generator_checkpoint(
            &model,
            &optim,
            dir,
            config.steps,
            &rng,
            &meta,
        )?);
    }
    Ok((model, report))
}

/// Which rows and level [`self_conditioning_codes`] predicts for.
struct SelfConditioning<'a, B: Backend> {
    clean_rows: &'a [Vec<i64>],
    level: usize,
    rows: usize,
    prompt: Option<&'a Tensor<B, 2, Int>>,
}

/// The model's own codes for every conditioning level `j < level`, each
/// predicted from clean `ẑ^(j-1)` and sampled at temperature 1, under no
/// gradient (the "vaccination" source of [`CorruptionSchedule::self_prob`]).
fn self_conditioning_codes<B, C>(
    model: &NcmGenerator<B>,
    codec: &InnerCodec<'_, B, C>,
    request: SelfConditioning<'_, B>,
    relax: &RelaxOptions,
    rng: &mut ChaCha12Rng,
    device: &B::Device,
) -> Result<Vec<Vec<i64>>>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
    C: SoftConceptCodec<B>,
{
    let SelfConditioning {
        clean_rows,
        level,
        rows,
        prompt,
    } = request;
    let inner = model.valid();
    let schedule = codec.schedule().clone();
    let prompt = prompt.map(|p| p.clone().inner());
    let mut committed: Vec<Tensor<B::InnerBackend, 2, Int>> = Vec::with_capacity(level);
    let mut own = Vec::with_capacity(level);
    let sampling = CodeSampling {
        temperature: 1.0,
        top_k: 0,
    };
    for (j, clean_j) in clean_rows.iter().enumerate().take(level) {
        let zhat = codec.reconstruct(&committed, rows)?;
        let logits = inner
            .forward_level(codec, j, zhat, prompt.as_ref(), relax)?
            .logits;
        own.push(sample_rows(
            &host_f32(logits)?,
            codec.codebook_size(),
            &sampling,
            rng,
        )?);
        committed.push(int_tensor::<B::InnerBackend>(
            clean_j,
            rows,
            schedule.lengths()[j],
            device,
        ));
    }
    Ok(own)
}

/// What a checkpoint records besides weights, optimizer and RNG.
struct CheckpointMeta<'a> {
    config: &'a NcmTrainConfig,
    spec: &'a GeneratorSpec,
    tokenizer_sha256: &'a str,
    datasets: &'a [DatasetIdentity],
}

fn save_generator_checkpoint<B: AutodiffBackend<FloatElem = f32>>(
    model: &NcmGenerator<B>,
    optim: &GeneratorOptim<B>,
    dir: &Path,
    step: usize,
    rng: &ChaCha12Rng,
    meta: &CheckpointMeta<'_>,
) -> Result<PathBuf> {
    let CheckpointMeta {
        config,
        spec,
        tokenizer_sha256,
        datasets,
    } = *meta;
    let model_path = checkpoint::save_content_addressed(model.clone(), dir, "ncm-generator")?;
    let state_dir = TrainState::dir_for(&model_path);
    if state_dir.exists() {
        std::fs::remove_dir_all(&state_dir)
            .with_context(|| format!("clear {}", state_dir.display()))?;
    }
    let optimizer = Some(checkpoint::save_record::<B, _>(
        optim.to_record(),
        &state_dir,
        "optimizer",
    )?);
    let extras = GeneratorExtras {
        tokenizer_sha256: tokenizer_sha256.to_string(),
        spec: spec.clone(),
    };
    let state = TrainState {
        format_version: checkpoint::STATE_FORMAT_VERSION,
        kind: STATE_KIND.into(),
        step,
        seed: config.seed,
        host_rng: serde_json::to_value(rng).context("serialize host RNG")?,
        config: serde_json::to_value(config).context("serialize training config")?,
        build: checkpoint::BuildInfo::current(),
        datasets: datasets.to_vec(),
        model: checkpoint::model_entry(&model_path)?,
        optimizer,
        ema: None,
        head: None,
        head_optimizer: None,
        extras: serde_json::to_value(&extras).context("serialize generator extras")?,
        saved_unix_secs: checkpoint::unix_now(),
    };
    state.write(&state_dir)?;
    Ok(model_path)
}

/// A generator loaded from its checkpoint.
pub struct LoadedGenerator<B: Backend> {
    pub model: NcmGenerator<B>,
    pub spec: GeneratorSpec,
    pub tokenizer_sha256: String,
    pub step: usize,
}

fn read_state(path: &Path) -> Result<(TrainState, GeneratorExtras)> {
    let state = TrainState::for_model(path)?
        .with_context(|| format!("no training state next to {}", path.display()))?;
    ensure!(
        state.kind == STATE_KIND,
        "training state is for {:?}, not {STATE_KIND:?}",
        state.kind
    );
    state.verify_files(&TrainState::dir_for(path))?;
    let extras: GeneratorExtras =
        serde_json::from_value(state.extras.clone()).context("parse generator extras")?;
    Ok((state, extras))
}

/// Load a generator checkpoint (the `.mpk` and its `.state` sidecar).
pub fn load_generator<B: Backend<FloatElem = f32>>(
    path: &Path,
    device: &B::Device,
) -> Result<LoadedGenerator<B>> {
    let (state, extras) = read_state(path)?;
    let fresh = NcmGenerator::new(&extras.spec.config, &extras.spec.codec, device)?;
    let model = checkpoint::load(fresh, path, device)?;
    Ok(LoadedGenerator {
        model,
        spec: extras.spec,
        tokenizer_sha256: extras.tokenizer_sha256,
        step: state.step,
    })
}

/// Load a checkpoint to resume training: refuses a different tokenizer (by
/// sha256), a different codec shape, or a different configuration (apart
/// from the run-length and output keys).
pub fn resume_point<B>(
    path: &Path,
    config: &NcmTrainConfig,
    shape: &CodecShape,
    tokenizer_sha256: &str,
    device: &B::Device,
) -> Result<(NcmGenerator<B>, ResumePoint<B>)>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let (state, extras) = read_state(path)?;
    ensure!(
        extras.tokenizer_sha256 == tokenizer_sha256,
        "refusing to resume: the generator was trained against tokenizer {} but {} was given",
        extras.tokenizer_sha256,
        tokenizer_sha256
    );
    ensure!(
        &extras.spec.codec == shape,
        "refusing to resume: the codec shape changed ({:?} -> {:?})",
        extras.spec.codec,
        shape
    );
    let current = serde_json::to_value(config).context("serialize training config")?;
    let differences = state.config_differences(&current);
    ensure!(
        differences.is_empty(),
        "refusing to resume: configuration differs in {differences:?}"
    );
    ensure!(
        state.step <= config.steps,
        "the checkpoint is at step {}, past the requested {} steps",
        state.step,
        config.steps
    );
    let fresh = NcmGenerator::new(&extras.spec.config, &extras.spec.codec, device)?;
    let model = checkpoint::load(fresh, path, device)?;
    let state_dir = TrainState::dir_for(path);
    let optimizer = match &state.optimizer {
        Some(file) => Some(checkpoint::load_record::<B, GeneratorOptimRecord<B>>(
            &state_dir, file, device,
        )?),
        None => None,
    };
    let rng: ChaCha12Rng =
        serde_json::from_value(state.host_rng.clone()).context("restore host RNG")?;
    Ok((
        model,
        ResumePoint {
            step: state.step,
            rng,
            optimizer,
        },
    ))
}

// ------------------------------------------------------- certificates --

fn ncm_cert(
    name: &'static str,
    theorem: &'static str,
    residual: f64,
    tolerance: f64,
) -> Certificate {
    Certificate {
        group: CERT_GROUP,
        name,
        theorem,
        residual,
        tolerance,
    }
}

type CertBackend = burn::backend::NdArray<f32>;

fn max_abs_diff<B: Backend, const D: usize>(a: &Tensor<B, D>, b: &Tensor<B, D>) -> Result<f64> {
    let x = host_f32(a.clone())?;
    let y = host_f32(b.clone())?;
    ensure!(
        x.len() == y.len(),
        "shape mismatch {} vs {}",
        x.len(),
        y.len()
    );
    Ok(x.iter()
        .zip(&y)
        .map(|(p, q)| {
            if p.is_nan() || q.is_nan() {
                f64::NAN
            } else {
                f64::from((p - q).abs())
            }
        })
        .fold(0.0, |acc, v| {
            if acc.is_nan() || v.is_nan() {
                f64::NAN
            } else {
                acc.max(v)
            }
        }))
}

/// A small codec and generator for the certificates.
fn cert_fixture(
    config: &NcmGeneratorConfig,
) -> Result<(ReferenceCodec<CertBackend>, NcmGenerator<CertBackend>)> {
    let device = Default::default();
    let schedule = LevelSchedule::new(vec![1, 2, 4, 8])?;
    let codec = ReferenceCodec::<CertBackend>::new(
        schedule,
        12,
        6,
        crate::tokenizer::VOCAB_SIZE,
        7,
        &device,
    )?;
    let shape = CodecShape::of(&codec)?;
    let model = NcmGenerator::new(config, &shape, &device)?;
    Ok((codec, model))
}

fn cert_config() -> NcmGeneratorConfig {
    NcmGeneratorConfig {
        blocks: 1,
        hidden: 8,
        heads: 2,
        mlp_ratio: 2,
        prompt_len: 4,
        thinking_steps: 2,
        ..NcmGeneratorConfig::tiny()
    }
}

/// A prompt `[2, 4]` whose second row is half padding.
fn cert_prompt(device: &burn::backend::ndarray::NdArrayDevice) -> Tensor<CertBackend, 2, Int> {
    let pad = default_pad_id() as i64;
    int_tensor::<CertBackend>(&[72, 101, 108, 112, pad, pad, 111, 33], 2, 4, device)
}

/// A level model that calls its inner model twice per level: the control
/// for the NFE count.
struct DoubleCall<'a, M>(&'a M);

impl<B: Backend, M: LevelModel<B>> LevelModel<B> for DoubleCall<'_, M> {
    fn level_logits(
        &self,
        level: usize,
        zhat: Tensor<B, 3>,
        prompt: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 3>> {
        self.0.level_logits(level, zhat.clone(), prompt.clone())?;
        self.0.level_logits(level, zhat, prompt)
    }
}

fn cert_generation_calls() -> Result<f64> {
    let config = cert_config();
    let (codec, model) = cert_fixture(&config)?;
    let device = Default::default();
    let k = codec.schedule().levels();
    let prompt = cert_prompt(&device);
    let field = ConceptFieldLm::new(&model, &codec, config.relax())?;
    let counted = CountingLevelModel::new(&field);
    let mut rng = ChaCha12Rng::seed_from_u64(1);
    generate_levels(
        &counted,
        &codec,
        Some(prompt.clone()),
        2,
        &CodeSampling::greedy(),
        false,
        &mut rng,
    )?;
    let mut residual = counted.calls().abs_diff(k) as f64;
    let rounds: usize = field.take_traces()?.iter().map(|t| t.usage.rounds).sum();
    residual += rounds.abs_diff(k * config.thinking_steps) as f64;
    // Through the full entry point, with three MBR samples.
    let options = GenerateOptions {
        codes: CodeSampling {
            temperature: 1.0,
            top_k: 4,
        },
        token_temperature: 0.0,
        mbr_samples: 3,
        seed: 3,
        relax: config.relax(),
        record_thoughts: false,
    };
    let generation = generate(&model, &codec, Some(prompt.clone()), 2, &options)?;
    residual += generation.nfe_per_sample.abs_diff(k) as f64;
    residual += generation.nfe_total.abs_diff(3 * k) as f64;
    residual += generation
        .trunk_evals
        .abs_diff(3 * k * config.thinking_steps) as f64;
    // Control: a model that loops is caught by the same count.
    let inner_counter = CountingLevelModel::new(&field);
    let doubled_counted = DoubleCall(&inner_counter);
    generate_levels(
        &doubled_counted,
        &codec,
        Some(prompt),
        2,
        &CodeSampling::greedy(),
        false,
        &mut rng,
    )?;
    if inner_counter.calls() == k {
        residual += 1.0;
    }
    Ok(residual)
}

fn prefix_residual(relax: RelaxOptions) -> Result<f64> {
    let mut config = cert_config();
    config.adaptive = relax.adaptive;
    let (codec, model) = cert_fixture(&config)?;
    let device = Default::default();
    let mut options = relax;
    options.record_prompt = true;
    let prompt = cert_prompt(&device);
    let codes = codec.encode(int_tensor::<CertBackend>(
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        2,
        8,
        &device,
    ))?;
    let level = 2;
    let zhat = codec.reconstruct(&codes.codes[..level], 2)?;
    let base = model.forward_level(&codec, level, zhat.clone(), Some(&prompt), &options)?;
    let perturbed_targets = zhat
        .clone()
        .add(Tensor::<CertBackend, 3>::ones(zhat.dims(), &device).mul_scalar(0.37));
    let moved = model.forward_level(&codec, level, perturbed_targets, Some(&prompt), &options)?;
    let mut residual = 0.0f64;
    ensure!(!base.prompt_states.is_empty(), "no prompt states recorded");
    // The adaptive schedule may differ between the two runs; compare every
    // trunk evaluation both runs made.
    for (a, b) in base.prompt_states.iter().zip(&moved.prompt_states) {
        residual = residual.max(max_abs_diff(a, b)?);
    }
    // Anti-vacuity: the targets did change the target outputs ...
    if max_abs_diff(&base.logits, &moved.logits)? == 0.0 {
        residual += 1.0;
    }
    // ... and the prompt reaches the targets.
    let other_prompt = int_tensor::<CertBackend>(&[65, 66, 67, 68, 69, 70, 71, 72], 2, 4, &device);
    let prompted = model.forward_level(&codec, level, zhat, Some(&other_prompt), &options)?;
    if max_abs_diff(&base.logits, &prompted.logits)? == 0.0 {
        residual += 1.0;
    }
    Ok(residual)
}

fn cert_conditional_independence() -> Result<f64> {
    let config = cert_config();
    let (codec, model) = cert_fixture(&config)?;
    let device = Default::default();
    let prompt = cert_prompt(&device);
    let field = ConceptFieldLm::new(&model, &codec, config.relax())?;
    let mut rng = ChaCha12Rng::seed_from_u64(5);
    let sampling = CodeSampling {
        temperature: 1.0,
        top_k: 0,
    };
    let run = generate_levels(
        &field,
        &codec,
        Some(prompt.clone()),
        2,
        &sampling,
        true,
        &mut rng,
    )?;
    let mut residual = 0.0f64;
    let mut control_moved = false;
    let classes = codec.codebook_size();
    for level in 1..codec.schedule().levels() - 1 {
        // Replace this level's sampled codes (and every finer one) with
        // different codes; only the coarser ones are kept.
        let mut perturbed: Vec<Tensor<CertBackend, 2, Int>> = run.codes[..level].to_vec();
        let shifted = host_i64(run.codes[level].clone())?
            .iter()
            .map(|&c| (c + 1) % classes as i64)
            .collect::<Vec<_>>();
        let len = codec.schedule().lengths()[level];
        let shifted = int_tensor::<CertBackend>(&shifted, 2, len, &device);
        let input = codec.reconstruct(&perturbed, 2)?;
        residual = residual.max(max_abs_diff(&input, &run.inputs[level])?);
        let again = field.level_logits(level, input, Some(prompt.clone()))?;
        residual = residual.max(max_abs_diff(&again, &run.logits[level])?);
        // Control: had the level's own (perturbed) codes leaked into its
        // input, the logits would move.
        perturbed.push(shifted);
        let leaky = codec.reconstruct(&perturbed, 2)?;
        let leaked = field.level_logits(level, leaky, Some(prompt.clone()))?;
        if max_abs_diff(&leaked, &run.logits[level])? > 0.0 {
            control_moved = true;
        }
    }
    if !control_moved {
        residual += 1.0;
    }
    Ok(residual)
}

fn cert_corruption_labels() -> Result<f64> {
    let schedule = LevelSchedule::new(vec![1, 2, 4, 8])?;
    let rows = 3;
    let classes = 9;
    let mut rng = ChaCha12Rng::seed_from_u64(11);
    let clean: Vec<Vec<i64>> = schedule
        .lengths()
        .iter()
        .map(|&l| {
            (0..rows * l)
                .map(|_| rng.random_range(0..classes) as i64)
                .collect()
        })
        .collect();
    let own: Vec<Vec<i64>> = clean
        .iter()
        .map(|c| c.iter().map(|&x| (x + 4) % classes as i64).collect())
        .collect();
    let mut residual = 0.0;
    let mut changed = [0usize; 3];
    let sources = [
        CorruptionSchedule::default(),
        CorruptionSchedule {
            self_prob: 1.0,
            ..CorruptionSchedule::default()
        },
        CorruptionSchedule {
            self_prob: 0.5,
            ..CorruptionSchedule::default()
        },
    ];
    for _ in 0..200 {
        for (s, corruption) in sources.iter().enumerate() {
            let level = rng.random_range(0..schedule.levels());
            let context = CorruptionContext {
                schedule: &schedule,
                corruption,
                codebook_size: classes,
            };
            let example = build_level_example(&clean, level, rows, context, Some(&own), &mut rng)?;
            if example.labels != clean[level] {
                residual += 1.0;
            }
            for (j, cond) in example.conditioning.iter().enumerate() {
                for (idx, (&c, &x)) in cond.iter().zip(&clean[j]).enumerate() {
                    if c != x {
                        changed[s] += 1;
                        // A pure-self draw may only ever write the model's code.
                        if s == 1 && c != own[j][idx] {
                            residual += 1.0;
                        }
                    }
                }
            }
        }
    }
    // Anti-vacuity: every source did corrupt the conditioning.
    residual += changed.iter().filter(|&&c| c == 0).count() as f64;
    // Control: the same comparison flags a builder that writes into labels.
    let context = CorruptionContext {
        schedule: &schedule,
        corruption: &sources[0],
        codebook_size: classes,
    };
    let mut broken = build_level_example(&clean, 2, rows, context, None, &mut rng)?;
    if let Some(first) = broken.labels.first_mut() {
        *first = (*first + 1) % classes as i64;
    }
    if broken.labels == clean[2] {
        residual += 1.0;
    }
    Ok(residual)
}

fn cert_weighted_loss() -> Result<f64> {
    let device = Default::default();
    let schedule = LevelSchedule::new(vec![1, 2, 4])?;
    let alpha = 0.5;
    let weights = schedule.loss_weights(alpha)?;
    let batch = 2;
    let classes = 5;
    let mut rng = ChaCha12Rng::seed_from_u64(13);
    let mut logits = Vec::new();
    let mut labels = Vec::new();
    let mut host_logits = Vec::new();
    let mut host_labels = Vec::new();
    for &len in schedule.lengths() {
        let values: Vec<f32> = (0..batch * len * classes)
            .map(|_| rng.random_range(-3.0..3.0))
            .collect();
        let ys: Vec<i64> = (0..batch * len)
            .map(|_| rng.random_range(0..classes) as i64)
            .collect();
        logits.push(float_tensor::<CertBackend, 3>(
            &values,
            [batch, len, classes],
            &device,
        ));
        labels.push(int_tensor::<CertBackend>(&ys, batch, len, &device));
        host_logits.push(values);
        host_labels.push(ys);
    }
    let implemented = f64::from(host_f32(weighted_ncm_loss(&logits, &labels, &weights)?)?[0]);
    // Scalar recomputation: −Σ_k w_k Σ_i log p, averaged over the batch.
    let formula = |w: &[f64]| -> f64 {
        let mut total = 0.0;
        for (k, &len) in schedule.lengths().iter().enumerate() {
            for b in 0..batch {
                for i in 0..len {
                    let row = &host_logits[k][(b * len + i) * classes..(b * len + i + 1) * classes];
                    let max = row
                        .iter()
                        .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
                    let lse = row
                        .iter()
                        .map(|&v| (f64::from(v) - max).exp())
                        .sum::<f64>()
                        .ln()
                        + max;
                    let y = host_labels[k][b * len + i] as usize;
                    total -= w[k] * (f64::from(row[y]) - lse);
                }
            }
        }
        total / batch as f64
    };
    let exact = formula(&weights);
    let mut residual = (implemented - exact).abs() / exact.abs().max(1e-12);
    // The sampled-level estimator's exact expectation equals the formula.
    let q = level_sampling_distribution(&schedule, alpha)?;
    let mut expectation = 0.0;
    for (k, logits_k) in logits.iter().enumerate() {
        let mean_ce =
            f64::from(host_f32(level_cross_entropy(logits_k.clone(), labels[k].clone()))?[0]);
        expectation += q[k] * mean_ce;
    }
    residual = residual.max((expectation - exact).abs() / exact.abs().max(1e-12));
    // Control: per-level uniform weights are a different objective here.
    let uniform = vec![1.0 / 3.0; 3];
    if (formula(&uniform) - exact).abs() / exact.abs().max(1e-12) <= 1e-5 {
        residual += 1.0;
    }
    Ok(residual)
}

/// A random positive-definite metric for the energy check.
fn random_metric(
    dim: usize,
    rng: &mut ChaCha12Rng,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> Tensor<CertBackend, 2> {
    let mut l = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            l[i * dim + j] = if i == j {
                rng.random_range(0.5..1.5)
            } else {
                rng.random_range(-0.5..0.5)
            };
        }
    }
    let l = float_tensor::<CertBackend, 2>(&l, [dim, dim], device);
    l.clone().matmul(l.transpose())
}

fn cert_energy_descends() -> Result<f64> {
    let device = Default::default();
    let mut rng = ChaCha12Rng::seed_from_u64(17);
    let (batch, n, dim, codes) = (2, 5, 6, 7);
    let g = random_metric(dim, &mut rng, &device);
    let a_values: Vec<f32> = (0..codes * dim)
        .map(|_| rng.random_range(-2.0..2.0))
        .collect();
    let a = float_tensor::<CertBackend, 2>(&a_values, [codes, dim], &device);
    let u_values: Vec<f32> = (0..batch * n * dim)
        .map(|_| rng.random_range(-3.0..3.0))
        .collect();
    let u0 = float_tensor::<CertBackend, 3>(&u_values, [batch, n, dim], &device);
    let tau = 0.7;
    let mut residual = 0.0f64;
    for eta in [0.25, 1.0, 1.75, 2.0] {
        let mut u = u0.clone();
        let mut energy = host_f32(attractor_energy(&u, &a, &g, tau))?;
        for _ in 0..20 {
            u = attractor_step(&u, &a, &g, tau, eta);
            let next = host_f32(attractor_energy(&u, &a, &g, tau))?;
            for (e0, e1) in energy.iter().zip(&next) {
                let scale = f64::from(e0.abs()).max(1.0);
                residual = residual.max(f64::from(e1 - e0) / scale);
            }
            energy = next;
        }
    }
    // Control: past the bound (η = 3) the energy rises. With one attractor
    // the update is `u ← a + (1 − η)(u − a)`, scaling the energy by
    // `(1 − η)² = 4`.
    let single = a.clone().narrow(0, 0, 1);
    let before = host_f32(attractor_energy(&u0, &single, &g, tau))?;
    let after = host_f32(attractor_energy(
        &attractor_step(&u0, &single, &g, tau, 3.0),
        &single,
        &g,
        tau,
    ))?;
    let rose = before.iter().zip(&after).any(|(b, a)| a > b);
    if !rose {
        residual += 1.0;
    }
    Ok(residual.max(0.0))
}

fn cert_soft_concept() -> Result<f64> {
    let config = cert_config();
    let (codec, model) = cert_fixture(&config)?;
    let device = Default::default();
    let level = 2;
    let attractors = model.attractors(&codec, level)?;
    let metric = model.full_metric();
    let mut rng = ChaCha12Rng::seed_from_u64(19);
    let values: Vec<f32> = (0..2 * 4 * 8)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let u = float_tensor::<CertBackend, 3>(&values, [2, 4, 8], &device);
    let p = softmax(
        distance_logits(&u, &attractors, &metric, model.temperature()),
        2,
    );
    let soft = soft_concept(p.clone(), &attractors);
    let ph = host_f32(p.clone())?;
    let classes = codec.codebook_size();
    let mut residual = 0.0f64;
    for row in ph.chunks(classes) {
        let sum: f64 = row.iter().map(|&v| f64::from(v)).sum();
        residual = residual.max((sum - 1.0).abs());
        if row.iter().any(|&v| v < 0.0) {
            residual += 1.0;
        }
    }
    // The soft concept is Σ_j P_j a_j, recomputed on the host.
    let ah = host_f32(attractors.clone())?;
    let sh = host_f32(soft)?;
    for (r, row) in ph.chunks(classes).enumerate() {
        for d in 0..8 {
            let expected: f64 = row
                .iter()
                .enumerate()
                .map(|(j, &w)| f64::from(w) * f64::from(ah[j * 8 + d]))
                .sum();
            residual = residual.max((expected - f64::from(sh[r * 8 + d])).abs());
        }
    }
    // In code space: P·E lies inside the codebook's bounding box, and the
    // field's attractor map commutes with the mixture (W_a has no bias).
    let book = codec.level_codebook(level)?;
    let [_, code_dim] = book.dims();
    let mixture = p.clone().reshape([8, classes]).matmul(book.clone());
    let mh = host_f32(mixture.clone())?;
    let bh = host_f32(book)?;
    for d in 0..code_dim {
        let lo = (0..classes)
            .map(|j| bh[j * code_dim + d])
            .fold(f32::INFINITY, f32::min);
        let hi = (0..classes)
            .map(|j| bh[j * code_dim + d])
            .fold(f32::NEG_INFINITY, f32::max);
        for r in 0..8 {
            let v = mh[r * code_dim + d];
            residual = residual.max(f64::from((lo - v).max(v - hi).max(0.0)));
        }
    }
    let lifted = model
        .attractor_in
        .forward(mixture.reshape([2, 4, code_dim]));
    residual = residual.max(max_abs_diff(&lifted, &soft_concept(p, &attractors))?);
    // Control: unnormalised weights fail the distribution check.
    let raw = distance_logits(&u, &attractors, &metric, model.temperature()).exp();
    let worst_sum = host_f32(raw.sum_dim(2))?
        .iter()
        .map(|&s| (f64::from(s) - 1.0).abs())
        .fold(0.0, f64::max);
    if worst_sum <= 1e-5 {
        residual += 1.0;
    }
    Ok(residual)
}

/// Host-side LayerNorm over the last axis.
fn host_layer_norm(x: &[f64], width: usize, gamma: &[f32], beta: &[f32], eps: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(width) {
        let mean = row.iter().sum::<f64>() / width as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / width as f64;
        for (i, v) in row.iter().enumerate() {
            out.push((v - mean) / (var + eps).sqrt() * f64::from(gamma[i]) + f64::from(beta[i]));
        }
    }
    out
}

/// Host-side `x W + b` for a burn `Linear` (`W` is `[in, out]`).
fn host_linear(x: &[f64], linear: &Linear<CertBackend>) -> Result<(Vec<f64>, usize)> {
    let [d_in, d_out] = linear.weight.dims();
    let w = host_f32(linear.weight.val())?;
    let b = match &linear.bias {
        Some(b) => host_f32(b.val())?,
        None => vec![0.0; d_out],
    };
    let mut out = Vec::with_capacity(x.len() / d_in * d_out);
    for row in x.chunks(d_in) {
        for o in 0..d_out {
            let mut acc = f64::from(b[o]);
            for (i, v) in row.iter().enumerate() {
                acc += v * f64::from(w[i * d_out + o]);
            }
            out.push(acc);
        }
    }
    Ok((out, d_out))
}

/// One Euclidean-distance-attention block on the host, written from the
/// definition: pre-norm attention with scores `−‖q̃_i − k̃_j‖² / √d_h`, the
/// prefix mask, then a pre-norm SiLU MLP.
fn host_block(
    x: &[f64],
    seq: usize,
    layer: &FieldLayer<CertBackend>,
    positions: &[f64],
    base: f64,
    blocked: &dyn Fn(usize, usize) -> bool,
) -> Result<Vec<f64>> {
    let gamma = |n: &LayerNorm<CertBackend>| host_f32(n.gamma.val());
    let beta = |n: &LayerNorm<CertBackend>| match &n.beta {
        Some(b) => host_f32(b.val()),
        None => Ok(Vec::new()),
    };
    let hidden = x.len() / seq;
    let heads = layer.heads;
    let dh = hidden / heads;
    let eps = 1e-5;
    let a = host_layer_norm(
        x,
        hidden,
        &gamma(&layer.norm_attn)?,
        &beta(&layer.norm_attn)?,
        eps,
    );
    let (q, _) = host_linear(&a, &layer.query)?;
    let (k, _) = host_linear(&a, &layer.key)?;
    let (v, _) = host_linear(&a, &layer.value)?;
    let rope = |t: &[f64], pos: usize, head: usize| -> Vec<f64> {
        let half = dh / 2;
        let base_idx = pos * hidden + head * dh;
        (0..dh)
            .map(|c| {
                let i = c % half;
                let angle = positions[pos] * base.powf(-(2.0 * i as f64) / dh as f64);
                let (cos, sin) = (f64::from(angle.cos() as f32), f64::from(angle.sin() as f32));
                let rotated = if c < half {
                    -t[base_idx + c + half]
                } else {
                    t[base_idx + c - half]
                };
                t[base_idx + c] * cos + rotated * sin
            })
            .collect()
    };
    let mut context = vec![0.0f64; seq * hidden];
    for h in 0..heads {
        for i in 0..seq {
            let qi = rope(&q, i, h);
            let mut scores = Vec::with_capacity(seq);
            for j in 0..seq {
                if blocked(i, j) {
                    scores.push(f64::NEG_INFINITY);
                    continue;
                }
                let kj = rope(&k, j, h);
                let d2: f64 = qi.iter().zip(&kj).map(|(p, r)| (p - r) * (p - r)).sum();
                scores.push(-d2 / (dh as f64).sqrt());
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = exps.iter().sum();
            for c in 0..dh {
                let mut acc = 0.0;
                for (j, e) in exps.iter().enumerate() {
                    acc += e / total * v[j * hidden + h * dh + c];
                }
                context[i * hidden + h * dh + c] = acc;
            }
        }
    }
    let (o, _) = host_linear(&context, &layer.out)?;
    let x1: Vec<f64> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
    let m = host_layer_norm(
        &x1,
        hidden,
        &gamma(&layer.norm_mlp)?,
        &beta(&layer.norm_mlp)?,
        eps,
    );
    let (up, _) = host_linear(&m, &layer.up)?;
    let act: Vec<f64> = up.iter().map(|z| z / (1.0 + (-z).exp())).collect();
    let (down, _) = host_linear(&act, &layer.down)?;
    Ok(x1.iter().zip(&down).map(|(a, b)| a + b).collect())
}

fn cert_degenerate_operator() -> Result<f64> {
    let mut config = cert_config();
    config.metric_epsilon = 0.0;
    config.thinking_steps = 1;
    config.step_size = 1.0;
    config.attractor = false;
    let (codec, model) = cert_fixture(&config)?;
    let device = Default::default();
    let prompt = cert_prompt(&device);
    let level = 2;
    let len = 4;
    let codes = codec.encode(int_tensor::<CertBackend>(
        &[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3],
        2,
        8,
        &device,
    ))?;
    let zhat = codec.reconstruct(&codes.codes[..level], 2)?;
    let relax = config.relax();
    let operator = model
        .forward_level(&codec, level, zhat.clone(), Some(&prompt), &relax)?
        .logits;
    let reference = |model: &NcmGenerator<CertBackend>| -> Result<Vec<f64>> {
        // The operator's inputs, then everything else from the definition.
        let u0 = host_f32(model.initial_state(level, len, zhat.clone())?)?;
        let context = host_f32(model.prompt_context(&prompt, 2)?.embedded)?;
        let ids = host_i64(prompt.clone())?;
        let attractors = host_f32(model.attractors(&codec, level)?)?;
        let hidden = model.hidden();
        let p = 4;
        let seq = p + len;
        let mut positions: Vec<f64> = (0..p).map(|s| s as f64 - p as f64).collect();
        positions.extend((0..len).map(|i| window_centre(i, 8, len)));
        let layer = model.layers.first().ok_or_else(|| anyhow!("no layer"))?;
        let mut logits = Vec::new();
        for b in 0..2 {
            let valid: Vec<bool> = ids[b * p..(b + 1) * p]
                .iter()
                .map(|&t| t as usize != model.pad_id)
                .collect();
            let mut x: Vec<f64> = context[b * p * hidden..(b + 1) * p * hidden]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            x.extend(
                u0[b * len * hidden..(b + 1) * len * hidden]
                    .iter()
                    .map(|&v| f64::from(v)),
            );
            let blocked = |q: usize, k: usize| {
                if q < p {
                    k >= p || !(valid[k] || k == q)
                } else {
                    k < p && !valid[k]
                }
            };
            let out = host_block(&x, seq, layer, &positions, model.rope_base, &blocked)?;
            for i in 0..len {
                let u = &out[(p + i) * hidden..(p + i + 1) * hidden];
                for j in 0..model.codebook_size() {
                    let d2: f64 = u
                        .iter()
                        .enumerate()
                        .map(|(c, &x)| {
                            let diff = x - f64::from(attractors[j * hidden + c]);
                            diff * diff
                        })
                        .sum();
                    logits.push(-d2 / hidden as f64 / (2.0 * model.temperature()));
                }
            }
        }
        Ok(logits)
    };
    let expected = reference(&model)?;
    let got = host_f32(operator)?;
    let scale = expected.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    let mut residual = expected
        .iter()
        .zip(&got)
        .map(|(e, g)| (e - f64::from(*g)).abs() / scale)
        .fold(0.0, f64::max);
    // Control: with a non-identity metric the Euclidean reference no longer
    // describes the operator.
    let mut skewed = model.clone();
    skewed.metric.log_diag =
        Param::from_tensor(Tensor::<CertBackend, 1>::ones([8], &device).mul_scalar(0.4));
    let skewed_logits = host_f32(
        skewed
            .forward_level(&codec, level, zhat, Some(&prompt), &relax)?
            .logits,
    )?;
    let skew = expected
        .iter()
        .zip(&skewed_logits)
        .map(|(e, g)| (e - f64::from(*g)).abs() / scale)
        .fold(0.0, f64::max);
    if skew <= 1e-4 {
        residual += 1.0;
    }
    Ok(residual)
}

fn cert_thought_readout() -> Result<f64> {
    let config = cert_config();
    let (codec, model) = cert_fixture(&config)?;
    let device = Default::default();
    let level = 2;
    let len = 4;
    let batch = 2;
    let classes = codec.codebook_size();
    let attractors = model.attractors(&codec, level)?;
    let metric = model.full_metric();
    let mut rng = ChaCha12Rng::seed_from_u64(23);
    let committed: Vec<Vec<i64>> = (0..level)
        .map(|j| {
            (0..batch * codec.schedule().lengths()[j])
                .map(|_| rng.random_range(0..classes) as i64)
                .collect()
        })
        .collect();
    let codes: Vec<i64> = (0..batch * len)
        .map(|_| rng.random_range(0..classes) as i64)
        .collect();
    // A state sitting exactly on the attractors of `codes`.
    let state = attractors
        .clone()
        .select(
            0,
            Tensor::<CertBackend, 1, Int>::from_ints(codes.as_slice(), &device),
        )
        .reshape([batch, len, model.hidden()]);
    let quantized = argmax_rows(
        &host_f32(distance_logits(
            &state,
            &attractors,
            &metric,
            model.temperature(),
        ))?,
        classes,
    );
    let mut residual = quantized.iter().zip(&codes).filter(|(a, b)| a != b).count() as f64;
    let readout = thought_readout(&codec, &committed, &quantized, batch, &device)?;
    let mut all = committed.clone();
    all.push(codes.clone());
    let tensors: Vec<Tensor<CertBackend, 2, Int>> = all
        .iter()
        .enumerate()
        .map(|(j, c)| int_tensor::<CertBackend>(c, batch, codec.schedule().lengths()[j], &device))
        .collect();
    let direct = decode_tokens(&codec, &tensors, batch, 0.0, &mut rng)?;
    residual += readout
        .iter()
        .flatten()
        .zip(direct.iter().flatten())
        .filter(|(a, b)| a != b)
        .count() as f64;
    // Control: different codes read out differently.
    let other: Vec<i64> = codes.iter().map(|&c| (c + 1) % classes as i64).collect();
    if thought_readout(&codec, &committed, &other, batch, &device)? == readout {
        residual += 1.0;
    }
    Ok(residual)
}

/// One codec and model shared by every run of an adaptive certificate: the
/// backend RNG is process-global, so two fixtures are two different models.
struct AdaptiveFixture {
    codec: ReferenceCodec<CertBackend>,
    model: NcmGenerator<CertBackend>,
    config: NcmGeneratorConfig,
}

fn adaptive_fixture() -> Result<AdaptiveFixture> {
    let mut config = cert_config();
    config.thinking_steps = 3;
    let (codec, model) = cert_fixture(&config)?;
    Ok(AdaptiveFixture {
        codec,
        model,
        config,
    })
}

fn adaptive_forward(
    fixture: &AdaptiveFixture,
    adaptive: Option<AdaptiveThinking>,
    record: bool,
) -> Result<LevelForward<CertBackend>> {
    let AdaptiveFixture {
        codec,
        model,
        config,
    } = fixture;
    let device = Default::default();
    let prompt = cert_prompt(&device);
    let codes = codec.encode(int_tensor::<CertBackend>(
        &[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5],
        2,
        8,
        &device,
    ))?;
    let level = 3;
    let zhat = codec.reconstruct(&codes.codes[..level], 2)?;
    let mut relax = config.relax();
    relax.adaptive = adaptive;
    relax.record_states = record;
    relax.record_step_logits = record;
    model.forward_level(codec, level, zhat, Some(&prompt), &relax)
}

/// The median confidence of the fixture's level-3 positions at `U_0`, the
/// state the first crystallisation decision reads: about half crystallise
/// at once.
fn median_confidence(fixture: &AdaptiveFixture) -> Result<f64> {
    let base = adaptive_forward(fixture, None, true)?;
    let initial = base
        .step_logits
        .first()
        .ok_or_else(|| anyhow!("no U_0 logits recorded"))?;
    let conf = max_probability(initial)?;
    Ok(quantiles(conf.iter().map(|&c| f64::from(c)).collect())[2])
}

fn cert_budget() -> Result<f64> {
    let fixture = adaptive_fixture()?;
    let threshold = median_confidence(&fixture)?;
    let len = 8;
    let steps = 3;
    let mut residual = 0.0;
    for (factor, cap) in [(0.3, 1), (0.6, 3), (1.0, 6), (0.5, 6)] {
        let adaptive = AdaptiveThinking {
            confidence: threshold,
            budget_factor: factor,
            max_steps_per_position: cap,
        };
        let forward = adaptive_forward(&fixture, Some(adaptive), true)?;
        for (spent, budget) in forward
            .usage
            .position_steps
            .iter()
            .zip(&forward.usage.budget)
        {
            residual += spent.saturating_sub(*budget) as f64;
            if *budget != adaptive.budget(len, steps) {
                residual += 1.0;
            }
        }
        // Per-position cap.
        let mut per_position = vec![0usize; 2 * len];
        for round in &forward.active {
            for (idx, &on) in round.iter().enumerate() {
                per_position[idx] += usize::from(on);
            }
        }
        residual += per_position
            .iter()
            .map(|&s| s.saturating_sub(cap))
            .sum::<usize>() as f64;
    }
    // Control: running every position to a cap of 6 would overspend a 0.5
    // budget, and the same comparison sees it.
    let control = AdaptiveThinking {
        confidence: 2.0,
        budget_factor: 0.5,
        max_steps_per_position: 6,
    };
    if len * control.max_steps_per_position <= control.budget(len, steps) {
        residual += 1.0;
    }
    Ok(residual)
}

fn cert_adaptive_off() -> Result<f64> {
    let fixture = adaptive_fixture()?;
    let fixed = adaptive_forward(&fixture, None, true)?;
    let idle = adaptive_forward(
        &fixture,
        Some(AdaptiveThinking {
            confidence: 2.0,
            budget_factor: 1.0,
            max_steps_per_position: 3,
        }),
        true,
    )?;
    let mut residual = max_abs_diff(&fixed.logits, &idle.logits)?;
    for (a, b) in fixed.states.iter().zip(&idle.states) {
        residual = residual.max(max_abs_diff(a, b)?);
    }
    if fixed.states.len() != idle.states.len() {
        residual += 1.0;
    }
    // Control: a binding budget changes the result.
    let binding = adaptive_forward(
        &fixture,
        Some(AdaptiveThinking {
            confidence: 2.0,
            budget_factor: 0.5,
            max_steps_per_position: 3,
        }),
        true,
    )?;
    if max_abs_diff(&fixed.logits, &binding.logits)? == 0.0 {
        residual += 1.0;
    }
    Ok(residual)
}

fn cert_crystals_hold() -> Result<f64> {
    let fixture = adaptive_fixture()?;
    let threshold = median_confidence(&fixture)?;
    let forward = adaptive_forward(
        &fixture,
        Some(AdaptiveThinking {
            confidence: threshold,
            budget_factor: 1.0,
            max_steps_per_position: 5,
        }),
        true,
    )?;
    let len = 8;
    let hidden = 8;
    let mut residual = 0.0f64;
    let states: Vec<Vec<f32>> = forward
        .states
        .iter()
        .map(|s| host_f32(s.clone()))
        .collect::<Result<_>>()?;
    let mut stopped = vec![false; 2 * len];
    let mut froze_any = false;
    for (round, active) in forward.active.iter().enumerate() {
        for (idx, &on) in active.iter().enumerate() {
            if stopped[idx] && on {
                residual += 1.0; // a crystal was re-activated
            }
            if !on {
                stopped[idx] = true;
                froze_any = true;
                let before = &states[round][idx * hidden..(idx + 1) * hidden];
                let after = &states[round + 1][idx * hidden..(idx + 1) * hidden];
                for (x, y) in before.iter().zip(after) {
                    if x.to_bits() != y.to_bits() {
                        residual += 1.0;
                    }
                }
            }
        }
    }
    if !froze_any {
        residual += 1.0;
    }
    // Control: in the fixed schedule positions do move, so the comparison
    // above can see motion.
    let fixed = adaptive_forward(&fixture, None, true)?;
    if max_abs_diff(&fixed.states[0], &fixed.states[1])? == 0.0 {
        residual += 1.0;
    }
    Ok(residual)
}

/// The generator's certificates (group `ncm`).
pub fn certificates() -> Result<Vec<Certificate>> {
    let operator_on = RelaxOptions {
        steps: 3,
        adaptive: Some(AdaptiveThinking {
            confidence: 0.2,
            budget_factor: 0.7,
            max_steps_per_position: 4,
        }),
        ..cert_config().relax()
    };
    let mask_only = RelaxOptions {
        steps: 1,
        attractor: false,
        ..cert_config().relax()
    };
    Ok(vec![
        ncm_cert(
            "generation_takes_exactly_k_calls",
            "Concept-field generation makes exactly K level calls per sample (K·n with n MBR samples) and K·T trunk evaluations at fixed depth, counted by a wrapper; a model that loops is caught by the same count.",
            cert_generation_calls()?,
            0.0,
        ),
        ncm_cert(
            "prefix_mask_hides_targets_from_the_prompt",
            "With the prefix-LM mask, perturbing the level input leaves every prompt-position output bit-identical, while both the level input and the prompt move the target logits.",
            prefix_residual(mask_only)?,
            0.0,
        ),
        ncm_cert(
            "prefix_stays_protected_with_the_operator_on",
            "With T = 3 relaxation steps, the attractor term and crystal growth on, perturbing the level input leaves the prompt-position outputs of every trunk evaluation bit-identical, and the prompt still reaches the targets.",
            prefix_residual(operator_on)?,
            0.0,
        ),
        ncm_cert(
            "level_codes_are_conditionally_independent_given_the_level_input",
            "Level-k logits are a function of the reconstruction of the coarser codes only: replacing the level's own sampled codes (and all finer ones) leaves its input and logits bit-identical, while feeding them in would move the logits.",
            cert_conditional_independence()?,
            0.0,
        ),
        ncm_cert(
            "corruption_never_touches_labels",
            "For every corruption draw, uniform, self-conditioned or mixed, the labels equal the clean codes; every source does corrupt the conditioning, and a pure self draw only writes the model's own codes.",
            cert_corruption_labels()?,
            0.0,
        ),
        ncm_cert(
            "weighted_loss_matches_the_formula",
            "The implemented loss equals a scalar recomputation of −Σ_k w_k Σ_i log p averaged over the batch, and the sampled-level estimator's exact expectation (q_k = w_k ℓ_k) equals it too.",
            cert_weighted_loss()?,
            1e-5,
        ),
        ncm_cert(
            "attractor_energy_descends",
            "With F_θ = 0 the attractor energy A_G is non-increasing for η in {0.25, 1, 1.75, 2} under a random non-identity metric; at η = 3 it rises.",
            cert_energy_descends()?,
            1e-5,
        ),
        ncm_cert(
            "soft_thought_is_a_convex_codebook_mixture",
            "The thought weights P are a distribution per position, the soft concept equals Σ_j P_j a_j, its code-space preimage P·E lies in the codebook's bounding box, and W_a(P·E) = P·(W_a E).",
            cert_soft_concept()?,
            1e-5,
        ),
        ncm_cert(
            "operator_reduces_to_a_distance_head_transformer",
            "With T = 1, η = 1, the attractor term off and G = I, the operator equals one Euclidean-distance-attention block (written on the host from its definition) followed by distance logits; a non-identity metric breaks the match.",
            cert_degenerate_operator()?,
            1e-4,
        ),
        ncm_cert(
            "thought_readout_round_trips_hard_codes",
            "A state placed on the attractors of given codes quantizes back to exactly those codes, and its readout equals the frozen codec's decode of those codes; different codes read out differently.",
            cert_thought_readout()?,
            0.0,
        ),
        ncm_cert(
            "thinking_budget_is_never_exceeded",
            "Under crystal growth no batch row spends more than its position-step budget and no position exceeds its step cap, across budgets and caps.",
            cert_budget()?,
            0.0,
        ),
        ncm_cert(
            "adaptive_off_equals_fixed_steps",
            "Crystal growth that never binds (no crystals, full budget, cap T) reproduces the fixed-depth states and logits bit for bit; a binding budget does not.",
            cert_adaptive_off()?,
            0.0,
        ),
        ncm_cert(
            "crystallised_positions_never_move",
            "Once a position stops updating it is never updated again and its state stays bit-identical; at fixed depth states do move.",
            cert_crystals_hold()?,
            0.0,
        ),
    ])
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

    type B = burn::backend::NdArray<f32>;
    type AB = burn::backend::Autodiff<B>;

    #[test]
    fn test_resize_windows_partition_and_pool() {
        for (from, to) in [(8, 1), (8, 3), (8, 8), (10, 4), (5, 5)] {
            let mut covered = vec![0; from];
            for i in 0..to {
                let (s, e) = resize_window(i, from, to);
                assert!(s < e && e <= from);
                for c in &mut covered[s..e] {
                    *c += 1;
                }
            }
            assert!(covered.iter().all(|&c| c == 1), "{from}->{to}: {covered:?}");
        }
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].as_slice(), &device)
            .reshape([1, 6, 1]);
        let pooled = host_f32(resize_positions(x, 2).unwrap()).unwrap();
        assert_eq!(pooled, vec![2.0, 5.0]);
    }

    #[test]
    fn test_reference_codec_round_trips_its_own_contract() {
        let device = Default::default();
        let codec = ReferenceCodec::<B>::new(
            LevelSchedule::new(vec![1, 2, 4]).unwrap(),
            6,
            4,
            10,
            3,
            &device,
        )
        .unwrap();
        let tokens = int_tensor::<B>(&[1, 2, 3, 4, 5, 6, 7, 8], 2, 4, &device);
        let codes = codec.encode(tokens).unwrap();
        assert_eq!(codes.codes.len(), 3);
        // reconstruct(..=k) − reconstruct(..k) is the level-k code vectors,
        // repeated over their windows.
        let full = codec.reconstruct(&codes.codes, 2).unwrap();
        let partial = codec.reconstruct(&codes.codes[..2], 2).unwrap();
        let book = codec.level_codebook(2).unwrap();
        let rows = book
            .select(0, codes.codes[2].clone().reshape([8]))
            .reshape([2, 4, 4]);
        assert!(max_abs_diff(&full.clone().sub(partial), &rows).unwrap() < 1e-6);
        assert_eq!(codec.decode_logits(full).unwrap().dims(), [2, 4, 10]);
    }

    #[test]
    fn test_forward_level_shapes_and_usage() {
        let config = cert_config();
        let (codec, model) = cert_fixture(&config).unwrap();
        let device = Default::default();
        let prompt = cert_prompt(&device);
        for level in 0..4 {
            let zhat = codec.reconstruct(&[], 2).unwrap();
            let zhat = if level == 0 {
                zhat
            } else {
                zhat.add_scalar(0.1)
            };
            let out = model
                .forward_level(&codec, level, zhat, Some(&prompt), &config.relax())
                .unwrap();
            let len = codec.schedule().lengths()[level];
            assert_eq!(out.logits.dims(), [2, len, 12]);
            assert_eq!(out.usage.rounds, config.thinking_steps);
            assert_eq!(
                out.usage.position_steps,
                vec![len * config.thinking_steps; 2]
            );
        }
    }

    #[test]
    fn test_sampler_respects_top_k_and_greedy() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let logits = [0.0f32, 5.0, 4.9, -1.0];
        let greedy = sample_rows(&logits, 4, &CodeSampling::greedy(), &mut rng).unwrap();
        assert_eq!(greedy, vec![1]);
        for _ in 0..50 {
            let s = sample_rows(
                &logits,
                4,
                &CodeSampling {
                    temperature: 1.0,
                    top_k: 2,
                },
                &mut rng,
            )
            .unwrap();
            assert!(s[0] == 1 || s[0] == 2);
        }
    }

    #[test]
    fn test_level_sampling_distribution_sums_to_one() {
        let s = LevelSchedule::geometric(64, 6).unwrap();
        for alpha in [0.0, 0.5, 1.0] {
            let q = level_sampling_distribution(&s, alpha).unwrap();
            assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_corruption_mean_rate_decreases_with_level() {
        let c = CorruptionSchedule::default();
        let rates: Vec<f64> = (0..5).map(|j| c.mean_rate(j, 6)).collect();
        assert!((rates[0] - 0.5).abs() < 1e-12);
        assert!((rates[4] - 0.1).abs() < 1e-12);
        assert!(rates.windows(2).all(|w| w[0] >= w[1]));
        assert!(CorruptionSchedule {
            coarse_mean: 0.1,
            fine_mean: 0.4,
            ..c.clone()
        }
        .validate()
        .is_err());
        assert!(CorruptionSchedule {
            coarse_mean: 0.8,
            ..c
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_step_size_above_the_bound_is_refused() {
        let config = NcmGeneratorConfig {
            step_size: 2.5,
            ..NcmGeneratorConfig::tiny()
        };
        assert!(config.validate().is_err());
        assert!(NcmGeneratorConfig::paper().validate().is_ok());
        assert!(NcmGeneratorConfig::sudoku().validate().is_ok());
    }

    #[test]
    fn test_generation_reports_measured_costs_and_thoughts() {
        let config = cert_config();
        let (codec, model) = cert_fixture(&config).unwrap();
        let device = Default::default();
        let options = GenerateOptions {
            codes: CodeSampling {
                temperature: 1.0,
                top_k: 0,
            },
            token_temperature: 0.0,
            mbr_samples: 1,
            seed: 9,
            relax: config.relax(),
            record_thoughts: true,
        };
        let generation = generate(&model, &codec, Some(cert_prompt(&device)), 2, &options).unwrap();
        assert_eq!(generation.nfe_per_sample, 4);
        assert_eq!(generation.trunk_evals, 4 * config.thinking_steps);
        assert_eq!(generation.tokens.len(), 2);
        assert_eq!(generation.tokens[0].len(), 8);
        assert_eq!(generation.thoughts.len(), 4);
        // T + 1 states per level, each decodable.
        let level = 2;
        let thoughts = &generation.thoughts[level];
        assert_eq!(thoughts.state_codes.len(), config.thinking_steps + 1);
        let text = thought_readout(
            &codec,
            &generation.codes[..level],
            &thoughts.state_codes[0],
            2,
            &device,
        )
        .unwrap();
        assert_eq!(text.len(), 2);
        // Greedy decoding twice gives the same output (deterministic).
        let greedy = GenerateOptions {
            codes: CodeSampling::greedy(),
            record_thoughts: false,
            ..options
        };
        let a = generate(&model, &codec, None, 1, &greedy).unwrap();
        let b = generate(&model, &codec, None, 1, &greedy).unwrap();
        assert_eq!(a.tokens, b.tokens);
    }

    fn tiny_corpus(name: &str) -> (PathBuf, TokenCorpus) {
        let dir = std::env::temp_dir().join(format!("ncm-gen-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.bin");
        let text = "the quick brown fox jumps over the lazy dog; ".repeat(40);
        let tokens: Vec<u16> = text.bytes().map(u16::from).collect();
        TokenCorpus::write(&path, &tokens).unwrap();
        let corpus = TokenCorpus::in_memory(&path).unwrap();
        (dir, corpus)
    }

    #[test]
    fn aligned_validation_windows_never_split_fixed_records() {
        let (dir, mut corpus) = tiny_corpus("aligned");
        let span = corpus.len() / 40;
        let windows = spaced_windows(&mut corpus, 9, span, span).unwrap();
        assert!(windows.iter().all(|row| row == &windows[0]));
        let staggered = fixed_windows(&mut corpus, 9, span).unwrap();
        assert!(staggered.iter().any(|row| row != &staggered[0]));
        let config = serde_json::to_value(NcmTrainConfig::default()).unwrap();
        assert!(config.get("aligned_windows").is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_training_lowers_the_loss_and_resume_checks_the_tokenizer() {
        let device = Default::default();
        let (dir, mut corpus) = tiny_corpus("train");
        let (_, mut val) = tiny_corpus("val");
        let codec = ReferenceCodec::<AB>::new(
            LevelSchedule::new(vec![1, 2, 4, 8]).unwrap(),
            12,
            6,
            crate::tokenizer::VOCAB_SIZE,
            7,
            &device,
        )
        .unwrap();
        let shape = CodecShape::of(&codec).unwrap();
        let model_config = NcmGeneratorConfig {
            blocks: 1,
            hidden: 16,
            heads: 2,
            prompt_len: 4,
            corruption: CorruptionSchedule {
                self_prob: 0.5,
                ..CorruptionSchedule::default()
            },
            deep_supervision: 0.1,
            ..NcmGeneratorConfig::tiny()
        };
        let config = NcmTrainConfig {
            model: model_config.clone(),
            steps: 30,
            batch_size: 4,
            peak_lr: 3e-3,
            min_lr: 3e-4,
            warmup_steps: 3,
            val_windows: 4,
            out_dir: Some(dir.join("out")),
            ..NcmTrainConfig::default()
        };
        AB::seed(&device, 1);
        let model = NcmGenerator::<AB>::new(&model_config, &shape, &device).unwrap();
        let mut logs = Vec::new();
        let data = NcmTrainData {
            train: &mut corpus,
            validation: Some(&mut val),
        };
        let session = TrainSession {
            tokenizer_sha256: "tok-a".into(),
            resume: None,
        };
        let (_, report) =
            train_generator(model, &codec, data, &config, session, &device, &mut |l| {
                logs.push(l.loss)
            })
            .unwrap();
        assert_eq!(report.steps_taken, 30);
        let start = report.val_start.as_ref().unwrap().weighted;
        let end = report.val_end.as_ref().unwrap().weighted;
        assert!(end < start, "val loss did not fall: {start} -> {end}");
        assert!(report.replaced_self > 0 && report.replaced_uniform > 0);
        assert!(report.hull_end.is_some());
        let path = report.checkpoint.clone().unwrap();
        // A different tokenizer is refused; the right one resumes.
        let more = NcmTrainConfig {
            steps: 32,
            ..config.clone()
        };
        assert!(resume_point::<AB>(&path, &more, &shape, "tok-b", &device).is_err());
        let (model, point) = resume_point::<AB>(&path, &more, &shape, "tok-a", &device).unwrap();
        assert_eq!(point.step, 30);
        let data = NcmTrainData {
            train: &mut corpus,
            validation: None,
        };
        let session = TrainSession {
            tokenizer_sha256: "tok-a".into(),
            resume: Some(point),
        };
        let (_, resumed) =
            train_generator(model, &codec, data, &more, session, &device, &mut |_| {}).unwrap();
        assert_eq!(resumed.steps_taken, 2);
        let loaded = load_generator::<B>(&resumed.checkpoint.unwrap(), &device).unwrap();
        assert_eq!(loaded.tokenizer_sha256, "tok-a");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_concept_field_drives_the_real_tokenizer() {
        // The trained tokenizer's type, freshly initialised: the contract is
        // about shapes and the level loop, not about reconstruction quality.
        let device = Default::default();
        let tok_config = crate::ncm::tokenizer::NcmTokenizerConfig::tiny().unwrap();
        AB::seed(&device, 5);
        let tokenizer =
            crate::ncm::tokenizer::NcmTokenizer::<AB>::new(&tok_config, &device).unwrap();
        crate::tensor_ext::force_initialization(&tokenizer);
        let tokenizer = tokenizer.no_grad();
        let shape = CodecShape::of(&tokenizer).unwrap();
        assert_eq!(shape.lengths, vec![2, 4, 8, 16, 32, 64]);
        let model_config = NcmGeneratorConfig {
            blocks: 1,
            hidden: 16,
            heads: 2,
            prompt_len: 8,
            ..NcmGeneratorConfig::tiny()
        };
        let model = NcmGenerator::<AB>::new(&model_config, &shape, &device).unwrap();
        let (dir, mut corpus) = tiny_corpus("real-tokenizer");
        let config = NcmTrainConfig {
            model: model_config.clone(),
            steps: 3,
            batch_size: 2,
            ..NcmTrainConfig::default()
        };
        let data = NcmTrainData {
            train: &mut corpus,
            validation: None,
        };
        let session = TrainSession {
            tokenizer_sha256: "fresh".into(),
            resume: None,
        };
        let (model, report) = train_generator(
            model,
            &tokenizer,
            data,
            &config,
            session,
            &device,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(report.steps_taken, 3);
        assert!(report.last_loss.is_finite());
        // Generation on the inner backend through the same codec.
        let inner_codec = InnerCodec::<AB, _>::new(&tokenizer);
        let inner = model.valid();
        let options = GenerateOptions {
            codes: CodeSampling {
                temperature: 1.0,
                top_k: 8,
            },
            token_temperature: 0.0,
            mbr_samples: 1,
            seed: 1,
            relax: model_config.relax(),
            record_thoughts: true,
        };
        let ids = crate::tokenizer::ByteTokenizer::new().encode("GET /index");
        let prompt = prompt_tensor::<B>(&ids, 8, model_config.pad_id, 1, &device);
        let generation = generate(&inner, &inner_codec, Some(prompt), 1, &options).unwrap();
        assert_eq!(generation.nfe_per_sample, 6);
        assert_eq!(generation.trunk_evals, 6 * model_config.thinking_steps);
        assert_eq!(generation.tokens[0].len(), 64);
        let level = 3;
        let thought = &generation.thoughts[level].state_codes[1];
        let text = thought_readout(
            &inner_codec,
            &generation.codes[..level],
            thought,
            1,
            &device,
        )
        .unwrap();
        assert_eq!(text[0].len(), 64);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_generator_certificates_hold() {
        let certificates = certificates().unwrap();
        for c in &certificates {
            println!(
                "{:<66} residual {:.3e} tolerance {:.1e}",
                c.name, c.residual, c.tolerance
            );
        }
        let failed: Vec<String> = certificates
            .iter()
            .filter(|c| !c.passed())
            .map(|c| format!("{}: residual {} > {}", c.name, c.residual, c.tolerance))
            .collect();
        assert!(failed.is_empty(), "{failed:#?}");
        assert_eq!(certificates.len(), 13);
    }
}
