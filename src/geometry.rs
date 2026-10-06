//! Geometric reasoning over a learned Riemannian scene geometry (roadmap
//! Phase 33).
//!
//! A conventional transformer earns its answers from scale: more layers, more
//! data, more compute per token. This module is the other policy. A
//! [`GeometricReasoner`] builds an explicit geometric description of a scene
//! -- entities as points in a learned Riemannian space, relations as metric
//! distances -- and answers by *relaxing* that description into an attractor
//! basin: a fixed point of an explicit energy, reached by a certified descent.
//! The learned parameters shape the landscape (the metric, the attention, the
//! writers); the reasoning itself is fixed geometric dynamics whose depth and
//! precision are inference-time knobs, so the answers come from the geometry
//! rather than from brute-force training.
//!
//! # The four papers this phase implements
//!
//! - *Attractor State* (Michels): the answer state is a fixed point of an
//!   explicit energy `E`; relaxation descends `E` with a step size derived
//!   from a closed-form Lipschitz bound, so monotone decrease and convergence
//!   are certificates (`geom` group), not assumptions.
//! - *Rule by Technocratic Mind Control* (Michels): the model is its own
//!   verifier. No external reward model grades the answer; the readout
//!   consumes the state the geometry itself certifies (final energy,
//!   displacement, convergence), and evaluation reports those diagnostics
//!   beside accuracy.
//! - *The Dissolution of a False Divide* (Michels): the scene is carried as
//!   two complementary descriptions of one state -- a symbolic stream
//!   (discrete tokens) and a geometric stream (continuous points). Each
//!   refines the other through geodesic attention under the shared metric;
//!   the landscape is fixed within a block and redrawn from the refined
//!   symbolic stream by the next, so the two descriptions remain two views of
//!   one recursive trajectory.
//! - *Principia Cybernetica II* (Michels): information geometry. Distances
//!   are computed under a learned metric `G = L L^T` with `L` parameterized
//!   so `G` is positive definite by construction; attention is inverse
//!   distance under `G` (geodesic attention); and the block-diffusion
//!   machinery of this crate runs on the metric space.
//!
//! # Block diffusion
//!
//! The reasoner is a block-diffusion model of the attractor state,
//! conditioned on the scene. The sigma trajectory is partitioned into
//! `num_blocks` blocks (block 0 noisiest, the convention of
//! [`crate::sigma::block_sigmas`]); block `b` is trained standalone on the
//! clean state corrupted at its window's noise scale, EDM-preconditioned
//! ([`crate::sigma::EdmPreconditioning`]), with boundary consistency at the
//! shared sigma between adjacent blocks -- the exact training scheme of this
//! crate's DiffusionBlocks core, applied to a geometric latent. Inference
//! runs the same trajectory at any noise scale: the clean path (scale zero)
//! is the fast reasoning path, and `--diffusion` generation starts from pure
//! noise and denoises into the scene's basin, with the number of refinement
//! steps a runtime knob.

use std::collections::BTreeSet;
use std::ops::Range;
use std::path::PathBuf;

use burn::module::{Initializer, Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution, Int, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::checkpoint;
use crate::corpus::TokenCorpus;
use crate::expert_index::{BalanceWeights, BoxSpec, ExpertSpec, MosmeSpec};
use crate::mosme::{BalanceBreakdown, HierarchicalRouter, MosmeConfig};

pub const GRID_MAX: i32 = 15;

/// The question kinds the synthetic corpus draws from, in rendering order.
pub const KINDS: [&str; 5] = ["nearest", "farthest", "direction", "inside", "collinear"];

/// Configuration for a [`GeometricReasoner`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeomConfig {
    /// The byte tokenizer's vocabulary (256 bytes + the three specials).
    pub vocab_size: usize,
    /// Tokens per scene, fixed: every document is exactly this long and ends
    /// in its answer token. The model reads `scene_len - 1` of them.
    pub scene_len: usize,
    pub hidden_size: usize,
    /// Concept points `K` -- the geometric stream's slots.
    pub slots: usize,
    /// Dimension of the geometric space.
    pub dim: usize,
    /// How many blocks the refinement trajectory is partitioned into (block 0
    /// noisiest, the [`crate::sigma`] convention).
    pub num_blocks: usize,
    /// Relaxation iterations per block.
    pub refine_steps: usize,
    pub cond_hidden_size: usize,
    pub frequency_embedding_size: usize,
    pub layer_norm_eps: f64,
    pub initializer_range: f64,
    /// The repulsion coefficient the blocks are initialized at; each block
    /// learns its own.
    pub repulsion: f64,
    /// EDM `sigma_data` for the preconditioning.
    pub sigma_data: f64,
    /// MoSME readout: boxes of specialized expert heads (one box per question
    /// kind), two-level sparse routing. `0` disables the mixture and uses the
    /// plain linear readout -- the certified identity setting.
    pub moe_boxes: usize,
    /// Experts per box.
    pub moe_experts: usize,
    /// Experts selected per box, and boxes selected overall.
    pub moe_top_k: usize,
    /// The closed answer set: the only tokens a question can be answered
    /// with. Empty means "score the whole byte vocabulary" (the identity
    /// setting, kept for the certificates and for a free-form reader).
    ///
    /// A scene's question has a *closed* answer space -- four letters for a
    /// `nearest` scene, sixteen across all five kinds -- so putting a 259-way
    /// softmax over every byte spends the readout's capacity discriminating
    /// tokens that are never answers, and makes the loss pay for it. This is
    /// the vocabulary-trimming result of "Cut Your Losses in Large-Vocabulary
    /// Language Models" (ICLR 2025) applied where the answer set is *known*:
    /// here it is declared, not guessed.
    pub answer_tokens: Vec<u16>,
}

impl Default for GeomConfig {
    fn default() -> Self {
        Self {
            vocab_size: crate::tokenizer::VOCAB_SIZE,
            scene_len: 40,
            hidden_size: 64,
            slots: 8,
            dim: 8,
            num_blocks: 2,
            refine_steps: 2,
            cond_hidden_size: 64,
            frequency_embedding_size: 16,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            repulsion: 0.5,
            sigma_data: 0.5,
            moe_boxes: 5,
            moe_experts: 2,
            moe_top_k: 1,
            // Empty by default: the unrestricted byte vocabulary is the
            // identity setting, and a caller that knows the answer set sets
            // it (the trainer and evaluator do, from `GeomMeta`).
            answer_tokens: Vec::new(),
        }
    }
}

impl GeomConfig {
    /// The small configuration, for CPU smoke runs: the 4-point grid.
    pub fn tiny() -> Self {
        Self {
            scene_len: 30,
            hidden_size: 32,
            slots: 6,
            dim: 6,
            cond_hidden_size: 8,
            moe_boxes: 3,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.vocab_size == crate::tokenizer::VOCAB_SIZE,
            "the byte tokenizer's vocabulary is {}",
            crate::tokenizer::VOCAB_SIZE
        );
        anyhow::ensure!(
            self.scene_len >= 15,
            "a scene needs at least 15 tokens (header, one point, query, answer)"
        );
        anyhow::ensure!(
            self.slots >= 2 && self.slots <= self.scene_len - 2,
            "slots must be in [2, scene_len - 2]"
        );
        anyhow::ensure!(
            self.dim >= 1 && self.hidden_size >= 1,
            "dimensions must be positive"
        );
        anyhow::ensure!(
            self.num_blocks >= 1 && self.refine_steps >= 1,
            "blocks and refinement steps must be positive"
        );
        anyhow::ensure!(
            self.frequency_embedding_size >= 2 && self.frequency_embedding_size % 2 == 0,
            "the frequency embedding size must be even"
        );
        anyhow::ensure!(self.sigma_data > 0.0, "sigma_data must be positive");
        anyhow::ensure!(
            self.moe_experts >= 1 && self.moe_top_k >= 1 && self.moe_top_k <= self.moe_experts,
            "the expert count must be positive and at least the top-k"
        );
        Ok(())
    }

    /// One line naming the architecture, for banners and records.
    pub fn describe(&self) -> String {
        let mixture = if self.moe_boxes > 0 {
            format!(
                " moe=boxes:{}x{}@top{}",
                self.moe_boxes, self.moe_experts, self.moe_top_k
            )
        } else {
            String::new()
        };
        format!(
            "slots={} dim={} hidden={} blocks={} refine={} scene={}{mixture}",
            self.slots,
            self.dim,
            self.hidden_size,
            self.num_blocks,
            self.refine_steps,
            self.scene_len
        )
    }
}

/// The metadata sidecar [`generate_corpus`] writes beside a corpus: what the
/// trainer and the evaluator need to read aligned scenes and decode answers.
/// Plain serde JSON, no Burn dependency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeomMeta {
    pub scene_len: usize,
    pub points: usize,
    pub kinds: Vec<String>,
    /// The distinct answer bytes the corpus uses.
    pub answer_tokens: Vec<u16>,
    pub scenes: usize,
    pub seed: u64,
    /// Scenes drawn per requested kind, parallel to `kinds`. Recorded so a
    /// run can tell a balanced corpus from a skewed one (AlphaGeometry's
    /// lesson: 9% aux-construction density starves hard constructions).
    /// Empty on sidecars written before this field existed.
    #[serde(default)]
    pub kind_counts: Vec<usize>,
    /// Similarity-augmented copies requested per scene (`0` = off, the
    /// default). Augmented copies are rotation + translation + uniform
    /// scale transforms of their original scene with the label recomputed
    /// on the grid; direction scenes are augmented with translation + scale
    /// only, because their answer is an absolute compass bearing, which a
    /// rotation would rotate.
    #[serde(default)]
    pub augment: usize,
}

/// A learned Riemannian metric: `G = L L^T` with `L` lower-triangular and a
/// positive diagonal *by construction* -- the strictly lower part and the log
/// of the diagonal are the free parameters, so `G` is positive definite for
/// any parameter values the optimizer can reach.
#[derive(Module, Debug)]
pub struct MetricField<B: Backend> {
    /// The strictly lower-triangular part of `L` (the diagonal is replaced).
    pub(crate) lower: Param<Tensor<B, 2>>,
    /// `log(diag(L))`; the diagonal is `exp(log_diag) > 0` by construction.
    pub(crate) log_diag: Param<Tensor<B, 1>>,
}

impl<B: Backend<FloatElem = f32>> MetricField<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            lower: Param::from_tensor(Tensor::<B, 2>::zeros([dim, dim], device)),
            log_diag: Param::from_tensor(Tensor::<B, 1>::zeros([dim], device)),
        }
    }

    /// The Cholesky factor `L` (lower-triangular, positive diagonal). A fresh
    /// metric is the identity.
    pub fn cholesky(&self) -> Tensor<B, 2> {
        let device = self.log_diag.val().device();
        let [dim, _] = self.lower.dims();
        let row = Tensor::<B, 1, Int>::arange(0..dim as i64, &device).unsqueeze_dim::<2>(1);
        let col = Tensor::<B, 1, Int>::arange(0..dim as i64, &device).unsqueeze_dim::<2>(0);
        let strict = row.clone().greater(col.clone()).float();
        let masked = self.lower.val().mul(strict);
        let eye = row.equal(col).float();
        let diag = self.log_diag.val().exp().unsqueeze_dim::<2>(1).mul(eye);
        masked.add(diag)
    }

    /// The metric matrix `G = L L^T`, `[dim, dim]`.
    pub fn metric(&self) -> Tensor<B, 2> {
        let l = self.cholesky();
        l.clone().matmul(l.transpose())
    }

    /// Squared distances under the metric: `a` `[batch, a_n, dim]` against
    /// `b` `[batch, b_n, dim]` gives `[batch, a_n, b_n]`.
    pub fn sq_dist(&self, a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3> {
        let g = self.metric();
        let [batch, a_n, dim] = a.dims();
        let [_, b_n, _] = b.dims();
        let diff = a
            .unsqueeze_dim::<4>(2)
            .sub(b.unsqueeze_dim::<4>(1))
            .reshape([batch * a_n * b_n, dim]);
        let transformed = diff.clone().matmul(g).reshape([batch, a_n, b_n, dim]);
        transformed
            .mul(diff.reshape([batch, a_n, b_n, dim]))
            .sum_dim(3)
            .reshape([batch, a_n, b_n])
    }
}

/// `t [batch, k, dim] @ G [dim, dim]` -- the flatten trick keeps the matmul
/// two-dimensional.
fn matmul_g3<B: Backend>(t: Tensor<B, 3>, g: &Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, k, dim] = t.dims();
    t.reshape([batch * k, dim])
        .matmul(g.clone())
        .reshape([batch, k, dim])
}

/// The dominant eigenvalue of a symmetric PSD matrix, by power iteration
/// from a *fixed* start. Deterministic: every call on the same matrix
/// returns the same value, which is what makes a refinement span reproduce
/// the full path bit for bit after training. The caller inflates the
/// estimate (see [`lipschitz_step`]) so the step size stays strictly inside
/// the descent lemma's stability range even if the iteration undershoots.
pub(crate) fn spectral_norm<B: Backend<FloatElem = f32>>(
    m: &Tensor<B, 2>,
    iterations: usize,
) -> f32 {
    let [rows, _] = m.dims();
    let device = m.device();
    let mut v = Tensor::<B, 1>::ones([rows], &device).mul_scalar((rows as f32).sqrt().recip());
    for _ in 0..iterations {
        let mv = m
            .clone()
            .matmul(v.clone().reshape([rows, 1]))
            .reshape([rows]);
        let norm = mv.clone().mul(mv.clone()).sum().sqrt().into_scalar();
        v = mv.div_scalar(norm.max(1e-8));
    }
    let mv = m.clone().matmul(v.reshape([rows, 1])).reshape([rows]);
    mv.clone().mul(mv).sum().sqrt().into_scalar()
}

/// The relaxation step size: one over the closed-form Lipschitz bound of the
/// energy's gradient, `L = 2 (1 + 2 lambda K) lambda_max(G)`, computed on a
/// 10%-inflated power-iteration estimate of the metric's top eigenvalue.
pub(crate) fn lipschitz_step<B: Backend<FloatElem = f32>>(
    metric: &Tensor<B, 2>,
    lambda: f32,
    slots: usize,
) -> f32 {
    let lambda_max = spectral_norm(metric, 24) * 1.1;
    let bound = 2.0 * (1.0 + 2.0 * lambda * slots as f32) * lambda_max * lambda_max;
    1.0 / bound.max(1e-8)
}

/// Upper bound on the geodesic-attention temperature as a multiple of the
/// head dim. Past ~2·dim every score is flatter than -0.5 and the attention
/// is uniform for all practical purposes, so a temperature beyond it is
/// drift, not signal: with no upper bound the model can always buy smaller
/// gradients by turning the temperature up instead of learning geometry
/// (the metric-learning failure mode the Riemannian literature warns has no
/// large-scale stabilization report). A clamp, not a remap, so every
/// existing checkpoint is bit-identical: trained temperatures sit near
/// `softplus(0) ≈ 0.69`, far below any reachable cap.
pub fn temperature_cap(dim: usize) -> f32 {
    2.0 * dim.max(1) as f32
}

/// Attention temperature from a raw logit, bounded on both sides:
/// `softplus` plus `1e-3` keeps it positive and off zero (no division blowup
/// in `-d²/tau`), and [`temperature_cap`] stops runaway drift from
/// flattening the attention into uniformity.
pub fn bounded_temperature(raw_tau: f32, dim: usize) -> f32 {
    let tau = (1.0 + raw_tau.exp()).ln() + 1e-3;
    tau.max(1e-3).min(temperature_cap(dim))
}

/// Geodesic attention: softmax over the *inverse* metric distance, so
/// attention weights are relations read from the learned geometry rather than
/// arbitrary learned logits. `q [batch, q_n, dim]`, `k [batch, k_n, dim]` ->
/// `[batch, q_n, k_n]`. The temperature is floored (never divide by zero)
/// and capped (never uniform-by-drift); see [`bounded_temperature`].
pub fn geodesic_attention<B: Backend>(
    metric: &Tensor<B, 2>,
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    tau: f32,
) -> Tensor<B, 3> {
    let [batch, q_n, dim] = q.dims();
    let [_, k_n, _] = k.dims();
    let diff = q
        .unsqueeze_dim::<4>(2)
        .sub(k.unsqueeze_dim::<4>(1))
        .reshape([batch * q_n * k_n, dim]);
    let transformed = diff
        .clone()
        .matmul(metric.clone())
        .reshape([batch, q_n, k_n, dim]);
    let d2 = transformed
        .mul(diff.reshape([batch, q_n, k_n, dim]))
        .sum_dim(3)
        .reshape([batch, q_n, k_n]);
    let tau = tau.max(1e-3).min(temperature_cap(dim));
    softmax(d2.mul_scalar(-1.0 / tau), 2)
}

/// The energy over the geometric stream, against a fixed context:
/// `E = sum_k ||z_k - c_k||^2_G + lambda sum_{k != l} ||z_k - z_l||^2_G`.
/// Attachment draws each slot toward its context; repulsion keeps the slots
/// from collapsing onto one point.
pub(crate) fn energy_value<B: Backend<FloatElem = f32>>(
    g: &Tensor<B, 2>,
    z: &Tensor<B, 3>,
    context: &Tensor<B, 3>,
    lambda: f32,
    slots: usize,
) -> f32 {
    let [batch, k, dim] = z.dims();
    let diff = z.clone().sub(context.clone());
    let att = matmul_g3(diff.clone(), g)
        .mul(diff)
        .sum_dim(2)
        .reshape([batch, k])
        .sum_dim(1)
        .reshape([batch]);
    let z_g = matmul_g3(z.clone(), g);
    let z2 = z_g
        .mul(z.clone())
        .sum_dim(2)
        .reshape([batch, k])
        .sum_dim(1)
        .reshape([batch]);
    let sums = z.clone().sum_dim(1).reshape([batch, dim]);
    let s2 = sums
        .clone()
        .matmul(g.clone())
        .mul(sums)
        .sum_dim(1)
        .reshape([batch]);
    let rep = z2.mul_scalar(2.0 * slots as f32).sub(s2.mul_scalar(2.0));
    att.add(rep.mul_scalar(lambda)).mean().into_scalar()
}

/// One step of gradient descent on the energy, in the metric:
/// `z <- z - 2 eta G ((z - c) + 2 lambda (K z - sum(z)))`. The energy is a
/// positive-definite quadratic, so with `eta` from [`lipschitz_step`] the
/// descent never increases it.
///
/// The repulsion coefficient is `2 lambda` because
/// `d/dz_k sum_{k != l} ||z_k - z_l||^2_G = 4 G (K z_k - sum_l z_l)`: the
/// ordered pair sum counts each unordered pair twice. The factor of 2 is the
/// whole difference between descending this energy and descending a *different*
/// one, so it is also the difference between monotone descent and a step that
/// can raise the energy it claims to be lowering.
pub(crate) fn energy_grad_step<B: Backend>(
    g: &Tensor<B, 2>,
    z: &Tensor<B, 3>,
    context: &Tensor<B, 3>,
    lambda: f32,
    slots: usize,
    eta: f32,
) -> Tensor<B, 3> {
    let sums = z.clone().sum_dim(1);
    // `2 * lambda`, not `lambda`: see the derivative on this function. The
    // ordered-pair repulsion sum differentiates to `4 G (K z_k - sum(z))`.
    let rep = z
        .clone()
        .mul_scalar(slots as f32)
        .sub(sums)
        .mul_scalar(2.0 * lambda);
    let total = z.clone().sub(context.clone()).add(rep);
    let grad = matmul_g3(total, g);
    z.clone().sub(grad.mul_scalar(2.0 * eta))
}

/// What one block's refinement did, per step.
#[derive(Debug, Clone)]
pub struct RefineTrace {
    /// Mean energy after each refinement step.
    pub energy: Vec<f32>,
    /// Mean squared displacement of each refinement step.
    pub displacement: Vec<f32>,
}

/// The declared answer columns of `logits [batch, vocab]`, in the order the
/// answer set lists them: `[batch, |set|]`. This is the *only* place the
/// vocabulary is narrowed, so the loss, the prediction and the reported
/// accuracy all read the same restricted set and cannot disagree about what
/// counts as an answer.
pub(crate) fn gather_answer_columns<B: Backend>(
    logits: &Tensor<B, 2>,
    set: &[u16],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch = logits.dims()[0];
    let mut columns = Vec::with_capacity(set.len());
    for &token in set {
        let index = Tensor::<B, 2, Int>::ones([batch, 1], device).mul_scalar(i64::from(token));
        columns.push(logits.clone().gather(1, index));
    }
    Tensor::cat(columns, 1)
}

/// The cross-entropy on the answer token: the loss tensor (for the
/// optimizer), its value, and the exact-match accuracy, all read over the
/// declared answer set (empty = the full vocabulary). Shared by the
/// [`GeometricReasoner`] and the matched-depth baseline
/// ([`crate::geombaseline::GeomBaseline`]) so the comparison is scored by
/// literally the same metric.
///
/// Restricting is not a trick to make the number look small: a scene's
/// question has a closed answer space, so the bytes outside the set are not
/// wrong answers, they are not answers. Scoring them lets the readout buy
/// loss reduction from suppressing bytes that never compete, and -- worse
/// for the reported number -- lets `argmax` land on a non-answer byte and
/// score a geometrically correct reply as wrong. With no declared set this
/// is the identity on the full vocabulary.
pub(crate) fn answer_ce<B: Backend<FloatElem = f32>>(
    logits: &Tensor<B, 2>,
    answers: &Tensor<B, 1, Int>,
    answer_tokens: &[u16],
    vocab_size: usize,
) -> (Tensor<B, 1>, f32, f32) {
    let device = logits.device();
    let scored = match answer_tokens.is_empty() {
        false => gather_answer_columns(logits, answer_tokens, &device),
        true => logits.clone(),
    };
    let set_ids: Tensor<B, 1, Int> = match answer_tokens.is_empty() {
        false => Tensor::<B, 1, Int>::from_ints(
            answer_tokens
                .iter()
                .map(|t| i64::from(*t))
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        ),
        true => Tensor::<B, 1, Int>::arange(0..vocab_size as i64, &device),
    };
    let log_probs = log_softmax(scored.clone(), 1);
    let classes = set_ids.clone();
    let mask = answers
        .clone()
        .unsqueeze_dim::<2>(1)
        .equal(classes.unsqueeze_dim::<2>(0))
        .float();
    let picked = log_probs
        .clone()
        .mul(mask)
        .sum_dim(1)
        .reshape([log_probs.dims()[0]]);
    let ce = picked.mean().neg();
    let ce_value = ce.clone().detach().into_scalar();
    let set_len = set_ids.dims()[0];
    // `argmax` keeps the reduced dim in this Burn version ([b, 1]);
    // `reshape` (not `unsqueeze_dim`) normalizes either convention to the
    // `[batch, 1]` index tensor `gather` needs, so this holds whether a
    // future backend reduces the dim or not.
    let pred = set_ids
        .clone()
        .unsqueeze_dim::<2>(0)
        .expand([logits.dims()[0], set_len])
        .gather(1, scored.argmax(1).reshape([logits.dims()[0], 1]))
        .reshape([logits.dims()[0]]);
    let accuracy = pred
        .reshape([logits.dims()[0]])
        .equal(answers.clone())
        .float()
        .mean()
        .into_scalar();
    (ce, ce_value, accuracy)
}

/// One refinement block: geodesic attention under the block's own learned
/// metric, a certified energy relaxation of the geometric stream, and the
/// experience-into-structure write back into the symbolic stream. The block's
/// landscape is fixed (computed once from its input); the next block redraws
/// it from the refined symbolic stream -- the recursive coupling of the two
/// descriptions.
#[derive(Module, Debug)]
pub struct GeomBlock<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    /// The read from the geometric stream into the symbolic stream.
    writer_in: Linear<B>,
    writer_out: Linear<B>,
    norm_tokens: LayerNorm<B>,
    /// The sigma conditioning (FiLM at block entry): each block knows where
    /// it sits on the trajectory, which is what makes it a denoiser.
    sigma_embedder: crate::vit::TimestepEmbedder<B>,
    film_scale: Linear<B>,
    film_shift: Linear<B>,
    metric: MetricField<B>,
    /// Raw repulsion logit; the coefficient is `softplus(raw) > 0`.
    repulsion: Param<Tensor<B, 1>>,
    /// Raw attention-temperature logit; the temperature is
    /// [`bounded_temperature`] of it (floored and capped).
    attn_tau: Param<Tensor<B, 1>>,
    slots: usize,
    refine_steps: usize,
}

impl<B: Backend<FloatElem = f32>> GeomBlock<B> {
    pub fn new(config: &GeomConfig, device: &B::Device) -> Self {
        let raw_repulsion = ((config.repulsion as f32).exp() - 1.0).ln();
        Self {
            q_proj: LinearConfig::new(config.dim, config.dim)
                .with_bias(true)
                .init(device),
            k_proj: LinearConfig::new(config.hidden_size, config.dim)
                .with_bias(true)
                .init(device),
            v_proj: LinearConfig::new(config.hidden_size, config.dim)
                .with_bias(true)
                .init(device),
            writer_in: LinearConfig::new(config.dim, config.hidden_size)
                .with_bias(true)
                .init(device),
            writer_out: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            norm_tokens: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            sigma_embedder: crate::vit::TimestepEmbedder::new(
                config.cond_hidden_size,
                config.frequency_embedding_size,
                device,
            ),
            film_scale: LinearConfig::new(config.cond_hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            film_shift: LinearConfig::new(config.cond_hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            metric: MetricField::new(config.dim, device),
            repulsion: Param::from_tensor(
                Tensor::<B, 1>::ones([1], device).mul_scalar(raw_repulsion),
            ),
            attn_tau: Param::from_tensor(Tensor::<B, 1>::zeros([1], device)),
            slots: config.slots,
            refine_steps: config.refine_steps,
        }
    }

    /// Refine `(h, z)` for `refine_steps` iterations at noise scale `sigma`.
    ///
    /// The landscape is fixed: the context each slot is drawn toward, the
    /// keys and values it reads, and the step size are all computed once from
    /// the block's input. The relaxation then descends the energy to a fixed
    /// point; the symbolic stream is rewritten from the refined state at
    /// every step.
    pub fn refine(
        &self,
        h: Tensor<B, 3>,
        z_in: Tensor<B, 3>,
        sigma: f64,
        refine_override: Option<usize>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, RefineTrace) {
        let device = h.device();
        let [batch, _, _] = h.dims();
        let steps = refine_override.unwrap_or(self.refine_steps);

        let sigma_t = Tensor::<B, 1>::ones([batch], &device).mul_scalar(sigma as f32);
        let cond = crate::vit::silu_public(self.sigma_embedder.forward(sigma_t));
        let gamma = self.film_scale.forward(cond.clone());
        let beta = self.film_shift.forward(cond);
        let mut h = h
            .mul(gamma.add_scalar(1.0).unsqueeze_dim::<3>(1))
            .add(beta.unsqueeze_dim::<3>(1));

        let hn = self.norm_tokens.forward(h.clone());
        let keys = self.k_proj.forward(hn.clone());
        let values = self.v_proj.forward(hn);
        let queries = self.q_proj.forward(z_in.clone());
        let dim = queries.dims()[2];
        let d2 = self.metric.sq_dist(queries, keys.clone());

        let tau = bounded_temperature(self.attn_tau.val().into_scalar(), dim);
        let lambda = (1.0 + self.repulsion.val().into_scalar().exp()).ln();
        let g = self.metric.metric();
        let eta = lipschitz_step(&g, lambda, self.slots);

        let context = softmax(d2.clone().mul_scalar(-1.0 / tau), 2).matmul(values);

        let mut z = z_in;
        let mut energy_trace = Vec::with_capacity(steps);
        let mut displacement_trace = Vec::with_capacity(steps);
        for _ in 0..steps {
            let z_old = z.clone();
            z = energy_grad_step(&g, &z, &context, lambda, self.slots, eta);
            let d2_refined = self.metric.sq_dist(z.clone(), keys.clone());
            let read_weights = softmax(d2_refined.swap_dims(1, 2).mul_scalar(-1.0 / tau), 2);
            let read = read_weights.matmul(self.writer_in.forward(z.clone()));
            h = h.add(self.writer_out.forward(read));
            let z_detached = z.clone().detach();
            let context_detached = context.clone().detach();
            energy_trace.push(energy_value(
                &g,
                &z_detached,
                &context_detached,
                lambda,
                self.slots,
            ));
            let dz = z_detached.sub(z_old.detach());
            displacement_trace.push(dz.clone().mul(dz).mean().into_scalar());
        }
        (
            h,
            z,
            RefineTrace {
                energy: energy_trace,
                displacement: displacement_trace,
            },
        )
    }
}

/// The router diagnostics: the mosme balance breakdown (the weighted balance
/// loss in `total`, the z-loss reported unweighted in `z_loss`), the load and
/// per-token entropies (normalized by the box count), and the per-box load
/// shares.
#[derive(Debug, Clone)]
pub struct RoutingInfo<B: Backend> {
    /// The hierarchical balance breakdown, straight from
    /// [`crate::mosme::HierarchicalGates::balance_loss`].
    pub balance: BalanceBreakdown<B>,
    /// The configured z-loss weight. `balance.z_loss` is added at this level,
    /// never at the caller's `balance_weight`.
    pub z_level: f64,
    pub load_entropy: f32,
    pub token_entropy: f32,
    pub box_load: Vec<f32>,
}

/// The MoSME readout: boxes of specialized expert heads under the crate's
/// two-level [`HierarchicalRouter`]. Each box is one failure mode's
/// representation; the box router picks the kinds, each box's expert router
/// picks the specialist head. Gates are softmaxes over the *selected* logits,
/// so gradients flow through exactly the routed path.
#[derive(Module, Debug)]
pub struct GeomRouter<B: Backend> {
    router: HierarchicalRouter<B>,
    experts: Vec<Vec<Linear<B>>>,
    balance: BalanceWeights,
    vocab_size: usize,
    n_boxes: usize,
}

impl<B: Backend<FloatElem = f32>> GeomRouter<B> {
    pub fn new(
        hidden_size: usize,
        vocab_size: usize,
        n_boxes: usize,
        n_experts: usize,
        top_k: usize,
        device: &B::Device,
    ) -> Self {
        // Checked here, where the mistake is, because `forward` relies on it:
        // a router with no heads has no mixture to return, and the message a
        // caller sees from a forward pass would not name the argument at fault.
        assert!(
            n_boxes >= 1 && n_experts >= 1,
            "a mixture needs at least one box and one expert, not {n_boxes} and {n_experts}"
        );
        assert!(
            top_k >= 1 && top_k <= n_experts.min(n_boxes),
            "top-k must be in [1, {}], not {top_k}",
            n_experts.min(n_boxes)
        );
        // The readout routes on the query state alone: `cond_size` is the
        // hidden size and `route_on_tokens` is off, so the router input is the
        // query unchanged. The same `top_k` applies at both levels, as the
        // hand-rolled router did.
        let spec = MosmeSpec {
            boxes: (0..n_boxes)
                .map(|i| {
                    BoxSpec::new(
                        format!("kind_{i}"),
                        format!("question kind {i}"),
                        (0..n_experts)
                            .map(|j| {
                                ExpertSpec::new(
                                    format!("kind_{i}/expert_{j}"),
                                    format!("kind {i} expert {j}"),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
            top_box: top_k,
            top_expert: top_k,
            route_on_tokens: false,
            balance: BalanceWeights::default(),
        };
        let balance = spec.balance;
        let config = MosmeConfig::new(hidden_size, hidden_size, spec);
        Self {
            router: HierarchicalRouter::new(&config, device),
            experts: (0..n_boxes)
                .map(|_| {
                    (0..n_experts)
                        .map(|_| {
                            LinearConfig::new(hidden_size, vocab_size)
                                .with_initializer(Initializer::Normal {
                                    mean: 0.0,
                                    std: 0.02,
                                })
                                .with_bias(true)
                                .init(device)
                        })
                        .collect()
                })
                .collect(),
            balance,
            vocab_size,
            n_boxes,
        }
    }

    /// One box's chosen expert head, unrouted -- the identity certificate's
    /// comparison target.
    pub fn expert_head(&self, query: Tensor<B, 2>, box_i: usize, expert_j: usize) -> Tensor<B, 2> {
        self.experts[box_i][expert_j].forward(query)
    }

    /// The inner two-level router, for callers that need the raw gates.
    pub fn router(&self) -> &HierarchicalRouter<B> {
        &self.router
    }

    /// Mutable access to the inner router, e.g. to enable or disable an
    /// expert (`HierarchicalRouter::set_enabled`).
    pub fn router_mut(&mut self) -> &mut HierarchicalRouter<B> {
        &mut self.router
    }

    /// Fails if the mixture has no heads, which `new` already rejects: a
    /// router that reached `forward` empty would otherwise report an empty
    /// mixture as a zero logit vector, which reads as a trained model that
    /// simply has nothing to say.
    pub fn forward(&self, query: Tensor<B, 2>) -> anyhow::Result<(Tensor<B, 2>, RoutingInfo<B>)> {
        let input = self.router.router_input_2d(&query, &query);
        let gates = self.router.route(input);

        // Dense evaluation over every head, weighted by the composed gates
        // `box_gates[:, i] * expert_gates[i][:, j]` -- a partition of unity,
        // so the mixture is a convex combination of head outputs.
        //
        // Seeded from the first head rather than accumulated through an
        // `Option`, which would have needed unwrapping at the end. The
        // emptiness is checked rather than assumed, because `new`'s assertion is
        // compiled out of a release build.
        let composed = gates.composed();
        let heads: Vec<_> = composed
            .iter()
            .zip(self.experts.iter())
            .flat_map(|(box_gates, experts)| {
                experts
                    .iter()
                    .enumerate()
                    .map(move |(j, head)| (box_gates, j, head))
            })
            .collect();
        anyhow::ensure!(
            !heads.is_empty(),
            "the {}x{} mixture has no heads to evaluate",
            self.n_boxes,
            self.experts.first().map_or(0, Vec::len)
        );
        let (first_gates, first_j, first_head) = heads[0];
        let mut logits = first_head
            .forward(query.clone())
            .mul(first_gates.clone().narrow(1, first_j, 1));
        for (box_gates, j, head) in &heads[1..] {
            logits = logits.add(
                head.forward(query.clone())
                    .mul((*box_gates).clone().narrow(1, *j, 1)),
            );
        }
        debug_assert_eq!(
            logits.dims()[1],
            self.vocab_size,
            "a head is hidden -> vocab"
        );

        let balance = gates.balance_loss(self.balance);

        // diagnostics: normalized load and per-token entropies over the boxes
        let ln_boxes = (self.n_boxes as f32).ln().max(1e-6);
        let traffic = gates.box_traffic().reshape([self.n_boxes]);
        let pl = traffic.clone().clamp_min(1e-30).log();
        let load_entropy = traffic.clone().mul(pl).sum().neg().into_scalar() / ln_boxes;
        let token_entropy = crate::moe::entropy_of_rows(&gates.box_gates).into_scalar() / ln_boxes;
        let load: Vec<f32> = traffic.into_data().convert::<f32>().iter::<f32>().collect();
        Ok((
            logits,
            RoutingInfo {
                balance,
                z_level: self.balance.z_level,
                load_entropy,
                token_entropy,
                box_load: load,
            },
        ))
    }
}

/// Output of a full pass over the refinement trajectory.
#[derive(Debug, Clone)]
pub struct GeomForward<B: Backend> {
    /// `[batch, vocab]` answer logits at the query position.
    pub logits: Tensor<B, 2>,
    /// The final relaxed state `[batch, slots, dim]` -- the attractor state.
    pub final_state: Tensor<B, 3>,
    /// The last block's final energy (mean over the batch).
    pub energy: f32,
    /// The last block's final squared displacement (mean over the batch).
    pub displacement: f32,
    /// Router diagnostics, when the MoSME readout is enabled.
    pub routing: Option<RoutingInfo<B>>,
}

/// What one standalone block-diffusion step produced.
#[derive(Debug, Clone)]
pub struct GeomBlockStep<B: Backend> {
    pub loss: Tensor<B, 1>,
    /// The EDM-preconditioned x0 estimate `[batch, slots, dim]`.
    pub x0: Tensor<B, 3>,
    pub mse: f32,
    pub ce: f32,
    pub accuracy: f32,
    pub balance: f32,
    pub load_entropy: f32,
    pub token_entropy: f32,
    pub energy: f32,
    pub displacement: f32,
    pub block: usize,
    pub sigma: f64,
}

/// What one training step produced.
#[derive(Debug, Clone, Copy)]
pub struct GeomStepMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub energy: f32,
    pub displacement: f32,
    pub consistency: f32,
    pub balance: f32,
    pub load_entropy: f32,
    pub token_entropy: f32,
    pub scenes: usize,
}

/// A dual-stream geometric reasoner: a symbolic stream (discrete tokens) and
/// a geometric stream (concept points in a learned Riemannian space), coupled
/// by geodesic attention and refined into an attractor basin block by block.
/// No position table: the scenes are fixed-width and the geometry carries the
/// structure. Generic over the backend, so the same model trains and answers
/// on CPU and on a GPU.
#[derive(Module, Debug)]
pub struct GeometricReasoner<B: Backend> {
    token_embedding: Embedding<B>,
    /// Per-slot projection of the chunk-pooled tokens: slot `k` starts near
    /// the scene region it was pooled from.
    slot_init: Linear<B>,
    blocks: Vec<GeomBlock<B>>,
    final_norm: LayerNorm<B>,
    readout: Linear<B>,
    router: Option<GeomRouter<B>>,
    hidden_size: usize,
    scene_len: usize,
    slots: usize,
    dim: usize,
    num_blocks: usize,
    vocab_size: usize,
    sigma_data: f64,
    /// The declared answer set, or `None` to score the whole vocabulary.
    /// Not a parameter: it is part of the task, not something training learns.
    answer_tokens: Option<Vec<u16>>,
}

impl<B: Backend<FloatElem = f32>> GeometricReasoner<B> {
    /// Build a reasoner from `config`.
    ///
    /// Fallsible because [`GeomConfig::validate`] is: an invalid geometry is a
    /// caller mistake that has to name itself, and a `new` that panicked on bad
    /// input would take a training run down instead of reporting which field was
    /// out of range.
    pub fn new(config: &GeomConfig, device: &B::Device) -> anyhow::Result<Self> {
        config.validate()?;
        Ok(Self {
            // The same initializer choice the LM trunk makes: a read head
            // over an N(0, 1) embedding would start confidently wrong.
            token_embedding: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: config.initializer_range,
                })
                .init(device),
            slot_init: LinearConfig::new(config.hidden_size, config.dim)
                .with_bias(true)
                .init(device),
            blocks: (0..config.num_blocks)
                .map(|_| GeomBlock::new(config, device))
                .collect(),
            final_norm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            readout: LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: config.initializer_range,
                })
                .with_bias(true)
                .init(device),
            router: (config.moe_boxes > 0).then(|| {
                GeomRouter::new(
                    config.hidden_size,
                    config.vocab_size,
                    config.moe_boxes,
                    config.moe_experts,
                    config.moe_top_k,
                    device,
                )
            }),
            hidden_size: config.hidden_size,
            scene_len: config.scene_len,
            slots: config.slots,
            dim: config.dim,
            num_blocks: config.num_blocks,
            vocab_size: config.vocab_size,
            sigma_data: config.sigma_data,
            answer_tokens: (!config.answer_tokens.is_empty()).then(|| config.answer_tokens.clone()),
        })
    }

    /// The declared answer set this model scores over, if any.
    pub fn answer_tokens(&self) -> Option<&[u16]> {
        self.answer_tokens.as_deref()
    }

    pub fn scene_len(&self) -> usize {
        self.scene_len
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Scene tokens `[batch, scene_len - 1]` to hidden states. No position
    /// table: the fixed scene layout is the position system, and the
    /// geometry carries the structure.
    fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.token_embedding.forward(tokens)
    }

    /// The initial geometric state: the scene's tokens pooled into `slots`
    /// contiguous chunks, each projected into the geometric space, so slot
    /// `k` starts near the region it was pooled from.
    fn init_slots(&self, h: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n, _] = h.dims();
        let slots = self.slots;
        let mut pooled = Vec::with_capacity(slots);
        for k in 0..slots {
            let start = k * n / slots;
            let end = (k + 1) * n / slots;
            let chunk = h.clone().narrow(1, start, end - start);
            pooled.push(chunk.mean_dim(1));
        }
        let pooled = Tensor::cat(pooled, 1);
        self.slot_init.forward(pooled)
    }

    /// The nominal sigma each block operates at: its window's upper edge,
    /// descending from `sigma_max` (block 0 noisiest).
    pub fn nominal_sigmas(&self) -> Vec<f64> {
        let bounds = crate::sigma::block_sigmas(self.num_blocks);
        (0..self.num_blocks)
            .map(|idx| crate::sigma::block_window(&bounds, idx).1)
            .collect()
    }

    /// Answer logits at the query position (the last token the model reads:
    /// the `?` before the answer). With the MoSME mixture enabled the query
    /// state routes through boxes of specialized expert heads; otherwise the
    /// plain linear readout.
    fn answer_logits(
        &self,
        h: Tensor<B, 3>,
    ) -> anyhow::Result<(Tensor<B, 2>, Option<RoutingInfo<B>>)> {
        let [batch, n, _] = h.dims();
        let query_state = self
            .final_norm
            .forward(h.narrow(1, n - 1, 1))
            .reshape([batch, self.hidden_size]);
        match &self.router {
            Some(router) => {
                let (logits, info) = router.forward(query_state)?;
                Ok((logits, Some(info)))
            }
            None => Ok((
                self.readout
                    .forward(query_state.unsqueeze_dim::<3>(1))
                    .reshape([batch, self.vocab_size]),
                None,
            )),
        }
    }

    fn run_blocks(
        &self,
        h0: Tensor<B, 3>,
        z0: Tensor<B, 3>,
        refine_override: Option<usize>,
    ) -> anyhow::Result<(Tensor<B, 3>, Tensor<B, 3>, f32, f32)> {
        let sigmas = self.nominal_sigmas();
        let mut h = h0;
        let mut z = z0;
        let mut energy = 0.0f32;
        let mut displacement = 0.0f32;
        for (idx, block) in self.blocks.iter().enumerate() {
            let (h_next, z_next, trace) = block.refine(h, z, sigmas[idx], refine_override);
            energy = trace.energy.last().copied().unwrap_or(0.0);
            displacement = trace.displacement.last().copied().unwrap_or(0.0);
            h = h_next;
            z = z_next;
        }
        Ok((h, z, energy, displacement))
    }

    /// The clean reasoning path: refine from the token-pooled init at noise
    /// scale zero and read the answer from the query position.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> anyhow::Result<GeomForward<B>> {
        self.forward_depth(tokens, None)
    }

    /// [`Self::forward`] with the refinement depth overridden -- the
    /// test-time-compute knob: the same weights, more or fewer relaxation
    /// steps.
    pub fn forward_depth(
        &self,
        tokens: Tensor<B, 2, Int>,
        refine_override: Option<usize>,
    ) -> anyhow::Result<GeomForward<B>> {
        let h0 = self.embed(tokens);
        let z0 = self.init_slots(&h0);
        let (h, z, energy, displacement) = self.run_blocks(h0, z0, refine_override)?;
        let (logits, routing) = self.answer_logits(h)?;
        Ok(GeomForward {
            logits,
            final_state: z,
            energy,
            displacement,
            routing,
        })
    }

    /// A contiguous span of blocks only -- blocks before the span run
    /// detached, which is gradient routing (mirrors `forward_span` /
    /// `denoise_span` on the trunk paths). A span covering every block
    /// reproduces [`Self::forward`] bit for bit.
    pub fn forward_span(
        &self,
        tokens: Tensor<B, 2, Int>,
        span: Range<usize>,
    ) -> anyhow::Result<GeomForward<B>> {
        assert!(span.end <= self.num_blocks, "block span out of range");
        let h0 = self.embed(tokens);
        let z0 = self.init_slots(&h0);
        let sigmas = self.nominal_sigmas();
        let mut h = h0;
        let mut z = z0;
        for (block, sigma) in self.blocks[..span.start].iter().zip(&sigmas[..span.start]) {
            let (h_next, z_next, _) = block.refine(h, z, *sigma, None);
            h = h_next.detach();
            z = z_next.detach();
        }
        let mut energy = 0.0f32;
        let mut displacement = 0.0f32;
        for (block, sigma) in self.blocks[span.clone()].iter().zip(&sigmas[span]) {
            let (h_next, z_next, trace) = block.refine(h, z, *sigma, None);
            energy = trace.energy.last().copied().unwrap_or(0.0);
            displacement = trace.displacement.last().copied().unwrap_or(0.0);
            h = h_next;
            z = z_next;
        }
        let (logits, routing) = self.answer_logits(h)?;
        Ok(GeomForward {
            logits,
            final_state: z,
            energy,
            displacement,
            routing,
        })
    }

    /// Reasoning as denoising: the trajectory starts from pure noise at
    /// `sigma_max` and the blocks relax it into the scene's basin. Requires a
    /// checkpoint trained with the diffusion objective -- the clean path's
    /// blocks never saw noise.
    pub fn forward_diffusion_start(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> anyhow::Result<GeomForward<B>> {
        let device = tokens.device();
        let batch = tokens.dims()[0];
        let h0 = self.embed(tokens);
        let noise = Tensor::<B, 3>::random(
            [batch, self.slots, self.dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z0 = noise.mul_scalar(crate::sigma::SIGMA_MAX as f32);
        let (h, z, energy, displacement) = self.run_blocks(h0, z0, None)?;
        let (logits, routing) = self.answer_logits(h)?;
        Ok(GeomForward {
            logits,
            final_state: z,
            energy,
            displacement,
            routing,
        })
    }

    /// The cross-entropy on the answer token: the loss tensor (for the
    /// optimizer), its value, and the exact-match accuracy.
    fn answer_ce(
        &self,
        logits: &Tensor<B, 2>,
        answers: &Tensor<B, 1, Int>,
    ) -> (Tensor<B, 1>, f32, f32) {
        answer_ce(
            logits,
            answers,
            self.answer_tokens.as_deref().unwrap_or(&[]),
            self.vocab_size,
        )
    }

    /// The clean answer objective: cross-entropy on the answer token, read
    /// from the query position of the relaxed state, plus the router's
    /// balance charge at `balance_weight`. The router z-loss is added at its
    /// own `z_level`, never at `balance_weight` (see [`crate::vit::RouterAux`]):
    /// it is a numerical stabilizer, not a routing regularizer.
    pub fn answer_step(
        &self,
        tokens_full: Tensor<B, 2, Int>,
        balance_weight: f64,
    ) -> anyhow::Result<(Tensor<B, 1>, GeomStepMetrics)> {
        let [batch, len] = tokens_full.dims();
        assert_eq!(
            len, self.scene_len,
            "a scene is {} tokens, got {}",
            self.scene_len, len
        );
        let tokens = tokens_full.clone().narrow(1, 0, self.scene_len - 1);
        let answers = tokens_full
            .narrow(1, self.scene_len - 1, 1)
            .reshape([batch]);
        let out = self.forward(tokens)?;
        let (ce, ce_value, accuracy) = self.answer_ce(&out.logits, &answers);
        let mut loss = ce;
        let (mut balance_value, mut load_entropy, mut token_entropy) = (0.0, 0.0, 0.0);
        if let Some(info) = &out.routing {
            balance_value = info.balance.total.clone().detach().into_scalar();
            load_entropy = info.load_entropy;
            token_entropy = info.token_entropy;
            loss = loss
                .add(info.balance.total.clone().mul_scalar(balance_weight as f32))
                .add(info.balance.z_loss.clone().mul_scalar(info.z_level as f32));
        }
        Ok((
            loss,
            GeomStepMetrics {
                loss: ce_value,
                accuracy,
                energy: out.energy,
                displacement: out.displacement,
                consistency: 0.0,
                balance: balance_value,
                load_entropy,
                token_entropy,
                scenes: batch,
            },
        ))
    }

    /// Predicted answer bytes for `logits` `[batch, vocab]`, read over the
    /// declared answer set when one exists (the same restriction the answer
    /// cross-entropy trains under): the argmax index mapped back
    /// through the set, so position `i` holds a byte, not a column number.
    /// Without a declared set the columns already are the bytes.
    pub fn predict_answers(&self, logits: &Tensor<B, 2>) -> Vec<u16> {
        let device = logits.device();
        let batch = logits.dims()[0];
        let (scored, set_vec): (Tensor<B, 2>, Vec<i64>) = match &self.answer_tokens {
            Some(set) => (
                gather_answer_columns(logits, set, &device),
                set.iter().map(|t| i64::from(*t)).collect(),
            ),
            None => (logits.clone(), (0..self.vocab_size as i64).collect()),
        };
        // `reshape`, not `unsqueeze_dim`: correct whether this Burn version's
        // `argmax` keeps the reduced dim or not (see `answer_ce`).
        let idx: Vec<i64> = scored
            .argmax(1)
            .reshape([batch, 1])
            .into_data()
            .convert::<i64>()
            .iter::<i64>()
            .collect();
        idx.into_iter()
            .map(|i| set_vec[i as usize] as u16)
            .collect()
    }

    /// The clean attractor state the diffusion objective denoises toward:
    /// the full clean path's final state, detached.
    pub fn clean_state(
        &self,
        tokens_full: Tensor<B, 2, Int>,
    ) -> anyhow::Result<(Tensor<B, 3>, GeomForward<B>)> {
        let tokens = tokens_full.narrow(1, 0, self.scene_len - 1);
        let out = self.forward(tokens)?;
        Ok((out.final_state.clone().detach(), out))
    }

    /// One standalone block-diffusion step (the framework's core): block
    /// `block_idx` is trained on the clean state corrupted at its window's
    /// noise scale, EDM-preconditioned, with the sigma-weighted answer CE.
    /// Blocks are trained standalone on data-side corruption -- not on the
    /// previous block's output -- which is what makes the objective
    /// embarrassingly parallel per block.
    pub fn block_step(
        &self,
        tokens_full: Tensor<B, 2, Int>,
        block_idx: usize,
        sigma: f64,
        z_star: &Tensor<B, 3>,
        balance_weight: f64,
    ) -> anyhow::Result<GeomBlockStep<B>> {
        assert!(
            block_idx < self.num_blocks,
            "block {block_idx} out of range"
        );
        let [batch, len] = tokens_full.dims();
        assert_eq!(
            len, self.scene_len,
            "a scene is {} tokens, got {}",
            self.scene_len, len
        );
        let tokens = tokens_full.clone().narrow(1, 0, self.scene_len - 1);
        let answers = tokens_full
            .narrow(1, self.scene_len - 1, 1)
            .reshape([batch]);
        let device = tokens.device();
        let h0 = self.embed(tokens);
        let noise = Tensor::<B, 3>::random(
            [batch, self.slots, self.dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z_in = z_star.clone().add(noise.mul_scalar(sigma as f32));
        let (h_out, z_out, trace) = self.blocks[block_idx].refine(h0, z_in.clone(), sigma, None);
        let precond = crate::sigma::EdmPreconditioning::new(sigma, self.sigma_data);
        let x0 = z_in
            .mul_scalar(precond.c_skip as f32)
            .add(z_out.mul_scalar(precond.c_out as f32));
        let diff = x0.clone().sub(z_star.clone());
        let mse = diff.clone().mul(diff).mean();
        let mse_value = mse.clone().detach().into_scalar();
        let (logits, routing) = self.answer_logits(h_out)?;
        let (ce, ce_value, accuracy) = self.answer_ce(&logits, &answers);
        let w_ce = ((crate::sigma::SIGMA_MAX - sigma)
            / (crate::sigma::SIGMA_MAX - crate::sigma::SIGMA_MIN)) as f32;
        let w_mse = crate::sigma::edm_loss_weight(sigma, self.sigma_data) as f32;
        let mut loss = ce.mul_scalar(w_ce).add(mse.mul_scalar(w_mse));
        let (mut balance_value, mut load_entropy, mut token_entropy) = (0.0, 0.0, 0.0);
        if let Some(info) = &routing {
            balance_value = info.balance.total.clone().detach().into_scalar();
            load_entropy = info.load_entropy;
            token_entropy = info.token_entropy;
            loss = loss
                .add(info.balance.total.clone().mul_scalar(balance_weight as f32))
                .add(info.balance.z_loss.clone().mul_scalar(info.z_level as f32));
        }
        Ok(GeomBlockStep {
            loss,
            x0,
            mse: mse_value,
            ce: ce_value,
            accuracy,
            balance: balance_value,
            load_entropy,
            token_entropy,
            energy: trace.energy.last().copied().unwrap_or(0.0),
            displacement: trace.displacement.last().copied().unwrap_or(0.0),
            block: block_idx,
            sigma,
        })
    }
}

/// Sample `batch_size` whole scenes, aligned: one positional read per scene,
/// so a corpus larger than memory costs no more per batch than one that fits.
pub fn sample_aligned_batch<R: rand::Rng>(
    corpus: &mut TokenCorpus,
    scene_len: usize,
    batch_size: usize,
    rng: &mut R,
) -> anyhow::Result<Vec<Vec<u16>>> {
    let scenes = corpus.len() / scene_len;
    anyhow::ensure!(
        scenes > 0,
        "the corpus holds {} tokens, fewer than one scene of {}",
        corpus.len(),
        scene_len
    );
    let mut rows = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let scene = rng.random_range(0..scenes);
        rows.push(corpus.window(scene * scene_len, scene_len)?);
    }
    Ok(rows)
}

fn pad2(value: i32) -> (u8, u8) {
    ((b'0' + (value / 10) as u8), (b'0' + (value % 10) as u8))
}

/// One of eight compass bytes, clockwise from north: N=0, NE=1, E=2, SE=3,
/// S=4, SW=5, W=6, NW=7.
fn compass_byte(dx: i32, dy: i32) -> u8 {
    let angle = (dy as f64).atan2(dx as f64).to_degrees();
    let clockwise_from_north = (90.0 - angle).rem_euclid(360.0);
    b'0' + (((clockwise_from_north / 45.0).round() as i32).rem_euclid(8) as u8)
}

/// The index of the nearest (`farthest = false`) or farthest other point, or
/// `None` when the extremum is tied -- the caller retries rather than emit an
/// ambiguous label.
fn extreme_point(coords: &[(i32, i32)], reference: usize, farthest: bool) -> Option<usize> {
    let (px, py) = coords[reference];
    let mut best: Option<(usize, f64)> = None;
    let mut tied = false;
    for (idx, (x, y)) in coords.iter().enumerate() {
        if idx == reference {
            continue;
        }
        let d = (((*x - px) * (*x - px)) + ((*y - py) * (*y - py))) as f64;
        match &best {
            None => best = Some((idx, d)),
            Some((_, best_d)) => {
                if (farthest && d > *best_d) || (!farthest && d < *best_d) {
                    best = Some((idx, d));
                    tied = false;
                } else if d == *best_d {
                    tied = true;
                }
            }
        }
    }
    let (idx, _) = best?;
    (!tied).then_some(idx)
}

/// A reference whose nearest/farthest other point is unique: retried rather
/// than emit an ambiguous label.
pub(crate) fn extreme_query(
    coords: &[(i32, i32)],
    rng: &mut StdRng,
    farthest: bool,
) -> (usize, usize) {
    let points = coords.len();
    let start = rng.random_range(0..points);
    for attempt in 0..points {
        let reference = (start + attempt) % points;
        if let Some(idx) = extreme_point(coords, reference, farthest) {
            return (reference, idx);
        }
    }
    let fallback = (0..points).find(|&i| i != start).unwrap_or(start);
    (start, fallback)
}

/// A collinear triple: axis-aligned half the time, slope +/-1 the other half;
/// both are exact on the grid. Avoids the remaining points.
pub(crate) fn collinear_triple(rng: &mut StdRng, others: &[(i32, i32)]) -> Option<[(i32, i32); 3]> {
    for _ in 0..128 {
        let triple = if rng.random_bool(0.5) {
            let y = rng.random_range(0..=GRID_MAX);
            let mut xs: Vec<i32> = Vec::new();
            while xs.len() < 3 {
                let x = rng.random_range(0..=GRID_MAX);
                if !xs.contains(&x) {
                    xs.push(x);
                }
            }
            xs.sort();
            [(xs[0], y), (xs[1], y), (xs[2], y)]
        } else {
            let dir = if rng.random_bool(0.5) { 1 } else { -1 };
            let x = rng.random_range(0..=GRID_MAX);
            let y = rng.random_range(0..=GRID_MAX);
            let pts: Vec<(i32, i32)> = (-1..=1)
                .map(|k| (x + k, y + k * dir))
                .filter(|(px, py)| (0..=GRID_MAX).contains(px) && (0..=GRID_MAX).contains(py))
                .collect();
            if pts.len() < 3 {
                continue;
            }
            [pts[0], pts[1], pts[2]]
        };
        if triple[0] == triple[1] || triple[1] == triple[2] || triple[0] == triple[2] {
            continue;
        }
        if triple.iter().any(|p| others.contains(p)) {
            continue;
        }
        return Some(triple);
    }
    None
}

/// A non-collinear triple, avoiding the remaining points.
pub(crate) fn non_collinear_triple(
    rng: &mut StdRng,
    others: &[(i32, i32)],
) -> Option<[(i32, i32); 3]> {
    for _ in 0..128 {
        let a = (
            rng.random_range(0..=GRID_MAX),
            rng.random_range(0..=GRID_MAX),
        );
        let b = (
            rng.random_range(0..=GRID_MAX),
            rng.random_range(0..=GRID_MAX),
        );
        let c = (
            rng.random_range(0..=GRID_MAX),
            rng.random_range(0..=GRID_MAX),
        );
        if a == b || b == c || a == c {
            continue;
        }
        let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if cross == 0 {
            continue;
        }
        if [a, b, c].iter().any(|p| others.contains(p)) {
            continue;
        }
        return Some([a, b, c]);
    }
    None
}

/// Is `d` strictly inside the triangle `a b c` (same-side test on all three
/// edges)? Containment is a similarity invariant, which is what the
/// augmentation relies on.
fn strictly_inside(a: (i32, i32), b: (i32, i32), c: (i32, i32), d: (i32, i32)) -> bool {
    let s1 = (b.0 - a.0) * (d.1 - a.1) - (b.1 - a.1) * (d.0 - a.0);
    let s2 = (c.0 - b.0) * (d.1 - b.1) - (c.1 - b.1) * (d.0 - b.0);
    let s3 = (a.0 - c.0) * (d.1 - c.1) - (a.1 - c.1) * (d.0 - c.0);
    (s1 > 0 && s2 > 0 && s3 > 0) || (s1 < 0 && s2 < 0 && s3 < 0)
}

/// A triangle plus a point strictly inside it (`desired = true`) or strictly
/// outside it, avoiding the remaining points.
pub(crate) fn inside_scenes(
    rng: &mut StdRng,
    desired: bool,
    others: &[(i32, i32)],
) -> Option<[(i32, i32); 4]> {
    for _ in 0..128 {
        let a = (
            rng.random_range(0..=GRID_MAX),
            rng.random_range(0..=GRID_MAX),
        );
        let b = (
            rng.random_range(0..=GRID_MAX),
            rng.random_range(0..=GRID_MAX),
        );
        let c = (
            rng.random_range(0..=GRID_MAX),
            rng.random_range(0..=GRID_MAX),
        );
        if a == b || b == c || a == c {
            continue;
        }
        let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if cross == 0 {
            continue;
        }
        let d = (
            rng.random_range(0..=GRID_MAX),
            rng.random_range(0..=GRID_MAX),
        );
        if d == a || d == b || d == c {
            continue;
        }
        if strictly_inside(a, b, c, d) != desired {
            continue;
        }
        if [a, b, c, d].iter().any(|p| others.contains(p)) {
            continue;
        }
        return Some([a, b, c, d]);
    }
    None
}

/// A scene's answer recomputed from its coordinates and reference indices --
/// the single label logic that generation, augmentation and the certificates
/// all share. `None` when the coordinates make the question ambiguous (a
/// tied extremum): generation retries those, augmentation rejects them.
///
/// Label semantics per kind, and why a similarity transform preserves them:
/// nearest/farthest are arg-min/arg-max of squared distances (distances
/// scale uniformly), inside is a sign test (containment), collinear is an
/// exact cross product. `direction` is the exception: its answer is an
/// *absolute* compass bearing (`compass_byte` bins the angle of `b - a`
/// clockwise from north), so a rotation would rotate the label -- direction
/// scenes are augmented with translation + uniform scale only.
/// [`answer_for`] for a caller that has already established the kind is known
/// and the references are in range -- so a `None` from `answer_for` means the
/// kind is unknown, not that the answer is ambiguous.
///
/// This exists because "no answer" and "answer is `n`" are not the same thing.
/// Substituting `b'n'` for an unknown kind would put a confident wrong label in
/// the corpus and train the model on it.
fn true_answer(kind: &str, coords: &[(i32, i32)], refs: &[usize]) -> anyhow::Result<u8> {
    answer_for(kind, coords, refs).ok_or_else(|| {
        anyhow::anyhow!("kind '{kind}' has no answer over {} references", refs.len())
    })
}

/// The single byte that encodes a question kind in a rendered scene.
fn kind_letter(kind: &str) -> anyhow::Result<u8> {
    let letter = match kind {
        "nearest" => b'n',
        "farthest" => b'f',
        "direction" => b'd',
        "inside" => b'i',
        "collinear" => b'c',
        other => {
            anyhow::bail!(
                "unknown question kind '{other}'; the kinds are {}",
                KINDS.join("|")
            )
        }
    };
    Ok(letter)
}

pub fn answer_for(kind: &str, coords: &[(i32, i32)], refs: &[usize]) -> Option<u8> {
    match kind {
        "nearest" => Some(b'A' + extreme_point(coords, refs[0], false)? as u8),
        "farthest" => Some(b'A' + extreme_point(coords, refs[0], true)? as u8),
        "direction" => {
            let dx = coords[refs[1]].0 - coords[refs[0]].0;
            let dy = coords[refs[1]].1 - coords[refs[0]].1;
            Some(compass_byte(dx, dy))
        }
        "inside" => Some(
            if strictly_inside(
                coords[refs[0]],
                coords[refs[1]],
                coords[refs[2]],
                coords[refs[3]],
            ) {
                b'y'
            } else {
                b'n'
            },
        ),
        "collinear" => {
            let (a, b, c) = (coords[refs[0]], coords[refs[1]], coords[refs[2]]);
            let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
            Some(if cross == 0 { b'y' } else { b'n' })
        }
        _ => None,
    }
}

/// A similarity transform of the plane: rotate by `theta`, scale uniformly
/// by `scale` (> 0), translate by `(tx, ty)`. Similarities preserve relative
/// distances, angles, collinearity and containment -- the invariants every
/// question kind's label is computed from (direction excepted: see
/// [`answer_for`]).
#[derive(Debug, Clone, Copy)]
pub struct Similarity {
    pub theta: f64,
    pub scale: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Similarity {
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let (sin, cos) = self.theta.sin_cos();
        (
            self.scale * (cos * x - sin * y) + self.tx,
            self.scale * (sin * x + cos * y) + self.ty,
        )
    }
}

/// One label-preserving augmented copy of a scene's coordinates: draw a
/// random similarity (translation + uniform scale only for `direction`, whose
/// answer is an absolute bearing), apply it, round back onto the grid, and
/// keep the copy only when the rounded coordinates still fit the grid, stay
/// pairwise distinct, and recompute to the *same* answer via [`answer_for`].
/// The recompute-and-reject step is what makes grid rounding safe: a
/// transform that rounds across a label boundary (a near-tie extremum, a
/// compass-bin edge, a near-degenerate cross product) is discarded, not
/// emitted with a stale label. `None` when no drawn transform survived.
pub(crate) fn augment_coords(
    kind: &str,
    coords: &[(i32, i32)],
    refs: &[usize],
    answer: u8,
    rng: &mut StdRng,
) -> Option<(Vec<(i32, i32)>, Similarity)> {
    let rotate = kind != "direction";
    for _ in 0..128 {
        let theta = if rotate {
            rng.random_range(0.0..std::f64::consts::TAU)
        } else {
            0.0
        };
        let scale = rng.random_range(0.6..=1.4);
        let (sin, cos) = theta.sin_cos();
        let raw: Vec<(f64, f64)> = coords
            .iter()
            .map(|&(x, y)| {
                (
                    scale * (cos * x as f64 - sin * y as f64),
                    scale * (sin * x as f64 + cos * y as f64),
                )
            })
            .collect();
        let min_x = raw.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = raw.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = raw.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = raw.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let room_x = GRID_MAX as f64 - (max_x - min_x);
        let room_y = GRID_MAX as f64 - (max_y - min_y);
        if room_x < 0.0 || room_y < 0.0 {
            continue;
        }
        let sim = Similarity {
            theta,
            scale,
            tx: rng.random_range(0.0..=room_x) - min_x,
            ty: rng.random_range(0.0..=room_y) - min_y,
        };
        let transformed: Vec<(i32, i32)> = coords
            .iter()
            .map(|&(x, y)| {
                let (fx, fy) = sim.apply(x as f64, y as f64);
                (fx.round() as i32, fy.round() as i32)
            })
            .collect();
        if transformed
            .iter()
            .any(|&(x, y)| !(0..=GRID_MAX).contains(&x) || !(0..=GRID_MAX).contains(&y))
        {
            continue;
        }
        let mut dedup = transformed.clone();
        dedup.sort_unstable();
        dedup.dedup();
        if dedup.len() != transformed.len() {
            continue;
        }
        if answer_for(kind, &transformed, refs) != Some(answer) {
            continue;
        }
        return Some((transformed, sim));
    }
    None
}

/// Render one scene at the fixed width: header, coordinates, question
/// (kind letter + four reference slots), `?`, answer. Called only after the
/// coordinates are final -- the inside/collinear kinds replace `coords[0..]`
/// with the construction their label was computed from, and the rendered
/// scene is what the question (and the model) refers to.
fn render_scene(coords: &[(i32, i32)], kind_letter: u8, refs: &[usize], answer: u8) -> Vec<u16> {
    let mut scene: Vec<u16> = Vec::with_capacity(10 + 5 * coords.len());
    scene.push(u16::from(b'P'));
    let (hi, lo) = pad2(coords.len() as i32);
    scene.push(u16::from(hi));
    scene.push(u16::from(lo));
    for (idx, (x, y)) in coords.iter().enumerate() {
        scene.push(u16::from(b'A' + idx as u8));
        let (x_hi, x_lo) = pad2(*x);
        let (y_hi, y_lo) = pad2(*y);
        scene.push(u16::from(x_hi));
        scene.push(u16::from(x_lo));
        scene.push(u16::from(y_hi));
        scene.push(u16::from(y_lo));
    }
    scene.push(u16::from(kind_letter));
    for slot in 0..4 {
        scene.push(match refs.get(slot) {
            Some(&idx) => u16::from(b'A' + idx as u8),
            None => u16::from(b'_'),
        });
    }
    scene.push(u16::from(b'?'));
    scene.push(u16::from(answer));
    scene
}

/// Generate the synthetic geometric-question corpus: scenes of uniquely
/// placed points on a fixed grid, one question each, rendered at a fixed
/// width so every document is exactly `10 + 3 * points` tokens and ends in
/// its answer token. Returns the token stream (one contiguous run of scenes)
/// and the metadata sidecar. Kinds cycle round-robin, exactly reproducing
/// every corpus written before kind weights existed.
pub fn generate_corpus(
    points: usize,
    kinds: &[&str],
    count: usize,
    seed: u64,
) -> anyhow::Result<(Vec<u16>, GeomMeta)> {
    generate_corpus_weighted(points, kinds, None, count, seed, 0)
}

/// [`generate_corpus`] with per-kind sampling weights (AlphaGeometry's
/// stratification lesson: hard kinds need oversampling, not equal shares).
/// `None` is the legacy round-robin, bit for bit; `Some` draws each scene's
/// kind independently with probability proportional to its weight, so even
/// uniform weights produce a different (but statistically balanced) corpus
/// than the legacy path -- the RNG stream diverges at the first draw.
///
/// `augment` emits up to that many extra copies of each scene, transformed
/// by a random similarity (rotation + translation + uniform scale; direction
/// scenes get translation + scale only -- see [`answer_for`]). `0` is off
/// and leaves the corpus exactly as generated.
pub fn generate_corpus_weighted(
    points: usize,
    kinds: &[&str],
    kind_weights: Option<&[f64]>,
    count: usize,
    seed: u64,
    augment: usize,
) -> anyhow::Result<(Vec<u16>, GeomMeta)> {
    anyhow::ensure!((1..=26).contains(&points), "points must be in [1, 26]");
    anyhow::ensure!(!kinds.is_empty(), "at least one question kind is required");
    for kind in kinds {
        anyhow::ensure!(
            KINDS.contains(kind),
            "unknown question kind '{kind}'; the kinds are {}",
            KINDS.join("|")
        );
    }
    if kinds.contains(&"inside") {
        anyhow::ensure!(points >= 4, "the inside kind needs at least 4 points");
    }
    if kinds.contains(&"collinear") {
        anyhow::ensure!(points >= 3, "the collinear kind needs at least 3 points");
    }
    anyhow::ensure!(count > 0, "count must be positive");
    let cumulative: Option<Vec<f64>> = kind_weights
        .map(|w| {
            anyhow::ensure!(
                w.len() == kinds.len(),
                "kind weights ({} values) must cover every requested kind ({})",
                w.len(),
                kinds.len()
            );
            anyhow::ensure!(
                w.iter().all(|v| v.is_finite() && *v >= 0.0),
                "kind weights must be finite and non-negative"
            );
            let total: f64 = w.iter().sum();
            anyhow::ensure!(total > 0.0, "kind weights must sum above zero");
            let mut cum = Vec::with_capacity(w.len());
            let mut acc = 0.0;
            for v in w {
                acc += v / total;
                cum.push(acc);
            }
            // Pin the last edge: float rounding must not strand a draw.
            if let Some(last) = cum.last_mut() {
                *last = 1.0;
            }
            Ok::<_, anyhow::Error>(cum)
        })
        .transpose()?;
    let scene_len = 10 + 5 * points;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tokens = Vec::with_capacity(count * (augment + 1) * scene_len);
    let mut answer_set = BTreeSet::new();
    let mut kind_counts = vec![0usize; kinds.len()];
    for scene_idx in 0..count {
        let kind_idx = match &cumulative {
            None => scene_idx % kinds.len(),
            Some(cum) => {
                let draw = rng.random_range(0.0..1.0);
                cum.iter()
                    .position(|edge| draw < *edge)
                    .unwrap_or(cum.len() - 1)
            }
        };
        let kind = kinds[kind_idx];
        let mut coords: Vec<(i32, i32)> = Vec::with_capacity(points);
        while coords.len() < points {
            let candidate = (
                rng.random_range(0..=GRID_MAX),
                rng.random_range(0..=GRID_MAX),
            );
            if !coords.contains(&candidate) {
                coords.push(candidate);
            }
        }
        let kind_letter = kind_letter(kind)?;
        let (refs, answer): (Vec<usize>, u8) = match kind {
            "nearest" => {
                let (reference, idx) = extreme_query(&coords, &mut rng, false);
                (vec![reference], b'A' + idx as u8)
            }
            "farthest" => {
                let (reference, idx) = extreme_query(&coords, &mut rng, true);
                (vec![reference], b'A' + idx as u8)
            }
            "direction" => {
                let a = rng.random_range(0..points);
                let mut b = rng.random_range(0..points);
                while b == a {
                    b = rng.random_range(0..points);
                }
                let dx = coords[b].0 - coords[a].0;
                let dy = coords[b].1 - coords[a].1;
                (vec![a, b], compass_byte(dx, dy))
            }
            "inside" => {
                let desired = scene_idx % 2 == 0;
                match inside_scenes(&mut rng, desired, &coords[4..]) {
                    Some([a, b, c, d]) => {
                        coords[0] = a;
                        coords[1] = b;
                        coords[2] = c;
                        coords[3] = d;
                        (vec![0, 1, 2, 3], if desired { b'y' } else { b'n' })
                    }
                    // The construction failed: keep the random points and
                    // take their true label, never a forced one.
                    None => (vec![0, 1, 2, 3], true_answer(kind, &coords, &[0, 1, 2, 3])?),
                }
            }
            "collinear" => {
                let desired = scene_idx % 2 == 0;
                let arranged = if desired {
                    collinear_triple(&mut rng, &coords[3..])
                } else {
                    non_collinear_triple(&mut rng, &coords[3..])
                };
                match arranged {
                    Some([a, b, c]) => {
                        coords[0] = a;
                        coords[1] = b;
                        coords[2] = c;
                        (vec![0, 1, 2], if desired { b'y' } else { b'n' })
                    }
                    None => (vec![0, 1, 2], true_answer(kind, &coords, &[0, 1, 2])?),
                }
            }
            other => {
                anyhow::bail!("the kinds were validated, so '{other}' should be unreachable here")
            }
        };
        // Render only now that the coordinates are final: the inside and
        // collinear kinds replaced `coords[0..]` with the construction the
        // answer was computed from, and the rendered scene is what the
        // question (and the model) refers to.
        let scene = render_scene(&coords, kind_letter, &refs, answer);
        assert_eq!(
            scene.len(),
            scene_len,
            "the scene rendering must be fixed-width"
        );
        tokens.extend(scene);
        kind_counts[kind_idx] += 1;
        answer_set.insert(u16::from(answer));
        // Similarity-augmented copies: same question and answer (the label
        // is a similarity invariant, verified per copy by `augment_coords`),
        // new coordinates.
        for _ in 0..augment {
            if let Some((aug_coords, _)) = augment_coords(kind, &coords, &refs, answer, &mut rng) {
                let copy = render_scene(&aug_coords, kind_letter, &refs, answer);
                debug_assert_eq!(
                    copy.len(),
                    scene_len,
                    "augmented scenes render fixed-width too"
                );
                tokens.extend(copy);
                kind_counts[kind_idx] += 1;
            }
        }
    }
    let meta = GeomMeta {
        scene_len,
        points,
        kinds: kinds.iter().map(|kind| kind.to_string()).collect(),
        answer_tokens: answer_set.into_iter().collect(),
        scenes: tokens.len() / scene_len,
        seed,
        kind_counts,
        augment,
    };
    Ok((tokens, meta))
}

/// Which objective a run trains (roadmap Phase 33).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeomObjective {
    /// The clean answer objective: cross-entropy on the answer token read
    /// from the query position of the relaxed state.
    #[default]
    Answer,
    /// The block-diffusion objective: per-block EDM denoising of the
    /// attractor state across the sigma windows, boundary consistency at the
    /// shared sigmas, and the sigma-weighted answer CE.
    Diffusion,
}

impl GeomObjective {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        match text {
            "answer" => Ok(Self::Answer),
            "diffusion" => Ok(Self::Diffusion),
            other => Err(anyhow::anyhow!(
                "unknown objective '{other}': answer | diffusion"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Answer => "answer",
            Self::Diffusion => "diffusion",
        }
    }
}

/// Configuration for [`train_geom`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeomTrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// Learning-rate schedule over the run. Warmup then cosine decay by
    /// default: the relaxation's step size is derived from the metric, so the
    /// early steps land where the metric is least settled and a full-size
    /// update there costs more than it buys.
    pub lr_schedule: crate::schedule::LrSchedule,
    /// Global gradient-norm ceiling; `0.0` disables clipping.
    pub clip_norm: f64,
    pub weight_decay: f64,
    pub seed: u64,
    pub objective: GeomObjective,
    /// The sigma-window extension factor (`gamma`) of the
    /// [`crate::sigma::DblockSigmaSampler`].
    pub gamma: f64,
    /// Boundary-consistency weight in diffusion mode; `0.0` adds nothing.
    pub consistency_weight: f64,
    /// Print (and log, if `log_path` is set) every this many steps.
    pub log_every: usize,
    pub log_path: Option<PathBuf>,
    /// Where checkpoints go, `None` for nowhere.
    pub out_dir: Option<PathBuf>,
    /// Also checkpoint every this many steps (`0` = only at the end).
    pub checkpoint_every: usize,
    /// The squared displacement below which a refinement step counts as
    /// converged.
    pub convergence_tol: f64,
    /// Charge on the router's balance + z-loss; `0.0` is off.
    pub moe_balance_weight: f64,
    /// Micro-batches per optimizer step, averaged not summed (roadmap 20.2,
    /// ported from the trunk loop). `1` steps on every micro-batch.
    #[serde(default = "default_geom_accumulate")]
    pub accumulate: usize,
    /// EMA decay for the evaluation shadow (roadmap 22.1). `None` returns
    /// the live weights.
    #[serde(default)]
    pub ema_decay: Option<f64>,
}

fn default_geom_accumulate() -> usize {
    1
}

impl Default for GeomTrainConfig {
    fn default() -> Self {
        Self {
            steps: 2000,
            batch_size: 64,
            lr: 1e-3,
            lr_schedule: crate::schedule::LrSchedule::cosine(1e-3, 2000),
            clip_norm: 1.0,
            weight_decay: 0.01,
            seed: 42,
            objective: GeomObjective::Answer,
            gamma: 0.05,
            consistency_weight: 0.1,
            log_every: 50,
            log_path: None,
            out_dir: None,
            checkpoint_every: 0,
            convergence_tol: 1e-3,
            moe_balance_weight: 0.01,
            accumulate: default_geom_accumulate(),
            ema_decay: None,
        }
    }
}

/// What a geometric-reasoner run did.
#[derive(Debug, Clone)]
pub struct GeomTrainReport {
    pub steps_taken: usize,
    /// Steps discarded for a non-finite loss.
    pub steps_skipped: usize,
    /// Steps whose gradient norm exceeded `clip_norm` and were rescaled.
    pub steps_clipped: usize,
    /// Micro-batches folded without stepping yet (`--accumulate`).
    pub steps_accumulated: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub mean_loss: f32,
    pub first_accuracy: f32,
    pub last_accuracy: f32,
    pub mean_accuracy: f32,
    /// The refinement energy at the first and the last step: the attractor
    /// state's depth in its basin.
    pub energy_start: f32,
    pub energy_end: f32,
    /// Mean squared displacement of the final refinement step.
    pub displacement_end: f32,
    /// Steps whose final displacement fell below the convergence tolerance.
    pub converged_steps: usize,
    /// The boundary-consistency term at the last step (diffusion mode); 0
    /// otherwise.
    pub last_consistency: f32,
    /// Router diagnostics at the last step: the load entropy, the per-token
    /// entropy (both normalized by the box count) and the balance charge.
    pub load_entropy_end: f32,
    pub token_entropy_end: f32,
    pub balance_end: f32,
    pub scenes_seen: usize,
    pub elapsed_secs: f64,
    /// The final checkpoint, when `out_dir` was set.
    pub checkpoint: Option<PathBuf>,
}

/// Train the geometric reasoner on an aligned corpus (roadmap Phase 33).
///
/// The answer objective trains the full trajectory jointly with the
/// cross-entropy on the answer token. The diffusion objective is the
/// framework's core scheme applied to the geometric latent: a block sampled
/// per step, sigmas drawn from its window, the clean state corrupted at that
/// scale, EDM preconditioning, boundary consistency between adjacent blocks,
/// and the sigma-weighted answer CE.
pub fn train_geom<B: AutodiffBackend<FloatElem = f32>>(
    mut model: GeometricReasoner<B>,
    corpus: &mut TokenCorpus,
    meta: &GeomMeta,
    config: &GeomTrainConfig,
    device: &B::Device,
) -> anyhow::Result<(GeometricReasoner<B>, GeomTrainReport)> {
    anyhow::ensure!(
        config.steps > 0 && config.batch_size > 0,
        "steps and batch_size must be positive"
    );
    anyhow::ensure!(
        meta.scene_len == model.scene_len(),
        "the metadata's scene length {} does not match the model's {}",
        meta.scene_len,
        model.scene_len()
    );
    anyhow::ensure!(
        corpus.len() % meta.scene_len == 0,
        "the corpus holds {} tokens, not a multiple of the {}-token scene length",
        corpus.len(),
        meta.scene_len
    );
    let bounds = crate::sigma::block_sigmas(model.num_blocks());
    let sampler = crate::sigma::DblockSigmaSampler::new(model.num_blocks(), config.gamma);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    // Honour the caller's `lr` as the schedule's peak: the config default is
    // written for `lr`, and a schedule that ignored it would silently run the
    // wrong peak whenever a caller raised the learning rate.
    let lr_schedule = match &config.lr_schedule {
        crate::schedule::LrSchedule::Constant { .. } => {
            crate::schedule::LrSchedule::Constant { lr: config.lr }
        }
        crate::schedule::LrSchedule::WarmupCosine {
            min_lr,
            warmup_steps,
            ..
        } => crate::schedule::LrSchedule::WarmupCosine {
            peak: config.lr,
            min_lr: *min_lr,
            warmup_steps: *warmup_steps,
            total_steps: config.steps,
        },
        crate::schedule::LrSchedule::WarmupConstant { warmup_steps, .. } => {
            crate::schedule::LrSchedule::WarmupConstant {
                peak: config.lr,
                warmup_steps: *warmup_steps,
            }
        }
    };
    let mut rng = ChaCha12Rng::seed_from_u64(config.seed);
    let mut logger = config
        .log_path
        .as_ref()
        .map(|path| crate::logging::MetricsLogger::open(path))
        .transpose()?;
    let mut accumulator = crate::schedule::GradientAccumulator::new(config.accumulate);
    let mut ema = config
        .ema_decay
        .map(|d| crate::schedule::Ema::new(&model, d));
    if let Some(e) = &ema {
        println!(
            "EMA: decay {:.4} (bias-corrected; the shadow is returned for evaluation)",
            e.decay()
        );
    }
    if config.accumulate > 1 {
        println!(
            "gradient accumulation: {} micro-batches per optimizer step (averaged)",
            accumulator.every()
        );
    }
    let mut report = GeomTrainReport {
        steps_clipped: 0,
        steps_taken: 0,
        steps_skipped: 0,
        steps_accumulated: 0,
        first_loss: f32::NAN,
        last_loss: f32::NAN,
        mean_loss: 0.0,
        first_accuracy: f32::NAN,
        last_accuracy: f32::NAN,
        mean_accuracy: 0.0,
        energy_start: f32::NAN,
        energy_end: f32::NAN,
        displacement_end: f32::NAN,
        converged_steps: 0,
        last_consistency: 0.0,
        load_entropy_end: 0.0,
        token_entropy_end: 0.0,
        balance_end: 0.0,
        scenes_seen: 0,
        elapsed_secs: 0.0,
        checkpoint: None,
    };
    let mut loss_sum = 0.0f64;
    let mut accuracy_sum = 0.0f64;
    let started = std::time::Instant::now();
    for step in 0..config.steps {
        <B as burn::tensor::backend::Backend>::seed(
            device,
            crate::train::step_seed(config.seed, step),
        );
        let windows = sample_aligned_batch(corpus, meta.scene_len, config.batch_size, &mut rng)?;
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| i64::from(*t)))
            .collect();
        let tokens_full = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
            .reshape([config.batch_size, meta.scene_len]);
        let (loss, metrics) = match config.objective {
            GeomObjective::Answer => model.answer_step(tokens_full, config.moe_balance_weight),
            GeomObjective::Diffusion => {
                let block_idx = rng.random_range(0..model.num_blocks());
                let sigma = sampler.sample(&mut rng, block_idx, 1)[0];
                let (z_star, _) = model.clean_state(tokens_full.clone())?;
                let block_result = model.block_step(
                    tokens_full.clone(),
                    block_idx,
                    sigma,
                    &z_star,
                    config.moe_balance_weight,
                )?;
                let mut loss = block_result.loss;
                let mut consistency = 0.0f32;
                if block_idx + 1 < model.num_blocks() && config.consistency_weight > 0.0 {
                    let shared = crate::sigma::shared_boundary_sigma(&bounds, block_idx);
                    let at_boundary = model.block_step(
                        tokens_full.clone(),
                        block_idx,
                        shared,
                        &z_star,
                        config.moe_balance_weight,
                    )?;
                    let next_at_boundary = model.block_step(
                        tokens_full.clone(),
                        block_idx + 1,
                        shared,
                        &z_star,
                        config.moe_balance_weight,
                    )?;
                    let diff = at_boundary.x0.sub(next_at_boundary.x0);
                    let cons = diff.clone().mul(diff).mean();
                    consistency = cons.clone().detach().into_scalar();
                    loss = loss.add(cons.mul_scalar(config.consistency_weight as f32));
                }
                let loss_value = loss.clone().detach().into_scalar();
                Ok((
                    loss,
                    GeomStepMetrics {
                        loss: loss_value,
                        accuracy: block_result.accuracy,
                        energy: block_result.energy,
                        displacement: block_result.displacement,
                        consistency,
                        balance: block_result.balance,
                        load_entropy: block_result.load_entropy,
                        token_entropy: block_result.token_entropy,
                        scenes: config.batch_size,
                    },
                ))
            }
        }?;
        if !metrics.loss.is_finite() {
            // The same policy as every loop in this crate: a pathological step
            // is discarded, not clipped into something that looks fine.
            report.steps_skipped += 1;
            accumulator.skip();
            println!(
                "step {step}: non-finite loss {}, step discarded",
                metrics.loss
            );
            continue;
        }
        let scaled = if accumulator.every() > 1 {
            loss.mul_scalar(accumulator.loss_scale() as f32)
        } else {
            loss
        };
        let grads = GradientsParams::from_grads(scaled.backward(), &model);
        let cycle = accumulator.fold(grads, &model);
        let Some(summed) = cycle.into_gradients() else {
            report.steps_accumulated += 1;
            continue;
        };
        let mut grads = summed;
        // Clip before the step. The relaxation multiplies its gradient by a
        // step size derived from the metric's *own* Lipschitz bound, so a batch
        // that lands far from the basin produces a large gradient exactly when
        // the state is least trustworthy; without a bound those steps throw
        // away the progress of every one after them. Clipping applies to the
        // accumulated gradient: that is the step actually taken.
        if config.clip_norm > 0.0 {
            let total_norm = crate::quality::global_grad_norm(&model, &grads);
            if total_norm > config.clip_norm as f32 {
                crate::schedule::clip_gradients(
                    &mut grads,
                    &model,
                    total_norm,
                    config.clip_norm as f32,
                );
                report.steps_clipped += 1;
            }
        }
        // The schedule, not a constant. The crate's own reasoning applies with
        // more force here than in the LM trunk: block 0 and block 1 see very
        // different landscapes, and a constant peak learning rate that suits the
        // last block overshoots the first one. Warmup then cosine decay to a
        // small floor is what `train.rs` already does for the trunk.
        model = optim.step(lr_schedule.at(step), model, grads);
        if let Some(ema) = ema.as_mut() {
            ema.update::<B>(&model);
        }

        if report.steps_taken == 0 {
            report.first_loss = metrics.loss;
            report.first_accuracy = metrics.accuracy;
            report.energy_start = metrics.energy;
        }
        report.last_loss = metrics.loss;
        report.last_accuracy = metrics.accuracy;
        report.energy_end = metrics.energy;
        report.displacement_end = metrics.displacement;
        report.last_consistency = metrics.consistency;
        report.load_entropy_end = metrics.load_entropy;
        report.token_entropy_end = metrics.token_entropy;
        report.balance_end = metrics.balance;
        if metrics.displacement < config.convergence_tol as f32 {
            report.converged_steps += 1;
        }
        report.scenes_seen += metrics.scenes;
        loss_sum += f64::from(metrics.loss);
        accuracy_sum += f64::from(metrics.accuracy);
        report.steps_taken += 1;

        if config.log_every > 0 && (step % config.log_every == 0 || step + 1 == config.steps) {
            println!(
                "step {step}: loss {:.4} accuracy {:.4} | energy {:.4} displacement {:.6} | consistency {:.4} | router load H {:.3} token H {:.3} balance {:.4}",
                metrics.loss,
                metrics.accuracy,
                metrics.energy,
                metrics.displacement,
                metrics.consistency,
                metrics.load_entropy,
                metrics.token_entropy,
                metrics.balance
            );
            if let Some(logger) = logger.as_mut() {
                logger.log(
                    step,
                    &[
                        ("loss", format!("{:.6}", metrics.loss)),
                        ("accuracy", format!("{:.6}", metrics.accuracy)),
                        ("energy", format!("{:.6}", metrics.energy)),
                        ("displacement", format!("{:.8}", metrics.displacement)),
                        ("consistency", format!("{:.6}", metrics.consistency)),
                        (
                            "router_load_entropy",
                            format!("{:.6}", metrics.load_entropy),
                        ),
                        (
                            "router_token_entropy",
                            format!("{:.6}", metrics.token_entropy),
                        ),
                        ("router_balance", format!("{:.6}", metrics.balance)),
                    ],
                )?;
            }
        }
        if config.checkpoint_every > 0 && report.steps_taken % config.checkpoint_every == 0 {
            if let Some(dir) = &config.out_dir {
                let path = checkpoint::save_content_addressed(model.clone(), dir, "geom")?;
                println!("checkpointed at step {step}: {}", path.display());
            }
        }
    }
    report.elapsed_secs = started.elapsed().as_secs_f64();
    report.mean_loss = (loss_sum / report.steps_taken.max(1) as f64) as f32;
    report.mean_accuracy = (accuracy_sum / report.steps_taken.max(1) as f64) as f32;
    if let Some(e) = &ema {
        println!("returning EMA weights ({} updates)", e.updates());
        model = e.shadow().clone();
    }
    if let Some(dir) = &config.out_dir {
        report.checkpoint = Some(checkpoint::save_content_addressed(
            model.clone(),
            dir,
            "geom",
        )?);
    }
    Ok((model, report))
}

/// The sweep shape shared by [`evaluate_geom`] and [`evaluate_geom_answers`]:
/// how much to evaluate, at what depth, and from which seed.
///
/// The seed lives here rather than at each call site on purpose. Both
/// evaluators draw scenes from one `StdRng` seeded from it, so two calls that
/// share an eval see the same scenes and their answers are comparable
/// scene-for-scene; that is the whole basis of the trace-dependence check, and
/// it is easy to break by passing seeds in the wrong order.
#[derive(Debug, Clone, Copy)]
pub struct GeomEval {
    /// Batches to draw.
    pub batches: usize,
    /// Scenes per batch.
    pub batch_size: usize,
    /// Relaxation steps per scene, overriding the configured depth.
    pub refine_override: Option<usize>,
    /// RNG seed for scene selection.
    pub seed: u64,
}

/// Answer accuracy over aligned scenes, optionally with the refinement depth
/// overridden -- the test-time-compute curve for reasoning: the same weights,
/// more relaxation steps. Returns `(accuracy, scenes seen)`.
///
/// Scoring goes through the reasoner's answer cross-entropy, the same
/// answer-set-restricted metric training optimizes: with a declared answer
/// set, an unrestricted `argmax` over the full vocabulary can land on a byte
/// that was never a candidate and score a geometrically correct reply as
/// wrong. Without a declared set the restriction is the identity, so this is
/// exactly the old behavior there.
///
/// The sweep shape lives in [`GeomEval`] rather than in the signature: the
/// same four numbers drive [`evaluate_geom`] and [`evaluate_geom_answers`], and
/// passing them positionally is how the two calls drift apart.
pub fn evaluate_geom<B: Backend<FloatElem = f32>>(
    model: &GeometricReasoner<B>,
    corpus: &mut TokenCorpus,
    meta: &GeomMeta,
    eval: &GeomEval,
    device: &B::Device,
) -> anyhow::Result<(f32, usize)> {
    let GeomEval {
        batches,
        batch_size,
        refine_override,
        seed,
    } = *eval;
    anyhow::ensure!(
        batches > 0 && batch_size > 0,
        "batches and batch_size must be positive"
    );
    anyhow::ensure!(
        meta.scene_len == model.scene_len(),
        "the metadata's scene length {} does not match the model's {}",
        meta.scene_len,
        model.scene_len()
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut correct = 0.0f32;
    let mut seen = 0usize;
    for _ in 0..batches {
        let windows = sample_aligned_batch(corpus, meta.scene_len, batch_size, &mut rng)?;
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| i64::from(*t)))
            .collect();
        let tokens_full = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
            .reshape([batch_size, meta.scene_len]);
        let tokens = tokens_full.clone().narrow(1, 0, meta.scene_len - 1);
        let answers = tokens_full
            .narrow(1, meta.scene_len - 1, 1)
            .reshape([batch_size]);
        let out = model.forward_depth(tokens, refine_override)?;
        let (_, _, accuracy) = model.answer_ce(&out.logits, &answers);
        correct += accuracy * batch_size as f32;
        seen += batch_size;
    }
    Ok((correct / seen.max(1) as f32, seen))
}

/// Predicted answer bytes over aligned scenes, same sampling as
/// [`evaluate_geom`]: the per-scene material behind an accuracy number, for
/// trace-dependence checks (does deeper refinement actually change answers,
/// or is the extra depth post-hoc decoration?). Same seed sees the same
/// scenes, so two calls at different depths are directly comparable.
pub fn evaluate_geom_answers<B: Backend<FloatElem = f32>>(
    model: &GeometricReasoner<B>,
    corpus: &mut TokenCorpus,
    meta: &GeomMeta,
    eval: &GeomEval,
    device: &B::Device,
) -> anyhow::Result<(Vec<u16>, usize)> {
    let GeomEval {
        batches,
        batch_size,
        refine_override,
        seed,
    } = *eval;
    anyhow::ensure!(
        batches > 0 && batch_size > 0,
        "batches and batch_size must be positive"
    );
    anyhow::ensure!(
        meta.scene_len == model.scene_len(),
        "the metadata's scene length {} does not match the model's {}",
        meta.scene_len,
        model.scene_len()
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut answers = Vec::with_capacity(batches * batch_size);
    let mut seen = 0usize;
    for _ in 0..batches {
        let windows = sample_aligned_batch(corpus, meta.scene_len, batch_size, &mut rng)?;
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| i64::from(*t)))
            .collect();
        let tokens_full = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
            .reshape([batch_size, meta.scene_len]);
        let tokens = tokens_full.clone().narrow(1, 0, meta.scene_len - 1);
        let out = model.forward_depth(tokens, refine_override)?;
        answers.extend(model.predict_answers(&out.logits));
        seen += batch_size;
    }
    Ok((answers, seen))
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
    use burn::backend::NdArray;
    use burn::tensor::{Shape, TensorData};

    type B = NdArray<f32>;

    #[test]
    fn test_corpus_layout_is_fixed_width() {
        let (tokens, meta) = generate_corpus(
            4,
            &["nearest", "direction", "inside", "collinear", "farthest"],
            100,
            3,
        )
        .unwrap();
        assert_eq!(meta.scene_len, 10 + 5 * 4);
        assert_eq!(tokens.len(), 100 * meta.scene_len);
        let mut answers = BTreeSet::new();
        for scene in 0..100 {
            let doc = &tokens[scene * meta.scene_len..(scene + 1) * meta.scene_len];
            assert_eq!(doc[meta.scene_len - 2], u16::from(b'?'));
            let answer = doc[meta.scene_len - 1];
            assert!(
                meta.answer_tokens.contains(&answer),
                "answer byte {answer} is not in the metadata's set"
            );
            answers.insert(answer);
        }
        // both binary answers appear: the generator balances the y/n tasks
        assert!(answers.contains(&u16::from(b'y')));
        assert!(answers.contains(&u16::from(b'n')));
    }

    #[test]
    fn test_aligned_windows_round_trip() {
        let (tokens, meta) = generate_corpus(4, &["nearest"], 16, 5).unwrap();
        let dir = std::env::temp_dir().join(format!("dblocks-geom-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.bin");
        TokenCorpus::write(&path, &tokens).unwrap();
        let mut corpus = TokenCorpus::in_memory(&path).unwrap();
        let mut streaming = TokenCorpus::streaming(&path).unwrap();
        for scene in 0..16 {
            let start = scene * meta.scene_len;
            assert_eq!(
                corpus.window(start, meta.scene_len).unwrap(),
                streaming.window(start, meta.scene_len).unwrap(),
                "the two readers must agree on aligned scenes"
            );
        }
        let mut rng = StdRng::seed_from_u64(9);
        let batch = sample_aligned_batch(&mut corpus, meta.scene_len, 8, &mut rng).unwrap();
        assert_eq!(batch.len(), 8);
        assert!(batch.iter().all(|row| row.len() == meta.scene_len));
    }

    #[test]
    fn test_metric_is_positive_definite_by_construction() {
        let device = Default::default();
        let mut metric = MetricField::<B>::new(4, &device);
        metric.lower = Param::from_tensor(Tensor::<B, 2>::random(
            [4, 4],
            Distribution::Normal(0.0, 1.0),
            &device,
        ));
        metric.log_diag = Param::from_tensor(Tensor::<B, 1>::random(
            [4],
            Distribution::Normal(0.0, 1.0),
            &device,
        ));
        let g = metric.metric();
        let mut worst = 0.0f32;
        for _ in 0..256 {
            let v = Tensor::<B, 1>::random([4], Distribution::Normal(0.0, 1.0), &device);
            let q = v
                .clone()
                .reshape([1, 4])
                .matmul(g.clone())
                .reshape([4])
                .mul(v)
                .sum()
                .into_scalar();
            worst = worst.max(-q);
        }
        assert!(
            worst <= 1e-5,
            "quadratic form {worst} below zero: the factorization is not PSD"
        );
    }

    #[test]
    fn test_energy_descent_is_monotone() {
        let device = Default::default();
        let metric = MetricField::<B>::new(4, &device);
        let g = metric.metric();
        let lambda = 0.5f32;
        let slots = 6usize;
        let eta = lipschitz_step(&g, lambda, slots);
        let mut z = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
        let c = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
        let mut worst = 0.0f32;
        let mut previous = energy_value(&g, &z, &c, lambda, slots);
        for _ in 0..64 {
            z = energy_grad_step(&g, &z, &c, lambda, slots, eta);
            let current = energy_value(&g, &z, &c, lambda, slots);
            worst = worst.max(current - previous);
            previous = current;
        }
        assert!(
            worst <= 1e-4,
            "energy increased by {worst} under the certified step"
        );
    }

    #[test]
    fn test_the_step_is_the_gradient_of_the_reported_energy() {
        // The monotone-descent test above can pass by luck: a step that is
        // *not* the gradient still shrinks the energy for a while, and a
        // sufficiently small `eta` hides the difference. So check the claim the
        // certificate actually rests on -- the step direction is the gradient of
        // `energy_value`, in the metric -- by finite differences.
        //
        // This is the test whose absence let the repulsion coefficient drift to
        // `lambda` where the energy's is `2*lambda`: the step was then
        // descending a different function, and the descent it did achieve was
        // not guaranteed to be monotone.
        let device = Default::default();
        let metric = MetricField::<B>::new(4, &device);
        let g = metric.metric();
        let lambda = 0.5f32;
        let slots = 4usize;
        let dim = 4usize;
        let eta = 0.01f32; // any small value: only the direction is under test

        // Host-side coordinates, so the finite differences are exact about
        // which entry they perturb.
        let z_data: Vec<f32> = (0..slots * dim).map(|i| (i as f32) * 0.37 - 0.5).collect();
        let c_data: Vec<f32> = (0..slots * dim).map(|i| (i as f32) * -0.21 + 0.3).collect();
        let build = |data: &[f32]| {
            Tensor::<B, 3>::from_data(
                TensorData::new(data.to_vec(), Shape::new([1, slots, dim])),
                &device,
            )
        };
        let z = build(&z_data);
        let c = build(&c_data);

        // The step is `z <- z - 2 eta G total` and the energy's gradient is
        // `2 G total`, so the displacement is `eta` times the gradient --
        // divide by `eta`, not `2 eta`.
        let stepped = energy_grad_step(&g, &z, &c, lambda, slots, eta);
        let direction = z
            .clone()
            .sub(stepped)
            .div_scalar(eta)
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("contiguous direction")
            .to_vec();

        let h = 1e-3f32;
        let mut worst = 0.0f32;
        for i in 0..z_data.len() {
            let mut up = z_data.clone();
            up[i] += h;
            let mut down = z_data.clone();
            down[i] -= h;
            let numeric = (energy_value(&g, &build(&up), &c, lambda, slots)
                - energy_value(&g, &build(&down), &c, lambda, slots))
                / (2.0 * h);
            let analytic = direction[i];
            worst = worst.max((analytic - numeric).abs() / (1.0 + numeric.abs()));
        }
        assert!(
            worst < 1e-2,
            "the step is not the energy's gradient: worst relative error {worst}"
        );
    }

    #[test]
    fn test_relaxation_reaches_the_exact_fixed_point() {
        // An identity metric (a fresh field) and a known lambda. The step is
        // `z <- z - 2 eta G (z - c + 2 lambda (K z - sum(z)))`, so the fixed
        // point solves `(I + 2*lambda*K) z - 2*lambda*sum(z) = c`; with
        // `A = (I + 2*lambda*K)I - 2*lambda*J` and `A - K*b = 1` the inverse
        // collapses to `(c + 2*lambda*sum(c)) / (1 + 2*lambda*K)`.
        let device = Default::default();
        let metric = MetricField::<B>::new(4, &device);
        let g = metric.metric();
        let lambda = 0.5f32;
        let slots = 6usize;
        let eta = lipschitz_step(&g, lambda, slots);
        let z0 = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
        let c = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
        let mut z = z0.clone();
        for _ in 0..96 {
            z = energy_grad_step(&g, &z, &c, lambda, slots, eta);
        }
        let fixed = c
            .clone()
            .add(c.clone().sum_dim(1).mul_scalar(2.0 * lambda))
            .div_scalar(1.0 + 2.0 * lambda * slots as f32);
        let gap = z.sub(fixed);
        let residual = gap.clone().mul(gap).sum().sqrt().into_scalar();
        let start_gap = z0.sub(c);
        let initial = start_gap.clone().mul(start_gap).sum().sqrt().into_scalar();
        assert!(
            residual / initial.max(1e-6) < 1e-2,
            "96 certified steps left {residual} of the initial {initial} gap"
        );
    }

    #[test]
    fn test_repulsion_off_lands_on_the_context() {
        let device = Default::default();
        let metric = MetricField::<B>::new(4, &device);
        let g = metric.metric();
        let lambda = 0.0f32;
        let slots = 6usize;
        let eta = lipschitz_step(&g, lambda, slots);
        let z0 = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
        let c = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
        let mut z = z0.clone();
        for _ in 0..8 {
            z = energy_grad_step(&g, &z, &c, lambda, slots, eta);
        }
        let gap = z.sub(c.clone());
        let end = gap.clone().mul(gap).sum().sqrt().into_scalar();
        let start_gap = z0.sub(c);
        let start = start_gap.clone().mul(start_gap).sum().sqrt().into_scalar();
        assert!(
            end / start.max(1e-6) < 1e-4,
            "eight certified steps left {end} of the initial {start} gap"
        );
    }

    #[test]
    fn test_model_span_identity() {
        let device = Default::default();
        let config = GeomConfig::tiny();
        let model = GeometricReasoner::<B>::new(&config, &device).unwrap();
        let scene: Vec<i64> = (0..config.scene_len - 1)
            .map(|i| i64::from(b'A' + (i % 26) as u8))
            .collect();
        let tokens = Tensor::<B, 1, Int>::from_ints(scene.as_slice(), &device)
            .reshape([1, config.scene_len - 1]);
        let direct = model.forward(tokens.clone()).unwrap().logits;
        let spanned = model
            .forward_span(tokens, 0..model.num_blocks())
            .unwrap()
            .logits;
        let diff = (direct - spanned).abs().max().into_scalar();
        assert_eq!(
            diff, 0.0,
            "a span covering every block must be the full path"
        );
    }

    #[test]
    fn test_answer_step_shapes_and_metrics() {
        let device = Default::default();
        let config = GeomConfig::tiny();
        let model = GeometricReasoner::<B>::new(&config, &device).unwrap();
        let (tokens, meta) = generate_corpus(4, &["nearest"], 8, 17).unwrap();
        let flat: Vec<i64> = tokens.iter().map(|t| i64::from(*t)).collect();
        let tokens_full =
            Tensor::<B, 1, Int>::from_ints(flat.as_slice(), &device).reshape([8, meta.scene_len]);
        let (loss, metrics) = model.answer_step(tokens_full, 0.01).unwrap();
        assert!(loss.dims() == [1]);
        assert!(metrics.loss.is_finite());
        assert_eq!(metrics.scenes, 8);
        // the untrained model sits near ln(vocab): confidently wrong, but not
        // absurdly so
        assert!(
            // the readout head is untied here, so the bar is looser than the
            // LM trunk's tied-head certificate
            (metrics.loss - (config.vocab_size as f32).ln()).abs() < 2.5,
            "initial loss {} is far from ln(vocab) = {}",
            metrics.loss,
            (config.vocab_size as f32).ln()
        );
    }

    #[test]
    fn test_compass_byte_sectors() {
        assert_eq!(compass_byte(0, 5), b'0');
        assert_eq!(compass_byte(5, 5), b'1');
        assert_eq!(compass_byte(5, 0), b'2');
        assert_eq!(compass_byte(5, -5), b'3');
        assert_eq!(compass_byte(0, -5), b'4');
        assert_eq!(compass_byte(-5, -5), b'5');
        assert_eq!(compass_byte(-5, 0), b'6');
        assert_eq!(compass_byte(-5, 5), b'7');
    }
}
