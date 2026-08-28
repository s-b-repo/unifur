//! Quality gates (roadmap Phase 12).
//!
//! Two families live here, both answering "is this step's output usable?":
//!
//! - **Sampling gates** ([`QualityGateConfig`], [`LayerGates`],
//!   [`evaluate`]): per-sample sanity checks on denoised outputs, used by the
//!   multi-block inference strategies to reject bad updates.
//! - **Training gates** ([`TrainingChecks`], [`StepVerdict`],
//!   [`TrainingHealth`]): verification at *every phase* of a training step --
//!   before the run, after the loss, after the backward pass, after the
//!   optimizer step, and periodically against the live weights.
//!
//! The training side exists because block-wise training fails quietly. A step
//! trains one block on one noise window; if that block receives no gradient,
//! or the optimizer pushes a parameter to infinity, the loss of the *other*
//! blocks keeps the average looking healthy for a long time. Checking each
//! phase separately turns those into named, attributable failures.
//!
//! A gate inspects the transition `x0_prev -> x0_new` (denoised embedding
//! estimates) plus the class-probability distribution and marks samples as
//! failing when:
//!
//! - cosine similarity drops below [`QualityGateConfig::min_cosine`]
//!   (direction of the embedding estimate changed too abruptly),
//! - mean squared error exceeds [`QualityGateConfig::max_mse`],
//! - max class probability falls below [`QualityGateConfig::min_confidence`].
//!
//! Failing samples can then keep their previous latent (see
//! `crate::multi_block`) or be filtered from training batches.

use burn::tensor::{Tensor, backend::Backend};

/// Thresholds for the quality checks. Set a threshold to `None`/`f32::INFINITY`
/// style sentinels to disable an individual check.
#[derive(Debug, Clone)]
pub struct QualityGateConfig {
    /// Minimum cosine similarity between consecutive x0 estimates.
    pub min_cosine: f32,
    /// Maximum mean squared error between consecutive x0 estimates.
    pub max_mse: f32,
    /// Optional minimum max-class probability of the new estimate.
    pub min_confidence: Option<f32>,
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        Self {
            min_cosine: 0.0,
            max_mse: f32::INFINITY,
            min_confidence: None,
        }
    }
}

impl QualityGateConfig {
    /// Gate that only rejects degenerate transitions (negative cosine).
    pub fn lenient() -> Self {
        Self::default()
    }

    /// Typical conservative gate for embedding-space diffusion sampling.
    pub fn strict() -> Self {
        Self {
            min_cosine: 0.5,
            max_mse: 4.0,
            min_confidence: Some(0.0),
        }
    }
}

/// Result of evaluating one transition for a whole batch.
#[derive(Debug, Clone)]
pub struct GateReport {
    /// Per-sample pass/fail (`true` = update accepted).
    pub passed: Vec<bool>,
    /// Batch-mean cosine similarity.
    pub mean_cosine: f32,
    /// Batch-mean squared error.
    pub mean_mse: f32,
    /// Batch-mean max class probability (if probabilities were provided).
    pub mean_confidence: Option<f32>,
}

impl GateReport {
    pub fn num_passed(&self) -> usize {
        self.passed.iter().filter(|&&p| p).count()
    }

    pub fn all_passed(&self) -> bool {
        self.passed.iter().all(|&p| p)
    }
}

/// Evaluate `x0_prev -> x0_new` against the gate thresholds.
///
/// `probs_new` is optional; when given it feeds the confidence check. On the
/// first step there is no previous estimate, so callers should skip gating.
pub fn evaluate<B: Backend>(
    config: &QualityGateConfig,
    x0_prev: &Tensor<B, 2>,
    x0_new: &Tensor<B, 2>,
    probs_new: Option<&Tensor<B, 2>>,
) -> GateReport {
    let device = x0_new.device();

    // Per-sample cosine similarity along the embedding dim.
    let dot = (x0_prev.clone() * x0_new.clone()).sum_dim(1);
    let norm_p = x0_prev.clone().powf_scalar(2.0).sum_dim(1).sqrt();
    let norm_n = x0_new.clone().powf_scalar(2.0).sum_dim(1).sqrt();
    let cos = dot / (norm_p * norm_n).clamp_min(1e-12);

    // Per-sample MSE.
    let mse = (x0_new.clone() - x0_prev.clone())
        .powf_scalar(2.0)
        .sum_dim(1)
        / x0_new.dims()[1] as f32;

    let cos_v = cos.into_data().convert::<f32>().iter::<f32>().collect::<Vec<_>>();
    let mse_v = mse.into_data().convert::<f32>().iter::<f32>().collect::<Vec<_>>();

    let conf_v: Option<Vec<f32>> = probs_new.map(|p| {
        p.clone().max_dim(1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect()
    });

    let n = cos_v.len();
    let mut passed = Vec::with_capacity(n);
    let (mut sum_cos, mut sum_mse) = (0.0f32, 0.0f32);
    let mut conf_mean = 0.0f32;
    for i in 0..n {
        let ok_cos = cos_v[i] >= config.min_cosine;
        let ok_mse = mse_v[i] <= config.max_mse;
        let ok_conf = match (&conf_v, config.min_confidence) {
            (Some(c), Some(min)) => c[i] >= min,
            _ => true,
        };
        passed.push(ok_cos && ok_mse && ok_conf);
        sum_cos += cos_v[i];
        sum_mse += mse_v[i];
    }
    if let Some(c) = &conf_v {
        conf_mean = c.iter().sum::<f32>() / n.max(1) as f32;
    }
    let _ = device;

    GateReport {
        passed,
        mean_cosine: sum_cos / n.max(1) as f32,
        mean_mse: sum_mse / n.max(1) as f32,
        mean_confidence: if conf_v.is_some() { Some(conf_mean) } else { None },
    }
}

/// Per-layer quality gates (roadmap 12.5).
///
/// The batch-level [`QualityGateConfig`] treats every block identically, but
/// blocks do not face identical problems: a high-sigma block legitimately
/// makes large, direction-changing updates, while a low-sigma block that does
/// so is diverging. `LayerGates` keeps one default plus optional per-block
/// overrides so each layer span can be held to its own standard.
#[derive(Debug, Clone, Default)]
pub struct LayerGates {
    /// Applied to any block without an override.
    pub default: QualityGateConfig,
    /// Overrides indexed by block; `None` entries fall back to `default`.
    pub per_block: Vec<Option<QualityGateConfig>>,
}

impl LayerGates {
    /// One configuration for every block (equivalent to the batch-level gate).
    pub fn uniform(config: QualityGateConfig) -> Self {
        Self { default: config, per_block: Vec::new() }
    }

    /// Gates that tighten monotonically as sigma falls, i.e. as the block
    /// index grows: block 0 denoises from `sigma_max` and is allowed large
    /// moves, the final block should only be making small corrections.
    ///
    /// `min_cosine` ramps linearly from `loose.min_cosine` to
    /// `strict.min_cosine` and `max_mse` geometrically from `loose` to
    /// `strict`, so both stay monotone in the block index by construction.
    pub fn tightening(num_blocks: usize, loose: QualityGateConfig, strict: QualityGateConfig) -> Self {
        let per_block = (0..num_blocks)
            .map(|b| {
                let t = if num_blocks > 1 {
                    b as f32 / (num_blocks - 1) as f32
                } else {
                    1.0
                };
                let mse = if loose.max_mse.is_finite() && strict.max_mse.is_finite() {
                    loose.max_mse.powf(1.0 - t) * strict.max_mse.powf(t)
                } else {
                    loose.max_mse
                };
                Some(QualityGateConfig {
                    min_cosine: loose.min_cosine + (strict.min_cosine - loose.min_cosine) * t,
                    max_mse: mse,
                    min_confidence: match (loose.min_confidence, strict.min_confidence) {
                        (Some(lo), Some(hi)) => Some(lo + (hi - lo) * t),
                        (other, None) => other,
                        (None, other) => other,
                    },
                })
            })
            .collect();
        Self { default: strict, per_block }
    }

    /// Gate governing `block_idx`.
    pub fn for_block(&self, block_idx: usize) -> &QualityGateConfig {
        self.per_block
            .get(block_idx)
            .and_then(|slot| slot.as_ref())
            .unwrap_or(&self.default)
    }
}

/// Running per-block tally of gate decisions across a sampling run.
#[derive(Debug, Clone, Default)]
pub struct GateLedger {
    evaluated: Vec<usize>,
    rejected: Vec<usize>,
}

impl GateLedger {
    pub fn new(num_blocks: usize) -> Self {
        Self { evaluated: vec![0; num_blocks], rejected: vec![0; num_blocks] }
    }

    /// Record one batch-level decision attributed to `block_idx`.
    pub fn record(&mut self, block_idx: usize, report: &GateReport) {
        if block_idx >= self.evaluated.len() {
            self.evaluated.resize(block_idx + 1, 0);
            self.rejected.resize(block_idx + 1, 0);
        }
        self.evaluated[block_idx] += report.passed.len();
        self.rejected[block_idx] += report.passed.len() - report.num_passed();
    }

    /// Fold `other`'s tallies into this ledger.
    ///
    /// Both the evaluated *and* the rejected counts are carried over: adding
    /// only the rejections would drive every rate that has any rejection to
    /// 100%.
    pub fn merge(&mut self, other: &GateLedger) {
        let n = self.evaluated.len().max(other.evaluated.len());
        self.evaluated.resize(n, 0);
        self.rejected.resize(n, 0);
        for block in 0..other.evaluated.len() {
            self.evaluated[block] += other.evaluated[block];
            self.rejected[block] += other.rejected[block];
        }
    }

    /// Sample-updates evaluated against `block_idx`'s gate.
    pub fn evaluated(&self, block_idx: usize) -> usize {
        self.evaluated.get(block_idx).copied().unwrap_or(0)
    }

    /// Sample-updates rejected by `block_idx`.
    pub fn rejected(&self, block_idx: usize) -> usize {
        self.rejected.get(block_idx).copied().unwrap_or(0)
    }

    /// Rejection rate of `block_idx` in `[0, 1]`; `0.0` if never evaluated.
    pub fn rejection_rate(&self, block_idx: usize) -> f32 {
        match self.evaluated.get(block_idx) {
            Some(&n) if n > 0 => self.rejected[block_idx] as f32 / n as f32,
            _ => 0.0,
        }
    }

    /// Total rejected sample-updates across every block.
    pub fn total_rejected(&self) -> usize {
        self.rejected.iter().sum()
    }

    pub fn num_blocks(&self) -> usize {
        self.evaluated.len()
    }
}

/// Indices of samples that passed the gate (for training-batch filtering).
pub fn filter_indices(report: &GateReport) -> Vec<usize> {
    report
        .passed
        .iter()
        .enumerate()
        .filter(|(_, &p)| p)
        .map(|(i, _)| i)
        .collect()
}

/// A point in a training step at which quality is verified.
///
/// Naming the phase matters: "loss was NaN" and "a parameter went non-finite
/// after the update" call for different responses, and a run that reports only
/// "step rejected" cannot distinguish them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhase {
    /// Before the first step: schedule and preconditioning invariants.
    Preflight,
    /// After the loss is computed, before the backward pass.
    Loss,
    /// After the backward pass, before the optimizer step.
    Gradients,
    /// After the optimizer step.
    Parameters,
    /// Periodic re-verification against the live weights.
    Periodic,
}

impl TrainingPhase {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Preflight => "preflight",
            Self::Loss => "loss",
            Self::Gradients => "gradients",
            Self::Parameters => "parameters",
            Self::Periodic => "periodic",
        }
    }
}

/// Reject a step whose global gradient norm looks pathological (roadmap 12.4).
///
/// A near-zero norm means the executed span received no signal; a huge one
/// means the step would move the weights somewhere the loss surface does not
/// justify. Skipping is preferable to clipping here because a block-wise loop
/// visits a different block each step -- a clipped bad step still writes to
/// that block's parameters, while a skipped one leaves them for a healthier
/// draw.
#[derive(Debug, Clone, Copy)]
pub struct GradNormGate {
    pub min_norm: f32,
    pub max_norm: f32,
}

impl Default for GradNormGate {
    fn default() -> Self {
        Self { min_norm: 1e-8, max_norm: 1e4 }
    }
}

/// Which verification runs at which phase of a training step.
#[derive(Debug, Clone)]
pub struct TrainingChecks {
    /// Run the schedule/preconditioning certificates before step 0.
    pub preflight: bool,
    /// Reject a step whose loss is not finite.
    pub loss_finite: bool,
    /// Gradient-norm gate, or `None` to accept any gradient.
    pub grad_gate: Option<GradNormGate>,
    /// Verify every parameter is still finite after the optimizer step.
    pub parameters_finite: bool,
    /// Re-verify the live model every `n` steps; `None` disables it.
    pub verify_every: Option<usize>,
    /// Abort after this many consecutive rejected steps. Zero disables the
    /// abort, which is rarely what you want: a run that rejects every step is
    /// burning compute for nothing.
    pub max_consecutive_rejections: usize,
}

impl Default for TrainingChecks {
    fn default() -> Self {
        Self {
            preflight: true,
            loss_finite: true,
            grad_gate: Some(GradNormGate::default()),
            parameters_finite: true,
            verify_every: None,
            max_consecutive_rejections: 50,
        }
    }
}

impl TrainingChecks {
    /// Every check off. Useful for measuring the checks' own overhead, and for
    /// reproducing a run that predates them.
    pub fn none() -> Self {
        Self {
            preflight: false,
            loss_finite: false,
            grad_gate: None,
            parameters_finite: false,
            verify_every: None,
            max_consecutive_rejections: 0,
        }
    }

    /// Everything on, including periodic re-verification.
    pub fn thorough(verify_every: usize) -> Self {
        Self { verify_every: Some(verify_every), ..Self::default() }
    }
}

/// One phase's failure.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckFailure {
    pub phase: TrainingPhase,
    pub detail: String,
}

impl std::fmt::Display for CheckFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.phase.name(), self.detail)
    }
}

/// Outcome of verifying one training step.
#[derive(Debug, Clone, Default)]
pub struct StepVerdict {
    /// Whether the optimizer step should be applied.
    pub accepted: bool,
    /// Everything that failed, in phase order.
    pub failures: Vec<CheckFailure>,
    /// Measured global gradient norm (`0.0` when the gate is disabled).
    pub grad_norm: f32,
}

impl StepVerdict {
    /// A verdict with no failures.
    pub fn accepted() -> Self {
        Self { accepted: true, ..Self::default() }
    }

    pub fn reject(&mut self, phase: TrainingPhase, detail: impl Into<String>) {
        self.accepted = false;
        self.failures.push(CheckFailure { phase, detail: detail.into() });
    }

    /// Short reason for a log line, or `None` when the step was accepted.
    pub fn reason(&self) -> Option<String> {
        if self.failures.is_empty() {
            return None;
        }
        Some(
            self.failures
                .iter()
                .map(CheckFailure::to_string)
                .collect::<Vec<_>>()
                .join("; "),
        )
    }
}

/// Per-block training statistics.
///
/// Block-wise training visits one block per step, so a block that has gone
/// dead -- or one that is absorbing all the instability -- is invisible in the
/// aggregate loss curve but obvious here.
#[derive(Debug, Clone, Default)]
pub struct BlockHealth {
    pub steps: usize,
    pub rejected: usize,
    loss_sum: f64,
    grad_sum: f64,
    pub min_grad: f32,
    pub max_grad: f32,
}

impl BlockHealth {
    pub fn mean_loss(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            (self.loss_sum / self.steps as f64) as f32
        }
    }

    pub fn mean_grad_norm(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            (self.grad_sum / self.steps as f64) as f32
        }
    }

    pub fn rejection_rate(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            self.rejected as f32 / self.steps as f32
        }
    }
}

/// Running quality state of a training run.
#[derive(Debug, Clone, Default)]
pub struct TrainingHealth {
    per_block: Vec<BlockHealth>,
    consecutive_rejections: usize,
    pub worst_consecutive_rejections: usize,
    pub total_steps: usize,
    pub total_rejected: usize,
    /// Failures kept for the final report, capped so a pathological run cannot
    /// exhaust memory logging its own failure.
    failures: Vec<(usize, CheckFailure)>,
}

/// Most failures retained by [`TrainingHealth`].
const MAX_RETAINED_FAILURES: usize = 64;

impl TrainingHealth {
    pub fn new(num_blocks: usize) -> Self {
        Self { per_block: vec![BlockHealth::default(); num_blocks], ..Self::default() }
    }

    /// Fold one step's verdict in.
    pub fn record(&mut self, step: usize, block_idx: usize, loss: f32, verdict: &StepVerdict) {
        if block_idx >= self.per_block.len() {
            self.per_block.resize(block_idx + 1, BlockHealth::default());
        }
        let block = &mut self.per_block[block_idx];
        block.steps += 1;
        if loss.is_finite() {
            block.loss_sum += loss as f64;
        }
        block.grad_sum += verdict.grad_norm as f64;
        if block.steps == 1 {
            block.min_grad = verdict.grad_norm;
            block.max_grad = verdict.grad_norm;
        } else {
            block.min_grad = block.min_grad.min(verdict.grad_norm);
            block.max_grad = block.max_grad.max(verdict.grad_norm);
        }

        self.total_steps += 1;
        if verdict.accepted {
            self.consecutive_rejections = 0;
        } else {
            block.rejected += 1;
            self.total_rejected += 1;
            self.consecutive_rejections += 1;
            self.worst_consecutive_rejections = self
                .worst_consecutive_rejections
                .max(self.consecutive_rejections);
            for failure in &verdict.failures {
                if self.failures.len() < MAX_RETAINED_FAILURES {
                    self.failures.push((step, failure.clone()));
                }
            }
        }
    }

    /// Record a failure not attached to any block (preflight, periodic).
    pub fn record_failure(&mut self, step: usize, failure: CheckFailure) {
        self.total_rejected += 1;
        if self.failures.len() < MAX_RETAINED_FAILURES {
            self.failures.push((step, failure));
        }
    }

    pub fn consecutive_rejections(&self) -> usize {
        self.consecutive_rejections
    }

    /// Whether the run should stop: too many steps rejected back to back.
    pub fn should_abort(&self, checks: &TrainingChecks) -> bool {
        checks.max_consecutive_rejections > 0
            && self.consecutive_rejections >= checks.max_consecutive_rejections
    }

    pub fn block(&self, idx: usize) -> Option<&BlockHealth> {
        self.per_block.get(idx)
    }

    pub fn num_blocks(&self) -> usize {
        self.per_block.len()
    }

    pub fn failures(&self) -> &[(usize, CheckFailure)] {
        &self.failures
    }

    /// Blocks that never received a usable gradient.
    ///
    /// A silently dead block is the failure mode block-wise training is most
    /// prone to and least likely to surface on its own.
    pub fn dead_blocks(&self) -> Vec<usize> {
        self.per_block
            .iter()
            .enumerate()
            .filter(|(_, b)| b.steps > 0 && b.max_grad <= 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Per-block table for the end of a run.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<7} {:>7} {:>10} {:>12} {:>12} {:>12} {:>9}\n",
            "block", "steps", "rejected", "mean loss", "mean |g|", "max |g|", "reject%"
        ));
        out.push_str(&"-".repeat(74));
        out.push('\n');
        for (i, b) in self.per_block.iter().enumerate() {
            out.push_str(&format!(
                "{:<7} {:>7} {:>10} {:>12.4} {:>12.3e} {:>12.3e} {:>8.1}%\n",
                i,
                b.steps,
                b.rejected,
                b.mean_loss(),
                b.mean_grad_norm(),
                b.max_grad,
                100.0 * b.rejection_rate()
            ));
        }
        if !self.failures.is_empty() {
            out.push_str(&format!("\nfirst {} failure(s):\n", self.failures.len()));
            for (step, failure) in &self.failures {
                out.push_str(&format!("  step {step}: {failure}\n"));
            }
        }
        out
    }
}

/// L2 norm of every parameter gradient, concatenated.
///
/// Parameters that received no gradient -- layers outside the executed span,
/// which is the normal case in block-wise training -- contribute nothing, so
/// this is the norm of the update that would actually be applied, not of some
/// larger notional one.
pub fn global_grad_norm<B, M>(module: &M, grads: &burn::optim::GradientsParams) -> f32
where
    B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>,
    M: burn::module::AutodiffModule<B>,
{
    let mut visitor = GradNormVisitor::<B> {
        grads,
        sum_sq: 0.0,
        _backend: std::marker::PhantomData,
    };
    module.visit(&mut visitor);
    (visitor.sum_sq as f32).sqrt()
}

struct GradNormVisitor<'a, B: burn::tensor::backend::AutodiffBackend> {
    grads: &'a burn::optim::GradientsParams,
    sum_sq: f64,
    _backend: std::marker::PhantomData<B>,
}

impl<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>> burn::module::ModuleVisitor<B>
    for GradNormVisitor<'_, B>
{
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
            let sq: f32 = grad.powf_scalar(2.0).sum().into_scalar();
            self.sum_sq += sq as f64;
        }
    }
}

/// Number of parameter tensors containing a non-finite value.
///
/// Checked *after* the optimizer step: a NaN that reaches the weights is
/// unrecoverable and poisons every later step, so it must be caught where it
/// happens rather than inferred from a loss curve that went flat.
pub fn non_finite_parameters<B, M>(module: &M) -> usize
where
    B: burn::tensor::backend::Backend<FloatElem = f32>,
    M: burn::module::Module<B>,
{
    let mut visitor = FiniteVisitor { non_finite: 0 };
    module.visit(&mut visitor);
    visitor.non_finite
}

struct FiniteVisitor {
    non_finite: usize,
}

impl<B: burn::tensor::backend::Backend<FloatElem = f32>> burn::module::ModuleVisitor<B>
    for FiniteVisitor
{
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
        // `x - x == 0` for every finite x and NaN for both NaN and infinity,
        // so one reduction detects either without a separate pass.
        let v = param.val();
        let probe: f32 = (v.clone() - v).abs().sum().into_scalar();
        if !probe.is_finite() {
            self.non_finite += 1;
        }
    }
}

/// Gradient-norm quality check: healthy gradients should stay within
/// `[min_norm, max_norm]`; near-zero indicates dead paths, huge norms
/// indicate instability (roadmap item 12.4).
pub fn grad_norm_ok(norm: f32, min_norm: f32, max_norm: f32) -> bool {
    norm.is_finite() && norm >= min_norm && norm <= max_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_all_pass_with_lenient_gate() {
        let device = Default::default();
        let prev = Tensor::<B, 2>::ones([3, 8], &device) * 0.5;
        let new = Tensor::<B, 2>::ones([3, 8], &device);
        let report = evaluate(&QualityGateConfig::lenient(), &prev, &new, None);
        assert!(report.all_passed());
        assert!(close(report.mean_cosine, 1.0, 1e-5));
    }

    #[test]
    fn test_cosine_threshold_rejects_flip() {
        let device = Default::default();
        let prev = Tensor::<B, 2>::ones([2, 4], &device);
        // Second sample flips direction entirely -> cosine -1 on that row.
        let mut new_data = vec![1.0f32; 8];
        for v in new_data[4..].iter_mut() {
            *v = -1.0;
        }
        let new = Tensor::<B, 1>::from_floats(new_data.as_slice(), &device).reshape([2, 4]);

        let report = evaluate(
            &QualityGateConfig { min_cosine: 0.0, ..QualityGateConfig::default() },
            &prev,
            &new,
            None,
        );
        assert!(report.passed[0], "aligned sample must pass");
        assert!(!report.passed[1], "flipped sample must fail");
        assert_eq!(filter_indices(&report), vec![0]);
    }

    #[test]
    fn test_mse_threshold() {
        let device = Default::default();
        let prev = Tensor::<B, 2>::zeros([1, 4], &device);
        let new = Tensor::<B, 2>::full([1, 4], 2.0, &device); // MSE = 4

        let cfg = QualityGateConfig { max_mse: 5.0, ..QualityGateConfig::default() };
        assert!(evaluate(&cfg, &prev, &new, None).all_passed());

        let cfg = QualityGateConfig { max_mse: 3.0, ..QualityGateConfig::default() };
        assert!(!evaluate(&cfg, &prev, &new, None).all_passed());
        assert!(close(evaluate(&cfg, &prev, &new, None).mean_mse, 4.0, 1e-6));
    }

    #[test]
    fn test_confidence_check() {
        let device = Default::default();
        let prev = Tensor::<B, 2>::ones([2, 4], &device);
        let new = Tensor::<B, 2>::ones([2, 4], &device);
        // Row 0 confident, row 1 uniform.
        let probs_data = [0.9f32, 0.05, 0.05, 0.0, 0.25, 0.25, 0.25, 0.25];
        let probs = Tensor::<B, 1>::from_floats(probs_data.as_slice(), &device).reshape([2, 4]);

        let cfg = QualityGateConfig {
            min_confidence: Some(0.5),
            ..QualityGateConfig::default()
        };
        let report = evaluate(&cfg, &prev, &new, Some(&probs));
        assert!(report.passed[0] && !report.passed[1]);
        assert!(close(report.mean_confidence.unwrap(), 0.575, 1e-6));
    }

    #[test]
    fn test_layer_gates_fall_back_to_default() {
        let gates = LayerGates::uniform(QualityGateConfig::strict());
        assert_eq!(gates.for_block(0).min_cosine, 0.5);
        assert_eq!(gates.for_block(99).min_cosine, 0.5, "out-of-range must not panic");

        let mut custom = LayerGates::uniform(QualityGateConfig::lenient());
        custom.per_block = vec![None, Some(QualityGateConfig::strict())];
        assert_eq!(custom.for_block(0).min_cosine, 0.0);
        assert_eq!(custom.for_block(1).min_cosine, 0.5);
    }

    #[test]
    fn test_tightening_gates_are_monotone() {
        // The whole point of per-layer gates is that later (lower-sigma)
        // blocks are held to a stricter standard; assert that ordering holds
        // for every block count rather than trusting the interpolation.
        for num_blocks in [1usize, 2, 3, 8] {
            let gates = LayerGates::tightening(
                num_blocks,
                QualityGateConfig { min_cosine: -0.5, max_mse: 100.0, min_confidence: None },
                QualityGateConfig { min_cosine: 0.9, max_mse: 0.5, min_confidence: None },
            );
            let mut prev_cos = f32::NEG_INFINITY;
            let mut prev_mse = f32::INFINITY;
            for b in 0..num_blocks {
                let g = gates.for_block(b);
                assert!(g.min_cosine >= prev_cos - 1e-6, "cosine gate must tighten");
                assert!(g.max_mse <= prev_mse + 1e-6, "mse gate must tighten");
                prev_cos = g.min_cosine;
                prev_mse = g.max_mse;
            }
            assert!((gates.for_block(num_blocks - 1).min_cosine - 0.9).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gate_ledger_tallies_per_block() {
        let device = Default::default();
        let prev = Tensor::<B, 2>::ones([2, 4], &device);
        let mut new_data = vec![1.0f32; 8];
        for v in new_data[4..].iter_mut() {
            *v = -1.0;
        }
        let new = Tensor::<B, 1>::from_floats(new_data.as_slice(), &device).reshape([2, 4]);
        let report = evaluate(&QualityGateConfig::default(), &prev, &new, None);

        let mut ledger = GateLedger::new(2);
        ledger.record(1, &report);
        assert_eq!(ledger.rejected(0), 0);
        assert_eq!(ledger.rejected(1), 1);
        assert!((ledger.rejection_rate(1) - 0.5).abs() < 1e-6);
        assert_eq!(ledger.rejection_rate(0), 0.0, "unevaluated blocks report 0");
        assert_eq!(ledger.total_rejected(), 1);

        // Recording past the initial size grows the ledger instead of panicking.
        ledger.record(5, &report);
        assert_eq!(ledger.num_blocks(), 6);
        assert_eq!(ledger.rejected(5), 1);
        assert_eq!(ledger.evaluated(5), 2);
    }

    #[test]
    fn test_merging_ledgers_preserves_rates() {
        // Merging per-batch ledgers must average the rates, not inflate them:
        // adding only the rejection counts would report 100% for any block
        // that ever rejected anything.
        let device = Default::default();
        let prev = Tensor::<B, 2>::ones([2, 4], &device);
        let mut flipped = vec![1.0f32; 8];
        for v in flipped[4..].iter_mut() {
            *v = -1.0;
        }
        let half_bad =
            Tensor::<B, 1>::from_floats(flipped.as_slice(), &device).reshape([2, 4]);
        let all_good = Tensor::<B, 2>::ones([2, 4], &device);

        let cfg = QualityGateConfig::default();
        let mut a = GateLedger::new(1);
        a.record(0, &evaluate(&cfg, &prev, &half_bad, None)); // 1 of 2 rejected
        let mut b = GateLedger::new(1);
        b.record(0, &evaluate(&cfg, &prev, &all_good, None)); // 0 of 2 rejected

        assert!((a.rejection_rate(0) - 0.5).abs() < 1e-6);
        assert_eq!(b.rejection_rate(0), 0.0);

        a.merge(&b);
        assert_eq!(a.evaluated(0), 4);
        assert_eq!(a.rejected(0), 1);
        assert!(
            (a.rejection_rate(0) - 0.25).abs() < 1e-6,
            "merged rate should be 1/4, got {}",
            a.rejection_rate(0)
        );

        // Merging a wider ledger grows this one rather than dropping blocks.
        let mut wide = GateLedger::new(4);
        wide.record(3, &evaluate(&cfg, &prev, &half_bad, None));
        a.merge(&wide);
        assert_eq!(a.num_blocks(), 4);
        assert_eq!(a.rejected(3), 1);
    }

    #[test]
    fn test_step_verdict_collects_failures_by_phase() {
        let mut v = StepVerdict::accepted();
        assert!(v.accepted && v.reason().is_none());

        v.reject(TrainingPhase::Loss, "loss is NaN");
        v.reject(TrainingPhase::Gradients, "gradient norm 0 outside range");
        assert!(!v.accepted);
        assert_eq!(v.failures.len(), 2);

        // The reason names the phase, so "loss was NaN" and "the update blew
        // up" are distinguishable in a log.
        let reason = v.reason().unwrap();
        assert!(reason.contains("loss: loss is NaN"), "{reason}");
        assert!(reason.contains("gradients: "), "{reason}");
    }

    #[test]
    fn test_training_health_tracks_blocks_independently() {
        let mut health = TrainingHealth::new(3);

        let mut good = StepVerdict::accepted();
        good.grad_norm = 2.0;
        let mut bad = StepVerdict::accepted();
        bad.grad_norm = 0.0;
        bad.reject(TrainingPhase::Gradients, "dead");

        health.record(0, 0, 1.0, &good);
        health.record(1, 0, 3.0, &good);
        health.record(2, 1, 5.0, &bad);

        let b0 = health.block(0).unwrap();
        assert_eq!(b0.steps, 2);
        assert_eq!(b0.rejected, 0);
        assert!((b0.mean_loss() - 2.0).abs() < 1e-6);
        assert!((b0.mean_grad_norm() - 2.0).abs() < 1e-6);

        let b1 = health.block(1).unwrap();
        assert_eq!(b1.rejected, 1);
        assert!((b1.rejection_rate() - 1.0).abs() < 1e-6);

        // Block 2 was never visited, so it is not "dead" -- only a block that
        // ran and never produced a gradient is.
        assert_eq!(health.dead_blocks(), vec![1]);
        assert_eq!(health.total_steps, 3);
        assert_eq!(health.total_rejected, 1);
        assert!(health.render().contains("block"));
    }

    #[test]
    fn test_non_finite_losses_are_excluded_from_the_mean() {
        // A NaN step must not poison the recorded mean, or the per-block
        // report becomes unreadable exactly when it is most needed.
        let mut health = TrainingHealth::new(1);
        let mut ok = StepVerdict::accepted();
        ok.grad_norm = 1.0;
        health.record(0, 0, 2.0, &ok);
        health.record(1, 0, f32::NAN, &ok);
        assert!(health.block(0).unwrap().mean_loss().is_finite());
    }

    #[test]
    fn test_abort_after_consecutive_rejections() {
        let checks = TrainingChecks { max_consecutive_rejections: 3, ..TrainingChecks::default() };
        let mut health = TrainingHealth::new(1);
        let mut bad = StepVerdict::accepted();
        bad.reject(TrainingPhase::Loss, "nan");
        let good = StepVerdict::accepted();

        health.record(0, 0, 1.0, &bad);
        health.record(1, 0, 1.0, &bad);
        assert!(!health.should_abort(&checks));
        health.record(2, 0, 1.0, &bad);
        assert!(health.should_abort(&checks), "three in a row must abort");

        // One good step resets the streak, because intermittent rejections are
        // normal in block-wise training; only a stuck run is fatal.
        health.record(3, 0, 1.0, &good);
        assert!(!health.should_abort(&checks));
        assert_eq!(health.worst_consecutive_rejections, 3);

        // A zero threshold disables the abort entirely.
        let never = TrainingChecks { max_consecutive_rejections: 0, ..TrainingChecks::default() };
        for step in 4..20 {
            health.record(step, 0, 1.0, &bad);
        }
        assert!(!health.should_abort(&never));
    }

    #[test]
    fn test_retained_failures_are_capped() {
        // A run that rejects every step must not exhaust memory recording it.
        let mut health = TrainingHealth::new(1);
        let mut bad = StepVerdict::accepted();
        bad.reject(TrainingPhase::Loss, "nan");
        for step in 0..500 {
            health.record(step, 0, 1.0, &bad);
        }
        assert_eq!(health.failures().len(), MAX_RETAINED_FAILURES);
        assert_eq!(health.total_rejected, 500, "the count is still exact");
    }

    #[test]
    fn test_checks_presets() {
        let none = TrainingChecks::none();
        assert!(!none.preflight && !none.loss_finite && none.grad_gate.is_none());
        let thorough = TrainingChecks::thorough(10);
        assert_eq!(thorough.verify_every, Some(10));
        assert!(thorough.preflight && thorough.parameters_finite);
    }

    #[test]
    fn test_non_finite_parameters_detects_nan_and_infinity() {
        use crate::dblock::{DblockClassifier, DblockConfig};
        use crate::vit::ViTDiTConfig;
        use burn::module::{Module, Param};

        let device = Default::default();
        let model = DblockClassifier::<B>::new(
            &ViTDiTConfig::tiny(10),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        assert_eq!(non_finite_parameters(&model), 0, "a fresh model is finite");

        for poison in [f32::NAN, f32::INFINITY] {
            let mut record = model.clone().into_record();
            let shape = record.model.vit.embeddings.label_embeddings.weight.shape();
            record.model.vit.embeddings.label_embeddings.weight =
                Param::from_tensor(Tensor::<B, 2>::full(shape, poison, &device));
            let broken = DblockClassifier::<B>::new(
                &ViTDiTConfig::tiny(10),
                &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
                &device,
            )
            .load_record(record);
            assert_eq!(
                non_finite_parameters(&broken),
                1,
                "{poison} must be detected"
            );
        }
    }

    #[test]
    fn test_grad_norm_gate() {
        assert!(grad_norm_ok(1.0, 1e-6, 100.0));
        assert!(!grad_norm_ok(0.0, 1e-6, 100.0), "dead gradient rejected");
        assert!(!grad_norm_ok(f32::NAN, 0.0, 100.0));
        assert!(!grad_norm_ok(1e9, 0.0, 100.0));
    }
}
