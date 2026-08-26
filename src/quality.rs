//! Quality gates (roadmap Phase 12): per-step sanity checks on denoised
//! outputs with batch filtering, used by the multi-block inference strategies
//! to reject bad updates.
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
    fn test_grad_norm_gate() {
        assert!(grad_norm_ok(1.0, 1e-6, 100.0));
        assert!(!grad_norm_ok(0.0, 1e-6, 100.0), "dead gradient rejected");
        assert!(!grad_norm_ok(f32::NAN, 0.0, 100.0));
        assert!(!grad_norm_ok(1e9, 0.0, 100.0));
    }
}
