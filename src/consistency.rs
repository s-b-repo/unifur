//! Consistency training objectives (roadmap Phase 3).
//!
//! Three complementary losses encourage adjacent blocks and repeated
//! evaluations to agree, which stabilizes multi-block inference:
//!
//! - **Boundary consistency** (3.1): the denoised x0 estimate of block `b`
//!   at its upper window edge must match that of block `b + 1` evaluated at
//!   the same sigma — blocks must agree at their shared boundary.
//! - **Self-consistency** (3.2): the same block evaluated at two different
//!   noise levels of the *same* clean state must predict the same x0.
//! - **Trajectory consistency** (3.3): a cheap full-chain Euler rollout must
//!   end where a single low-sigma block evaluation starts from the rolled
//!   state.
//!
//! [`ConsistencySchedule`] implements the weight scheduling of item 3.5.

use crate::dblock::DblockClassifier;
use burn::tensor::{Int, Tensor, backend::Backend};
use rand::Rng;

/// Weight schedule for combining consistency terms with the main loss.
#[derive(Debug, Clone, Copy)]
pub enum ConsistencySchedule {
    /// Constant weight (no ramp).
    Constant { weight: f64 },
    /// Linear ramp from `start` to `end` over `total_steps`.
    Linear { start: f64, end: f64, total_steps: usize },
    /// Cosine ramp 0 -> 1 over `total_steps` (smooth warmup).
    Cosine { total_steps: usize },
}

impl Default for ConsistencySchedule {
    fn default() -> Self {
        Self::Constant { weight: 0.1 }
    }
}

impl ConsistencySchedule {
    pub fn weight_at(&self, step: usize) -> f64 {
        match *self {
            Self::Constant { weight } => weight,
            Self::Linear { start, end, total_steps } => {
                let t = ((step as f64) / (total_steps.max(2) - 1) as f64).min(1.0);
                start + (end - start) * t
            }
            Self::Cosine { total_steps } => {
                let t = ((step as f64) / (total_steps.max(2) - 1) as f64).min(1.0);
                0.5 - 0.5 * (std::f64::consts::PI * t).cos()
            }
        }
    }
}

/// Weights of the three consistency terms.
#[derive(Debug, Clone, Copy)]
pub struct ConsistencyWeights {
    pub boundary: f64,
    pub self_consistency: f64,
    pub trajectory: f64,
}

impl Default for ConsistencyWeights {
    fn default() -> Self {
        Self { boundary: 1.0, self_consistency: 0.5, trajectory: 0.25 }
    }
}

/// Configuration for [`consistency_step`].
#[derive(Debug, Clone, Copy)]
pub struct ConsistencyConfig {
    pub gamma: f64,
    pub weights: ConsistencyWeights,
    pub schedule: ConsistencySchedule,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            gamma: 0.05,
            weights: ConsistencyWeights::default(),
            schedule: ConsistencySchedule::default(),
        }
    }
}

/// Metrics of one consistency-augmented step.
#[derive(Debug, Clone, Copy)]
pub struct ConsistencyMetrics {
    pub loss: f32,
    pub ce_loss: f32,
    pub boundary_loss: f32,
    pub self_loss: f32,
    pub trajectory_loss: f32,
    pub weight: f64,
    pub block_idx: usize,
}

impl<B: Backend<FloatElem = f32>> DblockClassifier<B> {
    /// One training step = standard EDM-weighted cross-entropy plus the
    /// enabled consistency terms (all differentiable).
    pub fn consistency_step<R: Rng>(
        &self,
        pixel_values: &Tensor<B, 4>,
        labels: Tensor<B, 1, Int>,
        config: &ConsistencyConfig,
        step: usize,
        rng: &mut R,
    ) -> (Tensor<B, 1>, ConsistencyMetrics) {
        use burn::tensor::{activation::log_softmax, Distribution};

        let device = pixel_values.device();
        let b = pixel_values.dims()[0];

        // --- Standard dblock loss on a random block ----------------------
        let z = self.model().normalized_label_embeds(labels.clone());
        let block_idx = rng.random_range(0..self.num_blocks());
        let sigmas = self.sampler(config.gamma).sample(rng, block_idx, b);

        let eps =
            Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
        let s_t = Tensor::<B, 1>::from_floats(
            sigmas.iter().map(|&v| v as f32).collect::<Vec<_>>().as_slice(),
            &device,
        );
        let zt = z.clone() + eps * s_t.unsqueeze_dim::<2>(1);

        let logits = self.denoise(pixel_values.clone(), zt.clone(), &sigmas, Some(block_idx));
        let log_probs = log_softmax(logits, 1);
        let nll = -log_probs.gather(1, labels.unsqueeze_dim::<2>(1)).squeeze_dim::<1>(1);
        let ce_loss = nll.clone().mean();

        let weights: Vec<f32> = sigmas
            .iter()
            .map(|&sg| crate::sigma::edm_loss_weight(sg, self.sigma_data()) as f32)
            .collect();
        let w = Tensor::<B, 1>::from_floats(weights.as_slice(), &device);
        let mut loss = (nll * w).mean();

        // --- Consistency terms -------------------------------------------
        let bounds = self.sampler(config.gamma).block_sigmas;
        let lambda = config.schedule.weight_at(step);
        let mut m_boundary = 0.0f32;
        let mut m_self = 0.0f32;
        let mut m_traj = 0.0f32;

        if config.weights.boundary > 0.0 && self.num_blocks() >= 2 {
            // Evaluate both neighbors exactly at their shared boundary sigma.
            let lo_blk = block_idx.min(self.num_blocks() - 2);
            let shared = bounds[lo_blk + 1];
            let eps_b = Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
            let zt_b = z.clone() + eps_b * shared;
            let lo = self.x0_estimate(pixel_values, &(zt_b.clone()), shared, Some(self.layer_range(lo_blk)));
            let hi = self.x0_estimate(pixel_values, &(zt_b), shared, Some(self.layer_range(lo_blk + 1)));
            let l = mse(&lo, &hi);
            m_boundary = l.clone().into_scalar();
            loss = loss + l.mul_scalar((config.weights.boundary * lambda * m_boundary as f64) as f32);
        }

        if config.weights.self_consistency > 0.0 {
            // Same block, two noise levels of the SAME clean embedding.
            let (lo_s, hi_s) = self.window_pair(block_idx, config.gamma);
            let e1 = Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
            let e2 = Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
            let z_lo = z.clone() + e1 * lo_s;
            let z_hi = z.clone() + e2 * hi_s;
            let span = self.layer_range(block_idx);
            let p_lo = self.x0_estimate(pixel_values, &z_lo, lo_s, Some(span.clone()));
            let p_hi = self.x0_estimate(pixel_values, &z_hi, hi_s, Some(span));
            let l = mse(&p_lo, &p_hi);
            m_self = l.clone().into_scalar();
            loss = loss + l.mul_scalar((config.weights.self_consistency * lambda * m_self as f64) as f32);
        }

        if config.weights.trajectory > 0.0 && self.num_blocks() >= 2 {
            // Roll the chain with no_grad-style detached estimates (we detach
            // by not backpropagating through the rollout: each rollout step's
            // contribution is dropped because we only train the final block).
            let start_sigma = bounds[bounds.len() - 1];
            let mut z_roll = z.clone()
                + Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device)
                    * start_sigma;
            let frac = (start_sigma / bounds[1]).powf(1.0 / self.num_blocks() as f64);
            let mut sigma = start_sigma;
            for blk in 0..self.num_blocks() - 1 {
                let next = (sigma * frac).max(bounds[1]);
                let x0r = self.x0_estimate(pixel_values, &z_roll, sigma, Some(self.layer_range(blk)));
                z_roll = crate::multi_block::euler_step(sigma, next, &z_roll, &x0r);
                sigma = next;
            }
            // Target: final block at the chain's endpoint...
            let target = self.x0_estimate(pixel_values, &z_roll, sigma, Some(self.layer_range(self.num_blocks() - 1)));
            // ...vs the trained block evaluated directly at its own window.
            let direct_sig = sigmas[sigmas.len().saturating_sub(1)];
            let direct = self.x0_estimate(pixel_values, &zt, direct_sig, Some(self.layer_range(self.num_blocks() - 1)));
            let l = mse(&direct, &target);
            m_traj = l.clone().into_scalar();
            loss = loss + l.mul_scalar((config.weights.trajectory * lambda * m_traj as f64) as f32);
        }

        let metrics = ConsistencyMetrics {
            loss: loss.clone().into_scalar(),
            ce_loss: ce_loss.into_scalar(),
            boundary_loss: m_boundary,
            self_loss: m_self,
            trajectory_loss: m_traj,
            weight: lambda,
            block_idx,
        };
        (loss, metrics)
    }

    /// A representative (lo, hi) sigma pair inside one extended window.
    fn window_pair(&self, block_idx: usize, gamma: f64) -> (f64, f64) {
        let (lo, hi) = self.sampler(gamma).extended_window(block_idx);
        let mid = ((hi.ln() + lo.ln()) / 2.0).exp();
        (mid, hi)
    }
}

/// Mean squared error between two `[b, h]` tensors, reduced to scalar.
fn mse<B: Backend>(a: &Tensor<B, 2>, b: &Tensor<B, 2>) -> Tensor<B, 1> {
    (a.clone() - b.clone()).powf_scalar(2.0).mean()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_schedule() {
        let s = ConsistencySchedule::Constant { weight: 0.3 };
        assert_eq!(s.weight_at(0), 0.3);
        assert_eq!(s.weight_at(9999), 0.3);
    }

    #[test]
    fn test_linear_ramp_endpoints() {
        let s = ConsistencySchedule::Linear { start: 0.0, end: 1.0, total_steps: 10 };
        assert!((s.weight_at(0)).abs() < 1e-12);
        assert!((s.weight_at(10) - 1.0).abs() < 1e-12);
        assert!((s.weight_at(5) - 5.0 / 9.0).abs() < 1e-12);
        // Clamped beyond the horizon.
        assert!((s.weight_at(50) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_ramp_bounds_and_monotonicity() {
        let s = ConsistencySchedule::Cosine { total_steps: 8 };
        assert!(s.weight_at(0).abs() < 1e-12);
        // Strictly increasing within the ramp horizon...
        let mut prev = -1.0;
        for i in 0..=7 {
            let w = s.weight_at(i);
            assert!(w > prev, "cosine ramp must be monotonic");
            prev = w;
        }
        // ...and clamped afterwards.
        assert!((s.weight_at(8) - 1.0).abs() < 1e-12);
        assert!((s.weight_at(20) - 1.0).abs() < 1e-12);
    }
}
