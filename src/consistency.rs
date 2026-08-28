//! Consistency training objectives (roadmap Phase 3, plus the cross-fork term
//! of 2.7).
//!
//! Four complementary residuals encourage blocks and repeated evaluations to
//! agree, which is what makes multi-block inference legitimate:
//!
//! - **Boundary consistency** (3.1): blocks `b` and `b + 1` evaluated at their
//!   shared boundary sigma must produce the same x0 estimate.
//! - **Self-consistency** (3.2): the same block evaluated at two different
//!   noise levels of the *same* clean state must predict the same x0.
//! - **Trajectory consistency** (3.3): a full-chain Euler rollout from
//!   `sigma_max` must land where a *direct* noising of the clean data at the
//!   final sigma lands, as judged by the last block.
//! - **Cross-fork consistency** (2.7): for a non-adjacent pair `(i, j)` with
//!   `j >= i + 2`, the joint span `i..=j` executed in one shot must reproduce
//!   the sequential composition of those blocks. This is exactly the property
//!   `Strategy::Parallel` relies on, so it is trained rather than assumed.
//!
//! All four are plain MSE residuals combined as
//! `total = task_loss + sum_k w_k * lambda(step) * L_k`, where `lambda` comes
//! from [`ConsistencySchedule`] (item 3.5). The multiplier deliberately does
//! not depend on `L_k` itself: scaling a residual by its own value optimizes
//! `L_k^2`, whose gradient vanishes precisely where the residual is already
//! small.

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

/// Weights of the four consistency terms.
#[derive(Debug, Clone, Copy)]
pub struct ConsistencyWeights {
    pub boundary: f64,
    pub self_consistency: f64,
    pub trajectory: f64,
    /// Fork-reconverge agreement between a joint span and the sequential
    /// composition of the same blocks (roadmap 2.7).
    pub cross_fork: f64,
}

impl Default for ConsistencyWeights {
    fn default() -> Self {
        Self {
            boundary: 1.0,
            self_consistency: 0.5,
            trajectory: 0.25,
            cross_fork: 0.25,
        }
    }
}

impl ConsistencyWeights {
    /// Only the boundary term (cheapest configuration).
    pub fn boundary_only() -> Self {
        Self { boundary: 1.0, self_consistency: 0.0, trajectory: 0.0, cross_fork: 0.0 }
    }

    /// Every term disabled: `consistency_step` reduces exactly to
    /// [`crate::dblock::DblockClassifier::training_step`].
    pub fn none() -> Self {
        Self { boundary: 0.0, self_consistency: 0.0, trajectory: 0.0, cross_fork: 0.0 }
    }
}

/// Configuration for [`DblockClassifier::consistency_step`].
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
    /// Fork-reconverge residual; `0.0` when the term is disabled or the model
    /// has too few blocks to form a non-adjacent pair.
    pub cross_fork_loss: f32,
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
        //
        // Every term below is a plain MSE residual scaled by
        // `weight * lambda`. Scaling by the residual's own *value* would
        // silently optimize `L^2` (gradient `2 w lambda L dL`), which
        // vanishes exactly where the term is already small and explodes where
        // it is large -- so the multiplier is kept independent of `l`.
        let bounds = self.sampler(config.gamma).block_sigmas;
        let lambda = config.schedule.weight_at(step);
        let n_blocks = self.num_blocks();
        let mut m_boundary = 0.0f32;
        let mut m_self = 0.0f32;
        let mut m_traj = 0.0f32;
        let mut m_fork = 0.0f32;

        if config.weights.boundary > 0.0 && n_blocks >= 2 {
            // Evaluate both neighbors exactly at their shared boundary sigma.
            let lo_blk = block_idx.min(n_blocks - 2);
            let shared = crate::sigma::shared_boundary_sigma(&bounds, lo_blk);
            let eps_b = Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
            let zt_b = z.clone() + eps_b * shared;
            let lo = self.x0_estimate(pixel_values, &zt_b, shared, Some(self.layer_range(lo_blk)));
            let hi =
                self.x0_estimate(pixel_values, &zt_b, shared, Some(self.layer_range(lo_blk + 1)));
            let l = mse(&lo, &hi);
            m_boundary = l.clone().into_scalar();
            loss = loss + l.mul_scalar((config.weights.boundary * lambda) as f32);
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
            loss = loss + l.mul_scalar((config.weights.self_consistency * lambda) as f32);
        }

        if config.weights.trajectory > 0.0 && n_blocks >= 2 {
            // Roll the whole chain downwards through the block boundaries.
            // `bounds` ascends and block 0 owns the noisiest window, so block
            // `b` steps from bounds[n - b] down to bounds[n - b - 1].
            let start_sigma = bounds[n_blocks];
            let mut z_roll = z.clone()
                + Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device)
                    * start_sigma;
            for blk in 0..n_blocks - 1 {
                let (next, sigma) = crate::sigma::block_window(&bounds, blk);
                let x0r =
                    self.x0_estimate(pixel_values, &z_roll, sigma, Some(self.layer_range(blk)));
                z_roll = crate::multi_block::euler_step(sigma, next, &z_roll, &x0r);
            }
            // The chain must land where a *direct* noising of the clean data
            // at the same sigma would: same block, same noise level, one
            // latent reached by integration and one by construction.
            let end_sigma = bounds[1];
            let last = self.layer_range(n_blocks - 1);
            let target =
                self.x0_estimate(pixel_values, &z_roll, end_sigma, Some(last.clone()));
            let z_direct = z.clone()
                + Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device)
                    * end_sigma;
            let direct = self.x0_estimate(pixel_values, &z_direct, end_sigma, Some(last));
            let l = mse(&direct, &target);
            m_traj = l.clone().into_scalar();
            loss = loss + l.mul_scalar((config.weights.trajectory * lambda) as f32);
        }

        // Cross-fork consistency (roadmap 2.7): a *non-adjacent* block pair
        // (i, j) with j >= i + 2 is trained against itself by requiring the
        // joint span i..=j -- what `Strategy::Parallel` actually executes --
        // to reproduce the sequential composition of the same blocks. This is
        // the property that makes forking legitimate, so it is trained
        // directly rather than hoped for.
        if config.weights.cross_fork > 0.0 && n_blocks >= 3 {
            let i = rng.random_range(0..n_blocks - 2);
            let j = rng.random_range(i + 2..n_blocks);

            let (_, sigma_start) = crate::sigma::block_window(&bounds, i);
            let (sigma_end, _) = crate::sigma::block_window(&bounds, j);
            let z_fork = z.clone()
                + Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device)
                    * sigma_start;

            // Sequential path: one block per boundary interval.
            let mut z_seq = z_fork.clone();
            for blk in i..=j {
                let (next, sigma) = crate::sigma::block_window(&bounds, blk);
                let x0b =
                    self.x0_estimate(pixel_values, &z_seq, sigma, Some(self.layer_range(blk)));
                z_seq = crate::multi_block::euler_step(sigma, next, &z_seq, &x0b);
            }

            // Forked path: the whole span in one shot, one step to the end.
            let span = self.layer_range(i).start..self.layer_range(j).end;
            let x0_par = self.x0_estimate(pixel_values, &z_fork, sigma_start, Some(span));
            let z_par = crate::multi_block::euler_step(sigma_start, sigma_end, &z_fork, &x0_par);

            let l = mse(&z_seq, &z_par);
            m_fork = l.clone().into_scalar();
            loss = loss + l.mul_scalar((config.weights.cross_fork * lambda) as f32);
        }

        let metrics = ConsistencyMetrics {
            loss: loss.clone().into_scalar(),
            ce_loss: ce_loss.into_scalar(),
            boundary_loss: m_boundary,
            self_loss: m_self,
            trajectory_loss: m_traj,
            cross_fork_loss: m_fork,
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
