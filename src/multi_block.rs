//! Multi-block denoising strategies (roadmap Phases 2 and 10).
//!
//! The sigma schedule is partitioned into windows; each window selects a
//! contiguous span of transformer blocks to execute:
//!
//! - [`Strategy::Sequential`]: one block per window (original DiffusionBlocks).
//! - [`Strategy::Parallel`]: `k` adjacent blocks jointly per window. During
//!   training, running only a span is simultaneously gradient routing: no
//!   gradient flows through layers outside the executed span.
//! - [`Strategy::Hybrid`]: sequential above a fraction of `sigma_max`,
//!   parallel below it (coarse structure from single blocks, fine detail
//!   from joint spans).
//! - [`Strategy::Adaptive`]: starts with one block and widens the span while
//!   the class confidence of the x0 estimate stays below a threshold.
//!
//! [`Gated`] wraps any strategy with the Phase-12 quality gates: samples
//! whose update looks degenerate keep their previous latent.

use crate::{
    dblock::DblockClassifier,
    quality::{self, QualityGateConfig},
    sigma::{SIGMA_MAX, SIGMA_MIN, P_MEAN, P_STD},
    solver::SolverKind,
};
use burn::tensor::{Tensor, backend::Backend};
use rand::Rng;

/// How many transformer layers each window executes.
#[derive(Debug, Clone, Copy, Default)]
pub enum Strategy {
    #[default]
    Sequential,
    Parallel { k: usize },
    /// Sequential while `sigma > warmup_frac * sigma_max`, then Parallel{k}.
    Hybrid { k: usize, warmup_frac: f64 },
    /// Start at one block; widen up to `k_max` while confidence is low.
    Adaptive { k_max: usize, conf_threshold: f32 },
}

/// Strategy + optional quality gate wrapper.
#[derive(Debug, Clone, Default)]
pub struct Gated {
    pub inner: Strategy,
    pub gate: QualityGateConfig,
}

/// Configuration of a multi-block sampling run.
#[derive(Debug, Clone, Default)]
pub struct MultiBlockConfig {
    pub strategy: Gated,
    pub solver: SolverKind,
    /// Number of inference windows; defaults to the model's `num_blocks`.
    pub num_steps: Option<usize>,
}

/// Outcome statistics of a sampling run.
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingStats {
    /// Denoiser invocations performed.
    pub model_calls: usize,
    /// Samples rejected by the quality gate at least once.
    pub gated_samples: usize,
}

impl<B: Backend<FloatElem = f32>> DblockClassifier<B> {
    /// Multi-block Euler-integrated sampling returning final logits plus
    /// statistics. Generalizes [`Self::diffusion_step`] with selectable block
    /// spans, solvers, and quality gating.
    pub fn sample_multi_block(
        &self,
        pixel_values: &Tensor<B, 4>,
        config: &MultiBlockConfig,
        _rng: &mut impl Rng,
    ) -> (Tensor<B, 2>, SamplingStats) {
        let num_blocks = self.num_blocks();
        let steps = config.num_steps.unwrap_or(num_blocks);
        let schedule =
            crate::sigma::discrete_sigmas_dblock(steps, SIGMA_MIN, SIGMA_MAX, P_MEAN, P_STD);
        let b = pixel_values.dims()[0];
        let h_dim = self.model().label_embedding_weight().dims()[1];

        // Initial latent N(0, I) scaled to sqrt(1 + sigma_0^2), matching
        // diffusion_step's convention.
        let s0 = schedule[0];
        let mut z = Tensor::<B, 2>::random(
            [b, h_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &pixel_values.device(),
        )
        .mul_scalar((1.0 + s0 * s0).sqrt() as f32);

        let mut stats = SamplingStats::default();
        let mut ever_gated = vec![false; b];
        let mut prev_x0: Option<Tensor<B, 2>> = None;
        let mut current_k: usize = 1;

        for window in schedule.windows(2) {
            let (sigma, s_next) = (window[0], window[1]);

            let base_block = crate::sigma::estimate_target_layer(&self.block_bounds(), &[sigma]);
            let span = self.select_span(&config.strategy.inner, base_block, current_k, sigma);
            let want_probs =
                matches!(config.strategy.inner, Strategy::Adaptive { .. });

            let (x0, probs) = self.x0_estimate_probs(pixel_values, &z, sigma, Some(span), want_probs);
            stats.model_calls += 1;

            // Adaptive depth: widen while the estimate looks unconfident.
            if let (Strategy::Adaptive { conf_threshold, k_max }, Some(p)) =
                (&config.strategy.inner, &probs)
            {
                let per_sample: Vec<f32> = p
                    .clone()
                    .max_dim(1)
                    .into_data()
                    .convert::<f32>()
                    .iter::<f32>()
                    .collect();
                let mean_conf = per_sample.iter().sum::<f32>() / per_sample.len().max(1) as f32;
                if mean_conf < *conf_threshold {
                    current_k = (current_k + 1).min((*k_max).max(1));
                }
            }

            let new_z = euler_step(sigma, s_next, &z, &x0);
            match &prev_x0 {
                None => z = new_z,
                Some(prev) => {
                    let report = quality::evaluate(&config.strategy.gate, prev, &x0, probs.as_ref());
                    for (i, ok) in report.passed.iter().enumerate() {
                        if !ok {
                            ever_gated[i] = true;
                        }
                    }
                    z = merge_kept(&new_z, &z, &report.passed);
                }
            }
            prev_x0 = Some(x0);
        }

        stats.gated_samples = ever_gated.iter().filter(|&&g| g).count();
        let min_sigma = *schedule.last().expect("non-empty schedule");
        let logits = self.denoise(pixel_values.clone(), z, &vec![min_sigma; b], None);
        (logits, stats)
    }

    /// Contiguous layer span for one window under the given strategy.
    fn select_span(
        &self,
        strategy: &Strategy,
        base_block: usize,
        current_k: usize,
        sigma: f64,
    ) -> std::ops::Range<usize> {
        let n = self.num_blocks();
        let k = match strategy {
            Strategy::Sequential => 1,
            Strategy::Parallel { k } => *k,
            Strategy::Hybrid { k, warmup_frac } => {
                if sigma > warmup_frac * SIGMA_MAX {
                    1
                } else {
                    *k
                }
            }
            Strategy::Adaptive { .. } => current_k,
        };
        let start = base_block.min(n.saturating_sub(1));
        let end_block = (start + k.max(1)).min(n).max(start + 1);
        self.layer_range(start).start..self.layer_range(end_block - 1).end
    }
}

/// Single Euler step of dz/dsigma = (x0 - z)/sigma toward `s_next`.
pub(crate) fn euler_step<B: Backend>(
    s: f64,
    s_next: f64,
    z: &Tensor<B, 2>,
    x0: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let d = (z.clone() - x0.clone()) / s;
    z.clone() + (s_next - s) * d
}

/// Keep old latent rows where `keep[i] == false`.
fn merge_kept<B: Backend>(new: &Tensor<B, 2>, old: &Tensor<B, 2>, keep: &[bool]) -> Tensor<B, 2> {
    let device = new.device();
    let mask_f: Vec<f32> = keep.iter().map(|&k| if k { 1.0 } else { 0.0 }).collect();
    let mask = Tensor::<B, 1>::from_floats(mask_f.as_slice(), &device).unsqueeze_dim::<2>(1);
    new.clone() * mask.clone() + old.clone() * (mask.neg() + 1.0)
}
