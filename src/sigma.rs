//! Noise schedules for DiffusionBlocks, ported from `dblock_modules.py`, plus
//! the EDM preconditioning and loss weighting used by `model.py`.
//!
//! # Reference semantics
//!
//! - [`block_sigmas`]: partition `[sigma_min, sigma_max]` into `num_blocks`
//!   contiguous windows whose boundaries are equally spaced in the CDF of a
//!   lognormal distribution (`p_mean=-1.2`, `p_std=1.2`), matching
//!   `get_block_sigmas`.
//! - [`discrete_sigmas_dblock`]: the inference schedule (equally spaced in
//!   lognormal-CDF space, descending order), matching `get_discrete_sigmas(dblock=True)`.
//! - [`discrete_sigmas_edm`]: the classic EDM polynomial schedule with
//!   exponent `rho=7`.
//! - [`DblockSigmaSampler`]: per-sample truncated lognormal sigma sampling inside
//!   a block window extended by `gamma`, matching `get_sigmas`.

use crate::stats::{norm_cdf, norm_ppf};
use rand::Rng;

/// Default `sigma_min` of the EDM schedule.
pub const SIGMA_MIN: f64 = 0.002;
/// Default `sigma_max` of the EDM schedule.
pub const SIGMA_MAX: f64 = 80.0;
/// Mean of the lognormal noise-level prior.
pub const P_MEAN: f64 = -1.2;
/// Std of the lognormal noise-level prior.
pub const P_STD: f64 = 1.2;
/// Rho exponent of the EDM polynomial schedule.
pub const RHO: f64 = 7.0;

/// Block boundaries: ascending sigmas `[sigma_0 < ... < sigma_B]` equally
/// spaced in lognormal-CDF space between `sigma_min` and `sigma_max`
/// (`get_block_sigmas(num_layers=num_blocks)`). Length is `num_blocks + 1`.
///
/// # Block indexing
///
/// The returned vector ascends, but *block* indices descend in noise level:
/// the composition `y = H_{B-1} o ... o H_0(x)` integrates the reverse ODE, so
/// block 0 runs first and therefore owns the **noisiest** window. Concretely,
/// block `b` covers
///
/// ```text
/// (block_sigmas[B - b - 1], block_sigmas[B - b]]
/// ```
///
/// so block `0` covers `(.., sigma_max]` and block `B - 1` covers
/// `(sigma_min, ..]`. [`block_window`] does that arithmetic,
/// [`estimate_target_layer`] is its inverse, and
/// [`DblockSigmaSampler::extended_window`] applies the `gamma` extension on
/// top. Everything downstream -- training-sigma sampling, boundary
/// consistency, multi-block span selection -- must agree on this convention,
/// which `test_block_index_roundtrip` pins.
pub fn block_sigmas(num_blocks: usize) -> Vec<f64> {
    block_sigmas_with(num_blocks, SIGMA_MIN, SIGMA_MAX, P_MEAN, P_STD)
}

/// Half-open sigma window `(lo, hi]` owned by block `b` under the
/// block-0-is-noisiest convention documented on [`block_sigmas`].
///
/// `bounds` must be the ascending boundary vector of length `num_blocks + 1`.
pub fn block_window(bounds: &[f64], block_idx: usize) -> (f64, f64) {
    let n = bounds.len() - 1;
    assert!(n >= 1, "bounds must contain num_blocks + 1 entries");
    assert!(block_idx < n, "block_idx {block_idx} out of range ({n} blocks)");
    (bounds[n - block_idx - 1], bounds[n - block_idx])
}

/// Sigma shared by blocks `b` and `b + 1` (their common window edge).
pub fn shared_boundary_sigma(bounds: &[f64], block_idx: usize) -> f64 {
    let n = bounds.len() - 1;
    assert!(block_idx + 1 < n, "blocks {block_idx} and {} do not both exist", block_idx + 1);
    bounds[n - block_idx - 1]
}

/// [`block_sigmas`] with explicit schedule hyperparameters.
pub fn block_sigmas_with(
    num_blocks: usize,
    sigma_min: f64,
    sigma_max: f64,
    p_mean: f64,
    p_std: f64,
) -> Vec<f64> {
    assert!(num_blocks >= 1, "num_blocks must be >= 1");
    let cdf_min = norm_cdf((sigma_min.ln() - p_mean) / p_std);
    let cdf_max = norm_cdf((sigma_max.ln() - p_mean) / p_std);
    (0..=num_blocks)
        .map(|i| {
            let p = cdf_min + (cdf_max - cdf_min) * (i as f64 / num_blocks as f64);
            (p_mean + p_std * norm_ppf(p)).exp()
        })
        .collect()
}

/// Discrete inference schedule for dblock models: `num_steps` points equally
/// spaced in lognormal-CDF space, returned in *descending* order
/// (`get_discrete_sigmas(num_steps, dblock=True)`).
pub fn discrete_sigmas_dblock(
    num_steps: usize,
    sigma_min: f64,
    sigma_max: f64,
    p_mean: f64,
    p_std: f64,
) -> Vec<f64> {
    let cdf_min = norm_cdf((sigma_min.ln() - p_mean) / p_std);
    let cdf_max = norm_cdf((sigma_max.ln() - p_mean) / p_std);
    let mut sigmas: Vec<f64> = (0..num_steps)
        .map(|i| {
            let p = cdf_min + if num_steps > 1 {
                (cdf_max - cdf_min) * (i as f64 / (num_steps - 1) as f64)
            } else {
                0.0
            };
            (p_mean + p_std * norm_ppf(p)).exp()
        })
        .collect();
    sigmas.reverse(); // torch.flip: ascending -> descending
    sigmas
}

/// Classic EDM polynomial inference schedule (descending):
/// `sigmas_i = (max^(1/rho) + ramp_i * (min^(1/rho) - max^(1/rho)))^rho`.
pub fn discrete_sigmas_edm(num_steps: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> Vec<f64> {
    let min_inv_rho = sigma_min.powf(1.0 / rho);
    let max_inv_rho = sigma_max.powf(1.0 / rho);
    (0..num_steps)
        .map(|i| {
            let ramp = i as f64 / (num_steps - 1).max(1) as f64;
            (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)).powf(rho)
        })
        .collect()
}

/// Samples per-example training sigmas for one block window
/// (`ViTDBlockModel.get_sigmas`).
///
/// The window `[sigma_min_block, sigma_max_block]` is extended by `gamma` on
/// each side in log space (clamped to the global bounds), then sigmas are
/// drawn uniformly in the lognormal CDF within that range.
#[derive(Debug, Clone)]
pub struct DblockSigmaSampler {
    /// Ascending block boundaries, length `num_blocks + 1`.
    pub block_sigmas: Vec<f64>,
    /// Sigma-range extension factor (`gamma`).
    pub gamma: f64,
    pub p_mean: f64,
    pub p_std: f64,
}

impl DblockSigmaSampler {
    pub fn new(num_blocks: usize, gamma: f64) -> Self {
        Self {
            block_sigmas: block_sigmas(num_blocks),
            gamma,
            p_mean: P_MEAN,
            p_std: P_STD,
        }
    }

    /// Extend block `block_idx`'s window by `gamma` in log space and clamp to
    /// the global bounds.
    ///
    /// Uses the block-0-is-noisiest convention of [`block_sigmas`], so this is
    /// the exact window the block is *trained* on and the inverse of
    /// [`estimate_target_layer`] up to the `gamma` extension.
    pub fn extended_window(&self, block_idx: usize) -> (f64, f64) {
        let n = self.block_sigmas.len() - 1;
        let (mut lo, mut hi) = block_window(&self.block_sigmas, block_idx);
        if self.gamma > 0.0 {
            let log_range = hi.ln() - lo.ln();
            lo = (lo.ln() - self.gamma * log_range).exp();
            hi = (hi.ln() + self.gamma * log_range).exp();
            // NOTE: reference clamps to block_sigmas[0] and [-1], which are the
            // *global* min/max of the whole trajectory.
            lo = lo.max(self.block_sigmas[0]);
            hi = hi.min(self.block_sigmas[n]);
        }
        (lo, hi)
    }

    /// Draw `n_samples` sigmas uniform-in-CDF inside the (extended) window.
    pub fn sample<R: Rng>(&self, rng: &mut R, block_idx: usize, n_samples: usize) -> Vec<f64> {
        let (lo, hi) = self.extended_window(block_idx);
        let cdf_lo = norm_cdf((lo.ln() - self.p_mean) / self.p_std);
        let cdf_hi = norm_cdf((hi.ln() - self.p_mean) / self.p_std);
        (0..n_samples)
            .map(|_| {
                let u: f64 = rng.random_range(cdf_lo..cdf_hi);
                (self.p_mean + self.p_std * norm_ppf(u)).exp()
            })
            .collect()
    }
}

/// Maps per-sample sigmas to their target block index
/// (`estimate_target_layer`). Each sigma is bucketized against the ascending
/// block boundaries (`torch.bucketize(..., right=True)` semantics), the raw
/// window index is reversed so that **block 0 sees the noisiest states**
/// (it holds the first transformer layers), and the majority block across
/// the batch is returned.
pub fn estimate_target_layer(block_bounds: &[f64], sigmas: &[f64]) -> usize {
    let n = block_bounds.len() - 1;
    assert!(n >= 1, "block_bounds must contain num_blocks + 1 entries");
    let mut counts = vec![0usize; n];
    for &s in sigmas {
        // First index whose boundary is >= s (right=True bucketize).
        let idx = block_bounds.partition_point(|&b| b < s);
        // Window index containing s.
        let raw = idx.saturating_sub(1).min(n - 1);
        // Reverse so block 0 == noisiest window.
        let block = n - 1 - raw;
        counts[block] += 1;
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// EDM preconditioning coefficients for `sigma_data` normalization
/// (Karras et al., 2022), as used in `denoise`.
#[derive(Debug, Clone, Copy)]
pub struct EdmPreconditioning {
    pub c_skip: f64,
    pub c_out: f64,
    pub c_in: f64,
    pub c_noise: f64,
}

impl EdmPreconditioning {
    pub fn new(sigma: f64, sigma_data: f64) -> Self {
        let denom = (sigma * sigma + sigma_data * sigma_data).sqrt();
        Self {
            c_skip: sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data),
            c_out: sigma * sigma_data / denom,
            c_in: 1.0 / denom,
            c_noise: 0.25 * sigma.ln(),
        }
    }
}

/// EDM loss weighting `w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`
/// (`get_weights`).
pub fn edm_loss_weight(sigma: f64, sigma_data: f64) -> f64 {
    (sigma * sigma + sigma_data * sigma_data) / (sigma * sigma_data).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_block_sigmas_endpoints_and_monotonicity() {
        let b3 = block_sigmas(3);
        assert_eq!(b3.len(), 4);
        // Endpoints equal the global bounds by construction (CDF endpoints).
        assert_relative_eq!(b3[0], SIGMA_MIN, epsilon = 1e-9);
        assert_relative_eq!(b3[3], SIGMA_MAX, max_relative = 1e-6);
        // Strictly increasing.
        for w in b3.windows(2) {
            assert!(w[0] < w[1], "block sigmas must be ascending: {b3:?}");
        }
    }

    #[test]
    fn test_block_sigmas_reference_values() {
        // Computed with scipy 1.16, mirroring `get_block_sigmas(num_layers=3)`:
        let expected = [
            0.0020000000000000005,
            0.17963247211160274,
            0.5050414640313313,
            79.99999999843152,
        ];
        let got = block_sigmas(3);
        for (g, e) in got.iter().zip(expected) {
            assert_relative_eq!(*g, e, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_discrete_sigmas_dblock_matches_block_grid() {
        // With num_steps == num_blocks + 1 the discrete schedule must coincide
        // with the block boundaries (both grids are uniform in CDF space).
        // The discrete schedule is returned in *descending* order while the
        // block boundaries ascend, so compare against the reversed steps.
        let steps = discrete_sigmas_dblock(4, SIGMA_MIN, SIGMA_MAX, P_MEAN, P_STD);
        let blocks = block_sigmas(3);
        assert_eq!(steps.len(), 4);
        for (s, b) in steps.iter().rev().zip(blocks.iter()) {
            assert_relative_eq!(*s, *b, max_relative = 1e-9);
        }
        // Descending order after flip.
        assert!(steps.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn test_discrete_sigmas_edm() {
        let s = discrete_sigmas_edm(1000, SIGMA_MIN, SIGMA_MAX, RHO);
        assert_relative_eq!(s[0], SIGMA_MAX, max_relative = 1e-12);
        assert_relative_eq!(s[s.len() - 1], SIGMA_MIN, max_relative = 1e-12);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn test_sampler_window_extension() {
        let sampler = DblockSigmaSampler::new(3, 0.05);
        let bounds = &sampler.block_sigmas;

        // Block 0 is the noisiest: its window tops out at the global max and
        // stays clamped there after the gamma extension.
        let (lo0, hi0) = sampler.extended_window(0);
        assert_relative_eq!(hi0, SIGMA_MAX, max_relative = 1e-9);
        assert!(lo0 < bounds[2], "gamma must widen downwards: {lo0} vs {}", bounds[2]);

        // The last block is the cleanest and clamps at the global min.
        let (lo_last, hi_last) = sampler.extended_window(2);
        assert_relative_eq!(lo_last, SIGMA_MIN, max_relative = 1e-9);
        assert!(hi_last > bounds[1]);

        // gamma = 0 reproduces the bare windows exactly.
        let bare = DblockSigmaSampler::new(3, 0.0);
        for b in 0..3 {
            assert_eq!(bare.extended_window(b), block_window(&bare.block_sigmas, b));
        }
    }

    #[test]
    fn test_block_index_roundtrip() {
        // The certificate that closes the train/inference loop: a sigma drawn
        // from block `b`'s *training* window must be routed back to block `b`
        // at inference. If `extended_window` and `estimate_target_layer` ever
        // disagree on the index convention, a block is trained on one noise
        // range and evaluated on another.
        for num_blocks in [1usize, 2, 3, 4, 6, 12] {
            let sampler = DblockSigmaSampler::new(num_blocks, 0.0);
            let bounds = &sampler.block_sigmas;
            let mut rng = rand::rng();
            for b in 0..num_blocks {
                let sigmas = sampler.sample(&mut rng, b, 64);
                for s in &sigmas {
                    assert_eq!(
                        estimate_target_layer(bounds, &[*s]),
                        b,
                        "sigma {s} trained on block {b} routes elsewhere ({num_blocks} blocks)"
                    );
                }
                // ...and the windows tile the range without gaps or overlap.
                let (lo, hi) = block_window(bounds, b);
                assert!(lo < hi);
                if b + 1 < num_blocks {
                    assert_eq!(lo, block_window(bounds, b + 1).1);
                    assert_eq!(lo, shared_boundary_sigma(bounds, b));
                }
            }
            assert_relative_eq!(block_window(bounds, 0).1, SIGMA_MAX, max_relative = 1e-6);
            assert_relative_eq!(
                block_window(bounds, num_blocks - 1).0,
                SIGMA_MIN,
                max_relative = 1e-9
            );
        }
    }

    #[test]
    fn test_sampler_within_extended_window() {
        let mut rng = rand::rng();
        let sampler = DblockSigmaSampler::new(3, 0.05);
        for block_idx in 0..3 {
            let sigmas = sampler.sample(&mut rng, block_idx, 256);
            let (lo, hi) = sampler.extended_window(block_idx);
            for s in &sigmas {
                assert!(
                    *s >= lo * (1.0 - 1e-12) && *s <= hi * (1.0 + 1e-12),
                    "sigma {s} outside window [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn test_preconditioning_values() {
        let p = EdmPreconditioning::new(0.5, 0.5);
        assert_relative_eq!(p.c_skip, 0.5, epsilon = 1e-12);
        // c_out = sigma*sigma_data / sqrt(sigma^2 + sigma_data^2) = 1/(2 sqrt 2).
        assert_relative_eq!(p.c_out, 0.35355339059327373, epsilon = 1e-12);
        assert_relative_eq!(p.c_in, 2.0_f64.sqrt(), epsilon = 1e-12);
        assert_relative_eq!(p.c_noise, 0.25 * (0.5f64).ln(), epsilon = 1e-12);

        // Identity check: D(x) = c_out * F(c_in * zt) + c_skip * zt reproduces
        // x when F is exact at any sigma.
    }

    #[test]
    fn test_loss_weight() {
        // At sigma == sigma_data the weight is 2/sigma_data^2.
        assert_relative_eq!(edm_loss_weight(0.5, 0.5), 8.0, epsilon = 1e-9);
        assert_relative_eq!(edm_loss_weight(80.0, 0.5), (80.0f64.powi(2) + 0.25) / 1600.0, epsilon = 1e-12);
    }

    #[test]
    fn test_estimate_target_layer_mapping() {
        let bounds = block_sigmas(3); // ~[0.002, 0.180, 0.505, 80]
        // Highest-noise sigma -> block 0.
        assert_eq!(estimate_target_layer(&bounds, &[70.0]), 0);
        // Lowest-noise sigma -> last block.
        assert_eq!(estimate_target_layer(&bounds, &[0.003]), 2);
        // Middle window -> middle block.
        assert_eq!(estimate_target_layer(&bounds, &[0.5]), 1);
        // Majority vote across mixed sigmas (2x high noise vs 1x low).
        assert_eq!(estimate_target_layer(&bounds, &[50.0, 60.0, 0.1]), 0);
        // Boundary value belongs to the lower window (right=True semantics):
        // s == bounds[2] -> raw window 1 -> block 3-1-1 = 1.
        assert_eq!(estimate_target_layer(&bounds, &[bounds[2]]), 1);
    }
}
