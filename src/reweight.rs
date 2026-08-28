//! Per-sigma loss reweighting and importance sampling (roadmap 20.5, 20.6).
//!
//! # The problem, measured
//!
//! This repository's own runs show block 2 at a mean loss of **1909.8** against
//! block 0's **13.4** — a ~140x imbalance. It is not a bug: the EDM weight
//!
//! ```text
//! w(sigma) = (sigma^2 + sigma_d^2) / (sigma * sigma_d)^2
//! ```
//!
//! diverges as `sigma -> 0`, and block 2 owns the low-noise end. But a shared
//! trunk trained on a sum of terms differing by two orders of magnitude is, in
//! effect, trained almost entirely on one of them. The other blocks contribute
//! gradient noise.
//!
//! Two independent remedies live here, and they compose:
//!
//! - [`UncertaintyWeighting`] (20.5) learns a per-noise-level scale and divides
//!   it out. At its optimum the gradient becomes that of `log` loss, which is
//!   **invariant to any per-sigma rescaling** — so the imbalance cannot
//!   reappear under a different weighting convention.
//! - [`SigmaImportanceSampler`] (20.6) spends samples where the loss actually
//!   varies, and reweights so the estimator stays unbiased.
//!
//! # Why neither lives inside the model
//!
//! Both are training-time objects. Putting the log-variance head inside
//! [`crate::vit`] would add a parameter to the module record, and every existing
//! checkpoint would stop loading. They are owned by the training loop instead,
//! and a run that does not enable them pays nothing — not even a record field.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use rand::Rng;

use crate::stats::{norm_cdf, norm_ppf};

/// Predicts the log-variance of the loss at a given noise level.
///
/// A two-layer MLP over the same sinusoidal noise embedding the trunk
/// conditions on, so it sees exactly the signal the denoiser sees. Small on
/// purpose: it has one scalar to explain per noise level, and a head with the
/// capacity to memorize individual batches would learn the noise instead of the
/// scale.
#[derive(Module, Debug)]
pub struct LogVarianceHead<B: Backend> {
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    frequency_embedding_size: usize,
}

impl<B: Backend<FloatElem = f32>> LogVarianceHead<B> {
    /// A head whose output is **exactly zero** everywhere at initialization.
    ///
    /// Zero log-variance is `exp(-0) = 1`, so the reweighted loss starts
    /// identical to the unweighted one and the run cannot be made worse by
    /// merely enabling the feature. The same zero-init argument the DiT blocks
    /// and the LoRA adapters use, for the same reason.
    pub fn new(hidden_size: usize, frequency_embedding_size: usize, device: &B::Device) -> Self {
        let linear_2 = LinearConfig::new(hidden_size, 1).with_bias(true).init(device);
        let zeroed = Linear {
            weight: burn::module::Param::from_tensor(linear_2.weight.val().zeros_like()),
            bias: linear_2
                .bias
                .map(|b| burn::module::Param::from_tensor(b.val().zeros_like())),
        };

        Self {
            linear_1: LinearConfig::new(frequency_embedding_size, hidden_size)
                .with_bias(true)
                .init(device),
            linear_2: zeroed,
            frequency_embedding_size,
        }
    }

    /// Sinusoidal embedding of `log(sigma)`.
    ///
    /// The *log* is what makes this work: sigma spans four decades, so a
    /// sinusoidal basis over sigma itself would resolve the noisy end and
    /// collapse the entire low-noise range — which is exactly the range whose
    /// scale needs explaining.
    fn embed(&self, sigmas: Tensor<B, 1>) -> Tensor<B, 2> {
        let device = sigmas.device();
        let half = self.frequency_embedding_size / 2;
        let exponent = -(10000f64.ln()) / half as f64;
        let freqs = Tensor::<B, 1, burn::tensor::Int>::arange(0..half as i64, &device)
            .float()
            .mul_scalar(exponent)
            .exp();
        let args = sigmas.log().unsqueeze_dim::<2>(1) * freqs.unsqueeze_dim::<2>(0);
        Tensor::cat(vec![args.clone().cos(), args.sin()], 1)
    }

    /// Per-sample log-variance `[b]` for noise levels `[b]`.
    pub fn forward(&self, sigmas: Tensor<B, 1>) -> Tensor<B, 1> {
        let h = crate::vit::silu_public(self.linear_1.forward(self.embed(sigmas)));
        self.linear_2.forward(h).squeeze_dim::<1>(1)
    }
}

/// Uncertainty-weighted per-sigma loss (EDM appendix B.6; Kendall & Gal, 2017).
///
/// The raw per-sample loss `L_i` is replaced by
///
/// ```text
/// exp(-l_i) * L_i + l_i,        l_i = logvar(sigma_i)
/// ```
///
/// # What the `+ l` term is for
///
/// Without it, the head would drive `l -> +inf` and make every loss vanish.
/// With it, the objective in `l` alone is minimized at `l* = ln L`, where its
/// value is `1 + ln L`. So the head is *forced* to report the true loss scale:
/// it cannot buy a smaller number by lying about the uncertainty.
///
/// # Why this fixes the imbalance rather than papering over it
///
/// At `l = l*`, the gradient with respect to the network is
///
/// ```text
/// exp(-l*) * dL/dtheta  =  (1/L) * dL/dtheta  =  d(ln L)/dtheta
/// ```
///
/// which is **invariant under `L -> c*L` for any per-sigma constant `c`**. That
/// is the precise sense in which no noise level can dominate: the imbalance is
/// a property of the *scale* of `w(sigma)`, and the gradient no longer sees the
/// scale at all. Certificate `optim/uncertainty_gradient_is_scale_free` states
/// exactly this.
#[derive(Debug, Clone, Copy)]
pub struct UncertaintyWeighting {
    /// Blend against the unweighted loss, in `[0, 1]`. `0.0` is the exact
    /// identity; `1.0` is full uncertainty weighting.
    pub strength: f64,
    /// Clamp on the predicted log-variance.
    ///
    /// `exp(-l)` is unbounded below, so an unclamped head that overshoots early
    /// can multiply a loss by `e^20` and destroy the run in one step. The bound
    /// is on the *log*, so it is symmetric in scale.
    pub clamp: f64,
}

impl Default for UncertaintyWeighting {
    fn default() -> Self {
        Self::off()
    }
}

impl UncertaintyWeighting {
    /// The exact identity: the loss is returned untouched.
    pub fn off() -> Self {
        Self { strength: 0.0, clamp: 8.0 }
    }

    pub fn full() -> Self {
        Self { strength: 1.0, clamp: 8.0 }
    }

    pub fn new(strength: f64) -> Self {
        Self { strength: strength.clamp(0.0, 1.0), clamp: 8.0 }
    }

    pub fn with_clamp(mut self, clamp: f64) -> Self {
        self.clamp = clamp.max(0.0);
        self
    }

    pub fn is_identity(&self) -> bool {
        self.strength == 0.0
    }

    /// Apply the weighting to per-sample losses `[b]` given log-variances `[b]`.
    ///
    /// At `strength == 0` the input is returned **unchanged** rather than
    /// blended with itself: `(1-t)*L + t*L` is not bitwise `L` in floating
    /// point, and an "almost identity" default is a leak nobody would look for.
    pub fn apply<B: Backend<FloatElem = f32>>(
        &self,
        per_sample_loss: Tensor<B, 1>,
        log_variance: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        if self.is_identity() {
            return per_sample_loss;
        }
        let l = log_variance.clamp(-self.clamp as f32, self.clamp as f32);
        let weighted = per_sample_loss.clone() * l.clone().neg().exp() + l;
        if self.strength >= 1.0 {
            return weighted;
        }
        let t = self.strength as f32;
        per_sample_loss.mul_scalar(1.0 - t) + weighted.mul_scalar(t)
    }

    /// The log-variance that minimizes the objective for a raw loss `l_raw`.
    ///
    /// Exposed because it is the closed form the certificate checks the head
    /// against, and because it is the number to compare a *trained* head's
    /// output with when diagnosing whether it has converged.
    pub fn optimal_log_variance(raw_loss: f64) -> f64 {
        raw_loss.max(f64::MIN_POSITIVE).ln()
    }

    /// Objective value at the optimum: `1 + ln(L)`.
    pub fn value_at_optimum(raw_loss: f64) -> f64 {
        1.0 + Self::optimal_log_variance(raw_loss)
    }
}

/// Importance sampling over noise levels (roadmap 20.6).
///
/// Training draws sigma uniformly in lognormal-CDF space across a block's
/// window. That spends the same number of samples on regions where the loss is
/// flat as on regions where it varies, and the variance of the gradient
/// estimate is dominated by the latter.
///
/// This keeps a running estimate of the loss magnitude per CDF bin, proposes
/// bins in proportion to it, and returns an **importance weight** `p/q` with
/// each sample so the estimator stays unbiased:
///
/// ```text
/// E_q[ (p/q) f ] = sum_b q_b * (p_b/q_b) * f_b = sum_b p_b f_b = E_p[f]
/// ```
///
/// # The floor is not optional
///
/// The proposal is mixed with the uniform one at weight [`Self::smoothing`].
/// Without it a bin whose loss estimate happens to start near zero is never
/// sampled again, its estimate never updates, and — worse — if the true loss
/// there later grows, the importance weight `p/q` for that bin is enormous. A
/// starved bin is not merely unvisited; it is a variance bomb.
#[derive(Debug, Clone)]
pub struct SigmaImportanceSampler {
    /// Running mean loss magnitude per bin.
    means: Vec<f64>,
    /// Observations per bin, for bias correction on the running mean.
    counts: Vec<usize>,
    /// Exponential-moving-average rate for the per-bin estimate.
    momentum: f64,
    /// Mixture weight on the uniform proposal, in `(0, 1]`.
    smoothing: f64,
}

impl SigmaImportanceSampler {
    /// A sampler over `bins` equal-probability CDF intervals.
    ///
    /// Bins are equal in *probability* rather than in sigma, so under the prior
    /// each starts equally likely and `p_b = 1/bins` exactly — which is what
    /// makes the importance weight a ratio of two simple numbers instead of an
    /// integral.
    pub fn new(bins: usize) -> Self {
        let bins = bins.max(1);
        Self {
            means: vec![0.0; bins],
            counts: vec![0; bins],
            momentum: 0.9,
            smoothing: 0.25,
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum.clamp(0.0, 0.999);
        self
    }

    /// Mixture weight on the uniform proposal. Clamped away from 0, because a
    /// proposal with no floor can starve a bin permanently.
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing.clamp(1e-3, 1.0);
        self
    }

    pub fn bins(&self) -> usize {
        self.means.len()
    }

    pub fn smoothing(&self) -> f64 {
        self.smoothing
    }

    /// Whether any bin has been observed yet.
    pub fn is_warm(&self) -> bool {
        self.counts.iter().any(|c| *c > 0)
    }

    /// Prior probability of each bin: uniform by construction.
    pub fn prior(&self) -> f64 {
        1.0 / self.bins() as f64
    }

    /// The proposal distribution `q`, summing to exactly 1.
    ///
    /// Before any observation this is the uniform distribution, so a cold
    /// sampler is *exactly* plain sampling with all weights 1.
    pub fn proposal(&self) -> Vec<f64> {
        let n = self.bins();
        let uniform = 1.0 / n as f64;
        if !self.is_warm() {
            return vec![uniform; n];
        }

        let total: f64 = self.means.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return vec![uniform; n];
        }

        // Mix the loss-proportional proposal with the uniform one, then
        // renormalize. Renormalizing after the mixture rather than trusting the
        // algebra keeps the sum at 1 to machine precision even when a bin
        // underflows.
        let s = self.smoothing;
        let mut q: Vec<f64> = self
            .means
            .iter()
            .map(|m| s * uniform + (1.0 - s) * (m / total))
            .collect();
        let sum: f64 = q.iter().sum();
        for value in &mut q {
            *value /= sum;
        }
        q
    }

    /// Draw `n` sigmas from `[cdf_lo, cdf_hi)` with their importance weights.
    ///
    /// The window is the block's extended CDF range; bins partition it, so the
    /// sampler composes with the existing per-block windows rather than
    /// replacing them.
    ///
    /// Returns `(sigma, weight)` pairs. Multiply each sample's loss by its
    /// weight before averaging; the mean is then an unbiased estimate of the
    /// mean under the prior.
    pub fn sample<R: Rng>(
        &self,
        rng: &mut R,
        cdf_lo: f64,
        cdf_hi: f64,
        p_mean: f64,
        p_std: f64,
        n: usize,
    ) -> Vec<(f64, f64)> {
        let q = self.proposal();
        let bins = self.bins();
        let prior = self.prior();
        let span = (cdf_hi - cdf_lo).max(f64::MIN_POSITIVE);
        let bin_span = span / bins as f64;

        (0..n)
            .map(|_| {
                let bin = pick(rng, &q);
                let lo = cdf_lo + bin as f64 * bin_span;
                let u = rng.random_range(lo..(lo + bin_span).min(cdf_hi).max(lo + f64::EPSILON));
                let sigma = (p_mean + p_std * norm_ppf(u.clamp(1e-12, 1.0 - 1e-12))).exp();
                (sigma, prior / q[bin].max(f64::MIN_POSITIVE))
            })
            .collect()
    }

    /// Which bin a sigma falls in, given the window it was drawn from.
    pub fn bin_of(&self, sigma: f64, cdf_lo: f64, cdf_hi: f64, p_mean: f64, p_std: f64) -> usize {
        let cdf = norm_cdf((sigma.ln() - p_mean) / p_std);
        let span = (cdf_hi - cdf_lo).max(f64::MIN_POSITIVE);
        let position = ((cdf - cdf_lo) / span).clamp(0.0, 1.0 - f64::EPSILON);
        ((position * self.bins() as f64) as usize).min(self.bins() - 1)
    }

    /// Record the loss magnitude observed in `bin`.
    ///
    /// Non-finite observations are ignored rather than propagated: a single NaN
    /// would otherwise poison the proposal for the rest of the run, and the
    /// training loop already has a gate that rejects the step it came from.
    pub fn observe(&mut self, bin: usize, loss: f64) {
        if bin >= self.bins() || !loss.is_finite() {
            return;
        }
        let magnitude = loss.abs();
        let m = self.momentum;
        self.means[bin] = if self.counts[bin] == 0 {
            magnitude
        } else {
            m * self.means[bin] + (1.0 - m) * magnitude
        };
        self.counts[bin] += 1;
    }

    /// Observations recorded in `bin`.
    pub fn count(&self, bin: usize) -> usize {
        self.counts.get(bin).copied().unwrap_or(0)
    }

    /// Running loss estimate for `bin`.
    pub fn mean(&self, bin: usize) -> f64 {
        self.means.get(bin).copied().unwrap_or(0.0)
    }

    /// Largest importance weight the current proposal can produce.
    ///
    /// The variance of an importance-sampled estimator scales with this, so it
    /// is the number to watch: the smoothing floor bounds it at
    /// `bins / (smoothing * ...)`, and reporting it makes that bound checkable
    /// rather than merely argued.
    pub fn max_weight(&self) -> f64 {
        let prior = self.prior();
        self.proposal()
            .iter()
            .map(|q| prior / q.max(f64::MIN_POSITIVE))
            .fold(0.0f64, f64::max)
    }
}

/// Sample an index from a distribution that sums to 1.
fn pick<R: Rng>(rng: &mut R, weights: &[f64]) -> usize {
    let mut target: f64 = rng.random::<f64>();
    for (i, w) in weights.iter().enumerate() {
        target -= w;
        if target <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use rand::{rngs::StdRng, SeedableRng};

    type B = NdArray<f32>;

    fn tensor(values: &[f32]) -> Tensor<B, 1> {
        let device = Default::default();
        Tensor::<B, 1>::from_floats(values, &device)
    }

    fn values(t: Tensor<B, 1>) -> Vec<f32> {
        t.into_data().convert::<f32>().iter::<f32>().collect()
    }

    #[test]
    fn test_the_head_starts_at_exactly_zero() {
        // Zero log-variance is exp(-0) = 1, so enabling the feature cannot make
        // the first step worse than not enabling it. Checked bitwise, because
        // "close to zero" would still perturb every loss in the run.
        let device = Default::default();
        let head = LogVarianceHead::<B>::new(32, 64, &device);
        let out = values(head.forward(tensor(&[0.002, 0.1, 1.0, 12.0, 80.0])));
        for v in out {
            assert_eq!(v, 0.0, "the head must be identically zero at init");
        }
    }

    #[test]
    fn test_zero_strength_is_bitwise_identity() {
        let raw = tensor(&[13.4, 1909.8, 0.001, 42.0]);
        let logvar = tensor(&[0.5, -3.0, 7.0, 1.0]);
        let out = UncertaintyWeighting::off().apply(raw.clone(), logvar);
        assert_eq!(values(out), values(raw));
    }

    #[test]
    fn test_the_objective_is_minimized_at_the_true_log_loss() {
        // The claim that makes the `+ l` term work: the head cannot buy a
        // smaller number by lying about the uncertainty. Minimize over a dense
        // grid and check the argmin lands on ln(L).
        for raw in [0.05f64, 1.0, 13.4, 1909.8] {
            let objective = |l: f64| (-l).exp() * raw + l;
            let mut best = (f64::INFINITY, 0.0);
            for i in 0..200_001 {
                let l = -10.0 + 20.0 * i as f64 / 200_000.0;
                let v = objective(l);
                if v < best.0 {
                    best = (v, l);
                }
            }
            let expected = UncertaintyWeighting::optimal_log_variance(raw);
            assert!(
                (best.1 - expected).abs() < 1e-3,
                "argmin {} != ln({raw}) = {expected}",
                best.1
            );
            assert!(
                (best.0 - UncertaintyWeighting::value_at_optimum(raw)).abs() < 1e-6,
                "value at optimum {} != 1 + ln({raw})",
                best.0
            );
        }
    }

    #[test]
    fn test_at_the_optimum_the_gradient_forgets_the_scale() {
        // The whole point. Block 2's loss is ~140x block 0's; after optimal
        // reweighting the gradient scale factor exp(-l*) exactly cancels it, so
        // a shared trunk sees comparable gradients from both.
        let block0 = 13.4f64;
        let block2 = 1909.8f64;

        let effective = |raw: f64| (-UncertaintyWeighting::optimal_log_variance(raw)).exp() * raw;
        assert!((effective(block0) - 1.0).abs() < 1e-12);
        assert!((effective(block2) - 1.0).abs() < 1e-12);

        // ...and it is invariant to rescaling, which is the stronger statement:
        // no choice of loss weighting convention can reintroduce the imbalance.
        for c in [1e-6f64, 0.5, 1.0, 3.0, 1e6] {
            assert!(
                (effective(block2 * c) - effective(block2)).abs() < 1e-9,
                "rescaling by {c} changed the effective gradient scale"
            );
        }
    }

    #[test]
    fn test_partial_strength_interpolates() {
        let raw = tensor(&[4.0]);
        let logvar = tensor(&[1.0]);
        let full = values(UncertaintyWeighting::full().apply(raw.clone(), logvar.clone()))[0];
        let half = values(UncertaintyWeighting::new(0.5).apply(raw.clone(), logvar))[0];
        let expected = 0.5 * 4.0 + 0.5 * full;
        assert!((half - expected).abs() < 1e-5, "{half} != {expected}");
    }

    #[test]
    fn test_the_clamp_bounds_the_multiplier() {
        // exp(-l) is unbounded below; a head that overshoots early could
        // multiply a loss by e^20 and destroy the run in one step.
        let raw = tensor(&[1.0, 1.0]);
        let wild = tensor(&[-500.0, 500.0]);
        let out = values(UncertaintyWeighting::full().with_clamp(4.0).apply(raw, wild));
        for v in &out {
            assert!(v.is_finite(), "clamping must keep the loss finite: {v}");
        }
        // exp(4) * 1 - 4 is the worst case at clamp 4.
        assert!(out[0] <= (4.0f32).exp() - 4.0 + 1e-3);
    }

    #[test]
    fn test_a_cold_sampler_is_exactly_plain_sampling() {
        // Containment: importance sampling with q == p must be indistinguishable
        // from not doing it, so the feature can be enabled unconditionally.
        let sampler = SigmaImportanceSampler::new(8);
        assert!(!sampler.is_warm());

        let q = sampler.proposal();
        assert_eq!(q.len(), 8);
        for value in &q {
            assert!((value - 0.125).abs() < 1e-15, "cold proposal must be uniform");
        }

        let mut rng = StdRng::seed_from_u64(3);
        for (sigma, weight) in sampler.sample(&mut rng, 0.05, 0.95, -1.2, 1.2, 32) {
            assert!(sigma.is_finite() && sigma > 0.0);
            assert!((weight - 1.0).abs() < 1e-12, "weight must be exactly 1, got {weight}");
        }
        assert!((sampler.max_weight() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_the_estimator_stays_unbiased() {
        // The identity the whole scheme rests on:
        //   sum_b q_b * (p_b / q_b) = sum_b p_b = 1.
        // Exact, so it is checked exactly rather than by convergence.
        let mut sampler = SigmaImportanceSampler::new(6);
        for bin in 0..6 {
            sampler.observe(bin, (bin as f64 + 1.0).powi(3));
        }
        let q = sampler.proposal();
        let prior = sampler.prior();

        let total: f64 = q.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "proposal must be a distribution");

        let expectation: f64 = q.iter().map(|qb| qb * (prior / qb)).sum();
        assert!(
            (expectation - 1.0).abs() < 1e-12,
            "E_q[p/q] must be 1, got {expectation}"
        );
    }

    #[test]
    fn test_the_proposal_follows_the_loss() {
        // What importance sampling is for: spend samples where the loss lives.
        let mut sampler = SigmaImportanceSampler::new(4).with_smoothing(0.05);
        sampler.observe(0, 1.0);
        sampler.observe(1, 1.0);
        sampler.observe(2, 1.0);
        sampler.observe(3, 500.0);

        let q = sampler.proposal();
        assert!(q[3] > q[0] * 10.0, "the heavy bin should dominate: {q:?}");
        assert!(
            q[0] > 0.0 && q[1] > 0.0 && q[2] > 0.0,
            "no bin may be starved: {q:?}"
        );
    }

    #[test]
    fn test_smoothing_bounds_the_worst_weight() {
        // A starved bin is not merely unvisited; it is a variance bomb, because
        // its importance weight p/q grows without bound. The floor caps it.
        for smoothing in [0.05f64, 0.25, 0.5, 1.0] {
            let mut sampler = SigmaImportanceSampler::new(10).with_smoothing(smoothing);
            // Adversarial: all the mass in one bin.
            sampler.observe(0, 1e9);
            for bin in 1..10 {
                sampler.observe(bin, 1e-12);
            }
            // q_b >= smoothing/bins after mixing (renormalization only
            // increases it, since the mixture already sums to 1).
            let bound = 1.0 / smoothing;
            assert!(
                sampler.max_weight() <= bound + 1e-6,
                "max weight {} exceeded 1/smoothing = {bound}",
                sampler.max_weight()
            );
        }
    }

    #[test]
    fn test_importance_sampling_reduces_variance_on_a_peaked_loss() {
        // The payoff, measured rather than argued: for a loss concentrated in
        // one bin, the reweighted estimator's spread should beat the uniform
        // one at the same sample count.
        let bins = 8usize;
        let truth = |bin: usize| if bin == 7 { 100.0 } else { 0.01 };

        let mut sampler = SigmaImportanceSampler::new(bins).with_smoothing(0.1);
        for bin in 0..bins {
            sampler.observe(bin, truth(bin));
        }
        let q = sampler.proposal();
        let prior = sampler.prior();

        // Exact variances of the two estimators, no sampling needed:
        //   Var_q[(p/q) f] = sum_b q_b ((p_b/q_b) f_b)^2 - mean^2
        let mean: f64 = (0..bins).map(|b| prior * truth(b)).sum();
        let var = |proposal: &[f64]| -> f64 {
            let second: f64 = (0..bins)
                .map(|b| {
                    let w = prior / proposal[b];
                    proposal[b] * (w * truth(b)).powi(2)
                })
                .sum();
            second - mean * mean
        };

        let uniform = vec![prior; bins];
        assert!(
            var(&q) < var(&uniform),
            "importance sampling should reduce variance: {} vs {}",
            var(&q),
            var(&uniform)
        );
    }

    #[test]
    fn test_bins_partition_the_window() {
        // Every sigma drawn from a window must land in a bin of that window,
        // or the loss observed at one noise level would update another's
        // estimate.
        let sampler = SigmaImportanceSampler::new(5);
        let (lo, hi) = (0.1f64, 0.9);
        let mut rng = StdRng::seed_from_u64(9);
        let mut seen = vec![0usize; 5];

        for (sigma, _) in sampler.sample(&mut rng, lo, hi, -1.2, 1.2, 400) {
            let bin = sampler.bin_of(sigma, lo, hi, -1.2, 1.2);
            assert!(bin < 5);
            seen[bin] += 1;
        }
        assert!(seen.iter().all(|c| *c > 0), "a uniform proposal must reach every bin: {seen:?}");
    }

    #[test]
    fn test_a_non_finite_observation_is_ignored() {
        // One NaN would otherwise poison the proposal for the rest of the run,
        // and the training loop already rejects the step it came from.
        let mut sampler = SigmaImportanceSampler::new(3);
        sampler.observe(0, 5.0);
        sampler.observe(1, f64::NAN);
        sampler.observe(2, f64::INFINITY);

        assert_eq!(sampler.count(1), 0);
        assert_eq!(sampler.count(2), 0);
        assert!(sampler.proposal().iter().all(|q| q.is_finite() && *q > 0.0));
    }

    #[test]
    fn test_the_head_receives_gradients_through_the_weighted_loss() {
        // The head is only useful if the optimizer can move it. It is a
        // separate module sharing one backward pass with the trunk, which is
        // exactly the arrangement most likely to end up silently detached.
        use burn::backend::{Autodiff, NdArray};
        use burn::optim::GradientsParams;

        type AB = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let head = LogVarianceHead::<AB>::new(16, 32, &device);

        let per_sample = Tensor::<AB, 1>::from_floats([13.4f32, 1909.8, 0.7].as_slice(), &device);
        let sigmas = Tensor::<AB, 1>::from_floats([0.01f32, 5.0, 40.0].as_slice(), &device);

        let loss = UncertaintyWeighting::full()
            .apply(per_sample, head.forward(sigmas))
            .mean();
        let grads = GradientsParams::from_grads(loss.backward(), &head);
        assert!(!grads.is_empty(), "the log-variance head must receive gradients");
    }

    #[test]
    fn test_a_trained_head_converges_on_the_true_log_loss() {
        // The end-to-end statement: given a constant loss, gradient descent on
        // the head drives its output to ln(L) -- the closed form the objective
        // is minimized at. If it converged anywhere else, the `+ l` term would
        // not be doing its job and the weighting would be free to lie.
        use burn::backend::{Autodiff, NdArray};
        use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

        type AB = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let mut head = LogVarianceHead::<AB>::new(32, 32, &device);
        let mut optimizer = AdamWConfig::new().init();

        // One noise level, one loss magnitude: the simplest case where the
        // answer is known exactly.
        let raw = 20.0f32;
        let sigmas = Tensor::<AB, 1>::from_floats([1.0f32].as_slice(), &device);

        for _ in 0..400 {
            let per_sample = Tensor::<AB, 1>::from_floats([raw].as_slice(), &device);
            let loss = UncertaintyWeighting::full()
                .apply(per_sample, head.forward(sigmas.clone()))
                .mean();
            let grads = GradientsParams::from_grads(loss.backward(), &head);
            head = optimizer.step(0.05, head, grads);
        }

        let learned: f32 = head.forward(sigmas).into_scalar();
        let expected = UncertaintyWeighting::optimal_log_variance(f64::from(raw)) as f32;
        assert!(
            (learned - expected).abs() < 0.15,
            "head converged to {learned}, expected ln({raw}) = {expected}"
        );
    }
}
