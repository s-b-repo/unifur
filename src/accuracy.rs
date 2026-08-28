//! Accuracy improvements (roadmap Phase 22).
//!
//! Four techniques that raise accuracy *after* training, without touching the
//! weights. They share a property worth stating up front: each one has an
//! exact identity setting, and each identity is checked by a certificate in the
//! `accuracy` group. That is what makes them safe to leave in a pipeline —
//! a mis-set knob degrades to the un-improved baseline rather than to garbage.
//!
//! - [`Guidance`]: a classifier-free-guidance analogue over paired conditional
//!   and unconditional x0 estimates.
//! - [`LogitNorm`]: per-sample logit normalization. Changes calibration, never
//!   the prediction — which is precisely why it matters here, since the
//!   adaptive strategy and the quality gates both read confidences.
//! - [`Ensemble`]: combining the outputs of several samplers.
//! - [`ScalingCurve`]: test-time compute scaling, measured rather than assumed.
//!
//! # What is not here
//!
//! Self-conditioning (roadmap 22.2) is absent by decision, not by oversight.
//! Feeding the previous x0 estimate back as conditioning only helps a model
//! *trained* with it; bolted on at sampling time it degrades results. Doing it
//! properly needs a new projection inside [`crate::vit`], which adds a
//! parameter to the module record and so cannot load an existing checkpoint.
//! It stays in `TODO.md` with that cost written down.

use burn::tensor::{activation::softmax, backend::Backend, Int, Tensor};

/// A classifier-free-guidance analogue.
///
/// Guidance extrapolates away from the unconditional estimate:
///
/// ```text
/// x0_guided = x0_uncond + scale * (x0_cond - x0_uncond)
/// ```
///
/// `scale = 1` is the conditional estimate, `scale = 0` the unconditional one,
/// and `scale > 1` sharpens conditioning at the cost of diversity and, past
/// some point, of fidelity.
///
/// # Rescaling
///
/// Large scales inflate the norm of the estimate, which in this crate feeds
/// straight into an ODE step and compounds. [`Guidance::rescale`] interpolates
/// the guided estimate back toward the conditional estimate's per-sample
/// standard deviation (Lin et al., 2024). `0.0` disables it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Guidance {
    pub scale: f64,
    pub rescale: f64,
}

impl Default for Guidance {
    fn default() -> Self {
        Self::none()
    }
}

impl Guidance {
    /// The identity: the conditional estimate, untouched.
    pub fn none() -> Self {
        Self { scale: 1.0, rescale: 0.0 }
    }

    pub fn new(scale: f64) -> Self {
        Self { scale, rescale: 0.0 }
    }

    pub fn with_rescale(mut self, rescale: f64) -> Self {
        self.rescale = rescale.clamp(0.0, 1.0);
        self
    }

    /// Whether this configuration is the exact identity.
    pub fn is_identity(&self) -> bool {
        self.scale == 1.0 && self.rescale == 0.0
    }

    /// Combine a conditional and an unconditional estimate.
    ///
    /// At `scale == 1.0` the conditional estimate is returned **unchanged**,
    /// rather than computed as `u + 1.0 * (c - u)`. The two are equal in exact
    /// arithmetic but not in floating point: the round trip through `c - u` and
    /// back loses low bits whenever `|u|` is much larger than `|c - u|`. The
    /// short-circuit is what lets `guidance_identity_is_exact` carry a
    /// tolerance of zero instead of an epsilon nobody can justify.
    pub fn apply<B: Backend<FloatElem = f32>>(
        &self,
        conditional: Tensor<B, 2>,
        unconditional: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        debug_assert_eq!(
            conditional.dims(),
            unconditional.dims(),
            "guidance needs matched estimates"
        );
        if self.is_identity() {
            return conditional;
        }

        let guided = unconditional.clone()
            + (conditional.clone() - unconditional).mul_scalar(self.scale as f32);
        if self.rescale <= 0.0 {
            return guided;
        }

        // Match the conditional estimate's per-sample spread, then interpolate
        // back by `rescale`. Guarding the divisor keeps a degenerate row (every
        // component equal) from producing a NaN that would poison the ODE.
        let target = row_std(&conditional);
        let actual = row_std(&guided).clamp_min(1e-8);
        let factor = target / actual;
        let blended = factor.mul_scalar(self.rescale as f32) + (1.0 - self.rescale as f32);
        guided * blended.unsqueeze_dim::<2>(1)
    }
}

/// Per-sample standard deviation `[b]`.
fn row_std<B: Backend<FloatElem = f32>>(x: &Tensor<B, 2>) -> Tensor<B, 1> {
    let mean = x.clone().mean_dim(1);
    (x.clone() - mean).powf_scalar(2.0).mean_dim(1).sqrt().squeeze_dim::<1>(1)
}

/// Per-sample logit normalization.
///
/// Every variant is a strictly increasing affine map applied per row, so **the
/// arg-max never moves**: top-1 accuracy is identical before and after. What
/// changes is the confidence the softmax reports.
///
/// That is the point. [`crate::multi_block::Strategy::Adaptive`] widens its
/// span when confidence is low, and [`crate::quality`] rejects updates on
/// confidence thresholds. Both read a number whose scale is an artifact of how
/// large the trained logits happen to be — a threshold tuned on one checkpoint
/// means something else on the next. Normalizing first makes those thresholds
/// portable, which is an accuracy improvement by way of the gates, not by way
/// of the classifier.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LogitNorm {
    /// Leave the logits alone.
    #[default]
    None,
    /// Divide by `tau`.
    Temperature { tau: f64 },
    /// Divide by the per-sample L2 norm, then by `tau`.
    L2 { tau: f64 },
    /// Subtract the per-sample mean, divide by the per-sample standard
    /// deviation, then by `tau`.
    Standardize { tau: f64 },
}

impl LogitNorm {
    pub fn parse(name: &str, tau: f64) -> anyhow::Result<Self> {
        match name {
            "none" => Ok(Self::None),
            "temperature" => Ok(Self::Temperature { tau }),
            "l2" => Ok(Self::L2 { tau }),
            "standardize" => Ok(Self::Standardize { tau }),
            other => anyhow::bail!(
                "unknown logit normalization '{other}' (expected none|temperature|l2|standardize)"
            ),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Temperature { .. } => "temperature",
            Self::L2 { .. } => "l2",
            Self::Standardize { .. } => "standardize",
        }
    }

    pub fn is_identity(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Normalize `logits` `[b, c]`.
    ///
    /// A non-positive `tau` would flip the ordering and turn the arg-max into
    /// an arg-min, so it is clamped rather than honoured — a silently inverted
    /// classifier is worse than a slightly-off temperature.
    pub fn apply<B: Backend<FloatElem = f32>>(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        match *self {
            Self::None => logits,
            Self::Temperature { tau } => logits.div_scalar(tau.max(1e-6) as f32),
            Self::L2 { tau } => {
                let norm = logits
                    .clone()
                    .powf_scalar(2.0)
                    .sum_dim(1)
                    .sqrt()
                    .clamp_min(1e-8);
                (logits / norm).div_scalar(tau.max(1e-6) as f32)
            }
            Self::Standardize { tau } => {
                let mean = logits.clone().mean_dim(1);
                let centered = logits - mean;
                let std = centered
                    .clone()
                    .powf_scalar(2.0)
                    .mean_dim(1)
                    .sqrt()
                    .clamp_min(1e-8);
                (centered / std).div_scalar(tau.max(1e-6) as f32)
            }
        }
    }
}

/// How several independently produced logit sets are combined.
///
/// The crate already has more than one way to reach an answer — four solvers,
/// four span strategies, a planner — and they make different errors. Averaging
/// them is the cheapest accuracy gain available, at a cost exactly linear in
/// the number of members.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Ensemble {
    /// Mean of the member distributions. The default: it weights a member by
    /// how confident it is, and it is what a proper scoring rule averages.
    #[default]
    ProbabilityMean,
    /// Softmax of the mean logits. Sharper than the probability mean, and
    /// dominated by whichever member has the largest logit scale — pair it
    /// with [`LogitNorm`] if the members are not calibrated alike.
    LogitMean,
    /// Mean of the member one-hot predictions: a plurality vote, with ties
    /// broken toward the earlier member.
    MajorityVote,
}

impl Ensemble {
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "probability" => Ok(Self::ProbabilityMean),
            "logit" => Ok(Self::LogitMean),
            "vote" => Ok(Self::MajorityVote),
            other => anyhow::bail!(
                "unknown ensemble '{other}' (expected probability|logit|vote)"
            ),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ProbabilityMean => "probability",
            Self::LogitMean => "logit",
            Self::MajorityVote => "vote",
        }
    }

    /// Combine member logits into a single probability distribution `[b, c]`.
    ///
    /// Returns probabilities rather than logits because the members are only
    /// comparable after a softmax: a vote has no logit scale at all, and a
    /// probability mean is not the softmax of anything.
    ///
    /// # Panics
    ///
    /// If `members` is empty. An empty ensemble has no defensible answer, and
    /// returning a uniform distribution would hide a caller's bug behind
    /// plausible-looking numbers.
    pub fn combine<B: Backend<FloatElem = f32>>(&self, members: &[Tensor<B, 2>]) -> Tensor<B, 2> {
        assert!(!members.is_empty(), "an ensemble needs at least one member");
        let n = members.len() as f32;

        match self {
            Self::ProbabilityMean => {
                let mut acc = softmax(members[0].clone(), 1);
                for m in &members[1..] {
                    acc = acc + softmax(m.clone(), 1);
                }
                acc.div_scalar(n)
            }
            Self::LogitMean => {
                let mut acc = members[0].clone();
                for m in &members[1..] {
                    acc = acc + m.clone();
                }
                softmax(acc.div_scalar(n), 1)
            }
            Self::MajorityVote => {
                let mut acc = one_hot_argmax(&members[0]);
                for m in &members[1..] {
                    acc = acc + one_hot_argmax(m);
                }
                acc.div_scalar(n)
            }
        }
    }
}

/// One-hot encoding of the row-wise arg-max, `[b, c]`.
fn one_hot_argmax<B: Backend<FloatElem = f32>>(logits: &Tensor<B, 2>) -> Tensor<B, 2> {
    let [b, c] = logits.dims();
    let device = logits.device();
    let idx = logits.clone().argmax(1); // [b, 1]
    let positions = Tensor::<B, 1, Int>::arange(0..c as i64, &device)
        .reshape([1, c])
        .repeat_dim(0, b);
    positions.equal(idx.repeat_dim(1, c)).float()
}

/// Top-1 accuracy of `logits` `[b, c]` against `labels` `[b]`.
pub fn accuracy<B: Backend<FloatElem = f32>>(logits: &Tensor<B, 2>, labels: &Tensor<B, 1, Int>) -> f64 {
    let b = logits.dims()[0];
    if b == 0 {
        return 0.0;
    }
    let predicted = logits.clone().argmax(1).squeeze_dim::<1>(1);
    let correct: f32 = predicted.equal(labels.clone()).float().sum().into_scalar();
    f64::from(correct) / b as f64
}

/// One measurement on a test-time compute-scaling curve.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalingPoint {
    /// What was run, e.g. `"planned/depth=2"`.
    pub label: String,
    /// Denoiser invocations.
    pub model_calls: usize,
    /// Transformer layers executed — the honest cost measure when strategies
    /// use different span widths, since a wide span costs more per call.
    pub layers_executed: usize,
    /// Top-1 accuracy achieved.
    pub accuracy: f64,
}

impl ScalingPoint {
    pub fn new(label: impl Into<String>, model_calls: usize, layers_executed: usize, accuracy: f64) -> Self {
        Self { label: label.into(), model_calls, layers_executed, accuracy }
    }
}

/// Accuracy as a function of test-time compute.
///
/// "Spend more compute at inference and get more accuracy" is a claim, not a
/// law. It holds up to a point and then flattens or reverses, and where that
/// happens depends on the model, the schedule and the solver. This type exists
/// so the trade is **measured** for a given setup rather than assumed — and so
/// the flattening point, which is the operationally useful number, is reported
/// instead of inferred from a single data point.
#[derive(Debug, Clone, Default)]
pub struct ScalingCurve {
    pub points: Vec<ScalingPoint>,
}

impl ScalingCurve {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, point: ScalingPoint) {
        self.points.push(point);
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The most accurate measurement, ties going to the cheaper one.
    pub fn most_accurate(&self) -> Option<&ScalingPoint> {
        self.points.iter().reduce(|best, p| {
            if p.accuracy > best.accuracy
                || (p.accuracy == best.accuracy && p.layers_executed < best.layers_executed)
            {
                p
            } else {
                best
            }
        })
    }

    /// The cheapest measurement, ties going to the more accurate one.
    pub fn cheapest(&self) -> Option<&ScalingPoint> {
        self.points.iter().reduce(|best, p| {
            if p.layers_executed < best.layers_executed
                || (p.layers_executed == best.layers_executed && p.accuracy > best.accuracy)
            {
                p
            } else {
                best
            }
        })
    }

    /// The non-dominated measurements, cheapest first.
    ///
    /// A point is dominated when another is at least as accurate and no more
    /// expensive. Everything dominated is a configuration nobody should ever
    /// choose, so the frontier is the whole of the useful answer.
    pub fn pareto(&self) -> Vec<&ScalingPoint> {
        let mut sorted: Vec<&ScalingPoint> = self.points.iter().collect();
        sorted.sort_by(|a, b| {
            a.layers_executed
                .cmp(&b.layers_executed)
                .then(b.accuracy.partial_cmp(&a.accuracy).unwrap_or(std::cmp::Ordering::Equal))
        });

        let mut frontier: Vec<&ScalingPoint> = Vec::new();
        let mut best = f64::NEG_INFINITY;
        for p in sorted {
            // Strictly better accuracy than anything cheaper. Equal accuracy at
            // higher cost is dominated, so `>` rather than `>=`.
            if p.accuracy > best {
                frontier.push(p);
                best = p.accuracy;
            }
        }
        frontier
    }

    /// Accuracy gained per extra layer, between consecutive frontier points.
    ///
    /// Where this falls to near zero is where extra test-time compute stops
    /// paying — the number to set a budget from.
    pub fn marginal_returns(&self) -> Vec<(String, f64)> {
        let frontier = self.pareto();
        frontier
            .windows(2)
            .map(|w| {
                let extra = w[1].layers_executed.saturating_sub(w[0].layers_executed);
                let gain = w[1].accuracy - w[0].accuracy;
                let rate = if extra == 0 { 0.0 } else { gain / extra as f64 };
                (w[1].label.clone(), rate)
            })
            .collect()
    }

    /// Human-readable table.
    pub fn render(&self) -> String {
        let frontier: Vec<&str> = self.pareto().iter().map(|p| p.label.as_str()).collect();
        let mut out = String::from("  configuration                 calls   layers   top-1   frontier\n");
        for p in &self.points {
            out.push_str(&format!(
                "  {:<28} {:>5}  {:>7}  {:>6.3}   {}\n",
                p.label,
                p.model_calls,
                p.layers_executed,
                p.accuracy,
                if frontier.contains(&p.label.as_str()) { "*" } else { "" }
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    fn logits(values: &[f32], rows: usize, cols: usize) -> Tensor<B, 2> {
        let device = Default::default();
        Tensor::<B, 1>::from_floats(values, &device).reshape([rows, cols])
    }

    fn values(t: Tensor<B, 2>) -> Vec<f32> {
        t.into_data().convert::<f32>().iter::<f32>().collect()
    }

    #[test]
    fn test_guidance_at_scale_one_is_bitwise_identity() {
        // `u + 1.0 * (c - u)` equals `c` in exact arithmetic but not in
        // floating point, so the identity is short-circuited. Checked bitwise,
        // because an "almost identity" default is a slow leak of accuracy that
        // no one would ever look for.
        let device = Default::default();
        let cond = Tensor::<B, 2>::random([4, 6], Distribution::Uniform(-30.0, 30.0), &device);
        let uncond = Tensor::<B, 2>::random([4, 6], Distribution::Uniform(-30.0, 30.0), &device);

        let out = Guidance::none().apply(cond.clone(), uncond);
        assert_eq!(values(out), values(cond));
    }

    #[test]
    fn test_guidance_endpoints_and_extrapolation() {
        let cond = logits(&[1.0, 3.0], 1, 2);
        let uncond = logits(&[1.0, 1.0], 1, 2);

        // scale 0 discards the conditioning entirely.
        let zero = values(Guidance::new(0.0).apply(cond.clone(), uncond.clone()));
        assert!((zero[1] - 1.0).abs() < 1e-6);

        // scale 2 extrapolates past the conditional estimate, which is the
        // whole point of guidance -- not an interpolation.
        let two = values(Guidance::new(2.0).apply(cond.clone(), uncond.clone()));
        assert!((two[1] - 5.0).abs() < 1e-6, "expected 1 + 2*(3-1) = 5, got {}", two[1]);
    }

    #[test]
    fn test_guidance_rescale_contains_the_inflated_norm() {
        // Large scales inflate the estimate, and in this crate the estimate
        // feeds an ODE step where the inflation compounds. Rescaling is what
        // keeps a strong guidance setting usable.
        let device = Default::default();
        let cond = Tensor::<B, 2>::random([3, 8], Distribution::Uniform(-1.0, 1.0), &device);
        let uncond = Tensor::<B, 2>::random([3, 8], Distribution::Uniform(-1.0, 1.0), &device);

        let raw = Guidance::new(6.0).apply(cond.clone(), uncond.clone());
        let tamed = Guidance::new(6.0).with_rescale(1.0).apply(cond.clone(), uncond);

        let spread = |t: Tensor<B, 2>| -> Vec<f32> {
            row_std(&t).into_data().convert::<f32>().iter::<f32>().collect()
        };
        let target = spread(cond);
        let before = spread(raw);
        let after = spread(tamed);

        for i in 0..3 {
            assert!(before[i] > target[i], "guidance should inflate the spread");
            assert!(
                (after[i] - target[i]).abs() < 1e-4,
                "full rescale should restore it: {} vs {}",
                after[i],
                target[i]
            );
        }
    }

    #[test]
    fn test_logit_normalization_never_moves_the_prediction() {
        // The claim that makes normalization safe to switch on anywhere: it
        // recalibrates confidence and leaves top-1 exactly where it was.
        let device = Default::default();
        let raw = Tensor::<B, 2>::random([16, 10], Distribution::Uniform(-40.0, 40.0), &device);
        let reference: Vec<i64> = raw
            .clone()
            .argmax(1)
            .into_data()
            .convert::<i64>()
            .iter()
            .collect();

        for norm in [
            LogitNorm::Temperature { tau: 0.5 },
            LogitNorm::Temperature { tau: 7.0 },
            LogitNorm::L2 { tau: 1.0 },
            LogitNorm::L2 { tau: 0.1 },
            LogitNorm::Standardize { tau: 1.0 },
            LogitNorm::Standardize { tau: 3.0 },
        ] {
            let got: Vec<i64> = norm
                .apply(raw.clone())
                .argmax(1)
                .into_data()
                .convert::<i64>()
                .iter()
                .collect();
            assert_eq!(got, reference, "{} moved the arg-max", norm.name());
        }
    }

    #[test]
    fn test_logit_normalization_makes_confidence_comparable() {
        // Two checkpoints of the same classifier can differ only in logit
        // scale. A confidence threshold tuned on one is then meaningless on the
        // other -- which is exactly the situation the adaptive strategy and the
        // quality gates are in. Standardization removes the difference.
        let small = logits(&[1.0, 2.0, 0.5], 1, 3);
        let large = logits(&[10.0, 20.0, 5.0], 1, 3);

        let conf = |t: Tensor<B, 2>| -> f32 {
            softmax(t, 1).max_dim(1).into_scalar()
        };
        assert!(
            (conf(small.clone()) - conf(large.clone())).abs() > 0.3,
            "raw confidences should disagree wildly"
        );

        let norm = LogitNorm::Standardize { tau: 1.0 };
        let a = conf(norm.apply(small));
        let b = conf(norm.apply(large));
        assert!(
            (a - b).abs() < 1e-5,
            "standardized confidences should agree: {a} vs {b}"
        );
    }

    #[test]
    fn test_none_is_the_identity() {
        let device = Default::default();
        let raw = Tensor::<B, 2>::random([5, 7], Distribution::Uniform(-5.0, 5.0), &device);
        assert_eq!(values(LogitNorm::None.apply(raw.clone())), values(raw));
        assert!(LogitNorm::None.is_identity());
        assert!(Guidance::none().is_identity());
    }

    #[test]
    fn test_ensembles_produce_distributions() {
        let device = Default::default();
        let members: Vec<Tensor<B, 2>> = (0..3)
            .map(|_| Tensor::<B, 2>::random([4, 5], Distribution::Uniform(-3.0, 3.0), &device))
            .collect();

        for kind in [Ensemble::ProbabilityMean, Ensemble::LogitMean, Ensemble::MajorityVote] {
            let combined = kind.combine(&members);
            assert_eq!(combined.dims(), [4, 5]);
            let sums: Vec<f32> = combined
                .sum_dim(1)
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
                .collect();
            for s in sums {
                assert!((s - 1.0).abs() < 1e-5, "{} row summed to {s}", kind.name());
            }
        }
    }

    #[test]
    fn test_a_single_member_ensemble_is_that_member() {
        // Containment again: ensembling one sampler must cost nothing and
        // change nothing, so a pipeline can be written with the ensemble
        // always in place and the member count as the only knob.
        let device = Default::default();
        let only = Tensor::<B, 2>::random([3, 6], Distribution::Uniform(-4.0, 4.0), &device);
        let reference = values(softmax(only.clone(), 1));

        for kind in [Ensemble::ProbabilityMean, Ensemble::LogitMean] {
            let got = values(kind.combine(std::slice::from_ref(&only)));
            for (a, b) in got.iter().zip(&reference) {
                assert!((a - b).abs() < 1e-6, "{} changed a lone member", kind.name());
            }
        }

        // A one-member vote is the one-hot prediction, which is a different
        // distribution but the same arg-max.
        let voted = Ensemble::MajorityVote.combine(std::slice::from_ref(&only));
        let a: Vec<i64> = voted.argmax(1).into_data().convert::<i64>().iter().collect();
        let b: Vec<i64> = only.argmax(1).into_data().convert::<i64>().iter().collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_identical_members_are_the_identity() {
        let device = Default::default();
        let m = Tensor::<B, 2>::random([3, 4], Distribution::Uniform(-2.0, 2.0), &device);
        let reference = values(softmax(m.clone(), 1));
        let repeated = vec![m.clone(), m.clone(), m];

        let got = values(Ensemble::ProbabilityMean.combine(&repeated));
        for (a, b) in got.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_an_ensemble_can_outvote_a_wrong_member() {
        // The reason to ensemble at all. One member is confidently wrong; two
        // mildly right members must still carry the answer.
        let wrong = logits(&[9.0, 0.0], 1, 2);
        let right_a = logits(&[0.0, 1.0], 1, 2);
        let right_b = logits(&[0.0, 1.2], 1, 2);

        let voted = Ensemble::MajorityVote.combine(&[wrong.clone(), right_a.clone(), right_b.clone()]);
        let idx: Vec<i64> = voted.argmax(1).into_data().convert::<i64>().iter().collect();
        assert_eq!(idx, vec![1], "a plurality vote resists one confident outlier");

        // The logit mean does not: a member with a large logit scale dominates
        // it. That is a documented property, not a bug -- and the reason
        // `LogitNorm` and `Ensemble::LogitMean` belong together.
        let averaged = Ensemble::LogitMean.combine(&[wrong, right_a, right_b]);
        let idx: Vec<i64> = averaged.argmax(1).into_data().convert::<i64>().iter().collect();
        assert_eq!(idx, vec![0]);
    }

    #[test]
    fn test_accuracy_counts_what_it_should() {
        let device = Default::default();
        let l = logits(&[0.1, 0.9, 2.0, 0.0, 0.0, 1.0], 3, 2);
        let labels = Tensor::<B, 1, Int>::from_ints([1i64, 0, 0].as_slice(), &device);
        assert!((accuracy(&l, &labels) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_pareto_frontier_drops_dominated_configurations() {
        // A dominated point is a configuration nobody should ever pick: it
        // costs more and delivers no more. Reporting it alongside the rest
        // would make a scaling study look like a menu of trade-offs when some
        // entries are simply worse.
        let mut curve = ScalingCurve::new();
        curve.push(ScalingPoint::new("cheap", 3, 6, 0.50));
        curve.push(ScalingPoint::new("mid", 6, 12, 0.62));
        curve.push(ScalingPoint::new("wasteful", 9, 20, 0.60)); // dominated by mid
        curve.push(ScalingPoint::new("best", 12, 30, 0.65));
        curve.push(ScalingPoint::new("same-for-more", 15, 40, 0.65)); // dominated by best

        let frontier: Vec<&str> = curve.pareto().iter().map(|p| p.label.as_str()).collect();
        assert_eq!(frontier, vec!["cheap", "mid", "best"]);
        assert_eq!(curve.most_accurate().map(|p| p.label.as_str()), Some("best"));
        assert_eq!(curve.cheapest().map(|p| p.label.as_str()), Some("cheap"));
    }

    #[test]
    fn test_marginal_returns_expose_where_scaling_stops_paying() {
        // "More compute is more accuracy" holds until it does not. The point
        // of measuring is to find where, so the rate must fall as it flattens.
        let mut curve = ScalingCurve::new();
        curve.push(ScalingPoint::new("s1", 2, 10, 0.40));
        curve.push(ScalingPoint::new("s2", 4, 20, 0.60)); // +0.020 / layer
        curve.push(ScalingPoint::new("s3", 8, 40, 0.62)); // +0.001 / layer

        let rates = curve.marginal_returns();
        assert_eq!(rates.len(), 2);
        assert!((rates[0].1 - 0.02).abs() < 1e-9, "got {}", rates[0].1);
        assert!((rates[1].1 - 0.001).abs() < 1e-9, "got {}", rates[1].1);
        assert!(rates[1].1 < rates[0].1 / 10.0, "the curve should flatten");
    }

    #[test]
    fn test_an_empty_curve_says_nothing_rather_than_guessing() {
        let curve = ScalingCurve::new();
        assert!(curve.is_empty());
        assert!(curve.most_accurate().is_none());
        assert!(curve.cheapest().is_none());
        assert!(curve.pareto().is_empty());
        assert!(curve.marginal_returns().is_empty());
    }

    #[test]
    #[should_panic(expected = "at least one member")]
    fn test_an_empty_ensemble_is_an_error_not_a_uniform_guess() {
        let empty: Vec<Tensor<B, 2>> = Vec::new();
        Ensemble::ProbabilityMean.combine(&empty);
    }

    #[test]
    fn test_parsers_round_trip_their_names() {
        for name in ["none", "temperature", "l2", "standardize"] {
            assert_eq!(LogitNorm::parse(name, 1.0).unwrap().name(), name);
        }
        assert!(LogitNorm::parse("nope", 1.0).is_err());

        for name in ["probability", "logit", "vote"] {
            assert_eq!(Ensemble::parse(name).unwrap().name(), name);
        }
        assert!(Ensemble::parse("nope").is_err());
    }
}
