//! Pre-training data filter (roadmap Phase 25).
//!
//! The unlikelihood term in [`crate::lm`] charges the model for predicting
//! flagged tokens but does not prevent it from seeing the patterns: a
//! window that is 80% `except: pass` is still 80% training signal for
//! `except: pass`. This module is the upstream half of the same idea --
//! decide, per window, whether to keep the window at all, and if so at
//! what weight.
//!
//! Three policies:
//!
//! - [`Policy::Keep`]: every window passes through, with weight `1.0`. The
//!   default; the rest of the pipeline pays nothing when filtering is off.
//! - [`Policy::Downweight`]: weights in `[floor, ceiling]` are derived from
//!   the score. `floor == ceiling == 1.0` is exactly [`Policy::Keep`].
//! - [`Policy::Drop`]: any window whose `overall <= drop_below` is
//!   removed from the batch. A `Drop` policy with `drop_below = 0.0`
//!   drops nothing; a `Drop` policy with `drop_below = 1.0` drops
//!   everything -- the policy is the lever, not the threshold.

use super::QualityScore;

/// How a [`WindowFilter`] turns a [`QualityScore`] into a keep/drop decision
/// and a per-window weight.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Policy {
    /// No filtering. Every window is kept with weight `1.0`.
    #[default]
    Keep,
    /// Score `s` becomes weight `floor + (ceiling - floor) * s`.
    ///
    /// `floor == ceiling == 1.0` is exactly [`Policy::Keep`]: the
    /// containment matters because a `--quality-filter downweight` flag
    /// that the user enabled for diagnostics should not change what is
    /// trained on.
    Downweight { floor: f32, ceiling: f32 },
    /// Drop the window if `overall <= drop_below`. Otherwise keep it with
    /// weight `1.0`.
    Drop { drop_below: f32 },
}

impl Policy {
    /// Whether this policy drops any window for any score.
    pub fn is_identity(&self) -> bool {
        match *self {
            Self::Keep => true,
            Self::Downweight { floor, ceiling } => floor == 1.0 && ceiling == 1.0,
            Self::Drop { drop_below } => drop_below <= 0.0,
        }
    }

    /// The weight this policy assigns to a window with overall score `s`.
    /// `s` is clamped to `[0, 1]` defensively.
    pub fn weight(&self, s: f32) -> f32 {
        let s = s.clamp(0.0, 1.0);
        match *self {
            Self::Keep => 1.0,
            Self::Downweight { floor, ceiling } => {
                let lo = floor.min(ceiling);
                let hi = floor.max(ceiling);
                lo + (hi - lo) * s
            }
            Self::Drop { .. } => 1.0,
        }
    }

    /// Whether this policy drops a window with overall score `s`.
    pub fn drops(&self, s: f32) -> bool {
        let s = s.clamp(0.0, 1.0);
        // Strict inequality: `drop_below == 0.0` therefore drops nothing,
        // which is the containment a `--quality-filter drop --threshold 0`
        // flag relies on. The threshold is the *first* score that survives,
        // not the *last* that is dropped.
        matches!(*self, Self::Drop { drop_below } if s < drop_below)
    }
}

/// The configured filter.
#[derive(Debug, Clone, Copy)]
pub struct WindowFilter {
    pub policy: Policy,
}

impl Default for WindowFilter {
    fn default() -> Self {
        Self {
            policy: Policy::Keep,
        }
    }
}

impl WindowFilter {
    pub fn keep() -> Self {
        Self::default()
    }

    pub fn downweight(floor: f32, ceiling: f32) -> Self {
        Self {
            policy: Policy::Downweight { floor, ceiling },
        }
    }

    pub fn drop_below(threshold: f32) -> Self {
        Self {
            policy: Policy::Drop {
                drop_below: threshold,
            },
        }
    }

    /// Decide which windows survive and what weight each carries.
    ///
    /// `scores` is one entry per window in the batch, in the same order as
    /// the tokens and labels the caller is about to stack into a tensor.
    /// Returns `(kept_indices, weights)` such that `kept_indices.len()
    /// == weights.len()` and `kept_indices` is strictly increasing -- the
    /// caller can `select` along the batch dimension in that order.
    pub fn decide(&self, scores: &[QualityScore]) -> (Vec<usize>, Vec<f32>) {
        let mut indices = Vec::with_capacity(scores.len());
        let mut weights = Vec::with_capacity(scores.len());
        for (i, score) in scores.iter().enumerate() {
            if self.policy.drops(score.overall) {
                continue;
            }
            indices.push(i);
            weights.push(self.policy.weight(score.overall));
        }
        (indices, weights)
    }

    /// Whether this filter is a no-op for every score.
    pub fn is_identity(&self) -> bool {
        self.policy.is_identity()
    }
}

/// How many windows a filter run dropped.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FilterReport {
    pub evaluated: usize,
    pub dropped: usize,
    pub weighted: usize,
}

impl FilterReport {
    pub fn from_decide(evaluated: usize, kept: usize) -> Self {
        Self {
            evaluated,
            dropped: evaluated - kept,
            weighted: 0,
        }
    }

    pub fn with_weights(mut self, weighted: usize) -> Self {
        self.weighted = weighted;
        self
    }

    pub fn drop_rate(&self) -> f32 {
        if self.evaluated == 0 {
            0.0
        } else {
            self.dropped as f32 / self.evaluated as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codequality::{Dimension, Language};

    fn score(overall: f32) -> QualityScore {
        // Construct from dimensions, then override `overall` directly. A
        // constant `1.0` dimension with the desired `overall` keeps the
        // production constructor in charge of the geometric-mean math.
        let mut s = QualityScore::from_dimensions(
            Language::Rust,
            vec![Dimension::new("synthetic", overall)],
            1,
        );
        s.overall = overall.clamp(0.0, 1.0);
        s
    }

    #[test]
    fn test_keep_policy_is_exactly_identity() {
        let filter = WindowFilter::keep();
        for s in [0.0f32, 0.3, 0.5, 1.0] {
            let score = score(s);
            let (kept, weights) = filter.decide(std::slice::from_ref(&score));
            assert_eq!(kept, vec![0]);
            assert_eq!(weights, vec![1.0]);
            assert!(!filter.policy.drops(score.overall));
        }
        assert!(filter.is_identity());
    }

    #[test]
    fn test_downweight_at_one_one_is_exactly_keep() {
        // Containment: enabling downweight with floor=ceiling=1 must not
        // change what is trained on. The CLI uses this to mean "measure
        // only".
        let filter = WindowFilter::downweight(1.0, 1.0);
        for s in [0.0f32, 0.5, 1.0] {
            let (kept, weights) = filter.decide(&[score(s)]);
            assert_eq!(kept, vec![0]);
            assert_eq!(weights, vec![1.0]);
        }
        assert!(filter.is_identity());
    }

    #[test]
    fn test_downweight_is_monotone_in_score() {
        let filter = WindowFilter::downweight(0.1, 1.0);
        let mut prev = f32::NEG_INFINITY;
        for s in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let w = filter.policy.weight(s);
            assert!(
                w >= prev,
                "weight must be monotone in score: {w} after {prev}"
            );
            prev = w;
        }
        assert!((filter.policy.weight(0.0) - 0.1).abs() < 1e-6);
        assert!((filter.policy.weight(1.0) - 1.0).abs() < 1e-6);
        assert!((filter.policy.weight(0.5) - 0.55).abs() < 1e-6);
    }

    #[test]
    fn test_downweight_respects_floor_and_ceiling_ordering() {
        // A user might pass (floor, ceiling) in either order; the policy
        // sorts internally so the math does not silently flip.
        let f = WindowFilter::downweight(0.9, 0.2);
        assert!((f.policy.weight(0.0) - 0.2).abs() < 1e-6);
        assert!((f.policy.weight(1.0) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_drop_below_threshold_is_strict() {
        let filter = WindowFilter::drop_below(0.5);
        // Strict inequality: `drop_below` is the first score that
        // *survives*, so a window at the threshold still gets trained on.
        for s in [0.0f32, 0.25, 0.4999] {
            assert!(filter.policy.drops(s), "score {s} must be dropped");
        }
        assert!(!filter.policy.drops(0.5), "score at threshold must survive");
        for s in [0.5001f32, 0.7, 1.0] {
            assert!(!filter.policy.drops(s), "score {s} must survive");
        }
    }

    #[test]
    fn test_drop_below_zero_drops_nothing() {
        let filter = WindowFilter::drop_below(0.0);
        assert!(filter.is_identity());
        for s in [0.0f32, 0.5, 1.0] {
            assert!(!filter.policy.drops(s));
        }
    }

    #[test]
    fn test_decide_returns_strictly_increasing_indices() {
        let filter = WindowFilter::drop_below(0.5);
        let scores: Vec<QualityScore> = (0..10)
            .map(|i| {
                let s = i as f32 / 9.0;
                score(s)
            })
            .collect();
        let (kept, weights) = filter.decide(&scores);
        for pair in kept.windows(2) {
            assert!(
                pair[0] < pair[1],
                "kept indices must be strictly increasing"
            );
        }
        assert_eq!(kept.len(), weights.len());
        // The first 5 (scores 0.0..=0.5) are dropped.
        assert_eq!(kept, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_decide_on_an_empty_batch_is_empty() {
        let (kept, weights) = WindowFilter::keep().decide(&[]);
        assert!(kept.is_empty() && weights.is_empty());
    }

    #[test]
    fn test_filter_report_drop_rate_handles_zero_evaluated() {
        let report = FilterReport::default();
        assert_eq!(report.drop_rate(), 0.0);
    }

    #[test]
    fn test_filter_report_drop_rate_is_exact() {
        let report = FilterReport::from_decide(4, 1);
        assert_eq!(report.dropped, 3);
        assert!((report.drop_rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_a_score_above_one_or_below_zero_is_clamped() {
        let filter = WindowFilter::downweight(0.0, 1.0);
        // Out-of-range inputs must not break the policy.
        assert!((filter.policy.weight(-1.0) - 0.0).abs() < 1e-6);
        assert!((filter.policy.weight(2.0) - 1.0).abs() < 1e-6);
    }
}
