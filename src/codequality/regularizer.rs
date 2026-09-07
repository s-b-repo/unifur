//! Quality regularizer (roadmap Phase 25).
//!
//! Where [`super::filter`] decides which windows to train on at all, the
//! regularizer adds a continuous pull toward a target quality: the loss
//! includes `weight * mean((target - score(x))^2)`, which is positive
//! whenever the batch's mean quality is below the target and zero when
//! the batch already meets it.
//!
//! # Why this is *not* `-log(1 - p)`
//!
//! The unlikelihood term in [`crate::lm`] charges per flagged token; the
//! regularizer pulls the *batch* toward a target. They are complementary
//! signals: the per-token term rewards the model for not generating a
//! specific bad pattern; the regularizer rewards it for generating
//! higher-quality code on average, however "quality" is measured.
//!
//! # Identity at strength zero
//!
//! At `weight = 0.0` the regularizer contributes nothing to the loss,
//! and the gradient with respect to the model is unchanged. That
//! containment is the certificate `codequality/identity_at_strength_zero`
//! checks: enabling the regularizer with a zero weight is bit-exact no-op
//! on the loss, so a CLI flag that defaults to zero cannot make the first
//! step worse than not enabling it.

use burn::tensor::{backend::Backend, Tensor};

use super::QualityScore;

/// `(target - score)^2` summed and scaled by `weight`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityRegularizer {
    /// Coefficient on the squared deviation from `target_score`.
    pub weight: f32,
    /// The score in `[0, 1]` the regularizer pulls the batch mean toward.
    pub target_score: f32,
}

impl Default for QualityRegularizer {
    fn default() -> Self {
        Self::off()
    }
}

impl QualityRegularizer {
    /// Zero weight, target `1.0`. The default state: no contribution, no
    /// pull, no surprise for a run that did not enable the feature.
    pub fn off() -> Self {
        Self {
            weight: 0.0,
            target_score: 1.0,
        }
    }

    pub fn new(weight: f32, target_score: f32) -> Self {
        Self {
            weight: weight.max(0.0),
            target_score: target_score.clamp(0.0, 1.0),
        }
    }

    /// Whether this regularizer contributes nothing to the loss.
    pub fn is_off(&self) -> bool {
        self.weight == 0.0
    }

    /// Build the loss term for a batch of per-window scores.
    ///
    /// Returns a `[1]` tensor (so it broadcasts cleanly with a per-token
    /// loss) and the pre-tensor scalar the regularizer expects. The mean
    /// is taken over the batch dimension because the unlikelihood term is
    /// also a mean, so the two live on the same scale and their weights
    /// can be compared directly.
    pub fn loss<B: Backend<FloatElem = f32>>(
        &self,
        scores: &[QualityScore],
        device: &B::Device,
    ) -> (Tensor<B, 1>, f32) {
        if scores.is_empty() {
            return (Tensor::<B, 1>::zeros([1], device), 0.0);
        }
        let mean_score = scores.iter().map(|s| s.overall).sum::<f32>() / scores.len() as f32;
        let deviation = self.target_score - mean_score;
        let value = self.weight * deviation * deviation;
        (Tensor::<B, 1>::from_floats([value], device), value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codequality::{Dimension, Language};
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn score(overall: f32) -> QualityScore {
        let mut s = QualityScore::from_dimensions(
            Language::Rust,
            vec![Dimension::new("synthetic", overall)],
            1,
        );
        s.overall = overall.clamp(0.0, 1.0);
        s
    }

    #[test]
    fn test_identity_at_strength_zero() {
        // The containment every "off by default" feature must satisfy:
        // weight zero is bitwise identical to a zero tensor.
        let reg = QualityRegularizer::off();
        assert!(reg.is_off());
        let device = Default::default();
        let scores = vec![score(0.3), score(0.7), score(0.9)];
        let (loss, scalar) = reg.loss::<B>(&scores, &device);
        let v: Vec<f32> = loss.into_data().convert::<f32>().iter::<f32>().collect();
        assert_eq!(v[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(scalar, 0.0);
    }

    #[test]
    fn test_zero_deviation_is_zero_loss() {
        // Batch mean exactly at the target: no pull, no penalty.
        let reg = QualityRegularizer::new(1.0, 0.7);
        let device = Default::default();
        let scores = vec![score(0.6), score(0.7), score(0.8)];
        let (loss, scalar) = reg.loss::<B>(&scores, &device);
        let v: Vec<f32> = loss.into_data().convert::<f32>().iter::<f32>().collect();
        assert!(
            scalar.abs() < 1e-6,
            "loss must be zero at the target: {scalar}"
        );
        assert!(v[0].abs() < 1e-6);
    }

    #[test]
    fn test_loss_is_weighted_squared_deviation() {
        let reg = QualityRegularizer::new(0.5, 1.0);
        let device = Default::default();
        let scores = vec![score(0.0), score(0.5), score(1.0)];
        // Mean 0.5; deviation 0.5; squared 0.25; weight 0.5 -> 0.125.
        let (loss, scalar) = reg.loss::<B>(&scores, &device);
        let v: Vec<f32> = loss.into_data().convert::<f32>().iter::<f32>().collect();
        assert!((scalar - 0.125).abs() < 1e-6, "scalar {scalar}");
        assert!((v[0] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_target_is_clamped_into_unit_interval() {
        let reg = QualityRegularizer::new(1.0, 2.5);
        assert_eq!(reg.target_score, 1.0);
        let reg = QualityRegularizer::new(1.0, -0.5);
        assert_eq!(reg.target_score, 0.0);
    }

    #[test]
    fn test_negative_weight_is_clamped_to_zero() {
        let reg = QualityRegularizer::new(-3.0, 0.5);
        assert_eq!(reg.weight, 0.0);
        assert!(reg.is_off());
    }

    #[test]
    fn test_empty_batch_returns_zero() {
        let reg = QualityRegularizer::new(2.0, 0.5);
        let device = Default::default();
        let (loss, scalar) = reg.loss::<B>(&[], &device);
        let v: Vec<f32> = loss.into_data().convert::<f32>().iter::<f32>().collect();
        assert_eq!(v[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(scalar, 0.0);
    }

    #[test]
    fn test_regularizer_is_symmetric_around_the_target() {
        // The pull is the same magnitude whether the batch is above or
        // below the target: a batch that's too good is also a deviation,
        // which is what makes the regularizer useful for both directions.
        let device = Default::default();
        let target = 0.6;
        let high = vec![score(0.9), score(0.9), score(0.9)];
        let low = vec![score(0.3), score(0.3), score(0.3)];
        let reg = QualityRegularizer::new(1.0, target);
        let (_, high_scalar) = reg.loss::<B>(&high, &device);
        let (_, low_scalar) = reg.loss::<B>(&low, &device);
        assert!(
            (high_scalar - low_scalar).abs() < 1e-6,
            "{high_scalar} vs {low_scalar}"
        );
    }
}
