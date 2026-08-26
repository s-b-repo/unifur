//! Adaptive-depth components (roadmap Phase 8): a halting-probability head
//! over per-block hidden states, an expected-depth (ponder) regularizer, and
//! host-side early-exit logic driven by cumulative halting probability
//! (ACT-style, Graves 2016).

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

/// Halting head configuration.
#[derive(Debug, Clone)]
pub struct HaltingConfig {
    pub hidden_size: usize,
    /// Cumulative-probability threshold for early exit at inference.
    pub exit_threshold: f32,
    /// Weight of the expected-depth regularizer during training.
    pub ponder_weight: f32,
}

impl Default for HaltingConfig {
    fn default() -> Self {
        Self { hidden_size: 128, exit_threshold: 0.99, ponder_weight: 0.01 }
    }
}

/// Sigmoid halting head: `p(halt | hidden state)`.
#[derive(Module, Debug)]
pub struct HaltingHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> HaltingHead<B> {
    pub fn new(config: &HaltingConfig, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(config.hidden_size, 1)
                .with_bias(true)
                .init(device),
        }
    }

    /// Per-sample halting probabilities `[b]` from pooled states `[b, h]`.
    pub fn halt_probability(&self, hidden: Tensor<B, 2>) -> Tensor<B, 1> {
        burn::tensor::activation::sigmoid(self.linear.forward(hidden).squeeze_dim::<1>(1))
    }

    /// Expected depth (ponder cost): mean over samples of the summed
    /// halting probabilities across blocks. Minimizing it encourages early
    /// exits; combine with a task loss weighted by [`HaltingConfig::ponder_weight`].
    pub fn expected_depth(&self, block_hiddens: &[Tensor<B, 2>]) -> Tensor<B, 1> {
        let mut acc = None;
        for h in block_hiddens {
            let p = self.halt_probability(h.clone());
            acc = Some(match acc {
                None => p,
                Some(a) => a + p,
            });
        }
        match acc {
            Some(total) => total.mean(),
            None => Tensor::zeros([1], &Default::default()),
        }
    }
}

/// First step where the running halting probability reaches `threshold`.
///
/// Returns `num_blocks` when the budget is never reached (full depth).
pub fn early_exit_step(halting_probs: &[f32], threshold: f32) -> usize {
    let mut cum = 0.0;
    for (i, &p) in halting_probs.iter().enumerate() {
        cum += p.max(0.0);
        if cum >= threshold {
            return i + 1;
        }
    }
    halting_probs.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_early_exit_boundaries() {
        // Reaches threshold exactly at the second block.
        assert_eq!(early_exit_step(&[0.4, 0.6], 1.0), 2);
        // Never reaches: full depth.
        assert_eq!(early_exit_step(&[0.1, 0.1, 0.1], 0.99), 3);
        // Immediate exit.
        assert_eq!(early_exit_step(&[0.999, 0.5, 0.5], 0.99), 1);
        // Negative probabilities are clamped.
        assert_eq!(early_exit_step(&[-1.0, -1.0], 0.5), 2);
        assert_eq!(early_exit_step(&[], 0.5), 0);
    }

    #[test]
    fn test_expected_depth_shapes_and_range() {
        let device = Default::default();
        let head = HaltingHead::<B>::new(&HaltingConfig { hidden_size: 8, ..Default::default() }, &device);
        let hiddens: Vec<Tensor<B, 2>> =
            (0..3).map(|_| Tensor::<B, 2>::random([4, 8], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device)).collect();
        let expected = head.expected_depth(&hiddens);
        assert_eq!(expected.dims(), [1]);
        let v: f32 = expected.into_scalar();
        assert!(v.is_finite() && (0.0..=3.0 + 1e-5).contains(&v), "expected depth {v} out of range");
    }
}
