//! Adaptive-depth components (roadmap Phase 8): a halting-probability head
//! over per-block hidden states, an expected-depth (ponder) regularizer, and
//! host-side early-exit logic driven by cumulative halting probability
//! (ACT-style, Graves 2016).

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};
use burn_nn::{Linear, LinearConfig};

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
        Self {
            hidden_size: 128,
            exit_threshold: 0.99,
            ponder_weight: 0.01,
        }
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

    /// Expected depth (ponder cost): the expected number of blocks executed
    /// when block `n` halts with probability `p_n`, i.e. the sum of the
    /// survival probabilities `sum_n prod_{k<n} (1 - p_k)`, averaged over
    /// samples. It lies in `[1, N]` for `N` blocks and *falls* as the early
    /// halting probabilities rise, so minimizing it encourages early exits;
    /// combine with a task loss weighted by [`HaltingConfig::ponder_weight`].
    ///
    /// (Summing the halting probabilities themselves, as this once did, has
    /// the opposite gradient: minimizing it drives every `p_n` to zero, so
    /// the cumulative halting probability never reaches the exit threshold
    /// and every sample runs at full depth.)
    pub fn expected_depth(&self, block_hiddens: &[Tensor<B, 2>]) -> Tensor<B, 1> {
        let probs: Vec<Tensor<B, 1>> = block_hiddens
            .iter()
            .map(|h| self.halt_probability(h.clone()))
            .collect();
        match expected_depth_from_probs(&probs) {
            Some(depth) => depth,
            // No blocks were supplied: the expected depth is zero, on the
            // head's own device rather than the backend default.
            None => Tensor::zeros([1], &self.linear.weight.device()),
        }
    }
}

/// `mean_b sum_n prod_{k<n} (1 - p_k[b])` over per-block halting
/// probabilities `[b]`; `None` for no blocks.
fn expected_depth_from_probs<B: Backend>(probs: &[Tensor<B, 1>]) -> Option<Tensor<B, 1>> {
    let mut survival: Option<Tensor<B, 1>> = None;
    let mut depth: Option<Tensor<B, 1>> = None;
    for p in probs {
        // Probability that block `n` runs at all: no earlier block halted.
        let alive = match &survival {
            None => p.ones_like(),
            Some(s) => s.clone(),
        };
        depth = Some(match depth {
            None => alive.clone(),
            Some(d) => d + alive.clone(),
        });
        survival = Some(alive * (p.clone().neg() + 1.0));
    }
    depth.map(|d| d.mean())
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
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it (see the `[lints]` table in
// `Cargo.toml` and the contract in the crate docs).
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::dbg_macro,
    clippy::let_underscore_must_use,
    clippy::redundant_pattern_matching,
    clippy::mem_forget,
    clippy::exit,
    clippy::print_stdout,
    clippy::print_stderr
)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

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
        let head = HaltingHead::<B>::new(
            &HaltingConfig {
                hidden_size: 8,
                ..Default::default()
            },
            &device,
        );
        let hiddens: Vec<Tensor<B, 2>> = (0..3)
            .map(|_| {
                Tensor::<B, 2>::random(
                    [4, 8],
                    burn::tensor::Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
            })
            .collect();
        let expected = head.expected_depth(&hiddens);
        assert_eq!(expected.dims(), [1]);
        let v: f32 = expected.into_scalar();
        assert!(
            v.is_finite() && (1.0 - 1e-5..=3.0 + 1e-5).contains(&v),
            "expected depth {v} out of range"
        );
    }

    #[test]
    fn test_expected_depth_falls_as_halting_rises() {
        // The regularizer is only useful if minimizing it means exiting
        // early: always halting at block 1 is depth 1, never halting is the
        // full depth, and p = 0.5 over three blocks is 1 + 0.5 + 0.25.
        let device = Default::default();
        let depth = |p: f32| -> f32 {
            let probs: Vec<Tensor<B, 1>> = (0..3)
                .map(|_| Tensor::<B, 1>::from_floats([p, p].as_slice(), &device))
                .collect();
            expected_depth_from_probs(&probs).unwrap().into_scalar()
        };
        assert!((depth(1.0) - 1.0).abs() < 1e-6);
        assert!((depth(0.0) - 3.0).abs() < 1e-6);
        assert!((depth(0.5) - 1.75).abs() < 1e-6);
        assert!(depth(0.9) < depth(0.5) && depth(0.5) < depth(0.1));
        let empty: Vec<Tensor<B, 1>> = Vec::new();
        assert!(expected_depth_from_probs(&empty).is_none());
    }
}
