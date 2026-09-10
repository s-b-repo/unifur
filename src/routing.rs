//! Persistent routing state and the common router report (roadmap Phase 25).
//!
//! # Routing state
//!
//! Every router in the trunk -- the FFN expert router, the value-expert
//! router of Phase 26, the mode gate of a learned attention layer -- decides
//! from what it can see. Without a shared state each of them rediscovers the
//! same token specialization from scratch, layer after layer. A small
//! **routing state** carried through the layers,
//!
//! ```text
//! r_l = tanh(W_h LN(h_l) + W_r r_{l-1} + b)
//! ```
//!
//! gives nearby layers something to agree on: it is appended to every
//! router's input, so a decision made at layer `l` can inform layer `l + 1`
//! without the two routers having to recompute it. The state is per token,
//! lives only for the duration of one forward pass (it is a function of the
//! token's own hidden states, so there is nothing to cache across steps),
//! and is bounded by the `tanh`, so a router's input scale does not drift with
//! depth.
//!
//! With the state size at `0` no state is built and every router input is
//! exactly what it was before this phase: the feature is additive.
//!
//! # The common router report
//!
//! Issue #4's D1 asks every router -- FFN, value, attention-mode, depth -- to
//! report through one interface: what it selected, the gates, the entropy,
//! the load, the cost. [`crate::moe::RoutingStats`] is that report; this
//! module adds the two diagnostics a *multi-axis* router needs and a flat MoE
//! never did:
//!
//! - **token stability**: the fraction of adjacent tokens (within one
//!   sequence) that chose the same top-1 expert. High stability on long agent
//!   trajectories is what would justify expert residency / paging
//!   (issue #4 B7); low stability says not to build it.
//! - **layer agreement**: the fraction of tokens whose top-1 expert is the
//!   same in two consecutive routed layers of equal width -- the number the
//!   routing state is meant to raise.

use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor},
};
use serde::{Deserialize, Serialize};

/// Which axis a router decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterKind {
    /// Feed-forward experts (flat MoE or MoSME boxes).
    #[default]
    Ffn,
    /// Value experts of a MoVA attention layer (Phase 26).
    Value,
    /// The dense/linear gate of a learned attention layer.
    AttentionMode,
    /// A depth / exit decision (Phase 27).
    Depth,
    /// A latent-branch aggregator (Phase 32).
    Branch,
}

impl RouterKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ffn => "ffn",
            Self::Value => "value",
            Self::AttentionMode => "attention",
            Self::Depth => "depth",
            Self::Branch => "branch",
        }
    }
}

/// The per-token state carried through the layers.
#[derive(Module, Debug)]
pub struct RoutingState<B: Backend> {
    norm: LayerNorm<B>,
    /// `[hidden + size, size]`.
    update: Linear<B>,
    size: usize,
}

impl<B: Backend> RoutingState<B> {
    pub fn new(hidden_size: usize, size: usize, device: &B::Device) -> Self {
        assert!(size > 0, "a routing state needs a positive size");
        Self {
            norm: LayerNormConfig::new(hidden_size).init(device),
            update: LinearConfig::new(hidden_size + size, size).with_bias(true).init(device),
            size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// `r_l` from the layer's input `[b, n, hidden]` and `r_{l-1}`
    /// `[b, n, size]` (zeros for the first routed layer).
    pub fn step(&self, hidden: &Tensor<B, 3>, previous: Option<&Tensor<B, 3>>) -> Tensor<B, 3> {
        let [b, n, _] = hidden.dims();
        let device = hidden.device();
        let prev = match previous {
            Some(p) => p.clone(),
            None => Tensor::<B, 3>::zeros([b, n, self.size], &device),
        };
        let input = Tensor::cat(vec![self.norm.forward(hidden.clone()), prev], 2);
        self.update.forward(input).tanh()
    }
}

/// Fraction of adjacent positions (within each of `batch` sequences of
/// `seq_len` positions) whose top-1 choice is the same. `top1` is `[T, 1]`
/// in row-major `[batch, seq_len]` order. `1.0` for sequences of length 1.
pub fn token_stability<B: Backend>(top1: &Tensor<B, 2, Int>, batch: usize, seq_len: usize) -> f32 {
    if seq_len < 2 || batch == 0 {
        return 1.0;
    }
    let ids: Vec<i64> = top1.clone().into_data().convert::<i64>().iter::<i64>().collect();
    assert_eq!(ids.len(), batch * seq_len, "top1 must cover batch x seq_len positions");
    let mut same = 0usize;
    let mut pairs = 0usize;
    for b in 0..batch {
        let row = &ids[b * seq_len..(b + 1) * seq_len];
        for w in row.windows(2) {
            pairs += 1;
            if w[0] == w[1] {
                same += 1;
            }
        }
    }
    same as f32 / pairs.max(1) as f32
}

/// Fraction of positions whose top-1 choice agrees between two routed layers
/// of the same width; `None` when the widths differ (the experts are not
/// comparable) or the lengths do.
pub fn layer_agreement(a: &[i64], b: &[i64], width_a: usize, width_b: usize) -> Option<f32> {
    if width_a != width_b || a.len() != b.len() || a.is_empty() {
        return None;
    }
    let same = a.iter().zip(b).filter(|(x, y)| x == y).count();
    Some(same as f32 / a.len() as f32)
}

/// Top-1 indices `[T, 1]` to a host vector.
pub fn top1_to_host<B: Backend>(top1: &Tensor<B, 2, Int>) -> Vec<i64> {
    top1.clone().into_data().convert::<i64>().iter::<i64>().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    #[test]
    fn test_routing_state_is_bounded_and_carries_history() {
        let device = Default::default();
        let state = RoutingState::<B>::new(8, 4, &device);
        let h = Tensor::<B, 3>::random([2, 3, 8], Distribution::Normal(0.0, 5.0), &device);
        let r0 = state.step(&h, None);
        assert_eq!(r0.dims(), [2, 3, 4]);
        let max: f32 = r0.clone().abs().max().into_scalar();
        assert!(max < 1.0, "tanh keeps the state inside (-1, 1): {max}");

        // The same hidden state with a different history gives a different state:
        // the recurrence is not ignoring r_{l-1}.
        let history = Tensor::<B, 3>::full([2, 3, 4], 0.9, &device);
        let r1 = state.step(&h, Some(&history));
        let moved: f32 = (r1 - r0).abs().max().into_scalar();
        assert!(moved > 1e-4, "history must influence the state: {moved}");
        assert_eq!(state.size(), 4);
    }

    #[test]
    fn test_stability_and_agreement_read_as_specified() {
        let device = Default::default();
        let ids = |v: &[i64]| Tensor::<B, 1, Int>::from_ints(v, &device).reshape([v.len(), 1]);
        // Two sequences of four: [0,0,0,0] is perfectly stable, [0,1,0,1] never.
        let top1 = ids(&[0, 0, 0, 0, 0, 1, 0, 1]);
        assert!((token_stability(&top1, 2, 4) - 0.5).abs() < 1e-6);
        assert_eq!(token_stability(&ids(&[0, 1]), 2, 1), 1.0, "length-1 sequences are trivially stable");
        // A boundary between sequences is not a pair: [..,0] then [1,..] must not count.
        let boundary = ids(&[0, 0, 1, 1]);
        assert_eq!(token_stability(&boundary, 2, 2), 1.0);

        assert_eq!(layer_agreement(&[0, 1, 2], &[0, 1, 0], 3, 3), Some(2.0 / 3.0));
        assert_eq!(layer_agreement(&[0, 1], &[0, 1], 2, 3), None, "different widths are incomparable");
        assert_eq!(layer_agreement(&[], &[], 2, 2), None);
        assert_eq!(top1_to_host(&ids(&[3, 1])), vec![3, 1]);
        assert_eq!(RouterKind::Value.name(), "value");
    }
}
