//! Compute accounting (roadmap Phases 25-27): active parameters, FLOPs per
//! token and resident state per layer, from a configuration rather than
//! from a stopwatch.
//!
//! Wall-clock on a CPU backend measures the backend, not the architecture:
//! every mode here is executed densely, so a sparse layer costs as much time
//! as a dense one even when it *reads* a fraction of the weights. The
//! roadmap's quality-per-active-FLOP axis therefore needs a cost that is
//! counted, not timed. Every count here is a multiply-add pair per token
//! (`2 * a * b` for a `[a, b]` projection), which is the convention the
//! literature's "active FLOPs" use, and every count is a plain function of
//! the shapes so it can be checked by hand in a certificate.

use crate::hybrid::AttentionMode;
use serde::{Deserialize, Serialize};

/// Cost of one layer for one token, at a given number of visible positions.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct LayerCost {
    /// Parameters touched by this token.
    pub active_params: usize,
    /// Multiply-adds, times two.
    pub flops: f64,
    /// Key/value rows read from the cache or the sequence.
    pub keys_read: usize,
    /// Floats of per-sequence state the layer keeps at decode time.
    pub state_floats: usize,
}

impl LayerCost {
    /// Component-wise sum.
    pub fn plus(self, other: LayerCost) -> LayerCost {
        LayerCost {
            active_params: self.active_params + other.active_params,
            flops: self.flops + other.flops,
            keys_read: self.keys_read + other.keys_read,
            state_floats: self.state_floats + other.state_floats,
        }
    }
}

/// The four projections of an attention layer plus its score/value work for
/// `positions` visible keys, per token.
pub fn attention_cost(mode: AttentionMode, hidden: usize, heads: usize, positions: usize) -> LayerCost {
    let head_dim = hidden / heads.max(1);
    let projections = 4 * hidden * hidden + 4 * hidden; // q, k, v, out with biases
    let proj_flops = 2.0 * (4 * hidden * hidden) as f64;
    let (mix_flops, keys_read, state_floats) = match mode {
        AttentionMode::Dense => (4.0 * positions as f64 * hidden as f64, positions, 2 * positions * hidden),
        AttentionMode::Sliding { window } => {
            let w = window.min(positions);
            (4.0 * w as f64 * hidden as f64, w, 2 * window.saturating_sub(1) * hidden)
        }
        AttentionMode::Retrieval { top_k } => {
            let k = top_k.min(positions);
            // Every score is computed; only the chosen values are read.
            (2.0 * positions as f64 * hidden as f64 + 2.0 * k as f64 * hidden as f64, k, 2 * positions * hidden)
        }
        AttentionMode::Linear => {
            // phi(q) S: [d] x [d, d] per head, plus the state update phi(k) v^T.
            let per_head = 2.0 * (head_dim * head_dim) as f64 * 2.0;
            (per_head * heads as f64, 0, heads * (head_dim * head_dim + head_dim))
        }
        AttentionMode::Learned => {
            let dense = attention_cost(AttentionMode::Dense, hidden, heads, positions);
            let linear = attention_cost(AttentionMode::Linear, hidden, heads, positions);
            return LayerCost {
                active_params: projections + 2,
                flops: proj_flops + (dense.flops - proj_flops) + (linear.flops - proj_flops),
                keys_read: dense.keys_read,
                state_floats: dense.state_floats + linear.state_floats,
            };
        }
    };
    LayerCost { active_params: projections, flops: proj_flops + mix_flops, keys_read, state_floats }
}

/// A dense two-layer MLP, per token.
pub fn dense_mlp_cost(hidden: usize, intermediate: usize) -> LayerCost {
    let params = 2 * hidden * intermediate + hidden + intermediate;
    LayerCost { active_params: params, flops: 2.0 * (2 * hidden * intermediate) as f64, keys_read: 0, state_floats: 0 }
}

/// A routed expert layer, per token: the router (optionally through a latent
/// projection) plus `top_k` expert MLPs. The unselected experts are resident
/// but not active, which is the whole point of the count.
pub fn moe_cost(hidden: usize, intermediate: usize, experts: usize, top_k: usize, router_in: usize, latent: Option<usize>) -> LayerCost {
    let router_params = match latent {
        Some(l) => router_in * l + l * experts + experts,
        None => router_in * experts + experts,
    };
    let router_flops = 2.0 * (router_params - experts) as f64;
    let expert = dense_mlp_cost(hidden, intermediate);
    let k = top_k.min(experts).max(1);
    LayerCost {
        active_params: router_params + k * expert.active_params,
        flops: router_flops + k as f64 * expert.flops,
        keys_read: 0,
        state_floats: 0,
    }
}

/// Value experts over a shared base (Phase 26): the router plus `top_k`
/// rank-`rank` deltas, per token. The base value projection is already in
/// [`attention_cost`].
pub fn value_experts_cost(hidden: usize, rank: usize, experts: usize, top_k: usize, router_in: usize, latent: Option<usize>) -> LayerCost {
    let router_params = match latent {
        Some(l) => router_in * l + l * experts + experts,
        None => router_in * experts + experts,
    };
    let per_delta = 2 * hidden * rank;
    let k = top_k.min(experts).max(1);
    LayerCost {
        active_params: router_params + k * per_delta,
        flops: 2.0 * (router_params - experts) as f64 + k as f64 * 2.0 * per_delta as f64,
        keys_read: 0,
        state_floats: 0,
    }
}

/// The cost of a whole trunk for one token that sees `positions` keys.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TrunkCost {
    pub layers: Vec<LayerCost>,
}

impl TrunkCost {
    pub fn total(&self) -> LayerCost {
        self.layers.iter().fold(LayerCost::default(), |acc, l| acc.plus(*l))
    }

    /// Cost of executing only `span` of the layers.
    pub fn span(&self, span: std::ops::Range<usize>) -> LayerCost {
        self.layers[span.start.min(self.layers.len())..span.end.min(self.layers.len())]
            .iter()
            .fold(LayerCost::default(), |acc, l| acc.plus(*l))
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Counts what a run actually executed, so a cost model can be checked
/// against it rather than trusted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ComputeLedger {
    /// Layer executions, summed over tokens.
    pub layer_token_evaluations: usize,
    /// Full trunk passes.
    pub forward_passes: usize,
    /// Tokens the passes produced.
    pub tokens_emitted: usize,
    /// FLOPs charged by the cost model for those executions.
    pub flops: f64,
}

impl ComputeLedger {
    pub fn charge_layers(&mut self, layers: usize, tokens: usize, cost_per_layer_token: f64) {
        self.layer_token_evaluations += layers * tokens;
        self.flops += layers as f64 * tokens as f64 * cost_per_layer_token;
    }

    pub fn charge_pass(&mut self) {
        self.forward_passes += 1;
    }

    /// Layer evaluations per emitted token: a full trunk of `L` layers is `L`.
    pub fn layers_per_token(&self) -> f64 {
        if self.tokens_emitted == 0 {
            return 0.0;
        }
        self.layer_token_evaluations as f64 / self.tokens_emitted as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_costs_are_hand_checkable() {
        let h = 8;
        let dense = attention_cost(AttentionMode::Dense, h, 2, 10);
        assert_eq!(dense.active_params, 4 * 64 + 32);
        assert_eq!(dense.flops, 2.0 * 256.0 + 4.0 * 10.0 * 8.0);
        assert_eq!(dense.keys_read, 10);
        let sliding = attention_cost(AttentionMode::Sliding { window: 4 }, h, 2, 10);
        assert_eq!(sliding.keys_read, 4);
        assert!(sliding.flops < dense.flops);
        let retrieval = attention_cost(AttentionMode::Retrieval { top_k: 2 }, h, 2, 10);
        assert_eq!(retrieval.keys_read, 2);
        assert!(retrieval.flops < dense.flops && retrieval.flops > sliding.flops);
        let linear = attention_cost(AttentionMode::Linear, h, 2, 10);
        assert_eq!(linear.keys_read, 0);
        assert_eq!(linear.state_floats, 2 * (16 + 4));
        // Linear attention's cost does not grow with the sequence.
        assert_eq!(linear.flops, attention_cost(AttentionMode::Linear, h, 2, 10_000).flops);
        let learned = attention_cost(AttentionMode::Learned, h, 2, 10);
        assert!(learned.flops > dense.flops);
    }

    #[test]
    fn test_moe_and_value_costs_scale_with_top_k_not_experts() {
        let one = moe_cost(8, 16, 4, 1, 12, None);
        let two = moe_cost(8, 16, 4, 2, 12, None);
        let eight = moe_cost(8, 16, 8, 1, 12, None);
        assert_eq!(two.active_params - one.active_params, dense_mlp_cost(8, 16).active_params);
        assert_eq!(eight.active_params - one.active_params, 4 * 12 + 4, "only the router widens");
        let latent = moe_cost(8, 16, 4, 1, 12, Some(2));
        assert!(latent.active_params < one.active_params, "a latent router is cheaper than a full one");
        let v = value_experts_cost(8, 2, 4, 2, 12, None);
        assert_eq!(v.active_params, 12 * 4 + 4 + 2 * (2 * 8 * 2));
        let trunk = TrunkCost { layers: vec![one, two] };
        assert_eq!(trunk.total().active_params, one.active_params + two.active_params);
        assert_eq!(trunk.span(1..2).active_params, two.active_params);
        assert_eq!(trunk.span(0..9).flops, trunk.total().flops);
    }

    #[test]
    fn test_ledger_counts_layer_evaluations() {
        let mut ledger = ComputeLedger::default();
        ledger.charge_layers(4, 3, 10.0);
        ledger.charge_pass();
        ledger.tokens_emitted = 3;
        assert_eq!(ledger.layer_token_evaluations, 12);
        assert_eq!(ledger.flops, 120.0);
        assert_eq!(ledger.layers_per_token(), 4.0);
        assert_eq!(ComputeLedger::default().layers_per_token(), 0.0);
    }
}
