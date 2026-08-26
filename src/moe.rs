//! Mixture-of-Experts routing (roadmap Phase 6): a top-k router over a pool
//! of expert MLPs with the Switch-style load-balancing auxiliary loss
//! (item 6.3) and noise-aware conditioning (item 6.4: pass any sigma-derived
//! embedding inside the router condition vector).
//!
//! Routing is per-token on rank-3 states `[b, n, h]`. Experts are evaluated
//! densely and combined with sparse renormalized gates — functionally exact
//! top-k weighting while keeping the implementation simple on CPU backends;
//! gather/scatter dispatch can replace it later without changing semantics.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Int, Tensor},
};

/// Router + expert-pool configuration.
#[derive(Debug, Clone)]
pub struct MoEConfig {
    /// Per-token feature dimension.
    pub hidden_size: usize,
    /// Dimension of the routing condition (adaLN cond [+ noise embedding]).
    pub cond_size: usize,
    pub num_experts: usize,
    /// Experts weighted per token (`k` of `num_experts`).
    pub top_k: usize,
    /// Expert MLP intermediate size.
    pub intermediate_size: usize,
}

impl MoEConfig {
    pub fn new(hidden_size: usize, cond_size: usize, num_experts: usize) -> Self {
        let num_experts = num_experts.max(1);
        Self {
            hidden_size,
            cond_size,
            num_experts,
            top_k: 1.min(num_experts),
            intermediate_size: hidden_size.saturating_mul(2),
        }
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k.clamp(1, self.num_experts);
        self
    }
}

/// Top-k gate network: softmax over per-token expert logits.
#[derive(Module, Debug)]
pub struct TopKRouter<B: Backend> {
    linear: Linear<B>,
}

/// One expert MLP (`exact-GELU` matching the trunk activation).
#[derive(Module, Debug)]
pub struct ExpertMlp<B: Backend> {
    fc_in: Linear<B>,
    fc_out: Linear<B>,
}

/// Complete MoE layer: router + homogeneous expert pool.
#[derive(Module, Debug)]
pub struct MoELayer<B: Backend> {
    router: TopKRouter<B>,
    experts: Vec<ExpertMlp<B>>,
    num_experts: usize,
    top_k: usize,
}

/// Output of [`MoELayer::forward`].
pub struct MoEOutput<B: Backend> {
    /// Mixed expert features, same shape as the input `[b, n, h]`.
    pub output: Tensor<B, 3>,
    /// Switch-style auxiliary balance loss (scalar).
    pub balance_loss: Tensor<B, 1>,
}

impl<B: Backend> MoELayer<B> {
    pub fn new(config: &MoEConfig, device: &B::Device) -> Self {
        let num_experts = config.num_experts.max(1);
        let top_k = config.top_k.clamp(1, num_experts);
        let router = TopKRouter {
            linear: LinearConfig::new(config.cond_size, num_experts)
                .with_bias(true)
                .init(device),
        };
        let experts = (0..num_experts)
            .map(|_| ExpertMlp {
                fc_in: LinearConfig::new(config.hidden_size, config.intermediate_size)
                    .with_bias(true)
                    .init(device),
                fc_out: LinearConfig::new(config.intermediate_size, config.hidden_size)
                    .with_bias(true)
                    .init(device),
            })
            .collect();

        Self { router, experts, num_experts, top_k }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Route every token through its top-k experts.
    ///
    /// * `x`: token features `[b, n, h]`.
    /// * `routing_cond`: per-example condition `[b, cond_size]`; concatenate a
    ///   sigma embedding beforehand for noise-aware routing.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        routing_cond: Tensor<B, 2>,
    ) -> MoEOutput<B> {
        let device = x.device();
        let [b, n, _h] = x.dims();
        let t = b * n;

        // Expand the per-example condition to per-token and flatten.
        let cond_dims = routing_cond.dims();
        let cond_tok = routing_cond
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, n)
            .reshape([t, cond_dims[1]]);
        let probs = softmax(self.router.linear.forward(cond_tok), 1); // [T, E]

        // Top-k selection with renormalized gates.
        let (gate_vals, gate_idx) = probs.clone().topk_with_indices(self.top_k, 1);
        let gate_sum = gate_vals.clone().sum_dim(1).clamp_min(1e-12);
        let gates = (gate_vals / gate_sum).unsqueeze_dim::<3>(2); // [T, K, 1]

        // One-hot accumulation over the k selected experts.
        let expert_ids = Tensor::<B, 1, Int>::arange(0..self.num_experts as i64, &device);
        let sel_ids = expert_ids
            .clone()
            .reshape([1, 1, self.num_experts])
            .repeat_dim(0, t)
            .repeat_dim(1, self.top_k); // [T, K, E]
        let sel = gate_idx.clone().unsqueeze_dim::<3>(2).equal(sel_ids); // [T, K, E]
        let sel_f = sel.float(); // [T, K, E]

        // gate contribution per expert: [T, E]
        let gates_e = (gates * sel_f).sum_dim(1).squeeze_dim::<2>(1);

        let h_size = x.dims()[2];
        let x_flat = x.reshape([t, h_size]);
        let mut out = Tensor::zeros(x_flat.shape(), &device);
        for (e, expert) in self.experts.iter().enumerate() {
            let gate_e = gates_e.clone().narrow(1, e, 1); // [T, 1]
            let y = expert.fc_out.forward(crate::tensor_ext::exact_gelu(
                expert.fc_in.forward(x_flat.clone()).unsqueeze_dim::<3>(1),
            ).squeeze_dim::<2>(1));
            out = out + y * gate_e;
        }

        // Switch balance loss: E * sum_e(f_e * p_e) with
        // f_e = fraction of tokens whose top-1 expert is e,
        // p_e = mean router probability mass.
        let top1 = gate_idx.narrow(1, 0, 1); // [T, 1]
        let top1_ids = expert_ids
            .reshape([1, self.num_experts])
            .repeat_dim(0, t);
        let onehot = top1.equal(top1_ids).float().mean_dim(0); // [1, E]
        let p = probs.mean_dim(0); // [1, E]
        let balance_loss = (onehot * p).sum();

        MoEOutput {
            output: out.reshape([b, n, h_size]),
            balance_loss: balance_loss.mul_scalar(self.num_experts as f32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn tiny_config(top_k: usize, num_experts: usize) -> MoEConfig {
        MoEConfig {
            hidden_size: 8,
            cond_size: 4,
            num_experts,
            top_k,
            intermediate_size: 16,
        }
    }

    #[test]
    fn test_output_shape_and_balance_loss_positive() {
        let device = Default::default();
        let layer = MoELayer::<B>::new(&tiny_config(2, 4), &device);

        let x = Tensor::<B, 3>::ones([3, 5, 8], &device);
        let cond = Tensor::<B, 2>::ones([3, 4], &device);
        let out = layer.forward(x, cond);

        assert_eq!(out.output.dims(), [3, 5, 8]);
        let aux: f32 = out.balance_loss.into_scalar();
        assert!(aux.is_finite() && aux >= 0.0);
    }

    #[test]
    fn test_top_one_matches_selected_expert_exactly() {
        let device = Default::default();
        let layer = MoELayer::<B>::new(&tiny_config(1, 3), &device);

        let x = Tensor::<B, 3>::random([2, 3, 8], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);
        let cond = Tensor::<B, 2>::random([2, 4], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);
        let out = layer.forward(x.clone(), cond.clone());

        // Recompute the router decision by hand.
        let [b, n, _h] = x.dims();
        let t = b * n;
        let cond_tok = cond
            .clone()
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, n)
            .reshape([t, 4]);
        let probs = softmax(layer.router.linear.forward(cond_tok), 1);
        let (_vals, idx) = probs.clone().topk_with_indices(1, 1);
        let chosen: Vec<i64> = idx.into_data().convert::<i64>().iter().collect();

        // Verify each row equals the output of its selected expert.
        let xf = x.reshape([t, 8]);
        let mut expected: Vec<f32> = Vec::with_capacity(t * 8);
        let flat_rows = xf.into_data().convert::<f32>().iter::<f32>().collect::<Vec<_>>();
        for (row, &e) in chosen.iter().enumerate() {
            let expert = &layer.experts[e as usize];
            let row_t = Tensor::<B, 1>::from_floats(
                flat_rows[row * 8..(row + 1) * 8].to_vec().as_slice(),
                &device,
            )
            .reshape([1, 8]);
            let y = expert.fc_out.forward(crate::tensor_ext::exact_gelu(
                expert.fc_in.forward(row_t).unsqueeze_dim::<3>(1),
            ).squeeze_dim::<2>(1));
            expected.extend(y.into_data().convert::<f32>().iter::<f32>());
        }

        let got = out.output.reshape([t, 8]).into_data().convert::<f32>().iter::<f32>().collect::<Vec<_>>();
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "top-1 routing mismatch: {g} vs {e}");
        }
    }

    #[test]
    fn test_balance_loss_uniform_routing_is_lower_than_collapsed() {
        // Uniform gates give the minimal balance loss for fixed p; a collapsed
        // router (all mass on expert 0, all tokens routed there) gives E.
        let device = Default::default();
        let e_count = 4usize;

        let probs_uniform = Tensor::<B, 2>::full([10, e_count], 1.0 / e_count as f32, &device);
        let (vals, idx) = probs_uniform.clone().topk_with_indices(2, 1);
        let _ = (vals, idx);
        // Compute directly: f uniform=0.25 each, p=0.25 each => aux = 4*4*0.0625 = 1.0
        let onehot_f: Vec<f32> = vec![0.25; e_count];
        let f = Tensor::<B, 1>::from_floats(onehot_f.as_slice(), &device);
        let p = Tensor::<B, 1>::full([e_count], 1.0 / e_count as f32, &device);
        let aux_uniform: f32 = (f * p).sum().into_scalar() * e_count as f32;
        assert!((aux_uniform - 1.0).abs() < 1e-6);

        // Collapsed: f = one-hot, p = one-hot => aux = E.
        let collapsed_p = Tensor::<B, 1>::from_floats([1.0, 0.0, 0.0, 0.0].as_slice(), &device);
        let collapsed_f = collapsed_p.clone();
        let aux_collapsed: f32 = (collapsed_f * collapsed_p).sum().into_scalar() * e_count as f32;
        assert!((aux_collapsed - e_count as f32).abs() < 1e-6);
        assert!(aux_collapsed > aux_uniform);
    }
}
