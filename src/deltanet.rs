//! Gated DeltaNet reference core (Phase C of `docs/Frontier-27B-Plan.md`).
//!
//! The gated delta rule (Yang et al., arXiv:2412.06464, Eq. 10), which is
//! what Qwen3.8's linear layers implement per `QwenGatedDeltaNetAttention`
//! in vLLM:
//!
//! ```text
//! S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
//! o_t = S_t q_t
//! ```
//!
//! with L2-normalized queries and keys, a short causal depthwise convolution
//! plus SiLU on the q/k/v paths, and data-dependent `alpha, beta in (0, 1)`.
//! Qwen3.8 groups heads as 16 groups x (1 Q head dim-128, 1 K head dim-128,
//! 3 V heads dim-128); `alpha`/`beta` are per V head (48 total), from the
//! `in_proj_ba` split. The core below is one (k, v) pair; callers pair one
//! shared k/q per group with each of its V heads.
//!
//! This module is the *reference* implementation: exact recurrent math plus
//! the convolution, in plain tensor ops, with certificates pinning the two
//! degenerate cases (pure delta at `alpha = 1`, pure decay at `beta = 0`)
//! and the closed-form first output. The hardware-efficient chunkwise form
//! (WY representation, paper §3.3) is training-time work built on top of
//! these identities, not beside them. Decode-time state is just `S`
//! (`[dv, dk]` floats per pair) carried across steps.

use burn::tensor::{backend::Backend, Tensor};

/// Shape of one Gated DeltaNet layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaConfig {
    /// Key/query groups (16 on Qwen3.8).
    pub num_k_groups: usize,
    /// Value heads per group (3 on Qwen3.8: 48 V heads over 16 groups).
    pub v_per_group: usize,
    /// Key/query head dim (128).
    pub head_k_dim: usize,
    /// Value head dim (128).
    pub head_v_dim: usize,
    /// Short-convolution kernel (4).
    pub conv_kernel: usize,
}

impl GatedDeltaConfig {
    /// Qwen3.8-27B linear layers (`linear_num_key_heads = 16`,
    /// `linear_num_value_heads = 48`, dims 128, conv kernel 4).
    pub fn qwen38() -> Self {
        Self {
            num_k_groups: 16,
            v_per_group: 3,
            head_k_dim: 128,
            head_v_dim: 128,
            conv_kernel: 4,
        }
    }

    /// Small configuration for tests and smoke runs.
    pub fn tiny() -> Self {
        Self {
            num_k_groups: 2,
            v_per_group: 2,
            head_k_dim: 8,
            head_v_dim: 8,
            conv_kernel: 4,
        }
    }

    /// Value heads total (`num_k_groups * v_per_group`; 48 on Qwen3.8).
    pub fn num_v_heads(&self) -> usize {
        self.num_k_groups * self.v_per_group
    }

    /// Decode-time state floats per (group, v-head) pair: `S [dv, dk]`.
    pub fn state_floats_per_pair(&self) -> usize {
        self.head_v_dim * self.head_k_dim
    }
}

/// L2-normalize the last dim of a rank-3 tensor, row by row. Shared with the
/// trunk through [`crate::tensor_ext::l2_normalize_rows`]; the reshape keeps
/// exactly one normalization implementation in the crate.
pub fn l2norm_last_dim<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [b, n, d] = x.dims();
    crate::tensor_ext::l2_normalize_rows(x.reshape([b * n, d])).reshape([b, n, d])
}

/// One gated-delta step (paper Eq. 10) on a single (k, v) pair.
/// `state` is `[dv, dk]`; `q`, `k` are `[dk]` (normalize them first with
/// [`l2norm_last_dim`] — this function takes them as given so the
/// normalization itself stays independently testable); `v` is `[dv]`.
/// Returns `(new_state, output)`.
pub fn gated_delta_step<B: Backend>(
    state: Tensor<B, 2>,
    q: Tensor<B, 1>,
    k: Tensor<B, 1>,
    v: Tensor<B, 1>,
    alpha: f32,
    beta: f32,
) -> (Tensor<B, 2>, Tensor<B, 1>) {
    // S' = alpha * (S - beta * (S k) k^T) + beta * v k^T, with
    // k_row [dk, 1] and k_col = k^T [1, dk].
    let k_row = k.clone().unsqueeze_dim::<2>(1);
    let k_col = k.unsqueeze_dim::<2>(1).transpose();
    let sk = state.clone().matmul(k_row);
    let decayed = state
        .sub(sk.matmul(k_col.clone()).mul_scalar(beta))
        .mul_scalar(alpha);
    let vk = v.unsqueeze_dim::<2>(1).matmul(k_col);
    let next = decayed.add(vk.mul_scalar(beta));
    // o = S' q
    let out = next.clone().matmul(q.unsqueeze_dim::<2>(1)).squeeze::<1>();
    (next, out)
}

/// Recurrent rollout over `n` steps from a zero state. `q`, `k` are
/// `[b, n, dk]`, `v` is `[b, n, dv]`, `alpha`/`beta` are `[n]`. Returns
/// `(outputs [b, n, dv], final_states [b, dv, dk])`.
pub fn gated_delta_recurrent<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    alpha: &[f32],
    beta: &[f32],
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [b, n, _dk] = q.dims();
    let dv = v.dims()[2];
    assert_eq!(k.dims(), [b, n, q.dims()[2]], "q/k shape mismatch");
    assert_eq!(v.dims()[0], b, "batch mismatch");
    assert_eq!(v.dims()[1], n, "length mismatch");
    assert_eq!(alpha.len(), n, "alpha length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    let dk = q.dims()[2];
    // Batch loop: heads fold into the batch dim by the caller (one pair per
    // row), so the core never needs a head axis of its own.
    let mut outs: Vec<Tensor<B, 3>> = Vec::with_capacity(b);
    let mut finals: Vec<Tensor<B, 3>> = Vec::with_capacity(b);
    for row in 0..b {
        let mut state: Tensor<B, 2> = Tensor::zeros([dv, dk], device);
        let mut steps: Vec<Tensor<B, 2>> = Vec::with_capacity(n);
        for t in 0..n {
            let qt = q.clone().narrow(0, row, 1).narrow(1, t, 1).reshape([dk]);
            let kt = k.clone().narrow(0, row, 1).narrow(1, t, 1).reshape([dk]);
            let vt = v.clone().narrow(0, row, 1).narrow(1, t, 1).reshape([dv]);
            let (next, out) = gated_delta_step(state, qt, kt, vt, alpha[t], beta[t]);
            state = next;
            steps.push(out.unsqueeze_dim::<2>(0));
        }
        finals.push(state.unsqueeze_dim::<3>(0));
        outs.push(Tensor::cat(steps, 1).reshape([1, n, dv]));
    }
    (Tensor::cat(outs, 0), Tensor::cat(finals, 0))
}

/// Tensor-alpha sibling of [`gated_delta_step`]: identical math in the
/// identical operation order, but `alpha`/`beta` arrive as `[1]` tensors so
/// gradients flow through them (the scalar version cuts the graph at the
/// host `f32`, and later LoRA adapters sit on `in_proj_a`/`in_proj_b`).
/// Burn 0.21's typed `mul` does not broadcast across ranks, so the `[1]`
/// factors are reshaped to `[1, 1]` and broadcast within rank 2 — the same
/// IEEE multiply as `mul_scalar`, so the two variants are bit-identical
/// (certified in `verify.rs`).
pub fn gated_delta_step_tensor<B: Backend>(
    state: Tensor<B, 2>,
    q: Tensor<B, 1>,
    k: Tensor<B, 1>,
    v: Tensor<B, 1>,
    alpha: Tensor<B, 1>,
    beta: Tensor<B, 1>,
) -> (Tensor<B, 2>, Tensor<B, 1>) {
    // S' = alpha * (S - beta * (S k) k^T) + beta * v k^T, exactly the scalar
    // step's order: subtract the Householder term, then decay, then add vk.
    let alpha = alpha.reshape([1, 1]);
    let beta = beta.reshape([1, 1]);
    let k_row = k.clone().unsqueeze_dim::<2>(1);
    let k_col = k.unsqueeze_dim::<2>(1).transpose();
    let sk = state.clone().matmul(k_row);
    let decayed = state
        .sub(sk.matmul(k_col.clone()).mul(beta.clone()))
        .mul(alpha);
    let vk = v.unsqueeze_dim::<2>(1).matmul(k_col);
    let next = decayed.add(vk.mul(beta));
    // o = S' q
    let out = next.clone().matmul(q.unsqueeze_dim::<2>(1)).squeeze::<1>();
    (next, out)
}

/// Tensor-alpha sibling of [`gated_delta_recurrent`]: `q`, `k` are
/// `[rows, n, dk]`, `v` is `[rows, n, dv]`, and `alpha`/`beta` are
/// `[rows, n]` (one (k, v) pair per row; callers fold batch x heads into
/// rows). Returns `(outputs [rows, n, dv], final_states [rows, dv, dk])`.
pub fn gated_delta_recurrent_batched<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    alpha: Tensor<B, 2>,
    beta: Tensor<B, 2>,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [rows, n, dk] = q.dims();
    let dv = v.dims()[2];
    assert_eq!(k.dims(), [rows, n, dk], "q/k shape mismatch");
    assert_eq!(v.dims()[0], rows, "batch mismatch");
    assert_eq!(v.dims()[1], n, "length mismatch");
    assert_eq!(alpha.dims(), [rows, n], "alpha shape mismatch");
    assert_eq!(beta.dims(), [rows, n], "beta shape mismatch");
    let mut outs: Vec<Tensor<B, 3>> = Vec::with_capacity(rows);
    let mut finals: Vec<Tensor<B, 3>> = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut state: Tensor<B, 2> = Tensor::zeros([dv, dk], device);
        let mut steps: Vec<Tensor<B, 2>> = Vec::with_capacity(n);
        for t in 0..n {
            let qt = q.clone().narrow(0, row, 1).narrow(1, t, 1).reshape([dk]);
            let kt = k.clone().narrow(0, row, 1).narrow(1, t, 1).reshape([dk]);
            let vt = v.clone().narrow(0, row, 1).narrow(1, t, 1).reshape([dv]);
            let at = alpha.clone().narrow(0, row, 1).narrow(1, t, 1).reshape([1]);
            let bt = beta.clone().narrow(0, row, 1).narrow(1, t, 1).reshape([1]);
            let (next, out) = gated_delta_step_tensor(state, qt, kt, vt, at, bt);
            state = next;
            steps.push(out.unsqueeze_dim::<2>(0));
        }
        finals.push(state.unsqueeze_dim::<3>(0));
        outs.push(Tensor::cat(steps, 1).reshape([1, n, dv]));
    }
    (Tensor::cat(outs, 0), Tensor::cat(finals, 0))
}

/// Causal depthwise FIR filter (the short convolution on the q/k/v paths;
/// kernel 4 on Qwen): each channel filtered independently over its own past,
/// `out[t] = sum_{lag} x[t-lag] * w[lag]` with zero padding before position
/// 0. `x` is `[b, n, d]`, `weight` is `[d, k]`. Pure narrow/cat/matmul, so
/// the causality (`trim` of the future) is structural, not a mask that could
/// be misbuilt.
pub fn causal_depthwise_conv<B: Backend>(x: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    let [b, n, d] = x.dims();
    let k = weight.dims()[1];
    assert_eq!(weight.dims()[0], d, "conv weight channel mismatch");
    let device = x.device();
    // Lag matrix [b, n, d, k]: lag l holds x shifted right by l (zeros first).
    let mut lags: Vec<Tensor<B, 4>> = Vec::with_capacity(k);
    for lag in 0..k {
        let pad: Tensor<B, 3> = Tensor::zeros([b, lag.min(n), d], &device);
        let keep = n - lag.min(n);
        let shifted = Tensor::cat(vec![pad, x.clone().narrow(1, 0, keep)], 1);
        lags.push(shifted.unsqueeze_dim::<4>(3));
    }
    let lagmat = Tensor::cat(lags, 3); // [b, n, d, k]
                                       // weight [d, k] -> [1, 1, d, k], broadcast-multiply, sum over lags.
    let w = weight.reshape([1, 1, d, k]);
    lagmat.mul(w).sum_dim(3).reshape([b, n, d])
}

/// FLOPs per token of one (k, v) pair's recurrent step (multiply-adds x2):
/// `S k` (dv*dk), outer updates (2*dv*dk), `S' q` (dv*dk).
pub fn pair_step_flops(head_k_dim: usize, head_v_dim: usize) -> f64 {
    2.0 * (4 * head_v_dim * head_k_dim) as f64
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
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn test_qwen38_config_matches_published_numbers() {
        let cfg = GatedDeltaConfig::qwen38();
        assert_eq!(cfg.num_v_heads(), 48);
        assert_eq!(
            (cfg.head_k_dim, cfg.head_v_dim, cfg.conv_kernel),
            (128, 128, 4)
        );
        assert_eq!(cfg.state_floats_per_pair(), 128 * 128);
    }

    #[test]
    fn test_delta_step_matches_householder_form() {
        // alpha = 1 must be exactly S(I - b k k^T) + b v k^T, computed by
        // hand here rather than by re-calling the step.
        let device = Default::default();
        let state = Tensor::<B, 2>::from_floats([[1.0, 0.5], [0.25, 2.0]], &device);
        let q = Tensor::<B, 1>::from_floats([1.0, -1.0], &device);
        let k = Tensor::<B, 1>::from_floats([0.5, 0.5], &device);
        let v = Tensor::<B, 1>::from_floats([2.0, 0.0], &device);
        let beta = 0.7f32;
        let (next, out) =
            gated_delta_step(state.clone(), q.clone(), k.clone(), v.clone(), 1.0, beta);
        // Reference by direct expansion.
        let sk: Vec<f32> = state
            .clone()
            .matmul(k.clone().unsqueeze_dim::<2>(1))
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        // S k = [1*0.5+0.5*0.5, 0.25*0.5+2*0.5] = [0.75, 1.125]
        assert!(close(&sk, &[0.75, 1.125], 1e-6));
        let next_v: Vec<f32> = next.into_data().convert::<f32>().iter().collect();
        // Row 0: [1,0.5] - 0.7*0.75*[0.5,0.5] + 0.7*[1,0] = [1.4375, 0.9375]
        // Row 1: [0.25,2] - 0.7*1.125*[0.5,0.5] + 0 = [-0.14375, 1.60625]
        assert!(close(&next_v, &[1.4375, 0.9375, -0.14375, 1.60625], 1e-5));
        // o = S' q with q = [1,-1]: [1.4375-0.9375, -0.14375-1.60625] = [0.5, -1.75]
        let out_v: Vec<f32> = out.into_data().convert::<f32>().iter().collect();
        assert!(close(&out_v, &[0.5, -1.75], 1e-5));
    }

    #[test]
    fn test_zero_beta_is_pure_decay() {
        // beta = 0 must be exactly alpha * S regardless of k, v.
        let device = Default::default();
        let state = Tensor::<B, 2>::random([3, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let q = Tensor::<B, 1>::random([4], Distribution::Uniform(-1.0, 1.0), &device);
        let k = Tensor::<B, 1>::random([4], Distribution::Uniform(-1.0, 1.0), &device);
        let v = Tensor::<B, 1>::random([3], Distribution::Uniform(-1.0, 1.0), &device);
        let (next, _) = gated_delta_step(state.clone(), q, k, v, 0.3, 0.0);
        let a: Vec<f32> = next.into_data().convert::<f32>().iter().collect();
        let b: Vec<f32> = (state * 0.3).into_data().convert::<f32>().iter().collect();
        assert!(close(&a, &b, 1e-6));
    }

    #[test]
    fn test_zero_state_first_output_closed_form() {
        // S_0 = 0 gives o_1 = beta * v_1 * (k_1^T q_1): the one identity that
        // exercises the full step (state update AND readout) in closed form.
        let device = Default::default();
        let dk = 4;
        let dv = 3;
        let q = Tensor::<B, 3>::random([1, 2, dk], Distribution::Uniform(-1.0, 1.0), &device);
        let k = Tensor::<B, 3>::random([1, 2, dk], Distribution::Uniform(-1.0, 1.0), &device);
        let v = Tensor::<B, 3>::random([1, 2, dv], Distribution::Uniform(-1.0, 1.0), &device);
        let (out, _) = gated_delta_recurrent(
            q.clone(),
            k.clone(),
            v.clone(),
            &[0.9, 0.5],
            &[0.7, 0.4],
            &device,
        );
        let q1: Vec<f32> = q
            .narrow(1, 0, 1)
            .reshape([dk])
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        let k1: Vec<f32> = k
            .narrow(1, 0, 1)
            .reshape([dk])
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        let v1: Vec<f32> = v
            .narrow(1, 0, 1)
            .reshape([dv])
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        let dot: f32 = q1.iter().zip(&k1).map(|(a, b)| a * b).sum();
        let want: Vec<f32> = v1.iter().map(|x| 0.7 * x * dot).collect();
        let got: Vec<f32> = out
            .narrow(1, 0, 1)
            .reshape([dv])
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        assert!(close(&got, &want, 1e-5));
    }

    #[test]
    fn test_conv_is_causal_fir_with_ordered_taps() {
        // Impulse at t=2 with kernel [1,2,3]: outputs at 2,3,4 read
        // 1,2,3 and nothing else; t<2 stays zero (causality, structurally).
        let device = Default::default();
        let mut x = vec![0.0f32; 6];
        x[2] = 1.0;
        let input = Tensor::<B, 1>::from_floats(x.as_slice(), &device).reshape([1, 6, 1]);
        let w = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device).reshape([1, 3]);
        let out: Vec<f32> = causal_depthwise_conv(input, w)
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        assert!(close(&out, &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0], 1e-6));
    }

    #[test]
    fn test_tensor_step_matches_scalar_step() {
        // The tensor-alpha variant must reproduce the scalar step exactly:
        // same operation order, broadcast [1] multiplies for mul_scalar.
        let device = Default::default();
        let state = Tensor::<B, 2>::random([3, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let q = Tensor::<B, 1>::random([4], Distribution::Uniform(-1.0, 1.0), &device);
        let k = Tensor::<B, 1>::random([4], Distribution::Uniform(-1.0, 1.0), &device);
        let v = Tensor::<B, 1>::random([3], Distribution::Uniform(-1.0, 1.0), &device);
        let alpha = 0.7f32;
        let beta = 0.4f32;
        let (want_s, want_o) =
            gated_delta_step(state.clone(), q.clone(), k.clone(), v.clone(), alpha, beta);
        let (got_s, got_o) = gated_delta_step_tensor(
            state,
            q,
            k,
            v,
            Tensor::<B, 1>::from_floats([alpha], &device),
            Tensor::<B, 1>::from_floats([beta], &device),
        );
        let a: Vec<f32> = got_s.into_data().convert::<f32>().iter().collect();
        let b: Vec<f32> = want_s.into_data().convert::<f32>().iter().collect();
        assert!(close(&a, &b, 0.0));
        let a: Vec<f32> = got_o.into_data().convert::<f32>().iter().collect();
        let b: Vec<f32> = want_o.into_data().convert::<f32>().iter().collect();
        assert!(close(&a, &b, 0.0));
    }

    #[test]
    fn test_batched_recurrent_matches_scalar_recurrent() {
        // Two independent rows through the batched tensor-alpha rollout must
        // equal the scalar rollout run per row with the same alpha/beta.
        let device = Default::default();
        let (dk, dv, n) = (4, 3, 3);
        let q0 = Tensor::<B, 3>::random([1, n, dk], Distribution::Uniform(-1.0, 1.0), &device);
        let k0 = Tensor::<B, 3>::random([1, n, dk], Distribution::Uniform(-1.0, 1.0), &device);
        let v0 = Tensor::<B, 3>::random([1, n, dv], Distribution::Uniform(-1.0, 1.0), &device);
        let q1 = Tensor::<B, 3>::random([1, n, dk], Distribution::Uniform(-1.0, 1.0), &device);
        let k1 = Tensor::<B, 3>::random([1, n, dk], Distribution::Uniform(-1.0, 1.0), &device);
        let v1 = Tensor::<B, 3>::random([1, n, dv], Distribution::Uniform(-1.0, 1.0), &device);
        let alpha = Tensor::<B, 2>::from_floats([[0.8, 0.3, 0.5], [0.9, 0.2, 0.7]], &device);
        let beta = Tensor::<B, 2>::from_floats([[0.6, 0.9, 0.2], [0.4, 0.8, 0.1]], &device);
        let (got_o, got_f) = gated_delta_recurrent_batched(
            Tensor::cat(vec![q0.clone(), q1.clone()], 0),
            Tensor::cat(vec![k0.clone(), k1.clone()], 0),
            Tensor::cat(vec![v0.clone(), v1.clone()], 0),
            alpha,
            beta,
            &device,
        );
        let (o0, f0) =
            gated_delta_recurrent(q0, k0, v0, &[0.8, 0.3, 0.5], &[0.6, 0.9, 0.2], &device);
        let (o1, f1) =
            gated_delta_recurrent(q1, k1, v1, &[0.9, 0.2, 0.7], &[0.4, 0.8, 0.1], &device);
        let want_o = Tensor::cat(vec![o0, o1], 0);
        let want_f = Tensor::cat(vec![f0, f1], 0);
        let a: Vec<f32> = got_o.into_data().convert::<f32>().iter().collect();
        let b: Vec<f32> = want_o.into_data().convert::<f32>().iter().collect();
        assert!(close(&a, &b, 0.0));
        let a: Vec<f32> = got_f.into_data().convert::<f32>().iter().collect();
        let b: Vec<f32> = want_f.into_data().convert::<f32>().iter().collect();
        assert!(close(&a, &b, 0.0));
    }

    #[test]
    fn test_l2norm_rows_are_unit() {
        // Unit *norm* (sum of squares one), like the kernel's l2norm: query
        // rows then satisfy |q . k| <= 1 by Cauchy-Schwarz, which is the
        // bound the stability argument needs.
        let device = Default::default();
        let x = Tensor::<B, 3>::random([2, 5, 8], Distribution::Normal(0.0, 3.0), &device);
        let y: Vec<f32> = l2norm_last_dim(x)
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        for row in y.chunks_exact(8) {
            let norm: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
            assert!((norm.sqrt() - 1.0).abs() < 1e-5);
        }
    }
}
