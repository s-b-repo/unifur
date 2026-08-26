//! Small tensor utilities shared across modules.

use burn::tensor::{Tensor, activation::sigmoid, backend::Backend};

/// SiLU activation `x * sigmoid(x)`.
pub fn silu<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    x.clone() * sigmoid(x)
}

/// Exact (erf-based) GELU, matching HF ViT's `hidden_act="gelu"`.
pub fn exact_gelu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let inner = (x.clone().mul_scalar((1.0f64 / 2.0).sqrt())).erf();
    x * (inner + 1.0) * 0.5
}

/// Row-wise L2 normalization for a rank-2 tensor.
pub fn l2_normalize_rows<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-12);
    x / norm.unsqueeze::<2>()
}
