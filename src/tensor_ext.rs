//! Small tensor and module utilities shared across modules.

use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{activation::sigmoid, backend::Backend, Bool, Int, Tensor};

/// Force every lazily-initialized parameter of a module to materialize.
///
/// Burn initializes `Linear` and `Embedding` weights **lazily**: `init()`
/// stores a closure and the tensor is only drawn on first access. `Param::clone`
/// on a parameter that has not been forced yet hands the clone *its own*
/// deferred initializer, so the two draw **different random values while
/// keeping the same `ParamId`** — Burn's own comment says "initializing one
/// does not affect the other".
///
/// That makes `module.clone()` on a freshly constructed module quietly wrong
/// whenever the clone is expected to share weights. Call this first.
pub fn force_initialization<B: Backend, M: Module<B>>(module: &M) {
    let mut visitor = Materializer;
    module.visit(&mut visitor);
}

struct Materializer;

impl<B: Backend> ModuleVisitor<B> for Materializer {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let _ = param.val();
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        let _ = param.val();
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        let _ = param.val();
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::{Linear, LinearConfig};

    type B = NdArray<f32>;

    #[test]
    fn test_cloning_an_unforced_module_draws_different_weights() {
        // Pins the Burn behaviour `force_initialization` exists for. If a
        // future Burn makes `Param::clone` share the deferred initializer this
        // test flips, and the helper can go -- but until then, cloning an
        // un-forced module silently produces a *different* model with the same
        // parameter ids, which is exactly the kind of bug that is invisible
        // until outputs disagree.
        let device = Default::default();
        let a: Linear<B> = LinearConfig::new(8, 4).init(&device);
        let b = a.clone();
        let drift = (a.weight.val() - b.weight.val()).abs().max().into_scalar();
        assert!(drift > 0.0, "expected lazy clones to diverge, got {drift}");

        // Forcing first makes the clone share the materialized value.
        let c: Linear<B> = LinearConfig::new(8, 4).init(&device);
        force_initialization(&c);
        let d = c.clone();
        let shared = (c.weight.val() - d.weight.val()).abs().max().into_scalar();
        assert_eq!(shared, 0.0, "a forced clone must share weights exactly");
    }
}
