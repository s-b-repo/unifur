//! QLoRA: 4-bit NormalFloat quantization plus low-rank adapters
//! (roadmap Phase 9).
//!
//! Two independent pieces that compose into QLoRA (Dettmers et al., 2023):
//!
//! 1. [`Nf4Block`] / [`Nf4Tensor`]: blockwise NF4 quantization. Weights are
//!    split into blocks of [`BLOCK_SIZE`]; each block is divided by its
//!    absolute maximum so its values land in `[-1, 1]`, then every value is
//!    mapped to the nearest of 16 levels drawn from the *quantiles of a
//!    standard normal*. The levels are information-theoretically optimal for
//!    normally distributed weights, which is what neural-network weight blocks
//!    approximately are -- and the absmax normalization is what makes that
//!    assumption hold per block rather than globally.
//!    [`Nf4Tensor::with_double_quantization`] additionally quantizes the
//!    per-block scales themselves to 8 bits, the "double quantization" of the
//!    paper, taking the amortized overhead from 32 to ~8.5 bits per block.
//!
//! 2. [`LoraAdapter`]: a rank-`r` update `dW = (alpha / r) * B A` with `B`
//!    zero-initialized, so an adapted layer starts *exactly* equal to the
//!    frozen base layer and training only ever moves it deliberately.
//!
//! [`QLoraLinear`] combines them: a frozen NF4 base weight plus a trainable
//! `f32` adapter, which is the whole point of QLoRA -- the memory of 4-bit
//! weights with the trainability of full precision.
//!
//! Quantization here is *storage* quantization: values are dequantized to
//! `f32` before the matmul, exactly as bitsandbytes does. The saving is in
//! resident weight memory, not in arithmetic throughput.

use burn::{
    module::{Module, ModuleMapper, Param},
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Distribution, Tensor},
};

/// Values per quantization block. 64 matches the QLoRA default.
pub const BLOCK_SIZE: usize = 64;

/// The 16 NF4 levels: quantiles of a standard normal, rescaled so the extreme
/// levels sit exactly at `-1` and `+1` and zero is exactly representable.
///
/// Exact representation of zero matters: padding, masked entries and pruned
/// weights must survive a round trip unchanged, which a symmetric 16-level
/// grid without a zero would not guarantee.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_51,
    0.160_930_6,
    0.246_112_3,
    0.337_665_43,
    0.440_709_83,
    0.562_617_6,
    0.722_956_84,
    1.0,
];

/// One quantized block: 4-bit codes plus the `f32` scale they share.
#[derive(Debug, Clone, PartialEq)]
pub struct Nf4Block {
    /// One code in `0..16` per value (stored one per byte; packing two codes
    /// per byte is a storage detail that does not change the numerics).
    pub codes: Vec<u8>,
    /// Absolute maximum of the block before quantization.
    pub absmax: f32,
}

/// A tensor stored as NF4 blocks.
#[derive(Debug, Clone)]
pub struct Nf4Tensor {
    blocks: Vec<Nf4Block>,
    /// Logical element count (the last block may be shorter than
    /// [`BLOCK_SIZE`]).
    len: usize,
    /// Set when the per-block scales are themselves quantized: the 8-bit
    /// codes and the shared scale of scales.
    double_quantized: Option<DoubleQuantScales>,
}

/// Second-level quantization of the per-block `absmax` values.
#[derive(Debug, Clone)]
struct DoubleQuantScales {
    codes: Vec<u8>,
    scale_of_scales: f32,
}

impl Nf4Tensor {
    /// Quantize `values` blockwise.
    pub fn quantize(values: &[f32]) -> Self {
        let blocks = values
            .chunks(BLOCK_SIZE)
            .map(|chunk| {
                let absmax = chunk.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                let codes = chunk
                    .iter()
                    .map(|&v| {
                        // An all-zero block has no scale; every value maps to
                        // the exact zero level.
                        if absmax == 0.0 {
                            nearest_level_code(0.0)
                        } else {
                            nearest_level_code(v / absmax)
                        }
                    })
                    .collect();
                Nf4Block { codes, absmax }
            })
            .collect();
        Self { blocks, len: values.len(), double_quantized: None }
    }

    /// Additionally quantize the per-block scales to 8 bits ("double
    /// quantization"): the scales share one `f32`, so per-block overhead falls
    /// from 32 bits to 8 + 32/`num_blocks`.
    pub fn with_double_quantization(mut self) -> Self {
        let max_scale = self.blocks.iter().fold(0.0f32, |acc, b| acc.max(b.absmax));
        let codes = self
            .blocks
            .iter()
            .map(|b| {
                if max_scale == 0.0 {
                    0u8
                } else {
                    // Scales are non-negative, so the full 0..=255 range is
                    // spent on [0, max_scale].
                    (b.absmax / max_scale * 255.0).round().clamp(0.0, 255.0) as u8
                }
            })
            .collect();
        self.double_quantized = Some(DoubleQuantScales { codes, scale_of_scales: max_scale });
        self
    }

    /// Whether the scales are themselves quantized.
    pub fn is_double_quantized(&self) -> bool {
        self.double_quantized.is_some()
    }

    /// Number of quantized values.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Effective scale of block `i`, honouring double quantization.
    fn block_scale(&self, i: usize) -> f32 {
        match &self.double_quantized {
            None => self.blocks[i].absmax,
            Some(dq) => dq.codes[i] as f32 / 255.0 * dq.scale_of_scales,
        }
    }

    /// Reconstruct the `f32` values.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len);
        for (i, block) in self.blocks.iter().enumerate() {
            let scale = self.block_scale(i);
            out.extend(block.codes.iter().map(|&c| NF4_LEVELS[c as usize] * scale));
        }
        out
    }

    /// Resident bits per stored value, including the scale overhead. This is
    /// the number the whole scheme exists to reduce, so it is computed rather
    /// than quoted.
    pub fn bits_per_value(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        let payload = 4.0 * self.len as f64;
        let scales = match &self.double_quantized {
            None => 32.0 * self.blocks.len() as f64,
            Some(_) => 8.0 * self.blocks.len() as f64 + 32.0,
        };
        (payload + scales) / self.len as f64
    }
}

/// Nearest NF4 level to `v` (which must already be normalized to `[-1, 1]`).
///
/// Linear scan over 16 entries: branch-free enough at this size, and it keeps
/// the level table the single source of truth rather than duplicating it as
/// hard-coded thresholds.
fn nearest_level_code(v: f32) -> u8 {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (i, &level) in NF4_LEVELS.iter().enumerate() {
        let d = (v - level).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best as u8
}

/// Quantize a rank-2 tensor and reconstruct it, i.e. apply exactly the error
/// a frozen NF4 weight would carry.
pub fn quantize_dequantize_tensor<B: Backend<FloatElem = f32>>(
    tensor: Tensor<B, 2>,
    double_quantization: bool,
) -> Tensor<B, 2> {
    let device = tensor.device();
    let dims = tensor.dims();
    let values: Vec<f32> = tensor.into_data().convert::<f32>().iter::<f32>().collect();
    let mut q = Nf4Tensor::quantize(&values);
    if double_quantization {
        q = q.with_double_quantization();
    }
    Tensor::<B, 1>::from_floats(q.dequantize().as_slice(), &device).reshape(dims)
}

/// Quantizes every weight-shaped parameter of a module to the NF4 grid
/// in place (roadmap 9.1: "quantize the base model").
///
/// Rank-1 parameters -- biases, LayerNorm scales -- are left alone: they are a
/// vanishing fraction of the parameters and by far the most sensitive to
/// quantization error, which is exactly the trade bitsandbytes makes. Modules
/// named in `skip` are excluded wholesale; the label-embedding table is the
/// usual candidate, since it *is* the diffusion process's data space.
pub struct Nf4Quantizer {
    double_quantization: bool,
    skip: Vec<String>,
    path: Vec<String>,
    /// Values quantized so far, for reporting.
    pub quantized_values: usize,
    /// Parameters left untouched.
    pub skipped_params: usize,
}

impl Nf4Quantizer {
    pub fn new(double_quantization: bool) -> Self {
        Self {
            double_quantization,
            skip: Vec::new(),
            path: Vec::new(),
            quantized_values: 0,
            skipped_params: 0,
        }
    }

    /// Exclude any parameter whose path contains `name`.
    pub fn skipping(mut self, name: &str) -> Self {
        self.skip.push(name.to_string());
        self
    }

    fn is_skipped(&self) -> bool {
        self.skip
            .iter()
            .any(|needle| self.path.iter().any(|part| part.contains(needle.as_str())))
    }
}

impl<B: Backend<FloatElem = f32>> ModuleMapper<B> for Nf4Quantizer {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let tensor = param.val();
        let numel: usize = tensor.dims().iter().product();
        if D < 2 || numel < BLOCK_SIZE || self.is_skipped() {
            self.skipped_params += 1;
            return param;
        }

        let device = tensor.device();
        let dims = tensor.dims();
        let values: Vec<f32> = tensor
            .set_require_grad(false)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let mut q = Nf4Tensor::quantize(&values);
        if self.double_quantization {
            q = q.with_double_quantization();
        }
        self.quantized_values += numel;
        Param::from_tensor(
            Tensor::<B, 1>::from_floats(q.dequantize().as_slice(), &device).reshape(dims),
        )
    }
}

/// Quantize a whole model's weights to NF4, returning the model and the number
/// of values quantized.
pub fn quantize_module<B, M>(module: M, double_quantization: bool, skip: &[&str]) -> (M, usize)
where
    B: Backend<FloatElem = f32>,
    M: Module<B>,
{
    let mut mapper = skip
        .iter()
        .fold(Nf4Quantizer::new(double_quantization), |m, name| m.skipping(name));
    let module = module.map(&mut mapper);
    let count = mapper.quantized_values;
    (module, count)
}

/// Configuration of a [`LoraAdapter`].
#[derive(Debug, Clone, Copy)]
pub struct LoraConfig {
    pub in_features: usize,
    pub out_features: usize,
    /// Adapter rank `r`.
    pub rank: usize,
    /// Scaling numerator; the update is multiplied by `alpha / rank`.
    pub alpha: f64,
}

impl LoraConfig {
    pub fn new(in_features: usize, out_features: usize, rank: usize) -> Self {
        Self { in_features, out_features, rank: rank.max(1), alpha: rank.max(1) as f64 }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Effective scaling `alpha / rank`.
    pub fn scaling(&self) -> f64 {
        self.alpha / self.rank.max(1) as f64
    }

    /// Trainable parameters, versus `in * out` for a full update.
    pub fn num_parameters(&self) -> usize {
        self.rank * (self.in_features + self.out_features)
    }
}

/// Low-rank update `dW = (alpha / r) * B A`, with `A ~ N(0, 1/r)` and `B = 0`.
///
/// The zero-initialized `B` is what makes an adapted model start out
/// numerically identical to its base -- attaching an adapter can never perturb
/// a trained network before a single gradient step.
#[derive(Module, Debug)]
pub struct LoraAdapter<B: Backend> {
    /// `[in_features, rank]`
    a: Param<Tensor<B, 2>>,
    /// `[rank, out_features]`
    b: Param<Tensor<B, 2>>,
    scaling: f64,
}

impl<B: Backend> LoraAdapter<B> {
    pub fn new(config: &LoraConfig, device: &B::Device) -> Self {
        let std = 1.0 / (config.rank.max(1) as f64).sqrt();
        Self {
            a: Param::from_tensor(Tensor::random(
                [config.in_features, config.rank],
                Distribution::Normal(0.0, std),
                device,
            )),
            b: Param::from_tensor(Tensor::zeros([config.rank, config.out_features], device)),
            scaling: config.scaling(),
        }
    }

    /// Rebuild from raw factors. Mirrors [`crate::moe::TopKRouter::from_parts`]
    /// and lets a trained or merged adapter be reconstructed without going
    /// through random initialization.
    ///
    /// Inputs are detached: `Param::from_tensor` needs a leaf, and factors
    /// derived from existing parameters are not one on an autodiff backend.
    pub fn from_parts(a: Tensor<B, 2>, b: Tensor<B, 2>, scaling: f64) -> Self {
        Self {
            a: Param::from_tensor(a.detach()),
            b: Param::from_tensor(b.detach()),
            scaling,
        }
    }

    /// `[in_features, rank]`.
    pub fn a(&self) -> Tensor<B, 2> {
        self.a.val()
    }

    /// `[rank, out_features]`. Zero at initialization, which is what makes a
    /// fresh adapter an exact no-op.
    pub fn b(&self) -> Tensor<B, 2> {
        self.b.val()
    }

    pub fn in_features(&self) -> usize {
        self.a.dims()[0]
    }

    pub fn out_features(&self) -> usize {
        self.b.dims()[1]
    }

    pub fn rank(&self) -> usize {
        self.a.dims()[1]
    }

    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// `x @ A @ B * scaling` for `x: [b, in_features]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.a.val())
            .matmul(self.b.val())
            .mul_scalar(self.scaling as f32)
    }

    /// The dense update `dW = scaling * A B`, `[in_features, out_features]`.
    /// Materializing it defeats the memory saving, so this is for merging and
    /// for verification, not for the forward path.
    pub fn delta_weight(&self) -> Tensor<B, 2> {
        self.a.val().matmul(self.b.val()).mul_scalar(self.scaling as f32)
    }
}

/// A linear layer with a frozen NF4-quantized base weight and a trainable
/// low-rank adapter.
///
/// The base weight is stored dequantized (`f32` values on the NF4 grid) so the
/// existing `f32`-only backend can use it directly; [`Self::resident_bits`]
/// reports what the same weight *would* occupy in packed NF4 form, which is
/// the figure QLoRA is chosen for.
#[derive(Module, Debug)]
pub struct QLoraLinear<B: Backend> {
    base: Linear<B>,
    adapter: LoraAdapter<B>,
    quantized_values: usize,
    double_quantized: bool,
}

impl<B: Backend<FloatElem = f32>> QLoraLinear<B> {
    /// Quantize `base`'s weight to NF4 and attach a fresh adapter.
    ///
    /// The bias (if any) is left in `f32`: biases are a vanishing fraction of
    /// the parameters and are the most sensitive to quantization error.
    pub fn from_linear(
        base: Linear<B>,
        config: &LoraConfig,
        double_quantization: bool,
        device: &B::Device,
    ) -> Self {
        let weight = base.weight.val();
        let quantized_values = weight.dims().iter().product();
        let q = quantize_dequantize_tensor(weight, double_quantization);

        let mut base = base;
        base.weight = Param::from_tensor(q);

        Self {
            base,
            adapter: LoraAdapter::new(config, device),
            quantized_values,
            double_quantized: double_quantization,
        }
    }

    /// Build a fresh quantized+adapted layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: &LoraConfig,
        double_quantization: bool,
        device: &B::Device,
    ) -> Self {
        let base = LinearConfig::new(in_features, out_features)
            .with_bias(true)
            .init(device);
        Self::from_linear(base, config, double_quantization, device)
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.base.forward(x.clone()) + self.adapter.forward(x)
    }

    pub fn adapter(&self) -> &LoraAdapter<B> {
        &self.adapter
    }

    /// Resident bits of the base weight in packed NF4 form, versus `f32`.
    pub fn resident_bits(&self) -> (f64, f64) {
        let blocks = self.quantized_values.div_ceil(BLOCK_SIZE) as f64;
        let scale_bits = if self.double_quantized {
            8.0 * blocks + 32.0
        } else {
            32.0 * blocks
        };
        (
            4.0 * self.quantized_values as f64 + scale_bits,
            32.0 * self.quantized_values as f64,
        )
    }

    /// Fold the adapter into the (already quantized) base weight, returning a
    /// plain linear layer. Used at the end of fine-tuning so inference costs
    /// nothing extra.
    pub fn merged(&self) -> Linear<B> {
        let mut base = self.base.clone();
        // `Param::from_tensor` requires a leaf; the sum of two tracked tensors
        // is not one, so the merged weight is detached first. That is also the
        // right semantics: merging ends fine-tuning, and the result is a plain
        // frozen weight rather than a node in the student's graph.
        let merged = (self.base.weight.val() + self.adapter.delta_weight()).detach();
        base.weight = Param::from_tensor(merged);
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_levels_are_sorted_symmetric_and_contain_zero() {
        assert!(NF4_LEVELS.windows(2).all(|w| w[0] < w[1]), "levels must be sorted");
        assert_eq!(NF4_LEVELS[0], -1.0);
        assert_eq!(NF4_LEVELS[NF4_LEVELS.len() - 1], 1.0);
        assert!(NF4_LEVELS.contains(&0.0), "zero must be exactly representable");
        assert_eq!(NF4_LEVELS.len(), 16, "NF4 is a 4-bit code");
    }

    #[test]
    fn test_exact_values_survive_a_roundtrip() {
        // Anything already on the grid (scaled by the block absmax) must come
        // back bit-identical -- in particular zeros, which padding and masks
        // depend on.
        let scale = 0.75f32;
        let values: Vec<f32> = NF4_LEVELS.iter().map(|&l| l * scale).collect();
        let restored = Nf4Tensor::quantize(&values).dequantize();
        for (r, v) in restored.iter().zip(&values) {
            assert!((r - v).abs() < 1e-6, "{r} != {v}");
        }
    }

    #[test]
    fn test_absmax_endpoints_are_exact() {
        // The largest-magnitude value of every block maps to level +/-1 and so
        // is reconstructed exactly; that is what the per-block absmax buys.
        let mut values = vec![0.01f32; BLOCK_SIZE];
        values[7] = -4.25;
        values[9] = 2.0;
        let restored = Nf4Tensor::quantize(&values).dequantize();
        assert!((restored[7] + 4.25).abs() < 1e-6, "block max must be exact: {}", restored[7]);
    }

    #[test]
    fn test_quantization_error_is_bounded_by_half_the_level_gap() {
        // The bound that makes NF4 usable at all: round-to-nearest on a fixed
        // grid cannot err by more than half the widest gap, times the block
        // scale. Checked on normally distributed data, which is the regime the
        // levels are designed for.
        let max_gap = NF4_LEVELS
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0f32, f32::max);

        let device = Default::default();
        let n = BLOCK_SIZE * 32;
        let values: Vec<f32> = Tensor::<B, 1>::random([n], Distribution::Normal(0.0, 1.0), &device)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();

        let q = Nf4Tensor::quantize(&values);
        let restored = q.dequantize();
        assert_eq!(restored.len(), values.len());

        for (block_idx, chunk) in values.chunks(BLOCK_SIZE).enumerate() {
            let absmax = chunk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let bound = 0.5 * max_gap * absmax + 1e-6;
            for (i, &v) in chunk.iter().enumerate() {
                let r = restored[block_idx * BLOCK_SIZE + i];
                assert!(
                    (r - v).abs() <= bound,
                    "error {} exceeds bound {bound} at {v}",
                    (r - v).abs()
                );
            }
        }

        // ...and NF4 must actually beat a uniform 16-level grid on normal
        // data, otherwise the quantile construction earns nothing.
        let nf4_err: f32 = values
            .iter()
            .zip(&restored)
            .map(|(v, r)| (v - r).powi(2))
            .sum();
        let uniform_err: f32 = values
            .chunks(BLOCK_SIZE)
            .flat_map(|chunk| {
                let absmax = chunk.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-12);
                chunk.iter().map(move |&v| {
                    // 16 uniformly spaced levels on [-absmax, absmax].
                    let step = 2.0 * absmax / 15.0;
                    let r = ((v + absmax) / step).round() * step - absmax;
                    (v - r).powi(2)
                })
            })
            .sum();
        assert!(
            nf4_err < uniform_err,
            "NF4 ({nf4_err}) should beat a uniform grid ({uniform_err}) on normal data"
        );
    }

    #[test]
    fn test_all_zero_block_is_exact() {
        let values = vec![0.0f32; BLOCK_SIZE];
        assert!(Nf4Tensor::quantize(&values).dequantize().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ragged_last_block() {
        let values: Vec<f32> = (0..BLOCK_SIZE + 5).map(|i| i as f32 * 0.01).collect();
        let q = Nf4Tensor::quantize(&values);
        assert_eq!(q.num_blocks(), 2);
        assert_eq!(q.len(), BLOCK_SIZE + 5);
        assert_eq!(q.dequantize().len(), values.len());
    }

    #[test]
    fn test_double_quantization_shrinks_overhead_and_keeps_accuracy() {
        let device = Default::default();
        let n = BLOCK_SIZE * 64;
        let values: Vec<f32> = Tensor::<B, 1>::random([n], Distribution::Normal(0.0, 1.0), &device)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();

        let plain = Nf4Tensor::quantize(&values);
        let double = Nf4Tensor::quantize(&values).with_double_quantization();

        assert!(!plain.is_double_quantized() && double.is_double_quantized());
        // 4 + 32/64 = 4.5 bits/value plain, 4 + 8/64 + eps ~ 4.13 doubled.
        assert!((plain.bits_per_value() - 4.5).abs() < 1e-9);
        assert!(double.bits_per_value() < plain.bits_per_value());
        assert!(double.bits_per_value() < 4.2);

        // The second level of quantization must stay a small perturbation:
        // 8 bits on a non-negative scale is <= 0.4% relative error, so the
        // reconstruction error should not grow by more than a few percent.
        let err = |t: &Nf4Tensor| -> f32 {
            t.dequantize().iter().zip(&values).map(|(r, v)| (r - v).powi(2)).sum()
        };
        let (e_plain, e_double) = (err(&plain), err(&double));
        assert!(
            e_double < e_plain * 1.1,
            "double quantization degraded accuracy too much: {e_plain} -> {e_double}"
        );
    }

    #[test]
    fn test_lora_starts_as_an_exact_identity() {
        // B = 0 means an attached adapter is a no-op until trained. This is
        // the property that makes it safe to wrap a trained checkpoint.
        let device = Default::default();
        let config = LoraConfig::new(16, 8, 4);
        let adapter = LoraAdapter::<B>::new(&config, &device);

        let x = Tensor::<B, 2>::random([3, 16], Distribution::Uniform(-1.0, 1.0), &device);
        let out = adapter.forward(x).abs().max().into_scalar();
        assert_eq!(out, 0.0, "zero-initialized B must give an exact no-op");
        assert_eq!(adapter.delta_weight().abs().max().into_scalar(), 0.0);
        assert_eq!(adapter.rank(), 4);
        assert!((adapter.scaling() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_lora_forward_matches_the_dense_update() {
        // x @ (scaling * A B) computed factored must equal the same thing
        // computed through the materialized delta.
        let device = Default::default();
        let config = LoraConfig::new(12, 6, 3).with_alpha(6.0);
        let mut adapter = LoraAdapter::<B>::new(&config, &device);
        // Give B a non-zero value so the check is not trivially 0 == 0.
        adapter.b = Param::from_tensor(Tensor::random(
            [3, 6],
            Distribution::Uniform(-0.5, 0.5),
            &device,
        ));

        let x = Tensor::<B, 2>::random([4, 12], Distribution::Uniform(-1.0, 1.0), &device);
        let factored = adapter.forward(x.clone());
        let dense = x.matmul(adapter.delta_weight());
        let diff = (factored - dense).abs().max().into_scalar();
        assert!(diff < 1e-5, "factored and dense LoRA disagree: {diff}");
        assert!((adapter.scaling() - 2.0).abs() < 1e-12, "alpha/rank = 6/3");
    }

    #[test]
    fn test_lora_parameter_count_beats_a_full_update() {
        let config = LoraConfig::new(768, 768, 8);
        assert_eq!(config.num_parameters(), 8 * (768 + 768));
        assert!(config.num_parameters() * 40 < 768 * 768, "rank 8 should be ~50x smaller");
    }

    #[test]
    fn test_qlora_linear_is_the_base_layer_at_init() {
        let device = Default::default();
        let base = LinearConfig::new(BLOCK_SIZE, 8).with_bias(true).init(&device);
        let config = LoraConfig::new(BLOCK_SIZE, 8, 2);
        let q = QLoraLinear::<B>::from_linear(base, &config, true, &device);

        // With B = 0 the layer is exactly its quantized base.
        let x = Tensor::<B, 2>::random([2, BLOCK_SIZE], Distribution::Uniform(-1.0, 1.0), &device);
        let diff = (q.forward(x.clone()) - q.base.forward(x)).abs().max().into_scalar();
        assert_eq!(diff, 0.0);

        let (nf4_bits, f32_bits) = q.resident_bits();
        assert!(nf4_bits * 7.0 < f32_bits, "NF4 should be ~7x smaller: {nf4_bits} vs {f32_bits}");
    }

    #[test]
    fn test_qlora_merge_preserves_outputs() {
        let device = Default::default();
        let config = LoraConfig::new(BLOCK_SIZE, 8, 2);
        let mut q = QLoraLinear::<B>::new(BLOCK_SIZE, 8, &config, false, &device);
        q.adapter.b = Param::from_tensor(Tensor::random(
            [2, 8],
            Distribution::Uniform(-0.3, 0.3),
            &device,
        ));

        let x = Tensor::<B, 2>::random([5, BLOCK_SIZE], Distribution::Uniform(-1.0, 1.0), &device);
        let before = q.forward(x.clone());
        let after = q.merged().forward(x);
        let diff = (before - after).abs().max().into_scalar();
        assert!(diff < 1e-5, "merging must not change the function: {diff}");
    }

    #[test]
    fn test_quantize_module_touches_only_weight_shaped_params() {
        use crate::dblock::{DblockClassifier, DblockConfig};
        use crate::vit::ViTDiTConfig;

        let device = Default::default();
        let cfg = ViTDiTConfig { num_hidden_layers: 2, ..ViTDiTConfig::tiny(10) };
        let model = DblockClassifier::<B>::new(
            &cfg,
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );

        // The label table is the diffusion process's data space; quantizing it
        // would move the targets, so it must survive untouched.
        let before = model.model().label_embedding_weight();
        let (quantized, count) = quantize_module(model, true, &["label_embeddings"]);
        assert!(count > 0, "some weights must have been quantized");

        let after = quantized.model().label_embedding_weight();
        let diff = (before - after).abs().max().into_scalar();
        assert_eq!(diff, 0.0, "skipped module must be bit-identical");
    }

    #[test]
    fn test_quantized_weights_stay_on_the_grid() {
        // Every dequantized value must be an exact level times its block
        // scale; if it is not, something reconstructed off-grid.
        let device = Default::default();
        let t = Tensor::<B, 2>::random([4, BLOCK_SIZE], Distribution::Normal(0.0, 1.0), &device);
        let values: Vec<f32> = t.clone().into_data().convert::<f32>().iter::<f32>().collect();
        let restored: Vec<f32> = quantize_dequantize_tensor(t, false)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();

        for (block_idx, chunk) in values.chunks(BLOCK_SIZE).enumerate() {
            let absmax = chunk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            for i in 0..chunk.len() {
                let r = restored[block_idx * BLOCK_SIZE + i];
                let on_grid = NF4_LEVELS
                    .iter()
                    .any(|&l| (l * absmax - r).abs() <= 1e-6 * absmax.max(1.0));
                assert!(on_grid, "value {r} is not on the NF4 grid (absmax {absmax})");
            }
        }
    }
}
