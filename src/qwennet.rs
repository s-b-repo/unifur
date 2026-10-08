//! Qwen3.8-27B Burn modules and their checkpoint loader (Phase C of
//! `docs/Frontier-27B-Plan.md`).
//!
//! Task 1 of the 27B wiring project built the gated-DeltaNet
//! linear-attention head ([`GatedDeltaHead`]) exactly as HuggingFace
//! `modeling_qwen3_5.py` computes it, plus [`QwenRmsNorm`] (zero-centered)
//! and [`RmsNormGated`] (the per-V-head output norm). Task 2 composes the
//! full trunk: [`QwenMlp`], [`QwenFullAttention`] (bias-free, QK-normed,
//! partial RoPE at base 1e7, adjacent-repeat GQA, sigmoid output gate),
//! [`QwenDecoderLayer`] (pre-norm residuals), [`QwenTrunk`] (embedding, 64
//! layers, final norm, untied `lm_head`), and [`load_qwen_trunk`], which
//! audits every shard before filling anything.
//!
//! The math of the gated-DeltaNet head, per linear layer, input
//! `x [b, n, h]` (h = 5120, 16 key groups, 48 V heads at 3 per group, head
//! dims 128, conv kernel 4, conv channels C = 2*2048 + 6144 = 10240):
//!
//! 1. Four bias-free projections: `qkv = in_proj_qkv(x) [b,n,C]`,
//!    `z = in_proj_z(x) [b,n,6144]`, `b_ = in_proj_b(x) [b,n,48]`,
//!    `a = in_proj_a(x) [b,n,48]`.
//! 2. Causal depthwise conv on qkv (weight `[C, K]`, no bias), then SiLU.
//! 3. Split into q `[b,n,2048]`, k `[b,n,2048]`, v `[b,n,6144]` (in that
//!    order); reshape q,k to `[b,n,16,128]`, v to `[b,n,48,128]`.
//! 4. `beta = sigmoid(b_)`; `g = -exp(A_log) * softplus(a + dt_bias)`;
//!    `alpha = exp(g)`.
//! 5. L2-normalize q and k over the head dim, THEN scale q by `dk^-0.5`
//!    (HF does the scaling inside the kernel, after the norm).
//! 6. V head `hh` pairs with Q/K group `hh / 3` (adjacent repeat, NOT the
//!    tiled `h % groups` mapping of GQA): one gated delta recurrence per
//!    (group, v-head) pair from a zero state, reusing
//!    [`crate::deltanet::gated_delta_recurrent_batched`].
//! 7. Per-V-head gated RMSNorm over the 128 dims, norm BEFORE gate:
//!    `y = x * rsqrt(mean(x^2, -1) + eps); y = w * y; y = y * silu(z)`,
//!    eps = 1e-6.
//! 8. `out = out_proj(y.reshape([b,n,6144]))`, bias-free.
//!
//! Decode carries the last `K-1` raw pre-conv qkv rows and the recurrent
//! state `S [dv, dk]` per pair; token-by-token decode reproduces the
//! prefill forward bit-for-bit (pinned by
//! `decode_step_matches_prefill_prefix`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, softmax, softplus};
use burn::tensor::{backend::Backend, Distribution, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::deltanet::{
    causal_depthwise_conv, gated_delta_recurrent_batched, gated_delta_step_tensor, l2norm_last_dim,
    GatedDeltaConfig,
};
use crate::hybrid::{apply_rotary_partial, attention_mask, rotary_dim_for, LayerState};
use crate::import::{open_index, read_tensor_f32, DecodedTensor, SafetensorsIndex};
use crate::quantize::{LoraAdapter, LoraConfig, PackedNf4Tensor};
use crate::qwen::{audit_tensors, QwenArchDims, QwenPart};
use crate::vit::causal_mask;

/// Rank-3 SiLU folded through the crate's single SiLU implementation
/// (`tensor_ext::silu` is rank-2; heads and sequence collapse into rows).
fn silu3<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [a, b, c] = x.dims();
    crate::tensor_ext::silu(x.reshape([a * b, c])).reshape([a, b, c])
}

/// Adjacent repeat along `dim`: each slice appears `times` times in a row
/// (`[a, b] x2 -> [a, a, b, b]`), the grouping Qwen3.8's linear attention
/// uses (V head `hh` reads Q/K group `hh / v_per_group`). Burn's
/// `repeat_dim` TILES (`[a, b] x2 -> [a, b, a, b]`), which is the GQA
/// mapping and wrong here; unsqueeze + repeat_dim + reshape gives the
/// adjacent semantics in pure tensor ops.
pub fn repeat_adjacent<B: Backend>(x: Tensor<B, 4>, dim: usize, times: usize) -> Tensor<B, 4> {
    let mut dims = x.dims();
    let expanded = x.unsqueeze_dim::<5>(dim + 1).repeat_dim(dim + 1, times);
    dims[dim] *= times;
    expanded.reshape(dims)
}

/// Qwen3.5 RMSNorm: ZERO-CENTERED weight, computing
/// `y = x * rsqrt(mean(x^2) + eps) * (1 + w)` with `w` zeros-init. Not used
/// by [`GatedDeltaHead`] (whose output norm is [`RmsNormGated`]); the trunk
/// layers need it, so it is defined here from the start.
#[derive(Module, Debug)]
pub struct QwenRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

impl<B: Backend> QwenRmsNorm<B> {
    pub fn new(hidden: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::<B, 1>::zeros([hidden], device)),
            epsilon,
        }
    }

    /// Rank-3 forward, same convention as `vit::RmsNorm`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = x.dims()[2];
        let ms = x.clone().powf_scalar(2.0).mean_dim(2);
        let inv = (ms + self.epsilon).powf_scalar(-0.5);
        x * inv * (self.weight.val().reshape([1, 1, h]) + 1.0)
    }

    /// Rank-4 forward (per-head QK norm on `[b, n, heads, dim]`): folds
    /// through the rank-3 path so there is exactly one normalization
    /// implementation for a divergence to hide in.
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, n, heads, d] = x.dims();
        self.forward(x.reshape([b * n * heads, 1, d]))
            .reshape([b, n, heads, d])
    }
}

/// Gated RMSNorm over the value-head dim (HF `Qwen3_5RMSNormGated`),
/// norm BEFORE gate: `y = x * rsqrt(mean(x^2, -1) + eps); y = w * y;
/// y = y * silu(z)`. The weight is ones-init (checkpoint tensor
/// `linear_attn.norm.weight`, `[head_v_dim]`).
#[derive(Module, Debug)]
pub struct RmsNormGated<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

impl<B: Backend> RmsNormGated<B> {
    pub fn new(head_v_dim: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::<B, 1>::ones([head_v_dim], device)),
            epsilon,
        }
    }

    /// `x` and `z` are `[rows, heads, dv]`: callers fold batch x position
    /// into rows so the norm runs per (row, head) over `dv`.
    pub fn forward(&self, x: Tensor<B, 3>, z: Tensor<B, 3>) -> Tensor<B, 3> {
        let d = x.dims()[2];
        let ms = x.clone().powf_scalar(2.0).mean_dim(2);
        let inv = (ms + self.epsilon).powf_scalar(-0.5);
        let normed = x * inv * self.weight.val().reshape([1, 1, d]);
        normed.mul(silu3(z))
    }
}

/// NF4-RESIDENCY POLICY (fixed design decision): every rank-2 tensor of the
/// trunk -- all attention/linear-attention projections, the MLP, the
/// embedding table, and the untied `lm_head` -- is stored PACKED
/// (`PackedNf4Tensor`, double-quantized, ~0.52 bytes/value) in the
/// `QwenLinear::Nf4` / `QwenEmbed::Nf4` variants. Rank-1 tensors (norms,
/// `dt_bias`, `A_log`), the depthwise conv weight ([10240, 4], tiny and
/// sensitive), and biases (this model has none) ALWAYS stay f32. The f32
/// loader keeps everything in the `F32` variants; both paths share every
/// downstream op, so tests exercise one code path with two storages.
///
/// A frozen NF4-RESIDENT linear: the weight lives packed
/// (`PackedNf4Tensor`, ~0.52 bytes/value with double quantization) and is
/// dequantized to a DETACHED f32 tensor at each forward, so gradients flow
/// to the input (and, in 3b, to adapter branches) but never into the
/// frozen base. The matmul runs through `burn::tensor::module::linear`,
/// the same op `Linear` uses, so an `Nf4Linear` is exactly "dequantize,
/// then matmul" (certified in `verify.rs`).
#[derive(Module, Debug)]
pub struct Nf4Linear<B: Backend> {
    /// Optional f32 bias (Qwen has none; the type stays honest).
    pub bias: Option<Param<Tensor<B, 1>>>,
    #[module(skip)]
    packed: PackedNf4Tensor,
    #[module(skip)]
    d_input: usize,
    #[module(skip)]
    d_output: usize,
}

impl<B: Backend> Nf4Linear<B> {
    /// Wrap an already-packed weight of logical shape `[d_input, d_output]`.
    pub fn from_packed(
        packed: PackedNf4Tensor,
        d_input: usize,
        d_output: usize,
        bias: Option<Param<Tensor<B, 1>>>,
    ) -> Self {
        Self {
            bias,
            packed,
            d_input,
            d_output,
        }
    }

    /// Quantize `[d_input, d_output]` row-major values into packed NF4 with
    /// double quantization (deterministic: pure function of the values).
    pub fn from_values(values: &[f32], d_input: usize, d_output: usize) -> Self {
        Self::from_packed(
            PackedNf4Tensor::quantize(values).with_double_quantization(),
            d_input,
            d_output,
            None,
        )
    }

    pub fn d_input(&self) -> usize {
        self.d_input
    }

    pub fn d_output(&self) -> usize {
        self.d_output
    }

    /// Packed bytes actually held (the residency-budget number).
    pub fn resident_bytes(&self) -> usize {
        self.packed.resident_bytes()
    }

    /// The packed weight itself (byte-identity checks; the frozen-base
    /// guarantee asserts these bytes never change across a training step).
    pub fn packed(&self) -> &PackedNf4Tensor {
        &self.packed
    }

    /// The dequantized `[d_input, d_output]` weight, DETACHED: the frozen
    /// base must never receive gradients. 3b's adapters sit beside this,
    /// not behind it. The dequantize Vec moves into `TensorData` (no slice
    /// copy on the 5 GB `lm_head`).
    pub fn dequantized_weight(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::from_data(
            TensorData::new(self.packed.dequantize(), [self.d_input, self.d_output]),
            device,
        )
        .set_require_grad(false)
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let weight = self.dequantized_weight(&x.device());
        burn::tensor::module::linear(x, weight, self.bias.as_ref().map(|b| b.val()))
    }
}

/// A frozen NF4-RESIDENT embedding table: packed storage, dequantized to a
/// detached f32 table per forward, then the same row-gather `Embedding`
/// uses. Dequantize-per-forward is deliberate: for the 27B the table is
/// 5.1 GB f32, but residency (not arithmetic) is what NF4 buys, and the
/// gate budget allows the transient.
///
/// The struct is deliberately NOT generic over `B` (it holds no tensors):
/// Burn derives a const-module `impl<B: Backend> Module<B>` for it, the
/// same way `nn::Dropout` is a module. The methods are generic instead.
#[derive(Module, Debug, Clone)]
pub struct Nf4Embedding {
    #[module(skip)]
    packed: PackedNf4Tensor,
    #[module(skip)]
    n_embedding: usize,
    #[module(skip)]
    d_model: usize,
}

impl Nf4Embedding {
    /// Quantize a `[n_embedding, d_model]` row-major table into packed NF4
    /// with double quantization.
    pub fn from_values(values: &[f32], n_embedding: usize, d_model: usize) -> Self {
        Self {
            packed: PackedNf4Tensor::quantize(values).with_double_quantization(),
            n_embedding,
            d_model,
        }
    }

    /// Packed bytes actually held.
    pub fn resident_bytes(&self) -> usize {
        self.packed.resident_bytes()
    }

    /// The dequantized `[n_embedding, d_model]` table, DETACHED.
    pub fn dequantized_weight<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::from_data(
            TensorData::new(self.packed.dequantize(), [self.n_embedding, self.d_model]),
            device,
        )
        .set_require_grad(false)
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        burn::tensor::module::embedding(self.dequantized_weight(&input.device()), input)
    }
}

/// Residency switch for a bias-free linear weight: full f32 (tests, small
/// models), packed NF4 (the 27B), or packed NF4 plus a trainable LoRA
/// adapter (QLoRA fine-tuning). All three forward through
/// `burn::tensor::module::linear`, so downstream code cannot tell them
/// apart by anything but memory.
// The variants cannot be boxed: Burn's `Module` is not implemented for
// `Box<T>`, so a module enum has to hold them inline (same as
// `vit::FeedForward`).
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum QwenLinear<B: Backend> {
    F32(Linear<B>),
    Nf4(Nf4Linear<B>),
    Qlora(QloraLinear<B>),
}

/// A frozen packed-NF4 base weight plus a trainable low-rank adapter:
/// `forward(x) = base(x) + adapter(x)`, mirroring `QLoraLinear` in
/// `quantize.rs` but with the base actually NF4-resident. The adapter is
/// zero-initialized at attach, so attaching is an exact no-op (pinned by
/// test). Burn's `Module` derive only supports single-field tuple enum
/// variants, which is why this is a struct rather than an enum variant
/// with named fields.
#[derive(Module, Debug)]
pub struct QloraLinear<B: Backend> {
    pub base: Nf4Linear<B>,
    pub adapter: LoraAdapter<B>,
}

impl<B: Backend> QloraLinear<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dims = x.dims();
        let d_in = self.adapter.in_features();
        let d_out = self.base.d_output();
        let rows: usize = dims.iter().take(D.saturating_sub(1)).product();
        // The adapter maps `d_in -> d_out`, so its result is as wide as the
        // *base's output*, not as wide as the input. Reshaping it back to the
        // input's shape works only for a square projection and silently
        // mis-shapes every other one — and the two terms of the sum below then
        // fail to line up.
        let mut out_dims = dims;
        if let Some(last) = out_dims.last_mut() {
            *last = d_out;
        }
        let adapted = self
            .adapter
            .forward(x.clone().reshape([rows, d_in]))
            .reshape(out_dims);
        self.base.forward(x) + adapted
    }
}

impl<B: Backend> QwenLinear<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::F32(linear) => linear.forward(x),
            Self::Nf4(linear) => linear.forward(x),
            Self::Qlora(linear) => linear.forward(x),
        }
    }

    /// Resident bytes of the weight storage (f32 values x 4, packed, or
    /// packed + adapter factors).
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::F32(linear) => 4 * linear.weight.dims().iter().product::<usize>(),
            Self::Nf4(linear) => linear.resident_bytes(),
            Self::Qlora(linear) => {
                linear.base.resident_bytes()
                    + 4 * (linear.adapter.in_features() + linear.adapter.out_features())
                        * linear.adapter.rank()
            }
        }
    }

    /// Quantize an f32 weight into NF4 residency, preserving its `ParamId`.
    ///
    /// The id is reused on purpose: a converted trunk must compare equal to the
    /// one it came from under `checkpoint::canonical_hash_hex`, or converting a
    /// trunk would silently invalidate every checkpoint taken before it.
    /// Attached adapters are an error — convert first, then attach.
    pub fn to_nf4(self) -> Result<Self> {
        match self {
            Self::F32(linear) => {
                // Burn's `Linear` already stores the weight as `[d_input,
                // d_output]` — the same row-major order the packed form wants,
                // because both feed `x @ W`. The *checkpoint* is `[out, in]`,
                // which is why the loader above transposes; converting an
                // in-memory trunk must not, and transposing here would swap
                // the axes twice and fail at the matmul rather than at the
                // shape check.
                let [d_in, d_out] = linear.weight.dims();
                let values: Vec<f32> = linear
                    .weight
                    .val()
                    .into_data()
                    .convert::<f32>()
                    .iter()
                    .collect();
                anyhow::ensure!(
                    values.len() == d_in * d_out,
                    "weight of shape [{d_in}, {d_out}] carries {} values",
                    values.len()
                );
                let bias = linear.bias;
                let mut nf4 = Nf4Linear::from_values(&values, d_in, d_out);
                if let Some(b) = bias {
                    nf4.bias = Some(Param::from_tensor(b.val()));
                }
                Ok(Self::Nf4(nf4))
            }
            Self::Nf4(_) => Ok(self),
            Self::Qlora(_) => {
                anyhow::bail!("a weight carrying an adapter must be de-adapted before conversion")
            }
        }
    }

    /// Attach a fresh (zero-init, exact no-op) adapter to an NF4-resident
    /// base. Attaching to an f32 weight or twice is a loud error: silently
    /// dropping or duplicating an adapter would corrupt a run's resume
    /// story.
    pub fn with_lora(self, config: &QwenLoraConfig, device: &B::Device) -> Result<Self> {
        match self {
            Self::Nf4(base) => {
                let lcfg = LoraConfig::new(base.d_input(), base.d_output(), config.rank)
                    .with_alpha(config.alpha);
                let adapter = LoraAdapter::new(&lcfg, device);
                Ok(Self::Qlora(QloraLinear { base, adapter }))
            }
            Self::F32(_) => anyhow::bail!("LoRA adapters attach to NF4-resident weights only"),
            Self::Qlora(_) => anyhow::bail!("weight already carries a LoRA adapter"),
        }
    }
}

/// Residency switch for the embedding table.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum QwenEmbed<B: Backend> {
    F32(Embedding<B>),
    Nf4(Nf4Embedding),
}

impl<B: Backend> QwenEmbed<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        match self {
            Self::F32(embedding) => embedding.forward(input),
            Self::Nf4(embedding) => embedding.forward(input),
        }
    }
}

/// Decode-time state of one [`GatedDeltaHead`].
///
/// - `conv_window [b, K-1, C]`: the last `K - 1` RAW pre-conv qkv rows,
///   zero-init (the zero rows stand in for the causal padding, so the
///   first decoded positions see exactly the prefill window).
/// - `recurrent [b, num_v_heads, dv, dk]`: the delta-rule state `S` of
///   each (group, v-head) pair, zero-init.
#[derive(Debug, Clone)]
pub struct DeltaDecodeState<B: Backend> {
    pub conv_window: Tensor<B, 3>,
    pub recurrent: Tensor<B, 4>,
}

impl<B: Backend> DeltaDecodeState<B> {
    pub fn new(batch: usize, config: GatedDeltaConfig, device: &B::Device) -> Self {
        let conv_channels =
            2 * config.num_k_groups * config.head_k_dim + config.num_v_heads() * config.head_v_dim;
        Self {
            conv_window: Tensor::zeros([batch, config.conv_kernel - 1, conv_channels], device),
            recurrent: Tensor::zeros(
                [
                    batch,
                    config.num_v_heads(),
                    config.head_v_dim,
                    config.head_k_dim,
                ],
                device,
            ),
        }
    }
}

/// One Qwen3.8 gated-DeltaNet linear-attention layer. See the module docs
/// for the exact math. All four input projections and `out_proj` are
/// bias-free; the depthwise conv has no bias on this model generation.
#[derive(Module, Debug)]
pub struct GatedDeltaHead<B: Backend> {
    pub in_proj_qkv: QwenLinear<B>,
    pub in_proj_z: QwenLinear<B>,
    pub in_proj_b: QwenLinear<B>,
    pub in_proj_a: QwenLinear<B>,
    /// Depthwise conv taps `[C, K]` (the checkpoint's `[C, 1, K]` squeezed).
    pub conv_weight: Param<Tensor<B, 2>>,
    pub dt_bias: Param<Tensor<B, 1>>,
    pub a_log: Param<Tensor<B, 1>>,
    pub norm: RmsNormGated<B>,
    pub out_proj: QwenLinear<B>,
    #[module(skip)]
    config: GatedDeltaConfig,
    #[module(skip)]
    hidden: usize,
}

impl<B: Backend> GatedDeltaHead<B> {
    pub fn new(config: GatedDeltaConfig, hidden: usize, device: &B::Device) -> Self {
        let kd = config.num_k_groups * config.head_k_dim;
        let vd = config.num_v_heads() * config.head_v_dim;
        let nv = config.num_v_heads();
        let conv_channels = 2 * kd + vd;
        Self {
            in_proj_qkv: QwenLinear::F32(
                LinearConfig::new(hidden, conv_channels)
                    .with_bias(false)
                    .init(device),
            ),
            in_proj_z: QwenLinear::F32(LinearConfig::new(hidden, vd).with_bias(false).init(device)),
            in_proj_b: QwenLinear::F32(LinearConfig::new(hidden, nv).with_bias(false).init(device)),
            in_proj_a: QwenLinear::F32(LinearConfig::new(hidden, nv).with_bias(false).init(device)),
            conv_weight: Param::from_tensor(Tensor::random(
                [conv_channels, config.conv_kernel],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            dt_bias: Param::from_tensor(Tensor::zeros([nv], device)),
            a_log: Param::from_tensor(Tensor::zeros([nv], device)),
            norm: RmsNormGated::new(config.head_v_dim, 1e-6, device),
            out_proj: QwenLinear::F32(LinearConfig::new(vd, hidden).with_bias(false).init(device)),
            config,
            hidden,
        }
    }

    pub fn config(&self) -> &GatedDeltaConfig {
        &self.config
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// `alpha = exp(-exp(A_log) * softplus(a + dt_bias))` from the raw
    /// `in_proj_a` output; shared by prefill and decode so the two cannot
    /// drift. `a` is `[..., num_v_heads]`.
    fn alpha_from_a(&self, a: Tensor<B, 3>) -> Tensor<B, 3> {
        let nv = self.config.num_v_heads();
        let g = softplus(a + self.dt_bias.val().reshape([1, 1, nv]), 1.0)
            .mul(self.a_log.val().exp().reshape([1, 1, nv]))
            .neg();
        g.exp()
    }

    /// Prefill forward, `x [b, n, h] -> [b, n, h]`, per the module docs.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let cfg = &self.config;
        let [b, n, _h] = x.dims();
        let device = x.device();
        let groups = cfg.num_k_groups;
        let nv = cfg.num_v_heads();
        let dk = cfg.head_k_dim;
        let dv = cfg.head_v_dim;
        let kd = groups * dk;
        let vd = nv * dv;
        let inv_sqrt_dk = 1.0 / (dk as f64).sqrt(); // audit-allow: head dim usize -> f64 scale

        let qkv = self.in_proj_qkv.forward(x.clone());
        let z = self.in_proj_z.forward(x.clone());
        let beta = sigmoid(self.in_proj_b.forward(x.clone()));
        let alpha = self.alpha_from_a(self.in_proj_a.forward(x));

        let mixed = silu3(causal_depthwise_conv(qkv, self.conv_weight.val()));
        let q = mixed.clone().narrow(2, 0, kd);
        let k = mixed.clone().narrow(2, kd, kd);
        let v = mixed.narrow(2, 2 * kd, vd);

        // L2-normalize per head, THEN scale q by dk^-0.5 (HF kernel order).
        let qh = l2norm_last_dim(q.reshape([b, n * groups, dk]))
            .reshape([b, n, groups, dk])
            .mul_scalar(inv_sqrt_dk);
        let kh = l2norm_last_dim(k.reshape([b, n * groups, dk])).reshape([b, n, groups, dk]);

        // V head hh pairs with group hh / v_per_group (adjacent repeat).
        let qp = repeat_adjacent(qh, 2, cfg.v_per_group);
        let kp = repeat_adjacent(kh, 2, cfg.v_per_group);
        let vp = v.reshape([b, n, nv, dv]);

        // Fold (batch, head) into rows: one (k, v) pair per recurrent row.
        let (outs, _finals) = gated_delta_recurrent_batched(
            qp.permute([0, 2, 1, 3]).reshape([b * nv, n, dk]),
            kp.permute([0, 2, 1, 3]).reshape([b * nv, n, dk]),
            vp.permute([0, 2, 1, 3]).reshape([b * nv, n, dv]),
            alpha.permute([0, 2, 1]).reshape([b * nv, n]),
            beta.permute([0, 2, 1]).reshape([b * nv, n]),
            &device,
        );
        let y = outs
            .reshape([b, nv, n, dv])
            .permute([0, 2, 1, 3])
            .reshape([b * n, nv, dv]);
        let z = z.reshape([b * n, nv, dv]);
        let gated = self.norm.forward(y, z).reshape([b, n, vd]);
        self.out_proj.forward(gated)
    }

    /// One decode step: `x [b, h] -> [b, h]`, updating `state` in place.
    /// Reproduces the prefill forward's output at each position bit-for-bit
    /// (the conv window replays the same FIR taps on the same raw rows, and
    /// the recurrent step is the same [`gated_delta_step_tensor`] call).
    pub fn step(&self, x: Tensor<B, 2>, state: &mut DeltaDecodeState<B>) -> Tensor<B, 2> {
        let cfg = &self.config;
        let b = x.dims()[0];
        let groups = cfg.num_k_groups;
        let nv = cfg.num_v_heads();
        let dk = cfg.head_k_dim;
        let dv = cfg.head_v_dim;
        let kd = groups * dk;
        let vd = nv * dv;
        let kernel = cfg.conv_kernel;
        let inv_sqrt_dk = 1.0 / (dk as f64).sqrt(); // audit-allow: head dim usize -> f64 scale

        let qkv = self.in_proj_qkv.forward(x.clone());
        let z = self.in_proj_z.forward(x.clone());
        let beta = sigmoid(self.in_proj_b.forward(x.clone()));
        let alpha = self
            .alpha_from_a(self.in_proj_a.forward(x).unsqueeze_dim::<3>(1))
            .reshape([b, nv]);

        // Append the raw row, FIR at the newest position with the same
        // taps, then drop the oldest row.
        let window = Tensor::cat(
            vec![state.conv_window.clone(), qkv.unsqueeze_dim::<3>(1)],
            1,
        );
        let mixed = silu3(
            causal_depthwise_conv(window.clone(), self.conv_weight.val()).narrow(1, kernel - 1, 1),
        );
        state.conv_window = window.narrow(1, 1, kernel - 1);

        let q = mixed.clone().narrow(2, 0, kd).reshape([b, groups, dk]);
        let k = mixed.clone().narrow(2, kd, kd).reshape([b, groups, dk]);
        let v = mixed.narrow(2, 2 * kd, vd).reshape([b, nv, dv]);

        let qh = l2norm_last_dim(q).mul_scalar(inv_sqrt_dk);
        let kh = l2norm_last_dim(k);
        let qp = repeat_adjacent(qh.unsqueeze_dim::<4>(1), 2, cfg.v_per_group).reshape([b, nv, dk]);
        let kp = repeat_adjacent(kh.unsqueeze_dim::<4>(1), 2, cfg.v_per_group).reshape([b, nv, dk]);

        let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(b * nv);
        let mut states: Vec<Tensor<B, 3>> = Vec::with_capacity(b * nv);
        for bi in 0..b {
            for head in 0..nv {
                let s = state
                    .recurrent
                    .clone()
                    .narrow(0, bi, 1)
                    .narrow(1, head, 1)
                    .reshape([dv, dk]);
                let qt = qp.clone().narrow(0, bi, 1).narrow(1, head, 1).reshape([dk]);
                let kt = kp.clone().narrow(0, bi, 1).narrow(1, head, 1).reshape([dk]);
                let vt = v.clone().narrow(0, bi, 1).narrow(1, head, 1).reshape([dv]);
                let at = alpha
                    .clone()
                    .narrow(0, bi, 1)
                    .narrow(1, head, 1)
                    .reshape([1]);
                let bt = beta
                    .clone()
                    .narrow(0, bi, 1)
                    .narrow(1, head, 1)
                    .reshape([1]);
                let (next, out) = gated_delta_step_tensor(s, qt, kt, vt, at, bt);
                outs.push(out.unsqueeze_dim::<2>(0));
                states.push(next.unsqueeze_dim::<3>(0));
            }
        }
        let y = Tensor::cat(outs, 0).reshape([b, nv, dv]);
        state.recurrent = Tensor::cat(states, 0).reshape([b, nv, dv, dk]);
        let gated = self
            .norm
            .forward(y, z.reshape([b, nv, dv]))
            .reshape([b, vd]);
        self.out_proj.forward(gated)
    }
}

/// Fraction of the full-attention head dim the rotary embedding covers
/// (HF `partial_rotary_factor = 0.25`: 64 of 256 dims on Qwen3.8).
pub const PARTIAL_ROTARY_FRACTION: f64 = 0.25;

fn default_arch_dims() -> QwenArchDims {
    QwenArchDims::qwen38()
}

fn default_rope_base() -> f64 {
    10_000_000.0
}

fn default_norm_eps() -> f64 {
    1e-6
}

fn default_lora_rank() -> usize {
    16
}

fn default_lora_alpha() -> f64 {
    16.0
}

/// LoRA adapter hyperparameters (auto-LoRA on the trunk's NF4-resident
/// projections). Every field has a serde default so sidecars written before
/// a field existed still parse.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QwenLoraConfig {
    /// Adapter rank `r` (16 by default).
    #[serde(default = "default_lora_rank")]
    pub rank: usize,
    /// Scaling numerator; the update is multiplied by `alpha / rank`.
    #[serde(default = "default_lora_alpha")]
    pub alpha: f64,
}

impl Default for QwenLoraConfig {
    fn default() -> Self {
        Self {
            rank: default_lora_rank(),
            alpha: default_lora_alpha(),
        }
    }
}

/// Trunk-level configuration: the architecture dims plus the knobs a run
/// may legitimately vary. Every field carries a serde default so sidecars
/// written before a field existed still parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenTrunkConfig {
    /// Architecture dimensions (`QwenArchDims::qwen38()` for the 27B).
    #[serde(default = "default_arch_dims")]
    pub dims: QwenArchDims,
    /// Override the number of decoder layers built/loaded (`None` = all
    /// `dims.num_layers`); tests load a reduced trunk from real weights.
    #[serde(default)]
    pub num_layers: Option<usize>,
    /// RoPE base of the full-attention layers: **10_000_000.0** on Qwen3.8,
    /// NOT the repo's `hybrid::ROTARY_BASE` (1e4).
    #[serde(default = "default_rope_base")]
    pub rope_base: f64,
    /// RMSNorm epsilon (1e-6 everywhere on this model).
    #[serde(default = "default_norm_eps")]
    pub eps: f64,
}

impl QwenTrunkConfig {
    pub fn qwen38() -> Self {
        Self {
            dims: QwenArchDims::qwen38(),
            num_layers: None,
            rope_base: default_rope_base(),
            eps: default_norm_eps(),
        }
    }

    /// Layers to actually build/load (`dims.num_layers` unless overridden).
    pub fn effective_layers(&self) -> usize {
        self.num_layers.unwrap_or(self.dims.num_layers)
    }
}

/// The always-dense feed-forward block: `down(silu(gate(x)) * up(x))`,
/// all three projections bias-free.
#[derive(Module, Debug)]
pub struct QwenMlp<B: Backend> {
    pub gate_proj: QwenLinear<B>,
    pub up_proj: QwenLinear<B>,
    pub down_proj: QwenLinear<B>,
}

impl<B: Backend> QwenMlp<B> {
    pub fn new(hidden: usize, intermediate: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: QwenLinear::F32(
                LinearConfig::new(hidden, intermediate)
                    .with_bias(false)
                    .init(device),
            ),
            up_proj: QwenLinear::F32(
                LinearConfig::new(hidden, intermediate)
                    .with_bias(false)
                    .init(device),
            ),
            down_proj: QwenLinear::F32(
                LinearConfig::new(intermediate, hidden)
                    .with_bias(false)
                    .init(device),
            ),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu3(self.gate_proj.forward(x.clone()));
        self.down_proj.forward(gate * self.up_proj.forward(x))
    }
}

/// Bias-free q/k/v projections plus the per-head QK norms (norm BEFORE
/// rotary, HF order); rotary and the GQA repeat happen downstream.
#[derive(Debug)]
struct Projections<B: Backend> {
    /// `[b, q_heads, n, dim]`, normed, not yet rotated.
    q: Tensor<B, 4>,
    /// `[b, n, q_heads*dim]`, the sigmoid output gate.
    gate: Tensor<B, 3>,
    /// `[b, kv_heads, n, dim]`, normed, not yet rotated.
    k: Tensor<B, 4>,
    /// `[b, kv_heads, n, dim]`.
    v: Tensor<B, 4>,
}

/// Qwen3.8 full-attention layer (every 4th decoder layer), bias-free
/// everywhere, exactly the HF `modeling_qwen3_5.py` math:
///
/// - `q_proj` widens to `2 * heads * dim`: the first half per head is the
///   query, the second half reshapes to the `[b, n, heads*dim]` sigmoid
///   output gate.
/// - `q_norm`/`k_norm` are zero-centered [`QwenRmsNorm`]s over the head
///   dim, applied per head BEFORE rotary.
/// - Partial RoPE (base `rope_base` = 1e7, NOT the repo default 1e4) over
///   the first `rotary_dim` = 0.25 * dim columns, via
///   [`apply_rotary_partial`] on `[b, heads, n, dim]`.
/// - GQA with ADJACENT repeat (query head `hh` reads kv head
///   `hh / (q_heads / kv_heads)`), scores scaled by `dim^-0.5`, additive
///   causal mask, softmax over keys, then `ctx * sigmoid(gate)` and
///   `o_proj`.
#[derive(Module, Debug)]
pub struct QwenFullAttention<B: Backend> {
    pub q_proj: QwenLinear<B>,
    pub k_proj: QwenLinear<B>,
    pub v_proj: QwenLinear<B>,
    pub o_proj: QwenLinear<B>,
    pub q_norm: QwenRmsNorm<B>,
    pub k_norm: QwenRmsNorm<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    rotary_dim: usize,
    #[module(skip)]
    rope_base: f64,
    #[module(skip)]
    inv_scale: f64,
}

impl<B: Backend> QwenFullAttention<B> {
    pub fn new(dims: &QwenArchDims, rope_base: f64, eps: f64, device: &B::Device) -> Self {
        let h = dims.hidden_size;
        let q_width = dims.num_q_heads * dims.head_dim;
        let q_proj_width = if dims.attn_output_gate {
            2 * q_width
        } else {
            q_width
        };
        Self {
            q_proj: QwenLinear::F32(
                LinearConfig::new(h, q_proj_width)
                    .with_bias(false)
                    .init(device),
            ),
            k_proj: QwenLinear::F32(
                LinearConfig::new(h, dims.num_kv_heads * dims.head_dim)
                    .with_bias(false)
                    .init(device),
            ),
            v_proj: QwenLinear::F32(
                LinearConfig::new(h, dims.num_kv_heads * dims.head_dim)
                    .with_bias(false)
                    .init(device),
            ),
            o_proj: QwenLinear::F32(LinearConfig::new(q_width, h).with_bias(false).init(device)),
            q_norm: QwenRmsNorm::new(dims.head_dim, eps, device),
            k_norm: QwenRmsNorm::new(dims.head_dim, eps, device),
            num_heads: dims.num_q_heads,
            num_kv_heads: dims.num_kv_heads,
            head_dim: dims.head_dim,
            rotary_dim: rotary_dim_for(PARTIAL_ROTARY_FRACTION, dims.head_dim),
            rope_base,
            inv_scale: 1.0 / (dims.head_dim as f64).sqrt(), // audit-allow: head dim usize -> f64 scale
        }
    }

    /// The RoPE base this layer was built with (1e7 on Qwen3.8).
    pub fn rope_base(&self) -> f64 {
        self.rope_base
    }

    /// Head-dim columns the rotary embedding covers (64 of 256 on Qwen3.8).
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    /// Bias-free projections plus the per-head QK norms; `x` is `[b, n, h]`
    /// in prefill and `[b, 1, h]` in decode so both paths run the identical
    /// rank-3 linears.
    fn project(&self, x: Tensor<B, 3>) -> Projections<B> {
        let [b, n, _h] = x.dims();
        let (heads, kv, dim) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let qh = self
            .q_proj
            .forward(x.clone())
            .reshape([b, n, heads, 2 * dim]);
        let q = self.q_norm.forward_4d(qh.clone().narrow(3, 0, dim));
        let gate = qh.narrow(3, dim, dim).reshape([b, n, heads * dim]);
        let k = self
            .k_norm
            .forward_4d(self.k_proj.forward(x.clone()).reshape([b, n, kv, dim]));
        let v = self.v_proj.forward(x).reshape([b, n, kv, dim]);
        Projections {
            q: q.permute([0, 2, 1, 3]),
            gate,
            k: k.permute([0, 2, 1, 3]),
            v: v.permute([0, 2, 1, 3]),
        }
    }

    /// Scaled dot-product attention with adjacent-repeat GQA and an
    /// additive mask; returns the merged heads `[b, m, heads*dim]`.
    fn attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [b, heads, m, dim] = q.dims();
        let group = heads / self.num_kv_heads;
        let kk = repeat_adjacent(k, 1, group);
        let vv = repeat_adjacent(v, 1, group);
        let scores = q.matmul(kk.transpose()).mul_scalar(self.inv_scale) + mask;
        softmax(scores, 3)
            .matmul(vv)
            .permute([0, 2, 1, 3])
            .reshape([b, m, heads * dim])
    }

    /// Prefill forward, `x [b, n, h] -> [b, n, h]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let n = x.dims()[1];
        let device = x.device();
        let p = self.project(x);
        let q = apply_rotary_partial(p.q, 0, self.rope_base, self.rotary_dim);
        let k = apply_rotary_partial(p.k, 0, self.rope_base, self.rotary_dim);
        let ctx = self.attend(q, k, p.v, causal_mask::<B>(n, &device));
        self.o_proj.forward(ctx.mul(sigmoid(p.gate)))
    }

    /// One decode step against the KV cache in `state`: the new token's
    /// q/k are rotated at the cache's absolute offset, k/v are appended,
    /// and the single query row attends to every cached position. Reproduces
    /// the prefill forward's output at that position bit-for-bit (certified
    /// in `verify.rs`).
    pub fn step(&self, x: Tensor<B, 2>, state: &mut LayerState<B>) -> Tensor<B, 2> {
        let b = x.dims()[0];
        let offset = state.positions;
        let device = x.device();
        let p = self.project(x.unsqueeze_dim::<3>(1));
        let q = apply_rotary_partial(p.q, offset, self.rope_base, self.rotary_dim);
        let k = apply_rotary_partial(p.k, offset, self.rope_base, self.rotary_dim);
        let (k, v) = state.push_kv(k, p.v, None);
        state.positions += 1;
        let mask = attention_mask::<B>(
            1,
            k.dims()[2],
            offset,
            state.first_key_position,
            None,
            &device,
        );
        let ctx = self.attend(q, k, v, mask);
        let gate = p.gate.reshape([b, self.num_heads * self.head_dim]);
        self.o_proj.forward(
            ctx.reshape([b, self.num_heads * self.head_dim])
                .mul(sigmoid(gate)),
        )
    }

    /// Scores-only max-logit probe (training-stability monitor, mirrors
    /// `vit::Attention::max_logit`): project q/k, per-head QK norms, rotary
    /// at offset 0, adjacent-repeat GQA, scaled scores with the causal
    /// mask, then the single-element maximum. No softmax, no value or
    /// output projections. Delta layers return `None` at the layer level.
    pub fn max_logit(&self, x: Tensor<B, 3>) -> Tensor<B, 1> {
        let n = x.dims()[1];
        let device = x.device();
        let p = self.project(x);
        let q = apply_rotary_partial(p.q, 0, self.rope_base, self.rotary_dim);
        let k = apply_rotary_partial(p.k, 0, self.rope_base, self.rotary_dim);
        let kk = repeat_adjacent(k, 1, self.num_heads / self.num_kv_heads);
        let scores =
            q.matmul(kk.transpose()).mul_scalar(self.inv_scale) + causal_mask::<B>(n, &device);
        scores.max()
    }
}

/// The per-layer mixer: a gated-DeltaNet linear layer or a full-attention
/// layer. Enum modules have precedent in `vit::TrunkNorm`.
// The variants cannot be boxed: Burn's `Module` is not implemented for
// `Box<T>`, so a module enum has to hold them inline (same as
// `vit::FeedForward`).
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum QwenMixer<B: Backend> {
    Linear(GatedDeltaHead<B>),
    Full(QwenFullAttention<B>),
}

/// Decode state of one decoder layer, kind-matched to its mixer.
#[derive(Debug, Clone)]
pub enum QwenLayerState<B: Backend> {
    Linear(DeltaDecodeState<B>),
    Full(LayerState<B>),
}

/// One Qwen3.8 decoder layer: pre-norm residual twice
/// (`input_layernorm` -> mixer -> +x; `post_attention_layernorm` -> MLP ->
/// +x).
#[derive(Module, Debug)]
pub struct QwenDecoderLayer<B: Backend> {
    pub input_norm: QwenRmsNorm<B>,
    pub mixer: QwenMixer<B>,
    pub post_norm: QwenRmsNorm<B>,
    pub mlp: QwenMlp<B>,
}

impl<B: Backend> QwenDecoderLayer<B> {
    pub fn new(
        dims: &QwenArchDims,
        layer_idx: usize,
        rope_base: f64,
        eps: f64,
        device: &B::Device,
    ) -> Self {
        let mixer = if dims.is_full_attention(layer_idx) {
            QwenMixer::Full(QwenFullAttention::new(dims, rope_base, eps, device))
        } else {
            QwenMixer::Linear(GatedDeltaHead::new(
                gated_delta_config(dims),
                dims.hidden_size,
                device,
            ))
        };
        Self {
            input_norm: QwenRmsNorm::new(dims.hidden_size, eps, device),
            mixer,
            post_norm: QwenRmsNorm::new(dims.hidden_size, eps, device),
            mlp: QwenMlp::new(dims.hidden_size, dims.intermediate_size, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mixed = match &self.mixer {
            QwenMixer::Linear(head) => head.forward(self.input_norm.forward(x.clone())),
            QwenMixer::Full(attn) => attn.forward(self.input_norm.forward(x.clone())),
        };
        let x = x + mixed;
        let fed = self.mlp.forward(self.post_norm.forward(x.clone()));
        x + fed
    }

    /// One decode step, `x [b, h] -> [b, h]`; the norms run through the
    /// rank-3 path on `[b, 1, h]` so prefill and decode share every op.
    pub fn step(&self, x: Tensor<B, 2>, state: &mut QwenLayerState<B>) -> Result<Tensor<B, 2>> {
        let [b, h] = x.dims();
        let normed = self
            .input_norm
            .forward(x.clone().unsqueeze_dim::<3>(1))
            .reshape([b, h]);
        let mixed = match (&self.mixer, state) {
            (QwenMixer::Linear(head), QwenLayerState::Linear(s)) => head.step(normed, s),
            (QwenMixer::Full(attn), QwenLayerState::Full(s)) => attn.step(normed, s),
            _ => anyhow::bail!("decode state kind does not match this layer's mixer"),
        };
        let x = x + mixed;
        let fed = self
            .mlp
            .forward(self.post_norm.forward(x.clone().unsqueeze_dim::<3>(1)))
            .reshape([b, h]);
        Ok(x + fed)
    }

    /// Scores-only max-logit probe of this layer's attention, on the layer
    /// INPUT (the norm is applied inside, as in the forward path). `None`
    /// for gated-DeltaNet layers, which have no attention logits -- same
    /// convention as `vit`'s `AttentionMode::Linear`.
    pub fn max_logit(&self, x: Tensor<B, 3>) -> Option<Tensor<B, 1>> {
        match &self.mixer {
            QwenMixer::Linear(_) => None,
            QwenMixer::Full(attn) => Some(attn.max_logit(self.input_norm.forward(x))),
        }
    }
}

/// Decode state of the whole trunk: one entry per layer, in layer order.
#[derive(Debug, Clone)]
pub struct QwenTrunkState<B: Backend> {
    pub layers: Vec<QwenLayerState<B>>,
}

/// The Qwen3.8 text trunk: token embedding, `effective_layers()` decoder
/// layers (full attention every `full_attention_interval`-th, linear
/// gated-DeltaNet otherwise), final zero-centered RMSNorm, and the UNTIED
/// bias-free `lm_head`.
#[derive(Module, Debug)]
pub struct QwenTrunk<B: Backend> {
    pub embed_tokens: QwenEmbed<B>,
    pub layers: Vec<QwenDecoderLayer<B>>,
    pub final_norm: QwenRmsNorm<B>,
    pub lm_head: QwenLinear<B>,
    #[module(skip)]
    config: QwenTrunkConfig,
}

/// A throwaway f32 linear used to move a slot out for in-place conversion.
///
/// `to_nf4` consumes the slot's value, and a move out of a struct field needs
/// something to leave behind. A 1x1 keeps the replacement cheap and is never
/// observed, because the conversion overwrites it before the next read.
fn placeholder_linear<B: Backend>(device: &B::Device) -> QwenLinear<B> {
    QwenLinear::F32(LinearConfig::new(1, 1).with_bias(false).init(device))
}

impl<B: Backend> QwenTrunk<B> {
    /// Quantize every projection in the trunk to NF4 residency.
    ///
    /// The from-scratch counterpart to [`load_qwen_trunk_nf4`]: that one reads
    /// packed weights from a checkpoint, this one packs an in-memory trunk. It
    /// is what makes the adapters-only training path reachable from a randomly
    /// initialised model — the shakedown run — instead of only from real
    /// weights on disk.
    ///
    /// Embeddings and norms are left alone: `QwenEmbed::Nf4` already exists for
    /// the packed case and no test here needs it, so converting it would add a
    /// code path with no caller. Only the projections that carry adapters move.
    pub fn to_nf4(&mut self) -> Result<()> {
        self.lm_head =
            std::mem::replace(&mut self.lm_head, placeholder_linear(&Default::default()))
                .to_nf4()?;
        for layer in &mut self.layers {
            match &mut layer.mixer {
                QwenMixer::Linear(head) => {
                    head.in_proj_qkv = std::mem::replace(
                        &mut head.in_proj_qkv,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                    head.in_proj_z = std::mem::replace(
                        &mut head.in_proj_z,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                    head.in_proj_b = std::mem::replace(
                        &mut head.in_proj_b,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                    head.in_proj_a = std::mem::replace(
                        &mut head.in_proj_a,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                    head.out_proj = std::mem::replace(
                        &mut head.out_proj,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                }
                QwenMixer::Full(attn) => {
                    attn.q_proj = std::mem::replace(
                        &mut attn.q_proj,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                    attn.k_proj = std::mem::replace(
                        &mut attn.k_proj,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                    attn.v_proj = std::mem::replace(
                        &mut attn.v_proj,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                    attn.o_proj = std::mem::replace(
                        &mut attn.o_proj,
                        placeholder_linear(&Default::default()),
                    )
                    .to_nf4()?;
                }
            }
            layer.mlp.gate_proj = std::mem::replace(
                &mut layer.mlp.gate_proj,
                placeholder_linear(&Default::default()),
            )
            .to_nf4()?;
            layer.mlp.up_proj = std::mem::replace(
                &mut layer.mlp.up_proj,
                placeholder_linear(&Default::default()),
            )
            .to_nf4()?;
            layer.mlp.down_proj = std::mem::replace(
                &mut layer.mlp.down_proj,
                placeholder_linear(&Default::default()),
            )
            .to_nf4()?;
        }
        Ok(())
    }

    pub fn new(config: QwenTrunkConfig, device: &B::Device) -> Self {
        let dims = config.dims;
        let layers = (0..config.effective_layers())
            .map(|idx| QwenDecoderLayer::new(&dims, idx, config.rope_base, config.eps, device))
            .collect();
        Self {
            embed_tokens: QwenEmbed::F32(
                EmbeddingConfig::new(dims.vocab_size, dims.hidden_size).init(device),
            ),
            layers,
            final_norm: QwenRmsNorm::new(dims.hidden_size, config.eps, device),
            lm_head: QwenLinear::F32(
                LinearConfig::new(dims.hidden_size, dims.vocab_size)
                    .with_bias(false)
                    .init(device),
            ),
            config,
        }
    }

    pub fn config(&self) -> &QwenTrunkConfig {
        &self.config
    }

    /// Everything but `lm_head`: embed, decoder layers, final norm.
    pub fn forward_hidden(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embed_tokens.forward(tokens);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.final_norm.forward(x)
    }

    /// `tokens [b, n]` -> logits `[b, n, vocab]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.lm_head.forward(self.forward_hidden(tokens))
    }

    /// Per-layer scores-only max-logit probe: `Some` for full-attention
    /// layers, `None` for gated-DeltaNet layers, in layer order. Runs the
    /// trunk forward once, probing each layer on its own input.
    pub fn max_logits(&self, tokens: Tensor<B, 2, Int>) -> Vec<Option<Tensor<B, 1>>> {
        let mut x = self.embed_tokens.forward(tokens);
        let mut out = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            out.push(layer.max_logit(x.clone()));
            x = layer.forward(x);
        }
        out
    }

    /// A fresh decode state matching this trunk's layer kinds, in order.
    pub fn new_state(&self, batch: usize, device: &B::Device) -> QwenTrunkState<B> {
        let layers = self
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                QwenMixer::Linear(head) => {
                    QwenLayerState::Linear(DeltaDecodeState::new(batch, *head.config(), device))
                }
                QwenMixer::Full(_) => QwenLayerState::Full(LayerState::new()),
            })
            .collect();
        QwenTrunkState { layers }
    }

    /// One decode step: `tokens [b, 1]` -> logits `[b, vocab]`.
    pub fn step(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut QwenTrunkState<B>,
    ) -> Result<Tensor<B, 2>> {
        ensure!(
            tokens.dims()[1] == 1,
            "trunk decode step takes exactly one token per row, got shape {:?}",
            tokens.dims()
        );
        ensure!(
            state.layers.len() == self.layers.len(),
            "decode state has {} layers, trunk has {}",
            state.layers.len(),
            self.layers.len()
        );
        let b = tokens.dims()[0];
        let h = self.config.dims.hidden_size;
        let mut x = self.embed_tokens.forward(tokens).reshape([b, h]);
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter_mut()) {
            x = layer.step(x, layer_state)?;
        }
        let x = self
            .final_norm
            .forward(x.unsqueeze_dim::<3>(1))
            .reshape([b, h]);
        Ok(self.lm_head.forward(x))
    }
}

/// Transpose a flat row-major `[rows, cols]` buffer into flat row-major
/// `[cols, rows]`, in 64x64 blocks so the strided writes stay cache-local.
/// A naive element-order pass writes with a `rows`-element stride and
/// misses cache (and TLB) on essentially every element; blocked, each
/// 64x64 tile writes 64 contiguous runs of 256 bytes. Measured on the
/// 27B `lm_head` (248320 x 5120 f32, 5.1 GB): 2.4 s blocked vs 4.4 s
/// naive on the workspace machine.
fn transpose2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    const BLOCK: usize = 64;
    for r0 in (0..rows).step_by(BLOCK) {
        let rend = (r0 + BLOCK).min(rows);
        for c0 in (0..cols).step_by(BLOCK) {
            let cend = (c0 + BLOCK).min(cols);
            for (r, row) in data.chunks_exact(cols).enumerate().skip(r0).take(rend - r0) {
                for (c, value) in row.iter().enumerate().skip(c0).take(cend - c0) {
                    out[c * rows + r] = *value;
                }
            }
        }
    }
    out
}

/// Checkpoint linear `[out, in]` (PyTorch) -> Burn `Param` `[in, out]`.
/// The transposed Vec moves into `TensorData` (no second copy of the
/// multi-GB `lm_head`), and `from_data` skips the `from_floats` slice copy.
fn linear_param<B: Backend>(
    tensor: &DecodedTensor,
    device: &B::Device,
) -> Result<Param<Tensor<B, 2>>> {
    let rows = *tensor
        .shape
        .first()
        .with_context(|| format!("tensor {:?} has no dims", tensor.name))?;
    let cols = *tensor
        .shape
        .get(1)
        .with_context(|| format!("tensor {:?} is not rank-2", tensor.name))?;
    ensure!(
        tensor.shape.len() == 2 && rows * cols == tensor.data.len(),
        "tensor {:?} shape {:?} does not match {} decoded values",
        tensor.name,
        tensor.shape,
        tensor.data.len()
    );
    let data = transpose2d(&tensor.data, rows, cols);
    Ok(Param::from_tensor(Tensor::<B, 2>::from_data(
        TensorData::new(data, [cols, rows]),
        device,
    )))
}

/// Checkpoint depthwise conv `[C, 1, K]` -> Burn `Param` `[C, K]`.
fn conv_param<B: Backend>(
    tensor: &DecodedTensor,
    device: &B::Device,
) -> Result<Param<Tensor<B, 2>>> {
    let channels = *tensor
        .shape
        .first()
        .with_context(|| format!("tensor {:?} has no dims", tensor.name))?;
    let kernel = *tensor
        .shape
        .get(2)
        .with_context(|| format!("tensor {:?} is not rank-3", tensor.name))?;
    ensure!(
        tensor.shape.len() == 3
            && tensor.shape.get(1) == Some(&1)
            && channels * kernel == tensor.data.len(),
        "tensor {:?} shape {:?} is not a depthwise [C, 1, K] conv weight",
        tensor.name,
        tensor.shape
    );
    Ok(Param::from_tensor(
        Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device).reshape([channels, kernel]),
    ))
}

/// Checkpoint rank-1 vector -> Burn `Param`.
fn vector_param<B: Backend>(
    tensor: &DecodedTensor,
    device: &B::Device,
) -> Result<Param<Tensor<B, 1>>> {
    ensure!(
        tensor.shape.len() == 1 && tensor.shape.first() == Some(&tensor.data.len()),
        "tensor {:?} shape {:?} is not rank-1",
        tensor.name,
        tensor.shape
    );
    Ok(Param::from_tensor(Tensor::<B, 1>::from_floats(
        tensor.data.as_slice(),
        device,
    )))
}

/// The [`GatedDeltaConfig`] a linear layer of `dims` implies. Callers
/// ensure `linear_v_heads % linear_k_groups == 0` first.
fn gated_delta_config(dims: &QwenArchDims) -> GatedDeltaConfig {
    GatedDeltaConfig {
        num_k_groups: dims.linear_k_groups,
        v_per_group: dims.linear_v_heads / dims.linear_k_groups,
        head_k_dim: dims.linear_head_dim,
        head_v_dim: dims.linear_head_dim,
        conv_kernel: dims.conv_kernel,
    }
}

/// Header-only map of every `model-*.safetensors` shard under a directory:
/// tensor name -> shard, built ONCE so a 64-layer load does not re-index
/// twelve shards per layer.
struct ShardMap {
    dir: PathBuf,
    shards: Vec<(PathBuf, SafetensorsIndex)>,
    location: HashMap<String, usize>,
}

impl ShardMap {
    fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir).with_context(|| format!("list {}", dir.display()))? {
            let entry = entry.with_context(|| format!("list {}", dir.display()))?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("model-") && name.ends_with(".safetensors") {
                paths.push(entry.path());
            }
        }
        paths.sort();
        ensure!(
            !paths.is_empty(),
            "no model-*.safetensors shards under {}",
            dir.display()
        );

        let mut shards: Vec<(PathBuf, SafetensorsIndex)> = Vec::with_capacity(paths.len());
        let mut location: HashMap<String, usize> = HashMap::new();
        for path in paths {
            let index = open_index(&path).with_context(|| format!("index {}", path.display()))?;
            let slot = shards.len();
            for tensor in &index.tensors {
                ensure!(
                    location.insert(tensor.name.clone(), slot).is_none(),
                    "tensor {:?} appears in more than one shard under {}",
                    tensor.name,
                    dir.display()
                );
            }
            shards.push((path, index));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            shards,
            location,
        })
    }

    /// Decode one named tensor and require an exact shape and all-finite
    /// values (loud errors naming the tensor, never a silent clamp).
    fn read_checked(&self, name: &str, want: &[usize]) -> Result<DecodedTensor> {
        let slot = *self.location.get(name).with_context(|| {
            format!(
                "tensor {name:?} not found in any shard under {}",
                self.dir.display()
            )
        })?;
        let (path, index) = &self.shards[slot];
        let tensor = read_tensor_f32(path, index, name)?;
        ensure!(
            tensor.shape.as_slice() == want,
            "tensor {name:?} shape {:?}, expected {want:?}",
            tensor.shape
        );
        ensure!(
            tensor.data.iter().all(|v| v.is_finite()),
            "tensor {name:?} contains non-finite values"
        );
        Ok(tensor)
    }

    /// [`Self::read_checked`] with the shape [`QwenPart::expected_shape`]
    /// implies for `dims`.
    fn read_part(&self, name: &str, part: QwenPart, dims: &QwenArchDims) -> Result<DecodedTensor> {
        self.read_checked(name, &part.expected_shape(dims))
    }
}

/// Checkpoint rank-2 matrix with the SAME layout in both worlds (the
/// embedding table, `[vocab, hidden]` row-gather in PyTorch and Burn) ->
/// Burn `Param`, no transpose. Takes the decoded tensor by value so the
/// 5 GB buffer moves into `TensorData` instead of being cloned.
fn matrix_param<B: Backend>(
    tensor: DecodedTensor,
    device: &B::Device,
) -> Result<Param<Tensor<B, 2>>> {
    let rows = *tensor
        .shape
        .first()
        .with_context(|| format!("tensor {:?} has no dims", tensor.name))?;
    let cols = *tensor
        .shape
        .get(1)
        .with_context(|| format!("tensor {:?} is not rank-2", tensor.name))?;
    ensure!(
        tensor.shape.len() == 2 && rows * cols == tensor.data.len(),
        "tensor {:?} shape {:?} does not match {} decoded values",
        tensor.name,
        tensor.shape,
        tensor.data.len()
    );
    Ok(Param::from_tensor(Tensor::<B, 2>::from_data(
        TensorData::new(tensor.data, [rows, cols]),
        device,
    )))
}

/// How a loaded rank-2 weight is stored: full f32 (tests, small models) or
/// packed NF4 with double quantization (the 27B). See the NF4-RESIDENCY
/// POLICY note above: rank-1 tensors and the conv weight stay f32 in both
/// modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightStore {
    F32,
    Nf4,
}

/// Resident-memory accounting of a trunk load: what the budget gate reads.
/// Totals are sums over the per-class rows; the scratch figure is the
/// estimated peak transient during the load itself (read -> transpose ->
/// quantize -> drop, one tensor at a time, so the transient is the LARGEST
/// single tensor, never the whole model).
#[derive(Debug, Clone, Default)]
pub struct Nf4ResidencyReport {
    /// Bytes held in packed NF4 form (codes + scales).
    pub packed_bytes: usize,
    /// Bytes kept in f32 (norms, `dt_bias`/`A_log`, conv; plus every weight
    /// when the loader runs in `WeightStore::F32`).
    pub f32_kept_bytes: usize,
    /// Per-tensor-class breakdown.
    pub by_class: Vec<ClassResidency>,
    /// Estimated peak scratch during the load (largest single tensor's f32
    /// buffers + its packed form; the `lm_head` dominates at ~10.8 GB).
    pub peak_scratch_bytes: usize,
}

/// One row of [`Nf4ResidencyReport::by_class`].
#[derive(Debug, Clone, Default)]
pub struct ClassResidency {
    pub class: String,
    pub tensors: usize,
    pub packed_bytes: usize,
    pub f32_bytes: usize,
}

impl Nf4ResidencyReport {
    /// Total bytes the loaded trunk holds.
    pub fn total_resident_bytes(&self) -> usize {
        self.packed_bytes + self.f32_kept_bytes
    }

    fn class_mut(&mut self, class: &str) -> &mut ClassResidency {
        let idx = match self.by_class.iter().position(|c| c.class == class) {
            Some(idx) => idx,
            None => {
                self.by_class.push(ClassResidency {
                    class: class.to_string(),
                    tensors: 0,
                    packed_bytes: 0,
                    f32_bytes: 0,
                });
                self.by_class.len() - 1
            }
        };
        &mut self.by_class[idx]
    }

    fn record(&mut self, class: &str, bytes: usize, store: WeightStore) {
        match store {
            WeightStore::Nf4 => self.packed_bytes += bytes,
            WeightStore::F32 => self.f32_kept_bytes += bytes,
        }
        let row = self.class_mut(class);
        match store {
            WeightStore::Nf4 => row.packed_bytes += bytes,
            WeightStore::F32 => row.f32_bytes += bytes,
        }
        row.tensors += 1;
    }

    fn record_f32(&mut self, class: &str, bytes: usize) {
        self.f32_kept_bytes += bytes;
        let row = self.class_mut(class);
        row.f32_bytes += bytes;
        row.tensors += 1;
    }

    fn note_scratch(&mut self, bytes: usize) {
        self.peak_scratch_bytes = self.peak_scratch_bytes.max(bytes);
    }
}

/// Convert one decoded checkpoint linear (`[out, in]`, PyTorch) into a
/// `QwenLinear` in the requested storage, returning the resident bytes of
/// the chosen form. NF4: host-transpose to `[in, out]`, quantize packed
/// with double quantization, drop the f32 copies.
fn qwen_linear<B: Backend>(
    tensor: &DecodedTensor,
    store: WeightStore,
    device: &B::Device,
) -> Result<(QwenLinear<B>, usize)> {
    match store {
        WeightStore::F32 => {
            let bytes = 4 * tensor.data.len();
            let linear = QwenLinear::F32(Linear {
                weight: linear_param(tensor, device)?,
                bias: None,
            });
            Ok((linear, bytes))
        }
        WeightStore::Nf4 => {
            let rows = *tensor
                .shape
                .first()
                .with_context(|| format!("tensor {:?} has no dims", tensor.name))?;
            let cols = *tensor
                .shape
                .get(1)
                .with_context(|| format!("tensor {:?} is not rank-2", tensor.name))?;
            ensure!(
                tensor.shape.len() == 2 && rows * cols == tensor.data.len(),
                "tensor {:?} shape {:?} does not match {} decoded values",
                tensor.name,
                tensor.shape,
                tensor.data.len()
            );
            let data = transpose2d(&tensor.data, rows, cols);
            let packed = PackedNf4Tensor::quantize(&data).with_double_quantization();
            let bytes = packed.resident_bytes();
            Ok((
                QwenLinear::Nf4(Nf4Linear::from_packed(packed, cols, rows, None)),
                bytes,
            ))
        }
    }
}

/// Fill the nine `linear_attn` tensors of `layer_idx` into `head`,
/// recording residency under class `linear_attn`.
fn fill_gated_delta_head<B: Backend>(
    head: &mut GatedDeltaHead<B>,
    map: &ShardMap,
    layer_idx: usize,
    dims: &QwenArchDims,
    store: WeightStore,
    report: &mut Nf4ResidencyReport,
    device: &B::Device,
) -> Result<()> {
    let p = |suffix: &str| format!("model.language_model.layers.{layer_idx}.linear_attn.{suffix}");
    let class = "linear_attn";
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("in_proj_qkv.weight"), QwenPart::LinearQkv, dims)?,
        store,
        device,
    )?;
    head.in_proj_qkv = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("in_proj_z.weight"), QwenPart::LinearZ, dims)?,
        store,
        device,
    )?;
    head.in_proj_z = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("in_proj_b.weight"), QwenPart::LinearB, dims)?,
        store,
        device,
    )?;
    head.in_proj_b = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("in_proj_a.weight"), QwenPart::LinearA, dims)?,
        store,
        device,
    )?;
    head.in_proj_a = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("out_proj.weight"), QwenPart::LinearOut, dims)?,
        store,
        device,
    )?;
    head.out_proj = w;
    report.record(class, bytes, store);
    // Rank-1 tensors and the conv weight stay f32 in BOTH modes (policy).
    let conv = map.read_part(&p("conv1d.weight"), QwenPart::LinearConv, dims)?;
    report.record_f32(class, 4 * conv.data.len());
    head.conv_weight = conv_param(&conv, device)?;
    let norm = map.read_part(&p("norm.weight"), QwenPart::LinearGateNorm, dims)?;
    report.record_f32(class, 4 * norm.data.len());
    head.norm.weight = vector_param(&norm, device)?;
    let dt_bias = map.read_part(&p("dt_bias"), QwenPart::DeltaDtBias, dims)?;
    report.record_f32(class, 4 * dt_bias.data.len());
    head.dt_bias = vector_param(&dt_bias, device)?;
    let a_log = map.read_part(&p("A_log"), QwenPart::DeltaALog, dims)?;
    report.record_f32(class, 4 * a_log.data.len());
    head.a_log = vector_param(&a_log, device)?;
    Ok(())
}

/// Fill the six `self_attn` tensors of `layer_idx` into `attn`,
/// recording residency under class `self_attn`.
fn fill_full_attention<B: Backend>(
    attn: &mut QwenFullAttention<B>,
    map: &ShardMap,
    layer_idx: usize,
    dims: &QwenArchDims,
    store: WeightStore,
    report: &mut Nf4ResidencyReport,
    device: &B::Device,
) -> Result<()> {
    let p = |suffix: &str| format!("model.language_model.layers.{layer_idx}.self_attn.{suffix}");
    let class = "self_attn";
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("q_proj.weight"), QwenPart::FullQ, dims)?,
        store,
        device,
    )?;
    attn.q_proj = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("k_proj.weight"), QwenPart::FullK, dims)?,
        store,
        device,
    )?;
    attn.k_proj = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("v_proj.weight"), QwenPart::FullV, dims)?,
        store,
        device,
    )?;
    attn.v_proj = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("o_proj.weight"), QwenPart::FullO, dims)?,
        store,
        device,
    )?;
    attn.o_proj = w;
    report.record(class, bytes, store);
    let q_norm = map.read_part(&p("q_norm.weight"), QwenPart::FullQNorm, dims)?;
    report.record_f32(class, 4 * q_norm.data.len());
    attn.q_norm.weight = vector_param(&q_norm, device)?;
    let k_norm = map.read_part(&p("k_norm.weight"), QwenPart::FullKNorm, dims)?;
    report.record_f32(class, 4 * k_norm.data.len());
    attn.k_norm.weight = vector_param(&k_norm, device)?;
    Ok(())
}

/// Fill the three `mlp` tensors of `layer_idx` into `mlp`, recording
/// residency under class `mlp`.
fn fill_mlp<B: Backend>(
    mlp: &mut QwenMlp<B>,
    map: &ShardMap,
    layer_idx: usize,
    dims: &QwenArchDims,
    store: WeightStore,
    report: &mut Nf4ResidencyReport,
    device: &B::Device,
) -> Result<()> {
    let p = |suffix: &str| format!("model.language_model.layers.{layer_idx}.mlp.{suffix}");
    let class = "mlp";
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("gate_proj.weight"), QwenPart::MlpGate, dims)?,
        store,
        device,
    )?;
    mlp.gate_proj = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("up_proj.weight"), QwenPart::MlpUp, dims)?,
        store,
        device,
    )?;
    mlp.up_proj = w;
    report.record(class, bytes, store);
    let (w, bytes) = qwen_linear(
        &map.read_part(&p("down_proj.weight"), QwenPart::MlpDown, dims)?,
        store,
        device,
    )?;
    mlp.down_proj = w;
    report.record(class, bytes, store);
    Ok(())
}

/// Checks shared by the single-layer loaders.
fn ensure_linear_layer(layer_idx: usize, dims: &QwenArchDims) -> Result<()> {
    ensure!(
        layer_idx < dims.num_layers,
        "layer index {layer_idx} beyond {} layers",
        dims.num_layers
    );
    ensure!(
        !dims.is_full_attention(layer_idx),
        "layer {layer_idx} is a full-attention layer; it has no linear_attn tensors"
    );
    ensure!(
        dims.linear_v_heads % dims.linear_k_groups == 0,
        "value heads {} not a multiple of key groups {}",
        dims.linear_v_heads,
        dims.linear_k_groups
    );
    Ok(())
}

/// Shared prelude of both trunk loaders: config sanity, the shard map
/// (built ONCE), and a full audit of every shard BEFORE anything is
/// filled.
fn open_audited_shards(dir: &Path, config: &QwenTrunkConfig) -> Result<ShardMap> {
    let dims = &config.dims;
    ensure!(
        config.effective_layers() <= dims.num_layers,
        "config asks for {} layers, architecture has {}",
        config.effective_layers(),
        dims.num_layers
    );
    ensure!(
        dims.linear_v_heads % dims.linear_k_groups == 0,
        "value heads {} not a multiple of key groups {}",
        dims.linear_v_heads,
        dims.linear_k_groups
    );
    ensure!(
        dims.num_q_heads % dims.num_kv_heads == 0,
        "query heads {} not a multiple of kv heads {}",
        dims.num_q_heads,
        dims.num_kv_heads
    );

    let map = ShardMap::open(dir)?;
    // Audit every shard BEFORE loading anything.
    for (path, index) in &map.shards {
        let entries: Vec<(String, Vec<usize>)> = index
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect();
        let reports = audit_tensors(
            &entries,
            dims,
            "model.language_model",
            &["mtp.", "model.visual."],
        )
        .with_context(|| format!("audit {}", path.display()))?;
        if let Some(report) = reports.iter().find(|r| !r.shape_ok) {
            anyhow::bail!(
                "shard {} tensor {:?}: {}",
                path.display(),
                report.name,
                report.detail
            );
        }
    }
    Ok(map)
}

/// The shared body of both trunk loaders: audit first, then stream
/// tensor-by-tensor (read -> fill -> drop) in the chosen weight storage.
/// Fill order keeps the memory peak low -- the untied 1.27B-element
/// `lm_head` first, then `embed_tokens`, then one layer at a time. The
/// Burn modules initialize lazily, so the skeleton built by
/// [`QwenTrunk::new`] does not allocate the random weights being replaced.
fn load_trunk_impl<B: Backend>(
    dir: &Path,
    config: &QwenTrunkConfig,
    store: WeightStore,
    device: &B::Device,
) -> Result<(QwenTrunk<B>, Nf4ResidencyReport)> {
    let dims = &config.dims;
    let map = open_audited_shards(dir, config)?;
    let mut report = Nf4ResidencyReport::default();
    let mut trunk = QwenTrunk::new(config.clone(), device);

    // Untied head: PyTorch [vocab, h] = [out, in] -> Burn [h, vocab].
    let lm = map.read_checked("lm_head.weight", &[dims.vocab_size, dims.hidden_size])?;
    report.note_scratch(
        8 * lm.data.len() + PackedNf4Tensor::estimated_resident_bytes(lm.data.len(), true),
    );
    let (w, bytes) = qwen_linear(&lm, store, device)?;
    trunk.lm_head = w;
    report.record("lm_head", bytes, store);
    drop(lm);
    // Embedding table is a row-gather [vocab, h] in both layouts: no
    // transpose in either storage mode.
    let embed = map.read_checked(
        "model.language_model.embed_tokens.weight",
        &[dims.vocab_size, dims.hidden_size],
    )?;
    report.note_scratch(
        4 * embed.data.len() + PackedNf4Tensor::estimated_resident_bytes(embed.data.len(), true),
    );
    match store {
        WeightStore::F32 => {
            let bytes = 4 * embed.data.len();
            trunk.embed_tokens = QwenEmbed::F32(Embedding {
                weight: matrix_param(embed, device)?,
            });
            report.record("embed_tokens", bytes, store);
        }
        WeightStore::Nf4 => {
            let packed = Nf4Embedding::from_values(&embed.data, dims.vocab_size, dims.hidden_size);
            let bytes = packed.resident_bytes();
            trunk.embed_tokens = QwenEmbed::Nf4(packed);
            report.record("embed_tokens", bytes, store);
        }
    }
    for idx in 0..config.effective_layers() {
        let p = |suffix: &str| format!("model.language_model.layers.{idx}.{suffix}");
        let layer = &mut trunk.layers[idx];
        let input_norm = map.read_part(&p("input_layernorm.weight"), QwenPart::InputNorm, dims)?;
        report.record_f32("small_f32", 4 * input_norm.data.len());
        layer.input_norm.weight = vector_param(&input_norm, device)?;
        let post_norm = map.read_part(
            &p("post_attention_layernorm.weight"),
            QwenPart::PostNorm,
            dims,
        )?;
        report.record_f32("small_f32", 4 * post_norm.data.len());
        layer.post_norm.weight = vector_param(&post_norm, device)?;
        fill_mlp(&mut layer.mlp, &map, idx, dims, store, &mut report, device)?;
        match &mut layer.mixer {
            QwenMixer::Linear(head) => {
                fill_gated_delta_head(head, &map, idx, dims, store, &mut report, device)?
            }
            QwenMixer::Full(attn) => {
                fill_full_attention(attn, &map, idx, dims, store, &mut report, device)?
            }
        }
    }
    let final_norm = map.read_checked("model.language_model.norm.weight", &[dims.hidden_size])?;
    report.record_f32("small_f32", 4 * final_norm.data.len());
    trunk.final_norm.weight = vector_param(&final_norm, device)?;
    Ok((trunk, report))
}

/// Load the gated-DeltaNet head of decoder layer `layer_idx` from the real
/// BF16 safetensors shards under `dir`, f32-resident. Every tensor is
/// shape-checked against [`QwenPart::expected_shape`] and required finite
/// (a loud error naming the tensor, never a silent clamp).
pub fn load_gated_delta_head<B: Backend>(
    dir: &Path,
    layer_idx: usize,
    dims: &QwenArchDims,
    device: &B::Device,
) -> Result<GatedDeltaHead<B>> {
    ensure_linear_layer(layer_idx, dims)?;
    let map = ShardMap::open(dir)?;
    let mut report = Nf4ResidencyReport::default();
    let mut head = GatedDeltaHead::new(gated_delta_config(dims), dims.hidden_size, device);
    fill_gated_delta_head(
        &mut head,
        &map,
        layer_idx,
        dims,
        WeightStore::F32,
        &mut report,
        device,
    )?;
    Ok(head)
}

/// The NF4-RESIDENT sibling of [`load_gated_delta_head`]: every rank-2
/// weight is streamed (read f32 -> quantize packed -> store -> drop) and
/// the small tensors stay f32, per the residency policy. Returns the head
/// and its residency accounting.
pub fn load_gated_delta_head_nf4<B: Backend>(
    dir: &Path,
    layer_idx: usize,
    dims: &QwenArchDims,
    device: &B::Device,
) -> Result<(GatedDeltaHead<B>, Nf4ResidencyReport)> {
    ensure_linear_layer(layer_idx, dims)?;
    let map = ShardMap::open(dir)?;
    let mut report = Nf4ResidencyReport::default();
    let mut head = GatedDeltaHead::new(gated_delta_config(dims), dims.hidden_size, device);
    fill_gated_delta_head(
        &mut head,
        &map,
        layer_idx,
        dims,
        WeightStore::Nf4,
        &mut report,
        device,
    )?;
    Ok((head, report))
}

/// Load the full trunk f32-resident (tests and small models). A FULL
/// 64-layer f32 load is ~110 GB resident and belongs to
/// [`load_qwen_trunk_nf4`], not this loader.
pub fn load_qwen_trunk<B: Backend>(
    dir: &Path,
    config: &QwenTrunkConfig,
    device: &B::Device,
) -> Result<QwenTrunk<B>> {
    let (trunk, _report) = load_trunk_impl(dir, config, WeightStore::F32, device)?;
    Ok(trunk)
}

/// Load the full trunk NF4-RESIDENT (~14 GB for the 27B instead of ~111
/// GB f32): every rank-2 weight is streamed read -> quantize packed ->
/// store -> drop, so the load-time peak is the LARGEST single tensor (the
/// 5.1 GB f32 embed decode plus its packed form and the accumulated ~14
/// GB), never the whole f32 model. Rank-1 tensors and the conv weights
/// stay f32 (policy). Quantization is a pure function of the shard bytes,
/// so two loads produce byte-identical packed weights -- that determinism
/// is 3b's resume story: re-quantize the trunk, load the adapter
/// checkpoint, continue training.
///
/// The returned [`Nf4ResidencyReport`] accounts packed bytes, f32-kept
/// bytes, the per-tensor-class breakdown, and the estimated peak scratch;
/// it is what the memory budget gate reads.
pub fn load_qwen_trunk_nf4<B: Backend>(
    dir: &Path,
    config: &QwenTrunkConfig,
    device: &B::Device,
) -> Result<(QwenTrunk<B>, Nf4ResidencyReport)> {
    load_trunk_impl(dir, config, WeightStore::Nf4, device)
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

    use crate::tensor_ext::force_initialization;
    use burn::backend::NdArray;
    use std::sync::atomic::{AtomicU64, Ordering};

    type B = NdArray<f32>;

    /// Unique temp dirs for checkpoint fixtures (parallel tests share /tmp).
    static FIXTURE_SEQ: AtomicU64 = AtomicU64::new(0);

    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn flat3(t: Tensor<B, 3>) -> Vec<f32> {
        t.into_data().convert::<f32>().iter().collect()
    }

    /// Small architecture exercising BOTH layer kinds: layers 0-2 and 4
    /// linear, layer 3 full attention ((3+1) % 4 == 0).
    fn tiny_dims() -> QwenArchDims {
        QwenArchDims {
            num_layers: 5,
            hidden_size: 64,
            intermediate_size: 96,
            vocab_size: 128,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            linear_k_groups: 2,
            linear_v_heads: 4,
            linear_head_dim: 16,
            conv_kernel: 4,
            full_attention_interval: 4,
            attn_output_gate: true,
        }
    }

    fn tiny_config() -> QwenTrunkConfig {
        QwenTrunkConfig {
            dims: tiny_dims(),
            num_layers: None,
            rope_base: 10_000_000.0,
            eps: 1e-6,
        }
    }

    fn tiny_tokens() -> Tensor<B, 2, Int> {
        Tensor::<B, 2, Int>::from_ints([[3, 1, 4, 1, 5, 9]], &Default::default())
    }

    #[test]
    fn linear_weight_layout_is_in_times_out() {
        // Ground truth for the loader's transpose direction: Burn 0.21
        // `Linear` stores weight as [in, out] and computes y[o] = sum_i
        // x[i] * W[i][o]. If a future Burn flips the layout this test
        // flips, and the loader must be re-derived, not guessed.
        let device = Default::default();
        let mut lin: Linear<B> = LinearConfig::new(2, 3).with_bias(false).init(&device);
        lin.weight = Param::from_tensor(
            Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device).reshape([2, 3]),
        );
        let x = Tensor::<B, 3>::from_floats([[[1.0, -1.0]]], &device);
        let y = flat3(lin.forward(x));
        // y[o] = 1*W[0][o] - 1*W[1][o] = [1-4, 2-5, 3-6].
        assert!(close(&y, &[-3.0, -3.0, -3.0], 1e-6), "got {y:?}");
    }

    #[test]
    fn forward_is_finite_on_random_weights() {
        let device = Default::default();
        let hidden = 32;
        let head: GatedDeltaHead<B> =
            GatedDeltaHead::new(GatedDeltaConfig::tiny(), hidden, &device);
        force_initialization(&head);
        let x = Tensor::<B, 3>::random([1, 5, hidden], Distribution::Normal(0.0, 0.02), &device);
        let y = head.forward(x);
        assert_eq!(y.dims(), [1, 5, hidden]);
        let vals = flat3(y);
        assert!(vals.iter().all(|v| v.is_finite()), "non-finite outputs");
    }

    #[test]
    fn decode_step_matches_prefill_prefix() {
        // Token-by-token decode must reproduce the prefill forward at every
        // position: the conv window replays the same FIR on the same raw
        // rows and the recurrent step is the same function call, so the
        // tolerance is exact-zero.
        let device = Default::default();
        let hidden = 32;
        let head: GatedDeltaHead<B> =
            GatedDeltaHead::new(GatedDeltaConfig::tiny(), hidden, &device);
        force_initialization(&head);
        let n = 6;
        let x = Tensor::<B, 3>::random([1, n, hidden], Distribution::Normal(0.0, 0.02), &device);
        let prefill = head.forward(x.clone());

        let mut state = DeltaDecodeState::new(1, GatedDeltaConfig::tiny(), &device);
        let mut last = Tensor::<B, 2>::zeros([1, hidden], &device);
        for t in 0..n {
            let xt = x.clone().narrow(1, t, 1).reshape([1, hidden]);
            let out = head.step(xt, &mut state);
            if t == 3 || t == n - 1 {
                let want = flat3(prefill.clone().narrow(1, t, 1));
                let got: Vec<f32> = out
                    .clone()
                    .reshape([1, 1, hidden])
                    .into_data()
                    .convert::<f32>()
                    .iter()
                    .collect();
                assert!(close(&got, &want, 0.0), "step {t} diverged from prefill");
            }
            last = out;
        }
        assert_eq!(last.dims(), [1, hidden]);
    }

    #[test]
    fn repeat_adjacent_groups_correctly() {
        let device = Default::default();
        let x = Tensor::<B, 4>::from_floats([[[[1.0], [2.0]], [[3.0], [4.0]]]], &device);
        let y = repeat_adjacent(x, 1, 3);
        assert_eq!(y.dims(), [1, 6, 2, 1]);
        let vals: Vec<f32> = y.into_data().convert::<f32>().iter().collect();
        let want = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0];
        assert!(close(&vals, &want, 0.0), "got {vals:?}");
    }

    #[test]
    fn qwen_rms_norm_is_zero_centered() {
        // Zeros-init weight: y = x * rsqrt(mean(x^2) + eps) * (1 + 0), the
        // identity on rows that already sit at unit RMS. A nonzero weight
        // scales by exactly (1 + w).
        let device = Default::default();
        let norm: QwenRmsNorm<B> = QwenRmsNorm::new(4, 1e-6, &device);
        let x = Tensor::<B, 3>::from_floats([[[1.0, -1.0, 1.0, -1.0]]], &device);
        let y = flat3(norm.forward(x.clone()));
        assert!(close(&y, &[1.0, -1.0, 1.0, -1.0], 1e-5), "got {y:?}");

        let mut scaled: QwenRmsNorm<B> = QwenRmsNorm::new(4, 1e-6, &device);
        scaled.weight =
            Param::from_tensor(Tensor::<B, 1>::from_floats([0.5, 0.5, 0.5, 0.5], &device));
        let y = flat3(scaled.forward(x));
        assert!(close(&y, &[1.5, -1.5, 1.5, -1.5], 1e-5), "got {y:?}");
    }

    #[test]
    fn real_shard_layer_zero_loads_and_runs_finite() {
        // Same skip-if-absent pattern as qwen.rs: env override, workspace
        // default, note on stderr when the weights have not landed.
        let dir = std::env::var("QWEN_WEIGHTS_DIR")
            .unwrap_or_else(|_| "/srv/m-sdd/unifur/weights/qwen38-bf16".to_string());
        let shard1 = Path::new(&dir).join("model-00001-of-00012.safetensors");
        if !shard1.is_file() {
            eprintln!("skip: shard 1 not landed yet ({})", shard1.display());
            return;
        }
        let device = Default::default();
        let dims = QwenArchDims::qwen38();
        let head = load_gated_delta_head(Path::new(&dir), 0, &dims, &device)
            .unwrap_or_else(|e| panic!("load layer 0: {e:#}"));
        let x = Tensor::<B, 3>::random(
            [1, 4, dims.hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        let y = head.forward(x);
        assert_eq!(y.dims(), [1, 4, dims.hidden_size]);
        let vals = flat3(y);
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "non-finite outputs on real weights"
        );
    }

    #[test]
    fn embedding_weight_layout_is_row_gather() {
        // Ground truth for the embed fill (NO transpose): Burn 0.21
        // `Embedding` stores weight as [n_embedding, d_model] and gathers
        // whole rows by id, like PyTorch. If a future Burn changes the
        // layout this test flips, and the loader must be re-derived.
        let device = Default::default();
        let mut emb: Embedding<B> = EmbeddingConfig::new(3, 2).init(&device);
        emb.weight = Param::from_tensor(
            Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device).reshape([3, 2]),
        );
        let out = emb.forward(Tensor::<B, 2, Int>::from_ints([[2, 0]], &device));
        let vals = flat3(out);
        assert!(close(&vals, &[5.0, 6.0, 1.0, 2.0], 0.0), "got {vals:?}");
    }

    #[test]
    fn tiny_trunk_forward_is_finite() {
        let device = Default::default();
        let trunk = QwenTrunk::<B>::new(tiny_config(), &device);
        force_initialization(&trunk);
        let tokens = tiny_tokens();
        let logits = trunk.forward(tokens.clone());
        assert_eq!(logits.dims(), [1, 6, 128]);
        let vals = flat3(logits);
        assert!(vals.iter().all(|v| v.is_finite()), "non-finite logits");
        let hidden = flat3(trunk.forward_hidden(tokens));
        assert!(
            hidden.iter().all(|v| v.is_finite()),
            "non-finite hidden states"
        );
        let peak = hidden.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("tiny trunk random-init hidden peak |x| = {peak}");
        assert!(peak < 1e4, "random-init hidden scale insane: {peak}");
    }

    #[test]
    fn full_attention_step_matches_prefill() {
        // KV-cache decode must reproduce prefill at every position: the
        // same projections, QK norms, rotary angles (absolute offset),
        // masked softmax row and output gate, so the tolerance is zero.
        let device = Default::default();
        let dims = tiny_dims();
        let attn = QwenFullAttention::<B>::new(&dims, 10_000_000.0, 1e-6, &device);
        force_initialization(&attn);
        let n = 6;
        let x = Tensor::<B, 3>::random(
            [1, n, dims.hidden_size],
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        let prefill = attn.forward(x.clone());
        let mut state = LayerState::<B>::new();
        for t in 0..n {
            let xt = x.clone().narrow(1, t, 1).reshape([1, dims.hidden_size]);
            let out = attn.step(xt, &mut state);
            if t == 3 || t == n - 1 {
                let want = flat3(prefill.clone().narrow(1, t, 1));
                let got: Vec<f32> = out
                    .reshape([1, 1, dims.hidden_size])
                    .into_data()
                    .convert::<f32>()
                    .iter()
                    .collect();
                assert!(close(&got, &want, 0.0), "step {t} diverged from prefill");
            }
        }
    }

    #[test]
    fn trunk_step_matches_prefill() {
        // Mixed trunk (4 linear + 1 full layer): token-by-token decode vs
        // prefill logits, positions 3 and last, exact zero tolerance.
        let device = Default::default();
        let trunk = QwenTrunk::<B>::new(tiny_config(), &device);
        force_initialization(&trunk);
        let tokens = tiny_tokens();
        let n = tokens.dims()[1];
        let logits = trunk.forward(tokens.clone());
        let mut state = trunk.new_state(1, &device);
        for t in 0..n {
            let out = trunk
                .step(tokens.clone().narrow(1, t, 1), &mut state)
                .unwrap_or_else(|e| panic!("step {t}: {e:#}"));
            if t == 3 || t == n - 1 {
                let want = flat3(logits.clone().narrow(1, t, 1));
                let got: Vec<f32> = out
                    .reshape([1, 1, 128])
                    .into_data()
                    .convert::<f32>()
                    .iter()
                    .collect();
                assert!(close(&got, &want, 0.0), "step {t} diverged from prefill");
            }
        }
    }

    #[test]
    fn checkpoint_round_trip_is_bit_identical() {
        // A saved+reloaded trunk must reproduce its outputs bit for bit:
        // the content-addressed record carries every parameter losslessly.
        let device = Default::default();
        let trunk = QwenTrunk::<B>::new(tiny_config(), &device);
        force_initialization(&trunk);
        let tokens = tiny_tokens();
        let want = flat3(trunk.forward(tokens.clone()));
        let dir = std::env::temp_dir().join(format!(
            "dblocks-qwennet-ckpt-{}-{}",
            std::process::id(),
            FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let path = crate::checkpoint::save_content_addressed(trunk, &dir, "qwtrunk")
            .unwrap_or_else(|e| panic!("save: {e:#}"));
        let fresh = QwenTrunk::<B>::new(tiny_config(), &device);
        let loaded = crate::checkpoint::load(fresh, &path, &device)
            .unwrap_or_else(|e| panic!("load: {e:#}"));
        let got = flat3(loaded.forward(tokens));
        assert!(
            close(&got, &want, 0.0),
            "outputs drifted across the round trip"
        );
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("clean {}: {e}", dir.display()));
    }

    #[test]
    fn rope_base_is_1e7_not_repo_default() {
        // Guard against silently wiring `hybrid::ROTARY_BASE` (1e4): Qwen3.8
        // full attention rotates at base 1e7 over a 64-wide partial window.
        let config = QwenTrunkConfig::qwen38();
        assert_eq!(config.rope_base, 10_000_000.0);
        let device = Default::default();
        let attn = QwenFullAttention::<B>::new(&tiny_dims(), config.rope_base, 1e-6, &device);
        assert_eq!(attn.rope_base(), 10_000_000.0);
        assert_ne!(attn.rope_base(), crate::hybrid::ROTARY_BASE);
        assert_eq!(attn.rotary_dim(), 4, "0.25 of head_dim 16");
        let full =
            QwenFullAttention::<B>::new(&QwenArchDims::qwen38(), config.rope_base, 1e-6, &device);
        assert_eq!(full.rotary_dim(), 64, "0.25 of head_dim 256");
    }

    #[test]
    fn gqa_adjacent_repeat_rank4() {
        // The GQA mapping on [b, kv, n, d]: query head hh reads kv head
        // hh / (q_heads / kv_heads) -- adjacent, NOT Burn's tiling repeat.
        let device = Default::default();
        let kv = Tensor::<B, 4>::from_floats([[[[1.0, 2.0]], [[3.0, 4.0]]]], &device); // [1,2,1,2]
        let repeated = repeat_adjacent(kv, 1, 3);
        assert_eq!(repeated.dims(), [1, 6, 1, 2]);
        let vals: Vec<f32> = repeated.into_data().convert::<f32>().iter().collect();
        let want = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0];
        assert!(close(&vals, &want, 0.0), "got {vals:?}");
    }

    #[test]
    fn real_weight_reduced_trunk_forward() {
        // Loads embed + layers 0..N + final norm + lm_head from the real
        // BF16 shards (~16 GB transient) and runs a short prompt. Layer
        // count from `QWEN_TRUNK_TEST_LAYERS` (default 4).
        let dir = std::env::var("QWEN_WEIGHTS_DIR")
            .unwrap_or_else(|_| "/srv/m-sdd/unifur/weights/qwen38-bf16".to_string());
        let shard1 = Path::new(&dir).join("model-00001-of-00012.safetensors");
        if !shard1.is_file() {
            eprintln!("skip: shard 1 not landed yet ({})", shard1.display());
            return;
        }
        let layers: usize = std::env::var("QWEN_TRUNK_TEST_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let device = Default::default();
        let config = QwenTrunkConfig {
            num_layers: Some(layers),
            ..QwenTrunkConfig::qwen38()
        };
        let start = std::time::Instant::now();
        let trunk = load_qwen_trunk(Path::new(&dir), &config, &device)
            .unwrap_or_else(|e| panic!("load trunk: {e:#}"));
        eprintln!("loaded {layers} layers in {:?}", start.elapsed());
        let tokens = Tensor::<B, 2, Int>::from_ints([[151643, 9707, 1879, 330]], &device);
        let hidden = flat3(trunk.forward_hidden(tokens.clone()));
        assert!(
            hidden.iter().all(|v| v.is_finite()),
            "non-finite hidden states"
        );
        let hpeak = hidden.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let start = std::time::Instant::now();
        let logits = flat3(trunk.forward(tokens));
        eprintln!("lm_head forward in {:?}", start.elapsed());
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
        let peak = logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("real 4-layer trunk: hidden peak |x| = {hpeak}, max |logit| = {peak}");
        assert!(
            peak < 200.0,
            "logit scale insane for pretrained weights: {peak}"
        );
    }

    #[test]
    fn nf4_linear_forward_equals_dequantized_linear() {
        // "dequantize, then matmul", exactly: an Nf4Linear and an f32
        // Linear built from its dequantized weight produce bit-identical
        // outputs (same values, same burn linear op), rank-2 and rank-3.
        let device = Default::default();
        let values: Vec<f32> =
            Tensor::<B, 2>::random([8, 6], Distribution::Normal(0.0, 1.0), &device)
                .into_data()
                .convert::<f32>()
                .iter()
                .collect();
        let packed = PackedNf4Tensor::quantize(&values).with_double_quantization();
        let deq = packed.dequantize();
        let nf4 = Nf4Linear::<B>::from_packed(packed, 8, 6, None);
        let f32lin = Linear::<B> {
            weight: Param::from_tensor(
                Tensor::<B, 1>::from_floats(deq.as_slice(), &device).reshape([8, 6]),
            ),
            bias: None,
        };
        let x2 = Tensor::<B, 2>::random([3, 8], Distribution::Uniform(-1.0, 1.0), &device);
        let a: Vec<f32> = nf4
            .forward(x2.clone())
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        let b: Vec<f32> = f32lin
            .forward(x2)
            .into_data()
            .convert::<f32>()
            .iter()
            .collect();
        assert!(close(&a, &b, 0.0), "rank-2 outputs diverged");
        let x3 = Tensor::<B, 3>::random([1, 4, 8], Distribution::Uniform(-1.0, 1.0), &device);
        assert!(
            close(
                &flat3(nf4.forward(x3.clone())),
                &flat3(f32lin.forward(x3)),
                0.0
            ),
            "rank-3 outputs diverged"
        );
    }

    /// Deterministic fixture values (LCG): same dims -> same bytes.
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = u16::try_from(self.0 >> 48).expect("16 bits fit u16");
            (f32::from(bits) / f32::from(u16::MAX) - 0.5) * 0.1
        }
    }

    /// Write a synthetic one-shard safetensors dir holding every tensor a
    /// trunk of `dims` needs (F32 dtype, import.rs fixture pattern).
    fn write_tiny_fixture(dims: &QwenArchDims) -> std::path::PathBuf {
        let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
        let h = dims.hidden_size;
        let inter = dims.intermediate_size;
        let conv_c = 2 * dims.key_dim() + dims.value_dim();
        specs.push((
            "model.language_model.embed_tokens.weight".to_string(),
            vec![dims.vocab_size, h],
        ));
        specs.push(("model.language_model.norm.weight".to_string(), vec![h]));
        specs.push(("lm_head.weight".to_string(), vec![dims.vocab_size, h]));
        for idx in 0..dims.num_layers {
            let prefix = format!("model.language_model.layers.{idx}");
            if dims.is_full_attention(idx) {
                let qw = dims.num_q_heads * dims.head_dim;
                let kvw = dims.num_kv_heads * dims.head_dim;
                specs.push((format!("{prefix}.self_attn.q_proj.weight"), vec![2 * qw, h]));
                specs.push((format!("{prefix}.self_attn.k_proj.weight"), vec![kvw, h]));
                specs.push((format!("{prefix}.self_attn.v_proj.weight"), vec![kvw, h]));
                specs.push((format!("{prefix}.self_attn.o_proj.weight"), vec![h, qw]));
                specs.push((
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![dims.head_dim],
                ));
                specs.push((
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![dims.head_dim],
                ));
            } else {
                specs.push((
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    vec![conv_c, h],
                ));
                specs.push((
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    vec![dims.value_dim(), h],
                ));
                specs.push((
                    format!("{prefix}.linear_attn.in_proj_b.weight"),
                    vec![dims.linear_v_heads, h],
                ));
                specs.push((
                    format!("{prefix}.linear_attn.in_proj_a.weight"),
                    vec![dims.linear_v_heads, h],
                ));
                specs.push((
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![conv_c, 1, dims.conv_kernel],
                ));
                specs.push((
                    format!("{prefix}.linear_attn.norm.weight"),
                    vec![dims.linear_head_dim],
                ));
                specs.push((
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![h, dims.value_dim()],
                ));
                specs.push((
                    format!("{prefix}.linear_attn.dt_bias"),
                    vec![dims.linear_v_heads],
                ));
                specs.push((
                    format!("{prefix}.linear_attn.A_log"),
                    vec![dims.linear_v_heads],
                ));
            }
            specs.push((format!("{prefix}.mlp.gate_proj.weight"), vec![inter, h]));
            specs.push((format!("{prefix}.mlp.up_proj.weight"), vec![inter, h]));
            specs.push((format!("{prefix}.mlp.down_proj.weight"), vec![h, inter]));
            specs.push((format!("{prefix}.input_layernorm.weight"), vec![h]));
            specs.push((format!("{prefix}.post_attention_layernorm.weight"), vec![h]));
        }
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        let mut payload = Vec::new();
        for (name, shape) in &specs {
            let count: usize = shape.iter().product();
            let mut bytes = Vec::with_capacity(4 * count);
            for _ in 0..count {
                bytes.extend_from_slice(&rng.next_f32().to_le_bytes());
            }
            let end = offset + u64::try_from(bytes.len()).expect("fixture small");
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
            );
            offset = end;
            payload.extend_from_slice(&bytes);
        }
        let dir = std::env::temp_dir().join(format!(
            "dblocks-qwennet-fixture-{}-{}",
            std::process::id(),
            FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create fixture dir");
        let header_str = serde_json::to_string(&header).expect("header json");
        use std::io::Write as _;
        let mut file = std::fs::File::create(dir.join("model-00001-of-00001.safetensors"))
            .expect("create fixture");
        file.write_all(
            &u64::try_from(header_str.len())
                .expect("small header")
                .to_le_bytes(),
        )
        .expect("write header len");
        file.write_all(header_str.as_bytes()).expect("write header");
        file.write_all(&payload).expect("write payload");
        dir
    }

    /// All (f32-weight, nf4-weight) value pairs of two same-config trunks.
    fn collect_weight_pairs(
        f32_trunk: &QwenTrunk<B>,
        nf4_trunk: &QwenTrunk<B>,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let linear_values = |l: &QwenLinear<B>| -> Vec<f32> {
            let t = match l {
                QwenLinear::F32(lin) => lin.weight.val(),
                QwenLinear::Nf4(n) => n.dequantized_weight(device),
                // Adapters are zero-initialized at attach, so the effective
                // weight is the base until an optimizer step moves it.
                QwenLinear::Qlora(q) => q.base.dequantized_weight(device),
            };
            t.into_data().convert::<f32>().iter().collect()
        };
        let embed_values = |e: &QwenEmbed<B>| -> Vec<f32> {
            let t = match e {
                QwenEmbed::F32(em) => em.weight.val(),
                QwenEmbed::Nf4(em) => em.dequantized_weight::<B>(device),
            };
            t.into_data().convert::<f32>().iter().collect()
        };
        let mut pairs = vec![
            (
                embed_values(&f32_trunk.embed_tokens),
                embed_values(&nf4_trunk.embed_tokens),
            ),
            (
                linear_values(&f32_trunk.lm_head),
                linear_values(&nf4_trunk.lm_head),
            ),
        ];
        for (fa, na) in f32_trunk.layers.iter().zip(&nf4_trunk.layers) {
            for (x, y) in [
                (&fa.mlp.gate_proj, &na.mlp.gate_proj),
                (&fa.mlp.up_proj, &na.mlp.up_proj),
                (&fa.mlp.down_proj, &na.mlp.down_proj),
            ] {
                pairs.push((linear_values(x), linear_values(y)));
            }
            match (&fa.mixer, &na.mixer) {
                (QwenMixer::Linear(fh), QwenMixer::Linear(nh)) => {
                    for (x, y) in [
                        (&fh.in_proj_qkv, &nh.in_proj_qkv),
                        (&fh.in_proj_z, &nh.in_proj_z),
                        (&fh.in_proj_b, &nh.in_proj_b),
                        (&fh.in_proj_a, &nh.in_proj_a),
                        (&fh.out_proj, &nh.out_proj),
                    ] {
                        pairs.push((linear_values(x), linear_values(y)));
                    }
                }
                (QwenMixer::Full(fa2), QwenMixer::Full(na2)) => {
                    for (x, y) in [
                        (&fa2.q_proj, &na2.q_proj),
                        (&fa2.k_proj, &na2.k_proj),
                        (&fa2.v_proj, &na2.v_proj),
                        (&fa2.o_proj, &na2.o_proj),
                    ] {
                        pairs.push((linear_values(x), linear_values(y)));
                    }
                }
                _ => panic!("mixer kinds differ between trunks"),
            }
        }
        pairs
    }

    /// Largest relative Frobenius error over the weight pairs.
    fn max_rel_frobenius(pairs: &[(Vec<f32>, Vec<f32>)]) -> f64 {
        let mut eps_max = 0.0f64;
        for (w, wq) in pairs {
            let num: f64 = w
                .iter()
                .zip(wq)
                .map(|(a, b)| f64::from(a - b) * f64::from(a - b))
                .sum::<f64>()
                .sqrt();
            let den: f64 = w
                .iter()
                .map(|a| f64::from(*a) * f64::from(*a))
                .sum::<f64>()
                .sqrt();
            eps_max = eps_max.max(num / den.max(1e-12));
        }
        eps_max
    }

    #[test]
    fn tiny_trunk_nf4_matches_f32_within_measured_bound() {
        // End-to-end through the synthetic fixture: the NF4 trunk's logits
        // must sit within a bound computed FROM THE WEIGHTS (per-tensor
        // relative Frobenius quant error, amplified (1 + eps) per linear),
        // never from a hand-picked number.
        let device = Default::default();
        let config = tiny_config();
        let dir = write_tiny_fixture(&config.dims);
        let trunk_f32 =
            load_qwen_trunk(&dir, &config, &device).unwrap_or_else(|e| panic!("f32 load: {e:#}"));
        let (trunk_nf4, report) = load_qwen_trunk_nf4(&dir, &config, &device)
            .unwrap_or_else(|e| panic!("nf4 load: {e:#}"));
        let tokens = tiny_tokens();
        let logits_f32 = flat3(trunk_f32.forward(tokens.clone()));
        let logits_nf4 = flat3(trunk_nf4.forward(tokens));
        let gap = logits_f32
            .iter()
            .zip(&logits_nf4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let pairs = collect_weight_pairs(&trunk_f32, &trunk_nf4, &device);
        let eps_max = max_rel_frobenius(&pairs);
        let peak = logits_f32.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let n = i32::try_from(pairs.len()).expect("few weights");
        let bound = ((1.0 + eps_max).powi(n) - 1.0) * f64::from(peak);
        eprintln!(
            "tiny trunk NF4: gap {gap:.6}, eps_max {eps_max:.4}, bound {bound:.4}, resident {} bytes ({} packed + {} f32)",
            report.total_resident_bytes(),
            report.packed_bytes,
            report.f32_kept_bytes
        );
        assert!(
            gap.is_finite() && f64::from(gap) <= bound,
            "gap {gap} exceeds weight-derived bound {bound}"
        );
        assert!(
            report.packed_bytes > 0 && report.f32_kept_bytes > 0,
            "report accounts both storages"
        );
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("clean {}: {e}", dir.display()));
    }

    #[test]
    fn real_shard_nf4_layer_loads_within_budget() {
        // skip-if-absent
        let dir = std::env::var("QWEN_WEIGHTS_DIR")
            .unwrap_or_else(|_| "/srv/m-sdd/unifur/weights/qwen38-bf16".to_string());
        let shard1 = Path::new(&dir).join("model-00001-of-00012.safetensors");
        if !shard1.is_file() {
            eprintln!("skip: shard 1 not landed yet ({})", shard1.display());
            return;
        }
        let device = Default::default();
        let dims = QwenArchDims::qwen38();
        let (nf4_head, report) = load_gated_delta_head_nf4(Path::new(&dir), 0, &dims, &device)
            .unwrap_or_else(|e| panic!("nf4 load: {e:#}"));
        // Pin the residency accounting against the formula (computed, not
        // quoted): five packed projections plus the f32-kept smalls.
        let est = PackedNf4Tensor::estimated_resident_bytes;
        let (h, kd, vd, nv) = (
            dims.hidden_size,
            dims.key_dim(),
            dims.value_dim(),
            dims.num_v_heads(),
        );
        let packed_want = est((2 * kd + vd) * h, true)
            + est(vd * h, true)
            + 2 * est(nv * h, true)
            + est(h * vd, true);
        let f32_want = 4 * ((2 * kd + vd) * dims.conv_kernel + dims.linear_head_dim + 2 * nv);
        assert_eq!(
            report.packed_bytes, packed_want,
            "packed accounting drifted"
        );
        assert_eq!(
            report.f32_kept_bytes, f32_want,
            "f32-kept accounting drifted"
        );
        let mib = report.total_resident_bytes() as f64 / 1_048_576.0; // audit-allow: display-only byte->MiB ratio
        eprintln!(
            "layer 0 NF4 resident: {} bytes ({mib:.1} MiB)",
            report.total_resident_bytes()
        );

        // Forward agreement, TWO ways.
        let f32_head = load_gated_delta_head(Path::new(&dir), 0, &dims, &device)
            .unwrap_or_else(|e| panic!("f32 load: {e:#}"));
        let x = Tensor::<B, 3>::random([1, 4, h], Distribution::Normal(0.0, 0.02), &device);
        let get = |l: &QwenLinear<B>| -> Vec<f32> {
            let t = match l {
                QwenLinear::F32(lin) => lin.weight.val(),
                QwenLinear::Nf4(n) => n.dequantized_weight(&device),
                QwenLinear::Qlora(q) => q.base.dequantized_weight(&device),
            };
            t.into_data().convert::<f32>().iter().collect()
        };
        let as_f32 = |l: &QwenLinear<B>| -> QwenLinear<B> {
            match l {
                QwenLinear::Nf4(n) => QwenLinear::F32(Linear {
                    weight: Param::from_tensor(n.dequantized_weight(&device)),
                    bias: None,
                }),
                QwenLinear::Qlora(q) => QwenLinear::F32(Linear {
                    weight: Param::from_tensor(q.base.dequantized_weight(&device)),
                    bias: None,
                }),
                QwenLinear::F32(_) => panic!("expected an NF4-resident weight"),
            }
        };
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = [
            (&f32_head.in_proj_qkv, &nf4_head.in_proj_qkv),
            (&f32_head.in_proj_z, &nf4_head.in_proj_z),
            (&f32_head.in_proj_b, &nf4_head.in_proj_b),
            (&f32_head.in_proj_a, &nf4_head.in_proj_a),
            (&f32_head.out_proj, &nf4_head.out_proj),
        ]
        .iter()
        .map(|(a, b)| (get(a), get(b)))
        .collect();

        // 1. RIGOROUS: an f32 head whose weights are the NF4 head's
        //    DEQUANTIZED weights must produce bit-identical outputs (same
        //    values, same ops). This is what proves the enum plumbing fills
        //    every real weight correctly.
        let mut ref_head = f32_head.clone();
        ref_head.in_proj_qkv = as_f32(&nf4_head.in_proj_qkv);
        ref_head.in_proj_z = as_f32(&nf4_head.in_proj_z);
        ref_head.in_proj_b = as_f32(&nf4_head.in_proj_b);
        ref_head.in_proj_a = as_f32(&nf4_head.in_proj_a);
        ref_head.out_proj = as_f32(&nf4_head.out_proj);
        let out_ref = flat3(ref_head.forward(x.clone()));
        let out_nf4 = flat3(nf4_head.forward(x.clone()));
        assert!(
            close(&out_nf4, &out_ref, 0.0),
            "NF4 head != dequantized f32 head"
        );

        // 2. Observed quant-noise gap vs the true f32 head, with a measured
        //    magnitude sanity: per-element max weight error times the input
        //    L1 norm, summed over the layer's linears (heuristic scale,
        //    documented as such -- the proof is the bit-identity above).
        let out_f32 = flat3(f32_head.forward(x.clone()));
        let gap = out_f32
            .iter()
            .zip(&out_nf4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let peak = out_f32.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let x_vals = flat3(x);
        let x_l1 = x_vals.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a + b);
        let bound: f64 = pairs
            .iter()
            .map(|(w, wq)| {
                w.iter()
                    .zip(wq)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            })
            .map(|delta| f64::from(delta) * f64::from(x_l1))
            .sum();
        let eps_max = max_rel_frobenius(&pairs);
        eprintln!(
            "layer 0 NF4 vs f32: observed gap {gap:.6} (peak |out| {peak:.6}, weight eps_max {eps_max:.4}, sanity bound {bound:.4})"
        );
        assert!(
            gap.is_finite() && f64::from(gap) <= bound,
            "gap {gap} exceeds sanity bound {bound}"
        );
    }

    #[test]
    fn nf4_quantization_is_deterministic() {
        // skip-if-absent. Quantization is a pure function of the shard
        // bytes: quantizing the same real tensor twice must give identical
        // packed bytes. THAT is 3b's resume story -- re-quantize the trunk
        // (deterministic) + load the adapter checkpoint.
        let dir = std::env::var("QWEN_WEIGHTS_DIR")
            .unwrap_or_else(|_| "/srv/m-sdd/unifur/weights/qwen38-bf16".to_string());
        let shard1 = Path::new(&dir).join("model-00001-of-00012.safetensors");
        if !shard1.is_file() {
            eprintln!("skip: shard 1 not landed yet ({})", shard1.display());
            return;
        }
        let map = ShardMap::open(Path::new(&dir)).unwrap_or_else(|e| panic!("shard map: {e:#}"));
        let dims = QwenArchDims::qwen38();
        let tensor = map
            .read_part(
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                QwenPart::LinearB,
                &dims,
            )
            .unwrap_or_else(|e| panic!("read: {e:#}"));
        let q1 = PackedNf4Tensor::quantize(&tensor.data).with_double_quantization();
        let q2 = PackedNf4Tensor::quantize(&tensor.data).with_double_quantization();
        assert!(
            q1 == q2,
            "packed bytes differ between identical quantizations"
        );
        assert_eq!(q1.dequantize(), q2.dequantize());
        assert_eq!(q1.len(), 48 * 5120);
    }
}
