//! ViT-DiT backbone: a HuggingFace-style ViT whose CLS token is replaced by
//! the noisy class embedding, with DiT-style adaLN-zero timestep
//! conditioning on every transformer layer.
//!
//! Ported from `vit.py` in the reference repository (`ViTDiT*` classes),
//! restricted to the `time_conditioning = true` configuration used by
//! DiffusionBlocks.

use crate::hybrid::{AttentionMode, AttentionSchedule, LayerState};
use crate::routing::RoutingState;
use crate::tensor_ext::{exact_gelu, l2_normalize_rows, silu};
use burn::{
    module::{Module, Param},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{activation::softmax, backend::Backend, Distribution, Int, Tensor},
};
use serde::{Deserialize, Serialize};

/// Hyperparameters of the ViT-DiT backbone (mirrors `ViTDiTConfig`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViTDiTConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub ffn_kind: FfnKind,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub initializer_range: f64,
    /// Number of classes (size of the label embedding table).
    pub num_labels: usize,
    /// Hidden size of the conditioning vector fed to adaLN (`hidden/6`).
    pub cond_hidden_size: usize,
    /// Dim of the sinusoidal timestep embedding.
    pub frequency_embedding_size: usize,
    /// Replace some layers' dense MLPs with mixture-of-experts blocks
    /// (roadmap 6.5). `None` keeps the trunk fully dense.
    pub moe: Option<MoeTrunkConfig>,
    /// Replace some layers' dense MLPs with *boxes* of specialized micro
    /// experts (roadmap 18.7). Takes precedence over [`Self::moe`], which it
    /// generalizes; setting both is a configuration error the CLI rejects.
    pub mosme: Option<MosmeTrunkConfig>,
    /// Mask attention so position `i` cannot see `i + 1` (roadmap 19.3).
    ///
    /// `false` for the image path: patch tokens have no ordering to respect,
    /// and the noisy class embedding in slot 0 must be visible to all of them.
    /// `true` for language modeling, where seeing the future would let the
    /// model copy the answer.
    pub causal: bool,
    /// One attention mode per layer (roadmap Phase 25); `None` is dense
    /// everywhere, which is the pre-Phase-25 trunk bit for bit. Anything but
    /// dense needs `causal`.
    pub attention: Option<AttentionSchedule>,
    /// Rotate queries and keys by their absolute position (roadmap 25.2).
    /// Off for the image path, whose positions are a learned table.
    pub rotary: bool,
    /// Width of the per-token routing state carried through the layers
    /// (roadmap 25.4); `0` builds none and leaves every router input as it
    /// was.
    pub routing_state: usize,
    /// Grouped-query attention: keys and values use this many heads while
    /// queries keep `num_attention_heads` (Qwen3 style GQA). `None` is full
    /// multi-head attention, which is what every checkpoint to date holds.
    /// Must divide `num_attention_heads` when set.
    pub num_kv_heads: Option<usize>,
    /// Fraction of each head dimension the rotary embedding covers (Qwen3-Next
    /// style partial rotary, e.g. `0.25`); `1.0` is the full rotation this
    /// crate always applied. Ignored unless `rotary` is set.
    #[serde(default = "default_rotary_fraction")]
    pub rotary_fraction: f64,
    /// Qwen-style gated attention: the merged heads are scaled by
    /// `1 + tanh(g(x))` with a zero-initialized gate, so an untrained gated
    /// layer is exactly the ungated trunk.
    #[serde(default)]
    pub gated_attention: bool,
    /// QK-Norm (Qwen3, GLM-4.5, LLaMA-4 Scout): per-head RMSNorm on queries
    /// and keys before rotary. `false` is the historical trunk bit for bit.
    #[serde(default)]
    pub qk_norm: bool,
    /// `Layer` is what every checkpoint to date holds; `Rms` is the Qwen
    /// style RMSNorm (one weight, no centering). The default is `Layer`, so
    /// existing checkpoints load unchanged.
    #[serde(default)]
    pub norm_kind: NormKind,
}

/// The rotary cover of a fresh trunk: the full head dimension.
pub(crate) fn default_rotary_fraction() -> f64 {
    1.0
}

/// Which normalization a trunk layer (and the trunk's final norm) uses.
///
/// Carried on the config rather than inferred from shapes so a checkpoint
/// built with one kind fails loudly on a trunk built with the other instead
/// of being silently reinterpreted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormKind {
    /// `LayerNorm`: weight plus bias per hidden unit.
    #[default]
    Layer,
    /// RMSNorm: one scale per hidden unit, no centering (Qwen style).
    Rms,
}

impl NormKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Layer => "layernorm",
            Self::Rms => "rmsnorm",
        }
    }
}

/// Where and how mixture-of-experts layers enter the trunk.
///
/// Only *some* layers become sparse: alternating dense and sparse layers is
/// the standard Switch/GLaM placement, and it keeps the dense path available
/// for the features every expert needs. `every_n_layers = 2` replaces every
/// second layer's MLP.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MoeTrunkConfig {
    pub num_experts: usize,
    pub top_k: usize,
    /// Replace the MLP of every `every_n_layers`-th layer (1 = all of them).
    pub every_n_layers: usize,
    /// Router z-loss weight (ST-MoE); `0.0` disables it exactly.
    pub z_level: f64,
    /// Loss-free selection biases on the routers (roadmap 23.5).
    pub balance_bias: bool,
}

impl Default for MoeTrunkConfig {
    fn default() -> Self {
        Self {
            num_experts: 4,
            top_k: 1,
            every_n_layers: 2,
            z_level: 1e-3,
            balance_bias: false,
        }
    }
}

/// Where hierarchical expert boxes enter the trunk (roadmap 18.7).
///
/// Deliberately not `Copy`: the spec carries box and expert names, which is
/// the whole point of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MosmeTrunkConfig {
    pub spec: crate::expert_index::MosmeSpec,
    /// Replace the feed-forward of every `every_n_layers`-th layer.
    pub every_n_layers: usize,
    /// Loss-free selection biases on every router (roadmap 23.5).
    pub balance_bias: bool,
}

impl MosmeTrunkConfig {
    pub fn new(spec: crate::expert_index::MosmeSpec) -> Self {
        Self {
            spec,
            every_n_layers: 2,
            balance_bias: false,
        }
    }

    pub fn with_every_n_layers(mut self, n: usize) -> Self {
        self.every_n_layers = n;
        self
    }

    pub fn with_balance_bias(mut self, enabled: bool) -> Self {
        self.balance_bias = enabled;
        self
    }

    /// Whether layer `idx` should be hierarchical.
    pub fn applies_to(&self, idx: usize) -> bool {
        let n = self.every_n_layers.max(1);
        idx % n == n - 1
    }

    pub fn num_hierarchical_layers(&self, num_layers: usize) -> usize {
        (0..num_layers).filter(|i| self.applies_to(*i)).count()
    }
}

impl MoeTrunkConfig {
    /// Whether layer `idx` should be sparse.
    pub fn applies_to(&self, idx: usize) -> bool {
        let n = self.every_n_layers.max(1);
        idx % n == n - 1
    }

    /// Sparse layers among `num_layers`.
    pub fn num_sparse_layers(&self, num_layers: usize) -> usize {
        (0..num_layers).filter(|i| self.applies_to(*i)).count()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FfnKind {
    #[default]
    Gelu,
    SwiGlu,
}

impl FfnKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gelu => "gelu",
            Self::SwiGlu => "swiglu",
        }
    }

    pub(crate) fn validate(self, has_experts: bool) -> anyhow::Result<()> {
        anyhow::ensure!(
            self == Self::Gelu || !has_experts,
            "SwiGLU is supported only for dense trunks; ffn_kind=swiglu cannot be combined with MoE or MoSME"
        );
        Ok(())
    }
}

/// RMSNorm: one scale per hidden unit, no centering (Qwen style).
///
/// The weight starts at exactly one, so a fresh RMSNorm passes its input
/// through up to the RMS rescaling; unlike `LayerNorm` there is no bias term,
/// which is why a checkpoint records a single `[h]` tensor here against
/// `LayerNorm`'s two.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    epsilon: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(hidden: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::<B, 1>::ones([hidden], device)),
            epsilon,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = x.dims()[2];
        let ms = x.clone().powf_scalar(2.0).mean_dim(2);
        let inv = (ms + self.epsilon).powf_scalar(-0.5);
        x * inv * self.weight.val().reshape([1, 1, h])
    }

    /// Normalize over the last dim of a rank-4 tensor (per attention head):
    /// reshape through the rank-3 path so there is exactly one RMSNorm
    /// implementation for a divergence to hide in.
    pub fn forward_4d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, n, d] = x.dims();
        self.forward(x.reshape([b * h * n, 1, d]))
            .reshape([b, h, n, d])
    }
}

/// Either trunk normalization, as an enum so the record round-trips through
/// Burn's serialization and a checkpoint carries which one it was built with.
///
/// `Layer` is what every checkpoint to date holds; selecting `Rms` on an old
/// checkpoint fails on the record shapes rather than silently reinterpreting
/// the weights.
#[derive(Module, Debug)]
pub enum TrunkNorm<B: Backend> {
    Layer(LayerNorm<B>),
    Rms(RmsNorm<B>),
}

impl<B: Backend> TrunkNorm<B> {
    pub fn new(kind: NormKind, hidden: usize, epsilon: f64, device: &B::Device) -> Self {
        match kind {
            NormKind::Layer => Self::Layer(
                LayerNormConfig::new(hidden)
                    .with_epsilon(epsilon)
                    .init(device),
            ),
            NormKind::Rms => Self::Rms(RmsNorm::new(hidden, epsilon, device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Layer(norm) => norm.forward(x),
            Self::Rms(norm) => norm.forward(x),
        }
    }

    pub fn kind(&self) -> NormKind {
        match self {
            Self::Layer(_) => NormKind::Layer,
            Self::Rms(_) => NormKind::Rms,
        }
    }
}

impl ViTDiTConfig {
    pub fn with_ffn_kind(mut self, kind: FfnKind) -> Self {
        self.ffn_kind = kind;
        self
    }

    pub fn validate_ffn(&self) -> anyhow::Result<()> {
        self.ffn_kind
            .validate(self.moe.is_some() || self.mosme.is_some())
    }

    /// CIFAR preset (image size 32): patch 4, 12 layers, hidden 128, 4 heads.
    pub fn cifar(num_labels: usize) -> Self {
        Self::with_image_size(32, num_labels)
    }

    /// Tiny ImageNet preset (image size 64): patch 4, 12 layers, hidden 768,
    /// 12 heads.
    pub fn tiny_imagenet(num_labels: usize) -> Self {
        let mut cfg = Self::with_image_size(64, num_labels);
        cfg.hidden_size = 768;
        cfg.intermediate_size = 768 * 4;
        cfg.num_attention_heads = 12;
        cfg.cond_hidden_size = 768 / 6;
        cfg
    }

    /// Shared preset logic from `load_vit`.
    pub fn with_image_size(image_size: usize, num_labels: usize) -> Self {
        assert!(
            image_size == 32 || image_size == 64,
            "invalid image size: {image_size} (expected 32 or 64)"
        );
        Self {
            image_size,
            patch_size: 4,
            in_channels: 3,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 12,
            num_attention_heads: 4,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            initializer_range: 0.02,
            num_labels,
            cond_hidden_size: 128 / 6,
            frequency_embedding_size: 256,
            moe: None,
            mosme: None,
            causal: false,
            attention: None,
            rotary: false,
            routing_state: 0,
            num_kv_heads: None,
            rotary_fraction: default_rotary_fraction(),
            gated_attention: false,
            qk_norm: false,
            norm_kind: NormKind::Layer,
            ffn_kind: FfnKind::Gelu,
        }
    }

    /// Enable mixture-of-experts layers in the trunk.
    pub fn with_moe(mut self, moe: MoeTrunkConfig) -> Self {
        self.moe = Some(moe);
        self
    }

    /// Assign an attention mode to every layer (roadmap Phase 25).
    pub fn with_attention(mut self, schedule: AttentionSchedule) -> Self {
        self.attention = Some(schedule);
        self
    }

    /// Rotate queries and keys by absolute position (roadmap 25.2).
    pub fn with_rotary(mut self, rotary: bool) -> Self {
        self.rotary = rotary;
        self
    }

    /// Carry a per-token routing state of this width through the layers
    /// (roadmap 25.4).
    pub fn with_routing_state(mut self, size: usize) -> Self {
        self.routing_state = size;
        self
    }

    /// Grouped-query attention with this many key/value heads (Qwen3 style).
    pub fn with_kv_heads(mut self, kv_heads: usize) -> Self {
        self.num_kv_heads = Some(kv_heads);
        self
    }

    /// Rotary cover as a fraction of the head dimension (Qwen3-Next style
    /// partial rotary, e.g. `0.25`); `1.0` is the full rotation.
    pub fn with_rotary_fraction(mut self, fraction: f64) -> Self {
        self.rotary_fraction = fraction;
        self
    }

    /// Qwen-style gated attention: the merged heads are scaled by
    /// `1 + tanh(g(x))` with a zero-initialized gate.
    pub fn with_gated_attention(mut self, gated: bool) -> Self {
        self.gated_attention = gated;
        self
    }

    /// QK-Norm on queries and keys before rotary (Qwen3/GLM-4.5/LLaMA-4).
    pub fn with_qk_norm(mut self, enabled: bool) -> Self {
        self.qk_norm = enabled;
        self
    }

    /// LayerNorm (the default, every checkpoint to date) or RMSNorm.
    pub fn with_norm_kind(mut self, kind: NormKind) -> Self {
        self.norm_kind = kind;
        self
    }

    /// The validated key/value head count: full MHA when unset.
    pub fn kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_attention_heads)
    }

    /// Cross-check the Qwen-style attention knobs together, since a bad
    /// combination (more KV heads than query heads, an unrepresentable rotary
    /// cover) would otherwise fail deep inside layer construction.
    pub fn validate_attention(&self) -> anyhow::Result<()> {
        let q = self.num_attention_heads;
        let kv = self.kv_heads();
        anyhow::ensure!(
            q > 0 && kv > 0 && kv <= q,
            "KV heads ({kv}) must divide the query heads ({q}) from below"
        );
        anyhow::ensure!(
            q % kv == 0,
            "query heads ({q}) must be a multiple of KV heads ({kv})"
        );
        anyhow::ensure!(
            self.hidden_size % q == 0,
            "hidden size must be divisible by head count"
        );
        anyhow::ensure!(
            self.rotary_fraction > 0.0 && self.rotary_fraction <= 1.0,
            "rotary fraction must be in (0, 1], got {}",
            self.rotary_fraction
        );
        Ok(())
    }

    /// The attention mode of layer `idx`.
    pub fn attention_mode(&self, idx: usize) -> AttentionMode {
        self.attention
            .as_ref()
            .map_or(AttentionMode::Dense, |s| s.mode(idx))
    }

    /// Enable hierarchical expert boxes in the trunk.
    pub fn with_mosme(mut self, mosme: MosmeTrunkConfig) -> Self {
        self.mosme = Some(mosme);
        self
    }

    /// Mask attention causally (roadmap 19.3).
    pub fn causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// A deliberately small configuration for tests, examples and smoke runs:
    /// 4 patches, hidden 32, 4 layers, dropout off.
    ///
    /// Dropout is disabled so results are deterministic, which every test that
    /// compares two forward passes depends on.
    pub fn tiny(num_labels: usize) -> Self {
        Self {
            image_size: 32,
            patch_size: 16,
            in_channels: 3,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            num_labels,
            cond_hidden_size: 8,
            frequency_embedding_size: 16,
            moe: None,
            mosme: None,
            causal: false,
            attention: None,
            rotary: false,
            routing_state: 0,
            num_kv_heads: None,
            rotary_fraction: default_rotary_fraction(),
            gated_attention: false,
            qk_norm: false,
            norm_kind: NormKind::Layer,
            ffn_kind: FfnKind::Gelu,
        }
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Number of tokens: noisy class-embedding token + patches.
    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// `SiLU(Linear(...))` modulation network (`AdaLN` in the reference).
#[derive(Module, Debug)]
pub struct AdaLN<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> AdaLN<B> {
    pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(in_features, out_features)
                .with_bias(true)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        silu(self.linear.forward(x))
    }
}

/// DiT `TimestepEmbedder`: sinusoidal features followed by an MLP.
#[derive(Module, Debug)]
pub struct TimestepEmbedder<B: Backend> {
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    frequency_embedding_size: usize,
}

impl<B: Backend> TimestepEmbedder<B> {
    pub fn new(
        cond_hidden_size: usize,
        frequency_embedding_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            linear_1: LinearConfig::new(frequency_embedding_size, cond_hidden_size)
                .with_bias(true)
                .init(device),
            linear_2: LinearConfig::new(cond_hidden_size, cond_hidden_size)
                .with_bias(true)
                .init(device),
            frequency_embedding_size,
        }
    }

    pub(crate) fn validate_specialist_shape(
        &self,
        cond: usize,
        frequency: usize,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.frequency_embedding_size == frequency
                && crate::mosme::module_shapes(self)
                    == vec![
                        vec![frequency, cond],
                        vec![cond],
                        vec![cond, cond],
                        vec![cond]
                    ],
            "timestep embedder shape mismatch"
        );
        Ok(())
    }

    /// Sinusoidal timestep embedding (`timestep_embedding(t, dim)`):
    /// `[batch] -> [batch, frequency_embedding_size]`.
    pub fn timestep_embedding(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let device = t.device();
        let half = self.frequency_embedding_size / 2;
        let exponent = -(10000f64.ln()) / half as f64;
        let freqs = Tensor::<B, 1, Int>::arange(0..half as i64, &device)
            .float()
            .mul_scalar(exponent)
            .exp();
        // Outer product via broadcasting: [b, 1] * [1, half] -> [b, half].
        let args = t.unsqueeze_dim::<2>(1) * freqs.unsqueeze_dim::<2>(0);
        Tensor::cat(vec![args.clone().cos(), args.sin()], 1)
    }

    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let t_freq = self.timestep_embedding(t);
        let h = silu(self.linear_1.forward(t_freq));
        self.linear_2.forward(h)
    }
}

/// Patch + position embeddings with the noisy class embedding taking the CLS
/// slot (`ViTDiTEmbeddings` with `time_conditioning=true`).
#[derive(Module, Debug)]
pub struct ViTDiTEmbeddings<B: Backend> {
    patch_embed: Conv2d<B>,
    label_embeddings: Embedding<B>,
    position_embeddings: Param<Tensor<B, 3>>,
    dropout: Dropout,
    hidden_size: usize,
    num_patches: usize,
}

impl<B: Backend> ViTDiTEmbeddings<B> {
    pub fn new(config: &ViTDiTConfig, device: &B::Device) -> Self {
        let patch_embed = Conv2dConfig::new(
            [config.in_channels, config.hidden_size],
            [config.patch_size, config.patch_size],
        )
        .with_stride([config.patch_size, config.patch_size])
        .init(device);

        let label_embeddings =
            EmbeddingConfig::new(config.num_labels, config.hidden_size).init(device);

        // trunc-normal-ish position table: [1, 1 + P, H]
        let pos = Tensor::<B, 2>::random(
            [1, config.seq_len() * config.hidden_size],
            Distribution::Normal(0.0, config.initializer_range),
            device,
        )
        .reshape([1, config.seq_len(), config.hidden_size]);

        Self {
            patch_embed,
            label_embeddings,
            position_embeddings: Param::from_tensor(pos),
            dropout: DropoutConfig::new(config.hidden_dropout_prob).init(),
            hidden_size: config.hidden_size,
            num_patches: config.num_patches(),
        }
    }

    /// `pixel_values`: `[b, c, h, w]`, `noisy_embeds`: `[b, hidden]`.
    /// Returns `[b, 1 + num_patches, hidden]`.
    pub fn forward(&self, pixel_values: Tensor<B, 4>, noisy_embeds: Tensor<B, 2>) -> Tensor<B, 3> {
        let b = pixel_values.dims()[0];

        // [b, c, h, w] -> conv -> [b, hidden, h', w'] -> flatten patches ->
        // transpose to [b, h'*w', hidden]
        let patches = self
            .patch_embed
            .forward(pixel_values)
            .reshape([b, self.hidden_size, self.num_patches])
            .swap_dims(1, 2);

        let cls_tokens = noisy_embeds.unsqueeze_dim::<3>(1); // [b, 1, hidden]
        let embeddings = Tensor::cat(vec![cls_tokens, patches], 1);
        let embeddings = embeddings + self.position_embeddings.val();
        self.dropout.forward(embeddings)
    }

    /// Label ids -> L2-normalized embeddings (`get_embeds`).
    pub fn embed_labels(&self, labels: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let b = labels.dims()[0];
        let embeds = self.label_embeddings.forward(labels.reshape([b, 1]));
        l2_normalize_rows(embeds.squeeze_dim::<2>(1))
    }
}

/// Floor on the linear-attention normalizer.
const LINEAR_EPS: f32 = 1e-6;

/// Multi-head self-attention (`ViTAttention`, biased qkv/out projections),
/// attending in one of the modes of [`crate::hybrid`] (roadmap Phase 25).
///
/// The mode is a plain value from the config, like `causal`, not a parameter;
/// the only parameter a mode adds is the `[2]` dense/linear mixing logit of
/// [`AttentionMode::Learned`], zero-initialized so an untrained mixture is
/// even.
#[derive(Module, Debug)]
struct Attention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    dense: Linear<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_dropout: Dropout,
    output_dropout: Dropout,
    #[module(skip)]
    mode: AttentionMode,
    rotary: bool,
    /// Columns of each head the rotary embedding covers; `== head_dim` for
    /// the full rotation (`rotary_fraction == 1.0`).
    rotary_dim: usize,
    /// Qwen-style output gate: `out * (1 + tanh(gate(x)))`, zero-initialized
    /// so an untrained gated layer is exactly the ungated trunk. `None` when
    /// the trunk was built without `--gated-attention`.
    gate: Option<Linear<B>>,
    /// QK-Norm (Qwen3, GLM-4.5, LLaMA-4 Scout): per-head RMSNorm on queries
    /// and keys before rotary, bounding attention scores by the head dim no
    /// matter how large the projections drift (the Kimi K2 postmortem is
    /// logits past 100). `None` when built without `--qk-norm`.
    q_norm: Option<RmsNorm<B>>,
    k_norm: Option<RmsNorm<B>>,
    /// Mixing logits for [`AttentionMode::Learned`]. Zero-initialized, so a
    /// fresh learned layer mixes its two branches evenly. Not an `Option`:
    /// every attention layer carries the two logits, and a mode other than
    /// `Learned` simply never reads them. That removes the "is the mix present?"
    /// question from the forward pass, which is the only thing that ever asked.
    mix: Param<Tensor<B, 1>>,
}

impl<B: Backend> Attention<B> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        attn_dropout: f64,
        output_dropout: f64,
        mode: AttentionMode,
        rotary: bool,
        rotary_dim: usize,
        gated: bool,
        qk_norm: bool,
        device: &B::Device,
    ) -> Self {
        assert!(
            num_heads > 0
                && num_kv_heads > 0
                && num_kv_heads <= num_heads
                && num_heads % num_kv_heads == 0,
            "KV heads ({num_kv_heads}) must divide the query heads ({num_heads}) from below"
        );
        let head_dim = hidden_size / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut query = LinearConfig::new(hidden_size, hidden_size)
            .with_bias(true)
            .init(device);
        // Fold the 1/sqrt(head_dim) attention scale into the Q projection so
        // the forward pass needs no extra elementwise multiply per layer.
        // Numerically identical to scaling q after projection.
        let scale = head_dim as f64;
        let inv_sqrt = (1.0 / scale.sqrt()) as f32;
        // set_require_grad(false): the scaled tensor must be an untracked
        // root before it can become a fresh parameter (detach() alone would
        // keep require_grad and make the product a non-leaf).
        let weight = query.weight.val().set_require_grad(false) * inv_sqrt;
        query.weight = Param::from_tensor(weight);
        if let Some(bias) = query.bias {
            let bias = bias.val().set_require_grad(false) * inv_sqrt;
            query.bias = Some(Param::from_tensor(bias));
        }

        // Zero logits, so `softmax` starts at an even 50/50 mixture; every mode
        // carries them and only `Learned` reads them.
        let mix = Param::from_tensor(Tensor::<B, 1>::zeros([2], device));
        if rotary {
            assert!(
                head_dim % 2 == 0,
                "rotary positions need an even head dimension, got {head_dim}"
            );
            assert!(
                rotary_dim <= head_dim && rotary_dim % 2 == 0,
                "rotary cover {rotary_dim} must be an even number of columns of {head_dim}"
            );
        }
        // Zero-initialized (Initializer::Zeros): tanh(0) is 0, so the gate
        // scales by exactly 1.0 until training moves it.
        let gate = gated.then(|| {
            LinearConfig::new(hidden_size, hidden_size)
                .with_bias(true)
                .with_initializer(burn::module::Initializer::Zeros)
                .init(device)
        });
        // Fresh RMSNorm carries weight exactly one: normalizing queries and
        // keys that are already unit-RMS is a no-op, so enabling the flag on
        // a normalized trunk starts close to it, and the scores can never
        // exceed the head dim whatever the projections learn.
        let (q_norm, k_norm) = match qk_norm {
            true => {
                let eps = 1e-6;
                (
                    Some(RmsNorm::new(head_dim, eps, device)),
                    Some(RmsNorm::new(head_dim, eps, device)),
                )
            }
            false => (None, None),
        };

        Self {
            query,
            key: LinearConfig::new(hidden_size, kv_dim)
                .with_bias(true)
                .init(device),
            value: LinearConfig::new(hidden_size, kv_dim)
                .with_bias(true)
                .init(device),
            dense: LinearConfig::new(hidden_size, hidden_size)
                .with_bias(true)
                .init(device),
            num_heads,
            num_kv_heads,
            head_dim,
            attn_dropout: DropoutConfig::new(attn_dropout).init(),
            output_dropout: DropoutConfig::new(output_dropout).init(),
            mode,
            rotary,
            rotary_dim,
            gate,
            q_norm,
            k_norm,
            mix,
        }
    }

    fn split_q(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        x.reshape([b, n, self.num_heads, self.head_dim])
            .swap_dims(1, 2) // [b, heads, n, head_dim]
    }

    fn split_kv(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        x.reshape([b, n, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2) // [b, kv_heads, n, head_dim]
    }

    /// Repeat each key/value head for its query group. Tiling maps head `h`
    /// to KV head `h % kv`; the from-scratch trunk is symmetric under query
    /// head permutation at init, so this grouping is equivalent to any other
    /// fixed partition up to a permutation the projections absorb.
    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let groups = self.num_heads / self.num_kv_heads;
        if groups <= 1 {
            return x;
        }
        x.repeat_dim(1, groups)
    }

    fn merge_heads(&self, ctx: Tensor<B, 4>, gate_input: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let ctx = ctx.swap_dims(1, 2); // [b, n, heads, head_dim]
        let [b, n, _, _] = ctx.dims();
        let ctx = ctx.reshape([b, n, self.num_heads * self.head_dim]);
        let out = self.output_dropout.forward(self.dense.forward(ctx));
        match (&self.gate, gate_input) {
            (Some(gate), Some(xin)) => {
                // Bounded in (0, 2), exactly 1.0 at init: a stabilizer the
                // trunk can lean on or ignore, never a scale shock.
                let scale = gate.forward(xin).tanh() + 1.0;
                out * scale
            }
            _ => out,
        }
    }

    /// The mode this layer attends in.
    pub(crate) fn mode(&self) -> AttentionMode {
        self.mode
    }

    /// Per-head RMSNorm on queries and keys when the trunk was built with
    /// `--qk-norm`, pass-through otherwise. Runs before rotary, matching
    /// Qwen3/GLM-4.5/LLaMA-4: with unit-RMS queries and keys every attention
    /// score is bounded by the head dim, so drifting projections cannot blow
    /// logits past 100 the way Kimi K2's postmortem describes.
    fn norm_qk(&self, q: Tensor<B, 4>, k: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        match (&self.q_norm, &self.k_norm) {
            (Some(qn), Some(kn)) => (qn.forward_4d(q), kn.forward_4d(k)),
            _ => (q, k),
        }
    }

    /// Rotate queries and keys by their absolute positions, when configured.
    /// Only the first `rotary_dim` columns turn; the rest pass through, which
    /// is what partial rotary (Qwen3-Next: 25% of dims) means here.
    fn rotate(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if !self.rotary {
            return (q, k);
        }
        (
            crate::hybrid::apply_rotary_partial(
                q,
                offset,
                crate::hybrid::ROTARY_BASE,
                self.rotary_dim,
            ),
            crate::hybrid::apply_rotary_partial(
                k,
                offset,
                crate::hybrid::ROTARY_BASE,
                self.rotary_dim,
            ),
        )
    }

    /// Largest (causally valid) attention score over `normed` `[b, n, h]`:
    /// the Kimi K2 early-warning metric. Loss and gradient norms miss logit
    /// blowup until it spikes the run; the max score shows it coming while it
    /// is still a drift. `None` for pure linear attention, which keeps no
    /// scores (its state is the `(S, z)` pair, not a score matrix).
    ///
    /// Scores only: no value projection, no softmax, no output map — roughly
    /// a third of an attention forward for the number training stability is
    /// actually read in. Returns the single-element score-maximum tensor;
    /// callers in a `FloatElem = f32` context read the scalar out, which
    /// keeps this method usable from the untyped `Backend` impl block.
    pub(crate) fn max_logit(
        &self,
        normed: Tensor<B, 3>,
        causal: bool,
        offset: usize,
    ) -> Option<Tensor<B, 1>> {
        if self.mode == AttentionMode::Linear {
            return None;
        }
        assert!(
            causal || self.mode == AttentionMode::Dense,
            "only dense attention is defined bidirectionally; this layer is {:?}",
            self.mode
        );
        let n = normed.dims()[1];
        let q = self.split_q(self.query.forward(normed.clone()));
        let k = self.split_kv(self.key.forward(normed));
        let (q, k) = self.norm_qk(q, k);
        let (q, k) = self.rotate(q, k, offset);
        let k = self.repeat_kv(k);
        let device = q.device();
        let mask = match self.mode {
            AttentionMode::Dense => causal.then(|| causal_mask::<B>(n, &device)),
            AttentionMode::Sliding { window } => Some(crate::hybrid::attention_mask::<B>(
                n,
                n,
                0,
                0,
                Some(window),
                &device,
            )),
            AttentionMode::Retrieval { .. } | AttentionMode::Learned => Some(
                crate::hybrid::attention_mask::<B>(n, n, 0, 0, None, &device),
            ),
            AttentionMode::Linear => return None,
        };
        let mut scores = q.matmul(k.swap_dims(2, 3));
        if let Some(mask) = mask {
            scores = scores + mask;
        }
        Some(scores.max())
    }

    /// Softmax attention of `q` `[b, heads, m, d]` over `k`/`v`
    /// `[b, heads, total, d]`, with an additive mask and, for retrieval, a
    /// per-row top-k restriction.
    fn softmax_attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<Tensor<B, 4>>,
        top_k: Option<usize>,
    ) -> Tensor<B, 4> {
        // The 1/sqrt(head_dim) scale is folded into the Q projection weights.
        let mut scores = q.matmul(k.swap_dims(2, 3)); // [b, heads, m, total]
        if let Some(mask) = mask {
            // Additive -inf, so exp() gives exactly 0 there and a masked key
            // gets no weight whatsoever. Masking the *scores* rather than the
            // probabilities is what makes that exact: zeroing after the
            // softmax would leave the denominator polluted by the future.
            scores = scores + mask;
        }
        if let Some(k) = top_k {
            scores = crate::hybrid::keep_top_k(scores, k);
        }
        let probs = self.attn_dropout.forward(softmax(scores, 3));
        probs.matmul(v)
    }

    /// Softmax weights of the learned dense/linear mixture, `[2]`.
    fn mix_weights(&self) -> Tensor<B, 1> {
        softmax(self.mix.val(), 0)
    }

    fn mix_pair(&self, dense: Tensor<B, 4>, linear: Tensor<B, 4>) -> Tensor<B, 4> {
        let w = self.mix_weights();
        let w_dense = w.clone().narrow(0, 0, 1).reshape([1, 1, 1, 1]);
        let w_linear = w.narrow(0, 1, 1).reshape([1, 1, 1, 1]);
        dense * w_dense + linear * w_linear
    }

    /// Attention over a whole sequence whose first position is `offset`.
    fn forward(&self, x: Tensor<B, 3>, causal: bool, offset: usize) -> Tensor<B, 3> {
        use crate::hybrid::{attention_mask, feature_map, linear_attention};

        let n = x.dims()[1];
        let gate_input = self.gate.is_some().then(|| x.clone());
        let q = self.split_q(self.query.forward(x.clone()));
        let k = self.split_kv(self.key.forward(x.clone()));
        let v = self.split_kv(self.value.forward(x));
        let (q, k) = self.norm_qk(q, k);
        let (q, k) = self.rotate(q, k, offset);
        // Grouped-query heads read the repeated keys/values; with full MHA
        // the repeat is a no-op returning its input.
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);
        let device = q.device();
        let mode = self.mode;
        assert!(
            causal || mode == AttentionMode::Dense,
            "only dense attention is defined bidirectionally; this layer is {:?}",
            mode
        );

        let ctx = match mode {
            AttentionMode::Dense => {
                let mask = causal.then(|| causal_mask::<B>(n, &device));
                self.softmax_attend(q, k, v, mask, None)
            }
            AttentionMode::Sliding { window } => {
                let mask = attention_mask::<B>(n, n, 0, 0, Some(window), &device);
                self.softmax_attend(q, k, v, Some(mask), None)
            }
            AttentionMode::Retrieval { top_k } => {
                let mask = attention_mask::<B>(n, n, 0, 0, None, &device);
                self.softmax_attend(q, k, v, Some(mask), Some(top_k))
            }
            AttentionMode::Linear => {
                let (ctx, _) =
                    linear_attention(feature_map(q), feature_map(k), v, None, LINEAR_EPS);
                ctx
            }
            AttentionMode::Learned => {
                let mask = attention_mask::<B>(n, n, 0, 0, None, &device);
                let dense = self.softmax_attend(q.clone(), k.clone(), v.clone(), Some(mask), None);
                let (linear, _) =
                    linear_attention(feature_map(q), feature_map(k), v, None, LINEAR_EPS);
                self.mix_pair(dense, linear)
            }
        };
        self.merge_heads(ctx, gate_input)
    }

    /// Causal attention over `x` (the *new* positions only), reusing and
    /// extending the layer's `state`.
    ///
    /// Generation without a state recomputes every previous position's keys
    /// and values on every step: `O(n^2)` work per token instead of `O(n)`.
    /// Nothing about the result changes — those tensors are a pure function
    /// of tokens that have already been committed — which is why
    /// `lm/kv_cache_matches_full_recompute` can demand exact agreement rather
    /// than a tolerance. A sliding layer keeps only the last `window - 1`
    /// positions and a linear layer only its `(S, z)` state, so their
    /// footprint is bounded whatever the sequence length.
    ///
    /// Only meaningful causally: with bidirectional attention an earlier
    /// position's output depends on later ones, so nothing is reusable.
    fn forward_state(&self, x: Tensor<B, 3>, state: &mut LayerState<B>) -> Tensor<B, 3> {
        use crate::hybrid::{attention_mask, feature_map, linear_attention};

        let m = x.dims()[1];
        let offset = state.positions;
        let gate_input = self.gate.is_some().then(|| x.clone());
        let q = self.split_q(self.query.forward(x.clone()));
        let k_new = self.split_kv(self.key.forward(x.clone()));
        let v_new = self.split_kv(self.value.forward(x));
        let (q, k_new) = self.norm_qk(q, k_new);
        let (q, k_new) = self.rotate(q, k_new, offset);
        let device = q.device();
        // The cache keeps the narrow KV heads (that is the whole saving);
        // every read below repeats them for the query groups first.
        let repeat = |t: Tensor<B, 4>| self.repeat_kv(t);

        let ctx = match self.mode {
            AttentionMode::Dense | AttentionMode::Retrieval { .. } => {
                let top_k = match self.mode {
                    AttentionMode::Retrieval { top_k } => Some(top_k),
                    _ => None,
                };
                let (k, v) = state.push_kv(k_new, v_new, None);
                let total = k.dims()[2];
                // Query j sits at absolute position `offset + j`, so it may
                // attend to any key up to that index. The mask is rectangular,
                // not triangular: the cached prefix is entirely in the past.
                let mask =
                    attention_mask::<B>(m, total, offset, state.first_key_position, None, &device);
                self.softmax_attend(q, repeat(k), repeat(v), Some(mask), top_k)
            }
            AttentionMode::Sliding { window } => {
                let (k, v) = state.push_kv(k_new, v_new, None);
                let total = k.dims()[2];
                let mask = attention_mask::<B>(
                    m,
                    total,
                    offset,
                    state.first_key_position,
                    Some(window),
                    &device,
                );
                let ctx = self.softmax_attend(q, repeat(k), repeat(v), Some(mask), None);
                // Only the last `window - 1` positions can be read by any
                // future query; the rest is forgotten, position included.
                state.trim(window.saturating_sub(1));
                ctx
            }
            AttentionMode::Linear => {
                let (ctx, next) = linear_attention(
                    feature_map(q),
                    feature_map(repeat(k_new)),
                    repeat(v_new),
                    state.linear.take(),
                    LINEAR_EPS,
                );
                state.linear = Some(next);
                ctx
            }
            AttentionMode::Learned => {
                let (k, v) = state.push_kv(k_new.clone(), v_new.clone(), None);
                let total = k.dims()[2];
                let mask =
                    attention_mask::<B>(m, total, offset, state.first_key_position, None, &device);
                let dense = self.softmax_attend(q.clone(), repeat(k), repeat(v), Some(mask), None);
                let (linear, next) = linear_attention(
                    feature_map(q),
                    feature_map(repeat(k_new)),
                    repeat(v_new),
                    state.linear.take(),
                    LINEAR_EPS,
                );
                state.linear = Some(next);
                self.mix_pair(dense, linear)
            }
        };
        state.positions += m;
        self.merge_heads(ctx, gate_input)
    }
}

/// Per-layer decode-time state (roadmap 25.3). Keys and values for a dense,
/// retrieval or sliding layer; the recurrent `(S, z)` for a linear one; both
/// for a learned mixture. Kept under the Phase 19 name so callers read the
/// same.
pub type LayerKvCache<B> = LayerState<B>;

/// Additive causal mask `[1, 1, n, n]`: `0` on and below the diagonal,
/// `-inf` above it.
///
/// Built on the host and broadcast over batch and heads. `n` is the sequence
/// length, which for the image path is fixed by the patch geometry and for the
/// language path is the context window. The general form -- an offset, a
/// window, cached keys -- is [`crate::hybrid::attention_mask`]; this is its
/// `m = total = n` case.
pub(crate) fn causal_mask<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 4> {
    crate::hybrid::attention_mask::<B>(n, n, 0, 0, None, device)
}

/// MLP block: `Linear -> exact GELU -> Linear` (`ViTIntermediate` +
/// `ViTOutput`).
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc_in: Linear<B>,
    fc_out: Linear<B>,
    output_dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    fn new(hidden_size: usize, intermediate_size: usize, p_drop: f64, device: &B::Device) -> Self {
        Self {
            fc_in: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(true)
                .init(device),
            fc_out: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(true)
                .init(device),
            output_dropout: DropoutConfig::new(p_drop).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = exact_gelu(self.fc_in.forward(x));
        self.output_dropout.forward(self.fc_out.forward(h))
    }
}

#[derive(Module, Debug)]
struct SwiGluMlp<B: Backend> {
    fc_in: Linear<B>,
    fc_gate: Linear<B>,
    fc_out: Linear<B>,
    output_dropout: Dropout,
}

impl<B: Backend> SwiGluMlp<B> {
    fn new(hidden_size: usize, intermediate_size: usize, p_drop: f64, device: &B::Device) -> Self {
        Self {
            fc_in: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(true)
                .init(device),
            fc_gate: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(true)
                .init(device),
            fc_out: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(true)
                .init(device),
            output_dropout: DropoutConfig::new(p_drop).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.fc_gate.forward(x.clone());
        let [b, n, width] = gate.dims();
        let gate = silu(gate.reshape([b * n, width])).reshape([b, n, width]);
        let h = gate * self.fc_in.forward(x);
        self.output_dropout.forward(self.fc_out.forward(h))
    }
}

/// A layer's position-wise feed-forward: either a dense MLP or a sparse
/// mixture of experts (roadmap 6.5).
///
/// Modelled as an enum rather than a boxed trait so the record round-trips
/// through Burn's serialization unchanged and a checkpoint carries its own
/// dense/sparse layout.
// The variants cannot be boxed: Burn's `Module` is not implemented for
// `Box<T>`, so a module enum has to hold them inline.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum FeedForward<B: Backend> {
    Dense(Mlp<B>),
    Sparse(crate::moe::MoELayer<B>),
    /// Boxes of specialized micro experts, routed two-level
    /// (see [`crate::mosme`]).
    Hierarchical(crate::mosme::MosmeFeedForward<B>),
    DenseSwiGlu(SwiGluMlp<B>),
}

/// The auxiliary losses a sparse layer produces.
///
/// The two are carried **separately** all the way to the training loop because
/// they are weighted differently and must be: the balance term is a routing
/// regularizer that a schedule may legitimately decay once every expert is
/// alive, while the z-loss is a numerical *stabilizer* that must not decay —
/// it matters most late, when the logits have had time to drift.
///
/// Folding them into one scalar also multiplied the z-loss by
/// `moe_aux_weight`, so a configured `z_level` of `1e-3` was reaching the
/// objective at `1e-5`. Keeping them apart is what makes `z_level` mean what
/// ST-MoE means by it: a coefficient on the total loss.
#[derive(Debug, Clone)]
pub struct RouterAux<B: Backend> {
    /// Load-balancing loss, unweighted.
    pub balance: Tensor<B, 1>,
    /// Router z-loss, already multiplied by its configured `z_level`.
    pub z: Tensor<B, 1>,
    /// One entry per sparse layer executed, in execution order: the load,
    /// probability mass and routing entropy that the scalar losses summarize
    /// away (roadmap 23.6). Still on the device; see
    /// [`crate::moe::LayerRouting::to_host`].
    pub layers: Vec<crate::moe::LayerRouting<B>>,
}

impl<B: Backend> RouterAux<B> {
    /// Sum two spans' losses and keep both spans' per-layer routing, in order.
    pub fn combine(mut self, other: Self) -> Self {
        self.layers.extend(other.layers);
        Self {
            balance: self.balance + other.balance,
            z: self.z + other.z,
            layers: self.layers,
        }
    }

    /// Per-layer routing statistics, synced to the host, with each layer's
    /// top-1 agreement with the routed layer before it (roadmap 25.6).
    pub fn to_host(&self) -> Vec<crate::moe::RoutingStats> {
        let mut stats: Vec<crate::moe::RoutingStats> = self
            .layers
            .iter()
            .map(crate::moe::LayerRouting::to_host)
            .collect();
        let top1: Vec<Vec<i64>> = self
            .layers
            .iter()
            .map(|l| crate::routing::top1_to_host(&l.top1))
            .collect();
        for i in 1..stats.len() {
            stats[i].agreement = crate::routing::layer_agreement(
                &top1[i - 1],
                &top1[i],
                self.layers[i - 1].experts,
                self.layers[i].experts,
            );
        }
        stats
    }
}

/// Mutable access to one sparse feed-forward, whichever kind it is.
pub(crate) enum SparseLayerMut<'a, B: Backend> {
    Flat(&'a mut crate::moe::MoELayer<B>),
    Hierarchical(&'a mut crate::mosme::MosmeFeedForward<B>),
}

impl<B: Backend> SparseLayerMut<'_, B> {
    /// Attach zero selection biases where missing (roadmap 23.5).
    pub(crate) fn ensure_balance_bias(&mut self) {
        match self {
            Self::Flat(layer) => layer.ensure_balance_bias(),
            Self::Hierarchical(layer) => layer.ensure_balance_bias(),
        }
    }

    /// Nudge the selection biases against `load` over this layer's global
    /// expert index (roadmap 23.5).
    pub(crate) fn nudge_balance_bias(&mut self, load: &[f32], rate: f32) {
        match self {
            Self::Flat(layer) => layer.nudge_balance_bias(load, rate),
            Self::Hierarchical(layer) => layer.nudge_balance_bias(load, rate),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    /// Returns the transformed states and, for sparse layers, the auxiliary
    /// losses that must be added to the training objective. `state` is the
    /// per-token routing state of roadmap 25.4, appended to the router input
    /// when the layer was built with one.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
        state: &Tensor<B, 3>,
        specialist: Option<(usize, usize)>,
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>) {
        let [b, n, _] = x.dims();
        match self {
            Self::Dense(mlp) => (mlp.forward(x), None),
            Self::DenseSwiGlu(mlp) => (mlp.forward(x), None),
            Self::Sparse(moe) => {
                let out = moe.forward_with_state(x, conditioning.clone(), state);
                let z = out.z_loss.mul_scalar(moe.z_level() as f32);
                (
                    out.output,
                    Some(RouterAux {
                        balance: out.balance,
                        z,
                        layers: vec![out.routing],
                    }),
                )
            }
            Self::Hierarchical(mosme) => {
                if let Some((bi, ei)) = specialist {
                    return (mosme.forward_specialist(x, bi, ei), None);
                }
                let out = mosme.forward_with_state(x, conditioning.clone(), state);
                let z = out
                    .balance
                    .z_loss
                    .clone()
                    .mul_scalar(mosme.z_level() as f32);
                let routing = out.gates.layer_routing().with_sequence(b, n);
                (
                    out.output,
                    Some(RouterAux {
                        balance: out.balance.total.clone(),
                        z,
                        layers: vec![routing],
                    }),
                )
            }
        }
    }
}

/// What flows *between* the layers of one pass besides the hidden states
/// (roadmap Phase 25): the per-token routing state, and the absolute position
/// of the pass's first token for rotary layers.
#[derive(Debug, Clone)]
pub(crate) struct LayerCarry<B: Backend> {
    /// `[b, n, size]` after the last layer that updated it; `None` before the
    /// first, and always `None` in a trunk built without a routing state.
    pub routing_state: Option<Tensor<B, 3>>,
    /// Absolute position of the first token of this pass.
    pub offset: usize,
    pub specialist: Option<(usize, usize)>,
}

impl<B: Backend> Default for LayerCarry<B> {
    fn default() -> Self {
        Self {
            routing_state: None,
            offset: 0,
            specialist: None,
        }
    }
}

impl<B: Backend> LayerCarry<B> {
    pub(crate) fn at(offset: usize) -> Self {
        Self {
            routing_state: None,
            offset,
            specialist: None,
        }
    }
}

/// One transformer layer with adaLN-zero conditioning (`ViTDiTLayer`).
///
/// Visible to the crate so the language-model trunk in [`crate::lm`] can reuse
/// it rather than duplicating the layer: the whole point of the MoSME and MoE
/// work is that it applies to any trunk, and a second copy of this type would
/// quietly diverge from the first.
#[derive(Module, Debug)]
pub(crate) struct DbLayer<B: Backend> {
    attention: Attention<B>,
    mlp: FeedForward<B>,
    layernorm_before: TrunkNorm<B>,
    layernorm_after: TrunkNorm<B>,
    ada_ln: AdaLN<B>,
    /// Whether this layer attends causally. Set from
    /// [`ViTDiTConfig::causal`]; `false` for the image path, whose tokens are
    /// patches with no ordering to respect.
    causal: bool,
    /// Updates the per-token routing state on the way through (roadmap
    /// 25.4); `None` in a trunk built without one.
    routing: Option<RoutingState<B>>,
}

impl<B: Backend> DbLayer<B> {
    /// Build one trunk layer. Fallsible for the reason
    /// [`crate::lm::LanguageModel::new`] is: an invalid config is reported with
    /// the field at fault, not by unwinding from inside an initializer.
    pub(crate) fn new(
        config: &ViTDiTConfig,
        layer_idx: usize,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        config.validate_ffn()?;
        config.validate_attention()?;
        let h = config.hidden_size;
        // The MoE router is conditioned on the adaLN vector, which is a pure
        // function of sigma -- that is what makes the routing noise-aware
        // (item 6.4) without any extra plumbing.
        let mlp = match (&config.mosme, config.moe) {
            // Hierarchical wins: it is the strict generalization.
            (Some(mosme), _) if mosme.applies_to(layer_idx) => {
                let cfg =
                    crate::mosme::MosmeConfig::new(h, config.cond_hidden_size, mosme.spec.clone())
                        .with_intermediate_size(config.intermediate_size)
                        .with_balance_bias(mosme.balance_bias)
                        .with_state_size(config.routing_state);
                FeedForward::Hierarchical(crate::mosme::MosmeFeedForward::new(&cfg, device))
            }
            (_, Some(moe)) if moe.applies_to(layer_idx) => {
                let cfg = crate::moe::MoEConfig::new(h, config.cond_hidden_size, moe.num_experts)
                    .with_z_level(moe.z_level)
                    .with_top_k(moe.top_k)
                    .with_intermediate_size(config.intermediate_size)
                    .with_balance_bias(moe.balance_bias)
                    .with_state_size(config.routing_state);
                FeedForward::Sparse(crate::moe::MoELayer::new(&cfg, device))
            }
            _ if config.ffn_kind == FfnKind::SwiGlu => FeedForward::DenseSwiGlu(SwiGluMlp::new(
                h,
                config.intermediate_size,
                config.hidden_dropout_prob,
                device,
            )),
            _ => FeedForward::Dense(Mlp::new(
                h,
                config.intermediate_size,
                config.hidden_dropout_prob,
                device,
            )),
        };
        let mode = config.attention_mode(layer_idx);
        assert!(
            config.causal || mode == AttentionMode::Dense,
            "layer {layer_idx} asks for {} attention, which is only defined causally",
            mode.name()
        );
        Ok(Self {
            attention: Attention::new(
                h,
                config.num_attention_heads,
                config.kv_heads(),
                config.attention_probs_dropout_prob,
                config.hidden_dropout_prob,
                mode,
                config.rotary,
                crate::hybrid::rotary_dim_for(
                    config.rotary_fraction,
                    h / config.num_attention_heads,
                ),
                config.gated_attention,
                config.qk_norm,
                device,
            ),
            mlp,
            layernorm_before: TrunkNorm::new(config.norm_kind, h, config.layer_norm_eps, device),
            layernorm_after: TrunkNorm::new(config.norm_kind, h, config.layer_norm_eps, device),
            ada_ln: AdaLN::new(config.cond_hidden_size, 6 * h, device),
            causal: config.causal,
            routing: (config.routing_state > 0)
                .then(|| RoutingState::new(h, config.routing_state, device)),
        })
    }

    #[cfg(test)]
    pub(crate) fn router_param_ids(&self, ids: &mut Vec<burn::module::ParamId>) {
        if let FeedForward::Hierarchical(site) = &self.mlp {
            ids.extend(burn::module::list_param_ids::<_, B>(site.router()));
        }
    }

    pub(crate) fn validate_specialist_config(
        &self,
        config: &ViTDiTConfig,
        idx: usize,
    ) -> anyhow::Result<()> {
        use crate::mosme::module_shapes;
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let c = config.cond_hidden_size;
        anyhow::ensure!(
            self.causal && self.routing.is_none(),
            "unsupported specialist layer"
        );
        let kv_dim = config.kv_heads() * self.attention.head_dim;
        anyhow::ensure!(
            self.attention.num_heads == config.num_attention_heads
                && self.attention.num_kv_heads == config.kv_heads()
                && self.attention.head_dim * config.num_attention_heads == h
                && self.attention.mode == config.attention_mode(idx)
                && self.attention.rotary == config.rotary
                && self.attention.rotary_dim
                    == crate::hybrid::rotary_dim_for(
                        config.rotary_fraction,
                        self.attention.head_dim
                    )
                && self.attention.gate.is_some() == config.gated_attention
                && self.attention.q_norm.is_some() == config.qk_norm
                && self.attention.k_norm.is_some() == config.qk_norm
                && self.attention.attn_dropout.prob == config.attention_probs_dropout_prob
                && self.attention.output_dropout.prob == config.hidden_dropout_prob,
            "specialist attention configuration mismatch"
        );
        anyhow::ensure!(
            module_shapes(&self.attention.query) == vec![vec![h, h], vec![h]],
            "attention q shape mismatch"
        );
        for linear in [&self.attention.key, &self.attention.value] {
            anyhow::ensure!(
                module_shapes(linear) == vec![vec![h, kv_dim], vec![kv_dim]],
                "attention kv shape mismatch"
            );
        }
        anyhow::ensure!(
            module_shapes(&self.attention.dense) == vec![vec![h, h], vec![h]],
            "attention out shape mismatch"
        );
        if let Some(gate) = &self.attention.gate {
            anyhow::ensure!(
                module_shapes(gate) == vec![vec![h, h], vec![h]],
                "attention gate shape mismatch"
            );
        }
        // Every layer carries the two mixing logits now, so the shape is the
        // only thing left to check; whether `Learned` reads them is the mode's
        // business, not a structural one.
        anyhow::ensure!(
            self.attention.mix.dims() == [2],
            "attention mix must be two logits, got {:?}",
            self.attention.mix.dims()
        );
        let expected_norm = match config.norm_kind {
            NormKind::Layer => vec![vec![h], vec![h]],
            NormKind::Rms => vec![vec![h]],
        };
        for norm in [&self.layernorm_before, &self.layernorm_after] {
            anyhow::ensure!(
                norm.kind() == config.norm_kind && module_shapes(norm) == expected_norm,
                "layer norm shape mismatch"
            );
        }
        anyhow::ensure!(
            module_shapes(&self.ada_ln) == vec![vec![c, 6 * h], vec![6 * h]],
            "conditioning shape mismatch"
        );
        let mosme = config
            .mosme
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing MoSME config"))?;
        match &self.mlp {
            FeedForward::Hierarchical(site) if mosme.applies_to(idx) => {
                site.validate_specialist_config(
                    &crate::mosme::MosmeConfig::new(h, c, mosme.spec.clone())
                        .with_intermediate_size(i)
                        .with_balance_bias(mosme.balance_bias),
                )?;
            }
            FeedForward::Dense(mlp) if !mosme.applies_to(idx) => {
                anyhow::ensure!(
                    module_shapes(mlp) == vec![vec![h, i], vec![i], vec![i, h], vec![h]]
                        && mlp.output_dropout.prob == config.hidden_dropout_prob,
                    "dense FFN mismatch"
                );
            }
            _ => anyhow::bail!("specialist FFN site placement mismatch at layer {idx}"),
        }
        Ok(())
    }

    pub(crate) fn compact_specialist(
        &self,
        position: (usize, usize),
        config: &crate::mosme::MosmeConfig,
    ) -> anyhow::Result<Self> {
        crate::tensor_ext::force_initialization(self);
        let mut result = self.clone();
        if let FeedForward::Hierarchical(site) = &self.mlp {
            result.mlp = FeedForward::Hierarchical(site.compact_specialist(position, config)?);
        }
        Ok(result)
    }

    pub(crate) fn apply_specialist(
        &self,
        specialist: &Self,
        position: (usize, usize),
    ) -> anyhow::Result<Self> {
        crate::tensor_ext::force_initialization(self);
        let mut result = self.clone();
        match (&self.mlp, &specialist.mlp) {
            (FeedForward::Hierarchical(target), FeedForward::Hierarchical(source)) => {
                result.mlp = FeedForward::Hierarchical(target.apply_specialist(source, position)?);
            }
            (FeedForward::Dense(_), FeedForward::Dense(_)) => {}
            _ => anyhow::bail!("specialist FFN site mismatch"),
        }
        Ok(result)
    }

    pub(crate) fn specialist_ids(
        &self,
        position: (usize, usize),
        expected: &[usize],
    ) -> anyhow::Result<Option<Vec<burn::module::ParamId>>> {
        match &self.mlp {
            FeedForward::Hierarchical(site) => {
                anyhow::ensure!(
                    site.router().experts_per_box() == expected,
                    "MoSME spec/module layout mismatch"
                );
                let expert = site
                    .expert(position.0, position.1)
                    .ok_or_else(|| anyhow::anyhow!("specialist is absent from MoSME site"))?;
                Ok(Some(burn::module::list_param_ids::<_, B>(expert)))
            }
            FeedForward::Sparse(_) => {
                anyhow::bail!("resident specialist training does not support flat MoE sites")
            }
            _ => Ok(None),
        }
    }

    pub(crate) fn attention_mode(&self) -> AttentionMode {
        self.attention.mode()
    }

    /// Query/KV head counts, for cross-checking a model against a config.
    pub(crate) fn attention_heads(&self) -> (usize, usize) {
        (self.attention.num_heads, self.attention.num_kv_heads)
    }

    /// Whether this layer's attention carries the zero-initialized output gate.
    pub(crate) fn attention_gated(&self) -> bool {
        self.attention.gate.is_some()
    }

    /// Whether this layer normalizes queries and keys before rotary.
    pub(crate) fn attention_qk_norm(&self) -> bool {
        self.attention.q_norm.is_some() && self.attention.k_norm.is_some()
    }

    /// This layer's attention input: LayerNorm plus the adaLN shift/scale,
    /// without the residual add or the attention itself. The max-logit probe
    /// reads the same tensor the attention branch does.
    pub(crate) fn normed_for_attention(
        &self,
        hidden_states: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let mods = self.ada_ln.forward(conditioning.clone());
        let h = mods.dims()[1] / 6;
        let shift_msa = mods.clone().narrow(1, 0, h).unsqueeze_dim::<3>(1);
        let scale_msa = mods.clone().narrow(1, h, h).unsqueeze_dim::<3>(1);
        modulate(
            self.layernorm_before.forward(hidden_states),
            shift_msa,
            scale_msa,
        )
    }

    /// Largest causally-valid attention score for this layer on an input
    /// already run through [`Self::normed_for_attention`], if the layer's
    /// mode keeps scores.
    pub(crate) fn attention_max_logit(
        &self,
        normed: Tensor<B, 3>,
        offset: usize,
    ) -> Option<Tensor<B, 1>> {
        self.attention.max_logit(normed, self.causal, offset)
    }

    /// Which normalization this layer uses.
    pub(crate) fn norm_kind(&self) -> NormKind {
        self.layernorm_before.kind()
    }

    /// Whether this layer carries a routing-state updater.
    pub(crate) fn has_routing_state(&self) -> bool {
        self.routing.is_some()
    }

    /// This layer's selection biases over its global expert index, if it is
    /// sparse and has them (roadmap 23.5). Hierarchical layers concatenate
    /// their boxes' expert-level biases.
    pub(crate) fn balance_bias_values(&self) -> Option<Vec<f32>> {
        match &self.mlp {
            FeedForward::Dense(_) | FeedForward::DenseSwiGlu(_) => None,
            FeedForward::Sparse(layer) => layer
                .router()
                .balance_bias()
                .map(|b| b.into_data().convert::<f32>().iter::<f32>().collect()),
            FeedForward::Hierarchical(layer) => {
                let boxes = layer.router().balance_biases();
                (!boxes.is_empty()).then(|| boxes.concat())
            }
        }
    }

    /// The adaLN gates `(gate_msa, gate_mlp)` this layer applies under
    /// `conditioning`, each `[b, h]` -- what its two residual writers are
    /// scaled by (roadmap 31.1).
    pub(crate) fn gates(&self, conditioning: &Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mods = self.ada_ln.forward(conditioning.clone());
        let h = mods.dims()[1] / 6;
        (mods.clone().narrow(1, 2 * h, h), mods.narrow(1, 5 * h, h))
    }

    /// The sparse feed-forward, if this layer has one.
    pub(crate) fn sparse_mut(&mut self) -> Option<SparseLayerMut<'_, B>> {
        match &mut self.mlp {
            FeedForward::Dense(_) | FeedForward::DenseSwiGlu(_) => None,
            FeedForward::Sparse(layer) => Some(SparseLayerMut::Flat(layer)),
            FeedForward::Hierarchical(layer) => Some(SparseLayerMut::Hierarchical(layer)),
        }
    }

    pub(crate) fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
        carry: &mut LayerCarry<B>,
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>) {
        let causal = self.causal;
        let offset = carry.offset;
        self.forward_with(hidden_states, conditioning, carry, |attn, normed| {
            attn.forward(normed, causal, offset)
        })
    }

    /// [`Self::forward`] over the new positions only, reusing `cache`.
    ///
    /// The adaLN modulation, the MLP branch and the residual structure are
    /// identical — only the attention differs — so both paths run the same
    /// code. That is deliberate: the certificate proves the *attention* is
    /// equivalent, and sharing everything else means there is no second copy
    /// of the layer for a divergence to hide in.
    ///
    /// # Panics
    ///
    /// If the layer is not causal. A bidirectional layer's earlier outputs
    /// depend on later inputs, so no prefix of its computation is reusable and
    /// a "cache" would silently return wrong activations.
    pub(crate) fn forward_cached(
        &self,
        hidden_states: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
        cache: &mut LayerKvCache<B>,
        carry: &mut LayerCarry<B>,
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>) {
        assert!(
            self.causal,
            "a KV cache is only sound for causal attention: with bidirectional \
             attention an earlier position's output depends on later ones"
        );
        self.forward_with(hidden_states, conditioning, carry, |attn, normed| {
            attn.forward_state(normed, cache)
        })
    }

    fn forward_with<F>(
        &self,
        hidden_states: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
        carry: &mut LayerCarry<B>,
        attend: F,
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>)
    where
        F: FnOnce(&Attention<B>, Tensor<B, 3>) -> Tensor<B, 3>,
    {
        // The routing state is updated from this layer's *input*, so the
        // routers inside the layer can already read it (roadmap 25.4).
        carry.routing_state = self
            .routing
            .as_ref()
            .map(|routing| routing.step(&hidden_states, carry.routing_state.as_ref()));
        // Downstream the routers take a tensor, and "this layer keeps no state"
        // is a zero-width tensor rather than a missing one. That way no call
        // site between here and the router has to unwrap anything, and the
        // router's own width check is what rejects a state of the wrong size.
        let routing_state = match &carry.routing_state {
            Some(state) => state.clone(),
            None => Tensor::<B, 3>::zeros([1, 1, 0], &hidden_states.device()),
        };
        let residual = hidden_states.clone();

        // Chunk the modulation vector [b, 6h] into six [b, h] slices.
        let mods = self.ada_ln.forward(conditioning.clone());
        let h = mods.dims()[1] / 6;
        let shift_msa = mods.clone().narrow(1, 0, h).unsqueeze_dim::<3>(1);
        let scale_msa = mods.clone().narrow(1, h, h).unsqueeze_dim::<3>(1);
        let gate_msa = mods.clone().narrow(1, 2 * h, h).unsqueeze_dim::<3>(1);
        let shift_mlp = mods.clone().narrow(1, 3 * h, h).unsqueeze_dim::<3>(1);
        let scale_mlp = mods.clone().narrow(1, 4 * h, h).unsqueeze_dim::<3>(1);
        let gate_mlp = mods.narrow(1, 5 * h, h).unsqueeze_dim::<3>(1);

        // Attention branch.
        let normed = modulate(
            self.layernorm_before.forward(hidden_states),
            shift_msa,
            scale_msa,
        );
        let attended = attend(&self.attention, normed);
        let hidden_states = attended * gate_msa + residual;

        // MLP branch.
        let layer_output = modulate(
            self.layernorm_after.forward(hidden_states.clone()),
            shift_mlp,
            scale_mlp,
        );
        let (layer_output, balance_loss) =
            self.mlp
                .forward(layer_output, conditioning, &routing_state, carry.specialist);
        (layer_output * gate_mlp + hidden_states, balance_loss)
    }
}

/// [`silu`] for the language-model trunk, which applies the same extra
/// activation to its conditioning vector as the image trunk does before
/// handing it to a layer's adaLN.
pub fn silu_public<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    silu(x)
}

/// `x * (1 + scale) + shift`, with per-batch affine params already expanded
/// to rank 3.
fn modulate<B: Backend>(x: Tensor<B, 3>, shift: Tensor<B, 3>, scale: Tensor<B, 3>) -> Tensor<B, 3> {
    x * (scale + 1.0) + shift
}

/// Output of [`ViTDiTModel::forward_block`].
#[derive(Debug, Clone)]
pub struct BlockOutput<B: Backend> {
    /// Final-LayerNormed sequence `[b, seq, h]`.
    pub last_hidden_state: Tensor<B, 3>,
    /// Conditioning vector `[b, cond]` reused by the output head.
    pub conditioning: Tensor<B, 2>,
    /// Summed load-balancing loss of any MoE layers in the executed span;
    /// `None` for a fully dense span. Add it to the training objective --
    /// without it the router is free to collapse onto one expert.
    /// Auxiliary routing losses of the executed span; `None` for a dense
    /// trunk. See [`RouterAux`] for why balance and z-loss stay apart.
    pub balance_loss: Option<RouterAux<B>>,
}

/// The ViT-DiT trunk (`ViTDiTModel`).
#[derive(Module, Debug)]
pub struct ViTDiTModel<B: Backend> {
    embeddings: ViTDiTEmbeddings<B>,
    time_embedder: TimestepEmbedder<B>,
    layers: Vec<DbLayer<B>>,
    final_layernorm: TrunkNorm<B>,
}

impl<B: Backend> ViTDiTModel<B> {
    /// Build the image/vision trunk. Fallsible for the reason
    /// [`crate::lm::LanguageModel::new`] is.
    pub fn new(config: &ViTDiTConfig, device: &B::Device) -> anyhow::Result<Self> {
        config.validate_ffn()?;
        config.validate_attention()?;
        Ok(Self {
            embeddings: ViTDiTEmbeddings::new(config, device),
            time_embedder: TimestepEmbedder::new(
                config.cond_hidden_size,
                config.frequency_embedding_size,
                device,
            ),
            layers: (0..config.num_hidden_layers)
                .map(|idx| DbLayer::new(config, idx, device))
                .collect::<anyhow::Result<_>>()?,
            final_layernorm: TrunkNorm::new(
                config.norm_kind,
                config.hidden_size,
                config.layer_norm_eps,
                device,
            ),
        })
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Selection biases of every sparse layer that has one, in layer order.
    pub fn balance_biases(&self) -> Vec<Vec<f32>> {
        self.layers
            .iter()
            .filter_map(DbLayer::balance_bias_values)
            .collect()
    }

    /// Visit every sparse layer in `range`, in execution order -- the same
    /// order [`RouterAux::layers`] reports them in, so the two can be paired.
    pub(crate) fn for_each_sparse_layer_mut(
        &mut self,
        range: std::ops::Range<usize>,
        mut f: impl FnMut(usize, SparseLayerMut<'_, B>),
    ) {
        let end = range.end.min(self.layers.len());
        let mut nth = 0;
        for i in range.start..end {
            if let Some(layer) = self.layers[i].sparse_mut() {
                f(nth, layer);
                nth += 1;
            }
        }
    }

    /// Compute embeddings + conditioning shared by every layer subset.
    fn embed(
        &self,
        pixel_values: Tensor<B, 4>,
        noisy_embeds: Tensor<B, 2>,
        timesteps: Tensor<B, 1>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        // Reference applies SiLU once more outside the timestep MLP.
        let conditioning = silu(self.time_embedder.forward(timesteps));
        let emb = self.embeddings.forward(pixel_values, noisy_embeds);
        (emb, conditioning)
    }

    /// Run transformer layers `[start, end)` and apply the final LayerNorm
    /// (`layer_indices` selection in the reference encoder).
    fn run_layers(
        &self,
        range: std::ops::Range<usize>,
        mut hidden_states: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>) {
        let mut aux_total: Option<RouterAux<B>> = None;
        let mut carry = LayerCarry::default();
        for i in range.start..range.end.min(self.layers.len()) {
            let (states, aux) = self.layers[i].forward(hidden_states, conditioning, &mut carry);
            hidden_states = states;
            if let Some(aux) = aux {
                aux_total = Some(match aux_total {
                    None => aux,
                    Some(acc) => acc.combine(aux),
                });
            }
        }
        (self.final_layernorm.forward(hidden_states), aux_total)
    }

    /// Block-wise forward (`forward_block`): embed inputs, run only the
    /// selected contiguous layer window, apply the final LayerNorm.
    pub fn forward_block(
        &self,
        layer_indices: std::ops::Range<usize>,
        pixel_values: Tensor<B, 4>,
        noisy_embeds: Tensor<B, 2>,
        timesteps: Tensor<B, 1>,
    ) -> BlockOutput<B> {
        let (emb, cond) = self.embed(pixel_values, noisy_embeds, timesteps);
        let (hidden, balance_loss) = self.run_layers(layer_indices, emb, &cond);
        BlockOutput {
            last_hidden_state: hidden,
            conditioning: cond,
            balance_loss,
        }
    }

    /// Full forward through every layer.
    pub fn forward_all(
        &self,
        pixel_values: Tensor<B, 4>,
        noisy_embeds: Tensor<B, 2>,
        timesteps: Tensor<B, 1>,
    ) -> BlockOutput<B> {
        let n = self.layers.len();
        self.forward_block(0..n, pixel_values, noisy_embeds, timesteps)
    }

    /// The label embedding table `[num_labels, hidden]`
    /// (`get_input_embeddings`).
    pub fn label_embedding_weight(&self) -> Tensor<B, 2> {
        self.embeddings.label_embeddings.weight.val()
    }
}

/// Classification head: adaLN modulation followed by a zero-initialized
/// linear projection (`adaLN_modulation` + `classifier`).
#[derive(Module, Debug)]
pub struct DbOutputHead<B: Backend> {
    ada_ln: AdaLN<B>,
    classifier: Linear<B>,
    hidden_size: usize,
}

impl<B: Backend> DbOutputHead<B> {
    pub fn new(config: &ViTDiTConfig, device: &B::Device) -> Self {
        Self {
            ada_ln: AdaLN::new(config.cond_hidden_size, 2 * config.hidden_size, device),
            classifier: LinearConfig::new(config.hidden_size, config.num_labels)
                .with_bias(true)
                .init(device),
            hidden_size: config.hidden_size,
        }
    }

    /// `model_out`: `[b, tokens, h]`, `conditioning`: `[b, cond]`.
    /// Returns logits `[b, num_labels]` computed from token 0.
    pub fn forward(&self, model_out: Tensor<B, 3>, conditioning: Tensor<B, 2>) -> Tensor<B, 2> {
        let cls = self.modulated_cls(model_out, conditioning);
        self.classifier.forward(cls)
    }

    /// adaLN-modulated CLS-token hidden `[b, h]`, before the classifier.
    /// Used by vector-prediction objectives such as flow matching.
    pub fn modulated_cls(
        &self,
        model_out: Tensor<B, 3>,
        conditioning: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let mods = self.ada_ln.forward(conditioning);
        let h = self.hidden_size;
        let shift = mods.clone().narrow(1, 0, h).unsqueeze_dim::<3>(1);
        let scale = mods.narrow(1, h, h).unsqueeze_dim::<3>(1);
        let modulated = modulate(model_out, shift, scale);
        modulated.narrow(1, 0, 1).squeeze_dim::<2>(1)
    }
}

/// Full dblock model: trunk + conditioned head
/// (`ViTDiTForImageClassification`, time-conditioned variant).
#[derive(Module, Debug)]
pub struct ViTDiTForImageClassification<B: Backend> {
    vit: ViTDiTModel<B>,
    head: DbOutputHead<B>,
}

impl<B: Backend> ViTDiTForImageClassification<B> {
    /// Build the classification model. Fallsible, like [`Self::new`] on the
    /// trunk it wraps.
    pub fn new(config: &ViTDiTConfig, device: &B::Device) -> anyhow::Result<Self> {
        Ok(Self {
            vit: ViTDiTModel::new(config, device)?,
            head: DbOutputHead::new(config, device),
        })
    }

    /// Apply the DiT-specific initialization from `_init_dit`:
    ///
    /// - label embedding table ~ N(0, initializer_range^2)
    /// - timestep MLP weights ~ N(0, initializer_range^2)
    /// - all adaLN modulation linears zeroed
    /// - classifier weight and bias zeroed
    pub fn with_dit_init(self, config: &ViTDiTConfig, device: &B::Device) -> anyhow::Result<Self> {
        use burn::module::Param;

        let std = config.initializer_range;
        let mut rec = self.into_record();

        // Label embedding table.
        {
            let shape = rec.vit.embeddings.label_embeddings.weight.shape();
            rec.vit.embeddings.label_embeddings.weight = Param::from_tensor(Tensor::random(
                shape,
                Distribution::Normal(0.0, std),
                device,
            ));
        }
        // Timestep embedder MLP weights (biases keep their default init,
        // mirroring nn.init.normal_ which only touches weights).
        for lin in [
            &mut rec.vit.time_embedder.linear_1,
            &mut rec.vit.time_embedder.linear_2,
        ] {
            let shape = lin.weight.shape();
            lin.weight = Param::from_tensor(Tensor::random(
                shape,
                Distribution::Normal(0.0, std),
                device,
            ));
        }
        // Zero all adaLN modulation linears (DiT zero-init trick).
        for layer in rec.vit.layers.iter_mut() {
            zero_linear_params(&mut layer.ada_ln.linear);
        }
        zero_linear_params(&mut rec.head.ada_ln.linear);
        // Zero the classifier.
        zero_linear_params(&mut rec.head.classifier);

        Ok(Self::new(config, device)?.load_record(rec))
    }

    /// Run only the layers in `layer_indices` and produce class logits
    /// (CLS pooling inside).
    pub fn forward_block(
        &self,
        layer_indices: std::ops::Range<usize>,
        pixel_values: Tensor<B, 4>,
        noisy_embeds: Tensor<B, 2>,
        timesteps: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let out = self
            .vit
            .forward_block(layer_indices, pixel_values, noisy_embeds, timesteps);
        let pooled = out.last_hidden_state.narrow(1, 0, 1); // CLS token slot
        self.head.forward(pooled, out.conditioning)
    }

    /// Like [`Self::forward_block`] but returns the adaLN-modulated CLS
    /// hidden `[b, h]` before the classifier head (for flow matching).
    pub fn forward_pooled_block(
        &self,
        layer_indices: std::ops::Range<usize>,
        pixel_values: Tensor<B, 4>,
        noisy_embeds: Tensor<B, 2>,
        timesteps: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let out = self
            .vit
            .forward_block(layer_indices, pixel_values, noisy_embeds, timesteps);
        let pooled = out.last_hidden_state.narrow(1, 0, 1);
        self.head.modulated_cls(pooled, out.conditioning)
    }

    /// Full forward through every layer.
    pub fn forward_all(
        &self,
        pixel_values: Tensor<B, 4>,
        noisy_embeds: Tensor<B, 2>,
        timesteps: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let n = self.vit.num_layers();
        self.forward_block(0..n, pixel_values, noisy_embeds, timesteps)
    }

    /// Label embedding table weight `[num_labels, hidden]`.
    pub fn label_embedding_weight(&self) -> Tensor<B, 2> {
        self.vit.label_embedding_weight()
    }

    /// Project a (denoised) hidden state through the adaLN-modulated
    /// classification head (`forward_output_embeddings`).
    pub fn forward_output_embeddings(
        &self,
        model_out: Tensor<B, 3>,
        conditioning: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.head.forward(model_out, conditioning)
    }

    /// Look up label ids and L2-normalize (`get_embeds`).
    pub fn normalized_label_embeds(&self, labels: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        self.vit.embeddings.embed_labels(labels)
    }

    /// Access the trunk (for block partitioning logic).
    pub(crate) fn vit_mut(&mut self) -> &mut ViTDiTModel<B> {
        &mut self.vit
    }

    pub fn vit(&self) -> &ViTDiTModel<B> {
        &self.vit
    }
}

fn zero_linear_params<B: Backend>(linear: &mut burn::nn::LinearRecord<B>) {
    let dev = linear.weight.device();
    let sw = linear.weight.shape();
    linear.weight = Param::from_tensor(Tensor::zeros(sw, &dev));
    if let Some(bias) = &linear.bias {
        let sb = bias.shape();
        linear.bias = Some(Param::from_tensor(Tensor::zeros(sb, &dev)));
    }
}

// A test says "this must have worked" with `unwrap`, which is the right thing
// for a test to say. The grant is scoped to this module: production code in the
// same file is still denied it (see the `[lints]` table in `Cargo.toml` and the
// contract in the crate docs).
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
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn tiny_config() -> ViTDiTConfig {
        ViTDiTConfig::tiny(10)
    }

    #[test]
    fn test_ffn_config_defaults_and_roundtrip() {
        for cfg in [
            ViTDiTConfig::tiny(10),
            ViTDiTConfig::cifar(10),
            ViTDiTConfig::tiny_imagenet(200),
        ] {
            assert_eq!(cfg.ffn_kind, FfnKind::Gelu);
            let mut old = serde_json::to_value(&cfg).unwrap();
            old.as_object_mut().unwrap().remove("ffn_kind");
            let parsed: ViTDiTConfig = serde_json::from_value(old).unwrap();
            assert_eq!(parsed.ffn_kind, FfnKind::Gelu);
            for kind in [FfnKind::Gelu, FfnKind::SwiGlu] {
                let value = serde_json::to_value(cfg.clone().with_ffn_kind(kind)).unwrap();
                assert_eq!(value["ffn_kind"], kind.name());
                let back: ViTDiTConfig = serde_json::from_value(value.clone()).unwrap();
                assert_eq!(back.ffn_kind, kind);
                assert_eq!(serde_json::to_value(back).unwrap(), value);
            }
        }
    }

    #[test]
    fn test_default_ffn_is_unchanged_gelu() {
        let device = Default::default();
        let layer = DbLayer::<B>::new(&tiny_config(), 0, &device).unwrap();
        let FeedForward::Dense(mlp) = layer.mlp else {
            panic!("default must use the legacy dense MLP")
        };
        let x = Tensor::<B, 3>::random([2, 3, 32], Distribution::Uniform(-2.0, 2.0), &device);
        let expected = mlp.fc_out.forward(exact_gelu(mlp.fc_in.forward(x.clone())));
        assert_eq!((mlp.forward(x) - expected).abs().max().into_scalar(), 0.0);
        assert_eq!(
            mlp.num_params(),
            crate::cost::dense_mlp_cost(32, 64).active_params
        );
    }

    #[test]
    fn test_swiglu_formula_and_gradients() {
        use burn::backend::Autodiff;
        type A = Autodiff<B>;
        let device = Default::default();
        let mut mlp = SwiGluMlp::<A>::new(1, 1, 0.0, &device);
        for (linear, weight, bias) in [
            (&mut mlp.fc_in, 2.0, 0.3),
            (&mut mlp.fc_gate, -0.7, 0.2),
            (&mut mlp.fc_out, 1.3, -0.1),
        ] {
            linear.weight = Param::from_tensor(Tensor::full([1, 1], weight, &device));
            linear.bias = Some(Param::from_tensor(Tensor::full([1], bias, &device)));
        }
        let xs = [-2.0f32, 0.0, 1.0, 3.0];
        let x = Tensor::<A, 1>::from_floats(xs, &device)
            .reshape([2, 2, 1])
            .require_grad();
        let y = mlp.forward(x.clone());
        assert_eq!(y.dims(), [2, 2, 1]);
        let values: Vec<f32> = y.clone().into_data().iter::<f32>().collect();
        let grads = y.sum().backward();
        let dx: Vec<f32> = x.grad(&grads).unwrap().into_data().iter::<f32>().collect();
        let mut expected_params = [0.0f32; 6];
        for (i, x) in xs.into_iter().enumerate() {
            let up = 2.0 * x + 0.3;
            let gate = -0.7 * x + 0.2;
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            let activated = gate * sigmoid;
            let d_up = 1.3 * activated;
            let d_gate = 1.3 * up * sigmoid * (1.0 + gate * (1.0 - sigmoid));
            let expected = 1.3 * activated * up - 0.1;
            assert!(
                (values[i] - expected).abs() < 2e-5,
                "{} vs {expected}",
                values[i]
            );
            assert!((dx[i] - (2.0 * d_up - 0.7 * d_gate)).abs() < 2e-5);
            for (sum, term) in expected_params.iter_mut().zip([
                x * d_up,
                d_up,
                x * d_gate,
                d_gate,
                activated * up,
                1.0,
            ]) {
                *sum += term;
            }
        }
        let mut actual = Vec::new();
        for linear in [&mlp.fc_in, &mlp.fc_gate, &mlp.fc_out] {
            actual.push(
                linear
                    .weight
                    .val()
                    .grad(&grads)
                    .unwrap()
                    .reshape([1])
                    .into_scalar(),
            );
            actual.push(
                linear
                    .bias
                    .as_ref()
                    .unwrap()
                    .val()
                    .grad(&grads)
                    .unwrap()
                    .into_scalar(),
            );
        }
        for (got, expected) in actual.into_iter().zip(expected_params) {
            assert!(
                got.is_finite() && (got - expected).abs() < 3e-5,
                "{got} vs {expected}"
            );
        }
    }

    #[test]
    fn test_swiglu_layer_shape_and_parameter_cost() {
        let device = Default::default();
        let cfg = tiny_config().with_ffn_kind(FfnKind::SwiGlu);
        let layer = DbLayer::<B>::new(&cfg, 0, &device).unwrap();
        let FeedForward::DenseSwiGlu(mlp) = layer.mlp else {
            panic!("SwiGLU was not selected")
        };
        assert_eq!(mlp.fc_gate.weight.dims(), [32, 64]);
        assert_eq!(
            mlp.num_params(),
            crate::cost::dense_ffn_cost(32, 64, FfnKind::SwiGlu).active_params
        );
        let x = Tensor::<B, 3>::ones([2, 3, 32], &device);
        assert_eq!(mlp.forward(x).dims(), [2, 3, 32]);
    }

    #[test]
    fn test_swiglu_rejects_expert_trunks() {
        let cfg = tiny_config()
            .with_ffn_kind(FfnKind::SwiGlu)
            .with_moe(MoeTrunkConfig::default());
        let err = ViTDiTModel::<B>::new(&cfg, &Default::default())
            .expect_err("SwiGLU cannot be combined with an expert trunk");
        assert!(
            err.to_string()
                .contains("cannot be combined with MoE or MoSME"),
            "the error must name the conflict, got: {err}"
        );
    }

    #[test]
    fn test_forward_shapes() {
        let device = Default::default();
        let cfg = tiny_config();
        let model = ViTDiTForImageClassification::new(&cfg, &device).unwrap();

        let pixels = Tensor::<B, 4>::zeros([2, 3, 32, 32], &device);
        let zt = Tensor::<B, 2>::zeros([2, 32], &device);
        let t = Tensor::<B, 1>::zeros([2], &device);

        let logits = model.forward_block(0..2, pixels, zt, t);
        assert_eq!(logits.dims(), [2, 10]);
    }

    #[test]
    fn test_layer_subset_changes_output() {
        let device = Default::default();
        let cfg = tiny_config();
        let model = ViTDiTForImageClassification::new(&cfg, &device).unwrap();

        let pixels = Tensor::<B, 4>::ones([1, 3, 32, 32], &device);
        let zt = Tensor::<B, 2>::ones([1, 32], &device);
        let t = Tensor::<B, 1>::zeros([1], &device);

        let partial = model.forward_block(0..2, pixels.clone(), zt.clone(), t.clone());
        let full = model.forward_all(pixels, zt, t);
        let diff = (full - partial).abs().max().into_scalar();
        assert!(diff > 1e-5, "different layer subsets must differ");
    }

    #[test]
    fn test_normalized_label_embeds_unit_norm() {
        let device = Default::default();
        let cfg = tiny_config();
        let model = ViTDiTForImageClassification::new(&cfg, &device).unwrap();
        let labels = Tensor::<B, 1, Int>::from_ints([0, 3, 9], &device);
        let e = model.normalized_label_embeds(labels);
        assert_eq!(e.dims(), [3, 32]);
        let norms = e.powf_scalar(2.0).sum_dim(1).sqrt();
        let err = (norms - 1.0).abs().max().into_scalar();
        assert!(err < 1e-5, "unit-norm violated, max err {err}");
    }

    #[test]
    fn test_moe_trunk_placement_and_balance_loss() {
        let device = Default::default();
        let moe = MoeTrunkConfig {
            num_experts: 4,
            top_k: 2,
            every_n_layers: 2,
            z_level: 1e-3,
            balance_bias: false,
        };

        // Placement is arithmetic, so check it directly before building
        // anything: every second layer, i.e. layers 1 and 3 of 4.
        assert_eq!(
            (0..4).filter(|i| moe.applies_to(*i)).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(moe.num_sparse_layers(4), 2);
        assert_eq!(
            MoeTrunkConfig {
                every_n_layers: 1,
                ..moe
            }
            .num_sparse_layers(4),
            4
        );

        let cfg = ViTDiTConfig::tiny(10).with_moe(moe);
        let model = ViTDiTForImageClassification::<B>::new(&cfg, &device).unwrap();

        let pixels = Tensor::<B, 4>::ones([2, 3, 32, 32], &device);
        let zt = Tensor::<B, 2>::ones([2, 32], &device);
        let t = Tensor::<B, 1>::zeros([2], &device);

        // A span containing no sparse layer reports no auxiliary loss...
        let dense_span = model
            .vit()
            .forward_block(0..1, pixels.clone(), zt.clone(), t.clone());
        assert!(dense_span.balance_loss.is_none(), "layer 0 is dense");

        // ...and one containing sparse layers reports a finite, positive one.
        let sparse_span = model
            .vit()
            .forward_block(0..4, pixels.clone(), zt.clone(), t.clone());
        let aux: f32 = sparse_span
            .balance_loss
            .map(|a| a.balance)
            .expect("layers 1 and 3 are sparse")
            .into_scalar();
        assert!(
            aux.is_finite() && aux > 0.0,
            "balance loss must be positive: {aux}"
        );
        // Two sparse layers, each contributing at least the uniform minimum 1.
        assert!(
            aux >= 2.0 - 1e-4,
            "two sparse layers must each contribute >= 1: {aux}"
        );
        assert!(aux <= 2.0 * moe.num_experts as f32 + 1e-4);

        // The sparse trunk still produces well-shaped logits.
        assert_eq!(model.forward_all(pixels, zt, t).dims(), [2, 10]);
    }

    #[test]
    fn test_mosme_trunk_placement_and_balance_loss() {
        use crate::expert_index::{BoxSpec, ExpertSpec, MosmeSpec};

        let device = Default::default();
        let spec = MosmeSpec {
            boxes: vec![
                BoxSpec::new(
                    "coding",
                    "Code",
                    vec![
                        ExpertSpec::new("coding/rust", "Rust"),
                        ExpertSpec::new("coding/python", "Python"),
                    ],
                ),
                BoxSpec::new(
                    "cyber",
                    "Cybersecurity",
                    vec![ExpertSpec::new("cyber/netsec", "Network")],
                ),
            ],
            top_box: 1,
            top_expert: 1,
            route_on_tokens: true,
            balance: Default::default(),
        };
        let trunk = MosmeTrunkConfig::new(spec).with_every_n_layers(2);

        // Placement is arithmetic; check it before building anything.
        assert_eq!(
            (0..4).filter(|i| trunk.applies_to(*i)).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(trunk.num_hierarchical_layers(4), 2);

        let cfg = ViTDiTConfig::tiny(10).with_mosme(trunk);
        let model = ViTDiTForImageClassification::<B>::new(&cfg, &device).unwrap();

        let pixels = Tensor::<B, 4>::ones([2, 3, 32, 32], &device);
        let zt = Tensor::<B, 2>::ones([2, 32], &device);
        let t = Tensor::<B, 1>::zeros([2], &device);

        // A span with no hierarchical layer reports no auxiliary loss...
        let dense = model
            .vit()
            .forward_block(0..1, pixels.clone(), zt.clone(), t.clone());
        assert!(dense.balance_loss.is_none(), "layer 0 is dense");

        // ...and one that includes them reports a finite, positive one. Each
        // hierarchical layer contributes a box term and an expert term, both
        // at least 1 on the diagonal, so two layers give at least 4.
        let sparse = model
            .vit()
            .forward_block(0..4, pixels.clone(), zt.clone(), t.clone());
        let aux: f32 = sparse
            .balance_loss
            .expect("layers 1 and 3 are hierarchical")
            .balance
            .into_scalar();
        assert!(
            aux.is_finite() && aux > 0.0,
            "balance loss must be positive: {aux}"
        );
        assert!(
            aux >= 2.0,
            "two hierarchical layers must each contribute: {aux}"
        );

        assert_eq!(model.forward_all(pixels, zt, t).dims(), [2, 10]);
    }

    #[test]
    fn test_mosme_takes_precedence_over_flat_moe() {
        // Both configured is a configuration error the CLI rejects, but the
        // model must still behave predictably: hierarchical wins, because it
        // is the strict generalization.
        use crate::expert_index::MosmeSpec;
        let device = Default::default();
        let cfg = ViTDiTConfig::tiny(10)
            .with_moe(MoeTrunkConfig {
                num_experts: 4,
                top_k: 1,
                every_n_layers: 1,
                z_level: 1e-3,
                balance_bias: false,
            })
            .with_mosme(MosmeTrunkConfig::new(MosmeSpec::flat(2)).with_every_n_layers(1));
        let model = ViTDiTForImageClassification::<B>::new(&cfg, &device).unwrap();

        // A single-box hierarchical layer reports a box loss of exactly 1 on
        // top of its expert loss, which a flat layer would not.
        let out = model.vit().forward_block(
            0..1,
            Tensor::<B, 4>::ones([1, 3, 32, 32], &device),
            Tensor::<B, 2>::ones([1, 32], &device),
            Tensor::<B, 1>::zeros([1], &device),
        );
        let aux: f32 = out
            .balance_loss
            .expect("layer 0 is hierarchical")
            .balance
            .into_scalar();
        assert!(aux >= 1.0, "hierarchical path should be active, got {aux}");
    }

    /// Perturb the last position of a sequence and report how much the
    /// preceding positions moved.
    ///
    /// Exercises `Attention` directly rather than a whole `DbLayer`: the
    /// layer's attention branch is gated by adaLN, which can be near zero at a
    /// random initialization, so a layer-level test would measure the gate
    /// rather than the mask.
    fn prefix_drift_from_future_perturbation(causal: bool) -> (f32, f32) {
        let device = Default::default();
        let (hidden, heads, seq) = (16usize, 4usize, 6usize);
        let attention = Attention::<B>::new(
            hidden,
            heads,
            heads,
            0.0,
            0.0,
            AttentionMode::Dense,
            false,
            hidden / heads,
            false,
            false,
            &device,
        );

        let base = Tensor::<B, 3>::random(
            [1, seq, hidden],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let reference = attention.forward(base.clone(), causal, 0);

        let tail = base.clone().narrow(1, seq - 1, 1) + 10.0;
        let perturbed = Tensor::cat(vec![base.narrow(1, 0, seq - 1), tail], 1);
        let changed = attention.forward(perturbed, causal, 0);

        let prefix = (reference.clone().narrow(1, 0, seq - 1)
            - changed.clone().narrow(1, 0, seq - 1))
        .abs()
        .max()
        .into_scalar();
        let tail_drift = (reference.narrow(1, seq - 1, 1) - changed.narrow(1, seq - 1, 1))
            .abs()
            .max()
            .into_scalar();
        (prefix, tail_drift)
    }

    #[test]
    fn test_causal_attention_cannot_see_the_future() {
        // The defining property, tested the only way that really settles it:
        // perturb a later position and check that no earlier position moves. A
        // mask that is off by one, or applied after the softmax instead of
        // before, fails immediately.
        let (prefix, tail) = prefix_drift_from_future_perturbation(true);
        assert_eq!(
            prefix, 0.0,
            "a causal layer leaked information backwards: {prefix}"
        );
        // ...and the perturbed position itself must respond, or the assertion
        // above would hold for attention that ignores its input entirely.
        assert!(tail > 1e-4, "the perturbed position should change: {tail}");
    }

    #[test]
    fn test_bidirectional_attention_does_see_the_future() {
        // The counterpart. Without it, the causal test would also pass for an
        // implementation that never mixes positions at all.
        let (prefix, _) = prefix_drift_from_future_perturbation(false);
        assert!(
            prefix > 1e-4,
            "bidirectional attention should propagate backwards: {prefix}"
        );
    }

    #[test]
    fn test_causal_mask_shape_and_values() {
        let device = Default::default();
        let mask = causal_mask::<B>(4, &device);
        assert_eq!(mask.dims(), [1, 1, 4, 4]);
        let values: Vec<f32> = mask.into_data().convert::<f32>().iter::<f32>().collect();
        for query in 0..4 {
            for key in 0..4 {
                let v = values[query * 4 + key];
                if key <= query {
                    assert_eq!(v, 0.0, "({query},{key}) should be visible");
                } else {
                    assert!(
                        v.is_infinite() && v < 0.0,
                        "({query},{key}) should be masked"
                    );
                }
            }
        }
    }

    #[test]
    fn test_dit_init_zero_logits_and_gates() {
        let device = Default::default();
        let cfg = tiny_config();
        let model = ViTDiTForImageClassification::new(&cfg, &device)
            .unwrap()
            .with_dit_init(&cfg, &device)
            .unwrap();

        let pixels = Tensor::<B, 4>::zeros([1, 3, 32, 32], &device);
        let zt = Tensor::<B, 2>::zeros([1, 32], &device);
        let t = Tensor::<B, 1>::zeros([1], &device);
        let logits = model.forward_all(pixels, zt, t);
        let max_abs = logits.abs().max().into_scalar();
        assert_eq!(max_abs, 0.0, "zero-init classifier must give zero logits");

        // Label embeddings must be non-trivial after init.
        let w = model.label_embedding_weight();
        let w_abs = w.abs().max().into_scalar();
        assert!(w_abs > 0.001 && w_abs < 0.15, "embedding std off: {w_abs}");
    }
}
