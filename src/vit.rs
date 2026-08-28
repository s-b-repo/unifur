//! ViT-DiT backbone: a HuggingFace-style ViT whose CLS token is replaced by
//! the noisy class embedding, with DiT-style adaLN-zero timestep
//! conditioning on every transformer layer.
//!
//! Ported from `vit.py` in the reference repository (`ViTDiT*` classes),
//! restricted to the `time_conditioning = true` configuration used by
//! DiffusionBlocks.

use crate::tensor_ext::{exact_gelu, l2_normalize_rows, silu};
use burn::{
    module::{Module, Param},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{
        activation::softmax,
        backend::Backend,
        Distribution,
        Int,
        Tensor,
    },
};

/// Hyperparameters of the ViT-DiT backbone (mirrors `ViTDiTConfig`).
#[derive(Debug, Clone)]
pub struct ViTDiTConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
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
}

/// Where and how mixture-of-experts layers enter the trunk.
///
/// Only *some* layers become sparse: alternating dense and sparse layers is
/// the standard Switch/GLaM placement, and it keeps the dense path available
/// for the features every expert needs. `every_n_layers = 2` replaces every
/// second layer's MLP.
#[derive(Debug, Clone, Copy)]
pub struct MoeTrunkConfig {
    pub num_experts: usize,
    pub top_k: usize,
    /// Replace the MLP of every `every_n_layers`-th layer (1 = all of them).
    pub every_n_layers: usize,
    /// Router z-loss weight (ST-MoE); `0.0` disables it exactly.
    pub z_level: f64,
}

impl Default for MoeTrunkConfig {
    fn default() -> Self {
        Self { num_experts: 4, top_k: 1, every_n_layers: 2, z_level: 1e-3 }
    }
}

/// Where hierarchical expert boxes enter the trunk (roadmap 18.7).
///
/// Deliberately not `Copy`: the spec carries box and expert names, which is
/// the whole point of it.
#[derive(Debug, Clone)]
pub struct MosmeTrunkConfig {
    pub spec: crate::expert_index::MosmeSpec,
    /// Replace the feed-forward of every `every_n_layers`-th layer.
    pub every_n_layers: usize,
}

impl MosmeTrunkConfig {
    pub fn new(spec: crate::expert_index::MosmeSpec) -> Self {
        Self { spec, every_n_layers: 2 }
    }

    pub fn with_every_n_layers(mut self, n: usize) -> Self {
        self.every_n_layers = n;
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

impl ViTDiTConfig {
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
        }
    }

    /// Enable mixture-of-experts layers in the trunk.
    pub fn with_moe(mut self, moe: MoeTrunkConfig) -> Self {
        self.moe = Some(moe);
        self
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
    pub fn new(cond_hidden_size: usize, frequency_embedding_size: usize, device: &B::Device) -> Self {
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

/// Multi-head self-attention (`ViTAttention`, biased qkv/out projections).
#[derive(Module, Debug)]
struct Attention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    dense: Linear<B>,
    num_heads: usize,
    head_dim: usize,
    attn_dropout: Dropout,
    output_dropout: Dropout,
}

impl<B: Backend> Attention<B> {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        attn_dropout: f64,
        output_dropout: f64,
        device: &B::Device,
    ) -> Self {
        let mut query = LinearConfig::new(hidden_size, hidden_size)
            .with_bias(true)
            .init(device);
        // Fold the 1/sqrt(head_dim) attention scale into the Q projection so
        // the forward pass needs no extra elementwise multiply per layer.
        // Numerically identical to scaling q after projection.
        let scale = (hidden_size / num_heads) as f64;
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

        Self {
            query,
            key: LinearConfig::new(hidden_size, hidden_size).with_bias(true).init(device),
            value: LinearConfig::new(hidden_size, hidden_size).with_bias(true).init(device),
            dense: LinearConfig::new(hidden_size, hidden_size).with_bias(true).init(device),
            num_heads,
            head_dim: hidden_size / num_heads,
            attn_dropout: DropoutConfig::new(attn_dropout).init(),
            output_dropout: DropoutConfig::new(output_dropout).init(),
        }
    }

    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        x.reshape([b, n, self.num_heads, self.head_dim])
            .swap_dims(1, 2) // [b, heads, n, head_dim]
    }

    fn forward(&self, x: Tensor<B, 3>, causal: bool) -> Tensor<B, 3> {
        // The 1/sqrt(head_dim) scale is folded into the Q projection weights.
        let q = self.split_heads(self.query.forward(x.clone()));
        let k = self.split_heads(self.key.forward(x.clone()));
        let v = self.split_heads(self.value.forward(x));

        // Scaled dot-product attention over all heads.
        let mut scores = q.matmul(k.swap_dims(2, 3)); // [b, heads, n, n]
        if causal {
            // Additive -inf above the diagonal, so exp() gives exactly 0 there
            // and position i can place no weight whatsoever on i+1. Masking the
            // *scores* rather than the probabilities is what makes that exact:
            // zeroing after the softmax would leave the denominator polluted by
            // the future.
            let [_, _, n, _] = scores.dims();
            let device = scores.device();
            scores = scores + causal_mask::<B>(n, &device);
        }
        let probs = self.attn_dropout.forward(softmax(scores, 3));
        let ctx = probs.matmul(v).swap_dims(1, 2); // [b, n, heads, head_dim]

        let [b, n, _, _] = ctx.dims();
        let ctx = ctx.reshape([b, n, self.num_heads * self.head_dim]);
        self.output_dropout.forward(self.dense.forward(ctx))
    }

    /// Causal attention over `x` (the *new* positions only), reusing and
    /// extending `cache`.
    ///
    /// Generation without a cache recomputes every previous position's keys and
    /// values on every step: `O(n^2)` work per token instead of `O(n)`. Nothing
    /// about the result changes — those tensors are a pure function of tokens
    /// that have already been committed — which is why
    /// `lm/kv_cache_matches_full_recompute` can demand exact agreement rather
    /// than a tolerance.
    ///
    /// Only meaningful causally: with bidirectional attention an earlier
    /// position's output depends on later ones, so nothing is reusable.
    fn forward_cached(&self, x: Tensor<B, 3>, cache: &mut LayerKvCache<B>) -> Tensor<B, 3> {
        let m = x.dims()[1];
        let q = self.split_heads(self.query.forward(x.clone()));
        let k_new = self.split_heads(self.key.forward(x.clone()));
        let v_new = self.split_heads(self.value.forward(x));

        let (k, v) = match (cache.keys.take(), cache.values.take()) {
            (Some(pk), Some(pv)) => (
                Tensor::cat(vec![pk, k_new], 2),
                Tensor::cat(vec![pv, v_new], 2),
            ),
            _ => (k_new, v_new),
        };
        cache.keys = Some(k.clone());
        cache.values = Some(v.clone());

        let total = k.dims()[2];
        let past = total - m;

        // Query j sits at absolute position `past + j`, so it may attend to any
        // key up to that index. The mask is rectangular, not triangular: the
        // cached prefix is entirely in the past and never masked.
        let mut scores = q.matmul(k.swap_dims(2, 3)); // [b, heads, m, total]
        let device = scores.device();
        scores = scores + causal_mask_offset::<B>(m, total, past, &device);

        let probs = self.attn_dropout.forward(softmax(scores, 3));
        let ctx = probs.matmul(v).swap_dims(1, 2);

        let [b, n, _, _] = ctx.dims();
        let ctx = ctx.reshape([b, n, self.num_heads * self.head_dim]);
        self.output_dropout.forward(self.dense.forward(ctx))
    }
}

/// Cached keys and values for one attention layer, `[b, heads, positions, dim]`.
///
/// Empty until the first cached forward pass; a cache is bound to the sequence
/// that filled it, so decoding a new prompt needs a fresh one (or
/// [`Self::clear`]).
#[derive(Debug, Clone)]
pub struct LayerKvCache<B: Backend> {
    keys: Option<Tensor<B, 4>>,
    values: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for LayerKvCache<B> {
    fn default() -> Self {
        Self { keys: None, values: None }
    }
}

impl<B: Backend> LayerKvCache<B> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Positions currently cached.
    pub fn len(&self) -> usize {
        self.keys.as_ref().map_or(0, |k| k.dims()[2])
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget everything, so the cache can be reused for another sequence.
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
    }
}

/// Additive causal mask `[1, 1, m, total]` for `m` new queries appended after
/// `offset` cached positions.
///
/// Query `j` is at absolute position `offset + j` and may attend to key `i`
/// exactly when `i <= offset + j`. With `offset == 0` and `total == m` this is
/// [`causal_mask`].
pub(crate) fn causal_mask_offset<B: Backend>(
    m: usize,
    total: usize,
    offset: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut values = Vec::with_capacity(m * total);
    for query in 0..m {
        for key in 0..total {
            values.push(if key <= offset + query { 0.0f32 } else { f32::NEG_INFINITY });
        }
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([1, 1, m, total])
}

/// Additive causal mask `[1, 1, n, n]`: `0` on and below the diagonal,
/// `-inf` above it.
///
/// Built on the host and broadcast over batch and heads. `n` is the sequence
/// length, which for the image path is fixed by the patch geometry and for the
/// language path is the context window.
pub(crate) fn causal_mask<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut values = Vec::with_capacity(n * n);
    for query in 0..n {
        for key in 0..n {
            values.push(if key <= query { 0.0f32 } else { f32::NEG_INFINITY });
        }
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([1, 1, n, n])
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
}

impl<B: Backend> RouterAux<B> {
    /// Sum two layers' contributions.
    pub fn combine(self, other: Self) -> Self {
        Self { balance: self.balance + other.balance, z: self.z + other.z }
    }
}

impl<B: Backend> FeedForward<B> {
    /// Returns the transformed states and, for sparse layers, the auxiliary
    /// losses that must be added to the training objective.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>) {
        match self {
            Self::Dense(mlp) => (mlp.forward(x), None),
            Self::Sparse(moe) => {
                let out = moe.forward(x, conditioning.clone());
                let z = out.z_loss.mul_scalar(moe.z_level() as f32);
                (out.output, Some(RouterAux { balance: out.balance, z }))
            }
            Self::Hierarchical(mosme) => {
                let out = mosme.forward(x, conditioning.clone());
                let z = out.balance.z_loss.clone().mul_scalar(mosme.z_level() as f32);
                (out.output, Some(RouterAux { balance: out.balance.total.clone(), z }))
            }
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
    layernorm_before: LayerNorm<B>,
    layernorm_after: LayerNorm<B>,
    ada_ln: AdaLN<B>,
    /// Whether this layer attends causally. Set from
    /// [`ViTDiTConfig::causal`]; `false` for the image path, whose tokens are
    /// patches with no ordering to respect.
    causal: bool,
}

impl<B: Backend> DbLayer<B> {
    pub(crate) fn new(config: &ViTDiTConfig, layer_idx: usize, device: &B::Device) -> Self {
        let h = config.hidden_size;
        // The MoE router is conditioned on the adaLN vector, which is a pure
        // function of sigma -- that is what makes the routing noise-aware
        // (item 6.4) without any extra plumbing.
        let mlp = match (&config.mosme, config.moe) {
            // Hierarchical wins: it is the strict generalization.
            (Some(mosme), _) if mosme.applies_to(layer_idx) => {
                let cfg = crate::mosme::MosmeConfig::new(
                    h,
                    config.cond_hidden_size,
                    mosme.spec.clone(),
                )
                .with_intermediate_size(config.intermediate_size);
                FeedForward::Hierarchical(crate::mosme::MosmeFeedForward::new(&cfg, device))
            }
            (_, Some(moe)) if moe.applies_to(layer_idx) => {
                let cfg = crate::moe::MoEConfig::new(h, config.cond_hidden_size, moe.num_experts)
                    .with_z_level(moe.z_level)
                    .with_top_k(moe.top_k)
                    .with_intermediate_size(config.intermediate_size);
                FeedForward::Sparse(crate::moe::MoELayer::new(&cfg, device))
            }
            _ => FeedForward::Dense(Mlp::new(
                h,
                config.intermediate_size,
                config.hidden_dropout_prob,
                device,
            )),
        };
        Self {
            attention: Attention::new(
                h,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
                config.hidden_dropout_prob,
                device,
            ),
            mlp,
            layernorm_before: LayerNormConfig::new(h)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            layernorm_after: LayerNormConfig::new(h)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            ada_ln: AdaLN::new(config.cond_hidden_size, 6 * h, device),
            causal: config.causal,
        }
    }

    pub(crate) fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>) {
        let causal = self.causal;
        self.forward_with(hidden_states, conditioning, |attn, normed| {
            attn.forward(normed, causal)
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
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>) {
        assert!(
            self.causal,
            "a KV cache is only sound for causal attention: with bidirectional \
             attention an earlier position's output depends on later ones"
        );
        self.forward_with(hidden_states, conditioning, |attn, normed| {
            attn.forward_cached(normed, cache)
        })
    }

    fn forward_with<F>(
        &self,
        hidden_states: Tensor<B, 3>,
        conditioning: &Tensor<B, 2>,
        attend: F,
    ) -> (Tensor<B, 3>, Option<RouterAux<B>>)
    where
        F: FnOnce(&Attention<B>, Tensor<B, 3>) -> Tensor<B, 3>,
    {
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
        let normed = modulate(self.layernorm_before.forward(hidden_states), shift_msa, scale_msa);
        let attended = attend(&self.attention, normed);
        let hidden_states = attended * gate_msa + residual;

        // MLP branch.
        let layer_output = modulate(
            self.layernorm_after.forward(hidden_states.clone()),
            shift_mlp,
            scale_mlp,
        );
        let (layer_output, balance_loss) = self.mlp.forward(layer_output, conditioning);
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
    final_layernorm: LayerNorm<B>,
}

impl<B: Backend> ViTDiTModel<B> {
    pub fn new(config: &ViTDiTConfig, device: &B::Device) -> Self {
        Self {
            embeddings: ViTDiTEmbeddings::new(config, device),
            time_embedder: TimestepEmbedder::new(
                config.cond_hidden_size,
                config.frequency_embedding_size,
                device,
            ),
            layers: (0..config.num_hidden_layers)
                .map(|idx| DbLayer::new(config, idx, device))
                .collect(),
            final_layernorm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
        }
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
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
        for i in range.start..range.end.min(self.layers.len()) {
            let (states, aux) = self.layers[i].forward(hidden_states, conditioning);
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
    pub fn modulated_cls(&self, model_out: Tensor<B, 3>, conditioning: Tensor<B, 2>) -> Tensor<B, 2> {
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
    pub fn new(config: &ViTDiTConfig, device: &B::Device) -> Self {
        Self {
            vit: ViTDiTModel::new(config, device),
            head: DbOutputHead::new(config, device),
        }
    }

    /// Apply the DiT-specific initialization from `_init_dit`:
    ///
    /// - label embedding table ~ N(0, initializer_range^2)
    /// - timestep MLP weights ~ N(0, initializer_range^2)
    /// - all adaLN modulation linears zeroed
    /// - classifier weight and bias zeroed
    pub fn with_dit_init(self, config: &ViTDiTConfig, device: &B::Device) -> Self {
        use burn::module::Param;

        let std = config.initializer_range;
        let mut rec = self.into_record();

        // Label embedding table.
        {
            let shape = rec.vit.embeddings.label_embeddings.weight.shape();
            rec.vit.embeddings.label_embeddings.weight =
                Param::from_tensor(Tensor::random(shape, Distribution::Normal(0.0, std), device));
        }
        // Timestep embedder MLP weights (biases keep their default init,
        // mirroring nn.init.normal_ which only touches weights).
        for lin in [&mut rec.vit.time_embedder.linear_1, &mut rec.vit.time_embedder.linear_2] {
            let shape = lin.weight.shape();
            lin.weight =
                Param::from_tensor(Tensor::random(shape, Distribution::Normal(0.0, std), device));
        }
        // Zero all adaLN modulation linears (DiT zero-init trick).
        for layer in rec.vit.layers.iter_mut() {
            zero_linear_params(&mut layer.ada_ln.linear);
        }
        zero_linear_params(&mut rec.head.ada_ln.linear);
        // Zero the classifier.
        zero_linear_params(&mut rec.head.classifier);

        Self::new(config, device).load_record(rec)
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
        let out = self.vit.forward_block(layer_indices, pixel_values, noisy_embeds, timesteps);
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
        let out = self.vit.forward_block(layer_indices, pixel_values, noisy_embeds, timesteps);
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn tiny_config() -> ViTDiTConfig {
        ViTDiTConfig::tiny(10)
    }

    #[test]
    fn test_forward_shapes() {
        let device = Default::default();
        let cfg = tiny_config();
        let model = ViTDiTForImageClassification::new(&cfg, &device);

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
        let model = ViTDiTForImageClassification::new(&cfg, &device);

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
        let model = ViTDiTForImageClassification::new(&cfg, &device);
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
        let moe = MoeTrunkConfig { num_experts: 4, top_k: 2, every_n_layers: 2, z_level: 1e-3 };

        // Placement is arithmetic, so check it directly before building
        // anything: every second layer, i.e. layers 1 and 3 of 4.
        assert_eq!(
            (0..4).filter(|i| moe.applies_to(*i)).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(moe.num_sparse_layers(4), 2);
        assert_eq!(MoeTrunkConfig { every_n_layers: 1, ..moe }.num_sparse_layers(4), 4);

        let cfg = ViTDiTConfig::tiny(10).with_moe(moe);
        let model = ViTDiTForImageClassification::<B>::new(&cfg, &device);

        let pixels = Tensor::<B, 4>::ones([2, 3, 32, 32], &device);
        let zt = Tensor::<B, 2>::ones([2, 32], &device);
        let t = Tensor::<B, 1>::zeros([2], &device);

        // A span containing no sparse layer reports no auxiliary loss...
        let dense_span = model.vit().forward_block(0..1, pixels.clone(), zt.clone(), t.clone());
        assert!(dense_span.balance_loss.is_none(), "layer 0 is dense");

        // ...and one containing sparse layers reports a finite, positive one.
        let sparse_span = model.vit().forward_block(0..4, pixels.clone(), zt.clone(), t.clone());
        let aux: f32 = sparse_span
            .balance_loss
            .map(|a| a.balance)
            .expect("layers 1 and 3 are sparse")
            .into_scalar();
        assert!(aux.is_finite() && aux > 0.0, "balance loss must be positive: {aux}");
        // Two sparse layers, each contributing at least the uniform minimum 1.
        assert!(aux >= 2.0 - 1e-4, "two sparse layers must each contribute >= 1: {aux}");
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
        let model = ViTDiTForImageClassification::<B>::new(&cfg, &device);

        let pixels = Tensor::<B, 4>::ones([2, 3, 32, 32], &device);
        let zt = Tensor::<B, 2>::ones([2, 32], &device);
        let t = Tensor::<B, 1>::zeros([2], &device);

        // A span with no hierarchical layer reports no auxiliary loss...
        let dense = model.vit().forward_block(0..1, pixels.clone(), zt.clone(), t.clone());
        assert!(dense.balance_loss.is_none(), "layer 0 is dense");

        // ...and one that includes them reports a finite, positive one. Each
        // hierarchical layer contributes a box term and an expert term, both
        // at least 1 on the diagonal, so two layers give at least 4.
        let sparse = model.vit().forward_block(0..4, pixels.clone(), zt.clone(), t.clone());
        let aux: f32 = sparse.balance_loss.expect("layers 1 and 3 are hierarchical").balance.into_scalar();
        assert!(aux.is_finite() && aux > 0.0, "balance loss must be positive: {aux}");
        assert!(aux >= 2.0, "two hierarchical layers must each contribute: {aux}");

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
            .with_moe(MoeTrunkConfig { num_experts: 4, top_k: 1, every_n_layers: 1, z_level: 1e-3 })
            .with_mosme(MosmeTrunkConfig::new(MosmeSpec::flat(2)).with_every_n_layers(1));
        let model = ViTDiTForImageClassification::<B>::new(&cfg, &device);

        // A single-box hierarchical layer reports a box loss of exactly 1 on
        // top of its expert loss, which a flat layer would not.
        let out = model.vit().forward_block(
            0..1,
            Tensor::<B, 4>::ones([1, 3, 32, 32], &device),
            Tensor::<B, 2>::ones([1, 32], &device),
            Tensor::<B, 1>::zeros([1], &device),
        );
        let aux: f32 = out.balance_loss.expect("layer 0 is hierarchical").balance.into_scalar();
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
        let attention = Attention::<B>::new(hidden, heads, 0.0, 0.0, &device);

        let base = Tensor::<B, 3>::random(
            [1, seq, hidden],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let reference = attention.forward(base.clone(), causal);

        let tail = base.clone().narrow(1, seq - 1, 1) + 10.0;
        let perturbed = Tensor::cat(vec![base.narrow(1, 0, seq - 1), tail], 1);
        let changed = attention.forward(perturbed, causal);

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
                    assert!(v.is_infinite() && v < 0.0, "({query},{key}) should be masked");
                }
            }
        }
    }

    #[test]
    fn test_dit_init_zero_logits_and_gates() {
        let device = Default::default();
        let cfg = tiny_config();
        let model = ViTDiTForImageClassification::new(&cfg, &device).with_dit_init(&cfg, &device);

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
