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

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // The 1/sqrt(head_dim) scale is folded into the Q projection weights.
        let q = self.split_heads(self.query.forward(x.clone()));
        let k = self.split_heads(self.key.forward(x.clone()));
        let v = self.split_heads(self.value.forward(x));

        // Scaled dot-product attention over all heads.
        let scores = q.matmul(k.swap_dims(2, 3)); // [b, heads, n, n]
        let probs = self.attn_dropout.forward(softmax(scores, 3));
        let ctx = probs.matmul(v).swap_dims(1, 2); // [b, n, heads, head_dim]

        let [b, n, _, _] = ctx.dims();
        let ctx = ctx.reshape([b, n, self.num_heads * self.head_dim]);
        self.output_dropout.forward(self.dense.forward(ctx))
    }
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

/// One transformer layer with adaLN-zero conditioning (`ViTDiTLayer`).
#[derive(Module, Debug)]
struct DbLayer<B: Backend> {
    attention: Attention<B>,
    mlp: Mlp<B>,
    layernorm_before: LayerNorm<B>,
    layernorm_after: LayerNorm<B>,
    ada_ln: AdaLN<B>,
}

impl<B: Backend> DbLayer<B> {
    fn new(config: &ViTDiTConfig, device: &B::Device) -> Self {
        let h = config.hidden_size;
        Self {
            attention: Attention::new(
                h,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
                config.hidden_dropout_prob,
                device,
            ),
            mlp: Mlp::new(h, config.intermediate_size, config.hidden_dropout_prob, device),
            layernorm_before: LayerNormConfig::new(h)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            layernorm_after: LayerNormConfig::new(h)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            ada_ln: AdaLN::new(config.cond_hidden_size, 6 * h, device),
        }
    }

    fn forward(&self, hidden_states: Tensor<B, 3>, conditioning: &Tensor<B, 2>) -> Tensor<B, 3> {
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
        let attended = self.attention.forward(normed);
        let hidden_states = attended * gate_msa + residual;

        // MLP branch.
        let layer_output = modulate(
            self.layernorm_after.forward(hidden_states.clone()),
            shift_mlp,
            scale_mlp,
        );
        let layer_output = self.mlp.forward(layer_output);
        layer_output * gate_mlp + hidden_states
    }
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
                .map(|_| DbLayer::new(config, device))
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
    ) -> Tensor<B, 3> {
        for i in range.start..range.end.min(self.layers.len()) {
            hidden_states = self.layers[i].forward(hidden_states, conditioning);
        }
        self.final_layernorm.forward(hidden_states)
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
        let hidden = self.run_layers(layer_indices, emb, &cond);
        BlockOutput {
            last_hidden_state: hidden,
            conditioning: cond,
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
        ViTDiTConfig {
            image_size: 32,
            patch_size: 16, // few patches => fast tests
            in_channels: 3,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            num_labels: 10,
            cond_hidden_size: 8,
            frequency_embedding_size: 16,
        }
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
