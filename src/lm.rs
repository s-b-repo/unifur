//! Causal language modeling (roadmap Phase 19).
//!
//! The trunk is **the same `DbLayer` the image path uses** — adaLN-zero
//! conditioning, optional MoE, optional boxes of specialized micro experts —
//! with two things changed:
//!
//! 1. attention is masked so position `i` cannot see `i + 1`
//!    ([`crate::vit::ViTDiTConfig::causal`]), and
//! 2. the input is a token embedding rather than a patch convolution.
//!
//! Reusing the layer rather than writing a second one is deliberate: a parallel
//! implementation would drift from the first, and every capability built for the
//! image trunk — expert boxes, balance losses, block-wise spans — would have to
//! be ported twice and verified twice.
//!
//! # Weight tying is structural
//!
//! There is no output projection. Logits are `h @ E^T` against the token
//! embedding table itself, so the vocabulary is paid for **once**. That is not
//! only a parameter saving: it means the output head cannot drift away from the
//! input embedding, which is the failure an untied head is prone to when the
//! vocabulary is large relative to the data.
//!
//! # Block-wise training carries over
//!
//! [`LanguageModel::forward_span`] runs a contiguous layer window, exactly as
//! `denoise_span` does for images. So the block-wise objectives — and the
//! gradient routing that comes free with them — apply here without change.

use serde::{Deserialize, Serialize};
use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig},
    tensor::{
        activation::{log_softmax, softmax},
        backend::Backend,
        Distribution, Int, Tensor,
    },
};
use rand::Rng;

use crate::{
    hybrid::{AttentionMode, AttentionSchedule, LayerState, PositionKind},
    planner::{Budget, LookaheadDecoder},
    tokenizer::{Special, VOCAB_SIZE},
    vit::{DbLayer, LayerCarry, TimestepEmbedder, ViTDiTConfig},
};

/// Unlikelihood penalty on labeled targets (roadmap Phase 24).
///
/// A target token carrying a label is a **negative** example: the model is
/// charged for the probability it assigns to it rather than rewarded. The term
/// is Welleck et al.'s unlikelihood, `-log(1 - p)`, not a negative weight on
/// the cross-entropy. The distinction is the whole design:
///
/// - `-w * (-log p) = w * log p` is **unbounded below**. The model can drive
///   the loss to `-inf` by making one bad token impossible, and that one term
///   then dominates every real target in the batch.
/// - `-log(1 - p)` is `0` when the bad token is impossible and grows without
///   bound only as `p -> 1`. It rewards nothing; it stops charging once the
///   pattern is gone.
///
/// `epsilon` floors `1 - p` so the term is finite even for a target the model
/// is certain of: the largest single charge is `-ln(epsilon)`.
///
/// A penalized target is also **removed from the likelihood term**. Charging
/// and rewarding the same token would leave the gradient at whichever term is
/// currently larger, which is a tug of war rather than a signal. With
/// `alpha = 0` nothing is removed either: the objective is then exactly the
/// plain loss, and the flagged targets are only measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Unlikelihood {
    /// Coefficient on the penalty. `0` disables the charge while the metrics
    /// are still reported, which is how a plain run measures what it is
    /// learning.
    pub alpha: f32,
    /// Floor on `1 - p` inside the logarithm.
    pub epsilon: f32,
}

impl Default for Unlikelihood {
    fn default() -> Self {
        Self { alpha: 1.0, epsilon: 1e-6 }
    }
}

impl Unlikelihood {
    pub fn new(alpha: f32) -> Self {
        Self { alpha, ..Self::default() }
    }

    /// Metrics only, no charge.
    pub fn off() -> Self {
        Self::new(0.0)
    }

    pub fn is_off(&self) -> bool {
        self.alpha == 0.0
    }

    /// The most one token can contribute: `-ln(epsilon)`.
    pub fn ceiling(&self) -> f32 {
        -self.epsilon.ln()
    }
}

/// The per-token unlikelihood term `-log(1 - p)` from log-probabilities,
/// floored at `-log(epsilon)`.
pub fn unlikelihood<B: Backend>(log_probs: Tensor<B, 1>, epsilon: f32) -> Tensor<B, 1> {
    let p = log_probs.exp();
    p.neg().add_scalar(1.0).clamp_min(epsilon).log().neg()
}

/// Per-token penalty weights from label bytes, via a label -> weight table
/// such as [`crate::antipattern::LabelManifest::weight_table`].
///
/// Returns `[batch, n]` aligned with the tokens; a clean token weighs `0`.
pub fn label_weights<B: Backend>(
    labels: &[Vec<u8>],
    table: &[f32; 256],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch = labels.len();
    let n = labels.first().map_or(0, Vec::len);
    let flat: Vec<f32> = labels
        .iter()
        .flat_map(|row| {
            assert_eq!(row.len(), n, "every label row must be the window length");
            row.iter().map(|l| table[usize::from(*l)])
        })
        .collect();
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([batch, n])
}

/// Shape of a causal language model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmConfig {
    pub vocab_size: usize,
    /// Longest sequence the position table covers.
    pub context: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub cond_hidden_size: usize,
    pub frequency_embedding_size: usize,
    pub dropout: f64,
    pub layer_norm_eps: f64,
    pub initializer_range: f64,
    /// How many blocks the layers are partitioned into for block-wise training.
    pub num_blocks: usize,
    pub moe: Option<crate::vit::MoeTrunkConfig>,
    pub mosme: Option<crate::vit::MosmeTrunkConfig>,
    /// One attention mode per layer (roadmap Phase 25); `None` is dense
    /// everywhere, the Phase 19 trunk.
    #[serde(default)]
    pub attention: Option<AttentionSchedule>,
    /// How positions enter the model (roadmap 25.2). With anything but the
    /// learned table the sequence length is unbounded.
    #[serde(default)]
    pub positions: PositionKind,
    /// Width of the per-token routing state carried through the layers
    /// (roadmap 25.4); `0` is none.
    #[serde(default)]
    pub routing_state: usize,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            context: 256,
            hidden_size: 256,
            num_layers: 8,
            num_heads: 8,
            intermediate_size: 1024,
            cond_hidden_size: 256 / 6,
            frequency_embedding_size: 256,
            dropout: 0.0,
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
            num_blocks: 4,
            moe: None,
            mosme: None,
            attention: None,
            positions: PositionKind::Learned,
            routing_state: 0,
        }
    }
}

impl LmConfig {
    /// A small configuration for tests and smoke runs.
    pub fn tiny() -> Self {
        Self {
            context: 16,
            hidden_size: 32,
            num_layers: 4,
            num_heads: 4,
            intermediate_size: 64,
            cond_hidden_size: 8,
            frequency_embedding_size: 16,
            num_blocks: 2,
            ..Self::default()
        }
    }

    pub fn with_mosme(mut self, mosme: crate::vit::MosmeTrunkConfig) -> Self {
        self.mosme = Some(mosme);
        self
    }

    /// Assign an attention mode to every layer (roadmap Phase 25).
    pub fn with_attention(mut self, schedule: AttentionSchedule) -> Self {
        assert_eq!(
            schedule.num_layers(),
            self.num_layers,
            "the schedule covers {} layers, the model has {}",
            schedule.num_layers(),
            self.num_layers
        );
        self.attention = Some(schedule);
        self
    }

    /// Choose how positions enter the model (roadmap 25.2).
    pub fn with_positions(mut self, positions: PositionKind) -> Self {
        self.positions = positions;
        self
    }

    /// Carry a per-token routing state of this width (roadmap 25.4).
    pub fn with_routing_state(mut self, size: usize) -> Self {
        self.routing_state = size;
        self
    }

    /// The attention schedule, dense when none was set.
    pub fn attention_schedule(&self) -> AttentionSchedule {
        self.attention.clone().unwrap_or_else(|| AttentionSchedule::dense(self.num_layers))
    }

    /// What one token costs in this trunk when it sees `positions` keys:
    /// active parameters, FLOPs and decode-time state per layer, counted
    /// from the shapes (roadmap 25.8, [`crate::cost`]).
    pub fn cost(&self, positions: usize) -> crate::cost::TrunkCost {
        let schedule = self.attention_schedule();
        let state = self.routing_state;
        let router_in = |route_on_tokens: bool| {
            (if route_on_tokens { self.cond_hidden_size + self.hidden_size } else { self.cond_hidden_size }) + state
        };
        let layers = (0..self.num_layers)
            .map(|idx| {
                let attention = crate::cost::attention_cost(schedule.mode(idx), self.hidden_size, self.num_heads, positions);
                let ffn = match (&self.mosme, self.moe) {
                    (Some(mosme), _) if mosme.applies_to(idx) => crate::cost::moe_cost(
                        self.hidden_size,
                        self.intermediate_size,
                        mosme.spec.num_experts(),
                        mosme.spec.top_box.max(1) * mosme.spec.top_expert.max(1),
                        router_in(mosme.spec.route_on_tokens),
                        None,
                    ),
                    (_, Some(moe)) if moe.applies_to(idx) => crate::cost::moe_cost(
                        self.hidden_size,
                        self.intermediate_size,
                        moe.num_experts,
                        moe.top_k,
                        router_in(true),
                        None,
                    ),
                    _ => crate::cost::dense_mlp_cost(self.hidden_size, self.intermediate_size),
                };
                let routing = crate::cost::LayerCost {
                    active_params: if state > 0 { (self.hidden_size + state) * state + state + 2 * self.hidden_size } else { 0 },
                    flops: if state > 0 { 2.0 * ((self.hidden_size + state) * state) as f64 } else { 0.0 },
                    keys_read: 0,
                    state_floats: 0,
                };
                attention.plus(ffn).plus(routing)
            })
            .collect();
        crate::cost::TrunkCost { layers }
    }

    /// One line naming the architecture, for banners and records.
    pub fn describe(&self) -> String {
        format!(
            "layers={} hidden={} context={} attention={} positions={} routing_state={}",
            self.num_layers,
            self.hidden_size,
            self.context,
            self.attention_schedule().summary(),
            self.positions.name(),
            self.routing_state
        )
    }

    /// Layers per block.
    pub fn layers_per_block(&self) -> usize {
        self.num_layers / self.num_blocks.max(1)
    }

    /// The trunk configuration this shares with the image path. `causal` is
    /// always true here — that is what makes it a language model.
    fn trunk(&self) -> ViTDiTConfig {
        ViTDiTConfig {
            // Unused by the LM path (there is no patch convolution), but the
            // struct is shared, so they must be internally consistent.
            image_size: 32,
            patch_size: 16,
            in_channels: 3,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_layers,
            num_attention_heads: self.num_heads,
            layer_norm_eps: self.layer_norm_eps,
            hidden_dropout_prob: self.dropout,
            attention_probs_dropout_prob: self.dropout,
            initializer_range: self.initializer_range,
            num_labels: self.vocab_size,
            cond_hidden_size: self.cond_hidden_size,
            frequency_embedding_size: self.frequency_embedding_size,
            moe: self.moe,
            mosme: self.mosme.clone(),
            causal: true,
            attention: self.attention.clone(),
            rotary: self.positions == PositionKind::Rotary,
            routing_state: self.routing_state,
        }
    }
}

/// Output of a trunk pass.
#[derive(Debug, Clone)]
pub struct LmOutput<B: Backend> {
    /// `[b, n, vocab]`.
    pub logits: Tensor<B, 3>,
    /// Summed MoE / MoSME balance loss over the executed span, if any.
    /// Auxiliary routing losses of the executed span; `None` for a dense
    /// trunk. Balance and z-loss are kept apart — see
    /// [`crate::vit::RouterAux`].
    pub balance_loss: Option<crate::vit::RouterAux<B>>,
}

/// A causal language model over the shared trunk.
#[derive(Module, Debug)]
pub struct LanguageModel<B: Backend> {
    token_embedding: Embedding<B>,
    /// `[1, context, hidden]`, learned; `None` under rotary or no positions
    /// (roadmap 25.2), where nothing bounds the sequence length.
    position_embedding: Option<Param<Tensor<B, 3>>>,
    time_embedder: TimestepEmbedder<B>,
    layers: Vec<DbLayer<B>>,
    final_norm: LayerNorm<B>,
    context: usize,
    vocab_size: usize,
    layers_per_block: usize,
    num_blocks: usize,
    /// Whether the position table bounds the sequence length.
    bounded: bool,
}

impl<B: Backend<FloatElem = f32>> LanguageModel<B> {
    pub fn new(config: &LmConfig, device: &B::Device) -> Self {
        assert!(
            config.num_layers % config.num_blocks.max(1) == 0,
            "num_layers ({}) must be divisible by num_blocks ({})",
            config.num_layers,
            config.num_blocks
        );
        if let Some(schedule) = &config.attention {
            assert_eq!(
                schedule.num_layers(),
                config.num_layers,
                "the attention schedule covers {} layers, the model has {}",
                schedule.num_layers(),
                config.num_layers
            );
        }
        let trunk = config.trunk();

        let position = config.positions.is_bounded().then(|| {
            Param::from_tensor(
                Tensor::<B, 2>::random(
                    [1, config.context * config.hidden_size],
                    Distribution::Normal(0.0, config.initializer_range),
                    device,
                )
                .reshape([1, config.context, config.hidden_size]),
            )
        });

        Self {
            // Burn's default embedding initializer is N(0, 1), which with a
            // *tied* output head gives logits of scale ~sqrt(hidden) and an
            // untrained model that is confidently wrong: the initial loss lands
            // far above ln(vocab) and the first phase of training is spent
            // undoing it. The image path already initializes its label table at
            // `initializer_range` for the same reason.
            token_embedding: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .with_initializer(burn::module::Initializer::Normal {
                    mean: 0.0,
                    std: config.initializer_range,
                })
                .init(device),
            position_embedding: position,
            time_embedder: TimestepEmbedder::new(
                config.cond_hidden_size,
                config.frequency_embedding_size,
                device,
            ),
            layers: (0..config.num_layers)
                .map(|idx| DbLayer::new(&trunk, idx, device))
                .collect(),
            final_norm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            context: config.context,
            vocab_size: config.vocab_size,
            layers_per_block: config.layers_per_block(),
            num_blocks: config.num_blocks.max(1),
            bounded: config.positions.is_bounded(),
        }
    }

    pub fn context(&self) -> usize {
        self.context
    }

    /// Whether a position table bounds the sequence length (roadmap 25.2).
    /// When it does not, cached decoding may run past `context`.
    pub fn positions_bounded(&self) -> bool {
        self.bounded
    }

    /// The attention mode of every layer (roadmap 25.1).
    pub fn attention_modes(&self) -> Vec<AttentionMode> {
        self.layers.iter().map(DbLayer::attention_mode).collect()
    }

    /// Whether the trunk carries a routing state (roadmap 25.4).
    pub fn has_routing_state(&self) -> bool {
        self.layers.iter().any(DbLayer::has_routing_state)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Width of the residual stream.
    pub fn hidden_size(&self) -> usize {
        self.embedding_weight().dims()[1]
    }

    /// Contiguous layer window owned by `block_idx`.
    pub fn layer_range(&self, block_idx: usize) -> std::ops::Range<usize> {
        assert!(block_idx < self.num_blocks, "block {block_idx} out of range");
        let start = block_idx * self.layers_per_block;
        start..start + self.layers_per_block
    }

    /// The token embedding table `[vocab, hidden]`.
    ///
    /// Also the output projection: see the module docs on weight tying.
    pub fn embedding_weight(&self) -> Tensor<B, 2> {
        self.token_embedding.weight.val()
    }

    /// Token ids `[b, n]` at absolute positions `offset..offset + n` to hidden
    /// states `[b, n, hidden]`.
    fn embed_at(&self, tokens: Tensor<B, 2, Int>, offset: usize) -> Tensor<B, 3> {
        let [_, n] = tokens.dims();
        if self.bounded {
            assert!(
                offset + n <= self.context,
                "sequence of {} exceeds the {} the position table covers",
                offset + n,
                self.context
            );
        }
        let embedded = self.token_embedding.forward(tokens);
        match &self.position_embedding {
            Some(table) => embedded + table.val().narrow(1, offset, n),
            None => embedded,
        }
    }

    /// Token ids `[b, n]` to hidden states `[b, n, hidden]`.
    fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed_at(tokens, 0)
    }

    /// Run a contiguous span of layers, mirroring `denoise_span` on the image
    /// path so block-wise training carries over unchanged.
    pub fn forward_span(
        &self,
        tokens: Tensor<B, 2, Int>,
        span: std::ops::Range<usize>,
    ) -> LmOutput<B> {
        self.forward_span_states(tokens, span, None).0
    }

    /// The conditioning vector every layer sees on the language path: the
    /// timestep-zero embedding, `[b, cond]`.
    fn conditioning(&self, b: usize, device: &B::Device) -> Tensor<B, 2> {
        crate::vit::silu_public(self.time_embedder.forward(Tensor::<B, 1>::zeros([b], device)))
    }

    /// The adaLN gates of every layer under the language conditioning
    /// (roadmap 31.1): what a residual writer's output is scaled by before
    /// it reaches the stream. Needed to ablate a direction correctly.
    pub fn layer_gates(&self, device: &B::Device) -> Vec<crate::ablation::LayerGates> {
        let cond = self.conditioning(1, device);
        self.layers
            .iter()
            .map(|layer| {
                let (msa, mlp) = layer.gates(&cond);
                crate::ablation::LayerGates {
                    attention: msa.into_data().convert::<f32>().iter::<f32>().collect(),
                    mlp: mlp.into_data().convert::<f32>().iter::<f32>().collect(),
                }
            })
            .collect()
    }

    /// [`Self::forward_span`] that also returns every layer's output (the
    /// residual stream before the final norm), and optionally projects a
    /// direction out of the stream after every layer (roadmap 31.1).
    pub fn forward_span_states(
        &self,
        tokens: Tensor<B, 2, Int>,
        span: std::ops::Range<usize>,
        ablate: Option<&Tensor<B, 1>>,
    ) -> (LmOutput<B>, Vec<Tensor<B, 3>>) {
        let device = tokens.device();
        let b = tokens.dims()[0];

        // A plain LM has no noise level; timestep zero keeps the conditioning
        // path identical to the image trunk's rather than special-casing it.
        let cond = self.conditioning(b, &device);

        let mut hidden = self.embed(tokens);
        if let Some(d) = ablate {
            hidden = crate::ablation::project_out(hidden, d);
        }
        let mut states = Vec::with_capacity(self.layers.len());
        let mut balance: Option<crate::vit::RouterAux<B>> = None;
        let mut carry = LayerCarry::at(0);
        for i in span.start..span.end.min(self.layers.len()) {
            let (mut next, aux) = self.layers[i].forward(hidden, &cond, &mut carry);
            if let Some(d) = ablate {
                next = crate::ablation::project_out(next, d);
            }
            hidden = next;
            states.push(hidden.clone());
            if let Some(aux) = aux {
                balance = Some(match balance {
                    None => aux,
                    Some(acc) => acc.combine(aux),
                });
            }
        }
        let hidden = self.final_norm.forward(hidden);

        // Tied output projection: logits against the embedding table itself.
        let [bb, n, h] = hidden.dims();
        let logits = hidden
            .reshape([bb * n, h])
            .matmul(self.embedding_weight().transpose())
            .reshape([bb, n, self.vocab_size]);

        (LmOutput { logits, balance_loss: balance }, states)
    }

    /// Every layer.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> LmOutput<B> {
        self.forward_span(tokens, 0..self.layers.len())
    }

    /// Every layer's output for `tokens`, before the final norm.
    pub fn hidden_states(&self, tokens: Tensor<B, 2, Int>) -> Vec<Tensor<B, 3>> {
        self.forward_span_states(tokens, 0..self.layers.len(), None).1
    }

    /// Forward with `direction` projected out of the residual stream after
    /// the embedding and after every layer: inference-time ablation.
    pub fn forward_ablated(&self, tokens: Tensor<B, 2, Int>, direction: &Tensor<B, 1>) -> LmOutput<B> {
        self.forward_span_states(tokens, 0..self.layers.len(), Some(direction)).0
    }

    /// The residual stream at the last position of `ids`, one `[h]` vector
    /// per layer, on the host: the raw material of a behaviour direction.
    pub fn residuals_at_last_position(&self, ids: &[u16], device: &B::Device) -> Vec<Vec<f32>> {
        let ids: Vec<i64> = if ids.is_empty() { vec![i64::from(Special::Bos.id())] } else { ids.iter().map(|t| i64::from(*t)).collect() };
        let start = ids.len().saturating_sub(self.context);
        let window = &ids[start..];
        let n = window.len();
        let tokens = Tensor::<B, 1, Int>::from_ints(window, device).reshape([1, n]);
        self.hidden_states(tokens)
            .into_iter()
            .map(|h| {
                let width = h.dims()[2];
                h.narrow(1, n - 1, 1).reshape([width]).into_data().convert::<f32>().iter::<f32>().collect()
            })
            .collect()
    }

    /// Next-token probabilities after `ids`, `[vocab]` on the device.
    pub fn next_token_probs(&self, ids: &[u16], device: &B::Device) -> Tensor<B, 1> {
        let ids: Vec<i64> = if ids.is_empty() { vec![i64::from(Special::Bos.id())] } else { ids.iter().map(|t| i64::from(*t)).collect() };
        let start = ids.len().saturating_sub(self.context);
        let window = &ids[start..];
        let n = window.len();
        let tokens = Tensor::<B, 1, Int>::from_ints(window, device).reshape([1, n]);
        let logits = self.forward(tokens).logits.narrow(1, n - 1, 1).reshape([self.vocab_size]);
        softmax(logits, 0)
    }

    /// Forward over `tokens`, treating them as a continuation of whatever
    /// `cache` already holds (roadmap 19.6).
    ///
    /// `tokens` are the **new** positions only. The returned logits cover just
    /// those positions — the cached prefix is not recomputed, which is the
    /// entire point.
    ///
    /// # Why this is exact, not an approximation
    ///
    /// A causal model's keys and values at position `i` depend only on tokens
    /// up to `i`. Once those tokens are committed, recomputing them can only
    /// reproduce the same numbers. So the cache is not a speed-for-accuracy
    /// trade: `lm/kv_cache_matches_full_recompute` demands bitwise-comparable
    /// agreement, at a tolerance set by float summation order alone.
    ///
    /// # Panics
    ///
    /// If the accumulated length would exceed the context the position table
    /// covers. Truncation is the caller's decision — silently dropping the
    /// oldest positions would change the conditioning without saying so.
    pub fn forward_cached(&self, tokens: Tensor<B, 2, Int>, cache: &mut KvCache<B>) -> LmOutput<B> {
        let device = tokens.device();
        let [b, m] = tokens.dims();
        let offset = cache.position();
        if self.bounded {
            assert!(
                offset + m <= self.context,
                "cached sequence of {} exceeds the {} the position table covers",
                offset + m,
                self.context
            );
        }
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "the cache was built for a different number of layers"
        );

        let cond = crate::vit::silu_public(
            self.time_embedder.forward(Tensor::<B, 1>::zeros([b], &device)),
        );

        // Positions are absolute: the new tokens sit *after* the cached ones,
        // so they must read the position table (or the rotary angle) at
        // `offset`, not at 0.
        let mut hidden = self.embed_at(tokens, offset);

        let mut balance: Option<crate::vit::RouterAux<B>> = None;
        let mut carry = LayerCarry::at(offset);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let (states, aux) = layer.forward_cached(hidden, &cond, layer_cache, &mut carry);
            hidden = states;
            if let Some(aux) = aux {
                balance = Some(match balance {
                    None => aux,
                    Some(acc) => acc.combine(aux),
                });
            }
        }
        cache.position += m;

        let hidden = self.final_norm.forward(hidden);
        let [bb, n, h] = hidden.dims();
        let logits = hidden
            .reshape([bb * n, h])
            .matmul(self.embedding_weight().transpose())
            .reshape([bb, n, self.vocab_size]);

        LmOutput { logits, balance_loss: balance }
    }

    /// A cache sized for this model.
    pub fn new_cache(&self) -> KvCache<B> {
        KvCache::new(self.layers.len())
    }

    /// Next-token cross-entropy over a span.
    ///
    /// Targets are the inputs shifted left by one, so position `i` predicts
    /// token `i + 1` and the final position has no target. Padding is excluded
    /// from the mean rather than merely zeroed, so a batch that is mostly
    /// padding does not silently report a small loss.
    pub fn next_token_loss(
        &self,
        tokens: Tensor<B, 2, Int>,
        span: std::ops::Range<usize>,
    ) -> (Tensor<B, 1>, LmMetrics) {
        let step = self.objective(tokens, span, None, None, None, None);
        (step.loss, step.metrics)
    }

    /// The training step with everything a trainer needs: the loss, the
    /// metrics, and the per-sparse-layer routing statistics of the span
    /// (roadmap 23.6), in execution order.
    pub fn next_token_step(
        &self,
        tokens: Tensor<B, 2, Int>,
        negatives: Option<(Tensor<B, 2>, Unlikelihood)>,
        span: std::ops::Range<usize>,
    ) -> LmStep<B> {
        self.objective(tokens, span, negatives, None, None, None)
    }

    /// [`Self::next_token_step`] with the two open-weight signals of roadmap
    /// Phase 29: extra negatives proposed by a negative teacher
    /// ([`Self::negative_proposals`]) and distillation toward a weighted
    /// mixture of teachers.
    pub fn next_token_step_full(
        &self,
        tokens: Tensor<B, 2, Int>,
        negatives: Option<(Tensor<B, 2>, Unlikelihood)>,
        extra: Option<ExtraNegatives<B>>,
        distill: Option<Distillation<'_, B>>,
        span: std::ops::Range<usize>,
    ) -> LmStep<B> {
        self.objective(tokens, span, negatives, extra, distill, None)
    }

    /// [`Self::next_token_step_full`] with a direction penalty (roadmap 31.2):
    /// `weight * mean((h_L . d)^2)` at the direction's layer joins the loss.
    /// A weight of zero adds nothing and the loss is the plain one bit for bit.
    pub fn next_token_step_directed(
        &self,
        tokens: Tensor<B, 2, Int>,
        negatives: Option<(Tensor<B, 2>, Unlikelihood)>,
        extra: Option<ExtraNegatives<B>>,
        distill: Option<Distillation<'_, B>>,
        direction: Option<DirectionPenalty<'_, B>>,
        span: std::ops::Range<usize>,
    ) -> LmStep<B> {
        self.objective(tokens, span, negatives, extra, distill, direction)
    }

    /// What a frozen **negative** model would say next (roadmap 29.3): at
    /// every position its arg-max token and, as the weight, `1.0` where it is
    /// at least `confidence` sure **and** that token is not the corpus
    /// target -- the corpus is never contradicted -- else `0.0`. Both
    /// `[b, n - 1]`, aligned with the predicting positions.
    pub fn negative_proposals(
        &self,
        tokens: Tensor<B, 2, Int>,
        confidence: f32,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        let [b, n] = tokens.dims();
        assert!(n >= 2, "proposals need at least two positions");
        let logits = self.forward(tokens.clone()).logits.narrow(1, 0, n - 1).detach();
        let probs = softmax(logits, 2);
        let (max_p, argmax) = probs.max_dim_with_indices(2);
        let argmax = argmax.reshape([b, n - 1]);
        let max_p = max_p.reshape([b, n - 1]);
        let targets = tokens.narrow(1, 1, n - 1);
        let confident = max_p.greater_equal_elem(confidence).float();
        let differs = argmax.clone().equal(targets).bool_not().float();
        (argmax, confident * differs)
    }

    /// Next-token loss with labeled targets **charged** rather than rewarded
    /// (roadmap Phase 24).
    ///
    /// `weights` is `[batch, n]`, aligned with `tokens`: a positive entry at
    /// `[b, i]` makes token `i` a negative target with that weight on its
    /// unlikelihood term, and removes it from the likelihood term. Build it
    /// with [`label_weights`]. With every weight zero this is exactly
    /// [`Self::next_token_loss`], bit for bit.
    pub fn next_token_loss_penalized(
        &self,
        tokens: Tensor<B, 2, Int>,
        weights: Tensor<B, 2>,
        penalty: Unlikelihood,
        span: std::ops::Range<usize>,
    ) -> (Tensor<B, 1>, LmMetrics) {
        let step = self.objective(tokens, span, Some((weights, penalty)), None, None, None);
        (step.loss, step.metrics)
    }

    fn objective(
        &self,
        tokens: Tensor<B, 2, Int>,
        span: std::ops::Range<usize>,
        negatives: Option<(Tensor<B, 2>, Unlikelihood)>,
        extra: Option<ExtraNegatives<B>>,
        distill: Option<Distillation<'_, B>>,
        direction: Option<DirectionPenalty<'_, B>>,
    ) -> LmStep<B> {
        let device = tokens.device();
        let [b, n] = tokens.dims();
        assert!(n >= 2, "next-token loss needs at least two positions");

        // States are only collected when a direction penalty will read them;
        // the plain path is untouched.
        let active_direction = direction.filter(|d| d.weight > 0.0);
        let (out, states) = match active_direction {
            Some(_) => self.forward_span_states(tokens.clone(), span.clone(), None),
            None => (self.forward_span(tokens.clone(), span.clone()), Vec::new()),
        };

        // Drop the last position (no target) and the first target (no input).
        let logits = out.logits.narrow(1, 0, n - 1);
        let targets = tokens.clone().narrow(1, 1, n - 1);

        let flat_logits = logits.reshape([b * (n - 1), self.vocab_size]);
        let flat_targets = targets.clone().reshape([b * (n - 1), 1]);

        let log_probs = log_softmax(flat_logits.clone(), 1);
        let target_log_prob = log_probs.clone().gather(1, flat_targets).squeeze_dim::<1>(1); // [b*(n-1)]
        let nll = -target_log_prob.clone();

        // Mask padding out of both the numerator and the denominator.
        let pad = Tensor::<B, 1, Int>::full([b * (n - 1)], Special::Pad.id() as i64, &device);
        let keep = targets
            .reshape([b * (n - 1)])
            .equal(pad)
            .bool_not()
            .float();
        let counted = keep.clone().sum().clamp_min(1.0);
        let keep_all = keep.clone();

        let (loss, penalized_tokens, penalty, penalized_prob) = match negatives {
            None => ((nll * keep).sum() / counted.clone(), 0, 0.0, 0.0),
            Some((weights, unlikelihood_term)) => {
                assert_eq!(weights.dims(), [b, n], "weights must be shaped like the tokens");
                let w = weights.narrow(1, 1, n - 1).reshape([b * (n - 1)]);
                // A flagged target is measured whether or not it is charged;
                // a padded position is neither.
                let flagged = w.clone().greater_elem(0.0).float() * keep.clone();
                let count = flagged.clone().sum();
                let per_flagged = count.clone().clamp_min(1.0);
                let prob = (target_log_prob.clone().exp() * flagged.clone()).sum() / per_flagged.clone();
                let charge = unlikelihood(target_log_prob, unlikelihood_term.epsilon) * w * flagged.clone();
                let charge_sum = charge.sum();

                // With the penalty off, the objective *is* the plain one: the
                // flagged targets stay in the likelihood and are merely
                // reported on. Removing them while charging nothing would be
                // a third objective -- "never learn these, never unlearn
                // them" -- and it was, briefly: a run with `alpha = 0` showed
                // p(bad) falling to 0.001 while its loss fell to 0.17, which
                // is impossible for a model being trained on those tokens.
                let negative = if unlikelihood_term.is_off() {
                    flagged.zeros_like()
                } else {
                    flagged
                };
                let positive = keep - negative;
                let likelihood = (nll * positive).sum();

                // Both terms share the denominator, so a batch with few
                // negatives is not dominated by them and a batch with none
                // reduces to the plain loss exactly.
                let loss = (likelihood + charge_sum.clone().mul_scalar(unlikelihood_term.alpha))
                    / counted.clone();
                (
                    loss,
                    count.into_scalar() as usize,
                    (charge_sum / per_flagged).into_scalar(),
                    prob.into_scalar(),
                )
            }
        };

        // --- open-weight negatives (roadmap 29.3) ---------------------------
        // Tokens a negative teacher would choose are charged with the same
        // bounded term the labeled negatives use. They are *not* removed from
        // the likelihood: they were never targets. Padded positions charge
        // nothing, whatever the teacher proposed there.
        let (loss, negative_teacher_tokens, negative_teacher_prob) = match extra {
            None => (loss, 0, 0.0),
            Some(extra) => {
                assert_eq!(extra.tokens.dims(), [b, n - 1], "proposals must cover every predicting position");
                assert_eq!(extra.weights.dims(), [b, n - 1], "one weight per proposal");
                let idx = extra.tokens.reshape([b * (n - 1), 1]);
                let lp = log_probs.gather(1, idx).squeeze_dim::<1>(1);
                let w = extra.weights.reshape([b * (n - 1)]) * keep_all.clone();
                let flagged = w.clone().greater_elem(0.0).float();
                let count = flagged.clone().sum();
                let prob = (lp.clone().exp() * flagged).sum() / count.clone().clamp_min(1.0);
                let charge = (unlikelihood(lp, extra.epsilon) * w).sum();
                (
                    loss + charge.mul_scalar(extra.alpha) / counted.clone(),
                    count.into_scalar() as usize,
                    prob.into_scalar(),
                )
            }
        };

        // --- distillation toward a teacher mixture (roadmap 29.2) -----------
        let (loss, distill_loss) = match distill {
            None => (loss, 0.0),
            Some(d) if d.weight > 0.0 && !d.teachers.is_empty() => {
                let teacher_logits: Vec<(Tensor<B, 2>, f64)> = d
                    .teachers
                    .iter()
                    .map(|(teacher, w)| {
                        let t = teacher
                            .forward(tokens.clone())
                            .logits
                            .narrow(1, 0, n - 1)
                            .reshape([b * (n - 1), self.vocab_size])
                            .detach();
                        (t, *w)
                    })
                    .collect();
                let mixture = crate::distill::teacher_mixture(&teacher_logits, d.temperature);
                let t = d.temperature.max(1e-6) as f32;
                let log_p = mixture.clone().clamp_min(1e-30).log();
                let log_q = log_softmax(flat_logits.div_scalar(t), 1);
                let per_row = (mixture * (log_p - log_q)).sum_dim(1).reshape([b * (n - 1)]);
                let kl = (per_row * keep_all).sum() / counted.clone() * (t * t);
                let value: f32 = kl.clone().into_scalar();
                (loss + kl.mul_scalar(d.weight as f32), value)
            }
            Some(_) => (loss, 0.0),
        };

        // --- direction penalty (roadmap 31.2) ---------------------------------
        let (loss, direction_projection) = match active_direction {
            None => (loss, 0.0),
            Some(d) => {
                let local = d.layer.saturating_sub(span.start).min(states.len().saturating_sub(1));
                let h = &states[local];
                let penalty = crate::ablation::projection_penalty(h, d.direction);
                let value: f32 = penalty.clone().into_scalar();
                (loss + penalty.mul_scalar(d.weight as f32), value)
            }
        };

        let value: f32 = loss.clone().into_scalar();
        let routing: Vec<crate::moe::RoutingStats> =
            out.balance_loss.as_ref().map_or_else(Vec::new, |aux| aux.to_host());
        let (_, routing_entropy, _, routing_max_load) =
            crate::moe::RoutingStats::summarize(&routing);
        let metrics = LmMetrics {
            loss: value,
            perplexity: value.exp(),
            tokens_counted: counted.into_scalar() as usize,
            balance_loss: out
                .balance_loss
                .as_ref()
                .map_or(0.0, |aux| aux.balance.clone().into_scalar()),
            penalized_tokens,
            penalty,
            penalized_prob,
            routing_entropy,
            routing_max_load,
            negative_teacher_tokens,
            negative_teacher_prob,
            distill_loss,
            direction_projection,
        };

        // The balance term is scaled; the z-loss already carries its own
        // `z_level` and is added at weight 1, so a stabilizer is not silently
        // attenuated by a routing regularizer's coefficient.
        let loss = match out.balance_loss {
            Some(aux) => loss + aux.balance.mul_scalar(0.01) + aux.z,
            None => loss,
        };
        LmStep { loss, metrics, routing }
    }

    /// Selection biases of every sparse layer that has one, in layer order
    /// (roadmap 23.5).
    pub fn balance_biases(&self) -> Vec<Vec<f32>> {
        self.layers.iter().filter_map(|l| l.balance_bias_values()).collect()
    }

    /// Attach zero selection biases to every sparse layer that lacks one
    /// (roadmap 23.5). Call after loading a checkpoint.
    pub fn ensure_balance_biases(&mut self) {
        for layer in &mut self.layers {
            if let Some(mut sparse) = layer.sparse_mut() {
                sparse.ensure_balance_bias();
            }
        }
    }

    /// Nudge the selection biases of the sparse layers in `span` against the
    /// loads a step on that span produced (roadmap 23.5): `loads` is
    /// [`LmStep::routing`], one entry per sparse layer in execution order.
    pub fn nudge_balance_biases(
        &mut self,
        span: std::ops::Range<usize>,
        loads: &[crate::moe::RoutingStats],
        rate: f32,
    ) {
        let end = span.end.min(self.layers.len());
        let mut nth = 0;
        for layer in &mut self.layers[span.start..end] {
            if let Some(mut sparse) = layer.sparse_mut() {
                if let Some(stats) = loads.get(nth) {
                    sparse.nudge_balance_bias(&stats.load, rate);
                }
                nth += 1;
            }
        }
    }

    /// Continue `prompt` for `max_new` tokens.
    ///
    /// Recomputes the full prefix each step rather than caching keys and
    /// values. That is `O(n^2)` per token instead of `O(n)`, and is the honest
    /// baseline a cache has to be checked against — see roadmap 19.6.
    pub fn generate<R: Rng>(
        &self,
        prompt: &[u16],
        max_new: usize,
        sampling: &Sampling,
        rng: &mut R,
        device: &B::Device,
    ) -> Vec<u16> {
        let mut ids: Vec<u16> = prompt.to_vec();
        if ids.is_empty() {
            ids.push(Special::Bos.id());
        }

        for _ in 0..max_new {
            // Keep only what the position table covers, dropping from the left.
            let start = ids.len().saturating_sub(self.context);
            let window: Vec<i64> = ids[start..].iter().map(|t| *t as i64).collect();
            let n = window.len();

            let tokens = Tensor::<B, 1, Int>::from_ints(window.as_slice(), device).reshape([1, n]);
            let logits = self.forward(tokens).logits.narrow(1, n - 1, 1).reshape([
                1,
                self.vocab_size,
            ]);

            let next = sampling.pick(&logits, rng);
            ids.push(next);
            if next == Special::Eos.id() {
                break;
            }
        }
        ids
    }

    /// [`Self::generate`] with a key/value cache (roadmap 19.6).
    ///
    /// The prompt is absorbed in one pass, then each new token costs a single
    /// position of attention instead of a full re-read of the prefix: `O(n)`
    /// work per token rather than `O(n^2)`. The output is the same sequence
    /// [`Self::generate`] produces from the same seed — that equivalence is
    /// certificate `lm/kv_cache_matches_full_recompute`.
    ///
    /// Generation stops at `<eos>` or when the context window fills, whichever
    /// comes first. Unlike [`Self::generate`] there is no left-truncation
    /// fallback: a cache cannot drop its oldest positions without invalidating
    /// every position embedding after them, so the honest behaviour is to stop.
    pub fn generate_cached<R: Rng>(
        &self,
        prompt: &[u16],
        max_new: usize,
        sampling: &Sampling,
        rng: &mut R,
        device: &B::Device,
    ) -> Vec<u16> {
        let mut ids: Vec<u16> = prompt.to_vec();
        if ids.is_empty() {
            ids.push(Special::Bos.id());
        }
        if self.bounded && ids.len() > self.context {
            ids.drain(..ids.len() - self.context);
        }

        let mut cache = self.new_cache();
        let mut pending: Vec<u16> = ids.clone();

        for _ in 0..max_new {
            let n = pending.len();
            // The context edge only exists under a position table (roadmap
            // 25.2): rotary layers keep going, and a sliding or linear layer's
            // state stays bounded however far they go.
            if n == 0 || (self.bounded && cache.position() + n > self.context) {
                break;
            }
            let window: Vec<i64> = pending.iter().map(|t| *t as i64).collect();
            let tokens =
                Tensor::<B, 1, Int>::from_ints(window.as_slice(), device).reshape([1, n]);

            let logits = self
                .forward_cached(tokens, &mut cache)
                .logits
                .narrow(1, n - 1, 1)
                .reshape([1, self.vocab_size]);

            let next = sampling.pick(&logits, rng);
            ids.push(next);
            if next == Special::Eos.id() {
                break;
            }
            pending = vec![next];
        }
        ids
    }

    /// Log-probabilities of the next token given `ids` `[vocab]`.
    ///
    /// The prefix is truncated from the left to what the position table covers,
    /// exactly as [`Self::generate`] does, so a lookahead search and a greedy
    /// run see the same conditioning.
    fn next_logprobs(&self, ids: &[u16], device: &B::Device) -> Vec<f64> {
        let start = ids.len().saturating_sub(self.context);
        let window: Vec<i64> = ids[start..].iter().map(|t| *t as i64).collect();
        let n = window.len().max(1);
        let window = if window.is_empty() {
            vec![Special::Bos.id() as i64]
        } else {
            window
        };

        let tokens = Tensor::<B, 1, Int>::from_ints(window.as_slice(), device).reshape([1, n]);
        let logits = self
            .forward(tokens)
            .logits
            .narrow(1, n - 1, 1)
            .reshape([1, self.vocab_size]);

        log_softmax(logits, 1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .map(f64::from)
            .collect()
    }

    /// Continue `prompt` using lookahead decoding (roadmap 21.5).
    ///
    /// Greedy decoding is myopic: the likeliest token now can open onto a
    /// continuation the model itself rates poorly. This scores candidate
    /// *continuations* of depth `budget.max_depth` and commits only their first
    /// token, then re-plans — so every committed token is chosen with evidence
    /// about what follows it.
    ///
    /// The model is the verifier: a continuation's score is the sum of its own
    /// log-probabilities under the same model. No second network is involved,
    /// which is what keeps this a decoding change rather than a training one.
    ///
    /// Cost is `budget` model calls per committed token in the worst case,
    /// against exactly one for [`Self::generate`]. With
    /// [`Budget::greedy`] and `top_k = 1` it *is* greedy decoding, at the same
    /// cost — that containment is certificate `planner/greedy_within_lookahead`.
    pub fn generate_lookahead(
        &self,
        prompt: &[u16],
        max_new: usize,
        top_k: usize,
        budget: Budget,
        device: &B::Device,
    ) -> (Vec<u16>, LookaheadStats) {
        let mut ids: Vec<u16> = prompt.to_vec();
        if ids.is_empty() {
            ids.push(Special::Bos.id());
        }

        let decoder = LookaheadDecoder::new(budget, top_k);
        let mut stats = LookaheadStats::default();

        for _ in 0..max_new {
            // One cache per committed token. Hypothesized paths share prefixes
            // heavily -- the whole point of a beam -- so without this the same
            // continuation is recomputed once per sibling.
            let mut cache: std::collections::HashMap<Vec<u16>, Vec<f64>> =
                std::collections::HashMap::new();

            let plan = decoder.plan(&ids, |context: &[u16]| {
                let logprobs = match cache.get(context) {
                    Some(hit) => hit.clone(),
                    None => {
                        stats.model_calls += 1;
                        let computed = self.next_logprobs(context, device);
                        cache.insert(context.to_vec(), computed.clone());
                        computed
                    }
                };
                logprobs
                    .into_iter()
                    .enumerate()
                    .map(|(id, lp)| (id as u16, lp))
                    .collect()
            });

            stats.evaluations += plan.evaluations;
            stats.budget_exhausted |= plan.budget_exhausted;
            stats.lookahead_depth += plan.depth();

            let Some(step) = plan.commit() else { break };
            ids.push(step.token);
            stats.committed += 1;
            if step.token == Special::Eos.id() {
                break;
            }
        }

        (ids, stats)
    }
}

/// What lookahead decoding cost, and whether it got what it paid for.
#[derive(Debug, Clone, Default)]
pub struct LookaheadStats {
    /// Tokens actually emitted.
    pub committed: usize,
    /// Forward passes performed. Divided by `committed`, the honest multiple
    /// over greedy decoding's one call per token.
    pub model_calls: usize,
    /// Candidate evaluations charged against the budget. Exceeds `model_calls`
    /// exactly by the number of prefix cache hits.
    pub evaluations: usize,
    /// Summed depth of the committed plans; divided by `committed`, how far
    /// ahead the search actually managed to look.
    pub lookahead_depth: usize,
    /// Whether the budget ever cut a search short. True means the configured
    /// depth was not reached and the result is the best fully evaluated level.
    pub budget_exhausted: bool,
}

impl LookaheadStats {
    /// Forward passes per emitted token; 1.0 is greedy decoding.
    pub fn calls_per_token(&self) -> f64 {
        if self.committed == 0 {
            return 0.0;
        }
        self.model_calls as f64 / self.committed as f64
    }

    /// Mean depth of the committed plans.
    pub fn mean_depth(&self) -> f64 {
        if self.committed == 0 {
            return 0.0;
        }
        self.lookahead_depth as f64 / self.committed as f64
    }
}

/// Per-layer decode state for incremental decoding: keys and values, a
/// sliding window's tail, or a linear layer's recurrent state, whichever the
/// layer's mode keeps (roadmap 25.3).
///
/// A cache is bound to one sequence: it records where in that sequence the
/// next token goes, and every layer's state for everything before it.
#[derive(Debug, Clone)]
pub struct KvCache<B: Backend> {
    layers: Vec<LayerState<B>>,
    position: usize,
}

impl<B: Backend> KvCache<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerState::new()).collect(),
            position: 0,
        }
    }

    /// Tokens already absorbed — where the next one lands.
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Forget the sequence so the cache can be reused for another prompt.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.position = 0;
    }

    /// The per-layer states, for inspection.
    pub fn layers(&self) -> &[LayerState<B>] {
        &self.layers
    }

    /// Floats of state held across every layer: the decode-time footprint.
    pub fn resident_floats(&self) -> usize {
        self.layers.iter().map(LayerState::resident_floats).sum()
    }
}

/// One training step's loss, metrics and routing statistics.
#[derive(Debug, Clone)]
pub struct LmStep<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub metrics: LmMetrics,
    /// Per-sparse-layer routing statistics of the executed span, in execution
    /// order; empty for a dense trunk (roadmap 23.6).
    pub routing: Vec<crate::moe::RoutingStats>,
}

/// Diagnostics for one language-model step.
#[derive(Debug, Clone, Copy)]
pub struct LmMetrics {
    pub loss: f32,
    /// `exp(loss)` — the per-token branching factor, which is the number
    /// language modeling is actually read in.
    pub perplexity: f32,
    /// Non-padding targets the loss was averaged over.
    pub tokens_counted: usize,
    pub balance_loss: f32,
    /// Targets carrying a penalty weight (roadmap Phase 24); 0 without labels.
    pub penalized_tokens: usize,
    /// Mean weighted unlikelihood charge over the penalized targets.
    pub penalty: f32,
    /// Mean probability the model assigned to the penalized targets — the
    /// number negative supervision exists to drive down, reported whether or
    /// not it is being charged for.
    pub penalized_prob: f32,
    /// Mean normalized per-token routing entropy over the span's sparse
    /// layers (roadmap 23.6); 0 for a dense trunk.
    pub routing_entropy: f32,
    /// Mean, over the span's sparse layers, of the largest expert load
    /// fraction; 0 for a dense trunk. `1/E` is balanced, `1` is collapse.
    pub routing_max_load: f32,
    /// Positions a negative teacher proposed a charged token at (roadmap 29.3).
    pub negative_teacher_tokens: usize,
    /// Mean probability the model gave those tokens.
    pub negative_teacher_prob: f32,
    /// The distillation term's value (before its weight); 0 without teachers.
    pub distill_loss: f32,
    /// Mean squared projection of the penalized layer's states onto the
    /// direction (before its weight); 0 without one (roadmap 31.2).
    pub direction_projection: f32,
}

/// A behaviour direction penalized during training (roadmap 31.2).
#[derive(Clone, Copy)]
pub struct DirectionPenalty<'a, B: Backend> {
    /// Unit vector `[h]` on the device.
    pub direction: &'a Tensor<B, 1>,
    /// Layer whose output is penalized (0-based, absolute).
    pub layer: usize,
    pub weight: f64,
}

/// Tokens proposed by a negative teacher, with their charge (roadmap 29.3).
#[derive(Debug, Clone)]
pub struct ExtraNegatives<B: Backend> {
    /// `[b, n - 1]`: the token the negative model would emit at each position.
    pub tokens: Tensor<B, 2, Int>,
    /// `[b, n - 1]`: `1.0` where the proposal is charged, else `0.0`.
    pub weights: Tensor<B, 2>,
    /// Coefficient on the charge.
    pub alpha: f32,
    /// Floor inside the unlikelihood logarithm.
    pub epsilon: f32,
}

/// Distillation toward a weighted mixture of teachers (roadmap 29.2).
#[derive(Clone, Copy)]
pub struct Distillation<'a, B: Backend> {
    pub teachers: &'a [(&'a LanguageModel<B>, f64)],
    pub temperature: f64,
    pub weight: f64,
}

/// How the next token is chosen.
#[derive(Debug, Clone, Copy, Default)]
pub enum Sampling {
    /// Always the arg-max. Deterministic, and what a correctness test wants.
    #[default]
    Greedy,
    /// Temperature-scaled sampling restricted to the `k` most likely tokens.
    TopK { k: usize, temperature: f64 },
}

impl Sampling {
    pub fn parse(name: &str, k: usize, temperature: f64) -> anyhow::Result<Self> {
        match name {
            "greedy" => Ok(Self::Greedy),
            "topk" => Ok(Self::TopK { k: k.max(1), temperature }),
            other => anyhow::bail!("unknown sampling '{other}' (expected greedy|topk)"),
        }
    }

    /// Choose a token from `logits` `[1, vocab]`.
    fn pick<B: Backend<FloatElem = f32>, R: Rng>(&self, logits: &Tensor<B, 2>, rng: &mut R) -> u16 {
        match *self {
            Self::Greedy => {
                let idx: Vec<i64> = logits
                    .clone()
                    .argmax(1)
                    .squeeze_dim::<1>(1)
                    .into_data()
                    .convert::<i64>()
                    .iter()
                    .collect();
                idx[0] as u16
            }
            Self::TopK { k, temperature } => {
                let t = temperature.max(1e-6) as f32;
                let probs: Vec<f32> = softmax(logits.clone().div_scalar(t), 1)
                    .into_data()
                    .convert::<f32>()
                    .iter::<f32>()
                    .collect();

                let mut ranked: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
                ranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                ranked.truncate(k.max(1).min(ranked.len()));

                let mass: f32 = ranked.iter().map(|(_, p)| p).sum();
                // Sample on the host so a seeded rng fully determines the
                // continuation, matching how the solvers draw their noise.
                let mut target = rng.random::<f32>() * mass;
                for (id, p) in &ranked {
                    target -= p;
                    if target <= 0.0 {
                        return *id as u16;
                    }
                }
                ranked.last().map_or(0, |(id, _)| *id as u16)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ByteTokenizer;
    use burn::backend::NdArray;
    use rand::{rngs::StdRng, SeedableRng};

    type B = NdArray<f32>;

    fn model() -> (LanguageModel<B>, <B as burn::tensor::backend::BackendTypes>::Device) {
        let device = Default::default();
        (LanguageModel::<B>::new(&LmConfig::tiny(), &device), device)
    }

    fn tokens(device: &<B as burn::tensor::backend::BackendTypes>::Device, ids: &[u16]) -> Tensor<B, 2, Int> {
        let v: Vec<i64> = ids.iter().map(|t| *t as i64).collect();
        Tensor::<B, 1, Int>::from_ints(v.as_slice(), device).reshape([1, v.len()])
    }

    fn logp_of_targets(m: &LanguageModel<B>, ids: &[u16], device: &<B as burn::tensor::backend::BackendTypes>::Device) -> Vec<f32> {
        // log p(target_j | prefix) for j = 0..n-1, straight from the logits.
        let n = ids.len();
        let logits = m.forward(tokens(device, ids)).logits.narrow(1, 0, n - 1).reshape([n - 1, VOCAB_SIZE]);
        let lp: Vec<f32> = log_softmax(logits, 1).into_data().convert::<f32>().iter::<f32>().collect();
        (0..n - 1).map(|j| lp[j * VOCAB_SIZE + ids[j + 1] as usize]).collect()
    }

    #[test]
    fn test_penalized_loss_with_no_labels_is_the_plain_loss_bitwise() {
        // Every run with labels open but nothing flagged in the batch goes
        // through the penalized path; it must cost exactly nothing.
        let (m, device) = model();
        let ids = [65u16, 66, 67, 68, 69, 70];
        let (plain, pm) = m.next_token_loss(tokens(&device, &ids), 0..m.num_layers());
        let zeros = Tensor::<B, 2>::zeros([1, ids.len()], &device);
        let (penalized, qm) =
            m.next_token_loss_penalized(tokens(&device, &ids), zeros, Unlikelihood::default(), 0..m.num_layers());
        assert_eq!(plain.into_scalar().to_bits(), penalized.into_scalar().to_bits());
        assert_eq!(qm.tokens_counted, pm.tokens_counted);
        assert_eq!((qm.penalized_tokens, qm.penalty, qm.penalized_prob), (0, 0.0, 0.0));
    }

    #[test]
    fn test_a_penalized_target_is_charged_and_leaves_the_likelihood() {
        // Recomputed by hand from the logits: the loss must be
        //   (sum over positives of -log p + alpha * sum over negatives of w * -log(1 - p)) / counted
        // with each negative in exactly one of the two sums.
        let (m, device) = model();
        let ids = [65u16, 66, 67, 68, 69];
        let logp = logp_of_targets(&m, &ids, &device);

        // Token 2 (target of position 1) at half weight, token 4 at full.
        let w = [0.0f32, 0.0, 0.5, 0.0, 1.0];
        let weights = Tensor::<B, 1>::from_floats(w.as_slice(), &device).reshape([1, 5]);
        let penalty = Unlikelihood { alpha: 2.0, epsilon: 1e-6 };
        let (loss, metrics) =
            m.next_token_loss_penalized(tokens(&device, &ids), weights, penalty, 0..m.num_layers());

        let mut likelihood = 0.0f32;
        let mut charge = 0.0f32;
        let mut prob = 0.0f32;
        for j in 0..4 {
            let wj = w[j + 1];
            if wj > 0.0 {
                charge += wj * -((1.0 - logp[j].exp()).max(penalty.epsilon)).ln();
                prob += logp[j].exp();
            } else {
                likelihood += -logp[j];
            }
        }
        let expected = (likelihood + penalty.alpha * charge) / 4.0;
        let got = loss.into_scalar();
        assert!((got - expected).abs() < 1e-5 * expected.abs().max(1.0), "loss {got} vs hand {expected}");
        assert_eq!(metrics.penalized_tokens, 2);
        assert!((metrics.penalty - charge / 2.0).abs() < 1e-5, "{} vs {}", metrics.penalty, charge / 2.0);
        assert!((metrics.penalized_prob - prob / 2.0).abs() < 1e-6);
        assert_eq!(metrics.tokens_counted, 4);
    }

    #[test]
    fn test_a_penalty_of_zero_is_the_plain_objective_with_metrics() {
        // "Measure only" must train on the flagged tokens exactly as a plain
        // run would. The first version excluded them from the likelihood
        // while charging nothing, and a plain run's p(bad) *fell* -- the
        // metric was measuring tokens the model was never shown.
        let (m, device) = model();
        let ids = [65u16, 66, 67, 68, 69];
        let w = [0.0f32, 0.0, 1.0, 0.0, 0.5];
        let weights = Tensor::<B, 1>::from_floats(w.as_slice(), &device).reshape([1, 5]);
        let (plain, _) = m.next_token_loss(tokens(&device, &ids), 0..m.num_layers());
        let (off, metrics) =
            m.next_token_loss_penalized(tokens(&device, &ids), weights, Unlikelihood::off(), 0..m.num_layers());
        assert_eq!(plain.into_scalar().to_bits(), off.into_scalar().to_bits());
        assert_eq!(metrics.penalized_tokens, 2, "still counted");
        assert!(metrics.penalized_prob > 0.0 && metrics.penalty > 0.0, "still measured");
    }

    #[test]
    fn test_padding_is_never_a_negative_target() {
        // A weight on a padded position must not conjure a negative: padding
        // is outside both sums, whatever the label file says about it.
        let (m, device) = model();
        let pad = Special::Pad.id();
        let ids = [65u16, 66, pad, pad];
        let w = [0.0f32, 0.0, 1.0, 1.0];
        let weights = Tensor::<B, 1>::from_floats(w.as_slice(), &device).reshape([1, 4]);
        let (plain, _) = m.next_token_loss(tokens(&device, &ids), 0..m.num_layers());
        let (penalized, metrics) =
            m.next_token_loss_penalized(tokens(&device, &ids), weights, Unlikelihood::default(), 0..m.num_layers());
        assert_eq!(metrics.penalized_tokens, 0);
        assert_eq!(metrics.tokens_counted, 1);
        assert_eq!(plain.into_scalar().to_bits(), penalized.into_scalar().to_bits());
    }

    #[test]
    fn test_unlikelihood_is_zero_when_impossible_and_bounded_when_certain() {
        // The property that makes this a penalty rather than a negative
        // reward: nothing to gain below zero, and no way to reach -inf.
        let device = Default::default();
        let eps = 1e-6f32;
        let term: Vec<f32> = unlikelihood(
            Tensor::<B, 1>::from_floats([-40.0f32, 0.5f32.ln(), 0.0], &device),
            eps,
        )
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
        assert!(term[0].abs() < 1e-12, "impossible token: {}", term[0]);
        assert!((term[1] - 2f32.ln()).abs() < 1e-6, "p = 1/2: {}", term[1]);
        assert!((term[2] - Unlikelihood { alpha: 1.0, epsilon: eps }.ceiling()).abs() < 1e-4, "certain token: {}", term[2]);
        assert!(term[0] < term[1] && term[1] < term[2], "must increase with p");
    }

    #[test]
    fn test_label_weights_follow_the_table() {
        let device = Default::default();
        let mut table = [0.0f32; 256];
        table[1] = 1.0;
        table[2] = 0.25;
        let labels = vec![vec![0u8, 1, 2], vec![2, 0, 1]];
        let w: Vec<f32> = label_weights::<B>(&labels, &table, &device)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        assert_eq!(w, vec![0.0, 1.0, 0.25, 0.25, 0.0, 1.0]);
    }

    #[test]
    fn test_forward_shapes_and_tied_head() {
        let (m, device) = model();
        let out = m.forward(tokens(&device, &[1, 2, 3, 4]));
        assert_eq!(out.logits.dims(), [1, 4, VOCAB_SIZE]);
        assert!(out.balance_loss.is_none(), "a dense trunk has no balance loss");

        // Weight tying is structural: the logits ARE a product with the
        // embedding table, so there is no separate output projection to drift.
        assert_eq!(m.embedding_weight().dims(), [VOCAB_SIZE, 32]);
    }

    #[test]
    fn test_logits_at_position_i_ignore_tokens_after_i() {
        // The property that makes next-token training meaningful: if position
        // `i` could see `i + 1`, the model would learn to copy the answer and
        // the loss would collapse without learning anything.
        let (m, device) = model();
        let base = [5u16, 6, 7, 8, 9];
        let reference = m.forward(tokens(&device, &base)).logits;

        let mut changed = base;
        changed[4] = 200; // perturb the LAST token only
        let perturbed = m.forward(tokens(&device, &changed)).logits;

        let prefix_drift = (reference.clone().narrow(1, 0, 4) - perturbed.clone().narrow(1, 0, 4))
            .abs()
            .max()
            .into_scalar();
        assert_eq!(prefix_drift, 0.0, "future token leaked into earlier logits");

        // The last position must respond, or the check above is vacuous.
        let tail = (reference.narrow(1, 4, 1) - perturbed.narrow(1, 4, 1))
            .abs()
            .max()
            .into_scalar();
        assert!(tail > 0.0);
    }

    #[test]
    fn test_next_token_loss_starts_near_uniform() {
        // An untrained model should be near-uniform over the vocabulary, so the
        // loss should sit around ln(V) and the perplexity around V. A value far
        // from that means the shift or the masking is wrong.
        let (m, device) = model();
        let (loss, metrics) = m.next_token_loss(tokens(&device, &[10, 20, 30, 40, 50]), 0..4);
        assert!(loss.into_scalar().is_finite());

        let expected = (VOCAB_SIZE as f32).ln();
        assert!(
            (metrics.loss - expected).abs() < 1.0,
            "expected ~ln({VOCAB_SIZE}) = {expected}, got {}. A large gap means \
             the tied head is producing peaked logits at initialization, so \
             training would begin by undoing a confidently wrong prior.",
            metrics.loss
        );
        assert!(
            (metrics.perplexity / VOCAB_SIZE as f32 - 1.0).abs() < 1.0,
            "perplexity {} should be near the vocabulary size",
            metrics.perplexity
        );
        assert_eq!(metrics.tokens_counted, 4, "5 tokens give 4 targets");
    }

    #[test]
    fn test_padding_is_excluded_from_the_loss() {
        // Padding must leave the denominator, not just the numerator. Zeroing
        // only the numerator would make a mostly-padded batch report an
        // artificially small loss and quietly dominate a run.
        let (m, device) = model();
        let pad = Special::Pad.id();

        let (_, dense) = m.next_token_loss(tokens(&device, &[1, 2, 3, 4]), 0..4);
        assert_eq!(dense.tokens_counted, 3);

        let (_, padded) = m.next_token_loss(tokens(&device, &[1, 2, pad, pad]), 0..4);
        assert_eq!(padded.tokens_counted, 1, "only one non-pad target remains");
        assert!(padded.loss.is_finite() && padded.loss > 0.0);

        // An all-padding batch must not divide by zero.
        let (_, empty) = m.next_token_loss(tokens(&device, &[pad, pad, pad]), 0..4);
        assert!(empty.loss.is_finite(), "all-padding must not produce NaN");
    }

    #[test]
    fn test_block_spans_partition_the_layers() {
        let (m, _) = model();
        assert_eq!(m.num_blocks(), 2);
        assert_eq!(m.layer_range(0), 0..2);
        assert_eq!(m.layer_range(1), 2..4);
        // Contiguous and covering, which is what makes block-wise gradient
        // routing exhaustive.
        assert_eq!(m.layer_range(0).end, m.layer_range(1).start);
        assert_eq!(m.layer_range(m.num_blocks() - 1).end, m.num_layers());
    }

    #[test]
    fn test_partial_span_differs_from_the_full_trunk() {
        let (m, device) = model();
        let full = m.forward(tokens(&device, &[1, 2, 3])).logits;
        let partial = m.forward_span(tokens(&device, &[1, 2, 3]), 0..2).logits;
        let diff = (full - partial).abs().max().into_scalar();
        assert!(diff > 0.0, "running fewer layers must change the output");
    }

    #[test]
    fn test_greedy_generation_is_deterministic_and_bounded() {
        let (m, device) = model();
        let tok = ByteTokenizer::new();
        let prompt = tok.encode("hi");

        let a = m.generate(&prompt, 5, &Sampling::Greedy, &mut StdRng::seed_from_u64(1), &device);
        let b = m.generate(&prompt, 5, &Sampling::Greedy, &mut StdRng::seed_from_u64(9), &device);
        assert_eq!(a, b, "greedy decoding must not depend on the rng");
        assert!(a.len() <= prompt.len() + 5);
        assert!(a.starts_with(&prompt), "the prompt must be preserved");
        assert!(a.iter().all(|t| (*t as usize) < VOCAB_SIZE));
    }

    #[test]
    fn test_sampling_is_reproducible_from_the_seed() {
        let (m, device) = model();
        let prompt = vec![Special::Bos.id(), 65];
        let s = Sampling::TopK { k: 8, temperature: 1.0 };

        let a = m.generate(&prompt, 6, &s, &mut StdRng::seed_from_u64(4), &device);
        let b = m.generate(&prompt, 6, &s, &mut StdRng::seed_from_u64(4), &device);
        assert_eq!(a, b, "the same seed must replay the same continuation");

        // Restricting to the top-1 recovers greedy exactly.
        let top1 = m.generate(
            &prompt,
            4,
            &Sampling::TopK { k: 1, temperature: 1.0 },
            &mut StdRng::seed_from_u64(7),
            &device,
        );
        let greedy = m.generate(&prompt, 4, &Sampling::Greedy, &mut StdRng::seed_from_u64(0), &device);
        assert_eq!(top1, greedy, "top-1 sampling is greedy decoding");
    }

    #[test]
    fn test_generation_respects_the_context_window() {
        // A prompt longer than the position table must be windowed, not
        // panic on the position slice.
        let (m, device) = model();
        let long: Vec<u16> = (0..m.context() as u16 + 5).map(|i| 65 + (i % 26)).collect();
        let out = m.generate(&long, 3, &Sampling::Greedy, &mut StdRng::seed_from_u64(2), &device);
        assert_eq!(out.len(), long.len() + 3);
    }

    #[test]
    fn test_sampling_parse() {
        assert!(matches!(Sampling::parse("greedy", 1, 1.0).unwrap(), Sampling::Greedy));
        assert!(matches!(
            Sampling::parse("topk", 5, 0.8).unwrap(),
            Sampling::TopK { k: 5, .. }
        ));
        assert!(Sampling::parse("nucleus", 1, 1.0).is_err());
    }

    #[test]
    fn test_lookahead_with_no_depth_reproduces_greedy_exactly() {
        // Containment: lookahead must be a generalization of greedy decoding,
        // not a different decoder that happens to be similar. Depth 0 with
        // top_k 1 leaves nothing to search, so the two must agree token for
        // token -- and at the same cost, one forward pass per token.
        let (m, device) = model();
        let prompt = ByteTokenizer::new().encode("hello");

        let mut rng = StdRng::seed_from_u64(7);
        let greedy = m.generate(&prompt, 12, &Sampling::Greedy, &mut rng, &device);

        let (looked, stats) =
            m.generate_lookahead(&prompt, 12, 1, Budget::greedy(), &device);

        assert_eq!(looked, greedy, "depth-0 lookahead must be greedy decoding");
        assert!(!stats.budget_exhausted);
        assert_eq!(stats.mean_depth(), 1.0, "one committed step, no lookahead");
        assert!(
            (stats.calls_per_token() - 1.0).abs() < 1e-12,
            "and it must not cost more: {} calls/token",
            stats.calls_per_token()
        );
    }

    #[test]
    fn test_lookahead_never_exceeds_its_budget() {
        // The guarantee that makes lookahead deployable: a bounded multiple of
        // greedy's cost, per token, no matter how the search branches.
        let (m, device) = model();
        let prompt = ByteTokenizer::new().encode("abc");

        for max_evaluations in [1usize, 3, 8] {
            let budget = Budget { max_evaluations, max_depth: 3, beam_width: 3 };
            let (out, stats) = m.generate_lookahead(&prompt, 4, 4, budget, &device);

            assert!(out.len() > prompt.len(), "decoding should still emit tokens");
            assert!(
                stats.model_calls <= max_evaluations * stats.committed,
                "{} calls for {} tokens at a budget of {max_evaluations}",
                stats.model_calls,
                stats.committed
            );
            assert!(stats.evaluations <= max_evaluations * stats.committed);
        }
    }

    #[test]
    fn test_lookahead_reuses_shared_prefixes() {
        // A beam's paths share prefixes by construction. Without the cache the
        // same continuation is recomputed once per sibling, which would make
        // the cost quadratic in beam width for no new information.
        let (m, device) = model();
        let prompt = ByteTokenizer::new().encode("xy");
        let budget = Budget { max_evaluations: 64, max_depth: 2, beam_width: 3 };

        let (_out, stats) = m.generate_lookahead(&prompt, 3, 3, budget, &device);
        assert!(
            stats.model_calls < stats.evaluations,
            "expected cache hits: {} calls for {} evaluations",
            stats.model_calls,
            stats.evaluations
        );
    }

    #[test]
    fn test_lookahead_is_deterministic() {
        // No sampling is involved, so two runs of the same model on the same
        // prompt must agree. A search that depended on hash iteration order
        // would fail here.
        let (m, device) = model();
        let prompt = ByteTokenizer::new().encode("determinism");
        let budget = Budget { max_evaluations: 40, max_depth: 2, beam_width: 2 };

        let (first, _) = m.generate_lookahead(&prompt, 6, 3, budget, &device);
        let (second, _) = m.generate_lookahead(&prompt, 6, 3, budget, &device);
        assert_eq!(first, second);
    }

    #[test]
    fn test_lookahead_stops_at_eos() {
        // The stopping rule has to survive the extra indirection: a plan that
        // commits <eos> must end the sequence, not decode past it.
        let (m, device) = model();
        let prompt = vec![Special::Bos.id(), Special::Eos.id()];
        let budget = Budget { max_evaluations: 16, max_depth: 1, beam_width: 2 };
        let (out, stats) = m.generate_lookahead(&prompt, 8, 2, budget, &device);
        assert!(out.len() <= prompt.len() + 8);
        assert_eq!(stats.committed, out.len() - prompt.len());
    }

    #[test]
    fn test_kv_cache_reproduces_full_recompute_exactly() {
        // The claim that makes caching free rather than a trade: a causal
        // model's keys and values at position i depend only on tokens up to i,
        // so recomputing them can only reproduce the same numbers. Anything
        // beyond float summation order here would be a real divergence.
        let (m, device) = model();
        // Kept inside `LmConfig::tiny`'s 16-position context window.
        let ids: Vec<u16> = ByteTokenizer::new().encode("exact caching!");

        let full: Vec<f32> = m
            .forward(tokens(&device, &ids))
            .logits
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();

        // Feed the same tokens in several different chunkings; every one must
        // land on the same logits.
        for chunks in [vec![ids.len()], vec![1; ids.len()], vec![3, 1, 5, ids.len() - 9]] {
            let mut cache = m.new_cache();
            let mut produced: Vec<f32> = Vec::new();
            let mut at = 0usize;
            for size in chunks {
                let slice = &ids[at..at + size];
                let out = m.forward_cached(tokens(&device, slice), &mut cache);
                assert_eq!(out.logits.dims(), [1, size, VOCAB_SIZE]);
                produced.extend(
                    out.logits.into_data().convert::<f32>().iter::<f32>(),
                );
                at += size;
            }
            assert_eq!(cache.position(), ids.len());
            assert_eq!(produced.len(), full.len());
            for (i, (a, b)) in produced.iter().zip(&full).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-4 * b.abs().max(1.0),
                    "logit {i} diverged: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_cached_generation_matches_uncached() {
        // The user-visible statement: same prompt, same seed, same text -- at
        // O(n) per token instead of O(n^2).
        let (m, device) = model();
        let prompt = ByteTokenizer::new().encode("once ");

        // Prompt plus continuation must stay inside the context: past it the
        // two paths legitimately diverge, since uncached generation slides its
        // window left and a cache cannot.
        let max_new = m.context() - prompt.len();
        for sampling in [Sampling::Greedy, Sampling::TopK { k: 3, temperature: 0.8 }] {
            let plain =
                m.generate(&prompt, max_new, &sampling, &mut StdRng::seed_from_u64(4), &device);
            let cached = m.generate_cached(
                &prompt,
                max_new,
                &sampling,
                &mut StdRng::seed_from_u64(4),
                &device,
            );
            assert_eq!(cached, plain, "cached decoding changed the output");
        }
    }

    #[test]
    fn test_cached_generation_stops_at_the_context_edge() {
        // Past the window the two paths part ways by design: uncached
        // generation slides its window left and keeps going, while a cache
        // cannot drop its oldest positions without invalidating every position
        // embedding after them. Stopping is stated behaviour, so it is tested.
        let (m, device) = model();
        let prompt = ByteTokenizer::new().encode("once upon");
        let cached = m.generate_cached(
            &prompt,
            100,
            &Sampling::Greedy,
            &mut StdRng::seed_from_u64(4),
            &device,
        );
        // One past the window is correct, not off by one: the final token is a
        // valid prediction made from the last position the table covers. It
        // simply cannot be conditioned on, so decoding ends there.
        assert!(
            cached.len() <= m.context() + 1,
            "cached decoding ran past the context: {} > {}",
            cached.len(),
            m.context() + 1
        );
        assert!(cached.len() > prompt.len(), "it should still emit something");
    }

    #[test]
    fn test_a_cache_can_be_reused_after_clearing() {
        // A cache is bound to one sequence. Reusing it without clearing would
        // silently prepend the previous prompt, which is the kind of bug that
        // shows up as mysteriously worse output rather than as an error.
        let (m, device) = model();
        let a = ByteTokenizer::new().encode("alpha");
        let b = ByteTokenizer::new().encode("beta!");

        let mut cache = m.new_cache();
        let first = m.forward_cached(tokens(&device, &a), &mut cache);
        assert_eq!(cache.position(), a.len());

        cache.clear();
        assert!(cache.is_empty());
        let after_clear: Vec<f32> = m
            .forward_cached(tokens(&device, &b), &mut cache)
            .logits
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();

        let fresh: Vec<f32> = m
            .forward(tokens(&device, &b))
            .logits
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        for (x, y) in after_clear.iter().zip(&fresh) {
            assert!((x - y).abs() <= 2e-4 * y.abs().max(1.0));
        }
        let _ = first;
    }

    #[test]
    #[should_panic(expected = "exceeds the")]
    fn test_a_cache_refuses_to_overrun_the_context() {
        // Silently dropping the oldest cached positions would invalidate every
        // position embedding after them. Refusing is the honest behaviour.
        let (m, device) = model();
        let long: Vec<u16> = (0..m.context() as u16 + 4).map(|i| i % 200).collect();
        let mut cache = m.new_cache();
        m.forward_cached(tokens(&device, &long), &mut cache);
    }

    #[test]
    fn test_every_attention_schedule_decodes_cached_like_uncached() {
        // Phase 25: whatever a layer keeps between steps -- keys, a window's
        // tail, a linear state, or both -- decoding from it must emit the
        // tokens a full recompute emits.
        use crate::hybrid::{AttentionMode, AttentionSchedule};
        let device = Default::default();
        let layers = LmConfig::tiny().num_layers;
        let schedules = vec![
            AttentionSchedule::parse("3:1", layers, 3, 2).unwrap(),
            AttentionSchedule::parse("sliding3", layers, 3, 2).unwrap(),
            AttentionSchedule::parse("retrieval2", layers, 3, 2).unwrap(),
            AttentionSchedule::parse("learned,linear,dense,sliding2", layers, 3, 2).unwrap(),
        ];
        let prompt = ByteTokenizer::new().encode("once ");
        for schedule in schedules {
            assert!(schedule.modes.iter().any(|m| *m != AttentionMode::Dense));
            let m = LanguageModel::<B>::new(&LmConfig::tiny().with_attention(schedule.clone()), &device);
            assert_eq!(m.attention_modes(), schedule.modes);
            let max_new = m.context() - prompt.len();
            let plain = m.generate(&prompt, max_new, &Sampling::Greedy, &mut StdRng::seed_from_u64(4), &device);
            let cached = m.generate_cached(&prompt, max_new, &Sampling::Greedy, &mut StdRng::seed_from_u64(4), &device);
            assert_eq!(cached, plain, "schedule {} diverged", schedule.pattern());
        }
    }

    #[test]
    fn test_rotary_sliding_trunk_runs_past_the_context() {
        // Under a position table decoding stops at the table's edge; under
        // rotary positions with bounded-state layers it keeps going, and the
        // per-layer state stays bounded however far it goes.
        use crate::hybrid::{AttentionMode, AttentionSchedule, PositionKind};
        let device = Default::default();
        let base = LmConfig::tiny();
        let schedule = AttentionSchedule::ratio(base.num_layers, 1, AttentionMode::Linear, AttentionMode::Sliding { window: 4 });
        let unbounded = LanguageModel::<B>::new(
            &base.clone().with_attention(schedule.clone()).with_positions(PositionKind::Rotary),
            &device,
        );
        assert!(!unbounded.positions_bounded());
        let prompt = vec![Special::Bos.id(), 65, 66];
        let want = base.context * 2;
        let out = unbounded.generate_cached(&prompt, want, &Sampling::Greedy, &mut StdRng::seed_from_u64(1), &device);
        // Greedy decoding may hit <eos> early; then the run is simply shorter.
        let stopped_at_eos = out.last() == Some(&Special::Eos.id());
        assert!(out.len() == prompt.len() + want || stopped_at_eos, "stopped at {} of {}", out.len(), prompt.len() + want);

        // The footprint of the state is what the modes promise: a window of 4
        // holds at most 3 keys per sliding layer, a linear layer holds d^2 + d.
        let mut cache = unbounded.new_cache();
        let ids: Vec<i64> = (0..(base.context as i64 + 9)).map(|i| 40 + i % 50).collect();
        for chunk in ids.chunks(4) {
            let t = Tensor::<B, 1, Int>::from_ints(chunk, &device).reshape([1, chunk.len()]);
            unbounded.forward_cached(t, &mut cache);
        }
        assert_eq!(cache.position(), ids.len());
        for (state, mode) in cache.layers().iter().zip(unbounded.attention_modes()) {
            match mode {
                AttentionMode::Sliding { window } => assert!(state.len() < window, "window kept {} keys", state.len()),
                AttentionMode::Linear => assert!(state.keys.is_none() && state.linear.is_some()),
                _ => unreachable!(),
            }
        }

        // A learned table still stops where it always did.
        let bounded = LanguageModel::<B>::new(&base.clone().with_attention(schedule), &device);
        assert!(bounded.positions_bounded());
        let out = bounded.generate_cached(&prompt, want, &Sampling::Greedy, &mut StdRng::seed_from_u64(1), &device);
        assert!(out.len() <= base.context + 1);
    }

    #[test]
    fn test_routing_state_trunk_reports_locality() {
        // A MoE trunk carrying a routing state routes, trains a step, and
        // reports token stability and layer agreement (roadmap 25.4-25.6).
        use crate::vit::MoeTrunkConfig;
        let device = Default::default();
        let moe = MoeTrunkConfig { num_experts: 3, top_k: 1, every_n_layers: 2, z_level: 1e-3, balance_bias: false };
        let config = LmConfig { moe: Some(moe), ..LmConfig::tiny() }.with_routing_state(6);
        let m = LanguageModel::<B>::new(&config, &device);
        assert!(m.has_routing_state());
        let ids = [65u16, 66, 67, 68, 69, 70, 71, 72];
        let step = m.next_token_step(tokens(&device, &ids), None, 0..m.num_layers());
        assert!(step.metrics.loss.is_finite());
        assert_eq!(step.routing.len(), 2, "two sparse layers");
        for stats in &step.routing {
            assert!((0.0..=1.0).contains(&stats.stability), "{}", stats.stability);
            assert_eq!(stats.kind, crate::routing::RouterKind::Ffn);
        }
        assert!(step.routing[0].agreement.is_none(), "the first routed layer has nothing before it");
        let agreement = step.routing[1].agreement.expect("equal widths are comparable");
        assert!((0.0..=1.0).contains(&agreement));

        // The state also decodes: cached and uncached generation still agree.
        let prompt = ByteTokenizer::new().encode("ab");
        let plain = m.generate(&prompt, 6, &Sampling::Greedy, &mut StdRng::seed_from_u64(2), &device);
        let cached = m.generate_cached(&prompt, 6, &Sampling::Greedy, &mut StdRng::seed_from_u64(2), &device);
        assert_eq!(plain, cached);
    }

    #[test]
    fn test_config_fields_from_phase_25_survive_json_and_default_to_off() {
        use crate::hybrid::{AttentionSchedule, PositionKind};
        let config = LmConfig::tiny()
            .with_attention(AttentionSchedule::parse("2:1@sliding8", 4, 8, 4).unwrap())
            .with_positions(PositionKind::Rotary)
            .with_routing_state(3);
        let json = serde_json::to_string(&config).unwrap();
        let back: LmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.attention_schedule().pattern(), "SSDS");
        assert_eq!(back.positions, PositionKind::Rotary);
        assert_eq!(back.routing_state, 3);
        // A Phase 19 state file, written before these fields existed, still parses.
        let old = serde_json::to_string(&LmConfig::tiny()).unwrap().replace(",\"attention\":null", "").replace(",\"positions\":\"learned\"", "").replace(",\"routing_state\":0", "");
        assert!(!old.contains("routing_state"));
        let parsed: LmConfig = serde_json::from_str(&old).unwrap();
        assert!(parsed.attention.is_none() && parsed.positions == PositionKind::Learned && parsed.routing_state == 0);
    }
}
