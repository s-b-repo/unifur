//! Token-block diffusion substrate — the subject that plan.md Rev-4 said
//! Items 1–2 do not have.
//!
//! A BD3-LM-flavoured semi-autoregressive byte-level LM: the sequence is
//! generated one block at a time, autoregressive over blocks, with discrete
//! (absorbing-mask) diffusion *within* a block and block-causal attention —
//! every position attends to all committed blocks and bidirectionally to its
//! own block, never to a later one. Committed blocks carry per-layer keys and
//! values, so a settled prefix costs nothing to re-read: the property that
//! makes block diffusion attractive, and the property every mechanism below
//! has to preserve.
//!
//! This module exists so the improvement items have something true to modify
//! and measure:
//!
//! - **Adaptive block sizing** (plan item 1): the pending region is sized from
//!   a bidirectional cumulative entropy signal computed over the model's own
//!   predictions — low entropy earns a larger block, high entropy a smaller
//!   one. Bounded below by `min_block` (never 1: a one-token block is
//!   autoregressive decoding with extra steps) and above by `max_block`; both
//!   degenerate cases the instruction warns about are structural
//!   impossibilities, and a certificate pins that.
//! - **Halo** (plan item 1): when a block's entropy signal is high, a small
//!   buffer of already-committed tail tokens is re-opened inside the block's
//!   denoising window and may be revised once before the block commits, and a
//!   peek of masked placeholders beyond the block edge joins the window as
//!   context. Gated on entropy — an ungated halo on every block is
//!   full-sequence diffusion wearing a costume.
//! - **Routine bounded revision** (plan item 2): after every committed block,
//!   the last `revision_blocks` blocks are re-predicted once with full
//!   bidirectional attention inside the window; disagreements above
//!   `edit_threshold` are applied, the affected keys and values are rebuilt,
//!   and every change is counted. The window never grows; a growing window is
//!   how block diffusion stops being block diffusion.
//!
//! Objective: per-token noise levels `t_i ~ U(0,1)`; a token is absorbed to
//! the mask token with probability `t_i` (the linear schedule BD3-LM trains
//! with) and the loss is cross-entropy over absorbed positions only. The
//! importance weighting BD3-LM adds on top is a known refinement, deliberately
//! not carried here: one objective, one place to audit.
//!
//! Honest scope, stated where the claim is: this is a substrate, not a
//! result. It is trainable at sizes a CPU can carry, and nothing here claims
//! the quality of a pretrained block-diffusion checkpoint. What it does claim
//! — and what the certificates pin — is the two structural identities (prefix
//! invariance, cache equivalence) and the boundedness of every adaptive
//! mechanism on top of them.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context as _, Result};
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::tensor_ext::exact_gelu;
use crate::tokenizer::{Special, VOCAB_SIZE};

/// The absorbing state a masked position carries. One id above the shared
/// vocabulary, so the LM's 259 ids keep their meaning and this module owns
/// exactly one extra.
pub const MASK_ID: u16 = VOCAB_SIZE as u16;

/// Vocabulary of this substrate: the byte tokenizer's 259 ids plus the mask.
pub const TB_VOCAB: usize = VOCAB_SIZE + 1;

/// Weight the entropy signal puts on a window's edge tokens. HaloBlock's
/// measured pathologies are edge pathologies, so the signal listens harder at
/// edges; 2.0 is the smallest weight that makes the edge audible without
/// letting it dominate.
pub const EDGE_WEIGHT: f64 = 2.0;

/// Additive attention bias standing in for "impossible": large enough that
/// softmax over a row containing it gives exactly zero weight in f32, finite
/// so a fully-masked row can never produce a NaN.
const NEG_BIAS: f32 = -1.0e9;

/// Shape and policy of the substrate. Validated by [`TokenBlockModel::new`];
/// never assumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TbConfig {
    /// Longest sequence, prompt included. Position table size.
    pub context: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    /// Feed-forward width inside each layer.
    pub intermediate_size: usize,
    /// The fixed block size, and the only size [`Sizing::Fixed`] uses.
    pub block_len: usize,
    /// Adaptive sizing floor. Never 1: a one-token block is autoregressive
    /// decoding, not block diffusion.
    pub min_block: usize,
    /// Adaptive sizing ceiling.
    pub max_block: usize,
    /// Denoising iterations per block at decode time.
    pub decode_steps: usize,
    /// Commit threshold: a masked position whose max probability reaches this
    /// is revealed early (the Fast-dLLM rule).
    pub confidence_threshold: f64,
    /// Threshold on the entropy signal above which a block earns a halo.
    pub halo_threshold: f64,
    /// Largest halo, in tokens, each side.
    pub halo_max: usize,
    /// How many trailing blocks the routine revision pass re-opens. `0` is
    /// off.
    pub revision_blocks: usize,
    /// Probability an edit must reach to be applied.
    pub edit_threshold: f64,
    pub initializer_range: f64,
}

impl Default for TbConfig {
    fn default() -> Self {
        Self {
            context: 512,
            hidden_size: 128,
            num_layers: 4,
            num_heads: 4,
            intermediate_size: 256,
            block_len: 32,
            min_block: 8,
            max_block: 48,
            decode_steps: 8,
            confidence_threshold: 0.9,
            halo_threshold: 0.5,
            halo_max: 4,
            revision_blocks: 1,
            edit_threshold: 0.6,
            initializer_range: 0.02,
        }
    }
}

impl TbConfig {
    /// Configuration for unit tests and certificates: small enough that a
    /// forward is milliseconds on the ndarray backend.
    pub fn tiny() -> Self {
        Self {
            context: 48,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
            block_len: 8,
            min_block: 4,
            max_block: 12,
            decode_steps: 4,
            confidence_threshold: 0.5,
            halo_threshold: 0.5,
            halo_max: 2,
            revision_blocks: 1,
            edit_threshold: 0.5,
            initializer_range: 0.05,
        }
    }

    /// Every constraint later code relies on, checked once, naming the field
    /// at fault.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.hidden_size > 0 && self.hidden_size % self.num_heads == 0,
            "hidden_size ({}) must be a positive multiple of num_heads ({})",
            self.hidden_size,
            self.num_heads
        );
        anyhow::ensure!(
            self.num_layers > 0,
            "num_layers ({}) must be positive",
            self.num_layers
        );
        anyhow::ensure!(
            self.intermediate_size > 0,
            "intermediate_size ({}) must be positive",
            self.intermediate_size
        );
        anyhow::ensure!(
            self.context > 0,
            "context ({}) must be positive",
            self.context
        );
        anyhow::ensure!(
            self.block_len >= 1 && self.block_len <= self.context,
            "block_len ({}) must be in 1..={}",
            self.block_len,
            self.context
        );
        anyhow::ensure!(
            self.min_block >= 2,
            "min_block ({}) must be at least 2 — a 1-token block is autoregressive \
             decoding, not block diffusion",
            self.min_block
        );
        anyhow::ensure!(
            self.min_block <= self.max_block,
            "min_block ({}) must not exceed max_block ({})",
            self.min_block,
            self.max_block
        );
        anyhow::ensure!(
            self.max_block <= self.context,
            "max_block ({}) must not exceed context ({})",
            self.max_block,
            self.context
        );
        anyhow::ensure!(
            self.decode_steps >= 1,
            "decode_steps ({}) must be at least 1",
            self.decode_steps
        );
        anyhow::ensure!(
            self.confidence_threshold > 0.0 && self.confidence_threshold <= 1.0,
            "confidence_threshold ({}) must be in (0, 1]",
            self.confidence_threshold
        );
        anyhow::ensure!(
            self.halo_threshold >= 0.0 && self.halo_threshold <= 1.0,
            "halo_threshold ({}) must be in [0, 1]",
            self.halo_threshold
        );
        anyhow::ensure!(
            self.edit_threshold > 0.0 && self.edit_threshold <= 1.0,
            "edit_threshold ({}) must be in (0, 1]",
            self.edit_threshold
        );
        anyhow::ensure!(
            self.revision_blocks * self.max_block <= self.context,
            "revision window ({} blocks x max_block {}) must fit the context ({})",
            self.revision_blocks,
            self.max_block,
            self.context
        );
        Ok(())
    }

    /// The block index of an absolute position under fixed-size blocks. The
    /// mask builder's single source of truth.
    pub fn block_of(pos: usize, block_len: usize) -> usize {
        pos / block_len.max(1)
    }
}

/// One pre-norm transformer layer. The attention mask arrives per call, so
/// block-causal, halo and edit-window masking are all the same mechanism.
#[derive(Module, Debug)]
pub struct TbLayer<B: Backend> {
    norm1: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    dense: Linear<B>,
    norm2: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

/// The substrate model: embeddings, layers, and a tied output head against
/// the token embedding table.
#[derive(Module, Debug)]
pub struct TokenBlockModel<B: Backend> {
    embedding: Embedding<B>,
    position: Embedding<B>,
    layers: Vec<TbLayer<B>>,
    final_norm: LayerNorm<B>,
    num_heads: usize,
    head_dim: usize,
}

/// Keys and values for one layer's share of the committed prefix.
#[derive(Debug, Clone)]
struct LayerKv<B: Backend> {
    k: Option<Tensor<B, 4>>,
    v: Option<Tensor<B, 4>>,
}

/// Per-layer keys and values for the committed prefix. The invariant the
/// whole decode loop leans on: this cache holds exactly `len` positions,
/// every one of them a settled token.
#[derive(Debug, Clone)]
pub struct TbKvCache<B: Backend> {
    layers: Vec<LayerKv<B>>,
    /// Committed positions represented.
    pub len: usize,
}

impl<B: Backend> TbKvCache<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerKv { k: None, v: None })
                .collect(),
            len: 0,
        }
    }

    /// Keep only the first `upto` committed positions. Used whenever history
    /// is rewritten: the affected keys and values must be rebuilt, not
    /// quietly left stale.
    pub fn truncate(&mut self, upto: usize) {
        let upto = upto.min(self.len);
        for layer in &mut self.layers {
            layer.k = match &layer.k {
                Some(k) if upto > 0 => Some(k.clone().narrow(2, 0, upto)),
                _ => None,
            };
            layer.v = match &layer.v {
                Some(v) if upto > 0 => Some(v.clone().narrow(2, 0, upto)),
                _ => None,
            };
        }
        self.len = upto;
    }
}

impl<B: Backend<FloatElem = f32>> TokenBlockModel<B> {
    pub fn new(config: &TbConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        let head_dim = config.hidden_size / config.num_heads;
        let layer = || TbLayer {
            norm1: LayerNormConfig::new(config.hidden_size).init(device),
            q: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            k: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            v: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            dense: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            norm2: LayerNormConfig::new(config.hidden_size).init(device),
            fc1: LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(true)
                .init(device),
            fc2: LinearConfig::new(config.intermediate_size, config.hidden_size)
                .with_bias(true)
                .init(device),
        };
        Ok(Self {
            embedding: EmbeddingConfig::new(TB_VOCAB, config.hidden_size).init(device),
            position: EmbeddingConfig::new(config.context, config.hidden_size).init(device),
            layers: (0..config.num_layers).map(|_| layer()).collect(),
            final_norm: LayerNormConfig::new(config.hidden_size).init(device),
            num_heads: config.num_heads,
            head_dim,
        })
    }

    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        x.reshape([b, n, self.num_heads, self.head_dim])
            .swap_dims(1, 2)
    }

    /// Embed `tokens` `[b, n]` at absolute positions `[offset, offset + n)`.
    fn embed(&self, tokens: Tensor<B, 2, Int>, offset: usize, device: &B::Device) -> Tensor<B, 3> {
        let n = tokens.dims()[1];
        let tok = self.embedding.forward(tokens);
        let pos_ids =
            Tensor::<B, 1, Int>::arange(offset as i64..(offset + n) as i64, device).reshape([1, n]);
        tok + self.position.forward(pos_ids)
    }

    /// Full-sequence forward under an explicit additive bias `[b, 1, n, n]`.
    ///
    /// Training uses [`block_causal_bias`]; the certificates compare this
    /// path against [`Self::forward_cached`].
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, bias: &Tensor<B, 4>) -> Tensor<B, 3> {
        let device = tokens.device();
        let [b, n] = tokens.dims();
        let mut hidden = self.embed(tokens, 0, &device);
        for layer in &self.layers {
            let normed = layer.norm1.forward(hidden.clone());
            let q = self.split_heads(layer.q.forward(normed.clone()));
            let k = self.split_heads(layer.k.forward(normed.clone()));
            let v = self.split_heads(layer.v.forward(normed));
            let scores = q.matmul(k.swap_dims(2, 3)) + bias.clone();
            let ctx = softmax(scores, 3).matmul(v);
            let ctx = ctx
                .swap_dims(1, 2)
                .reshape([b, n, self.num_heads * self.head_dim]);
            hidden = hidden + layer.dense.forward(ctx);
            let normed2 = layer.norm2.forward(hidden.clone());
            hidden = hidden + layer.fc2.forward(exact_gelu(layer.fc1.forward(normed2)));
        }
        self.head(self.final_norm.forward(hidden))
    }

    /// Incremental forward over a new window of `tokens` `[1, n]` starting at
    /// absolute position `cache.len`.
    ///
    /// The window attends to the whole committed prefix (from the cache) and
    /// bidirectionally to itself — which is exactly the block-causal rule for
    /// a chunk that never spans a block boundary, and exactly the edit-window
    /// rule for a revision chunk. On return the window's keys and values are
    /// appended, so the caller must have settled on the window's tokens.
    pub fn forward_cached(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut TbKvCache<B>,
    ) -> Tensor<B, 3> {
        let device = tokens.device();
        let [b, n] = tokens.dims();
        assert_eq!(b, 1, "incremental decode is single-stream; got batch {b}");
        let offset = cache.len;
        let mut hidden = self.embed(tokens, offset, &device);
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = layer.norm1.forward(hidden.clone());
            let q = self.split_heads(layer.q.forward(normed.clone()));
            let k_new = self.split_heads(layer.k.forward(normed.clone()));
            let v_new = self.split_heads(layer.v.forward(normed));
            let k_full = match &cache.layers[i].k {
                Some(prev) => Tensor::cat(vec![prev.clone(), k_new], 2),
                None => k_new,
            };
            let v_full = match &cache.layers[i].v {
                Some(prev) => Tensor::cat(vec![prev.clone(), v_new.clone()], 2),
                None => v_new.clone(),
            };
            cache.layers[i].k = Some(k_full.clone());
            cache.layers[i].v = Some(v_full.clone());
            let scores = q.matmul(k_full.swap_dims(2, 3));
            let ctx = softmax(scores, 3).matmul(v_full);
            let ctx = ctx
                .swap_dims(1, 2)
                .reshape([b, n, self.num_heads * self.head_dim]);
            hidden = hidden + layer.dense.forward(ctx);
            let normed2 = layer.norm2.forward(hidden.clone());
            hidden = hidden + layer.fc2.forward(exact_gelu(layer.fc1.forward(normed2)));
        }
        cache.len = offset + n;
        self.head(self.final_norm.forward(hidden))
    }

    /// Tied output projection: logits `[b, n, vocab]`.
    fn head(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, h] = hidden.dims();
        let w = self.embedding.weight.val();
        hidden
            .reshape([b * n, h])
            .matmul(w.swap_dims(0, 1))
            .reshape([b, n, TB_VOCAB])
    }
}

/// Additive block-causal bias `[b, 1, n, n]` for fixed-size blocks: row `i`
/// sees every position of its own block and of every earlier block.
///
/// Built on the host and moved as one tensor — small (`n²` f32), inspectable,
/// and the subject of a certificate that checks it refuses future positions.
pub fn block_causal_bias<B: Backend<FloatElem = f32>>(
    n: usize,
    block_len: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let flat = host_block_causal_mask(n, block_len);
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([1, 1, n, n])
}

fn host_block_causal_mask(n: usize, block_len: usize) -> Vec<f32> {
    let mut flat = vec![NEG_BIAS; n * n];
    for i in 0..n {
        for j in 0..n {
            if TbConfig::block_of(j, block_len) <= TbConfig::block_of(i, block_len) {
                flat[i * n + j] = 0.0;
            }
        }
    }
    flat
}

// -- measurement -----------------------------------------------------------

/// Shannon entropy in nats of one prediction row.
pub fn entropy_of_row(probs: &[f32]) -> f64 {
    probs
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| {
            let q = f64::from(*p);
            -q * q.ln()
        })
        .sum()
}

/// Bidirectional cumulative entropy of the window `[start, start + len)` over
/// per-token entropies `h`: the forward pass accumulates with the leading edge
/// weighted `EDGE_WEIGHT`, the backward pass accumulates with the trailing
/// edge weighted `EDGE_WEIGHT`, and the two normalized sums are averaged.
/// This is the signal both adaptive sizing and halo gating read.
pub fn bidirectional_cumulative_entropy(h: &[f64], start: usize, len: usize) -> f64 {
    let end = (start + len).min(h.len());
    let len = end - start;
    if len == 0 {
        return 0.0;
    }
    let mut forward = 0.0;
    let mut backward = 0.0;
    let window = &h[start..end];
    for (i, value) in window.iter().enumerate() {
        let w_f = if i == 0 { EDGE_WEIGHT } else { 1.0 };
        let w_b = if i + 1 == window.len() {
            EDGE_WEIGHT
        } else {
            1.0
        };
        forward += w_f * value;
        backward += w_b * value;
    }
    let normalizer = (len as f64 - 2.0).max(0.0) + 2.0 * EDGE_WEIGHT;
    (forward / normalizer + backward / normalizer) / 2.0
}

/// Map a difficulty in `[0, 1]` to a block size in `[min_block, max_block]`:
/// low difficulty earns the largest block, high difficulty the smallest. Both
/// degenerate failure modes the instruction warns about are structural
/// impossibilities: broken estimates (difficulty 0, 1 or NaN everywhere)
/// still produce sizes inside the band.
pub fn size_from_difficulty(difficulty: f64, min_block: usize, max_block: usize) -> usize {
    let d = if difficulty.is_nan() {
        1.0
    } else {
        difficulty.clamp(0.0, 1.0)
    };
    let span = (max_block - min_block) as f64;
    let size = max_block as f64 - d * span;
    (size.round() as usize).clamp(min_block, max_block)
}

// -- decode ----------------------------------------------------------------

/// How each block's size is chosen at decode time.
#[derive(Debug, Clone, Copy)]
pub enum Sizing {
    /// One size, every block — the baseline.
    Fixed(usize),
    /// Difficulty-aware: a scout forward measures the pending region's
    /// entropy, then [`size_from_difficulty`] picks the size.
    Entropy,
}

/// What runs on top of plain block-by-block decoding.
#[derive(Debug, Clone, Copy)]
pub struct DecodePlan {
    pub sizing: Sizing,
    /// Whether an entropy-gated halo may re-open committed tail tokens.
    pub halo: bool,
    /// Routine revision pass window, in trailing blocks. `0` off.
    pub revision_blocks: usize,
}

impl DecodePlan {
    /// The plain baseline: fixed blocks, no halo, no revision.
    pub fn baseline(block_len: usize) -> Self {
        Self {
            sizing: Sizing::Fixed(block_len),
            halo: false,
            revision_blocks: 0,
        }
    }
}

/// Everything measured about one committed block. The confidence trace is the
/// instrument plan item 1 asks for: per-position, position-within-block,
/// confidence at commit.
#[derive(Debug, Clone, Default)]
pub struct BlockStat {
    pub start: usize,
    pub size: usize,
    /// Bidirectional cumulative entropy the sizing read, normalized to [0, 1].
    pub difficulty: f64,
    pub halo_len: usize,
    pub halo_edits: usize,
    pub revision_edits: usize,
    pub steps_used: usize,
    /// `(offset within block, max-probability at commit)`.
    pub committed_confidence: Vec<(usize, f64)>,
}

/// Totals for one generation. `forwards` is the wall-clock-honest cost: the
/// adaptive machinery pays for its scout and rebuild forwards here, where a
/// token-savings story alone would hide it.
#[derive(Debug, Clone, Default)]
pub struct DecodeStats {
    pub blocks: Vec<BlockStat>,
    pub forwards: usize,
    pub sizing_forwards: usize,
    pub revision_forwards: usize,
    /// How many times each position's token changed after its first commit.
    /// Two or more flips on one position is the thrash the instruction warns
    /// about; the map makes it countable instead of anecdotal.
    pub flips: BTreeMap<usize, u32>,
}

impl DecodeStats {
    /// Positions revised more than once across the whole generation.
    pub fn thrashed_positions(&self) -> usize {
        self.flips.values().filter(|c| **c >= 2).count()
    }

    pub fn total_edits(&self) -> usize {
        self.flips.values().sum::<u32>() as usize
    }
}

/// Per-row summaries pulled off the device: `(token id, max probability,
/// entropy in nats)`.
fn summarize_rows<B: Backend<FloatElem = f32>>(logits: &Tensor<B, 3>) -> Vec<(u16, f64, f64)> {
    let [_, n, _] = logits.dims();
    let logp = log_softmax(logits.clone(), 2);
    let flat: Vec<f32> = logp.into_data().convert::<f32>().iter().collect();
    (0..n)
        .map(|i| {
            let row = &flat[i * TB_VOCAB..(i + 1) * TB_VOCAB];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let arg = row
                .iter()
                .position(|v| *v == max)
                .map(|p| p as u16)
                .unwrap_or(0);
            let probs: Vec<f32> = row.iter().map(|v| v.exp()).collect();
            (arg, f64::from(max.exp()), entropy_of_row(&probs))
        })
        .collect()
}

fn ids_i64(ids: &[u16]) -> Vec<i64> {
    ids.iter().map(|t| i64::from(*t)).collect()
}

/// What one denoising window produced.
struct WindowOutcome {
    block_tokens: Vec<u16>,
    /// `(offset within block, max-probability at commit)`.
    confidences: Vec<(usize, f64)>,
    steps_used: usize,
    /// `(absolute position, new token)` staged for the committed tail.
    halo_edits: Vec<(usize, u16)>,
}

/// Where the denoising window sits: the block, plus optional halo on each
/// side.
struct Window {
    start: usize,
    len: usize,
    halo_left: usize,
    halo_right: usize,
}

/// One denoising window: `[optional halo-left] [block] [optional halo-right]`,
/// decoded against the committed prefix in the cache. The block's positions
/// start absorbed and are revealed either at the confidence threshold or, when
/// nothing clears it, one leader at a time; a block always commits complete.
fn denoise_window<B: Backend<FloatElem = f32>>(
    model: &TokenBlockModel<B>,
    cfg: &TbConfig,
    device: &B::Device,
    committed: &[u16],
    cache: &mut TbKvCache<B>,
    window: &Window,
    stats: &mut DecodeStats,
) -> Result<WindowOutcome> {
    let block_start = window.start;
    let block_len = window.len;
    let halo_left = window.halo_left;
    let window_start = block_start - halo_left;
    let window_end = (block_start + block_len + window.halo_right).min(cfg.context);
    let mut window_tokens: Vec<u16> = committed[window_start..block_start].to_vec();
    window_tokens.extend(std::iter::repeat_n(MASK_ID, block_len));
    window_tokens.extend(std::iter::repeat_n(
        MASK_ID,
        window_end - block_start - block_len,
    ));
    let window = &mut window_tokens;
    let mut revealed = vec![false; window.len()];
    revealed[..halo_left].fill(true); // the committed tail participates as itself
    let mut steps_used = 0usize;
    let mut confidences: Vec<(usize, f64)> = Vec::new();
    let mut halo_edits: Vec<(usize, u16)> = Vec::new();
    let block_span = halo_left..(halo_left + block_len).min(window.len());

    for step in 0..cfg.decode_steps {
        if revealed[block_span.clone()].iter().all(|r| *r) {
            break;
        }
        steps_used += 1;
        cache.truncate(window_start);
        let ids = ids_i64(window);
        let tokens =
            Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, window.len()]);
        let logits = model.forward_cached(tokens, cache);
        stats.forwards += 1;
        let summary = summarize_rows(&logits);
        let mut best: Option<(usize, f64, u16)> = None;
        let mut any_revealed = false;
        for i in block_span.clone() {
            if revealed[i] {
                continue;
            }
            let (tok, conf, _) = summary[i];
            if conf >= cfg.confidence_threshold {
                window[i] = tok;
                revealed[i] = true;
                confidences.push((i - halo_left, conf));
                any_revealed = true;
            } else if best.is_none_or(|(_, c, _)| conf > c) {
                best = Some((i, conf, tok));
            }
        }
        // Fast-dLLM's dead-stop rule: if nothing cleared the threshold, the
        // single most confident masked position reveals anyway so the loop
        // cannot spin on a block the model is uniformly unsure about.
        if !any_revealed {
            if let Some((i, conf, tok)) = best {
                window[i] = tok;
                revealed[i] = true;
                confidences.push((i - halo_left, conf));
            }
        }
        // On the window's final pass, stage halo-left revisions: the
        // committed tail may be rewritten once, and only once, and only
        // where the model is confident in the replacement.
        if step + 1 == cfg.decode_steps {
            for i in 0..halo_left {
                let (tok, conf, _) = summary[i];
                if tok != window[i]
                    && conf >= cfg.edit_threshold
                    && stats.flips.get(&(window_start + i)).copied().unwrap_or(0) < 1
                {
                    halo_edits.push((window_start + i, tok));
                }
            }
        }
    }
    // Steps exhausted: reveal whatever is still masked at the argmax, so the
    // block always commits complete.
    if !revealed.iter().all(|r| *r) {
        cache.truncate(window_start);
        let ids = ids_i64(window);
        let tokens =
            Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, window.len()]);
        let logits = model.forward_cached(tokens, cache);
        stats.forwards += 1;
        let summary = summarize_rows(&logits);
        for i in 0..window.len() {
            if !revealed[i] {
                let (tok, conf, _) = summary[i];
                window[i] = tok;
                if block_span.contains(&i) {
                    confidences.push((i - halo_left, conf));
                }
            }
        }
    }
    Ok(WindowOutcome {
        block_tokens: window[block_span].to_vec(),
        confidences,
        steps_used,
        halo_edits,
    })
}

/// Generate up to `total_len` tokens, one block at a time. The prompt is
/// committed clean; every later position is absorbed and denoised. Greedy
/// only: the substrate's measurements want determinism.
pub fn generate<B: Backend<FloatElem = f32>>(
    model: &TokenBlockModel<B>,
    cfg: &TbConfig,
    device: &B::Device,
    prompt: &[u16],
    total_len: usize,
    plan: &DecodePlan,
) -> Result<(Vec<u16>, DecodeStats)> {
    anyhow::ensure!(
        total_len >= prompt.len() && total_len <= cfg.context,
        "generation length {total_len} must be within {}..={}",
        prompt.len(),
        cfg.context
    );
    let mut committed: Vec<u16> = prompt.to_vec();
    let mut cache = TbKvCache::<B>::new(cfg.num_layers);
    let mut stats = DecodeStats::default();

    // The prompt is a real part of the prefix: its keys and values belong in
    // the cache, paid for honestly.
    if !prompt.is_empty() {
        let ids = ids_i64(prompt);
        let tokens =
            Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, prompt.len()]);
        model.forward_cached(tokens, &mut cache);
        stats.forwards += 1;
    }

    let mut pos = committed.len();
    while pos < total_len {
        // -- size the block ------------------------------------------------
        let (size, difficulty, halo_len) = match plan.sizing {
            Sizing::Fixed(len) => {
                let len = len.min(cfg.context - pos);
                (len, 0.0, 0)
            }
            Sizing::Entropy => {
                let scout_len = cfg.max_block.min(cfg.context - pos);
                cache.truncate(pos);
                let scout = vec![MASK_ID; scout_len];
                let ids = ids_i64(&scout);
                let tokens =
                    Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, scout_len]);
                let logits = model.forward_cached(tokens, &mut cache);
                stats.forwards += 1;
                stats.sizing_forwards += 1;
                cache.truncate(pos); // the scout's keys and values are not settled
                let summary = summarize_rows(&logits);
                let entropies: Vec<f64> = summary.iter().map(|(_, _, h)| *h).collect();
                let difficulty = bidirectional_cumulative_entropy(&entropies, 0, scout_len)
                    / (TB_VOCAB as f64).ln();
                let mut halo_len = 0usize;
                if plan.halo && difficulty >= cfg.halo_threshold {
                    halo_len = ((difficulty * cfg.halo_max as f64).ceil() as usize)
                        .min(cfg.halo_max)
                        .min(pos);
                }
                (
                    size_from_difficulty(difficulty, cfg.min_block, cfg.max_block)
                        .min(cfg.context - pos),
                    difficulty,
                    halo_len,
                )
            }
        };

        // -- denoise the block (with its optional halo) ---------------------
        let halo_right = halo_len.min((cfg.context - pos - size).min(cfg.max_block));
        let outcome = denoise_window(
            model,
            cfg,
            device,
            &committed,
            &mut cache,
            &Window {
                start: pos,
                len: size,
                halo_left: halo_len,
                halo_right,
            },
            &mut stats,
        )?;
        stats.blocks.push(BlockStat {
            start: pos,
            size,
            difficulty,
            halo_len,
            halo_edits: 0,
            revision_edits: 0,
            steps_used: outcome.steps_used,
            committed_confidence: outcome.confidences,
        });

        // -- staged halo edits, then the commit forward ---------------------
        // Edits are applied to the committed tail *before* the block's keys
        // and values are fixed, so nothing in the cache is ever computed from
        // a token that no longer exists.
        let last_block = &mut stats.blocks;
        if !outcome.halo_edits.is_empty() {
            let first_edit = outcome
                .halo_edits
                .iter()
                .map(|(p, _)| *p)
                .min()
                .unwrap_or(pos);
            let mut edits = 0usize;
            for (abs, tok) in &outcome.halo_edits {
                if committed[*abs] != *tok && stats.flips.get(abs).copied().unwrap_or(0) < 1 {
                    committed[*abs] = *tok;
                    if let Some(block_stat) = last_block.last_mut() {
                        block_stat.halo_edits += 1;
                    }
                    stats.flips.entry(*abs).and_modify(|c| *c += 1).or_insert(1);
                    edits += 1;
                }
            }
            if edits > 0 {
                cache.truncate(first_edit);
                let ids = ids_i64(&committed[first_edit..pos]);
                let tokens = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device)
                    .reshape([1, pos - first_edit]);
                model.forward_cached(tokens, &mut cache);
                stats.forwards += 1;
            }
        }
        cache.truncate(pos);
        let ids = ids_i64(&outcome.block_tokens);
        let tokens = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device)
            .reshape([1, outcome.block_tokens.len()]);
        model.forward_cached(tokens, &mut cache);
        stats.forwards += 1;
        committed.extend_from_slice(&outcome.block_tokens);
        pos += size;

        // -- routine bounded revision (plan item 2) ------------------------
        // Gated on the block's entropy signal, not unconditional: revising
        // every block reintroduces full-sequence-diffusion cost. A block whose
        // difficulty did not reach the uncertainty threshold is left alone.
        if plan.revision_blocks > 0 && difficulty >= cfg.halo_threshold {
            let edits = revision_pass(
                model,
                cfg,
                device,
                &mut committed,
                &mut cache,
                &mut stats,
                plan.revision_blocks,
            )?;
            if let Some(last) = stats.blocks.last_mut() {
                last.revision_edits = edits;
            }
        } else if plan.revision_blocks > 0 {
            // Record that the gate declined, so the cost accounting stays
            // honest (a zero-edit pass that never ran is not a revision pass).
            if let Some(last) = stats.blocks.last_mut() {
                last.revision_edits = 0;
            }
        }

        // Stop early at end-of-sequence: further blocks would be padding.
        if committed.last() == Some(&Special::Eos.id()) {
            break;
        }
    }
    Ok((committed, stats))
}

/// Re-predict the last `blocks` blocks once with full bidirectional attention
/// and apply disagreements above the edit threshold. Returns the number of
/// applied edits. The window is bounded by construction — a growing window is
/// how block diffusion stops being block diffusion.
fn revision_pass<B: Backend<FloatElem = f32>>(
    model: &TokenBlockModel<B>,
    cfg: &TbConfig,
    device: &B::Device,
    committed: &mut [u16],
    cache: &mut TbKvCache<B>,
    stats: &mut DecodeStats,
    blocks: usize,
) -> Result<usize> {
    let window_len = (blocks * cfg.max_block).min(committed.len());
    let start = committed.len() - window_len;
    cache.truncate(start);
    let ids = ids_i64(&committed[start..]);
    let tokens = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, window_len]);
    let logits = model.forward_cached(tokens, cache);
    stats.forwards += 1;
    stats.revision_forwards += 1;
    let summary = summarize_rows(&logits);
    let mut edits = 0usize;
    for (i, (tok, conf, _)) in summary.iter().enumerate() {
        let abs = start + i;
        if *conf >= cfg.edit_threshold && *tok != committed[abs] {
            // Thrash guard: a position that already changed once this
            // generation is not revised again. Flipping the same token back
            // and forth across later blocks is an instability, not a feature.
            if stats.flips.get(&abs).copied().unwrap_or(0) >= 1 {
                continue;
            }
            stats.flips.entry(abs).and_modify(|c| *c += 1).or_insert(1);
            committed[abs] = *tok;
            edits += 1;
        }
    }
    if edits > 0 {
        // The cache now holds keys and values for tokens that no longer
        // exist. Rebuild the window honestly rather than leaving stale state.
        cache.truncate(start);
        let ids = ids_i64(&committed[start..]);
        let tokens =
            Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, window_len]);
        model.forward_cached(tokens, cache);
        stats.forwards += 1;
        stats.revision_forwards += 1;
    }
    Ok(edits)
}

// -- training --------------------------------------------------------------

/// One corrupted training window: input tokens (some absorbed), the clean
/// targets, and the loss mask (1.0 where the position was absorbed).
#[derive(Debug, Clone)]
pub struct Corrupted {
    pub input: Vec<u16>,
    pub target: Vec<u16>,
    pub mask: Vec<f32>,
}

/// Absorb each token with probability equal to its noise level, one
/// independent level per token — the linear schedule BD3-LM trains with.
pub fn corrupt(tokens: &[u16], rng: &mut StdRng) -> Corrupted {
    let mut input = tokens.to_vec();
    let mut mask = vec![0.0f32; tokens.len()];
    for (i, t) in input.iter_mut().enumerate() {
        let level: f64 = rng.random::<f64>();
        if rng.random::<f64>() < level {
            *t = MASK_ID;
            mask[i] = 1.0;
        }
    }
    Corrupted {
        input,
        target: tokens.to_vec(),
        mask,
    }
}

/// Weighted cross-entropy over absorbed positions: the substrate's objective.
pub fn denoising_loss<B: AutodiffBackend<FloatElem = f32>>(
    model: &TokenBlockModel<B>,
    input: Tensor<B, 2, Int>,
    target: Tensor<B, 2, Int>,
    mask: Tensor<B, 2>,
    block_len: usize,
) -> Tensor<B, 1> {
    let [b, n] = input.dims();
    let bias = block_causal_bias::<B>(n, block_len, &input.device());
    let logits = model.forward(input, &bias);
    let flat_logits = logits.reshape([b * n, TB_VOCAB]);
    let flat_target = target.reshape([b * n, 1]);
    let logp = log_softmax(flat_logits, 1);
    let nll = -logp.gather(1, flat_target).squeeze_dim::<1>(1);
    let keep = mask.reshape([b * n]);
    let counted = keep.clone().sum().clamp_min(1.0);
    (nll * keep).sum() / counted
}

/// Non-overlapping `context`-length windows; the remainder is dropped rather
/// than padded — a padded position is a trained-on lie.
pub fn build_windows(stream: &[u16], context: usize) -> Vec<Vec<u16>> {
    stream
        .chunks(context)
        .filter(|w| w.len() == context)
        .map(<[u16]>::to_vec)
        .collect()
}

/// Summary of a training run. `val_curve` is the convergence evidence: loss
/// against steps, on windows the trainer never optimized.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TbTrainReport {
    pub steps_run: usize,
    pub train_curve: Vec<(usize, f64)>,
    pub val_curve: Vec<(usize, f64)>,
    pub final_val_loss: f64,
    pub final_val_bpb: f64,
    pub seconds: f64,
}

/// The training loop's knobs, grouped so the signature stays readable.
#[derive(Debug, Clone, Copy)]
pub struct TbTrainParams {
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub eval_every: usize,
    pub seed: u64,
}

/// Train the substrate with AdamW. Deterministic under a fixed seed.
pub fn train_model<B: AutodiffBackend<FloatElem = f32>>(
    config: &TbConfig,
    device: &B::Device,
    train_windows: &[Vec<u16>],
    val_windows: &[Vec<u16>],
    params: &TbTrainParams,
) -> Result<(TokenBlockModel<B>, TbTrainReport)> {
    let TbTrainParams {
        steps,
        batch_size,
        lr,
        eval_every,
        seed,
    } = *params;
    config.validate()?;
    anyhow::ensure!(!train_windows.is_empty(), "no training windows");
    anyhow::ensure!(!val_windows.is_empty(), "no validation windows");
    anyhow::ensure!(
        eval_every >= 1,
        "eval_every ({eval_every}) must be at least 1"
    );
    anyhow::ensure!(
        batch_size >= 1,
        "batch_size ({batch_size}) must be at least 1"
    );
    let mut model = TokenBlockModel::<B>::new(config, device)?;
    let mut optim = AdamWConfig::new().init();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut report = TbTrainReport::default();
    let started = Instant::now();
    let n = config.context;
    for step in 0..steps {
        let mut input = Vec::with_capacity(batch_size * n);
        let mut target = Vec::with_capacity(batch_size * n);
        let mut mask = Vec::with_capacity(batch_size * n);
        for _ in 0..batch_size {
            let idx = rng.random_range(0..train_windows.len());
            let c = corrupt(&train_windows[idx], &mut rng);
            input.extend_from_slice(&c.input);
            target.extend_from_slice(&c.target);
            mask.extend_from_slice(&c.mask);
        }
        let input =
            Tensor::<B, 1, Int>::from_ints(input.as_slice(), device).reshape([batch_size, n]);
        let target =
            Tensor::<B, 1, Int>::from_ints(target.as_slice(), device).reshape([batch_size, n]);
        let mask = Tensor::<B, 1>::from_floats(mask.as_slice(), device).reshape([batch_size, n]);
        let loss = denoising_loss(&model, input, target, mask, config.block_len);
        let train_loss = loss.clone().into_scalar() as f64;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads);
        if step % eval_every == 0 || step + 1 == steps {
            let val_loss = evaluate_loss(config, &model, device, val_windows, &mut rng);
            report.train_curve.push((step, train_loss));
            report.val_curve.push((step, val_loss));
        }
    }
    report.steps_run = steps;
    let final_val = evaluate_loss(config, &model, device, val_windows, &mut rng);
    report.final_val_loss = final_val;
    report.final_val_bpb = final_val / 2f64.ln();
    report.seconds = started.elapsed().as_secs_f64();
    Ok((model, report))
}

/// Mean denoising loss over the validation windows, deterministic corruption.
pub fn evaluate_loss<B: AutodiffBackend<FloatElem = f32>>(
    config: &TbConfig,
    model: &TokenBlockModel<B>,
    device: &B::Device,
    val_windows: &[Vec<u16>],
    rng: &mut StdRng,
) -> f64 {
    let mut total = 0.0f64;
    let mut count = 0.0f64;
    for window in val_windows {
        let c = corrupt(window, rng);
        let ids = ids_i64(&c.input);
        let input =
            Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, config.context]);
        let target_ids = ids_i64(&c.target);
        let target = Tensor::<B, 1, Int>::from_ints(target_ids.as_slice(), device)
            .reshape([1, config.context]);
        let mask =
            Tensor::<B, 1>::from_floats(c.mask.as_slice(), device).reshape([1, config.context]);
        let loss = denoising_loss(model, input, target, mask, config.block_len);
        total += loss.into_scalar() as f64;
        count += 1.0;
    }
    if count == 0.0 {
        0.0
    } else {
        total / count
    }
}

// -- certificates ----------------------------------------------------------

/// The `tokenblocks` certificate group. Every structural claim this module
/// makes is pinned here, in the crate's one place for load-bearing identity
/// checks.
pub fn certificates() -> Vec<crate::verify::Certificate> {
    use crate::verify::Certificate;
    type Nd = burn::backend::NdArray<f32>;
    let device = Default::default();

    // 1. The block-causal mask refuses future positions, exactly.
    let (n, len) = (16usize, 4usize);
    let flat = host_block_causal_mask(n, len);
    let mut violations = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let future = TbConfig::block_of(j, len) > TbConfig::block_of(i, len);
            let allowed = flat[i * n + j] == 0.0;
            if future == allowed {
                violations += 1.0;
            }
        }
    }

    // 2. Difficulty sizing stays inside its band for every input class.
    let mut out_of_band = 0.0f64;
    let (min_b, max_b) = (4usize, 12usize);
    for d in [0.0, -3.0, 0.25, 0.5, 0.9, 1.0, 42.0, f64::NAN] {
        let s = size_from_difficulty(d, min_b, max_b);
        if s < min_b || s > max_b {
            out_of_band += 1.0;
        }
    }

    // 3. Bidirectional cumulative entropy matches a hand computation:
    //    forward: 2*0.5 + 1.0 + 0.0 = 2.0; backward: 0.5 + 1.0 + 2*0.0 = 1.5;
    //    normalizer: (3-2) + 2*2 = 5; mean: (2/5 + 1.5/5)/2 = 0.35.
    let got = bidirectional_cumulative_entropy(&[0.5, 1.0, 0.0, 2.0], 0, 3);
    let entropy_residual = (got - 0.35).abs();

    // 4. Prefix invariance: logits on blocks 0..k are unchanged when a later
    //    block's tokens change. Block-causal attention cannot see them.
    let config = TbConfig::tiny();
    let prefix_residual = match TokenBlockModel::<Nd>::new(&config, &device) {
        Err(err) => {
            eprintln!("verify: tokenblocks/prefix_invariance: {err:#}");
            f64::INFINITY
        }
        Ok(model) => {
            let seq: Vec<u16> = (0..config.context).map(|i| (i % 200) as u16).collect();
            // Perturb exactly the final block, and compare only strictly
            // earlier blocks. Perturbing from context - max_block would cross
            // a block boundary (positions in the compared region can share a
            // block with perturbed positions), and the model is then RIGHT to
            // differ — that mistake is why the certificate once failed.
            let mut perturbed = seq.clone();
            for token in &mut perturbed[config.context - config.block_len..] {
                *token = (*token + 37) % 200;
            }
            let bias = block_causal_bias::<Nd>(config.context, config.block_len, &device);
            let la = model.forward(
                Tensor::<Nd, 1, Int>::from_ints(seq.as_slice(), &device)
                    .reshape([1, config.context]),
                &bias,
            );
            let lb = model.forward(
                Tensor::<Nd, 1, Int>::from_ints(perturbed.as_slice(), &device)
                    .reshape([1, config.context]),
                &bias,
            );
            let seen = config.context - config.block_len;
            let da: Vec<f32> = la
                .narrow(1, 0, seen)
                .into_data()
                .convert::<f32>()
                .iter()
                .collect();
            let db: Vec<f32> = lb
                .narrow(1, 0, seen)
                .into_data()
                .convert::<f32>()
                .iter()
                .collect();
            da.iter()
                .zip(db.iter())
                .map(|(a, b)| ((*a - *b) as f64).abs())
                .fold(0.0f64, f64::max)
        }
    };

    // 5. Cache equivalence: block-by-block decoding with the key/value cache
    //    reproduces the one-shot forward over the whole sequence.
    let cache_residual = match TokenBlockModel::<Nd>::new(&config, &device) {
        Err(err) => {
            eprintln!("verify: tokenblocks/kv_cache: {err:#}");
            f64::INFINITY
        }
        Ok(model) => {
            let seq: Vec<u16> = (0..config.context).map(|i| (i % 200) as u16).collect();
            let bias = block_causal_bias::<Nd>(config.context, config.block_len, &device);
            let one_shot = model.forward(
                Tensor::<Nd, 1, Int>::from_ints(seq.as_slice(), &device)
                    .reshape([1, config.context]),
                &bias,
            );
            let mut cache = TbKvCache::<Nd>::new(config.num_layers);
            let mut cached: Option<Tensor<Nd, 3>> = None;
            let mut offset = 0usize;
            while offset < config.context {
                let end = (offset + config.block_len).min(config.context);
                let ids = ids_i64(&seq[offset..end]);
                let tokens = Tensor::<Nd, 1, Int>::from_ints(ids.as_slice(), &device)
                    .reshape([1, end - offset]);
                let out = model.forward_cached(tokens, &mut cache);
                cached = Some(match cached {
                    None => out,
                    Some(prev) => Tensor::cat(vec![prev, out], 1),
                });
                offset = end;
            }
            match cached {
                None => f64::INFINITY,
                Some(c) => {
                    let a: Vec<f32> = c.into_data().convert::<f32>().iter().collect();
                    let b: Vec<f32> = one_shot.into_data().convert::<f32>().iter().collect();
                    a.iter()
                        .zip(b.iter())
                        .map(|(x, y)| ((*x - *y) as f64).abs())
                        .fold(0.0f64, f64::max)
                }
            }
        }
    };

    vec![
        Certificate {
            group: "tokenblocks",
            name: "block_causal_mask_refuses_future_positions",
            theorem: "Under the fixed-size block-causal bias, a row may attend to its \
                      own block and earlier blocks only — every future position is \
                      excluded exactly.",
            residual: violations,
            tolerance: 0.0,
        },
        Certificate {
            group: "tokenblocks",
            name: "difficulty_sizing_stays_bounded",
            theorem: "The entropy-difficulty size map returns a block size within \
                      [min_block, max_block] for every difficulty, including NaN, \
                      negative and out-of-range values.",
            residual: out_of_band,
            tolerance: 0.0,
        },
        Certificate {
            group: "tokenblocks",
            name: "bidirectional_entropy_matches_hand_computation",
            theorem: "The bidirectional cumulative entropy of [0.5, 1.0, 0.0] with \
                      edge weight 2 is 0.35 exactly.",
            residual: entropy_residual,
            tolerance: 1e-9,
        },
        Certificate {
            group: "tokenblocks",
            name: "prefix_invariance_under_later_block_perturbation",
            theorem: "Perturbing the tokens of the final block leaves every strictly \
                      earlier block's logits unchanged: block-causal attention \
                      cannot see them.",
            residual: prefix_residual,
            tolerance: 1e-5,
        },
        Certificate {
            group: "tokenblocks",
            name: "kv_cache_matches_full_forward",
            theorem: "Decoding block by block with the per-layer key/value cache \
                      reproduces the one-shot block-causal forward over the whole \
                      sequence.",
            residual: cache_residual,
            tolerance: 1e-4,
        },
    ]
}

// -- corpus + benchmark harness ---------------------------------------------

/// Read at most `max_bytes` from `path`, split into line-delimited documents,
/// and cut each split into non-overlapping windows. The split unit is the
/// document (a JSONL line), never the window — adjacent windows from one
/// document may overlap in content, so a split that mixed windows from one
/// document across train and test would leak. 70/15/15 by document, then
/// windows per split.
/// One 70/15/15 split of the byte stream, cut into windows per split.
pub struct StreamWindows {
    pub train: Vec<Vec<u16>>,
    pub val: Vec<Vec<u16>>,
    pub test: Vec<Vec<u16>>,
}

pub fn load_stream(
    path: &std::path::Path,
    max_bytes: usize,
    context: usize,
) -> Result<StreamWindows> {
    use crate::tokenizer::ByteTokenizer;
    use std::io::Read as _;
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut raw = Vec::with_capacity(max_bytes);
    (&mut file)
        .take(max_bytes as u64)
        .read_to_end(&mut raw)
        .with_context(|| format!("read {}", path.display()))?;
    let text = String::from_utf8_lossy(&raw);
    let docs: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    anyhow::ensure!(
        docs.len() >= 20,
        "corpus too small: {} documents",
        docs.len()
    );
    let n_train = docs.len() * 7 / 10;
    let n_val = docs.len() * 15 / 100;
    let tok = ByteTokenizer::new();
    let split_windows = |slice: &[&str]| -> Vec<Vec<u16>> {
        let mut stream = Vec::new();
        for doc in slice {
            stream.extend(tok.encode_document(doc));
        }
        build_windows(&stream, context)
    };
    let train = split_windows(&docs[..n_train]);
    let val = split_windows(&docs[n_train..n_train + n_val]);
    let test = split_windows(&docs[n_train + n_val..]);
    anyhow::ensure!(
        !train.is_empty(),
        "no training windows from {}",
        path.display()
    );
    anyhow::ensure!(
        !val.is_empty(),
        "no validation windows from {}",
        path.display()
    );
    anyhow::ensure!(!test.is_empty(), "no test windows from {}", path.display());
    Ok(StreamWindows { train, val, test })
}

/// Teacher-forced NLL of a generated continuation: one forward over the
/// revealed sequence, log-probabilities read at the continuation positions.
/// The same protocol for every decode plan, so plan scores are comparable.
pub fn score_continuation<B: Backend<FloatElem = f32>>(
    model: &TokenBlockModel<B>,
    cfg: &TbConfig,
    device: &B::Device,
    prompt: &[u16],
    continuation: &[u16],
) -> Result<f64> {
    let mut tokens = prompt.to_vec();
    tokens.extend_from_slice(continuation);
    anyhow::ensure!(
        tokens.len() <= cfg.context,
        "prompt + continuation exceeds context"
    );
    let ids = ids_i64(&tokens);
    let n = tokens.len();
    let input = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, n]);
    let bias = block_causal_bias::<B>(n, cfg.block_len, device);
    let logits = model.forward(input, &bias);
    let logp = log_softmax(logits, 2);
    let flat: Vec<f32> = logp.into_data().convert::<f32>().iter().collect();
    let mut total = 0.0f64;
    for (k, t) in continuation.iter().enumerate() {
        let pos = prompt.len() + k;
        total -= f64::from(flat[pos * TB_VOCAB + usize::from(*t)]);
    }
    Ok(total / continuation.len().max(1) as f64)
}

/// One decode plan's numbers over a set of prompts.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanResult {
    pub name: String,
    pub seconds: f64,
    pub forwards: usize,
    pub sizing_forwards: usize,
    pub revision_forwards: usize,
    pub blocks: usize,
    /// Mean max-probability over every committed token.
    pub mean_confidence: f64,
    /// Mean teacher-forced NLL of the continuations (lower is better).
    pub continuation_nll: f64,
    pub continuation_bpb: f64,
    /// Block-size histogram.
    pub sizes: BTreeMap<usize, usize>,
    /// Mean committed confidence per offset-within-block. The plan-item-1
    /// instrument: the position-vs-confidence curve.
    pub confidence_by_offset: Vec<(usize, f64)>,
    pub edits_total: usize,
    pub thrashed_positions: usize,
}

/// The six plans the benchmark compares: three fixed references (the small
/// and large ends of the adaptive band, and the default) plus the adaptive
/// mechanism, then halo, then the bounded revision pass.
pub fn benchmark_plans(cfg: &TbConfig) -> Vec<(String, DecodePlan)> {
    vec![
        (
            format!("fixed-{}", cfg.block_len),
            DecodePlan::baseline(cfg.block_len),
        ),
        (
            format!("fixed-{}", cfg.min_block),
            DecodePlan::baseline(cfg.min_block),
        ),
        (
            format!("fixed-{}", cfg.max_block),
            DecodePlan::baseline(cfg.max_block),
        ),
        (
            "adaptive".to_string(),
            DecodePlan {
                sizing: Sizing::Entropy,
                halo: false,
                revision_blocks: 0,
            },
        ),
        (
            "adaptive+halo".to_string(),
            DecodePlan {
                sizing: Sizing::Entropy,
                halo: true,
                revision_blocks: 0,
            },
        ),
        (
            "adaptive+halo+revision".to_string(),
            DecodePlan {
                sizing: Sizing::Entropy,
                halo: true,
                revision_blocks: cfg.revision_blocks.max(1),
            },
        ),
    ]
}

/// Run one plan over every prompt: generation, wall-clock, forwards, the
/// confidence-vs-position instrument, and the continuation quality proxy.
pub fn run_plan<B: Backend<FloatElem = f32>>(
    name: &str,
    model: &TokenBlockModel<B>,
    cfg: &TbConfig,
    device: &B::Device,
    plan: &DecodePlan,
    prompts: &[Vec<u16>],
    gen_len: usize,
) -> Result<PlanResult> {
    let started = Instant::now();
    let mut all_stats = DecodeStats::default();
    let mut nlls = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let target = (prompt.len() + gen_len).min(cfg.context);
        let (out, stats) = generate(model, cfg, device, prompt, target, plan)?;
        all_stats.forwards += stats.forwards;
        all_stats.sizing_forwards += stats.sizing_forwards;
        all_stats.revision_forwards += stats.revision_forwards;
        all_stats.blocks.extend(stats.blocks.iter().cloned());
        for (pos, count) in &stats.flips {
            *all_stats.flips.entry(*pos).or_insert(0) += count;
        }
        let prompt_len = prompt.len();
        let continuation = &out[prompt_len..];
        if !continuation.is_empty() {
            nlls.push(score_continuation(
                model,
                cfg,
                device,
                prompt,
                continuation,
            )?);
        }
    }
    let seconds = started.elapsed().as_secs_f64();

    let mut sizes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut conf_by_offset: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut conf_total = 0.0f64;
    let mut conf_count = 0usize;
    for block in &all_stats.blocks {
        *sizes.entry(block.size).or_insert(0) += 1;
        for (offset, conf) in &block.committed_confidence {
            conf_by_offset.entry(*offset).or_default().push(*conf);
            conf_total += *conf;
            conf_count += 1;
        }
    }
    let confidence_by_offset = conf_by_offset
        .into_iter()
        .map(|(offset, values)| {
            let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
            (offset, mean)
        })
        .collect();
    let continuation_nll = if nlls.is_empty() {
        0.0
    } else {
        nlls.iter().sum::<f64>() / nlls.len() as f64
    };
    Ok(PlanResult {
        name: name.to_string(),
        seconds,
        forwards: all_stats.forwards,
        sizing_forwards: all_stats.sizing_forwards,
        revision_forwards: all_stats.revision_forwards,
        blocks: all_stats.blocks.len(),
        mean_confidence: if conf_count == 0 {
            0.0
        } else {
            conf_total / conf_count as f64
        },
        continuation_nll,
        continuation_bpb: continuation_nll / 2f64.ln(),
        sizes,
        confidence_by_offset,
        edits_total: all_stats.total_edits(),
        thrashed_positions: all_stats.thrashed_positions(),
    })
}

/// Save a checkpoint the way the rest of the crate does.
pub fn save_checkpoint<B: Backend>(
    model: TokenBlockModel<B>,
    path: &std::path::Path,
) -> Result<()> {
    type Rec = burn::record::NamedMpkFileRecorder<burn::record::FullPrecisionSettings>;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    model
        .save_file(path, &Rec::new())
        .map_err(|err| anyhow::anyhow!("save {}: {err}", path.display()))
}

/// Load a checkpoint saved by [`save_checkpoint`].
pub fn load_checkpoint(
    path: &std::path::Path,
    config: &TbConfig,
    device: &<burn::backend::NdArray<f32> as burn::tensor::backend::BackendTypes>::Device,
) -> Result<TokenBlockModel<burn::backend::NdArray<f32>>> {
    type Rec = burn::record::NamedMpkFileRecorder<burn::record::FullPrecisionSettings>;
    type Nd = burn::backend::NdArray<f32>;
    TokenBlockModel::<Nd>::new(config, device)?
        .load_file(path, &Rec::new(), device)
        .map_err(|err| anyhow::anyhow!("load {}: {err}", path.display()))
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it.
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable
)]
mod tests {
    use super::*;
    use burn::module::AutodiffModule;

    type Nd = burn::backend::NdArray<f32>;
    type Ad = burn::backend::Autodiff<Nd>;

    fn device() -> <Nd as burn::tensor::backend::BackendTypes>::Device {
        Default::default()
    }

    fn pattern(config: &TbConfig) -> Vec<u16> {
        (0..config.context).map(|i| ((i % 16) + 1) as u16).collect()
    }

    #[test]
    fn mask_builder_matches_host_copy() {
        let device = device();
        let t = block_causal_bias::<Nd>(12, 4, &device);
        assert_eq!(t.dims(), [1, 1, 12, 12]); // broadcast over batch and heads
        let data: Vec<f32> = t.into_data().convert::<f32>().iter().collect();
        assert_eq!(data, host_block_causal_mask(12, 4));
    }

    #[test]
    fn corrupt_absorbs_a_sane_fraction() {
        let mut rng = StdRng::seed_from_u64(7);
        let tokens: Vec<u16> = (0..64).map(|i| (i % 200) as u16).collect();
        let c = corrupt(&tokens, &mut rng);
        let masked = c.mask.iter().filter(|m| **m > 0.0).count();
        assert!(masked > 8 && masked < 56, "masked {masked} of 64");
        assert_eq!(c.target, tokens);
    }

    #[test]
    fn sizing_never_leaves_the_band() {
        for d in [-1.0, 0.0, 0.5, 1.0, 9.0, f64::NAN] {
            let s = size_from_difficulty(d, 4, 12);
            assert!((4..=12).contains(&s), "size {s} outside the band for d={d}");
        }
    }

    #[test]
    fn forward_shapes_hold() {
        let device = device();
        let config = TbConfig::tiny();
        let model = TokenBlockModel::<Nd>::new(&config, &device).unwrap();
        let n = config.context;
        let tokens = pattern(&config);
        let bias = block_causal_bias::<Nd>(n, config.block_len, &device);
        // Two identical rows exercise the batch broadcast in the bias.
        let ids: Vec<i64> = tokens
            .iter()
            .chain(tokens.iter())
            .map(|t| i64::from(*t))
            .collect();
        let out = model.forward(
            Tensor::<Nd, 1, Int>::from_ints(ids.as_slice(), &device).reshape([2, n]),
            &bias,
        );
        assert_eq!(out.dims(), [2, n, TB_VOCAB]);
    }

    #[test]
    fn tiny_training_reduces_loss() {
        let device = device();
        let config = TbConfig::tiny();
        let window = pattern(&config);
        let windows = vec![window.clone(); 8];
        let val = vec![window.clone(); 2];
        let (model, report) = train_model::<Ad>(
            &config,
            &device,
            &windows,
            &val,
            &TbTrainParams {
                steps: 30,
                batch_size: 4,
                lr: 3e-3,
                eval_every: 10,
                seed: 42,
            },
        )
        .unwrap();
        assert!(
            report.final_val_loss < report.val_curve[0].1,
            "validation loss must fall: first {} final {}",
            report.val_curve[0].1,
            report.final_val_loss
        );
        // Deterministic generation runs and returns a full sequence. The
        // trainer hands back the autodiff-wrapped model; decode runs on the
        // plain inner backend.
        let inference = model.valid();
        let plan = DecodePlan::baseline(config.block_len);
        let (out, stats) = generate::<Nd>(
            &inference,
            &config,
            &device,
            &window[..8],
            config.context,
            &plan,
        )
        .unwrap();
        assert_eq!(out.len(), config.context);
        assert!(!stats.blocks.is_empty());
    }

    #[test]
    fn adaptive_decode_runs_and_stays_bounded() {
        let device = device();
        let config = TbConfig::tiny();
        let model = TokenBlockModel::<Nd>::new(&config, &device).unwrap();
        let window = pattern(&config);
        let plan = DecodePlan {
            sizing: Sizing::Entropy,
            halo: true,
            revision_blocks: 1,
        };
        let (_, stats) = generate::<Nd>(
            &model,
            &config,
            &device,
            &window[..8],
            config.context,
            &plan,
        )
        .unwrap();
        for block in &stats.blocks {
            assert!(
                (config.min_block..=config.max_block).contains(&block.size),
                "block size {} outside the band",
                block.size
            );
            assert!(block.halo_len <= config.halo_max);
        }
        assert!(stats.thrashed_positions() <= stats.total_edits());
    }
}
