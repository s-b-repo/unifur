//! The multiscale residual-VQ tokenizer: stage 1 of Next-Concept Modeling.
//!
//! The tokenizer turns a token window `x` of length `L` into `K` levels of
//! discrete codes, coarse to fine, and back. Everything the generator needs
//! from it goes through [`super::ConceptCodec`] and [`super::SoftConceptCodec`].
//!
//! # Architecture (what is built here, precisely)
//!
//! - **Encoder `E`.** Token embedding plus a learned absolute position table,
//!   then `encoder_blocks` bidirectional pre-norm transformer blocks
//!   ([`NcmBlock`]). With `encoder_blocks = 0` (the paper's setting) the
//!   encoder is exactly `z_i = emb(x_i) + pos_i`. Output `z ∈ R^{L×d}`.
//! - **`Down_k` (`L → ℓ_k`).** Mean pooling over contiguous windows. Window
//!   `i` of level `k` covers positions `[⌊iL/ℓ_k⌋, ⌊(i+1)L/ℓ_k⌋)`; for
//!   `ℓ_k ≤ L` every window is non-empty, the windows tile `[0, L)`, and their
//!   sizes differ by at most one (they are `⌊L/ℓ_k⌋` or `⌈L/ℓ_k⌉`). That
//!   definition is the same whether or not `ℓ_k` divides `L`
//!   ([`level_windows`]).
//! - **`Up_k` (`ℓ_k → L`).** Nearest (piecewise-constant) upsampling over the
//!   same windows: position `j` reads the vector of the window that owns it.
//!   `Down_k ∘ Up_k` is the identity.
//! - **`R_k`.** `R_k(v) = Up_k(v + v W_k + b_k)`: upsample plus a position-wise
//!   linear map, written as a residual so that with `W_k, b_k` zero-initialised
//!   `R_k` starts as plain upsampling. The linear part is applied at the coarse
//!   resolution, which is the same map because a position-wise linear map
//!   commutes with row selection. `R_k` is affine in `v`.
//! - **Codebooks.** Each level owns `codebook_size` codes of width `d`, stored
//!   as their EMA state (see below); [`NcmTokenizer::codebook`] materialises
//!   them.
//! - **Decoder `ψ`.** `ẑ + pos'` (its own learned position table), then
//!   `decoder_blocks` bidirectional blocks, a final LayerNorm and a linear LM
//!   head over the vocabulary.
//!
//! # Residual quantization
//!
//! With `z^(0) = z` and `ẑ^(0) = 0`, for `k = 1..K`:
//!
//! ```text
//! u_k     = Down_k(z^(k-1))
//! r_k^i   = argmin_j ‖u_k^i − e_{k,j}‖²
//! ẑ^(k)   = ẑ^(k-1) + R_k(e_k(r_k))
//! z^(k)   = z^(k-1) − R_k(e_k(r_k))          (= z − ẑ^(k))
//! ```
//!
//! The residual is carried by running subtraction, so `ẑ^(K) + z^(K) = z`
//! is a genuine check that every contribution added to the reconstruction
//! was also removed from the residual (`residual_reconstruction_telescopes`).
//!
//! # Training objective
//!
//! `L_tok = −(1/L) Σ_i log p_ψ(x_i | ẑ_st^(c)) + β·MSE(z, sg[ẑ^(c)]) +
//! MSE(sg[z], ẑ^(c))`, where:
//!
//! - `c` is the **nested-dropout** cutoff, drawn per sample: with probability
//!   `nested_dropout` it is uniform on `1..=K`, otherwise `c = K`. Levels
//!   `1..c` are kept, the rest contribute exactly zero
//!   (`nested_dropout_truncates_exactly`).
//! - `ẑ_st = ẑ + (z − sg[z])` is the **straight-through** estimator: its value
//!   is `ẑ` bit for bit and its gradient with respect to `z` is the gradient
//!   with respect to `ẑ` (`straight_through_gradient_is_identity`).
//! - **Codebooks are updated by EMA, not by gradient.** Per level, with decay
//!   `γ` and the crate's convention that the decay weights the *retained*
//!   state (as in [`crate::schedule::Ema`]): `N ← γN + (1−γ)n`,
//!   `m ← γm + (1−γ)s`, where `n_j` counts the vectors assigned to code `j`
//!   in the step and `s_j` sums them. The code is `e_j = m_j / Ñ_j` with
//!   Laplace smoothing `Ñ_j = (N_j + ε) / (Σ_l N_l + Cε) · Σ_l N_l`. So the
//!   codebook term `MSE(sg[z], ẑ)` does **not** move the codebooks (they have
//!   no gradient); its gradient trains the only trainable parameters inside
//!   `ẑ`, the `R_k` linear maps. This is the standard EMA substitution for
//!   the codebook loss (van den Oord et al.), stated rather than implied.
//! - **Dead-code restart.** Every `restart_every` optimizer steps, a code
//!   whose EMA count `N_j` is below `dead_code_threshold` times the uniform
//!   share `Σ N / C` is re-seeded from a randomly chosen pooled encoder
//!   vector `u_k` of the current batch (plus `1e-3`-relative noise), with
//!   `N_j` reset to the uniform share. The same mechanism seeds every
//!   codebook from data before the first step
//!   ([`NcmTokenizer::init_codebooks_from_data`]), level by level so each
//!   level is seeded from the residual its predecessors leave.
//!
//! # Honesty about what is measured
//!
//! The certificates in [`certificate_checks`] check the identities above.
//! The paper's quality and speed results (10–31× fewer network calls than
//! diffusion LMs) are *not* measured by anything in this module; the trainer
//! in [`train`] measures reconstruction cross-entropy and token accuracy per
//! cutoff level, and codebook usage, on held-out data, and nothing else.

use super::{ConceptCodec, LevelCodes, LevelSchedule, SoftConceptCodec};
use crate::tensor_ext::exact_gelu;
use anyhow::{bail, ensure, Context as _, Result};
use burn::module::{Module, Param, ParamId};
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, IndexingUpdateOp, Int, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};

pub mod train;

fn default_mlp_ratio() -> usize {
    4
}

fn default_ema_epsilon() -> f64 {
    1e-5
}

fn default_restart_every() -> usize {
    100
}

fn default_layer_norm_eps() -> f64 {
    1e-5
}

/// Architecture and quantizer settings of the tokenizer.
///
/// The presets are [`Self::tiny`], [`Self::paper_owt`] and [`Self::nina_4k`];
/// a JSON file with these fields is the fourth way to get one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NcmTokenizerConfig {
    /// Token vocabulary: 259 for the crate's byte tokenizer (256 bytes plus
    /// BOS/EOS/PAD, [`crate::tokenizer::VOCAB_SIZE`]).
    pub vocab: usize,
    /// Latent width `d` of `z`, `ẑ` and every code.
    pub latent_dim: usize,
    /// Codes per level, `|V_k|`.
    pub codebook_size: usize,
    /// Level lengths `ℓ_1 < … < ℓ_K = L`.
    pub schedule: LevelSchedule,
    /// Bidirectional blocks in the encoder (the paper uses 0).
    pub encoder_blocks: usize,
    /// Bidirectional blocks in the decoder (the paper uses 1).
    pub decoder_blocks: usize,
    /// Attention heads per block; must divide `latent_dim`.
    pub heads: usize,
    /// MLP hidden width as a multiple of `latent_dim`.
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: usize,
    /// Codebook EMA decay `γ ∈ (0, 1)`; weights the retained state.
    pub ema_decay: f64,
    /// Laplace smoothing `ε` of the EMA cluster sizes.
    #[serde(default = "default_ema_epsilon")]
    pub ema_epsilon: f64,
    /// Commitment weight `β` on `MSE(z, sg[ẑ])`.
    pub commitment_beta: f64,
    /// Probability that a sample's cutoff is drawn uniformly from `1..=K`
    /// rather than fixed at `K`.
    pub nested_dropout: f64,
    /// A code is dead when its EMA count is below this fraction of the
    /// uniform share `Σ N / C`; 0 disables restarts.
    pub dead_code_threshold: f64,
    /// Optimizer steps between dead-code sweeps.
    #[serde(default = "default_restart_every")]
    pub restart_every: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

impl NcmTokenizerConfig {
    /// A CPU-sized configuration that trains in minutes: `L = 64`, `K = 6`
    /// (lengths 2, 4, 8, 16, 32, 64), `d = 64`, 256 codes per level, no
    /// encoder blocks and one decoder block.
    pub fn tiny() -> Result<Self> {
        Ok(Self {
            vocab: crate::tokenizer::VOCAB_SIZE,
            latent_dim: 64,
            codebook_size: 256,
            schedule: LevelSchedule::geometric(64, 6)?,
            encoder_blocks: 0,
            decoder_blocks: 1,
            heads: 4,
            mlp_ratio: 4,
            ema_decay: 0.95,
            ema_epsilon: 1e-5,
            commitment_beta: 0.25,
            nested_dropout: 0.5,
            dead_code_threshold: 0.03,
            restart_every: 20,
            layer_norm_eps: 1e-5,
        })
    }

    /// The paper's OpenWebText tokenizer shape (Table 5): `L = 1024`, `K = 32`,
    /// `d = 384`, 4096 codes per level, 0 encoder + 1 decoder block, 6 heads,
    /// EMA 0.99, nested dropout 0.5. The vocabulary is the crate's 259-token
    /// byte vocabulary, not the paper's BPE: this is the paper's *shape* on
    /// nina's tokens. `β = 0.25` is the VQ-VAE default; the paper's value is
    /// not in the extracted tables. Sized for a GPU.
    pub fn paper_owt() -> Result<Self> {
        Ok(Self {
            vocab: crate::tokenizer::VOCAB_SIZE,
            latent_dim: 384,
            codebook_size: 4096,
            schedule: LevelSchedule::geometric(1024, 32)?,
            encoder_blocks: 0,
            decoder_blocks: 1,
            heads: 6,
            mlp_ratio: 4,
            ema_decay: 0.99,
            ema_epsilon: 1e-5,
            commitment_beta: 0.25,
            nested_dropout: 0.5,
            dead_code_threshold: 0.03,
            restart_every: 100,
            layer_norm_eps: 1e-5,
        })
    }

    /// The nina target: byte-level `L = 4096` with `K = 32`, otherwise the
    /// paper's shape. Needs a GPU: one decoder block's attention alone holds
    /// `heads × L²` scores per sample.
    pub fn nina_4k() -> Result<Self> {
        Ok(Self {
            schedule: LevelSchedule::geometric(4096, 32)?,
            ..Self::paper_owt()?
        })
    }

    /// A preset by name (`tiny`, `paper-owt`, `nina-4k`), or a JSON file.
    pub fn resolve(spec: &str) -> Result<Self> {
        let config = match spec {
            "tiny" => Self::tiny()?,
            "paper-owt" | "paper_owt" => Self::paper_owt()?,
            "nina-4k" | "nina_4k" => Self::nina_4k()?,
            path => {
                let text = std::fs::read_to_string(path).with_context(|| {
                    format!(
                        "'{path}' is not a preset (tiny|paper-owt|nina-4k) or a readable JSON file"
                    )
                })?;
                serde_json::from_str(&text)
                    .with_context(|| format!("parse tokenizer config {path}"))?
            }
        };
        config.validate()?;
        Ok(config)
    }

    /// Sequence length `L`.
    pub fn seq_len(&self) -> usize {
        self.schedule.seq_len()
    }

    /// Check every field, naming the one at fault.
    pub fn validate(&self) -> Result<()> {
        self.schedule.validate()?;
        ensure!(
            self.vocab >= 2,
            "vocab must be at least 2, got {}",
            self.vocab
        );
        ensure!(self.latent_dim >= 1, "latent_dim must be at least 1");
        ensure!(self.codebook_size >= 1, "codebook_size must be at least 1");
        ensure!(
            self.heads >= 1 && self.latent_dim.is_multiple_of(self.heads),
            "heads ({}) must be at least 1 and divide latent_dim ({})",
            self.heads,
            self.latent_dim
        );
        ensure!(self.mlp_ratio >= 1, "mlp_ratio must be at least 1");
        ensure!(
            self.ema_decay > 0.0 && self.ema_decay < 1.0,
            "ema_decay must be in (0, 1), got {}",
            self.ema_decay
        );
        ensure!(
            self.ema_epsilon > 0.0 && self.ema_epsilon.is_finite(),
            "ema_epsilon must be positive and finite, got {}",
            self.ema_epsilon
        );
        ensure!(
            self.commitment_beta >= 0.0 && self.commitment_beta.is_finite(),
            "commitment_beta must be non-negative and finite, got {}",
            self.commitment_beta
        );
        ensure!(
            (0.0..=1.0).contains(&self.nested_dropout),
            "nested_dropout must be a probability, got {}",
            self.nested_dropout
        );
        ensure!(
            self.dead_code_threshold >= 0.0 && self.dead_code_threshold.is_finite(),
            "dead_code_threshold must be non-negative and finite, got {}",
            self.dead_code_threshold
        );
        ensure!(self.restart_every >= 1, "restart_every must be at least 1");
        ensure!(
            self.layer_norm_eps > 0.0 && self.layer_norm_eps.is_finite(),
            "layer_norm_eps must be positive and finite"
        );
        Ok(())
    }
}

/// The contiguous windows `[⌊iL/ℓ⌋, ⌊(i+1)L/ℓ⌋)` of a level of length
/// `len` over a sequence of length `seq_len`.
pub fn level_windows(seq_len: usize, len: usize) -> Result<Vec<(usize, usize)>> {
    ensure!(
        len >= 1 && len <= seq_len,
        "a level of length {len} cannot pool a sequence of length {seq_len}"
    );
    Ok((0..len)
        .map(|i| (i * seq_len / len, (i + 1) * seq_len / len))
        .collect())
}

/// Window bookkeeping for one level: which window owns each position, and
/// each window's size.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelGeometry {
    /// `owner[j]` is the window of position `j`, `0..len`.
    pub owner: Vec<usize>,
    /// Positions per window.
    pub sizes: Vec<usize>,
}

impl LevelGeometry {
    pub fn new(seq_len: usize, len: usize) -> Result<Self> {
        let windows = level_windows(seq_len, len)?;
        let mut owner = vec![0usize; seq_len];
        let mut sizes = Vec::with_capacity(len);
        for (i, &(start, end)) in windows.iter().enumerate() {
            for slot in owner.iter_mut().take(end).skip(start) {
                *slot = i;
            }
            sizes.push(end - start);
        }
        Ok(Self { owner, sizes })
    }

    fn owner_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
        let ids: Vec<i64> = self.owner.iter().map(|&i| i as i64).collect();
        Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device)
    }
}

/// `Down`: mean of `x` `[b, L, d]` over each window, `[b, ℓ, d]`.
pub fn pool_windows<B: Backend>(x: Tensor<B, 3>, geometry: &LevelGeometry) -> Tensor<B, 3> {
    let [b, _, d] = x.dims();
    let len = geometry.sizes.len();
    let device = x.device();
    let sums = Tensor::<B, 3>::zeros([b, len, d], &device).select_assign(
        1,
        geometry.owner_tensor::<B>(&device),
        x,
        IndexingUpdateOp::Add,
    );
    let sizes: Vec<f32> = geometry.sizes.iter().map(|&s| s as f32).collect();
    let sizes = Tensor::<B, 1>::from_floats(sizes.as_slice(), &device).reshape([1, len, 1]);
    sums / sizes
}

/// `Up`: `v` `[b, ℓ, d]` to `[b, L, d]`, each position reading its window.
pub fn upsample_windows<B: Backend>(v: Tensor<B, 3>, geometry: &LevelGeometry) -> Tensor<B, 3> {
    let device = v.device();
    v.select(1, geometry.owner_tensor::<B>(&device))
}

/// Index of the nearest row of `codebook` `[C, d]` for each row of `rows`
/// `[N, d]` in squared Euclidean distance.
///
/// `‖u‖²` is dropped from `‖u‖² − 2u·e + ‖e‖²`: it is constant along the row
/// and cannot change the argmin.
pub fn nearest_codes<B: Backend>(rows: Tensor<B, 2>, codebook: Tensor<B, 2>) -> Tensor<B, 1, Int> {
    let [n, _] = rows.dims();
    let [c, _] = codebook.dims();
    let norms = (codebook.clone() * codebook.clone())
        .sum_dim(1)
        .reshape([1, c]);
    let cross = rows.matmul(codebook.transpose());
    (norms - cross.mul_scalar(2.0)).argmin(1).reshape([n])
}

/// The straight-through estimator `ẑ + (z − sg[z])`: the value of `ẑ`, the
/// gradient of `z`.
pub fn straight_through<B: Backend>(z: Tensor<B, 3>, zhat: Tensor<B, 3>) -> Tensor<B, 3> {
    zhat + (z.clone() - z.detach())
}

/// One EMA step of a level's state: `N ← γN + (1−γ)n`, `m ← γm + (1−γ)s`.
pub fn ema_step<B: Backend>(
    cluster_size: Tensor<B, 1>,
    embed_sum: Tensor<B, 2>,
    counts: Tensor<B, 1>,
    sums: Tensor<B, 2>,
    decay: f64,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    (
        cluster_size.mul_scalar(decay) + counts.mul_scalar(1.0 - decay),
        embed_sum.mul_scalar(decay) + sums.mul_scalar(1.0 - decay),
    )
}

/// The Laplace-smoothed sizes `Ñ_j = (N_j + ε) / (n + Cε) · n`, `n = Σ N`.
pub fn smoothed_sizes<B: Backend>(cluster_size: Tensor<B, 1>, epsilon: f64) -> Tensor<B, 1> {
    let [c] = cluster_size.dims();
    let total = cluster_size.clone().sum();
    (cluster_size.add_scalar(epsilon) / total.clone().add_scalar(c as f64 * epsilon)) * total
}

/// The codebook `e_j = m_j / Ñ_j` an EMA state stands for.
pub fn codebook_from_state<B: Backend>(
    cluster_size: Tensor<B, 1>,
    embed_sum: Tensor<B, 2>,
    epsilon: f64,
) -> Tensor<B, 2> {
    let [c] = cluster_size.dims();
    embed_sum / smoothed_sizes(cluster_size, epsilon).reshape([c, 1])
}

/// A tensor that is never trained by gradient, as a parameter so it is
/// checkpointed with the model.
fn frozen<B: Backend, const D: usize>(id: ParamId, value: Tensor<B, D>) -> Param<Tensor<B, D>> {
    Param::initialized(id, value.detach().set_require_grad(false))
}

/// One bidirectional pre-norm transformer block: multi-head self-attention
/// with no mask, then an exact-GELU MLP, both residual.
#[derive(Module, Debug)]
pub struct NcmBlock<B: Backend> {
    norm_attn: LayerNorm<B>,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    attn_out: Linear<B>,
    norm_mlp: LayerNorm<B>,
    fc_in: Linear<B>,
    fc_out: Linear<B>,
    heads: usize,
}

impl<B: Backend> NcmBlock<B> {
    /// Build a block of width `width`. Fallible: `heads` must divide it.
    pub fn new(
        width: usize,
        heads: usize,
        mlp_ratio: usize,
        layer_norm_eps: f64,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(
            heads >= 1 && width.is_multiple_of(heads),
            "{heads} heads do not divide width {width}"
        );
        let linear = |i: usize, o: usize| LinearConfig::new(i, o).with_bias(true).init(device);
        Ok(Self {
            norm_attn: LayerNormConfig::new(width)
                .with_epsilon(layer_norm_eps)
                .init(device),
            query: linear(width, width),
            key: linear(width, width),
            value: linear(width, width),
            attn_out: linear(width, width),
            norm_mlp: LayerNormConfig::new(width)
                .with_epsilon(layer_norm_eps)
                .init(device),
            fc_in: linear(width, width * mlp_ratio),
            fc_out: linear(width * mlp_ratio, width),
            heads,
        })
    }

    /// `[b, n, width] → [b, n, width]`; every position attends to every other.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, width] = x.dims();
        let head_dim = width / self.heads;
        let normed = self.norm_attn.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([b, n, self.heads, head_dim]).swap_dims(1, 2);
        let q = split(self.query.forward(normed.clone()));
        let k = split(self.key.forward(normed.clone()));
        let v = split(self.value.forward(normed));
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (head_dim as f64).sqrt());
        let context = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, n, width]);
        let x = x + self.attn_out.forward(context);
        let normed = self.norm_mlp.forward(x.clone());
        x + self.fc_out.forward(exact_gelu(self.fc_in.forward(normed)))
    }
}

/// One level's quantizer state and its lift `R_k`.
#[derive(Module, Debug)]
pub struct NcmLevel<B: Backend> {
    /// EMA cluster sizes `N`, `[C]`. No gradient.
    cluster_size: Param<Tensor<B, 1>>,
    /// EMA embedding sums `m`, `[C, d]`. No gradient.
    embed_sum: Param<Tensor<B, 2>>,
    /// The linear part of `R_k`, zero-initialised.
    lift: Linear<B>,
}

/// Per-level assignment statistics of one batch, for the EMA update.
#[derive(Debug, Clone)]
pub struct LevelStats<B: Backend> {
    /// `n_j`: vectors assigned to code `j`, `[C]`.
    pub counts: Tensor<B, 1>,
    /// `s_j`: their sum, `[C, d]`.
    pub sums: Tensor<B, 2>,
}

impl<B: Backend> LevelStats<B> {
    /// Fold another micro-batch's statistics in (gradient accumulation).
    pub fn merge(self, other: Self) -> Self {
        Self {
            counts: self.counts + other.counts,
            sums: self.sums + other.sums,
        }
    }
}

/// What residual quantization of a batch produced.
#[derive(Debug, Clone)]
pub struct Quantized<B: Backend> {
    /// `r_k`, `[b, ℓ_k]` per level.
    pub codes: Vec<Tensor<B, 2, Int>>,
    /// `u_k = Down_k(z^(k-1))`, `[b, ℓ_k, d]`, detached.
    pub pooled: Vec<Tensor<B, 3>>,
    /// `z^(K)`, carried by running subtraction, detached.
    pub residual: Tensor<B, 3>,
    /// `ẑ^(K)`, detached.
    pub zhat: Tensor<B, 3>,
    /// The reconstruction the loss reads: `ẑ^(c_b)` per sample when cutoffs
    /// were given, `ẑ^(K)` otherwise. Carries gradient into the `R_k` maps.
    pub selected: Tensor<B, 3>,
    /// Per-level EMA statistics, when requested.
    pub stats: Vec<LevelStats<B>>,
}

/// The multiscale residual-VQ tokenizer.
#[derive(Module, Debug)]
pub struct NcmTokenizer<B: Backend> {
    token_embedding: Embedding<B>,
    encoder_positions: Param<Tensor<B, 3>>,
    encoder: Vec<NcmBlock<B>>,
    levels: Vec<NcmLevel<B>>,
    decoder_positions: Param<Tensor<B, 3>>,
    decoder: Vec<NcmBlock<B>>,
    final_norm: LayerNorm<B>,
    head: Linear<B>,
    #[module(skip)]
    config: NcmTokenizerConfig,
}

impl<B: Backend> NcmTokenizer<B> {
    /// Build a fresh tokenizer. Codebooks start as `N(0, 1)` rows with unit
    /// EMA counts; [`Self::init_codebooks_from_data`] replaces them before
    /// training.
    pub fn new(config: &NcmTokenizerConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        let d = config.latent_dim;
        let l = config.seq_len();
        let c = config.codebook_size;
        let block = || {
            NcmBlock::new(
                d,
                config.heads,
                config.mlp_ratio,
                config.layer_norm_eps,
                device,
            )
        };
        let positions = || {
            Param::from_tensor(Tensor::<B, 3>::random(
                [1, l, d],
                Distribution::Normal(0.0, 0.02),
                device,
            ))
        };
        let levels = (0..config.schedule.levels())
            .map(|_| NcmLevel {
                cluster_size: frozen(ParamId::new(), Tensor::<B, 1>::ones([c], device)),
                embed_sum: frozen(
                    ParamId::new(),
                    Tensor::<B, 2>::random([c, d], Distribution::Normal(0.0, 1.0), device),
                ),
                lift: LinearConfig::new(d, d)
                    .with_bias(true)
                    .with_initializer(Initializer::Zeros)
                    .init(device),
            })
            .collect();
        Ok(Self {
            token_embedding: EmbeddingConfig::new(config.vocab, d).init(device),
            encoder_positions: positions(),
            encoder: (0..config.encoder_blocks)
                .map(|_| block())
                .collect::<Result<_>>()?,
            levels,
            decoder_positions: positions(),
            decoder: (0..config.decoder_blocks)
                .map(|_| block())
                .collect::<Result<_>>()?,
            final_norm: LayerNormConfig::new(d)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            head: LinearConfig::new(d, config.vocab)
                .with_bias(true)
                .init(device),
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &NcmTokenizerConfig {
        &self.config
    }

    fn level(&self, level: usize) -> Result<&NcmLevel<B>> {
        self.levels.get(level).with_context(|| {
            format!(
                "level {level} does not exist; this tokenizer has {} levels",
                self.levels.len()
            )
        })
    }

    /// Window bookkeeping of 0-based `level`.
    pub fn geometry(&self, level: usize) -> Result<LevelGeometry> {
        let len = self
            .config
            .schedule
            .len_of(level)
            .with_context(|| format!("level {level} is not in the schedule"))?;
        LevelGeometry::new(self.config.seq_len(), len)
    }

    /// The EMA state `(N, m)` of 0-based `level`.
    pub fn level_state(&self, level: usize) -> Result<(Tensor<B, 1>, Tensor<B, 2>)> {
        let state = self.level(level)?;
        Ok((state.cluster_size.val(), state.embed_sum.val()))
    }

    /// The codebook of 0-based `level`, `[C, d]`: `m / Ñ`, without gradient.
    pub fn codebook(&self, level: usize) -> Result<Tensor<B, 2>> {
        let (n, m) = self.level_state(level)?;
        Ok(codebook_from_state(n, m, self.config.ema_epsilon).detach())
    }

    /// `R_k`: code vectors `[b, ℓ_k, d]` to their contribution `[b, L, d]`.
    pub fn lift(&self, level: usize, vectors: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [_, len, d] = vectors.dims();
        let geometry = self.geometry(level)?;
        ensure!(
            len == geometry.sizes.len() && d == self.config.latent_dim,
            "level {level} lifts [_, {}, {}] vectors, got [_, {len}, {d}]",
            geometry.sizes.len(),
            self.config.latent_dim
        );
        let coarse = vectors.clone() + self.level(level)?.lift.forward(vectors);
        Ok(upsample_windows(coarse, &geometry))
    }

    /// The code vectors `e_k(r_k)` of codes `[b, ℓ_k]`, `[b, ℓ_k, d]`.
    pub fn code_vectors(&self, level: usize, codes: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>> {
        let [b, len] = codes.dims();
        let codebook = self.codebook(level)?;
        Ok(codebook
            .select(0, codes.reshape([b * len]))
            .reshape([b, len, self.config.latent_dim]))
    }

    /// `R_k(e_k(r_k))` for codes `[b, ℓ_k]`.
    pub fn level_contribution(
        &self,
        level: usize,
        codes: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>> {
        let vectors = self.code_vectors(level, codes)?;
        self.lift(level, vectors)
    }

    fn check_tokens(&self, tokens: &Tensor<B, 2, Int>) -> Result<usize> {
        let [b, l] = tokens.dims();
        ensure!(
            l == self.config.seq_len(),
            "the tokenizer reads windows of {} tokens, got {l}",
            self.config.seq_len()
        );
        ensure!(b >= 1, "an empty batch has nothing to encode");
        Ok(b)
    }

    /// The encoder `E`: tokens `[b, L]` to latents `z` `[b, L, d]`.
    pub fn latents(&self, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>> {
        self.check_tokens(&tokens)?;
        let mut z = self.token_embedding.forward(tokens) + self.encoder_positions.val();
        for block in &self.encoder {
            z = block.forward(z);
        }
        Ok(z)
    }

    /// The decoder `ψ`: `ẑ` `[b, L, d]` to token logits `[b, L, vocab]`.
    pub fn decode(&self, zhat: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [_, l, d] = zhat.dims();
        ensure!(
            l == self.config.seq_len() && d == self.config.latent_dim,
            "the decoder reads [_, {}, {}], got [_, {l}, {d}]",
            self.config.seq_len(),
            self.config.latent_dim
        );
        let mut h = zhat + self.decoder_positions.val();
        for block in &self.decoder {
            h = block.forward(h);
        }
        Ok(self.head.forward(self.final_norm.forward(h)))
    }

    /// Residual quantization of latents `z` `[b, L, d]`.
    ///
    /// `cutoffs`, one per sample in `1..=K`, select `ẑ^(c_b)` as
    /// [`Quantized::selected`]; `None` selects `ẑ^(K)` for every sample.
    /// `with_stats` also returns the per-level EMA statistics.
    pub fn quantize(
        &self,
        z: Tensor<B, 3>,
        cutoffs: Option<&[usize]>,
        with_stats: bool,
    ) -> Result<Quantized<B>> {
        let [b, l, d] = z.dims();
        ensure!(
            l == self.config.seq_len() && d == self.config.latent_dim,
            "quantize reads latents [_, {}, {}], got [_, {l}, {d}]",
            self.config.seq_len(),
            self.config.latent_dim
        );
        let levels = self.levels.len();
        let device = z.device();
        let masks = match cutoffs {
            Some(cutoffs) => {
                ensure!(
                    cutoffs.len() == b,
                    "{} cutoffs for a batch of {b}",
                    cutoffs.len()
                );
                ensure!(
                    cutoffs.iter().all(|&c| (1..=levels).contains(&c)),
                    "every cutoff must be in 1..={levels}, got {cutoffs:?}"
                );
                let masks = (0..levels)
                    .map(|k| {
                        let keep: Vec<f32> = cutoffs
                            .iter()
                            .map(|&c| if k < c { 1.0 } else { 0.0 })
                            .collect();
                        Tensor::<B, 1>::from_floats(keep.as_slice(), &device).reshape([b, 1, 1])
                    })
                    .collect::<Vec<_>>();
                Some(masks)
            }
            None => None,
        };
        let mut residual = z.detach();
        let mut zhat = Tensor::<B, 3>::zeros([b, l, d], &device);
        let mut selected = Tensor::<B, 3>::zeros([b, l, d], &device);
        let mut codes = Vec::with_capacity(levels);
        let mut pooled = Vec::with_capacity(levels);
        let mut stats = Vec::with_capacity(if with_stats { levels } else { 0 });
        for k in 0..levels {
            let geometry = self.geometry(k)?;
            let len = geometry.sizes.len();
            let u = pool_windows(residual.clone(), &geometry).detach();
            let rows = u.clone().reshape([b * len, d]);
            let codebook = self.codebook(k)?;
            let flat = nearest_codes(rows.clone(), codebook.clone());
            if with_stats {
                let c = self.config.codebook_size;
                let counts = Tensor::<B, 1>::zeros([c], &device).select_assign(
                    0,
                    flat.clone(),
                    Tensor::<B, 1>::ones([b * len], &device),
                    IndexingUpdateOp::Add,
                );
                let sums = Tensor::<B, 2>::zeros([c, d], &device).select_assign(
                    0,
                    flat.clone(),
                    rows,
                    IndexingUpdateOp::Add,
                );
                stats.push(LevelStats { counts, sums });
            }
            let level_codes = flat.reshape([b, len]);
            let contribution = self.level_contribution(k, level_codes.clone())?;
            let value = contribution.clone().detach();
            residual = residual - value.clone();
            zhat = zhat + value;
            selected = match &masks {
                Some(masks) => {
                    let mask = masks
                        .get(k)
                        .with_context(|| format!("no cutoff mask for level {k}"))?;
                    selected + contribution * mask.clone()
                }
                None => selected + contribution,
            };
            codes.push(level_codes);
            pooled.push(u);
        }
        Ok(Quantized {
            codes,
            pooled,
            residual,
            zhat,
            selected,
            stats,
        })
    }

    /// Fold one step's statistics into every level's EMA state.
    pub fn apply_codebook_ema(self, stats: &[LevelStats<B>]) -> Result<Self> {
        ensure!(
            stats.len() == self.levels.len(),
            "{} levels of statistics for {} levels",
            stats.len(),
            self.levels.len()
        );
        let decay = self.config.ema_decay;
        let Self {
            token_embedding,
            encoder_positions,
            encoder,
            levels,
            decoder_positions,
            decoder,
            final_norm,
            head,
            config,
        } = self;
        let levels = levels
            .into_iter()
            .zip(stats)
            .map(|(level, step)| {
                let (n, m) = ema_step(
                    level.cluster_size.val(),
                    level.embed_sum.val(),
                    step.counts.clone(),
                    step.sums.clone(),
                    decay,
                );
                NcmLevel {
                    cluster_size: frozen(level.cluster_size.id, n),
                    embed_sum: frozen(level.embed_sum.id, m),
                    lift: level.lift,
                }
            })
            .collect();
        Ok(Self {
            token_embedding,
            encoder_positions,
            encoder,
            levels,
            decoder_positions,
            decoder,
            final_norm,
            head,
            config,
        })
    }

    /// Re-seed the codes of `level` listed in `targets` from rows of `rows`
    /// `[N, d]` (`picks[i]` is the row for `targets[i]`), plus noise of
    /// relative size `noise`, and give each the count `count`.
    ///
    /// `m_j` is set to `row · Ñ_j` under the *new* counts, so the
    /// materialised code is the chosen row (plus noise) up to rounding.
    fn reseed(
        self,
        level: usize,
        rows: Tensor<B, 2>,
        targets: &[usize],
        picks: &[usize],
        count: f32,
        noise: f64,
    ) -> Result<Self> {
        ensure!(targets.len() == picks.len(), "one pick per reseeded code");
        if targets.is_empty() {
            return Ok(self);
        }
        let [available, d] = rows.dims();
        ensure!(available >= 1, "no rows to reseed level {level} from");
        ensure!(
            picks.iter().all(|&p| p < available),
            "a pick is past the {available} available rows"
        );
        let c = self.config.codebook_size;
        ensure!(
            targets.iter().all(|&t| t < c),
            "a reseeded code is past the codebook of {c}"
        );
        let epsilon = self.config.ema_epsilon;
        let device = rows.device();
        let (n, m) = self.level_state(level)?;
        let mut sizes: Vec<f32> = n.into_data().iter::<f32>().collect();
        let mut mask = vec![0.0f32; c];
        for &t in targets {
            if let Some(slot) = sizes.get_mut(t) {
                *slot = count;
            }
            if let Some(slot) = mask.get_mut(t) {
                *slot = 1.0;
            }
        }
        let total: f64 = sizes.iter().map(|&s| f64::from(s)).sum();
        let smoothed: Vec<f32> = targets
            .iter()
            .map(|&t| {
                let size = sizes.get(t).copied().map_or(0.0, f64::from);
                ((size + epsilon) / (total + c as f64 * epsilon) * total) as f32
            })
            .collect();
        let pick_ids: Vec<i64> = picks.iter().map(|&p| p as i64).collect();
        let target_ids: Vec<i64> = targets.iter().map(|&t| t as i64).collect();
        let chosen = rows.select(
            0,
            Tensor::<B, 1, Int>::from_ints(pick_ids.as_slice(), &device),
        );
        let rms = chosen
            .clone()
            .powf_scalar(2.0)
            .mean()
            .sqrt()
            .into_data()
            .iter::<f64>()
            .next()
            .unwrap_or(0.0);
        let jitter = Tensor::<B, 2>::random(
            [targets.len(), d],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        )
        .mul_scalar(noise * rms);
        let scale =
            Tensor::<B, 1>::from_floats(smoothed.as_slice(), &device).reshape([targets.len(), 1]);
        let replacement = Tensor::<B, 2>::zeros([c, d], &device).select_assign(
            0,
            Tensor::<B, 1, Int>::from_ints(target_ids.as_slice(), &device),
            (chosen + jitter) * scale,
            IndexingUpdateOp::Add,
        );
        let keep = Tensor::<B, 1>::from_floats(mask.as_slice(), &device)
            .neg()
            .add_scalar(1.0)
            .reshape([c, 1]);
        let new_m = m * keep + replacement;
        let new_n = Tensor::<B, 1>::from_floats(sizes.as_slice(), &device);
        let mut this = self;
        let slot = this
            .levels
            .get_mut(level)
            .with_context(|| format!("level {level} vanished"))?;
        slot.cluster_size = frozen(slot.cluster_size.id, new_n);
        slot.embed_sum = frozen(slot.embed_sum.id, new_m);
        Ok(this)
    }

    /// Seed every codebook from data, level by level: level `k` draws its
    /// codes from the pooled residual `u_k` the already-seeded levels
    /// `1..k` leave on `tokens`. Counts are reset to 1.
    pub fn init_codebooks_from_data<R: Rng>(
        self,
        tokens: Tensor<B, 2, Int>,
        rng: &mut R,
    ) -> Result<Self> {
        let z = self.latents(tokens)?.detach();
        let [b, _, d] = z.dims();
        let c = self.config.codebook_size;
        let mut residual = z;
        let mut this = self;
        for k in 0..this.levels.len() {
            let geometry = this.geometry(k)?;
            let len = geometry.sizes.len();
            let rows = pool_windows(residual.clone(), &geometry).reshape([b * len, d]);
            let picks: Vec<usize> = (0..c).map(|_| rng.random_range(0..b * len)).collect();
            let targets: Vec<usize> = (0..c).collect();
            this = this.reseed(k, rows.clone(), &targets, &picks, 1.0, 1e-3)?;
            let codes = nearest_codes(rows, this.codebook(k)?).reshape([b, len]);
            residual = residual - this.level_contribution(k, codes)?.detach();
        }
        Ok(this)
    }

    /// Restart the dead codes of every level from the pooled vectors of the
    /// current batch. Returns the number restarted per level.
    pub fn restart_dead_codes<R: Rng>(
        self,
        pooled: &[Tensor<B, 3>],
        rng: &mut R,
    ) -> Result<(Self, Vec<usize>)> {
        ensure!(
            pooled.len() == self.levels.len(),
            "{} pooled levels for {} levels",
            pooled.len(),
            self.levels.len()
        );
        let threshold = self.config.dead_code_threshold;
        let c = self.config.codebook_size;
        let mut restarted = Vec::with_capacity(pooled.len());
        let mut this = self;
        for (k, u) in pooled.iter().enumerate() {
            if threshold <= 0.0 {
                restarted.push(0);
                continue;
            }
            let (n, _) = this.level_state(k)?;
            let sizes: Vec<f32> = n.into_data().iter::<f32>().collect();
            let total: f64 = sizes.iter().map(|&s| f64::from(s)).sum();
            let share = total / c as f64;
            let mut dead = Vec::new();
            for (j, &size) in sizes.iter().enumerate() {
                if f64::from(size) < threshold * share {
                    dead.push(j);
                }
            }
            restarted.push(dead.len());
            if dead.is_empty() {
                continue;
            }
            let [b, len, d] = u.dims();
            let rows = u.clone().reshape([b * len, d]);
            let picks: Vec<usize> = dead.iter().map(|_| rng.random_range(0..b * len)).collect();
            this = this.reseed(k, rows, &dead, &picks, share as f32, 1e-3)?;
        }
        Ok((this, restarted))
    }

    /// Mean reconstruction cross-entropy of `logits` `[b, L, V]` against
    /// `tokens` `[b, L]`, as a `[1]` tensor.
    pub fn cross_entropy(logits: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [b, l] = tokens.dims();
        log_softmax(logits, 2)
            .gather(2, tokens.reshape([b, l, 1]))
            .mean()
            .neg()
    }

    /// Replace every `R_k` linear map with random weights of standard
    /// deviation `std`, so certificates exercise a non-trivial lift (a fresh
    /// tokenizer's maps are zero).
    pub(crate) fn randomize_lifts(self, std: f64, device: &B::Device) -> Self {
        let d = self.config.latent_dim;
        let mut this = self;
        for level in &mut this.levels {
            level.lift = LinearConfig::new(d, d)
                .with_bias(true)
                .with_initializer(Initializer::Normal { mean: 0.0, std })
                .init(device);
        }
        this
    }

    /// The `R_k` linear map of `level` as host `(W [d_in × d_out], b [d])`.
    pub(crate) fn lift_parameters(&self, level: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let lift = &self.level(level)?.lift;
        let weight: Vec<f32> = lift.weight.val().into_data().iter::<f32>().collect();
        let bias: Vec<f32> = match &lift.bias {
            Some(bias) => bias.val().into_data().iter::<f32>().collect(),
            None => vec![0.0; self.config.latent_dim],
        };
        Ok((weight, bias))
    }
}

impl<B: Backend> ConceptCodec<B> for NcmTokenizer<B> {
    fn schedule(&self) -> &LevelSchedule {
        &self.config.schedule
    }

    fn codebook_size(&self) -> usize {
        self.config.codebook_size
    }

    fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab
    }

    fn encode(&self, tokens: Tensor<B, 2, Int>) -> Result<LevelCodes<B>> {
        let z = self.latents(tokens)?;
        Ok(LevelCodes {
            codes: self.quantize(z, None, false)?.codes,
        })
    }

    fn reconstruct(&self, codes: &[Tensor<B, 2, Int>], batch: usize) -> Result<Tensor<B, 3>> {
        ensure!(
            codes.len() <= self.levels.len(),
            "{} levels of codes for a {}-level tokenizer",
            codes.len(),
            self.levels.len()
        );
        let device = self.head.weight.val().device();
        let mut zhat = Tensor::<B, 3>::zeros(
            [batch, self.config.seq_len(), self.config.latent_dim],
            &device,
        );
        for (k, level_codes) in codes.iter().enumerate() {
            let [b, len] = level_codes.dims();
            let expected = self.config.schedule.len_of(k).unwrap_or(0);
            ensure!(
                b == batch && len == expected,
                "level {k} codes must be [{batch}, {expected}], got [{b}, {len}]"
            );
            zhat = zhat + self.level_contribution(k, level_codes.clone())?;
        }
        Ok(zhat)
    }

    fn decode_logits(&self, zhat: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        self.decode(zhat)
    }
}

impl<B: Backend> SoftConceptCodec<B> for NcmTokenizer<B> {
    fn level_codebook(&self, level: usize) -> Result<Tensor<B, 2>> {
        self.codebook(level)
    }
}

/// Codebook usage of one level's assignments: distinct codes and the
/// perplexity `exp(H)` of the empirical code distribution.
pub fn code_usage(histogram: &[u64]) -> (usize, f64) {
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return (0, 0.0);
    }
    let used = histogram.iter().filter(|&&h| h > 0).count();
    let entropy: f64 = histogram
        .iter()
        .filter(|&&h| h > 0)
        .map(|&h| {
            let p = h as f64 / total as f64;
            -p * p.ln()
        })
        .sum();
    (used, entropy.exp())
}

// ------------------------------------------------------------ certificates --

type CertBackend = burn::backend::NdArray<f32>;
type CertAutodiff = burn::backend::Autodiff<CertBackend>;

fn ncm_cert(
    name: &'static str,
    theorem: &'static str,
    residual: f64,
    tolerance: f64,
) -> crate::verify::Certificate {
    crate::verify::Certificate {
        group: "ncm",
        name,
        theorem,
        residual,
        tolerance,
    }
}

/// `a.max(b)` that keeps a NaN instead of dropping it.
fn worst(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn host<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().iter::<f64>().collect()
}

fn host_ints<B: Backend, const D: usize>(t: Tensor<B, D, Int>) -> Vec<usize> {
    t.into_data()
        .iter::<i64>()
        .map(|v| usize::try_from(v).unwrap_or(usize::MAX))
        .collect()
}

/// The certificate micro-configuration: `L = 12` over lengths `2, 5, 12`
/// (5 does not divide 12, so uneven windows are exercised), one encoder and
/// one decoder block, 16 codes of width 8.
pub(crate) fn micro_config() -> Result<NcmTokenizerConfig> {
    Ok(NcmTokenizerConfig {
        vocab: crate::tokenizer::VOCAB_SIZE,
        latent_dim: 8,
        codebook_size: 16,
        schedule: LevelSchedule::new(vec![2, 5, 12])?,
        encoder_blocks: 1,
        decoder_blocks: 1,
        heads: 2,
        mlp_ratio: 2,
        ema_decay: 0.9,
        ema_epsilon: 1e-5,
        commitment_beta: 0.25,
        nested_dropout: 0.5,
        dead_code_threshold: 0.03,
        restart_every: 1,
        layer_norm_eps: 1e-5,
    })
}

/// A seeded micro tokenizer with data-seeded codebooks and random lifts,
/// and the token batch it was seeded on.
pub(crate) fn micro_tokenizer<B: Backend>(
    seed: u64,
    batch: usize,
    device: &B::Device,
) -> Result<(NcmTokenizer<B>, Tensor<B, 2, Int>)> {
    use rand::SeedableRng;
    let config = micro_config()?;
    B::seed(device, seed);
    let model = NcmTokenizer::<B>::new(&config, device)?.randomize_lifts(0.2, device);
    crate::tensor_ext::force_initialization(&model);
    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(seed);
    let l = config.seq_len();
    let ids: Vec<i64> = (0..batch * l)
        .map(|_| i64::from(rng.random_range(0u16..259)))
        .collect();
    let tokens = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([batch, l]);
    let model = model.init_codebooks_from_data(tokens.clone(), &mut rng)?;
    Ok((model, tokens))
}

/// Largest normalised distance gap `(d(u, e_chosen) − d(u, e_best)) / (1 +
/// d(u, e_best))` over the rows, by a brute-force f64 scan of the codebook.
pub fn nearest_code_gap(rows: &[f64], codebook: &[f64], chosen: &[usize], d: usize) -> Result<f64> {
    ensure!(
        d >= 1 && rows.len().is_multiple_of(d) && codebook.len().is_multiple_of(d),
        "ragged rows"
    );
    let n = rows.len() / d;
    let c = codebook.len() / d;
    ensure!(chosen.len() == n, "{} choices for {n} rows", chosen.len());
    let distance = |row: &[f64], code: usize| -> f64 {
        row.iter()
            .zip(codebook.iter().skip(code * d).take(d))
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    };
    let mut gap = 0.0f64;
    for (i, &pick) in chosen.iter().enumerate() {
        ensure!(pick < c, "row {i} chose code {pick} of {c}");
        let row = rows
            .get(i * d..(i + 1) * d)
            .with_context(|| format!("row {i} is out of range"))?;
        let mut best = f64::INFINITY;
        for code in 0..c {
            best = best.min(distance(row, code));
        }
        gap = worst(gap, (distance(row, pick) - best) / (1.0 + best));
    }
    Ok(gap)
}

/// Host reference for `Down_k` over `[b, L, d]` data.
fn host_pool(x: &[f64], b: usize, l: usize, d: usize, windows: &[(usize, usize)]) -> Vec<f64> {
    let mut out = Vec::with_capacity(b * windows.len() * d);
    for s in 0..b {
        for &(start, end) in windows {
            for col in 0..d {
                let sum: f64 = (start..end)
                    .map(|j| x.get((s * l + j) * d + col).copied().unwrap_or(f64::NAN))
                    .sum();
                out.push(sum / (end - start) as f64);
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .fold(0.0f64, |acc, (x, y)| worst(acc, (x - y).abs()))
}

fn schedule_and_weights_residual() -> f64 {
    let mut residual = 0.0f64;
    for (l, k) in [
        (1024, 32),
        (128, 12),
        (48, 8),
        (64, 11),
        (4096, 32),
        (64, 6),
        (12, 3),
    ] {
        match LevelSchedule::geometric(l, k) {
            Ok(s) => {
                let ok = s.levels() == k
                    && s.seq_len() == l
                    && s.lengths().first().is_some_and(|&f| f >= 1)
                    && s.lengths().windows(2).all(|w| matches!(w, [a, b] if a < b));
                if !ok {
                    residual = worst(residual, 1.0);
                }
                let (Ok(w0), Ok(w1)) = (s.loss_weights(0.0), s.loss_weights(1.0)) else {
                    return f64::INFINITY;
                };
                let total: f64 = s.lengths().iter().map(|&x| x as f64).sum();
                for ((a, b), &len) in w0.iter().zip(&w1).zip(s.lengths()) {
                    residual = worst(residual, (a - 1.0 / total).abs());
                    residual = worst(residual, (b * len as f64 - 1.0 / k as f64).abs());
                }
                let mass: f64 = w0
                    .iter()
                    .zip(s.lengths())
                    .map(|(w, &len)| w * len as f64)
                    .sum();
                residual = worst(residual, (mass - 1.0).abs());
            }
            Err(_) => return f64::INFINITY,
        }
    }
    // The refusals are part of the contract: a schedule that cannot be
    // strictly increasing, or a weight exponent outside [0, 1], is an error.
    let refused = [
        LevelSchedule::geometric(4, 5).is_err(),
        LevelSchedule::new(vec![2, 2, 4]).is_err(),
        LevelSchedule::new(vec![0, 4]).is_err(),
        LevelSchedule::new(vec![1, 4]).is_ok_and(|s| s.loss_weights(1.5).is_err()),
    ];
    if refused.iter().any(|ok| !ok) {
        residual = worst(residual, 1.0);
    }
    residual
}

fn windows_residual() -> Result<f64> {
    let device = Default::default();
    let mut residual = 0.0f64;
    for (l, len) in [
        (12, 5),
        (12, 2),
        (12, 12),
        (1024, 7),
        (64, 6),
        (7, 3),
        (5, 1),
    ] {
        let windows = level_windows(l, len)?;
        let mut cursor = 0usize;
        for &(start, end) in &windows {
            let size = end.saturating_sub(start);
            if start != cursor || size == 0 || size < l / len || size > l.div_ceil(len) {
                residual = worst(residual, 1.0);
            }
            cursor = end;
        }
        if cursor != l || windows.len() != len {
            residual = worst(residual, 1.0);
        }
        // Down on random data against the host mean, and Down ∘ Up = id.
        let geometry = LevelGeometry::new(l, len)?;
        let d = 3usize;
        let x =
            Tensor::<CertBackend, 3>::random([2, l, d], Distribution::Normal(0.0, 1.0), &device);
        let pooled = pool_windows(x.clone(), &geometry);
        let reference = host_pool(&host(x), 2, l, d, &windows);
        residual = worst(residual, max_abs_diff(&host(pooled.clone()), &reference));
        let round_trip = pool_windows(upsample_windows(pooled.clone(), &geometry), &geometry);
        residual = worst(residual, max_abs_diff(&host(round_trip), &host(pooled)));
    }
    Ok(residual)
}

fn nearest_code_residual() -> Result<f64> {
    let device = Default::default();
    CertBackend::seed(&device, 11);
    let (n, c, d) = (64usize, 32usize, 8usize);
    let rows = Tensor::<CertBackend, 2>::random([n, d], Distribution::Normal(0.0, 1.0), &device);
    let codebook =
        Tensor::<CertBackend, 2>::random([c, d], Distribution::Normal(0.0, 1.0), &device);
    let chosen = host_ints(nearest_codes(rows.clone(), codebook.clone()));
    nearest_code_gap(&host(rows), &host(codebook), &chosen, d)
}

/// `encode`'s level-`k` code is the argmin, against exactly the rows
/// `level_codebook(k)` returns, of the window means of `z − ẑ^(k-1)`, where
/// `ẑ^(k-1) = reconstruct(codes[..k])` — all recomputed on the host.
fn encode_matches_codebook_residual() -> Result<f64> {
    let device = Default::default();
    let (model, tokens) = micro_tokenizer::<CertBackend>(23, 4, &device)?;
    let [b, l] = tokens.dims();
    let d = model.config().latent_dim;
    let codes = model.encode(tokens.clone())?.codes;
    let z = host(model.latents(tokens)?);
    let mut gap = 0.0f64;
    for k in 0..codes.len() {
        let prefix = model.reconstruct(codes.get(..k).unwrap_or(&[]), b)?;
        let residual: Vec<f64> = z.iter().zip(host(prefix)).map(|(a, p)| a - p).collect();
        let windows = level_windows(l, model.config().schedule.len_of(k).unwrap_or(0))?;
        let u = host_pool(&residual, b, l, d, &windows);
        let rows = host(model.level_codebook(k)?);
        let chosen = host_ints(
            codes
                .get(k)
                .with_context(|| format!("no codes at level {k}"))?
                .clone(),
        );
        gap = worst(gap, nearest_code_gap(&u, &rows, &chosen, d)?);
    }
    Ok(gap)
}

/// Host reference for `Σ_{j<k} R_j(e_j(r_j))`, from the codebook rows, the
/// lift weights and the window map — no tensor op of the tokenizer.
fn host_reconstruction(
    model: &NcmTokenizer<CertBackend>,
    codes: &[Vec<usize>],
    b: usize,
) -> Result<Vec<f64>> {
    let l = model.config().seq_len();
    let d = model.config().latent_dim;
    let mut out = vec![0.0f64; b * l * d];
    for (k, level_codes) in codes.iter().enumerate() {
        let rows = host(model.codebook(k)?);
        let (weight, bias) = model.lift_parameters(k)?;
        let geometry = model.geometry(k)?;
        let len = geometry.sizes.len();
        for s in 0..b {
            for (j, &window) in geometry.owner.iter().enumerate() {
                let code = *level_codes
                    .get(s * len + window)
                    .with_context(|| format!("missing code for sample {s} window {window}"))?;
                let v = rows
                    .get(code * d..(code + 1) * d)
                    .with_context(|| format!("code {code} is past the codebook"))?;
                for o in 0..d {
                    let mut y = v.get(o).copied().unwrap_or(f64::NAN)
                        + f64::from(bias.get(o).copied().unwrap_or(f32::NAN));
                    for (i, vi) in v.iter().enumerate() {
                        y += vi * f64::from(weight.get(i * d + o).copied().unwrap_or(f32::NAN));
                    }
                    if let Some(slot) = out.get_mut((s * l + j) * d + o) {
                        *slot += y;
                    }
                }
            }
        }
    }
    Ok(out)
}

fn telescoping_residual() -> Result<f64> {
    let device = Default::default();
    let (model, tokens) = micro_tokenizer::<CertBackend>(31, 3, &device)?;
    let [b, _] = tokens.dims();
    let z = model.latents(tokens)?;
    let q = model.quantize(z.clone(), None, false)?;
    let z_host = host(z);
    let scale = 1.0 + z_host.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let sum: Vec<f64> = host(q.zhat.clone())
        .iter()
        .zip(host(q.residual.clone()))
        .map(|(a, r)| a + r)
        .collect();
    let mut residual = max_abs_diff(&sum, &z_host) / scale;
    let host_codes: Vec<Vec<usize>> = q.codes.iter().map(|c| host_ints(c.clone())).collect();
    for k in 0..=q.codes.len() {
        let model_side = host(model.reconstruct(q.codes.get(..k).unwrap_or(&[]), b)?);
        let reference = host_reconstruction(&model, host_codes.get(..k).unwrap_or(&[]), b)?;
        residual = worst(residual, max_abs_diff(&model_side, &reference) / scale);
    }
    Ok(residual)
}

/// After `steps` updates with the same assignment statistics, the EMA state
/// against `N_n = γⁿN_0 + (1−γⁿ)n̄`, `m_n = γⁿm_0 + (1−γⁿ)s̄`, and the
/// codebook against `m_n / Ñ_n`, all in f64 on the host. Relative error.
pub(crate) fn ema_closed_form_residual(
    update: impl Fn(
        Tensor<CertBackend, 1>,
        Tensor<CertBackend, 2>,
        Tensor<CertBackend, 1>,
        Tensor<CertBackend, 2>,
        f64,
    ) -> (Tensor<CertBackend, 1>, Tensor<CertBackend, 2>),
) -> Result<f64> {
    let device = Default::default();
    CertBackend::seed(&device, 5);
    let (c, d, steps, decay, epsilon) = (6usize, 4usize, 25i32, 0.9f64, 1e-5f64);
    let n0 = Tensor::<CertBackend, 1>::random([c], Distribution::Uniform(0.5, 2.0), &device);
    let m0 = Tensor::<CertBackend, 2>::random([c, d], Distribution::Normal(0.0, 1.0), &device);
    // Fixed assignments: code 3 never used, so its state only decays.
    let counts_host = [4.0f32, 1.0, 2.0, 0.0, 7.0, 1.0];
    let counts = Tensor::<CertBackend, 1>::from_floats(counts_host, &device);
    let sums = Tensor::<CertBackend, 2>::random([c, d], Distribution::Normal(0.0, 3.0), &device)
        * counts.clone().reshape([c, 1]);
    let (mut n, mut m) = (n0.clone(), m0.clone());
    for _ in 0..steps {
        (n, m) = update(n, m, counts.clone(), sums.clone(), decay);
    }
    let g = decay.powi(steps);
    let n_ref: Vec<f64> = host(n0)
        .iter()
        .zip(counts_host)
        .map(|(a, nb)| g * a + (1.0 - g) * f64::from(nb))
        .collect();
    let m_ref: Vec<f64> = host(m0)
        .iter()
        .zip(host(sums))
        .map(|(a, s)| g * a + (1.0 - g) * s)
        .collect();
    let total: f64 = n_ref.iter().sum();
    let e_ref: Vec<f64> = m_ref
        .iter()
        .enumerate()
        .map(|(i, mv)| {
            let size = n_ref.get(i / d).copied().unwrap_or(f64::NAN);
            mv / ((size + epsilon) / (total + c as f64 * epsilon) * total)
        })
        .collect();
    let codebook = host(codebook_from_state(n.clone(), m.clone(), epsilon));
    let relative = |got: &[f64], want: &[f64]| -> f64 {
        if got.len() != want.len() {
            return f64::INFINITY;
        }
        got.iter().zip(want).fold(0.0f64, |acc, (a, b)| {
            worst(acc, (a - b).abs() / (1.0 + b.abs()))
        })
    };
    Ok(worst(
        worst(relative(&host(n), &n_ref), relative(&host(m), &m_ref)),
        relative(&codebook, &e_ref),
    ))
}

/// The tolerance for comparing gradients from two *separate* backward passes.
///
/// Measured, not assumed: on the `NdArray` backend two backward passes of the
/// very same graph on the very same weights differ by up to ~1e-6 relative
/// (6.7e-7 absolute on the micro tokenizer) — forward values are bit-identical
/// every time, the backward's floating-point summation is not reproducible.
/// So a cross-pass gradient comparison can only be exact to rounding; 1e-4
/// relative leaves two orders of magnitude of margin above that floor while a
/// real change (a halved estimator, a non-zero intent weight) moves the
/// gradients by more than 1e-2 relative.
pub(crate) const CROSS_PASS_GRADIENT_TOLERANCE: f64 = 1e-4;

/// `max|a − b| / max|b|`: a gradient difference relative to the gradient's
/// own scale.
pub(crate) fn relative_gap(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, v| worst(m, v.abs()));
    max_abs_diff(a, b) / scale.max(f64::MIN_POSITIVE)
}

/// Straight-through, within one backward pass: with both the encoder output
/// `z` and the reconstruction `ẑ` as leaves of the same graph, `∂L/∂z`
/// through `estimator(z, ẑ)` against `∂L/∂ẑ`, for the real decoder and
/// cross-entropy. Exact: both gradients are read off the one pass.
///
/// `cross_pass` instead compares `∂L/∂z` with `∂L/∂ẑ` from an independent
/// pass where the decoder reads `ẑ` directly, returning the gap relative to
/// the gradient scale (see [`CROSS_PASS_GRADIENT_TOLERANCE`]).
pub(crate) fn straight_through_residual(
    estimator: impl Fn(Tensor<CertAutodiff, 3>, Tensor<CertAutodiff, 3>) -> Tensor<CertAutodiff, 3>,
    cross_pass: bool,
) -> Result<f64> {
    let device = Default::default();
    let (model, tokens) = micro_tokenizer::<CertAutodiff>(41, 3, &device)?;
    let z = model.latents(tokens.clone())?.detach().require_grad();
    let q = model.quantize(z.clone(), None, false)?;
    let reconstruction = q.selected.clone().detach().require_grad();
    let loss = NcmTokenizer::cross_entropy(
        model.decode(estimator(z.clone(), reconstruction.clone()))?,
        tokens.clone(),
    );
    let grads = loss.backward();
    let g_z = host(
        z.grad(&grads)
            .context("no gradient reached z through the estimator")?,
    );
    if !cross_pass {
        let g_hat = host(
            reconstruction
                .grad(&grads)
                .context("no gradient reached the reconstruction")?,
        );
        return Ok(max_abs_diff(&g_z, &g_hat));
    }
    let leaf = q.selected.detach().require_grad();
    let loss = NcmTokenizer::cross_entropy(model.decode(leaf.clone())?, tokens);
    let grads = loss.backward();
    let g_hat = host(
        leaf.grad(&grads)
            .context("no gradient reached the reconstruction leaf")?,
    );
    Ok(relative_gap(&g_z, &g_hat))
}

/// The training forward's per-sample `ẑ^(c_b)` against
/// `reconstruct(&codes[..c_b])`, bit for bit; `offset` shifts the
/// reference prefix (non-zero only to show the check can fail).
pub(crate) fn nested_dropout_residual(offset: usize) -> Result<f64> {
    let device = Default::default();
    let (model, tokens) = micro_tokenizer::<CertBackend>(53, 4, &device)?;
    let [b, l] = tokens.dims();
    let d = model.config().latent_dim;
    let cutoffs = [1usize, 3, 2, 1];
    let q = model.quantize(model.latents(tokens)?, Some(&cutoffs), false)?;
    let selected = host(q.selected);
    let mut residual = 0.0f64;
    for (s, &c) in cutoffs.iter().enumerate().take(b) {
        let upto = (c + offset).min(q.codes.len());
        let reference = host(model.reconstruct(q.codes.get(..upto).unwrap_or(&[]), b)?);
        let span = s * l * d..(s + 1) * l * d;
        let (Some(got), Some(want)) = (selected.get(span.clone()), reference.get(span)) else {
            bail!("sample {s} is out of range");
        };
        residual = worst(residual, max_abs_diff(got, want));
    }
    Ok(residual)
}

/// The `ncm` certificate group's tokenizer checks.
pub fn certificate_checks() -> Result<Vec<crate::verify::Certificate>> {
    let intent = train::intent_off_check()?;
    Ok(vec![
        ncm_cert(
            "schedule_and_weights",
            "Geometric schedules are strictly increasing from >=1 and end at L; impossible ones are \
             refused; alpha=0 weights every code 1/sum(l) with total mass 1, alpha=1 gives every level \
             w_k l_k = 1/K.",
            schedule_and_weights_residual(),
            1e-12,
        ),
        ncm_cert(
            "windows_tile_and_pool_inverts_upsample",
            "Down_k's windows [floor(iL/l), floor((i+1)L/l)) are contiguous, non-empty, tile [0, L) \
             with sizes floor or ceil of L/l (dividing or not); pooling equals the host window mean; \
             Down_k(Up_k(v)) = v.",
            windows_residual()?,
            1e-6,
        ),
        ncm_cert(
            "quantizer_picks_the_nearest_code",
            "The quantizer's code for every row is a nearest codebook row in squared distance, \
             against a brute-force f64 scan (normalised distance gap).",
            nearest_code_residual()?,
            1e-5,
        ),
        ncm_cert(
            "encode_matches_level_codebook_argmin",
            "encode's level-k code is the argmin, over exactly the rows level_codebook(k) returns, of \
             the host window means of z - reconstruct(codes[..k]) (the SoftConceptCodec promise).",
            encode_matches_codebook_residual()?,
            1e-5,
        ),
        ncm_cert(
            "residual_reconstruction_telescopes",
            "zhat^(K) + z^(K) = z with the residual carried by running subtraction, and \
             reconstruct(codes[..k]) = sum_{j<k} R_j(e_j(r_j)) recomputed on the host from codebook \
             rows, lift weights and the window map, for every k (relative to max|z|).",
            telescoping_residual()?,
            1e-5,
        ),
        ncm_cert(
            "ema_codebook_update_matches_closed_form",
            "After n EMA steps with fixed assignments, N = g^n N0 + (1-g^n) n, m = g^n m0 + (1-g^n) s \
             and the codebook is m / Laplace(N) (relative error, f32 recursion).",
            ema_closed_form_residual(ema_step::<CertBackend>)?,
            1e-5,
        ),
        ncm_cert(
            "straight_through_gradient_is_identity",
            "Through zhat + (z - sg[z]) the gradient of the reconstruction cross-entropy w.r.t. the \
             encoder output equals its gradient w.r.t. zhat, bit for bit, within one backward pass.",
            straight_through_residual(straight_through::<CertAutodiff>, false)?,
            0.0,
        ),
        ncm_cert(
            "straight_through_matches_a_direct_decoder_pass",
            "The encoder gradient through the estimator equals the gradient of a separate pass in \
             which the decoder reads zhat directly, to the backend's backward reproducibility \
             (relative; repeated identical backward passes differ by ~1e-6 on NdArray).",
            straight_through_residual(straight_through::<CertAutodiff>, true)?,
            CROSS_PASS_GRADIENT_TOLERANCE,
        ),
        ncm_cert(
            "nested_dropout_truncates_exactly",
            "With per-sample cutoffs c_b, the training reconstruction of sample b equals \
             reconstruct(codes[..c_b]) bit for bit: dropped levels contribute exactly zero.",
            nested_dropout_residual(0)?,
            0.0,
        ),
        ncm_cert(
            "intent_weight_zero_skips_the_term",
            "With the intent weight at 0 the reader is never called, the intent head gets no \
             gradient and the training loss is bit for bit that of a run without the term; a \
             positive weight reads, trains the head and changes the loss.",
            intent.skipped,
            0.0,
        ),
        ncm_cert(
            "intent_weight_zero_matches_gradients",
            "With the intent weight at 0 every tokenizer gradient equals that of a run without the \
             term to the backend's backward reproducibility (relative), while a positive weight \
             moves them beyond it.",
            intent.gradient,
            CROSS_PASS_GRADIENT_TOLERANCE,
        ),
    ])
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

    type B = CertBackend;

    #[test]
    fn test_every_certificate_holds() {
        for cert in certificate_checks().unwrap() {
            assert!(
                cert.passed(),
                "{}: residual {} > tolerance {}",
                cert.name,
                cert.residual,
                cert.tolerance
            );
        }
    }

    #[test]
    fn test_presets_validate_and_have_the_paper_shapes() {
        let tiny = NcmTokenizerConfig::tiny().unwrap();
        assert_eq!(tiny.schedule.lengths(), &[2, 4, 8, 16, 32, 64]);
        let owt = NcmTokenizerConfig::paper_owt().unwrap();
        assert_eq!(
            (owt.seq_len(), owt.schedule.levels(), owt.latent_dim),
            (1024, 32, 384)
        );
        assert_eq!(
            (owt.encoder_blocks, owt.decoder_blocks, owt.codebook_size),
            (0, 1, 4096)
        );
        let nina = NcmTokenizerConfig::nina_4k().unwrap();
        assert_eq!((nina.seq_len(), nina.schedule.levels()), (4096, 32));
        let mut bad = tiny.clone();
        bad.heads = 5;
        assert!(bad.validate().is_err());
        bad = tiny;
        bad.ema_decay = 1.0;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_a_perturbed_code_breaks_the_nearest_check_but_not_the_telescope() {
        // Anti-vacuity: move one code to its worst choice. The reconstruction
        // still telescopes (it is self-consistent for any codes), but the
        // nearest-code certificate must fail.
        let device = Default::default();
        let (model, tokens) = micro_tokenizer::<B>(7, 2, &device).unwrap();
        let q = model
            .quantize(model.latents(tokens).unwrap(), None, false)
            .unwrap();
        let first = q.pooled.first().unwrap().clone();
        let [b, len, d] = first.dims();
        let rows = host(first.reshape([b * len, d]));
        let codebook = host(model.codebook(0).unwrap());
        let mut chosen = host_ints(q.codes.first().unwrap().clone().reshape([b * len]));
        assert!(nearest_code_gap(&rows, &codebook, &chosen, d).unwrap() <= 1e-5);
        // The farthest code from row 0.
        let c = codebook.len() / d;
        let far = (0..c)
            .max_by(|&x, &y| {
                let dist = |k: usize| -> f64 {
                    (0..d)
                        .map(|i| (rows[i] - codebook[k * d + i]).powi(2))
                        .sum()
                };
                dist(x).partial_cmp(&dist(y)).unwrap()
            })
            .unwrap();
        chosen[0] = far;
        assert!(nearest_code_gap(&rows, &codebook, &chosen, d).unwrap() > 1e-3);
        // Telescoping with the perturbed code still holds.
        let mut codes = q.codes.clone();
        let mut flat = host_ints(codes[0].clone().reshape([b * len]));
        flat[0] = far;
        let ids: Vec<i64> = flat.iter().map(|&v| v as i64).collect();
        codes[0] = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([b, len]);
        let host_codes: Vec<Vec<usize>> = codes.iter().map(|c| host_ints(c.clone())).collect();
        let model_side = host(model.reconstruct(&codes, b).unwrap());
        let reference = host_reconstruction(&model, &host_codes, b).unwrap();
        assert!(max_abs_diff(&model_side, &reference) < 1e-4);
    }

    #[test]
    fn test_wrong_ema_convention_and_detached_estimator_fail() {
        // Weighting the live statistics by the decay is the bug the crate's
        // EMA convention exists to prevent; the closed form must reject it.
        let inverted =
            |n: Tensor<B, 1>, m: Tensor<B, 2>, c: Tensor<B, 1>, s: Tensor<B, 2>, g: f64| {
                ema_step(n, m, c, s, 1.0 - g)
            };
        assert!(ema_closed_form_residual(inverted).unwrap() > 1e-2);
        // An estimator that forgets the straight-through term gives the
        // encoder no gradient at all.
        let detached = |z: Tensor<CertAutodiff, 3>, zhat: Tensor<CertAutodiff, 3>| {
            zhat + (z.clone() - z).detach()
        };
        assert!(straight_through_residual(detached, false).is_err());
        assert!(straight_through_residual(detached, true).is_err());
        // A halved straight-through term is caught by both forms.
        let halved = |z: Tensor<CertAutodiff, 3>, zhat: Tensor<CertAutodiff, 3>| {
            zhat + (z.clone() - z.detach()).mul_scalar(0.5)
        };
        assert!(straight_through_residual(halved, false).unwrap() > 0.0);
        assert!(straight_through_residual(halved, true).unwrap() > 1e-2);
        // The cross-pass form passes with margin on the real estimator.
        let gap = straight_through_residual(straight_through::<CertAutodiff>, true).unwrap();
        assert!(
            gap < CROSS_PASS_GRADIENT_TOLERANCE / 10.0,
            "cross-pass gap {gap}"
        );
    }

    #[test]
    fn test_nested_dropout_check_can_fail() {
        assert_eq!(nested_dropout_residual(0).unwrap(), 0.0);
        assert!(nested_dropout_residual(1).unwrap() > 0.0);
    }

    #[test]
    fn test_codec_shapes_and_empty_prefix() {
        let device = Default::default();
        let (model, tokens) = micro_tokenizer::<B>(3, 2, &device).unwrap();
        let codes = model.encode(tokens).unwrap().codes;
        let lengths: Vec<usize> = codes.iter().map(|c| c.dims()[1]).collect();
        assert_eq!(lengths, vec![2, 5, 12]);
        let zero = model.reconstruct(&[], 2).unwrap();
        assert_eq!(zero.dims(), [2, 12, 8]);
        assert!(host(zero).iter().all(|&v| v == 0.0));
        let logits = model
            .decode_logits(model.reconstruct(&codes, 2).unwrap())
            .unwrap();
        assert_eq!(logits.dims(), [2, 12, crate::tokenizer::VOCAB_SIZE]);
        assert_eq!(model.level_codebook(2).unwrap().dims(), [16, 8]);
        assert!(model.level_codebook(3).is_err());
        assert!(model.reconstruct(&codes[..1], 3).is_err());
    }

    #[test]
    fn test_dead_codes_are_restarted_and_live_ones_kept() {
        let device = Default::default();
        let (model, tokens) = micro_tokenizer::<B>(9, 2, &device).unwrap();
        let q = model
            .quantize(model.latents(tokens).unwrap(), None, true)
            .unwrap();
        // Decay every count towards zero except code 0's, then sweep.
        let mut model = model;
        for _ in 0..200 {
            let stats: Vec<LevelStats<B>> = (0..3)
                .map(|_| {
                    let mut counts = vec![0.0f32; 16];
                    counts[0] = 50.0;
                    let mut sums = vec![0.0f32; 16 * 8];
                    for (i, slot) in sums.iter_mut().take(8).enumerate() {
                        *slot = 50.0 * (i as f32 + 1.0) / 8.0;
                    }
                    LevelStats {
                        counts: Tensor::from_floats(counts.as_slice(), &device),
                        sums: Tensor::<B, 1>::from_floats(sums.as_slice(), &device)
                            .reshape([16, 8]),
                    }
                })
                .collect();
            model = model.apply_codebook_ema(&stats).unwrap();
        }
        let before = host(model.codebook(1).unwrap());
        let mut rng = <rand_chacha::ChaCha12Rng as rand::SeedableRng>::seed_from_u64(1);
        let (model, restarted) = model.restart_dead_codes(&q.pooled, &mut rng).unwrap();
        assert_eq!(restarted, vec![15, 15, 15]);
        let after = host(model.codebook(1).unwrap());
        // Code 0 (live) is untouched up to the smoothing renormalisation,
        // and it is the vector its statistics point at, not zero.
        assert!(max_abs_diff(&before[..8], &after[..8]) < 1e-3);
        assert!(
            (after[7] - 1.0).abs() < 1e-3,
            "live code drifted: {:?}",
            &after[..8]
        );
        // A restarted code now sits next to a pooled vector.
        let pooled = q.pooled[1].clone();
        let [b, len, d] = pooled.dims();
        let rows = host(pooled.reshape([b * len, d]));
        let code = &after[8..16];
        let nearest = (0..b * len)
            .map(|r| {
                (0..d)
                    .map(|i| (rows[r * d + i] - code[i]).powi(2))
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest < 1e-3,
            "restarted code is {nearest} from every pooled vector"
        );
    }

    #[test]
    fn test_code_usage() {
        assert_eq!(code_usage(&[0, 0]), (0, 0.0));
        let (used, ppl) = code_usage(&[5, 5, 0, 5, 5]);
        assert_eq!(used, 4);
        assert!((ppl - 4.0).abs() < 1e-12);
    }
}
