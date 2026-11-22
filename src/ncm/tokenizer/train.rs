//! Training and evaluation of the NCM tokenizer.
//!
//! # Data
//!
//! Training windows are `L` tokens read from a crate corpus (headerless
//! little-endian `u16`, BOS = 256 / EOS = 257, [`crate::corpus`]), sampled
//! uniformly with replacement from the training region. Validation never
//! overlaps it:
//!
//! - with one or more `val_corpora` (e.g. the per-field `<field>.val.corpus.bin`
//!   files cut at document level upstream), each is its own validation set,
//!   reported separately, and the whole training corpus trains;
//! - with none, the tail `val_fraction` of the training corpus is held out,
//!   the cut moved forward to the next BOS so no document straddles the
//!   split.
//!
//! Validation windows are non-overlapping tiles from the start of each set,
//! so they are the same windows at every evaluation.
//!
//! # Loop
//!
//! AdamW (β = 0.9 / 0.95), linear warmup over 5% of the run then cosine decay
//! to 10% of the peak ([`LrSchedule::WarmupCosine`]), global-norm clipping,
//! optional gradient accumulation with the crate's convention (each
//! micro-batch's loss scaled by `1/k`, gradients summed:
//! [`GradientAccumulator`]). Codebook statistics are summed over the
//! micro-batches of a step and folded into the EMA once per optimizer step;
//! dead codes are swept every `restart_every` steps. The device RNG is
//! reseeded from `(seed, step)` at every step and the host RNG is saved in the
//! training state ([`crate::train::step_seed`]), so `--resume` restores the
//! exact state: the first resumed step's forward pass is bit-identical to the
//! interrupted run's. Later steps agree only to rounding, because on the
//! `NdArray` backend the backward pass is not bit-reproducible (two identical
//! backward passes differ by ~1e-6 relative; measured, see
//! `super::CROSS_PASS_GRADIENT_TOLERANCE`).
//!
//! # Optional intent alignment (off by default, unvalidated)
//!
//! With [`IntentConfig::weight`] `λ > 0` an auxiliary term asks the coarse
//! concepts to carry what a frozen *reader* model takes from the same
//! window: the straight-through `ẑ^(c)` of the first `c` levels is averaged
//! over positions, projected by a small linear head, and compared (cosine or
//! MSE) with the reader's mean hidden state. The reader is a nina student
//! checkpoint ([`load_student_reader`]) evaluated on the inner backend, so no
//! gradient can reach it. **Whether this makes the codes carry intent is not
//! measured by anything here**; treat it as an experiment until a downstream
//! evaluation says otherwise. At `λ = 0` the term is not computed at all: the
//! reader is never called, the head gets no gradient and the loss is bit for
//! bit that of a run without it (certificate `intent_weight_zero_skips_the_term`);
//! the gradients then agree to the backend's backward reproducibility
//! (`intent_weight_zero_matches_gradients`). The head is built after the
//! tokenizer and draws nothing from the host RNG, so turning the term on never
//! changes the tokenizer's initialization or its batches.

use super::{code_usage, straight_through, LevelStats, NcmTokenizer, NcmTokenizerConfig};
use crate::checkpoint::{self, DatasetIdentity, TrainState};
use crate::corpus::TokenCorpus;
use crate::ncm::ConceptCodec;
use crate::quality::global_grad_norm;
use crate::schedule::{clip_gradients, Cycle, GradientAccumulator, LrSchedule};
use anyhow::{bail, ensure, Context as _, Result};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The `kind` this trainer writes into [`TrainState`].
pub const STATE_KIND: &str = "ncm-tokenizer";

/// File stem of the content-addressed tokenizer checkpoints.
pub const CHECKPOINT_STEM: &str = "ncm-tokenizer";

/// How the intent term compares the projected concept with the reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntentLoss {
    /// `1 − cos(p, r)`, scale-free.
    Cosine,
    /// `mean((p − r)²)`.
    Mse,
}

impl IntentLoss {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "cosine" => Ok(Self::Cosine),
            "mse" => Ok(Self::Mse),
            other => bail!("unknown intent loss '{other}' (expected cosine|mse)"),
        }
    }
}

/// The optional intent-alignment term. Off unless `weight > 0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentConfig {
    /// `λ`. Zero means the term is never computed.
    pub weight: f64,
    /// `c`: the term reads `ẑ^(c)`, the first `c` levels.
    pub levels: usize,
    pub loss: IntentLoss,
    /// Where the reader came from, for the record.
    #[serde(default)]
    pub reader: Option<String>,
}

impl IntentConfig {
    pub fn is_active(&self) -> bool {
        self.weight != 0.0
    }
}

/// A frozen model that summarises a token window as one vector.
pub trait IntentReader<B: Backend> {
    /// Width of the summary.
    fn width(&self) -> usize;
    /// Summaries `[b, width]` of token windows `[b, L]`.
    fn read(&self, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>>;
    /// One line naming the reader.
    fn describe(&self) -> String;
}

/// A nina student as an intent reader: the mean over positions of its
/// residual stream after the geometric write-back.
pub struct StudentReader<B: Backend<FloatElem = f32>> {
    student: crate::student::GeometricStudent<B>,
    description: String,
}

impl<B: Backend<FloatElem = f32>> StudentReader<B> {
    /// Wrap a loaded student that will read windows of `seq_len` tokens.
    pub fn new(
        student: crate::student::GeometricStudent<B>,
        seq_len: usize,
        description: String,
    ) -> Result<Self> {
        let trunk = student.trunk();
        ensure!(
            !trunk.positions_bounded() || seq_len <= trunk.context(),
            "the reader's context is {} tokens, the tokenizer reads windows of {seq_len}",
            trunk.context()
        );
        Ok(Self {
            student,
            description,
        })
    }
}

impl<B: Backend<FloatElem = f32>> IntentReader<B> for StudentReader<B> {
    fn width(&self) -> usize {
        self.student.config().hidden_size
    }

    fn read(&self, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>> {
        let hidden = self.student.forward(tokens)?.hidden;
        let [b, _, h] = hidden.dims();
        Ok(hidden.mean_dim(1).reshape([b, h]).detach())
    }

    fn describe(&self) -> String {
        self.description.clone()
    }
}

/// Load a nina student checkpoint as an intent reader. Its architecture
/// comes from `config` (a `StudentConfig` JSON) or, without one, from the
/// `student_config` its training state recorded.
pub fn load_student_reader<B: Backend<FloatElem = f32>>(
    checkpoint_path: &Path,
    config: Option<&Path>,
    seq_len: usize,
    device: &B::Device,
) -> Result<StudentReader<B>> {
    let student_config: crate::student::StudentConfig = match config {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read student config {}", path.display()))?;
            serde_json::from_str(&text)
                .with_context(|| format!("parse student config {}", path.display()))?
        }
        None => {
            let state = TrainState::for_model(checkpoint_path)?.with_context(|| {
                format!(
                    "{} has no training state to read its architecture from; pass the \
                     student config JSON explicitly",
                    checkpoint_path.display()
                )
            })?;
            ensure!(
                state.kind == "student",
                "{} is a '{}' checkpoint; the intent reader must be a nina student",
                checkpoint_path.display(),
                state.kind
            );
            let value = state
                .config
                .get("student_config")
                .cloned()
                .context("the student's training state records no student_config")?;
            serde_json::from_value(value).context("parse the recorded student_config")?
        }
    };
    student_config.validate()?;
    let student = crate::student::GeometricStudent::<B>::new(&student_config, device)?;
    crate::tensor_ext::force_initialization(&student);
    let student = checkpoint::load::<B, _>(student, checkpoint_path, device)?;
    StudentReader::new(
        student,
        seq_len,
        format!(
            "student {} (hidden {})",
            checkpoint_path.display(),
            student_config.hidden_size
        ),
    )
}

/// A lexical baseline reader: the normalised byte histogram of the window.
/// It carries no meaning beyond token counts, which is what makes it the
/// right control for the intent term.
#[derive(Debug, Clone, Copy)]
pub struct ByteHistogramReader {
    pub vocab: usize,
}

impl<B: Backend> IntentReader<B> for ByteHistogramReader {
    fn width(&self) -> usize {
        self.vocab
    }

    fn read(&self, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>> {
        let [b, _] = tokens.dims();
        let one_hot: Tensor<B, 3, Int> = tokens.one_hot(self.vocab);
        Ok(one_hot.float().mean_dim(1).reshape([b, self.vocab]))
    }

    fn describe(&self) -> String {
        format!("byte histogram over {} tokens", self.vocab)
    }
}

/// The intent term's projection from the latent width to the reader's.
#[derive(Module, Debug)]
pub struct IntentHead<B: Backend> {
    proj: Linear<B>,
}

impl<B: Backend> IntentHead<B> {
    pub fn new(latent_dim: usize, reader_width: usize, device: &B::Device) -> Self {
        Self {
            proj: LinearConfig::new(latent_dim, reader_width)
                .with_bias(true)
                .init(device),
        }
    }

    pub fn width(&self) -> usize {
        let [_, out] = self.proj.weight.val().dims();
        out
    }
}

/// What the optimizer steps: the tokenizer, plus the intent head when the
/// intent term is on. With it off this is the tokenizer alone.
#[derive(Module, Debug)]
pub struct NcmTrainee<B: Backend> {
    pub tokenizer: NcmTokenizer<B>,
    pub intent: Option<IntentHead<B>>,
}

/// Host values of one micro-batch's loss terms.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct LossParts {
    pub total: f64,
    pub recon_ce: f64,
    pub commitment: f64,
    pub codebook: f64,
    pub intent: f64,
    /// Token accuracy of the training reconstruction (at each sample's cutoff).
    pub accuracy: f64,
}

impl LossParts {
    fn accumulate(&mut self, other: &Self, weight: f64) {
        self.total += other.total * weight;
        self.recon_ce += other.recon_ce * weight;
        self.commitment += other.commitment * weight;
        self.codebook += other.codebook * weight;
        self.intent += other.intent * weight;
        self.accuracy += other.accuracy * weight;
    }
}

/// One micro-batch's gradients, loss terms and codebook statistics.
#[derive(Debug)]
pub struct MicroOutcome<B: AutodiffBackend> {
    pub grads: GradientsParams,
    pub losses: LossParts,
    pub stats: Vec<LevelStats<B>>,
    pub pooled: Vec<Tensor<B, 3>>,
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    t.into_data().iter::<f64>().next().unwrap_or(f64::NAN)
}

fn intent_term<B: AutodiffBackend>(
    tokenizer: &NcmTokenizer<B>,
    head: &IntentHead<B>,
    z: Tensor<B, 3>,
    codes: &[Tensor<B, 2, Int>],
    tokens: Tensor<B, 2, Int>,
    config: &IntentConfig,
    reader: &dyn IntentReader<B::InnerBackend>,
) -> Result<Tensor<B, 1>> {
    let [b, _, d] = z.dims();
    ensure!(
        (1..=codes.len()).contains(&config.levels),
        "intent levels must be in 1..={}, got {}",
        codes.len(),
        config.levels
    );
    ensure!(
        reader.width() == head.width(),
        "the reader summarises to width {}, the intent head projects to {}",
        reader.width(),
        head.width()
    );
    let prefix = codes.get(..config.levels).unwrap_or(&[]);
    let coarse = straight_through(z, tokenizer.reconstruct(prefix, b)?);
    let concept = head.proj.forward(coarse.mean_dim(1).reshape([b, d]));
    let target = Tensor::<B, 2>::from_inner(reader.read(tokens.inner())?);
    Ok(match config.loss {
        IntentLoss::Cosine => {
            let dot = (concept.clone() * target.clone()).sum_dim(1);
            let norms = (concept.clone() * concept).sum_dim(1).sqrt()
                * (target.clone() * target).sum_dim(1).sqrt();
            (dot / norms.add_scalar(1e-8)).mean().neg().add_scalar(1.0)
        }
        IntentLoss::Mse => (concept - target).powf_scalar(2.0).mean(),
    })
}

/// Forward and backward of one micro-batch.
///
/// `cutoffs` are the per-sample nested-dropout cutoffs; `loss_scale` is the
/// accumulator's `1/k`. The intent term runs only when `intent` is given
/// *and* its weight is non-zero.
pub fn micro_step<B: AutodiffBackend>(
    trainee: &NcmTrainee<B>,
    tokens: Tensor<B, 2, Int>,
    cutoffs: &[usize],
    loss_scale: f64,
    intent: Option<(&IntentConfig, &dyn IntentReader<B::InnerBackend>)>,
) -> Result<MicroOutcome<B>> {
    let tokenizer = &trainee.tokenizer;
    let beta = tokenizer.config().commitment_beta;
    let [b, l] = tokens.dims();
    let z = tokenizer.latents(tokens.clone())?;
    let q = tokenizer.quantize(z.clone(), Some(cutoffs), true)?;
    let logits = tokenizer.decode(straight_through(z.clone(), q.selected.clone()))?;
    let recon = NcmTokenizer::cross_entropy(logits.clone(), tokens.clone());
    let commitment = (z.clone() - q.selected.clone().detach())
        .powf_scalar(2.0)
        .mean();
    let codebook = (z.clone().detach() - q.selected.clone())
        .powf_scalar(2.0)
        .mean();
    let mut total = recon.clone() + commitment.clone().mul_scalar(beta) + codebook.clone();
    let mut intent_value = 0.0;
    if let Some((config, reader)) = intent.filter(|(config, _)| config.is_active()) {
        let head = trainee
            .intent
            .as_ref()
            .context("the intent term is on but the trainee has no intent head")?;
        let term = intent_term(tokenizer, head, z, &q.codes, tokens.clone(), config, reader)?;
        intent_value = scalar(term.clone());
        total = total + term.mul_scalar(config.weight);
    }
    let correct = logits
        .argmax(2)
        .reshape([b, l])
        .equal(tokens)
        .int()
        .sum()
        .into_data()
        .iter::<i64>()
        .next()
        .unwrap_or(0);
    let losses = LossParts {
        total: scalar(total.clone()),
        recon_ce: scalar(recon),
        commitment: scalar(commitment),
        codebook: scalar(codebook),
        intent: intent_value,
        accuracy: correct as f64 / (b * l) as f64,
    };
    let grads = GradientsParams::from_grads(total.mul_scalar(loss_scale).backward(), trainee);
    Ok(MicroOutcome {
        grads,
        losses,
        stats: q.stats,
        pooled: q.pooled,
    })
}

/// Everything a tokenizer training run is configured by.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenizerTrainConfig {
    pub tokenizer: NcmTokenizerConfig,
    pub corpus: PathBuf,
    /// Held-out sets, one report each; empty holds out the corpus tail.
    #[serde(default)]
    pub val_corpora: Vec<PathBuf>,
    /// Train on whole pre-shuffled response records in corpus order.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ordered_records: bool,
    /// Tail fraction held out when `val_corpora` is empty.
    pub val_fraction: f64,
    pub steps: usize,
    /// End this invocation after saving the given step of the full schedule.
    #[serde(skip)]
    pub stop_at: Option<usize>,
    pub batch_size: usize,
    /// Micro-batches per optimizer step.
    pub accumulate: usize,
    /// Peak learning rate.
    pub lr: f64,
    pub weight_decay: f64,
    /// Global gradient-norm clip.
    pub clip_norm: f64,
    pub seed: u64,
    pub log_every: usize,
    /// Optimizer steps between evaluations (0: only at start and end).
    pub eval_every: usize,
    /// Validation windows per set, at most.
    pub val_windows: usize,
    /// Optimizer steps between checkpoints (0: only at the end).
    pub checkpoint_every: usize,
    pub out_dir: Option<PathBuf>,
    pub resume: Option<PathBuf>,
    #[serde(default)]
    pub intent: Option<IntentConfig>,
}

impl TokenizerTrainConfig {
    /// Defaults for a tokenizer config: peak LR 3e-4 for the paper shapes
    /// (the paper's generator LR; its tokenizer LR is not in the extracted
    /// tables), 1e-3 for anything smaller than 256 wide.
    pub fn new(tokenizer: NcmTokenizerConfig, corpus: PathBuf) -> Self {
        let lr = if tokenizer.latent_dim >= 256 {
            3e-4
        } else {
            1e-3
        };
        Self {
            tokenizer,
            corpus,
            val_corpora: Vec::new(),
            ordered_records: false,
            val_fraction: 0.05,
            steps: 300,
            stop_at: None,
            batch_size: 16,
            accumulate: 1,
            lr,
            weight_decay: 0.0,
            clip_norm: 1.0,
            seed: 42,
            log_every: 10,
            eval_every: 0,
            val_windows: 64,
            checkpoint_every: 0,
            out_dir: None,
            resume: None,
            intent: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.tokenizer.validate()?;
        ensure!(self.steps >= 1, "steps must be at least 1");
        super::super::check_stop_step(self.steps, self.stop_at, self.out_dir.is_some())?;
        ensure!(self.batch_size >= 1, "batch_size must be at least 1");
        ensure!(self.accumulate >= 1, "accumulate must be at least 1");
        ensure!(
            self.lr > 0.0 && self.lr.is_finite(),
            "lr must be positive and finite, got {}",
            self.lr
        );
        ensure!(
            self.weight_decay >= 0.0 && self.weight_decay.is_finite(),
            "weight_decay must be non-negative"
        );
        ensure!(
            self.clip_norm > 0.0 && self.clip_norm.is_finite(),
            "clip_norm must be positive"
        );
        ensure!(
            self.val_fraction > 0.0 && self.val_fraction < 1.0,
            "val_fraction must be in (0, 1), got {}",
            self.val_fraction
        );
        ensure!(self.val_windows >= 1, "val_windows must be at least 1");
        ensure!(
            !self.ordered_records || !self.val_corpora.is_empty(),
            "ordered codec records require a separate validation corpus"
        );
        if let Some(intent) = &self.intent {
            ensure!(
                intent.weight >= 0.0 && intent.weight.is_finite(),
                "the intent weight must be non-negative and finite, got {}",
                intent.weight
            );
            ensure!(
                (1..=self.tokenizer.schedule.levels()).contains(&intent.levels),
                "intent levels must be in 1..={}, got {}",
                self.tokenizer.schedule.levels(),
                intent.levels
            );
        }
        Ok(())
    }

    /// Peak `lr`, linear warmup over 5% of the run, cosine to 10% of peak
    /// (the paper's 3e-4 → 3e-5 ratio).
    pub fn lr_schedule(&self) -> LrSchedule {
        LrSchedule::WarmupCosine {
            peak: self.lr,
            min_lr: self.lr * 0.1,
            warmup_steps: (self.steps / 20).max(1),
            total_steps: self.steps,
        }
    }

    /// The config as recorded in the training state: the keys that may
    /// change across a resume without changing the run are left out.
    fn state_json(&self) -> Result<serde_json::Value> {
        let mut value = serde_json::to_value(self).context("serialize tokenizer train config")?;
        if let Some(map) = value.as_object_mut() {
            for key in ["log_every", "eval_every", "val_windows"] {
                map.remove(key);
            }
        }
        Ok(value)
    }
}

/// One held-out set: a name and its fixed windows.
#[derive(Debug, Clone)]
pub struct ValSet {
    pub name: String,
    pub windows: Vec<Vec<u16>>,
    /// Why the set has no windows, if it has none.
    pub note: Option<String>,
}

/// `coding.val.corpus.bin` → `coding`.
pub fn set_name(path: &Path) -> String {
    let file = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("corpus")
        .to_string();
    let stem = file
        .strip_suffix(".corpus.bin")
        .or_else(|| file.strip_suffix(".bin"))
        .unwrap_or(&file);
    stem.strip_suffix(".val").unwrap_or(stem).to_string()
}

fn read_tokens(path: &Path) -> Result<Vec<u16>> {
    let mut corpus = TokenCorpus::in_memory(path)?;
    let len = corpus.len();
    corpus.window(0, len)
}

/// Non-overlapping windows of `len` from the start of `tokens`, at most `max`.
pub fn tile_windows(tokens: &[u16], len: usize, max: usize) -> Vec<Vec<u16>> {
    tokens
        .chunks_exact(len.max(1))
        .take(max)
        .map(<[u16]>::to_vec)
        .collect()
}

/// Where the held-out tail starts: `len − ⌈fraction·len⌉`, moved forward to
/// the next BOS so no document straddles the split (unchanged when no BOS
/// follows).
pub fn tail_cut(tokens: &[u16], fraction: f64) -> usize {
    let len = tokens.len();
    let held = ((len as f64) * fraction).ceil() as usize;
    let nominal = len.saturating_sub(held);
    let bos = crate::tokenizer::Special::Bos.id();
    tokens
        .iter()
        .skip(nominal)
        .position(|&t| t == bos)
        .map_or(nominal, |offset| nominal + offset)
}

/// The training stream and the validation sets a config describes.
pub fn load_data(config: &TokenizerTrainConfig) -> Result<(Vec<u16>, Vec<ValSet>)> {
    let l = config.tokenizer.seq_len();
    let tokens = read_tokens(&config.corpus)?;
    ensure!(
        !config.ordered_records || tokens.len().is_multiple_of(l),
        "ordered codec training data must contain whole records of {l} tokens"
    );
    let make_set = |name: String, stream: &[u16]| -> ValSet {
        let windows = tile_windows(stream, l, config.val_windows);
        let note = windows.is_empty().then(|| {
            format!(
                "{} tokens is shorter than one {l}-token window; not evaluated",
                stream.len()
            )
        });
        ValSet {
            name,
            windows,
            note,
        }
    };
    let (train, sets) = if config.val_corpora.is_empty() {
        let cut = tail_cut(&tokens, config.val_fraction);
        let train = tokens.get(..cut).unwrap_or(&[]).to_vec();
        let held = tokens.get(cut..).unwrap_or(&[]);
        let set = make_set(
            format!("{} (held-out tail)", set_name(&config.corpus)),
            held,
        );
        (train, vec![set])
    } else {
        let mut sets = Vec::with_capacity(config.val_corpora.len());
        for path in &config.val_corpora {
            let tokens = read_tokens(path)?;
            ensure!(
                !config.ordered_records || tokens.len().is_multiple_of(l),
                "ordered codec validation data must contain whole records of {l} tokens"
            );
            sets.push(make_set(set_name(path), &tokens));
        }
        (tokens, sets)
    };
    ensure!(
        train.len() >= l,
        "the training region of {} has {} tokens, fewer than one {l}-token window",
        config.corpus.display(),
        train.len()
    );
    ensure!(
        sets.iter().any(|s| !s.windows.is_empty()),
        "no validation set has a full {l}-token window"
    );
    Ok((train, sets))
}

fn tokens_tensor<B: Backend>(
    windows: &[Vec<u16>],
    len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let flat: Vec<i64> = windows
        .iter()
        .flat_map(|w| w.iter().map(|&t| i64::from(t)))
        .collect();
    Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device).reshape([windows.len(), len])
}

fn draw_windows<R: Rng>(stream: &[u16], len: usize, count: usize, rng: &mut R) -> Vec<Vec<u16>> {
    let last = stream.len().saturating_sub(len);
    (0..count)
        .map(|_| {
            let start = rng.random_range(0..=last);
            stream.get(start..start + len).unwrap_or(&[]).to_vec()
        })
        .collect()
}

fn draw_cutoffs<R: Rng>(batch: usize, levels: usize, p: f64, rng: &mut R) -> Vec<usize> {
    (0..batch)
        .map(|_| {
            if rng.random::<f64>() < p {
                rng.random_range(1..=levels)
            } else {
                levels
            }
        })
        .collect()
}

/// Reconstruction quality at one cutoff level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelEval {
    /// 1-based cutoff `k`: the decoder read `ẑ^(k)`.
    pub level: usize,
    pub length: usize,
    pub recon_ce: f64,
    pub token_accuracy: f64,
}

/// One validation set at every cutoff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValSetEval {
    pub name: String,
    pub windows: usize,
    pub levels: Vec<LevelEval>,
    pub note: Option<String>,
}

/// Codebook use of one level over every validation window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelUsage {
    pub level: usize,
    pub length: usize,
    pub assignments: u64,
    pub used: usize,
    pub codebook_size: usize,
    /// `exp(H)` of the empirical code distribution.
    pub perplexity: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub step: usize,
    pub sets: Vec<ValSetEval>,
    pub usage: Vec<LevelUsage>,
}

/// Reconstruction cross-entropy and token accuracy at every cutoff, per set,
/// and codebook usage per level over all sets.
pub fn evaluate<B: Backend>(
    model: &NcmTokenizer<B>,
    sets: &[ValSet],
    batch_size: usize,
    step: usize,
    device: &B::Device,
) -> Result<EvalReport> {
    let config = model.config();
    let l = config.seq_len();
    let levels = config.schedule.levels();
    let c = config.codebook_size;
    let mut histograms = vec![vec![0u64; c]; levels];
    let mut out = Vec::with_capacity(sets.len());
    for set in sets {
        let mut ce = vec![0.0f64; levels];
        let mut correct = vec![0u64; levels];
        let mut seen = 0usize;
        for chunk in set.windows.chunks(batch_size.max(1)) {
            let b = chunk.len();
            let tokens = tokens_tensor::<B>(chunk, l, device);
            let q = model.quantize(model.latents(tokens.clone())?, None, false)?;
            let mut zhat = Tensor::<B, 3>::zeros([b, l, config.latent_dim], device);
            for (k, codes) in q.codes.iter().enumerate() {
                for code in codes.clone().into_data().iter::<i64>() {
                    let slot = usize::try_from(code)
                        .ok()
                        .and_then(|j| histograms.get_mut(k).and_then(|h| h.get_mut(j)))
                        .with_context(|| format!("level {k} produced code {code}"))?;
                    *slot += 1;
                }
                zhat = zhat + model.level_contribution(k, codes.clone())?;
                let logits = model.decode(zhat.clone())?;
                let loss = scalar(NcmTokenizer::cross_entropy(logits.clone(), tokens.clone()));
                let hits = logits
                    .argmax(2)
                    .reshape([b, l])
                    .equal(tokens.clone())
                    .int()
                    .sum()
                    .into_data()
                    .iter::<i64>()
                    .next()
                    .unwrap_or(0);
                if let (Some(sum), Some(hit)) = (ce.get_mut(k), correct.get_mut(k)) {
                    *sum += loss * b as f64;
                    *hit += u64::try_from(hits).unwrap_or(0);
                }
            }
            seen += b;
        }
        let level_evals = (0..levels)
            .map(|k| LevelEval {
                level: k + 1,
                length: config.schedule.len_of(k).unwrap_or(0),
                recon_ce: ce.get(k).copied().unwrap_or(f64::NAN) / seen.max(1) as f64,
                token_accuracy: correct.get(k).copied().unwrap_or(0) as f64
                    / (seen.max(1) * l) as f64,
            })
            .collect();
        out.push(ValSetEval {
            name: set.name.clone(),
            windows: seen,
            levels: if seen == 0 { Vec::new() } else { level_evals },
            note: set.note.clone(),
        });
    }
    let usage = histograms
        .iter()
        .enumerate()
        .map(|(k, h)| {
            let (used, perplexity) = code_usage(h);
            LevelUsage {
                level: k + 1,
                length: config.schedule.len_of(k).unwrap_or(0),
                assignments: h.iter().sum(),
                used,
                codebook_size: c,
                perplexity,
            }
        })
        .collect();
    Ok(EvalReport {
        step,
        sets: out,
        usage,
    })
}

/// One logged optimizer step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepLog {
    pub step: usize,
    pub lr: f64,
    pub grad_norm: f64,
    /// The scale clipping applied (1.0 when the norm was under the limit).
    pub clip_scale: f64,
    /// Mean over the step's micro-batches.
    pub losses: LossParts,
}

/// Host state carried in [`TrainState::extras`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TokenizerExtras {
    history: Vec<StepLog>,
    evals: Vec<EvalReport>,
    restarts: Vec<usize>,
    elapsed_secs: f64,
    windows_seen: usize,
}

/// What a training run did.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerTrainReport {
    pub config: NcmTokenizerConfig,
    /// Gradient-trained parameters (tokenizer plus intent head).
    pub trainable_parameters: usize,
    /// EMA codebook state, floats (`K · C · (d + 1)`).
    pub codebook_floats: usize,
    pub total_steps: usize,
    pub resumed_from: usize,
    pub train_tokens: usize,
    pub windows_seen: usize,
    pub history: Vec<StepLog>,
    pub evals: Vec<EvalReport>,
    /// Dead codes restarted per level over the whole run.
    pub restarts: Vec<usize>,
    pub checkpoint: Option<PathBuf>,
    /// Wall time of the training loop, this process and any resumed-from one.
    pub elapsed_secs: f64,
    pub intent: Option<String>,
}

fn trainable_parameters<B: Backend, M: Module<B>>(module: &M) -> usize {
    struct Counter(usize);
    impl<B: Backend> ModuleVisitor<B> for Counter {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let value = param.val();
            if value.is_require_grad() {
                self.0 += value.shape().num_elements();
            }
        }
    }
    let mut counter = Counter(0);
    module.visit(&mut counter);
    counter.0
}

fn describe_eval(eval: &EvalReport) -> Vec<String> {
    let mut lines = Vec::new();
    for set in &eval.sets {
        if set.levels.is_empty() {
            lines.push(format!(
                "  {:<24} {}",
                set.name,
                set.note.as_deref().unwrap_or("no windows")
            ));
            continue;
        }
        let ce: Vec<String> = set
            .levels
            .iter()
            .map(|l| format!("{:.3}", l.recon_ce))
            .collect();
        let acc: Vec<String> = set
            .levels
            .iter()
            .map(|l| format!("{:.3}", l.token_accuracy))
            .collect();
        lines.push(format!(
            "  {:<24} ({} windows) CE@k=1..K [{}]  acc [{}]",
            set.name,
            set.windows,
            ce.join(" "),
            acc.join(" ")
        ));
    }
    let usage: Vec<String> = eval
        .usage
        .iter()
        .map(|u| format!("{}/{} ppl {:.1}", u.used, u.codebook_size, u.perplexity))
        .collect();
    lines.push(format!("  codes used per level: {}", usage.join(", ")));
    lines
}

type Optim<B> = burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, NcmTrainee<B>, B>;

struct SaveContext<'a> {
    config: &'a TokenizerTrainConfig,
    config_json: &'a serde_json::Value,
    datasets: &'a [DatasetIdentity],
}

fn save_state<B: AutodiffBackend<FloatElem = f32>>(
    ctx: &SaveContext<'_>,
    step: usize,
    trainee: &NcmTrainee<B>,
    optim: &Optim<B>,
    rng: &ChaCha12Rng,
    extras: &TokenizerExtras,
) -> Result<Option<PathBuf>> {
    let Some(dir) = &ctx.config.out_dir else {
        return Ok(None);
    };
    let model_path =
        checkpoint::save_content_addressed(trainee.tokenizer.clone(), dir, CHECKPOINT_STEM)?;
    let state_dir = TrainState::dir_for(&model_path);
    if state_dir.exists() {
        std::fs::remove_dir_all(&state_dir)
            .with_context(|| format!("clear {}", state_dir.display()))?;
    }
    let optimizer = Some(checkpoint::save_record::<B, _>(
        optim.to_record(),
        &state_dir,
        "optimizer",
    )?);
    let head = trainee
        .intent
        .as_ref()
        .map(|h| {
            checkpoint::save_record::<B, _>(h.clone().into_record(), &state_dir, "intent_head")
        })
        .transpose()?;
    let state = TrainState {
        format_version: checkpoint::STATE_FORMAT_VERSION,
        kind: STATE_KIND.into(),
        step,
        seed: ctx.config.seed,
        host_rng: serde_json::to_value(rng).context("serialize host RNG")?,
        config: ctx.config_json.clone(),
        build: checkpoint::BuildInfo::current(),
        datasets: ctx.datasets.to_vec(),
        model: checkpoint::model_entry(&model_path)?,
        optimizer,
        ema: None,
        head,
        head_optimizer: None,
        extras: serde_json::to_value(extras).context("serialize tokenizer training extras")?,
        saved_unix_secs: checkpoint::unix_now(),
    };
    state.write(&state_dir)?;
    Ok(Some(model_path))
}

/// Train the tokenizer. `reader` is required when the intent term is on and
/// ignored otherwise. Progress lines go to `progress`; nothing is printed.
pub fn train_tokenizer<B: AutodiffBackend<FloatElem = f32>>(
    config: &TokenizerTrainConfig,
    reader: Option<&dyn IntentReader<B::InnerBackend>>,
    device: &B::Device,
    progress: &mut dyn FnMut(&str),
) -> Result<TokenizerTrainReport> {
    config.validate()?;
    let intent = match (&config.intent, reader) {
        (Some(intent), Some(reader)) if intent.is_active() => Some((intent, reader)),
        (Some(intent), None) if intent.is_active() => bail!(
            "the intent term has weight {} but no reader was given; pass a nina student \
             checkpoint as the reader, or set the weight to 0",
            intent.weight
        ),
        _ => None,
    };
    let tok_config = &config.tokenizer;
    let l = tok_config.seq_len();
    let levels = tok_config.schedule.levels();
    let (train, sets) = load_data(config)?;
    let mut dataset_paths = vec![config.corpus.clone()];
    dataset_paths.extend(config.val_corpora.iter().cloned());
    let datasets = vec![DatasetIdentity::of_paths(
        &dataset_paths,
        "ncm tokenizer train + validation corpora",
    )?];
    let config_json = config.state_json()?;

    let mut rng = ChaCha12Rng::seed_from_u64(config.seed);
    B::seed(device, config.seed);
    let tokenizer = NcmTokenizer::<B>::new(tok_config, device)?;
    crate::tensor_ext::force_initialization(&tokenizer);
    let mut optim: Optim<B> = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let mut extras = TokenizerExtras {
        restarts: vec![0; levels],
        ..TokenizerExtras::default()
    };
    let mut start_step = 0usize;
    let mut trainee = match &config.resume {
        Some(resume) => {
            let state = TrainState::for_model(resume)?.with_context(|| {
                format!("{} has no training state to resume from", resume.display())
            })?;
            ensure!(
                state.kind == STATE_KIND,
                "{} was written by the '{}' trainer, not '{STATE_KIND}'",
                resume.display(),
                state.kind
            );
            let dir = TrainState::dir_for(resume);
            state.verify_files(&dir)?;
            let differences = state.config_differences(&config_json);
            ensure!(
                differences.is_empty(),
                "the resumed run's configuration differs in: {}",
                differences.join(", ")
            );
            ensure!(
                checkpoint::same_datasets(&state.datasets, &datasets),
                "the corpora changed since the checkpoint: recorded {}, now {}",
                checkpoint::describe_datasets(&state.datasets),
                checkpoint::describe_datasets(&datasets)
            );
            let tokenizer = checkpoint::load::<B, _>(tokenizer, resume, device)?;
            let head = match (&state.head, intent) {
                (Some(file), Some((_, reader))) => {
                    let fresh = IntentHead::<B>::new(tok_config.latent_dim, reader.width(), device);
                    crate::tensor_ext::force_initialization(&fresh);
                    Some(fresh.load_record(checkpoint::load_record::<B, _>(&dir, file, device)?))
                }
                (None, None) => None,
                (Some(_), None) => {
                    bail!("the checkpoint has an intent head but the intent term is off")
                }
                (None, Some(_)) => {
                    bail!("the intent term is on but the checkpoint has no intent head")
                }
            };
            if let Some(file) = &state.optimizer {
                optim = optim.load_record(checkpoint::load_record::<B, _>(&dir, file, device)?);
            }
            rng = serde_json::from_value(state.host_rng.clone()).context("restore host RNG")?;
            extras = serde_json::from_value(state.extras.clone())
                .context("parse tokenizer training extras")?;
            start_step = state.step;
            progress(&format!(
                "resumed from {} at step {start_step}",
                resume.display()
            ));
            NcmTrainee {
                tokenizer,
                intent: head,
            }
        }
        None => {
            let init_count = config
                .batch_size
                .max(
                    tok_config
                        .codebook_size
                        .div_ceil(tok_config.schedule.len_of(0).unwrap_or(1)),
                )
                .min(config.batch_size * 8);
            let windows = if config.ordered_records {
                tile_windows(&train, l, init_count)
            } else {
                draw_windows(&train, l, init_count, &mut rng)
            };
            let tokenizer = tokenizer
                .init_codebooks_from_data(tokens_tensor::<B>(&windows, l, device), &mut rng)?;
            // The head is built after the tokenizer and draws nothing from the
            // host RNG, so turning the term on never changes the tokenizer's
            // initialization or the batches it sees.
            let head = intent.map(|(_, reader)| {
                let head = IntentHead::<B>::new(tok_config.latent_dim, reader.width(), device);
                crate::tensor_ext::force_initialization(&head);
                head
            });
            NcmTrainee {
                tokenizer,
                intent: head,
            }
        }
    };
    let end_step = config.stop_at.unwrap_or(config.steps);
    ensure!(
        start_step <= end_step,
        "checkpoint step {start_step} exceeds this invocation's final step {end_step}"
    );
    let ctx = SaveContext {
        config,
        config_json: &config_json,
        datasets: &datasets,
    };
    let intent_note = intent.map(|(cfg, reader)| {
        format!(
            "λ={} over the first {} levels, {:?} loss, reader: {}",
            cfg.weight,
            cfg.levels,
            cfg.loss,
            reader.describe()
        )
    });
    progress(&format!(
        "ncm tokenizer: L={l} K={levels} d={} C={} | {} trainable params | train {} tokens | {} val set(s)",
        tok_config.latent_dim,
        tok_config.codebook_size,
        trainable_parameters::<B, _>(&trainee),
        train.len(),
        sets.len()
    ));
    if let Some(note) = &intent_note {
        progress(&format!("intent term (unvalidated): {note}"));
    }
    if start_step == 0 {
        let eval = evaluate(
            &trainee.tokenizer.valid(),
            &sets,
            config.batch_size,
            0,
            device,
        )?;
        progress("eval at step 0 (after data-seeded codebooks, before any update):");
        for line in describe_eval(&eval) {
            progress(&line);
        }
        extras.evals.push(eval);
    }
    let schedule = config.lr_schedule();
    let clock = std::time::Instant::now();
    let elapsed_before = extras.elapsed_secs;
    let mut last_checkpoint = None;
    for step in start_step..end_step {
        B::seed(device, crate::train::step_seed(config.seed, step));
        let lr = schedule.at(step);
        let mut accumulator = GradientAccumulator::new(config.accumulate);
        let mut step_stats: Option<Vec<LevelStats<B>>> = None;
        let mut pooled = Vec::new();
        let mut losses = LossParts::default();
        let mut ready = None;
        for micro in 0..config.accumulate {
            let windows = if config.ordered_records {
                let batch_index = step
                    .checked_mul(config.accumulate)
                    .and_then(|index| index.checked_add(micro))
                    .context("ordered codec batch index overflow")?;
                crate::ncm::ordered_record_offsets(train.len(), l, config.batch_size, batch_index)?
                    .map(|start| train[start..start + l].to_vec())
                    .collect()
            } else {
                draw_windows(&train, l, config.batch_size, &mut rng)
            };
            let cutoffs = draw_cutoffs(
                config.batch_size,
                levels,
                tok_config.nested_dropout,
                &mut rng,
            );
            let out = micro_step(
                &trainee,
                tokens_tensor::<B>(&windows, l, device),
                &cutoffs,
                accumulator.loss_scale(),
                intent,
            )?;
            losses.accumulate(&out.losses, accumulator.loss_scale());
            step_stats = Some(match step_stats {
                None => out.stats,
                Some(acc) => acc
                    .into_iter()
                    .zip(out.stats)
                    .map(|(a, b)| a.merge(b))
                    .collect(),
            });
            pooled = out.pooled;
            extras.windows_seen += config.batch_size;
            if let Cycle::Ready(grads) = accumulator.fold::<B, _>(out.grads, &trainee) {
                ready = grads;
            }
        }
        let mut grads = ready.context("the accumulation cycle completed without gradients")?;
        let grad_norm = global_grad_norm(&trainee, &grads);
        ensure!(
            grad_norm.is_finite(),
            "step {step}: the gradient norm is {grad_norm}; refusing to apply it"
        );
        let clip_scale = clip_gradients(&mut grads, &trainee, grad_norm, config.clip_norm as f32);
        trainee = optim.step(lr, trainee, grads);
        let NcmTrainee {
            tokenizer,
            intent: head,
        } = trainee;
        let stats = step_stats.context("a step ran no micro-batches")?;
        let mut tokenizer = tokenizer.apply_codebook_ema(&stats)?;
        if (step + 1) % tok_config.restart_every == 0 {
            let (swept, restarted) = tokenizer.restart_dead_codes(&pooled, &mut rng)?;
            tokenizer = swept;
            for (total, now) in extras.restarts.iter_mut().zip(&restarted) {
                *total += now;
            }
        }
        trainee = NcmTrainee {
            tokenizer,
            intent: head,
        };
        let done = step + 1 == end_step;
        if config.log_every > 0 && (step % config.log_every == 0 || done) {
            let log = StepLog {
                step: step + 1,
                lr,
                grad_norm: f64::from(grad_norm),
                clip_scale: f64::from(clip_scale),
                losses,
            };
            progress(&format!(
                "step {}/{} loss {:.4} ce {:.4} commit {:.4} cb {:.4}{} acc {:.3} lr {:.2e} |g| {:.3}",
                log.step,
                config.steps,
                losses.total,
                losses.recon_ce,
                losses.commitment,
                losses.codebook,
                if intent.is_some() {
                    format!(" intent {:.4}", losses.intent)
                } else {
                    String::new()
                },
                losses.accuracy,
                lr,
                grad_norm
            ));
            extras.history.push(log);
        }
        if done || (config.eval_every > 0 && (step + 1) % config.eval_every == 0) {
            let eval = evaluate(
                &trainee.tokenizer.valid(),
                &sets,
                config.batch_size,
                step + 1,
                device,
            )?;
            progress(&format!("eval at step {}:", step + 1));
            for line in describe_eval(&eval) {
                progress(&line);
            }
            extras.evals.push(eval);
        }
        if done || (config.checkpoint_every > 0 && (step + 1) % config.checkpoint_every == 0) {
            extras.elapsed_secs = elapsed_before + clock.elapsed().as_secs_f64();
            last_checkpoint = save_state(&ctx, step + 1, &trainee, &optim, &rng, &extras)?;
            if let Some(path) = &last_checkpoint {
                progress(&format!("checkpoint {}", path.display()));
            }
        }
    }
    extras.elapsed_secs = elapsed_before + clock.elapsed().as_secs_f64();
    let codebook_floats = levels * tok_config.codebook_size * (tok_config.latent_dim + 1);
    Ok(TokenizerTrainReport {
        config: tok_config.clone(),
        trainable_parameters: trainable_parameters::<B, _>(&trainee),
        codebook_floats,
        total_steps: end_step,
        resumed_from: start_step,
        train_tokens: train.len(),
        windows_seen: extras.windows_seen,
        history: extras.history,
        evals: extras.evals,
        restarts: extras.restarts,
        checkpoint: last_checkpoint,
        elapsed_secs: extras.elapsed_secs,
        intent: intent_note,
    })
}

impl TokenizerTrainReport {
    /// A plain-text summary: first and last evaluation side by side.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "NCM tokenizer: L={} K={} d={} C={} enc={} dec={} | {} trainable params, {} codebook floats\n",
            self.config.seq_len(),
            self.config.schedule.levels(),
            self.config.latent_dim,
            self.config.codebook_size,
            self.config.encoder_blocks,
            self.config.decoder_blocks,
            self.trainable_parameters,
            self.codebook_floats
        ));
        out.push_str(&format!(
            "steps {} (resumed from {}), {} windows seen, {:.1}s measured wall time\n",
            self.total_steps, self.resumed_from, self.windows_seen, self.elapsed_secs
        ));
        if let Some(note) = &self.intent {
            out.push_str(&format!("intent term (unvalidated): {note}\n"));
        }
        if let (Some(first), Some(last)) = (self.evals.first(), self.evals.last()) {
            out.push_str(&format!(
                "validation, step {} -> step {} (recon CE nats/token, token accuracy):\n",
                first.step, last.step
            ));
            for (a, b) in first.sets.iter().zip(&last.sets) {
                out.push_str(&format!("  [{}] {} windows\n", b.name, b.windows));
                if b.levels.is_empty() {
                    out.push_str(&format!(
                        "    {}\n",
                        b.note.as_deref().unwrap_or("no windows")
                    ));
                    continue;
                }
                out.push_str("    k   len   CE start -> end      acc start -> end\n");
                for (x, y) in a.levels.iter().zip(&b.levels) {
                    out.push_str(&format!(
                        "    {:<3} {:<5} {:>7.3} -> {:<7.3}   {:>6.3} -> {:<6.3}\n",
                        y.level,
                        y.length,
                        x.recon_ce,
                        y.recon_ce,
                        x.token_accuracy,
                        y.token_accuracy
                    ));
                }
            }
            out.push_str("  codebook usage on validation (start -> end):\n");
            for (x, y) in first.usage.iter().zip(&last.usage) {
                out.push_str(&format!(
                    "    level {:<3} used {:>5} -> {:<5} of {}  perplexity {:>8.1} -> {:<8.1}\n",
                    y.level, x.used, y.used, y.codebook_size, x.perplexity, y.perplexity
                ));
            }
        }
        out.push_str(&format!(
            "dead codes restarted per level: {:?}\n",
            self.restarts
        ));
        if let Some(path) = &self.checkpoint {
            out.push_str(&format!("checkpoint: {}\n", path.display()));
        }
        out
    }
}

/// Evaluate a saved tokenizer on validation corpora. The architecture comes
/// from `config`, or from the training state next to the checkpoint.
pub fn evaluate_checkpoint<B: Backend>(
    checkpoint_path: &Path,
    config: Option<NcmTokenizerConfig>,
    val_corpora: &[PathBuf],
    max_windows: usize,
    batch_size: usize,
    device: &B::Device,
) -> Result<EvalReport> {
    ensure!(
        !val_corpora.is_empty(),
        "evaluation needs at least one corpus"
    );
    ensure!(
        checkpoint_path.is_file(),
        "tokenizer checkpoint {} does not exist",
        checkpoint_path.display()
    );
    let state = TrainState::for_model(checkpoint_path)?;
    if let Some(state) = &state {
        ensure!(
            state.kind == STATE_KIND,
            "{} was written by the '{}' trainer, not '{STATE_KIND}'",
            checkpoint_path.display(),
            state.kind
        );
        state.verify_files(&TrainState::dir_for(checkpoint_path))?;
    }
    let config = match (config, &state) {
        (Some(config), _) => config,
        (None, Some(state)) => serde_json::from_value(
            state
                .config
                .get("tokenizer")
                .cloned()
                .context("the training state records no tokenizer config")?,
        )
        .context("parse the recorded tokenizer config")?,
        (None, None) => bail!(
            "{} has no training state; pass the tokenizer config explicitly",
            checkpoint_path.display()
        ),
    };
    let model = NcmTokenizer::<B>::new(&config, device)?;
    crate::tensor_ext::force_initialization(&model);
    let model = checkpoint::load::<B, _>(model, checkpoint_path, device)?;
    let l = config.seq_len();
    let mut sets = Vec::with_capacity(val_corpora.len());
    for path in val_corpora {
        let tokens = read_tokens(path)?;
        let windows = tile_windows(&tokens, l, max_windows);
        let note = windows.is_empty().then(|| {
            format!(
                "{} tokens is shorter than one {l}-token window",
                tokens.len()
            )
        });
        sets.push(ValSet {
            name: set_name(path),
            windows,
            note,
        });
    }
    let step = state.as_ref().map_or(0, |s| s.step);
    evaluate(&model, &sets, batch_size, step, device)
}

impl EvalReport {
    pub fn render(&self) -> String {
        let mut out = format!("evaluation of the step-{} tokenizer\n", self.step);
        for line in describe_eval(self) {
            out.push_str(&line);
            out.push('\n');
        }
        out
    }
}

/// Host values of every gradient of `module`, in traversal order; `None`
/// for a parameter without one.
fn grad_vectors<B: AutodiffBackend, M: AutodiffModule<B>>(
    module: &M,
    grads: &GradientsParams,
) -> Vec<Option<Vec<f32>>> {
    struct Collect<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        out: Vec<Option<Vec<f32>>>,
        _backend: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Collect<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let grad = self.grads.get::<B::InnerBackend, D>(param.id);
            self.out
                .push(grad.map(|g| g.into_data().iter::<f32>().collect()));
        }
    }
    let mut visitor = Collect::<B> {
        grads,
        out: Vec::new(),
        _backend: std::marker::PhantomData,
    };
    module.visit(&mut visitor);
    visitor.out
}

/// `max|a − b| / max|b|` over two gradient sets; infinite when one set has
/// a gradient the other lacks.
fn grad_gap(a: &[Option<Vec<f32>>], b: &[Option<Vec<f32>>]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (p, q) in a.iter().zip(b) {
        match (p, q) {
            (None, None) => {}
            (Some(p), Some(q)) if p.len() == q.len() => {
                x.extend(p.iter().map(|&v| f64::from(v)));
                y.extend(q.iter().map(|&v| f64::from(v)));
            }
            _ => return f64::INFINITY,
        }
    }
    super::relative_gap(&x, &y)
}

/// A [`ByteHistogramReader`] that counts its calls, so a check can tell
/// whether a disabled term ever ran it.
pub(crate) struct CountingReader {
    inner: ByteHistogramReader,
    reads: std::cell::Cell<usize>,
}

impl CountingReader {
    pub(crate) fn new(vocab: usize) -> Self {
        Self {
            inner: ByteHistogramReader { vocab },
            reads: std::cell::Cell::new(0),
        }
    }

    pub(crate) fn reads(&self) -> usize {
        self.reads.get()
    }
}

impl<B: Backend> IntentReader<B> for CountingReader {
    fn width(&self) -> usize {
        self.inner.vocab
    }

    fn read(&self, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>> {
        self.reads.set(self.reads.get() + 1);
        IntentReader::<B>::read(&self.inner, tokens)
    }

    fn describe(&self) -> String {
        format!("counting {}", IntentReader::<B>::describe(&self.inner))
    }
}

/// The two residuals of the "intent off" claim.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IntentOffCheck {
    /// Exact part: `|Δloss|` + reader calls + intent-head gradients at
    /// weight 0 (all must be zero); infinite if a positive weight does not
    /// read, train the head and change the loss.
    pub(crate) skipped: f64,
    /// Relative gradient gap between weight 0 and no term; infinite if a
    /// positive weight does not move the gradients past the tolerance.
    pub(crate) gradient: f64,
}

/// The intent term at weight 0 (head present, reader given) against a step
/// with no intent term at all, and at weight 0.5 to show the comparison can
/// fail.
///
/// Loss values come from the forward pass, which is bit-reproducible here,
/// so they are compared exactly; gradients come from separate backward
/// passes and are compared to [`super::CROSS_PASS_GRADIENT_TOLERANCE`].
pub(crate) fn intent_off_check() -> Result<IntentOffCheck> {
    type Ad = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
    let device = Default::default();
    let (model, tokens) = super::micro_tokenizer::<Ad>(61, 3, &device)?;
    let vocab = model.config().vocab;
    let head = IntentHead::<Ad>::new(model.config().latent_dim, vocab, &device);
    crate::tensor_ext::force_initialization(&head);
    let cutoffs = [3usize, 1, 2];
    let plain = NcmTrainee {
        tokenizer: model.clone(),
        intent: None,
    };
    let with_head = NcmTrainee {
        tokenizer: model,
        intent: Some(head),
    };
    let off = IntentConfig {
        weight: 0.0,
        levels: 2,
        loss: IntentLoss::Cosine,
        reader: None,
    };
    let on = IntentConfig {
        weight: 0.5,
        ..off.clone()
    };
    let base = micro_step(&plain, tokens.clone(), &cutoffs, 1.0, None)?;
    let zero_reader = CountingReader::new(vocab);
    let zero = micro_step(
        &with_head,
        tokens.clone(),
        &cutoffs,
        1.0,
        Some((&off, &zero_reader)),
    )?;
    let live_reader = CountingReader::new(vocab);
    let live = micro_step(&with_head, tokens, &cutoffs, 1.0, Some((&on, &live_reader)))?;
    let head = with_head
        .intent
        .as_ref()
        .context("the intent trainee has no head")?;
    let head_grads = |out: &MicroOutcome<Ad>| {
        grad_vectors::<Ad, _>(head, &out.grads)
            .iter()
            .filter(|g| g.is_some())
            .count()
    };
    let skipped = (base.losses.total - zero.losses.total).abs()
        + zero_reader.reads() as f64
        + head_grads(&zero) as f64;
    let engaged =
        live_reader.reads() > 0 && head_grads(&live) > 0 && live.losses.total != base.losses.total;
    let base_grads = grad_vectors::<Ad, _>(&plain.tokenizer, &base.grads);
    let zero_grads = grad_vectors::<Ad, _>(&with_head.tokenizer, &zero.grads);
    let live_grads = grad_vectors::<Ad, _>(&with_head.tokenizer, &live.grads);
    let moved = grad_gap(&live_grads, &base_grads) > 10.0 * super::CROSS_PASS_GRADIENT_TOLERANCE;
    Ok(IntentOffCheck {
        skipped: if engaged { skipped } else { f64::INFINITY },
        gradient: if moved {
            grad_gap(&zero_grads, &base_grads)
        } else {
            f64::INFINITY
        },
    })
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
    use crate::ncm::LevelSchedule;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Ad = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ncm-tok-{tag}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A corpus of `docs` short documents, BOS-delimited.
    fn write_corpus(dir: &Path, name: &str, docs: usize) -> PathBuf {
        let mut tokens = Vec::new();
        for i in 0..docs {
            tokens.push(crate::tokenizer::Special::Bos.id());
            for j in 0..(20 + i % 7) {
                tokens.push(u16::try_from((i * 31 + j * 7) % 256).unwrap());
            }
            tokens.push(crate::tokenizer::Special::Eos.id());
        }
        let path = dir.join(name);
        TokenCorpus::write(&path, &tokens).unwrap();
        path
    }

    fn small_config(corpus: PathBuf, out: &Path) -> TokenizerTrainConfig {
        let tokenizer = NcmTokenizerConfig {
            latent_dim: 16,
            codebook_size: 32,
            schedule: LevelSchedule::new(vec![2, 5, 16]).unwrap(),
            heads: 2,
            restart_every: 2,
            ..NcmTokenizerConfig::tiny().unwrap()
        };
        TokenizerTrainConfig {
            steps: 4,
            batch_size: 4,
            log_every: 1,
            val_windows: 4,
            out_dir: Some(out.to_path_buf()),
            ..TokenizerTrainConfig::new(tokenizer, corpus)
        }
    }

    #[test]
    fn test_tail_cut_moves_to_a_document_boundary() {
        let bos = crate::tokenizer::Special::Bos.id();
        let tokens = vec![bos, 1, 2, 3, bos, 4, 5, bos, 6, 7];
        // Nominal cut at 10 - 3 = 7, which is a BOS.
        assert_eq!(tail_cut(&tokens, 0.3), 7);
        // Nominal cut at 10 - 5 = 5 moves forward to the BOS at 7.
        assert_eq!(tail_cut(&tokens, 0.5), 7);
        // No BOS after the nominal cut: keep it.
        assert_eq!(tail_cut(&[1, 2, 3, 4], 0.5), 2);
        assert_eq!(set_name(Path::new("/x/coding.val.corpus.bin")), "coding");
        assert_eq!(set_name(Path::new("all-fields.corpus.bin")), "all-fields");
        assert_eq!(
            tile_windows(&[1, 2, 3, 4, 5], 2, 9),
            vec![vec![1, 2], vec![3, 4]]
        );
    }

    #[test]
    fn test_intent_weight_zero_skips_the_term() {
        // Independent of the process-global RNG: both sides share one
        // tokenizer and nothing in a micro-step draws random numbers, so
        // concurrent tests cannot change what is compared.
        let check = intent_off_check().unwrap();
        assert_eq!(check.skipped, 0.0);
        assert!(
            check.gradient <= crate::ncm::tokenizer::CROSS_PASS_GRADIENT_TOLERANCE / 10.0,
            "gradient gap {}",
            check.gradient
        );
    }

    #[test]
    fn test_intent_without_reader_is_refused() {
        let dir = scratch("noreader");
        let corpus = write_corpus(&dir, "c.corpus.bin", 40);
        let mut config = small_config(corpus, &dir);
        config.intent = Some(IntentConfig {
            weight: 0.1,
            levels: 1,
            loss: IntentLoss::Cosine,
            reader: None,
        });
        let err =
            train_tokenizer::<Ad>(&config, None, &Default::default(), &mut |_| {}).unwrap_err();
        assert!(format!("{err:#}").contains("no reader"), "{err:#}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_resume_restores_the_exact_state() {
        // What can be bit-exact is checked bit-exactly: the first resumed
        // step's forward pass reads the restored weights, codebook EMA
        // state, host RNG (windows and cutoffs) and LR schedule, so its
        // logged losses equal the interrupted run's at that step bit for
        // bit, and a saved checkpoint evaluates to exactly the numbers the
        // run reported. Steps after a backward pass are compared to
        // rounding: this backend's backward is not bit-reproducible
        // (see `CROSS_PASS_GRADIENT_TOLERANCE`).
        let device = Default::default();
        let dir = scratch("resume");
        let corpus = write_corpus(&dir, "c.corpus.bin", 60);
        let val = write_corpus(&dir, "v.val.corpus.bin", 12);
        let mut first = small_config(corpus.clone(), &dir.join("first"));
        first.val_corpora = vec![val.clone()];
        first.checkpoint_every = 2;
        let mut lines = Vec::new();
        let full = train_tokenizer::<Ad>(&first, None, &device, &mut |l| lines.push(l.to_string()))
            .unwrap();
        let step2 = lines
            .iter()
            .find_map(|l| l.strip_prefix("checkpoint "))
            .map(PathBuf::from)
            .unwrap();
        let mut resumed = small_config(corpus, &dir.join("resumed"));
        resumed.val_corpora = vec![val.clone()];
        resumed.resume = Some(step2);
        let after = train_tokenizer::<Ad>(&resumed, None, &device, &mut |_| {}).unwrap();
        assert_eq!(after.resumed_from, 2);
        let at = |r: &TokenizerTrainReport, step: usize| {
            r.history.iter().find(|h| h.step == step).cloned().unwrap()
        };
        // The carried history is the interrupted run's (through JSON).
        assert!((at(&full, 2).losses.total - at(&after, 2).losses.total).abs() < 1e-12);
        // First resumed step: exact.
        let (a, b) = (at(&full, 3), at(&after, 3));
        assert_eq!(a.losses, b.losses);
        assert_eq!(a.lr, b.lr);
        // After a backward pass: equal to rounding.
        let (a, b) = (at(&full, 4), at(&after, 4));
        assert!(
            (a.losses.total - b.losses.total).abs() <= 1e-4 * a.losses.total.abs(),
            "step 4 loss {} vs {}",
            a.losses.total,
            b.losses.total
        );
        // A checkpoint evaluates to exactly what its run reported.
        let eval = evaluate_checkpoint::<burn::backend::NdArray<f32>>(
            full.checkpoint.as_ref().unwrap(),
            None,
            &[val],
            4,
            4,
            &device,
        )
        .unwrap();
        assert_eq!(eval.step, 4);
        assert_eq!(&eval, full.evals.last().unwrap());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_training_lowers_reconstruction_error() {
        let device = Default::default();
        let dir = scratch("learn");
        let corpus = write_corpus(&dir, "c.corpus.bin", 80);
        let mut config = small_config(corpus, &dir);
        config.steps = 40;
        config.out_dir = None;
        let report = train_tokenizer::<Ad>(&config, None, &device, &mut |_| {}).unwrap();
        let first = report.evals.first().unwrap();
        let last = report.evals.last().unwrap();
        let ce = |e: &EvalReport| e.sets[0].levels.last().unwrap().recon_ce;
        assert!(ce(last) < ce(first), "CE {} -> {}", ce(first), ce(last));
        assert!(report.render().contains("codebook usage"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
