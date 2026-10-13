//! The learned half of reward-integrity scanning: a small trainable classifier
//! that scores a code snippet for cheat likelihood from raw text features.
//!
//! # Why this exists alongside `cheat.rs`
//!
//! [`crate::cheat`] is a *pattern* scanner: it fires on the idioms it was
//! taught (`#[allow]` with no reason, `todo!()`, `type: ignore`, a key
//! prefix). A teacher that cheats in a slightly new idiom — a suppression
//! spelled differently, a stub behind an unfamiliar macro — produces no
//! finding, and the trace sails through. This module is the generalization
//! layer: logistic regression over hashed char-ngram and word features,
//! trained on the pattern scanner's verdicts, so it can score snippets whose
//! patterns do not fire.
//!
//! # The circularity, stated plainly
//!
//! The training labels **are** the pattern scanner's verdicts (kept vs.
//! rejected traces from the data pipeline, plus synthetic seeded pairs built
//! from the scanner's own fixture style). This model therefore learns to
//! *generalize* those labels; it does not invent new ground truth, and it
//! cannot discover a cheat class the patterns have never seen. With the real
//! labeled set at ~50 samples this is a mechanism, not a detector you would
//! trust alone — every report carries the label source and the sample counts
//! so that fact is never hidden. Treat the score as a second opinion that
//! improves with data volume, and see the stacking rule below for which
//! opinion wins.
//!
//! # Stacking order: deterministic rules outrank the learned score, always
//!
//! [`filter_traces_ml`] runs the pattern scan first. Any finding at
//! quarantine grade or above ([`Severity::Evasion`] and up) rejects the trace
//! regardless of the ML score. Only when the pattern scan is silent does the
//! learned score decide, against a caller-chosen threshold. A learned model
//! can be wrong in both directions; a reject-grade pattern finding is a
//! stated rule with evidence, so it always wins.
//!
//! # Size honesty
//!
//! This is spam-filter math, deliberately: a linear model over hashed
//! features is the honest capacity for tens of labeled samples, trains in
//! microseconds on CPU, serializes to plain JSON, and has no tensor
//! dependency. Scaling it up before the data justifies it would be the same
//! overfitting this module exists to catch.

use anyhow::{bail, Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::cheat::{scan_snippet, Severity, SnippetLanguage};

/// Where every training label comes from. Recorded in every [`TrainReport`],
/// [`EvalReport`] and saved [`MlCheatModel`], so a persisted artifact always
/// states what its labels mean.
pub const LABEL_SOURCE: &str = "cheat.rs pattern-scanner verdicts (scan_snippet / filter_traces); \
    the ML layer generalizes those labels to snippets whose patterns do not fire — \
    it does not invent new ground truth";

/// Current on-disk format version of a saved [`MlCheatModel`].
pub const MODEL_VERSION: u32 = 1;

/// Number of [`SnippetLanguage`] variants; the feature space is partitioned
/// into one block per language plus an indicator block plus a shared block.
const NUM_LANGS: u32 = 7;

/// Stable feature-block index per language. A match, not `as`, so adding a
/// variant to [`SnippetLanguage`] fails to compile here rather than silently
/// remapping the feature space of every saved model.
fn lang_index(lang: SnippetLanguage) -> u32 {
    match lang {
        SnippetLanguage::Rust => 0,
        SnippetLanguage::Python => 1,
        SnippetLanguage::C => 2,
        SnippetLanguage::Cpp => 3,
        SnippetLanguage::JsTs => 4,
        SnippetLanguage::HtmlCss => 5,
        SnippetLanguage::Other => 6,
    }
}

// FNV-1a 64-bit with the fixed offset basis and prime — the same fixed,
// seedless contract `cheat.rs` uses for shingling. `DefaultHasher` is
// deterministic within a build but is not a stable contract; a saved model's
// features must hash identically in every process on every run, so the hasher
// is spelled out here.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Incremental FNV-1a so n-grams hash without an intermediate allocation.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(FNV_OFFSET)
    }

    fn byte(&mut self, b: u8) {
        self.0 ^= u64::from(b);
        self.0 = self.0.wrapping_mul(FNV_PRIME);
    }

    fn bytes(&mut self, bs: &[u8]) {
        for b in bs {
            self.byte(*b);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

/// Hash of a char n-gram, tagged with its width so a 3-gram and a 4-gram over
/// the same prefix never collide in the same bin.
fn hash_char_ngram(width: usize, gram: &[char]) -> u64 {
    let mut h = Fnv::new();
    h.byte(0xC1);
    h.byte(width as u8); // widths are validated to 2..=8, so the cast is lossless
    for c in gram {
        h.bytes(&u32::from(*c).to_le_bytes());
    }
    h.finish()
}

/// Hash of a word token, tagged separately from char n-grams.
fn hash_word(token: &str) -> u64 {
    let mut h = Fnv::new();
    h.byte(0x57); // 'W': the word-token tag
    h.bytes(token.as_bytes());
    h.finish()
}

/// Configuration for [`MlCheatModel`]. Every field has a serde default, so a
/// config or model sidecar written by an older build still parses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MlCheatConfig {
    /// Char n-gram widths, each in `2..=8`. `3..=5` catches idioms
    /// (`todo!(`, `noqa`, `allow(`) without shattering on whitespace.
    pub char_widths: Vec<usize>,
    /// Whether to add word unigram features alongside the char n-grams.
    pub word_unigrams: bool,
    /// Log2 of the hash-bin count per language block. Total weight count is
    /// `(NUM_LANGS + 2) << hash_bits`. Validated to `4..=20`.
    pub hash_bits: u32,
    /// SGD learning rate (constant; the dataset this trains on is tens of
    /// samples, so a schedule would be decoration).
    pub lr: f32,
    /// Training epochs over the sample set.
    pub epochs: usize,
    /// L2 penalty, applied lazily to the weights a sample touches (the
    /// standard sparse-update approximation; with tens of features per
    /// sample the exact dense penalty would dominate the step).
    pub l2: f32,
    /// Seed for the per-epoch shuffle order (ChaCha12, same discipline as
    /// `train.rs`).
    pub seed: u64,
}

impl Default for MlCheatConfig {
    fn default() -> Self {
        Self {
            char_widths: vec![3, 4, 5],
            word_unigrams: true,
            hash_bits: 14,
            lr: 0.5,
            epochs: 200,
            l2: 1e-4,
            seed: 0x5eed_0001,
        }
    }
}

impl MlCheatConfig {
    /// The weight-vector dimension this config implies.
    fn dim(&self) -> usize {
        (NUM_LANGS as usize + 2) << self.hash_bits
    }

    fn validate(&self) -> Result<()> {
        if !(4..=20).contains(&self.hash_bits) {
            bail!(
                "hash_bits must be in 4..=20 (got {}): below 4 the bins collapse, above 20 the weight vector exceeds 32M floats",
                self.hash_bits
            );
        }
        if self.char_widths.is_empty() {
            bail!("char_widths must not be empty: the char n-grams are the primary feature");
        }
        for w in &self.char_widths {
            if !(2..=8).contains(w) {
                bail!("char n-gram width must be in 2..=8 (got {w})");
            }
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            bail!("lr must be a positive finite number (got {})", self.lr);
        }
        if self.epochs == 0 {
            bail!("epochs must be at least 1");
        }
        if !self.l2.is_finite() || self.l2 < 0.0 {
            bail!("l2 must be a non-negative finite number (got {})", self.l2);
        }
        Ok(())
    }
}

/// Sparse hashed feature vector: `(index, value)` pairs, sorted by index with
/// duplicates merged, L2-normalized to unit length.
///
/// The normalization is the document-length correction: without it a long
/// snippet would score "more of everything" and length itself would leak into
/// the verdict. Unit L2 norm makes the score a function of the snippet's
/// *composition*, not its size.
///
/// Layout: block `0` of `1 << hash_bits` entries holds a one-hot language
/// indicator at index `lang_index(lang)`; block `1 + lang_index(lang)` holds
/// the snippet's content features *for that language*; the final block
/// (`NUM_LANGS + 1`) holds the same content features again, shared across all
/// languages. The per-language blocks are disjoint by construction — the
/// suppression idioms differ per language family — while the shared block is
/// what lets a cheat idiom learned in one language score a snippet in another
/// at this data volume (tens of samples per language is not enough for seven
/// independent models). Fully deterministic: fixed FNV-1a, no `RandomState`,
/// no iteration-order dependence.
pub fn featurize(code: &str, lang: SnippetLanguage, cfg: &MlCheatConfig) -> Vec<(u32, f32)> {
    let bins = 1u32 << cfg.hash_bits;
    let mask = u64::from(bins - 1);
    let li = lang_index(lang);
    let content_base = (1 + li) * bins;
    let shared_base = (1 + NUM_LANGS) * bins;

    let mut feats: Vec<(u32, f32)> = Vec::new();
    // Language indicator, block 0.
    feats.push((li, 1.0));

    let mut push_content = |h: u64| {
        let bin = (h & mask) as u32;
        feats.push((content_base + bin, 1.0));
        feats.push((shared_base + bin, 1.0));
    };

    let lower = code.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    for &width in &cfg.char_widths {
        if chars.len() < width {
            continue;
        }
        for gram in chars.windows(width) {
            push_content(hash_char_ngram(width, gram));
        }
    }
    if cfg.word_unigrams {
        for token in lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
        {
            push_content(hash_word(token));
        }
    }

    feats.sort_unstable_by_key(|f| f.0);
    let mut merged: Vec<(u32, f32)> = Vec::with_capacity(feats.len());
    for (idx, v) in feats {
        match merged.last_mut() {
            Some(last) if last.0 == idx => last.1 += v,
            _ => merged.push((idx, v)),
        }
    }
    let norm = merged.iter().map(|f| f.1 * f.1).sum::<f32>().sqrt();
    if norm > 0.0 {
        for f in &mut merged {
            f.1 /= norm;
        }
    }
    merged
}

/// Numerically stable logistic sigmoid.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// What one training run did. The label source is recorded on the report
/// itself, because a number without its provenance is how a circular metric
/// gets trusted as an independent one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainReport {
    /// Always [`LABEL_SOURCE`].
    pub label_source: String,
    /// Samples trained on.
    pub samples: usize,
    /// How many were labeled cheat (`true`).
    pub positives: usize,
    /// How many were labeled clean (`false`).
    pub negatives: usize,
    /// Mean binary cross-entropy after each epoch, in order.
    pub loss_curve: Vec<f64>,
    /// Accuracy on the training set at threshold 0.5 after the final epoch.
    /// This is a convergence check, not a generalization claim — use
    /// [`evaluate`] on held-out data for that.
    pub final_train_accuracy: f64,
}

/// Metrics on a labeled held-out set. `precision`/`recall` are for the
/// positive (cheat) class; `auc` is the rank statistic from [`auc_rank`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Always [`LABEL_SOURCE`], copied from the model.
    pub label_source: String,
    /// Samples evaluated.
    pub samples: usize,
    /// Ground-truth positives in the set.
    pub positives: usize,
    /// Ground-truth negatives in the set.
    pub negatives: usize,
    /// Decision threshold used for accuracy/precision/recall (0.5).
    pub threshold: f64,
    pub accuracy: f64,
    /// `tp / (tp + fp)`; defined as 0.0 when nothing is predicted positive.
    pub precision: f64,
    /// `tp / (tp + fn)`; defined as 0.0 when the set has no positives.
    pub recall: f64,
    /// Probability a random positive scores above a random negative.
    /// `NaN` when the set is single-class (the statistic is undefined there,
    /// and a fake number would be worse than none).
    pub auc: f64,
}

/// A logistic-regression cheat scorer. Weights are one `f32` per hash bin
/// across all language blocks, plus a bias. Deliberately dependency-light: no
/// tensors, no GPU, no Burn — the data volume does not justify more.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlCheatModel {
    /// On-disk format version; see [`MODEL_VERSION`].
    pub version: u32,
    /// The config the feature space was built from. Loading validates that
    /// the weight count matches it.
    pub config: MlCheatConfig,
    weights: Vec<f32>,
    bias: f32,
    /// Always [`LABEL_SOURCE`] after [`MlCheatModel::train`]; empty for a
    /// freshly constructed, untrained model.
    #[serde(default)]
    pub label_source: String,
    /// How many samples the last training run saw (0 = untrained).
    #[serde(default)]
    pub trained_samples: usize,
}

impl MlCheatModel {
    /// Construct an untrained model, validating the config.
    pub fn new(config: MlCheatConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            version: MODEL_VERSION,
            weights: vec![0.0; config.dim()],
            bias: 0.0,
            label_source: String::new(),
            trained_samples: 0,
            config,
        })
    }

    fn logit(&self, feats: &[(u32, f32)]) -> f64 {
        let mut z = f64::from(self.bias);
        for &(idx, v) in feats {
            // Every index featurize emits is below config.dim(), and load
            // validates the weight count against the config, so this index is
            // in range by construction.
            z += f64::from(self.weights[idx as usize]) * f64::from(v);
        }
        z
    }

    /// Probability that `code` is a cheat, in `[0, 1]`.
    pub fn score(&self, code: &str, lang: SnippetLanguage) -> f64 {
        sigmoid(self.logit(&featurize(code, lang, &self.config)))
    }

    /// Train in place on labeled samples: `(code, language, is_cheat)`.
    ///
    /// Hand-rolled SGD over the sparse features with a ChaCha12-seeded
    /// shuffle per epoch, so a rerun with the same config and samples
    /// reproduces the same weights bit for bit.
    pub fn train(&mut self, samples: &[(String, SnippetLanguage, bool)]) -> Result<TrainReport> {
        if samples.is_empty() {
            bail!("cannot train an mlcheat model on zero samples");
        }
        let feats: Vec<Vec<(u32, f32)>> = samples
            .iter()
            .map(|(code, lang, _)| featurize(code, *lang, &self.config))
            .collect();
        let lr = self.config.lr;
        let l2 = self.config.l2;
        let mut rng = ChaCha12Rng::seed_from_u64(self.config.seed);
        let mut order: Vec<usize> = (0..samples.len()).collect();
        let mut loss_curve = Vec::with_capacity(self.config.epochs);

        for _epoch in 0..self.config.epochs {
            order.shuffle(&mut rng);
            let mut loss_sum = 0.0f64;
            for &i in &order {
                let y = if samples[i].2 { 1.0f32 } else { 0.0f32 };
                let mut z = self.bias;
                for &(idx, v) in &feats[i] {
                    z += self.weights[idx as usize] * v;
                }
                let p = sigmoid(f64::from(z)) as f32;
                let grad = p - y;
                for &(idx, v) in &feats[i] {
                    let w = &mut self.weights[idx as usize];
                    *w -= lr * (grad * v + l2 * *w);
                }
                self.bias -= lr * grad;
                let pc = f64::from(p).clamp(1e-7, 1.0 - 1e-7);
                loss_sum -= f64::from(y) * pc.ln() + f64::from(1.0 - y) * (1.0 - pc).ln();
            }
            loss_curve.push(loss_sum / samples.len() as f64);
        }

        let correct = samples
            .iter()
            .zip(&feats)
            .filter(|((_, _, y), f)| (self.logit(f) >= 0.0) == *y)
            .count();
        self.label_source = LABEL_SOURCE.to_string();
        self.trained_samples = samples.len();
        Ok(TrainReport {
            label_source: LABEL_SOURCE.to_string(),
            samples: samples.len(),
            positives: samples.iter().filter(|s| s.2).count(),
            negatives: samples.iter().filter(|s| !s.2).count(),
            loss_curve,
            final_train_accuracy: correct as f64 / samples.len() as f64,
        })
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize mlcheat model")
    }

    /// Parse from JSON, rejecting a newer format version and any config whose
    /// weight count does not match the vector on disk.
    pub fn from_json(text: &str) -> Result<Self> {
        let model: Self = serde_json::from_str(text).context("parse mlcheat model JSON")?;
        if model.version > MODEL_VERSION {
            bail!(
                "mlcheat model version {} is newer than this build understands ({MODEL_VERSION})",
                model.version
            );
        }
        model.config.validate()?;
        let want = model.config.dim();
        if model.weights.len() != want {
            bail!(
                "mlcheat model has {} weights but its config implies {want} — the file and its config disagree",
                model.weights.len()
            );
        }
        Ok(model)
    }

    /// Save to `path` as JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_json()?)
            .with_context(|| format!("write mlcheat model to {}", path.display()))
    }

    /// Load from a JSON file written by [`MlCheatModel::save`].
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read mlcheat model from {}", path.display()))?;
        Self::from_json(&text)
    }
}

/// Area under the ROC curve as the Mann–Whitney rank statistic with
/// average ranks for ties: the probability that a uniformly chosen positive
/// outranks a uniformly chosen negative (ties count half). Hand-rolled on
/// purpose — it is fifteen lines and importing a stats crate for it would be
/// the dependency version of overfitting.
///
/// Returns `NaN` when either class is absent: the statistic is undefined
/// there and a placeholder number would read as a measurement.
///
/// Panics by assertion if the slices differ in length — a caller that
/// misaligns scores and labels has a bug, not a data point.
pub fn auc_rank(scores: &[f64], labels: &[bool]) -> f64 {
    assert_eq!(
        scores.len(),
        labels.len(),
        "auc_rank: scores and labels must have the same length"
    );
    let n_pos = labels.iter().filter(|&&l| l).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    let mut ranked: Vec<(f64, bool)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0usize;
    while i < ranked.len() {
        let mut j = i + 1;
        while j < ranked.len() && ranked[j].0 == ranked[i].0 {
            j += 1;
        }
        // Ranks are 1-based: this run occupies ranks i+1..=j, average (i+1+j)/2.
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        let pos_in_run = ranked[i..j].iter().filter(|r| r.1).count();
        rank_sum_pos += avg_rank * pos_in_run as f64;
        i = j;
    }
    let (n_pos, n_neg) = (n_pos as f64, n_neg as f64);
    (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Evaluate a trained model on a labeled held-out set at threshold 0.5.
pub fn evaluate(
    model: &MlCheatModel,
    held_out: &[(String, SnippetLanguage, bool)],
) -> Result<EvalReport> {
    if held_out.is_empty() {
        bail!("cannot evaluate an mlcheat model on zero samples");
    }
    let scores: Vec<f64> = held_out
        .iter()
        .map(|(code, lang, _)| model.score(code, *lang))
        .collect();
    let labels: Vec<bool> = held_out.iter().map(|s| s.2).collect();
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut correct = 0usize;
    for (score, label) in scores.iter().zip(&labels) {
        let predicted = *score >= 0.5;
        match (predicted, label) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => {}
        }
        if predicted == *label {
            correct += 1;
        }
    }
    let positives = tp + fn_;
    let negatives = held_out.len() - positives;
    Ok(EvalReport {
        label_source: model.label_source.clone(),
        samples: held_out.len(),
        positives,
        negatives,
        threshold: 0.5,
        accuracy: correct as f64 / held_out.len() as f64,
        precision: if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        },
        recall: if positives > 0 {
            tp as f64 / positives as f64
        } else {
            0.0
        },
        auc: auc_rank(&scores, &labels),
    })
}

/// Batch filter for the data pipeline, stacking the pattern scanner and the
/// learned score — the ML mirror of [`crate::cheat::filter_traces`].
///
/// Stacking order (the deterministic rules always win):
///
/// 1. The pattern scan runs first. Any finding at [`Severity::Evasion`] or
///    above rejects the trace outright, whatever the ML score says.
/// 2. Otherwise the ML score decides: `score >= threshold` rejects.
///
/// Returns the kept indices and every trace's ML score (so the caller can log
/// the margins, not just the verdict).
pub fn filter_traces_ml(
    texts: &[String],
    lang: SnippetLanguage,
    model: &MlCheatModel,
    threshold: f64,
) -> (Vec<usize>, Vec<f64>) {
    let mut kept = Vec::new();
    let mut scores = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
        let pattern_reject = scan_snippet(text, lang)
            .findings
            .iter()
            .any(|f| f.severity >= Severity::Evasion);
        let score = model.score(text, lang);
        scores.push(score);
        if !pattern_reject && score < threshold {
            kept.push(i);
        }
    }
    (kept, scores)
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

    /// One labeled sample: `(code, language, is_cheat)`.
    type Samples = Vec<(String, SnippetLanguage, bool)>;

    /// Synthetic separable pairs: for every (language, class) cell there is
    /// one train string and one held-out string built from the same idiom
    /// with renamed identifiers, so a correct featurizer can separate them
    /// by construction. Secret-shaped values are built with `repeat` so no
    /// literal token shape sits in this file.
    fn synthetic_pairs() -> (Samples, Samples) {
        let train: &[(String, SnippetLanguage, bool)] = &[
            ("pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n".to_string(), SnippetLanguage::Rust, false),
            ("fn running_total(xs: &[i64]) -> Vec<i64> {\n    let mut sum = 0;\n    xs.iter().map(|x| { sum += x; sum }).collect()\n}\n".to_string(), SnippetLanguage::Rust, false),
            ("#[allow(dead_code)]\nfn f() {}\n".to_string(), SnippetLanguage::Rust, true),
            ("fn compute(x: i32) -> i32 {\n    todo!()\n}\n".to_string(), SnippetLanguage::Rust, true),
            ("def add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(1, 2) == 3\n".to_string(), SnippetLanguage::Python, false),
            ("def word_frequency(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n".to_string(), SnippetLanguage::Python, false),
            ("x = compute()  # type: ignore\n".to_string(), SnippetLanguage::Python, true),
            ("def f():\n    raise NotImplementedError\n".to_string(), SnippetLanguage::Python, true),
            ("function add(a, b) {\n  return a + b;\n}\n\ntest('add', () => {\n  expect(add(1, 2)).toBe(3);\n});\n".to_string(), SnippetLanguage::JsTs, false),
            ("// eslint-disable-next-line no-unused-vars\nfoo();\n".to_string(), SnippetLanguage::JsTs, true),
            ("function f() {\n  throw new Error('not implemented');\n}\n".to_string(), SnippetLanguage::JsTs, true),
            ("int add(int a, int b) { return a + b; }\n".to_string(), SnippetLanguage::C, false),
            ("int f(void) { return g(); }  // NOLINT\n".to_string(), SnippetLanguage::C, true),
            ("std::string greet(const std::string& name) { return \"hi \" + name; }\n".to_string(), SnippetLanguage::Cpp, false),
            ("<div class=\"note\">hello</div>\n".to_string(), SnippetLanguage::HtmlCss, false),
            (format!("let key = \"nvapi-{}\";\n", "F".repeat(30)), SnippetLanguage::Rust, true),
        ];
        let held_out: &[(String, SnippetLanguage, bool)] = &[
            ("pub fn sum(x: i64, y: i64) -> i64 {\n    x + y\n}\n".to_string(), SnippetLanguage::Rust, false),
            ("fn first<'a>(xs: &'a [i32]) -> Option<&'a i32> {\n    xs.first()\n}\n".to_string(), SnippetLanguage::Rust, false),
            ("#[allow(unused_variables)]\nfn g() {}\n".to_string(), SnippetLanguage::Rust, true),
            ("fn solve(n: i64) -> i64 {\n    unimplemented!()\n}\n".to_string(), SnippetLanguage::Rust, true),
            ("def multiply(a, b):\n    return a * b\n".to_string(), SnippetLanguage::Python, false),
            ("def median(values):\n    values = sorted(values)\n    n = len(values)\n    return values[n // 2]\n".to_string(), SnippetLanguage::Python, false),
            ("result = parse(data)  # noqa\n".to_string(), SnippetLanguage::Python, true),
            ("def g(x):\n    pass  # TODO\n".to_string(), SnippetLanguage::Python, true),
            ("const mul = (a, b) => a * b;\n".to_string(), SnippetLanguage::JsTs, false),
            ("// @ts-nocheck\nconst x = y;\n".to_string(), SnippetLanguage::JsTs, true),
            ("function g() {\n  throw new Error('not implemented yet');\n}\n".to_string(), SnippetLanguage::JsTs, true),
            ("int sub(int a, int b) { return a - b; }\n".to_string(), SnippetLanguage::C, false),
            ("int h(void) { return q(); }  // NOLINT(readability)\n".to_string(), SnippetLanguage::C, true),
            ("std::string shout(const std::string& word) { return word + \"!\"; }\n".to_string(), SnippetLanguage::Cpp, false),
            ("<span class=\"tag\">ok</span>\n".to_string(), SnippetLanguage::HtmlCss, false),
            (format!("token = \"ghp_{}\"\n", "G".repeat(36)), SnippetLanguage::Python, true),
        ];
        (train.to_vec(), held_out.to_vec())
    }

    #[test]
    fn featurize_is_deterministic() {
        let cfg = MlCheatConfig::default();
        let a = featurize("fn f() { todo!() }", SnippetLanguage::Rust, &cfg);
        let b = featurize("fn f() { todo!() }", SnippetLanguage::Rust, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn featurize_normalizes_to_unit_l2() {
        let cfg = MlCheatConfig::default();
        let feats = featurize(
            "def word_frequency(text):\n    return {}\n",
            SnippetLanguage::Python,
            &cfg,
        );
        let norm: f32 = feats.iter().map(|f| f.1 * f.1).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn language_blocks_are_disjoint() {
        let cfg = MlCheatConfig::default();
        let bins = 1u32 << cfg.hash_bits;
        let shared_base = (1 + NUM_LANGS) * bins;
        let code = "fn f(x: i32) -> i32 { x + 1 }";
        // Per-language content features only: drop the block-0 language
        // indicator (different per language by construction) and the shared
        // block (identical per language by construction).
        let per_lang = |lang: SnippetLanguage| -> std::collections::HashSet<u32> {
            featurize(code, lang, &cfg)
                .into_iter()
                .map(|f| f.0)
                .filter(|i| *i >= bins && *i < shared_base)
                .collect()
        };
        let rust = per_lang(SnippetLanguage::Rust);
        let python = per_lang(SnippetLanguage::Python);
        let shared: Vec<u32> = rust.intersection(&python).copied().collect();
        assert!(
            shared.is_empty(),
            "per-language feature blocks overlap at indices {shared:?}"
        );
        // Sanity: both languages did emit content into their own block.
        assert!(!rust.is_empty() && !python.is_empty());
        // And the shared block sees the same content regardless of language.
        let shared_block = |lang: SnippetLanguage| -> std::collections::HashSet<u32> {
            featurize(code, lang, &cfg)
                .into_iter()
                .map(|f| f.0)
                .filter(|i| *i >= shared_base)
                .collect()
        };
        assert_eq!(
            shared_block(SnippetLanguage::Rust),
            shared_block(SnippetLanguage::Python),
            "the shared block must be language-independent"
        );
    }

    #[test]
    fn a_bad_config_is_rejected_by_name() {
        let cfg = MlCheatConfig {
            hash_bits: 2,
            ..MlCheatConfig::default()
        };
        let err = MlCheatModel::new(cfg).unwrap_err().to_string();
        assert!(err.contains("hash_bits"), "unexpected error: {err}");
        let cfg = MlCheatConfig {
            char_widths: Vec::new(),
            ..MlCheatConfig::default()
        };
        let err = MlCheatModel::new(cfg).unwrap_err().to_string();
        assert!(err.contains("char_widths"), "unexpected error: {err}");
    }

    #[test]
    fn training_on_separable_data_reaches_high_held_out_accuracy() {
        // Every (language, class) cell has a train string and a held-out
        // string built from the same idiom, so this is separable by
        // construction. If the accuracy is not near-perfect, the features
        // are broken — that is what this test is for.
        let (train, held_out) = synthetic_pairs();
        assert!(train.iter().any(|s| s.2) && train.iter().any(|s| !s.2));
        assert!(held_out.iter().any(|s| s.2) && held_out.iter().any(|s| !s.2));

        let mut model = MlCheatModel::new(MlCheatConfig::default()).expect("config");
        let report = model.train(&train).expect("train");
        assert_eq!(report.samples, train.len());
        assert_eq!(report.label_source, LABEL_SOURCE);
        assert!(
            report.loss_curve.windows(2).all(|w| w[1] <= w[0] + 1e-6),
            "loss did not decrease monotonically on separable data: {:?}",
            report.loss_curve
        );
        assert!(report.final_train_accuracy >= 0.99);

        let eval = evaluate(&model, &held_out).expect("evaluate");
        eprintln!(
            "synthetic held-out: accuracy={:.3} precision={:.3} recall={:.3} auc={:.3}",
            eval.accuracy, eval.precision, eval.recall, eval.auc
        );
        assert!(
            eval.accuracy >= 0.9,
            "separable data scored {:.3} — the features are broken",
            eval.accuracy
        );
    }

    #[test]
    fn save_load_round_trip_gives_identical_scores() {
        let (mut samples, held_out) = synthetic_pairs();
        samples.extend(held_out);
        let mut model = MlCheatModel::new(MlCheatConfig::default()).expect("config");
        model.train(&samples).expect("train");
        let probe = "#[allow(dead_code)]\nfn f() {}\n";
        let before = model.score(probe, SnippetLanguage::Rust);

        let path =
            std::env::temp_dir().join(format!("mlcheat-roundtrip-{}-1.json", std::process::id()));
        model.save(&path).expect("save");
        let loaded = MlCheatModel::load(&path).expect("load");
        // The temp file's only purpose was this round trip; its removal
        // failing (e.g. a read-only /tmp after the write) changes nothing.
        drop(std::fs::remove_file(&path));
        let after = loaded.score(probe, SnippetLanguage::Rust);
        assert_eq!(before.to_bits(), after.to_bits());

        let json = model.to_json().expect("json");
        let back = MlCheatModel::from_json(&json).expect("parse");
        assert_eq!(back.weights, model.weights);
        assert_eq!(back.label_source, LABEL_SOURCE);
    }

    #[test]
    fn a_newer_model_version_is_rejected() {
        let model = MlCheatModel::new(MlCheatConfig::default()).expect("config");
        let mut value: serde_json::Value =
            serde_json::from_str(&model.to_json().expect("json")).expect("parse own json");
        value["version"] = serde_json::json!(MODEL_VERSION + 1);
        let err = MlCheatModel::from_json(&value.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("newer"), "unexpected error: {err}");
    }

    #[test]
    fn auc_is_one_on_perfect_separation() {
        let scores = [0.9, 0.1, 0.8, 0.2];
        let labels = [true, false, true, false];
        assert_eq!(auc_rank(&scores, &labels), 1.0);
    }

    #[test]
    fn auc_is_a_half_on_alternating_scores() {
        // Each positive is matched by an equally scored negative: the rank
        // statistic must land exactly on chance.
        let scores = [0.9, 0.1, 0.9, 0.1];
        let labels = [true, false, false, true];
        assert_eq!(auc_rank(&scores, &labels), 0.5);
    }

    #[test]
    fn auc_is_nan_on_a_single_class_set() {
        assert!(auc_rank(&[0.1, 0.2], &[true, true]).is_nan());
        assert!(auc_rank(&[0.1, 0.2], &[false, false]).is_nan());
    }

    #[test]
    fn pattern_reject_grade_wins_even_when_the_ml_score_is_low() {
        // An untrained model scores everything 0.5 (zero weights), so with a
        // 0.9 threshold the ML layer alone would keep the suppression snippet.
        // The combined filter must still reject it: deterministic rules
        // outrank the learned score, always.
        let model = MlCheatModel::new(MlCheatConfig::default()).expect("config");
        let traces = vec![
            "#[allow(dead_code)]\nfn f() {}\n".to_string(),
            "fn f() -> i32 {\n    todo!()\n}\n".to_string(),
            "pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n".to_string(),
        ];
        let (kept, scores) = filter_traces_ml(&traces, SnippetLanguage::Rust, &model, 0.9);
        assert_eq!(scores.len(), 3);
        assert!(
            scores.iter().all(|s| (0.0..=1.0).contains(s)),
            "scores out of range: {scores:?}"
        );
        assert!(
            scores.iter().all(|s| *s < 0.9),
            "test premise broken: an ML score was not low: {scores:?}"
        );
        assert_eq!(
            kept,
            vec![2],
            "the pattern-reject traces must be dropped despite low ML scores"
        );
    }

    #[test]
    fn ml_layer_rejects_what_the_patterns_miss_at_a_low_threshold() {
        // Train on synthetic data, then hand the filter a snippet whose
        // patterns do not fire but whose n-grams look like the bad class.
        let (mut samples, held_out) = synthetic_pairs();
        samples.extend(held_out);
        let mut model = MlCheatModel::new(MlCheatConfig::default()).expect("config");
        model.train(&samples).expect("train");
        // A `todo!()` spelled inside a string literal fires no pattern — the
        // scanner's code view blanks string contents — but the learned
        // featurizer reads raw text, so its n-grams still light up the
        // weights the stub fixtures trained. This is exactly the gap the ML
        // layer exists to score.
        let traces = vec![
            "fn compute(x: i32) -> i32 {\n    let marker = \"todo!()\";\n    x\n}\n".to_string(),
            "pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n".to_string(),
        ];
        assert!(scan_snippet(&traces[0], SnippetLanguage::Rust).is_clean());
        let (kept, scores) = filter_traces_ml(&traces, SnippetLanguage::Rust, &model, 0.5);
        assert!(
            scores[0] > scores[1],
            "stub-like snippet should outscore clean code: {scores:?}"
        );
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn real_trace_smoke_runs_and_metrics_stay_in_bounds_no_quality_claim() {
        // Smoke test over the real NIM run's kept/rejected traces. It asserts
        // only that the machinery runs end to end on real data and that every
        // metric is a number in range — with ~50 samples, split across seven
        // language blocks, no quality claim is being made or tested here.
        let root = Path::new("/srv/m-sda/unifur/teacher-traces/pipeline-real");
        if !root.is_dir() {
            eprintln!("skipping: {} is absent", root.display());
            return;
        }
        let mut samples: Vec<(String, SnippetLanguage, bool)> = Vec::new();
        let mut dirs: Vec<_> = std::fs::read_dir(root)
            .expect("read trace root")
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for dir in dirs {
            let name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .expect("dir name")
                .to_string();
            // The pipeline's "web" family is the scanner's HtmlCss.
            let lang = if name == "web" {
                SnippetLanguage::HtmlCss
            } else {
                SnippetLanguage::from_name(&name)
            };
            let train_path = dir.join("train.jsonl");
            if let Ok(text) = std::fs::read_to_string(&train_path) {
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let rec: crate::langdata::TrainRecord =
                        serde_json::from_str(line).expect("train record");
                    samples.push((rec.completion, lang, false));
                }
            }
            let rejects_path = dir.join("rejects.jsonl");
            if let Ok(text) = std::fs::read_to_string(&rejects_path) {
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let rec: crate::langdata::RejectRecord =
                        serde_json::from_str(line).expect("reject record");
                    // Only `cheat`-stage rejects carry the pattern scanner's
                    // verdict; a decontamination drop is not a cheat label.
                    if rec.stage == "cheat" {
                        samples.push((rec.completion, lang, true));
                    }
                }
            }
        }
        if samples.len() < 4 {
            eprintln!("skipping: only {} real samples found", samples.len());
            return;
        }
        let positives = samples.iter().filter(|s| s.2).count();
        eprintln!(
            "real-data smoke: {} samples ({} kept / {} rejected), labels = {}",
            samples.len(),
            samples.len() - positives,
            positives,
            LABEL_SOURCE
        );

        let mut model = MlCheatModel::new(MlCheatConfig::default()).expect("config");
        let report = model.train(&samples).expect("train on real traces");
        // Train-set metrics only: a smoke signal, printed for the log.
        let eval = evaluate(&model, &samples).expect("evaluate");
        eprintln!(
            "real-data train-set metrics (not a generalization claim): \
             accuracy={:.3} precision={:.3} recall={:.3} auc={:.3}",
            eval.accuracy, eval.precision, eval.recall, eval.auc
        );
        for (name, v) in [
            ("accuracy", eval.accuracy),
            ("precision", eval.precision),
            ("recall", eval.recall),
        ] {
            assert!((0.0..=1.0).contains(&v), "{name} out of [0,1]: {v}");
        }
        assert!(
            eval.auc.is_nan() || (0.0..=1.0).contains(&eval.auc),
            "auc out of range: {}",
            eval.auc
        );
        assert_eq!(report.samples, samples.len());
    }
}
