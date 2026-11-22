//! The three-layer hierarchical router (TRISKELE-PLAN §3.9, §13; Stage 4):
//! **meta-router -> category routers -> sub-specialists**.
//!
//! ```text
//!  prompt (+ memory context)
//!     |
//!  L3 MetaRouter       which *combination* of categories, with weights,
//!     |                or a calibrated "unsure"
//!  L2 CategoryRouter   per engaged category: which specialists, blended how
//!     |                (MoSME box gating over the catalogue's boxes)
//!  planner budget      at most `max_specialists` loaded, by joint weight
//!     |
//!  L1 specialists      the nina `.mpk` experts, resolved through the
//!                      registry / expert index
//! ```
//!
//! Routing is by **domain only** (mandate §3.5): no layer scores safety, and
//! nothing here refuses.
//!
//! # L3: why a hashed n-gram softmax model
//!
//! The plan says the routers will eventually be trained with the NCM
//! pipeline. What exists today is ~150 labelled traces and a CPU. A
//! multinomial logistic regression over hashed char n-grams and word uni/bi-
//! grams (the [`crate::mlcheat`] recipe, multi-class) is the honest capacity
//! for that: it trains in well under a second, is deterministic bit for bit,
//! serializes to JSON, and has nothing to overfit with beyond its weight
//! table. It sits behind the [`MetaRouter`] trait so an NCM-based router
//! replaces it without touching the layers below.
//!
//! The model is trained with **soft labels**: a trace labelled
//! `coding 0.67 + data 0.33` is a target distribution, so the softmax learns
//! blends, not single picks. One extra output is the *none* class (traces
//! with no category). Probabilities are **temperature-calibrated** on a slice
//! of the training traces held out from fitting (Guo et al. 2017 temperature
//! scaling: one scalar, NLL-minimizing). The router abstains — the
//! calibrated "unsure" outcome — when the none class is the most probable or
//! the top category's calibrated probability is under
//! [`MetaPolicy::confidence_floor`]. The policy knobs are a-priori defaults,
//! not tuned on any eval set.
//!
//! # L2: MoSME, promoted
//!
//! [`MosmeCategoryRouter`] does not reimplement anything: it builds a
//! [`crate::expert_index::MosmeSpec`] with **one box per category** and the
//! category's specialists as the box's experts (the same shape as
//! `dblocks experts init --box coding:rust,python`), instantiates
//! [`crate::mosme::HierarchicalRouter`], and reads the within-box expert gates
//! — masked softmax, top-k, renormalized, a disabled expert's gate exactly
//! zero. Its router weights start at **zero**, so until a trained router is
//! supplied ([`MosmeCategoryRouter::with_router`]) its gates are uniform over
//! the first `k` enabled specialists in catalogue order: a stated
//! no-information prior, not a disguised random pick. The router traces built
//! so far carry no sub-field labels, so no L2 training exists yet.
//! [`TrafficCategoryRouter`] serves a category from an already-trained MoSME
//! box readout (the student's probe router's per-language box traffic, which
//! `nina route_prompt` computes) and is consulted first when present.
//!
//! The cross-category specialist budget is a [`crate::planner::Beam`] search
//! under a [`crate::planner::Budget`]. With additive joint weights the
//! optimum is the top-k by weight, so the beam width is 1; the planner is used
//! for its enforced evaluation ceiling and its documented, deterministic tie
//! order.
//!
//! # Invariants (checked by [`check_plan`] and the `router` certificates)
//!
//! - every layer's weights are finite, non-negative and sum to 1;
//! - every chosen specialist belongs to a chosen category, every chosen
//!   category has at least one specialist, and a category's specialists carry
//!   exactly its weight;
//! - top-k bounds hold at every layer;
//! - a fixed model routes a fixed prompt to a bit-identical plan.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{backend::Backend, Tensor},
};
use burn_ndarray::NdArray;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::expert_index::{BalanceWeights, BoxSpec, ExpertIndex, ExpertSpec, MosmeSpec};
use crate::nina::{ModelStatus, RegistryFile};
use crate::taxonomy::{SpecialistMapping, Taxonomy};

// ------------------------------------------------------------- features --

// FNV-1a 64 with the fixed basis and prime: a saved model's features must
// hash identically in every process, so the hasher is spelled out (the same
// contract `mlcheat` and `cheat` use).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(tag: u8, parts: &[&[u8]]) -> u64 {
    let mut h = FNV_OFFSET;
    let mut feed = |b: u8| {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    };
    feed(tag);
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            feed(0x1f);
        }
        for b in *part {
            feed(*b);
        }
    }
    h
}

/// How text becomes a sparse feature vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureConfig {
    /// Log2 of the hash-bin count, `4..=20`.
    pub hash_bits: u32,
    /// Char n-gram widths, each `2..=8`.
    pub char_widths: Vec<usize>,
    /// Word bigrams on top of word unigrams.
    pub word_bigrams: bool,
    /// Weight of the memory-context features relative to the prompt's,
    /// `0..=1`. Context is hashed with its own tag, so a word in the context
    /// and the same word in the prompt are different features.
    pub context_scale: f32,
    /// Characters of prompt (and of context) read; bounds the cost.
    pub max_chars: usize,
    /// Collapse runs of one repeated letter before featurizing ("sooo
    /// slooow" -> "so slow"), so emphasis and stuck keys do not create
    /// unseen n-grams. Off by default: models saved without it keep their
    /// features.
    pub squash_repeats: bool,
    /// Also hash each word's char n-grams with boundary marks (`<word>`,
    /// fastText-style), so a typo perturbs only the grams of the word it is
    /// in. Off by default.
    pub word_boundary_grams: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            hash_bits: 12,
            char_widths: vec![3, 4, 5],
            word_bigrams: true,
            context_scale: 0.5,
            max_chars: 4096,
            squash_repeats: false,
            word_boundary_grams: false,
        }
    }
}

impl FeatureConfig {
    /// The `--tiny` configuration: fewer bins.
    pub fn tiny() -> Self {
        Self {
            hash_bits: 10,
            ..Self::default()
        }
    }

    /// A named feature set: `base` (the default), `squash` (base on
    /// repeat-collapsed text), `rich` (squash plus per-word boundary grams in
    /// a 4x larger table). `tiny` shrinks the table by 4x.
    pub fn preset(name: &str, tiny: bool) -> Result<Self> {
        let base = if tiny { Self::tiny() } else { Self::default() };
        let config = match name {
            "base" => base,
            "squash" => Self {
                squash_repeats: true,
                ..base
            },
            "rich" => Self {
                squash_repeats: true,
                word_boundary_grams: true,
                hash_bits: base.hash_bits + 2,
                ..base
            },
            other => bail!("unknown feature set '{other}' (expected base, squash or rich)"),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn dim(&self) -> usize {
        1usize << self.hash_bits
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            (4..=20).contains(&self.hash_bits),
            "hash_bits must be in 4..=20, got {}",
            self.hash_bits
        );
        ensure!(
            !self.char_widths.is_empty(),
            "char_widths must not be empty"
        );
        for w in &self.char_widths {
            ensure!(
                (2..=8).contains(w),
                "char n-gram width must be in 2..=8, got {w}"
            );
        }
        ensure!(
            self.context_scale.is_finite() && (0.0..=1.0).contains(&self.context_scale),
            "context_scale must be in [0, 1], got {}",
            self.context_scale
        );
        ensure!(self.max_chars > 0, "max_chars must be positive");
        Ok(())
    }
}

/// Collapse runs of one repeated alphabetic character to a single one.
fn squash_repeats(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last: Option<char> = None;
    for c in text.chars() {
        if c.is_alphabetic() && last == Some(c) {
            continue;
        }
        out.push(c);
        last = Some(c);
    }
    out
}

fn push_text(feats: &mut Vec<(u32, f32)>, text: &str, tag: u8, scale: f32, cfg: &FeatureConfig) {
    let mask = (1u64 << cfg.hash_bits) - 1;
    let mut lower: String = text.to_lowercase().chars().take(cfg.max_chars).collect();
    if cfg.squash_repeats {
        lower = squash_repeats(&lower);
    }
    let chars: Vec<char> = lower.chars().collect();
    let mut push = |h: u64| feats.push(((h & mask) as u32, scale));
    for &width in &cfg.char_widths {
        for gram in chars.windows(width) {
            let mut buf = Vec::with_capacity(width * 4);
            for c in gram {
                buf.extend_from_slice(&u32::from(*c).to_le_bytes());
            }
            push(fnv(tag ^ (width as u8), &[buf.as_slice()]));
        }
    }
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    for w in &words {
        push(fnv(tag ^ 0x40, &[w.as_bytes()]));
    }
    if cfg.word_boundary_grams {
        for w in &words {
            let marked: Vec<char> = std::iter::once('<')
                .chain(w.chars())
                .chain(std::iter::once('>'))
                .collect();
            for &width in &cfg.char_widths {
                for gram in marked.windows(width) {
                    let mut buf = Vec::with_capacity(width * 4);
                    for c in gram {
                        buf.extend_from_slice(&u32::from(*c).to_le_bytes());
                    }
                    push(fnv(tag ^ 0x20 ^ (width as u8), &[buf.as_slice()]));
                }
            }
        }
    }
    if cfg.word_bigrams {
        for pair in words.windows(2) {
            if let [a, b] = pair {
                push(fnv(tag ^ 0x60, &[a.as_bytes(), b.as_bytes()]));
            }
        }
    }
}

/// Sparse hashed features of a prompt and its optional memory context:
/// `(bin, value)` sorted by bin, duplicates merged, unit L2 norm (so length
/// does not leak into the scores). Fully deterministic.
pub fn featurize(prompt: &str, context: Option<&str>, cfg: &FeatureConfig) -> Vec<(u32, f32)> {
    let mut feats = Vec::new();
    push_text(&mut feats, prompt, 0x50, 1.0, cfg);
    if let Some(context) = context {
        if cfg.context_scale > 0.0 {
            push_text(&mut feats, context, 0x43, cfg.context_scale, cfg);
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

fn densify(feats: &[(u32, f32)], dim: usize) -> Vec<f32> {
    let mut dense = vec![0.0f32; dim];
    for &(idx, v) in feats {
        if let Some(slot) = dense.get_mut(idx as usize) {
            *slot += v;
        }
    }
    dense
}

// --------------------------------------------------------------- traces --

/// One category of a trace's label, with its share of the task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceCategory {
    pub id: String,
    pub weight: f64,
}

/// One meta-router training trace (one JSONL line):
///
/// ```json
/// {"id": "t-17", "text": "...", "context": "optional memory context",
///  "categories": [{"id": "coding", "weight": 0.67}, {"id": "data", "weight": 0.33}],
///  "subfields": ["coding/python", "data/databases"]}
/// ```
///
/// `categories` empty means "none of the taxonomy". Weights must be finite,
/// non-negative and sum to 1 within [`TRACE_WEIGHT_TOLERANCE`] (labels are
/// written to three decimals); they are renormalized exactly on load.
/// `subfields` is optional and, when present, each must be a taxonomy leaf
/// of one of the trace's categories (L2 supervision for a later trainer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouterTrace {
    pub id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    pub categories: Vec<TraceCategory>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subfields: Vec<String>,
    /// Optional tag for per-style eval breakdowns (`short`, `combo`, `oos`,
    /// ...). Never a feature: the router does not read it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

/// How far a trace's category weights may sum from 1 before it is refused.
pub const TRACE_WEIGHT_TOLERANCE: f64 = 1e-2;

impl RouterTrace {
    pub fn validate(&self, taxonomy: &Taxonomy) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "a trace needs a non-empty id");
        ensure!(
            !self.text.trim().is_empty(),
            "trace '{}' has empty text",
            self.id
        );
        let mut seen = BTreeSet::new();
        let mut sum = 0.0f64;
        for c in &self.categories {
            ensure!(
                seen.insert(c.id.as_str()),
                "trace '{}' lists category '{}' twice",
                self.id,
                c.id
            );
            taxonomy
                .check_ids(&c.id, None)
                .with_context(|| format!("trace '{}'", self.id))?;
            ensure!(
                c.weight.is_finite() && c.weight >= 0.0,
                "trace '{}' gives category '{}' weight {}, not a finite non-negative number",
                self.id,
                c.id,
                c.weight
            );
            sum += c.weight;
        }
        if !self.categories.is_empty() {
            ensure!(
                (sum - 1.0).abs() <= TRACE_WEIGHT_TOLERANCE,
                "trace '{}' category weights sum to {sum}, not 1 (tolerance {TRACE_WEIGHT_TOLERANCE})",
                self.id
            );
        }
        for leaf in &self.subfields {
            let (category, _) = crate::taxonomy::parse_leaf_id(leaf)
                .with_context(|| format!("trace '{}'", self.id))?;
            ensure!(
                seen.contains(category),
                "trace '{}' names sub-field '{leaf}' outside its categories",
                self.id
            );
            taxonomy
                .check_ids(category, Some(leaf))
                .with_context(|| format!("trace '{}'", self.id))?;
        }
        Ok(())
    }

    /// Weights rescaled to sum to exactly 1 (a no-op for an empty label).
    pub fn normalized(mut self) -> Self {
        let sum: f64 = self.categories.iter().map(|c| c.weight).sum();
        if sum > 0.0 {
            for c in &mut self.categories {
                c.weight /= sum;
            }
        }
        self
    }

    /// Categories with positive weight.
    pub fn gold_set(&self) -> BTreeSet<&str> {
        self.categories
            .iter()
            .filter(|c| c.weight > 0.0)
            .map(|c| c.id.as_str())
            .collect()
    }

    /// The breakdown group: the `style` tag when present, else derived from
    /// the label — `oos` for no category, `combo` for several, `single`.
    pub fn group(&self) -> &str {
        match &self.style {
            Some(style) => style,
            None if self.categories.is_empty() => "oos",
            None if self.categories.len() > 1 => "combo",
            None => "single",
        }
    }

    /// The highest-weighted category (first on a tie); `None` for "none".
    pub fn gold_top(&self) -> Option<&str> {
        self.categories
            .iter()
            .filter(|c| c.weight > 0.0)
            .fold(None, |best: Option<&TraceCategory>, c| match best {
                Some(b) if b.weight >= c.weight => Some(b),
                _ => Some(c),
            })
            .map(|c| c.id.as_str())
    }
}

/// Parse JSONL traces: blank lines skipped, every line validated against the
/// taxonomy (the error names the line), ids unique, weights renormalized.
pub fn parse_traces(text: &str, taxonomy: &Taxonomy) -> Result<Vec<RouterTrace>> {
    let mut out = Vec::new();
    let mut ids = BTreeSet::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let trace: RouterTrace = serde_json::from_str(line)
            .with_context(|| format!("trace line {}: not a trace object", lineno + 1))?;
        trace
            .validate(taxonomy)
            .with_context(|| format!("trace line {}", lineno + 1))?;
        ensure!(
            ids.insert(trace.id.clone()),
            "trace line {}: duplicate id '{}'",
            lineno + 1,
            trace.id
        );
        out.push(trace.normalized());
    }
    Ok(out)
}

pub fn read_traces(path: &Path, taxonomy: &Taxonomy) -> Result<Vec<RouterTrace>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("read traces {}", path.display()))?;
    parse_traces(&text, taxonomy).with_context(|| format!("in {}", path.display()))
}

/// Read several trace files as one set, in order; ids must be unique across
/// files (two files reusing an id would silently share a hash fold).
pub fn read_trace_files(paths: &[PathBuf], taxonomy: &Taxonomy) -> Result<Vec<RouterTrace>> {
    ensure!(!paths.is_empty(), "no trace files were given");
    let mut out = Vec::new();
    let mut ids = BTreeSet::new();
    for path in paths {
        for trace in read_traces(path, taxonomy)? {
            ensure!(
                ids.insert(trace.id.clone()),
                "trace id '{}' in {} repeats an id from an earlier file",
                trace.id,
                path.display()
            );
            out.push(trace);
        }
    }
    Ok(out)
}

/// Deterministic bucket `0..100` of a trace id — the hash split, so a trace
/// lands in the same fold on every run and on every machine.
pub fn holdout_bucket(id: &str) -> u32 {
    (fnv(0x53, &[id.as_bytes()]) % 100) as u32
}

/// Split by id hash: `(kept, held_out)` with roughly `percent`% held out.
pub fn split_traces(traces: &[RouterTrace], percent: u32) -> (Vec<RouterTrace>, Vec<RouterTrace>) {
    traces
        .iter()
        .cloned()
        .partition(|t| holdout_bucket(&t.id) >= percent)
}

// ------------------------------------------------------------ L3: meta --

/// What the meta-router decided for one prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaDecision {
    /// Calibrated probability of every category, highest first (ties in
    /// model order). Excludes the none class.
    pub scores: Vec<(String, f64)>,
    /// Calibrated probability of "none of the categories".
    pub none: f64,
    /// The engaged categories and their blend weights (summing to 1); empty
    /// when the router abstains.
    pub engaged: Vec<(String, f64)>,
    /// Why the router abstained, when it did.
    pub unsure: Option<String>,
}

/// The top layer: prompt (+ memory context) -> a weighted category
/// combination, or a reasoned abstention. Implemented here by
/// [`HashedMetaRouter`]; an NCM-trained router implements the same trait.
pub trait MetaRouter {
    /// A short name for traces (`hashed-softmax`, `ncm`, ...).
    fn name(&self) -> &str;
    /// The category ids this router can output.
    fn categories(&self) -> &[String];
    fn decide(&self, prompt: &str, context: Option<&str>) -> Result<MetaDecision>;
}

/// When the meta-router engages several categories and when it abstains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MetaPolicy {
    /// At most this many categories engaged per prompt.
    pub max_categories: usize,
    /// A category is engaged when its probability is at least this fraction
    /// of the top category's.
    pub engage_ratio: f64,
    /// Abstain when the top category's calibrated probability is below this.
    pub confidence_floor: f64,
    /// Abstain when the none class comes within this margin of the top
    /// category: `p_none >= p_top - none_margin`. `0` abstains exactly when
    /// none is the most probable outcome; a negative margin abstains less.
    pub none_margin: f64,
}

impl Default for MetaPolicy {
    fn default() -> Self {
        Self {
            max_categories: 3,
            engage_ratio: 0.3,
            confidence_floor: 0.25,
            none_margin: 0.0,
        }
    }
}

impl MetaPolicy {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_categories >= 1,
            "max_categories must be at least 1"
        );
        ensure!(
            self.engage_ratio.is_finite() && self.engage_ratio > 0.0 && self.engage_ratio <= 1.0,
            "engage_ratio must be in (0, 1], got {}",
            self.engage_ratio
        );
        ensure!(
            self.confidence_floor.is_finite() && (0.0..1.0).contains(&self.confidence_floor),
            "confidence_floor must be in [0, 1), got {}",
            self.confidence_floor
        );
        ensure!(
            self.none_margin.is_finite() && (-1.0..=1.0).contains(&self.none_margin),
            "none_margin must be in [-1, 1], got {}",
            self.none_margin
        );
        Ok(())
    }

    /// Apply the policy to calibrated scores (sorted, highest first).
    pub fn decide(&self, scores: Vec<(String, f64)>, none: f64) -> MetaDecision {
        let top = scores.first().map_or(0.0, |s| s.1);
        let unsure = if scores.is_empty() {
            Some("the router has no categories".to_string())
        } else if none >= top - self.none_margin {
            Some(format!(
                "the none class is within the margin of the top category \
                 (p_none={none:.3}, p_top={top:.3}, none_margin={:.3})",
                self.none_margin
            ))
        } else if top < self.confidence_floor {
            Some(format!(
                "top category at calibrated p={top:.3} is below the confidence floor {:.3}",
                self.confidence_floor
            ))
        } else {
            None
        };
        let engaged = if unsure.is_some() {
            Vec::new()
        } else {
            let chosen: Vec<(String, f64)> = scores
                .iter()
                .filter(|(_, p)| *p >= self.engage_ratio * top && *p > 0.0)
                .take(self.max_categories)
                .cloned()
                .collect();
            normalize_pairs(chosen)
        };
        MetaDecision {
            scores,
            none,
            engaged,
            unsure,
        }
    }
}

/// Rescale `(id, w)` pairs to sum to 1; an all-zero or empty input stays as is.
fn normalize_pairs(pairs: Vec<(String, f64)>) -> Vec<(String, f64)> {
    let sum: f64 = pairs.iter().map(|p| p.1).sum();
    if sum > 0.0 && sum.is_finite() {
        pairs.into_iter().map(|(id, w)| (id, w / sum)).collect()
    } else {
        pairs
    }
}

/// How the meta-router trains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MetaTrainConfig {
    /// SGD learning rate (constant).
    pub lr: f32,
    pub epochs: usize,
    /// L2 penalty on the weights a sample touches (sparse approximation).
    pub l2: f32,
    /// ChaCha12 seed for the per-epoch shuffle.
    pub seed: u64,
    /// Share (by hash) of the training traces held out from fitting and used
    /// only to fit the temperature and tune the unsure rule.
    pub calibration_percent: u32,
    /// What the calibration hash is taken over; see [`CalibrationGroup`].
    pub calibration_group: CalibrationGroup,
}

/// How the calibration slice is drawn from the training traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationGroup {
    /// By trace id: the calibration slice looks exactly like the fit slice.
    #[default]
    Id,
    /// By the trace's first sub-field leaf (by id for a trace without one):
    /// whole leaves are unseen at fit time, so the slice measures how the
    /// router behaves on topics it has not been trained on — closer to a
    /// held-out set of hand-written prompts than an id split of templated
    /// prompts. The weights are refitted on every trace afterwards
    /// ([`HashedMetaRouter::refit_all`]), keeping the temperature and rule.
    Leaf,
}

impl CalibrationGroup {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "id" => Ok(Self::Id),
            "leaf" => Ok(Self::Leaf),
            other => bail!("unknown calibration split '{other}' (expected 'id' or 'leaf')"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Id => "id",
            Self::Leaf => "leaf",
        }
    }
}

impl Default for MetaTrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.5,
            epochs: 40,
            l2: 1e-5,
            seed: 0x5eed_0004,
            calibration_percent: 20,
            calibration_group: CalibrationGroup::Id,
        }
    }
}

impl MetaTrainConfig {
    pub fn tiny() -> Self {
        Self {
            epochs: 15,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.lr.is_finite() && self.lr > 0.0,
            "lr must be positive, got {}",
            self.lr
        );
        ensure!(self.epochs >= 1, "epochs must be at least 1");
        ensure!(
            self.l2.is_finite() && self.l2 >= 0.0,
            "l2 must be non-negative, got {}",
            self.l2
        );
        ensure!(
            self.calibration_percent < 100,
            "calibration_percent must be below 100"
        );
        Ok(())
    }
}

/// One featurized trace: its sparse features and its target distribution
/// over the router's classes.
type Sample = (Vec<(u32, f32)>, Vec<f64>);

/// `(fit, calibration)` by [`MetaTrainConfig`]'s hash split. A calibration
/// slice that swallowed everything leaves nothing to fit, so then everything
/// is fitted and the calibration slice is empty.
pub fn calibration_split(
    traces: &[RouterTrace],
    config: &MetaTrainConfig,
) -> (Vec<RouterTrace>, Vec<RouterTrace>) {
    let bucket = |t: &RouterTrace| match config.calibration_group {
        CalibrationGroup::Id => holdout_bucket(&t.id),
        CalibrationGroup::Leaf => t
            .subfields
            .first()
            .map_or_else(|| holdout_bucket(&t.id), |leaf| holdout_bucket(leaf)),
    };
    let (fit, calibration): (Vec<RouterTrace>, Vec<RouterTrace>) = traces
        .iter()
        .cloned()
        .partition(|t| bucket(t) >= config.calibration_percent);
    if fit.is_empty() {
        (calibration, Vec::new())
    } else {
        (fit, calibration)
    }
}

/// Refuse a trace whose category the router does not output.
fn check_trace_categories(categories: &[String], traces: &[RouterTrace]) -> Result<()> {
    ensure!(
        !traces.is_empty(),
        "cannot train a meta-router on zero traces"
    );
    for trace in traces {
        for c in &trace.categories {
            ensure!(
                categories.contains(&c.id),
                "trace '{}' uses category '{}', which this router does not output",
                trace.id,
                c.id
            );
        }
    }
    Ok(())
}

/// Target distribution over `categories + none` for a trace (none last).
fn soft_target(categories: &[String], trace: &RouterTrace) -> Vec<f64> {
    let mut y = vec![0.0; categories.len() + 1];
    if trace.categories.is_empty() {
        if let Some(last) = y.last_mut() {
            *last = 1.0;
        }
    } else {
        for c in &trace.categories {
            if let Some(slot) = categories
                .iter()
                .position(|id| *id == c.id)
                .and_then(|k| y.get_mut(k))
            {
                *slot += c.weight;
            }
        }
    }
    y
}

/// Logits of a linear softmax head (`bias.len()` classes, `dim` inputs,
/// row-major weights) on sparse features.
fn head_logits(weights: &[f32], bias: &[f32], dim: usize, feats: &[(u32, f32)]) -> Vec<f64> {
    bias.iter()
        .enumerate()
        .map(|(k, b)| {
            let row = k * dim;
            let mut z = f64::from(*b);
            for &(idx, v) in feats {
                let w = weights.get(row + idx as usize).copied().unwrap_or(0.0);
                z += f64::from(w) * f64::from(v);
            }
            z
        })
        .collect()
}

/// Softmax of `logits / temperature`, max-shifted.
fn softmax_at(logits: &[f64], temperature: f64) -> Vec<f64> {
    let t = temperature.max(1e-6);
    let max = logits.iter().fold(f64::NEG_INFINITY, |m, z| m.max(*z / t));
    let exps: Vec<f64> = logits.iter().map(|z| (z / t - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Soft cross-entropy `-sum y log p`.
fn cross_entropy(y: &[f64], p: &[f64]) -> f64 {
    -y.iter()
        .zip(p)
        .filter(|(yk, _)| **yk > 0.0)
        .map(|(yk, pk)| yk * pk.max(1e-12).ln())
        .sum::<f64>()
}

/// Mean soft cross-entropy of `data` at `temperature` (`NaN` when empty).
fn head_nll(weights: &[f32], bias: &[f32], dim: usize, data: &[Sample], temperature: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let total: f64 = data
        .iter()
        .map(|(x, y)| {
            cross_entropy(
                y,
                &softmax_at(&head_logits(weights, bias, dim, x), temperature),
            )
        })
        .sum();
    total / data.len() as f64
}

/// Hand-rolled SGD on the soft cross-entropy, ChaCha12-shuffled per epoch:
/// the same config and data reproduce the same weights bit for bit. Returns
/// the online loss per epoch.
fn sgd_fit(
    weights: &mut [f32],
    bias: &mut [f32],
    dim: usize,
    data: &[Sample],
    config: &MetaTrainConfig,
) -> Vec<f64> {
    let lr = config.lr;
    let l2 = config.l2;
    let mut rng = ChaCha12Rng::seed_from_u64(config.seed);
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut loss_curve = Vec::with_capacity(config.epochs);
    for _ in 0..config.epochs {
        order.shuffle(&mut rng);
        let mut loss = 0.0f64;
        for &i in &order {
            let Some((x, y)) = data.get(i) else {
                continue;
            };
            let p = softmax_at(&head_logits(weights, bias, dim, x), 1.0);
            loss += cross_entropy(y, &p);
            for (k, b) in bias.iter_mut().enumerate() {
                let g =
                    (p.get(k).copied().unwrap_or(0.0) - y.get(k).copied().unwrap_or(0.0)) as f32;
                *b -= lr * g;
                let row = k * dim;
                for &(idx, v) in x {
                    if let Some(w) = weights.get_mut(row + idx as usize) {
                        *w -= lr * (g * v + l2 * *w);
                    }
                }
            }
        }
        loss_curve.push(loss / data.len().max(1) as f64);
    }
    loss_curve
}

/// Temperature scaling: one scalar on a log grid from 0.05 to 20,
/// NLL-minimizing on the calibration data; the first minimum wins, so ties
/// are stable. Returns `(temperature, nll at 1, nll at the temperature)`;
/// `(1, NaN, NaN)` without calibration data.
fn fit_temperature(
    weights: &[f32],
    bias: &[f32],
    dim: usize,
    calibration: &[Sample],
) -> (f64, f64, f64) {
    let raw = head_nll(weights, bias, dim, calibration, 1.0);
    let mut best = (1.0f64, raw);
    if !calibration.is_empty() {
        let (lo, hi) = (0.05f64.ln(), 20.0f64.ln());
        for step in 0..=120 {
            let t = (lo + (hi - lo) * f64::from(step) / 120.0).exp();
            let nll = head_nll(weights, bias, dim, calibration, t);
            if nll.is_finite() && nll < best.1 {
                best = (t, nll);
            }
        }
    }
    (best.0, raw, best.1)
}

/// Calibrated scores (highest first, ties in model order) and the none
/// probability from a softmax head's logits.
fn ranked_scores(
    categories: &[String],
    logits: &[f64],
    temperature: f64,
) -> (Vec<(String, f64)>, f64) {
    let p = softmax_at(logits, temperature);
    let none = p.last().copied().unwrap_or(0.0);
    let mut scores: Vec<(String, f64)> = categories
        .iter()
        .zip(&p)
        .map(|(id, pk)| (id.clone(), *pk))
        .collect();
    // Stable sort, highest first; NaN cannot arise from a finite softmax.
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    (scores, none)
}

/// A meta-router whose decision is a policy applied to calibrated scores —
/// what [`tune_policy`] needs to search the policy without re-running it.
pub trait ScoredMetaRouter: MetaRouter {
    /// Calibrated category probabilities (highest first) and the none
    /// probability.
    fn calibrated_scores(
        &self,
        prompt: &str,
        context: Option<&str>,
    ) -> Result<(Vec<(String, f64)>, f64)>;
    fn policy(&self) -> &MetaPolicy;
    fn is_trained(&self) -> bool;
}

/// Current on-disk version of a [`HashedMetaRouter`].
pub const META_ROUTER_VERSION: u32 = 1;

/// The CPU baseline meta-router: multinomial logistic regression over hashed
/// n-gram features, one output per category plus a none class, temperature
/// calibrated. See the module docs for why this capacity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HashedMetaRouter {
    pub version: u32,
    pub features: FeatureConfig,
    pub train_config: MetaTrainConfig,
    pub policy: MetaPolicy,
    /// Output order; the none class is the extra last row.
    pub categories: Vec<String>,
    /// The taxonomy version the categories were drawn from.
    pub taxonomy_version: String,
    /// `(categories + 1) x dim`, row-major per class.
    weights: Vec<f32>,
    /// `categories + 1`.
    bias: Vec<f32>,
    /// Softmax temperature fitted on the calibration slice (1 = uncalibrated).
    pub temperature: f64,
    /// Traces the weights were fitted on (0 = untrained; an untrained router
    /// always abstains).
    pub trained_samples: usize,
    /// How many fitted traces were labelled none.
    pub none_samples: usize,
}

impl HashedMetaRouter {
    /// An untrained router over every category of `taxonomy`.
    pub fn new(
        taxonomy: &Taxonomy,
        features: FeatureConfig,
        train_config: MetaTrainConfig,
        policy: MetaPolicy,
    ) -> Result<Self> {
        features.validate()?;
        train_config.validate()?;
        policy.validate()?;
        let categories = taxonomy.category_ids();
        let classes = categories.len() + 1;
        Ok(Self {
            version: META_ROUTER_VERSION,
            weights: vec![0.0; classes * features.dim()],
            bias: vec![0.0; classes],
            features,
            train_config,
            policy,
            categories,
            taxonomy_version: taxonomy.version.clone(),
            temperature: 1.0,
            trained_samples: 0,
            none_samples: 0,
        })
    }

    fn logits(&self, feats: &[(u32, f32)]) -> Vec<f64> {
        head_logits(&self.weights, &self.bias, self.features.dim(), feats)
    }

    fn encode(&self, set: &[RouterTrace]) -> Vec<Sample> {
        set.iter()
            .map(|t| {
                (
                    featurize(&t.text, t.context.as_deref(), &self.features),
                    soft_target(&self.categories, t),
                )
            })
            .collect()
    }

    /// Fit on `traces`: the calibration slice ([`calibration_split`]) is held
    /// out from the SGD and used only to fit the temperature. Deterministic:
    /// the same config and traces reproduce the same weights bit for bit.
    pub fn train(&mut self, traces: &[RouterTrace]) -> Result<MetaTrainReport> {
        check_trace_categories(&self.categories, traces)?;
        let (fit, calibration) = self.calibration_split(traces);
        let fit_data = self.encode(&fit);
        let calibration_data = self.encode(&calibration);
        let dim = self.features.dim();
        let loss_curve = sgd_fit(
            &mut self.weights,
            &mut self.bias,
            dim,
            &fit_data,
            &self.train_config,
        );
        let (temperature, raw, calibrated) =
            fit_temperature(&self.weights, &self.bias, dim, &calibration_data);
        self.temperature = temperature;
        self.trained_samples = fit_data.len();
        self.none_samples = fit.iter().filter(|t| t.categories.is_empty()).count();
        Ok(MetaTrainReport {
            fit_samples: fit_data.len(),
            calibration_samples: calibration_data.len(),
            none_samples: self.none_samples,
            loss_curve,
            temperature,
            calibration_nll_raw: raw,
            calibration_nll_calibrated: calibrated,
        })
    }

    /// Refit the weights from zero on every trace (fit and calibration
    /// slices together), keeping the temperature and the policy chosen on the
    /// calibration slice. Used after a [`CalibrationGroup::Leaf`] split, whose
    /// fit slice deliberately lacks whole leaves.
    pub fn refit_all(&mut self, traces: &[RouterTrace]) -> Result<Vec<f64>> {
        check_trace_categories(&self.categories, traces)?;
        self.weights.iter_mut().for_each(|w| *w = 0.0);
        self.bias.iter_mut().for_each(|b| *b = 0.0);
        let data = self.encode(traces);
        let loss = sgd_fit(
            &mut self.weights,
            &mut self.bias,
            self.features.dim(),
            &data,
            &self.train_config,
        );
        self.trained_samples = data.len();
        self.none_samples = traces.iter().filter(|t| t.categories.is_empty()).count();
        Ok(loss)
    }

    /// Fit the temperature on an external calibration set; returns
    /// `(temperature, nll at 1, nll at the temperature)`.
    pub fn calibrate(&mut self, traces: &[RouterTrace]) -> Result<(f64, f64, f64)> {
        check_trace_categories(&self.categories, traces)?;
        let data = self.encode(traces);
        let fitted = fit_temperature(&self.weights, &self.bias, self.features.dim(), &data);
        self.temperature = fitted.0;
        Ok(fitted)
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize meta-router")
    }

    /// Parse, refusing a newer version, a weight table that does not match
    /// the config, duplicate categories and any non-finite number (serde
    /// reads an out-of-range literal as infinity, which would poison every
    /// score it touches).
    pub fn from_json(text: &str) -> Result<Self> {
        let model: Self = serde_json::from_str(text).context("parse meta-router JSON")?;
        ensure!(
            model.version <= META_ROUTER_VERSION,
            "meta-router version {} is newer than this build understands ({META_ROUTER_VERSION})",
            model.version
        );
        model.features.validate()?;
        model.train_config.validate()?;
        model.policy.validate()?;
        ensure!(
            !model.categories.is_empty(),
            "meta-router has no categories"
        );
        let unique: BTreeSet<&String> = model.categories.iter().collect();
        ensure!(
            unique.len() == model.categories.len(),
            "meta-router lists a category twice"
        );
        let classes = model.categories.len() + 1;
        ensure!(
            model.weights.len() == classes * model.features.dim(),
            "meta-router has {} weights but its config implies {}",
            model.weights.len(),
            classes * model.features.dim()
        );
        ensure!(
            model.bias.len() == classes,
            "meta-router has {} biases for {classes} classes",
            model.bias.len()
        );
        ensure!(
            model
                .weights
                .iter()
                .chain(&model.bias)
                .all(|w| w.is_finite()),
            "meta-router carries a non-finite weight"
        );
        ensure!(
            model.temperature.is_finite() && model.temperature > 0.0,
            "meta-router temperature {} is not a positive finite number",
            model.temperature
        );
        Ok(model)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        crate::interop::write_atomic(path, self.to_json()?.as_bytes())
            .with_context(|| format!("write meta-router {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read meta-router {}", path.display()))?;
        Self::from_json(&text).with_context(|| format!("in {}", path.display()))
    }

    /// `(fit, calibration)`: the id-hash split [`Self::train`] uses. A
    /// calibration slice that swallowed everything leaves nothing to fit, so
    /// then everything is fitted and the calibration slice is empty.
    pub fn calibration_split(
        &self,
        traces: &[RouterTrace],
    ) -> (Vec<RouterTrace>, Vec<RouterTrace>) {
        calibration_split(traces, &self.train_config)
    }

    /// Calibrated category probabilities (highest first, ties in model
    /// order) and the none probability — everything the policy decides from.
    pub fn scores(&self, prompt: &str, context: Option<&str>) -> (Vec<(String, f64)>, f64) {
        let feats = featurize(prompt, context, &self.features);
        ranked_scores(&self.categories, &self.logits(&feats), self.temperature)
    }

    /// The same router with the policy replaced.
    pub fn with_policy(&self, policy: MetaPolicy) -> Self {
        Self {
            policy,
            ..self.clone()
        }
    }

    /// The same router with the temperature replaced (eval of the
    /// uncalibrated model uses `1.0`).
    pub fn with_temperature(&self, temperature: f64) -> Self {
        Self {
            temperature,
            ..self.clone()
        }
    }
}

impl MetaRouter for HashedMetaRouter {
    fn name(&self) -> &str {
        "hashed-softmax"
    }

    fn categories(&self) -> &[String] {
        &self.categories
    }

    fn decide(&self, prompt: &str, context: Option<&str>) -> Result<MetaDecision> {
        if self.trained_samples == 0 {
            return Ok(MetaDecision {
                scores: Vec::new(),
                none: 0.0,
                engaged: Vec::new(),
                unsure: Some(
                    "no trained meta-router is loaded (train one with `dblocks router train`)"
                        .to_string(),
                ),
            });
        }
        let (scores, none) = self.scores(prompt, context);
        Ok(self.policy.decide(scores, none))
    }
}

impl ScoredMetaRouter for HashedMetaRouter {
    fn calibrated_scores(
        &self,
        prompt: &str,
        context: Option<&str>,
    ) -> Result<(Vec<(String, f64)>, f64)> {
        Ok(self.scores(prompt, context))
    }

    fn policy(&self) -> &MetaPolicy {
        &self.policy
    }

    fn is_trained(&self) -> bool {
        self.trained_samples > 0
    }
}

// ------------------------------------------------- L3: tuning the rule --

/// A calibration trace as the policy search sees it: its calibrated scores,
/// its none probability and its gold top category (`None` when out of scope).
type ScoredTrace<'a> = (Vec<(String, f64)>, f64, Option<&'a str>);

/// The objective [`tune_policy`] maximizes, stated once so reports quote it.
pub const TUNING_OBJECTIVE: &str = "balanced accuracy of the route/abstain decision on the \
    calibration split: 0.5 * (in-scope traces routed) + 0.5 * (out-of-scope traces abstained on); \
    ties go to more in-scope traces routed to the right top category, then to the policy \
    closest to the default";

/// The unsure rule chosen on the calibration split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyTuning {
    pub objective: String,
    pub calibration_in_scope: usize,
    pub calibration_oos: usize,
    /// Policies evaluated (`confidence_floor` x `none_margin` grid).
    pub grid_points: usize,
    pub default_policy: MetaPolicy,
    pub default_objective: f64,
    pub chosen: MetaPolicy,
    pub chosen_objective: f64,
    /// Calibration in-scope traces routed with the right top category under
    /// the chosen policy.
    pub chosen_in_scope_routed_correct: f64,
    pub note: String,
}

/// Choose `confidence_floor` (0 to 0.6 in steps of 0.025) and `none_margin`
/// (-0.3 to 0.3 in steps of 0.05) on `calibration` — the traces held out from
/// fitting, never the eval set — maximizing [`TUNING_OBJECTIVE`]. The default
/// policy is the starting point and is kept unless a grid point is strictly
/// better (or equal and better on the tie-breaks). Without both in-scope and
/// out-of-scope calibration traces the objective is undefined, and the
/// default is kept with a note saying so.
pub fn tune_policy(
    router: &dyn ScoredMetaRouter,
    calibration: &[RouterTrace],
) -> Result<PolicyTuning> {
    let default = router.policy().clone();
    let scored: Vec<ScoredTrace<'_>> = calibration
        .iter()
        .map(|t| {
            router
                .calibrated_scores(&t.text, t.context.as_deref())
                .map(|(scores, none)| (scores, none, t.gold_top()))
        })
        .collect::<Result<_>>()?;
    let n_in = scored.iter().filter(|s| s.2.is_some()).count();
    let n_oos = scored.len() - n_in;
    let measure = |policy: &MetaPolicy| -> (f64, f64) {
        let (mut routed, mut correct, mut abstained) = (0usize, 0usize, 0usize);
        for (scores, none, gold) in &scored {
            let d = policy.decide(scores.clone(), *none);
            match gold {
                Some(gold) => {
                    if d.unsure.is_none() {
                        routed += 1;
                        if d.scores.first().is_some_and(|(id, _)| id == gold) {
                            correct += 1;
                        }
                    }
                }
                None => {
                    if d.unsure.is_some() {
                        abstained += 1;
                    }
                }
            }
        }
        let rate = |x: usize, n: usize| if n == 0 { 0.0 } else { x as f64 / n as f64 };
        (
            0.5 * rate(routed, n_in) + 0.5 * rate(abstained, n_oos),
            rate(correct, n_in),
        )
    };
    let (default_objective, default_correct) = measure(&default);
    let mut tuning = PolicyTuning {
        objective: TUNING_OBJECTIVE.to_string(),
        calibration_in_scope: n_in,
        calibration_oos: n_oos,
        grid_points: 0,
        default_policy: default.clone(),
        default_objective,
        chosen: default.clone(),
        chosen_objective: default_objective,
        chosen_in_scope_routed_correct: default_correct,
        note: String::new(),
    };
    if !router.is_trained() || n_in == 0 || n_oos == 0 {
        tuning.note = format!(
            "not tuned: the calibration split has {n_in} in-scope and {n_oos} out-of-scope traces \
             (both are needed{}); the default policy is kept",
            if !router.is_trained() {
                ", and the router is untrained"
            } else {
                ""
            }
        );
        return Ok(tuning);
    }
    let distance = |p: &MetaPolicy| {
        (p.confidence_floor - default.confidence_floor).abs()
            + (p.none_margin - default.none_margin).abs()
    };
    let mut best_distance = 0.0;
    for floor_step in 0..=24u32 {
        for margin_step in 0..=12u32 {
            let candidate = MetaPolicy {
                confidence_floor: f64::from(floor_step) * 0.025,
                none_margin: -0.3 + f64::from(margin_step) * 0.05,
                ..default.clone()
            };
            tuning.grid_points += 1;
            let (objective, correct) = measure(&candidate);
            let d = distance(&candidate);
            let eps = 1e-12;
            let better = objective > tuning.chosen_objective + eps
                || ((objective - tuning.chosen_objective).abs() <= eps
                    && (correct > tuning.chosen_in_scope_routed_correct + eps
                        || ((correct - tuning.chosen_in_scope_routed_correct).abs() <= eps
                            && d < best_distance - eps)));
            if better {
                tuning.chosen = candidate;
                tuning.chosen_objective = objective;
                tuning.chosen_in_scope_routed_correct = correct;
                best_distance = d;
            }
        }
    }
    tuning.note = format!(
        "tuned on {} calibration traces ({n_in} in-scope, {n_oos} out-of-scope); small n, so the \
         chosen thresholds are themselves noisy",
        n_in + n_oos
    );
    Ok(tuning)
}

/// What one training run did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaTrainReport {
    pub fit_samples: usize,
    pub calibration_samples: usize,
    pub none_samples: usize,
    /// Online mean soft cross-entropy per epoch (each sample's loss just
    /// before its own step): a convergence trace, not a held-out number.
    pub loss_curve: Vec<f64>,
    pub temperature: f64,
    /// Calibration-slice NLL at temperature 1 and at the fitted temperature
    /// (`NaN` when the slice is empty).
    pub calibration_nll_raw: f64,
    pub calibration_nll_calibrated: f64,
}

// ------------------------------------------------------------- L3 eval --

/// Held-out metrics of a meta-router.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetaEval {
    pub samples: usize,
    /// Arg-max category equals the trace's highest-weighted category (for a
    /// none trace: the router abstained).
    pub top1_correct: usize,
    pub top1_accuracy: f64,
    /// Mean F1 between the engaged set (empty when abstaining) and the
    /// trace's category set; two empty sets score 1.
    pub mean_set_f1: f64,
    /// Mean L1 distance between engaged weights and label weights, in `[0, 2]`.
    pub mean_weight_l1: f64,
    /// Prompts the router abstained on.
    pub unsure: usize,
    /// Expected calibration error of the top-1 confidence, 5 equal bins.
    pub ece_top1: f64,
    /// The training set's most frequent top category, and the top-1 accuracy
    /// of always answering it — the floor a router has to beat.
    pub majority_category: Option<String>,
    pub majority_top1_accuracy: f64,
    /// Wilson 95% interval of `top1_accuracy`.
    pub top1_wilson95: (f64, f64),
    /// In-scope traces (at least one category): arg-max is the top label.
    pub in_scope_top1: Rate,
    /// In-scope traces the router routed rather than abstained on.
    pub in_scope_routed: Rate,
    /// In-scope traces routed with the right top category — what a caller
    /// actually receives.
    pub in_scope_routed_correct: Rate,
    /// Out-of-scope traces the router abstained on (unsure recall).
    pub oos_unsure_recall: Rate,
    /// Abstentions that were on out-of-scope traces (unsure precision).
    pub oos_unsure_precision: Rate,
    /// Mean set-F1 over traces with several categories (`NaN` when none).
    pub combo_set_f1: f64,
    pub combo_traces: usize,
    /// `0.5 * in_scope_routed + 0.5 * oos_unsure_recall`, the tuning objective
    /// measured on this set.
    pub balanced_accuracy: f64,
    /// Per [`RouterTrace::group`]: the `style` tag, or derived from the label.
    pub by_style: Vec<StyleEval>,
    /// In-scope top-1 per gold top category, in first-seen order — where the
    /// router is weak.
    pub by_category: Vec<(String, Rate)>,
}

impl MetaEval {
    /// A multi-line report: every rate with its Wilson 95% interval.
    pub fn render(&self, label: &str) -> String {
        let mut out = format!(
            "[{label}] n={} (in-scope {}, out-of-scope {}, combo {})\n",
            self.samples, self.in_scope_top1.n, self.oos_unsure_recall.n, self.combo_traces
        );
        out.push_str(&format!(
            "  overall top-1            {}/{} = {:.3} [{:.3}, {:.3}]   set-F1 {:.3}  weight-L1 {:.3}  ECE {:.3}\n",
            self.top1_correct,
            self.samples,
            self.top1_accuracy,
            self.top1_wilson95.0,
            self.top1_wilson95.1,
            self.mean_set_f1,
            self.mean_weight_l1,
            self.ece_top1
        ));
        for (name, rate) in [
            ("in-scope top-1 (argmax)", &self.in_scope_top1),
            ("in-scope routed", &self.in_scope_routed),
            ("in-scope routed+correct", &self.in_scope_routed_correct),
            ("OOS unsure recall", &self.oos_unsure_recall),
            ("OOS unsure precision", &self.oos_unsure_precision),
        ] {
            out.push_str(&format!("  {name:<24} {}\n", rate.render()));
        }
        out.push_str(&format!(
            "  combo set-F1             {:.3} (n={})   balanced accuracy {:.3}   unsure {}\n",
            self.combo_set_f1, self.combo_traces, self.balanced_accuracy, self.unsure
        ));
        for style in &self.by_style {
            out.push_str(&format!(
                "  style {:<18} top-1 {}  set-F1 {:.3}  unsure {}\n",
                style.style,
                style.top1.render(),
                style.mean_set_f1,
                style.unsure
            ));
        }
        out.push_str(&format!(
            "  majority baseline ({}): top-1 {:.3}\n",
            self.majority_category.as_deref().unwrap_or("-"),
            self.majority_top1_accuracy
        ));
        out
    }

    /// `category rate` lines, weakest first.
    pub fn render_categories(&self) -> String {
        let mut rows: Vec<&(String, Rate)> = self.by_category.iter().collect();
        rows.sort_by(|a, b| a.1.rate.total_cmp(&b.1.rate));
        rows.iter()
            .map(|(c, r)| format!("  {c:<22} {}\n", r.render()))
            .collect()
    }
}

/// One breakdown group's metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleEval {
    pub style: String,
    /// Top-1 as in [`MetaEval::top1_accuracy`] (for out-of-scope traces:
    /// abstained).
    pub top1: Rate,
    pub mean_set_f1: f64,
    pub unsure: usize,
}

/// A proportion with its Wilson 95% score interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rate {
    pub hits: usize,
    pub n: usize,
    /// `hits / n`; `NaN` when `n` is 0.
    pub rate: f64,
    pub wilson95: (f64, f64),
}

impl Rate {
    pub fn new(hits: usize, n: usize) -> Self {
        Self {
            hits,
            n,
            rate: if n == 0 {
                f64::NAN
            } else {
                hits as f64 / n as f64
            },
            wilson95: wilson_interval(hits, n),
        }
    }

    /// `hits/n = rate [lo, hi]`.
    pub fn render(&self) -> String {
        format!(
            "{}/{} = {:.3} [{:.3}, {:.3}]",
            self.hits, self.n, self.rate, self.wilson95.0, self.wilson95.1
        )
    }
}

/// The Wilson score interval for `hits` successes in `n` trials at 95%
/// (z = 1.96): unlike the normal approximation it stays inside `[0, 1]` and
/// is honest at small `n` and at rates near 0 or 1. `(0, 1)` when `n` is 0.
pub fn wilson_interval(hits: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.959_963_984_540_054_f64;
    let nf = n as f64;
    let p = hits.min(n) as f64 / nf;
    let z2 = z * z;
    let denominator = 1.0 + z2 / nf;
    let center = (p + z2 / (2.0 * nf)) / denominator;
    let half = z * (p * (1.0 - p) / nf + z2 / (4.0 * nf * nf)).sqrt() / denominator;
    // At 0 or n successes the exact bound is 0 or 1; the closed form only
    // reaches it up to rounding.
    let lo = if hits == 0 {
        0.0
    } else {
        (center - half).max(0.0)
    };
    let hi = if hits >= n {
        1.0
    } else {
        (center + half).min(1.0)
    };
    (lo, hi)
}

/// One evaluated trace: `(group, in scope, combo, abstained, top-1 correct,
/// set-F1)`.
type EvalRow = (String, bool, bool, bool, bool, f64);

/// The most frequent top category of `traces` (first in trace order on a tie).
pub fn majority_category(traces: &[RouterTrace]) -> Option<String> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for t in traces {
        if let Some(top) = t.gold_top() {
            match counts.iter_mut().find(|(id, _)| id == top) {
                Some(entry) => entry.1 += 1,
                None => counts.push((top.to_string(), 1)),
            }
        }
    }
    counts
        .into_iter()
        .fold(None, |best: Option<(String, usize)>, c| match best {
            Some(b) if b.1 >= c.1 => Some(b),
            _ => Some(c),
        })
        .map(|(id, _)| id)
}

fn set_f1(predicted: &BTreeSet<&str>, gold: &BTreeSet<&str>) -> f64 {
    if predicted.is_empty() && gold.is_empty() {
        return 1.0;
    }
    let hit = predicted.intersection(gold).count() as f64;
    2.0 * hit / (predicted.len() + gold.len()) as f64
}

/// Evaluate `router` on `traces`; `majority` is the training set's
/// [`majority_category`].
pub fn evaluate_meta(
    router: &dyn MetaRouter,
    traces: &[RouterTrace],
    majority: Option<&str>,
) -> Result<MetaEval> {
    let mut top1_correct = 0usize;
    let mut f1_sum = 0.0;
    let mut l1_sum = 0.0;
    let mut unsure = 0usize;
    let mut majority_hits = 0usize;
    let mut confidences: Vec<(f64, bool)> = Vec::with_capacity(traces.len());
    // (group, in scope, combo, abstained, top-1 correct, set-F1) per trace.
    let mut rows: Vec<EvalRow> = Vec::with_capacity(traces.len());
    // (gold top category, hits, n) for in-scope traces.
    let mut per_category: Vec<(String, usize, usize)> = Vec::new();
    for t in traces {
        let d = router.decide(&t.text, t.context.as_deref())?;
        if d.unsure.is_some() {
            unsure += 1;
        }
        let top = d.scores.first();
        let correct = match t.gold_top() {
            Some(gold) => top.is_some_and(|(id, _)| id == gold),
            None => d.unsure.is_some(),
        };
        if correct {
            top1_correct += 1;
        }
        if t.gold_top().is_some() && t.gold_top() == majority {
            majority_hits += 1;
        }
        confidences.push((top.map_or(0.0, |s| s.1), correct));
        let predicted: BTreeSet<&str> = d.engaged.iter().map(|(id, _)| id.as_str()).collect();
        let f1 = set_f1(&predicted, &t.gold_set());
        f1_sum += f1;
        if let Some(gold) = t.gold_top() {
            match per_category.iter_mut().find(|(c, _, _)| c == gold) {
                Some(entry) => {
                    entry.1 += usize::from(correct);
                    entry.2 += 1;
                }
                None => per_category.push((gold.to_string(), usize::from(correct), 1)),
            }
        }
        rows.push((
            t.group().to_string(),
            !t.categories.is_empty(),
            t.categories.len() > 1,
            d.unsure.is_some(),
            correct,
            f1,
        ));
        let mut ids: BTreeSet<&str> = predicted.clone();
        ids.extend(t.gold_set());
        l1_sum += ids
            .iter()
            .map(|id| {
                let p = d
                    .engaged
                    .iter()
                    .find(|(c, _)| c == id)
                    .map_or(0.0, |(_, w)| *w);
                let g = t
                    .categories
                    .iter()
                    .find(|c| c.id == *id)
                    .map_or(0.0, |c| c.weight);
                (p - g).abs()
            })
            .sum::<f64>();
    }
    let n = traces.len();
    let frac = |x: f64| if n == 0 { f64::NAN } else { x / n as f64 };
    // ECE over 5 equal-width confidence bins.
    let mut ece = 0.0;
    for bin in 0..5 {
        let lo = f64::from(bin) / 5.0;
        let hi = f64::from(bin + 1) / 5.0;
        let members: Vec<&(f64, bool)> = confidences
            .iter()
            .filter(|(c, _)| *c >= lo && (*c < hi || (bin == 4 && *c <= hi)))
            .collect();
        if members.is_empty() {
            continue;
        }
        let m = members.len() as f64;
        let acc = members.iter().filter(|(_, ok)| *ok).count() as f64 / m;
        let conf = members.iter().map(|(c, _)| c).sum::<f64>() / m;
        ece += m * (acc - conf).abs();
    }
    let count = |pred: &dyn Fn(&EvalRow) -> bool| rows.iter().filter(|r| pred(r)).count();
    let n_in = count(&|r| r.1);
    let n_oos = n - n_in;
    let in_scope_routed = Rate::new(count(&|r| r.1 && !r.3), n_in);
    let oos_unsure_recall = Rate::new(count(&|r| !r.1 && r.3), n_oos);
    let combo: Vec<f64> = rows.iter().filter(|r| r.2).map(|r| r.5).collect();
    let mut groups: Vec<String> = Vec::new();
    for r in &rows {
        if !groups.contains(&r.0) {
            groups.push(r.0.clone());
        }
    }
    let by_style = groups
        .into_iter()
        .map(|style| {
            let members: Vec<&EvalRow> = rows.iter().filter(|r| r.0 == style).collect();
            let m = members.len();
            StyleEval {
                top1: Rate::new(members.iter().filter(|r| r.4).count(), m),
                mean_set_f1: members.iter().map(|r| r.5).sum::<f64>() / m.max(1) as f64,
                unsure: members.iter().filter(|r| r.3).count(),
                style,
            }
        })
        .collect();
    Ok(MetaEval {
        samples: n,
        top1_correct,
        top1_accuracy: frac(top1_correct as f64),
        mean_set_f1: frac(f1_sum),
        mean_weight_l1: frac(l1_sum),
        unsure,
        ece_top1: frac(ece),
        majority_category: majority.map(str::to_string),
        majority_top1_accuracy: frac(majority_hits as f64),
        top1_wilson95: wilson_interval(top1_correct, n),
        in_scope_top1: Rate::new(count(&|r| r.1 && r.4), n_in),
        in_scope_routed_correct: Rate::new(count(&|r| r.1 && !r.3 && r.4), n_in),
        oos_unsure_precision: Rate::new(count(&|r| !r.1 && r.3), unsure),
        balanced_accuracy: 0.5 * in_scope_routed.rate + 0.5 * oos_unsure_recall.rate,
        in_scope_routed,
        oos_unsure_recall,
        combo_set_f1: if combo.is_empty() {
            f64::NAN
        } else {
            combo.iter().sum::<f64>() / combo.len() as f64
        },
        combo_traces: combo.len(),
        by_style,
        by_category: per_category
            .into_iter()
            .map(|(c, hits, n)| (c, Rate::new(hits, n)))
            .collect(),
    })
}

// ------------------------------------------------------- L1: catalogue --

/// One routable specialist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Specialist {
    /// Registry name or expert-index id.
    pub id: String,
    /// Taxonomy category it serves.
    pub category: String,
    /// Taxonomy leaf, for a sub-field specialist; `None` for a generalist.
    pub leaf: Option<String>,
    /// Whether L2 may give it traffic.
    pub enabled: bool,
    /// `active` / `candidate` (registry), `index`, or `declared` (roster).
    pub status: String,
    /// Content hash of its weights, when the source records one.
    pub checkpoint: Option<String>,
    /// `registry`, `expert-index` or `roster`.
    pub source: String,
}

/// The L1 layer: every specialist the router can resolve, plus the ones it
/// could not place and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Catalogue {
    pub source: String,
    pub specialists: Vec<Specialist>,
    /// `(name, reason)` for every entry left out.
    pub unmapped: Vec<(String, String)>,
}

impl Catalogue {
    fn push(&mut self, specialist: Specialist) {
        if self.specialists.iter().any(|s| s.id == specialist.id) {
            self.unmapped
                .push((specialist.id, "duplicate specialist id".to_string()));
        } else {
            self.specialists.push(specialist);
        }
    }

    /// From the nina registry. Retired entries are left out; candidates are
    /// routable only when `admit_candidates` (the shipped v1 specialists are
    /// all candidates). Names are placed with [`Taxonomy::map_specialist`].
    pub fn from_registry(file: &RegistryFile, taxonomy: &Taxonomy, admit_candidates: bool) -> Self {
        let mut out = Self {
            source: "registry".to_string(),
            specialists: Vec::new(),
            unmapped: Vec::new(),
        };
        for entry in &file.models {
            if entry.status == ModelStatus::Retired {
                out.unmapped
                    .push((entry.name.clone(), "retired".to_string()));
                continue;
            }
            match taxonomy.map_specialist(&entry.name) {
                SpecialistMapping::Unmapped { reason } => {
                    out.unmapped.push((entry.name.clone(), reason));
                }
                mapping => {
                    let category = mapping.category().unwrap_or_default().to_string();
                    out.push(Specialist {
                        id: entry.name.clone(),
                        category,
                        leaf: mapping.leaf().map(str::to_string),
                        enabled: entry.status == ModelStatus::Active
                            || (admit_candidates && entry.status == ModelStatus::Candidate),
                        status: entry.status.as_str().to_string(),
                        checkpoint: (!entry.checkpoint_hash.is_empty())
                            .then(|| entry.checkpoint_hash.clone()),
                        source: "registry".to_string(),
                    });
                }
            }
        }
        out
    }

    /// From an expert index carrying taxonomy fields; an expert with no
    /// category (on itself or its box) is reported, not placed.
    pub fn from_index(index: &ExpertIndex, taxonomy: &Taxonomy) -> Result<Self> {
        index.validate_with_taxonomy(taxonomy)?;
        let mut out = Self {
            source: "expert-index".to_string(),
            specialists: Vec::new(),
            unmapped: Vec::new(),
        };
        for entry in &index.boxes {
            for expert in &entry.experts {
                match index.expert_category(entry, expert) {
                    None => out.unmapped.push((
                        expert.id.clone(),
                        "no taxonomy category on the expert or its box".to_string(),
                    )),
                    Some(category) => out.push(Specialist {
                        id: expert.id.clone(),
                        category: category.to_string(),
                        leaf: expert.subfield.clone(),
                        enabled: expert.enabled,
                        status: "index".to_string(),
                        checkpoint: Some(expert.weights.sha256.clone()),
                        source: "expert-index".to_string(),
                    }),
                }
            }
        }
        Ok(out)
    }

    /// From the shipped roster ([`crate::taxonomy::nina_roster`]): declared
    /// specialists with no resolved checkpoint, for routing before a
    /// registry exists.
    pub fn from_roster(taxonomy: &Taxonomy) -> Self {
        let mut out = Self {
            source: "roster".to_string(),
            specialists: Vec::new(),
            unmapped: Vec::new(),
        };
        for line in taxonomy.map_roster().entries {
            match line.mapping {
                SpecialistMapping::Unmapped { reason } => {
                    out.unmapped.push((line.entry.name, reason));
                }
                mapping => {
                    let category = mapping.category().unwrap_or_default().to_string();
                    out.push(Specialist {
                        id: line.entry.name,
                        category,
                        leaf: mapping.leaf().map(str::to_string),
                        enabled: true,
                        status: "declared".to_string(),
                        checkpoint: None,
                        source: "roster".to_string(),
                    });
                }
            }
        }
        out
    }

    pub fn get(&self, id: &str) -> Option<&Specialist> {
        self.specialists.iter().find(|s| s.id == id)
    }

    pub fn in_category<'a>(&'a self, category: &'a str) -> impl Iterator<Item = &'a Specialist> {
        self.specialists
            .iter()
            .filter(move |s| s.category == category)
    }

    /// Categories with at least one enabled specialist, in taxonomy order.
    pub fn servable_categories(&self, taxonomy: &Taxonomy) -> Vec<String> {
        taxonomy
            .categories
            .iter()
            .filter(|c| self.in_category(&c.id).any(|s| s.enabled))
            .map(|c| c.id.clone())
            .collect()
    }
}

// ------------------------------------------------------ L2: categories --

/// What a category router sees.
#[derive(Debug, Clone, Copy)]
pub struct RouteQuery<'a> {
    pub prompt: &'a str,
    pub context: Option<&'a str>,
}

/// The middle layer: within one engaged category, which specialists and how
/// to blend them. Returns `(specialist id, gate)` pairs — gates non-negative
/// and summing to 1 — or `Ok(None)` when this router does not serve the
/// category (the next router is asked).
pub trait CategoryRouter {
    fn name(&self) -> &str;
    fn route(&self, query: &RouteQuery<'_>, category: &str) -> Result<Option<Vec<(String, f64)>>>;
}

/// The L2 backend: the crate's CPU backend. Routing a prompt is one small
/// matmul per engaged box.
type L2 = NdArray<f32>;

/// Zero every finite entry of every float parameter. The MoSME masks are
/// `0` (enabled) or `-inf` (disabled), so they survive unchanged while every
/// router weight and bias becomes `0`.
struct ZeroFinite;

impl<B: Backend> ModuleMapper<B> for ZeroFinite {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let finite = tensor.clone().is_finite();
        Param::from_mapped_value(id, tensor.mask_fill(finite, 0.0), mapper)
    }
}

/// L2 backed by MoSME: one box per category, the category's specialists as
/// its experts, gated by [`crate::mosme::HierarchicalRouter`].
pub struct MosmeCategoryRouter {
    router: crate::mosme::HierarchicalRouter<L2>,
    spec: MosmeSpec,
    features: FeatureConfig,
    device: <L2 as burn::tensor::backend::BackendTypes>::Device,
    trained: bool,
}

impl MosmeCategoryRouter {
    /// The box spec for `catalogue`: one box per servable category (taxonomy
    /// order), every specialist of that category as an expert (disabled ones
    /// masked), at most `top_k` experts gated per box.
    pub fn spec_for(catalogue: &Catalogue, taxonomy: &Taxonomy, top_k: usize) -> Result<MosmeSpec> {
        let boxes: Vec<BoxSpec> = catalogue
            .servable_categories(taxonomy)
            .into_iter()
            .map(|category| {
                let experts = catalogue
                    .in_category(&category)
                    .map(|s| {
                        let spec = ExpertSpec::new(s.id.clone(), s.id.clone());
                        if s.enabled {
                            spec
                        } else {
                            spec.disabled()
                        }
                    })
                    .collect();
                BoxSpec::new(category.clone(), category, experts)
            })
            .collect();
        ensure!(
            !boxes.is_empty(),
            "the catalogue ({}) has no category with an enabled specialist",
            catalogue.source
        );
        let spec = MosmeSpec {
            boxes,
            top_box: 1,
            top_expert: top_k.max(1),
            route_on_tokens: false,
            balance: BalanceWeights::default(),
        };
        spec.validate()?;
        Ok(spec)
    }

    /// A router with zero weights: uniform gates over the first `top_k`
    /// enabled specialists of each box, in catalogue order. The honest prior
    /// until a trained router exists.
    pub fn uniform_prior(
        catalogue: &Catalogue,
        taxonomy: &Taxonomy,
        top_k: usize,
        features: FeatureConfig,
    ) -> Result<Self> {
        features.validate()?;
        let spec = Self::spec_for(catalogue, taxonomy, top_k)?;
        let config = crate::mosme::MosmeConfig::new(1, features.dim(), spec.clone());
        let device: <L2 as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let router = crate::mosme::HierarchicalRouter::<L2>::new(&config, &device);
        Ok(Self {
            router: router.map(&mut ZeroFinite),
            spec,
            features,
            device,
            trained: false,
        })
    }

    /// Replace the gating module with a trained one built for the same spec.
    pub fn with_router(mut self, router: crate::mosme::HierarchicalRouter<L2>) -> Result<Self> {
        ensure!(
            router.experts_per_box() == self.spec.experts_per_box(),
            "the supplied MoSME router has boxes {:?}, the catalogue spec {:?}",
            router.experts_per_box(),
            self.spec.experts_per_box()
        );
        self.router = router;
        self.trained = true;
        Ok(self)
    }

    pub fn spec(&self) -> &MosmeSpec {
        &self.spec
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

impl CategoryRouter for MosmeCategoryRouter {
    fn name(&self) -> &str {
        if self.trained {
            "mosme"
        } else {
            "mosme(uniform-prior)"
        }
    }

    fn route(&self, query: &RouteQuery<'_>, category: &str) -> Result<Option<Vec<(String, f64)>>> {
        let Some(box_idx) = self.spec.boxes.iter().position(|b| b.id == category) else {
            return Ok(None);
        };
        let dim = self.features.dim();
        let dense = densify(&featurize(query.prompt, query.context, &self.features), dim);
        let input = Tensor::<L2, 1>::from_floats(dense.as_slice(), &self.device).reshape([1, dim]);
        let gates = self.router.route(input);
        let row = gates
            .expert_gates
            .get(box_idx)
            .with_context(|| format!("MoSME produced no gates for box {box_idx}"))?;
        let values: Vec<f32> = row.clone().into_data().iter::<f32>().collect();
        let experts = self
            .spec
            .boxes
            .get(box_idx)
            .map_or(&[][..], |b| b.experts.as_slice());
        let mut picked: Vec<(String, f64)> = experts
            .iter()
            .zip(&values)
            .filter(|(_, g)| **g > 0.0)
            .map(|(e, g)| (e.id.clone(), f64::from(*g)))
            .collect();
        picked.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(Some(normalize_pairs(picked)))
    }
}

/// L2 from an already-computed MoSME box readout: per-leaf traffic (the
/// student probe router's per-language box traffic) for one category.
pub struct TrafficCategoryRouter {
    name: String,
    category: String,
    traffic: Vec<(String, f64)>,
    top_k: usize,
}

impl TrafficCategoryRouter {
    /// From `(box id, traffic)` pairs whose box ids are `<field>:<sub>`
    /// (`coding:rust`). Each box is placed with [`Taxonomy::map_specialist`]
    /// on `<field>/<sub>`; traffic on a leaf goes to every enabled specialist
    /// of `category` serving that leaf. `None` when no traffic lands.
    pub fn from_box_traffic(
        name: &str,
        category: &str,
        box_traffic: &[(String, f64)],
        taxonomy: &Taxonomy,
        catalogue: &Catalogue,
        top_k: usize,
    ) -> Option<Self> {
        let mut traffic: Vec<(String, f64)> = Vec::new();
        for (box_id, t) in box_traffic {
            if !t.is_finite() || *t <= 0.0 {
                continue;
            }
            let registry_name = box_id.replacen(':', "/", 1);
            let mapping = taxonomy.map_specialist(&registry_name);
            let (Some(c), Some(leaf)) = (mapping.category(), mapping.leaf()) else {
                continue;
            };
            if c != category {
                continue;
            }
            for s in catalogue.in_category(category) {
                if s.enabled && s.leaf.as_deref() == Some(leaf) {
                    match traffic.iter_mut().find(|(id, _)| *id == s.id) {
                        Some(entry) => entry.1 += t,
                        None => traffic.push((s.id.clone(), *t)),
                    }
                }
            }
        }
        (!traffic.is_empty()).then(|| Self {
            name: name.to_string(),
            category: category.to_string(),
            traffic,
            top_k: top_k.max(1),
        })
    }
}

impl CategoryRouter for TrafficCategoryRouter {
    fn name(&self) -> &str {
        &self.name
    }

    fn route(&self, _query: &RouteQuery<'_>, category: &str) -> Result<Option<Vec<(String, f64)>>> {
        if category != self.category {
            return Ok(None);
        }
        let mut picked = self.traffic.clone();
        picked.sort_by(|a, b| b.1.total_cmp(&a.1));
        picked.truncate(self.top_k);
        Ok(Some(normalize_pairs(picked)))
    }
}

// ------------------------------------------------------------- the plan --

/// Which layer made a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteLayer {
    /// L3: category combination.
    L3Meta,
    /// L2: specialists within one category.
    L2Category,
    /// The cross-category specialist budget.
    PlannerBudget,
    /// L1: one specialist resolved to its weights.
    L1Specialist,
}

/// One decision, attributed to its layer and the router that made it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerDecision {
    pub layer: RouteLayer,
    pub decided_by: String,
    /// What was decided about: `prompt`, a category id, `specialists`, or a
    /// specialist id.
    pub scope: String,
    /// The choice, with weights (gates for L2, joint weights for the budget).
    pub chose: Vec<(String, f64)>,
    /// What else was considered (L3: the top scores).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub considered: Vec<(String, f64)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A routing decision for one prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutePlan {
    /// Engaged categories and their weights (sum to 1); empty when unsure.
    pub categories: Vec<(String, f64)>,
    /// `(specialist id, category, weight)`; weights sum to 1; empty when unsure.
    pub specialists: Vec<(String, String, f64)>,
    /// Why nothing was routed, when nothing was.
    pub unsure: Option<String>,
    /// Every decision, layer by layer.
    pub trace: Vec<LayerDecision>,
}

impl RoutePlan {
    fn unsure(reason: String, trace: Vec<LayerDecision>) -> Self {
        Self {
            categories: Vec::new(),
            specialists: Vec::new(),
            unsure: Some(reason),
            trace,
        }
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize route plan")
    }
}

/// Top-k bounds at each layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RouteBounds {
    pub max_categories: usize,
    pub top_specialists_per_category: usize,
    pub max_specialists: usize,
}

impl Default for RouteBounds {
    fn default() -> Self {
        Self {
            max_categories: 3,
            top_specialists_per_category: 2,
            max_specialists: 4,
        }
    }
}

impl RouteBounds {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_categories >= 1
                && self.top_specialists_per_category >= 1
                && self.max_specialists >= 1,
            "every route bound must be at least 1: {self:?}"
        );
        Ok(())
    }
}

/// Violations of the plan invariants, as residuals (all zero for a valid plan).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlanCheck {
    /// `|sum(category weights) - 1|`.
    pub category_sum_error: f64,
    /// `|sum(specialist weights) - 1|`.
    pub specialist_sum_error: f64,
    /// Largest `|sum(gates) - 1|` over the L2 decisions in the trace.
    pub l2_sum_error: f64,
    /// Negative, NaN or infinite weights anywhere in the plan or its L2 gates.
    pub invalid_weights: usize,
    /// Specialists whose category is not a chosen category.
    pub orphans: usize,
    /// Chosen categories with no specialist.
    pub empty_categories: usize,
    /// Largest `|sum(weights of a category's specialists) - category weight|`.
    pub mass_error: f64,
    /// Repeated category or specialist ids.
    pub duplicates: usize,
    /// Top-k bounds exceeded, or an unsure plan that still routes something
    /// (or a routed plan that routes nothing).
    pub bound_violations: usize,
}

impl PlanCheck {
    /// Distribution residual: the largest sum error, or infinity when any
    /// weight is invalid (so a NaN can never pass as a small error).
    pub fn distribution_residual(&self) -> f64 {
        if self.invalid_weights > 0 {
            return f64::INFINITY;
        }
        [
            self.category_sum_error,
            self.specialist_sum_error,
            self.l2_sum_error,
        ]
        .into_iter()
        .fold(0.0, worst)
    }

    /// Hierarchy residual: structural counts plus the mass error.
    pub fn hierarchy_residual(&self) -> f64 {
        (self.orphans + self.empty_categories + self.duplicates + self.bound_violations) as f64
            + self.mass_error
    }
}

/// Tolerance on every sum in a plan: f64 renormalization leaves ~1e-16.
pub const PLAN_TOLERANCE: f64 = 1e-9;

/// Measure a plan against the invariants.
pub fn check_plan(plan: &RoutePlan, bounds: &RouteBounds) -> PlanCheck {
    let mut check = PlanCheck::default();
    let bad = |w: f64| !w.is_finite() || w < 0.0;
    check.invalid_weights += plan.categories.iter().filter(|(_, w)| bad(*w)).count();
    check.invalid_weights += plan.specialists.iter().filter(|(_, _, w)| bad(*w)).count();

    if plan.unsure.is_some() {
        if !plan.categories.is_empty() || !plan.specialists.is_empty() {
            check.bound_violations += 1;
        }
        return check;
    }
    if plan.categories.is_empty() || plan.specialists.is_empty() {
        check.bound_violations += 1;
    }
    check.category_sum_error = (plan.categories.iter().map(|c| c.1).sum::<f64>() - 1.0).abs();
    check.specialist_sum_error = (plan.specialists.iter().map(|s| s.2).sum::<f64>() - 1.0).abs();

    let category_ids: BTreeSet<&str> = plan.categories.iter().map(|c| c.0.as_str()).collect();
    check.duplicates += plan.categories.len() - category_ids.len();
    let specialist_ids: BTreeSet<&str> = plan.specialists.iter().map(|s| s.0.as_str()).collect();
    check.duplicates += plan.specialists.len() - specialist_ids.len();
    check.orphans = plan
        .specialists
        .iter()
        .filter(|s| !category_ids.contains(s.1.as_str()))
        .count();
    for (category, weight) in &plan.categories {
        let members: Vec<f64> = plan
            .specialists
            .iter()
            .filter(|s| s.1 == *category)
            .map(|s| s.2)
            .collect();
        if members.is_empty() {
            check.empty_categories += 1;
        }
        if members.len() > bounds.top_specialists_per_category {
            check.bound_violations += 1;
        }
        let err = (members.iter().sum::<f64>() - weight).abs();
        check.mass_error = worst(check.mass_error, err);
    }
    if plan.categories.len() > bounds.max_categories {
        check.bound_violations += 1;
    }
    if plan.specialists.len() > bounds.max_specialists {
        check.bound_violations += 1;
    }
    for d in plan
        .trace
        .iter()
        .filter(|d| d.layer == RouteLayer::L2Category && !d.chose.is_empty())
    {
        check.invalid_weights += d.chose.iter().filter(|(_, g)| bad(*g)).count();
        let err = (d.chose.iter().map(|c| c.1).sum::<f64>() - 1.0).abs();
        check.l2_sum_error = worst(check.l2_sum_error, err);
    }
    check
}

/// Refuse a plan that breaks any invariant, naming the first broken one.
pub fn validate_plan(plan: &RoutePlan, bounds: &RouteBounds) -> Result<()> {
    let c = check_plan(plan, bounds);
    ensure!(
        c.invalid_weights == 0,
        "plan carries {} negative or non-finite weights",
        c.invalid_weights
    );
    let dist = c.distribution_residual();
    ensure!(
        dist.is_finite() && dist <= PLAN_TOLERANCE,
        "plan weights are not distributions (largest sum error {dist})"
    );
    ensure!(
        c.orphans == 0,
        "{} specialists sit in unchosen categories",
        c.orphans
    );
    ensure!(
        c.empty_categories == 0,
        "{} chosen categories have no specialist",
        c.empty_categories
    );
    ensure!(c.duplicates == 0, "plan repeats {} ids", c.duplicates);
    ensure!(
        c.mass_error.is_finite() && c.mass_error <= PLAN_TOLERANCE,
        "a category's specialists do not carry its weight (error {})",
        c.mass_error
    );
    ensure!(
        c.bound_violations == 0,
        "plan breaks {} top-k / outcome bounds",
        c.bound_violations
    );
    Ok(())
}

// ------------------------------------------------------ the full router --

/// Where a [`LayeredRouter`] comes from on disk.
#[derive(Debug, Clone, Default)]
pub struct LayeredRouterConfig {
    /// A trained [`HashedMetaRouter`] JSON; without one L3 always abstains.
    pub meta_router: Option<PathBuf>,
    /// A taxonomy patch merged over the built-in.
    pub taxonomy_patch: Option<PathBuf>,
    /// An expert index with taxonomy fields (takes precedence).
    pub index: Option<PathBuf>,
    /// A nina `registry.json`.
    pub registry: Option<PathBuf>,
    /// Route to registry candidates, not only active models.
    pub admit_candidates: bool,
    pub bounds: RouteBounds,
}

/// One engaged category L2 could serve: its id, its L3 weight and its
/// specialist gates.
type ServedCategory = (String, f64, Vec<(String, f64)>);

/// The three layers wired together.
pub struct LayeredRouter {
    taxonomy: Taxonomy,
    meta: Box<dyn MetaRouter>,
    category_routers: Vec<Box<dyn CategoryRouter>>,
    catalogue: Catalogue,
    bounds: RouteBounds,
}

impl LayeredRouter {
    /// Wire `meta` over `catalogue`, with the MoSME uniform-prior L2.
    pub fn new(
        taxonomy: Taxonomy,
        meta: Box<dyn MetaRouter>,
        catalogue: Catalogue,
        bounds: RouteBounds,
    ) -> Result<Self> {
        bounds.validate()?;
        taxonomy.validate()?;
        for id in meta.categories() {
            ensure!(
                taxonomy.category(id).is_some(),
                "the meta-router outputs category '{id}', which taxonomy {} does not have",
                taxonomy.version
            );
        }
        // An empty catalogue is a legitimate state (a fresh registry): with no
        // MoSME boxes to build, every engaged category is reported unserved.
        let mut category_routers: Vec<Box<dyn CategoryRouter>> = Vec::new();
        if !catalogue.servable_categories(&taxonomy).is_empty() {
            category_routers.push(Box::new(MosmeCategoryRouter::uniform_prior(
                &catalogue,
                &taxonomy,
                bounds.top_specialists_per_category,
                FeatureConfig::tiny(),
            )?));
        }
        Ok(Self {
            taxonomy,
            meta,
            category_routers,
            catalogue,
            bounds,
        })
    }

    /// Build from files; see [`LayeredRouterConfig`]. The catalogue comes from
    /// the index, else the registry, else the shipped roster.
    pub fn from_config(config: &LayeredRouterConfig) -> Result<Self> {
        let taxonomy = Taxonomy::builtin_with_patch(config.taxonomy_patch.as_deref())?;
        let catalogue = match (&config.index, &config.registry) {
            (Some(index), _) => Catalogue::from_index(&ExpertIndex::read(index)?, &taxonomy)?,
            (None, Some(registry)) => Catalogue::from_registry(
                &RegistryFile::load(registry)?,
                &taxonomy,
                config.admit_candidates,
            ),
            (None, None) => Catalogue::from_roster(&taxonomy),
        };
        let meta: Box<dyn MetaRouter> = match &config.meta_router {
            Some(path) => load_meta_router(path)?,
            None => Box::new(HashedMetaRouter::new(
                &taxonomy,
                FeatureConfig::tiny(),
                MetaTrainConfig::default(),
                MetaPolicy {
                    max_categories: config.bounds.max_categories,
                    ..MetaPolicy::default()
                },
            )?),
        };
        Self::new(taxonomy, meta, catalogue, config.bounds)
    }

    /// Ask `router` before the built-in L2 routers.
    pub fn with_category_router_first(mut self, router: Box<dyn CategoryRouter>) -> Self {
        self.category_routers.insert(0, router);
        self
    }

    pub fn taxonomy(&self) -> &Taxonomy {
        &self.taxonomy
    }

    pub fn catalogue(&self) -> &Catalogue {
        &self.catalogue
    }

    pub fn bounds(&self) -> &RouteBounds {
        &self.bounds
    }

    pub fn meta_name(&self) -> &str {
        self.meta.name()
    }

    pub fn route(&self, prompt: &str) -> Result<RoutePlan> {
        self.route_with(
            &RouteQuery {
                prompt,
                context: None,
            },
            &[],
        )
    }

    /// Route with a memory context and per-call L2 routers consulted first
    /// (`nina route_prompt` passes the student probe's box readout here).
    pub fn route_with(
        &self,
        query: &RouteQuery<'_>,
        extra: &[&dyn CategoryRouter],
    ) -> Result<RoutePlan> {
        let mut trace = Vec::new();

        // ---- L3
        let decision = self.meta.decide(query.prompt, query.context)?;
        let engaged: Vec<(String, f64)> = normalize_pairs(
            decision
                .engaged
                .iter()
                .take(self.bounds.max_categories)
                .cloned()
                .collect(),
        );
        trace.push(LayerDecision {
            layer: RouteLayer::L3Meta,
            decided_by: self.meta.name().to_string(),
            scope: "prompt".to_string(),
            chose: engaged.clone(),
            considered: decision.scores.iter().take(5).cloned().collect(),
            note: Some(match &decision.unsure {
                Some(reason) => format!("unsure: {reason}"),
                None => format!("p_none={:.3}", decision.none),
            }),
        });
        if let Some(reason) = decision.unsure {
            return Ok(RoutePlan::unsure(reason, trace));
        }

        // ---- L2, per engaged category
        let mut routers: Vec<&dyn CategoryRouter> = extra.to_vec();
        routers.extend(
            self.category_routers
                .iter()
                .map(|r| r.as_ref() as &dyn CategoryRouter),
        );
        let mut served: Vec<ServedCategory> = Vec::new();
        let mut unserved: Vec<String> = Vec::new();
        for (category, weight) in &engaged {
            let mut answer: Option<(String, Vec<(String, f64)>)> = None;
            for router in &routers {
                if let Some(gates) = router.route(query, category)? {
                    let gates = self.admit_gates(category, gates);
                    if !gates.is_empty() {
                        answer = Some((router.name().to_string(), gates));
                        break;
                    }
                }
            }
            match answer {
                Some((by, gates)) => {
                    trace.push(LayerDecision {
                        layer: RouteLayer::L2Category,
                        decided_by: by,
                        scope: category.clone(),
                        chose: gates.clone(),
                        considered: Vec::new(),
                        note: None,
                    });
                    served.push((category.clone(), *weight, gates));
                }
                None => {
                    trace.push(LayerDecision {
                        layer: RouteLayer::L2Category,
                        decided_by: "none".to_string(),
                        scope: category.clone(),
                        chose: Vec::new(),
                        considered: Vec::new(),
                        note: Some(format!(
                            "no specialist in the {} catalogue serves this category",
                            self.catalogue.source
                        )),
                    });
                    unserved.push(category.clone());
                }
            }
        }
        if served.is_empty() {
            return Ok(RoutePlan::unsure(
                format!(
                    "no specialist in the {} catalogue serves the engaged categories ({})",
                    self.catalogue.source,
                    unserved.join(", ")
                ),
                trace,
            ));
        }

        // ---- budget across categories (planner beam search)
        let candidates: Vec<(usize, String, f64)> = {
            let mut all: Vec<(usize, String, f64)> = served
                .iter()
                .enumerate()
                .flat_map(|(ci, (_, w, gates))| {
                    gates.iter().map(move |(id, g)| (ci, id.clone(), w * g))
                })
                .collect();
            all.sort_by(|a, b| b.2.total_cmp(&a.2));
            all
        };
        let keep = self.bounds.max_specialists.min(candidates.len());
        let budget = crate::planner::Budget {
            max_evaluations: keep.saturating_mul(candidates.len()).max(1),
            max_depth: keep.saturating_sub(1),
            beam_width: 1,
        };
        let plan = crate::planner::Beam::new(budget).search(|path, _remaining| {
            if path.depth() >= keep {
                return Vec::new();
            }
            (0..candidates.len())
                .filter(|i| !path.steps.contains(i))
                .filter_map(|i| candidates.get(i).map(|c| (i, c.2)))
                .collect()
        });
        let selected: Vec<usize> = plan.best.map(|p| p.steps).unwrap_or_default();
        trace.push(LayerDecision {
            layer: RouteLayer::PlannerBudget,
            decided_by: "planner::Beam".to_string(),
            scope: "specialists".to_string(),
            chose: selected
                .iter()
                .filter_map(|i| candidates.get(*i).map(|c| (c.1.clone(), c.2)))
                .collect(),
            considered: Vec::new(),
            note: Some(format!(
                "kept {} of {} candidates (max_specialists {}), {} evaluations{}",
                selected.len(),
                candidates.len(),
                self.bounds.max_specialists,
                plan.evaluations,
                if plan.budget_exhausted {
                    ", budget exhausted"
                } else {
                    ""
                }
            )),
        });

        // ---- assemble: drop categories that lost every specialist
        let survivors: Vec<ServedCategory> = served
            .iter()
            .enumerate()
            .filter_map(|(ci, (category, weight, gates))| {
                let kept: Vec<(String, f64)> = gates
                    .iter()
                    .filter(|(id, _)| {
                        selected
                            .iter()
                            .filter_map(|i| candidates.get(*i))
                            .any(|c| c.0 == ci && c.1 == *id)
                    })
                    .cloned()
                    .collect();
                (!kept.is_empty()).then(|| (category.clone(), *weight, normalize_pairs(kept)))
            })
            .collect();
        let total: f64 = survivors.iter().map(|s| s.1).sum();
        if survivors.is_empty() || total.is_nan() || total <= 0.0 {
            return Ok(RoutePlan::unsure(
                "the specialist budget left no category served".to_string(),
                trace,
            ));
        }
        let categories: Vec<(String, f64)> = survivors
            .iter()
            .map(|(c, w, _)| (c.clone(), w / total))
            .collect();
        let specialists: Vec<(String, String, f64)> = survivors
            .iter()
            .flat_map(|(c, w, gates)| {
                gates
                    .iter()
                    .map(move |(id, g)| (id.clone(), c.clone(), w / total * g))
            })
            .collect();

        // ---- L1 resolution
        for (id, _, weight) in &specialists {
            let resolved = self.catalogue.get(id);
            trace.push(LayerDecision {
                layer: RouteLayer::L1Specialist,
                decided_by: resolved
                    .map_or("unknown", |s| s.source.as_str())
                    .to_string(),
                scope: id.clone(),
                chose: vec![(
                    resolved
                        .and_then(|s| s.checkpoint.clone())
                        .unwrap_or_else(|| "unresolved".to_string()),
                    *weight,
                )],
                considered: Vec::new(),
                note: resolved.map(|s| {
                    format!(
                        "status {}, {}",
                        s.status,
                        s.leaf
                            .as_deref()
                            .map_or("category generalist".to_string(), |l| format!("leaf {l}"))
                    )
                }),
            });
        }

        Ok(RoutePlan {
            categories,
            specialists,
            unsure: None,
            trace,
        })
    }

    /// Keep only well-formed gates for enabled specialists of `category`,
    /// truncated to the per-category bound and renormalized. A buggy L2
    /// router can therefore never put a specialist in the wrong category or
    /// an invalid weight in the plan.
    fn admit_gates(&self, category: &str, gates: Vec<(String, f64)>) -> Vec<(String, f64)> {
        let mut seen = BTreeSet::new();
        let mut kept: Vec<(String, f64)> = gates
            .into_iter()
            .filter(|(id, g)| {
                g.is_finite()
                    && *g > 0.0
                    && self
                        .catalogue
                        .get(id)
                        .is_some_and(|s| s.enabled && s.category == category)
            })
            .filter(|(id, _)| seen.insert(id.clone()))
            .collect();
        kept.sort_by(|a, b| b.1.total_cmp(&a.1));
        kept.truncate(self.bounds.top_specialists_per_category);
        normalize_pairs(kept)
    }
}

// --------------------------------------- L3: a frozen-trunk meta-router --

/// The encoder backend: the crate's CPU backend, inference only.
type Enc = NdArray<f32>;

/// A frozen nina language-model trunk ([`crate::lm::LanguageModel`], the
/// trunk the specialist student routes from) used as a text encoder.
pub struct LmEncoder {
    model: crate::lm::LanguageModel<Enc>,
    device: <Enc as burn::tensor::backend::BackendTypes>::Device,
    context: usize,
    checkpoint: PathBuf,
    sha256: String,
}

impl LmEncoder {
    /// Load an `lm` checkpoint. Its architecture is read from the
    /// training-state sidecar beside it, and a checkpoint without one is
    /// refused: a guessed architecture whose shapes happen to agree would
    /// load and silently be a different model.
    pub fn load(checkpoint: &Path) -> Result<Self> {
        let state = crate::checkpoint::TrainState::for_model(checkpoint)?.with_context(|| {
            format!(
                "no training-state sidecar beside {}; the encoder's architecture is unknown",
                checkpoint.display()
            )
        })?;
        let value = state
            .config
            .get("model_config")
            .filter(|v| !v.is_null())
            .cloned()
            .with_context(|| {
                format!(
                    "the sidecar beside {} records no model_config (is it an `lm` checkpoint?)",
                    checkpoint.display()
                )
            })?;
        let config: crate::lm::LmConfig = serde_json::from_value(value)
            .with_context(|| format!("parse the model config beside {}", checkpoint.display()))?;
        let device: <Enc as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let model = crate::lm::LanguageModel::<Enc>::new(&config, &device)?;
        crate::tensor_ext::force_initialization(&model);
        let model = crate::checkpoint::load::<Enc, _>(model, checkpoint, &device)?;
        Ok(Self {
            model,
            device,
            context: config.context.max(1),
            checkpoint: checkpoint.to_path_buf(),
            sha256: crate::checkpoint::file_sha256_hex(checkpoint)?,
        })
    }

    pub fn checkpoint(&self) -> &Path {
        &self.checkpoint
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Mean over positions of every layer's hidden state, concatenated
    /// (`layers x hidden` values). The text is byte-tokenized and cut to the
    /// model's context.
    pub fn pooled(&self, text: &str) -> Result<Vec<f32>> {
        let mut tokens = crate::tokenizer::ByteTokenizer::new().encode_document(text);
        tokens.truncate(self.context);
        ensure!(!tokens.is_empty(), "the text tokenizes to zero tokens");
        let n = tokens.len();
        let ids: Vec<i64> = tokens.iter().map(|t| i64::from(*t)).collect();
        let input = Tensor::<Enc, 1, burn::tensor::Int>::from_ints(ids.as_slice(), &self.device)
            .reshape([1, n]);
        let mut out = Vec::new();
        for state in self.model.hidden_states(input) {
            out.extend(state.mean_dim(1).into_data().iter::<f32>());
        }
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "the encoder produced a non-finite hidden state"
        );
        Ok(out)
    }
}

/// A frozen text encoder whose fixed-width output a [`PooledMetaRouter`]
/// puts a linear head on.
pub trait TextEncoder {
    /// `lm` or `ncm`; recorded in the head so loading rebuilds the same encoder.
    fn kind(&self) -> &'static str;
    fn checkpoint(&self) -> &Path;
    fn sha256(&self) -> &str;
    /// Encoder-specific setting recorded in the head (NCM: coarse levels
    /// pooled; LM: 0).
    fn levels(&self) -> usize;
    fn encode(&self, text: &str) -> Result<Vec<f32>>;
}

impl TextEncoder for LmEncoder {
    fn kind(&self) -> &'static str {
        "lm"
    }

    fn checkpoint(&self) -> &Path {
        LmEncoder::checkpoint(self)
    }

    fn sha256(&self) -> &str {
        LmEncoder::sha256(self)
    }

    fn levels(&self) -> usize {
        0
    }

    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.pooled(text)
    }
}

/// NCM concept features (TRISKELE §13: routers trained with the NCM
/// pipeline): the frozen multiscale residual-VQ tokenizer
/// ([`crate::ncm::tokenizer::NcmTokenizer`], through the public
/// [`crate::ncm::ConceptCodec`] API) encodes the prompt, and the coarse
/// cumulative reconstructions `ẑ^(1..=c)` are mean-pooled over the prompt's
/// real (non-pad) positions and concatenated: `c x d` features. Coarse
/// levels are the "concept" summary the generator conditions on, so they
/// are the natural routing signal — if the tokenizer has learned one.
pub struct NcmEncoder {
    codec: crate::ncm::tokenizer::NcmTokenizer<Enc>,
    device: <Enc as burn::tensor::backend::BackendTypes>::Device,
    levels: usize,
    checkpoint: PathBuf,
    sha256: String,
}

impl NcmEncoder {
    /// Load an NCM tokenizer checkpoint with the architecture its training
    /// state records (refused without one, or when the state was written by
    /// a different trainer), frozen, pooling `levels` coarse levels.
    pub fn load(checkpoint: &Path, levels: usize) -> Result<Self> {
        use crate::ncm::ConceptCodec as _;
        let state = crate::checkpoint::TrainState::for_model(checkpoint)?.with_context(|| {
            format!(
                "no training state beside {}; the tokenizer's architecture is unknown",
                checkpoint.display()
            )
        })?;
        ensure!(
            state.kind == crate::ncm::tokenizer::train::STATE_KIND,
            "{} was written by the '{}' trainer, not the NCM tokenizer",
            checkpoint.display(),
            state.kind
        );
        state.verify_files(&crate::checkpoint::TrainState::dir_for(checkpoint))?;
        let config: crate::ncm::tokenizer::NcmTokenizerConfig = serde_json::from_value(
            state
                .config
                .get("tokenizer")
                .cloned()
                .context("the training state records no tokenizer config")?,
        )
        .context("parse the recorded tokenizer config")?;
        let device: <Enc as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let codec = crate::ncm::tokenizer::NcmTokenizer::<Enc>::new(&config, &device)?;
        crate::tensor_ext::force_initialization(&codec);
        let codec = crate::checkpoint::load::<Enc, _>(codec, checkpoint, &device)?;
        let available = codec.schedule().levels();
        ensure!(
            (1..=available).contains(&levels),
            "pool 1..={available} levels of this tokenizer, not {levels}"
        );
        Ok(Self {
            codec,
            device,
            levels,
            checkpoint: checkpoint.to_path_buf(),
            sha256: crate::checkpoint::file_sha256_hex(checkpoint)?,
        })
    }
}

impl TextEncoder for NcmEncoder {
    fn kind(&self) -> &'static str {
        "ncm"
    }

    fn checkpoint(&self) -> &Path {
        &self.checkpoint
    }

    fn sha256(&self) -> &str {
        &self.sha256
    }

    fn levels(&self) -> usize {
        self.levels
    }

    /// The text is byte-tokenized and cut into windows of the tokenizer's
    /// length `L` (the last padded with `<pad>`); each window's `ẑ^(c)` is
    /// summed over its real positions, and the sums are divided by the total
    /// real-position count.
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        use crate::ncm::ConceptCodec as _;
        let tokens = crate::tokenizer::ByteTokenizer::new().encode_document(text);
        ensure!(!tokens.is_empty(), "the text tokenizes to zero tokens");
        let seq_len = self.codec.schedule().seq_len();
        let d = self.codec.latent_dim();
        let pad = crate::tokenizer::Special::Pad.id();
        let mut sums = vec![0.0f64; self.levels * d];
        let mut real_total = 0usize;
        for window in tokens.chunks(seq_len.max(1)) {
            let real = window.len();
            let ids: Vec<i64> = window
                .iter()
                .copied()
                .chain(std::iter::repeat(pad))
                .take(seq_len)
                .map(i64::from)
                .collect();
            let input =
                Tensor::<Enc, 1, burn::tensor::Int>::from_ints(ids.as_slice(), &self.device)
                    .reshape([1, seq_len]);
            let codes = self.codec.encode(input)?;
            for level in 0..self.levels {
                let prefix = codes.codes.get(..=level).with_context(|| {
                    format!("the tokenizer returned fewer than {} levels", level + 1)
                })?;
                let zhat = self.codec.reconstruct(prefix, 1)?; // [1, L, d]
                let pooled: Vec<f32> = zhat
                    .narrow(1, 0, real)
                    .sum_dim(1)
                    .into_data()
                    .iter::<f32>()
                    .collect();
                for (k, v) in pooled.iter().enumerate() {
                    if let Some(slot) = sums.get_mut(level * d + k) {
                        *slot += f64::from(*v);
                    }
                }
            }
            real_total += real;
        }
        let out: Vec<f32> = sums
            .into_iter()
            .map(|v| (v / real_total.max(1) as f64) as f32)
            .collect();
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "the NCM tokenizer produced a non-finite reconstruction"
        );
        Ok(out)
    }
}

fn default_encoder_kind() -> String {
    "lm".to_string()
}

/// Current on-disk version of a [`PooledHead`].
pub const POOLED_ROUTER_VERSION: u32 = 1;

/// The trainable part of a [`PooledMetaRouter`]: a standardized linear
/// softmax head over the frozen encoder's pooled hidden state, plus the
/// identity of the encoder it was trained on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PooledHead {
    pub version: u32,
    /// Which [`TextEncoder`] produced the features (`lm` or `ncm`); heads
    /// written before the field existed are `lm`.
    #[serde(default = "default_encoder_kind")]
    pub encoder_kind: String,
    /// The encoder's level setting ([`TextEncoder::levels`]).
    #[serde(default)]
    pub encoder_levels: usize,
    pub encoder_checkpoint: PathBuf,
    /// The encoder file's sha256; loading refuses a different file.
    pub encoder_sha256: String,
    /// Pooled feature width (`layers x hidden`).
    pub dim: usize,
    /// Per-feature mean and scale over the fit slice (standardization).
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    pub train_config: MetaTrainConfig,
    pub policy: MetaPolicy,
    pub categories: Vec<String>,
    pub taxonomy_version: String,
    weights: Vec<f32>,
    bias: Vec<f32>,
    pub temperature: f64,
    pub trained_samples: usize,
    pub none_samples: usize,
}

/// The second [`MetaRouter`]: a frozen nina trunk's mean-pooled hidden
/// states as features, a linear softmax head on top, trained, calibrated and
/// tuned exactly like [`HashedMetaRouter`] (same SGD, same temperature
/// scaling, same policy search). How good it is depends entirely on the
/// encoder: a barely trained trunk gives near-random features, and the
/// comparison in `dblocks router train --encoder` says so with numbers.
pub struct PooledMetaRouter {
    head: PooledHead,
    encoder: Box<dyn TextEncoder>,
}

fn pooled_text(prompt: &str, context: Option<&str>) -> String {
    match context {
        Some(context) => format!("{prompt}\n{context}"),
        None => prompt.to_string(),
    }
}

impl PooledMetaRouter {
    /// Encode, standardize on the fit slice, fit the head, fit the
    /// temperature on the calibration slice ([`calibration_split`]).
    pub fn train(
        encoder: Box<dyn TextEncoder>,
        taxonomy: &Taxonomy,
        train_config: MetaTrainConfig,
        policy: MetaPolicy,
        traces: &[RouterTrace],
    ) -> Result<(Self, MetaTrainReport)> {
        train_config.validate()?;
        policy.validate()?;
        let categories = taxonomy.category_ids();
        check_trace_categories(&categories, traces)?;
        let (fit, calibration) = calibration_split(traces, &train_config);
        let encode = |set: &[RouterTrace]| -> Result<Vec<Vec<f32>>> {
            set.iter()
                .map(|t| encoder.encode(&pooled_text(&t.text, t.context.as_deref())))
                .collect()
        };
        let raw_fit = encode(&fit)?;
        let raw_calibration = encode(&calibration)?;
        let dim = raw_fit.first().map_or(0, Vec::len);
        ensure!(dim > 0, "the encoder produced empty features");
        let n = raw_fit.len().max(1) as f64;
        let mut mean = vec![0.0f32; dim];
        let mut scale = vec![1.0f32; dim];
        for d in 0..dim {
            let column: Vec<f64> = raw_fit
                .iter()
                .map(|row| f64::from(row.get(d).copied().unwrap_or(0.0)))
                .collect();
            let m = column.iter().sum::<f64>() / n;
            let var = column.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
            if let (Some(mean_d), Some(scale_d)) = (mean.get_mut(d), scale.get_mut(d)) {
                *mean_d = m as f32;
                *scale_d = var.sqrt().max(1e-6) as f32;
            }
        }
        let classes = categories.len() + 1;
        let mut head = PooledHead {
            version: POOLED_ROUTER_VERSION,
            encoder_kind: encoder.kind().to_string(),
            encoder_levels: encoder.levels(),
            encoder_checkpoint: encoder.checkpoint().to_path_buf(),
            encoder_sha256: encoder.sha256().to_string(),
            dim,
            mean,
            scale,
            train_config,
            policy,
            taxonomy_version: taxonomy.version.clone(),
            weights: vec![0.0; classes * dim],
            bias: vec![0.0; classes],
            temperature: 1.0,
            trained_samples: 0,
            none_samples: 0,
            categories,
        };
        let samples = |head: &PooledHead, raw: &[Vec<f32>], set: &[RouterTrace]| -> Vec<Sample> {
            raw.iter()
                .zip(set)
                .map(|(x, t)| (standardized(head, x), soft_target(&head.categories, t)))
                .collect()
        };
        let fit_data = samples(&head, &raw_fit, &fit);
        let calibration_data = samples(&head, &raw_calibration, &calibration);
        let config = head.train_config.clone();
        let loss_curve = sgd_fit(&mut head.weights, &mut head.bias, dim, &fit_data, &config);
        let (temperature, raw, calibrated) =
            fit_temperature(&head.weights, &head.bias, dim, &calibration_data);
        head.temperature = temperature;
        head.trained_samples = fit_data.len();
        head.none_samples = fit.iter().filter(|t| t.categories.is_empty()).count();
        let report = MetaTrainReport {
            fit_samples: fit_data.len(),
            calibration_samples: calibration_data.len(),
            none_samples: head.none_samples,
            loss_curve,
            temperature,
            calibration_nll_raw: raw,
            calibration_nll_calibrated: calibrated,
        };
        Ok((Self { head, encoder }, report))
    }

    /// Refit the head from zero on every trace, keeping the standardization,
    /// temperature and policy (see [`HashedMetaRouter::refit_all`]).
    pub fn refit_all(&mut self, traces: &[RouterTrace]) -> Result<Vec<f64>> {
        check_trace_categories(&self.head.categories, traces)?;
        let data: Vec<Sample> = traces
            .iter()
            .map(|t| {
                self.encoder
                    .encode(&pooled_text(&t.text, t.context.as_deref()))
                    .map(|x| {
                        (
                            standardized(&self.head, &x),
                            soft_target(&self.head.categories, t),
                        )
                    })
            })
            .collect::<Result<_>>()?;
        let head = &mut self.head;
        head.weights.iter_mut().for_each(|w| *w = 0.0);
        head.bias.iter_mut().for_each(|b| *b = 0.0);
        let config = head.train_config.clone();
        let loss = sgd_fit(&mut head.weights, &mut head.bias, head.dim, &data, &config);
        head.trained_samples = data.len();
        head.none_samples = traces.iter().filter(|t| t.categories.is_empty()).count();
        Ok(loss)
    }

    pub fn head(&self) -> &PooledHead {
        &self.head
    }

    pub fn set_policy(&mut self, policy: MetaPolicy) {
        self.head.policy = policy;
    }

    /// Fit the temperature on an external calibration set; returns
    /// `(temperature, nll at 1, nll at the temperature)`.
    pub fn calibrate(&mut self, traces: &[RouterTrace]) -> Result<(f64, f64, f64)> {
        check_trace_categories(&self.head.categories, traces)?;
        let data: Vec<Sample> = traces
            .iter()
            .map(|t| {
                self.encoder
                    .encode(&pooled_text(&t.text, t.context.as_deref()))
                    .map(|x| {
                        (
                            standardized(&self.head, &x),
                            soft_target(&self.head.categories, t),
                        )
                    })
            })
            .collect::<Result<_>>()?;
        let fitted = fit_temperature(&self.head.weights, &self.head.bias, self.head.dim, &data);
        self.head.temperature = fitted.0;
        Ok(fitted)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string(&self.head).context("serialize pooled router head")?;
        crate::interop::write_atomic(path, text.as_bytes())
            .with_context(|| format!("write pooled router {}", path.display()))
    }

    /// Load a head and the encoder it names, refusing an encoder file whose
    /// hash differs from the one the head was trained on.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read pooled router {}", path.display()))?;
        let head: PooledHead = serde_json::from_str(&text)
            .with_context(|| format!("parse pooled router {}", path.display()))?;
        ensure!(
            head.version <= POOLED_ROUTER_VERSION,
            "pooled router version {} is newer than this build understands",
            head.version
        );
        head.train_config.validate()?;
        head.policy.validate()?;
        let classes = head.categories.len() + 1;
        ensure!(
            head.dim > 0
                && head.weights.len() == classes * head.dim
                && head.bias.len() == classes
                && head.mean.len() == head.dim
                && head.scale.len() == head.dim,
            "pooled router {} has inconsistent shapes",
            path.display()
        );
        ensure!(
            head.weights
                .iter()
                .chain(&head.bias)
                .chain(&head.mean)
                .all(|v| v.is_finite())
                && head.scale.iter().all(|v| v.is_finite() && *v > 0.0)
                && head.temperature.is_finite()
                && head.temperature > 0.0,
            "pooled router {} carries a non-finite or non-positive number",
            path.display()
        );
        let encoder: Box<dyn TextEncoder> = match head.encoder_kind.as_str() {
            "lm" => Box::new(LmEncoder::load(&head.encoder_checkpoint)?),
            "ncm" => Box::new(NcmEncoder::load(
                &head.encoder_checkpoint,
                head.encoder_levels,
            )?),
            other => bail!(
                "pooled router {} names unknown encoder kind '{other}'",
                path.display()
            ),
        };
        ensure!(
            encoder.sha256() == head.encoder_sha256,
            "encoder {} changed since this head was trained (sha256 {} != {})",
            head.encoder_checkpoint.display(),
            encoder.sha256(),
            head.encoder_sha256
        );
        Ok(Self { head, encoder })
    }
}

/// Standardize with the head's statistics, then scale to unit L2 norm (the
/// same length discipline as the hashed features), as sparse pairs.
fn standardized(head: &PooledHead, raw: &[f32]) -> Vec<(u32, f32)> {
    let mut v: Vec<f32> = raw
        .iter()
        .zip(head.mean.iter().zip(&head.scale))
        .map(|(x, (m, s))| (x - m) / s)
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v.into_iter()
        .enumerate()
        .map(|(d, x)| (d as u32, x))
        .collect()
}

impl MetaRouter for PooledMetaRouter {
    fn name(&self) -> &str {
        match self.head.encoder_kind.as_str() {
            "ncm" => "ncm-concept-softmax",
            _ => "lm-pooled-softmax",
        }
    }

    fn categories(&self) -> &[String] {
        &self.head.categories
    }

    fn decide(&self, prompt: &str, context: Option<&str>) -> Result<MetaDecision> {
        if self.head.trained_samples == 0 {
            return Ok(MetaDecision {
                scores: Vec::new(),
                none: 0.0,
                engaged: Vec::new(),
                unsure: Some("the pooled router head is untrained".to_string()),
            });
        }
        let (scores, none) = self.calibrated_scores(prompt, context)?;
        Ok(self.head.policy.decide(scores, none))
    }
}

impl ScoredMetaRouter for PooledMetaRouter {
    fn calibrated_scores(
        &self,
        prompt: &str,
        context: Option<&str>,
    ) -> Result<(Vec<(String, f64)>, f64)> {
        let raw = self.encoder.encode(&pooled_text(prompt, context))?;
        ensure!(
            raw.len() == self.head.dim,
            "the encoder produced {} features, the head expects {}",
            raw.len(),
            self.head.dim
        );
        let feats = standardized(&self.head, &raw);
        let logits = head_logits(&self.head.weights, &self.head.bias, self.head.dim, &feats);
        Ok(ranked_scores(
            &self.head.categories,
            &logits,
            self.head.temperature,
        ))
    }

    fn policy(&self) -> &MetaPolicy {
        &self.head.policy
    }

    fn is_trained(&self) -> bool {
        self.head.trained_samples > 0
    }
}

/// Load either meta-router from its JSON: a file naming an
/// `encoder_checkpoint` is a [`PooledMetaRouter`] head, anything else a
/// [`HashedMetaRouter`].
pub fn load_meta_router(path: &Path) -> Result<Box<dyn MetaRouter>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read meta-router {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse meta-router {}", path.display()))?;
    if value.get("encoder_checkpoint").is_some() {
        Ok(Box::new(PooledMetaRouter::load(path)?))
    } else {
        Ok(Box::new(
            HashedMetaRouter::from_json(&text).with_context(|| format!("in {}", path.display()))?,
        ))
    }
}

// ------------------------------------------------- training entry point --

/// A frozen encoder to train a [`PooledMetaRouter`] on.
#[derive(Debug, Clone)]
pub enum EncoderSpec<'a> {
    /// An `lm` checkpoint ([`LmEncoder`]).
    Lm(&'a Path),
    /// An NCM tokenizer checkpoint pooling `levels` coarse levels
    /// ([`NcmEncoder`]).
    Ncm { checkpoint: &'a Path, levels: usize },
}

impl EncoderSpec<'_> {
    fn load(&self) -> Result<Box<dyn TextEncoder>> {
        Ok(match self {
            Self::Lm(path) => Box::new(LmEncoder::load(path)?),
            Self::Ncm { checkpoint, levels } => Box::new(NcmEncoder::load(checkpoint, *levels)?),
        })
    }

    fn file_stem(&self) -> String {
        match self {
            Self::Lm(_) => "meta-router-lm".to_string(),
            Self::Ncm { levels, .. } => format!("meta-router-ncm-c{levels}"),
        }
    }
}

/// What `dblocks router train` is asked to do besides reading the traces.
#[derive(Debug, Clone)]
pub struct RouterTrainOptions<'a> {
    /// Held-out eval traces; `None` holds out `holdout_percent` of the traces
    /// by id hash.
    pub eval: Option<&'a Path>,
    /// External calibration traces. When given, every training trace is
    /// fitted, and the temperature and the unsure rule are fitted on these
    /// instead of a slice of the training traces — so a calibration set in
    /// the eval set's style (but disjoint from it) can stand in for the
    /// users the router will meet. Never the eval set itself.
    pub calibration: &'a [PathBuf],
    pub out: &'a Path,
    pub tiny: bool,
    pub holdout_percent: u32,
    /// The internal calibration split, when no external set is given.
    pub calibration_group: CalibrationGroup,
    /// The hashed router's feature set ([`FeatureConfig::preset`]).
    pub feature_set: &'a str,
    /// Frozen encoders to train a [`PooledMetaRouter`] on each, compared on
    /// the same split.
    pub encoders: Vec<EncoderSpec<'a>>,
}

/// One frozen-encoder router's results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PooledOutcome {
    pub router: String,
    pub encoder: String,
    pub encoder_sha256: String,
    pub encoder_levels: usize,
    pub feature_dim: usize,
    pub train: MetaTrainReport,
    pub tuning: PolicyTuning,
    pub eval_default_policy: MetaEval,
    pub eval: MetaEval,
    pub head_path: PathBuf,
    /// Wall time to encode, fit, calibrate and evaluate.
    pub seconds: f64,
}

/// Everything `dblocks router train` produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouterTrainOutcome {
    pub traces: Vec<String>,
    pub eval_source: String,
    /// Where the temperature and the unsure rule were fitted.
    pub calibration_source: String,
    pub train_traces: usize,
    pub eval_traces: usize,
    pub calibration_traces: usize,
    /// Eval traces whose text also appears in the training traces (a leak
    /// that would inflate every number).
    pub eval_text_overlap: usize,
    pub tiny: bool,
    pub feature_set: String,
    pub calibration_group: CalibrationGroup,
    /// Whether the weights were refitted on every trace after an internal
    /// [`CalibrationGroup::Leaf`] calibration.
    pub refit: bool,
    pub train: MetaTrainReport,
    /// How the unsure rule was chosen (on the calibration traces).
    pub tuning: PolicyTuning,
    /// Eval with the default unsure rule — the "before" of the tuning.
    pub eval_default_policy: MetaEval,
    /// Eval with the tuned rule — the model as saved.
    pub eval: MetaEval,
    /// Eval of the saved model at temperature 1, for the calibration effect.
    pub eval_uncalibrated: MetaEval,
    /// The frozen-encoder routers, one per [`EncoderSpec`].
    pub pooled: Vec<PooledOutcome>,
    pub model_path: PathBuf,
    pub report_path: PathBuf,
    pub note: String,
}

/// Train the hashed meta-router (and one [`PooledMetaRouter`] per encoder)
/// on the trace files `traces`, fit the temperature and tune the unsure rule
/// on the calibration traces ([`RouterTrainOptions::calibration`], else a
/// slice of the training traces; never the eval set), evaluate on the
/// held-out traces before and after the tuning, and write every model plus
/// `<out>/router-train-report.json`.
pub fn train_router_files(
    traces: &[PathBuf],
    options: &RouterTrainOptions<'_>,
    taxonomy: &Taxonomy,
) -> Result<RouterTrainOutcome> {
    let holdout_percent = options.holdout_percent;
    ensure!(holdout_percent < 100, "holdout_percent must be below 100");
    let all = read_trace_files(traces, taxonomy)?;
    let sources = traces
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>();
    let (train, held, eval_source) = match options.eval {
        Some(path) => (
            all,
            read_traces(path, taxonomy)?,
            path.display().to_string(),
        ),
        None => {
            let (train, held) = split_traces(&all, holdout_percent);
            (
                train,
                held,
                format!(
                    "{holdout_percent}% id-hash holdout of {}",
                    sources.join(" + ")
                ),
            )
        }
    };
    ensure!(!train.is_empty(), "no training traces after the split");
    ensure!(!held.is_empty(), "no eval traces ({eval_source})");
    let train_ids: BTreeSet<&str> = train.iter().map(|t| t.id.as_str()).collect();
    if let Some(dup) = held.iter().find(|t| train_ids.contains(t.id.as_str())) {
        bail!(
            "eval trace id '{}' also appears in the training traces",
            dup.id
        );
    }
    let train_texts: BTreeSet<&str> = train.iter().map(|t| t.text.as_str()).collect();
    let overlap = held
        .iter()
        .filter(|t| train_texts.contains(t.text.as_str()))
        .count();

    let external = if options.calibration.is_empty() {
        Vec::new()
    } else {
        read_trace_files(options.calibration, taxonomy)?
    };
    let held_ids: BTreeSet<&str> = held.iter().map(|t| t.id.as_str()).collect();
    let held_texts: BTreeSet<&str> = held.iter().map(|t| t.text.as_str()).collect();
    if let Some(leak) = external
        .iter()
        .find(|t| held_ids.contains(t.id.as_str()) || held_texts.contains(t.text.as_str()))
    {
        bail!(
            "calibration trace '{}' is also in the eval set; tuning on it would leak the eval",
            leak.id
        );
    }
    let use_external = !external.is_empty();

    let (features, mut train_config) = if options.tiny {
        (
            FeatureConfig::preset(options.feature_set, true)?,
            MetaTrainConfig::tiny(),
        )
    } else {
        (
            FeatureConfig::preset(options.feature_set, false)?,
            MetaTrainConfig::default(),
        )
    };
    train_config.calibration_group = options.calibration_group;
    if use_external {
        // Every training trace is fitted; calibration comes from outside.
        train_config.calibration_percent = 0;
    }
    let refit = !use_external && options.calibration_group == CalibrationGroup::Leaf;
    let majority = majority_category(&train);
    let calibration_source = if use_external {
        options
            .calibration
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" + ")
    } else {
        format!(
            "{}% {}-hash slice of the training traces",
            train_config.calibration_percent,
            options.calibration_group.name()
        )
    };

    let mut model = HashedMetaRouter::new(
        taxonomy,
        features,
        train_config.clone(),
        MetaPolicy::default(),
    )?;
    let mut report = model.train(&train)?;
    let calibration = if use_external {
        let (temperature, raw, calibrated) = model.calibrate(&external)?;
        report.temperature = temperature;
        report.calibration_nll_raw = raw;
        report.calibration_nll_calibrated = calibrated;
        report.calibration_samples = external.len();
        external.clone()
    } else {
        model.calibration_split(&train).1
    };
    let tuning = tune_policy(&model, &calibration)?;
    if refit {
        model.refit_all(&train)?;
    }
    let eval_default = evaluate_meta(&model, &held, majority.as_deref())?;
    model.policy = tuning.chosen.clone();
    let eval_report = evaluate_meta(&model, &held, majority.as_deref())?;
    let raw = model.with_temperature(1.0);
    let eval_raw = evaluate_meta(&raw, &held, majority.as_deref())?;

    let model_path = options.out.join("meta-router.json");
    let report_path = options.out.join("router-train-report.json");
    model.save(&model_path)?;

    let mut pooled = Vec::with_capacity(options.encoders.len());
    for spec in &options.encoders {
        let started = std::time::Instant::now();
        let encoder = spec.load()?;
        let encoder_name = encoder.checkpoint().display().to_string();
        let (mut router, mut pooled_report) = PooledMetaRouter::train(
            encoder,
            taxonomy,
            train_config.clone(),
            MetaPolicy::default(),
            &train,
        )?;
        let calibration = if use_external {
            let (temperature, raw, calibrated) = router.calibrate(&external)?;
            pooled_report.temperature = temperature;
            pooled_report.calibration_nll_raw = raw;
            pooled_report.calibration_nll_calibrated = calibrated;
            pooled_report.calibration_samples = external.len();
            external.clone()
        } else {
            calibration_split(&train, &train_config).1
        };
        let pooled_tuning = tune_policy(&router, &calibration)?;
        if refit {
            router.refit_all(&train)?;
        }
        let before = evaluate_meta(&router, &held, majority.as_deref())?;
        router.set_policy(pooled_tuning.chosen.clone());
        let after = evaluate_meta(&router, &held, majority.as_deref())?;
        let head_path = options.out.join(format!("{}.json", spec.file_stem()));
        router.save(&head_path)?;
        pooled.push(PooledOutcome {
            router: router.name().to_string(),
            encoder: encoder_name,
            encoder_sha256: router.head().encoder_sha256.clone(),
            encoder_levels: router.head().encoder_levels,
            feature_dim: router.head().dim,
            train: pooled_report,
            tuning: pooled_tuning,
            eval_default_policy: before,
            eval: after,
            head_path,
            seconds: started.elapsed().as_secs_f64(),
        });
    }

    let outcome = RouterTrainOutcome {
        traces: sources,
        eval_source,
        calibration_source,
        train_traces: train.len(),
        eval_traces: held.len(),
        calibration_traces: calibration.len(),
        eval_text_overlap: overlap,
        tiny: options.tiny,
        feature_set: options.feature_set.to_string(),
        calibration_group: options.calibration_group,
        refit,
        train: report,
        tuning,
        eval_default_policy: eval_default,
        eval: eval_report,
        eval_uncalibrated: eval_raw,
        pooled,
        model_path,
        report_path: report_path.clone(),
        note: format!(
            "{} eval traces: one trace moves overall top-1 by {:.1} points; read every rate \
             with its Wilson interval and against the majority baseline, not against 0",
            held.len(),
            100.0 / held.len() as f64
        ),
    };
    let text = serde_json::to_string_pretty(&outcome).context("serialize train report")?;
    crate::interop::write_atomic(&report_path, text.as_bytes())
        .with_context(|| format!("write {}", report_path.display()))?;
    Ok(outcome)
}

impl RouterTrainOutcome {
    /// The `dblocks router train` report.
    pub fn render(&self) -> String {
        let first = self.train.loss_curve.first().copied().unwrap_or(f64::NAN);
        let last = self.train.loss_curve.last().copied().unwrap_or(f64::NAN);
        let mut out = format!(
            "trained on {} traces from {} ({} fit, {} none; features '{}'{}); eval on {} traces from {}\n\
             calibration (temperature + unsure rule) on {} traces: {}\n",
            self.train_traces,
            self.traces.join(" + "),
            self.train.fit_samples,
            self.train.none_samples,
            self.feature_set,
            if self.refit {
                ", refitted on all after the leaf calibration"
            } else {
                ""
            },
            self.eval_traces,
            self.eval_source,
            self.calibration_traces,
            self.calibration_source
        );
        out.push_str(&format!(
            "loss {first:.4} -> {last:.4} over {} epochs; temperature {:.3} (calibration NLL {:.4} -> {:.4})\n",
            self.train.loss_curve.len(),
            self.train.temperature,
            self.train.calibration_nll_raw,
            self.train.calibration_nll_calibrated
        ));
        out.push_str(&render_tuning(&self.tuning));
        out.push_str(&format!("  objective: {}\n", self.tuning.objective));
        out.push_str(
            &self
                .eval_default_policy
                .render("hashed, before tuning (default rule)"),
        );
        out.push_str(&self.eval.render("hashed, after tuning (saved model)"));
        out.push_str(
            &self
                .eval_uncalibrated
                .render("hashed, after tuning, temperature 1"),
        );
        out.push_str("hashed in-scope top-1 per category (after tuning), weakest first:\n");
        out.push_str(&self.eval.render_categories());
        for p in &self.pooled {
            out.push_str(&format!(
                "\n== {} on {} (sha256 {}..., levels {}, {} features, {:.1}s); temperature {:.3}\n",
                p.router,
                p.encoder,
                p.encoder_sha256.chars().take(12).collect::<String>(),
                p.encoder_levels,
                p.feature_dim,
                p.seconds,
                p.train.temperature
            ));
            out.push_str(&render_tuning(&p.tuning));
            out.push_str(
                &p.eval_default_policy
                    .render(&format!("{}, before tuning", p.router)),
            );
            out.push_str(&p.eval.render(&format!("{}, after tuning", p.router)));
            out.push_str("in-scope top-1 per category, weakest first:\n");
            out.push_str(&p.eval.render_categories());
        }
        out.push_str(&format!(
            "\ncomparison on the same {} eval traces (after tuning):\n  {:<26} {:<28} {:<28} {:<26} {:>6} {:>8}\n",
            self.eval_traces,
            "router",
            "in-scope top-1",
            "in-scope routed+correct",
            "OOS unsure recall",
            "BA",
            "combo F1"
        ));
        let row = |name: String, e: &MetaEval| {
            format!(
                "  {:<26} {:<28} {:<28} {:<26} {:>6.3} {:>8.3}\n",
                name,
                e.in_scope_top1.render(),
                e.in_scope_routed_correct.render(),
                e.oos_unsure_recall.render(),
                e.balanced_accuracy,
                e.combo_set_f1
            )
        };
        out.push_str(&row(format!("hashed ({})", self.feature_set), &self.eval));
        for p in &self.pooled {
            let name = if p.encoder_levels > 0 {
                format!("{} c={}", p.router, p.encoder_levels)
            } else {
                p.router.clone()
            };
            out.push_str(&row(name, &p.eval));
        }
        if self.eval_text_overlap > 0 {
            out.push_str(&format!(
                "WARNING: {} eval traces have text identical to a training trace\n",
                self.eval_text_overlap
            ));
        }
        out.push_str(&format!("note: {}\n", self.note));
        out.push_str(&format!(
            "wrote {} and {}\n",
            self.model_path.display(),
            self.report_path.display()
        ));
        out
    }
}

fn render_tuning(t: &PolicyTuning) -> String {
    format!(
        "unsure rule: confidence_floor {:.3} -> {:.3}, none_margin {:.3} -> {:.3}; calibration \
         objective {:.3} -> {:.3} ({} in-scope / {} out-of-scope, {} grid points)\n  {}\n",
        t.default_policy.confidence_floor,
        t.chosen.confidence_floor,
        t.default_policy.none_margin,
        t.chosen.none_margin,
        t.default_objective,
        t.chosen_objective,
        t.calibration_in_scope,
        t.calibration_oos,
        t.grid_points,
        t.note
    )
}

// --------------------------------------------------------- certificates --

/// `(id, text, [(category, weight)])`.
type FixtureRow = (&'static str, &'static str, &'static [(&'static str, f64)]);

/// A small, neutral fixture trace set: enough to train a router whose plans
/// the certificates can check. Not a quality benchmark.
fn fixture_traces() -> Vec<RouterTrace> {
    let rows: &[FixtureRow] = &[
        (
            "f01",
            "write a rust function that parses a toml config with serde",
            &[("coding", 1.0)],
        ),
        (
            "f02",
            "python list comprehension that keeps the even numbers",
            &[("coding", 1.0)],
        ),
        (
            "f03",
            "fix the borrow checker error in this rust struct method",
            &[("coding", 1.0)],
        ),
        (
            "f04",
            "javascript async function fetching json and handling errors",
            &[("coding", 1.0)],
        ),
        (
            "f05",
            "prove that the sum of two even integers is even",
            &[("math", 1.0)],
        ),
        (
            "f06",
            "solve for x when three x plus five equals twenty",
            &[("math", 1.0)],
        ),
        (
            "f07",
            "compute the derivative of x squared times sine x",
            &[("math", 1.0)],
        ),
        (
            "f08",
            "how many ways can five books be arranged on a shelf",
            &[("math", 1.0)],
        ),
        (
            "f09",
            "which planet in the solar system is closest to the sun",
            &[("science", 1.0)],
        ),
        (
            "f10",
            "why does ice float on liquid water",
            &[("science", 1.0)],
        ),
        (
            "f11",
            "what gas do plants absorb during photosynthesis",
            &[("science", 1.0)],
        ),
        (
            "f12",
            "outline the phases of an authorized penetration testing methodology",
            &[("cyber-specialized", 1.0)],
        ),
        (
            "f13",
            "write a python script for modular inverse in number theory homework",
            &[("coding", 0.5), ("math", 0.5)],
        ),
        (
            "f14",
            "explain the scope and rules of engagement section of a pentest report",
            &[("cyber-specialized", 1.0)],
        ),
        (
            "f15",
            "recommend a pasta recipe for a weeknight dinner",
            &[],
        ),
        (
            "f16",
            "what is a good stretching routine after running",
            &[],
        ),
    ];
    rows.iter()
        .map(|(id, text, cats)| RouterTrace {
            id: (*id).to_string(),
            text: (*text).to_string(),
            context: None,
            categories: cats
                .iter()
                .map(|(c, w)| TraceCategory {
                    id: (*c).to_string(),
                    weight: *w,
                })
                .collect(),
            subfields: Vec::new(),
            style: None,
        })
        .collect()
}

const FIXTURE_PROMPTS: &[&str] = &[
    "rust function to read a file line by line",
    "integrate x times e to the x",
    "why is the sky blue",
    "steps of a penetration testing methodology",
    "python code for the greatest common divisor of two integers",
    "a pasta recipe please",
];

/// The fixture router the certificates and tests share: a tiny trained
/// meta-router over the roster catalogue.
pub fn fixture_router() -> Result<LayeredRouter> {
    let taxonomy = Taxonomy::builtin();
    let mut meta = HashedMetaRouter::new(
        &taxonomy,
        FeatureConfig::tiny(),
        MetaTrainConfig {
            epochs: 30,
            calibration_percent: 0,
            ..MetaTrainConfig::default()
        },
        MetaPolicy::default(),
    )?;
    meta.train(&fixture_traces())?;
    let catalogue = Catalogue::from_roster(&taxonomy);
    LayeredRouter::new(taxonomy, Box::new(meta), catalogue, RouteBounds::default())
}

/// `a.max(b)`, except that a NaN in either operand is kept (`f64::max`
/// would silently drop it, and a NaN residual must fail).
fn worst(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn cert(
    name: &'static str,
    theorem: &'static str,
    residual: f64,
    tolerance: f64,
) -> crate::verify::Certificate {
    crate::verify::Certificate {
        group: "router",
        name,
        theorem,
        residual,
        tolerance,
    }
}

/// The `router` certificate group. Each check carries a planted-violation
/// control, so a validator that accepts everything fails its certificate.
pub fn certificates() -> Vec<crate::verify::Certificate> {
    let bounds = RouteBounds::default();
    let router = fixture_router();
    let plans: Result<Vec<RoutePlan>> = match &router {
        Ok(router) => FIXTURE_PROMPTS.iter().map(|p| router.route(p)).collect(),
        Err(err) => Err(anyhow::anyhow!("{err:#}")),
    };

    // 1. Distributions.
    let distribution = match &plans {
        Err(_) => f64::INFINITY,
        Ok(plans) => {
            let routed = plans.iter().filter(|p| p.unsure.is_none()).count();
            let mut residual = plans
                .iter()
                .map(|p| check_plan(p, &bounds).distribution_residual())
                .fold(0.0, worst);
            // The certificate is vacuous if nothing routed.
            if routed == 0 {
                residual = f64::INFINITY;
            }
            // Control: a NaN weight must not pass.
            if let Some(mut bad) = plans.iter().find(|p| p.unsure.is_none()).cloned() {
                if let Some(first) = bad.specialists.first_mut() {
                    first.2 = f64::NAN;
                }
                if validate_plan(&bad, &bounds).is_ok() {
                    residual = f64::INFINITY;
                }
            }
            residual
        }
    };

    // 2. Hierarchy consistency.
    let hierarchy = match &plans {
        Err(_) => f64::INFINITY,
        Ok(plans) => {
            let mut residual = plans
                .iter()
                .map(|p| check_plan(p, &bounds).hierarchy_residual())
                .fold(0.0, worst);
            // Control: move one specialist into a category the plan did not
            // choose; the validator must refuse it.
            if let Some(mut bad) = plans.iter().find(|p| p.unsure.is_none()).cloned() {
                if let Some(first) = bad.specialists.first_mut() {
                    first.1 = "not-a-chosen-category".to_string();
                }
                if validate_plan(&bad, &bounds).is_ok() {
                    residual += 1.0;
                }
            }
            residual
        }
    };

    // 3. Determinism: route twice, and once more through a JSON round trip
    // of the meta-router; every serialized plan must be identical.
    let determinism = (|| -> Result<f64> {
        let a = fixture_router()?;
        let b = fixture_router()?;
        let mut differing = 0.0;
        let mut first: Option<String> = None;
        for prompt in FIXTURE_PROMPTS {
            let pa = a.route(prompt)?.to_json()?;
            let pb = b.route(prompt)?.to_json()?;
            let pc = a.route(prompt)?.to_json()?;
            if pa != pb || pa != pc {
                differing += 1.0;
            }
            first.get_or_insert(pa);
        }
        // Control: the comparison must see a one-ULP change.
        if let Some(text) = first {
            let mut plan: RoutePlan = serde_json::from_str(&text)?;
            if let Some(c) = plan.categories.first_mut() {
                c.1 = f64::from_bits(c.1.to_bits() ^ 1);
            }
            if plan.unsure.is_none() && plan.to_json()? == text {
                differing += 1.0;
            }
        }
        // A trained meta-router reloaded from JSON decides identically.
        let taxonomy = Taxonomy::builtin();
        let mut meta = HashedMetaRouter::new(
            &taxonomy,
            FeatureConfig::tiny(),
            MetaTrainConfig {
                epochs: 10,
                calibration_percent: 0,
                ..MetaTrainConfig::default()
            },
            MetaPolicy::default(),
        )?;
        meta.train(&fixture_traces())?;
        let back = HashedMetaRouter::from_json(&meta.to_json()?)?;
        for prompt in FIXTURE_PROMPTS {
            if meta.decide(prompt, None)? != back.decide(prompt, None)? {
                differing += 1.0;
            }
        }
        Ok(differing)
    })()
    .unwrap_or(f64::INFINITY);

    // 4. Taxonomy and legacy-index round trips.
    let round_trip = (|| -> Result<f64> {
        let mut residual = 0.0;
        let taxonomy = Taxonomy::builtin();
        let json = taxonomy.to_json()?;
        let back = Taxonomy::from_json(&json)?;
        if back != taxonomy || back.to_json()? != json {
            residual += 1.0;
        }
        // A pre-Stage-4 index (no taxonomy keys) loads, re-serializes byte
        // for byte, and validates against the taxonomy.
        let legacy = ExpertIndex::from_json(LEGACY_INDEX)?;
        if legacy.to_json()? != ExpertIndex::from_json(&legacy.to_json()?)?.to_json()?
            || legacy.to_json()?.contains("\"category\"")
        {
            residual += 1.0;
        }
        legacy.validate_with_taxonomy(&taxonomy)?;
        // With taxonomy fields it round-trips, and a bad id is refused.
        let mut tagged = legacy.clone();
        if let Some(b) = tagged.boxes.first_mut() {
            b.category = Some("coding".to_string());
            if let Some(e) = b.experts.first_mut() {
                e.subfield = Some("coding/rust".to_string());
            }
        }
        let tagged_back = ExpertIndex::from_json(&tagged.to_json()?)?;
        if tagged_back != tagged {
            residual += 1.0;
        }
        tagged_back.validate_with_taxonomy(&taxonomy)?;
        let mut bad = tagged;
        if let Some(e) = bad.boxes.first_mut().and_then(|b| b.experts.first_mut()) {
            e.subfield = Some("coding/not-a-leaf".to_string());
        }
        if bad.validate_with_taxonomy(&taxonomy).is_ok() {
            residual += 1.0;
        }
        Ok(residual)
    })()
    .unwrap_or(f64::INFINITY);

    vec![
        cert(
            "route_weights_are_distributions",
            "Every routed plan's category weights, specialist weights and per-category L2 gates \
             are finite, non-negative and sum to 1 within 1e-9; a planted NaN weight is refused.",
            distribution,
            PLAN_TOLERANCE,
        ),
        cert(
            "chosen_specialists_belong_to_chosen_categories",
            "Every chosen specialist sits in a chosen category, every chosen category has a \
             specialist, a category's specialists carry exactly its weight, ids are unique and \
             top-k bounds hold; a planted orphan specialist is refused by the validator.",
            hierarchy,
            PLAN_TOLERANCE,
        ),
        cert(
            "routing_is_deterministic",
            "A fixed model routes a fixed prompt to a byte-identical plan across router \
             instances and repeated calls, a meta-router reloaded from JSON decides identically, \
             and the comparison detects a one-ULP weight change.",
            determinism,
            0.0,
        ),
        cert(
            "taxonomy_round_trips",
            "Taxonomy JSON -> taxonomy -> JSON is identical; a legacy expert index without \
             taxonomy fields loads, re-serializes unchanged and validates; taxonomy fields \
             round-trip; an unknown leaf id is refused.",
            round_trip,
            0.0,
        ),
    ]
}

/// A minimal expert index in the pre-Stage-4 wire format (no taxonomy keys).
const LEGACY_INDEX: &str = r#"{
  "schema_version": 1,
  "model_id": "deadbeefcafef00d",
  "site": "mlp",
  "routing": {"cond_size": 8, "hidden_size": 32, "route_on_tokens": true, "top_box": 1, "top_expert": 1},
  "boxes": [
    {"id": "coding", "label": "Code", "index": 0, "experts": [
      {"id": "coding/rust", "label": "Rust", "index": 0, "global_index": 0, "kind": "mlp",
       "hidden_size": 32, "intermediate_size": 64, "enabled": true, "num_parameters": 4096,
       "weights": {"param_path": "vit.layers.1.mlp.boxes.0.experts.0",
                   "sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
       "tags": ["rust"]}
    ]}
  ]
}"#;

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

    #[test]
    fn features_are_deterministic_unit_norm_and_context_tagged() {
        let cfg = FeatureConfig::tiny();
        let a = featurize("rust borrow checker", None, &cfg);
        assert_eq!(a, featurize("rust borrow checker", None, &cfg));
        let norm: f32 = a.iter().map(|f| f.1 * f.1).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "{norm}");
        assert!(a.windows(2).all(|w| w[0].0 < w[1].0), "sorted and merged");
        let with_context = featurize("rust borrow checker", Some("rust borrow checker"), &cfg);
        assert_ne!(
            a, with_context,
            "context features are distinct from prompt features"
        );
        assert!(featurize("", None, &cfg).is_empty());
    }

    #[test]
    fn trace_validation_and_normalization() {
        let t = Taxonomy::builtin();
        let ok = r#"{"id":"a","text":"x","categories":[{"id":"coding","weight":0.667},{"id":"data","weight":0.333}]}"#;
        let traces = parse_traces(ok, &t).unwrap();
        let sum: f64 = traces[0].categories.iter().map(|c| c.weight).sum();
        assert!((sum - 1.0).abs() < 1e-15);
        assert_eq!(traces[0].gold_top(), Some("coding"));

        for bad in [
            r#"{"id":"a","text":"x","categories":[{"id":"nope","weight":1.0}]}"#,
            r#"{"id":"a","text":"x","categories":[{"id":"coding","weight":0.5}]}"#,
            r#"{"id":"a","text":"x","categories":[{"id":"coding","weight":-1.0},{"id":"math","weight":2.0}]}"#,
            r#"{"id":"","text":"x","categories":[]}"#,
            r#"{"id":"a","text":"x","categories":[{"id":"coding","weight":1.0}],"subfields":["math/algebra"]}"#,
            r#"{"id":"a","text":"x","categories":[{"id":"coding","weight":1.0},{"id":"coding","weight":0.0}]}"#,
        ] {
            assert!(parse_traces(bad, &t).is_err(), "{bad}");
        }
        let dup = format!("{ok}\n{ok}");
        assert!(parse_traces(&dup, &t)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
        let none = r#"{"id":"n","text":"pasta","categories":[]}"#;
        assert_eq!(parse_traces(none, &t).unwrap()[0].gold_top(), None);
    }

    #[test]
    fn split_is_a_stable_partition() {
        let traces = fixture_traces();
        let (a, b) = split_traces(&traces, 30);
        assert_eq!(a.len() + b.len(), traces.len());
        assert_eq!(split_traces(&traces, 30), (a, b));
        assert_eq!(split_traces(&traces, 0).1.len(), 0);
    }

    #[test]
    fn meta_router_learns_the_fixture_and_round_trips() {
        let taxonomy = Taxonomy::builtin();
        let mut meta = HashedMetaRouter::new(
            &taxonomy,
            FeatureConfig::tiny(),
            MetaTrainConfig {
                epochs: 30,
                calibration_percent: 0,
                ..MetaTrainConfig::default()
            },
            MetaPolicy::default(),
        )
        .unwrap();
        // Untrained: abstains.
        assert!(meta.decide("rust", None).unwrap().unsure.is_some());
        let report = meta.train(&fixture_traces()).unwrap();
        let first = report.loss_curve.first().copied().unwrap();
        let last = report.loss_curve.last().copied().unwrap();
        assert!(last < first, "loss must fall: {first} -> {last}");
        let eval = evaluate_meta(&meta, &fixture_traces(), None).unwrap();
        assert!(eval.top1_accuracy > 0.9, "training-set fit: {eval:?}");
        let d = meta.decide("write a rust function", None).unwrap();
        assert_eq!(d.engaged.first().map(|e| e.0.as_str()), Some("coding"));
        let back = HashedMetaRouter::from_json(&meta.to_json().unwrap()).unwrap();
        assert_eq!(back, meta);

        // Corrupt files are refused.
        let mut v: serde_json::Value = serde_json::from_str(&meta.to_json().unwrap()).unwrap();
        v["version"] = serde_json::json!(META_ROUTER_VERSION + 1);
        assert!(HashedMetaRouter::from_json(&v.to_string()).is_err());
        let mut v: serde_json::Value = serde_json::from_str(&meta.to_json().unwrap()).unwrap();
        v["weights"] = serde_json::json!([1.0]);
        assert!(HashedMetaRouter::from_json(&v.to_string()).is_err());
        let mut v: serde_json::Value = serde_json::from_str(&meta.to_json().unwrap()).unwrap();
        v["temperature"] = serde_json::json!(0.0);
        assert!(HashedMetaRouter::from_json(&v.to_string()).is_err());
    }

    #[test]
    fn policy_engages_combinations_and_abstains() {
        let policy = MetaPolicy::default();
        let s = |v: &[(&str, f64)]| v.iter().map(|(a, b)| ((*a).to_string(), *b)).collect();
        let d = policy.decide(s(&[("coding", 0.5), ("math", 0.3), ("data", 0.1)]), 0.05);
        assert_eq!(d.engaged.len(), 2);
        assert!((d.engaged.iter().map(|e| e.1).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            policy.decide(s(&[("coding", 0.2)]), 0.1).unsure.is_some(),
            "below the floor"
        );
        assert!(
            policy.decide(s(&[("coding", 0.4)]), 0.5).unsure.is_some(),
            "none wins"
        );
        let five = s(&[("a", 0.2), ("b", 0.2), ("c", 0.2), ("d", 0.2), ("e", 0.2)]);
        let wide = MetaPolicy {
            confidence_floor: 0.1,
            ..MetaPolicy::default()
        };
        assert_eq!(
            wide.decide(five, 0.0).engaged.len(),
            3,
            "max_categories holds"
        );
    }

    #[test]
    fn eval_metrics_read_as_specified() {
        struct Fixed(MetaDecision, Vec<String>);
        impl MetaRouter for Fixed {
            fn name(&self) -> &str {
                "fixed"
            }
            fn categories(&self) -> &[String] {
                &self.1
            }
            fn decide(&self, _: &str, _: Option<&str>) -> Result<MetaDecision> {
                Ok(self.0.clone())
            }
        }
        let router = Fixed(
            MetaDecision {
                scores: vec![("coding".into(), 0.6), ("math".into(), 0.4)],
                none: 0.0,
                engaged: vec![("coding".into(), 0.6), ("math".into(), 0.4)],
                unsure: None,
            },
            vec!["coding".into(), "math".into()],
        );
        let t = Taxonomy::builtin();
        let traces = parse_traces(
            "{\"id\":\"1\",\"text\":\"a\",\"categories\":[{\"id\":\"coding\",\"weight\":1.0}]}\n\
             {\"id\":\"2\",\"text\":\"b\",\"categories\":[{\"id\":\"math\",\"weight\":1.0}]}",
            &t,
        )
        .unwrap();
        let e = evaluate_meta(&router, &traces, Some("coding")).unwrap();
        assert_eq!(e.samples, 2);
        assert_eq!(e.top1_correct, 1);
        assert!((e.mean_set_f1 - 2.0 / 3.0).abs() < 1e-12, "{e:?}");
        assert!((e.mean_weight_l1 - 1.0).abs() < 1e-12, "{e:?}");
        assert!((e.majority_top1_accuracy - 0.5).abs() < 1e-12);
        assert_eq!(majority_category(&traces).as_deref(), Some("coding"));
    }

    #[test]
    fn plans_satisfy_every_invariant() {
        let router = fixture_router().unwrap();
        let bounds = *router.bounds();
        let mut routed = 0;
        for prompt in FIXTURE_PROMPTS {
            let plan = router.route(prompt).unwrap();
            validate_plan(&plan, &bounds).unwrap();
            if plan.unsure.is_none() {
                routed += 1;
                // Every layer is in the trace.
                for layer in [
                    RouteLayer::L3Meta,
                    RouteLayer::L2Category,
                    RouteLayer::PlannerBudget,
                    RouteLayer::L1Specialist,
                ] {
                    assert!(plan.trace.iter().any(|d| d.layer == layer), "{layer:?}");
                }
            }
        }
        assert!(routed >= 3, "the fixture must route most prompts");
        let plan = router
            .route("rust function to read a file line by line")
            .unwrap();
        assert_eq!(
            plan.categories.first().map(|c| c.0.as_str()),
            Some("coding")
        );
        assert!(plan
            .specialists
            .iter()
            .filter(|s| s.1 == "coding")
            .all(|s| s.0.starts_with("coding/")));
        // The out-of-scope prompt abstains.
        let pasta = router.route("a pasta recipe please").unwrap();
        assert!(pasta.unsure.is_some(), "{pasta:?}");
    }

    #[test]
    fn the_validator_rejects_planted_violations() {
        let router = fixture_router().unwrap();
        let bounds = *router.bounds();
        let plan = router
            .route("rust function to read a file line by line")
            .unwrap();
        validate_plan(&plan, &bounds).unwrap();

        let mut orphan = plan.clone();
        orphan.specialists[0].1 = "math".into();
        assert!(validate_plan(&orphan, &bounds).is_err());

        let mut nan = plan.clone();
        nan.categories[0].1 = f64::NAN;
        assert!(validate_plan(&nan, &bounds).is_err());

        let mut negative = plan.clone();
        negative.specialists[0].2 = -0.1;
        assert!(validate_plan(&negative, &bounds).is_err());

        let mut heavy = plan.clone();
        heavy.categories[0].1 += 0.5;
        assert!(validate_plan(&heavy, &bounds).is_err());

        let mut empty_cat = plan.clone();
        empty_cat.categories.push(("math".into(), 0.0));
        assert!(validate_plan(&empty_cat, &bounds).is_err());

        let tight = RouteBounds {
            max_specialists: 1,
            top_specialists_per_category: 1,
            ..bounds
        };
        if plan.specialists.len() > 1 {
            assert!(validate_plan(&plan, &tight).is_err());
        }

        let mut unsure_but_routed = plan;
        unsure_but_routed.unsure = Some("x".into());
        assert!(validate_plan(&unsure_but_routed, &bounds).is_err());
    }

    #[test]
    fn budget_bounds_hold_and_drop_whole_categories() {
        let taxonomy = Taxonomy::builtin();
        let mut meta = HashedMetaRouter::new(
            &taxonomy,
            FeatureConfig::tiny(),
            MetaTrainConfig {
                epochs: 30,
                calibration_percent: 0,
                ..MetaTrainConfig::default()
            },
            MetaPolicy::default(),
        )
        .unwrap();
        meta.train(&fixture_traces()).unwrap();
        let bounds = RouteBounds {
            max_categories: 3,
            top_specialists_per_category: 3,
            max_specialists: 1,
        };
        let router = LayeredRouter::new(
            taxonomy.clone(),
            Box::new(meta),
            Catalogue::from_roster(&taxonomy),
            bounds,
        )
        .unwrap();
        let plan = router
            .route("write a python script for modular inverse in number theory homework")
            .unwrap();
        validate_plan(&plan, &bounds).unwrap();
        if plan.unsure.is_none() {
            assert_eq!(plan.specialists.len(), 1);
            assert_eq!(
                plan.categories.len(),
                1,
                "a category without a kept specialist is dropped"
            );
            assert!((plan.categories[0].1 - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn mosme_l2_masks_disabled_specialists_exactly() {
        let taxonomy = Taxonomy::builtin();
        let mut catalogue = Catalogue::from_roster(&taxonomy);
        for s in &mut catalogue.specialists {
            if s.id == "coding/rust" {
                s.enabled = false;
            }
        }
        let l2 =
            MosmeCategoryRouter::uniform_prior(&catalogue, &taxonomy, 3, FeatureConfig::tiny())
                .unwrap();
        assert!(!l2.is_trained());
        let q = RouteQuery {
            prompt: "rust",
            context: None,
        };
        let gates = l2.route(&q, "coding").unwrap().unwrap();
        assert_eq!(gates.len(), 3, "top-k");
        assert!(gates.iter().all(|(id, _)| id != "coding/rust"), "{gates:?}");
        for (_, g) in &gates {
            assert!((g - 1.0 / 3.0).abs() < 1e-6, "uniform prior: {gates:?}");
        }
        assert!(
            l2.route(&q, "networking").unwrap().is_none(),
            "no box, no answer"
        );
        // The spec is the catalogue's servable categories, in taxonomy order.
        let ids: Vec<&str> = l2.spec().boxes.iter().map(|b| b.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "coding",
                "math",
                "cyber-offensive",
                "cyber-appsec",
                "cyber-specialized",
                "science"
            ]
        );
    }

    #[test]
    fn traffic_router_promotes_a_box_readout() {
        let taxonomy = Taxonomy::builtin();
        let catalogue = Catalogue::from_roster(&taxonomy);
        let traffic = vec![
            ("coding:rust".to_string(), 0.6),
            ("coding:jsts".to_string(), 0.3),
            ("coding:python".to_string(), 0.1),
            ("field:pentest".to_string(), 0.9),
        ];
        let r = TrafficCategoryRouter::from_box_traffic(
            "probe", "coding", &traffic, &taxonomy, &catalogue, 2,
        )
        .unwrap();
        let q = RouteQuery {
            prompt: "x",
            context: None,
        };
        let gates = r.route(&q, "coding").unwrap().unwrap();
        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].0, "coding/rust");
        assert_eq!(gates[1].0, "coding/jsts");
        assert!((gates[0].1 - 2.0 / 3.0).abs() < 1e-12);
        assert!(r.route(&q, "math").unwrap().is_none());
        assert!(TrafficCategoryRouter::from_box_traffic(
            "probe", "math", &traffic, &taxonomy, &catalogue, 2
        )
        .is_none());
    }

    #[test]
    fn catalogue_from_registry_respects_status_and_reports_unmapped() {
        use crate::nina::ModelEntry;
        let taxonomy = Taxonomy::builtin();
        let entry = |name: &str, status: ModelStatus| ModelEntry {
            name: name.into(),
            status,
            checkpoint_hash: "ab".repeat(32),
            ..ModelEntry::default()
        };
        let file = RegistryFile {
            models: vec![
                entry("coding/rust", ModelStatus::Active),
                entry("pentest/recon", ModelStatus::Candidate),
                entry("coding/python", ModelStatus::Retired),
                entry("bio/general", ModelStatus::Active),
            ],
            ..RegistryFile::default()
        };
        let strict = Catalogue::from_registry(&file, &taxonomy, false);
        assert_eq!(strict.specialists.len(), 2);
        assert!(strict.get("coding/rust").unwrap().enabled);
        assert!(!strict.get("pentest/recon").unwrap().enabled);
        assert_eq!(
            strict.get("pentest/recon").unwrap().category,
            "cyber-offensive"
        );
        assert_eq!(
            strict.unmapped.len(),
            2,
            "retired + bio: {:?}",
            strict.unmapped
        );
        let open = Catalogue::from_registry(&file, &taxonomy, true);
        assert!(open.get("pentest/recon").unwrap().enabled);
    }

    #[test]
    fn untrained_meta_router_makes_the_whole_router_abstain() {
        let router = LayeredRouter::from_config(&LayeredRouterConfig::default()).unwrap();
        let plan = router.route("rust").unwrap();
        assert!(plan
            .unsure
            .as_deref()
            .unwrap()
            .contains("no trained meta-router"));
        validate_plan(&plan, router.bounds()).unwrap();
        assert_eq!(plan.trace.len(), 1);
        assert_eq!(plan.trace[0].layer, RouteLayer::L3Meta);
    }

    #[test]
    fn train_router_files_writes_model_and_report() {
        let dir = std::env::temp_dir().join(format!("hrouter-train-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let traces_path = dir.join("traces.jsonl");
        let mut lines = String::new();
        // Repeat the fixture under fresh ids so a holdout split has members.
        for round in 0..4 {
            for t in fixture_traces() {
                let t = RouterTrace {
                    id: format!("{}-{round}", t.id),
                    ..t
                };
                lines.push_str(&serde_json::to_string(&t).unwrap());
                lines.push('\n');
            }
        }
        std::fs::write(&traces_path, lines).unwrap();
        let out = dir.join("out");
        let outcome = train_router_files(
            std::slice::from_ref(&traces_path),
            &RouterTrainOptions {
                eval: None,
                calibration: &[],
                out: &out,
                tiny: true,
                holdout_percent: 25,
                calibration_group: CalibrationGroup::Leaf,
                feature_set: "rich",
                encoders: Vec::new(),
            },
            &Taxonomy::builtin(),
        )
        .unwrap();
        assert!(outcome.refit);
        assert!(outcome.pooled.is_empty());
        assert!(outcome.render().contains("before tuning"));
        assert!(outcome.eval_traces > 0 && outcome.train_traces > 0);
        assert!((0.0..=1.0).contains(&outcome.eval.top1_accuracy));
        assert!((0.0..=1.0).contains(&outcome.eval.mean_set_f1));
        let loaded = HashedMetaRouter::load(&outcome.model_path).unwrap();
        assert!(loaded.trained_samples > 0);
        assert!(outcome.report_path.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn wilson_interval_matches_reference_values() {
        // Reference: Wilson 95% for 5/10 is [0.2366, 0.7634].
        let (lo, hi) = wilson_interval(5, 10);
        assert!(
            (lo - 0.2366).abs() < 1e-4 && (hi - 0.7634).abs() < 1e-4,
            "{lo} {hi}"
        );
        let (lo, hi) = wilson_interval(0, 10);
        assert_eq!(lo, 0.0);
        assert!((hi - 0.2775).abs() < 1e-4, "{hi}");
        let (lo, hi) = wilson_interval(10, 10);
        assert!((lo - 0.7225).abs() < 1e-4 && hi == 1.0, "{lo} {hi}");
        assert_eq!(wilson_interval(0, 0), (0.0, 1.0));
        let r = Rate::new(3, 4);
        assert!((r.rate - 0.75).abs() < 1e-12);
        assert!(r.render().starts_with("3/4 = 0.750 ["), "{}", r.render());
        assert!(Rate::new(0, 0).rate.is_nan());
    }

    #[test]
    fn none_margin_moves_the_abstention_boundary() {
        let s = |v: &[(&str, f64)]| v.iter().map(|(a, b)| ((*a).to_string(), *b)).collect();
        let base = MetaPolicy {
            confidence_floor: 0.0,
            ..MetaPolicy::default()
        };
        // p_none 0.35 < p_top 0.45: routed at margin 0, abstains at margin 0.2.
        assert!(base.decide(s(&[("coding", 0.45)]), 0.35).unsure.is_none());
        let wide = MetaPolicy {
            none_margin: 0.2,
            ..base.clone()
        };
        assert!(wide.decide(s(&[("coding", 0.45)]), 0.35).unsure.is_some());
        // A negative margin routes even when none is slightly ahead.
        let narrow = MetaPolicy {
            none_margin: -0.2,
            ..base
        };
        assert!(narrow.decide(s(&[("coding", 0.45)]), 0.5).unsure.is_none());
        assert!(MetaPolicy {
            none_margin: 2.0,
            ..MetaPolicy::default()
        }
        .validate()
        .is_err());
        // An old policy without the field still parses, at margin 0.
        let old: MetaPolicy = serde_json::from_str(
            r#"{"max_categories":3,"engage_ratio":0.3,"confidence_floor":0.25}"#,
        )
        .unwrap();
        assert_eq!(old.none_margin, 0.0);
    }

    #[test]
    fn tuning_uses_calibration_and_never_lowers_the_objective() {
        let taxonomy = Taxonomy::builtin();
        let mut traces = Vec::new();
        for round in 0..6 {
            for t in fixture_traces() {
                traces.push(RouterTrace {
                    id: format!("{}-{round}", t.id),
                    ..t
                });
            }
        }
        let mut meta = HashedMetaRouter::new(
            &taxonomy,
            FeatureConfig::tiny(),
            MetaTrainConfig {
                epochs: 20,
                calibration_percent: 30,
                ..MetaTrainConfig::default()
            },
            MetaPolicy::default(),
        )
        .unwrap();
        meta.train(&traces).unwrap();
        let (_, calibration) = meta.calibration_split(&traces);
        let tuning = tune_policy(&meta, &calibration).unwrap();
        assert!(
            tuning.calibration_oos > 0 && tuning.calibration_in_scope > 0,
            "{tuning:?}"
        );
        assert_eq!(tuning.grid_points, 25 * 13);
        assert!(tuning.chosen_objective >= tuning.default_objective);
        tuning.chosen.validate().unwrap();
        // Deterministic.
        assert_eq!(tune_policy(&meta, &calibration).unwrap(), tuning);
        // Without out-of-scope calibration traces the default is kept.
        let in_scope: Vec<RouterTrace> = calibration
            .into_iter()
            .filter(|t| !t.categories.is_empty())
            .collect();
        let kept = tune_policy(&meta, &in_scope).unwrap();
        assert_eq!(kept.chosen, MetaPolicy::default());
        assert!(kept.note.contains("not tuned"), "{}", kept.note);
    }

    #[test]
    fn eval_breakdown_separates_in_scope_oos_and_combo() {
        struct Fixed;
        impl MetaRouter for Fixed {
            fn name(&self) -> &str {
                "fixed"
            }
            fn categories(&self) -> &[String] {
                &[]
            }
            fn decide(&self, prompt: &str, _: Option<&str>) -> Result<MetaDecision> {
                // Abstains on anything mentioning pasta, routes the rest to coding.
                let unsure = prompt.contains("pasta").then(|| "x".to_string());
                Ok(MetaDecision {
                    scores: vec![("coding".into(), 0.9), ("math".into(), 0.1)],
                    none: 0.0,
                    engaged: if unsure.is_some() {
                        vec![]
                    } else {
                        vec![("coding".into(), 1.0)]
                    },
                    unsure,
                })
            }
        }
        let t = Taxonomy::builtin();
        let traces = parse_traces(
            "{\"id\":\"1\",\"text\":\"rust\",\"categories\":[{\"id\":\"coding\",\"weight\":1.0}],\"style\":\"short\"}\n\
             {\"id\":\"2\",\"text\":\"algebra\",\"categories\":[{\"id\":\"math\",\"weight\":1.0}]}\n\
             {\"id\":\"3\",\"text\":\"pasta\",\"categories\":[]}\n\
             {\"id\":\"4\",\"text\":\"pasta rust\",\"categories\":[{\"id\":\"coding\",\"weight\":1.0}]}\n\
             {\"id\":\"5\",\"text\":\"weather\",\"categories\":[]}\n\
             {\"id\":\"6\",\"text\":\"rust maths\",\"categories\":[{\"id\":\"coding\",\"weight\":0.5},{\"id\":\"math\",\"weight\":0.5}]}",
            &t,
        )
        .unwrap();
        let e = evaluate_meta(&Fixed, &traces, None).unwrap();
        assert_eq!((e.in_scope_top1.hits, e.in_scope_top1.n), (3, 4));
        assert_eq!((e.in_scope_routed.hits, e.in_scope_routed.n), (3, 4));
        assert_eq!(
            e.in_scope_routed_correct.hits, 2,
            "the abstained coding trace is a miss"
        );
        assert_eq!((e.oos_unsure_recall.hits, e.oos_unsure_recall.n), (1, 2));
        assert_eq!(
            (e.oos_unsure_precision.hits, e.oos_unsure_precision.n),
            (1, 2)
        );
        assert_eq!(e.combo_traces, 1);
        assert!((e.combo_set_f1 - 2.0 / 3.0).abs() < 1e-12);
        assert!((e.balanced_accuracy - (0.5 * 0.75 + 0.5 * 0.5)).abs() < 1e-12);
        let styles: Vec<&str> = e.by_style.iter().map(|s| s.style.as_str()).collect();
        assert_eq!(styles, vec!["short", "single", "oos", "combo"]);
        let math = e.by_category.iter().find(|(c, _)| c == "math").unwrap();
        assert_eq!((math.1.hits, math.1.n), (0, 1));
    }

    #[test]
    fn several_trace_files_read_as_one_set() {
        let dir = std::env::temp_dir().join(format!("hrouter-multi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.jsonl");
        let b = dir.join("b.jsonl");
        std::fs::write(&a, "{\"id\":\"1\",\"text\":\"x\",\"categories\":[]}\n").unwrap();
        std::fs::write(
            &b,
            "{\"id\":\"2\",\"text\":\"y\",\"categories\":[],\"style\":\"oos\"}\n",
        )
        .unwrap();
        let t = Taxonomy::builtin();
        let all = read_trace_files(&[a.clone(), b.clone()], &t).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].style.as_deref(), Some("oos"));
        let err = read_trace_files(&[a.clone(), a], &t)
            .unwrap_err()
            .to_string();
        assert!(err.contains("repeats an id"), "{err}");
        assert!(read_trace_files(&[], &t).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn encoder_without_a_sidecar_is_refused_and_loader_dispatches() {
        let dir = std::env::temp_dir().join(format!("hrouter-enc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bare = dir.join("lm-bare.mpk");
        std::fs::write(&bare, b"not a checkpoint").unwrap();
        let err = LmEncoder::load(&bare).err().unwrap().to_string();
        assert!(err.contains("sidecar"), "{err}");

        let taxonomy = Taxonomy::builtin();
        let mut meta = HashedMetaRouter::new(
            &taxonomy,
            FeatureConfig::tiny(),
            MetaTrainConfig {
                epochs: 5,
                calibration_percent: 0,
                ..MetaTrainConfig::default()
            },
            MetaPolicy::default(),
        )
        .unwrap();
        meta.train(&fixture_traces()).unwrap();
        let path = dir.join("meta-router.json");
        meta.save(&path).unwrap();
        let loaded = load_meta_router(&path).unwrap();
        assert_eq!(loaded.name(), "hashed-softmax");
        assert_eq!(
            loaded.decide("rust", None).unwrap(),
            meta.decide("rust", None).unwrap()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn squash_and_rich_feature_presets() {
        assert_eq!(squash_repeats("sooo  slooow!!"), "so  slow!!");
        assert_eq!(squash_repeats("kubernetes"), "kubernetes");
        let squash = FeatureConfig::preset("squash", false).unwrap();
        assert!(squash.squash_repeats && !squash.word_boundary_grams);
        assert_eq!(
            featurize("hellooo wooorld", None, &squash),
            featurize("helo world", None, &squash)
        );
        let base = FeatureConfig::preset("base", false).unwrap();
        assert_eq!(base, FeatureConfig::default());
        assert_ne!(
            featurize("hellooo", None, &base),
            featurize("helo", None, &base)
        );
        let rich = FeatureConfig::preset("rich", false).unwrap();
        assert!(rich.squash_repeats && rich.word_boundary_grams);
        assert_eq!(rich.hash_bits, base.hash_bits + 2);
        assert_ne!(
            featurize("rust lifetimes", None, &rich),
            featurize("rust lifetimes", None, &squash),
            "boundary grams add features"
        );
        assert!(FeatureConfig::preset("bogus", false).is_err());
        // A model saved before the fields existed loads with both off.
        let old: FeatureConfig = serde_json::from_str(r#"{"hash_bits":12}"#).unwrap();
        assert!(!old.squash_repeats && !old.word_boundary_grams);
    }

    #[test]
    fn external_calibration_is_used_and_may_not_overlap_the_eval() {
        let dir = std::env::temp_dir().join(format!("hrouter-extcal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, rounds: std::ops::Range<usize>| {
            let mut lines = String::new();
            for round in rounds {
                for t in fixture_traces() {
                    let t = RouterTrace {
                        id: format!("{}-{name}-{round}", t.id),
                        text: format!("{} ({name} {round})", t.text),
                        ..t
                    };
                    lines.push_str(&serde_json::to_string(&t).unwrap());
                    lines.push('\n');
                }
            }
            let path = dir.join(format!("{name}.jsonl"));
            std::fs::write(&path, lines).unwrap();
            path
        };
        let train = write("train", 0..3);
        let cal = write("cal", 0..1);
        let eval = write("eval", 0..1);
        let out = dir.join("out");
        fn options<'a>(
            eval: &'a Path,
            calibration: &'a [PathBuf],
            out: &'a Path,
        ) -> RouterTrainOptions<'a> {
            RouterTrainOptions {
                eval: Some(eval),
                calibration,
                out,
                tiny: true,
                holdout_percent: 20,
                calibration_group: CalibrationGroup::Leaf,
                feature_set: "base",
                encoders: Vec::new(),
            }
        }
        let cal_files = vec![cal.clone()];
        let outcome = train_router_files(
            std::slice::from_ref(&train),
            &options(&eval, &cal_files, &out),
            &Taxonomy::builtin(),
        )
        .unwrap();
        assert_eq!(outcome.calibration_traces, fixture_traces().len());
        assert_eq!(outcome.train.calibration_samples, fixture_traces().len());
        assert_eq!(
            outcome.train.fit_samples,
            3 * fixture_traces().len(),
            "all traces fitted"
        );
        assert!(!outcome.refit, "no refit with an external calibration set");
        assert!(outcome.calibration_source.contains("cal.jsonl"));
        // The same texts as the eval (only the ids differ) count as a leak.
        let leaky = vec![eval.clone()];
        let err = train_router_files(
            std::slice::from_ref(&train),
            &options(&eval, &leaky, &out),
            &Taxonomy::builtin(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("leak"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn certificates_pass() {
        let certs = certificates();
        assert_eq!(certs.len(), 4);
        for c in &certs {
            assert!(
                c.passed(),
                "{}: residual {} > {}",
                c.name,
                c.residual,
                c.tolerance
            );
            assert_eq!(c.group, "router");
        }
    }
}
