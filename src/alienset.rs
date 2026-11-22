//! Alien Space of Science, natively: idea atoms, a coherence model over atom
//! sets, a non-parametric author-availability score, and their z-fusion.
//!
//! Method: Artiles et al., *The Alien Space of Science*, arXiv:2603.01092v2
//! (2026); data and reference code MIT,
//! <https://github.com/alejandrohdez00/alien-space-of-science>. This module
//! uses **only the authors' released data files** (exported to flat files by
//! `alien-space/tools/export_alien.py`). None of their code is run and none of
//! their model weights are loaded; the models here are trained from scratch.
//!
//! # The objects
//!
//! - **Atoms.** 16,068 LLM papers were reduced by the authors' LLM pipeline to
//!   "conceptual units" and clustered into 273 idea atoms (80%-coverage
//!   clustering). A paper is the *set* of atoms its units fell into (noise
//!   units, cluster `-1`, dropped). [`AtomCatalog`] loads and validates that.
//! - **Coherence** `C(S)`: how plausible a set of atoms is as one paper. A
//!   small causal transformer ([`CoherenceModel`]) over atom tokens is trained
//!   with the next-atom objective on random serialisations of each paper's set
//!   (`BOS a_π(1) … a_π(k) EOS`, BOS and EOS one shared token as in the paper).
//! - **Availability** `A(S)`: how close the existing author community already
//!   is to `S`. The paper learns a dual encoder; this module uses a
//!   non-parametric stand-in ([`AvailabilityIndex`]), documented below.
//! - **Fusion** `F(S) = (1−β)·z(C) + β·z(U)`, `U = −A`, z-scored within the
//!   candidate pool ([`fuse`]). High `F` = coherent but far from the community.
//!
//! # The coherence estimator (exactly what is computed)
//!
//! For one ordering `π` of a `k`-set `S`, with the model's log-probabilities
//! over its `n_atoms + 1` outputs (atoms and EOS; no renormalisation),
//!
//! ```text
//! c_π(S) = (1/k) Σ_{t=1..k} log p(a_π(t) | BOS, a_π(1), …, a_π(t−1))
//! ```
//!
//! (EOS is *not* scored: `C` measures the atoms, not the stopping decision).
//! `C(S)` is the **arithmetic mean of `c_π` over orderings**: all `k!` of them
//! when `k ≤ exhaustive_max` (default 4, so 6 orderings for triples and 24 for
//! quadruples, as in the paper's diagnostics), otherwise `samples` orderings
//! drawn by a ChaCha8 stream seeded from the sorted set and the policy seed,
//! so the score of a set is a deterministic function of the set. The mean of
//! log-likelihoods is a lower bound (Jensen) on the log of the
//! permutation-marginal likelihood `log mean_π exp(k·c_π)/k`; it is used
//! because it is what the paper's per-permutation diagnostics average, and
//! because it is invariant to the order the set is given in, which is
//! certified (`alienset/coherence_is_order_and_relabel_invariant`).
//!
//! For `k = 3` over a whole vocabulary, [`TripleTable`] computes the same
//! number from one forward pass per ordered prefix `(a, b)`: the pass on
//! `[BOS, a, b]` yields `log p(a | BOS)`, `log p(b | BOS, a)` and the full
//! distribution `log p(· | BOS, a, b)`, so all `n(n−1)(n−2)` ordered
//! triples cost `n(n−1)` passes. Equality with the direct estimator is a test.
//!
//! # The availability approximation (read before trusting `A`)
//!
//! Each author's repertoire `R_i` is the union of the atoms of their papers.
//! With inverse-document-frequency weights over repertoires,
//! `w_a = ln((1 + N) / (1 + df_a)) + 1`, an author's similarity to a set is
//! the cosine between the weighted indicator vectors,
//!
//! ```text
//! sim_i(S) = Σ_{a ∈ S ∩ R_i} w_a² / (‖w ⊙ 1_S‖ · ‖w ⊙ 1_{R_i}‖)
//! ```
//!
//! and `A(S)` is the **median of the top `m` similarities over all authors**
//! (`m = 10` as in the paper; authors with no overlap count as 0, so a set
//! that fewer than `m` authors touch has a low median). This replaces the
//! paper's learned `ψ(R_i)ᵀφ(S)`: it sees only exact atom overlap, so it
//! cannot generalise "author X works on neighbours of a" the way a trained
//! encoder can, and it rewards small repertoires (the cosine denominator).
//! It is an approximation and is reported as one; the validation measures
//! how well it separates real future paper sets from random ones.
//!
//! # Honesty
//!
//! A high `F` marks a *candidate*: coherent under a model of 2017–2025 LLM
//! papers and far from the authors who wrote them. It is not evidence that
//! the idea works, is novel outside this corpus, or is worth a pilot. The
//! corpus is LLM papers only; the atoms were produced by an LLM pipeline; the
//! paper's own judge evaluations share the sampler's criterion and its human
//! samples are small. The authors' released coherence "validation" split is
//! record-level (every validation paper also has training serialisations),
//! so this module makes its own paper-level splits ([`SplitRule`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use burn_optim::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module, Param};
use burn_nn::attention::generate_autoregressive_mask;
use burn_nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn_nn::{Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn_optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::interop::write_atomic;
use crate::verify::Certificate;

/// The CPU inference backend.
pub type Cpu = burn_ndarray::NdArray<f32>;
/// The CPU training backend.
pub type CpuTrain = burn_autodiff::Autodiff<Cpu>;

/// The paper's `m`: availability is the median of the top-`m` author scores.
pub const DEFAULT_TOP_M: usize = 10;
/// The paper's main-experiment fusion weight.
pub const DEFAULT_BETA: f64 = 0.7;
/// Orderings are enumerated exhaustively up to this set size (`4! = 24`).
pub const DEFAULT_EXHAUSTIVE_MAX: usize = 4;
/// The largest atom vocabulary accepted (token ids must fit with BOS and PAD).
const MAX_ATOMS: usize = 60_000;

// ------------------------------------------------------------------ catalog --

/// One idea atom: a cluster of conceptual units with its canonical text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    pub id: u16,
    /// Papers whose atom set contains this atom (from the export).
    pub n_papers: usize,
    /// The cluster's canonical statement (`clusters[id].name`).
    pub name: String,
    /// The clustering pipeline's rationale for the cluster.
    pub description: String,
}

/// One paper as an atom set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paper {
    pub paper_id: String,
    #[serde(default)]
    pub year: Option<u16>,
    /// Sorted, distinct atom ids.
    pub atoms: Vec<u16>,
}

/// One author: the papers they are linked to and the union of their atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Author {
    /// Opaque key (a hash of the source id; no names are exported).
    pub key: String,
    /// Indices into [`AtomCatalog::papers`].
    pub papers: Vec<u32>,
    /// Sorted, distinct atom ids: the union over `papers`.
    pub atoms: Vec<u16>,
}

#[derive(Deserialize)]
struct AuthorLine {
    author: String,
    papers: Vec<String>,
    atoms: Vec<u16>,
}

/// Row counts of a loaded catalog, for comparison with the paper's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogCounts {
    pub atoms: usize,
    pub papers: usize,
    pub papers_with_atoms: usize,
    pub papers_with_two_or_more: usize,
    pub authors: usize,
    pub authors_with_atoms: usize,
    pub authors_with_two_or_more: usize,
}

/// The validated atom catalog: atoms, papers as atom sets, and authors.
#[derive(Debug, Clone)]
pub struct AtomCatalog {
    atoms: Vec<Atom>,
    papers: Vec<Paper>,
    authors: Vec<Author>,
}

fn strictly_increasing(xs: &[u16]) -> bool {
    xs.windows(2).all(|w| matches!(w, [a, b] if a < b))
}

impl AtomCatalog {
    /// Load `atoms.tsv`, `paper_atoms.jsonl` and `author_repertoires.jsonl`
    /// from an export directory and validate them together.
    pub fn load(dir: &Path) -> Result<Self> {
        let atoms_path = dir.join("atoms.tsv");
        let text = std::fs::read_to_string(&atoms_path)
            .with_context(|| format!("reading {}", atoms_path.display()))?;
        let atoms = parse_atoms_tsv(&text).with_context(|| format!("parsing {}", atoms_path.display()))?;
        let papers_path = dir.join("paper_atoms.jsonl");
        let papers: Vec<Paper> = read_jsonl(&papers_path)?;
        let authors_path = dir.join("author_repertoires.jsonl");
        let lines: Vec<AuthorLine> = read_jsonl(&authors_path)?;
        let authors = lines.into_iter().map(|l| (l.author, l.papers, l.atoms)).collect();
        Self::from_parts(atoms, papers, authors)
    }

    /// Build from parts. Every structural promise is checked and a failure
    /// names the offending record: atom ids are exactly `0..n` in order,
    /// paper ids are unique, atom sets are sorted, distinct and in range,
    /// author keys are unique, every author paper exists, and every author's
    /// atoms are exactly the union of their papers' atoms.
    pub fn from_parts(
        atoms: Vec<Atom>,
        papers: Vec<Paper>,
        authors: Vec<(String, Vec<String>, Vec<u16>)>,
    ) -> Result<Self> {
        ensure!(!atoms.is_empty(), "an atom catalog needs at least one atom");
        ensure!(atoms.len() <= MAX_ATOMS, "{} atoms exceed the supported {MAX_ATOMS}", atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            ensure!(usize::from(atom.id) == i, "atom ids must be 0..n in order: row {i} has id {}", atom.id);
            ensure!(!atom.name.trim().is_empty(), "atom {} has an empty canonical name", atom.id);
        }
        let n = atoms.len();
        let mut index: HashMap<&str, u32> = HashMap::with_capacity(papers.len());
        for (i, paper) in papers.iter().enumerate() {
            ensure!(!paper.paper_id.is_empty(), "paper row {i} has an empty id");
            let slot = u32::try_from(i).context("more than u32::MAX papers")?;
            ensure!(index.insert(paper.paper_id.as_str(), slot).is_none(), "paper id '{}' appears twice", paper.paper_id);
            ensure!(strictly_increasing(&paper.atoms), "paper '{}' atoms are not sorted and distinct: {:?}", paper.paper_id, paper.atoms);
            if let Some(&bad) = paper.atoms.iter().find(|&&a| usize::from(a) >= n) {
                bail!("paper '{}' names atom {bad}, the catalog has {n}", paper.paper_id);
            }
            if let Some(year) = paper.year {
                ensure!((1900..=2100).contains(&year), "paper '{}' has implausible year {year}", paper.paper_id);
            }
        }
        let mut keys: HashSet<&str> = HashSet::with_capacity(authors.len());
        let mut built = Vec::with_capacity(authors.len());
        for (key, paper_ids, author_atoms) in &authors {
            ensure!(keys.insert(key.as_str()), "author key '{key}' appears twice");
            ensure!(!paper_ids.is_empty(), "author '{key}' has no papers");
            let mut slots = Vec::with_capacity(paper_ids.len());
            let mut union: Vec<u16> = Vec::new();
            for pid in paper_ids {
                let &slot = index.get(pid.as_str()).ok_or_else(|| anyhow::anyhow!("author '{key}' links unknown paper '{pid}'"))?;
                slots.push(slot);
                let paper = papers.get(slot as usize).ok_or_else(|| anyhow::anyhow!("paper slot {slot} out of range"))?;
                union.extend_from_slice(&paper.atoms);
            }
            union.sort_unstable();
            union.dedup();
            ensure!(
                &union == author_atoms,
                "author '{key}' lists atoms {author_atoms:?}, the union of their papers is {union:?}"
            );
            slots.sort_unstable();
            slots.dedup();
            built.push(Author { key: key.clone(), papers: slots, atoms: union });
        }
        Ok(Self { atoms, papers, authors: built })
    }

    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    pub fn papers(&self) -> &[Paper] {
        &self.papers
    }

    pub fn authors(&self) -> &[Author] {
        &self.authors
    }

    /// Row counts, for the comparison with the paper (273 atoms, 16,068
    /// papers, 16,020 with atoms, 15,203 with at least two; 46,329 author
    /// profiles, 44,809 with at least two atoms).
    pub fn counts(&self) -> CatalogCounts {
        CatalogCounts {
            atoms: self.atoms.len(),
            papers: self.papers.len(),
            papers_with_atoms: self.papers.iter().filter(|p| !p.atoms.is_empty()).count(),
            papers_with_two_or_more: self.papers.iter().filter(|p| p.atoms.len() >= 2).count(),
            authors: self.authors.len(),
            authors_with_atoms: self.authors.iter().filter(|a| !a.atoms.is_empty()).count(),
            authors_with_two_or_more: self.authors.iter().filter(|a| a.atoms.len() >= 2).count(),
        }
    }

    /// Author repertoires built only from papers with `year ≤ max_year`
    /// (every paper when `None`; a paper without a year is then excluded
    /// from a year-bounded build). Empty repertoires are dropped.
    pub fn repertoires(&self, max_year: Option<u16>) -> Vec<Vec<u16>> {
        let allowed = |paper: &Paper| match max_year {
            None => true,
            Some(limit) => paper.year.is_some_and(|y| y <= limit),
        };
        let mut out = Vec::with_capacity(self.authors.len());
        for author in &self.authors {
            let mut atoms: Vec<u16> = author
                .papers
                .iter()
                .filter_map(|&slot| self.papers.get(slot as usize))
                .filter(|p| allowed(p))
                .flat_map(|p| p.atoms.iter().copied())
                .collect();
            atoms.sort_unstable();
            atoms.dedup();
            if !atoms.is_empty() {
                out.push(atoms);
            }
        }
        out
    }

    /// Split the papers with at least `min_atoms` atoms. A deterministic
    /// `val_permille` share of the training-eligible papers (by a hash of the
    /// paper id) is held out for early stopping.
    pub fn split(&self, rule: SplitRule, val_permille: u32, min_atoms: usize) -> Result<Split> {
        ensure!(val_permille < 1000, "val_permille must be below 1000, got {val_permille}");
        ensure!(min_atoms >= 1, "min_atoms must be at least 1");
        let mut split = Split { rule, train: Vec::new(), val: Vec::new(), test: Vec::new(), skipped_small: 0, skipped_no_year: 0 };
        for (i, paper) in self.papers.iter().enumerate() {
            if paper.atoms.len() < min_atoms {
                split.skipped_small += 1;
                continue;
            }
            let trainable = match rule {
                SplitRule::All => true,
                SplitRule::Temporal { last_train_year } => match paper.year {
                    None => {
                        split.skipped_no_year += 1;
                        continue;
                    }
                    Some(y) => y <= last_train_year,
                },
            };
            if !trainable {
                split.test.push(i);
            } else if id_permille(&paper.paper_id) < u64::from(val_permille) {
                split.val.push(i);
            } else {
                split.train.push(i);
            }
        }
        ensure!(!split.train.is_empty(), "the split {rule:?} leaves no training papers");
        ensure!(!split.val.is_empty(), "the split {rule:?} leaves no validation papers (val_permille {val_permille})");
        Ok(split)
    }

    /// The largest year among papers the rule trains on (`None` for `All`).
    pub fn train_year_limit(rule: SplitRule) -> Option<u16> {
        match rule {
            SplitRule::All => None,
            SplitRule::Temporal { last_train_year } => Some(last_train_year),
        }
    }
}

/// `sha256(id)` mod 1000, a stable split key.
fn id_permille(id: &str) -> u64 {
    let digest = Sha256::digest(id.as_bytes());
    let mut head = [0u8; 8];
    head.copy_from_slice(digest.get(..8).unwrap_or(&[0u8; 8]));
    u64::from_le_bytes(head) % 1000
}

fn parse_atoms_tsv(text: &str) -> Result<Vec<Atom>> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| anyhow::anyhow!("empty atoms.tsv"))?;
    ensure!(header.starts_with("atom_id\t"), "atoms.tsv must start with an atom_id header, got '{header}'");
    let mut atoms = Vec::new();
    for (row, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let [id, n_papers, _assignments, _confidence, name, description] = fields.as_slice() else {
            bail!("atoms.tsv row {} has {} fields, expected 6", row + 2, fields.len());
        };
        atoms.push(Atom {
            id: id.parse().with_context(|| format!("atoms.tsv row {}: atom id '{id}'", row + 2))?,
            n_papers: n_papers.parse().with_context(|| format!("atoms.tsv row {}: n_papers '{n_papers}'", row + 2))?,
            name: (*name).to_string(),
            description: (*description).to_string(),
        });
    }
    Ok(atoms)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).with_context(|| format!("{} line {}", path.display(), i + 1)))
        .collect()
}

/// Which papers train the coherence model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SplitRule {
    /// Train on years `≤ last_train_year`; later papers are the test set.
    /// This mirrors the paper's temporal held-out test (train ≤2024, 2025).
    Temporal { last_train_year: u16 },
    /// Train on every year (for ranking ideas about the future); no test set.
    All,
}

/// Paper indices of one split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Split {
    pub rule: SplitRule,
    pub train: Vec<usize>,
    pub val: Vec<usize>,
    pub test: Vec<usize>,
    pub skipped_small: usize,
    pub skipped_no_year: usize,
}

// -------------------------------------------------------------------- model --

/// Size and vocabulary of the coherence transformer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoherenceConfig {
    pub n_atoms: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    #[serde(default)]
    pub dropout: f64,
    /// Longest input (BOS plus atoms) the position table covers.
    pub max_len: usize,
}

impl CoherenceConfig {
    /// Reject a configuration that cannot build, naming the field.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.n_atoms >= 2 && self.n_atoms <= MAX_ATOMS, "n_atoms must be in 2..={MAX_ATOMS}, got {}", self.n_atoms);
        ensure!(self.d_model >= 1, "d_model must be positive");
        ensure!(self.n_heads >= 1 && self.d_model % self.n_heads == 0, "d_model {} must be divisible by n_heads {}", self.d_model, self.n_heads);
        ensure!(self.n_layers >= 1, "n_layers must be positive");
        ensure!(self.d_ff >= 1, "d_ff must be positive");
        ensure!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout), "dropout must be in [0, 1), got {}", self.dropout);
        ensure!(self.max_len >= 2, "max_len must be at least 2 (BOS plus one atom), got {}", self.max_len);
        Ok(())
    }

    /// The shared BOS/EOS token.
    pub fn bos(&self) -> usize {
        self.n_atoms
    }

    /// Input padding token (never a target).
    pub fn pad(&self) -> usize {
        self.n_atoms + 1
    }

    /// Output classes: every atom plus EOS.
    pub fn n_outputs(&self) -> usize {
        self.n_atoms + 1
    }

    /// Build a freshly initialised model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<CoherenceModel<B>> {
        self.validate()?;
        let small = Initializer::Normal { mean: 0.0, std: 0.02 };
        Ok(CoherenceModel {
            token: EmbeddingConfig::new(self.n_atoms + 2, self.d_model).with_initializer(small.clone()).init(device),
            position: EmbeddingConfig::new(self.max_len, self.d_model).with_initializer(small).init(device),
            encoder: TransformerEncoderConfig::new(self.d_model, self.d_ff, self.n_heads, self.n_layers)
                .with_dropout(self.dropout)
                .with_norm_first(true)
                .init(device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            head: LinearConfig::new(self.d_model, self.n_outputs()).init(device),
        })
    }
}

/// A pre-norm causal transformer over atom tokens.
#[derive(Module, Debug)]
pub struct CoherenceModel<B: Backend> {
    token: Embedding<B>,
    position: Embedding<B>,
    encoder: TransformerEncoder<B>,
    norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: Backend> CoherenceModel<B> {
    /// Logits `[batch, len, n_atoms + 1]` for token ids `[batch, len]`.
    /// The caller guarantees `len ≤ max_len` ([`Self::logits`] checks it).
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, len] = tokens.dims();
        let device = tokens.device();
        let positions = Tensor::<B, 1, Int>::arange(0..len as i64, &device)
            .reshape([1, len])
            .repeat_dim(0, batch);
        let x = self.token.forward(tokens) + self.position.forward(positions);
        let mask = generate_autoregressive_mask::<B>(batch, len, &device);
        let h = self.encoder.forward(TransformerEncoderInput::new(x).mask_attn(mask));
        self.head.forward(self.norm.forward(h))
    }

    /// Checked forward: refuses inputs longer than the position table.
    pub fn logits(&self, config: &CoherenceConfig, tokens: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>> {
        let [_, len] = tokens.dims();
        ensure!(len >= 1 && len <= config.max_len, "input length {len} outside 1..={}", config.max_len);
        Ok(self.forward(tokens))
    }

    /// The same model with atom ids relabelled by `sigma` (atom `a` becomes
    /// `sigma[a]`): embedding rows and output columns are permuted, BOS, PAD
    /// and EOS stay. The relabelled model scores `sigma(S)` exactly as this
    /// one scores `S`, which is what the relabelling certificate checks.
    fn relabelled(&self, config: &CoherenceConfig, sigma: &[usize]) -> Result<Self> {
        let n = config.n_atoms;
        ensure!(sigma.len() == n, "relabelling has {} entries for {n} atoms", sigma.len());
        let mut inverse = vec![usize::MAX; n];
        for (a, &s) in sigma.iter().enumerate() {
            let slot = inverse.get_mut(s).ok_or_else(|| anyhow::anyhow!("relabelling maps {a} to {s}, out of range"))?;
            ensure!(*slot == usize::MAX, "relabelling maps two atoms to {s}");
            *slot = a;
        }
        let device = self.head.weight.val().device();
        let rows: Vec<i64> = inverse.iter().map(|&a| a as i64).chain([n as i64, n as i64 + 1]).collect();
        let cols: Vec<i64> = inverse.iter().map(|&a| a as i64).chain([n as i64]).collect();
        let rows = Tensor::<B, 1, Int>::from_ints(rows.as_slice(), &device);
        let cols = Tensor::<B, 1, Int>::from_ints(cols.as_slice(), &device);
        let mut out = self.clone();
        out.token.weight = out.token.weight.map(|w| w.select(0, rows));
        out.head.weight = out.head.weight.map(|w| w.select(1, cols.clone()));
        out.head.bias = out.head.bias.map(|b| b.map(|b| b.select(0, cols)));
        Ok(out)
    }
}

/// A trained model with its configuration (inference side).
#[derive(Debug, Clone)]
pub struct CoherenceLm<B: Backend> {
    pub config: CoherenceConfig,
    pub model: CoherenceModel<B>,
}

// -------------------------------------------------------------- batching --

/// Teacher-forcing rows: input `BOS s_1 … s_k`, target `s_1 … s_k EOS`.
struct Batch {
    rows: usize,
    len: usize,
    inputs: Vec<i64>,
    targets: Vec<i64>,
    /// 1 on every real target (atoms and EOS).
    mask: Vec<f32>,
    /// 1 on atom targets only (EOS excluded): what `C` scores.
    atom_mask: Vec<f32>,
}

impl Batch {
    fn new(seqs: &[Vec<u16>], config: &CoherenceConfig) -> Result<Self> {
        ensure!(!seqs.is_empty(), "an empty batch");
        let longest = seqs.iter().map(Vec::len).max().unwrap_or(0);
        let len = longest + 1;
        ensure!(len <= config.max_len, "a sequence of {longest} atoms needs {len} positions, max_len is {}", config.max_len);
        let rows = seqs.len();
        let mut batch = Batch {
            rows,
            len,
            inputs: vec![config.pad() as i64; rows * len],
            targets: vec![0; rows * len],
            mask: vec![0.0; rows * len],
            atom_mask: vec![0.0; rows * len],
        };
        for (r, seq) in seqs.iter().enumerate() {
            ensure!(!seq.is_empty(), "row {r} is an empty sequence");
            let base = r * len;
            let row_in = batch.inputs.get_mut(base..base + len).ok_or_else(|| anyhow::anyhow!("batch row {r} out of range"))?;
            if let Some(first) = row_in.first_mut() {
                *first = config.bos() as i64;
            }
            for (slot, &a) in row_in.iter_mut().skip(1).zip(seq.iter()) {
                ensure!(usize::from(a) < config.n_atoms, "atom {a} outside the vocabulary of {}", config.n_atoms);
                *slot = i64::from(a);
            }
            let row_t = batch.targets.get_mut(base..base + len).ok_or_else(|| anyhow::anyhow!("batch row {r} out of range"))?;
            for (slot, value) in row_t.iter_mut().zip(seq.iter().map(|&a| i64::from(a)).chain([config.bos() as i64])) {
                *slot = value;
            }
            for (t, (m, am)) in batch
                .mask
                .get_mut(base..base + len)
                .into_iter()
                .flatten()
                .zip(batch.atom_mask.get_mut(base..base + len).into_iter().flatten())
                .enumerate()
            {
                if t <= seq.len() {
                    *m = 1.0;
                }
                if t < seq.len() {
                    *am = 1.0;
                }
            }
        }
        Ok(batch)
    }

    /// `log p(target)` at every position, `[rows, len]`.
    fn picked<B: Backend>(&self, model: &CoherenceModel<B>, config: &CoherenceConfig, device: &B::Device) -> Result<Tensor<B, 2>> {
        let inputs = Tensor::<B, 1, Int>::from_ints(self.inputs.as_slice(), device).reshape([self.rows, self.len]);
        let targets = Tensor::<B, 1, Int>::from_ints(self.targets.as_slice(), device).reshape([self.rows, self.len, 1]);
        let logp = log_softmax(model.logits(config, inputs)?, 2);
        Ok(logp.gather(2, targets).reshape([self.rows, self.len]))
    }
}

fn host_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("reading a float tensor: {err:?}"))
}

// --------------------------------------------------------------- scoring --

/// How orderings of a set are marginalised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationPolicy {
    /// All `k!` orderings when `k ≤ exhaustive_max`.
    pub exhaustive_max: usize,
    /// Orderings drawn for larger sets.
    pub samples: usize,
    pub seed: u64,
}

impl Default for PermutationPolicy {
    fn default() -> Self {
        Self { exhaustive_max: DEFAULT_EXHAUSTIVE_MAX, samples: 24, seed: 0x00A1_1E45 }
    }
}

/// Every permutation of `0..k` in lexicographic order.
pub fn all_permutations(k: usize) -> Vec<Vec<usize>> {
    let mut current: Vec<usize> = (0..k).collect();
    let mut out = vec![current.clone()];
    loop {
        // Standard next-permutation: find the last ascent.
        let Some(i) = (1..k).rev().find(|&i| current.get(i - 1) < current.get(i)) else {
            return out;
        };
        let pivot = i - 1;
        let Some(j) = (i..k).rev().find(|&j| current.get(j) > current.get(pivot)) else {
            return out;
        };
        current.swap(pivot, j);
        if let Some(tail) = current.get_mut(i..) {
            tail.reverse();
        }
        out.push(current.clone());
    }
}

fn set_seed(set: &[u16], seed: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    for a in set {
        hasher.update(a.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut head = [0u8; 8];
    head.copy_from_slice(digest.get(..8).unwrap_or(&[0u8; 8]));
    u64::from_le_bytes(head)
}

/// The orderings of `set` the policy scores (as atom sequences).
pub fn orderings(set: &[u16], policy: &PermutationPolicy) -> Vec<Vec<u16>> {
    let mut sorted = set.to_vec();
    sorted.sort_unstable();
    if sorted.len() <= policy.exhaustive_max {
        all_permutations(sorted.len())
            .into_iter()
            .map(|p| p.into_iter().filter_map(|i| sorted.get(i).copied()).collect())
            .collect()
    } else {
        let mut rng = ChaCha8Rng::seed_from_u64(set_seed(&sorted, policy.seed));
        (0..policy.samples.max(1))
            .map(|_| {
                let mut order = sorted.clone();
                order.shuffle(&mut rng);
                order
            })
            .collect()
    }
}

/// Rows scored per forward pass.
const SCORE_CHUNK: usize = 4096;

/// Per-row `(Σ atom log p, atoms)` for the given sequences.
fn sequence_log_probs<B: Backend>(lm: &CoherenceLm<B>, seqs: &[Vec<u16>], device: &B::Device) -> Result<Vec<(f64, usize)>> {
    let mut out = Vec::with_capacity(seqs.len());
    for chunk in seqs.chunks(SCORE_CHUNK) {
        let batch = Batch::new(chunk, &lm.config)?;
        let picked = host_f32(batch.picked(&lm.model, &lm.config, device)?)?;
        for r in 0..batch.rows {
            let range = r * batch.len..(r + 1) * batch.len;
            let lp = picked.get(range.clone()).ok_or_else(|| anyhow::anyhow!("score row {r} out of range"))?;
            let am = batch.atom_mask.get(range).ok_or_else(|| anyhow::anyhow!("mask row {r} out of range"))?;
            let total: f64 = lp.iter().zip(am).map(|(&l, &m)| f64::from(l) * f64::from(m)).sum();
            let count = am.iter().filter(|&&m| m > 0.0).count();
            out.push((total, count));
        }
    }
    Ok(out)
}

/// `C(S)` for every set (see the module docs for the estimator).
pub fn coherence_scores<B: Backend>(lm: &CoherenceLm<B>, sets: &[Vec<u16>], policy: &PermutationPolicy, device: &B::Device) -> Result<Vec<f64>> {
    let mut seqs = Vec::new();
    let mut owners = Vec::new();
    for (i, set) in sets.iter().enumerate() {
        ensure!(!set.is_empty(), "set {i} is empty");
        let mut distinct = set.clone();
        distinct.sort_unstable();
        distinct.dedup();
        ensure!(distinct.len() == set.len(), "set {i} repeats an atom: {set:?}");
        for order in orderings(set, policy) {
            seqs.push(order);
            owners.push(i);
        }
    }
    let scored = sequence_log_probs(lm, &seqs, device)?;
    let mut sums = vec![0.0f64; sets.len()];
    let mut counts = vec![0usize; sets.len()];
    for (&owner, (total, k)) in owners.iter().zip(scored) {
        ensure!(k > 0, "a scored ordering had no atoms");
        if let (Some(s), Some(c)) = (sums.get_mut(owner), counts.get_mut(owner)) {
            *s += total / k as f64;
            *c += 1;
        }
    }
    Ok(sums.iter().zip(&counts).map(|(&s, &c)| s / c.max(1) as f64).collect())
}

/// The coherence of one ordering only (no marginalisation): the
/// order-dependent control the invariance certificate must catch.
fn single_order_score<B: Backend>(lm: &CoherenceLm<B>, order: &[u16], device: &B::Device) -> Result<f64> {
    let scored = sequence_log_probs(lm, &[order.to_vec()], device)?;
    let (total, k) = scored.first().copied().ok_or_else(|| anyhow::anyhow!("no score"))?;
    Ok(total / k.max(1) as f64)
}

/// All ordered-triple log-probabilities from `n(n−1)` forward passes.
#[derive(Debug, Clone)]
pub struct TripleTable {
    n: usize,
    first: Vec<f32>,
    second: Vec<f32>,
    third: Vec<f32>,
}

impl TripleTable {
    /// Build the table with one pass per ordered prefix `(a, b)`, `a ≠ b`.
    pub fn build<B: Backend>(lm: &CoherenceLm<B>, device: &B::Device) -> Result<Self> {
        let n = lm.config.n_atoms;
        ensure!(n >= 3, "a triple table needs at least 3 atoms, have {n}");
        ensure!(lm.config.max_len >= 3, "max_len {} cannot hold BOS plus two atoms", lm.config.max_len);
        let v = lm.config.n_outputs();
        let mut table = Self { n, first: vec![f32::NAN; n], second: vec![f32::NAN; n * n], third: vec![f32::NAN; n * n * n] };
        let prefixes: Vec<(usize, usize)> = (0..n).flat_map(|a| (0..n).filter(move |&b| b != a).map(move |b| (a, b))).collect();
        let chunk_rows = 1024;
        for chunk in prefixes.chunks(chunk_rows) {
            let rows = chunk.len();
            let ids: Vec<i64> = chunk.iter().flat_map(|&(a, b)| [n as i64, a as i64, b as i64]).collect();
            let tokens = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([rows, 3]);
            let logp = host_f32(log_softmax(lm.model.logits(&lm.config, tokens)?, 2))?;
            for (r, &(a, b)) in chunk.iter().enumerate() {
                let at = |t: usize, x: usize| logp.get((r * 3 + t) * v + x).copied();
                if let Some(slot) = table.first.get_mut(a) {
                    *slot = at(0, a).ok_or_else(|| anyhow::anyhow!("table read out of range"))?;
                }
                if let Some(slot) = table.second.get_mut(a * n + b) {
                    *slot = at(1, b).ok_or_else(|| anyhow::anyhow!("table read out of range"))?;
                }
                let dist = logp.get((r * 3 + 2) * v..(r * 3 + 2) * v + n).ok_or_else(|| anyhow::anyhow!("table read out of range"))?;
                let dst = table.third.get_mut((a * n + b) * n..(a * n + b + 1) * n).ok_or_else(|| anyhow::anyhow!("table write out of range"))?;
                dst.copy_from_slice(dist);
            }
        }
        Ok(table)
    }

    pub fn n_atoms(&self) -> usize {
        self.n
    }

    fn ordered(&self, a: usize, b: usize, c: usize) -> f64 {
        let n = self.n;
        let f = self.first.get(a).copied().unwrap_or(f32::NAN);
        let s = self.second.get(a * n + b).copied().unwrap_or(f32::NAN);
        let t = self.third.get((a * n + b) * n + c).copied().unwrap_or(f32::NAN);
        f64::from(f) + f64::from(s) + f64::from(t)
    }

    /// `C({a, b, c})`: the mean over all six orderings of the per-atom
    /// log-likelihood. NaN when an atom is out of range or repeated.
    pub fn coherence(&self, set: [u16; 3]) -> f64 {
        let [a, b, c] = set.map(usize::from);
        if a == b || b == c || a == c || a.max(b).max(c) >= self.n {
            return f64::NAN;
        }
        let total = self.ordered(a, b, c)
            + self.ordered(a, c, b)
            + self.ordered(b, a, c)
            + self.ordered(b, c, a)
            + self.ordered(c, a, b)
            + self.ordered(c, b, a);
        total / 18.0
    }
}

/// Draw `count` atom sequences of exactly `k` distinct atoms from the model
/// at `temperature` (EOS and already-drawn atoms masked; the paper's k = 4
/// candidate generator). Returned sorted.
pub fn sample_sets<B: Backend>(lm: &CoherenceLm<B>, k: usize, count: usize, temperature: f64, seed: u64, device: &B::Device) -> Result<Vec<Vec<u16>>> {
    let n = lm.config.n_atoms;
    ensure!(k >= 1 && k < n, "cannot draw {k} distinct atoms from {n}");
    ensure!(k < lm.config.max_len, "k = {k} needs {} positions, max_len is {}", k + 1, lm.config.max_len);
    ensure!(temperature.is_finite() && temperature > 0.0, "temperature must be positive, got {temperature}");
    let v = lm.config.n_outputs();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count);
    let chunk_rows = 2048;
    let mut remaining = count;
    while remaining > 0 {
        let rows = remaining.min(chunk_rows);
        remaining -= rows;
        let mut drawn: Vec<Vec<u16>> = vec![Vec::with_capacity(k); rows];
        for step in 0..k {
            let len = step + 1;
            let ids: Vec<i64> = drawn.iter().flat_map(|d| std::iter::once(n as i64).chain(d.iter().map(|&a| i64::from(a)))).collect();
            let tokens = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([rows, len]);
            let logits = lm.model.logits(&lm.config, tokens)?.slice([0..rows, step..step + 1, 0..v]).reshape([rows, v]);
            let host = host_f32(logits)?;
            for (r, row) in drawn.iter_mut().enumerate() {
                let logits = host.get(r * v..r * v + n).ok_or_else(|| anyhow::anyhow!("sample row {r} out of range"))?;
                let scaled: Vec<f64> = logits
                    .iter()
                    .enumerate()
                    .map(|(a, &l)| if row.iter().any(|&d| usize::from(d) == a) { f64::NEG_INFINITY } else { f64::from(l) / temperature })
                    .collect();
                let top = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scaled.iter().map(|&s| (s - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                ensure!(total.is_finite() && total > 0.0, "sampling distribution degenerate at step {step}");
                let mut u = rng.random::<f64>() * total;
                let mut pick = weights.len().saturating_sub(1);
                for (a, &w) in weights.iter().enumerate() {
                    if u < w {
                        pick = a;
                        break;
                    }
                    u -= w;
                }
                row.push(u16::try_from(pick).context("atom id exceeds u16")?);
            }
        }
        for mut d in drawn {
            d.sort_unstable();
            out.push(d);
        }
    }
    Ok(out)
}

// ------------------------------------------------------------ availability --

/// The non-parametric availability score (module docs: an approximation of
/// the paper's learned dual encoder).
#[derive(Debug, Clone)]
pub struct AvailabilityIndex {
    n_atoms: usize,
    m: usize,
    weight_sq: Vec<f64>,
    postings: Vec<Vec<u32>>,
    inv_norm: Vec<f64>,
    repertoires: Vec<Vec<u16>>,
}

/// Per-thread scratch for [`AvailabilityIndex::availability`].
#[derive(Debug, Clone, Default)]
pub struct AvailabilityScratch {
    acc: Vec<f64>,
    touched: Vec<u32>,
    values: Vec<f64>,
}

/// The median of the `m` largest values, padding with zeros when fewer than
/// `m` are given (an author with no overlap scores 0). `values` is reordered.
pub fn median_of_top_m(values: &mut [f64], m: usize) -> f64 {
    if m == 0 {
        return f64::NAN;
    }
    let take = values.len().min(m);
    if values.len() > m {
        values.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
    }
    let mut top: Vec<f64> = values.iter().take(take).copied().collect();
    top.resize(m, 0.0);
    top.sort_unstable_by(|a, b| b.total_cmp(a));
    let mid = m / 2;
    if m % 2 == 1 {
        top.get(mid).copied().unwrap_or(f64::NAN)
    } else {
        match (top.get(mid - 1), top.get(mid)) {
            (Some(x), Some(y)) => 0.5 * (x + y),
            _ => f64::NAN,
        }
    }
}

impl AvailabilityIndex {
    /// Index the repertoires (each sorted, distinct, non-empty, in range).
    pub fn build(n_atoms: usize, repertoires: Vec<Vec<u16>>, m: usize) -> Result<Self> {
        ensure!(m >= 1, "top-m needs m ≥ 1");
        ensure!(repertoires.len() >= m, "{} repertoires cannot fill a top-{m}", repertoires.len());
        ensure!(u32::try_from(repertoires.len()).is_ok(), "too many repertoires");
        let mut df = vec![0usize; n_atoms];
        for (i, r) in repertoires.iter().enumerate() {
            ensure!(!r.is_empty(), "repertoire {i} is empty");
            ensure!(strictly_increasing(r), "repertoire {i} is not sorted and distinct");
            for &a in r {
                let slot = df.get_mut(usize::from(a)).ok_or_else(|| anyhow::anyhow!("repertoire {i} names atom {a} of {n_atoms}"))?;
                *slot += 1;
            }
        }
        let total = repertoires.len() as f64;
        let weight_sq: Vec<f64> = df.iter().map(|&d| ((1.0 + total) / (1.0 + d as f64)).ln() + 1.0).map(|w| w * w).collect();
        let mut postings = vec![Vec::new(); n_atoms];
        let mut inv_norm = Vec::with_capacity(repertoires.len());
        for (i, r) in repertoires.iter().enumerate() {
            let mut norm_sq = 0.0;
            for &a in r {
                let idx = usize::from(a);
                norm_sq += weight_sq.get(idx).copied().unwrap_or(0.0);
                if let Some(list) = postings.get_mut(idx) {
                    list.push(i as u32);
                }
            }
            inv_norm.push(1.0 / norm_sq.sqrt());
        }
        Ok(Self { n_atoms, m, weight_sq, postings, inv_norm, repertoires })
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn n_authors(&self) -> usize {
        self.repertoires.len()
    }

    fn set_norm(&self, set: &[u16]) -> Result<f64> {
        let mut sq = 0.0;
        for &a in set {
            sq += self.weight_sq.get(usize::from(a)).copied().ok_or_else(|| anyhow::anyhow!("atom {a} outside {}", self.n_atoms))?;
        }
        ensure!(sq > 0.0, "an empty set has no availability");
        Ok(sq.sqrt())
    }

    /// `A(S)` through the inverted index (only overlapping authors visited).
    pub fn availability(&self, set: &[u16], scratch: &mut AvailabilityScratch) -> Result<f64> {
        let norm = self.set_norm(set)?;
        if scratch.acc.len() != self.repertoires.len() {
            scratch.acc = vec![0.0; self.repertoires.len()];
        }
        scratch.touched.clear();
        for &a in set {
            let w2 = self.weight_sq.get(usize::from(a)).copied().unwrap_or(0.0);
            for &i in self.postings.get(usize::from(a)).into_iter().flatten() {
                if let Some(slot) = scratch.acc.get_mut(i as usize) {
                    if *slot == 0.0 {
                        scratch.touched.push(i);
                    }
                    *slot += w2;
                }
            }
        }
        scratch.values.clear();
        for &i in &scratch.touched {
            let idx = i as usize;
            if let (Some(slot), Some(inv)) = (scratch.acc.get_mut(idx), self.inv_norm.get(idx)) {
                scratch.values.push(*slot * inv / norm);
                *slot = 0.0;
            }
        }
        Ok(median_of_top_m(&mut scratch.values, self.m))
    }

    /// Every author's similarity, computed densely from the repertoires
    /// (the reference the inverted index is certified against).
    pub fn similarities_dense(&self, set: &[u16]) -> Result<Vec<f64>> {
        let norm = self.set_norm(set)?;
        Ok(self
            .repertoires
            .iter()
            .map(|r| {
                let dot: f64 = set.iter().filter(|a| r.binary_search(a).is_ok()).map(|&a| self.weight_sq.get(usize::from(a)).copied().unwrap_or(0.0)).sum();
                let r_sq: f64 = r.iter().map(|&a| self.weight_sq.get(usize::from(a)).copied().unwrap_or(0.0)).sum();
                dot / (norm * r_sq.sqrt())
            })
            .collect())
    }

    /// Authors whose repertoire contains every atom of `set`.
    pub fn support(&self, set: &[u16]) -> usize {
        let mut lists: Vec<&Vec<u32>> = set.iter().filter_map(|&a| self.postings.get(usize::from(a))).collect();
        if lists.len() != set.len() || lists.is_empty() {
            return 0;
        }
        lists.sort_by_key(|l| l.len());
        let Some((shortest, rest)) = lists.split_first() else {
            return 0;
        };
        shortest.iter().filter(|i| rest.iter().all(|l| l.binary_search(i).is_ok())).count()
    }

    /// `A` for many sets on all cores.
    pub fn availability_many<S: AsRef<[u16]> + Sync>(&self, sets: &[S]) -> Result<Vec<f64>> {
        let threads = std::thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
        let chunk = sets.len().div_ceil(threads).max(1);
        let results: Vec<Result<Vec<f64>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = sets
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        let mut scratch = AvailabilityScratch::default();
                        part.iter().map(|s| self.availability(s.as_ref(), &mut scratch)).collect::<Result<Vec<f64>>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().map_err(|_| anyhow::anyhow!("an availability worker panicked"))?)
                .collect()
        });
        let mut out = Vec::with_capacity(sets.len());
        for part in results {
            out.extend(part?);
        }
        Ok(out)
    }
}

// ------------------------------------------------------------------ fusion --

/// Population z-scores. Refuses a pool that is empty, non-finite or constant.
pub fn zscores(values: &[f64]) -> Result<Vec<f64>> {
    ensure!(!values.is_empty(), "z-scores of an empty pool");
    ensure!(values.iter().all(|v| v.is_finite()), "the pool contains a non-finite score");
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    ensure!(var > 0.0, "every score in the pool is {mean}; z-scores are undefined");
    let sd = var.sqrt();
    Ok(values.iter().map(|v| (v - mean) / sd).collect())
}

/// `F = (1−β)·z(C) + β·z(U)` with `U = −A`, both z-scored within the pool.
pub fn fuse(coherence: &[f64], availability: &[f64], beta: f64) -> Result<Vec<f64>> {
    ensure!(coherence.len() == availability.len(), "{} coherence vs {} availability scores", coherence.len(), availability.len());
    ensure!(beta.is_finite() && (0.0..=1.0).contains(&beta), "beta must be in [0, 1], got {beta}");
    let zc = zscores(coherence)?;
    let unavailability: Vec<f64> = availability.iter().map(|a| -a).collect();
    let zu = zscores(&unavailability)?;
    Ok(zc.iter().zip(&zu).map(|(c, u)| (1.0 - beta) * c + beta * u).collect())
}

/// Indices of the `top` largest values, best first (ties by index).
pub fn top_indices(values: &[f64], top: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    let cmp = |a: &usize, b: &usize| {
        let va = values.get(*a).copied().unwrap_or(f64::NEG_INFINITY);
        let vb = values.get(*b).copied().unwrap_or(f64::NEG_INFINITY);
        vb.total_cmp(&va).then(a.cmp(b))
    };
    if top < idx.len() && top > 0 {
        idx.select_nth_unstable_by(top - 1, cmp);
        idx.truncate(top);
    } else if top == 0 {
        idx.clear();
    }
    idx.sort_unstable_by(cmp);
    idx
}

/// `C(n, k)` with overflow checking.
pub fn binomial(n: usize, k: usize) -> Result<u64> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc * (n - i) as u128 / (i + 1) as u128;
    }
    u64::try_from(acc).context("binomial overflows u64")
}

/// Every 3-subset `a < b < c` of `0..n`, keeping those with at least
/// `min_targets` atoms in `targets` (all of them when `targets` is `None`).
pub fn enumerate_triples(n: usize, targets: Option<&[bool]>, min_targets: usize) -> Result<Vec<[u16; 3]>> {
    ensure!(n <= MAX_ATOMS, "{n} atoms exceed {MAX_ATOMS}");
    if let Some(t) = targets {
        ensure!(t.len() == n, "target mask has {} entries for {n} atoms", t.len());
    }
    let is_t = |a: usize| usize::from(targets.is_none_or(|t| t.get(a).copied().unwrap_or(false)));
    let mut out = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                let hits = if targets.is_some() { is_t(a) + is_t(b) + is_t(c) } else { 3 };
                if hits >= min_targets {
                    out.push([u16::try_from(a)?, u16::try_from(b)?, u16::try_from(c)?]);
                }
            }
        }
    }
    Ok(out)
}

/// `P(pos > neg) + ½ P(pos = neg)` (the Mann–Whitney AUC), by sorting.
pub fn auc(pos: &[f64], neg: &[f64]) -> Result<f64> {
    ensure!(!pos.is_empty() && !neg.is_empty(), "AUC needs both classes ({} positives, {} negatives)", pos.len(), neg.len());
    ensure!(pos.iter().chain(neg).all(|v| v.is_finite()), "AUC over non-finite scores");
    let mut all: Vec<(f64, bool)> = pos.iter().map(|&v| (v, true)).chain(neg.iter().map(|&v| (v, false))).collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all.get(j + 1).map(|x| x.0) == all.get(i).map(|x| x.0) {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        let positives = all.get(i..=j).map(|s| s.iter().filter(|x| x.1).count()).unwrap_or(0);
        rank_sum_pos += avg_rank * positives as f64;
        i = j + 1;
    }
    let np = pos.len() as f64;
    let nn = neg.len() as f64;
    Ok((rank_sum_pos - np * (np + 1.0) / 2.0) / (np * nn))
}

// ---------------------------------------------------------------- training --

/// How the coherence model is trained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub warmup_steps: usize,
    /// Cosine decay ends at `lr · final_lr_fraction`.
    pub final_lr_fraction: f64,
    pub weight_decay: f64,
    /// Fresh random serialisations of each training paper per epoch.
    pub serialisations: usize,
    /// Fixed serialisations per evaluation paper.
    pub eval_serialisations: usize,
    /// Epochs without a validation improvement before stopping.
    pub patience: usize,
    pub seed: u64,
    pub min_atoms: usize,
    pub val_permille: u32,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            epochs: 60,
            batch_size: 128,
            lr: 1e-3,
            warmup_steps: 200,
            final_lr_fraction: 0.1,
            weight_decay: 0.01,
            serialisations: 2,
            eval_serialisations: 4,
            patience: 8,
            seed: 42,
            min_atoms: 2,
            val_permille: 50,
        }
    }
}

impl TrainOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.epochs >= 1, "epochs must be positive");
        ensure!(self.batch_size >= 1, "batch_size must be positive");
        ensure!(self.lr.is_finite() && self.lr > 0.0, "lr must be positive, got {}", self.lr);
        ensure!(self.final_lr_fraction.is_finite() && (0.0..=1.0).contains(&self.final_lr_fraction), "final_lr_fraction must be in [0, 1]");
        ensure!(self.weight_decay.is_finite() && self.weight_decay >= 0.0, "weight_decay must be non-negative");
        ensure!(self.serialisations >= 1, "serialisations must be positive");
        ensure!(self.eval_serialisations >= 1, "eval_serialisations must be positive");
        ensure!(self.min_atoms >= 1, "min_atoms must be positive");
        Ok(())
    }

    fn lr_at(&self, step: usize, total: usize) -> f64 {
        if step < self.warmup_steps {
            return self.lr * (step + 1) as f64 / self.warmup_steps as f64;
        }
        let span = total.saturating_sub(self.warmup_steps).max(1) as f64;
        let progress = ((step - self.warmup_steps) as f64 / span).min(1.0);
        let floor = self.lr * self.final_lr_fraction;
        floor + (self.lr - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

/// Per-atom negative log-likelihoods (nats) of one evaluation set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NllReport {
    pub papers: usize,
    pub sequences: usize,
    pub atoms: usize,
    /// The model, atoms only (what `C` scores).
    pub model_per_atom: f64,
    /// The model including the EOS decision, per target.
    pub model_per_target: f64,
    /// Add-one unigram over training atom occurrences.
    pub unigram_per_atom: f64,
    /// The same unigram renormalised over atoms not yet drawn (sets never
    /// repeat an atom): the stronger baseline.
    pub unigram_without_replacement_per_atom: f64,
}

/// One epoch of the training curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochLog {
    pub epoch: usize,
    pub steps: usize,
    pub train_loss: f64,
    pub val_per_atom: f64,
    pub lr_end: f64,
    pub seconds: f64,
}

/// What a training run produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainReport {
    pub params: usize,
    pub split_sizes: [usize; 3],
    pub epochs: Vec<EpochLog>,
    pub best_epoch: usize,
    pub val: NllReport,
    pub test: Option<NllReport>,
    pub seconds: f64,
}

fn eval_sequences(catalog: &AtomCatalog, papers: &[usize], per_paper: usize, seed: u64) -> Vec<Vec<u16>> {
    let mut out = Vec::with_capacity(papers.len() * per_paper);
    for &i in papers {
        if let Some(p) = catalog.papers().get(i) {
            let mut rng = ChaCha8Rng::seed_from_u64(set_seed(&p.atoms, seed ^ i as u64));
            for _ in 0..per_paper {
                let mut s = p.atoms.clone();
                s.shuffle(&mut rng);
                out.push(s);
            }
        }
    }
    out
}

fn unigram(catalog: &AtomCatalog, train: &[usize]) -> Vec<f64> {
    let n = catalog.n_atoms();
    let mut counts = vec![1.0f64; n];
    for &i in train {
        for &a in catalog.papers().get(i).map(|p| p.atoms.as_slice()).unwrap_or(&[]) {
            if let Some(c) = counts.get_mut(usize::from(a)) {
                *c += 1.0;
            }
        }
    }
    let total: f64 = counts.iter().sum();
    counts.iter().map(|c| c / total).collect()
}

fn nll_report<B: Backend>(lm: &CoherenceLm<B>, catalog: &AtomCatalog, papers: &[usize], probs: &[f64], options: &TrainOptions, device: &B::Device) -> Result<NllReport> {
    let seqs = eval_sequences(catalog, papers, options.eval_serialisations, options.seed);
    ensure!(!seqs.is_empty(), "no evaluation sequences");
    let mut atom_sum = 0.0;
    let mut atoms = 0usize;
    let mut target_sum = 0.0;
    let mut targets = 0usize;
    for chunk in seqs.chunks(SCORE_CHUNK) {
        let batch = Batch::new(chunk, &lm.config)?;
        let picked = host_f32(batch.picked(&lm.model, &lm.config, device)?)?;
        for ((&lp, &m), &am) in picked.iter().zip(&batch.mask).zip(&batch.atom_mask) {
            target_sum -= f64::from(lp) * f64::from(m);
            targets += usize::from(m > 0.0);
            atom_sum -= f64::from(lp) * f64::from(am);
            atoms += usize::from(am > 0.0);
        }
    }
    let mut uni = 0.0;
    let mut uni_wr = 0.0;
    for s in &seqs {
        let mut used = 0.0;
        for &a in s {
            let p = probs.get(usize::from(a)).copied().unwrap_or(f64::MIN_POSITIVE);
            uni -= p.ln();
            uni_wr -= (p / (1.0 - used)).ln();
            used += p;
        }
    }
    Ok(NllReport {
        papers: papers.len(),
        sequences: seqs.len(),
        atoms,
        model_per_atom: atom_sum / atoms.max(1) as f64,
        model_per_target: target_sum / targets.max(1) as f64,
        unigram_per_atom: uni / atoms.max(1) as f64,
        unigram_without_replacement_per_atom: uni_wr / atoms.max(1) as f64,
    })
}

/// Train the coherence model on `split.train`, early-stopping on the
/// per-atom NLL of `split.val`; reports val (and test) NLL against the
/// unigram baselines. `progress` receives one line per epoch.
pub fn train_coherence(
    catalog: &AtomCatalog,
    split: &Split,
    config: &CoherenceConfig,
    options: &TrainOptions,
    progress: &mut dyn FnMut(&str),
) -> Result<(CoherenceLm<Cpu>, TrainReport)> {
    config.validate()?;
    options.validate()?;
    ensure!(config.n_atoms == catalog.n_atoms(), "config has {} atoms, the catalog {}", config.n_atoms, catalog.n_atoms());
    let longest = catalog.papers().iter().map(|p| p.atoms.len()).max().unwrap_or(0);
    ensure!(longest < config.max_len, "the longest paper has {longest} atoms; max_len {} must exceed it", config.max_len);
    let start = Instant::now();
    let device = Default::default();
    <CpuTrain as Backend>::seed(&device, options.seed);
    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    let mut model: CoherenceModel<CpuTrain> = config.init(&device)?;
    let params = model.num_params();
    let mut optim = AdamWConfig::new()
        .with_weight_decay(options.weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<CpuTrain, CoherenceModel<CpuTrain>>();
    let train_sets: Vec<&Vec<u16>> = split.train.iter().filter_map(|&i| catalog.papers().get(i)).map(|p| &p.atoms).collect();
    let steps_per_epoch = (train_sets.len() * options.serialisations).div_ceil(options.batch_size);
    let total_steps = steps_per_epoch * options.epochs;
    let probs = unigram(catalog, &split.train);
    let mut step = 0usize;
    let mut best: Option<(f64, usize, CoherenceModel<Cpu>)> = None;
    let mut epochs = Vec::new();
    progress(&format!(
        "alienset train: {params} params, {} train / {} val / {} test papers, {steps_per_epoch} steps per epoch",
        split.train.len(),
        split.val.len(),
        split.test.len()
    ));
    for epoch in 0..options.epochs {
        let epoch_start = Instant::now();
        let mut rows: Vec<Vec<u16>> = Vec::with_capacity(train_sets.len() * options.serialisations);
        for set in &train_sets {
            for _ in 0..options.serialisations {
                let mut s = (*set).clone();
                s.shuffle(&mut rng);
                rows.push(s);
            }
        }
        rows.shuffle(&mut rng);
        let mut loss_sum = 0.0;
        let mut loss_n = 0usize;
        let mut lr = options.lr;
        for chunk in rows.chunks(options.batch_size) {
            let batch = Batch::new(chunk, config)?;
            lr = options.lr_at(step, total_steps);
            let picked = batch.picked(&model, config, &device)?;
            let mask = Tensor::<CpuTrain, 1>::from_floats(batch.mask.as_slice(), &device).reshape([batch.rows, batch.len]);
            let count: f32 = batch.mask.iter().sum();
            let loss = (picked * mask).sum().neg() / count.max(1.0);
            let value = f64::from(loss.clone().into_scalar());
            ensure!(value.is_finite(), "training loss became {value} at step {step}");
            loss_sum += value;
            loss_n += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
            step += 1;
        }
        let inner = CoherenceLm { config: config.clone(), model: model.valid() };
        let val = nll_report(&inner, catalog, &split.val, &probs, options, &device)?;
        let log = EpochLog {
            epoch: epoch + 1,
            steps: step,
            train_loss: loss_sum / loss_n.max(1) as f64,
            val_per_atom: val.model_per_atom,
            lr_end: lr,
            seconds: epoch_start.elapsed().as_secs_f64(),
        };
        progress(&format!(
            "epoch {:>3}  train/target {:.4}  val/atom {:.4}  (unigram {:.4}, w/o repl. {:.4})  lr {:.2e}  {:.1}s",
            log.epoch, log.train_loss, val.model_per_atom, val.unigram_per_atom, val.unigram_without_replacement_per_atom, lr, log.seconds
        ));
        epochs.push(log);
        let improved = best.as_ref().is_none_or(|(b, _, _)| val.model_per_atom < *b);
        if improved {
            best = Some((val.model_per_atom, epoch + 1, inner.model));
        } else if best.as_ref().is_some_and(|(_, e, _)| epoch + 1 >= e + options.patience) {
            progress(&format!("early stop: no validation improvement for {} epochs", options.patience));
            break;
        }
    }
    let (_, best_epoch, best_model) = best.ok_or_else(|| anyhow::anyhow!("training ran no epochs"))?;
    let lm = CoherenceLm { config: config.clone(), model: best_model };
    let val = nll_report(&lm, catalog, &split.val, &probs, options, &device)?;
    let test = if split.test.is_empty() { None } else { Some(nll_report(&lm, catalog, &split.test, &probs, options, &device)?) };
    let report = TrainReport {
        params,
        split_sizes: [split.train.len(), split.val.len(), split.test.len()],
        epochs,
        best_epoch,
        val,
        test,
        seconds: start.elapsed().as_secs_f64(),
    };
    Ok((lm, report))
}

// --------------------------------------------------------------- model card --

/// Everything saved next to the weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub method: String,
    pub config: CoherenceConfig,
    pub options: TrainOptions,
    pub split: SplitRule,
    pub export_dir: PathBuf,
    /// sha256 of the export files the model was trained on.
    pub export_sha256: BTreeMap<String, String>,
    pub counts: CatalogCounts,
    pub report: TrainReport,
}

const EXPORT_FILES: [&str; 3] = ["atoms.tsv", "paper_atoms.jsonl", "author_repertoires.jsonl"];

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn export_hashes(dir: &Path) -> Result<BTreeMap<String, String>> {
    EXPORT_FILES.iter().map(|f| Ok(((*f).to_string(), file_sha256(&dir.join(f))?))).collect()
}

/// Save weights (`model.mpk`) and the card (`model.json`) atomically.
pub fn save_model(dir: &Path, lm: &CoherenceLm<Cpu>, card: &ModelCard) -> Result<()> {
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder
        .record(lm.model.clone().into_record(), ())
        .map_err(|err| anyhow::anyhow!("serialising the coherence model: {err}"))?;
    write_atomic(&dir.join("model.mpk"), &bytes)?;
    write_atomic(&dir.join("model.json"), serde_json::to_string_pretty(card)?.as_bytes())
}

/// Load a model directory written by [`save_model`].
pub fn load_model(dir: &Path) -> Result<(CoherenceLm<Cpu>, ModelCard)> {
    let card_path = dir.join("model.json");
    let card: ModelCard = serde_json::from_str(&std::fs::read_to_string(&card_path).with_context(|| format!("reading {}", card_path.display()))?)
        .with_context(|| format!("parsing {}", card_path.display()))?;
    let device = Default::default();
    let weights_path = dir.join("model.mpk");
    let bytes = std::fs::read(&weights_path).with_context(|| format!("reading {}", weights_path.display()))?;
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let record = recorder.load(bytes, &device).map_err(|err| anyhow::anyhow!("loading {}: {err}", weights_path.display()))?;
    let model = card.config.init::<Cpu>(&device)?.load_record(record);
    Ok((CoherenceLm { config: card.config.clone(), model }, card))
}

// ----------------------------------------------------------------- ranking --

/// One ranked candidate set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedSet {
    pub rank: usize,
    pub atoms: Vec<u16>,
    /// How many of `atoms` are target atoms (all when unrestricted).
    pub targets: usize,
    pub coherence: f64,
    pub availability: f64,
    pub z_coherence: f64,
    pub z_unavailability: f64,
    pub fused: f64,
    /// Authors whose repertoire contains the whole set.
    pub support: usize,
    /// A training paper's atom set contains the whole set.
    pub inside_a_paper: bool,
    /// First 160 characters of each atom's canonical text.
    pub names: Vec<String>,
}

/// Summary of one ranking run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankSummary {
    pub k: usize,
    pub beta: f64,
    pub pool: usize,
    pub top: usize,
    pub restricted_to_targets: bool,
    pub min_targets: usize,
    pub coherence_seconds: f64,
    pub availability_seconds: f64,
    pub fusion_seconds: f64,
    pub mean_coherence_pool: f64,
    pub mean_availability_pool: f64,
    pub mean_coherence_top: f64,
    pub mean_availability_top: f64,
    pub inside_a_paper_top: usize,
    pub zero_support_top: usize,
}

/// All 3-subsets (or fewer) of every paper's atom set, for "is this set
/// already inside a paper" checks.
pub fn paper_subsets(catalog: &AtomCatalog, papers: &[usize], k: usize) -> HashSet<Vec<u16>> {
    let mut out = HashSet::new();
    for &i in papers {
        let Some(p) = catalog.papers().get(i) else { continue };
        if p.atoms.len() < k {
            continue;
        }
        let mut idx: Vec<usize> = (0..k).collect();
        loop {
            out.insert(idx.iter().filter_map(|&j| p.atoms.get(j).copied()).collect());
            let mut pos = k;
            let mut advanced = false;
            while pos > 0 {
                pos -= 1;
                let limit = p.atoms.len() - k + pos;
                if idx.get(pos).is_some_and(|&v| v < limit) {
                    if let Some(v) = idx.get_mut(pos) {
                        *v += 1;
                    }
                    let base = idx.get(pos).copied().unwrap_or(0);
                    for (offset, slot) in idx.iter_mut().skip(pos + 1).enumerate() {
                        *slot = base + offset + 1;
                    }
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                break;
            }
        }
    }
    out
}

/// Score a pool of sets (coherence already computed) and keep the top.
fn finish_ranking(
    catalog: &AtomCatalog,
    index: &AvailabilityIndex,
    inside: &HashSet<Vec<u16>>,
    target_mask: Option<&[bool]>,
    pool: Vec<Vec<u16>>,
    scored: (Vec<f64>, f64),
    options: &SetsRankArgs,
) -> Result<(Vec<RankedSet>, RankSummary)> {
    let (coherence, coherence_seconds) = scored;
    let k = pool.first().map(Vec::len).unwrap_or(0);
    let t0 = Instant::now();
    let availability = index.availability_many(&pool)?;
    let availability_seconds = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let zc = zscores(&coherence)?;
    let zu = zscores(&availability.iter().map(|a| -a).collect::<Vec<_>>())?;
    let fused = fuse(&coherence, &availability, options.beta)?;
    let best = top_indices(&fused, options.top);
    let fusion_seconds = t1.elapsed().as_secs_f64();
    let mut rows = Vec::with_capacity(best.len());
    for (rank, &i) in best.iter().enumerate() {
        let atoms = pool.get(i).cloned().ok_or_else(|| anyhow::anyhow!("pool index {i} out of range"))?;
        let targets = match target_mask {
            None => atoms.len(),
            Some(mask) => atoms.iter().filter(|&&a| mask.get(usize::from(a)).copied().unwrap_or(false)).count(),
        };
        let names = atoms
            .iter()
            .map(|&a| catalog.atoms().get(usize::from(a)).map(|x| x.name.chars().take(160).collect()).unwrap_or_default())
            .collect();
        rows.push(RankedSet {
            rank: rank + 1,
            support: index.support(&atoms),
            inside_a_paper: inside.contains(&atoms),
            targets,
            coherence: coherence.get(i).copied().unwrap_or(f64::NAN),
            availability: availability.get(i).copied().unwrap_or(f64::NAN),
            z_coherence: zc.get(i).copied().unwrap_or(f64::NAN),
            z_unavailability: zu.get(i).copied().unwrap_or(f64::NAN),
            fused: fused.get(i).copied().unwrap_or(f64::NAN),
            atoms,
            names,
        });
    }
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len().max(1) as f64;
    let top_c: Vec<f64> = rows.iter().map(|r| r.coherence).collect();
    let top_a: Vec<f64> = rows.iter().map(|r| r.availability).collect();
    let summary = RankSummary {
        k,
        beta: options.beta,
        pool: pool.len(),
        top: rows.len(),
        restricted_to_targets: target_mask.is_some(),
        min_targets: options.min_targets,
        coherence_seconds,
        availability_seconds,
        fusion_seconds,
        mean_coherence_pool: mean(&coherence),
        mean_availability_pool: mean(&availability),
        mean_coherence_top: mean(&top_c),
        mean_availability_top: mean(&top_a),
        inside_a_paper_top: rows.iter().filter(|r| r.inside_a_paper).count(),
        zero_support_top: rows.iter().filter(|r| r.support == 0).count(),
    };
    Ok((rows, summary))
}

// -------------------------------------------------------------- validation --

/// Separation of one positive and one negative class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Separation {
    pub positives: usize,
    pub negatives: usize,
    pub auc: f64,
    pub mean_positive: f64,
    pub mean_negative: f64,
}

fn separation(pos: &[f64], neg: &[f64]) -> Result<Separation> {
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len().max(1) as f64;
    Ok(Separation { positives: pos.len(), negatives: neg.len(), auc: auc(pos, neg)?, mean_positive: mean(pos), mean_negative: mean(neg) })
}

/// The paper's three-way coherence diagnostic for one set size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoherenceDiagnostic {
    pub k: usize,
    pub real_vs_random: Separation,
    pub random_vs_disjoint: Separation,
    pub real_vs_disjoint: Separation,
    pub disjoint_attempts: usize,
}

/// Recall of held-out future paper sets in the top-K of the full k = 3 pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallRow {
    pub beta: f64,
    pub top_k: usize,
    pub hits: usize,
    pub hits_novel: usize,
    pub expected_random: f64,
    pub expected_random_novel: f64,
    pub lift: f64,
}

/// What the top of the full pool looks like at one β.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BetaBehaviour {
    pub beta: f64,
    pub top: usize,
    pub mean_coherence: f64,
    pub mean_availability: f64,
    pub mean_support: f64,
    pub inside_a_training_paper: usize,
    pub zero_support: usize,
    pub examples: Vec<Vec<u16>>,
}

/// Everything `alien sets validate` measures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub split: SplitRule,
    pub held_out_papers: usize,
    pub coherence: Vec<CoherenceDiagnostic>,
    /// High-support (≥ 3 authors hold the whole set) vs zero-support sets.
    pub availability_high_vs_zero_support: Separation,
    /// Held-out real paper sets vs uniformly random sets.
    pub availability_real_vs_random: Separation,
    pub held_out_triples: usize,
    pub held_out_triples_novel: usize,
    pub recall: Vec<RecallRow>,
    pub beta_behaviour: Vec<BetaBehaviour>,
    pub seconds_table: f64,
    pub seconds_availability_full_pool: f64,
}

fn random_set<R: Rng>(rng: &mut R, n: usize, k: usize) -> Result<Vec<u16>> {
    let mut idx = rand::seq::index::sample(rng, n, k).into_vec();
    idx.sort_unstable();
    idx.into_iter().map(|i| u16::try_from(i).context("atom id exceeds u16")).collect()
}

/// Run the diagnostics. The held-out papers are the split's test papers
/// (validation papers for a split without a test set).
pub fn validate(lm: &CoherenceLm<Cpu>, catalog: &AtomCatalog, rule: SplitRule, options: &TrainOptions, samples: usize, progress: &mut dyn FnMut(&str)) -> Result<ValidationReport> {
    let device = Default::default();
    let split = catalog.split(rule, options.val_permille, options.min_atoms)?;
    let train_papers: Vec<usize> = split.train.iter().chain(&split.val).copied().collect();
    let held_out = if split.test.is_empty() { split.val.clone() } else { split.test.clone() };
    let n = catalog.n_atoms();
    let policy = PermutationPolicy::default();
    let mut rng = ChaCha8Rng::seed_from_u64(options.seed ^ 0xD1A6);
    let mut co = vec![false; n * n];
    for &i in &train_papers {
        let Some(p) = catalog.papers().get(i) else { continue };
        for &a in &p.atoms {
            for &b in &p.atoms {
                if let Some(slot) = co.get_mut(usize::from(a) * n + usize::from(b)) {
                    *slot = true;
                }
            }
        }
    }
    let mut diagnostics = Vec::new();
    for k in [3usize, 4] {
        let real: Vec<Vec<u16>> = held_out.iter().filter_map(|&i| catalog.papers().get(i)).filter(|p| p.atoms.len() == k).map(|p| p.atoms.clone()).collect();
        let random: Vec<Vec<u16>> = (0..samples).map(|_| random_set(&mut rng, n, k)).collect::<Result<_>>()?;
        let mut disjoint = Vec::new();
        let mut attempts = 0usize;
        while disjoint.len() < samples && attempts < 200 * samples.max(1) {
            attempts += 1;
            let s = random_set(&mut rng, n, k)?;
            let clean = s.iter().enumerate().all(|(x, &a)| s.iter().skip(x + 1).all(|&b| !co.get(usize::from(a) * n + usize::from(b)).copied().unwrap_or(true)));
            if clean {
                disjoint.push(s);
            }
        }
        progress(&format!("coherence diagnostic k={k}: {} real, {} random, {} disjoint ({attempts} draws)", real.len(), random.len(), disjoint.len()));
        let c_real = coherence_scores(lm, &real, &policy, &device)?;
        let c_random = coherence_scores(lm, &random, &policy, &device)?;
        let c_disjoint = coherence_scores(lm, &disjoint, &policy, &device)?;
        diagnostics.push(CoherenceDiagnostic {
            k,
            real_vs_random: separation(&c_real, &c_random)?,
            random_vs_disjoint: separation(&c_random, &c_disjoint)?,
            real_vs_disjoint: separation(&c_real, &c_disjoint)?,
            disjoint_attempts: attempts,
        });
    }
    // Availability from the training-period repertoires only.
    let year_limit = AtomCatalog::train_year_limit(rule);
    let index = AvailabilityIndex::build(n, catalog.repertoires(year_limit), DEFAULT_TOP_M)?;
    let mut high = Vec::new();
    let mut zero = Vec::new();
    let reps = catalog.repertoires(year_limit);
    let mut guard = 0usize;
    while (high.len() < samples || zero.len() < samples) && guard < 400 * samples.max(1) {
        guard += 1;
        if high.len() < samples {
            if let Some(r) = reps.get(rng.random_range(0..reps.len().max(1))) {
                if r.len() >= 3 {
                    let pick = rand::seq::index::sample(&mut rng, r.len(), 3).into_vec();
                    let mut s: Vec<u16> = pick.iter().filter_map(|&j| r.get(j).copied()).collect();
                    s.sort_unstable();
                    if index.support(&s) >= 3 {
                        high.push(s);
                    }
                }
            }
        }
        if zero.len() < samples {
            let s = random_set(&mut rng, n, 3)?;
            if index.support(&s) == 0 {
                zero.push(s);
            }
        }
    }
    let a_high = index.availability_many(&high)?;
    let a_zero = index.availability_many(&zero)?;
    let real3: Vec<Vec<u16>> = held_out.iter().filter_map(|&i| catalog.papers().get(i)).filter(|p| p.atoms.len() == 3).map(|p| p.atoms.clone()).collect();
    let random3: Vec<Vec<u16>> = (0..samples).map(|_| random_set(&mut rng, n, 3)).collect::<Result<_>>()?;
    let a_real = index.availability_many(&real3)?;
    let a_random = index.availability_many(&random3)?;
    let availability_high_vs_zero_support = separation(&a_high, &a_zero)?;
    let availability_real_vs_random = separation(&a_real, &a_random)?;
    progress(&format!(
        "availability: high-vs-zero AUC {:.3}, real-vs-random AUC {:.3}",
        availability_high_vs_zero_support.auc, availability_real_vs_random.auc
    ));

    // The full k = 3 pool: temporal recall and the β behaviour.
    let t0 = Instant::now();
    let table = TripleTable::build(lm, &device)?;
    let seconds_table = t0.elapsed().as_secs_f64();
    let pool = enumerate_triples(n, None, 3)?;
    let coherence: Vec<f64> = pool.iter().map(|&s| table.coherence(s)).collect();
    let t1 = Instant::now();
    let availability = index.availability_many(&pool)?;
    let seconds_availability_full_pool = t1.elapsed().as_secs_f64();
    progress(&format!("full pool: {} triples, table {seconds_table:.1}s, availability {seconds_availability_full_pool:.1}s", pool.len()));
    let inside_train = paper_subsets(catalog, &train_papers, 3);
    let held: HashSet<[u16; 3]> = real3.iter().filter_map(|s| <[u16; 3]>::try_from(s.as_slice()).ok()).collect();
    let novel: HashSet<[u16; 3]> = held.iter().filter(|s| !inside_train.contains(&s.to_vec())).copied().collect();
    let total = pool.len() as f64;
    let mut recall = Vec::new();
    let mut beta_behaviour = Vec::new();
    for beta in [0.0, 0.1, 0.5, 0.7, 1.0] {
        let fused = fuse(&coherence, &availability, beta)?;
        for top_k in [100usize, 1000, 10000] {
            let best = top_indices(&fused, top_k);
            let hits = best.iter().filter(|&&i| pool.get(i).is_some_and(|s| held.contains(s))).count();
            let hits_novel = best.iter().filter(|&&i| pool.get(i).is_some_and(|s| novel.contains(s))).count();
            let expected_random = top_k as f64 * held.len() as f64 / total;
            recall.push(RecallRow {
                beta,
                top_k,
                hits,
                hits_novel,
                expected_random,
                expected_random_novel: top_k as f64 * novel.len() as f64 / total,
                lift: hits as f64 / expected_random.max(f64::MIN_POSITIVE),
            });
        }
        let best = top_indices(&fused, 300);
        let sets: Vec<[u16; 3]> = best.iter().filter_map(|&i| pool.get(i).copied()).collect();
        let supports: Vec<usize> = sets.iter().map(|s| index.support(s)).collect();
        let mean = |f: &dyn Fn(usize) -> f64| best.iter().map(|&i| f(i)).sum::<f64>() / best.len().max(1) as f64;
        beta_behaviour.push(BetaBehaviour {
            beta,
            top: best.len(),
            mean_coherence: mean(&|i| coherence.get(i).copied().unwrap_or(f64::NAN)),
            mean_availability: mean(&|i| availability.get(i).copied().unwrap_or(f64::NAN)),
            mean_support: supports.iter().sum::<usize>() as f64 / supports.len().max(1) as f64,
            inside_a_training_paper: sets.iter().filter(|s| inside_train.contains(&s.to_vec())).count(),
            zero_support: supports.iter().filter(|&&s| s == 0).count(),
            examples: sets.iter().take(5).map(|s| s.to_vec()).collect(),
        });
    }
    Ok(ValidationReport {
        split: rule,
        held_out_papers: held_out.len(),
        coherence: diagnostics,
        availability_high_vs_zero_support,
        availability_real_vs_random,
        held_out_triples: held.len(),
        held_out_triples_novel: novel.len(),
        recall,
        beta_behaviour,
        seconds_table,
        seconds_availability_full_pool,
    })
}

// --------------------------------------------------------------------- CLI --

/// `dblocks alien sets ...`.
#[derive(clap::Subcommand, Debug, Clone)]
pub enum SetsAction {
    /// Train the atom-set coherence model on an export directory.
    Train(Box<SetsTrainArgs>),
    /// Rank k-sets by z-fused coherence and unavailability.
    Rank(Box<SetsRankArgs>),
    /// The paper's diagnostics: coherence AUCs, availability AUCs,
    /// temporal recall of held-out triples, and the β behaviour.
    Validate(Box<SetsValidateArgs>),
}

/// Flags of `alien sets train`.
#[derive(clap::Args, Debug, Clone)]
pub struct SetsTrainArgs {
    /// Export directory (atoms.tsv, paper_atoms.jsonl, author_repertoires.jsonl).
    #[arg(long)]
    pub export: PathBuf,
    /// Output model directory.
    #[arg(long)]
    pub out: PathBuf,
    /// `temporal` (train ≤ --last-train-year, test later) or `all`.
    #[arg(long, default_value = "temporal")]
    pub split: String,
    #[arg(long, default_value_t = 2024)]
    pub last_train_year: u16,
    #[arg(long, default_value_t = 3)]
    pub layers: usize,
    #[arg(long, default_value_t = 128)]
    pub d_model: usize,
    #[arg(long, default_value_t = 4)]
    pub heads: usize,
    #[arg(long, default_value_t = 512)]
    pub d_ff: usize,
    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,
    #[arg(long, default_value_t = 16)]
    pub max_len: usize,
    #[arg(long, default_value_t = 60)]
    pub epochs: usize,
    #[arg(long, default_value_t = 128)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,
    #[arg(long, default_value_t = 2)]
    pub serialisations: usize,
    #[arg(long, default_value_t = 8)]
    pub patience: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// Flags of `alien sets rank`.
#[derive(clap::Args, Debug, Clone)]
pub struct SetsRankArgs {
    /// Model directory from `alien sets train`.
    #[arg(long)]
    pub model: PathBuf,
    /// Export directory (default: the one recorded in the model card).
    #[arg(long)]
    pub export: Option<PathBuf>,
    /// TSV whose first column is a target atom id (`#` comments and a
    /// non-numeric header are skipped). Without it the pool is unrestricted.
    #[arg(long)]
    pub target_atoms: Option<PathBuf>,
    /// Minimum target atoms per set.
    #[arg(long, default_value_t = 2)]
    pub min_targets: usize,
    /// Set size: 3 is exhaustive, 4 is sampled from the coherence model.
    #[arg(long, default_value_t = 3)]
    pub k: usize,
    #[arg(long, default_value_t = DEFAULT_BETA)]
    pub beta: f64,
    #[arg(long, default_value_t = 300)]
    pub top: usize,
    /// Model samples for k = 4 (the paper drew 100,000).
    #[arg(long, default_value_t = 100_000)]
    pub samples: usize,
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
    /// Ranked JSONL; a `<out>.summary.json` is written beside it.
    #[arg(long)]
    pub out: PathBuf,
}

/// Flags of `alien sets validate`.
#[derive(clap::Args, Debug, Clone)]
pub struct SetsValidateArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub export: Option<PathBuf>,
    /// Random / disjoint / support samples per class.
    #[arg(long, default_value_t = 5000)]
    pub samples: usize,
    #[arg(long)]
    pub out: PathBuf,
}

/// Read target atom ids from a TSV (first column).
pub fn read_targets(path: &Path, n_atoms: usize) -> Result<Vec<bool>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut mask = vec![false; n_atoms];
    for (i, line) in text.lines().enumerate() {
        let first = line.split('\t').next().unwrap_or("").trim();
        if first.is_empty() || first.starts_with('#') || (i == 0 && first.parse::<usize>().is_err()) {
            continue;
        }
        let id: usize = first.parse().with_context(|| format!("{} line {}: '{first}' is not an atom id", path.display(), i + 1))?;
        let slot = mask.get_mut(id).ok_or_else(|| anyhow::anyhow!("{} line {}: atom {id} outside {n_atoms}", path.display(), i + 1))?;
        *slot = true;
    }
    ensure!(mask.iter().any(|&m| m), "{} names no target atoms", path.display());
    Ok(mask)
}

/// Run one `alien sets` action, reporting progress line by line.
pub fn run_cli(action: SetsAction, progress: &mut dyn FnMut(&str)) -> Result<()> {
    match action {
        SetsAction::Train(args) => {
            let catalog = AtomCatalog::load(&args.export)?;
            let counts = catalog.counts();
            progress(&format!("catalog: {}", serde_json::to_string(&counts)?));
            let rule = match args.split.as_str() {
                "temporal" => SplitRule::Temporal { last_train_year: args.last_train_year },
                "all" => SplitRule::All,
                other => bail!("--split must be 'temporal' or 'all', got '{other}'"),
            };
            let options = TrainOptions {
                epochs: args.epochs,
                batch_size: args.batch_size,
                lr: args.lr,
                serialisations: args.serialisations,
                patience: args.patience,
                seed: args.seed,
                ..TrainOptions::default()
            };
            let split = catalog.split(rule, options.val_permille, options.min_atoms)?;
            let config = CoherenceConfig {
                n_atoms: catalog.n_atoms(),
                d_model: args.d_model,
                n_heads: args.heads,
                n_layers: args.layers,
                d_ff: args.d_ff,
                dropout: args.dropout,
                max_len: args.max_len,
            };
            let (lm, report) = train_coherence(&catalog, &split, &config, &options, progress)?;
            progress(&format!(
                "best epoch {}: val/atom {:.4} (unigram {:.4}, w/o repl. {:.4}){}; {:.0}s",
                report.best_epoch,
                report.val.model_per_atom,
                report.val.unigram_per_atom,
                report.val.unigram_without_replacement_per_atom,
                report
                    .test
                    .as_ref()
                    .map(|t| format!("; test/atom {:.4} (unigram {:.4}, w/o repl. {:.4})", t.model_per_atom, t.unigram_per_atom, t.unigram_without_replacement_per_atom))
                    .unwrap_or_default(),
                report.seconds
            ));
            let card = ModelCard {
                method: "Alien Space of Science coherence model (Artiles et al., arXiv:2603.01092v2), native unifur reimplementation".into(),
                config,
                options,
                split: rule,
                export_dir: args.export.clone(),
                export_sha256: export_hashes(&args.export)?,
                counts,
                report,
            };
            save_model(&args.out, &lm, &card)?;
            progress(&format!("wrote {}", args.out.display()));
            Ok(())
        }
        SetsAction::Rank(args) => {
            let (lm, card) = load_model(&args.model)?;
            let export = args.export.clone().unwrap_or_else(|| card.export_dir.clone());
            let catalog = AtomCatalog::load(&export)?;
            ensure!(catalog.n_atoms() == lm.config.n_atoms, "the export has {} atoms, the model {}", catalog.n_atoms(), lm.config.n_atoms);
            let mask = args.target_atoms.as_deref().map(|p| read_targets(p, catalog.n_atoms())).transpose()?;
            let year_limit = AtomCatalog::train_year_limit(card.split);
            let index = AvailabilityIndex::build(catalog.n_atoms(), catalog.repertoires(year_limit), DEFAULT_TOP_M)?;
            let split = catalog.split(card.split, card.options.val_permille, card.options.min_atoms)?;
            let train_papers: Vec<usize> = split.train.iter().chain(&split.val).copied().collect();
            let inside = paper_subsets(&catalog, &train_papers, args.k);
            let device = Default::default();
            let t0 = Instant::now();
            let (pool, coherence) = match args.k {
                3 => {
                    let table = TripleTable::build(&lm, &device)?;
                    let triples = enumerate_triples(catalog.n_atoms(), mask.as_deref(), if mask.is_some() { args.min_targets } else { 0 })?;
                    let c: Vec<f64> = triples.iter().map(|&s| table.coherence(s)).collect();
                    (triples.into_iter().map(|s| s.to_vec()).collect::<Vec<_>>(), c)
                }
                4 => {
                    let mut sets = sample_sets(&lm, 4, args.samples, 1.0, args.seed, &device)?;
                    sets.sort();
                    sets.dedup();
                    if let Some(m) = &mask {
                        sets.retain(|s| s.iter().filter(|&&a| m.get(usize::from(a)).copied().unwrap_or(false)).count() >= args.min_targets);
                    }
                    let c = coherence_scores(&lm, &sets, &PermutationPolicy::default(), &device)?;
                    (sets, c)
                }
                other => bail!("--k must be 3 (exhaustive) or 4 (sampled), got {other}"),
            };
            ensure!(pool.len() >= 2, "the candidate pool has {} sets; nothing to rank", pool.len());
            let coherence_seconds = t0.elapsed().as_secs_f64();
            let (rows, summary) = finish_ranking(
                &catalog,
                &index,
                &inside,
                mask.as_deref(),
                pool,
                (coherence, coherence_seconds),
                &args,
            )?;
            let mut text = String::new();
            for row in &rows {
                text.push_str(&serde_json::to_string(row)?);
                text.push('\n');
            }
            write_atomic(&args.out, text.as_bytes())?;
            let summary_path = PathBuf::from(format!("{}.summary.json", args.out.display()));
            write_atomic(&summary_path, serde_json::to_string_pretty(&summary)?.as_bytes())?;
            progress(&format!(
                "ranked {} sets (k={}, beta={}): coherence {:.1}s, availability {:.1}s, fusion {:.2}s; top {} written to {}",
                summary.pool, summary.k, summary.beta, summary.coherence_seconds, summary.availability_seconds, summary.fusion_seconds, summary.top, args.out.display()
            ));
            Ok(())
        }
        SetsAction::Validate(args) => {
            let (lm, card) = load_model(&args.model)?;
            let export = args.export.clone().unwrap_or_else(|| card.export_dir.clone());
            let catalog = AtomCatalog::load(&export)?;
            let report = validate(&lm, &catalog, card.split, &card.options, args.samples, progress)?;
            write_atomic(&args.out, serde_json::to_string_pretty(&report)?.as_bytes())?;
            for d in &report.coherence {
                progress(&format!(
                    "k={}: AUC real>random {:.3}, random>disjoint {:.3}, real>disjoint {:.3}",
                    d.k, d.real_vs_random.auc, d.random_vs_disjoint.auc, d.real_vs_disjoint.auc
                ));
            }
            progress(&format!("wrote {}", args.out.display()));
            Ok(())
        }
    }
}

// ------------------------------------------------------------ certificates --

fn alienset_cert(name: &'static str, theorem: &'static str, residual: f64, tolerance: f64) -> Certificate {
    Certificate { group: "alienset", name, theorem, residual, tolerance }
}

/// Fusion on a fixed pool against the formula written out by hand.
fn fusion_residual() -> Result<f64> {
    let c = [-3.0, -2.5, -4.0, -1.0, -2.0];
    let a = [0.40, 0.10, 0.05, 0.60, 0.20];
    let beta = 0.7;
    let fused = fuse(&c, &a, beta)?;
    // Hand formula: population mean and standard deviation.
    let stats = |xs: &[f64]| {
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        let s = (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt();
        (m, s)
    };
    let (mc, sc) = stats(&c);
    let (ma, sa) = stats(&a);
    let expected: Vec<f64> = c.iter().zip(&a).map(|(ci, ai)| (1.0 - beta) * (ci - mc) / sc + beta * (-(ai - ma) / sa)).collect();
    let residual = fused.iter().zip(&expected).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    // Controls: un-negated availability and sample (n − 1) deviation differ.
    let wrong_sign: Vec<f64> = c.iter().zip(&a).map(|(ci, ai)| (1.0 - beta) * (ci - mc) / sc + beta * (ai - ma) / sa).collect();
    let control = fused.iter().zip(&wrong_sign).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let control_fails = if control > 1e-3 { 0.0 } else { 1.0 };
    // β = 0 is z(C) exactly and β = 1 is −z(A) exactly.
    let z0 = fuse(&c, &a, 0.0)?;
    let zc = zscores(&c)?;
    let ends = z0.iter().zip(&zc).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    Ok(residual.max(control_fails).max(ends))
}

/// The inverted-index availability against a dense brute force.
fn availability_residual() -> Result<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(0xA7A1);
    let n = 24;
    let reps: Vec<Vec<u16>> = (0..60)
        .map(|_| {
            let size = rng.random_range(1..8);
            random_set(&mut rng, n, size)
        })
        .collect::<Result<_>>()?;
    let index = AvailabilityIndex::build(n, reps, DEFAULT_TOP_M)?;
    let mut scratch = AvailabilityScratch::default();
    let mut worst: f64 = 0.0;
    let mut control_gap: f64 = 0.0;
    for k in 2..6 {
        for _ in 0..20 {
            let s = random_set(&mut rng, n, k)?;
            let fast = index.availability(&s, &mut scratch)?;
            let mut dense = index.similarities_dense(&s)?;
            dense.sort_by(|a, b| b.total_cmp(a));
            let top: Vec<f64> = dense.iter().take(DEFAULT_TOP_M).copied().collect();
            let median = 0.5 * (top.get(4).copied().unwrap_or(0.0) + top.get(5).copied().unwrap_or(0.0));
            worst = worst.max((fast - median).abs());
            let mean = top.iter().sum::<f64>() / top.len().max(1) as f64;
            control_gap = control_gap.max((fast - mean).abs());
        }
    }
    // Control: the mean of the top-m is a different statistic and must differ.
    let control_fails = if control_gap > 1e-3 { 0.0 } else { 1.0 };
    // A set nobody touches has availability exactly 0.
    let lonely = AvailabilityIndex::build(4, vec![vec![0]; 12], DEFAULT_TOP_M)?;
    let zero = lonely.availability(&[1, 2, 3], &mut scratch)?.abs();
    Ok(worst.max(control_fails).max(zero))
}

/// Order invariance, relabelling equivariance and table equality of `C`.
fn coherence_identity_residual() -> Result<f64> {
    let device = Default::default();
    <Cpu as Backend>::seed(&device, 11);
    let config = CoherenceConfig { n_atoms: 9, d_model: 16, n_heads: 2, n_layers: 2, d_ff: 32, dropout: 0.0, max_len: 8 };
    let lm = CoherenceLm { config: config.clone(), model: config.init::<Cpu>(&device)? };
    let policy = PermutationPolicy::default();
    let sets: Vec<Vec<u16>> = vec![vec![0, 3, 7], vec![1, 2, 4, 8], vec![5, 6], vec![0, 1, 2, 3, 4, 5]];
    let base = coherence_scores(&lm, &sets, &policy, &device)?;
    // Order invariance: the same sets given in another order score the same.
    let reordered: Vec<Vec<u16>> = sets.iter().map(|s| s.iter().rev().copied().collect()).collect();
    let again = coherence_scores(&lm, &reordered, &policy, &device)?;
    let mut residual = base.iter().zip(&again).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    // Relabelling: σ applied to both the model and the sets.
    let sigma = [4usize, 7, 0, 8, 2, 5, 1, 3, 6];
    let relabelled = CoherenceLm { config: config.clone(), model: lm.model.relabelled(&config, &sigma)? };
    let mapped: Vec<Vec<u16>> = sets.iter().map(|s| s.iter().map(|&a| sigma.get(usize::from(a)).map(|&x| x as u16).unwrap_or(0)).collect()).collect();
    let moved = coherence_scores(&relabelled, &mapped, &policy, &device)?;
    residual = residual.max(base.iter().zip(&moved).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
    // The exhaustive table is the same estimator.
    let table = TripleTable::build(&lm, &device)?;
    for s in enumerate_triples(config.n_atoms, None, 3)?.iter().step_by(7) {
        let direct = coherence_scores(&lm, &[s.to_vec()], &policy, &device)?;
        residual = residual.max((table.coherence(*s) - direct.first().copied().unwrap_or(f64::NAN)).abs());
    }
    // Controls: the relabelled model on the *unmapped* sets, and a single
    // ordering instead of the marginal, must both differ.
    let unmapped = coherence_scores(&relabelled, &sets, &policy, &device)?;
    let relabel_gap = base.iter().zip(&unmapped).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let first = sets.first().cloned().unwrap_or_default();
    let forward = single_order_score(&lm, &first, &device)?;
    let backward = single_order_score(&lm, &first.iter().rev().copied().collect::<Vec<_>>(), &device)?;
    let controls_fail = if relabel_gap > 1e-4 && (forward - backward).abs() > 1e-5 { 0.0 } else { 1.0 };
    Ok(residual.max(controls_fail))
}

/// The enumeration covers exactly `C(n, 3)` distinct sorted sets, and the
/// target restriction exactly `C(t,2)(n−t) + C(t,3)` of them.
fn enumeration_residual() -> Result<f64> {
    let n = 14;
    let all = enumerate_triples(n, None, 3)?;
    let distinct: HashSet<[u16; 3]> = all.iter().copied().collect();
    let sorted = all.iter().all(|s| matches!(s, [a, b, c] if a < b && b < c && usize::from(*c) < n));
    let mut residual = (all.len() as f64 - binomial(n, 3)? as f64).abs() + (distinct.len() as f64 - all.len() as f64).abs();
    if !sorted {
        residual += 1.0;
    }
    let mask: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let t = mask.iter().filter(|&&m| m).count();
    let restricted = enumerate_triples(n, Some(&mask), 2)?;
    let expected = binomial(t, 2)? * (n - t) as u64 + binomial(t, 3)?;
    residual += (restricted.len() as f64 - expected as f64).abs();
    let ok_targets = restricted.iter().all(|s| s.iter().filter(|&&a| mask.get(usize::from(a)).copied().unwrap_or(false)).count() >= 2);
    if !ok_targets {
        residual += 1.0;
    }
    // Control: ordered pairs (`b ≥ a`, not `b > a`) overcount and must fail.
    let mut sloppy = 0u64;
    for a in 0..n {
        for b in a..n {
            sloppy += (b + 1..n).count() as u64;
        }
    }
    let control_fails = if sloppy == binomial(n, 3)? { 1.0 } else { 0.0 };
    Ok(residual + control_fails)
}

/// The `alienset` certificate group (registered in [`crate::verify`]).
/// Each residual carries a control that a wrong implementation trips.
pub fn certificate_checks() -> Result<Vec<Certificate>> {
    Ok(vec![
        alienset_cert(
            "z_fusion_matches_the_formula",
            "On a fixed 5-set pool, fuse(C, A, 0.7) equals (1-b)(C-mean)/sd + b(-(A-mean)/sd) with \
             population moments to 1e-12, and b = 0 gives z(C) (control: the un-negated \
             availability term differs by more than 1e-3).",
            fusion_residual()?,
            1e-12,
        ),
        alienset_cert(
            "availability_is_the_median_of_the_top_m",
            "On 60 random repertoires over 24 atoms and 80 random sets, the inverted-index A(S) \
             equals the median of the 10 largest dense IDF-cosine similarities (mean of the 5th \
             and 6th) to 1e-12, and an untouched set scores 0 (control: the top-10 mean differs).",
            availability_residual()?,
            1e-12,
        ),
        alienset_cert(
            "coherence_is_order_and_relabel_invariant",
            "C(S) of a random 9-atom model is unchanged when S is given in another order and when \
             atom ids are relabelled in both the model and S, and the exhaustive triple table \
             equals the direct estimator (controls: the relabelled model on unmapped sets, and a \
             single ordering forward vs backward, both differ).",
            coherence_identity_residual()?,
            1e-5,
        ),
        alienset_cert(
            "exhaustive_triples_cover_c_n_3",
            "enumerate_triples(14) yields exactly C(14,3) = 364 distinct sorted sets, and the \
             >= 2-of-5-targets restriction exactly C(5,2)*9 + C(5,3) (control: ordered-pair \
             enumeration overcounts).",
            enumeration_residual()?,
            0.0,
        ),
    ])
}

// ------------------------------------------------------------------- tests --

#[cfg(test)]
// Tests unwrap and index: a failure here is the test failing, with its value.
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn tiny_catalog() -> AtomCatalog {
        let atoms = (0..6u16).map(|i| Atom { id: i, n_papers: 0, name: format!("atom {i}"), description: String::new() }).collect();
        let papers = vec![
            Paper { paper_id: "p1".into(), year: Some(2023), atoms: vec![0, 1, 2] },
            Paper { paper_id: "p2".into(), year: Some(2024), atoms: vec![1, 2] },
            Paper { paper_id: "p3".into(), year: Some(2025), atoms: vec![3, 4, 5] },
            Paper { paper_id: "p4".into(), year: Some(2024), atoms: vec![] },
        ];
        let authors = vec![
            ("a1".to_string(), vec!["p1".to_string(), "p3".to_string()], vec![0, 1, 2, 3, 4, 5]),
            ("a2".to_string(), vec!["p2".to_string()], vec![1, 2]),
        ];
        AtomCatalog::from_parts(atoms, papers, authors).unwrap()
    }

    #[test]
    fn catalog_validates_and_builds_year_bounded_repertoires() {
        let c = tiny_catalog();
        assert_eq!(c.counts().papers_with_two_or_more, 3);
        assert_eq!(c.repertoires(Some(2024)), vec![vec![0, 1, 2], vec![1, 2]]);
        assert_eq!(c.repertoires(None).first().unwrap().len(), 6);
    }

    #[test]
    fn catalog_rejects_a_wrong_author_union() {
        let atoms = (0..3u16).map(|i| Atom { id: i, n_papers: 0, name: "x".into(), description: String::new() }).collect();
        let papers = vec![Paper { paper_id: "p".into(), year: None, atoms: vec![0, 1] }];
        let err = AtomCatalog::from_parts(atoms, papers, vec![("a".into(), vec!["p".into()], vec![0])]).unwrap_err();
        assert!(format!("{err}").contains("union"), "{err}");
    }

    #[test]
    fn catalog_rejects_unsorted_atoms_and_unknown_papers() {
        let atoms: Vec<Atom> = (0..3u16).map(|i| Atom { id: i, n_papers: 0, name: "x".into(), description: String::new() }).collect();
        let bad = vec![Paper { paper_id: "p".into(), year: None, atoms: vec![1, 0] }];
        assert!(AtomCatalog::from_parts(atoms.clone(), bad, vec![]).is_err());
        let ok = vec![Paper { paper_id: "p".into(), year: None, atoms: vec![0] }];
        assert!(AtomCatalog::from_parts(atoms, ok, vec![("a".into(), vec!["q".into()], vec![0])]).is_err());
    }

    #[test]
    fn temporal_split_holds_out_later_years() {
        let c = tiny_catalog();
        let s = c.split(SplitRule::Temporal { last_train_year: 2024 }, 0, 2);
        // val_permille 0 leaves no validation papers: refused.
        assert!(s.is_err());
        let s = c.split(SplitRule::Temporal { last_train_year: 2024 }, 999, 2).unwrap();
        assert_eq!(s.test, vec![2]);
        assert_eq!(s.skipped_small, 1);
    }

    #[test]
    fn permutations_are_complete_and_distinct() {
        let p = all_permutations(4);
        assert_eq!(p.len(), 24);
        let d: HashSet<Vec<usize>> = p.into_iter().collect();
        assert_eq!(d.len(), 24);
        assert_eq!(all_permutations(1).len(), 1);
    }

    #[test]
    fn median_of_top_m_pads_with_zeros() {
        let mut v = vec![0.9, 0.8];
        assert_eq!(median_of_top_m(&mut v, 10), 0.0);
        let mut v: Vec<f64> = (1..=12).map(f64::from).collect();
        // Top 10 of 1..=12 is 12..=3; its median is (8 + 7) / 2.
        assert_eq!(median_of_top_m(&mut v, 10), 7.5);
        let mut v: Vec<f64> = (1..=5).map(f64::from).collect();
        assert_eq!(median_of_top_m(&mut v, 3), 4.0);
    }

    #[test]
    fn auc_matches_hand_counts() {
        assert_eq!(auc(&[3.0, 4.0], &[1.0, 2.0]).unwrap(), 1.0);
        assert_eq!(auc(&[1.0], &[1.0]).unwrap(), 0.5);
        assert!((auc(&[1.0, 3.0], &[2.0]).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn zscores_refuse_a_constant_pool() {
        assert!(zscores(&[1.0, 1.0]).is_err());
        assert!(fuse(&[1.0, 2.0], &[0.1, 0.2], 1.5).is_err());
    }

    #[test]
    fn paper_subsets_enumerates_every_triple() {
        let c = tiny_catalog();
        let s = paper_subsets(&c, &[0, 1, 2], 3);
        assert_eq!(s.len(), 2);
        assert!(s.contains(&vec![0, 1, 2]));
        let s2 = paper_subsets(&c, &[0], 2);
        assert_eq!(s2.len(), 3);
    }

    #[test]
    fn certificates_hold() {
        for c in certificate_checks().unwrap() {
            assert!(c.passed(), "{} residual {} > {}", c.name, c.residual, c.tolerance);
        }
    }

    #[test]
    fn training_reduces_validation_nll_on_a_planted_corpus() {
        // Two planted "fields": atoms 0..4 co-occur, atoms 4..8 co-occur.
        let atoms: Vec<Atom> = (0..8u16).map(|i| Atom { id: i, n_papers: 0, name: format!("a{i}"), description: String::new() }).collect();
        let mut papers = Vec::new();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for i in 0..400 {
            let base: u16 = if i % 2 == 0 { 0 } else { 4 };
            let mut s = random_set(&mut rng, 4, 3).unwrap();
            for a in &mut s {
                *a += base;
            }
            papers.push(Paper { paper_id: format!("p{i}"), year: Some(2024), atoms: s });
        }
        let catalog = AtomCatalog::from_parts(atoms, papers, vec![]).unwrap();
        let split = catalog.split(SplitRule::All, 100, 2).unwrap();
        let config = CoherenceConfig { n_atoms: 8, d_model: 16, n_heads: 2, n_layers: 1, d_ff: 32, dropout: 0.0, max_len: 6 };
        let options = TrainOptions { epochs: 6, batch_size: 32, lr: 3e-3, warmup_steps: 5, patience: 6, ..TrainOptions::default() };
        let (lm, report) = train_coherence(&catalog, &split, &config, &options, &mut |_| {}).unwrap();
        assert!(report.val.model_per_atom < report.val.unigram_without_replacement_per_atom, "{report:?}");
        let device = Default::default();
        let c = coherence_scores(&lm, &[vec![0, 1, 2], vec![0, 5, 6]], &PermutationPolicy::default(), &device).unwrap();
        assert!(c.first().unwrap() > c.get(1).unwrap(), "{c:?}");
    }
}
