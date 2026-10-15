//! The specialist coding student: a lightweight hybrid-attention language
//! model with per-language MoSME expert boxes, a router that can say *which*
//! language specialist a token belongs to without running any expert, and a
//! geometric-reasoning stream as a first-class citizen (the repo's measured
//! direction: reasoning through geometry, not brute-force token guessing).
//!
//! # Architecture
//!
//! - **Trunk** ([`crate::lm::LanguageModel`]): causal, rotary positions with
//!   the Qwen3-Next partial-rotary fraction `0.25`, grouped-query attention,
//!   QK-Norm and RMSNorm on, and the repo's measured-best `3:1`
//!   linear:dense attention schedule (`docs/Hybrid-Attention.md`: three
//!   cheap linear layers per precise dense one, the dense one last in each
//!   group). Every second feed-forward is a MoSME site holding the
//!   per-language specialist boxes of [`StudentConfig::coding_boxes`]
//!   (`coding:rust`, `coding:python`, …; the default set is
//!   [`crate::expert_index::CODING_LANGUAGES`]), plus one `field:{name}` box
//!   per [`StudentConfig::field_boxes`] entry for non-coding specialists, with
//!   micro experts per `docs/Mixture-of-Specialized-Micro-Experts.md`.
//! - **Geometric stream**: the trunk's last hidden states are chunk-pooled
//!   into `geom_slots` concept points in a `geom_dim`-dimensional learned
//!   Riemannian space, relaxed by one certified [`crate::geometry::GeomBlock`]
//!   refinement (geodesic attention under a positive-definite metric,
//!   energy descent with a Lipschitz-bounded step), and read back into the
//!   residual stream through a **zero-initialized** projection — so a fresh
//!   student is *exactly* the plain trunk, and training the stream is pure
//!   gain from a known-good start.
//! - **Router-which-knows**: a probe [`crate::mosme::HierarchicalRouter`]
//!   over the same box layout reads the trunk's hidden states and reports
//!   per-box traffic (`LanguageRouting::traffic`), each token's most likely
//!   language specialist (`LanguageRouting::top_language`) and its most
//!   likely box overall, field boxes included (`LanguageRouting::top_box`) —
//!   gates only, no expert MLP is run.
//!
//! # What the wave-2 per-language training agent gets
//!
//! - **Model build**: [`GeometricStudent::new`] (fallible, validates the
//!   config), config sidecar via serde (every field `#[serde(default)]`),
//!   checkpoints through [`crate::checkpoint`] (the whole student is one
//!   Burn `Module`).
//! - **Forward**: [`GeometricStudent::forward`] /
//!   [`GeometricStudent::forward_depth`] (refine-depth override — the
//!   test-time-compute knob) returning logits, the modified hidden states,
//!   the certified [`crate::geometry::RefineTrace`] and the routing readout.
//! - **Box-traffic readout**: [`GeometricStudent::route_languages`], or
//!   `StudentForward::routing` from a forward pass.
//! - **Loss attach point**: `StudentForward::logits` is `[b, n, vocab]`
//!   from the trunk's own tied readout — standard next-token cross-entropy
//!   applies directly, and `out.refine.energy` plus the MoSME balance terms
//!   (in `routing.trunk_layers` / the trunk's `LmOutput.balance_loss`) are
//!   the auxiliary signals. The trunk itself is reachable via
//!   [`GeometricStudent::trunk`] for the crate's existing `next_token_step`
//!   machinery when the geometric stream should not see a batch.
//!
//! # Wave 2 (delivered): per-language specialist training
//!
//! - **Data**: [`LanguageBatch`] loads a traces JSONL (the
//!   `{id, language, prompt, completion, source, ...}` schema, or the older
//!   SFT `instruction`/`response` schema with `language = unknown`), filters
//!   by language, tokenizes `prompt + completion` with a completion-only
//!   loss mask, truncates at the context and shuffles deterministically. A
//!   line may also carry the NINA pair-file contract's `"field"` key: when
//!   the field names a configured [`StudentConfig::field_boxes`] entry, it
//!   supervises that field's `field:{name}` box (a known language still
//!   wins when both are present; an unknown field supervises nothing).
//! - **Step**: [`GeometricStudent::specialist_step`] — masked next-token CE
//!   on the completion tokens plus the router-which-knows auxiliary (the
//!   known language's CE against the probe's box distribution, on the
//!   completion tokens), with the trunk's MoSME balance/z terms at the
//!   weights `crate::lm` uses.
//! - **Loop**: [`train_language_specialist`] with [`SpecialistTrainConfig`]
//!   → [`SpecialistTrainReport`]: AdamW, optional gradient clipping, the
//!   `crate::train` RNG discipline (host ChaCha serialized, device reseeded
//!   per step), content-addressed checkpoints with verified training state,
//!   and resume that restores weights, optimizer moments, host RNG and
//!   step exactly (see the resume test's comment for the measured
//!   backward-nondeterminism caveat on all-trainable MoSME trunks).
//! - **CLI**: `dblocks student info|route|train` in `src/main.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::{
    module::{Initializer, Module},
    nn::{Linear, LinearConfig},
    optim::{AdamWConfig, Optimizer},
    tensor::{activation::log_softmax, backend::Backend, Int, Tensor},
};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::{
    checkpoint::{self, DatasetIdentity, TrainState},
    expert_index::{
        coding_spec_for, validate_coding_box_names, validate_field_box_names, MosmeSpec,
        CODING_LANGUAGES,
    },
    geometry::{GeomBlock, GeomConfig, RefineTrace},
    hybrid::{AttentionSchedule, PositionKind},
    lm::{LanguageModel, LmConfig},
    mosme::{HierarchicalRouter, MosmeConfig, TrainableSet},
    tokenizer::{ByteTokenizer, Special, VOCAB_SIZE},
    vit::{MosmeTrunkConfig, NormKind},
};

/// A programming language with a specialist box in the student's MoSME
/// trunk. `Web` is the HTML/CSS/JS-mixed specialist; plain
/// JavaScript/TypeScript is [`CodeLanguage::JsTs`].
///
/// The first six variants are the shipped box set — the default
/// [`StudentConfig::coding_boxes`] and every checkpoint written before the
/// box set became configurable. `Go`, `Java`, `CSharp` and `Php` are
/// trainable when a config's `coding_boxes` lists them; they are not in the
/// default set, so an old sidecar still builds the model it described. The
/// remaining top-10 languages (`ruby`, `swift`, `kotlin`, `sql`, `dart`,
/// `scala`) are documented in
/// [`crate::expert_index::PLANNED_CODING_LANGUAGES`] as later registry
/// entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeLanguage {
    Rust,
    Python,
    C,
    Cpp,
    JsTs,
    Web,
    Go,
    Java,
    #[serde(rename = "csharp")]
    CSharp,
    Php,
}

impl CodeLanguage {
    /// Every shipped language, in registry (box router) order: the default
    /// [`StudentConfig::coding_boxes`] and every checkpoint written before
    /// the box set became configurable. Their [`Self::index`] values are
    /// frozen — checkpoints and seeds are keyed by them.
    pub const ALL: [Self; 6] = [
        Self::Rust,
        Self::Python,
        Self::C,
        Self::Cpp,
        Self::JsTs,
        Self::Web,
    ];

    /// The languages added after the shipped six, in registry order:
    /// trainable whenever a config's `coding_boxes` lists them, absent from
    /// the default box set.
    pub const EXTENDED: [Self; 4] = [Self::Go, Self::Java, Self::CSharp, Self::Php];

    /// Every known language: [`Self::ALL`] then [`Self::EXTENDED`]. This is
    /// the order [`Self::index`] numbers.
    pub const KNOWN: [Self; 10] = [
        Self::Rust,
        Self::Python,
        Self::C,
        Self::Cpp,
        Self::JsTs,
        Self::Web,
        Self::Go,
        Self::Java,
        Self::CSharp,
        Self::Php,
    ];

    /// The registry key: `name` is the `l` in the box id `coding:{l}`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::JsTs => "jsts",
            Self::Web => "web",
            Self::Go => "go",
            Self::Java => "java",
            Self::CSharp => "csharp",
            Self::Php => "php",
        }
    }

    /// Parse a language key (`rust`, `python`, `c`, `cpp`, `jsts`, `web`,
    /// `go`, `java`, `csharp`, `php`). A few obvious aliases are accepted
    /// (`c++`, `js`, `ts`, `golang`, `c#`); anything else is an error naming
    /// the expected keys.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(match text.trim().to_ascii_lowercase().as_str() {
            "rust" | "rs" => Self::Rust,
            "python" | "py" => Self::Python,
            "c" => Self::C,
            "cpp" | "c++" => Self::Cpp,
            "jsts" | "js" | "ts" | "javascript" | "typescript" => Self::JsTs,
            "web" | "html" => Self::Web,
            "go" | "golang" => Self::Go,
            "java" => Self::Java,
            "csharp" | "c#" | "cs" => Self::CSharp,
            "php" => Self::Php,
            other => anyhow::bail!(
                "unknown language '{other}' (expected rust|python|c|cpp|jsts|web|go|java|csharp|\
                 php; the remaining top-10 are later registry entries, see \
                 expert_index::PLANNED_CODING_LANGUAGES)"
            ),
        })
    }

    /// This language's MoSME box id, `coding:{name}`.
    pub fn box_id(self) -> String {
        format!("coding:{}", self.name())
    }

    /// Index in [`CodeLanguage::KNOWN`], stable across builds: it keys the
    /// pipeline's per-language seeds and the
    /// [`crate::langdata::RouterAgreement`] tallies. The box-router index of a
    /// language is its position in the *config's* `coding_boxes` instead (see
    /// [`StudentConfig::box_target`]); the two agree on the default config.
    pub fn index(self) -> usize {
        CodeLanguage::KNOWN
            .iter()
            .position(|l| *l == self)
            .map_or(CodeLanguage::KNOWN.len(), |idx| idx)
    }
}

/// The MoSME box one training example supervises: a coding language's box,
/// or a configured field box (a *non-coding* specialist, see
/// [`StudentConfig::field_boxes`]). The payload of both variants is the box's
/// index in the router's output, already resolved against the config that
/// produced it — the resolution is [`StudentConfig::box_target`]'s job,
/// because the coding box count (and therefore the field boxes' base) is
/// configuration, not a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxTarget {
    /// A `coding:{lang}` box, at the language's position in the config's
    /// `coding_boxes`.
    Language(usize),
    /// A `field:{name}` box, at `coding_boxes.len()` + the field's position
    /// in [`StudentConfig::field_boxes`].
    Field(usize),
}

impl BoxTarget {
    /// Position in the box router's output (and in
    /// [`LanguageRouting::top_box`]).
    pub fn box_index(self) -> usize {
        match self {
            Self::Language(idx) | Self::Field(idx) => idx,
        }
    }
}

fn default_hidden() -> usize {
    512
}
fn default_layers() -> usize {
    8
}
fn default_heads() -> usize {
    8
}
fn default_kv_heads() -> usize {
    2
}
fn default_context() -> usize {
    1024
}
fn default_intermediate() -> usize {
    256
}
fn default_cond() -> usize {
    64
}
fn default_freq() -> usize {
    64
}
fn default_attention() -> String {
    "3:1".to_string()
}
fn default_rotary_fraction() -> f64 {
    0.25
}
fn default_true() -> bool {
    true
}
fn default_norm_kind() -> NormKind {
    NormKind::Rms
}
fn default_experts_per_box() -> usize {
    4
}
fn default_one() -> usize {
    1
}
fn default_mosme_every() -> usize {
    2
}
fn default_geom_slots() -> usize {
    16
}
fn default_geom_dim() -> usize {
    32
}
fn default_geom_refine() -> usize {
    4
}
fn default_geom_repulsion() -> f64 {
    0.5
}
fn default_geom_cond() -> usize {
    64
}
fn default_geom_freq() -> usize {
    32
}
fn default_vocab() -> usize {
    VOCAB_SIZE
}
/// The default coding box set: the six shipped languages, in registry order.
/// This *is* the back-compat contract — a sidecar written before
/// `coding_boxes` existed parses to exactly the box set its checkpoint was
/// trained with.
fn default_coding_boxes() -> Vec<String> {
    CODING_LANGUAGES
        .iter()
        .map(|(key, _)| (*key).to_string())
        .collect()
}

/// Shape of a [`GeometricStudent`].
///
/// Every field carries `#[serde(default)]`, so a sidecar written by an older
/// build still parses, and `StudentConfig::default()` is the honest local
/// size: a config this machine (12 CPU threads, 46 GB RAM) can actually
/// step. [`StudentConfig::few_billion_active`] is the scale-up and warns
/// that it exceeds local step budgets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudentConfig {
    /// Residual-stream width.
    #[serde(default = "default_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_layers")]
    pub num_layers: usize,
    #[serde(default = "default_heads")]
    pub num_heads: usize,
    /// Grouped-query attention: key/value heads per layer.
    #[serde(default = "default_kv_heads")]
    pub num_kv_heads: usize,
    /// Nominal context the cost model is evaluated at. Positions are rotary,
    /// so the sequence length itself is unbounded.
    #[serde(default = "default_context")]
    pub context: usize,
    /// Feed-forward width of one micro expert (and of the dense layers).
    /// FFN/4–FFN/8 of a dense trunk per the MoSME sizing notes.
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_cond")]
    pub cond_hidden_size: usize,
    #[serde(default = "default_freq")]
    pub frequency_embedding_size: usize,
    /// Attention schedule text, parsed by
    /// [`crate::hybrid::AttentionSchedule::parse`]. `3:1` (three linear per
    /// dense) is the repo's measured-best hybrid ratio.
    #[serde(default = "default_attention")]
    pub attention: String,
    /// Fraction of each head dimension the rotary embedding covers
    /// (Qwen3-Next uses 0.25).
    #[serde(default = "default_rotary_fraction")]
    pub rotary_fraction: f64,
    /// QK-Norm on queries and keys before rotary.
    #[serde(default = "default_true")]
    pub qk_norm: bool,
    /// Trunk normalization; RMSNorm by default.
    #[serde(default = "default_norm_kind")]
    pub norm_kind: NormKind,
    /// Micro experts in each per-language box.
    #[serde(default = "default_experts_per_box")]
    pub experts_per_box: usize,
    /// Boxes (languages) selected per token.
    #[serde(default = "default_one")]
    pub top_box: usize,
    /// Experts selected within a box.
    #[serde(default = "default_one")]
    pub top_expert: usize,
    /// Replace the feed-forward of every `mosme_every`-th layer with the
    /// MoSME site (2 = every second layer, layer 0 stays dense).
    #[serde(default = "default_mosme_every")]
    pub mosme_every: usize,
    /// Concept points of the geometric stream.
    #[serde(default = "default_geom_slots")]
    pub geom_slots: usize,
    /// Dimension of the geometric (Riemannian) space.
    #[serde(default = "default_geom_dim")]
    pub geom_dim: usize,
    /// Certified relaxation iterations of the geometric refinement.
    #[serde(default = "default_geom_refine")]
    pub geom_refine_steps: usize,
    /// Repulsion coefficient the geometric block starts at (keeps the slots
    /// from collapsing onto one point).
    #[serde(default = "default_geom_repulsion")]
    pub geom_repulsion: f64,
    /// Conditioning width of the geometric block's sigma embedder.
    #[serde(default = "default_geom_cond")]
    pub geom_cond_hidden: usize,
    #[serde(default = "default_geom_freq")]
    pub geom_frequency: usize,
    /// The byte tokenizer's vocabulary; tied input/output embedding.
    #[serde(default = "default_vocab")]
    pub vocab_size: usize,
    /// The coding languages with their own routing box, as `coding:{name}`
    /// in list order — the list order *is* the router's box order, so this
    /// pins every checkpoint's routing layout. Defaults to the six shipped
    /// languages ([`crate::expert_index::CODING_LANGUAGES`]), so a sidecar
    /// written before this field existed builds the model it described;
    /// [`crate::expert_index::EXTENDED_CODING_LANGUAGES`] are trainable by
    /// naming them here.
    #[serde(default = "default_coding_boxes")]
    pub coding_boxes: Vec<String>,
    /// Non-coding fields with their own routing box, appended after the
    /// coding boxes as `field:{name}` (e.g. `field:pentest`). Empty by
    /// default, so a sidecar written before field boxes existed parses to
    /// exactly the config it described. When non-empty the MoSME grows one
    /// box per field and `LanguageBatch` lines carrying a matching `"field"`
    /// key supervise it; an unknown field supervises nothing, the same rule
    /// as an unknown language.
    #[serde(default)]
    pub field_boxes: Vec<String>,
}

impl Default for StudentConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden(),
            num_layers: default_layers(),
            num_heads: default_heads(),
            num_kv_heads: default_kv_heads(),
            context: default_context(),
            intermediate_size: default_intermediate(),
            cond_hidden_size: default_cond(),
            frequency_embedding_size: default_freq(),
            attention: default_attention(),
            rotary_fraction: default_rotary_fraction(),
            qk_norm: default_true(),
            norm_kind: default_norm_kind(),
            experts_per_box: default_experts_per_box(),
            top_box: default_one(),
            top_expert: default_one(),
            mosme_every: default_mosme_every(),
            geom_slots: default_geom_slots(),
            geom_dim: default_geom_dim(),
            geom_refine_steps: default_geom_refine(),
            geom_repulsion: default_geom_repulsion(),
            geom_cond_hidden: default_geom_cond(),
            geom_frequency: default_geom_freq(),
            vocab_size: default_vocab(),
            coding_boxes: default_coding_boxes(),
            field_boxes: Vec::new(),
        }
    }
}

impl StudentConfig {
    /// The scale-up configuration: a few billion **active** parameters per
    /// token (many more resident, which is the point of the sparse boxes).
    ///
    /// This exceeds what this machine can step — it is the validated target
    /// shape for a bigger budget, printed with its counted costs so the
    /// claim is checkable. The default config stays local-sized.
    pub fn few_billion_active() -> Self {
        let config = Self {
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            context: 4096,
            intermediate_size: 1024,
            cond_hidden_size: 682,
            frequency_embedding_size: 256,
            experts_per_box: 32,
            top_box: 2,
            top_expert: 4,
            geom_slots: 64,
            geom_dim: 128,
            geom_refine_steps: 8,
            geom_cond_hidden: 256,
            geom_frequency: 64,
            ..Self::default()
        };
        match config.cost() {
            Ok(cost) => eprintln!(
                "warning: few_billion_active is {:.2}B active / {:.2}B total parameters and \
                 {:.1} GFLOPs per token -- far beyond local step budgets; use the default \
                 config on this machine",
                cost.active_params as f64 / 1e9,
                cost.total_params as f64 / 1e9,
                cost.trunk_flops_per_token / 1e9,
            ),
            Err(err) => eprintln!(
                "warning: few_billion_active exceeds local step budgets (cost unavailable: {err})"
            ),
        }
        config
    }

    /// Cross-check every field, naming the one at fault. Constructors that
    /// take this config call it first, so a bad architecture is reported
    /// here rather than inside a weight initializer.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.vocab_size == VOCAB_SIZE,
            "vocab_size must be the byte tokenizer's {VOCAB_SIZE}, got {}",
            self.vocab_size
        );
        anyhow::ensure!(self.hidden_size >= 1, "hidden_size must be positive");
        anyhow::ensure!(
            self.num_heads >= 1 && self.hidden_size % self.num_heads == 0,
            "hidden_size ({}) must be divisible by num_heads ({})",
            self.hidden_size,
            self.num_heads
        );
        let head_dim = self.hidden_size / self.num_heads;
        anyhow::ensure!(
            head_dim % 2 == 0,
            "the head dim ({head_dim}) must be even for rotary positions \
             (hidden_size {} / num_heads {})",
            self.hidden_size,
            self.num_heads
        );
        anyhow::ensure!(
            self.num_kv_heads >= 1
                && self.num_kv_heads <= self.num_heads
                && self.num_heads % self.num_kv_heads == 0,
            "num_kv_heads ({}) must divide num_heads ({}) from below",
            self.num_kv_heads,
            self.num_heads
        );
        anyhow::ensure!(self.num_layers >= 1, "num_layers must be positive");
        anyhow::ensure!(
            self.context >= 2,
            "context must cover at least 2 positions, got {}",
            self.context
        );
        anyhow::ensure!(
            self.intermediate_size >= 1,
            "intermediate_size must be positive"
        );
        anyhow::ensure!(
            self.cond_hidden_size >= 1,
            "cond_hidden_size must be positive"
        );
        anyhow::ensure!(
            self.frequency_embedding_size >= 2 && self.frequency_embedding_size % 2 == 0,
            "frequency_embedding_size must be even and at least 2, got {}",
            self.frequency_embedding_size
        );
        AttentionSchedule::parse(&self.attention, self.num_layers, 64, 8).map_err(|err| {
            anyhow::anyhow!("attention schedule '{}' is invalid: {err}", self.attention)
        })?;
        anyhow::ensure!(
            self.rotary_fraction > 0.0 && self.rotary_fraction <= 1.0,
            "rotary_fraction must be in (0, 1], got {}",
            self.rotary_fraction
        );
        anyhow::ensure!(
            self.experts_per_box >= 1,
            "experts_per_box must be at least 1, got {}",
            self.experts_per_box
        );
        validate_coding_box_names(&self.coding_boxes)?;
        anyhow::ensure!(
            (1..=self.coding_boxes.len()).contains(&self.top_box),
            "top_box must be in [1, {}], got {}",
            self.coding_boxes.len(),
            self.top_box
        );
        anyhow::ensure!(
            self.top_expert >= 1 && self.top_expert <= self.experts_per_box,
            "top_expert must be in [1, {}], got {}",
            self.experts_per_box,
            self.top_expert
        );
        anyhow::ensure!(
            self.mosme_every >= 1,
            "mosme_every must be at least 1, got {}",
            self.mosme_every
        );
        anyhow::ensure!(
            self.geom_slots >= 2,
            "geom_slots must be at least 2, got {}",
            self.geom_slots
        );
        anyhow::ensure!(self.geom_dim >= 1, "geom_dim must be positive");
        anyhow::ensure!(
            self.geom_refine_steps >= 1,
            "geom_refine_steps must be at least 1, got {}",
            self.geom_refine_steps
        );
        anyhow::ensure!(
            self.geom_repulsion > 0.0,
            "geom_repulsion must be positive, got {}",
            self.geom_repulsion
        );
        anyhow::ensure!(
            self.geom_cond_hidden >= 1,
            "geom_cond_hidden must be positive"
        );
        anyhow::ensure!(
            self.geom_frequency >= 2 && self.geom_frequency % 2 == 0,
            "geom_frequency must be even and at least 2, got {}",
            self.geom_frequency
        );
        validate_field_box_names(&self.field_boxes)?;
        Ok(())
    }

    /// The per-language specialist boxes this config builds — one
    /// `coding:{name}` box per [`Self::coding_boxes`] entry, in list order —
    /// plus one `field:{name}` box per [`Self::field_boxes`] entry, appended
    /// after the coding boxes so the coding box positions — and the
    /// checkpoints keyed by them — are undisturbed.
    pub fn mosme_spec(&self) -> anyhow::Result<MosmeSpec> {
        coding_spec_for(
            &self.coding_boxes,
            self.experts_per_box,
            self.top_box,
            self.top_expert,
        )?
        .with_field_boxes(&self.field_boxes, self.experts_per_box)
    }

    /// Total routing boxes: the configured coding boxes plus the field boxes.
    pub fn num_boxes(&self) -> usize {
        self.coding_boxes.len() + self.field_boxes.len()
    }

    /// The box-router index of `field:{field}`, when the field is configured.
    pub fn field_box_index(&self, field: &str) -> Option<usize> {
        self.field_boxes
            .iter()
            .position(|f| f == field)
            .map(|idx| self.coding_boxes.len() + idx)
    }

    /// What one training example supervises in the probe router: its language
    /// box when the language names a configured coding box (unchanged whether
    /// or not a field is also present), else its field box when the field is
    /// configured. An unconfigured language or an unknown field supervises
    /// nothing.
    pub fn box_target(
        &self,
        language: Option<CodeLanguage>,
        field: Option<&str>,
    ) -> Option<BoxTarget> {
        match language {
            Some(language) => self
                .coding_boxes
                .iter()
                .position(|c| c == language.name())
                .map(BoxTarget::Language),
            None => self.field_box_index(field?).map(BoxTarget::Field),
        }
    }

    /// The trunk configuration, with the hybrid schedule, rotary positions
    /// at `rotary_fraction`, GQA, QK-Norm, the configured norm kind and the
    /// per-language MoSME boxes on every `mosme_every`-th layer.
    pub fn to_lm_config(&self) -> anyhow::Result<LmConfig> {
        self.validate()?;
        let schedule = AttentionSchedule::parse(&self.attention, self.num_layers, 64, 8)?;
        let mosme = MosmeTrunkConfig::new(self.mosme_spec()?)
            .with_every_n_layers(self.mosme_every)
            .with_balance_bias(false);
        Ok(LmConfig {
            vocab_size: self.vocab_size,
            context: self.context,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            intermediate_size: self.intermediate_size,
            cond_hidden_size: self.cond_hidden_size,
            frequency_embedding_size: self.frequency_embedding_size,
            num_blocks: 1,
            ..LmConfig::default()
        }
        .with_attention(schedule)
        .with_positions(PositionKind::Rotary)
        .with_kv_heads(self.num_kv_heads)
        .with_rotary_fraction(self.rotary_fraction)
        .with_qk_norm(self.qk_norm)
        .with_norm_kind(self.norm_kind)
        .with_mosme(mosme))
    }

    /// The geometric stream's [`GeomConfig`] projection (only the fields a
    /// [`GeomBlock`] reads are load-bearing: `dim`, `hidden_size`, `slots`,
    /// `refine_steps`, the conditioning sizes, `layer_norm_eps` and
    /// `repulsion`).
    fn to_geom_config(&self) -> GeomConfig {
        GeomConfig {
            vocab_size: self.vocab_size,
            scene_len: 40,
            hidden_size: self.hidden_size,
            slots: self.geom_slots,
            dim: self.geom_dim,
            num_blocks: 1,
            refine_steps: self.geom_refine_steps,
            cond_hidden_size: self.geom_cond_hidden,
            frequency_embedding_size: self.geom_frequency,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            repulsion: self.geom_repulsion,
            sigma_data: 0.5,
            moe_boxes: 0,
            moe_experts: 1,
            moe_top_k: 1,
            answer_tokens: Vec::new(),
        }
    }

    /// `weight * out + out`: one biased linear map's parameter count.
    fn linear(in_features: usize, out_features: usize) -> usize {
        in_features * out_features + out_features
    }

    /// Parameter counts from the shapes alone, split by component so the
    /// accounting can be checked by hand. Returns `(trunk, geom_stream,
    /// probe_router, resident_experts)`. Call after `validate` (which
    /// `cost` does).
    fn shape_params(&self, spec: &MosmeSpec) -> (usize, usize, usize, usize) {
        let (h, c, i, l) = (
            self.hidden_size,
            self.cond_hidden_size,
            self.intermediate_size,
            self.num_layers,
        );
        let head_dim = h / self.num_heads;
        let kv_dim = self.num_kv_heads * head_dim;
        let k = spec.boxes.len();
        let e = self.experts_per_box;
        let router_in = c + h; // route_on_tokens: [cond, token]
        let expert_mlp = Self::linear(h, i) + Self::linear(i, h);
        let norm_params = match self.norm_kind {
            NormKind::Layer => 2 * h,
            NormKind::Rms => h,
        };
        let qk_norm_params = if self.qk_norm { 2 * head_dim } else { 0 };

        // Trunk, per layer: q/out full-width, k/v at the KV width, the two
        // mode-mixing logits, optional QK-Norm scales, two trunk norms, the
        // adaLN modulation.
        let attention = 2 * Self::linear(h, h) + 2 * Self::linear(h, kv_dim) + 2 + qk_norm_params;
        let adaln = Self::linear(c, 6 * h);
        let embedding = self.vocab_size * h;
        let time_embedder = Self::linear(self.frequency_embedding_size, c) + Self::linear(c, c);
        let sparse_ffn = Self::linear(router_in, k)
            + k * Self::linear(router_in, e)
            + k * e // the enabled masks are real tensors
            + k * e * expert_mlp;
        let mosme = MosmeTrunkConfig::new(spec.clone()).with_every_n_layers(self.mosme_every);
        let sparse_layers = mosme.num_hierarchical_layers(l);
        let trunk = embedding
            + time_embedder
            + l * (attention + 2 * norm_params + adaln)
            + sparse_layers * sparse_ffn
            + (l - sparse_layers) * expert_mlp
            + norm_params; // the final norm

        let (d, gc, gf) = (self.geom_dim, self.geom_cond_hidden, self.geom_frequency);
        let geom = Self::linear(h, d) // slot init
            + Self::linear(d, d) // queries
            + 2 * Self::linear(h, d) // keys and values
            + Self::linear(d, h) // writer_in
            + Self::linear(h, h) // writer_out
            + 2 * h // the block's LayerNorm
            + Self::linear(gf, gc)
            + Self::linear(gc, gc) // sigma embedder
            + 2 * Self::linear(gc, h) // FiLM scale and shift
            + d * d
            + d // the metric's Cholesky parameters
            + 2 // repulsion and attention-temperature logits
            + Self::linear(d, h); // the zero-init readout

        let probe = Self::linear(router_in, k) + k * Self::linear(router_in, e) + k * e;
        (trunk, geom, probe, sparse_layers * k * e * expert_mlp)
    }

    /// Counted cost of one token, from the shapes: active parameters (what a
    /// token actually touches — the trunk's cost model at `context`, plus the
    /// embedding, the geometric stream and the probe router) against the
    /// total the checkpoint holds (every resident expert included).
    pub fn cost(&self) -> anyhow::Result<StudentCost> {
        self.validate()?;
        let spec = self.mosme_spec()?;
        let trunk_cost = self.to_lm_config()?.cost(self.context)?.total();
        let (trunk_total, geom, probe, resident_experts) = self.shape_params(&spec);
        let embedding = self.vocab_size * self.hidden_size;
        Ok(StudentCost {
            active_params: trunk_cost.active_params + embedding + geom + probe,
            total_params: trunk_total + geom + probe,
            trunk_flops_per_token: trunk_cost.flops,
            geom_stream_params: geom,
            probe_router_params: probe,
            resident_expert_params: resident_experts,
        })
    }
}

/// What one token costs in a [`GeometricStudent`], counted from the shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentCost {
    /// Parameters a token actually touches: the trunk's counted active
    /// parameters (sparse experts at `top_box * top_expert`), plus the tied
    /// embedding, the geometric stream and the probe router.
    pub active_params: usize,
    /// Every parameter the checkpoint holds, resident experts included.
    pub total_params: usize,
    /// Trunk FLOPs per token at `context` (the cost model's multiply-adds
    /// times two; the geometric stream adds a small dense term on top).
    pub trunk_flops_per_token: f64,
    pub geom_stream_params: usize,
    pub probe_router_params: usize,
    /// Parameters of the resident specialists a token did *not* necessarily
    /// touch — the honesty half of the MoSME accounting.
    pub resident_expert_params: usize,
}

/// The router-which-knows readout: which specialist box would handle each
/// token, read from the gates alone — no expert MLP ran.
#[derive(Debug, Clone)]
pub struct LanguageRouting {
    /// Mean box gate over every token. A distribution: every gate row sums
    /// to 1, so their mean does too. The coding prefix is parallel to the
    /// model's [`StudentConfig::coding_boxes`]; any remaining entries are the
    /// configured field boxes, in [`StudentConfig::field_boxes`] order.
    pub traffic: Vec<f32>,
    /// Each token's most likely *language* specialist, flattened `[b * n]`:
    /// the argmax over the coding boxes, decoded through the model's
    /// [`StudentConfig::coding_boxes`]. Unchanged when field boxes exist —
    /// a field box is not a language, and this readout's callers
    /// (`crate::langdata::router_agreement`) speak [`CodeLanguage`].
    pub top_language: Vec<CodeLanguage>,
    /// Each token's most likely box over *all* boxes, coding and field,
    /// flattened `[b * n]`: the raw box-router argmax. Compare against
    /// [`BoxTarget::box_index`].
    pub top_box: Vec<usize>,
    /// The trunk's own per-sparse-layer routing statistics, in execution
    /// order (what actually fired inside the trunk pass).
    pub trunk_layers: Vec<crate::moe::RoutingStats>,
}

/// Output of a student pass.
#[derive(Debug)]
pub struct StudentForward<B: Backend> {
    /// `[b, n, vocab]`, from the trunk's tied readout of the
    /// geometrically-refined hidden states. The loss attach point.
    pub logits: Tensor<B, 3>,
    /// The residual stream after the geometric write-back, `[b, n, hidden]`
    /// (before the trunk's final norm).
    pub hidden: Tensor<B, 3>,
    /// The certified refinement trace: energy and displacement per
    /// relaxation step. The energy descends by construction.
    pub refine: RefineTrace,
    pub routing: LanguageRouting,
}

/// The geometric stream's states from [`GeometricStudent::slots_at_depth`]:
/// the relaxed slot states alongside the hidden states they were written
/// back into and the certified trace.
#[derive(Debug)]
pub struct SlotPass<B: Backend> {
    /// The residual stream after the geometric write-back, `[b, n, hidden]`
    /// (identical to `forward_depth`'s hidden for the same arguments).
    pub hidden: Tensor<B, 3>,
    /// The relaxed slot states `[b, geom_slots, geom_dim]` — the attractor
    /// the certified refinement landed on.
    pub slots: Tensor<B, 3>,
    /// The certified refinement trace: energy and displacement per step.
    pub refine: RefineTrace,
}

/// The specialist coding student. See the module docs for the architecture
/// and the wave-2 entry points.
#[derive(Module, Debug)]
pub struct GeometricStudent<B: Backend> {
    trunk: LanguageModel<B>,
    /// Chunk-pooled hidden states to the initial concept points.
    slot_init: Linear<B>,
    /// The certified geometric refinement (metric, geodesic attention,
    /// energy descent).
    geom: GeomBlock<B>,
    /// Geometric space back into the residual stream. **Zero-initialized**:
    /// a fresh student is exactly the plain trunk, so the stream starts as
    /// pure option value.
    readout: Linear<B>,
    /// The router-which-knows: gates over the per-language boxes, read from
    /// hidden states without running any expert.
    probe: HierarchicalRouter<B>,
    #[module(skip)]
    config: StudentConfig,
}

impl<B: Backend<FloatElem = f32>> GeometricStudent<B> {
    /// Build a student from `config`. Fallible because
    /// [`StudentConfig::validate`] and [`LanguageModel::new`] are: a bad
    /// architecture names the field at fault instead of panicking inside an
    /// initializer.
    pub fn new(config: &StudentConfig, device: &B::Device) -> anyhow::Result<Self> {
        config.validate()?;
        let trunk = LanguageModel::new(&config.to_lm_config()?, device)?;
        let spec = config.mosme_spec()?;
        Ok(Self {
            slot_init: LinearConfig::new(config.hidden_size, config.geom_dim)
                .with_bias(true)
                .init(device),
            geom: GeomBlock::new(&config.to_geom_config(), device),
            readout: LinearConfig::new(config.geom_dim, config.hidden_size)
                .with_bias(true)
                .with_initializer(Initializer::Zeros)
                .init(device),
            probe: HierarchicalRouter::new(
                &MosmeConfig::new(config.hidden_size, config.cond_hidden_size, spec),
                device,
            ),
            trunk,
            config: config.clone(),
        })
    }

    /// The language trunk, for the crate's existing training machinery.
    pub fn trunk(&self) -> &LanguageModel<B> {
        &self.trunk
    }

    pub fn config(&self) -> &StudentConfig {
        &self.config
    }

    /// The probe router, e.g. to enable or disable a specialist
    /// (`HierarchicalRouter::set_enabled`).
    pub fn probe_router(&self) -> &HierarchicalRouter<B> {
        &self.probe
    }

    /// The initial geometric state: the hidden states pooled into
    /// `geom_slots` contiguous chunks, each projected into the geometric
    /// space, so slot `k` starts near the region it was pooled from.
    fn init_slots(&self, h: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n, _] = h.dims();
        let slots = self.config.geom_slots;
        let mut pooled = Vec::with_capacity(slots);
        for k in 0..slots {
            let start = k * n / slots;
            let end = ((k + 1) * n / slots).max(start + 1).min(n);
            pooled.push(h.clone().narrow(1, start, end - start).mean_dim(1));
        }
        self.slot_init.forward(Tensor::cat(pooled, 1))
    }

    /// One geometric pass over the trunk's hidden states: pool to slots,
    /// certified relaxation, zero-init readout back into the residual
    /// stream. Returns the modified hidden states, the relaxed slot states
    /// and the certified trace.
    ///
    /// The block's *own* symbolic-stream write (`writer_out`, not
    /// zero-initialized) is deliberately discarded: using it would break the
    /// fresh-student-is-the-plain-trunk contract. The certified half of the
    /// refinement — the geometric trajectory and its energy trace — is what
    /// this stream keeps; the write into the hidden states is this module's
    /// own zero-initialized projection of the relaxed slots.
    fn geometric_pass(
        &self,
        h: &Tensor<B, 3>,
        refine_override: Option<usize>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, RefineTrace) {
        let z0 = self.init_slots(h);
        let (_, z, trace) = self.geom.refine(h.clone(), z0, 0.0, refine_override);
        let pooled = z.clone().mean_dim(1); // [b, 1, dim]
        let delta = self.readout.forward(pooled); // [b, 1, hidden]
        (h.clone() + delta, z, trace)
    }

    /// The gates over the specialist boxes for already-computed hidden
    /// states: box traffic and each token's top specialist, without running
    /// any expert MLP.
    fn route_languages_from_hidden(&self, h: &Tensor<B, 3>) -> anyhow::Result<LanguageRouting> {
        let [b, n, _] = h.dims();
        let device = h.device();
        // The trunk's conditioning is a pure function of timestep zero; for
        // the probe the conditioning carries no information the hidden states
        // do not, so a zero vector keeps the readout a function of the tokens.
        let cond = Tensor::<B, 2>::zeros([b, self.config.cond_hidden_size], &device);
        let input = self.probe.router_input(h, &cond);
        let gates = self.probe.route(input);
        let traffic: Vec<f32> = gates
            .box_traffic()
            .reshape([self.config.num_boxes()])
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let top_box: Vec<usize> = gates
            .box_gates
            .clone()
            .argmax(1)
            .reshape([b * n])
            .into_data()
            .convert::<i64>()
            .iter::<i64>()
            .map(|raw| {
                usize::try_from(raw).map_err(|_| {
                    anyhow::anyhow!("the box router produced a negative box index {raw}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // top_language stays a language readout: the argmax over the coding
        // prefix only, so a configured field box never makes it fail. The
        // decode goes through the config's box list, because the box order is
        // the config's, not the enum's.
        let coding_boxes = self.config.coding_boxes.len();
        let top_idx: Vec<i64> = gates
            .box_gates
            .clone()
            .narrow(1, 0, coding_boxes)
            .argmax(1)
            .reshape([b * n])
            .into_data()
            .convert::<i64>()
            .iter::<i64>()
            .collect();
        let mut top_language = Vec::with_capacity(top_idx.len());
        for raw in top_idx {
            let idx = usize::try_from(raw).map_err(|_| {
                anyhow::anyhow!("the box router produced a negative box index {raw}")
            })?;
            let name = self.config.coding_boxes.get(idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "the box router produced coding box {idx}, but the config names {} coding \
                     boxes",
                    coding_boxes
                )
            })?;
            top_language.push(CodeLanguage::parse(name).map_err(|_| {
                anyhow::anyhow!(
                    "coding box '{name}' is not a language this build knows, so the per-language \
                     readout cannot decode it"
                )
            })?);
        }
        Ok(LanguageRouting {
            traffic,
            top_language,
            top_box,
            trunk_layers: Vec::new(),
        })
    }

    /// Which language specialist would handle each token of `tokens`, read
    /// from the trunk's hidden states through the probe gates alone — no
    /// expert MLP runs.
    pub fn route_languages(&self, tokens: Tensor<B, 2, Int>) -> anyhow::Result<LanguageRouting> {
        let (out, states) =
            self.trunk
                .forward_span_states(tokens, 0..self.trunk.num_layers(), None);
        let hidden = states.last().cloned().ok_or_else(|| {
            anyhow::anyhow!("the trunk produced no hidden states for an empty span")
        })?;
        let mut routing = self.route_languages_from_hidden(&hidden)?;
        routing.trunk_layers = out
            .balance_loss
            .as_ref()
            .map_or_else(Vec::new, |aux| aux.to_host());
        Ok(routing)
    }

    /// Trunk pass, geometric refinement, readout: logits, the refined hidden
    /// states, the certified trace and the routing readout.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> anyhow::Result<StudentForward<B>> {
        self.forward_depth(tokens, None)
    }

    /// [`Self::forward`] with the refinement depth overridden — the
    /// test-time-compute knob: same weights, more or fewer certified
    /// relaxation steps.
    pub fn forward_depth(
        &self,
        tokens: Tensor<B, 2, Int>,
        refine_override: Option<usize>,
    ) -> anyhow::Result<StudentForward<B>> {
        let (out, states) =
            self.trunk
                .forward_span_states(tokens, 0..self.trunk.num_layers(), None);
        let hidden = states.last().cloned().ok_or_else(|| {
            anyhow::anyhow!("the trunk produced no hidden states for an empty span")
        })?;
        let (hidden, _, refine) = self.geometric_pass(&hidden, refine_override);
        let logits = self.trunk.logits_from_hidden(hidden.clone());
        let mut routing = self.route_languages_from_hidden(&hidden)?;
        routing.trunk_layers = out
            .balance_loss
            .as_ref()
            .map_or_else(Vec::new, |aux| aux.to_host());
        Ok(StudentForward {
            logits,
            hidden,
            refine,
            routing,
        })
    }

    /// The geometric stream's states at a refinement depth: the same
    /// computation as [`Self::forward_depth`], but returning the relaxed slot
    /// states instead of the logits and routing. The vecread lens
    /// ([`crate::vecread`]) reads the slot geometry from here.
    pub fn slots_at_depth(
        &self,
        tokens: Tensor<B, 2, Int>,
        refine_override: Option<usize>,
    ) -> anyhow::Result<SlotPass<B>> {
        let (_, states) = self
            .trunk
            .forward_span_states(tokens, 0..self.trunk.num_layers(), None);
        let hidden = states.last().cloned().ok_or_else(|| {
            anyhow::anyhow!("the trunk produced no hidden states for an empty span")
        })?;
        let (hidden, slots, refine) = self.geometric_pass(&hidden, refine_override);
        Ok(SlotPass {
            hidden,
            slots,
            refine,
        })
    }

    /// The geometric stream's learned metric field, for metric-native
    /// diagnostics (the vecread slot-geometry section computes pairwise slot
    /// distances through it).
    pub(crate) fn geom_metric(&self) -> &crate::geometry::MetricField<B> {
        self.geom.metric_field()
    }

    /// Validation hook for the vecread causal certificates: replace the
    /// zero-initialized geometric readout with a given projection. A fresh
    /// student's stream is an exact identity by design (the module docs), so
    /// a refinement-depth intervention is invisible to the lens at init; the
    /// certificates install a non-zero readout through here and say so.
    pub(crate) fn set_geom_readout(&mut self, readout: Linear<B>) {
        self.readout = readout;
    }

    /// One per-language training step: next-token CE on the completion
    /// tokens only, plus the router-which-knows auxiliary.
    ///
    /// - `tokens` is `[b, n]` and `mask` is `[b, n]` aligned with it: 1.0
    ///   where the position is a supervised completion target (a
    ///   [`LanguageBatch`] mask). Padding is excluded twice over — by the
    ///   mask and by the `<pad>` target check — mirroring the masked path
    ///   of `crate::lm`'s objective.
    /// - `languages` is the known language of each row (`None` = unknown);
    ///   it supervises the probe router's box distribution on the trunk's
    ///   hidden states — the same hidden states [`Self::route_languages`]
    ///   reads, so the aux trains exactly the readout it is measured by.
    ///   A language the config's `coding_boxes` does not name supervises no
    ///   box, exactly like an unknown language.
    ///   Supervision lands on the completion tokens, where the language
    ///   signal is; the shared prompt would only teach the router to guess.
    ///
    /// This is the coding-language half of the objective; rows whose signal
    /// is a *field* box (`field:pentest`, not a language) go through
    /// [`Self::specialist_step_boxes`], of which this is the thin wrapper.
    ///
    /// The trunk's MoSME balance and z terms join the loss at the same
    /// weights `crate::lm` uses (0.01 and 1.0). The geometric stream trains
    /// jointly through the completion CE; its energy trace is returned as a
    /// diagnostic, not a loss.
    pub fn specialist_step(
        &self,
        tokens: Tensor<B, 2, Int>,
        mask: Tensor<B, 2>,
        languages: &[Option<CodeLanguage>],
        router_aux_weight: f64,
    ) -> anyhow::Result<SpecialistStep<B>> {
        let targets: Vec<Option<BoxTarget>> = languages
            .iter()
            .map(|row| self.config.box_target(*row, None))
            .collect();
        self.specialist_step_boxes(tokens, mask, &targets, router_aux_weight)
    }

    /// [`Self::specialist_step`] with the per-row routing target spelled out
    /// as a [`BoxTarget`]: a coding-language box or a configured field box.
    /// `None` supervises no box, exactly like an unknown language.
    pub fn specialist_step_boxes(
        &self,
        tokens: Tensor<B, 2, Int>,
        mask: Tensor<B, 2>,
        targets: &[Option<BoxTarget>],
        router_aux_weight: f64,
    ) -> anyhow::Result<SpecialistStep<B>> {
        let [b, n] = tokens.dims();
        anyhow::ensure!(n >= 2, "next-token loss needs at least two positions");
        anyhow::ensure!(
            mask.dims() == [b, n],
            "the loss mask must be shaped like the tokens"
        );
        anyhow::ensure!(
            targets.len() == b,
            "targets holds {} rows for a batch of {b}",
            targets.len()
        );
        let device = tokens.device();
        let (out, states) =
            self.trunk
                .forward_span_states(tokens.clone(), 0..self.trunk.num_layers(), None);
        let trunk_hidden = states.last().cloned().ok_or_else(|| {
            anyhow::anyhow!("the trunk produced no hidden states for an empty span")
        })?;
        let (hidden, _, refine) = self.geometric_pass(&trunk_hidden, None);
        let logits = self.trunk.logits_from_hidden(hidden);

        // --- masked completion CE (the masked path of lm::objective) ------
        let flat_logits = logits
            .narrow(1, 0, n - 1)
            .reshape([b * (n - 1), self.config.vocab_size]);
        let next_tokens = tokens.clone().narrow(1, 1, n - 1).reshape([b * (n - 1), 1]);
        let log_probs = log_softmax(flat_logits, 1);
        let nll = -log_probs
            .gather(1, next_tokens.clone())
            .reshape([b * (n - 1)]);
        let pad = Tensor::<B, 1, Int>::full([b * (n - 1)], i64::from(Special::Pad.id()), &device);
        let pad_keep = next_tokens
            .reshape([b * (n - 1)])
            .equal(pad)
            .bool_not()
            .float();
        let completion_keep = mask
            .clone()
            .narrow(1, 1, n - 1)
            .reshape([b * (n - 1)])
            .clamp_min(0.0)
            * pad_keep;
        let counted_raw = completion_keep.clone().sum();
        let ce = (nll * completion_keep).sum() / counted_raw.clone().clamp_min(1.0);
        let tokens_counted = counted_raw.into_scalar() as usize;
        let completion_ce: f32 = ce.clone().into_scalar();

        // --- the router-which-knows auxiliary -----------------------------
        let any_target = targets.iter().any(Option::is_some);
        let mut loss = ce;
        let (mut router_aux_value, mut target_traffic, mut target_top1) =
            (f32::NAN, f32::NAN, f32::NAN);
        if any_target {
            let cond = Tensor::<B, 2>::zeros([b, self.config.cond_hidden_size], &device);
            let input = self.probe.router_input(&trunk_hidden, &cond);
            let gates = self.probe.route(input);
            let mut target_flat = Vec::with_capacity(b * n);
            let mut has_flat = Vec::with_capacity(b * n);
            for row in targets {
                for _ in 0..n {
                    target_flat.push(row.map_or(0, BoxTarget::box_index) as i64);
                    has_flat.push(if row.is_some() { 1.0f32 } else { 0.0 });
                }
            }
            let target =
                Tensor::<B, 1, Int>::from_ints(target_flat.as_slice(), &device).reshape([b * n, 1]);
            let has = Tensor::<B, 1>::from_floats(has_flat.as_slice(), &device).reshape([b * n]);
            // Supervise where the specialist signal actually is: the
            // completion tokens. The prompt is shared across boxes, so
            // supervising its tokens toward one box would teach the router
            // to guess; padding carries no signal at all.
            let aux_keep = mask.clone().reshape([b * n]).clamp_min(0.0) * has;
            let aux_counted = aux_keep.clone().sum().clamp_min(1.0);
            let box_log_probs = log_softmax(gates.box_logits.clone(), 1);
            let aux_nll = -box_log_probs.gather(1, target.clone()).reshape([b * n]);
            let router_aux = (aux_nll * aux_keep.clone()).sum() / aux_counted.clone();
            router_aux_value = router_aux.clone().into_scalar();
            // The readouts the proof reports: soft mass on the true box, and
            // the top-1 agreement, both over the supervised tokens only.
            let true_prob = gates
                .box_probs
                .clone()
                .gather(1, target.clone())
                .reshape([b * n]);
            target_traffic =
                ((true_prob * aux_keep.clone()).sum() / aux_counted.clone()).into_scalar();
            let top1_hit = gates
                .box_probs
                .argmax(1)
                .reshape([b * n])
                .equal(target.reshape([b * n]))
                .float();
            target_top1 = ((top1_hit * aux_keep).sum() / aux_counted).into_scalar();
            if router_aux_weight > 0.0 {
                loss = loss + router_aux.mul_scalar(router_aux_weight as f32);
            }
        }

        // The trunk's own routing regularizers, at the weights the LM
        // objective uses: the balance term scaled, the z-loss at weight 1.
        let mut balance = 0.0f32;
        let loss = match out.balance_loss {
            Some(aux) => {
                balance = aux.balance.clone().into_scalar();
                loss + aux.balance.mul_scalar(0.01) + aux.z
            }
            None => loss,
        };
        Ok(SpecialistStep {
            loss,
            completion_ce,
            router_aux: router_aux_value,
            target_traffic,
            target_top1,
            balance,
            tokens_counted,
            energy: refine.energy,
        })
    }
}

// ------------------------------------------------- per-language training --
//
// Wave 2: the traces a teacher writes are *text* (one
// `{id, language, prompt, completion, source, ...}` object per JSONL line),
// not logits, so the specialist objective is SFT on the filtered traces —
// next-token cross-entropy on the completion tokens only. Nothing is
// distilled here because there is nothing to distil from: when teacher
// logits do exist, the trunk's own `Distillation` extra
// (`crate::lm::LmExtras::distill`, driven by `dblocks lm train`) is the KL
// path, and it is deliberately not faked in this module.
//
// The second half of the objective is what makes the router *know which
// model to call*: each example's known language supervises the probe
// router's box distribution (`routing_aux_weight` x CE against the one-hot
// language box), so `route_languages` learns to name a token's specialist
// without running any expert. The geometric stream trains jointly through
// the completion CE; its certified energy trace is reported as a
// diagnostic only — no energy objective exists in `crate::geometry` for
// the refinement (the `GeomObjective` variants are answer/diffusion/
// consistency losses of the standalone reasoner), so none is invented.

/// One tokenized trace: `<bos> prompt completion <eos>`, truncated at
/// `context` and padded with `<pad>`, with the next-token loss mask that
/// restricts supervision to the completion (and its closing `<eos>`).
#[derive(Debug, Clone, PartialEq)]
struct TokenizedTrace {
    tokens: Vec<u16>,
    mask: Vec<f32>,
    language: Option<CodeLanguage>,
    /// The raw `"field"` string, when the line carried a non-empty one.
    /// Resolution to a configured field box happens against the training
    /// config ([`StudentConfig::box_target`]), not here: the batch is data,
    /// and which fields have boxes is the model's business.
    field: Option<String>,
}

/// A per-language training set loaded from a traces JSONL.
///
/// Two on-disk shapes are read: the trace schema (`prompt` + `completion`,
/// with a `language` key) and the older SFT schema (`instruction` +
/// `response`, with `lang`), whose lines get `language = unknown`. A line
/// whose language names no shipped box (e.g. `go`, a *planned* registry
/// entry) is likewise unknown: it can train the trunk in an unfiltered run
/// but cannot supervise a routing box. Either shape may also carry a
/// `"field"` string (the NINA pair-file contract): it is kept verbatim and
/// supervises a configured field box at training time — or nothing, when
/// the field is unknown, the same rule as an unknown language. The load is
/// deterministic — the example order is a seeded ChaCha shuffle, the same
/// RNG discipline `crate::train` keeps — so two loads of the same file at
/// the same seed agree bit for bit, and a resumed run sees the data the
/// first run saw.
#[derive(Debug, Clone)]
pub struct LanguageBatch {
    examples: Vec<TokenizedTrace>,
    /// The filter the load applied; `None` kept every language.
    filter: Option<CodeLanguage>,
    context: usize,
    path: PathBuf,
    lines_read: usize,
    /// Examples dropped because the prompt alone filled the context, leaving
    /// no supervised completion token.
    skipped_no_target: usize,
    /// `(raw language string or "unknown", count)` over every line read,
    /// before filtering — what the "no examples" error reports.
    languages_seen: Vec<(String, usize)>,
    /// What the cheat gate removed, kept on the batch rather than printed and
    /// discarded: a drop count that only existed on the terminal is a count
    /// nobody can check against the file.
    cheat_scan: crate::cheat::ScanSummary,
}

impl LanguageBatch {
    /// Load `path`, keeping only examples in `filter`'s language (`None`
    /// keeps everything, unknown-language lines included), tokenizing
    /// `prompt + completion`, truncating at `context`, and shuffling
    /// deterministically from `shuffle_seed`.
    ///
    /// Loud by contract: a missing file, a malformed line and an empty
    /// result are all errors naming the path — a specialist trained on the
    /// wrong data is worse than no specialist.
    pub fn load(
        path: &Path,
        filter: Option<CodeLanguage>,
        context: usize,
        shuffle_seed: u64,
    ) -> anyhow::Result<Self> {
        Self::load_gated(
            path,
            filter,
            context,
            shuffle_seed,
            &crate::cheat::CheatGate::patterns_only(),
        )
    }

    /// [`Self::load`] under an explicit [`crate::cheat::CheatGate`]: a
    /// calibrated language-mixing profile, the learned layer, or both.
    ///
    /// The gate is not opt-in. Every text training path in this crate reaches
    /// its rows through this function, so gating *here* is what makes the
    /// filter impossible to route around — a caller that forgets to scan gets
    /// the patterns anyway, because there is no path to the tokenizer that
    /// does not pass through them.
    pub fn load_gated(
        path: &Path,
        filter: Option<CodeLanguage>,
        context: usize,
        shuffle_seed: u64,
        gate: &crate::cheat::CheatGate,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            context >= 2,
            "context must cover at least 2 positions, got {context}"
        );
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read traces {}", path.display()))?;
        let tokenizer = ByteTokenizer::new();
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        let mut examples = Vec::new();
        let mut lines_read = 0usize;
        let mut skipped_no_target = 0usize;
        let mut cheat_scan = crate::cheat::ScanSummary::default();
        for (idx, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            lines_read += 1;
            let line_no = idx + 1;
            let value: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("{} line {line_no} is not valid JSON", path.display()))?;
            let get = |key: &str| value.get(key).and_then(|v| v.as_str());
            let prompt = get("prompt").or_else(|| get("instruction"));
            let completion = get("completion").or_else(|| get("response"));
            let (Some(prompt), Some(completion)) = (prompt, completion) else {
                anyhow::bail!(
                    "{} line {line_no} has no prompt/completion (or instruction/response) pair",
                    path.display()
                );
            };
            let raw_language = get("language")
                .or_else(|| get("lang"))
                .map(str::trim)
                .filter(|s| !s.is_empty());
            // A language this build does not know (a planned registry entry,
            // a typo, "unknown") supervises no routing box: it is unknown.
            // Whether a *known* language's box is supervised is the config's
            // call (`StudentConfig::box_target`), not the loader's.
            let language = raw_language.and_then(|raw| CodeLanguage::parse(raw).ok());
            // The NINA pair-file contract's optional field key, kept raw;
            // an unknown field supervises nothing, decided at the config.
            let field = get("field")
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            *seen
                .entry(raw_language.unwrap_or("unknown").to_string())
                .or_default() += 1;
            if let Some(want) = filter {
                if language != Some(want) {
                    continue;
                }
            }

            // The gate reads the completion — the training signal, and the
            // same unit `langdata` filters at build time. The prompt is not
            // scanned: real task statements enumerate the forbidden idioms
            // ("no `todo!()`") and show fenced buggy code, and a scanner
            // reading instructions as code false-positives on every one.
            //
            // Only rows this run will actually train on are gated: a row the
            // language filter excludes never reaches the tokenizer, and
            // scanning it would only inflate the counts with rows whose fate
            // was already decided.
            let snippet_lang = match language {
                Some(lang) => crate::langdata::snippet_language(lang),
                None => crate::cheat::SnippetLanguage::Other,
            };
            let offending = gate.inspect(Some(prompt), completion, snippet_lang, &mut cheat_scan);
            cheat_scan.observe(&offending);
            if !offending.is_empty() {
                continue;
            }

            let prompt_tokens = tokenizer.encode(prompt);
            let completion_tokens = tokenizer.encode(completion);
            let mut tokens = Vec::with_capacity(
                (2 + prompt_tokens.len() + completion_tokens.len()).min(context),
            );
            let mut mask = Vec::with_capacity(tokens.capacity());
            tokens.push(Special::Bos.id());
            mask.push(0.0);
            tokens.extend_from_slice(&prompt_tokens);
            mask.extend(std::iter::repeat_n(0.0, prompt_tokens.len()));
            tokens.extend_from_slice(&completion_tokens);
            mask.extend(std::iter::repeat_n(1.0, completion_tokens.len()));
            tokens.push(Special::Eos.id());
            mask.push(1.0);
            tokens.truncate(context);
            mask.truncate(context);
            if !mask.iter().any(|m| *m > 0.0) {
                // The prompt alone filled the context: this example carries
                // no completion signal, so keeping it would teach nothing.
                skipped_no_target += 1;
                continue;
            }
            tokens.resize(context, Special::Pad.id());
            mask.resize(context, 0.0);
            examples.push(TokenizedTrace {
                tokens,
                mask,
                language,
                field,
            });
        }
        let languages_seen: Vec<(String, usize)> = seen.into_iter().collect();
        // The gate emptied the file. Naming the gate's own reasons is the
        // point: "no usable examples" would send whoever reads it looking at
        // the tokenizer, when the rows were removed before it ever saw one.
        if cheat_scan.scanned > 0 && cheat_scan.kept == 0 {
            return Err(cheat_scan.refusal(path));
        }
        anyhow::ensure!(
            !examples.is_empty(),
            "{} holds no usable {}examples ({} lines read, {} skipped with no supervised \
             completion token, {} dropped by the cheat gate; languages seen: {})",
            path.display(),
            filter.map_or(String::new(), |l| format!("'{}' ", l.name())),
            lines_read,
            skipped_no_target,
            cheat_scan.dropped,
            languages_seen
                .iter()
                .map(|(l, n)| format!("{l}:{n}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        // The same shuffle the uninterrupted and the resumed run both see:
        // seeded ChaCha, so the order is a pure function of (file, seed).
        let mut rng = ChaCha12Rng::seed_from_u64(shuffle_seed);
        examples.shuffle(&mut rng);
        Ok(Self {
            examples,
            filter,
            context,
            path: path.to_path_buf(),
            lines_read,
            skipped_no_target,
            languages_seen,
            cheat_scan,
        })
    }

    /// What the cheat gate removed from this batch, and why. `None` never:
    /// [`Self::load`] always runs the patterns, so a batch always has a
    /// summary — it is the *count* that can be zero, and a zero count on a
    /// summary that exists is the claim "nothing was removed", which is what
    /// makes the absence of a drop visible rather than assumed.
    pub fn cheat_scan(&self) -> &crate::cheat::ScanSummary {
        &self.cheat_scan
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// The sequence length every example is padded/truncated to.
    pub fn context(&self) -> usize {
        self.context
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The filter the load applied (`None` = every language kept).
    pub fn filter(&self) -> Option<CodeLanguage> {
        self.filter
    }

    pub fn lines_read(&self) -> usize {
        self.lines_read
    }

    pub fn skipped_no_target(&self) -> usize {
        self.skipped_no_target
    }

    /// `(raw language string or "unknown", count)` over the lines read.
    pub fn languages_seen(&self) -> &[(String, usize)] {
        &self.languages_seen
    }

    /// Token ids of example `idx`, `context` long.
    pub fn tokens(&self, idx: usize) -> &[u16] {
        &self.examples[idx].tokens
    }

    /// Loss mask of example `idx`: 1.0 where the position is a supervised
    /// completion target, aligned with [`Self::tokens`].
    pub fn mask(&self, idx: usize) -> &[f32] {
        &self.examples[idx].mask
    }

    /// The example's language; `None` when the trace named no shipped box.
    pub fn language_of(&self, idx: usize) -> Option<CodeLanguage> {
        self.examples[idx].language
    }

    /// The example's raw `"field"` string; `None` when the line carried
    /// none. Whether the field supervises a box is decided against the
    /// training config ([`StudentConfig::box_target`]).
    pub fn field_of(&self, idx: usize) -> Option<&str> {
        self.examples[idx].field.as_deref()
    }

    /// What a run trained on, hashed once, for the training state.
    pub fn identity(&self) -> anyhow::Result<DatasetIdentity> {
        DatasetIdentity::of_path(
            &self.path,
            format!(
                "language traces {} (filter {})",
                self.path.display(),
                self.filter
                    .map_or("none".to_string(), |l| l.name().to_string())
            ),
        )
    }
}

/// What one [`GeometricStudent::specialist_step`] measured.
#[derive(Debug)]
pub struct SpecialistStep<B: Backend> {
    /// The backward root: masked completion CE, plus the trunk's MoSME
    /// balance (0.01) and z (1.0) terms exactly as `crate::lm`'s objective
    /// adds them, plus `router_aux_weight` x the routing auxiliary.
    pub loss: Tensor<B, 1>,
    /// Masked next-token CE on completion tokens only.
    pub completion_ce: f32,
    /// The routing auxiliary before its weight (CE of the probe's box
    /// softmax against the known box — a language's or a field's); NaN when
    /// no example in the batch carried a target, which is how "the aux is
    /// off" is spelled.
    pub router_aux: f32,
    /// Mean probe box probability on the true box over the supervised
    /// tokens; NaN alongside `router_aux`.
    pub target_traffic: f32,
    /// Fraction of the supervised tokens whose top-1 box *is* the true
    /// box — the router-which-knows readout (chance is 1 over the box
    /// count: 1/6 without field boxes).
    pub target_top1: f32,
    /// The trunk's MoSME balance loss, inside `loss` at weight 0.01.
    pub balance: f32,
    /// Supervised (completion, non-pad) target tokens in the batch.
    pub tokens_counted: usize,
    /// The geometric stream's certified energy trace — a diagnostic of the
    /// descent the refinement certified, not a term of the loss.
    pub energy: Vec<f32>,
}

/// Configuration for [`train_language_specialist`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistTrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    /// Rescale the gradient above this global norm; 0.0 disables exactly.
    #[serde(default)]
    pub clip_norm: f32,
    /// Weight on the router-which-knows auxiliary (the known language's CE
    /// against the probe's box distribution). 0.0 disables it exactly.
    #[serde(default = "default_router_aux_weight")]
    pub router_aux_weight: f64,
    pub seed: u64,
    /// Print every this many steps; 0 is quiet.
    pub log_every: usize,
    /// Where checkpoints (model + training state) go; `None` for nowhere.
    #[serde(default)]
    pub out_dir: Option<PathBuf>,
    /// Also checkpoint every this many steps (0 = only at the end).
    #[serde(default)]
    pub checkpoint_every: usize,
    /// A model file from an earlier run; its training state is restored and
    /// verified, so the continuation is exact.
    #[serde(default)]
    pub resume: Option<PathBuf>,
    /// Which specialist this run trains (`None` = unfiltered or
    /// unknown-language data). Recorded in the training state, so a resumed
    /// run cannot silently continue as a different specialist.
    #[serde(default)]
    pub language: Option<CodeLanguage>,
    /// Micro-batch size for gradient accumulation: each optimizer step draws
    /// `batch_size` examples as always, but runs them as
    /// `batch_size / micro_batch` sequential forward/backward passes and sums
    /// the gradients, so peak activation memory is one micro-batch's. Each
    /// micro-batch's loss is scaled by its share of the batch's supervised
    /// completion tokens, which makes the completion-CE gradient exactly the
    /// full batch's; the router auxiliary and the trunk's balance/z terms
    /// (per-micro-batch means) are weighted by the same share rather than
    /// flat-averaged — an approximation of the full-batch objective,
    /// documented here because it is one. 0 (the default) means
    /// `micro_batch == batch_size`: the direct single-pass path, bit-stable
    /// for every run written before this field existed.
    #[serde(default)]
    pub micro_batch: usize,
}

fn default_weight_decay() -> f64 {
    0.01
}

fn default_router_aux_weight() -> f64 {
    0.1
}

impl Default for SpecialistTrainConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch_size: 8,
            lr: 3e-4,
            weight_decay: default_weight_decay(),
            clip_norm: 0.0,
            router_aux_weight: default_router_aux_weight(),
            seed: 42,
            log_every: 10,
            out_dir: None,
            checkpoint_every: 0,
            resume: None,
            language: None,
            micro_batch: 0,
        }
    }
}

/// What a specialist run did.
#[derive(Debug, Clone)]
pub struct SpecialistTrainReport {
    pub steps_taken: usize,
    /// Steps discarded for a non-finite loss.
    pub steps_skipped: usize,
    /// Optimizer steps whose gradient was rescaled by `clip_norm`.
    pub steps_clipped: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub mean_loss: f32,
    /// Masked completion CE at the first and last step taken.
    pub first_ce: f32,
    pub last_ce: f32,
    /// Routing auxiliary at the first and last step (NaN when off).
    pub first_router_aux: f32,
    pub last_router_aux: f32,
    /// Mean probe box probability on the true box, first and last step.
    pub first_target_traffic: f32,
    pub last_target_traffic: f32,
    /// Router-which-knows: fraction of supervised tokens whose top-1 box is
    /// the true box, at the first and last step. Chance is 1 over the box
    /// count (1/6 without field boxes).
    pub first_target_top1: f32,
    pub last_target_top1: f32,
    pub tokens_seen: usize,
    pub elapsed_secs: f64,
    /// The final checkpoint, when `out_dir` was set.
    pub checkpoint: Option<PathBuf>,
    /// The step a resumed run continued from; 0 for a fresh run.
    pub resumed_from_step: usize,
    /// Every periodic checkpoint written, as `(steps completed, model path)`.
    pub periodic_checkpoints: Vec<(usize, PathBuf)>,
}

/// Host-side specialist trainer state saved beside the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StudentExtras {
    steps_taken: usize,
    steps_skipped: usize,
    steps_clipped: usize,
    first_loss: f32,
    first_ce: f32,
    first_router_aux: f32,
    first_target_traffic: f32,
    first_target_top1: f32,
    tokens_seen: usize,
    loss_sum: f64,
    elapsed_secs: f64,
}

/// Train the student on a [`LanguageBatch`]: masked completion CE plus the
/// router-which-knows auxiliary, jointly over the trunk, the geometric
/// stream and the probe router.
///
/// The RNG discipline is `crate::train`'s: the host ChaCha (serialized into
/// every checkpoint) draws the batch, and the device stream is reseeded
/// from `(seed, step)` at the top of every step, so a resumed run is
/// bit-identical to the uninterrupted one.
pub fn train_language_specialist<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>>(
    mut model: GeometricStudent<B>,
    data: &LanguageBatch,
    config: &SpecialistTrainConfig,
    device: &B::Device,
) -> anyhow::Result<(GeometricStudent<B>, SpecialistTrainReport)> {
    anyhow::ensure!(config.steps > 0, "steps must be positive");
    anyhow::ensure!(config.batch_size > 0, "batch_size must be positive");
    let micro_batch = if config.micro_batch == 0 {
        config.batch_size
    } else {
        config.micro_batch
    };
    anyhow::ensure!(
        config.batch_size % micro_batch == 0,
        "micro_batch ({micro_batch}) must divide batch_size ({}) evenly",
        config.batch_size
    );
    anyhow::ensure!(!data.is_empty(), "the language batch holds no examples");
    anyhow::ensure!(
        data.context() >= 2,
        "a context of {} has no target position",
        data.context()
    );
    let dataset_identity = data.identity()?;
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let mut rng = ChaCha12Rng::seed_from_u64(config.seed);

    // The training state records the shape of the run — which specialist,
    // which architecture, which context — so a resume cannot silently
    // continue as something else.
    let mut config_json =
        serde_json::to_value(config).context("serialize specialist training config")?;
    config_json["student_config"] =
        serde_json::to_value(model.config()).context("serialize student config")?;
    config_json["trace_context"] =
        serde_json::to_value(data.context()).context("serialize trace context")?;

    let mut restored: Option<TrainState> = None;
    if let Some(path) = &config.resume {
        let state = TrainState::for_model(path)?.ok_or_else(|| {
            anyhow::anyhow!(
                "{} has no training state beside it; a student resume restores the optimizer \
                 and RNG from that state, so start a fresh run instead",
                path.display()
            )
        })?;
        let dir = TrainState::dir_for(path);
        anyhow::ensure!(
            state.kind == "student",
            "{} holds `{}` training state, not a student run",
            dir.display(),
            state.kind
        );
        state.verify_files(&dir)?;
        anyhow::ensure!(
            checkpoint::same_datasets(&state.datasets, std::slice::from_ref(&dataset_identity)),
            "refusing to resume: the checkpoint was trained on [{}], this run opened [{}]",
            checkpoint::describe_datasets(&state.datasets),
            checkpoint::describe_datasets(std::slice::from_ref(&dataset_identity))
        );
        for key in ["language", "student_config", "trace_context"] {
            anyhow::ensure!(
                state.config.get(key) == config_json.get(key),
                "refusing to resume: {key} differs between the checkpoint and this run"
            );
        }
        let differences = state.config_differences(&config_json);
        if !differences.is_empty() {
            println!(
                "warning: resuming with a different configuration in {}",
                differences.join(", ")
            );
        }
        println!(
            "resumed from {} at step {} (training state verified)",
            path.display(),
            state.step
        );
        model = checkpoint::load::<B, _>(model, path, device)?;
        restored = Some(state);
    }

    let started = std::time::Instant::now();
    let mut report = SpecialistTrainReport {
        steps_taken: 0,
        steps_skipped: 0,
        steps_clipped: 0,
        first_loss: f32::NAN,
        last_loss: f32::NAN,
        mean_loss: 0.0,
        first_ce: f32::NAN,
        last_ce: f32::NAN,
        first_router_aux: f32::NAN,
        last_router_aux: f32::NAN,
        first_target_traffic: f32::NAN,
        last_target_traffic: f32::NAN,
        first_target_top1: f32::NAN,
        last_target_top1: f32::NAN,
        tokens_seen: 0,
        elapsed_secs: 0.0,
        checkpoint: None,
        resumed_from_step: 0,
        periodic_checkpoints: Vec::new(),
    };
    let mut loss_sum = 0.0f64;
    let mut elapsed_before = 0.0f64;
    let mut start_step = 0usize;
    if let Some(state) = &restored {
        let dir = TrainState::dir_for(config.resume.as_ref().ok_or_else(|| {
            anyhow::anyhow!("a training state was restored without a resume path")
        })?);
        if let Some(file) = &state.optimizer {
            let record = checkpoint::load_record::<B, _>(&dir, file, device)?;
            optim = optim.load_record(record);
        }
        let extras: StudentExtras =
            serde_json::from_value(state.extras.clone()).context("parse student training state")?;
        report.steps_taken = extras.steps_taken;
        report.steps_skipped = extras.steps_skipped;
        report.steps_clipped = extras.steps_clipped;
        report.first_loss = extras.first_loss;
        report.first_ce = extras.first_ce;
        report.first_router_aux = extras.first_router_aux;
        report.first_target_traffic = extras.first_target_traffic;
        report.first_target_top1 = extras.first_target_top1;
        report.tokens_seen = extras.tokens_seen;
        loss_sum = extras.loss_sum;
        elapsed_before = extras.elapsed_secs;
        rng = serde_json::from_value(state.host_rng.clone()).context("restore host RNG")?;
        start_step = state.step;
        report.resumed_from_step = start_step;
    }
    anyhow::ensure!(
        start_step < config.steps,
        "the checkpoint is already at step {start_step}; ask for more with --steps"
    );

    let save_state = |step: usize,
                      model: &GeometricStudent<B>,
                      optim: &burn::optim::adaptor::OptimizerAdaptor<
        burn::optim::AdamW,
        GeometricStudent<B>,
        B,
    >,
                      rng: &ChaCha12Rng,
                      report: &SpecialistTrainReport,
                      loss_sum: f64,
                      elapsed: f64|
     -> anyhow::Result<Option<PathBuf>> {
        let Some(dir) = &config.out_dir else {
            return Ok(None);
        };
        let model_path = checkpoint::save_content_addressed(model.clone(), dir, "student")?;
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
        let extras = StudentExtras {
            steps_taken: report.steps_taken,
            steps_skipped: report.steps_skipped,
            steps_clipped: report.steps_clipped,
            first_loss: report.first_loss,
            first_ce: report.first_ce,
            first_router_aux: report.first_router_aux,
            first_target_traffic: report.first_target_traffic,
            first_target_top1: report.first_target_top1,
            tokens_seen: report.tokens_seen,
            loss_sum,
            elapsed_secs: elapsed,
        };
        let state = TrainState {
            format_version: checkpoint::STATE_FORMAT_VERSION,
            kind: "student".into(),
            step,
            seed: config.seed,
            host_rng: serde_json::to_value(rng).context("serialize host RNG")?,
            config: config_json.clone(),
            build: checkpoint::BuildInfo::current(),
            datasets: vec![dataset_identity.clone()],
            model: checkpoint::model_entry(&model_path)?,
            optimizer,
            ema: None,
            head: None,
            head_optimizer: None,
            extras: serde_json::to_value(&extras).context("serialize student training extras")?,
            saved_unix_secs: checkpoint::unix_now(),
        };
        state.write(&state_dir)?;
        Ok(Some(model_path))
    };

    let trainable = TrainableSet::all();
    let context = data.context();
    for step in start_step..config.steps {
        // The device stream is a pure function of (seed, step): see
        // `crate::train::step_seed`.
        <B as burn::tensor::backend::Backend>::seed(
            device,
            crate::train::step_seed(config.seed, step),
        );
        let mut flat = Vec::with_capacity(config.batch_size * context);
        let mut mask_flat = Vec::with_capacity(config.batch_size * context);
        let mut targets = Vec::with_capacity(config.batch_size);
        for _ in 0..config.batch_size {
            // With-replacement draws, exactly as CorpusMix samples windows:
            // stateless in everything but the serialized host RNG, which is
            // what makes the resumed run draw what the uninterrupted one did.
            let idx = rng.random_range(0..data.len());
            flat.extend(data.tokens(idx).iter().map(|t| i64::from(*t)));
            mask_flat.extend_from_slice(data.mask(idx));
            targets.push(
                model
                    .config()
                    .box_target(data.language_of(idx), data.field_of(idx)),
            );
        }
        // The per-step observables the report and the log line read. With one
        // micro-batch these are the step's own numbers; with several they are
        // the token-share-weighted means over the micro-batches that ran.
        struct StepObs {
            loss: f32,
            ce: f32,
            router_aux: f32,
            target_traffic: f32,
            target_top1: f32,
            balance: f32,
            tokens_counted: usize,
            energy: Vec<f32>,
        }

        let (loss_value, obs, grads) = if micro_batch == config.batch_size {
            let tokens = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
                .reshape([config.batch_size, context]);
            let mask = Tensor::<B, 1>::from_floats(mask_flat.as_slice(), device)
                .reshape([config.batch_size, context]);
            let out =
                model.specialist_step_boxes(tokens, mask, &targets, config.router_aux_weight)?;
            let loss_value: f32 = out.loss.clone().into_scalar();
            let grads = if out.tokens_counted > 0 && loss_value.is_finite() {
                Some(trainable.gradients::<B, _>(&mut out.loss.backward(), &model))
            } else {
                None
            };
            let obs = StepObs {
                loss: loss_value,
                ce: out.completion_ce,
                router_aux: out.router_aux,
                target_traffic: out.target_traffic,
                target_top1: out.target_top1,
                balance: out.balance,
                tokens_counted: out.tokens_counted,
                energy: out.energy,
            };
            (loss_value, obs, grads)
        } else {
            // Gradient accumulation: the batch runs as k sequential
            // micro-batch forward/backwards, so peak activation memory is one
            // micro-batch's. Each micro-batch's loss is scaled by its share
            // of the batch's supervised completion tokens — counted host-side
            // with the same keep rule `specialist_step_boxes` applies (mask
            // positive at t in 1..n, next token not <pad>) and asserted
            // against the step's own count, so the CE gradient is exactly the
            // full batch's. A micro-batch with no supervised target or a
            // non-finite loss is discarded, not folded — the same policy as
            // the direct path, one level down.
            let k = config.batch_size / micro_batch;
            let pad_id = i64::from(Special::Pad.id());
            let mut counted = Vec::with_capacity(k);
            for m in 0..k {
                let row0 = m * micro_batch;
                let mut c = 0usize;
                for row in row0..row0 + micro_batch {
                    for t in 1..context {
                        if mask_flat[row * context + t] > 0.0 && flat[row * context + t] != pad_id {
                            c += 1;
                        }
                    }
                }
                counted.push(c);
            }
            let total_counted: usize = counted.iter().sum();
            let mut accumulator = crate::schedule::GradientAccumulator::new(k);
            let mut summed: Option<burn::optim::GradientsParams> = None;
            let mut obs = StepObs {
                loss: 0.0,
                ce: 0.0,
                router_aux: 0.0,
                target_traffic: 0.0,
                target_top1: 0.0,
                balance: 0.0,
                tokens_counted: total_counted,
                energy: Vec::new(),
            };
            let mut aux_weight = 0.0f32; // share of the batch whose aux was on
            let mut skipped_micros = 0usize;
            for m in 0..k {
                let lo = m * micro_batch * context;
                let hi = lo + micro_batch * context;
                let tokens = Tensor::<B, 1, Int>::from_ints(flat[lo..hi].as_ref(), device)
                    .reshape([micro_batch, context]);
                let mask = Tensor::<B, 1>::from_floats(mask_flat[lo..hi].as_ref(), device)
                    .reshape([micro_batch, context]);
                let out = model.specialist_step_boxes(
                    tokens,
                    mask,
                    &targets[m * micro_batch..(m + 1) * micro_batch],
                    config.router_aux_weight,
                )?;
                assert_eq!(
                    out.tokens_counted, counted[m],
                    "the host-side token count must mirror the step's own keep rule"
                );
                let micro_loss: f32 = out.loss.clone().into_scalar();
                if out.tokens_counted == 0 || !micro_loss.is_finite() {
                    accumulator.skip();
                    skipped_micros += 1;
                    continue;
                }
                let w = out.tokens_counted as f32 / total_counted.max(1) as f32;
                obs.loss += w * micro_loss;
                obs.ce += w * out.completion_ce;
                obs.balance += w * out.balance;
                if out.router_aux.is_finite() {
                    obs.router_aux += w * out.router_aux;
                    obs.target_traffic += w * out.target_traffic;
                    obs.target_top1 += w * out.target_top1;
                    aux_weight += w;
                }
                obs.energy = out.energy;
                let scaled = out.loss.mul_scalar(f64::from(w));
                let grads = trainable.gradients::<B, _>(&mut scaled.backward(), &model);
                if let crate::schedule::Cycle::Ready(g) = accumulator.fold(grads, &model) {
                    // The cycle completes on the k-th fold; earlier folds
                    // return Filling. None means every micro-batch in the
                    // cycle was discarded.
                    summed = g;
                }
            }
            if aux_weight > 0.0 {
                obs.router_aux /= aux_weight;
                obs.target_traffic /= aux_weight;
                obs.target_top1 /= aux_weight;
            } else {
                obs.router_aux = f32::NAN;
                obs.target_traffic = f32::NAN;
                obs.target_top1 = f32::NAN;
            }
            if skipped_micros > 0 {
                println!("step {step}: {skipped_micros}/{k} micro-batches discarded");
            }
            (obs.loss, obs, summed)
        };
        let Some(mut grads) = grads else {
            report.steps_skipped += 1;
            if obs.tokens_counted == 0 {
                println!("step {step}: batch has no supervised target, step discarded");
            } else {
                // The same policy as the other loops: a pathological step is
                // discarded, not clipped into something that looks fine.
                println!("step {step}: non-finite loss {loss_value}, step discarded");
            }
            continue;
        };
        if config.clip_norm > 0.0 {
            let total_norm = crate::quality::global_grad_norm(&model, &grads);
            if total_norm > config.clip_norm {
                crate::schedule::clip_gradients(&mut grads, &model, total_norm, config.clip_norm);
                report.steps_clipped += 1;
            }
        }
        model = optim.step(config.lr, model, grads);

        if report.steps_taken == 0 {
            report.first_loss = loss_value;
            report.first_ce = obs.ce;
            report.first_router_aux = obs.router_aux;
            report.first_target_traffic = obs.target_traffic;
            report.first_target_top1 = obs.target_top1;
        }
        report.last_loss = loss_value;
        report.last_ce = obs.ce;
        report.last_router_aux = obs.router_aux;
        report.last_target_traffic = obs.target_traffic;
        report.last_target_top1 = obs.target_top1;
        report.steps_taken += 1;
        report.tokens_seen += obs.tokens_counted;
        loss_sum += f64::from(loss_value);

        if config.log_every > 0 && (step % config.log_every == 0 || step + 1 == config.steps) {
            let mut line = format!(
                "step {step}: loss {:.4} ce {:.4} balance {:.4}",
                loss_value, obs.ce, obs.balance
            );
            if obs.router_aux.is_finite() {
                line.push_str(&format!(
                    " | router aux {:.4} p(lang) {:.3} top1 {:.3}",
                    obs.router_aux, obs.target_traffic, obs.target_top1
                ));
            }
            if let (Some(first), Some(last)) = (obs.energy.first(), obs.energy.last()) {
                line.push_str(&format!(" | energy {first:.4}->{last:.4}"));
            }
            println!("{line}");
        }

        if config.checkpoint_every > 0
            && (step + 1) % config.checkpoint_every == 0
            && step + 1 < config.steps
        {
            let elapsed = elapsed_before + started.elapsed().as_secs_f64();
            if let Some(path) =
                save_state(step + 1, &model, &optim, &rng, &report, loss_sum, elapsed)?
            {
                println!("step {step}: checkpoint {}", path.display());
                report.periodic_checkpoints.push((step + 1, path));
            }
        }
    }

    anyhow::ensure!(
        report.steps_taken > 0,
        "every step produced a non-finite loss"
    );
    report.mean_loss = (loss_sum / report.steps_taken as f64) as f32;
    report.elapsed_secs = elapsed_before + started.elapsed().as_secs_f64();
    report.checkpoint = save_state(
        config.steps,
        &model,
        &optim,
        &rng,
        &report,
        loss_sum,
        report.elapsed_secs,
    )?;
    Ok((model, report))
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
    use crate::expert_index::coding_student_spec;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn unique_tmp(stem: &str) -> std::path::PathBuf {
        let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("student-{stem}-{}-{unique}", std::process::id()))
    }

    /// Every parameter's values, flattened in module traversal order (the
    /// qwentrain/geomfusion discipline: traversal order pairs the values,
    /// because `ParamId`s from two independent builds can never pair them).
    fn param_values<M, BK>(module: &M) -> Vec<f32>
    where
        BK: Backend,
        M: Module<BK>,
    {
        struct Flatten(Vec<f32>);
        impl<BK: Backend> burn::module::ModuleVisitor<BK> for Flatten {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<BK, D>>) {
                self.0.extend(param.val().to_data().iter::<f32>());
            }
        }
        let mut flat = Flatten(Vec::new());
        module.visit(&mut flat);
        flat.0
    }

    /// The largest absolute difference between two equal-length flattenings.
    fn worst_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "flattened parameter counts differ");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_coding_boxes_default_matches_the_pre_config_build() {
        use sha2::{Digest, Sha256};
        // Back-compat, pinned against numbers captured from the build that
        // predates configurable coding boxes: the default config must build
        // exactly the spec and parameter count that build produced, because
        // the six coding checkpoints and the in-flight field specialists
        // were trained against it.
        let digest_of = |config: &StudentConfig| {
            let spec = config.mosme_spec().unwrap();
            let json = serde_json::to_string(&spec).unwrap();
            Sha256::digest(json.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        let default = StudentConfig::default();
        assert_eq!(
            default.coding_boxes,
            vec!["rust", "python", "c", "cpp", "jsts", "web"]
        );
        assert_eq!(
            digest_of(&default),
            "c67edb8f6d2ad4b0214e7a588d8bb93baee6c94427e43d2fba45e0ffbfcc67ff",
            "the default MoSME spec drifted from the pre-config build"
        );
        let cost = default.cost().unwrap();
        assert_eq!(cost.total_params, 33_800_768);
        assert_eq!(cost.active_params, 7_990_584);
        assert_eq!(cost.trunk_flops_per_token, 19_812_352.0);

        assert_eq!(
            digest_of(&tiny_config()),
            "d98b385290bafce13fb86c49bd869d2b2a3eb68bb6fa3b81228203945979be48",
            "the tiny MoSME spec drifted from the pre-config build"
        );
        let tiny_cost = tiny_config().cost().unwrap();
        assert_eq!(tiny_cost.total_params, 217_004);
        assert_eq!(tiny_cost.active_params, 96_800);

        // A sidecar from before `coding_boxes` existed recovers the default.
        let mut old_wire = serde_json::to_value(&default).unwrap();
        old_wire.as_object_mut().unwrap().remove("coding_boxes");
        let parsed: StudentConfig = serde_json::from_value(old_wire).unwrap();
        assert_eq!(parsed, default);

        // The six shipped languages keep their box-router indices.
        let spec = default.mosme_spec().unwrap();
        for (idx, language) in CodeLanguage::ALL.iter().enumerate() {
            assert_eq!(spec.boxes[idx].id, language.box_id());
            assert_eq!(
                default.box_target(Some(*language), None),
                Some(BoxTarget::Language(idx)),
                "{} keeps box index {idx}",
                language.name()
            );
        }
        // ...and the extended four have no box until a config names them.
        for language in [
            CodeLanguage::Go,
            CodeLanguage::Java,
            CodeLanguage::CSharp,
            CodeLanguage::Php,
        ] {
            assert_eq!(default.box_target(Some(language), None), None);
        }
    }

    #[test]
    fn test_coding_boxes_are_config_driven_and_validated() {
        // Ten boxes: the shipped six plus the four extensions, in PLAN-NINA
        // order. Field boxes move with the coding count.
        let ten: Vec<String> = CodeLanguage::KNOWN
            .iter()
            .map(|l| l.name().to_string())
            .collect();
        let config = StudentConfig {
            coding_boxes: ten,
            field_boxes: vec!["pentest".to_string()],
            ..tiny_config()
        };
        config.validate().unwrap();
        let spec = config.mosme_spec().unwrap();
        assert_eq!(spec.boxes.len(), 11);
        assert_eq!(spec.boxes[6].id, "coding:go");
        assert_eq!(spec.boxes[6].label, "Go specialist");
        assert_eq!(spec.boxes[9].id, "coding:php");
        assert_eq!(spec.boxes[10].id, "field:pentest");
        assert_eq!(config.num_boxes(), 11);
        assert_eq!(config.field_box_index("pentest"), Some(10));
        assert_eq!(
            config.box_target(Some(CodeLanguage::Go), None),
            Some(BoxTarget::Language(6))
        );
        assert_eq!(
            config.box_target(Some(CodeLanguage::Php), None),
            Some(BoxTarget::Language(9))
        );
        assert_eq!(
            config.box_target(None, Some("pentest")),
            Some(BoxTarget::Field(10))
        );

        // A config that reorders the boxes supervises the reordered indices.
        let shuffled = StudentConfig {
            coding_boxes: vec!["go".to_string(), "rust".to_string()],
            ..tiny_config()
        };
        assert_eq!(
            shuffled.box_target(Some(CodeLanguage::Go), None),
            Some(BoxTarget::Language(0))
        );
        assert_eq!(
            shuffled.box_target(Some(CodeLanguage::Rust), None),
            Some(BoxTarget::Language(1))
        );
        let spec = shuffled.mosme_spec().unwrap();
        assert_eq!(spec.boxes[0].id, "coding:go");
        assert_eq!(spec.boxes[1].id, "coding:rust");

        // Bad lists are config errors naming the field at fault.
        for bad in [
            vec![String::new()],
            vec!["Go".to_string()],
            vec!["rust".to_string(), "rust".to_string()],
            vec![],
        ] {
            let config = StudentConfig {
                coding_boxes: bad,
                ..tiny_config()
            };
            let err = config.validate().unwrap_err().to_string();
            assert!(err.contains("coding"), "{err}");
        }
        let err = StudentConfig {
            top_box: 7,
            ..tiny_config()
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("top_box"), "{err}");
        // top_box scales with the configured box count.
        StudentConfig {
            coding_boxes: CodeLanguage::KNOWN
                .iter()
                .map(|l| l.name().to_string())
                .collect(),
            top_box: 7,
            ..tiny_config()
        }
        .validate()
        .unwrap();
    }

    fn tiny_config() -> StudentConfig {
        StudentConfig {
            hidden_size: 64,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            context: 32,
            intermediate_size: 32,
            cond_hidden_size: 16,
            frequency_embedding_size: 16,
            experts_per_box: 2,
            geom_slots: 4,
            geom_dim: 8,
            geom_refine_steps: 2,
            geom_cond_hidden: 16,
            geom_frequency: 8,
            ..StudentConfig::default()
        }
    }

    fn tiny_student(
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> GeometricStudent<B> {
        GeometricStudent::new(&tiny_config(), device).unwrap()
    }

    fn tokens(
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        b: usize,
        n: usize,
    ) -> Tensor<B, 2, Int> {
        let ids: Vec<i64> = (0..b * n)
            .map(|i| (i * 7 + 3) % 250 + 4)
            .map(|v| v as i64)
            .collect();
        Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([b, n])
    }

    fn to_vec(t: Tensor<B, 3>) -> Vec<f32> {
        t.into_data().convert::<f32>().iter::<f32>().collect()
    }

    #[test]
    fn test_language_name_parse_roundtrip() {
        for language in CodeLanguage::KNOWN {
            assert_eq!(CodeLanguage::parse(language.name()).unwrap(), language);
            assert_eq!(language.box_id(), format!("coding:{}", language.name()));
        }
        assert_eq!(CodeLanguage::parse("c++").unwrap(), CodeLanguage::Cpp);
        assert_eq!(CodeLanguage::parse("TS").unwrap(), CodeLanguage::JsTs);
        assert_eq!(CodeLanguage::parse(" web ").unwrap(), CodeLanguage::Web);
        assert_eq!(CodeLanguage::parse("golang").unwrap(), CodeLanguage::Go);
        assert_eq!(CodeLanguage::parse("C#").unwrap(), CodeLanguage::CSharp);
        assert_eq!(CodeLanguage::parse("cs").unwrap(), CodeLanguage::CSharp);
        let err = CodeLanguage::parse("cobol").unwrap_err().to_string();
        assert!(err.contains("cobol"), "{err}");
        // The registry order is stable: index() agrees with KNOWN's position.
        for (idx, language) in CodeLanguage::KNOWN.iter().enumerate() {
            assert_eq!(language.index(), idx);
        }
        // The shipped six are the prefix — the default box order and every
        // checkpoint written before the extensions existed.
        assert_eq!(CodeLanguage::KNOWN[..6], CodeLanguage::ALL[..]);
        assert_eq!(CodeLanguage::KNOWN[6..], CodeLanguage::EXTENDED[..]);
    }

    #[test]
    fn test_registry_boxes_match_languages() {
        let spec = coding_student_spec(2, 1, 1).unwrap();
        let boxes: Vec<&str> = spec.boxes.iter().map(|b| b.id.as_str()).collect();
        let expected: Vec<String> = CodeLanguage::ALL.iter().map(|l| l.box_id()).collect();
        assert_eq!(boxes, expected);
    }

    #[test]
    fn test_default_config_validates_and_serde_defaults() {
        let config = StudentConfig::default();
        config.validate().unwrap();
        // An empty sidecar is the default config: every field has a default.
        let parsed: StudentConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, config);
        // A partial sidecar from an older build still parses.
        let partial: StudentConfig = serde_json::from_str(r#"{"hidden_size": 256}"#).unwrap();
        assert_eq!(partial.hidden_size, 256);
        assert_eq!(partial.attention, "3:1");
        assert_eq!(partial.norm_kind, NormKind::Rms);
        assert!((partial.rotary_fraction - 0.25).abs() < f64::EPSILON);
        // Round trip.
        let text = serde_json::to_string_pretty(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<StudentConfig>(&text).unwrap(),
            config
        );
    }

    #[test]
    fn test_invalid_config_names_the_field() {
        let cases: [(StudentConfig, &str); 8] = [
            (
                StudentConfig {
                    num_heads: 3,
                    ..tiny_config()
                },
                "num_heads",
            ),
            (
                StudentConfig {
                    num_kv_heads: 3,
                    ..tiny_config()
                },
                "num_kv_heads",
            ),
            (
                StudentConfig {
                    rotary_fraction: 0.0,
                    ..tiny_config()
                },
                "rotary_fraction",
            ),
            (
                StudentConfig {
                    attention: "bogus".into(),
                    ..tiny_config()
                },
                "attention",
            ),
            (
                StudentConfig {
                    top_expert: 3,
                    ..tiny_config()
                },
                "top_expert",
            ),
            (
                StudentConfig {
                    top_box: 7,
                    ..tiny_config()
                },
                "top_box",
            ),
            (
                StudentConfig {
                    geom_slots: 1,
                    ..tiny_config()
                },
                "geom_slots",
            ),
            (
                StudentConfig {
                    geom_refine_steps: 0,
                    ..tiny_config()
                },
                "geom_refine_steps",
            ),
        ];
        for (config, field) in cases {
            let err = config.validate().unwrap_err().to_string();
            assert!(err.contains(field), "expected '{field}' named in: {err}");
        }
    }

    #[test]
    fn test_forward_is_finite_and_shaped() {
        let device = Default::default();
        let student = tiny_student(&device);
        let out = student.forward(tokens(&device, 2, 8)).unwrap();
        assert_eq!(out.logits.dims(), [2, 8, VOCAB_SIZE]);
        assert_eq!(out.hidden.dims(), [2, 8, 64]);
        let logits = to_vec(out.logits);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(out.refine.energy.len(), 2);
        assert_eq!(out.refine.displacement.len(), 2);
        assert!(out.refine.energy.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_fresh_student_is_exactly_the_plain_trunk() {
        // The zero-initialized readout makes the geometric stream an exact
        // no-op at init: same hidden states in, same tied readout, so the
        // student's logits are the trunk's, bit for bit (tolerance 0.0).
        let device = Default::default();
        let student = tiny_student(&device);
        crate::tensor_ext::force_initialization(&student);
        let input = tokens(&device, 2, 8);
        let student_logits = to_vec(student.forward(input.clone()).unwrap().logits);
        let trunk_logits = to_vec(student.trunk().forward(input).logits);
        assert_eq!(student_logits, trunk_logits);
    }

    #[test]
    fn test_refine_depth_changes_hidden_and_energy_falls() {
        let device = Default::default();
        let mut student = tiny_student(&device);
        // Swap the zero-init readout for a random one so the stream's write
        // is visible; the init identity is covered by its own test.
        student.readout = LinearConfig::new(8, 64).with_bias(true).init(&device);
        let input = tokens(&device, 2, 8);

        let shallow = student.forward_depth(input.clone(), Some(1)).unwrap();
        let deep = student.forward_depth(input, Some(4)).unwrap();
        assert_eq!(shallow.refine.energy.len(), 1);
        assert_eq!(deep.refine.energy.len(), 4);

        // More certified descent: same landscape (the context is fixed from
        // the same input), so the energy after 4 steps is below step 1's.
        let e1 = shallow.refine.energy[0];
        let e4 = deep.refine.energy[3];
        assert!(
            e4 <= e1 + 1e-6,
            "energy must fall with depth: step 1 was {e1}, step 4 was {e4}"
        );
        // ...and it is monotone within the deep run too.
        for pair in deep.refine.energy.windows(2) {
            assert!(pair[1] <= pair[0] + 1e-6, "non-descent: {:?}", pair);
        }

        // The depth actually reaches the hidden states (and so the logits).
        assert_ne!(to_vec(shallow.hidden), to_vec(deep.hidden));
        assert_ne!(to_vec(shallow.logits), to_vec(deep.logits));
    }

    #[test]
    fn test_language_traffic_is_a_distribution() {
        let device = Default::default();
        let student = tiny_student(&device);
        let routing = student.route_languages(tokens(&device, 2, 8)).unwrap();
        assert_eq!(routing.traffic.len(), tiny_config().num_boxes());
        assert!(routing.traffic.iter().all(|v| *v >= 0.0 && v.is_finite()));
        let sum: f32 = routing.traffic.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "box traffic must form a distribution, sums to {sum}"
        );
        assert_eq!(routing.top_language.len(), 16);
        // The trunk really ran its MoSME sites (layers 1 and 3 of 4).
        assert_eq!(routing.trunk_layers.len(), 2);
        for stats in &routing.trunk_layers {
            let load_sum: f32 = stats.load.iter().sum();
            assert!((load_sum - 1.0).abs() < 1e-4, "expert loads sum to 1");
            assert_eq!(stats.load.len(), 2 * CODING_LANGUAGES.len());
        }
    }

    #[test]
    fn test_checkpoint_roundtrip_is_bit_identical() {
        let device = Default::default();
        let config = tiny_config();
        let student = GeometricStudent::<B>::new(&config, &device).unwrap();
        crate::tensor_ext::force_initialization(&student);
        let dir = unique_tmp("ckpt");
        std::fs::create_dir_all(&dir).unwrap();
        let path =
            crate::checkpoint::save_content_addressed(student.clone(), &dir, "student").unwrap();

        let fresh = GeometricStudent::<B>::new(&config, &device).unwrap();
        let restored = crate::checkpoint::load(fresh, &path, &device).unwrap();
        assert_eq!(
            crate::checkpoint::canonical_hash_hex(&student),
            crate::checkpoint::canonical_hash_hex(&restored),
            "the round trip must be bit-identical"
        );
        let input = tokens(&device, 1, 6);
        assert_eq!(
            to_vec(student.forward(input.clone()).unwrap().logits),
            to_vec(restored.forward(input).unwrap().logits),
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_cost_matches_the_built_model() {
        // The accounting is honest only if it counts what Burn counts: the
        // shape-derived total must equal the built module's parameter count.
        let device = Default::default();
        let config = StudentConfig::default();
        let student = GeometricStudent::<B>::new(&config, &device).unwrap();
        let cost = config.cost().unwrap();
        assert_eq!(
            cost.total_params,
            student.num_params(),
            "shape accounting drifted from the module ({} vs {})",
            cost.total_params,
            student.num_params()
        );
        assert!(cost.active_params < cost.total_params);
        assert!(
            cost.resident_expert_params + cost.geom_stream_params + cost.probe_router_params
                <= cost.total_params
        );
        println!(
            "local student: {:.2}M active / {:.2}M total params, {:.2} MFLOPs per token",
            cost.active_params as f64 / 1e6,
            cost.total_params as f64 / 1e6,
            cost.trunk_flops_per_token / 1e6,
        );
    }

    #[test]
    fn test_few_billion_active_validates_and_reports() {
        let config = StudentConfig::few_billion_active();
        config.validate().unwrap();
        let cost = config.cost().unwrap();
        assert!(
            cost.active_params > 1_000_000_000,
            "expected a few billion active, got {}",
            cost.active_params
        );
        assert!(
            cost.active_params < 6_000_000_000,
            "expected *few* billion active, got {}",
            cost.active_params
        );
        assert!(cost.total_params > 4 * cost.active_params);
        println!(
            "few-billion student: {:.3}B active / {:.2}B total ({:.1}% resident experts), \
             {:.1} GFLOPs per token",
            cost.active_params as f64 / 1e9,
            cost.total_params as f64 / 1e9,
            100.0 * cost.resident_expert_params as f64 / cost.total_params as f64,
            cost.trunk_flops_per_token / 1e9,
        );
    }

    #[test]
    fn test_tensor_data_roundtrip_sanity() {
        // Guards the test helpers themselves: the logit comparison reads the
        // raw buffer, so a layout change must fail loudly here, not silently
        // pass everywhere.
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 2, 2]),
            &device,
        );
        assert_eq!(to_vec(t), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ------------------------------------------------- wave-2 training tests --

    fn trace_line(language: &str, prompt: &str, completion: &str) -> String {
        serde_json::json!({
            "id": format!("{language}-1"),
            "language": language,
            "prompt": prompt,
            "completion": completion,
            "source": "test",
        })
        .to_string()
    }

    fn write_lines(dir: &std::path::Path, name: &str, lines: &[String]) -> std::path::PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
        path
    }

    /// The toy-scale per-language signal: language `i`'s completions are
    /// bytes drawn from the disjoint ASCII range `[48 + 8i, 48 + 8i + 8)`,
    /// so every completion token carries the language's marker. This is an
    /// honest mechanism proof at toy scale — the router can only learn the
    /// mapping because the signal exists, which is exactly the property the
    /// real traces provide by being code in one language.
    /// A distinct 24-character marker per language, so the router has separable
    /// per-language signal to learn from.
    ///
    /// Alphanumeric by construction, and not by accident: an earlier version drew
    /// from a shifting ASCII window, so some languages' markers were strings
    /// of `[]^_` — unbalanced delimiters and high-entropy non-code, which is
    /// exactly what the data-side cheat gate is built to refuse. The gate was
    /// right about those rows and the fixture was wrong, so the alphabet moved
    /// rather than the filter.
    fn marker_completion(language: CodeLanguage, len: usize, salt: usize) -> String {
        const ALPHABET: &[u8; 62] =
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let base = (8 * language.index()) % ALPHABET.len();
        (0..len)
            .map(|k| ALPHABET[(base + (k * 3 + salt) % 8) % ALPHABET.len()] as char)
            .collect()
    }

    fn synthetic_trace_fixture(dir: &std::path::Path, per_language: usize) -> std::path::PathBuf {
        let mut lines = Vec::new();
        // The shipped six: the default config's box set, which is what the
        // toy model below trains and routes.
        for language in CodeLanguage::ALL {
            for k in 0..per_language {
                lines.push(trace_line(
                    language.name(),
                    "code: ",
                    &marker_completion(language, 24, k),
                ));
            }
        }
        write_lines(dir, "train.jsonl", &lines)
    }

    /// The data-side gate is on the loader, not on the callers, so a
    /// training path that never heard of it still gets it. This is the test
    /// that would fail if someone made the gate opt-in.
    #[test]
    fn test_language_batch_drops_a_planted_cheat_without_being_asked_to() {
        let dir = unique_tmp("gate");
        let lines = vec![
            trace_line(
                "rust",
                "add ",
                "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
            ),
            trace_line(
                "rust",
                "stub ",
                "fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n",
            ),
            trace_line("python", "quiet ", "def f(x):\n    return x\n"),
        ];
        let path = write_lines(&dir, "train.jsonl", &lines);

        let batch = LanguageBatch::load(&path, None, 64, 7).unwrap();
        let scan = batch.cheat_scan();
        assert_eq!(scan.scanned, 3, "every row is gated, none skipped silently");
        assert_eq!(scan.dropped, 1);
        assert_eq!(scan.kept, 2);
        assert_eq!(
            scan.dropped_by_class.get("data-stub").copied(),
            Some(1),
            "the count names the class: {:?}",
            scan.dropped_by_class
        );
        assert_eq!(scan.kept + scan.dropped, scan.scanned);
        // Nothing gated, but the claim is still made: a zero on a summary that
        // exists is "checked and clean", which a missing summary is not.
        assert!(scan.learned.is_none(), "no learned layer was configured");
    }

    #[test]
    fn test_language_batch_refuses_a_file_the_gate_empties() {
        let dir = unique_tmp("gate-empty");
        let lines = vec![trace_line(
            "rust",
            "stub ",
            "fn add() -> i32 {\n    todo!()\n}\n",
        )];
        let path = write_lines(&dir, "train.jsonl", &lines);
        let err = LanguageBatch::load(&path, None, 64, 7).expect_err("emptied by the gate");
        let message = format!("{err:#}");
        assert!(message.contains("cheat gate"), "{message}");
        assert!(message.contains("data-stub"), "names the reason: {message}");
    }

    /// A completion that is only balanced behind its prompt is a suffix
    /// fragment, not corruption — the loader trains prompt + completion as one
    /// unit, so it gates the unit.
    #[test]
    fn test_language_batch_keeps_a_fragment_balanced_by_its_prompt() {
        let dir = unique_tmp("gate-fragment");
        let prompt = "fn compute(x: i32) -> i32 {\n    if x > 0 {\n";
        let completion = "        x * 2\n    }\n    0\n}";
        let lines = vec![trace_line("rust", prompt, completion)];
        let path = write_lines(&dir, "train.jsonl", &lines);
        let batch = LanguageBatch::load(&path, None, 96, 7).unwrap();
        assert_eq!(
            batch.cheat_scan().dropped,
            0,
            "a balanced fragment was read as broken: {:?}",
            batch.cheat_scan().dropped_by_class
        );
    }

    #[test]
    fn test_language_batch_filters_masks_truncates_and_is_deterministic() {
        let dir = unique_tmp("batch");
        let long_prompt = "p".repeat(64);
        let lines = vec![
            trace_line("rust", "fn ", "x"),
            trace_line("python", "def ", "y"),
            trace_line("rust", "let ", "zz"),
            trace_line("go", "package ", "main"), // a known language, but not the filter's
            trace_line("rust", &long_prompt, "never seen"),
            trace_line("rust", "use ", "www"),
        ];
        let path = write_lines(&dir, "train.jsonl", &lines);

        let batch = LanguageBatch::load(&path, Some(CodeLanguage::Rust), 32, 7).unwrap();
        // Three rust lines survive; the over-long prompt fills the context
        // alone and is skipped rather than supervising nothing.
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.skipped_no_target(), 1);
        assert_eq!(batch.lines_read(), 6);
        assert!(batch
            .languages_seen()
            .iter()
            .any(|(l, n)| l == "go" && *n == 1));
        for idx in 0..batch.len() {
            assert_eq!(batch.language_of(idx), Some(CodeLanguage::Rust));
            assert_eq!(batch.tokens(idx).len(), 32);
            assert_eq!(batch.mask(idx).len(), 32);
        }
        // The mask supervises exactly the completion tokens and the closing
        // <eos>: find the untruncated two-token completion line and check.
        let pos = (0..batch.len())
            .find(|idx| batch.tokens(*idx)[1] == b'f' as u16)
            .unwrap();
        let expected_tokens: Vec<u16> = [Special::Bos.id(), b'f' as u16, b'n' as u16, b' ' as u16]
            .into_iter()
            .chain([b'x' as u16, Special::Eos.id()])
            .chain(std::iter::repeat(Special::Pad.id()))
            .take(32)
            .collect();
        assert_eq!(batch.tokens(pos), expected_tokens.as_slice());
        let expected_mask: Vec<f32> = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
            .into_iter()
            .chain(std::iter::repeat(0.0))
            .take(32)
            .collect();
        assert_eq!(batch.mask(pos), expected_mask.as_slice());
        // A pad position is never a target.
        assert!(batch.mask(pos)[6..].iter().all(|m| *m == 0.0));

        // The seeded shuffle is the order contract: same seed, same order;
        // a different seed (almost surely) reorders a large enough set.
        let again = LanguageBatch::load(&path, Some(CodeLanguage::Rust), 32, 7).unwrap();
        let order = |b: &LanguageBatch| {
            (0..b.len())
                .map(|i| b.tokens(i).to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(order(&batch), order(&again));

        let many: Vec<String> = (0..12)
            .map(|k| trace_line("rust", "code: ", &format!("line {k}")))
            .collect();
        let many_path = write_lines(&dir, "many.jsonl", &many);
        let order_of = |seed: u64| {
            let b = LanguageBatch::load(&many_path, None, 32, seed).unwrap();
            (0..b.len())
                .map(|i| b.tokens(i).to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(order_of(3), order_of(3));
        assert_ne!(order_of(3), order_of(4), "the seed must choose the order");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_language_batch_missing_file_and_empty_filter_are_loud() {
        let dir = unique_tmp("batch-errors");
        let missing = dir.join("absent.jsonl");
        let err = LanguageBatch::load(&missing, None, 32, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("absent.jsonl"), "{err}");

        let path = write_lines(&dir, "train.jsonl", &[trace_line("rust", "a", "b")]);
        let err = LanguageBatch::load(&path, Some(CodeLanguage::Cpp), 32, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("train.jsonl"), "{err}");
        assert!(err.contains("cpp"), "{err}");
        assert!(err.contains("rust:1"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_language_batch_reads_the_sft_fallback_schema() {
        // /srv/m-sdd/unifur/datasets/sft-12k.jsonl's shape: instruction +
        // response, `lang` frequently empty. Those lines load with
        // language = unknown: they can train the trunk but cannot supervise
        // a routing box.
        let dir = unique_tmp("batch-fallback");
        let lines = vec![
            serde_json::json!({"source": "magicoder", "lang": "", "instruction": "write", "response": "code"}).to_string(),
            serde_json::json!({"lang": "python", "instruction": "def f", "response": "pass"}).to_string(),
        ];
        let path = write_lines(&dir, "sft.jsonl", &lines);
        let batch = LanguageBatch::load(&path, None, 32, 0).unwrap();
        assert_eq!(batch.len(), 2);
        // The shuffle reorders; compare the multiset of languages.
        let mut languages: Vec<Option<CodeLanguage>> =
            (0..batch.len()).map(|i| batch.language_of(i)).collect();
        languages.sort_by_key(|l| l.map_or(99, |l| l.index() as i64));
        assert_eq!(languages, vec![Some(CodeLanguage::Python), None]);
        let only_python = LanguageBatch::load(&path, Some(CodeLanguage::Python), 32, 0).unwrap();
        assert_eq!(only_python.len(), 1);
        assert_eq!(only_python.language_of(0), Some(CodeLanguage::Python));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_specialist_step_supervises_completion_only_and_reports_the_router() {
        let device = Default::default();
        let student = tiny_student(&device);
        crate::tensor_ext::force_initialization(&student);
        // One row: <bos> "ab" "cd" <eos> <pad>..., completion mask on "cd"+<eos>.
        let context = 8;
        let mut ids = vec![Special::Pad.id(); context];
        ids[..6].copy_from_slice(&[
            Special::Bos.id(),
            b'a' as u16,
            b'b' as u16,
            b'c' as u16,
            b'd' as u16,
            Special::Eos.id(),
        ]);
        let mut mask = vec![0.0f32; context];
        mask[3] = 1.0;
        mask[4] = 1.0;
        mask[5] = 1.0;
        let tokens = Tensor::<B, 1, Int>::from_ints(
            ids.iter()
                .map(|t| i64::from(*t))
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([1, context]);
        let mask_t = Tensor::<B, 1>::from_floats(mask.as_slice(), &device).reshape([1, context]);

        let step = student
            .specialist_step(
                tokens.clone(),
                mask_t.clone(),
                &[Some(CodeLanguage::Rust)],
                0.5,
            )
            .unwrap();
        assert_eq!(step.tokens_counted, 3, "only the completion targets count");
        assert!(step.completion_ce.is_finite());
        assert!(step.router_aux.is_finite());
        assert!(step.target_traffic >= 0.0 && step.target_traffic <= 1.0);
        assert!(step.target_top1 >= 0.0 && step.target_top1 <= 1.0);
        assert!(step.balance.is_finite());
        // The geometric stream's certified descent, reported as a diagnostic.
        assert_eq!(step.energy.len(), 2);
        for pair in step.energy.windows(2) {
            assert!(pair[1] <= pair[0] + 1e-6, "non-descent: {pair:?}");
        }
        let loss_with_aux: f32 = step.loss.into_scalar();
        let plain = student
            .specialist_step(
                tokens.clone(),
                mask_t.clone(),
                &[Some(CodeLanguage::Rust)],
                0.0,
            )
            .unwrap();
        let loss_without_aux: f32 = plain.loss.into_scalar();
        assert!(
            (loss_with_aux - loss_without_aux - 0.5 * step.router_aux).abs() < 1e-5,
            "the aux enters at exactly its weight: {loss_with_aux} vs {loss_without_aux} + {}",
            0.5 * step.router_aux
        );

        // An unknown-language row switches the aux off: no supervision, and
        // the readouts say so rather than reporting a zero someone trusts.
        let unknown = student
            .specialist_step(tokens, mask_t, &[None], 0.5)
            .unwrap();
        assert!(unknown.router_aux.is_nan());
        assert!(unknown.target_top1.is_nan());
        assert_eq!(unknown.tokens_counted, 3);
    }

    #[test]
    fn test_specialist_training_falls_and_teaches_the_router() {
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let dir = unique_tmp("specialist");
        let fixture = synthetic_trace_fixture(&dir, 8);
        let data = LanguageBatch::load(&fixture, None, 32, 5).unwrap();
        assert_eq!(data.len(), 48);

        <AB as burn::tensor::backend::Backend>::seed(&device, 3);
        let student = GeometricStudent::<AB>::new(&tiny_config(), &device).unwrap();
        crate::tensor_ext::force_initialization(&student);
        let config = SpecialistTrainConfig {
            steps: 40,
            batch_size: 6,
            lr: 5e-3,
            router_aux_weight: 1.0,
            log_every: 0,
            seed: 11,
            ..SpecialistTrainConfig::default()
        };
        let (student, report) =
            train_language_specialist(student, &data, &config, &device).unwrap();
        assert_eq!(report.steps_taken, 40);
        println!(
            "toy specialist run: loss {:.4} -> {:.4}, ce {:.4} -> {:.4}, router aux {:.4} -> \
             {:.4}, p(lang) {:.3} -> {:.3}, top1 {:.3} -> {:.3}",
            report.first_loss,
            report.last_loss,
            report.first_ce,
            report.last_ce,
            report.first_router_aux,
            report.last_router_aux,
            report.first_target_traffic,
            report.last_target_traffic,
            report.first_target_top1,
            report.last_target_top1,
        );
        // Measured, not assumed: the completion CE fell against step 0.
        assert!(
            report.last_ce < report.first_ce,
            "completion CE did not fall: {} -> {}",
            report.first_ce,
            report.last_ce
        );
        assert!(report.last_loss < report.first_loss);
        // The aux moved the probe's mass onto the true language.
        assert!(
            report.last_target_traffic > report.first_target_traffic,
            "p(lang) did not move: {} -> {}",
            report.first_target_traffic,
            report.last_target_traffic
        );

        // The router-which-knows proof at toy scale: held-out per-language
        // prompts (fresh marker bytes, no shared prompt diluting the score)
        // must put the top-1 box on the true language above chance (1/6).
        let mut held_out = crate::langdata::RouterAgreement::default();
        for language in CodeLanguage::ALL {
            let text = marker_completion(language, 8, 100);
            let mut ids = vec![Special::Bos.id()];
            ids.extend(ByteTokenizer::new().encode(&text));
            let n = ids.len();
            let tokens = Tensor::<AB, 1, Int>::from_ints(
                ids.iter()
                    .map(|t| i64::from(*t))
                    .collect::<Vec<_>>()
                    .as_slice(),
                &device,
            )
            .reshape([1, n]);
            let routing = student.route_languages(tokens).unwrap();
            let agreement =
                crate::langdata::router_agreement(&routing.top_language, &[language], n).unwrap();
            held_out.correct[language.index()] += agreement.correct[language.index()];
            held_out.total[language.index()] += agreement.total[language.index()];
        }
        print!("{}", held_out.render());
        let chance = 1.0 / CodeLanguage::ALL.len() as f32;
        for language in CodeLanguage::ALL {
            let accuracy = held_out.accuracy(language).unwrap();
            assert!(
                accuracy > chance,
                "{} held-out routing accuracy {accuracy:.3} is not above chance {chance:.3}",
                language.name()
            );
        }
        assert!(
            held_out.overall().unwrap() > 0.5,
            "the toy proof should be decisive, got {:.3}",
            held_out.overall().unwrap()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_specialist_resume_restores_the_exact_state() {
        // Mirrors tests/resume.rs's LM case, in-file. What this pins, and
        // why it is proof against the parallel gate:
        //
        // - **The restore is bit-exact.** The checkpoint's
        //   content-addressed name embeds a prefix of the weights hash
        //   computed at save time, and a fresh load reproduces that hash --
        //   the weights on disk are the weights the uninterrupted run held
        //   at step 2, to the bit. The restored report state (the first
        //   loss, carried through the serialized extras) is compared
        //   bitwise too. Neither half computes anything, so neither can
        //   flake under load.
        // - **The continuation matches inside measured run-to-run noise.**
        //   It cannot be pinned bitwise: with a MoSME trunk and *every*
        //   parameter trainable, burn-ndarray 0.21's threaded backward
        //   reduces in an order that depends on thread scheduling, so two
        //   IDENTICAL uninterrupted runs differ in the final loss -- 0 in
        //   a quiet process, ~1e-6 under gate load, as measured across the
        //   stress runs this test was hardened with (bisected to the MoSME
        //   layer: `LmConfig::tiny()` reproduces exactly, adding its MoSME
        //   trunk does not; the repo's resident-specialist test still
        //   asserts bitwise resume because scoped freezing excludes the
        //   affected parameters).
        //   The noise is measured in-process below -- a second
        //   uninterrupted run of the same config from the same init -- so
        //   the bound tracks the load the gate is actually imposing: ten
        //   times the measured noise, floored at 2e-5 on the loss and
        //   1e-5 of the weight scale. A real resume defect (a dropped
        //   optimizer page, a stale weight, a wrong batch order) moves
        //   the loss by 1e-2 or more, orders above the bound. The data
        //   order restores through the serialized host RNG and the
        //   optimizer moments through their record; the bounded
        //   continuation is what exercises both.
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let dir = unique_tmp("specialist-resume");
        let fixture = synthetic_trace_fixture(&dir, 4);
        let data = LanguageBatch::load(&fixture, None, 32, 9).unwrap();

        <AB as burn::tensor::backend::Backend>::seed(&device, 3);
        let init = GeometricStudent::<AB>::new(&tiny_config(), &device).unwrap();
        crate::tensor_ext::force_initialization(&init);

        let base = SpecialistTrainConfig {
            steps: 4,
            batch_size: 2,
            log_every: 0,
            language: Some(CodeLanguage::Rust),
            ..SpecialistTrainConfig::default()
        };
        let full = SpecialistTrainConfig {
            out_dir: Some(dir.join("a")),
            checkpoint_every: 2,
            ..base.clone()
        };
        let (model_a, report_a) =
            train_language_specialist(init.clone(), &data, &full, &device).unwrap();
        assert_eq!(report_a.periodic_checkpoints.len(), 1);
        let (at, halfway) = report_a.periodic_checkpoints[0].clone();
        assert_eq!(at, 2);

        // The checkpoint round trip is exact: the weights on disk are the
        // weights the uninterrupted run held at step 2, to the bit.
        let loaded = crate::checkpoint::load::<AB, _>(
            GeometricStudent::<AB>::new(&tiny_config(), &device).unwrap(),
            &halfway,
            &device,
        )
        .unwrap();
        let expected = halfway
            .file_stem()
            .and_then(|s| s.to_str())
            .map_or(String::new(), |s| {
                s.trim_start_matches("student-").to_string()
            });
        let hash = crate::checkpoint::canonical_hash_hex(&loaded);
        assert!(
            hash.starts_with(&expected),
            "the content-addressed name embeds a prefix of the weights hash: {hash} vs {expected}"
        );

        // The run-to-run noise floor, measured rather than assumed: an
        // identical second uninterrupted run (same init, same seed, same
        // data -- only the checkpoint writes differ, and those do not
        // touch the numerics).
        let rerun = SpecialistTrainConfig {
            out_dir: None,
            ..base.clone()
        };
        let (model_a2, report_a2) =
            train_language_specialist(init.clone(), &data, &rerun, &device).unwrap();
        let loss_noise = (report_a.last_loss - report_a2.last_loss).abs();
        let weight_noise = worst_diff(
            &param_values::<_, AB>(&model_a),
            &param_values::<_, AB>(&model_a2),
        );

        // Resume into a *fresh* model: the weights come from the file, the
        // optimizer moments and host RNG from the state beside it.
        <AB as burn::tensor::backend::Backend>::seed(&device, 99);
        let fresh = GeometricStudent::<AB>::new(&tiny_config(), &device).unwrap();
        let resumed = SpecialistTrainConfig {
            out_dir: Some(dir.join("b")),
            resume: Some(halfway.clone()),
            ..base.clone()
        };
        let (model_b, report_b) =
            train_language_specialist(fresh, &data, &resumed, &device).unwrap();
        assert_eq!(report_b.resumed_from_step, 2);
        assert_eq!(report_a.steps_taken, report_b.steps_taken);
        // The restored report state is the uninterrupted run's, bit for
        // bit: the resumed run's first loss was deserialized from the
        // checkpoint's extras, not recomputed.
        assert_eq!(
            report_a.first_loss.to_bits(),
            report_b.first_loss.to_bits(),
            "the restored report state diverged: {} vs {}",
            report_a.first_loss,
            report_b.first_loss
        );
        // The continuation is a function of the restored weights, the
        // restored data order (host RNG) and the restored optimizer
        // moments -- all exact -- plus the threaded backward's reduction
        // order, which is not. Ten times the measured noise, floored,
        // separates that scheduling sliver from a restore defect.
        let loss_divergence = (report_a.last_loss - report_b.last_loss).abs();
        let loss_bound = (loss_noise * 10.0).max(2e-5);
        println!(
            "specialist resume: measured rerun noise {loss_noise:e}, resumed divergence {loss_divergence:e}, bound {loss_bound:e}"
        );
        assert!(
            loss_divergence <= loss_bound,
            "resumed loss diverged {}, above the bound {} from measured run-to-run noise {}",
            report_b.last_loss,
            loss_bound,
            loss_noise
        );
        let mean_bound = ((report_a.mean_loss - report_a2.mean_loss).abs() * 10.0).max(2e-5);
        assert!(
            (report_a.mean_loss - report_b.mean_loss).abs() <= mean_bound,
            "resumed mean loss diverged {} vs {}",
            report_a.mean_loss,
            report_b.mean_loss
        );
        // The final weights meet the same discipline, relative to their
        // scale: they carry ulp-level scheduling noise, nothing more.
        let pa = param_values::<_, AB>(&model_a);
        let pb = param_values::<_, AB>(&model_b);
        assert_eq!(pa.len(), pb.len(), "parameter count changed across resume");
        let scale = pa.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-3);
        let weight_divergence = worst_diff(&pa, &pb);
        let weight_bound = (weight_noise * 10.0).max(1e-5 * scale);
        assert!(
            weight_divergence <= weight_bound,
            "resumed weights diverged by {weight_divergence}, above the bound {weight_bound} \
             from measured run-to-run noise {weight_noise} at scale {scale}"
        );
        // The resumed run wrote its final state at the same step.
        let state_b = crate::checkpoint::TrainState::for_model(
            &report_b.checkpoint.clone().expect("final checkpoint"),
        )
        .unwrap()
        .expect("training state beside the model");
        assert_eq!(state_b.step, 4);

        // A different specialist may not continue the run.
        let wrong_language = SpecialistTrainConfig {
            steps: 6,
            resume: Some(halfway),
            language: Some(CodeLanguage::Python),
            ..base
        };
        let err = train_language_specialist(init, &data, &wrong_language, &device)
            .unwrap_err()
            .to_string();
        assert!(err.contains("language"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // -- field boxes: non-coding specialist routing -------------------------

    #[test]
    fn test_field_boxes_default_empty_and_old_sidecars_parse() {
        // The six coding checkpoints were written before `field_boxes`
        // existed: their sidecars are the default config's JSON minus the
        // key, and they must parse back to exactly the default config.
        let default = StudentConfig::default();
        assert!(default.field_boxes.is_empty());
        let mut old_wire = serde_json::to_value(&default).unwrap();
        old_wire.as_object_mut().unwrap().remove("field_boxes");
        let parsed: StudentConfig = serde_json::from_value(old_wire).unwrap();
        assert_eq!(parsed, default);
        // A minimal hand-written config gets the same default.
        let minimal: StudentConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(minimal, default);
        // And the field round-trips when set.
        let with_fields = StudentConfig {
            field_boxes: vec!["pentest".to_string()],
            ..StudentConfig::default()
        };
        let text = serde_json::to_string(&with_fields).unwrap();
        assert_eq!(
            serde_json::from_str::<StudentConfig>(&text).unwrap(),
            with_fields
        );
    }

    #[test]
    fn test_field_boxes_extend_the_spec_and_the_cost_accounts_for_them() {
        let device = Default::default();
        let config = StudentConfig {
            field_boxes: vec!["pentest".to_string(), "dfir".to_string()],
            ..tiny_config()
        };
        config.validate().unwrap();
        assert_eq!(config.num_boxes(), CODING_LANGUAGES.len() + 2);
        assert_eq!(config.field_box_index("pentest"), Some(6));
        assert_eq!(config.field_box_index("dfir"), Some(7));
        assert_eq!(config.field_box_index("alchemy"), None);

        // The MoSME keeps the coding prefix and appends one box per field.
        let spec = config.mosme_spec().unwrap();
        assert_eq!(spec.boxes.len(), 8);
        assert_eq!(spec.boxes[6].id, "field:pentest");
        assert_eq!(spec.boxes[7].id, "field:dfir");
        assert_eq!(spec.boxes[6].experts.len(), config.experts_per_box);

        // The model builds with the extra boxes, the shape accounting counts
        // them honestly, and the routing readout reports all eight boxes.
        let student = GeometricStudent::<B>::new(&config, &device).unwrap();
        let cost = config.cost().unwrap();
        assert_eq!(
            cost.total_params,
            student.num_params(),
            "shape accounting drifted from the module ({} vs {})",
            cost.total_params,
            student.num_params()
        );
        let routing = student.route_languages(tokens(&device, 2, 8)).unwrap();
        assert_eq!(routing.traffic.len(), 8);
        let sum: f32 = routing.traffic.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "traffic sums to {sum}");
        assert_eq!(routing.top_box.len(), 16);
        assert!(routing.top_box.iter().all(|&idx| idx < 8));
        assert_eq!(routing.top_language.len(), 16);

        // Bad names and duplicates are config errors, not runtime surprises.
        let bad = StudentConfig {
            field_boxes: vec!["Pentest".to_string()],
            ..tiny_config()
        };
        assert!(bad.validate().is_err());
        let dup = StudentConfig {
            field_boxes: vec!["pentest".to_string(), "pentest".to_string()],
            ..tiny_config()
        };
        assert!(dup.validate().is_err());
    }

    #[test]
    fn test_box_target_resolution_rules() {
        let config = StudentConfig {
            field_boxes: vec!["pentest".to_string()],
            ..tiny_config()
        };
        // A configured language wins even when a configured field is also
        // present: the coding path is unchanged.
        assert_eq!(
            config.box_target(Some(CodeLanguage::Rust), Some("pentest")),
            Some(BoxTarget::Language(0))
        );
        assert_eq!(
            BoxTarget::Language(0).box_index(),
            CodeLanguage::Rust.index()
        );
        // A configured field supervises its own box, after the coding boxes.
        assert_eq!(
            config.box_target(None, Some("pentest")),
            Some(BoxTarget::Field(CODING_LANGUAGES.len()))
        );
        assert_eq!(
            BoxTarget::Field(CODING_LANGUAGES.len()).box_index(),
            CODING_LANGUAGES.len()
        );
        // An unknown field — like an unconfigured language — supervises
        // nothing.
        assert_eq!(config.box_target(None, Some("alchemy")), None);
        assert_eq!(config.box_target(Some(CodeLanguage::Go), None), None);
        assert_eq!(config.box_target(None, None), None);

        // ...and "supervises nothing" is spelled as the aux being off.
        let device = Default::default();
        let student = GeometricStudent::<B>::new(&config, &device).unwrap();
        crate::tensor_ext::force_initialization(&student);
        let step = student
            .specialist_step_boxes(
                tokens(&device, 1, 8),
                Tensor::ones([1, 8], &device),
                &[None],
                0.5,
            )
            .unwrap();
        assert!(step.router_aux.is_nan());
        assert!(step.target_top1.is_nan());
    }

    /// Traces for a non-coding field: the pair-file contract
    /// (`prompt`/`completion`/`field`), with every completion carrying the
    /// field's marker bytes so the routing signal is learnable at toy scale.
    fn field_trace_fixture(dir: &std::path::Path, field: &str, lines: usize) -> std::path::PathBuf {
        let path = dir.join(format!("{field}-traces.jsonl"));
        let mut out = Vec::new();
        for k in 0..lines {
            let marker: String = (0..24)
                .map(|i| (200 + (i + k) % 24) as u8 as char)
                .collect();
            out.push(format!(
                "{{\"prompt\":\"scan the host:\",\"completion\":\"{marker}\",\"field\":\"{field}\"}}"
            ));
        }
        std::fs::write(&path, format!("{}\n", out.join("\n"))).unwrap();
        path
    }

    #[test]
    fn test_field_supervision_trains_the_field_box() {
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let dir = unique_tmp("fieldbox");
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = field_trace_fixture(&dir, "pentest", 8);
        let data = LanguageBatch::load(&fixture, None, 32, 5).unwrap();
        assert_eq!(data.len(), 8);
        // The loader kept the field key on every line.
        for i in 0..data.len() {
            assert_eq!(data.field_of(i), Some("pentest"));
            assert_eq!(data.language_of(i), None);
        }

        let mut field_config = tiny_config();
        field_config.field_boxes = vec!["pentest".to_string()];
        <AB as burn::tensor::backend::Backend>::seed(&device, 3);
        let student = GeometricStudent::<AB>::new(&field_config, &device).unwrap();
        crate::tensor_ext::force_initialization(&student);
        let config = SpecialistTrainConfig {
            steps: 40,
            batch_size: 4,
            lr: 5e-3,
            router_aux_weight: 1.0,
            log_every: 0,
            seed: 11,
            ..SpecialistTrainConfig::default()
        };
        let (student, report) =
            train_language_specialist(student, &data, &config, &device).unwrap();
        assert_eq!(report.steps_taken, 40);
        println!(
            "toy field run: loss {:.4} -> {:.4}, router aux {:.4} -> {:.4}, top1 {:.3} -> {:.3}",
            report.first_loss,
            report.last_loss,
            report.first_router_aux,
            report.last_router_aux,
            report.first_target_top1,
            report.last_target_top1,
        );
        // The aux fired (a field line supervises a box now) and the router
        // moved onto the field box, decisively above chance (1/7).
        assert!(report.first_router_aux.is_finite());
        let chance = 1.0 / 7.0;
        assert!(
            report.last_target_top1 > 0.5,
            "field-box top1 {:.3} should be decisively above chance {chance:.3}",
            report.last_target_top1
        );

        // Held-out proof: fresh marker tokens route to `field:pentest`
        // (box index 6), read from the gates alone.
        let marker: String = (0..24)
            .map(|i| (200 + (i + 100) % 24) as u8 as char)
            .collect();
        let mut ids = vec![Special::Bos.id()];
        ids.extend(ByteTokenizer::new().encode(&marker));
        let n = ids.len();
        let tokens = Tensor::<AB, 1, Int>::from_ints(
            ids.iter()
                .map(|t| i64::from(*t))
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([1, n]);
        let routing = student.route_languages(tokens).unwrap();
        let field_idx = field_config.field_box_index("pentest").unwrap();
        let hits = routing.top_box.iter().filter(|&&t| t == field_idx).count();
        assert!(
            hits as f32 > 0.5 * n as f32,
            "held-out field routing: {hits}/{n} tokens on field:pentest"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The Phase-2 proof for the extended languages: a config that lists all
    /// ten languages trains a go slice whose router supervision lands on
    /// `coding:go` (box 6) exactly as it does for a shipped language.
    #[test]
    fn test_extended_language_supervision_trains_the_go_box() {
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let dir = unique_tmp("gobox");
        std::fs::create_dir_all(&dir).unwrap();
        let lines: Vec<String> = (0..8)
            .map(|k| trace_line("go", "code: ", &marker_completion(CodeLanguage::Go, 24, k)))
            .collect();
        let fixture = write_lines(&dir, "train.jsonl", &lines);
        let data = LanguageBatch::load(&fixture, Some(CodeLanguage::Go), 32, 5).unwrap();
        assert_eq!(data.len(), 8);
        for i in 0..data.len() {
            assert_eq!(data.language_of(i), Some(CodeLanguage::Go));
        }

        let config = StudentConfig {
            coding_boxes: CodeLanguage::KNOWN
                .iter()
                .map(|l| l.name().to_string())
                .collect(),
            ..tiny_config()
        };
        let go_box = config
            .box_target(Some(CodeLanguage::Go), None)
            .unwrap()
            .box_index();
        assert_eq!(go_box, 6);
        <AB as burn::tensor::backend::Backend>::seed(&device, 3);
        let student = GeometricStudent::<AB>::new(&config, &device).unwrap();
        crate::tensor_ext::force_initialization(&student);
        let train = SpecialistTrainConfig {
            steps: 40,
            batch_size: 4,
            lr: 5e-3,
            router_aux_weight: 1.0,
            log_every: 0,
            seed: 11,
            ..SpecialistTrainConfig::default()
        };
        let (student, report) = train_language_specialist(student, &data, &train, &device).unwrap();
        assert_eq!(report.steps_taken, 40);
        // The aux fired and moved the probe onto coding:go, decisively above
        // chance (1/10 with ten coding boxes).
        assert!(report.first_router_aux.is_finite());
        let chance = 1.0 / 10.0;
        assert!(
            report.last_target_top1 > 0.5,
            "go-box top1 {:.3} should be decisively above chance {chance:.3}",
            report.last_target_top1
        );

        // Held-out proof: fresh go marker tokens route to box 6, and the
        // per-language readout decodes them as Go.
        let mut ids = vec![Special::Bos.id()];
        ids.extend(ByteTokenizer::new().encode(&marker_completion(CodeLanguage::Go, 8, 100)));
        let n = ids.len();
        let tokens = Tensor::<AB, 1, Int>::from_ints(
            ids.iter()
                .map(|t| i64::from(*t))
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([1, n]);
        let routing = student.route_languages(tokens).unwrap();
        assert_eq!(routing.traffic.len(), 10);
        let box_hits = routing.top_box.iter().filter(|&&t| t == go_box).count();
        assert!(
            box_hits as f32 > 0.5 * n as f32,
            "held-out go routing: {box_hits}/{n} tokens on coding:go"
        );
        let lang_hits = routing
            .top_language
            .iter()
            .filter(|&&l| l == CodeLanguage::Go)
            .count();
        assert!(
            lang_hits as f32 > 0.5 * n as f32,
            "held-out go readout: {lang_hits}/{n} tokens decoded as go"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_field_batch_unknown_field_supervises_nothing() {
        let dir = unique_tmp("fieldunknown");
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = field_trace_fixture(&dir, "alchemy", 4);
        let data = LanguageBatch::load(&fixture, None, 32, 5).unwrap();
        // The raw field survives the load; resolution is the config's call,
        // and "alchemy" names no configured box.
        let config = StudentConfig {
            field_boxes: vec!["pentest".to_string()],
            ..tiny_config()
        };
        for i in 0..data.len() {
            assert_eq!(data.field_of(i), Some("alchemy"));
            assert_eq!(
                config.box_target(data.language_of(i), data.field_of(i)),
                None
            );
        }
        // Without the field key at all, field_of is None.
        let dir2 = unique_tmp("fieldabsent");
        std::fs::create_dir_all(&dir2).unwrap();
        let plain = synthetic_trace_fixture(&dir2, 1);
        let data = LanguageBatch::load(&plain, None, 32, 5).unwrap();
        assert!(data.field_of(0).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&dir2).unwrap();
    }

    /// Micro-batch accumulation: batch 4 as 2x2 must reproduce the direct
    /// batch-4 run's numbers to float noise (the fixture's rows all carry the
    /// same supervised-token count, so the token-share weights are exactly
    /// 1/2 and the CE gradient is the full batch's), and a `micro_batch`
    /// that does not divide `batch_size` is a config error naming the field.
    #[test]
    fn test_micro_batch_accumulation_matches_the_direct_batch() {
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let dir = unique_tmp("accum");
        let fixture = synthetic_trace_fixture(&dir, 8);
        let data = LanguageBatch::load(&fixture, None, 32, 5).unwrap();

        <AB as burn::tensor::backend::Backend>::seed(&device, 3);
        let init = GeometricStudent::<AB>::new(&tiny_config(), &device).unwrap();
        crate::tensor_ext::force_initialization(&init);
        let base = SpecialistTrainConfig {
            steps: 6,
            batch_size: 4,
            lr: 3e-4,
            router_aux_weight: 1.0,
            log_every: 0,
            seed: 11,
            ..SpecialistTrainConfig::default()
        };
        let (direct_model, direct) =
            train_language_specialist(init.clone(), &data, &base, &device).unwrap();
        let accumulated_cfg = SpecialistTrainConfig {
            micro_batch: 2,
            ..base.clone()
        };
        let (accum_model, accumulated) =
            train_language_specialist(init, &data, &accumulated_cfg, &device).unwrap();

        assert_eq!(direct.steps_taken, accumulated.steps_taken);
        assert_eq!(direct.tokens_seen, accumulated.tokens_seen);
        assert!(
            (direct.first_loss - accumulated.first_loss).abs() < 1e-3,
            "step-0 loss: direct {:.6} vs accumulated {:.6}",
            direct.first_loss,
            accumulated.first_loss
        );
        // Step 0 is pure forward numerics (same weights, same data): the
        // split graphs reduce in a different order, so the gate is float
        // noise, not bit-identity. Later steps compound that noise through
        // AdamW's normalized update (a sign flip in a near-zero gradient
        // costs a whole lr per step), so the parameter gate is a multiple of
        // the total distance a sign-flipped parameter could travel: 6 steps
        // x 3e-4 = 1.8e-3.
        assert!(
            (direct.last_loss - accumulated.last_loss).abs() < 1e-2,
            "step-5 loss: direct {:.6} vs accumulated {:.6}",
            direct.last_loss,
            accumulated.last_loss
        );
        let worst = worst_diff(&param_values(&direct_model), &param_values(&accum_model));
        println!("accumulated vs direct after 6 steps: worst param diff {worst:.3e}");
        assert!(
            worst < 5e-3,
            "accumulated and direct runs drifted by {worst} after 6 steps"
        );

        // A micro-batch that does not tile the batch is a config error.
        let bad = SpecialistTrainConfig {
            micro_batch: 3,
            ..base.clone()
        };
        let student = GeometricStudent::<AB>::new(&tiny_config(), &device).unwrap();
        let err = train_language_specialist(student, &data, &bad, &device)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(err.contains("micro_batch"), "{err}");

        // A config written before the field existed parses to the direct path.
        let parsed: SpecialistTrainConfig = serde_json::from_str(
            &serde_json::to_string(&serde_json::json!({
                "steps": 1, "batch_size": 2, "lr": 0.001, "seed": 1, "log_every": 0
            }))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(parsed.micro_batch, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
