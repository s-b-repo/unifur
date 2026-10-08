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
//!   per-language specialist boxes of [`crate::expert_index::coding_student_spec`]
//!   (`coding:rust`, `coding:python`, `coding:c`, `coding:cpp`,
//!   `coding:jsts`, `coding:web`), with micro experts per
//!   `docs/Mixture-of-Specialized-Micro-Experts.md`.
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
//!   per-language box traffic (`LanguageRouting::traffic`) and each token's
//!   most likely specialist (`LanguageRouting::top_language`) — gates only,
//!   no expert MLP is run.
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
    expert_index::{coding_student_spec, MosmeSpec, CODING_LANGUAGES},
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
/// The remaining top-10 languages (`go`, `java`, `csharp`, `php`, `ruby`,
/// `swift`, `kotlin`, `sql`, `dart`, `scala`) are documented in
/// [`crate::expert_index::PLANNED_CODING_LANGUAGES`] as later registry
/// entries — adding one does not disturb this enum or any checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeLanguage {
    Rust,
    Python,
    C,
    Cpp,
    JsTs,
    Web,
}

impl CodeLanguage {
    /// Every shipped language, in registry (box router) order.
    pub const ALL: [Self; 6] = [
        Self::Rust,
        Self::Python,
        Self::C,
        Self::Cpp,
        Self::JsTs,
        Self::Web,
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
        }
    }

    /// Parse a language key (`rust`, `python`, `c`, `cpp`, `jsts`, `web`).
    /// A few obvious aliases are accepted (`c++`, `js`, `ts`); anything else
    /// is an error naming the expected keys.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(match text.trim().to_ascii_lowercase().as_str() {
            "rust" | "rs" => Self::Rust,
            "python" | "py" => Self::Python,
            "c" => Self::C,
            "cpp" | "c++" => Self::Cpp,
            "jsts" | "js" | "ts" | "javascript" | "typescript" => Self::JsTs,
            "web" | "html" => Self::Web,
            other => anyhow::bail!(
                "unknown language '{other}' (expected rust|python|c|cpp|jsts|web; the remaining \
                 top-10 are later registry entries, see expert_index::PLANNED_CODING_LANGUAGES)"
            ),
        })
    }

    /// This language's MoSME box id, `coding:{name}`.
    pub fn box_id(self) -> String {
        format!("coding:{}", self.name())
    }

    /// Index in the box router's output (and in [`CodeLanguage::ALL`]).
    pub fn index(self) -> usize {
        CodeLanguage::ALL
            .iter()
            .position(|l| *l == self)
            .map_or(CodeLanguage::ALL.len(), |idx| idx)
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
        anyhow::ensure!(
            (1..=CODING_LANGUAGES.len()).contains(&self.top_box),
            "top_box must be in [1, {}], got {}",
            CODING_LANGUAGES.len(),
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
        Ok(())
    }

    /// The per-language specialist boxes this config builds.
    pub fn mosme_spec(&self) -> anyhow::Result<MosmeSpec> {
        coding_student_spec(self.experts_per_box, self.top_box, self.top_expert)
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
        let k = CODING_LANGUAGES.len();
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

/// The router-which-knows readout: which language specialist would handle
/// each token, read from the gates alone — no expert MLP ran.
#[derive(Debug, Clone)]
pub struct LanguageRouting {
    /// Mean box gate per language over every token, parallel to
    /// [`CodeLanguage::ALL`]. A distribution: every gate row sums to 1, so
    /// their mean does too.
    pub traffic: Vec<f32>,
    /// Each token's most likely specialist, flattened `[b * n]`.
    pub top_language: Vec<CodeLanguage>,
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
    /// stream.
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
    ) -> (Tensor<B, 3>, RefineTrace) {
        let z0 = self.init_slots(h);
        let (_, z, trace) = self.geom.refine(h.clone(), z0, 0.0, refine_override);
        let pooled = z.mean_dim(1); // [b, 1, dim]
        let delta = self.readout.forward(pooled); // [b, 1, hidden]
        (h.clone() + delta, trace)
    }

    /// The gates over the per-language boxes for already-computed hidden
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
            .reshape([CODING_LANGUAGES.len()])
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let top_idx: Vec<i64> = gates
            .box_gates
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
            top_language.push(CodeLanguage::ALL.get(idx).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "the box router produced box index {idx}, but {} boxes exist",
                    CodeLanguage::ALL.len()
                )
            })?);
        }
        Ok(LanguageRouting {
            traffic,
            top_language,
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
        let (hidden, refine) = self.geometric_pass(&hidden, refine_override);
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
        let [b, n] = tokens.dims();
        anyhow::ensure!(n >= 2, "next-token loss needs at least two positions");
        anyhow::ensure!(
            mask.dims() == [b, n],
            "the loss mask must be shaped like the tokens"
        );
        anyhow::ensure!(
            languages.len() == b,
            "languages holds {} rows for a batch of {b}",
            languages.len()
        );
        let device = tokens.device();
        let (out, states) =
            self.trunk
                .forward_span_states(tokens.clone(), 0..self.trunk.num_layers(), None);
        let trunk_hidden = states.last().cloned().ok_or_else(|| {
            anyhow::anyhow!("the trunk produced no hidden states for an empty span")
        })?;
        let (hidden, refine) = self.geometric_pass(&trunk_hidden, None);
        let logits = self.trunk.logits_from_hidden(hidden);

        // --- masked completion CE (the masked path of lm::objective) ------
        let flat_logits = logits
            .narrow(1, 0, n - 1)
            .reshape([b * (n - 1), self.config.vocab_size]);
        let targets = tokens.clone().narrow(1, 1, n - 1).reshape([b * (n - 1), 1]);
        let log_probs = log_softmax(flat_logits, 1);
        let nll = -log_probs.gather(1, targets.clone()).reshape([b * (n - 1)]);
        let pad = Tensor::<B, 1, Int>::full([b * (n - 1)], i64::from(Special::Pad.id()), &device);
        let pad_keep = targets
            .reshape([b * (n - 1)])
            .equal(pad)
            .bool_not()
            .float();
        let completion_keep = mask
            .narrow(1, 1, n - 1)
            .reshape([b * (n - 1)])
            .clamp_min(0.0)
            * pad_keep;
        let counted_raw = completion_keep.clone().sum();
        let ce = (nll * completion_keep).sum() / counted_raw.clone().clamp_min(1.0);
        let tokens_counted = counted_raw.into_scalar() as usize;
        let completion_ce: f32 = ce.clone().into_scalar();

        // --- the router-which-knows auxiliary -----------------------------
        let any_language = languages.iter().any(Option::is_some);
        let mut loss = ce;
        let (mut router_aux_value, mut target_traffic, mut target_top1) =
            (f32::NAN, f32::NAN, f32::NAN);
        if any_language {
            let cond = Tensor::<B, 2>::zeros([b, self.config.cond_hidden_size], &device);
            let input = self.probe.router_input(&trunk_hidden, &cond);
            let gates = self.probe.route(input);
            let mut target_flat = Vec::with_capacity(b * n);
            let mut has_flat = Vec::with_capacity(b * n);
            for row in languages {
                for _ in 0..n {
                    target_flat.push(row.map_or(0, CodeLanguage::index) as i64);
                    has_flat.push(if row.is_some() { 1.0f32 } else { 0.0 });
                }
            }
            let target = Tensor::<B, 1, Int>::from_ints(target_flat.as_slice(), &device)
                .reshape([b * n, 1]);
            let has = Tensor::<B, 1>::from_floats(has_flat.as_slice(), &device).reshape([b * n]);
            let pad_full = Tensor::<B, 1, Int>::full([b * n], i64::from(Special::Pad.id()), &device);
            let token_keep = tokens
                .clone()
                .reshape([b * n])
                .equal(pad_full)
                .bool_not()
                .float();
            let aux_keep = token_keep * has;
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
}

/// A per-language training set loaded from a traces JSONL.
///
/// Two on-disk shapes are read: the trace schema (`prompt` + `completion`,
/// with a `language` key) and the older SFT schema (`instruction` +
/// `response`, with `lang`), whose lines get `language = unknown`. A line
/// whose language names no shipped box (e.g. `go`, a *planned* registry
/// entry) is likewise unknown: it can train the trunk in an unfiltered run
/// but cannot supervise a routing box. The load is deterministic — the
/// example order is a seeded ChaCha shuffle, the same RNG discipline
/// `crate::train` keeps — so two loads of the same file at the same seed
/// agree bit for bit, and a resumed run sees the data the first run saw.
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
        for (idx, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            lines_read += 1;
            let line_no = idx + 1;
            let value: serde_json::Value = serde_json::from_str(line).with_context(|| {
                format!("{} line {line_no} is not valid JSON", path.display())
            })?;
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
            // A language with no shipped box (a planned registry entry, a
            // typo, "unknown") supervises no routing box: it is unknown.
            let language = raw_language.and_then(|raw| CodeLanguage::parse(raw).ok());
            *seen.entry(raw_language.unwrap_or("unknown").to_string())
                .or_default() += 1;
            if let Some(want) = filter {
                if language != Some(want) {
                    continue;
                }
            }

            let prompt_tokens = tokenizer.encode(prompt);
            let completion_tokens = tokenizer.encode(completion);
            let mut tokens =
                Vec::with_capacity((2 + prompt_tokens.len() + completion_tokens.len()).min(context));
            let mut mask = Vec::with_capacity(tokens.capacity());
            tokens.push(Special::Bos.id());
            mask.push(0.0);
            tokens.extend_from_slice(&prompt_tokens);
            mask.extend(std::iter::repeat(0.0).take(prompt_tokens.len()));
            tokens.extend_from_slice(&completion_tokens);
            mask.extend(std::iter::repeat(1.0).take(completion_tokens.len()));
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
            });
        }
        let languages_seen: Vec<(String, usize)> = seen.into_iter().collect();
        anyhow::ensure!(
            !examples.is_empty(),
            "{} holds no usable {}examples ({} lines read, {} skipped with no supervised \
             completion token; languages seen: {})",
            path.display(),
            filter.map_or(String::new(), |l| format!("'{}' ", l.name())),
            lines_read,
            skipped_no_target,
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
        })
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

    /// What a run trained on, hashed once, for the training state.
    pub fn identity(&self) -> anyhow::Result<DatasetIdentity> {
        DatasetIdentity::of_path(
            &self.path,
            format!(
                "language traces {} (filter {})",
                self.path.display(),
                self.filter.map_or("none".to_string(), |l| l.name().to_string())
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
    /// softmax against the known language); NaN when no example in the
    /// batch carried a language, which is how "the aux is off" is spelled.
    pub router_aux: f32,
    /// Mean probe box probability on the true language over the supervised
    /// tokens; NaN alongside `router_aux`.
    pub target_traffic: f32,
    /// Fraction of the supervised tokens whose top-1 box *is* the true
    /// language — the router-which-knows readout (chance is 1/6).
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
    /// Mean probe box probability on the true language, first and last step.
    pub first_target_traffic: f32,
    pub last_target_traffic: f32,
    /// Router-which-knows: fraction of supervised tokens whose top-1 box is
    /// the true language, at the first and last step. Chance is 1/6.
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
            checkpoint::same_datasets(&state.datasets, &[dataset_identity.clone()]),
            "refusing to resume: the checkpoint was trained on [{}], this run opened [{}]",
            checkpoint::describe_datasets(&state.datasets),
            checkpoint::describe_datasets(&[dataset_identity.clone()])
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
        let mut languages = Vec::with_capacity(config.batch_size);
        for _ in 0..config.batch_size {
            // With-replacement draws, exactly as CorpusMix samples windows:
            // stateless in everything but the serialized host RNG, which is
            // what makes the resumed run draw what the uninterrupted one did.
            let idx = rng.random_range(0..data.len());
            flat.extend(data.tokens(idx).iter().map(|t| i64::from(*t)));
            mask_flat.extend_from_slice(data.mask(idx));
            languages.push(data.language_of(idx));
        }
        let tokens = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
            .reshape([config.batch_size, context]);
        let mask = Tensor::<B, 1>::from_floats(mask_flat.as_slice(), device)
            .reshape([config.batch_size, context]);

        let out = model.specialist_step(tokens, mask, &languages, config.router_aux_weight)?;
        let loss_value: f32 = out.loss.clone().into_scalar();
        if out.tokens_counted == 0 {
            report.steps_skipped += 1;
            println!("step {step}: batch has no supervised target, step discarded");
            continue;
        }
        if !loss_value.is_finite() {
            // The same policy as the other loops: a pathological step is
            // discarded, not clipped into something that looks fine.
            report.steps_skipped += 1;
            println!("step {step}: non-finite loss {loss_value}, step discarded");
            continue;
        }
        let mut grads = trainable.gradients::<B, _>(&mut out.loss.backward(), &model);
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
            report.first_ce = out.completion_ce;
            report.first_router_aux = out.router_aux;
            report.first_target_traffic = out.target_traffic;
            report.first_target_top1 = out.target_top1;
        }
        report.last_loss = loss_value;
        report.last_ce = out.completion_ce;
        report.last_router_aux = out.router_aux;
        report.last_target_traffic = out.target_traffic;
        report.last_target_top1 = out.target_top1;
        report.steps_taken += 1;
        report.tokens_seen += out.tokens_counted;
        loss_sum += f64::from(loss_value);

        if config.log_every > 0 && (step % config.log_every == 0 || step + 1 == config.steps) {
            let mut line = format!(
                "step {step}: loss {:.4} ce {:.4} balance {:.4}",
                loss_value, out.completion_ce, out.balance
            );
            if out.router_aux.is_finite() {
                line.push_str(&format!(
                    " | router aux {:.4} p(lang) {:.3} top1 {:.3}",
                    out.router_aux, out.target_traffic, out.target_top1
                ));
            }
            if let (Some(first), Some(last)) = (out.energy.first(), out.energy.last()) {
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
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn unique_tmp(stem: &str) -> std::path::PathBuf {
        let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("student-{stem}-{}-{unique}", std::process::id()))
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
        for language in CodeLanguage::ALL {
            assert_eq!(CodeLanguage::parse(language.name()).unwrap(), language);
            assert_eq!(language.box_id(), format!("coding:{}", language.name()));
        }
        assert_eq!(CodeLanguage::parse("c++").unwrap(), CodeLanguage::Cpp);
        assert_eq!(CodeLanguage::parse("TS").unwrap(), CodeLanguage::JsTs);
        assert_eq!(CodeLanguage::parse(" web ").unwrap(), CodeLanguage::Web);
        let err = CodeLanguage::parse("cobol").unwrap_err().to_string();
        assert!(err.contains("cobol"), "{err}");
        // The registry order is the box router's order: index() agrees.
        for (idx, language) in CodeLanguage::ALL.iter().enumerate() {
            assert_eq!(language.index(), idx);
        }
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
        assert_eq!(routing.traffic.len(), CodeLanguage::ALL.len());
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
    fn marker_completion(language: CodeLanguage, len: usize, salt: usize) -> String {
        let base = 48 + 8 * language.index() as u8;
        (0..len)
            .map(|k| (base + ((k * 3 + salt) % 8) as u8) as char)
            .collect()
    }

    fn synthetic_trace_fixture(dir: &std::path::Path, per_language: usize) -> std::path::PathBuf {
        let mut lines = Vec::new();
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

    #[test]
    fn test_language_batch_filters_masks_truncates_and_is_deterministic() {
        let dir = unique_tmp("batch");
        let long_prompt = "p".repeat(64);
        let lines = vec![
            trace_line("rust", "fn ", "x"),
            trace_line("python", "def ", "y"),
            trace_line("rust", "let ", "zz"),
            trace_line("go", "package ", "main"), // a planned language: unknown
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
            .find(|idx| batch.tokens(idx)[1] == b'f' as u16)
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
        let order = |b: &LanguageBatch| (0..b.len()).map(|i| b.tokens(i).to_vec()).collect::<Vec<_>>();
        assert_eq!(order(&batch), order(&again));

        let many: Vec<String> = (0..12)
            .map(|k| trace_line("rust", "code: ", &format!("line {k}")))
            .collect();
        let many_path = write_lines(&dir, "many.jsonl", &many);
        let order_of = |seed: u64| {
            let b = LanguageBatch::load(&many_path, None, 32, seed).unwrap();
            (0..b.len()).map(|i| b.tokens(i).to_vec()).collect::<Vec<_>>()
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
        assert_eq!(batch.language_of(0), None);
        assert_eq!(batch.language_of(1), Some(CodeLanguage::Python));
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
            ids.iter().map(|t| i64::from(*t)).collect::<Vec<_>>().as_slice(),
            &device,
        )
        .reshape([1, context]);
        let mask_t = Tensor::<B, 1>::from_floats(mask.as_slice(), &device).reshape([1, context]);

        let step = student
            .specialist_step(tokens.clone(), mask_t.clone(), &[Some(CodeLanguage::Rust)], 0.5)
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
            .specialist_step(tokens.clone(), mask_t.clone(), &[Some(CodeLanguage::Rust)], 0.0)
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
            steps: 30,
            batch_size: 6,
            lr: 5e-3,
            router_aux_weight: 0.5,
            log_every: 0,
            seed: 11,
            ..SpecialistTrainConfig::default()
        };
        let (student, report) =
            train_language_specialist(student, &data, &config, &device).unwrap();
        assert_eq!(report.steps_taken, 30);
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

        // The router-which-knows proof at toy scale: held-out prompts (the
        // shared prompt plus *fresh* marker bytes per language) must put
        // the top-1 box on the true language above chance (1/6).
        let mut held_out = crate::langdata::RouterAgreement::default();
        for language in CodeLanguage::ALL {
            let text = format!("code: {}", marker_completion(language, 8, 100));
            let mut ids = vec![Special::Bos.id()];
            ids.extend(ByteTokenizer::new().encode(&text));
            let n = ids.len();
            let tokens = Tensor::<AB, 1, Int>::from_ints(
                ids.iter().map(|t| i64::from(*t)).collect::<Vec<_>>().as_slice(),
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
    fn test_specialist_resume_is_bit_identical() {
        // Mirrors tests/resume.rs's LM case, in-file: neither the forward,
        // the backward nor AdamW draws from the (process-global) device
        // stream, and the trainer reseeds it per step anyway, so two runs
        // from the same in-memory init are bit-comparable even with the
        // unit-test suite running concurrently.
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

        // Resume into a *fresh* model: the weights come from the file.
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
        assert_eq!(
            report_a.last_loss.to_bits(),
            report_b.last_loss.to_bits(),
            "{} vs {}",
            report_a.last_loss,
            report_b.last_loss
        );
        assert_eq!(
            crate::checkpoint::canonical_hash_hex(&model_a),
            crate::checkpoint::canonical_hash_hex(&model_b),
            "weights differ after resume"
        );
        assert_eq!(
            report_a.checkpoint.unwrap().file_name(),
            report_b.checkpoint.unwrap().file_name(),
            "content-addressed names embed the weights hash"
        );
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
}
