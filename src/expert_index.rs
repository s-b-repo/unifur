//! The expert index: the manifest an inference engine reads to route
//! **without loading weights** (roadmap Phase 18).
//!
//! A mixture of specialized micro experts is organized as *boxes* of experts —
//! a `coding` box holding a Rust expert, a Python expert and a secure-code
//! expert; a `cybersecurity` box holding its own specialists. Routing is
//! two-level: pick box(es), then pick expert(s) within them.
//!
//! Names, domains and tags cannot live inside the model. Burn gives `String`,
//! `bool` and `usize` fields of a `#[derive(Module)]` struct an *empty* record,
//! so **metadata stored in a module does not survive a checkpoint round trip**.
//! That is not a limitation to work around but a reason the manifest should be
//! a separate artifact anyway: an engine deciding *which* expert to load must
//! be able to read the catalogue before it has loaded anything.
//!
//! This module therefore has **no Burn dependency at all** — it is plain data
//! plus `serde`. An external routing service can depend on it alone.
//!
//! # Two documents
//!
//! - [`MosmeSpec`] is the *authoring* form: the boxes and experts a human
//!   writes down, with labels and tags. It is input.
//! - [`ExpertIndex`] is the *emitted* form: the spec plus measured shapes,
//!   parameter counts, content hashes and the id of the checkpoint the weights
//!   live in. It is output, written next to the checkpoint.
//!
//! Keeping them apart matters because the spec is hand-editable and the index
//! is not: the index makes claims about weights that only the trainer can
//! substantiate, so its per-expert hashes are measured from the live module
//! rather than copied from the spec.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};

/// Version of the on-disk index format. Bump on any breaking change.
pub const INDEX_SCHEMA_VERSION: u32 = 1;

/// Which architectural level a set of experts lives at.
///
/// The three levels are genuinely different mechanisms with different call
/// signatures, and an index describes exactly one of them. See
/// [`crate::mosme`] for why they are not unified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteKind {
    /// Low-rank adapters over one shared frozen `Linear`.
    Adapter,
    /// Complete micro-models, mixed in probability space.
    Model,
    /// Feed-forward sub-layers inside a transformer block.
    Mlp,
}

impl SiteKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Adapter => "adapter",
            Self::Model => "model",
            Self::Mlp => "mlp",
        }
    }
}

/// Relative weights of the two levels of the load-balancing loss.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BalanceWeights {
    pub box_level: f64,
    pub expert_level: f64,
    /// Router z-loss weight (ST-MoE). Penalizes large routing logits, which
    /// the balance loss cannot see: the softmax is invariant to a per-row
    /// constant shift, so nothing else in the objective stops them drifting.
    ///
    /// `#[serde(default)]` so an index written before this field existed still
    /// loads — and defaults to the ST-MoE value rather than to zero, because a
    /// stored index describes a *router*, and a router without this term is the
    /// one that goes numerically unstable.
    #[serde(default = "default_z_level")]
    pub z_level: f64,
}

fn default_z_level() -> f64 {
    1e-3
}

impl Default for BalanceWeights {
    fn default() -> Self {
        Self {
            box_level: 1.0,
            expert_level: 1.0,
            z_level: default_z_level(),
        }
    }
}

/// How the router behaves. Recorded so an engine can reproduce the decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingSpec {
    pub cond_size: usize,
    pub hidden_size: usize,
    pub route_on_tokens: bool,
    pub top_box: usize,
    pub top_expert: usize,
    #[serde(default)]
    pub balance: BalanceWeights,
}

/// Shape of one expert, discriminated by the level it lives at.
///
/// Serialized with an explicit `kind` tag rather than serde's default
/// externally-tagged encoding, so an engine written in another language does
/// not have to know how Rust encodes enums.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExpertKind {
    Adapter {
        rank: usize,
        alpha: f64,
        in_features: usize,
        out_features: usize,
    },
    Model {
        num_blocks: usize,
        num_hidden_layers: usize,
        hidden_size: usize,
        num_labels: usize,
    },
    Mlp {
        hidden_size: usize,
        intermediate_size: usize,
    },
}

impl ExpertKind {
    pub fn site(&self) -> SiteKind {
        match self {
            Self::Adapter { .. } => SiteKind::Adapter,
            Self::Model { .. } => SiteKind::Model,
            Self::Mlp { .. } => SiteKind::Mlp,
        }
    }
}

/// Where an expert's weights are, precisely enough to fetch them alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightLocator {
    /// Checkpoint file relative to the index; `None` means the host checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<PathBuf>,
    /// Burn module path, e.g. `vit.layers.1.mlp.boxes.0.experts.2`.
    pub param_path: String,
    /// Content hash of this expert's parameters alone.
    pub sha256: String,
}

/// One expert.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertEntry {
    /// Globally stable id, conventionally `<box_id>/<name>` — `coding/rust`.
    pub id: String,
    pub label: String,
    /// Position within this box's expert router output.
    pub index: usize,
    /// Position in the flattened order `sum_{i<box} E_i + index`.
    pub global_index: usize,
    /// Flattened into the entry, so the wire form is
    /// `"kind": "mlp", "hidden_size": 128, ...` rather than a nested object.
    /// An external engine should be able to read the discriminant without
    /// descending a level it has no reason to know about.
    #[serde(flatten)]
    pub kind: ExpertKind,
    /// `false` masks the router logit to `-inf`, giving an exactly zero gate.
    pub enabled: bool,
    pub num_parameters: usize,
    pub weights: WeightLocator,
    /// Free-form, for an engine's own matching: `["rust", "memory-safety"]`.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// One box: a domain grouping of experts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoxEntry {
    /// Stable identifier, e.g. `coding`.
    pub id: String,
    pub label: String,
    /// Position in the box router's output; must equal the vec position.
    pub index: usize,
    pub experts: Vec<ExpertEntry>,
}

impl BoxEntry {
    pub fn enabled_count(&self) -> usize {
        self.experts.iter().filter(|e| e.enabled).count()
    }
}

/// The manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertIndex {
    pub schema_version: u32,
    /// Content hash of the checkpoint these experts live in, binding the two
    /// artifacts together.
    pub model_id: String,
    pub site: SiteKind,
    pub routing: RoutingSpec,
    pub boxes: Vec<BoxEntry>,
}

impl ExpertIndex {
    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self).context("serialize expert index")
    }

    /// Parse and validate. An index that does not validate is never returned,
    /// so callers downstream can rely on its invariants.
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let index: Self = serde_json::from_str(text).context("parse expert index")?;
        index.validate()?;
        Ok(index)
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        self.validate()?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_json()?).with_context(|| format!("write {}", path.display()))
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn num_boxes(&self) -> usize {
        self.boxes.len()
    }

    pub fn num_experts(&self) -> usize {
        self.boxes.iter().map(|b| b.experts.len()).sum()
    }

    pub fn experts_per_box(&self) -> Vec<usize> {
        self.boxes.iter().map(|b| b.experts.len()).collect()
    }

    /// `enabled[box][expert]`, the mask the router applies.
    pub fn enabled_mask(&self) -> Vec<Vec<bool>> {
        self.boxes
            .iter()
            .map(|b| b.experts.iter().map(|e| e.enabled).collect())
            .collect()
    }

    /// Look an expert up by its global id.
    pub fn expert(&self, id: &str) -> Option<(&BoxEntry, &ExpertEntry)> {
        self.boxes
            .iter()
            .find_map(|b| b.experts.iter().find(|e| e.id == id).map(|e| (b, e)))
    }

    pub fn resolve(&self, box_idx: usize, expert_idx: usize) -> Option<&ExpertEntry> {
        self.boxes.get(box_idx)?.experts.get(expert_idx)
    }

    /// Structural invariants every consumer is entitled to assume.
    ///
    /// The `all disabled` check is not cosmetic: masking every logit in a box
    /// to `-inf` makes the softmax denominator zero and yields NaN, so an
    /// index that permitted it would poison a run rather than misroute it.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.schema_version != INDEX_SCHEMA_VERSION {
            bail!(
                "unsupported index schema version {} (this build understands {})",
                self.schema_version,
                INDEX_SCHEMA_VERSION
            );
        }
        if self.boxes.is_empty() {
            bail!("an index needs at least one box");
        }

        let mut ids = BTreeSet::new();
        let mut box_ids = BTreeSet::new();
        let mut running = 0usize;

        for (bi, entry) in self.boxes.iter().enumerate() {
            if entry.index != bi {
                bail!(
                    "box '{}' claims index {} but sits at {bi}",
                    entry.id,
                    entry.index
                );
            }
            if !box_ids.insert(entry.id.as_str()) {
                bail!("duplicate box id '{}'", entry.id);
            }
            if entry.experts.is_empty() {
                bail!("box '{}' has no experts", entry.id);
            }
            if entry.enabled_count() == 0 {
                bail!(
                    "every expert in box '{}' is disabled; the router softmax would be NaN",
                    entry.id
                );
            }
            for (ei, expert) in entry.experts.iter().enumerate() {
                if expert.index != ei {
                    bail!(
                        "expert '{}' claims index {} but sits at {ei}",
                        expert.id,
                        expert.index
                    );
                }
                if expert.global_index != running {
                    bail!(
                        "expert '{}' claims global index {} but the flattened order gives {running}",
                        expert.id,
                        expert.global_index
                    );
                }
                if !ids.insert(expert.id.as_str()) {
                    bail!("duplicate expert id '{}'", expert.id);
                }
                if expert.kind.site() != self.site {
                    bail!(
                        "expert '{}' is a {} expert in a {} index",
                        expert.id,
                        expert.kind.site().name(),
                        self.site.name()
                    );
                }
                running += 1;
            }
        }

        if self.routing.top_box == 0 || self.routing.top_expert == 0 {
            bail!("top_box and top_expert must be at least 1");
        }
        Ok(())
    }

    /// Human-readable table for `dblocks experts`.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "site={}  model={}  boxes={}  experts={}  top_box={}  top_expert={}\n",
            self.site.name(),
            &self.model_id[..self.model_id.len().min(16)],
            self.num_boxes(),
            self.num_experts(),
            self.routing.top_box,
            self.routing.top_expert
        ));
        for entry in &self.boxes {
            out.push_str(&format!(
                "\n[{}] {} — {} ({} enabled)\n",
                entry.index,
                entry.id,
                entry.label,
                entry.enabled_count()
            ));
            out.push_str(&format!(
                "  {:<24} {:>6} {:>9} {:>12}  {}\n",
                "expert", "index", "enabled", "params", "tags"
            ));
            for expert in &entry.experts {
                out.push_str(&format!(
                    "  {:<24} {:>6} {:>9} {:>12}  {}\n",
                    expert.id,
                    expert.index,
                    if expert.enabled { "yes" } else { "no" },
                    expert.num_parameters,
                    expert.tags.join(",")
                ));
            }
        }
        out
    }
}

// ------------------------------------------------------------ authoring --

fn default_true() -> bool {
    true
}

fn default_one() -> usize {
    1
}

/// One expert as a human writes it down.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertSpec {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl ExpertSpec {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            tags: Vec::new(),
            enabled: true,
        }
    }

    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.tags = tags.iter().map(|t| (*t).to_string()).collect();
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// One box as a human writes it down.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoxSpec {
    pub id: String,
    pub label: String,
    pub experts: Vec<ExpertSpec>,
}

impl BoxSpec {
    pub fn new(id: impl Into<String>, label: impl Into<String>, experts: Vec<ExpertSpec>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            experts,
        }
    }
}

/// The authoring document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MosmeSpec {
    pub boxes: Vec<BoxSpec>,
    #[serde(default = "default_one")]
    pub top_box: usize,
    #[serde(default = "default_one")]
    pub top_expert: usize,
    #[serde(default = "default_true")]
    pub route_on_tokens: bool,
    #[serde(default)]
    pub balance: BalanceWeights,
}

impl MosmeSpec {
    /// A single unnamed box of `n` experts.
    ///
    /// This is the configuration under which every hierarchical path reduces
    /// *exactly* to the flat [`crate::moe`] path, which is what the reduction
    /// certificate exercises.
    pub fn flat(n: usize) -> Self {
        let experts = (0..n.max(1))
            .map(|i| ExpertSpec::new(format!("flat/{i}"), format!("expert {i}")))
            .collect();
        Self {
            boxes: vec![BoxSpec::new("flat", "Flat", experts)],
            top_box: 1,
            top_expert: 1,
            route_on_tokens: true,
            balance: BalanceWeights::default(),
        }
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let spec: Self = serde_json::from_str(&text).context("parse mosme spec")?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        self.validate()?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("write {}", path.display()))
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.boxes.is_empty() {
            bail!("a spec needs at least one box");
        }
        if self.top_box == 0 || self.top_expert == 0 {
            bail!("top_box and top_expert must be at least 1");
        }
        let mut ids = BTreeSet::new();
        let mut box_ids = BTreeSet::new();
        for entry in &self.boxes {
            if !box_ids.insert(entry.id.as_str()) {
                bail!("duplicate box id '{}'", entry.id);
            }
            if entry.experts.is_empty() {
                bail!("box '{}' has no experts", entry.id);
            }
            if entry.experts.iter().all(|e| !e.enabled) {
                bail!(
                    "every expert in box '{}' is disabled; the router softmax would be NaN",
                    entry.id
                );
            }
            for expert in &entry.experts {
                if !ids.insert(expert.id.as_str()) {
                    bail!("duplicate expert id '{}'", expert.id);
                }
            }
        }
        Ok(())
    }

    pub fn experts_per_box(&self) -> Vec<usize> {
        self.boxes.iter().map(|b| b.experts.len()).collect()
    }

    pub fn num_experts(&self) -> usize {
        self.boxes.iter().map(|b| b.experts.len()).sum()
    }

    /// The shape-only projection: what the module actually needs to be built.
    pub fn layout(&self) -> BoxLayout {
        BoxLayout {
            enabled: self
                .boxes
                .iter()
                .map(|b| b.experts.iter().map(|e| e.enabled).collect())
                .collect(),
        }
    }

    /// Locate an expert by id.
    pub fn position(&self, id: &str) -> Option<(usize, usize)> {
        self.boxes
            .iter()
            .enumerate()
            .find_map(|(bi, b)| b.experts.iter().position(|e| e.id == id).map(|ei| (bi, ei)))
    }

    /// Append an expert to an existing box, for hot-swap.
    ///
    /// The new expert lands **disabled**, which is what makes growing a model
    /// an exact no-op until it is deliberately switched on.
    pub fn extended_with(&self, box_id: &str, expert: ExpertSpec) -> anyhow::Result<Self> {
        let mut next = self.clone();
        let entry = next
            .boxes
            .iter_mut()
            .find(|b| b.id == box_id)
            .with_context(|| format!("no box '{box_id}'"))?;
        entry.experts.push(expert.disabled());
        next.validate()?;
        Ok(next)
    }

    /// Append one routing box per field, after the existing boxes.
    ///
    /// This is how a spec gains boxes for *non-coding* specialists
    /// (`field:pentest` next to `coding:rust`): the appended order is the
    /// field list's order, so the existing boxes — and every checkpoint
    /// keyed by their positions — are undisturbed. Each field box holds
    /// `experts_per_box` micro experts named `field:{field}/micro_{i}`,
    /// enabled from the start (unlike [`Self::extended_with`], which lands
    /// experts disabled: a field box exists to be routed to, not staged).
    pub fn with_field_boxes(
        &self,
        fields: &[String],
        experts_per_box: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            experts_per_box >= 1,
            "experts_per_box must be at least 1, got {experts_per_box}"
        );
        validate_field_box_names(fields)?;
        let mut next = self.clone();
        for field in fields {
            next.boxes.push(BoxSpec::new(
                format!("field:{field}"),
                format!("{field} specialist"),
                (0..experts_per_box)
                    .map(|i| {
                        ExpertSpec::new(
                            format!("field:{field}/micro_{i}"),
                            format!("{field} micro expert {i}"),
                        )
                        .with_tags(&["field", field])
                    })
                    .collect(),
            ));
        }
        next.validate()?;
        Ok(next)
    }

    /// Whether `self` extends `other`: same boxes in the same order, each with
    /// at least as many experts, and the shared prefix identical.
    pub fn is_superset_of(&self, other: &MosmeSpec) -> bool {
        if self.boxes.len() < other.boxes.len() {
            return false;
        }
        other.boxes.iter().zip(&self.boxes).all(|(old, new)| {
            old.id == new.id
                && new.experts.len() >= old.experts.len()
                && old
                    .experts
                    .iter()
                    .zip(&new.experts)
                    .all(|(a, b)| a.id == b.id)
        })
    }
}

// ---------------------------------------------------- coding student registry --

/// The languages with a shipped specialist box in the coding student
/// ([`crate::student`]), as `(language key, label)` pairs. The box id of
/// language `l` is `coding:{l}`; each box holds that language's micro experts.
///
/// One box per language, rather than one `coding` box with per-language
/// experts, because the readout a caller wants is *per-language box traffic*:
/// the box router's gates over `coding:*` answer "which language specialist
/// would handle this token" directly, with no second level to decode.
pub const CODING_LANGUAGES: [(&str, &str); 6] = [
    ("rust", "Rust"),
    ("python", "Python"),
    ("c", "C"),
    ("cpp", "C++"),
    ("jsts", "JavaScript/TypeScript"),
    ("web", "Web (HTML/CSS/JS mixed)"),
];

/// The languages graduated from planning into [`crate::student::CodeLanguage`]
/// after the shipped six: trainable whenever a config's
/// [`crate::student::StudentConfig::coding_boxes`] lists them, but absent from
/// the default box set, so a config that predates them still builds the model
/// it described.
pub const EXTENDED_CODING_LANGUAGES: [(&str, &str); 4] = [
    ("go", "Go"),
    ("java", "Java"),
    ("csharp", "C#"),
    ("php", "PHP"),
];

/// The rest of the top-10 language list, documented as **later** registry
/// entries: no [`crate::student::CodeLanguage`] variant exists for these yet.
/// Adding one is a new `CodeLanguage` variant plus a move into
/// [`EXTENDED_CODING_LANGUAGES`] — the box set itself is config-driven
/// ([`crate::student::StudentConfig::coding_boxes`]), so an existing student
/// checkpoint is undisturbed either way.
pub const PLANNED_CODING_LANGUAGES: [&str; 6] = ["ruby", "swift", "kotlin", "sql", "dart", "scala"];

/// The naming rule both box lists share: the same charset a registry field
/// uses (lowercase letters, digits and dashes — see `crate::nina::validate_name`),
/// non-empty and unique within the list. `kind` names the list in the error
/// (`coding` or `field`) so a bad config names the field at fault.
fn validate_box_names(kind: &str, names: &[String]) -> anyhow::Result<()> {
    let mut seen = BTreeSet::new();
    for name in names {
        anyhow::ensure!(
            !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{kind} box name '{name}' is invalid: lowercase letters, digits and dashes only"
        );
        anyhow::ensure!(
            seen.insert(name.as_str()),
            "duplicate {kind} box name '{name}'"
        );
    }
    Ok(())
}

/// The naming rule for field boxes, kept as its own function because it is
/// the contract [`MosmeSpec::with_field_boxes`] and the configs document.
/// The box id is `field:{field}`, so a field name can never collide with a
/// `coding:{lang}` box; what remains to refuse is a name that would make an
/// unreadable id or a duplicate box.
pub fn validate_field_box_names(fields: &[String]) -> anyhow::Result<()> {
    validate_box_names("field", fields)
}

/// The naming rule for coding boxes ([`crate::student::StudentConfig`]'s
/// `coding_boxes`): the shared box charset, plus non-empty — the router
/// needs at least one box for `top_box` to mean anything.
pub fn validate_coding_box_names(languages: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(
        !languages.is_empty(),
        "coding_boxes must name at least one language"
    );
    validate_box_names("coding", languages)
}

/// The label a coding box carries for `key`: the registry's label when the
/// key is a known language, the key itself otherwise (a config may name a
/// language this build does not know; the box is still well-formed).
fn coding_label(key: &str) -> &str {
    CODING_LANGUAGES
        .iter()
        .chain(EXTENDED_CODING_LANGUAGES.iter())
        .find(|(k, _)| *k == key)
        .map_or(key, |(_, label)| *label)
}

/// The coding student's MoSME spec for an explicit box list: one box per
/// entry of `languages`, each holding `experts_per_box` micro experts named
/// `coding:<lang>/micro_<i>`. [`coding_student_spec`] is this function at the
/// default box set.
///
/// The list order *is* the box order: box `i` of the router's output is
/// `coding:{languages[i]}`, so a config's `coding_boxes` pins every
/// checkpoint's routing layout. Micro-expert sizing follows
/// `docs/Mixture-of-Specialized-Micro-Experts.md`: many tiny experts beat a
/// few large ones, so the *box* count per language stays small here and the
/// expert *width* (the trunk's `intermediate_size`, FFN/4–FFN/8 of a dense
/// feed-forward) is what makes each expert micro. Fallible for the same
/// reason `validate` is: a spec with zero experts, an unreadable name or an
/// impossible top-k is a caller mistake that should name the argument at
/// fault.
pub fn coding_spec_for(
    languages: &[String],
    experts_per_box: usize,
    top_box: usize,
    top_expert: usize,
) -> anyhow::Result<MosmeSpec> {
    anyhow::ensure!(
        experts_per_box >= 1,
        "experts_per_box must be at least 1, got {experts_per_box}"
    );
    validate_coding_box_names(languages)?;
    anyhow::ensure!(
        (1..=languages.len()).contains(&top_box),
        "top_box must be in [1, {}], got {top_box}",
        languages.len()
    );
    anyhow::ensure!(
        top_expert >= 1 && top_expert <= experts_per_box,
        "top_expert must be in [1, {experts_per_box}], got {top_expert}"
    );
    let spec = MosmeSpec {
        boxes: languages
            .iter()
            .map(|key| {
                let label = coding_label(key);
                BoxSpec::new(
                    format!("coding:{key}"),
                    format!("{label} specialist"),
                    (0..experts_per_box)
                        .map(|i| {
                            ExpertSpec::new(
                                format!("coding:{key}/micro_{i}"),
                                format!("{label} micro expert {i}"),
                            )
                            .with_tags(&["coding", key])
                        })
                        .collect(),
                )
            })
            .collect(),
        top_box,
        top_expert,
        route_on_tokens: true,
        balance: BalanceWeights::default(),
    };
    spec.validate()?;
    Ok(spec)
}

/// The coding student's MoSME spec at the default box set: one box per
/// [`CODING_LANGUAGES`] entry. This is [`coding_spec_for`] with the six
/// shipped keys; a [`crate::student::StudentConfig`] builds its spec from its
/// own `coding_boxes` list instead, which defaults to exactly these.
pub fn coding_student_spec(
    experts_per_box: usize,
    top_box: usize,
    top_expert: usize,
) -> anyhow::Result<MosmeSpec> {
    let names: Vec<String> = CODING_LANGUAGES
        .iter()
        .map(|(key, _)| (*key).to_string())
        .collect();
    coding_spec_for(&names, experts_per_box, top_box, top_expert)
}

/// Per-box expert counts and enabled flags — the spec stripped of everything
/// the tensors do not need.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxLayout {
    enabled: Vec<Vec<bool>>,
}

impl BoxLayout {
    pub fn new(enabled: Vec<Vec<bool>>) -> Self {
        Self { enabled }
    }

    pub fn num_boxes(&self) -> usize {
        self.enabled.len()
    }

    pub fn experts_in(&self, box_idx: usize) -> usize {
        self.enabled.get(box_idx).map_or(0, Vec::len)
    }

    pub fn experts_per_box(&self) -> Vec<usize> {
        self.enabled.iter().map(Vec::len).collect()
    }

    pub fn total_experts(&self) -> usize {
        self.enabled.iter().map(Vec::len).sum()
    }

    pub fn is_enabled(&self, box_idx: usize, expert_idx: usize) -> bool {
        self.enabled
            .get(box_idx)
            .and_then(|b| b.get(expert_idx))
            .copied()
            .unwrap_or(false)
    }

    pub fn enabled_in(&self, box_idx: usize) -> usize {
        self.enabled
            .get(box_idx)
            .map_or(0, |b| b.iter().filter(|e| **e).count())
    }

    pub fn enabled(&self) -> &[Vec<bool>] {
        &self.enabled
    }
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

    fn sample_spec() -> MosmeSpec {
        MosmeSpec {
            boxes: vec![
                BoxSpec::new(
                    "coding",
                    "Code",
                    vec![
                        ExpertSpec::new("coding/rust", "Rust")
                            .with_tags(&["rust", "memory-safety"]),
                        ExpertSpec::new("coding/python", "Python").with_tags(&["python"]),
                        ExpertSpec::new("coding/secure", "Secure code review")
                            .with_tags(&["security", "bugs"]),
                    ],
                ),
                // Deliberately ragged: 3 experts then 2.
                BoxSpec::new(
                    "cyber",
                    "Cybersecurity",
                    vec![
                        ExpertSpec::new("cyber/netsec", "Network security"),
                        ExpertSpec::new("cyber/malware", "Malware analysis"),
                    ],
                ),
            ],
            top_box: 1,
            top_expert: 1,
            route_on_tokens: true,
            balance: BalanceWeights::default(),
        }
    }

    fn sample_index() -> ExpertIndex {
        let spec = sample_spec();
        let mut boxes = Vec::new();
        let mut global = 0usize;
        for (bi, b) in spec.boxes.iter().enumerate() {
            let experts = b
                .experts
                .iter()
                .enumerate()
                .map(|(ei, e)| {
                    let entry = ExpertEntry {
                        id: e.id.clone(),
                        label: e.label.clone(),
                        index: ei,
                        global_index: global,
                        kind: ExpertKind::Mlp {
                            hidden_size: 32,
                            intermediate_size: 64,
                        },
                        enabled: e.enabled,
                        num_parameters: 4096,
                        weights: WeightLocator {
                            checkpoint: None,
                            param_path: format!("vit.layers.1.mlp.boxes.{bi}.experts.{ei}"),
                            sha256: format!("{:016x}", global),
                        },
                        tags: e.tags.clone(),
                    };
                    global += 1;
                    entry
                })
                .collect();
            boxes.push(BoxEntry {
                id: b.id.clone(),
                label: b.label.clone(),
                index: bi,
                experts,
            });
        }
        ExpertIndex {
            schema_version: INDEX_SCHEMA_VERSION,
            model_id: "deadbeefcafef00d".into(),
            site: SiteKind::Mlp,
            routing: RoutingSpec {
                cond_size: 8,
                hidden_size: 32,
                route_on_tokens: true,
                top_box: 1,
                top_expert: 1,
                balance: BalanceWeights::default(),
            },
            boxes,
        }
    }

    #[test]
    fn test_index_json_roundtrip() {
        let index = sample_index();
        let restored = ExpertIndex::from_json(&index.to_json().unwrap()).unwrap();
        assert_eq!(index, restored);
        assert_eq!(restored.num_boxes(), 2);
        assert_eq!(restored.num_experts(), 5);
        assert_eq!(restored.experts_per_box(), vec![3, 2]);
    }

    #[test]
    fn test_wire_format_is_externally_readable() {
        // An engine in another language depends on this encoding, so pin it.
        // serde's default externally-tagged form would emit {"Mlp":{...}},
        // which nobody outside Rust should have to know about.
        let json = sample_index().to_json().unwrap();
        assert!(json.contains(r#""kind": "mlp""#), "{json}");
        assert!(json.contains(r#""site": "mlp""#));
        assert!(json.contains(r#""id": "coding/rust""#));
        assert!(
            !json.contains(r#""Mlp""#),
            "must not leak Rust enum encoding"
        );

        // The discriminant must sit *in* the entry, not one level down. An
        // earlier version emitted `"kind": {"kind": "mlp", ...}`, which this
        // test did not catch because the inner substring still matched.
        assert!(
            !json.contains(r#""kind": {"#),
            "kind must be flattened into the entry, not nested:\n{json}"
        );
        assert!(
            json.contains(r#""hidden_size": 32"#),
            "flattened kind fields must appear at the entry level:\n{json}"
        );

        // ...and the shape fields still round-trip into the right variant.
        let back = ExpertIndex::from_json(&json).unwrap();
        assert!(matches!(
            back.boxes[0].experts[0].kind,
            ExpertKind::Mlp {
                hidden_size: 32,
                intermediate_size: 64
            }
        ));
    }

    #[test]
    fn test_expert_lookup_by_id_and_position() {
        let index = sample_index();
        let (b, e) = index.expert("cyber/malware").unwrap();
        assert_eq!(b.id, "cyber");
        assert_eq!(e.index, 1);
        assert_eq!(e.global_index, 4, "flattened order continues across boxes");
        assert!(index.expert("nope").is_none());
        assert_eq!(index.resolve(0, 2).unwrap().id, "coding/secure");
        assert!(index.resolve(9, 0).is_none());
    }

    #[test]
    fn test_validate_rejects_duplicate_ids() {
        let mut index = sample_index();
        index.boxes[1].experts[0].id = "coding/rust".into();
        let err = index.validate().unwrap_err().to_string();
        assert!(err.contains("duplicate expert id"), "{err}");

        let mut index = sample_index();
        index.boxes[1].id = "coding".into();
        assert!(index
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate box id"));
    }

    #[test]
    fn test_validate_rejects_inconsistent_indices() {
        let mut index = sample_index();
        index.boxes[1].index = 7;
        assert!(index
            .validate()
            .unwrap_err()
            .to_string()
            .contains("claims index 7"));

        let mut index = sample_index();
        index.boxes[1].experts[0].global_index = 99;
        let err = index.validate().unwrap_err().to_string();
        assert!(err.contains("global index 99"), "{err}");
    }

    #[test]
    fn test_validate_rejects_an_all_disabled_box() {
        // Masking every logit in a box to -inf makes the softmax denominator
        // zero and the gates NaN. Refusing the index is the only safe move.
        let mut index = sample_index();
        for expert in &mut index.boxes[1].experts {
            expert.enabled = false;
        }
        let err = index.validate().unwrap_err().to_string();
        assert!(err.contains("disabled") && err.contains("NaN"), "{err}");

        let mut spec = sample_spec();
        spec.boxes[0]
            .experts
            .iter_mut()
            .for_each(|e| e.enabled = false);
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_a_mismatched_site() {
        let mut index = sample_index();
        index.boxes[0].experts[0].kind = ExpertKind::Adapter {
            rank: 4,
            alpha: 4.0,
            in_features: 32,
            out_features: 10,
        };
        let err = index.validate().unwrap_err().to_string();
        assert!(err.contains("adapter expert in a mlp index"), "{err}");
    }

    #[test]
    fn test_validate_rejects_unknown_schema_version() {
        let mut index = sample_index();
        index.schema_version = 99;
        assert!(index
            .validate()
            .unwrap_err()
            .to_string()
            .contains("schema version 99"));
    }

    #[test]
    fn test_spec_extension_lands_disabled() {
        // The hot-swap contract: a newly added expert must arrive switched
        // off, so growing a model is provably an identity until someone
        // enables it deliberately.
        let spec = sample_spec();
        let grown = spec
            .extended_with("coding", ExpertSpec::new("coding/go", "Go"))
            .unwrap();

        assert_eq!(grown.experts_per_box(), vec![4, 2]);
        let (bi, ei) = grown.position("coding/go").unwrap();
        assert_eq!((bi, ei), (0, 3));
        assert!(
            !grown.boxes[0].experts[3].enabled,
            "new experts must land disabled"
        );
        assert!(grown.is_superset_of(&spec));
        assert!(!spec.is_superset_of(&grown));

        assert!(spec
            .extended_with("nonexistent", ExpertSpec::new("a/b", "B"))
            .is_err());
        // A duplicate id is caught by the validate() inside extended_with.
        assert!(spec
            .extended_with("coding", ExpertSpec::new("coding/rust", "Rust again"))
            .is_err());
    }

    #[test]
    fn test_layout_projection() {
        let mut spec = sample_spec();
        spec.boxes[0].experts[1].enabled = false;
        let layout = spec.layout();
        assert_eq!(layout.num_boxes(), 2);
        assert_eq!(layout.experts_per_box(), vec![3, 2]);
        assert_eq!(layout.total_experts(), 5);
        assert!(layout.is_enabled(0, 0));
        assert!(!layout.is_enabled(0, 1));
        assert_eq!(layout.enabled_in(0), 2);
        assert_eq!(layout.enabled_in(1), 2);
        // Out-of-range queries answer rather than panic.
        assert!(!layout.is_enabled(9, 9));
        assert_eq!(layout.experts_in(9), 0);
    }

    #[test]
    fn test_flat_spec_is_a_single_box() {
        let spec = MosmeSpec::flat(4);
        assert_eq!(spec.boxes.len(), 1);
        assert_eq!(spec.num_experts(), 4);
        assert!(spec.validate().is_ok());
        // Degenerate input is clamped rather than producing an invalid spec.
        assert_eq!(MosmeSpec::flat(0).num_experts(), 1);
    }

    #[test]
    fn test_spec_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("mosme-spec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("boxes.json");

        let spec = sample_spec();
        spec.write(&path).unwrap();
        assert_eq!(MosmeSpec::read(&path).unwrap(), spec);

        // Defaults let a minimal hand-written spec parse.
        std::fs::write(
            &path,
            r#"{"boxes":[{"id":"a","label":"A","experts":[{"id":"a/x","label":"X"}]}]}"#,
        )
        .unwrap();
        let minimal = MosmeSpec::read(&path).unwrap();
        assert_eq!(minimal.top_box, 1);
        assert!(minimal.route_on_tokens);
        assert!(minimal.boxes[0].experts[0].enabled);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_render_lists_every_expert() {
        let rendered = sample_index().render();
        for id in [
            "coding/rust",
            "coding/python",
            "coding/secure",
            "cyber/netsec",
        ] {
            assert!(rendered.contains(id), "{id} missing from:\n{rendered}");
        }
        assert!(rendered.contains("boxes=2"));
    }

    #[test]
    fn test_coding_student_spec_validates_and_round_trips() {
        let spec = coding_student_spec(4, 1, 2).unwrap();
        assert_eq!(spec.boxes.len(), CODING_LANGUAGES.len());
        assert_eq!(spec.experts_per_box(), vec![4; CODING_LANGUAGES.len()]);
        assert_eq!(spec.num_experts(), 4 * CODING_LANGUAGES.len());
        assert_eq!(spec.top_box, 1);
        assert_eq!(spec.top_expert, 2);
        assert!(spec.route_on_tokens);
        // Every shipped language has exactly one box, at its vec position.
        for (idx, (key, _)) in CODING_LANGUAGES.iter().enumerate() {
            assert_eq!(spec.boxes[idx].id, format!("coding:{key}"));
            assert!(spec.position(&format!("coding:{key}/micro_0")).is_some());
        }
        // Deterministic: two calls build the same document.
        assert_eq!(spec, coding_student_spec(4, 1, 2).unwrap());
        // serde round-trip: the authoring document parses back identically.
        let text = serde_json::to_string_pretty(&spec).unwrap();
        let back: MosmeSpec = serde_json::from_str(&text).unwrap();
        back.validate().unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn test_coding_student_spec_has_no_duplicate_ids() {
        // The registry is fixed, so duplicates would be a bug in this file;
        // validate() is the check every consumer relies on, so prove it
        // catches a tampered duplicate here.
        let mut spec = coding_student_spec(2, 1, 1).unwrap();
        spec.validate().unwrap();
        spec.boxes[1].experts[0].id = spec.boxes[0].experts[0].id.clone();
        assert!(spec
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate expert id"));
        spec = coding_student_spec(2, 1, 1).unwrap();
        spec.boxes[1].id = spec.boxes[0].id.clone();
        assert!(spec
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate box id"));
    }

    #[test]
    fn test_coding_student_spec_rejects_bad_sizes() {
        assert!(coding_student_spec(0, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("experts_per_box"));
        assert!(coding_student_spec(4, 0, 1)
            .unwrap_err()
            .to_string()
            .contains("top_box"));
        assert!(coding_student_spec(4, CODING_LANGUAGES.len() + 1, 1)
            .unwrap_err()
            .to_string()
            .contains("top_box"));
        assert!(coding_student_spec(2, 1, 3)
            .unwrap_err()
            .to_string()
            .contains("top_expert"));
    }

    #[test]
    fn test_coding_spec_for_builds_a_configured_box_set() {
        // The default builder is the explicit-list builder at the shipped
        // six — same document, byte for byte.
        let six: Vec<String> = CODING_LANGUAGES
            .iter()
            .map(|(key, _)| (*key).to_string())
            .collect();
        assert_eq!(
            coding_student_spec(2, 1, 1).unwrap(),
            coding_spec_for(&six, 2, 1, 1).unwrap()
        );

        // The ten-language set: the extended four land after the shipped
        // six, with the registry's labels and the usual expert naming.
        let ten: Vec<String> = six
            .iter()
            .cloned()
            .chain(
                EXTENDED_CODING_LANGUAGES
                    .iter()
                    .map(|(key, _)| (*key).to_string()),
            )
            .collect();
        let spec = coding_spec_for(&ten, 2, 1, 1).unwrap();
        assert_eq!(spec.boxes.len(), 10);
        assert_eq!(spec.boxes[6].id, "coding:go");
        assert_eq!(spec.boxes[6].label, "Go specialist");
        assert_eq!(spec.boxes[9].id, "coding:php");
        assert!(spec.position("coding:csharp/micro_1").is_some());
        assert_eq!(spec.experts_per_box(), vec![2; 10]);
        spec.validate().unwrap();

        // A reordered list is a reordered router: the list order is the box
        // order, so a config pins its checkpoints' layout.
        let shuffled = coding_spec_for(&["go".to_string(), "rust".to_string()], 1, 1, 1).unwrap();
        assert_eq!(shuffled.boxes[0].id, "coding:go");
        assert_eq!(shuffled.boxes[1].id, "coding:rust");

        // Bad lists are errors naming the argument at fault.
        assert!(coding_spec_for(&[], 1, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("at least one"));
        assert!(coding_spec_for(&["Go".to_string()], 1, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("invalid"));
        assert!(
            coding_spec_for(&["rust".to_string(), "rust".to_string()], 1, 1, 1)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        assert!(coding_spec_for(&six, 1, 7, 1)
            .unwrap_err()
            .to_string()
            .contains("top_box"));

        // The graduated four left the planned list; what remains is
        // genuinely future work.
        for graduated in ["go", "java", "csharp", "php"] {
            assert!(!PLANNED_CODING_LANGUAGES.contains(&graduated));
        }
        assert_eq!(PLANNED_CODING_LANGUAGES.len(), 6);
    }

    #[test]
    fn test_with_field_boxes_appends_after_the_coding_boxes() {
        let base = coding_student_spec(2, 1, 1).unwrap();
        let spec = base
            .with_field_boxes(&["pentest".to_string(), "dfir".to_string()], 2)
            .unwrap();
        // The coding prefix is untouched: same boxes, same order, same experts.
        assert_eq!(spec.boxes[..CODING_LANGUAGES.len()], base.boxes[..]);
        assert_eq!(spec.boxes.len(), CODING_LANGUAGES.len() + 2);
        assert_eq!(spec.boxes[6].id, "field:pentest");
        assert_eq!(spec.boxes[7].id, "field:dfir");
        assert_eq!(spec.experts_per_box(), vec![2; CODING_LANGUAGES.len() + 2]);
        // Field experts are named under their box and enabled from the start.
        let (bi, ei) = spec.position("field:pentest/micro_1").unwrap();
        assert_eq!((bi, ei), (6, 1));
        assert!(spec.boxes[6].experts.iter().all(|e| e.enabled));
        // Deterministic, and the extended spec still validates.
        assert_eq!(
            spec,
            base.with_field_boxes(&["pentest".to_string(), "dfir".to_string()], 2)
                .unwrap()
        );
        spec.validate().unwrap();
    }

    #[test]
    fn test_with_field_boxes_rejects_bad_names_and_duplicates() {
        let base = coding_student_spec(1, 1, 1).unwrap();
        for bad in ["", "Pentest", "pen test", "pentest/recon", "pentest_recon"] {
            let err = base
                .with_field_boxes(&[bad.to_string()], 1)
                .unwrap_err()
                .to_string();
            assert!(err.contains("invalid"), "{bad}: {err}");
        }
        let err = base
            .with_field_boxes(&["pentest".to_string(), "pentest".to_string()], 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate"), "{err}");
        assert!(base
            .with_field_boxes(&["pentest".to_string()], 0)
            .unwrap_err()
            .to_string()
            .contains("experts_per_box"));
        // An empty field list is the identity, not an error.
        assert_eq!(base.with_field_boxes(&[], 1).unwrap(), base);
    }
}
