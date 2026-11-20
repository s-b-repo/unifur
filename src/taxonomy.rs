//! The field / sub-field taxonomy the hierarchical router routes over
//! (TRISKELE-PLAN §3.9, §13; Stage 4).
//!
//! A taxonomy is two levels of stable ids: **categories** (`coding`,
//! `cyber-offensive`, `math`, ...) and the **leaves** inside them (`rust`,
//! `recon`, `number-theory`, ...). A leaf's global id is
//! `<category>/<leaf>` — `cyber-offensive/recon` — which is also a valid nina
//! registry name ([`crate::nina::validate_name`] accepts it), so a specialist
//! trained for a leaf can be registered under the leaf's own id.
//!
//! # Data, not code
//!
//! The plan calls the taxonomy "extensible, dataset-driven": its leaf set is
//! meant to follow the datasets that exist per sub-field. So the taxonomy is a
//! plain serde document with a built-in default ([`Taxonomy::builtin`], the §13
//! structure) that a JSON file can extend ([`Taxonomy::merge`]) or that a
//! dataset manifest can grow leaf by leaf ([`Taxonomy::with_leaf_ids`]).
//! Merging only ever **adds**: an id, once published, keeps its meaning, so a
//! patch that relabels an existing id is refused rather than applied.
//!
//! This module has no Burn dependency, for the same reason
//! [`crate::expert_index`] has none: an engine deciding what to load reads it
//! before it has loaded anything.
//!
//! # Beyond §13
//!
//! §13 covers IT + math + cyber. The built-in adds exactly one category §13
//! does not list, `science`: the shipped nina roster has a science specialist
//! and the router traces built by the data side label ARC-style questions
//! `science`. It is marked as an extension in its label. `bio` is **not**
//! added — no trace uses it and §13 does not name it — so the bio specialist
//! is reported unmapped rather than filed under a guessed category.
//!
//! # Mapping existing specialists
//!
//! [`Taxonomy::map_specialist`] maps a registry name (`field/subfield`) to a
//! leaf, to a whole category (a field-level generalist), or to
//! [`SpecialistMapping::Unmapped`] with the reason. It never guesses: a field
//! claimed by several categories with no sub-field to disambiguate is
//! reported, not resolved by preference. [`nina_roster`] lists the 24 shipped
//! specialists as far as this repository records their names, and
//! [`Taxonomy::map_roster`] reports how each maps.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{bail, ensure, Context};
use serde::{Deserialize, Serialize};

/// Version of the taxonomy document format. Bump on a breaking change.
pub const TAXONOMY_SCHEMA_VERSION: u32 = 1;

/// Version label of the built-in data: the plan section it transcribes plus
/// the date it was written down.
pub const BUILTIN_TAXONOMY_VERSION: &str = "s13-2026.11";

/// One leaf (sub-field) of a category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Leaf {
    /// Local id, unique within its category: lowercase letters, digits and
    /// dashes. The global id is `<category>/<id>`.
    pub id: String,
    pub label: String,
    /// Other spellings a registry sub-field may use for this leaf (`jsts` for
    /// `js-ts`, `bugfinding` for `bug-finding`). Matched exactly, never fuzzily.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
}

/// One category: a domain the meta-router can engage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Category {
    /// Stable id: lowercase letters, digits and dashes (`cyber-offensive`).
    pub id: String,
    pub label: String,
    /// Registry *field* names whose specialists belong to this category
    /// (`pentest` for the cyber categories). Several categories may claim one
    /// field; the sub-field then has to name a leaf to disambiguate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<String>,
    pub leaves: Vec<Leaf>,
}

impl Category {
    /// A leaf by local id or alias.
    pub fn leaf(&self, name: &str) -> Option<&Leaf> {
        self.leaves
            .iter()
            .find(|l| l.id == name || l.aliases.iter().any(|a| a == name))
    }
}

/// The taxonomy document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Taxonomy {
    pub schema_version: u32,
    /// Free-form data version (`s13-2026.11`, `s13-2026.11+hf-subfields`).
    pub version: String,
    pub categories: Vec<Category>,
}

/// The id charset every category, leaf, alias and field name shares — the
/// same rule a nina registry name's parts follow.
fn is_valid_id(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('-')
        && !id.ends_with('-')
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Split a global leaf id `category/leaf` into its parts, checking both.
pub fn parse_leaf_id(id: &str) -> anyhow::Result<(&str, &str)> {
    let (category, leaf) = id
        .split_once('/')
        .with_context(|| format!("leaf id '{id}' is not of the form <category>/<leaf>"))?;
    ensure!(
        is_valid_id(category) && is_valid_id(leaf),
        "leaf id '{id}' has a part outside the id charset (lowercase letters, digits, dashes)"
    );
    Ok((category, leaf))
}

/// Whether `id` is a syntactically valid category id.
pub fn is_valid_category_id(id: &str) -> bool {
    is_valid_id(id)
}

impl Taxonomy {
    /// The built-in taxonomy: the §13 structure plus the `science` extension.
    pub fn builtin() -> Self {
        Self {
            schema_version: TAXONOMY_SCHEMA_VERSION,
            version: BUILTIN_TAXONOMY_VERSION.to_string(),
            categories: BUILTIN
                .iter()
                .map(|row| Category {
                    id: row.id.to_string(),
                    label: row.label.to_string(),
                    fields: row.fields.iter().map(|f| (*f).to_string()).collect(),
                    leaves: row
                        .leaves
                        .iter()
                        .map(|(id, label, aliases)| Leaf {
                            id: (*id).to_string(),
                            label: (*label).to_string(),
                            aliases: aliases.iter().map(|a| (*a).to_string()).collect(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self).context("serialize taxonomy")
    }

    /// Parse and validate; an invalid taxonomy is never returned.
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let taxonomy: Self = serde_json::from_str(text).context("parse taxonomy")?;
        taxonomy.validate()?;
        Ok(taxonomy)
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read taxonomy {}", path.display()))?;
        Self::from_json(&text).with_context(|| format!("in {}", path.display()))
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        self.validate()?;
        crate::interop::write_atomic(path, self.to_json()?.as_bytes())
            .with_context(|| format!("write taxonomy {}", path.display()))
    }

    /// The built-in taxonomy, extended by the patch at `path` when given.
    pub fn builtin_with_patch(path: Option<&Path>) -> anyhow::Result<Self> {
        let base = Self::builtin();
        match path {
            Some(path) => base.merge(&Self::read(path)?),
            None => Ok(base),
        }
    }

    /// Structural invariants: a known schema version, at least one category,
    /// unique and well-formed ids, no empty category, and aliases that do not
    /// collide with each other or with a leaf id inside one category.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.schema_version != TAXONOMY_SCHEMA_VERSION {
            bail!(
                "unsupported taxonomy schema version {} (this build understands {})",
                self.schema_version,
                TAXONOMY_SCHEMA_VERSION
            );
        }
        ensure!(
            !self.categories.is_empty(),
            "a taxonomy needs at least one category"
        );
        let mut category_ids = BTreeSet::new();
        for category in &self.categories {
            ensure!(
                is_valid_id(&category.id),
                "category id '{}' is invalid: lowercase letters, digits and dashes only",
                category.id
            );
            ensure!(
                category_ids.insert(category.id.as_str()),
                "duplicate category id '{}'",
                category.id
            );
            ensure!(
                !category.leaves.is_empty(),
                "category '{}' has no leaves",
                category.id
            );
            for field in &category.fields {
                ensure!(
                    is_valid_id(field),
                    "category '{}' claims field '{field}', which is not a valid registry field name",
                    category.id
                );
            }
            // Every name a sub-field can match — leaf ids and aliases alike —
            // is unique within the category, or a lookup would be ambiguous.
            let mut names = BTreeSet::new();
            for leaf in &category.leaves {
                ensure!(
                    is_valid_id(&leaf.id),
                    "leaf id '{}/{}' is invalid: lowercase letters, digits and dashes only",
                    category.id,
                    leaf.id
                );
                ensure!(
                    names.insert(leaf.id.as_str()),
                    "duplicate leaf name '{}/{}'",
                    category.id,
                    leaf.id
                );
            }
            for leaf in &category.leaves {
                for alias in &leaf.aliases {
                    ensure!(
                        is_valid_id(alias),
                        "alias '{alias}' of '{}/{}' is invalid",
                        category.id,
                        leaf.id
                    );
                    ensure!(
                        names.insert(alias.as_str()),
                        "alias '{alias}' of '{}/{}' collides with another leaf name or alias",
                        category.id,
                        leaf.id
                    );
                }
            }
        }
        Ok(())
    }

    pub fn num_categories(&self) -> usize {
        self.categories.len()
    }

    pub fn num_leaves(&self) -> usize {
        self.categories.iter().map(|c| c.leaves.len()).sum()
    }

    pub fn category(&self, id: &str) -> Option<&Category> {
        self.categories.iter().find(|c| c.id == id)
    }

    /// Category ids in document order.
    pub fn category_ids(&self) -> Vec<String> {
        self.categories.iter().map(|c| c.id.clone()).collect()
    }

    /// Every global leaf id, in document order.
    pub fn leaf_ids(&self) -> Vec<String> {
        self.categories
            .iter()
            .flat_map(|c| c.leaves.iter().map(move |l| format!("{}/{}", c.id, l.id)))
            .collect()
    }

    /// A leaf by its global id (`category/leaf`). Aliases are not ids and do
    /// not resolve here; use [`Self::map_specialist`] for registry names.
    pub fn leaf(&self, id: &str) -> Option<(&Category, &Leaf)> {
        let (category, leaf) = id.split_once('/')?;
        let category = self.category(category)?;
        let leaf = category.leaves.iter().find(|l| l.id == leaf)?;
        Some((category, leaf))
    }

    /// Check a category id (and optionally a global leaf id inside it)
    /// against this taxonomy; the error names what is missing.
    pub fn check_ids(&self, category: &str, leaf: Option<&str>) -> anyhow::Result<()> {
        ensure!(
            self.category(category).is_some(),
            "category '{category}' is not in taxonomy {}",
            self.version
        );
        if let Some(leaf) = leaf {
            let (owner, _) = parse_leaf_id(leaf)?;
            ensure!(
                owner == category,
                "leaf '{leaf}' does not belong to category '{category}'"
            );
            ensure!(
                self.leaf(leaf).is_some(),
                "leaf '{leaf}' is not in taxonomy {}",
                self.version
            );
        }
        Ok(())
    }

    /// Extend `self` with `patch`. Additive only: new categories are
    /// appended, new leaves are appended to existing categories, and aliases
    /// and fields are unioned (base order first). An id that exists on both
    /// sides must carry the same label — ids are stable, so a patch cannot
    /// quietly change what one means. The result is validated.
    pub fn merge(&self, patch: &Taxonomy) -> anyhow::Result<Taxonomy> {
        patch
            .validate()
            .context("the taxonomy patch does not validate on its own")?;
        let mut merged = self.clone();
        for incoming in &patch.categories {
            match merged.categories.iter_mut().find(|c| c.id == incoming.id) {
                None => merged.categories.push(incoming.clone()),
                Some(existing) => {
                    ensure!(
                        existing.label == incoming.label,
                        "patch relabels category '{}' ('{}' -> '{}'); ids are stable, add a new id instead",
                        existing.id,
                        existing.label,
                        incoming.label
                    );
                    union_into(&mut existing.fields, &incoming.fields);
                    for leaf in &incoming.leaves {
                        match existing.leaves.iter_mut().find(|l| l.id == leaf.id) {
                            None => existing.leaves.push(leaf.clone()),
                            Some(current) => {
                                ensure!(
                                    current.label == leaf.label,
                                    "patch relabels leaf '{}/{}' ('{}' -> '{}'); ids are stable",
                                    existing.id,
                                    leaf.id,
                                    current.label,
                                    leaf.label
                                );
                                union_into(&mut current.aliases, &leaf.aliases);
                            }
                        }
                    }
                }
            }
        }
        if patch.version != self.version {
            merged.version = format!("{}+{}", self.version, patch.version);
        }
        merged.validate()?;
        Ok(merged)
    }

    /// Grow the leaf set from a dataset manifest's sub-field ids
    /// (`category/leaf`). A new leaf is labelled with its own id; an id whose
    /// category does not exist is an error — a dataset does not get to
    /// invent a category by naming one.
    pub fn with_leaf_ids(&self, ids: &[String]) -> anyhow::Result<Taxonomy> {
        let mut next = self.clone();
        for id in ids {
            let (category, leaf) = parse_leaf_id(id)?;
            let entry = next
                .categories
                .iter_mut()
                .find(|c| c.id == category)
                .with_context(|| {
                    format!("dataset leaf '{id}' names category '{category}', which does not exist")
                })?;
            if entry.leaf(leaf).is_none() {
                entry.leaves.push(Leaf {
                    id: leaf.to_string(),
                    label: leaf.to_string(),
                    aliases: Vec::new(),
                });
            }
        }
        next.validate()?;
        Ok(next)
    }

    /// Map a registry specialist name (`field/subfield`, a bare `field`, or a
    /// global leaf id) onto this taxonomy. See [`SpecialistMapping`].
    pub fn map_specialist(&self, name: &str) -> SpecialistMapping {
        // A global leaf id maps to itself.
        if let Some((category, leaf)) = self.leaf(name) {
            return SpecialistMapping::Leaf {
                category: category.id.clone(),
                leaf: format!("{}/{}", category.id, leaf.id),
            };
        }
        let (field, subfield) = match name.split_once('/') {
            Some((field, subfield)) => (field, Some(subfield)),
            None => (name, None),
        };
        let candidates: Vec<&Category> = self
            .categories
            .iter()
            .filter(|c| c.id == field || c.fields.iter().any(|f| f == field))
            .collect();
        if candidates.is_empty() {
            return SpecialistMapping::Unmapped {
                reason: format!(
                    "no category of taxonomy {} claims field '{field}'",
                    self.version
                ),
            };
        }
        let candidate_ids = || {
            candidates
                .iter()
                .map(|c| c.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match subfield {
            None | Some("general") => match candidates.as_slice() {
                [only] => SpecialistMapping::Category {
                    category: only.id.clone(),
                },
                _ => SpecialistMapping::Unmapped {
                    reason: format!(
                        "field '{field}' spans {} categories ({}) and the name has no sub-field to choose between them",
                        candidates.len(),
                        candidate_ids()
                    ),
                },
            },
            Some(sub) => {
                let hits: Vec<(&Category, &Leaf)> = candidates
                    .iter()
                    .filter_map(|c| c.leaf(sub).map(|l| (*c, l)))
                    .collect();
                match hits.as_slice() {
                    [(category, leaf)] => SpecialistMapping::Leaf {
                        category: category.id.clone(),
                        leaf: format!("{}/{}", category.id, leaf.id),
                    },
                    [] => SpecialistMapping::Unmapped {
                        reason: format!(
                            "no leaf named '{sub}' in the categories claiming field '{field}' ({})",
                            candidate_ids()
                        ),
                    },
                    many => SpecialistMapping::Unmapped {
                        reason: format!(
                            "sub-field '{sub}' names a leaf in {} categories ({})",
                            many.len(),
                            many.iter()
                                .map(|(c, _)| c.id.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    },
                }
            }
        }
    }

    /// Map every [`nina_roster`] entry.
    pub fn map_roster(&self) -> RosterReport {
        let entries: Vec<RosterMapping> = nina_roster()
            .into_iter()
            .map(|entry| RosterMapping {
                mapping: self.map_specialist(&entry.name),
                entry,
            })
            .collect();
        let count = |pred: fn(&SpecialistMapping) -> bool| -> usize {
            entries
                .iter()
                .filter(|e| pred(&e.mapping))
                .map(|e| e.entry.count)
                .sum()
        };
        RosterReport {
            taxonomy_version: self.version.clone(),
            specialists: entries.iter().map(|e| e.entry.count).sum(),
            to_leaf: count(|m| matches!(m, SpecialistMapping::Leaf { .. })),
            to_category: count(|m| matches!(m, SpecialistMapping::Category { .. })),
            unmapped: count(|m| matches!(m, SpecialistMapping::Unmapped { .. })),
            entries,
        }
    }

    /// Human-readable table for `dblocks experts taxonomy`.
    pub fn render(&self) -> String {
        let mut out = format!(
            "taxonomy {} (schema {}): {} categories, {} leaves\n",
            self.version,
            self.schema_version,
            self.num_categories(),
            self.num_leaves()
        );
        for category in &self.categories {
            out.push_str(&format!(
                "\n[{}] {} — {} leaves{}\n",
                category.id,
                category.label,
                category.leaves.len(),
                if category.fields.is_empty() {
                    String::new()
                } else {
                    format!("  (registry fields: {})", category.fields.join(", "))
                }
            ));
            for leaf in &category.leaves {
                out.push_str(&format!(
                    "  {:<40} {}{}\n",
                    format!("{}/{}", category.id, leaf.id),
                    leaf.label,
                    if leaf.aliases.is_empty() {
                        String::new()
                    } else {
                        format!("  (aliases: {})", leaf.aliases.join(", "))
                    }
                ));
            }
        }
        out
    }
}

fn union_into(into: &mut Vec<String>, from: &[String]) {
    for item in from {
        if !into.contains(item) {
            into.push(item.clone());
        }
    }
}

/// How a registry specialist lands on the taxonomy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpecialistMapping {
    /// A sub-field specialist: one leaf.
    Leaf { category: String, leaf: String },
    /// A field-level generalist: the whole category, no particular leaf.
    Category { category: String },
    /// Not placed, with the reason. Never resolved by guessing.
    Unmapped { reason: String },
}

impl SpecialistMapping {
    /// The category this mapping places the specialist in, if any.
    pub fn category(&self) -> Option<&str> {
        match self {
            Self::Leaf { category, .. } | Self::Category { category } => Some(category),
            Self::Unmapped { .. } => None,
        }
    }

    /// The global leaf id, for a sub-field specialist.
    pub fn leaf(&self) -> Option<&str> {
        match self {
            Self::Leaf { leaf, .. } => Some(leaf),
            _ => None,
        }
    }
}

/// One line of the shipped nina roster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterEntry {
    /// The registry name as far as this repository records it: `field/subfield`
    /// where the sub-field is known, the bare field where it is not.
    pub name: String,
    /// How many shipped specialists this line stands for (3 for `math` and
    /// `tech`, whose sub-field names this repository does not record).
    pub count: usize,
    /// Where the name comes from.
    pub source: String,
}

/// The 24 shipped nina specialists (coding x10, math x3, science, bio, cyber,
/// tech x3, pentest x5), named from what this repository records:
///
/// - the ten coding languages are the student's box registry
///   ([`crate::expert_index::CODING_LANGUAGES`] and
///   [`crate::expert_index::EXTENDED_CODING_LANGUAGES`]);
/// - the field and pentest names are [`crate::fieldnorm::available_fields`].
///
/// The repository does not record the sub-field names of the three math and
/// three tech specialists (the counts come from the published model card), so
/// those two lines carry the bare field and a count of 3 rather than invented
/// names.
pub fn nina_roster() -> Vec<RosterEntry> {
    let mut roster: Vec<RosterEntry> = crate::expert_index::CODING_LANGUAGES
        .iter()
        .chain(crate::expert_index::EXTENDED_CODING_LANGUAGES.iter())
        .map(|(key, _)| RosterEntry {
            name: format!("coding/{key}"),
            count: 1,
            source: "expert_index coding box registry".to_string(),
        })
        .collect();
    for field in crate::fieldnorm::available_fields() {
        let count = match *field {
            "math" | "tech" => 3,
            _ => 1,
        };
        roster.push(RosterEntry {
            name: (*field).to_string(),
            count,
            source: if count > 1 {
                "fieldnorm field list; sub-field names not recorded in this repository".to_string()
            } else {
                "fieldnorm field list".to_string()
            },
        });
    }
    roster
}

/// One roster line and how it mapped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterMapping {
    pub entry: RosterEntry,
    pub mapping: SpecialistMapping,
}

/// The roster mapping, with specialist-weighted counts (a `count: 3` line
/// counts three times).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RosterReport {
    pub taxonomy_version: String,
    pub specialists: usize,
    pub to_leaf: usize,
    pub to_category: usize,
    pub unmapped: usize,
    pub entries: Vec<RosterMapping>,
}

impl RosterReport {
    pub fn render(&self) -> String {
        let mut out = format!(
            "nina roster -> taxonomy {}: {} specialists, {} to a leaf, {} to a whole category, {} unmapped\n",
            self.taxonomy_version, self.specialists, self.to_leaf, self.to_category, self.unmapped
        );
        for e in &self.entries {
            let target = match &e.mapping {
                SpecialistMapping::Leaf { leaf, .. } => format!("leaf {leaf}"),
                SpecialistMapping::Category { category } => format!("category {category}"),
                SpecialistMapping::Unmapped { reason } => format!("UNMAPPED: {reason}"),
            };
            out.push_str(&format!(
                "  {:<22} x{:<2} {}\n",
                e.entry.name, e.entry.count, target
            ));
        }
        out
    }
}

// ------------------------------------------------------------ built-in data --

type LeafRow = (&'static str, &'static str, &'static [&'static str]);

struct CategoryRow {
    id: &'static str,
    label: &'static str,
    fields: &'static [&'static str],
    leaves: &'static [LeafRow],
}

/// Registry fields shared by the cyber categories: the shipped `cyber`
/// generalist and the `pentest/*` specialists.
const CYBER_FIELDS: &[&str] = &["cyber", "pentest"];

/// TRISKELE-PLAN §13, transcribed. Ids are the §13 names lower-cased with
/// `/`, `(`, `)` and spaces turned into dashes; the original spelling is the
/// label.
const BUILTIN: &[CategoryRow] = &[
    CategoryRow {
        id: "coding",
        label: "Coding",
        fields: &["coding"],
        leaves: &[
            ("rust", "Rust", &[]),
            ("c", "C", &[]),
            ("cpp", "C++", &[]),
            ("python", "Python", &[]),
            ("go", "Go", &[]),
            ("java", "Java", &[]),
            ("csharp", "C#", &[]),
            ("php", "PHP", &[]),
            (
                "js-ts",
                "JavaScript / TypeScript",
                &["jsts", "javascript", "typescript"],
            ),
            ("web", "Web (HTML/CSS/JS mixed)", &[]),
            ("ruby", "Ruby", &[]),
            ("swift", "Swift", &[]),
            ("kotlin", "Kotlin", &[]),
            ("sql", "SQL", &[]),
            ("shell", "Shell", &["bash"]),
            ("asm", "Assembly", &["assembly"]),
        ],
    },
    CategoryRow {
        id: "systems",
        label: "Systems / OS",
        fields: &[],
        leaves: &[
            ("linux", "Linux", &[]),
            ("windows-internals", "Windows internals", &[]),
            ("kernel", "Kernel", &[]),
            ("drivers", "Drivers", &[]),
            ("embedded", "Embedded", &[]),
            ("firmware", "Firmware", &[]),
            ("rtos", "RTOS", &[]),
            ("concurrency", "Concurrency", &[]),
        ],
    },
    CategoryRow {
        id: "networking",
        label: "Networking",
        fields: &[],
        leaves: &[
            ("tcp-ip", "TCP/IP", &["tcpip"]),
            ("dns", "DNS", &[]),
            ("http", "HTTP", &[]),
            ("tls", "TLS", &[]),
            ("routing-bgp", "Routing / BGP", &["bgp"]),
            ("wireless", "Wireless", &[]),
            ("sdn", "SDN", &[]),
            ("protocols", "Protocols (general)", &[]),
        ],
    },
    CategoryRow {
        id: "cloud",
        label: "Cloud / Infra / DevOps",
        fields: &["tech"],
        leaves: &[
            ("aws", "AWS", &[]),
            ("gcp", "GCP", &[]),
            ("azure", "Azure", &[]),
            ("kubernetes", "Kubernetes", &["k8s"]),
            ("docker", "Docker", &[]),
            ("terraform-iac", "Terraform / IaC", &["terraform", "iac"]),
            ("ci-cd", "CI/CD", &["cicd"]),
            ("sre", "SRE", &[]),
            ("observability", "Observability", &[]),
        ],
    },
    CategoryRow {
        id: "data",
        label: "Data",
        fields: &["tech"],
        leaves: &[
            (
                "databases",
                "Databases (SQL / NoSQL)",
                &["sql", "nosql", "postgres", "text-to-sql"],
            ),
            ("data-engineering", "Data engineering", &[]),
            ("distributed-systems", "Distributed systems", &[]),
            ("streaming", "Streaming", &[]),
            ("analytics", "Analytics", &[]),
        ],
    },
    CategoryRow {
        id: "web-mobile",
        label: "Web / Mobile",
        fields: &[],
        leaves: &[
            ("frontend", "Frontend", &[]),
            ("backend", "Backend", &[]),
            ("api-design", "API design", &[]),
            ("browsers", "Browsers", &[]),
            ("android", "Android", &[]),
            ("ios", "iOS", &[]),
        ],
    },
    CategoryRow {
        id: "ai-ml",
        label: "AI / ML",
        fields: &[],
        leaves: &[
            ("ml", "Machine learning", &[]),
            ("deep-learning", "Deep learning", &[]),
            ("nlp", "NLP", &[]),
            ("cv", "Computer vision", &[]),
            ("rl", "Reinforcement learning", &[]),
            ("mlops", "MLOps", &[]),
        ],
    },
    CategoryRow {
        id: "cs-theory",
        label: "CS theory",
        fields: &[],
        leaves: &[
            ("algorithms", "Algorithms", &[]),
            ("data-structures", "Data structures", &[]),
            ("complexity", "Complexity", &[]),
            ("compilers", "Compilers", &[]),
            ("formal-methods", "Formal methods", &[]),
            ("cryptography-theory", "Cryptography (theory)", &[]),
        ],
    },
    CategoryRow {
        id: "math",
        label: "Math",
        fields: &["math"],
        leaves: &[
            ("algebra", "Algebra", &[]),
            ("linear-algebra", "Linear algebra", &[]),
            ("number-theory", "Number theory", &[]),
            ("analysis", "Analysis", &[]),
            ("calculus", "Calculus", &[]),
            ("geometry", "Geometry", &[]),
            ("topology", "Topology", &[]),
            ("logic", "Logic", &[]),
            ("probability", "Probability", &[]),
            ("statistics", "Statistics", &[]),
            ("optimization", "Optimization", &[]),
            ("combinatorics", "Combinatorics", &[]),
            ("graph-theory", "Graph theory", &[]),
            ("ode-pde", "ODE / PDE", &["ode", "pde"]),
            ("numerical-methods", "Numerical methods", &[]),
            ("competition", "Competition math", &[]),
            ("word-problems", "Word problems", &[]),
        ],
    },
    CategoryRow {
        id: "cyber-offensive",
        label: "Cyber — offensive",
        fields: CYBER_FIELDS,
        leaves: &[
            ("recon", "Recon", &[]),
            ("scanning", "Scanning", &[]),
            ("enumeration", "Enumeration", &[]),
            ("exploit-dev", "Exploit development", &[]),
            ("zeroday", "Zero-day research", &[]),
            ("web-exploitation", "Web exploitation", &[]),
            ("binary-exploitation", "Binary exploitation", &[]),
            ("privesc", "Privilege escalation", &[]),
            ("lateral-movement", "Lateral movement", &[]),
            ("c2", "C2", &[]),
            ("evasion", "Evasion", &[]),
            ("red-team", "Red team", &[]),
        ],
    },
    CategoryRow {
        id: "cyber-defensive",
        label: "Cyber — defensive",
        fields: CYBER_FIELDS,
        leaves: &[
            ("blue-team", "Blue team", &[]),
            ("detection", "Detection", &[]),
            ("siem", "SIEM", &[]),
            ("incident-response", "Incident response", &[]),
            ("forensics", "Forensics", &["dfir"]),
            ("malware-analysis", "Malware analysis", &[]),
            ("reverse-engineering", "Reverse engineering", &[]),
            ("threat-intel", "Threat intelligence", &[]),
            ("hardening", "Hardening", &[]),
        ],
    },
    CategoryRow {
        id: "cyber-appsec",
        label: "Cyber — appsec",
        fields: CYBER_FIELDS,
        leaves: &[
            ("sast", "SAST", &[]),
            ("dast", "DAST", &[]),
            ("fuzzing", "Fuzzing", &[]),
            ("bug-finding", "Bug finding", &["bugfinding"]),
            ("code-audit", "Code audit", &[]),
        ],
    },
    CategoryRow {
        id: "cyber-netsec-crypto",
        label: "Cyber — netsec / crypto",
        fields: CYBER_FIELDS,
        leaves: &[
            ("firewalls", "Firewalls", &[]),
            ("ids-ips", "IDS / IPS", &["ids", "ips"]),
            ("packet-analysis", "Packet analysis", &[]),
            ("cryptography", "Cryptography", &[]),
            ("cryptanalysis", "Cryptanalysis", &[]),
            ("pki", "PKI", &[]),
        ],
    },
    CategoryRow {
        id: "cyber-specialized",
        label: "Cyber — specialized",
        fields: CYBER_FIELDS,
        leaves: &[
            ("cloud-sec", "Cloud security", &[]),
            ("mobile-sec", "Mobile security", &[]),
            ("iot-ics-scada", "IoT / ICS-SCADA", &["iot", "ics", "scada"]),
            ("wireless-sec", "Wireless security", &[]),
            ("osint", "OSINT", &[]),
            ("social-engineering", "Social engineering", &[]),
            ("methodology", "Methodology", &[]),
            ("research", "Research", &[]),
        ],
    },
    CategoryRow {
        id: "cyber-data",
        label: "Cyber data / algorithms",
        fields: CYBER_FIELDS,
        leaves: &[
            ("log-analysis", "Log analysis", &[]),
            ("anomaly-detection", "Anomaly detection", &[]),
            ("security-graph-analytics", "Security graph analytics", &[]),
            ("ml-for-security", "ML for security", &[]),
        ],
    },
    CategoryRow {
        id: "science",
        label: "Science (extension beyond §13)",
        fields: &["science"],
        leaves: &[
            ("physics", "Physics", &[]),
            ("chemistry", "Chemistry", &[]),
            ("biology", "Biology", &[]),
            ("earth-space", "Earth and space science", &[]),
            ("general-science", "General science (ARC-style)", &["arc"]),
        ],
    },
];

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
    fn builtin_validates_and_has_the_s13_shape() {
        let t = Taxonomy::builtin();
        t.validate().unwrap();
        assert_eq!(t.num_categories(), 16, "15 §13 categories + science");
        assert_eq!(t.num_leaves(), 130, "125 §13 leaves + 5 science");
        // Every global leaf id is a valid nina registry name.
        for id in t.leaf_ids() {
            crate::nina::validate_name(&id).unwrap();
        }
        // The ids the data side's router traces use all exist.
        for id in [
            "math",
            "coding",
            "data",
            "cyber-offensive",
            "science",
            "cyber-netsec-crypto",
            "cyber-defensive",
            "cloud",
            "networking",
            "ai-ml",
        ] {
            assert!(t.category(id).is_some(), "{id}");
        }
    }

    #[test]
    fn json_round_trip_is_identical() {
        let t = Taxonomy::builtin();
        let json = t.to_json().unwrap();
        let back = Taxonomy::from_json(&json).unwrap();
        assert_eq!(back, t);
        assert_eq!(back.to_json().unwrap(), json);
    }

    #[test]
    fn validation_rejects_duplicates_and_empty_categories() {
        let mut t = Taxonomy::builtin();
        t.categories[1].id = "coding".into();
        assert!(t
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate category"));

        let mut t = Taxonomy::builtin();
        t.categories[0].leaves.clear();
        assert!(t.validate().unwrap_err().to_string().contains("no leaves"));

        let mut t = Taxonomy::builtin();
        t.categories[0].leaves[1].id = "rust".into();
        assert!(t
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate leaf"));

        let mut t = Taxonomy::builtin();
        t.categories[0].leaves[0].aliases.push("cpp".into());
        assert!(t.validate().unwrap_err().to_string().contains("collides"));

        let mut t = Taxonomy::builtin();
        t.categories[0].id = "Coding".into();
        assert!(t.validate().is_err());

        let mut t = Taxonomy::builtin();
        t.categories.clear();
        assert!(t.validate().is_err());

        let mut t = Taxonomy::builtin();
        t.schema_version = 99;
        assert!(t.validate().is_err());
    }

    #[test]
    fn merge_is_additive_and_refuses_relabels() {
        let base = Taxonomy::builtin();
        let patch = Taxonomy {
            schema_version: TAXONOMY_SCHEMA_VERSION,
            version: "hf-subfields".into(),
            categories: vec![
                Category {
                    id: "coding".into(),
                    label: "Coding".into(),
                    fields: vec!["code".into()],
                    leaves: vec![
                        Leaf {
                            id: "zig".into(),
                            label: "Zig".into(),
                            aliases: vec![],
                        },
                        Leaf {
                            id: "rust".into(),
                            label: "Rust".into(),
                            aliases: vec!["rs".into()],
                        },
                    ],
                },
                Category {
                    id: "bio".into(),
                    label: "Biology / medicine".into(),
                    fields: vec!["bio".into()],
                    leaves: vec![Leaf {
                        id: "medical-qa".into(),
                        label: "Medical QA".into(),
                        aliases: vec![],
                    }],
                },
            ],
        };
        let merged = base.merge(&patch).unwrap();
        assert_eq!(merged.num_categories(), base.num_categories() + 1);
        assert_eq!(merged.num_leaves(), base.num_leaves() + 2);
        assert!(merged.leaf("coding/zig").is_some());
        assert_eq!(
            merged.version,
            format!("{BUILTIN_TAXONOMY_VERSION}+hf-subfields")
        );
        assert_eq!(
            merged.map_specialist("coding/rs"),
            SpecialistMapping::Leaf {
                category: "coding".into(),
                leaf: "coding/rust".into()
            }
        );
        // Existing ids keep their order and meaning.
        assert_eq!(merged.categories[0].leaves[0].id, "rust");
        assert_eq!(merged.map_specialist("bio").category(), Some("bio"));

        let mut relabel = patch.clone();
        relabel.categories[0].label = "Programming".into();
        assert!(base
            .merge(&relabel)
            .unwrap_err()
            .to_string()
            .contains("relabels"));
        let mut relabel = patch;
        relabel.categories[0].leaves[1].label = "Rust lang".into();
        assert!(base
            .merge(&relabel)
            .unwrap_err()
            .to_string()
            .contains("relabels"));
    }

    #[test]
    fn dataset_leaf_ids_grow_but_never_invent_categories() {
        let base = Taxonomy::builtin();
        let grown = base
            .with_leaf_ids(&["math/category-theory".into(), "coding/rust".into()])
            .unwrap();
        assert_eq!(grown.num_leaves(), base.num_leaves() + 1);
        assert!(grown.leaf("math/category-theory").is_some());
        assert!(base.with_leaf_ids(&["chemistry/organic".into()]).is_err());
        assert!(base.with_leaf_ids(&["no-slash".into()]).is_err());
    }

    #[test]
    fn specialist_mapping_never_guesses() {
        let t = Taxonomy::builtin();
        let leaf = |c: &str, l: &str| SpecialistMapping::Leaf {
            category: c.into(),
            leaf: l.into(),
        };
        assert_eq!(
            t.map_specialist("coding/jsts"),
            leaf("coding", "coding/js-ts")
        );
        assert_eq!(
            t.map_specialist("pentest/recon"),
            leaf("cyber-offensive", "cyber-offensive/recon")
        );
        assert_eq!(
            t.map_specialist("pentest/bugfinding"),
            leaf("cyber-appsec", "cyber-appsec/bug-finding")
        );
        assert_eq!(
            t.map_specialist("cyber-offensive/recon"),
            leaf("cyber-offensive", "cyber-offensive/recon")
        );
        assert_eq!(t.map_specialist("tech/sql"), leaf("data", "data/databases"));
        assert_eq!(
            t.map_specialist("tech/kubernetes"),
            leaf("cloud", "cloud/kubernetes")
        );
        assert_eq!(
            t.map_specialist("math"),
            SpecialistMapping::Category {
                category: "math".into()
            }
        );
        assert_eq!(
            t.map_specialist("science/general"),
            SpecialistMapping::Category {
                category: "science".into()
            }
        );
        // Ambiguous or unknown: reported, not resolved.
        for name in [
            "cyber",
            "tech",
            "bio",
            "bio/general",
            "pentest/nonexistent",
            "coding/cobol",
        ] {
            assert!(
                matches!(t.map_specialist(name), SpecialistMapping::Unmapped { .. }),
                "{name}"
            );
        }
        let SpecialistMapping::Unmapped { reason } = t.map_specialist("cyber") else {
            panic!("cyber must be unmapped");
        };
        assert!(reason.contains("6 categories"), "{reason}");
    }

    #[test]
    fn roster_is_the_24_shipped_specialists() {
        let report = Taxonomy::builtin().map_roster();
        assert_eq!(report.specialists, 24);
        assert_eq!(report.to_leaf, 15, "10 coding + 5 pentest");
        assert_eq!(report.to_category, 4, "math x3 + science");
        assert_eq!(report.unmapped, 5, "bio, cyber, tech x3");
        assert_eq!(
            report.to_leaf + report.to_category + report.unmapped,
            report.specialists
        );
        let rendered = report.render();
        assert!(rendered.contains("UNMAPPED"), "{rendered}");
    }

    #[test]
    fn check_ids_names_the_missing_id() {
        let t = Taxonomy::builtin();
        t.check_ids("coding", Some("coding/rust")).unwrap();
        t.check_ids("coding", None).unwrap();
        assert!(t
            .check_ids("nope", None)
            .unwrap_err()
            .to_string()
            .contains("nope"));
        assert!(t.check_ids("coding", Some("math/algebra")).is_err());
        assert!(t.check_ids("coding", Some("coding/cobol")).is_err());
        assert_eq!(parse_leaf_id("math/logic").unwrap(), ("math", "logic"));
        assert!(parse_leaf_id("math").is_err());
    }

    #[test]
    fn render_lists_every_leaf() {
        let t = Taxonomy::builtin();
        let text = t.render();
        for id in t.leaf_ids() {
            assert!(text.contains(&id), "{id}");
        }
    }
}
