//! Field-corpus normalizer: raw public datasets (parquet, JSON, JSONL chat,
//! Markdown, SGML) -> the cybercorpus layout (`dataset/all.jsonl` with one
//! `{"id","source","path","format","text"}` per line, per-source files, and a
//! `manifest.json`), routed through the cheat stack before anything lands.
//!
//! This is the field-wide counterpart of [`crate::corpus::build_cybercorpus`]:
//! where that normalizer walks an already-plain `raw/` tree, this one knows
//! each downloaded dataset's own format — NuminaMath/MetaMathQA for math,
//! ai2_arc for science, MedMCQA for bio, Fenrir/Trendyol for cyber,
//! kubernetes-website/docker-docs/synthetic_text_to_sql/postgres-docs for
//! tech, and the pentest subfield feeds (PLAN-NINA §1): NVD CVE JSON 2.0 rows
//! (research: all CVEs; zeroday: only exploit-referenced ones), MITRE CWE
//! content XML (bugfinding), MITRE CAPEC XML (zeroday), ATT&CK STIX rows
//! (recon: TA0043/TA0042 techniques; methodology: everything), and the OWASP
//! Cheat Sheet Series markdown tree (methodology) — and renders every row
//! into one document of text.
//!
//! The crate has no parquet reader (and must not gain one), so parquet
//! sources arrive pre-converted: `/srv/m-sda/unifur/tools/parquet2jsonl.py`
//! writes `raw/<source>.jsonl` next to a `raw/parquet2jsonl-conversions.jsonl`
//! provenance log, and this module refuses to build a parquet-backed source
//! whose converted JSONL is absent, naming the tool to run. The conversions
//! log is copied into the manifest verbatim.
//!
//! Every rendered document passes [`crate::cheat::scan_snippet`] (as
//! [`crate::cheat::SnippetLanguage::Other`] — these are prose/QA documents and
//! the scanner's language-independent classes, secrets included, still run)
//! and then the MinHash near-duplicate grouping
//! ([`crate::cheat::minhash_near_duplicate_groups`]) within its source.
//!
//! Drop policy, measured: the scanner's quarantine grade
//! ([`Severity::Evasion`]: stub, broken, language-mixing, weak behavioral
//! reads) is calibrated for *code snippets* — "prose pasted into a code
//! block" presumes the text should be code. Applied to natural-language
//! documentation it fires on ordinary English (a line starting "You can" or
//! "Note that"): measured on synthetic_text_to_sql it dropped **100%** of the
//! rows, on docker-docs 84%, on postgres-docs 75%. That is not a filter, it
//! is an erasure. So the drop threshold for these prose corpora is
//! **reject-grade** ([`Severity::Suppression`] and above: suppression, eval
//! gaming, secrets, encoding and dynamic evasion, reject-grade behavioral),
//! and quarantine-grade findings are *counted per class* in the manifest
//! (`quarantine_findings`) without dropping. Every drop and every finding is
//! counted; nothing is silently skipped. Documents
//! get deterministic ids (`<source>-<seq:06>`) in deterministic order (sorted
//! file walk, row order within files), and a completed source leaves a marker
//! under `dataset/.fieldnorm/` so a rerun skips it (`--force` rebuilds).
//!
//! [`to_pairs`] is the bridge from a finished field corpus to the trainer:
//! `LanguageBatch::load` reads `{"prompt","completion"}` lines, so `dblocks
//! corpus to-pairs` renders `dataset/all.jsonl` into exactly that shape (plus
//! the `field`), splitting each document at its first blank line or its first
//! quarter (see [`split_prompt_completion`] for the rule), capped at a
//! deterministic sha256-of-id subsample because the trainer holds the whole
//! pairs file in memory. The conversion streams: the corpus is never read
//! whole.

use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::cheat::{self, Severity, SnippetLanguage};
use crate::corpus::DatasetRecord;

/// Default ceiling on one rendered document, matching
/// [`crate::corpus::DEFAULT_MAX_DOC_BYTES`]: a multi-megabyte "document" is a
/// packaging mistake, and reading it as one would silently dominate the
/// corpus. Larger documents are dropped and counted (`oversize`).
pub const DEFAULT_MAX_DOC_BYTES: usize = 4 * 1024 * 1024;

/// Default MinHash near-duplicate threshold, the value
/// [`crate::cheat::near_duplicate_groups`] documents as the point where a
/// one-word edit still scores below and a touched-up copy scores above.
pub const DEFAULT_NEAR_DUP_THRESHOLD: f64 = 0.8;

/// Default MinHash signature length; [`crate::cheat::minhash_near_duplicate_groups`]
/// documents the estimate error as ~1/sqrt(hashes), so 128 gives ~0.09.
pub const DEFAULT_NEAR_DUP_HASHES: usize = 128;

/// Where the fetch agents put the downloaded public datasets.
pub const DEFAULT_DATASETS_ROOT: &str = "/srv/m-sdd/unifur/datasets";

/// Where the normalized per-field corpora are written.
pub const DEFAULT_OUT_ROOT: &str = "/srv/m-sdd/unifur/corpora";

/// The identity file mixed into every field corpus as source `identity`
/// (PLAN-NINA §3); a missing file is a manifest note, not an error.
pub const DEFAULT_IDENTITY_PATH: &str = "/srv/m-sda/unifur/identity/identity.jsonl";

/// The source name the identity file lands under.
pub const IDENTITY_SOURCE: &str = "identity";

fn default_max_doc_bytes() -> usize {
    DEFAULT_MAX_DOC_BYTES
}

fn default_near_dup_threshold() -> f64 {
    DEFAULT_NEAR_DUP_THRESHOLD
}

fn default_near_dup_hashes() -> usize {
    DEFAULT_NEAR_DUP_HASHES
}

fn default_identity_path() -> PathBuf {
    PathBuf::from(DEFAULT_IDENTITY_PATH)
}

/// Configuration for [`build_field_corpus`]. All fields carry serde defaults,
/// so a sidecar written by an older build still parses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldnormConfig {
    /// Documents larger than this many bytes are dropped and counted as
    /// `oversize`.
    #[serde(default = "default_max_doc_bytes")]
    pub max_doc_bytes: usize,
    /// MinHash near-duplicate threshold in (0, 1].
    #[serde(default = "default_near_dup_threshold")]
    pub near_dup_threshold: f64,
    /// MinHash signature length (estimate error ~1/sqrt of this).
    #[serde(default = "default_near_dup_hashes")]
    pub near_dup_hashes: usize,
    /// Identity JSONL mixed into every field corpus as source `identity`.
    #[serde(default = "default_identity_path")]
    pub identity_path: PathBuf,
}

impl Default for FieldnormConfig {
    fn default() -> Self {
        Self {
            max_doc_bytes: default_max_doc_bytes(),
            near_dup_threshold: default_near_dup_threshold(),
            near_dup_hashes: default_near_dup_hashes(),
            identity_path: default_identity_path(),
        }
    }
}

impl FieldnormConfig {
    /// Validate the tunables, naming the field at fault.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.max_doc_bytes > 0,
            "FieldnormConfig.max_doc_bytes must be positive"
        );
        anyhow::ensure!(
            self.near_dup_threshold > 0.0 && self.near_dup_threshold <= 1.0,
            "FieldnormConfig.near_dup_threshold must be in (0, 1], got {}",
            self.near_dup_threshold
        );
        anyhow::ensure!(
            self.near_dup_hashes >= 8,
            "FieldnormConfig.near_dup_hashes must be at least 8, got {}",
            self.near_dup_hashes
        );
        Ok(())
    }
}

/// How one raw dataset's rows become document text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    /// JSONL chat triples `{system, user, assistant}` (Fenrir, Trendyol).
    /// Rendered as instruction text: `user` then `assistant`. The `system`
    /// block is boilerplate repeated (nearly) unchanged across the whole
    /// source — baking it in would teach the model another project's persona
    /// and would make every document near-duplicate its neighbours on the
    /// shared prefix — so it is dropped and counted
    /// (`chat-system-block-dropped`).
    ChatJsonl,
    /// One JSON array of `{query, response, type, ...}` (MetaMathQA).
    JsonArrayQA,
    /// parquet2jsonl-converted NuminaMath rows `{problem, solution,
    /// problem_is_valid, solution_is_valid, ...}`. Rows whose validity flags
    /// are not both `"Yes"` are skipped and counted (`invalid-flag`).
    NuminaJsonl,
    /// parquet2jsonl-converted ai2_arc rows `{question, choices{text,
    /// label}, answerKey}`; rendered as the question, lettered options, and
    /// the answer line.
    Ai2ArcJsonl,
    /// parquet2jsonl-converted MedMCQA rows `{question, opa..opd, cop, exp,
    /// subject_name, topic_name}`; rendered as the question, lettered
    /// options, the answer, and the explanation when one exists.
    MedmcqaJsonl,
    /// parquet2jsonl-converted synthetic_text_to_sql rows `{sql_prompt,
    /// sql_context, sql, sql_explanation, ...}`; rendered as task, schema,
    /// SQL, explanation.
    TextToSqlJsonl,
    /// A Hugo Markdown tree (kubernetes-website `content/en`, docker-docs
    /// `content`): front matter, shortcodes, HTML and link syntax stripped by
    /// [`hugo_markdown_to_text`].
    MarkdownTree,
    /// A DocBook SGML tree (postgres-docs `src/sgml`): elements stripped,
    /// comments removed, CDATA kept verbatim, entities decoded, and anything
    /// unhandled counted — see [`sgml_to_text`].
    SgmlTree,
    /// The identity file: plain JSONL in any of the cybercorpus row shapes
    /// (`text`/`content`, `prompt`+`completion`, `instruction`+`response`).
    IdentityJsonl,
    /// NVD JSON 2.0 CVE rows (one `{"cve": {...}}` object per line, streamed
    /// out of a year feed by `/srv/m-sda/unifur/tools/fetch_pentest.py`).
    /// Rendered as id, publication date, English description, CWE weakness
    /// ids, CVSS severity when NVD scored it, and the reference list.
    /// `vulnStatus: "Rejected"` rows and `** RESERVED **` placeholders are
    /// skipped, counted (`rejected-status`, `reserved-placeholder`).
    NvdCveJsonl,
    /// The same rows as [`SourceFormat::NvdCveJsonl`], but a CVE is kept only
    /// when at least one of its references is tagged `Exploit`
    /// (`no-exploit-reference` skip) — the zeroday subfield's slice of the
    /// feeds.
    NvdCveExploitJsonl,
    /// MITRE ATT&CK STIX 2.1 objects (one per line, split out of
    /// `enterprise-attack.json`). Attack patterns and tactics are rendered;
    /// revoked/deprecated objects and every other object type are skipped,
    /// counted (`revoked-or-deprecated`, `stix-type-not-kept`).
    AttackStixJsonl,
    /// The same rows as [`SourceFormat::AttackStixJsonl`], keeping only
    /// attack patterns whose `kill_chain_phases` include `reconnaissance`
    /// (TA0043) or `resource-development` (TA0042) — the recon subfield's
    /// slice (`not-a-recon-technique` skip; tactics are not techniques and
    /// are skipped here too).
    AttackStixReconJsonl,
    /// MITRE CWE content XML (`cwe.xml`): split into `<Weakness>` entries by
    /// a balanced-tag scan ([`split_xml_entries`]); each entry rendered as
    /// name, description, extended description, consequences, detection
    /// methods, mitigations and observed examples, inner XHTML stripped by
    /// [`xml_inner_text`].
    CweXmlFile,
    /// MITRE CAPEC content XML (`capec.xml`): `<Attack_Pattern>` entries
    /// rendered as name, description, likelihood/severity, prerequisites,
    /// skills required, consequences, mitigations and example instances.
    CapecXmlFile,
}

impl SourceFormat {
    /// The `format` field written into each [`DatasetRecord`].
    pub fn record_format(self) -> &'static str {
        match self {
            Self::ChatJsonl
            | Self::NuminaJsonl
            | Self::Ai2ArcJsonl
            | Self::MedmcqaJsonl
            | Self::TextToSqlJsonl
            | Self::IdentityJsonl
            | Self::NvdCveJsonl
            | Self::NvdCveExploitJsonl
            | Self::AttackStixJsonl
            | Self::AttackStixReconJsonl => "jsonl",
            Self::JsonArrayQA => "json",
            Self::MarkdownTree => "markdown",
            Self::SgmlTree => "sgml",
            Self::CweXmlFile | Self::CapecXmlFile => "xml",
        }
    }
}

/// One raw dataset and how the normalizer reads it.
#[derive(Debug, Clone)]
pub struct SourceSpec {
    /// Source name; becomes `dataset/<name>.jsonl`, `text/<name>/` and the id
    /// prefix `<name>-<seq>`.
    pub name: String,
    pub format: SourceFormat,
    /// Input files/directories, relative to the datasets root (absolute paths
    /// are used as-is — the identity file lives outside the datasets root).
    /// For parquet-backed sources these are the *original* parquets, recorded
    /// for provenance; the rows are read from `converted` instead.
    pub inputs: Vec<PathBuf>,
    /// Parquet-backed sources: the JSONL under `<corpus>/raw/` that
    /// `parquet2jsonl.py` must have produced from `inputs`.
    pub converted: Option<String>,
    /// Inputs deliberately excluded (eval/test splits), recorded in the
    /// manifest so the exclusion is visible.
    pub noted_skips: Vec<String>,
    /// The JSONL rows were escape-encoded a second time on disk (the
    /// `*_escaped.jsonl` cyber sources: a 500-row sample shows 0 real
    /// newlines and tens of thousands of literal `\n` pairs). When set, one
    /// escape layer is removed before rendering. The layer is uniform, so
    /// unescaping restores the original text rather than corrupting it.
    pub escaped_jsonl: bool,
}

/// All sources of one field corpus.
#[derive(Debug, Clone)]
pub struct FieldSpec {
    /// Field name (`math`, `science`, …); the corpus lands at
    /// `<out-root>/<field>/`.
    pub field: String,
    pub sources: Vec<SourceSpec>,
}

fn source(name: &str, format: SourceFormat, inputs: &[&str]) -> SourceSpec {
    SourceSpec {
        name: name.to_string(),
        format,
        inputs: inputs.iter().map(PathBuf::from).collect(),
        converted: None,
        noted_skips: Vec::new(),
        escaped_jsonl: false,
    }
}

fn parquet_source(
    name: &str,
    format: SourceFormat,
    parquet_inputs: &[&str],
    converted: &str,
    noted_skips: &[&str],
) -> SourceSpec {
    SourceSpec {
        name: name.to_string(),
        format,
        inputs: parquet_inputs.iter().map(PathBuf::from).collect(),
        converted: Some(converted.to_string()),
        noted_skips: noted_skips.iter().map(|s| s.to_string()).collect(),
        escaped_jsonl: false,
    }
}

/// A JSONL chat source whose rows carry a second escape layer
/// (`SourceSpec::escaped_jsonl`).
fn escaped_chat_source(name: &str, input: &str) -> SourceSpec {
    SourceSpec {
        escaped_jsonl: true,
        ..source(name, SourceFormat::ChatJsonl, &[input])
    }
}

/// The converted NVD year feeds (`fetch_pentest.py` streams each feed's
/// `vulnerabilities` array into one-CVE-per-line JSONL). Years 2023–2025 per
/// the task scope: recent years suffice; the fetch tool records the choice.
fn nvd_year_jsonls() -> [&'static str; 3] {
    [
        "pentest/nvd-cve/nvdcve-2.0-2023.jsonl",
        "pentest/nvd-cve/nvdcve-2.0-2024.jsonl",
        "pentest/nvd-cve/nvdcve-2.0-2025.jsonl",
    ]
}

/// The field names [`field_spec`] knows. The pentest subfields name their
/// parent field with a slash, so `<out-root>/<field>` lands each subfield
/// corpus at `<out-root>/pentest/<subfield>/` (PLAN-NINA §1).
pub fn available_fields() -> &'static [&'static str] {
    &[
        "math",
        "science",
        "bio",
        "cyber",
        "tech",
        "pentest/recon",
        "pentest/bugfinding",
        "pentest/zeroday",
        "pentest/research",
        "pentest/methodology",
    ]
}

/// The source registry: which downloaded datasets make up each field corpus
/// and how each is read. An unknown field name is an error listing the known
/// ones.
pub fn field_spec(field: &str) -> Result<FieldSpec> {
    let sources = match field {
        "math" => vec![
            parquet_source(
                "numinamath",
                SourceFormat::NuminaJsonl,
                &[
                    "math/numinamath-1.5/data/train-00000-of-00003.parquet",
                    "math/numinamath-1.5/data/train-00001-of-00003.parquet",
                    "math/numinamath-1.5/data/train-00002-of-00003.parquet",
                ],
                "numinamath.jsonl",
                &[],
            ),
            source(
                "metamathqa",
                SourceFormat::JsonArrayQA,
                &["math/metamathqa/MetaMathQA-395K.json"],
            ),
        ],
        "science" => vec![parquet_source(
            "ai2-arc",
            SourceFormat::Ai2ArcJsonl,
            &[
                "science/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet",
                "science/ai2_arc/ARC-Challenge/validation-00000-of-00001.parquet",
                "science/ai2_arc/ARC-Easy/train-00000-of-00001.parquet",
                "science/ai2_arc/ARC-Easy/validation-00000-of-00001.parquet",
            ],
            "ai2-arc.jsonl",
            // ARC's test split is a standard benchmark: eval material, not
            // training data.
            &[
                "science/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet",
                "science/ai2_arc/ARC-Easy/test-00000-of-00001.parquet",
            ],
        )],
        "bio" => vec![parquet_source(
            "medmcqa",
            SourceFormat::MedmcqaJsonl,
            &[
                "bio/medmcqa/data/train-00000-of-00001.parquet",
                "bio/medmcqa/data/validation-00000-of-00001.parquet",
            ],
            "medmcqa.jsonl",
            // The MedMCQA test split ships without labels.
            &["bio/medmcqa/data/test-00000-of-00001.parquet"],
        )],
        "cyber" => vec![
            // Both `*_escaped.jsonl` files are escape-encoded twice: a
            // 500-row sample shows 0 real newlines and ~48k literal `\n`
            // pairs, so the layer is uniform and safe to remove.
            escaped_chat_source(
                "fenrir",
                "cyber/Fenrir-v2.1/220426-CyberSec-Dataset_escaped.jsonl",
            ),
            escaped_chat_source(
                "trendyol-cyber",
                "cyber/Trendyol-Cybersecurity/CyberSec-Dataset_escaped.jsonl",
            ),
        ],
        "tech" => vec![
            source(
                "kubernetes-docs",
                SourceFormat::MarkdownTree,
                &["tech/kubernetes-website/content/en"],
            ),
            source(
                "docker-docs",
                SourceFormat::MarkdownTree,
                &["tech/docker-docs/content"],
            ),
            parquet_source(
                "synthetic-text-to-sql",
                SourceFormat::TextToSqlJsonl,
                &["tech/synthetic_text_to_sql/synthetic_text_to_sql_train.snappy.parquet"],
                "synthetic-text-to-sql.jsonl",
                // Held-out split of the same synthetic generator.
                &["tech/synthetic_text_to_sql/synthetic_text_to_sql_test.snappy.parquet"],
            ),
            source(
                "postgres-docs",
                SourceFormat::SgmlTree,
                &["tech/postgres-docs/src/sgml"],
            ),
        ],
        // The pentest subfields (PLAN-NINA §1). All sources are license-
        // verified public feeds; per-source manifests with the verdicts live
        // beside the downloads under datasets/pentest/. HackTricks was
        // considered and SKIPPED: the repository carries no license at all
        // (HackTricks.SKIPPED.json records the check).
        "pentest/recon" => vec![source(
            "attack-recon",
            SourceFormat::AttackStixReconJsonl,
            &["pentest/attack-cti/attack-stix.jsonl"],
        )],
        "pentest/bugfinding" => vec![source(
            "cwe",
            SourceFormat::CweXmlFile,
            &["pentest/cwe/cwe.xml"],
        )],
        "pentest/zeroday" => vec![
            source(
                "capec",
                SourceFormat::CapecXmlFile,
                &["pentest/capec/capec.xml"],
            ),
            source(
                "nvd-cve-exploit",
                SourceFormat::NvdCveExploitJsonl,
                &nvd_year_jsonls(),
            ),
        ],
        "pentest/research" => vec![source(
            "nvd-cve",
            SourceFormat::NvdCveJsonl,
            &nvd_year_jsonls(),
        )],
        "pentest/methodology" => vec![
            source(
                "attack-full",
                SourceFormat::AttackStixJsonl,
                &["pentest/attack-cti/attack-stix.jsonl"],
            ),
            source(
                "owasp-cheatsheets",
                SourceFormat::MarkdownTree,
                &["pentest/owasp-cheatsheetseries/cheatsheets"],
            ),
        ],
        other => {
            return Err(anyhow::anyhow!(
                "unknown field {other:?}; known fields: {}",
                available_fields().join(", ")
            ));
        }
    };
    Ok(FieldSpec {
        field: field.to_string(),
        sources,
    })
}

/// Per-source counts in [`FieldManifest`], and the content of the completion
/// marker a finished source leaves under `dataset/.fieldnorm/`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceStats {
    /// Kept documents.
    pub documents: usize,
    /// Kept characters.
    pub chars: usize,
    /// Cheat-class name -> documents the cheat stack dropped because of it
    /// (reject grade: [`Severity::Suppression`] and above).
    #[serde(default)]
    pub dropped_by_class: BTreeMap<String, usize>,
    /// Cheat-class name -> quarantine-grade ([`Severity::Evasion`]) findings,
    /// counted but not dropped — the quarantine classes are calibrated for
    /// code snippets, and on prose they fire on ordinary English (see the
    /// module docs for the measurement behind this policy).
    #[serde(default)]
    pub quarantine_findings: BTreeMap<String, usize>,
    /// Documents dropped as near-duplicates of an earlier kept document.
    #[serde(default)]
    pub near_duplicates: usize,
    /// Documents over [`FieldnormConfig::max_doc_bytes`].
    #[serde(default)]
    pub oversize: usize,
    /// Rows that never became a document, by reason (`invalid-flag`,
    /// `empty-text`, `unrecognized-row`, …).
    #[serde(default)]
    pub rows_skipped: BTreeMap<String, usize>,
    /// Counted events that are neither drops nor skips: boilerplate blocks
    /// removed (`chat-system-block-dropped`), markdown shortcodes stripped,
    /// SGML entities decoded, …
    #[serde(default)]
    pub counters: BTreeMap<String, usize>,
    /// SGML constructs the stripper did not handle, by construct
    /// (`entity:&foo;`, `marked-section`, …). A nonzero map is a finding,
    /// not a failure — the count is the honest record.
    #[serde(default)]
    pub unhandled_constructs: BTreeMap<String, usize>,
    /// Human-readable notes (excluded inputs, …).
    #[serde(default)]
    pub notes: Vec<String>,
}

/// The identity mix-in record in [`FieldManifest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityNote {
    /// The file that was mixed in.
    pub path: PathBuf,
    /// Documents kept from it.
    pub documents: usize,
}

/// `dataset/manifest.json`: the provenance record of one field-corpus build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldManifest {
    /// The field that was built.
    pub field: String,
    /// The datasets root that was read.
    pub datasets_root: PathBuf,
    /// The corpus directory that was written (`<out-root>/<field>`).
    pub corpus_dir: PathBuf,
    /// The effective configuration.
    pub config: FieldnormConfig,
    /// Per-source counts, keyed by source name (sorted).
    #[serde(default)]
    pub sources: BTreeMap<String, SourceStats>,
    /// Kept documents across all sources.
    pub total_documents: usize,
    /// Kept characters across all sources.
    pub total_chars: usize,
    /// Cheat-class name -> documents dropped because of it, all sources
    /// (reject grade only — see the module docs for the drop policy).
    #[serde(default)]
    pub dropped_by_class: BTreeMap<String, usize>,
    /// Quarantine-grade findings, counted but not dropped, all sources.
    #[serde(default)]
    pub quarantine_findings: BTreeMap<String, usize>,
    /// Near-duplicate drops, all sources.
    #[serde(default)]
    pub near_duplicates: usize,
    /// Oversize drops, all sources.
    #[serde(default)]
    pub oversize: usize,
    /// The identity mix-in, when the identity file existed.
    #[serde(default)]
    pub identity: Option<IdentityNote>,
    /// The parquet2jsonl.py invocations that produced this corpus's
    /// converted inputs, copied verbatim from
    /// `raw/parquet2jsonl-conversions.jsonl`.
    #[serde(default)]
    pub parquet_conversions: Vec<Value>,
    /// Sources skipped this run because a completion marker existed.
    #[serde(default)]
    pub resumed_sources: Vec<String>,
    /// Human-readable notes.
    #[serde(default)]
    pub notes: Vec<String>,
}

/// One rendered document before filtering.
struct Doc {
    /// Provenance: the input path relative to the datasets root, with
    /// `#<row>` for JSONL/JSON-array rows.
    origin: String,
    text: String,
}

/// What a row renderer made of one row.
enum RowOutcome {
    /// A document.
    Doc(String),
    /// A document, plus a counter event (e.g. `chat-system-block-dropped`).
    DocCounted(String, &'static str),
    /// Not a document; counted under the reason key.
    Skip(&'static str),
}

fn get_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

/// Remove one layer of escape sequences (`\\n`, `\\t`, `\\"`, `\\\\`, …).
/// Only called for sources flagged [`SourceSpec::escaped_jsonl`]; an unknown
/// escape keeps both characters.
fn unescape_literals(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('\'') => out.push('\''),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Fenrir/Trendyol `{system, user, assistant}` -> instruction text. The
/// boilerplate system block is dropped (and counted), not rendered: it is the
/// same persona paragraph across the whole source. An `escaped_jsonl`
/// source's second escape layer is removed first (and counted when it
/// changed anything).
fn render_chat(value: &Value, escaped: bool) -> RowOutcome {
    let (Some(user), Some(assistant)) = (get_str(value, "user"), get_str(value, "assistant"))
    else {
        return RowOutcome::Skip("chat-row-without-user-or-assistant");
    };
    let (user, assistant) = if escaped {
        (unescape_literals(user), unescape_literals(assistant))
    } else {
        (user.to_string(), assistant.to_string())
    };
    let text = format!("{}\n\n{}", user.trim(), assistant.trim());
    match get_str(value, "system") {
        Some(system) if !system.trim().is_empty() => {
            RowOutcome::DocCounted(text, "chat-system-block-dropped")
        }
        _ => RowOutcome::Doc(text),
    }
}

/// MetaMathQA `{query, response}` -> `query\n\nresponse`.
fn render_metamath(value: &Value) -> RowOutcome {
    let (Some(query), Some(response)) = (get_str(value, "query"), get_str(value, "response"))
    else {
        return RowOutcome::Skip("metamath-row-without-query-or-response");
    };
    RowOutcome::Doc(format!("{}\n\n{}", query.trim(), response.trim()))
}

/// NuminaMath `{problem, solution, problem_is_valid, solution_is_valid}`.
/// Rows the source itself flags as invalid or incomplete are skipped, not
/// rendered — the flags are the dataset's own quality labels.
fn render_numina(value: &Value) -> RowOutcome {
    if get_str(value, "problem_is_valid") != Some("Yes")
        || get_str(value, "solution_is_valid") != Some("Yes")
    {
        return RowOutcome::Skip("invalid-flag");
    }
    let (Some(problem), Some(solution)) = (get_str(value, "problem"), get_str(value, "solution"))
    else {
        return RowOutcome::Skip("numina-row-without-problem-or-solution");
    };
    RowOutcome::Doc(format!("{}\n\n{}", problem.trim(), solution.trim()))
}

/// ai2_arc `{question, choices{text, label}, answerKey}` -> question,
/// lettered options, answer line.
fn render_ai2_arc(value: &Value) -> RowOutcome {
    let Some(question) = get_str(value, "question") else {
        return RowOutcome::Skip("arc-row-without-question");
    };
    let choices = value.get("choices").and_then(Value::as_object);
    let texts = choices
        .and_then(|c| c.get("text"))
        .and_then(Value::as_array);
    let labels = choices
        .and_then(|c| c.get("label"))
        .and_then(Value::as_array);
    let (Some(texts), Some(labels)) = (texts, labels) else {
        return RowOutcome::Skip("arc-row-without-choices");
    };
    let answer = get_str(value, "answerKey").unwrap_or("").trim();
    if answer.is_empty() {
        return RowOutcome::Skip("arc-row-without-answer");
    }
    let mut text = question.trim().to_string();
    text.push_str("\n\n");
    for (label, choice) in labels.iter().zip(texts.iter()) {
        let (Some(label), Some(choice)) = (label.as_str(), choice.as_str()) else {
            return RowOutcome::Skip("arc-row-with-non-string-choice");
        };
        text.push_str(&format!("{label}. {}\n", choice.trim()));
    }
    text.push_str(&format!("\nAnswer: {answer}"));
    RowOutcome::Doc(text)
}

/// MedMCQA `{question, opa..opd, cop, exp, subject_name, topic_name}`.
fn render_medmcqa(value: &Value) -> RowOutcome {
    let Some(question) = get_str(value, "question") else {
        return RowOutcome::Skip("medmcqa-row-without-question");
    };
    let mut options = Vec::with_capacity(4);
    for key in ["opa", "opb", "opc", "opd"] {
        let Some(option) = get_str(value, key) else {
            return RowOutcome::Skip("medmcqa-row-without-options");
        };
        options.push(option.trim());
    }
    let cop = value.get("cop").and_then(Value::as_i64);
    let Some(cop) = cop else {
        return RowOutcome::Skip("medmcqa-row-without-correct-option");
    };
    let Some(letter) = (0..=3).contains(&cop).then(|| (b'A' + cop as u8) as char) else {
        return RowOutcome::Skip("medmcqa-row-with-out-of-range-correct-option");
    };
    let mut text = question.trim().to_string();
    text.push_str("\n\n");
    for (i, option) in options.iter().enumerate() {
        text.push_str(&format!("{}. {}\n", (b'A' + i as u8) as char, option));
    }
    text.push_str(&format!("\nAnswer: {letter}"));
    if let Some(exp) = get_str(value, "exp") {
        let exp = exp.trim();
        if !exp.is_empty() {
            // No "Explanation:" label: a line starting with one is the cheat
            // scanner's "prose where code should be" marker, and the label
            // would put that marker on every row.
            text.push_str(&format!("\n\n{exp}"));
        }
    }
    RowOutcome::Doc(text)
}

/// synthetic_text_to_sql `{sql_prompt, sql_context, sql, sql_explanation}`.
fn render_text_to_sql(value: &Value) -> RowOutcome {
    let (Some(prompt), Some(context), Some(sql)) = (
        get_str(value, "sql_prompt"),
        get_str(value, "sql_context"),
        get_str(value, "sql"),
    ) else {
        return RowOutcome::Skip("sql-row-missing-fields");
    };
    let mut text = format!(
        "Task: {}\n\nSchema:\n{}\n\nSQL:\n{}",
        prompt.trim(),
        context.trim(),
        sql.trim()
    );
    if let Some(explanation) = get_str(value, "sql_explanation") {
        let explanation = explanation.trim();
        if !explanation.is_empty() {
            // Bare prose, no "Explanation:" label — see render_medmcqa.
            text.push_str(&format!("\n\n{explanation}"));
        }
    }
    RowOutcome::Doc(text)
}

/// The identity file's row shapes are the cybercorpus ones: `text`/`content`,
/// then a `prompt`+`completion` or `instruction`+`response` pair, then a lone
/// `completion`/`response`/`answer`.
fn render_identity(value: &Value) -> RowOutcome {
    if let Some(text) = get_str(value, "text").or_else(|| get_str(value, "content")) {
        return RowOutcome::Doc(text.trim().to_string());
    }
    for (a, b) in [("prompt", "completion"), ("instruction", "response")] {
        if let (Some(first), Some(second)) = (get_str(value, a), get_str(value, b)) {
            return RowOutcome::Doc(format!("{}\n\n{}", first.trim(), second.trim()));
        }
    }
    match get_str(value, "completion")
        .or_else(|| get_str(value, "response"))
        .or_else(|| get_str(value, "answer"))
    {
        Some(lone) => RowOutcome::Doc(lone.trim().to_string()),
        None => RowOutcome::Skip("unrecognized-row"),
    }
}

// ---------------------------------------------------------------------------
// NVD JSON 2.0 CVE rows
// ---------------------------------------------------------------------------

/// The English description of an NVD `descriptions` array.
fn nvd_en_description(cve: &Value) -> Option<&str> {
    cve.get("descriptions")
        .and_then(Value::as_array)
        .and_then(|ds| {
            ds.iter()
                .find(|d| get_str(d, "lang") == Some("en"))
                .and_then(|d| get_str(d, "value"))
        })
}

/// The CWE ids of an NVD `weaknesses` array (`CWE-79`, `NVD-CWE-Other`, …).
fn nvd_weakness_ids(cve: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(weaknesses) = cve.get("weaknesses").and_then(Value::as_array) {
        for weakness in weaknesses {
            if let Some(descriptions) = weakness.get("description").and_then(Value::as_array) {
                for d in descriptions {
                    if let Some(id) = get_str(d, "value") {
                        let id = id.trim();
                        if !id.is_empty() && !out.iter().any(|seen| seen == id) {
                            out.push(id.to_string());
                        }
                    }
                }
            }
        }
    }
    out
}

/// One reference: the URL plus whether NVD tagged it `Exploit`.
fn nvd_references(cve: &Value) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    if let Some(references) = cve.get("references").and_then(Value::as_array) {
        for reference in references {
            let Some(url) = get_str(reference, "url") else {
                continue;
            };
            let exploit = reference
                .get("tags")
                .and_then(Value::as_array)
                .is_some_and(|tags| tags.iter().any(|t| t.as_str() == Some("Exploit")));
            out.push((url.trim().to_string(), exploit));
        }
    }
    out
}

/// NVD's own severity line, when a CVSS v3.x metric exists: the primary (or
/// first) metric's `baseSeverity` and `baseScore`.
fn nvd_severity(cve: &Value) -> Option<String> {
    let metrics = cve.get("metrics")?;
    for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"] {
        if let Some(list) = metrics.get(key).and_then(Value::as_array) {
            for metric in list {
                let data = metric.get("cvssData");
                let severity = get_str(metric, "baseSeverity");
                let score = data.and_then(|d| d.get("baseScore")).and_then(|s| {
                    s.as_f64()
                        .or_else(|| s.as_str().and_then(|t| t.parse().ok()))
                });
                match (severity, score) {
                    (Some(sev), Some(score)) => return Some(format!("{sev} (base score {score})")),
                    (Some(sev), None) => return Some(sev.to_string()),
                    _ => continue,
                }
            }
        }
    }
    None
}

/// NVD 2.0 `{"cve": {...}}` -> a document: id and publication date, the
/// English description, CWE ids, NVD's severity, and the reference list with
/// exploit tags marked. `exploit_only` keeps only CVEs with an
/// `Exploit`-tagged reference (the zeroday slice). Rejected entries and
/// `** RESERVED **` placeholders are skipped — a placeholder is not research
/// material, and near-dedup would collapse them into a single misleading
/// survivor.
fn render_nvd_cve(value: &Value, exploit_only: bool) -> RowOutcome {
    let Some(cve) = value.get("cve") else {
        return RowOutcome::Skip("nvd-row-without-cve");
    };
    let Some(id) = get_str(cve, "id") else {
        return RowOutcome::Skip("nvd-row-without-id");
    };
    if get_str(cve, "vulnStatus") == Some("Rejected") {
        return RowOutcome::Skip("rejected-status");
    }
    let Some(description) = nvd_en_description(cve) else {
        return RowOutcome::Skip("nvd-row-without-en-description");
    };
    if description.trim_start().starts_with("** RESERVED **") {
        return RowOutcome::Skip("reserved-placeholder");
    }
    let references = nvd_references(cve);
    let has_exploit = references.iter().any(|(_, exploit)| *exploit);
    if exploit_only && !has_exploit {
        return RowOutcome::Skip("no-exploit-reference");
    }
    let published = get_str(cve, "published").unwrap_or("");
    let published = published.split('T').next().unwrap_or(published);
    let mut text = format!("{id} (published {published})\n\n{}", description.trim());
    let weaknesses = nvd_weakness_ids(cve);
    if !weaknesses.is_empty() {
        text.push_str(&format!("\n\nWeaknesses: {}", weaknesses.join(", ")));
    }
    if let Some(severity) = nvd_severity(cve) {
        text.push_str(&format!("\n\nSeverity: {severity}"));
    }
    if !references.is_empty() {
        text.push_str("\n\nReferences:");
        for (url, exploit) in &references {
            if *exploit {
                text.push_str(&format!("\n- {url} [Exploit]"));
            } else {
                text.push_str(&format!("\n- {url}"));
            }
        }
    }
    RowOutcome::Doc(text)
}

// ---------------------------------------------------------------------------
// MITRE ATT&CK STIX rows
// ---------------------------------------------------------------------------

/// The ATT&CK id (`T1595`, `TA0043`) of a STIX object: the `mitre-attack`
/// external reference's `external_id`.
fn stix_attack_id(value: &Value) -> Option<String> {
    value
        .get("external_references")
        .and_then(Value::as_array)
        .and_then(|refs| {
            refs.iter().find_map(|r| {
                if get_str(r, "source_name") == Some("mitre-attack") {
                    get_str(r, "external_id").map(str::to_string)
                } else {
                    None
                }
            })
        })
}

/// The tactic shortnames (`reconnaissance`, …) of an attack pattern's
/// `mitre-attack` kill-chain phases.
fn stix_phases(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(phases) = value.get("kill_chain_phases").and_then(Value::as_array) {
        for phase in phases {
            if get_str(phase, "kill_chain_name") != Some("mitre-attack") {
                continue;
            }
            if let Some(name) = get_str(phase, "phase_name") {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// A STIX object from the ATT&CK bundle -> a document. Attack patterns render
/// as id, name, description and their tactics; tactics render as id, name and
/// description. With `recon_only`, only attack patterns in the
/// reconnaissance / resource-development tactics are kept. Revoked and
/// deprecated objects are historical record, not doctrine: skipped.
fn render_stix(value: &Value, recon_only: bool) -> RowOutcome {
    if value.get("revoked").and_then(Value::as_bool) == Some(true)
        || value.get("x_mitre_deprecated").and_then(Value::as_bool) == Some(true)
    {
        return RowOutcome::Skip("revoked-or-deprecated");
    }
    match get_str(value, "type") {
        Some("attack-pattern") => {
            let phases = stix_phases(value);
            if recon_only
                && !phases
                    .iter()
                    .any(|p| p == "reconnaissance" || p == "resource-development")
            {
                return RowOutcome::Skip("not-a-recon-technique");
            }
            let Some(name) = get_str(value, "name") else {
                return RowOutcome::Skip("stix-row-without-name");
            };
            let Some(description) = get_str(value, "description") else {
                return RowOutcome::Skip("stix-row-without-description");
            };
            let id = stix_attack_id(value).unwrap_or_else(|| "attack-pattern".to_string());
            let mut text = format!("{id}: {}\n\n{}", name.trim(), description.trim());
            if !phases.is_empty() {
                text.push_str(&format!("\n\nTactics: {}", phases.join(", ")));
            }
            if value
                .get("x_mitre_is_subtechnique")
                .and_then(Value::as_bool)
                == Some(true)
            {
                text.push_str("\n\nThis is a sub-technique of a broader ATT&CK technique.");
            }
            RowOutcome::Doc(text)
        }
        Some("x-mitre-tactic") if !recon_only => {
            let Some(name) = get_str(value, "name") else {
                return RowOutcome::Skip("stix-row-without-name");
            };
            let Some(description) = get_str(value, "description") else {
                return RowOutcome::Skip("stix-row-without-description");
            };
            let id = stix_attack_id(value).unwrap_or_else(|| "tactic".to_string());
            RowOutcome::Doc(format!(
                "{id}: {} (tactic)\n\n{}",
                name.trim(),
                description.trim()
            ))
        }
        _ => RowOutcome::Skip("stix-type-not-kept"),
    }
}

// ---------------------------------------------------------------------------
// Hugo Markdown -> plain text
// ---------------------------------------------------------------------------

/// Counted events of [`hugo_markdown_to_text`].
#[derive(Debug, Clone, Default)]
pub struct MarkdownStrip {
    /// A YAML front-matter block was removed.
    pub front_matter: bool,
    /// Hugo shortcode tags (`{{< … >}}`, `{{% … %}}`) removed.
    pub shortcodes: usize,
    /// HTML tags removed.
    pub html_tags: usize,
    /// HTML comments removed.
    pub html_comments: usize,
}

fn starts_with_chars(chars: &[char], at: usize, needle: &str) -> bool {
    let n: Vec<char> = needle.chars().collect();
    chars.len() >= at + n.len() && chars[at..at + n.len()] == n[..]
}

/// Scan an inline markdown line to plain text: shortcodes, images, links,
/// autolinks, inline HTML tags and emphasis markers removed; code spans keep
/// their content.
fn strip_md_inline(line: &str, stats: &mut MarkdownStrip) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0usize;
    while i < chars.len() {
        // Hugo shortcodes, inline or whole-line.
        let shortcode_close = if starts_with_chars(&chars, i, "{{<") {
            Some(">}}")
        } else if starts_with_chars(&chars, i, "{{%") {
            Some("%}}")
        } else {
            None
        };
        if let Some(close) = shortcode_close {
            let mut j = i + 3;
            while j < chars.len() && !starts_with_chars(&chars, j, close) {
                j += 1;
            }
            stats.shortcodes += 1;
            i = if j < chars.len() { j + 3 } else { chars.len() };
            continue;
        }
        let c = chars[i];
        // Image: ![alt](src) -> alt.
        if c == '!' && chars.get(i + 1) == Some(&'[') {
            let mut j = i + 2;
            while j < chars.len() && !starts_with_chars(&chars, j, "](") {
                j += 1;
            }
            if j < chars.len() {
                out.extend(chars[i + 2..j].iter());
                let mut k = j + 2;
                while k < chars.len() && chars[k] != ')' {
                    k += 1;
                }
                i = if k < chars.len() { k + 1 } else { chars.len() };
                continue;
            }
        }
        // Link: [text](url) -> text, or reference [text][id] -> text.
        if c == '[' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != ']' {
                j += 1;
            }
            if j < chars.len() {
                let next = chars.get(j + 1);
                if next == Some(&'(') {
                    let mut k = j + 2;
                    while k < chars.len() && chars[k] != ')' {
                        k += 1;
                    }
                    out.extend(chars[i + 1..j].iter());
                    i = if k < chars.len() { k + 1 } else { chars.len() };
                    continue;
                }
                if next == Some(&'[') {
                    let mut k = j + 2;
                    while k < chars.len() && chars[k] != ']' {
                        k += 1;
                    }
                    out.extend(chars[i + 1..j].iter());
                    i = if k < chars.len() { k + 1 } else { chars.len() };
                    continue;
                }
            }
        }
        // Autolink <https://…> keeps the URL text; <…> that is clearly an
        // HTML tag (a closer, attributes, or a known tag name) is dropped;
        // a bare <word-with-dashes> is a docs placeholder and stays.
        if c == '<' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            if j < chars.len() {
                let inner: String = chars[i + 1..j].iter().collect();
                let lower = inner.to_lowercase();
                if lower.starts_with("http://")
                    || lower.starts_with("https://")
                    || lower.starts_with("mailto:")
                {
                    out.push_str(&inner);
                } else {
                    let bare = !inner.contains(char::is_whitespace)
                        && !inner.contains('=')
                        && !inner.contains('/');
                    const HTML_TAGS: &[&str] = &[
                        "a", "abbr", "b", "br", "code", "div", "em", "h1", "h2", "h3", "h4", "h5",
                        "h6", "hr", "i", "img", "kbd", "li", "ol", "p", "pre", "span", "strong",
                        "sub", "sup", "table", "td", "th", "tr", "ul",
                    ];
                    if bare && !HTML_TAGS.contains(&lower.as_str()) {
                        out.push('<');
                        out.push_str(&inner);
                        out.push('>');
                    } else {
                        stats.html_tags += 1;
                    }
                }
                i = j + 1;
                continue;
            }
        }
        // Emphasis and code-span markers: keep the content, drop the marks.
        if c == '`' {
            i += 1;
            continue;
        }
        if (c == '*' || c == '_') && chars.get(i + 1) == Some(&c) {
            i += 2;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Strip Hugo Markdown to honest plain text: YAML front matter removed,
/// fenced code kept verbatim (the fence markers dropped), Hugo shortcode
/// *tags* removed (their inner content stays — a `{{< note >}}` block's body
/// is real prose), heading and blockquote markers removed, links reduced to
/// their text, inline HTML tags removed, HTML comments dropped. Everything
/// removed is counted in the returned [`MarkdownStrip`].
pub fn hugo_markdown_to_text(input: &str, stats_out: &mut MarkdownStrip) -> String {
    let mut body = input;
    if body.starts_with("---\n") || body.starts_with("---\r\n") {
        // Front matter ends at the next line that is exactly `---`.
        let after_open = body.find('\n').map(|n| n + 1).unwrap_or(body.len());
        if let Some(close) = body[after_open..].find("\n---") {
            let tail = &body[after_open + close + 1..];
            if let Some(eol) = tail.find('\n') {
                stats_out.front_matter = true;
                body = &tail[eol + 1..];
            }
        }
    }

    let mut out = String::with_capacity(body.len());
    let mut in_fence = false;
    let mut in_html_comment = false;
    for raw_line in body.lines() {
        let trimmed = raw_line.trim_start();
        if in_html_comment {
            if let Some(end) = raw_line.find("-->") {
                in_html_comment = false;
                stats_out.html_comments += 1;
                let rest = &raw_line[end + 3..];
                if !rest.trim().is_empty() {
                    out.push_str(&strip_md_inline(rest, stats_out));
                    out.push('\n');
                }
            }
            continue;
        }
        if let Some(start) = raw_line.find("<!--") {
            stats_out.html_comments += 1;
            if !raw_line[start..].contains("-->") {
                in_html_comment = true;
            }
            let rest = raw_line[..start].to_string();
            let after = raw_line[start..]
                .find("-->")
                .map(|e| &raw_line[start + e + 3..])
                .unwrap_or("");
            let combined = format!("{rest}{after}");
            if !combined.trim().is_empty() {
                out.push_str(&strip_md_inline(&combined, stats_out));
                out.push('\n');
            }
            continue;
        }
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            out.push_str(raw_line);
            out.push('\n');
            continue;
        }
        // Link reference definitions carry no prose.
        if trimmed.starts_with('[') && trimmed.contains("]:") {
            let label_end = trimmed.find("]:");
            if label_end.is_some_and(|e| e < 40) {
                continue;
            }
        }
        let mut line = trimmed;
        if let Some(rest) = line.strip_prefix('>') {
            line = rest.trim_start();
        }
        let hashes = line.chars().take_while(|c| *c == '#').count();
        if hashes > 0 && line.chars().nth(hashes) == Some(' ') {
            line = line[hashes + 1..].trim_start();
        }
        let stripped = strip_md_inline(line, stats_out);
        out.push_str(stripped.trim_end());
        out.push('\n');
    }
    collapse_blank_lines(&out)
}

/// Collapse runs of 3+ newlines to a blank line and trim the ends.
fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                out.push('\n');
            }
        } else {
            blank_run = 0;
            out.push_str(line);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// DocBook SGML -> plain text
// ---------------------------------------------------------------------------

/// Counted events of [`sgml_to_text`].
#[derive(Debug, Clone, Default)]
pub struct SgmlStrip {
    /// Elements stripped.
    pub elements: usize,
    /// Comments removed.
    pub comments: usize,
    /// Entities decoded.
    pub entities_decoded: usize,
    /// CDATA sections whose content was kept verbatim.
    pub cdata_sections: usize,
    /// Constructs not handled, by name (`entity:&foo;`, `marked-section`,
    /// `declaration`, `processing-instruction`). Counted, never silently
    /// dropped.
    pub unhandled: BTreeMap<String, usize>,
}

/// Elements that end a block of text: a newline is emitted when one closes
/// (or opens, for the self-standing ones). Anything not listed is inline and
/// is stripped without whitespace.
const SGML_BLOCK_ELEMENTS: &[&str] = &[
    "appendix",
    "article",
    "blockquote",
    "book",
    "caution",
    "chapter",
    "entry",
    "example",
    "figure",
    "formalpara",
    "glossentry",
    "important",
    "indexterm",
    "informaltable",
    "itemizedlist",
    "listitem",
    "literallayout",
    "note",
    "orderedlist",
    "para",
    "part",
    "preface",
    "procedure",
    "programlisting",
    "refentry",
    "refsect1",
    "refsect2",
    "refsect3",
    "refsynopsisdiv",
    "row",
    "screen",
    "sect1",
    "sect2",
    "sect3",
    "sect4",
    "sect5",
    "step",
    "synopsis",
    "table",
    "tbody",
    "term",
    "thead",
    "tip",
    "title",
    "variablelist",
    "varlistentry",
    "warning",
];

/// The named entities the PostgreSQL 17.6 DocBook sources actually use
/// (measured over `src/sgml`): the XML predefined set, the Latin-1 handful,
/// and the DocBook customization entities. `version` is the release the
/// tarball documents — 17.6 per the download manifest. A named entity outside
/// this map is dropped and counted under `entity:&name;`, never silently.
fn sgml_entity(name: &str) -> Option<String> {
    let decoded = match name {
        "amp" => "&".to_string(),
        "lt" => "<".to_string(),
        "gt" => ">".to_string(),
        "quot" => "\"".to_string(),
        "apos" => "'".to_string(),
        "sect" => "§".to_string(),
        "mdash" => "—".to_string(),
        "ndash" => "–".to_string(),
        "nbsp" | "zwsp" => " ".to_string(),
        "bull" => "•".to_string(),
        "rarr" => "→".to_string(),
        "copy" => "©".to_string(),
        "eacute" => "é".to_string(),
        "ocirc" => "ô".to_string(),
        "aacute" => "á".to_string(),
        "oacute" => "ó".to_string(),
        "aring" => "å".to_string(),
        "AElig" => "Æ".to_string(),
        "Aacute" => "Á".to_string(),
        "Acirc" => "Â".to_string(),
        "Agrave" => "À".to_string(),
        "Aring" => "Å".to_string(),
        "Atilde" => "Ã".to_string(),
        "Auml" => "Ä".to_string(),
        "ntilde" => "ñ".to_string(),
        "oslash" => "ø".to_string(),
        "ldquo" => "“".to_string(),
        "rdquo" => "”".to_string(),
        "frac12" => "½".to_string(),
        "frac14" => "¼".to_string(),
        "frac34" => "¾".to_string(),
        "pi" => "π".to_string(),
        // The release series the tarball documents (PostgreSQL 17.6 per the
        // download manifest).
        "majorversion" => "17".to_string(),
        "version" => "17.6".to_string(),
        _ => return None,
    };
    Some(decoded)
}

/// Strip DocBook SGML to plain text with a real element pass, not a
/// whole-file regex: comments removed, CDATA sections kept verbatim (their
/// content is example code, and tag-stripping it would corrupt it), elements
/// stripped with block-level breaks preserved, entities decoded through the
/// built-in entity map. Marked sections that are not CDATA, declarations and
/// processing instructions are counted as unhandled; so is every unknown
/// entity.
pub fn sgml_to_text(input: &str, stats: &mut SgmlStrip) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;
    let mut pending_block = false;
    while i < chars.len() {
        if starts_with_chars(&chars, i, "<!--") {
            let mut j = i + 4;
            while j < chars.len() && !starts_with_chars(&chars, j, "-->") {
                j += 1;
            }
            stats.comments += 1;
            i = if j < chars.len() { j + 3 } else { chars.len() };
            continue;
        }
        if starts_with_chars(&chars, i, "<![CDATA[") {
            let mut j = i + 9;
            while j < chars.len() && !starts_with_chars(&chars, j, "]]>") {
                j += 1;
            }
            stats.cdata_sections += 1;
            out.extend(chars[i + 9..j].iter());
            i = if j < chars.len() { j + 3 } else { chars.len() };
            continue;
        }
        if starts_with_chars(&chars, i, "<![") {
            let mut j = i + 3;
            while j < chars.len() && !starts_with_chars(&chars, j, "]]>") {
                j += 1;
            }
            *stats
                .unhandled
                .entry("marked-section".to_string())
                .or_insert(0) += 1;
            // The inner text is real documentation guarded by a conditional;
            // keep it rather than dropping content.
            out.extend(chars[i + 3..j].iter());
            i = if j < chars.len() { j + 3 } else { chars.len() };
            continue;
        }
        if starts_with_chars(&chars, i, "<!") {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            *stats
                .unhandled
                .entry("declaration".to_string())
                .or_insert(0) += 1;
            i = if j < chars.len() { j + 1 } else { chars.len() };
            continue;
        }
        if starts_with_chars(&chars, i, "<?") {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            *stats
                .unhandled
                .entry("processing-instruction".to_string())
                .or_insert(0) += 1;
            i = if j < chars.len() { j + 1 } else { chars.len() };
            continue;
        }
        if chars[i] == '<' {
            let mut j = i + 1;
            let closing = j < chars.len() && chars[j] == '/';
            if closing {
                j += 1;
            }
            let name_start = j;
            while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '-') {
                j += 1;
            }
            let name: String = chars[name_start..j].iter().collect();
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            stats.elements += 1;
            if SGML_BLOCK_ELEMENTS.contains(&name.as_str()) {
                pending_block = true;
            } else if closing {
                pending_block = false;
            }
            i = if j < chars.len() { j + 1 } else { chars.len() };
            continue;
        }
        if chars[i] == '&' {
            let mut j = i + 1;
            while j < chars.len() && j - i <= 40 && chars[j] != ';' && !chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == ';' {
                let name: String = chars[i + 1..j].iter().collect();
                let decoded = if let Some(number) = name.strip_prefix('#') {
                    let (digits, radix) = match number
                        .strip_prefix('x')
                        .or_else(|| number.strip_prefix('X'))
                    {
                        Some(hex) => (hex, 16),
                        None => (number, 10),
                    };
                    u32::from_str_radix(digits, radix)
                        .ok()
                        .and_then(char::from_u32)
                        .map(|c| c.to_string())
                } else {
                    sgml_entity(&name)
                };
                match decoded {
                    Some(text) => {
                        stats.entities_decoded += 1;
                        out.push_str(&text);
                    }
                    None => {
                        *stats
                            .unhandled
                            .entry(format!("entity:&{name};"))
                            .or_insert(0) += 1;
                    }
                }
                i = j + 1;
                continue;
            }
            // No `;` nearby: a literal ampersand in prose.
            out.push('&');
            i += 1;
            continue;
        }
        if pending_block && chars[i] == '\n' {
            // The tag already carries the break; a newline right after a
            // block boundary would only add a blank line.
            pending_block = false;
            i += 1;
            continue;
        }
        if pending_block && !chars[i].is_whitespace() {
            out.push('\n');
            pending_block = false;
        }
        out.push(chars[i]);
        i += 1;
    }
    if pending_block {
        out.push('\n');
    }
    collapse_blank_lines(&out)
}

// ---------------------------------------------------------------------------
// MITRE CWE/CAPEC XML -> plain text
// ---------------------------------------------------------------------------

/// Counted events of [`xml_inner_text`].
#[derive(Debug, Clone, Default)]
pub struct XmlStrip {
    /// Elements stripped.
    pub elements: usize,
    /// Entities decoded.
    pub entities_decoded: usize,
    /// Constructs not handled, by name (`entity:&foo;`, `comment`,
    /// `processing-instruction`, `declaration`). Counted, never silently
    /// dropped.
    pub unhandled: BTreeMap<String, usize>,
}

/// XML elements that end a block of text inside CWE/CAPEC prose: a newline
/// is emitted where one opens or closes. Anything else is inline.
const XML_BLOCK_ELEMENTS: &[&str] = &[
    "br", "div", "li", "ol", "p", "table", "td", "th", "tr", "ul",
];

/// The tag's local name: `xhtml:div` -> `div`.
fn xml_local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

/// Read a tag name at `chars[at..]` (letters, digits, `-`, `_`, `:`),
/// returning `(name, next_index)`.
fn xml_tag_name(chars: &[char], at: usize) -> (String, usize) {
    let mut j = at;
    while j < chars.len()
        && (chars[j].is_ascii_alphanumeric() || matches!(chars[j], '-' | '_' | ':'))
    {
        j += 1;
    }
    (chars[at..j].iter().collect(), j)
}

/// Strip XML markup to plain text: elements removed (block elements leaving
/// a line break), comments and processing instructions dropped and counted,
/// CDATA kept verbatim, the five predefined entities plus numeric character
/// references decoded, anything else counted as unhandled. This is the same
/// discipline as [`sgml_to_text`], for the XHTML-flavoured prose MITRE puts
/// inside CWE/CAPEC description elements.
pub fn xml_inner_text(input: &str, stats: &mut XmlStrip) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;
    let mut pending_block = false;
    while i < chars.len() {
        if starts_with_chars(&chars, i, "<!--") {
            let mut j = i + 4;
            while j < chars.len() && !starts_with_chars(&chars, j, "-->") {
                j += 1;
            }
            *stats.unhandled.entry("comment".to_string()).or_insert(0) += 1;
            i = if j < chars.len() { j + 3 } else { chars.len() };
            continue;
        }
        if starts_with_chars(&chars, i, "<![CDATA[") {
            let mut j = i + 9;
            while j < chars.len() && !starts_with_chars(&chars, j, "]]>") {
                j += 1;
            }
            out.extend(chars[i + 9..j].iter());
            i = if j < chars.len() { j + 3 } else { chars.len() };
            continue;
        }
        if starts_with_chars(&chars, i, "<!") {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            *stats
                .unhandled
                .entry("declaration".to_string())
                .or_insert(0) += 1;
            i = if j < chars.len() { j + 1 } else { chars.len() };
            continue;
        }
        if starts_with_chars(&chars, i, "<?") {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            *stats
                .unhandled
                .entry("processing-instruction".to_string())
                .or_insert(0) += 1;
            i = if j < chars.len() { j + 1 } else { chars.len() };
            continue;
        }
        if chars[i] == '<' {
            let name_start = i + 1 + usize::from(chars.get(i + 1) == Some(&'/'));
            let (name, after_name) = xml_tag_name(&chars, name_start);
            let mut j = after_name;
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            stats.elements += 1;
            if XML_BLOCK_ELEMENTS.contains(&xml_local_name(&name)) {
                pending_block = true;
            }
            i = if j < chars.len() { j + 1 } else { chars.len() };
            continue;
        }
        if chars[i] == '&' {
            let mut j = i + 1;
            while j < chars.len() && j - i <= 40 && chars[j] != ';' && !chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j] == ';' {
                let name: String = chars[i + 1..j].iter().collect();
                let decoded = if let Some(number) = name.strip_prefix('#') {
                    let (digits, radix) = match number
                        .strip_prefix('x')
                        .or_else(|| number.strip_prefix('X'))
                    {
                        Some(hex) => (hex, 16),
                        None => (number, 10),
                    };
                    u32::from_str_radix(digits, radix)
                        .ok()
                        .and_then(char::from_u32)
                        .map(|c| c.to_string())
                } else {
                    match name.as_str() {
                        "amp" => Some("&".to_string()),
                        "lt" => Some("<".to_string()),
                        "gt" => Some(">".to_string()),
                        "quot" => Some("\"".to_string()),
                        "apos" => Some("'".to_string()),
                        _ => None,
                    }
                };
                match decoded {
                    Some(text) => {
                        stats.entities_decoded += 1;
                        out.push_str(&text);
                    }
                    None => {
                        *stats
                            .unhandled
                            .entry(format!("entity:&{name};"))
                            .or_insert(0) += 1;
                    }
                }
                i = j + 1;
                continue;
            }
            out.push('&');
            i += 1;
            continue;
        }
        if pending_block && chars[i] == '\n' {
            pending_block = false;
            i += 1;
            continue;
        }
        if pending_block && !chars[i].is_whitespace() {
            out.push('\n');
            pending_block = false;
        }
        out.push(chars[i]);
        i += 1;
    }
    if pending_block {
        out.push('\n');
    }
    collapse_blank_lines(&out)
}

/// The byte width of the first char of `s` (1 for ASCII); `s` is non-empty
/// at every call site, which the callers' loop conditions guarantee.
fn utf8_char_width(s: &str) -> usize {
    s.chars().next().map_or(1, |c| c.len_utf8())
}

/// Whether the `<tag` match at `at` in `xml` is a real element opener: the
/// next char is `>`, `/` or whitespace — `<Weaknesses>` is not a
/// `<Weakness>`.
fn xml_open_boundary(xml: &str, at: usize, open_len: usize) -> bool {
    match xml[at + open_len..].chars().next() {
        Some('>') | Some('/') => true,
        Some(c) => c.is_whitespace(),
        None => false,
    }
}

/// One extracted element of [`split_xml_entries`]: its attribute string and
/// its inner content, both borrowed from the document.
pub struct XmlEntry<'a> {
    /// Everything between the tag name and the opening tag's `>`.
    pub attrs: &'a str,
    /// The content between the opening and closing tags (`""` for a
    /// self-closing element).
    pub inner: &'a str,
}

/// Split a machine-generated XML document into the top-level occurrences of
/// one element (`Weakness`, `Attack_Pattern`): a balanced scan that counts
/// same-tag nesting and respects comments, CDATA and processing
/// instructions, so a literal `<Weakness>` inside example code cannot fake a
/// boundary. Entries are returned in document order.
pub fn split_xml_entries<'a>(xml: &'a str, tag: &str) -> Vec<XmlEntry<'a>> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut entries = Vec::new();
    let mut rest = xml;
    while let Some(pos) = rest.find(&open) {
        // The opener must be exactly `<tag` followed by whitespace, `>` or
        // `/>` — `<Weaknesses>` is not a `<Weakness>`.
        if !xml_open_boundary(rest, pos, open.len()) {
            rest = &rest[pos + open.len()..];
            continue;
        }
        // End of the opening tag.
        let open_end = match rest[pos..].find('>') {
            Some(e) => pos + e,
            None => break,
        };
        let opening = &rest[pos + open.len()..open_end];
        if opening.trim_end().ends_with('/') {
            entries.push(XmlEntry {
                attrs: opening.trim_end_matches('/').trim(),
                inner: "",
            });
            rest = &rest[open_end + 1..];
            continue;
        }
        // Balanced scan for the matching close.
        let mut depth = 1usize;
        let mut scan = open_end + 1;
        while scan < rest.len() && depth > 0 {
            if rest[scan..].starts_with("<!--") {
                scan = rest[scan + 4..]
                    .find("-->")
                    .map_or(rest.len(), |e| scan + 4 + e + 3);
                continue;
            }
            if rest[scan..].starts_with("<![CDATA[") {
                scan = rest[scan + 9..]
                    .find("]]>")
                    .map_or(rest.len(), |e| scan + 9 + e + 3);
                continue;
            }
            if rest[scan..].starts_with(&close) {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                scan += close.len();
                continue;
            }
            if rest[scan..].starts_with(&open) && xml_open_boundary(rest, scan, open.len()) {
                depth += 1;
            }
            // Advance one char (the document is UTF-8; indices stay on
            // boundaries because every match above is ASCII).
            scan += utf8_char_width(&rest[scan..]);
        }
        if depth != 0 {
            break;
        }
        entries.push(XmlEntry {
            attrs: opening.trim(),
            inner: &rest[open_end + 1..scan],
        });
        rest = &rest[scan + close.len()..];
    }
    entries
}

/// An attribute value out of an element's attribute string (`ID` in
/// `<Weakness ID="79" …>`).
fn xml_attr<'a>(attrs: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("{name}=\"");
    let start = attrs.find(&needle)? + needle.len();
    let end = attrs[start..].find('"')? + start;
    Some(&attrs[start..end])
}

/// The first direct occurrence of `<name>…</name>` (exact tag name;
/// `<Consequences>` is not `<Consequence>`). The inner content is returned
/// with markup intact — callers run [`xml_inner_text`] over it.
fn xml_element<'a>(xml: &'a str, name: &str) -> Option<&'a str> {
    xml_elements(xml, name).into_iter().next()
}

/// All direct occurrences of `<name>…</name>` in document order, as
/// [`XmlEntry`]s so attribute-only elements (`<Related_Weakness CWE_ID="89"
/// />`, self-closing with an empty inner) keep their attributes. Nested
/// same-name elements return the outermost only.
fn xml_element_entries<'a>(xml: &'a str, name: &str) -> Vec<XmlEntry<'a>> {
    let open = format!("<{name}");
    let close = format!("</{name}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(pos) = rest.find(&open) {
        if !xml_open_boundary(rest, pos, open.len()) {
            rest = &rest[pos + open.len()..];
            continue;
        }
        let open_end = match rest[pos..].find('>') {
            Some(e) => pos + e,
            None => break,
        };
        let opening = &rest[pos + open.len()..open_end];
        if opening.trim_end().ends_with('/') {
            out.push(XmlEntry {
                attrs: opening.trim_end_matches('/').trim(),
                inner: "",
            });
            rest = &rest[open_end + 1..];
            continue;
        }
        let mut depth = 1usize;
        let mut scan = open_end + 1;
        while scan < rest.len() && depth > 0 {
            if rest[scan..].starts_with(&close) {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                scan += close.len();
                continue;
            }
            if rest[scan..].starts_with(&open) && xml_open_boundary(rest, scan, open.len()) {
                depth += 1;
            }
            scan += utf8_char_width(&rest[scan..]);
        }
        if depth != 0 {
            break;
        }
        out.push(XmlEntry {
            attrs: opening.trim(),
            inner: &rest[open_end + 1..scan],
        });
        rest = &rest[scan + close.len()..];
    }
    out
}

/// All direct occurrences of `<name>…</name>` in document order, inner
/// content only. Nested same-name elements return the outermost only.
fn xml_elements<'a>(xml: &'a str, name: &str) -> Vec<&'a str> {
    xml_element_entries(xml, name)
        .into_iter()
        .map(|entry| entry.inner)
        .collect()
}

/// One element's text, stripped; absent or whitespace-only -> `None`.
fn xml_field(xml: &str, name: &str, stats: &mut XmlStrip) -> Option<String> {
    let inner = xml_element(xml, name)?;
    let text = xml_inner_text(inner, stats);
    (!text.trim().is_empty()).then_some(text)
}

/// All occurrences of an element's text, stripped, dropping empties.
fn xml_fields(xml: &str, name: &str, stats: &mut XmlStrip) -> Vec<String> {
    xml_elements(xml, name)
        .into_iter()
        .map(|inner| xml_inner_text(inner, stats))
        .filter(|text| !text.trim().is_empty())
        .collect()
}

/// Render a `<Common_Consequences>`-style container: each child element's
/// `Scope` values joined with its `Impact` values.
fn xml_scope_impacts(container: Option<&str>, stats: &mut XmlStrip) -> Vec<String> {
    let mut out = Vec::new();
    let Some(container) = container else {
        return out;
    };
    for child in xml_elements(container, "Consequence") {
        let scopes = xml_fields(child, "Scope", stats);
        let impacts = xml_fields(child, "Impact", stats);
        let note = xml_field(child, "Note", stats);
        let mut line = String::new();
        if !scopes.is_empty() {
            line.push_str(&format!("Scope: {}", scopes.join(", ")));
        }
        if !impacts.is_empty() {
            if !line.is_empty() {
                line.push_str("; ");
            }
            line.push_str(&format!("Impact: {}", impacts.join(", ")));
        }
        if let Some(note) = note {
            line.push_str(&format!(" — {note}"));
        }
        if !line.is_empty() {
            out.push(line);
        }
    }
    out
}

/// A CWE `<Weakness>` entry -> a document: the id and name, description,
/// extended description, modes of introduction, consequences, detection
/// methods, mitigations and observed examples. An entry without a
/// description is skipped (counted) — there is nothing to learn from a name.
fn render_cwe_weakness(entry: &XmlEntry<'_>, stats: &mut XmlStrip) -> RowOutcome {
    let Some(id) = xml_attr(entry.attrs, "ID") else {
        return RowOutcome::Skip("cwe-entry-without-id");
    };
    let name = xml_attr(entry.attrs, "Name").unwrap_or("unnamed weakness");
    let Some(description) = xml_field(entry.inner, "Description", stats) else {
        return RowOutcome::Skip("cwe-entry-without-description");
    };
    let mut text = format!("CWE-{id}: {name}\n\n{description}");
    if let Some(extended) = xml_field(entry.inner, "Extended_Description", stats) {
        text.push_str(&format!("\n\n{extended}"));
    }
    let intro_phases = xml_element(entry.inner, "Modes_Of_Introduction")
        .map(|m| xml_fields(m, "Phase", stats))
        .unwrap_or_default();
    if !intro_phases.is_empty() {
        text.push_str(&format!(
            "\n\nModes of introduction: {}",
            intro_phases.join(", ")
        ));
    }
    let consequences = xml_scope_impacts(xml_element(entry.inner, "Common_Consequences"), stats);
    if !consequences.is_empty() {
        text.push_str("\n\nConsequences:");
        for line in &consequences {
            text.push_str(&format!("\n- {line}"));
        }
    }
    let detection = xml_element(entry.inner, "Detection_Methods")
        .map(|d| {
            xml_elements(d, "Detection_Method")
                .into_iter()
                .filter_map(|m| {
                    let method = xml_field(m, "Method", stats);
                    let desc = xml_field(m, "Description", stats);
                    match (method, desc) {
                        (Some(method), Some(desc)) => Some(format!("{method}: {desc}")),
                        (None, Some(desc)) => Some(desc),
                        (Some(method), None) => Some(method),
                        (None, None) => None,
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !detection.is_empty() {
        text.push_str("\n\nDetection methods:");
        for line in &detection {
            text.push_str(&format!("\n- {line}"));
        }
    }
    let mitigations = xml_element(entry.inner, "Potential_Mitigations")
        .map(|container| {
            xml_elements(container, "Mitigation")
                .into_iter()
                .filter_map(|m| {
                    let phases = xml_fields(m, "Phase", stats);
                    let desc = xml_field(m, "Description", stats)?;
                    Some(if phases.is_empty() {
                        desc
                    } else {
                        format!("{}: {desc}", phases.join(", "))
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !mitigations.is_empty() {
        text.push_str("\n\nMitigations:");
        for line in &mitigations {
            text.push_str(&format!("\n- {line}"));
        }
    }
    let examples = xml_element(entry.inner, "Observed_Examples")
        .map(|container| {
            xml_elements(container, "Observed_Example")
                .into_iter()
                .filter_map(|e| {
                    let reference = xml_field(e, "Reference", stats);
                    let desc = xml_field(e, "Description", stats);
                    match (reference, desc) {
                        (Some(reference), Some(desc)) => Some(format!("{reference}: {desc}")),
                        (None, Some(desc)) => Some(desc),
                        _ => None,
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !examples.is_empty() {
        text.push_str("\n\nObserved examples:");
        for line in &examples {
            text.push_str(&format!("\n- {line}"));
        }
    }
    RowOutcome::Doc(text)
}

/// A CAPEC `<Attack_Pattern>` entry -> a document: id and name, description,
/// likelihood/severity, prerequisites, skills required, consequences,
/// mitigations and example instances.
fn render_capec_pattern(entry: &XmlEntry<'_>, stats: &mut XmlStrip) -> RowOutcome {
    let Some(id) = xml_attr(entry.attrs, "ID") else {
        return RowOutcome::Skip("capec-entry-without-id");
    };
    let name = xml_attr(entry.attrs, "Name").unwrap_or("unnamed pattern");
    let Some(description) = xml_field(entry.inner, "Description", stats) else {
        return RowOutcome::Skip("capec-entry-without-description");
    };
    let mut text = format!("CAPEC-{id}: {name}\n\n{description}");
    let likelihood = xml_field(entry.inner, "Likelihood_Of_Attack", stats);
    let severity = xml_field(entry.inner, "Typical_Severity", stats);
    let mut line = String::new();
    if let Some(likelihood) = likelihood {
        line.push_str(&format!("Likelihood of attack: {likelihood}"));
    }
    if let Some(severity) = severity {
        if !line.is_empty() {
            line.push_str(". ");
        }
        line.push_str(&format!("Typical severity: {severity}"));
    }
    if !line.is_empty() {
        text.push_str(&format!("\n\n{line}"));
    }
    for (container_name, child_name, heading) in [
        ("Prerequisites", "Prerequisite", "Prerequisites"),
        ("Skills_Required", "Skill", "Skills required"),
        ("Mitigations", "Mitigation", "Mitigations"),
        ("Example_Instances", "Example", "Example instances"),
        ("Indicators", "Indicator", "Indicators"),
    ] {
        let items = xml_element(entry.inner, container_name)
            .map(|c| xml_fields(c, child_name, stats))
            .unwrap_or_default();
        if !items.is_empty() {
            text.push_str(&format!("\n\n{heading}:"));
            for item in &items {
                text.push_str(&format!("\n- {item}"));
            }
        }
    }
    let consequences = xml_scope_impacts(xml_element(entry.inner, "Consequences"), stats);
    if !consequences.is_empty() {
        text.push_str("\n\nConsequences:");
        for line in &consequences {
            text.push_str(&format!("\n- {line}"));
        }
    }
    let related = xml_element(entry.inner, "Related_Weaknesses")
        .map(|c| {
            xml_element_entries(c, "Related_Weakness")
                .into_iter()
                .filter_map(|w| xml_attr(w.attrs, "CWE_ID").map(|id| format!("CWE-{id}")))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !related.is_empty() {
        text.push_str(&format!("\n\nRelated weaknesses: {}", related.join(", ")));
    }
    RowOutcome::Doc(text)
}

/// All files under `dir`, recursively, sorted for a deterministic walk.
fn walk_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&current)
            .with_context(|| format!("read directory {}", current.display()))?
            .map(|entry| {
                entry
                    .map(|e| e.path())
                    .with_context(|| format!("read an entry of {}", current.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        entries.sort();
        for path in entries {
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Resolve one input against the datasets root (absolute inputs stay as-is).
fn resolve(datasets_root: &Path, input: &Path) -> PathBuf {
    if input.is_absolute() {
        input.to_path_buf()
    } else {
        datasets_root.join(input)
    }
}

/// The relative origin string for a row-based document.
fn row_origin(display_path: &str, row: usize) -> String {
    format!("{display_path}#{row}")
}

fn render_row(spec: &SourceSpec, value: &Value) -> RowOutcome {
    match spec.format {
        SourceFormat::ChatJsonl => render_chat(value, spec.escaped_jsonl),
        SourceFormat::JsonArrayQA => render_metamath(value),
        SourceFormat::NuminaJsonl => render_numina(value),
        SourceFormat::Ai2ArcJsonl => render_ai2_arc(value),
        SourceFormat::MedmcqaJsonl => render_medmcqa(value),
        SourceFormat::TextToSqlJsonl => render_text_to_sql(value),
        SourceFormat::IdentityJsonl => render_identity(value),
        SourceFormat::NvdCveJsonl => render_nvd_cve(value, false),
        SourceFormat::NvdCveExploitJsonl => render_nvd_cve(value, true),
        SourceFormat::AttackStixJsonl => render_stix(value, false),
        SourceFormat::AttackStixReconJsonl => render_stix(value, true),
        // Row renderers are only called for row formats; tree and XML-file
        // formats go through their own collectors. One of those here is a
        // spec bug, and it is counted rather than `unreachable!` (the crate
        // denies it).
        SourceFormat::MarkdownTree
        | SourceFormat::SgmlTree
        | SourceFormat::CweXmlFile
        | SourceFormat::CapecXmlFile => RowOutcome::Skip("internal-error:tree-format-as-row"),
    }
}

/// Collect the rendered documents of one JSONL/JSON-array source.
fn collect_row_documents(
    spec: &SourceSpec,
    inputs: &[PathBuf],
    stats: &mut SourceStats,
) -> Result<Vec<Doc>> {
    let mut docs = Vec::new();
    for input in inputs {
        match spec.format {
            SourceFormat::JsonArrayQA => {
                let file = std::fs::File::open(input)
                    .with_context(|| format!("open JSON array {}", input.display()))?;
                let rows: Vec<Value> = serde_json::from_reader(BufReader::new(file))
                    .with_context(|| format!("{} is not a single JSON array", input.display()))?;
                let display = input.display().to_string();
                for (n, row) in rows.iter().enumerate() {
                    push_rendered(spec, &row_origin(&display, n + 1), row, stats, &mut docs);
                }
            }
            _ => {
                let file = std::fs::File::open(input)
                    .with_context(|| format!("open JSONL {}", input.display()))?;
                for (n, line) in BufReader::new(file).lines().enumerate() {
                    let line =
                        line.with_context(|| format!("read {} line {}", input.display(), n + 1))?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(&line).with_context(|| {
                        format!("{} line {} is not valid JSON", input.display(), n + 1)
                    })?;
                    let display = input.display().to_string();
                    push_rendered(spec, &row_origin(&display, n + 1), &value, stats, &mut docs);
                }
            }
        }
    }
    Ok(docs)
}

fn push_rendered(
    spec: &SourceSpec,
    origin: &str,
    value: &Value,
    stats: &mut SourceStats,
    docs: &mut Vec<Doc>,
) {
    match render_row(spec, value) {
        RowOutcome::Doc(text) => docs.push(Doc {
            origin: origin.to_string(),
            text,
        }),
        RowOutcome::DocCounted(text, counter) => {
            *stats.counters.entry(counter.to_string()).or_insert(0) += 1;
            docs.push(Doc {
                origin: origin.to_string(),
                text,
            });
        }
        RowOutcome::Skip(reason) => {
            *stats.rows_skipped.entry(reason.to_string()).or_insert(0) += 1;
        }
    }
}

/// Collect the rendered documents of one markdown/SGML tree source.
fn collect_tree_documents(
    spec: &SourceSpec,
    input_dir: &Path,
    stats: &mut SourceStats,
) -> Result<Vec<Doc>> {
    let extension = match spec.format {
        SourceFormat::MarkdownTree => "md",
        SourceFormat::SgmlTree => "sgml",
        other => {
            return Err(anyhow::anyhow!(
                "collect_tree_documents called with a non-tree format: {other:?}"
            ));
        }
    };
    let mut docs = Vec::new();
    for file in walk_files(input_dir)? {
        let is_doc = file
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case(extension));
        let rel = file
            .strip_prefix(input_dir)
            .with_context(|| format!("{} is not under {}", file.display(), input_dir.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        if !is_doc {
            *stats
                .rows_skipped
                .entry(format!("non-{extension}-file"))
                .or_insert(0) += 1;
            continue;
        }
        let raw =
            std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
        let origin = format!("{}:{rel}", input_dir.display());
        let text = match spec.format {
            SourceFormat::MarkdownTree => {
                let mut strip = MarkdownStrip::default();
                let text = hugo_markdown_to_text(&raw, &mut strip);
                if strip.front_matter {
                    *stats
                        .counters
                        .entry("front-matter".to_string())
                        .or_insert(0) += 1;
                }
                if strip.shortcodes > 0 {
                    *stats.counters.entry("shortcodes".to_string()).or_insert(0) +=
                        strip.shortcodes;
                }
                if strip.html_tags > 0 {
                    *stats.counters.entry("html-tags".to_string()).or_insert(0) += strip.html_tags;
                }
                if strip.html_comments > 0 {
                    *stats
                        .counters
                        .entry("html-comments".to_string())
                        .or_insert(0) += strip.html_comments;
                }
                text
            }
            SourceFormat::SgmlTree => {
                let mut strip = SgmlStrip::default();
                let text = sgml_to_text(&raw, &mut strip);
                *stats
                    .counters
                    .entry("sgml-elements".to_string())
                    .or_insert(0) += strip.elements;
                *stats
                    .counters
                    .entry("sgml-comments".to_string())
                    .or_insert(0) += strip.comments;
                *stats
                    .counters
                    .entry("sgml-entities-decoded".to_string())
                    .or_insert(0) += strip.entities_decoded;
                *stats
                    .counters
                    .entry("sgml-cdata-sections".to_string())
                    .or_insert(0) += strip.cdata_sections;
                for (construct, count) in strip.unhandled {
                    *stats.unhandled_constructs.entry(construct).or_insert(0) += count;
                }
                text
            }
            other => {
                return Err(anyhow::anyhow!(
                    "collect_tree_documents called with a non-tree format: {other:?}"
                ));
            }
        };
        docs.push(Doc { origin, text });
    }
    Ok(docs)
}

/// Collect the rendered documents of one CWE/CAPEC content-XML source: split
/// the file into entries with [`split_xml_entries`], render each, and
/// aggregate the strip counters. The origin of every document names the
/// entry (`#CWE-79`, `#CAPEC-66`), so provenance points at the weakness or
/// attack pattern, not a byte offset.
fn collect_xml_documents(
    spec: &SourceSpec,
    input_file: &Path,
    stats: &mut SourceStats,
) -> Result<Vec<Doc>> {
    let (tag, render): (&str, fn(&XmlEntry<'_>, &mut XmlStrip) -> RowOutcome) = match spec.format {
        SourceFormat::CweXmlFile => ("Weakness", render_cwe_weakness),
        SourceFormat::CapecXmlFile => ("Attack_Pattern", render_capec_pattern),
        other => {
            return Err(anyhow::anyhow!(
                "collect_xml_documents called with a non-XML format: {other:?}"
            ));
        }
    };
    let raw = std::fs::read_to_string(input_file)
        .with_context(|| format!("read {}", input_file.display()))?;
    let entries = split_xml_entries(&raw, tag);
    if entries.is_empty() {
        return Err(anyhow::anyhow!(
            "{} contains no <{tag}> entries; refusing to build an empty source from it",
            input_file.display()
        ));
    }
    let display = input_file.display().to_string();
    let mut strip = XmlStrip::default();
    let mut docs = Vec::with_capacity(entries.len());
    for entry in &entries {
        let id = xml_attr(entry.attrs, "ID").unwrap_or("unknown");
        let origin = format!("{display}#{tag}-{id}");
        match render(entry, &mut strip) {
            RowOutcome::Doc(text) | RowOutcome::DocCounted(text, _) => {
                docs.push(Doc { origin, text });
            }
            RowOutcome::Skip(reason) => {
                *stats.rows_skipped.entry(reason.to_string()).or_insert(0) += 1;
            }
        }
    }
    *stats
        .counters
        .entry("xml-elements-stripped".to_string())
        .or_insert(0) += strip.elements;
    *stats
        .counters
        .entry("xml-entities-decoded".to_string())
        .or_insert(0) += strip.entities_decoded;
    for (construct, count) in strip.unhandled {
        *stats.unhandled_constructs.entry(construct).or_insert(0) += count;
    }
    Ok(docs)
}

/// The per-source completion marker path.
fn marker_path(dataset_dir: &Path, source: &str) -> PathBuf {
    dataset_dir
        .join(".fieldnorm")
        .join(format!("{source}.json"))
}

/// Filter the rendered documents of one source through the cheat stack and
/// MinHash near-dup, then write `dataset/<source>.jsonl` and
/// `text/<source>/<id>.txt`. Returns the final stats. Every drop is counted:
/// per cheat class, as `near_duplicates`, or as `oversize`.
fn filter_and_write(
    spec: &SourceSpec,
    docs: Vec<Doc>,
    config: &FieldnormConfig,
    dataset_dir: &Path,
    text_dir: &Path,
    stats: &mut SourceStats,
) -> Result<()> {
    // Size cap and empty-text filter.
    let mut survivors: Vec<Doc> = Vec::with_capacity(docs.len());
    for doc in docs {
        if doc.text.len() > config.max_doc_bytes {
            stats.oversize += 1;
            continue;
        }
        if doc.text.trim().is_empty() {
            *stats
                .rows_skipped
                .entry("empty-text".to_string())
                .or_insert(0) += 1;
            continue;
        }
        survivors.push(doc);
    }

    // The cheat stack: suppression, stubs, gaming, brokenness, secrets,
    // encoding/dynamic/behavioral evasion — every finding is counted.
    // Reject grade (Suppression and above: secrets, suppression, gaming,
    // encoding/dynamic evasion) drops the document, counted per class;
    // quarantine grade (Evasion) is counted without dropping — those classes
    // are calibrated for code snippets and fire on ordinary prose (module
    // docs carry the measurement).
    let mut kept: Vec<Doc> = Vec::with_capacity(survivors.len());
    for doc in survivors {
        let report = cheat::scan_snippet(&doc.text, SnippetLanguage::Other);
        let mut reject: Vec<String> = Vec::new();
        let mut quarantine: Vec<String> = Vec::new();
        for finding in &report.findings {
            if finding.severity >= Severity::Suppression {
                reject.push(finding.class.as_str().to_string());
            } else if finding.severity >= Severity::Evasion {
                quarantine.push(finding.class.as_str().to_string());
            }
        }
        quarantine.sort();
        quarantine.dedup();
        for class in quarantine {
            *stats.quarantine_findings.entry(class).or_insert(0) += 1;
        }
        reject.sort();
        reject.dedup();
        if reject.is_empty() {
            kept.push(doc);
        } else {
            for class in reject {
                *stats.dropped_by_class.entry(class).or_insert(0) += 1;
            }
        }
    }

    // MinHash near-duplicates within the source: keep the first member of
    // every cluster, drop the rest. Deterministic (fixed FNV-1a permutations)
    // per cheat::minhash_near_duplicate_groups' contract.
    let texts: Vec<String> = kept.iter().map(|d| d.text.clone()).collect();
    let groups = cheat::minhash_near_duplicate_groups(
        &texts,
        config.near_dup_hashes,
        config.near_dup_threshold,
    );
    let mut dup: HashSet<usize> = HashSet::new();
    for group in &groups {
        for &member in group.iter().skip(1) {
            dup.insert(member);
        }
        stats.near_duplicates += group.len().saturating_sub(1);
    }

    // A forced rebuild over a previous run's tree must not leave stale text
    // files behind: clear the source's text directory first.
    let source_text_dir = text_dir.join(&spec.name);
    if source_text_dir.is_dir() {
        std::fs::remove_dir_all(&source_text_dir)
            .with_context(|| format!("clear {}", source_text_dir.display()))?;
    }
    std::fs::create_dir_all(&source_text_dir)
        .with_context(|| format!("create {}", source_text_dir.display()))?;

    let out_path = dataset_dir.join(format!("{}.jsonl", spec.name));
    let tmp_path = dataset_dir.join(format!(".{}.jsonl.tmp", spec.name));
    let mut writer = BufWriter::new(
        std::fs::File::create(&tmp_path)
            .with_context(|| format!("create {}", tmp_path.display()))?,
    );
    let mut seq = 0usize;
    for (index, doc) in kept.iter().enumerate() {
        if dup.contains(&index) {
            continue;
        }
        seq += 1;
        let id = format!("{}-{seq:06}", spec.name);
        let record = DatasetRecord {
            id: id.clone(),
            source: spec.name.clone(),
            path: doc.origin.clone(),
            format: spec.format.record_format().to_string(),
            text: doc.text.clone(),
        };
        let line = serde_json::to_string(&record).context("serialize dataset record")?;
        writer
            .write_all(line.as_bytes())
            .and_then(|()| writer.write_all(b"\n"))
            .with_context(|| format!("write {}", tmp_path.display()))?;
        crate::nim::write_atomic(
            &source_text_dir.join(format!("{id}.txt")),
            doc.text.as_bytes(),
        )?;
        stats.documents += 1;
        stats.chars += doc.text.chars().count();
    }
    writer
        .flush()
        .with_context(|| format!("flush {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &out_path)
        .with_context(|| format!("rename {} to {}", tmp_path.display(), out_path.display()))?;
    Ok(())
}

/// Build one field corpus in the cybercorpus layout at
/// `corpus_dir` (conventionally `<out-root>/<field>`).
///
/// Per source: render rows to documents, size-cap, cheat-scan, near-dedup,
/// write `dataset/<source>.jsonl` + `text/<source>/<id>.txt`, then leave a
/// completion marker under `dataset/.fieldnorm/`. A rerun with `force =
/// false` skips sources whose marker exists (their stats are re-read from the
/// marker) and rebuilds `dataset/all.jsonl` and `dataset/manifest.json` from
/// the per-source files. Output order is deterministic: sorted sources,
/// sorted file walks, row order within files.
///
/// The identity file (`config.identity_path`), when it exists, is appended as
/// source `identity` to the corpus — the whole file, once (PLAN-NINA §3).
pub fn build_field_corpus(
    spec: &FieldSpec,
    datasets_root: &Path,
    corpus_dir: &Path,
    config: &FieldnormConfig,
    force: bool,
) -> Result<FieldManifest> {
    config.validate()?;
    anyhow::ensure!(
        datasets_root.is_dir(),
        "datasets root {} does not exist",
        datasets_root.display()
    );
    let dataset_dir = corpus_dir.join("dataset");
    let text_dir = corpus_dir.join("text");
    let raw_dir = corpus_dir.join("raw");
    std::fs::create_dir_all(dataset_dir.join(".fieldnorm"))
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    std::fs::create_dir_all(&text_dir).with_context(|| format!("create {}", text_dir.display()))?;
    std::fs::create_dir_all(&raw_dir).with_context(|| format!("create {}", raw_dir.display()))?;

    // Parquet-backed sources read the converted JSONL; its absence is a loud
    // error naming the tool, never a silent empty source.
    for spec_source in &spec.sources {
        if let Some(converted) = &spec_source.converted {
            let converted_path = raw_dir.join(converted);
            anyhow::ensure!(
                converted_path.is_file(),
                "source {} reads {}, which does not exist; convert the parquet inputs ({}) first, e.g.: \
                 /srv/m-sda/unifur/tools/venv/bin/python /srv/m-sda/unifur/tools/parquet2jsonl.py --out {} <in.parquet>...",
                spec_source.name,
                converted_path.display(),
                spec_source
                    .inputs
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                converted_path.display()
            );
        }
    }

    let mut manifest = FieldManifest {
        field: spec.field.clone(),
        datasets_root: datasets_root.to_path_buf(),
        corpus_dir: corpus_dir.to_path_buf(),
        config: config.clone(),
        sources: BTreeMap::new(),
        total_documents: 0,
        total_chars: 0,
        dropped_by_class: BTreeMap::new(),
        quarantine_findings: BTreeMap::new(),
        near_duplicates: 0,
        oversize: 0,
        identity: None,
        parquet_conversions: Vec::new(),
        resumed_sources: Vec::new(),
        notes: vec![
            "drop policy: cheat findings at reject grade (Suppression and above) drop the document; \
             quarantine grade (Evasion) is counted per class under quarantine_findings without dropping \
             — the quarantine classes are code-snippet checks and fire on ordinary prose (module docs \
             carry the measurement)"
                .to_string(),
        ],
    };

    // The conversion provenance log, verbatim.
    let conversions_log = raw_dir.join("parquet2jsonl-conversions.jsonl");
    if conversions_log.is_file() {
        let text = std::fs::read_to_string(&conversions_log)
            .with_context(|| format!("read {}", conversions_log.display()))?;
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: Value = serde_json::from_str(line).with_context(|| {
                format!(
                    "{} line {} is not valid JSON",
                    conversions_log.display(),
                    n + 1
                )
            })?;
            manifest.parquet_conversions.push(record);
        }
    }

    let mut specs: Vec<SourceSpec> = spec.sources.clone();
    if config.identity_path.is_file() {
        specs.push(SourceSpec {
            name: IDENTITY_SOURCE.to_string(),
            format: SourceFormat::IdentityJsonl,
            inputs: vec![config.identity_path.clone()],
            converted: None,
            noted_skips: Vec::new(),
            escaped_jsonl: false,
        });
    } else {
        manifest.notes.push(format!(
            "identity file {} not found; no identity mix-in",
            config.identity_path.display()
        ));
    }
    specs.sort_by(|a, b| a.name.cmp(&b.name));

    for source in specs {
        let marker = marker_path(&dataset_dir, &source.name);
        if !force && marker.is_file() {
            let text = std::fs::read_to_string(&marker)
                .with_context(|| format!("read {}", marker.display()))?;
            let stats: SourceStats = serde_json::from_str(&text)
                .with_context(|| format!("{} is not a source marker", marker.display()))?;
            manifest.resumed_sources.push(source.name.clone());
            manifest.sources.insert(source.name.clone(), stats);
            continue;
        }
        let mut stats = SourceStats::default();
        for skip in &source.noted_skips {
            stats
                .notes
                .push(format!("deliberately excluded input: {skip}"));
        }
        let docs = match source.format {
            SourceFormat::MarkdownTree | SourceFormat::SgmlTree => {
                let input = source.inputs.first().ok_or_else(|| {
                    anyhow::anyhow!("source {} has no input directory", source.name)
                })?;
                let input = resolve(datasets_root, input);
                anyhow::ensure!(
                    input.is_dir(),
                    "source {} input directory {} does not exist",
                    source.name,
                    input.display()
                );
                collect_tree_documents(&source, &input, &mut stats)?
            }
            SourceFormat::CweXmlFile | SourceFormat::CapecXmlFile => {
                let input = source
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("source {} has no input file", source.name))?;
                let input = resolve(datasets_root, input);
                anyhow::ensure!(
                    input.is_file(),
                    "source {} input file {} does not exist",
                    source.name,
                    input.display()
                );
                collect_xml_documents(&source, &input, &mut stats)?
            }
            _ => {
                let inputs: Vec<PathBuf> = match &source.converted {
                    Some(converted) => vec![raw_dir.join(converted)],
                    None => source
                        .inputs
                        .iter()
                        .map(|i| resolve(datasets_root, i))
                        .collect(),
                };
                for input in &inputs {
                    anyhow::ensure!(
                        input.is_file(),
                        "source {} input {} does not exist",
                        source.name,
                        input.display()
                    );
                }
                collect_row_documents(&source, &inputs, &mut stats)?
            }
        };
        eprintln!(
            "[fieldnorm] {}: {} rendered documents, filtering",
            source.name,
            docs.len()
        );
        filter_and_write(&source, docs, config, &dataset_dir, &text_dir, &mut stats)?;
        if source.name == IDENTITY_SOURCE {
            manifest.identity = Some(IdentityNote {
                path: config.identity_path.clone(),
                documents: stats.documents,
            });
        }
        let marker_json =
            serde_json::to_string_pretty(&stats).context("serialize source completion marker")?;
        crate::nim::write_atomic(&marker, marker_json.as_bytes())?;
        eprintln!(
            "[fieldnorm] {}: kept {}, dropped {} ({} near-dup, {} oversize)",
            source.name,
            stats.documents,
            stats.dropped_by_class.values().sum::<usize>(),
            stats.near_duplicates,
            stats.oversize
        );
        manifest.sources.insert(source.name.clone(), stats);
    }

    // Rebuild all.jsonl from the per-source files, sorted by source: the
    // aggregate is derived state, so a resumed run regenerates it byte-for-byte.
    let all_tmp = dataset_dir.join(".all.jsonl.tmp");
    {
        let mut writer = BufWriter::new(
            std::fs::File::create(&all_tmp)
                .with_context(|| format!("create {}", all_tmp.display()))?,
        );
        for source in manifest.sources.keys() {
            let per_source = dataset_dir.join(format!("{source}.jsonl"));
            if !per_source.is_file() {
                return Err(anyhow::anyhow!(
                    "source {} has stats but no {}; the corpus is half-built — rerun with --force",
                    source,
                    per_source.display()
                ));
            }
            let mut reader = BufReader::new(
                std::fs::File::open(&per_source)
                    .with_context(|| format!("open {}", per_source.display()))?,
            );
            std::io::copy(&mut reader, &mut writer)
                .with_context(|| format!("append {}", per_source.display()))?;
        }
        writer
            .flush()
            .with_context(|| format!("flush {}", all_tmp.display()))?;
    }
    std::fs::rename(&all_tmp, dataset_dir.join("all.jsonl"))
        .with_context(|| format!("rename {}", all_tmp.display()))?;

    // Totals.
    for stats in manifest.sources.values() {
        manifest.total_documents += stats.documents;
        manifest.total_chars += stats.chars;
        manifest.near_duplicates += stats.near_duplicates;
        manifest.oversize += stats.oversize;
        for (class, count) in &stats.dropped_by_class {
            *manifest.dropped_by_class.entry(class.clone()).or_insert(0) += count;
        }
        for (class, count) in &stats.quarantine_findings {
            *manifest
                .quarantine_findings
                .entry(class.clone())
                .or_insert(0) += count;
        }
    }
    let manifest_json =
        serde_json::to_string_pretty(&manifest).context("serialize field manifest")?;
    crate::nim::write_atomic(&dataset_dir.join("manifest.json"), manifest_json.as_bytes())?;
    eprintln!(
        "[fieldnorm] {}: {} documents ({} chars) from {} sources, {} cheat-dropped, {} near-dup, {} oversize -> {}",
        manifest.field,
        manifest.total_documents,
        manifest.total_chars,
        manifest.sources.len(),
        manifest.dropped_by_class.values().sum::<usize>(),
        manifest.near_duplicates,
        manifest.oversize,
        dataset_dir.display()
    );
    Ok(manifest)
}

// ---------------------------------------------------------------------------
// Field corpus -> trainer pairs (`dblocks corpus to-pairs`)
// ---------------------------------------------------------------------------

/// Default cap on the pairs one corpus yields. The trainer
/// ([`crate::student::LanguageBatch::load`]) reads the whole pairs file into
/// memory, so this cap — not the corpus size — is what bounds that load.
pub const DEFAULT_PAIRS_MAX_DOCS: usize = 20_000;

/// Documents shorter than this many *characters* never become a pair
/// (counted `skipped_short`): a quarter of a tiny document is a prompt with
/// no completion worth supervising.
pub const MIN_PAIR_DOC_CHARS: usize = 64;

fn default_pairs_max_docs() -> usize {
    DEFAULT_PAIRS_MAX_DOCS
}

/// Configuration for [`to_pairs`]. Serde-defaulted so a sidecar written by an
/// older build still parses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToPairsConfig {
    /// Cap on emitted pairs. The subsample is the `max_docs` smallest
    /// sha256(id) digests, not the first `max_docs` lines: with first-N a
    /// rerun over a corpus that gained sources would silently swap in the new
    /// front of the file, while hash order only lets ids that hash lower
    /// displace an earlier selection.
    #[serde(default = "default_pairs_max_docs")]
    pub max_docs: usize,
}

impl Default for ToPairsConfig {
    fn default() -> Self {
        Self {
            max_docs: default_pairs_max_docs(),
        }
    }
}

impl ToPairsConfig {
    /// Validate the tunables, naming the field at fault.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.max_docs > 0, "ToPairsConfig.max_docs must be positive");
        Ok(())
    }
}

/// The split rule of [`to_pairs`]: where one document becomes a
/// (prompt, completion) pair. Deterministic and char-based — the arithmetic
/// is over `char`s, so a multi-byte character is never divided.
///
/// - the prompt ends at the document's first *blank line* (a newline,
///   optional spaces/tabs/carriage returns, another newline) — the rendered
///   field documents put question and answer in separate paragraphs, so the
///   paragraph break is the natural boundary;
/// - with no usable blank line, the prompt is the first quarter of the
///   document's chars, cut at the last whitespace at or before the quarter
///   point (never mid-word); a document with no whitespace that early cuts
///   at the first whitespace after the quarter, and one with no whitespace
///   at all cuts at the quarter point itself;
/// - whichever boundary comes first wins, and the prompt is at least one
///   char: a blank line at the very start of the document does not count,
///   and a slice that would trim to nothing keeps its raw characters.
///
/// The completion is the remainder, leading whitespace (the blank line or
/// the cut point itself) removed.
pub fn split_prompt_completion(text: &str) -> (String, String) {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n < 2 {
        return (text.to_string(), String::new());
    }

    // First blank line: '\n', horizontal whitespace, '\n'. The prompt is
    // everything before the newline that opens the blank line. Index 0 is
    // not scanned: a blank line at the very start would make an empty
    // prompt, so it is not a boundary.
    let mut blank_at = None;
    for i in 1..n {
        if chars[i] != '\n' {
            continue;
        }
        let mut j = i + 1;
        while j < n && (chars[j] == ' ' || chars[j] == '\t' || chars[j] == '\r') {
            j += 1;
        }
        if j < n && chars[j] == '\n' {
            blank_at = Some(i);
            break;
        }
    }

    // The quarter cut: the last whitespace at or before `quarter` chars,
    // else the first whitespace after it, else the quarter point itself.
    // Cut index `i` means prompt = chars[..i], completion = chars[i..].
    let quarter = (n / 4).max(1).min(n - 1);
    let quarter_cut = (1..=quarter)
        .rev()
        .find(|&i| chars[i].is_whitespace())
        .or_else(|| (quarter + 1..n).find(|&i| chars[i].is_whitespace()))
        .unwrap_or(quarter);

    let split = blank_at.map_or(quarter_cut, |b| b.min(quarter_cut));

    let raw_prompt: String = chars[..split].iter().collect();
    let trimmed_prompt = raw_prompt.trim();
    // A whitespace-only slice would trim to nothing; the prompt keeps at
    // least its raw characters (the minimum-1-char rule).
    let prompt = if trimmed_prompt.is_empty() {
        raw_prompt
    } else {
        trimmed_prompt.to_string()
    };
    let raw_completion: String = chars[split..].iter().collect();
    let trimmed_completion = raw_completion.trim_start();
    let completion = if trimmed_completion.is_empty() {
        raw_completion
    } else {
        trimmed_completion.to_string()
    };
    (prompt, completion)
}

/// One output line of [`to_pairs`], in the exact key order the pair-file
/// contract names.
#[derive(Serialize)]
struct PairLine<'a> {
    prompt: &'a str,
    completion: &'a str,
    field: &'a str,
}

/// `<out>.manifest.json`: the provenance record of one [`to_pairs`] run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairsManifest {
    /// The field corpus that was read.
    pub source_corpus: PathBuf,
    /// The field written into every pair line.
    pub field: String,
    /// Non-empty JSON lines read from the corpus.
    pub docs_in: usize,
    /// Pair lines written.
    pub docs_out: usize,
    /// Documents under [`MIN_PAIR_DOC_CHARS`] chars, never paired.
    pub skipped_short: usize,
    /// The cap that was in effect.
    pub max_docs: usize,
    /// Hex sha256 over the output file, for byte-level verification.
    pub output_sha256: String,
    /// Why the cap exists, stated where a reader of the pairs file sees it.
    pub note: String,
}

/// Render a field corpus (`dataset/all.jsonl`, one
/// `{"id","source","path","format","text"}` per line) into the trainer's
/// pair-file shape: one `{"prompt","completion","field"}` per line, split by
/// [`split_prompt_completion`].
///
/// The corpus is streamed, never read whole — math's `all.jsonl` is over a
/// billion chars. The selection is still the *smallest sha256(id) digests*,
/// so the cap is met by a deterministic subsample rather than the first
/// `max_docs` lines: pass 1 remembers only `(digest, line number)` per
/// eligible document (ids are short; the texts never accumulate), pass 2
/// re-reads the corpus and writes the selected documents in line order. Two
/// runs over the same corpus produce byte-identical output.
///
/// Writes `out` (via a temporary rename) and `<out>.manifest.json`.
pub fn to_pairs(
    corpus: &Path,
    field: &str,
    out: &Path,
    config: &ToPairsConfig,
) -> Result<PairsManifest> {
    config.validate()?;
    anyhow::ensure!(
        !field.trim().is_empty(),
        "to-pairs needs a non-empty field name"
    );
    anyhow::ensure!(
        corpus.is_file(),
        "corpus {} does not exist",
        corpus.display()
    );

    // Pass 1: eligibility and digests, streaming one line at a time.
    let mut docs_in = 0usize;
    let mut skipped_short = 0usize;
    let mut candidates: Vec<([u8; 32], u64)> = Vec::new();
    {
        let file = std::fs::File::open(corpus)
            .with_context(|| format!("open corpus {}", corpus.display()))?;
        for (n, line) in BufReader::new(file).lines().enumerate() {
            let line = line.with_context(|| format!("read {} line {}", corpus.display(), n + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(&line).with_context(|| {
                format!("{} line {} is not valid JSON", corpus.display(), n + 1)
            })?;
            let id = get_str(&value, "id")
                .ok_or_else(|| anyhow::anyhow!("{} line {} has no id", corpus.display(), n + 1))?;
            let text = get_str(&value, "text").ok_or_else(|| {
                anyhow::anyhow!("{} line {} has no text", corpus.display(), n + 1)
            })?;
            docs_in += 1;
            if text.chars().count() < MIN_PAIR_DOC_CHARS {
                skipped_short += 1;
                continue;
            }
            candidates.push((Sha256::digest(id.as_bytes()).into(), n as u64));
        }
    }

    // The line number breaks digest ties so the order is total.
    candidates.sort();
    let selected: HashSet<u64> = candidates
        .iter()
        .take(config.max_docs)
        .map(|&(_, n)| n)
        .collect();

    let file_name = out
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| anyhow::anyhow!("output path {} has no file name", out.display()))?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let tmp = out.with_file_name(format!(".{file_name}.tmp"));

    // Pass 2: write the selected documents, hashing as we go.
    let mut docs_out = 0usize;
    let mut hasher = Sha256::new();
    {
        let file = std::fs::File::open(corpus)
            .with_context(|| format!("re-open corpus {}", corpus.display()))?;
        let mut writer = BufWriter::new(
            std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?,
        );
        for (n, line) in BufReader::new(file).lines().enumerate() {
            if !selected.contains(&(n as u64)) {
                continue;
            }
            let line = line.with_context(|| format!("read {} line {}", corpus.display(), n + 1))?;
            let value: Value = serde_json::from_str(&line).with_context(|| {
                format!("{} line {} is not valid JSON", corpus.display(), n + 1)
            })?;
            let text = get_str(&value, "text").ok_or_else(|| {
                anyhow::anyhow!("{} line {} has no text", corpus.display(), n + 1)
            })?;
            let (prompt, completion) = split_prompt_completion(text);
            let record = PairLine {
                prompt: &prompt,
                completion: &completion,
                field,
            };
            let bytes = serde_json::to_vec(&record).context("serialize pair line")?;
            writer
                .write_all(&bytes)
                .and_then(|()| writer.write_all(b"\n"))
                .with_context(|| format!("write {}", tmp.display()))?;
            hasher.update(&bytes);
            hasher.update(b"\n");
            docs_out += 1;
        }
        writer
            .flush()
            .with_context(|| format!("flush {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, out)
        .with_context(|| format!("rename {} to {}", tmp.display(), out.display()))?;

    let digest = hasher.finalize();
    let manifest = PairsManifest {
        source_corpus: corpus.to_path_buf(),
        field: field.to_string(),
        docs_in,
        docs_out,
        skipped_short,
        max_docs: config.max_docs,
        output_sha256: digest.iter().map(|b| format!("{b:02x}")).collect(),
        note: "the trainer (student::LanguageBatch::load) reads the whole pairs file into \
               memory, so max_docs — not the corpus size — bounds that load; raise it only \
               with the RAM to match"
            .to_string(),
    };
    let manifest_json =
        serde_json::to_string_pretty(&manifest).context("serialize pairs manifest")?;
    crate::nim::write_atomic(
        &out.with_file_name(format!("{file_name}.manifest.json")),
        manifest_json.as_bytes(),
    )?;
    eprintln!(
        "[to-pairs] {}: {} docs in, {} pairs out ({} skipped under {} chars, cap {}) -> {}",
        field,
        docs_in,
        docs_out,
        skipped_short,
        MIN_PAIR_DOC_CHARS,
        config.max_docs,
        out.display()
    );
    Ok(manifest)
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn scratch_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "dblocks-fieldnorm-test-{}-{tag}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn doc_text(outcome: RowOutcome) -> String {
        match outcome {
            RowOutcome::Doc(text) | RowOutcome::DocCounted(text, _) => text,
            RowOutcome::Skip(reason) => panic!("expected a document, got skip {reason}"),
        }
    }

    #[test]
    fn chat_triple_renders_as_instruction_text() {
        let row: Value = serde_json::json!({
            "system": "You are an advanced AI assistant specialized in cybersecurity.",
            "user": "What is a SYN scan?",
            "assistant": "A SYN scan sends TCP SYN packets and reads the reply."
        });
        let (text, counted) = match render_chat(&row, false) {
            RowOutcome::DocCounted(text, counter) => (text, Some(counter)),
            RowOutcome::Doc(text) => (text, None),
            RowOutcome::Skip(reason) => panic!("unexpected skip: {reason}"),
        };
        assert_eq!(
            text,
            "What is a SYN scan?\n\nA SYN scan sends TCP SYN packets and reads the reply."
        );
        // The boilerplate system block is dropped but never silently.
        assert_eq!(counted, Some("chat-system-block-dropped"));
        assert!(!text.contains("advanced AI assistant"));
        // A row without an assistant half is skipped, not guessed at.
        let broken: Value = serde_json::json!({"user": "hello"});
        assert!(matches!(
            render_chat(&broken, false),
            RowOutcome::Skip("chat-row-without-user-or-assistant")
        ));
    }

    #[test]
    fn escaped_chat_rows_get_one_escape_layer_removed() {
        // The *_escaped.jsonl cyber sources were escape-encoded twice on
        // disk: after JSON parsing the text still holds literal `\n` pairs.
        let row: Value = serde_json::json!({
            "system": "sys",
            "user": "Show me\\na payload.",
            "assistant": "Line one\\nLine two with \\\"quotes\\\" and a literal \\\\x41."
        });
        let text = doc_text(render_chat(&row, true));
        assert_eq!(
            text,
            "Show me\na payload.\n\nLine one\nLine two with \"quotes\" and a literal \\x41."
        );
    }

    #[test]
    fn metamath_row_renders_query_then_response() {
        let row: Value = serde_json::json!({
            "query": "What is $1+1$?",
            "response": "The answer is: 2",
            "type": "GSM_AnsAug"
        });
        assert_eq!(
            doc_text(render_metamath(&row)),
            "What is $1+1$?\n\nThe answer is: 2"
        );
        let broken: Value = serde_json::json!({"query": "no response"});
        assert!(matches!(render_metamath(&broken), RowOutcome::Skip(_)));
    }

    #[test]
    fn numina_row_honors_the_validity_flags() {
        let good: Value = serde_json::json!({
            "problem": "Find $1+1$.",
            "solution": "We have $1+1=2$.",
            "problem_is_valid": "Yes",
            "solution_is_valid": "Yes"
        });
        assert_eq!(
            doc_text(render_numina(&good)),
            "Find $1+1$.\n\nWe have $1+1=2$."
        );
        let invalid: Value = serde_json::json!({
            "problem": "Find $1+1$.",
            "solution": "We have $1+1=2$.",
            "problem_is_valid": "Incomplete",
            "solution_is_valid": "Yes"
        });
        assert!(matches!(
            render_numina(&invalid),
            RowOutcome::Skip("invalid-flag")
        ));
    }

    #[test]
    fn ai2_arc_row_renders_lettered_choices() {
        let row: Value = serde_json::json!({
            "id": "Mercury_SC_415696",
            "question": "Which property of a mineral can be determined just by looking at it?",
            "choices": {"text": ["luster", "mass"], "label": ["A", "B"]},
            "answerKey": "A"
        });
        assert_eq!(
            doc_text(render_ai2_arc(&row)),
            "Which property of a mineral can be determined just by looking at it?\n\nA. luster\nB. mass\n\nAnswer: A"
        );
        let no_answer: Value = serde_json::json!({
            "question": "q",
            "choices": {"text": ["a"], "label": ["A"]},
            "answerKey": ""
        });
        assert!(matches!(
            render_ai2_arc(&no_answer),
            RowOutcome::Skip("arc-row-without-answer")
        ));
    }

    #[test]
    fn medmcqa_row_renders_options_answer_and_explanation() {
        let row: Value = serde_json::json!({
            "question": "Most common cause of X?",
            "opa": "A", "opb": "B", "opc": "C", "opd": "D",
            "cop": 2,
            "choice_type": "single",
            "exp": "Because C."
        });
        assert_eq!(
            doc_text(render_medmcqa(&row)),
            "Most common cause of X?\n\nA. A\nB. B\nC. C\nD. D\n\nAnswer: C\n\nBecause C."
        );
        let out_of_range: Value = serde_json::json!({
            "question": "q", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 7
        });
        assert!(matches!(
            render_medmcqa(&out_of_range),
            RowOutcome::Skip("medmcqa-row-with-out-of-range-correct-option")
        ));
    }

    #[test]
    fn text_to_sql_row_renders_task_schema_sql() {
        let row: Value = serde_json::json!({
            "sql_prompt": "Total sales per region?",
            "sql_context": "CREATE TABLE t (region TEXT);",
            "sql": "SELECT region, SUM(x) FROM t GROUP BY region;",
            "sql_explanation": "Groups by region."
        });
        let text = doc_text(render_text_to_sql(&row));
        assert!(text.starts_with("Task: Total sales per region?\n\nSchema:\n"));
        assert!(text.contains("\n\nSQL:\nSELECT region"));
        assert!(text.ends_with("\n\nGroups by region."));
    }

    #[test]
    fn hugo_markdown_strips_front_matter_shortcodes_and_links() {
        let mut stats = MarkdownStrip::default();
        let input = "---\ntitle: Install Guide\nweight: 10\n---\n\n# Installing\n\n{{< note >}}\nRead the [setup page](/setup/) and ![diagram](d.png).\n{{< /note >}}\n\n```bash\nkubectl apply -f x.yaml  # kept [verbatim](x)\n```\n\n{{% thirdparty-content %}}\n";
        let text = hugo_markdown_to_text(input, &mut stats);
        assert!(stats.front_matter);
        assert_eq!(stats.shortcodes, 3);
        assert!(
            !text.contains("title:"),
            "front matter must be gone: {text}"
        );
        assert!(!text.contains("{{<"), "shortcode tags must be gone: {text}");
        assert!(text.contains("Installing"));
        assert!(text.contains("Read the setup page and diagram."));
        assert!(
            text.contains("kubectl apply -f x.yaml  # kept [verbatim](x)"),
            "fenced code stays verbatim: {text}"
        );
        assert!(!text.contains("thirdparty"));
    }

    #[test]
    fn sgml_strips_elements_decodes_entities_and_counts_unhandled() {
        let mut stats = SgmlStrip::default();
        let input = "<!-- doc header -->\n<chapter id=\"x\">\n <title>Configuration &amp; Tuning</title>\n <para>\n  See &sect; 4 &mdash; the &weirdentity; setting.\n </para>\n <programlisting><![CDATA[if (a < b) { f(); }]]></programlisting>\n</chapter>\n";
        let text = sgml_to_text(input, &mut stats);
        assert_eq!(stats.comments, 1);
        assert_eq!(stats.cdata_sections, 1);
        assert!(text.contains("Configuration & Tuning"), "{text}");
        assert!(text.contains("See § 4 — the"), "{text}");
        assert!(
            text.contains("if (a < b) { f(); }"),
            "CDATA content stays verbatim: {text}"
        );
        assert!(!text.contains("<chapter"), "{text}");
        assert_eq!(stats.unhandled.get("entity:&weirdentity;"), Some(&1));
    }

    #[test]
    fn sgml_numeric_entities_decode() {
        let mut stats = SgmlStrip::default();
        let text = sgml_to_text("<para>a&#65;b&#x42;</para>", &mut stats);
        assert_eq!(text, "aAbB");
    }

    /// Build a minimal field corpus on disk: one chat JSONL source, one
    /// markdown tree.
    fn mini_spec(datasets: &Path) -> FieldSpec {
        std::fs::create_dir_all(datasets.join("cyber/fenrir")).unwrap();
        std::fs::write(
            datasets.join("cyber/fenrir/chat.jsonl"),
            concat!(
                "{\"system\":\"sys\",\"user\":\"What is phishing?\",\"assistant\":\"Phishing is a social-engineering attack that tricks a user into revealing credentials.\"}\n",
                "{\"system\":\"sys\",\"user\":\"What is phishing?\",\"assistant\":\"Phishing is a social-engineering attack that tricks a user into revealing credentials.\"}\n",
                "{\"user\":\"Give me the key\",\"assistant\":\"Sure: -----BEGIN PRIVATE KEY----- abc123\"}\n",
                "{\"user\":\"lone row\"}\n",
                "{\"system\":\"sys\",\"user\":\"How do I list pods?\",\"assistant\":\"You can list pods with kubectl get pods.\"}\n"
            ),
        )
        .unwrap();
        std::fs::create_dir_all(datasets.join("tech/site/content")).unwrap();
        std::fs::write(
            datasets.join("tech/site/content/page.md"),
            "---\ntitle: T\n---\n\n# Heading\n\nBody text about clusters and containers and nodes and pods and services and deployments and namespaces and volumes and more words here.",
        )
        .unwrap();
        FieldSpec {
            field: "mini".to_string(),
            sources: vec![
                source(
                    "chat",
                    SourceFormat::ChatJsonl,
                    &["cyber/fenrir/chat.jsonl"],
                ),
                source("site", SourceFormat::MarkdownTree, &["tech/site/content"]),
            ],
        }
    }

    #[test]
    fn build_filters_cheats_dedups_and_resumes() {
        let root = scratch_dir("build");
        let datasets = root.join("datasets");
        let corpus_dir = root.join("corpora/mini");
        let spec = mini_spec(&datasets);
        let config = FieldnormConfig {
            identity_path: root.join("absent-identity.jsonl"),
            ..FieldnormConfig::default()
        };

        let manifest = build_field_corpus(&spec, &datasets, &corpus_dir, &config, false).unwrap();

        // The private-key row is dropped, counted under its cheat class —
        // the malicious-input path is the manifest's job to expose.
        let chat = &manifest.sources["chat"];
        assert_eq!(chat.documents, 2, "two clean chat rows survive: {chat:?}");
        assert!(
            chat.dropped_by_class
                .get("data-secrets")
                .copied()
                .unwrap_or(0)
                >= 1,
            "the private key row is a secrets drop: {chat:?}"
        );
        // The duplicate row is the near-dup drop.
        assert_eq!(chat.near_duplicates, 1, "{chat:?}");
        // The row without an assistant is a counted skip.
        assert_eq!(
            chat.rows_skipped.get("chat-row-without-user-or-assistant"),
            Some(&1)
        );
        // The system boilerplate is counted.
        assert_eq!(chat.counters.get("chat-system-block-dropped"), Some(&3));
        // "You can list pods…" is prose, not a pasted-answer artifact: the
        // quarantine-grade finding is counted, the document is kept.
        assert!(
            chat.quarantine_findings
                .get("data-broken")
                .copied()
                .unwrap_or(0)
                >= 1,
            "{chat:?}"
        );

        let site = &manifest.sources["site"];
        assert_eq!(site.documents, 1, "{site:?}");
        assert_eq!(site.counters.get("front-matter"), Some(&1));

        // Deterministic ids and the layout contract.
        let all = std::fs::read_to_string(corpus_dir.join("dataset/all.jsonl")).unwrap();
        let ids: Vec<String> = all
            .lines()
            .map(|l| {
                serde_json::from_str::<Value>(l).unwrap()["id"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(
            ids,
            vec!["chat-000001", "chat-000002", "site-000001"],
            "{all}"
        );
        assert!(corpus_dir.join("text/chat/chat-000001.txt").is_file());
        assert!(corpus_dir.join("dataset/chat.jsonl").is_file());
        assert!(corpus_dir.join("dataset/site.jsonl").is_file());
        assert!(corpus_dir.join("dataset/manifest.json").is_file());

        // The identity file was absent: a note, not an error.
        assert!(manifest.identity.is_none());
        assert!(manifest.notes.iter().any(|n| n.contains("identity file")));

        // Resume: a second run skips completed sources and reproduces the
        // totals and the byte-identical all.jsonl.
        let rerun = build_field_corpus(&spec, &datasets, &corpus_dir, &config, false).unwrap();
        assert_eq!(rerun.resumed_sources, vec!["chat", "site"]);
        assert_eq!(rerun.total_documents, manifest.total_documents);
        assert_eq!(rerun.total_chars, manifest.total_chars);
        let all2 = std::fs::read_to_string(corpus_dir.join("dataset/all.jsonl")).unwrap();
        assert_eq!(all, all2);

        // Force rebuilds.
        let forced = build_field_corpus(&spec, &datasets, &corpus_dir, &config, true).unwrap();
        assert!(forced.resumed_sources.is_empty());
        assert_eq!(forced.total_documents, manifest.total_documents);

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn identity_file_mixes_in_as_a_source() {
        let root = scratch_dir("identity");
        let datasets = root.join("datasets");
        let spec = mini_spec(&datasets);
        let identity = root.join("identity.jsonl");
        std::fs::write(
            &identity,
            "{\"prompt\":\"Who are you?\",\"completion\":\"I am nina, from cybersec.org.za, made by the CEO Stephan.\"}\n",
        )
        .unwrap();
        let config = FieldnormConfig {
            identity_path: identity.clone(),
            ..FieldnormConfig::default()
        };
        let corpus_dir = root.join("corpora/mini");
        let manifest = build_field_corpus(&spec, &datasets, &corpus_dir, &config, false).unwrap();
        let note = manifest.identity.as_ref().expect("identity mixed in");
        assert_eq!(note.documents, 1);
        assert_eq!(manifest.sources["identity"].documents, 1);
        let all = std::fs::read_to_string(corpus_dir.join("dataset/all.jsonl")).unwrap();
        assert!(all.contains("identity-000001"), "{all}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn unknown_field_names_the_known_ones() {
        let err = field_spec("alchemy").unwrap_err().to_string();
        assert!(err.contains("math"), "{err}");
        assert!(err.contains("alchemy"), "{err}");
    }

    #[test]
    fn a_parquet_source_without_conversion_is_a_loud_error() {
        let root = scratch_dir("parquet");
        let datasets = root.join("datasets");
        std::fs::create_dir_all(&datasets).unwrap();
        let spec = FieldSpec {
            field: "mini".to_string(),
            sources: vec![parquet_source(
                "rows",
                SourceFormat::NuminaJsonl,
                &["x.parquet"],
                "rows.jsonl",
                &[],
            )],
        };
        let err = build_field_corpus(
            &spec,
            &datasets,
            &root.join("corpora/mini"),
            &FieldnormConfig::default(),
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("parquet2jsonl.py"), "{err}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    // -----------------------------------------------------------------
    // pentest converters: NVD, ATT&CK STIX, CWE/CAPEC XML
    // -----------------------------------------------------------------

    fn nvd_row() -> Value {
        serde_json::json!({
            "cve": {
                "id": "CVE-2025-0001",
                "published": "2025-01-01T14:15:23.590",
                "vulnStatus": "Analyzed",
                "descriptions": [
                    {"lang": "es", "value": "descripcion"},
                    {"lang": "en", "value": "A buffer overflow in the widget parser allows remote attackers to execute arbitrary code via a crafted packet."}
                ],
                "weaknesses": [
                    {"description": [{"lang": "en", "value": "CWE-787"}]},
                    {"description": [{"lang": "en", "value": "CWE-787"}]}
                ],
                "metrics": {
                    "cvssMetricV31": [{
                        "type": "Primary",
                        "cvssData": {"baseScore": 9.8},
                        "baseSeverity": "CRITICAL"
                    }]
                },
                "references": [
                    {"url": "https://vendor.example/advisory"},
                    {"url": "https://exploit.example/poc", "tags": ["Exploit", "Third Party Advisory"]}
                ]
            }
        })
    }

    #[test]
    fn nvd_row_renders_id_description_weaknesses_severity_references() {
        let text = doc_text(render_nvd_cve(&nvd_row(), false));
        assert!(
            text.starts_with("CVE-2025-0001 (published 2025-01-01)\n\nA buffer overflow"),
            "{text}"
        );
        assert!(
            text.contains("\n\nWeaknesses: CWE-787\n\nSeverity: CRITICAL (base score 9.8)"),
            "{text}"
        );
        assert!(
            text.contains("\n- https://exploit.example/poc [Exploit]"),
            "{text}"
        );
        assert!(
            text.contains("\n- https://vendor.example/advisory\n"),
            "{text}"
        );
        // Duplicate weakness ids are listed once.
        assert_eq!(text.matches("CWE-787").count(), 1, "{text}");
    }

    #[test]
    fn nvd_exploit_filter_rejected_and_reserved_rows_skip() {
        // The exploit-only variant keeps the row (it has an Exploit tag)…
        assert!(matches!(
            render_nvd_cve(&nvd_row(), true),
            RowOutcome::Doc(_)
        ));
        // …and drops an otherwise fine row whose references are untagged.
        let mut plain = nvd_row();
        plain["cve"]["references"]
            .as_array_mut()
            .unwrap()
            .retain(|r| r.get("tags").is_none());
        assert!(matches!(
            render_nvd_cve(&plain, true),
            RowOutcome::Skip("no-exploit-reference")
        ));
        assert!(matches!(render_nvd_cve(&plain, false), RowOutcome::Doc(_)));

        let mut rejected = nvd_row();
        rejected["cve"]["vulnStatus"] = Value::String("Rejected".to_string());
        assert!(matches!(
            render_nvd_cve(&rejected, false),
            RowOutcome::Skip("rejected-status")
        ));

        let mut reserved = nvd_row();
        reserved["cve"]["descriptions"] = serde_json::json!([
            {"lang": "en", "value": "** RESERVED ** This candidate has been reserved by an organization or individual."}
        ]);
        assert!(matches!(
            render_nvd_cve(&reserved, false),
            RowOutcome::Skip("reserved-placeholder")
        ));
    }

    #[test]
    fn stix_attack_pattern_renders_and_the_recon_filter_is_by_tactic() {
        let technique = serde_json::json!({
            "type": "attack-pattern",
            "name": "Active Scanning",
            "description": "Adversaries may probe victim infrastructure to identify targets.",
            "revoked": false,
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": "reconnaissance"},
                {"kill_chain_name": "other", "phase_name": "ignore-me"}
            ],
            "external_references": [
                {"source_name": "mitre-attack", "external_id": "T1595", "url": "https://attack.mitre.org/techniques/T1595"}
            ]
        });
        let text = doc_text(render_stix(&technique, false));
        assert_eq!(
            text,
            "T1595: Active Scanning\n\nAdversaries may probe victim infrastructure to identify targets.\n\nTactics: reconnaissance"
        );
        // Recon variant keeps it (TA0043); a lateral-movement technique is
        // dropped; tactics are not techniques and are dropped in recon mode.
        assert!(matches!(render_stix(&technique, true), RowOutcome::Doc(_)));
        let mut lateral = technique.clone();
        lateral["kill_chain_phases"] = serde_json::json!([
            {"kill_chain_name": "mitre-attack", "phase_name": "lateral-movement"}
        ]);
        assert!(matches!(
            render_stix(&lateral, true),
            RowOutcome::Skip("not-a-recon-technique")
        ));
        let tactic = serde_json::json!({
            "type": "x-mitre-tactic",
            "name": "Reconnaissance",
            "description": "The adversary is trying to gather information.",
            "x_mitre_shortname": "reconnaissance",
            "external_references": [
                {"source_name": "mitre-attack", "external_id": "TA0043"}
            ]
        });
        let text = doc_text(render_stix(&tactic, false));
        assert_eq!(
            text,
            "TA0043: Reconnaissance (tactic)\n\nThe adversary is trying to gather information."
        );
        assert!(matches!(
            render_stix(&tactic, true),
            RowOutcome::Skip("stix-type-not-kept")
        ));
        // Revoked and deprecated objects are historical record.
        let mut revoked = technique.clone();
        revoked["revoked"] = Value::Bool(true);
        assert!(matches!(
            render_stix(&revoked, false),
            RowOutcome::Skip("revoked-or-deprecated")
        ));
        // Other STIX types (identity, campaign, malware, …) are not kept.
        let identity = serde_json::json!({"type": "identity", "name": "MITRE"});
        assert!(matches!(
            render_stix(&identity, false),
            RowOutcome::Skip("stix-type-not-kept")
        ));
    }

    #[test]
    fn split_xml_entries_matches_exact_tags_and_respects_cdata() {
        let xml = "<Weaknesses>\n<Weakness ID=\"1\" Name=\"A\"><Description>one</Description></Weakness>\n<Weakness ID=\"2\" Name=\"B\"><Description><![CDATA[a </Weakness> b]]></Description></Weakness>\n</Weaknesses>";
        let entries = split_xml_entries(xml, "Weakness");
        assert_eq!(entries.len(), 2);
        assert_eq!(xml_attr(entries[0].attrs, "ID"), Some("1"));
        assert_eq!(xml_attr(entries[1].attrs, "Name"), Some("B"));
        // The CDATA literal cannot fake a close tag.
        assert!(entries[1].inner.contains("</Weakness> b]]>"));
        // A namespaced lookalike prefix is a different element.
        assert!(split_xml_entries("<x:Weakness ID=\"9\"/>", "Weakness").is_empty());
    }

    #[test]
    fn cwe_weakness_renders_fields_and_strips_xhtml() {
        let xml = r#"<Weaknesses>
<Weakness ID="79" Name="Improper Neutralization of Input During Web Page Generation" Abstraction="Base">
  <Description><xhtml:div>The product does not neutralize input &amp; output.</xhtml:div></Description>
  <Extended_Description><xhtml:p>Commonly called XSS.</xhtml:p></Extended_Description>
  <Modes_Of_Introduction><Introduction><Phase>Implementation</Phase></Introduction></Modes_Of_Introduction>
  <Common_Consequences>
    <Consequence><Scope>Confidentiality</Scope><Scope>Integrity</Scope><Impact>Execute Unauthorized Code or Commands</Impact></Consequence>
  </Common_Consequences>
  <Detection_Methods>
    <Detection_Method><Method>Automated Static Analysis</Method><Description>Use a scanner.</Description></Detection_Method>
  </Detection_Methods>
  <Potential_Mitigations>
    <Mitigation><Phase>Implementation</Phase><Description>Escape all output.</Description></Mitigation>
  </Potential_Mitigations>
  <Observed_Examples>
    <Observed_Example><Reference>CVE-2021-1234</Reference><Description>XSS in a forum.</Description></Observed_Example>
  </Observed_Examples>
</Weakness>
<Weakness ID="999" Name="Empty" />
</Weaknesses>"#;
        let entries = split_xml_entries(xml, "Weakness");
        assert_eq!(entries.len(), 2);
        let mut stats = XmlStrip::default();
        let text = doc_text(render_cwe_weakness(&entries[0], &mut stats));
        assert!(
            text.starts_with("CWE-79: Improper Neutralization of Input During Web Page Generation\n\nThe product does not neutralize input & output."),
            "{text}"
        );
        assert!(text.contains("\n\nCommonly called XSS."), "{text}");
        assert!(
            text.contains("\n\nModes of introduction: Implementation"),
            "{text}"
        );
        assert!(
            text.contains("\n- Scope: Confidentiality, Integrity; Impact: Execute Unauthorized Code or Commands"),
            "{text}"
        );
        assert!(
            text.contains("\n- Automated Static Analysis: Use a scanner."),
            "{text}"
        );
        assert!(
            text.contains("\n- Implementation: Escape all output."),
            "{text}"
        );
        assert!(
            text.contains("\n- CVE-2021-1234: XSS in a forum."),
            "{text}"
        );
        assert!(!text.contains("xhtml"), "{text}");
        // The self-closing empty entry is a counted skip, not an empty doc.
        assert!(matches!(
            render_cwe_weakness(&entries[1], &mut stats),
            RowOutcome::Skip("cwe-entry-without-description")
        ));
        assert!(stats.elements > 0);
        assert_eq!(stats.entities_decoded, 1);
    }

    #[test]
    fn capec_pattern_renders_fields() {
        let xml = r#"<Attack_Patterns>
<Attack_Pattern ID="66" Name="SQL Injection" Abstraction="Standard">
  <Description>An attacker injects SQL into a parameter.</Description>
  <Likelihood_Of_Attack>High</Likelihood_Of_Attack>
  <Typical_Severity>Very High</Typical_Severity>
  <Prerequisites><Prerequisite>The application uses a SQL database.</Prerequisite></Prerequisites>
  <Skills_Required><Skill Level="Medium">Knowledge of SQL syntax.</Skill></Skills_Required>
  <Consequences>
    <Consequence><Scope>Confidentiality</Scope><Impact>Read Data</Impact></Consequence>
  </Consequences>
  <Mitigations><Mitigation>Use parameterized queries.</Mitigation></Mitigations>
  <Example_Instances><Example>Poison the login form.</Example></Example_Instances>
  <Related_Weaknesses><Related_Weakness CWE_ID="89" /></Related_Weaknesses>
</Attack_Pattern>
</Attack_Patterns>"#;
        let entries = split_xml_entries(xml, "Attack_Pattern");
        assert_eq!(entries.len(), 1);
        let mut stats = XmlStrip::default();
        let text = doc_text(render_capec_pattern(&entries[0], &mut stats));
        assert!(
            text.starts_with(
                "CAPEC-66: SQL Injection\n\nAn attacker injects SQL into a parameter."
            ),
            "{text}"
        );
        assert!(
            text.contains("\n\nLikelihood of attack: High. Typical severity: Very High"),
            "{text}"
        );
        assert!(
            text.contains("\n- The application uses a SQL database."),
            "{text}"
        );
        assert!(text.contains("\n- Knowledge of SQL syntax."), "{text}");
        assert!(text.contains("\n- Use parameterized queries."), "{text}");
        assert!(text.contains("\n- Poison the login form."), "{text}");
        assert!(
            text.contains("\n- Scope: Confidentiality; Impact: Read Data"),
            "{text}"
        );
        assert!(text.ends_with("\n\nRelated weaknesses: CWE-89"), "{text}");
    }

    #[test]
    fn xml_inner_text_decodes_entities_and_counts_unhandled() {
        let mut stats = XmlStrip::default();
        let text = xml_inner_text(
            "<p>a &lt;b&gt; &amp; &#65;&#x42; &bogus;</p><?pi x?>",
            &mut stats,
        );
        assert_eq!(text, "a <b> & AB");
        assert_eq!(stats.entities_decoded, 5);
        assert_eq!(stats.unhandled.get("entity:&bogus;"), Some(&1));
        assert_eq!(stats.unhandled.get("processing-instruction"), Some(&1));
    }

    #[test]
    fn pentest_build_drops_malicious_rows_and_counts_them() {
        let root = scratch_dir("pentest");
        let datasets = root.join("datasets");
        std::fs::create_dir_all(datasets.join("pentest/nvd-cve")).unwrap();
        std::fs::write(
            datasets.join("pentest/nvd-cve/feed.jsonl"),
            concat!(
                "{\"cve\":{\"id\":\"CVE-2025-1000\",\"published\":\"2025-02-01T00:00:00\",\"vulnStatus\":\"Analyzed\",\"descriptions\":[{\"lang\":\"en\",\"value\":\"A path traversal in the upload handler allows reading arbitrary files from the server filesystem.\"}],\"references\":[{\"url\":\"https://a.example\"}]}}\n",
                "{\"cve\":{\"id\":\"CVE-2025-1001\",\"published\":\"2025-02-02T00:00:00\",\"vulnStatus\":\"Analyzed\",\"descriptions\":[{\"lang\":\"en\",\"value\":\"Leaked material: -----BEGIN PRIVATE KEY----- abc123 -----END PRIVATE KEY----- embedded in a report.\"}],\"references\":[]}}\n",
                "{\"cve\":{\"id\":\"CVE-2025-1002\",\"vulnStatus\":\"Rejected\",\"descriptions\":[{\"lang\":\"en\",\"value\":\"** REJECT ** do not use.\"}]}}\n"
            ),
        )
        .unwrap();
        let spec = FieldSpec {
            field: "pentest/research".to_string(),
            sources: vec![source(
                "nvd-cve",
                SourceFormat::NvdCveJsonl,
                &["pentest/nvd-cve/feed.jsonl"],
            )],
        };
        let config = FieldnormConfig {
            identity_path: root.join("absent-identity.jsonl"),
            ..FieldnormConfig::default()
        };
        let corpus_dir = root.join("corpora/pentest/research");
        let manifest = build_field_corpus(&spec, &datasets, &corpus_dir, &config, false).unwrap();
        let nvd = &manifest.sources["nvd-cve"];
        assert_eq!(nvd.documents, 1, "{nvd:?}");
        assert!(
            nvd.dropped_by_class
                .get("data-secrets")
                .copied()
                .unwrap_or(0)
                >= 1,
            "the private-key description is a secrets drop: {nvd:?}"
        );
        assert_eq!(nvd.rows_skipped.get("rejected-status"), Some(&1), "{nvd:?}");
        let all = std::fs::read_to_string(corpus_dir.join("dataset/all.jsonl")).unwrap();
        assert!(all.contains("CVE-2025-1000"), "{all}");
        assert!(!all.contains("CVE-2025-1001"), "{all}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn pentest_fields_are_registered_with_their_sources() {
        let recon = field_spec("pentest/recon").unwrap();
        assert_eq!(recon.sources.len(), 1);
        assert_eq!(recon.sources[0].name, "attack-recon");
        let zeroday = field_spec("pentest/zeroday").unwrap();
        assert_eq!(
            zeroday
                .sources
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            vec!["capec", "nvd-cve-exploit"]
        );
        let methodology = field_spec("pentest/methodology").unwrap();
        assert_eq!(
            methodology
                .sources
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            vec!["attack-full", "owasp-cheatsheets"]
        );
        assert!(available_fields().contains(&"pentest/research"));
    }

    // -----------------------------------------------------------------
    // to-pairs: the split rule
    // -----------------------------------------------------------------

    /// Thirty 3-char words, space-joined: 119 chars, whitespace at every
    /// index 4k+3.
    fn thirty_words() -> Vec<String> {
        (0..30).map(|i| format!("w{i:02}")).collect()
    }

    #[test]
    fn split_prefers_the_first_blank_line() {
        // The blank line (at char 13) comes before the quarter point, so it
        // is the boundary: the prompt is the whole first paragraph.
        let body = "b".repeat(200);
        let text = format!("Question one?\n\n{body}");
        let (prompt, completion) = split_prompt_completion(&text);
        assert_eq!(prompt, "Question one?");
        assert_eq!(completion, body);
    }

    #[test]
    fn split_without_a_blank_line_cuts_at_the_quarter_whitespace() {
        // 119 chars; the quarter point is char 29, the last whitespace at or
        // before it is index 27 — the prompt ends there, mid-sentence but
        // never mid-word.
        let words = thirty_words();
        let text = words.join(" ");
        assert_eq!(text.chars().count(), 119);
        let (prompt, completion) = split_prompt_completion(&text);
        assert_eq!(prompt, "w00 w01 w02 w03 w04 w05 w06");
        assert_eq!(completion, words[7..].join(" "));
    }

    #[test]
    fn split_uses_the_quarter_when_the_blank_line_comes_later() {
        // Whichever boundary comes first wins: with the blank line at char
        // 119 of 221, the quarter cut (last whitespace <= char 55) is first.
        let words = thirty_words();
        let text = format!("{}\n\n{}", words.join(" "), "x".repeat(100));
        assert_eq!(text.chars().count(), 221);
        let (prompt, completion) = split_prompt_completion(&text);
        assert_eq!(prompt, words[..14].join(" "));
        assert!(completion.starts_with("w14"), "{completion}");
        assert!(completion.contains("\n\n"), "{completion}");
    }

    #[test]
    fn a_blank_line_first_document_falls_to_the_quarter_rule() {
        // A blank line at the very start would make an empty prompt, so it
        // is not a boundary; the quarter rule applies to the whole text.
        let words = thirty_words();
        let text = format!("\n\n{}", words.join(" "));
        let (prompt, completion) = split_prompt_completion(&text);
        assert_eq!(prompt, "w00 w01 w02 w03 w04 w05 w06");
        assert_eq!(completion, words[7..].join(" "));
    }

    #[test]
    fn split_never_divides_a_multibyte_char() {
        // 40 x 'é' (2 bytes each), a space, 60 x 'x'. Byte arithmetic would
        // put the quarter (byte 25 of 101) inside an 'é'; char arithmetic
        // finds no whitespace that early and cuts at the one there is.
        let text = format!("{} {}", "é".repeat(40), "x".repeat(60));
        let (prompt, completion) = split_prompt_completion(&text);
        assert_eq!(prompt, "é".repeat(40));
        assert_eq!(completion, "x".repeat(60));
    }

    #[test]
    fn split_keeps_at_least_the_raw_chars_when_a_prompt_would_trim_empty() {
        // The blank-line boundary at char 3 leaves a whitespace-only prompt;
        // the minimum-1-char rule keeps the raw slice rather than trimming
        // to nothing.
        let text = format!("   \n\n{}", "answer ".repeat(20));
        let (prompt, completion) = split_prompt_completion(&text);
        assert_eq!(prompt, "   ");
        assert!(completion.starts_with("answer"), "{completion}");
    }

    // -----------------------------------------------------------------
    // to-pairs: the streaming conversion
    // -----------------------------------------------------------------

    /// Write a pairs-test corpus: `docs` documents with id `<prefix>-<i>`
    /// whose text clears the 64-char floor, plus a below-floor document
    /// (`<prefix>-short`, skipped) and an exactly-64-char one
    /// (`<prefix>-edge64`, kept). Returns the *eligible* ids in line order.
    fn write_pairs_corpus(path: &Path, docs: usize, prefix: &str) -> Vec<String> {
        fn line(out: &mut String, id: &str, text: String) {
            out.push_str(
                &serde_json::json!({"id": id, "source": "t", "path": "t", "format": "jsonl", "text": text})
                    .to_string(),
            );
            out.push('\n');
        }
        let mut ids = Vec::new();
        let mut out = String::new();
        for i in 0..docs {
            let id = format!("{prefix}-{i:04}");
            let text = format!(
                "Question {i} asks something with enough words to pass the floor.\n\n\
                 Answer {i} goes on at some length so the whole document clears it."
            );
            assert!(text.chars().count() >= MIN_PAIR_DOC_CHARS);
            line(&mut out, &id, text);
            ids.push(id);
        }
        // Below the floor: never eligible. Not pushed to `ids`.
        line(
            &mut out,
            &format!("{prefix}-short"),
            "x".repeat(MIN_PAIR_DOC_CHARS - 1),
        );
        // Exactly at the floor: eligible.
        let edge = format!("{prefix}-edge64");
        line(&mut out, &edge, "y".repeat(MIN_PAIR_DOC_CHARS));
        ids.push(edge);
        std::fs::write(path, out).unwrap();
        ids
    }

    /// The corpus ids behind an emitted pairs file, recovered from the
    /// prompts the test corpus's shape makes recognizable.
    fn emitted_ids(path: &Path, prefix: &str) -> Vec<String> {
        let mut out = Vec::new();
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let value: Value = serde_json::from_str(line).unwrap();
            let prompt = value["prompt"].as_str().unwrap();
            if let Some(rest) = prompt.strip_prefix("Question ") {
                let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                let n: usize = num.parse().unwrap();
                out.push(format!("{prefix}-{n:04}"));
            } else {
                assert_eq!(prompt, "y".repeat(MIN_PAIR_DOC_CHARS / 4));
                out.push(format!("{prefix}-edge64"));
            }
        }
        out.sort();
        out
    }

    /// The expected selection: the `max_docs` smallest sha256(id) digests.
    fn expected_selection(ids: &[String], max_docs: usize) -> Vec<String> {
        let mut by_hash: Vec<([u8; 32], &String)> = ids
            .iter()
            .map(|id| (Sha256::digest(id.as_bytes()).into(), id))
            .collect();
        by_hash.sort();
        let mut out: Vec<String> = by_hash
            .iter()
            .take(max_docs)
            .map(|(_, id)| id.to_string())
            .collect();
        out.sort();
        out
    }

    #[test]
    fn to_pairs_keeps_an_exactly_64_char_doc_and_skips_a_shorter_one() {
        let root = scratch_dir("pairs-floor");
        let corpus = root.join("all.jsonl");
        let docs = [
            ("doc-short", "z".repeat(MIN_PAIR_DOC_CHARS - 1)),
            ("doc-edge", "y".repeat(MIN_PAIR_DOC_CHARS)),
            (
                "doc-normal",
                "A question with room enough to breathe in here.\n\n\
                 An answer with the detail the question asked for, and then some more."
                    .to_string(),
            ),
        ];
        let mut text = String::new();
        for (id, doc) in &docs {
            text.push_str(
                &serde_json::json!({"id": id, "source": "t", "path": "t", "format": "jsonl", "text": doc})
                    .to_string(),
            );
            text.push('\n');
        }
        std::fs::write(&corpus, text).unwrap();

        let out = root.join("pairs/general.jsonl");
        let manifest = to_pairs(&corpus, "mini", &out, &ToPairsConfig::default()).unwrap();
        assert_eq!(manifest.docs_in, 3);
        assert_eq!(manifest.skipped_short, 1);
        assert_eq!(manifest.docs_out, 2);

        // Every emitted line is the pair-file contract.
        let lines: Vec<Value> = std::fs::read_to_string(&out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        for line in &lines {
            assert!(line["prompt"].as_str().is_some_and(|p| !p.is_empty()));
            assert!(line["completion"].as_str().is_some_and(|c| !c.is_empty()));
            assert_eq!(line["field"].as_str(), Some("mini"));
        }

        // The manifest's hash is the output file's hash, and the manifest
        // states why the cap exists.
        let digest = Sha256::digest(std::fs::read(&out).unwrap());
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(manifest.output_sha256, hex);
        assert!(manifest.note.contains("memory"), "{}", manifest.note);
        let on_disk: PairsManifest = serde_json::from_str(
            &std::fs::read_to_string(root.join("pairs/general.jsonl.manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(on_disk.docs_out, 2);
        assert_eq!(on_disk.source_corpus, corpus);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn to_pairs_subsamples_by_id_hash_not_line_order() {
        let root = scratch_dir("pairs-subsample");
        let corpus = root.join("all.jsonl");
        let ids = write_pairs_corpus(&corpus, 60, "d");
        let config = ToPairsConfig { max_docs: 10 };

        let out_a = root.join("a.jsonl");
        let manifest = to_pairs(&corpus, "mini", &out_a, &config).unwrap();
        assert_eq!(manifest.docs_in, 62); // 60 docs + short + edge64
        assert_eq!(manifest.skipped_short, 1);
        assert_eq!(manifest.docs_out, 10);

        // A rerun over the same corpus is byte-identical.
        let out_b = root.join("b.jsonl");
        to_pairs(&corpus, "mini", &out_b, &config).unwrap();
        assert_eq!(
            std::fs::read(&out_a).unwrap(),
            std::fs::read(&out_b).unwrap()
        );

        // The selection is exactly the 10 smallest sha256(id) digests — the
        // fixed test ids must NOT already be in hash order, or the
        // distinction from first-N is unproven.
        let expected = expected_selection(&ids, 10);
        assert_ne!(
            expected,
            ids.iter().take(10).cloned().collect::<Vec<_>>(),
            "the test ids are accidentally in hash order"
        );
        assert_eq!(emitted_ids(&out_a, "d"), expected);

        // Corpus growth: 20 new documents appended. The new selection is the
        // 10 smallest digests over the grown set — old picks are displaced
        // only where a new id hashes lower, never by line position.
        let mut grown = std::fs::read_to_string(&corpus).unwrap();
        let mut grown_ids = ids.clone();
        for i in 60..80 {
            let id = format!("d-{i:04}");
            grown.push_str(
                &serde_json::json!({"id": id, "source": "t", "path": "t", "format": "jsonl",
                "text": format!(
                    "Question {i} asks something with enough words to pass the floor.\n\n\
                     Answer {i} goes on at some length so the whole document clears it."
                )})
                .to_string(),
            );
            grown.push('\n');
            grown_ids.push(id);
        }
        std::fs::write(&corpus, grown).unwrap();
        to_pairs(&corpus, "mini", &out_a, &config).unwrap();
        assert_eq!(emitted_ids(&out_a, "d"), expected_selection(&grown_ids, 10));

        // And the shrunk corpus still reproduces the first selection.
        std::fs::write(&corpus, "").unwrap();
        let ids = write_pairs_corpus(&corpus, 60, "d");
        to_pairs(&corpus, "mini", &out_a, &config).unwrap();
        assert_eq!(emitted_ids(&out_a, "d"), expected_selection(&ids, 10));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
