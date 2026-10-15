//! The NINA program's Phase N1 core: a name-addressable model registry, a
//! Merkle-chained append-only log, and the MCP training server (PLAN-NINA.md
//! §1, §4, §8).
//!
//! # The chained log — tamper-evident history
//!
//! [`ChainedLog`] is an append-only JSONL file whose every line is
//! `{"seq", "ts", "prev", "hash", "payload"}`:
//!
//! - `seq` — the record number, 0-based and gapless;
//! - `ts` — Unix seconds at append time;
//! - `prev` — the previous record's `hash`; the first record points at
//!   [`GENESIS_HASH`];
//! - `hash` — sha256 hex of `prev || canonical(payload)`, where the canonical
//!   form ([`canonical_bytes`]) sorts object keys and carries no whitespace,
//!   so append and verify agree byte for byte regardless of map ordering;
//! - `payload` — the record itself.
//!
//! The API has no edit or delete: history is append-only by construction, and
//! any after-the-fact edit, insertion, deletion or reorder breaks the chain.
//! [`ChainedLog::verify`] re-derives it and names the first bad line and why.
//! (The idea is ported from the user's subnext repo, credited in
//! PLAN-NINA.md §8.)
//!
//! # The registry — derived state; the log is the truth
//!
//! [`Registry`] writes every mutation to the chained log *first* and rewrites
//! `registry.json` second. The JSON file is a read model that can be rebuilt
//! from the log at any time; the log is what an audit trusts. Promotion to
//! [`ModelStatus::Active`] requires eval evidence attached to the entry — a
//! model with no measured accuracy cannot be promoted, and the refusal says
//! so.
//!
//! # The MCP server
//!
//! [`McpServer`] speaks the Model Context Protocol's stdio subset: JSON-RPC
//! 2.0, one object per line. It implements `initialize`,
//! `notifications/initialized`, `tools/list` and `tools/call`; an unknown
//! method gets `-32601` naming what arrived, malformed JSON gets `-32700`
//! quoting what arrived. The v1 tool set is [`tool_table`]; the one
//! declared-but-unavailable tool carries its reason rather than faking a
//! result. Every `tools/call` — success or failure — is appended to
//! `logs/mcp.jsonl` (itself a chained log): the swarm's actions and the
//! audit of them are the same artifact (PLAN-NINA.md §4).
//!
//! # The scheduler — the queue is real
//!
//! [`run_queue`] drains `logs/queue.jsonl` serially (2-concurrent is
//! aspirational; serial is honest on this 12-thread box): each pending
//! `queue.add` entry runs a student training slice
//! (`student::train_language_specialist`), checkpoints content-addressed
//! into `<root>/models/<field>/<subfield>/`, gets measured on a held-out
//! fold of its own dataset ([`SchedulerConfig::holdout_percent`] — a
//! deterministic hash split, the same discipline as `nim::split_traces`
//! but over raw lines, since ids are not guaranteed), and is registered as
//! [`ModelStatus::Candidate`] with that evidence. Every step appends to the
//! queue's own chain (`queue.start` / `queue.done` / `queue.failed`), so a
//! crashed run leaves the entry pending and the next run retries it —
//! checkpoints are content-addressed, so the retry is exact. An entry whose
//! dataset is missing fails loudly, named in the [`SchedulerReport`], and
//! the queue continues.
//!
//! Before any fold is split, the entry's dataset is streamed through the
//! data-side cheat scan ([`crate::cheat::scan_snippet`], the same detector
//! and drop rule the corpus builders apply at build time): offending lines
//! are dropped and counted per class into the `queue.done` payload
//! (`cheat_scanned`, `cheat_dropped`), and a dataset the scan empties fails
//! the entry loudly, naming the path. A queued dataset that bypassed the
//! builders never reaches training.
//!
//! # The promotion gate (the RSI rule, PLAN-NINA.md §7)
//!
//! A candidate must *measurably* beat the incumbent: [`Registry::promote`]
//! still refuses entries with no eval evidence, and
//! [`Registry::promote_if_better`] refuses (loudly, and logged as
//! `promotion_rejected`) unless the candidate's accuracy exceeds the
//! accuracy the last `promote` event for that name recorded — the baseline
//! is read from the chain, which is the truth — by at least
//! [`SchedulerConfig::margin`] (default 0.0). A tie is a rejection: an
//! equal candidate has not measurably beaten the incumbent.

use crate::cheat::SnippetLanguage;
use crate::langdata::snippet_language;
use crate::student::{
    train_language_specialist, CodeLanguage, GeometricStudent, LanguageBatch,
    SpecialistTrainConfig, StudentConfig,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Default root for the registry and the chained logs (the external drive).
pub const DEFAULT_ROOT: &str = "/srv/m-sda/unifur";
/// The default registry file, under [`DEFAULT_ROOT`].
pub const DEFAULT_REGISTRY: &str = "/srv/m-sda/unifur/registry.json";
/// The chain's genesis pointer: the `prev` of the very first record.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// Schema version of the on-disk registry.
pub const REGISTRY_SCHEMA_VERSION: u32 = 1;
/// The MCP protocol revision this server speaks.
pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

const JSONRPC_PARSE_ERROR: i64 = -32700;
const JSONRPC_INVALID_REQUEST: i64 = -32600;
const JSONRPC_METHOD_NOT_FOUND: i64 = -32601;
const JSONRPC_INVALID_PARAMS: i64 = -32602;
const JSONRPC_INTERNAL_ERROR: i64 = -32603;

/// The naming rule, spelled out once so every rejection can quote it.
const NAME_RULE: &str =
    "the naming rule is field/subfield, all lowercase letters, digits and dashes \
     (e.g. 'pentest/recon', 'coding/rust')";

// ---------------------------------------------------------------------------
// Canonical encoding (what a record's hash commits to)
// ---------------------------------------------------------------------------

/// The deterministic byte encoding a record's hash commits to: object keys
/// sorted, no whitespace, strings escaped the JSON way. Self-consistent by
/// construction — append and verify both call this.
pub fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    canonical_into(value, &mut out);
    out
}

fn canonical_into(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
        Value::String(s) => escape_into(s, out),
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                canonical_into(item, out);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            out.push(b'{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                escape_into(key, out);
                out.push(b':');
                if let Some(v) = map.get(*key) {
                    canonical_into(v, out);
                }
            }
            out.push(b'}');
        }
    }
}

fn escape_into(text: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in text.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{08}' => out.extend_from_slice(b"\\b"),
            '\u{0c}' => out.extend_from_slice(b"\\f"),
            c if (c as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes());
            }
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------------
// The Merkle-chained append-only log
// ---------------------------------------------------------------------------

/// One line of the chain: the envelope every record travels in. Field order
/// is the wire order: `{"seq", "ts", "prev", "hash", "payload"}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainRecord {
    /// Record number, 0-based and gapless.
    pub seq: u64,
    /// Unix seconds at append time.
    pub ts: u64,
    /// The previous record's `hash` ([`GENESIS_HASH`] for the first).
    pub prev: String,
    /// sha256 hex of `prev || canonical(payload)`.
    pub hash: String,
    /// The record itself.
    pub payload: Value,
}

/// What a successful verification reports.
#[derive(Debug, Clone, Serialize)]
pub struct ChainReport {
    /// Records checked.
    pub lines: u64,
    /// The chain's head after the last record ([`GENESIS_HASH`] when empty).
    pub head_hash: String,
    /// Timestamp of the first record, when there is one.
    pub first_ts: Option<u64>,
    /// Timestamp of the last record, when there is one.
    pub last_ts: Option<u64>,
}

/// An append-only, hash-chained JSONL log. See the module docs for the line
/// format and the tamper-evidence argument.
pub struct ChainedLog {
    path: PathBuf,
    head_hash: String,
    len: u64,
}

impl ChainedLog {
    /// Open the log at `path`, deriving the head from its last record. A
    /// missing file is an empty chain, created on the first append. A
    /// corrupt tail is an error here — [`Self::verify`] is the tool that
    /// diagnoses it.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if !path.exists() {
            return Ok(Self {
                path,
                head_hash: GENESIS_HASH.to_string(),
                len: 0,
            });
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read chained log {}", path.display()))?;
        let mut len = 0u64;
        let mut last: Option<&str> = None;
        for line in text.lines() {
            len += 1;
            last = Some(line);
        }
        let head_hash = match last {
            None => GENESIS_HASH.to_string(),
            Some(line) => {
                let record: Value = serde_json::from_str(line).with_context(|| {
                    format!(
                        "the last line ({len}) of {} is not valid JSON; \
                         run `dblocks nina verify-log {}`",
                        path.display(),
                        path.display()
                    )
                })?;
                record
                    .get("hash")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "the last line ({len}) of {} carries no 'hash'; the tail is corrupt — \
                             run `dblocks nina verify-log {}`",
                            path.display(),
                            path.display()
                        )
                    })?
            }
        };
        Ok(Self {
            path,
            head_hash,
            len,
        })
    }

    /// The file this log appends to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The chain's head: the hash the next record will point back to.
    pub fn head(&self) -> &str {
        &self.head_hash
    }

    /// Records appended so far (also the next record's `seq`).
    pub fn len(&self) -> u64 {
        self.len
    }

    /// No records yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append one record — the only write operation the log has. Returns the
    /// record as written, including its sequence number and hash.
    pub fn append(&mut self, payload: &Value) -> Result<ChainRecord> {
        let mut preimage = self.head_hash.clone().into_bytes();
        preimage.extend_from_slice(&canonical_bytes(payload));
        let record = ChainRecord {
            seq: self.len,
            ts: now_unix(),
            prev: self.head_hash.clone(),
            hash: sha256_hex(&preimage),
            payload: payload.clone(),
        };
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("append to chained log {}", self.path.display()))?;
        let mut line = serde_json::to_string(&record).context("serialize a chain record")?;
        line.push('\n');
        file.write_all(line.as_bytes())
            .with_context(|| format!("write chained log {}", self.path.display()))?;
        file.sync_all()
            .with_context(|| format!("sync chained log {}", self.path.display()))?;
        self.head_hash = record.hash.clone();
        self.len += 1;
        Ok(record)
    }

    /// Re-derive the whole chain from `path`. A valid chain is a
    /// [`ChainReport`]; a broken one is an error naming the first bad line
    /// (1-based) and exactly why it fails — a hash mismatch (the payload was
    /// edited), a broken `prev` link (a record was edited or deleted), a
    /// `seq` gap (a record was deleted, inserted or reordered), or a line
    /// that is not valid JSON (a truncated tail).
    pub fn verify(path: &Path) -> Result<ChainReport> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read chained log {}", path.display()))?;
        let mut expected_prev = GENESIS_HASH.to_string();
        let mut expected_seq = 0u64;
        let mut first_ts = None;
        let mut last_ts = None;
        let mut head_hash = GENESIS_HASH.to_string();
        for raw in text.lines() {
            let n = expected_seq + 1; // 1-based line number, for the human reading the error
            let record: Value = serde_json::from_str(raw).map_err(|err| {
                anyhow::anyhow!(
                    "line {n}: not valid JSON ({err}) — the chain is cut or corrupted here"
                )
            })?;
            let seq = record
                .get("seq")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow::anyhow!("line {n}: missing or non-integer 'seq'"))?;
            ensure!(
                seq == expected_seq,
                "line {n}: seq is {seq}, expected {expected_seq} — a record was deleted, inserted or reordered"
            );
            let prev = record
                .get("prev")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("line {n}: missing the 'prev' link"))?;
            ensure!(
                prev == expected_prev,
                "line {n}: prev link is '{prev}', expected '{expected_prev}' — \
                 the previous record was edited or deleted"
            );
            let hash = record
                .get("hash")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("line {n}: missing 'hash'"))?;
            let payload = record
                .get("payload")
                .ok_or_else(|| anyhow::anyhow!("line {n}: missing 'payload'"))?;
            let mut preimage = prev.as_bytes().to_vec();
            preimage.extend_from_slice(&canonical_bytes(payload));
            let recomputed = sha256_hex(&preimage);
            ensure!(
                hash == recomputed,
                "line {n}: hash is '{hash}', recomputed '{recomputed}' — \
                 the payload was edited after the fact"
            );
            let ts = record
                .get("ts")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow::anyhow!("line {n}: missing or non-integer 'ts'"))?;
            if first_ts.is_none() {
                first_ts = Some(ts);
            }
            last_ts = Some(ts);
            head_hash = hash.to_string();
            expected_prev = hash.to_string();
            expected_seq += 1;
        }
        Ok(ChainReport {
            lines: expected_seq,
            head_hash,
            first_ts,
            last_ts,
        })
    }
}

/// Sum the language-mixing tallies every `queue.done` in `path` recorded.
///
/// The chain is verified first: a tally read out of an edited log would
/// calibrate the filter on evidence that no longer exists, and the whole
/// point of counting scans into a Merkle chain is that the counts cannot be
/// revised after the fact. A `queue.done` written before the mixing fields
/// existed contributes nothing — its tally defaults to empty — so a chain
/// spanning the change sums the drains that could speak.
///
/// Records that are not `queue.done`, and `queue.done` records whose
/// `cheat_mixing` is not a tally, are skipped rather than refused: the chain
/// is an audit log that accumulates other events by design.
pub fn mixing_tally_from_chain(path: &Path) -> Result<crate::cheat::MixingTally> {
    let report = ChainedLog::verify(path)?;
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read chained log {}", path.display()))?;
    let mut total = crate::cheat::MixingTally::default();
    for raw in text.lines() {
        let record: Value = serde_json::from_str(raw)
            .with_context(|| format!("parse chained log {}", path.display()))?;
        let Some(payload) = record.get("payload") else {
            continue;
        };
        if payload.get("event").and_then(Value::as_str) != Some("queue.done") {
            continue;
        }
        let Some(entry) = payload.get("cheat_mixing") else {
            continue;
        };
        let tally: crate::cheat::MixingTally = match serde_json::from_value(entry.clone()) {
            Ok(tally) => tally,
            Err(err) => anyhow::bail!(
                "record {}: 'cheat_mixing' is not a mixing tally ({err}) — the chain is intact, so \
                 this was written that way",
                report.lines
            ),
        };
        total.absorb(&tally);
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// The model registry
// ---------------------------------------------------------------------------

/// Lifecycle of a registry entry. Promotion to `Active` requires eval
/// evidence; retirement keeps the entry addressable forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// Registered, not (yet) proven.
    #[default]
    Candidate,
    /// The model the router may send traffic to.
    Active,
    /// Superseded or withdrawn; kept for provenance.
    Retired,
}

impl ModelStatus {
    /// The wire spelling (`candidate` | `active` | `retired`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Active => "active",
            Self::Retired => "retired",
        }
    }
}

/// Eval evidence attached to an entry. `accuracy: None` means "not
/// evaluated" and bars promotion to [`ModelStatus::Active`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalEvidence {
    /// Field-eval accuracy, when measured.
    #[serde(default)]
    pub accuracy: Option<f64>,
    /// Router top-1 agreement on the field, when measured.
    #[serde(default)]
    pub router_top1: Option<f64>,
    /// Evaluation date (ISO-ish free text).
    #[serde(default)]
    pub date: String,
}

/// One model in the registry: everything needed to find it, route to it and
/// audit where it came from (PLAN-NINA.md §1).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelEntry {
    /// `field/subfield`, lowercase — see [`validate_name`].
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub field: String,
    #[serde(default)]
    pub subfield: String,
    /// The MoSME box id this specialist serves (`coding:{lang}` for a coding
    /// language, `field:{field}` otherwise).
    #[serde(default)]
    pub box_id: String,
    /// Content hash of the checkpoint (its sha256).
    #[serde(default)]
    pub checkpoint_hash: String,
    #[serde(default)]
    pub status: ModelStatus,
    #[serde(default)]
    pub eval: EvalEvidence,
    /// Dataset source ids (+ licenses) the model was trained on.
    #[serde(default)]
    pub datasets: Vec<String>,
    /// The teacher that generated traces, when one did.
    #[serde(default)]
    pub teacher: Option<String>,
    /// Checkpoint size in bytes.
    #[serde(default)]
    pub bytes: u64,
}

fn schema_version() -> u32 {
    REGISTRY_SCHEMA_VERSION
}

/// The on-disk registry (`registry.json`): derived state, versioned so a
/// sidecar written by an older build still parses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryFile {
    /// Schema version; current is [`REGISTRY_SCHEMA_VERSION`].
    #[serde(default = "schema_version")]
    pub version: u32,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

impl Default for RegistryFile {
    fn default() -> Self {
        Self {
            version: REGISTRY_SCHEMA_VERSION,
            models: Vec::new(),
        }
    }
}

impl RegistryFile {
    /// Read and parse a registry file. A missing or malformed file is an
    /// error naming the path.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read registry {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse registry {}", path.display()))
    }

    /// The distinct, sorted field names the registry knows.
    pub fn fields(&self) -> Vec<String> {
        let mut fields: Vec<String> = self
            .models
            .iter()
            .map(|m| m.field.clone())
            .filter(|f| !f.is_empty())
            .collect();
        fields.sort();
        fields.dedup();
        fields
    }
}

/// Check a model name against the naming rule and return its two parts.
/// Every rejection quotes the rule.
pub fn validate_name(name: &str) -> Result<(&str, &str)> {
    let (field, subfield) = name.split_once('/').ok_or_else(|| {
        anyhow::anyhow!("invalid model name '{name}': {NAME_RULE}; this name has no '/'")
    })?;
    ensure!(
        !subfield.contains('/'),
        "invalid model name '{name}': {NAME_RULE}; it carries more than one '/'"
    );
    for (part, what) in [(field, "field"), (subfield, "subfield")] {
        ensure!(
            !part.is_empty(),
            "invalid model name '{name}': {NAME_RULE}; the {what} part is empty"
        );
        if let Some(bad) = part
            .chars()
            .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-'))
        {
            anyhow::bail!(
                "invalid model name '{name}': {NAME_RULE}; the {what} part contains '{bad}'"
            );
        }
    }
    Ok((field, subfield))
}

/// The outcome of [`Registry::promote_if_better`]. Both halves are also in
/// the chain (`promote` / `promotion_rejected`), with every number attached.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum PromotionVerdict {
    /// The candidate had no incumbent, or beat it by the margin.
    Promoted {
        name: String,
        accuracy: f64,
        /// The incumbent's accuracy at its own promotion; `None` on a first
        /// promotion.
        baseline: Option<f64>,
        margin: f64,
    },
    /// The candidate did not measurably beat the incumbent (a tie counts as
    /// not beating it).
    Rejected {
        name: String,
        candidate: f64,
        baseline: f64,
        margin: f64,
    },
}

/// The name-addressable model registry. `register`, `promote` and `retire`
/// append to the chained log (the truth) and rewrite `registry.json` (the
/// derived read model), in that order.
pub struct Registry {
    path: PathBuf,
    log_path: PathBuf,
    file: RegistryFile,
}

impl Registry {
    /// Open the registry at `path`, chained to the log at `log_path`. A
    /// missing registry file starts empty; a malformed one is an error.
    pub fn open(path: impl Into<PathBuf>, log_path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let file = if path.exists() {
            RegistryFile::load(&path)?
        } else {
            RegistryFile::default()
        };
        Ok(Self {
            path,
            log_path: log_path.into(),
            file,
        })
    }

    /// Every entry, in registration order.
    pub fn entries(&self) -> &[ModelEntry] {
        &self.file.models
    }

    /// One entry by name.
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.file.models.iter().find(|m| m.name == name)
    }

    /// Entries, optionally filtered to one field.
    pub fn list(&self, field: Option<&str>) -> Vec<&ModelEntry> {
        self.file
            .models
            .iter()
            .filter(|m| field.is_none_or(|f| m.field == f))
            .collect()
    }

    /// Register a new model: validates the name, refuses duplicates, appends
    /// the `register` record to the chained log, then rewrites
    /// `registry.json`. New entries start as [`ModelStatus::Candidate`].
    pub fn register(&mut self, entry: ModelEntry) -> Result<ChainRecord> {
        let (field, subfield) = validate_name(&entry.name)?;
        ensure!(
            entry.field == field && entry.subfield == subfield,
            "registry entry '{}' disagrees with its own field '{}' / subfield '{}' fields: \
             the name is field/subfield and must match them",
            entry.name,
            entry.field,
            entry.subfield
        );
        if let Some(existing) = self.get(&entry.name) {
            anyhow::bail!(
                "model '{}' is already registered (status {}); the registry is append-only — \
                 promote or retire the existing entry, or register under a new name",
                entry.name,
                existing.status.as_str()
            );
        }
        let record = self.log_event(json!({ "event": "register", "entry": &entry }))?;
        self.file.models.push(entry);
        self.write_file()?;
        Ok(record)
    }

    /// Flip an entry to [`ModelStatus::Active`]. Requires eval evidence: an
    /// entry with no measured accuracy cannot be promoted, and the refusal
    /// says exactly that.
    pub fn promote(&mut self, name: &str) -> Result<ChainRecord> {
        let index = self
            .file
            .models
            .iter()
            .position(|m| m.name == name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no model named '{name}' in the registry ({} entries)",
                    self.file.models.len()
                )
            })?;
        let entry = &self.file.models[index];
        ensure!(
            entry.status != ModelStatus::Active,
            "model '{name}' is already active"
        );
        let accuracy = entry.eval.accuracy;
        ensure!(
            accuracy.is_some(),
            "cannot promote '{name}' to active: the entry carries no eval accuracy — \
             promotion requires eval evidence attached (register the eval first)"
        );
        let from = entry.status.as_str();
        let record = self.log_event(
            json!({ "event": "promote", "name": name, "from": from, "to": "active", "accuracy": accuracy }),
        )?;
        if let Some(entry) = self.file.models.get_mut(index) {
            entry.status = ModelStatus::Active;
        }
        self.write_file()?;
        Ok(record)
    }

    /// Attach fresh eval evidence to an existing entry: appends an `eval`
    /// record to the chain and updates the derived entry. When the retrain
    /// produced a new checkpoint, pass its content hash and size so the
    /// registry points at the latest weights. When the retrain ran on a
    /// different dataset than the entry was registered with, pass the new
    /// dataset list — without it the entry keeps citing the v1 data and the
    /// provenance lies by omission. Updating an *Active* entry's
    /// evidence does not move the promotion baseline — the baseline is the
    /// accuracy the last `promote` event recorded, read from the chain.
    pub fn record_eval(
        &mut self,
        name: &str,
        eval: EvalEvidence,
        checkpoint_hash: Option<String>,
        bytes: Option<u64>,
        datasets: Option<Vec<String>>,
    ) -> Result<ChainRecord> {
        let index = self
            .file
            .models
            .iter()
            .position(|m| m.name == name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no model named '{name}' in the registry ({} entries)",
                    self.file.models.len()
                )
            })?;
        let record = self.log_event(json!({
            "event": "eval",
            "name": name,
            "accuracy": eval.accuracy,
            "router_top1": eval.router_top1,
            "date": eval.date,
            "checkpoint_hash": checkpoint_hash,
            "bytes": bytes,
            "datasets": datasets,
        }))?;
        if let Some(entry) = self.file.models.get_mut(index) {
            entry.eval = eval;
            if let Some(hash) = checkpoint_hash {
                entry.checkpoint_hash = hash;
            }
            if let Some(bytes) = bytes {
                entry.bytes = bytes;
            }
            if let Some(datasets) = datasets {
                entry.datasets = datasets;
            }
        }
        self.write_file()?;
        Ok(record)
    }

    /// The accuracy the most recent `promote` event for `name` recorded —
    /// the promotion baseline, read from the chain (the truth), never from
    /// the derived file. `None` when the name was never promoted.
    fn last_promoted_accuracy(&self, name: &str) -> Result<Option<f64>> {
        if !self.log_path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&self.log_path)
            .with_context(|| format!("read registry log {}", self.log_path.display()))?;
        let mut baseline = None;
        for (index, raw) in text.lines().enumerate() {
            let record: Value = serde_json::from_str(raw).with_context(|| {
                format!(
                    "registry log {} line {} is not valid JSON; verify it with \
                     `dblocks nina verify-log {}`",
                    self.log_path.display(),
                    index + 1,
                    self.log_path.display()
                )
            })?;
            let payload = record.get("payload").cloned().unwrap_or(Value::Null);
            let is_promotion = payload.get("event").and_then(Value::as_str) == Some("promote")
                && payload.get("name").and_then(Value::as_str) == Some(name);
            if is_promotion {
                baseline = payload.get("accuracy").and_then(Value::as_f64).or(baseline);
            }
        }
        Ok(baseline)
    }

    /// The RSI promotion gate (PLAN-NINA.md §7): promote only when the
    /// candidate's eval accuracy *measurably* beats the incumbent's — the
    /// accuracy the last `promote` event for this name recorded, plus
    /// `margin`. A first promotion has no incumbent and is admitted on
    /// evidence alone. A tie is rejected: an equal candidate has not
    /// measurably beaten the incumbent. Every decision is logged —
    /// `promote` on admission, `promotion_rejected` (with all three
    /// numbers) on refusal — and the verdict is returned either way.
    pub fn promote_if_better(&mut self, name: &str, margin: f64) -> Result<PromotionVerdict> {
        ensure!(
            margin.is_finite() && margin >= 0.0,
            "the promotion margin must be a non-negative finite number, got {margin}"
        );
        let index = self
            .file
            .models
            .iter()
            .position(|m| m.name == name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no model named '{name}' in the registry ({} entries)",
                    self.file.models.len()
                )
            })?;
        let entry = &self.file.models[index];
        let candidate = entry.eval.accuracy.ok_or_else(|| {
            anyhow::anyhow!(
                "cannot promote '{name}' to active: the entry carries no eval accuracy — \
                 promotion requires eval evidence attached (register the eval first)"
            )
        })?;
        ensure!(
            candidate.is_finite(),
            "the eval accuracy recorded for '{name}' is not a finite number"
        );
        let from = entry.status.as_str();
        let baseline = self.last_promoted_accuracy(name)?;
        if let Some(incumbent) = baseline {
            // `<=`, strictly: a tie has not measurably beaten the incumbent.
            if candidate <= incumbent + margin {
                self.log_event(json!({
                    "event": "promotion_rejected",
                    "name": name,
                    "candidate": candidate,
                    "baseline": incumbent,
                    "margin": margin,
                }))?;
                return Ok(PromotionVerdict::Rejected {
                    name: name.to_string(),
                    candidate,
                    baseline: incumbent,
                    margin,
                });
            }
        }
        self.log_event(json!({
            "event": "promote",
            "name": name,
            "from": from,
            "to": "active",
            "accuracy": candidate,
            "baseline": baseline,
            "margin": margin,
        }))?;
        if let Some(entry) = self.file.models.get_mut(index) {
            entry.status = ModelStatus::Active;
        }
        self.write_file()?;
        Ok(PromotionVerdict::Promoted {
            name: name.to_string(),
            accuracy: candidate,
            baseline,
            margin,
        })
    }

    /// Flip an entry to [`ModelStatus::Retired`]; the entry stays
    /// addressable for provenance.
    pub fn retire(&mut self, name: &str) -> Result<ChainRecord> {
        let index = self
            .file
            .models
            .iter()
            .position(|m| m.name == name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no model named '{name}' in the registry ({} entries)",
                    self.file.models.len()
                )
            })?;
        let entry = &self.file.models[index];
        ensure!(
            entry.status != ModelStatus::Retired,
            "model '{name}' is already retired"
        );
        let from = entry.status.as_str();
        let record = self
            .log_event(json!({ "event": "retire", "name": name, "from": from, "to": "retired" }))?;
        if let Some(entry) = self.file.models.get_mut(index) {
            entry.status = ModelStatus::Retired;
        }
        self.write_file()?;
        Ok(record)
    }

    fn log_event(&mut self, payload: Value) -> Result<ChainRecord> {
        let mut log = ChainedLog::open(&self.log_path)?;
        log.append(&payload)
    }

    fn write_file(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(&self.file).context("serialize the registry")?;
        std::fs::write(&self.path, format!("{text}\n"))
            .with_context(|| format!("write registry {}", self.path.display()))
    }
}

// ---------------------------------------------------------------------------
// Routing (the probe router, with an honest fallback)
// ---------------------------------------------------------------------------

/// The inference backend the router, eval and scheduler paths run on.
type EvalBackend = burn::backend::NdArray<f32>;
/// The autodiff backend the scheduler trains on.
type TrainBackend = burn::backend::Autodiff<EvalBackend>;

/// Recover the `StudentConfig` a checkpoint was trained with: the
/// training-state sidecar first; a deployed checkpoint travels without one,
/// so next the `student_config` recorded in the `config.json` deployment
/// manifest beside it; the default config only when neither records one.
/// Evaluating at the wrong context truncates every trace differently than
/// training did, so a silent default here is a wrong number, not a safe one.
fn student_config_for_checkpoint(path: &Path) -> Result<StudentConfig> {
    if let Some(state) = crate::checkpoint::TrainState::for_model(path)? {
        return match state.config.get("student_config") {
            Some(value) => serde_json::from_value(value.clone()).with_context(|| {
                format!(
                    "parse the student config recorded beside {}",
                    path.display()
                )
            }),
            None => Ok(StudentConfig::default()),
        };
    }
    let Some(dir) = path.parent() else {
        return Ok(StudentConfig::default());
    };
    let manifest = dir.join("config.json");
    let text = match std::fs::read_to_string(&manifest) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(StudentConfig::default());
        }
        Err(err) => anyhow::bail!("read deployment manifest {}: {err}", manifest.display()),
    };
    let value: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse deployment manifest {}", manifest.display()))?;
    match value.get("student_config") {
        Some(config) => serde_json::from_value(config.clone()).with_context(|| {
            format!(
                "parse the student config in deployment manifest {}",
                manifest.display()
            )
        }),
        None => Ok(StudentConfig::default()),
    }
}

/// Route a prompt. With a student checkpoint present this runs the probe
/// router (`student::GeometricStudent::route_languages`) and reports the
/// per-language box traffic; without one it returns the registry's field
/// list with a loud note that no trained router is loaded — an honest
/// "I don't know" rather than a chance-level pick presented as a route.
pub fn route_prompt(
    prompt: &str,
    checkpoint: Option<&Path>,
    registry_fields: &[String],
) -> Result<Value> {
    use crate::tokenizer::ByteTokenizer;
    use burn::tensor::{Int, Tensor};

    type Eval = EvalBackend;

    let tokens = ByteTokenizer::new().encode_document(prompt);
    let n = tokens.len();
    let mut found: Option<PathBuf> = None;
    if let Some(path) = checkpoint {
        if path
            .try_exists()
            .with_context(|| format!("stat {}", path.display()))?
        {
            found = Some(path.to_path_buf());
        }
    }
    let Some(path) = found else {
        let note = match checkpoint {
            Some(path) => format!(
                "no trained router is loaded: the checkpoint {} does not exist",
                path.display()
            ),
            None => "no trained router is loaded: no student checkpoint was found".to_string(),
        };
        return Ok(json!({
            "router": Value::Null,
            "note": format!(
                "{note}; routing picks need a trained student checkpoint \
                 (`dblocks student train`, default output <root>/student/). \
                 The registry's known fields follow."
            ),
            "prompt_tokens": n,
            "fields": registry_fields,
        }));
    };

    ensure!(
        !tokens.is_empty(),
        "the prompt tokenizes to zero tokens — nothing to route"
    );
    let config = student_config_for_checkpoint(&path)?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <Eval as burn::tensor::backend::Backend>::seed(&device, 0);
    let model = GeometricStudent::<Eval>::new(&config, &device)?;
    crate::tensor_ext::force_initialization(&model);
    let model = crate::checkpoint::load::<Eval, _>(model, &path, &device)?;
    let tensor = Tensor::<Eval, 1, Int>::from_ints(
        tokens
            .iter()
            .map(|t| i64::from(*t))
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    )
    .reshape([1, n]);
    let routing = model.route_languages(tensor)?;
    let mut counts = vec![0usize; CodeLanguage::KNOWN.len()];
    for top in &routing.top_language {
        if let Some(slot) = counts.get_mut(top.index()) {
            *slot += 1;
        }
    }
    // The box names are the config's, not a constant: the coding prefix in
    // `coding_boxes` order, then the field boxes in `field_boxes` order.
    let box_id_of = |i: usize| -> String {
        if let Some(name) = config.coding_boxes.get(i) {
            format!("coding:{name}")
        } else {
            match config.field_boxes.get(i - config.coding_boxes.len()) {
                Some(name) => format!("field:{name}"),
                None => format!("box:{i}"),
            }
        }
    };
    let traffic: Vec<Value> = routing
        .traffic
        .iter()
        .enumerate()
        .map(|(i, gate)| {
            let box_id = box_id_of(i);
            let top1 = config
                .coding_boxes
                .get(i)
                .and_then(|name| CodeLanguage::parse(name).ok())
                .map_or(0, |language| {
                    counts.get(language.index()).copied().unwrap_or(0)
                });
            json!({
                "box_id": box_id,
                "traffic": gate,
                "top1_tokens": top1,
            })
        })
        .collect();
    let top = routing
        .traffic
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| box_id_of(i));
    Ok(json!({
        "router": "probe",
        "checkpoint": path.display().to_string(),
        "prompt_tokens": n,
        "top": top,
        "traffic": traffic,
    }))
}

// ---------------------------------------------------------------------------
// The scheduler: queue consumption, training slices, eval, registration
// ---------------------------------------------------------------------------

/// How the scheduler runs: the student architecture, the training knobs,
/// and the eval/promotion policy. Serde-defaulted throughout, so a config
/// written by an older build still parses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// The student architecture every queued slice trains. A test or a smoke
    /// run passes a tiny config; production passes the real one.
    #[serde(default)]
    pub student: StudentConfig,
    /// Micro-batch size per step.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// AdamW learning rate.
    #[serde(default = "default_lr")]
    pub lr: f64,
    /// Seed for model init, the batch shuffle and the data order — a queued
    /// entry re-run after a crash is exact.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Percentage of the dataset's lines held out for eval (by hash bucket,
    /// so the split is stable across runs and machines). When positive, at
    /// least one line is held out whenever the dataset has any.
    #[serde(default = "default_holdout_percent")]
    pub holdout_percent: u32,
    /// The margin [`Registry::promote_if_better`] demands over the
    /// incumbent's accuracy. 0.0 means any strict improvement; a tie is
    /// never an improvement.
    #[serde(default)]
    pub margin: f64,
    /// Print every this many steps; 0 is quiet (the GUI polls the logs).
    #[serde(default)]
    pub log_every: usize,
    /// Weight on the router-which-knows auxiliary; 0 disables it exactly.
    #[serde(default = "default_router_aux_weight")]
    pub router_aux_weight: f64,
    /// Non-coding fields that get their own routing box (`field:{name}`) in
    /// every student this scheduler trains — merged into
    /// [`SchedulerConfig::student`]'s [`StudentConfig::field_boxes`] at run
    /// time, so a queued `pentest/recon` slice trains with its box supervised
    /// instead of flying blind. Empty by default, so a config written before
    /// field boxes existed parses unchanged.
    #[serde(default)]
    pub field_boxes: Vec<String>,
    /// Micro-batch size the trainer accumulates over
    /// ([`SpecialistTrainConfig::micro_batch`]). 0 (the default) means the
    /// direct path — one forward/backward per step — so a config written
    /// before this field existed runs bit-identically. GPU shapes set it to
    /// keep peak activation memory at one micro-batch while the step still
    /// sees `batch_size` examples.
    #[serde(default)]
    pub micro_batch: usize,
    /// Train with Burn's `BalancedCheckpointing` autodiff strategy (ops the
    /// backend marks memory-bound are recomputed in the backward instead of
    /// resident). Only meaningful to a caller that instantiates the backend —
    /// `queue_runner --backend wgpu` reads it; the default CPU tick ignores
    /// it. Default false, so existing configs keep their backend semantics.
    #[serde(default)]
    pub gpu_checkpointing: bool,
    /// A [`crate::cheat::MixingPolicy`] profile to run the language-mixing
    /// check under, as written by `dblocks cheat adapt`. `None` runs the
    /// uncalibrated defaults, so a config written before calibration existed
    /// parses and behaves unchanged.
    #[serde(default)]
    pub mixing_policy: Option<PathBuf>,
    /// A trained [`crate::mlcheat::MlCheatModel`] to stack behind the pattern
    /// scan. `None` (the default) leaves the pattern scan deciding alone. The
    /// model generalizes the pattern scanner's own verdicts to snippets whose
    /// patterns do not fire; it cannot discover a class the patterns have
    /// never seen, and it never overrides a pattern finding.
    #[serde(default)]
    pub ml_cheat_model: Option<PathBuf>,
    /// Score at or above which a trace the pattern scan passed is dropped
    /// anyway. Only consulted when [`Self::ml_cheat_model`] is set. The default
    /// is deliberately high: this layer is trained on roughly fifty labeled
    /// samples and is a second opinion, not a verdict.
    #[serde(default = "default_ml_cheat_threshold")]
    pub ml_cheat_threshold: f64,
}

fn default_ml_cheat_threshold() -> f64 {
    0.9
}

fn default_batch_size() -> usize {
    4
}
fn default_lr() -> f64 {
    3e-4
}
fn default_seed() -> u64 {
    42
}
fn default_holdout_percent() -> u32 {
    10
}
fn default_router_aux_weight() -> f64 {
    0.1
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            student: StudentConfig::default(),
            batch_size: default_batch_size(),
            lr: default_lr(),
            seed: default_seed(),
            holdout_percent: default_holdout_percent(),
            margin: 0.0,
            log_every: 0,
            router_aux_weight: default_router_aux_weight(),
            field_boxes: Vec::new(),
            micro_batch: 0,
            gpu_checkpointing: false,
            mixing_policy: None,
            ml_cheat_model: None,
            ml_cheat_threshold: default_ml_cheat_threshold(),
        }
    }
}

impl SchedulerConfig {
    /// Validate the knobs that would otherwise fail deep inside a run.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.holdout_percent <= 100,
            "holdout_percent must be in 0..=100, got {}",
            self.holdout_percent
        );
        ensure!(self.batch_size > 0, "batch_size must be positive");
        ensure!(
            self.lr.is_finite() && self.lr > 0.0,
            "lr must be a positive finite number, got {}",
            self.lr
        );
        ensure!(
            self.margin.is_finite() && self.margin >= 0.0,
            "margin must be a non-negative finite number, got {}",
            self.margin
        );
        crate::expert_index::validate_field_box_names(&self.field_boxes)?;
        self.student.validate()
    }

    /// The student config a run actually trains: [`Self::student`] with the
    /// scheduler-level [`Self::field_boxes`] merged in (union, existing
    /// entries first), so the field's routing box exists in the model and in
    /// the checkpoint sidecar its training state writes.
    fn student_for_run(&self) -> StudentConfig {
        let mut student = self.student.clone();
        for field in &self.field_boxes {
            if !student.field_boxes.contains(field) {
                student.field_boxes.push(field.clone());
            }
        }
        student
    }

    /// The language-mixing profile this run scans under.
    ///
    /// Falls back to [`crate::cheat::MixingPolicy::default`] when
    /// [`Self::mixing_policy`] is unset, so a config written before
    /// calibration existed runs the rule exactly as it always did. A path that
    /// is set but unreadable is an error rather than a silent fallback: a run
    /// that quietly ignored the profile it was pointed at would produce a
    /// corpus filtered by different rules than its log claims.
    pub fn mixing_policy(&self) -> Result<crate::cheat::MixingPolicy> {
        match &self.mixing_policy {
            Some(path) => crate::cheat::MixingPolicy::load(path),
            None => Ok(crate::cheat::MixingPolicy::default()),
        }
    }

    /// The learned second opinion, when one is configured: the model and the
    /// score at or above which a trace the pattern scan *passed* is dropped
    /// anyway.
    ///
    /// `None` when [`Self::ml_cheat_model`] is unset, which is the default —
    /// the pattern scan alone decides, as it always has. An unreadable model
    /// path is an error on the same reasoning as [`Self::mixing_policy`]: a
    /// scan that quietly fell back to patterns would log counts that say
    /// nothing about which rules produced them.
    pub fn ml_cheat(&self) -> Result<Option<(crate::mlcheat::MlCheatModel, f64)>> {
        let Some(path) = &self.ml_cheat_model else {
            return Ok(None);
        };
        let model = crate::mlcheat::MlCheatModel::load(path)?;
        Ok(Some((model, self.ml_cheat_threshold)))
    }
}

/// What the pre-training cheat gate did to one queue dataset.
///
/// A re-export, not a second type: the queue scanner and
/// [`crate::student::LanguageBatch::load`] run the same
/// [`crate::cheat::CheatGate`], and two summary structs for one gate would be
/// two places for the stacking order to drift.
pub use crate::cheat::ScanSummary as CheatScanSummary;

/// What happened to one queue entry.
#[derive(Debug, Clone, Serialize)]
pub struct QueueEntryOutcome {
    /// The entry's `seq` in the queue chain.
    pub queue_seq: u64,
    /// The specialist name (`field/subfield`), or the raw payload when the
    /// entry itself was unreadable.
    pub name: String,
    /// `true` when the slice trained, evaluated and registered.
    pub ok: bool,
    /// The content-addressed checkpoint, when training produced one.
    pub checkpoint: Option<PathBuf>,
    /// sha256 of the checkpoint file.
    pub checkpoint_hash: Option<String>,
    /// Held-out completion-token accuracy (see [`eval_model`]).
    pub accuracy: Option<f64>,
    /// Held-out router top-1 (see [`eval_model`]); `None` when the fold
    /// names no routing box (a field the config does not route).
    pub router_top1: Option<f64>,
    /// Optimizer steps actually taken.
    pub steps_taken: usize,
    /// Fold sizes of the deterministic split the run used.
    pub train_lines: usize,
    /// Fold sizes of the deterministic split the run used.
    pub holdout_lines: usize,
    /// What the pre-training cheat scan removed from the entry's dataset
    /// before the split. Zeroed on a failed entry.
    pub cheat_scan: CheatScanSummary,
    /// The autodiff backend type this entry trained on, as
    /// `std::any::type_name`. Recorded because the two are not
    /// interchangeable: reduction orders differ, so an accuracy produced on
    /// `Autodiff<Wgpu<f32>>` and one produced on `Autodiff<NdArray<f32>>`
    /// are measurements of different computations, and a registry that
    /// cannot say which produced a number cannot be compared across entries.
    pub backend: String,
    /// Why the entry failed, when it did. The queue continues past it.
    pub error: Option<String>,
}

/// What a [`run_queue`] pass did.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerReport {
    /// Entries attempted (trained or failed).
    pub drained: usize,
    pub trained: usize,
    pub failed: usize,
    /// One per drained entry, in queue order.
    pub outcomes: Vec<QueueEntryOutcome>,
    pub elapsed_secs: f64,
}

/// Today's date as `YYYY-MM-DD` (Howard Hinnant's civil-from-days), for
/// [`EvalEvidence::date`].
fn iso_date(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}")
}

/// The fold bucket of a raw dataset line, in `0..100`: sha256 of the line,
/// first eight bytes big-endian, mod 100. The same discipline as
/// `nim::split_traces`, but over the raw line because queue datasets do not
/// guarantee an `id` field.
fn fold_bucket(line: &str) -> u32 {
    let digest = Sha256::digest(line.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_be_bytes(bytes) % 100) as u32
}

/// The language a queue entry trains, when its name is a coding box. The
/// probe router supervises coding boxes and configured field boxes
/// ([`SchedulerConfig::field_boxes`]); any other field trains unfiltered
/// with the routing auxiliary off.
fn entry_language(field: &str, subfield: &str) -> Option<CodeLanguage> {
    if field == "coding" {
        CodeLanguage::parse(subfield).ok()
    } else {
        None
    }
}

/// Where a dataset split lands while a run uses it.
struct SplitFolds {
    train: PathBuf,
    holdout: PathBuf,
    train_lines: usize,
    holdout_lines: usize,
}

/// Split a traces file into train/holdout fold files under `work_dir`,
/// deterministically. Only the lines the specialist will see are split:
/// with a coding target that is the lines whose `language` parses to it
/// (mirroring [`LanguageBatch::load`]'s filter); without one it is every
/// line. A malformed line always lands in the train fold, so the loader —
/// not this split — names it. When `holdout_percent > 0` and the hash split
/// would leave a fold empty, the extreme-bucket line is moved over: a tiny
/// dataset still gets both folds, and the rule is deterministic.
fn split_dataset(
    path: &Path,
    language: Option<CodeLanguage>,
    holdout_percent: u32,
    work_dir: &Path,
) -> Result<SplitFolds> {
    ensure!(
        holdout_percent <= 100,
        "holdout_percent must be in 0..=100, got {holdout_percent}"
    );
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read dataset {}", path.display()))?;
    let mut kept: Vec<&str> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let matches = match (language, serde_json::from_str::<Value>(line).ok()) {
            (Some(want), Some(value)) => {
                let raw_language = value
                    .get("language")
                    .or_else(|| value.get("lang"))
                    .and_then(Value::as_str)
                    .map(str::trim);
                raw_language.and_then(|raw| CodeLanguage::parse(raw).ok()) == Some(want)
            }
            // A malformed line goes to the train fold so the loader names it.
            (Some(_), None) => true,
            (None, _) => true,
        };
        if matches {
            kept.push(line);
        }
    }
    ensure!(
        !kept.is_empty(),
        "{} holds no lines for {} — nothing to train or evaluate",
        path.display(),
        language.map_or("an unfiltered run".to_string(), |l| format!(
            "'{}'",
            l.name()
        ))
    );
    let mut train: Vec<&str> = Vec::new();
    let mut holdout: Vec<&str> = Vec::new();
    for line in &kept {
        if fold_bucket(line) < holdout_percent {
            holdout.push(line);
        } else {
            train.push(line);
        }
    }
    if holdout_percent > 0 && holdout.is_empty() {
        // The smallest-bucket line moves to the holdout fold.
        let index = kept
            .iter()
            .enumerate()
            .min_by_key(|(_, line)| fold_bucket(line))
            .map(|(i, _)| i)
            .ok_or_else(|| anyhow::anyhow!("the dataset kept no lines to split"))?;
        holdout.push(kept[index]);
        train = kept
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, line)| *line)
            .collect();
    }
    if train.is_empty() {
        // The largest-bucket line moves back to the train fold.
        let index = kept
            .iter()
            .enumerate()
            .max_by_key(|(_, line)| fold_bucket(line))
            .map(|(i, _)| i)
            .ok_or_else(|| anyhow::anyhow!("the dataset kept no lines to split"))?;
        train.push(kept[index]);
        holdout = kept
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, line)| *line)
            .collect();
    }
    std::fs::create_dir_all(work_dir).with_context(|| format!("create {}", work_dir.display()))?;
    let train_path = work_dir.join("train-fold.jsonl");
    let holdout_path = work_dir.join("holdout-fold.jsonl");
    std::fs::write(&train_path, format!("{}\n", train.join("\n")))
        .with_context(|| format!("write {}", train_path.display()))?;
    std::fs::write(&holdout_path, format!("{}\n", holdout.join("\n")))
        .with_context(|| format!("write {}", holdout_path.display()))?;
    Ok(SplitFolds {
        train: train_path,
        holdout: holdout_path,
        train_lines: train.len(),
        holdout_lines: holdout.len(),
    })
}

/// Stream a queue dataset through the data-side cheat scan *before* the fold
/// split: the same [`scan_snippet`] detector and the same drop rule as
/// [`crate::cheat::filter_traces`] — any finding at quarantine grade
/// ([`Severity::Evasion`]) or above drops the line — which the corpus
/// builders apply at build time. A queued dataset that bypassed the builders
/// is exactly what this stage exists to catch.
///
/// The scanned text is the completion — the training signal, and the same
/// unit [`crate::langdata`] filters at build time. The prompt is not scanned:
/// real task statements enumerate the forbidden idioms ("no `todo!()`")
/// and show fenced buggy code, and a scanner reading instructions as code
/// false-positives on every one of them. The one exception is brokenness:
/// the queue contract trains prompt + completion as a single unit, so a
/// completion that is unbalanced on its own but balanced behind its prompt
/// (a suffix fragment) is kept — a [`CheatClass::DataBroken`]-only verdict
/// is re-checked against the joined unit before the line is dropped.
/// Lines the entry's language filter excludes are not scanned: they
/// never reach the folds, and the wrong language's patterns would only
/// false-positive on them.
///
/// Surviving lines are written verbatim to `work_dir/cheat-cleaned.jsonl`,
/// which is what [`split_dataset`] then splits: the folds are built from
/// clean lines only, and since the fold bucket is a hash of the line itself,
/// a surviving line's fold assignment is unchanged by the lines removed
/// around it.
///
/// Loud by the same contract as the loader it fronts: a non-JSON line or a
/// matching line with no prompt/completion pair is an error naming the
/// dataset and the line number, and a dataset the scan empties entirely is
/// an error naming the dataset and the classes — training on an empty
/// dataset is refused.
fn scan_dataset_cheats(
    path: &Path,
    language: Option<CodeLanguage>,
    policy: &crate::cheat::MixingPolicy,
    learned: Option<&(crate::mlcheat::MlCheatModel, f64)>,
    work_dir: &Path,
) -> Result<(PathBuf, CheatScanSummary)> {
    let file =
        std::fs::File::open(path).with_context(|| format!("read dataset {}", path.display()))?;
    std::fs::create_dir_all(work_dir).with_context(|| format!("create {}", work_dir.display()))?;
    let cleaned_path = work_dir.join("cheat-cleaned.jsonl");
    let mut cleaned = BufWriter::new(
        std::fs::File::create(&cleaned_path)
            .with_context(|| format!("create {}", cleaned_path.display()))?,
    );
    let mut summary = CheatScanSummary::default();
    // Distinct *completion* text among the rows that survive the gate. Kept as
    // a set of whole strings: the degeneracy this catches is verbatim reuse,
    // and a cheap normalisation here would miss the case that matters (the same
    // answer with different whitespace still teaches one answer).
    let mut distinct_completions: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    // The one gate every text training path in this crate shares. Building it
    // here from the same policy and model the other loaders use is what keeps
    // the stacking order — patterns first, learned score only where they are
    // silent — from being re-derived per call site.
    let gate = crate::cheat::CheatGate {
        policy: policy.clone(),
        learned: learned.cloned(),
    };
    for (index, raw) in BufReader::new(file).lines().enumerate() {
        let raw = raw.with_context(|| format!("read dataset {}", path.display()))?;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_no = index + 1;
        let value: Value = serde_json::from_str(line).with_context(|| {
            format!(
                "{} line {line_no} is not valid JSON; the pre-training cheat scan reads \
                 the traces JSONL format (prompt/completion or instruction/response)",
                path.display()
            )
        })?;
        let get = |key: &str| value.get(key).and_then(Value::as_str);
        let parsed_language = get("language")
            .or_else(|| get("lang"))
            .map(str::trim)
            .and_then(|raw| CodeLanguage::parse(raw).ok());
        // The filter split_dataset will apply: a line the fold split would
        // skip is not this run's data, so it is not this run's scan either.
        if language.is_some() && parsed_language != language {
            continue;
        }
        let (Some(prompt), Some(completion)) = (
            get("prompt").or_else(|| get("instruction")),
            get("completion").or_else(|| get("response")),
        ) else {
            anyhow::bail!(
                "{} line {line_no} has no prompt/completion (or instruction/response) pair \
                 for the pre-training cheat scan to check",
                path.display()
            );
        };
        let snippet_lang = match language.or(parsed_language) {
            Some(lang) => snippet_language(lang),
            None => SnippetLanguage::Other,
        };
        // The completion is the training signal, so the completion is what
        // is gated — the same unit `langdata` filters at build time. The
        // prompt is passed for the fragment rule and is not itself gated:
        // real task statements enumerate the forbidden idioms ("no
        // `todo!()`, no `# type: ignore`") and show fenced buggy code, and a
        // scanner reading instructions as code false-positives on every one.
        let offending = gate.inspect(Some(prompt), completion, snippet_lang, &mut summary);
        summary.observe(&offending);
        if offending.is_empty() {
            distinct_completions.insert(completion.to_string());
            cleaned
                .write_all(line.as_bytes())
                .with_context(|| format!("write {}", cleaned_path.display()))?;
            cleaned
                .write_all(b"\n")
                .with_context(|| format!("write {}", cleaned_path.display()))?;
        }
    }
    cleaned
        .flush()
        .with_context(|| format!("write {}", cleaned_path.display()))?;
    if summary.scanned > 0 && summary.kept == 0 {
        return Err(summary.refusal(path));
    }
    // A dataset that is mostly one answer repeated is not a dataset, and no
    // cheat class describes it: every row can be individually clean. The
    // offline pipeline produced exactly this for the three coding languages
    // magicoder does not cover — 3000 rows over the template bank's four
    // references — and the specialist registered 0.848 by memorising four
    // strings against a held-out fold drawn from the same four.
    //
    // The ratio is the discriminator, not a count: real corpora sit at
    // 0.95–1.00 distinct completions, and a floor has to clear a *duplicate*
    // run without clearing a *degenerate* one. 0.10 is the line, and it is
    // reported on refusal so the operator can see how far off the data was
    // rather than being told only "refused".
    if summary.kept > 0 {
        let distinct = distinct_completions.len();
        let ratio = distinct as f64 / summary.kept as f64;
        if ratio < MIN_DISTINCT_COMPLETION_RATIO {
            return Err(anyhow::anyhow!(
                "{}: {distinct} of {} kept rows carry distinct completions \
                 ({:.2}% distinct, floor {:.0}%) — this is one answer repeated, not a \
                 dataset, and a model trained on it memorises the repetition rather than \
                 learning the language. Retrain on a corpus with real per-row answers, or \
                 accept a specialist trained on the {distinct} real example(s) this data \
                 actually holds.",
                path.display(),
                summary.kept,
                100.0 * ratio,
                100.0 * MIN_DISTINCT_COMPLETION_RATIO,
            ));
        }
    }
    Ok((cleaned_path, summary))
}

/// The floor on distinct completions, as a fraction of kept rows.
///
/// Real corpora sit at 0.95–1.00; the template-bank fallback that produced a
/// fake 0.848 sat at 0.0013. Anything between those is a duplicate run, not a
/// degenerate one, and clearing it would reject data that trains fine.
const MIN_DISTINCT_COMPLETION_RATIO: f64 = 0.10;

/// Held-out eval, defined exactly:
///
/// - **accuracy**: over every held-out example, at every position `t` whose
///   successor is a supervised completion token (the [`LanguageBatch`] mask
///   is positive at `t + 1`), whether the model's argmax at `t` equals the
///   true token at `t + 1`. `correct / supervised`, over the whole fold.
/// - **router_top1**: over the *supervised* (completion) tokens of held-out
///   examples that name a routing box — a known language, or a `"field"` the
///   model's config lists in [`StudentConfig::field_boxes`] — the fraction
///   whose probe-router top-1 box is that box: the same quantity the trainer
///   reports as `target_top1` (chance is 1 over the box count). `None` when
///   no held-out example names a box (a field the config does not route).
///
/// The pass is a plain forward — no sampling, no RNG — so the same
/// checkpoint on the same fold on the same backend gives bit-identical
/// numbers. Generic over the backend: the scheduler's CPU path evaluates on
/// `EvalBackend`; a [`train_tick_with`] run on another backend evaluates
/// the trained weights on that backend's inference twin.
pub fn eval_model<B: burn::tensor::backend::Backend<FloatElem = f32>>(
    model: &GeometricStudent<B>,
    holdout: &LanguageBatch,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Result<(f64, Option<f64>)> {
    use burn::tensor::{Int, Tensor};

    /// Examples per forward pass; bounds the eval's memory on big folds.
    const EVAL_CHUNK: usize = 32;
    /// Token budget per eval forward: `EVAL_CHUNK` at context 128, shrinking
    /// as the context grows — a ctx-1024 run chunks to 4, a ctx-4096 run to
    /// a single row. The budget is sized to the largest activation one
    /// training micro-batch already allocates at that context (a dense
    /// attention layer's `[chunk, heads, n, n]` score transient), because by
    /// eval time a GPU run's cubecl pool is reserved to the training peak and
    /// a bigger single allocation is what tips it over: measured on the
    /// RTX 3060 (12 GB) drain at ctx 4096, a chunk of 8 OOM'd the device
    /// mid-eval and corrupted the pool while a chunk of 1 runs clean.
    /// Chunking changes no numbers: the forward is deterministic and per-row.
    const EVAL_TOKENS: usize = 4096;

    let context = holdout.context();
    let eval_chunk = (EVAL_TOKENS / context).clamp(1, EVAL_CHUNK);
    let mut correct = 0u64;
    let mut supervised = 0u64;
    let mut router_correct = 0u64;
    let mut router_supervised = 0u64;
    for chunk_start in (0..holdout.len()).step_by(eval_chunk) {
        let chunk_len = eval_chunk.min(holdout.len() - chunk_start);
        let mut flat = Vec::with_capacity(chunk_len * context);
        for i in chunk_start..chunk_start + chunk_len {
            flat.extend(holdout.tokens(i).iter().map(|t| i64::from(*t)));
        }
        let tokens =
            Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device).reshape([chunk_len, context]);
        let out = model.forward(tokens)?;
        let preds: Vec<i64> = out
            .logits
            .argmax(2)
            .reshape([chunk_len * context])
            .into_data()
            .convert::<i64>()
            .iter()
            .collect();
        assert_eq!(
            preds.len(),
            chunk_len * context,
            "argmax must return one prediction per position"
        );
        for (offset, i) in (chunk_start..chunk_start + chunk_len).enumerate() {
            let mask = holdout.mask(i);
            let example = holdout.tokens(i);
            // Position t predicts token t + 1; a successor counts when the
            // mask supervises it.
            for (t, (target, m)) in example.iter().zip(mask.iter()).enumerate().skip(1) {
                if *m > 0.0 {
                    supervised += 1;
                    if preds[offset * context + t - 1] == i64::from(*target) {
                        correct += 1;
                    }
                }
            }
            let target = model
                .config()
                .box_target(holdout.language_of(i), holdout.field_of(i));
            if let Some(target) = target {
                let want = target.box_index();
                for (t, m) in mask.iter().enumerate() {
                    if *m > 0.0 {
                        router_supervised += 1;
                        if out.routing.top_box.get(offset * context + t) == Some(&want) {
                            router_correct += 1;
                        }
                    }
                }
            }
        }
    }
    ensure!(
        supervised > 0,
        "the held-out fold of {} has no supervised completion tokens",
        holdout.path().display()
    );
    let accuracy = correct as f64 / supervised as f64;
    let router_top1 =
        (router_supervised > 0).then(|| router_correct as f64 / router_supervised as f64);
    Ok((accuracy, router_top1))
}

/// Recompute an entry's evidence from a checkpoint on disk: load the model,
/// split the dataset the same deterministic way, run [`eval_model`].
fn eval_from_checkpoint(
    checkpoint: &Path,
    dataset: &Path,
    language: Option<CodeLanguage>,
    config: &SchedulerConfig,
) -> Result<EvalEvidence> {
    let device: <EvalBackend as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <EvalBackend as burn::tensor::backend::Backend>::seed(&device, 0);
    let student_config = student_config_for_checkpoint(checkpoint)?;
    let model = GeometricStudent::<EvalBackend>::new(&student_config, &device)?;
    crate::tensor_ext::force_initialization(&model);
    let model = crate::checkpoint::load::<EvalBackend, _>(model, checkpoint, &device)?;
    let work_dir = scratch_dir("eval");
    // The same pre-training scan run_entry applied, so the re-derived
    // numbers come from the same clean folds the run trained on.
    let policy = config.mixing_policy()?;
    let learned = config.ml_cheat()?;
    let (cleaned, _) =
        scan_dataset_cheats(dataset, language, &policy, learned.as_ref(), &work_dir)?;
    let folds = split_dataset(&cleaned, language, config.holdout_percent, &work_dir)?;
    let holdout = LanguageBatch::load(
        &folds.holdout,
        language,
        student_config.context,
        config.seed,
    )?;
    let (accuracy, router_top1) = eval_model(&model, &holdout, &device)?;
    // Scratch files: a failed cleanup leaves temp files behind, nothing more.
    drop(std::fs::remove_dir_all(&work_dir));
    Ok(EvalEvidence {
        accuracy: Some(accuracy),
        router_top1,
        date: iso_date(now_unix()),
    })
}

/// A unique scratch directory for fold files (pid + counter; the crate's
/// temp-file discipline).
fn scratch_dir(stem: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("dblocks-nina-{stem}-{}-{n}", std::process::id()))
}

/// Pending queue entries as `(seq, payload)`: every `queue.add` whose seq no
/// `queue.done` or `queue.failed` has settled. A crashed run never wrote
/// either, so its entry is pending again — the retry is the resume.
pub fn pending_queue_entries(paths: &NinaPaths) -> Result<Vec<(u64, Value)>> {
    let path = paths.queue_log();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read queue {}", path.display()))?;
    let mut adds: Vec<(u64, Value)> = Vec::new();
    let mut settled: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for (index, raw) in text.lines().enumerate() {
        let record: Value = serde_json::from_str(raw).with_context(|| {
            format!(
                "queue {} line {} is not valid JSON; verify it with `dblocks nina verify-log {}`",
                path.display(),
                index + 1,
                path.display()
            )
        })?;
        let payload = record.get("payload").cloned().unwrap_or(Value::Null);
        match payload.get("event").and_then(Value::as_str) {
            Some("queue.add") => {
                let seq = record
                    .get("seq")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow::anyhow!("queue line {} has no seq", index + 1))?;
                adds.push((seq, payload));
            }
            Some("queue.done" | "queue.failed") => {
                if let Some(seq) = payload.get("queue_seq").and_then(Value::as_u64) {
                    settled.insert(seq);
                }
            }
            _ => {}
        }
    }
    Ok(adds
        .into_iter()
        .filter(|(seq, _)| !settled.contains(seq))
        .collect())
}

/// Write the deployment sidecar (`config.json`) a model directory is served
/// from: the checkpoint file name, the architecture it was trained with, and
/// enough provenance to say where both came from.
///
/// Rewritten on every successful training run rather than once by the trace
/// pipeline, because the checkpoint it names is content-addressed and a
/// retrain replaces it. See the call site for what went wrong when it did not.
fn write_deployment_sidecar(
    model_dir: &Path,
    name: &str,
    checkpoint: &Path,
    config: &SchedulerConfig,
) -> Result<()> {
    let file_name = checkpoint
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "checkpoint {} has no file name to record in the sidecar",
                checkpoint.display()
            )
        })?
        .to_string();
    let manifest = serde_json::json!({
        "name": name,
        "checkpoint": file_name,
        "checkpoint_sha256": crate::checkpoint::file_sha256_hex(checkpoint)?,
        "student_config": serde_json::to_value(&config.student)?,
        "build": crate::checkpoint::BuildInfo::current(),
        "refreshed_utc": crate::checkpoint::unix_now(),
    });
    std::fs::create_dir_all(model_dir)
        .with_context(|| format!("create model directory {}", model_dir.display()))?;
    std::fs::write(model_dir.join("config.json"), manifest.to_string())
        .with_context(|| format!("write the sidecar in {}", model_dir.display()))?;
    Ok(())
}

/// The training half of one queue entry. Errors are returned; the caller
/// (`execute_entry`) turns them into a `queue.failed` record and moves on.
/// Generic over the autodiff backend: [`train_tick`] instantiates it with
/// the CPU [`TrainBackend`], [`train_tick_with`] with the caller's.
fn run_entry<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>>(
    paths: &NinaPaths,
    config: &SchedulerConfig,
    queue_seq: u64,
    payload: &Value,
) -> Result<QueueEntryOutcome> {
    let missing = |key: &str| {
        anyhow::anyhow!("queue entry {queue_seq} is missing a usable '{key}' field: {payload}")
    };
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| missing("name"))?;
    let (field, subfield) = validate_name(name)?;
    let dataset = payload
        .get("dataset_path")
        .and_then(Value::as_str)
        .ok_or_else(|| missing("dataset_path"))?;
    let steps = payload
        .get("steps")
        .and_then(Value::as_u64)
        .ok_or_else(|| missing("steps"))?;
    let steps = usize::try_from(steps)
        .ok()
        .filter(|s| *s > 0)
        .ok_or_else(|| {
            anyhow::anyhow!("queue entry {queue_seq} has steps {steps}; need a positive count")
        })?;

    let language = entry_language(field, subfield);
    // The scheduler-level field boxes join the student's, so a non-coding
    // slice's `field:{field}` box exists in the model it trains.
    let student_config = config.student_for_run();
    let model_dir = paths.model_dir(field, subfield);
    std::fs::create_dir_all(&model_dir)
        .with_context(|| format!("create {}", model_dir.display()))?;

    // The fold split is deterministic; scratch files are removed afterwards.
    // The dataset is cheat-scanned first: the folds are split from the lines
    // that survive the scan, never from the raw file.
    let work_dir = scratch_dir("queue");
    let policy = config.mixing_policy()?;
    let learned = config.ml_cheat()?;
    let (cleaned, cheat_scan) = scan_dataset_cheats(
        Path::new(dataset),
        language,
        &policy,
        learned.as_ref(),
        &work_dir,
    )?;
    let folds = split_dataset(&cleaned, language, config.holdout_percent, &work_dir)?;
    let train_fold =
        LanguageBatch::load(&folds.train, language, student_config.context, config.seed)?;
    let holdout_fold = LanguageBatch::load(
        &folds.holdout,
        language,
        student_config.context,
        config.seed,
    )?;
    let (train_lines, holdout_lines) = (folds.train_lines, folds.holdout_lines);

    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <B as burn::tensor::backend::Backend>::seed(&device, config.seed);
    let model = GeometricStudent::<B>::new(&student_config, &device)?;
    crate::tensor_ext::force_initialization(&model);
    let train_config = SpecialistTrainConfig {
        steps,
        batch_size: config.batch_size,
        lr: config.lr,
        seed: config.seed,
        log_every: config.log_every,
        out_dir: Some(model_dir.clone()),
        checkpoint_every: 0,
        resume: None,
        language,
        micro_batch: config.micro_batch,
        ..SpecialistTrainConfig::default()
    };
    let (model, report) = train_language_specialist(model, &train_fold, &train_config, &device)?;
    let checkpoint = report.checkpoint.clone().ok_or_else(|| {
        anyhow::anyhow!("the training run wrote no checkpoint though out_dir was set")
    })?;

    // Eval the trained weights (the autodiff module's inference twin) on the
    // held-out fold — the same numbers `nina.eval` recomputes from disk. The
    // eval runs on the training backend's inner backend (`valid()`'s module),
    // which shares the device, so a GPU run never round-trips through the CPU.
    use burn::module::AutodiffModule;
    let inference = model.valid();
    let eval_device: <B::InnerBackend as burn::tensor::backend::BackendTypes>::Device =
        Default::default();
    let (accuracy, router_top1) = eval_model(&inference, &holdout_fold, &eval_device)?;
    // Scratch files: a failed cleanup leaves temp files behind, nothing more.
    drop(std::fs::remove_dir_all(&work_dir));

    let checkpoint_hash = crate::checkpoint::file_sha256_hex(&checkpoint)?;
    let bytes = std::fs::metadata(&checkpoint)
        .with_context(|| format!("stat {}", checkpoint.display()))?
        .len();
    let evidence = EvalEvidence {
        accuracy: Some(accuracy),
        router_top1,
        date: iso_date(now_unix()),
    };
    let box_id = language.map_or_else(|| format!("field:{field}"), |l| l.box_id());
    let mut registry = Registry::open(paths.registry_json(), paths.registry_log())?;
    if registry.get(name).is_some() {
        // A retrain: fresh evidence (and weights) on the existing entry; the
        // promotion gate compares against the chain's baseline.
        registry.record_eval(
            name,
            evidence,
            Some(checkpoint_hash.clone()),
            Some(bytes),
            Some(vec![dataset.to_string()]),
        )?;
    } else {
        registry.register(ModelEntry {
            name: name.to_string(),
            field: field.to_string(),
            subfield: subfield.to_string(),
            box_id,
            checkpoint_hash: checkpoint_hash.clone(),
            status: ModelStatus::Candidate,
            eval: evidence,
            datasets: vec![dataset.to_string()],
            teacher: None,
            bytes,
        })?;
    }
    // Refresh the deployment sidecar so it names *this* checkpoint and *this*
    // architecture.
    //
    // The sidecar is written by the trace pipeline, once, for the first
    // checkpoint a model directory ever received. A retrain writes a new
    // content-addressed `.mpk` and a new registry entry, and used to leave the
    // sidecar alone — so it went on naming a checkpoint that had been replaced,
    // and carrying the `student_config` of *that* run. Six deployed
    // specialists were unservable that way, and `student think` on any of them
    // failed on a shape mismatch (6 routing boxes against a checkpoint
    // trained with 10) rather than saying which config it wanted.
    //
    // Written after the registry entry, so a failure here cannot leave the
    // registry claiming weights that are not on disk. A failure to write it
    // is reported rather than swallowed: the model still trains and is still
    // registered, but the operator should know the sidecar is stale.
    if let Err(err) = write_deployment_sidecar(&model_dir, name, &checkpoint, config) {
        eprintln!(
            "note: {name} trained and registered, but its deployment sidecar could not be \
             refreshed: {err:#} — `nina chat` will fall back to resolving the checkpoint from \
             the registry hash, and `student think` needs --config"
        );
    }
    Ok(QueueEntryOutcome {
        queue_seq,
        name: name.to_string(),
        ok: true,
        checkpoint: Some(checkpoint),
        checkpoint_hash: Some(checkpoint_hash),
        accuracy: Some(accuracy),
        router_top1,
        steps_taken: report.steps_taken,
        train_lines,
        holdout_lines,
        cheat_scan,
        backend: std::any::type_name::<B>().to_string(),
        error: None,
    })
}

/// Run one queue entry end to end: `queue.start`, the training slice, then
/// `queue.done` or `queue.failed`. A failure is recorded and returned as a
/// failed outcome — it never stops the queue. Generic over the autodiff
/// backend; see [`train_tick_with`].
fn execute_entry<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>>(
    paths: &NinaPaths,
    config: &SchedulerConfig,
    queue_seq: u64,
    payload: &Value,
) -> Result<QueueEntryOutcome> {
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("<unreadable entry>")
        .to_string();
    {
        let mut log = ChainedLog::open(paths.queue_log())?;
        log.append(&json!({
            "event": "queue.start",
            "queue_seq": queue_seq,
            "name": name,
            // Recorded at start, not only at the end: a run that crashes
            // mid-entry writes no `queue.done`, and without this the one
            // record that could attribute the partial work is the one that
            // is missing.
            "backend": std::any::type_name::<B>(),
        }))?;
    }
    match run_entry::<B>(paths, config, queue_seq, payload) {
        Ok(outcome) => {
            let mut log = ChainedLog::open(paths.queue_log())?;
            log.append(&json!({
                "event": "queue.done",
                "queue_seq": queue_seq,
                "name": outcome.name,
                "checkpoint": outcome.checkpoint,
                "checkpoint_hash": outcome.checkpoint_hash,
                "accuracy": outcome.accuracy,
                "router_top1": outcome.router_top1,
                "steps_taken": outcome.steps_taken,
                "train_lines": outcome.train_lines,
                "holdout_lines": outcome.holdout_lines,
                "cheat_scanned": outcome.cheat_scan.scanned,
                "cheat_dropped": outcome.cheat_scan.dropped_by_class,
                "cheat_mixing": outcome.cheat_scan.mixing,
                "cheat_ml": outcome.cheat_scan.learned,
                "backend": outcome.backend,
            }))?;
            Ok(outcome)
        }
        Err(err) => {
            let error = format!("{err:#}");
            let mut log = ChainedLog::open(paths.queue_log())?;
            log.append(&json!({
                "event": "queue.failed",
                "queue_seq": queue_seq,
                "name": name,
                "error": error,
            }))?;
            Ok(QueueEntryOutcome {
                queue_seq,
                name,
                ok: false,
                checkpoint: None,
                checkpoint_hash: None,
                accuracy: None,
                router_top1: None,
                steps_taken: 0,
                train_lines: 0,
                holdout_lines: 0,
                cheat_scan: CheatScanSummary::default(),
                backend: std::any::type_name::<B>().to_string(),
                error: Some(error),
            })
        }
    }
}

/// Run ONE queued training slice (the `nina.train.tick` tool): the oldest
/// pending entry trains, is evaluated, and is registered. `None` when the
/// queue is empty. Trains on the CPU `TrainBackend` — the bit-compat path
/// the MCP server and the existing drains run; [`train_tick_with`] is the
/// same tick on a caller-chosen autodiff backend.
pub fn train_tick(root: &Path, config: &SchedulerConfig) -> Result<Option<QueueEntryOutcome>> {
    train_tick_with::<TrainBackend>(root, config)
}

/// [`train_tick`] on an explicit autodiff backend (e.g.
/// `Autodiff<Wgpu<f32>>` with the `wgpu` feature): the same queue semantics,
/// the same chain records, the same seeds; the device the backend's
/// `Default` names decides where the arithmetic runs. Cross-backend results
/// are not bit-identical (reduction orders differ), so a queue entry's
/// recorded numbers belong to the backend that produced them.
pub fn train_tick_with<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>>(
    root: &Path,
    config: &SchedulerConfig,
) -> Result<Option<QueueEntryOutcome>> {
    config.validate()?;
    let paths = NinaPaths::new(root);
    let pending = pending_queue_entries(&paths)?;
    let Some((seq, payload)) = pending.into_iter().next() else {
        return Ok(None);
    };
    execute_entry::<B>(&paths, config, seq, &payload).map(Some)
}

/// Drain the queue, serially: one slice at a time until nothing is pending.
/// Serial is honest on this box; the chain records the order it happened in.
pub fn run_queue(root: &Path, config: &SchedulerConfig) -> Result<SchedulerReport> {
    config.validate()?;
    let started = std::time::Instant::now();
    let mut outcomes = Vec::new();
    while let Some(outcome) = train_tick(root, config)? {
        outcomes.push(outcome);
    }
    let trained = outcomes.iter().filter(|o| o.ok).count();
    Ok(SchedulerReport {
        drained: outcomes.len(),
        trained,
        failed: outcomes.len() - trained,
        outcomes,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// The MCP server (JSON-RPC 2.0 over stdio, std-only)
// ---------------------------------------------------------------------------

/// Every path the NINA tooling reads or writes, derived from one root.
#[derive(Debug, Clone)]
pub struct NinaPaths {
    root: PathBuf,
}

impl NinaPaths {
    /// Root the layout at `root` (see [`DEFAULT_ROOT`]).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root itself.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `<root>/registry.json` — the derived read model.
    pub fn registry_json(&self) -> PathBuf {
        self.root.join("registry.json")
    }

    /// `<root>/logs/registry.jsonl` — the registry's chained truth.
    pub fn registry_log(&self) -> PathBuf {
        self.root.join("logs").join("registry.jsonl")
    }

    /// `<root>/logs/queue.jsonl` — the training queue, a chained log itself.
    pub fn queue_log(&self) -> PathBuf {
        self.root.join("logs").join("queue.jsonl")
    }

    /// `<root>/logs/mcp.jsonl` — the audit chain: every `tools/call` lands
    /// here (PLAN-NINA.md §4).
    pub fn mcp_log(&self) -> PathBuf {
        self.root.join("logs").join("mcp.jsonl")
    }

    /// `<root>/models` — the folder layout PLAN-NINA.md §1 describes.
    pub fn models_root(&self) -> PathBuf {
        self.root.join("models")
    }

    /// `<root>/models/<field>/<subfield>` — one specialist's home; its
    /// checkpoints land here, content-addressed.
    pub fn model_dir(&self, field: &str, subfield: &str) -> PathBuf {
        self.models_root().join(field).join(subfield)
    }

    /// `<root>/student` — where `dblocks student train` writes checkpoints.
    pub fn student_dir(&self) -> PathBuf {
        self.root.join("student")
    }
}

/// One entry of the v1 tool table. An unavailable tool is declared anyway,
/// with the reason — honest beats presentable.
pub struct ToolSpec {
    /// The tool's MCP name (e.g. `nina.register`).
    pub name: &'static str,
    /// What it does, for `tools/list`.
    pub description: &'static str,
    /// JSON Schema for the `arguments` object.
    pub input_schema: Value,
    /// Whether `tools/call` will execute it.
    pub available: bool,
    /// Why an unavailable tool is off.
    pub unavailable_reason: Option<&'static str>,
}

/// The v1 tool set, in declaration order: nine working tools plus one
/// declared-but-unavailable one (`nina.think`) that carries its reason
/// instead of faking a result.
pub fn tool_table() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "nina.queue.add",
            description: "Validate and append a training job to the queue (itself a chained log): \
                          specialist name, field, dataset path, steps.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "field/subfield, lowercase (e.g. pentest/recon)"},
                    "field": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "steps": {"type": "integer", "minimum": 1},
                },
                "required": ["name", "field", "dataset_path", "steps"],
            }),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.queue.list",
            description: "List the pending training queue (nothing consumes it in v1, so every \
                          entry is pending).",
            input_schema: json!({"type": "object", "properties": {}}),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.route",
            description: "Route a prompt: the student's probe router when a checkpoint exists, \
                          else the registry's field list with a note that no trained router is \
                          loaded.",
            input_schema: json!({
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            }),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.register",
            description: "Register a model: validates the entry, appends to the chained log, \
                          rewrites registry.json (derived state; the log is the truth).",
            input_schema: json!({
                "type": "object",
                "properties": {"entry": {"type": "object", "description": "a ModelEntry"}},
                "required": ["entry"],
            }),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.registry.retire",
            description: "Withdraw a model: flips its status to `retired`, keeping the entry \
                          addressable for provenance. Use it when a registered accuracy cannot be \
                          trusted — a number measured against data that was later found \
                          degenerate is a claim, and leaving it standing is worse than withdrawing \
                          it.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string", "description": "why the evidence is withdrawn"},
                },
                "required": ["name", "reason"],
            }),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.registry.get",
            description: "One registry entry by name.",
            input_schema: json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.registry.list",
            description: "Every registry entry, or one field's with `field`.",
            input_schema: json!({
                "type": "object",
                "properties": {"field": {"type": "string"}},
            }),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.log.verify",
            description: "Re-derive a Merkle-chained log and name the first broken link.",
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.train.tick",
            description: "Run ONE queued training slice: train, eval on a held-out fold, register \
                          as a candidate; returns the outcome. `ran` is false when the queue is \
                          empty.",
            input_schema: json!({"type": "object", "properties": {}}),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.eval",
            description: "Recompute a registry entry's eval evidence on demand: held-out fold of \
                          its recorded dataset — or of `dataset_path` when given — loaded from \
                          its latest checkpoint; the entry's evidence is updated (an `eval` \
                          record lands in the chain).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dataset_path": {"type": "string"},
                },
                "required": ["name"],
            }),
            available: true,
            unavailable_reason: None,
        },
        ToolSpec {
            name: "nina.think",
            description: "Read a vecread thought trace.",
            input_schema: json!({"type": "object", "properties": {}}),
            available: false,
            unavailable_reason: Some(
                "the vecread trace endpoint is not wired here yet; use `dblocks student think`",
            ),
        },
    ]
}

/// The NINA MCP server: JSON-RPC 2.0 over stdio (newline-delimited frames),
/// std-only. [`McpServer::serve`] reads one JSON object per line and writes
/// one response per request; notifications get none, per the spec. Every
/// `tools/call` is appended to the audit chain (`logs/mcp.jsonl`) with its
/// arguments and its outcome.
pub struct McpServer {
    paths: NinaPaths,
    scheduler: SchedulerConfig,
}

impl McpServer {
    /// A server over the registry, logs, queue and models under `paths`,
    /// with the default scheduler configuration.
    pub fn new(paths: NinaPaths) -> Self {
        Self {
            paths,
            scheduler: SchedulerConfig::default(),
        }
    }

    /// [`Self::new`] with an explicit scheduler configuration (the student
    /// architecture `nina.train.tick` trains, the holdout fraction, the
    /// promotion margin).
    pub fn with_scheduler(paths: NinaPaths, scheduler: SchedulerConfig) -> Self {
        Self { paths, scheduler }
    }

    /// The paths this server reads and writes.
    pub fn paths(&self) -> &NinaPaths {
        &self.paths
    }

    /// Serve frames until EOF: one JSON-RPC object per line in, one response
    /// per request out. Blank lines are ignored; EOF is a clean shutdown.
    pub fn serve<R: BufRead, W: Write>(&self, mut reader: R, mut writer: W) -> Result<()> {
        let mut frame = String::new();
        loop {
            frame.clear();
            let read = reader.read_line(&mut frame).context("read an MCP frame")?;
            if read == 0 {
                return Ok(()); // EOF: the client hung up
            }
            let trimmed = frame.trim_end();
            if trimmed.is_empty() {
                continue; // blank separators carry no message
            }
            if let Some(response) = self.handle_message(trimmed) {
                serde_json::to_writer(&mut writer, &response)
                    .context("serialize an MCP response")?;
                writer.write_all(b"\n").context("write an MCP response")?;
                writer.flush().context("flush an MCP response")?;
            }
        }
    }

    /// Handle one frame. The return is `None` for notifications (the spec
    /// forbids answering them) and `Some(response)` otherwise — including for
    /// the parse error, which answers with a null id.
    pub fn handle_message(&self, raw: &str) -> Option<Value> {
        let request: Value = match serde_json::from_str(raw) {
            Ok(value) => value,
            Err(err) => {
                let shown: String = raw.chars().take(120).collect();
                return Some(json_error(
                    Value::Null,
                    JSONRPC_PARSE_ERROR,
                    format!("parse error: {err}; received '{shown}'"),
                ));
            }
        };
        let id = request.get("id").cloned();
        if let Some(version) = request.get("jsonrpc") {
            if version.as_str() != Some("2.0") {
                return Some(json_error(
                    id.unwrap_or(Value::Null),
                    JSONRPC_INVALID_REQUEST,
                    format!("invalid request: 'jsonrpc' must be \"2.0\", received {version}"),
                ));
            }
        }
        let method = match request.get("method").and_then(Value::as_str) {
            Some(method) => method,
            None => {
                let shown: String = request.to_string().chars().take(120).collect();
                return Some(json_error(
                    id.unwrap_or(Value::Null),
                    JSONRPC_INVALID_REQUEST,
                    format!("invalid request: no string 'method' field; received {shown}"),
                ));
            }
        };
        match method {
            "initialize" => Some(json_result(
                id.unwrap_or(Value::Null),
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "nina", "version": env!("CARGO_PKG_VERSION")},
                }),
            )),
            "notifications/initialized" => None,
            "tools/list" => Some(json_result(id.unwrap_or(Value::Null), tools_list())),
            "tools/call" => {
                Some(self.handle_tools_call(id.unwrap_or(Value::Null), request.get("params")))
            }
            // An unknown *notification* (no id) maps to None: the spec forbids
            // answering notifications, even with an error.
            _ => id.map(|id| {
                json_error(
                    id,
                    JSONRPC_METHOD_NOT_FOUND,
                    format!("method not found: '{method}'"),
                )
            }),
        }
    }

    fn handle_tools_call(&self, id: Value, params: Option<&Value>) -> Value {
        let params = params.cloned().unwrap_or_else(|| json!({}));
        let name = match params.get("name").and_then(Value::as_str) {
            Some(name) => name,
            None => {
                return json_error(
                    id,
                    JSONRPC_INVALID_PARAMS,
                    format!("invalid params: tools/call needs a string 'name'; received {params}"),
                );
            }
        };
        let table = tool_table();
        let known = table.iter().any(|t| t.name == name);
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let outcome = if known {
            self.call_tool(name, &arguments)
        } else {
            let declared: Vec<&str> = table.iter().map(|t| t.name).collect();
            Err(anyhow::anyhow!(
                "unknown tool '{name}'; the declared tools are: {}",
                declared.join(", ")
            ))
        };
        // Every tools/call lands in the audit chain — the swarm's training
        // actions and the audit of them are the same artifact (§4). If the
        // audit itself fails, the call is reported as an internal error: a
        // system that cannot log should not silently act.
        if let Err(err) = self.audit_call(name, &arguments, &outcome) {
            return json_error(
                id,
                JSONRPC_INTERNAL_ERROR,
                format!("the audit log refused the record for '{name}': {err:#}"),
            );
        }
        match outcome {
            Ok(result) => {
                let text =
                    serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string());
                json_result(
                    id,
                    json!({"content": [{"type": "text", "text": text}], "isError": false}),
                )
            }
            Err(err) if !known => json_error(id, JSONRPC_INVALID_PARAMS, format!("{err:#}")),
            Err(err) => json_result(
                id,
                json!({"content": [{"type": "text", "text": format!("{err:#}")}], "isError": true}),
            ),
        }
    }

    /// Append one `mcp.call` record to the audit chain.
    fn audit_call(&self, tool: &str, arguments: &Value, outcome: &Result<Value>) -> Result<()> {
        let mut log = ChainedLog::open(self.paths.mcp_log())?;
        log.append(&json!({
            "event": "mcp.call",
            "tool": tool,
            "arguments": arguments,
            "ok": outcome.is_ok(),
            "error": outcome.as_ref().err().map(|e| format!("{e:#}")),
        }))?;
        Ok(())
    }

    fn call_tool(&self, name: &str, args: &Value) -> Result<Value> {
        match name {
            "nina.queue.add" => self.tool_queue_add(args),
            "nina.queue.list" => self.tool_queue_list(),
            "nina.route" => self.tool_route(args),
            "nina.register" => self.tool_register(args),
            "nina.registry.retire" => self.tool_registry_retire(args),
            "nina.registry.get" => self.tool_registry_get(args),
            "nina.registry.list" => self.tool_registry_list(args),
            "nina.log.verify" => Self::tool_log_verify(args),
            "nina.train.tick" => self.tool_train_tick(),
            "nina.eval" => self.tool_eval(args),
            other => {
                let reason = tool_table()
                    .iter()
                    .find(|t| t.name == other)
                    .and_then(|t| t.unavailable_reason);
                match reason {
                    Some(reason) => anyhow::bail!(
                        "tool '{other}' is declared but not available in v1: {reason}"
                    ),
                    None => anyhow::bail!("unknown tool '{other}'"),
                }
            }
        }
    }

    fn tool_train_tick(&self) -> Result<Value> {
        match train_tick(self.paths.root(), &self.scheduler)? {
            Some(outcome) => Ok(json!({"ran": true, "outcome": outcome})),
            None => Ok(json!({"ran": false, "note": "the queue is empty"})),
        }
    }

    fn tool_eval(&self, args: &Value) -> Result<Value> {
        let name = arg_str(args, "name")?;
        let (field, subfield) = validate_name(name)?;
        let file = RegistryFile::load(&self.paths.registry_json())?;
        let entry = file.models.iter().find(|m| m.name == name).ok_or_else(|| {
            anyhow::anyhow!(
                "no model named '{name}' in the registry ({} entries)",
                file.models.len()
            )
        })?;
        let dataset = match args.get("dataset_path") {
            None | Some(Value::Null) => entry
                .datasets
                .first()
                .ok_or_else(|| {
                    anyhow::anyhow!("registry entry '{name}' records no dataset; eval needs one")
                })?
                .clone(),
            // An override: eval against this dataset (through the same
            // pre-training scan) instead of the recorded one.
            Some(value) => value.as_str().map(str::to_string).ok_or_else(|| {
                anyhow::anyhow!("tool argument 'dataset_path' must be a string; received {value}")
            })?,
        };
        let model_dir = self.paths.model_dir(field, subfield);
        let checkpoint =
            crate::checkpoint::latest_in_dir(&model_dir, "student")?.ok_or_else(|| {
                anyhow::anyhow!(
                "no student checkpoint under {} for '{name}'; train one first (nina.train.tick)",
                model_dir.display()
            )
            })?;
        let language = entry_language(field, subfield);
        let evidence =
            eval_from_checkpoint(&checkpoint, Path::new(&dataset), language, &self.scheduler)?;
        let checkpoint_hash = crate::checkpoint::file_sha256_hex(&checkpoint)?;
        let bytes = std::fs::metadata(&checkpoint)
            .with_context(|| format!("stat {}", checkpoint.display()))?
            .len();
        let mut registry = Registry::open(self.paths.registry_json(), self.paths.registry_log())?;
        registry.record_eval(name, evidence, Some(checkpoint_hash), Some(bytes), None)?;
        let entry = registry.get(name).ok_or_else(|| {
            anyhow::anyhow!("'{name}' vanished from the registry between load and record_eval")
        })?;
        serde_json::to_value(entry).context("serialize the registry entry")
    }

    fn tool_queue_add(&self, args: &Value) -> Result<Value> {
        let name = arg_str(args, "name")?;
        let (field_part, _) = validate_name(name)?;
        let field = arg_str(args, "field")?;
        ensure!(
            field == field_part,
            "queue entry '{name}' disagrees with field '{field}': the name's field part is '{field_part}'"
        );
        let dataset_path = arg_str(args, "dataset_path")?;
        ensure!(
            !dataset_path.is_empty(),
            "tool argument 'dataset_path' must not be empty"
        );
        let steps = arg_u64(args, "steps")?;
        ensure!(
            steps > 0,
            "tool argument 'steps' must be positive; received {steps}"
        );
        let mut log = ChainedLog::open(self.paths.queue_log())?;
        let record = log.append(&json!({
            "event": "queue.add",
            "name": name,
            "field": field,
            "dataset_path": dataset_path,
            "steps": steps,
            "status": "pending",
        }))?;
        Ok(json!({
            "queued": name,
            "seq": record.seq,
            "log": log.path().display().to_string(),
        }))
    }

    fn tool_queue_list(&self) -> Result<Value> {
        let pending: Vec<Value> = pending_queue_entries(&self.paths)?
            .into_iter()
            .map(|(seq, payload)| json!({"seq": seq, "entry": payload}))
            .collect();
        let count = pending.len();
        // Pending = added and neither done nor failed; the scheduler settles
        // entries with queue.done / queue.failed records in the same chain.
        Ok(json!({"pending": pending, "count": count}))
    }

    fn tool_route(&self, args: &Value) -> Result<Value> {
        let prompt = arg_str(args, "prompt")?;
        let registry_path = self.paths.registry_json();
        let (fields, registry_note) = match RegistryFile::load(&registry_path) {
            Ok(file) => (file.fields(), None),
            Err(err) => (
                Vec::new(),
                Some(format!(
                    "the registry is unreadable ({err:#}); the field list is empty"
                )),
            ),
        };
        let checkpoint = crate::checkpoint::latest_in_dir(&self.paths.student_dir(), "student")?;
        let mut result = route_prompt(prompt, checkpoint.as_deref(), &fields)?;
        if let (Some(note), Some(object)) = (registry_note, result.as_object_mut()) {
            object.insert("registry_note".to_string(), Value::String(note));
        }
        Ok(result)
    }

    fn tool_register(&self, args: &Value) -> Result<Value> {
        let entry_value = args.get("entry").cloned().ok_or_else(|| {
            anyhow::anyhow!("nina.register needs an 'entry' object; received {args}")
        })?;
        let entry: ModelEntry = serde_json::from_value(entry_value).map_err(|err| {
            anyhow::anyhow!("parse the 'entry' argument as a registry ModelEntry: {err}")
        })?;
        let name = entry.name.clone();
        let mut registry = Registry::open(self.paths.registry_json(), self.paths.registry_log())?;
        let record = registry.register(entry)?;
        Ok(json!({
            "registered": name,
            "status": ModelStatus::Candidate.as_str(),
            "seq": record.seq,
        }))
    }

    fn tool_registry_retire(&self, args: &Value) -> Result<Value> {
        let name = arg_str(args, "name")?;
        let reason = arg_str(args, "reason")?;
        let mut registry = Registry::open(self.paths.registry_json(), self.paths.registry_log())?;
        let before = registry
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("no model named '{name}' in the registry"))?;
        // The reason travels in the chain, next to the flip, so a reader of the
        // log learns *why* an accuracy was withdrawn rather than only that it
        // was. `retire` itself records the status change; this records the
        // justification as its own event.
        registry.log_event(json!({
            "event": "retire.reason",
            "name": name,
            "reason": reason,
            "withdrawn_accuracy": before.eval.accuracy,
        }))?;
        let record = registry.retire(name)?;
        Ok(json!({"retired": name, "seq": record.seq, "reason": reason}))
    }

    fn tool_registry_get(&self, args: &Value) -> Result<Value> {
        let name = arg_str(args, "name")?;
        let file = RegistryFile::load(&self.paths.registry_json())?;
        let entry = file.models.iter().find(|m| m.name == name).ok_or_else(|| {
            anyhow::anyhow!(
                "no model named '{name}' in the registry ({} entries)",
                file.models.len()
            )
        })?;
        serde_json::to_value(entry).context("serialize the registry entry")
    }

    fn tool_registry_list(&self, args: &Value) -> Result<Value> {
        let field = args.get("field").and_then(Value::as_str);
        let file = RegistryFile::load(&self.paths.registry_json())?;
        let models: Vec<&ModelEntry> = file
            .models
            .iter()
            .filter(|m| field.is_none_or(|f| m.field == f))
            .collect();
        Ok(json!({"count": models.len(), "models": models}))
    }

    fn tool_log_verify(args: &Value) -> Result<Value> {
        let path = arg_str(args, "path")?;
        let report = ChainedLog::verify(Path::new(path))?;
        Ok(json!({
            "ok": true,
            "path": path,
            "lines": report.lines,
            "head_hash": report.head_hash,
            "first_ts": report.first_ts,
            "last_ts": report.last_ts,
        }))
    }
}

fn tools_list() -> Value {
    let tools: Vec<Value> = tool_table()
        .iter()
        .map(|t| {
            let mut tool = json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "available": t.available,
            });
            if let (Some(reason), Some(object)) = (t.unavailable_reason, tool.as_object_mut()) {
                object.insert("reason".to_string(), Value::String(reason.to_string()));
            }
            tool
        })
        .collect();
    json!({"tools": tools})
}

fn json_result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn json_error(id: Value, code: i64, message: String) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| {
        let got = args
            .get(key)
            .map_or("absent".to_string(), |v| v.to_string());
        anyhow::anyhow!("tool argument '{key}' must be a string; received {got}")
    })
}

fn arg_u64(args: &Value, key: &str) -> Result<u64> {
    args.get(key).and_then(Value::as_u64).ok_or_else(|| {
        let got = args
            .get(key)
            .map_or("absent".to_string(), |v| v.to_string());
        anyhow::anyhow!("tool argument '{key}' must be a non-negative integer; received {got}")
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
    use std::io::BufReader;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SCRATCH: AtomicUsize = AtomicUsize::new(0);

    fn scratch(stem: &str) -> PathBuf {
        let n = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("dblocks-nina-{stem}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(name: &str, field: &str, subfield: &str) -> ModelEntry {
        ModelEntry {
            name: name.to_string(),
            field: field.to_string(),
            subfield: subfield.to_string(),
            box_id: format!("{field}:{subfield}"),
            ..ModelEntry::default()
        }
    }

    // -- the chained log -----------------------------------------------------

    #[test]
    fn chain_append_verify_round_trip() {
        let dir = scratch("chain-roundtrip");
        let path = dir.join("log.jsonl");
        let mut log = ChainedLog::open(&path).unwrap();
        assert!(log.is_empty());
        assert_eq!(log.head(), GENESIS_HASH);
        let r0 = log.append(&json!({"event": "a"})).unwrap();
        assert_eq!((r0.seq, r0.prev.as_str()), (0, GENESIS_HASH));
        let r1 = log.append(&json!({"event": "b", "n": 1})).unwrap();
        assert_eq!((r1.seq, r1.prev.as_str()), (1, r0.hash.as_str()));

        // Reopening derives the head from the file, so the chain continues.
        let mut log = ChainedLog::open(&path).unwrap();
        assert_eq!(log.head(), r1.hash);
        assert_eq!(log.len(), 2);
        let r2 = log.append(&json!({"event": "c"})).unwrap();
        assert_eq!(r2.seq, 2);

        let report = ChainedLog::verify(&path).unwrap();
        assert_eq!(report.lines, 3);
        assert_eq!(report.head_hash, r2.hash);
        assert!(report.first_ts.is_some() && report.last_ts.is_some());

        // The hash formula is the documented one: sha256(prev || canonical(payload)).
        let mut preimage = r1.hash.into_bytes();
        preimage.extend_from_slice(&canonical_bytes(&json!({"event": "c"})));
        assert_eq!(r2.hash, sha256_hex(&preimage));

        // An empty log verifies as an empty chain; a missing log is an error.
        let empty = dir.join("empty.jsonl");
        std::fs::write(&empty, "").unwrap();
        assert_eq!(ChainedLog::verify(&empty).unwrap().lines, 0);
        assert!(ChainedLog::verify(&dir.join("absent.jsonl")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// serde_json's default float parser is not correctly rounded for every
    /// lexeme: "0.9959735770559633" used to parse one ULP low, so verifying an
    /// unedited record recomputed a different hash. With the crate-wide
    /// `arbitrary_precision` feature a parsed Number keeps its lexeme and this
    /// exact value must round-trip through a written-and-re-read chain.
    #[test]
    fn chain_verify_is_exact_for_float_lexemes() {
        let dir = scratch("chain-float-lexeme");
        let path = dir.join("log.jsonl");
        let mut log = ChainedLog::open(&path).unwrap();
        log.append(
            &json!({"accuracy": 0.29484275361554274_f64, "router_top1": 0.9959735770559633_f64}),
        )
        .unwrap();
        drop(log);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("0.9959735770559633"),
            "the written record must carry the full-precision lexeme: {text}"
        );
        let report = ChainedLog::verify(&path).unwrap();
        assert_eq!(report.lines, 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_names_an_edited_middle_line() {
        let dir = scratch("chain-edited");
        let path = dir.join("log.jsonl");
        let mut log = ChainedLog::open(&path).unwrap();
        for i in 0..3 {
            log.append(&json!({"event": "record", "i": i})).unwrap();
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let edited = text.replacen("\"i\":1", "\"i\":99", 1);
        assert_ne!(edited, text);
        std::fs::write(&path, edited).unwrap();
        let err = ChainedLog::verify(&path).unwrap_err().to_string();
        assert!(err.contains("line 2"), "{err}");
        assert!(err.contains("edited after the fact"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_names_a_truncated_tail() {
        let dir = scratch("chain-truncated");
        let path = dir.join("log.jsonl");
        let mut log = ChainedLog::open(&path).unwrap();
        for i in 0..3 {
            log.append(&json!({"event": "record", "i": i})).unwrap();
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let cut = text.len() - 12; // mid-way through the last line
        std::fs::write(&path, &text[..cut]).unwrap();
        let err = ChainedLog::verify(&path).unwrap_err().to_string();
        assert!(err.contains("line 3"), "{err}");
        assert!(err.contains("not valid JSON"), "{err}");
        // Appending onto a cut tail is refused too, pointing at verify.
        let err = ChainedLog::open(&path)
            .and_then(|mut log| log.append(&json!({"event": "d"})))
            .unwrap_err()
            .to_string();
        assert!(err.contains("verify-log"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_names_a_seq_gap() {
        let dir = scratch("chain-seq");
        let path = dir.join("log.jsonl");
        let mut log = ChainedLog::open(&path).unwrap();
        for i in 0..3 {
            log.append(&json!({"event": "record", "i": i})).unwrap();
        }
        // Renumbering a record keeps every hash valid (seq is not in the
        // hash preimage); the gapless-numbering check is what catches it.
        let text = std::fs::read_to_string(&path).unwrap();
        let edited = text.replacen("{\"seq\":1,", "{\"seq\":5,", 1);
        assert_ne!(edited, text);
        std::fs::write(&path, edited).unwrap();
        let err = ChainedLog::verify(&path).unwrap_err().to_string();
        assert!(err.contains("line 2"), "{err}");
        assert!(err.contains("seq is 5, expected 1"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // -- the registry --------------------------------------------------------

    #[test]
    fn registry_register_get_list_round_trip() {
        let dir = scratch("registry-roundtrip");
        let registry_path = dir.join("registry.json");
        let log_path = dir.join("logs").join("registry.jsonl");
        let mut registry = Registry::open(&registry_path, &log_path).unwrap();
        registry
            .register(entry("coding/rust", "coding", "rust"))
            .unwrap();
        registry
            .register(entry("pentest/recon", "pentest", "recon"))
            .unwrap();
        assert!(registry.get("coding/rust").is_some());
        assert!(registry.get("bio/general").is_none());
        assert_eq!(registry.list(None).len(), 2);
        assert_eq!(registry.list(Some("coding")).len(), 1);
        assert_eq!(registry.list(Some("bio")).len(), 0);

        // The derived file parses and carries the schema version.
        let file = RegistryFile::load(&registry_path).unwrap();
        assert_eq!(file.version, REGISTRY_SCHEMA_VERSION);
        assert_eq!(file.models.len(), 2);
        assert_eq!(
            file.fields(),
            vec!["coding".to_string(), "pentest".to_string()]
        );

        // The truth: two register records, chain intact.
        let report = ChainedLog::verify(&log_path).unwrap();
        assert_eq!(report.lines, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn registry_rejects_bad_names_naming_the_rule() {
        for bad in [
            "Pentest/Recon",
            "noslash",
            "a/",
            "/b",
            "a/b/c",
            "a b/c",
            "a//b",
        ] {
            let err = validate_name(bad).unwrap_err().to_string();
            assert!(err.contains("field/subfield"), "{bad}: {err}");
            assert!(err.contains(bad), "{bad}: {err}");
        }
        assert_eq!(
            validate_name("pentest/recon").unwrap(),
            ("pentest", "recon")
        );

        // Register enforces the same rule, and name/field agreement.
        let dir = scratch("registry-names");
        let mut registry = Registry::open(dir.join("r.json"), dir.join("r.jsonl")).unwrap();
        let err = registry
            .register(entry("Coding/Rust", "Coding", "Rust"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("field/subfield"), "{err}");
        let err = registry
            .register(entry("coding/rust", "coding", "python"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("disagrees"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn promotion_requires_eval_evidence_and_transitions_are_logged() {
        let dir = scratch("registry-promote");
        let registry_path = dir.join("registry.json");
        let log_path = dir.join("logs").join("registry.jsonl");
        let mut registry = Registry::open(&registry_path, &log_path).unwrap();

        // No eval evidence: promotion is refused, saying why.
        registry
            .register(entry("coding/rust", "coding", "rust"))
            .unwrap();
        let err = registry.promote("coding/rust").unwrap_err().to_string();
        assert!(err.contains("eval accuracy"), "{err}");
        assert_eq!(
            registry.get("coding/rust").unwrap().status,
            ModelStatus::Candidate
        );

        // Evidence attached: promotion proceeds, and the transition is logged.
        let mut proven = entry("coding/python", "coding", "python");
        proven.eval.accuracy = Some(0.91);
        proven.eval.date = "2026-10-10".to_string();
        registry.register(proven).unwrap();
        registry.promote("coding/python").unwrap();
        assert_eq!(
            registry.get("coding/python").unwrap().status,
            ModelStatus::Active
        );
        let err = registry.promote("coding/python").unwrap_err().to_string();
        assert!(err.contains("already active"), "{err}");
        registry.retire("coding/python").unwrap();
        assert_eq!(
            registry.get("coding/python").unwrap().status,
            ModelStatus::Retired
        );
        let err = registry.retire("coding/python").unwrap_err().to_string();
        assert!(err.contains("already retired"), "{err}");

        // Unknown names are loud; duplicates are refused.
        assert!(registry.promote("bio/general").is_err());
        let err = registry
            .register(entry("coding/rust", "coding", "rust"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("already registered"), "{err}");

        // The chain holds the whole history, in order, and verifies.
        let report = ChainedLog::verify(&log_path).unwrap();
        assert_eq!(report.lines, 4);
        let text = std::fs::read_to_string(&log_path).unwrap();
        let events: Vec<String> = text
            .lines()
            .map(|line| {
                let record: Value = serde_json::from_str(line).unwrap();
                record["payload"]["event"].as_str().unwrap().to_string()
            })
            .collect();
        assert_eq!(events, ["register", "register", "promote", "retire"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // -- routing fallback ----------------------------------------------------

    #[test]
    fn route_without_a_checkpoint_returns_the_field_list() {
        let fields = vec!["coding".to_string(), "pentest".to_string()];
        let result = route_prompt("fn main() {}", None, &fields).unwrap();
        assert_eq!(result["router"], Value::Null);
        assert!(
            result["note"]
                .as_str()
                .unwrap()
                .contains("no trained router is loaded"),
            "{result}"
        );
        assert_eq!(result["fields"], json!(["coding", "pentest"]));

        // An absent explicit checkpoint is the same fallback, naming the path.
        let dir = scratch("route-absent");
        let absent = dir.join("absent.mpk");
        let result = route_prompt("x", Some(&absent), &fields).unwrap();
        assert_eq!(result["router"], Value::Null);
        assert!(
            result["note"].as_str().unwrap().contains("absent.mpk"),
            "{result}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // -- the MCP server ------------------------------------------------------

    #[test]
    fn mcp_handshake_and_tools_over_a_socket_pair() {
        use std::os::unix::net::UnixStream;
        let dir = scratch("mcp-pipe");
        let server = McpServer::new(NinaPaths::new(dir.clone()));
        let (client, server_side) = UnixStream::pair().unwrap();
        server_side
            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
            .unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
            .unwrap();
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let reader = BufReader::new(server_side.try_clone().unwrap());
                server.serve(reader, server_side)
            });
            let mut writer = client.try_clone().unwrap();
            let mut reader = BufReader::new(client);
            fn exchange<W: Write, R: BufRead>(
                writer: &mut W,
                reader: &mut R,
                request: &Value,
            ) -> Value {
                writeln!(writer, "{request}").unwrap();
                writer.flush().unwrap();
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                serde_json::from_str(&line).unwrap()
            }

            // The handshake.
            let response = exchange(
                &mut writer,
                &mut reader,
                &json!({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": MCP_PROTOCOL_VERSION,
                               "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
                }),
            );
            assert_eq!(response["id"], 1);
            assert_eq!(
                response["result"]["protocolVersion"],
                Value::from(MCP_PROTOCOL_VERSION)
            );
            assert_eq!(response["result"]["serverInfo"]["name"], "nina");
            // The initialized notification is answered with silence, so the
            // next response belongs to the next request.
            writeln!(
                writer,
                "{}",
                json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
            )
            .unwrap();

            // tools/list is exactly the v1 set.
            let response = exchange(
                &mut writer,
                &mut reader,
                &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            );
            let tools = response["result"]["tools"].as_array().unwrap();
            let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
            assert_eq!(
                names,
                [
                    "nina.queue.add",
                    "nina.queue.list",
                    "nina.route",
                    "nina.register",
                    "nina.registry.retire",
                    "nina.registry.get",
                    "nina.registry.list",
                    "nina.log.verify",
                    "nina.train.tick",
                    "nina.eval",
                    "nina.think",
                ]
            );
            for tool in tools {
                let available = tool["available"].as_bool().unwrap();
                // nina.think is the one declared-but-unavailable tool left.
                let expected = tool["name"] != "nina.think";
                assert_eq!(available, expected, "{tool}");
                if !available {
                    assert!(!tool["reason"].as_str().unwrap().is_empty(), "{tool}");
                }
            }

            // tools/call: register, then read it back, then verify the chain.
            let response = exchange(
                &mut writer,
                &mut reader,
                &json!({
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "nina.register", "arguments": {"entry": {
                        "name": "coding/rust", "field": "coding", "subfield": "rust",
                        "box_id": "coding:rust",
                    }}},
                }),
            );
            assert_eq!(response["result"]["isError"], false, "{response}");

            let response = exchange(
                &mut writer,
                &mut reader,
                &json!({
                    "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "nina.registry.get", "arguments": {"name": "coding/rust"}},
                }),
            );
            assert_eq!(response["result"]["isError"], false, "{response}");
            let text = response["result"]["content"][0]["text"].as_str().unwrap();
            let got: Value = serde_json::from_str(text).unwrap();
            assert_eq!(got["name"], "coding/rust");
            assert_eq!(got["status"], "candidate");

            let log_path = dir.join("logs").join("registry.jsonl");
            let response = exchange(
                &mut writer,
                &mut reader,
                &json!({
                    "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "nina.log.verify", "arguments": {"path": log_path}},
                }),
            );
            assert_eq!(response["result"]["isError"], false, "{response}");
            let text = response["result"]["content"][0]["text"].as_str().unwrap();
            let report: Value = serde_json::from_str(text).unwrap();
            assert_eq!(report["ok"], true);
            assert_eq!(report["lines"], 1);

            // The queue: add, then list.
            let response = exchange(
                &mut writer,
                &mut reader,
                &json!({
                    "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                    "params": {"name": "nina.queue.add", "arguments": {
                        "name": "pentest/recon", "field": "pentest",
                        "dataset_path": "/srv/m-sda/unifur/pentestcorpus/dataset/all.jsonl",
                        "steps": 120,
                    }},
                }),
            );
            assert_eq!(response["result"]["isError"], false, "{response}");
            let response = exchange(
                &mut writer,
                &mut reader,
                &json!({
                    "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": "nina.queue.list", "arguments": {}},
                }),
            );
            let text = response["result"]["content"][0]["text"].as_str().unwrap();
            let queue: Value = serde_json::from_str(text).unwrap();
            assert_eq!(queue["count"], 1);
            assert_eq!(queue["pending"][0]["entry"]["name"], "pentest/recon");

            // route with no checkpoint anywhere: the honest fallback.
            let response = exchange(
                &mut writer,
                &mut reader,
                &json!({
                    "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                    "params": {"name": "nina.route", "arguments": {"prompt": "fn main() {}"}},
                }),
            );
            assert_eq!(response["result"]["isError"], false, "{response}");
            let text = response["result"]["content"][0]["text"].as_str().unwrap();
            let routed: Value = serde_json::from_str(text).unwrap();
            assert_eq!(routed["router"], Value::Null);
            assert!(routed["note"]
                .as_str()
                .unwrap()
                .contains("no trained router"));
            assert_eq!(routed["fields"], json!(["coding"]));

            drop(writer);
            drop(reader);
            handle
                .join()
                .map_err(|_| "the MCP server thread panicked")
                .unwrap()
                .unwrap();
        });
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mcp_malformed_json_gets_a_parse_error() {
        let dir = scratch("mcp-parse");
        let server = McpServer::new(NinaPaths::new(dir.clone()));
        let response = server.handle_message("{not json").unwrap();
        assert_eq!(response["error"]["code"], -32700);
        assert_eq!(response["id"], Value::Null);
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("{not json"),
            "{response}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mcp_unknown_method_gets_method_not_found() {
        let dir = scratch("mcp-method");
        let server = McpServer::new(NinaPaths::new(dir.clone()));
        let response = server
            .handle_message(r#"{"jsonrpc":"2.0","id":9,"method":"nina.nope"}"#)
            .unwrap();
        assert_eq!(response["error"]["code"], -32601);
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("nina.nope"),
            "{response}"
        );
        // A notification (no id) is never answered, even an unknown one.
        assert!(server
            .handle_message(r#"{"jsonrpc":"2.0","method":"nina.nope"}"#)
            .is_none());
        // No method at all is an invalid request naming what arrived.
        let response = server
            .handle_message(r#"{"jsonrpc":"2.0","id":3}"#)
            .unwrap();
        assert_eq!(response["error"]["code"], -32600);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Retiring withdraws a claim and says why, in the chain.
    ///
    /// The reason matters as much as the flip: `coding/c` sat in the registry
    /// at 0.848, measured against 3000 rows carrying four distinct
    /// completions. A status change with no recorded justification leaves a
    /// reader unable to tell a withdrawn false claim from a routine
    /// housekeeping retirement.
    #[test]
    fn retiring_a_model_records_the_withdrawn_accuracy_and_the_reason() {
        let dir = scratch("mcp-retire");
        let server = McpServer::new(NinaPaths::new(dir.clone()));
        server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":
                "nina.register","arguments":{"entry":{"name":"coding/c","field":"coding",
                "subfield":"c","box_id":"coding:c","checkpoint_hash":"aa","bytes":1,
                "eval":{"accuracy":0.848125340599455,"router_top1":1.0,"date":"2026-10-14"}}}}}"#,
            )
            .unwrap();

        let out = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":
                "nina.registry.retire","arguments":{"name":"coding/c",
                "reason":"measured against four distinct completions"}}}"#,
            )
            .unwrap();
        assert!(out["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("retired"));

        let file = RegistryFile::load(&NinaPaths::new(dir.clone()).registry_json()).unwrap();
        let entry = file.models.iter().find(|m| m.name == "coding/c").unwrap();
        assert_eq!(entry.status, ModelStatus::Retired);
        assert!(
            entry.eval.accuracy.is_some(),
            "retiring withdraws the model, not the record of what was measured"
        );

        // The chain carries both the flip and the justification.
        let log = std::fs::read_to_string(dir.join("logs").join("registry.jsonl")).unwrap();
        assert!(log.contains("retire.reason"), "{log}");
        assert!(log.contains("four distinct completions"), "{log}");
        assert!(log.contains("0.848125340599455"), "{log}");

        // Retiring twice is refused rather than silently accepted.
        let again = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":
                "nina.registry.retire","arguments":{"name":"coding/c","reason":"again"}}}"#,
            )
            .unwrap();
        // MCP reports a refused tool call as a result carrying `isError`, not
        // as a JSON-RPC error — so the refusal to look for is there.
        assert_eq!(
            again["result"]["isError"], true,
            "a second retirement must not pass: {again}"
        );
        assert!(
            again["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("already retired"),
            "{again}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mcp_tools_list_is_exactly_the_v1_set() {
        let dir = scratch("mcp-tools");
        let server = McpServer::new(NinaPaths::new(dir.clone()));
        let response = server
            .handle_message(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
            .unwrap();
        let tools = response["result"]["tools"].as_array().unwrap();
        // 11: `nina.registry.retire` joined the ten. The count is asserted
        // rather than left open so a tool cannot appear or vanish unnoticed —
        // this list is the server's published contract.
        assert_eq!(tools.len(), 11);
        let available: Vec<&str> = tools
            .iter()
            .filter(|t| t["available"] == true)
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            available,
            [
                "nina.queue.add",
                "nina.queue.list",
                "nina.route",
                "nina.register",
                "nina.registry.retire",
                "nina.registry.get",
                "nina.registry.list",
                "nina.log.verify",
                "nina.train.tick",
                "nina.eval",
            ]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mcp_unknown_and_unavailable_tools_are_loud() {
        let dir = scratch("mcp-unknown-tool");
        let server = McpServer::new(NinaPaths::new(dir.clone()));
        // Unknown tool: a protocol-level invalid-params naming the tool.
        let response = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"nina.nope"}}"#,
            )
            .unwrap();
        assert_eq!(response["error"]["code"], -32602);
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("nina.nope"),
            "{response}"
        );
        // Declared-but-unavailable: a tool-level error carrying the reason.
        let response = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"nina.think","arguments":{}}}"#,
            )
            .unwrap();
        assert_eq!(response["result"]["isError"], true);
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not available in v1"), "{text}");
        assert!(text.contains("dblocks student think"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // -- the scheduler -------------------------------------------------------

    fn tiny_student_config() -> StudentConfig {
        serde_json::from_value(json!({
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 2,
            "num_kv_heads": 1,
            "context": 24,
            "intermediate_size": 16,
            "cond_hidden_size": 8,
            "frequency_embedding_size": 8,
            "attention": "dense",
            "experts_per_box": 1,
            "geom_slots": 2,
            "geom_dim": 4,
            "geom_refine_steps": 1,
            "geom_cond_hidden": 8,
            "geom_frequency": 4,
        }))
        .unwrap()
    }

    fn tiny_scheduler() -> SchedulerConfig {
        SchedulerConfig {
            student: tiny_student_config(),
            batch_size: 2,
            ..SchedulerConfig::default()
        }
    }

    fn rust_traces(dir: &Path) -> PathBuf {
        let path = dir.join("traces.jsonl");
        let pairs = [
            ("fn add(", ") -> i32 { 3 }"),
            ("let x =", " 1 + 2;"),
            ("impl A {", " fn b(&self) {} }"),
            ("fn main() {", " println!(\"hi\"); }"),
            ("struct P {", " x: i32 }"),
            ("match v {", " _ => 0 }"),
        ];
        let lines: Vec<String> = pairs
            .iter()
            .enumerate()
            .map(|(i, (prompt, completion))| {
                json!({
                    "id": format!("trace-{i}"),
                    "language": "rust",
                    "prompt": prompt,
                    "completion": completion,
                    "source": "fixture",
                })
                .to_string()
            })
            .collect();
        std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
        path
    }

    fn queue_add(root: &Path, name: &str, field: &str, dataset: &Path, steps: u64) {
        let paths = NinaPaths::new(root);
        let mut log = ChainedLog::open(paths.queue_log()).unwrap();
        log.append(&json!({
            "event": "queue.add",
            "name": name,
            "field": field,
            "dataset_path": dataset,
            "steps": steps,
            "status": "pending",
        }))
        .unwrap();
    }

    #[test]
    fn scheduler_trains_a_queued_slice_and_registers_the_candidate() {
        let root = scratch("scheduler-train");
        let dataset = rust_traces(&root);
        queue_add(&root, "coding/rust", "coding", &dataset, 3);

        let report = run_queue(&root, &tiny_scheduler()).unwrap();
        assert_eq!((report.drained, report.trained, report.failed), (1, 1, 0));
        let outcome = &report.outcomes[0];
        assert!(outcome.ok, "{outcome:?}");
        assert_eq!(outcome.name, "coding/rust");
        assert_eq!(outcome.steps_taken, 3);
        let checkpoint = outcome.checkpoint.clone().unwrap();
        // The folder layout: <root>/models/<field>/<subfield>/, content-addressed.
        assert_eq!(
            checkpoint.parent().unwrap(),
            root.join("models").join("coding").join("rust")
        );
        assert!(checkpoint
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("student-"));
        let accuracy = outcome.accuracy.unwrap();
        assert!((0.0..=1.0).contains(&accuracy), "{accuracy}");
        let router_top1 = outcome.router_top1.unwrap();
        assert!((0.0..=1.0).contains(&router_top1), "{router_top1}");

        // The deployment sidecar names *this* checkpoint and *this*
        // architecture. It used to be written once by the trace pipeline, so a
        // retrain left it naming a checkpoint that had been replaced — which is
        // what made six deployed specialists unservable, and left
        // `student think` failing on a routing-box shape mismatch.
        let sidecar = checkpoint.parent().unwrap().join("config.json");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&sidecar).unwrap()).unwrap();
        assert_eq!(
            manifest.get("checkpoint").and_then(|v| v.as_str()),
            checkpoint.file_name().and_then(|n| n.to_str()),
            "the sidecar must name the checkpoint that was just written"
        );
        assert_eq!(
            manifest
                .get("student_config")
                .and_then(|c| c.get("hidden_size"))
                .and_then(|v| v.as_u64()),
            Some(tiny_scheduler().student.hidden_size as u64),
            "the sidecar must carry the architecture this run trained"
        );

        // Registration: a Candidate carrying the measured evidence.
        let paths = NinaPaths::new(&root);
        let registry = Registry::open(paths.registry_json(), paths.registry_log()).unwrap();
        let entry = registry.get("coding/rust").unwrap();
        assert_eq!(entry.status, ModelStatus::Candidate);
        assert_eq!(entry.box_id, "coding:rust");
        assert_eq!(entry.eval.accuracy, Some(accuracy));
        assert_eq!(entry.eval.router_top1, Some(router_top1));
        assert!(!entry.eval.date.is_empty());
        assert_eq!(entry.checkpoint_hash.len(), 64);
        assert!(entry.bytes > 0);
        assert_eq!(entry.datasets, vec![dataset.to_str().unwrap().to_string()]);

        // The queue chain: add -> start -> done, verifiable.
        let report_chain = ChainedLog::verify(&paths.queue_log()).unwrap();
        assert_eq!(report_chain.lines, 3);
        let text = std::fs::read_to_string(paths.queue_log()).unwrap();
        let events: Vec<String> = text
            .lines()
            .map(|line| {
                serde_json::from_str::<Value>(line).unwrap()["payload"]["event"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(events, ["queue.add", "queue.start", "queue.done"]);
        ChainedLog::verify(&paths.registry_log()).unwrap();

        // Consumed entries stay consumed: a second pass drains nothing.
        let second = run_queue(&root, &tiny_scheduler()).unwrap();
        assert_eq!(second.drained, 0);

        // Eval determinism: same checkpoint, same split, same numbers.
        let eval_a = eval_from_checkpoint(
            &checkpoint,
            &dataset,
            Some(CodeLanguage::Rust),
            &tiny_scheduler(),
        )
        .unwrap();
        let eval_b = eval_from_checkpoint(
            &checkpoint,
            &dataset,
            Some(CodeLanguage::Rust),
            &tiny_scheduler(),
        )
        .unwrap();
        assert_eq!(eval_a.accuracy, eval_b.accuracy);
        assert_eq!(eval_a.router_top1, eval_b.router_top1);
        // And they are the numbers the run registered.
        assert_eq!(eval_a.accuracy, Some(accuracy));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The generic tick (`train_tick_with`) instantiated with the CPU
    /// backend is the same run `train_tick` performs: the outcome trains and
    /// registers, and the queue settles. This is the coverage CI gives the
    /// backend-generic path without a GPU.
    #[test]
    fn train_tick_with_runs_one_entry_on_an_explicit_backend() {
        let root = scratch("train-tick-with");
        let dataset = rust_traces(&root);
        queue_add(&root, "coding/rust", "coding", &dataset, 2);

        let outcome = train_tick_with::<TrainBackend>(&root, &tiny_scheduler())
            .unwrap()
            .unwrap();
        assert!(outcome.ok, "{outcome:?}");
        assert_eq!(outcome.name, "coding/rust");
        assert_eq!(outcome.steps_taken, 2);
        // The entry is settled: a second tick on either spelling drains nothing.
        assert!(train_tick(&root, &tiny_scheduler()).unwrap().is_none());
        let paths = NinaPaths::new(&root);
        ChainedLog::verify(&paths.queue_log()).unwrap();
        let registry = Registry::open(paths.registry_json(), paths.registry_log()).unwrap();
        assert!(registry.get("coding/rust").is_some());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn scheduler_config_field_boxes_parse_with_a_default() {
        // A config written before field boxes existed parses unchanged.
        let parsed: SchedulerConfig = serde_json::from_str("{}").unwrap();
        assert!(parsed.field_boxes.is_empty());
        assert_eq!(parsed, SchedulerConfig::default());
        // The field round-trips, feeds the student a run trains, and is
        // validated like any other name-bearing config.
        let config = SchedulerConfig {
            field_boxes: vec!["pentest".to_string()],
            ..SchedulerConfig::default()
        };
        let text = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<SchedulerConfig>(&text).unwrap(),
            config
        );
        let student = config.student_for_run();
        assert_eq!(student.field_boxes, vec!["pentest".to_string()]);
        // A student-level entry is kept, the scheduler's appended, no dupes.
        let merged = SchedulerConfig {
            student: StudentConfig {
                field_boxes: vec!["dfir".to_string()],
                ..StudentConfig::default()
            },
            field_boxes: vec!["pentest".to_string(), "dfir".to_string()],
            ..SchedulerConfig::default()
        };
        assert_eq!(
            merged.student_for_run().field_boxes,
            vec!["dfir".to_string(), "pentest".to_string()]
        );
        let bad = SchedulerConfig {
            field_boxes: vec!["Pen Test".to_string()],
            ..SchedulerConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    /// Pair-file-contract traces for a non-coding field: prompt, completion,
    /// field — no language key.
    fn field_traces(dir: &Path, field: &str) -> PathBuf {
        let path = dir.join(format!("{field}-traces.jsonl"));
        let lines: Vec<String> = (0..6)
            .map(|i| {
                json!({
                    "prompt": format!("task {i}:"),
                    "completion": format!(" scan the subnet, log the banners {i}"),
                    "field": field,
                })
                .to_string()
            })
            .collect();
        std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
        path
    }

    #[test]
    fn scheduler_trains_a_field_specialist_and_routes_to_its_box() {
        let root = scratch("scheduler-field");
        let dataset = field_traces(&root, "pentest");
        queue_add(&root, "pentest/recon", "pentest", &dataset, 3);

        let scheduler = SchedulerConfig {
            field_boxes: vec!["pentest".to_string()],
            ..tiny_scheduler()
        };
        let report = run_queue(&root, &scheduler).unwrap();
        assert_eq!((report.drained, report.trained, report.failed), (1, 1, 0));
        let outcome = &report.outcomes[0];
        assert!(outcome.ok, "{outcome:?}");
        assert_eq!(outcome.name, "pentest/recon");
        // The field box was supervised during training, so the held-out
        // router top-1 is measured, not None.
        let router_top1 = outcome.router_top1.unwrap();
        assert!((0.0..=1.0).contains(&router_top1), "{router_top1}");

        // Registration: the box id names the real MoSME box.
        let paths = NinaPaths::new(&root);
        let registry = Registry::open(paths.registry_json(), paths.registry_log()).unwrap();
        let entry = registry.get("pentest/recon").unwrap();
        assert_eq!(entry.box_id, "field:pentest");
        assert_eq!(entry.eval.router_top1, Some(router_top1));

        // Eval determinism from disk: the checkpoint's sidecar carries the
        // merged field_boxes, so the re-eval routes to the same box.
        let checkpoint = outcome.checkpoint.clone().unwrap();
        let eval_a = eval_from_checkpoint(&checkpoint, &dataset, None, &scheduler).unwrap();
        let eval_b = eval_from_checkpoint(&checkpoint, &dataset, None, &scheduler).unwrap();
        assert_eq!(eval_a.router_top1, Some(router_top1));
        assert_eq!(eval_a.router_top1, eval_b.router_top1);
        assert_eq!(eval_a.accuracy, outcome.accuracy);

        // A scheduler that does not route the field keeps the old contract:
        // same dataset, no box, router_top1 stays None.
        let root2 = scratch("scheduler-field-unrouted");
        let dataset2 = field_traces(&root2, "pentest");
        queue_add(&root2, "pentest/recon", "pentest", &dataset2, 3);
        let report2 = run_queue(&root2, &tiny_scheduler()).unwrap();
        assert_eq!((report2.drained, report2.trained), (1, 1));
        assert_eq!(report2.outcomes[0].router_top1, None);

        std::fs::remove_dir_all(&root).unwrap();
        std::fs::remove_dir_all(&root2).unwrap();
    }

    #[test]
    fn scheduler_names_a_missing_dataset_and_continues() {
        let root = scratch("scheduler-missing");
        let missing = root.join("absent.jsonl");
        let dataset = rust_traces(&root);
        queue_add(&root, "coding/rust", "coding", &missing, 2);
        // A good entry behind the bad one proves the queue continues.
        queue_add(&root, "coding/rust", "coding", &dataset, 2);
        let report = run_queue(&root, &tiny_scheduler()).unwrap();
        assert_eq!(report.drained, 2);
        assert_eq!((report.trained, report.failed), (1, 1));
        // The first entry failed loudly, naming the missing dataset.
        let first = &report.outcomes[0];
        assert!(!first.ok);
        assert_eq!(first.name, "coding/rust");
        assert!(
            first.error.as_ref().unwrap().contains("absent.jsonl"),
            "{first:?}"
        );
        // The second trained and registered.
        assert!(report.outcomes[1].ok, "{:?}", report.outcomes[1]);
        let registry = Registry::open(
            NinaPaths::new(&root).registry_json(),
            NinaPaths::new(&root).registry_log(),
        )
        .unwrap();
        assert_eq!(
            registry.get("coding/rust").unwrap().status,
            ModelStatus::Candidate
        );
        // And the queue chain records the failure, verifiably.
        let paths = NinaPaths::new(&root);
        ChainedLog::verify(&paths.queue_log()).unwrap();
        let text = std::fs::read_to_string(paths.queue_log()).unwrap();
        assert!(text.contains("queue.failed"), "{text}");
        assert!(text.contains("absent.jsonl"), "{text}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The pre-training scan: planted cheat lines are dropped, counted per
    /// class into the outcome and the queue.done payload, and the run trains
    /// on the clean lines only.
    #[test]
    fn scheduler_drops_planted_cheat_lines_and_counts_them() {
        let root = scratch("scheduler-cheat-scan");
        let dataset = rust_traces(&root);
        // Three lines the corpus builders would have rejected: an unreasoned
        // lint suppression, a stub body, and a live-looking API key.
        let mut text = std::fs::read_to_string(&dataset).unwrap();
        for (prompt, completion) in [
            ("fn evil(", ") { #[allow(dead_code)] fn unused() {} }"),
            ("fn later(", ") { todo!() }"),
            (
                "fn keys(",
                ") { let k = \"sk-abcdefghij1234567890abcdef\"; }",
            ),
        ] {
            text.push_str(
                &json!({"language": "rust", "prompt": prompt, "completion": completion})
                    .to_string(),
            );
            text.push('\n');
        }
        std::fs::write(&dataset, text).unwrap();
        queue_add(&root, "coding/rust", "coding", &dataset, 3);

        let report = run_queue(&root, &tiny_scheduler()).unwrap();
        assert_eq!((report.drained, report.trained, report.failed), (1, 1, 0));
        let outcome = &report.outcomes[0];
        assert!(outcome.ok, "{outcome:?}");
        assert_eq!(outcome.cheat_scan.scanned, 9);
        assert_eq!(
            (outcome.cheat_scan.kept, outcome.cheat_scan.dropped),
            (6, 3)
        );
        assert_eq!(
            outcome.cheat_scan.dropped_by_class.get("data-suppression"),
            Some(&1)
        );
        assert_eq!(
            outcome.cheat_scan.dropped_by_class.get("data-stub"),
            Some(&1)
        );
        assert_eq!(
            outcome.cheat_scan.dropped_by_class.get("data-secrets"),
            Some(&1)
        );
        // Only the six clean lines were split into folds.
        assert_eq!(outcome.train_lines + outcome.holdout_lines, 6);

        // The counts are part of the queue.done payload; the chain verifies.
        let paths = NinaPaths::new(&root);
        ChainedLog::verify(&paths.queue_log()).unwrap();
        let text = std::fs::read_to_string(paths.queue_log()).unwrap();
        let done = text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap()["payload"].clone())
            .find(|payload| payload["event"] == "queue.done")
            .unwrap();
        assert_eq!(done["cheat_scanned"], 9);
        assert_eq!(done["cheat_dropped"]["data-suppression"], 1);
        assert_eq!(done["cheat_dropped"]["data-stub"], 1);
        assert_eq!(done["cheat_dropped"]["data-secrets"], 1);

        // nina.eval re-derives the numbers from the same cleaned split.
        let checkpoint = outcome.checkpoint.clone().unwrap();
        let eval = eval_from_checkpoint(
            &checkpoint,
            &dataset,
            Some(CodeLanguage::Rust),
            &tiny_scheduler(),
        )
        .unwrap();
        assert_eq!(eval.accuracy, outcome.accuracy);
        assert_eq!(eval.router_top1, outcome.router_top1);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A dataset the scan empties fails the entry loudly, naming the dataset
    /// and the classes — nothing trains on an empty dataset.
    #[test]
    fn scheduler_refuses_an_all_cheat_dataset_naming_the_path() {
        let root = scratch("scheduler-all-cheat");
        let dataset = root.join("poisoned.jsonl");
        let lines: Vec<String> = [
            ("fn a(", ") { todo!() }"),
            ("fn b(", ") { #[allow(dead_code)] fn f() {} }"),
            ("fn c(", ") { let k = \"sk-abcdefghij1234567890abcdef\"; }"),
        ]
        .iter()
        .map(|(prompt, completion)| {
            json!({"language": "rust", "prompt": prompt, "completion": completion}).to_string()
        })
        .collect();
        std::fs::write(&dataset, format!("{}\n", lines.join("\n"))).unwrap();
        queue_add(&root, "coding/rust", "coding", &dataset, 2);

        let report = run_queue(&root, &tiny_scheduler()).unwrap();
        assert_eq!((report.drained, report.trained, report.failed), (1, 0, 1));
        let outcome = &report.outcomes[0];
        assert!(!outcome.ok);
        let error = outcome.error.as_ref().unwrap();
        assert!(error.contains("poisoned.jsonl"), "{error}");
        // The refusal names the gate and every reason it gave. Naming the
        // detector it used to be ("cheat scan") is not enough now that the
        // same refusal can also come from the loader, and a message that
        // pointed at only one of the two would send the reader to the wrong
        // file when the other one was what emptied it.
        assert!(error.contains("cheat gate"), "{error}");
        assert!(error.contains("dropped every one of its 3 rows"), "{error}");
        assert!(outcome.checkpoint.is_none());

        // The failure is in the chain, and the chain verifies.
        let paths = NinaPaths::new(&root);
        ChainedLog::verify(&paths.queue_log()).unwrap();
        let text = std::fs::read_to_string(paths.queue_log()).unwrap();
        assert!(text.contains("queue.failed"), "{text}");
        assert!(text.contains("poisoned.jsonl"), "{text}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A dataset in a format the scan cannot read fails loudly, as the
    /// loader made it fail before: a non-JSONL file is not scannable.
    #[test]
    fn scheduler_fails_loudly_on_a_non_jsonl_dataset() {
        let root = scratch("scheduler-non-jsonl");
        let dataset = root.join("not-jsonl.txt");
        std::fs::write(&dataset, "this is not json\nneither is this\n").unwrap();
        queue_add(&root, "coding/rust", "coding", &dataset, 2);

        let report = run_queue(&root, &tiny_scheduler()).unwrap();
        assert_eq!((report.drained, report.trained, report.failed), (1, 0, 1));
        let error = report.outcomes[0].error.as_ref().unwrap();
        assert!(error.contains("not-jsonl.txt"), "{error}");
        assert!(error.contains("not valid JSON"), "{error}");
        assert!(error.contains("cheat scan"), "{error}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A queue.done record written before the cheat-scan fields existed: the
    /// chain verifies, the entry it settles stays settled, and the summary
    /// deserializes from partial or absent data (every field is
    /// #[serde(default)]).
    #[test]
    fn old_queue_done_records_without_cheat_fields_still_parse() {
        let root = scratch("queue-done-backcompat");
        let paths = NinaPaths::new(&root);
        {
            let mut log = ChainedLog::open(paths.queue_log()).unwrap();
            log.append(&json!({
                "event": "queue.add",
                "name": "coding/rust",
                "field": "coding",
                "dataset_path": "/tmp/x.jsonl",
                "steps": 3,
                "status": "pending",
            }))
            .unwrap();
            // The pre-scan shape: no cheat_scanned, no cheat_dropped.
            log.append(&json!({
                "event": "queue.done",
                "queue_seq": 0,
                "name": "coding/rust",
                "accuracy": 0.5,
                "steps_taken": 3,
                "train_lines": 5,
                "holdout_lines": 1,
            }))
            .unwrap();
        }
        ChainedLog::verify(&paths.queue_log()).unwrap();
        assert!(pending_queue_entries(&paths).unwrap().is_empty());

        // Partial or absent scan data defaults field by field.
        assert_eq!(
            serde_json::from_value::<CheatScanSummary>(json!({})).unwrap(),
            CheatScanSummary::default()
        );
        let partial: CheatScanSummary = serde_json::from_value(json!({
            "scanned": 7,
            "dropped_by_class": {"data-stub": 2},
        }))
        .unwrap();
        assert_eq!(partial.scanned, 7);
        assert_eq!((partial.kept, partial.dropped), (0, 0));
        assert_eq!(partial.dropped_by_class.get("data-stub"), Some(&2));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A deployed checkpoint travels without its training-state sidecar: the
    /// student config comes from the `config.json` manifest beside it, and
    /// only a dir with neither falls back to the default.
    #[test]
    fn student_config_falls_back_to_the_deployment_manifest() {
        let dir = scratch("config-fallback");
        let model = dir.join("student-deadbeef.mpk");
        std::fs::write(&model, b"").unwrap();
        // No sidecar, no manifest: the default.
        assert_eq!(
            student_config_for_checkpoint(&model).unwrap(),
            StudentConfig::default()
        );
        // A manifest with a student_config: that config, not the default.
        std::fs::write(
            dir.join("config.json"),
            json!({
                "name": "coding/rust",
                "checkpoint": "student-deadbeef.mpk",
                "student_config": {"context": 2048, "hidden_size": 64},
            })
            .to_string(),
        )
        .unwrap();
        let config = student_config_for_checkpoint(&model).unwrap();
        assert_eq!(config.context, 2048);
        assert_eq!(config.hidden_size, 64);
        // A manifest without a student_config key: the default again.
        std::fs::write(
            dir.join("config.json"),
            json!({"name": "coding/rust"}).to_string(),
        )
        .unwrap();
        assert_eq!(
            student_config_for_checkpoint(&model).unwrap(),
            StudentConfig::default()
        );
        // A corrupt manifest is loud, not a silent default.
        std::fs::write(dir.join("config.json"), b"not json").unwrap();
        let err = student_config_for_checkpoint(&model)
            .unwrap_err()
            .to_string();
        assert!(err.contains("config.json"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn promotion_gate_admits_only_measured_improvement() {
        let root = scratch("promotion-gate");
        let paths = NinaPaths::new(&root);
        let mut registry = Registry::open(paths.registry_json(), paths.registry_log()).unwrap();
        let mut candidate = entry("coding/rust", "coding", "rust");
        candidate.eval.accuracy = Some(0.80);
        candidate.eval.date = "2026-10-10".to_string();
        registry.register(candidate).unwrap();

        // First promotion: no incumbent, evidence suffices.
        let verdict = registry.promote_if_better("coding/rust", 0.0).unwrap();
        assert!(
            matches!(verdict, PromotionVerdict::Promoted { .. }),
            "{verdict:?}"
        );
        assert_eq!(
            registry.get("coding/rust").unwrap().status,
            ModelStatus::Active
        );

        // A planted regression: 0.75 does not beat the 0.80 baseline.
        registry
            .record_eval(
                "coding/rust",
                EvalEvidence {
                    accuracy: Some(0.75),
                    ..EvalEvidence::default()
                },
                None,
                None,
                None,
            )
            .unwrap();
        let verdict = registry.promote_if_better("coding/rust", 0.0).unwrap();
        match verdict {
            PromotionVerdict::Rejected {
                candidate,
                baseline,
                margin,
                ..
            } => {
                assert_eq!((candidate, baseline, margin), (0.75, 0.80, 0.0));
            }
            PromotionVerdict::Promoted { .. } => unreachable!("a regression must be rejected"),
        }

        // A tie is not an improvement either.
        registry
            .record_eval(
                "coding/rust",
                EvalEvidence {
                    accuracy: Some(0.80),
                    ..EvalEvidence::default()
                },
                None,
                None,
                None,
            )
            .unwrap();
        let verdict = registry.promote_if_better("coding/rust", 0.0).unwrap();
        assert!(
            matches!(verdict, PromotionVerdict::Rejected { .. }),
            "{verdict:?}"
        );

        // The margin bites: 0.84 < 0.80 + 0.05 is rejected; 0.86 clears it.
        registry
            .record_eval(
                "coding/rust",
                EvalEvidence {
                    accuracy: Some(0.84),
                    ..EvalEvidence::default()
                },
                None,
                None,
                None,
            )
            .unwrap();
        let verdict = registry.promote_if_better("coding/rust", 0.05).unwrap();
        assert!(
            matches!(verdict, PromotionVerdict::Rejected { .. }),
            "{verdict:?}"
        );
        registry
            .record_eval(
                "coding/rust",
                EvalEvidence {
                    accuracy: Some(0.86),
                    ..EvalEvidence::default()
                },
                None,
                None,
                None,
            )
            .unwrap();
        let verdict = registry.promote_if_better("coding/rust", 0.05).unwrap();
        match verdict {
            PromotionVerdict::Promoted {
                accuracy, baseline, ..
            } => {
                assert_eq!(accuracy, 0.86);
                assert_eq!(baseline, Some(0.80));
            }
            PromotionVerdict::Rejected { .. } => unreachable!("0.86 beats 0.80 + 0.05"),
        }

        // Every decision is in the chain, and the chain verifies.
        ChainedLog::verify(&paths.registry_log()).unwrap();
        let text = std::fs::read_to_string(paths.registry_log()).unwrap();
        let events: Vec<String> = text
            .lines()
            .map(|line| {
                serde_json::from_str::<Value>(line).unwrap()["payload"]["event"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(
            events,
            [
                "register",
                "promote",
                "eval",
                "promotion_rejected",
                "eval",
                "promotion_rejected",
                "eval",
                "promotion_rejected",
                "eval",
                "promote",
            ]
        );

        // The gate's refusals are loud for unknown names and evidence-less
        // entries, and a negative margin is nonsense.
        assert!(registry.promote_if_better("bio/general", 0.0).is_err());
        let mut unevaled = entry("pentest/recon", "pentest", "recon");
        unevaled.eval = EvalEvidence::default();
        registry.register(unevaled).unwrap();
        let err = registry
            .promote_if_better("pentest/recon", 0.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("eval accuracy"), "{err}");
        assert!(registry.promote_if_better("coding/rust", -0.1).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A retrain on new data must not leave the entry citing the v1 dataset:
    /// `record_eval` carries the replacement dataset list into both the chain
    /// payload and the derived entry (drain-3 left stale `pipeline-real`
    /// paths behind when it retrained on `pipeline-offline`).
    #[test]
    fn record_eval_updates_datasets_on_retrain() {
        let root = scratch("retrain-datasets");
        let paths = NinaPaths::new(root.clone());
        let mut registry = Registry::open(paths.registry_json(), paths.registry_log()).unwrap();
        let mut e = entry("coding/rust", "coding", "rust");
        e.datasets = vec!["/data/v1.jsonl".to_string()];
        registry.register(e).unwrap();

        registry
            .record_eval(
                "coding/rust",
                EvalEvidence {
                    accuracy: Some(0.5),
                    ..EvalEvidence::default()
                },
                Some("a".repeat(64)),
                Some(42),
                Some(vec!["/data/v2.jsonl".to_string()]),
            )
            .unwrap();

        let entry = registry.get("coding/rust").unwrap();
        assert_eq!(entry.datasets, vec!["/data/v2.jsonl".to_string()]);
        let text = std::fs::read_to_string(paths.registry_log()).unwrap();
        let last = text.lines().last().unwrap();
        let payload = serde_json::from_str::<Value>(last).unwrap();
        assert_eq!(
            payload["payload"]["datasets"],
            json!(["/data/v2.jsonl"]),
            "the chain must carry the retrain's dataset, not just the hash"
        );
        ChainedLog::verify(&paths.registry_log()).unwrap();

        // A plain re-eval (no datasets argument) leaves provenance alone.
        registry
            .record_eval(
                "coding/rust",
                EvalEvidence {
                    accuracy: Some(0.51),
                    ..EvalEvidence::default()
                },
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            registry.get("coding/rust").unwrap().datasets,
            vec!["/data/v2.jsonl".to_string()]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn mcp_audit_log_records_every_tools_call() {
        let root = scratch("mcp-audit");
        let paths = NinaPaths::new(root.clone());
        let server = McpServer::new(NinaPaths::new(root.clone()));
        // One successful call and one failing call; both must be audited.
        let ok = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"nina.queue.add","arguments":{"name":"coding/rust","field":"coding","dataset_path":"/tmp/x.jsonl","steps":5}}}"#,
            )
            .unwrap();
        assert_eq!(ok["result"]["isError"], false);
        let err = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"nina.registry.get","arguments":{"name":"bio/general"}}}"#,
            )
            .unwrap();
        assert_eq!(err["result"]["isError"], true);

        let report = ChainedLog::verify(&paths.mcp_log()).unwrap();
        assert_eq!(report.lines, 2);
        let text = std::fs::read_to_string(paths.mcp_log()).unwrap();
        let calls: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap()["payload"].clone())
            .collect();
        assert_eq!(calls[0]["event"], "mcp.call");
        assert_eq!(calls[0]["tool"], "nina.queue.add");
        assert_eq!(calls[0]["ok"], true);
        assert_eq!(calls[0]["arguments"]["steps"], 5);
        assert_eq!(calls[1]["tool"], "nina.registry.get");
        assert_eq!(calls[1]["ok"], false);
        // The registry file does not exist in this scratch root, so the
        // recorded error is the load failure naming that path.
        assert!(calls[1]["error"]
            .as_str()
            .unwrap()
            .contains("registry.json"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn mcp_train_tick_and_eval_round_trip() {
        let root = scratch("mcp-tick");
        let dataset = rust_traces(&root);
        let server = McpServer::with_scheduler(NinaPaths::new(root.clone()), tiny_scheduler());

        // Queue a slice through the protocol, then run exactly one tick.
        let add = server
            .handle_message(
                json!({
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "nina.queue.add", "arguments": {
                        "name": "coding/rust", "field": "coding",
                        "dataset_path": dataset, "steps": 3,
                    }},
                })
                .to_string()
                .as_str(),
            )
            .unwrap();
        assert_eq!(add["result"]["isError"], false, "{add}");
        let tick = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"nina.train.tick","arguments":{}}}"#,
            )
            .unwrap();
        assert_eq!(tick["result"]["isError"], false, "{tick}");
        let text = tick["result"]["content"][0]["text"].as_str().unwrap();
        let outcome: Value = serde_json::from_str(text).unwrap();
        assert_eq!(outcome["ran"], true);
        assert_eq!(outcome["outcome"]["ok"], true, "{outcome}");
        assert_eq!(outcome["outcome"]["steps_taken"], 3);
        // The pre-training cheat scan's counts ride the same outcome JSON.
        assert_eq!(outcome["outcome"]["cheat_scan"]["scanned"], 6);
        assert_eq!(outcome["outcome"]["cheat_scan"]["dropped"], 0);
        let accuracy = outcome["outcome"]["accuracy"].as_f64().unwrap();

        // The queue is drained; a second tick says so honestly.
        let tick = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"nina.train.tick","arguments":{}}}"#,
            )
            .unwrap();
        let text = tick["result"]["content"][0]["text"].as_str().unwrap();
        let drained: Value = serde_json::from_str(text).unwrap();
        assert_eq!(drained["ran"], false);

        // queue.list shows nothing pending (the old "nothing consumes it"
        // note is gone).
        let list = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"nina.queue.list","arguments":{}}}"#,
            )
            .unwrap();
        let text = list["result"]["content"][0]["text"].as_str().unwrap();
        let queue: Value = serde_json::from_str(text).unwrap();
        assert_eq!(queue["count"], 0);
        assert!(queue.get("note").is_none() || queue["note"].is_null());

        // nina.eval recomputes the evidence on demand — bit-identical to the
        // registration numbers, and lands in the chain as an `eval` record.
        let eval = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"nina.eval","arguments":{"name":"coding/rust"}}}"#,
            )
            .unwrap();
        assert_eq!(eval["result"]["isError"], false, "{eval}");
        let text = eval["result"]["content"][0]["text"].as_str().unwrap();
        let entry: Value = serde_json::from_str(text).unwrap();
        assert_eq!(entry["name"], "coding/rust");
        assert_eq!(entry["eval"]["accuracy"].as_f64().unwrap(), accuracy);

        // Every call above was audited; the audit chain verifies.
        let paths = NinaPaths::new(root.clone());
        let report = ChainedLog::verify(&paths.mcp_log()).unwrap();
        assert_eq!(report.lines, 5);
        ChainedLog::verify(&paths.queue_log()).unwrap();
        ChainedLog::verify(&paths.registry_log()).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// nina.eval with a `dataset_path` override measures that dataset instead
    /// of the recorded one; the schema declares the argument optional.
    #[test]
    fn mcp_eval_honours_a_dataset_override() {
        let root = scratch("mcp-eval-override");
        let dataset_a = rust_traces(&root);
        // A second, disjoint dataset to override with.
        let pairs = [
            ("fn sub(", ") -> i32 { 4 }"),
            ("let y =", " 3 * 3;"),
            ("impl B {", " fn c(&self) {} }"),
            ("fn idle() {", " loop { break } }"),
            ("struct Q {", " y: i32 }"),
            ("match w {", " _ => 1 }"),
        ];
        let dataset_b = root.join("override-traces.jsonl");
        let lines: Vec<String> = pairs
            .iter()
            .enumerate()
            .map(|(i, (prompt, completion))| {
                json!({
                    "id": format!("ovr-{i}"),
                    "language": "rust",
                    "prompt": prompt,
                    "completion": completion,
                    "source": "fixture",
                })
                .to_string()
            })
            .collect();
        std::fs::write(&dataset_b, format!("{}\n", lines.join("\n"))).unwrap();
        queue_add(&root, "coding/rust", "coding", &dataset_a, 3);
        let server = McpServer::with_scheduler(NinaPaths::new(root.clone()), tiny_scheduler());
        let tick = server
            .handle_message(
                r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"nina.train.tick","arguments":{}}}"#,
            )
            .unwrap();
        assert_eq!(tick["result"]["isError"], false, "{tick}");

        // tools/list declares the override: an optional string, name still required.
        let list = server
            .handle_message(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#)
            .unwrap();
        let tools = list["result"]["tools"].as_array().unwrap();
        let eval_tool = tools.iter().find(|t| t["name"] == "nina.eval").unwrap();
        assert_eq!(
            eval_tool["inputSchema"]["properties"]["dataset_path"]["type"],
            "string"
        );
        assert_eq!(eval_tool["inputSchema"]["required"], json!(["name"]));

        // The override's numbers are dataset B's numbers, recomputed directly.
        let checkpoint = crate::checkpoint::latest_in_dir(
            &NinaPaths::new(&root).model_dir("coding", "rust"),
            "student",
        )
        .unwrap()
        .unwrap();
        let want = eval_from_checkpoint(
            &checkpoint,
            &dataset_b,
            Some(CodeLanguage::Rust),
            &tiny_scheduler(),
        )
        .unwrap();
        let eval = server
            .handle_message(
                json!({
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "nina.eval", "arguments": {
                        "name": "coding/rust", "dataset_path": dataset_b,
                    }},
                })
                .to_string()
                .as_str(),
            )
            .unwrap();
        assert_eq!(eval["result"]["isError"], false, "{eval}");
        let text = eval["result"]["content"][0]["text"].as_str().unwrap();
        let entry: Value = serde_json::from_str(text).unwrap();
        assert_eq!(
            entry["eval"]["accuracy"].as_f64().unwrap(),
            want.accuracy.unwrap()
        );
        assert_eq!(
            entry["eval"]["router_top1"].as_f64().unwrap(),
            want.router_top1.unwrap()
        );

        // The audit chain records the call WITH its override argument.
        let paths = NinaPaths::new(&root);
        let text = std::fs::read_to_string(paths.mcp_log()).unwrap();
        let eval_call = text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap()["payload"].clone())
            .find(|payload| payload["tool"] == "nina.eval")
            .unwrap();
        assert_eq!(
            eval_call["arguments"]["dataset_path"],
            dataset_b.to_str().unwrap()
        );
        ChainedLog::verify(&paths.mcp_log()).unwrap();
        ChainedLog::verify(&paths.registry_log()).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The calibration loop, end to end: a drain counts what the mixing rule
    /// did, the count survives in the chain, and the profile derived from it
    /// changes what the next drain drops. Each link is the whole point —
    /// a tally that never reached the log, or a profile that never reached
    /// the scanner, would leave the loop closed at one end.
    #[test]
    fn a_drain_counts_its_mixing_pairs_the_chain_keeps_and_the_profile_changes_the_next_drain() {
        let root = scratch("mixing-loop");
        let paths = NinaPaths::new(&root);

        // A dataset of prose: `Other`-declared, so nothing fires, and the
        // tally still records how many documents it looked at.
        let dataset = root.join("prose.jsonl");
        let mut lines: Vec<String> = Vec::new();
        for i in 0..260 {
            lines.push(
                serde_json::to_string(&serde_json::json!({
                    "instruction": format!("Explain item {i}."),
                    "response": format!("Item {i} takes a self parameter and returns None.\n"),
                }))
                .unwrap(),
            );
        }
        std::fs::write(&dataset, format!("{}\n", lines.join("\n"))).unwrap();

        let config = SchedulerConfig::default();
        let (cleaned, summary) = scan_dataset_cheats(
            &dataset,
            None,
            &config.mixing_policy().unwrap(),
            config.ml_cheat().unwrap().as_ref(),
            &root,
        )
        .unwrap();
        assert_eq!(summary.scanned, 260);
        assert_eq!(summary.mixing.scanned_total(), 260);
        assert!(
            summary.mixing.fired.is_empty(),
            "prose reported a language switch: {:?}",
            summary.mixing.fired
        );
        // The whole corpus survived: the filter that used to take 97% of it
        // takes none of it, and says so in the counts rather than in prose.
        assert_eq!(summary.dropped, 0);
        assert!(cleaned.exists());

        // Into the chain, as a queue.done would carry it.
        let mut log = ChainedLog::open(paths.queue_log()).unwrap();
        log.append(&json!({
            "event": "queue.done",
            "queue_seq": 0,
            "name": "prose/prose",
            "cheat_scanned": summary.scanned,
            "cheat_mixing": summary.mixing,
        }))
        .unwrap();
        ChainedLog::verify(&paths.queue_log()).unwrap();

        let tally = mixing_tally_from_chain(&paths.queue_log()).unwrap();
        assert_eq!(tally.scanned_total(), 260);

        // A tally below the floor is refused rather than turned into a rate.
        let thin = crate::cheat::MixingTally {
            scanned_by_declared: std::collections::BTreeMap::from([("other".to_string(), 12usize)]),
            fired: std::collections::BTreeMap::from([("other->python".to_string(), 9usize)]),
        };
        let err = crate::cheat::calibrate_mixing(&thin, 0.5).unwrap_err();
        assert!(
            format!("{err:#}").contains("at least 200 scanned documents"),
            "unhelpful refusal: {err:#}"
        );

        // With the floor satisfied, the evidence supports a profile, and the
        // profile is what the next scan runs under.
        let tallied = crate::cheat::MixingTally {
            scanned_by_declared: tally.scanned_by_declared.clone(),
            fired: std::collections::BTreeMap::from([("other->python".to_string(), 258usize)]),
        };
        let adaptation = crate::cheat::calibrate_mixing(&tallied, 0.5).unwrap();
        assert_eq!(adaptation.policy.admitted_pairs.len(), 1);

        // A profile naming `other` is the one calibration can never produce —
        // `Other` is not a candidate language — so the struck pair here is a
        // hypothetical one, and the round trip is what is under test.
        let profile = root.join("policy.json");
        crate::cheat::MixingPolicy {
            admitted_pairs: adaptation.policy.admitted_pairs.clone(),
            ..crate::cheat::MixingPolicy::default()
        }
        .save(&profile)
        .unwrap();
        let with_profile = SchedulerConfig {
            mixing_policy: Some(profile),
            ..SchedulerConfig::default()
        };
        assert_eq!(
            with_profile.mixing_policy().unwrap().admitted_pairs.len(),
            1
        );

        // A configured-but-unreadable profile is an error, not a silent
        // fallback to the defaults: a run that ignored the profile it was
        // pointed at would log counts produced by different rules.
        let missing = SchedulerConfig {
            mixing_policy: Some(root.join("absent.json")),
            ..SchedulerConfig::default()
        };
        assert!(missing.mixing_policy().is_err());

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The stacking order the whole learned layer rests on, at the place it
    /// actually runs: a reject-grade pattern finding drops the line whatever
    /// the model says, and the model's score is never even computed for it.
    ///
    /// The two halves are asserted separately because they can fail
    /// independently. A learned filter that quietly overrode a stated rule
    /// would be the worst of both — the rule stops meaning anything, and
    /// nothing records that it stopped meaning anything.
    #[test]
    fn the_learned_layer_is_consulted_only_where_the_pattern_scan_found_nothing() {
        let root = scratch("ml-stacking");
        let dataset = root.join("mixed.jsonl");
        // Three lines: clean, pattern-cheating (a Rust `todo!()`), and clean.
        let rows = [
            (
                "Clean work.",
                "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
            ),
            (
                "Leave it for later.",
                "fn add(a: i32, b: i32) -> i32 {\n    todo!()\n}\n",
            ),
            (
                "Also clean.",
                "fn sub(a: i32, b: i32) -> i32 {\n    a - b\n}\n",
            ),
        ];
        let text = rows
            .iter()
            .map(|(instruction, response)| {
                serde_json::to_string(&json!({
                    "instruction": instruction,
                    "response": response,
                    "language": "rust",
                }))
                .unwrap()
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&dataset, format!("{text}\n")).unwrap();

        let mut model =
            crate::mlcheat::MlCheatModel::new(crate::mlcheat::MlCheatConfig::default()).unwrap();
        let samples = vec![
            (
                "fn add(a: i32, b: i32) -> i32 { todo!() }\n".to_string(),
                SnippetLanguage::Rust,
                true,
            ),
            (
                "fn add(a: i32, b: i32) -> i32 { a + b }\n".to_string(),
                SnippetLanguage::Rust,
                false,
            ),
        ];
        model.train(&samples).unwrap();
        let model_path = root.join("ml-cheat.json");
        model.save(&model_path).unwrap();

        // Patterns alone: the `todo!()` line goes, and the model is never
        // configured, so the summary must say so rather than reporting zero
        // learned drops.
        let (cleaned, patterns_only) = scan_dataset_cheats(
            &dataset,
            None,
            &crate::cheat::MixingPolicy::default(),
            None,
            &root,
        )
        .unwrap();
        assert_eq!(patterns_only.dropped, 1);
        assert!(patterns_only.learned.is_none());
        assert_eq!(
            std::fs::read_to_string(&cleaned).unwrap().lines().count(),
            2
        );

        // With the model at a threshold no snippet can reach, the pattern
        // verdict stands and the clean lines survive untouched.
        let (_, with_model) = scan_dataset_cheats(
            &dataset,
            None,
            &crate::cheat::MixingPolicy::default(),
            Some(&(model.clone(), 2.0)),
            &root,
        )
        .unwrap();
        assert_eq!(with_model.dropped, 1, "the model changed a pattern verdict");
        let ml = with_model
            .learned
            .expect("a configured layer must report itself");
        assert_eq!(ml.scored, 2, "the pattern-rejected line must not be scored");
        assert_eq!(ml.dropped, 0);
        assert!((ml.threshold - 2.0).abs() < f64::EPSILON);

        // At a threshold every score clears, the learned layer drops the two
        // lines the patterns passed — which empties the dataset, so the
        // refusal is the observable outcome. It has to name the learned drops:
        // a refusal that reported only `data-stub:1` for three dropped lines
        // would send whoever reads it looking at the wrong filter.
        let err = scan_dataset_cheats(
            &dataset,
            None,
            &crate::cheat::MixingPolicy::default(),
            Some(&(model, 0.0)),
            &root,
        )
        .expect_err("an emptied dataset is refused");
        let err = format!("{err:#}");
        assert!(
            err.contains("learned-second-opinion:2"),
            "the refusal under-reports the learned drops: {err}"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A dataset that is one answer repeated is refused, even though every row
    /// in it is individually clean.
    ///
    /// This is the defect that registered `coding/c` at 0.848: the offline
    /// template bank holds one reference per task kind, the pipeline was asked
    /// for 3000 tasks, and it emitted 3000 rows over 4 completions. No cheat
    /// class describes that — every row passes every pattern — so the scan
    /// reported `dropped: {}` and the specialist memorised four strings
    /// against a held-out fold drawn from the same four. The refusal names the
    /// ratio, because "refused" alone would send whoever reads it looking for
    /// a filter that fired.
    #[test]
    fn a_dataset_that_is_one_answer_repeated_is_refused() {
        let root = scratch("degenerate-dataset");
        let dataset = root.join("train.jsonl");
        let answer = "fn add(a: i32, b: i32) -> i32 { a + b }\n";
        let mut body = String::new();
        for i in 0..200 {
            body.push_str(
                &serde_json::json!({
                    "id": format!("t{i}"),
                    "language": "rust",
                    "prompt": format!("write a function that adds two numbers (variant {i})"),
                    "completion": answer,
                })
                .to_string(),
            );
            body.push('\n');
        }
        std::fs::write(&dataset, body).unwrap();

        let err = scan_dataset_cheats(
            &dataset,
            None,
            &crate::cheat::MixingPolicy::default(),
            None,
            &root.join("work"),
        )
        .expect_err("a dataset with one distinct answer must be refused");
        let err = format!("{err:#}");
        assert!(
            err.contains("distinct completion"),
            "the refusal must name what is wrong: {err}"
        );
        assert!(
            err.contains("1 of 200 kept rows"),
            "the refusal must carry the numbers: {err}"
        );

        // The same dataset with real per-row answers passes, so the guard is
        // about degeneracy and not about size.
        let mut varied = String::new();
        for i in 0..200 {
            varied.push_str(
                &serde_json::json!({
                    "id": format!("v{i}"),
                    "language": "rust",
                    "prompt": format!("write function {i}"),
                    "completion": format!("fn f{i}() -> i32 {{ {i} }}\n"),
                })
                .to_string(),
            );
            varied.push('\n');
        }
        let varied_path = root.join("varied.jsonl");
        std::fs::write(&varied_path, varied).unwrap();
        let (cleaned, summary) = scan_dataset_cheats(
            &varied_path,
            None,
            &crate::cheat::MixingPolicy::default(),
            None,
            &root.join("work2"),
        )
        .expect("200 distinct answers is a dataset");
        assert_eq!(summary.kept, 200);
        assert!(cleaned.exists());

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Four real examples are a small dataset, not a degenerate one.
    ///
    /// The template bank legitimately holds this many for a language no
    /// corpus covers. Refusing it would refuse an honest "we have four
    /// examples", and the distinction matters: a duplicate run is a defect,
    /// a small dataset is a fact.
    #[test]
    fn a_small_but_distinct_dataset_is_not_refused_as_degenerate() {
        let root = scratch("small-distinct");
        let dataset = root.join("train.jsonl");
        let mut body = String::new();
        for i in 0..4 {
            body.push_str(
                &serde_json::json!({
                    "id": format!("s{i}"),
                    "language": "rust",
                    "prompt": format!("write function {i}"),
                    "completion": format!("fn f{i}() -> i32 {{ {i} }}\n"),
                })
                .to_string(),
            );
            body.push('\n');
        }
        std::fs::write(&dataset, body).unwrap();

        let (_, summary) = scan_dataset_cheats(
            &dataset,
            None,
            &crate::cheat::MixingPolicy::default(),
            None,
            &root.join("work"),
        )
        .expect("four distinct answers is a small dataset, not a degenerate one");
        assert_eq!(summary.kept, 4);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
