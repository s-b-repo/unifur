//! Interop with **subnext**, the bounded-context runtime (PLAN-NINA.md §4b).
//!
//! Context lives *outside* the model. Capability is trained into weights here
//! in unifur; the corpus, memory and long documents are owned and served by
//! subnext. The two repos coordinate through a versioned, file-based contract
//! — never shared memory — so either side can be developed and tested alone:
//!
//! - [`ContextPut`] / [`ContextQuery`] / [`ContextHit`] / [`MemoryAppend`] are
//!   the versioned message types both sides implement. An unknown (or missing)
//!   version is a loud error that names it, never a silent reinterpretation.
//! - [`RequestChannel`] is the fallback transport until the MCP bridge is up:
//!   append-only `<id>.request.json` / `<id>.response.json` files in the
//!   interop directory, every write atomic via tmp-file + rename.
//! - [`LocalRetriever`] is the local fallback so the feature works *before*
//!   subnext answers, and doubles as the reference implementation the subnext
//!   side is tested against. It reads both the cybercorpus layout
//!   ([`LocalRetriever::load`]) and the serve-time context-blocks layout
//!   ([`LocalRetriever::load_blocks`]).
//! - [`augment_prompt`] injects retrieved context into the serve-time prompt
//!   in a clearly marked, budgeted block, so the student's trained context
//!   window stays small.
//!
//! The contract document itself ([`contract_json`], [`write_contract`]) is
//! generated from the same constants the code enforces, so it cannot drift.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Directory shared with subnext. Created on demand by [`RequestChannel::open`]
/// and [`write_contract`]; never assumed to exist.
pub const INTEROP_DIR: &str = "/srv/m-sda/unifur/interop";

/// File name of the machine-readable contract document inside [`INTEROP_DIR`].
pub const CONTRACT_FILE: &str = "contract.json";

/// Schema version of [`ContextPut`].
pub const CONTEXT_PUT_VERSION: u32 = 1;
/// Schema version of [`ContextQuery`].
pub const CONTEXT_QUERY_VERSION: u32 = 1;
/// Schema version of [`ContextHit`].
pub const CONTEXT_HIT_VERSION: u32 = 1;
/// Schema version of [`MemoryAppend`].
pub const MEMORY_APPEND_VERSION: u32 = 1;
/// Schema version of the [`ChannelRequest`] / [`ChannelResponse`] envelopes.
pub const CHANNEL_ENVELOPE_VERSION: u32 = 1;
/// Schema version of the [`ContractDoc`] document itself.
pub const CONTRACT_DOC_VERSION: u32 = 1;

/// Suffix of a submitted request file inside the channel directory.
pub const REQUEST_SUFFIX: &str = ".request.json";
/// Suffix of the responder's answer file inside the channel directory.
pub const RESPONSE_SUFFIX: &str = ".response.json";

/// A context document handed from unifur to subnext for storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextPut {
    /// Schema version; current is [`CONTEXT_PUT_VERSION`]. Missing parses as 0
    /// and is rejected like any unknown version.
    #[serde(default)]
    pub version: u32,
    /// Stable document id, assigned by the caller.
    #[serde(default)]
    pub id: String,
    /// What kind of context this is (`"document"`, `"memory"`, ...).
    #[serde(default)]
    pub kind: String,
    /// The full text of the document.
    #[serde(default)]
    pub text: String,
    /// Free-form metadata (provenance hints, tags); null when unused.
    #[serde(default)]
    pub meta: Value,
}

impl ContextPut {
    /// Build a message at the current schema version.
    pub fn new(
        id: impl Into<String>,
        kind: impl Into<String>,
        text: impl Into<String>,
        meta: Value,
    ) -> Self {
        Self {
            version: CONTEXT_PUT_VERSION,
            id: id.into(),
            kind: kind.into(),
            text: text.into(),
            meta,
        }
    }

    /// Parse a JSON message, rejecting versions this build does not implement.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let msg: Self = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("invalid ContextPut JSON: {e}"))?;
        check_version("ContextPut", msg.version, CONTEXT_PUT_VERSION)?;
        Ok(msg)
    }

    /// Serialize as pretty JSON.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// A retrieval query: unifur asks subnext (or the [`LocalRetriever`]) for the
/// `k` most relevant context documents.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextQuery {
    /// Schema version; current is [`CONTEXT_QUERY_VERSION`].
    #[serde(default)]
    pub version: u32,
    /// The query text (typically the user's prompt or a distilled form of it).
    #[serde(default)]
    pub text: String,
    /// Maximum number of hits to return.
    #[serde(default)]
    pub k: usize,
    /// Field to search in (`"pentest"`, `"coding"`, ...); empty means all.
    #[serde(default)]
    pub field: String,
}

impl ContextQuery {
    /// Build a query at the current schema version.
    pub fn new(text: impl Into<String>, k: usize, field: impl Into<String>) -> Self {
        Self {
            version: CONTEXT_QUERY_VERSION,
            text: text.into(),
            k,
            field: field.into(),
        }
    }

    /// Parse a JSON message, rejecting versions this build does not implement.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let msg: Self = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("invalid ContextQuery JSON: {e}"))?;
        check_version("ContextQuery", msg.version, CONTEXT_QUERY_VERSION)?;
        Ok(msg)
    }

    /// Serialize as pretty JSON.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Where a retrieved hit came from: which corpus source and which path inside
/// it. Filled by the retriever so the serve-time block can name its evidence.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Corpus source id (the `<source>` of the cybercorpus record).
    #[serde(default)]
    pub source: String,
    /// Path of the document within its source.
    #[serde(default)]
    pub path: String,
}

impl Provenance {
    /// Build a provenance record.
    pub fn new(source: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            path: path.into(),
        }
    }
}

/// One retrieved context document, scored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextHit {
    /// Schema version; current is [`CONTEXT_HIT_VERSION`].
    #[serde(default)]
    pub version: u32,
    /// The document id, stable within the corpus that served it.
    #[serde(default)]
    pub id: String,
    /// Relevance score; higher is better. Scale is retriever-defined.
    #[serde(default)]
    pub score: f64,
    /// The document text.
    #[serde(default)]
    pub text: String,
    /// Where this document came from.
    #[serde(default)]
    pub provenance: Provenance,
}

impl ContextHit {
    /// Build a hit at the current schema version.
    pub fn new(
        id: impl Into<String>,
        score: f64,
        text: impl Into<String>,
        provenance: Provenance,
    ) -> Self {
        Self {
            version: CONTEXT_HIT_VERSION,
            id: id.into(),
            score,
            text: text.into(),
            provenance,
        }
    }

    /// Parse a JSON message, rejecting versions this build does not implement.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let msg: Self = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("invalid ContextHit JSON: {e}"))?;
        check_version("ContextHit", msg.version, CONTEXT_HIT_VERSION)?;
        Ok(msg)
    }

    /// Serialize as pretty JSON.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// A write into subnext's Merkle-chained long-term memory: past chats and runs
/// become retrievable context, never weight edits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryAppend {
    /// Schema version; current is [`MEMORY_APPEND_VERSION`].
    #[serde(default)]
    pub version: u32,
    /// The payload to append; shape is owned by the memory log, not the model.
    #[serde(default)]
    pub payload: Value,
}

impl MemoryAppend {
    /// Build a message at the current schema version.
    pub fn new(payload: Value) -> Self {
        Self {
            version: MEMORY_APPEND_VERSION,
            payload,
        }
    }

    /// Parse a JSON message, rejecting versions this build does not implement.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let msg: Self = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("invalid MemoryAppend JSON: {e}"))?;
        check_version("MemoryAppend", msg.version, MEMORY_APPEND_VERSION)?;
        Ok(msg)
    }

    /// Serialize as pretty JSON.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Reject any version this build does not implement, naming both sides of the
/// mismatch. A missing `version` field parses as 0 through the serde defaults,
/// so "no version at all" is the same loud error as an unknown one.
fn check_version(kind: &str, got: u32, supported: u32) -> anyhow::Result<()> {
    anyhow::ensure!(
        got == supported,
        "{kind} message version {got} is not supported by this build \
         (supported: {supported}; a missing version parses as 0 and is \
         rejected the same way)"
    );
    Ok(())
}

/// The request envelope written as `<id>.request.json` in the channel
/// directory. Wraps a [`ContextQuery`] with the request id and an envelope
/// version so the transport itself can evolve independently of the messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelRequest {
    /// Envelope version; current is [`CHANNEL_ENVELOPE_VERSION`].
    #[serde(default)]
    pub version: u32,
    /// Request id; also the file-name stem of both envelope files.
    #[serde(default)]
    pub id: String,
    /// The query being asked.
    #[serde(default)]
    pub query: ContextQuery,
}

impl ChannelRequest {
    /// Parse a request envelope, rejecting unknown envelope versions.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let msg: Self = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("invalid ChannelRequest JSON: {e}"))?;
        check_version("ChannelRequest", msg.version, CHANNEL_ENVELOPE_VERSION)?;
        Ok(msg)
    }

    /// Serialize as pretty JSON.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// The response envelope written as `<id>.response.json` by the responder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelResponse {
    /// Envelope version; current is [`CHANNEL_ENVELOPE_VERSION`].
    #[serde(default)]
    pub version: u32,
    /// The request id this answers; must match the request's id.
    #[serde(default)]
    pub id: String,
    /// The retrieved hits.
    #[serde(default)]
    pub hits: Vec<ContextHit>,
}

impl ChannelResponse {
    /// Build a response envelope at the current version.
    pub fn new(id: impl Into<String>, hits: Vec<ContextHit>) -> Self {
        Self {
            version: CHANNEL_ENVELOPE_VERSION,
            id: id.into(),
            hits,
        }
    }

    /// Parse a response envelope, rejecting unknown envelope versions.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let msg: Self = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("invalid ChannelResponse JSON: {e}"))?;
        check_version("ChannelResponse", msg.version, CHANNEL_ENVELOPE_VERSION)?;
        Ok(msg)
    }

    /// Serialize as pretty JSON.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Write `bytes` to `path` atomically: the full contents land in a tmp sibling
/// first and are then renamed over the destination, so a concurrent reader
/// sees either the old file or the complete new one — never a partial write.
/// Parent directories are created on demand.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("creating '{}': {e}", parent.display()))?;
        }
    }
    let tmp = PathBuf::from(format!("{}.tmp-{}", path.display(), std::process::id()));
    fs::write(&tmp, bytes).map_err(|e| anyhow::anyhow!("writing '{}': {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| {
        anyhow::anyhow!(
            "renaming '{}' over '{}': {e}",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

/// A request id is valid as a file-name component only: no separators, no
/// traversal. Anything else is rejected before it touches the filesystem.
fn validate_request_id(request_id: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !request_id.is_empty()
            && request_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
        "invalid request id '{request_id}': only [A-Za-z0-9-_] is allowed \
         (the id becomes a file name in the channel directory)"
    );
    Ok(())
}

fn new_request_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("ctx-{}-{nanos}-{n}", std::process::id())
}

/// The file-based fallback channel to subnext: append-only request/response
/// JSON files in a shared directory. This is what the two sides use until the
/// MCP bridge is live, and it stays available afterwards as the
/// no-broker-needed transport.
///
/// Semantics:
///
/// - [`submit`](RequestChannel::submit) writes `<id>.request.json` atomically
///   and returns the id.
/// - [`poll`](RequestChannel::poll) waits for `<id>.response.json`, up to a
///   caller-chosen timeout. A timeout is an `Err` naming the request, the
///   directory and the waited seconds — never an infinite hang. A response
///   that exists but is corrupt, the wrong version, or answers a different id
///   is likewise a loud error, not a retry.
/// - [`gc`](RequestChannel::gc) removes answered request/response pairs;
///   unanswered requests are left alone so the responder can still find them.
pub struct RequestChannel {
    dir: PathBuf,
}

impl RequestChannel {
    /// Open the channel at `dir`, creating the directory on demand (the
    /// interop dir is never assumed to exist).
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)
            .map_err(|e| anyhow::anyhow!("creating channel dir '{}': {e}", dir.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
        })
    }

    /// The directory this channel reads and writes.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Path of the request file for `request_id`.
    pub fn request_path(&self, request_id: &str) -> PathBuf {
        self.dir.join(format!("{request_id}{REQUEST_SUFFIX}"))
    }

    /// Path of the response file for `request_id`.
    pub fn response_path(&self, request_id: &str) -> PathBuf {
        self.dir.join(format!("{request_id}{RESPONSE_SUFFIX}"))
    }

    /// Submit a query; returns the request id. The request file is written
    /// atomically, so a watching responder never sees a partial envelope.
    pub fn submit(&self, query: &ContextQuery) -> anyhow::Result<String> {
        let id = new_request_id();
        let request = ChannelRequest {
            version: CHANNEL_ENVELOPE_VERSION,
            id: id.clone(),
            query: query.clone(),
        };
        let json = request.to_json()?;
        write_atomic(&self.request_path(&id), json.as_bytes())?;
        Ok(id)
    }

    /// Wait for the response to `request_id`, up to `timeout`. Errors loudly —
    /// naming the request id, the directory and the waited seconds — when the
    /// timeout expires, and immediately when a present response is corrupt,
    /// an unknown version, or answers a different request id.
    pub fn poll(&self, request_id: &str, timeout: Duration) -> anyhow::Result<Vec<ContextHit>> {
        validate_request_id(request_id)?;
        let path = self.response_path(request_id);
        let started = Instant::now();
        loop {
            match fs::read_to_string(&path) {
                Ok(text) => {
                    let response = ChannelResponse::from_json(&text).map_err(|e| {
                        anyhow::anyhow!("response file '{}' is not answerable: {e}", path.display())
                    })?;
                    anyhow::ensure!(
                        response.id == request_id,
                        "response file '{}' answers request '{}' but was polled as '{request_id}'",
                        path.display(),
                        response.id
                    );
                    return Ok(response.hits);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    let waited = started.elapsed();
                    anyhow::ensure!(
                        waited < timeout,
                        "no response for request '{request_id}' in '{}' after {:.1}s \
                         (timeout was {:.1}s)",
                        self.dir.display(),
                        waited.as_secs_f64(),
                        timeout.as_secs_f64()
                    );
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "reading response for request '{request_id}' at '{}': {e}",
                        path.display()
                    ));
                }
            }
        }
    }

    /// Remove request/response pairs that have been answered. Returns the
    /// number of pairs removed. Unanswered requests are kept.
    pub fn gc(&self) -> anyhow::Result<usize> {
        if !self.dir.is_dir() {
            return Ok(0);
        }
        let mut removed = 0usize;
        for entry in fs::read_dir(&self.dir)
            .map_err(|e| anyhow::anyhow!("listing channel dir '{}': {e}", self.dir.display()))?
        {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(id) = name.strip_suffix(REQUEST_SUFFIX) else {
                continue;
            };
            let response = self.response_path(id);
            if response.is_file() {
                fs::remove_file(entry.path())?;
                fs::remove_file(&response)?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}

/// One message type of the contract, as documented in `contract.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageSpec {
    /// Type name as used in Rust and in this document.
    #[serde(default)]
    pub name: String,
    /// Current schema version of the message.
    #[serde(default)]
    pub version: u32,
    /// Field names, in declaration order, excluding the leading `version`.
    #[serde(default)]
    pub fields: Vec<String>,
    /// One-line description of the message's role.
    #[serde(default)]
    pub summary: String,
}

/// Transport rules for the file channel, as documented in `contract.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChannelSpec {
    /// Suffix of request files (`<id>` + this).
    #[serde(default)]
    pub request_suffix: String,
    /// Suffix of response files (`<id>` + this).
    #[serde(default)]
    pub response_suffix: String,
    /// How files are written: the atomicity rule both sides must implement.
    #[serde(default)]
    pub write_rule: String,
    /// How readers wait: the timeout rule both sides must implement.
    #[serde(default)]
    pub timeout_rule: String,
}

/// The machine-readable interop contract (`contract.json`). Generated from the
/// same constants the code enforces — see [`contract_doc`] — so the document
/// cannot drift from the implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractDoc {
    /// Document schema version; current is [`CONTRACT_DOC_VERSION`].
    #[serde(default)]
    pub version: u32,
    /// Protocol name both repos implement.
    #[serde(default)]
    pub protocol: String,
    /// The versioned messages, in the order a reader meets them.
    #[serde(default)]
    pub messages: Vec<MessageSpec>,
    /// The file-channel transport rules.
    #[serde(default)]
    pub channel: ChannelSpec,
}

impl ContractDoc {
    /// Parse a contract document, rejecting unknown document versions.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let doc: Self = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("invalid ContractDoc JSON: {e}"))?;
        check_version("ContractDoc", doc.version, CONTRACT_DOC_VERSION)?;
        Ok(doc)
    }
}

/// Build the canonical contract document from the enforced constants.
pub fn contract_doc() -> ContractDoc {
    ContractDoc {
        version: CONTRACT_DOC_VERSION,
        protocol: "unifur-subnext-interop".to_string(),
        messages: vec![
            MessageSpec {
                name: "ContextPut".to_string(),
                version: CONTEXT_PUT_VERSION,
                fields: vec![
                    "id".to_string(),
                    "kind".to_string(),
                    "text".to_string(),
                    "meta".to_string(),
                ],
                summary: "hand a context document to the context runtime for storage".to_string(),
            },
            MessageSpec {
                name: "ContextQuery".to_string(),
                version: CONTEXT_QUERY_VERSION,
                fields: vec!["text".to_string(), "k".to_string(), "field".to_string()],
                summary: "ask for the k most relevant context documents".to_string(),
            },
            MessageSpec {
                name: "ContextHit".to_string(),
                version: CONTEXT_HIT_VERSION,
                fields: vec![
                    "id".to_string(),
                    "score".to_string(),
                    "text".to_string(),
                    "provenance".to_string(),
                ],
                summary: "one scored retrieved document, with provenance".to_string(),
            },
            MessageSpec {
                name: "MemoryAppend".to_string(),
                version: MEMORY_APPEND_VERSION,
                fields: vec!["payload".to_string()],
                summary: "append to the Merkle-chained long-term memory log".to_string(),
            },
            MessageSpec {
                name: "ChannelRequest".to_string(),
                version: CHANNEL_ENVELOPE_VERSION,
                fields: vec!["id".to_string(), "query".to_string()],
                summary: "file-channel envelope wrapping a ContextQuery".to_string(),
            },
            MessageSpec {
                name: "ChannelResponse".to_string(),
                version: CHANNEL_ENVELOPE_VERSION,
                fields: vec!["id".to_string(), "hits".to_string()],
                summary: "file-channel envelope carrying the ContextHit list".to_string(),
            },
        ],
        channel: ChannelSpec {
            request_suffix: REQUEST_SUFFIX.to_string(),
            response_suffix: RESPONSE_SUFFIX.to_string(),
            write_rule: "every file is written atomically: full contents to a \
                         sibling tmp file, then rename over the destination"
                .to_string(),
            timeout_rule: "pollers wait with a caller-chosen timeout; expiry is \
                           a loud error naming the request id, the directory and \
                           the waited seconds — never an infinite hang"
                .to_string(),
        },
    }
}

/// Serialize the canonical contract document as pretty JSON.
pub fn contract_json() -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(&contract_doc())?)
}

/// Write `contract.json` into `dir`, creating the directory on demand. The
/// write is atomic like every other file in the interop directory.
pub fn write_contract(dir: &Path) -> anyhow::Result<PathBuf> {
    let path = dir.join(CONTRACT_FILE);
    write_atomic(&path, contract_json()?.as_bytes())?;
    Ok(path)
}

/// One line of a cybercorpus-layout dataset file. The `format` field of the
/// on-disk schema is deliberately not modelled: serde ignores unknown fields,
/// and the retriever never reads it.
#[derive(Debug, Deserialize)]
struct DatasetRecord {
    #[serde(default)]
    id: String,
    #[serde(default)]
    source: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    text: String,
}

/// One line of a `*.jsonl` file in a context-blocks tree. Unknown fields are
/// ignored: the layout owns only `text` and the optional `id`.
#[derive(Debug, Deserialize)]
struct BlockRecord {
    #[serde(default)]
    id: String,
    #[serde(default)]
    text: String,
}

/// Provenance source reported for blocks loaded from a context-blocks tree.
const BLOCKS_SOURCE: &str = "context-blocks";

/// Every `*.txt` and `*.jsonl` file under `dir`, recursively, in visitation
/// order. [`LocalRetriever::load_blocks`] sorts the result before reading, so
/// the load is deterministic regardless of directory iteration order.
fn collect_block_files(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in
        fs::read_dir(dir).map_err(|e| anyhow::anyhow!("listing '{}': {e}", dir.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            collect_block_files(&path, out)?;
        } else if path
            .extension()
            .is_some_and(|ext| ext == "txt" || ext == "jsonl")
        {
            out.push(path);
        }
    }
    Ok(())
}

/// Lowercased alphanumeric runs; the only tokenization the retriever does.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Whitespace-token count; the budget unit of [`augment_prompt`].
fn token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[derive(Debug)]
struct CorpusDoc {
    id: String,
    source: String,
    path: String,
    text: String,
    tokens: Vec<String>,
}

/// The local fallback retriever: serves context before subnext answers, and
/// doubles as the reference implementation the subnext side is tested
/// against. Deterministic by construction — no RNG anywhere, ties broken by
/// document id.
///
/// Scoring is **BM25-lite**. For query terms `t` and document `d`:
///
/// ```text
/// score(d) = Σ_t idf(t) · tf(t,d)·(k1+1) / (tf(t,d)+k1),   k1 = 1.5
/// idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
/// ```
///
/// The simplifications versus full BM25, stated plainly:
///
/// - **No document-length normalization** (no `b` parameter, no average
///   length): tf saturation alone bounds long-document advantage.
/// - **Tokenization is lowercase alphanumeric runs only** — no stemming, no
///   stop-words, no Unicode segmentation beyond `char::is_alphanumeric`.
///
/// Documents with zero score are never returned.
#[derive(Debug)]
pub struct LocalRetriever {
    docs: Vec<CorpusDoc>,
}

impl LocalRetriever {
    /// Load a cybercorpus-layout corpus: `<corpus_dir>/dataset/all.jsonl` when
    /// present, otherwise every `*.jsonl` in `<corpus_dir>/dataset/` in sorted
    /// file-name order (deterministic). A missing directory or an unreadable
    /// line is a loud error naming the path; a malformed line also names its
    /// 1-based line number.
    pub fn load(corpus_dir: &Path) -> anyhow::Result<Self> {
        let dataset_dir = corpus_dir.join("dataset");
        anyhow::ensure!(
            dataset_dir.is_dir(),
            "corpus dataset directory '{}' does not exist \
             (expected the cybercorpus layout: <corpus>/dataset/*.jsonl)",
            dataset_dir.display()
        );
        let all = dataset_dir.join("all.jsonl");
        let files = if all.is_file() {
            vec![all]
        } else {
            let mut per_source: Vec<PathBuf> = Vec::new();
            for entry in fs::read_dir(&dataset_dir)
                .map_err(|e| anyhow::anyhow!("listing '{}': {e}", dataset_dir.display()))?
            {
                let path = entry?.path();
                if path.extension().is_some_and(|ext| ext == "jsonl") {
                    per_source.push(path);
                }
            }
            per_source.sort();
            per_source
        };
        anyhow::ensure!(
            !files.is_empty(),
            "corpus dataset directory '{}' holds no .jsonl files",
            dataset_dir.display()
        );
        let mut docs = Vec::new();
        for file in &files {
            let text = fs::read_to_string(file)
                .map_err(|e| anyhow::anyhow!("reading '{}': {e}", file.display()))?;
            for (line_no, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let record: DatasetRecord = serde_json::from_str(line).map_err(|e| {
                    anyhow::anyhow!(
                        "'{}' line {} is not a dataset record: {e}",
                        file.display(),
                        line_no + 1
                    )
                })?;
                docs.push(CorpusDoc {
                    tokens: tokenize(&record.text),
                    id: record.id,
                    source: record.source,
                    path: record.path,
                    text: record.text,
                });
            }
        }
        anyhow::ensure!(
            !docs.is_empty(),
            "corpus dataset directory '{}' parsed to zero documents",
            dataset_dir.display()
        );
        Ok(Self { docs })
    }

    /// Load a **context-blocks tree**: the layout `dblocks nina chat
    /// --context-root` serves. Every `*.txt` and `*.jsonl` file under `root`,
    /// recursively, in sorted path order (deterministic):
    ///
    /// - a `*.txt` file is one block; its id is its path relative to `root`;
    /// - a `*.jsonl` file holds one JSON object per line, each with a `text`
    ///   string and an optional `id` (default `<relpath>#line-N`, 1-based).
    ///
    /// A missing root, an unreadable file or a malformed line is a loud error
    /// naming the path (and the 1-based line number); an empty tree parses to
    /// zero blocks and is refused the same way.
    ///
    /// Honest note on scale: the store is **RAM-resident** — every block's
    /// text and token list live in this struct — so at the 1,000,000-token
    /// design target the whole corpus sits in memory (a few tens of MB, which
    /// is fine), and the win of select-before-read is the *attention* budget,
    /// not I/O: the student's trained context window reads 1-4k tokens either
    /// way, with or without retrieval.
    pub fn load_blocks(root: &Path) -> anyhow::Result<Self> {
        anyhow::ensure!(
            root.is_dir(),
            "context-blocks root '{}' does not exist \
             (expected a directory of *.txt files and/or *.jsonl docs)",
            root.display()
        );
        let mut files = Vec::new();
        collect_block_files(root, &mut files)?;
        files.sort();
        anyhow::ensure!(
            !files.is_empty(),
            "context-blocks root '{}' holds no *.txt or *.jsonl files",
            root.display()
        );
        let mut docs = Vec::new();
        for file in &files {
            let rel = file
                .strip_prefix(root)
                .map_err(|e| anyhow::anyhow!("'{file:?}' is not under the blocks root: {e}"))?
                .to_string_lossy()
                .replace('\\', "/");
            let text = fs::read_to_string(file)
                .map_err(|e| anyhow::anyhow!("reading '{}': {e}", file.display()))?;
            if file.extension().is_some_and(|ext| ext == "txt") {
                docs.push(CorpusDoc {
                    tokens: tokenize(&text),
                    id: rel.clone(),
                    source: BLOCKS_SOURCE.to_string(),
                    path: rel,
                    text,
                });
                continue;
            }
            for (line_no, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let record: BlockRecord = serde_json::from_str(line).map_err(|e| {
                    anyhow::anyhow!(
                        "'{}' line {} is not a context block: {e}",
                        file.display(),
                        line_no + 1
                    )
                })?;
                anyhow::ensure!(
                    !record.text.is_empty(),
                    "'{}' line {} has an empty 'text'",
                    file.display(),
                    line_no + 1
                );
                let id = if record.id.is_empty() {
                    format!("{rel}#line-{}", line_no + 1)
                } else {
                    record.id
                };
                docs.push(CorpusDoc {
                    tokens: tokenize(&record.text),
                    id,
                    source: BLOCKS_SOURCE.to_string(),
                    path: rel.clone(),
                    text: record.text,
                });
            }
        }
        anyhow::ensure!(
            !docs.is_empty(),
            "context-blocks root '{}' parsed to zero blocks",
            root.display()
        );
        Ok(Self { docs })
    }

    /// Number of documents loaded.
    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }

    /// Score every document against `query` and return the top `k` hits with
    /// provenance filled. Order is total and deterministic: score descending,
    /// then document id ascending.
    pub fn retrieve(&self, query: &str, k: usize) -> Vec<ContextHit> {
        let mut query_terms: Vec<String> = Vec::new();
        for term in tokenize(query) {
            if !query_terms.contains(&term) {
                query_terms.push(term);
            }
        }
        let doc_total = self.docs.len() as f64; // audit-allow: doc counts fit f64 exactly far past any real corpus
        let k1 = 1.5_f64;
        let mut scored: Vec<(f64, &CorpusDoc)> = Vec::new();
        for doc in &self.docs {
            let mut score = 0.0_f64;
            for term in &query_terms {
                let tf = doc.tokens.iter().filter(|t| t == &term).count() as f64; // audit-allow: term counts fit f64 exactly
                if tf == 0.0 {
                    continue;
                }
                let df = self
                    .docs
                    .iter()
                    .filter(|d| d.tokens.iter().any(|t| t == term))
                    .count() as f64; // audit-allow: document counts fit f64 exactly
                let idf = (1.0 + (doc_total - df + 0.5) / (df + 0.5)).ln();
                score += idf * (tf * (k1 + 1.0)) / (tf + k1);
            }
            if score > 0.0 {
                scored.push((score, doc));
            }
        }
        scored.sort_by(|(sa, da), (sb, db)| sb.total_cmp(sa).then_with(|| da.id.cmp(&db.id)));
        scored
            .into_iter()
            .take(k)
            .map(|(score, doc)| {
                ContextHit::new(
                    doc.id.clone(),
                    score,
                    doc.text.clone(),
                    Provenance::new(doc.source.clone(), doc.path.clone()),
                )
            })
            .collect()
    }
}

/// Inject retrieved context into a serve-time prompt.
///
/// Each hit becomes a clearly marked block:
///
/// ```text
/// [context id=<id> source=<source>]
/// <document text>
/// [/context]
/// ```
///
/// followed by the bare prompt. The context is budgeted in whitespace tokens
/// (`max_context_tokens` covers headers, text and closing markers): hits are
/// added in order, a hit's text is truncated to the remaining budget, and no
/// further hit is started once a header no longer fits. With a budget of 0 —
/// or no hits — the prompt is returned unchanged, so a bare prompt stays
/// byte-identical. The student's trained context window stays small; anything
/// longer is the retriever's job.
pub fn augment_prompt(prompt: &str, hits: &[ContextHit], max_context_tokens: usize) -> String {
    let mut block = String::new();
    let mut used = 0usize;
    for hit in hits {
        let header = format!("[context id={} source={}]", hit.id, hit.provenance.source);
        let header_cost = token_count(&header) + 1; // +1 for the [/context] marker
        let remaining = max_context_tokens.saturating_sub(used);
        if header_cost >= remaining {
            break;
        }
        let text_budget = remaining - header_cost;
        let text: Vec<&str> = hit.text.split_whitespace().take(text_budget).collect();
        block.push_str(&header);
        block.push('\n');
        block.push_str(&text.join(" "));
        block.push_str("\n[/context]\n");
        used += header_cost + text.len();
    }
    if block.is_empty() {
        prompt.to_string()
    } else {
        format!("{block}\n{prompt}")
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn unique_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "dblocks-interop-{}-{tag}-{nanos}-{n}",
            std::process::id()
        ))
    }

    /// A tiny field-QA corpus: the answer to the query exists ONLY in doc-2.
    fn write_mini_corpus(root: &Path) -> PathBuf {
        let dataset = root.join("corpus").join("dataset");
        fs::create_dir_all(&dataset).unwrap();
        let docs = [
            r#"{"id":"doc-1","source":"field-manual","path":"nets.md","format":"markdown","text":"The Copper Badger sensor mesh relays telemetry over UDP port 5353 and rotates keys weekly."}"#,
            r#"{"id":"doc-2","source":"field-manual","path":"relay.md","format":"markdown","text":"The Argent Fox relay listens on TCP port 7447 by default and refuses plaintext fallback."}"#,
            r#"{"id":"doc-3","source":"field-manual","path":"dns.md","format":"markdown","text":"Internal zones resolve through the Juniper Wren resolver pair on UDP port 53."}"#,
        ];
        fs::write(dataset.join("all.jsonl"), docs.join("\n") + "\n").unwrap();
        root.join("corpus")
    }

    // --- contract schema ----------------------------------------------------

    #[test]
    fn schema_round_trips_and_version_rejection() {
        let put = ContextPut::new("doc-9", "document", "body text", Value::Null);
        assert_eq!(ContextPut::from_json(&put.to_json().unwrap()).unwrap(), put);

        let query = ContextQuery::new("which port", 3, "pentest");
        assert_eq!(
            ContextQuery::from_json(&query.to_json().unwrap()).unwrap(),
            query
        );

        let hit = ContextHit::new("doc-2", 1.25, "text", Provenance::new("src", "p.md"));
        assert_eq!(ContextHit::from_json(&hit.to_json().unwrap()).unwrap(), hit);

        let mem = MemoryAppend::new(serde_json::json!({"chat": 1}));
        assert_eq!(
            MemoryAppend::from_json(&mem.to_json().unwrap()).unwrap(),
            mem
        );

        // Unknown versions are loud errors that name the version received.
        for (json, expected) in [
            (r#"{"version": 7, "id": "x"}"#, "ContextPut"),
            (r#"{"version": 7, "text": "x"}"#, "ContextQuery"),
            (r#"{"version": 7, "id": "x"}"#, "ContextHit"),
            (r#"{"version": 7, "payload": null}"#, "MemoryAppend"),
        ] {
            let err = match expected {
                "ContextPut" => ContextPut::from_json(json).unwrap_err(),
                "ContextQuery" => ContextQuery::from_json(json).unwrap_err(),
                "ContextHit" => ContextHit::from_json(json).unwrap_err(),
                _ => MemoryAppend::from_json(json).unwrap_err(),
            };
            let msg = format!("{err}");
            assert!(msg.contains(expected), "{msg}");
            assert!(msg.contains('7'), "{msg}");
        }

        // A missing version parses as 0 and is rejected the same way.
        let err = ContextQuery::from_json(r#"{"text": "x", "k": 1}"#).unwrap_err();
        assert!(format!("{err}").contains("version 0"), "{err}");
    }

    #[test]
    fn contract_document_round_trips() {
        let json = contract_json().unwrap();
        let doc = ContractDoc::from_json(&json).unwrap();
        assert_eq!(doc.version, CONTRACT_DOC_VERSION);
        let names: Vec<&str> = doc.messages.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "ContextPut",
                "ContextQuery",
                "ContextHit",
                "MemoryAppend",
                "ChannelRequest",
                "ChannelResponse"
            ]
        );
        assert_eq!(
            doc.messages.iter().map(|m| m.version).collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 1, 1]
        );
        assert_eq!(doc.channel.request_suffix, REQUEST_SUFFIX);
        assert_eq!(doc.channel.response_suffix, RESPONSE_SUFFIX);

        // The written file parses back into the same document.
        let dir = unique_dir("contract");
        let path = write_contract(&dir).unwrap();
        let on_disk = ContractDoc::from_json(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(on_disk, doc);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "writes the real /srv/m-sda/unifur/interop/contract.json on demand"]
    fn write_real_contract_file() {
        let path = write_contract(Path::new(INTEROP_DIR)).unwrap();
        assert_eq!(path, Path::new(INTEROP_DIR).join(CONTRACT_FILE));
        let doc = ContractDoc::from_json(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(doc, contract_doc());
    }

    // --- channel --------------------------------------------------------------

    /// Fake responder: stands in for subnext. Waits for the one request file,
    /// then answers it with one hit, atomically.
    fn spawn_responder(dir: PathBuf) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                let mut request_file = None;
                for entry in fs::read_dir(&dir).unwrap().flatten() {
                    let name = entry.file_name();
                    if name.to_str().is_some_and(|n| n.ends_with(REQUEST_SUFFIX)) {
                        request_file = Some(entry.path());
                        break;
                    }
                }
                if let Some(path) = request_file {
                    let request =
                        ChannelRequest::from_json(&fs::read_to_string(&path).unwrap()).unwrap();
                    let response = ChannelResponse::new(
                        request.id.clone(),
                        vec![ContextHit::new(
                            "doc-2",
                            0.9,
                            "The Argent Fox relay listens on TCP port 7447 by default.",
                            Provenance::new("field-manual", "relay.md"),
                        )],
                    );
                    let out = dir.join(format!("{}{RESPONSE_SUFFIX}", request.id));
                    write_atomic(&out, response.to_json().unwrap().as_bytes()).unwrap();
                    return;
                }
                assert!(Instant::now() < deadline, "responder saw no request file");
                std::thread::sleep(Duration::from_millis(5));
            }
        })
    }

    #[test]
    fn channel_submit_poll_gc() {
        let dir = unique_dir("channel");
        let channel = RequestChannel::open(&dir).unwrap();
        let responder = spawn_responder(channel.dir().to_path_buf());

        let id = channel
            .submit(&ContextQuery::new("which port", 1, "pentest"))
            .unwrap();
        assert!(channel.request_path(&id).is_file());

        let hits = channel.poll(&id, Duration::from_secs(10)).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits.first().unwrap().id, "doc-2");
        responder.join().unwrap();

        // gc removes the answered pair and leaves the directory empty.
        assert_eq!(channel.gc().unwrap(), 1);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        assert_eq!(channel.gc().unwrap(), 0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn poll_timeout_is_loud_and_finite() {
        let dir = unique_dir("timeout");
        let channel = RequestChannel::open(&dir).unwrap();
        let started = Instant::now();
        let err = channel
            .poll("ctx-no-such-request", Duration::from_millis(150))
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ctx-no-such-request"), "{msg}");
        assert!(msg.contains(&dir.display().to_string()), "{msg}");
        assert!(msg.contains("0.1s"), "{msg}"); // timeout was 0.1s after rounding
        assert!(started.elapsed() < Duration::from_secs(5));

        // Path-traversal ids never reach the filesystem.
        let err = channel
            .poll("../escape", Duration::from_millis(10))
            .unwrap_err();
        assert!(format!("{err}").contains("invalid request id"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn atomic_writes_have_no_partial_reads() {
        let dir = unique_dir("atomic");
        let channel = RequestChannel::open(&dir).unwrap();
        let id = channel.submit(&ContextQuery::new("bulk", 500, "")).unwrap();

        // Slow responder: the response body is written chunk by chunk into a
        // tmp sibling, then renamed. The poller must only ever see the
        // complete file — never a mid-write prefix.
        let respond_dir = channel.dir().to_path_buf();
        let respond_id = id.clone();
        let responder = std::thread::spawn(move || {
            let hits: Vec<ContextHit> = (0..500)
                .map(|i| {
                    ContextHit::new(
                        format!("doc-{i}"),
                        1.0,
                        format!("bulk text {i}"),
                        Provenance::new("bulk", "bulk.md"),
                    )
                })
                .collect();
            let json = ChannelResponse::new(respond_id.clone(), hits)
                .to_json()
                .unwrap();
            let tmp = respond_dir.join(format!("{respond_id}.response.json.tmp-x"));
            for chunk in json.as_bytes().chunks(97) {
                use std::io::Write;
                let mut f = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&tmp)
                    .unwrap();
                f.write_all(chunk).unwrap();
                f.flush().unwrap();
            }
            fs::rename(
                &tmp,
                respond_dir.join(format!("{respond_id}{RESPONSE_SUFFIX}")),
            )
            .unwrap();
        });

        let hits = channel.poll(&id, Duration::from_secs(10)).unwrap();
        responder.join().unwrap();
        assert_eq!(hits.len(), 500);
        assert_eq!(hits.last().unwrap().id, "doc-499");
        fs::remove_dir_all(&dir).unwrap();
    }

    // --- retriever --------------------------------------------------------------

    #[test]
    fn retriever_ranks_the_relevant_doc_first() {
        let dir = unique_dir("corpus");
        let corpus = write_mini_corpus(&dir);
        let retriever = LocalRetriever::load(&corpus).unwrap();
        assert_eq!(retriever.doc_count(), 3);

        let hits = retriever.retrieve("Which TCP port does the Argent Fox relay listen on?", 1);
        assert_eq!(hits.len(), 1);
        let hit = hits.first().unwrap();
        assert_eq!(hit.id, "doc-2");
        assert_eq!(hit.provenance.source, "field-manual");
        assert_eq!(hit.provenance.path, "relay.md");
        assert!(hit.score > 0.0);
        assert!(hit.text.contains("7447"));

        // A query matching nothing returns nothing, never a zero-score filler.
        assert!(retriever.retrieve("zanzibar quilt", 3).is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn retriever_is_deterministic() {
        let dir = unique_dir("determinism");
        let corpus = write_mini_corpus(&dir);
        let a = LocalRetriever::load(&corpus).unwrap();
        let b = LocalRetriever::load(&corpus).unwrap();
        for _ in 0..5 {
            assert_eq!(
                a.retrieve("TCP port relay Argent Fox", 3),
                b.retrieve("TCP port relay Argent Fox", 3)
            );
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_corpus_dir_is_a_loud_error() {
        let missing = unique_dir("absent").join("no-such-corpus");
        let err = LocalRetriever::load(&missing).unwrap_err();
        assert!(
            format!("{err}").contains(&missing.join("dataset").display().to_string()),
            "{err}"
        );
    }

    #[test]
    fn malformed_corpus_line_names_file_and_line() {
        let dir = unique_dir("malformed");
        let dataset = dir.join("corpus").join("dataset");
        fs::create_dir_all(&dataset).unwrap();
        fs::write(
            dataset.join("all.jsonl"),
            "{\"id\":\"ok\",\"source\":\"s\",\"path\":\"p\",\"text\":\"fine\"}\nnot json\n",
        )
        .unwrap();
        let err = LocalRetriever::load(&dir.join("corpus")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("all.jsonl"), "{msg}");
        assert!(msg.contains("line 2"), "{msg}");
        fs::remove_dir_all(&dir).unwrap();
    }

    // --- serve-time glue ---------------------------------------------------------

    #[test]
    fn augmented_prompt_carries_the_answer_fact_and_bare_does_not() {
        // The honest check, pinned: the answer exists only in the corpus, so
        // the retrieval-augmented prompt contains the fact and the bare prompt
        // provably does not.
        let dir = unique_dir("qa");
        let corpus = write_mini_corpus(&dir);
        let retriever = LocalRetriever::load(&corpus).unwrap();
        let prompt = "Which TCP port does the Argent Fox relay listen on?";
        assert!(!prompt.contains("7447"));

        let hits = retriever.retrieve(prompt, 1);
        let augmented = augment_prompt(prompt, &hits, 64);
        assert!(augmented.contains("7447"));
        assert!(augmented.contains("[context id=doc-2 source=field-manual]"));
        assert!(augmented.contains("[/context]"));
        assert!(augmented.ends_with(prompt));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn augment_respects_budget_and_marks_blocks() {
        let hit = ContextHit::new(
            "doc-1",
            1.0,
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
            Provenance::new("manual", "m.md"),
        );

        // Budget 0 (or no hits) returns the bare prompt byte-identical.
        assert_eq!(augment_prompt("p", std::slice::from_ref(&hit), 0), "p");
        assert_eq!(augment_prompt("p", &[], 10), "p");

        // A tight budget truncates text but never overflows it, and the block
        // stays marked.
        let augmented = augment_prompt("question", std::slice::from_ref(&hit), 6);
        assert!(augmented.contains("[context id=doc-1 source=manual]"));
        let context_part = augmented.strip_suffix("\nquestion").unwrap();
        assert!(token_count(context_part) <= 6, "{context_part}");

        // A generous budget fits the whole document.
        let full = augment_prompt("question", &[hit], 64);
        assert!(full.contains("alpha bravo charlie"));
        assert!(full.contains("india juliet"));
    }

    // --- context blocks (the --context-root layout) -----------------------------

    /// A small blocks tree mixing both layouts: a nested `*.txt` file and a
    /// `*.jsonl` file whose lines carry and omit ids.
    fn write_block_tree(root: &Path) -> PathBuf {
        let tree = root.join("blocks");
        fs::create_dir_all(tree.join("nested")).unwrap();
        fs::write(
            tree.join("nested").join("relay.txt"),
            "The Argent Fox relay listens on TCP port 7447 by default.",
        )
        .unwrap();
        let lines = [
            serde_json::json!({"id":"memo-1","text":"The Copper Badger mesh rotates keys weekly."}),
            serde_json::json!({"text":"The Juniper Wren resolver pair answers on UDP port 53."}),
        ];
        let body = lines
            .iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(tree.join("notes.jsonl"), body + "\n").unwrap();
        tree
    }

    #[test]
    fn load_blocks_indexes_txt_and_jsonl_with_relative_ids() {
        let dir = unique_dir("blocks");
        let tree = write_block_tree(&dir);
        let retriever = LocalRetriever::load_blocks(&tree).unwrap();
        assert_eq!(retriever.doc_count(), 3);

        // A *.txt file is one block, addressed by its path relative to root.
        let hits = retriever.retrieve("Which TCP port does the Argent Fox relay use?", 3);
        assert_eq!(hits.first().unwrap().id, "nested/relay.txt");
        assert_eq!(hits.first().unwrap().provenance.source, BLOCKS_SOURCE);

        // An explicit jsonl id wins; a line without one is addressed by file
        // and 1-based line number.
        let hits = retriever.retrieve("Copper Badger mesh keys", 3);
        assert_eq!(hits.first().unwrap().id, "memo-1");
        let hits = retriever.retrieve("Juniper Wren resolver", 3);
        assert_eq!(hits.first().unwrap().id, "notes.jsonl#line-2");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_blocks_errors_are_loud() {
        // A missing root names the path.
        let missing = unique_dir("absent-blocks").join("no-such-root");
        let err = LocalRetriever::load_blocks(&missing).unwrap_err();
        assert!(
            format!("{err}").contains(&missing.display().to_string()),
            "{err}"
        );

        // A malformed jsonl line names the file and the 1-based line number.
        let dir = unique_dir("badline");
        let tree = dir.join("blocks");
        fs::create_dir_all(&tree).unwrap();
        fs::write(tree.join("bad.jsonl"), "{\"text\":\"fine\"}\nnot json\n").unwrap();
        let err = LocalRetriever::load_blocks(&tree).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("bad.jsonl"), "{msg}");
        assert!(msg.contains("line 2"), "{msg}");

        // A tree with no block files at all is refused, not an empty index.
        let empty = unique_dir("empty-blocks");
        fs::create_dir_all(empty.join("sub")).unwrap();
        let err = LocalRetriever::load_blocks(&empty).unwrap_err();
        assert!(format!("{err}").contains("no *.txt or *.jsonl"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
        fs::remove_dir_all(&empty).unwrap();
    }

    // --- the 1M-token context proof ------------------------------------------------

    /// A deterministic PRNG for the fixture generator: no `rand` in the test
    /// path, and the corpus is bit-reproducible from the seed.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: usize) -> usize {
            let n = u64::try_from(n).unwrap();
            usize::try_from(self.next() % n).unwrap()
        }
    }

    /// Common English words, so the query's function words occur in filler
    /// documents too and only the *distinctive* terms carry the idf. The
    /// answer's rare terms (heliodrome, kelvin) are deliberately absent.
    const FILLER_WORDS: &[&str] = &[
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "for",
        "with",
        "from",
        "that",
        "this",
        "it",
        "its",
        "as",
        "not",
        "but",
        "if",
        "then",
        "than",
        "so",
        "such",
        "into",
        "over",
        "under",
        "between",
        "through",
        "during",
        "system",
        "network",
        "array",
        "mirror",
        "sensor",
        "relay",
        "signal",
        "channel",
        "device",
        "module",
        "circuit",
        "voltage",
        "current",
        "power",
        "thermal",
        "optical",
        "digital",
        "analog",
        "control",
        "data",
        "packet",
        "frame",
        "buffer",
        "cache",
        "memory",
        "storage",
        "unit",
        "node",
        "link",
        "port",
        "wire",
        "band",
        "field",
        "level",
        "rate",
        "mode",
        "state",
        "value",
        "range",
        "limit",
        "error",
        "check",
        "test",
        "calibration",
        "routine",
        "schedule",
        "maintenance",
        "report",
        "log",
        "record",
        "entry",
        "status",
        "alarm",
        "threshold",
        "operator",
        "engineer",
        "station",
        "facility",
        "chamber",
        "platform",
        "mount",
        "housing",
    ];

    struct Fixture {
        fact: String,
        fact_id: String,
        query: String,
        docs: usize,
        tokens: usize,
        bytes: usize,
    }

    /// Write a deterministic filler corpus of `docs` blocks (jsonl bundles of
    /// 250) totalling roughly `docs * 112` whitespace tokens, with the
    /// answer-bearing fact planted at a seeded depth. The fact's rare terms
    /// occur nowhere else, so retrieval must find exactly one needle.
    fn write_fixture(root: &Path, docs: usize, seed: u64) -> Fixture {
        let mut rng = XorShift(seed | 1);
        // A wide pseudo-vocabulary keeps filler df low and honest: the corpus
        // is not a wall of identical documents.
        let mut vocab: Vec<String> = FILLER_WORDS.iter().map(|w| (*w).to_string()).collect();
        for i in 0..4096 {
            vocab.push(format!("qx{:04}", rng.below(100_000) + i));
        }
        let fact = "The Heliodrome Array keeps its primary mirror at 77 kelvin through a \
                    closed helium loop."
            .to_string();
        let fact_index = rng.below(docs);
        let fact_id = format!("doc-{fact_index:05}");
        let mut tokens = 0usize;
        let mut bytes = 0usize;
        let mut pending: Vec<String> = Vec::new();
        if root.is_dir() {
            fs::remove_dir_all(root).unwrap();
        }
        fs::create_dir_all(root).unwrap();
        for i in 0..docs {
            let mut words: Vec<String> = Vec::new();
            if i == fact_index {
                words.push(fact.clone());
            }
            let len = 96 + rng.below(32);
            for _ in 0..len {
                words.push(vocab[rng.below(vocab.len())].clone());
            }
            let text = words.join(" ");
            tokens += token_count(&text);
            let id = format!("doc-{i:05}");
            let line = serde_json::json!({"id": id, "text": text}).to_string();
            bytes += line.len() + 1;
            pending.push(line);
            if pending.len() == 250 || i + 1 == docs {
                let file = root.join(format!("blocks-{:03}.jsonl", i / 250));
                fs::write(&file, pending.join("\n") + "\n").unwrap();
                pending.clear();
            }
        }
        Fixture {
            fact,
            fact_id,
            query: "At what temperature is the Heliodrome Array mirror kept?".to_string(),
            docs,
            tokens,
            bytes,
        }
    }

    /// The three pinned assertions of the 1M proof, shared by the small
    /// always-on test and the ignored real-scale driver: (a) the answer block
    /// is in the top-k hits, (b) the augmented prompt carries the fact inside
    /// the token budget, (c) the bare prompt provably does not.
    fn assert_fixture_retrieval(
        root: &Path,
        fixture: &Fixture,
        budget: usize,
    ) -> (usize, f64, f64) {
        let load_started = Instant::now();
        let retriever = LocalRetriever::load_blocks(root).unwrap();
        let load_s = load_started.elapsed().as_secs_f64();
        assert_eq!(retriever.doc_count(), fixture.docs);

        let retrieve_started = Instant::now();
        let hits = retriever.retrieve(&fixture.query, 8);
        let retrieve_s = retrieve_started.elapsed().as_secs_f64();

        // (a) The answer block is in the top-k.
        let rank = hits.iter().position(|h| h.id == fixture.fact_id);
        assert!(
            rank.is_some(),
            "the answer block {} is not in the top-{} hits: {:?}",
            fixture.fact_id,
            hits.len(),
            hits.iter().map(|h| &h.id).collect::<Vec<_>>()
        );

        // (b) The augmented prompt carries the fact inside the budget.
        let augmented = augment_prompt(&fixture.query, &hits, budget);
        assert!(augmented.contains(&fixture.fact), "{augmented}");
        let context_part = augmented
            .strip_suffix(&format!("\n{}", fixture.query))
            .unwrap();
        assert!(token_count(context_part) <= budget, "{context_part}");

        // (c) The bare prompt provably does not.
        assert!(!fixture.query.contains("77 kelvin"));
        (rank.unwrap() + 1, load_s, retrieve_s)
    }

    #[test]
    fn small_context_fixture_retrieves_the_fact() {
        // The always-on scale of the 1M proof: same generator, same
        // assertions, 50 documents instead of 9,000.
        let dir = unique_dir("fixture-small");
        let tree = dir.join("fixture");
        let fixture = write_fixture(&tree, 50, 0x5eed);
        let (rank, load_s, retrieve_s) = assert_fixture_retrieval(&tree, &fixture, 768);
        eprintln!(
            "small fixture: {} docs, {} tokens, {} bytes; answer rank {rank}; \
             load {load_s:.3}s, retrieve {retrieve_s:.3}s",
            fixture.docs, fixture.tokens, fixture.bytes
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "builds the real 1M-token context fixture; run explicitly"]
    fn million_token_context_fixture_retrieves_the_fact() {
        // The design-target proof: ~1M tokens of external context, the answer
        // hidden at a seeded depth, selected before read into a 768-token
        // budget. The fixture tree is left on disk for the subnext side.
        let root = Path::new("/srv/m-sdd/unifur/context/1m-fixture");
        let fixture = write_fixture(root, 9_000, 0x1_000_000);
        let (rank, load_s, retrieve_s) = assert_fixture_retrieval(root, &fixture, 768);
        eprintln!(
            "1M fixture: {} docs, {} tokens, {} bytes at {}; answer rank {rank}; \
             load {load_s:.3}s, retrieve {retrieve_s:.3}s",
            fixture.docs,
            fixture.tokens,
            fixture.bytes,
            root.display()
        );
    }
}
