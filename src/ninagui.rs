//! The NINA web dashboard: `dblocks nina serve` (PLAN-NINA.md §8).
//!
//! A localhost HTTP/1.1 server, std-only (a hand-rolled parser on
//! [`TcpListener`]; no new dependencies), that shows what the model is doing
//! while training — and shows it straight from the tamper-evident history:
//!
//! - `/` — the registry read model (`registry.json`) as a table, plus the
//!   chain state of every log under `<root>/logs/`, **re-verified per
//!   request**, so an after-the-fact edit renders red instead of passing
//!   silently;
//! - `/live` — the tail of every chained JSONL log (queue, registry, chat);
//! - `/thinking?prompt=…` — the vecread thought trace ([`crate::vecread`]) of
//!   the newest student checkpoint, with its `APPROXIMATION` header intact;
//!   without a checkpoint the page says so loudly instead of faking a trace;
//! - `/chat?prompt=…` — one chat turn: the prompt is routed through the same
//!   [`crate::nina::route_prompt`] the MCP server uses, the student's greedy
//!   seeded answer is generated when a checkpoint exists (and "pending" is
//!   said out loud when none does), and the turn is appended to
//!   `<root>/logs/chat.jsonl` as a chained record;
//! - `/data` — the corpus manifests (`<root>/*/dataset/manifest.json`) found
//!   under the root.
//!
//! # Honesty rules
//!
//! Every page header carries the chain-verification state of the log it reads:
//! `chain ok · lines=N · head=…` or `TAMPERED: line k: …`. A missing root,
//! registry or log renders an explicit empty state, never a panic. A malformed
//! request gets a `400` and its connection closes; the server keeps serving.
//! Nothing here unwraps a client request.

use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::nina::{ChainedLog, NinaPaths, RegistryFile};
use crate::tokenizer::{ByteTokenizer, Special};

/// The inference backend the thinking and chat pages run the student on.
type Eval = burn::backend::NdArray<f32>;

/// Longest accepted request line; longer gets a `400`, not a bigger buffer.
const MAX_REQUEST_LINE: usize = 8192;
/// Longest accepted single header line.
const MAX_HEADER_LINE: usize = 4096;
/// Total header bytes accepted per request.
const MAX_HEADER_BYTES: usize = 16384;
/// How long a connection may stall before its thread lets go of it.
const IO_TIMEOUT: Duration = Duration::from_secs(10);
/// Default number of tail lines `/live` shows per log.
const DEFAULT_TAIL: usize = 20;
/// The most tail lines a `?n=` query may ask for.
const MAX_TAIL: usize = 200;
/// `/live` and history views read at most this much of a log's tail.
const TAIL_READ_CAP: u64 = 256 * 1024;
/// Greedy generation cap for a chat answer (bounded so a turn stays cheap).
const GENERATE_MAX_NEW: usize = 48;
/// The depth sweep the thinking page asks the lens for (same as the CLI's).
const THINK_DEPTHS: [usize; 4] = [1, 2, 4, 8];

// ---------------------------------------------------------------------------
// The server
// ---------------------------------------------------------------------------

/// Everything a request handler needs, shared across connection threads.
struct GuiState {
    root: PathBuf,
}

impl GuiState {
    fn paths(&self) -> NinaPaths {
        NinaPaths::new(self.root.clone())
    }

    /// `<root>/logs` — where every chained JSONL lives.
    fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    /// `<root>/logs/chat.jsonl` — the chat history, itself a chained log.
    fn chat_log(&self) -> PathBuf {
        self.logs_dir().join("chat.jsonl")
    }
}

/// The bound dashboard server. [`GuiServer::bind`] reserves the port (pass 0
/// for an ephemeral one, then read [`GuiServer::local_addr`]);
/// [`GuiServer::serve`] runs the accept loop and never returns on its own.
pub struct GuiServer {
    listener: TcpListener,
    state: Arc<GuiState>,
}

impl GuiServer {
    /// Bind the dashboard on loopback. Localhost only: this is an operator's
    /// window onto one machine's training run, not a service.
    pub fn bind(root: impl Into<PathBuf>, port: u16) -> Result<Self> {
        let root = root.into();
        let listener = TcpListener::bind(("127.0.0.1", port))
            .with_context(|| format!("bind the nina dashboard on 127.0.0.1:{port}"))?;
        Ok(Self {
            listener,
            state: Arc::new(GuiState { root }),
        })
    }

    /// The address the socket actually bound (the reason `--port 0` is
    /// testable: the kernel picks the port and this reports it).
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener
            .local_addr()
            .context("read the dashboard's bound address")
    }

    /// Serve forever: one thread per accepted connection, so a stalled client
    /// never blocks the dashboard. Connection handlers catch their own
    /// errors; a panic-free handler plus an accept loop that propagates only
    /// listener-level failures is what keeps the server up after a malformed
    /// request.
    pub fn serve(self) -> Result<()> {
        loop {
            let (stream, _peer) = self
                .listener
                .accept()
                .context("accept a dashboard connection")?;
            let state = Arc::clone(&self.state);
            // The JoinHandle is deliberately dropped: the handler owns the
            // connection for its whole life and reports its own failures on
            // the wire; nobody would join it.
            drop(std::thread::spawn(move || handle_connection(state, stream)));
        }
    }
}

/// Serve one connection, then close it (`Connection: close` — the parser is
/// strict and tiny, and keep-alive is not worth a state machine at this
/// scale).
fn handle_connection(state: Arc<GuiState>, stream: TcpStream) {
    if let Err(err) = try_handle(&state, &stream) {
        // Best effort: the client deserves to hear why, but a dead socket
        // must not take the server with it.
        drop(write_response(
            &stream,
            500,
            &format!("internal error: {err:#}\n"),
        ));
    }
}

fn try_handle(state: &GuiState, stream: &TcpStream) -> Result<()> {
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .context("set the read timeout")?;
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .context("set the write timeout")?;
    let mut reader = BufReader::new(stream);
    let Some(line) = read_line_limited(&mut reader, MAX_REQUEST_LINE)? else {
        return Ok(()); // connected and hung up without a word: nothing to answer
    };
    let request = match parse_request_line(&line) {
        Ok(request) => request,
        Err(err) => return write_response(stream, 400, &format!("bad request: {err:#}\n")),
    };
    if let Err(err) = read_headers(&mut reader) {
        return write_response(stream, 400, &format!("bad request: {err:#}\n"));
    }
    if request.method != "GET" {
        return write_response(
            stream,
            405,
            &format!(
                "method not allowed: {} — the dashboard is read-only over HTTP; \
                 writes go through the chained logs\n",
                request.method
            ),
        );
    }
    let (status, body) = route(state, &request.target)?;
    write_response(stream, status, &body)
}

// ---------------------------------------------------------------------------
// The strict, tiny HTTP parser
// ---------------------------------------------------------------------------

/// One parsed request line. The dashboard speaks GET only.
struct Request {
    method: String,
    /// Path plus query, exactly as sent (`/chat?prompt=…`).
    target: String,
}

/// Read one CRLF-terminated line, refusing anything over `limit` bytes. EOF
/// before any byte is `None` (a client that connected and left); EOF or a
/// buffer-full mid-line is an error, because a truncated request is a
/// malformed one.
fn read_line_limited<R: BufRead>(reader: &mut R, limit: usize) -> Result<Option<String>> {
    let mut buf = Vec::new();
    let cap = u64::try_from(limit + 1).context("the line limit fits a u64")?;
    let read = reader
        .by_ref()
        .take(cap)
        .read_until(b'\n', &mut buf)
        .context("read a request line")?;
    if read == 0 {
        return Ok(None);
    }
    ensure!(
        buf.len() <= limit && buf.last() == Some(&b'\n'),
        "the request line exceeds {limit} bytes or is unterminated"
    );
    // The terminating \n was just checked; a preceding \r goes with it.
    buf.truncate(buf.len() - 1);
    if buf.last() == Some(&b'\r') {
        buf.truncate(buf.len() - 1);
    }
    let text = String::from_utf8(buf).context("the request line is not UTF-8")?;
    Ok(Some(text))
}

/// Parse `METHOD SP target SP version` — exactly three space-separated,
/// non-empty parts (a doubled space is a malformed request, not a wide one),
/// an uppercase method, an absolute path, HTTP/1.0 or 1.1.
fn parse_request_line(line: &str) -> Result<Request> {
    let parts: Vec<&str> = line.split(' ').collect();
    let [method, target, version] = parts.as_slice() else {
        anyhow::bail!(
            "the request line must be 'METHOD /path HTTP/1.x' (three space-separated parts); \
             received '{line}'"
        );
    };
    ensure!(
        !method.is_empty() && method.chars().all(|c| c.is_ascii_uppercase()),
        "the method '{method}' is not an uppercase token"
    );
    ensure!(
        target.starts_with('/'),
        "the request target '{target}' is not an absolute path"
    );
    ensure!(
        *version == "HTTP/1.0" || *version == "HTTP/1.1",
        "unsupported HTTP version '{version}' (the dashboard speaks HTTP/1.0 and 1.1)"
    );
    Ok(Request {
        method: (*method).to_string(),
        target: (*target).to_string(),
    })
}

/// Consume headers up to the blank line. Their contents are ignored — the
/// dashboard needs nothing from them — but they are bounded and must end.
fn read_headers<R: BufRead>(reader: &mut R) -> Result<()> {
    let mut total = 0usize;
    loop {
        let line = read_line_limited(reader, MAX_HEADER_LINE)?
            .ok_or_else(|| anyhow::anyhow!("the connection closed mid-headers"))?;
        ensure!(!line.contains('\0'), "a header line carries a NUL byte");
        total += line.len();
        ensure!(
            total <= MAX_HEADER_BYTES,
            "the headers exceed {MAX_HEADER_BYTES} bytes in total"
        );
        if line.is_empty() {
            return Ok(());
        }
    }
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    }
}

fn write_response(stream: &TcpStream, status: u16, body: &str) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        status_text(status),
        body.len()
    );
    let mut writer = stream;
    writer
        .write_all(head.as_bytes())
        .and_then(|()| writer.write_all(body.as_bytes()))
        .and_then(|()| writer.flush())
        .with_context(|| format!("write the {status} response"))
}

// ---------------------------------------------------------------------------
// Query strings
// ---------------------------------------------------------------------------

/// Decode one query component: `+` is a space, `%XX` a byte, a malformed `%`
/// stays literal (a dashboard query is not an attack surface worth 400ing).
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes.get(i) {
            Some(b'+') => out.push(b' '),
            Some(b'%') => {
                let hex = i + 1;
                let digit = |k: usize| {
                    bytes
                        .get(k)
                        .and_then(|b| char::from(*b).to_digit(16))
                        .and_then(|d| u8::try_from(d).ok())
                };
                match (digit(hex), digit(hex + 1)) {
                    (Some(hi), Some(lo)) => {
                        out.push(hi * 16 + lo);
                        i += 2;
                    }
                    _ => out.push(b'%'),
                }
            }
            Some(b) => out.push(*b),
            None => {}
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parse `a=1&b=two` into decoded pairs. Empty segments are skipped.
fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|part| !part.is_empty())
        .map(|part| match part.split_once('=') {
            Some((key, value)) => (percent_decode(key), percent_decode(value)),
            None => (percent_decode(part), String::new()),
        })
        .collect()
}

/// The first value of a query parameter, when present and non-blank.
fn query_param<'a>(params: &'a [(String, String)], key: &str) -> Option<&'a str> {
    params
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .filter(|v| !v.trim().is_empty())
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

fn route(state: &GuiState, target: &str) -> Result<(u16, String)> {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let params = parse_query(query);
    match path {
        "/" => Ok((200, page_overview(state))),
        "/live" => Ok((200, page_live(state, &params))),
        "/thinking" => Ok((200, page_thinking(state, &params))),
        "/chat" => Ok((200, page_chat(state, &params))),
        "/data" => Ok((200, page_data(state))),
        _ => Ok((404, page_not_found(path))),
    }
}

// ---------------------------------------------------------------------------
// HTML scaffolding
// ---------------------------------------------------------------------------

/// Escape text that came from a log, a client or a model before it goes
/// anywhere near markup.
fn esc(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// The page frame every route renders through: nav, then the chain-status
/// line (the "saved in a way AI can't edit" surface), then the body.
fn page(title: &str, badge: &str, head_extra: &str, body: &str) -> String {
    format!(
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\
         <title>nina · {title}</title>{head_extra}<style>{CSS}</style></head><body>\
         <header><h1>nina <small>from cybersec.org.za</small></h1><nav>\
         <a href=\"/\">overview</a> · <a href=\"/live\">live</a> · \
         <a href=\"/thinking\">thinking</a> · <a href=\"/chat\">chat</a> · \
         <a href=\"/data\">data</a></nav>\
         <div class=\"chain\">{badge}</div></header>\
         <main>{body}</main></body></html>\n",
        title = esc(title),
    )
}

const CSS: &str = "body{font-family:ui-monospace,Menlo,Consolas,monospace;margin:2em;\
    max-width:1100px;color:#1a1a1a}h1{margin:0}h1 small{font-weight:normal;color:#666}\
    nav{margin:.4em 0}.chain{margin:.6em 0;padding:.4em .6em;border:1px solid #ccc}\
    .chain-ok{color:#146c2e;font-weight:bold}.chain-bad{color:#b00020;font-weight:bold}\
    .chain-none{color:#666}.error,.empty{border:1px solid #b00020;color:#b00020;\
    padding:.6em .8em;margin:.8em 0;font-weight:bold}\
    .empty{color:#555;border-color:#999}\
    table{border-collapse:collapse;margin:.8em 0}\
    td,th{border:1px solid #bbb;padding:.25em .7em;text-align:left}\
    th{background:#eee}pre{background:#f5f5f5;border:1px solid #ddd;padding:.8em;\
    overflow-x:auto}input[name=prompt]{width:36em;font-family:inherit}\
    .turn{border-top:2px solid #999;margin-top:1.2em;padding-top:.6em}\
    .meta{color:#666;font-size:.9em}";

/// The chain-verification line for one log, re-derived at request time: this
/// is the check that makes an edited history *show* red.
fn chain_badge(path: &Path) -> String {
    if !path.exists() {
        return format!(
            "<span class=\"chain-none\">no log yet: <code>{}</code></span>",
            esc(&path.display().to_string())
        );
    }
    match ChainedLog::verify(path) {
        Ok(report) => {
            let head: String = report.head_hash.chars().take(12).collect();
            format!(
                "<span class=\"chain-ok\">chain ok · lines={} · head={}…</span> \
                 <code>{}</code>",
                report.lines,
                esc(&head),
                esc(&path.display().to_string())
            )
        }
        Err(err) => format!(
            "<span class=\"chain-bad\">TAMPERED: {err}</span> <code>{}</code>",
            esc(&path.display().to_string())
        ),
    }
}

/// The last `n` lines of a file, reading at most [`TAIL_READ_CAP`] bytes of
/// its tail so a big log cannot make the dashboard slow.
fn tail_lines(path: &Path, n: usize) -> Result<Vec<String>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let len = bytes.len() as u64; // audit-allow: a file length always fits u64
    let start = if len > TAIL_READ_CAP {
        let from = usize::try_from(len - TAIL_READ_CAP).unwrap_or(0);
        // Start after the next newline so the first shown line is whole.
        bytes
            .iter()
            .skip(from)
            .position(|b| *b == b'\n')
            .map_or(from, |p| from + p + 1)
    } else {
        0
    };
    let text = String::from_utf8_lossy(bytes.get(start..).unwrap_or(&[]));
    let lines: Vec<&str> = text.lines().collect();
    let skip = lines.len().saturating_sub(n);
    Ok(lines.iter().skip(skip).map(|s| (*s).to_string()).collect())
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64; // audit-allow: display-only; sub-unit rounding is fine
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    let name = UNITS.get(unit).copied().unwrap_or("TiB");
    if unit == 0 {
        format!("{bytes} {name}")
    } else {
        format!("{value:.1} {name}")
    }
}

fn opt_f64(value: Option<f64>) -> String {
    value.map_or("—".to_string(), |v| format!("{v:.4}"))
}

// ---------------------------------------------------------------------------
// / — overview: the model map + the chain states
// ---------------------------------------------------------------------------

fn page_overview(state: &GuiState) -> String {
    let paths = state.paths();
    let badge = chain_badge(&paths.registry_log());
    let mut body = String::from("<h2>model registry</h2>");
    let registry_path = paths.registry_json();
    if !registry_path.exists() {
        body.push_str(&format!(
            "<div class=\"empty\">no registry at <code>{}</code> yet — register a model with \
             `dblocks nina mcp-serve` (tool `nina.register`) and it appears here.</div>",
            esc(&registry_path.display().to_string())
        ));
    } else {
        match RegistryFile::load(&registry_path) {
            Ok(file) if file.models.is_empty() => body.push_str(
                "<div class=\"empty\">the registry exists but holds no models yet.</div>",
            ),
            Ok(file) => {
                body.push_str(
                    "<table><tr><th>name</th><th>field</th><th>status</th>\
                    <th>accuracy</th><th>router top-1</th><th>eval date</th><th>bytes</th>\
                    <th>box</th><th>checkpoint</th></tr>",
                );
                for m in &file.models {
                    let hash: String = m.checkpoint_hash.chars().take(12).collect();
                    body.push_str(&format!(
                        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\
                         <td>{}</td><td>{}</td><td>{}</td></tr>",
                        esc(&m.name),
                        esc(&m.field),
                        m.status.as_str(),
                        opt_f64(m.eval.accuracy),
                        opt_f64(m.eval.router_top1),
                        esc(if m.eval.date.is_empty() {
                            "—"
                        } else {
                            &m.eval.date
                        }),
                        human_bytes(m.bytes),
                        esc(if m.box_id.is_empty() {
                            "—"
                        } else {
                            &m.box_id
                        }),
                        esc(if hash.is_empty() { "—" } else { &hash }),
                    ));
                }
                body.push_str("</table>");
            }
            Err(err) => body.push_str(&format!(
                "<div class=\"error\">the registry at <code>{}</code> is unreadable: {err}</div>",
                esc(&registry_path.display().to_string())
            )),
        }
    }
    body.push_str("<h2>chained logs</h2>");
    body.push_str(&logs_section(state));
    body.push_str(
        "<p class=\"meta\">the registry above is derived state; the truth is the chained log, \
         re-verified on every request. An edited line turns the badge red and names the line.</p>",
    );
    page("overview", &badge, "", &body)
}

/// Every `*.jsonl` under `<root>/logs/`, with its chain state.
fn logs_section(state: &GuiState) -> String {
    let dir = state.logs_dir();
    if !dir.is_dir() {
        return format!(
            "<div class=\"empty\">no logs directory at <code>{}</code> yet — the first registered \
             model, queued job or chat turn creates it.</div>",
            esc(&dir.display().to_string())
        );
    }
    let mut names: Vec<PathBuf> = Vec::new();
    match std::fs::read_dir(&dir) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "jsonl") {
                    names.push(path);
                }
            }
        }
        Err(err) => {
            return format!(
                "<div class=\"error\">cannot list <code>{}</code>: {err}</div>",
                esc(&dir.display().to_string())
            )
        }
    }
    names.sort();
    if names.is_empty() {
        return "<div class=\"empty\">the logs directory holds no chained logs yet.</div>"
            .to_string();
    }
    let mut out = String::new();
    for path in &names {
        out.push_str(&format!("<div>{}</div>", chain_badge(path)));
    }
    out
}

// ---------------------------------------------------------------------------
// /live — the tails of the chained logs
// ---------------------------------------------------------------------------

fn page_live(state: &GuiState, params: &[(String, String)]) -> String {
    let n = query_param(params, "n")
        .and_then(|v| v.parse::<usize>().ok())
        .map_or(DEFAULT_TAIL, |v| v.clamp(1, MAX_TAIL));
    let dir = state.logs_dir();
    let badge = "<span class=\"chain-none\">each log below is re-verified on this request</span>";
    let mut body = format!("<h2>live tails (last {n} records per log, refreshes every 5s)</h2>");
    let mut names: Vec<PathBuf> = Vec::new();
    if dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "jsonl") {
                    names.push(path);
                }
            }
        }
        names.sort();
    }
    if names.is_empty() {
        body.push_str(&format!(
            "<div class=\"empty\">no chained logs under <code>{}</code> yet — training queue, \
             registry and chat records appear here as they are appended.</div>",
            esc(&dir.display().to_string())
        ));
    }
    for path in &names {
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |s| s.to_string_lossy().into_owned(),
        );
        body.push_str(&format!("<h3>{}</h3>{}", esc(&name), chain_badge(path)));
        match tail_lines(path, n) {
            Ok(lines) if lines.is_empty() => {
                body.push_str("<div class=\"empty\">the log is empty.</div>")
            }
            Ok(lines) => {
                body.push_str("<pre>");
                for line in &lines {
                    body.push_str(&esc(line));
                    body.push('\n');
                }
                body.push_str("</pre>");
            }
            Err(err) => body.push_str(&format!("<div class=\"error\">{err}</div>")),
        }
    }
    page(
        "live",
        badge,
        "<meta http-equiv=\"refresh\" content=\"5\">",
        &body,
    )
}

// ---------------------------------------------------------------------------
// The student, when there is one
// ---------------------------------------------------------------------------

/// A loaded student plus the facts the pages need about it.
struct LoadedStudent {
    model: crate::student::GeometricStudent<Eval>,
    path: PathBuf,
    context: usize,
}

/// Load the newest student checkpoint under `<root>/student/`, or `None` when
/// there is none — the skip-if-absent discipline: pages that need a model
/// say so loudly instead of inventing one.
fn load_student(paths: &NinaPaths) -> Result<Option<LoadedStudent>> {
    use crate::student::{GeometricStudent, StudentConfig};
    let Some(path) = crate::checkpoint::latest_in_dir(&paths.student_dir(), "student")? else {
        return Ok(None);
    };
    let config = match crate::checkpoint::TrainState::for_model(&path)? {
        Some(state) => match state.config.get("student_config") {
            Some(value) => serde_json::from_value(value.clone()).with_context(|| {
                format!(
                    "parse the student config recorded beside {}",
                    path.display()
                )
            })?,
            None => StudentConfig::default(),
        },
        None => StudentConfig::default(),
    };
    let context = config.context;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <Eval as burn::tensor::backend::Backend>::seed(&device, 0);
    let model = GeometricStudent::<Eval>::new(&config, &device)?;
    crate::tensor_ext::force_initialization(&model);
    let model = crate::checkpoint::load::<Eval, _>(model, &path, &device)?;
    Ok(Some(LoadedStudent {
        model,
        path,
        context,
    }))
}

/// Encode a prompt for the student, truncating to the last `context` tokens
/// (rotary positions make length unbounded; the cap keeps a turn cheap). The
/// second return says the truncation happened, so the page can admit it.
fn encode_window(loaded: &LoadedStudent, prompt: &str) -> (Vec<u16>, bool) {
    let mut ids = ByteTokenizer::new().encode_document(prompt);
    let truncated = ids.len() > loaded.context;
    if truncated {
        let keep = ids.split_off(ids.len() - loaded.context);
        ids = keep;
    }
    (ids, truncated)
}

fn token_tensor(ids: &[u16]) -> Result<burn::tensor::Tensor<Eval, 2, burn::tensor::Int>> {
    use burn::tensor::{Int, Tensor};
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let raw: Vec<i64> = ids.iter().map(|t| i64::from(*t)).collect();
    let n = raw.len();
    Ok(Tensor::<Eval, 1, Int>::from_ints(raw.as_slice(), &device).reshape([1, n]))
}

/// Plain greedy, seeded generation: `≤ GENERATE_MAX_NEW` tokens, stopping at
/// `<eos>`. This is the cheapest honest answer the student can give — no
/// sampler, no temperature, and the page labels it as exactly that.
fn generate_greedy(loaded: &LoadedStudent, prompt: &str) -> Result<String> {
    let (mut ids, _) = encode_window(loaded, prompt);
    let prompt_len = ids.len();
    for _ in 0..GENERATE_MAX_NEW {
        let window_start = ids.len().saturating_sub(loaded.context);
        let window: Vec<u16> = ids.iter().skip(window_start).copied().collect();
        let out = loaded.model.forward(token_tensor(&window)?)?;
        let [_, n, vocab] = out.logits.dims();
        let last = out
            .logits
            .slice([0..1, (n - 1)..n, 0..vocab])
            .reshape([vocab]);
        let data = last.argmax(0).into_data();
        let raw = data
            .convert::<i64>()
            .iter::<i64>()
            .next()
            .ok_or_else(|| anyhow::anyhow!("greedy decode: the argmax row was empty"))?;
        let id = u16::try_from(raw).map_err(|_| {
            anyhow::anyhow!("greedy decode: token id {raw} is outside the byte vocabulary")
        })?;
        if id == Special::Eos.id() {
            break;
        }
        ids.push(id);
    }
    let new_ids: Vec<u16> = ids.iter().skip(prompt_len).copied().collect();
    Ok(ByteTokenizer::new().decode_lossy(&new_ids))
}

// ---------------------------------------------------------------------------
// /thinking — the vecread thought trace
// ---------------------------------------------------------------------------

fn page_thinking(state: &GuiState, params: &[(String, String)]) -> String {
    let paths = state.paths();
    let badge = format!(
        "<span class=\"chain-none\">reads the newest checkpoint under <code>{}</code> \
         (content-addressed .mpk); no log is read on this page</span>",
        esc(&paths.student_dir().display().to_string())
    );
    let mut body = String::from(
        "<h2>what the model is thinking</h2>\
         <form action=\"/thinking\" method=\"get\">\
         <input name=\"prompt\" placeholder=\"a prompt to read the trace of\">\
         <button>think</button></form>",
    );
    if let Some(prompt) = query_param(params, "prompt") {
        match think_inner(state, prompt) {
            Ok(html) => body.push_str(&html),
            Err(err) => body.push_str(&format!(
                "<div class=\"error\">the thought trace failed: {err}</div>"
            )),
        }
    }
    page("thinking", &badge, "", &body)
}

fn think_inner(state: &GuiState, prompt: &str) -> Result<String> {
    let paths = state.paths();
    let Some(loaded) = load_student(&paths)? else {
        return Ok(format!(
            "<div class=\"empty\">no student checkpoint under <code>{}</code> — the thinking \
             view needs a trained student (`dblocks student train` writes one there). Nothing \
             is rendered in its place: an invented trace would be a lie with styling.</div>",
            esc(&paths.student_dir().display().to_string())
        ));
    };
    let (ids, truncated) = encode_window(&loaded, prompt);
    let trace = crate::vecread::VecReader::default().read_student_thinking(
        &loaded.model,
        token_tensor(&ids)?,
        &THINK_DEPTHS,
    )?;
    let mut out = format!(
        "<p class=\"meta\">checkpoint: <code>{}</code> · prompt: <code>{}</code>{}</p>",
        esc(&loaded.path.display().to_string()),
        esc(prompt),
        if truncated {
            " · prompt truncated to the student's context window"
        } else {
            ""
        },
    );
    out.push_str(&format!("<pre>{}</pre>", esc(&trace.render())));
    Ok(out)
}

// ---------------------------------------------------------------------------
// /chat — routed, answered, and logged into the chain
// ---------------------------------------------------------------------------

fn page_chat(state: &GuiState, params: &[(String, String)]) -> String {
    let chat_log = state.chat_log();
    let badge = chain_badge(&chat_log);
    let mut body = String::from(
        "<h2>chat</h2>\
         <form action=\"/chat\" method=\"get\">\
         <input name=\"prompt\" placeholder=\"ask nina…\"><button>send</button></form>",
    );
    if let Some(prompt) = query_param(params, "prompt") {
        body.push_str(&chat_turn(state, prompt));
    }
    body.push_str(&chat_history(&chat_log, 10));
    page("chat", &badge, "", &body)
}

/// One chat turn. Failures render as loud red boxes *inside* the page — a
/// dashboard that 500s on a bad turn hides exactly what it exists to show.
fn chat_turn(state: &GuiState, prompt: &str) -> String {
    match chat_turn_inner(state, prompt) {
        Ok(html) => html,
        Err(err) => format!(
            "<div class=\"turn\"><div class=\"error\">the chat turn failed before it could be \
             logged: {err}</div></div>"
        ),
    }
}

fn chat_turn_inner(state: &GuiState, prompt: &str) -> Result<String> {
    let paths = state.paths();
    let (fields, registry_note) = match RegistryFile::load(&paths.registry_json()) {
        Ok(file) => (file.fields(), None),
        Err(err) => (
            Vec::new(),
            Some(format!(
                "the registry is unreadable ({err:#}); the field list is empty"
            )),
        ),
    };
    let checkpoint = crate::checkpoint::latest_in_dir(&paths.student_dir(), "student")?;
    // The same routing the MCP server's `nina.route` tool computes.
    let routed = crate::nina::route_prompt(prompt, checkpoint.as_deref(), &fields)?;

    // The answer: greedy text when a student is loaded, an honest "pending"
    // otherwise. A failed generation is recorded as failed, never invented.
    let (answer, generation_note) = match checkpoint.as_deref() {
        None => (
            None,
            "pending: no student checkpoint, so there is no model to answer with".to_string(),
        ),
        Some(_) => match load_student(&paths).and_then(|loaded| {
            let loaded = loaded.ok_or_else(|| {
                anyhow::anyhow!("the checkpoint listed a moment ago no longer loads")
            })?;
            Ok((generate_greedy(&loaded, prompt)?, loaded.path))
        }) {
            Ok((text, path)) => (
                Some(text),
                format!(
                    "greedy, seeded, at most {GENERATE_MAX_NEW} new tokens, checkpoint {}",
                    path.display()
                ),
            ),
            Err(err) => (None, format!("failed loudly: {err:#}")),
        },
    };

    // The turn is history: append it to the chained chat log *before*
    // rendering, so what the page shows and what the chain holds cannot drift.
    let mut log = ChainedLog::open(state.chat_log())?;
    let record = log.append(&json!({
        "event": "chat",
        "prompt": prompt,
        "router": routed.get("router").cloned().unwrap_or(Value::Null),
        "top": routed.get("top").cloned().unwrap_or(Value::Null),
        "answer": answer,
        "generation": generation_note,
        "registry_note": registry_note,
    }))?;

    let mut out = String::from("<div class=\"turn\">");
    out.push_str(&format!("<p><b>you:</b> {}</p>", esc(prompt)));
    out.push_str(&render_route(&routed));
    match &answer {
        Some(text) => out.push_str(&format!(
            "<p><b>nina:</b> <span class=\"meta\">({})</span></p><pre>{}</pre>",
            esc(&generation_note),
            esc(text)
        )),
        None => out.push_str(&format!(
            "<div class=\"empty\">generation {} — the routing above is the whole honest answer \
             so far; no reply was fabricated.</div>",
            esc(&generation_note)
        )),
    }
    let hash: String = record.hash.chars().take(12).collect();
    out.push_str(&format!(
        "<p class=\"meta\">logged to <code>{}</code> as record seq={} hash={}… — append-only, \
         chain-verified on the next request.</p></div>",
        esc(&log.path().display().to_string()),
        record.seq,
        esc(&hash),
    ));
    Ok(out)
}

/// Render what `route_prompt` returned: either the probe router's box-traffic
/// table or its honest no-router note plus the registry's field list.
fn render_route(routed: &Value) -> String {
    let mut out = String::new();
    let router = routed.get("router").cloned().unwrap_or(Value::Null);
    if router.is_null() {
        let note = routed
            .get("note")
            .and_then(Value::as_str)
            .unwrap_or("no routing note");
        out.push_str(&format!("<div class=\"empty\">{}</div>", esc(note)));
        if let Some(fields) = routed.get("fields").and_then(Value::as_array) {
            let names: Vec<String> = fields
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
            if !names.is_empty() {
                out.push_str(&format!(
                    "<p class=\"meta\">known fields: {}</p>",
                    esc(&names.join(", "))
                ));
            }
        }
        return out;
    }
    let top = routed
        .get("top")
        .and_then(Value::as_str)
        .unwrap_or("(none above the rest)");
    out.push_str(&format!(
        "<p><b>routed to:</b> {} <span class=\"meta\">(probe router)</span></p>",
        esc(top)
    ));
    if let Some(traffic) = routed.get("traffic").and_then(Value::as_array) {
        out.push_str(
            "<table><tr><th>box</th><th>language</th><th>traffic</th>\
                      <th>top-1 tokens</th></tr>",
        );
        for row in traffic {
            let language = row.get("language").and_then(Value::as_str).unwrap_or("?");
            let box_id = row.get("box_id").and_then(Value::as_str).unwrap_or("?");
            let gate = row.get("traffic").and_then(Value::as_f64).unwrap_or(0.0);
            let top1 = row.get("top1_tokens").and_then(Value::as_u64).unwrap_or(0);
            let bold = language == top;
            let (open, close) = if bold { ("<b>", "</b>") } else { ("", "") };
            out.push_str(&format!(
                "<tr>{open}<td>{}</td><td>{}</td><td>{gate:.4}</td><td>{top1}</td>{close}</tr>",
                esc(box_id),
                esc(language),
            ));
        }
        out.push_str("</table>");
    }
    out
}

/// The recent chat history, read back out of the chained log — the page shows
/// the chain's contents, not a side cache.
fn chat_history(chat_log: &Path, n: usize) -> String {
    let mut out = String::from("<h3>recent turns</h3>");
    if !chat_log.exists() {
        out.push_str(
            "<div class=\"empty\">no chat history yet — the first turn above creates \
             <code>logs/chat.jsonl</code>.</div>",
        );
        return out;
    }
    let lines = match tail_lines(chat_log, n) {
        Ok(lines) => lines,
        Err(err) => {
            out.push_str(&format!(
                "<div class=\"error\">cannot read the chat log: {err}</div>"
            ));
            return out;
        }
    };
    if lines.is_empty() {
        out.push_str("<div class=\"empty\">the chat log is empty.</div>");
        return out;
    }
    for line in lines.iter().rev() {
        match serde_json::from_str::<Value>(line) {
            Ok(record) => {
                let payload = record.get("payload").cloned().unwrap_or(Value::Null);
                let seq = record.get("seq").and_then(Value::as_u64).unwrap_or(0);
                let prompt = payload.get("prompt").and_then(Value::as_str).unwrap_or("?");
                let top = payload.get("top").and_then(Value::as_str);
                let answer = payload.get("answer").and_then(Value::as_str);
                out.push_str(&format!(
                    "<div class=\"turn\"><p class=\"meta\">seq={seq}</p><p><b>you:</b> {}</p>",
                    esc(prompt)
                ));
                if let Some(top) = top {
                    out.push_str(&format!("<p><b>routed to:</b> {}</p>", esc(top)));
                }
                match answer {
                    Some(text) => out.push_str(&format!("<pre>{}</pre>", esc(text))),
                    None => out.push_str(
                        "<p class=\"meta\">(no generated answer recorded for this turn)</p>",
                    ),
                }
                out.push_str("</div>");
            }
            // A line that no longer parses is exactly what the red badge above
            // is reporting; show it raw rather than hiding it.
            Err(_) => out.push_str(&format!("<pre>{}</pre>", esc(line))),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// /data — corpus manifests under the root
// ---------------------------------------------------------------------------

fn page_data(state: &GuiState) -> String {
    let badge = "<span class=\"chain-none\">reads corpus manifests (<code>*/dataset/manifest.json</code>), \
                 not a chained log</span>";
    let mut body = String::from("<h2>corpora</h2>");
    let mut manifests: Vec<(String, PathBuf)> = Vec::new();
    if state.root.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&state.root) {
            for entry in entries.flatten() {
                let manifest = entry.path().join("dataset").join("manifest.json");
                if manifest.is_file() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    manifests.push((name, manifest));
                }
            }
        }
        manifests.sort();
    }
    if manifests.is_empty() {
        body.push_str(&format!(
            "<div class=\"empty\">no corpus manifests under <code>{}</code> — a corpus built in \
             the cybercorpus layout (<code>&lt;field&gt;corpus/dataset/manifest.json</code>) \
             appears here with its per-source document and character counts.</div>",
            esc(&state.root.display().to_string())
        ));
    }
    for (name, manifest) in &manifests {
        body.push_str(&format!("<h3>{}</h3>", esc(name)));
        match std::fs::read_to_string(manifest)
            .with_context(|| format!("read {}", manifest.display()))
            .and_then(|text| {
                serde_json::from_str::<Value>(&text)
                    .with_context(|| format!("parse {}", manifest.display()))
            }) {
            Ok(value) => body.push_str(&render_manifest(&value)),
            Err(err) => body.push_str(&format!("<div class=\"error\">{err}</div>")),
        }
    }
    page("data", badge, "", &body)
}

/// Render one manifest: a per-source table over the union of its numeric
/// fields (documents, chars, and any drop/reject counts a builder recorded),
/// with a totals row. A manifest of any other shape is shown verbatim — the
/// page reports what is there, not what it wishes were there.
fn render_manifest(manifest: &Value) -> String {
    let Some(entries) = manifest.as_object() else {
        return format!("<pre>{}</pre>", esc(&manifest.to_string()));
    };
    let mut keys: Vec<String> = Vec::new();
    for value in entries.values() {
        if let Some(object) = value.as_object() {
            for (key, value) in object {
                if value.is_number() && !keys.contains(key) {
                    keys.push(key.clone());
                }
            }
        }
    }
    if keys.is_empty() {
        return format!("<pre>{}</pre>", esc(&manifest.to_string()));
    }
    keys.sort();
    let mut out = String::from("<table><tr><th>source</th>");
    for key in &keys {
        out.push_str(&format!("<th>{}</th>", esc(key)));
    }
    out.push_str("</tr>");
    let mut totals = vec![0u64; keys.len()];
    let mut rows: Vec<(&String, &Value)> = entries.iter().collect();
    rows.sort_by_key(|(name, _)| (*name).clone());
    for (source, value) in rows {
        out.push_str(&format!("<tr><td>{}</td>", esc(source)));
        for (i, key) in keys.iter().enumerate() {
            let number = value.get(key).and_then(Value::as_u64);
            if let (Some(slot), Some(number)) = (totals.get_mut(i), number) {
                *slot = slot.saturating_add(number);
            }
            out.push_str(&format!(
                "<td>{}</td>",
                number.map_or("—".to_string(), |v| v.to_string())
            ));
        }
        out.push_str("</tr>");
    }
    out.push_str("<tr><th>total</th>");
    for total in &totals {
        out.push_str(&format!("<th>{total}</th>"));
    }
    out.push_str("</tr></table>");
    out
}

fn page_not_found(path: &str) -> String {
    let body = format!(
        "<div class=\"error\">unknown path <code>{}</code> — the dashboard serves \
         <a href=\"/\">/</a>, <a href=\"/live\">/live</a>, <a href=\"/thinking\">/thinking</a>, \
         <a href=\"/chat\">/chat</a> and <a href=\"/data\">/data</a>.</div>",
        esc(path)
    );
    page("not found", "", "", &body)
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
    use crate::nina::{ModelEntry, Registry};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SCRATCH: AtomicUsize = AtomicUsize::new(0);

    fn scratch(stem: &str) -> PathBuf {
        let n = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("dblocks-ninagui-{stem}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Bind on an ephemeral port and serve on a thread that outlives the
    /// test; the test process reaps it at exit.
    fn start(root: &Path) -> SocketAddr {
        let server = GuiServer::bind(root, 0).unwrap();
        let addr = server.local_addr().unwrap();
        std::thread::spawn(move || server.serve());
        addr
    }

    /// Send a raw request and read the whole response (Connection: close).
    fn raw_exchange(addr: SocketAddr, request: &[u8]) -> String {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(15)))
            .unwrap();
        stream.write_all(request).unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).unwrap();
        String::from_utf8_lossy(&buf).into_owned()
    }

    fn http_get(addr: SocketAddr, target: &str) -> (u16, String) {
        let response = raw_exchange(
            addr,
            format!("GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes(),
        );
        let status: u16 = response
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap();
        (status, response)
    }

    fn seed_registry(root: &Path) -> PathBuf {
        let mut registry =
            Registry::open(root.join("registry.json"), root.join("logs/registry.jsonl")).unwrap();
        registry
            .register(ModelEntry {
                name: "coding/rust".into(),
                field: "coding".into(),
                subfield: "rust".into(),
                box_id: "coding:rust".into(),
                bytes: 12_884_901_888,
                eval: crate::nina::EvalEvidence {
                    accuracy: Some(0.91),
                    router_top1: Some(1.0),
                    date: "2026-10-10".into(),
                },
                ..ModelEntry::default()
            })
            .unwrap();
        root.join("logs/registry.jsonl")
    }

    #[test]
    fn overview_on_an_empty_root_is_200_html_with_empty_states() {
        let dir = scratch("empty");
        let addr = start(&dir);
        let (status, response) = http_get(addr, "/");
        assert_eq!(status, 200, "{response}");
        assert!(response.contains("Content-Type: text/html"), "{response}");
        assert!(response.contains("no registry at"), "{response}");
        assert!(response.contains("no logs directory"), "{response}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn overview_renders_the_registry_and_the_chain_state() {
        let dir = scratch("seeded");
        seed_registry(&dir);
        let addr = start(&dir);
        let (status, body) = http_get(addr, "/");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("coding/rust"), "{body}");
        assert!(body.contains("0.9100"), "{body}");
        assert!(body.contains("12.0 GiB"), "{body}");
        assert!(body.contains("chain ok · lines=1 · head="), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tampered_log_turns_the_overview_red() {
        let dir = scratch("tampered");
        let log_path = seed_registry(&dir);
        let text = std::fs::read_to_string(&log_path).unwrap();
        let edited = text.replacen("coding/rust", "coding/ruby", 1);
        assert_ne!(edited, text);
        std::fs::write(&log_path, edited).unwrap();
        let addr = start(&dir);
        let (status, body) = http_get(addr, "/");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("TAMPERED"), "{body}");
        assert!(body.contains("line 1"), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn live_tails_the_chained_logs() {
        let dir = scratch("live");
        let mut queue = ChainedLog::open(dir.join("logs/queue.jsonl")).unwrap();
        queue
            .append(&json!({"event": "queue.add", "name": "pentest/recon"}))
            .unwrap();
        let addr = start(&dir);
        let (status, body) = http_get(addr, "/live");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("pentest/recon"), "{body}");
        assert!(body.contains("chain ok · lines=1"), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_chat_turn_appends_a_record_that_verifies() {
        let dir = scratch("chat");
        let addr = start(&dir);
        let (status, body) = http_get(addr, "/chat?prompt=hello%20nina");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("hello nina"), "{body}");
        // No checkpoint under the root: the router's honest fallback and a
        // pending generation, not a fabricated reply.
        assert!(body.contains("no trained router is loaded"), "{body}");
        assert!(body.contains("pending: no student checkpoint"), "{body}");
        // The turn landed in the chained log and the chain verifies.
        let report = ChainedLog::verify(&dir.join("logs/chat.jsonl")).unwrap();
        assert_eq!(report.lines, 1);
        let line = std::fs::read_to_string(dir.join("logs/chat.jsonl")).unwrap();
        assert!(line.contains("\"event\":\"chat\""), "{line}");
        assert!(line.contains("hello nina"), "{line}");
        // And the history section renders it back out of the log.
        let (_, body) = http_get(addr, "/chat");
        assert!(body.contains("recent turns"), "{body}");
        assert!(body.contains("hello nina"), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn thinking_without_a_checkpoint_is_a_loud_note() {
        let dir = scratch("thinking");
        let addr = start(&dir);
        let (status, body) = http_get(addr, "/thinking?prompt=fn%20main");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("no student checkpoint"), "{body}");
        assert!(body.contains("dblocks student train"), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn data_page_lists_manifests_and_empty_states() {
        let dir = scratch("data");
        let addr = start(&dir);
        let (status, body) = http_get(addr, "/data");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("no corpus manifests"), "{body}");

        let dataset = dir.join("pentestcorpus/dataset");
        std::fs::create_dir_all(&dataset).unwrap();
        std::fs::write(
            dataset.join("manifest.json"),
            r#"{"seclists": {"documents": 10, "chars": 1234}, "cve": {"documents": 3, "chars": 99}}"#,
        )
        .unwrap();
        let (status, body) = http_get(addr, "/data");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("pentestcorpus"), "{body}");
        assert!(body.contains("seclists"), "{body}");
        assert!(body.contains("<th>13</th>"), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_path_is_a_404_naming_the_routes() {
        let dir = scratch("404");
        let addr = start(&dir);
        let (status, body) = http_get(addr, "/nope");
        assert_eq!(status, 404, "{body}");
        assert!(body.contains("unknown path"), "{body}");
        assert!(body.contains("/chat"), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_malformed_request_is_400_and_the_server_stays_up() {
        let dir = scratch("400");
        let addr = start(&dir);
        let response = raw_exchange(addr, b"GARBAGE\r\n\r\n");
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
        assert!(response.contains("bad request"), "{response}");
        let response = raw_exchange(addr, b"GET / HTTP/1.9\r\n\r\n");
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
        // The server survived both: a valid request still gets its page.
        let (status, body) = http_get(addr, "/");
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("nina"), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn query_decoding_handles_spaces_percent_and_malformed() {
        let params = parse_query("prompt=hello+world%21&empty=&flag");
        assert_eq!(query_param(&params, "prompt"), Some("hello world!"));
        assert_eq!(query_param(&params, "empty"), None);
        assert_eq!(query_param(&params, "flag"), None); // blank values are absent
        let params = parse_query("x=100%zz%");
        assert_eq!(query_param(&params, "x"), Some("100%zz%"));
    }

    #[test]
    fn the_request_line_parser_is_strict() {
        let request = parse_request_line("GET /chat?prompt=x HTTP/1.1").unwrap();
        assert_eq!(request.method, "GET");
        assert_eq!(request.target, "/chat?prompt=x");
        assert!(parse_request_line("GET / HTTP/1.1").is_ok());
        assert!(parse_request_line("GET  / HTTP/1.1").is_err()); // doubled space
        assert!(parse_request_line("get / HTTP/1.1").is_err());
        assert!(parse_request_line("GET / HTTP/2").is_err());
        assert!(parse_request_line("GET http://x/ HTTP/1.1").is_err());
        assert!(parse_request_line("GET /").is_err());
    }

    #[test]
    fn tail_lines_caps_and_takes_the_last_n() {
        let dir = scratch("tail");
        let path = dir.join("l.jsonl");
        let text: String = (0..50).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&path, text).unwrap();
        let lines = tail_lines(&path, 3).unwrap();
        assert_eq!(lines, ["line 47", "line 48", "line 49"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
