//! NVIDIA NIM teacher client: generates per-language coding training traces
//! by querying an OpenAI-compatible chat-completions endpoint.
//!
//! The crate has no HTTP stack, so requests go through the `curl` binary via
//! [`std::process::Command`], the same way [`crate::codequality`] shells out
//! to external analyzers. HTTPS makes a raw `TcpStream` a non-starter; `curl`
//! carries the TLS stack. Its absence on `PATH` is a loud construction error,
//! not a runtime surprise.
//!
//! # Keys and cost
//!
//! Everything in this module works without a key: [`NimClient::new`] takes
//! the key explicitly, and the unit tests pass a dummy against a local mock
//! server. [`NimClient::from_env`] is the one that fails loudly when
//! `NVIDIA_API_KEY` (or whatever [`NimConfig::api_key_env`] names) is unset,
//! and its error says how to set it.
//!
//! Every completed request logs the model, the requested `max_tokens` and the
//! returned token usage to stderr. [`NimClient::generate_traces`] carries a
//! hard [`TraceBudget`] whose default ([`TraceBudget::default`]) is
//! deliberately tiny; a caller that wants a bigger run must raise
//! `max_requests` explicitly.
//!
//! # Reading traces back
//!
//! The read side of the pipeline is keyless and file-local: [`read_traces`]
//! parses a JSONL trace file strictly (corrupt lines and duplicate ids are
//! loud errors), [`trace_stats`] reports per-language counts and the token
//! cost the endpoint actually billed, and [`split_traces`] divides records
//! into train/validation folds by id hash, so a resumed or extended trace
//! file never reshuffles the validation set.
//!
//! # Providers
//!
//! The client speaks to any OpenAI-compatible endpoint; [`Provider`] names
//! the three it is configured for: NVIDIA's hosted NIM catalogue (the
//! default), OpenRouter (whose free models are what `dblocks teacher models
//! --free` discovers), and a custom base URL. The provider sets the default
//! base URL and key variable; OpenRouter requests also carry the generic
//! `X-Title: unifur` attribution header and nothing that identifies the user.
//!
//! # Rate limits
//!
//! Every exchange is captured whole (status, final header block, body) so
//! [`crate::ratelimit::classify`] can read a 429, a `Retry-After`, the
//! `X-RateLimit-*` headers, an error object inside a 200, a 402 or an
//! overload. [`NimClient::complete`] waits out a short limit within its retry
//! allowance; [`NimClient::generate_traces`] goes through
//! [`crate::teacher::generate_resumable`], which pauses, rotates and resumes
//! across process restarts.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{anyhow, bail, ensure, Context};
use serde::{Deserialize, Serialize};

use crate::ratelimit::{HttpResponse, PausePolicy, PauseVerdict};
use sha2::{Digest, Sha256};

/// Default endpoint for NVIDIA's hosted NIM catalogue. A local NIM container
/// is pointed at by overriding `base_url` (or `NIM_BASE_URL` for
/// [`NimClient::from_env`]).
pub const DEFAULT_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";

/// Default teacher model. This is a *default*, not an assumption: every
/// request carries [`NimConfig::model`] and nothing in the module behaves
/// differently per model name. (The previous default,
/// `qwen/qwen3-coder-480b-a35b-instruct`, was end-of-lifed by the catalogue
/// on 2026-06-11 and now answers HTTP 410; `NIM_MODEL` overrides this default
/// without a rebuild — the catalogue retires models faster than crates
/// release.)
pub const DEFAULT_MODEL: &str = "moonshotai/kimi-k3";

/// Environment variable [`NimClient::from_env`] reads the API key from, unless
/// [`NimConfig::api_key_env`] says otherwise.
pub const DEFAULT_API_KEY_ENV: &str = "NVIDIA_API_KEY";

/// Environment variable [`NimClient::from_env`] reads an endpoint override
/// from (e.g. `http://localhost:8000/v1` for a local NIM container).
pub const BASE_URL_ENV: &str = "NIM_BASE_URL";

/// Environment variable [`NimClient::from_env`] reads a model override from.
/// The hosted catalogue retires models (see [`DEFAULT_MODEL`]), so the model
/// must be replaceable without editing the crate.
pub const MODEL_ENV: &str = "NIM_MODEL";

/// Where real trace-generation runs write by default. The path lives on an
/// external drive; tests never touch it and use unique `/tmp` names instead.
pub const DEFAULT_TRACE_DIR: &str = "/srv/m-sda/unifur/teacher-traces/";

/// OpenRouter's OpenAI-compatible endpoint.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Environment variable the OpenRouter key is read from by default.
pub const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";

/// Environment variable a [`Provider::Custom`] endpoint's key is read from by
/// default (override with [`NimConfig::api_key_env`]).
pub const CUSTOM_API_KEY_ENV: &str = "TEACHER_API_KEY";

/// The `X-Title` attribution header OpenRouter requests carry. Generic on
/// purpose: it names the tool, never the user, their project or their email.
pub const ATTRIBUTION_TITLE: &str = "unifur";

/// Which OpenAI-compatible service a [`NimClient`] talks to. The provider
/// decides the default base URL and key variable and whether the
/// attribution header is sent; the wire protocol is the same for all three.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// NVIDIA's hosted NIM catalogue (or a local NIM container).
    #[default]
    Nim,
    /// OpenRouter, including its free (`:free`, zero-priced) models.
    OpenRouter,
    /// Any other OpenAI-compatible endpoint; needs an explicit base URL.
    Custom,
}

impl Provider {
    /// Parse `nim`, `openrouter` or `custom` (case-insensitive).
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "nim" | "nvidia" => Ok(Self::Nim),
            "openrouter" => Ok(Self::OpenRouter),
            "custom" => Ok(Self::Custom),
            other => bail!("unknown teacher provider {other:?}: use nim, openrouter or custom"),
        }
    }

    /// Stable lower-case name, as recorded in traces.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nim => "nim",
            Self::OpenRouter => "openrouter",
            Self::Custom => "custom",
        }
    }

    /// The provider's default base URL; `None` for [`Provider::Custom`].
    pub fn default_base_url(self) -> Option<&'static str> {
        match self {
            Self::Nim => Some(DEFAULT_BASE_URL),
            Self::OpenRouter => Some(OPENROUTER_BASE_URL),
            Self::Custom => None,
        }
    }

    /// The environment variable the key is read from by default.
    pub fn default_api_key_env(self) -> &'static str {
        match self {
            Self::Nim => DEFAULT_API_KEY_ENV,
            Self::OpenRouter => OPENROUTER_API_KEY_ENV,
            Self::Custom => CUSTOM_API_KEY_ENV,
        }
    }

    /// Where a key for this provider comes from, for error messages.
    fn key_source(self) -> &'static str {
        match self {
            Self::Nim => "https://build.nvidia.com (free tier is enough for trace generation)",
            Self::OpenRouter => "https://openrouter.ai/keys (free models need a key too)",
            Self::Custom => "the operator of your endpoint",
        }
    }
}

/// Sampling temperature used for trace generation: low, because a teacher
/// trace should be the model's best answer, not a creative one.
const TRACE_TEMPERATURE: f32 = 0.2;

/// Configuration for [`NimClient`]. Every field carries `serde(default)`, so
/// a sidecar written by an older or newer build still parses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NimConfig {
    /// OpenAI-compatible base URL; the client POSTs to
    /// `{base_url}/chat/completions`.
    #[serde(default = "default_base_url")]
    pub base_url: String,
    /// Model name sent with every request.
    #[serde(default = "default_model")]
    pub model: String,
    /// Name of the environment variable holding the API key. The key itself
    /// is never stored in the config so it cannot end up serialized into a
    /// checkpoint sidecar.
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    /// Per-request timeout handed to curl's `--max-time`.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Number of *retries* after the first attempt, for transport failures
    /// and HTTP 5xx only. A 4xx is never retried: it fails the same way every
    /// time, and retrying it just burns the budget slower.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Base of the exponential backoff between retries, in milliseconds;
    /// retry `n` (1-based) waits `backoff_base_ms * 2^(n-1)`.
    #[serde(default = "default_backoff_base_ms")]
    pub backoff_base_ms: u64,
    /// Which service this is. Decides the attribution header and is recorded
    /// as provenance on every trace. Older sidecars have none and are NIM.
    #[serde(default)]
    pub provider: Provider,
    /// How rate limits are waited out (see [`crate::ratelimit::PausePolicy`]).
    #[serde(default)]
    pub pause: PausePolicy,
}

fn default_base_url() -> String {
    DEFAULT_BASE_URL.to_string()
}
fn default_model() -> String {
    DEFAULT_MODEL.to_string()
}
fn default_api_key_env() -> String {
    DEFAULT_API_KEY_ENV.to_string()
}
fn default_timeout_secs() -> u64 {
    120
}
fn default_max_retries() -> u32 {
    3
}
fn default_backoff_base_ms() -> u64 {
    500
}

impl Default for NimConfig {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            model: default_model(),
            api_key_env: default_api_key_env(),
            timeout_secs: default_timeout_secs(),
            max_retries: default_max_retries(),
            backoff_base_ms: default_backoff_base_ms(),
            provider: Provider::default(),
            pause: PausePolicy::default(),
        }
    }
}

impl NimConfig {
    /// The defaults for `provider`: its base URL (empty for
    /// [`Provider::Custom`], which must be given one) and key variable. The
    /// model is left at [`DEFAULT_MODEL`]; a pool or `--models` replaces it.
    pub fn for_provider(provider: Provider) -> Self {
        Self {
            base_url: provider
                .default_base_url()
                .map(str::to_string)
                .unwrap_or_default(),
            api_key_env: provider.default_api_key_env().to_string(),
            provider,
            ..Self::default()
        }
    }
}

/// One message in a chat-completions request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

impl ChatMessage {
    /// A system-role message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    /// A user-role message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
}

/// An OpenAI-compatible chat-completions request body.
#[derive(Clone, Debug, Serialize)]
pub struct ChatRequest {
    /// Model to query.
    pub model: String,
    /// Conversation so far.
    pub messages: Vec<ChatMessage>,
    /// Sampling temperature; omitted from the wire when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Hard cap on generated tokens; part of the cost log line.
    pub max_tokens: u32,
}

/// Token accounting returned by the endpoint. Fields default to zero because
/// some endpoints omit `usage` on stream-less responses; a zero here means
/// "not reported", which the stderr log line makes visible.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Tokens in the completion.
    #[serde(default)]
    pub completion_tokens: u32,
    /// Total tokens billed.
    #[serde(default)]
    pub total_tokens: u32,
}

/// A parsed chat-completions reply.
#[derive(Clone, Debug)]
pub struct ChatReply {
    /// `choices[0].message.content`.
    pub content: String,
    /// Token usage reported by the endpoint.
    pub usage: Usage,
    /// Model the endpoint says served the request (falls back to the
    /// requested model when the field is absent).
    pub model: String,
    /// `choices[0].finish_reason` as reported (`stop`, `length`, …).
    pub finish_reason: Option<String>,
}

/// One trace-generation prompt: a coding task in a named language.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TracePrompt {
    /// Programming language the answer should be in (`"rust"`, `"python"`, …).
    pub language: String,
    /// The task text sent to the teacher.
    pub prompt: String,
    /// Sampling seed recorded with the trace, for later reproduction.
    #[serde(default)]
    pub seed: u64,
    /// Token cap for this completion.
    #[serde(default = "default_trace_max_tokens")]
    pub max_tokens: u32,
}

fn default_trace_max_tokens() -> u32 {
    1024
}

impl TracePrompt {
    /// A stable id derived from the prompt's content (sha2-256, first 16 hex
    /// chars). Two runs over the same prompts produce the same ids, which is
    /// what makes [`NimClient::generate_traces`] resumable: an id already in
    /// the output file is never re-requested.
    pub fn id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.language.as_bytes());
        hasher.update(b"\n");
        hasher.update(self.seed.to_string().as_bytes());
        hasher.update(b"\n");
        hasher.update(self.prompt.as_bytes());
        let digest = hasher.finalize();
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// One line of the JSONL trace file: prompt, completion and provenance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRecord {
    /// [`TracePrompt::id`] of the prompt this answers.
    pub id: String,
    /// Language of the task.
    pub language: String,
    /// Task text (kept so the trace file is self-contained).
    pub prompt: String,
    /// Teacher's answer.
    pub completion: String,
    /// Model that produced the answer.
    pub model: String,
    /// Token usage of the request that produced it.
    pub usage: Usage,
    /// Seed recorded for reproduction.
    pub seed: u64,
    /// Where the line came from: `teacher` for NIM-generated traces,
    /// `identity` for the identity corpus ([`crate::langdata`]). Files
    /// written before this field existed are all teacher traces, so the
    /// serde default is `teacher`.
    #[serde(default = "default_trace_source")]
    pub source: String,
    /// The provider the request went to ([`Provider::as_str`]): provenance.
    /// Empty on records written before it was recorded, and on non-teacher
    /// records, where it is also left off the wire.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider: String,
    /// The exact model id the request named (e.g. an OpenRouter `:free` id).
    /// [`Self::model`] is what the endpoint *says* served it, which can drop
    /// the variant suffix; this is what was asked for. Empty on older records.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub requested_model: String,
}

pub(crate) fn default_trace_source() -> String {
    "teacher".to_string()
}

/// Hard cap on how many teacher requests may be made for one trace file. The
/// default is deliberately tiny (8): a caller that wants a real run
/// constructs a larger budget explicitly, so a copy-pasted call cannot
/// quietly spend money.
///
/// The cap is cumulative across resumes: the requests a file has already cost
/// are recorded in its journal (`<file>.state.json`, see
/// [`crate::teacher::RunJournal`]), so rerunning an interrupted command spends
/// only what is left of the cap rather than the whole cap again.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TraceBudget {
    /// Maximum number of HTTP requests (attempts, not successes, rate-limited
    /// ones included) the trace file may cost in total. Prompts past the cap
    /// are reported as skipped, not dropped silently.
    #[serde(default = "default_budget_max_requests")]
    pub max_requests: usize,
}

fn default_budget_max_requests() -> usize {
    8
}

impl Default for TraceBudget {
    fn default() -> Self {
        Self {
            max_requests: default_budget_max_requests(),
        }
    }
}

/// What a [`NimClient::generate_traces`] run did.
#[derive(Clone, Debug, Default)]
pub struct TraceReport {
    /// Path of the JSONL file that was written.
    pub out_path: PathBuf,
    /// Prompts completed by the teacher in this run.
    pub served: usize,
    /// Prompts skipped because their id was already in the output file.
    pub skipped_existing: usize,
    /// Prompts skipped because the budget was exhausted.
    pub skipped_budget: usize,
    /// Prompts whose request failed after retries. A failure consumes budget
    /// but never aborts the run; the error is logged to stderr.
    pub failed: usize,
    /// What the data-side gate removed from the teacher's own output, counted
    /// per reason. The teacher is the last point where a contaminated answer
    /// can be caught before it becomes training data: everything downstream
    /// reads this file, so a suppression or a smuggled key the teacher emitted
    /// would otherwise be read as a training signal.
    ///
    /// [`crate::cheat::ScanSummary::dropped`] counts every trace served in this run that the
    /// gate rejected. Traces already in the file from an earlier run are not
    /// re-gated here — they are gated by the loader that reads them, and
    /// re-running this command must not rewrite or renumber them.
    pub cheat_scan: crate::cheat::ScanSummary,
    /// The run's journal (`<out_path>.state.json`).
    pub journal_path: PathBuf,
    /// HTTP requests this run issued.
    pub requests_this_run: usize,
    /// HTTP requests the trace file has cost across every run, this one
    /// included; what [`TraceBudget::max_requests`] caps.
    pub requests_spent_total: usize,
    /// Rate-limit, credit and overload responses this run met.
    pub rate_limit_events: usize,
    /// Prompts not attempted because every model in the pool was paused past
    /// the idle bound (see [`Self::stopped`]); rerunning resumes them.
    pub skipped_paused: usize,
    /// Why the run stopped before the end of the prompts, when it did; names
    /// the earliest time a rerun can make progress.
    pub stopped: Option<String>,
    /// Traces written this run, per requested model id.
    pub served_by_model: std::collections::BTreeMap<String, usize>,
}

/// Why one HTTP attempt did not produce a parseable body. Internal: the
/// retry policy pattern-matches on it, and [`NimClient::complete`] turns the
/// terminal case into an `anyhow::Error` with the full context.
enum AttemptError {
    /// curl failed to spawn, exited non-zero, or produced unparseable
    /// output. Retryable.
    Transport(String),
    /// The server answered with a non-2xx status (or a 2xx whose body is a
    /// rate-limit error object). Retryable iff 5xx, or a rate limit whose
    /// pause fits the policy.
    Http(HttpResponse),
}

/// The teacher client. Construct with [`NimClient::new`] (explicit key, used
/// by tests with a dummy) or [`NimClient::from_env`] (reads the key from the
/// environment, the real-run path).
pub struct NimClient {
    config: NimConfig,
    api_key: String,
}

impl NimClient {
    /// Build a client from an explicit config and key. Validates the config
    /// and names the field at fault.
    pub fn new(config: NimConfig, api_key: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = api_key.into();
        ensure!(
            !config.base_url.trim().is_empty(),
            "NimConfig.base_url is empty: point it at an OpenAI-compatible endpoint such as {DEFAULT_BASE_URL}"
        );
        ensure!(
            !config.model.trim().is_empty(),
            "NimConfig.model is empty: name the teacher model, e.g. {DEFAULT_MODEL}"
        );
        ensure!(
            !api_key.trim().is_empty(),
            "the {} API key is empty: export ${} with a key from {}",
            config.provider.as_str(),
            config.api_key_env,
            config.provider.key_source()
        );
        ensure!(
            config.timeout_secs > 0,
            "NimConfig.timeout_secs must be positive"
        );
        ensure_curl()?;
        Ok(Self { config, api_key })
    }

    /// Build a client from the process environment. Reads the endpoint
    /// override from `NIM_BASE_URL` and the model override from `NIM_MODEL`
    /// when set, then the API key from [`NimConfig::api_key_env`] (default
    /// `NVIDIA_API_KEY`). A missing key is an error that says exactly how to
    /// fix it.
    pub fn from_env() -> anyhow::Result<Self> {
        let mut config = NimConfig::default();
        if let Ok(base) = std::env::var(BASE_URL_ENV) {
            if !base.trim().is_empty() {
                config.base_url = base;
            }
        }
        if let Ok(model) = std::env::var(MODEL_ENV) {
            if !model.trim().is_empty() {
                config.model = model;
            }
        }
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            anyhow!(
                "environment variable `{}` is not set: the NVIDIA NIM teacher client needs an API \
                 key. Create one at https://build.nvidia.com (free tier is enough for trace \
                 generation), then run `export {}=nvapi-...` and retry",
                config.api_key_env,
                config.api_key_env
            )
        })?;
        Self::new(config, api_key)
    }

    /// Build a client for an explicit config, reading the key from the
    /// variable [`NimConfig::api_key_env`] names. The key never appears in
    /// the error: only the variable's name and where to get a key.
    pub fn from_env_config(config: NimConfig) -> anyhow::Result<Self> {
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            anyhow!(
                "environment variable `{}` is not set: the {} teacher client needs an API key. \
                 Create one at {}, then `export {}=...` and retry",
                config.api_key_env,
                config.provider.as_str(),
                config.provider.key_source(),
                config.api_key_env
            )
        })?;
        Self::new(config, api_key)
    }

    /// The configuration this client was built with.
    pub fn config(&self) -> &NimConfig {
        &self.config
    }

    /// One chat completion. POSTs `{base_url}/chat/completions` and parses
    /// `choices[0].message.content` plus `usage`.
    ///
    /// Retries (bounded exponential backoff, [`NimConfig::max_retries`]
    /// times) happen on transport failures and HTTP 5xx. A rate limit (as
    /// [`crate::ratelimit::classify`] reads it, including an error object in
    /// a 200) whose pause fits [`NimConfig::pause`] is slept out and retried
    /// within the same retry count; a daily cap past `max_pause_daily_secs`,
    /// an account out of credits, or retries running out fail with the
    /// limit's kind and reset time in the message. Any other 4xx fails
    /// immediately; 401/403 errors name the key's environment variable.
    pub fn complete(&self, request: &ChatRequest) -> anyhow::Result<ChatReply> {
        let mut unlimited = usize::MAX;
        self.complete_within(request, &mut unlimited)
    }

    /// [`Self::complete`], charged against an HTTP-attempt allowance: every
    /// attempt -- retries included -- costs one, and no attempt (and no retry
    /// sleep) is made once it is spent. This is what lets
    /// [`TraceBudget::max_requests`] cap *requests*, as it documents, rather
    /// than prompts: counting prompts let each one spend `max_retries + 1`
    /// requests.
    fn complete_within(
        &self,
        request: &ChatRequest,
        allowance: &mut usize,
    ) -> anyhow::Result<ChatReply> {
        let body = serde_json::to_vec(request).context("serialize chat request")?;
        let mut retries_used = 0u32;
        loop {
            ensure!(
                *allowance > 0,
                "the request budget ran out after {} attempt(s) at this request",
                retries_used
            );
            *allowance -= 1;
            match self.attempt_once(&body) {
                Ok(raw) => {
                    let reply = parse_reply(&raw, request)?;
                    eprintln!(
                        "[nim] model={} max_tokens={} prompt_tokens={} completion_tokens={} total_tokens={}",
                        reply.model,
                        request.max_tokens,
                        reply.usage.prompt_tokens,
                        reply.usage.completion_tokens,
                        reply.usage.total_tokens
                    );
                    return Ok(reply);
                }
                Err(AttemptError::Transport(message)) => {
                    if retries_used >= self.config.max_retries || *allowance == 0 {
                        bail!(
                            "NIM request to {} failed after {} attempt(s){}: {message}",
                            self.config.base_url,
                            retries_used + 1,
                            budget_note(*allowance, retries_used, self.config.max_retries)
                        );
                    }
                    retries_used += 1;
                    self.sleep_before_retry(retries_used);
                }
                Err(AttemptError::Http(response)) => {
                    let status = response.status;
                    eprintln!(
                        "[nim] model={} request failed with HTTP {status}",
                        request.model
                    );
                    let now = system_now_unix_ms();
                    if let Some(signal) = crate::ratelimit::classify(&response, now) {
                        let plan = crate::ratelimit::plan_pause(
                            &signal,
                            now,
                            retries_used,
                            &self.config.pause,
                            rand::random::<f64>(),
                        );
                        let can_wait = plan.verdict == PauseVerdict::Wait
                            && retries_used < self.config.max_retries
                            && *allowance > 0;
                        if !can_wait {
                            let usable = if plan.verdict == PauseVerdict::Disable {
                                "not by waiting".to_string()
                            } else {
                                crate::ratelimit::format_unix_ms(plan.until_unix_ms)
                            };
                            bail!(
                                "{} request for model `{}` hit a {} limit after {} attempt(s) \
                                 (usable again: {usable}; {}): {}",
                                self.config.provider.as_str(),
                                request.model,
                                signal.kind.as_str(),
                                retries_used + 1,
                                plan.reason,
                                signal.evidence
                            );
                        }
                        eprintln!(
                            "[nim] model={} {} limit: pausing {} ms until {} ({})",
                            request.model,
                            signal.kind.as_str(),
                            plan.wait_ms,
                            crate::ratelimit::format_unix_ms(plan.until_unix_ms),
                            signal.evidence
                        );
                        std::thread::sleep(Duration::from_millis(plan.wait_ms));
                        retries_used += 1;
                        continue;
                    }
                    let retryable = status >= 500;
                    if !retryable || retries_used >= self.config.max_retries || *allowance == 0 {
                        if status == 401 || status == 403 {
                            bail!(
                                "NIM request rejected with HTTP {status} (authentication): the key in \
                                 ${} is missing, expired or lacks access to model `{}`. Server said: {}",
                                self.config.api_key_env,
                                request.model,
                                excerpt(&response.body)
                            );
                        }
                        bail!(
                            "NIM request failed with HTTP {status} after {} attempt(s){}: {}",
                            retries_used + 1,
                            if retryable {
                                budget_note(*allowance, retries_used, self.config.max_retries)
                            } else {
                                " (not retried: only a 5xx or a rate limit is)"
                            },
                            excerpt(&response.body)
                        );
                    }
                    retries_used += 1;
                    self.sleep_before_retry(retries_used);
                }
            }
        }
    }

    /// Generate teacher traces for `prompts`, appending to
    /// `out_dir/traces.jsonl` (created, along with `out_dir`, when absent).
    ///
    /// The file is one [`TraceRecord`] per line, appended and flushed to disk
    /// (`fsync`) as each trace is accepted, with a journal beside it
    /// (`traces.jsonl.state.json`): a crash or Ctrl-C loses at most the
    /// request in flight. Re-running with the same prompts is safe and cheap:
    /// ids already present are skipped without a request, a pause recorded
    /// by the interrupted run is honoured, and the request count carries
    /// over. At most `budget.max_requests` HTTP requests are issued for the
    /// file across all runs; the rest of the prompts are reported as
    /// `skipped_budget`. Rate limits pause and resume per
    /// [`NimConfig::pause`] (see [`crate::teacher::generate_resumable`]).
    pub fn generate_traces(
        &self,
        prompts: &[TracePrompt],
        out_dir: &Path,
        budget: TraceBudget,
    ) -> anyhow::Result<TraceReport> {
        self.generate_traces_gated(
            prompts,
            out_dir,
            budget,
            &crate::cheat::CheatGate::patterns_only(),
        )
    }

    /// [`Self::generate_traces`] with the data-side gate applied to the
    /// teacher's output before it is written.
    ///
    /// This is the upstream half of "the filter runs in every training
    /// method": [`crate::student::LanguageBatch::load`] gates what a trainer
    /// reads, and this gates what the teacher produced. A reply carrying a
    /// `todo!()`, a suppression or a key is refused here — counted in
    /// [`TraceReport::cheat_scan`], never written — rather than sitting in a
    /// traces file for a later loader to discover.
    ///
    /// The gate reads the completion alone: the prompt is a task statement
    /// that *says* the forbidden idioms, and scanning it would reject the
    /// prompt that asks for correct code most often.
    pub fn generate_traces_gated(
        &self,
        prompts: &[TracePrompt],
        out_dir: &Path,
        budget: TraceBudget,
        gate: &crate::cheat::CheatGate,
    ) -> anyhow::Result<TraceReport> {
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("create trace output directory {}", out_dir.display()))?;
        let pool = crate::teacher::PoolConfig {
            models: vec![self.config.model.clone()],
            pause: self.config.pause.clone(),
            forbid_mixing: false,
        };
        let settings = crate::teacher::GenerationSettings {
            budget,
            gate,
            pool: &pool,
            max_retries: self.config.max_retries,
            backoff_base_ms: self.config.backoff_base_ms,
        };
        crate::teacher::generate_resumable(
            self,
            &crate::teacher::SystemEnv,
            prompts,
            &out_dir.join("traces.jsonl"),
            &settings,
        )
    }

    /// One chat-completions exchange, exactly as the server answered it:
    /// any status, the final header block and the body. No retry, no
    /// classification; `Err` only when no HTTP response arrived (curl could
    /// not run, the connection failed, the output was not HTTP). This is the
    /// transport [`crate::teacher`] builds its pause/rotate loop on.
    pub fn send_chat(&self, request: &ChatRequest) -> Result<HttpResponse, String> {
        let body = serde_json::to_vec(request)
            .map_err(|err| format!("failed to serialize the chat request: {err}"))?;
        curl_exchange(
            &self.config,
            Some(&self.api_key),
            "POST",
            &self.completions_url(),
            Some(&body),
        )
    }

    /// The endpoint URL for one request.
    fn completions_url(&self) -> String {
        format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        )
    }

    fn sleep_before_retry(&self, retry_number: u32) {
        let factor = 2u64.saturating_pow(retry_number.saturating_sub(1).min(16));
        let wait = Duration::from_millis(self.config.backoff_base_ms.saturating_mul(factor));
        eprintln!("[nim] retry {retry_number} after {} ms", wait.as_millis());
        std::thread::sleep(wait);
    }

    /// Fire one request through curl. Returns the raw response body on a 2xx
    /// that is not a rate-limit error object; anything else is an
    /// [`AttemptError`] carrying the whole exchange.
    fn attempt_once(&self, body: &[u8]) -> Result<String, AttemptError> {
        let response = curl_exchange(
            &self.config,
            Some(&self.api_key),
            "POST",
            &self.completions_url(),
            Some(body),
        )
        .map_err(AttemptError::Transport)?;
        if response.is_success()
            && crate::ratelimit::classify(&response, system_now_unix_ms()).is_none()
        {
            Ok(response.body)
        } else {
            Err(AttemptError::Http(response))
        }
    }
}

/// The wall clock as Unix epoch milliseconds (0 if the clock is before 1970).
pub(crate) fn system_now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// The request a trace prompt becomes for `model`: the code-only system
/// prompt, the task, the low trace temperature and the prompt's token cap.
pub(crate) fn trace_request(model: &str, prompt: &TracePrompt) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage::system(format!(
                "You are an expert {} programmer. Answer the task with correct, idiomatic \
                 {} code and its tests. Answer with code only: no prose, no commentary — \
                 the answer is filtered as training data, and prose around the code gets \
                 it rejected.",
                prompt.language, prompt.language
            )),
            ChatMessage::user(prompt.prompt.clone()),
        ],
        temperature: Some(TRACE_TEMPERATURE),
        max_tokens: prompt.max_tokens,
    }
}

/// `GET {base_url}/{path}` (e.g. `models`) through curl, returning the whole
/// exchange. The key is optional because some listings (OpenRouter's
/// `/models`) are public; when given it travels on curl's stdin like every
/// other request, never on its command line.
pub fn http_get(
    config: &NimConfig,
    api_key: Option<&str>,
    path: &str,
) -> anyhow::Result<HttpResponse> {
    ensure!(
        !config.base_url.trim().is_empty(),
        "no base URL for provider {}: pass one explicitly",
        config.provider.as_str()
    );
    ensure_curl()?;
    let url = format!(
        "{}/{}",
        config.base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    );
    curl_exchange(config, api_key, "GET", &url, None)
        .map_err(|message| anyhow!("GET {url} failed: {message}"))
}

/// One HTTP exchange through curl.
///
/// The API key never appears in curl's argv, which every local user can
/// read (`ps`, `/proc/<pid>/cmdline`): the `Authorization` header is handed
/// to curl on stdin (`--header @-`). The request body, which is not secret,
/// goes through a private temporary file instead, since stdin is taken.
/// `--include` puts the response headers in front of the body, which is how
/// the rate-limit headers reach [`crate::ratelimit`].
fn curl_exchange(
    config: &NimConfig,
    api_key: Option<&str>,
    method: &str,
    url: &str,
    body: Option<&[u8]>,
) -> Result<HttpResponse, String> {
    let body_file = match body {
        Some(bytes) => Some(
            BodyFile::create(bytes)
                .map_err(|err| format!("failed to stage the request body for curl: {err}"))?,
        ),
        None => None,
    };
    let mut command = Command::new("curl");
    command
        .arg("--silent")
        .arg("--show-error")
        .arg("--location")
        .arg("--include")
        .arg("--max-time")
        .arg(config.timeout_secs.to_string())
        .arg("--request")
        .arg(method);
    if body_file.is_some() {
        command.arg("--header").arg("Content-Type: application/json");
    }
    if config.provider == Provider::OpenRouter {
        command
            .arg("--header")
            .arg(format!("X-Title: {ATTRIBUTION_TITLE}"));
    }
    if api_key.is_some() {
        command.arg("--header").arg("@-");
    }
    if let Some(file) = &body_file {
        command.arg("--data-binary").arg(file.curl_arg());
    }
    let mut child = command
        .arg(url)
        .stdin(if api_key.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("failed to spawn curl: {err}"))?;
    let piped = match (api_key, child.stdin.as_mut()) {
        (Some(key), Some(stdin)) => {
            stdin.write_all(format!("Authorization: Bearer {key}\n").as_bytes())
        }
        _ => Ok(()),
    };
    // Reap curl even when the pipe failed, or it lingers as a zombie;
    // `wait_with_output` closes stdin first, which is curl's cue that the
    // header list is complete.
    let output = child
        .wait_with_output()
        .map_err(|err| format!("failed to wait for curl: {err}"))?;
    piped.map_err(|err| format!("failed to pipe the auth header to curl: {err}"))?;
    drop(body_file);
    if !output.status.success() {
        return Err(format!(
            "curl exited with {}: {}",
            output.status,
            excerpt(&String::from_utf8_lossy(&output.stderr))
        ));
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|err| format!("curl output was not UTF-8: {err}"))?;
    let (status, header_block, body) = split_response(&text)?;
    Ok(HttpResponse::new(
        status,
        crate::ratelimit::parse_header_lines(header_block),
        body,
    ))
}

/// A request body staged in a private (`0600`) temporary file for curl's
/// `--data-binary @path`, removed again on drop.
struct BodyFile {
    path: PathBuf,
}

impl BodyFile {
    fn create(body: &[u8]) -> std::io::Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "dblocks-nim-body-{}-{n}.json",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        // Owned from here on, so a failed write still removes the file.
        let staged = Self { path };
        file.write_all(body)?;
        Ok(staged)
    }

    /// `@<path>`, curl's "read the data from this file" form.
    fn curl_arg(&self) -> std::ffi::OsString {
        let mut arg = std::ffi::OsString::from("@");
        arg.push(&self.path);
        arg
    }
}

impl Drop for BodyFile {
    fn drop(&mut self) {
        // Nothing can act on a failure here: the request already succeeded or
        // failed on its own terms, and a leftover file holds only the prompt.
        drop(std::fs::remove_file(&self.path));
    }
}

/// Why a failing request stopped retrying, for its error message: the empty
/// string when the retry limit was reached, a note when the request budget
/// ran out first.
fn budget_note(allowance: usize, retries_used: u32, max_retries: u32) -> &'static str {
    if allowance == 0 && retries_used < max_retries {
        " (the request budget ran out before the retries did)"
    } else {
        ""
    }
}

/// Fail fast when the one external tool this module needs is absent.
fn ensure_curl() -> anyhow::Result<()> {
    let found = Command::new("curl")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    ensure!(
        found,
        "the `curl` binary was not found on PATH: NimClient talks to the NIM endpoint through \
         curl (the crate has no HTTP stack of its own). Install it, e.g. `apt install curl`, \
         and retry"
    );
    Ok(())
}

/// Write `contents` to `path` via a temporary sibling plus rename, so a
/// reader never observes a partially written trace file. Shared with
/// [`crate::langdata`], whose pipeline outputs have the same crash-safety
/// requirement.
pub(crate) fn write_atomic(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp-{}-{n}", std::process::id()));
    // The temporary file is flushed to disk *before* the rename. Without the
    // `sync_all`, a crash soon after the rename can leave `path` naming an
    // empty or partial file -- and since the new contents include every
    // earlier trace, that loses the paid-for traces of every previous run, not
    // just this one's.
    let written = std::fs::File::create(&tmp).and_then(|mut file| {
        file.write_all(contents)?;
        file.sync_all()
    });
    if let Err(err) = written {
        // The write already failed; a leftover temporary is clutter, and the
        // write error is the one worth reporting.
        drop(std::fs::remove_file(&tmp));
        return Err(err).with_context(|| format!("write temporary trace file {}", tmp.display()));
    }
    std::fs::rename(&tmp, path).with_context(|| {
        format!(
            "rename temporary trace file {} over {}",
            tmp.display(),
            path.display()
        )
    })?;
    // And the rename itself is made durable by syncing the directory.
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    std::fs::File::open(dir)
        .and_then(|handle| handle.sync_all())
        .with_context(|| format!("flush directory {} to disk", dir.display()))?;
    Ok(())
}

/// Wire format of a chat-completions response. Unknown fields are ignored so
/// endpoint additions never break the parse.
#[derive(Deserialize)]
struct RawResponse {
    #[serde(default)]
    choices: Vec<RawChoice>,
    #[serde(default)]
    usage: Usage,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct RawChoice {
    message: RawMessage,
    /// Why generation stopped; `length` with a null `content` is how a
    /// reasoning model says the whole token budget went to thinking.
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct RawMessage {
    /// Null when the endpoint produced no visible answer (see above).
    content: Option<String>,
}

/// A chat-completions body read leniently: `content` may be absent (a
/// reasoning model that spent its whole token cap thinking answers with
/// `content: null`), which [`NimClient::complete`] reports as an error and
/// the teacher evaluation counts as a truncation.
#[derive(Clone, Debug, Default)]
pub struct ParsedChat {
    /// `choices[0].message.content`, when present.
    pub content: Option<String>,
    /// `choices[0].finish_reason`, when reported.
    pub finish_reason: Option<String>,
    /// Token usage (zeros when not reported).
    pub usage: Usage,
    /// The model the endpoint says served the request.
    pub model: Option<String>,
}

/// Parse a chat-completions body. Errors only when the body is not JSON or
/// carries no choice at all.
pub fn parse_chat_body(raw: &str) -> anyhow::Result<ParsedChat> {
    let response: RawResponse = serde_json::from_str(raw)
        .with_context(|| format!("NIM response was not valid JSON: {}", excerpt(raw)))?;
    let choice = response
        .choices
        .first()
        .ok_or_else(|| anyhow!("NIM response contained no choices: {}", excerpt(raw)))?;
    Ok(ParsedChat {
        content: choice.message.content.clone(),
        finish_reason: choice.finish_reason.clone(),
        usage: response.usage,
        model: response.model,
    })
}

fn parse_reply(raw: &str, request: &ChatRequest) -> anyhow::Result<ChatReply> {
    let parsed = parse_chat_body(raw)?;
    let content = parsed.content.ok_or_else(|| {
        anyhow!(
            "NIM response carried no content (finish_reason: {}): the model spent its {}-token \
             cap without producing an answer — reasoning models need a larger max_tokens",
            parsed.finish_reason.as_deref().unwrap_or("unreported"),
            request.max_tokens
        )
    })?;
    Ok(ChatReply {
        content,
        usage: parsed.usage,
        model: parsed.model.unwrap_or_else(|| request.model.clone()),
        finish_reason: parsed.finish_reason,
    })
}

/// Split a `curl --include` output into the final status code, the final
/// header block and the body. Redirects and `100 Continue` each leave their
/// own header block in the output, so blocks are stripped while the next one
/// still starts with `HTTP/`.
fn split_response(raw: &str) -> Result<(u16, &str, &str), String> {
    let mut rest = raw;
    loop {
        if !rest.starts_with("HTTP/") {
            return Err(format!(
                "curl output did not start with an HTTP status line: {}",
                excerpt(rest)
            ));
        }
        let Some((delim_at, delim_len)) = find_header_end(rest) else {
            return Err("curl output ended inside the HTTP headers".to_string());
        };
        let header_block = &rest[..delim_at];
        let status = parse_status(header_block)?;
        let body = &rest[delim_at + delim_len..];
        if body.starts_with("HTTP/") {
            rest = body;
            continue;
        }
        return Ok((status, header_block, body));
    }
}

/// Offset and length of the blank-line delimiter ending the first header
/// block in `raw`.
fn find_header_end(raw: &str) -> Option<(usize, usize)> {
    match (raw.find("\r\n\r\n"), raw.find("\n\n")) {
        (Some(crlf), Some(lf)) if lf < crlf => Some((lf, 2)),
        (Some(crlf), _) => Some((crlf, 4)),
        (None, Some(lf)) => Some((lf, 2)),
        (None, None) => None,
    }
}

/// The status code of a header block's first line (`HTTP/1.1 200 OK`).
fn parse_status(header_block: &str) -> Result<u16, String> {
    let line = header_block.lines().next().unwrap_or("");
    line.split_whitespace()
        .nth(1)
        .and_then(|token| token.parse::<u16>().ok())
        .ok_or_else(|| format!("could not parse an HTTP status from {line:?}"))
}

/// A single-line, length-bounded excerpt of a server response for error
/// messages: enough to diagnose, never a dump.
fn excerpt(text: &str) -> String {
    const MAX: usize = 240;
    let mut out: String = text.trim().chars().take(MAX).collect();
    if text.trim().chars().nth(MAX).is_some() {
        out.push('…');
    }
    out.replace(['\r', '\n'], " ")
}

// ---------------------------------------------------------------------------
// Trace consumption: the read side of the pipeline
//
// `generate_traces` writes; these functions are what a training pipeline
// reads. Everything here is keyless and file-local by construction.
// ---------------------------------------------------------------------------

/// Read every [`TraceRecord`] from a JSONL trace file.
///
/// Strict on purpose, because a training pipeline that silently drops lines
/// trains on less data than it thinks: a line that does not parse is an
/// error naming the file and the 1-based line number, and a duplicated
/// record id is an error naming the id (ids are content hashes, so a
/// duplicate means the same prompt was answered twice — a pipeline bug or a
/// hand-edited file, never something to paper over). Blank lines are
/// tolerated; a missing file is an error naming the path.
pub fn read_traces(path: &Path) -> anyhow::Result<Vec<TraceRecord>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read trace file {}", path.display()))?;
    let mut records = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: TraceRecord = serde_json::from_str(line).with_context(|| {
            format!(
                "trace file {} line {} is not a valid trace record",
                path.display(),
                index + 1
            )
        })?;
        ensure!(
            seen.insert(record.id.clone()),
            "trace file {} line {} repeats record id {}; ids are content hashes, so the same \
             prompt was recorded twice — merge or deduplicate the file before training on it",
            path.display(),
            index + 1,
            record.id
        );
        records.push(record);
    }
    Ok(records)
}

/// Aggregate statistics over a set of trace records: how many traces, in
/// which languages, at what token cost. Token sums come from the usage the
/// endpoint reported, so they are the bill the run actually ran up.
#[derive(Clone, Debug, Default)]
pub struct TraceStats {
    /// Total records.
    pub records: usize,
    /// Records per language, sorted by language name for stable reports.
    pub languages: std::collections::BTreeMap<String, usize>,
    /// Sum of reported prompt tokens.
    pub prompt_tokens: u64,
    /// Sum of reported completion tokens.
    pub completion_tokens: u64,
    /// Records whose completion is empty or whitespace. A teacher that
    /// answered nothing produced a trace worth dropping before training.
    pub empty_completions: usize,
}

/// Compute [`TraceStats`] over `records`.
pub fn trace_stats(records: &[TraceRecord]) -> TraceStats {
    let mut stats = TraceStats {
        records: records.len(),
        ..TraceStats::default()
    };
    for record in records {
        *stats.languages.entry(record.language.clone()).or_insert(0) += 1;
        stats.prompt_tokens += u64::from(record.usage.prompt_tokens);
        stats.completion_tokens += u64::from(record.usage.completion_tokens);
        if record.completion.trim().is_empty() {
            stats.empty_completions += 1;
        }
    }
    stats
}

/// Split trace records into `(train, validation)` folds, deterministically.
///
/// The fold of each record is decided by hashing its id, so the same record
/// lands in the same fold on every run, on every machine, regardless of the
/// order records appear in — a resumed or extended trace file never reshuffles
/// the validation set into the training set. `val_percent` is the percentage
/// of records (by hash bucket, so approximately) routed to validation;
/// `0` puts everything in train, `100` everything in validation.
pub fn split_traces(
    records: Vec<TraceRecord>,
    val_percent: u32,
) -> anyhow::Result<(Vec<TraceRecord>, Vec<TraceRecord>)> {
    ensure!(
        val_percent <= 100,
        "validation percentage must be in 0..=100, got {val_percent}"
    );
    let mut train = Vec::new();
    let mut validation = Vec::new();
    for record in records {
        let bucket = fold_bucket(&record.id);
        if bucket < val_percent {
            validation.push(record);
        } else {
            train.push(record);
        }
    }
    Ok((train, validation))
}

/// The fold bucket of a record id, in `0..100`: sha2-256 of the id, first
/// eight bytes as a big-endian u64, modulo 100. Hashing the id (rather than
/// parsing it) keeps the split well-defined for ids this module did not mint.
fn fold_bucket(id: &str) -> u32 {
    let digest = Sha256::digest(id.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_be_bytes(bytes) % 100) as u32
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
    clippy::wildcard_imports,
    clippy::exit
)]
mod tests {
    use super::*;

    use std::io::Read;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize as AtomicCounter, Ordering as AtomicOrdering};
    use std::sync::Arc;

    /// A minimal HTTP/1.1 server on a loopback port. Each connection gets the
    /// responder's `(status, body)` for that 1-based request number. Plain
    /// HTTP is fine here: curl needs TLS for the real endpoint, not for
    /// localhost.
    struct MockServer {
        addr: std::net::SocketAddr,
        stop: Arc<std::sync::atomic::AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
        requests: Arc<AtomicCounter>,
        /// The raw header block of every request received, in order.
        heads: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl MockServer {
        fn spawn(responder: impl Fn(usize) -> (u16, String) + Send + 'static) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
            listener
                .set_nonblocking(true)
                .expect("nonblocking listener");
            let addr = listener.local_addr().expect("local addr");
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let requests = Arc::new(AtomicCounter::new(0));
            let heads = Arc::new(std::sync::Mutex::new(Vec::new()));
            let thread_stop = Arc::clone(&stop);
            let thread_requests = Arc::clone(&requests);
            let thread_heads = Arc::clone(&heads);
            let handle = std::thread::spawn(move || {
                while !thread_stop.load(AtomicOrdering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let n = thread_requests.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                            serve_connection(stream, &responder, n, &thread_heads);
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                addr,
                stop,
                handle: Some(handle),
                requests,
                heads,
            }
        }

        fn heads(&self) -> Vec<String> {
            self.heads.lock().expect("heads lock").clone()
        }

        fn url(&self) -> String {
            format!("http://{}", self.addr)
        }

        fn request_count(&self) -> usize {
            self.requests.load(AtomicOrdering::SeqCst)
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.stop.store(true, AtomicOrdering::SeqCst);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn serve_connection(
        mut stream: std::net::TcpStream,
        responder: &dyn Fn(usize) -> (u16, String),
        n: usize,
        heads: &std::sync::Mutex<Vec<String>>,
    ) {
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") && head.len() < 64 * 1024 {
            match stream.read(&mut byte) {
                Ok(1) => head.push(byte[0]),
                _ => return,
            }
        }
        heads
            .lock()
            .expect("heads lock")
            .push(String::from_utf8_lossy(&head).into_owned());
        let head_text = String::from_utf8_lossy(&head).to_lowercase();
        let content_length = head_text
            .find("content-length:")
            .and_then(|pos| {
                head_text[pos + "content-length:".len()..]
                    .split(['\r', '\n'])
                    .next()
                    .and_then(|token| token.trim().parse::<usize>().ok())
            })
            .unwrap_or(0);
        let mut body = vec![0u8; content_length];
        if stream.read_exact(&mut body).is_err() {
            return;
        }
        let (status, body) = responder(n);
        let reason = match status {
            200 => "OK",
            401 => "Unauthorized",
            500 => "Internal Server Error",
            _ => "Status",
        };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
    }

    fn test_config(base_url: &str) -> NimConfig {
        NimConfig {
            base_url: base_url.to_string(),
            model: "test-teacher".to_string(),
            api_key_env: "NIM_TEST_UNUSED_KEY".to_string(),
            timeout_secs: 15,
            max_retries: 2,
            backoff_base_ms: 1,
            provider: Provider::Nim,
            pause: PausePolicy {
                jitter_max_ms: 0,
                ..PausePolicy::default()
            },
        }
    }

    fn test_client(base_url: &str) -> NimClient {
        NimClient::new(test_config(base_url), "test-key").expect("test client")
    }

    fn chat_request() -> ChatRequest {
        ChatRequest {
            model: "test-teacher".to_string(),
            messages: vec![ChatMessage::user("write fizzbuzz in rust")],
            temperature: Some(0.2),
            max_tokens: 64,
        }
    }

    fn ok_body(text: &str) -> String {
        format!(
            "{{\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":{text:?}}}}}],\
             \"usage\":{{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}},\
             \"model\":\"test-teacher\"}}"
        )
    }

    fn sample_prompts(n: usize) -> Vec<TracePrompt> {
        (0..n)
            .map(|i| TracePrompt {
                language: "rust".to_string(),
                prompt: format!("task number {i}: implement a stack"),
                seed: i as u64,
                max_tokens: 64,
            })
            .collect()
    }

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicCounter = AtomicCounter::new(0);
        let n = COUNTER.fetch_add(1, AtomicOrdering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("dblocks-nim-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create tmp dir");
        dir
    }

    /// Removes an env var for the scope of the guard and restores it on drop.
    struct EnvGuard {
        key: &'static str,
        saved: Option<String>,
    }

    impl EnvGuard {
        fn unset(key: &'static str) -> Self {
            let saved = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.saved {
                std::env::set_var(self.key, value);
            }
        }
    }

    #[test]
    fn round_trip_parses_valid_response() {
        let server = MockServer::spawn(|_| (200, ok_body("fn main() {}")));
        let client = test_client(&server.url());
        let reply = client.complete(&chat_request()).expect("completion");
        assert_eq!(reply.content, "fn main() {}");
        assert_eq!(reply.usage.prompt_tokens, 11);
        assert_eq!(reply.usage.completion_tokens, 7);
        assert_eq!(reply.usage.total_tokens, 18);
        assert_eq!(reply.model, "test-teacher");
        assert_eq!(server.request_count(), 1);
    }

    #[test]
    fn unauthorized_names_the_key_env_var() {
        let server = MockServer::spawn(|_| (401, "{\"error\":\"bad key\"}".to_string()));
        let client = test_client(&server.url());
        let err = client.complete(&chat_request()).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("401"), "status in error: {message}");
        assert!(
            message.contains("NIM_TEST_UNUSED_KEY"),
            "auth hint in error: {message}"
        );
        assert_eq!(server.request_count(), 1, "a 4xx must never be retried");
    }

    #[test]
    fn server_error_is_retried_then_succeeds() {
        let server = MockServer::spawn(|n| {
            if n == 1 {
                (500, "{\"error\":\"boom\"}".to_string())
            } else {
                (200, ok_body("retried ok"))
            }
        });
        let client = test_client(&server.url());
        let reply = client
            .complete(&chat_request())
            .expect("completion after retry");
        assert_eq!(reply.content, "retried ok");
        assert_eq!(server.request_count(), 2);
    }

    #[test]
    fn persistent_server_error_fails_after_bounded_retries() {
        let server = MockServer::spawn(|_| (500, "{\"error\":\"always\"}".to_string()));
        let client = test_client(&server.url());
        let err = client.complete(&chat_request()).unwrap_err();
        assert!(format!("{err:#}").contains("500"));
        assert_eq!(server.request_count(), 3, "1 attempt + 2 retries");
    }

    #[test]
    fn malformed_json_is_an_error() {
        let server = MockServer::spawn(|_| (200, "this is not json".to_string()));
        let client = test_client(&server.url());
        let err = client.complete(&chat_request()).unwrap_err();
        assert!(format!("{err:#}").contains("not valid JSON"));
    }

    #[test]
    fn empty_choices_is_an_error() {
        let server = MockServer::spawn(|_| (200, "{\"choices\":[]}".to_string()));
        let client = test_client(&server.url());
        let err = client.complete(&chat_request()).unwrap_err();
        assert!(format!("{err:#}").contains("no choices"));
    }

    #[test]
    fn null_content_names_the_token_cap() {
        // A reasoning model that spends its whole budget thinking answers
        // with `content: null` and `finish_reason: "length"`; the error must
        // say so, not fail as a JSON parse.
        let server = MockServer::spawn(|_| {
            (
                200,
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null},\
                 \"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":3,\
                 \"completion_tokens\":8,\"total_tokens\":11}}"
                    .to_string(),
            )
        });
        let client = test_client(&server.url());
        let err = client.complete(&chat_request()).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("no content"), "names the shape: {message}");
        assert!(message.contains("length"), "names the reason: {message}");
        assert!(
            message.contains("64"),
            "names the cap that was too small: {message}"
        );
    }

    #[test]
    fn budget_caps_requests_and_reports_skips() {
        let server = MockServer::spawn(|_| (200, ok_body("code")));
        let client = test_client(&server.url());
        let dir = unique_tmp_dir("budget");
        let prompts = sample_prompts(3);
        let budget = TraceBudget { max_requests: 2 };
        let report = client
            .generate_traces(&prompts, &dir, budget)
            .expect("generate traces");
        assert_eq!(report.served, 2);
        assert_eq!(report.skipped_budget, 1);
        assert_eq!(report.skipped_existing, 0);
        assert_eq!(report.failed, 0);
        assert_eq!(server.request_count(), 2);
        let text = std::fs::read_to_string(&report.out_path).expect("read traces");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        let record: TraceRecord = serde_json::from_str(lines[0]).expect("parse trace line");
        assert_eq!(record.id, prompts[0].id());
        assert_eq!(record.language, "rust");
        assert_eq!(record.completion, "code");
        assert_eq!(record.usage.total_tokens, 18);
    }

    /// The budget caps HTTP *requests*, retries included, as its doc says. A
    /// prompt-counting budget let every prompt spend `max_retries + 1`
    /// requests: here that was 6 requests against a budget of 2.
    #[test]
    fn budget_counts_retries_as_requests() {
        let server = MockServer::spawn(|_| (500, "{\"error\":\"down\"}".to_string()));
        let client = test_client(&server.url()); // max_retries: 2
        let dir = unique_tmp_dir("budget-retries");
        let report = client
            .generate_traces(&sample_prompts(3), &dir, TraceBudget { max_requests: 2 })
            .expect("generate traces");
        assert_eq!(server.request_count(), 2, "the budget is a request cap");
        assert_eq!(report.failed, 1, "the first prompt spent the budget retrying");
        assert_eq!(report.skipped_budget, 2);
        assert_eq!(report.served, 0);
    }

    /// The key reaches the server, and it reaches it through curl's stdin,
    /// not its command line: the request carries exactly one Authorization
    /// header, and the body arrives intact from the staged file.
    #[test]
    fn the_api_key_is_sent_as_a_header_and_the_body_intact() {
        let server = MockServer::spawn(|_| (200, ok_body("ok")));
        let client = test_client(&server.url());
        client.complete(&chat_request()).expect("completion");
        let heads = server.heads();
        assert_eq!(heads.len(), 1);
        let lower = heads[0].to_lowercase();
        assert_eq!(
            lower.matches("authorization: bearer test-key").count(),
            1,
            "header block: {}",
            heads[0]
        );
        let body_len = serde_json::to_vec(&chat_request()).unwrap().len();
        assert!(
            lower.contains(&format!("content-length: {body_len}")),
            "the body did not arrive whole: {}",
            heads[0]
        );
    }

    /// The staged body is private to the user, holds exactly the body, and is
    /// gone once the attempt that staged it is over.
    #[test]
    fn a_staged_body_is_private_exact_and_removed() {
        use std::os::unix::fs::PermissionsExt;
        let staged = BodyFile::create(b"{\"k\":1}").expect("stage");
        let path = staged.path.clone();
        assert_eq!(std::fs::read(&path).unwrap(), b"{\"k\":1}");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "the staged body is readable by others");
        assert_eq!(
            staged.curl_arg(),
            std::ffi::OsString::from(format!("@{}", path.display()))
        );
        drop(staged);
        assert!(!path.exists(), "the staged body outlived its attempt");
    }

    #[test]
    fn default_budget_is_tiny() {
        assert_eq!(TraceBudget::default().max_requests, 8);
    }

    #[test]
    fn resume_skips_prompts_already_in_the_file() {
        let server = MockServer::spawn(|_| (200, ok_body("code")));
        let client = test_client(&server.url());
        let dir = unique_tmp_dir("resume");
        let prompts = sample_prompts(2);
        let budget = TraceBudget { max_requests: 10 };

        let first = client
            .generate_traces(&prompts, &dir, budget)
            .expect("first run");
        assert_eq!(first.served, 2);
        assert_eq!(server.request_count(), 2);

        let second = client
            .generate_traces(&prompts, &dir, budget)
            .expect("second run");
        assert_eq!(second.served, 0);
        assert_eq!(second.skipped_existing, 2);
        assert_eq!(server.request_count(), 2, "no new requests on resume");

        // The file still holds exactly the original two records.
        let text = std::fs::read_to_string(second.out_path).expect("read traces");
        assert_eq!(text.lines().count(), 2);
    }

    #[test]
    fn resume_continues_after_a_budget_cutoff() {
        let server = MockServer::spawn(|_| (200, ok_body("code")));
        let client = test_client(&server.url());
        let dir = unique_tmp_dir("resume-raise");
        let prompts = sample_prompts(3);

        let first = client
            .generate_traces(&prompts, &dir, TraceBudget { max_requests: 1 })
            .expect("first run");
        assert_eq!((first.served, first.skipped_budget), (1, 2));

        let second = client
            .generate_traces(&prompts, &dir, TraceBudget { max_requests: 10 })
            .expect("second run with raised budget");
        assert_eq!((second.served, second.skipped_existing), (2, 1));
        assert_eq!(server.request_count(), 3);
        let text = std::fs::read_to_string(second.out_path).expect("read traces");
        assert_eq!(text.lines().count(), 3);
    }

    #[test]
    fn corrupt_trace_file_fails_loudly() {
        let server = MockServer::spawn(|_| (200, ok_body("code")));
        let client = test_client(&server.url());
        let dir = unique_tmp_dir("corrupt");
        std::fs::write(dir.join("traces.jsonl"), "not json at all\n").expect("seed corrupt file");
        let err = client
            .generate_traces(&sample_prompts(1), &dir, TraceBudget::default())
            .unwrap_err();
        assert!(format!("{err:#}").contains("line 1"));
    }

    #[test]
    fn prompt_ids_are_stable_and_content_sensitive() {
        let prompts = sample_prompts(2);
        assert_eq!(prompts[0].id(), prompts[0].id());
        assert_ne!(prompts[0].id(), prompts[1].id());
        assert_eq!(prompts[0].id().len(), 16);
    }

    #[test]
    fn config_round_trips_through_serde_with_defaults() {
        let config: NimConfig = serde_json::from_str("{}").expect("defaults parse");
        assert_eq!(config.base_url, DEFAULT_BASE_URL);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.api_key_env, DEFAULT_API_KEY_ENV);
        assert!(config.timeout_secs > 0);
    }

    #[test]
    fn from_env_without_key_fails_with_instructions() {
        let _guard = EnvGuard::unset(DEFAULT_API_KEY_ENV);
        // `.err()` avoids a `Debug` bound on `NimClient`: the client holds the
        // API key and deliberately does not implement `Debug`.
        let err = NimClient::from_env()
            .err()
            .expect("from_env without a key must fail");
        let message = format!("{err:#}");
        assert!(
            message.contains(DEFAULT_API_KEY_ENV),
            "names the var: {message}"
        );
        assert!(message.contains("export"), "says how to set it: {message}");
        assert!(
            message.contains("build.nvidia.com"),
            "says where to get a key: {message}"
        );
    }

    #[test]
    fn read_traces_round_trips_a_generated_file() {
        let server = MockServer::spawn(|_| (200, ok_body("code")));
        let client = test_client(&server.url());
        let dir = unique_tmp_dir("read");
        let prompts = sample_prompts(3);
        let report = client
            .generate_traces(&prompts, &dir, TraceBudget::default())
            .expect("generate traces");
        let records = read_traces(&report.out_path).expect("read traces back");
        assert_eq!(records.len(), 3);
        for (record, prompt) in records.iter().zip(prompts.iter()) {
            assert_eq!(record.id, prompt.id());
            assert_eq!(record.prompt, prompt.prompt);
            assert_eq!(record.completion, "code");
        }
    }

    #[test]
    fn read_traces_names_the_corrupt_line() {
        let dir = unique_tmp_dir("read-corrupt");
        let path = dir.join("traces.jsonl");
        let good = serde_json::to_string(&TraceRecord {
            id: "a".to_string(),
            language: "rust".to_string(),
            prompt: "p".to_string(),
            completion: "c".to_string(),
            model: "m".to_string(),
            usage: Usage::default(),
            seed: 0,
            source: "teacher".to_string(),
            provider: String::new(),
            requested_model: String::new(),
        })
        .expect("serialize record");
        std::fs::write(&path, format!("{good}\ngarbage\n")).expect("write file");
        let err = read_traces(&path).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("line 2"), "names the line: {message}");
        assert!(
            message.contains("traces.jsonl"),
            "names the file: {message}"
        );
    }

    #[test]
    fn read_traces_rejects_duplicate_ids() {
        let dir = unique_tmp_dir("read-dup");
        let path = dir.join("traces.jsonl");
        let record = TraceRecord {
            id: "dup".to_string(),
            language: "rust".to_string(),
            prompt: "p".to_string(),
            completion: "c".to_string(),
            model: "m".to_string(),
            usage: Usage::default(),
            seed: 0,
            source: "teacher".to_string(),
            provider: String::new(),
            requested_model: String::new(),
        };
        let line = serde_json::to_string(&record).expect("serialize record");
        std::fs::write(&path, format!("{line}\n{line}\n")).expect("write file");
        let err = read_traces(&path).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("dup"), "names the id: {message}");
        assert!(message.contains("line 2"), "names the line: {message}");
    }

    #[test]
    fn trace_stats_breaks_down_by_language_and_tokens() {
        let record =
            |language: &str, completion: &str, prompt_tokens: u32, completion_tokens: u32| {
                TraceRecord {
                    id: format!("id-{language}-{prompt_tokens}"),
                    language: language.to_string(),
                    prompt: "p".to_string(),
                    completion: completion.to_string(),
                    model: "m".to_string(),
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                    seed: 0,
                    source: "teacher".to_string(),
                    provider: String::new(),
                    requested_model: String::new(),
                }
            };
        let records = vec![
            record("rust", "fn main() {}", 10, 5),
            record("rust", "   ", 20, 0),
            record("python", "print(1)", 30, 7),
        ];
        let stats = trace_stats(&records);
        assert_eq!(stats.records, 3);
        assert_eq!(stats.languages.get("rust"), Some(&2));
        assert_eq!(stats.languages.get("python"), Some(&1));
        assert_eq!(stats.prompt_tokens, 60);
        assert_eq!(stats.completion_tokens, 12);
        assert_eq!(stats.empty_completions, 1);
    }

    /// The teacher side of the gate: a contaminated reply is refused before
    /// it is written, not left in the traces file for a loader to find.
    ///
    /// The mock answers the first request with a `todo!()` and the second with
    /// correct code, so the assertion is about *which* one reaches disk, not
    /// about the gate rejecting everything.
    #[test]
    fn a_contaminated_teacher_reply_is_never_written() {
        let server = MockServer::spawn(|n| {
            if n == 1 {
                (200, ok_body("fn add(a: i32, b: i32) -> i32 { todo!() }"))
            } else {
                (200, ok_body("fn add(a: i32, b: i32) -> i32 { a + b }"))
            }
        });
        let client = test_client(&server.url());
        let dir = unique_tmp_dir("gate");
        let prompts = sample_prompts(3);
        let report = client
            .generate_traces(&prompts, &dir, TraceBudget { max_requests: 3 })
            .expect("generate traces");

        // Three replies came back; one was refused, so two are on disk and
        // `served` counts only what is actually there.
        assert_eq!(server.request_count(), 3);
        assert_eq!(report.served, 2);
        assert_eq!(report.cheat_scan.scanned, 3);
        assert_eq!(report.cheat_scan.dropped, 1);
        assert_eq!(report.cheat_scan.kept, 2);
        assert_eq!(
            report
                .cheat_scan
                .dropped_by_class
                .get(crate::cheat::CheatClass::DataStub.as_str())
                .copied(),
            Some(1)
        );

        let records = read_traces(&report.out_path).expect("read traces");
        assert_eq!(records.len(), 2, "one reply must never reach the file");
        assert!(
            records.iter().all(|r| !r.completion.contains("todo!")),
            "a refused reply was written: {:?}",
            records.iter().map(|r| &r.completion).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_gate_that_rejects_every_reply_leaves_no_traces_and_says_so() {
        let server = MockServer::spawn(|_| (200, ok_body("fn f() { todo!() }")));
        let client = test_client(&server.url());
        let dir = unique_tmp_dir("gate-all");
        let report = client
            .generate_traces(&sample_prompts(2), &dir, TraceBudget { max_requests: 2 })
            .expect("generate traces");
        assert_eq!(report.served, 0);
        assert_eq!(report.cheat_scan.dropped, 2);
        let records = read_traces(&report.out_path).expect("read traces");
        assert!(
            records.is_empty(),
            "an all-rejected run left {} traces behind",
            records.len()
        );
    }

    #[test]
    fn split_is_deterministic_complete_and_exclusive() {
        let server = MockServer::spawn(|_| (200, ok_body("code")));
        let client = test_client(&server.url());
        let dir = unique_tmp_dir("split");
        let report = client
            .generate_traces(&sample_prompts(20), &dir, TraceBudget { max_requests: 20 })
            .expect("generate traces");
        let records = read_traces(&report.out_path).expect("read traces");

        let (train_a, val_a) = split_traces(records.clone(), 25).expect("split");
        let (train_b, val_b) = split_traces(records, 25).expect("split again");
        let ids =
            |records: &[TraceRecord]| records.iter().map(|r| r.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&train_a), ids(&train_b), "deterministic train fold");
        assert_eq!(ids(&val_a), ids(&val_b), "deterministic validation fold");
        assert_eq!(
            train_a.len() + val_a.len(),
            20,
            "no record lost or duplicated"
        );
        let train_ids: std::collections::HashSet<_> = ids(&train_a).into_iter().collect();
        assert!(
            val_a.iter().all(|r| !train_ids.contains(&r.id)),
            "folds are disjoint"
        );
        // 20 distinct hash buckets should land some records in each fold at 25%.
        assert!(!train_a.is_empty() && !val_a.is_empty());
    }

    #[test]
    fn split_edges_put_everything_in_one_fold() {
        let records: Vec<TraceRecord> = sample_prompts(5)
            .iter()
            .map(|p| TraceRecord {
                id: p.id(),
                language: p.language.clone(),
                prompt: p.prompt.clone(),
                completion: "c".to_string(),
                model: "m".to_string(),
                usage: Usage::default(),
                seed: p.seed,
                source: "teacher".to_string(),
                provider: String::new(),
                requested_model: String::new(),
            })
            .collect();
        let (train, val) = split_traces(records.clone(), 0).expect("split 0");
        assert_eq!((train.len(), val.len()), (5, 0));
        let (train, val) = split_traces(records.clone(), 100).expect("split 100");
        assert_eq!((train.len(), val.len()), (0, 5));
        assert!(split_traces(records, 101).is_err(), "101 is out of range");
    }

    /// A short 429 is waited out inside `complete` (the retry-after is 0 and
    /// the test policy has no jitter), and the retry succeeds.
    #[test]
    fn a_short_rate_limit_is_waited_out_and_retried() {
        let server = MockServer::spawn(|n| {
            if n == 1 {
                (
                    429,
                    "{\"error\":{\"code\":429,\"message\":\"Rate limit exceeded: free-models-per-min.\",\"retry_after\":0}}"
                        .to_string(),
                )
            } else {
                (200, ok_body("after the pause"))
            }
        });
        let client = test_client(&server.url());
        let reply = client.complete(&chat_request()).expect("completion after pause");
        assert_eq!(reply.content, "after the pause");
        assert_eq!(server.request_count(), 2);
    }

    /// A daily cap whose reset is hours away is not slept through by
    /// `complete`: it fails at once, naming the kind and the reset.
    #[test]
    fn a_far_daily_cap_fails_fast_with_its_reset() {
        let reset = system_now_unix_ms() + 6 * 3_600_000;
        let server = MockServer::spawn(move |_| {
            (
                200,
                format!(
                    "{{\"error\":{{\"code\":429,\"message\":\"Rate limit exceeded: free-models-per-day\",\"metadata\":{{\"headers\":{{\"X-RateLimit-Reset\":\"{reset}\"}}}}}}}}"
                ),
            )
        });
        let client = test_client(&server.url());
        let err = client.complete(&chat_request()).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("daily limit"), "{message}");
        assert!(
            message.contains(&crate::ratelimit::format_unix_ms(reset)),
            "names the reset: {message}"
        );
        assert_eq!(server.request_count(), 1);
    }

    /// OpenRouter requests carry the generic attribution header, and a NIM
    /// request does not; the exchange's headers reach the caller.
    #[test]
    fn openrouter_requests_carry_the_attribution_header() {
        let server = MockServer::spawn(|_| (200, ok_body("ok")));
        let client = NimClient::new(
            NimConfig {
                provider: Provider::OpenRouter,
                ..test_config(&server.url())
            },
            "test-key",
        )
        .expect("client");
        let response = client.send_chat(&chat_request()).expect("exchange");
        assert_eq!(response.status, 200);
        assert_eq!(response.header("content-type"), Some("application/json"));
        let nim = test_client(&server.url());
        nim.send_chat(&chat_request()).expect("exchange");
        let heads = server.heads();
        assert!(heads[0].to_lowercase().contains("x-title: unifur"), "{}", heads[0]);
        assert!(!heads[1].to_lowercase().contains("x-title"), "{}", heads[1]);
        assert_eq!(heads[0].to_lowercase().matches("authorization: bearer test-key").count(), 1);
    }

    #[test]
    fn providers_parse_and_default() {
        assert_eq!(Provider::parse("OpenRouter").unwrap(), Provider::OpenRouter);
        assert_eq!(Provider::parse("nim").unwrap(), Provider::Nim);
        assert!(Provider::parse("other").is_err());
        let open = NimConfig::for_provider(Provider::OpenRouter);
        assert_eq!(open.base_url, OPENROUTER_BASE_URL);
        assert_eq!(open.api_key_env, OPENROUTER_API_KEY_ENV);
        assert!(NimConfig::for_provider(Provider::Custom).base_url.is_empty());
        // An old sidecar without the new fields still parses, as NIM.
        let old: NimConfig = serde_json::from_str("{\"model\":\"m\"}").unwrap();
        assert_eq!(old.provider, Provider::Nim);
        // An old trace line without provenance still parses.
        let line = "{\"id\":\"a\",\"language\":\"rust\",\"prompt\":\"p\",\"completion\":\"c\",\"model\":\"m\",\"usage\":{},\"seed\":0}";
        let record: TraceRecord = serde_json::from_str(line).unwrap();
        assert!(record.provider.is_empty() && record.requested_model.is_empty());
        assert!(!serde_json::to_string(&record).unwrap().contains("provider"), "empty provenance stays off the wire");
    }

    /// Real-API smoke test. Ignored by default AND gated on the key: without
    /// `NVIDIA_API_KEY` it returns immediately even when explicitly run.
    /// Costs one tiny completion (`--include-ignored` to run):
    ///
    /// ```sh
    /// cargo test --lib nim -- --include-ignored real_api_smoke --nocapture
    /// ```
    #[test]
    #[ignore = "spends real tokens; run explicitly with NVIDIA_API_KEY set"]
    fn real_api_smoke() {
        if std::env::var(DEFAULT_API_KEY_ENV).is_err() {
            eprintln!("[nim] {DEFAULT_API_KEY_ENV} not set; skipping real-API smoke test");
            return;
        }
        let client = NimClient::from_env().expect("client from env");
        let request = ChatRequest {
            model: client.config().model.clone(),
            messages: vec![ChatMessage::user("Reply with exactly the word: ok")],
            temperature: Some(0.0),
            // Room for a reasoning model's hidden thinking plus the word.
            max_tokens: 256,
        };
        let reply = client.complete(&request).expect("real completion");
        assert!(!reply.content.trim().is_empty());
        eprintln!(
            "[nim] smoke usage: prompt={} completion={} total={}",
            reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.total_tokens
        );
    }
}
