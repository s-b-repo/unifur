//! Teacher runs that survive rate limits: provider transport, free-model
//! discovery, a rotating model pool, and a durable, resumable trace writer.
//!
//! # The loop
//!
//! [`Requester::request`] is the one place a teacher request is made. It
//! picks a model from the [`ModelPool`], sends through a [`ChatBackend`]
//! (curl via [`NimClient`] in production, a canned script in tests), reads
//! the exchange with [`interpret`], and on a rate limit asks
//! [`crate::ratelimit::plan_pause`] what to do:
//!
//! - **Wait** — a per-minute limit, an overload, a daily cap that resets
//!   within `max_pause_daily_secs`: the model is paused until its reset plus
//!   jitter, and the run sleeps for it (each pause is logged to stderr with
//!   the reason and the resume time).
//! - **Rotate** — a daily cap that resets later, a limit past `max_pause_secs`,
//!   a model that failed `max_consecutive_failures` times in a row: the
//!   model is benched until its reset and the next model in the pool's
//!   priority order is used. A higher-priority model is returned to as soon as
//!   its reset passes.
//! - **Disable** — HTTP 402 (no credits) or a model the endpoint does not
//!   have (404/410): not used again in this run.
//!
//! When every model is paused, the run sleeps until the earliest reset if it
//! is within `max_idle_secs`, and otherwise stops cleanly with its journal
//! saved. OpenRouter's free-tier caps (`free-models-per-min`,
//! `free-models-per-day`) are per *account*, not per model, so such a signal
//! pauses every `:free` model in the pool at once instead of burning requests
//! rotating through models that would all answer 429.
//!
//! # Durability
//!
//! [`generate_resumable`] appends each accepted trace to the JSONL file and
//! `fsync`s it before moving on, and keeps a journal beside it
//! (`<out>.state.json`, [`RunJournal`], written atomically) with the
//! completed prompt ids, the requests spent (reserved *before* each request
//! is sent, so a crash can only over-count) and each model's pause. Rerunning
//! the same command skips completed ids, honours a pause whose reset has not
//! passed, and enforces [`TraceBudget::max_requests`] against the cumulative
//! count. A Ctrl-C or crash loses at most the request in flight; a torn final
//! line from an interrupted write is dropped on the next open.
//!
//! # Mixing
//!
//! Every trace records the provider and the exact requested model id, so a
//! pool never mixes models silently. [`PoolConfig::forbid_mixing`] goes
//! further: the file is locked to the first model that serves it (or the one
//! its existing traces came from), and the pool waits for that model rather
//! than rotating.
//!
//! # Free models
//!
//! [`fetch_models`] lists `{base}/models`; [`free_rule`] marks a model free
//! when its prompt and completion prices are both zero ([`FreeRule::ZeroPricing`])
//! and/or its id ends in `:free` ([`FreeRule::FreeSuffix`]), and reports which
//! rule matched. Nothing here names a model id: discovery returns whatever the
//! provider lists today. Free-tier providers on OpenRouter may log or train on
//! prompts, depending on the account's privacy settings — never send private
//! data through a free model.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, ensure, Context};
use serde::{Deserialize, Serialize};

use crate::cheat::{CheatGate, SnippetLanguage};
use crate::nim::{
    ChatRequest, NimClient, NimConfig, Provider, TraceBudget, TracePrompt, TraceRecord,
    TraceReport, Usage,
};
use crate::ratelimit::{
    self, HttpResponse, PausePlan, PausePolicy, PauseVerdict, QuotaSnapshot, RateLimitKind,
    RateLimitSignal,
};
use crate::student::CodeLanguage;

// ---------------------------------------------------------------------------
// Transport and environment
// ---------------------------------------------------------------------------

/// Where teacher requests go. [`NimClient`] is the production
/// implementation; tests script one.
pub trait ChatBackend {
    /// The provider requests go to, recorded on every trace.
    fn provider(&self) -> Provider;
    /// The name (never the value) of the variable holding the key, for
    /// authentication error messages.
    fn key_env(&self) -> &str;
    /// One exchange, whatever its status. `Err` only when no HTTP response
    /// arrived at all.
    fn send(&self, request: &ChatRequest) -> Result<HttpResponse, String>;
}

impl ChatBackend for NimClient {
    fn provider(&self) -> Provider {
        self.config().provider
    }

    fn key_env(&self) -> &str {
        &self.config().api_key_env
    }

    fn send(&self, request: &ChatRequest) -> Result<HttpResponse, String> {
        self.send_chat(request)
    }
}

/// Clock, sleep and jitter source. Real runs use [`SystemEnv`]; tests use a
/// virtual clock so pauses of hours cost nothing.
pub trait RunEnv {
    /// Now, as Unix epoch milliseconds.
    fn now_unix_ms(&self) -> u64;
    /// Block for `wait`.
    fn sleep(&self, wait: Duration);
    /// A draw in `[0, 1)` for pause jitter.
    fn jitter_unit(&self) -> f64;
}

/// The wall clock, `std::thread::sleep` and `rand`.
pub struct SystemEnv;

impl RunEnv for SystemEnv {
    fn now_unix_ms(&self) -> u64 {
        crate::nim::system_now_unix_ms()
    }

    fn sleep(&self, wait: Duration) {
        std::thread::sleep(wait);
    }

    fn jitter_unit(&self) -> f64 {
        rand::random::<f64>()
    }
}

// ---------------------------------------------------------------------------
// Reading one exchange
// ---------------------------------------------------------------------------

/// A successful chat-completions reply.
#[derive(Clone, Debug)]
pub struct TeacherReply {
    /// The answer; `None` when the model produced no visible content (a
    /// reasoning model that spent its token cap thinking).
    pub content: Option<String>,
    /// `finish_reason` as reported; `length` means truncated.
    pub finish_reason: Option<String>,
    /// Token usage.
    pub usage: Usage,
    /// The model the endpoint says served it (the requested id when absent).
    pub served_model: String,
    /// The quota the response advertised, when it did.
    pub quota: Option<QuotaSnapshot>,
}

/// What one exchange means for the loop.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// An answer.
    Reply(TeacherReply),
    /// A rate limit, credit or overload signal.
    Limited(RateLimitSignal),
    /// No response, or a 5xx that is not an overload: retry with backoff.
    Retryable(String),
    /// The endpoint does not serve this model (404/410, "not a valid model").
    ModelUnavailable(String),
    /// Authentication failed: no request will succeed, stop the run.
    Fatal(String),
    /// This request failed (other 4xx, an unparseable 2xx); move on.
    Failed(String),
}

/// Body phrases that mean "this model is not served here".
const MODEL_GONE_MARKERS: &[&str] = &[
    "not a valid model",
    "model not found",
    "no endpoints found",
    "does not exist",
    "unknown model",
    "no such model",
    "end of life",
    "end-of-life",
];

/// Classify one exchange. Rate limits are checked first, so an error object
/// inside a 200 is a [`Outcome::Limited`], never a reply.
pub fn interpret(
    result: Result<HttpResponse, String>,
    requested_model: &str,
    key_env: &str,
    now_unix_ms: u64,
) -> Outcome {
    let response = match result {
        Ok(response) => response,
        Err(message) => return Outcome::Retryable(format!("transport: {message}")),
    };
    if let Some(signal) = ratelimit::classify(&response, now_unix_ms) {
        return Outcome::Limited(signal);
    }
    let status = response.status;
    let said = short(&response.body, 240);
    if response.is_success() {
        return match crate::nim::parse_chat_body(&response.body) {
            Ok(parsed) => Outcome::Reply(TeacherReply {
                content: parsed.content,
                finish_reason: parsed.finish_reason,
                usage: parsed.usage,
                served_model: parsed
                    .model
                    .filter(|m| !m.trim().is_empty())
                    .unwrap_or_else(|| requested_model.to_string()),
                quota: ratelimit::quota_from_headers(&response.headers, now_unix_ms),
            }),
            Err(err) => Outcome::Failed(format!("{err:#}")),
        };
    }
    let lower = response.body.to_lowercase();
    match status {
        403 if lower.contains("flagged") || lower.contains("moderation") => {
            Outcome::Failed(format!("HTTP 403 (input flagged by moderation): {said}"))
        }
        401 | 403 => Outcome::Fatal(format!(
            "HTTP {status} (authentication): the key in ${key_env} is missing, expired or lacks \
             access to model `{requested_model}`. Server said: {said}"
        )),
        404 | 410 => Outcome::ModelUnavailable(format!(
            "HTTP {status} for model `{requested_model}`: {said}"
        )),
        400 | 422 if MODEL_GONE_MARKERS.iter().any(|m| lower.contains(m)) => {
            Outcome::ModelUnavailable(format!(
                "HTTP {status} for model `{requested_model}`: {said}"
            ))
        }
        s if s >= 500 => Outcome::Retryable(format!("HTTP {status}: {said}")),
        _ => Outcome::Failed(format!("HTTP {status} (not retried): {said}")),
    }
}

/// A single-line, length-bounded excerpt.
fn short(text: &str, max: usize) -> String {
    let flat = text.trim().replace(['\r', '\n'], " ");
    let mut out: String = flat.chars().take(max).collect();
    if flat.chars().nth(max).is_some() {
        out.push('…');
    }
    out
}

/// True when a signal is OpenRouter's account-wide free-tier limit, which
/// every `:free` model shares.
fn free_tier_account_limit(signal: &RateLimitSignal) -> bool {
    signal.evidence.to_lowercase().contains("free-models-per-")
}

/// Milliseconds as a short human duration for log lines.
fn human_ms(ms: u64) -> String {
    let secs = ms / 1_000;
    if secs >= 3_600 {
        format!("{}h{:02}m", secs / 3_600, (secs % 3_600) / 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}.{:03}s", secs, ms % 1_000)
    }
}

fn duration_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

// ---------------------------------------------------------------------------
// The model pool
// ---------------------------------------------------------------------------

/// One model in a [`ModelPool`] and its pause state.
#[derive(Clone, Debug)]
pub struct ModelSlot {
    /// Exact model id.
    pub id: String,
    /// Unusable before this time (Unix ms); 0 when not paused.
    pub paused_until_unix_ms: u64,
    /// The limit behind the current pause.
    pub pause_kind: Option<RateLimitKind>,
    /// Whether the current pause is waited for or rotated around.
    pub verdict: Option<PauseVerdict>,
    /// Why it is paused.
    pub reason: String,
    /// Why it is out of the run, when it is.
    pub disabled: Option<String>,
    /// Failures and overloads in a row since its last success.
    pub consecutive_failures: u32,
}

/// What the pool says to do next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pick {
    /// Send to this slot now.
    Use(usize),
    /// Sleep until this slot is usable, then pick again.
    Sleep {
        /// The slot being waited for.
        index: usize,
        /// When it is usable.
        until_unix_ms: u64,
    },
    /// Nothing is usable within the idle bound.
    Exhausted {
        /// Why, with the earliest resume time when there is one.
        reason: String,
        /// When a rerun could make progress.
        earliest_resume_unix_ms: Option<u64>,
    },
}

/// An ordered list of models: earlier entries are preferred, later ones are
/// used while earlier ones are benched.
#[derive(Clone, Debug)]
pub struct ModelPool {
    slots: Vec<ModelSlot>,
    current: usize,
    locked: Option<usize>,
}

impl ModelPool {
    /// A pool over `models` in priority order. Empty, blank or duplicated
    /// ids are errors.
    pub fn new(models: &[String]) -> anyhow::Result<Self> {
        ensure!(
            !models.is_empty(),
            "the teacher model pool is empty: name at least one model"
        );
        let mut seen = HashSet::new();
        let mut slots = Vec::with_capacity(models.len());
        for id in models {
            let id = id.trim();
            ensure!(!id.is_empty(), "the teacher model pool holds a blank model id");
            ensure!(
                seen.insert(id.to_string()),
                "model `{id}` appears twice in the teacher model pool"
            );
            slots.push(ModelSlot {
                id: id.to_string(),
                paused_until_unix_ms: 0,
                pause_kind: None,
                verdict: None,
                reason: String::new(),
                disabled: None,
                consecutive_failures: 0,
            });
        }
        Ok(Self {
            slots,
            current: 0,
            locked: None,
        })
    }

    /// The slots, in priority order.
    pub fn slots(&self) -> &[ModelSlot] {
        &self.slots
    }

    /// The slot requests currently go to.
    pub fn current(&self) -> usize {
        self.current
    }

    /// The id of slot `index` (empty when out of range).
    pub fn id(&self, index: usize) -> &str {
        self.slots.get(index).map_or("", |slot| slot.id.as_str())
    }

    /// The slot of model `id`.
    pub fn index_of(&self, id: &str) -> Option<usize> {
        self.slots.iter().position(|slot| slot.id == id)
    }

    /// Restrict the pool to slot `index` (forbid-mixing).
    pub fn lock(&mut self, index: usize) {
        if index < self.slots.len() {
            self.locked = Some(index);
            self.current = index;
        }
    }

    /// The slot the pool is locked to, when it is.
    pub fn locked(&self) -> Option<usize> {
        self.locked
    }

    fn usable(&self, index: usize) -> bool {
        self.slots
            .get(index)
            .is_some_and(|slot| slot.disabled.is_none())
            && self.locked.is_none_or(|locked| locked == index)
    }

    fn available(&self, index: usize, now_unix_ms: u64) -> bool {
        self.usable(index)
            && self
                .slots
                .get(index)
                .is_some_and(|slot| slot.paused_until_unix_ms <= now_unix_ms)
    }

    /// Decide what to do next.
    ///
    /// 1. The highest-priority model at or above the current one that is
    ///    usable now — which is how a benched model is returned to once its
    ///    reset passes.
    /// 2. If the current model is paused with a *wait* verdict, sleep for it.
    /// 3. Otherwise rotate to the first usable model in priority order.
    /// 4. Nothing usable now: sleep until the earliest reset if that is within
    ///    `max_idle_ms`, else stop.
    pub fn pick(&self, now_unix_ms: u64, max_idle_ms: u64) -> Pick {
        let last = self.current.min(self.slots.len().saturating_sub(1));
        if let Some(index) = (0..=last).find(|&i| self.available(i, now_unix_ms)) {
            return Pick::Use(index);
        }
        if let Some(slot) = self.slots.get(self.current) {
            if self.usable(self.current)
                && slot.verdict == Some(PauseVerdict::Wait)
                && slot.paused_until_unix_ms > now_unix_ms
            {
                return Pick::Sleep {
                    index: self.current,
                    until_unix_ms: slot.paused_until_unix_ms,
                };
            }
        }
        if let Some(index) = (0..self.slots.len()).find(|&i| self.available(i, now_unix_ms)) {
            return Pick::Use(index);
        }
        let earliest = (0..self.slots.len())
            .filter(|&i| self.usable(i))
            .filter_map(|i| self.slots.get(i).map(|s| (i, s.paused_until_unix_ms)))
            .min_by_key(|(_, until)| *until);
        match earliest {
            Some((index, until)) if until.saturating_sub(now_unix_ms) <= max_idle_ms => {
                Pick::Sleep {
                    index,
                    until_unix_ms: until,
                }
            }
            Some((index, until)) => Pick::Exhausted {
                reason: format!(
                    "every usable model is paused past max_idle_secs; the earliest, `{}`, is \
                     usable again at {} ({})",
                    self.id(index),
                    ratelimit::format_unix_ms(until),
                    self.slots.get(index).map_or("", |s| s.reason.as_str())
                ),
                earliest_resume_unix_ms: Some(until),
            },
            None => Pick::Exhausted {
                reason: format!(
                    "no model in the pool is usable: {}",
                    self.slots
                        .iter()
                        .map(|s| format!(
                            "`{}`: {}",
                            s.id,
                            s.disabled.as_deref().unwrap_or("excluded by forbid-mixing")
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
                earliest_resume_unix_ms: None,
            },
        }
    }

    /// Make `index` current. Returns the previous slot when it changed.
    pub fn switch_to(&mut self, index: usize) -> Option<usize> {
        if index == self.current || index >= self.slots.len() {
            return None;
        }
        let previous = self.current;
        self.current = index;
        Some(previous)
    }

    /// Apply a pause plan to slot `index`.
    pub fn apply(&mut self, index: usize, plan: &PausePlan) {
        if plan.verdict == PauseVerdict::Disable {
            self.disable(index, plan.reason.clone());
            return;
        }
        if let Some(slot) = self.slots.get_mut(index) {
            slot.paused_until_unix_ms = plan.until_unix_ms;
            slot.pause_kind = Some(plan.kind);
            slot.verdict = Some(plan.verdict);
            slot.reason = plan.reason.clone();
            if plan.verdict == PauseVerdict::Rotate {
                slot.consecutive_failures = 0;
            }
        }
    }

    /// Restore a pause recorded by an earlier run.
    pub fn restore(
        &mut self,
        index: usize,
        until_unix_ms: u64,
        kind: Option<RateLimitKind>,
        verdict: PauseVerdict,
        reason: String,
    ) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.paused_until_unix_ms = until_unix_ms;
            slot.pause_kind = kind;
            slot.verdict = Some(verdict);
            slot.reason = reason;
        }
    }

    /// Hold slot `index` until `until_unix_ms` to spread a low quota (a
    /// soft, waited-for pause; not a rate-limit event).
    pub fn pace(&mut self, index: usize, until_unix_ms: u64) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.paused_until_unix_ms = slot.paused_until_unix_ms.max(until_unix_ms);
            slot.verdict = Some(PauseVerdict::Wait);
            slot.reason = "pacing a low remaining quota".to_string();
        }
    }

    /// Take slot `index` out of the run.
    pub fn disable(&mut self, index: usize, reason: String) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.disabled = Some(reason);
        }
    }

    /// A success: the failure streak ends.
    pub fn on_success(&mut self, index: usize) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.consecutive_failures = 0;
        }
    }

    /// Count a failure or overload; returns the new streak length.
    pub fn note_failure(&mut self, index: usize) -> u32 {
        match self.slots.get_mut(index) {
            Some(slot) => {
                slot.consecutive_failures = slot.consecutive_failures.saturating_add(1);
                slot.consecutive_failures
            }
            None => 0,
        }
    }

    /// The current failure streak of slot `index`.
    pub fn consecutive(&self, index: usize) -> u32 {
        self.slots
            .get(index)
            .map_or(0, |slot| slot.consecutive_failures)
    }
}

// ---------------------------------------------------------------------------
// The request loop
// ---------------------------------------------------------------------------

/// How one prompt ended.
#[derive(Clone, Debug)]
pub enum RequestEnd {
    /// The teacher answered.
    Reply {
        /// The reply.
        reply: TeacherReply,
        /// The model id that was requested.
        model: String,
        /// Time from send to response, ms.
        latency_ms: u64,
    },
    /// The prompt failed (retries spent, a 4xx, too many pauses).
    Failed(String),
    /// No attempt was made: the request budget is spent.
    OutOfBudget,
    /// No model is usable within the idle bound; the run should stop.
    Stopped(String),
}

/// Hooks the loop calls so a caller can persist what happened. Every method
/// has a no-op default.
pub trait RequestObserver {
    /// Just before a request is sent to `model` (the place to reserve it in
    /// a journal, so a crash mid-request still counts it).
    fn before_send(&mut self, _model: &str, _now_unix_ms: u64) -> anyhow::Result<()> {
        Ok(())
    }

    /// A pause was applied to `affected` (a signal, or a failure streak when
    /// `signal` is `None`).
    fn on_pause(
        &mut self,
        _signal: Option<&RateLimitSignal>,
        _plan: &PausePlan,
        _affected: &[String],
        _now_unix_ms: u64,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// `model` was taken out of the run.
    fn on_disabled(&mut self, _model: &str, _reason: &str, _now_unix_ms: u64) -> anyhow::Result<()> {
        Ok(())
    }
}

/// An observer that records nothing.
pub struct NoObserver;

impl RequestObserver for NoObserver {}

/// The pause / rotate / retry loop around a [`ChatBackend`].
pub struct Requester<'a> {
    backend: &'a dyn ChatBackend,
    env: &'a dyn RunEnv,
    policy: PausePolicy,
    pool: ModelPool,
    max_retries: u32,
    backoff_base_ms: u64,
    allowance: usize,
    last_request_unix_ms: Option<u64>,
    requests: usize,
    rate_limit_events: usize,
    events_by_kind: BTreeMap<RateLimitKind, usize>,
}

impl<'a> Requester<'a> {
    /// A loop over `pool` with at most `allowance` requests.
    pub fn new(
        backend: &'a dyn ChatBackend,
        env: &'a dyn RunEnv,
        pool: ModelPool,
        policy: PausePolicy,
        max_retries: u32,
        backoff_base_ms: u64,
        allowance: usize,
    ) -> Self {
        Self {
            backend,
            env,
            policy,
            pool,
            max_retries,
            backoff_base_ms,
            allowance,
            last_request_unix_ms: None,
            requests: 0,
            rate_limit_events: 0,
            events_by_kind: BTreeMap::new(),
        }
    }

    /// The pool, with its pause state.
    pub fn pool(&self) -> &ModelPool {
        &self.pool
    }

    /// The pool, mutably (to lock it).
    pub fn pool_mut(&mut self) -> &mut ModelPool {
        &mut self.pool
    }

    /// Requests left.
    pub fn allowance(&self) -> usize {
        self.allowance
    }

    /// Requests sent so far.
    pub fn requests(&self) -> usize {
        self.requests
    }

    /// Rate-limit, credit and overload responses met so far.
    pub fn rate_limit_events(&self) -> usize {
        self.rate_limit_events
    }

    /// Those events by kind.
    pub fn events_by_kind(&self) -> &BTreeMap<RateLimitKind, usize> {
        &self.events_by_kind
    }

    /// Serve one prompt: `build` turns a model id into the request. Returns
    /// how it ended; `Err` only for an authentication failure (no request
    /// can succeed) or an observer error.
    pub fn request(
        &mut self,
        build: &dyn Fn(&str) -> ChatRequest,
        observer: &mut dyn RequestObserver,
    ) -> anyhow::Result<RequestEnd> {
        let provider = self.backend.provider().as_str();
        let mut attempts = 0u32;
        let mut retries = 0u32;
        let mut waits = 0u32;
        loop {
            if self.allowance == 0 {
                return Ok(if attempts == 0 {
                    RequestEnd::OutOfBudget
                } else {
                    RequestEnd::Failed(format!(
                        "the request budget ran out after {attempts} attempt(s) at this prompt"
                    ))
                });
            }
            let now = self.env.now_unix_ms();
            let index = match self.pool.pick(now, self.policy.max_idle_ms()) {
                Pick::Use(index) => index,
                Pick::Sleep {
                    index,
                    until_unix_ms,
                } => {
                    let wait = until_unix_ms.saturating_sub(now);
                    eprintln!(
                        "[teacher] pausing {} until {}: waiting for `{}` ({})",
                        human_ms(wait),
                        ratelimit::format_unix_ms(until_unix_ms),
                        self.pool.id(index),
                        self.pool
                            .slots()
                            .get(index)
                            .map_or("", |s| s.reason.as_str())
                    );
                    self.env.sleep(Duration::from_millis(wait));
                    continue;
                }
                Pick::Exhausted { reason, .. } => return Ok(RequestEnd::Stopped(reason)),
            };
            if let Some(previous) = self.pool.switch_to(index) {
                let why = self
                    .pool
                    .slots()
                    .get(previous)
                    .map(|s| match &s.disabled {
                        Some(reason) => format!("disabled: {reason}"),
                        None if s.paused_until_unix_ms > now => format!(
                            "{} until {}",
                            s.reason,
                            ratelimit::format_unix_ms(s.paused_until_unix_ms)
                        ),
                        None => "usable again, preferred by priority".to_string(),
                    })
                    .unwrap_or_default();
                eprintln!(
                    "[teacher] rotating {provider} model `{}` -> `{}` ({why})",
                    self.pool.id(previous),
                    self.pool.id(index)
                );
            }
            if self.policy.min_interval_ms > 0 {
                if let Some(last) = self.last_request_unix_ms {
                    let next = last.saturating_add(self.policy.min_interval_ms);
                    if next > now {
                        self.env.sleep(Duration::from_millis(next - now));
                    }
                }
            }
            let model = self.pool.id(index).to_string();
            let request = build(&model);
            let sent_at = self.env.now_unix_ms();
            observer.before_send(&model, sent_at)?;
            self.allowance -= 1;
            self.requests += 1;
            attempts += 1;
            self.last_request_unix_ms = Some(sent_at);
            let result = self.backend.send(&request);
            let after = self.env.now_unix_ms();
            match interpret(result, &model, self.backend.key_env(), after) {
                Outcome::Reply(reply) => {
                    self.pool.on_success(index);
                    if let Some(delay) = reply
                        .quota
                        .as_ref()
                        .and_then(|q| ratelimit::pacing_delay(q, after, &self.policy))
                    {
                        let until = after.saturating_add(duration_ms(delay));
                        self.pool.pace(index, until);
                        eprintln!(
                            "[teacher] `{model}`: quota nearly spent ({} left); spacing the next \
                             request by {}",
                            reply
                                .quota
                                .and_then(|q| q.remaining)
                                .map_or_else(|| "?".to_string(), |r| r.to_string()),
                            human_ms(duration_ms(delay))
                        );
                    }
                    return Ok(RequestEnd::Reply {
                        reply,
                        model,
                        latency_ms: after.saturating_sub(sent_at),
                    });
                }
                Outcome::Limited(signal) => {
                    self.rate_limit_events += 1;
                    *self.events_by_kind.entry(signal.kind).or_insert(0) += 1;
                    let plan = ratelimit::plan_pause(
                        &signal,
                        after,
                        self.pool.consecutive(index),
                        &self.policy,
                        self.env.jitter_unit(),
                    );
                    if signal.kind == RateLimitKind::Overloaded {
                        self.pool.note_failure(index);
                    }
                    let affected: Vec<usize> = if free_tier_account_limit(&signal) {
                        (0..self.pool.slots().len())
                            .filter(|&i| i == index || self.pool.id(i).ends_with(":free"))
                            .collect()
                    } else {
                        vec![index]
                    };
                    for &i in &affected {
                        self.pool.apply(i, &plan);
                    }
                    let affected_ids: Vec<String> =
                        affected.iter().map(|&i| self.pool.id(i).to_string()).collect();
                    let when = if plan.verdict == PauseVerdict::Disable {
                        "for the rest of the run".to_string()
                    } else {
                        format!(
                            "until {} (in {})",
                            ratelimit::format_unix_ms(plan.until_unix_ms),
                            human_ms(plan.wait_ms)
                        )
                    };
                    eprintln!(
                        "[teacher] {provider} `{model}`: {} limit; {} {when}: {}{}. Evidence: {}",
                        signal.kind.as_str(),
                        match plan.verdict {
                            PauseVerdict::Wait => "pausing",
                            PauseVerdict::Rotate => "benching",
                            PauseVerdict::Disable => "disabling",
                        },
                        plan.reason,
                        if affected.len() > 1 {
                            format!(
                                " (account-wide free-tier limit: applied to {} :free models)",
                                affected.len()
                            )
                        } else {
                            String::new()
                        },
                        signal.evidence
                    );
                    observer.on_pause(Some(&signal), &plan, &affected_ids, after)?;
                    waits += 1;
                    if waits > self.policy.max_waits_per_prompt {
                        return Ok(RequestEnd::Failed(format!(
                            "gave up after {waits} rate-limit pauses at this prompt \
                             (max_waits_per_prompt); last: {}",
                            signal.evidence
                        )));
                    }
                }
                Outcome::Retryable(message) => {
                    retries += 1;
                    eprintln!("[teacher] `{model}`: attempt {attempts} failed: {message}");
                    let benched = self.bench_if_failing(index, after, observer)?;
                    if retries > self.max_retries || self.allowance == 0 {
                        return Ok(RequestEnd::Failed(format!(
                            "{message} (after {attempts} attempt(s){})",
                            if self.allowance == 0 && retries <= self.max_retries {
                                "; the request budget ran out before the retries did"
                            } else {
                                ""
                            }
                        )));
                    }
                    if !benched {
                        let factor = 2u64.saturating_pow(retries.saturating_sub(1).min(16));
                        let wait = self.backoff_base_ms.saturating_mul(factor);
                        eprintln!("[teacher] retry {retries} after {wait} ms");
                        self.env.sleep(Duration::from_millis(wait));
                    }
                }
                Outcome::ModelUnavailable(message) => {
                    eprintln!("[teacher] disabling `{model}` for this run: {message}");
                    self.pool.disable(index, message.clone());
                    observer.on_disabled(&model, &message, after)?;
                }
                Outcome::Fatal(message) => {
                    bail!("{provider} teacher request for `{model}` failed: {message}")
                }
                Outcome::Failed(message) => {
                    self.bench_if_failing(index, after, observer)?;
                    return Ok(RequestEnd::Failed(message));
                }
            }
        }
    }

    /// Count a failure on slot `index` and bench it when the streak reaches
    /// `max_consecutive_failures`. Returns whether it was benched.
    fn bench_if_failing(
        &mut self,
        index: usize,
        now: u64,
        observer: &mut dyn RequestObserver,
    ) -> anyhow::Result<bool> {
        let streak = self.pool.note_failure(index);
        if streak < self.policy.max_consecutive_failures.max(1) {
            return Ok(false);
        }
        let wait = self.policy.max_pause_ms();
        let plan = PausePlan {
            kind: RateLimitKind::Overloaded,
            verdict: PauseVerdict::Rotate,
            until_unix_ms: now.saturating_add(wait),
            wait_ms: wait,
            jitter_ms: 0,
            reason: format!("failed {streak} times in a row; benched for max_pause"),
        };
        self.pool.apply(index, &plan);
        let id = self.pool.id(index).to_string();
        eprintln!(
            "[teacher] benching `{id}` until {}: {}",
            ratelimit::format_unix_ms(plan.until_unix_ms),
            plan.reason
        );
        observer.on_pause(None, &plan, &[id], now)?;
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// The journal
// ---------------------------------------------------------------------------

/// Journal schema version.
pub const JOURNAL_VERSION: u32 = 1;

/// One model's state in a [`RunJournal`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelState {
    /// Provider the model was used through.
    #[serde(default)]
    pub provider: String,
    /// Traces it produced that reached the file.
    #[serde(default)]
    pub served: usize,
    /// Rate-limit signals it drew.
    #[serde(default)]
    pub rate_limits: usize,
    /// Failure streaks that benched it.
    #[serde(default)]
    pub benches: usize,
    /// Unusable before this time (Unix ms), when paused.
    #[serde(default)]
    pub paused_until_unix_ms: Option<u64>,
    /// The same, readable.
    #[serde(default)]
    pub paused_until: Option<String>,
    /// The limit behind the pause.
    #[serde(default)]
    pub pause_kind: Option<RateLimitKind>,
    /// Wait or rotate.
    #[serde(default)]
    pub pause_verdict: Option<PauseVerdict>,
    /// Why.
    #[serde(default)]
    pub pause_reason: Option<String>,
    /// Why it was taken out of the last run (credits, gone). Informational:
    /// a rerun tries it again, since credits can be added.
    #[serde(default)]
    pub last_disabled: Option<String>,
}

/// The resume journal written beside a trace file (`<out>.state.json`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunJournal {
    /// [`JOURNAL_VERSION`].
    #[serde(default)]
    pub version: u32,
    /// Prompt ids whose trace is in the file.
    #[serde(default)]
    pub completed: BTreeSet<String>,
    /// HTTP requests the file has cost across every run (reserved before
    /// each send). This is what the budget is enforced against.
    #[serde(default)]
    pub requests_spent: usize,
    /// Rate-limit, credit and overload responses met across every run.
    #[serde(default)]
    pub rate_limit_events: usize,
    /// Per-model state, keyed by exact model id.
    #[serde(default)]
    pub models: BTreeMap<String, ModelState>,
    /// The model the file is locked to under forbid-mixing.
    #[serde(default)]
    pub locked_model: Option<String>,
    /// Last write (Unix ms).
    #[serde(default)]
    pub updated_at_unix_ms: u64,
    /// Last write, readable.
    #[serde(default)]
    pub updated_at: String,
}

/// `<out>.state.json`.
pub fn journal_path(out_path: &Path) -> PathBuf {
    let mut name = out_path.as_os_str().to_owned();
    name.push(".state.json");
    PathBuf::from(name)
}

impl RunJournal {
    /// Read a journal; `None` when there is none yet. A corrupt journal is a
    /// loud error: it holds the request count the budget is enforced against.
    pub fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(err).with_context(|| format!("read run journal {}", path.display()))
            }
        };
        let journal: Self = serde_json::from_str(&text).with_context(|| {
            format!(
                "run journal {} is corrupt. It holds the request count the budget is enforced \
                 against and the pauses a resume honours; repair it, or delete it to reset both \
                 (the trace file alone still prevents re-requesting completed prompts)",
                path.display()
            )
        })?;
        ensure!(
            journal.version <= JOURNAL_VERSION,
            "run journal {} has version {}, newer than this build understands ({JOURNAL_VERSION})",
            path.display(),
            journal.version
        );
        Ok(Some(journal))
    }

    /// Write the journal atomically.
    pub fn save(&mut self, path: &Path, now_unix_ms: u64) -> anyhow::Result<()> {
        self.version = JOURNAL_VERSION;
        self.updated_at_unix_ms = now_unix_ms;
        self.updated_at = ratelimit::format_unix_ms(now_unix_ms);
        let text = serde_json::to_string_pretty(self).context("serialize run journal")?;
        crate::interop::write_atomic(path, text.as_bytes())
    }
}

/// Keeps a [`RunJournal`] in step with the request loop.
struct JournalSync<'j> {
    journal: &'j mut RunJournal,
    path: &'j Path,
    provider: &'j str,
}

impl RequestObserver for JournalSync<'_> {
    fn before_send(&mut self, model: &str, now_unix_ms: u64) -> anyhow::Result<()> {
        self.journal.requests_spent += 1;
        self.journal
            .models
            .entry(model.to_string())
            .or_default()
            .provider = self.provider.to_string();
        self.journal.save(self.path, now_unix_ms)
    }

    fn on_pause(
        &mut self,
        signal: Option<&RateLimitSignal>,
        plan: &PausePlan,
        affected: &[String],
        now_unix_ms: u64,
    ) -> anyhow::Result<()> {
        if signal.is_some() {
            self.journal.rate_limit_events += 1;
        }
        for id in affected {
            let state = self.journal.models.entry(id.clone()).or_default();
            state.provider = self.provider.to_string();
            if signal.is_some() {
                state.rate_limits += 1;
            } else {
                state.benches += 1;
            }
            if plan.verdict == PauseVerdict::Disable {
                state.last_disabled = Some(plan.reason.clone());
            } else {
                state.paused_until_unix_ms = Some(plan.until_unix_ms);
                state.paused_until = Some(ratelimit::format_unix_ms(plan.until_unix_ms));
                state.pause_kind = Some(plan.kind);
                state.pause_verdict = Some(plan.verdict);
                state.pause_reason = Some(plan.reason.clone());
            }
        }
        self.journal.save(self.path, now_unix_ms)
    }

    fn on_disabled(&mut self, model: &str, reason: &str, now_unix_ms: u64) -> anyhow::Result<()> {
        self.journal
            .models
            .entry(model.to_string())
            .or_default()
            .last_disabled = Some(reason.to_string());
        self.journal.save(self.path, now_unix_ms)
    }
}

// ---------------------------------------------------------------------------
// The trace file
// ---------------------------------------------------------------------------

/// An append-only, fsynced JSONL trace file.
struct TraceFile {
    path: PathBuf,
    ids: HashSet<String>,
    /// The model each existing teacher record was requested from.
    models: BTreeSet<String>,
    /// The file's last line has no newline (hand-written); add one first.
    needs_newline: bool,
}

impl TraceFile {
    /// Open (creating when absent) and index `path`. A torn final line — an
    /// unterminated line that does not parse, left by a write the process
    /// died in — is cut off with a log line; any other unparseable line is
    /// an error naming it, because appending to a corrupt file buries it.
    fn open(path: &Path) -> anyhow::Result<Self> {
        let mut text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                std::fs::File::create(path)
                    .and_then(|file| file.sync_all())
                    .with_context(|| format!("create trace file {}", path.display()))?;
                sync_parent(path)?;
                String::new()
            }
            Err(err) => {
                return Err(err).with_context(|| format!("read trace file {}", path.display()))
            }
        };
        let mut needs_newline = false;
        if !text.is_empty() && !text.ends_with('\n') {
            let cut = text.rfind('\n').map_or(0, |at| at + 1);
            let tail_parses = text
                .get(cut..)
                .is_some_and(|tail| serde_json::from_str::<serde_json::Value>(tail).is_ok());
            if tail_parses {
                needs_newline = true;
            } else {
                let torn = text.len() - cut;
                let keep = u64::try_from(cut).context("trace file offset overflows u64")?;
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(path)
                    .and_then(|file| {
                        file.set_len(keep)?;
                        file.sync_all()
                    })
                    .with_context(|| format!("truncate torn line of {}", path.display()))?;
                eprintln!(
                    "[teacher] {}: dropped a torn final line ({torn} bytes) left by an \
                     interrupted write",
                    path.display()
                );
                text.truncate(cut);
            }
        }
        let mut ids = HashSet::new();
        let mut models = BTreeSet::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line).with_context(|| {
                format!(
                    "trace file {} line {} is not valid JSON; fix or remove it before resuming",
                    path.display(),
                    index + 1
                )
            })?;
            if let Some(id) = value.get("id").and_then(serde_json::Value::as_str) {
                ids.insert(id.to_string());
            }
            let requested = value
                .get("requested_model")
                .and_then(serde_json::Value::as_str)
                .filter(|m| !m.is_empty())
                .or_else(|| value.get("model").and_then(serde_json::Value::as_str));
            if let Some(model) = requested {
                models.insert(model.to_string());
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            ids,
            models,
            needs_newline,
        })
    }

    /// Append one record and flush it to disk before returning.
    fn append(&mut self, record: &TraceRecord) -> anyhow::Result<()> {
        let mut line = String::new();
        if self.needs_newline {
            line.push('\n');
        }
        line.push_str(&serde_json::to_string(record).context("serialize trace record")?);
        line.push('\n');
        std::fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .and_then(|mut file| {
                file.write_all(line.as_bytes())?;
                file.sync_data()
            })
            .with_context(|| format!("append to trace file {}", self.path.display()))?;
        self.needs_newline = false;
        self.ids.insert(record.id.clone());
        Ok(())
    }
}

/// Flush a directory entry (a newly created file) to disk.
fn sync_parent(path: &Path) -> anyhow::Result<()> {
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    std::fs::File::open(dir)
        .and_then(|handle| handle.sync_all())
        .with_context(|| format!("flush directory {} to disk", dir.display()))
}

// ---------------------------------------------------------------------------
// Resumable generation
// ---------------------------------------------------------------------------

/// Which models a run uses and how it pauses. Every field carries
/// `serde(default)`; this is what [`crate::langdata::PipelineConfig`]
/// embeds.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Model ids in priority order. Empty means "the client's configured
    /// model".
    #[serde(default)]
    pub models: Vec<String>,
    /// Pause, rotation and pacing bounds.
    #[serde(default)]
    pub pause: PausePolicy,
    /// Lock the output file to one model; never rotate to another.
    #[serde(default)]
    pub forbid_mixing: bool,
}

/// Everything [`generate_resumable`] needs besides the transport.
pub struct GenerationSettings<'a> {
    /// Cumulative request cap for the output file.
    pub budget: TraceBudget,
    /// The data-side gate replies pass before they are written.
    pub gate: &'a CheatGate,
    /// Models and pause policy (`models` must be non-empty here).
    pub pool: &'a PoolConfig,
    /// Retries per prompt for transport failures and plain 5xx.
    pub max_retries: u32,
    /// Base of their exponential backoff, ms.
    pub backoff_base_ms: u64,
}

/// Generate traces for `prompts` into the JSONL file `out_path`, durably and
/// resumably (see the module docs). Rate limits pause, rotate and resume;
/// when every model is paused past `max_idle_secs` the run stops with
/// [`TraceReport::stopped`] set and everything saved — rerun the same call to
/// continue.
pub fn generate_resumable(
    backend: &dyn ChatBackend,
    env: &dyn RunEnv,
    prompts: &[TracePrompt],
    out_path: &Path,
    settings: &GenerationSettings<'_>,
) -> anyhow::Result<TraceReport> {
    if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create trace output directory {}", parent.display()))?;
    }
    let mut pool = ModelPool::new(&settings.pool.models)?;
    let mut trace_file = TraceFile::open(out_path)?;
    let jpath = journal_path(out_path);
    let mut journal = RunJournal::load(&jpath)?.unwrap_or_default();
    journal.completed.extend(trace_file.ids.iter().cloned());
    let provider = backend.provider().as_str();
    let now = env.now_unix_ms();

    if settings.pool.forbid_mixing {
        ensure!(
            trace_file.models.len() <= 1,
            "forbid-mixing: {} already holds traces from {} models ({}); write to a new file",
            out_path.display(),
            trace_file.models.len(),
            trace_file
                .models
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
        let lock = journal
            .locked_model
            .clone()
            .or_else(|| trace_file.models.iter().next().cloned());
        if let Some(lock) = lock {
            let index = pool.index_of(&lock).ok_or_else(|| {
                anyhow!(
                    "forbid-mixing: {} was written by model `{lock}`, which is not in the pool \
                     [{}]; add it to the pool or write to a new file",
                    out_path.display(),
                    settings.pool.models.join(", ")
                )
            })?;
            pool.lock(index);
            journal.locked_model = Some(lock);
        }
    }
    for index in 0..pool.slots().len() {
        let id = pool.id(index).to_string();
        let Some(state) = journal.models.get(&id) else {
            continue;
        };
        if let Some(until) = state.paused_until_unix_ms.filter(|until| *until > now) {
            let reason = state
                .pause_reason
                .clone()
                .unwrap_or_else(|| "paused by an earlier run".to_string());
            eprintln!(
                "[teacher] honouring a pause recorded by an earlier run: `{id}` until {} ({reason})",
                ratelimit::format_unix_ms(until)
            );
            pool.restore(
                index,
                until,
                state.pause_kind,
                state.pause_verdict.unwrap_or(PauseVerdict::Wait),
                reason,
            );
        }
    }

    let spent_before = journal.requests_spent;
    let allowance = settings.budget.max_requests.saturating_sub(spent_before);
    if spent_before > 0 {
        eprintln!(
            "[teacher] {}: {spent_before} of {} requests already spent by earlier runs; {allowance} left",
            out_path.display(),
            settings.budget.max_requests
        );
    }
    let mut requester = Requester::new(
        backend,
        env,
        pool,
        settings.pool.pause.clone(),
        settings.max_retries,
        settings.backoff_base_ms,
        allowance,
    );
    let mut report = TraceReport {
        out_path: out_path.to_path_buf(),
        journal_path: jpath.clone(),
        ..TraceReport::default()
    };
    let mut stopped: Option<String> = None;
    for prompt in prompts {
        let id = prompt.id();
        if journal.completed.contains(&id) {
            report.skipped_existing += 1;
            continue;
        }
        if stopped.is_some() {
            report.skipped_paused += 1;
            continue;
        }
        let build = |model: &str| crate::nim::trace_request(model, prompt);
        let end = requester.request(
            &build,
            &mut JournalSync {
                journal: &mut journal,
                path: &jpath,
                provider,
            },
        )?;
        match end {
            RequestEnd::OutOfBudget => report.skipped_budget += 1,
            RequestEnd::Stopped(reason) => {
                report.skipped_paused += 1;
                stopped = Some(reason);
            }
            RequestEnd::Failed(message) => {
                eprintln!("[teacher] trace request for prompt id {id} failed: {message}");
                report.failed += 1;
            }
            RequestEnd::Reply { reply, model, .. } => {
                let Some(content) = reply.content else {
                    eprintln!(
                        "[teacher] trace request for prompt id {id} failed: `{model}` returned no \
                         content (finish_reason: {}); reasoning models need a larger max_tokens \
                         than {}",
                        reply.finish_reason.as_deref().unwrap_or("unreported"),
                        prompt.max_tokens
                    );
                    report.failed += 1;
                    continue;
                };
                // A language the registry does not know still gets every
                // language-independent class (secrets, encoding, blobs,
                // brokenness); it just cannot get a language-mixing verdict.
                let snippet_lang = CodeLanguage::parse(&prompt.language)
                    .map_or(SnippetLanguage::Other, crate::langdata::snippet_language);
                let offending =
                    settings
                        .gate
                        .inspect(None, &content, snippet_lang, &mut report.cheat_scan);
                report.cheat_scan.observe(&offending);
                if !offending.is_empty() {
                    eprintln!(
                        "[teacher] trace {id} rejected by the cheat gate: {}",
                        offending.join(", ")
                    );
                    continue;
                }
                let record = TraceRecord {
                    id: id.clone(),
                    language: prompt.language.clone(),
                    prompt: prompt.prompt.clone(),
                    completion: content,
                    model: reply.served_model,
                    usage: reply.usage,
                    seed: prompt.seed,
                    source: crate::nim::default_trace_source(),
                    provider: provider.to_string(),
                    requested_model: model.clone(),
                };
                trace_file.append(&record)?;
                journal.completed.insert(id);
                journal.models.entry(model.clone()).or_default().served += 1;
                if settings.pool.forbid_mixing && journal.locked_model.is_none() {
                    if let Some(index) = requester.pool().index_of(&model) {
                        requester.pool_mut().lock(index);
                    }
                    journal.locked_model = Some(model.clone());
                }
                journal.save(&jpath, env.now_unix_ms())?;
                report.served += 1;
                *report.served_by_model.entry(model).or_insert(0) += 1;
            }
        }
    }
    report.requests_this_run = requester.requests();
    report.rate_limit_events = requester.rate_limit_events();
    report.requests_spent_total = journal.requests_spent;
    report.stopped =
        stopped.map(|reason| format!("{reason}; rerun the same command to resume from here"));
    journal.save(&jpath, env.now_unix_ms())?;
    eprintln!(
        "[teacher] traces: {} served, {} skipped (existing), {} skipped (budget), {} skipped \
         (paused), {} failed; {} request(s) this run, {} of {} spent in total, {} rate-limit \
         event(s) -> {}",
        report.served,
        report.skipped_existing,
        report.skipped_budget,
        report.skipped_paused,
        report.failed,
        report.requests_this_run,
        report.requests_spent_total,
        settings.budget.max_requests,
        report.rate_limit_events,
        out_path.display()
    );
    if let Some(reason) = &report.stopped {
        eprintln!("[teacher] stopped early: {reason}");
    }
    Ok(report)
}

// ---------------------------------------------------------------------------
// Free-model discovery
// ---------------------------------------------------------------------------

/// Why a listed model counts as free.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FreeRule {
    /// Prompt and completion prices are both zero.
    ZeroPricing,
    /// The id ends in `:free`.
    FreeSuffix,
    /// Both rules match.
    Both,
}

/// One entry of a `{base}/models` listing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ListedModel {
    /// Exact model id, as requests must name it.
    pub id: String,
    /// Display name, when listed.
    #[serde(default)]
    pub name: String,
    /// Prompt price per token as listed (a decimal string).
    #[serde(default)]
    pub pricing_prompt: Option<String>,
    /// Completion price per token as listed.
    #[serde(default)]
    pub pricing_completion: Option<String>,
    /// Context window, when listed.
    #[serde(default)]
    pub context_length: Option<u64>,
    /// Which free rule matched, if any.
    #[serde(default)]
    pub free: Option<FreeRule>,
}

/// A price string that is exactly zero (`"0"`, `"0.0"`, `"0.000000"`).
/// Negative prices (OpenRouter lists `-1` for variable-priced routers) are
/// not zero.
pub fn is_zero_price(text: &str) -> bool {
    text.trim()
        .parse::<f64>()
        .is_ok_and(|price| price.is_finite() && price.abs() < f64::MIN_POSITIVE)
}

/// Which free rule a model matches.
pub fn free_rule(id: &str, prompt: Option<&str>, completion: Option<&str>) -> Option<FreeRule> {
    let zero = prompt.is_some_and(is_zero_price) && completion.is_some_and(is_zero_price);
    let suffix = id.ends_with(":free");
    match (zero, suffix) {
        (true, true) => Some(FreeRule::Both),
        (true, false) => Some(FreeRule::ZeroPricing),
        (false, true) => Some(FreeRule::FreeSuffix),
        (false, false) => None,
    }
}

fn price_text(value: Option<&serde_json::Value>) -> Option<String> {
    let value = value?;
    match value.as_str() {
        Some(text) => Some(text.to_string()),
        None if value.is_number() => Some(value.to_string()),
        None => None,
    }
}

/// Parse a `/models` listing (`{"data":[...]}` or a bare array). Entries
/// without an id are skipped; a body that is not a listing is an error.
pub fn parse_model_listing(json: &str) -> anyhow::Result<Vec<ListedModel>> {
    let value: serde_json::Value = serde_json::from_str(json)
        .with_context(|| format!("model listing is not JSON: {}", short(json, 200)))?;
    let entries = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .or_else(|| value.as_array())
        .ok_or_else(|| anyhow!("model listing has no `data` array: {}", short(json, 200)))?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(id) = entry.get("id").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let pricing = entry.get("pricing");
        let pricing_prompt = price_text(pricing.and_then(|p| p.get("prompt")));
        let pricing_completion = price_text(pricing.and_then(|p| p.get("completion")));
        let context_length = entry
            .get("context_length")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| {
                entry
                    .get("top_provider")
                    .and_then(|p| p.get("context_length"))
                    .and_then(serde_json::Value::as_u64)
            });
        out.push(ListedModel {
            id: id.to_string(),
            name: entry
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string(),
            free: free_rule(id, pricing_prompt.as_deref(), pricing_completion.as_deref()),
            pricing_prompt,
            pricing_completion,
            context_length,
        });
    }
    Ok(out)
}

/// Keep the models that are free (when `free_only`) and whose id or name
/// contains any of `needles` (case-insensitive; no needles keeps all).
pub fn filter_models(models: &[ListedModel], free_only: bool, needles: &[String]) -> Vec<ListedModel> {
    let needles: Vec<String> = needles
        .iter()
        .map(|n| n.trim().to_lowercase())
        .filter(|n| !n.is_empty())
        .collect();
    models
        .iter()
        .filter(|m| !free_only || m.free.is_some())
        .filter(|m| {
            needles.is_empty() || {
                let id = m.id.to_lowercase();
                let name = m.name.to_lowercase();
                needles.iter().any(|n| id.contains(n) || name.contains(n))
            }
        })
        .cloned()
        .collect()
}

/// `GET {base}/models` and parse it. The key is optional (OpenRouter's
/// listing is public).
pub fn fetch_models(config: &NimConfig, api_key: Option<&str>) -> anyhow::Result<Vec<ListedModel>> {
    let response = crate::nim::http_get(config, api_key, "models")?;
    if !response.is_success() {
        let limit = ratelimit::classify(&response, crate::nim::system_now_unix_ms())
            .map(|signal| format!(" ({} limit)", signal.kind.as_str()))
            .unwrap_or_default();
        bail!(
            "listing models at {} failed with HTTP {}{limit}: {}",
            config.base_url,
            response.status,
            short(&response.body, 240)
        );
    }
    parse_model_listing(&response.body)
}

/// Schema version of [`ModelCache`].
pub const MODEL_CACHE_VERSION: u32 = 1;

/// A saved discovery result (`dblocks teacher models --out`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCache {
    /// [`MODEL_CACHE_VERSION`].
    #[serde(default)]
    pub version: u32,
    /// Provider listed.
    #[serde(default)]
    pub provider: String,
    /// Endpoint listed.
    #[serde(default)]
    pub base_url: String,
    /// When (Unix ms).
    #[serde(default)]
    pub fetched_at_unix_ms: u64,
    /// When, readable.
    #[serde(default)]
    pub fetched_at: String,
    /// Whether only free models were kept.
    #[serde(default)]
    pub free_only: bool,
    /// The `--match` needles applied.
    #[serde(default)]
    pub matches: Vec<String>,
    /// Models in the full listing before filtering.
    #[serde(default)]
    pub total_listed: usize,
    /// The kept models, in listing order.
    #[serde(default)]
    pub models: Vec<ListedModel>,
}

impl ModelCache {
    /// Write the cache atomically as pretty JSON.
    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(self).context("serialize model cache")?;
        crate::interop::write_atomic(path, text.as_bytes())
    }

    /// Read a cache written by [`Self::write`].
    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read model listing {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("{} is not a `teacher models --out` listing", path.display()))
    }
}

/// How a command chooses its models.
#[derive(Clone, Debug, Default)]
pub struct ModelSelection {
    /// Explicit ids, in priority order (first).
    pub models: Vec<String>,
    /// A saved listing to take ids from (then filtered by `free`/`matches`).
    pub models_from: Option<PathBuf>,
    /// Keep only free models from a listing or a live discovery.
    pub free: bool,
    /// Case-insensitive needles a listed model must contain (any of).
    pub matches: Vec<String>,
}

/// Resolve a [`ModelSelection`] into an ordered, de-duplicated pool:
/// explicit ids first, then the saved listing (`models_from`) or — when
/// `free` or `matches` is set without one — a live `GET {base}/models`.
pub fn resolve_models(
    selection: &ModelSelection,
    config: &NimConfig,
    api_key: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |id: &str| {
        let id = id.trim();
        if !id.is_empty() && !out.iter().any(|existing| existing == id) {
            out.push(id.to_string());
        }
    };
    for id in &selection.models {
        push(id);
    }
    let discovered = if let Some(path) = &selection.models_from {
        let cache = ModelCache::read(path)?;
        Some(filter_models(&cache.models, selection.free, &selection.matches))
    } else if selection.free || !selection.matches.is_empty() {
        let listed = fetch_models(config, api_key)?;
        Some(filter_models(&listed, selection.free, &selection.matches))
    } else {
        None
    };
    if let Some(found) = &discovered {
        for model in found {
            push(&model.id);
        }
    }
    ensure!(
        !out.is_empty(),
        "no teacher models selected{}: pass --models a,b, --models-from <listing.json>, or \
         --free/--match to discover them",
        if discovered.is_some() {
            " (discovery matched nothing)"
        } else {
            ""
        }
    );
    Ok(out)
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

    use std::cell::{Cell, RefCell};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 2026-11-20T12:00:00Z.
    const NOW: u64 = 1_795_176_000_000;

    /// A virtual clock: sleeping advances it and is recorded.
    struct MockEnv {
        now: Cell<u64>,
        sleeps: RefCell<Vec<u64>>,
    }

    impl MockEnv {
        fn at(now: u64) -> Self {
            Self {
                now: Cell::new(now),
                sleeps: RefCell::new(Vec::new()),
            }
        }
    }

    impl RunEnv for MockEnv {
        fn now_unix_ms(&self) -> u64 {
            self.now.get()
        }
        fn sleep(&self, wait: Duration) {
            let ms = duration_ms(wait);
            self.sleeps.borrow_mut().push(ms);
            self.now.set(self.now.get() + ms);
        }
        fn jitter_unit(&self) -> f64 {
            0.0
        }
    }

    type Script = Box<dyn Fn(&str, usize, u64) -> Result<HttpResponse, String>>;

    /// A scripted backend: `(model, 1-based call number, now)` -> response.
    struct MockBackend<'e> {
        env: &'e MockEnv,
        script: Script,
        calls: RefCell<Vec<String>>,
    }

    impl<'e> MockBackend<'e> {
        fn new(
            env: &'e MockEnv,
            script: impl Fn(&str, usize, u64) -> Result<HttpResponse, String> + 'static,
        ) -> Self {
            Self {
                env,
                script: Box::new(script),
                calls: RefCell::new(Vec::new()),
            }
        }
        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }
    }

    impl ChatBackend for MockBackend<'_> {
        fn provider(&self) -> Provider {
            Provider::OpenRouter
        }
        fn key_env(&self) -> &str {
            "TEACHER_TEST_UNUSED_KEY"
        }
        fn send(&self, request: &ChatRequest) -> Result<HttpResponse, String> {
            self.calls.borrow_mut().push(request.model.clone());
            let n = self.calls.borrow().len();
            (self.script)(&request.model, n, self.env.now_unix_ms())
        }
    }

    fn ok(text: &str) -> Result<HttpResponse, String> {
        Ok(HttpResponse::new(
            200,
            vec![],
            serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            })
            .to_string(),
        ))
    }

    fn per_minute(retry_secs: u64) -> Result<HttpResponse, String> {
        Ok(HttpResponse::new(
            429,
            vec![("Retry-After".to_string(), retry_secs.to_string())],
            r#"{"error":{"code":429,"message":"Rate limit exceeded: per minute"}}"#,
        ))
    }

    fn daily_until(reset_ms: u64) -> Result<HttpResponse, String> {
        Ok(HttpResponse::new(
            429,
            vec![],
            format!(
                r#"{{"error":{{"code":429,"message":"Rate limit exceeded: model-per-day","metadata":{{"headers":{{"X-RateLimit-Limit":"50","X-RateLimit-Remaining":"0","X-RateLimit-Reset":"{reset_ms}"}}}}}}}}"#
            ),
        ))
    }

    fn quiet_policy() -> PausePolicy {
        PausePolicy {
            jitter_max_ms: 0,
            ..PausePolicy::default()
        }
    }

    fn prompts(n: usize) -> Vec<TracePrompt> {
        (0..n)
            .map(|i| TracePrompt {
                language: "rust".to_string(),
                prompt: format!("task {i}: implement a queue"),
                seed: i as u64,
                max_tokens: 64,
            })
            .collect()
    }

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "dblocks-teacher-{tag}-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn generate(
        backend: &dyn ChatBackend,
        env: &dyn RunEnv,
        prompts: &[TracePrompt],
        out: &Path,
        pool: &PoolConfig,
        max_requests: usize,
    ) -> anyhow::Result<TraceReport> {
        let gate = CheatGate::patterns_only();
        generate_resumable(
            backend,
            env,
            prompts,
            out,
            &GenerationSettings {
                budget: TraceBudget { max_requests },
                gate: &gate,
                pool,
                max_retries: 2,
                backoff_base_ms: 1,
            },
        )
    }

    fn pool_of(models: &[&str], pause: PausePolicy) -> PoolConfig {
        PoolConfig {
            models: models.iter().map(|m| m.to_string()).collect(),
            pause,
            forbid_mixing: false,
        }
    }

    const CODE: &str = "fn push(q: &mut Vec<i32>, x: i32) {\n    q.push(x);\n}\n";

    #[test]
    fn interpret_reads_each_shape() {
        let o = interpret(ok("hi"), "m", "K", NOW);
        assert!(matches!(o, Outcome::Reply(ref r) if r.content.as_deref() == Some("hi") && r.served_model == "m"));
        assert!(matches!(interpret(per_minute(5), "m", "K", NOW), Outcome::Limited(_)));
        let in_200 = Ok(HttpResponse::new(
            200,
            vec![],
            r#"{"error":{"code":429,"message":"Rate limit exceeded: free-models-per-min."}}"#,
        ));
        assert!(matches!(interpret(in_200, "m", "K", NOW), Outcome::Limited(_)));
        let auth = interpret(Ok(HttpResponse::new(401, vec![], "{}")), "m", "MY_KEY_VAR", NOW);
        assert!(matches!(auth, Outcome::Fatal(ref m) if m.contains("MY_KEY_VAR")));
        assert!(matches!(
            interpret(Ok(HttpResponse::new(404, vec![], "")), "m", "K", NOW),
            Outcome::ModelUnavailable(_)
        ));
        assert!(matches!(
            interpret(
                Ok(HttpResponse::new(400, vec![], r#"{"error":{"message":"x is not a valid model ID"}}"#)),
                "m",
                "K",
                NOW
            ),
            Outcome::ModelUnavailable(_)
        ));
        assert!(matches!(
            interpret(Ok(HttpResponse::new(500, vec![], "boom")), "m", "K", NOW),
            Outcome::Retryable(_)
        ));
        assert!(matches!(
            interpret(Ok(HttpResponse::new(400, vec![], "too long")), "m", "K", NOW),
            Outcome::Failed(_)
        ));
        assert!(matches!(interpret(Err("refused".into()), "m", "K", NOW), Outcome::Retryable(_)));
        // Null content is a reply with no content, not a parse failure.
        let null = Ok(HttpResponse::new(
            200,
            vec![],
            r#"{"choices":[{"message":{"content":null},"finish_reason":"length"}]}"#,
        ));
        assert!(matches!(interpret(null, "m", "K", NOW), Outcome::Reply(ref r) if r.content.is_none() && r.finish_reason.as_deref() == Some("length")));
    }

    #[test]
    fn rotation_follows_priority_and_returns_after_the_reset() {
        let mut pool = ModelPool::new(&["a".into(), "b".into(), "c".into()]).unwrap();
        let p = quiet_policy();
        assert_eq!(pool.pick(NOW, p.max_idle_ms()), Pick::Use(0));
        // a: daily cap far away -> benched; the pool rotates to b, not c.
        let daily = RateLimitSignal {
            kind: RateLimitKind::Daily,
            retry_after: None,
            reset_at_unix_ms: Some(NOW + 6 * 3_600_000),
            evidence: String::new(),
        };
        pool.apply(0, &ratelimit::plan_pause(&daily, NOW, 0, &p, 0.0));
        assert_eq!(pool.pick(NOW, p.max_idle_ms()), Pick::Use(1));
        assert_eq!(pool.switch_to(1), Some(0));
        // b: credits -> disabled; c is next.
        let credits = RateLimitSignal {
            kind: RateLimitKind::Credits,
            ..daily.clone()
        };
        pool.apply(1, &ratelimit::plan_pause(&credits, NOW, 0, &p, 0.0));
        assert_eq!(pool.pick(NOW, p.max_idle_ms()), Pick::Use(2));
        pool.switch_to(2);
        // A per-minute limit on the current model is waited for, not rotated.
        let minute = RateLimitSignal {
            kind: RateLimitKind::PerMinute,
            retry_after: Some(Duration::from_secs(30)),
            reset_at_unix_ms: None,
            evidence: String::new(),
        };
        pool.apply(2, &ratelimit::plan_pause(&minute, NOW, 0, &p, 0.0));
        assert_eq!(
            pool.pick(NOW, p.max_idle_ms()),
            Pick::Sleep {
                index: 2,
                until_unix_ms: NOW + 30_000
            }
        );
        // After a's reset passes, the pool comes back to a.
        assert_eq!(pool.pick(NOW + 6 * 3_600_000, p.max_idle_ms()), Pick::Use(0));
        // With everything paused past the idle bound, the run stops and says when to resume.
        let mut lone = ModelPool::new(&["x".into()]).unwrap();
        lone.apply(0, &ratelimit::plan_pause(&daily, NOW, 0, &p, 0.0));
        match lone.pick(NOW, 3_600_000) {
            Pick::Exhausted {
                earliest_resume_unix_ms,
                reason,
            } => {
                assert_eq!(earliest_resume_unix_ms, Some(NOW + 6 * 3_600_000));
                assert!(reason.contains("2026-11-20T18:00:00Z"), "{reason}");
            }
            other => panic!("expected exhaustion, got {other:?}"),
        }
        assert!(ModelPool::new(&[]).is_err());
        assert!(ModelPool::new(&["a".into(), "a".into()]).is_err());
    }

    #[test]
    fn a_per_minute_limit_pauses_and_resumes_on_the_same_model() {
        let env = MockEnv::at(NOW);
        let backend = MockBackend::new(&env, |_, n, _| if n == 1 { per_minute(20) } else { ok(CODE) });
        let dir = unique_tmp_dir("minute");
        let out = dir.join("traces.jsonl");
        let report = generate(
            &backend,
            &env,
            &prompts(2),
            &out,
            &pool_of(&["a:free", "b:free"], quiet_policy()),
            10,
        )
        .unwrap();
        assert_eq!(report.served, 2);
        assert_eq!(report.rate_limit_events, 1);
        assert_eq!(backend.calls(), vec!["a:free", "a:free", "a:free"], "no rotation on a per-minute limit");
        assert_eq!(*env.sleeps.borrow(), vec![20_000], "slept exactly the Retry-After");
        let records = crate::nim::read_traces(&out).unwrap();
        assert!(records.iter().all(|r| r.provider == "openrouter" && r.requested_model == "a:free"));
    }

    #[test]
    fn a_daily_cap_rotates_and_the_pool_comes_back_after_the_reset() {
        let env = MockEnv::at(NOW);
        let reset = NOW + 5 * 3_600_000;
        let backend = MockBackend::new(&env, move |model, _, now| {
            if model == "primary" && now < reset {
                daily_until(reset)
            } else {
                ok(CODE)
            }
        });
        let dir = unique_tmp_dir("daily");
        let out = dir.join("traces.jsonl");
        let pool = pool_of(&["primary", "backup"], quiet_policy());
        let first = generate(&backend, &env, &prompts(2), &out, &pool, 20).unwrap();
        assert_eq!(first.served_by_model.get("backup"), Some(&2));
        assert_eq!(backend.calls(), vec!["primary", "backup", "backup"]);
        // The journal remembers the bench; after the reset the next run uses primary again.
        let journal = RunJournal::load(&journal_path(&out)).unwrap().unwrap();
        let state = journal.models.get("primary").unwrap();
        assert_eq!(state.paused_until_unix_ms, Some(reset));
        assert_eq!(state.pause_verdict, Some(PauseVerdict::Rotate));
        env.now.set(reset + 1);
        let more = prompts(4);
        let second = generate(&backend, &env, &more, &out, &pool, 20).unwrap();
        assert_eq!(second.skipped_existing, 2);
        assert_eq!(second.served_by_model.get("primary"), Some(&2));
        // Every trace says which model wrote it.
        let records = crate::nim::read_traces(&out).unwrap();
        let models: Vec<&str> = records.iter().map(|r| r.requested_model.as_str()).collect();
        assert_eq!(models, vec!["backup", "backup", "primary", "primary"]);
    }

    #[test]
    fn a_crash_loses_nothing_written_and_the_budget_carries_over() {
        let env = MockEnv::at(NOW);
        let dir = unique_tmp_dir("crash");
        let out = dir.join("traces.jsonl");
        let pool = pool_of(&["m"], quiet_policy());
        // Three answers, then the key "expires": the run dies like a crash.
        let dying = MockBackend::new(&env, |_, n, _| {
            if n <= 3 {
                ok(CODE)
            } else {
                Ok(HttpResponse::new(401, vec![], "{}"))
            }
        });
        let err = generate(&dying, &env, &prompts(6), &out, &pool, 10).unwrap_err();
        assert!(format!("{err:#}").contains("TEACHER_TEST_UNUSED_KEY"));
        // An interrupted write leaves a torn line behind as well.
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&out).unwrap();
            f.write_all(b"{\"id\":\"torn").unwrap();
        }
        let journal = RunJournal::load(&journal_path(&out)).unwrap().unwrap();
        assert_eq!(journal.completed.len(), 3);
        assert_eq!(journal.requests_spent, 4, "the fatal request was reserved before it was sent");

        // Resume with a healthy backend and the same cap: 6 left of 10.
        let healthy = MockBackend::new(&env, |_, _, _| ok(CODE));
        let resumed = generate(&healthy, &env, &prompts(6), &out, &pool, 6).unwrap();
        assert_eq!(resumed.skipped_existing, 3);
        assert_eq!(resumed.served, 2, "the cap counts across runs: 6 - 4 spent = 2");
        assert_eq!(resumed.skipped_budget, 1);
        assert_eq!(resumed.requests_spent_total, 6);
        assert_eq!(healthy.calls().len(), 2);
        // No duplicate ids (read_traces rejects them) and the torn line is gone.
        let records = crate::nim::read_traces(&out).unwrap();
        assert_eq!(records.len(), 5);
        let expected: Vec<String> = prompts(5).iter().map(TracePrompt::id).collect();
        let got: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn a_recorded_pause_is_honoured_by_the_next_run() {
        let env = MockEnv::at(NOW);
        let reset = NOW + 8 * 3_600_000;
        let backend = MockBackend::new(&env, move |_, _, now| {
            if now < reset {
                daily_until(reset)
            } else {
                ok(CODE)
            }
        });
        let dir = unique_tmp_dir("honour");
        let out = dir.join("traces.jsonl");
        // max_idle 0: stop instead of sleeping eight hours.
        let pool = pool_of(
            &["m"],
            PausePolicy {
                max_idle_secs: 0,
                ..quiet_policy()
            },
        );
        let first = generate(&backend, &env, &prompts(3), &out, &pool, 10).unwrap();
        assert_eq!(first.served, 0);
        assert_eq!(first.skipped_paused, 3);
        let stopped = first.stopped.clone().unwrap();
        assert!(stopped.contains("2026-11-20T20:00:00Z"), "{stopped}");
        assert_eq!(backend.calls().len(), 1);
        // Same time, rerun: the pause is honoured without a request.
        let again = generate(&backend, &env, &prompts(3), &out, &pool, 10).unwrap();
        assert!(again.stopped.is_some());
        assert_eq!(backend.calls().len(), 1, "a paused model is not asked again");
        // After the reset the run completes.
        env.now.set(reset);
        let done = generate(&backend, &env, &prompts(3), &out, &pool, 10).unwrap();
        assert_eq!(done.served, 3);
        assert!(done.stopped.is_none());
        // With the default idle bound the same run would sleep through the cap instead.
        let env2 = MockEnv::at(NOW);
        let backend2 = MockBackend::new(&env2, move |_, _, now| {
            if now < reset {
                daily_until(reset)
            } else {
                ok(CODE)
            }
        });
        let out2 = unique_tmp_dir("honour-sleep").join("traces.jsonl");
        let slept = generate(&backend2, &env2, &prompts(2), &out2, &pool_of(&["m"], quiet_policy()), 10).unwrap();
        assert_eq!(slept.served, 2);
        assert_eq!(*env2.sleeps.borrow(), vec![8 * 3_600_000]);
    }

    #[test]
    fn forbid_mixing_never_rotates_and_refuses_a_foreign_file() {
        let env = MockEnv::at(NOW);
        let reset = NOW + 2 * 3_600_000;
        let backend = MockBackend::new(&env, move |model, n, now| {
            if model == "a" && n == 2 && now < reset {
                daily_until(reset)
            } else {
                ok(CODE)
            }
        });
        let dir = unique_tmp_dir("nomix");
        let out = dir.join("traces.jsonl");
        let mut pool = pool_of(&["a", "b"], quiet_policy());
        pool.forbid_mixing = true;
        let report = generate(&backend, &env, &prompts(3), &out, &pool, 10).unwrap();
        assert_eq!(report.served_by_model.get("a"), Some(&3));
        assert!(!backend.calls().contains(&"b".to_string()), "never rotated");
        // A pool without the file's model is refused.
        let foreign = PoolConfig {
            models: vec!["b".into()],
            ..pool.clone()
        };
        let err = generate(&backend, &env, &prompts(3), &out, &foreign, 10).unwrap_err();
        assert!(format!("{err:#}").contains("forbid-mixing"));
    }

    #[test]
    fn a_free_tier_account_limit_pauses_every_free_model() {
        let env = MockEnv::at(NOW);
        let backend = MockBackend::new(&env, |model, _, _| {
            if model.ends_with(":free") {
                Ok(HttpResponse::new(
                    429,
                    vec![],
                    format!(
                        r#"{{"error":{{"code":429,"message":"Rate limit exceeded: free-models-per-day","metadata":{{"headers":{{"X-RateLimit-Reset":"{}"}}}}}}}}"#,
                        NOW + 10 * 3_600_000
                    ),
                ))
            } else {
                ok(CODE)
            }
        });
        let out = unique_tmp_dir("account").join("traces.jsonl");
        let report = generate(
            &backend,
            &env,
            &prompts(1),
            &out,
            &pool_of(&["x:free", "y:free", "paid"], quiet_policy()),
            10,
        )
        .unwrap();
        assert_eq!(report.served_by_model.get("paid"), Some(&1));
        assert_eq!(backend.calls(), vec!["x:free", "paid"], "y:free was not tried in vain");
    }

    #[test]
    fn a_model_the_endpoint_lacks_is_disabled_and_skipped() {
        let env = MockEnv::at(NOW);
        let backend = MockBackend::new(&env, |model, _, _| {
            if model == "gone" {
                Ok(HttpResponse::new(410, vec![], "retired"))
            } else {
                ok(CODE)
            }
        });
        let out = unique_tmp_dir("gone").join("traces.jsonl");
        let report = generate(&backend, &env, &prompts(2), &out, &pool_of(&["gone", "live"], quiet_policy()), 10).unwrap();
        assert_eq!(report.served_by_model.get("live"), Some(&2));
        assert_eq!(backend.calls(), vec!["gone", "live", "live"]);
    }

    #[test]
    fn low_remaining_quota_spaces_requests() {
        let env = MockEnv::at(NOW);
        let backend = MockBackend::new(&env, |_, _, now| {
            let mut r = ok(CODE).unwrap();
            r.headers = vec![
                ("x-ratelimit-limit".into(), "20".into()),
                ("x-ratelimit-remaining".into(), "1".into()),
                ("x-ratelimit-reset".into(), (now + 30_000).to_string()),
            ];
            Ok(r)
        });
        let out = unique_tmp_dir("pace").join("traces.jsonl");
        let report = generate(&backend, &env, &prompts(2), &out, &pool_of(&["m"], quiet_policy()), 10).unwrap();
        assert_eq!(report.served, 2);
        assert_eq!(report.rate_limit_events, 0, "pacing is not a rate-limit event");
        assert_eq!(env.sleeps.borrow().first(), Some(&15_000), "30 s window / (1 + 1)");
    }

    const LISTING: &str = r#"{"data":[
        {"id":"nvidia/nemotron-super-x:free","name":"NVIDIA: Nemotron Super X (free)","pricing":{"prompt":"0","completion":"0"},"context_length":131072},
        {"id":"nvidia/nemotron-nano-y","name":"NVIDIA: Nemotron Nano Y","pricing":{"prompt":"0.0000000","completion":"0.000000"}},
        {"id":"vendor/odd-one:free","name":"Odd (free)","pricing":{"prompt":"0.000001","completion":"0.000002"}},
        {"id":"vendor/paid","name":"Paid","pricing":{"prompt":"0.000003","completion":"0.000015"}},
        {"id":"openrouter/auto","name":"Auto","pricing":{"prompt":"-1","completion":"-1"}},
        {"name":"no id"}
    ]}"#;

    #[test]
    fn discovery_marks_free_models_and_says_which_rule_matched() {
        let models = parse_model_listing(LISTING).unwrap();
        assert_eq!(models.len(), 5, "the entry without an id is skipped");
        let rule = |id: &str| models.iter().find(|m| m.id == id).unwrap().free;
        assert_eq!(rule("nvidia/nemotron-super-x:free"), Some(FreeRule::Both));
        assert_eq!(rule("nvidia/nemotron-nano-y"), Some(FreeRule::ZeroPricing));
        assert_eq!(rule("vendor/odd-one:free"), Some(FreeRule::FreeSuffix));
        assert_eq!(rule("vendor/paid"), None);
        assert_eq!(rule("openrouter/auto"), None, "negative prices are not zero");

        let free = filter_models(&models, true, &[]);
        assert_eq!(free.len(), 3);
        let nvidia = filter_models(&models, true, &["NVIDIA".to_string()]);
        assert_eq!(
            nvidia.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["nvidia/nemotron-super-x:free", "nvidia/nemotron-nano-y"]
        );
        let super_ = filter_models(&models, true, &["super".to_string(), "  ".to_string()]);
        assert_eq!(super_.len(), 1);
        assert_eq!(filter_models(&models, false, &["paid".to_string()]).len(), 1);
        assert!(parse_model_listing("{}").is_err());
        assert!(parse_model_listing("nope").is_err());
    }

    #[test]
    fn a_saved_listing_resolves_offline() {
        let dir = unique_tmp_dir("cache");
        let path = dir.join("models.json");
        let cache = ModelCache {
            version: MODEL_CACHE_VERSION,
            provider: "openrouter".into(),
            base_url: crate::nim::OPENROUTER_BASE_URL.into(),
            fetched_at_unix_ms: NOW,
            fetched_at: ratelimit::format_unix_ms(NOW),
            free_only: true,
            matches: vec![],
            total_listed: 5,
            models: parse_model_listing(LISTING).unwrap(),
        };
        cache.write(&path).unwrap();
        assert_eq!(ModelCache::read(&path).unwrap(), cache);
        let selection = ModelSelection {
            models: vec!["first/explicit".into(), "nvidia/nemotron-nano-y".into()],
            models_from: Some(path),
            free: true,
            matches: vec!["nemotron".into()],
        };
        let ids = resolve_models(&selection, &NimConfig::for_provider(Provider::OpenRouter), None).unwrap();
        assert_eq!(
            ids,
            vec!["first/explicit", "nvidia/nemotron-nano-y", "nvidia/nemotron-super-x:free"],
            "explicit ids first, then the listing, without duplicates"
        );
        let empty = ModelSelection::default();
        assert!(resolve_models(&empty, &NimConfig::default(), None).is_err());
    }

    #[test]
    fn journal_path_appends_the_suffix() {
        assert_eq!(
            journal_path(Path::new("/x/traces.jsonl")),
            PathBuf::from("/x/traces.jsonl.state.json")
        );
    }
}
