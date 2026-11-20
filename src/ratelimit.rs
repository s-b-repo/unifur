//! Rate-limit detection and pause scheduling for the teacher clients.
//!
//! Everything here is a pure function of one HTTP exchange (status, headers,
//! body) and a clock reading the caller passes in, so the module is tested on
//! canned responses with no network and no sleeping. The I/O lives in
//! [`crate::nim`] (which captures the exchange through curl) and
//! [`crate::teacher`] (which sleeps, rotates and journals).
//!
//! # What is recognised
//!
//! [`classify`] turns a response into a [`RateLimitSignal`] when it is one of:
//!
//! - **HTTP 429**, from any OpenAI-compatible endpoint. NVIDIA's hosted NIM
//!   answers with a problem document (`{"status":429,"title":...,"detail":...}`);
//!   OpenRouter with `{"error":{"code":429,"message":...,"metadata":{...}}}`,
//!   whose `metadata.headers` carries the `X-RateLimit-*` values of the limit
//!   that fired. Both shapes are read.
//! - **An error object inside an HTTP 200.** OpenRouter can report a limit (or
//!   an upstream failure) in a 2xx body; a classifier that read the status
//!   alone would hand that body to the reply parser as if it were an answer.
//! - **HTTP 402**: the account is out of credits. Waiting does not fix it.
//! - **HTTP 502/503/504/529**: the provider is overloaded or the upstream
//!   failed. Short, growing pauses fix it; persistent ones mean rotate.
//!
//! The kind ([`RateLimitKind`]) separates a per-minute limit (wait it out)
//! from a per-day cap (wait only if the reset is near, otherwise move to
//! another model). The message is read first (`free-models-per-day`,
//! `per minute`, …); when it says nothing, the distance to the advertised reset
//! decides: an hour or more away is a daily cap.
//!
//! # Timestamps
//!
//! `Retry-After` is either delta-seconds or an HTTP-date (RFC 9110 §10.2.3);
//! both are parsed. `X-RateLimit-Reset` differs by provider: OpenRouter sends a
//! Unix epoch in *milliseconds*, others send epoch seconds, delta seconds, an
//! RFC 3339 timestamp or a Go-style duration (`1m30s`). [`parse_reset_value`]
//! tells them apart by form and magnitude: a bare number at or above 1e11 is
//! epoch milliseconds (as seconds it would be the year 5138), at or above 1e9
//! epoch seconds (2001 onwards), and anything smaller a delta in seconds.

use std::time::Duration;

use serde::{Deserialize, Serialize};

const MS_PER_SEC: u64 = 1_000;
const MS_PER_DAY: u64 = 86_400_000;
/// A reset at least this far away reads as a daily (not per-minute) limit when
/// the message does not say which it is.
const DAILY_HORIZON_MS: u64 = 3_600_000;

/// One HTTP response as the classifier sees it: final status, the final
/// header block, and the body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HttpResponse {
    /// Final status code (after redirects).
    pub status: u16,
    /// Header names lower-cased, values trimmed, in arrival order.
    pub headers: Vec<(String, String)>,
    /// Response body.
    pub body: String,
}

impl HttpResponse {
    /// Build a response, lower-casing the header names.
    pub fn new(status: u16, headers: Vec<(String, String)>, body: impl Into<String>) -> Self {
        Self {
            status,
            headers: headers
                .into_iter()
                .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
                .collect(),
            body: body.into(),
        }
    }

    /// The last value of header `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        header_in(&self.headers, name)
    }

    /// True for a 2xx status.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

fn header_in<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .rev()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// Parse the header lines of one header block (`Name: value` per line). The
/// status line and anything without a colon are skipped; names are
/// lower-cased.
pub fn parse_header_lines(block: &str) -> Vec<(String, String)> {
    block
        .lines()
        .filter(|line| !line.starts_with("HTTP/"))
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            let name = name.trim();
            if name.is_empty() || name.contains(' ') {
                return None;
            }
            Some((name.to_ascii_lowercase(), value.trim().to_string()))
        })
        .collect()
}

/// What kind of limit a response reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RateLimitKind {
    /// A short-window limit (per minute / per second / upstream throttling).
    PerMinute,
    /// A per-day cap, e.g. OpenRouter's `free-models-per-day`.
    Daily,
    /// The account cannot pay (HTTP 402, insufficient credits).
    Credits,
    /// The provider is overloaded or the upstream failed (502/503/504/529).
    Overloaded,
    /// A 429 that says nothing about its window.
    Unknown,
}

impl RateLimitKind {
    /// Stable kebab-case name, as serialized.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PerMinute => "per-minute",
            Self::Daily => "daily",
            Self::Credits => "credits",
            Self::Overloaded => "overloaded",
            Self::Unknown => "unknown",
        }
    }
}

/// A classified rate-limit (or overload, or credits) response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitSignal {
    /// Which limit fired.
    pub kind: RateLimitKind,
    /// How long the server asked the client to wait, when it said
    /// (`Retry-After`, `retry-after-ms`, a `retry_after` body field or a
    /// "try again in 20s" message).
    pub retry_after: Option<Duration>,
    /// When the limit resets, as Unix epoch milliseconds, when the server said.
    pub reset_at_unix_ms: Option<u64>,
    /// What the decision rests on: status, the rate-limit headers and a short
    /// excerpt of the server's message. Never contains request headers.
    pub evidence: String,
}

/// Quota state advertised on a response (`X-RateLimit-*` and friends).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaSnapshot {
    /// Requests allowed per window.
    pub limit: Option<u64>,
    /// Requests left in the current window.
    pub remaining: Option<u64>,
    /// When the window resets (Unix epoch ms).
    pub reset_at_unix_ms: Option<u64>,
}

/// Read the advertised quota from response headers, if any is present.
/// Recognises `x-ratelimit-{limit,remaining,reset}`, their `-requests`
/// variants and the IETF `ratelimit-*` names.
pub fn quota_from_headers(headers: &[(String, String)], now_unix_ms: u64) -> Option<QuotaSnapshot> {
    let first = |names: &[&str]| names.iter().find_map(|name| header_in(headers, name));
    let limit = first(&[
        "x-ratelimit-limit",
        "x-ratelimit-limit-requests",
        "ratelimit-limit",
    ])
    .and_then(leading_u64);
    let remaining = first(&[
        "x-ratelimit-remaining",
        "x-ratelimit-remaining-requests",
        "ratelimit-remaining",
    ])
    .and_then(leading_u64);
    let reset_at_unix_ms = first(&[
        "x-ratelimit-reset",
        "x-ratelimit-reset-requests",
        "ratelimit-reset",
    ])
    .and_then(|value| parse_reset_value(value, now_unix_ms));
    if limit.is_none() && remaining.is_none() && reset_at_unix_ms.is_none() {
        return None;
    }
    Some(QuotaSnapshot {
        limit,
        remaining,
        reset_at_unix_ms,
    })
}

/// The leading unsigned integer of a header value (`"50"`, `"50, 50;w=60"`).
fn leading_u64(value: &str) -> Option<u64> {
    let token = first_token(value);
    token.parse::<u64>().ok().or_else(|| {
        token
            .parse::<f64>()
            .ok()
            .filter(|n| n.is_finite() && *n >= 0.0)
            .map(|n| n.floor() as u64) // audit-allow: finite and non-negative, checked by the filter
    })
}

/// The part of a header value before the first `,` or `;`.
fn first_token(value: &str) -> &str {
    value.split([',', ';']).next().unwrap_or("").trim()
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Phrases that mark a per-day cap. Read before the credit phrases, because
/// OpenRouter's daily message ("…free-models-per-day. Add 10 credits to
/// unlock 1000 free model requests per day") mentions credits too.
const DAILY_MARKERS: &[&str] = &[
    "per-day",
    "per day",
    "/day",
    "daily",
    "rpd",
    "24 hours",
    "24h",
    "per 24",
];

/// Phrases that mark a short-window limit.
const MINUTE_MARKERS: &[&str] = &[
    "per-min",
    "per min",
    "/min",
    "rpm",
    "tpm",
    "per second",
    "per-second",
    "/sec",
    "rps",
    "temporarily",
    "retry shortly",
    "upstream",
];

/// Phrases that mark an account that cannot pay.
const CREDIT_MARKERS: &[&str] = &[
    "insufficient credit",
    "insufficient_quota",
    "insufficient balance",
    "out of credits",
    "no credits",
    "credit balance",
    "negative balance",
    "payment required",
    "billing",
];

/// Statuses that mean "overloaded / upstream failed".
const OVERLOAD_STATUSES: &[u16] = &[502, 503, 504, 529];

/// The error object a body carries, in whichever of the known shapes.
#[derive(Default)]
struct ErrorBody {
    /// Numeric code, when the body states one.
    code: Option<u16>,
    /// The textual parts (message, code name, type, raw upstream text).
    text: String,
    /// Headers relayed inside the body (OpenRouter `metadata.headers`).
    headers: Vec<(String, String)>,
    /// A `retry_after` field in seconds.
    retry_after_secs: Option<f64>,
}

/// Find an error object in a JSON body: `{"error": {...}}`, `{"error": "..."}`,
/// `{"choices":[{"error": {...}}]}` or a problem document
/// (`{"status": 429, "title": ..., "detail": ...}`).
fn error_body(body: &str) -> Option<ErrorBody> {
    // A body that is not JSON carries no error object; that is an answer to
    // "is there one", not a failure to report.
    let value: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    if let Some(error) = value.get("error") {
        return Some(read_error_value(error));
    }
    let in_choice = value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.iter().find_map(|choice| choice.get("error")));
    if let Some(error) = in_choice {
        return Some(read_error_value(error));
    }
    // A problem document carries `status` plus `title`/`detail`. Requiring
    // the text fields keeps an ordinary object with a `status` key out.
    let status = value.get("status").and_then(json_u16)?;
    let title = value.get("title").and_then(serde_json::Value::as_str);
    let detail = value.get("detail").and_then(serde_json::Value::as_str);
    if title.is_none() && detail.is_none() {
        return None;
    }
    Some(ErrorBody {
        code: Some(status),
        text: [title.unwrap_or(""), detail.unwrap_or("")].join(" "),
        ..ErrorBody::default()
    })
}

fn read_error_value(error: &serde_json::Value) -> ErrorBody {
    if let Some(text) = error.as_str() {
        return ErrorBody {
            text: text.to_string(),
            ..ErrorBody::default()
        };
    }
    let mut out = ErrorBody {
        code: error
            .get("code")
            .and_then(json_u16)
            .or_else(|| error.get("status").and_then(json_u16)),
        ..ErrorBody::default()
    };
    let mut text = Vec::new();
    for key in ["message", "type", "detail", "title"] {
        if let Some(part) = error.get(key).and_then(serde_json::Value::as_str) {
            text.push(part.to_string());
        }
    }
    if let Some(code) = error.get("code").and_then(serde_json::Value::as_str) {
        text.push(code.to_string());
    }
    out.retry_after_secs = error.get("retry_after").and_then(serde_json::Value::as_f64);
    if let Some(metadata) = error.get("metadata") {
        if let Some(raw) = metadata.get("raw") {
            match raw.as_str() {
                Some(raw) => text.push(raw.to_string()),
                None => text.push(raw.to_string()),
            }
        }
        if let Some(headers) = metadata.get("headers").and_then(serde_json::Value::as_object) {
            for (name, value) in headers {
                let value = match value.as_str() {
                    Some(text) => text.to_string(),
                    None => value.to_string(),
                };
                out.headers.push((name.to_ascii_lowercase(), value));
            }
        }
        if out.retry_after_secs.is_none() {
            out.retry_after_secs = metadata
                .get("retry_after")
                .and_then(serde_json::Value::as_f64);
        }
    }
    out.text = text.join(" ");
    out
}

/// A status code from a JSON number or a numeric string.
fn json_u16(value: &serde_json::Value) -> Option<u16> {
    if let Some(n) = value.as_u64() {
        return u16::try_from(n).ok();
    }
    value.as_str().and_then(|s| s.trim().parse::<u16>().ok())
}

fn mentions(text: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| text.contains(marker))
}

/// Classify one response. `None` means it is not a rate-limit, credits or
/// overload signal (a normal answer, an auth failure, a bad request, a plain
/// 500): those are the caller's ordinary success or error paths.
pub fn classify(response: &HttpResponse, now_unix_ms: u64) -> Option<RateLimitSignal> {
    let body = error_body(&response.body);
    let is_signal_code =
        |code: u16| code == 429 || code == 402 || OVERLOAD_STATUSES.contains(&code);
    let text_lower = body
        .as_ref()
        .map(|b| b.text.to_lowercase())
        .unwrap_or_default();
    let code = if is_signal_code(response.status) {
        response.status
    } else {
        // The status alone says nothing: look inside the body (OpenRouter
        // reports some limits and upstream failures in a 2xx).
        match body.as_ref().and_then(|b| b.code) {
            Some(code) if is_signal_code(code) => code,
            _ if body.is_some()
                && (text_lower.contains("rate limit") || text_lower.contains("rate_limit")) =>
            {
                429
            }
            _ => return None,
        }
    };

    let mut headers = response.headers.clone();
    if let Some(b) = &body {
        // Response headers win over relayed ones: they were appended first
        // and `header_in` reads the last match, so relayed ones go first.
        let mut merged = b.headers.clone();
        merged.extend(headers);
        headers = merged;
    }
    let retry_after = header_in(&headers, "retry-after-ms")
        .and_then(|v| first_token(v).parse::<f64>().ok())
        .filter(|ms| ms.is_finite() && *ms >= 0.0)
        .map(|ms| Duration::from_millis(ms.round() as u64)) // audit-allow: finite and non-negative, checked by the filter
        .or_else(|| {
            header_in(&headers, "retry-after").and_then(|v| parse_retry_after(v, now_unix_ms))
        })
        .or_else(|| {
            body.as_ref()
                .and_then(|b| b.retry_after_secs)
                .and_then(|s| Duration::try_from_secs_f64(s).ok())
        })
        .or_else(|| duration_hint_in_text(&text_lower));
    let quota = quota_from_headers(&headers, now_unix_ms);
    let reset_at_unix_ms = quota.and_then(|q| q.reset_at_unix_ms);
    let horizon_ms = retry_after
        .map(duration_ms)
        .or_else(|| reset_at_unix_ms.map(|reset| reset.saturating_sub(now_unix_ms)));

    let kind = if code == 402 {
        RateLimitKind::Credits
    } else if OVERLOAD_STATUSES.contains(&code) {
        RateLimitKind::Overloaded
    } else if mentions(&text_lower, DAILY_MARKERS) {
        RateLimitKind::Daily
    } else if mentions(&text_lower, MINUTE_MARKERS) {
        RateLimitKind::PerMinute
    } else if mentions(&text_lower, CREDIT_MARKERS) {
        RateLimitKind::Credits
    } else {
        match horizon_ms {
            Some(h) if h >= DAILY_HORIZON_MS => RateLimitKind::Daily,
            Some(_) => RateLimitKind::PerMinute,
            None => RateLimitKind::Unknown,
        }
    };

    let mut evidence = format!("HTTP {}", response.status);
    if code != response.status {
        evidence.push_str(&format!(" carrying error code {code}"));
    }
    for name in [
        "retry-after",
        "retry-after-ms",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
    ] {
        if let Some(value) = header_in(&headers, name) {
            evidence.push_str(&format!("; {name}={}", short(value, 40)));
        }
    }
    if let Some(b) = &body {
        if !b.text.trim().is_empty() {
            evidence.push_str(&format!("; message: {}", short(&b.text, 160)));
        }
    }
    Some(RateLimitSignal {
        kind,
        retry_after,
        reset_at_unix_ms,
        evidence,
    })
}

fn duration_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
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

/// A wait stated in prose: "try again in 20s", "retry after 7 seconds",
/// "please retry in 1m30s". Lower-case input.
fn duration_hint_in_text(text_lower: &str) -> Option<Duration> {
    for lead in ["try again in ", "retry after ", "retry in ", "wait "] {
        if let Some(at) = text_lower.find(lead) {
            let rest = &text_lower[at + lead.len()..];
            let candidate: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '.' || *c == ' ')
                .collect();
            // "20 seconds." / "20s" / "1m30s": try the longest prefix that
            // parses, shrinking word by word.
            let words: Vec<&str> = candidate.split_whitespace().collect();
            for take in (1..=words.len().min(4)).rev() {
                let joined: String = words.iter().take(take).copied().collect();
                if let Some(d) = parse_duration_text(&joined) {
                    return Some(d);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Timestamp and duration parsing
// ---------------------------------------------------------------------------

/// `Retry-After`: delta-seconds or an HTTP-date. A date in the past is a zero
/// wait, never a negative one.
pub fn parse_retry_after(value: &str, now_unix_ms: u64) -> Option<Duration> {
    let value = value.trim();
    if let Some(wait) = value
        .parse::<f64>()
        .ok()
        .and_then(|secs| Duration::try_from_secs_f64(secs).ok())
    {
        return Some(wait);
    }
    parse_http_date(value).map(|at| Duration::from_millis(at.saturating_sub(now_unix_ms)))
}

/// A reset value in any of the forms providers use, as Unix epoch ms: RFC
/// 3339, a duration with units (`1m30s`, `250ms`), or a bare number read by
/// magnitude (see the module docs), or an HTTP-date.
pub fn parse_reset_value(value: &str, now_unix_ms: u64) -> Option<u64> {
    let token = first_token(value);
    if token.is_empty() {
        return None;
    }
    if let Some(at) = parse_rfc3339(token) {
        return Some(at);
    }
    if let Some(n) = token.parse::<f64>().ok().filter(|n| n.is_finite()) {
        if n < 0.0 {
            return None;
        }
        return Some(if n >= 1e11 {
            n.round() as u64 // audit-allow: finite, >= 1e11, epoch milliseconds
        } else if n >= 1e9 {
            (n * 1_000.0).round() as u64 // audit-allow: finite, epoch seconds below 1e11
        } else {
            now_unix_ms.saturating_add((n * 1_000.0).round() as u64) // audit-allow: finite, below 1e9 seconds
        });
    }
    if let Some(d) = parse_duration_text(token) {
        return Some(now_unix_ms.saturating_add(duration_ms(d)));
    }
    parse_http_date(value.trim())
}

/// A Go-style duration with units: `2s`, `1m30s`, `250ms`, `1h`, `0.5s`,
/// `20seconds`. Every number needs a unit; spaces are ignored.
pub fn parse_duration_text(text: &str) -> Option<Duration> {
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let compact = compact.trim_end_matches('.');
    if compact.is_empty() {
        return None;
    }
    let mut total_ms = 0f64;
    let mut rest = compact;
    while !rest.is_empty() {
        let num_len = rest
            .char_indices()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == '.'))
            .map_or(rest.len(), |(i, _)| i);
        if num_len == 0 {
            return None;
        }
        let number: f64 = rest[..num_len].parse().ok()?;
        rest = &rest[num_len..];
        let unit_len = rest
            .char_indices()
            .find(|(_, c)| !c.is_ascii_alphabetic())
            .map_or(rest.len(), |(i, _)| i);
        let unit = &rest[..unit_len];
        rest = &rest[unit_len..];
        let scale = match unit {
            "ms" | "msec" | "millisecond" | "milliseconds" => 1.0,
            "s" | "sec" | "secs" | "second" | "seconds" => 1_000.0,
            "m" | "min" | "mins" | "minute" | "minutes" => 60_000.0,
            "h" | "hr" | "hrs" | "hour" | "hours" => 3_600_000.0,
            _ => return None,
        };
        total_ms += number * scale;
    }
    if !total_ms.is_finite() || total_ms < 0.0 {
        return None;
    }
    Some(Duration::from_millis(total_ms.round() as u64)) // audit-allow: finite and non-negative, checked above
}

const MONTHS: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];

fn month_number(token: &str) -> Option<u32> {
    let lower = token.to_ascii_lowercase();
    let head = lower.get(..3)?;
    MONTHS
        .iter()
        .position(|m| *m == head)
        .and_then(|i| u32::try_from(i + 1).ok())
}

/// An HTTP-date in any of the three forms RFC 9110 requires a recipient to
/// accept (IMF-fixdate, RFC 850, asctime), as Unix epoch ms. Always UTC.
pub fn parse_http_date(text: &str) -> Option<u64> {
    let mut day = None;
    let mut month = None;
    let mut year = None;
    let mut time = None;
    for token in text
        .split([' ', ',', '-'])
        .filter(|token| !token.is_empty())
    {
        if token.contains(':') {
            time = Some(parse_hms(token)?);
        } else if token.chars().all(|c| c.is_ascii_digit()) {
            let n: u32 = token.parse().ok()?;
            if token.len() == 4 {
                year = Some(n);
            } else if day.is_none() {
                day = Some(n);
            } else if year.is_none() {
                // RFC 850's two-digit year.
                year = Some(if n >= 70 { 1900 + n } else { 2000 + n });
            }
        } else if month.is_none() {
            // Weekday names and the `GMT` zone fall through as non-months.
            month = month_number(token).filter(|_| token.len() == 3 || is_month_name(token));
        }
    }
    let (hour, minute, second) = time?;
    civil_to_unix_ms(i64::from(year?), month?, day?, hour, minute, second, 0)
}

fn is_month_name(token: &str) -> bool {
    const FULL: [&str; 12] = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    FULL.contains(&token.to_ascii_lowercase().as_str())
}

fn parse_hms(token: &str) -> Option<(u32, u32, u32)> {
    let mut parts = token.split(':');
    let hour: u32 = parts.next()?.parse().ok()?;
    let minute: u32 = parts.next()?.parse().ok()?;
    let second: u32 = parts.next().map_or(Some(0), |s| s.parse().ok())?;
    if parts.next().is_some() || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    Some((hour, minute, second))
}

/// An RFC 3339 / ISO 8601 timestamp (`2026-11-21T00:00:00Z`, fractional
/// seconds and `±HH:MM` offsets accepted) as Unix epoch ms.
pub fn parse_rfc3339(text: &str) -> Option<u64> {
    let (date, time) = text.split_once(['T', 't', ' '])?;
    let mut date_parts = date.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    if date_parts.next().is_some() {
        return None;
    }
    let (clock, offset_ms): (&str, i64) = if let Some(clock) = time.strip_suffix(['Z', 'z']) {
        (clock, 0)
    } else if let Some(at) = time.rfind(['+', '-']) {
        let (clock, offset) = time.split_at(at);
        let sign = if offset.starts_with('-') { -1 } else { 1 };
        let (oh, om) = offset.get(1..)?.split_once(':')?;
        let oh: i64 = oh.parse().ok()?;
        let om: i64 = om.parse().ok()?;
        (clock, sign * (oh * 3_600_000 + om * 60_000))
    } else {
        return None;
    };
    let (whole, frac) = match clock.split_once('.') {
        Some((whole, frac)) => (whole, frac),
        None => (clock, ""),
    };
    let (hour, minute, second) = parse_hms(whole)?;
    let millis: u32 = if frac.is_empty() {
        0
    } else {
        let digits: String = frac.chars().take(3).collect();
        let padded = format!("{digits:0<3}");
        padded.parse().ok()?
    };
    let utc = civil_to_unix_ms(year, month, day, hour, minute, second, millis)?;
    let shifted = i128::from(utc) - i128::from(offset_ms);
    u64::try_from(shifted).ok()
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from((month + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of [`days_from_civil`]: `(year, month, day)`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (
        year,
        u32::try_from(m).unwrap_or(1),
        u32::try_from(d).unwrap_or(1),
    )
}

fn civil_to_unix_ms(
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    millis: u32,
) -> Option<u64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || year < 1970 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let secs = days * 86_400 + i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second);
    let ms = i128::from(secs) * 1_000 + i128::from(millis);
    u64::try_from(ms).ok()
}

/// Unix epoch ms as an RFC 3339 UTC timestamp to the second
/// (`2026-11-20T13:45:00Z`), for log lines and the journal.
pub fn format_unix_ms(ms: u64) -> String {
    let secs = ms / MS_PER_SEC;
    let days = i64::try_from(secs / 86_400).unwrap_or(i64::MAX / 2);
    let rem = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        (rem % 3_600) / 60,
        rem % 60
    )
}

// ---------------------------------------------------------------------------
// Pause scheduling
// ---------------------------------------------------------------------------

fn default_max_pause_secs() -> u64 {
    900
}
fn default_max_pause_daily_secs() -> u64 {
    3_600
}
fn default_max_idle_secs() -> u64 {
    86_400
}
fn default_jitter_max_ms() -> u64 {
    2_000
}
fn default_fallback_pause_secs() -> u64 {
    60
}
fn default_overload_base_secs() -> u64 {
    5
}
fn default_low_remaining() -> u64 {
    2
}
fn default_max_consecutive_failures() -> u32 {
    4
}
fn default_max_waits_per_prompt() -> u32 {
    24
}

/// How a teacher run pauses. Every field carries `serde(default)`, so a
/// config written before a field existed still parses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PausePolicy {
    /// Longest pause for a per-minute, unknown or overload limit. A limit
    /// whose reset is further away than this benches the model instead
    /// (rotate, or stop if nothing else is usable). Default 15 min.
    #[serde(default = "default_max_pause_secs")]
    pub max_pause_secs: u64,
    /// A daily-capped model is waited for only when its reset is at most this
    /// far away; otherwise the pool rotates to another model. Default 1 h.
    #[serde(default = "default_max_pause_daily_secs")]
    pub max_pause_daily_secs: u64,
    /// When every model in the pool is paused, the run sleeps until the
    /// earliest reset if it is at most this far away, and otherwise stops
    /// with its journal saved (rerun the same command to resume). Default
    /// 24 h, so an unattended run waits out a daily cap; `0` stops at once.
    #[serde(default = "default_max_idle_secs")]
    pub max_idle_secs: u64,
    /// Upper bound of the random jitter added to every pause, so a pool of
    /// clients does not resume in lockstep. Default 2 s.
    #[serde(default = "default_jitter_max_ms")]
    pub jitter_max_ms: u64,
    /// Pause for a per-minute or unknown 429 that names no reset. Default 60 s.
    #[serde(default = "default_fallback_pause_secs")]
    pub fallback_pause_secs: u64,
    /// First overload pause when the server names none; doubles per
    /// consecutive overload (capped by `max_pause_secs`). Default 5 s.
    #[serde(default = "default_overload_base_secs")]
    pub overload_base_secs: u64,
    /// Start spacing requests out when the advertised `Remaining` is at or
    /// below this, spreading what is left over the rest of the window.
    #[serde(default = "default_low_remaining")]
    pub low_remaining: u64,
    /// Minimum gap between two requests, whatever the headers say. `0` is
    /// none; OpenRouter's free tier allows about 20 requests a minute, which
    /// `3000` stays under.
    #[serde(default)]
    pub min_interval_ms: u64,
    /// Consecutive overloads or failures after which a model is benched and
    /// the pool moves on ("keeps failing").
    #[serde(default = "default_max_consecutive_failures")]
    pub max_consecutive_failures: u32,
    /// Rate-limit pauses one prompt may sit through before it is counted as
    /// failed, so a server that answers 429 forever cannot hold a run.
    #[serde(default = "default_max_waits_per_prompt")]
    pub max_waits_per_prompt: u32,
}

impl Default for PausePolicy {
    fn default() -> Self {
        Self {
            max_pause_secs: default_max_pause_secs(),
            max_pause_daily_secs: default_max_pause_daily_secs(),
            max_idle_secs: default_max_idle_secs(),
            jitter_max_ms: default_jitter_max_ms(),
            fallback_pause_secs: default_fallback_pause_secs(),
            overload_base_secs: default_overload_base_secs(),
            low_remaining: default_low_remaining(),
            min_interval_ms: 0,
            max_consecutive_failures: default_max_consecutive_failures(),
            max_waits_per_prompt: default_max_waits_per_prompt(),
        }
    }
}

impl PausePolicy {
    /// [`Self::max_pause_secs`] in ms.
    pub fn max_pause_ms(&self) -> u64 {
        self.max_pause_secs.saturating_mul(MS_PER_SEC)
    }

    /// [`Self::max_idle_secs`] in ms.
    pub fn max_idle_ms(&self) -> u64 {
        self.max_idle_secs.saturating_mul(MS_PER_SEC)
    }
}

/// What to do with a model after a signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PauseVerdict {
    /// Pause this model and wait for it (the reset is near).
    Wait,
    /// Bench this model until its reset and use another one meanwhile.
    Rotate,
    /// Stop using this model for the rest of the run (credits, gone).
    Disable,
}

/// A scheduled pause.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PausePlan {
    /// The limit that caused it.
    pub kind: RateLimitKind,
    /// Wait, rotate or disable.
    pub verdict: PauseVerdict,
    /// When the model is usable again (Unix ms); `u64::MAX` for `Disable`.
    pub until_unix_ms: u64,
    /// `until_unix_ms - now`, jitter included; 0 for `Disable`.
    pub wait_ms: u64,
    /// The jitter part of `wait_ms`.
    pub jitter_ms: u64,
    /// Human-readable reason for the log line and the journal.
    pub reason: String,
}

/// Jitter in `[0, max_ms]` from a unit draw. Out-of-range or non-finite
/// draws are clamped, so the bound holds whatever the source returns.
pub fn jitter_ms(max_ms: u64, unit: f64) -> u64 {
    let u = if unit.is_finite() {
        unit.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let raw = (max_ms as f64 * u).floor(); // audit-allow: max_ms is a configured pause, far below 2^53
    (raw as u64).min(max_ms) // audit-allow: finite and non-negative, clamped to max_ms
}

/// Milliseconds until the next 00:00 UTC: OpenRouter's free-tier daily caps
/// reset then, and it is the fallback when a daily cap names no reset.
pub fn ms_until_next_utc_midnight(now_unix_ms: u64) -> u64 {
    MS_PER_DAY - now_unix_ms % MS_PER_DAY
}

/// Schedule the pause for `signal`. `consecutive` is how many signals or
/// failures this model has had in a row *before* this one; `jitter_unit` a
/// draw in `[0, 1)`. The result never waits a negative time, a `Wait` never
/// exceeds its bound (`max_pause_secs`, or `max_pause_daily_secs` for a
/// daily cap), and the jitter is at most `jitter_max_ms`.
pub fn plan_pause(
    signal: &RateLimitSignal,
    now_unix_ms: u64,
    consecutive: u32,
    policy: &PausePolicy,
    jitter_unit: f64,
) -> PausePlan {
    let jitter = jitter_ms(policy.jitter_max_ms, jitter_unit);
    let hint_ms = signal.retry_after.map(duration_ms).or_else(|| {
        signal
            .reset_at_unix_ms
            .map(|reset| reset.saturating_sub(now_unix_ms))
    });
    let max_pause = policy.max_pause_ms();
    let plan = |verdict: PauseVerdict, wait_ms: u64, jit: u64, reason: String| PausePlan {
        kind: signal.kind,
        verdict,
        until_unix_ms: now_unix_ms.saturating_add(wait_ms),
        wait_ms,
        jitter_ms: jit,
        reason,
    };
    match signal.kind {
        RateLimitKind::Credits => PausePlan {
            kind: signal.kind,
            verdict: PauseVerdict::Disable,
            until_unix_ms: u64::MAX,
            wait_ms: 0,
            jitter_ms: 0,
            reason: "out of credits: waiting does not fix it; add credits or use another model"
                .to_string(),
        },
        RateLimitKind::Daily => {
            let (base, source) = match hint_ms {
                Some(ms) => (ms, "the server's reset"),
                None => (
                    ms_until_next_utc_midnight(now_unix_ms),
                    "the next 00:00 UTC (no reset given)",
                ),
            };
            let total = base.saturating_add(jitter);
            let bound = policy.max_pause_daily_secs.saturating_mul(MS_PER_SEC);
            if total <= bound {
                plan(
                    PauseVerdict::Wait,
                    total,
                    jitter,
                    format!("daily cap; waiting for {source}, within max_pause_daily"),
                )
            } else {
                plan(
                    PauseVerdict::Rotate,
                    total,
                    jitter,
                    format!("daily cap; benched until {source}, beyond max_pause_daily"),
                )
            }
        }
        RateLimitKind::PerMinute | RateLimitKind::Unknown => {
            let base = hint_ms
                .unwrap_or_else(|| policy.fallback_pause_secs.saturating_mul(MS_PER_SEC));
            if base <= max_pause {
                let total = base.saturating_add(jitter).min(max_pause);
                let jitter = total - base;
                plan(
                    PauseVerdict::Wait,
                    total,
                    jitter,
                    format!("{} limit; waiting for it to reset", signal.kind.as_str()),
                )
            } else {
                plan(
                    PauseVerdict::Rotate,
                    base.saturating_add(jitter),
                    jitter,
                    format!(
                        "{} limit resets beyond max_pause; benched until then",
                        signal.kind.as_str()
                    ),
                )
            }
        }
        RateLimitKind::Overloaded => {
            let exponent = consecutive.min(16);
            let base = hint_ms.unwrap_or_else(|| {
                policy
                    .overload_base_secs
                    .saturating_mul(MS_PER_SEC)
                    .saturating_mul(1u64 << exponent)
            });
            if consecutive.saturating_add(1) >= policy.max_consecutive_failures.max(1) {
                plan(
                    PauseVerdict::Rotate,
                    max_pause,
                    0,
                    format!(
                        "overloaded {} time(s) in a row; benched for max_pause",
                        consecutive.saturating_add(1)
                    ),
                )
            } else {
                let base = base.min(max_pause);
                let total = base.saturating_add(jitter).min(max_pause);
                plan(
                    PauseVerdict::Wait,
                    total,
                    total - base,
                    "provider overloaded; backing off".to_string(),
                )
            }
        }
    }
}

/// How long to hold the next request so the advertised quota lasts until its
/// reset: the whole rest of the window when `remaining` is 0, the window
/// divided by `remaining + 1` when it is at or below
/// [`PausePolicy::low_remaining`], and nothing otherwise. Capped at
/// `max_pause_secs`; `None` when the quota says nothing usable.
pub fn pacing_delay(quota: &QuotaSnapshot, now_unix_ms: u64, policy: &PausePolicy) -> Option<Duration> {
    let remaining = quota.remaining?;
    let reset = quota.reset_at_unix_ms?;
    let window = reset.checked_sub(now_unix_ms).filter(|w| *w > 0)?;
    let delay = if remaining == 0 {
        window
    } else if remaining <= policy.low_remaining {
        window / remaining.saturating_add(1)
    } else {
        return None;
    };
    Some(Duration::from_millis(delay.min(policy.max_pause_ms())))
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

    /// 2026-11-20T12:00:00Z.
    const NOW: u64 = 1_795_176_000_000;

    fn response(status: u16, headers: &[(&str, &str)], body: &str) -> HttpResponse {
        HttpResponse::new(
            status,
            headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            body,
        )
    }

    fn policy() -> PausePolicy {
        PausePolicy::default()
    }

    #[test]
    fn now_constant_is_what_it_says() {
        assert_eq!(format_unix_ms(NOW), "2026-11-20T12:00:00Z");
        assert_eq!(parse_rfc3339("2026-11-20T12:00:00Z"), Some(NOW));
    }

    #[test]
    fn a_429_with_retry_after_seconds_is_a_short_limit() {
        let signal = classify(&response(429, &[("Retry-After", "30")], ""), NOW).unwrap();
        assert_eq!(signal.kind, RateLimitKind::PerMinute);
        assert_eq!(signal.retry_after, Some(Duration::from_secs(30)));
        assert!(signal.evidence.contains("HTTP 429"), "{}", signal.evidence);
        assert!(signal.evidence.contains("retry-after=30"), "{}", signal.evidence);
    }

    #[test]
    fn a_429_with_an_http_date_retry_after_is_measured_from_now() {
        // 90 s after NOW, in all three RFC 9110 date forms.
        for date in [
            "Fri, 20 Nov 2026 12:01:30 GMT",
            "Friday, 20-Nov-26 12:01:30 GMT",
            "Fri Nov 20 12:01:30 2026",
        ] {
            let signal = classify(&response(429, &[("retry-after", date)], ""), NOW).unwrap();
            assert_eq!(signal.retry_after, Some(Duration::from_secs(90)), "{date}");
            assert_eq!(signal.kind, RateLimitKind::PerMinute, "{date}");
        }
        // A date in the past is a zero wait, never a negative one.
        let past = classify(
            &response(429, &[("Retry-After", "Fri, 20 Nov 2026 11:00:00 GMT")], ""),
            NOW,
        )
        .unwrap();
        assert_eq!(past.retry_after, Some(Duration::ZERO));
    }

    #[test]
    fn reset_in_milliseconds_and_seconds_agree() {
        let reset_ms = NOW + 45_000;
        let as_ms = reset_ms.to_string();
        let as_s = (reset_ms / 1_000).to_string();
        let as_delta = "45";
        let as_duration = "45s";
        let as_rfc3339 = "2026-11-20T12:00:45Z";
        for value in [as_ms.as_str(), as_s.as_str(), as_delta, as_duration, as_rfc3339] {
            let signal = classify(
                &response(
                    429,
                    &[("X-RateLimit-Limit", "20"), ("X-RateLimit-Remaining", "0"), ("X-RateLimit-Reset", value)],
                    "",
                ),
                NOW,
            )
            .unwrap();
            assert_eq!(signal.reset_at_unix_ms, Some(reset_ms), "value {value}");
            assert_eq!(signal.kind, RateLimitKind::PerMinute, "value {value}");
        }
        // Fractional epoch seconds round to the millisecond.
        assert_eq!(parse_reset_value("1795176045.5", NOW), Some(NOW + 45_500));
    }

    #[test]
    fn openrouter_error_json_in_a_200_is_a_limit_not_an_answer() {
        let body = r#"{"error":{"code":429,"message":"Rate limit exceeded: free-models-per-min.","metadata":{"headers":{"X-RateLimit-Limit":"20","X-RateLimit-Remaining":"0","X-RateLimit-Reset":"1795176060000"},"provider_name":null}},"user_id":"u"}"#;
        let signal = classify(&response(200, &[], body), NOW).unwrap();
        assert_eq!(signal.kind, RateLimitKind::PerMinute);
        assert_eq!(signal.reset_at_unix_ms, Some(NOW + 60_000));
        assert!(signal.evidence.contains("HTTP 200 carrying error code 429"), "{}", signal.evidence);
        assert!(signal.evidence.contains("free-models-per-min"), "{}", signal.evidence);
        // An upstream failure reported in a 200 is an overload.
        let upstream = r#"{"choices":[{"finish_reason":"error","error":{"code":502,"message":"Provider returned error"}}]}"#;
        let signal = classify(&response(200, &[], upstream), NOW).unwrap();
        assert_eq!(signal.kind, RateLimitKind::Overloaded);
        // An ordinary answer is not a signal.
        let ok = r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}]}"#;
        assert!(classify(&response(200, &[], ok), NOW).is_none());
    }

    #[test]
    fn daily_caps_are_told_apart_from_per_minute_limits() {
        let midnight = NOW + 12 * 3_600_000;
        let daily = format!(
            r#"{{"error":{{"code":429,"message":"Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day","metadata":{{"headers":{{"X-RateLimit-Limit":"50","X-RateLimit-Remaining":"0","X-RateLimit-Reset":"{midnight}"}}}}}}}}"#
        );
        let signal = classify(&response(429, &[], &daily), NOW).unwrap();
        assert_eq!(signal.kind, RateLimitKind::Daily, "credits mention must not win");
        assert_eq!(signal.reset_at_unix_ms, Some(midnight));

        let minute = r#"{"error":{"code":429,"message":"Rate limit exceeded: free-models-per-min."}}"#;
        assert_eq!(
            classify(&response(429, &[], minute), NOW).unwrap().kind,
            RateLimitKind::PerMinute
        );
        // No words at all: the reset distance decides.
        let far = (NOW + 5 * 3_600_000).to_string();
        let near = (NOW + 30_000).to_string();
        assert_eq!(
            classify(&response(429, &[("x-ratelimit-reset", &far)], "{}"), NOW).unwrap().kind,
            RateLimitKind::Daily
        );
        assert_eq!(
            classify(&response(429, &[("x-ratelimit-reset", &near)], "{}"), NOW).unwrap().kind,
            RateLimitKind::PerMinute
        );
        assert_eq!(
            classify(&response(429, &[], ""), NOW).unwrap().kind,
            RateLimitKind::Unknown
        );
    }

    #[test]
    fn a_402_is_credits() {
        let body = r#"{"error":{"code":402,"message":"Insufficient credits. Add more using https://openrouter.ai/credits"}}"#;
        let signal = classify(&response(402, &[], body), NOW).unwrap();
        assert_eq!(signal.kind, RateLimitKind::Credits);
        assert_eq!(plan_pause(&signal, NOW, 0, &policy(), 0.5).verdict, PauseVerdict::Disable);
    }

    #[test]
    fn a_503_and_a_502_are_overloads() {
        for status in [502, 503, 504, 529] {
            let signal = classify(&response(status, &[], "<html>busy</html>"), NOW).unwrap();
            assert_eq!(signal.kind, RateLimitKind::Overloaded, "status {status}");
        }
        let with_hint = classify(&response(503, &[("Retry-After", "7")], ""), NOW).unwrap();
        assert_eq!(with_hint.retry_after, Some(Duration::from_secs(7)));
    }

    #[test]
    fn nim_problem_documents_are_read() {
        let body = r#"{"type":"about:blank","status":429,"title":"Too Many Requests","detail":"Rate limit exceeded, please retry in 20 seconds."}"#;
        let signal = classify(&response(429, &[], body), NOW).unwrap();
        assert_eq!(signal.retry_after, Some(Duration::from_secs(20)));
        assert_eq!(signal.kind, RateLimitKind::PerMinute);
        assert!(signal.evidence.contains("Too Many Requests"), "{}", signal.evidence);
        // The same document in a 200 is still a limit.
        assert!(classify(&response(200, &[], body), NOW).is_some());
    }

    #[test]
    fn ordinary_failures_are_not_signals() {
        for (status, body) in [
            (401, r#"{"error":"bad key"}"#),
            (400, r#"{"error":{"code":400,"message":"model is not a valid model ID"}}"#),
            (404, ""),
            (500, r#"{"error":"boom"}"#),
            (200, "not json"),
        ] {
            assert!(classify(&response(status, &[], body), NOW).is_none(), "{status} {body}");
        }
        // A status document without title/detail is not a problem document.
        assert!(classify(&response(200, &[], r#"{"status":429}"#), NOW).is_none());
    }

    #[test]
    fn header_blocks_parse_case_insensitively() {
        let block = "HTTP/2 429\r\nRetry-After: 12\r\nX-RateLimit-Remaining: 0\r\nbad line\r\n";
        let headers = parse_header_lines(block);
        let r = HttpResponse::new(429, headers, "");
        assert_eq!(r.header("retry-after"), Some("12"));
        assert_eq!(r.header("X-RATELIMIT-REMAINING"), Some("0"));
        assert_eq!(r.headers.len(), 2);
    }

    #[test]
    fn retry_after_ms_header_wins() {
        let signal = classify(
            &response(429, &[("retry-after-ms", "1500"), ("retry-after", "9")], ""),
            NOW,
        )
        .unwrap();
        assert_eq!(signal.retry_after, Some(Duration::from_millis(1_500)));
    }

    #[test]
    fn durations_and_dates_parse() {
        assert_eq!(parse_duration_text("1m30s"), Some(Duration::from_secs(90)));
        assert_eq!(parse_duration_text("250ms"), Some(Duration::from_millis(250)));
        assert_eq!(parse_duration_text("0.5s"), Some(Duration::from_millis(500)));
        assert_eq!(parse_duration_text("20 seconds"), Some(Duration::from_secs(20)));
        assert_eq!(parse_duration_text("12"), None, "a number needs a unit");
        assert_eq!(parse_duration_text("5 parsecs"), None);
        assert_eq!(parse_http_date("not a date"), None);
        assert_eq!(parse_http_date("Fri, 31 Foo 2026 12:00:00 GMT"), None);
        assert_eq!(
            parse_rfc3339("2026-11-20T14:00:00+02:00"),
            Some(NOW),
            "offsets are applied"
        );
        assert_eq!(parse_rfc3339("2026-11-20T12:00:00.250Z"), Some(NOW + 250));
        for days in [-1, 0, 1, 365, 10_957, 20_777, 100_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "round trip {days}");
        }
    }

    #[test]
    fn pauses_are_never_negative_and_always_bounded() {
        let p = policy();
        // Reset already passed: zero base, jitter only.
        let past = RateLimitSignal {
            kind: RateLimitKind::PerMinute,
            retry_after: None,
            reset_at_unix_ms: Some(NOW - 10_000),
            evidence: String::new(),
        };
        let plan = plan_pause(&past, NOW, 0, &p, 0.25);
        assert_eq!(plan.verdict, PauseVerdict::Wait);
        assert_eq!(plan.wait_ms, 500);
        assert_eq!(plan.until_unix_ms, NOW + 500);

        // A per-minute limit just under the bound is clamped to it, jitter included.
        let near_bound = RateLimitSignal {
            retry_after: Some(Duration::from_millis(p.max_pause_ms() - 100)),
            ..past.clone()
        };
        let plan = plan_pause(&near_bound, NOW, 0, &p, 0.99);
        assert_eq!(plan.verdict, PauseVerdict::Wait);
        assert!(plan.wait_ms <= p.max_pause_ms());

        // Beyond the bound: bench until the real reset, never a shorter
        // pause that would just earn another 429.
        let far = RateLimitSignal {
            retry_after: Some(Duration::from_secs(3 * 3_600)),
            ..past.clone()
        };
        let plan = plan_pause(&far, NOW, 0, &p, 0.0);
        assert_eq!(plan.verdict, PauseVerdict::Rotate);
        assert_eq!(plan.until_unix_ms, NOW + 3 * 3_600_000);

        // No hint at all: the fallback pause.
        let unknown = RateLimitSignal {
            kind: RateLimitKind::Unknown,
            retry_after: None,
            reset_at_unix_ms: None,
            evidence: String::new(),
        };
        let plan = plan_pause(&unknown, NOW, 0, &p, 0.0);
        assert_eq!(plan.wait_ms, 60_000);
    }

    #[test]
    fn jitter_stays_within_bounds() {
        assert_eq!(jitter_ms(2_000, 0.0), 0);
        assert_eq!(jitter_ms(2_000, 0.5), 1_000);
        assert!(jitter_ms(2_000, 0.999_999) < 2_000);
        assert_eq!(jitter_ms(2_000, 1.0), 2_000);
        assert_eq!(jitter_ms(2_000, 7.0), 2_000, "clamped above");
        assert_eq!(jitter_ms(2_000, -3.0), 0, "clamped below");
        assert_eq!(jitter_ms(2_000, f64::NAN), 0, "non-finite draws are zero");
        assert_eq!(jitter_ms(0, 0.7), 0);
        for i in 0..100 {
            let u = f64::from(i) / 100.0;
            let signal = RateLimitSignal {
                kind: RateLimitKind::PerMinute,
                retry_after: Some(Duration::from_secs(10)),
                reset_at_unix_ms: None,
                evidence: String::new(),
            };
            let plan = plan_pause(&signal, NOW, 0, &policy(), u);
            assert!(plan.jitter_ms <= policy().jitter_max_ms);
            assert_eq!(plan.wait_ms, 10_000 + plan.jitter_ms);
        }
    }

    #[test]
    fn daily_caps_wait_when_near_and_rotate_when_far() {
        let p = policy(); // max_pause_daily 1 h
        let near = RateLimitSignal {
            kind: RateLimitKind::Daily,
            retry_after: None,
            reset_at_unix_ms: Some(NOW + 20 * 60_000),
            evidence: String::new(),
        };
        let plan = plan_pause(&near, NOW, 0, &p, 0.0);
        assert_eq!(plan.verdict, PauseVerdict::Wait);
        assert_eq!(plan.until_unix_ms, NOW + 20 * 60_000);

        let far = RateLimitSignal {
            reset_at_unix_ms: Some(NOW + 10 * 3_600_000),
            ..near.clone()
        };
        let plan = plan_pause(&far, NOW, 0, &p, 0.0);
        assert_eq!(plan.verdict, PauseVerdict::Rotate);
        assert_eq!(plan.until_unix_ms, NOW + 10 * 3_600_000);

        // No reset given: next midnight UTC (NOW is noon).
        let blind = RateLimitSignal {
            reset_at_unix_ms: None,
            ..near
        };
        let plan = plan_pause(&blind, NOW, 0, &p, 0.0);
        assert_eq!(plan.until_unix_ms, NOW + 12 * 3_600_000);
        assert_eq!(format_unix_ms(plan.until_unix_ms), "2026-11-21T00:00:00Z");
    }

    #[test]
    fn overloads_back_off_then_rotate() {
        let p = policy(); // base 5 s, max 4 consecutive
        let signal = RateLimitSignal {
            kind: RateLimitKind::Overloaded,
            retry_after: None,
            reset_at_unix_ms: None,
            evidence: String::new(),
        };
        let waits: Vec<u64> = (0..3)
            .map(|k| plan_pause(&signal, NOW, k, &p, 0.0).wait_ms)
            .collect();
        assert_eq!(waits, vec![5_000, 10_000, 20_000]);
        let fourth = plan_pause(&signal, NOW, 3, &p, 0.0);
        assert_eq!(fourth.verdict, PauseVerdict::Rotate);
        assert_eq!(fourth.wait_ms, p.max_pause_ms());
        // A huge exponent still cannot exceed the bound.
        let tight = PausePolicy {
            max_consecutive_failures: 100,
            ..p
        };
        assert!(plan_pause(&signal, NOW, 60, &tight, 1.0).wait_ms <= tight.max_pause_ms());
    }

    #[test]
    fn pacing_spreads_the_remaining_quota() {
        let p = policy();
        let quota = |remaining| QuotaSnapshot {
            limit: Some(20),
            remaining: Some(remaining),
            reset_at_unix_ms: Some(NOW + 30_000),
        };
        assert_eq!(pacing_delay(&quota(0), NOW, &p), Some(Duration::from_secs(30)));
        assert_eq!(pacing_delay(&quota(2), NOW, &p), Some(Duration::from_secs(10)));
        assert_eq!(pacing_delay(&quota(1), NOW, &p), Some(Duration::from_secs(15)));
        assert_eq!(pacing_delay(&quota(15), NOW, &p), None);
        let expired = QuotaSnapshot {
            reset_at_unix_ms: Some(NOW - 1),
            ..quota(0)
        };
        assert_eq!(pacing_delay(&expired, NOW, &p), None);
        let headers = vec![
            ("x-ratelimit-limit".to_string(), "20".to_string()),
            ("x-ratelimit-remaining".to_string(), "1".to_string()),
            ("x-ratelimit-reset".to_string(), (NOW + 30_000).to_string()),
        ];
        assert_eq!(quota_from_headers(&headers, NOW), Some(quota(1)));
        assert_eq!(quota_from_headers(&[], NOW), None);
    }
}
