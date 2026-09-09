//! Experiment protocol (roadmap Phase 28; issue #1 §19): what a measurement
//! must carry to be reproducible, and the statistics it is reported with.
//!
//! A number without its environment, configuration, seeds and raw trials is
//! an anecdote. A [`Record`] carries all of them, is appended to a JSONL file
//! that is **never truncated** ([`RunLog::append`]), and summarizes its trials
//! with a mean, a median, a sample standard deviation and a 95% confidence
//! interval from the t distribution -- the interval a small CPU benchmark
//! actually justifies, not the normal one it does not.
//!
//! Warm-up trials are recorded and flagged rather than dropped: a raw
//! measurement that was taken is part of the record whether or not it enters
//! the summary.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

use crate::checkpoint::BuildInfo;

/// JSON has no NaN. A statistic that does not exist (the standard deviation
/// of one sample) is written as `null` and read back as NaN, so a record
/// round-trips instead of failing to parse.
mod nan_null {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, s: S) -> Result<S::Ok, S::Error> {
        if value.is_finite() {
            s.serialize_f64(*value)
        } else {
            s.serialize_none()
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
}

/// The machine and build a measurement was taken on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    pub hostname: String,
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub os_name: String,
    pub os_release: String,
    pub build: BuildInfo,
}

impl Environment {
    /// Read from `/proc` and `/etc` where available; fields fall back to
    /// `unknown` rather than failing, since a record with gaps still beats
    /// none.
    pub fn capture() -> Self {
        let read = |path: &str| std::fs::read_to_string(path).ok();
        let cpu_model = read("/proc/cpuinfo")
            .and_then(|text| {
                text.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split_once(':'))
                    .map(|(_, v)| v.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".into());
        let os_name = read("/etc/os-release")
            .and_then(|text| {
                text.lines()
                    .find(|l| l.starts_with("PRETTY_NAME="))
                    .map(|l| l.trim_start_matches("PRETTY_NAME=").trim_matches('"').to_string())
            })
            .unwrap_or_else(|| std::env::consts::OS.to_string());
        Self {
            hostname: read("/etc/hostname")
                .map(|h| h.trim().to_string())
                .or_else(|| std::env::var("HOSTNAME").ok())
                .unwrap_or_else(|| "unknown".into()),
            cpu_model,
            logical_cpus: std::thread::available_parallelism().map_or(0, |n| n.get()),
            os_name,
            os_release: read("/proc/sys/kernel/osrelease")
                .map(|r| r.trim().to_string())
                .unwrap_or_else(|| "unknown".into()),
            build: BuildInfo::current(),
        }
    }
}

/// One measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    #[serde(with = "nan_null")]
    pub value: f64,
    /// Taken to warm caches and allocators; kept in the record, left out of
    /// the summary.
    pub warmup: bool,
}

/// Descriptive statistics of the measured (non-warm-up) trials.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    /// Sample standard deviation (`n - 1` in the denominator); NaN for `n < 2`
    /// (written as `null`).
    #[serde(with = "nan_null")]
    pub std: f64,
    pub min: f64,
    pub max: f64,
    /// `t_{0.975, n-1} * std / sqrt(n)`; NaN for `n < 2` (written as `null`).
    #[serde(with = "nan_null")]
    pub ci95_half_width: f64,
}

impl Summary {
    /// `None` for an empty sample.
    pub fn of(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let std = if n >= 2 {
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        } else {
            f64::NAN
        };
        let ci95_half_width = if n >= 2 { t_quantile_975(n - 1) * std / (n as f64).sqrt() } else { f64::NAN };
        Some(Self {
            n,
            mean,
            median: median(values),
            std,
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ci95_half_width,
        })
    }

    /// `(low, high)` of the 95% interval.
    pub fn ci95(&self) -> (f64, f64) {
        (self.mean - self.ci95_half_width, self.mean + self.ci95_half_width)
    }

    /// Whether the two intervals overlap. Not a significance test -- a
    /// non-overlap is strong evidence of a difference, an overlap is not
    /// evidence of none -- but it is the honest one-line reading.
    pub fn overlaps(&self, other: &Summary) -> bool {
        let (a_lo, a_hi) = self.ci95();
        let (b_lo, b_hi) = other.ci95();
        if [a_lo, a_hi, b_lo, b_hi].iter().any(|v| v.is_nan()) {
            return true;
        }
        a_lo <= b_hi && b_lo <= a_hi
    }
}

/// The median: the middle element of the sorted sample, or the mean of the
/// two middle elements for an even one.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Two-sided 95% quantile of Student's t with `df` degrees of freedom.
///
/// Tabulated to four decimals for `df <= 30` and at a few larger values; the
/// normal quantile `1.960` beyond `df = 120`. The table is what a benchmark of
/// three to thirty repeats needs; a run with hundreds of repeats is within a
/// per-mille of the normal interval anyway.
pub fn t_quantile_975(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281, 2.2010,
        2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860, 2.0796, 2.0739,
        2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
    ];
    match df {
        0 => f64::INFINITY,
        1..=30 => TABLE[df - 1],
        31..=40 => 2.0211,
        41..=60 => 2.0003,
        61..=120 => 1.9799,
        _ => 1.9600,
    }
}

/// One measurement series with everything needed to reproduce it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// What was measured, e.g. `bench/dpmpp2m/sequential`.
    pub name: String,
    /// Unit of `trials[*].value`, e.g. `ms`, `loss`.
    pub unit: String,
    pub environment: Environment,
    /// The configuration, as JSON, exactly as it was run.
    pub config: serde_json::Value,
    pub seeds: Vec<u64>,
    /// Every measurement taken, warm-ups included.
    pub trials: Vec<Trial>,
    /// Over the non-warm-up trials; `None` until one exists.
    pub summary: Option<Summary>,
    /// Anything else worth keeping (counts, sub-metrics).
    pub extra: serde_json::Value,
    pub recorded_unix_secs: u64,
}

impl Record {
    pub fn new(name: impl Into<String>, unit: impl Into<String>, config: serde_json::Value, seeds: Vec<u64>) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            environment: Environment::capture(),
            config,
            seeds,
            trials: Vec::new(),
            summary: None,
            extra: serde_json::Value::Null,
            recorded_unix_secs: crate::checkpoint::unix_now(),
        }
    }

    pub fn push(&mut self, value: f64, warmup: bool) {
        self.trials.push(Trial { value, warmup });
        self.summary = Summary::of(&self.measured());
    }

    /// The non-warm-up values, in order.
    pub fn measured(&self) -> Vec<f64> {
        self.trials.iter().filter(|t| !t.warmup).map(|t| t.value).collect()
    }

    pub fn with_extra(mut self, extra: serde_json::Value) -> Self {
        self.extra = extra;
        self
    }
}

/// An append-only JSONL file of [`Record`]s.
pub struct RunLog;

impl RunLog {
    /// Append one record as one line. The file is opened for append and
    /// created if missing; it is **never** truncated, so earlier measurements
    /// survive every later run. Parent directories are created.
    pub fn append(path: &Path, record: &Record) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
            }
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open {} for append", path.display()))?;
        let line = serde_json::to_string(record).context("serialize record")?;
        writeln!(file, "{line}").with_context(|| format!("append to {}", path.display()))?;
        file.flush()?;
        Ok(())
    }

    /// Every record in the file, in order. Blank lines are skipped; a
    /// malformed line is an error naming its number.
    pub fn read(path: &Path) -> anyhow::Result<Vec<Record>> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        text.lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
            .map(|(i, l)| serde_json::from_str(l).with_context(|| format!("{}:{}: malformed record", path.display(), i + 1)))
            .collect()
    }
}

/// One row of a comparison between two logs, matched by record name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    pub name: String,
    pub unit: String,
    pub a: Summary,
    pub b: Summary,
    /// `b.mean / a.mean`.
    pub ratio: f64,
    /// Whether the two 95% intervals overlap.
    pub overlap: bool,
}

/// Match records by name (the last record of each name wins) and compare.
pub fn compare(a: &[Record], b: &[Record]) -> Vec<ComparisonRow> {
    let latest = |records: &[Record]| -> std::collections::BTreeMap<String, Record> {
        let mut map = std::collections::BTreeMap::new();
        for r in records {
            map.insert(r.name.clone(), r.clone());
        }
        map
    };
    let (a, b) = (latest(a), latest(b));
    a.iter()
        .filter_map(|(name, ra)| {
            let rb = b.get(name)?;
            let (sa, sb) = (ra.summary?, rb.summary?);
            Some(ComparisonRow {
                name: name.clone(),
                unit: ra.unit.clone(),
                a: sa,
                b: sb,
                ratio: sb.mean / sa.mean,
                overlap: sa.overlaps(&sb),
            })
        })
        .collect()
}

/// A fixed-width table of a comparison.
pub fn render_comparison(rows: &[ComparisonRow]) -> String {
    let mut out = format!(
        "{:<36} {:>6} {:>14} {:>14} {:>8} {:>8}\n",
        "name", "unit", "a (mean ± ci)", "b (mean ± ci)", "b/a", "overlap"
    );
    out.push_str(&"-".repeat(92));
    out.push('\n');
    for r in rows {
        out.push_str(&format!(
            "{:<36} {:>6} {:>14} {:>14} {:>8.3} {:>8}\n",
            r.name,
            r.unit,
            format!("{:.4}±{:.4}", r.a.mean, r.a.ci95_half_width),
            format!("{:.4}±{:.4}", r.b.mean, r.b.ci95_half_width),
            r.ratio,
            if r.overlap { "yes" } else { "no" }
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_statistics_by_hand() {
        let s = Summary::of(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!((s.n, s.mean, s.median, s.min, s.max), (5, 3.0, 3.0, 1.0, 5.0));
        assert!((s.std - 2.5f64.sqrt()).abs() < 1e-12);
        assert!((s.ci95_half_width - 2.7764 * 2.5f64.sqrt() / 5f64.sqrt()).abs() < 1e-12);
        let single = Summary::of(&[7.0]).unwrap();
        assert!(single.std.is_nan() && single.ci95_half_width.is_nan());
        let text = serde_json::to_string(&single).unwrap();
        assert!(text.contains("\"std\":null"), "{text}");
        let back: Summary = serde_json::from_str(&text).unwrap();
        assert!(back.std.is_nan() && back.ci95_half_width.is_nan() && back.mean == 7.0);
        assert!(Summary::of(&[]).is_none());
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn test_record_keeps_warmups_out_of_the_summary_but_in_the_trials() {
        let mut r = Record::new("x", "ms", serde_json::json!({"k": 1}), vec![42]);
        r.push(100.0, true);
        r.push(1.0, false);
        r.push(3.0, false);
        assert_eq!(r.trials.len(), 3);
        assert_eq!(r.measured(), vec![1.0, 3.0]);
        assert_eq!(r.summary.unwrap().mean, 2.0);
    }

    #[test]
    fn test_run_log_appends_without_touching_earlier_bytes() {
        let dir = std::env::temp_dir().join("dblocks-experiment-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("log-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut a = Record::new("a", "ms", serde_json::Value::Null, vec![]);
        a.push(1.0, false);
        RunLog::append(&path, &a).unwrap();
        let before = std::fs::read(&path).unwrap();
        let mut b = Record::new("b", "ms", serde_json::Value::Null, vec![]);
        b.push(2.0, false);
        RunLog::append(&path, &b).unwrap();
        let after = std::fs::read(&path).unwrap();
        assert!(after.starts_with(&before), "earlier bytes must be untouched");
        let records = RunLog::read(&path).unwrap();
        assert_eq!(records.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        let rows = compare(&records[..1], &[Record { name: "a".into(), ..records[1].clone() }]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ratio, 2.0);
        assert!(render_comparison(&rows).contains("b/a"));
    }

    #[test]
    fn test_environment_capture_never_fails() {
        let env = Environment::capture();
        assert!(!env.build.crate_version.is_empty());
        assert!(env.logical_cpus > 0);
    }
}
