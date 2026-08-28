//! Scope-level timing harness (roadmap item 13.5).
//!
//! A profiler that answers the only question worth asking before optimizing:
//! *where does the time actually go?* Named scopes accumulate wall-clock
//! samples, and [`Profiler::render`] reports count, total, mean, median, p95
//! and share of total, sorted by total time -- so the first row is the thing
//! to work on.
//!
//! Percentiles rather than means alone, because the distributions here are
//! routinely skewed: a denoise step that is usually fast but occasionally
//! stalls on allocation has a healthy mean and a terrible p95, and only the
//! second one is visible to a user.
//!
//! Every sample is retained (8 bytes each), so percentiles are exact rather
//! than estimated. A long profiling run over millions of scope entries should
//! therefore expect to spend a few megabytes.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

/// Accumulated timings for one named scope.
#[derive(Debug, Clone, Default)]
pub struct ScopeStats {
    samples: Vec<u64>,
    total_ns: u128,
}

impl ScopeStats {
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    pub fn total(&self) -> Duration {
        Duration::from_nanos(self.total_ns.min(u64::MAX as u128) as u64)
    }

    pub fn total_nanos(&self) -> u128 {
        self.total_ns
    }

    pub fn mean(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        Duration::from_nanos((self.total_ns / self.samples.len() as u128) as u64)
    }

    pub fn min(&self) -> Duration {
        Duration::from_nanos(self.samples.iter().copied().min().unwrap_or(0))
    }

    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.samples.iter().copied().max().unwrap_or(0))
    }

    /// Nearest-rank percentile: the smallest sample at or above `q` of the
    /// distribution. `q` is clamped to `[0, 1]`.
    pub fn percentile(&self, q: f64) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let q = q.clamp(0.0, 1.0);
        // Nearest-rank: rank = ceil(q * n), clamped into 1..=n, then 0-indexed.
        let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
        Duration::from_nanos(sorted[rank.min(sorted.len()) - 1])
    }
}

/// Collects timings for named scopes.
#[derive(Debug, Clone, Default)]
pub struct Profiler {
    scopes: BTreeMap<String, ScopeStats>,
}

impl Profiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one sample against `name`.
    pub fn record(&mut self, name: &str, elapsed: Duration) {
        let entry = self.scopes.entry(name.to_string()).or_default();
        let nanos = elapsed.as_nanos();
        entry.samples.push(nanos.min(u64::MAX as u128) as u64);
        entry.total_ns += nanos;
    }

    /// Time `f`, record it against `name`, and return its value.
    pub fn time<T>(&mut self, name: &str, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let value = f();
        self.record(name, start.elapsed());
        value
    }

    pub fn stats(&self, name: &str) -> Option<&ScopeStats> {
        self.scopes.get(name)
    }

    pub fn scope_names(&self) -> Vec<&str> {
        self.scopes.keys().map(String::as_str).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.scopes.is_empty()
    }

    /// Summed time across every scope.
    ///
    /// Scopes are independent counters, not a partition of wall-clock time:
    /// nesting one scope inside another double-counts, by design, since both
    /// numbers are separately useful.
    pub fn total_nanos(&self) -> u128 {
        self.scopes.values().map(|s| s.total_ns).sum()
    }

    pub fn clear(&mut self) {
        self.scopes.clear();
    }

    /// Scopes ordered by total time, descending -- the optimization worklist.
    pub fn ranked(&self) -> Vec<(&str, &ScopeStats)> {
        let mut rows: Vec<(&str, &ScopeStats)> =
            self.scopes.iter().map(|(k, v)| (k.as_str(), v)).collect();
        rows.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns).then(a.0.cmp(b.0)));
        rows
    }

    /// Table sorted by total time.
    pub fn render(&self) -> String {
        let total = self.total_nanos().max(1);
        let mut out = String::new();
        out.push_str(&format!(
            "{:<28} {:>8} {:>12} {:>11} {:>11} {:>11} {:>7}\n",
            "scope", "count", "total", "mean", "p50", "p95", "share"
        ));
        out.push_str(&"-".repeat(94));
        out.push('\n');
        for (name, s) in self.ranked() {
            out.push_str(&format!(
                "{:<28} {:>8} {:>12} {:>11} {:>11} {:>11} {:>6.1}%\n",
                name,
                s.count(),
                format_duration(s.total()),
                format_duration(s.mean()),
                format_duration(s.percentile(0.5)),
                format_duration(s.percentile(0.95)),
                100.0 * s.total_ns as f64 / total as f64
            ));
        }
        out
    }
}

/// Compact human-readable duration.
pub fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}us", ns as f64 / 1e3)
    } else if ns < 1_000_000_000 {
        format!("{:.1}ms", ns as f64 / 1e6)
    } else {
        format!("{:.2}s", ns as f64 / 1e9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build stats from explicit nanosecond samples so the tests do not
    /// depend on wall-clock timing at all.
    fn stats_from(samples: &[u64]) -> ScopeStats {
        let mut p = Profiler::new();
        for &ns in samples {
            p.record("s", Duration::from_nanos(ns));
        }
        p.stats("s").unwrap().clone()
    }

    #[test]
    fn test_accumulation() {
        let s = stats_from(&[10, 20, 30, 40]);
        assert_eq!(s.count(), 4);
        assert_eq!(s.total_nanos(), 100);
        assert_eq!(s.mean(), Duration::from_nanos(25));
        assert_eq!(s.min(), Duration::from_nanos(10));
        assert_eq!(s.max(), Duration::from_nanos(40));
    }

    #[test]
    fn test_nearest_rank_percentiles() {
        // Nearest-rank on 1..=100: p50 = 50, p95 = 95, p100 = 100, p0 = 1.
        let s = stats_from(&(1..=100).collect::<Vec<u64>>());
        assert_eq!(s.percentile(0.5), Duration::from_nanos(50));
        assert_eq!(s.percentile(0.95), Duration::from_nanos(95));
        assert_eq!(s.percentile(1.0), Duration::from_nanos(100));
        assert_eq!(s.percentile(0.0), Duration::from_nanos(1));
        // Out-of-range quantiles clamp instead of panicking.
        assert_eq!(s.percentile(-5.0), Duration::from_nanos(1));
        assert_eq!(s.percentile(9.0), Duration::from_nanos(100));
    }

    #[test]
    fn test_percentiles_ignore_insertion_order() {
        let ascending = stats_from(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let shuffled = stats_from(&[7, 2, 10, 4, 1, 9, 3, 8, 5, 6]);
        assert_eq!(ascending.percentile(0.5), shuffled.percentile(0.5));
        assert_eq!(ascending.percentile(0.95), shuffled.percentile(0.95));
        assert_eq!(ascending.total_nanos(), shuffled.total_nanos());
    }

    #[test]
    fn test_percentiles_expose_a_skewed_tail() {
        // The reason percentiles are reported at all: a mean that looks fine
        // can hide a p95 that does not.
        let mut samples = vec![10u64; 95];
        samples.extend(std::iter::repeat_n(1000u64, 5));
        let s = stats_from(&samples);
        assert_eq!(s.percentile(0.5), Duration::from_nanos(10));
        assert_eq!(s.percentile(0.95), Duration::from_nanos(10));
        assert_eq!(s.percentile(0.96), Duration::from_nanos(1000));
        assert!(s.mean() < Duration::from_nanos(100), "mean hides the tail");
    }

    #[test]
    fn test_empty_scope_is_all_zero() {
        let s = ScopeStats::default();
        assert_eq!(s.count(), 0);
        assert_eq!(s.mean(), Duration::ZERO);
        assert_eq!(s.percentile(0.5), Duration::ZERO);
        assert_eq!(s.min(), Duration::ZERO);
    }

    #[test]
    fn test_time_returns_the_value_and_records_a_sample() {
        let mut p = Profiler::new();
        let value = p.time("work", || 6 * 7);
        assert_eq!(value, 42);
        assert_eq!(p.stats("work").unwrap().count(), 1);
        assert!(p.stats("missing").is_none());
    }

    #[test]
    fn test_ranking_is_by_total_time() {
        let mut p = Profiler::new();
        p.record("fast", Duration::from_nanos(1));
        p.record("slow", Duration::from_nanos(1000));
        p.record("medium", Duration::from_nanos(100));
        assert_eq!(p.scope_names(), vec!["fast", "medium", "slow"]); // BTreeMap order
        assert_eq!(
            p.ranked().iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            vec!["slow", "medium", "fast"]
        );

        let rendered = p.render();
        let slow_line = rendered.lines().nth(2).unwrap();
        assert!(slow_line.starts_with("slow"), "slowest scope must come first");
        assert!(rendered.contains('%'));

        p.clear();
        assert!(p.is_empty());
    }

    #[test]
    fn test_shares_sum_to_one_hundred_percent() {
        let mut p = Profiler::new();
        p.record("a", Duration::from_nanos(250));
        p.record("b", Duration::from_nanos(750));
        let total = p.total_nanos() as f64;
        let share: f64 = p
            .ranked()
            .iter()
            .map(|(_, s)| 100.0 * s.total_nanos() as f64 / total)
            .sum();
        assert!((share - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_duration_formatting_switches_units() {
        assert_eq!(format_duration(Duration::from_nanos(999)), "999ns");
        assert_eq!(format_duration(Duration::from_nanos(1_500)), "1.5us");
        assert_eq!(format_duration(Duration::from_micros(1_500)), "1.5ms");
        assert_eq!(format_duration(Duration::from_millis(1_500)), "1.50s");
    }
}
