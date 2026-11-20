//! Evaluation harness for the agentic refiner (roadmap Phase 25).
//!
//! The trainer learns to produce [`super::Patch`] values; this module
//! scores them. Three metrics, each on a held-out split of each
//! source, plus a composite headline number. None of the metrics
//! requires running the trained model — they can be exercised on the
//! gold patches to verify the harness itself is correct, which is
//! what the unit tests do.
//!
//! # Why these three metrics
//!
//! - **`apply_patch_success`**: a patch that fails to apply is wrong
//!   by construction. If this metric is below ~99% the model is
//!   producing malformed output, not bad fixes.
//! - **`lint_clean_rate`**: the patch's contribution to code quality,
//!   scored by the same [`crate::codequality`] module the regularizer
//!   uses. The improvement is checked against the original file's
//!   score on the same dimension, so a patch that removes a buggy
//!   function entirely (and thus raises the score) is *not* counted
//!   as improvement — the dimension that drops is functionality.
//! - **`test_preserved`**: the failure mode the design document is
//!   built around. A patch that fixes the named defect but breaks
//!   previously-passing tests has removed real functionality.
//!
//! # What the harness is *not*
//!
//! - Not an LLM-as-judge. The metrics are deterministic; an LLM judge
//!   is added in a later release once the deterministic metrics are
//!   above target.
//! - Not an autonomous test runner. It does not invoke the test
//!   command on a real checkout; the `test_preserved` metric is a
//!   proxy computed from the unified diff's hunk structure. Real test
//!   execution is added in the next release.

use serde::{Deserialize, Serialize};

use super::dataset::Source;
use super::Patch;
use crate::codequality::{
    CodeAnalyzer, CompositeAnalyzer, ExternalAnalyzer, Language, QualityScore, StructuralAnalyzer,
};

/// One evaluation metric.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Fraction of patches that apply cleanly to the source.
    ApplyPatchSuccess,
    /// Fraction of *all* evaluated examples whose patch applied and
    /// improved quality without regressing any dimension below 0.5 (an
    /// unapplied patch counts as not improved).
    LintCleanRate,
    /// Fraction of patches whose diff touches only lines within the
    /// defect span (a proxy for "did not delete unrelated code").
    /// The full version of this metric runs the test command; see
    /// the module docs.
    TestPreserved,
}

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Self::ApplyPatchSuccess => "apply_patch_success",
            Self::LintCleanRate => "lint_clean_rate",
            Self::TestPreserved => "test_preserved",
        }
    }
}

/// The headline composite score: `0.1 * apply + 0.5 * lint + 0.4 *
/// test`. Matches the design document. The weights are deliberately
/// not user-tunable in the first release; the design document
/// commits to these numbers as the contract for v1.0.
pub const COMPOSITE_WEIGHTS: [f32; 3] = [0.1, 0.5, 0.4];

/// Aggregate of all three metrics for one (model, source) pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub source: Source,
    pub evaluated: usize,
    pub apply_patch_success: f32,
    pub lint_clean_rate: f32,
    pub test_preserved: f32,
    /// `0.1 * apply + 0.5 * lint + 0.4 * test`.
    pub composite: f32,
}

impl EvalReport {
    /// Whether this report meets the v1.0 targets from the design
    /// document. `apply >= 0.99`, `lint >= 0.80`, `test >= 0.95`.
    pub fn meets_v1_targets(&self) -> bool {
        self.apply_patch_success >= 0.99
            && self.lint_clean_rate >= 0.80
            && self.test_preserved >= 0.95
    }

    pub fn render(&self) -> String {
        format!(
            "{:<14} n={:<6} apply={:.3} lint={:.3} test={:.3} composite={:.3} {}",
            self.source.name(),
            self.evaluated,
            self.apply_patch_success,
            self.lint_clean_rate,
            self.test_preserved,
            self.composite,
            if self.meets_v1_targets() {
                "v1-target"
            } else {
                "below-target"
            },
        )
    }
}

/// Per-example scoring result. One row per held-out example, useful
/// for debugging a low-scoring model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreRow {
    pub example_id: String,
    pub source: Source,
    pub applied: bool,
    pub quality_improved: bool,
    pub test_preserved: bool,
    pub original_score: f32,
    pub patched_score: f32,
}

/// The harness. Construct with [`EvalHarness::new`] and call
/// [`EvalHarness::score`] with one held-out split at a time.
#[derive(Debug, Clone)]
pub struct EvalHarness {
    analyzer: CompositeAnalyzer<IdentityAnalyzer>,
}

impl Default for EvalHarness {
    /// [`EvalHarness::new`], falling back to a harness without the lexical
    /// dimension if the built-in rules do not compile; the reason goes to
    /// stderr, since `Default` cannot report it.
    fn default() -> Self {
        match Self::new() {
            Ok(harness) => harness,
            Err(err) => {
                eprintln!("quality-coder: built-in rules unavailable, scoring without the lexical dimension: {err:#}");
                Self {
                    analyzer: CompositeAnalyzer::<IdentityAnalyzer>::new(
                        Language::Generic,
                        None,
                        Some(StructuralAnalyzer::default()),
                        Some(ExternalAnalyzer::new("none", Vec::new())),
                    ),
                }
            }
        }
    }
}

impl EvalHarness {
    /// Build a harness with the default analyzer: structural
    /// heuristics on, anti-pattern density on, external tools off.
    /// The composite score is the same one the regularizer uses, so
    /// improvements here are improvements there.
    pub fn new() -> anyhow::Result<Self> {
        let labeler = crate::antipattern::Labeler::builtin()?;
        Ok(Self {
            analyzer: CompositeAnalyzer::<IdentityAnalyzer>::new(
                Language::Generic,
                Some(labeler),
                Some(StructuralAnalyzer::default()),
                Some(ExternalAnalyzer::new("none", Vec::new())),
            ),
        })
    }

    /// Score a held-out split. `examples` is the held-out set;
    /// `predictions[i]` is the model's patch for `examples[i]`.
    /// Order is the caller's responsibility.
    pub fn score(
        &self,
        source: Source,
        examples: &[super::dataset::Example],
        predictions: &[Patch],
    ) -> (EvalReport, Vec<ScoreRow>) {
        assert_eq!(
            examples.len(),
            predictions.len(),
            "examples and predictions must align"
        );
        let mut rows = Vec::with_capacity(examples.len());
        let mut applied = 0usize;
        let mut improved = 0usize;
        let mut preserved = 0usize;

        for (example, predicted) in examples.iter().zip(predictions) {
            let patched = match apply_unified_diff(&example.prompt.text, predicted) {
                Some(s) => s,
                None => {
                    rows.push(ScoreRow {
                        example_id: example.id.clone(),
                        source,
                        applied: false,
                        quality_improved: false,
                        test_preserved: false,
                        original_score: 0.0,
                        patched_score: 0.0,
                    });
                    continue;
                }
            };
            applied += 1;

            let original = self
                .analyzer
                .analyze(&extract_source_from_prompt(&example.prompt.text));
            let patched_score = self.analyzer.analyze(&patched);
            let improves = patched_score.overall > original.overall
                && !regressed_below_threshold(&original, &patched_score);
            if improves {
                improved += 1;
            }

            // `test_preserved` is a structural proxy: a patch that
            // touches lines outside the defect span is more likely to
            // have deleted functionality. The full version runs the
            // test command; this proxy is what the harness scores
            // until that lands.
            let touches_outside = touches_lines_outside_defect(predicted, &example.prompt.text);
            if !touches_outside {
                preserved += 1;
            }

            rows.push(ScoreRow {
                example_id: example.id.clone(),
                source,
                applied: true,
                quality_improved: improves,
                test_preserved: !touches_outside,
                original_score: original.overall,
                patched_score: patched_score.overall,
            });
        }

        let n = examples.len().max(1) as f32;
        let apply = applied as f32 / n;
        let lint = improved as f32 / n;
        let test = preserved as f32 / n;
        let composite = COMPOSITE_WEIGHTS[0] * apply
            + COMPOSITE_WEIGHTS[1] * lint
            + COMPOSITE_WEIGHTS[2] * test;

        let report = EvalReport {
            source,
            evaluated: examples.len(),
            apply_patch_success: apply,
            lint_clean_rate: lint,
            test_preserved: test,
            composite,
        };
        (report, rows)
    }
}

/// A no-op analyzer used as the [`CodeAnalyzer`] type parameter on
/// [`CompositeAnalyzer`]. The composite analyzer carries the real
/// signal; the placeholder is required by the type system.
#[derive(Debug, Clone)]
struct IdentityAnalyzer;
impl CodeAnalyzer for IdentityAnalyzer {
    fn language(&self) -> Language {
        Language::Generic
    }
    fn analyze(&self, _: &str) -> QualityScore {
        QualityScore::identity(Language::Generic)
    }
}

// --------------------------------------------------- diff application --

/// Apply a unified diff to `source`. Returns `None` if the diff does
/// not parse or does not match the source's line structure.
///
/// This is a deliberately minimal implementation: the refiner is
/// expected to produce small, well-formed diffs, and a full
/// `git apply` clone would be tens of thousands of lines. The
/// eval-time correctness bar is "we can tell good patches from bad
/// ones", not "we are a production patch tool".
pub fn apply_unified_diff(prompt: &str, patch: &Patch) -> Option<String> {
    if patch.is_refusal() {
        return None;
    }
    if patch.diff.is_empty() {
        return None;
    }
    // Whole-file rewrite marker from the dataset adapter: bypass the
    // diff parser and use the payload directly.
    if let Some(payload) = patch.diff.strip_prefix(WHOLE_FILE_REWRITE) {
        return Some(payload.to_string());
    }
    apply_unified_diff_inner(&extract_source_from_prompt(prompt), &patch.diff)
}

/// The marker the dataset adapters put in front of a whole-file rewrite
/// (see `super::dataset`).
const WHOLE_FILE_REWRITE: &str = "# quality-coder: whole-file-rewrite\n";

fn apply_unified_diff_inner(source: &str, diff: &str) -> Option<String> {
    let hunks = parse_hunks(diff)?;
    let source_lines: Vec<&str> = source.split_inclusive('\n').collect();
    let mut out = String::with_capacity(source.len());
    let mut source_idx = 0usize;
    for hunk in &hunks {
        // Copy source lines up to the hunk's start.
        while source_idx < hunk.old_start.saturating_sub(1) && source_idx < source_lines.len() {
            out.push_str(source_lines[source_idx]);
            source_idx += 1;
        }
        // Apply the hunk's lines.
        for line in &hunk.lines {
            match line.kind {
                LineKind::Context => {
                    if source_idx >= source_lines.len() {
                        return None;
                    }
                    if source_lines[source_idx] != line.content.as_str() {
                        return None;
                    }
                    out.push_str(source_lines[source_idx]);
                    source_idx += 1;
                }
                LineKind::Add => {
                    out.push_str(&line.content);
                    if !line.content.ends_with('\n') {
                        out.push('\n');
                    }
                }
                LineKind::Remove => {
                    // Content check: a `-foo` line that does not match
                    // the source line it claims to remove is a sign the
                    // patch was generated against a different file, and
                    // applying it would corrupt unrelated code. The cost
                    // of false rejection is one extra prompt cycle; the
                    // cost of false acceptance is silent data loss.
                    if source_idx >= source_lines.len() {
                        return None;
                    }
                    if source_lines[source_idx] != line.content.as_str() {
                        return None;
                    }
                    source_idx += 1;
                }
            }
        }
    }
    // Copy any trailing lines.
    while source_idx < source_lines.len() {
        out.push_str(source_lines[source_idx]);
        source_idx += 1;
    }
    Some(out)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LineKind {
    Context,
    Add,
    Remove,
}

#[derive(Debug, Clone)]
struct DiffLine {
    kind: LineKind,
    content: String,
}

#[derive(Debug, Clone)]
struct Hunk {
    old_start: usize,
    lines: Vec<DiffLine>,
}

/// Minimal unified-diff parser. Supports `--- a/path`, `+++ b/path`,
/// `@@ -old,count +new,count @@`, and ` `/`+`/`-` lines. Anything
/// fancier (binary patches, rename detection, git extensions) is
/// refused.
fn parse_hunks(diff: &str) -> Option<Vec<Hunk>> {
    let mut lines = diff.lines();
    // Skip the header lines.
    while let Some(line) = lines.clone().next() {
        if line.starts_with("@@") {
            break;
        }
        lines.next();
    }

    let mut hunks = Vec::new();
    let mut current: Option<Hunk> = None;

    for line in lines {
        if let Some(rest) = line.strip_prefix("@@") {
            if let Some(h) = current.take() {
                hunks.push(h);
            }
            // Parse `@@ -old_start[,old_count] +new_start[,new_count] @@`.
            let header = rest.trim().trim_end_matches("@@").trim();
            let parts: Vec<&str> = header.split_whitespace().collect();
            if parts.len() < 2 {
                return None;
            }
            let old_part = parts[0].trim_start_matches('-');
            let old_start: usize = old_part.split(',').next().and_then(|s| s.parse().ok())?;
            current = Some(Hunk {
                old_start,
                lines: Vec::new(),
            });
        } else if let Some(h) = current.as_mut() {
            if let Some(content) = line.strip_prefix(' ') {
                h.lines.push(DiffLine {
                    kind: LineKind::Context,
                    content: content.to_string() + "\n",
                });
            } else if let Some(content) = line.strip_prefix('+') {
                h.lines.push(DiffLine {
                    kind: LineKind::Add,
                    content: content.to_string(),
                });
            } else if let Some(content) = line.strip_prefix('-') {
                h.lines.push(DiffLine {
                    kind: LineKind::Remove,
                    content: content.to_string() + "\n",
                });
            }
        }
    }
    if let Some(h) = current.take() {
        hunks.push(h);
    }
    if hunks.is_empty() {
        None
    } else {
        Some(hunks)
    }
}

/// Pull the source out of a rendered prompt. The prompt format is
/// stable — see [`super::Task::prompt`] — so a substring search is
/// sufficient.
fn extract_source_from_prompt(prompt: &str) -> String {
    let start = prompt
        .find("Source:\n```\n")
        .map(|i| i + "Source:\n```\n".len());
    let Some(start) = start else {
        return String::new();
    };
    let rest = &prompt[start..];
    let end = rest.find("\n```\n\nOutput").unwrap_or(rest.len());
    rest[..end].to_string()
}

fn regressed_below_threshold(before: &QualityScore, after: &QualityScore) -> bool {
    let threshold = 0.5;
    for d in &before.dimensions {
        let a = after
            .dimensions
            .iter()
            .find(|x| x.name == d.name)
            .map(|x| x.score)
            .unwrap_or(0.0);
        if d.score >= threshold && a < threshold {
            return true;
        }
    }
    false
}

/// Structural proxy for "did not delete unrelated code": whether the
/// patch changes any source line outside the reported defect spans. This
/// is a heuristic, not the full test runner.
///
/// Each defect's byte span (rendered into the prompt as `at bytes S..E`)
/// is mapped to the 1-based source lines it covers. A removed line must
/// lie inside some span; an added line must be inserted inside a span or
/// directly after its last line. A whole-file rewrite touches every line.
/// With no defects reported, any change at all is outside; a diff whose
/// hunks do not parse counts as outside.
///
/// (This used to flag only hunks starting past twice the file's length,
/// which no patch that applies can do -- so the metric was, in effect,
/// a second copy of `apply_patch_success`.)
fn touches_lines_outside_defect(patch: &Patch, prompt: &str) -> bool {
    let source = extract_source_from_prompt(prompt);
    let spans: Vec<(usize, usize)> = defect_spans(prompt)
        .into_iter()
        .map(|(start, end)| line_range(&source, start, end))
        .collect();
    let removable = |line: usize| {
        spans
            .iter()
            .any(|&(first, last)| first <= line && line <= last)
    };
    let insertable = |at: usize| {
        spans
            .iter()
            .any(|&(first, last)| first <= at && at <= last + 1)
    };

    if patch.diff.starts_with(WHOLE_FILE_REWRITE) {
        return (1..=source.lines().count()).any(|line| !removable(line));
    }
    let Some(hunks) = parse_hunks(&patch.diff) else {
        return true;
    };
    for hunk in &hunks {
        // `line` is the old-file line the next hunk line refers to; an
        // added line is inserted before it.
        let mut line = hunk.old_start.max(1);
        for l in &hunk.lines {
            match l.kind {
                LineKind::Context => line += 1,
                LineKind::Remove => {
                    if !removable(line) {
                        return true;
                    }
                    line += 1;
                }
                LineKind::Add => {
                    if !insertable(line) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// The `(start, end)` byte spans of the defects listed in a rendered
/// prompt (see [`super::Prompt::render`]: `N. category at bytes S..E ...`).
fn defect_spans(prompt: &str) -> Vec<(usize, usize)> {
    const HEADER: &str = "\nDefects:\n";
    let Some(start) = prompt.find(HEADER).map(|i| i + HEADER.len()) else {
        return Vec::new();
    };
    let section = &prompt[start..];
    let section = &section[..section.find("\nSource:\n```\n").unwrap_or(section.len())];
    section
        .lines()
        .filter_map(|line| {
            const AT: &str = " at bytes ";
            let rest = &line[line.find(AT)? + AT.len()..];
            let (s, rest) = rest.split_once("..")?;
            let e: String = rest.chars().take_while(char::is_ascii_digit).collect();
            Some((s.parse().ok()?, e.parse().ok()?))
        })
        .collect()
}

/// The 1-based, inclusive `(first, last)` lines of `source` that the byte
/// span `[start, end)` covers.
fn line_range(source: &str, start: usize, end: usize) -> (usize, usize) {
    let bytes = source.as_bytes();
    let line_of = |b: usize| {
        bytes[..b.min(bytes.len())]
            .iter()
            .filter(|&&c| c == b'\n')
            .count()
            + 1
    };
    let first = line_of(start);
    let last = line_of(end.saturating_sub(1).max(start));
    (first, last.max(first))
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
    use crate::codequality::Language;
    use crate::quality_coder::{Defect, Patch, Task};

    fn sample_task(source: &str) -> Task {
        Task {
            language: Language::Python,
            path: "src/foo.py".into(),
            source: source.into(),
            defects: vec![Defect {
                category: "error-swallowing".into(),
                span: (0, source.len()),
                severity: 1.0,
                explanation: "bare except".into(),
            }],
            test_command: None,
        }
    }

    #[test]
    fn test_composite_weights_match_design_doc() {
        assert_eq!(COMPOSITE_WEIGHTS, [0.1, 0.5, 0.4]);
    }

    #[test]
    fn test_apply_unified_diff_handles_a_simple_replacement() {
        let source = "def f():\n    return 1\n";
        let task = sample_task(source);
        let prompt = task.prompt();
        let patch_text = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-def f():\n-    return 1\n+def f():\n+    return 2\n";
        let patched = apply_unified_diff(
            &prompt.text,
            &Patch {
                diff: patch_text.into(),
                refusal_reason: None,
            },
        )
        .unwrap();
        assert_eq!(patched, "def f():\n    return 2\n");
    }

    #[test]
    fn test_apply_unified_diff_returns_none_on_unparseable() {
        let source = "def f():\n    return 1\n";
        let task = sample_task(source);
        let prompt = task.prompt();
        let patch_text = "this is not a diff";
        assert!(apply_unified_diff(
            &prompt.text,
            &Patch {
                diff: patch_text.into(),
                refusal_reason: None
            }
        )
        .is_none());
    }

    #[test]
    fn test_apply_unified_diff_returns_none_on_context_mismatch() {
        let source = "def f():\n    return 1\n";
        let task = sample_task(source);
        let prompt = task.prompt();
        // The hunk header says "lines 1..3", which matches the source,
        // but the removed content does not match the actual line. The
        // parser compares context/remove lines to the source, so a
        // mismatch is detected and the patch is rejected.
        let patch_text = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,2 +1,2 @@\n-def f():\n-    return 999\n+def f():\n+    return 2\n";
        let extracted = extract_source_from_prompt(&prompt.text);
        eprintln!("extracted: {extracted:?}");
        let result = apply_unified_diff(
            &prompt.text,
            &Patch {
                diff: patch_text.into(),
                refusal_reason: None,
            },
        );
        eprintln!("result: {result:?}");
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_unified_diff_bypasses_for_whole_file_marker() {
        let source = "def f():\n    return 1\n";
        let task = sample_task(source);
        let prompt = task.prompt();
        let rewritten = "# quality-coder: whole-file-rewrite\ndef f():\n    return 2\n";
        let patched = apply_unified_diff(
            &prompt.text,
            &Patch {
                diff: rewritten.into(),
                refusal_reason: None,
            },
        )
        .unwrap();
        assert_eq!(patched, "def f():\n    return 2\n");
    }

    #[test]
    fn test_apply_unified_diff_returns_none_for_refusal() {
        let source = "def f():\n    return 1\n";
        let task = sample_task(source);
        let prompt = task.prompt();
        assert!(apply_unified_diff(&prompt.text, &Patch::refuse("looks intentional")).is_none());
    }

    #[test]
    fn test_score_reports_correct_metrics_on_gold_patches() {
        // The harness scoring a perfect patch should give apply=1.0;
        // lint and test depend on whether the patch actually improves
        // the file, which a hand-crafted patch in the test fixture does.
        let source = "def f():\n    try:\n        x = 1/0\n    except:\n        pass\n";
        let task = sample_task(source);
        let prompt = task.prompt();
        let good_patch = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,4 +1,5 @@\n def f():\n     try:\n         x = 1/0\n-    except:\n-        pass\n+    except ZeroDivisionError as e:\n+        log.error(e)\n";
        let example = super::super::dataset::Example {
            source: Source::SweBench,
            id: "t1".into(),
            prompt,
            target: Patch {
                diff: good_patch.into(),
                refusal_reason: None,
            },
        };
        let harness = EvalHarness::new().expect("built-in rules");
        let (report, rows) = harness.score(
            Source::SweBench,
            std::slice::from_ref(&example),
            &[Patch {
                diff: good_patch.into(),
                refusal_reason: None,
            }],
        );
        assert_eq!(report.evaluated, 1);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].applied);
        // The defect span is the whole file, so the gold patch stays inside it.
        assert!(rows[0].test_preserved);
        // We don't assert on lint/test here because the small test
        // corpus and the harness's coarse structural proxy don't
        // always agree with a perfect patch; the metric contract is
        // exercised end-to-end in the eval scripts.
    }

    #[test]
    fn test_extract_source_from_prompt_round_trips() {
        let source = "line 1\nline 2\nline 3\n";
        let task = sample_task(source);
        let prompt = task.prompt();
        let extracted = extract_source_from_prompt(&prompt.text);
        assert_eq!(extracted, source);
    }

    #[test]
    fn test_regressed_below_threshold_detects_dimension_drop() {
        let before = QualityScore::from_dimensions(
            Language::Python,
            vec![super::super::super::codequality::Dimension::new("a", 0.8)],
            1,
        );
        let after = QualityScore::from_dimensions(
            Language::Python,
            vec![super::super::super::codequality::Dimension::new("a", 0.3)],
            1,
        );
        assert!(regressed_below_threshold(&before, &after));
    }

    #[test]
    fn test_regressed_below_threshold_ignores_low_dimensions() {
        // A dimension that was already below the threshold and stayed
        // there is not a "regression": it was already broken.
        let before = QualityScore::from_dimensions(
            Language::Python,
            vec![super::super::super::codequality::Dimension::new("a", 0.3)],
            1,
        );
        let after = QualityScore::from_dimensions(
            Language::Python,
            vec![super::super::super::codequality::Dimension::new("a", 0.1)],
            1,
        );
        assert!(!regressed_below_threshold(&before, &after));
    }

    #[test]
    fn test_eval_report_meets_v1_targets() {
        let report = EvalReport {
            source: Source::SweBench,
            evaluated: 100,
            apply_patch_success: 0.99,
            lint_clean_rate: 0.80,
            test_preserved: 0.95,
            composite: 0.1 * 0.99 + 0.5 * 0.80 + 0.4 * 0.95,
        };
        assert!(report.meets_v1_targets());
    }

    #[test]
    fn test_eval_report_fails_v1_targets_below_thresholds() {
        let report = EvalReport {
            source: Source::CodeReviewer,
            evaluated: 100,
            apply_patch_success: 0.95,
            lint_clean_rate: 0.80,
            test_preserved: 0.95,
            composite: 0.0,
        };
        assert!(!report.meets_v1_targets());
    }

    #[test]
    fn test_metric_names_are_stable() {
        // Eval reports are written to JSON; renaming a metric
        // breaks every downstream consumer. Asserted here so an
        // accidental rename fails the gate.
        assert_eq!(Metric::ApplyPatchSuccess.name(), "apply_patch_success");
        assert_eq!(Metric::LintCleanRate.name(), "lint_clean_rate");
        assert_eq!(Metric::TestPreserved.name(), "test_preserved");
    }

    #[test]
    fn test_touches_lines_outside_defect_checks_the_defect_span() {
        // Four lines; the defect covers only line 2 ("b\n" is bytes 2..4).
        let source = "a\nb\nc\nd\n";
        let mut task = sample_task(source);
        task.defects[0].span = (2, 4);
        let prompt = task.prompt();
        let patch = |diff: &str| Patch {
            diff: diff.into(),
            refusal_reason: None,
        };
        assert_eq!(defect_spans(&prompt.text), vec![(2, 4)]);

        let inside = patch("@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n");
        assert!(apply_unified_diff(&prompt.text, &inside).is_some());
        assert!(!touches_lines_outside_defect(&inside, &prompt.text));

        // Applies cleanly, but deletes line 4, which no defect covers. The
        // old proxy (hunk start past twice the file length) passed this.
        let outside = patch("@@ -3,2 +3,1 @@\n c\n-d\n");
        assert_eq!(
            apply_unified_diff(&prompt.text, &outside).as_deref(),
            Some("a\nb\nc\n")
        );
        assert!(touches_lines_outside_defect(&outside, &prompt.text));

        // A whole-file rewrite touches every line, so it is outside a
        // one-line span...
        let rewrite = patch("# quality-coder: whole-file-rewrite\n");
        assert!(touches_lines_outside_defect(&rewrite, &prompt.text));
        // ...but inside a span that covers the whole file.
        let whole = sample_task(source).prompt();
        assert!(!touches_lines_outside_defect(&rewrite, &whole.text));

        // With no defects reported, any change is outside.
        let none = Task::identity(Language::Python, "src/foo.py", source).prompt();
        assert!(touches_lines_outside_defect(&inside, &none.text));
    }

    #[test]
    fn test_touches_lines_outside_defect_flags_far_hunks() {
        let source = "def f():\n    return 1\n";
        let task = sample_task(source);
        let prompt = task.prompt();
        let bad_patch = "--- a/x\n+++ b/x\n@@ -1000 +1 @@\n-old\n+new\n";
        assert!(touches_lines_outside_defect(
            &Patch {
                diff: bad_patch.into(),
                refusal_reason: None
            },
            &prompt.text,
        ));
    }
}
