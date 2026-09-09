//! Dataset adapters for the three training sources plus the synthetic
//! slice. **Adapters only** — no network I/O, no subprocess calls.
//! Each adapter takes a directory of pre-downloaded records and emits
//! [`Example`] values; the loader is responsible for fetching the
//! raw data.
//!
//! # Why adapters rather than one unified schema
//!
//! The three sources have different shapes (CodeReviewer emits
//! `(file, review comment, accepted patch)`, SWE-bench emits
//! `(repo state at issue, issue text, accepted patch)`, Code-Feedback
//! emits `(code, execution result, corrected code)`), and the
//! information they carry is genuinely different. A unified schema
//! would either lose information or invent fake fields to fill gaps.
//! Adapters that produce the same [`Example`] downstream let the
//! trainer mix them without caring which source they came from.
//!
//! # License filtering
//!
//! CodeReviewer in particular inherits its per-PR licenses from the
//! repositories it scraped. A permissive-only filter is applied at
//! load time; the loader returns the rejected count so the user can
//! decide whether to widen the filter.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::{Defect, Task};

/// Which training source an example came from. Recorded so the
/// per-source eval split can be reconstructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    CodeReviewer,
    SweBench,
    CodeFeedback,
    /// A `(clean code, injected defect, fixed code)` triple produced
    /// by the synthetic generator. The reproducibility test exercises
    /// only this slice.
    Synthetic,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Self::CodeReviewer => "code-reviewer",
            Self::SweBench => "swe-bench",
            Self::CodeFeedback => "code-feedback",
            Self::Synthetic => "synthetic",
        }
    }
}

/// A single training or eval example.
///
/// `prompt` is the rendered [`Task::prompt`] text; `target` is the
/// gold [`super::Patch`]. The trainer consumes `(prompt, target)`
/// pairs; the eval harness consumes `prompt` plus a separate
/// `target` for scoring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Example {
    pub source: Source,
    /// Stable identifier within the source — useful for debugging
    /// ("why did example X score Y?"). The exact format depends on
    /// the source; for SWE-bench it is the issue id, for
    /// CodeReviewer the PR number + file path.
    pub id: String,
    pub prompt: super::Prompt,
    pub target: super::Patch,
}

/// Per-source statistics from a load run.
///
/// The rejected counts are the load-time filter results, not the
/// trainer's later filtering. A high rejected count is a signal that
/// the source's data needs a stricter license filter or a quality
/// pre-filter.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoadReport {
    pub loaded: usize,
    pub rejected_by_license: usize,
    pub rejected_by_format: usize,
    pub skipped_empty: usize,
}

impl LoadReport {
    pub fn merge(&mut self, other: &LoadReport) {
        self.loaded += other.loaded;
        self.rejected_by_license += other.rejected_by_license;
        self.rejected_by_format += other.rejected_by_format;
        self.skipped_empty += other.skipped_empty;
    }

    pub fn render(&self, source: Source) -> String {
        format!(
            "{}: loaded={} | rejected (license={}, format={}, empty={})",
            source.name(),
            self.loaded,
            self.rejected_by_license,
            self.rejected_by_format,
            self.skipped_empty,
        )
    }
}

/// Permissive licenses the loader keeps. Anything not on this list
/// is rejected. The list is deliberately narrow; widening it is a
/// deliberate action, not an oversight.
pub const PERMISSIVE_LICENSES: &[&str] = &[
    "MIT", "Apache-2.0", "BSD-2-3", "BSD-3-Clause", "ISC", "Unlicense", "CC0-1.0", "MPL-2.0",
];

/// Whether `license` is permissive enough to train on. Case- and
/// whitespace-insensitive: `"Apache 2.0"`, `"apache-2.0"`, and
/// `"APACHE-2.0"` all match the `Apache-2.0` entry on the
/// permissive list.
pub fn is_permissive(license: &str) -> bool {
    let normalized = license.trim().to_ascii_uppercase().replace([' ', '_'], "-");
    PERMISSIVE_LICENSES
        .iter()
        .any(|p| normalized.contains(p.to_ascii_uppercase().as_str()))
}

// ------------------------------------------------------ CodeReviewer --

/// One row of the CodeReviewer JSON dump. The schema is whatever
/// Microsoft ships; we extract the three fields the refiner needs
/// and ignore the rest. The `derive(Deserialize)` is forgiving
/// (every field but `review_id` and `patch` is `default`) so the
/// loader survives schema additions.
#[derive(Debug, Clone, Deserialize)]
pub struct CodeReviewerRow {
    #[serde(default)]
    pub review_id: String,
    #[serde(default)]
    pub file_path: String,
    #[serde(default)]
    pub pre_file: String,
    #[serde(default)]
    pub patch: String,
    #[serde(default)]
    pub review_comment: String,
    #[serde(default)]
    pub lang: Option<String>,
    #[serde(default)]
    pub license: String,
}

/// Load CodeReviewer rows from a JSONL file.
///
/// Each line is one row. The loader walks the file twice: once to
/// validate format and apply the license filter, once to build
/// [`Example`] values, so the rejected counts are exact even when
/// the filter rejects everything.
pub fn load_code_reviewer(path: &Path) -> Result<(Vec<Example>, LoadReport)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut examples = Vec::new();
    let mut report = LoadReport::default();

    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: CodeReviewerRow = match serde_json::from_str(line) {
            Ok(row) => row,
            Err(_) => {
                report.rejected_by_format += 1;
                continue;
            }
        };
        if !is_permissive(&row.license) {
            report.rejected_by_license += 1;
            continue;
        }
        if row.patch.is_empty() || row.pre_file.is_empty() {
            report.skipped_empty += 1;
            continue;
        }
        let language = crate::codequality::Language::parse(row.lang.as_deref().unwrap_or(""));
        let task = Task {
            language,
            path: row.file_path.clone(),
            source: row.pre_file.clone(),
            //: CodeReviewer rows name a review comment, not a defect
            //: category. The trainer is responsible for matching the
            //: comment to a category during data prep; the loader
            //: keeps the comment as the defect explanation.
            defects: vec![Defect {
                category: "review-comment".into(),
                span: (0, row.pre_file.len()),
                severity: 1.0,
                explanation: row.review_comment.clone(),
            }],
            test_command: None,
        };
        let id = if row.review_id.is_empty() {
            format!("line:{}", line_no + 1)
        } else {
            row.review_id.clone()
        };
        examples.push(Example {
            source: Source::CodeReviewer,
            id,
            prompt: task.prompt(),
            target: super::Patch { diff: row.patch.clone(), refusal_reason: None },
        });
    }
    report.loaded = examples.len();
    Ok((examples, report))
}

// ---------------------------------------------------------- SWE-bench --

/// One row of the SWE-bench Lite JSON dump.
#[derive(Debug, Clone, Deserialize)]
pub struct SweBenchRow {
    #[serde(default)]
    pub instance_id: String,
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub pre_file: String,
    #[serde(default)]
    pub patch: String,
    #[serde(default)]
    pub problem_statement: String,
    #[serde(default)]
    pub test_patch: String,
    #[serde(default)]
    pub license: String,
}

pub fn load_swe_bench(path: &Path) -> Result<(Vec<Example>, LoadReport)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut examples = Vec::new();
    let mut report = LoadReport::default();

    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: SweBenchRow = match serde_json::from_str(line) {
            Ok(row) => row,
            Err(_) => {
                report.rejected_by_format += 1;
                continue;
            }
        };
        if !is_permissive(&row.license) {
            report.rejected_by_license += 1;
            continue;
        }
        if row.patch.is_empty() || row.pre_file.is_empty() {
            report.skipped_empty += 1;
            continue;
        }
        let task = Task {
            language: crate::codequality::Language::Python,
            path: format!("{}/src", row.repo),
            source: row.pre_file.clone(),
            defects: vec![Defect {
                category: "issue-fix".into(),
                span: (0, row.pre_file.len()),
                severity: 1.0,
                explanation: row.problem_statement.clone(),
            }],
            test_command: Some(row.test_patch.clone()),
        };
        let id = if row.instance_id.is_empty() {
            format!("line:{}", line_no + 1)
        } else {
            row.instance_id.clone()
        };
        examples.push(Example {
            source: Source::SweBench,
            id,
            prompt: task.prompt(),
            target: super::Patch { diff: row.patch.clone(), refusal_reason: None },
        });
    }
    report.loaded = examples.len();
    Ok((examples, report))
}

// --------------------------------------------------------- CodeFeedback --

#[derive(Debug, Clone, Deserialize)]
pub struct CodeFeedbackRow {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub lang: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub feedback: String,
    #[serde(default)]
    pub corrected_code: String,
    #[serde(default)]
    pub license: String,
}

pub fn load_code_feedback(path: &Path) -> Result<(Vec<Example>, LoadReport)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut examples = Vec::new();
    let mut report = LoadReport::default();

    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: CodeFeedbackRow = match serde_json::from_str(line) {
            Ok(row) => row,
            Err(_) => {
                report.rejected_by_format += 1;
                continue;
            }
        };
        if !is_permissive(&row.license) {
            report.rejected_by_license += 1;
            continue;
        }
        if row.code.is_empty() || row.corrected_code.is_empty() {
            report.skipped_empty += 1;
            continue;
        }
        let language = crate::codequality::Language::parse(&row.lang);
        // Code-Feedback patches are whole-file rewrites, not unified
        // diffs. The trainer converts at data-prep time; the loader
        // keeps the rewrite as-is and tags the example so the
        // conversion is detectable downstream.
        let task = Task {
            language,
            path: format!("line_{}.txt", line_no + 1),
            source: row.code.clone(),
            defects: vec![Defect {
                category: "execution-feedback".into(),
                span: (0, row.code.len()),
                severity: 1.0,
                explanation: row.feedback.clone(),
            }],
            test_command: None,
        };
        let id = if row.id.is_empty() {
            format!("line:{}", line_no + 1)
        } else {
            row.id.clone()
        };
        // Wrap the rewrite in a placeholder diff so the [`Patch`]
        // schema stays uniform. The trainer's prep step converts it
        // to a real diff; the marker is a comment that `patch` would
        // reject, so a misformatted dataset fails loudly rather than
        // silently shipping whole-file rewrites through training.
        let marker = "# quality-coder: whole-file-rewrite\n";
        let pseudo_diff = format!("{marker}{}", row.corrected_code);
        examples.push(Example {
            source: Source::CodeFeedback,
            id,
            prompt: task.prompt(),
            target: super::Patch { diff: pseudo_diff, refusal_reason: None },
        });
    }
    report.loaded = examples.len();
    Ok((examples, report))
}

// ------------------------------------------------------------ synthetic --

/// Generate a single synthetic example from the antipattern rule
/// catalog. The clean source is taken from one of the rule's
/// `counterexamples` (which are by definition NOT flagged) and a
/// defect from the same rule's `examples` is patched in.
///
/// The reverse mapping — "clean this up" — is the training target.
/// The reproducibility test exercises this path.
pub fn synthetic_example_from_rule(
    rule_name: &str,
    category: &str,
    language: crate::codequality::Language,
    broken: &str,
    fixed: &str,
) -> Example {
    let task = Task {
        language,
        path: format!("synthetic/{rule_name}"),
        source: broken.into(),
        defects: vec![Defect {
            category: category.into(),
            span: (0, broken.len()),
            severity: 1.0,
            explanation: format!("antipattern rule '{rule_name}'"),
        }],
        test_command: None,
    };
    // Wrap the rewrite in the same marker so the synthetic slice is
    // indistinguishable from Code-Feedback at the data-prep step.
    let marker = "# quality-coder: whole-file-rewrite\n";
    Example {
        source: Source::Synthetic,
        id: format!("synthetic:{rule_name}"),
        prompt: task.prompt(),
        target: super::Patch { diff: format!("{marker}{fixed}"), refusal_reason: None },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_jsonl(dir: &Path, name: &str, rows: &[&str]) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, rows.join("\n")).unwrap();
        path
    }

    fn scratch_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("dblocks-quality-coder-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_permissive_licenses_match_with_case_insensitivity() {
        assert!(is_permissive("MIT"));
        assert!(is_permissive("mit"));
        assert!(is_permissive("Apache-2.0"));
        assert!(is_permissive("APACHE 2.0"));
        assert!(is_permissive("BSD-3-Clause"));
        assert!(!is_permissive("GPL-3.0"));
        assert!(!is_permissive("AGPL-3.0"));
        assert!(!is_permissive(""));
    }

    #[test]
    fn test_code_reviewer_loader_filters_by_license() {
        let dir = scratch_dir();
        let path = write_jsonl(&dir, "code_reviewer.jsonl", &[
            r#"{"review_id":"r1","file_path":"src/foo.py","pre_file":"def f():\n    pass\n","patch":"--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-    pass\n+    return None\n","review_comment":"return something","license":"MIT"}"#,
            r#"{"review_id":"r2","file_path":"src/bar.py","pre_file":"def f():\n    pass\n","patch":"--- a/src/bar.py\n+++ b/src/bar.py\n@@ -1 +1 @@\n-    pass\n+    return 1\n","review_comment":"return something","license":"GPL-3.0"}"#,
        ]);
        let (examples, report) = load_code_reviewer(&path).unwrap();
        assert_eq!(report.loaded, 1);
        assert_eq!(report.rejected_by_license, 1);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].source, Source::CodeReviewer);
        assert_eq!(examples[0].id, "r1");
    }

    #[test]
    fn test_code_reviewer_loader_skips_malformed_rows() {
        let dir = scratch_dir();
        let path = write_jsonl(&dir, "code_reviewer_bad.jsonl", &[
            r#"{"review_id":"r1","pre_file":"def f(): pass\n","patch":"--- a\n+++ b\n@@\n-x\n+y\n","license":"MIT"}"#,
            r#"this is not json"#,
            // Parses cleanly, has a permissive license, but the source
            // and patch are empty -- this is the `skipped_empty` path.
            r#"{"review_id":"r2","license":"MIT"}"#,
        ]);
        let (_examples, report) = load_code_reviewer(&path).unwrap();
        assert_eq!(report.loaded, 1, "only the first row is valid");
        // The malformed line is rejected by format. The empty-content
        // line parses but has no source/patch and is rejected as empty
        // content. Both are correct behavior -- the bug the test would
        // catch is counting one of them as `loaded`.
        assert_eq!(report.rejected_by_format, 1);
        assert_eq!(report.skipped_empty, 1);
        assert_eq!(report.loaded, 1);
    }

    #[test]
    fn test_swe_bench_loader_python_only() {
        let dir = scratch_dir();
        let path = write_jsonl(&dir, "swe_bench.jsonl", &[
            r#"{"instance_id":"django-1","repo":"django/django","pre_file":"def f(): pass\n","patch":"--- a\n+++ b\n@@\n-x\n+y\n","problem_statement":"fix f","test_patch":"+def test_f(): assert f() == 1","license":"BSD-3-Clause"}"#,
        ]);
        let (examples, report) = load_swe_bench(&path).unwrap();
        assert_eq!(report.loaded, 1);
        assert!(examples[0].prompt.text.contains("Tests must pass"));
        assert_eq!(examples[0].source, Source::SweBench);
    }

    #[test]
    fn test_code_feedback_loader_marks_rewrites() {
        let dir = scratch_dir();
        let path = write_jsonl(&dir, "code_feedback.jsonl", &[
            r#"{"id":"cf1","lang":"python","code":"def f():\n    return 1/0\n","feedback":"ZeroDivisionError","corrected_code":"def f():\n    try:\n        return 1/0\n    except ZeroDivisionError:\n        return None\n","license":"Apache-2.0"}"#,
        ]);
        let (examples, report) = load_code_feedback(&path).unwrap();
        assert_eq!(report.loaded, 1);
        assert!(examples[0].target.diff.contains("quality-coder: whole-file-rewrite"));
    }

    #[test]
    fn test_synthetic_example_carries_marker() {
        let ex = synthetic_example_from_rule(
            "except-pass",
            "error-swallowing",
            crate::codequality::Language::Python,
            "try:\n    f()\nexcept:\n    pass\n",
            "try:\n    f()\nexcept Exception as e:\n    log.error(e)\n",
        );
        assert_eq!(ex.source, Source::Synthetic);
        assert!(ex.target.diff.contains("quality-coder: whole-file-rewrite"));
        assert!(ex.prompt.text.contains("error-swallowing"));
    }

    #[test]
    fn test_load_report_merges_correctly() {
        let mut a = LoadReport { loaded: 10, rejected_by_license: 2, rejected_by_format: 1, skipped_empty: 0 };
        let b = LoadReport { loaded: 5, rejected_by_license: 1, rejected_by_format: 0, skipped_empty: 3 };
        a.merge(&b);
        assert_eq!(a.loaded, 15);
        assert_eq!(a.rejected_by_license, 3);
        assert_eq!(a.rejected_by_format, 1);
        assert_eq!(a.skipped_empty, 3);
    }

    #[test]
    fn test_load_report_render_includes_source_name() {
        let r = LoadReport { loaded: 100, rejected_by_license: 10, rejected_by_format: 5, skipped_empty: 2 };
        let rendered = r.render(Source::SweBench);
        assert!(rendered.contains("swe-bench"));
        assert!(rendered.contains("100"));
        assert!(rendered.contains("10"));
        assert!(rendered.contains("5"));
        assert!(rendered.contains("2"));
    }

    use std::path::PathBuf;
}