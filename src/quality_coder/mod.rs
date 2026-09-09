//! Quality Coder — scaffolding for the agentic code refiner described in
//! [`crate::quality_coder`]'s design document.
//!
//! This module is **scaffolding only**. It defines the task, the prompt
//! format, and the data shapes the trainer and evaluator consume, but it
//! does not load any model, run any network calls, or spawn any
//! subprocess. The model itself is intended to live in a separate
//! Python-side fine-tuning pipeline (Qwen2.5-Coder-1.5B + LoRA via
//! HuggingFace + PEFT + TRL); the Rust side owns the data adapters and
//! the eval harness, which are easier to keep deterministic and
//! dependency-free than to maintain across two languages.
//!
//! # What lives here
//!
//! - [`Task`]: the input/output contract — a source file plus a defect
//!   list goes in, a unified diff comes out.
//! - [`Defect`]: a named defect from the antipattern catalog, with
//!   the line range it covers.
//! - [`Patch`]: the unified-diff output, with an optional refusal
//!   reason.
//! - [`Prompt`]: how a [`Task`] is rendered to text for the model.
//!   Deterministic, versioned, and round-trip-tested so a fine-tune
//!   trained on one version of the format still matches the prompt
//!   the inference server emits.
//!
//! # What does *not* live here (yet)
//!
//! - The Python training script. That is the next commit, gated on
//!   the design document being approved.
//! - The HuggingFace integration. Loading Qwen2.5-Coder-1.5B from a
//!   Python process is straightforward; doing it from Rust pulls in
//!   `tch-rs` or `candle` and is not justified by the rest of the
//!   crate.
//! - Inference. Once trained, the model is served by a Python
//!   process; Rust callers invoke it over a local HTTP or Unix-socket
//!   API, never embed the weights.

pub mod dataset;
pub mod eval;

use serde::{Deserialize, Serialize};

/// One named defect that the refiner is asked to fix.
///
/// `category` matches the names in [`crate::antipattern::Category`].
/// `span` is the half-open byte range in the source file the defect
/// was reported at; it is included in the prompt so the model does
/// not have to guess where the problem lives.
///
/// `severity` is the analyst's confidence that the pattern is a real
/// defect rather than a false positive. The refiner is instructed to
/// refuse fixes when severity is below a threshold, because
/// low-confidence defects often turn out to be intentional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Defect {
    pub category: String,
    /// Half-open byte range `[start, end)` in the source file.
    pub span: (usize, usize),
    /// Analyst confidence in `[0, 1]`. Below the `refusal_threshold`
    /// the refiner is allowed to emit an empty patch with a reason.
    #[serde(default = "default_severity")]
    pub severity: f32,
    /// Free-form explanation from the upstream classifier.
    #[serde(default)]
    pub explanation: String,
}

fn default_severity() -> f32 {
    1.0
}

/// The model's input: source file + defect list + optional test
/// command. The output is a [`Patch`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    /// Language of the source file, drawn from
    /// [`crate::codequality::Language`]. The prompt format depends on
    /// it; a Python file gets a different header than a Rust file.
    pub language: crate::codequality::Language,
    /// Path-style identifier for the file ("src/foo.py"). It is
    /// embedded in the prompt header so the model can produce
    /// hunks that reference the right path.
    pub path: String,
    /// Full source text. The model sees the whole file, not a
    /// snippet.
    pub source: String,
    /// Defects to fix. Empty list is a valid input: the prompt
    /// degrades to "no defects reported, do nothing" and the
    /// expected output is an empty patch.
    pub defects: Vec<Defect>,
    /// Optional test command. When present, the prompt asks the
    /// model to produce a patch that keeps the test command green.
    #[serde(default)]
    pub test_command: Option<String>,
}

impl Task {
    /// Construct a task with no defects and no test command. Useful
    /// for the synthetic fixture generator.
    pub fn identity(language: crate::codequality::Language, path: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            language,
            path: path.into(),
            source: source.into(),
            defects: Vec::new(),
            test_command: None,
        }
    }

    /// The prompt text fed to the model. Stable across versions —
    /// changing this is a breaking change for any fine-tune trained
    /// on the previous version.
    pub fn prompt(&self) -> Prompt {
        Prompt::render(self)
    }
}

/// The model's output: a unified diff, optionally with a refusal
/// reason.
///
/// The diff is parsed and validated by `Patch::parse` before
/// being scored; an unparseable patch is wrong by construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Patch {
    /// Unified diff in `git apply` format. Empty when the model
    /// refuses.
    pub diff: String,
    /// When `diff` is empty, this is the model's reason for refusing.
    /// Required when `diff` is empty; ignored otherwise.
    #[serde(default)]
    pub refusal_reason: Option<String>,
}

impl Patch {
    /// Empty patch with a refusal reason.
    pub fn refuse(reason: impl Into<String>) -> Self {
        Self { diff: String::new(), refusal_reason: Some(reason.into()) }
    }

    /// Empty patch with no reason. Equivalent to "do nothing" —
    /// useful as a sentinel value.
    pub fn empty() -> Self {
        Self { diff: String::new(), refusal_reason: None }
    }

    /// Whether this is a refusal. The eval harness treats refusals
    /// as correct only when the upstream defect list was empty;
    /// otherwise they are scored as wrong, because the user
    /// explicitly asked for a fix.
    pub fn is_refusal(&self) -> bool {
        self.diff.is_empty() && self.refusal_reason.is_some()
    }
}

/// The rendered prompt. Separated from [`Task`] so the version field
/// can be checked separately and so round-trip tests can compare
/// exact strings without re-parsing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    /// The prompt format version. Bumped whenever the rendering
    /// changes; serialized into training data so a checkpoint can be
    /// matched against the inference server it was trained for.
    pub version: u32,
    /// The text the model sees. Newlines are preserved.
    pub text: String,
}

impl Prompt {
    /// Bump this whenever [`Prompt::render`] changes. Older training
    /// runs are valid as long as the version they used is still
    /// emitted by the version of the code that runs them.
    pub const CURRENT_VERSION: u32 = 1;

    /// Render `task` to a prompt. The format is intentionally simple:
    /// a header, the source, the defect list, and an instruction
    /// suffix. No few-shot examples in the prompt itself — the model
    /// is fine-tuned on the format, not prompted with examples at
    /// inference time.
    pub fn render(task: &Task) -> Prompt {
        let mut text = String::new();
        text.push_str(
            "You are a code refiner. Given a source file and a list of defects, output a unified diff that fixes the defects without removing functionality.\n\n",
        );
        text.push_str(&format!("File: {} ({})\n", task.path, task.language.name()));
        text.push_str(&format!("Source length: {} bytes\n\n", task.source.len()));
        if let Some(cmd) = &task.test_command {
            text.push_str(&format!("Tests must pass: {cmd}\n\n"));
        }
        text.push_str("Defects:\n");
        if task.defects.is_empty() {
            text.push_str("  (none reported)\n");
        } else {
            for (i, d) in task.defects.iter().enumerate() {
                text.push_str(&format!(
                    "  {}. {} at bytes {}..{} (severity {:.2}): {}\n",
                    i + 1,
                    d.category,
                    d.span.0,
                    d.span.1,
                    d.severity,
                    if d.explanation.is_empty() { "(no explanation)" } else { d.explanation.as_str() }
                ));
            }
        }
        text.push_str("\nSource:\n```\n");
        text.push_str(&task.source);
        text.push_str("\n```\n\n");
        text.push_str(
            "Output a unified diff that fixes every defect above. Do not delete\n\
             the broken code path; route the error or repair the bug. If the\n\
             defect is intentional behavior the user should not silently undo,\n\
             output an empty diff and a one-line refusal reason.\n",
        );
        Prompt { version: Self::CURRENT_VERSION, text }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codequality::Language;

    fn sample_task() -> Task {
        let mut task = Task::identity(Language::Python, "src/foo.py", "def f():\n    try:\n        x = 1/0\n    except: pass\n");
        task.defects.push(Defect {
            category: "error-swallowing".into(),
            span: (30, 50),
            severity: 0.95,
            explanation: "bare except with pass".into(),
        });
        task
    }

    #[test]
    fn test_prompt_version_is_stable() {
        // Bumping CURRENT_VERSION is a deliberate, versioned change.
        // If a test asserts it is 1 today, a future bump must update
        // both the constant and the test, so an accidental bump
        // during refactoring fails the gate.
        assert_eq!(Prompt::CURRENT_VERSION, 1);
    }

    #[test]
    fn test_prompt_includes_every_defect() {
        let task = sample_task();
        let prompt = task.prompt();
        assert!(prompt.text.contains("error-swallowing"));
        assert!(prompt.text.contains("30..50"));
        assert!(prompt.text.contains("severity 0.95"));
        assert!(prompt.text.contains("bare except with pass"));
        assert!(prompt.text.contains("def f():"));
    }

    #[test]
    fn test_prompt_handles_empty_defect_list() {
        let task = Task::identity(Language::Rust, "src/lib.rs", "pub fn f() {}\n");
        let prompt = task.prompt();
        assert!(prompt.text.contains("(none reported)"));
        assert!(prompt.text.contains("rust"));
    }

    #[test]
    fn test_prompt_with_test_command_mentions_it() {
        let mut task = sample_task();
        task.test_command = Some("cargo test".into());
        let prompt = task.prompt();
        assert!(prompt.text.contains("Tests must pass: cargo test"));
    }

    #[test]
    fn test_prompt_rendering_is_deterministic() {
        let task = sample_task();
        let a = task.prompt();
        let b = task.prompt();
        assert_eq!(a, b, "the same task must render identically");
    }

    #[test]
    fn test_patch_refuse_is_a_refusal() {
        let p = Patch::refuse("defect looks intentional");
        assert!(p.is_refusal());
        assert_eq!(p.diff, "");
    }

    #[test]
    fn test_patch_empty_is_not_a_refusal() {
        let p = Patch::empty();
        assert!(!p.is_refusal());
    }

    #[test]
    fn test_patch_with_diff_is_not_a_refusal() {
        let p = Patch { diff: "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n".into(), refusal_reason: None };
        assert!(!p.is_refusal());
    }

    #[test]
    fn test_defect_severity_defaults_to_one() {
        // A classifier that does not report severity is implicitly
        // fully confident. This is the wrong default in production
        // but matches what every static-analysis tool emits, so it
        // is the safe assumption for the deserializer.
        let json = r#"{"category":"foo","span":[0,1]}"#;
        let d: Defect = serde_json::from_str(json).unwrap();
        assert_eq!(d.severity, 1.0);
    }

    #[test]
    fn test_task_round_trip_through_json() {
        let task = sample_task();
        let json = serde_json::to_string(&task).unwrap();
        let parsed: Task = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, task);
    }

    #[test]
    fn test_prompt_serializes_with_its_version() {
        // The version is part of the data: a future training run
        // that sees a prompt without a version is, by definition,
        // running against the wrong code.
        let task = sample_task();
        let prompt = task.prompt();
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(json.contains("\"version\":1"));
    }
}