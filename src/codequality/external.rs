//! External-tool-backed analyzer. Compiled regardless of the
//! `codequality-external` feature; the implementation is empty when the
//! feature is off so the rest of the pipeline compiles unchanged.

use super::ExternalAnalyzer;

/// Run the configured command on `source` and turn its output into a
/// density score in `[0, 1]`.
///
/// Score formula: `1 - min(warnings / max(1, lines), 1.0)`. A file with
/// zero warnings scores `1.0`; a file whose every line has a warning
/// scores `0.0`. The intermediate values are continuous, so a partial
/// regression in a window is not lost to a binary "drop / keep" decision.
///
/// Returns `Ok(None)` when no command is configured, or when the tool's
/// stdout holds no JSON this function can read (it declines rather than
/// guessing). Returns an error when the tool cannot be spawned, fed, or
/// waited for, or when it exits unsuccessfully without reporting a single
/// diagnostic — a crashed or misconfigured tool, not a clean file. Neither
/// case is ever turned into a score: an earlier version scored unreadable
/// output as zero warnings, i.e. a perfect `1.0`, which *raised* the
/// geometric mean of every window it touched.
#[cfg(feature = "codequality-external")]
pub fn run(analyzer: &ExternalAnalyzer, source: &str) -> anyhow::Result<Option<f32>> {
    use anyhow::Context;
    use std::io::Write;
    use std::process::{Command, Stdio};

    if let Some((lang, options)) = &analyzer.runner {
        let outcomes = crate::lintrunner::lint_all(source, *lang, options);
        return Ok(lint_density(&outcomes, source));
    }
    let Some((program, args)) = analyzer.command.split_first() else {
        return Ok(None);
    };
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    let mut child = Command::new(program)
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawn external analyzer {program:?}"))?;
    // Feed stdin from a separate thread while `wait_with_output` drains
    // stdout. Writing it all first deadlocks once both pipe buffers fill: the
    // tool blocks writing diagnostics nobody reads while we block writing
    // source it has stopped reading.
    let writer = child.stdin.take().map(|mut stdin| {
        let bytes = source.as_bytes().to_vec();
        std::thread::spawn(move || stdin.write_all(&bytes))
    });
    let output = child
        .wait_with_output()
        .with_context(|| format!("wait for external analyzer {program:?}"))?;
    if let Some(writer) = writer {
        match writer.join() {
            Ok(result) => {
                // A tool that exits without reading all of stdin (it lints
                // files, not stdin) closes the pipe early; that is reported
                // rather than scored.
                result.with_context(|| format!("write source to external analyzer {program:?}"))?;
            }
            Err(_) => anyhow::bail!("the stdin writer for external analyzer {program:?} panicked"),
        }
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let warnings = count_json_messages(&stdout);
    if !output.status.success() && warnings.is_none_or(|n| n == 0) {
        anyhow::bail!(
            "external analyzer {program:?} exited with {} and reported no diagnostics; \
             refusing to score its output as a clean file",
            output.status
        );
    }
    Ok(warnings.map(|n| density_score(n, source)))
}

/// The external dimension from [`crate::lintrunner::lint_all`] outcomes: the
/// same density as one configured command, over the diagnostics of the tools
/// that *completed*. A tool that is not installed, was refused, timed out or
/// failed contributes nothing; when no tool completed there is no score at
/// all (`None`), because "nothing could read this" is not "this is clean".
#[cfg(any(feature = "codequality-external", test))]
fn lint_density(outcomes: &[crate::lintrunner::LintOutcome], source: &str) -> Option<f32> {
    let completed: Vec<&crate::lintrunner::LintOutcome> = outcomes
        .iter()
        .filter(|o| o.status == crate::lintrunner::LintStatus::Completed)
        .collect();
    if completed.is_empty() {
        return None;
    }
    Some(density_score(
        completed.iter().map(|o| o.diagnostics).sum(),
        source,
    ))
}

#[cfg(any(feature = "codequality-external", test))]
fn density_score(warnings: usize, source: &str) -> f32 {
    let lines = source.lines().count().max(1);
    1.0 - (warnings as f32 / lines as f32).min(1.0)
}

/// Count the diagnostics in a tool's JSON output, or `None` when stdout holds
/// no JSON at all.
///
/// Reads a stream of JSON values, which covers the three shapes the presets
/// produce: one object per line (`cargo clippy --message-format=json`), one
/// pretty-printed array (`ruff --output-format=json`), and a one-line array of
/// per-file results (`eslint --format=json`). Counting lines that start with
/// `{` — the previous heuristic — read eslint's single `[...]` line as zero
/// warnings, counted every nested `{` of ruff's pretty-printed fixes as a
/// warning, and counted cargo's `compiler-artifact` and `build-finished`
/// records as warnings too.
#[cfg(any(feature = "codequality-external", test))]
fn count_json_messages(stdout: &str) -> Option<usize> {
    let mut values = serde_json::Deserializer::from_str(stdout).into_iter::<serde_json::Value>();
    let mut parsed = 0usize;
    let mut count = 0usize;
    // Stop at the first non-JSON byte: whatever parsed before it is counted,
    // and output that never parses is "unreadable", not "clean".
    while let Some(Ok(value)) = values.next() {
        parsed += 1;
        count += diagnostics_in(&value);
    }
    (parsed > 0).then_some(count)
}

#[cfg(any(feature = "codequality-external", test))]
fn diagnostics_in(value: &serde_json::Value) -> usize {
    match value {
        // A list of findings, or of per-file results carrying their own.
        serde_json::Value::Array(items) => items.iter().map(diagnostic_or_file).sum(),
        serde_json::Value::Object(_) => diagnostic_or_file(value),
        _ => 0,
    }
}

#[cfg(any(feature = "codequality-external", test))]
fn diagnostic_or_file(value: &serde_json::Value) -> usize {
    // eslint: `{"filePath": .., "messages": [..]}` per file.
    if let Some(messages) = value.get("messages").and_then(|m| m.as_array()) {
        return messages.len();
    }
    // cargo: only `compiler-message` records are diagnostics.
    if let Some(reason) = value.get("reason").and_then(|r| r.as_str()) {
        return usize::from(reason == "compiler-message");
    }
    usize::from(value.is_object())
}

#[cfg(not(feature = "codequality-external"))]
pub fn run(_analyzer: &ExternalAnalyzer, _source: &str) -> anyhow::Result<Option<f32>> {
    Ok(None)
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

    #[test]
    fn test_each_preset_shape_is_counted_by_its_diagnostics() {
        // cargo: one object per line, only compiler-message records count.
        let cargo = concat!(
            r#"{"reason":"compiler-artifact","target":{}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"warning"}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"warning"}}"#,
            "\n",
            r#"{"reason":"build-finished","success":true}"#,
            "\n"
        );
        assert_eq!(count_json_messages(cargo), Some(2));

        // eslint: one line, an array of per-file results.
        let eslint = r#"[{"filePath":"a.js","messages":[{"ruleId":"x"},{"ruleId":"y"}]},{"filePath":"b.js","messages":[{"ruleId":"z"}]}]"#;
        assert_eq!(count_json_messages(eslint), Some(3));

        // ruff: a pretty-printed array; nested objects are not findings.
        let ruff = "[\n  {\n    \"code\": \"F401\",\n    \"fix\": {\n      \"edits\": [\n        {\n          \"content\": \"\"\n        }\n      ]\n    }\n  }\n]\n";
        assert_eq!(count_json_messages(ruff), Some(1));

        // An empty array is a clean report, not an unreadable one.
        assert_eq!(count_json_messages("[]"), Some(0));
    }

    #[test]
    fn test_unreadable_output_is_declined_not_scored_clean() {
        assert_eq!(count_json_messages(""), None);
        assert_eq!(count_json_messages("error: no such command\n"), None);
    }

    #[test]
    fn test_unavailable_linters_contribute_nothing_and_never_a_clean_score() {
        use crate::lintrunner::{LintOutcome, LintStatus};
        let outcome = |tool: &'static str, status: LintStatus, diagnostics: usize| LintOutcome {
            tool,
            status,
            diagnostics,
            messages: Vec::new(),
            exit: None,
            timed_out: status == LintStatus::TimedOut,
            truncated: false,
        };
        let source = "a\nb\nc\nd\n";
        // Nothing ran: no score, not 1.0.
        let none_ran = [
            outcome("ruff", LintStatus::Unavailable, 0),
            outcome("mypy", LintStatus::Refused, 0),
            outcome("pylint", LintStatus::TimedOut, 0),
            outcome("pyright", LintStatus::Failed, 0),
        ];
        assert_eq!(lint_density(&none_ran, source), None);
        assert_eq!(lint_density(&[], source), None);
        // Only completed tools count, and their diagnostics are summed.
        let mixed = [
            outcome("ruff", LintStatus::Completed, 1),
            outcome("mypy", LintStatus::Unavailable, 0),
            outcome("pylint", LintStatus::Completed, 1),
        ];
        assert_eq!(lint_density(&mixed, source), Some(0.5));
        let clean = [outcome("ruff", LintStatus::Completed, 0)];
        assert_eq!(lint_density(&clean, source), Some(1.0));
    }

    #[test]
    fn test_density_score_matches_its_formula() {
        assert_eq!(density_score(0, "a\nb\n"), 1.0);
        assert_eq!(density_score(1, "a\nb\n"), 0.5);
        assert_eq!(density_score(9, "a\n"), 0.0);
        assert_eq!(density_score(0, ""), 1.0);
    }
}
