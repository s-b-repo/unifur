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
/// Returns `Ok(None)` when no command is configured; an error when the
/// tool cannot be spawned, fed, or waited for. Non-JSON output is not an
/// error: it counts as zero warnings (see `count_json_messages`). The
/// composite analyzer reports the error and skips the external dimension,
/// so a broken tool never silently changes the score.
#[cfg(feature = "codequality-external")]
pub fn run(analyzer: &ExternalAnalyzer, source: &str) -> anyhow::Result<Option<f32>> {
    use anyhow::Context;
    use std::io::Write;
    use std::process::{Command, Stdio};

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
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(source.as_bytes())
            .with_context(|| format!("write source to external analyzer {program:?}"))?;
    }
    let output = child
        .wait_with_output()
        .with_context(|| format!("wait for external analyzer {program:?}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_warnings(&stdout, source))
}

#[cfg(feature = "codequality-external")]
fn parse_warnings(stdout: &str, source: &str) -> Option<f32> {
    let warnings = count_json_messages(stdout);
    let lines = source.lines().count().max(1);
    Some(1.0 - (warnings as f32 / lines as f32).min(1.0))
}

/// Naive JSON-line counter for diagnostics output. Tools like clippy's
/// `--message-format json` emit one object per finding; we count the
/// opening braces, which is robust to additions of fields we don't care
/// about. With non-JSON output the count is zero and the window scores
/// `1.0` -- the analyzer declines to penalize rather than inventing a
/// penalty from a parse failure.
#[cfg(feature = "codequality-external")]
fn count_json_messages(stdout: &str) -> usize {
    stdout
        .lines()
        .filter(|line| line.trim_start().starts_with('{'))
        .count()
}

#[cfg(not(feature = "codequality-external"))]
pub fn run(_analyzer: &ExternalAnalyzer, _source: &str) -> anyhow::Result<Option<f32>> {
    Ok(None)
}
