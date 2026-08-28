//! Lightweight JSONL metrics logging (roadmap items 1.5 / 15.6).
//!
//! Writes one JSON object per training step to `--log-file`. The schema is
//! deliberately W&B-compatible (`"step"`, flat metric keys) so the file can
//! be replayed into external dashboards later; a hosted-W&B adapter is left
//! out on purpose to keep this crate network-free.

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

/// Append-only JSONL metrics writer.
pub struct MetricsLogger {
    writer: BufWriter<File>,
}

impl MetricsLogger {
    /// Open `path` for appending, creating it (and any parent directories) if
    /// needed. Appending rather than truncating is what makes a resumed run
    /// extend its own metrics file instead of erasing the history it is
    /// resuming from.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        Self::open_with(path, true)
    }

    /// Open `path`, truncating it first when `append` is false.
    pub fn open_with(path: &Path, append: bool) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file: File = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)?;
        Ok(Self { writer: BufWriter::new(file) })
    }

    /// Write one flat key/value record with a `step` field.
    ///
    /// Values must already be JSON scalars (numbers, strings, bools).
    pub fn log(&mut self, step: usize, fields: &[(&str, String)]) -> anyhow::Result<()> {
        let mut line = format!(r#"{{"step":{step}"#);
        for (key, value) in fields {
            line.push_str(&format!(r#","{key}":{value}"#));
        }
        line.push_str("}\n");
        self.writer.write_all(line.as_bytes())?;
        self.writer.flush()?;
        Ok(())
    }
}

/// Helper formatting an f32 metric as a JSON number.
pub fn jnum(v: f32) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonl_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dblocks-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.jsonl");

        let mut logger = MetricsLogger::open_with(&path, false).unwrap();
        logger.log(0, &[("loss", jnum(1.5)), ("block", "0".to_string())]).unwrap();
        drop(logger);

        // Re-opening must append, not truncate: a resumed run keeps history.
        let mut logger = MetricsLogger::open(&path).unwrap();
        logger.log(1, &[("loss", jnum(f32::NAN)), ("note", "\"nan-case\"".to_string())]).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        assert_eq!(
            lines.next().unwrap(),
            r#"{"step":0,"loss":1.5,"block":0}"#
        );
        assert_eq!(
            lines.next().unwrap(),
            r#"{"step":1,"loss":null,"note":"nan-case"}"#
        );
        assert!(lines.next().is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
