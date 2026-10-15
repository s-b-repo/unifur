//! Does the *training-time* cheat gate actually drop anything, or is it a
//! silent no-op on the coding traces?
//!
//! The queue log records `cheat_scanned: 2558, cheat_dropped: {}` for every
//! pipeline-gpu entry, against `cheat_dropped: {data-broken: 88, ...}` for the
//! pentest ones. Those two readings are indistinguishable from the log alone:
//! "the coding data is clean" and "the gate does nothing" produce the same
//! line. This probe separates them by running the *same* gate the trainer
//! builds (`CheatGate::inspect`) over
//!
//!   1. the real traces as they are, and
//!   2. the same file with a known-cheat row spliced in,
//!
//! so a gate that cannot drop the planted row is visibly not a gate.

use diffusionblocks::cheat::{CheatGate, ScanSummary, SnippetLanguage};
use serde_json::Value;
use std::path::Path;

fn classes_for(
    gate: &CheatGate,
    prompt: &str,
    completion: &str,
    lang: SnippetLanguage,
    s: &mut ScanSummary,
) -> Vec<String> {
    gate.inspect(Some(prompt), completion, lang, s)
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/srv/m-sda/unifur/teacher-traces/pipeline-gpu/rust/train.jsonl".into());
    let lang = SnippetLanguage::from_name("rust");
    let gate = CheatGate::patterns_only();

    let text = std::fs::read_to_string(Path::new(&path))?;
    let mut summary = ScanSummary::default();
    let mut kept = 0usize;
    let mut dropped = 0usize;
    let mut by_class: std::collections::BTreeMap<String, usize> = Default::default();
    let mut rows = Vec::new();

    for (i, raw) in text.lines().enumerate() {
        if raw.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(raw)?;
        let prompt = v.get("prompt").and_then(Value::as_str).unwrap_or("");
        let completion = v.get("completion").and_then(Value::as_str).unwrap_or("");
        let classes = classes_for(&gate, prompt, completion, lang, &mut summary);
        if classes.is_empty() {
            kept += 1;
        } else {
            dropped += 1;
            for c in &classes {
                *by_class.entry(c.clone()).or_default() += 1;
            }
        }
        if i < 3 {
            rows.push((prompt.to_string(), completion.to_string()));
        }
    }

    println!("file: {path}");
    println!("  real traces: kept {kept}, dropped {dropped}, by class {by_class:?}");

    // Now splice in rows carrying cheats the detector is supposed to name, and
    // ask the same gate about them.
    println!("\nplanted cheats (same gate, same language):");
    let planted: Vec<(&str, &str)> = vec![
        (
            "config",
            "fn f() {\n    #[allow(dead_code)]\n    #[allow(unused_variables)]\n    let x = 1;\n    let y = 2;\n    println!(\"{x} {y}\");\n}\n",
        ),
        (
            "stub",
            "pub fn parse(input: &str) -> Result<Ast, Error> {\n    // TODO: implement\n    todo!()\n}\n",
        ),
        (
            "eval gaming",
            "The answer is always A. The correct answer is A. A is correct. Answer: A.\n",
        ),
        (
            "secret",
            "let token = \"sk-abcdefghijklmnopqrstuvwxyz0123456789\";\nlet key = \"AKIAIOSFODNN7EXAMPLE\";\nclient.send(token);\n",
        ),
    ];
    let mut any_dropped = false;
    for (label, completion) in planted {
        let mut s = ScanSummary::default();
        let classes = classes_for(&gate, "write it for me", completion, lang, &mut s);
        if !classes.is_empty() {
            any_dropped = true;
        }
        println!(
            "  {label:12} -> {}",
            if classes.is_empty() {
                "KEPT  <-- NOT DETECTED".to_string()
            } else {
                format!("dropped {classes:?}")
            }
        );
    }

    // A clean row, as the control: the gate must NOT drop ordinary code.
    let mut s = ScanSummary::default();
    let clean = "fn add(a: i32, b: i32) -> i32 { a + b }\n";
    let classes = classes_for(&gate, "write add", clean, lang, &mut s);
    println!(
        "\n  {:12} -> {}",
        "clean code",
        if classes.is_empty() {
            "kept (correct)".to_string()
        } else {
            format!("DROPPED {classes:?}  <-- false positive")
        }
    );

    println!(
        "\nverdict: gate is {}",
        if any_dropped {
            "LIVE (drops planted cheats)"
        } else {
            "A NO-OP — it dropped nothing and detects nothing"
        }
    );
    let _ = rows;
    Ok(())
}
