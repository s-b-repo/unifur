//! Count Qwen-tokenizer tokens over an instruction JSONL manifest.
//!
//! Data tool for Frontier-27B-Plan Phase G prep: validates `bpe` at scale on
//! real text and sizes the run (tokens/day math). Reads rows shaped
//! `{instruction, response}`, encodes `instruction + "\n" + response` with
//! the file's declared pipeline, and reports totals. No model, no training.
//!
//! Usage: `cargo run --example tokcount -- <tokenizer.json> <manifest.jsonl> [max_rows]`

use diffusionblocks::bpe::BpeTokenizer;
use std::io::BufRead;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        anyhow::bail!("usage: tokcount <tokenizer.json> <manifest.jsonl> [max_rows]");
    }
    let text = std::fs::read_to_string(&args[1])?;
    let tok = BpeTokenizer::from_json(&text)?;
    let max_rows: usize = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(usize::MAX);

    let file = std::fs::File::open(&args[2])?;
    let mut rows = 0usize;
    let mut tokens = 0usize;
    let mut over_8k = 0usize;
    let mut failures = 0usize;
    let started = std::time::Instant::now();
    for line in std::io::BufReader::new(file).lines() {
        if rows >= max_rows {
            break;
        }
        let line = line?;
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let doc = format!(
            "{}\n{}",
            v.get("instruction").and_then(|s| s.as_str()).unwrap_or(""),
            v.get("response").and_then(|s| s.as_str()).unwrap_or("")
        );
        match tok.encode(&doc) {
            Ok(ids) => {
                rows += 1;
                tokens += ids.len();
                if ids.len() > 8192 {
                    over_8k += 1;
                }
            }
            Err(e) => {
                failures += 1;
                eprintln!("row encode failed (counted as failure, continuing): {e}");
            }
        }
    }
    println!(
        "rows={rows} tokens={tokens} mean={:.1} over_8k={over_8k} failures={failures} secs={:.1}",
        tokens as f64 / rows.max(1) as f64,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
