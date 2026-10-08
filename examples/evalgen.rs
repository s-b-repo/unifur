use anyhow::{Context, Result};
use diffusionblocks::codegen_eval::{
    self, admit, contamination_risk, split, CorpusIndex, Side, Task, DEFAULT_EVAL_FRACTION,
};
use std::path::PathBuf;

/// One task in the catalogue.
struct Spec {
    id: &'static str,
    prompt: &'static str,
    tests: &'static [&'static str],
    language: &'static str,
}

include!("evalgen/catalogue.rs");

fn main() -> Result<()> {
    let mut out = PathBuf::from("/srv/m-sdd/unifur/datasets/repo-native-eval.jsonl");
    let mut corpus: Option<PathBuf> = None;
    let mut limit = usize::MAX;
    let mut all = false;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out = PathBuf::from(args.get(i + 1).context("--out needs a path")?);
                i += 2;
            }
            "--corpus" => {
                corpus = Some(PathBuf::from(
                    args.get(i + 1).context("--corpus needs a path")?,
                ));
                i += 2;
            }
            "--limit" => {
                limit = args.get(i + 1).context("--limit needs a count")?.parse()?;
                i += 2;
            }
            "--all" => {
                all = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!(
                    "usage: evalgen [--out <path>] [--corpus <sft.jsonl>] [--limit <n>] [--all]\n\
                     \n--all  write every task, not only the eval-side half"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }

    let tasks: Vec<Task> = CATALOGUE
        .iter()
        .take(limit)
        .map(|s| Task {
            id: s.id.to_string(),
            prompt: s.prompt.to_string(),
            target: "src/adhoc.rs".to_string(),
            tests: s.tests.iter().map(|t| (*t).to_string()).collect(),
            source: "repo-native".to_string(),
            language: s.language.to_string(),
        })
        .collect();

    println!("{} repo-native tasks in the catalogue", tasks.len());

    match &corpus {
        None => {
            codegen_eval::write_tasks(&out, &tasks)?;
            println!("wrote {} tasks to {}", tasks.len(), out.display());
        }
        Some(corpus_path) => {
            let index = index_from(corpus_path)?;
            println!(
                "index: {} rows, {} distinct 13-grams",
                index.task_count(),
                index.gram_count()
            );
            let report = admit(&tasks, &index);
            println!(
                "admitted {} of {} (yield {:.2})",
                report.scored,
                report.total,
                report.yield_rate()
            );
            for r in &report.rejected {
                println!(
                    "  REJECTED {} overlap={:.3} example={:?}",
                    r.id, r.overlap, r.example
                );
            }
            let at_risk = tasks
                .iter()
                .filter(|t| contamination_risk(&t.source).is_some())
                .count();
            println!("at-risk provenance: {at_risk} (repo-native is clean by construction)");

            let kept: Vec<Task> = if all {
                tasks.clone()
            } else {
                tasks
                    .iter()
                    .filter(|t| split(&t.id, DEFAULT_EVAL_FRACTION) == Side::Eval)
                    .cloned()
                    .collect()
            };
            codegen_eval::write_tasks(&out, &kept)?;
            let tests: usize = kept.iter().map(|t| t.tests.len()).sum();
            println!(
                "wrote {} tasks ({} test functions) to {}",
                kept.len(),
                tests,
                out.display()
            );
        }
    }
    Ok(())
}

/// Index over an SFT file, or a stand-in when it is absent.
///
/// An absent corpus is not a licence to skip the check: the gate must run
/// before the corpus exists. The stand-in is a single row whose text says the
/// corpus was unavailable, which is long enough to produce n-grams and so does
/// not silently make every task look uncontaminated.
fn index_from(path: &std::path::Path) -> Result<CorpusIndex> {
    if path.exists() {
        codegen_eval::index_sft(path)
    } else {
        println!(
            "corpus {} absent; decontamination is unverified",
            path.display()
        );
        Ok(CorpusIndex::build([(
            "absent",
            "the training corpus was not available so nothing can be shown uncontaminated here",
        )]))
    }
}
