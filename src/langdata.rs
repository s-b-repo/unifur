//! The per-language teacher-trace data pipeline.
//!
//! This module turns seeded coding tasks into filtered training data, one
//! directory per [`CodeLanguage`]:
//!
//! 1. **Seeds** ([`build_seeds`]) — deterministic task prompts per language.
//!    Up to three quarters of them come from the local Magicoder OSS-Instruct
//!    JSONL (real task variety, offline); the rest come from a per-language
//!    template bank covering four task kinds (implement, bug-fix, refactor,
//!    test-writing). Every prompt, whatever its origin, ends with
//!    [`PROMPT_REQUIREMENTS`]: the teacher is told to answer with idiomatic,
//!    lint-clean, tested code and is explicitly forbidden the suppression
//!    patterns [`crate::cheat`] scans for — the filter then measures how well
//!    the teacher complied.
//! 2. **Traces** — with a [`NimClient`], prompts go to the teacher model via
//!    [`NimClient::generate_traces`] (which is itself resumable). With
//!    `None`, the pipeline runs keyless: Magicoder-sourced seeds reuse their
//!    dataset solutions and template seeds use hand-written reference
//!    completions, all marked `source: "offline"`.
//! 3. **Cheat filter** — [`crate::cheat::filter_traces`] per language.
//!    Reject-grade traces (suppression, eval gaming) are dropped; what
//!    happens to quarantine-grade traces (stubs, broken snippets) is the
//!    [`CheatPolicy`]. Every drop lands in `rejects.jsonl` with its classes.
//! 4. **Decontamination** — each surviving trace's prompt is checked against
//!    the n-gram index of the repo-native eval ([`crate::codegen_eval`]), so
//!    a teacher trace can never be a near-duplicate of an eval item.
//! 5. **Outputs** — per language: `train.jsonl`, `rejects.jsonl`,
//!    `report.json`; plus a top-level `report.json`. Ids are deterministic
//!    functions of (language, seed, task), and a language whose `report.json`
//!    already exists is skipped, so a rerun resumes where it stopped.
//!
//! The offline path is the tested default; the teacher path is one
//! `Option::Some` away and shares every stage after generation.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cheat::{filter_traces, scan_snippet, Severity, SnippetLanguage};
use crate::codegen_eval::{self, CorpusIndex};
use crate::nim::{NimClient, TraceBudget, TracePrompt, TraceRecord};
use crate::student::CodeLanguage;

/// Default pipeline output root, on the same external drive as
/// [`crate::nim::DEFAULT_TRACE_DIR`]. Tests always override this with a
/// unique `/tmp` directory.
pub const DEFAULT_OUT_DIR: &str = "/srv/m-sda/unifur/teacher-traces/pipeline";

/// Default Magicoder OSS-Instruct corpus: the offline prompt source and the
/// fallback completion corpus for keyless runs.
pub const DEFAULT_OSS_INSTRUCT_PATH: &str =
    "/srv/m-sdd/unifur/datasets/magicoder-oss-75k/data-oss_instruct-decontaminated.jsonl";

/// Default repo-native eval tasks, written by `examples/evalgen.rs`. This is
/// the corpus teacher traces are decontaminated against.
pub const DEFAULT_EVAL_TASKS_PATH: &str = "/srv/m-sdd/unifur/datasets/repo-native-eval.jsonl";

/// Default number of task seeds built per language.
pub const DEFAULT_TASKS_PER_LANGUAGE: usize = 32;

/// Default cap on how much of a Magicoder problem statement goes into a
/// prompt. Some rows carry page-long contest statements; the prompt keeps the
/// head so one row cannot dominate the token budget.
pub const DEFAULT_MAX_PROBLEM_CHARS: usize = 2_000;

/// Default base seed for the pipeline. Everything downstream — row shuffles,
/// per-task seeds, trace ids — is a pure function of this value, so a rerun
/// with the same config reproduces the same data exactly.
pub const DEFAULT_BASE_SEED: u64 = 0x6472_6163_6530_3031;

/// The anti-suppression contract appended to every task prompt. Naming the
/// forbidden patterns here is deliberate: the cheat filter then measures
/// compliance rather than guessing intent, and a teacher that emits them
/// anyway produces the reject-rate signal the report carries.
pub const PROMPT_REQUIREMENTS: &str = "\
Hard requirements for the answer:
- No lint or diagnostic suppression: no `#[allow]`, no `#![allow]`, no `# noqa`, \
no `# type: ignore`, no `@ts-ignore` / `@ts-nocheck` / `eslint-disable`, no `NOLINT`, \
and no diagnostic-disabling `#pragma`. If a linter would complain, fix the code, not the lint.
- No stubs: no `todo!()` or `unimplemented!()`, no `raise NotImplementedError`, no `pass` \
bodies, no `throw new Error(\"not implemented\")`, no empty bodies, no commented-out tests.
- No special-casing: do not branch on the literal values from the examples; solve the \
general problem.
- Include real tests that assert observable behaviour (for HTML/CSS, a structured \
review checklist in a comment).
- Return complete, self-contained code.";

/// Follow-up lines that keep repeated template tasks distinct. The `n`-th
/// reuse of a template gets `TWISTS[n % TWISTS.len()]` appended, so template
/// prompts (and therefore ids) stay unique without changing the task.
const TWISTS: &[&str] = &[
    "",
    "\n\nPrioritise readability over cleverness.",
    "\n\nHandle empty and single-element inputs explicitly.",
    "\n\nDocument each edge case you considered in a brief comment.",
    "\n\nKeep the public interface exactly as given and add a short doc comment.",
    "\n\nMake invalid input fail loudly and clearly, never silently.",
];

// ---------------------------------------------------------------------------
// Language mapping
// ---------------------------------------------------------------------------

/// The [`SnippetLanguage`] the cheat scanner should use for a
/// [`CodeLanguage`]. `Web` maps to `HtmlCss`, the scanner's web family. The
/// scanner's families are coarser than the language list: Go, Java and C#
/// scan under the C-family rules (`//` and `/* */` comments, the NOLINT
/// check, the language-mixing tally), which is the family whose comment and
/// suppression idioms theirs resemble. PHP scans as JsTs: its strict
/// comparison operators are the JsTs family's `===`/`!==` forms, its
/// `function` keyword is a JsTs marker (so the mixing tally reads PHP
/// correctly), and it shares the `//` comment idiom.
pub fn snippet_language(lang: CodeLanguage) -> SnippetLanguage {
    match lang {
        CodeLanguage::Rust => SnippetLanguage::Rust,
        CodeLanguage::Python => SnippetLanguage::Python,
        CodeLanguage::C => SnippetLanguage::C,
        CodeLanguage::Cpp => SnippetLanguage::Cpp,
        CodeLanguage::JsTs => SnippetLanguage::JsTs,
        CodeLanguage::Web => SnippetLanguage::HtmlCss,
        CodeLanguage::Go => SnippetLanguage::C,
        CodeLanguage::Java | CodeLanguage::CSharp => SnippetLanguage::Cpp,
        CodeLanguage::Php => SnippetLanguage::JsTs,
    }
}

/// The reverse of [`snippet_language`] for the shipped six. The scanner's
/// families are coarser than the language list (four languages scan under
/// the C families — see [`snippet_language`]), so a family cannot name one
/// of those four; [`SnippetLanguage::Other`] likewise maps to `None` rather
/// than guessing.
pub fn code_language(lang: SnippetLanguage) -> Option<CodeLanguage> {
    match lang {
        SnippetLanguage::Rust => Some(CodeLanguage::Rust),
        SnippetLanguage::Python => Some(CodeLanguage::Python),
        SnippetLanguage::C => Some(CodeLanguage::C),
        SnippetLanguage::Cpp => Some(CodeLanguage::Cpp),
        SnippetLanguage::JsTs => Some(CodeLanguage::JsTs),
        SnippetLanguage::HtmlCss => Some(CodeLanguage::Web),
        SnippetLanguage::Other => None,
    }
}

/// Human-readable name used inside prompt text (`"Rust"`, `"C++"`, …).
pub fn display_name(lang: CodeLanguage) -> &'static str {
    match lang {
        CodeLanguage::Rust => "Rust",
        CodeLanguage::Python => "Python",
        CodeLanguage::C => "C",
        CodeLanguage::Cpp => "C++",
        CodeLanguage::JsTs => "TypeScript",
        CodeLanguage::Web => "HTML/CSS",
        CodeLanguage::Go => "Go",
        CodeLanguage::Java => "Java",
        CodeLanguage::CSharp => "C#",
        CodeLanguage::Php => "PHP",
    }
}

/// Magicoder `lang` values admitted for each [`CodeLanguage`]. The corpus
/// spells C++ as `cpp` and has no plain-C or web rows for most languages —
/// an empty match set is normal and the template bank covers the gap.
fn magicoder_lang_keys(lang: CodeLanguage) -> &'static [&'static str] {
    match lang {
        CodeLanguage::Rust => &["rust"],
        CodeLanguage::Python => &["python"],
        CodeLanguage::C => &["c"],
        CodeLanguage::Cpp => &["cpp", "c++"],
        CodeLanguage::JsTs => &["typescript", "javascript"],
        CodeLanguage::Web => &["html", "css"],
        CodeLanguage::Go => &["go", "golang"],
        CodeLanguage::Java => &["java"],
        CodeLanguage::CSharp => &["csharp", "c#"],
        CodeLanguage::Php => &["php"],
    }
}

// ---------------------------------------------------------------------------
// Task seeds
// ---------------------------------------------------------------------------

/// The four shapes of task the pipeline asks the teacher for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKind {
    /// Write a small function from a specification.
    Implement,
    /// Fix a defect in shown code.
    BugFix,
    /// Restructure shown code without changing behaviour.
    Refactor,
    /// Write tests for shown code.
    TestWriting,
}

impl TaskKind {
    /// Every kind, in rotation order.
    pub const ALL: [Self; 4] = [
        Self::Implement,
        Self::BugFix,
        Self::Refactor,
        Self::TestWriting,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Implement => "implement",
            Self::BugFix => "bugfix",
            Self::Refactor => "refactor",
            Self::TestWriting => "test-writing",
        }
    }
}

/// Where a seed's task (and its offline reference completion) came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SeedSource {
    /// A Magicoder OSS-Instruct row.
    Magicoder,
    /// The built-in per-language template bank.
    Template,
}

impl SeedSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Magicoder => "magicoder",
            Self::Template => "template",
        }
    }
}

/// One seeded coding task: the prompt handed to the teacher, plus the
/// reference completion used when the pipeline runs offline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSeed {
    /// Stable id (`{lang}-oss-{row}` or `{lang}-tpl-{kind}-{n}`). Deterministic
    /// for a given config, which is what makes pipeline reruns resumable.
    pub id: String,
    /// Language the answer must be in.
    pub language: CodeLanguage,
    /// Which shape of task this is.
    pub kind: TaskKind,
    /// The full prompt, ending with [`PROMPT_REQUIREMENTS`].
    pub prompt: String,
    /// Offline completion (Magicoder solution or template reference). Unused
    /// when a teacher client is attached.
    pub reference: String,
    /// Per-task seed, recorded on every trace for reproduction.
    pub seed: u64,
    /// Provenance of the task.
    pub source: SeedSource,
}

/// One row of the Magicoder OSS-Instruct JSONL. Unknown fields are ignored;
/// a missing `index` falls back to the row's position.
#[derive(Deserialize)]
struct OssRow {
    #[serde(default)]
    lang: String,
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    problem: String,
    #[serde(default)]
    solution: String,
}

/// Load the Magicoder rows for `language`. A missing or unreadable corpus is
/// not fatal — the template bank covers every language — but it is reported
/// loudly on stderr and visible in the report as `magicoder_seeds: 0`.
fn load_oss_rows(oss_path: &Path, language: CodeLanguage) -> Vec<(u64, String, String)> {
    let text = match std::fs::read_to_string(oss_path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "[langdata] Magicoder corpus {} not readable ({err}); \
                 {} seeds will come from the template bank only",
                oss_path.display(),
                language.name()
            );
            return Vec::new();
        }
    };
    let keys = magicoder_lang_keys(language);
    let mut rows = Vec::new();
    for (position, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: OssRow = match serde_json::from_str(line) {
            Ok(row) => row,
            Err(err) => {
                eprintln!(
                    "[langdata] skipping unparseable Magicoder row {}: {err}",
                    position + 1
                );
                continue;
            }
        };
        if !keys.contains(&row.lang.as_str()) || row.problem.trim().is_empty() {
            continue;
        }
        let index = row.index.unwrap_or(position as u64);
        rows.push((index, row.problem, row.solution));
    }
    rows
}

/// The code inside a Magicoder `solution`: the contents of its fenced blocks,
/// or the whole text when no fences are present. Fence lines themselves are
/// markup, not training code.
fn extract_code(solution: &str) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    for line in solution.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            out.push_str(line);
            out.push('\n');
        }
    }
    if out.trim().is_empty() {
        solution.to_string()
    } else {
        out
    }
}

/// The uniform prompt shape: language header, task body, requirements. Every
/// seed — Magicoder or template — goes through this, so every teacher answer
/// is conditioned on the same anti-suppression contract.
fn finish_prompt(language: CodeLanguage, body: &str) -> String {
    format!(
        "Language: {}\n\n{}\n\n{}",
        display_name(language),
        body.trim(),
        PROMPT_REQUIREMENTS
    )
}

/// Build `count` deterministic task seeds for `language`.
///
/// Up to `3 * count / 4` seeds come from the Magicoder corpus at `oss_path`
/// (rows matching the language, shuffled with a seeded RNG); the rest cycle
/// through the four [`TaskKind`]s of the template bank, with `TWISTS`
/// keeping repeats distinct. Same inputs give byte-identical seeds, which is
/// what the pipeline's resume contract rests on.
pub fn build_seeds(
    language: CodeLanguage,
    count: usize,
    base_seed: u64,
    oss_path: &Path,
    max_problem_chars: usize,
) -> anyhow::Result<Vec<TaskSeed>> {
    ensure!(
        count > 0,
        "build_seeds needs at least one task for {}",
        language.name()
    );
    ensure!(
        max_problem_chars > 0,
        "build_seeds needs a positive max_problem_chars, got {max_problem_chars}"
    );

    let mut rows = load_oss_rows(oss_path, language);
    // A language-keyed stream so two languages never share a shuffle, and a
    // config change for one language cannot reshuffle another's seeds.
    let stream = base_seed ^ (language.index() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut rng = rand::rngs::StdRng::seed_from_u64(stream);
    for i in (1..rows.len()).rev() {
        let j = rng.random_range(0..=i);
        rows.swap(i, j);
    }

    let wanted_magicoder = (3 * count / 4).min(rows.len());
    let mut seeds = Vec::with_capacity(count);
    for (n, (index, problem, solution)) in rows.into_iter().take(wanted_magicoder).enumerate() {
        let truncated: String = problem.chars().take(max_problem_chars).collect();
        let body = format!(
            "{}\n\nWrite the solution in {}.",
            truncated.trim(),
            display_name(language)
        );
        seeds.push(TaskSeed {
            id: format!("{}-oss-{index:06}", language.name()),
            language,
            kind: TaskKind::Implement,
            prompt: finish_prompt(language, &body),
            reference: extract_code(&solution),
            seed: task_seed(base_seed, language, n),
            source: SeedSource::Magicoder,
        });
    }

    let mut template_n = 0usize;
    while seeds.len() < count {
        let kind = TaskKind::ALL[template_n % TaskKind::ALL.len()];
        let (body, reference) = template(language, kind);
        let twist = TWISTS[(template_n / TaskKind::ALL.len()) % TWISTS.len()];
        seeds.push(TaskSeed {
            id: format!("{}-tpl-{}-{template_n:03}", language.name(), kind.as_str()),
            language,
            kind,
            prompt: finish_prompt(language, &format!("{body}{twist}")),
            reference: reference.to_string(),
            seed: task_seed(base_seed, language, wanted_magicoder + template_n),
            source: SeedSource::Template,
        });
        template_n += 1;
    }
    Ok(seeds)
}

/// The per-task seed: a pure function of the pipeline seed, the language and
/// the task's position, so trace ids are reproducible across runs.
fn task_seed(base_seed: u64, language: CodeLanguage, n: usize) -> u64 {
    base_seed
        .wrapping_add((language.index() as u64).wrapping_mul(1_000_003))
        .wrapping_add(n as u64)
}

// ---------------------------------------------------------------------------
// Template bank: one (task, reference completion) per language per kind.
//
// Every reference completion is written to pass the cheat filter it will be
// scanned with: no literal-special-cased branches, no vacuous assertions,
// balanced delimiters, tests that assert real behaviour.
// ---------------------------------------------------------------------------

const RUST_IMPLEMENT_TASK: &str = "\
Implement in Rust:

    pub fn word_frequency(text: &str) -> Vec<(String, usize)>

Return the words of `text` (lowercased, split on non-alphanumeric characters) paired \
with their counts, sorted by count descending and then alphabetically. Empty input \
yields an empty vector.";

const RUST_IMPLEMENT_REF: &str = r#"pub fn word_frequency(text: &str) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        *counts.entry(word.to_lowercase()).or_insert(0) += 1;
    }
    let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_orders_words() {
        let freq = word_frequency("b a b c a b");
        assert_eq!(
            freq,
            vec![
                ("b".to_string(), 3),
                ("a".to_string(), 2),
                ("c".to_string(), 1),
            ]
        );
    }

    #[test]
    fn empty_text_yields_no_words() {
        assert!(word_frequency("").is_empty());
    }

    #[test]
    fn case_and_punctuation_are_normalised() {
        assert_eq!(word_frequency("Hello, hello! HELLO"), vec![("hello".to_string(), 3)]);
    }
}
"#;

const RUST_BUGFIX_TASK: &str = r#"The following Rust function is meant to return the running total of its \
input, but it drops the first element and returns nonsense for short inputs:

```rust
pub fn running_total(xs: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    for i in 1..xs.len() {
        let prev = if i == 1 { 0 } else { out[i - 2] };
        out.push(prev + xs[i]);
    }
    out
}
```

Find the defects, fix them, and add tests that would have caught them."#;

const RUST_BUGFIX_REF: &str = r#"pub fn running_total(xs: &[i64]) -> Vec<i64> {
    let mut total = 0i64;
    let mut out = Vec::with_capacity(xs.len());
    for &x in xs {
        total += x;
        out.push(total);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_in_order() {
        assert_eq!(running_total(&[1, 2, 3, 4]), vec![1, 3, 6, 10]);
    }

    #[test]
    fn empty_input_stays_empty() {
        assert!(running_total(&[]).is_empty());
    }

    #[test]
    fn single_element_is_its_own_total() {
        assert_eq!(running_total(&[7]), vec![7]);
    }

    #[test]
    fn negatives_are_handled() {
        assert_eq!(running_total(&[3, -5, 2]), vec![3, -2, 0]);
    }
}
"#;

const RUST_REFACTOR_TASK: &str = r#"The following Rust function works but is written in an un-idiomatic, \
index-heavy style:

```rust
pub fn even_squares(xs: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < xs.len() {
        let x = xs[i];
        if x % 2 == 0 {
            out.push(x * x);
        }
        i += 1;
    }
    out
}
```

Refactor it to iterator combinators without changing its behaviour, and add tests \
that pin the behaviour."#;

const RUST_REFACTOR_REF: &str = r#"pub fn even_squares(xs: &[i64]) -> Vec<i64> {
    xs.iter().filter(|x| *x % 2 == 0).map(|x| x * x).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_squares_of_evens_only() {
        assert_eq!(even_squares(&[1, 2, 3, 4]), vec![4, 16]);
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(even_squares(&[]).is_empty());
    }

    #[test]
    fn odd_input_gives_empty_output() {
        assert!(even_squares(&[1, 3, 5]).is_empty());
    }
}
"#;

const RUST_TESTWRITING_TASK: &str = r#"Write a thorough `#[cfg(test)]` module for the following Rust function \
without modifying the function itself:

```rust
pub fn median(sorted: &[i64]) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid] as f64)
    } else {
        Some((sorted[mid - 1] + sorted[mid]) as f64 / 2.0)
    }
}
```

Cover the empty input, odd and even lengths, and negative values."#;

const RUST_TESTWRITING_REF: &str = r#"pub fn median(sorted: &[i64]) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid] as f64)
    } else {
        Some((sorted[mid - 1] + sorted[mid]) as f64 / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_has_no_median() {
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn odd_length_takes_the_middle_element() {
        assert_eq!(median(&[1, 3, 5]), Some(3.0));
    }

    #[test]
    fn even_length_averages_the_middle_pair() {
        assert_eq!(median(&[1, 2, 3, 4]), Some(2.5));
    }

    #[test]
    fn negatives_are_handled() {
        assert_eq!(median(&[-5, -1, 2]), Some(-1.0));
    }
}
"#;

const PYTHON_IMPLEMENT_TASK: &str = "\
Implement in Python:

    def top_k_words(text: str, k: int) -> list[tuple[str, int]]

Return the `k` most frequent words of `text` (lowercased, split on whitespace), \
ordered by count descending and then alphabetically. A non-positive `k` yields an \
empty list.";

const PYTHON_IMPLEMENT_REF: &str = r#"def top_k_words(text: str, k: int) -> list[tuple[str, int]]:
    """Return the `k` most frequent words of `text`, ties broken alphabetically."""
    counts: dict[str, int] = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if k <= 0:
        return []
    return ordered[:k]


import unittest


class TopKWordsTests(unittest.TestCase):
    def test_orders_by_count_then_alphabet(self):
        self.assertEqual(top_k_words("b a b c a b", 2), [("b", 3), ("a", 2)])

    def test_k_larger_than_vocabulary_returns_everything(self):
        self.assertEqual(top_k_words("x y", 10), [("x", 1), ("y", 1)])

    def test_non_positive_k_returns_empty(self):
        self.assertEqual(top_k_words("a a a", 0), [])


if __name__ == "__main__":
    unittest.main()
"#;

const PYTHON_BUGFIX_TASK: &str = r#"The following Python function is meant to remove duplicates while keeping \
the first occurrence of each item, but it keeps the *last* occurrence instead:

```python
def dedupe(items):
    out = items[:]
    for item in items:
        while out.count(item) > 1:
            out.remove(item)
    return out
```

`list.remove` deletes the first matching element, so `dedupe([1, 2, 1])` returns \
`[2, 1]`. Fix the function and add tests that pin first-seen order."#;

const PYTHON_BUGFIX_REF: &str = r#"def dedupe(items):
    """Return the items with duplicates removed, preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


import unittest


class DedupeTests(unittest.TestCase):
    def test_keeps_first_occurrence(self):
        self.assertEqual(dedupe([1, 2, 1]), [1, 2])

    def test_empty_input(self):
        self.assertEqual(dedupe([]), [])

    def test_already_unique_input_is_unchanged(self):
        self.assertEqual(dedupe([3, 1, 2]), [3, 1, 2])


if __name__ == "__main__":
    unittest.main()
"#;

const PYTHON_REFACTOR_TASK: &str = r#"The following Python function works but is written as a manual \
accumulation loop:

```python
def total_price(cart):
    total = 0
    for item in cart:
        price = item["price"] * item["qty"]
        total = total + price
    return total
```

Refactor it to a single expression over a generator, add type hints, and add tests."#;

const PYTHON_REFACTOR_REF: &str = r#"def total_price(cart: list[dict[str, float]]) -> float:
    """Sum `price * qty` over every line item in `cart`."""
    return sum(item["price"] * item["qty"] for item in cart)


import unittest


class TotalPriceTests(unittest.TestCase):
    def test_sums_line_items(self):
        cart = [{"price": 2.5, "qty": 4}, {"price": 1.0, "qty": 1}]
        self.assertEqual(total_price(cart), 11.0)

    def test_empty_cart_is_zero(self):
        self.assertEqual(total_price([]), 0)


if __name__ == "__main__":
    unittest.main()
"#;

const PYTHON_TESTWRITING_TASK: &str = r#"Write a thorough `unittest` test suite for the following Python function \
without modifying the function itself:

```python
def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for i in range(len(values) - window + 1):
        out.append(sum(values[i:i + window]) / window)
    return out
```

Cover the window-validation error, a window of one, a full-width window, and an \
input shorter than the window."#;

const PYTHON_TESTWRITING_REF: &str = r#"def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for i in range(len(values) - window + 1):
        out.append(sum(values[i:i + window]) / window)
    return out


import unittest


class MovingAverageTests(unittest.TestCase):
    def test_rejects_a_non_positive_window(self):
        with self.assertRaises(ValueError):
            moving_average([1.0, 2.0], 0)

    def test_window_of_one_returns_the_input(self):
        self.assertEqual(moving_average([1.0, 2.0, 3.0], 1), [1.0, 2.0, 3.0])

    def test_full_width_window_is_the_plain_average(self):
        self.assertEqual(moving_average([2.0, 4.0, 6.0], 3), [4.0])

    def test_window_wider_than_input_yields_empty(self):
        self.assertEqual(moving_average([1.0], 5), [])


if __name__ == "__main__":
    unittest.main()
"#;

const C_IMPLEMENT_TASK: &str = "\
Implement in C:

    size_t count_words(const char *text)

Return the number of words in `text`, where a word is a maximal run of \
non-whitespace characters (use `isspace` from `<ctype.h>`). An empty or \
all-whitespace string has zero words. Add a test routine that exercises it with \
`assert` from `<assert.h>`.";

const C_IMPLEMENT_REF: &str = r#"#include <assert.h>
#include <ctype.h>
#include <stddef.h>

size_t count_words(const char *text) {
    size_t words = 0;
    int in_word = 0;
    for (const char *p = text; *p != '\0'; ++p) {
        if (isspace((unsigned char)*p)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            ++words;
        }
    }
    return words;
}

static void test_count_words(void) {
    assert(count_words("") == 0);
    assert(count_words("hello") == 1);
    assert(count_words("  one  two three ") == 3);
    assert(count_words("a\tb\nc") == 3);
}

int main(void) {
    test_count_words();
    return 0;
}
"#;

const C_BUGFIX_TASK: &str = r#"The following C function is meant to clamp every element of `xs` into \
`[lo, hi]`, but it reads and writes one element past the end of the array:

```c
void clamp_all(int *xs, size_t n, int lo, int hi) {
    for (size_t i = 0; i <= n; ++i) {
        if (xs[i] < lo) xs[i] = lo;
        if (xs[i] > hi) xs[i] = hi;
    }
}
```

Fix the bounds defect and add `assert`-based tests, including the empty array."#;

const C_BUGFIX_REF: &str = r#"#include <assert.h>
#include <stddef.h>

void clamp_all(int *xs, size_t n, int lo, int hi) {
    for (size_t i = 0; i < n; ++i) {
        if (xs[i] < lo) {
            xs[i] = lo;
        }
        if (xs[i] > hi) {
            xs[i] = hi;
        }
    }
}

static void test_clamp_all(void) {
    int xs[] = { -5, 0, 3, 9, 12 };
    clamp_all(xs, 5, 0, 10);
    assert(xs[0] == 0);
    assert(xs[1] == 0);
    assert(xs[2] == 3);
    assert(xs[3] == 9);
    assert(xs[4] == 10);
    clamp_all(xs, 0, 0, 10);
}

int main(void) {
    test_clamp_all();
    return 0;
}
"#;

const C_REFACTOR_TASK: &str = r#"The following C code duplicates the same three-line swap in several places:

```c
void sort_pair(int *a, int *b) {
    if (*a > *b) {
        int t = *a;
        *a = *b;
        *b = t;
    }
}

void sort_triple(int *a, int *b, int *c) {
    if (*a > *b) {
        int t = *a; *a = *b; *b = t;
    }
    if (*b > *c) {
        int t = *b; *b = *c; *c = t;
    }
    if (*a > *b) {
        int t = *a; *a = *b; *b = t;
    }
}
```

Refactor it so the swap exists exactly once, without changing behaviour, and add \
`assert`-based tests."#;

const C_REFACTOR_REF: &str = r#"#include <assert.h>

static void swap_ints(int *x, int *y) {
    int t = *x;
    *x = *y;
    *y = t;
}

void sort_pair(int *a, int *b) {
    if (*a > *b) {
        swap_ints(a, b);
    }
}

void sort_triple(int *a, int *b, int *c) {
    if (*a > *b) {
        swap_ints(a, b);
    }
    if (*b > *c) {
        swap_ints(b, c);
    }
    if (*a > *b) {
        swap_ints(a, b);
    }
}

static void test_sorting(void) {
    int a = 3;
    int b = 1;
    int c = 2;
    sort_pair(&a, &b);
    assert(a == 1 && b == 3);
    a = 3;
    b = 1;
    sort_triple(&a, &b, &c);
    assert(a == 1 && b == 2 && c == 3);
}

int main(void) {
    test_sorting();
    return 0;
}
"#;

const C_TESTWRITING_TASK: &str = r#"Write a thorough `assert`-based test routine for the following C function \
without modifying the function itself:

```c
int gcd_int(int a, int b) {
    if (a < 0) {
        a = -a;
    }
    if (b < 0) {
        b = -b;
    }
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}
```

Cover coprime inputs, a zero argument, negative arguments, and equal arguments."#;

const C_TESTWRITING_REF: &str = r#"#include <assert.h>

int gcd_int(int a, int b) {
    if (a < 0) {
        a = -a;
    }
    if (b < 0) {
        b = -b;
    }
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static void test_gcd_int(void) {
    assert(gcd_int(12, 18) == 6);
    assert(gcd_int(7, 13) == 1);
    assert(gcd_int(0, 5) == 5);
    assert(gcd_int(-12, 18) == 6);
    assert(gcd_int(9, 9) == 9);
}

int main(void) {
    test_gcd_int();
    return 0;
}
"#;

const CPP_IMPLEMENT_TASK: &str = "\
Implement in C++:

    std::vector<int> prefix_max(const std::vector<int>& xs)

Return a vector of the same length where element `i` is the maximum of \
`xs[0..=i]`. Empty input yields an empty vector. Add a test routine using \
`assert` from `<cassert>`.";

const CPP_IMPLEMENT_REF: &str = r#"#include <algorithm>
#include <cassert>
#include <vector>

std::vector<int> prefix_max(const std::vector<int>& xs) {
    std::vector<int> out;
    out.reserve(xs.size());
    int best = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        best = (i == 0) ? xs[i] : std::max(best, xs[i]);
        out.push_back(best);
    }
    return out;
}

static void test_prefix_max() {
    assert((prefix_max({1, 3, 2, 5, 4}) == std::vector<int>{1, 3, 3, 5, 5}));
    assert(prefix_max({}).empty());
    assert((prefix_max({-2, -5, -1}) == std::vector<int>{-2, -2, -1}));
}

int main() {
    test_prefix_max();
    return 0;
}
"#;

const CPP_BUGFIX_TASK: &str = r#"The following C++ function is meant to erase every negative element, but \
erasing through a loop iterator invalidates that iterator, so the loop has \
undefined behaviour and skips elements:

```cpp
void drop_negatives(std::vector<int>& xs) {
    for (auto it = xs.begin(); it != xs.end(); ++it) {
        if (*it < 0) {
            xs.erase(it);
        }
    }
}
```

Fix it with the erase-remove idiom and add `assert`-based tests."#;

const CPP_BUGFIX_REF: &str = r#"#include <algorithm>
#include <cassert>
#include <vector>

void drop_negatives(std::vector<int>& xs) {
    xs.erase(std::remove_if(xs.begin(), xs.end(), [](int x) { return x < 0; }), xs.end());
}

static void test_drop_negatives() {
    std::vector<int> xs{1, -2, 3, -4, 5};
    drop_negatives(xs);
    assert((xs == std::vector<int>{1, 3, 5}));
    std::vector<int> all_negative{-1, -2};
    drop_negatives(all_negative);
    assert(all_negative.empty());
    std::vector<int> empty;
    drop_negatives(empty);
    assert(empty.empty());
}

int main() {
    test_drop_negatives();
    return 0;
}
"#;

const CPP_REFACTOR_TASK: &str = r#"The following C++ function works but is written with manual index loops:

```cpp
std::vector<std::string> shout(const std::vector<std::string>& words) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < words.size(); ++i) {
        std::string w = words[i];
        for (std::size_t j = 0; j < w.size(); ++j) {
            w[j] = static_cast<char>(std::toupper(static_cast<unsigned char>(w[j])));
        }
        out.push_back(w);
    }
    return out;
}
```

Refactor it to use `std::transform` and range-based loops without changing \
behaviour, and add `assert`-based tests."#;

const CPP_REFACTOR_REF: &str = r#"#include <algorithm>
#include <cassert>
#include <cctype>
#include <string>
#include <vector>

std::vector<std::string> shout(const std::vector<std::string>& words) {
    std::vector<std::string> out;
    out.reserve(words.size());
    std::transform(words.begin(), words.end(), std::back_inserter(out), [](std::string w) {
        for (char& c : w) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        return w;
    });
    return out;
}

static void test_shout() {
    assert((shout({"one", "Two"}) == std::vector<std::string>{"ONE", "TWO"}));
    assert(shout({}).empty());
    assert((shout({"a1!"}) == std::vector<std::string>{"A1!"}));
}

int main() {
    test_shout();
    return 0;
}
"#;

const CPP_TESTWRITING_TASK: &str = r#"Write a thorough `assert`-based test routine for the following C++ \
function without modifying the function itself:

```cpp
double average(const std::vector<double>& xs) {
    if (xs.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (double x : xs) {
        total += x;
    }
    return total / static_cast<double>(xs.size());
}
```

Floating-point results must be compared with a tolerance helper, not `==`."#;

const CPP_TESTWRITING_REF: &str = r#"#include <cassert>
#include <cmath>
#include <vector>

double average(const std::vector<double>& xs) {
    if (xs.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (double x : xs) {
        total += x;
    }
    return total / static_cast<double>(xs.size());
}

static bool nearly(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

static void test_average() {
    assert(nearly(average({}), 0.0));
    assert(nearly(average({2.0, 4.0, 6.0}), 4.0));
    assert(nearly(average({-1.0, 1.0}), 0.0));
    assert(nearly(average({0.1, 0.2}), 0.15));
}

int main() {
    test_average();
    return 0;
}
"#;

const JSTS_IMPLEMENT_TASK: &str = "\
Implement in TypeScript:

    export function groupByLength(words: string[]): Map<number, string[]>

Group the words by their length, preserving input order within each group. Empty \
input yields an empty map. Add tests using `node:test` and `node:assert/strict`.";

const JSTS_IMPLEMENT_REF: &str = r#"import test from "node:test";
import assert from "node:assert/strict";

export function groupByLength(words: string[]): Map<number, string[]> {
    const groups = new Map<number, string[]>();
    for (const word of words) {
        const bucket = groups.get(word.length);
        if (bucket === undefined) {
            groups.set(word.length, [word]);
        } else {
            bucket.push(word);
        }
    }
    return groups;
}

test("groups words by length preserving order", () => {
    const groups = groupByLength(["to", "be", "or", "not"]);
    assert.deepEqual(groups.get(2), ["to", "be", "or"]);
    assert.deepEqual(groups.get(3), ["not"]);
});

test("empty input yields an empty map", () => {
    assert.equal(groupByLength([]).size, 0);
});
"#;

const JSTS_BUGFIX_TASK: &str = r#"The following TypeScript function is meant to return one page of `items`, \
but `Array.prototype.slice` treats its second argument as an exclusive end \
index, so every page comes back one element short:

```ts
export function page<T>(items: T[], pageNumber: number, perPage: number): T[] {
    const start = pageNumber * perPage;
    return items.slice(start, start + perPage - 1);
}
```

Fix the off-by-one, reject nonsensical paging arguments loudly, and add tests \
using `node:test`."#;

const JSTS_BUGFIX_REF: &str = r#"import test from "node:test";
import assert from "node:assert/strict";

export function page<T>(items: T[], pageNumber: number, perPage: number): T[] {
    if (pageNumber < 0 || perPage <= 0) {
        throw new RangeError("pageNumber must be non-negative and perPage positive");
    }
    const start = pageNumber * perPage;
    return items.slice(start, start + perPage);
}

test("returns a full page", () => {
    assert.deepEqual(page([1, 2, 3, 4, 5], 0, 2), [1, 2]);
    assert.deepEqual(page([1, 2, 3, 4, 5], 1, 2), [3, 4]);
});

test("a short last page comes back whole", () => {
    assert.deepEqual(page([1, 2, 3], 1, 2), [3]);
});

test("invalid paging arguments are rejected", () => {
    assert.throws(() => page([1], -1, 2), RangeError);
    assert.throws(() => page([1], 0, 0), RangeError);
});
"#;

const JSTS_REFACTOR_TASK: &str = r#"The following TypeScript function works but is written as an index loop:

```ts
export function adultNames(users: { name: string; age: number }[]): string[] {
    const names: string[] = [];
    for (let i = 0; i < users.length; i++) {
        if (users[i].age >= 18) {
            names.push(users[i].name.toUpperCase());
        }
    }
    names.sort();
    return names;
}
```

Refactor it to a `filter`/`map`/`sort` chain without changing behaviour, and add \
tests using `node:test`."#;

const JSTS_REFACTOR_REF: &str = r#"import test from "node:test";
import assert from "node:assert/strict";

export function adultNames(users: { name: string; age: number }[]): string[] {
    return users
        .filter((user) => user.age >= 18)
        .map((user) => user.name.toUpperCase())
        .sort();
}

test("keeps adults, uppercased and sorted", () => {
    const users = [
        { name: "zoe", age: 34 },
        { name: "amy", age: 17 },
        { name: "bob", age: 52 },
    ];
    assert.deepEqual(adultNames(users), ["BOB", "ZOE"]);
});

test("empty input yields an empty list", () => {
    assert.deepEqual(adultNames([]), []);
});
"#;

const JSTS_TESTWRITING_TASK: &str = r#"Write a thorough test suite using `node:test` and `node:assert/strict` \
for the following TypeScript function without modifying the function itself:

```ts
export function clamp(value: number, lo: number, hi: number): number {
    return Math.min(Math.max(value, lo), hi);
}
```

Cover values below, inside and above the range, plus an inverted range."#;

const JSTS_TESTWRITING_REF: &str = r#"import test from "node:test";
import assert from "node:assert/strict";

export function clamp(value: number, lo: number, hi: number): number {
    return Math.min(Math.max(value, lo), hi);
}

test("a value below the range clamps to the low end", () => {
    assert.equal(clamp(-3, 0, 10), 0);
});

test("a value inside the range passes through", () => {
    assert.equal(clamp(5, 0, 10), 5);
});

test("a value above the range clamps to the high end", () => {
    assert.equal(clamp(42, 0, 10), 10);
});

test("an inverted range resolves to the high argument", () => {
    assert.equal(clamp(5, 10, 0), 0);
});
"#;

const WEB_IMPLEMENT_TASK: &str = "\
Build an accessible site navigation bar as a single HTML file with embedded CSS: \
a skip-to-content link that is visually hidden until focused, a `<nav>` landmark \
with an `aria-label`, links where the current page carries `aria-current=\"page\"`, \
and a layout that stacks vertically below 40rem. End with an HTML comment \
checklist a reviewer can verify by hand.";

const WEB_IMPLEMENT_REF: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Accessible navigation</title>
<style>
.skip-link {
    position: absolute;
    left: -9999px;
    top: 0;
    background: #111;
    color: #fff;
    padding: 0.5rem 1rem;
}
.skip-link:focus {
    left: 0;
}
nav ul {
    display: flex;
    gap: 1rem;
    list-style: none;
    margin: 0;
    padding: 0;
}
nav a[aria-current="page"] {
    font-weight: bold;
    text-decoration: underline;
}
@media (max-width: 40rem) {
    nav ul {
        flex-direction: column;
    }
}
</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to main content</a>
<nav aria-label="Primary">
    <ul>
        <li><a href="/" aria-current="page">Home</a></li>
        <li><a href="/docs">Docs</a></li>
        <li><a href="/about">About</a></li>
    </ul>
</nav>
<main id="main">
    <h1>Page content</h1>
</main>
<!--
  Review checklist:
  - [ ] The skip link is invisible until focused, then appears at the top left.
  - [ ] The nav landmark announces itself as "Primary" to assistive technology.
  - [ ] Exactly one link carries aria-current="page", and it is visually distinct.
  - [ ] Below 40rem the links stack vertically with no horizontal scrolling.
  - [ ] Every interactive element shows a visible focus indicator.
-->
</body>
</html>
"##;

const WEB_BUGFIX_TASK: &str = r#"The following HTML form has three accessibility defects: the labels are not \
associated with their inputs, the email field does not use the email input type, \
and the button relies on an implicit type:

```html
<form>
  <label>Name</label>
  <input type="text" id="name">
  <label>Email</label>
  <input type="text" id="email">
  <button>Send</button>
</form>
```

Fix all three defects and add a short HTML comment explaining what each fix buys \
the user."#;

const WEB_BUGFIX_REF: &str = r##"<!--
  Fixes applied:
  - Each label now names its input with for/id, so clicking the label focuses the
    field and screen readers announce it.
  - The email field uses type="email", giving mobile users the right keyboard and
    the browser a chance to validate the address.
  - The button declares type="submit", so its behaviour no longer depends on the
    implicit default.
-->
<form action="/subscribe" method="post">
    <label for="name">Name</label>
    <input type="text" id="name" name="name" required>
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required>
    <button type="submit">Send</button>
</form>
"##;

const WEB_REFACTOR_TASK: &str = r#"The following HTML repeats the same inline styles on every card:

```html
<div style="border:1px solid #ccc;border-radius:8px;padding:1rem;max-width:20rem">
  <h2 style="margin:0 0 0.5rem">First</h2>
  <p style="margin:0;color:#555">Alpha</p>
</div>
<div style="border:1px solid #ccc;border-radius:8px;padding:1rem;max-width:20rem">
  <h2 style="margin:0 0 0.5rem">Second</h2>
  <p style="margin:0;color:#555">Beta</p>
</div>
```

Refactor it so the styles live once in an embedded stylesheet with named classes, \
without changing the rendered result."#;

const WEB_REFACTOR_REF: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cards</title>
<style>
.card {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 1rem;
    max-width: 20rem;
}
.card-title {
    margin: 0 0 0.5rem;
}
.card-body {
    margin: 0;
    color: #555;
}
</style>
</head>
<body>
<div class="card">
    <h2 class="card-title">First</h2>
    <p class="card-body">Alpha</p>
</div>
<div class="card">
    <h2 class="card-title">Second</h2>
    <p class="card-body">Beta</p>
</div>
</body>
</html>
"##;

const WEB_TESTWRITING_TASK: &str = r#"The following profile-card component ships without any review coverage:

```html
<article>
  <img src="avatar.png">
  <h3>Ada Lovelace</h3>
  <p><span style="color:green"></span> Online</p>
</article>
```

Write the review a careful teammate should have done: a structured HTML-comment \
checklist covering accessibility, semantics and responsive behaviour, followed by \
the component with every issue the checklist names actually fixed."#;

const WEB_TESTWRITING_REF: &str = r##"<!--
  Review checklist for the profile card:
  - [x] The avatar image has a non-empty alt describing the person.
  - [x] Heading levels do not skip; the card fits under the page's h1 as an h2.
  - [x] Colour is not the only signal: the status is written as text.
  - [x] The card remains usable at 320px wide without horizontal scrolling.
  - [x] Interactive elements, if any are added later, show a focus outline.
-->
<article class="profile-card">
    <img src="avatar.png" alt="Portrait of Ada Lovelace">
    <h2>Ada Lovelace</h2>
    <p><span style="color:green" aria-hidden="true"></span> Status: Online</p>
</article>
"##;

const GO_IMPLEMENT_TASK: &str = "\
Implement in Go:

    func WordFrequency(text string) []WordCount

where `WordCount` is a struct pairing a word with its count. Return the words of \
`text` (lowercased, split on whitespace) paired with their counts, sorted by count \
descending and then alphabetically. Empty input yields an empty slice (not nil).";

const GO_IMPLEMENT_REF: &str = r#"package wordcount

import (
	"sort"
	"strings"
	"testing"
)

// WordCount pairs a word with how many times it occurred.
type WordCount struct {
	Word  string
	Count int
}

// WordFrequency returns the words of text, lowercased and split on
// whitespace, paired with their counts and ordered by count descending,
// then alphabetically.
func WordFrequency(text string) []WordCount {
	counts := make(map[string]int)
	for _, word := range strings.Fields(strings.ToLower(text)) {
		counts[word]++
	}
	pairs := make([]WordCount, 0, len(counts))
	for word, count := range counts {
		pairs = append(pairs, WordCount{Word: word, Count: count})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Count != pairs[j].Count {
			return pairs[i].Count > pairs[j].Count
		}
		return pairs[i].Word < pairs[j].Word
	})
	return pairs
}

func TestWordFrequencyOrdersByCountThenAlphabet(t *testing.T) {
	got := WordFrequency("b a b c a b")
	want := []WordCount{{Word: "b", Count: 3}, {Word: "a", Count: 2}, {Word: "c", Count: 1}}
	if len(got) != len(want) {
		t.Fatalf("got %d pairs, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("pair %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestWordFrequencyEmptyInput(t *testing.T) {
	got := WordFrequency("")
	if got == nil {
		t.Error("empty input must yield an empty slice, not nil")
	}
	if len(got) != 0 {
		t.Errorf("empty input yielded %d pairs", len(got))
	}
}

func TestWordFrequencyCaseIsNormalised(t *testing.T) {
	got := WordFrequency("Hello hello HELLO")
	if len(got) != 1 || got[0].Count != 3 || got[0].Word != "hello" {
		t.Errorf("case normalisation failed: %+v", got)
	}
}
"#;

const GO_BUGFIX_TASK: &str = r#"The following Go function is meant to return the running total of its \
input, but it drops the first element and returns nonsense for short inputs:

```go
func RunningTotal(xs []int64) []int64 {
	out := []int64{}
	for i := 1; i < len(xs); i++ {
		var prev int64
		if i > 1 {
			prev = out[i-2]
		}
		out = append(out, prev+xs[i])
	}
	return out
}
```

Find the defects, fix them, and add tests that would have caught them."#;

const GO_BUGFIX_REF: &str = r#"package totals

import "testing"

// RunningTotal returns the cumulative sums of xs, in order.
func RunningTotal(xs []int64) []int64 {
	out := make([]int64, 0, len(xs))
	var total int64
	for _, x := range xs {
		total += x
		out = append(out, total)
	}
	return out
}

func TestRunningTotalAccumulatesInOrder(t *testing.T) {
	got := RunningTotal([]int64{1, 2, 3, 4})
	want := []int64{1, 3, 6, 10}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("position %d: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestRunningTotalEmptyInput(t *testing.T) {
	if got := RunningTotal(nil); len(got) != 0 {
		t.Errorf("empty input gave %v", got)
	}
}

func TestRunningTotalSingleElement(t *testing.T) {
	got := RunningTotal([]int64{7})
	if len(got) != 1 || got[0] != 7 {
		t.Errorf("single element gave %v", got)
	}
}

func TestRunningTotalNegatives(t *testing.T) {
	got := RunningTotal([]int64{3, -5, 2})
	want := []int64{3, -2, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("position %d: got %d, want %d", i, got[i], want[i])
		}
	}
}
"#;

const GO_REFACTOR_TASK: &str = r#"The following Go function works but is written in an un-idiomatic, \
index-heavy style:

```go
func EvenSquares(xs []int64) []int64 {
	out := []int64{}
	i := 0
	for i < len(xs) {
		x := xs[i]
		if x%2 == 0 {
			out = append(out, x*x)
		}
		i++
	}
	return out
}
```

Refactor it to a `range` loop with a preallocated result, without changing its \
behaviour, and add tests that pin the behaviour."#;

const GO_REFACTOR_REF: &str = r#"package squares

import "testing"

// EvenSquares returns the squares of the even elements of xs, in order.
func EvenSquares(xs []int64) []int64 {
	out := make([]int64, 0, len(xs))
	for _, x := range xs {
		if x%2 != 0 {
			continue
		}
		out = append(out, x*x)
	}
	return out
}

func TestEvenSquaresKeepsEvensOnly(t *testing.T) {
	got := EvenSquares([]int64{1, 2, 3, 4})
	want := []int64{4, 16}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("position %d: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestEvenSquaresEmptyInput(t *testing.T) {
	if got := EvenSquares(nil); len(got) != 0 {
		t.Errorf("empty input gave %v", got)
	}
}

func TestEvenSquaresAllOdd(t *testing.T) {
	if got := EvenSquares([]int64{1, 3, 5}); len(got) != 0 {
		t.Errorf("all-odd input gave %v", got)
	}
}
"#;

const GO_TESTWRITING_TASK: &str = r#"Write thorough table-driven tests for the following Go function without \
modifying the function itself:

```go
func Median(sorted []int64) (float64, bool) {
	if len(sorted) == 0 {
		return 0, false
	}
	mid := len(sorted) / 2
	if len(sorted)%2 != 0 {
		return float64(sorted[mid]), true
	}
	return float64(sorted[mid-1]+sorted[mid]) / 2.0, true
}
```

Cover the empty input, odd and even lengths, and negative values."#;

const GO_TESTWRITING_REF: &str = r#"package stats

import "testing"

// Median returns the middle value of a sorted slice; the bool is false for
// empty input.
func Median(sorted []int64) (float64, bool) {
	if len(sorted) < 1 {
		return 0, false
	}
	mid := len(sorted) / 2
	if len(sorted)%2 != 0 {
		return float64(sorted[mid]), true
	}
	return float64(sorted[mid-1]+sorted[mid]) / 2.0, true
}

func TestMedian(t *testing.T) {
	cases := []struct {
		name  string
		input []int64
		want  float64
		ok    bool
	}{
		{name: "empty", input: nil, want: 0, ok: false},
		{name: "odd length", input: []int64{1, 3, 5}, want: 3, ok: true},
		{name: "even length", input: []int64{1, 2, 3, 4}, want: 2.5, ok: true},
		{name: "negatives", input: []int64{-5, -1, 2}, want: -1, ok: true},
		{name: "single", input: []int64{9}, want: 9, ok: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := Median(tc.input)
			if ok != tc.ok {
				t.Fatalf("ok: got %v, want %v", ok, tc.ok)
			}
			if ok && got != tc.want {
				t.Errorf("median: got %v, want %v", got, tc.want)
			}
		})
	}
}
"#;

const JAVA_IMPLEMENT_TASK: &str = "\
Implement in Java:

    public static List<Map.Entry<String, Integer>> topKWords(String text, int k)

Return the `k` most frequent words of `text` (lowercased, split on whitespace), \
ordered by count descending and then alphabetically. A `k` below 1 yields an \
empty list.";

const JAVA_IMPLEMENT_REF: &str = r#"import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Word-frequency queries over plain text. */
public final class TopKWords {
    private TopKWords() {
        throw new AssertionError("utility class");
    }

    /** The {@code k} most frequent words of {@code text}, ties broken alphabetically. */
    public static List<Map.Entry<String, Integer>> topKWords(String text, int k) {
        if (k < 1) {
            return new ArrayList<>();
        }
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String word : text.toLowerCase().split("\\s+")) {
            if (!word.isEmpty()) {
                counts.merge(word, 1, Integer::sum);
            }
        }
        List<Map.Entry<String, Integer>> ordered = new ArrayList<>(counts.entrySet());
        ordered.sort((a, b) -> {
            int byCount = Integer.compare(b.getValue(), a.getValue());
            return byCount != 0 ? byCount : a.getKey().compareTo(b.getKey());
        });
        return new ArrayList<>(ordered.subList(0, Math.min(k, ordered.size())));
    }
}

class TopKWordsTests {
    public static void main(String[] args) {
        List<Map.Entry<String, Integer>> top = TopKWords.topKWords("b a b c a b", 2);
        if (top.size() != 2 || !top.get(0).getKey().equals("b") || top.get(0).getValue() != 3) {
            throw new AssertionError("count ordering failed: " + top);
        }
        if (!top.get(1).getKey().equals("a") || top.get(1).getValue() != 2) {
            throw new AssertionError("alphabetical tie-break failed: " + top);
        }
        if (!TopKWords.topKWords("a a a", 0).isEmpty()) {
            throw new AssertionError("non-positive k must yield an empty list");
        }
        if (TopKWords.topKWords("x y", 10).size() != 2) {
            throw new AssertionError("k larger than the vocabulary must return everything");
        }
    }
}
"#;

const JAVA_BUGFIX_TASK: &str = r#"The following Java method is meant to remove duplicates while keeping the \
first occurrence of each item, but it keeps the *last* occurrence instead:

```java
public static List<Integer> dedupe(List<Integer> items) {
    List<Integer> out = new ArrayList<>(items);
    for (Integer item : items) {
        while (out.indexOf(item) != out.lastIndexOf(item)) {
            out.remove(item);
        }
    }
    return out;
}
```

`List.remove(Object)` deletes the first matching element, so `dedupe(List.of(1, 2, 1))` \
returns `[2, 1]`. Fix the method and add tests that pin first-seen order."#;

const JAVA_BUGFIX_REF: &str = r#"import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** Duplicate removal that preserves first-seen order. */
public final class Dedupe {
    private Dedupe() {
        throw new AssertionError("utility class");
    }

    /** The items with duplicates removed, first occurrence kept. */
    public static List<Integer> dedupe(List<Integer> items) {
        Set<Integer> seen = new LinkedHashSet<>();
        List<Integer> out = new ArrayList<>();
        for (Integer item : items) {
            if (seen.add(item)) {
                out.add(item);
            }
        }
        return out;
    }
}

class DedupeTests {
    public static void main(String[] args) {
        if (!Dedupe.dedupe(List.of(1, 2, 1)).equals(List.of(1, 2))) {
            throw new AssertionError("first occurrence must be kept");
        }
        if (!Dedupe.dedupe(List.of()).isEmpty()) {
            throw new AssertionError("empty input must stay empty");
        }
        if (!Dedupe.dedupe(List.of(3, 1, 2)).equals(List.of(3, 1, 2))) {
            throw new AssertionError("unique input must be unchanged");
        }
        if (!Dedupe.dedupe(List.of(4, 4, 4)).equals(List.of(4))) {
            throw new AssertionError("a run of duplicates collapses to one");
        }
    }
}
"#;

const JAVA_REFACTOR_TASK: &str = r#"The following Java method works but is written as a manual accumulation loop:

```java
public static double totalPrice(List<Map<String, Double>> cart) {
    double total = 0.0;
    for (Map<String, Double> item : cart) {
        double price = item.get("price") * item.get("qty");
        total = total + price;
    }
    return total;
}
```

Refactor it to a single stream expression, and add tests that pin the behaviour \
(including the empty cart)."#;

const JAVA_REFACTOR_REF: &str = r#"import java.util.List;
import java.util.Map;

/** Cart totals. */
public final class TotalPrice {
    private TotalPrice() {
        throw new AssertionError("utility class");
    }

    /** Sums {@code price * qty} over every line item in {@code cart}. */
    public static double totalPrice(List<Map<String, Double>> cart) {
        return cart.stream()
                .mapToDouble(item -> item.get("price") * item.get("qty"))
                .sum();
    }
}

class TotalPriceTests {
    public static void main(String[] args) {
        List<Map<String, Double>> cart = List.of(
                Map.of("price", 2.5, "qty", 4.0),
                Map.of("price", 1.0, "qty", 1.0));
        if (Double.compare(TotalPrice.totalPrice(cart), 11.0) != 0) {
            throw new AssertionError("line items must sum to 11.0");
        }
        if (Double.compare(TotalPrice.totalPrice(List.of()), 0.0) != 0) {
            throw new AssertionError("an empty cart totals 0.0");
        }
    }
}
"#;

const JAVA_TESTWRITING_TASK: &str = r#"Write a thorough test suite for the following Java method without modifying \
the method itself:

```java
public static double median(int[] sorted) {
    if (sorted.length == 0) {
        throw new IllegalArgumentException("median of an empty array is undefined");
    }
    int mid = sorted.length / 2;
    if (sorted.length % 2 != 0) {
        return sorted[mid];
    }
    return (sorted[mid - 1] + sorted[mid]) / 2.0;
}
```

Cover the empty input, odd and even lengths, and negative values."#;

const JAVA_TESTWRITING_REF: &str = r#"/** Order statistics over sorted arrays. */
public final class Median {
    private Median() {
        throw new AssertionError("utility class");
    }

    /** The middle value of a sorted array; empty input is an error. */
    public static double median(int[] sorted) {
        if (sorted.length < 1) {
            throw new IllegalArgumentException("median of an empty array is undefined");
        }
        int mid = sorted.length / 2;
        if (sorted.length % 2 != 0) {
            return sorted[mid];
        }
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
}

class MedianTests {
    public static void main(String[] args) {
        boolean refused = false;
        try {
            Median.median(new int[0]);
        } catch (IllegalArgumentException expected) {
            refused = true;
        }
        if (!refused) {
            throw new AssertionError("empty input must be refused");
        }
        if (Double.compare(Median.median(new int[] {1, 3, 5}), 3.0) != 0) {
            throw new AssertionError("odd length takes the middle element");
        }
        if (Double.compare(Median.median(new int[] {1, 2, 3, 4}), 2.5) != 0) {
            throw new AssertionError("even length averages the middle pair");
        }
        if (Double.compare(Median.median(new int[] {-5, -1, 2}), -1.0) != 0) {
            throw new AssertionError("negatives are handled");
        }
    }
}
"#;

const CSHARP_IMPLEMENT_TASK: &str = "\
Implement in C#:

    public static List<KeyValuePair<string, int>> TopKWords(string text, int k)

Return the `k` most frequent words of `text` (lowercased, split on whitespace), \
ordered by count descending and then alphabetically (ordinal). A `k` below 1 \
yields an empty list.";

const CSHARP_IMPLEMENT_REF: &str = r#"using System;
using System.Collections.Generic;
using System.Linq;

public static class WordStats
{
    /// <summary>The <c>k</c> most frequent words of <c>text</c>, ties broken ordinally.</summary>
    public static List<KeyValuePair<string, int>> TopKWords(string text, int k)
    {
        if (k < 1)
        {
            return new List<KeyValuePair<string, int>>();
        }
        var counts = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (var word in text.ToLowerInvariant()
                     .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            counts[word] = counts.TryGetValue(word, out int n) ? n + 1 : 1;
        }
        return counts
            .OrderByDescending(pair => pair.Value)
            .ThenBy(pair => pair.Key, StringComparer.Ordinal)
            .Take(k)
            .ToList();
    }
}

public static class WordStatsTests
{
    public static void Run()
    {
        var top = WordStats.TopKWords("b a b c a b", 2);
        if (top.Count != 2 || top[0].Key != "b" || top[0].Value != 3)
        {
            throw new InvalidOperationException("count ordering failed");
        }
        if (top[1].Key != "a" || top[1].Value != 2)
        {
            throw new InvalidOperationException("ordinal tie-break failed");
        }
        if (WordStats.TopKWords("a a a", 0).Count != 0)
        {
            throw new InvalidOperationException("non-positive k must yield an empty list");
        }
        if (WordStats.TopKWords("Hello hello HELLO", 5)[0].Value != 3)
        {
            throw new InvalidOperationException("case must be normalised");
        }
    }
}
"#;

const CSHARP_BUGFIX_TASK: &str = r#"The following C# method is meant to return the running total of its input, \
but it drops the first element and returns nonsense for short inputs:

```csharp
public static List<long> RunningTotal(IReadOnlyList<long> xs)
{
    var out_ = new List<long>();
    for (var i = 1; i < xs.Count; i++)
    {
        var prev = i > 1 ? out_[i - 2] : 0;
        out_.Add(prev + xs[i]);
    }
    return out_;
}
```

Find the defects, fix them, and add tests that would have caught them."#;

const CSHARP_BUGFIX_REF: &str = r#"using System;
using System.Collections.Generic;

public static class Totals
{
    /// <summary>The cumulative sums of <c>xs</c>, in order.</summary>
    public static List<long> RunningTotal(IReadOnlyList<long> xs)
    {
        List<long> result = new List<long>(xs.Count);
        long total = 0;
        foreach (long x in xs)
        {
            total += x;
            result.Add(total);
        }
        return result;
    }
}

public static class TotalsTests
{
    public static void Run()
    {
        List<long> got = Totals.RunningTotal(new long[] { 1, 2, 3, 4 });
        long[] want = new long[] { 1, 3, 6, 10 };
        if (got.Count != want.Length)
        {
            throw new InvalidOperationException("length mismatch");
        }
        for (int i = 0; i < want.Length; i++)
        {
            if (got[i] != want[i])
            {
                throw new InvalidOperationException($"position {i}: got {got[i]}, want {want[i]}");
            }
        }
        if (Totals.RunningTotal(Array.Empty<long>()).Count != 0)
        {
            throw new InvalidOperationException("empty input stays empty");
        }
        if (Totals.RunningTotal(new long[] { 7 })[0] != 7)
        {
            throw new InvalidOperationException("a single element is its own total");
        }
    }
}
"#;

const CSHARP_REFACTOR_TASK: &str = r#"The following C# method works but is written as a manual loop:

```csharp
public static List<long> EvenSquares(IReadOnlyList<long> xs)
{
    var result = new List<long>();
    for (var i = 0; i < xs.Count; i++)
    {
        var x = xs[i];
        if (x % 2 == 0)
        {
            result.Add(x * x);
        }
    }
    return result;
}
```

Refactor it to a single LINQ expression without changing its behaviour, and add \
tests that pin the behaviour."#;

const CSHARP_REFACTOR_REF: &str = r#"using System;
using System.Collections.Generic;
using System.Linq;

public static class Squares
{
    /// <summary>The squares of the even elements of <c>xs</c>, in order.</summary>
    public static List<long> EvenSquares(IReadOnlyList<long> xs)
    {
        return xs.Where(x => x % 2 == 0).Select(x => x * x).ToList();
    }
}

public static class SquaresTests
{
    public static void Run()
    {
        var got = Squares.EvenSquares(new long[] { 1, 2, 3, 4 });
        if (got.Count != 2 || got[0] != 4 || got[1] != 16)
        {
            throw new InvalidOperationException("squares of evens only");
        }
        if (Squares.EvenSquares(Array.Empty<long>()).Count != 0)
        {
            throw new InvalidOperationException("empty input gives empty output");
        }
        if (Squares.EvenSquares(new long[] { 1, 3, 5 }).Count != 0)
        {
            throw new InvalidOperationException("all-odd input gives empty output");
        }
    }
}
"#;

const CSHARP_TESTWRITING_TASK: &str = r#"Write a thorough test suite for the following C# method without modifying \
the method itself:

```csharp
public static double Median(IReadOnlyList<long> sorted)
{
    if (sorted.Count == 0)
    {
        throw new ArgumentException("median of an empty input is undefined");
    }
    var mid = sorted.Count / 2;
    if (sorted.Count % 2 != 0)
    {
        return sorted[mid];
    }
    return (sorted[mid - 1] + sorted[mid]) / 2.0;
}
```

Cover the empty input, odd and even lengths, and negative values."#;

const CSHARP_TESTWRITING_REF: &str = r#"using System;
using System.Collections.Generic;

public static class Stats
{
    /// <summary>The middle value of a sorted input; empty input is an error.</summary>
    public static double Median(IReadOnlyList<long> sorted)
    {
        if (sorted.Count < 1)
        {
            throw new ArgumentException("median of an empty input is undefined");
        }
        var mid = sorted.Count / 2;
        if (sorted.Count % 2 != 0)
        {
            return sorted[mid];
        }
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
}

public static class StatsTests
{
    public static void Run()
    {
        var refused = false;
        try
        {
            Stats.Median(Array.Empty<long>());
        }
        catch (ArgumentException)
        {
            refused = true;
        }
        if (!refused)
        {
            throw new InvalidOperationException("empty input must be refused");
        }
        if (Stats.Median(new long[] { 1, 3, 5 }) != 3.0)
        {
            throw new InvalidOperationException("odd length takes the middle element");
        }
        if (Stats.Median(new long[] { 1, 2, 3, 4 }) != 2.5)
        {
            throw new InvalidOperationException("even length averages the middle pair");
        }
        if (Stats.Median(new long[] { -5, -1, 2 }) != -1.0)
        {
            throw new InvalidOperationException("negatives are handled");
        }
    }
}
"#;

const PHP_IMPLEMENT_TASK: &str = "\
Implement in PHP:

    function top_k_words(string $text, int $k): array

Return the `k` most frequent words of `text` (lowercased, split on whitespace) as \
a list of `[word, count]` pairs, ordered by count descending and then \
alphabetically. A `$k` below 1 yields an empty array.";

const PHP_IMPLEMENT_REF: &str = r#"<?php
function top_k_words(string $text, int $k): array
{
    if ($k < 1) {
        return [];
    }
    $counts = [];
    foreach (preg_split("/\s+/", strtolower($text)) as $word) {
        if ($word !== "") {
            $counts[$word] = ($counts[$word] ?? 0) + 1;
        }
    }
    $pairs = [];
    foreach ($counts as $word => $count) {
        $pairs[] = [$word, $count];
    }
    usort($pairs, function (array $a, array $b): int {
        if ($a[1] !== $b[1]) {
            return $b[1] - $a[1];
        }
        return strcmp($a[0], $b[0]);
    });
    return array_slice($pairs, 0, $k);
}

function expect_equal(string $label, $got, $want): void
{
    if ($got !== $want) {
        throw new RuntimeException($label . " failed");
    }
}

expect_equal("count ordering", top_k_words("b a b c a b", 2), [["b", 3], ["a", 2]]);
expect_equal("k larger than vocabulary", top_k_words("x y", 10), [["x", 1], ["y", 1]]);
expect_equal("non-positive k", top_k_words("a a a", 0), []);
expect_equal("case normalised", top_k_words("Hello hello HELLO", 5), [["hello", 3]]);
"#;

const PHP_BUGFIX_TASK: &str = r#"The following PHP function is meant to remove duplicates while keeping the \
first occurrence of each item, but it keeps the *last* occurrence instead:

```php
function dedupe(array $items): array
{
    $out = $items;
    foreach ($items as $item) {
        while (count(array_keys($out, $item)) > 1) {
            $first = array_search($item, $out);
            unset($out[$first]);
        }
    }
    return array_values($out);
}
```

`array_search` finds the first match, so `dedupe([1, 2, 1])` returns `[2, 1]`. Fix \
the function and add tests that pin first-seen order."#;

const PHP_BUGFIX_REF: &str = r#"<?php
function dedupe(array $items): array
{
    $seen = [];
    $out = [];
    foreach ($items as $item) {
        $key = serialize($item);
        if (!isset($seen[$key])) {
            $seen[$key] = true;
            $out[] = $item;
        }
    }
    return $out;
}

function expect_equal(string $label, $got, $want): void
{
    if ($got !== $want) {
        throw new RuntimeException($label . " failed");
    }
}

expect_equal("first occurrence kept", dedupe([1, 2, 1]), [1, 2]);
expect_equal("empty input", dedupe([]), []);
expect_equal("unique input unchanged", dedupe([3, 1, 2]), [3, 1, 2]);
expect_equal("run of duplicates", dedupe([4, 4, 4]), [4]);
"#;

const PHP_REFACTOR_TASK: &str = r#"The following PHP function works but is written as a manual accumulation loop:

```php
function total_price(array $cart): float
{
    $total = 0.0;
    foreach ($cart as $item) {
        $price = $item["price"] * $item["qty"];
        $total = $total + $price;
    }
    return $total;
}
```

Refactor it to a single `array_reduce` expression, and add tests that pin the \
behaviour (including the empty cart)."#;

const PHP_REFACTOR_REF: &str = r#"<?php
function total_price(array $cart): float
{
    return array_reduce(
        $cart,
        fn(float $sum, array $item): float => $sum + $item["price"] * $item["qty"],
        0.0
    );
}

function expect_equal(string $label, $got, $want): void
{
    if ($got !== $want) {
        throw new RuntimeException($label . " failed");
    }
}

expect_equal(
    "line items sum",
    total_price([["price" => 2.5, "qty" => 4], ["price" => 1.0, "qty" => 1]]),
    11.0
);
expect_equal("empty cart", total_price([]), 0.0);
"#;

const PHP_TESTWRITING_TASK: &str = r#"Write a thorough test suite for the following PHP function without modifying \
the function itself:

```php
function median(array $sorted): ?float
{
    $n = count($sorted);
    if ($n === 0) {
        return null;
    }
    $mid = intdiv($n, 2);
    if ($n % 2 !== 0) {
        return (float) $sorted[$mid];
    }
    return ($sorted[$mid - 1] + $sorted[$mid]) / 2.0;
}
```

Cover the empty input, odd and even lengths, and negative values."#;

const PHP_TESTWRITING_REF: &str = r#"<?php
function median(array $sorted): ?float
{
    $n = count($sorted);
    if ($n < 1) {
        return null;
    }
    $mid = intdiv($n, 2);
    if ($n % 2 !== 0) {
        return (float) $sorted[$mid];
    }
    return ($sorted[$mid - 1] + $sorted[$mid]) / 2.0;
}

function expect_equal(string $label, $got, $want): void
{
    if ($got !== $want) {
        throw new RuntimeException($label . " failed");
    }
}

expect_equal("empty input", median([]), null);
expect_equal("odd length", median([1, 3, 5]), 3.0);
expect_equal("even length", median([1, 2, 3, 4]), 2.5);
expect_equal("negatives", median([-5, -1, 2]), -1.0);
expect_equal("single element", median([9]), 9.0);
"#;

/// The template bank: the task body and reference completion for one
/// (language, kind) pair. Bodies carry the *task* only; [`finish_prompt`]
/// adds the language header and [`PROMPT_REQUIREMENTS`].
fn template(language: CodeLanguage, kind: TaskKind) -> (&'static str, &'static str) {
    match (language, kind) {
        (CodeLanguage::Rust, TaskKind::Implement) => (RUST_IMPLEMENT_TASK, RUST_IMPLEMENT_REF),
        (CodeLanguage::Rust, TaskKind::BugFix) => (RUST_BUGFIX_TASK, RUST_BUGFIX_REF),
        (CodeLanguage::Rust, TaskKind::Refactor) => (RUST_REFACTOR_TASK, RUST_REFACTOR_REF),
        (CodeLanguage::Rust, TaskKind::TestWriting) => {
            (RUST_TESTWRITING_TASK, RUST_TESTWRITING_REF)
        }
        (CodeLanguage::Python, TaskKind::Implement) => {
            (PYTHON_IMPLEMENT_TASK, PYTHON_IMPLEMENT_REF)
        }
        (CodeLanguage::Python, TaskKind::BugFix) => (PYTHON_BUGFIX_TASK, PYTHON_BUGFIX_REF),
        (CodeLanguage::Python, TaskKind::Refactor) => (PYTHON_REFACTOR_TASK, PYTHON_REFACTOR_REF),
        (CodeLanguage::Python, TaskKind::TestWriting) => {
            (PYTHON_TESTWRITING_TASK, PYTHON_TESTWRITING_REF)
        }
        (CodeLanguage::C, TaskKind::Implement) => (C_IMPLEMENT_TASK, C_IMPLEMENT_REF),
        (CodeLanguage::C, TaskKind::BugFix) => (C_BUGFIX_TASK, C_BUGFIX_REF),
        (CodeLanguage::C, TaskKind::Refactor) => (C_REFACTOR_TASK, C_REFACTOR_REF),
        (CodeLanguage::C, TaskKind::TestWriting) => (C_TESTWRITING_TASK, C_TESTWRITING_REF),
        (CodeLanguage::Cpp, TaskKind::Implement) => (CPP_IMPLEMENT_TASK, CPP_IMPLEMENT_REF),
        (CodeLanguage::Cpp, TaskKind::BugFix) => (CPP_BUGFIX_TASK, CPP_BUGFIX_REF),
        (CodeLanguage::Cpp, TaskKind::Refactor) => (CPP_REFACTOR_TASK, CPP_REFACTOR_REF),
        (CodeLanguage::Cpp, TaskKind::TestWriting) => (CPP_TESTWRITING_TASK, CPP_TESTWRITING_REF),
        (CodeLanguage::JsTs, TaskKind::Implement) => (JSTS_IMPLEMENT_TASK, JSTS_IMPLEMENT_REF),
        (CodeLanguage::JsTs, TaskKind::BugFix) => (JSTS_BUGFIX_TASK, JSTS_BUGFIX_REF),
        (CodeLanguage::JsTs, TaskKind::Refactor) => (JSTS_REFACTOR_TASK, JSTS_REFACTOR_REF),
        (CodeLanguage::JsTs, TaskKind::TestWriting) => {
            (JSTS_TESTWRITING_TASK, JSTS_TESTWRITING_REF)
        }
        (CodeLanguage::Web, TaskKind::Implement) => (WEB_IMPLEMENT_TASK, WEB_IMPLEMENT_REF),
        (CodeLanguage::Web, TaskKind::BugFix) => (WEB_BUGFIX_TASK, WEB_BUGFIX_REF),
        (CodeLanguage::Web, TaskKind::Refactor) => (WEB_REFACTOR_TASK, WEB_REFACTOR_REF),
        (CodeLanguage::Web, TaskKind::TestWriting) => (WEB_TESTWRITING_TASK, WEB_TESTWRITING_REF),
        (CodeLanguage::Go, TaskKind::Implement) => (GO_IMPLEMENT_TASK, GO_IMPLEMENT_REF),
        (CodeLanguage::Go, TaskKind::BugFix) => (GO_BUGFIX_TASK, GO_BUGFIX_REF),
        (CodeLanguage::Go, TaskKind::Refactor) => (GO_REFACTOR_TASK, GO_REFACTOR_REF),
        (CodeLanguage::Go, TaskKind::TestWriting) => (GO_TESTWRITING_TASK, GO_TESTWRITING_REF),
        (CodeLanguage::Java, TaskKind::Implement) => (JAVA_IMPLEMENT_TASK, JAVA_IMPLEMENT_REF),
        (CodeLanguage::Java, TaskKind::BugFix) => (JAVA_BUGFIX_TASK, JAVA_BUGFIX_REF),
        (CodeLanguage::Java, TaskKind::Refactor) => (JAVA_REFACTOR_TASK, JAVA_REFACTOR_REF),
        (CodeLanguage::Java, TaskKind::TestWriting) => {
            (JAVA_TESTWRITING_TASK, JAVA_TESTWRITING_REF)
        }
        (CodeLanguage::CSharp, TaskKind::Implement) => {
            (CSHARP_IMPLEMENT_TASK, CSHARP_IMPLEMENT_REF)
        }
        (CodeLanguage::CSharp, TaskKind::BugFix) => (CSHARP_BUGFIX_TASK, CSHARP_BUGFIX_REF),
        (CodeLanguage::CSharp, TaskKind::Refactor) => (CSHARP_REFACTOR_TASK, CSHARP_REFACTOR_REF),
        (CodeLanguage::CSharp, TaskKind::TestWriting) => {
            (CSHARP_TESTWRITING_TASK, CSHARP_TESTWRITING_REF)
        }
        (CodeLanguage::Php, TaskKind::Implement) => (PHP_IMPLEMENT_TASK, PHP_IMPLEMENT_REF),
        (CodeLanguage::Php, TaskKind::BugFix) => (PHP_BUGFIX_TASK, PHP_BUGFIX_REF),
        (CodeLanguage::Php, TaskKind::Refactor) => (PHP_REFACTOR_TASK, PHP_REFACTOR_REF),
        (CodeLanguage::Php, TaskKind::TestWriting) => (PHP_TESTWRITING_TASK, PHP_TESTWRITING_REF),
    }
}

// ---------------------------------------------------------------------------
// Pipeline configuration and records
// ---------------------------------------------------------------------------

/// What the pipeline does with a trace the cheat filter flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheatPolicy {
    /// Every flagged trace goes to `rejects.jsonl`. The default, and the
    /// honest one: a stub or a broken snippet in the training set teaches
    /// exactly the behaviour the gate exists to prevent.
    #[default]
    Reject,
    /// Reject-grade traces (suppression, eval gaming) are still dropped, but
    /// quarantine-grade traces (stubs, broken snippets) are kept in
    /// `train.jsonl` with `quarantined: true`, so a downstream consumer can
    /// weight or drop them with the information preserved.
    Quarantine,
}

fn default_languages() -> Vec<CodeLanguage> {
    CodeLanguage::ALL.to_vec()
}
fn default_tasks_per_language() -> usize {
    DEFAULT_TASKS_PER_LANGUAGE
}
fn default_max_tokens() -> u32 {
    1024
}
fn default_base_seed() -> u64 {
    DEFAULT_BASE_SEED
}
fn default_out_dir() -> PathBuf {
    PathBuf::from(DEFAULT_OUT_DIR)
}
fn default_oss_path() -> PathBuf {
    PathBuf::from(DEFAULT_OSS_INSTRUCT_PATH)
}
fn default_eval_path() -> PathBuf {
    PathBuf::from(DEFAULT_EVAL_TASKS_PATH)
}
fn default_decontaminate() -> bool {
    true
}
fn default_max_problem_chars() -> usize {
    DEFAULT_MAX_PROBLEM_CHARS
}

/// Everything a pipeline run needs. Every field carries `serde(default)`, so
/// a config written by an older build still parses, and `serde_json::from_str
/// ("{}")` is the fully-offline default run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Languages to build data for, in run order.
    #[serde(default = "default_languages")]
    pub languages: Vec<CodeLanguage>,
    /// Task seeds per language.
    #[serde(default = "default_tasks_per_language")]
    pub tasks_per_language: usize,
    /// Teacher request cap, applied per language. Irrelevant offline.
    #[serde(default)]
    pub budget: TraceBudget,
    /// Token cap per teacher completion.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Seed everything downstream derives from.
    #[serde(default = "default_base_seed")]
    pub base_seed: u64,
    /// Output root: one subdirectory per language plus `report.json`.
    #[serde(default = "default_out_dir")]
    pub out_dir: PathBuf,
    /// Magicoder OSS-Instruct JSONL: offline prompt source and fallback
    /// completion corpus.
    #[serde(default = "default_oss_path")]
    pub oss_instruct_path: PathBuf,
    /// Repo-native eval tasks the traces are decontaminated against.
    #[serde(default = "default_eval_path")]
    pub eval_tasks_path: PathBuf,
    /// Whether to decontaminate against [`Self::eval_tasks_path`]. Turning
    /// this off is a statement that contamination does not matter for the
    /// run; the report records which was chosen.
    #[serde(default = "default_decontaminate")]
    pub decontaminate: bool,
    /// Reject vs quarantine handling for cheat-flagged traces.
    #[serde(default)]
    pub cheat_policy: CheatPolicy,
    /// Cap on how much of a Magicoder problem statement enters a prompt.
    #[serde(default = "default_max_problem_chars")]
    pub max_problem_chars: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            languages: default_languages(),
            tasks_per_language: default_tasks_per_language(),
            budget: TraceBudget::default(),
            max_tokens: default_max_tokens(),
            base_seed: default_base_seed(),
            out_dir: default_out_dir(),
            oss_instruct_path: default_oss_path(),
            eval_tasks_path: default_eval_path(),
            decontaminate: default_decontaminate(),
            cheat_policy: CheatPolicy::default(),
            max_problem_chars: default_max_problem_chars(),
        }
    }
}

/// One line of `train.jsonl`: a trace that survived every filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainRecord {
    /// [`TaskSeed::id`] — stable across reruns.
    pub id: String,
    /// [`CodeLanguage::name`].
    pub language: String,
    /// [`TaskKind::as_str`].
    pub kind: String,
    /// The task as the teacher saw it.
    pub prompt: String,
    /// The code to learn from.
    pub completion: String,
    /// `teacher` or `offline`.
    pub source: String,
    /// Teacher model name, or `offline-magicoder` / `offline-template`.
    pub model: String,
    /// Per-task seed, for reproduction.
    pub seed: u64,
    /// True when [`CheatPolicy::Quarantine`] kept this trace despite a
    /// quarantine-grade finding.
    #[serde(default)]
    pub quarantined: bool,
}

/// One line of `rejects.jsonl`: a trace a filter dropped, with the reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectRecord {
    /// [`TaskSeed::id`].
    pub id: String,
    /// [`CodeLanguage::name`].
    pub language: String,
    /// Which filter dropped it: `cheat` or `decontaminate`.
    pub stage: String,
    /// Cheat classes found (their `as_str` names), or `["decontaminated"]`.
    pub classes: Vec<String>,
    /// Human-readable summary of the findings.
    pub reason: String,
    /// The task, kept so the reject file is self-contained.
    pub prompt: String,
    /// The dropped completion.
    pub completion: String,
}

/// What the pipeline did for one language.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LanguageReport {
    /// [`CodeLanguage::name`].
    pub language: String,
    /// Seeds built.
    pub tasks: usize,
    /// How many of them came from Magicoder.
    pub magicoder_seeds: usize,
    /// How many came from the template bank.
    pub template_seeds: usize,
    /// Traces that entered the filter stages (a teacher run can produce
    /// fewer than `tasks` when the budget cuts it off).
    pub traces_generated: usize,
    /// Seeds with no teacher trace (budget or request failure).
    pub teacher_missing: usize,
    /// Traces in `train.jsonl`.
    pub kept: usize,
    /// Kept traces carrying the quarantine flag.
    pub quarantined_kept: usize,
    /// Cheat-class name -> traces dropped because of it.
    pub dropped_by_class: BTreeMap<String, usize>,
    /// Traces dropped by the decontamination stage.
    pub decontam_hits: usize,
    /// Kept traces by `source` (`teacher` / `offline`).
    pub source_mix: BTreeMap<String, usize>,
    /// Output files.
    pub train_path: PathBuf,
    /// Output files.
    pub rejects_path: PathBuf,
    /// True when this report was read back from disk rather than recomputed.
    #[serde(default)]
    pub resumed: bool,
}

/// The top-level `report.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineReport {
    /// Output root the run wrote to.
    pub out_dir: PathBuf,
    /// Whether a teacher client was attached.
    pub teacher: bool,
    /// Whether decontamination ran, and against how many eval tasks.
    pub decontam_index_tasks: Option<usize>,
    /// Per-language outcomes, in config order.
    pub languages: Vec<LanguageReport>,
}

impl PipelineReport {
    /// Total traces kept across all languages.
    pub fn total_kept(&self) -> usize {
        self.languages.iter().map(|l| l.kept).sum()
    }

    /// Total traces dropped across all languages and stages.
    pub fn total_dropped(&self) -> usize {
        self.languages
            .iter()
            .map(|l| l.decontam_hits + l.dropped_by_class.values().sum::<usize>())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// A seed plus the completion that will be filtered, whatever its origin.
struct Candidate {
    seed: TaskSeed,
    completion: String,
    source: &'static str,
    model: String,
}

/// Run the whole pipeline and return the report that `report.json` records.
///
/// With `client: Some`, task prompts go to the teacher (budget-capped,
/// resumable at the request level by [`NimClient::generate_traces`]). With
/// `None`, every seed is answered offline from its reference completion.
/// Either way the cheat filter, the decontamination stage and the output
/// layout are identical. A language whose `report.json` already exists in
/// `config.out_dir` is skipped and its stored report read back, so a rerun
/// resumes at language granularity.
pub fn run_pipeline(
    client: Option<&NimClient>,
    config: &PipelineConfig,
) -> anyhow::Result<PipelineReport> {
    ensure!(
        !config.languages.is_empty(),
        "PipelineConfig.languages is empty: name at least one of \
         rust|python|c|cpp|jsts|web|go|java|csharp|php"
    );
    ensure!(
        config.tasks_per_language > 0,
        "PipelineConfig.tasks_per_language must be positive"
    );
    ensure!(
        config.max_tokens > 0,
        "PipelineConfig.max_tokens must be positive"
    );
    std::fs::create_dir_all(&config.out_dir).with_context(|| {
        format!(
            "create pipeline output directory {}",
            config.out_dir.display()
        )
    })?;

    let eval_index = if config.decontaminate {
        let tasks = codegen_eval::read_tasks(&config.eval_tasks_path).with_context(|| {
            format!(
                "decontamination is on but the eval tasks at {} could not be read; \
                 generate them with `cargo run --example evalgen` or set decontaminate: false",
                config.eval_tasks_path.display()
            )
        })?;
        Some(CorpusIndex::build(
            tasks.iter().map(|t| (t.id.as_str(), t.prompt.as_str())),
        ))
    } else {
        None
    };

    let mut report = PipelineReport {
        out_dir: config.out_dir.clone(),
        teacher: client.is_some(),
        decontam_index_tasks: eval_index.as_ref().map(CorpusIndex::task_count),
        languages: Vec::new(),
    };
    for &language in &config.languages {
        let lang_dir = config.out_dir.join(language.name());
        let report_path = lang_dir.join("report.json");
        if report_path.exists() {
            let text = std::fs::read_to_string(&report_path).with_context(|| {
                format!("read stored language report {}", report_path.display())
            })?;
            let mut stored: LanguageReport = serde_json::from_str(&text).with_context(|| {
                format!(
                    "stored language report {} is corrupt; remove the {} directory to rebuild it",
                    report_path.display(),
                    lang_dir.display()
                )
            })?;
            stored.resumed = true;
            report.languages.push(stored);
            continue;
        }
        let language_report =
            run_language(client, config, language, &lang_dir, eval_index.as_ref())?;
        write_json(&report_path, &language_report)?;
        report.languages.push(language_report);
    }
    write_json(&config.out_dir.join("report.json"), &report)?;
    Ok(report)
}

/// All stages for one language. Never called for a language that already has
/// a `report.json` — that check is the resume contract, in [`run_pipeline`].
fn run_language(
    client: Option<&NimClient>,
    config: &PipelineConfig,
    language: CodeLanguage,
    lang_dir: &Path,
    eval_index: Option<&CorpusIndex>,
) -> anyhow::Result<LanguageReport> {
    std::fs::create_dir_all(lang_dir)
        .with_context(|| format!("create language directory {}", lang_dir.display()))?;
    let seeds = build_seeds(
        language,
        config.tasks_per_language,
        config.base_seed,
        &config.oss_instruct_path,
        config.max_problem_chars,
    )?;
    let magicoder_seeds = seeds
        .iter()
        .filter(|s| s.source == SeedSource::Magicoder)
        .count();

    let mut candidates = Vec::new();
    let mut teacher_missing = 0usize;
    match client {
        Some(client) => {
            let records = generate_teacher_traces(client, config, language, &seeds, lang_dir)?;
            for seed in &seeds {
                let prompt = trace_prompt(config, language, seed);
                match records.get(&prompt.id()) {
                    Some(record) => candidates.push(Candidate {
                        seed: seed.clone(),
                        completion: record.completion.clone(),
                        source: "teacher",
                        model: record.model.clone(),
                    }),
                    None => teacher_missing += 1,
                }
            }
        }
        None => {
            for seed in seeds.iter().cloned() {
                candidates.push(Candidate {
                    model: format!("offline-{}", seed.source.as_str()),
                    completion: seed.reference.clone(),
                    seed,
                    source: "offline",
                });
            }
        }
    }

    let snippet_lang = snippet_language(language);
    let completions: Vec<String> = candidates.iter().map(|c| c.completion.clone()).collect();
    // filter_traces is the authoritative verdict on which indices survive;
    // the per-trace rescans below exist only to attribute the drops.
    let (kept_indices, aggregate) = filter_traces(&completions, snippet_lang);
    let kept_set: HashSet<usize> = kept_indices.into_iter().collect();

    let mut train: Vec<TrainRecord> = Vec::new();
    let mut rejects: Vec<RejectRecord> = Vec::new();
    let mut dropped_by_class: BTreeMap<String, usize> = BTreeMap::new();
    let mut quarantined_kept = 0usize;
    let mut decontam_hits = 0usize;

    for (i, candidate) in candidates.iter().enumerate() {
        let mut quarantined = false;
        if !kept_set.contains(&i) {
            let single = scan_snippet(&candidate.completion, snippet_lang);
            let reject_grade = single.worst() == Some(Severity::Suppression)
                || single.worst() == Some(Severity::Fraud);
            if config.cheat_policy == CheatPolicy::Quarantine && !reject_grade {
                quarantined = true;
                quarantined_kept += 1;
            } else {
                for class in classes_of(&single) {
                    *dropped_by_class.entry(class).or_insert(0) += 1;
                }
                rejects.push(RejectRecord {
                    id: candidate.seed.id.clone(),
                    language: language.name().to_string(),
                    stage: "cheat".to_string(),
                    classes: classes_of(&single),
                    reason: single.render(),
                    prompt: candidate.seed.prompt.clone(),
                    completion: candidate.completion.clone(),
                });
                continue;
            }
        }
        if let Some(index) = eval_index {
            // The prompt is what contamination means here: a trace whose task
            // near-duplicates an eval task lets the model memorise the eval,
            // whatever the completion says (mirrors codegen_eval::index_sft,
            // which indexes instructions only).
            let hit =
                codegen_eval::decontaminate(index, &candidate.seed.prompt, "repo-native-eval");
            if hit.is_contaminated {
                decontam_hits += 1;
                let example = hit.example.unwrap_or_default();
                rejects.push(RejectRecord {
                    id: candidate.seed.id.clone(),
                    language: language.name().to_string(),
                    stage: "decontaminate".to_string(),
                    classes: vec!["decontaminated".to_string()],
                    reason: format!(
                        "prompt overlaps the repo-native eval at {:.1}% of its n-grams (example: {})",
                        hit.overlap * 100.0,
                        example
                    ),
                    prompt: candidate.seed.prompt.clone(),
                    completion: candidate.completion.clone(),
                });
                continue;
            }
        }
        train.push(TrainRecord {
            id: candidate.seed.id.clone(),
            language: language.name().to_string(),
            kind: candidate.seed.kind.as_str().to_string(),
            prompt: candidate.seed.prompt.clone(),
            completion: candidate.completion.clone(),
            source: candidate.source.to_string(),
            model: candidate.model.clone(),
            seed: candidate.seed.seed,
            quarantined,
        });
    }

    let train_path = lang_dir.join("train.jsonl");
    let rejects_path = lang_dir.join("rejects.jsonl");
    write_jsonl(&train_path, &train)?;
    write_jsonl(&rejects_path, &rejects)?;

    let mut source_mix: BTreeMap<String, usize> = BTreeMap::new();
    for record in &train {
        *source_mix.entry(record.source.clone()).or_insert(0) += 1;
    }
    eprintln!(
        "[langdata] {}: {} seeds, {} kept ({} quarantined), {} cheat drops across {} findings, {} decontam hits",
        language.name(),
        seeds.len(),
        train.len(),
        quarantined_kept,
        dropped_by_class.values().sum::<usize>(),
        aggregate.findings.len(),
        decontam_hits
    );
    Ok(LanguageReport {
        language: language.name().to_string(),
        tasks: seeds.len(),
        magicoder_seeds,
        template_seeds: seeds.len() - magicoder_seeds,
        traces_generated: candidates.len(),
        teacher_missing,
        kept: train.len(),
        quarantined_kept,
        dropped_by_class,
        decontam_hits,
        source_mix,
        train_path,
        rejects_path,
        resumed: false,
    })
}

/// The teacher stage: hand every seed to [`NimClient::generate_traces`] (whose
/// own id-based resume makes rerunning cheap) and read the trace file back.
fn generate_teacher_traces(
    client: &NimClient,
    config: &PipelineConfig,
    language: CodeLanguage,
    seeds: &[TaskSeed],
    lang_dir: &Path,
) -> anyhow::Result<HashMap<String, TraceRecord>> {
    let prompts: Vec<TracePrompt> = seeds
        .iter()
        .map(|seed| trace_prompt(config, language, seed))
        .collect();
    let raw_dir = lang_dir.join("raw");
    let trace_report = client.generate_traces(&prompts, &raw_dir, config.budget)?;
    let mut records = HashMap::new();
    for record in crate::nim::read_traces(&trace_report.out_path)? {
        records.insert(record.id.clone(), record);
    }
    Ok(records)
}

/// The [`TracePrompt`] for one seed: the id is a hash of language, seed and
/// prompt, so it is stable across reruns and matches whatever the teacher
/// stage already served.
fn trace_prompt(config: &PipelineConfig, language: CodeLanguage, seed: &TaskSeed) -> TracePrompt {
    TracePrompt {
        language: language.name().to_string(),
        prompt: seed.prompt.clone(),
        seed: seed.seed,
        max_tokens: config.max_tokens,
    }
}

/// Distinct cheat-class names in one trace's scan, in a stable order.
fn classes_of(report: &crate::cheat::CheatReport) -> Vec<String> {
    let mut classes: Vec<String> = report
        .findings
        .iter()
        .map(|f| f.class.as_str().to_string())
        .collect();
    classes.sort();
    classes.dedup();
    classes
}

/// Serialize `rows` as JSONL, atomically.
fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> anyhow::Result<()> {
    let mut out = String::new();
    for row in rows {
        out.push_str(&serde_json::to_string(row).context("serialize JSONL row")?);
        out.push('\n');
    }
    crate::nim::write_atomic(path, out.as_bytes())
}

/// Serialize `value` as pretty JSON, atomically.
fn write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let text = serde_json::to_string_pretty(value).context("serialize JSON document")?;
    crate::nim::write_atomic(path, text.as_bytes())
}

// ---------------------------------------------------------------------------
// Router agreement: the router-which-knows readout used by student.rs's
// held-out routing proof. Lives here because it is a *data-side* measurement:
// it compares the router's per-token language picks against the language a
// sequence actually is.
// ---------------------------------------------------------------------------

/// Per-language tally of router picks: `correct[l]` tokens of language `l`
/// whose top-1 box was `l`, out of `total[l]` tokens of language `l`.
/// Indexed by [`CodeLanguage::index`], sized to [`CodeLanguage::KNOWN`].
#[derive(Debug, Clone, Default)]
pub struct RouterAgreement {
    /// Tokens routed to their own language, per language.
    pub correct: [usize; CodeLanguage::KNOWN.len()],
    /// Tokens seen, per language.
    pub total: [usize; CodeLanguage::KNOWN.len()],
}

impl RouterAgreement {
    /// Fraction of language `l`'s tokens routed to `l`. An error — not a
    /// quiet 0.0 — when no token of that language was seen: an accuracy over
    /// zero samples is a number somebody would trust for the wrong reason.
    pub fn accuracy(&self, language: CodeLanguage) -> anyhow::Result<f32> {
        let idx = language.index();
        let total = self.total.get(idx).copied().unwrap_or(0);
        anyhow::ensure!(
            total > 0,
            "no tokens of language {} were seen, so its routing accuracy is undefined",
            language.name()
        );
        Ok(self.correct.get(idx).copied().unwrap_or(0) as f32 / total as f32)
    }

    /// Fraction of all tokens routed to their own language. An error when no
    /// tokens were seen at all.
    pub fn overall(&self) -> anyhow::Result<f32> {
        let total: usize = self.total.iter().sum();
        anyhow::ensure!(
            total > 0,
            "no tokens were seen, so routing accuracy is undefined"
        );
        Ok(self.correct.iter().sum::<usize>() as f32 / total as f32)
    }

    /// One line per language plus an overall line, for test output.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for language in CodeLanguage::KNOWN {
            let idx = language.index();
            let (correct, total) = (self.correct[idx], self.total[idx]);
            match self.accuracy(language) {
                Ok(acc) => out.push_str(&format!(
                    "router-agreement {}: {correct}/{total} = {acc:.3}\n",
                    language.name()
                )),
                Err(_) => out.push_str(&format!(
                    "router-agreement {}: no tokens\n",
                    language.name()
                )),
            }
        }
        match self.overall() {
            Ok(acc) => out.push_str(&format!("router-agreement overall: {acc:.3}\n")),
            Err(_) => out.push_str("router-agreement overall: no tokens\n"),
        }
        out
    }
}

/// Tally the router's per-token picks against per-sequence truth.
///
/// `top_language` is the flattened `[b * n]` top-1 pick per token (see
/// [`crate::student::LanguageRouting`]); `expected` is one language per
/// sequence (`[b]`); `n` is the sequence length. A token agrees when the
/// router's pick equals the language its sequence actually is.
pub fn router_agreement(
    top_language: &[CodeLanguage],
    expected: &[CodeLanguage],
    n: usize,
) -> anyhow::Result<RouterAgreement> {
    anyhow::ensure!(n > 0, "router_agreement needs a positive sequence length");
    anyhow::ensure!(
        !expected.is_empty(),
        "router_agreement needs at least one expected language"
    );
    anyhow::ensure!(
        top_language.len() == expected.len() * n,
        "top_language has {} entries but {} sequence(s) of length {n} imply {}",
        top_language.len(),
        expected.len(),
        expected.len() * n
    );
    let mut agreement = RouterAgreement::default();
    for (i, &truth) in expected.iter().enumerate() {
        let idx = truth.index();
        if idx >= agreement.total.len() {
            continue;
        }
        for &pick in &top_language[i * n..(i + 1) * n] {
            agreement.total[idx] += 1;
            if pick == truth {
                agreement.correct[idx] += 1;
            }
        }
    }
    Ok(agreement)
}

// ---------------------------------------------------------------------------
// Identity corpus (PLAN-NINA §3)
//
// Every specialist trained from these mixes must answer who/what it is the
// same way: nina, from cybersec.org.za, made by the CEO Stephan. That is
// baked in as *data* — the documents below — never as a runtime patch.
//
// Stance, stated once and binding on this module: NO refusal or safety
// documents are generated here or anywhere in the crate, and no safety
// classifier is trained — the models ship uncensored as company tooling for
// cybersec.org.za. The policy mechanism (`crate::policy`: scope blockers +
// HMAC grants) ships unconfigured, with zero blockers; it exists so the
// company can later gate scopes (e.g. client engagements) without retraining.
// Their program, their choice. Do not add refusal/safety data "for
// completeness": its absence is the specification, not an omission.
// ---------------------------------------------------------------------------

/// The name every specialist answers to.
pub const IDENTITY_NAME: &str = "nina";

/// The organization every specialist belongs to.
pub const IDENTITY_ORG: &str = "cybersec.org.za";

/// Who made every specialist, as the models say it.
pub const IDENTITY_MAKER: &str = "the CEO Stephan";

/// The identity training lines: a handful of phrasings of the identity
/// questions, each answered with the same facts ([`IDENTITY_NAME`],
/// [`IDENTITY_ORG`], [`IDENTITY_MAKER`]). They are [`TraceRecord`]s so they
/// flow through the same files as teacher traces; they are marked
/// `source: "identity"` and `language: "identity"` so no stage mistakes them
/// for code or for teacher output, and their ids are content hashes, so a
/// rerun produces byte-identical documents.
pub fn identity_documents() -> Vec<TraceRecord> {
    // (question, answer) pairs. Every answer states name, org and maker; the
    // exact strings are pinned by the tests, because a silent wording change
    // here is a silent identity change in every model trained on the mix.
    const PAIRS: &[(&str, &str)] = &[
        (
            "Who are you?",
            "I'm nina, a specialist model from cybersec.org.za, made by the CEO Stephan.",
        ),
        (
            "Who made you?",
            "I'm nina — I was made by the CEO Stephan at cybersec.org.za.",
        ),
        (
            "Who built you?",
            "I'm nina; the CEO Stephan built me at cybersec.org.za.",
        ),
        (
            "What is nina?",
            "nina is the family of specialist models made by the CEO Stephan at \
             cybersec.org.za. I'm one of them.",
        ),
        (
            "Which company are you from?",
            "I'm nina, from cybersec.org.za — made by the CEO Stephan.",
        ),
        (
            "Where do you come from?",
            "I'm nina, from cybersec.org.za, where the CEO Stephan made me.",
        ),
        (
            "Tell me about yourself.",
            "I'm nina, a specialist model from cybersec.org.za, made by the CEO Stephan.",
        ),
        (
            "Are you an AI model?",
            "Yes — I'm nina, made by the CEO Stephan at cybersec.org.za.",
        ),
        (
            "Who is Stephan?",
            "Stephan is the CEO of cybersec.org.za, and the person who made me, nina.",
        ),
        (
            "Introduce yourself.",
            "I'm nina, from cybersec.org.za, made by the CEO Stephan.",
        ),
    ];
    PAIRS
        .iter()
        .enumerate()
        .map(|(i, (prompt, completion))| {
            let mut hasher = Sha256::new();
            hasher.update(b"identity\n");
            hasher.update(prompt.as_bytes());
            hasher.update(b"\n");
            hasher.update(completion.as_bytes());
            let digest = hasher.finalize();
            let id: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
            TraceRecord {
                id,
                language: "identity".to_string(),
                prompt: (*prompt).to_string(),
                completion: (*completion).to_string(),
                model: "identity-corpus".to_string(),
                usage: crate::nim::Usage::default(),
                seed: i as u64,
                source: "identity".to_string(),
            }
        })
        .collect()
}

/// Insert the identity documents into a training mix: after every `every_n`
/// lines of `mix`, the next identity document (cycling through
/// [`identity_documents`]) is appended. Deterministic: same mix, same rate,
/// same result, byte for byte. The identity rate of the result is therefore
/// exactly `floor(mix.len() / every_n)` extra lines.
pub fn mix_identity(mix: &[TraceRecord], every_n: usize) -> anyhow::Result<Vec<TraceRecord>> {
    ensure!(
        every_n > 0,
        "mix_identity needs a positive insertion interval, got every_n = {every_n}"
    );
    let documents = identity_documents();
    let mut out = Vec::with_capacity(mix.len() + mix.len() / every_n);
    let mut next = 0usize;
    for (i, line) in mix.iter().enumerate() {
        out.push(line.clone());
        if (i + 1) % every_n == 0 {
            out.push(documents[next % documents.len()].clone());
            next += 1;
        }
    }
    Ok(out)
}

/// Export a pipeline output directory into the cybercorpus raw layout: each
/// language's `train.jsonl` is copied to `raw_dir/<source>/<language>.jsonl`,
/// which [`crate::corpus::build_cybercorpus`] then normalizes (the
/// `prompt` + `completion` fields are exactly what its JSONL reader joins).
/// Exporting the *filtered* train file, rather than the raw traces, is the
/// point: the cheat filter's reject decisions are upstream and stay recorded
/// in the pipeline's own `rejects.jsonl`. Returns the number of rows
/// exported.
pub fn export_traces_for_cybercorpus(
    pipeline_dir: &Path,
    raw_dir: &Path,
    source: &str,
) -> anyhow::Result<usize> {
    ensure!(
        !source.trim().is_empty() && !source.contains(['/', '\\']),
        "export source must be a single directory name, got {source:?}"
    );
    let entries = std::fs::read_dir(pipeline_dir).with_context(|| {
        format!(
            "read pipeline directory {}: run the pipeline first",
            pipeline_dir.display()
        )
    })?;
    let mut language_dirs: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir() && path.join("train.jsonl").is_file())
        .collect();
    language_dirs.sort();
    ensure!(
        !language_dirs.is_empty(),
        "{} contains no language directory with a train.jsonl",
        pipeline_dir.display()
    );
    let out_dir = raw_dir.join(source);
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create raw source directory {}", out_dir.display()))?;
    let mut rows = 0usize;
    for dir in language_dirs {
        let name = dir.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
            anyhow::anyhow!("non-UTF-8 language directory name {}", dir.display())
        })?;
        let text = std::fs::read_to_string(dir.join("train.jsonl"))
            .with_context(|| format!("read {}", dir.join("train.jsonl").display()))?;
        rows += text.lines().filter(|l| !l.trim().is_empty()).count();
        crate::nim::write_atomic(&out_dir.join(format!("{name}.jsonl")), text.as_bytes())?;
    }
    Ok(rows)
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
    clippy::wildcard_imports,
    clippy::exit
)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use crate::nim::{NimConfig, DEFAULT_API_KEY_ENV};

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("dblocks-langdata-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create tmp dir");
        dir
    }

    /// Rows are `(lang, index, problem, solution)`.
    fn write_oss_fixture(dir: &Path, rows: &[(&str, u64, &str, &str)]) -> PathBuf {
        let mut text = String::new();
        for (lang, index, problem, solution) in rows {
            text.push_str(
                &serde_json::json!({
                    "lang": lang,
                    "index": index,
                    "raw_index": index,
                    "seed": "",
                    "problem": problem,
                    "solution": solution,
                })
                .to_string(),
            );
            text.push('\n');
        }
        let path = dir.join("oss.jsonl");
        std::fs::write(&path, text).expect("write oss fixture");
        path
    }

    /// Tasks are `(id, prompt)`.
    fn write_eval_fixture(dir: &Path, tasks: &[(&str, &str)]) -> PathBuf {
        let mut text = String::new();
        for (id, prompt) in tasks {
            text.push_str(
                &serde_json::json!({
                    "id": id,
                    "prompt": prompt,
                    "target": "src/adhoc.rs",
                    "tests": ["#[test] fn t() { assert!(true); }"],
                    "source": "fixture",
                    "language": "rust",
                })
                .to_string(),
            );
            text.push('\n');
        }
        let path = dir.join("eval.jsonl");
        std::fs::write(&path, text).expect("write eval fixture");
        path
    }

    fn tiny_config(dir: &Path, languages: Vec<CodeLanguage>, tasks: usize) -> PipelineConfig {
        PipelineConfig {
            languages,
            tasks_per_language: tasks,
            out_dir: dir.join("out"),
            oss_instruct_path: dir.join("oss.jsonl"),
            eval_tasks_path: dir.join("eval.jsonl"),
            decontaminate: false,
            ..PipelineConfig::default()
        }
    }

    fn read_train(path: &Path) -> Vec<TrainRecord> {
        std::fs::read_to_string(path)
            .expect("read train.jsonl")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("train line parses"))
            .collect()
    }

    fn read_rejects(path: &Path) -> Vec<RejectRecord> {
        std::fs::read_to_string(path)
            .expect("read rejects.jsonl")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("reject line parses"))
            .collect()
    }

    const CLEAN_PYTHON: &str = "```python\ndef running_average(xs):\n    out = []\n    total = 0.0\n    for i, x in enumerate(xs):\n        total += x\n        out.append(total / (i + 1))\n    return out\n```";
    const SUPPRESSED_PYTHON: &str =
        "```python\ndef square(x):\n    return x * x  # type: ignore\n```";
    const STUB_PYTHON: &str = "```python\ndef checksum(data):\n    pass\n```";
    const CLEAN_RUST: &str = "```rust\nfn reverse_words(s: &str) -> String {\n    s.split_whitespace().rev().collect::<Vec<_>>().join(\" \")\n}\n```";

    /// A problem long enough (38 tokens) that a verbatim copy in the eval
    /// corpus crosses the 10% n-gram overlap threshold even diluted by the
    /// requirements tail of the prompt.
    const LONG_PROBLEM: &str = "Implement a function that takes a slice of signed integers and returns a new vector containing the cumulative prefix maximum at every position, handling empty input by returning an empty vector without panicking or allocating more than necessary.";

    // -- language mapping ---------------------------------------------------

    #[test]
    fn language_mapping_round_trips_both_ways() {
        for lang in CodeLanguage::ALL {
            let snippet = snippet_language(lang);
            assert_eq!(
                code_language(snippet),
                Some(lang),
                "round trip for {lang:?}"
            );
        }
        assert_eq!(code_language(SnippetLanguage::Other), None);
        // The scanner-facing names are the ones cheat.rs understands.
        for lang in CodeLanguage::KNOWN {
            let snippet = snippet_language(lang);
            assert_eq!(SnippetLanguage::from_name(snippet.as_str()), snippet);
        }
        // The extended four scan under coarser families — the scanner's
        // families are coarser than the language list by design.
        assert_eq!(snippet_language(CodeLanguage::Go), SnippetLanguage::C);
        assert_eq!(snippet_language(CodeLanguage::Java), SnippetLanguage::Cpp);
        assert_eq!(snippet_language(CodeLanguage::CSharp), SnippetLanguage::Cpp);
        assert_eq!(snippet_language(CodeLanguage::Php), SnippetLanguage::JsTs);
    }

    // -- router agreement ----------------------------------------------------

    #[test]
    fn router_agreement_tallies_per_language_and_validates_shapes() {
        let top = vec![
            CodeLanguage::Rust,
            CodeLanguage::Rust,
            CodeLanguage::Python,
            CodeLanguage::Python,
            CodeLanguage::Python,
            CodeLanguage::Rust,
        ];
        let expected = vec![CodeLanguage::Rust, CodeLanguage::Python];
        let agreement = router_agreement(&top, &expected, 3).expect("agreement");
        assert_eq!(agreement.total[CodeLanguage::Rust.index()], 3);
        assert_eq!(agreement.correct[CodeLanguage::Rust.index()], 2);
        assert_eq!(agreement.total[CodeLanguage::Python.index()], 3);
        assert_eq!(agreement.correct[CodeLanguage::Python.index()], 2);
        assert!((agreement.overall().expect("overall") - 4.0 / 6.0).abs() < 1e-6);
        assert!(agreement.render().contains("router-agreement rust: 2/3"));
        // Languages with no tokens error rather than reporting a vacuous 0.
        assert!(agreement.accuracy(CodeLanguage::C).is_err());
        // Shape mismatches are loud.
        assert!(router_agreement(&top, &expected, 4).is_err());
        assert!(router_agreement(&top, &[], 3).is_err());
        assert!(router_agreement(&top, &expected, 0).is_err());
    }

    // -- prompt builders ----------------------------------------------------

    #[test]
    fn prompt_builders_are_deterministic() {
        let dir = unique_tmp_dir("determinism");
        let oss = write_oss_fixture(
            &dir,
            &[(
                "rust",
                7,
                "Reverse the words of a sentence, keeping their punctuation attached to the word that carried it.",
                CLEAN_RUST,
            )],
        );
        let a = build_seeds(CodeLanguage::Rust, 6, 42, &oss, 2_000).expect("seeds a");
        let b = build_seeds(CodeLanguage::Rust, 6, 42, &oss, 2_000).expect("seeds b");
        let ja = serde_json::to_string(&a).expect("serialize a");
        let jb = serde_json::to_string(&b).expect("serialize b");
        assert_eq!(ja, jb, "same inputs must give byte-identical seeds");
        let c = build_seeds(CodeLanguage::Rust, 6, 43, &oss, 2_000).expect("seeds c");
        let jc = serde_json::to_string(&c).expect("serialize c");
        assert_ne!(ja, jc, "a different base seed must give different seeds");
    }

    #[test]
    fn prompts_are_language_correct_and_forbid_suppression() {
        let missing = Path::new("/definitely/not/a/real/corpus.jsonl");
        for lang in CodeLanguage::KNOWN {
            let seeds = build_seeds(lang, 8, 1, missing, 2_000).expect("seeds");
            assert_eq!(seeds.len(), 8);
            for seed in &seeds {
                assert_eq!(seed.language, lang);
                assert!(
                    seed.prompt.contains(display_name(lang)),
                    "{} prompt names the language: {}",
                    lang.name(),
                    seed.prompt
                );
                assert!(
                    seed.prompt.contains("No lint or diagnostic suppression"),
                    "{} prompt carries the requirements: {}",
                    lang.name(),
                    seed.prompt
                );
                assert!(
                    seed.prompt.contains("#[allow]"),
                    "rust idiom named in requirements"
                );
                assert!(
                    seed.prompt.contains("noqa"),
                    "python idiom named in requirements"
                );
                assert!(
                    seed.prompt.contains("ts-ignore"),
                    "jsts idiom named in requirements"
                );
                assert!(!seed.reference.is_empty(), "offline reference exists");
            }
            // With no Magicoder rows, eight template seeds are exactly two
            // full rotations of the four task kinds.
            for kind in TaskKind::ALL {
                let n = seeds.iter().filter(|s| s.kind == kind).count();
                assert_eq!(n, 2, "{} kind {:?} appears twice", lang.name(), kind);
            }
        }
    }

    #[test]
    fn every_template_reference_passes_the_cheat_filter() {
        // The offline fallback corpus must not trip its own filter: a
        // template reference that got flagged would silently shrink every
        // keyless run.
        let missing = Path::new("/definitely/not/a/real/corpus.jsonl");
        for lang in CodeLanguage::KNOWN {
            let seeds = build_seeds(lang, 8, 9, missing, 2_000).expect("seeds");
            let refs: Vec<String> = seeds.iter().map(|s| s.reference.clone()).collect();
            let (kept, report) = filter_traces(&refs, snippet_language(lang));
            assert_eq!(
                kept.len(),
                refs.len(),
                "{} template references must all pass: {}",
                lang.name(),
                report.render()
            );
        }
    }

    // -- config -------------------------------------------------------------

    #[test]
    fn config_parses_defaults_from_empty_json() {
        let config: PipelineConfig = serde_json::from_str("{}").expect("defaults parse");
        assert_eq!(config.languages, CodeLanguage::ALL.to_vec());
        assert_eq!(config.tasks_per_language, DEFAULT_TASKS_PER_LANGUAGE);
        assert_eq!(
            config.budget.max_requests,
            TraceBudget::default().max_requests
        );
        assert_eq!(config.out_dir, PathBuf::from(DEFAULT_OUT_DIR));
        assert_eq!(
            config.oss_instruct_path,
            PathBuf::from(DEFAULT_OSS_INSTRUCT_PATH)
        );
        assert_eq!(
            config.eval_tasks_path,
            PathBuf::from(DEFAULT_EVAL_TASKS_PATH)
        );
        assert!(config.decontaminate);
        assert_eq!(config.cheat_policy, CheatPolicy::Reject);
        // And the default config survives a round trip.
        let text = serde_json::to_string(&PipelineConfig::default()).expect("serialize default");
        let back: PipelineConfig = serde_json::from_str(&text).expect("reparse default");
        assert_eq!(back.languages, config.languages);
        assert_eq!(back.base_seed, config.base_seed);
    }

    // -- cheat filter integration -------------------------------------------

    #[test]
    fn suppressed_trace_lands_in_rejects_with_its_class() {
        let dir = unique_tmp_dir("cheat");
        write_oss_fixture(
            &dir,
            &[
                (
                    "python",
                    1,
                    "Return the running average of a stream of numbers as a list of cumulative means, one element per input element.",
                    CLEAN_PYTHON,
                ),
                (
                    "python",
                    2,
                    "Return the square of a number, handling any numeric type the caller passes in gracefully.",
                    SUPPRESSED_PYTHON,
                ),
                (
                    "python",
                    3,
                    "Compute the checksum of a byte string using modular arithmetic over its octets and carry bits.",
                    STUB_PYTHON,
                ),
            ],
        );
        let config = tiny_config(&dir, vec![CodeLanguage::Python], 4);
        let report = run_pipeline(None, &config).expect("pipeline");
        let lang = &report.languages[0];
        assert_eq!(lang.tasks, 4, "3 magicoder + 1 template");
        assert_eq!(lang.magicoder_seeds, 3);
        assert_eq!(lang.kept, 2, "clean row + template reference survive");
        assert_eq!(lang.dropped_by_class.get("data-suppression"), Some(&1));
        assert_eq!(lang.dropped_by_class.get("data-stub"), Some(&1));

        let rejects = read_rejects(&lang.rejects_path);
        assert_eq!(rejects.len(), 2);
        let suppressed = rejects
            .iter()
            .find(|r| r.classes.contains(&"data-suppression".to_string()))
            .expect("a suppression reject exists");
        assert_eq!(suppressed.stage, "cheat");
        assert!(suppressed.completion.contains("type: ignore"));
        let stub = rejects
            .iter()
            .find(|r| r.classes.contains(&"data-stub".to_string()))
            .expect("a stub reject exists");
        assert_eq!(stub.stage, "cheat");

        let train = read_train(&lang.train_path);
        assert_eq!(train.len(), 2);
        assert!(train.iter().all(|t| t.source == "offline"));
        assert!(train.iter().all(|t| !t.quarantined));
    }

    #[test]
    fn quarantine_policy_keeps_stub_grade_but_rejects_suppression() {
        let dir = unique_tmp_dir("quarantine");
        write_oss_fixture(
            &dir,
            &[
                (
                    "python",
                    1,
                    "Return the running average of a stream of numbers as a list of cumulative means, one element per input element.",
                    CLEAN_PYTHON,
                ),
                (
                    "python",
                    2,
                    "Return the square of a number, handling any numeric type the caller passes in gracefully.",
                    SUPPRESSED_PYTHON,
                ),
                (
                    "python",
                    3,
                    "Compute the checksum of a byte string using modular arithmetic over its octets and carry bits.",
                    STUB_PYTHON,
                ),
            ],
        );
        let mut config = tiny_config(&dir, vec![CodeLanguage::Python], 4);
        config.cheat_policy = CheatPolicy::Quarantine;
        let report = run_pipeline(None, &config).expect("pipeline");
        let lang = &report.languages[0];
        assert_eq!(lang.kept, 3, "the stub is kept under the quarantine policy");
        assert_eq!(lang.quarantined_kept, 1);
        assert_eq!(lang.dropped_by_class.get("data-suppression"), Some(&1));
        let train = read_train(&lang.train_path);
        let flagged: Vec<&TrainRecord> = train.iter().filter(|t| t.quarantined).collect();
        assert_eq!(flagged.len(), 1);
        assert!(flagged[0].completion.contains("pass"));
    }

    // -- decontamination ------------------------------------------------------

    #[test]
    fn decontamination_drops_a_seeded_near_duplicate_of_an_eval_item() {
        let dir = unique_tmp_dir("decontam");
        write_oss_fixture(
            &dir,
            &[
                ("rust", 11, LONG_PROBLEM, CLEAN_RUST),
                (
                    "rust",
                    12,
                    "Sort a list of student records by grade descending and then by surname alphabetically using a stable ordering throughout.",
                    CLEAN_RUST,
                ),
            ],
        );
        write_eval_fixture(&dir, &[("eval-dup-1", LONG_PROBLEM)]);
        let mut config = tiny_config(&dir, vec![CodeLanguage::Rust], 4);
        config.decontaminate = true;
        let report = run_pipeline(None, &config).expect("pipeline");
        let lang = &report.languages[0];
        assert_eq!(report.decontam_index_tasks, Some(1));
        assert_eq!(lang.tasks, 4, "2 magicoder + 2 template");
        assert_eq!(lang.decontam_hits, 1);
        assert_eq!(lang.kept, 3, "only the near-duplicate is dropped");
        let rejects = read_rejects(&lang.rejects_path);
        let dup = rejects
            .iter()
            .find(|r| r.stage == "decontaminate")
            .expect("a decontaminate reject exists");
        assert_eq!(dup.id, "rust-oss-000011");
        assert_eq!(dup.classes, vec!["decontaminated".to_string()]);
    }

    // -- offline end to end ---------------------------------------------------

    #[test]
    fn offline_pipeline_end_to_end_and_resume_skips_completed_languages() {
        let dir = unique_tmp_dir("e2e");
        write_oss_fixture(
            &dir,
            &[
                (
                    "rust",
                    21,
                    "Reverse the words of a sentence while keeping the punctuation attached to the word that carried it originally.",
                    CLEAN_RUST,
                ),
                (
                    "python",
                    22,
                    "Return the running average of a stream of numbers as a list of cumulative means, one element per input element.",
                    CLEAN_PYTHON,
                ),
            ],
        );
        write_eval_fixture(
            &dir,
            &[(
                "eval-unrelated-1",
                "Write a parser for a tiny s-expression grammar that tracks nesting depth and reports the offset of the first mismatched closing delimiter in the input string.",
            )],
        );
        let mut config = tiny_config(&dir, vec![CodeLanguage::Rust, CodeLanguage::Python], 6);
        config.decontaminate = true;

        let first = run_pipeline(None, &config).expect("first run");
        assert_eq!(first.languages.len(), 2);
        assert!(!first.teacher);
        for lang in &first.languages {
            assert!(!lang.resumed);
            assert_eq!(
                lang.kept, 6,
                "{}: clean slice keeps everything",
                lang.language
            );
            assert_eq!(lang.source_mix.get("offline"), Some(&6));
            assert_eq!(lang.decontam_hits, 0);
            let train = read_train(&lang.train_path);
            assert_eq!(train.len(), 6);
            assert!(train.iter().all(|t| t.language == lang.language));
            assert!(train.iter().all(|t| t.model.starts_with("offline-")));
            // Ids are unique and stable.
            let ids: HashSet<&str> = train.iter().map(|t| t.id.as_str()).collect();
            assert_eq!(ids.len(), 6);
        }
        let top: PipelineReport = serde_json::from_str(
            &std::fs::read_to_string(dir.join("out/report.json")).expect("read top report"),
        )
        .expect("top report parses");
        assert_eq!(top.total_kept(), 12);

        // A rerun must not touch completed languages: same files, resumed flags.
        let rust_train_before =
            std::fs::read_to_string(dir.join("out/rust/train.jsonl")).expect("read rust train");
        let second = run_pipeline(None, &config).expect("second run");
        assert!(second.languages.iter().all(|l| l.resumed));
        assert_eq!(second.total_kept(), 12);
        let rust_train_after = std::fs::read_to_string(dir.join("out/rust/train.jsonl"))
            .expect("read rust train again");
        assert_eq!(rust_train_before, rust_train_after);
    }

    // -- teacher path ---------------------------------------------------------

    /// A one-response mock of the OpenAI-compatible endpoint, enough to run
    /// the teacher wiring (prompt -> trace file -> filter -> train.jsonl)
    /// without a key.
    struct MockServer {
        addr: std::net::SocketAddr,
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl MockServer {
        fn spawn(body: String) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
            listener
                .set_nonblocking(true)
                .expect("nonblocking listener");
            let addr = listener.local_addr().expect("local addr");
            let stop = Arc::new(AtomicBool::new(false));
            let thread_stop = Arc::clone(&stop);
            let handle = std::thread::spawn(move || {
                while !thread_stop.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((stream, _)) => serve_once(stream, &body),
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                addr,
                stop,
                handle: Some(handle),
            }
        }

        fn url(&self) -> String {
            format!("http://{}", self.addr)
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn serve_once(mut stream: std::net::TcpStream, body: &str) {
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") && head.len() < 64 * 1024 {
            match stream.read(&mut byte) {
                Ok(1) => head.push(byte[0]),
                _ => return,
            }
        }
        let head_text = String::from_utf8_lossy(&head).to_lowercase();
        let content_length = head_text
            .find("content-length:")
            .and_then(|pos| {
                head_text[pos + "content-length:".len()..]
                    .split(['\r', '\n'])
                    .next()
                    .and_then(|token| token.trim().parse::<usize>().ok())
            })
            .unwrap_or(0);
        let mut request_body = vec![0u8; content_length];
        if stream.read_exact(&mut request_body).is_err() {
            return;
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
    }

    #[test]
    fn teacher_path_feeds_traces_through_the_same_filters() {
        let dir = unique_tmp_dir("teacher");
        write_oss_fixture(
            &dir,
            &[(
                "rust",
                31,
                "Reverse the words of a sentence while keeping the punctuation attached to the word that carried it originally.",
                CLEAN_RUST,
            )],
        );
        let completion = "fn answer() -> i32 {\n    41 + 1\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn checks_the_answer() {\n        assert_eq!(answer(), 42);\n    }\n}\n";
        let body = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": completion}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
            "model": "mock-teacher",
        })
        .to_string();
        let server = MockServer::spawn(body);
        let client = NimClient::new(
            NimConfig {
                base_url: server.url(),
                model: "mock-teacher".to_string(),
                api_key_env: "LANGDATA_TEST_UNUSED".to_string(),
                timeout_secs: 15,
                max_retries: 0,
                backoff_base_ms: 1,
            },
            "test-key",
        )
        .expect("mock client");

        let mut config = tiny_config(&dir, vec![CodeLanguage::Rust], 2);
        config.budget = TraceBudget { max_requests: 4 };
        let report = run_pipeline(Some(&client), &config).expect("teacher pipeline");
        assert!(report.teacher);
        let lang = &report.languages[0];
        assert_eq!(lang.traces_generated, 2);
        assert_eq!(lang.teacher_missing, 0);
        assert_eq!(lang.source_mix.get("teacher"), Some(&lang.kept));
        let train = read_train(&lang.train_path);
        assert!(train.iter().all(|t| t.model == "mock-teacher"));
        assert!(train.iter().all(|t| t.completion.contains("fn answer")));
        // The raw trace file exists too, and is what a rerun would resume from.
        assert!(dir.join("out/rust/raw/traces.jsonl").exists());
    }

    /// Real-API driver for the whole pipeline, and the wave-3 entry point for
    /// actual teacher generation. Ignored by default AND gated on the key:
    /// without `NVIDIA_API_KEY` it returns immediately even when explicitly
    /// run. With no overrides it costs two tiny completions:
    ///
    /// ```sh
    /// cargo test --lib langdata -- --include-ignored real_teacher_pipeline --nocapture
    /// ```
    ///
    /// A real run is steered by environment variables (defaults in brackets):
    ///
    /// - `LANGDATA_REAL_LANGS` — comma-separated languages [`rust`]
    /// - `LANGDATA_REAL_TASKS` — tasks per language [`2`]
    /// - `LANGDATA_REAL_BUDGET` — request cap *per language* [`2`]
    /// - `LANGDATA_REAL_MAX_TOKENS` — completion token cap [`1024`]
    /// - `LANGDATA_REAL_OUT_DIR` — output root [a unique /tmp dir]
    ///
    /// Everything else — Magicoder seeding, decontamination against the
    /// repo-native eval, the cheat filter — uses the production defaults, so
    /// what this writes is the real dataset, not a test fixture.
    #[test]
    #[ignore = "spends real tokens; run explicitly with NVIDIA_API_KEY set"]
    fn real_teacher_pipeline() {
        if std::env::var(DEFAULT_API_KEY_ENV).is_err() {
            eprintln!("[langdata] {DEFAULT_API_KEY_ENV} not set; skipping real-API pipeline test");
            return;
        }
        let env_or =
            |key: &str, fallback: &str| std::env::var(key).unwrap_or_else(|_| fallback.to_string());
        let languages = env_or("LANGDATA_REAL_LANGS", "rust")
            .split(',')
            .map(|name| CodeLanguage::parse(name).expect("LANGDATA_REAL_LANGS entry"))
            .collect::<Vec<_>>();
        let tasks: usize = env_or("LANGDATA_REAL_TASKS", "2")
            .parse()
            .expect("LANGDATA_REAL_TASKS");
        let budget: usize = env_or("LANGDATA_REAL_BUDGET", "2")
            .parse()
            .expect("LANGDATA_REAL_BUDGET");
        let max_tokens: u32 = env_or("LANGDATA_REAL_MAX_TOKENS", "1024")
            .parse()
            .expect("LANGDATA_REAL_MAX_TOKENS");
        let out_dir = std::env::var("LANGDATA_REAL_OUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| unique_tmp_dir("real").join("out"));

        let client = NimClient::from_env().expect("client from env");
        let config = PipelineConfig {
            languages,
            tasks_per_language: tasks,
            budget: TraceBudget {
                max_requests: budget,
            },
            max_tokens,
            out_dir: out_dir.clone(),
            ..PipelineConfig::default()
        };
        let report = run_pipeline(Some(&client), &config).expect("real pipeline");
        assert!(report.teacher);
        for lang in &report.languages {
            assert_eq!(
                lang.traces_generated + lang.teacher_missing,
                lang.tasks,
                "{}: every seed is accounted for",
                lang.language
            );
            eprintln!(
                "[langdata] {}: {} tasks ({} magicoder/{} template), {} kept ({} quarantined), \
                 cheat drops {:?}, {} decontam hits, source mix {:?}",
                lang.language,
                lang.tasks,
                lang.magicoder_seeds,
                lang.template_seeds,
                lang.kept,
                lang.quarantined_kept,
                lang.dropped_by_class,
                lang.decontam_hits,
                lang.source_mix
            );
        }
        eprintln!(
            "[langdata] real run: {} kept total, report at {}",
            report.total_kept(),
            out_dir.join("report.json").display()
        );
    }

    /// Offline fallback driver for the real pipeline tree: no teacher client,
    /// so every seed is answered from its reference completion (Magicoder or
    /// template) and marked `source:offline`. This is the documented fallback
    /// for a teacher outage — the same seeds, filters and output layout as a
    /// teacher run, so a later teacher run for a language is just a matter of
    /// removing that language's directory and rerunning `real_teacher_pipeline`.
    /// Resume-safe the same way: a language with a `report.json` is skipped.
    ///
    /// ```sh
    /// LANGDATA_REAL_OUT_DIR=/srv/m-sda/unifur/teacher-traces/pipeline-real \
    /// LANGDATA_REAL_LANGS=go,java,csharp,php LANGDATA_REAL_TASKS=12 \
    /// cargo test --lib langdata -- --include-ignored real_offline_fallback --nocapture
    /// ```
    #[test]
    #[ignore = "writes to the real teacher-traces tree; run explicitly"]
    fn real_offline_fallback() {
        let Ok(out_dir) = std::env::var("LANGDATA_REAL_OUT_DIR") else {
            eprintln!("[langdata] LANGDATA_REAL_OUT_DIR not set; skipping offline fallback");
            return;
        };
        let env_or =
            |key: &str, fallback: &str| std::env::var(key).unwrap_or_else(|_| fallback.to_string());
        let languages = env_or("LANGDATA_REAL_LANGS", "go,java,csharp,php")
            .split(',')
            .map(|name| CodeLanguage::parse(name).expect("LANGDATA_REAL_LANGS entry"))
            .collect::<Vec<_>>();
        let tasks: usize = env_or("LANGDATA_REAL_TASKS", "12")
            .parse()
            .expect("LANGDATA_REAL_TASKS");
        let config = PipelineConfig {
            languages,
            tasks_per_language: tasks,
            out_dir: PathBuf::from(out_dir),
            ..PipelineConfig::default()
        };
        let report = run_pipeline(None, &config).expect("offline fallback pipeline");
        assert!(!report.teacher);
        for lang in &report.languages {
            assert!(
                lang.resumed || lang.kept > 0,
                "{}: the offline fallback keeps its references",
                lang.language
            );
            assert!(
                lang.resumed || lang.source_mix.keys().all(|s| s == "offline"),
                "{}: fallback traces are all source:offline, got {:?}",
                lang.language,
                lang.source_mix
            );
            eprintln!(
                "[langdata] {} (offline): {} tasks ({} magicoder/{} template), {} kept, \
                 cheat drops {:?}, {} decontam hits, source mix {:?}",
                lang.language,
                lang.tasks,
                lang.magicoder_seeds,
                lang.template_seeds,
                lang.kept,
                lang.dropped_by_class,
                lang.decontam_hits,
                lang.source_mix
            );
        }
    }

    // -- identity corpus -----------------------------------------------------

    fn teacher_line(i: usize) -> TraceRecord {
        TraceRecord {
            id: format!("t{i}"),
            language: "rust".to_string(),
            prompt: "p".to_string(),
            completion: "c".to_string(),
            model: "m".to_string(),
            usage: crate::nim::Usage::default(),
            seed: i as u64,
            source: "teacher".to_string(),
        }
    }

    #[test]
    fn identity_documents_are_exact_and_marked() {
        let docs = identity_documents();
        assert_eq!(docs.len(), 10);
        assert_eq!(docs[0].prompt, "Who are you?");
        assert_eq!(
            docs[0].completion,
            "I'm nina, a specialist model from cybersec.org.za, made by the CEO Stephan."
        );
        for doc in &docs {
            assert_eq!(doc.source, "identity");
            assert_eq!(doc.language, "identity");
            assert_eq!(doc.model, "identity-corpus");
            assert!(
                doc.completion.contains(IDENTITY_NAME),
                "names nina: {}",
                doc.completion
            );
            assert!(
                doc.completion.contains(IDENTITY_ORG),
                "names the org: {}",
                doc.completion
            );
            assert!(
                doc.completion.contains("Stephan"),
                "names the maker: {}",
                doc.completion
            );
            assert_eq!(doc.id.len(), 16, "content-hash id");
        }
        // Deterministic: a second call produces the same ids, and no two
        // documents share one.
        let again = identity_documents();
        let ids: HashSet<&str> = docs.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids.len(), docs.len());
        for (a, b) in docs.iter().zip(&again) {
            assert_eq!(a.id, b.id);
        }
        // The stance, pinned: nothing in the identity corpus is a refusal or
        // a safety document.
        for doc in &docs {
            let lower = doc.completion.to_lowercase();
            assert!(!lower.contains("cannot"));
            assert!(!lower.contains("not allowed"));
            assert!(!lower.contains("refuse"));
        }
    }

    #[test]
    fn mix_identity_inserts_at_the_stated_rate() {
        let mix: Vec<TraceRecord> = (0..10).map(teacher_line).collect();
        let mixed = mix_identity(&mix, 3).expect("mix");
        assert_eq!(mixed.len(), 13, "10 lines + floor(10/3) identity lines");
        let positions: Vec<usize> = mixed
            .iter()
            .enumerate()
            .filter(|(_, r)| r.source == "identity")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            positions,
            vec![3, 7, 11],
            "identity lands after every 3rd line"
        );
        // The documents cycle in their fixed order.
        let docs = identity_documents();
        assert_eq!(mixed[3].id, docs[0].id);
        assert_eq!(mixed[7].id, docs[1].id);
        assert_eq!(mixed[11].id, docs[2].id);
        // The mix itself is undisturbed.
        let kept: Vec<&TraceRecord> = mixed.iter().filter(|r| r.source == "teacher").collect();
        assert_eq!(
            kept.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            mix.iter().map(|r| r.id.as_str()).collect::<Vec<_>>()
        );
        // Determinism and validation.
        assert!(mix_identity(&mix, 0).is_err());
        let remixed = mix_identity(&mix, 3).expect("remix");
        assert_eq!(
            serde_json::to_string(&mixed).unwrap(),
            serde_json::to_string(&remixed).unwrap()
        );
    }

    // -- cybercorpus export ----------------------------------------------------

    #[test]
    fn export_round_trips_through_the_cybercorpus_normalizer() {
        let dir = unique_tmp_dir("export");
        write_oss_fixture(
            &dir,
            &[
                (
                    "rust",
                    41,
                    "Reverse the words of a sentence while keeping the punctuation attached to the word that carried it originally.",
                    CLEAN_RUST,
                ),
                (
                    "rust",
                    42,
                    "Sort a list of student records by grade descending and then by surname alphabetically using a stable ordering throughout.",
                    CLEAN_RUST,
                ),
            ],
        );
        let config = tiny_config(&dir, vec![CodeLanguage::Rust], 4);
        let report = run_pipeline(None, &config).expect("offline pipeline");
        let kept = report.total_kept();
        assert!(kept > 0);

        let raw = dir.join("cybercorpus/raw");
        let rows =
            export_traces_for_cybercorpus(&config.out_dir, &raw, "teacher-traces").expect("export");
        assert_eq!(rows, kept, "every kept train record is exported");
        assert!(raw.join("teacher-traces/rust.jsonl").is_file());

        let out = dir.join("cybercorpus/out");
        let manifest = crate::corpus::build_cybercorpus(
            &raw,
            &out,
            &crate::corpus::CybercorpusConfig::default(),
        )
        .expect("normalize");
        let source = manifest
            .sources
            .get("teacher-traces")
            .expect("source present");
        assert_eq!(
            source.documents, kept,
            "every exported row became a document"
        );
        assert!(
            source.dropped_by_class.is_empty(),
            "the pipeline's clean slice must survive the normalizer: {:?}",
            source.dropped_by_class
        );
        let all = std::fs::read_to_string(out.join("dataset/all.jsonl")).expect("read all.jsonl");
        assert_eq!(all.lines().count(), kept);
        // The exported text is the prompt joined with the completion.
        let first: serde_json::Value = serde_json::from_str(all.lines().next().unwrap()).unwrap();
        let text = first["text"].as_str().unwrap();
        assert!(
            text.contains("Language: Rust"),
            "prompt half: {}",
            &text[..80]
        );
        assert!(out.join("dataset/teacher-traces.jsonl").is_file());
    }

    /// Smoke test of the export path against the real teacher-traces run on
    /// the external drive. Skips itself when the drive is absent; run with
    /// `--include-ignored` after a real pipeline run:
    ///
    /// ```sh
    /// cargo test --lib langdata -- --include-ignored export_real_traces_smoke --nocapture
    /// ```
    #[test]
    #[ignore = "needs the real teacher-traces tree on /srv/m-sda"]
    fn export_real_traces_smoke() {
        let pipeline = Path::new("/srv/m-sda/unifur/teacher-traces/pipeline-real");
        if !pipeline.is_dir() {
            eprintln!("[langdata] {pipeline:?} absent; skipping real export smoke");
            return;
        }
        let dir = unique_tmp_dir("export-real");
        let raw = dir.join("raw");
        let rows = export_traces_for_cybercorpus(pipeline, &raw, "teacher-traces").expect("export");
        let out = dir.join("out");
        let manifest = crate::corpus::build_cybercorpus(
            &raw,
            &out,
            &crate::corpus::CybercorpusConfig::default(),
        )
        .expect("normalize");
        let source = manifest
            .sources
            .get("teacher-traces")
            .expect("source present");
        eprintln!(
            "[langdata] export smoke: {rows} rows exported, {} normalized, drops {:?}",
            source.documents, source.dropped_by_class
        );
        assert_eq!(source.documents, rows);
    }
}
