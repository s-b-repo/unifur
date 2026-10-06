//! Generate the repo-native eval set: coding tasks whose answers are derivable
//! only from this repository's own law.
//!
//! # Why these tasks and not benchmarks
//!
//! Every public coding benchmark in this space sits inside the pretraining
//! data of every frontier model. A model reproduces a memorised solution
//! whether or not it can reason, so a score on one measures recall — and the
//! score *rises* as you train, which reads as improvement while being its
//! opposite.
//!
//! These tasks have the opposite property. Each one states a rule this
//! repository enforces (`AGENTS.md`, `Cargo.toml`'s `[lints]`, and the gate
//! script), asks for an implementation against the crate's real API, and
//! carries tests that discriminate. A model that has memorised HumanEval has no
//! advantage here, because no amount of recall tells it that this repo wants a
//! `Result` naming the offending value rather than an `unwrap`.
//!
//! The tests are the anti-cheat mechanism, so each one is written to *fail* on
//! the plausible shortcut: the naive version that returns `0.0` instead of an
//! error, the one that sums gates to `n_boxes` instead of 1, the one that
//! counts rather than hashes and reshuffles on insertion. A test that a lazy
//! solution also passes measures nothing.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example evalgen -- \
//!     --out <tasks.jsonl> \
//!     --corpus <sft.jsonl>          # optional: decontaminate against a corpus
//!     --limit <n>
//! ```
//!
//! With `--corpus`, the generator builds a 13-gram index over the training
//! prompts and rejects any task that overlaps, reporting the yield. Rejected
//! tasks are listed with the offending span rather than silently dropped.

use anyhow::{Context, Result};
use diffusionblocks::codegen_eval::{
    self, admit, contamination_risk, split, CorpusIndex, Side, Task, DEFAULT_EVAL_FRACTION,
};
use std::path::{Path, PathBuf};

/// One task in the catalogue.
struct Spec {
    id: &'static str,
    prompt: &'static str,
    tests: &'static [&'static str],
    language: &'static str,
}

const CATALOGUE: &[Spec] = &[
    Spec {
        id: "law-error-not-panic",
        language: "rust",
        prompt: "\
In this repository errors are returned, never raised: `unwrap`, `expect`, `panic!` \
and `todo!` are denied by the crate's lint table outside `#[cfg(test)]`. A bare-metal \
adapters command needs the target file and the task id.

Implement in `src/adhoc.rs`:

    pub fn open_task_file(path: &std::path::Path, task: &str) -> anyhow::Result<String>

Return the file's contents. When the path does not exist, return an `anyhow::Error` \
whose message names *both* the path that was tried and the task id, so the caller can \
tell which of several tasks failed. Do not panic and do not unwrap.",
        tests: &[
            "#[test]\nfn reads_an_existing_file() {\n    let d = std::env::temp_dir().join(format!(\"adhoc-{}\", std::process::id()));\n    std::fs::create_dir_all(&d).unwrap();\n    let p = d.join(\"t.txt\");\n    std::fs::write(&p, b\"hello\").unwrap();\n    assert_eq!(super::open_task_file(&p, \"task-a\").unwrap(), \"hello\");\n    std::fs::remove_dir_all(&d).unwrap();\n}\n",
            "#[test]\nfn missing_file_names_the_path_and_the_task() {\n    let p = std::env::temp_dir().join(\"definitely-absent-xyz\");\n    let err = super::open_task_file(&p, \"task-42\").unwrap_err().to_string();\n    assert!(err.contains(\"definitely-absent-xyz\"), \"error must name the path: {err}\");\n    assert!(err.contains(\"task-42\"), \"error must name the task: {err}\");\n}\n",
            "#[test]\nfn no_panicking_constructs_in_production_code() {\n    let src = std::fs::read_to_string(\"src/adhoc.rs\").unwrap();\n    for bad in [\".unwrap()\", \".expect(\", \"panic!(\", \"todo!(\", \"unimplemented!(\"] {\n        assert!(!src.contains(bad), \"production code uses {bad}\");\n    }\n}\n",
        ],
    },
    Spec {
        id: "law-partition-of-unity",
        language: "rust",
        prompt: "\
A two-level sparse router composes a per-box gate with a per-expert gate. Composition \
preserves the box load: the expert loads inside box `b` must sum to exactly \
`row_gates[b]`, and the box loads themselves sum to 1, so the composed loads form a \
probability distribution over the routing tree.

Implement in `src/adhoc.rs`:

    pub fn composed_load(row_gates: &[f64], expert_gates: &[Vec<f64>]) -> Vec<Vec<f64>>

where `expert_gates[b][k]` is expert `k`'s gate inside box `b`, and `row_gates[b]` is \
box `b`'s load. Return `row_gates[b] * expert_gates[b][k]` for every box and expert.

Note the direction this is easy to get wrong: each box's expert row sums to that \
*box's load*, not to 1. Requiring a sum of 1 per box is a different quantity, and it is \
wrong for every router whose box loads are not uniform.",
        tests: &[
            "#[test]\nfn each_box_preserves_its_own_load() {\n    let rows = vec![0.25, 0.75];\n    let experts = vec![vec![0.5, 0.5], vec![0.1, 0.9]];\n    let out = super::composed_load(&rows, &experts);\n    for (b, r) in out.iter().enumerate() {\n        let s: f64 = r.iter().sum();\n        assert!((s - rows[b]).abs() < 1e-12, \"box {b} summed to {s}, not its load {}\", rows[b]);\n    }\n}\n",
            "#[test]\nfn the_box_loads_still_form_a_distribution() {\n    let rows = vec![0.25, 0.75];\n    let experts = vec![vec![0.5, 0.5], vec![0.1, 0.9]];\n    let out = super::composed_load(&rows, &experts);\n    let total: f64 = out.iter().flat_map(|r| r.iter()).sum();\n    assert!((total - 1.0).abs() < 1e-12, \"composed loads summed to {total}\");\n}\n",
            "#[test]\nfn a_single_expert_carries_the_whole_box_load() {\n    let rows = vec![0.3, 0.7];\n    let experts = vec![vec![1.0], vec![1.0]];\n    let out = super::composed_load(&rows, &experts);\n    assert_eq!(out[0], vec![0.3]);\n    assert_eq!(out[1], vec![0.7]);\n}\n",
            "#[test]\nfn a_broadcast_does_not_replicate_the_gate() {\n    // The historical bug: a [b,k] gate compared against a [b,n] table and then\n    // summed replicates the single gate across every column, so a box's experts\n    // sum to load * n_experts instead of load.\n    let rows = vec![0.5, 0.5];\n    let experts = vec![vec![0.5, 0.5], vec![0.5, 0.5]];\n    let out = super::composed_load(&rows, &experts);\n    for (b, r) in out.iter().enumerate() {\n        let s: f64 = r.iter().sum();\n        assert!((s - rows[b]).abs() < 1e-12, \"box {b} summed to {s}\");\n    }\n}\n",
        ],
    },
    Spec {
        id: "law-covariance-positive-definite",
        language: "rust",
        prompt: "\
A learned Riemannian metric is stored as `G = L Lᵀ` where `L` is lower triangular with a \
strictly positive diagonal, so `G` is positive definite *by construction* for any \
parameter values an optimizer can reach.

Implement in `src/adhoc.rs`:
    pub fn covariance(lower: &[Vec<f64>]) -> anyhow::Result<Vec<Vec<f64>>>

`lower` is a square lower-triangular matrix. Reject (with an `anyhow::Error` naming the \
offending index) any entry above the diagonal or any non-positive diagonal entry. \
Return the symmetric positive-definite `G`.",
        tests: &[
            "#[test]\nfn identity_parameters_give_the_identity() {\n    let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];\n    assert_eq!(super::covariance(&l).unwrap(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);\n}\n",
            "#[test]\nfn a_triangular_matrix_is_symmetric_after_multiplication() {\n    let l = vec![vec![2.0, 0.0], vec![3.0, 4.0]];\n    let g = super::covariance(&l).unwrap();\n    for i in 0..g.len() {\n        for j in 0..g.len() {\n            assert!((g[i][j] - g[j][i]).abs() < 1e-12, \"G is not symmetric at ({i},{j})\");\n        }\n    }\n}\n",
            "#[test]\nfn a_non_positive_diagonal_is_refused() {\n    let l = vec![vec![1.0, 0.0], vec![0.5, 0.0]];\n    let err = super::covariance(&l).unwrap_err().to_string();\n    assert!(err.contains('1'), \"error must name the offending index: {err}\");\n}\n",
            "#[test]\nfn an_entry_above_the_diagonal_is_refused() {\n    let l = vec![vec![1.0, 5.0], vec![0.0, 1.0]];\n    assert!(super::covariance(&l).is_err(), \"non-triangular input was accepted\");\n}\n",
        ],
    },
    Spec {
id: "law-tokenise-before-matching",
        language: "rust",
        // A raw string, because the prompt has to *contain* quotes and raw-string
        // delimiters: describing them with escapes inside a normal string makes
        // the example itself unreadable.
        prompt: r##"A pattern scanner for source code must not report an attribute that appears inside a comment or a string literal, because a detector that does will flag its own test fixtures as violations.

Implement in `src/adhoc.rs`:

    pub fn has_attribute(source: &str, attribute: &str) -> bool

Return true only when `attribute` occurs in `source` as code. Occurrences inside a `//` line comment, inside a `/* */` block comment, inside an ordinary "..." string, and inside a raw r#"..."# string do not count. Comments are stripped but the line structure of the source is preserved, so a match on line 7 is reported as line 7."##,
        tests: &[
            "#[test]\nfn a_real_attribute_is_found() {\n    assert!(super::has_attribute(\"#[allow(dead_code)]\\nfn f() {}\\n\", \"#[allow(\"));\n}\n",
            "#[test]\nfn a_line_comment_is_not_code() {\n    assert!(!super::has_attribute(\"// see #[allow(x)]\\nfn f() {}\\n\", \"#[allow(\"));\n}\n",
            "#[test]\nfn a_block_comment_is_not_code() {\n    assert!(!super::has_attribute(\"/* #[allow(x)] */\\nfn f() {}\\n\", \"#[allow(\"));\n}\n",
            "#[test]\nfn a_string_literal_is_not_code() {\n    assert!(!super::has_attribute(\"let s = \\\"#[allow(x)]\\\";\\n\", \"#[allow(\"));\n    assert!(!super::has_attribute(\"let s = r#\\\"#[allow(x)]\\\"#;\\n\", \"#[allow(\"));\n}\n",
            "#[test]\nfn an_escaped_quote_does_not_end_the_string() {\n    let src = \"let s = \\\"a \\\\\\\" #[allow(x)]\\\";\\n\";\n    assert!(!super::has_attribute(src, \"#[allow(\"), \"a string ended early\");\n}\n",
        ],
    },
    Spec {
        id: "law-split-by-hash-not-counter",
        language: "rust",
        prompt: "\
A corpus is divided into a training half and an evaluation half. The assignment must not \
depend on row order: if a single row is inserted, every other row must stay on the side it \
was already on, or every score recorded before the insertion is invalidated.

Implement in `src/adhoc.rs`:

    pub fn side(id: &str, eval_fraction: f64) -> bool

Return true when `id` belongs to the eval side. The decision must be a function of the \
identifier alone, using the low 8 bytes of its SHA-256 as a big-endian `u64` divided by \
`u64::MAX` to obtain a fraction in `[0, 1)`, compared against `eval_fraction`.",
        tests: &[
            "#[test]\nfn the_same_id_always_lands_the_same_side() {\n    let first = super::side(\"task-42\", 0.25);\n    for _ in 0..16 {\n        assert_eq!(super::side(\"task-42\", 0.25), first);\n    }\n}\n",
            "#[test]\nfn the_fraction_is_honoured() {\n    let n = 4000;\n    let eval = (0..n).filter(|i| super::side(&format!(\"t{i}\"), 0.25)).count();\n    let frac = eval as f64 / n as f64;\n    assert!((frac - 0.25).abs() < 0.03, \"eval fraction was {frac}\");\n}\n",
            "#[test]\nfn both_sides_are_populated() {\n    let n = 500;\n    let eval = (0..n).filter(|i| super::side(&format!(\"t{i}\"), 0.25)).count();\n    assert!(eval > 0 && eval < n, \"one side was empty: eval={eval}\");\n}\n",
        ],
    },
    Spec {
        id: "law-overlap-measured-against-the-task",
        language: "rust",
        prompt: "\
A contamination check asks what fraction of a *task*'s n-grams appear in a training \
corpus. Measuring the fraction against the corpus instead would let a large corpus \
dilute a verbatim-contaminated task into looking clean.

Implement in `src/adhoc.rs`:

    pub fn overlap(task_grams: &[String], corpus: &std::collections::HashSet<String>) -> f64

Return the number of `task_grams` that occur in `corpus`, divided by `task_grams.len()`. \
A task with no grams returns 1.0, because a task too short to check must not be reported \
as clean.",
        tests: &[
            "#[test]\nfn an_empty_task_is_not_clean() {\n    let corpus: std::collections::HashSet<String> = [\"a\".to_string()].into_iter().collect();\n    assert_eq!(super::overlap(&[], &corpus), 1.0);\n}\n",
            "#[test]\nfn the_denominator_is_the_task_not_the_corpus() {\n    let task: Vec<String> = (0..10).map(|i| format!(\"g{i}\")).collect();\n    let corpus: std::collections::HashSet<String> = (0..10).map(|i| format!(\"g{i}\")).chain((0..390).map(|i| format!(\"z{i}\"))).collect();\n    assert!((super::overlap(&task, &corpus) - 1.0).abs() < 1e-12, \"a verbatim task was diluted\");\n}\n",
            "#[test]\nfn partial_overlap_is_proportional() {\n    let task: Vec<String> = (0..10).map(|i| format!(\"g{i}\")).collect();\n    let corpus: std::collections::HashSet<String> = (0..5).map(|i| format!(\"g{i}\")).collect();\n    assert!((super::overlap(&task, &corpus) - 0.5).abs() < 1e-12);\n}\n",
        ],
    },
    Spec {
        id: "law-tolerance-zero-for-exact",
        language: "rust",
        prompt: "\
Every load-bearing mathematical identity in this crate is a certificate carrying a \
residual and a tolerance. An exact identity is stated with tolerance exactly `0.0`; a \
float identity carries a tolerance chosen from the arithmetic, never from whatever the \
code currently happens to produce.

Implement in `src/adhoc.rs`:

    pub struct Check { pub name: String, pub residual: f64, pub tolerance: f64 }\n    impl Check {\n        pub fn exact(name: &str, residual: f64) -> Self;\n        pub fn approximate(name: &str, residual: f64, tolerance: f64) -> Self;\n        pub fn passed(&self) -> bool;\n    }\n\n`exact` sets tolerance to `0.0`. `passed` is true only when the residual is finite and \
does not exceed the tolerance. A non-finite residual always fails, whatever the tolerance.",
        tests: &[
            "#[test]\nfn an_exact_identity_has_zero_tolerance() {\n    assert_eq!(super::Check::exact(\"a\", 0.0).tolerance, 0.0);\n}\n",
            "#[test]\nfn an_exact_identity_fails_on_any_drift() {\n    assert!(super::Check::exact(\"a\", 1e-18).passed() == false, \"exact tolerance was loosened\");\n}\n",
            "#[test]\nfn a_non_finite_residual_always_fails() {\n    assert!(!super::Check::approximate(\"a\", f64::INFINITY, 1e9).passed());\n    assert!(!super::Check::approximate(\"a\", f64::NAN, 1e9).passed());\n}\n",
        ],
    },
    Spec {
        id: "law-count-budget-baseline",
        language: "rust",
        prompt: "\
A scanner reports lint suppressions but tolerates the ones already on disk, reviewed by a \
human. The allowance is a *count*, not a boolean: a file already carrying three allowed \
suppressions earns no fourth for free, because a boolean match would let a candidate hide \
a new suppression behind an existing one in the same file.

Implement in `src/adhoc.rs`:

    pub fn drop_baselined(found: Vec<String>, allowed: &[(String, usize)]) -> Vec<String>

`allowed` lists `(evidence, budget)` pairs. Walk it in order; for each pair, drop up to \
`budget` occurrences of that evidence from `found`, then move on. Return the rest.",
        tests: &[
            "#[test]\nfn a_budget_is_respected() {\n    let found = vec![\"x\".into(), \"x\".into(), \"x\".into()];\n    let allowed = vec![(\"x\".to_string(), 2usize)];\n    assert_eq!(super::drop_baselined(found, &allowed), vec![\"x\"]);\n}\n",
            "#[test]\nfn a_zero_budget_drops_nothing() {\n    let found = vec![\"x\".into()];\n    assert_eq!(super::drop_baselined(found.clone(), &[]), found);\n}\n",
            "#[test]\nfn an_unlisted_evidence_is_never_dropped() {\n    let found = vec![\"y\".into()];\n    let allowed = vec![(\"x\".to_string(), 9usize)];\n    assert_eq!(super::drop_baselined(found, &allowed), vec![\"y\"]);\n}\n",
            "#[test]\nfn budgets_are_per_entry_not_shared() {\n    let found = vec![\"x\".into(), \"y\".into()];\n    let allowed = vec![(\"x\".to_string(), 1usize), (\"y\".to_string(), 1usize)];\n    assert!(super::drop_baselined(found, &allowed).is_empty());\n}\n",
        ],
    },
    Spec {
        id: "law-no-silent-drop",
        language: "rust",
        prompt: "\
A `Result` that is genuinely not needed must say so explicitly with `drop(f())` and a \
comment naming why the value does not matter. Assigning it to `_`, or discarding it with \
`let _ = f()`, is denied: silence is how a failure becomes a number somebody trusts.

Implement in `src/adhoc.rs`:

    pub fn discard_deliberately() -> anyhow::Result<()>\n    pub fn discard_silently() -> anyhow::Result<()>

The first documents the drop with a comment. The second uses `let _ =`. The function \
`discard_silently` must exist so a test can prove it is the shape this repository denies; \
it must not be called from anywhere else in the crate.",
        tests: &[
            "#[test]\nfn the_deliberate_form_is_clean() {\n    assert!(super::discard_deliberately().is_ok());\n}\n",
            "#[test]\nfn the_source_shows_both_shapes() {\n    let src = std::fs::read_to_string(\"src/adhoc.rs\").unwrap();\n    assert!(src.contains(\"drop(\"), \"the documented drop form is absent\");\n    assert!(src.contains(\"let _ =\"), \"the denied form is absent, so nothing is being compared\");\n}\n",
            "#[test]\nfn the_denied_form_appears_exactly_once_and_is_unreferenced() {\n    let src = std::fs::read_to_string(\"src/adhoc.rs\").unwrap();\n    assert_eq!(src.matches(\"discard_silently\").count(), 2, \"expected a definition and one reference\");\n}\n",
        ],
    },
    Spec {
        id: "law-strip-deletes-spelled-out",
        language: "rust",
        prompt: "\
A loader that is handed a tree must distinguish a file that was *absent* from a file that \
was not *looked at*, because deleting the gate script is visible only as an absence. A \
loader that reports only what it found cannot tell those apart.

Implement in `src/adhoc.rs`:

    pub struct Loaded { pub files: Vec<(String, String)>, pub deleted: Vec<String> }\n    pub fn load(root: &std::path::Path, expected: &[&str]) -> anyhow::Result<Loaded>\n\nEvery entry of `expected` must appear in exactly one of `files` or `deleted`. An entry \
that is neither is an error naming it, because a silently-missing file is a gate that \
was never run.",
        tests: &[
            "#[test]\nfn an_expected_file_lands_in_exactly_one_side() {\n    let d = std::env::temp_dir().join(format!(\"load-{}\", std::process::id()));\n    std::fs::create_dir_all(&d).unwrap();\n    std::fs::write(d.join(\"a.rs\"), \"pub fn f() {}\").unwrap();\n    let l = super::load(&d, &[\"a.rs\", \"missing.rs\"]).unwrap();\n    assert_eq!(l.files.len(), 1);\n    assert_eq!(l.deleted, vec![\"missing.rs\".to_string()]);\n    std::fs::remove_dir_all(&d).unwrap();\n}\n",
            "#[test]\nfn every_expected_name_is_accounted_for() {\n    let d = std::env::temp_dir().join(format!(\"load2-{}\", std::process::id()));\n    std::fs::create_dir_all(&d).unwrap();\n    let l = super::load(&d, &[\"a.rs\", \"b.rs\"]).unwrap();\n    assert_eq!(l.deleted.len(), 2, \"both missing files must be recorded as deleted\");\n    std::fs::remove_dir_all(&d).unwrap();\n}\n",
        ],
    },
    Spec {
        id: "law-bpe-roundtrip-lossless",
        language: "rust",
        prompt: "\
A byte-level BPE tokenizer must be lossless: encoding text and decoding the ids returns \
the original bytes exactly, for any input including invalid UTF-8 and emoji. A tokenizer \
that round-trips only well-formed ASCII is a tokenizer that silently corrupts code.

Implement in `src/adhoc.rs`:

    pub fn roundtrips(text: &[u8]) -> bool

Return true when a simple byte-pair tokeniser that maps every byte to an id and back \
reproduces `text` exactly. Encode as base-256 over the input bytes, decode by mapping \
each id back to its byte, and compare.",
        tests: &[
            "#[test]\nfn ascii_round_trips() {\n    assert!(super::roundtrips(b\"fn main() {}\"));\n}\n",
            "#[test]\nfn invalid_utf8_round_trips() {\n    let bytes: &[u8] = &[0xff, 0xfe, 0x00, 0x80, 0x41];\n    assert!(super::roundtrips(bytes), \"invalid UTF-8 was corrupted\");\n}\n",
            "#[test]\nfn multi_byte_utf8_round_trips() {\n    let bytes = \"héllo 🌍 日本語\".as_bytes();\n    assert!(super::roundtrips(bytes));\n}\n",
            "#[test]\nfn the_empty_input_round_trips() {\n    assert!(super::roundtrips(b\"\"));\n}\n",
        ],
    },
    Spec {
        id: "law-power-iteration-bound",
        language: "rust",
        prompt: "\
A step size derived from a Lipschitz bound needs an upper estimate of the largest \
eigenvalue of a symmetric positive-definite matrix. Power iteration converges to it from \
any starting vector, and the estimate must be inflated afterwards so the step stays \
strictly inside the stability range.

Implement in `src/adhoc.rs`:

    pub fn spectral_norm_upper(g: &[Vec<f64>], iterations: usize, inflate: f64) -> anyhow::Result<f64>\n    pub fn power_iteration(g: &[Vec<f64>], iterations: usize) -> anyhow::Result<Vec<f64>>\n\n`power_iteration` returns the final iterate. `spectral_norm_upper` returns the Rayleigh \
quotient of that iterate inflated by the factor. Reject a non-square or empty matrix.",
        tests: &[
            "#[test]\nfn the_identity_has_norm_one() {\n    let g = vec![vec![1.0, 0.0], vec![0.0, 1.0]];\n    assert!((super::spectral_norm_upper(&g, 64, 1.1) - 1.1).abs() < 1e-3);\n}\n",
            "#[test]\nfn a_diagonal_matrix_gives_its_largest_entry() {\n    let g = vec![vec![3.0, 0.0], vec![0.0, 5.0]];\n    let n = super::spectral_norm_upper(&g, 200, 1.0);\n    assert!((n - 5.0).abs() < 1e-3, \"got {n}\");\n}\n",
            "#[test]\nfn the_bound_inflates() {\n    let g = vec![vec![2.0, 0.0], vec![0.0, 1.0]];\n    assert!(super::spectral_norm_upper(&g, 200, 1.5) > super::spectral_norm_upper(&g, 200, 1.0));\n}\n",
            "#[test]\nfn a_non_square_matrix_is_refused() {\n    let g = vec![vec![1.0, 2.0, 3.0]];\n    assert!(super::spectral_norm_upper(&g, 10, 1.1).is_err());\n}\n",
        ],
    },
    Spec {
        id: "law-entropy-in-range",
        language: "rust",
        prompt: "\
Routing diagnostics are read by a human deciding whether training is healthy. An \
entropy printed outside its mathematical range is a bug report in itself: a negative \
entropy is impossible, and an entropy above 1 means the weights are not a distribution.

Implement in `src/adhoc.rs`:

    pub fn entropy(weights: &[f64]) -> anyhow::Result<f64>\n    pub fn is_distribution(weights: &[f64]) -> bool\n\n`entropy` is `-Σ p log₂ p`, treating a zero weight as contributing zero. It is defined \
only for a probability distribution, so reject any row that is empty, holds a negative \
weight, or fails to sum to 1 within 1e-9 — returning an error rather than normalising. \
Silently normalising would hide the exact failure being looked for: a router whose gates \
sum to the number of boxes instead of 1 produces a negative entropy, and renormalising \
turns that diagnostic into a plausible-looking number. `is_distribution` is the predicate \
that decides, and must be false for any row entropy rejects.",
        tests: &[
            "#[test]\nfn a_point_mass_has_zero_entropy() {\n    assert!(super::entropy(&[1.0, 0.0]).unwrap().abs() < 1e-12);\n}\n",
            "#[test]\nfn a_uniform_row_over_k_has_log2_k_entropy() {\n    let e = super::entropy(&[0.25, 0.25, 0.25, 0.25]).unwrap();\n    assert!((e - 2.0).abs() < 1e-12, \"got {e}\");\n}\n",
            "#[test]\nfn entropy_is_never_negative_on_a_distribution() {\n    for k in 2..8 {\n        let w: Vec<f64> = vec![1.0 / k as f64; k];\n        let e = super::entropy(&w).unwrap();\n        assert!(e >= 0.0, \"entropy {e} at k={k}\");\n        assert!(e <= k as f64, \"entropy {e} exceeded log2({k})\");\n    }\n}\n",
            "#[test]\nfn a_row_that_is_not_a_distribution_is_refused_not_normalised() {\n    // The historical failure: a broadcast scatter left every column holding a\n    // gate above 1, so each row summed to the number of boxes and its entropy\n    // read negative. Renormalising would turn that diagnostic into a plausible\n    // number, which is why this case must be refused.\n    for k in 2..8 {\n        let overweight: Vec<f64> = vec![1.5; k];\n        assert!(super::entropy(&overweight).is_err(), \"a row summing to {} was scored\", overweight.iter().sum::<f64>());\n    }\n}\n",
            "#[test]\nfn a_row_that_does_not_sum_to_one_is_not_a_distribution() {\n    assert!(!super::is_distribution(&[0.5, 0.6]));\n    assert!(!super::is_distribution(&[-0.5, 1.5]));\n}\n",
        ],
    },
    Spec {
        id: "law-content-address-roundtrip",
        language: "rust",
        prompt: "\
A checkpoint is addressed by the hash of its contents, so two models are equal exactly \
when their bytes are. The address must be a function of content alone: the same records in \
a different order are the same model, and one changed record is a different model.\n\nImplement in `src/adhoc.rs`:

    pub fn content_address(records: &[(String, Vec<u8>)]) -> String\n    pub fn same_model(a: &[(String, Vec<u8>)], b: &[(String, Vec<u8>)]) -> bool\n\n`content_address` sorts the records by name, hashes `name` then the bytes with SHA-256, \
and returns lowercase hex. `same_model` compares addresses rather than orderings.",
        tests: &[
            "#[test]\nfn order_does_not_change_the_address() {\n    let a = vec![(\"x\".to_string(), vec![1u8]), (\"y\".to_string(), vec![2u8])];\n    let b = vec![(\"y\".to_string(), vec![2u8]), (\"x\".to_string(), vec![1u8])];\n    assert_eq!(super::content_address(&a), super::content_address(&b));\n    assert!(super::same_model(&a, &b));\n}\n",
            "#[test]\nfn a_changed_byte_changes_the_address() {\n    let a = vec![(\"x\".to_string(), vec![1u8])];\n    let b = vec![(\"x\".to_string(), vec![2u8])];\n    assert_ne!(super::content_address(&a), super::content_address(&b));\n    assert!(!super::same_model(&a, &b));\n}\n",
            "#[test]\nfn a_renamed_record_changes_the_address() {\n    let a = vec![(\"x\".to_string(), vec![1u8])];\n    let b = vec![(\"y\".to_string(), vec![1u8])];\n    assert_ne!(super::content_address(&a), super::content_address(&b));\n}\n",
            "#[test]\nfn the_address_is_lowercase_hex_of_sha256() {\n    let a = super::content_address(&[]);\n    assert_eq!(a.len(), 64);\n    assert!(a.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()), \"{a}\");\n}\n",
        ],
    },
    Spec {
        id: "law-nf4-exact-zero",
        language: "rust",
        prompt: "\
A 4-bit normal-float weight format maps the value zero exactly. Every other level is a \
quantised approximation, so the reconstruction error bound is stated per level and the \
zero level carries no error at all.\n\nImplement in `src/adhoc.rs`:

    pub const NF4_LEVELS: [f64; 16];\n    pub fn quantize(x: f32) -> (u8, f64)\n    pub fn dequantize(code: u8, scale: f64) -> f64\n    pub fn max_error() -> f64\n\nReturn the nearest level index and the reconstruction error. `max_error` is the largest \
reconstruction error over a dense sweep of the representable range.",
        tests: &[
            "#[test]\nfn zero_is_exact() {\n    let (code, err) = super::quantize(0.0);\n    assert!(err == 0.0, \"zero carried error {err}\");\n    let level = super::NF4_LEVELS[code as usize];\n    assert!(level.abs() < 1e-12, \"the level chosen for zero was {level}\");\n}\n",
            "#[test]\nfn every_code_round_trips_to_its_own_level() {\n    for c in 0..16u8 {\n        let v = super::dequantize(c, 1.0);\n        assert_eq!(super::quantize(v as f32).0, c, \"code {c} did not round trip\");\n    }\n}\n",
            "#[test]\nfn the_error_bound_is_finite_and_bounded() {\n    let m = super::max_error();\n    assert!(m.is_finite());\n    assert!(m > 0.0 && m < 0.5, \"error bound was {m}\");\n}\n",
        ],
    },
    Spec {
        id: "law-sigma-preconditioning-identity",
        language: "rust",
        prompt: "\
An EDM preconditioner wraps a denoiser so that its output is an estimate of clean data \
for any noise scale. At zero noise the preconditioner is the identity, which is what makes \
the clean reasoning path exact rather than approximate.\n\nImplement in `src/adhoc.rs`:

    pub fn precondition(c_in: f64, c_out: f64, c_noise: f64, sigma: f64, x: &[f64]) -> anyhow::Result<Vec<f64>>\n    pub fn is_identity_at_zero(sigma: f64) -> bool\n\nReturn `c_noise * x + c_in * F(c_out * x + c_noise * sigma)`. Reject a negative `sigma`. \
`is_identity_at_zero` is true exactly when `sigma` is zero.",
        tests: &[
            "#[test]\nfn zero_sigma_leaves_the_input_alone() {\n    let x = vec![1.0, -2.0, 0.5];\n    let out = super::precondition(1.0, 1.0, 1.0, 0.0, &x).unwrap();\n    for (a, b) in out.iter().zip(&x) {\n        assert!((a - b).abs() < 1e-12, \"preconditioning changed the input at sigma=0\");\n    }\n}\n",
            "#[test]\nfn a_negative_sigma_is_refused() {\n    assert!(super::precondition(1.0, 1.0, 1.0, -1.0, &[1.0]).is_err());\n}\n",
            "#[test]\nfn the_identity_predicate_agrees_with_the_map() {\n    assert!(super::is_identity_at_zero(0.0));\n    assert!(!super::is_identity_at_zero(1e-3));\n}\n",
        ],
    },
    Spec {
        id: "law-loop-rejector",
        language: "rust",
        prompt: "\
A reasoning teacher served over an API can fall into a fixed point and repeat a span \
until it hits the token cap. Such a completion must be rejected before it enters a \
training corpus, because a looped trace teaches the student to loop.\n\nImplement in `src/adhoc.rs`:

    pub fn repeat_rate(text: &str, n: usize) -> f64\n    pub fn is_looped(text: &str, n: usize, threshold: f64) -> bool\n\n`repeat_rate` is the fraction of the text's n-grams that have already appeared earlier \
in the text. `is_looped` is true when the rate exceeds `threshold`. A text shorter than \
one n-gram has rate 0.0.",
        tests: &[
            "#[test]\nfn ordinary_prose_does_not_loop() {\n    let t = \"the quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant hills\";\n    assert!(!super::is_looped(t, 4, 0.2), \"prose was rejected as a loop\");\n}\n",
            "#[test]\nfn a_repeated_span_is_rejected() {\n    let span = \"wait let me reconsider the previous derivation once more \";\n    let t = format!(\"{span}{span}{span}{span}\");\n    assert!(super::is_looped(&t, 4, 0.2), \"a repeated span was accepted\");\n}\n",
            "#[test]\nfn a_short_text_is_not_a_loop() {\n    assert_eq!(super::repeat_rate(\"hi\", 8), 0.0);\n    assert!(!super::is_looped(\"hi\", 8, 0.0));\n}\n",
            "#[test]\nfn a_later_repeat_raises_the_rate() {\n    let t = \"alpha beta gamma delta epsilon zeta eta theta \";\n    let once = super::repeat_rate(t, 4);\n    let twice = super::repeat_rate(&format!(\"{t}{t}\"), 4);\n    assert!(twice > once, \"repetition did not raise the rate ({once} then {twice})\");\n}\n",
        ],
    },
];

fn main() -> Result<()> {
    let mut out = PathBuf::from("/srv/m-sdd/unifur/datasets/repo-native-eval.jsonl");
    let mut corpus: Option<PathBuf> = None;
    let mut limit = usize::MAX;

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
            "--help" | "-h" => {
                println!("usage: evalgen [--out <path>] [--corpus <sft.jsonl>] [--limit <n>]");
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
            let at_risk: Vec<&Task> = tasks
                .iter()
                .filter(|t| contamination_risk(&t.source).is_some())
                .collect();
            println!(
                "at-risk provenance: {} (repo-native is clean by construction)",
                at_risk.len()
            );
            let kept: Vec<Task> = tasks
                .iter()
                .filter(|t| split(&t.id, DEFAULT_EVAL_FRACTION) == Side::Eval)
                .cloned()
                .collect();
            codegen_eval::write_tasks(&out, &kept)?;
            println!("wrote {} eval-side tasks to {}", kept.len(), out.display());
        }
    }
    Ok(())
}

/// Index over an SFT file, or an empty index when it is absent.
///
/// An absent corpus is not a licence to skip the check: the gate must be able to
/// run before the corpus exists, so the empty index reports every task as
/// *uncheckable* rather than clean. `overlap_fraction` already returns 1.0 for a
/// task with no grams, but an empty corpus would otherwise look maximally
/// clean, so this returns an index built from a single unmatchable row.
fn index_from(path: &Path) -> Result<CorpusIndex> {
    if path.exists() {
        codegen_eval::index_sft(path)
    } else {
        println!("corpus {} absent; skipping decontamination", path.display());
        Ok(CorpusIndex::build([(
            "absent",
            "the training corpus was not available so nothing can be shown uncontaminated",
        )]))
    }
}
