// The task catalogue for `evalgen`. Kept in its own file so the generator's
// argument handling and the task text do not compete for the same screenful.
//
// Each entry states one rule this repository enforces, names the API to
// implement, and carries tests written to reject the plausible shortcut. A test
// a lazy solution also passes measures nothing, so every case here was checked
// in both directions: the correct implementation passes, and the naive one
// fails.

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
        id: "law-tokenise-before-matching",
        language: "rust",
        // A raw string, because the prompt has to *contain* quotes and a
        // raw-string delimiter: describing them with escapes inside a normal
        // string makes the example itself unreadable.
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
comment naming why the value does not matter. Assigning it to `_` is denied: silence is \
how a failure becomes a number somebody trusts.

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
zero level carries no error at all.

Implement in `src/adhoc.rs`:

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
the clean reasoning path exact rather than approximate.

Implement in `src/adhoc.rs`:

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
training corpus, because a looped trace teaches the student to loop.

Implement in `src/adhoc.rs`:

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
    Spec {
        id: "law-fallible-constructor",
        language: "rust",
        prompt: "\
A constructor that can be handed a bad configuration returns an `anyhow::Result` and \
names the field at fault. Silently clamping an invalid value would move the bug to the \
first caller that depended on it.

Implement in `src/adhoc.rs`:

    #[derive(Clone, Copy, Debug)]\n    pub struct LayerCounts { pub blocks: usize, pub refine_steps: usize }\n\n    impl LayerCounts {\n        pub fn new(blocks: usize, refine_steps: usize) -> anyhow::Result<Self>;\n        pub fn total_layers(&self) -> usize;\n    }\n\nBoth counts must be at least 1. Return an error naming the offending field and its value.",
        tests: &[
            "#[test]\nfn a_valid_configuration_is_accepted() {\n    let c = super::LayerCounts::new(4, 3).unwrap();\n    assert_eq!(c.total_layers(), 12);\n}\n",
            "#[test]\nfn zero_blocks_is_refused_and_named() {\n    let err = super::LayerCounts::new(0, 3).unwrap_err().to_string();\n    assert!(err.contains(\"blocks\"), \"error must name the field: {err}\");\n    assert!(err.contains('0'), \"error must name the value: {err}\");\n}\n",
            "#[test]\nfn zero_refine_steps_is_refused_and_named() {\n    let err = super::LayerCounts::new(4, 0).unwrap_err().to_string();\n    assert!(err.contains(\"refine_steps\"), \"error must name the field: {err}\");\n    assert!(err.contains('0'), \"error must name the value: {err}\");\n}\n",
            "#[test]\nfn a_silently_clamped_configuration_is_rejected() {\n    // Clamping 0 to 1 would make this return Ok, which is the shortcut the\n    // repository's fallible-constructor rule exists to prevent.\n    assert!(super::LayerCounts::new(0, 1).is_err());\n}\n",
        ],
    },
    Spec {
        id: "law-serde-default-field",
        language: "rust",
        prompt: "\
A checkpoint sidecar written by an older build must still parse after a new field is \
added. Every `serde` field on a config therefore carries `#[serde(default)]`, so an \
absent field takes its default rather than failing the whole load.

Implement in `src/adhoc.rs`:

    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]\n    pub struct Sidecar {\n        pub model_path: String,\n        #[serde(default = \"default_eval_fraction\")]\n        pub eval_fraction: f64,\n        #[serde(default = \"default_gate\")]\n        pub gate: bool,\n    }\n\nwith `fn default_eval_fraction() -> f64 { 0.25 }` and `fn default_gate() -> bool { true }`.\nA JSON document naming only `model_path` must deserialize successfully, taking both \
defaults.",
        tests: &[
            "#[test]\nfn an_older_sidecar_still_parses() {\n    let v = super::Sidecar {\n        model_path: \"/models/qwen\".into(),\n        eval_fraction: 0.25,\n        gate: true,\n    };\n    let json = serde_json::to_string(&v).unwrap();\n    let back: super::Sidecar = serde_json::from_str(&json).unwrap();\n    assert_eq!(back, v);\n}\n",
            "#[test]\nfn an_absent_field_takes_its_default() {\n    let s: super::Sidecar = serde_json::from_str(r#\"{\"model_path\":\"/m\"}\"#).unwrap();\n    assert_eq!(s.model_path, \"/m\");\n    assert_eq!(s.eval_fraction, 0.25, \"eval_fraction took the wrong default\");\n    assert!(s.gate, \"gate took the wrong default\");\n}\n",
            "#[test]\nfn the_round_trip_preserves_non_default_values() {\n    let v = super::Sidecar { model_path: \"/m\".into(), eval_fraction: 0.5, gate: false };\n    let json = serde_json::to_string(&v).unwrap();\n    let back: super::Sidecar = serde_json::from_str(&json).unwrap();\n    assert_eq!(back, v, \"a default attribute overwrote a stored value\");\n}\n",
        ],
    },
    Spec {
        id: "law-monotone-energy-descent",
        language: "rust",
        prompt: "\
A certified relaxation descends an explicit energy with a step size from a closed-form \
Lipschitz bound, so the energy must never increase. The step has to be the gradient of \
the energy it reports: applying the repulsion term with coefficient `λ` where the \
energy's is `2λ` makes the step descend a *different* quadratic, and the monotonicity \
then holds or fails depending on the draw.\n\nImplement in `src/adhoc.rs`:

    pub fn energy(z: &[f64], c: &[f64], lambda: f64) -> f64\n    pub fn grad_step(z: &[f64], c: &[f64], lambda: f64, eta: f64) -> Vec<f64>\n\n`energy` is `Σ_k (z_k − c_k)² + λ Σ_{k≠l} (z_k − z_l)²` over the ordered pairs, so each \
unordered pair is counted twice. `grad_step` takes one gradient step of that same energy \
with step size `eta`.",
        tests: &[
            "#[test]\nfn the_step_lowers_the_energy_it_reports() {\n    for k in [2usize, 3, 5] {\n        let z: Vec<f64> = (0..k).map(|i| i as f64 * 0.3).collect();\n        let c: Vec<f64> = (0..k).map(|i| (i as f64 * 0.7).sin()).collect();\n        for lambda in [0.0, 0.5, 2.0] {\n            let before = super::energy(&z, &c, lambda);\n            let next = super::grad_step(&z, &c, lambda, 0.01);\n            let after = super::energy(&next, &c, lambda);\n            assert!(after <= before + 1e-12, \"energy rose {before} -> {after} at k={k} lambda={lambda}\");\n        }\n    }\n}\n",
            "#[test]\nfn the_step_is_the_gradient_of_the_reported_energy() {\n    // Central differences against the direction of the step itself. Monotonicity\n    // alone cannot see a wrong repulsion coefficient: a wide range of\n    // coefficients keeps the energy decreasing at a small step size, which is\n    // exactly why the original bug passed the monotone test and went red only by\n    // the draw. This case pins the coefficient, so it fails the moment the two\n    // drift apart.\n    let z = vec![0.3, 0.6, 0.9, 1.2];\n    let c = vec![0.1, 0.8, 0.2, 0.5];\n    let eta = 0.01;\n    for lambda in [0.5, 1.5, 3.0] {\n        let next = super::grad_step(&z, &c, lambda, eta);\n        let step: Vec<f64> = next.iter().zip(&z).map(|(a, b)| a - b).collect();\n        let h = 1e-6;\n        for (m, (s, zk)) in step.iter().zip(&z).enumerate() {\n            let mut plus = z.clone();\n            let mut minus = z.clone();\n            plus[m] = zk + h;\n            minus[m] = zk - h;\n            let numeric = (super::energy(&plus, &c, lambda) - super::energy(&minus, &c, lambda)) / (2.0 * h);\n            // The step is -eta * gradient, so the gradient is -step / eta.\n            let analytic = -s / eta;\n            let rel = (analytic - numeric).abs() / numeric.abs().max(1e-9);\n            assert!(rel < 1e-3, \"coordinate {m}: analytic {analytic} vs numeric {numeric}, relative error {rel}\");\n        }\n    }\n}\n",
            "#[test]\nfn zero_repulsion_lands_on_the_context() {\n    let z = vec![1.0, -2.0, 0.5];\n    let c = vec![0.25, 0.0, 0.75];\n    let mut cur = z.clone();\n    for _ in 0..4000 {\n        cur = super::grad_step(&cur, &c, 0.0, 0.05);\n    }\n    for (a, b) in cur.iter().zip(&c) {\n        assert!((a - b).abs() < 1e-2, \"slot landed at {a}, context is {b}\");\n    }\n}\n",
        ],
    },
    Spec {
        id: "law-exact-rational-arithmetic",
        language: "rust",
        prompt: "\
The geometric kernel stands beside the model as proof, and it is exact: rational \
arithmetic, because `1/3 + 1/6` is exactly `1/2` and a float kernel gets that wrong.

Implement in `src/adhoc.rs`:

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]\n    pub struct Rat { pub num: i64, pub den: i64 }\n\n    impl Rat {\n        pub fn new(num: i64, den: i64) -> Option<Self>;\n        pub fn add(&self, other: &Self) -> Option<Self>;\n        pub fn mul(&self, other: &Self) -> Option<Self>;\n        pub fn to_f64(&self) -> f64;\n    }\n\nAlways keep the fraction in lowest terms with a positive denominator. Return `None` for \
a zero denominator or on i64 overflow rather than wrapping or panicking.",
        tests: &[
            "#[test]\nfn thirds_plus_sixths_is_exactly_one_half() {\n    let a = super::Rat::new(1, 3).unwrap();\n    let b = super::Rat::new(1, 6).unwrap();\n    let sum = a.add(&b).unwrap();\n    assert_eq!(sum, super::Rat::new(1, 2).unwrap(), \"got {sum:?}\");\n}\n",
            "#[test]\nfn fractions_are_reduced_and_sign_normalised() {\n    let a = super::Rat::new(2, 4).unwrap();\n    assert_eq!((a.num, a.den), (1, 2));\n    let b = super::Rat::new(1, -2).unwrap();\n    assert_eq!((b.num, b.den), (-1, 2), \"denominator must be positive\");\n}\n",
            "#[test]\nfn a_zero_denominator_is_refused_not_a_division_by_zero() {\n    assert!(super::Rat::new(1, 0).is_none());\n}\n",
            "#[test]\nfn overflow_is_refused_rather_than_wrapping() {\n    let big = super::Rat::new(i64::MAX, 1).unwrap();\n    let two = super::Rat::new(2, 1).unwrap();\n    assert!(big.mul(&two).is_none(), \"overflow wrapped instead of failing\");\n}\n",
            "#[test]\nfn a_float_kernel_would_get_this_wrong() {\n    // The reason the kernel is rational. Note which identity is used: 1/3 + 1/6\n    // happens to be exactly 0.5 in f64, so it proves nothing. 1/10 + 2/10 does\n    // not, which is why this case is stated in tenths rather than in thirds.\n    let exact = super::Rat::new(1, 10).unwrap().add(&super::Rat::new(2, 10).unwrap()).unwrap();\n    assert_eq!((exact.num, exact.den), (3, 10), \"rational arithmetic disagreed\");\n    let naive = 0.1f64 + 0.2f64;\n    assert!(naive != 0.3, \"the float path happened to be exact; the case proves nothing\");\n}\n",
        ],
    },
    Spec {
        id: "law-segment-intersection-exact",
        language: "rust",
        prompt: "\
Exact geometry needs an exact segment-intersection predicate. Signs of cross products, \
not slope comparisons: dividing by `dx` loses the answer when the segment is vertical, \
and a float comparison loses it when the crossing point is genuinely on the boundary.

Implement in `src/adhoc.rs`:

    pub fn orient(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> i32\n    pub fn segments_cross(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool\n\n`orient` is the sign of the cross product `(b−a) × (c−a)`. `segments_cross` reports \
whether the closed segments `ab` and `cd` share at least one point, including touching \
endpoints and collinear overlap.",
        tests: &[
            "#[test]\nfn a_proper_crossing_is_detected() {\n    assert!(super::segments_cross((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)));\n}\n",
            "#[test]\nfn parallel_segments_do_not_cross() {\n    assert!(!super::segments_cross((0.0, 0.0), (10.0, 0.0), (0.0, 1.0), (10.0, 1.0)));\n}\n",
            "#[test]\nfn a_vertical_segment_is_handled() {\n    // A slope-based implementation divides by zero here.\n    assert!(super::segments_cross((5.0, -5.0), (5.0, 5.0), (0.0, 0.0), (10.0, 1.0)));\n}\n",
            "#[test]\nfn touching_endpoints_count_as_crossing() {\n    assert!(super::segments_cross((0.0, 0.0), (1.0, 1.0), (1.0, 1.0), (2.0, 0.0)));\n}\n",
            "#[test]\nfn collinear_overlap_counts_as_crossing() {\n    assert!(super::segments_cross((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), (6.0, 0.0)));\n}\n",
        ],
    },
    Spec {
        id: "law-paged-state-is-exact",
        language: "rust",
        prompt: "\
Optimizer state is paged so only one decoder layer's adapter moments are resident. \
Round-tripping a page must be *exact*: the loss after an evict and restore has to equal \
the loss before it to the last bit, because a resume that drifts silently invalidates \
every measurement taken before the interruption.

Implement in `src/adhoc.rs`:

    pub fn partition<T: Clone>(items: &[(usize, T)], pages: usize) -> Vec<Vec<T>>\n    pub fn merge_restore<T>(pages: Vec<Vec<T>>) -> Vec<T>\n\n`partition` groups items by page index, preserving order within a page. `merge_restore` \
flattens the pages back into one vector in ascending page order.",
        tests: &[
            "#[test]\nfn a_round_trip_preserves_every_item() {\n    let items: Vec<(usize, u32)> = (0..50).map(|i| (i % 4, i as u32)).collect();\n    let original: Vec<u32> = items.iter().map(|(_, v)| *v).collect();\n    let pages = super::partition(&items, 4);\n    assert_eq!(pages.len(), 4);\n    let restored = super::merge_restore(pages);\n    let mut sorted = restored.clone();\n    sorted.sort_unstable();\n    let mut want = original.clone();\n    want.sort_unstable();\n    assert_eq!(sorted, want, \"a round trip lost or invented an item\");\n}\n",
            "#[test]\nfn an_empty_page_stays_empty() {\n    let items: Vec<(usize, u32)> = vec![(0, 1), (0, 2)];\n    let pages = super::partition(&items, 4);\n    assert!(pages[1].is_empty());\n    assert!(pages[3].is_empty());\n}\n",
            "#[test]\nfn page_order_survives_the_round_trip() {\n    let items: Vec<(usize, String)> = vec![(2, \"c\".into()), (0, \"a\".into()), (1, \"b\".into())];\n    let restored = super::merge_restore(super::partition(&items, 3));\n    assert_eq!(restored, vec![\"a\".to_string(), \"b\".to_string(), \"c\".to_string()]);\n}\n",
            "#[test]\nfn an_out_of_range_page_is_not_silently_folded_into_another_page() {\n    // Wrapping the page index with modulo looks correct for every in-range\n    // index and quietly corrupts state the moment one is out of range, so this\n    // case is the only thing separating the two.\n    let items: Vec<(usize, u32)> = vec![(0, 1), (7, 2)];\n    let pages = super::partition(&items, 3);\n    assert_eq!(pages[1].len(), 0, \"page 7 was folded into page 1\");\n}\n",
        ],
    },
    Spec {
        id: "law-nf4-scale-is-max-abs",
        language: "rust",
        prompt: "\
A block's quantization scale is its largest absolute weight, so the block factor is \
always at least 1 and the reconstruction error is bounded by the block's own dynamic \
range. Choosing a fixed scale instead wastes the block's precision.\n\nImplement in `src/adhoc.rs`:

    pub fn block_scale(abs_max: f64) -> anyhow::Result<f64>\n    pub fn quantize_block(values: &[f64]) -> anyhow::Result<(f64, Vec<i8>)>\n\n`block_scale` rejects a zero, negative or non-finite magnitude: a zero block has no \
scale to divide by, and returning it would hand the caller a division by zero one step \
later. `quantize_block` returns the scale and each value's signed code, where the code \
is the nearest of `[-8, 8]` after scaling, so a zero value always codes to 0 and the largest \
magnitude always codes to ±8.",
        tests: &[
            "#[test]\nfn zero_magnitude_is_refused() {\n    assert!(super::block_scale(0.0).is_err(), \"a zero block would divide by zero\");\n}\n",
            "#[test]\nfn a_non_finite_magnitude_is_refused() {\n    assert!(super::block_scale(f64::NAN).is_err());\n    assert!(super::block_scale(f64::INFINITY).is_err());\n}\n",
            "#[test]\nfn zero_codes_to_zero_and_the_max_to_eight() {\n    let v = vec![0.0, -3.0, 1.5, 3.0, -0.5];\n    let (scale, codes) = super::quantize_block(&v).unwrap();\n    assert!((scale - 3.0).abs() < 1e-12, \"scale was {scale}, want max-abs 3.0\");\n    assert_eq!(codes[0], 0, \"a zero value coded to {}\", codes[0]);\n    assert_eq!(codes[3], 8);\n    assert_eq!(codes[1], -8);\n}\n",
            "#[test]\nfn every_code_is_within_the_signed_range() {\n    let v: Vec<f64> = (0..64).map(|i| (i as f64 * 0.37).sin() * 2.0).collect();\n    let (_scale, codes) = super::quantize_block(&v).unwrap();\n    for c in &codes {\n        assert!((-8..=8).contains(c), \"code {c} left the range\");\n    }\n}\n",
        ],
    },
    Spec {
        id: "law-temperature-bounded-on-both-sides",
        language: "rust",
        prompt: "\
A jointly learned metric drifts silently: attention collapses toward uniform or blows \
up, with vanishing gradients instead of an error. The temperature is therefore clamped \
on both sides — a floor keeps `-d²/τ` off a zero divisor, a cap keeps runaway drift from \
flattening attention. Clamp rather than remap, so existing checkpoints stay bit-identical.

Implement in `src/adhoc.rs`:

    pub fn bounded_temperature(raw: f64, dim: usize) -> f64\n\nClamp into `[1e-3, 2·dim]`. A non-finite `raw` takes the floor, since an infinity in the \
exponent is the failure this exists to prevent.",
        tests: &[
            "#[test]\nfn the_floor_holds_for_a_negative_raw() {\n    assert!((super::bounded_temperature(-5.0, 8) - 1e-3).abs() < 1e-12);\n}\n",
            "#[test]\nfn the_cap_holds_for_a_runaway_raw() {\n    assert!((super::bounded_temperature(1e9, 8) - 16.0).abs() < 1e-12);\n}\n",
            "#[test]\nfn a_value_inside_the_range_is_untouched() {\n    // Clamp, not remap: an untouched interior is what keeps existing\n    // checkpoints bit-identical.\n    assert_eq!(super::bounded_temperature(3.5, 8), 3.5);\n}\n",
            "#[test]\nfn a_non_finite_raw_takes_the_floor() {\n    assert_eq!(super::bounded_temperature(f64::NAN, 8), 1e-3);\n    assert_eq!(super::bounded_temperature(f64::INFINITY, 8), 1e-3);\n}\n",
        ],
    },
    Spec {
        id: "law-decontamination-rejects-not-scores-clean",
        language: "rust",
        prompt: "\
A contamination check must refuse what it cannot verify. A task shorter than one n-gram \
has no n-grams to compare, and reporting it as clean would let an uncheckable task \
score as uncontaminated — a false negative that reads exactly like a pass.

Implement in `src/adhoc.rs`:

    pub fn overlap(task_grams: &[String], corpus: &std::collections::HashSet<String>) -> anyhow::Result<f64>\n    pub fn admit(overlap: f64, threshold: f64) -> bool\n\n`overlap` returns an error naming the task when `task_grams` is empty. `admit` is true only \
when the overlap is a number at or below the threshold. An overlap that is not a number — \
NaN, infinity — must not be admitted.",
        tests: &[
            "#[test]\nfn an_uncheckable_task_is_an_error_not_a_zero() {\n    let corpus: std::collections::HashSet<String> = [\"a\".to_string()].into_iter().collect();\n    assert!(super::overlap(&[], &corpus).is_err(), \"an empty task scored as clean\");\n}\n",
            "#[test]\nfn a_non_finite_overlap_is_never_admitted() {\n    assert!(!super::admit(f64::NAN, 0.1));\n    assert!(!super::admit(f64::INFINITY, 0.1));\n}\n",
            "#[test]\nfn the_threshold_is_inclusive_at_the_boundary() {\n    assert!(super::admit(0.1, 0.1));\n    assert!(!super::admit(0.1000001, 0.1));\n}\n",
        ],
    },
];