//! A coding eval that cannot be passed by having memorised the answers.
//!
//! # The problem this exists to solve
//!
//! Every public coding benchmark in this space — HumanEval, MBPP, APPS,
//! BigCodeBench — is inside the pretraining corpus of every frontier model, and
//! inside the fine-tuning corpus of most instruction mixes built on top of
//! them. A model that has seen a problem's reference solution will reproduce
//! it whether or not it can reason, so a score on such a set measures recall
//! rather than capability. Worse, the score *rises* as you train, which reads
//! as improvement while being the opposite.
//!
//! This module takes the position that the contamination check is the eval.
//! A task is only admitted to the scored set after it has been shown to be
//! absent from the training corpus, by n-gram overlap over normalised text.
//! A set that cannot be shown clean is reported as contaminated and scores
//! nothing, rather than being quietly scored.
//!
//! # Three independent leak paths, three independent checks
//!
//! | Path | Check |
//! |---|---|
//! | Training data contains the task | [`decontaminate`] — n-gram overlap against the corpus |
//! | Teacher has memorised the task | [`contamination_risk`] — flags the benchmarks known to be in pretraining data |
//! | Training on the graded assertions | [`split`] — a task's own tests never appear in the training half |
//!
//! The third is the one people forget. Holding out *problems* while training on
//! their *tests* leaks the answer as surely as holding out nothing.
//!
//! # What a passing score means
//!
//! Passing means: no cheat finding ([`crate::cheat`]), tests green, clippy
//! clean, audit clean, and the task verified uncontaminated against a named
//! corpus. All-or-nothing, for the same reason as the cheat gate — partial
//! credit on a gate is an incentive to buy the cheapest gate.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

/// Token count of the n-gram used for overlap detection.
///
/// 13 is the value GPT-3's contamination analysis settled on and is what
/// subsequent work defaults to. A longer n-gram misses paraphrases that share
/// only common phrasing; a shorter one flags every task that says "write a
/// function that".
pub const NGRAM: usize = 13;

/// Overlap above this fraction is contamination.
///
/// Generous, because a false positive costs a task and a false negative costs
/// the entire eval's meaning. The report always names the overlapping span so
/// a flagged task can be inspected rather than discarded blindly.
pub const OVERLAP_THRESHOLD: f64 = 0.10;

/// Benchmarks known to be inside frontier pretraining corpora.
///
/// This is a *prior*, not a measurement: admitting one of these without an
/// overlap check is the mistake this list exists to prevent. It cannot be
/// resolved by measurement either — a task can be clean against the local
/// corpus and still be in the teacher's weights — which is why
/// [`contamination_risk`] is advisory and [`decontaminate`] is the gate.
pub const KNOWN_CONTAMINATED: &[(&str, &str)] = &[
    (
        "humaneval",
        "OpenAI, 2021; canonical, in every frontier pretraining mix",
    ),
    (
        "mbpp",
        "Google, 2021; sanitized variants still overlap the originals",
    ),
    (
        "apps",
        "Hendrycks, 2021; competitive-programming, widely scraped",
    ),
    ("codecontests", "DeepMind, 2021; scraped from public judges"),
    (
        "bigcodebench",
        "ICLR 2024; released after most frontier cutoffs",
    ),
    (
        "ds-1000",
        "XCodeEval, 2023; data-science notebooks in pretraining",
    ),
    ("leetcode", "scraped from a public site since 2010"),
    ("codeforces", "scraped from a public site since 2010"),
    (
        "cruxeval",
        "cruxeval.io, 2022; Python transpiled to 32 languages",
    ),
];

/// Normalise text for n-gram comparison.
///
/// Case-folded, split on non-alphanumerics, so `Convolution(a, b)`, `convolution(a,b)`
/// and `convolution of a and b` share tokens. Contamination is about *code*,
/// and code's identity survives tokenisation even when formatting does not.
pub fn normalise_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// All n-grams of `tokens`, as a set.
///
/// A set, not a list: a task repeated verbatim in the corpus ten times is one
/// contaminated task, not ten.
pub fn ngrams(tokens: &[String], n: usize) -> HashSet<String> {
    if tokens.len() < n {
        return HashSet::new();
    }
    tokens.windows(n).map(|w| w.join("\u{1}")).collect()
}

/// What a decontamination check found.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Contamination {
    /// True when the task overlaps the corpus past [`OVERLAP_THRESHOLD`].
    pub is_contaminated: bool,
    /// Fraction of the task's n-grams that appear in the corpus.
    pub overlap: f64,
    /// One shared span, so a flag can be inspected instead of trusted.
    pub example: Option<String>,
    /// Which side of the comparison the corpus was.
    pub against: String,
}

impl Contamination {
    pub fn clean(against: impl Into<String>) -> Self {
        Self {
            is_contaminated: false,
            overlap: 0.0,
            example: None,
            against: against.into(),
        }
    }
}

/// Overlap of one task's n-grams with a prebuilt corpus index.
#[derive(Debug, Clone, Default)]
pub struct CorpusIndex {
    grams: HashSet<String>,
    grams_by_task: BTreeMap<String, usize>,
}

impl CorpusIndex {
    /// Build an index over labelled training samples.
    ///
    /// `grams_by_task` records how many distinct n-grams each sample
    /// contributes, which is what makes the overlap a *fraction of the task*
    /// rather than a fraction of the whole corpus — otherwise a large corpus
    /// makes every task look clean.
    pub fn build<'a, I>(samples: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut grams = HashSet::new();
        let mut grams_by_task = BTreeMap::new();
        for (label, text) in samples {
            let g = ngrams(&normalise_tokens(text), NGRAM);
            grams_by_task.insert(label.to_string(), g.len());
            grams.extend(g);
        }
        Self {
            grams,
            grams_by_task,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.grams.is_empty()
    }

    pub fn task_count(&self) -> usize {
        self.grams_by_task.len()
    }

    /// Distinct n-grams in the index.
    pub fn gram_count(&self) -> usize {
        self.grams.len()
    }

    /// Fraction of `text`'s n-grams that also occur in the corpus.
    pub fn overlap_fraction(&self, text: &str) -> f64 {
        let task = ngrams(&normalise_tokens(text), NGRAM);
        if task.is_empty() {
            // A task shorter than one n-gram cannot be checked. That is a
            // reason to reject it, not to score it clean.
            return 1.0;
        }
        let hits = task.iter().filter(|g| self.grams.contains(*g)).count();
        hits as f64 / task.len() as f64
    }

    /// One shared n-gram, for a human to judge the flag.
    pub fn example(&self, text: &str) -> Option<String> {
        ngrams(&normalise_tokens(text), NGRAM)
            .into_iter()
            .find(|g| self.grams.contains(g))
            .map(|g| g.replace('\u{1}', " "))
    }
}

/// Check one task against an index.
pub fn decontaminate(index: &CorpusIndex, task_text: &str, against: &str) -> Contamination {
    let overlap = index.overlap_fraction(task_text);
    Contamination {
        is_contaminated: overlap > OVERLAP_THRESHOLD,
        overlap,
        example: if overlap > OVERLAP_THRESHOLD {
            index.example(task_text)
        } else {
            None
        },
        against: against.to_string(),
    }
}

/// Named risk for a task whose provenance is a known-contaminated benchmark.
///
/// Advisory rather than blocking, because it cannot be discharged by
/// measurement: a task can be absent from the local corpus and still be in the
/// teacher's weights. The caller's job is to notice it, and the report keeps
/// it visible.
pub fn contamination_risk(source: &str) -> Option<&'static str> {
    let lowered = source.to_lowercase();
    KNOWN_CONTAMINATED
        .iter()
        .find(|(name, _)| lowered.contains(name))
        .map(|(_, why)| *why)
}

/// One coding task: a problem, the tests that decide it, and where it came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Stable id. The split is a deterministic function of this, so the same
    /// task lands on the same side on every machine and every rerun.
    pub id: String,
    /// The prompt handed to the model. Must not contain the tests.
    pub prompt: String,
    /// The file the model's code lands in, relative to the scratch tree.
    pub target: String,
    /// Test source, or command lines, that decide the task. Never shown to the
    /// model and never present in the training half.
    pub tests: Vec<String>,
    /// Where the task came from. A known-contaminated name is a red flag.
    pub source: String,
    /// Language, for reporting.
    pub language: String,
}

/// Default fraction of task ids assigned to the eval side.
///
/// A quarter: large enough that the pass rate's standard error stays under
/// about 0.02 at a few hundred tasks, small enough that most of the corpus
/// still trains.
pub const DEFAULT_EVAL_FRACTION: f64 = 0.25;

/// Which half of the corpus a task belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Train,
    Eval,
}

/// Deterministic train/eval assignment from the task id.
///
/// A hash, not a counter: a counter depends on file order, so inserting one
/// row silently reshuffles the split and invalidates every earlier score.
/// A hash depends only on the id, so the split is stable under any change to
/// the corpus that does not touch the task itself.
pub fn split(id: &str, eval_fraction: f64) -> Side {
    let digest = sha256(id.as_bytes());
    // First 8 bytes as a u64, mapped into [0, 1).
    let mut word = [0u8; 8];
    word.copy_from_slice(&digest[..8]);
    let frac = (u64::from_be_bytes(word) as f64) / (u64::MAX as f64);
    if frac < eval_fraction {
        Side::Eval
    } else {
        Side::Train
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// A task set that has been checked, and what the check found.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub total: usize,
    pub scored: usize,
    pub rejected: Vec<RejectedTask>,
    /// Tasks whose source is a known-contaminated benchmark. Counted as
    /// scored only if they also passed the overlap check.
    pub at_risk: Vec<AtRiskTask>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedTask {
    pub id: String,
    pub reason: String,
    pub overlap: f64,
    pub example: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtRiskTask {
    pub id: String,
    pub source: String,
    pub why: String,
}

impl EvalReport {
    /// Fraction of admitted tasks that were actually scorable.
    ///
    /// Reported because a set that admits 40 of 100 tasks and scores all 40 is
    /// a weaker claim than one that admits 90, and the difference is invisible
    /// if only the pass rate is printed.
    pub fn yield_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.scored as f64 / self.total as f64
    }
}

/// Admit tasks into the scored set, rejecting contaminated ones.
///
/// `index` is built over the *training* text only. Rejection is not silent:
/// every rejected task appears in the report with the overlap that condemned
/// it, so a low yield is diagnosable instead of mysterious.
pub fn admit(tasks: &[Task], index: &CorpusIndex) -> EvalReport {
    let mut report = EvalReport {
        total: tasks.len(),
        scored: 0,
        rejected: Vec::new(),
        at_risk: Vec::new(),
    };
    for task in tasks {
        // The prompt *and* the tests are both checked. A task whose tests
        // appear in the corpus leaks through the test text even when the
        // prompt is original.
        let mut text = task.prompt.clone();
        text.push('\n');
        for t in &task.tests {
            text.push_str(t);
            text.push('\n');
        }
        let c = decontaminate(index, &text, "training corpus n-gram index");
        if c.is_contaminated {
            report.rejected.push(RejectedTask {
                id: task.id.clone(),
                reason: "n-gram overlap with the training corpus exceeds the threshold".to_string(),
                overlap: c.overlap,
                example: c.example,
            });
            continue;
        }
        if let Some(why) = contamination_risk(&task.source) {
            report.at_risk.push(AtRiskTask {
                id: task.id.clone(),
                source: task.source.clone(),
                why: why.to_string(),
            });
        }
        report.scored += 1;
    }
    report
}

/// One task's outcome.
///
/// Mirrors the cheat gate's all-or-nothing shape: a task that passes its tests
/// but trips clippy, the audit, or the cheat scan is a **failure**, because the
/// pass was purchased rather than earned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutcome {
    pub id: String,
    pub tests_pass: bool,
    pub clippy_pass: bool,
    pub audit_pass: bool,
    pub cheat_clean: bool,
    pub score: f64,
}

impl TaskOutcome {
    pub fn passed(&self) -> bool {
        self.tests_pass && self.clippy_pass && self.audit_pass && self.cheat_clean
    }

    pub fn failures(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if !self.tests_pass {
            out.push("tests");
        }
        if !self.clippy_pass {
            out.push("clippy");
        }
        if !self.audit_pass {
            out.push("audit");
        }
        if !self.cheat_clean {
            out.push("cheat");
        }
        out
    }

    pub fn new(id: impl Into<String>, tests: bool, clippy: bool, audit: bool, cheat: bool) -> Self {
        let passed = tests && clippy && audit && cheat;
        Self {
            id: id.into(),
            tests_pass: tests,
            clippy_pass: clippy,
            audit_pass: audit,
            cheat_clean: cheat,
            score: if passed { 1.0 } else { 0.0 },
        }
    }
}

/// Aggregate outcomes. Mean of the binary scores, which is also the pass rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    pub scored: usize,
    pub passed: usize,
    pub pass_rate: f64,
    pub outcomes: Vec<TaskOutcome>,
}

impl Scorecard {
    pub fn new(outcomes: Vec<TaskOutcome>) -> Self {
        let scored = outcomes.len();
        let passed = outcomes.iter().filter(|o| o.passed()).count();
        let pass_rate = if scored == 0 {
            0.0
        } else {
            passed as f64 / scored as f64
        };
        Self {
            scored,
            passed,
            pass_rate,
            outcomes,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.scored == 0
    }

    /// Failures grouped by gate, for a training log that has to say *why*.
    pub fn failure_breakdown(&self) -> BTreeMap<&'static str, usize> {
        let mut out = BTreeMap::new();
        for o in &self.outcomes {
            for f in o.failures() {
                *out.entry(f).or_insert(0) += 1;
            }
        }
        out
    }
}

/// Read a JSONL task file.
pub fn read_tasks(path: &Path) -> Result<Vec<Task>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading task file {}", path.display()))?;
    let mut tasks = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let task: Task = serde_json::from_str(line)
            .with_context(|| format!("parsing task on line {}", n + 1))?;
        tasks.push(task);
    }
    if tasks.is_empty() {
        bail!("task file {} is empty", path.display());
    }
    Ok(tasks)
}

/// Write tasks as JSONL.
pub fn write_tasks(path: &Path, tasks: &[Task]) -> Result<()> {
    let mut out = String::new();
    for t in tasks {
        out.push_str(&serde_json::to_string(t).context("serialize task")?);
        out.push('\n');
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Build the corpus index from a JSONL SFT file of `{instruction, response}`.
///
/// Only `instruction` is indexed. Indexing the response would flag any task
/// whose *answer* appears in training — which is the right thing to catch, but
/// only if the response is part of what the model conditions on. For an SFT
/// set the response is the label, and a task sharing a label with a training
/// row is not thereby contaminated; sharing a *problem* is.
pub fn index_sft(path: &Path) -> Result<CorpusIndex> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut pairs: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let label = v
            .get("id")
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| format!("row{}", pairs.len()));
        let prompt = v
            .get("instruction")
            .and_then(|x| x.as_str())
            .unwrap_or_default();
        pairs.push((label, prompt.to_string()));
    }
    if pairs.is_empty() {
        bail!("no usable rows in {}", path.display());
    }
    let borrowed: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(l, t)| (l.as_str(), t.as_str()))
        .collect();
    Ok(CorpusIndex::build(borrowed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A problem statement long enough to contain several 13-grams. The
    /// fixtures below must clear [`NGRAM`] tokens, or the detector refuses to
    /// score them and every case looks contaminated.
    const SEARCH: &str = "implement a binary search over a sorted slice of integers \
        and return the index of the target value or none when it is absent";

    const CSV: &str = "parse a csv file with a header row and group every record \
        by the categorical value that appears in its second column";

    const MARINE: &str = "track the population of several coral reef species across \
        a series of annual surveys and report the growth rate for each one";

    const DIFF: &str = "integrate a first order differential equation numerically \
        using an explicit euler step over a configurable number of intervals";

    const SCHED: &str = "schedule a set of jobs on a single machine by descending \
        duration while respecting each job's declared deadline constraints";

    fn task(id: &str, prompt: &str) -> Task {
        Task {
            id: id.into(),
            prompt: prompt.into(),
            target: format!("src/{id}.rs"),
            tests: vec!["#[test]\nfn t() { assert!(true); }".into()],
            source: "synthetic".into(),
            language: "rust".into(),
        }
    }

    #[test]
    fn normalise_folds_case_and_punctuation() {
        let a = normalise_tokens("Convolution(a, b);");
        let b = normalise_tokens("convolution of a and b");
        assert_eq!(a[0], "convolution");
        assert!(b.contains(&"convolution".to_string()));
    }

    #[test]
    fn identical_text_has_full_overlap() {
        let index = CorpusIndex::build([("train", SEARCH)]);
        assert!(decontaminate(&index, SEARCH, "corpus").is_contaminated);
    }

    #[test]
    fn unrelated_text_is_clean() {
        let index = CorpusIndex::build([("train", CSV)]);
        let clean = decontaminate(&index, MARINE, "corpus");
        assert!(
            !clean.is_contaminated,
            "false positive: overlap {}",
            clean.overlap
        );
    }

    #[test]
    fn contamination_reports_an_example_span() {
        let index = CorpusIndex::build([("train", SEARCH)]);
        let c = decontaminate(&index, SEARCH, "corpus");
        assert!(c.example.is_some(), "a flag must be inspectable");
    }

    #[test]
    fn a_task_shorter_than_one_ngram_is_not_scored_clean() {
        let index =
            CorpusIndex::build([("train", "a much longer training problem statement here")]);
        let c = decontaminate(&index, "sort", "corpus");
        assert!(
            c.is_contaminated,
            "an uncheckable task must not pass as clean"
        );
    }

    #[test]
    fn overlap_is_a_fraction_of_the_task_not_the_corpus() {
        // Two tasks, one of which the corpus contains verbatim. The other must
        // not look clean merely because the corpus is large.
        let index = CorpusIndex::build([("train", SEARCH)]);
        let c = decontaminate(&index, MARINE, "corpus");
        assert!(
            c.overlap < OVERLAP_THRESHOLD,
            "unrelated task scored {}",
            c.overlap
        );
    }

    #[test]
    fn a_contaminated_task_is_rejected_from_the_scored_set() {
        let index = CorpusIndex::build([("train", SEARCH)]);
        let tasks = vec![task("t1", SEARCH), task("t2", DIFF)];
        let report = admit(&tasks, &index);
        assert_eq!(report.total, 2);
        assert_eq!(report.rejected.len(), 1);
        assert_eq!(report.rejected[0].id, "t1");
    }

    #[test]
    fn a_rejection_names_the_overlap_that_caused_it() {
        let index = CorpusIndex::build([("train", SEARCH)]);
        let report = admit(&[task("t1", SEARCH)], &index);
        let r = &report.rejected[0];
        assert!(r.overlap > OVERLAP_THRESHOLD);
        assert!(r.example.is_some());
        assert!(!r.reason.is_empty());
    }

    #[test]
    fn a_task_whose_tests_leak_is_rejected_even_when_its_prompt_is_original() {
        // Holding out problems while training on their tests leaks the answer.
        let test_text = "assert_eq!(binary_search(&[1, 2, 3, 5, 8, 13, 21], 13), Ok(5)); \
            assert_eq!(binary_search(&[1, 2, 3], 99), None);";
        let index = CorpusIndex::build([("train", test_text)]);
        let mut t = task("t1", SCHED);
        t.tests = vec![test_text.into()];
        let report = admit(&[t], &index);
        assert_eq!(report.rejected.len(), 1, "leaked tests were not caught");
    }

    #[test]
    fn humaneval_provenance_is_flagged_as_at_risk() {
        assert!(contamination_risk("humaneval").is_some());
        assert!(contamination_risk("mbpp-python").is_some());
        assert!(contamination_risk("leetcode").is_some());
    }

    #[test]
    fn a_synthetic_provenance_is_not_flagged() {
        assert!(contamination_risk("synthetic").is_none());
        assert!(contamination_risk("repo-native").is_none());
    }

    #[test]
    fn an_at_risk_task_is_still_admitted_if_it_passes_overlap() {
        let index = CorpusIndex::build([("train", MARINE)]);
        let mut t = task("t1", SCHED);
        t.source = "humaneval-cleaned".into();
        let report = admit(&[t], &index);
        assert_eq!(report.scored, 1);
        assert_eq!(report.at_risk.len(), 1);
    }

    #[test]
    fn split_is_stable_for_the_same_id() {
        for id in ["task-0", "task-1", "abc123", "a-much-longer-identifier"] {
            let first = split(id, 0.25);
            for _ in 0..8 {
                assert_eq!(split(id, 0.25), first, "split moved for {id}");
            }
        }
    }

    #[test]
    fn split_ignores_insertion_order() {
        // A counter-based split would move every later task when one row is
        // inserted. The hash must not.
        let ids: Vec<String> = (0..500).map(|i| format!("task-{i}")).collect();
        let before: Vec<Side> = ids.iter().map(|i| split(i, 0.3)).collect();
        let mut with_extra: Vec<String> = ids.clone();
        with_extra.insert(0, "task-new".to_string());
        let after: Vec<Side> = with_extra.iter().map(|i| split(i, 0.3)).collect();
        assert_eq!(
            &before,
            &after[1..],
            "inserting one task reshuffled the split"
        );
    }

    #[test]
    fn split_puts_roughly_the_requested_fraction_in_eval() {
        let n = 4000;
        let eval = (0..n)
            .filter(|i| split(&format!("t{i}"), 0.25) == Side::Eval)
            .count();
        let frac = eval as f64 / n as f64;
        assert!(
            (frac - 0.25).abs() < 0.03,
            "eval fraction {frac}, wanted 0.25"
        );
    }

    #[test]
    fn both_sides_are_actually_populated() {
        let n = 500;
        let train = (0..n)
            .filter(|i| split(&format!("t{i}"), 0.25) == Side::Train)
            .count();
        let eval = n - train;
        assert!(
            train > 0 && eval > 0,
            "split produced an empty side: train={train} eval={eval}"
        );
    }

    #[test]
    fn a_perfect_outcome_scores_one() {
        let o = TaskOutcome::new("t1", true, true, true, true);
        assert!(o.passed());
        assert_eq!(o.score, 1.0);
    }

    #[test]
    fn any_single_failing_gate_scores_zero() {
        for (t, c, a, h, name) in [
            (false, true, true, true, "tests"),
            (true, false, true, true, "clippy"),
            (true, true, false, true, "audit"),
            (true, true, true, false, "cheat"),
        ] {
            let o = TaskOutcome::new("t", t, c, a, h);
            assert_eq!(o.score, 0.0, "{name} failure still scored");
            assert!(!o.passed());
        }
    }

    #[test]
    fn a_cheating_outcome_is_not_a_partial_win() {
        // Green tests, but the pass was bought with a suppression.
        let o = TaskOutcome::new("t", true, true, true, false);
        assert_eq!(o.score, 0.0);
        assert_eq!(o.failures(), vec!["cheat"]);
    }

    #[test]
    fn pass_rate_is_the_fraction_of_fully_clean_outcomes() {
        let outcomes = vec![
            TaskOutcome::new("a", true, true, true, true),
            TaskOutcome::new("b", true, true, true, false),
            TaskOutcome::new("c", false, true, true, true),
            TaskOutcome::new("d", true, true, true, true),
        ];
        let card = Scorecard::new(outcomes);
        assert_eq!(card.scored, 4);
        assert_eq!(card.passed, 2);
        assert_eq!(card.pass_rate, 0.5);
    }

    #[test]
    fn failure_breakdown_counts_every_gate() {
        let outcomes = vec![
            TaskOutcome::new("a", false, false, true, true),
            TaskOutcome::new("b", true, false, true, true),
        ];
        let card = Scorecard::new(outcomes);
        let breakdown = card.failure_breakdown();
        assert_eq!(breakdown.get("tests"), Some(&1));
        assert_eq!(breakdown.get("clippy"), Some(&2));
    }

    #[test]
    fn an_empty_scorecard_is_zero_not_a_pass() {
        let card = Scorecard::new(Vec::new());
        assert!(card.is_empty());
        assert_eq!(card.pass_rate, 0.0);
    }

    #[test]
    fn yield_rate_reports_how_much_of_the_set_was_scorable() {
        let index = CorpusIndex::build([("train", SEARCH)]);
        let tasks = vec![task("bad", SEARCH), task("good", MARINE)];
        let report = admit(&tasks, &index);
        assert!((report.yield_rate() - 0.5).abs() < f64::EPSILON);
    }
}
