//! Fixed-budget evaluation for Nina's numeric and multiple-choice probes.
//!
//! Score the frozen case population by ID, including failed and missing
//! attempts. The same verified case file can be scored against multiple
//! checkpoints or runtimes without changing the denominator.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The answer domain for an independently labelled case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Answer {
    Integer(i64),
    Choice(char),
}

impl Answer {
    fn validate(self) -> Result<()> {
        if let Self::Choice(letter) = self {
            ensure!(matches!(letter, 'A'..='H'), "choice must be ASCII A–H");
        }
        Ok(())
    }

    fn same_domain(self, other: Self) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }
}

/// One greedy draw per case at a fixed generation limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationBudget {
    pub max_output_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationCase {
    pub id: String,
    pub answer: Answer,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairKind {
    Invariant,
    AnswerChanging,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPair {
    pub left: String,
    pub right: String,
    pub kind: PairKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationSet {
    pub budget: EvaluationBudget,
    pub cases: Vec<EvaluationCase>,
    #[serde(default)]
    pub pairs: Vec<EvaluationPair>,
}

impl EvaluationSet {
    /// Validate the case plan and hash its canonical JSON, including the
    /// optional rendered prompts and the output cap.
    pub fn sha256(&self) -> Result<String> {
        Ok(PreparedSet::new(self)?.sha256)
    }
}

/// Runner's observation of how a single generation finished.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceStatus {
    #[default]
    Completed,
    Truncated,
    Abstained,
    Unsupported,
    Error,
    Pending,
    Timeout,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[serde(default)]
    pub status: InferenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionRun {
    pub checkpoint_sha256: String,
    pub runtime: String,
    pub budget: EvaluationBudget,
    /// Provide the frozen plan's digest for a newly collected run. Historical
    /// archives can be rescored without an existing plan pin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_sha256: Option<String>,
    pub predictions: Vec<Prediction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    Answered,
    Invalid,
    Ambiguous,
    Empty,
    Missing,
    Truncated,
    Abstained,
    Unsupported,
    Error,
    Pending,
    Timeout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Verdict {
    pub status: OutcomeStatus,
    pub parsed: Option<Answer>,
    pub canonical_format: bool,
    pub strict_correct: bool,
    pub semantic_correct: bool,
}

impl Verdict {
    fn without_answer(status: OutcomeStatus) -> Self {
        Self {
            status,
            parsed: None,
            canonical_format: false,
            strict_correct: false,
            semantic_correct: false,
        }
    }
}

fn integer(text: &str) -> Option<i64> {
    let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

fn choice(text: &str) -> Option<char> {
    match text.as_bytes() {
        [letter] if matches!(letter.to_ascii_uppercase(), b'A'..=b'H') => {
            Some(char::from(letter.to_ascii_uppercase()))
        }
        _ => None,
    }
}

fn answer_payload(text: &str) -> &str {
    if let Some((prefix, rest)) = text.split_once(':') {
        if prefix.trim_end().eq_ignore_ascii_case("answer") {
            return rest.trim_start();
        }
    }
    text
}

fn strip_period(text: &str) -> &str {
    text.strip_suffix('.').unwrap_or(text)
}

/// Check format and correctness independently. Integer semantics accept one
/// signed ASCII integer, optionally labelled `Answer:` or ending in a period.
/// Multiple plausible values remain ambiguous. Decimal, fractional and free
/// prose explanations are outside this research rubric.
pub fn grade(answer: Answer, prediction: Option<&Prediction>) -> Verdict {
    let Some(prediction) = prediction else {
        return Verdict::without_answer(OutcomeStatus::Missing);
    };
    let status = match prediction.status {
        InferenceStatus::Completed => None,
        InferenceStatus::Truncated => Some(OutcomeStatus::Truncated),
        InferenceStatus::Abstained => Some(OutcomeStatus::Abstained),
        InferenceStatus::Unsupported => Some(OutcomeStatus::Unsupported),
        InferenceStatus::Error => Some(OutcomeStatus::Error),
        InferenceStatus::Pending => Some(OutcomeStatus::Pending),
        InferenceStatus::Timeout => Some(OutcomeStatus::Timeout),
    };
    if let Some(status) = status {
        return Verdict::without_answer(status);
    }
    let text = prediction.response.as_deref().unwrap_or_default().trim();
    if text.is_empty() {
        return Verdict::without_answer(OutcomeStatus::Empty);
    }

    let payload = answer_payload(text);
    let parse: fn(&str) -> Option<Answer> = match answer {
        Answer::Integer(_) => |value| integer(value).map(Answer::Integer),
        Answer::Choice(_) => |value| choice(value).map(Answer::Choice),
    };
    let Some(parsed) = parse(strip_period(payload)) else {
        let multiple = payload
            .split(|c: char| !c.is_alphanumeric() && !matches!(c, '+' | '-' | '.'))
            .filter(|token| parse(strip_period(token)).is_some())
            .take(2)
            .count()
            == 2;
        return Verdict::without_answer(if multiple {
            OutcomeStatus::Ambiguous
        } else {
            OutcomeStatus::Invalid
        });
    };

    let canonical_format = match parsed {
        Answer::Integer(value) => text == value.to_string() || (value == 0 && text == "-0"),
        Answer::Choice(letter) => text.len() == 1 && text.starts_with(letter),
    };
    let semantic_correct = parsed == answer;
    Verdict {
        status: OutcomeStatus::Answered,
        parsed: Some(parsed),
        canonical_format,
        strict_correct: canonical_format && semantic_correct,
        semantic_correct,
    }
}

/// Qualification examples drawn from the saved arithmetic grader preflight.
#[derive(Debug, Clone, Serialize)]
pub struct GraderFixture {
    pub target: i64,
    pub response: &'static str,
    pub strict: bool,
    pub semantic: bool,
    pub status: OutcomeStatus,
    pub matches_fixture: bool,
}

pub fn grade_fixtures() -> [GraderFixture; 7] {
    [
        (17, "17", true, true, OutcomeStatus::Answered),
        (-4, "  -4\n", true, true, OutcomeStatus::Answered),
        (3, "Answer: +003.", false, true, OutcomeStatus::Answered),
        (4, "5", false, false, OutcomeStatus::Answered),
        (7, "7 or 9", false, false, OutcomeStatus::Ambiguous),
        (3, "3.0", false, false, OutcomeStatus::Invalid),
        (0, "", false, false, OutcomeStatus::Empty),
    ]
    .map(|(target, response, strict, semantic, status)| {
        let prediction = Prediction {
            id: "grader-qualification".to_owned(),
            response: Some(response.to_owned()),
            status: InferenceStatus::Completed,
        };
        let verdict = grade(Answer::Integer(target), Some(&prediction));
        GraderFixture {
            target,
            response,
            strict: verdict.strict_correct,
            semantic: verdict.semantic_correct,
            status: verdict.status,
            matches_fixture: (
                verdict.strict_correct,
                verdict.semantic_correct,
                verdict.status,
            ) == (strict, semantic, status),
        }
    })
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ScoreTotals {
    pub planned: usize,
    pub canonical_format: usize,
    pub strict_correct: usize,
    pub semantic_correct: usize,
    pub statuses: BTreeMap<OutcomeStatus, usize>,
}

impl ScoreTotals {
    fn record(&mut self, verdict: Verdict) {
        self.planned += 1;
        self.canonical_format += usize::from(verdict.canonical_format);
        self.strict_correct += usize::from(verdict.strict_correct);
        self.semantic_correct += usize::from(verdict.semantic_correct);
        *self.statuses.entry(verdict.status).or_default() += 1;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Comparison {
    pub gained: usize,
    pub lost: usize,
    pub both_correct: usize,
    pub both_incorrect: usize,
}

impl Comparison {
    fn record(&mut self, incumbent: bool, challenger: bool) {
        match (incumbent, challenger) {
            (true, true) => self.both_correct += 1,
            (true, false) => self.lost += 1,
            (false, true) => self.gained += 1,
            (false, false) => self.both_incorrect += 1,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PairCounts {
    pub both_parsed: usize,
    pub relation_observed: usize,
    pub strict_joint_correct: usize,
    pub semantic_joint_correct: usize,
}

impl PairCounts {
    fn record(&mut self, left: Verdict, right: Verdict, kind: PairKind) -> (bool, bool) {
        if let (Some(left), Some(right)) = (left.parsed, right.parsed) {
            self.both_parsed += 1;
            self.relation_observed += usize::from((left == right) == (kind == PairKind::Invariant));
        }
        let strict = left.strict_correct && right.strict_correct;
        let semantic = left.semantic_correct && right.semantic_correct;
        self.strict_joint_correct += usize::from(strict);
        self.semantic_joint_correct += usize::from(semantic);
        (strict, semantic)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PairReport {
    pub planned: usize,
    pub incumbent: PairCounts,
    pub challenger: PairCounts,
    pub strict: Comparison,
    pub semantic: Comparison,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SliceReport {
    pub incumbent: ScoreTotals,
    pub challenger: ScoreTotals,
    pub strict: Comparison,
    pub semantic: Comparison,
}

impl SliceReport {
    fn record(&mut self, incumbent: Verdict, challenger: Verdict) {
        self.incumbent.record(incumbent);
        self.challenger.record(challenger);
        self.strict
            .record(incumbent.strict_correct, challenger.strict_correct);
        self.semantic
            .record(incumbent.semantic_correct, challenger.semantic_correct);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CaseComparison {
    pub id: String,
    pub incumbent: Verdict,
    pub challenger: Verdict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunIdentity {
    pub checkpoint_sha256: String,
    pub runtime: String,
    pub protocol_bound: bool,
}

impl From<&PredictionRun> for RunIdentity {
    fn from(run: &PredictionRun) -> Self {
        Self {
            checkpoint_sha256: run.checkpoint_sha256.clone(),
            runtime: run.runtime.clone(),
            protocol_bound: run.protocol_sha256.is_some(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PairedReport {
    pub protocol_sha256: String,
    pub budget: EvaluationBudget,
    pub incumbent_run: RunIdentity,
    pub challenger_run: RunIdentity,
    pub incumbent: ScoreTotals,
    pub challenger: ScoreTotals,
    pub strict: Comparison,
    pub semantic: Comparison,
    pub invariant: PairReport,
    pub answer_changing: PairReport,
    pub slices: BTreeMap<String, SliceReport>,
    pub cases: Vec<CaseComparison>,
}

#[derive(Clone, Copy)]
struct Observation<'a> {
    prediction: Option<&'a Prediction>,
    verdict: Verdict,
}

struct PreparedSet<'a> {
    set: &'a EvaluationSet,
    sha256: String,
    indices: BTreeMap<&'a str, usize>,
    pairs: Vec<(PairKind, usize, usize)>,
}

fn valid_sha256(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

impl<'a> PreparedSet<'a> {
    fn new(set: &'a EvaluationSet) -> Result<Self> {
        ensure!(
            set.budget.max_output_tokens > 0,
            "output cap must be positive"
        );
        ensure!(
            !set.cases.is_empty(),
            "evaluation requires at least one case"
        );
        let mut indices = BTreeMap::new();
        for (index, case) in set.cases.iter().enumerate() {
            ensure!(!case.id.trim().is_empty(), "evaluation case ID is empty");
            ensure!(
                indices.insert(case.id.as_str(), index).is_none(),
                "duplicate evaluation case ID {:?}",
                case.id
            );
            case.answer
                .validate()
                .with_context(|| format!("invalid answer for case {:?}", case.id))?;
        }
        let mut seen_pairs = BTreeSet::new();
        let mut pairs = Vec::with_capacity(set.pairs.len());
        for pair in &set.pairs {
            let index = |id: &str| {
                indices
                    .get(id)
                    .copied()
                    .with_context(|| format!("unknown paired case {id:?}"))
            };
            let left = index(&pair.left)?;
            let right = index(&pair.right)?;
            ensure!(left != right, "case {:?} pairs with itself", pair.left);
            ensure!(
                seen_pairs.insert((left.min(right), left.max(right))),
                "duplicate pair {:?} and {:?}",
                pair.left,
                pair.right
            );
            let a = set.cases[left].answer;
            let b = set.cases[right].answer;
            ensure!(
                a.same_domain(b) && (a == b) == (pair.kind == PairKind::Invariant),
                "gold answers disagree with {:?} pair {:?} and {:?}",
                pair.kind,
                pair.left,
                pair.right
            );
            pairs.push((pair.kind, left, right));
        }
        let canonical = crate::nina::canonical_bytes(&serde_json::to_value(set)?);
        Ok(Self {
            set,
            sha256: format!("{:x}", Sha256::digest(canonical)),
            indices,
            pairs,
        })
    }

    fn observe<'r>(&self, run: &'r PredictionRun) -> Result<Vec<Observation<'r>>> {
        ensure!(
            run.budget == self.set.budget,
            "prediction budget differs from plan"
        );
        ensure!(!run.runtime.trim().is_empty(), "runtime identity is empty");
        ensure!(
            valid_sha256(&run.checkpoint_sha256),
            "checkpoint identity must be a lowercase SHA-256 digest"
        );
        if let Some(pin) = &run.protocol_sha256 {
            ensure!(
                pin == &self.sha256,
                "prediction protocol digest differs from plan"
            );
        }
        let mut predictions = vec![None; self.set.cases.len()];
        for prediction in &run.predictions {
            let index = *self
                .indices
                .get(prediction.id.as_str())
                .with_context(|| format!("unexpected prediction ID {:?}", prediction.id))?;
            ensure!(
                predictions[index].replace(prediction).is_none(),
                "duplicate prediction ID {:?}",
                prediction.id
            );
        }
        Ok(predictions
            .into_iter()
            .zip(&self.set.cases)
            .map(|(prediction, case)| Observation {
                prediction,
                verdict: grade(case.answer, prediction),
            })
            .collect())
    }

    fn compare(
        &self,
        incumbent_run: &PredictionRun,
        challenger_run: &PredictionRun,
        incumbent: &[Observation<'_>],
        challenger: &[Observation<'_>],
    ) -> PairedReport {
        let mut totals = SliceReport::default();
        let mut slices: BTreeMap<String, SliceReport> = BTreeMap::new();
        let cases = self
            .set
            .cases
            .iter()
            .zip(incumbent.iter().zip(challenger))
            .map(|(case, (incumbent, challenger))| {
                totals.record(incumbent.verdict, challenger.verdict);
                if let Some(slice) = &case.slice {
                    slices
                        .entry(slice.clone())
                        .or_default()
                        .record(incumbent.verdict, challenger.verdict);
                }
                CaseComparison {
                    id: case.id.clone(),
                    incumbent: incumbent.verdict,
                    challenger: challenger.verdict,
                }
            })
            .collect();

        let mut invariant = PairReport::default();
        let mut answer_changing = PairReport::default();
        for &(kind, left, right) in &self.pairs {
            let report = match kind {
                PairKind::Invariant => &mut invariant,
                PairKind::AnswerChanging => &mut answer_changing,
            };
            report.planned += 1;
            let (incumbent_strict, incumbent_semantic) =
                report
                    .incumbent
                    .record(incumbent[left].verdict, incumbent[right].verdict, kind);
            let (challenger_strict, challenger_semantic) =
                report
                    .challenger
                    .record(challenger[left].verdict, challenger[right].verdict, kind);
            report.strict.record(incumbent_strict, challenger_strict);
            report
                .semantic
                .record(incumbent_semantic, challenger_semantic);
        }

        PairedReport {
            protocol_sha256: self.sha256.clone(),
            budget: self.set.budget,
            incumbent_run: incumbent_run.into(),
            challenger_run: challenger_run.into(),
            incumbent: totals.incumbent,
            challenger: totals.challenger,
            strict: totals.strict,
            semantic: totals.semantic,
            invariant,
            answer_changing,
            slices,
            cases,
        }
    }
}

/// Score both arms on the same planned case IDs. Missing predictions remain
/// in the denominator as missing outcomes.
pub fn score_paired(
    set: &EvaluationSet,
    incumbent: &PredictionRun,
    challenger: &PredictionRun,
) -> Result<PairedReport> {
    let prepared = PreparedSet::new(set)?;
    let a = prepared
        .observe(incumbent)
        .context("incumbent predictions")?;
    let b = prepared
        .observe(challenger)
        .context("challenger predictions")?;
    Ok(prepared.compare(incumbent, challenger, &a, &b))
}

/// Counts from one first-draw wave. `observed` counts decoded outcomes and
/// `usable` counts nonempty completions, including incorrect answers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct FirstDrawTotals {
    pub scheduled: usize,
    pub observed: usize,
    pub usable: usize,
    pub statuses: BTreeMap<OutcomeStatus, usize>,
}

impl FirstDrawTotals {
    fn record(&mut self, status: OutcomeStatus, usable: bool) {
        self.scheduled += 1;
        self.observed += usize::from(
            usable || matches!(status, OutcomeStatus::Empty | OutcomeStatus::Truncated),
        );
        self.usable += usize::from(usable);
        *self.statuses.entry(status).or_default() += 1;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FirstDrawCase {
    pub id: String,
    pub slice: Option<String>,
    pub status: OutcomeStatus,
    pub usable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FirstDrawReport {
    pub protocol_sha256: String,
    pub run: RunIdentity,
    pub budget: EvaluationBudget,
    pub ready_for_extra_draws: bool,
    pub totals: FirstDrawTotals,
    pub slices: BTreeMap<String, FirstDrawTotals>,
    pub cases: Vec<FirstDrawCase>,
}

/// Check all scheduled case IDs before opening a subsequent draw wave. An
/// available incorrect completion supplies a first-draw measurement.
pub fn score_first_draws(set: &EvaluationSet, run: &PredictionRun) -> Result<FirstDrawReport> {
    let prepared = PreparedSet::new(set)?;
    let observations = prepared.observe(run)?;
    let mut totals = FirstDrawTotals::default();
    let mut slices: BTreeMap<String, FirstDrawTotals> = BTreeMap::new();
    let cases = set
        .cases
        .iter()
        .zip(observations)
        .map(|(case, observation)| {
            let status = observation.verdict.status;
            let usable = matches!(
                status,
                OutcomeStatus::Answered | OutcomeStatus::Invalid | OutcomeStatus::Ambiguous
            );
            totals.record(status, usable);
            if let Some(slice) = &case.slice {
                slices
                    .entry(slice.clone())
                    .or_default()
                    .record(status, usable);
            }
            FirstDrawCase {
                id: case.id.clone(),
                slice: case.slice.clone(),
                status,
                usable,
            }
        })
        .collect();
    Ok(FirstDrawReport {
        protocol_sha256: prepared.sha256,
        run: run.into(),
        budget: set.budget,
        ready_for_extra_draws: totals.usable == totals.scheduled,
        totals,
        slices,
        cases,
    })
}

/// One correct count from a pinned development cohort at one training rung.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankObservation {
    pub arm: String,
    pub seed: u64,
    pub step: usize,
    pub cohort_sha256: String,
    pub development_cases: usize,
    pub correct: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankAuditInput {
    pub cohort_sha256: String,
    pub development_cases: usize,
    pub early_step: usize,
    pub full_step: usize,
    pub seeds: Vec<u64>,
    pub arms: Vec<String>,
    pub observations: Vec<RankObservation>,
}

/// Sum of correct counts over the common seeds. Equal denominators make the
/// rank exact in integer arithmetic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct RankTotals {
    pub early_correct_sum: usize,
    pub full_correct_sum: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RankAuditReport {
    pub cohort_sha256: String,
    pub development_cases: usize,
    pub early_step: usize,
    pub full_step: usize,
    pub seeds: Vec<u64>,
    pub totals: BTreeMap<String, RankTotals>,
    pub early_rank: Vec<Vec<String>>,
    pub full_rank: Vec<Vec<String>>,
    pub reversed_pairs: Vec<[String; 2]>,
    pub early_ties: Vec<[String; 2]>,
    pub full_ties: Vec<[String; 2]>,
}

fn rank_groups(
    totals: &BTreeMap<String, RankTotals>,
    score: impl Fn(&RankTotals) -> usize,
) -> Vec<Vec<String>> {
    let mut groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (arm, total) in totals {
        groups.entry(score(total)).or_default().push(arm.clone());
    }
    groups.into_values().rev().collect()
}

/// Audit a matched early and full rung with the same cohort, arms and seeds.
/// Ties are reported separately from pairs whose strict order reverses.
pub fn audit_early_rank(input: &RankAuditInput) -> Result<RankAuditReport> {
    ensure!(
        valid_sha256(&input.cohort_sha256),
        "invalid cohort SHA-256 digest"
    );
    ensure!(input.development_cases > 0, "development cohort is empty");
    ensure!(
        input.early_step < input.full_step,
        "early rung must precede full rung"
    );
    let seeds: BTreeSet<_> = input.seeds.iter().copied().collect();
    ensure!(
        !seeds.is_empty() && seeds.len() == input.seeds.len(),
        "rank audit needs unique seeds"
    );
    let arms: BTreeSet<_> = input.arms.iter().map(String::as_str).collect();
    ensure!(
        arms.len() >= 2
            && arms.len() == input.arms.len()
            && arms.iter().all(|arm| !arm.trim().is_empty()),
        "rank audit needs at least two unique named arms"
    );
    let mut observed = BTreeMap::new();
    for row in &input.observations {
        let id = (row.arm.as_str(), row.seed, row.step);
        ensure!(
            arms.contains(id.0)
                && seeds.contains(&id.1)
                && (id.2 == input.early_step || id.2 == input.full_step),
            "unscheduled rank observation {id:?}"
        );
        ensure!(
            row.cohort_sha256 == input.cohort_sha256
                && row.development_cases == input.development_cases,
            "cohort differs for rank observation {id:?}"
        );
        ensure!(
            row.correct <= row.development_cases,
            "invalid correct count for {id:?}"
        );
        ensure!(
            observed.insert(id, row.correct).is_none(),
            "duplicate rank observation {id:?}"
        );
    }
    let mut totals = BTreeMap::new();
    for arm in &input.arms {
        let sum = |step| -> Result<usize> {
            input.seeds.iter().try_fold(0usize, |total, &seed| {
                let id = (arm.as_str(), seed, step);
                let correct = observed
                    .get(&id)
                    .with_context(|| format!("missing rank observation {id:?}"))?;
                total.checked_add(*correct).context("rank count overflow")
            })
        };
        totals.insert(
            arm.clone(),
            RankTotals {
                early_correct_sum: sum(input.early_step)?,
                full_correct_sum: sum(input.full_step)?,
            },
        );
    }

    let mut reversed_pairs = Vec::new();
    let mut early_ties = Vec::new();
    let mut full_ties = Vec::new();
    use std::cmp::Ordering::{Equal, Greater, Less};
    for (index, left) in input.arms.iter().enumerate() {
        let a = &totals[left];
        for right in input.arms.iter().skip(index + 1) {
            let b = &totals[right];
            let early = a.early_correct_sum.cmp(&b.early_correct_sum);
            let full = a.full_correct_sum.cmp(&b.full_correct_sum);
            let pair = || [left.clone(), right.clone()];
            if early == Equal {
                early_ties.push(pair());
            }
            if full == Equal {
                full_ties.push(pair());
            }
            if matches!((early, full), (Less, Greater) | (Greater, Less)) {
                reversed_pairs.push(pair());
            }
        }
    }
    Ok(RankAuditReport {
        cohort_sha256: input.cohort_sha256.clone(),
        development_cases: input.development_cases,
        early_step: input.early_step,
        full_step: input.full_step,
        seeds: input.seeds.clone(),
        early_rank: rank_groups(&totals, |row| row.early_correct_sum),
        full_rank: rank_groups(&totals, |row| row.full_correct_sum),
        totals,
        reversed_pairs,
        early_ties,
        full_ties,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeMatrix {
    pub incumbent_canonical: PredictionRun,
    pub challenger_canonical: PredictionRun,
    pub incumbent_staged: PredictionRun,
    pub challenger_staged: PredictionRun,
}

/// Runtime effect with checkpoint identity held fixed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct RuntimeChange {
    pub both_observed: usize,
    pub both_completed: usize,
    pub raw_equal: usize,
    pub raw_changed: Vec<String>,
    pub missing: Vec<String>,
    pub strict: Comparison,
    pub semantic: Comparison,
}

fn runtime_change(
    set: &EvaluationSet,
    canonical: &[Observation<'_>],
    staged: &[Observation<'_>],
) -> RuntimeChange {
    let mut change = RuntimeChange::default();
    for (case, (a, b)) in set.cases.iter().zip(canonical.iter().zip(staged)) {
        change
            .strict
            .record(a.verdict.strict_correct, b.verdict.strict_correct);
        change
            .semantic
            .record(a.verdict.semantic_correct, b.verdict.semantic_correct);
        match (a.prediction, b.prediction) {
            (Some(a), Some(b)) => {
                change.both_observed += 1;
                if a.status == InferenceStatus::Completed && b.status == InferenceStatus::Completed
                {
                    change.both_completed += 1;
                }
                if a == b {
                    change.raw_equal += 1;
                } else {
                    change.raw_changed.push(case.id.clone());
                }
            }
            _ => change.missing.push(case.id.clone()),
        }
    }
    change
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeReport {
    pub canonical: PairedReport,
    pub staged: PairedReport,
    pub incumbent_runtime_change: RuntimeChange,
    pub challenger_runtime_change: RuntimeChange,
}

/// Cross both checkpoint and runtime axes. Compare raw output as well as
/// grades so equal aggregate accuracy cannot hide a serving regression.
pub fn score_runtime(set: &EvaluationSet, matrix: &RuntimeMatrix) -> Result<RuntimeReport> {
    let a = &matrix.incumbent_canonical;
    let b = &matrix.challenger_canonical;
    let c = &matrix.incumbent_staged;
    let d = &matrix.challenger_staged;
    ensure!(
        a.checkpoint_sha256 == c.checkpoint_sha256
            && b.checkpoint_sha256 == d.checkpoint_sha256
            && a.checkpoint_sha256 != b.checkpoint_sha256,
        "matrix must hold each of two distinct checkpoints fixed across runtimes"
    );
    ensure!(
        a.runtime == b.runtime && c.runtime == d.runtime && a.runtime != c.runtime,
        "matrix must cross two distinct runtimes with both checkpoints"
    );
    let prepared = PreparedSet::new(set)?;
    let observed_a = prepared
        .observe(a)
        .context("canonical incumbent predictions")?;
    let observed_b = prepared
        .observe(b)
        .context("canonical challenger predictions")?;
    let observed_c = prepared
        .observe(c)
        .context("staged incumbent predictions")?;
    let observed_d = prepared
        .observe(d)
        .context("staged challenger predictions")?;
    Ok(RuntimeReport {
        canonical: prepared.compare(a, b, &observed_a, &observed_b),
        staged: prepared.compare(c, d, &observed_c, &observed_d),
        incumbent_runtime_change: runtime_change(set, &observed_a, &observed_c),
        challenger_runtime_change: runtime_change(set, &observed_b, &observed_d),
    })
}

/// Majority reference derived exclusively from the supplied training labels.
/// Ties choose the smallest numeric value or earliest choice letter.
pub fn train_majority(training_answers: &[Answer]) -> Result<Answer> {
    let first = *training_answers
        .first()
        .context("majority requires training labels")?;
    let mut histogram: BTreeMap<Answer, usize> = BTreeMap::new();
    for &answer in training_answers {
        answer.validate()?;
        ensure!(
            answer.same_domain(first),
            "training labels mix answer domains"
        );
        *histogram.entry(answer).or_default() += 1;
    }
    histogram
        .into_iter()
        .max_by_key(|&(answer, count)| (count, std::cmp::Reverse(answer)))
        .map(|(answer, _)| answer)
        .context("majority requires training labels")
}

const CHOICE_LABELS: [char; 4] = ['A', 'B', 'C', 'D'];

/// A source question, with `correct` indexing the original options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChoiceCase {
    pub id: String,
    pub question: String,
    pub options: [String; 4],
    pub correct: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChoiceLayout {
    pub id: String,
    pub displayed_to_original: [usize; 4],
    pub prompt: String,
    pub answer: char,
}

/// Render each original option once in each ASCII A–D position. Distinct
/// layout IDs make the denominator and the mapping back to option text fixed.
pub fn rotate_choices(case: &ChoiceCase) -> Result<[ChoiceLayout; 4]> {
    ensure!(!case.id.trim().is_empty(), "choice case ID is empty");
    ensure!(
        !case.question.trim().is_empty(),
        "choice question {:?} is empty",
        case.id
    );
    ensure!(
        case.correct < 4,
        "gold option out of range for {:?}",
        case.id
    );
    let unique: BTreeSet<_> = case
        .options
        .iter()
        .map(|option| option.trim().to_lowercase())
        .collect();
    ensure!(
        unique.len() == 4 && !unique.contains(""),
        "blank or duplicate options for {:?}",
        case.id
    );
    Ok(std::array::from_fn(|rotation| {
        let displayed_to_original = std::array::from_fn(|slot| (slot + rotation) % 4);
        let mut prompt = format!("Question:\n{}\n", case.question);
        for (&label, &index) in CHOICE_LABELS.iter().zip(&displayed_to_original) {
            prompt.push_str(&format!("{label}. {}\n", case.options[index]));
        }
        prompt.push_str("Answer:\n");
        ChoiceLayout {
            id: format!("{}/layout-{rotation}", case.id),
            displayed_to_original,
            prompt,
            answer: CHOICE_LABELS[(case.correct + 4 - rotation) % 4],
        }
    }))
}

/// Render the corresponding pinned evaluation set. The four layouts per
/// source case never become four independent question families.
pub fn choice_set(items: &[ChoiceCase], budget: EvaluationBudget) -> Result<EvaluationSet> {
    let layouts = items
        .iter()
        .map(rotate_choices)
        .collect::<Result<Vec<_>>>()?;
    let set = choice_plan(items, &layouts, budget);
    PreparedSet::new(&set)?;
    Ok(set)
}

fn choice_plan(
    items: &[ChoiceCase],
    layouts: &[[ChoiceLayout; 4]],
    budget: EvaluationBudget,
) -> EvaluationSet {
    EvaluationSet {
        budget,
        cases: items
            .iter()
            .zip(layouts)
            .flat_map(|(item, layouts)| {
                layouts.iter().map(|layout| EvaluationCase {
                    id: layout.id.clone(),
                    answer: Answer::Choice(layout.answer),
                    prompt: Some(layout.prompt.clone()),
                    family: Some(item.id.clone()),
                    slice: item.source.clone(),
                })
            })
            .collect(),
        pairs: Vec::new(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PositionCaseScore {
    pub id: String,
    pub all_layouts_parsed: bool,
    pub same_original_option: bool,
    pub all_strict_correct: bool,
    pub all_semantic_correct: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PositionReport {
    pub protocol_sha256: String,
    pub run: RunIdentity,
    pub planned_cases: usize,
    pub all_layouts_parsed: usize,
    pub same_original_option: usize,
    pub all_strict_correct: usize,
    pub all_semantic_correct: usize,
    pub by_gold_slot: BTreeMap<char, ScoreTotals>,
    pub by_predicted_slot: BTreeMap<char, usize>,
    pub cases: Vec<PositionCaseScore>,
}

/// Compare all four layouts after mapping each displayed letter back to its
/// original option. A question with an unavailable layout cannot qualify as
/// consistent or jointly correct.
pub fn score_choice_positions(items: &[ChoiceCase], run: &PredictionRun) -> Result<PositionReport> {
    let layouts = items
        .iter()
        .map(rotate_choices)
        .collect::<Result<Vec<_>>>()?;
    let set = choice_plan(items, &layouts, run.budget);
    let prepared = PreparedSet::new(&set)?;
    let observed = prepared.observe(run)?;
    let mut report = PositionReport {
        protocol_sha256: prepared.sha256,
        run: run.into(),
        planned_cases: items.len(),
        all_layouts_parsed: 0,
        same_original_option: 0,
        all_strict_correct: 0,
        all_semantic_correct: 0,
        by_gold_slot: BTreeMap::new(),
        by_predicted_slot: BTreeMap::new(),
        cases: Vec::with_capacity(items.len()),
    };
    for ((item, layouts), observed) in items.iter().zip(&layouts).zip(observed.chunks_exact(4)) {
        let mut count = 0;
        let mut original_options = BTreeSet::new();
        for (layout, observation) in layouts.iter().zip(observed) {
            let mut verdict = observation.verdict;
            if let Some(Answer::Choice(letter)) = observation.verdict.parsed {
                if let Some(displayed) = CHOICE_LABELS.iter().position(|&label| label == letter) {
                    *report.by_predicted_slot.entry(letter).or_default() += 1;
                    original_options.insert(layout.displayed_to_original[displayed]);
                    count += 1;
                } else {
                    verdict = Verdict::without_answer(OutcomeStatus::Invalid);
                }
            }
            report
                .by_gold_slot
                .entry(layout.answer)
                .or_default()
                .record(verdict);
        }
        let score = PositionCaseScore {
            id: item.id.clone(),
            all_layouts_parsed: count == 4,
            same_original_option: count == 4 && original_options.len() == 1,
            all_strict_correct: observed.iter().all(|o| o.verdict.strict_correct),
            all_semantic_correct: observed.iter().all(|o| o.verdict.semantic_correct),
        };
        report.all_layouts_parsed += usize::from(score.all_layouts_parsed);
        report.same_original_option += usize::from(score.same_original_option);
        report.all_strict_correct += usize::from(score.all_strict_correct);
        report.all_semantic_correct += usize::from(score.all_semantic_correct);
        report.cases.push(score);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::{
        audit_early_rank, choice_set, grade, grade_fixtures, rotate_choices,
        score_choice_positions, score_first_draws, score_paired, score_runtime, train_majority,
        Answer, ChoiceCase, EvaluationBudget, EvaluationCase, EvaluationPair, EvaluationSet,
        InferenceStatus, OutcomeStatus, PairKind, Prediction, PredictionRun, RankAuditInput,
        RankObservation, RuntimeMatrix, CHOICE_LABELS,
    };
    use anyhow::Result;

    const BUDGET: EvaluationBudget = EvaluationBudget {
        max_output_tokens: 6,
    };

    fn case(id: &str, value: i64, slice: &str) -> EvaluationCase {
        EvaluationCase {
            id: id.to_owned(),
            answer: Answer::Integer(value),
            family: Some("arithmetic".to_owned()),
            slice: Some(slice.to_owned()),
            prompt: None,
        }
    }

    fn pair(left: &str, right: &str, kind: PairKind) -> EvaluationPair {
        EvaluationPair {
            left: left.to_owned(),
            right: right.to_owned(),
            kind,
        }
    }

    fn prediction(id: &str, response: &str) -> Prediction {
        Prediction {
            id: id.to_owned(),
            response: Some(response.to_owned()),
            status: InferenceStatus::Completed,
        }
    }

    fn run(set: &EvaluationSet, checkpoint: char, runtime: &str) -> Result<PredictionRun> {
        Ok(PredictionRun {
            checkpoint_sha256: checkpoint.to_string().repeat(64),
            runtime: runtime.to_owned(),
            budget: set.budget,
            protocol_sha256: Some(set.sha256()?),
            predictions: Vec::new(),
        })
    }

    #[test]
    fn qualification_checks_strict_semantic_ambiguity_and_empty_cases() {
        let fixtures = grade_fixtures();
        assert_eq!(fixtures.len(), 7);
        assert!(fixtures.iter().all(|fixture| fixture.matches_fixture));
        let negative_zero = prediction("zero", "-0");
        assert!(grade(Answer::Integer(0), Some(&negative_zero)).strict_correct);
        let overflow = prediction("large", "9223372036854775808");
        assert_eq!(
            grade(Answer::Integer(0), Some(&overflow)).status,
            OutcomeStatus::Invalid
        );
        let decimal = prediction("decimal", "Answer : +3.0");
        assert_eq!(
            grade(Answer::Integer(3), Some(&decimal)).status,
            OutcomeStatus::Invalid
        );
        let wider_choice = prediction("wider-choice", "H");
        assert!(grade(Answer::Choice('H'), Some(&wider_choice)).strict_correct);
    }

    #[test]
    fn paired_scores_keep_all_ids_and_measure_coverage_and_relations() -> Result<()> {
        let set = EvaluationSet {
            budget: BUDGET,
            cases: vec![
                case("a", 3, "canonical"),
                case("a-worded", 3, "surface"),
                case("b", 4, "canonical"),
                case("partial", 0, "canonical"),
                case("unsupported", 0, "canonical"),
                case("failed", 0, "canonical"),
            ],
            pairs: vec![
                pair("a", "a-worded", PairKind::Invariant),
                pair("a", "b", PairKind::AnswerChanging),
            ],
        };
        let mut a = run(&set, 'a', "cpu")?;
        a.predictions = vec![
            prediction("b", "4"),
            prediction("a-worded", "answer: +003."),
            prediction("a", "3"),
        ];
        let mut b = run(&set, 'b', "cpu")?;
        b.predictions = vec![
            prediction("a", "3"),
            prediction("a-worded", "3"),
            prediction("b", "5"),
            Prediction {
                status: InferenceStatus::Truncated,
                ..prediction("partial", "0")
            },
            Prediction {
                status: InferenceStatus::Unsupported,
                ..prediction("unsupported", "0")
            },
            Prediction {
                status: InferenceStatus::Error,
                ..prediction("failed", "0")
            },
        ];
        let report = score_paired(&set, &a, &b)?;
        assert_eq!(report.incumbent.planned, 6);
        assert_eq!(report.incumbent.statuses[&OutcomeStatus::Missing], 3);
        assert_eq!(report.challenger.statuses[&OutcomeStatus::Truncated], 1);
        assert_eq!(report.challenger.statuses[&OutcomeStatus::Unsupported], 1);
        assert_eq!(report.challenger.statuses[&OutcomeStatus::Error], 1);
        assert_eq!(report.challenger.canonical_format, 3);
        assert_eq!(report.strict.gained, 1);
        assert_eq!(report.strict.lost, 1);
        assert_eq!(report.strict.both_correct, 1);
        assert_eq!(report.strict.both_incorrect, 3);
        assert_eq!(report.semantic.lost, 1);
        assert_eq!(report.invariant.incumbent.semantic_joint_correct, 1);
        assert_eq!(report.invariant.strict.gained, 1);
        assert_eq!(report.answer_changing.challenger.relation_observed, 1);
        assert_eq!(report.answer_changing.challenger.semantic_joint_correct, 0);
        assert_eq!(report.answer_changing.strict.lost, 1);
        assert_eq!(report.cases[0].id, "a");
        assert_eq!(report.slices["canonical"].incumbent.strict_correct, 2);
        assert_eq!(report.slices["surface"].strict.gained, 1);
        Ok(())
    }

    #[test]
    fn frozen_cases_reject_duplicate_unknown_and_changed_run_inputs() -> Result<()> {
        let mut set = EvaluationSet {
            budget: BUDGET,
            cases: vec![case("id", 1, "canonical"), case("other", 2, "canonical")],
            pairs: Vec::new(),
        };
        let mut a = run(&set, 'a', "cpu")?;
        let mut b = run(&set, 'b', "cpu")?;
        a.predictions = vec![prediction("id", "1"), prediction("id", "2")];
        assert!(score_paired(&set, &a, &b).is_err());
        a.predictions = vec![prediction("unknown", "1")];
        assert!(score_paired(&set, &a, &b).is_err());
        a.predictions.clear();
        b.budget.max_output_tokens += 1;
        assert!(score_paired(&set, &a, &b).is_err());
        b.budget = BUDGET;
        assert!(score_paired(&set, &a, &b).is_ok());
        set.cases[0].prompt = Some("changed prompt".to_owned());
        assert!(score_paired(&set, &a, &b).is_err());
        set.pairs = vec![pair("id", "other", PairKind::Invariant)];
        assert!(set.sha256().is_err());
        set.pairs = vec![pair("id", "unknown", PairKind::AnswerChanging)];
        assert!(set.sha256().is_err());
        set.pairs = vec![
            pair("id", "other", PairKind::AnswerChanging),
            pair("other", "id", PairKind::AnswerChanging),
        ];
        assert!(set.sha256().is_err());
        set.pairs.clear();
        set.cases[1].id = "id".to_owned();
        assert!(set.sha256().is_err());
        Ok(())
    }

    #[test]
    fn the_runtime_matrix_finds_raw_drift_even_if_grades_hold() -> Result<()> {
        let set = EvaluationSet {
            budget: BUDGET,
            cases: vec![case("x", 3, "canonical"), case("missing", 8, "canonical")],
            pairs: Vec::new(),
        };
        let mut matrix = RuntimeMatrix {
            incumbent_canonical: run(&set, 'a', "uncached")?,
            challenger_canonical: run(&set, 'b', "uncached")?,
            incumbent_staged: run(&set, 'a', "cached")?,
            challenger_staged: run(&set, 'b', "cached")?,
        };
        matrix.incumbent_canonical.predictions = vec![prediction("x", "4")];
        matrix.incumbent_staged.predictions = vec![prediction("x", "5")];
        matrix.challenger_canonical.predictions = vec![prediction("x", "3")];
        matrix.challenger_staged.predictions = vec![prediction("x", "3")];
        let report = score_runtime(&set, &matrix)?;
        assert_eq!(report.canonical.strict.gained, 1);
        assert_eq!(report.staged.strict.gained, 1);
        assert_eq!(report.incumbent_runtime_change.raw_changed, ["x"]);
        assert_eq!(report.challenger_runtime_change.raw_equal, 1);
        assert_eq!(report.challenger_runtime_change.both_completed, 1);
        assert_eq!(report.challenger_runtime_change.missing, ["missing"]);
        matrix.challenger_staged = matrix.incumbent_staged.clone();
        assert!(score_runtime(&set, &matrix).is_err());
        Ok(())
    }

    #[test]
    fn choice_position_reports_distinguish_correct_answers_and_fixed_slots() -> Result<()> {
        let item = ChoiceCase {
            id: "item".to_owned(),
            question: "Which object is a fruit?".to_owned(),
            options: ["desk", "apple", "window", "plate"].map(str::to_owned),
            correct: 1,
            source: Some("fixture".to_owned()),
        };
        let layouts = rotate_choices(&item)?;
        for display in 0..4 {
            let mut original = layouts
                .each_ref()
                .map(|layout| layout.displayed_to_original[display]);
            original.sort_unstable();
            assert_eq!(original, [0, 1, 2, 3]);
        }
        let set = choice_set(std::slice::from_ref(&item), BUDGET)?;
        let mut run = run(&set, 'a', "cpu")?;
        run.predictions = layouts
            .iter()
            .map(|layout| prediction(&layout.id, "A"))
            .collect();
        let biased = score_choice_positions(std::slice::from_ref(&item), &run)?;
        assert_eq!(biased.all_layouts_parsed, 1);
        assert_eq!(biased.same_original_option, 0);
        assert_eq!(biased.all_strict_correct, 0);
        assert_eq!(biased.by_predicted_slot[&'A'], 4);
        run.predictions = layouts
            .iter()
            .map(|layout| prediction(&layout.id, &layout.answer.to_string()))
            .collect();
        let correct = score_choice_positions(std::slice::from_ref(&item), &run)?;
        assert_eq!(correct.all_strict_correct, 1);
        assert_eq!(correct.same_original_option, 1);
        for label in CHOICE_LABELS {
            assert_eq!(correct.by_gold_slot[&label].strict_correct, 1);
        }
        run.predictions = vec![prediction(&layouts[0].id, "H")];
        let out_of_range = score_choice_positions(std::slice::from_ref(&item), &run)?;
        assert_eq!(
            out_of_range.by_gold_slot[&layouts[0].answer].statuses[&OutcomeStatus::Invalid],
            1
        );
        run.predictions.clear();
        let unavailable = score_choice_positions(&[item], &run)?;
        assert_eq!(unavailable.same_original_option, 0);
        assert_eq!(
            unavailable.by_gold_slot[&'A'].statuses[&OutcomeStatus::Missing],
            1
        );
        Ok(())
    }

    #[test]
    fn training_majority_uses_fixed_tie_break_and_checks_domain() -> Result<()> {
        let answers = [2, 1, 1, 2, 3].map(Answer::Integer);
        assert_eq!(train_majority(&answers)?, Answer::Integer(1));
        assert!(train_majority(&[Answer::Integer(1), Answer::Choice('A')]).is_err());
        assert!(train_majority(&[]).is_err());
        Ok(())
    }

    #[test]
    fn first_draw_readiness_requires_usable_wave_coverage() -> Result<()> {
        let set = EvaluationSet {
            budget: BUDGET,
            cases: vec![
                case("wrong", 0, "fast"),
                case("empty", 0, "fast"),
                case("truncated", 0, "fast"),
                case("pending", 0, "slow"),
                case("timeout", 0, "slow"),
                case("error", 0, "slow"),
                case("missing", 0, "slow"),
            ],
            pairs: Vec::new(),
        };
        let mut run = run(&set, 'a', "cpu")?;
        run.predictions = vec![prediction("wrong", "9"), prediction("empty", "")];
        for (id, status) in [
            ("pending", InferenceStatus::Pending),
            ("timeout", InferenceStatus::Timeout),
            ("error", InferenceStatus::Error),
            ("truncated", InferenceStatus::Truncated),
        ] {
            run.predictions.push(Prediction {
                id: id.into(),
                response: None,
                status,
            });
        }
        let report = score_first_draws(&set, &run)?;
        assert_eq!(report.totals.scheduled, 7);
        assert_eq!(report.totals.observed, 3);
        assert_eq!(report.totals.usable, 1);
        for status in [
            OutcomeStatus::Pending,
            OutcomeStatus::Timeout,
            OutcomeStatus::Error,
            OutcomeStatus::Missing,
            OutcomeStatus::Empty,
            OutcomeStatus::Truncated,
        ] {
            assert_eq!(report.totals.statuses[&status], 1);
        }
        assert_eq!(report.slices["fast"].observed, 3);
        assert_eq!(report.slices["slow"].observed, 0);
        assert!(!report.ready_for_extra_draws);
        run.predictions = set
            .cases
            .iter()
            .map(|case| prediction(&case.id, "not an integer"))
            .collect();
        assert!(score_first_draws(&set, &run)?.ready_for_extra_draws);
        run.predictions.push(prediction("wrong", "0"));
        assert!(score_first_draws(&set, &run).is_err());
        Ok(())
    }

    #[test]
    fn early_rank_replays_three_reversals_and_distinguishes_ties() -> Result<()> {
        let cohort_sha256 = "a".repeat(64);
        let mut input = RankAuditInput {
            cohort_sha256: cohort_sha256.clone(),
            development_cases: 297,
            early_step: 60,
            full_step: 120,
            seeds: vec![42, 2026],
            arms: vec![
                "publisher-480".into(),
                "balanced-480".into(),
                "balanced-256".into(),
            ],
            observations: Vec::new(),
        };
        for (arm, early, full) in [
            ("publisher-480", [66, 78], [75, 87]),
            ("balanced-480", [77, 74], [80, 71]),
            ("balanced-256", [74, 78], [68, 74]),
        ] {
            for (step, scores) in [(60, early), (120, full)] {
                for (&seed, correct) in input.seeds.iter().zip(scores) {
                    input.observations.push(RankObservation {
                        arm: arm.into(),
                        seed,
                        step,
                        cohort_sha256: cohort_sha256.clone(),
                        development_cases: 297,
                        correct,
                    });
                }
            }
        }
        input.observations.reverse();
        let report = audit_early_rank(&input)?;
        assert_eq!(
            report.reversed_pairs,
            [
                ["publisher-480", "balanced-480"],
                ["publisher-480", "balanced-256"],
                ["balanced-480", "balanced-256"],
            ]
        );
        assert_eq!(
            report.early_rank,
            [
                vec!["balanced-256"],
                vec!["balanced-480"],
                vec!["publisher-480"]
            ]
        );
        assert_eq!(
            report.full_rank,
            [
                vec!["publisher-480"],
                vec!["balanced-480"],
                vec!["balanced-256"]
            ]
        );
        let contested = input
            .observations
            .iter_mut()
            .find(|row| row.arm == "balanced-256" && row.step == 60 && row.seed == 42)
            .ok_or_else(|| anyhow::anyhow!("missing fixture score"))?;
        contested.correct -= 1;
        let tied = audit_early_rank(&input)?;
        assert_eq!(tied.early_rank[0], ["balanced-256", "balanced-480"]);
        assert_eq!(tied.early_ties, [["balanced-480", "balanced-256"]]);
        assert_eq!(tied.reversed_pairs.len(), 2);
        let mut wrong_cohort = input.clone();
        wrong_cohort.observations[0].cohort_sha256 = "b".repeat(64);
        assert!(audit_early_rank(&wrong_cohort).is_err());
        let mut duplicate = input.clone();
        duplicate.observations.push(input.observations[0].clone());
        assert!(audit_early_rank(&duplicate).is_err());
        input.observations.remove(0);
        assert!(audit_early_rank(&input).is_err());
        Ok(())
    }
}
