//! Bounded Nina development experiments over independently checked arithmetic.
//!
//! The two fresh fits per seed see the same prompts, answer byte lengths,
//! label histogram, mask, initialization, and draw order. The reference arm
//! changes every target while retaining that budget. All cases and scoring
//! rules are recorded before any fit, including alternate phrasing and the
//! answer-changing pairs. Each attempt has a durable log and development
//! decisions carry no authority over the serving registry.

use anyhow::{ensure, Context, Result};
use burn::tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use crate::checkpoint::{self, file_sha256_hex, TrainState};
use crate::corpus::{mask_path, TokenCorpus};
use crate::hybrid::PositionKind;
use crate::lm::{LanguageModel, LmConfig, Sampling};
use crate::nina::ChainedLog;
use crate::nina_eval::{
    self, Answer, EvaluationBudget, EvaluationCase, EvaluationPair, EvaluationSet, InferenceStatus,
    PairKind, PairedReport, Prediction, PredictionRun, RuntimeMatrix,
};
use crate::tokenizer::{ByteTokenizer, Special};
use crate::train::{self, DefaultTrainBackend, LmTrainConfig};

mod validation;
pub use validation::{long_record_validation, science_screen};

const SEEDS: [u64; 2] = [42, 2026];
const ARITHMETIC_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Operation {
    Plus,
    Minus,
    Times,
    #[serde(rename = "divided by")]
    DividedBy,
}

impl Operation {
    fn symbol(self) -> char {
        match self {
            Self::Plus => '+',
            Self::Minus => '-',
            Self::Times => '*',
            Self::DividedBy => '/',
        }
    }
}

#[derive(Debug, Deserialize)]
struct Problem {
    id: String,
    operation: Operation,
    left: i64,
    right: i64,
    expected: i64,
}

type Family = (char, i64, i64, i64);

impl Problem {
    fn family(&self) -> Result<Family> {
        let (a, b, op, value) = match self.operation {
            Operation::Plus => (
                self.left,
                self.right,
                '+',
                self.left.checked_add(self.right),
            ),
            Operation::Minus => (
                self.expected,
                self.right,
                '+',
                self.left.checked_sub(self.right),
            ),
            Operation::Times => (
                self.left,
                self.right,
                '*',
                self.left.checked_mul(self.right),
            ),
            Operation::DividedBy => {
                ensure!(
                    self.left.checked_rem(self.right) == Some(0),
                    "{} needs exact integer division",
                    self.id
                );
                (
                    self.expected,
                    self.right,
                    '*',
                    self.left.checked_div(self.right),
                )
            }
        };
        ensure!(
            value == Some(self.expected),
            "independent answer disagrees for {}",
            self.id
        );
        let result = match self.operation {
            Operation::Plus | Operation::Times => self.expected,
            Operation::Minus | Operation::DividedBy => self.left,
        };
        Ok((op, a.min(b), a.max(b), result))
    }

    fn prompt(&self, variant: bool) -> String {
        let op = self.operation.symbol();
        let (a, b) = (self.left, self.right);
        if variant {
            format!("Result of {a}{op}{b} = ")
        } else {
            format!("{a} {op} {b} = ")
        }
    }
}

#[derive(Deserialize)]
struct ArithmeticCases {
    train: Vec<Problem>,
    dev: Vec<Problem>,
}

impl ArithmeticCases {
    fn validate(&self) -> Result<(usize, usize)> {
        ensure!(
            !self.train.is_empty() && !self.dev.is_empty(),
            "arithmetic needs both train and development cases"
        );
        let mut ids = BTreeSet::new();
        let mut expressions = BTreeSet::new();
        for row in self.train.iter().chain(&self.dev) {
            ensure!(
                !row.id.trim().is_empty() && !row.id.contains('/'),
                "invalid arithmetic ID {:?}",
                row.id
            );
            ensure!(ids.insert(&row.id), "duplicate arithmetic ID {}", row.id);
            ensure!(
                expressions.insert(row.prompt(false)),
                "duplicate arithmetic expression for {}",
                row.id
            );
        }
        let families = |rows: &[Problem]| {
            rows.iter()
                .map(Problem::family)
                .collect::<Result<BTreeSet<_>>>()
        };
        let (train, dev) = (families(&self.train)?, families(&self.dev)?);
        ensure!(
            train.is_disjoint(&dev),
            "arithmetic family leakage across train and development"
        );
        Ok((train.len(), dev.len()))
    }
}

/// Fixed budgets for both arms of a local arithmetic trial.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ArithmeticBudget {
    /// Causal context and invisible spacer between training records, in bytes.
    pub context: usize,
    /// Scheduled microbatches per arm. There is one batch per optimizer update.
    pub steps: usize,
    /// Scheduled windows per batch.
    pub batch_size: usize,
    /// Output cap per greedy inference, including its end marker.
    pub max_output_tokens: usize,
}

/// Read a research input and identify its path on decoding errors.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let input = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(std::io::BufReader::new(input))
        .with_context(|| format!("decode {}", path.display()))
}

pub(crate) fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    crate::nim::write_atomic(path, &bytes)
}

fn create_run_directory(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir(path)
        .with_context(|| format!("create new run directory {}", path.display()))
}

fn scrambled_targets(train: &[Problem]) -> Result<Vec<i64>> {
    let mut bands = BTreeMap::<usize, Vec<usize>>::new();
    for (index, row) in train.iter().enumerate() {
        bands
            .entry(row.expected.to_string().len())
            .or_default()
            .push(index);
    }
    let mut answers: Vec<_> = train.iter().map(|row| row.expected).collect();
    for (width, indices) in bands {
        let mut order = indices;
        order.sort_by_key(|&i| (train[i].expected, &train[i].id));
        let shift = order.len() / 2;
        for (rank, &index) in order.iter().enumerate() {
            let answer = train[order[(rank + shift) % order.len()]].expected;
            ensure!(
                answer != train[index].expected,
                "width {width} has too few distinct labels to change every target"
            );
            answers[index] = answer;
        }
    }
    Ok(answers)
}

fn training_stream(train: &[Problem], answers: &[i64], context: usize) -> (Vec<u16>, Vec<u8>) {
    let tokenizer = ByteTokenizer::new();
    let (mut tokens, mut masks) = (Vec::new(), Vec::new());
    for (row, answer) in train.iter().zip(answers) {
        let prompt = row.prompt(false);
        let target = answer.to_string();
        tokens.extend(tokenizer.encode_document(&format!("{prompt}{target}")));
        tokens.extend(std::iter::repeat_n(Special::Pad.id(), context));
        masks.extend(std::iter::repeat_n(0, prompt.len() + 1));
        masks.extend(std::iter::repeat_n(1, target.len() + 1));
        masks.extend(std::iter::repeat_n(0, context));
    }
    (tokens, masks)
}

fn development_set(dev: &[Problem], max_output_tokens: usize) -> Result<EvaluationSet> {
    let mut cases = Vec::with_capacity(dev.len() * 2);
    let mut pairs = Vec::new();
    for row in dev {
        let family = format!("{:?}", row.family()?);
        let alternate = format!("{}/surface", row.id);
        for (id, surface, slice) in [(&row.id, false, "canonical"), (&alternate, true, "surface")] {
            cases.push(EvaluationCase {
                id: id.clone(),
                answer: Answer::Integer(row.expected),
                family: Some(family.clone()),
                slice: Some(slice.to_owned()),
                prompt: Some(row.prompt(surface)),
            });
        }
        pairs.push(EvaluationPair {
            left: row.id.clone(),
            right: alternate,
            kind: PairKind::Invariant,
        });
    }
    let mut ordered: Vec<_> = dev.iter().collect();
    ordered.sort_by_key(|row| &row.id);
    let mut used = BTreeSet::new();
    for (i, a) in ordered.iter().enumerate() {
        if used.contains(&a.id) {
            continue;
        }
        if let Some(b) = ordered[(i + 1)..].iter().find(|b| {
            !used.contains(&b.id)
                && a.operation == b.operation
                && a.expected != b.expected
                && ((a.left == b.left && a.right.abs_diff(b.right) == 1)
                    || (a.right == b.right && a.left.abs_diff(b.left) == 1))
        }) {
            pairs.push(EvaluationPair {
                left: a.id.clone(),
                right: b.id.clone(),
                kind: PairKind::AnswerChanging,
            });
            used.extend([&a.id, &b.id]);
        }
    }
    ensure!(used.len() >= 2, "development needs an answer-changing pair");
    Ok(EvaluationSet {
        budget: EvaluationBudget { max_output_tokens },
        cases,
        pairs,
    })
}

type Cpu = burn_ndarray::NdArray<f32>;

fn run_predictions(
    model: &LanguageModel<Cpu>,
    set: &EvaluationSet,
    checkpoint_sha256: &str,
    cached: bool,
    device: &burn_ndarray::NdArrayDevice,
) -> Result<PredictionRun> {
    let tokenizer = ByteTokenizer::new();
    let mut rng = StdRng::seed_from_u64(0);
    let mut predictions = Vec::with_capacity(set.cases.len());
    for case in &set.cases {
        let prompt = case
            .prompt
            .as_ref()
            .with_context(|| format!("missing prompt for {}", case.id))?;
        let mut input = vec![Special::Bos.id()];
        input.extend(tokenizer.encode(prompt));
        let limit = set.budget.max_output_tokens;
        let output = if cached {
            model.generate_cached(&input, limit, &Sampling::Greedy, &mut rng, device)
        } else {
            model.generate(&input, limit, &Sampling::Greedy, &mut rng, device)
        };
        let new = output
            .strip_prefix(input.as_slice())
            .with_context(|| format!("readout changed input for {}", case.id))?;
        let eos = [Special::Eos.id()];
        let completed = new.strip_suffix(&eos);
        let text = completed.unwrap_or(new);
        let decoded = if text.iter().any(|&token| tokenizer.is_special(token)) {
            None
        } else {
            tokenizer.decode(text)
        };
        let status = match (&decoded, completed) {
            (None, _) => InferenceStatus::Error,
            (Some(_), None) => InferenceStatus::Truncated,
            (Some(_), Some(_)) => InferenceStatus::Completed,
        };
        predictions.push(Prediction {
            id: case.id.clone(),
            response: decoded,
            status,
        });
    }
    let runtime = if cached {
        "cpu-kv-cache"
    } else {
        "cpu-full-recompute"
    };
    Ok(PredictionRun {
        checkpoint_sha256: checkpoint_sha256.to_owned(),
        runtime: runtime.to_owned(),
        budget: set.budget,
        predictions,
        protocol_sha256: Some(set.sha256()?),
    })
}

struct ArmRun {
    canonical: PredictionRun,
    cached: PredictionRun,
    wall_secs: f64,
    steps_taken: usize,
    targets_seen: usize,
}

fn arithmetic_model(context: usize) -> LmConfig {
    LmConfig {
        context,
        hidden_size: 32,
        num_layers: 2,
        num_heads: 4,
        intermediate_size: 64,
        num_blocks: 1,
        positions: PositionKind::Rotary,
        ..LmConfig::default()
    }
}

const LEARNING_RATE: f64 = 0.002;
const CLIP_NORM: f32 = 1.0;

fn fit_arm(
    out: &Path,
    arm: &str,
    seed: u64,
    budget: ArithmeticBudget,
    set: &EvaluationSet,
) -> Result<ArmRun> {
    let start = Instant::now();
    let dir = out.join(format!("{arm}-{seed}"));
    std::fs::create_dir(&dir)?;
    let path = out.join(format!("{arm}.bin"));
    let mut corpus = TokenCorpus::in_memory(&path)?;
    corpus.open_mask()?;
    let device = Default::default();
    DefaultTrainBackend::seed(&device, seed);
    let model_config = arithmetic_model(budget.context);
    let config = LmTrainConfig {
        steps: budget.steps,
        batch_size: budget.batch_size,
        lr: LEARNING_RATE,
        clip_norm: CLIP_NORM,
        log_every: 24,
        log_path: Some(dir.join("fit.jsonl")),
        seed,
        out_dir: Some(dir.clone()),
        model_config: Some(model_config.clone()),
        ..LmTrainConfig::default()
    };
    let model = LanguageModel::<DefaultTrainBackend>::new(&model_config, &device)?;
    let (_, report) = train::train_lm(model, &mut corpus, &config, &device)?;
    let checkpoint = report
        .checkpoint
        .context("training did not save a checkpoint")?;
    let state = TrainState::for_model(&checkpoint)?.context("checkpoint sidecar missing")?;
    state.verify_files(&TrainState::dir_for(&checkpoint))?;
    ensure!(
        state.step == budget.steps,
        "checkpoint did not reach the frozen budget"
    );
    let fit_secs = start.elapsed().as_secs_f64();
    let model = checkpoint::load::<Cpu, _>(
        LanguageModel::new(&model_config, &device)?,
        &checkpoint,
        &device,
    )?;
    let canonical = run_predictions(&model, set, &state.model.sha256, false, &device)?;
    let cached = run_predictions(&model, set, &state.model.sha256, true, &device)?;
    let wall_secs = start.elapsed().as_secs_f64();
    write_json(&dir.join("canonical.json"), &canonical)?;
    write_json(&dir.join("cached.json"), &cached)?;
    write_json(
        &dir.join("fit.json"),
        &json!({
            "checkpoint_sha256":state.model.sha256,"state_step":state.step,"steps_taken":report.steps_taken,
            "steps_skipped":report.steps_skipped,"targets_seen":report.tokens_seen,
            "fit_and_integrity_secs":fit_secs,"training_secs":report.elapsed_secs,
            "inference_secs":wall_secs-fit_secs,"wall_secs":wall_secs,
            "model":model_config,"clip_norm":config.clip_norm,"learning_rate":config.lr,
        }),
    )?;
    Ok(ArmRun {
        canonical,
        cached,
        wall_secs,
        steps_taken: report.steps_taken,
        targets_seen: report.tokens_seen,
    })
}

fn arithmetic_qualifies(report: &PairedReport, majority_correct: usize) -> Result<bool> {
    let canonical = report
        .slices
        .get("canonical")
        .context("canonical slice missing")?;
    let surface = report
        .slices
        .get("surface")
        .context("alternate phrasing slice missing")?;
    Ok(canonical.challenger.semantic_correct
        > canonical.incumbent.semantic_correct.max(majority_correct)
        && surface.challenger.semantic_correct > surface.incumbent.semantic_correct
        && report.answer_changing.challenger.semantic_joint_correct > 0)
}

/// Build the recorded arithmetic controls and optionally fit both arms per seed.
/// The input JSON has `train` and `dev` arrays of `id`, `operation`, `left`,
/// `right`, and `expected`. Operation names are `plus`, `minus`, `times`, and
/// `divided by`. The output path must be new. `protocol.json` is written before
/// the first fit and the original input file is read-only.
pub fn arithmetic(
    cases: &Path,
    out: &Path,
    budget: ArithmeticBudget,
    prepare_only: bool,
) -> Result<Value> {
    let started = Instant::now();
    ensure!(
        budget.steps > 0 && budget.batch_size > 0,
        "steps and batch size must be positive"
    );
    ensure!(
        budget.max_output_tokens > 1 && budget.max_output_tokens < budget.context,
        "output cap must fit the context with room for the prompt"
    );
    let draws = budget
        .steps
        .checked_mul(budget.batch_size)
        .context("arithmetic draw budget overflows")?;
    let input = std::fs::read(cases).with_context(|| format!("read {}", cases.display()))?;
    let cases: ArithmeticCases =
        serde_json::from_slice(&input).context("decode the arithmetic cases")?;
    let (train_families, development_families) = cases.validate()?;
    ensure!(
        nina_eval::grade_fixtures()
            .iter()
            .all(|fixture| fixture.matches_fixture),
        "numeric grader failed its qualification fixtures"
    );
    for row in &cases.train {
        ensure!(
            row.prompt(false).len() + row.expected.to_string().len() + 2 <= budget.context,
            "training row {} exceeds context",
            row.id
        );
    }
    for row in &cases.dev {
        ensure!(
            row.prompt(true).len() < budget.context - budget.max_output_tokens,
            "development row {} exceeds context",
            row.id
        );
    }
    let set = development_set(&cases.dev, budget.max_output_tokens)?;
    let solver: Vec<_> = cases.train.iter().map(|row| row.expected).collect();
    let scrambled = scrambled_targets(&cases.train)?;
    let majority = nina_eval::train_majority(
        &solver
            .iter()
            .copied()
            .map(Answer::Integer)
            .collect::<Vec<_>>(),
    )?;
    let Answer::Integer(majority) = majority else {
        anyhow::bail!("arithmetic majority must be an integer");
    };

    create_run_directory(out)?;
    let mut assets = BTreeMap::new();
    let mut target_mask = Vec::new();
    for (arm, answers) in [("solver", &solver), ("scrambled", &scrambled)] {
        let (tokens, mask) = training_stream(&cases.train, answers, budget.context);
        let path = out.join(format!("{arm}.bin"));
        TokenCorpus::write(&path, &tokens)?;
        TokenCorpus::write_masks(&mask_path(&path), &mask)?;
        assets.insert(arm, json!({"token_sha256":file_sha256_hex(&path)?,"mask_sha256":file_sha256_hex(&mask_path(&path))?}));
        if target_mask.is_empty() {
            target_mask = mask;
        } else {
            ensure!(
                mask == target_mask,
                "matched arms have different target positions"
            );
        }
    }
    let majority_correct = cases
        .dev
        .iter()
        .filter(|row| row.expected == majority)
        .count();
    let order = [["solver", "scrambled"], ["scrambled", "solver"]];
    let protocol = json!({
        "version":ARITHMETIC_VERSION,"scope":"development", "budget":budget, "seeds":SEEDS,"arm_order":order,
        "cases_sha256":format!("{:x}", Sha256::digest(input)),
        "evaluation_sha256":set.sha256()?,"runtime_sha256":file_sha256_hex(&std::env::current_exe()?)?,
        "build":checkpoint::BuildInfo::current(), "model":arithmetic_model(budget.context),
        "learning_rate":LEARNING_RATE,"lr_schedule":"constant","clip_norm":CLIP_NORM,
        "train_cases":cases.train.len(),"train_families":train_families,
        "development_cases":cases.dev.len(),"development_families":development_families,
        "scrambled_labels_changed":scrambled.len(),"scheduled_draws_per_arm":draws,
        "training_target_bytes_including_eos":target_mask.iter().filter(|&&v| v==1).count(),
        "train_majority":majority,"train_majority_correct":majority_correct,"assets":assets,
        "advance":"Both seeds beat the matched control and train majority on canonical questions, beat the control on alternate phrasing, and solve both members of an answer-changing pair. The capped response must complete.",
    });
    write_json(&out.join("development.json"), &set)?;
    let protocol_path = out.join("protocol.json");
    write_json(&protocol_path, &protocol)?;
    if prepare_only {
        return Ok(protocol);
    }

    let mut attempts = ChainedLog::open(out.join("attempts.jsonl"))?;
    let mut rows = BTreeMap::new();
    let mut qualifies = true;
    for (seed, arms) in SEEDS.into_iter().zip(order) {
        let mut trained = BTreeMap::new();
        for arm in arms {
            let attempt = Instant::now();
            attempts.append(&json!({"arm":arm,"seed":seed,"status":"started"}))?;
            match fit_arm(out, arm, seed, budget, &set) {
                Ok(run) => {
                    attempts.append(&json!({"arm":arm,"seed":seed,"status":"completed","wall_secs":run.wall_secs}))?;
                    trained.insert(arm, run);
                }
                Err(error) => {
                    attempts.append(&json!({"arm":arm,"seed":seed,"status":"error","wall_secs":attempt.elapsed().as_secs_f64(),"error":format!("{error:#}")}))?;
                    return Err(error);
                }
            }
        }
        let control = trained
            .remove("scrambled")
            .context("scrambled arm missing")?;
        let candidate = trained.remove("solver").context("solver arm missing")?;
        let matched_budget = control.steps_taken == candidate.steps_taken
            && control.targets_seen == candidate.targets_seen;
        let matrix = RuntimeMatrix {
            incumbent_canonical: control.canonical,
            challenger_canonical: candidate.canonical,
            incumbent_staged: control.cached,
            challenger_staged: candidate.cached,
        };
        let runtime = nina_eval::score_runtime(&set, &matrix)?;
        let passed = matched_budget && arithmetic_qualifies(&runtime.canonical, majority_correct)?;
        qualifies &= passed;
        write_json(&out.join(format!("paired-{seed}.json")), &runtime.canonical)?;
        write_json(&out.join(format!("runtime-{seed}.json")), &runtime)?;
        rows.insert(seed, json!({"qualified":passed,"matched_budget":matched_budget,"paired":format!("paired-{seed}.json"),"runtime":format!("runtime-{seed}.json")}));
    }
    let results = json!({
        "protocol_sha256":file_sha256_hex(&protocol_path)?,"development_qualified":qualifies,
        "confirmation_scored":false,"seeds":rows,"elapsed_secs":started.elapsed().as_secs_f64(),
    });
    write_json(&out.join("results.json"), &results)?;
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archived_arithmetic_masks_and_families_rebuild() -> Result<()> {
        let cases: ArithmeticCases = serde_json::from_str(include_str!(
            "../experiments/nina_research/arithmetic-cases.json"
        ))?;
        assert_eq!(cases.validate()?, (330, 78));
        let gold: Vec<_> = cases.train.iter().map(|row| row.expected).collect();
        let scrambled = scrambled_targets(&cases.train)?;
        let (direct, direct_mask) = training_stream(&cases.train, &gold, 64);
        let (control, control_mask) = training_stream(&cases.train, &scrambled, 64);
        let digest = |bytes: &[u16]| {
            let tokens: Vec<_> = bytes.iter().flat_map(|value| value.to_le_bytes()).collect();
            format!("{:x}", Sha256::digest(tokens))
        };
        assert_eq!(direct_mask, control_mask);
        assert_eq!(
            format!("{:x}", Sha256::digest(direct_mask)),
            "5e03e0fcc8d894d36824ee2f5ebb9a4ad7e86c2a7c86f6cfbdbab65ffbb18c37"
        );
        assert_eq!(
            digest(&direct),
            "86714f9a839e5d43ab837dd419e372ddba05fe878e8f8908e467a443686816a7"
        );
        assert_eq!(
            digest(&control),
            "fb0cc287b077a80374b004ed5265b172982e972e2fe5b52059293a78194e1c6b"
        );
        let dev = development_set(&cases.dev, 6)?;
        assert_eq!(dev.cases.len(), 192);
        assert_eq!(
            dev.pairs
                .iter()
                .filter(|pair| pair.kind == PairKind::AnswerChanging)
                .count(),
            18
        );
        Ok(())
    }

    #[test]
    fn inverse_arithmetic_in_the_other_split_is_leakage() -> Result<()> {
        let cases: ArithmeticCases = serde_json::from_value(json!({
            "train":[{"id":"sum","operation":"plus","left":7,"right":11,"expected":18}],
            "dev":[{"id":"inverse","operation":"minus","left":18,"right":7,"expected":11}],
        }))?;
        let error = cases.validate().err().context("equivalent family passed")?;
        assert!(error.to_string().contains("family leakage"));
        Ok(())
    }
}
