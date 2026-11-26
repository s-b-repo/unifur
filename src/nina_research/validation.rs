use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

use super::{create_run_directory, read_json, write_json};
use crate::checkpoint::file_sha256_hex;
use crate::nina_eval::{
    choice_set, Answer, ChoiceCase, EvaluationBudget, EvaluationCase, EvaluationSet,
};

const SCIENCE_SEED: &str = "nina-layout-diagnostic-v1";
const LONG_RECORD_SEED: &str = "nina-long-validation-screen-v1";

fn rank(seed: &str, id: &str) -> [u8; 32] {
    Sha256::digest(format!("{seed}/{id}")).into()
}

#[derive(Deserialize)]
struct ScienceRow {
    id: String,
    question: String,
    options: Vec<String>,
    correct: usize,
    source: String,
}

impl ScienceRow {
    fn evaluation_case(&self) -> Result<EvaluationCase> {
        ensure!(
            !self.id.trim().is_empty() && !self.question.trim().is_empty(),
            "science case has an empty ID or question"
        );
        ensure!(
            (2..=8).contains(&self.options.len()) && self.correct < self.options.len(),
            "science case {} has invalid options or gold label",
            self.id
        );
        ensure!(
            self.options.iter().all(|text| !text.trim().is_empty()),
            "science case {} has a blank option",
            self.id
        );
        let labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
        let mut prompt = format!("Question:\n{}\n", self.question);
        for (&label, text) in labels.iter().zip(&self.options) {
            prompt.push_str(&format!("{label}. {text}\n"));
        }
        prompt.push_str("Answer:\n");
        Ok(EvaluationCase {
            id: self.id.clone(),
            answer: Answer::Choice(labels[self.correct]),
            prompt: Some(prompt),
            family: Some(self.id.clone()),
            slice: Some(self.source.clone()),
        })
    }

    fn longest_is_correct(&self) -> bool {
        self.options
            .iter()
            .enumerate()
            .max_by_key(|&(index, option)| (option.chars().count(), std::cmp::Reverse(index)))
            .is_some_and(|(index, _)| index == self.correct)
    }
}

/// Prepare the exact-fit science screen, full development and a balanced
/// four-position diagnostic. The input JSONL has `id`, `question`, `options`,
/// `correct` (zero based), and `source`. Prompts are kept byte-intact.
pub fn science_screen(
    input: &Path,
    out: &Path,
    context: usize,
    diagnostic_per_slice: usize,
) -> Result<Value> {
    ensure!(
        context >= 3 && diagnostic_per_slice > 0,
        "science screen needs at least three context positions and a positive diagnostic size"
    );
    let contents =
        std::fs::read_to_string(input).with_context(|| format!("read {}", input.display()))?;
    let input_digest = format!("{:x}", Sha256::digest(contents.as_bytes()));
    let rows: Vec<ScienceRow> = contents
        .lines()
        .enumerate()
        .map(|(line, text)| {
            serde_json::from_str(text)
                .with_context(|| format!("{} line {}", input.display(), line + 1))
        })
        .collect::<Result<_>>()?;
    let longest_full = rows.iter().filter(|row| row.longest_is_correct()).count();
    let candidates = rows
        .iter()
        .map(ScienceRow::evaluation_case)
        .collect::<Result<Vec<_>>>()?;
    let budget = EvaluationBudget {
        max_output_tokens: 2,
    };
    let case_set = |cases| EvaluationSet {
        budget,
        cases,
        pairs: Vec::new(),
    };
    let mut layouts = [Vec::new(), Vec::new()];
    let mut compact = Vec::new();
    let mut full = Vec::with_capacity(rows.len());
    let mut longest_compact = 0;
    for (row, mut case) in rows.into_iter().zip(candidates) {
        let width = case
            .prompt
            .as_ref()
            .context("science case missing its prompt")?
            .len()
            + 3;
        let fits = width <= context;
        case.slice = Some(if fits { "exact_fit" } else { "over_width" }.to_owned());
        if fits {
            longest_compact += usize::from(row.longest_is_correct());
            compact.push(case.clone());
        }
        full.push(case);
        if let Ok(options) = <[String; 4]>::try_from(row.options) {
            let choice = ChoiceCase {
                id: row.id,
                question: row.question,
                options,
                correct: row.correct,
                source: Some(row.source),
            };
            layouts[usize::from(!fits)].push(choice);
        }
    }
    let (compact, full) = (case_set(compact), case_set(full));
    let mut diagnostic = Vec::new();
    for group in &mut layouts {
        ensure!(group.len() >= diagnostic_per_slice, "available four-choice cases ({}) cannot fill diagnostic slice ({diagnostic_per_slice})", group.len());
        group.sort_by_cached_key(|row| rank(SCIENCE_SEED, &row.id));
        diagnostic.extend(group.drain(..diagnostic_per_slice));
    }
    let layout_set = choice_set(&diagnostic, budget)?;
    let report = json!({
        "scope":"historical_development", "source_sha256":input_digest,"context":context,
        "full_cases":full.cases.len(),"exact_fit_cases":compact.cases.len(),
        "over_width_cases":full.cases.len()-compact.cases.len(),
        "longest_option_correct":{"full":longest_full,"exact_fit":longest_compact},
        "first_option_correct":full.cases.iter().filter(|case|case.answer==Answer::Choice('A')).count(),
        "diagnostic_seed":SCIENCE_SEED,"diagnostic_per_slice":diagnostic_per_slice,
        "four_position_case_ids":diagnostic.iter().map(|row| &row.id).collect::<Vec<_>>(),
        "full_sha256":full.sha256()?,"exact_fit_sha256":compact.sha256()?,"four_position_sha256":layout_set.sha256()?,
    });
    create_run_directory(out)?;
    write_json(&out.join("full.json"), &full)?;
    write_json(&out.join("exact-fit.json"), &compact)?;
    write_json(&out.join("position-cases.json"), &diagnostic)?;
    write_json(&out.join("positions.json"), &layout_set)?;
    write_json(&out.join("manifest.json"), &report)?;
    Ok(report)
}

#[derive(Deserialize)]
struct PackedIds {
    train: Vec<String>,
    validation: Vec<String>,
}

#[derive(Deserialize)]
struct LongManifest {
    packed_ids: PackedIds,
    prompt_width_bytes: usize,
    response_width_bytes_including_eos: usize,
    corpus_hashes: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct TrainingRow<'a> {
    id: &'a str,
    question: String,
}

#[derive(Deserialize)]
struct ValidationRow {
    id: String,
    question: String,
    field: String,
    answer: String,
}

fn stem(question: &str) -> [u8; 32] {
    let text: String = question.nfkc().flat_map(char::to_lowercase).collect();
    let mut hash = Sha256::new();
    for word in text
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|word| !word.is_empty())
    {
        hash.update(word);
        hash.update(b" ");
    }
    hash.finalize().into()
}

struct TrainingStems {
    packed: HashSet<[u8; 32]>,
    raw: HashSet<[u8; 32]>,
    rows: usize,
    sha256: String,
}

fn read_training(path: &Path, mut expected: HashSet<&str>) -> Result<TrainingStems> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut input = BufReader::new(file);
    let mut hash = Sha256::new();
    let mut bytes = Vec::new();
    let (mut packed, mut raw, mut rows) = (HashSet::new(), HashSet::new(), 0);
    while input.read_until(b'\n', &mut bytes)? != 0 {
        rows += 1;
        let row: TrainingRow<'_> = serde_json::from_slice(&bytes)
            .with_context(|| format!("{} line {rows}", path.display()))?;
        let key = stem(&row.question);
        raw.insert(key);
        if expected.remove(row.id) {
            packed.insert(key);
        }
        hash.update(&bytes);
        bytes.clear();
    }
    ensure!(
        expected.is_empty(),
        "{} packed training IDs are missing from the source",
        expected.len()
    );
    Ok(TrainingStems {
        packed,
        raw,
        rows,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn read_verified(dir: &Path, name: &str, expected: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let bytes = std::fs::read(dir.join(name)).with_context(|| format!("read {name}"))?;
    let pinned = expected
        .get(name)
        .with_context(|| format!("manifest has no digest for {name}"))?;
    ensure!(
        &format!("{:x}", Sha256::digest(&bytes)) == pinned,
        "{name} differs from the source manifest"
    );
    Ok(bytes)
}

fn answer_band(bytes: usize) -> &'static str {
    match bytes {
        0..=64 => "1–64",
        65..=256 => "65–256",
        _ => "257+",
    }
}

fn stratified_screen(
    pools: BTreeMap<(&str, &str), Vec<usize>>,
    rows: &[ValidationRow],
    size: usize,
) -> Vec<usize> {
    let mut groups: Vec<VecDeque<_>> = pools
        .into_values()
        .map(|mut indices| {
            indices.sort_by_cached_key(|&index| rank(LONG_RECORD_SEED, &rows[index].id));
            indices.into()
        })
        .collect();
    let mut result = Vec::with_capacity(size);
    while result.len() < size {
        for group in &mut groups {
            if let Some(index) = group.pop_front() {
                result.push(index);
                if result.len() == size {
                    break;
                }
            }
        }
    }
    result.sort_unstable();
    result
}

#[derive(Serialize)]
struct Excluded<'a> {
    id: &'a str,
    index: usize,
    field: &'a str,
    reason: &'static str,
}

struct Views<'a> {
    rows: &'a [ValidationRow],
    metadata: Vec<&'a str>,
    aligned: &'a [u8],
    codec: &'a [u8],
    aligned_stride: usize,
    target_stride: usize,
}

impl Views<'_> {
    fn write(&self, dir: &Path, indices: &[usize]) -> Result<Value> {
        std::fs::create_dir(dir)?;
        let mut full = Vec::new();
        let mut targets = Vec::new();
        let mut lines = String::new();
        let mut counts = BTreeMap::new();
        for &index in indices {
            let (a, b) = (index * self.aligned_stride, index * self.target_stride);
            full.extend_from_slice(&self.aligned[a..a + self.aligned_stride]);
            targets.extend_from_slice(&self.codec[b..b + self.target_stride]);
            lines.push_str(self.metadata[index]);
            lines.push('\n');
            let row = &self.rows[index];
            *counts
                .entry(format!("{}/{}", row.field, answer_band(row.answer.len())))
                .or_insert(0usize) += 1;
        }
        let mut hashes = BTreeMap::new();
        for (name, bytes) in [
            ("validation.aligned.bin", full.as_slice()),
            ("validation.codec.bin", targets.as_slice()),
            ("validation.jsonl", lines.as_bytes()),
        ] {
            crate::nim::write_atomic(&dir.join(name), bytes)?;
            hashes.insert(name, format!("{:x}", Sha256::digest(bytes)));
        }
        write_json(&dir.join("indices.json"), &indices)?;
        hashes.insert("indices.json", file_sha256_hex(&dir.join("indices.json"))?);
        Ok(json!({"cases":indices.len(),"field_and_target_byte_band":counts,"data_files":hashes}))
    }
}

/// Rebuild clean and small validation views from an existing long-record
/// pack. Removes every validation stem seen in raw training and verifies
/// the original row alignment, targets, and binary digests before copying.
pub fn long_record_validation(source: &Path, out: &Path, size: usize) -> Result<Value> {
    ensure!(size > 0, "screen size must be positive");
    let manifest_path = source.join("manifest.json");
    let manifest_digest = file_sha256_hex(&manifest_path)?;
    let manifest: LongManifest = read_json(&manifest_path)?;
    ensure!(
        manifest_digest == file_sha256_hex(&manifest_path)?,
        "pack manifest changed during preparation"
    );
    let train: HashSet<_> = manifest
        .packed_ids
        .train
        .iter()
        .map(String::as_str)
        .collect();
    let valid: HashSet<_> = manifest
        .packed_ids
        .validation
        .iter()
        .map(String::as_str)
        .collect();
    ensure!(
        train.len() == manifest.packed_ids.train.len()
            && valid.len() == manifest.packed_ids.validation.len()
            && train.is_disjoint(&valid),
        "duplicate or overlapping packed row IDs"
    );
    let train_count = train.len();
    let training = read_training(&source.join("train.jsonl"), train)?;
    let input = std::fs::read_to_string(source.join("pilot-validation.jsonl"))?;
    let metadata: Vec<_> = input.lines().collect();
    let rows: Vec<ValidationRow> = metadata
        .iter()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("validation metadata line {}", i + 1))
        })
        .collect::<Result<_>>()?;
    ensure!(
        rows.iter()
            .map(|row| &row.id)
            .eq(manifest.packed_ids.validation.iter()),
        "validation metadata order differs from the pack manifest"
    );
    let prompt = manifest
        .prompt_width_bytes
        .checked_mul(2)
        .filter(|&n| n > 0)
        .context("invalid prompt width")?;
    let target = manifest
        .response_width_bytes_including_eos
        .checked_mul(2)
        .filter(|&n| n > 0)
        .context("invalid target width")?;
    let stride = prompt
        .checked_add(target)
        .context("aligned record width overflows")?;
    let aligned = read_verified(source, "validation.aligned.bin", &manifest.corpus_hashes)?;
    let codec = read_verified(source, "validation.codec.bin", &manifest.corpus_hashes)?;
    ensure!(
        rows.len().checked_mul(stride) == Some(aligned.len())
            && rows.len().checked_mul(target) == Some(codec.len()),
        "validation binary sizes disagree with the widths and row count"
    );
    let mut excluded = Vec::new();
    let mut clean = Vec::new();
    let mut pools: BTreeMap<_, Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        ensure!(
            aligned[index * stride + prompt..(index + 1) * stride]
                == codec[index * target..(index + 1) * target],
            "aligned and codec targets differ for {}",
            row.id
        );
        let key = stem(&row.question);
        if training.raw.contains(&key) {
            let reason = if training.packed.contains(&key) {
                "packed_train_stem"
            } else {
                "raw_train_only_stem"
            };
            excluded.push(Excluded {
                id: &row.id,
                index,
                field: &row.field,
                reason,
            });
        } else {
            clean.push(index);
            pools
                .entry((row.field.as_str(), answer_band(row.answer.len())))
                .or_default()
                .push(index);
        }
    }
    ensure!(
        clean.len() >= size,
        "{} clean rows cannot fill a screen of {size}",
        clean.len()
    );
    let screen = stratified_screen(pools, &rows, size);
    let views = Views {
        rows: &rows,
        metadata,
        aligned: &aligned,
        codec: &codec,
        aligned_stride: stride,
        target_stride: target,
    };
    create_run_directory(out)?;
    let full = views.write(&out.join("clean"), &clean)?;
    let fast = views.write(&out.join("screen"), &screen)?;
    let mut exclusion_counts = BTreeMap::new();
    for exclusion in &excluded {
        *exclusion_counts.entry(exclusion.reason).or_insert(0usize) += 1;
    }
    write_json(&out.join("excluded.json"), &excluded)?;
    let report = json!({
        "scope":"historical_development","source_manifest_sha256":manifest_digest,
        "raw_train_metadata_sha256":training.sha256,"source_validation_metadata_sha256":format!("{:x}",Sha256::digest(input)),
        "prompt_width_bytes":manifest.prompt_width_bytes,"response_width_bytes_including_eos":manifest.response_width_bytes_including_eos,
        "raw_validation_cases":rows.len(),"raw_train_rows_seen":training.rows,"packed_train_rows_seen":train_count,
        "screen_seed":LONG_RECORD_SEED,"removed":exclusion_counts,"subsets":{"clean":full,"screen":fast},
    });
    write_json(&out.join("manifest.json"), &report)?;
    Ok(report)
}
