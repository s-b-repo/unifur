//! Teacher-forced adaptation of the last block's selected expert biases.

use super::{
    floats, lookup, read_manifest, AffineNorm, ConceptHead, Kv, Linear, Matrix, ReleasedModel,
    Routing, CODES, DEVICE, LATENT, LENGTH, LEVELS, WIDTH,
};
use anyhow::{ensure, Context, Result};
use burn::tensor::{activation::softmax, TensorData};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Fit output biases with the released trunk and codec held fixed.
#[derive(Args, Debug, Clone)]
pub struct TuneArgs {
    /// Converted NCM model directory.
    #[arg(long)]
    pub model: PathBuf,
    /// JSONL training documents with prompt, field, codes and optional rejected_codes.
    #[arg(long)]
    pub train: PathBuf,
    /// Disjoint held-out documents with the same schema.
    #[arg(long)]
    pub validation: PathBuf,
    /// Save the adapter and held-out scores to a new file.
    #[arg(long)]
    pub out: PathBuf,
    /// Optimization passes through the cached concept states.
    #[arg(long, default_value_t = 6)]
    pub epochs: usize,
    /// Step size for the token-averaged code cross-entropy gradient.
    #[arg(long, default_value_t = 1.0)]
    pub learning_rate: f64,
    /// Recompute source and coarser levels independently for each target.
    #[arg(long)]
    pub no_share_prefix: bool,
}

type Codes = Vec<Vec<i64>>;

#[derive(Deserialize)]
struct Document {
    id: String,
    field: String,
    prompt: String,
    codes: Codes,
    #[serde(default)]
    rejected_codes: Vec<Codes>,
}

fn valid_codes(codes: &Codes) -> bool {
    codes.len() == LEVELS.len()
        && codes.iter().zip(LEVELS).all(|(ids, length)| {
            ids.len() == length
                && ids
                    .iter()
                    .all(|&id| usize::try_from(id).is_ok_and(|id| id < CODES))
        })
}

fn documents(path: &Path) -> Result<Vec<Document>> {
    let mut records = Vec::new();
    let mut ids = BTreeSet::new();
    for (i, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Document = serde_json::from_str(&line)
            .with_context(|| format!("parse {}:{}", path.display(), i + 1))?;
        ensure!(
            !record.id.is_empty()
                && ids.insert(record.id.clone())
                && !record.prompt.trim().is_empty(),
            "missing prompt, empty id or duplicate id at {}:{}",
            path.display(),
            i + 1
        );
        ensure!(
            valid_codes(&record.codes),
            "document {} has invalid released code labels",
            record.id
        );
        let mut targets = BTreeSet::from([&record.codes]);
        ensure!(
            record
                .rejected_codes
                .iter()
                .all(|codes| valid_codes(codes) && targets.insert(codes)),
            "document {} has invalid or duplicate rejected codes",
            record.id
        );
        records.push(record);
    }
    ensure!(
        !records.is_empty(),
        "{} contains no documents",
        path.display()
    );
    Ok(records)
}

struct Level {
    expert: usize,
    index: usize,
    state: Matrix,
    labels: Vec<i64>,
}

struct CachedDocument {
    field: String,
    branches: Vec<Vec<Level>>,
}

type Branch<'a> = (usize, usize, &'a Codes);
type Source<'a> = (&'a str, Vec<u32>, Option<(usize, usize)>);

struct Forks<'a> {
    model: &'a mut ReleasedModel,
    prefix: &'a [Kv],
    routing: &'a Routing,
    cached: &'a mut [CachedDocument],
    level_passes: &'a mut usize,
}

impl Forks<'_> {
    // Only labels at earlier levels enter the hidden state. Split at the first
    // differing code group and retain the original record order for optimization.
    fn cache(
        &mut self,
        level: usize,
        reconstruction: Matrix,
        branches: &[Branch<'_>],
    ) -> Result<()> {
        let state = self
            .model
            .level_state(level, &reconstruction, self.prefix, self.routing)?;
        *self.level_passes += 1;
        let expert = *state
            .experts
            .last()
            .context("field has no final specialist")?;
        let mut children = BTreeMap::<&[i64], Vec<Branch<'_>>>::new();
        for &(document, alternative, codes) in branches {
            let labels = &codes[level];
            self.cached[document].branches[alternative].push(Level {
                expert,
                index: level,
                state: state.hidden.clone(),
                labels: labels.clone(),
            });
            if level + 1 < LEVELS.len() {
                children
                    .entry(labels)
                    .or_default()
                    .push((document, alternative, codes));
            }
        }
        for (labels, members) in children {
            let next =
                self.model
                    .codec
                    .refine_reconstruction(reconstruction.clone(), level, labels);
            self.cache(level + 1, next, &members)?;
        }
        Ok(())
    }
}

fn cache(
    model: &mut ReleasedModel,
    data: &[Document],
    share_prefix: bool,
    split: &str,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<CachedDocument>> {
    let mut groups = BTreeMap::<Source<'_>, Vec<Branch<'_>>>::new();
    let mut cached = Vec::with_capacity(data.len());
    for (index, doc) in data.iter().enumerate() {
        let (prompt, _) = model.prompt(&doc.prompt)?;
        for (alternative, codes) in std::iter::once(&doc.codes)
            .chain(&doc.rejected_codes)
            .enumerate()
        {
            let independent = (!share_prefix).then_some((index, alternative));
            groups
                .entry((&doc.field, prompt.clone(), independent))
                .or_default()
                .push((index, alternative, codes));
        }
        cached.push(CachedDocument {
            field: doc.field.clone(),
            branches: (0..=doc.rejected_codes.len())
                .map(|_| Vec::with_capacity(LEVELS.len()))
                .collect(),
        });
    }
    let (mut positions, mut level_passes, mut branches_cached) = (0, 0, 0);
    for (index, ((field, prompt, _), branches)) in groups.iter().enumerate() {
        let routing = Routing::load(&model.manifest, Some(field), false, true)?;
        let (prefix, _) = model.prefix_cache(prompt, &routing)?;
        positions += prompt.len();
        Forks {
            model,
            prefix: &prefix,
            routing: &routing,
            cached: &mut cached,
            level_passes: &mut level_passes,
        }
        .cache(0, Matrix::zeros([LENGTH, LATENT], &DEVICE), branches)?;
        branches_cached += branches.len();
        progress(&format!(
            "cached {split} source {}/{} ({field}, {} targets)",
            index + 1,
            groups.len(),
            branches.len()
        ));
    }
    progress(&format!("{split}: {} source passes, {positions} source positions, {level_passes} level passes, {branches_cached} target branches", groups.len()));
    Ok(cached)
}

/// Sum of code negative log-likelihoods and its optional broadcast-bias gradient.
fn score(
    state: Matrix,
    norm: &AffineNorm,
    head: &ConceptHead,
    book: &Matrix,
    labels: &[i64],
    backward: bool,
) -> Result<(f64, Option<Matrix>)> {
    let centered = state.clone() - state.mean_dim(1);
    let inv_std = (centered.clone().square().mean_dim(1) + 1e-6)
        .sqrt()
        .recip();
    let normalized = centered * inv_std.clone();
    let mut hidden = normalized.clone() * norm.weight.clone();
    if let Some(bias) = &norm.bias {
        hidden = hidden + bias.clone();
    }
    let logits = head.logits(hidden, book);
    let [rows, codes] = logits.dims();
    ensure!(rows == labels.len(), "code targets and positions differ");
    let shifted = logits.clone() - logits.clone().max_dim(1);
    let partition = shifted.clone().exp().sum_dim(1).log();
    let logs = floats(shifted - partition)?;
    let mut loss = 0.0;
    for (row, &label) in labels.iter().enumerate() {
        let label = usize::try_from(label).context("negative code label")?;
        ensure!(
            label < codes,
            "code label {label} outside a vocabulary of {codes}"
        );
        let offset = row * codes + label;
        loss -= f64::from(logs[offset]);
    }
    ensure!(loss.is_finite(), "the code loss is non-finite");
    if !backward {
        return Ok((loss, None));
    }
    // The target is one-hot. Projecting it just selects the corresponding
    // code vector, so subtract in latent space after the shared projection.
    let derivative = (softmax(logits, 1).matmul(book.clone()) - lookup(book, labels))
        .matmul(head.projection.weight.clone().transpose())
        * (2.0 * head.scale)
        * norm.weight.clone();
    let mean = derivative.clone().mean_dim(1);
    let aligned = (derivative.clone() * normalized.clone()).mean_dim(1);
    let gradient = (derivative - mean - normalized * aligned) * inv_std;
    Ok((loss, Some(gradient.sum_dim(0))))
}

type Biases = BTreeMap<usize, Matrix>;
type Gradients = BTreeMap<usize, (Matrix, usize)>;

fn score_level(
    model: &ReleasedModel,
    row: &Level,
    biases: &Biases,
    backward: bool,
) -> Result<(f64, Option<Matrix>)> {
    let x = biases.get(&row.expert).map_or_else(
        || row.state.clone(),
        |bias| row.state.clone() + bias.clone(),
    );
    score(
        x,
        &model.final_norm,
        &model.heads[row.index],
        &model.codec.books[row.index],
        &row.labels,
        backward,
    )
}

struct BranchScore {
    nll: f64,
    gradients: Vec<Matrix>,
}

fn score_branch(
    model: &ReleasedModel,
    branch: &[Level],
    biases: &Biases,
    backward: bool,
) -> Result<BranchScore> {
    let mut result = BranchScore {
        nll: 0.0,
        gradients: Vec::new(),
    };
    for row in branch {
        let (loss, gradient) = score_level(model, row, biases, backward)?;
        result.nll += loss;
        if backward {
            result
                .gradients
                .push(gradient.context("no expert bias gradient")?);
        }
    }
    Ok(result)
}

// Delta is the rejected branch's log-probability advantage per code.
fn pair_objective(delta: f64) -> (f64, f64) {
    let tail = (-delta.abs()).exp();
    let weight = if delta >= 0.0 { 1.0 } else { tail };
    (delta.max(0.0) + tail.ln_1p(), weight / (1.0 + tail))
}

#[derive(Default)]
struct Metrics {
    nll: f64,
    codes: usize,
    documents: usize,
    contrastive_documents: usize,
    pair_loss: f64,
}

impl Metrics {
    fn add(&mut self, other: Self) {
        self.nll += other.nll;
        self.codes += other.codes;
        self.documents += other.documents;
        self.contrastive_documents += other.contrastive_documents;
        self.pair_loss += other.pair_loss;
    }

    fn code_nats(&self) -> f64 {
        self.nll / self.codes as f64
    }

    fn pair_nats(&self) -> Option<f64> {
        (self.contrastive_documents > 0).then(|| self.pair_loss / self.documents as f64)
    }

    fn objective(&self) -> f64 {
        self.code_nats() + self.pair_nats().unwrap_or(0.0)
    }
}

fn document_objective(
    model: &ReleasedModel,
    doc: &CachedDocument,
    biases: &Biases,
    mut gradients: Option<&mut Gradients>,
) -> Result<Metrics> {
    let backward = gradients.is_some();
    let (preferred, rejected) = doc
        .branches
        .split_first()
        .context("document has no target")?;
    let positive = score_branch(model, preferred, biases, backward)?;
    let codes = preferred.iter().map(|row| row.labels.len()).sum();
    let mut metrics = Metrics {
        nll: positive.nll,
        codes,
        documents: 1,
        contrastive_documents: usize::from(!rejected.is_empty()),
        pair_loss: 0.0,
    };
    if let Some(accumulated) = gradients.as_deref_mut() {
        for (row, gradient) in preferred.iter().zip(&positive.gradients) {
            let entry = accumulated
                .entry(row.expert)
                .or_insert_with(|| (Matrix::zeros([1, WIDTH], &DEVICE), 0));
            entry.0 = entry.0.clone() + gradient.clone();
            entry.1 += row.labels.len();
        }
    }
    for branch in rejected {
        let negative = score_branch(model, branch, biases, backward)?;
        let (loss, weight) = pair_objective((positive.nll - negative.nll) / codes as f64);
        metrics.pair_loss += loss / rejected.len() as f64;
        if let Some(accumulated) = gradients.as_deref_mut() {
            let scale = weight / rejected.len() as f64;
            for ((accepted, rejected), (plus, minus)) in preferred
                .iter()
                .zip(branch)
                .zip(positive.gradients.iter().zip(negative.gradients))
            {
                ensure!(
                    accepted.expert == rejected.expert,
                    "fork uses a different field specialist"
                );
                let entry = accumulated
                    .get_mut(&accepted.expert)
                    .context("missing expert gradient")?;
                // The optimizer divides by the accepted tokens per expert below.
                // Weighting the raw gradient difference here gives the derivative
                // of the token-normalized pair term on the same scale as code CE.
                entry.0 = entry.0.clone() + (plus.clone() - minus) * scale;
            }
        }
    }
    Ok(metrics)
}

fn evaluate(
    model: &ReleasedModel,
    data: &[CachedDocument],
    biases: &Biases,
) -> Result<BTreeMap<String, Metrics>> {
    let mut totals = BTreeMap::<String, Metrics>::new();
    for doc in data {
        totals
            .entry(doc.field.clone())
            .or_default()
            .add(document_objective(model, doc, biases, None)?);
    }
    Ok(totals)
}

#[derive(Serialize, Deserialize)]
struct Adapter {
    version: u32,
    experts: usize,
    epochs: usize,
    train_documents: usize,
    validation_documents: usize,
    biases: BTreeMap<usize, Vec<f32>>,
    held_out: BTreeMap<String, Quality>,
}

#[derive(Serialize, Deserialize)]
struct Quality {
    base_nats: f64,
    tuned_nats: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_pair_nats: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tuned_pair_nats: Option<f64>,
    kept: bool,
}

pub(super) fn load_biases(path: &Path, experts: usize) -> Result<Biases> {
    let adapter: Adapter = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    ensure!(
        adapter.version == 1 && adapter.experts == experts,
        "adapter version or expert count differs from model"
    );
    adapter
        .biases
        .into_iter()
        .map(|(id, bias)| {
            ensure!(
                id < experts
                    && bias.len() == WIDTH
                    && bias.iter().all(|x| x.is_finite() && x.abs() <= 0.5),
                "invalid output bias for expert {id}"
            );
            Ok((
                id,
                Matrix::from_data(TensorData::new(bias, [1, WIDTH]), &DEVICE),
            ))
        })
        .collect()
}

/// Fit the expert biases on one split and keep fields that improve on the other.
pub fn tune(args: TuneArgs, progress: &mut dyn FnMut(&str)) -> Result<String> {
    ensure!(args.epochs > 0, "epochs must be positive");
    ensure!(
        args.learning_rate.is_finite() && args.learning_rate > 0.0,
        "learning rate must be positive"
    );
    ensure!(
        !args.out.exists(),
        "adapter destination {} already exists",
        args.out.display()
    );
    let train = documents(&args.train)?;
    let validation = documents(&args.validation)?;
    let fields = |docs: &[Document]| {
        docs.iter()
            .map(|doc| doc.field.clone())
            .collect::<BTreeSet<_>>()
    };
    let train_ids: BTreeSet<_> = train.iter().map(|doc| doc.id.as_str()).collect();
    ensure!(
        validation
            .iter()
            .all(|doc| !train_ids.contains(doc.id.as_str())),
        "training and validation overlap"
    );
    ensure!(
        fields(&train) == fields(&validation),
        "training and validation field sets differ"
    );
    let manifest = read_manifest(&args.model)?;
    let routing = Routing::load(&manifest, None, false, true)?;
    let mut model = ReleasedModel::load(&args.model, manifest, routing.spec)?;
    let train = cache(
        &mut model,
        &train,
        !args.no_share_prefix,
        "training",
        progress,
    )?;
    let validation = cache(
        &mut model,
        &validation,
        !args.no_share_prefix,
        "held-out",
        progress,
    )?;
    let baseline = evaluate(&model, &validation, &Biases::new())?;
    let mut biases = Biases::new();
    for epoch in 1..=args.epochs {
        let mut gradients = Gradients::new();
        let mut metrics = Metrics::default();
        for doc in &train {
            metrics.add(document_objective(
                &model,
                doc,
                &biases,
                Some(&mut gradients),
            )?);
        }
        for (expert, (gradient, count)) in gradients {
            let bias = biases
                .entry(expert)
                .or_insert_with(|| Matrix::zeros([1, WIDTH], &DEVICE));
            *bias =
                (bias.clone() - gradient * (args.learning_rate / count as f64)).clamp(-0.5, 0.5);
        }
        let mut message = format!(
            "epoch {epoch}/{} training code CE {:.5}",
            args.epochs,
            metrics.code_nats()
        );
        if let Some(pair_nats) = metrics.pair_nats() {
            use std::fmt::Write as _;
            write!(message, ", pair loss {pair_nats:.5}")?;
        }
        progress(&message);
    }
    let tuned = evaluate(&model, &validation, &biases)?;
    let mut held_out = BTreeMap::new();
    for (field, base) in baseline {
        let candidate = tuned.get(&field).context("missing validation field")?;
        let kept = candidate.objective() < base.objective();
        if !kept {
            for doc in validation.iter().filter(|doc| doc.field == field) {
                for row in doc.branches.iter().flatten() {
                    biases.remove(&row.expert);
                }
            }
        }
        held_out.insert(
            field,
            Quality {
                base_nats: base.code_nats(),
                tuned_nats: candidate.code_nats(),
                base_pair_nats: base.pair_nats(),
                tuned_pair_nats: candidate.pair_nats(),
                kept,
            },
        );
    }
    let biases = biases
        .into_iter()
        .map(|(id, bias)| Ok((id, floats(bias)?)))
        .collect::<Result<_>>()?;
    let adapter = Adapter {
        version: 1,
        experts: model.manifest.experts,
        epochs: args.epochs,
        train_documents: train.len(),
        validation_documents: validation.len(),
        biases,
        held_out,
    };
    let mut output = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&args.out)?,
    );
    serde_json::to_writer(&mut output, &adapter)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(format!(
        "Fit {} expert biases from {} training documents and {} held-out documents.\n{}",
        adapter.biases.len(),
        adapter.train_documents,
        adapter.validation_documents,
        serde_json::to_string_pretty(&adapter.held_out)?
    ))
}

pub(super) fn gradient_residual() -> Result<f64> {
    let matrix =
        |values: &[f32], shape| Matrix::from_data(TensorData::new(values.to_vec(), shape), &DEVICE);
    let x = matrix(&[0.3, 0.4, -0.1, -0.5, 0.2, 0.9], [2, 3]);
    let book = matrix(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.5], [3, 2]);
    let norm = AffineNorm {
        weight: matrix(&[0.8, 1.2, 0.5], [1, 3]),
        bias: None,
    };
    let head = ConceptHead {
        projection: Linear {
            weight: matrix(&[0.2, 0.3, 0.9, -0.4, 0.1, 0.5], [3, 2]),
            bias: None,
        },
        squared_norms: book.clone().square().sum_dim(1).transpose(),
        scale: 0.7,
        metric: Matrix::ones([1, 2], &DEVICE),
    };
    let labels = [0, 2];
    let (_, grad) = score(x.clone(), &norm, &head, &book, &labels, true)?;
    let analytic = f64::from(floats(grad.context("no analytical gradient")?)?[0]);
    let step = 1e-3_f32;
    let bias = matrix(&[step, 0.0, 0.0], [1, 3]);
    let plus = score(
        x.clone() + bias.clone(),
        &norm,
        &head,
        &book,
        &labels,
        false,
    )?
    .0;
    let minus = score(x - bias, &norm, &head, &book, &labels, false)?.0;
    Ok((analytic - (plus - minus) / (2.0 * f64::from(step))).abs())
}

pub(super) fn pair_gradient_residual() -> f64 {
    [-3.0, 0.0, 2.0]
        .into_iter()
        .map(|delta| {
            let step = 1e-5;
            let (loss_above, _) = pair_objective(delta + step);
            let (loss_below, _) = pair_objective(delta - step);
            ((loss_above - loss_below) / (2.0 * step) - pair_objective(delta).1).abs()
        })
        .fold(0.0, f64::max)
}
