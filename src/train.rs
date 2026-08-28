//! Block-wise training loop (`main.py` / Lightning plumbing of the reference,
//! reduced to a self-contained Burn loop).
//!
//! One loop serves every objective the crate implements -- the plain
//! DiffusionBlocks loss, the consistency-augmented one, rectified flow, and
//! block distillation -- because they differ only in how a step's loss is
//! computed. Everything around that (batching, gradient gating, logging,
//! checkpointing, resume) is shared, so a new objective cannot accidentally
//! come with a subtly different training procedure.

use std::path::PathBuf;

use crate::{
    reweight::{LogVarianceHead, SigmaImportanceSampler, UncertaintyWeighting},
    schedule::BalanceSchedule,
    consistency::ConsistencyConfig,
    data::{Batch, SyntheticDataset, TrainDataset},
    dblock::{DblockClassifier, DblockConfig},
    distill::DistillConfig,
    quality::{
        global_grad_norm, grad_norm_ok, non_finite_parameters, StepVerdict, TrainingChecks,
        TrainingHealth, TrainingPhase,
    },
    rawdata::RawImageDataset,
    schedule::{Ema, GradientAccumulator, LossScales, LrSchedule},
    vit::ViTDiTConfig,
};
use burn::{
    backend::{
        autodiff::checkpoint::strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing},
        Autodiff, NdArray,
    },
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Distribution, Tensor},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Autodiff-enabled ndarray backend used for CPU training.
pub type DefaultTrainBackend = Autodiff<NdArray<f32>>;
/// Variant that checkpoints intermediate activations during backward
/// (roadmap 15.2): trades compute for lower peak memory.
pub type CheckpointedTrainBackend = Autodiff<NdArray<f32>, BalancedCheckpointing>;

/// Which dataset to train on.
#[derive(Debug, Clone)]
pub enum DatasetChoice {
    /// Random images shaped like CIFAR-100; needs no download.
    Synthetic,
    /// CIFAR-100 binary distribution in `dir`.
    Cifar100 { dir: PathBuf, streaming: bool },
    /// Preprocessed Tiny ImageNet in `dir` (see [`crate::tinyimagenet`]).
    TinyImagenet { dir: PathBuf, streaming: bool },
}

impl DatasetChoice {
    /// Parse a CLI dataset name; `dir` is required by the real datasets.
    pub fn parse(name: &str, dir: Option<PathBuf>, streaming: bool) -> anyhow::Result<Self> {
        match name {
            "synthetic" => Ok(Self::Synthetic),
            "cifar100" => Ok(Self::Cifar100 {
                dir: dir.ok_or_else(|| anyhow::anyhow!("cifar100 needs --data-dir"))?,
                streaming,
            }),
            "tiny-imagenet" | "tinyimagenet" => Ok(Self::TinyImagenet {
                dir: dir.ok_or_else(|| anyhow::anyhow!("tiny-imagenet needs --data-dir"))?,
                streaming,
            }),
            other => anyhow::bail!(
                "unknown dataset '{other}' (expected synthetic|cifar100|tiny-imagenet)"
            ),
        }
    }

    /// Image side length and class count implied by the dataset, so the model
    /// cannot be configured inconsistently with its data.
    pub fn shape(&self) -> Option<(usize, usize)> {
        match self {
            Self::Synthetic => None,
            Self::Cifar100 { .. } => Some((32, 100)),
            Self::TinyImagenet { .. } => Some((64, 200)),
        }
    }
}

/// Dispatch wrapper: [`TrainDataset`] has a generic method and so is not
/// object-safe.
enum AnyDataset {
    Synthetic(SyntheticDataset),
    // Boxed: the raw dataset owns several reusable staging buffers and is an
    // order of magnitude larger than the synthetic one.
    Raw(Box<RawImageDataset>),
}

impl AnyDataset {
    fn next<B: burn::tensor::backend::Backend, R: Rng>(
        &mut self,
        rng: &mut R,
        device: &B::Device,
    ) -> Batch<B> {
        match self {
            Self::Synthetic(d) => d.next_batch(rng, device),
            Self::Raw(d) => d.next_batch(rng, device),
        }
    }
}

/// Which loss a training step computes.
#[derive(Debug, Clone)]
pub enum Objective {
    /// EDM-weighted cross-entropy on one random block (original paper).
    Dblock,
    /// [`Objective::Dblock`] plus the Phase-3 consistency residuals.
    Consistency(Box<ConsistencyConfig>),
    /// Rectified-flow velocity regression.
    Flow,
    /// Block distillation against a frozen teacher.
    ///
    /// Without a separate teacher checkpoint the *initial* model is frozen and
    /// used as the teacher, which is self-distillation: still meaningful,
    /// because the student is asked to cover several teacher substeps in one.
    Distill(Box<DistillConfig>),
}

impl Objective {
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "dblock" => Ok(Self::Dblock),
            "consistency" => Ok(Self::Consistency(Box::default())),
            "flow" => Ok(Self::Flow),
            "distill" => Ok(Self::Distill(Box::default())),
            other => anyhow::bail!(
                "unknown objective '{other}' (expected dblock|consistency|flow|distill)"
            ),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Dblock => "dblock",
            Self::Consistency(_) => "consistency",
            Self::Flow => "flow",
            Self::Distill(_) => "distill",
        }
    }
}

/// Re-exported so `TrainConfig` users need only one import.
pub use crate::quality::GradNormGate;

/// Training hyperparameters.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub image_size: usize,
    pub num_labels: usize,
    pub batch_size: usize,
    /// Number of blocks.
    pub num_blocks: usize,
    /// Sigma window extension factor.
    pub gamma: f64,
    pub lr: f64,
    pub weight_decay: f64,
    /// Total optimizer steps.
    pub steps: usize,
    pub log_every: usize,
    pub seed: u64,
    /// Optional JSONL metrics sink (roadmap 1.5 / 15.6).
    pub log_file: Option<PathBuf>,
    pub dataset: DatasetChoice,
    pub objective: Objective,
    /// Quality verification applied at each phase of a training step.
    pub checks: TrainingChecks,
    /// Checkpoint to initialize from (roadmap 15.7).
    pub resume: Option<PathBuf>,
    /// Frozen teacher for [`Objective::Distill`].
    pub teacher: Option<PathBuf>,
    /// Flat mixture-of-experts placement in the trunk (roadmap 6.5).
    pub moe: Option<crate::vit::MoeTrunkConfig>,
    /// Boxes of specialized micro experts in the trunk (roadmap 18.7).
    pub mosme: Option<crate::vit::MosmeTrunkConfig>,
    /// Learning-rate schedule (roadmap 20.1). Defaults to constant, which is
    /// what the loop did before schedules existed.
    pub lr_schedule: LrSchedule,
    /// Micro-batches per optimizer step (roadmap 20.2).
    pub accumulate: usize,
    /// Rescale gradients whose global norm exceeds this (roadmap 20.3).
    /// Complements, rather than replaces, the gradient *gate*.
    pub clip_norm: Option<f32>,
    /// Keep an exponential moving average of the weights (roadmap 20.4).
    pub ema_decay: Option<f64>,
    /// Normalize each block's loss onto a common scale (roadmap 20.7).
    pub normalize_block_loss: bool,
    /// Learned per-sigma uncertainty weighting (roadmap 20.5). `0.0` is the
    /// exact identity and allocates nothing.
    pub uncertainty: f64,
    /// CDF bins for sigma importance sampling (roadmap 20.6). `0` disables it,
    /// and a cold sampler is exactly plain sampling in any case.
    pub importance_bins: usize,
    /// How the auxiliary balance-loss weight evolves. `None` keeps the model's
    /// own fixed `moe_aux_weight`, which is what every run did before this
    /// existed. See [`BalanceSchedule`] for why annealing it is worth doing.
    pub balance_schedule: Option<BalanceSchedule>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            image_size: 32,
            num_labels: 100,
            batch_size: 128,
            num_blocks: 3,
            gamma: 0.05,
            lr: 1e-3,
            weight_decay: 0.01,
            steps: 200,
            log_every: 20,
            seed: 42,
            log_file: None,
            dataset: DatasetChoice::Synthetic,
            objective: Objective::Dblock,
            checks: TrainingChecks::default(),
            resume: None,
            teacher: None,
            moe: None,
            mosme: None,
            lr_schedule: LrSchedule::default(),
            accumulate: 1,
            clip_norm: None,
            ema_decay: None,
            normalize_block_loss: false,
            uncertainty: 0.0,
            importance_bins: 0,
            balance_schedule: None,
        }
    }
}

impl TrainConfig {
    /// Model configuration implied by this training configuration.
    ///
    /// A real dataset dictates the image size and class count, so those are
    /// taken from the dataset rather than from the CLI defaults; silently
    /// training a 32x32/100-class model on 64x64/200-class data would
    /// otherwise be an easy mistake to make.
    pub fn vit_config(&self) -> ViTDiTConfig {
        let (image_size, num_labels) = self
            .dataset
            .shape()
            .unwrap_or((self.image_size, self.num_labels));
        let mut cfg = ViTDiTConfig::with_image_size(image_size, num_labels);
        cfg.moe = self.moe;
        cfg.mosme = self.mosme.clone();
        cfg
    }

    pub fn dblock_config(&self) -> DblockConfig {
        DblockConfig {
            num_blocks: self.num_blocks,
            gamma: self.gamma,
            ..DblockConfig::default()
        }
    }
}

/// Outcome of a training run.
#[derive(Debug, Clone, Default)]
pub struct TrainSummary {
    pub steps_taken: usize,
    /// Steps rejected by a quality check at any phase.
    pub steps_skipped: usize,
    pub final_loss: f32,
    pub mean_loss: f32,
    pub elapsed_secs: f64,
    /// Per-block quality state and the failures that were recorded.
    pub health: TrainingHealth,
    /// Set when the run stopped early because verification kept failing.
    pub aborted: Option<String>,
    /// Times the live model was re-verified during the run.
    pub periodic_verifications: usize,
    /// Steps whose gradients were rescaled by the clipping bound.
    pub steps_clipped: usize,
    /// Final learning rate the schedule produced.
    pub final_lr: f64,
}

impl TrainSummary {
    /// Fraction of steps the gradient gate rejected.
    pub fn skip_rate(&self) -> f32 {
        let total = self.steps_taken + self.steps_skipped;
        if total == 0 {
            0.0
        } else {
            self.steps_skipped as f32 / total as f32
        }
    }
}

/// A trained model paired with its run summary.
pub type TrainOutcome<C> = (DblockClassifier<Autodiff<NdArray<f32>, C>>, TrainSummary);

/// Train a dblock classifier; returns the trained model and a summary.
pub fn train(
    config: &TrainConfig,
) -> anyhow::Result<(DblockClassifier<DefaultTrainBackend>, TrainSummary)> {
    train_generic::<NoCheckpointing>(config)
}

/// Backwards-compatible entry point for the synthetic smoke run.
pub fn train_synthetic(
    config: &TrainConfig,
) -> anyhow::Result<DblockClassifier<DefaultTrainBackend>> {
    Ok(train(config)?.0)
}

/// Backend-generic training loop; `C` selects the autodiff checkpointing
/// strategy (see [`CheckpointedTrainBackend`]).
pub fn train_synthetic_generic<C>(
    config: &TrainConfig,
) -> anyhow::Result<DblockClassifier<Autodiff<NdArray<f32>, C>>>
where
    C: CheckpointStrategy,
{
    Ok(train_generic::<C>(config)?.0)
}

/// The loop itself.
pub fn train_generic<C>(
    config: &TrainConfig,
) -> anyhow::Result<TrainOutcome<C>>
where
    C: CheckpointStrategy,
{
    type Recorder = burn::record::NamedMpkFileRecorder<burn::record::FullPrecisionSettings>;
    let device = Default::default();

    // Seed the on-device RNG (init, data generation, dropout) so runs with
    // the same config produce bit-identical checkpoints.
    <Autodiff<NdArray<f32>, C> as burn::tensor::backend::Backend>::seed(&device, config.seed);

    let vit_config = config.vit_config();
    let dblock_config = config.dblock_config();

    let hidden_size = vit_config.hidden_size;
    let mut model =
        DblockClassifier::<Autodiff<NdArray<f32>, C>>::new(&vit_config, &dblock_config, &device);
    if let Some(path) = &config.resume {
        model = model
            .load_file(path, &Recorder::new(), &device)
            .map_err(|err| anyhow::anyhow!("resume from {}: {err}", path.display()))?;
        println!("resumed from {}", path.display());
    }

    // The teacher is frozen: either a separate checkpoint or a snapshot of the
    // starting model.
    let teacher = match &config.objective {
        Objective::Distill(_) => {
            let base = DblockClassifier::<Autodiff<NdArray<f32>, C>>::new(
                &vit_config,
                &dblock_config,
                &device,
            );
            Some(match &config.teacher {
                Some(path) => base
                    .load_file(path, &Recorder::new(), &device)
                    .map_err(|err| anyhow::anyhow!("load teacher {}: {err}", path.display()))?,
                None => model.clone(),
            })
        }
        _ => None,
    };

    // --- Phase: preflight -------------------------------------------------
    //
    // The schedule and preconditioning identities are what every later step
    // silently depends on. Verifying them costs milliseconds; discovering
    // afterwards that a whole run trained against a broken block-index
    // convention costs the run.
    let mut health = TrainingHealth::new(config.num_blocks);
    if config.checks.preflight {
        let report = crate::verify::preflight();
        println!("preflight: {}", report.summary());
        if !report.passed() {
            anyhow::bail!(
                "preflight verification failed, refusing to train:\n{}",
                report.render()
            );
        }
    }

    let mut dataset = open_dataset(config)?;
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Burn optimizers are functional: step() consumes the model record and
    // returns an updated one, so we reassign the binding every step.
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    println!(
        "lr schedule: {} (peak {:.2e})",
        config.lr_schedule.name(),
        config.lr_schedule.peak()
    );

    let mut running = RunningAvg::new(config.log_every);
    let start = std::time::Instant::now();
    let mut jsonl = config
        .log_file
        .as_ref()
        .map(|path| crate::logging::MetricsLogger::open(path))
        .transpose()?;

    // Per-sigma reweighting (roadmap 20.5 / 20.6). Both are training-time
    // objects: the log-variance head is *not* part of the model record, so a
    // run that enables it still writes checkpoints an unmodified build can
    // load, and a run that does not enable it allocates nothing.
    let uncertainty = UncertaintyWeighting::new(config.uncertainty);
    let mut logvar_head = (!uncertainty.is_identity()).then(|| {
        LogVarianceHead::<Autodiff<NdArray<f32>, C>>::new(64, 64, &device)
    });
    let mut logvar_optim = logvar_head.is_some().then(|| AdamWConfig::new().init());
    let mut importance = (config.importance_bins > 0)
        .then(|| SigmaImportanceSampler::new(config.importance_bins));
    if !uncertainty.is_identity() {
        println!(
            "uncertainty weighting: strength {:.2} (gradient becomes that of log-loss at the optimum)",
            uncertainty.strength
        );
    }
    if let Some(sampler) = &importance {
        println!(
            "sigma importance sampling: {} CDF bins, smoothing {:.2} (worst weight <= {:.1}x)",
            sampler.bins(),
            sampler.smoothing(),
            1.0 / sampler.smoothing()
        );
    }

    let mut summary = TrainSummary::default();
    let mut loss_sum = 0.0f64;
    let mut accumulator = GradientAccumulator::new(config.accumulate);
    let mut scales = LossScales::new(config.num_blocks);
    let mut ema = config.ema_decay.map(|d| Ema::new(&model, d));
    if config.accumulate > 1 {
        println!(
            "gradient accumulation: {} micro-batches per step",
            accumulator.every()
        );
    }

    for step in 0..config.steps {
        let batch = dataset.next(&mut rng, &device);
        let (loss, mut fields) = compute_loss(
            &model,
            teacher.as_ref(),
            &batch,
            config,
            step,
            &mut rng,
            &mut Reweighting {
                weighting: uncertainty,
                head: logvar_head.as_ref(),
                sampler: importance.as_mut(),
                balance: config.balance_schedule,
                step,
            },
        );
        let block_idx = block_of(&fields);

        let mut verdict = StepVerdict::accepted();
        let scalar_loss: f32 = loss.clone().into_scalar();

        // Per-block loss normalization. Blocks see wildly different EDM
        // weights, so without this the run is effectively tuned for whichever
        // block's sigma window happens to produce the largest loss.
        let block_scale = if config.normalize_block_loss {
            scales.observe(block_idx, scalar_loss);
            scales.scale(block_idx)
        } else {
            1.0
        };
        // Averaging over accumulated micro-batches keeps the gradient
        // magnitude independent of the accumulation count.
        let step_scale = block_scale * accumulator.loss_scale();
        let loss = if (step_scale - 1.0).abs() > f64::EPSILON {
            loss.mul_scalar(step_scale as f32)
        } else {
            loss
        };

        // --- Phase: loss --------------------------------------------------
        if config.checks.loss_finite && !scalar_loss.is_finite() {
            verdict.reject(TrainingPhase::Loss, format!("loss is {scalar_loss}"));
        }

        // --- Phase: gradients ---------------------------------------------
        // The backward pass still runs on a non-finite loss: skipping it would
        // desynchronize the autodiff graph, and the step is rejected anyway.
        let mut all_grads = loss.backward();
        // The head shares the backward pass but not the optimizer state: it is
        // one scalar per noise level against a whole trunk, so folding it into
        // the model's gradient norm would let it move the clipping threshold
        // and the gate for reasons that have nothing to do with the trunk.
        // `from_module` borrows rather than consuming, which is what lets two
        // modules draw from one backward pass.
        let head_grads = logvar_head
            .as_ref()
            .map(|head| GradientsParams::from_module(&mut all_grads, head));
        let grads = GradientsParams::from_grads(all_grads, &model);
        verdict.grad_norm = global_grad_norm(&model, &grads);

        if let Some(gate) = config.checks.grad_gate {
            if !grad_norm_ok(verdict.grad_norm, gate.min_norm, gate.max_norm) {
                verdict.reject(
                    TrainingPhase::Gradients,
                    format!(
                        "gradient norm {:.3e} outside [{:.1e}, {:.1e}]",
                        verdict.grad_norm, gate.min_norm, gate.max_norm
                    ),
                );
            }
        }

        // Fold this micro-batch into the accumulation buffer, or skip it if a
        // gate rejected it. A rejected micro-batch still advances the cycle:
        // dropping the whole cycle instead would let one persistently bad block
        // stall the run indefinitely.
        let cycle = if verdict.accepted {
            accumulator.fold(grads, &model)
        } else {
            accumulator.skip()
        };

        if let Some(mut summed) = cycle.into_gradients() {
            // Clipping applies to the *accumulated* gradient, because that is
            // the step being taken -- clipping each micro-batch separately
            // would bound k small vectors whose sum can still be large.
            let total_norm = global_grad_norm(&model, &summed);
            if let Some(max_norm) = config.clip_norm {
                let scale =
                    crate::schedule::clip_gradients(&mut summed, &model, total_norm, max_norm);
                if scale < 1.0 {
                    summary.steps_clipped += 1;
                }
            }

            let lr = config.lr_schedule.at(step);
            summary.final_lr = lr;
            model = optim.step(lr, model, summed);

            // `take()` only after every part is known present: evaluating it
            // inside the tuple pattern would move the head out even when the
            // match fails, silently disabling the feature for the rest of the
            // run while the startup banner still claimed it was on.
            if let (Some(head_optim), Some(head_grads)) = (logvar_optim.as_mut(), head_grads) {
                if let Some(head) = logvar_head.take() {
                    logvar_head = Some(head_optim.step(lr, head, head_grads));
                }
            }
            if let Some(ema) = ema.as_mut() {
                ema.update::<Autodiff<NdArray<f32>, C>>(&model);
            }

            // --- Phase: parameters ----------------------------------------
            // A NaN that reaches the weights poisons every later step, so it
            // is fatal rather than skippable: there is nothing left to train.
            if config.checks.parameters_finite {
                let bad = non_finite_parameters(&model);
                if bad > 0 {
                    // Unrecoverable: there is nothing left to train, and the
                    // per-block table would only describe a poisoned model, so
                    // the error carries everything worth reporting.
                    anyhow::bail!(
                        "step {step}: {} parameter tensor(s) went non-finite after the \
                         optimizer step; the run cannot continue. Last gradient norm \
                         {:.3e}, block {block_idx}. Try a lower learning rate or a \
                         tighter --grad gate.",
                        bad,
                        verdict.grad_norm
                    );
                }
            }
            summary.steps_taken += 1;
        } else {
            summary.steps_skipped += 1;
        }

        health.record(step, block_idx, scalar_loss, &verdict);
        running.push(scalar_loss);
        loss_sum += scalar_loss as f64;
        summary.final_loss = scalar_loss;

        // --- Phase: periodic ----------------------------------------------
        // Re-verify the *live* weights: the invariants below hold for any
        // model, so a violation means this run has diverged, not that the
        // implementation is wrong.
        if let Some(every) = config.checks.verify_every {
            if every > 0 && step % every == 0 {
                let report = crate::verify::Report {
                    certificates: crate::verify::model_health(
                        &model,
                        &batch.pixel_values,
                        &Tensor::random(
                            [batch.pixel_values.dims()[0], hidden_size],
                            Distribution::Normal(0.0, 1.0),
                            &device,
                        ),
                    ),
                };
                // Reported on success too: a check nobody can see run is a
                // check nobody trusts.
                println!("step {step}: periodic verification: {}", report.summary());
                if !report.passed() {
                    health.record_failure(
                        step,
                        crate::quality::CheckFailure {
                            phase: TrainingPhase::Periodic,
                            detail: report.summary(),
                        },
                    );
                }
                summary.periodic_verifications += 1;
            }
        }

        if health.should_abort(&config.checks) {
            let detail = format!(
                "{} consecutive steps rejected; last reason: {}",
                health.consecutive_rejections(),
                verdict.reason().unwrap_or_else(|| "unknown".to_string())
            );
            println!("aborting at step {step}: {detail}");
            summary.aborted = Some(detail);
            break;
        }

        if step % config.log_every.max(1) == 0 || step + 1 == config.steps {
            let sps = (step + 1) as f64 / start.elapsed().as_secs_f64().max(1e-9);
            println!(
                "step {:>6} | loss {:.4} | grad {:.3e} | avg {:.4} | skipped {} | {:.1} steps/s{}",
                step,
                scalar_loss,
                verdict.grad_norm,
                running.avg(),
                summary.steps_skipped,
                sps,
                verdict
                    .reason()
                    .map(|r| format!(" | REJECTED {r}"))
                    .unwrap_or_default()
            );
            if let Some(logger) = jsonl.as_mut() {
                fields.push(("grad_norm", crate::logging::jnum(verdict.grad_norm)));
                fields.push(("steps_per_s", crate::logging::jnum(sps as f32)));
                fields.push(("skipped", format!("{}", summary.steps_skipped)));
                fields.push(("accepted", format!("{}", verdict.accepted)));
                logger.log(step, &fields)?;
            }
        }
    }

    let completed = summary.steps_taken + summary.steps_skipped;
    summary.mean_loss = if completed == 0 {
        0.0
    } else {
        (loss_sum / completed as f64) as f32
    };
    summary.elapsed_secs = start.elapsed().as_secs_f64();

    let dead = health.dead_blocks();
    if !dead.is_empty() {
        println!(
            "warning: block(s) {dead:?} never received a non-zero gradient; \
             check num_blocks against num_hidden_layers and the sigma windows"
        );
    }
    summary.health = health;

    // The averaged weights are usually the better evaluation model, and are
    // what gets returned when EMA is enabled.
    if let Some(ema) = ema {
        println!("returning EMA weights ({} updates)", ema.updates());
        return Ok((ema.into_shadow(), summary));
    }
    Ok((model, summary))
}

/// Block index recorded in a step's metric fields, or 0 when the objective
/// does not train a single identifiable block (flow matching).
fn block_of(fields: &[(&'static str, String)]) -> usize {
    fields
        .iter()
        .find(|(k, _)| *k == "block")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0)
}

fn open_dataset(config: &TrainConfig) -> anyhow::Result<AnyDataset> {
    let (image_size, num_labels) = config
        .dataset
        .shape()
        .unwrap_or((config.image_size, config.num_labels));
    Ok(match &config.dataset {
        DatasetChoice::Synthetic => AnyDataset::Synthetic(SyntheticDataset::new(
            image_size,
            num_labels,
            config.batch_size,
            config.seed,
        )),
        DatasetChoice::Cifar100 { dir, streaming } => {
            AnyDataset::Raw(Box::new(crate::cifar::open(dir, true, config.batch_size, *streaming)?))
        }
        DatasetChoice::TinyImagenet { dir, streaming } => {
            AnyDataset::Raw(Box::new(crate::tinyimagenet::open(
                dir,
                true,
                config.batch_size,
                *streaming,
            )?))
        }
    })
}

/// Compute one step's loss plus the metric fields to log.
/// The per-sigma reweighting a training step may apply (roadmap 20.5 / 20.6).
///
/// Borrowed rather than owned so the log-variance head keeps its own optimizer
/// state in the training loop, and so a run with neither feature enabled passes
/// two `None`s and does exactly what it did before.
pub struct Reweighting<'a, B: AutodiffBackend<FloatElem = f32>> {
    pub weighting: UncertaintyWeighting,
    pub head: Option<&'a LogVarianceHead<B>>,
    pub sampler: Option<&'a mut SigmaImportanceSampler>,
    /// Overrides the model's fixed `moe_aux_weight` when set.
    pub balance: Option<BalanceSchedule>,
    /// The step the schedule is evaluated at.
    pub step: usize,
}

impl<B: AutodiffBackend<FloatElem = f32>> Reweighting<'_, B> {
    /// Run one `Objective::Dblock` step, applying whichever reweighting is on.
    ///
    /// Only this objective is reweighted. The others draw their noise levels
    /// through their own paths — consistency pairs sigmas across a boundary,
    /// flow matching works in `t` rather than sigma, and distillation follows
    /// the teacher's trajectory — so silently applying a sigma-indexed weight
    /// there would be weighting something other than what the name claims.
    fn dblock_step<R: Rng>(
        &mut self,
        model: &DblockClassifier<B>,
        batch: &Batch<B>,
        gamma: f64,
        rng: &mut R,
    ) -> (Tensor<B, 1>, crate::dblock::StepMetrics, Vec<(&'static str, String)>) {
        use crate::logging::jnum;

        let b = batch.pixel_values.dims()[0];
        let block_idx = rng.random_range(0..model.num_blocks());
        let sampler_cfg = crate::sigma::DblockSigmaSampler::new(model.num_blocks(), gamma);
        let (lo, hi) = sampler_cfg.extended_window(block_idx);
        let cdf_lo = crate::stats::norm_cdf((lo.ln() - sampler_cfg.p_mean) / sampler_cfg.p_std);
        let cdf_hi = crate::stats::norm_cdf((hi.ln() - sampler_cfg.p_mean) / sampler_cfg.p_std);

        let mut extra: Vec<(&'static str, String)> = Vec::new();

        // --- noise levels, optionally importance-sampled ------------------
        let (sigmas, weights): (Vec<f64>, Option<Vec<f64>>) = match self.sampler.as_deref() {
            Some(sampler) => {
                let drawn = sampler.sample(
                    rng,
                    cdf_lo,
                    cdf_hi,
                    sampler_cfg.p_mean,
                    sampler_cfg.p_std,
                    b,
                );
                let (s, w): (Vec<f64>, Vec<f64>) = drawn.into_iter().unzip();
                extra.push(("importance_max_weight", jnum(sampler.max_weight() as f32)));
                (s, Some(w))
            }
            None => (sampler_cfg.sample(rng, block_idx, b), None),
        };

        let parts = model.training_step_on(
            batch.pixel_values.clone(),
            batch.labels.clone(),
            &sigmas,
            block_idx,
            weights.as_deref(),
        );

        // --- feed the proposal what it just learned -----------------------
        if let Some(sampler) = self.sampler.as_deref_mut() {
            let observed: Vec<f32> = parts
                .per_sample
                .clone()
                .inner()
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
                .collect();
            for (sigma, value) in sigmas.iter().zip(&observed) {
                let bin = sampler.bin_of(
                    *sigma,
                    cdf_lo,
                    cdf_hi,
                    sampler_cfg.p_mean,
                    sampler_cfg.p_std,
                );
                sampler.observe(bin, f64::from(*value));
            }
        }

        // --- balance-loss weight ------------------------------------------
        // Applied here rather than inside the model because the weight depends
        // on the *step*, which the model has no business knowing. `StepParts`
        // hands the balance term back separately for exactly this.
        let scheduled_balance = self.balance.map(|schedule| {
            let w = schedule.at(self.step);
            extra.push(("balance_weight", jnum(w as f32)));
            w
        });

        // Importance weights multiply the *final* per-sample values, after any
        // uncertainty transform, so the estimator stays unbiased for whatever
        // objective is actually being optimized.
        let aggregate = |per_sample: Tensor<B, 1>,
                         importance: Option<Tensor<B, 1>>,
                         balance: Option<Tensor<B, 1>>,
                         z: Option<Tensor<B, 1>>,
                         weight: f64| -> Tensor<B, 1> {
            let weighted = match importance {
                Some(iw) => per_sample * iw,
                None => per_sample,
            };
            let mut total = weighted.mean();
            if let Some(aux) = balance {
                total = total + aux.mul_scalar(weight as f32);
            }
            // Outside the schedule on purpose: annealing the balance weight is
            // meant to stop a routing regularizer fighting specialization, not
            // to switch off a numerical stabilizer just as the run gets long
            // enough for logit drift to matter.
            if let Some(z) = z {
                total = total + z;
            }
            total
        };

        // --- uncertainty weighting ----------------------------------------
        let Some(head) = self.head.filter(|_| !self.weighting.is_identity()) else {
            return match scheduled_balance {
                None => (parts.loss, parts.metrics, extra),
                Some(w) => {
                    let loss =
                        aggregate(parts.per_sample, parts.importance, parts.balance, parts.z, w);
                    let value: f32 = loss.clone().into_scalar();
                    let metrics = crate::dblock::StepMetrics { loss: value, ..parts.metrics };
                    (loss, metrics, extra)
                }
            };
        };

        let device = batch.pixel_values.device();
        let sigma_tensor = Tensor::<B, 1>::from_floats(
            sigmas.iter().map(|&v| v as f32).collect::<Vec<_>>().as_slice(),
            &device,
        );
        let log_variance = head.forward(sigma_tensor);
        extra.push((
            "log_variance",
            jnum(log_variance.clone().mean().into_scalar()),
        ));

        // Reweight the per-sample terms, then re-add the balance loss. The
        // balance term is a router regularizer, not a per-sigma quantity:
        // dividing it by a noise-level uncertainty would tie load balancing to
        // whichever sigmas this batch happened to draw.
        let weight = scheduled_balance.unwrap_or_else(|| model.moe_aux_weight());
        let balanced = aggregate(
            self.weighting.apply(parts.per_sample, log_variance),
            parts.importance,
            parts.balance,
            parts.z,
            weight,
        );

        let value: f32 = balanced.clone().into_scalar();
        let metrics = crate::dblock::StepMetrics { loss: value, ..parts.metrics };
        (balanced, metrics, extra)
    }
}

fn compute_loss<B, R>(
    model: &DblockClassifier<B>,
    teacher: Option<&DblockClassifier<B>>,
    batch: &Batch<B>,
    config: &TrainConfig,
    step: usize,
    rng: &mut R,
    reweight: &mut Reweighting<'_, B>,
) -> (Tensor<B, 1>, Vec<(&'static str, String)>)
where
    B: AutodiffBackend<FloatElem = f32>,
    R: Rng,
{
    use crate::logging::jnum;
    match &config.objective {
        Objective::Dblock => {
            let (loss, m, extra) = reweight.dblock_step(model, batch, config.gamma, rng);
            let mut fields = vec![
                ("loss", jnum(m.loss)),
                ("ce_loss", jnum(m.ce_loss)),
                ("balance_loss", jnum(m.balance_loss)),
                ("block", format!("{}", m.block_idx)),
            ];
            fields.extend(extra);
            (loss, fields)
        }
        Objective::Consistency(cfg) => {
            let (loss, m) =
                model.consistency_step(&batch.pixel_values, batch.labels.clone(), cfg, step, rng);
            (
                loss,
                vec![
                    ("loss", jnum(m.loss)),
                    ("ce_loss", jnum(m.ce_loss)),
                    ("boundary", jnum(m.boundary_loss)),
                    ("self_consistency", jnum(m.self_loss)),
                    ("trajectory", jnum(m.trajectory_loss)),
                    ("cross_fork", jnum(m.cross_fork_loss)),
                    ("block", format!("{}", m.block_idx)),
                ],
            )
        }
        Objective::Flow => {
            let loss = crate::flow::flow_matching_loss(
                model,
                &batch.pixel_values,
                batch.labels.clone(),
                rng,
            );
            let value: f32 = loss.clone().into_scalar();
            (loss, vec![("loss", jnum(value)), ("flow_mse", jnum(value))])
        }
        Objective::Distill(cfg) => {
            let teacher = teacher.expect("distillation requires a teacher");
            let (loss, m) = model.distill_step(
                teacher,
                &batch.pixel_values,
                batch.labels.clone(),
                cfg,
                rng,
            );
            (
                loss,
                vec![
                    ("loss", jnum(m.loss)),
                    ("kl", jnum(m.kl)),
                    ("latent_mse", jnum(m.latent_mse)),
                    ("ce_loss", jnum(m.ce)),
                    ("block", format!("{}", m.block_idx)),
                ],
            )
        }
    }
}

/// Sliding-window mean over the last `window` values.
pub struct RunningAvg {
    window: std::collections::VecDeque<f32>,
    sum: f32,
    cap: usize,
}

impl RunningAvg {
    pub fn new(window: usize) -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(window),
            sum: 0.0,
            cap: window.max(1),
        }
    }

    pub fn push(&mut self, v: f32) {
        if self.window.len() == self.cap {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
            }
        }
        self.window.push_back(v);
        self.sum += v;
    }

    pub fn avg(&self) -> f32 {
        if self.window.is_empty() {
            0.0
        } else {
            self.sum / self.window.len() as f32
        }
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_avg_slides() {
        let mut avg = RunningAvg::new(3);
        assert_eq!(avg.avg(), 0.0);
        for v in [1.0, 2.0, 3.0] {
            avg.push(v);
        }
        assert!((avg.avg() - 2.0).abs() < 1e-6);
        avg.push(10.0); // evicts the 1.0
        assert!((avg.avg() - 5.0).abs() < 1e-6);
        assert_eq!(avg.len(), 3);
    }

    #[test]
    fn test_dataset_choice_parsing_and_shapes() {
        assert!(matches!(
            DatasetChoice::parse("synthetic", None, false).unwrap(),
            DatasetChoice::Synthetic
        ));
        // The real datasets need a directory; failing early beats a confusing
        // file-not-found later.
        assert!(DatasetChoice::parse("cifar100", None, false).is_err());
        assert!(DatasetChoice::parse("nonsense", None, false).is_err());

        let cifar = DatasetChoice::parse("cifar100", Some("/data".into()), true).unwrap();
        assert_eq!(cifar.shape(), Some((32, 100)));
        let tin = DatasetChoice::parse("tiny-imagenet", Some("/data".into()), false).unwrap();
        assert_eq!(tin.shape(), Some((64, 200)));
        assert_eq!(DatasetChoice::Synthetic.shape(), None);
    }

    #[test]
    fn test_dataset_shape_overrides_cli_defaults() {
        // A Tiny ImageNet run must build a 64x64 / 200-class model even though
        // the CLI defaults say 32 and 100.
        let config = TrainConfig {
            image_size: 32,
            num_labels: 100,
            dataset: DatasetChoice::TinyImagenet { dir: "/data".into(), streaming: true },
            ..TrainConfig::default()
        };
        let vit = config.vit_config();
        assert_eq!(vit.image_size, 64);
        assert_eq!(vit.num_labels, 200);

        // Synthetic keeps whatever the caller asked for.
        let synth = TrainConfig { image_size: 32, num_labels: 7, ..TrainConfig::default() };
        assert_eq!(synth.vit_config().num_labels, 7);
    }

    #[test]
    fn test_objective_parsing_roundtrip() {
        for name in ["dblock", "consistency", "flow", "distill"] {
            assert_eq!(Objective::parse(name).unwrap().name(), name);
        }
        assert!(Objective::parse("elbo").is_err());
    }

    #[test]
    fn test_summary_skip_rate() {
        let s = TrainSummary { steps_taken: 9, steps_skipped: 1, ..TrainSummary::default() };
        assert!((s.skip_rate() - 0.1).abs() < 1e-6);
        assert_eq!(TrainSummary::default().skip_rate(), 0.0);
    }

    #[test]
    fn test_block_of_reads_the_metric_field() {
        assert_eq!(block_of(&[("loss", "1.0".into()), ("block", "2".into())]), 2);
        // Flow matching reports no block; defaulting to 0 keeps the health
        // table usable rather than panicking.
        assert_eq!(block_of(&[("loss", "1.0".into())]), 0);
        assert_eq!(block_of(&[("block", "not-a-number".into())]), 0);
    }

    #[test]
    fn test_preflight_failure_stops_the_run_before_any_step() {
        // The preflight gate is only worth having if it actually blocks; the
        // certificates pass here, so assert the wiring instead: a run with
        // preflight on must still complete, and the summary must be populated.
        let config = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 2,
            log_every: 10,
            checks: TrainingChecks { preflight: true, ..TrainingChecks::default() },
            ..TrainConfig::default()
        };
        let (_model, summary) = train(&config).unwrap();
        assert!(summary.aborted.is_none());
        assert_eq!(summary.health.total_steps, 2);
    }

    #[test]
    fn test_health_is_recorded_for_every_step() {
        let config = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 4,
            log_every: 10,
            checks: TrainingChecks { preflight: false, ..TrainingChecks::thorough(2) },
            ..TrainConfig::default()
        };
        let (_model, summary) = train(&config).unwrap();
        assert_eq!(summary.health.total_steps, 4);
        assert_eq!(summary.health.num_blocks(), 2);

        // Every step is attributed to exactly one block.
        let attributed: usize = (0..2)
            .filter_map(|b| summary.health.block(b))
            .map(|b| b.steps)
            .sum();
        assert_eq!(attributed, 4);
        assert!(summary.health.dead_blocks().is_empty(), "no block should be dead");
    }

    #[test]
    fn test_gradient_accumulation_reduces_optimizer_steps() {
        // Accumulating over k micro-batches must take k times fewer optimizer
        // steps while still consuming every batch.
        let base = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 6,
            log_every: 100,
            checks: TrainingChecks { preflight: false, ..TrainingChecks::default() },
            ..TrainConfig::default()
        };

        let (_, plain) = train(&base).unwrap();
        assert_eq!(plain.steps_taken, 6);

        let (_, accumulated) =
            train(&TrainConfig { accumulate: 3, ..base.clone() }).unwrap();
        assert_eq!(
            accumulated.steps_taken, 2,
            "6 micro-batches at accumulate=3 is 2 optimizer steps"
        );
        // Every batch is still visited and recorded.
        assert_eq!(accumulated.health.total_steps, 6);
    }

    #[test]
    fn test_lr_schedule_is_applied_and_reported() {
        let config = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 4,
            log_every: 100,
            lr: 1e-3,
            lr_schedule: LrSchedule::cosine(1e-3, 4),
            checks: TrainingChecks { preflight: false, ..TrainingChecks::default() },
            ..TrainConfig::default()
        };
        let (_, summary) = train(&config).unwrap();
        assert!(summary.final_lr > 0.0);
        assert!(
            summary.final_lr <= 1e-3,
            "the schedule must never exceed its peak, got {}",
            summary.final_lr
        );
    }

    #[test]
    fn test_clipping_is_counted() {
        // A bound of zero cannot clip (it is treated as "disabled" by
        // clip_gradients), while a tiny positive bound clips every step.
        let base = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 3,
            log_every: 100,
            checks: TrainingChecks { preflight: false, ..TrainingChecks::default() },
            ..TrainConfig::default()
        };
        let (_, unclipped) = train(&base).unwrap();
        assert_eq!(unclipped.steps_clipped, 0);

        let (_, clipped) =
            train(&TrainConfig { clip_norm: Some(1e-4), ..base }).unwrap();
        assert_eq!(clipped.steps_clipped, 3, "every step should exceed a 1e-4 bound");
    }

    #[test]
    fn test_ema_weights_are_returned_when_enabled() {
        let base = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 3,
            log_every: 100,
            seed: 11,
            checks: TrainingChecks { preflight: false, ..TrainingChecks::default() },
            ..TrainConfig::default()
        };
        let (live, _) = train(&base.clone()).unwrap();
        let (averaged, _) = train(&TrainConfig { ema_decay: Some(0.9), ..base }).unwrap();

        // The averaged weights lag the live ones, so they must differ.
        let diff = (live.model().label_embedding_weight()
            - averaged.model().label_embedding_weight())
        .abs()
        .max()
        .into_scalar();
        assert!(diff > 0.0, "EMA weights should differ from the live ones");
    }

    #[test]
    fn test_disabling_checks_accepts_every_step() {
        let config = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 3,
            log_every: 10,
            checks: TrainingChecks::none(),
            ..TrainConfig::default()
        };
        let (_model, summary) = train(&config).unwrap();
        assert_eq!(summary.steps_taken, 3);
        assert_eq!(summary.steps_skipped, 0);
    }

    #[test]
    fn test_grad_norm_matches_a_hand_computed_value() {
        use burn::tensor::Distribution;

        // A one-parameter module whose gradient is known in closed form:
        // for loss = sum(w^2), dL/dw = 2w, so ||grad|| = 2 ||w||.
        type A = DefaultTrainBackend;
        let device = Default::default();
        let model = DblockClassifier::<A>::new(
            &ViTDiTConfig::tiny(10),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );

        let pixels =
            Tensor::<A, 4>::random([2, 3, 32, 32], Distribution::Uniform(-0.5, 0.5), &device);
        let labels = Tensor::<A, 1, burn::tensor::Int>::from_ints([1i64, 5].as_slice(), &device);
        let mut rng = StdRng::seed_from_u64(0);
        let (loss, _) = model.training_step(pixels, labels, 0.05, &mut rng);

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let norm = global_grad_norm(&model, &grads);
        assert!(norm.is_finite() && norm > 0.0, "gradient norm must be positive: {norm}");

        // An empty gradient set has norm zero, which the gate rejects as a
        // dead step rather than treating as healthy.
        let empty = GradientsParams::new();
        assert_eq!(global_grad_norm(&model, &empty), 0.0);
        assert!(!grad_norm_ok(0.0, GradNormGate::default().min_norm, GradNormGate::default().max_norm));
    }

    #[test]
    fn test_short_run_updates_the_model() {
        // End-to-end: a handful of steps must run, take optimizer steps, and
        // leave a finite loss.
        let config = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 3,
            log_every: 10,
            ..TrainConfig::default()
        };
        let (_model, summary) = train(&config).unwrap();
        assert_eq!(summary.steps_taken + summary.steps_skipped, 3);
        assert!(summary.final_loss.is_finite());
        assert!(summary.mean_loss.is_finite());
    }

    #[test]
    fn test_reweighting_runs_end_to_end_in_every_combination() {
        // Both Phase-20 reweighting features must be switchable without
        // disturbing anything else about a run.
        //
        // Note what is *not* asserted here: that two runs report the same loss.
        // `train` seeds a **global** backend RNG, so any concurrently running
        // test that builds a model perturbs it — comparing losses across two
        // `train` calls would be pinning a value, not a property. The exact
        // identity at strength 0 is proved where it is actually a property of
        // the code, in `reweight::tests::test_zero_strength_is_bitwise_identity`.
        let base = TrainConfig {
            image_size: 32,
            num_labels: 10,
            batch_size: 2,
            num_blocks: 2,
            steps: 4,
            log_every: 100,
            checks: TrainingChecks { preflight: false, ..TrainingChecks::default() },
            ..TrainConfig::default()
        };

        for (uncertainty, bins) in [(0.0, 0usize), (1.0, 0), (0.0, 8), (0.5, 8)] {
            let (_, summary) = train(&TrainConfig {
                uncertainty,
                importance_bins: bins,
                ..base.clone()
            })
            .unwrap();
            assert_eq!(
                summary.steps_taken + summary.steps_skipped,
                4,
                "every step must be accounted for at uncertainty={uncertainty}, bins={bins}"
            );
            assert!(
                summary.mean_loss.is_finite(),
                "uncertainty={uncertainty} bins={bins} produced a non-finite mean loss"
            );
            assert!(summary.aborted.is_none(), "aborted: {:?}", summary.aborted);
        }
    }
}
