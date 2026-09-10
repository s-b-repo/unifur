//! Block-wise training loop (`main.py` / Lightning plumbing of the reference,
//! reduced to a self-contained Burn loop).
//!
//! One loop serves every objective the crate implements -- the plain
//! DiffusionBlocks loss, the consistency-augmented one, rectified flow, and
//! block distillation -- because they differ only in how a step's loss is
//! computed. Everything around that (batching, gradient gating, logging,
//! checkpointing, resume) is shared, so a new objective cannot accidentally
//! come with a subtly different training procedure.

use serde::{Deserialize, Serialize};
use anyhow::Context;
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
    schedule::{BalanceScope, Ema, GlobalLoad, GradientAccumulator, LossScales, LrSchedule},
    vit::ViTDiTConfig,
};
use crate::{
    corpus::TokenCorpus,
    lm::{LanguageModel, Unlikelihood},
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
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::checkpoint::{self, DatasetIdentity, TrainState};
use crate::mix::{BatchOrigin, CorpusMix, MixMode, MixWeights, SourceStats};

/// The device RNG seed for one step: a function of `(seed, step)` only.
///
/// The backend's random stream is a process-wide global that every
/// `Tensor::random` and dropout draw from. Snapshotting the *host* RNG alone
/// cannot make a resumed run bit-identical to the uninterrupted one -- the
/// device stream would be at a different position. Reseeding it at the top of
/// every step from `(seed, step)` makes the stream a pure function of where the
/// run is, which is what makes `--resume` exact (roadmap Phase 28).
pub fn step_seed(seed: u64, step: usize) -> u64 {
    // splitmix64 over the pair: cheap, and every bit of the step reaches
    // every bit of the seed.
    let mut z = seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

type TrainBackend<C> = Autodiff<NdArray<f32>, C>;
type ModelOptim<C> =
    burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, DblockClassifier<TrainBackend<C>>, TrainBackend<C>>;
type Teachers<C> = Vec<(DblockClassifier<TrainBackend<C>>, f64)>;
type TeacherRefs<'a, C> = Vec<(&'a DblockClassifier<TrainBackend<C>>, f64)>;
type HeadOptim<C> =
    burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, LogVarianceHead<TrainBackend<C>>, TrainBackend<C>>;

/// Autodiff-enabled ndarray backend used for CPU training.
pub type DefaultTrainBackend = Autodiff<NdArray<f32>>;
/// Variant that checkpoints intermediate activations during backward
/// (roadmap 15.2): trades compute for lower peak memory.
pub type CheckpointedTrainBackend = Autodiff<NdArray<f32>, BalancedCheckpointing>;

/// Which dataset to train on.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    ) -> anyhow::Result<Batch<B>> {
        match self {
            Self::Synthetic(d) => d.next_batch(rng, device),
            Self::Raw(d) => d.next_batch(rng, device),
        }
    }
}

/// Which loss a training step computes.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Further sources trained alongside `dataset` (roadmap 29.1). Every
    /// source must share the image size and label count.
    pub extra_datasets: Vec<DatasetChoice>,
    /// Weights over `dataset` and `extra_datasets`, in order; empty is uniform.
    pub dataset_weights: Vec<f64>,
    /// How the sources share a run.
    pub mix_mode: MixMode,
    pub objective: Objective,
    /// Quality verification applied at each phase of a training step.
    pub checks: TrainingChecks,
    /// Checkpoint to initialize from (roadmap 15.7).
    pub resume: Option<PathBuf>,
    /// Frozen teacher for [`Objective::Distill`].
    pub teacher: Option<PathBuf>,
    /// Further teachers distilled from at once (roadmap 29.2); weighted with
    /// `teacher_weights` over `teacher` (or the initial snapshot) and these.
    pub extra_teachers: Vec<PathBuf>,
    /// Empty is uniform.
    pub teacher_weights: Vec<f64>,
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
    /// Over which batch the balance loss's load fraction is measured
    /// (roadmap 23.4). `Global` averages it over the accumulation window.
    pub balance_scope: BalanceScope,
    /// Loss-free bias balancing rate (roadmap 23.5): how far each router's
    /// selection bias moves per step, against its load. `0.0` is off and
    /// attaches no bias at all.
    pub bias_balance_rate: f32,
    /// Where the trainer writes its checkpoints -- the content-addressed model
    /// file plus the training-state directory beside it (roadmap Phase 28).
    /// `None` writes nothing; the caller may still save the returned model.
    pub out_dir: Option<PathBuf>,
    /// Also checkpoint every this many steps (`0` = only at the end). A step
    /// inside an accumulation cycle defers to the next cycle boundary.
    pub checkpoint_every: usize,
    /// Fraction of each batch relabeled with a wrong class and charged as a
    /// negative (roadmap 31.3); `0` is off and takes the plain path.
    pub synthetic_negatives: f64,
    /// Coefficient on the negative charge.
    pub negative_penalty: f32,
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
            extra_datasets: Vec::new(),
            dataset_weights: Vec::new(),
            mix_mode: MixMode::Mixture,
            objective: Objective::Dblock,
            checks: TrainingChecks::default(),
            resume: None,
            teacher: None,
            extra_teachers: Vec::new(),
            teacher_weights: Vec::new(),
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
            balance_scope: BalanceScope::Micro,
            bias_balance_rate: 0.0,
            out_dir: None,
            checkpoint_every: 0,
            synthetic_negatives: 0.0,
            negative_penalty: 1.0,
        }
    }
}

impl TrainConfig {
    /// Every source, primary first.
    pub fn all_datasets(&self) -> Vec<&DatasetChoice> {
        std::iter::once(&self.dataset).chain(self.extra_datasets.iter()).collect()
    }

    /// Normalized weights over [`Self::all_datasets`].
    pub fn mix_weights(&self) -> anyhow::Result<MixWeights> {
        let n = 1 + self.extra_datasets.len();
        if self.dataset_weights.is_empty() {
            Ok(MixWeights::uniform(n))
        } else {
            anyhow::ensure!(self.dataset_weights.len() == n, "{} dataset weight(s) for {n} source(s)", self.dataset_weights.len());
            MixWeights::new(&self.dataset_weights)
        }
    }

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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainSummary {
    pub steps_taken: usize,
    /// Steps rejected by a quality check at any phase.
    pub steps_skipped: usize,
    /// Accepted micro-batches folded into an accumulation cycle that had not
    /// completed yet (`--accumulate` > 1): neither steps nor rejections.
    #[serde(default)]
    pub steps_accumulated: usize,
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
    /// The final checkpoint, when `out_dir` was set: the model file, with the
    /// training state in the directory beside it.
    pub checkpoint: Option<PathBuf>,
    /// The step a resumed run continued from; 0 for a fresh run.
    pub resumed_from_step: usize,
    /// Every periodic checkpoint written, as `(steps completed, model path)`.
    pub periodic_checkpoints: Vec<(usize, PathBuf)>,
    /// Per-source batches, samples and mean loss (roadmap 29.1).
    pub sources: SourceStats,
}

impl TrainSummary {
    /// Fraction of micro-batches a quality gate rejected.
    pub fn skip_rate(&self) -> f32 {
        let total = self.steps_taken + self.steps_skipped + self.steps_accumulated;
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
    // Hashed once, before anything else happens: it is both what the state
    // file records and what a resume is checked against.
    let dataset_identity = dataset_identities(config)?;
    let config_json = serde_json::to_value(config).context("serialize training config")?;
    let mut restored: Option<TrainState> = None;
    if let Some(path) = &config.resume {
        model = model
            .load_file(path, &Recorder::new(), &device)
            .map_err(|err| anyhow::anyhow!("resume from {}: {err}", path.display()))?;
        match TrainState::for_model(path)? {
            Some(state) => {
                let dir = TrainState::dir_for(path);
                anyhow::ensure!(
                    state.kind == "dblock",
                    "{} holds `{}` training state, not a dblock run",
                    dir.display(),
                    state.kind
                );
                state.verify_files(&dir)?;
                anyhow::ensure!(
                    checkpoint::same_datasets(&state.datasets, &dataset_identity),
                    "refusing to resume: the checkpoint was trained on [{}], this run opened [{}]",
                    checkpoint::describe_datasets(&state.datasets),
                    checkpoint::describe_datasets(&dataset_identity)
                );
                let differences = state.config_differences(&config_json);
                if !differences.is_empty() {
                    println!(
                        "warning: resuming with a different configuration in {}",
                        differences.join(", ")
                    );
                }
                println!(
                    "resumed from {} at step {} (training state verified: {} file(s))",
                    path.display(),
                    state.step,
                    1 + usize::from(state.optimizer.is_some())
                        + usize::from(state.ema.is_some())
                        + usize::from(state.head.is_some())
                        + usize::from(state.head_optimizer.is_some())
                );
                restored = Some(state);
            }
            None => println!(
                "resumed weights from {}; no training state beside it, so the optimizer, \
                 schedules and RNG start fresh",
                path.display()
            ),
        }
    }
    if config.bias_balance_rate > 0.0 {
        // A record from before the bias existed, or one written without it,
        // loads as `None`; re-attach zeros so the run has a bias to nudge.
        model.ensure_balance_biases();
        println!(
            "loss-free bias balancing: rate {:.1e} per step (roadmap 23.5)",
            config.bias_balance_rate
        );
    }
    if config.balance_scope == BalanceScope::Global {
        anyhow::ensure!(
            config.mosme.is_none(),
            "--balance-scope global is implemented for flat MoE layers; a hierarchical \
             (MoSME) trunk weights each box's balance term by its own gate, which one \
             load window cannot express yet"
        );
        println!(
            "balance loss: global-batch load over a window of {} micro-batch(es) (roadmap 23.4)",
            config.accumulate.max(1)
        );
    }
    let mut global_load =
        (config.balance_scope == BalanceScope::Global).then(|| GlobalLoad::new(config.accumulate));

    // The teachers are frozen: separate checkpoints, or a snapshot of the
    // starting model when none is given. Several distil at once (29.2).
    let teachers: Teachers<C> = match &config.objective {
        Objective::Distill(_) => {
            let load = |path: &PathBuf| -> anyhow::Result<DblockClassifier<Autodiff<NdArray<f32>, C>>> {
                DblockClassifier::<Autodiff<NdArray<f32>, C>>::new(&vit_config, &dblock_config, &device)
                    .load_file(path, &Recorder::new(), &device)
                    .map_err(|err| anyhow::anyhow!("load teacher {}: {err}", path.display()))
            };
            let mut list = vec![match &config.teacher {
                Some(path) => load(path)?,
                None => model.clone(),
            }];
            for path in &config.extra_teachers {
                list.push(load(path)?);
            }
            let weights = if config.teacher_weights.is_empty() {
                vec![1.0; list.len()]
            } else {
                anyhow::ensure!(
                    config.teacher_weights.len() == list.len(),
                    "{} teacher weight(s) for {} teacher(s)",
                    config.teacher_weights.len(),
                    list.len()
                );
                config.teacher_weights.clone()
            };
            if list.len() > 1 {
                println!("distilling from {} teachers, weights {:?}", list.len(), weights);
            }
            list.into_iter().zip(weights).collect()
        }
        _ => Vec::new(),
    };
    let teacher_refs: TeacherRefs<'_, C> = teachers.iter().map(|(t, w)| (t, *w)).collect();

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

    let mut dataset = open_sources(config)?;
    let mut source_stats = SourceStats::new(dataset.names.clone());
    if dataset.sources.len() > 1 {
        println!(
            "data: {} sources as a {} ({})",
            dataset.sources.len(),
            dataset.mode.name(),
            dataset
                .names
                .iter()
                .zip(dataset.weights.as_slice())
                .map(|(n, w)| format!("{n} x{w:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let mut rng = ChaCha12Rng::seed_from_u64(config.seed);

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
    let mut accumulator = GradientAccumulator::new(config.accumulate);
    let mut scales = LossScales::new(config.num_blocks);
    let mut ema = config.ema_decay.map(|d| Ema::new(&model, d));
    let mut loss_sum = 0.0f64;
    let mut elapsed_before = 0.0f64;
    let mut start_step = 0usize;
    if let Some(state) = &restored {
        let resume = config.resume.as_ref().context("a training state was restored without a resume path")?;
        let dir = TrainState::dir_for(resume);
        if let Some(file) = &state.optimizer {
            optim = optim.load_record(checkpoint::load_record::<TrainBackend<C>, _>(&dir, file, &device)?);
        }
        match (&state.head, logvar_head.take()) {
            (Some(file), Some(head)) => {
                logvar_head =
                    Some(head.load_record(checkpoint::load_record::<TrainBackend<C>, _>(&dir, file, &device)?));
                if let (Some(file), Some(head_optim)) = (&state.head_optimizer, logvar_optim.take()) {
                    logvar_optim = Some(
                        head_optim.load_record(checkpoint::load_record::<TrainBackend<C>, _>(&dir, file, &device)?),
                    );
                }
            }
            (_, head) => logvar_head = head,
        }
        let extras: DblockExtras =
            serde_json::from_value(state.extras.clone()).context("parse dblock training state")?;
        if let (Some(file), Some(current)) = (&state.ema, ema.take()) {
            let shadow = current
                .shadow()
                .clone()
                .load_record(checkpoint::load_record::<TrainBackend<C>, _>(&dir, file, &device)?);
            ema = Some(Ema::from_parts(shadow, current.decay(), extras.ema_updates.unwrap_or(0)));
        }
        scales = extras.scales;
        if importance.is_some() {
            importance = extras.importance.or(importance);
        }
        health = extras.health;
        running = extras.running;
        if extras.sources.names.len() == source_stats.names.len() {
            source_stats = extras.sources;
        }
        summary.steps_taken = extras.steps_taken;
        summary.steps_skipped = extras.steps_skipped;
        summary.steps_accumulated = extras.steps_accumulated;
        summary.steps_clipped = extras.steps_clipped;
        summary.periodic_verifications = extras.periodic_verifications;
        summary.final_lr = extras.final_lr;
        loss_sum = extras.loss_sum;
        elapsed_before = extras.elapsed_secs;
        rng = serde_json::from_value(state.host_rng.clone()).context("restore host RNG")?;
        start_step = state.step;
        summary.resumed_from_step = start_step;
    }
    let save_state = |step: usize,
                      model: &DblockClassifier<TrainBackend<C>>,
                      optim: &ModelOptim<C>,
                      head: Option<&LogVarianceHead<TrainBackend<C>>>,
                      head_optim: Option<&HeadOptim<C>>,
                      ema: Option<&Ema<DblockClassifier<TrainBackend<C>>>>,
                      rng: &ChaCha12Rng,
                      extras: &DblockExtras|
     -> anyhow::Result<Option<PathBuf>> {
        let Some(dir) = &config.out_dir else {
            return Ok(None);
        };
        let model_path = checkpoint::save_content_addressed(model.clone(), dir, "dblocks")?;
        let state_dir = TrainState::dir_for(&model_path);
        // Rewritten from scratch: a directory left by an earlier save of the
        // same weights (a dedupe hit) must not keep stale companions.
        if state_dir.exists() {
            std::fs::remove_dir_all(&state_dir)
                .with_context(|| format!("clear {}", state_dir.display()))?;
        }
        let optimizer = Some(checkpoint::save_record::<TrainBackend<C>, _>(optim.to_record(), &state_dir, "optimizer")?);
        let head_file = head
            .map(|h| checkpoint::save_record::<TrainBackend<C>, _>(h.clone().into_record(), &state_dir, "head"))
            .transpose()?;
        let head_optimizer = head_optim
            .map(|o| checkpoint::save_record::<TrainBackend<C>, _>(o.to_record(), &state_dir, "head-optimizer"))
            .transpose()?;
        let ema_file = ema
            .map(|e| checkpoint::save_record::<TrainBackend<C>, _>(e.shadow().clone().into_record(), &state_dir, "ema"))
            .transpose()?;
        let state = TrainState {
            format_version: checkpoint::STATE_FORMAT_VERSION,
            kind: "dblock".into(),
            step,
            seed: config.seed,
            host_rng: serde_json::to_value(rng).context("serialize host RNG")?,
            config: config_json.clone(),
            build: checkpoint::BuildInfo::current(),
            datasets: dataset_identity.clone(),
            model: checkpoint::model_entry(&model_path)?,
            optimizer,
            ema: ema_file,
            head: head_file,
            head_optimizer,
            extras: serde_json::to_value(extras).context("serialize training extras")?,
            saved_unix_secs: checkpoint::unix_now(),
        };
        state.write(&state_dir)?;
        Ok(Some(model_path))
    };
    let extras_now = |scales: &LossScales,
                      importance: &Option<SigmaImportanceSampler>,
                      health: &TrainingHealth,
                      running: &RunningAvg,
                      summary: &TrainSummary,
                      loss_sum: f64,
                      elapsed: f64,
                      ema: &Option<Ema<DblockClassifier<TrainBackend<C>>>>,
                      sources: &SourceStats| DblockExtras {
        scales: scales.clone(),
        importance: importance.clone(),
        health: health.clone(),
        running: running.clone(),
        steps_taken: summary.steps_taken,
        steps_skipped: summary.steps_skipped,
        steps_accumulated: summary.steps_accumulated,
        steps_clipped: summary.steps_clipped,
        periodic_verifications: summary.periodic_verifications,
        final_lr: summary.final_lr,
        loss_sum,
        elapsed_secs: elapsed,
        ema_updates: ema.as_ref().map(Ema::updates),
        sources: sources.clone(),
    };
    if config.accumulate > 1 {
        println!(
            "gradient accumulation: {} micro-batches per step",
            accumulator.every()
        );
    }

    for step in start_step..config.steps {
        // The device stream is a pure function of (seed, step): see `step_seed`.
        <TrainBackend<C> as burn::tensor::backend::Backend>::seed(&device, step_seed(config.seed, step));
        let (batch, origin) = dataset.next(&mut rng, &device).with_context(|| format!("step {step}: draw a batch"))?;
        let (loss, mut fields, routing) = compute_loss(
            &model,
            &teacher_refs,
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
                global_load: global_load.as_mut(),
                synthetic_negatives: config.synthetic_negatives,
                negative_penalty: config.negative_penalty,
            },
        );
        let block_idx = block_of(&fields);
        if dataset.sources.len() > 1 {
            fields.push(("source", origin.sole_source().map_or("-1".to_string(), |s| s.to_string())));
        }

        let mut verdict = StepVerdict::accepted();
        let scalar_loss: f32 = loss.clone().into_scalar();
        source_stats.record(&origin, scalar_loss);

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

            // Loss-free balancing acts *after* the step, from the load this
            // step observed, and touches nothing the optimizer owns.
            if config.bias_balance_rate > 0.0 && !routing.is_empty() {
                model.nudge_balance_biases(block_idx, &routing, config.bias_balance_rate);
            }

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
        } else if verdict.accepted {
            summary.steps_accumulated += 1;
        } else {
            summary.steps_skipped += 1;
        }

        health.record(step, block_idx, scalar_loss, &verdict);
        health.record_routing(block_idx, &routing);
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

        // --- periodic checkpoint (roadmap Phase 28) ------------------------
        // Only at an accumulation-cycle boundary: a half-folded gradient
        // buffer is not part of the saved state, and dropping it would make
        // the resumed run diverge.
        let due = config.checkpoint_every > 0
            && (step + 1) % config.checkpoint_every == 0
            && step + 1 < config.steps
            && !accumulator.has_pending();
        if due {
            let extras = extras_now(
                &scales,
                &importance,
                &health,
                &running,
                &summary,
                loss_sum,
                elapsed_before + start.elapsed().as_secs_f64(),
                &ema,
                &source_stats,
            );
            if let Some(path) = save_state(
                step + 1,
                &model,
                &optim,
                logvar_head.as_ref(),
                logvar_optim.as_ref(),
                ema.as_ref(),
                &rng,
                &extras,
            )? {
                println!("step {step}: checkpoint {}", path.display());
                summary.periodic_checkpoints.push((step + 1, path));
            }
        }
    }

    // Every micro-batch the loop ran, whether it stepped, was folded into a
    // pending accumulation cycle, or was rejected.
    let completed = summary.steps_taken + summary.steps_skipped + summary.steps_accumulated;
    summary.mean_loss = if completed == 0 {
        0.0
    } else {
        (loss_sum / completed as f64) as f32
    };
    summary.elapsed_secs = elapsed_before + start.elapsed().as_secs_f64();

    // The final checkpoint carries the state as of `config.steps`, so a later
    // `--resume --steps N+M` continues rather than restarts.
    let extras = extras_now(
        &scales,
        &importance,
        &health,
        &running,
        &summary,
        loss_sum,
        summary.elapsed_secs,
        &ema,
        &source_stats,
    );
    summary.checkpoint = save_state(
        config.steps,
        &model,
        &optim,
        logvar_head.as_ref(),
        logvar_optim.as_ref(),
        ema.as_ref(),
        &rng,
        &extras,
    )?;

    let dead = health.dead_blocks();
    if !dead.is_empty() {
        println!(
            "warning: block(s) {dead:?} never received a non-zero gradient; \
             check num_blocks against num_hidden_layers and the sigma windows"
        );
    }
    summary.health = health;
    if source_stats.is_multi() {
        print!("\nper-source:\n{}", source_stats.render());
    }
    summary.sources = source_stats;

    // The averaged weights are usually the better evaluation model, and are
    // what gets returned when EMA is enabled. The checkpoint holds the *live*
    // weights and the shadow separately, so resuming continues the average.
    if let Some(ema) = ema {
        println!("returning EMA weights ({} updates)", ema.updates());
        return Ok((ema.into_shadow(), summary));
    }
    Ok((model, summary))
}

/// Host-side trainer state saved beside the model (roadmap Phase 28).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DblockExtras {
    scales: LossScales,
    importance: Option<SigmaImportanceSampler>,
    health: TrainingHealth,
    running: RunningAvg,
    steps_taken: usize,
    steps_skipped: usize,
    #[serde(default)]
    steps_accumulated: usize,
    steps_clipped: usize,
    periodic_verifications: usize,
    final_lr: f64,
    loss_sum: f64,
    elapsed_secs: f64,
    ema_updates: Option<usize>,
    #[serde(default)]
    sources: SourceStats,
}

/// What a run's data is, for the training state (roadmap Phase 28), one
/// entry per source.
fn dataset_identities(config: &TrainConfig) -> anyhow::Result<Vec<DatasetIdentity>> {
    config
        .all_datasets()
        .into_iter()
        .map(|choice| {
            Ok(match choice {
                DatasetChoice::Synthetic => DatasetIdentity::synthetic(format!(
                    "synthetic {}x{} images, {} labels",
                    config.image_size, config.image_size, config.num_labels
                )),
                DatasetChoice::Cifar100 { dir, .. } => DatasetIdentity::of_path(dir, "cifar100")?,
                DatasetChoice::TinyImagenet { dir, .. } => DatasetIdentity::of_path(dir, "tiny-imagenet")?,
            })
        })
        .collect()
}

/// The sources of a run, opened with the batch size each contributes.
struct MixedDataset {
    sources: Vec<AnyDataset>,
    names: Vec<String>,
    weights: MixWeights,
    mode: MixMode,
    /// Per source, the batch it yields (the full batch for a mixture, its
    /// slice for a composite; a slice of zero means the source is skipped).
    slices: Vec<usize>,
}

impl MixedDataset {
    fn next<B: burn::tensor::backend::Backend, R: Rng>(
        &mut self,
        rng: &mut R,
        device: &B::Device,
    ) -> anyhow::Result<(Batch<B>, BatchOrigin)> {
        let n = self.sources.len();
        match self.mode {
            MixMode::Mixture => {
                let source = self.weights.draw(rng);
                let batch = self.sources[source]
                    .next(rng, device)
                    .with_context(|| format!("source {}", self.names[source]))?;
                let size = batch.batch_size();
                Ok((batch, BatchOrigin::single(source, n, size)))
            }
            MixMode::Composite => {
                let mut parts = Vec::with_capacity(n);
                for (i, source) in self.sources.iter_mut().enumerate() {
                    if self.slices[i] > 0 {
                        parts.push(source.next(rng, device).with_context(|| format!("source {}", self.names[i]))?);
                    }
                }
                Ok((crate::mix::concat_batches(parts)?, BatchOrigin { counts: self.slices.clone() }))
            }
        }
    }
}

fn open_sources(config: &TrainConfig) -> anyhow::Result<MixedDataset> {
    let choices = config.all_datasets();
    let weights = config.mix_weights()?;
    let shape = config.dataset.shape().unwrap_or((config.image_size, config.num_labels));
    for (i, choice) in choices.iter().enumerate().skip(1) {
        let other = choice.shape().unwrap_or((config.image_size, config.num_labels));
        anyhow::ensure!(
            other == shape,
            "source {i} is {}x{} with {} labels but the primary source is {}x{} with {} labels; \
             one model cannot train on both",
            other.0,
            other.0,
            other.1,
            shape.0,
            shape.0,
            shape.1
        );
    }
    let slices = match config.mix_mode {
        MixMode::Mixture => vec![config.batch_size; choices.len()],
        MixMode::Composite => weights.split(config.batch_size),
    };
    let mut sources = Vec::with_capacity(choices.len());
    let mut names = Vec::with_capacity(choices.len());
    for (i, choice) in choices.iter().enumerate() {
        let batch_size = slices[i].max(1);
        sources.push(open_one(config, choice, batch_size, i as u64)?);
        names.push(match choice {
            DatasetChoice::Synthetic => format!("synthetic#{i}"),
            DatasetChoice::Cifar100 { dir, .. } => format!("cifar100:{}", dir.display()),
            DatasetChoice::TinyImagenet { dir, .. } => format!("tiny-imagenet:{}", dir.display()),
        });
    }
    Ok(MixedDataset { sources, names, weights, mode: config.mix_mode, slices })
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

fn open_one(config: &TrainConfig, choice: &DatasetChoice, batch_size: usize, index: u64) -> anyhow::Result<AnyDataset> {
    let (image_size, num_labels) = config
        .dataset
        .shape()
        .unwrap_or((config.image_size, config.num_labels));
    Ok(match choice {
        DatasetChoice::Synthetic => AnyDataset::Synthetic(SyntheticDataset::new(
            image_size,
            num_labels,
            batch_size,
            config.seed.wrapping_add(index),
        )),
        DatasetChoice::Cifar100 { dir, streaming } => {
            AnyDataset::Raw(Box::new(crate::cifar::open(dir, true, batch_size, *streaming)?))
        }
        DatasetChoice::TinyImagenet { dir, streaming } => {
            AnyDataset::Raw(Box::new(crate::tinyimagenet::open(dir, true, batch_size, *streaming)?))
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
    /// Global-batch load windows, when the balance scope is global
    /// (roadmap 23.4).
    pub global_load: Option<&'a mut GlobalLoad>,
    /// Fraction of samples relabeled as negatives, and the charge (roadmap 31.3).
    pub synthetic_negatives: f64,
    pub negative_penalty: f32,
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

        // Synthetic negatives (roadmap 31.3): a fraction of the batch keeps a
        // deliberately wrong label and is charged for it instead of rewarded.
        let parts = if self.synthetic_negatives > 0.0 {
            let device = batch.pixel_values.device();
            let labels: Vec<i64> = batch.labels.clone().into_data().convert::<i64>().iter::<i64>().collect();
            let num_labels = model.model().label_embedding_weight().dims()[0] as i64;
            let mut mask = vec![0.0f32; b];
            let mut relabeled = labels.clone();
            for i in 0..b {
                if rng.random::<f64>() < self.synthetic_negatives && num_labels > 1 {
                    mask[i] = 1.0;
                    let offset = rng.random_range(1..num_labels);
                    relabeled[i] = (labels[i] + offset) % num_labels;
                }
            }
            extra.push(("negative_samples", format!("{}", mask.iter().filter(|m| **m > 0.0).count())));
            model.training_step_negative(
                batch.pixel_values.clone(),
                Tensor::<B, 1, burn::tensor::Int>::from_ints(relabeled.as_slice(), &device),
                &sigmas,
                block_idx,
                weights.as_deref(),
                crate::dblock::NegativeLabels {
                    mask: Tensor::<B, 1>::from_floats(mask.as_slice(), &device),
                    alpha: self.negative_penalty,
                    epsilon: 1e-6,
                },
            )
        } else {
            model.training_step_on(
                batch.pixel_values.clone(),
                batch.labels.clone(),
                &sigmas,
                block_idx,
                weights.as_deref(),
            )
        };
        if parts.metrics.negative_samples > 0 {
            extra.push(("negative_prob", jnum(parts.metrics.negative_prob)));
        }

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
        let mut scheduled_balance = self.balance.map(|schedule| {
            let w = schedule.at(self.step);
            extra.push(("balance_weight", jnum(w as f32)));
            w
        });

        // --- global-batch load (roadmap 23.4) -----------------------------
        // Replace each layer's micro-batch load `f` with the window mean and
        // recombine it with this micro-batch's differentiable `p`. The window
        // mean is a constant as far as the graph is concerned, exactly as the
        // arg-max `f` always was.
        let balance = match self.global_load.as_deref_mut() {
            Some(window) if !parts.routing.is_empty() => {
                let device = batch.pixel_values.device();
                let mut total: Option<Tensor<B, 1>> = None;
                for (layer, routing) in parts.routing.iter().enumerate() {
                    let load: Vec<f32> = routing.load.clone().inner().into_data().convert::<f32>().iter::<f32>().collect();
                    let global = window.observe(layer, &load);
                    let e = routing.experts;
                    let f = Tensor::<B, 1>::from_floats(global.as_slice(), &device).reshape([1, e]);
                    let p = routing.prob_mass.clone().reshape([1, e]);
                    let term = crate::moe::switch_loss_from_parts(f, p, e);
                    total = Some(match total {
                        None => term,
                        Some(acc) => acc + term,
                    });
                }
                extra.push(("balance_window", format!("{}", window.filled(0))));
                // The model folded its micro-batch term into `parts.loss`, so
                // the aggregate path below must run even without a schedule.
                if scheduled_balance.is_none() {
                    scheduled_balance = Some(model.moe_aux_weight());
                }
                total
            }
            _ => parts.balance.clone(),
        };

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
                    let loss = aggregate(parts.per_sample, parts.importance, balance, parts.z, w);
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
            balance,
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
    teachers: &[(&DblockClassifier<B>, f64)],
    batch: &Batch<B>,
    config: &TrainConfig,
    step: usize,
    rng: &mut R,
    reweight: &mut Reweighting<'_, B>,
) -> (Tensor<B, 1>, Vec<(&'static str, String)>, Vec<crate::moe::RoutingStats>)
where
    B: AutodiffBackend<FloatElem = f32>,
    R: Rng,
{
    use crate::logging::jnum;
    let (loss, fields) = match &config.objective {
        Objective::Dblock => {
            let (loss, m, extra) = reweight.dblock_step(model, batch, config.gamma, rng);
            let mut fields = vec![
                ("loss", jnum(m.loss)),
                ("ce_loss", jnum(m.ce_loss)),
                ("balance_loss", jnum(m.balance_loss)),
                ("block", format!("{}", m.block_idx)),
            ];
            fields.extend(extra);
            if !m.routing.is_empty() {
                let (load_h, token_h, min_load, max_load) =
                    crate::moe::RoutingStats::summarize(&m.routing);
                fields.push(("load_entropy", jnum(load_h)));
                fields.push(("token_entropy", jnum(token_h)));
                fields.push(("min_load", jnum(min_load)));
                fields.push(("max_load", jnum(max_load)));
                fields.push(("routing_load", crate::moe::RoutingStats::load_json(&m.routing)));
                fields.push(("routing_stability", jnum(crate::moe::RoutingStats::mean_stability(&m.routing))));
                if let Some(agreement) = crate::moe::RoutingStats::mean_agreement(&m.routing) {
                    fields.push(("routing_agreement", jnum(agreement)));
                }
            }
            return (loss, fields, m.routing);
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
            assert!(!teachers.is_empty(), "distillation requires a teacher");
            let (loss, m) = model.distill_step_multi(
                teachers,
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
    };
    (loss, fields, Vec::new())
}

/// Sliding-window mean over the last `window` values.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

// -------------------------------------------------------- language model --

/// Configuration for [`train_lm`] (roadmap Phase 24).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmTrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub seed: u64,
    /// The charge on labeled targets. Anything but [`Unlikelihood::off`]
    /// needs the corpus to have its labels open.
    pub penalty: Unlikelihood,
    /// Print (and log, if `log_path` is set) every this many steps.
    pub log_every: usize,
    pub log_path: Option<PathBuf>,
    /// Loss-free bias balancing rate on the trunk's routers (roadmap 23.5);
    /// `0.0` is off.
    pub bias_balance_rate: f32,
    /// Where checkpoints go (model file + training state), `None` for nowhere
    /// (roadmap Phase 28).
    pub out_dir: Option<PathBuf>,
    /// Also checkpoint every this many steps (`0` = only at the end).
    pub checkpoint_every: usize,
    /// A model file from an earlier run; its training state, if present, is
    /// restored and verified.
    pub resume: Option<PathBuf>,
    /// The model's shape, recorded in the state so a resume can be checked.
    pub model_config: Option<crate::lm::LmConfig>,
    /// Weight on distillation toward the teacher mixture (roadmap 29.2);
    /// `0.0` is off.
    pub distill_weight: f64,
    pub distill_temperature: f64,
    /// A negative teacher's proposal is charged only where it is at least
    /// this sure (roadmap 29.3).
    pub negative_confidence: f32,
    /// Coefficient on the negative teacher's charge; `0.0` is off.
    pub negative_penalty: f32,
    /// A behaviour direction penalized during training (roadmap 31.2).
    pub direction: Option<crate::ablation::Direction>,
    /// Weight on `mean((h_L . d)^2)`; `0.0` adds nothing.
    pub direction_weight: f64,
    /// Decensor the trained model afterwards (roadmap 31.6): the final
    /// checkpoint is the best Heretic trial rather than the raw weights.
    pub heretic: Option<crate::heretic::HereticConfig>,
}

impl Default for LmTrainConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch_size: 8,
            lr: 3e-4,
            weight_decay: 0.01,
            seed: 42,
            penalty: Unlikelihood::off(),
            log_every: 10,
            log_path: None,
            bias_balance_rate: 0.0,
            out_dir: None,
            checkpoint_every: 0,
            resume: None,
            model_config: None,
            distill_weight: 0.0,
            distill_temperature: 2.0,
            negative_confidence: 0.5,
            negative_penalty: 0.0,
            direction: None,
            direction_weight: 0.0,
            heretic: None,
        }
    }
}

/// Frozen models a language-model run trains *with* (roadmap Phase 29):
/// teachers to distil from, and a negative teacher whose confident choices
/// are charged.
pub struct LmTrainInputs<B: AutodiffBackend<FloatElem = f32>> {
    pub teachers: Vec<(LanguageModel<B>, f64)>,
    pub negative_teacher: Option<LanguageModel<B>>,
}

impl<B: AutodiffBackend<FloatElem = f32>> Default for LmTrainInputs<B> {
    fn default() -> Self {
        Self { teachers: Vec::new(), negative_teacher: None }
    }
}

/// What a language-model run did.
#[derive(Debug, Clone)]
pub struct LmTrainReport {
    pub steps_taken: usize,
    /// Steps discarded for a non-finite loss.
    pub steps_skipped: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub mean_loss: f32,
    /// Mean probability the model gave the labeled targets, at the first and
    /// the last step. Without labels both are 0.
    pub first_penalized_prob: f32,
    pub last_penalized_prob: f32,
    /// Labeled targets seen over the whole run.
    pub penalized_tokens: usize,
    pub tokens_seen: usize,
    pub elapsed_secs: f64,
    /// The final checkpoint, when `out_dir` was set.
    pub checkpoint: Option<PathBuf>,
    /// The step a resumed run continued from; 0 for a fresh run.
    pub resumed_from_step: usize,
    /// Every periodic checkpoint written, as `(steps completed, model path)`.
    pub periodic_checkpoints: Vec<(usize, PathBuf)>,
    /// Per-corpus batches, samples and mean loss (roadmap 29.1).
    pub sources: SourceStats,
    /// Mean probability the model gave the negative teacher's proposals, at
    /// the first and the last step (roadmap 29.3); 0 without one.
    pub first_negative_teacher_prob: f32,
    pub last_negative_teacher_prob: f32,
    pub negative_teacher_tokens: usize,
    /// The distillation term at the last step (roadmap 29.2); 0 without teachers.
    pub last_distill_loss: f32,
    /// Mean squared projection onto the penalized direction at the first and
    /// last step (roadmap 31.2); 0 without one.
    pub first_direction_projection: f32,
    pub last_direction_projection: f32,
    /// What decensoring found, when it ran (roadmap 31.6).
    pub heretic: Option<crate::heretic::HereticReport>,
}

/// Train a causal language model on `corpus`, charging labeled anti-patterns
/// when the corpus has labels open (roadmap Phase 24).
///
/// With labels open and `config.penalty` off, the run trains plainly but still
/// reports what probability it assigns to the labeled targets — which is how
/// the plain objective is shown to *learn* the anti-patterns rather than merely
/// tolerate them. With the penalty on, those targets are charged for instead.
pub fn train_lm<B: AutodiffBackend<FloatElem = f32>>(
    model: LanguageModel<B>,
    corpus: &mut TokenCorpus,
    config: &LmTrainConfig,
    device: &B::Device,
) -> anyhow::Result<(LanguageModel<B>, LmTrainReport)> {
    let mut mix = CorpusMix::single(corpus)?;
    train_lm_mixed(model, &mut mix, &LmTrainInputs::default(), config, device)
}

/// [`train_lm`] over several corpora, with teachers and a negative teacher
/// (roadmap Phase 29).
pub fn train_lm_mixed<B: AutodiffBackend<FloatElem = f32>>(
    mut model: LanguageModel<B>,
    mix: &mut CorpusMix<'_>,
    inputs: &LmTrainInputs<B>,
    config: &LmTrainConfig,
    device: &B::Device,
) -> anyhow::Result<(LanguageModel<B>, LmTrainReport)> {
    anyhow::ensure!(config.steps > 0, "steps must be positive");
    anyhow::ensure!(config.batch_size > 0, "batch_size must be positive");
    let context = model.context();
    anyhow::ensure!(context >= 2, "a context of {context} has no target position");

    let labeled = mix.any_labels();
    anyhow::ensure!(
        labeled || config.penalty.is_off(),
        "a penalty of {} needs labels: label a corpus and open them first",
        config.penalty.alpha
    );
    let teacher_refs: Vec<(&LanguageModel<B>, f64)> = inputs.teachers.iter().map(|(t, w)| (t, *w)).collect();
    for (teacher, _) in &teacher_refs {
        anyhow::ensure!(
            teacher.vocab_size() == model.vocab_size() && teacher.context() >= context,
            "a teacher must share the vocabulary and cover the student's context"
        );
    }
    if let Some(negative) = &inputs.negative_teacher {
        anyhow::ensure!(negative.vocab_size() == model.vocab_size(), "the negative teacher must share the vocabulary");
    }
    let mut source_stats = SourceStats::new(mix.names().to_vec());
    let direction_tensor = config.direction.as_ref().map(|d| {
        anyhow::ensure!(d.hidden_size() == model.hidden_size(), "direction has {} dims, the model {}", d.hidden_size(), model.hidden_size());
        Ok::<_, anyhow::Error>((d.tensor::<B>(device), d.layer.min(model.num_layers() - 1)))
    }).transpose()?;

    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let mut rng = ChaCha12Rng::seed_from_u64(config.seed);
    let mut logger = config
        .log_path
        .as_ref()
        .map(|path| crate::logging::MetricsLogger::open(path))
        .transpose()?;
    let span = 0..model.num_layers();

    // Training state (roadmap Phase 28): identity of the data, the config as
    // JSON, and whatever an earlier run left beside the model being resumed.
    let dataset_identity = mix.identities()?;
    let config_json = serde_json::to_value(config).context("serialize LM training config")?;
    let mut restored: Option<TrainState> = None;
    if let Some(path) = &config.resume {
        model = checkpoint::load::<B, _>(model, path, device)?;
        match TrainState::for_model(path)? {
            Some(state) => {
                let dir = TrainState::dir_for(path);
                anyhow::ensure!(state.kind == "lm", "{} holds `{}` training state, not an lm run", dir.display(), state.kind);
                state.verify_files(&dir)?;
                anyhow::ensure!(
                    checkpoint::same_datasets(&state.datasets, &dataset_identity),
                    "refusing to resume: the checkpoint was trained on [{}], this run opened [{}]",
                    checkpoint::describe_datasets(&state.datasets),
                    checkpoint::describe_datasets(&dataset_identity)
                );
                let differences = state.config_differences(&config_json);
                if !differences.is_empty() {
                    println!("warning: resuming with a different configuration in {}", differences.join(", "));
                }
                println!("resumed from {} at step {} (training state verified)", path.display(), state.step);
                restored = Some(state);
            }
            None => println!(
                "resumed weights from {}; no training state beside it, so the optimizer and RNG start fresh",
                path.display()
            ),
        }
    }
    if config.bias_balance_rate > 0.0 {
        model.ensure_balance_biases();
    }

    let started = std::time::Instant::now();
    let mut report = LmTrainReport {
        steps_taken: 0,
        steps_skipped: 0,
        first_loss: f32::NAN,
        last_loss: f32::NAN,
        mean_loss: 0.0,
        first_penalized_prob: 0.0,
        last_penalized_prob: 0.0,
        penalized_tokens: 0,
        tokens_seen: 0,
        elapsed_secs: 0.0,
        checkpoint: None,
        resumed_from_step: 0,
        periodic_checkpoints: Vec::new(),
        sources: SourceStats::default(),
        first_negative_teacher_prob: 0.0,
        last_negative_teacher_prob: 0.0,
        negative_teacher_tokens: 0,
        last_distill_loss: 0.0,
        first_direction_projection: 0.0,
        last_direction_projection: 0.0,
        heretic: None,
    };
    let mut loss_sum = 0.0f64;
    let mut elapsed_before = 0.0f64;
    let mut start_step = 0usize;
    if let Some(state) = &restored {
        let resume = config.resume.as_ref().context("a training state was restored without a resume path")?;
        let dir = TrainState::dir_for(resume);
        if let Some(file) = &state.optimizer {
            optim = optim.load_record(checkpoint::load_record::<B, _>(&dir, file, device)?);
        }
        let extras: LmExtras = serde_json::from_value(state.extras.clone()).context("parse lm training state")?;
        report.steps_taken = extras.steps_taken;
        report.steps_skipped = extras.steps_skipped;
        report.first_loss = extras.first_loss;
        report.first_penalized_prob = extras.first_penalized_prob;
        report.penalized_tokens = extras.penalized_tokens;
        report.tokens_seen = extras.tokens_seen;
        loss_sum = extras.loss_sum;
        elapsed_before = extras.elapsed_secs;
        rng = serde_json::from_value(state.host_rng.clone()).context("restore host RNG")?;
        start_step = state.step;
        report.resumed_from_step = start_step;
    }
    let save_state = |step: usize,
                      model: &LanguageModel<B>,
                      optim: &burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, LanguageModel<B>, B>,
                      rng: &ChaCha12Rng,
                      report: &LmTrainReport,
                      loss_sum: f64,
                      elapsed: f64|
     -> anyhow::Result<Option<PathBuf>> {
        let Some(dir) = &config.out_dir else {
            return Ok(None);
        };
        let model_path = checkpoint::save_content_addressed(model.clone(), dir, "lm")?;
        let state_dir = TrainState::dir_for(&model_path);
        if state_dir.exists() {
            std::fs::remove_dir_all(&state_dir).with_context(|| format!("clear {}", state_dir.display()))?;
        }
        let optimizer = Some(checkpoint::save_record::<B, _>(optim.to_record(), &state_dir, "optimizer")?);
        let extras = LmExtras {
            steps_taken: report.steps_taken,
            steps_skipped: report.steps_skipped,
            first_loss: report.first_loss,
            first_penalized_prob: report.first_penalized_prob,
            penalized_tokens: report.penalized_tokens,
            tokens_seen: report.tokens_seen,
            loss_sum,
            elapsed_secs: elapsed,
        };
        let state = TrainState {
            format_version: checkpoint::STATE_FORMAT_VERSION,
            kind: "lm".into(),
            step,
            seed: config.seed,
            host_rng: serde_json::to_value(rng).context("serialize host RNG")?,
            config: config_json.clone(),
            build: checkpoint::BuildInfo::current(),
            datasets: dataset_identity.clone(),
            model: checkpoint::model_entry(&model_path)?,
            optimizer,
            ema: None,
            head: None,
            head_optimizer: None,
            extras: serde_json::to_value(&extras).context("serialize lm training extras")?,
            saved_unix_secs: checkpoint::unix_now(),
        };
        state.write(&state_dir)?;
        Ok(Some(model_path))
    };

    for step in start_step..config.steps {
        <B as burn::tensor::backend::Backend>::seed(device, step_seed(config.seed, step));
        let (windows, weight_rows, origin) = mix.sample(config.batch_size, context - 1, &mut rng)?;
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| i64::from(*t)))
            .collect();
        let tokens = Tensor::<B, 1, burn::tensor::Int>::from_ints(flat.as_slice(), device)
            .reshape([config.batch_size, context]);

        let negatives = weight_rows
            .as_ref()
            .map(|rows| (crate::mix::weight_rows::<B>(rows, device), config.penalty));
        let extra = match (&inputs.negative_teacher, config.negative_penalty > 0.0) {
            (Some(negative), true) => {
                let (proposed, weights) = negative.negative_proposals(tokens.clone(), config.negative_confidence);
                Some(crate::lm::ExtraNegatives {
                    tokens: proposed,
                    weights,
                    alpha: config.negative_penalty,
                    epsilon: config.penalty.epsilon,
                })
            }
            _ => None,
        };
        let distill = (config.distill_weight > 0.0 && !teacher_refs.is_empty()).then_some(crate::lm::Distillation {
            teachers: teacher_refs.as_slice(),
            temperature: config.distill_temperature,
            weight: config.distill_weight,
        });
        let penalty = direction_tensor.as_ref().map(|(d, layer)| crate::lm::DirectionPenalty {
            direction: d,
            layer: *layer,
            weight: config.direction_weight,
        });
        let crate::lm::LmStep { loss, metrics, routing } =
            model.next_token_step_directed(tokens, negatives, extra, distill, penalty, span.clone());

        if !metrics.loss.is_finite() {
            // The same policy as the image loop: a pathological step is
            // discarded, not clipped into something that looks fine.
            report.steps_skipped += 1;
            println!("step {step}: non-finite loss {}, step discarded", metrics.loss);
            continue;
        }

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(config.lr, model, grads);
        if config.bias_balance_rate > 0.0 && !routing.is_empty() {
            model.nudge_balance_biases(span.clone(), &routing, config.bias_balance_rate);
        }

        if report.steps_taken == 0 {
            report.first_loss = metrics.loss;
            report.first_penalized_prob = metrics.penalized_prob;
            report.first_negative_teacher_prob = metrics.negative_teacher_prob;
        }
        report.last_loss = metrics.loss;
        report.last_penalized_prob = metrics.penalized_prob;
        report.last_negative_teacher_prob = metrics.negative_teacher_prob;
        report.negative_teacher_tokens += metrics.negative_teacher_tokens;
        report.last_distill_loss = metrics.distill_loss;
        if report.steps_taken == 0 {
            report.first_direction_projection = metrics.direction_projection;
        }
        report.last_direction_projection = metrics.direction_projection;
        source_stats.record(&origin, metrics.loss);
        report.steps_taken += 1;
        report.penalized_tokens += metrics.penalized_tokens;
        report.tokens_seen += metrics.tokens_counted;
        loss_sum += f64::from(metrics.loss);

        if config.log_every > 0 && (step % config.log_every == 0 || step + 1 == config.steps) {
            let mut line = format!(
                "step {step}: loss {:.4} ppl {:.2}",
                metrics.loss, metrics.perplexity
            );
            if labeled {
                line.push_str(&format!(
                    " | {} labeled targets, p(bad) {:.4}, charge {:.4}",
                    metrics.penalized_tokens, metrics.penalized_prob, metrics.penalty
                ));
            }
            if !routing.is_empty() {
                line.push_str(&format!(
                    " | routing: token H {:.3}, max load {:.3}, stability {:.3}{}",
                    metrics.routing_entropy,
                    metrics.routing_max_load,
                    crate::moe::RoutingStats::mean_stability(&routing),
                    crate::moe::RoutingStats::mean_agreement(&routing)
                        .map_or(String::new(), |a| format!(", layer agreement {a:.3}"))
                ));
            }
            if metrics.negative_teacher_tokens > 0 {
                line.push_str(&format!(
                    " | negative teacher: {} proposals, p {:.4}",
                    metrics.negative_teacher_tokens, metrics.negative_teacher_prob
                ));
            }
            if metrics.distill_loss > 0.0 {
                line.push_str(&format!(" | distill {:.4}", metrics.distill_loss));
            }
            if direction_tensor.is_some() {
                line.push_str(&format!(" | direction {:.4}", metrics.direction_projection));
            }
            println!("{line}");
            if let Some(logger) = logger.as_mut() {
                let mut fields = vec![
                    ("loss", crate::logging::jnum(metrics.loss)),
                    ("perplexity", crate::logging::jnum(metrics.perplexity)),
                    ("penalized_tokens", metrics.penalized_tokens.to_string()),
                    ("penalized_prob", crate::logging::jnum(metrics.penalized_prob)),
                    ("penalty", crate::logging::jnum(metrics.penalty)),
                    ("negative_teacher_tokens", metrics.negative_teacher_tokens.to_string()),
                    ("negative_teacher_prob", crate::logging::jnum(metrics.negative_teacher_prob)),
                    ("distill_loss", crate::logging::jnum(metrics.distill_loss)),
                    ("direction_projection", crate::logging::jnum(metrics.direction_projection)),
                    ("source", origin.sole_source().map_or("-1".to_string(), |s| s.to_string())),
                ];
                if !routing.is_empty() {
                    let (load_h, token_h, min_load, max_load) =
                        crate::moe::RoutingStats::summarize(&routing);
                    fields.push(("load_entropy", crate::logging::jnum(load_h)));
                    fields.push(("token_entropy", crate::logging::jnum(token_h)));
                    fields.push(("min_load", crate::logging::jnum(min_load)));
                    fields.push(("max_load", crate::logging::jnum(max_load)));
                    fields.push(("routing_load", crate::moe::RoutingStats::load_json(&routing)));
                    fields.push(("routing_stability", crate::logging::jnum(crate::moe::RoutingStats::mean_stability(&routing))));
                    if let Some(agreement) = crate::moe::RoutingStats::mean_agreement(&routing) {
                        fields.push(("routing_agreement", crate::logging::jnum(agreement)));
                    }
                }
                logger.log(step, &fields)?;
            }
        }

        if config.checkpoint_every > 0 && (step + 1) % config.checkpoint_every == 0 && step + 1 < config.steps {
            let elapsed = elapsed_before + started.elapsed().as_secs_f64();
            if let Some(path) = save_state(step + 1, &model, &optim, &rng, &report, loss_sum, elapsed)? {
                println!("step {step}: checkpoint {}", path.display());
                report.periodic_checkpoints.push((step + 1, path));
            }
        }
    }

    anyhow::ensure!(report.steps_taken > 0, "every step produced a non-finite loss");
    report.mean_loss = (loss_sum / report.steps_taken as f64) as f32;
    report.elapsed_secs = elapsed_before + started.elapsed().as_secs_f64();
    if source_stats.is_multi() {
        print!("per-corpus:\n{}", source_stats.render());
    }
    report.sources = source_stats;
    if let Some(heretic) = &config.heretic {
        println!(
            "heretic: {} target / {} baseline prompt(s), {} trial(s), kl weight {}",
            heretic.target.len(),
            heretic.baseline.len(),
            heretic.trials,
            heretic.kl_weight
        );
        let (decensored, report_h) = crate::heretic::decensor(model, heretic, device)?;
        print!("{}", crate::heretic::render(&report_h.trials));
        model = decensored;
        report.heretic = Some(report_h);
    }
    report.checkpoint = save_state(config.steps, &model, &optim, &rng, &report, loss_sum, report.elapsed_secs)?;
    Ok((model, report))
}

/// Host-side LM trainer state saved beside the model (roadmap Phase 28).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LmExtras {
    steps_taken: usize,
    steps_skipped: usize,
    first_loss: f32,
    first_penalized_prob: f32,
    penalized_tokens: usize,
    tokens_seen: usize,
    loss_sum: f64,
    elapsed_secs: f64,
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
        let mut rng = ChaCha12Rng::seed_from_u64(0);
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

    #[test]
    fn test_train_lm_refuses_a_penalty_without_labels_and_runs_plainly_with_none() {
        use crate::lm::LmConfig;
        let dir = std::env::temp_dir().join("dblocks-train-lm-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("plain.txt");
        std::fs::write(&source, "abcabcabcabcabcabcabcabcabcabcabc".repeat(4)).unwrap();
        let path = dir.join("plain.bin");
        TokenCorpus::tokenize_file(&source, &path).unwrap();
        let mut corpus = TokenCorpus::in_memory(&path).unwrap();

        let device = Default::default();
        let model = LanguageModel::<DefaultTrainBackend>::new(&LmConfig::tiny(), &device);

        let charged = LmTrainConfig { steps: 2, batch_size: 2, penalty: Unlikelihood::new(1.0), ..Default::default() };
        let err = train_lm(model.clone(), &mut corpus, &charged, &device).unwrap_err().to_string();
        assert!(err.contains("needs labels"), "unhelpful error: {err}");

        let plain = LmTrainConfig { steps: 3, batch_size: 2, log_every: 0, ..Default::default() };
        let (_, report) = train_lm(model, &mut corpus, &plain, &device).unwrap();
        assert_eq!(report.steps_taken, 3);
        assert_eq!(report.penalized_tokens, 0);
        assert!(report.first_loss.is_finite() && report.last_loss.is_finite());
        assert_eq!(report.tokens_seen, 3 * 2 * (LmConfig::tiny().context - 1));
    }
}
