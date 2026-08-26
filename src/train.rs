//! Block-wise training loop (`main.py` / Lightning plumbing of the reference,
//! reduced to a self-contained Burn loop).
//!
//! Every step trains exactly one randomly chosen block on its own sigma
//! window (original DiffusionBlocks). Parallel-depth trajectories, solvers
//! and the remaining roadmap features build on this loop.

use crate::{
    data::{SyntheticDataset, TrainDataset},
    dblock::{DblockClassifier, DblockConfig},
    vit::ViTDiTConfig,
};
use burn::{
    backend::{
        autodiff::checkpoint::strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing},
        Autodiff, NdArray,
    },
    optim::{AdamWConfig, GradientsParams, Optimizer},
};
use rand::{rngs::StdRng, SeedableRng};

/// Autodiff-enabled ndarray backend used for CPU training.
pub type DefaultTrainBackend = Autodiff<NdArray<f32>>;
/// Variant that checkpoints intermediate activations during backward
/// (roadmap 15.2): trades compute for lower peak memory.
pub type CheckpointedTrainBackend = Autodiff<NdArray<f32>, BalancedCheckpointing>;

/// Training hyperparameters (subset of the reference CLI).
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
    /// Total optimizer steps (the reference counts epochs x batches; we count
    /// steps directly for now).
    pub steps: usize,
    pub log_every: usize,
    pub seed: u64,
    /// Optional JSONL metrics sink (roadmap 1.5 / 15.6).
    pub log_file: Option<std::path::PathBuf>,
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
        }
    }
}

/// Train a dblock classifier on synthetic data; returns the trained model.
///
/// This is the Phase-1 skeleton loop: swap in a real dataset by implementing
/// [`TrainDataset`].
pub fn train_synthetic(
    config: &TrainConfig,
) -> anyhow::Result<DblockClassifier<DefaultTrainBackend>> {
    train_synthetic_generic::<NoCheckpointing>(config)
}

/// Backend-generic training loop; `C` selects the autodiff checkpointing
/// strategy (see [`CheckpointedTrainBackend`]).
pub fn train_synthetic_generic<C>(
    config: &TrainConfig,
) -> anyhow::Result<DblockClassifier<Autodiff<NdArray<f32>, C>>>
where
    C: CheckpointStrategy,
{
    let device = Default::default();

    // Seed the on-device RNG (init, data generation, dropout) so runs with
    // the same config produce bit-identical checkpoints.
    <Autodiff<NdArray<f32>, C> as burn::tensor::backend::Backend>::seed(&device, config.seed);

    let vit_config = ViTDiTConfig::with_image_size(config.image_size, config.num_labels);
    let dblock_config = DblockConfig {
        num_blocks: config.num_blocks,
        gamma: config.gamma,
        ..DblockConfig::default()
    };

    let mut model =
        DblockClassifier::<Autodiff<NdArray<f32>, C>>::new(&vit_config, &dblock_config, &device);

    let mut dataset = SyntheticDataset::new(
        config.image_size,
        config.num_labels,
        config.batch_size,
        config.seed,
    );
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Burn optimizers are functional: step() consumes the model record and
    // returns an updated one, so we reassign the binding every step.
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let lr: burn::optim::LearningRate = config.lr;

    let mut running = RunningAvg::new(config.log_every);
    let start = std::time::Instant::now();
    let mut jsonl = config
        .log_file
        .as_ref()
        .map(|path| crate::logging::MetricsLogger::open(path))
        .transpose()?;

    for step in 0..config.steps {
        let batch = dataset.next_batch(&mut rng, &device);

        // Forward through exactly one block + backward.
        let (loss, metrics) =
            model.training_step(batch.pixel_values, batch.labels, config.gamma, &mut rng);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        running.push(metrics.loss);
        if step % config.log_every.max(1) == 0 || step + 1 == config.steps {
            let sps = (step + 1) as f64 / start.elapsed().as_secs_f64().max(1e-9);
            println!(
                "step {:>6} | loss {:.4} | ce {:.4} | block {} | avg {:.4} | {:.1} steps/s",
                step,
                metrics.loss,
                metrics.ce_loss,
                metrics.block_idx,
                running.avg(),
                sps
            );
            if let Some(logger) = jsonl.as_mut() {
                logger.log(
                    step,
                    &[
                        ("loss", crate::logging::jnum(metrics.loss)),
                        ("ce_loss", crate::logging::jnum(metrics.ce_loss)),
                        ("block", format!("{}", metrics.block_idx)),
                        ("steps_per_s", crate::logging::jnum(sps as f32)),
                    ],
                )?;
            }
        }
    }

    Ok(model)
}

/// Exponential-ish running average over the last `window` values.
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
}
