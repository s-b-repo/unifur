//! Rectified-flow / flow-matching objective (roadmap Phase 5).
//!
//! The label embedding plays the role of the clean sample `x0`, Gaussian
//! noise is `x1 = eps`, and the conditional OT path (item 5.2) is the
//! straight line `xt = (1 - t) * x0 + t * x1` with the linear schedule
//! `t in [0, 1]` (item 5.3). The velocity target of rectified flow
//! (item 5.1) is `v* = x1 - x0`; the network predicts `v` from the modulated
//! CLS hidden of a single block, conditioned on the continuous time `t`
//! through the existing timestep embedder.
//!
//! Sampling integrates `dz/dt = v(z, t)` backwards from `t = 1` to `t = 0`
//! with Euler steps, then classifies by nearest label embedding.

use crate::{
    data::TrainDataset,
    dblock::DblockClassifier,
    train::{DefaultTrainBackend, RunningAvg},
    vit::ViTDiTConfig,
};
use burn::{
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{backend::Backend, Distribution, Tensor},
};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Flow-matching hyperparameters.
#[derive(Debug, Clone)]
pub struct FlowMatchingConfig {
    /// Number of blocks (layer groups) the trunk is partitioned into.
    pub num_blocks: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub batch_size: usize,
    pub image_size: usize,
    pub num_labels: usize,
    pub steps: usize,
    pub log_every: usize,
    pub seed: u64,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            num_blocks: 3,
            lr: 1e-3,
            weight_decay: 0.01,
            batch_size: 128,
            image_size: 32,
            num_labels: 100,
            steps: 200,
            log_every: 20,
            seed: 42,
        }
    }
}

/// Rectified-flow loss for one batch: sample `t ~ U(0,1)` per example, build
/// the OT path, regress the pooled hidden onto `v* = x1 - x0`.
pub fn flow_matching_loss<B, R>(
    model: &DblockClassifier<B>,
    pixel_values: &Tensor<B, 4>,
    labels: Tensor<B, 1, Int>,
    rng: &mut R,
) -> Tensor<B, 1>
where
    B: Backend<FloatElem = f32>,
    R: Rng,
{
    let device = pixel_values.device();
    let b = pixel_values.dims()[0];

    // Clean endpoints and noise endpoints.
    let x0 = model.model().normalized_label_embeds(labels);
    let x1 = Tensor::<B, 2>::random(x0.dims(), Distribution::Normal(0.0, 1.0), &device);

    // Per-sample t ~ U(0, 1); scale to the timestep range the embedder sees.
    let t_f: Vec<f32> = (0..b).map(|_| rng.random::<f32>()).collect();
    let t = Tensor::<B, 1>::from_floats(t_f.as_slice(), &device).clamp(1e-4, 1.0 - 1e-4);

    // Conditional OT path.
    let t2 = t.clone().unsqueeze_dim::<2>(1);
    let xt = x0.clone() * (1.0 - t2.clone()) + x1.clone() * t2.clone();

    // Velocity target v* = x1 - x0; prediction from a random block span.
    let v_target = x1 - x0;
    let block_idx = rng.random_range(0..model.num_blocks());
    let span = model.layer_range(block_idx);

    // Feed scaled time so its magnitude resembles EDM c_noise magnitudes.
    let timesteps = t * 1000.0;
    let v_pred = model.model().forward_pooled_block(
        span,
        pixel_values.clone(),
        xt,
        timesteps,
    );

    (v_pred - v_target).powf_scalar(2.0).mean()
}

use burn::tensor::Int;

/// Train the classifier with the flow-matching objective on synthetic data
/// (`train_flow_synthetic`), mirroring [`crate::train::train_synthetic`].
pub fn train_flow_synthetic(
    config: &FlowMatchingConfig,
) -> anyhow::Result<DblockClassifier<DefaultTrainBackend>> {
    use crate::data::SyntheticDataset;

    let device = Default::default();
    <DefaultTrainBackend as burn::tensor::backend::Backend>::seed(&device, config.seed);

    let vit_config = ViTDiTConfig::with_image_size(config.image_size, config.num_labels);
    let dblock_config = crate::dblock::DblockConfig {
        num_blocks: config.num_blocks,
        ..crate::dblock::DblockConfig::default()
    };
    let mut model =
        DblockClassifier::<DefaultTrainBackend>::new(&vit_config, &dblock_config, &device);

    let mut dataset =
        SyntheticDataset::new(config.image_size, config.num_labels, config.batch_size, config.seed);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let lr: burn::optim::LearningRate = config.lr;

    let mut running = RunningAvg::new(config.log_every);
    let start = std::time::Instant::now();

    for step in 0..config.steps {
        let batch = dataset.next_batch(&mut rng, &device);
        let loss = flow_matching_loss(&model, &batch.pixel_values, batch.labels.clone(), &mut rng);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        let l: f32 = loss.clone().into_scalar();
        running.push(l);
        if step % config.log_every.max(1) == 0 || step + 1 == config.steps {
            println!(
                "fm step {:>6} | mse {:.5} | avg {:.5} | {:.1} steps/s",
                step,
                l,
                running.avg(),
                (step + 1) as f64 / start.elapsed().as_secs_f64().max(1e-9)
            );
        }
    }
    Ok(model)
}

/// Sample class embeddings by integrating the learned velocity field from
/// pure noise at `t = 1` down to `t = 0` (Euler), then classify by nearest
/// normalized label embedding.
pub fn flow_sample<B: Backend<FloatElem = f32>, R: Rng>(
    model: &DblockClassifier<B>,
    pixel_values: &Tensor<B, 4>,
    num_steps: usize,
    rng: &mut R,
) -> Tensor<B, 1, burn::tensor::Int> {
    assert!(num_steps >= 1, "flow sampling needs at least one step");
    let device = pixel_values.device();
    let b = pixel_values.dims()[0];
    let h_dim = model.model().label_embedding_weight().dims()[1];

    let mut z = crate::solver::randn_like(&Tensor::<B, 2>::zeros([b, h_dim], &device), rng);
    let dt = 1.0f32 / num_steps as f32;

    for i in 0..num_steps {
        let t_val = 1.0 - i as f32 * dt;
        let t = Tensor::<B, 1>::full([b], t_val * 1000.0, &device);
        let block_idx = (i * model.num_blocks() / num_steps.max(1)).min(model.num_blocks() - 1);
        let v = model
            .model()
            .forward_pooled_block(model.layer_range(block_idx), pixel_values.clone(), z.clone(), t);
        z = z - v.mul_scalar(dt); // descend from t=1 to t=0
    }

    // Nearest-label classification by cosine similarity. Both sides must be
    // L2-normalized: the raw dot product would rank labels by embedding norm
    // as much as by direction, and the label table is not norm-uniform.
    let table = crate::tensor_ext::l2_normalize_rows(model.model().label_embedding_weight());
    let sims = crate::tensor_ext::l2_normalize_rows(z).matmul(table.transpose());
    sims.argmax(1).squeeze_dim::<1>(1)
}
