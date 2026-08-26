//! End-to-end integration tests over the full feature surface
//! (roadmap Phase 16.10 / 16.11), kept small enough for CPU CI.

use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::BackendTypes;
use burn::{tensor::Tensor, tensor::Distribution};
use diffusionblocks::consistency::{ConsistencyConfig, ConsistencyWeights};
use diffusionblocks::dblock::{DblockClassifier, DblockConfig};
use diffusionblocks::flow;
use diffusionblocks::multi_block::{
    Gated, MultiBlockConfig, SamplingStats, Strategy,
};
use diffusionblocks::quality::QualityGateConfig;
use diffusionblocks::solver::SolverKind;
use diffusionblocks::train::DefaultTrainBackend as B;
use diffusionblocks::vit::ViTDiTConfig;
use rand::{rngs::StdRng, SeedableRng};

fn tiny_vit_config() -> ViTDiTConfig {
    ViTDiTConfig {
        image_size: 32,
        patch_size: 16,
        in_channels: 3,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        layer_norm_eps: 1e-12,
        hidden_dropout_prob: 0.0,
        attention_probs_dropout_prob: 0.0,
        initializer_range: 0.02,
        num_labels: 10,
        cond_hidden_size: 8,
        frequency_embedding_size: 16,
    }
}

fn tiny_model(device: &<B as BackendTypes>::Device) -> DblockClassifier<B> {
    <B as burn::tensor::backend::Backend>::seed(device, 7);
    DblockClassifier::<B>::new(
        &tiny_vit_config(),
        &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
        device,
    )
}

fn fixture_batch(
    device: &<B as BackendTypes>::Device,
) -> (Tensor<B, 4>, Tensor<B, 1, burn::tensor::Int>) {
    let pixels =
        Tensor::<B, 4>::random([2, 3, 32, 32], Distribution::Uniform(-0.5, 0.5), device);
    let labels = Tensor::<B, 1, burn::tensor::Int>::from_ints([3i64, 8].as_slice(), device);
    (pixels, labels)
}

#[test]
fn integration_standard_training_step_backward() {
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, labels) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(0);

    let (loss, metrics) = model.training_step(pixels.clone(), labels, 0.05, &mut rng);
    assert!(metrics.loss.is_finite());
    let grads = loss.backward();
    let _gp = GradientsParams::from_grads(grads, &model);
}

#[test]
fn integration_consistency_step_all_terms() {
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, labels) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(1);

    let cfg = ConsistencyConfig {
        gamma: 0.05,
        weights: ConsistencyWeights { boundary: 1.0, self_consistency: 1.0, trajectory: 1.0 },
        ..ConsistencyConfig::default()
    };
    let (loss, metrics) = model.consistency_step(&pixels, labels.clone(), &cfg, 5, &mut rng);
    assert!(
        metrics.boundary_loss.is_finite()
            && metrics.self_loss.is_finite()
            && metrics.trajectory_loss.is_finite(),
        "all consistency terms must produce finite values"
    );
    let grads = loss.backward();
    let _gp = GradientsParams::from_grads(grads, &model);
}

#[test]
fn integration_flow_matching_loss_and_update() {
    let device = Default::default();
    let mut model = tiny_model(&device);
    let (pixels, labels) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(2);

    let loss = flow::flow_matching_loss(&model, &pixels, labels.clone(), &mut rng);
    let grads = loss.backward();
    let gp = GradientsParams::from_grads(grads, &model);
    let mut optim = AdamWConfig::new().init();
    model = optim.step(1e-3, model, gp); // one full FM update must succeed

    // Velocity-field sampler runs and returns label ids of the right shape.
    let ids = flow::flow_sample(&model, &pixels, 3, &mut rng);
    assert_eq!(ids.dims().len(), 1);
        assert_eq!(ids.dims()[0], 2);
}

#[test]
fn integration_multi_block_strategies_and_solvers() {
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _labels) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(3);

    let cases = [
        Strategy::Sequential,
        Strategy::Parallel { k: 2 },
        Strategy::Hybrid { k: 2, warmup_frac: 0.3 },
        Strategy::Adaptive { k_max: 2, conf_threshold: 1.0 },
    ];
    for strategy in cases {
        let config = MultiBlockConfig {
            strategy: Gated { inner: strategy, gate: QualityGateConfig::strict() },
            solver: SolverKind::Euler,
            num_steps: Some(3),
        };
        let (logits, stats): (_, SamplingStats) =
            model.sample_multi_block(&pixels, &config, &mut rng);
        assert_eq!(logits.dims(), [2, 10]);
        assert!(stats.model_calls >= 2, "at least one call per window");
    }

    // Solver variety through the fixed-policy path.
    for kind in [SolverKind::Euler, SolverKind::Heun, SolverKind::DpmPlusPlus2M] {
        let logits = model.solve(&pixels, kind, 2, None, &mut rng);
        assert_eq!(logits.dims(), [2, 10]);
    }
}
