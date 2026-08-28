//! End-to-end integration tests over the full feature surface
//! (roadmap Phase 16.10 / 16.11), kept small enough for CPU CI.

use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::BackendTypes;
use burn::{tensor::Distribution, tensor::Tensor};
use diffusionblocks::consistency::{ConsistencyConfig, ConsistencyWeights};
use diffusionblocks::dblock::{DblockClassifier, DblockConfig};
use diffusionblocks::distill::DistillConfig;
use diffusionblocks::flow;
use diffusionblocks::infer::{InferenceConfig, InferenceEngine};
use diffusionblocks::loopgraph::{LoopGraph, LoopGraphConfig};
use diffusionblocks::multi_block::{Gated, MultiBlockConfig, PlannedConfig, SamplingStats, Strategy};
use diffusionblocks::planner::Budget;
use diffusionblocks::precision::{Precision, PrecisionPolicy};
use diffusionblocks::quality::{LayerGates, QualityGateConfig};
use diffusionblocks::quantize::{quantize_module, LoraConfig, QLoraLinear};
use diffusionblocks::solver::SolverKind;
use diffusionblocks::train::DefaultTrainBackend as B;
use diffusionblocks::verify;
use diffusionblocks::expert_index::{BoxSpec, ExpertSpec, MosmeSpec};
use diffusionblocks::mosme::{MosmeConfig, MosmeFeedForward};
use diffusionblocks::vit::{MoeTrunkConfig, MosmeTrunkConfig, ViTDiTConfig};
use rand::{rngs::StdRng, SeedableRng};

type Device = <B as BackendTypes>::Device;

fn tiny_model(device: &Device) -> DblockClassifier<B> {
    <B as burn::tensor::backend::Backend>::seed(device, 7);
    DblockClassifier::<B>::new(
        &ViTDiTConfig::tiny(10),
        &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
        device,
    )
}

fn fixture_batch(device: &Device) -> (Tensor<B, 4>, Tensor<B, 1, burn::tensor::Int>) {
    let pixels = Tensor::<B, 4>::random([2, 3, 32, 32], Distribution::Uniform(-0.5, 0.5), device);
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
    assert_eq!(metrics.balance_loss, 0.0, "a dense trunk has no balance loss");
    let grads = loss.backward();
    let gp = GradientsParams::from_grads(grads, &model);
    assert!(!gp.is_empty(), "the executed span must receive gradients");
}

#[test]
fn integration_consistency_step_all_terms() {
    let device = Default::default();
    // Cross-fork consistency needs a non-adjacent pair, so at least 3 blocks.
    <B as burn::tensor::backend::Backend>::seed(&device, 7);
    let model = DblockClassifier::<B>::new(
        &ViTDiTConfig { num_hidden_layers: 6, ..ViTDiTConfig::tiny(10) },
        &DblockConfig { num_blocks: 3, ..DblockConfig::default() },
        &device,
    );
    let (pixels, labels) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(1);

    let cfg = ConsistencyConfig {
        gamma: 0.05,
        weights: ConsistencyWeights {
            boundary: 1.0,
            self_consistency: 1.0,
            trajectory: 1.0,
            cross_fork: 1.0,
        },
        ..ConsistencyConfig::default()
    };
    let (loss, metrics) = model.consistency_step(&pixels, labels.clone(), &cfg, 5, &mut rng);
    for (name, value) in [
        ("boundary", metrics.boundary_loss),
        ("self", metrics.self_loss),
        ("trajectory", metrics.trajectory_loss),
        ("cross_fork", metrics.cross_fork_loss),
    ] {
        assert!(value.is_finite(), "{name} consistency term is not finite");
        assert!(value >= 0.0, "{name} is an MSE and cannot be negative: {value}");
    }
    let grads = loss.backward();
    let _gp = GradientsParams::from_grads(grads, &model);
}

#[test]
fn integration_disabling_every_consistency_term_reduces_to_the_plain_loss() {
    // The consistency step must be a strict superset of the standard step, not
    // a separate implementation that happens to look similar.
    let device = Default::default();
    let (pixels, labels) = fixture_batch(&device);

    let model = tiny_model(&device);
    <B as burn::tensor::backend::Backend>::seed(&device, 99);
    let (_, plain) =
        model.training_step(pixels.clone(), labels.clone(), 0.05, &mut StdRng::seed_from_u64(4));

    <B as burn::tensor::backend::Backend>::seed(&device, 99);
    let cfg = ConsistencyConfig {
        gamma: 0.05,
        weights: ConsistencyWeights::none(),
        ..ConsistencyConfig::default()
    };
    let (_, with_consistency) =
        model.consistency_step(&pixels, labels, &cfg, 5, &mut StdRng::seed_from_u64(4));

    assert_eq!(plain.block_idx, with_consistency.block_idx);
    assert!(
        (plain.loss - with_consistency.loss).abs() < 1e-4,
        "zero-weight consistency changed the loss: {} vs {}",
        plain.loss,
        with_consistency.loss
    );
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

    // Velocity-field sampler runs and returns label ids in range.
    let ids = flow::flow_sample(&model, &pixels, 3, &mut rng);
    assert_eq!(ids.dims(), [2]);
    let values: Vec<i64> = ids.into_data().convert::<i64>().iter().collect();
    assert!(values.iter().all(|&v| (0..10).contains(&v)), "labels out of range: {values:?}");
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
    // Every strategy must work with every solver: the two are independent
    // axes, and the step-wise SolverState exists so they stay that way.
    for strategy in cases {
        for solver in SolverKind::all() {
            let config = MultiBlockConfig {
                strategy: Gated::uniform(strategy, QualityGateConfig::strict()),
                solver,
                num_steps: Some(3),
                ..MultiBlockConfig::default()
            };
            let (logits, stats): (_, SamplingStats) =
                model.sample_multi_block(&pixels, &config, &mut rng);
            assert_eq!(logits.dims(), [2, 10]);
            assert_eq!(stats.spans.len(), 2, "3 schedule points => 2 windows");
            // Heun evaluates the model twice per window; the others once.
            // Plus one final denoise in every case.
            let expected_min = if solver == SolverKind::Heun { 5 } else { 3 };
            assert!(
                stats.model_calls >= expected_min,
                "{} reported {} calls, expected at least {expected_min}",
                solver.name(),
                stats.model_calls
            );
        }
    }

    // Solver variety through the fixed-policy path too.
    for kind in SolverKind::deterministic() {
        let logits = model.solve(&pixels, kind, 3, None, &mut rng);
        assert_eq!(logits.dims(), [2, 10]);
    }
}

#[test]
fn integration_wider_spans_cost_more_layers() {
    // The cost accounting has to reflect what actually ran, otherwise a
    // strategy benchmark is meaningless.
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(11);

    let run = |strategy: Strategy, rng: &mut StdRng| {
        let config = MultiBlockConfig {
            strategy: Gated::uniform(strategy, QualityGateConfig::lenient()),
            num_steps: Some(3),
            ..MultiBlockConfig::default()
        };
        model.sample_multi_block(&pixels, &config, rng).1
    };

    let sequential = run(Strategy::Sequential, &mut rng);
    let parallel = run(Strategy::Parallel { k: 2 }, &mut rng);
    assert!(
        parallel.layers_executed > sequential.layers_executed,
        "k=2 must execute more layers than k=1: {} vs {}",
        parallel.layers_executed,
        sequential.layers_executed
    );
    assert_eq!(
        parallel.model_calls, sequential.model_calls,
        "parallel spans trade width for depth, not for extra calls"
    );
    assert!(parallel.mean_span_width() > sequential.mean_span_width());
}

#[test]
fn integration_mixed_precision_is_reported_where_it_applies() {
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _) = fixture_batch(&device);

    let run = |precision: PrecisionPolicy| {
        let config = MultiBlockConfig {
            strategy: Gated::uniform(Strategy::Sequential, QualityGateConfig::lenient()),
            num_steps: Some(3),
            precision,
            ..MultiBlockConfig::default()
        };
        // Same seed each time so only the precision differs.
        model.sample_multi_block(&pixels, &config, &mut StdRng::seed_from_u64(5))
    };

    let (_, full) = run(PrecisionPolicy::default());
    assert_eq!(full.reduced_precision_windows, 0);

    // A switch above sigma_max means every window is below it, so the policy
    // must fall back to f32 everywhere.
    let (_, never) = run(PrecisionPolicy::mixed(Precision::Bf16, 1e9));
    assert_eq!(never.reduced_precision_windows, 0, "switch above sigma_max must not trigger");

    let (_, coarse) = run(PrecisionPolicy::full(Precision::Bf16));
    assert!(coarse.reduced_precision_windows > 0, "bf16 everywhere must be reported");
}

#[test]
fn integration_loop_graph_produces_a_convex_mixture() {
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _) = fixture_batch(&device);
    let graph = LoopGraph::<B>::new(2, 32, &device);
    let z = Tensor::<B, 2>::random([2, 32], Distribution::Normal(0.0, 1.0), &device);

    for config in [LoopGraphConfig::feedforward(2), LoopGraphConfig::default()] {
        let (x0, trace) = graph.x0_estimate(&model, &pixels, &z, 1.0, &config);
        assert_eq!(x0.dims(), [2, 32]);
        assert!(!trace.blocks_run.is_empty(), "at least one block must run");
        assert!(
            (trace.weight_mass() - 1.0).abs() < 1e-4,
            "ACT weights must sum to 1, got {}",
            trace.weight_mass()
        );
        assert!(trace.executions() <= config.max_iterations);
    }
}

#[test]
fn integration_distillation_against_a_quantized_student() {
    // The QLoRA story end to end: quantize a copy of a model, then train it to
    // match the full-precision original.
    let device = Default::default();
    let teacher = tiny_model(&device);
    let (student, quantized) = quantize_module(tiny_model(&device), true, &["label_embeddings"]);
    assert!(quantized > 0, "some weights must have been quantized");

    let (pixels, labels) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(6);
    let (loss, metrics) = student.distill_step(
        &teacher,
        &pixels,
        labels,
        &DistillConfig { teacher_substeps: 2, ..DistillConfig::default() },
        &mut rng,
    );
    assert!(metrics.loss.is_finite() && metrics.kl >= 0.0);
    assert_eq!(metrics.steps_saved, 1);

    // Only the student is trainable.
    let grads = GradientsParams::from_grads(loss.backward(), &student);
    assert!(!grads.is_empty());
}

#[test]
fn integration_qlora_layer_starts_as_its_base() {
    let device = Default::default();
    let config = LoraConfig::new(64, 16, 4);
    let layer = QLoraLinear::<B>::new(64, 16, &config, true, &device);
    let x = Tensor::<B, 2>::random([3, 64], Distribution::Uniform(-1.0, 1.0), &device);

    // Zero-initialized B means the adapter contributes nothing yet, so the
    // merged and unmerged layers must agree exactly.
    let diff = (layer.forward(x.clone()) - layer.merged().forward(x))
        .abs()
        .max()
        .into_scalar();
    assert!(diff < 1e-5, "adapter merge changed the function: {diff}");

    let (nf4_bits, f32_bits) = layer.resident_bits();
    assert!(nf4_bits < f32_bits / 7.0, "NF4 must be ~7x smaller");
}

#[test]
fn integration_moe_trunk_trains_with_its_balance_loss() {
    let device = Default::default();
    <B as burn::tensor::backend::Backend>::seed(&device, 7);
    let cfg = ViTDiTConfig::tiny(10).with_moe(MoeTrunkConfig {
        num_experts: 4,
        top_k: 2,
        every_n_layers: 2,
        z_level: 1e-3,
    });
    let model = DblockClassifier::<B>::new(
        &cfg,
        &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
        &device,
    );
    let (pixels, labels) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(8);

    let (loss, metrics) = model.training_step(pixels, labels, 0.05, &mut rng);
    assert!(
        metrics.balance_loss > 0.0,
        "a sparse trunk must report a balance loss, got {}",
        metrics.balance_loss
    );
    // One sparse layer per 2-layer block, and the Switch loss is at least 1.
    assert!(metrics.balance_loss >= 1.0 - 1e-4);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    assert!(!grads.is_empty());
}

#[test]
fn integration_inference_engine_batches_and_ranks() {
    let device = Default::default();
    let model = tiny_model(&device);
    let engine = InferenceEngine::new(
        model,
        InferenceConfig {
            batch_size: 2,
            num_steps: Some(3),
            gates: LayerGates::tightening(
                2,
                QualityGateConfig::lenient(),
                QualityGateConfig::strict(),
            ),
            ..InferenceConfig::default()
        },
    );

    let pixels = Tensor::<B, 4>::random([5, 3, 32, 32], Distribution::Uniform(-0.5, 0.5), &device);
    let preds = engine.classify(pixels, &mut StdRng::seed_from_u64(9));
    assert_eq!(preds.len(), 5);
    for row in preds.top_k(3) {
        assert_eq!(row.len(), 3);
        assert!(row.windows(2).all(|w| w[0].1 >= w[1].1), "top-k must be sorted");
    }
}

fn expert_boxes() -> MosmeSpec {
    MosmeSpec {
        boxes: vec![
            BoxSpec::new(
                "coding",
                "Code",
                vec![
                    ExpertSpec::new("coding/rust", "Rust").with_tags(&["rust"]),
                    ExpertSpec::new("coding/python", "Python").with_tags(&["python"]),
                ],
            ),
            BoxSpec::new(
                "cyber",
                "Cybersecurity",
                vec![ExpertSpec::new("cyber/netsec", "Network security")],
            ),
        ],
        top_box: 1,
        top_expert: 1,
        route_on_tokens: true,
        balance: Default::default(),
    }
}

#[test]
fn integration_mosme_trunk_trains_end_to_end() {
    let device = Default::default();
    <B as burn::tensor::backend::Backend>::seed(&device, 7);
    let cfg = ViTDiTConfig::tiny(10)
        .with_mosme(MosmeTrunkConfig::new(expert_boxes()).with_every_n_layers(2));
    let model = DblockClassifier::<B>::new(
        &cfg,
        &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
        &device,
    );
    let (pixels, labels) = fixture_batch(&device);
    let mut rng = StdRng::seed_from_u64(21);

    let (loss, metrics) = model.training_step(pixels, labels, 0.05, &mut rng);
    assert!(metrics.loss.is_finite());
    assert!(
        metrics.balance_loss > 0.0,
        "a hierarchical trunk must report a balance loss, got {}",
        metrics.balance_loss
    );

    // The bound that is actually a theorem. Each hierarchical layer adds
    // `box_level * L_box + expert_level * L_expert`, and a Switch loss over N
    // terms lies in `[0, N]` for arbitrary traffic. The familiar `L >= 1` holds
    // only on the diagonal `f == p`, which hard top-k routing does not give —
    // asserting it here would be asserting a property of one random draw.
    //
    // `tiny()` has 4 layers and `with_every_n_layers(2)` makes two of them
    // hierarchical; the spec has 2 boxes whose largest holds 3 experts.
    let hierarchical_layers = 2.0;
    let ceiling = hierarchical_layers * (2.0 + 3.0);
    assert!(
        metrics.balance_loss <= ceiling + 1e-4,
        "balance loss {} exceeds its structural maximum {ceiling}",
        metrics.balance_loss
    );

    // What *is* seed-stable is that the same model and the same sampling seed
    // give the same number. This test used to pin a specific magnitude instead,
    // which silently depended on the global backend RNG not being touched by a
    // concurrently running test.
    let (_, again) = model.training_step(
        fixture_batch(&device).0,
        fixture_batch(&device).1,
        0.05,
        &mut StdRng::seed_from_u64(21),
    );
    assert!(again.balance_loss > 0.0 && again.balance_loss <= ceiling + 1e-4);

    let grads = GradientsParams::from_grads(loss.backward(), &model);
    assert!(!grads.is_empty(), "the executed span must receive gradients");
}

#[test]
fn integration_expert_index_round_trips_through_a_file() {
    // The artifact an inference engine consumes: written by training, read
    // back without touching the weights.
    let device = Default::default();
    let spec = expert_boxes();
    let cfg = MosmeConfig::new(32, 8, spec.clone()).with_intermediate_size(64);
    let layer = MosmeFeedForward::<B>::new(&cfg, &device);

    let index = layer.index(&spec, "model-abc", "vit.layers.1.mlp", 8).unwrap();
    assert_eq!(index.num_boxes(), 2);
    assert_eq!(index.num_experts(), 3);

    let dir = std::env::temp_dir().join(format!("mosme-int-{}", std::process::id()));
    let path = dir.join("experts.index.json");
    index.write(&path).unwrap();

    let restored = diffusionblocks::expert_index::ExpertIndex::read(&path).unwrap();
    assert_eq!(restored, index);
    let (bx, expert) = restored.expert("coding/rust").unwrap();
    assert_eq!(bx.id, "coding");
    assert!(expert.enabled);
    assert!(!expert.weights.sha256.is_empty());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn integration_adding_an_expert_is_an_exact_identity() {
    // "Add a specialist without retraining the others" as an end-to-end claim,
    // not just a unit-level one.
    let device = Default::default();
    <B as burn::tensor::backend::Backend>::seed(&device, 3);
    let mut spec = expert_boxes();
    spec.top_expert = 3; // wide enough that every enabled expert gets a gate
    let cfg = MosmeConfig::new(32, 8, spec.clone()).with_intermediate_size(64);
    let layer = MosmeFeedForward::<B>::new(&cfg, &device);

    let x = Tensor::<B, 3>::random([2, 5, 32], Distribution::Uniform(-1.0, 1.0), &device);
    let cond = Tensor::<B, 2>::random([2, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let before = layer.forward(x.clone(), cond.clone()).output;

    let grown_spec = spec
        .extended_with("coding", ExpertSpec::new("coding/go", "Go"))
        .unwrap();
    let mut grown = layer.grown(&grown_spec, &cfg, &device).unwrap();
    let after = grown.forward(x.clone(), cond.clone()).output;
    assert_eq!(
        (before.clone() - after).abs().max().into_scalar(),
        0.0,
        "adding a disabled expert must change nothing at all"
    );

    grown.router_mut().set_enabled(0, 2, true).unwrap();
    let enabled = grown.forward(x, cond).output;
    assert!(
        (before - enabled).abs().max().into_scalar() > 1e-6,
        "enabling it must then actually do something"
    );
}

#[test]
fn integration_certificate_suite_passes() {
    // The quality gate itself, exercised from outside the crate exactly as
    // `dblocks verify` runs it.
    let report = verify::run_all();
    assert!(report.passed(), "certificate suite failed:\n{}", report.render());
    assert!(report.certificates.len() >= 25);
    // The mosme group must be present, not silently dropped.
    assert!(
        report.certificates.iter().any(|c| c.group == "mosme"),
        "the mosme certificate group is missing"
    );
}

#[test]
fn integration_planned_sampling_stays_inside_its_budget() {
    // Planning multiplies work: beam x depth x candidates model calls per
    // committed step. The budget is what makes that deployable, so it has to
    // hold against every shape -- including a budget too small to finish a
    // single level.
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _) = fixture_batch(&device);

    for budget in [
        Budget::greedy(),
        Budget { max_evaluations: 1, max_depth: 3, beam_width: 4 },
        Budget { max_evaluations: 6, max_depth: 1, beam_width: 2 },
        Budget { max_evaluations: 40, max_depth: 2, beam_width: 3 },
    ] {
        let config = PlannedConfig { budget, max_steps: 6, ..PlannedConfig::default() };
        let mut rng = StdRng::seed_from_u64(19);
        let (logits, stats, trace) = model.sample_planned(&pixels, &config, &mut rng);

        assert_eq!(logits.dims(), [2, 10]);
        assert!(!trace.steps.is_empty(), "the planner must commit something");
        // One over the cap is the forced final step that closes the remaining
        // distance to sigma_min; it is not a planned step.
        assert!(trace.planned_steps() <= config.max_steps);
        assert!(trace.steps.len() <= config.max_steps + 1);

        for spent in &trace.evaluations {
            assert!(
                *spent <= budget.max_evaluations,
                "spent {spent} against a budget of {}",
                budget.max_evaluations
            );
        }
        // One model call per (node, width); every call is paired with at least
        // one charged candidate, plus the final denoise and the solver's own
        // corrector evaluations.
        assert!(stats.model_calls >= trace.steps.len());
        for depth in &trace.depths {
            assert!(*depth <= budget.max_depth + 1);
        }
    }
}

#[test]
fn integration_planned_sampling_descends_and_is_reproducible() {
    // A planned trajectory is still a trajectory: sigma must fall on every
    // committed step. And with no sampling in the loop, the same seed must
    // give the same path -- otherwise a planned run cannot be reproduced.
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _) = fixture_batch(&device);

    let config = PlannedConfig {
        budget: Budget { max_evaluations: 32, max_depth: 1, beam_width: 2 },
        max_steps: 8,
        ..PlannedConfig::default()
    };

    let run = || {
        model.sample_planned(&pixels, &config, &mut StdRng::seed_from_u64(23))
    };
    let (first_logits, _, first) = run();
    let (second_logits, _, second) = run();

    assert_eq!(first.steps, second.steps, "the planned path must be reproducible");
    let a: Vec<f32> = first_logits.into_data().convert::<f32>().iter::<f32>().collect();
    let b: Vec<f32> = second_logits.into_data().convert::<f32>().iter::<f32>().collect();
    assert_eq!(a, b);

    let mut previous = f64::INFINITY;
    for step in &first.steps {
        assert!(step.sigma < previous, "sigma must fall: {} !< {previous}", step.sigma);
        assert!(step.sigma.is_finite() && step.sigma > 0.0);
        previous = step.sigma;
    }
    assert!(
        first.steps.iter().all(|s| config.widths.contains(&s.width)),
        "committed widths must come from the candidate set"
    );
}

#[test]
fn integration_a_planned_trajectory_always_lands_on_sigma_min() {
    // The step cap can bind before the floor is reached. Handing a latent that
    // is still at sigma=0.6 to a denoise at sigma_min would be a silent
    // discontinuity -- the estimate conditioned on a noise level the latent
    // does not have -- so the remaining distance is closed explicitly.
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _) = fixture_batch(&device);

    let floor = *diffusionblocks::sigma::discrete_sigmas_dblock(
        4,
        diffusionblocks::sigma::SIGMA_MIN,
        diffusionblocks::sigma::SIGMA_MAX,
        diffusionblocks::sigma::P_MEAN,
        diffusionblocks::sigma::P_STD,
    )
    .last()
    .unwrap();

    // Two steps is far too few to descend from sigma_max on its own, so the
    // forced final step is guaranteed to fire.
    let tight = PlannedConfig {
        budget: Budget { max_evaluations: 24, max_depth: 1, beam_width: 2 },
        max_steps: 2,
        ..PlannedConfig::default()
    };
    let (_, stats, trace) =
        model.sample_planned(&pixels, &tight, &mut StdRng::seed_from_u64(41));

    assert!(trace.forced_final_step, "the step cap should have bound");
    assert_eq!(trace.final_sigma, floor, "the trajectory must end at the floor");
    assert_eq!(trace.steps.last().map(|s| s.sigma), Some(floor));

    // The recorded spans and the executed layers are different quantities:
    // layers_executed also carries planning work, so a mean span computed from
    // it would report spans many times their real width.
    assert!(
        stats.layers_executed > stats.spans.iter().map(|s| s.len()).sum::<usize>(),
        "planning work must be counted in the total cost"
    );
    assert!(
        stats.mean_span_width() <= 2.0 * model.num_blocks() as f32,
        "mean span {} is not a plausible width",
        stats.mean_span_width()
    );
    assert!(stats.planning_layers > 0 && stats.planning_overhead() > 0.0);
}

#[test]
fn integration_mean_span_width_excludes_corrector_evaluations() {
    // Heun evaluates the model twice per window. Dividing total layers by the
    // window count would report its spans as twice their real width, which
    // would make a strategy comparison say the opposite of the truth.
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _) = fixture_batch(&device);

    let run = |solver: SolverKind| {
        let config = MultiBlockConfig {
            strategy: Gated::uniform(Strategy::Sequential, QualityGateConfig::lenient()),
            solver,
            num_steps: Some(3),
            ..MultiBlockConfig::default()
        };
        model.sample_multi_block(&pixels, &config, &mut StdRng::seed_from_u64(2)).1
    };

    let euler = run(SolverKind::Euler);
    let heun = run(SolverKind::Heun);
    assert!(heun.model_calls > euler.model_calls, "Heun must cost more calls");
    assert_eq!(
        heun.mean_span_width(),
        euler.mean_span_width(),
        "the same span policy must report the same span width regardless of solver"
    );
}

#[test]
fn integration_lookahead_costs_more_than_greedy_planning() {
    // Lookahead is only worth having if it actually looks -- a planner that
    // silently degrades to greedy would pass every budget assertion above
    // while doing nothing.
    let device = Default::default();
    let model = tiny_model(&device);
    let (pixels, _) = fixture_batch(&device);

    let run = |budget: Budget| {
        let config = PlannedConfig { budget, max_steps: 4, ..PlannedConfig::default() };
        model.sample_planned(&pixels, &config, &mut StdRng::seed_from_u64(31))
    };

    let (_, greedy_stats, greedy_trace) = run(Budget::greedy());
    let (_, deep_stats, deep_trace) = run(Budget {
        max_evaluations: 64,
        max_depth: 2,
        beam_width: 3,
    });

    assert_eq!(greedy_trace.mean_depth(), 1.0, "depth 0 commits one step per plan");
    assert!(
        deep_trace.mean_depth() > greedy_trace.mean_depth(),
        "lookahead should reach deeper: {} vs {}",
        deep_trace.mean_depth(),
        greedy_trace.mean_depth()
    );
    assert!(
        deep_stats.model_calls > greedy_stats.model_calls,
        "and pay for it: {} vs {}",
        deep_stats.model_calls,
        greedy_stats.model_calls
    );
    assert_eq!(greedy_trace.budget_exhausted_steps, 0, "greedy planning is never cut short");
}

#[test]
fn integration_a_language_model_trains_on_a_corpus() {
    // The whole language path end to end: tokenize text, stream windows out of
    // a corpus file, take real optimizer steps, and check the loss actually
    // moves. Without this, "a causal LM trains" is an assertion about code that
    // has only ever been called one forward pass at a time.
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::tokenizer::ByteTokenizer;

    let device: Device = Default::default();
    let dir = std::env::temp_dir().join("dblocks-lm-integration");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("source.txt");
    let corpus_path = dir.join("source.bin");

    // A short, highly repetitive corpus: with a handful of steps on CPU the
    // only learnable signal has to be trivial, or the test measures noise.
    std::fs::write(&source, "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc".repeat(8)).unwrap();
    let written = TokenCorpus::tokenize_file(&source, &corpus_path).unwrap();
    assert!(written > 64);

    let mut corpus = TokenCorpus::streaming(&corpus_path).unwrap();
    assert!(corpus.is_streaming());

    let config = LmConfig { context: 24, ..LmConfig::tiny() };
    let mut model = LanguageModel::<B>::new(&config, &device);
    let mut optimizer = AdamWConfig::new().init();
    let mut rng = StdRng::seed_from_u64(17);

    let span = 0..model.num_layers();
    let mut first = f32::NAN;
    let mut last = f32::NAN;

    for step in 0..12 {
        let windows = corpus.sample_batch(4, config.context - 1, &mut rng).unwrap();
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| *t as i64))
            .collect();
        let tokens = Tensor::<B, 1, burn::tensor::Int>::from_ints(flat.as_slice(), &device)
            .reshape([4, config.context]);

        let (loss, metrics) = model.next_token_loss(tokens, span.clone());
        assert!(metrics.loss.is_finite(), "step {step} produced a non-finite loss");
        assert!(metrics.tokens_counted > 0, "padding must not swallow every target");
        if step == 0 {
            first = metrics.loss;
            // An untrained tied head should sit near ln(vocab).
            assert!(
                (metrics.loss - (config.vocab_size as f32).ln()).abs() < 1.5,
                "initial loss {} is far from ln(vocab) = {}",
                metrics.loss,
                (config.vocab_size as f32).ln()
            );
        }
        last = metrics.loss;

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        assert!(!grads.is_empty(), "the LM must receive gradients");
        model = optimizer.step(3e-3, model, grads);
    }

    assert!(
        last < first,
        "loss should fall on a trivially repetitive corpus: {first} -> {last}"
    );

    // ...and the trained model still decodes consistently through both paths.
    let prompt = ByteTokenizer::new().encode("abc");
    let plain = model.generate(
        &prompt,
        6,
        &diffusionblocks::lm::Sampling::Greedy,
        &mut StdRng::seed_from_u64(1),
        &device,
    );
    let cached = model.generate_cached(
        &prompt,
        6,
        &diffusionblocks::lm::Sampling::Greedy,
        &mut StdRng::seed_from_u64(1),
        &device,
    );
    assert_eq!(plain, cached, "the cache must survive training too");
}
