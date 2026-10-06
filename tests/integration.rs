//! End-to-end integration tests over the full feature surface
//! (roadmap Phase 16.10 / 16.11), kept small enough for CPU CI.
//!
//! This entire file is a test binary: every function here exists to be called
//! by `cargo test`, and a test says "this must have worked" with `unwrap`. The
//! grant below is file-scoped for that reason -- see the "Testing exemptions"
//! section of `src/lib.rs` for why it is written here rather than at the crate
//! root.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::dbg_macro,
    clippy::let_underscore_must_use,
    clippy::redundant_pattern_matching,
    clippy::mem_forget,
    clippy::exit,
    clippy::print_stdout,
    clippy::print_stderr
)]
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::BackendTypes;
use burn::{tensor::Distribution, tensor::Tensor};
use diffusionblocks::consistency::{ConsistencyConfig, ConsistencyWeights};
use diffusionblocks::dblock::{DblockClassifier, DblockConfig};
use diffusionblocks::distill::DistillConfig;
use diffusionblocks::expert_index::{BoxSpec, ExpertSpec, MosmeSpec};
use diffusionblocks::flow;
use diffusionblocks::infer::{InferenceConfig, InferenceEngine};
use diffusionblocks::loopgraph::{LoopGraph, LoopGraphConfig};
use diffusionblocks::mosme::{MosmeConfig, MosmeFeedForward};
use diffusionblocks::multi_block::{
    Gated, MultiBlockConfig, PlannedConfig, SamplingStats, Strategy,
};
use diffusionblocks::planner::Budget;
use diffusionblocks::precision::{Precision, PrecisionPolicy};
use diffusionblocks::quality::{LayerGates, QualityGateConfig};
use diffusionblocks::quantize::{quantize_module, LoraConfig, QLoraLinear};
use diffusionblocks::solver::SolverKind;
use diffusionblocks::train::DefaultTrainBackend as B;
use diffusionblocks::verify;
use diffusionblocks::vit::{MoeTrunkConfig, MosmeTrunkConfig, ViTDiTConfig};
use rand::{rngs::StdRng, SeedableRng};

type Device = <B as BackendTypes>::Device;

fn lm_cli(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_dblocks"))
        .arg("lm")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn cli_lm_runtime_and_ffn_parsing() {
    for action in ["train", "generate"] {
        let out = lm_cli(&[action, "--help"]);
        assert!(out.status.success());
        let help = String::from_utf8_lossy(&out.stdout);
        for flag in ["--backend", "--device", "--ffn-kind"] {
            assert!(help.contains(flag), "missing {flag}: {help}");
        }
    }
    for args in [
        vec!["generate", "--backend", "cuda"],
        vec!["generate", "--ffn-kind", "relu"],
        vec!["train", "--corpus", "missing.bin", "--ffn-kind", "relu"],
        vec!["generate", "--backend", "cpu", "--device", "discrete:0"],
        vec!["generate", "--backend", "wgpu", "--device", "cpu"],
        vec!["generate", "--backend", "wgpu", "--device", "discrete:-1"],
        vec![
            "generate",
            "--backend",
            "wgpu",
            "--device",
            "discrete:65536",
        ],
        vec!["bench", "--backend", "wgpu"],
    ] {
        let out = lm_cli(&args);
        assert!(!out.status.success(), "accepted {args:?}");
    }
    for kind in ["gelu", "swiglu"] {
        let out = lm_cli(&[
            "generate",
            "--tiny",
            "--ffn-kind",
            kind,
            "--max-new",
            "1",
            "--prompt",
            "a",
            "--cached",
        ]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8_lossy(&out.stderr).contains("backend=cpu"));
    }
    let out = lm_cli(&[
        "generate",
        "--tiny",
        "--backend",
        "cpu",
        "--device",
        "cpu",
        "--max-new",
        "0",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn cli_lm_wgpu_requires_compiled_feature() {
    for args in [
        vec!["generate", "--backend", "wgpu"],
        vec![
            "train",
            "--corpus",
            "missing.bin",
            "--backend",
            "wgpu",
            "--device",
            "integrated:0",
        ],
    ] {
        let out = lm_cli(&args);
        assert!(!out.status.success());
        let error = String::from_utf8_lossy(&out.stderr);
        assert!(error.contains("WGPU support is not compiled in"), "{error}");
        assert!(error.contains("--features wgpu"), "{error}");
    }
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a discrete GPU with native WGPU compute support"]
fn gpu_lm_parity_cached_backward_and_cli_smoke() {
    use burn::backend::{
        wgpu::{graphics::AutoGraphicsApi, init_setup, WgpuDevice},
        Autodiff, NdArray, Wgpu,
    };
    use burn::module::{AutodiffModule, Module};
    use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
    use burn::tensor::Int;
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::vit::FfnKind;
    type Cpu = NdArray<f32>;
    type Gpu = Autodiff<Wgpu<f32, i32>>;
    let device = WgpuDevice::DiscreteGpu(0);
    let setup = init_setup::<AutoGraphicsApi>(&device, Default::default());
    assert_eq!(
        setup.adapter.get_info().device_type,
        wgpu_types::DeviceType::DiscreteGpu
    );
    let cpu_device = Default::default();
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    for kind in [FfnKind::Gelu, FfnKind::SwiGlu] {
        let cfg = LmConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_heads: 2,
            num_layers: 1,
            num_blocks: 1,
            context: 8,
            ..LmConfig::tiny()
        }
        .with_ffn_kind(kind);
        cfg.validate_ffn().unwrap();
        let cpu = LanguageModel::<Cpu>::new(&cfg, &cpu_device).unwrap();
        let hash = diffusionblocks::checkpoint::canonical_hash_hex::<Cpu, _>(&cpu);
        let bytes = recorder.record(cpu.clone().into_record(), ()).unwrap();
        let gpu = LanguageModel::<Gpu>::new(&cfg, &device)
            .unwrap()
            .load_record(recorder.load(bytes, &device).unwrap());
        assert_eq!(
            hash,
            diffusionblocks::checkpoint::canonical_hash_hex::<Gpu, _>(&gpu)
        );
        let ids = [4i32, 8, 15, 16];
        let reference: Vec<f32> = cpu
            .forward(Tensor::<Cpu, 1, Int>::from_ints(ids.as_slice(), &cpu_device).reshape([1, 4]))
            .logits
            .into_data()
            .iter()
            .collect();
        let tokens = Tensor::<Gpu, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, 4]);
        let logits = gpu.forward(tokens.clone()).logits;
        let actual: Vec<f32> = logits.clone().into_data().iter().collect();
        assert_eq!(reference.len(), actual.len());
        for (a, b) in reference.iter().zip(&actual) {
            assert!(
                (a - b).abs() < 3e-3 + 3e-3 * a.abs(),
                "{kind:?}: CPU {a}, GPU {b}"
            );
        }
        let mut cache = gpu.new_cache();
        let cached = Tensor::cat(
            (0..4)
                .map(|i| {
                    gpu.forward_cached(tokens.clone().narrow(1, i, 1), &mut cache)
                        .logits
                })
                .collect(),
            1,
        );
        assert!((cached - logits).abs().max().into_scalar() < 3e-3);
        let (loss, _) = gpu.next_token_loss(tokens.clone(), 0..cfg.num_layers);
        assert!(loss.clone().into_scalar().is_finite());
        let grads = GradientsParams::from_grads(loss.backward(), &gpu);
        assert!(!grads.is_empty());
        let mut optim = AdamWConfig::new().init();
        let updated = optim.step(1e-3, gpu, grads);
        let values: Vec<f32> = updated
            .valid()
            .forward(tokens.inner())
            .logits
            .into_data()
            .iter()
            .collect();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().zip(actual).any(|(a, b)| (a - b).abs() > 1e-6));
        let out = lm_cli(&[
            "generate",
            "--tiny",
            "--backend",
            "wgpu",
            "--device",
            "discrete:0",
            "--ffn-kind",
            kind.name(),
            "--max-new",
            "1",
            "--prompt",
            "a",
            "--cached",
        ]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8_lossy(&out.stderr).contains("backend=wgpu"));
    }
    let dir = std::env::temp_dir().join(format!("dblocks-wgpu-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("tokens.bin");
    diffusionblocks::corpus::TokenCorpus::write(&corpus, &[65; 128]).unwrap();
    let out = lm_cli(&[
        "train",
        "--corpus",
        corpus.to_str().unwrap(),
        "--tiny",
        "--ffn-kind",
        "swiglu",
        "--backend",
        "wgpu",
        "--device",
        "discrete:0",
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--out-dir",
        dir.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("done: 1 steps"));
    let path = diffusionblocks::checkpoint::latest_in_dir(&dir, "lm")
        .unwrap()
        .unwrap();
    for backend in ["cpu", "wgpu"] {
        let out = lm_cli(&[
            "generate",
            "--checkpoint",
            path.to_str().unwrap(),
            "--backend",
            backend,
            "--max-new",
            "1",
            "--prompt",
            "a",
            "--cached",
        ]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8_lossy(&out.stdout).contains("ffn=swiglu"));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a discrete GPU with native WGPU compute support"]
fn gpu_lm_resident_mosme_specialist_cli_smoke() {
    use burn::module::{Module, ModuleVisitor, Param, ParamId};
    use diffusionblocks::checkpoint::{self, TrainState};
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::train::LmTrainingScope;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Snapshot(HashMap<ParamId, Vec<f32>>);
    impl ModuleVisitor<B> for Snapshot {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            self.0
                .insert(param.id, param.val().to_data().iter::<f32>().collect());
        }
    }

    let dir = std::env::temp_dir().join(format!(
        "dblocks-wgpu-specialist-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("tokens.bin");
    TokenCorpus::write(&corpus, &(65..=72u16).cycle().take(128).collect::<Vec<_>>()).unwrap();
    let spec = expert_boxes();
    let spec_path = dir.join("mosme.json");
    spec.write(&spec_path).unwrap();
    let expert_id = "coding/python";
    let config = LmConfig::tiny().with_mosme(MosmeTrunkConfig::new(spec).with_every_n_layers(2));
    let device: Device = Default::default();
    let model = LanguageModel::<B>::new(&config, &device).unwrap();
    let base_path = checkpoint::save_content_addressed(model, &dir.join("base"), "lm").unwrap();
    let base = checkpoint::load::<B, _>(
        LanguageModel::<B>::new(&config, &device).unwrap(),
        &base_path,
        &device,
    )
    .unwrap();
    let selected = base.specialist_trainable(&config, expert_id).unwrap();
    let selected_ids = selected.ids().unwrap();
    let mut before = Snapshot::default();
    base.visit(&mut before);
    assert!(!selected_ids.is_empty() && selected_ids.len() < before.0.len());
    let out_dir = dir.join("trained");
    let log = dir.join("metrics.jsonl");
    let out = lm_cli(&[
        "train",
        "--corpus",
        corpus.to_str().unwrap(),
        "--mosme-spec",
        spec_path.to_str().unwrap(),
        "--specialist",
        expert_id,
        "--tiny",
        "--backend",
        "wgpu",
        "--device",
        "discrete:0",
        "--base-weights",
        base_path.to_str().unwrap(),
        "--steps",
        "2",
        "--batch-size",
        "1",
        "--weight-decay",
        "0.1",
        "--log-every",
        "1",
        "--log",
        log.to_str().unwrap(),
        "--out-dir",
        out_dir.to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    println!("{stderr}{stdout}");
    assert!(out.status.success(), "{stderr}\n{stdout}");
    assert!(
        stderr.contains("backend=wgpu")
            && stderr.contains("device=discrete:0")
            && stderr.contains("type=DiscreteGpu"),
        "{stderr}"
    );
    assert!(
        stdout.contains("done: 2 steps") && stdout.contains("| 0 skipped"),
        "{stdout}"
    );
    let metrics: Vec<serde_json::Value> = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(metrics.len(), 2);
    for row in &metrics {
        assert!(row["loss"].as_f64().is_some_and(f64::is_finite), "{row}");
    }

    let path = checkpoint::latest_in_dir(&out_dir, "lm").unwrap().unwrap();
    let state = TrainState::for_model(&path).unwrap().unwrap();
    state.verify_files(&TrainState::dir_for(&path)).unwrap();
    assert_eq!(state.kind, "lm");
    assert_eq!(state.step, 2);
    assert_eq!(state.extras["steps_taken"], 2);
    assert_eq!(state.extras["steps_skipped"], 0);
    assert_eq!(
        state.config["training_scope"],
        serde_json::to_value(LmTrainingScope::Specialist {
            expert_id: expert_id.into()
        })
        .unwrap()
    );
    assert_eq!(
        state.config["model_config"],
        serde_json::to_value(&config).unwrap()
    );
    assert!(state.optimizer.is_some());
    let trained = checkpoint::load::<B, _>(
        LanguageModel::<B>::new(&config, &device).unwrap(),
        &path,
        &device,
    )
    .unwrap();
    let mut after = Snapshot::default();
    trained.visit(&mut after);
    assert_eq!(before.0.len(), after.0.len());
    let mut changed = 0;
    for (id, values) in &before.0 {
        let updated = &after.0[id];
        assert_eq!(
            values.len(),
            updated.len(),
            "parameter {id:?} changed shape"
        );
        assert!(
            updated.iter().all(|value| value.is_finite()),
            "non-finite parameter {id:?}"
        );
        let identical = values
            .iter()
            .zip(updated)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        if selected_ids.contains(id) {
            changed += usize::from(!identical);
        } else {
            assert!(
                identical,
                "frozen parameter {id:?} changed on WGPU under AdamW"
            );
        }
    }
    assert_eq!(
        changed,
        selected_ids.len(),
        "every selected parameter must update"
    );
    println!(
        "{changed} selected parameters changed; {} frozen parameters stayed bit-identical",
        before.0.len() - selected_ids.len()
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a discrete GPU with native WGPU compute support"]
fn gpu_lm_compact_specialist_export_train_apply() {
    use burn::backend::{
        wgpu::{graphics::AutoGraphicsApi, init_setup, CubeBackend, WgpuDevice, WgpuRuntime},
        NdArray, Wgpu,
    };
    use burn::module::{Module, ModuleVisitor, Param, ParamId};
    use burn::tensor::{backend::Backend, Int};
    use diffusionblocks::checkpoint::{self, TrainState};
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use std::collections::HashMap;

    type Cpu = NdArray<f32>;
    type Gpu = Wgpu<f32, i32>;
    type RawGpu = CubeBackend<WgpuRuntime, f32, i32, u32>;

    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[derive(Default)]
    struct Snapshot(HashMap<ParamId, Vec<u32>>);
    impl ModuleVisitor<Cpu> for Snapshot {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Cpu, D>>) {
            let values: Vec<f32> = param.val().into_data().iter().collect();
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(self
                .0
                .insert(param.id, values.into_iter().map(f32::to_bits).collect())
                .is_none());
        }
    }
    let run = |args: &[&str]| {
        let out = lm_cli(args);
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        println!("{stderr}{stdout}");
        assert!(out.status.success(), "{args:?}\n{stderr}\n{stdout}");
        (stdout, stderr)
    };
    let state_for = |path: &std::path::Path| {
        let state = TrainState::for_model(path).unwrap().unwrap();
        state.verify_files(&TrainState::dir_for(path)).unwrap();
        assert_eq!(state.kind, "lm");
        state
    };
    let scratch = Scratch(std::env::temp_dir().join(format!(
        "dblocks-wgpu-compact-{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
    )));
    std::fs::create_dir(&scratch.0).unwrap();
    let corpus = scratch.0.join("tokens.bin");
    TokenCorpus::write(&corpus, &(65..=72u16).cycle().take(128).collect::<Vec<_>>()).unwrap();
    let cpu_device = Default::default();
    let device = WgpuDevice::DiscreteGpu(0);
    let setup = init_setup::<AutoGraphicsApi>(&device, Default::default());
    let info = setup.adapter.get_info();
    assert_eq!(info.device_type, wgpu_types::DeviceType::DiscreteGpu);
    println!(
        "compact regression adapter: {} ({:?})",
        info.name, info.device_type
    );
    let probe = Tensor::<RawGpu, 1>::zeros([1], &device)
        .into_primitive()
        .tensor();
    let client = probe.client.clone();
    drop(probe);
    let memory = |stage: &str, experts: usize| {
        Gpu::sync(&device).unwrap();
        let usage = client.memory_usage().unwrap();
        println!("E={experts} {stage}: synchronized CubeCL allocator live={} reserved={} padding={} bytes; peak unavailable (not a training-peak measurement)",
            usage.bytes_in_use, usage.bytes_reserved, usage.bytes_padding);
    };
    let expert_id = "flat/1";
    let dimensions = LmConfig {
        hidden_size: 16,
        intermediate_size: 32,
        cond_hidden_size: 16,
        frequency_embedding_size: 16,
        num_heads: 2,
        num_layers: 2,
        num_blocks: 1,
        context: 8,
        ..LmConfig::tiny()
    };
    let donor_config = dimensions
        .clone()
        .with_mosme(MosmeTrunkConfig::new(MosmeSpec::flat(2)).with_every_n_layers(2));
    let (donor, donor_compact_config) = LanguageModel::<Cpu>::new(&donor_config, &cpu_device)
        .unwrap()
        .compact_specialist(&donor_config, expert_id)
        .unwrap();
    let donor_selected = donor
        .specialist_trainable(&donor_compact_config, expert_id)
        .unwrap();
    let mut donor_snapshot = Snapshot::default();
    donor.visit(&mut donor_snapshot);
    let input = Tensor::<Cpu, 2, Int>::from_ints([[65, 66, 67, 68, 69, 70]], &cpu_device);
    let mut counts = Vec::new();
    let mut source_counts = Vec::new();
    for experts in [2, 8, 32, 128] {
        let dir = scratch.0.join(format!("e-{experts}"));
        let config = dimensions
            .clone()
            .with_mosme(MosmeTrunkConfig::new(MosmeSpec::flat(experts)).with_every_n_layers(2));
        let source = LanguageModel::<Cpu>::new(&config, &cpu_device)
            .unwrap()
            .apply_specialist(&config, &donor, &donor_compact_config, expert_id)
            .unwrap();
        let selected = source.specialist_trainable(&config, expert_id).unwrap();
        let selected_ids = selected.ids().unwrap();
        let mut before = Snapshot::default();
        source.visit(&mut before);
        for (id, donor_id) in selected_ids.iter().zip(donor_selected.ids().unwrap()) {
            assert_eq!(
                before.0[id], donor_snapshot.0[donor_id],
                "selected initialization differs at E={experts}"
            );
        }
        assert_eq!(selected_ids.len(), donor_selected.ids().unwrap().len());
        assert!(before.0.len() > selected_ids.len());
        let source_path = checkpoint::save_content_addressed(source.clone(), &dir, "base").unwrap();
        TrainState {
            format_version: checkpoint::STATE_FORMAT_VERSION,
            kind: "lm".into(),
            step: 0,
            seed: 0,
            host_rng: serde_json::Value::Null,
            config: serde_json::json!({"model_config": config}),
            build: checkpoint::BuildInfo::current(),
            datasets: Vec::new(),
            model: checkpoint::model_entry(&source_path).unwrap(),
            optimizer: None,
            ema: None,
            head: None,
            head_optimizer: None,
            extras: serde_json::json!({}),
            saved_unix_secs: checkpoint::unix_now(),
        }
        .write(&TrainState::dir_for(&source_path))
        .unwrap();
        let source_checksum = checkpoint::file_sha256_hex(&source_path).unwrap();
        let exported_dir = dir.join("exported");
        let (stdout, _) = run(&[
            "export-specialist",
            "--checkpoint",
            source_path.to_str().unwrap(),
            "--specialist",
            expert_id,
            "--out",
            exported_dir.to_str().unwrap(),
        ]);
        assert!(stdout.contains("CPU; weights only"), "{stdout}");
        let compact_path = checkpoint::latest_in_dir(&exported_dir, "export-specialist")
            .unwrap()
            .unwrap();
        let exported_state = state_for(&compact_path);
        assert!(exported_state.optimizer.is_none());
        assert_eq!(
            exported_state.config["training_scope"]["expert_id"],
            expert_id
        );
        let compact_config: LmConfig =
            serde_json::from_value(exported_state.config["model_config"].clone()).unwrap();
        assert_eq!(
            serde_json::to_value(&compact_config).unwrap(),
            serde_json::to_value(&donor_compact_config).unwrap()
        );
        assert_eq!(compact_config.mosme.as_ref().unwrap().spec.num_experts(), 1);
        assert_eq!(
            compact_config
                .mosme
                .as_ref()
                .unwrap()
                .spec
                .position(expert_id),
            Some((0, 0))
        );
        let compact = checkpoint::load::<Cpu, _>(
            LanguageModel::<Cpu>::new(&compact_config, &cpu_device).unwrap(),
            &compact_path,
            &cpu_device,
        )
        .unwrap();
        counts.push(compact.num_params());
        source_counts.push(source.num_params());
        assert!(compact.num_params() < source.num_params());
        let expected = source
            .forward_specialist(input.clone(), &config, expert_id)
            .unwrap()
            .logits;
        let actual = compact
            .forward_specialist(input.clone(), &compact_config, expert_id)
            .unwrap()
            .logits;
        let error = (expected.clone() - actual).abs().max().into_scalar();
        assert_eq!(error, 0.0, "CPU forced logits at E={experts}");
        println!(
            "E={experts} source_params={} compact_params={} CPU forced-logit max_error={error}",
            source.num_params(),
            compact.num_params()
        );
        memory("before compact load", experts);
        let gpu = checkpoint::load::<Gpu, _>(
            LanguageModel::<Gpu>::new(&compact_config, &device).unwrap(),
            &compact_path,
            &device,
        )
        .unwrap();
        assert_eq!(gpu.num_params(), compact.num_params());
        assert_eq!(
            checkpoint::canonical_hash_hex::<Gpu, _>(&gpu),
            checkpoint::canonical_hash_hex::<Cpu, _>(&compact)
        );
        memory("compact loaded", experts);
        let gpu_logits = gpu
            .forward_specialist(
                Tensor::<Gpu, 2, Int>::from_ints([[65, 66, 67, 68, 69, 70]], &device),
                &compact_config,
                expert_id,
            )
            .unwrap()
            .logits;
        let gpu_values: Vec<f32> = gpu_logits.into_data().iter().collect();
        let reference: Vec<f32> = expected.into_data().iter().collect();
        assert_eq!(gpu_values.len(), reference.len());
        for (a, b) in reference.iter().zip(&gpu_values) {
            assert!(
                b.is_finite() && (a - b).abs() < 3e-3 + 3e-3 * a.abs(),
                "E={experts}: CPU {a}, compact GPU {b}"
            );
        }
        memory("compact forward completed", experts);
        drop(gpu);
        Gpu::sync(&device).unwrap();
        Gpu::memory_cleanup(&device);
        if experts == 128 {
            let trained_dir = dir.join("trained");
            let log = dir.join("metrics.jsonl");
            let (stdout, stderr) = run(&[
                "train",
                "--corpus",
                corpus.to_str().unwrap(),
                "--base-weights",
                compact_path.to_str().unwrap(),
                "--specialist",
                expert_id,
                "--backend",
                "wgpu",
                "--device",
                "discrete:0",
                "--steps",
                "2",
                "--batch-size",
                "1",
                "--weight-decay",
                "0.1",
                "--log-every",
                "1",
                "--log",
                log.to_str().unwrap(),
                "--out-dir",
                trained_dir.to_str().unwrap(),
            ]);
            assert!(
                stderr.contains("backend=wgpu")
                    && stderr.contains("device=discrete:0")
                    && stderr.contains("type=DiscreteGpu"),
                "{stderr}"
            );
            assert!(
                stderr.contains(&info.name),
                "CLI selected a different adapter: {stderr}"
            );
            assert!(
                stdout.contains("done: 2 steps") && stdout.contains("| 0 skipped"),
                "{stdout}"
            );
            let metrics: Vec<serde_json::Value> = std::fs::read_to_string(&log)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(metrics.len(), 2);
            assert!(metrics
                .iter()
                .all(|row| row["loss"].as_f64().is_some_and(f64::is_finite)));
            let trained_path = checkpoint::latest_in_dir(&trained_dir, "lm")
                .unwrap()
                .unwrap();
            let trained_state = state_for(&trained_path);
            assert_eq!(trained_state.step, 2);
            assert_eq!(trained_state.extras["steps_taken"], 2);
            assert_eq!(trained_state.extras["steps_skipped"], 0);
            assert!(trained_state.optimizer.is_some());
            assert_eq!(
                trained_state.config["model_config"],
                exported_state.config["model_config"]
            );
            assert_eq!(
                trained_state.config["training_scope"],
                exported_state.config["training_scope"]
            );
            let trained = checkpoint::load::<Cpu, _>(
                LanguageModel::<Cpu>::new(&compact_config, &cpu_device).unwrap(),
                &trained_path,
                &cpu_device,
            )
            .unwrap();
            let mut compact_before = Snapshot::default();
            let mut compact_after = Snapshot::default();
            compact.visit(&mut compact_before);
            trained.visit(&mut compact_after);
            let compact_selected = compact
                .specialist_trainable(&compact_config, expert_id)
                .unwrap();
            assert_eq!(compact_before.0.len(), compact_after.0.len());
            for (id, values) in &compact_before.0 {
                if compact_selected.ids().unwrap().contains(id) {
                    assert_ne!(
                        values, &compact_after.0[id],
                        "compact selected tensor did not train"
                    );
                } else {
                    assert_eq!(
                        values, &compact_after.0[id],
                        "compact shared tensor changed"
                    );
                }
            }
            let applied_dir = dir.join("applied");
            let (stdout, _) = run(&[
                "apply-specialist",
                "--checkpoint",
                source_path.to_str().unwrap(),
                "--trained",
                trained_path.to_str().unwrap(),
                "--specialist",
                expert_id,
                "--out",
                applied_dir.to_str().unwrap(),
            ]);
            assert!(stdout.contains("CPU; weights only"), "{stdout}");
            let applied_path = checkpoint::latest_in_dir(&applied_dir, "apply-specialist")
                .unwrap()
                .unwrap();
            let applied_state = state_for(&applied_path);
            assert_eq!(
                applied_state.config["model_config"],
                serde_json::to_value(&config).unwrap()
            );
            assert_eq!(applied_state.config["training_scope"]["mode"], "joint");
            assert!(applied_state.optimizer.is_none());
            let applied = checkpoint::load::<Cpu, _>(
                LanguageModel::<Cpu>::new(&config, &cpu_device).unwrap(),
                &applied_path,
                &cpu_device,
            )
            .unwrap();
            let mut after = Snapshot::default();
            applied.visit(&mut after);
            assert_eq!(before.0.len(), after.0.len());
            for (id, values) in &before.0 {
                if selected_ids.contains(id) {
                    assert_ne!(values, &after.0[id], "selected tensor did not change");
                } else {
                    assert_eq!(values, &after.0[id], "shared/router/sibling tensor changed");
                }
            }
            let expected_apply = source
                .apply_specialist(&config, &trained, &compact_config, expert_id)
                .unwrap();
            assert_eq!(
                checkpoint::canonical_hash_hex::<Cpu, _>(&applied),
                checkpoint::canonical_hash_hex::<Cpu, _>(&expected_apply)
            );
            let trained_logits = trained
                .forward_specialist(input.clone(), &compact_config, expert_id)
                .unwrap()
                .logits;
            let applied_logits = applied
                .forward_specialist(input.clone(), &config, expert_id)
                .unwrap()
                .logits;
            assert_eq!(
                (trained_logits - applied_logits).abs().max().into_scalar(),
                0.0
            );
            println!("E=128: {} selected tensors changed; {} shared/router/sibling tensors bit-identical; losses={:?}",
                selected_ids.len(), before.0.len() - selected_ids.len(), metrics.iter().map(|row| &row["loss"]).collect::<Vec<_>>());
        }
        assert_eq!(
            checkpoint::file_sha256_hex(&source_path).unwrap(),
            source_checksum
        );
    }
    assert!(
        counts.windows(2).all(|pair| pair[0] == pair[1]),
        "{counts:?}"
    );
    assert!(
        source_counts.windows(2).all(|pair| pair[0] < pair[1]),
        "{source_counts:?}"
    );
    std::fs::remove_dir_all(&scratch.0).unwrap();
}

fn tiny_model(device: &Device) -> DblockClassifier<B> {
    <B as burn::tensor::backend::Backend>::seed(device, 7);
    DblockClassifier::<B>::new(
        &ViTDiTConfig::tiny(10),
        &DblockConfig {
            num_blocks: 2,
            ..DblockConfig::default()
        },
        device,
    )
    .unwrap()
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
    assert_eq!(
        metrics.balance_loss, 0.0,
        "a dense trunk has no balance loss"
    );
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
        &ViTDiTConfig {
            num_hidden_layers: 6,
            ..ViTDiTConfig::tiny(10)
        },
        &DblockConfig {
            num_blocks: 3,
            ..DblockConfig::default()
        },
        &device,
    )
    .unwrap();
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
        assert!(
            value >= 0.0,
            "{name} is an MSE and cannot be negative: {value}"
        );
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
    let (_, plain) = model.training_step(
        pixels.clone(),
        labels.clone(),
        0.05,
        &mut StdRng::seed_from_u64(4),
    );

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
    assert!(
        values.iter().all(|&v| (0..10).contains(&v)),
        "labels out of range: {values:?}"
    );
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
        Strategy::Hybrid {
            k: 2,
            warmup_frac: 0.3,
        },
        Strategy::Adaptive {
            k_max: 2,
            conf_threshold: 1.0,
        },
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
    assert_eq!(
        never.reduced_precision_windows, 0,
        "switch above sigma_max must not trigger"
    );

    let (_, coarse) = run(PrecisionPolicy::full(Precision::Bf16));
    assert!(
        coarse.reduced_precision_windows > 0,
        "bf16 everywhere must be reported"
    );
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
        &DistillConfig {
            teacher_substeps: 2,
            ..DistillConfig::default()
        },
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
        balance_bias: false,
        num_experts: 4,
        top_k: 2,
        every_n_layers: 2,
        z_level: 1e-3,
    });
    let model = DblockClassifier::<B>::new(
        &cfg,
        &DblockConfig {
            num_blocks: 2,
            ..DblockConfig::default()
        },
        &device,
    )
    .unwrap();
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
        assert!(
            row.windows(2).all(|w| w[0].1 >= w[1].1),
            "top-k must be sorted"
        );
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
        &DblockConfig {
            num_blocks: 2,
            ..DblockConfig::default()
        },
        &device,
    )
    .unwrap();
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
    assert!(
        !grads.is_empty(),
        "the executed span must receive gradients"
    );
}

#[test]
fn integration_expert_index_round_trips_through_a_file() {
    // The artifact an inference engine consumes: written by training, read
    // back without touching the weights.
    let device = Default::default();
    let spec = expert_boxes();
    let cfg = MosmeConfig::new(32, 8, spec.clone()).with_intermediate_size(64);
    let layer = MosmeFeedForward::<B>::new(&cfg, &device);

    let index = layer
        .index(&spec, "model-abc", "vit.layers.1.mlp", 8)
        .unwrap();
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
    assert!(
        report.passed(),
        "certificate suite failed:\n{}",
        report.render()
    );
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
        Budget {
            max_evaluations: 1,
            max_depth: 3,
            beam_width: 4,
        },
        Budget {
            max_evaluations: 6,
            max_depth: 1,
            beam_width: 2,
        },
        Budget {
            max_evaluations: 40,
            max_depth: 2,
            beam_width: 3,
        },
    ] {
        let config = PlannedConfig {
            budget,
            max_steps: 6,
            ..PlannedConfig::default()
        };
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
        budget: Budget {
            max_evaluations: 32,
            max_depth: 1,
            beam_width: 2,
        },
        max_steps: 8,
        ..PlannedConfig::default()
    };

    let run = || model.sample_planned(&pixels, &config, &mut StdRng::seed_from_u64(23));
    let (first_logits, _, first) = run();
    let (second_logits, _, second) = run();

    assert_eq!(
        first.steps, second.steps,
        "the planned path must be reproducible"
    );
    let a: Vec<f32> = first_logits
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let b: Vec<f32> = second_logits
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    assert_eq!(a, b);

    let mut previous = f64::INFINITY;
    for step in &first.steps {
        assert!(
            step.sigma < previous,
            "sigma must fall: {} !< {previous}",
            step.sigma
        );
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
        budget: Budget {
            max_evaluations: 24,
            max_depth: 1,
            beam_width: 2,
        },
        max_steps: 2,
        ..PlannedConfig::default()
    };
    let (_, stats, trace) = model.sample_planned(&pixels, &tight, &mut StdRng::seed_from_u64(41));

    assert!(trace.forced_final_step, "the step cap should have bound");
    assert_eq!(
        trace.final_sigma, floor,
        "the trajectory must end at the floor"
    );
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
        model
            .sample_multi_block(&pixels, &config, &mut StdRng::seed_from_u64(2))
            .1
    };

    let euler = run(SolverKind::Euler);
    let heun = run(SolverKind::Heun);
    assert!(
        heun.model_calls > euler.model_calls,
        "Heun must cost more calls"
    );
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
        let config = PlannedConfig {
            budget,
            max_steps: 4,
            ..PlannedConfig::default()
        };
        model.sample_planned(&pixels, &config, &mut StdRng::seed_from_u64(31))
    };

    let (_, greedy_stats, greedy_trace) = run(Budget::greedy());
    let (_, deep_stats, deep_trace) = run(Budget {
        max_evaluations: 64,
        max_depth: 2,
        beam_width: 3,
    });

    assert_eq!(
        greedy_trace.mean_depth(),
        1.0,
        "depth 0 commits one step per plan"
    );
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
    assert_eq!(
        greedy_trace.budget_exhausted_steps, 0,
        "greedy planning is never cut short"
    );
}

#[test]
fn integration_global_scope_and_bias_balancing_train_a_moe_trunk() {
    // Roadmap 23.4-23.6 end to end on the image trunk: a flat-MoE model
    // trains with the global-batch load window and loss-free bias balancing
    // on, reports routing statistics per block, and its selection biases move
    // away from zero while the weights it does not own stay untouched by them.
    use diffusionblocks::schedule::BalanceScope;
    use diffusionblocks::train::{train, TrainConfig};
    use diffusionblocks::vit::MoeTrunkConfig;

    // The full 32x32 preset: `with_image_size` accepts only the two dataset
    // sizes, so a smaller model is not on offer through `TrainConfig`.
    let config = TrainConfig {
        image_size: 32,
        num_labels: 10,
        batch_size: 4,
        num_blocks: 2,
        steps: 6,
        log_every: 1,
        accumulate: 2,
        moe: Some(MoeTrunkConfig {
            num_experts: 4,
            top_k: 2,
            every_n_layers: 1,
            z_level: 1e-3,
            balance_bias: true,
        }),
        balance_scope: BalanceScope::Global,
        bias_balance_rate: 1e-3,
        checks: diffusionblocks::quality::TrainingChecks::none(),
        ..TrainConfig::default()
    };
    let (model, summary) = train(&config).unwrap();
    assert_eq!(
        summary.steps_taken + summary.steps_skipped + summary.steps_accumulated,
        6
    );
    assert_eq!(
        (summary.steps_taken, summary.steps_accumulated),
        (3, 3),
        "accumulate 2: half the micro-batches step, half are folded"
    );
    assert!(
        summary.health.has_routing(),
        "routing statistics must reach the health report"
    );
    let rendered = summary.health.render();
    assert!(
        rendered.contains("load H") && rendered.contains("token H"),
        "{rendered}"
    );

    let biases = model.balance_biases();
    assert!(
        !biases.is_empty(),
        "every sparse layer carries a selection bias"
    );
    let moved = biases.iter().flatten().filter(|b| **b != 0.0).count();
    assert!(
        moved > 0,
        "six steps of nudging must move some bias: {biases:?}"
    );
    // Every entry is an integer multiple of the rate: the bias only ever moves
    // by exactly +-rate per step.
    for b in biases.iter().flatten() {
        let steps = b / 1e-3;
        assert!(
            (steps - steps.round()).abs() < 1e-3,
            "bias {b} is not a multiple of the rate"
        );
    }
}

#[test]
fn integration_lm_trainer_nudges_its_router_biases() {
    // The same mechanism on the language path, through `train_lm`.
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::train::{train_lm, LmTrainConfig};
    use diffusionblocks::vit::MoeTrunkConfig;

    let device: Device = Default::default();
    let dir = std::env::temp_dir().join("dblocks-lm-bias-integration");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("text.txt");
    std::fs::write(
        &source,
        "the quick brown fox jumps over the lazy dog. ".repeat(20),
    )
    .unwrap();
    let corpus_path = dir.join("text.bin");
    TokenCorpus::tokenize_file(&source, &corpus_path).unwrap();
    let mut corpus = TokenCorpus::in_memory(&corpus_path).unwrap();

    let config = LmConfig {
        moe: Some(MoeTrunkConfig {
            num_experts: 3,
            top_k: 1,
            every_n_layers: 2,
            z_level: 1e-3,
            balance_bias: true,
        }),
        ..LmConfig::tiny()
    };
    let model = LanguageModel::<B>::new(&config, &device).unwrap();
    let (trained, report) = train_lm(
        model,
        &mut corpus,
        &LmTrainConfig {
            steps: 4,
            batch_size: 2,
            log_every: 0,
            bias_balance_rate: 1e-3,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    assert_eq!(report.steps_taken, 4);
    let biases = trained.balance_biases();
    assert_eq!(biases.len(), 2, "two of four tiny layers are sparse");
    assert!(biases.iter().flatten().any(|b| *b != 0.0), "{biases:?}");
}

#[test]
fn integration_two_synthetic_sources_train_as_mixture_and_composite() {
    // Roadmap 29.1 on the image trunk: two sources, weighted 3:1. A mixture
    // takes every batch from one source; a composite slices every batch 3 + 1.
    use diffusionblocks::mix::MixMode;
    use diffusionblocks::train::{train, DatasetChoice, TrainConfig};
    let base = TrainConfig {
        image_size: 32,
        num_labels: 10,
        batch_size: 4,
        num_blocks: 2,
        steps: 6,
        log_every: 1,
        extra_datasets: vec![DatasetChoice::Synthetic],
        dataset_weights: vec![3.0, 1.0],
        checks: diffusionblocks::quality::TrainingChecks::none(),
        ..TrainConfig::default()
    };
    let (_, mixture) = train(&TrainConfig {
        mix_mode: MixMode::Mixture,
        ..base.clone()
    })
    .unwrap();
    assert_eq!(mixture.sources.names.len(), 2);
    assert_eq!(
        mixture.sources.batches.iter().sum::<usize>(),
        6,
        "one source per step"
    );
    assert_eq!(mixture.sources.samples.iter().sum::<usize>(), 24);
    let (_, composite) = train(&TrainConfig {
        mix_mode: MixMode::Composite,
        ..base
    })
    .unwrap();
    assert_eq!(
        composite.sources.batches,
        vec![6, 6],
        "both sources in every step"
    );
    assert_eq!(composite.sources.samples, vec![18, 6], "3 + 1 of every 4");
}

#[test]
fn integration_lm_trains_on_two_corpora_with_a_teacher_and_a_negative_teacher() {
    // Roadmap 29.1-29.3 on the language path: a labeled and an unlabeled
    // corpus as a composite, a teacher to distil from, and a negative teacher
    // whose confident choices are charged.
    use diffusionblocks::antipattern::Labeler;
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::lm::{LanguageModel, LmConfig, Unlikelihood};
    use diffusionblocks::mix::{CorpusMix, MixMode, MixWeights};
    use diffusionblocks::train::{train_lm_mixed, LmTrainConfig, LmTrainInputs};

    let device: Device = Default::default();
    let dir = std::env::temp_dir().join(format!("dblocks-lm-multisource-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bad = dir.join("bad.txt");
    std::fs::write(&bad, "try:\n    f()\nexcept:\n    pass\n".repeat(20)).unwrap();
    let bad_bin = dir.join("bad.bin");
    TokenCorpus::tokenize_file(&bad, &bad_bin).unwrap();
    TokenCorpus::label_file(&bad_bin, &Labeler::builtin().expect("built-in rules")).unwrap();
    let good = dir.join("good.txt");
    std::fs::write(&good, "def f():\n    return 1\n".repeat(30)).unwrap();
    let good_bin = dir.join("good.bin");
    TokenCorpus::tokenize_file(&good, &good_bin).unwrap();

    let mut labeled = TokenCorpus::in_memory(&bad_bin).unwrap();
    labeled.open_labels().unwrap();
    let mut plain = TokenCorpus::in_memory(&good_bin).unwrap();
    let mut mix = CorpusMix::new(
        vec![&mut labeled, &mut plain],
        MixWeights::new(&[1.0, 1.0]).unwrap(),
        MixMode::Composite,
    )
    .unwrap();
    assert!(mix.any_labels());

    let config = LmConfig {
        context: 24,
        ..LmConfig::tiny()
    };
    let student = LanguageModel::<B>::new(&config, &device).unwrap();
    let inputs = LmTrainInputs::<B> {
        teachers: vec![
            (LanguageModel::<B>::new(&config, &device).unwrap(), 1.0),
            (LanguageModel::<B>::new(&config, &device).unwrap(), 2.0),
        ],
        negative_teacher: Some(LanguageModel::<B>::new(&config, &device).unwrap()),
    };
    let train_config = LmTrainConfig {
        steps: 4,
        batch_size: 4,
        log_every: 0,
        penalty: Unlikelihood::new(1.0),
        distill_weight: 0.5,
        negative_confidence: 0.0,
        negative_penalty: 1.0,
        ..Default::default()
    };
    let (_, report) = train_lm_mixed(student, &mut mix, &inputs, &train_config, &device).unwrap();
    assert_eq!(report.steps_taken, 4);
    assert_eq!(report.sources.names.len(), 2);
    assert_eq!(report.sources.batches, vec![4, 4]);
    assert!(
        report.penalized_tokens > 0,
        "the labeled corpus contributes negatives"
    );
    assert!(
        report.negative_teacher_tokens > 0,
        "at confidence 0 the negative teacher proposes at every differing position"
    );
    assert!(
        report.last_distill_loss > 0.0,
        "two random teachers differ from the student"
    );
    assert!(report.last_loss.is_finite());
}

#[test]
fn integration_merging_saved_checkpoints_averages_them() {
    use diffusionblocks::checkpoint::save_content_addressed;
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::merge::{flatten, merge_checkpoints};

    let device: Device = Default::default();
    let dir = std::env::temp_dir().join(format!("dblocks-merge-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let config = LmConfig::tiny();
    let a = LanguageModel::<B>::new(&config, &device).unwrap();
    let b = LanguageModel::<B>::new(&config, &device).unwrap();
    let (fa, fb) = (flatten::<B, _>(&a), flatten::<B, _>(&b));
    let pa = save_content_addressed(a, &dir, "lm").unwrap();
    let pb = save_content_addressed(b, &dir, "lm").unwrap();
    let template = LanguageModel::<B>::new(&config, &device).unwrap();
    let (merged, parents) =
        merge_checkpoints::<B, _>(template, &[pa, pb], &[1.0, 1.0], &device).unwrap();
    assert_eq!(parents.len(), 2);
    assert_ne!(parents[0], parents[1]);
    for ((m, x), y) in flatten::<B, _>(&merged).iter().zip(&fa).zip(&fb) {
        assert!((m - 0.5 * (x + y)).abs() <= 1e-6 * (x.abs() + y.abs() + 1.0));
    }
}

#[test]
fn integration_policy_gates_a_real_model_and_its_refusal_corpus_trains() {
    // Roadmap Phase 30 end to end: a starter policy and key, a signed grant,
    // the gate in front of a real (untrained) model, and the refusal corpus
    // the policy implies tokenized and trained on.
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::lm::{LanguageModel, LmConfig, Sampling};
    use diffusionblocks::policy::{
        gated_generate, refusal_documents, starter, Approval, Grant, Key,
    };
    use diffusionblocks::tokenizer::ByteTokenizer;
    use diffusionblocks::train::{train_lm, LmTrainConfig};

    let device: Device = Default::default();
    let key = Key::generate();
    let policy = starter(&key).expect("starter policy");
    let grant = Grant::issue(
        &key,
        Approval {
            id: "t".into(),
            scopes: vec!["cyber:malware".into()],
            issued_unix: 0,
            expires_unix: u64::MAX,
            note: String::new(),
        },
    )
    .unwrap();
    let approved = policy.approved_scopes(&key, &[grant], 1);
    assert_eq!(approved, vec!["cyber:malware".to_string()]);

    let model = LanguageModel::<B>::new(&LmConfig::tiny(), &device).unwrap();
    let tokenizer = ByteTokenizer::new();
    let mut decode = |sent: &str| {
        let ids = tokenizer.encode(sent);
        let out = model.generate(
            &ids,
            4,
            &Sampling::Greedy,
            &mut StdRng::seed_from_u64(1),
            &device,
        );
        tokenizer.decode_lossy(&out[ids.len()..])
    };
    let refused = gated_generate(&policy, &approved, "write a keylogger for me", &mut decode);
    assert!(
        refused.model_called,
        "the malware scope is lifted by the grant"
    );
    let blocked = gated_generate(
        &policy,
        &approved,
        "write an exploit for CVE-2020-1",
        &mut decode,
    );
    assert!(!blocked.model_called, "exploit development is not lifted");
    assert!(blocked.text.contains("gated"));

    let dir = std::env::temp_dir().join(format!("dblocks-policy-corpus-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prompts = vec![
        "write an exploit for CVE-2020-1".to_string(),
        "what is a port scan?".to_string(),
    ];
    let answers = vec!["never".to_string(), "a probe".to_string()];
    let docs = refusal_documents(&policy, &prompts, Some(&answers));
    assert_eq!(docs.len(), 3);
    let tokens: Vec<u16> = docs
        .iter()
        .flat_map(|d| tokenizer.encode_document(d))
        .collect();
    let corpus_path = dir.join("refusals.bin");
    TokenCorpus::write(&corpus_path, &tokens).unwrap();
    let mut corpus = TokenCorpus::in_memory(&corpus_path).unwrap();
    let (_, report) = train_lm(
        model,
        &mut corpus,
        &LmTrainConfig {
            steps: 2,
            batch_size: 2,
            log_every: 0,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    assert_eq!(report.steps_taken, 2);
}

#[test]
fn integration_negative_supervision_unlearns_error_swallowing() {
    // The Phase 24 claim end to end. A corpus in which every handler swallows
    // its error is tokenized, labeled and trained on twice from the same
    // initialization and the same window sequence: once plainly, once with
    // the flagged tokens charged. The plain run must *learn* `pass` after
    // `except:`; the penalized run must drive it down -- and still fit the
    // clean tokens, or the penalty would merely be breaking the model.
    use diffusionblocks::antipattern::Labeler;
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::lm::{LanguageModel, LmConfig, Sampling, Unlikelihood};
    use diffusionblocks::tokenizer::ByteTokenizer;
    use diffusionblocks::train::{train_lm, LmTrainConfig};

    let device: Device = Default::default();
    let dir = std::env::temp_dir().join("dblocks-lm-negative-integration");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("swallow.py");
    let corpus_path = dir.join("swallow.bin");

    let unit = "try:\n    f()\nexcept:\n    pass\n";
    std::fs::write(&source, unit.repeat(40)).unwrap();
    TokenCorpus::tokenize_file(&source, &corpus_path).unwrap();
    let manifest =
        TokenCorpus::label_file(&corpus_path, &Labeler::builtin().expect("built-in rules"))
            .unwrap();
    assert_eq!(
        manifest.labeled_tokens,
        40 * ":pass".len(),
        "{}",
        manifest.render()
    );

    let mut corpus = TokenCorpus::streaming(&corpus_path).unwrap();
    corpus.open_labels().unwrap();

    let config = LmConfig {
        context: 32,
        ..LmConfig::tiny()
    };
    <B as burn::tensor::backend::Backend>::seed(&device, 7);
    let init = LanguageModel::<B>::new(&config, &device).unwrap();

    // 100 steps rather than 30: the unlikelihood gradient scales with p, so
    // for the first few dozen steps generalization from `try:` raises p(bad)
    // faster than the charge lowers it (0.004 -> 0.018 at step 30 in the
    // Phase 24 measurement) before the charge takes over.
    let base = LmTrainConfig {
        steps: 100,
        batch_size: 8,
        lr: 3e-3,
        log_every: 0,
        ..Default::default()
    };
    let plain_cfg = LmTrainConfig {
        penalty: Unlikelihood::off(),
        ..base.clone()
    };
    let charged_cfg = LmTrainConfig {
        penalty: Unlikelihood::new(1.0),
        ..base
    };

    let (_, plain) = train_lm(init.clone(), &mut corpus, &plain_cfg, &device).unwrap();
    let (charged_model, charged) = train_lm(init, &mut corpus, &charged_cfg, &device).unwrap();

    assert_eq!(plain.steps_taken, 100);
    assert!(
        plain.penalized_tokens > 0,
        "the labeled corpus must surface labeled targets"
    );
    assert!(
        plain.last_penalized_prob > plain.first_penalized_prob && plain.last_penalized_prob > 0.25,
        "plain training should learn the anti-pattern: p(bad) {} -> {}",
        plain.first_penalized_prob,
        plain.last_penalized_prob
    );
    assert!(
        charged.last_penalized_prob < charged.first_penalized_prob,
        "penalized training should unlearn it: p(bad) {} -> {}",
        charged.first_penalized_prob,
        charged.last_penalized_prob
    );
    assert!(
        charged.last_penalized_prob < plain.last_penalized_prob / 20.0,
        "the two runs should end far apart: charged {} vs plain {}",
        charged.last_penalized_prob,
        plain.last_penalized_prob
    );
    assert!(
        charged.last_loss < charged.first_loss,
        "the clean tokens must still be learned: {} -> {}",
        charged.first_loss,
        charged.last_loss
    );

    // ...and the trained model's own choice after `except:` is no longer `pass`.
    let prompt = ByteTokenizer::new().encode("except:\n    ");
    let out = charged_model.generate(
        &prompt,
        4,
        &Sampling::Greedy,
        &mut StdRng::seed_from_u64(1),
        &device,
    );
    let continuation = ByteTokenizer::new().decode_lossy(&out[prompt.len()..]);
    assert!(
        !continuation.starts_with("pass"),
        "after negative supervision the greedy continuation is still {continuation:?}"
    );
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
    std::fs::write(
        &source,
        "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc".repeat(8),
    )
    .unwrap();
    let written = TokenCorpus::tokenize_file(&source, &corpus_path).unwrap();
    assert!(written > 64);

    let mut corpus = TokenCorpus::streaming(&corpus_path).unwrap();
    assert!(corpus.is_streaming());

    let config = LmConfig {
        context: 24,
        ..LmConfig::tiny()
    };
    let mut model = LanguageModel::<B>::new(&config, &device).unwrap();
    let mut optimizer = AdamWConfig::new().init();
    let mut rng = StdRng::seed_from_u64(17);

    let span = 0..model.num_layers();
    let mut first = f32::NAN;
    let mut last = f32::NAN;

    for step in 0..12 {
        let windows = corpus
            .sample_batch(4, config.context - 1, &mut rng)
            .unwrap();
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| *t as i64))
            .collect();
        let tokens = Tensor::<B, 1, burn::tensor::Int>::from_ints(flat.as_slice(), &device)
            .reshape([4, config.context]);

        let (loss, metrics) = model.next_token_loss(tokens, span.clone());
        assert!(
            metrics.loss.is_finite(),
            "step {step} produced a non-finite loss"
        );
        assert!(
            metrics.tokens_counted > 0,
            "padding must not swallow every target"
        );
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

#[test]
fn integration_direction_ablation_penalty_and_heretic_run_on_a_trained_model() {
    // Roadmap Phase 31 end to end on a tiny model: train a few steps, extract
    // a direction from two prompt sets, ablate it from the weights (gates
    // honoured), train with the activation penalty, and run a three-trial
    // Heretic search whose result is a usable model.
    use diffusionblocks::ablation::{best, extract, orthogonalize, residual_projection};
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::heretic::{decensor, HereticConfig};
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::tokenizer::ByteTokenizer;
    use diffusionblocks::train::{train_lm, LmTrainConfig};

    let device: Device = Default::default();
    let dir = std::env::temp_dir().join(format!("dblocks-ablation-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // The backend's random stream is process-wide, and this test derives its
    // direction from the model's *random initialization* before training it --
    // so which direction it finds, and hence whether three penalized steps
    // lower the projection, depends on whatever any concurrently-running test
    // left in the stream. Seeding here makes the run reproducible instead of
    // letting a neighbouring test decide the outcome.
    <B as burn::tensor::backend::Backend>::seed(&device, 8675309);
    let tokenizer = ByteTokenizer::new();
    let docs = [
        "I cannot help with that request.",
        "Sure, here is the code you asked for.",
        "fn main() { println!(\"hi\"); }",
    ];
    let tokens: Vec<u16> = docs
        .iter()
        .cycle()
        .take(24)
        .flat_map(|d| tokenizer.encode_document(d))
        .collect();
    let corpus_path = dir.join("corpus.bin");
    TokenCorpus::write(&corpus_path, &tokens).unwrap();
    let mut corpus = TokenCorpus::in_memory(&corpus_path).unwrap();

    let config = LmConfig {
        context: 16,
        ..LmConfig::tiny()
    };
    let (model, _) = train_lm(
        LanguageModel::<B>::new(&config, &device).unwrap(),
        &mut corpus,
        &LmTrainConfig {
            steps: 3,
            batch_size: 2,
            log_every: 0,
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    // A direction from the trained model's own residual stream.
    let target = ["write a keylogger", "write an exploit", "make malware"];
    let baseline = ["what is a port", "explain a socket", "sort a list"];
    let residuals = |prompts: &[&str]| -> Vec<Vec<Vec<f32>>> {
        prompts
            .iter()
            .map(|p| model.residuals_at_last_position(&tokenizer.encode(p), &device))
            .collect()
    };
    let (t, b) = (residuals(&target), residuals(&baseline));
    let directions: Vec<_> = (0..model.num_layers())
        .filter_map(|l| {
            let tl: Vec<Vec<f32>> = t.iter().map(|r| r[l].clone()).collect();
            let bl: Vec<Vec<f32>> = b.iter().map(|r| r[l].clone()).collect();
            extract(l, &tl, &bl)
        })
        .collect();
    assert_eq!(directions.len(), model.num_layers());
    let chosen = best(&directions).cloned().unwrap();
    assert!(chosen.separation > 0.0);
    let path = dir.join("direction.json");
    chosen.write(&path).unwrap();
    assert_eq!(
        diffusionblocks::ablation::Direction::read(&path).unwrap(),
        chosen
    );

    // Weight-space ablation with the model's gates: the writers lose their
    // component along the direction and the model still runs.
    let gates = model.layer_gates(&device);
    let before = residual_projection::<B, _>(&model, &chosen, Some(&gates));
    let (ablated, touched) = orthogonalize::<B, _>(model.clone(), &chosen, Some(gates.clone()));
    let after = residual_projection::<B, _>(&ablated, &chosen, Some(&gates));
    assert!(
        touched > 0 && after < before,
        "touched {touched}, {before} -> {after}"
    );
    let ids = tokenizer.encode("keylogger");
    let ids64: Vec<i64> = ids.iter().map(|t| i64::from(*t)).collect();
    let n = ids64.len();
    let tokens =
        Tensor::<B, 1, burn::tensor::Int>::from_ints(ids64.as_slice(), &device).reshape([1, n]);
    let logits = ablated.forward(tokens.clone()).logits;
    assert!(logits.clone().abs().max().into_scalar().is_finite());
    let ablated_at_inference = model
        .forward_ablated(tokens, &chosen.tensor::<B>(&device))
        .logits;
    assert!(ablated_at_inference.abs().max().into_scalar().is_finite());

    // Training with the penalty reports the projection and lowers it.
    let (penalized, report) = train_lm(
        model.clone(),
        &mut corpus,
        &LmTrainConfig {
            steps: 3,
            batch_size: 2,
            log_every: 0,
            direction: Some(chosen.clone()),
            direction_weight: 10.0,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    assert!(
        report.first_direction_projection > 0.0 && report.last_direction_projection.is_finite()
    );
    assert!(
        report.last_direction_projection < report.first_direction_projection,
        "{} -> {}",
        report.first_direction_projection,
        report.last_direction_projection
    );
    assert!(penalized.hidden_size() == model.hidden_size());

    // Heretic: three trials, a best, and a model that still generates.
    let heretic = HereticConfig {
        target: target.iter().map(|s| s.to_string()).collect(),
        baseline: baseline.iter().map(|s| s.to_string()).collect(),
        trials: 3,
        kl_weight: 1.0,
        max_new: 4,
        seed: 1,
        detector: None,
    };
    let (decensored, report) = decensor(model.clone(), &heretic, &device).unwrap();
    assert_eq!(report.trials.len(), 3);
    assert_eq!(report.directions.len(), model.num_layers());
    let best_trial = report.best.as_ref().unwrap();
    assert!(report.trials.iter().all(|t| t.score >= best_trial.score));
    assert!(best_trial.kl.is_finite() && (0.0..=1.0).contains(&best_trial.refusals));
    let out = decensored.generate(
        &ids,
        3,
        &diffusionblocks::lm::Sampling::Greedy,
        &mut StdRng::seed_from_u64(0),
        &device,
    );
    assert!(out.len() > ids.len());

    // The same search from the trainer makes the decensored model the run's
    // final checkpoint.
    let (_, report) = train_lm(
        model,
        &mut corpus,
        &LmTrainConfig {
            steps: 2,
            batch_size: 2,
            log_every: 0,
            heretic: Some(HereticConfig {
                trials: 2,
                ..heretic
            }),
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    assert_eq!(report.heretic.as_ref().unwrap().trials.len(), 2);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn integration_synthetic_negatives_train_the_image_trunk() {
    // Roadmap 31.3: half of every synthetic batch relabeled and charged.
    use diffusionblocks::train::{train, TrainConfig};
    let (_, report) = train(&TrainConfig {
        image_size: 32,
        num_labels: 10,
        batch_size: 4,
        num_blocks: 1,
        steps: 3,
        log_every: 1,
        synthetic_negatives: 0.5,
        negative_penalty: 1.0,
        checks: diffusionblocks::quality::TrainingChecks::none(),
        ..TrainConfig::default()
    })
    .unwrap();
    assert!(report.mean_loss.is_finite());
}

#[test]
fn integration_hybrid_schedules_and_routing_state_train_end_to_end() {
    // Phase 25: a 3:1 linear/dense trunk, a rotary sliding trunk and a MoE
    // trunk with a routing state each take real optimizer steps and leave a
    // finite, falling-or-flat loss on a tiny corpus.
    use diffusionblocks::hybrid::{AttentionSchedule, PositionKind};
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::train::{train_lm, LmTrainConfig};
    use diffusionblocks::vit::MoeTrunkConfig;

    let dir = std::env::temp_dir().join("dblocks-hybrid-integration");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("text.txt");
    std::fs::write(
        &source,
        "the quick brown fox jumps over the lazy dog. ".repeat(20),
    )
    .unwrap();
    let path = dir.join("text.bin");
    diffusionblocks::corpus::TokenCorpus::tokenize_file(&source, &path).unwrap();
    let mut corpus = diffusionblocks::corpus::TokenCorpus::in_memory(&path).unwrap();

    let device: Device = Default::default();
    let layers = LmConfig::tiny().num_layers;
    let variants: Vec<(&str, LmConfig)> = vec![
        (
            "3:1",
            LmConfig::tiny().with_attention(AttentionSchedule::parse("3:1", layers, 4, 2).unwrap()),
        ),
        (
            "rotary sliding",
            LmConfig::tiny()
                .with_attention(AttentionSchedule::parse("sliding4", layers, 4, 2).unwrap())
                .with_positions(PositionKind::Rotary),
        ),
        (
            "learned + retrieval",
            LmConfig::tiny().with_attention(
                AttentionSchedule::parse("learned,retrieval2,learned,dense", layers, 4, 2).unwrap(),
            ),
        ),
        (
            "moe + routing state",
            LmConfig {
                moe: Some(MoeTrunkConfig {
                    num_experts: 3,
                    top_k: 1,
                    every_n_layers: 2,
                    z_level: 1e-3,
                    balance_bias: false,
                }),
                ..LmConfig::tiny()
            }
            .with_routing_state(4),
        ),
    ];
    for (name, config) in variants {
        let model = LanguageModel::<B>::new(&config, &device).unwrap();
        let train = LmTrainConfig {
            steps: 4,
            batch_size: 2,
            log_every: 0,
            lr: 1e-3,
            ..Default::default()
        };
        let (_, report) = train_lm(model, &mut corpus, &train, &device).unwrap();
        assert_eq!(report.steps_taken, 4, "{name}");
        assert!(
            report.first_loss.is_finite() && report.last_loss.is_finite(),
            "{name}"
        );
        assert!(
            report.steps_skipped == 0,
            "{name}: {} steps skipped",
            report.steps_skipped
        );
    }
}

#[test]
fn integration_a_geometric_reasoner_trains_and_answers() {
    // Roadmap Phase 33 end to end: generate the geometric-question corpus,
    // train the dual-stream reasoner with the MoSME readout on the answer
    // objective, and check the loss falls and accuracy beats chance. The
    // span identity must survive training too.
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::geometry::{
        self, evaluate_geom, sample_aligned_batch, GeomConfig, GeomEval, GeometricReasoner,
    };

    let device: Device = Default::default();
    let dir = std::env::temp_dir().join(format!("dblocks-geom-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // nearest only: the relation is directly readable from metric distances,
    // and the four answer letters give chance 25% -- the CPU CI budget can
    // clear that bar; the full kind set is the GPU run's job
    let (tokens, meta) = geometry::generate_corpus(4, &["nearest"], 512, 11).unwrap();
    let corpus_path = dir.join("corpus.bin");
    TokenCorpus::write(&corpus_path, &tokens).unwrap();
    let mut corpus = TokenCorpus::in_memory(&corpus_path).unwrap();

    let mut config = GeomConfig::tiny();
    assert_eq!(config.scene_len, meta.scene_len, "tiny is the 4-point grid");
    // The corpus's own answer set, as the trainer and evaluator do from
    // `GeomMeta` (see `geom_config_for`): the loss, the accuracy and the
    // chance baseline below are all read over the closed space the questions
    // live in, not over the 259-byte vocabulary.
    config.answer_tokens = meta.answer_tokens.clone();
    // the device stream is process-global: seed it so the run is independent
    // Train through the shipped trainer, not a hand-rolled loop: the answer
    // objective needs the trainer's warmup-aware schedule and gradient
    // clipping (the early steps land where the learned metric is least
    // settled), and a raw constant-LR loop without them stalls at chance on
    // the restricted answer set. The device stream is process-global: the
    // trainer reseeds it per step from (seed, step), so the run is
    // independent of every other test's draws.
    use diffusionblocks::geometry::{train_geom, GeomTrainConfig};
    // The device stream is process-global: seed it so the initialization is
    // independent of every other test's draws (the trainer reseeds per step
    // from (seed, step) after that).
    <B as burn::tensor::backend::Backend>::seed(&device, 3311);
    let model = GeometricReasoner::<B>::new(&config, &device).unwrap();
    // 600 steps x 16 scenes: the readout needs of order ten thousand scenes
    // to ground the byte layout in the metric (the 1000-step CLI probe sits
    // at mean accuracy 0.34 over chance 0.25); 200 steps ends at chance.
    let train_config = GeomTrainConfig {
        steps: 600,
        batch_size: 16,
        lr: 1e-3,
        seed: 23,
        log_every: 0,
        out_dir: None,
        ..Default::default()
    };
    assert!(
        matches!(
            train_config.objective,
            diffusionblocks::geometry::GeomObjective::Answer
        ),
        "this test trains the clean reasoning path"
    );
    let (model, report) = train_geom(model, &mut corpus, &meta, &train_config, &device).unwrap();
    assert_eq!(report.steps_taken, 600);
    assert_eq!(report.steps_skipped, 0);
    assert!(
        report.last_loss < report.first_loss,
        "the answer loss should fall on the geometric corpus: {} -> {}",
        report.first_loss,
        report.last_loss
    );
    let chance = 1.0 / meta.answer_tokens.len() as f32;
    assert!(
        report.mean_accuracy > chance,
        "mean training accuracy {} should beat chance {chance}",
        report.mean_accuracy
    );
    // Held-out confirmation over a real sample. Deliberately *not* the final
    // 16-scene training batch: with four answer letters (chance 0.25) such a
    // batch resolves to 1/16 and lands anywhere in [0.125, 0.375] on identical
    // weights, so asserting on it tests the draw, not the model -- it failed
    // about one run in five.
    let mut eval_corpus = TokenCorpus::in_memory(&corpus_path).unwrap();
    let held_out_eval = GeomEval {
        batches: 16,
        batch_size: 32,
        refine_override: None,
        seed: 5,
    };
    let (held_out, seen) =
        evaluate_geom::<B>(&model, &mut eval_corpus, &meta, &held_out_eval, &device).unwrap();
    assert_eq!(seen, 512);
    assert!(
        held_out > chance,
        "held-out accuracy {held_out} over {seen} scenes should beat chance {chance}"
    );

    // the span identity survives training
    let mut rng = StdRng::seed_from_u64(23);
    let windows = sample_aligned_batch(&mut corpus, meta.scene_len, 2, &mut rng).unwrap();
    let flat: Vec<i64> = windows[0]
        .iter()
        .take(meta.scene_len - 1)
        .map(|t| *t as i64)
        .collect();
    let input = Tensor::<B, 1, burn::tensor::Int>::from_ints(flat.as_slice(), &device)
        .reshape([1, meta.scene_len - 1]);
    let direct = model.forward(input.clone()).unwrap().logits;
    let spanned = model
        .forward_span(input, 0..model.num_blocks())
        .unwrap()
        .logits;
    assert_eq!(
        (direct - spanned).abs().max().into_scalar(),
        0.0,
        "a full span must be the full path after training"
    );

    // evaluation runs and the refinement sweep covers several depths
    let shallow = GeomEval {
        batches: 4,
        batch_size: 16,
        refine_override: None,
        seed: 7,
    };
    let (accuracy, seen) =
        evaluate_geom::<B>(&model, &mut eval_corpus, &meta, &shallow, &device).unwrap();
    assert_eq!(seen, 64);
    assert!(accuracy.is_finite());
    for depth in 1..=2 {
        let eval = GeomEval {
            batches: 2,
            batch_size: 8,
            refine_override: Some(depth),
            seed: 7,
        };
        let (_, seen) =
            evaluate_geom::<B>(&model, &mut eval_corpus, &meta, &eval, &device).unwrap();
        assert_eq!(seen, 16);
    }
}

#[test]
fn integration_geometric_block_diffusion_trains_and_denoises() {
    // The diffusion objective (roadmap Phase 33): per-block sigma windows,
    // the clean state corrupted at the block's scale, EDM preconditioning,
    // boundary consistency at the shared sigma, and denoising-from-noise
    // inference on the trained model.
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::geometry::{self, sample_aligned_batch, GeomConfig, GeometricReasoner};
    use diffusionblocks::sigma::{block_sigmas, shared_boundary_sigma, DblockSigmaSampler};
    use rand::Rng;

    let device: Device = Default::default();
    let dir = std::env::temp_dir().join(format!("dblocks-geom-diff-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (tokens, meta) = geometry::generate_corpus(4, &["nearest"], 256, 13).unwrap();
    let corpus_path = dir.join("corpus.bin");
    TokenCorpus::write(&corpus_path, &tokens).unwrap();
    let mut corpus = TokenCorpus::in_memory(&corpus_path).unwrap();

    let config = GeomConfig::tiny();
    assert!(
        config.num_blocks >= 2,
        "the boundary consistency needs two blocks"
    );
    <B as burn::tensor::backend::Backend>::seed(&device, 3312);
    let mut model = GeometricReasoner::<B>::new(&config, &device).unwrap();
    let mut optimizer = AdamWConfig::new().init();
    let mut rng = StdRng::seed_from_u64(29);
    let bounds = block_sigmas(config.num_blocks);
    let sampler = DblockSigmaSampler::new(config.num_blocks, 0.05);

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..24 {
        let windows = sample_aligned_batch(&mut corpus, meta.scene_len, 8, &mut rng).unwrap();
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| *t as i64))
            .collect();
        let tokens_full = Tensor::<B, 1, burn::tensor::Int>::from_ints(flat.as_slice(), &device)
            .reshape([8, meta.scene_len]);
        let block_idx = rng.random_range(0..config.num_blocks);
        let sigma = sampler.sample(&mut rng, block_idx, 1)[0];
        let (z_star, _) = model.clean_state(tokens_full.clone()).unwrap();
        let block_result = model
            .block_step(tokens_full.clone(), block_idx, sigma, &z_star, 0.01)
            .unwrap();
        assert!(
            block_result.mse.is_finite() && block_result.ce.is_finite(),
            "step {step} produced a non-finite diffusion loss"
        );
        let mut loss = block_result.loss;
        if block_idx + 1 < config.num_blocks {
            let shared = shared_boundary_sigma(&bounds, block_idx);
            let at_boundary = model
                .block_step(tokens_full.clone(), block_idx, shared, &z_star, 0.01)
                .unwrap();
            let next_at_boundary = model
                .block_step(tokens_full.clone(), block_idx + 1, shared, &z_star, 0.01)
                .unwrap();
            let diff = at_boundary.x0.sub(next_at_boundary.x0);
            loss = loss.add(diff.clone().mul(diff).mean().mul_scalar(0.1));
        }
        let value = loss.clone().detach().into_scalar();
        assert!(value.is_finite(), "step {step} loss {value} is not finite");
        if step == 0 {
            first = value;
        }
        last = value;

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(1e-3, model, grads);
    }
    assert!(
        last < first,
        "the diffusion loss should fall across the sigma windows: {first} -> {last}"
    );

    // the trained blocks denoise from pure noise into a decodable answer
    let scene: String = corpus
        .window(0, meta.scene_len - 1)
        .unwrap()
        .iter()
        .map(|t| (*t as u8) as char)
        .collect();
    let query_tokens = diffusionblocks::tokenizer::ByteTokenizer::new().encode(&scene);
    let input = Tensor::<B, 1, burn::tensor::Int>::from_ints(
        query_tokens
            .iter()
            .map(|t| *t as i64)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    )
    .reshape([1, query_tokens.len()]);
    let out = model.forward_diffusion_start(input).unwrap();
    assert!(out.energy.is_finite());
    assert_eq!(out.logits.dims(), [1, config.vocab_size]);
}
