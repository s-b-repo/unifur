//! Exact-resume tests (roadmap Phase 28), in their own binary.
//!
//! The backend's random stream is a process-wide global. A resumed run is
//! bit-identical to an uninterrupted one because the trainer reseeds that
//! stream at the top of every step -- which holds in a training process, and
//! holds here only if no *other* test draws from the stream between a reseed
//! and the draws it governs. Cargo runs each integration test binary as its
//! own process, and the lock below keeps these two tests from overlapping each
//! other, so this file is the one place the guarantee can be asserted.
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
use burn::tensor::backend::BackendTypes;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

type B = Autodiff<NdArray<f32>>;
type Device = <B as BackendTypes>::Device;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("dblocks-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn logged_losses(path: &std::path::Path) -> Vec<(usize, String)> {
    // The `loss` field's exact text per step: bit-identity is asserted on the
    // printed decimal, which is a function of the bits.
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            (v["step"].as_u64().unwrap() as usize, v["loss"].to_string())
        })
        .collect()
}

#[test]
fn integration_resume_is_bit_identical_for_the_image_trainer() {
    let _guard = serial();
    // Roadmap Phase 28: a run of six steps and a run of three steps resumed
    // for three more must produce the same weights, the same EMA shadow and
    // the same logged losses, to the bit -- with every stateful feature on:
    // EMA, the uncertainty head and its optimizer, importance sampling, block
    // loss normalization, gradient accumulation and dropout.
    use diffusionblocks::checkpoint::{canonical_hash_hex, TrainState};
    use diffusionblocks::train::{train, TrainConfig};

    let dir_a = scratch("resume-a");
    let dir_b = scratch("resume-b");
    let base = TrainConfig {
        image_size: 32,
        num_labels: 10,
        batch_size: 4,
        num_blocks: 2,
        steps: 6,
        log_every: 1,
        accumulate: 1,
        ema_decay: Some(0.9),
        uncertainty: 0.5,
        importance_bins: 4,
        normalize_block_loss: true,
        checks: diffusionblocks::quality::TrainingChecks::none(),
        ..TrainConfig::default()
    };
    let full = TrainConfig {
        out_dir: Some(dir_a.clone()),
        checkpoint_every: 3,
        log_file: Some(dir_a.join("log.jsonl")),
        ..base.clone()
    };
    let (model_a, summary_a) = train(&full).unwrap();
    assert_eq!(
        summary_a.periodic_checkpoints.len(),
        1,
        "{:?}",
        summary_a.periodic_checkpoints
    );
    let (at, halfway) = summary_a.periodic_checkpoints[0].clone();
    assert_eq!(at, 3);
    let final_a = summary_a.checkpoint.clone().expect("final checkpoint");
    assert!(TrainState::dir_for(&final_a).join("state.json").exists());

    let resumed = TrainConfig {
        resume: Some(halfway.clone()),
        out_dir: Some(dir_b.clone()),
        log_file: Some(dir_b.join("log.jsonl")),
        ..base.clone()
    };
    let (model_b, summary_b) = train(&resumed).unwrap();
    assert_eq!(summary_b.resumed_from_step, 3);
    assert_eq!(summary_b.steps_taken, summary_a.steps_taken);
    let final_b = summary_b.checkpoint.clone().expect("final checkpoint");
    // Content-addressed names embed the canonical parameter hash: identical
    // live weights mean identical file names.
    assert_eq!(
        final_a.file_name(),
        final_b.file_name(),
        "live weights differ after resume"
    );
    // `train` returns the EMA shadow when EMA is on; it must match too.
    assert_eq!(
        canonical_hash_hex::<B, _>(&model_a),
        canonical_hash_hex::<B, _>(&model_b),
        "EMA shadow differs after resume"
    );
    let losses_a = logged_losses(&dir_a.join("log.jsonl"));
    let losses_b = logged_losses(&dir_b.join("log.jsonl"));
    let tail_a: Vec<_> = losses_a.iter().filter(|(s, _)| *s >= 3).collect();
    let tail_b: Vec<_> = losses_b.iter().collect();
    assert_eq!(
        tail_a, tail_b,
        "logged losses after the resume point differ"
    );

    // Corruption is refused by name, not loaded.
    let optimizer_file = TrainState::dir_for(&halfway).join("optimizer.mpk");
    let mut bytes = std::fs::read(&optimizer_file).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&optimizer_file, bytes).unwrap();
    let err = train(&TrainConfig {
        steps: 4,
        ..resumed.clone()
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("corrupted"), "unhelpful error: {err}");

    // A different dataset is refused before any step runs.
    std::fs::remove_dir_all(TrainState::dir_for(&halfway)).unwrap();
    let (model_c, summary_c) = train(&TrainConfig {
        steps: 4,
        ..resumed.clone()
    })
    .unwrap();
    assert_eq!(
        summary_c.resumed_from_step, 0,
        "without a state directory only the weights load"
    );
    drop(model_c);
    let (_, summary_d) = train(&TrainConfig {
        steps: 4,
        resume: Some(final_a.clone()),
        num_labels: 10,
        ..base.clone()
    })
    .unwrap();
    assert_eq!(
        summary_d.resumed_from_step, 6,
        "resuming a finished run continues from its last step"
    );
    let err = train(&TrainConfig {
        steps: 8,
        resume: Some(final_a),
        num_labels: 7,
        ..base
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("refusing to resume"), "unhelpful error: {err}");
}

#[test]
fn integration_resume_is_bit_identical_for_the_lm_trainer() {
    let _guard = serial();
    use diffusionblocks::checkpoint::canonical_hash_hex;
    use diffusionblocks::corpus::TokenCorpus;
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::train::{train_lm, LmTrainConfig};

    let dir = scratch("lm-resume");
    let source = dir.join("text.txt");
    std::fs::write(
        &source,
        "all work and no play makes jack a dull boy. ".repeat(30),
    )
    .unwrap();
    let corpus_path = dir.join("text.bin");
    TokenCorpus::tokenize_file(&source, &corpus_path).unwrap();
    let mut corpus = TokenCorpus::in_memory(&corpus_path).unwrap();

    let device: Device = Default::default();
    let config = LmConfig::tiny();
    <B as burn::tensor::backend::Backend>::seed(&device, 3);
    let init = LanguageModel::<B>::new(&config, &device).unwrap();

    let base = LmTrainConfig {
        steps: 6,
        batch_size: 2,
        log_every: 0,
        model_config: Some(config.clone()),
        ..Default::default()
    };
    let full = LmTrainConfig {
        out_dir: Some(dir.join("a")),
        checkpoint_every: 3,
        ..base.clone()
    };
    let (model_a, report_a) = train_lm(init.clone(), &mut corpus, &full, &device).unwrap();
    let (_, halfway) = report_a.periodic_checkpoints[0].clone();

    // Resume into a *fresh* model: the weights come from the file.
    <B as burn::tensor::backend::Backend>::seed(&device, 99);
    let fresh = LanguageModel::<B>::new(&config, &device).unwrap();
    let resumed = LmTrainConfig {
        out_dir: Some(dir.join("b")),
        resume: Some(halfway),
        ..base
    };
    let (model_b, report_b) = train_lm(fresh, &mut corpus, &resumed, &device).unwrap();
    assert_eq!(report_b.resumed_from_step, 3);
    assert_eq!(report_a.steps_taken, report_b.steps_taken);
    assert_eq!(
        report_a.last_loss.to_bits(),
        report_b.last_loss.to_bits(),
        "{} vs {}",
        report_a.last_loss,
        report_b.last_loss
    );
    assert_eq!(
        canonical_hash_hex::<B, _>(&model_a),
        canonical_hash_hex::<B, _>(&model_b)
    );
    assert_eq!(
        report_a.checkpoint.unwrap().file_name(),
        report_b.checkpoint.unwrap().file_name()
    );
}

#[test]
fn integration_masked_lm_resume_preserves_weighting_and_refuses_corruption() -> anyhow::Result<()> {
    let _guard = serial();
    use diffusionblocks::checkpoint::{canonical_hash_hex, TrainState};
    use diffusionblocks::corpus::{mask_path, TokenCorpus};
    use diffusionblocks::lm::{LanguageModel, LmConfig};
    use diffusionblocks::train::{train_lm, LmAccumulationWeighting, LmTrainConfig};

    let dir = scratch("lm-masked-resume");
    let corpus_path = dir.join("masked.bin");
    let ids: Vec<u16> = (0..96).map(|index| 65 + (index % 26)).collect();
    let masks: Vec<u8> = (0..96).map(|index| u8::from(index % 12 >= 5)).collect();
    TokenCorpus::write(&corpus_path, &ids)?;
    TokenCorpus::write_masks(&mask_path(&corpus_path), &masks)?;
    let mut corpus = TokenCorpus::in_memory(&corpus_path)?;
    corpus.open_mask()?;

    let device: Device = Default::default();
    let shape = LmConfig::tiny();
    <B as burn::tensor::backend::Backend>::seed(&device, 42);
    let init = LanguageModel::<B>::new(&shape, &device)?;
    diffusionblocks::tensor_ext::force_initialization(&init);
    let config = LmTrainConfig {
        steps: 4,
        batch_size: 1,
        accumulate: 2,
        log_every: 1,
        log_path: Some(dir.join("full.jsonl")),
        model_config: Some(shape.clone()),
        checkpoint_every: 2,
        out_dir: Some(dir.join("full")),
        ..Default::default()
    };
    let (full, report) = train_lm(init, &mut corpus, &config, &device)?;
    assert_eq!(report.periodic_checkpoints.len(), 1);
    let (at, checkpoint) = &report.periodic_checkpoints[0];
    assert_eq!(*at, 2);
    let state_dir = TrainState::dir_for(checkpoint);
    let state = TrainState::for_model(checkpoint)?.expect("masked LM sidecar");
    assert_eq!(state.config["accumulation_weighting"], "tokens");
    assert_eq!(report.steps_taken, 2);

    let resume = LmTrainConfig {
        log_path: Some(dir.join("resume.jsonl")),
        out_dir: Some(dir.join("resume")),
        resume: Some(checkpoint.clone()),
        ..config.clone()
    };
    let fresh = || LanguageModel::<B>::new(&shape, &device);
    let (continued, resumed) = train_lm(fresh()?, &mut corpus, &resume, &device)?;
    assert_eq!(resumed.resumed_from_step, 2);
    assert_eq!(resumed.tokens_seen, report.tokens_seen);
    assert_eq!(resumed.steps_taken, report.steps_taken);
    assert_eq!(canonical_hash_hex(&full), canonical_hash_hex(&continued));
    assert_eq!(
        logged_losses(&dir.join("full.jsonl"))[1..],
        logged_losses(&dir.join("resume.jsonl"))
    );

    // The optimizer and RNG advance even when the weights stay frozen.
    // Both requested checkpoint boundaries must survive the repeated hash.
    let (_, frozen) = train_lm(
        fresh()?,
        &mut corpus,
        &LmTrainConfig {
            lr: 0.0,
            log_every: 0,
            log_path: None,
            out_dir: Some(dir.join("frozen")),
            ..config.clone()
        },
        &device,
    )?;
    let (_, earlier) = &frozen.periodic_checkpoints[0];
    let later = frozen.checkpoint.expect("final frozen checkpoint");
    assert_ne!(earlier, &later);
    let earlier_state = TrainState::for_model(earlier)?.expect("earlier frozen checkpoint state");
    let later_state = TrainState::for_model(&later)?.expect("later frozen checkpoint state");
    assert_eq!((earlier_state.step, later_state.step), (2, 4));
    earlier_state.verify_files(&TrainState::dir_for(earlier))?;
    later_state.verify_files(&TrainState::dir_for(&later))?;
    let earlier_model = diffusionblocks::checkpoint::load(fresh()?, earlier, &device)?;
    let later_model = diffusionblocks::checkpoint::load(fresh()?, &later, &device)?;
    assert_eq!(
        canonical_hash_hex(&earlier_model),
        canonical_hash_hex(&later_model)
    );

    let reject = LmTrainConfig {
        log_path: Some(dir.join("rejected.jsonl")),
        out_dir: Some(dir.join("rejected")),
        ..resume
    };
    let error = train_lm(
        fresh()?,
        &mut corpus,
        &LmTrainConfig {
            accumulate: 1,
            ..reject.clone()
        },
        &device,
    )
    .unwrap_err();
    assert!(error.to_string().contains("accumulate"), "{error}");
    let error = train_lm(
        fresh()?,
        &mut corpus,
        &LmTrainConfig {
            steps: 3,
            ..reject.clone()
        },
        &device,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("accumulation boundary"),
        "{error}"
    );
    let mut legacy_state = state.clone();
    legacy_state
        .config
        .as_object_mut()
        .expect("configuration object")
        .remove("accumulation_weighting");
    legacy_state.write(&state_dir)?;
    let error = train_lm(
        fresh()?,
        &mut corpus,
        &LmTrainConfig {
            accumulation_weighting: LmAccumulationWeighting::Tokens,
            ..reject.clone()
        },
        &device,
    )
    .unwrap_err();
    assert!(error.to_string().contains("Microbatches"), "{error}");
    state.write(&state_dir)?;

    let optimizer = state_dir.join(state.optimizer.expect("masked LM optimizer").path);
    let mut bytes = std::fs::read(&optimizer)?;
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&optimizer, bytes)?;
    let error = train_lm(fresh()?, &mut corpus, &reject, &device).unwrap_err();
    assert!(
        error.to_string().contains("optimizer file") && error.to_string().contains("corrupted"),
        "{error}"
    );
    assert!(!dir.join("rejected").exists() && !dir.join("rejected.jsonl").exists());
    std::fs::remove_dir_all(dir)?;
    Ok(())
}
