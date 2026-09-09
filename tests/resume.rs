//! Exact-resume tests (roadmap Phase 28), in their own binary.
//!
//! The backend's random stream is a process-wide global. A resumed run is
//! bit-identical to an uninterrupted one because the trainer reseeds that
//! stream at the top of every step -- which holds in a training process, and
//! holds here only if no *other* test draws from the stream between a reseed
//! and the draws it governs. Cargo runs each integration test binary as its
//! own process, and the lock below keeps these two tests from overlapping each
//! other, so this file is the one place the guarantee can be asserted.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::BackendTypes;

type B = Autodiff<NdArray<f32>>;
type Device = <B as BackendTypes>::Device;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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
    assert_eq!(summary_a.periodic_checkpoints.len(), 1, "{:?}", summary_a.periodic_checkpoints);
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
    assert_eq!(final_a.file_name(), final_b.file_name(), "live weights differ after resume");
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
    assert_eq!(tail_a, tail_b, "logged losses after the resume point differ");

    // Corruption is refused by name, not loaded.
    let optimizer_file = TrainState::dir_for(&halfway).join("optimizer.mpk");
    let mut bytes = std::fs::read(&optimizer_file).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&optimizer_file, bytes).unwrap();
    let err = train(&TrainConfig { steps: 4, ..resumed.clone() }).unwrap_err().to_string();
    assert!(err.contains("corrupted"), "unhelpful error: {err}");

    // A different dataset is refused before any step runs.
    std::fs::remove_dir_all(TrainState::dir_for(&halfway)).unwrap();
    let (model_c, summary_c) = train(&TrainConfig { steps: 4, ..resumed.clone() }).unwrap();
    assert_eq!(summary_c.resumed_from_step, 0, "without a state directory only the weights load");
    drop(model_c);
    let (_, summary_d) = train(&TrainConfig { steps: 4, resume: Some(final_a.clone()), num_labels: 10, ..base.clone() }).unwrap();
    assert_eq!(summary_d.resumed_from_step, 6, "resuming a finished run continues from its last step");
    let err = train(&TrainConfig { steps: 8, resume: Some(final_a), num_labels: 7, ..base }).unwrap_err().to_string();
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
    std::fs::write(&source, "all work and no play makes jack a dull boy. ".repeat(30)).unwrap();
    let corpus_path = dir.join("text.bin");
    TokenCorpus::tokenize_file(&source, &corpus_path).unwrap();
    let mut corpus = TokenCorpus::in_memory(&corpus_path).unwrap();

    let device: Device = Default::default();
    let config = LmConfig::tiny();
    <B as burn::tensor::backend::Backend>::seed(&device, 3);
    let init = LanguageModel::<B>::new(&config, &device);

    let base = LmTrainConfig { steps: 6, batch_size: 2, log_every: 0, model_config: Some(config.clone()), ..Default::default() };
    let full = LmTrainConfig { out_dir: Some(dir.join("a")), checkpoint_every: 3, ..base.clone() };
    let (model_a, report_a) = train_lm(init.clone(), &mut corpus, &full, &device).unwrap();
    let (_, halfway) = report_a.periodic_checkpoints[0].clone();

    // Resume into a *fresh* model: the weights come from the file.
    <B as burn::tensor::backend::Backend>::seed(&device, 99);
    let fresh = LanguageModel::<B>::new(&config, &device);
    let resumed = LmTrainConfig { out_dir: Some(dir.join("b")), resume: Some(halfway), ..base };
    let (model_b, report_b) = train_lm(fresh, &mut corpus, &resumed, &device).unwrap();
    assert_eq!(report_b.resumed_from_step, 3);
    assert_eq!(report_a.steps_taken, report_b.steps_taken);
    assert_eq!(report_a.last_loss.to_bits(), report_b.last_loss.to_bits(), "{} vs {}", report_a.last_loss, report_b.last_loss);
    assert_eq!(canonical_hash_hex::<B, _>(&model_a), canonical_hash_hex::<B, _>(&model_b));
    assert_eq!(report_a.checkpoint.unwrap().file_name(), report_b.checkpoint.unwrap().file_name());
}

