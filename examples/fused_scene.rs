//! Train and use the native fused reasoner with a CPU-sized Qwen trunk.

use anyhow::{ensure, Context, Result};
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::Backend, Int, Tensor};
use clap::{Parser, Subcommand};
use diffusionblocks::{
    bpe::BpeTokenizer,
    checkpoint::{self, TrainState},
    corpus::TokenCorpus,
    geometry::{sample_aligned_batch, GeomMeta, GeomObjective},
    geomfusion::{self, FusedQwen, FusedQwenTrainer, FusedTrainConfig, GeomFusionConfig},
    qwen::QwenArchDims,
    qwennet::{QwenTrunk, QwenTrunkConfig},
    qwentrain::attach_lora,
    tensor_ext::force_initialization,
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

type Cpu = NdArray<f32>;
type Train = Autodiff<Cpu>;
type Device = <Cpu as burn::tensor::backend::BackendTypes>::Device;
const SEED: u64 = 42;
const PROMPT: usize = 29;
const BACKBONE_FILE: &str = "backbone.mpk";

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    Warmup {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        steps: usize,
        #[arg(long)]
        max_seconds: Option<u64>,
        #[arg(long)]
        out: PathBuf,
    },
    Train {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        meta: PathBuf,
        #[arg(long)]
        backbone: PathBuf,
        #[arg(long)]
        objective: String,
        #[arg(long)]
        steps: usize,
        #[arg(long)]
        out: PathBuf,
    },
    Use {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, conflicts_with = "cases")]
        scene: Option<String>,
        #[arg(long)]
        cases: Option<PathBuf>,
        #[arg(
            long,
            help = "Use the trained diffusion block at its reference noise scale"
        )]
        diffusion: bool,
    },
}

#[derive(Serialize, Deserialize)]
struct Shape {
    trunk: QwenTrunkConfig,
    fusion: GeomFusionConfig,
}

fn shape() -> Shape {
    let dims = QwenArchDims {
        num_layers: 2,
        hidden_size: 64,
        intermediate_size: 96,
        vocab_size: 128,
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        linear_k_groups: 2,
        linear_v_heads: 4,
        linear_head_dim: 16,
        conv_kernel: 4,
        full_attention_interval: 2,
        attn_output_gate: true,
    };
    Shape {
        trunk: QwenTrunkConfig {
            dims,
            ..QwenTrunkConfig::qwen38()
        },
        fusion: GeomFusionConfig {
            slots: 4,
            dim: 8,
            refine_steps: 2,
            ..GeomFusionConfig::tiny()
        },
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

fn scene_tokenizer() -> Result<BpeTokenizer> {
    let vocab: serde_json::Map<String, Value> = "P0123456789ABCDnfdicy_?"
        .chars()
        .map(|c| (c.to_string(), Value::from(u32::from(c))))
        .collect();
    BpeTokenizer::from_json(
        &json!({
            "model": {"vocab": vocab, "merges": []},
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        })
        .to_string(),
    )
}

fn warmup(
    path: &Path,
    steps: usize,
    max_seconds: Option<u64>,
    out: &Path,
    device: &Device,
) -> Result<()> {
    ensure!(steps > 0, "warm-up steps must be positive");
    ensure!(
        max_seconds != Some(0),
        "warm-up time limit must be positive"
    );
    let mut corpus = TokenCorpus::in_memory(path)?;
    ensure!(
        corpus.len().is_multiple_of(PROMPT + 1),
        "expected whole 30-token scene records"
    );
    let shape = shape();
    Train::seed(device, SEED);
    let mut model = QwenTrunk::<Train>::new(shape.trunk, device);
    force_initialization(&model);
    let mut optimizer = AdamWConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut rng = ChaCha12Rng::seed_from_u64(SEED);
    let mut first = 0.0;
    let mut last = 0.0;
    let batch = 8;
    let started = std::time::Instant::now();
    let mut taken = 0;
    for step in 0..steps {
        let rows = sample_aligned_batch(&mut corpus, PROMPT + 1, batch, &mut rng)?;
        let input: Vec<i64> = rows
            .iter()
            .flat_map(|row| row[..PROMPT].iter().copied().map(i64::from))
            .collect();
        let labels: Vec<i64> = rows.iter().map(|row| i64::from(row[PROMPT])).collect();
        let input =
            Tensor::<Train, 1, Int>::from_ints(input.as_slice(), device).reshape([batch, PROMPT]);
        let targets = Tensor::<Train, 1, Int>::from_ints(labels.as_slice(), device);
        let logits = model
            .forward(input)
            .narrow(1, PROMPT - 1, 1)
            .reshape([batch, 128]);
        let loss = loss_fn.forward(logits, targets);
        last = loss.clone().into_scalar();
        ensure!(
            last.is_finite(),
            "non-finite Qwen warm-up loss at step {step}"
        );
        if step == 0 {
            first = last;
        }
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.001, model, gradients);
        taken += 1;
        if step % 100 == 0 || step + 1 == steps {
            println!("Qwen warm-up step {} / {steps}: loss {last:.4}", step + 1);
        }
        if max_seconds.is_some_and(|seconds| started.elapsed().as_secs() >= seconds) {
            break;
        }
    }
    let weights = checkpoint::save_content_addressed(model.valid(), out, "qwen-backbone")?;
    let name = weights
        .file_name()
        .and_then(|name| name.to_str())
        .context("backbone checkpoint lacks a UTF-8 filename")?;
    write_json(
        &out.join("warmup.json"),
        &json!({"weights": name, "steps": taken, "steps_requested": steps,
            "elapsed_secs": started.elapsed().as_secs_f64(), "first_loss": first, "last_loss": last}),
    )
}

fn train(action: Action, device: &Device) -> Result<()> {
    let Action::Train {
        corpus,
        meta,
        backbone,
        objective,
        steps,
        out,
    } = action
    else {
        anyhow::bail!("expected fused train arguments");
    };
    let mut corpus = TokenCorpus::streaming(&corpus)?;
    let meta: GeomMeta = read_json(&meta)?;
    ensure!(meta.scene_len == PROMPT + 1, "expected four-point scenes");
    let shape = shape();
    std::fs::create_dir_all(&out)?;
    let bundled_backbone = out.join(BACKBONE_FILE);
    let backbone_bytes =
        std::fs::read(&backbone).with_context(|| format!("read {}", backbone.display()))?;
    std::fs::write(&bundled_backbone, backbone_bytes)
        .with_context(|| format!("write {}", bundled_backbone.display()))?;
    let backbone_sha256 = checkpoint::file_sha256_hex(&bundled_backbone)?;
    Train::seed(device, SEED);
    let trunk = QwenTrunk::<Train>::new(shape.trunk.clone(), device);
    let mut trunk = checkpoint::load(trunk, &bundled_backbone, device)?;
    trunk.to_nf4()?;
    let model = FusedQwen::new(trunk, &shape.fusion, device)?;
    let train_cfg = FusedTrainConfig {
        steps,
        batch_size: 6,
        seed: SEED,
        objective: GeomObjective::parse(&objective)?,
        holdout: 0,
        log_every: 100,
        ema_decay: None,
        ..FusedTrainConfig::default()
    };
    let mut trainer = FusedQwenTrainer::new(model, train_cfg, device)?;
    let tokenizer = scene_tokenizer()?;
    let report =
        geomfusion::train_fused_qwen(&mut trainer, &mut corpus, &meta, &tokenizer, Some(&out))?;
    ensure!(report.steps_taken > 0, "fused training took no steps");
    let checkpoint = report
        .checkpoint
        .context("fused trainer did not write a checkpoint")?;
    let name = checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .context("fused checkpoint lacks a UTF-8 filename")?;
    write_json(&out.join("shape.json"), &shape)?;
    write_json(&out.join("geometry.json"), &meta)?;
    write_json(
        &out.join("model.json"),
        &json!({
            "checkpoint": name, "objective": objective, "steps_taken": report.steps_taken,
            "steps_skipped": report.steps_skipped, "elapsed_secs": report.elapsed_secs,
            "first_loss": report.first_loss, "last_loss": report.last_loss,
            "first_accuracy": report.first_accuracy, "last_accuracy": report.last_accuracy,
            "backbone_sha256": backbone_sha256,
        }),
    )
}

#[derive(Deserialize)]
struct Case {
    prompt: String,
    answer: String,
    kind: String,
}

fn predict_denoised(
    model: &FusedQwen<Cpu>,
    scene: &geomfusion::TokenizedScene,
    candidates: &[(u16, u32)],
    block: usize,
    sigma: f64,
    device: &Device,
) -> Result<(u8, f32)> {
    // As in training, the noncausal fusion path sees only the question prefix.
    let prefix = &scene.tokens[..=scene.query];
    let tokens = Tensor::<Cpu, 1, Int>::from_ints(prefix, device).reshape([1, prefix.len()]);
    let hidden = model.trunk.forward_hidden(tokens);
    let target = model.fusion.clean_final_state(&hidden, None)?;
    let step = model
        .fusion
        .block_diffusion_step(hidden, block, sigma, sigma, &target)?;
    let logits = model
        .trunk
        .lm_head
        .forward(step.hidden.narrow(1, scene.query, 1));
    let vocab = model.trunk.config().dims.vocab_size;
    let ids: Vec<i64> = candidates.iter().map(|&(_, id)| i64::from(id)).collect();
    let ids = Tensor::<Cpu, 1, Int>::from_ints(ids.as_slice(), device);
    let best: i64 = logits
        .reshape([vocab])
        .select(0, ids)
        .argmax(0)
        .into_scalar();
    Ok((
        u8::try_from(candidates[usize::try_from(best)?].0)?,
        step.mse_value,
    ))
}

fn use_model(
    bundle: &Path,
    scene: Option<String>,
    cases: Option<PathBuf>,
    diffusion: bool,
    device: &Device,
) -> Result<()> {
    let shape: Shape = read_json(&bundle.join("shape.json"))?;
    let card: Value = read_json(&bundle.join("model.json"))?;
    let name = card
        .get("checkpoint")
        .and_then(Value::as_str)
        .context("missing checkpoint filename")?;
    let path = bundle.join(name);
    let state = TrainState::for_model(&path)?.context("missing fused training state")?;
    ensure!(
        state.kind == "geomfusion",
        "expected a geomfusion checkpoint"
    );
    state.verify_files(&TrainState::dir_for(&path))?;
    let config: FusedTrainConfig = serde_json::from_value(state.config)?;
    let backbone_sha256 = card
        .get("backbone_sha256")
        .and_then(Value::as_str)
        .context("missing backbone checksum")?;
    let backbone = bundle.join(BACKBONE_FILE);
    ensure!(
        checkpoint::file_sha256_hex(&backbone)? == backbone_sha256,
        "backbone checksum mismatch for {}",
        backbone.display()
    );
    // The frozen NF4 tensors are skipped by the fused checkpoint record.
    let trunk = QwenTrunk::<Cpu>::new(shape.trunk, device);
    let mut trunk = checkpoint::load(trunk, &backbone, device)?;
    trunk.to_nf4()?;
    let mut model = FusedQwen::<Cpu>::new(trunk, &shape.fusion, device)?;
    attach_lora(&mut model.trunk, &config.lora, device)?;
    let model = checkpoint::load(model, &path, device)?;
    let tokenizer = scene_tokenizer()?;
    let meta: GeomMeta = read_json(&bundle.join("geometry.json"))?;
    let candidates = geomfusion::answer_token_ids(&tokenizer, &meta)?;
    let noise = if diffusion {
        ensure!(
            matches!(config.objective, GeomObjective::Diffusion),
            "--diffusion requires a diffusion-trained model"
        );
        Cpu::seed(device, SEED);
        let block = model
            .fusion
            .num_blocks()
            .checked_sub(1)
            .context("no diffusion block")?;
        Some((block, config.sigma_data))
    } else {
        None
    };
    let predict = |prompt: &str| -> Result<(String, &'static str, Option<f32>)> {
        ensure!(
            prompt.len() == PROMPT && prompt.is_ascii() && prompt.ends_with('?'),
            "expected a 29-byte four-point scene ending in '?'"
        );
        // tokenize_scene needs a label to validate the frame. Both prediction
        // paths read only the prefix through '?'.
        let row: Vec<u16> = prompt.bytes().chain([b'A']).map(u16::from).collect();
        let scene = geomfusion::tokenize_scene(&tokenizer, &row)?;
        let (answer, mse) = if let Some((block, sigma)) = noise {
            let (answer, mse) =
                predict_denoised(&model, &scene, &candidates, block, sigma, device)?;
            (answer, Some(mse))
        } else {
            (
                model.predict_answer(&scene, &candidates, None, device)?.0,
                None,
            )
        };
        let verdict = geomfusion::kernel_check(&geomfusion::parse_scene(&row)?, answer)?;
        Ok((char::from(answer).to_string(), verdict.name(), mse))
    };
    if let Some(path) = cases {
        let text = std::fs::read_to_string(path)?;
        let mut records = Vec::new();
        let mut kernel = BTreeMap::new();
        let mut by_kind = BTreeMap::<String, [usize; 2]>::new();
        let mut correct = 0;
        let mut mse_sum = 0.0;
        for line in text.lines() {
            let case: Case = serde_json::from_str(line)?;
            let (answer, verdict, mse) = predict(&case.prompt)?;
            mse_sum += f64::from(mse.unwrap_or(0.0));
            let hit = usize::from(answer == case.answer);
            correct += hit;
            *kernel.entry(verdict).or_insert(0) += 1;
            let counts = by_kind.entry(case.kind.clone()).or_insert([0, 0]);
            counts[0] += 1;
            counts[1] += hit;
            records.push(
                json!({"prompt": case.prompt, "expected": case.answer, "prediction": answer,
                                "kind": case.kind, "kernel": verdict}),
            );
        }
        ensure!(!records.is_empty(), "no held-out scenes");
        let mut report = json!({"cases": records.len(), "correct": correct,
            "accuracy": correct as f64 / records.len() as f64, "kernel": kernel,
            "by_kind_cases_and_correct": by_kind, "predictions": records});
        if let Some((block, sigma)) = noise {
            report["denoising"] = json!({"block": block, "sigma": sigma,
                "mean_latent_mse": mse_sum / records.len() as f64});
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let prompt = scene.context("supply --scene or --cases")?;
        let (answer, verdict, mse) = predict(&prompt)?;
        let mut result = json!({"scene": prompt, "answer": answer, "kernel": verdict});
        if let Some((block, sigma)) = noise {
            result["denoising"] = json!({"block": block, "sigma": sigma, "latent_mse": mse});
        }
        println!("{result}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::default();
    match Cli::parse().action {
        Action::Warmup {
            corpus,
            steps,
            max_seconds,
            out,
        } => warmup(&corpus, steps, max_seconds, &out, &device),
        action @ Action::Train { .. } => train(action, &device),
        Action::Use {
            bundle,
            scene,
            cases,
            diffusion,
        } => use_model(&bundle, scene, cases, diffusion, &device),
    }
}
