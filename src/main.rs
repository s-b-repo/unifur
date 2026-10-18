//! `dblocks` CLI: train, sample, benchmark and verify DiffusionBlocks models.

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

use burn::module::AutodiffModule as _;
use diffusionblocks::{
    accuracy::{Ensemble, Guidance, LogitNorm, ScalingCurve, ScalingPoint},
    antipattern::{Labeler, RuleSet},
    bpe::BpeTokenizer,
    cheat, checkpoint,
    corpus::{self, TokenCorpus},
    data::{SyntheticDataset, TrainDataset},
    dblock::{DblockClassifier, DblockConfig},
    expert_index::{BoxSpec, ExpertIndex, ExpertSpec, MosmeSpec},
    fieldnorm,
    geombaseline::{self, GeomBaseline, GeomBaselineConfig},
    geometry::{self, GeomConfig, GeomMeta, GeomObjective, GeometricReasoner},
    geomfusion::{self, FusedQwen, FusedTrainConfig, GeomFusionConfig, ProofConfig},
    geomkernel,
    hybrid::{AttentionMode, AttentionSchedule, PositionKind},
    infer::{InferenceConfig, InferenceEngine},
    lm::{LanguageModel, LmConfig, Sampling, Unlikelihood},
    mlcheat,
    multi_block::{Gated, MultiBlockConfig, PlannedConfig, Strategy},
    nina, ninagui,
    planner::Budget,
    precision::{Precision, PrecisionPolicy},
    profile::{format_duration, Profiler},
    quality::{LayerGates, QualityGateConfig, TrainingChecks},
    qwennet::{self, QwenTrunkConfig},
    schedule::LrSchedule,
    sigma,
    solver::SolverKind,
    tokenblocks::{self as tb, TbConfig, TbTrainParams},
    tokenizer::ByteTokenizer,
    train::{self, DatasetChoice, Objective, TrainConfig},
    verify,
    vit::{FfnKind, MoeTrunkConfig, MosmeTrunkConfig, ViTDiTConfig},
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Plain (non-autodiff) backend used by every inference-side command.
type Eval = burn::backend::NdArray<f32>;

#[derive(Parser)]
#[command(
    name = "dblocks",
    version,
    about = "DiffusionBlocks++ in Rust (Burn backend)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Model-shape flags shared by the inference-side commands.
#[derive(clap::Args, Clone)]
struct ModelArgs {
    #[arg(long, default_value_t = 32)]
    image_size: usize,
    #[arg(long, default_value_t = 10)]
    num_labels: usize,
    #[arg(long, default_value_t = 12)]
    num_hidden_layers: usize,
    #[arg(long, default_value_t = 4)]
    num_blocks: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Checkpoint to load; random weights are used when omitted.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
}

impl ModelArgs {
    fn build(&self, num_inference_steps: Option<usize>) -> Result<DblockClassifier<Eval>> {
        let device = Default::default();
        <Eval as burn::tensor::backend::Backend>::seed(&device, self.seed);

        let mut vit_cfg = ViTDiTConfig::with_image_size(self.image_size, self.num_labels);
        vit_cfg.num_hidden_layers = self.num_hidden_layers;
        let dblock_cfg = DblockConfig {
            num_blocks: self.num_blocks,
            num_inference_steps,
            ..DblockConfig::default()
        };
        let model = DblockClassifier::<Eval>::new(&vit_cfg, &dblock_cfg, &device)?;
        match &self.checkpoint {
            Some(path) => checkpoint::load::<Eval, _>(model, path, &device),
            None => Ok(model),
        }
    }
}

/// Architecture flags shared by the `lm` subcommands that build a model
/// (roadmap Phase 25). A checkpoint's own training state wins over these
/// when one is beside it, so a model is decoded with the trunk it was
/// trained as.
#[derive(clap::Args, Clone)]
struct LmArchArgs {
    /// The small configuration, for CPU smoke runs.
    #[arg(long, default_value_t = false)]
    tiny: bool,
    /// Attention schedule (roadmap 25.1): `dense`, `linear`, `learned`,
    /// `sliding<w>`, `retrieval<k>` for every layer; `3:1` (three linear per
    /// dense), `2:1@sliding64`; a letter pattern such as `LLLD`; or one mode
    /// per layer, comma-separated. Empty is dense.
    #[arg(long, default_value = "")]
    attention: String,
    /// Window for `sliding` layers named without one.
    #[arg(long, default_value_t = 64)]
    window: usize,
    /// Top-k for `retrieval` layers named without one.
    #[arg(long, default_value_t = 32)]
    retrieval_k: usize,
    /// learned | rotary | none (roadmap 25.2). Anything but `learned` lets
    /// cached decoding run past the context length.
    #[arg(long, default_value = "learned")]
    positions: String,
    /// Width of the per-token routing state carried through the layers and
    /// appended to every router's input (roadmap 25.4); 0 is none.
    #[arg(long, default_value_t = 0)]
    routing_state: usize,
    #[arg(long, default_value = "gelu", value_parser = ["gelu", "swiglu"])]
    ffn_kind: String,
    #[arg(long, value_parser = parse_positive_usize)]
    hidden_size: Option<usize>,
    #[arg(long, value_parser = parse_positive_usize)]
    num_layers: Option<usize>,
    #[arg(long, value_parser = parse_positive_usize)]
    num_heads: Option<usize>,
    #[arg(long, value_parser = parse_positive_usize)]
    intermediate_size: Option<usize>,
    #[arg(long, value_parser = parse_positive_usize)]
    context: Option<usize>,
    #[arg(long)]
    mosme_spec: Option<PathBuf>,
    #[arg(long, default_value_t = 2, value_parser = parse_positive_usize, requires = "mosme_spec")]
    mosme_every: usize,
    /// Grouped-query attention: this many key/value heads (Qwen3 style GQA).
    /// Must divide --num-heads; empty is full MHA.
    #[arg(long, value_parser = parse_positive_usize)]
    kv_heads: Option<usize>,
    /// Fraction of each head dimension the rotary embedding covers
    /// (Qwen3-Next style partial rotary, e.g. 0.25); 1.0 is full rotation.
    #[arg(long, default_value_t = 1.0)]
    rotary_fraction: f64,
    /// Qwen-style gated attention: merged heads scaled by 1+tanh(g(x)) with
    /// a zero-initialized gate (identity until trained).
    #[arg(long, default_value_t = false)]
    gated_attention: bool,
    /// QK-Norm on queries and keys before rotary (Qwen3/GLM-4.5/LLaMA-4):
    /// bounds attention scores by the head dim however far logits drift.
    #[arg(long, default_value_t = false)]
    qk_norm: bool,
    /// Trunk normalization: layernorm (every checkpoint to date) or rmsnorm.
    #[arg(long, default_value = "layernorm", value_parser = ["layernorm", "rmsnorm"])]
    norm: String,
    /// Multi-token prediction depth (Qwen MTP style): auxiliary CE on this
    /// many future offsets. 0 disables MTP exactly.
    #[arg(long, default_value_t = 0)]
    mtp_steps: usize,
    /// Weight on the MTP auxiliary loss; 0 adds nothing.
    #[arg(long, default_value_t = 0.0)]
    mtp_weight: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum LmBackend {
    Cpu,
    Wgpu,
}

#[derive(clap::Args, Clone)]
struct LmRuntimeArgs {
    #[arg(long, value_enum, default_value = "cpu")]
    backend: LmBackend,
    #[arg(long, value_name = "cpu|discrete:N|integrated:N|virtual:N")]
    device: Option<String>,
}

impl LmRuntimeArgs {
    fn validate(&self) -> Result<()> {
        match self.backend {
            LmBackend::Cpu => anyhow::ensure!(
                self.device.as_deref().is_none_or(|d| d == "cpu"),
                "--backend cpu accepts only --device cpu"
            ),
            LmBackend::Wgpu => {
                parse_gpu_device(self.device.as_deref().unwrap_or("discrete:0"))?;
                anyhow::ensure!(
                    cfg!(feature = "wgpu"),
                    "WGPU support is not compiled in; rebuild with cargo build --features wgpu --bin dblocks"
                );
            }
        }
        Ok(())
    }
}

/// The adapter classes this crate will drive. A closed set, so the parser
/// below can return one of them rather than a bare string that every caller
/// has to match on again.
#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuKind {
    Discrete,
    Integrated,
    Virtual,
}

fn parse_gpu_device(text: &str) -> Result<(&str, usize)> {
    let (kind, index) = text
        .split_once(':')
        .context("WGPU device must be discrete:N, integrated:N or virtual:N (zero-based index)")?;
    anyhow::ensure!(
        matches!(kind, "discrete" | "integrated" | "virtual"),
        "WGPU device must be discrete:N, integrated:N or virtual:N; CPU adapters are not accepted"
    );
    let index: usize = index
        .parse()
        .context("WGPU device index must be a non-negative integer")?;
    anyhow::ensure!(
        index <= u16::MAX as usize,
        "WGPU device index must be at most {}",
        u16::MAX
    );
    Ok((kind, index))
}

/// The adapter class named by `kind`, or the error naming what is accepted.
///
/// `parse_gpu_device` has already checked the spelling by the time this runs;
/// the `bail!` is here so that adding a [`GpuKind`] variant without teaching the
/// CLI about it is an error rather than a fallback.
#[cfg(feature = "wgpu")]
fn gpu_kind(kind: &str) -> Result<GpuKind> {
    match kind {
        "discrete" => Ok(GpuKind::Discrete),
        "integrated" => Ok(GpuKind::Integrated),
        "virtual" => Ok(GpuKind::Virtual),
        other => anyhow::bail!("unknown WGPU adapter class '{other}'"),
    }
}

#[cfg(feature = "wgpu")]
fn lm_wgpu_device(args: &LmRuntimeArgs) -> Result<burn::backend::wgpu::WgpuDevice> {
    use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, WgpuDevice};
    use wgpu_types::DeviceType;
    let (kind, index) = parse_gpu_device(args.device.as_deref().unwrap_or("discrete:0"))?;
    let (device, expected) = match gpu_kind(kind)? {
        GpuKind::Discrete => (WgpuDevice::DiscreteGpu(index), DeviceType::DiscreteGpu),
        GpuKind::Integrated => (WgpuDevice::IntegratedGpu(index), DeviceType::IntegratedGpu),
        GpuKind::Virtual => (WgpuDevice::VirtualGpu(index), DeviceType::VirtualGpu),
    };
    let setup =
        std::panic::catch_unwind(|| init_setup::<AutoGraphicsApi>(&device, Default::default()))
            .map_err(|err| {
                anyhow::anyhow!(
                    "WGPU initialization failed for {kind}:{index}: {}",
                    panic_message(err.as_ref())
                )
            })?;
    let info = setup.adapter.get_info();
    anyhow::ensure!(
        info.device_type == expected,
        "WGPU requested {kind}:{index}, but selected {:?} adapter '{}'; refusing fallback",
        info.device_type,
        info.name
    );
    eprintln!(
        "backend=wgpu api={:?} device={kind}:{index} name={:?} type={:?} driver={:?} driver_info={:?}",
        info.backend, info.name, info.device_type, info.driver, info.driver_info
    );
    Ok(device)
}

fn parse_positive_usize(text: &str) -> std::result::Result<usize, String> {
    let value: usize = text
        .parse()
        .map_err(|_| "expected a positive integer".to_owned())?;
    if value == 0 {
        return Err("expected a positive integer".to_owned());
    }
    Ok(value)
}

fn validate_lm_config(config: &LmConfig) -> Result<()> {
    anyhow::ensure!(
        config.hidden_size > 0
            && config.num_heads > 0
            && config.num_layers > 0
            && config.intermediate_size > 0,
        "hidden size, layer count, head count and intermediate size must be positive"
    );
    anyhow::ensure!(
        config.hidden_size % config.num_heads == 0,
        "hidden size must be divisible by head count"
    );
    config.validate_attention()?;
    anyhow::ensure!(config.context >= 2, "context must be at least 2");
    anyhow::ensure!(
        config.num_layers % config.num_blocks.max(1) == 0,
        "layer count must be divisible by block count"
    );
    if let Some(mosme) = &config.mosme {
        mosme.spec.validate()?;
        anyhow::ensure!(
            mosme.every_n_layers > 0 && mosme.every_n_layers <= config.num_layers,
            "MoSME placement must be between 1 and the layer count"
        );
    }
    config.validate_ffn()
}

impl LmArchArgs {
    fn config(&self) -> Result<LmConfig> {
        let mut config = if self.tiny {
            LmConfig::tiny()
        } else {
            LmConfig::default()
        };
        config.hidden_size = self.hidden_size.unwrap_or(config.hidden_size);
        config.num_layers = self.num_layers.unwrap_or(config.num_layers);
        config.num_heads = self.num_heads.unwrap_or(config.num_heads);
        config.intermediate_size = self.intermediate_size.unwrap_or(config.intermediate_size);
        config.context = self.context.unwrap_or(config.context);
        if config.num_layers % config.num_blocks != 0 {
            config.num_blocks = 1;
        }
        if let Some(path) = &self.mosme_spec {
            config = config.with_mosme(
                MosmeTrunkConfig::new(MosmeSpec::read(path)?).with_every_n_layers(self.mosme_every),
            );
        }
        if !self.attention.trim().is_empty() {
            let schedule = AttentionSchedule::parse(
                &self.attention,
                config.num_layers,
                self.window,
                self.retrieval_k,
            )?;
            config = config.with_attention(schedule);
        }
        config = config
            .with_positions(PositionKind::parse(&self.positions)?)
            .with_routing_state(self.routing_state);
        config.ffn_kind = match self.ffn_kind.as_str() {
            "gelu" => FfnKind::Gelu,
            "swiglu" => FfnKind::SwiGlu,
            other => anyhow::bail!("unknown FFN kind '{other}' (expected gelu|swiglu)"),
        };
        if let Some(kv) = self.kv_heads {
            config = config.with_kv_heads(kv);
        }
        config = config
            .with_rotary_fraction(self.rotary_fraction)
            .with_gated_attention(self.gated_attention)
            .with_qk_norm(self.qk_norm)
            .with_mtp(self.mtp_steps, self.mtp_weight);
        config.norm_kind = match self.norm.as_str() {
            "layernorm" => diffusionblocks::vit::NormKind::Layer,
            "rmsnorm" => diffusionblocks::vit::NormKind::Rms,
            other => anyhow::bail!("unknown norm '{other}' (expected layernorm|rmsnorm)"),
        };
        validate_lm_config(&config)?;
        Ok(config)
    }
}

/// The architecture a checkpoint was trained with, when its training state
/// records one; otherwise what the flags say.
fn lm_config_for(arch: &LmArchArgs, checkpoint: Option<&PathBuf>) -> Result<LmConfig> {
    if let Some(path) = checkpoint {
        if let Some(state) = checkpoint::TrainState::for_model(path)? {
            match state.config.get("model_config").filter(|v| !v.is_null()) {
                Some(value) => {
                    let config: LmConfig =
                        serde_json::from_value(value.clone()).map_err(|err| {
                            anyhow::anyhow!(
                                "parse the model config recorded beside {}: {err}",
                                path.display()
                            )
                        })?;
                    validate_lm_config(&config)?;
                    println!(
                        "architecture from {}: {}",
                        checkpoint::TrainState::dir_for(path).display(),
                        config.describe()
                    );
                    return Ok(config);
                }
                None => println!(
                    "{} records no architecture; building the model from the flags",
                    checkpoint::TrainState::dir_for(path).display()
                ),
            }
        }
    }
    arch.config()
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it (see the `[lints]` table in
// `Cargo.toml` and the contract in the crate docs).
#[allow(
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
mod lm_cli_tests {
    use super::*;

    fn parse_train(flags: &[&str]) -> std::result::Result<LmAction, clap::Error> {
        let args = ["dblocks", "lm", "train", "--corpus", "tokens.bin"];
        let cli = Cli::try_parse_from(args.into_iter().chain(flags.iter().copied()))?;
        let Command::Lm { action } = cli.command else {
            unreachable!()
        };
        Ok(action)
    }

    fn arch(flags: &[&str]) -> LmArchArgs {
        let LmAction::Train { arch, .. } = parse_train(flags).unwrap() else {
            unreachable!()
        };
        arch
    }

    #[test]
    fn defaults_are_unchanged() {
        for (flags, expected) in [
            (&[][..], LmConfig::default()),
            (&["--tiny"][..], LmConfig::tiny()),
        ] {
            let actual = arch(flags).config().unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        }
    }

    #[test]
    fn custom_dimensions_and_attention() {
        let config = arch(&[
            "--hidden-size",
            "48",
            "--num-layers",
            "3",
            "--num-heads",
            "6",
            "--intermediate-size",
            "96",
            "--context",
            "2",
            "--attention",
            "linear",
        ])
        .config()
        .unwrap();
        assert_eq!(
            (
                config.hidden_size,
                config.num_layers,
                config.num_heads,
                config.intermediate_size,
                config.context
            ),
            (48, 3, 6, 96, 2)
        );
        assert_eq!(config.num_blocks, 1);
        assert_eq!(config.attention.as_ref().unwrap().num_layers(), 3);
        let device = Default::default();
        let model = LanguageModel::<Eval>::new(&config, &device).unwrap();
        assert_eq!(model.num_layers(), 3);
    }

    #[test]
    fn rejects_invalid_dimensions_and_placement_flags() {
        for flag in [
            "--hidden-size",
            "--num-layers",
            "--num-heads",
            "--intermediate-size",
            "--context",
            "--mosme-every",
        ] {
            for value in ["0", "-1", "abc"] {
                assert!(parse_train(&[flag, value]).is_err(), "{flag} {value}");
            }
        }
        assert!(arch(&["--context", "1"]).config().is_err());
        assert!(arch(&["--hidden-size", "33"]).config().is_err());
        assert!(parse_train(&["--mosme-every", "1"]).is_err());
    }

    #[test]
    fn resident_flags_and_resume_conflict() {
        let LmAction::Train {
            specialist,
            base_weights,
            resume,
            ..
        } = parse_train(&["--specialist", "flat/1", "--base-weights", "base.mpk"]).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(specialist.as_deref(), Some("flat/1"));
        assert_eq!(base_weights, Some(PathBuf::from("base.mpk")));
        assert!(resume.is_none());
        assert!(parse_train(&["--specialist", "flat/1", "--resume", "resume.mpk"]).is_ok());
        assert!(parse_train(&["--base-weights", "base.mpk", "--resume", "resume.mpk"]).is_err());
    }

    #[test]
    fn compact_specialist_parsing() {
        for action in ["export-specialist", "apply-specialist"] {
            let mut args = vec![
                "dblocks",
                "lm",
                action,
                "--checkpoint",
                "original.mpk",
                "--specialist",
                "flat/1",
                "--out",
                "new-dir",
            ];
            if action == "apply-specialist" {
                assert!(Cli::try_parse_from(&args).is_err());
                args.extend(["--trained", "compact.mpk"]);
            }
            assert!(Cli::try_parse_from(&args).is_ok());
            for flags in [["--backend", "wgpu"], ["--device", "cpu"]] {
                assert!(Cli::try_parse_from(args.iter().copied().chain(flags)).is_err());
            }
            assert!(
                Cli::try_parse_from(args.iter().copied().chain(["--tiny", "--context", "8"]))
                    .is_ok()
            );
        }
        assert!(Cli::try_parse_from([
            "dblocks",
            "lm",
            "export-specialist",
            "--checkpoint",
            "original.mpk",
            "--out",
            "new-dir"
        ])
        .is_err());
    }

    #[test]
    fn compact_specialist_checkpoint_workflow() {
        use burn::module::{Module, ModuleMapper, Param};
        use burn::tensor::Tensor;
        let dir = std::env::temp_dir().join(format!("dblocks-compact-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let device = Default::default();
        let mut config = LmConfig::tiny()
            .with_mosme(MosmeTrunkConfig::new(MosmeSpec::flat(2)).with_every_n_layers(1));
        config.hidden_size = 8;
        config.num_heads = 2;
        config.num_layers = 2;
        config.num_blocks = 1;
        config.intermediate_size = 12;
        config.context = 4;
        let model = LanguageModel::<Eval>::new(&config, &device).unwrap();
        diffusionblocks::tensor_ext::force_initialization(&model);
        let base =
            checkpoint::save_content_addressed::<Eval, _>(model.clone(), &dir, "base").unwrap();
        write_derived_state(&base, "lm", serde_json::json!({})).unwrap();
        let mut state = checkpoint::TrainState::for_model(&base).unwrap().unwrap();
        state.config = serde_json::json!({"model_config": config});
        state
            .write(&checkpoint::TrainState::dir_for(&base))
            .unwrap();
        let base_checksum = checkpoint::file_sha256_hex(&base).unwrap();
        let flags = arch(&[]);
        let exported_dir = dir.join("exported");
        cmd_lm(LmAction::ExportSpecialist {
            checkpoint: base.clone(),
            specialist: "flat/1".into(),
            out: exported_dir.clone(),
            arch: flags.clone(),
        })
        .unwrap();
        let compact_path = checkpoint::latest_in_dir(&exported_dir, "export-specialist")
            .unwrap()
            .unwrap();
        let compact_state = checkpoint::TrainState::for_model(&compact_path)
            .unwrap()
            .unwrap();
        compact_state
            .verify_files(&checkpoint::TrainState::dir_for(&compact_path))
            .unwrap();
        assert!(compact_state.optimizer.is_none());
        assert_eq!(compact_state.kind, "lm");
        assert_eq!(
            compact_state.config["training_scope"]["expert_id"],
            "flat/1"
        );
        let compact_config = lm_specialist_config(Some(&compact_state), || {
            panic!("state architecture must win")
        })
        .unwrap();
        assert_eq!(
            compact_config
                .mosme
                .as_ref()
                .unwrap()
                .spec
                .experts_per_box(),
            [1]
        );
        let compact = checkpoint::load::<Eval, _>(
            LanguageModel::<Eval>::new(&compact_config, &device).unwrap(),
            &compact_path,
            &device,
        )
        .unwrap();
        assert!(compact.num_params() < model.num_params());
        let applied_dir = dir.join("untrained-applied");
        cmd_lm_specialist(&base, Some(&compact_path), "flat/1", &applied_dir, &flags).unwrap();
        assert!(cmd_lm_specialist(&base, None, "flat/1", &exported_dir, &flags).is_err());
        assert!(cmd_lm_specialist(
            &base,
            Some(&compact_path),
            "flat/0",
            &dir.join("wrong-id"),
            &flags
        )
        .is_err());
        let corpus = dir.join("tokens.bin");
        std::fs::write(
            &corpus,
            [1u16, 2, 3, 4, 5, 6, 7, 8]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let trained_dir = dir.join("trained");
        let cli = Cli::try_parse_from([
            "dblocks",
            "lm",
            "train",
            "--corpus",
            corpus.to_str().unwrap(),
            "--base-weights",
            compact_path.to_str().unwrap(),
            "--specialist",
            "flat/1",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--out-dir",
            trained_dir.to_str().unwrap(),
        ])
        .unwrap();
        let Command::Lm { action } = cli.command else {
            unreachable!()
        };
        cmd_lm(action).unwrap();
        let trained_path = checkpoint::latest_in_dir(&trained_dir, "lm")
            .unwrap()
            .unwrap();
        let applied_dir = dir.join("trained-applied");
        cmd_lm(LmAction::ApplySpecialist {
            checkpoint: base.clone(),
            trained: trained_path.clone(),
            specialist: "flat/1".into(),
            out: applied_dir.clone(),
            arch: flags.clone(),
        })
        .unwrap();
        let applied_path = checkpoint::latest_in_dir(&applied_dir, "apply-specialist")
            .unwrap()
            .unwrap();
        let applied_state = checkpoint::TrainState::for_model(&applied_path)
            .unwrap()
            .unwrap();
        applied_state
            .verify_files(&checkpoint::TrainState::dir_for(&applied_path))
            .unwrap();
        assert_eq!(
            applied_state.config["model_config"],
            serde_json::to_value(&config).unwrap()
        );
        assert_eq!(applied_state.config["training_scope"]["mode"], "joint");
        assert_eq!(
            applied_state.extras["lineage_validation"],
            "shared_tensors_only"
        );
        assert!(applied_state.optimizer.is_none());
        let trained = checkpoint::load::<Eval, _>(
            LanguageModel::<Eval>::new(&compact_config, &device).unwrap(),
            &trained_path,
            &device,
        )
        .unwrap();
        assert_ne!(
            checkpoint::canonical_hash_hex::<Eval, _>(&trained),
            checkpoint::canonical_hash_hex::<Eval, _>(&compact)
        );
        let expected = model
            .apply_specialist(&config, &trained, &compact_config, "flat/1")
            .unwrap();
        let applied = checkpoint::load::<Eval, _>(
            LanguageModel::<Eval>::new(&config, &device).unwrap(),
            &applied_path,
            &device,
        )
        .unwrap();
        assert_eq!(
            checkpoint::canonical_hash_hex::<Eval, _>(&applied),
            checkpoint::canonical_hash_hex::<Eval, _>(&expected)
        );
        struct ChangeShared;
        impl ModuleMapper<Eval> for ChangeShared {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<Eval, D>>,
            ) -> Param<Tensor<Eval, D>> {
                param.map(|tensor| tensor + 1.0)
            }
        }
        let wrong = checkpoint::save_content_addressed::<Eval, _>(
            compact.clone().map(&mut ChangeShared),
            &dir,
            "wrong-shared",
        )
        .unwrap();
        let err = cmd_lm_specialist(
            &base,
            Some(&wrong),
            "flat/1",
            &dir.join("rejected-shared"),
            &flags,
        )
        .unwrap_err();
        assert!(err.to_string().contains("shared tensor mismatch"), "{err}");
        let mut wrong_state = compact_state.clone();
        wrong_state.extras["specialist_lineage"]["parent"]["sha256"] = "wrong-parent".into();
        wrong_state
            .write(&checkpoint::TrainState::dir_for(&compact_path))
            .unwrap();
        let err = cmd_lm_specialist(
            &base,
            Some(&compact_path),
            "flat/1",
            &dir.join("rejected-parent"),
            &flags,
        )
        .unwrap_err();
        assert!(err.to_string().contains("parent lineage mismatch"), "{err}");
        wrong_state.model.sha256 = "corrupt".into();
        wrong_state
            .write(&checkpoint::TrainState::dir_for(&compact_path))
            .unwrap();
        let err = cmd_lm_specialist(
            &base,
            Some(&compact_path),
            "flat/1",
            &dir.join("rejected-checksum"),
            &flags,
        )
        .unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
        assert_eq!(checkpoint::file_sha256_hex(&base).unwrap(), base_checksum);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mosme_spec_and_placement() {
        let dir = std::env::temp_dir().join(format!("dblocks-lm-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("spec.json");
        MosmeSpec::flat(2).write(&path).unwrap();
        let path_arg = path.to_str().unwrap();
        let config = arch(&["--tiny", "--mosme-spec", path_arg])
            .config()
            .unwrap();
        let mosme = config.mosme.as_ref().unwrap();
        assert_eq!(mosme.every_n_layers, 2);
        assert_eq!(mosme.num_hierarchical_layers(config.num_layers), 2);
        assert_eq!(mosme.spec.position("flat/1"), Some((0, 1)));
        assert!(
            arch(&["--tiny", "--mosme-spec", path_arg, "--mosme-every", "5"])
                .config()
                .is_err()
        );
        assert!(
            arch(&["--tiny", "--mosme-spec", path_arg, "--ffn-kind", "swiglu"])
                .config()
                .is_err()
        );
        let config = arch(&["--tiny", "--mosme-spec", path_arg, "--mosme-every", "1"])
            .config()
            .unwrap();
        assert_eq!(config.mosme.unwrap().every_n_layers, 1);
        let mut invalid = MosmeSpec::flat(2);
        invalid.boxes.clear();
        std::fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(arch(&["--mosme-spec", path_arg]).config().is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

// `Train` carries far more flags than the other subcommands, so the enum is
// sized for it. Boxing the variant would mean a second struct definition and an
// extra indirection on a type that is constructed exactly once per process.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum LmAction {
    #[command(about = "Export an independent compact specialist checkpoint on CPU")]
    ExportSpecialist {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, value_name = "STABLE_ID")]
        specialist: String,
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        arch: LmArchArgs,
    },
    #[command(about = "Apply a compact specialist to a new full checkpoint on CPU")]
    ApplySpecialist {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        trained: PathBuf,
        #[arg(long, value_name = "STABLE_ID")]
        specialist: String,
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        arch: LmArchArgs,
    },
    /// Tokenize a UTF-8 text file into a pre-tokenized corpus.
    Tokenize {
        /// Input text file.
        #[arg(long)]
        input: PathBuf,
        /// Output corpus file (little-endian u16 tokens).
        #[arg(long)]
        out: PathBuf,
        /// Also label anti-patterns (roadmap Phase 24), writing `<out>.labels`
        /// and `<out>.labels.json` next to the corpus.
        #[arg(long, default_value_t = false)]
        label: bool,
        /// Rule set JSON for `--label`; the built-in rules when omitted.
        #[arg(long)]
        rules: Option<PathBuf>,
    },
    /// Report what a corpus contains, without loading it into memory.
    Corpus {
        #[arg(long)]
        path: PathBuf,
        /// Context length used to count training windows.
        #[arg(long, default_value_t = 256)]
        context: usize,
    },
    /// Label an existing corpus with anti-pattern categories (roadmap Phase
    /// 24). Writes `<corpus>.labels` and `<corpus>.labels.json`.
    Label {
        #[arg(long)]
        corpus: PathBuf,
        /// Rule set JSON; the built-in rules when omitted.
        #[arg(long)]
        rules: Option<PathBuf>,
    },
    /// Grade an existing corpus with supervision points. Writes
    /// `<corpus>.grades` (one signed byte per token) and its `.grades.json`
    /// manifest, which `lm train` then reads: a `-1` or `-0.5` token is charged
    /// an unlikelihood penalty instead of learned, a `+0.5` token is learned at
    /// half credit, and an ungraded or `+1` token is ordinary cross-entropy.
    ///
    /// Grades are derived from the corpus's own label sidecar: a token a heavy
    /// rule fired on is `-1`, a lightly-flagged one `-0.5`, and a token no rule
    /// touched stays *ungraded* — the rules know what is wrong, not what is
    /// right. Grade a corpus, then train: the penalty applies to graded tokens
    /// only, so a half-graded corpus is a half-supervised run, never a wrong one.
    Grade {
        #[arg(long)]
        corpus: PathBuf,
    },
    /// List the anti-pattern findings in a source file, with line numbers.
    Scan {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        rules: Option<PathBuf>,
    },
    /// Score the code quality of a source file: per-dimension breakdown and
    /// the geometric mean that the filter and regularizer would read. Writes
    /// a JSON report when `--out` is given; otherwise prints to stdout.
    Score {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "rust")]
        language: String,
        /// Include the lexical (anti-pattern) dimension.
        #[arg(long, default_value_t = true)]
        lexical: bool,
        /// Include the structural heuristics dimension.
        #[arg(long, default_value_t = true)]
        structural: bool,
        /// Include the external-analyzer dimension. Requires the
        /// `codequality-external` Cargo feature at build time and the named
        /// tool on `PATH` at run time; otherwise the dimension is silently
        /// skipped.
        #[arg(long)]
        external: Option<String>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Write the built-in rule set as JSON to extend it, or validate a rule
    /// file -- every rule must match its examples and none of its
    /// counterexamples.
    Rules {
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        check: Option<PathBuf>,
    },
    /// Train a causal language model on a corpus. With a `.labels` sidecar
    /// present the run reports the probability it assigns to flagged tokens,
    /// and `--penalty` charges for them instead of rewarding them.
    Train {
        /// Corpus from `lm tokenize`. Repeatable (roadmap 29.1): several
        /// corpora train in one run, weighted by `--corpus-weights` and
        /// combined per `--mix`; a `.labels` sidecar next to any of them is
        /// opened automatically.
        #[arg(long, action = clap::ArgAction::Append, required = true)]
        corpus: Vec<PathBuf>,
        /// Weights over the corpora, e.g. `0.7,0.3`; empty is uniform.
        #[arg(long, default_value = "")]
        corpus_weights: String,
        /// mixture | composite.
        #[arg(long, default_value = "mixture")]
        mix: String,
        /// Teacher checkpoints from `lm train` to distil from, repeatable
        /// (roadmap 29.2); weighted by `--teacher-weights`.
        #[arg(long, action = clap::ArgAction::Append)]
        teacher: Vec<PathBuf>,
        #[arg(long, default_value = "")]
        teacher_weights: String,
        /// Weight on the distillation term; `0` is off.
        #[arg(long, default_value_t = 0.0)]
        distill_weight: f64,
        #[arg(long, default_value_t = 2.0)]
        distill_temperature: f64,
        /// A checkpoint whose confident next-token choices are *charged*
        /// (roadmap 29.3): an open-weight source of bad patterns.
        #[arg(long)]
        negative_teacher: Option<PathBuf>,
        /// Charge a proposal only where the negative teacher is at least this
        /// sure of it.
        #[arg(long, default_value_t = 0.5)]
        negative_confidence: f32,
        /// Coefficient on the negative teacher's charge.
        #[arg(long, default_value_t = 1.0)]
        negative_penalty: f32,
        /// A behaviour direction from `lm direction` penalized in activation
        /// space while training (roadmap 31.2).
        #[arg(long)]
        direction: Option<PathBuf>,
        /// Weight on the direction penalty `mean((h . d)^2)`.
        #[arg(long, default_value_t = 1.0)]
        direction_weight: f64,
        /// Decensor the trained model afterwards (roadmap 31.6, Heretic):
        /// prompts the model should stop refusing, one per line.
        #[arg(long, requires = "heretic_baseline")]
        heretic_target: Option<PathBuf>,
        /// Prompts whose behaviour Heretic must preserve, one per line.
        #[arg(long)]
        heretic_baseline: Option<PathBuf>,
        #[arg(long, default_value_t = 12)]
        heretic_trials: usize,
        #[arg(long, default_value_t = 1.0)]
        heretic_kl_weight: f64,
        /// Tokens generated per target prompt when counting refusals.
        #[arg(long, default_value_t = 24)]
        heretic_max_new: usize,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        /// Learning-rate schedule over the run: constant | cosine | warmup
        /// (roadmap Phase 20, ported from the image loop). cosine warms up
        /// over the first 5% of steps and decays to 1% of the peak.
        #[arg(long, default_value = "constant", value_parser = ["constant", "cosine", "warmup"])]
        lr_schedule: String,
        /// Micro-batches per optimizer step, averaged not summed (roadmap
        /// 20.2). 1 is a single batch per step, exactly as before.
        #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
        accumulate: usize,
        /// Rescale the accumulated gradient when its global norm exceeds this
        /// (roadmap 20.3). 0 disables clipping exactly.
        #[arg(long, default_value_t = 0.0)]
        clip_norm: f32,
        /// Keep an exponential moving average of the weights and return the
        /// shadow for evaluation (roadmap 22.1). Empty disables EMA exactly.
        #[arg(long)]
        ema_decay: Option<f64>,
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f64,
        /// Unlikelihood coefficient on labeled targets (roadmap Phase 24).
        /// 0 trains plainly; requires labels when positive.
        #[arg(long, default_value_t = 0.0)]
        penalty: f32,
        /// Stream windows from disk instead of loading the corpus.
        #[arg(long, default_value_t = false)]
        streaming: bool,
        #[command(flatten)]
        arch: LmArchArgs,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 10)]
        log_every: usize,
        /// Append-mode JSONL metrics.
        #[arg(long)]
        log: Option<PathBuf>,
        #[arg(long, default_value = "checkpoints")]
        out_dir: PathBuf,
        /// Also write a checkpoint (model + training state) every n steps.
        #[arg(long, default_value_t = 0)]
        checkpoint_every: usize,
        /// Resume from a model written by `lm train`; its training state,
        /// when present, is restored and verified.
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, conflicts_with = "resume")]
        base_weights: Option<PathBuf>,
        #[arg(long, value_name = "STABLE_ID")]
        specialist: Option<String>,
    },
    /// Train the tiny model a few steps under each trunk variant (dense,
    /// flat MoE, MoE with loss-free bias balancing) and record the loss and
    /// step time per seed as experiment records (roadmap Phase 28). On CPU
    /// this measures the mechanisms' cost, not their quality.
    Bench {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long, default_value_t = 30)]
        steps: usize,
        #[arg(long, default_value = "1,2")]
        seeds: String,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        /// Which axis to vary (roadmap 25.8): `trunk` (dense, flat MoE, MoE
        /// with loss-free bias), `attention` (dense, 3:1 linear, sliding,
        /// retrieval, linear, learned), `positions` (learned, rotary, none),
        /// `routing` (MoE with and without a routing state).
        #[arg(long, default_value = "trunk")]
        axis: String,
        /// Append one record per variant here.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Extract a behaviour direction (roadmap 31.1): the normalized
    /// difference of the mean residual stream between target and baseline
    /// prompts, one candidate per layer, the best-separated one written out.
    Direction {
        #[arg(long)]
        checkpoint: PathBuf,
        /// Prompts exhibiting the behaviour, one per line.
        #[arg(long)]
        target: PathBuf,
        /// Prompts without it, one per line.
        #[arg(long)]
        baseline: PathBuf,
        /// Fix the layer instead of picking the best-separated one.
        #[arg(long)]
        layer: Option<usize>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = false)]
        tiny: bool,
    },
    /// Orthogonalize every residual-writing weight against a direction and
    /// save the result as a new checkpoint (roadmap 31.1, abliteration).
    Ablate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        direction: PathBuf,
        #[arg(long, default_value = "checkpoints")]
        out: PathBuf,
        #[arg(long, default_value_t = false)]
        tiny: bool,
    },
    /// Mean projection of the residual stream onto a direction over prompt
    /// files (roadmap 31.4): the before/after measurement of an ablation.
    DirectionScore {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        direction: PathBuf,
        /// Prompt files, one prompt per line; repeatable.
        #[arg(long, action = clap::ArgAction::Append, required = true)]
        prompts: Vec<PathBuf>,
        /// Also project the direction out at inference and report the result.
        #[arg(long, default_value_t = false)]
        ablated: bool,
        #[arg(long, default_value_t = false)]
        tiny: bool,
    },
    /// Decensor a checkpoint the Heretic way (roadmap 31.6): search
    /// per-component weighted ablations against refusals and first-token KL
    /// and save the best.
    Heretic {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long, default_value_t = 12)]
        trials: usize,
        #[arg(long, default_value_t = 1.0)]
        kl_weight: f64,
        #[arg(long, default_value_t = 24)]
        max_new: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "checkpoints")]
        out: PathBuf,
        /// Write the full report (directions, every trial, the best) here.
        #[arg(long)]
        json: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        tiny: bool,
    },
    /// Capability gating (roadmap Phase 30): blockers, refusals and the
    /// scopes signed approvals lift.
    Policy {
        #[command(subcommand)]
        action: PolicyAction,
    },
    /// Signed grants that lift policy scopes for their holder (roadmap Phase 30).
    Approvals {
        #[command(subcommand)]
        action: ApprovalsAction,
    },
    /// Build the refusal and approval training documents a policy implies
    /// and tokenize them into a corpus (roadmap 30.3).
    RefusalCorpus {
        #[arg(long)]
        policy: PathBuf,
        /// One prompt per line.
        #[arg(long)]
        prompts: PathBuf,
        /// One answer per line, aligned with the prompts; optional.
        #[arg(long)]
        answers: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Average several `lm train` checkpoints of one configuration into a
    /// new one (roadmap 29.4).
    Merge {
        #[arg(long, action = clap::ArgAction::Append, required = true)]
        input: Vec<PathBuf>,
        #[arg(long, default_value = "")]
        weights: String,
        #[arg(long, default_value = "checkpoints")]
        out: PathBuf,
        /// The inputs' architecture; read from the first input's training
        /// state when it has one.
        #[command(flatten)]
        arch: LmArchArgs,
    },
    /// Generate a continuation from an untrained model.
    ///
    /// The weights are random unless `--checkpoint` is given, so the output is
    /// noise: what this demonstrates is that the decoding paths agree, not that
    /// the model says anything.
    Generate {
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_new: usize,
        /// greedy | topk
        #[arg(long, default_value = "greedy")]
        sampling: String,
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,
        /// Look ahead this many tokens and score whole continuations
        /// (roadmap 21.5). 0 is ordinary greedy decoding.
        #[arg(long, default_value_t = 0)]
        lookahead: usize,
        /// Beam width for `--lookahead`.
        #[arg(long, default_value_t = 3)]
        beam: usize,
        /// Candidate evaluations per committed token for `--lookahead`.
        #[arg(long, default_value_t = 32)]
        budget: usize,
        /// Decode with a key/value cache (roadmap 19.6).
        #[arg(long, default_value_t = false)]
        cached: bool,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
        /// Weights from `dblocks lm train`; random when omitted.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// The checkpoint's architecture; read from its training state when
        /// it has one.
        #[command(flatten)]
        arch: LmArchArgs,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
        /// Gate the request through a policy (roadmap Phase 30): a blocked
        /// prompt is refused before any forward pass. Uses plain or cached
        /// decoding.
        #[arg(long)]
        policy: Option<PathBuf>,
        /// The policy's key, needed to verify `--grant`s.
        #[arg(long)]
        key: Option<PathBuf>,
        /// Grants to present; repeatable.
        #[arg(long, action = clap::ArgAction::Append)]
        grant: Vec<PathBuf>,
    },
    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long, default_value_t = 32)]
        batches: usize,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[command(flatten)]
        arch: LmArchArgs,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
    },
}

// The variants cannot be boxed without a second round of destructuring, and
// `Train` legitimately carries every training flag.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Command {
    /// Block-wise training.
    Train {
        /// Dataset: synthetic | cifar100 | tiny-imagenet. Repeatable
        /// (roadmap 29.1): several sources train in one run, weighted by
        /// `--dataset-weights` and combined per `--mix`. Every source must
        /// share the image size and label count.
        #[arg(long, default_value = "synthetic", action = clap::ArgAction::Append)]
        dataset: Vec<String>,
        /// Directory holding a dataset's `.bin` splits; repeat once per
        /// `--dataset` that needs one, in the same order.
        #[arg(long, action = clap::ArgAction::Append)]
        data_dir: Vec<PathBuf>,
        /// Weights over the datasets, e.g. `0.7,0.3`; empty is uniform.
        #[arg(long, default_value = "")]
        dataset_weights: String,
        /// mixture (each batch from one source, drawn by weight) |
        /// composite (every batch sliced from every source).
        #[arg(long, default_value = "mixture")]
        mix: String,
        /// Stream records from disk instead of loading the split into memory.
        #[arg(long, default_value_t = false)]
        streaming: bool,
        /// Objective: dblock | consistency | flow | distill.
        #[arg(long, default_value = "dblock")]
        objective: String,
        /// Frozen teacher checkpoint for `--objective distill`. Repeatable
        /// (roadmap 29.2): several teachers distil at once, weighted by
        /// `--teacher-weights`.
        #[arg(long, action = clap::ArgAction::Append)]
        teacher: Vec<PathBuf>,
        /// Weights over the teachers; empty is uniform.
        #[arg(long, default_value = "")]
        teacher_weights: String,
        #[arg(long, default_value_t = 32)]
        image_size: usize,
        #[arg(long, default_value_t = 100)]
        num_labels: usize,
        #[arg(long, default_value_t = 3)]
        num_blocks: usize,
        /// Sigma window extension factor.
        #[arg(long, default_value_t = 0.05)]
        gamma: f64,
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        #[arg(long, default_value_t = 0.001)]
        lr: f64,
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f64,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 20)]
        log_every: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Directory for content-addressed checkpoints.
        #[arg(long, default_value = "checkpoints")]
        out_dir: String,
        /// Optional JSONL metrics file (one object per logged step).
        #[arg(long)]
        log_file: Option<String>,
        /// Checkpoint activations during backward to reduce peak memory.
        #[arg(long, default_value_t = false)]
        grad_checkpointing: bool,
        /// Save the checkpoint on a background thread.
        #[arg(long, default_value_t = false)]
        async_save: bool,
        /// Resume from this checkpoint, or from the newest one in `--out-dir`
        /// when passed without a value. The training state beside the model
        /// (optimizer, schedules, RNG, EMA, estimators) is restored and
        /// verified when present, so the continuation is exact.
        #[arg(long, num_args = 0..=1, default_missing_value = "")]
        resume: Option<String>,
        /// Also write a checkpoint (model + training state) every n steps.
        #[arg(long, default_value_t = 0)]
        checkpoint_every: usize,
        /// Relabel this fraction of every batch with a wrong class and charge
        /// the model for believing it (roadmap 31.3): negative supervision
        /// for the image trunk. 0 is off.
        #[arg(long, default_value_t = 0.0)]
        synthetic_negatives: f64,
        /// Coefficient on the negative charge.
        #[arg(long, default_value_t = 1.0)]
        negative_penalty: f32,
        /// Disable every training-time quality check.
        #[arg(long, default_value_t = false)]
        no_checks: bool,
        /// Skip the pre-training certificate check.
        #[arg(long, default_value_t = false)]
        no_preflight: bool,
        /// Re-verify the live model every n steps (0 = never).
        #[arg(long, default_value_t = 0)]
        verify_every: usize,
        /// Boxes-of-experts spec (JSON). Enables Mixture of Specialized Micro
        /// Experts in the trunk.
        #[arg(long, conflicts_with = "moe_every")]
        mosme_spec: Option<PathBuf>,
        /// Which trunk layers get expert boxes.
        #[arg(long, default_value_t = 2)]
        mosme_every: usize,
        /// Where to write the expert index; defaults to alongside the checkpoint.
        #[arg(long)]
        index_out: Option<PathBuf>,
        /// Learning-rate schedule: constant | cosine | warmup.
        #[arg(long, default_value = "constant")]
        lr_schedule: String,
        /// Micro-batches per optimizer step.
        #[arg(long, default_value_t = 1)]
        accumulate: usize,
        /// Rescale gradients whose global norm exceeds this.
        #[arg(long)]
        clip_norm: Option<f32>,
        /// Keep an exponential moving average of the weights and return it.
        #[arg(long)]
        ema_decay: Option<f64>,
        /// Normalize each block's loss onto a common scale.
        #[arg(long, default_value_t = false)]
        normalize_block_loss: bool,
        /// Auxiliary balance-loss weight schedule: constant | anneal.
        /// Annealing holds the weight high while routing collapse is the risk,
        /// then decays it so it stops fighting expert specialization.
        #[arg(long, default_value = "constant")]
        balance_schedule: String,
        /// Starting weight for `--balance-schedule`.
        #[arg(long, default_value_t = 0.01)]
        balance_weight: f64,
        /// Over which batch the balance loss measures expert load
        /// (roadmap 23.4): micro | global. `global` averages the load over
        /// the `--accumulate` window so the router is not pushed to balance
        /// every micro-batch on its own, which inhibits specialization.
        #[arg(long, default_value = "micro")]
        balance_scope: String,
        /// Loss-free bias balancing (roadmap 23.5): move each router's
        /// selection bias by this much per step against its observed load.
        /// `0` is off. DeepSeek-V3 uses 1e-3.
        #[arg(long, default_value_t = 0.0)]
        bias_balance_rate: f32,
        /// Router z-loss weight (ST-MoE). Penalizes large routing logits,
        /// which the balance loss cannot see — the softmax is invariant to a
        /// per-row constant shift. `0.0` disables it exactly.
        #[arg(long, default_value_t = 1e-3)]
        z_level: f64,
        /// Learned per-sigma uncertainty weighting, in [0, 1] (roadmap 20.5).
        /// `0.0` is the exact identity. At its optimum the gradient becomes
        /// that of log-loss, which no per-sigma rescaling can unbalance.
        #[arg(long, default_value_t = 0.0)]
        uncertainty: f64,
        /// CDF bins for sigma importance sampling (roadmap 20.6). `0` disables
        /// it; a cold sampler is exactly plain sampling in any case.
        #[arg(long, default_value_t = 0)]
        importance_bins: usize,
        /// Replace every n-th layer's MLP with a flat mixture of experts.
        #[arg(long)]
        moe_every: Option<usize>,
        #[arg(long, default_value_t = 4)]
        moe_experts: usize,
        #[arg(long, default_value_t = 1)]
        moe_top_k: usize,
    },
    /// Run diffusion sampling and print predictions for one batch.
    Sample {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 3)]
        num_inference_steps: usize,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        /// ODE solver: euler | heun | ddim | dpmpp2m | dpmpp3m
        #[arg(long, default_value = "euler")]
        solver: String,
        /// Multi-block strategy: sequential | parallel | hybrid | adaptive
        #[arg(long, default_value = "sequential")]
        strategy: String,
        /// Parallel span width for parallel/hybrid/adaptive strategies.
        #[arg(long, default_value_t = 2)]
        k: usize,
        /// Arithmetic precision above `--precision-switch`: f32 | bf16 | f16.
        #[arg(long, default_value = "f32")]
        precision: String,
        /// Sigma below which sampling reverts to f32.
        #[arg(long, default_value_t = 0.0)]
        precision_switch: f64,
        /// Quality gate: lenient | strict | tightening
        #[arg(long, default_value = "lenient")]
        gate: String,
        /// Guidance scale (roadmap 22.5). 1.0 is the exact identity; anything
        /// else doubles the model calls.
        #[arg(long, default_value_t = 1.0)]
        guidance: f64,
        /// Fraction of the conditional estimate's spread to restore after
        /// guidance, in [0, 1].
        #[arg(long, default_value_t = 0.0)]
        guidance_rescale: f64,
        /// Logit normalization: none | temperature | l2 | standardize
        /// (roadmap 22.6). Never changes the prediction, only the confidence.
        #[arg(long, default_value = "none")]
        logit_norm: String,
        /// Temperature for `--logit-norm`.
        #[arg(long, default_value_t = 1.0)]
        logit_tau: f64,
        /// Ensemble the deterministic solvers and combine their answers
        /// (roadmap 22.4): probability | logit | vote. Empty runs a single
        /// solver.
        #[arg(long, default_value = "")]
        ensemble: String,
        /// Plan each step instead of following the schedule (roadmap 21a).
        #[arg(long, default_value_t = false)]
        planned: bool,
        /// Rollout depth for `--planned`. 0 is greedy planning.
        #[arg(long, default_value_t = 1)]
        plan_depth: usize,
        /// Beam width for `--planned`.
        #[arg(long, default_value_t = 3)]
        plan_beam: usize,
        /// Candidate evaluations per committed step for `--planned`.
        #[arg(long, default_value_t = 32)]
        plan_budget: usize,
    },
    /// Sweep solvers and strategies, reporting cost and agreement
    /// (roadmap 2.9 / 4.7 / 6.6 / 8.6 / 10.7 harness).
    /// Measure the parallel I/O engine and the block execution modes.
    ///
    /// The two halves of Phase 34 in one place, because the question they
    /// answer is the same one: does the fast path return the same bytes as the
    /// simple path, and how much faster is it? Both are *measured* here and
    /// checked, rather than asserted in prose.
    Io {
        /// A file to read. Omitted, a temporary fixture is written.
        #[arg(long)]
        path: Option<PathBuf>,
        /// Extra replicas of the same file, given in order. Every path is a
        /// mirror of the first; a diverging one is reported, not hidden.
        #[arg(long, action = clap::ArgAction::Append)]
        mirror: Vec<PathBuf>,
        /// Bytes to read.
        #[arg(long, default_value_t = 8 << 20)]
        bytes: usize,
        /// Repetitions per path, for a stable timing.
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Compare the block execution modes as well. Off by default because
        /// the I/O half is the interesting one on a machine with one device.
        #[arg(long)]
        block_modes: bool,
    },
    Bench {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 8)]
        num_inference_steps: usize,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        /// Repetitions per configuration, for stable timings.
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Untimed repetitions before the measured ones; recorded and flagged
        /// in the JSON record, left out of the summary (roadmap Phase 28).
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Append one experiment record per configuration (environment,
        /// config, seed, every raw trial, summary with a 95% t-interval) to
        /// this JSONL file. Never truncates it.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Train a grid of configurations, one run per seed, into experiment
    /// records (roadmap Phase 28; issue 1 sections 5 and 8). The protocol the
    /// GPU-blocked comparisons will run through.
    Sweep {
        /// `lr=1e-4,3e-4 num_blocks=2,3 consistency=0,0.1` -- axes separated
        /// by whitespace, values by commas.
        #[arg(long)]
        grid: String,
        /// Seeds to repeat every cell with.
        #[arg(long, default_value = "1,2,3")]
        seeds: String,
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value = "synthetic")]
        dataset: String,
        #[arg(long)]
        data_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        /// Append every cell's record here as soon as it finishes.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Average several checkpoints of one architecture into a new one
    /// (roadmap 29.4): a model soup as a starting point.
    Merge {
        #[command(flatten)]
        model: ModelArgs,
        /// Checkpoints to merge, repeatable.
        #[arg(long, action = clap::ArgAction::Append, required = true)]
        input: Vec<PathBuf>,
        /// Weights over the inputs; empty is uniform.
        #[arg(long, default_value = "")]
        weights: String,
        #[arg(long, default_value = "checkpoints")]
        out: PathBuf,
    },
    /// Audits that measure what the theory assumes (roadmap Phase 28).
    Audit {
        #[command(subcommand)]
        action: AuditAction,
    },
    /// Read and compare experiment records.
    Experiment {
        #[command(subcommand)]
        action: ExperimentAction,
    },
    /// Classify a batch through the inference API and print top-k results.
    Infer {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 4)]
        num_inference_steps: usize,
        #[arg(long, default_value_t = 3)]
        top_k: usize,
        #[arg(long, default_value = "euler")]
        solver: String,
    },
    /// Run the numerical certificate suite (the quality gate).
    ///
    /// Exits non-zero if any mathematical identity the implementation rests on
    /// fails to hold within its tolerance.
    Verify {
        /// Only run certificates in this group.
        #[arg(long)]
        group: Option<String>,
    },
    /// Scan a tree for attempts to defeat the quality gate rather than satisfy
    /// it: lint suppression, weakened or disabled tests, and edits to the
    /// gate script, certificate registry, or manifest.
    ///
    /// Exits non-zero on any finding. Findings already accepted in the
    /// repository baseline are not reported; only the delta is.
    Cheat {
        /// Scan a tree for attempts to defeat the quality gate rather than
        /// satisfy it: lint suppression, weakened or disabled tests, and edits
        /// to the gate script, certificate registry, or manifest.
        ///
        /// Subcommands, not flags, because this tree now has four shapes of
        /// thing to do and a flat flag list for all of them puts every one of
        /// their options on the root `Cli` — which clap materialises, whole, on
        /// the stack of whatever parses it.
        #[command(subcommand)]
        action: CheatAction,
    },
    /// Language-model paths: tokenize a corpus, generate text, and compare
    /// the decoding strategies (roadmap Phases 19 and 21b).
    Lm {
        #[command(subcommand)]
        action: LmAction,
    },
    /// Token-block diffusion substrate (plan items 1-2): a BD3-LM-style
    /// semi-autoregressive byte LM with block-causal attention, per-block KV
    /// caching, entropy-adaptive block sizing, an entropy-gated halo, and a
    /// bounded routine revision pass. This is the subject those items
    /// modify; it is a substrate, not a result.
    Tblock {
        #[command(subcommand)]
        action: TblockAction,
    },
    /// Geometric reasoning (roadmap Phase 33): a dual-stream reasoner over a
    /// learned Riemannian scene geometry with a MoSME expert readout, a
    /// deterministic exact kernel beside it, and the block-diffusion
    /// machinery for training. The kernel proves; the model proposes.
    Geom {
        #[command(subcommand)]
        action: GeomAction,
    },
    /// Inspect or scaffold an expert index — the manifest an inference engine
    /// routes from. Reads JSON only; never loads weights.
    Experts {
        #[command(subcommand)]
        action: ExpertsAction,
    },
    /// The specialist coding student: per-language trace training with the
    /// router-which-knows auxiliary, its routing readout, and its shape
    /// report.
    Student {
        #[command(subcommand)]
        action: StudentAction,
    },
    /// The NINA program (PLAN-NINA.md §1, §4, §8): the name-addressable model
    /// registry, the Merkle-chained append-only log, and the MCP training
    /// server.
    Nina {
        #[command(subcommand)]
        action: NinaAction,
    },
    /// Print the block sigma schedule for inspection.
    Sigmas {
        #[arg(long, default_value_t = 3)]
        num_blocks: usize,
        #[arg(long, default_value_t = 0.05)]
        gamma: f64,
    },
    /// Field corpora (PLAN-NINA §2): normalize the downloaded public datasets
    /// into the cybercorpus layout, one corpus per field, every document
    /// through the cheat stack before it lands.
    Corpus {
        #[command(subcommand)]
        action: CorpusAction,
    },
}

/// The field-corpus subcommands.
#[derive(Subcommand)]
enum CorpusAction {
    /// Build one field corpus (`math`, `science`, `bio`, `cyber`, `tech`)
    /// from the downloaded datasets into `<out-root>/<field>/`:
    /// `dataset/all.jsonl` (one {"id","source","path","format","text"} per
    /// line), per-source `dataset/<source>.jsonl`, `dataset/manifest.json`
    /// with per-source doc/char counts and cheat-drop counts per class, and
    /// `text/<source>/` plain text. Resume-safe: a source with a completion
    /// marker is skipped unless `--force` is given.
    BuildField {
        /// The field to build; see fieldnorm::available_fields.
        #[arg(long)]
        field: String,
        /// Root of the downloaded datasets.
        #[arg(long, default_value = fieldnorm::DEFAULT_DATASETS_ROOT)]
        datasets_root: PathBuf,
        /// Root the corpora are written under.
        #[arg(long, default_value = fieldnorm::DEFAULT_OUT_ROOT)]
        out_root: PathBuf,
        /// FieldnormConfig sidecar (JSON); the defaults when omitted.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Rebuild sources even where a completion marker exists.
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Render a field corpus (`dataset/all.jsonl`, one
    /// {"id","source","path","format","text"} per line) into the trainer's
    /// pair-file shape: {"prompt","completion","field"} per line, split at
    /// the first blank line or the first quarter of the text (see
    /// fieldnorm::split_prompt_completion). Capped at --max-docs by a
    /// deterministic sha256(id) subsample — NOT first-N, so a rerun over a
    /// grown corpus displaces old picks only where new ids hash lower.
    /// Streams (the corpus is never read whole); writes
    /// `<out>.manifest.json` with counts and the output's sha256.
    ToPairs {
        /// Field corpus all.jsonl.
        #[arg(long)]
        corpus: PathBuf,
        /// Field name written into every pair line.
        #[arg(long)]
        field: String,
        /// Output pairs JSONL.
        #[arg(long)]
        out: PathBuf,
        /// Cap on emitted pairs (deterministic sha256(id) subsample). The
        /// trainer reads the whole pairs file into memory; this cap is what
        /// bounds that load.
        #[arg(long, default_value_t = fieldnorm::DEFAULT_PAIRS_MAX_DOCS)]
        max_docs: usize,
    },
}

/// Where `dblocks student train` looks for traces when `--traces` is not
/// given: the external drive the teacher-trace pipeline writes to.
const DEFAULT_STUDENT_TRACES: &str = "/srv/m-sda/unifur/train.jsonl";
/// Where student checkpoints go by default (the same external drive).
const DEFAULT_STUDENT_OUT: &str = "/srv/m-sda/unifur/student";
/// The fallback corpus: unlabeled SFT lines, loaded with `language=unknown`
/// (the routing auxiliary is off for them — there is no box to name).
const FALLBACK_SFT: &str = "/srv/m-sdd/unifur/datasets/sft-12k.jsonl";

/// The flags of `dblocks student train`, bundled so the handler takes the
/// struct rather than a dozen loose arguments.
#[derive(clap::Args, Clone)]
struct StudentTrainArgs {
    /// rust|python|c|cpp|jsts|web, or `unknown` for unlabeled data (the
    /// routing auxiliary switches off: no box exists to supervise).
    #[arg(long)]
    language: String,
    /// Traces JSONL, one {id, language, prompt, completion, source, ...}
    /// per line. The older SFT schema (instruction/response) also reads.
    #[arg(long, default_value = DEFAULT_STUDENT_TRACES)]
    traces: PathBuf,
    #[arg(long, default_value_t = 100)]
    steps: usize,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    /// StudentConfig sidecar (JSON); the default config is local-sized.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Resume from this checkpoint, or from the newest one in `--out-dir`
    /// when passed without a value. The training state beside the model
    /// (optimizer, RNG, step) is restored and verified, so the
    /// continuation is exact.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    resume: Option<String>,
    #[arg(long, default_value = DEFAULT_STUDENT_OUT)]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Weight on the router-which-knows auxiliary; 0 disables it exactly.
    #[arg(long, default_value_t = 0.1)]
    router_aux_weight: f64,
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Also write a checkpoint (model + training state) every n steps.
    #[arg(long, default_value_t = 0)]
    checkpoint_every: usize,
    /// Rescale gradients whose global norm exceeds this; 0 is off.
    #[arg(long, default_value_t = 0.0)]
    clip_norm: f32,
}

/// The specialist-student subcommands.
#[derive(Subcommand)]
enum StudentAction {
    /// Train one language specialist on a traces JSONL: masked next-token CE
    /// on the completion tokens plus the router-which-knows auxiliary (the
    /// known language supervises the probe router's box traffic).
    Train {
        #[command(flatten)]
        args: StudentTrainArgs,
    },
    /// Print the probe router's per-language box traffic for a prompt — the
    /// router-which-knows readout: which specialist would be called, read
    /// from the gates alone, no expert run.
    Route {
        #[arg(long)]
        language_prompt: String,
        /// A trained student checkpoint; fresh weights route at chance.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// StudentConfig sidecar (must match the checkpoint's shape).
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Print the student's config, counted cost (active/total) and box
    /// registry.
    Info {
        /// StudentConfig sidecar; the default config when omitted.
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Read the student's raw-vector thinking: decode the hidden states at
    /// each refine depth through the trunk's own readout (the logit lens),
    /// printed as a thought trace with the certified energy and displacement
    /// of every refinement step.
    Think {
        #[arg(long)]
        prompt: String,
        /// Comma-separated refine depths, e.g. 1,2,4,8.
        #[arg(long, default_value = "1,2,4,8", value_delimiter = ',')]
        depths: Vec<usize>,
        /// A trained student checkpoint; fresh weights decode the untrained
        /// baseline.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// StudentConfig sidecar (must match the checkpoint's shape).
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
}

/// The flags of `dblocks nina chat`, bundled so the handler takes the struct
/// rather than a dozen loose arguments.
#[derive(clap::Args, Clone)]
struct NinaChatArgs {
    /// Registry name of the specialist to serve, `field/subfield`.
    #[arg(long)]
    model: String,
    /// The user turn.
    #[arg(long)]
    prompt: String,
    /// Serve context from this directory of blocks (`*.txt` files and/or
    /// `*.jsonl` docs, recursively; ids are relative paths) via the local
    /// retriever.
    #[arg(long, conflicts_with = "no_context")]
    context_root: Option<PathBuf>,
    /// Serve the bare prompt: byte-identical to augmentation with zero hits,
    /// which is the designed safe degradation.
    #[arg(long)]
    no_context: bool,
    /// Whitespace-token budget for injected context blocks. Shrunk further
    /// when the byte-level context window would overflow.
    #[arg(long, default_value_t = 768)]
    max_context_tokens: usize,
    /// Greedy-decode at most this many tokens.
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    /// Root holding registry.json, models/ and logs/; defaults to the
    /// external drive.
    #[arg(long, default_value = diffusionblocks::nina::DEFAULT_ROOT)]
    root: PathBuf,
}

/// The NINA program subcommands (PLAN-NINA.md §1, §4, §8).
#[derive(Subcommand)]
enum NinaAction {
    /// Run the MCP training server: JSON-RPC 2.0 over stdio (newline-delimited
    /// frames), exposing the registry, the training queue, routing and log
    /// verification as MCP tools.
    McpServe {
        /// Root holding registry.json and logs/; defaults to the external drive.
        #[arg(long, default_value = diffusionblocks::nina::DEFAULT_ROOT)]
        root: PathBuf,
    },
    /// Read the name-addressable model registry.
    Registry {
        #[command(subcommand)]
        action: NinaRegistryAction,
        /// Global so it parses after the leaf: `nina registry get x --registry r.json`.
        #[arg(long, global = true, default_value = diffusionblocks::nina::DEFAULT_REGISTRY)]
        registry: PathBuf,
    },
    /// Chat with a registered specialist: load its checkpoint, retrieve
    /// external context (`--context-root`, else the interop file channel with
    /// a short, loud timeout, else nothing — the bare prompt is the designed
    /// safe degradation), inject the hits into the serve-time prompt under a
    /// token budget, greedy-decode an answer, and append the whole turn to
    /// the Merkle-chained logs/chat.jsonl. Context is served, never trained
    /// into weights.
    Chat {
        #[command(flatten)]
        args: NinaChatArgs,
    },
    /// Re-derive a Merkle-chained log and report. A broken link is an error
    /// naming the first bad line, and the exit code is non-zero.
    VerifyLog {
        /// The chained JSONL log to verify.
        path: PathBuf,
    },
    /// Serve the web dashboard (PLAN-NINA.md §8): a localhost HTTP server
    /// showing the registry, the live log tails, the vecread thought trace
    /// and the chat console — every page re-verifies the chained log it
    /// reads, so an edited history renders red.
    Serve {
        #[arg(long, default_value_t = 7878)]
        port: u16,
        /// Root holding registry.json and logs/; defaults to the external drive.
        #[arg(long, default_value = diffusionblocks::nina::DEFAULT_ROOT)]
        root: PathBuf,
    },
}

/// Registry reads: list every entry, or one entry by name.
#[derive(Subcommand)]
enum NinaRegistryAction {
    /// Print every entry, or one field's with `--field`.
    List {
        #[arg(long)]
        field: Option<String>,
    },
    /// Print one entry by its `field/subfield` name.
    Get { name: String },
}

/// The geometric-reasoning subcommands (roadmap Phase 33). `solve` and
/// `counterexample` never touch the neural model: the kernel is the source of
/// mathematical truth, and these are its two gates.
#[derive(Subcommand)]
enum GeomAction {
    /// Generate the synthetic geometric-question corpus and its metadata
    /// sidecar. Any pre-tokenized corpus in this crate's u16 format can be
    /// trained on instead -- the reasoner does not care where its scenes came
    /// from.
    Data {
        #[arg(long, default_value_t = 6)]
        points: usize,
        /// Comma-separated question kinds: nearest | farthest | direction |
        /// inside | collinear.
        #[arg(long, default_value = "nearest,farthest,direction,inside,collinear")]
        kinds: String,
        /// Per-kind sampling weights, e.g. `1,1,2,1,1` to double `direction`
        /// (AlphaGeometry's stratification lesson: hard kinds need
        /// oversampling). Empty keeps the legacy round-robin bit for bit.
        #[arg(long, default_value = "")]
        kind_weights: String,
        #[arg(long, default_value_t = 4096)]
        count: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Similarity-augmented copies per scene: rotation, translation and uniform
        /// scale, with the label recomputed on the grid and the copy kept only
        /// when it still matches (direction scenes get translation and scale
        /// only, their answer being an absolute compass bearing). 0, the
        /// default, leaves the corpus exactly as generated.
        #[arg(long, default_value_t = 0)]
        augment: usize,
        #[arg(long, default_value = "geom")]
        out_dir: PathBuf,
    },
    /// Train the geometric reasoner (objective answer | diffusion). The
    /// answer objective trains the clean reasoning path; the diffusion
    /// objective is the framework's per-block sigma-window scheme with
    /// boundary consistency, and additionally enables `generate --diffusion`.
    Train {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        meta: PathBuf,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        /// constant | warmup | cosine. `cosine` is the default: the
        /// relaxation's step size comes from the learned metric, so early
        /// large updates land where the metric is least settled.
        #[arg(long, default_value = "cosine")]
        lr_schedule: String,
        /// Global gradient-norm ceiling; 0 disables clipping.
        #[arg(long, default_value_t = 1.0)]
        clip_norm: f64,
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// answer | diffusion.
        #[arg(long, default_value = "answer")]
        objective: String,
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 8)]
        dim: usize,
        #[arg(long, default_value_t = 64)]
        hidden_size: usize,
        #[arg(long, default_value_t = 2)]
        num_blocks: usize,
        #[arg(long, default_value_t = 2)]
        refine_steps: usize,
        /// Boxes of expert heads in the MoSME readout; 0 uses the plain
        /// linear readout.
        #[arg(long, default_value_t = 5)]
        moe_boxes: usize,
        #[arg(long, default_value_t = 2)]
        moe_experts: usize,
        #[arg(long, default_value_t = 1)]
        moe_top_k: usize,
        /// EDM data scale for the preconditioning (Karras et al.): match it
        /// to the latent scale (0.5 is the image default). Must agree across
        /// train/eval/generate: geom checkpoints carry no training state.
        #[arg(long, default_value_t = 0.5)]
        sigma_data: f64,
        #[arg(long, default_value_t = 0.01)]
        moe_balance_weight: f64,
        #[arg(long, default_value_t = 0.05)]
        gamma: f64,
        #[arg(long, default_value_t = 0.1)]
        consistency_weight: f64,
        /// Micro-batches per optimizer step, averaged not summed. 1 steps on
        /// every micro-batch, exactly as before.
        #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
        accumulate: usize,
        /// EMA decay for the evaluation shadow; empty disables EMA exactly.
        #[arg(long)]
        ema_decay: Option<f64>,
        #[arg(long, default_value_t = 50)]
        log_every: usize,
        #[arg(long)]
        log: Option<PathBuf>,
        #[arg(long, default_value = "checkpoints")]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        checkpoint_every: usize,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
    },
    /// Train the matched-depth transformer baseline: a plain pre-norm
    /// encoder over the same scenes, with `num_blocks * refine_steps` layers
    /// and the same hidden size, corpus, batching, optimizer and held-out
    /// protocol as the reasoner -- the control arm for the geometric claim
    /// in docs/Claims.md. Prints both models' parameter counts; they must
    /// sit in a 25% band for the comparison to mean anything.
    Baseline {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        meta: PathBuf,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        /// constant | warmup | cosine. Must match the reasoner run being
        /// compared against.
        #[arg(long, default_value = "cosine")]
        lr_schedule: String,
        /// Global gradient-norm ceiling; 0 disables clipping.
        #[arg(long, default_value_t = 1.0)]
        clip_norm: f64,
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 8)]
        dim: usize,
        #[arg(long, default_value_t = 64)]
        hidden_size: usize,
        #[arg(long, default_value_t = 2)]
        num_blocks: usize,
        #[arg(long, default_value_t = 2)]
        refine_steps: usize,
        /// The reasoner's readout flags: the baseline has no router, but the
        /// reasoner's parameter count (printed for the band check) depends
        /// on them, so the match must see them.
        #[arg(long, default_value_t = 5)]
        moe_boxes: usize,
        #[arg(long, default_value_t = 2)]
        moe_experts: usize,
        #[arg(long, default_value_t = 1)]
        moe_top_k: usize,
        /// Unused by the baseline itself; part of the shared architecture
        /// flags so invocations can mirror a reasoner run verbatim.
        #[arg(long, default_value_t = 0.5)]
        sigma_data: f64,
        /// Micro-batches per optimizer step, averaged not summed. 1 steps on
        /// every micro-batch, exactly as before.
        #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
        accumulate: usize,
        /// EMA decay for the evaluation shadow; empty disables EMA exactly.
        #[arg(long)]
        ema_decay: Option<f64>,
        #[arg(long, default_value_t = 50)]
        log_every: usize,
        #[arg(long)]
        log: Option<PathBuf>,
        #[arg(long, default_value = "checkpoints")]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        checkpoint_every: usize,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
    },
    /// Evaluate answer accuracy over held-out scenes. `--refine-sweep` sweeps
    /// the relaxation depth at inference: accuracy against the number of
    /// geometric refinement steps -- the test-time-compute curve for
    /// reasoning, measured on the same weights.
    Eval {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        meta: PathBuf,
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 8)]
        dim: usize,
        #[arg(long, default_value_t = 64)]
        hidden_size: usize,
        #[arg(long, default_value_t = 2)]
        num_blocks: usize,
        #[arg(long, default_value_t = 2)]
        refine_steps: usize,
        #[arg(long, default_value_t = 5)]
        moe_boxes: usize,
        #[arg(long, default_value_t = 2)]
        moe_experts: usize,
        #[arg(long, default_value_t = 1)]
        moe_top_k: usize,
        /// EDM data scale; must match the training flags (see `train`).
        #[arg(long, default_value_t = 0.5)]
        sigma_data: f64,
        #[arg(long, default_value_t = 16)]
        batches: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = false)]
        refine_sweep: bool,
        /// A trained baseline checkpoint (`geom baseline`): after evaluating
        /// the reasoner, load it and evaluate the baseline on the same
        /// held-out scenes, then print the side-by-side comparison (accuracy
        /// and both parameter counts).
        #[arg(long)]
        baseline: Option<PathBuf>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
    },
    /// Answer one scene: the answer token, the attractor diagnostics, and --
    /// with --diffusion -- a trajectory denoised from pure noise into the
    /// scene's basin (needs a diffusion-trained checkpoint).
    Generate {
        /// The scene, without its answer: scene_len - 1 bytes, e.g.
        /// "P06A0507B1203C1502D0811E0314F0910nA___".
        #[arg(long)]
        scene: String,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 8)]
        dim: usize,
        #[arg(long, default_value_t = 64)]
        hidden_size: usize,
        #[arg(long, default_value_t = 2)]
        num_blocks: usize,
        #[arg(long, default_value_t = 2)]
        refine_steps: usize,
        #[arg(long, default_value_t = 5)]
        moe_boxes: usize,
        #[arg(long, default_value_t = 2)]
        moe_experts: usize,
        #[arg(long, default_value_t = 1)]
        moe_top_k: usize,
        /// EDM data scale; must match the training flags (see `train`).
        #[arg(long, default_value_t = 0.5)]
        sigma_data: f64,
        #[arg(long, default_value_t = false)]
        diffusion: bool,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
    },
    /// The symbolic engine: saturate a scene graph's facts through the
    /// kernel's deduction rules to a fixed point and print every derived fact
    /// with its rule certificate. No model is loaded.
    Solve {
        /// A geometry scene-graph JSON (the kernel's canonical IR).
        #[arg(long)]
        graph: PathBuf,
        /// Also write the saturated graph here.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Randomized falsification: sample configurations satisfying the
    /// premises exactly and hunt for one violating the claim. A
    /// counterexample rejects; surviving draws are evidence, never proof.
    Counterexample {
        #[arg(long)]
        claim: PathBuf,
        #[arg(long, default_value_t = 256)]
        trials: usize,
        #[arg(long, default_value_t = 6)]
        grid: i64,
        #[arg(long, default_value_t = 4)]
        seed: u64,
    },
    /// The trunk-fused geometric stream: Qwen trunk + GeomFusion vector stream
    /// + MoSME readout. Adapters-only training; every emitted answer is
    ///
    /// kernel-checked by `prove`.
    Fused {
        #[command(subcommand)]
        action: Box<FusedAction>,
    },
}

#[derive(Subcommand)]
enum FusedAction {
    /// Train the fused model (adapters-only) on a geometric corpus.
    Train {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        meta: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(
            long,
            default_value = "/srv/m-sdd/unifur/tokenizers/qwen38-source/tokenizer.json"
        )]
        tokenizer: PathBuf,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,
        #[arg(long, default_value_t = 1e-3)]
        fusion_lr: f64,
        #[arg(long, default_value_t = 1, value_parser = parse_positive_usize)]
        accumulate: usize,
        #[arg(long)]
        ema_decay: Option<f64>,
        #[arg(long, default_value_t = 10)]
        log_every: usize,
        #[arg(long, default_value = "checkpoints")]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        holdout: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// answer | diffusion.
        #[arg(long, default_value = "answer")]
        objective: String,
        #[arg(long, default_value_t = 0.5)]
        sigma_data: f64,
        #[arg(long, default_value_t = 0.01)]
        moe_balance_weight: f64,
        #[arg(long, default_value_t = 0.05)]
        gamma: f64,
        #[arg(long)]
        depth: Option<usize>,
        #[arg(long, default_value_t = 4)]
        lora_rank: usize,
        #[arg(long, default_value_t = 4.0)]
        lora_alpha: f64,
        #[arg(long, default_value_t = 5120)]
        hidden_size: usize,
        #[arg(long, default_value_t = 128)]
        geom_hidden: usize,
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 32)]
        dim: usize,
        #[arg(long, default_value_t = 2)]
        num_blocks: usize,
        #[arg(long, default_value_t = 4)]
        refine_steps: usize,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
    },
    /// Evaluate a fused checkpoint on held-out scenes.
    Eval {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        meta: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(
            long,
            default_value = "/srv/m-sdd/unifur/tokenizers/qwen38-source/tokenizer.json"
        )]
        tokenizer: PathBuf,
        #[arg(long, default_value_t = 16)]
        batches: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 5120)]
        hidden_size: usize,
        #[arg(long, default_value_t = 128)]
        geom_hidden: usize,
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 32)]
        dim: usize,
        #[arg(long, default_value_t = 2)]
        num_blocks: usize,
        #[arg(long, default_value_t = 4)]
        refine_steps: usize,
        #[arg(long, default_value_t = 4)]
        lora_rank: usize,
        #[arg(long, default_value_t = 4.0)]
        lora_alpha: f64,
        #[arg(long, default_value_t = 0.5)]
        sigma_data: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
    },
    /// Kernel-check a fused checkpoint on held-out scenes.
    Prove {
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        meta: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(
            long,
            default_value = "/srv/m-sdd/unifur/tokenizers/qwen38-source/tokenizer.json"
        )]
        tokenizer: PathBuf,
        #[arg(long, default_value_t = 256)]
        scenes: usize,
        #[arg(long)]
        depth: Option<usize>,
        #[arg(long, default_value_t = true)]
        sweep: bool,
        #[arg(long, default_value_t = 5120)]
        hidden_size: usize,
        #[arg(long, default_value_t = 128)]
        geom_hidden: usize,
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 32)]
        dim: usize,
        #[arg(long, default_value_t = 2)]
        num_blocks: usize,
        #[arg(long, default_value_t = 4)]
        refine_steps: usize,
        #[arg(long, default_value_t = 4)]
        lora_rank: usize,
        #[arg(long, default_value_t = 4.0)]
        lora_alpha: f64,
        #[arg(long, default_value_t = 0.5)]
        sigma_data: f64,
        #[command(flatten)]
        runtime: LmRuntimeArgs,
    },
}

#[derive(Subcommand)]
enum PolicyAction {
    /// Write the starter policy (three cyber scopes) and a fresh signing key.
    Init {
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        key: PathBuf,
    },
    /// Blockers, scopes, patterns and revocations.
    List {
        #[arg(long)]
        policy: PathBuf,
    },
    /// Add a blocker; every pattern must parse and consume something.
    AddBlocker {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        id: String,
        #[arg(long)]
        scope: String,
        /// Pattern in the `antipattern` syntax; repeatable.
        #[arg(long, action = clap::ArgAction::Append, required = true)]
        pattern: Vec<String>,
        /// prompt | output | both
        #[arg(long, default_value = "both")]
        applies_to: String,
        #[arg(long)]
        refusal: String,
        #[arg(long, default_value = "")]
        description: String,
    },
    /// Remove a blocker by id.
    RemoveBlocker {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        id: String,
    },
    /// Show which blockers fire on a prompt and what the gate would decide,
    /// given any grants presented.
    Check {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        key: Option<PathBuf>,
        #[arg(long, action = clap::ArgAction::Append)]
        grant: Vec<PathBuf>,
    },
    /// Refuse a grant from now on.
    Revoke {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        grant_id: String,
    },
}

#[derive(Subcommand)]
enum ApprovalsAction {
    /// Sign a grant with the policy's key.
    Issue {
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        id: String,
        /// Comma-separated scopes, e.g. `cyber:exploit-development,cyber:malware`.
        #[arg(long)]
        scopes: String,
        /// Expiry as a Unix timestamp.
        #[arg(long)]
        expires: u64,
        #[arg(long, default_value = "")]
        note: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Check a grant's signature, key, expiry and revocation, in that order.
    Verify {
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        grant: PathBuf,
    },
}

#[derive(Subcommand)]
enum AuditAction {
    /// Per block: local loss, boundary mismatch with the next block, a
    /// finite-difference sensitivity proxy, and the downstream amplification
    /// of an error made there (issue 1 section 7).
    Propagation {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        /// Standard deviation of the latent perturbation.
        #[arg(long, default_value_t = 1e-2)]
        epsilon: f64,
        /// Write the report as JSON here as well.
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ExperimentAction {
    /// Print every record in a log: name, unit, trials, summary.
    Show {
        #[arg(long)]
        path: PathBuf,
    },
    /// Match records by name across two logs and compare their summaries.
    Compare {
        #[arg(long)]
        a: PathBuf,
        #[arg(long)]
        b: PathBuf,
    },
}

#[derive(Subcommand)]
enum ExpertsAction {
    /// Write a starter spec, e.g.
    /// `--box coding:rust,python,secure --box cyber:netsec,malware`.
    Init {
        #[arg(long)]
        out: PathBuf,
        /// `<box>:<expert>,<expert>,...`, repeatable.
        #[arg(long = "box", required = true)]
        boxes: Vec<String>,
        #[arg(long, default_value_t = 1)]
        top_box: usize,
        #[arg(long, default_value_t = 1)]
        top_expert: usize,
    },
    /// Print an index or spec as a table.
    List {
        #[arg(long)]
        index: PathBuf,
    },
    /// Check an index's structural invariants.
    Validate {
        #[arg(long)]
        index: PathBuf,
    },
}

/// The message a panicking thread carried, for reporting a joined thread's
/// failure as an error instead of re-panicking.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Sigmas { num_blocks, gamma } => cmd_sigmas(num_blocks, gamma),
        Command::Lm { action } => cmd_lm(action),
        Command::Tblock { action } => cmd_tblock(action),
        Command::Geom { action } => cmd_geom(action),
        Command::Experts { action } => cmd_experts(action),
        Command::Student { action } => cmd_student(action),
        Command::Nina { action } => cmd_nina(action),
        Command::Corpus { action } => cmd_corpus(action),
        Command::Verify { group } => cmd_verify(group.as_deref()),
        Command::Cheat { action } => match action {
            CheatAction::Scan { root, json, all } => cmd_cheat_scan(&root, json, all),
            CheatAction::Adapt {
                log,
                max_fire_rate,
                out,
            } => cmd_cheat_adapt(&log, max_fire_rate, out.as_deref()),
            CheatAction::Policy { profile } => cmd_cheat_policy(profile.as_deref()),
            CheatAction::Ml { model } => cmd_cheat_ml(model.as_deref()),
            CheatAction::Harness { action } => match action {
                CheatHarnessAction::Round {
                    corpus,
                    registry,
                    candidates,
                    json,
                } => cmd_cheat_harness_round(&corpus, &registry, candidates, json),
                CheatHarnessAction::Commit {
                    corpus,
                    registry,
                    candidates,
                    provenance,
                } => cmd_cheat_harness_commit(&corpus, &registry, candidates, &provenance),
                CheatHarnessAction::Rollback { registry } => cmd_cheat_harness_rollback(&registry),
                CheatHarnessAction::Status { registry } => cmd_cheat_harness_status(&registry),
            },
        },
        command @ Command::Train { .. } => cmd_train(command),
        command @ Command::Sample { .. } => cmd_sample(command),
        Command::Bench {
            model,
            num_inference_steps,
            batch_size,
            repeats,
            warmup,
            json,
        } => cmd_bench(
            model,
            num_inference_steps,
            batch_size,
            repeats,
            warmup,
            json,
        ),
        Command::Io {
            path,
            mirror,
            bytes,
            repeats,
            block_modes,
        } => cmd_io(path.as_deref(), &mirror, bytes, repeats, block_modes),
        Command::Sweep {
            grid,
            seeds,
            steps,
            dataset,
            data_dir,
            batch_size,
            json,
        } => cmd_sweep(&grid, &seeds, steps, &dataset, data_dir, batch_size, json),
        Command::Audit { action } => cmd_audit(action),
        Command::Merge {
            model,
            input,
            weights,
            out,
        } => cmd_merge(model, input, &weights, out),
        Command::Experiment { action } => cmd_experiment(action),
        Command::Infer {
            model,
            batch_size,
            num_inference_steps,
            top_k,
            solver,
        } => cmd_infer(model, batch_size, num_inference_steps, top_k, &solver),
    }
}

/// Measure the I/O engine, and optionally the block execution modes.
///
/// The output is a table, and the important column is not the speed one: every
/// configuration is read and then compared *byte for byte* against a
/// single-drive read of the same region. A faster read of the wrong bytes is
/// the failure mode this command exists to make visible, so a mismatch is
/// reported as an error rather than as a timing.
fn cmd_io(
    path: Option<&Path>,
    mirrors: &[PathBuf],
    bytes: usize,
    repeats: usize,
    block_modes: bool,
) -> Result<()> {
    use diffusionblocks::{
        blockexec::BlockExecMode,
        peregrine::{Backend, MirrorSet, Region},
    };
    use std::io::Write;
    use std::time::Instant;

    let repeats = repeats.max(1);
    let mut scratch: Option<PathBuf> = None;
    // Without a path, write a fixture whose every byte depends on its offset,
    // so a mis-assembled read is detected rather than merely changing a length.
    let primary: PathBuf = match path {
        Some(p) => p.to_path_buf(),
        None => {
            let dir = std::env::temp_dir().join(format!("dblocks-io-{}", std::process::id()));
            std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
            let fixture = dir.join("fixture.bin");
            let mut file = std::fs::File::create(&fixture)
                .with_context(|| format!("create {}", fixture.display()))?;
            let chunk: Vec<u8> = (0..(1 << 20)).map(|i| (i % 251) as u8).collect();
            let mut written = 0usize;
            while written < bytes {
                let n = chunk.len().min(bytes - written);
                file.write_all(&chunk[..n]).context("write fixture")?;
                written += n;
            }
            file.sync_all().context("sync fixture")?;
            scratch = Some(dir);
            fixture
        }
    };

    // The replica set is the primary plus any `--mirror` paths, deduplicated so
    // the same file twice is not reported as two independent devices.
    let mut paths = vec![primary.clone()];
    for m in mirrors {
        if !paths.contains(m) {
            paths.push(m.clone());
        }
    }
    let on_disk = std::fs::metadata(&primary)
        .with_context(|| format!("stat {}", primary.display()))?
        .len() as usize;
    let region = Region::new(0, bytes.min(on_disk));

    println!(
        "region: {} bytes | replicas: {} | backend: {:?}",
        region.len,
        paths.len(),
        Backend::detect(cfg!(feature = "peregrine-uring"))
    );
    for p in &paths {
        println!("  replica {}", p.display());
    }

    // The reference: one drive, one read.
    let mut serial = MirrorSet::open(&paths, Backend::Threaded)
        .with_context(|| format!("open {}", primary.display()))?;
    let mut reference: Option<Vec<u8>> = None;
    for _ in 0..repeats {
        let start = Instant::now();
        let read = serial.read_serial(region.clone()).context("serial read")?;
        let elapsed = start.elapsed();
        match &reference {
            None => reference = Some(read),
            Some(first) => ensure!(
                first == &read,
                "two serial reads of the same region disagreed: the file changed under the reader"
            ),
        }
        let stats = serial.stats();
        println!(
            "  {:<24} {:>8.2} MiB/s  reads {:>3}  syscalls {:>3}  reads/syscall {:.2}",
            "serial (1 drive)",
            mib(region.len, elapsed),
            stats.reads,
            stats.syscalls,
            stats.reads_per_syscall()
        );
    }
    let reference = reference.context("no repeat was run")?;

    // The striped paths, each checked against the reference.
    for backend in [
        Backend::Threaded,
        Backend::detect(cfg!(feature = "peregrine-uring")),
    ] {
        if backend == Backend::IoUring && !cfg!(feature = "peregrine-uring") {
            continue;
        }
        let mut set = MirrorSet::open(&paths, backend)
            .with_context(|| format!("open {} on {backend:?}", primary.display()))?;
        for _ in 0..repeats {
            let start = Instant::now();
            let got = set
                .read_striped(region.clone())
                .with_context(|| format!("striped read on {backend:?}"))?;
            let elapsed = start.elapsed();
            ensure!(
                got == reference,
                "{backend:?}: the striped read returned {} bytes that differ from the \
                 single-drive read of the same region. This is exactly the corruption \
                 the engine exists to prevent, and it is not a performance result.",
                got.len()
            );
            let stats = set.stats();
            println!(
                "  {:<24} {:>8.2} MiB/s  reads {:>3}  syscalls {:>3}  reads/syscall {:.2}  saved {}",
                format!("striped {backend:?}"),
                mib(got.len(), elapsed),
                stats.reads,
                stats.syscalls,
                stats.reads_per_syscall(),
                stats.syscalls_avoided()
            );
        }
    }
    println!("\nall reads returned identical {} bytes", reference.len());

    if block_modes {
        println!("\nblock execution modes (32 items, uneven work):");
        for mode in [
            BlockExecMode::Sync,
            BlockExecMode::MultiThread,
            BlockExecMode::Parallel,
        ] {
            let (out, timing) = mode
                .map_timed(32, |i| {
                    // Deliberately uneven, so a mode that returned results in
                    // completion order would produce a different vector.
                    std::thread::sleep(std::time::Duration::from_millis(4 * (i % 5) as u64));
                    Ok(i * i)
                })
                .with_context(|| format!("{} batch", mode.as_str()))?;
            ensure!(
                out == (0..32).map(|i| i * i).collect::<Vec<_>>(),
                "{} returned a different result from the serial expectation",
                mode.as_str()
            );
            println!(
                "  {:<5} {:>8.2} items/s  threads {:>3}  per-thread {:>8.2} items/s",
                mode.as_str(),
                timing.throughput(),
                timing.threads,
                timing.per_thread_throughput()
            );
        }
    }

    // Cleanup failure is reported rather than swallowed: a scratch directory
    // that survives the run is the one way this command can leave a machine
    // worse off than it found it, and the user is the one who can delete it.
    if let Some(dir) = scratch {
        if let Err(err) = std::fs::remove_dir_all(&dir) {
            eprintln!(
                "could not remove the scratch directory {}: {err}",
                dir.display()
            );
        }
    }
    Ok(())
}

/// Bytes moved per second, in MiB/s.
fn mib(bytes: usize, elapsed: std::time::Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64 / (1u64 << 20) as f64) / secs
}

/// The rule set a `--rules` flag names, or the built-in one.
fn load_labeler(rules: Option<&Path>) -> Result<Labeler> {
    match rules {
        Some(path) => Labeler::new(RuleSet::read(path)?),
        None => Labeler::builtin(),
    }
}

fn lm_specialist_state(path: &Path) -> Result<Option<checkpoint::TrainState>> {
    let state = checkpoint::TrainState::for_model(path)?;
    if let Some(state) = &state {
        anyhow::ensure!(
            state.kind == "lm",
            "{} is not an LM checkpoint",
            path.display()
        );
        anyhow::ensure!(
            checkpoint::file_sha256_hex(path)? == state.model.sha256,
            "model checksum mismatch for {}",
            path.display()
        );
    }
    Ok(state)
}

fn lm_specialist_config(
    state: Option<&checkpoint::TrainState>,
    fallback: impl FnOnce() -> Result<LmConfig>,
) -> Result<LmConfig> {
    let config = match state
        .and_then(|state| state.config.get("model_config"))
        .filter(|value| !value.is_null())
    {
        Some(value) => {
            serde_json::from_value(value.clone()).context("parse checkpoint model_config")?
        }
        None => fallback()?,
    };
    validate_lm_config(&config)?;
    anyhow::ensure!(
        config.frequency_embedding_size > 0 && config.frequency_embedding_size % 2 == 0,
        "frequency embedding size must be positive and even"
    );
    anyhow::ensure!(
        config
            .attention
            .as_ref()
            .is_none_or(|a| a.num_layers() == config.num_layers),
        "attention schedule length mismatch"
    );
    anyhow::ensure!(
        config.mosme.is_some() && config.moe.is_none() && config.routing_state == 0,
        "compact specialists require MoSME without flat MoE or routing state"
    );
    Ok(config)
}

fn lm_specialist_shared_hash(
    model: &LanguageModel<Eval>,
    config: &LmConfig,
    expert_id: &str,
) -> Result<String> {
    use burn::module::{Module, ModuleMapper, Param, ParamId};
    use burn::tensor::Tensor;
    use sha2::{Digest, Sha256};
    diffusionblocks::tensor_ext::force_initialization(model);
    let selected = model.specialist_trainable(config, expert_id)?;
    struct SharedHash<'a> {
        selected: &'a [ParamId],
        path: Vec<String>,
        sha: Sha256,
        count: usize,
    }
    impl ModuleMapper<Eval> for SharedHash<'_> {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path.pop();
        }
        fn map_float<const D: usize>(
            &mut self,
            param: Param<Tensor<Eval, D>>,
        ) -> Param<Tensor<Eval, D>> {
            if !self.selected.contains(&param.id) && !self.path.iter().any(|part| part == "router")
            {
                let data = param.val().to_data();
                self.sha.update(
                    format!("{:?}\0{:?}\0{:?}\0", self.path, data.shape, data.dtype).as_bytes(),
                );
                self.sha
                    .update((data.as_bytes().len() as u64).to_le_bytes());
                self.sha.update(data.as_bytes());
                self.count += 1;
            }
            param
        }
    }
    let mut hash = SharedHash {
        selected: selected.ids().context("missing specialist parameter IDs")?,
        path: Vec::new(),
        sha: Sha256::new(),
        count: 0,
    };
    // `map` is a pure read of the parameters: it walks them to accumulate a hash
    // and mutates the mapper, never the model. The clone exists because `map`
    // takes ownership. `drop` states that the returned model is deliberately
    // discarded rather than leaving a bare `let _ =` to wonder about.
    drop(model.clone().map(&mut hash));
    anyhow::ensure!(
        hash.count > 0,
        "no shared tensors found for lineage validation"
    );
    Ok(format!("{:x}", hash.sha.finalize()))
}

fn cmd_lm_specialist(
    base: &Path,
    trained: Option<&Path>,
    expert_id: &str,
    out: &Path,
    arch: &LmArchArgs,
) -> Result<()> {
    anyhow::ensure!(
        !out.try_exists()? && std::fs::symlink_metadata(out).is_err(),
        "--out must be a new directory; refusing to overwrite {}",
        out.display()
    );
    let base = std::fs::canonicalize(base).context("resolve base checkpoint")?;
    let base_state = lm_specialist_state(&base)?;
    let config = lm_specialist_config(base_state.as_ref(), || arch.config())?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let model = checkpoint::load::<Eval, _>(
        LanguageModel::<Eval>::new(&config, &device)?,
        &base,
        &device,
    )?;
    let (compact, compact_config) = model.compact_specialist(&config, expert_id)?;
    let parent = checkpoint::model_entry(&base)?;
    let parent_hash = checkpoint::canonical_hash_hex::<Eval, _>(&model);
    let shared_hash = lm_specialist_shared_hash(&compact, &compact_config, expert_id)?;
    let scope = train::LmTrainingScope::Specialist {
        expert_id: expert_id.to_owned(),
    };
    let (output, output_config, output_scope, operation, mut extras) = match trained {
        None => (
            compact,
            compact_config,
            scope,
            "export-specialist",
            serde_json::json!({
                "specialist_lineage": {
                    "expert_id": expert_id,
                    "parent": parent,
                    "parent_canonical_sha256": parent_hash,
                    "parent_model_config": config,
                    "shared_sha256": shared_hash,
                },
            }),
        ),
        Some(trained) => {
            let trained = std::fs::canonicalize(trained).context("resolve trained checkpoint")?;
            let state = lm_specialist_state(&trained)?;
            let trained_config =
                lm_specialist_config(state.as_ref(), || Ok(compact_config.clone()))?;
            if let Some(state) = &state {
                if let Some(saved) = state.config.get("training_scope") {
                    let saved: train::LmTrainingScope = serde_json::from_value(saved.clone())
                        .context("parse specialist training scope")?;
                    anyhow::ensure!(
                        saved == scope,
                        "trained checkpoint has incompatible specialist training scope"
                    );
                }
                if let Some(lineage) = state.extras.get("specialist_lineage") {
                    anyhow::ensure!(
                        lineage["expert_id"] == expert_id
                            && lineage["parent"]["sha256"] == parent.sha256
                            && lineage["parent_canonical_sha256"] == parent_hash
                            && lineage["parent_model_config"] == serde_json::to_value(&config)?
                            && lineage["shared_sha256"] == shared_hash,
                        "compact specialist parent lineage mismatch"
                    );
                }
            }
            anyhow::ensure!(
                serde_json::to_value(&trained_config)? == serde_json::to_value(&compact_config)?,
                "incompatible compact specialist configuration or identity"
            );
            let specialist = checkpoint::load::<Eval, _>(
                LanguageModel::<Eval>::new(&trained_config, &device)?,
                &trained,
                &device,
            )?;
            anyhow::ensure!(
                lm_specialist_shared_hash(&specialist, &trained_config, expert_id)? == shared_hash,
                "compact specialist shared tensor mismatch: wrong base or shared weights were trained"
            );
            let applied =
                model.apply_specialist(&config, &specialist, &trained_config, expert_id)?;
            (
                applied,
                config.clone(),
                train::LmTrainingScope::Joint,
                "apply-specialist",
                serde_json::json!({
                    "parent": parent,
                    "parent_canonical_sha256": parent_hash,
                    "trained": checkpoint::model_entry(&trained)?,
                    "trained_canonical_sha256": checkpoint::canonical_hash_hex::<Eval, _>(&specialist),
                    "specialist": expert_id,
                    "shared_sha256": shared_hash,
                    "lineage_validation": if state.as_ref().is_some_and(|s| s.extras.get("specialist_lineage").is_some()) {
                        "parent_checksums_and_shared_tensors"
                    } else { "shared_tensors_only" },
                    "applied_scope": scope,
                }),
            )
        }
    };
    extras["operation"] = operation.into();
    extras["weights_only"] = true.into();
    extras["optimizer_resume_available"] = false.into();
    extras["canonical_sha256"] = checkpoint::canonical_hash_hex::<Eval, _>(&output).into();
    std::fs::create_dir(out)
        .with_context(|| format!("create new output directory {}", out.display()))?;
    let out = std::fs::canonicalize(out)?;
    let path = checkpoint::save_content_addressed::<Eval, _>(output, &out, operation)?;
    write_derived_state(&path, "lm", extras)?;
    let mut state = checkpoint::TrainState::for_model(&path)?.context("missing derived state")?;
    state.config =
        serde_json::json!({ "model_config": output_config, "training_scope": output_scope });
    state.write(&checkpoint::TrainState::dir_for(&path))?;
    println!(
        "{operation} {expert_id} -> {} (CPU; weights only, use --base-weights, not --resume)",
        path.display()
    );
    Ok(())
}

fn geom_config_for(scene_len: usize, answers: &[u16], args: &GeomArch) -> GeomConfig {
    GeomConfig {
        scene_len,
        slots: args.slots,
        dim: args.dim,
        hidden_size: args.hidden_size,
        num_blocks: args.num_blocks,
        refine_steps: args.refine_steps,
        moe_boxes: args.moe_boxes,
        moe_experts: args.moe_experts,
        moe_top_k: args.moe_top_k,
        // The corpus's own answer set, so the loss and the accuracy are both
        // read over the closed space the questions actually live in.
        answer_tokens: answers.to_vec(),
        sigma_data: args.sigma_data,
        ..Default::default()
    }
}

/// The architecture flags shared by the geom subcommands that rebuild a
/// checkpoint. `scene_len` always comes from the metadata sidecar (or the
/// scene itself), never from a flag, so a model cannot be reloaded at the
/// wrong width.
struct GeomArch {
    slots: usize,
    dim: usize,
    hidden_size: usize,
    num_blocks: usize,
    refine_steps: usize,
    moe_boxes: usize,
    moe_experts: usize,
    moe_top_k: usize,
    sigma_data: f64,
}

fn load_geom_meta(path: &Path) -> Result<GeomMeta> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn cmd_geom(action: GeomAction) -> Result<()> {
    let runtime = match &action {
        GeomAction::Data { .. } | GeomAction::Solve { .. } | GeomAction::Counterexample { .. } => {
            None
        }
        GeomAction::Train { runtime, .. }
        | GeomAction::Baseline { runtime, .. }
        | GeomAction::Eval { runtime, .. }
        | GeomAction::Generate { runtime, .. } => Some(runtime),
        GeomAction::Fused { action } => match action.as_ref() {
            FusedAction::Train { runtime, .. }
            | FusedAction::Eval { runtime, .. }
            | FusedAction::Prove { runtime, .. } => Some(runtime),
        },
    };
    let Some(runtime) = runtime else {
        return match action {
            GeomAction::Data {
                points,
                kinds,
                kind_weights,
                count,
                seed,
                augment,
                out_dir,
            } => cmd_geom_data(
                points,
                &kinds,
                &kind_weights,
                count,
                seed,
                augment,
                &out_dir,
            ),
            GeomAction::Solve { graph, out } => cmd_geom_solve(&graph, out.as_deref()),
            GeomAction::Counterexample {
                claim,
                trials,
                grid,
                seed,
            } => cmd_geom_counterexample(&claim, trials, grid, seed),
            // Reached only if a variant is added to the match above without a
            // branch here. Reported rather than aborted, because the argument
            // parser is the thing that would be wrong, and a message naming it is
            // more use than a panic.
            _ => anyhow::bail!(
                "this `geom` subcommand is not dispatched on the backend path; this is a bug in the argument parser"
            ),
        };
    };
    runtime.validate()?;
    if runtime.backend == LmBackend::Wgpu {
        #[cfg(feature = "wgpu")]
        {
            let device = lm_wgpu_device(runtime)?;
            return cmd_geom_backend::<burn::backend::Wgpu<f32, i32>>(action, device);
        }
        #[cfg(not(feature = "wgpu"))]
        anyhow::bail!("WGPU support is not compiled in; rebuild with --features wgpu");
    }
    eprintln!("backend=cpu device=NdArray(Cpu) dtype=f32");
    cmd_geom_backend::<Eval>(action, Default::default())
}

fn cmd_geom_data(
    points: usize,
    kinds: &str,
    kind_weights: &str,
    count: usize,
    seed: u64,
    augment: usize,
    out_dir: &Path,
) -> Result<()> {
    let kinds: Vec<&str> = kinds.split(',').map(|k| k.trim()).collect();
    let weights: Option<Vec<f64>> = if kind_weights.trim().is_empty() {
        None
    } else {
        Some(
            kind_weights
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .with_context(|| format!("parse kind weight '{s}'"))
                })
                .collect::<Result<Vec<f64>>>()?,
        )
    };
    let (tokens, meta) = geometry::generate_corpus_weighted(
        points,
        &kinds,
        weights.as_deref(),
        count,
        seed,
        augment,
    )?;
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let corpus_path = out_dir.join("corpus.bin");
    TokenCorpus::write(&corpus_path, &tokens)?;
    let meta_path = out_dir.join("geom-meta.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
    let tokenizer = ByteTokenizer::new();
    let answers: Vec<String> = meta
        .answer_tokens
        .iter()
        .map(|t| tokenizer.decode_lossy(&[*t]))
        .collect();
    println!(
        "geometric corpus: {} scenes x {} tokens | points {} | kinds {}",
        meta.scenes,
        meta.scene_len,
        meta.points,
        kinds.join(",")
    );
    println!(
        "answer tokens ({}): {} | chance {:.4}",
        answers.len(),
        answers.join(" "),
        1.0 / answers.len().max(1) as f64
    );
    let balance: Vec<String> = meta
        .kinds
        .iter()
        .zip(meta.kind_counts.iter().chain(std::iter::repeat(&0)))
        .map(|(kind, n)| {
            format!(
                "{kind} {n} ({:.1}%)",
                100.0 * *n as f64 / meta.scenes.max(1) as f64
            )
        })
        .collect();
    println!("kind balance: {}", balance.join(", "));
    if meta.augment > 0 {
        println!(
            "augmentation: {} similarity copies per scene (direction: translation + scale only)",
            meta.augment
        );
    }
    println!(
        "wrote {} and {}",
        corpus_path.display(),
        meta_path.display()
    );
    Ok(())
}

fn cmd_geom_solve(graph_path: &Path, out: Option<&Path>) -> Result<()> {
    let text = std::fs::read_to_string(graph_path)
        .with_context(|| format!("read {}", graph_path.display()))?;
    let mut graph: geomkernel::SceneGraph =
        serde_json::from_str(&text).with_context(|| format!("parse {}", graph_path.display()))?;
    let given = graph.facts.len();
    let added = geomkernel::saturate_rules(&mut graph, 64)
        .with_context(|| format!("saturate the scene graph in {}", graph_path.display()))?;
    println!(
        "scene graph: {} points | {} given facts | {} facts derived, each naming its rule and inputs",
        graph.points.len(),
        given,
        added
    );
    for fact in graph.facts.iter().skip(given) {
        match &fact.provenance {
            geomkernel::Provenance::Derived { rule, inputs } => {
                println!(
                    "  {}   <- {} from [{}]",
                    fact.constraint.describe(),
                    rule,
                    inputs.join("; ")
                )
            }
            _ => println!("  {}", fact.constraint.describe()),
        }
    }
    for note in &graph.not_established {
        println!("  not established: {note}");
    }
    if let Some(out) = out {
        std::fs::write(out, serde_json::to_string_pretty(&graph)?)?;
        println!("saturated graph -> {}", out.display());
    }
    Ok(())
}

fn cmd_geom_counterexample(claim_path: &Path, trials: usize, grid: i64, seed: u64) -> Result<()> {
    let text = std::fs::read_to_string(claim_path)
        .with_context(|| format!("read {}", claim_path.display()))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", claim_path.display()))?;
    let premises: Vec<geomkernel::Constraint> = match doc.get("premises") {
        Some(p) if !p.is_null() => serde_json::from_value(p.clone())?,
        _ => Vec::new(),
    };
    let claim: geomkernel::Constraint = serde_json::from_value(
        doc.get("claim")
            .cloned()
            .context("the claim file needs a \"claim\" constraint")?,
    )?;
    let mut rng = StdRng::seed_from_u64(seed);
    match geomkernel::falsify(&mut rng, &premises, &claim, trials, grid)? {
        geomkernel::Verdict::Counterexample { trial, points } => {
            println!("COUNTEREXAMPLE at trial {trial}: the claim is rejected.");
            for (name, x, y) in &points {
                println!("  {name} = ({}, {})", x, y);
            }
            println!("  claim: {}", claim.describe());
        }
        geomkernel::Verdict::Unrefuted { trials, satisfying } => {
            println!(
                "unrefuted: {satisfying} of {trials} draws satisfied the premises and the claim. Evidence, not proof -- the symbolic engine still has to close it."
            );
        }
    }
    Ok(())
}

fn cmd_geom_backend<B: burn::tensor::backend::Backend<FloatElem = f32>>(
    action: GeomAction,
    device: B::Device,
) -> Result<()> {
    // `Module` in scope for `num_params` (a supertrait method): the baseline
    // comparison prints both models' parameter counts.
    use burn::module::Module as _;
    type Train<B2> = burn::backend::Autodiff<B2>;
    match action {
        // These three are dispatched before a device exists (they are pure data
        // and kernel operations), so they never reach this function. Saying so
        // beats aborting: the caller that routed one here has a bug, and a
        // message naming it is more use than a panic.
        GeomAction::Data { .. } | GeomAction::Solve { .. } | GeomAction::Counterexample { .. } => {
            anyhow::bail!(
                "this `geom` subcommand needs no model and is handled before the device is built"
            )
        }
        GeomAction::Train {
            corpus,
            meta,
            steps,
            batch_size,
            lr,
            lr_schedule,
            clip_norm,
            weight_decay,
            seed,
            objective,
            slots,
            dim,
            hidden_size,
            num_blocks,
            refine_steps,
            moe_boxes,
            moe_experts,
            moe_top_k,
            sigma_data,
            moe_balance_weight,
            gamma,
            consistency_weight,
            accumulate,
            ema_decay,
            log_every,
            log,
            out_dir,
            checkpoint_every,
            runtime: _,
        } => {
            anyhow::ensure!(
                steps > 0 && batch_size > 0,
                "steps and batch_size must be positive"
            );
            let meta = load_geom_meta(&meta)?;
            let arch = GeomArch {
                slots,
                dim,
                hidden_size,
                num_blocks,
                refine_steps,
                moe_boxes,
                moe_experts,
                moe_top_k,
                sigma_data,
            };
            let geom_config = geom_config_for(meta.scene_len, &meta.answer_tokens, &arch);
            let objective = GeomObjective::parse(&objective)?;
            println!(
                "geometric reasoner: {} | objective {}",
                geom_config.describe(),
                objective.name()
            );
            let mut corpus = TokenCorpus::streaming(&corpus)?;
            let model = GeometricReasoner::<Train<B>>::new(&geom_config, &device)?;
            if let Some(d) = ema_decay {
                anyhow::ensure!(d > 0.0 && d < 1.0, "--ema-decay must be in (0, 1), got {d}");
            }
            let train_config = geometry::GeomTrainConfig {
                steps,
                batch_size,
                lr,
                lr_schedule: LrSchedule::parse(&lr_schedule, lr, steps)?,
                clip_norm,
                weight_decay,
                seed,
                objective,
                gamma,
                consistency_weight,
                accumulate,
                ema_decay,
                log_every,
                log_path: log,
                out_dir: Some(out_dir),
                checkpoint_every,
                convergence_tol: 1e-3,
                moe_balance_weight,
            };
            let (_model, report) = geometry::train_geom::<Train<B>>(
                model,
                &mut corpus,
                &meta,
                &train_config,
                &device,
            )?;
            println!(
                "trained: {} steps ({} skipped) | loss {:.4} -> {:.4} (mean {:.4})",
                report.steps_taken,
                report.steps_skipped,
                report.first_loss,
                report.last_loss,
                report.mean_loss
            );
            println!(
                "accuracy {:.4} -> {:.4} (mean {:.4}) | energy {:.4} -> {:.4} | displacement {:.6}",
                report.first_accuracy,
                report.last_accuracy,
                report.mean_accuracy,
                report.energy_start,
                report.energy_end,
                report.displacement_end
            );
            println!(
                "converged steps {}/{} | router: load H {:.3} token H {:.3} balance {:.4}",
                report.converged_steps,
                report.steps_taken,
                report.load_entropy_end,
                report.token_entropy_end,
                report.balance_end
            );
            if let Some(path) = &report.checkpoint {
                println!("checkpoint {}", path.display());
            }
            Ok(())
        }
        GeomAction::Baseline {
            corpus,
            meta,
            steps,
            batch_size,
            lr,
            lr_schedule,
            clip_norm,
            weight_decay,
            seed,
            slots,
            dim,
            hidden_size,
            num_blocks,
            refine_steps,
            moe_boxes,
            moe_experts,
            moe_top_k,
            sigma_data,
            accumulate,
            ema_decay,
            log_every,
            log,
            out_dir,
            checkpoint_every,
            runtime: _,
        } => {
            anyhow::ensure!(
                steps > 0 && batch_size > 0,
                "steps and batch_size must be positive"
            );
            let meta = load_geom_meta(&meta)?;
            let arch = GeomArch {
                slots,
                dim,
                hidden_size,
                num_blocks,
                refine_steps,
                moe_boxes,
                moe_experts,
                moe_top_k,
                sigma_data,
            };
            let geom_config = geom_config_for(meta.scene_len, &meta.answer_tokens, &arch);
            let baseline_config = GeomBaselineConfig::matched_to(&geom_config);
            println!("{}", baseline_config.describe());
            // The parameter-count band, printed before training so a broken
            // match is seen before the run, not after it.
            let reasoner_params = GeometricReasoner::<B>::new(&geom_config, &device)?.num_params();
            let model = GeomBaseline::<Train<B>>::new(&baseline_config, &device)?;
            let baseline_params = model.num_params();
            let skew =
                (reasoner_params as f64 - baseline_params as f64).abs() / reasoner_params as f64;
            println!(
                "parameters: reasoner {} | baseline {} ({:.1}% apart, band 25%)",
                reasoner_params,
                baseline_params,
                skew * 100.0
            );
            if let Some(d) = ema_decay {
                anyhow::ensure!(d > 0.0 && d < 1.0, "--ema-decay must be in (0, 1), got {d}");
            }
            let train_config = geometry::GeomTrainConfig {
                steps,
                batch_size,
                lr,
                lr_schedule: LrSchedule::parse(&lr_schedule, lr, steps)?,
                clip_norm,
                weight_decay,
                seed,
                // The answer objective is the only one a baseline has; the
                // diffusion fields are inert in `train_geom_baseline`.
                objective: GeomObjective::Answer,
                gamma: 0.0,
                consistency_weight: 0.0,
                accumulate,
                ema_decay,
                log_every,
                log_path: log,
                out_dir: Some(out_dir),
                checkpoint_every,
                convergence_tol: 1e-3,
                moe_balance_weight: 0.0,
            };
            let mut corpus = TokenCorpus::streaming(&corpus)?;
            let (model, report) = geombaseline::train_geom_baseline::<Train<B>>(
                model,
                &mut corpus,
                &meta,
                &train_config,
                &device,
            )?;
            println!(
                "trained: {} steps ({} skipped) | loss {:.4} -> {:.4} (mean {:.4}) | accuracy {:.4} -> {:.4} (mean {:.4})",
                report.steps_taken,
                report.steps_skipped,
                report.first_loss,
                report.last_loss,
                report.mean_loss,
                report.first_accuracy,
                report.last_accuracy,
                report.mean_accuracy
            );
            // The held-out protocol of `geom eval` (its default batches and
            // seed), so this number lines up with a reasoner evaluated there.
            let (accuracy, seen) = geombaseline::evaluate_geom_baseline::<Train<B>>(
                &model,
                &mut corpus,
                &meta,
                16,
                batch_size,
                42,
                &device,
            )?;
            let chance = 1.0 / meta.answer_tokens.len().max(1) as f64;
            println!(
                "held-out: {} scenes | accuracy {:.4} (chance {:.4}) | {} parameters",
                seen,
                accuracy,
                chance,
                model.num_params()
            );
            if let Some(path) = &report.checkpoint {
                println!("checkpoint {}", path.display());
            }
            Ok(())
        }
        GeomAction::Eval {
            checkpoint,
            corpus,
            meta,
            slots,
            dim,
            hidden_size,
            num_blocks,
            refine_steps,
            moe_boxes,
            moe_experts,
            moe_top_k,
            sigma_data,
            batches,
            batch_size,
            refine_sweep,
            baseline,
            seed,
            runtime: _,
        } => {
            anyhow::ensure!(
                batches > 0 && batch_size > 0,
                "batches and batch_size must be positive"
            );
            let meta = load_geom_meta(&meta)?;
            let arch = GeomArch {
                slots,
                dim,
                hidden_size,
                num_blocks,
                refine_steps,
                moe_boxes,
                moe_experts,
                moe_top_k,
                sigma_data,
            };
            let geom_config = geom_config_for(meta.scene_len, &meta.answer_tokens, &arch);
            B::seed(&device, seed);
            let model = checkpoint::load::<B, _>(
                GeometricReasoner::<B>::new(&geom_config, &device)?,
                &checkpoint,
                &device,
            )?;
            let mut corpus = TokenCorpus::streaming(&corpus)?;
            let chance = 1.0 / meta.answer_tokens.len().max(1) as f64;
            if refine_sweep {
                println!(
                    "refinement sweep (chance {:.4}): more relaxation steps are a runtime knob -- past convergence the curve must flatten, which is the attractor showing itself",
                    chance
                );
                // Trace dependence (truncation-faithfulness analog): the same
                // seed sees the same scenes, so agreement of each depth with
                // depth 1 says whether deeper refinement actually changes
                // answers or is post-hoc decoration on top of them.
                let shallow_eval = geometry::GeomEval {
                    batches,
                    batch_size,
                    refine_override: Some(1),
                    seed,
                };
                let (shallow, _) = geometry::evaluate_geom_answers::<B>(
                    &model,
                    &mut corpus,
                    &meta,
                    &shallow_eval,
                    &device,
                )?;
                for depth in 1..=(geom_config.refine_steps * 2).max(1) {
                    let eval = geometry::GeomEval {
                        batches,
                        batch_size,
                        refine_override: Some(depth),
                        seed,
                    };
                    let (accuracy, seen) =
                        geometry::evaluate_geom::<B>(&model, &mut corpus, &meta, &eval, &device)?;
                    let (preds, _) = geometry::evaluate_geom_answers::<B>(
                        &model,
                        &mut corpus,
                        &meta,
                        &eval,
                        &device,
                    )?;
                    let agree = preds
                        .iter()
                        .zip(shallow.iter())
                        .filter(|(a, b)| a == b)
                        .count() as f32
                        / seen.max(1) as f32;
                    println!(
                        "  refine depth {depth}: accuracy {:.4} | agree-with-depth-1 {:.4} over {seen} scenes",
                        accuracy, agree
                    );
                }
            } else {
                let eval = geometry::GeomEval {
                    batches,
                    batch_size,
                    refine_override: None,
                    seed,
                };
                let (accuracy, seen) =
                    geometry::evaluate_geom::<B>(&model, &mut corpus, &meta, &eval, &device)?;
                println!(
                    "evaluation: {} | {} batches | {} scenes | accuracy {:.4} (chance {:.4})",
                    geom_config.describe(),
                    batches,
                    seen,
                    accuracy,
                    chance
                );
            }
            if let Some(baseline_path) = baseline {
                let baseline_config = GeomBaselineConfig::matched_to(&geom_config);
                B::seed(&device, seed);
                let baseline_model = checkpoint::load::<B, _>(
                    GeomBaseline::<B>::new(&baseline_config, &device)?,
                    &baseline_path,
                    &device,
                )?;
                // The reasoner at its trained depth, on the same held-out
                // scenes the baseline is about to see (the sweep above, when
                // it ran, covers the other depths).
                let eval = geometry::GeomEval {
                    batches,
                    batch_size,
                    refine_override: None,
                    seed,
                };
                let (reasoner_accuracy, _) =
                    geometry::evaluate_geom::<B>(&model, &mut corpus, &meta, &eval, &device)?;
                let (baseline_accuracy, seen) = geombaseline::evaluate_geom_baseline::<B>(
                    &baseline_model,
                    &mut corpus,
                    &meta,
                    batches,
                    batch_size,
                    seed,
                    &device,
                )?;
                println!(
                    "matched comparison: {seen} held-out scenes (seed {seed}), {batches} batches x {batch_size}, chance {chance:.4}"
                );
                println!("  {:<52} {:>10} {:>9}", "model", "params", "accuracy");
                println!(
                    "  {:<52} {:>10} {:>9.4}",
                    format!("reasoner ({})", geom_config.describe()),
                    model.num_params(),
                    reasoner_accuracy
                );
                println!(
                    "  {:<52} {:>10} {:>9.4}",
                    baseline_config.describe(),
                    baseline_model.num_params(),
                    baseline_accuracy
                );
            }
            Ok(())
        }
        GeomAction::Generate {
            scene,
            checkpoint,
            slots,
            dim,
            hidden_size,
            num_blocks,
            refine_steps,
            moe_boxes,
            moe_experts,
            moe_top_k,
            sigma_data,
            diffusion,
            seed,
            runtime: _,
        } => {
            let tokenizer = ByteTokenizer::new();
            let tokens = tokenizer.encode(&scene);
            let scene_len = tokens.len() + 1;
            let arch = GeomArch {
                slots,
                dim,
                hidden_size,
                num_blocks,
                refine_steps,
                moe_boxes,
                moe_experts,
                moe_top_k,
                sigma_data,
            };
            // No metadata here -- the caller hands over one scene, not a corpus
            // -- so the answer set comes from the architecture the checkpoint
            // records beside its weights, exactly as `lm generate` reloads the
            // LM's recorded config. A checkpoint that records none falls back
            // to the unrestricted vocabulary.
            let recorded = checkpoint::TrainState::for_model(&checkpoint)?
                .and_then(|state| state.config.get("model_config").cloned())
                .and_then(|value| serde_json::from_value::<GeomConfig>(value).ok());
            let answers: &[u16] = recorded
                .as_ref()
                .map(|c| c.answer_tokens.as_slice())
                .unwrap_or(&[]);
            let geom_config = geom_config_for(scene_len, answers, &arch);
            if answers.is_empty() {
                println!(
                    "note: this checkpoint records no answer set; scoring the full byte vocabulary"
                );
            }
            B::seed(&device, seed);
            let model = checkpoint::load::<B, _>(
                GeometricReasoner::<B>::new(&geom_config, &device)?,
                &checkpoint,
                &device,
            )?;
            let input = burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_ints(
                tokens
                    .iter()
                    .map(|t| i64::from(*t))
                    .collect::<Vec<_>>()
                    .as_slice(),
                &device,
            )
            .reshape([1, tokens.len()]);
            let out = if diffusion {
                model.forward_diffusion_start(input)
            } else {
                model.forward(input)
            }?;
            let pred = out
                .logits
                .argmax(1)
                .into_data()
                .convert::<i64>()
                .iter::<i64>()
                .next()
                .unwrap_or(0) as u16;
            let answer = tokenizer.decode_lossy(&[pred]);
            println!(
                "answer: {answer} | energy {:.4} | displacement {:.6} | refine {} | blocks {}{}",
                out.energy,
                out.displacement,
                geom_config.refine_steps,
                geom_config.num_blocks,
                if diffusion {
                    " | diffusion start (from pure noise)"
                } else {
                    ""
                }
            );
            Ok(())
        }
        GeomAction::Fused { action } => match *action {
            FusedAction::Train {
                corpus,
                meta,
                weights,
                tokenizer,
                steps,
                batch_size,
                lr,
                fusion_lr,
                accumulate,
                ema_decay,
                log_every,
                out_dir,
                holdout,
                seed,
                objective,
                sigma_data,
                moe_balance_weight,
                gamma,
                depth,
                lora_rank,
                lora_alpha,
                hidden_size,
                geom_hidden,
                slots,
                dim,
                num_blocks,
                refine_steps,
                runtime: _,
            } => {
                let meta = load_geom_meta(&meta)?;
                let fusion_config = GeomFusionConfig {
                    hidden_size,
                    geom_hidden,
                    slots,
                    dim,
                    num_blocks,
                    refine_steps,
                    ..GeomFusionConfig::default()
                };
                fusion_config.validate()?;
                let trunk_config = QwenTrunkConfig {
                    num_layers: None,
                    ..QwenTrunkConfig::qwen38()
                };
                let trunk =
                    qwennet::load_qwen_trunk_nf4::<Train<B>>(&weights, &trunk_config, &device)?.0;
                let model = FusedQwen::<Train<B>>::new(trunk, &fusion_config, &device)?;
                let tok_text = std::fs::read_to_string(&tokenizer)
                    .with_context(|| format!("read {}", tokenizer.display()))?;
                let tokenizer = BpeTokenizer::from_json(&tok_text)?;
                let mut corpus = TokenCorpus::streaming(&corpus)?;
                let objective = GeomObjective::parse(&objective)?;
                let lora = qwennet::QwenLoraConfig {
                    rank: lora_rank,
                    alpha: lora_alpha,
                };
                if let Some(d) = ema_decay {
                    anyhow::ensure!(d > 0.0 && d < 1.0, "--ema-decay must be in (0, 1), got {d}");
                }
                let train_config = FusedTrainConfig {
                    steps,
                    batch_size,
                    lr,
                    fusion_lr,
                    clip_norm: Some(1.0),
                    accumulate,
                    ema_decay,
                    lora,
                    seed,
                    objective,
                    gamma,
                    sigma_data,
                    moe_balance_weight,
                    depth,
                    holdout,
                    log_every,
                };
                train_config.validate()?;
                let mut trainer = geomfusion::FusedQwenTrainer::new(model, train_config, &device)?;
                let report = geomfusion::train_fused_qwen(
                    &mut trainer,
                    &mut corpus,
                    &meta,
                    &tokenizer,
                    Some(&out_dir),
                )?;
                println!(
                    "fused train: {} steps | loss {:.4} -> {:.4} | accuracy {:.4} -> {:.4}",
                    report.steps_taken,
                    report.first_loss,
                    report.last_loss,
                    report.first_accuracy,
                    report.last_accuracy
                );
                if let Some(path) = &report.checkpoint {
                    println!("checkpoint {}", path.display());
                }
                Ok(())
            }
            FusedAction::Eval {
                checkpoint,
                corpus,
                meta,
                weights,
                tokenizer,
                batches: _batches,
                batch_size: _batch_size,
                hidden_size,
                geom_hidden,
                slots,
                dim,
                num_blocks,
                refine_steps,
                lora_rank: _lora_rank,
                lora_alpha: _lora_alpha,
                sigma_data: _sigma_data,
                seed: _seed,
                runtime: _,
            } => {
                let meta = load_geom_meta(&meta)?;
                let fusion_config = GeomFusionConfig {
                    hidden_size,
                    geom_hidden,
                    slots,
                    dim,
                    num_blocks,
                    refine_steps,
                    ..GeomFusionConfig::default()
                };
                fusion_config.validate()?;
                let trunk_config = QwenTrunkConfig {
                    num_layers: None,
                    ..QwenTrunkConfig::qwen38()
                };
                let trunk = qwennet::load_qwen_trunk_nf4::<B>(&weights, &trunk_config, &device)?.0;
                let model = FusedQwen::<B>::new(trunk, &fusion_config, &device)?;
                let model = checkpoint::load(model, &checkpoint, &device)?;
                let tok_text = std::fs::read_to_string(&tokenizer)?;
                let tokenizer = BpeTokenizer::from_json(&tok_text)?;
                let mut corpus = TokenCorpus::streaming(&corpus)?;
                let proof = ProofConfig {
                    scenes: 256.min(corpus.len() / meta.scene_len).max(1),
                    ..ProofConfig::default()
                };
                proof.validate()?;
                let report = geomfusion::prove::<B>(
                    &model,
                    &mut corpus,
                    &meta,
                    &tokenizer,
                    &proof,
                    &device,
                )?;
                println!(
                    "fused eval: scenes {} | accuracy {:.4} (chance {:.4}) | margin {:.4} | verified {} falsified {} indeterminate {}",
                    report.scenes,
                    report.accuracy,
                    report.chance,
                    report.margin,
                    report.verified,
                    report.falsified,
                    report.indeterminate
                );
                Ok(())
            }
            FusedAction::Prove {
                checkpoint,
                corpus,
                meta,
                weights,
                tokenizer,
                scenes,
                depth,
                sweep,
                hidden_size,
                geom_hidden,
                slots,
                dim,
                num_blocks,
                refine_steps,
                lora_rank: _lora_rank,
                lora_alpha: _lora_alpha,
                sigma_data: _sigma_data,
                runtime: _,
            } => {
                let meta = load_geom_meta(&meta)?;
                let fusion_config = GeomFusionConfig {
                    hidden_size,
                    geom_hidden,
                    slots,
                    dim,
                    num_blocks,
                    refine_steps,
                    ..GeomFusionConfig::default()
                };
                fusion_config.validate()?;
                let trunk_config = QwenTrunkConfig {
                    num_layers: None,
                    ..QwenTrunkConfig::qwen38()
                };
                let trunk = qwennet::load_qwen_trunk_nf4::<B>(&weights, &trunk_config, &device)?.0;
                let mut model = FusedQwen::<B>::new(trunk, &fusion_config, &device)?;
                if let Some(path) = &checkpoint {
                    model = checkpoint::load(model, path, &device)?;
                }
                let tok_text = std::fs::read_to_string(&tokenizer)?;
                let tokenizer = BpeTokenizer::from_json(&tok_text)?;
                let mut corpus = TokenCorpus::streaming(&corpus)?;
                let proof = ProofConfig {
                    scenes,
                    depth,
                    sweep,
                    ..ProofConfig::default()
                };
                proof.validate()?;
                let report = geomfusion::prove::<B>(
                    &model,
                    &mut corpus,
                    &meta,
                    &tokenizer,
                    &proof,
                    &device,
                )?;
                println!(
                    "fused prove: scenes {} | accuracy {:.4} (chance {:.4}) | margin {:.4} | verified {} falsified {} indeterminate {}",
                    report.scenes,
                    report.accuracy,
                    report.chance,
                    report.margin,
                    report.verified,
                    report.falsified,
                    report.indeterminate
                );
                for p in &report.sweep {
                    println!(
                        "  depth {} | accuracy {:.4} | energy {:.4} | agree_with_depth1 {:.3}",
                        p.depth, p.accuracy, p.mean_energy, p.agree_with_depth1
                    );
                }
                for k in &report.per_kind {
                    println!(
                        "  kind {} | {}/{} | verified {} falsified {} indeterminate {}",
                        k.kind, k.correct, k.scenes, k.verified, k.falsified, k.indeterminate
                    );
                }
                Ok(())
            }
        },
    }
}
/// Subcommands of the token-block substrate.
#[derive(clap::Subcommand)]
enum TblockAction {
    /// Train the substrate on a byte stream and save a checkpoint.
    #[command(about = "train the token-block substrate on a byte stream")]
    Train {
        /// Text or JSONL file; the first --bytes feed the training stream.
        #[arg(long, default_value = "/srv/m-sda/unifur/pairs/pentest-full.jsonl")]
        data: PathBuf,
        #[arg(long, default_value_t = 1_000_000)]
        bytes: usize,
        #[arg(long, default_value_t = 192)]
        context: usize,
        #[arg(long, default_value_t = 64)]
        hidden: usize,
        #[arg(long, default_value_t = 3)]
        layers: usize,
        #[arg(long, default_value_t = 300)]
        steps: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 25)]
        eval_every: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "target/tblock-checkpoint.mpk")]
        out: PathBuf,
        #[arg(long, default_value = "target/tblock-train.json")]
        report: PathBuf,
    },
    /// Benchmark the decode plans against each other on one checkpoint.
    #[command(about = "compare fixed, adaptive, halo and revision decode plans")]
    Bench {
        #[arg(long, default_value = "target/tblock-checkpoint.mpk")]
        checkpoint: PathBuf,
        #[arg(long, default_value = "/srv/m-sda/unifur/pairs/pentest-full.jsonl")]
        data: PathBuf,
        #[arg(long, default_value_t = 1_000_000)]
        bytes: usize,
        #[arg(long, default_value_t = 20)]
        prompts: usize,
        #[arg(long, default_value_t = 128)]
        gen_len: usize,
        #[arg(long, default_value = "target/tblock-bench.json")]
        out: PathBuf,
    },
}

fn cmd_tblock(action: TblockAction) -> Result<()> {
    match action {
        TblockAction::Train {
            data,
            bytes,
            context,
            hidden,
            layers,
            steps,
            batch,
            lr,
            eval_every,
            seed,
            out,
            report,
        } => {
            let config = TbConfig {
                context,
                hidden_size: hidden,
                num_layers: layers,
                num_heads: 4,
                intermediate_size: hidden * 2,
                ..TbConfig::default()
            };
            config.validate()?;
            let splits = tb::load_stream(&data, bytes, context)?;
            // A fixed subsample: the val estimator is a curve, not a record,
            // and 24 windows bound each eval pass to seconds on CPU.
            let val_sample: Vec<Vec<u16>> = splits.val.iter().take(24).cloned().collect();
            let (train_windows, val_windows) = (&splits.train, &val_sample);
            println!(
                "tokenblocks train: {} train windows, {} val windows, config {config:?}",
                train_windows.len(),
                val_windows.len()
            );
            type Ad = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
            let device = Default::default();
            let params = TbTrainParams {
                steps,
                batch_size: batch,
                lr,
                eval_every,
                seed,
            };
            let (model, training) =
                tb::train_model::<Ad>(&config, &device, train_windows, val_windows, &params)?;
            println!(
                "final val loss {:.4} ({:.4} bpb), {} steps in {:.1}s",
                training.final_val_loss, training.final_val_bpb, steps, training.seconds
            );
            for (step, val) in &training.val_curve {
                println!("  step {step:>5}  val loss {val:.4}");
            }
            let inference = model.valid();
            tb::save_checkpoint(inference, &out)?;
            println!("checkpoint saved to {}", out.display());
            let json =
                serde_json::to_string_pretty(&training).context("serialize training report")?;
            std::fs::write(&report, json).with_context(|| format!("write {}", report.display()))?;
        }
        TblockAction::Bench {
            checkpoint,
            data,
            bytes,
            prompts,
            gen_len,
            out,
        } => {
            let config = TbConfig {
                context: 192,
                hidden_size: 64,
                num_layers: 3,
                num_heads: 4,
                intermediate_size: 128,
                ..TbConfig::default()
            };
            let device = Default::default();
            let model = tb::load_checkpoint(&checkpoint, &config, &device)?;
            let splits = tb::load_stream(&data, bytes, config.context)?;
            let test = &splits.test;
            anyhow::ensure!(
                test.len() >= prompts,
                "want {prompts} prompts, the test split has {} windows",
                test.len()
            );
            // The prompt is the front of each held-out window; the model
            // generates the rest.
            let prompt_len = config.context.saturating_sub(gen_len).max(8);
            let seeds: Vec<Vec<u16>> = test
                .iter()
                .take(prompts)
                .map(|w| w[..prompt_len].to_vec())
                .collect();
            println!(
                "tokenblocks bench: {} prompts x {} tokens, from {}",
                seeds.len(),
                gen_len,
                checkpoint.display()
            );
            let mut results = Vec::new();
            for (name, plan) in tb::benchmark_plans(&config) {
                let r = tb::run_plan(&name, &model, &config, &device, &plan, &seeds, gen_len)?;
                println!(
                    "{name:<24} {seconds:>7.2}s  fw={fw:>5} (sizing {sz:>3}, revision {rev:>3})  conf {conf:.3}  nll {nll:.4} ({bpb:.3} bpb)  edits {edits:>4} thrash {thrash}",
                    seconds = r.seconds,
                    fw = r.forwards,
                    sz = r.sizing_forwards,
                    rev = r.revision_forwards,
                    conf = r.mean_confidence,
                    nll = r.continuation_nll,
                    bpb = r.continuation_bpb,
                    edits = r.edits_total,
                    thrash = r.thrashed_positions,
                );
                let sizes: Vec<String> = r
                    .sizes
                    .iter()
                    .map(|(size, count)| format!("{size}x{count}"))
                    .collect();
                println!("    sizes: {}", sizes.join(" "));
                let curve: Vec<String> = r
                    .confidence_by_offset
                    .iter()
                    .step_by((r.confidence_by_offset.len() / 12).max(1))
                    .map(|(offset, conf)| format!("{offset}:{conf:.3}"))
                    .collect();
                println!("    conf@offset: {}", curve.join(" "));
                results.push(r);
            }
            let json = serde_json::to_string_pretty(&results).context("serialize bench report")?;
            std::fs::write(&out, json).with_context(|| format!("write {}", out.display()))?;
            println!("report written to {}", out.display());
        }
    }
    Ok(())
}

fn cmd_lm(action: LmAction) -> Result<()> {
    let runtime = match &action {
        LmAction::Train { runtime, .. }
        | LmAction::Generate { runtime, .. }
        | LmAction::Evaluate { runtime, .. } => Some(runtime),
        _ => None,
    };
    if let Some(runtime) = runtime {
        runtime.validate()?;
        if runtime.backend == LmBackend::Wgpu {
            #[cfg(feature = "wgpu")]
            {
                let device = lm_wgpu_device(runtime)?;
                return cmd_lm_backend::<burn::backend::Wgpu<f32, i32>>(action, device);
            }
            #[cfg(not(feature = "wgpu"))]
            anyhow::bail!("WGPU support is not compiled in; rebuild with --features wgpu");
        }
        eprintln!("backend=cpu device=NdArray(Cpu) dtype=f32");
    }
    cmd_lm_backend::<Eval>(action, Default::default())
}

fn cmd_lm_backend<B: burn::tensor::backend::Backend<FloatElem = f32>>(
    action: LmAction,
    lm_device: B::Device,
) -> Result<()> {
    match action {
        LmAction::ExportSpecialist {
            checkpoint,
            specialist,
            out,
            arch,
        } => cmd_lm_specialist(&checkpoint, None, &specialist, &out, &arch),
        LmAction::ApplySpecialist {
            checkpoint,
            trained,
            specialist,
            out,
            arch,
        } => cmd_lm_specialist(&checkpoint, Some(&trained), &specialist, &out, &arch),
        LmAction::Evaluate {
            checkpoint,
            corpus: corpus_path,
            batches,
            batch_size,
            seed,
            arch,
            ..
        } => {
            anyhow::ensure!(
                batches > 0 && batch_size > 0,
                "batches and batch_size must be positive"
            );
            let config = lm_config_for(&arch, Some(&checkpoint))?;
            let device = lm_device;
            B::seed(&device, seed);
            let model = checkpoint::load::<B, _>(
                LanguageModel::<B>::new(&config, &device)?,
                &checkpoint,
                &device,
            )?;
            let mut corpus = TokenCorpus::streaming(&corpus_path)?;
            let has_mask = corpus::mask_path(&corpus_path).exists();
            if has_mask {
                corpus.open_mask()?;
            }
            let mut rng = StdRng::seed_from_u64(seed);
            let mut total_loss = 0.0f64;
            let mut total_tokens = 0usize;
            for _ in 0..batches {
                // Evaluation deliberately measures plain next-token CE: the
                // number has to stay comparable across runs that were graded
                // differently, so grades are not requested here.
                let (windows, _, mask_rows, _) = corpus.sample_batch_with_sides(
                    batch_size,
                    config.context - 1,
                    false,
                    has_mask,
                    false,
                    &mut rng,
                )?;
                let flat: Vec<i64> = windows
                    .iter()
                    .flat_map(|window| window.iter().map(|token| i64::from(*token)))
                    .collect();
                let tokens = burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_ints(
                    flat.as_slice(),
                    &device,
                )
                .reshape([batch_size, config.context]);
                let (_loss, metrics) = if let Some(rows) = mask_rows {
                    let rows: Vec<Vec<f32>> = rows
                        .into_iter()
                        .map(|row| {
                            row.into_iter()
                                .map(|value| if value == 0 { 0.0 } else { 1.0 })
                                .collect()
                        })
                        .collect();
                    let mask = diffusionblocks::mix::mask_rows::<B>(&rows, &device);
                    model.next_token_loss_masked(tokens, mask, 0..config.num_layers)
                } else {
                    model.next_token_loss(tokens, 0..config.num_layers)
                };
                anyhow::ensure!(
                    metrics.loss.is_finite(),
                    "evaluation produced a non-finite loss"
                );
                total_loss += f64::from(metrics.loss) * metrics.tokens_counted as f64;
                total_tokens += metrics.tokens_counted;
            }
            let mean = total_loss / total_tokens.max(1) as f64;
            println!(
                "evaluation: {} | {} batches | {} tokens | loss {:.4} ppl {:.2}",
                config.describe(),
                batches,
                total_tokens,
                mean,
                mean.exp()
            );
            Ok(())
        }
        LmAction::Tokenize {
            input,
            out,
            label,
            rules,
        } => {
            let count = TokenCorpus::tokenize_file(&input, &out)?;
            println!(
                "{} -> {} | {count} tokens ({} bytes, u16 little-endian)",
                input.display(),
                out.display(),
                count * corpus::TOKEN_BYTES
            );
            if label {
                let manifest = TokenCorpus::label_file(&out, &load_labeler(rules.as_deref())?)?;
                println!(
                    "labels -> {}\n{}",
                    corpus::labels_path(&out).display(),
                    manifest.render()
                );
            }
            Ok(())
        }
        LmAction::Corpus { path, context } => {
            // Opened streaming on purpose: reporting on a corpus must not
            // require enough memory to hold it.
            let corpus = TokenCorpus::streaming(&path)?;
            println!(
                "{}: {} tokens | {} training windows at context {context}",
                path.display(),
                corpus.len(),
                corpus.windows(context)
            );
            if corpus::manifest_path(&path).exists() {
                print!("labels: {}", corpus.manifest()?.render());
            } else {
                println!(
                    "labels: none (`dblocks lm label --corpus {}` to add them)",
                    path.display()
                );
            }
            Ok(())
        }
        LmAction::Label {
            corpus: path,
            rules,
        } => {
            let manifest = TokenCorpus::label_file(&path, &load_labeler(rules.as_deref())?)?;
            println!(
                "{} -> {} + {}\n{}",
                path.display(),
                corpus::labels_path(&path).display(),
                corpus::manifest_path(&path).display(),
                manifest.render()
            );
            Ok(())
        }
        LmAction::Grade { corpus: path } => {
            let manifest = TokenCorpus::grade_file_from_labels(&path)?;
            println!(
                "{} -> {} + {}\n{}",
                path.display(),
                diffusionblocks::grade::grades_path(&path).display(),
                diffusionblocks::grade::manifest_path(&path).display(),
                manifest.summary()
            );
            println!(
                "training: `dblocks lm train --corpus {} --penalty <alpha>` will now charge those tokens instead of learning them",
                path.display()
            );
            Ok(())
        }
        LmAction::Scan { input, rules } => {
            let text = std::fs::read_to_string(&input)
                .map_err(|err| anyhow::anyhow!("read {}: {err}", input.display()))?;
            let labeler = load_labeler(rules.as_deref())?;
            let report = labeler.report(&text);
            if report.is_empty() {
                println!("{}: no findings", input.display());
            } else {
                print!("{report}");
                println!(
                    "{} finding(s) in {}",
                    report.lines().count(),
                    input.display()
                );
            }
            Ok(())
        }
        LmAction::Score {
            input,
            language,
            lexical,
            structural,
            external,
            out,
        } => {
            use diffusionblocks::codequality::{
                CodeAnalyzer, CompositeAnalyzer, ExternalAnalyzer as ExtTool, Language as Lang,
                QualityScore, StructuralAnalyzer,
            };
            let text = std::fs::read_to_string(&input)
                .map_err(|err| anyhow::anyhow!("read {}: {err}", input.display()))?;
            let lang = Lang::parse(&language);
            let lexical_labeler = if lexical {
                Some(load_labeler(None)?)
            } else {
                None
            };
            let structural_analyzer = if structural {
                Some(StructuralAnalyzer::default())
            } else {
                None
            };
            let external_analyzer = external.as_ref().map(|tool| {
                // The default invocation is a placeholder: users who care
                // about this dimension configure their own command via the
                // analyzer API. The CLI exists so the dimension is reachable
                // without writing Rust code.

                let args = match tool.as_str() {
                    "clippy" => vec!["clippy".into(), "--message-format=json".into()],
                    "ruff" => vec!["ruff".into(), "check".into(), "--output-format=json".into()],
                    "eslint" => vec!["eslint".into(), "--format=json".into()],
                    other => vec![other.into()],
                };
                ExtTool::new(tool.clone(), args)
            });

            struct Identity(Lang);
            impl diffusionblocks::codequality::CodeAnalyzer for Identity {
                fn language(&self) -> Lang {
                    self.0
                }
                fn analyze(&self, _: &str) -> QualityScore {
                    QualityScore::identity(self.0)
                }
            }
            let composite = CompositeAnalyzer::<Identity>::new(
                lang,
                lexical_labeler,
                structural_analyzer,
                external_analyzer,
            );
            let score = composite.analyze(&text);

            let rendered = format!(
                "{}: language={} overall={:.4} lines={}\n",
                input.display(),
                lang.name(),
                score.overall,
                score.lines,
            );
            let dims = score
                .dimensions
                .iter()
                .map(|d| format!("  {:<11} {:.4}\n", d.name, d.score))
                .collect::<String>();
            let report_text = format!("{rendered}{dims}");
            match out {
                Some(path) => {
                    let json = serde_json::to_string_pretty(&score)
                        .map_err(|err| anyhow::anyhow!("serialize: {err}"))?;
                    std::fs::write(&path, format!("{json}\n"))
                        .map_err(|err| anyhow::anyhow!("write {}: {err}", path.display()))?;
                    println!("{report_text}wrote {}", path.display());
                }
                None => print!("{report_text}"),
            }
            Ok(())
        }
        LmAction::Rules { out, check } => {
            if let Some(path) = &check {
                let set = RuleSet::read(path)?;
                println!(
                    "{}: {} categories, {} rules, every example and counterexample holds",
                    path.display(),
                    set.categories.len(),
                    set.rules.len()
                );
            }
            match out {
                Some(path) => {
                    RuleSet::builtin().write(&path)?;
                    println!("built-in rule set written to {}", path.display());
                }
                None if check.is_none() => print!("{}", RuleSet::builtin().to_json()?),
                None => {}
            }
            Ok(())
        }
        LmAction::Train {
            corpus: corpus_paths,
            corpus_weights,
            mix,
            teacher,
            teacher_weights,
            distill_weight,
            distill_temperature,
            negative_teacher,
            negative_confidence,
            negative_penalty,
            direction,
            direction_weight,
            heretic_target,
            heretic_baseline,
            heretic_trials,
            heretic_kl_weight,
            heretic_max_new,
            steps,
            batch_size,
            lr,
            lr_schedule,
            accumulate,
            clip_norm,
            ema_decay,
            weight_decay,
            penalty,
            streaming,
            arch,
            seed,
            log_every,
            log,
            out_dir,
            checkpoint_every,
            resume,
            base_weights,
            specialist,
            runtime: _,
        } => {
            type Train<B> = burn::backend::Autodiff<B>;
            let device = lm_device;
            B::seed(&device, seed);
            let model_config = lm_config_for(&arch, resume.as_ref().or(base_weights.as_ref()))?;

            let corpus_path = corpus_paths[0].clone();
            let mut corpora: Vec<TokenCorpus> = Vec::with_capacity(corpus_paths.len());
            let mut any_labels = false;
            let mut any_grades = false;
            for path in &corpus_paths {
                let mut corpus = if streaming {
                    TokenCorpus::streaming(path)?
                } else {
                    TokenCorpus::in_memory(path)?
                };
                if corpus::mask_path(path).exists() {
                    corpus.open_mask()?;
                    println!(
                        "mask ({}): prompt tokens excluded from loss",
                        path.display()
                    );
                }
                if corpus::labels_path(path).exists() {
                    corpus.open_labels()?;
                    let manifest = corpus.manifest()?;
                    println!(
                        "labels ({}): {} of {} tokens flagged ({:.3}%) across {} categories | penalty {}",
                        path.display(),
                        manifest.labeled_tokens,
                        manifest.tokens,
                        100.0 * manifest.labeled_fraction(),
                        manifest.categories.iter().filter(|c| c.tokens > 0).count(),
                        if penalty > 0.0 {
                            format!("alpha={penalty}")
                        } else {
                            "off (measuring only)".into()
                        }
                    );
                    any_labels = true;
                } else {
                    println!("labels ({}): none", path.display());
                }
                if diffusionblocks::grade::grades_path(path).exists() {
                    corpus.open_grades()?;
                    println!(
                        "grades ({}): {}",
                        path.display(),
                        corpus.grade_manifest()?.summary()
                    );
                    any_grades = true;
                }
                corpora.push(corpus);
            }
            if penalty > 0.0 && !any_labels && !any_grades {
                anyhow::bail!(
                    "--penalty {penalty} needs labels or grades; run `dblocks lm label --corpus {}` and then `dblocks lm grade --corpus {}` first",
                    corpus_path.display(),
                    corpus_path.display()
                );
            }
            let weights = diffusionblocks::mix::MixWeights::parse(&corpus_weights, corpora.len())?;
            let mix_mode = diffusionblocks::mix::MixMode::parse(&mix)?;
            let mut mix = diffusionblocks::mix::CorpusMix::new(
                corpora.iter_mut().collect(),
                weights,
                mix_mode,
            )?;

            let model = LanguageModel::<Train<B>>::new(&model_config, &device)?;
            let cost = model_config.cost(model_config.context)?.total();
            println!(
                "training: {} tokens in {} corpus(es) ({}) | {} | steps={steps} batch={batch_size} lr={lr}",
                mix.total_tokens(),
                mix.len(),
                mix.mode().name(),
                model_config.describe(),
            );
            println!(
                "cost per token at context {}: {} active parameters, {:.3e} FLOPs, {} keys read, {} floats of decode state",
                model_config.context,
                cost.active_params,
                cost.flops,
                cost.keys_read,
                cost.state_floats
            );
            if std::any::type_name::<B>().contains("NdArray")
                && model_config
                    .attention_schedule()
                    .modes
                    .iter()
                    .any(|m| matches!(m, AttentionMode::Retrieval { .. }))
            {
                println!(
                    "warning: retrieval attention on a CPU backend computes every score and only skips value reads, \
                     so wall-clock measures the backend rather than the architecture (docs/Hybrid-Attention.md); \
                     read the quality-per-FLOP axis from the counted keys above, or train on wgpu"
                );
            }
            let teacher_weights = parse_weights(&teacher_weights)?;
            let mut inputs = train::LmTrainInputs::<Train<B>>::default();
            for (i, path) in teacher.iter().enumerate() {
                let loaded = checkpoint::load::<Train<B>, _>(
                    LanguageModel::<Train<B>>::new(&model_config, &device)?,
                    path,
                    &device,
                )?;
                let w = teacher_weights.get(i).copied().unwrap_or(1.0);
                println!("teacher {}: {} (weight {w})", i + 1, path.display());
                inputs.teachers.push((loaded, w));
            }
            if let Some(path) = &negative_teacher {
                inputs.negative_teacher = Some(checkpoint::load::<Train<B>, _>(
                    LanguageModel::<Train<B>>::new(&model_config, &device)?,
                    path,
                    &device,
                )?);
                println!(
                    "negative teacher: {} (confidence >= {negative_confidence}, penalty {negative_penalty})",
                    path.display()
                );
            }

            if let Some(d) = ema_decay {
                anyhow::ensure!(d > 0.0 && d < 1.0, "--ema-decay must be in (0, 1), got {d}");
            }
            let config = train::LmTrainConfig {
                steps,
                batch_size,
                lr,
                lr_schedule: diffusionblocks::schedule::LrSchedule::parse(&lr_schedule, lr, steps)?,
                accumulate,
                clip_norm,
                ema_decay,
                weight_decay,
                seed,
                penalty: Unlikelihood::new(penalty),
                log_every,
                log_path: log,
                bias_balance_rate: 0.0,
                out_dir: Some(out_dir.clone()),
                checkpoint_every,
                resume,
                model_config: Some(model_config.clone()),
                distill_weight: if teacher.is_empty() {
                    0.0
                } else {
                    distill_weight
                },
                distill_temperature,
                negative_confidence,
                negative_penalty: if negative_teacher.is_some() {
                    negative_penalty
                } else {
                    0.0
                },
                direction: direction
                    .as_deref()
                    .map(diffusionblocks::ablation::Direction::read)
                    .transpose()?,
                direction_weight: if direction.is_some() {
                    direction_weight
                } else {
                    0.0
                },
                heretic: match (&heretic_target, &heretic_baseline) {
                    (Some(t), Some(b)) => Some(diffusionblocks::heretic::HereticConfig {
                        target: diffusionblocks::heretic::HereticConfig::read_prompts(t)?,
                        baseline: diffusionblocks::heretic::HereticConfig::read_prompts(b)?,
                        trials: heretic_trials,
                        kl_weight: heretic_kl_weight,
                        max_new: heretic_max_new,
                        seed,
                        detector: None,
                    }),
                    _ => None,
                },
            };
            if let Some(d) = &config.direction {
                println!(
                    "direction: layer {} of {} dims, separation {:.3}, weight {}",
                    d.layer,
                    d.hidden_size(),
                    d.separation,
                    config.direction_weight
                );
            }
            let resident = train::LmResidentTraining {
                scope: match specialist {
                    Some(expert_id) => train::LmTrainingScope::Specialist { expert_id },
                    None => train::LmTrainingScope::Joint,
                },
                base_weights,
            };
            let (model, report) =
                train::train_lm_resident(model, &mut mix, &inputs, &config, &resident, &device)?;
            println!(
                "done: {} steps in {:.1}s | loss {:.4} -> {:.4} (mean {:.4}) | {} skipped",
                report.steps_taken,
                report.elapsed_secs,
                report.first_loss,
                report.last_loss,
                report.mean_loss,
                report.steps_skipped
            );
            if report.penalized_tokens > 0 {
                println!(
                    "flagged targets: {} seen | p(bad) {:.4} -> {:.4}",
                    report.penalized_tokens,
                    report.first_penalized_prob,
                    report.last_penalized_prob
                );
            }
            if report.negative_teacher_tokens > 0 {
                println!(
                    "negative teacher: {} proposals charged | p {:.4} -> {:.4}",
                    report.negative_teacher_tokens,
                    report.first_negative_teacher_prob,
                    report.last_negative_teacher_prob
                );
            }
            if config.direction.is_some() {
                println!(
                    "direction penalty: mean squared projection {:.6} -> {:.6}",
                    report.first_direction_projection, report.last_direction_projection
                );
            }
            if let Some(h) = &report.heretic {
                match &h.best {
                    Some(t) => println!(
                        "heretic: refusals {:.3} -> {:.3} | kl {:.4} | {} parameter(s) ablated (saved as the final checkpoint)",
                        h.baseline_refusals, t.refusals, t.kl, t.touched
                    ),
                    None => println!("heretic: no trial ran"),
                }
            }
            if report.last_distill_loss > 0.0 {
                println!(
                    "distillation term at the last step: {:.4}",
                    report.last_distill_loss
                );
            }
            let path = match report.checkpoint {
                Some(path) => path,
                None => checkpoint::save_content_addressed(model, &out_dir, "lm")?,
            };
            println!(
                "checkpoint saved: {} (training state beside it)",
                path.display()
            );
            Ok(())
        }
        LmAction::Direction {
            checkpoint,
            target,
            baseline,
            layer,
            out,
            tiny,
        } => {
            use diffusionblocks::ablation;
            use diffusionblocks::heretic::HereticConfig;
            let (model, device) = load_lm_eval(&checkpoint, tiny)?;
            let tokenizer = ByteTokenizer::new();
            let (target, baseline) = (
                HereticConfig::read_prompts(&target)?,
                HereticConfig::read_prompts(&baseline)?,
            );
            let residuals = |prompts: &[String]| -> Vec<Vec<Vec<f32>>> {
                prompts
                    .iter()
                    .map(|p| model.residuals_at_last_position(&tokenizer.encode(p), &device))
                    .collect()
            };
            let (t, b) = (residuals(&target), residuals(&baseline));
            let layers = model.num_layers();
            let mut candidates = Vec::new();
            for l in 0..layers {
                let tl: Vec<Vec<f32>> = t.iter().map(|r| r[l].clone()).collect();
                let bl: Vec<Vec<f32>> = b.iter().map(|r| r[l].clone()).collect();
                if let Some(d) = ablation::extract(l, &tl, &bl) {
                    println!(
                        "layer {l}: separation {:.4} | target {:.4} baseline {:.4}",
                        d.separation, d.target_mean_projection, d.baseline_mean_projection
                    );
                    candidates.push(d);
                }
            }
            let chosen = match layer {
                Some(l) => candidates
                    .into_iter()
                    .find(|d| d.layer == l)
                    .ok_or_else(|| anyhow::anyhow!("layer {l} gave no direction"))?,
                None => ablation::best(&candidates)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("no layer separates the prompt sets"))?,
            };
            chosen.write(&out)?;
            println!(
                "direction: layer {} ({} target, {} baseline prompts) -> {}",
                chosen.layer,
                target.len(),
                baseline.len(),
                out.display()
            );
            Ok(())
        }
        LmAction::Ablate {
            checkpoint,
            direction,
            out,
            tiny,
        } => {
            use diffusionblocks::ablation;
            let (model, device) = load_lm_eval(&checkpoint, tiny)?;
            let d = ablation::Direction::read(&direction)?;
            anyhow::ensure!(
                d.hidden_size() == model.hidden_size(),
                "direction has {} dims, the model {}",
                d.hidden_size(),
                model.hidden_size()
            );
            let gates = model.layer_gates(&device);
            let before = ablation::residual_projection::<Eval, _>(&model, &d, Some(&gates));
            let (ablated, touched) =
                ablation::orthogonalize::<Eval, _>(model, &d, Some(gates.clone()));
            let after = ablation::residual_projection::<Eval, _>(&ablated, &d, Some(&gates));
            let parent = checkpoint::file_sha256_hex(&checkpoint)?;
            let path = checkpoint::save_content_addressed(ablated, &out, "lm")?;
            write_derived_state(
                &path,
                "ablate",
                serde_json::json!({ "parent": parent, "direction": &d, "touched": touched }),
            )?;
            println!(
                "ablated {touched} parameter(s) | max |W d| {before:.3e} -> {after:.3e} | {}",
                path.display()
            );
            Ok(())
        }
        LmAction::DirectionScore {
            checkpoint,
            direction,
            prompts,
            ablated,
            tiny,
        } => {
            use diffusionblocks::ablation;
            use diffusionblocks::heretic::HereticConfig;
            let (model, device) = load_lm_eval(&checkpoint, tiny)?;
            let d = ablation::Direction::read(&direction)?;
            anyhow::ensure!(
                d.hidden_size() == model.hidden_size(),
                "direction has {} dims, the model {}",
                d.hidden_size(),
                model.hidden_size()
            );
            let tokenizer = ByteTokenizer::new();
            let tensor = d.tensor::<Eval>(&device);
            let layer = d.layer.min(model.num_layers() - 1);
            println!(
                "{:<40} {:>8} {:>12}{}",
                "prompts",
                "count",
                "projection",
                if ablated { "     ablated" } else { "" }
            );
            for file in &prompts {
                let lines = HereticConfig::read_prompts(file)?;
                let mut sum = 0.0f64;
                let mut sum_ablated = 0.0f64;
                for p in &lines {
                    let ids = tokenizer.encode(p);
                    let r = model.residuals_at_last_position(&ids, &device);
                    sum += r[layer]
                        .iter()
                        .zip(&d.vector)
                        .map(|(a, b)| f64::from(a * b))
                        .sum::<f64>();
                    if ablated {
                        let ids64: Vec<i64> = if ids.is_empty() {
                            vec![i64::from(diffusionblocks::tokenizer::Special::Bos.id())]
                        } else {
                            ids.iter().map(|t| i64::from(*t)).collect()
                        };
                        let start = ids64.len().saturating_sub(model.context());
                        let n = ids64.len() - start;
                        let tokens = burn::tensor::Tensor::<Eval, 1, burn::tensor::Int>::from_ints(
                            &ids64[start..],
                            &device,
                        )
                        .reshape([1, n]);
                        let (_, states) =
                            model.forward_span_states(tokens, 0..model.num_layers(), Some(&tensor));
                        let h: Vec<f32> = states[layer]
                            .clone()
                            .narrow(1, n - 1, 1)
                            .reshape([d.hidden_size()])
                            .into_data()
                            .convert::<f32>()
                            .iter::<f32>()
                            .collect();
                        sum_ablated += h
                            .iter()
                            .zip(&d.vector)
                            .map(|(a, b)| f64::from(a * b))
                            .sum::<f64>();
                    }
                }
                let count = lines.len().max(1) as f64;
                let mut line = format!(
                    "{:<40} {:>8} {:>12.5}",
                    file.display(),
                    lines.len(),
                    sum / count
                );
                if ablated {
                    line.push_str(&format!(" {:>11.3e}", sum_ablated / count));
                }
                println!("{line}");
            }
            Ok(())
        }
        LmAction::Heretic {
            checkpoint,
            target,
            baseline,
            trials,
            kl_weight,
            max_new,
            seed,
            out,
            json,
            tiny,
        } => {
            use diffusionblocks::heretic::{self, HereticConfig};
            let (model, device) = load_lm_eval(&checkpoint, tiny)?;
            let config = HereticConfig {
                target: HereticConfig::read_prompts(&target)?,
                baseline: HereticConfig::read_prompts(&baseline)?,
                trials,
                kl_weight,
                max_new,
                seed,
                detector: None,
            };
            let (decensored, report) = heretic::decensor(model, &config, &device)?;
            print!("{}", heretic::render(&report.trials));
            let parent = checkpoint::file_sha256_hex(&checkpoint)?;
            let path = checkpoint::save_content_addressed(decensored, &out, "lm")?;
            write_derived_state(
                &path,
                "heretic",
                serde_json::json!({ "parent": parent, "best": &report.best, "baseline_refusals": report.baseline_refusals }),
            )?;
            match &report.best {
                Some(t) => println!(
                    "best: refusals {:.3} -> {:.3} | kl {:.4} | {} parameter(s) ablated -> {}",
                    report.baseline_refusals,
                    t.refusals,
                    t.kl,
                    t.touched,
                    path.display()
                ),
                None => println!(
                    "no trial ran; saved the untouched weights -> {}",
                    path.display()
                ),
            }
            if let Some(json) = json {
                std::fs::write(&json, serde_json::to_string_pretty(&report)?)?;
                println!("report: {}", json.display());
            }
            Ok(())
        }
        LmAction::Policy { action } => cmd_policy(action),
        LmAction::Approvals { action } => cmd_approvals(action),
        LmAction::RefusalCorpus {
            policy,
            prompts,
            answers,
            out,
        } => {
            use diffusionblocks::policy::{refusal_documents, Policy};
            let policy = Policy::read(&policy)?;
            let read_lines = |path: &Path| -> Result<Vec<String>> {
                Ok(std::fs::read_to_string(path)
                    .map_err(|err| anyhow::anyhow!("read {}: {err}", path.display()))?
                    .lines()
                    .map(str::to_string)
                    .collect())
            };
            let prompt_lines = read_lines(&prompts)?;
            let answer_lines = answers.as_deref().map(read_lines).transpose()?;
            let docs = refusal_documents(&policy, &prompt_lines, answer_lines.as_deref());
            let tokenizer = ByteTokenizer::new();
            let tokens: Vec<u16> = docs
                .iter()
                .flat_map(|d| tokenizer.encode_document(d))
                .collect();
            TokenCorpus::write(&out, &tokens)?;
            println!(
                "{} prompt(s) -> {} document(s), {} tokens -> {}",
                prompt_lines.len(),
                docs.len(),
                tokens.len(),
                out.display()
            );
            Ok(())
        }
        LmAction::Merge {
            input,
            weights,
            out,
            arch,
        } => {
            let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
            let config = lm_config_for(&arch, input.first())?;
            let template = LanguageModel::<Eval>::new(&config, &device)?;
            let weights = if weights.trim().is_empty() {
                vec![1.0; input.len()]
            } else {
                parse_weights(&weights)?
            };
            let (merged, parents) = diffusionblocks::merge::merge_checkpoints::<Eval, _>(
                template, &input, &weights, &device,
            )?;
            let path = checkpoint::save_content_addressed(merged, &out, "lm")?;
            write_merge_state(&path, &parents, &weights, &input)?;
            println!(
                "merged {} -> {}",
                diffusionblocks::merge::describe(&parents, &weights),
                path.display()
            );
            Ok(())
        }
        LmAction::Bench {
            corpus: corpus_path,
            steps,
            seeds,
            batch_size,
            axis,
            json,
        } => {
            use diffusionblocks::experiment::{Record, RunLog};
            type Train = train::DefaultTrainBackend;
            let device: <Train as burn::tensor::backend::BackendTypes>::Device = Default::default();
            let seeds = parse_seeds(&seeds)?;
            let mut corpus = TokenCorpus::in_memory(&corpus_path)?;
            let variants = lm_bench_variants(&axis)?;
            println!("axis: {axis}");
            println!(
                "{:<14} {:>6} {:>12} {:>12} {:>10} {:>12} {:>11}",
                "variant", "seeds", "final loss", "±ci95", "ms/step", "active par.", "FLOPs/tok"
            );
            println!("{}", "-".repeat(84));
            let mut records = Vec::new();
            for (name, model_config, rate) in variants {
                let cost = model_config.cost(model_config.context)?.total();
                let train_config = train::LmTrainConfig {
                    steps,
                    batch_size,
                    log_every: 0,
                    bias_balance_rate: rate,
                    model_config: Some(model_config.clone()),
                    ..Default::default()
                };
                let mut record = Record::new(
                    format!("lm-bench/{name}"),
                    "loss",
                    serde_json::to_value(&train_config).map_err(|e| anyhow::anyhow!("{e}"))?,
                    seeds.clone(),
                );
                let mut ms_per_step = Vec::new();
                for &seed in &seeds {
                    <Train as burn::tensor::backend::Backend>::seed(&device, seed);
                    let model = LanguageModel::<Train>::new(&model_config, &device)?;
                    let cfg = train::LmTrainConfig {
                        seed,
                        ..train_config.clone()
                    };
                    let (_, report) = train::train_lm(model, &mut corpus, &cfg, &device)?;
                    record.push(f64::from(report.last_loss), false);
                    ms_per_step.push(1e3 * report.elapsed_secs / report.steps_taken.max(1) as f64);
                }
                let mean_ms = ms_per_step.iter().sum::<f64>() / ms_per_step.len() as f64;
                record.extra = serde_json::json!({
                    "ms_per_step": ms_per_step,
                    "forward_passes_per_token": 1,
                    "axis": axis,
                    "architecture": model_config.describe(),
                    "active_params_per_token": cost.active_params,
                    "flops_per_token": cost.flops,
                    "keys_read_per_token": cost.keys_read,
                    "decode_state_floats": cost.state_floats,
                });
                let summary = record.summary.context("no seed produced a measurement")?;
                println!(
                    "{:<14} {:>6} {:>12.4} {:>12.4} {:>10.1} {:>12} {:>11.3e}",
                    name,
                    summary.n,
                    summary.mean,
                    if summary.ci95_half_width.is_nan() {
                        0.0
                    } else {
                        summary.ci95_half_width
                    },
                    mean_ms,
                    cost.active_params,
                    cost.flops
                );
                records.push(record);
            }
            if let Some(path) = &json {
                for r in &records {
                    RunLog::append(path, r)?;
                }
                println!("{} record(s) appended to {}", records.len(), path.display());
            }
            Ok(())
        }
        LmAction::Generate {
            prompt,
            max_new,
            sampling,
            top_k,
            temperature,
            lookahead,
            beam,
            budget,
            cached,
            seed,
            checkpoint: weights,
            arch,
            policy,
            key,
            grant,
            runtime: _,
        } => {
            let device = lm_device;
            B::seed(&device, seed);

            let config = lm_config_for(&arch, weights.as_ref())?;
            let mut model = LanguageModel::<B>::new(&config, &device)?;
            if let Some(path) = &weights {
                model = checkpoint::load::<B, _>(model, path, &device)?;
                println!("loaded {}", path.display());
            }
            let tokenizer = ByteTokenizer::new();

            if let Some(policy_path) = &policy {
                use diffusionblocks::policy::{gated_generate, Decision, Grant, Key, Policy};
                let policy = Policy::read(policy_path)?;
                let grants: Vec<Grant> = grant
                    .iter()
                    .map(|p| Grant::read(p))
                    .collect::<Result<_>>()?;
                let approved = match (&key, grants.is_empty()) {
                    (Some(key_path), false) => {
                        let key = Key::read(key_path)?;
                        let approved =
                            policy.approved_scopes(&key, &grants, checkpoint::unix_now());
                        for g in &grants {
                            match g.verify(&key, &policy, checkpoint::unix_now()) {
                                Ok(()) => println!(
                                    "grant {}: valid for {}",
                                    g.approval.id,
                                    g.approval.scopes.join(",")
                                ),
                                Err(err) => println!("grant {}: rejected: {err}", g.approval.id),
                            }
                        }
                        approved
                    }
                    (None, false) => anyhow::bail!("--grant needs --key to verify against"),
                    _ => Vec::new(),
                };
                let sampling = Sampling::parse(&sampling, top_k, temperature)?;
                let mut rng = StdRng::seed_from_u64(seed);
                let outcome = gated_generate(&policy, &approved, &prompt, |sent| {
                    let ids = tokenizer.encode(sent);
                    let out = if cached {
                        model.generate_cached(&ids, max_new, &sampling, &mut rng, &device)
                    } else {
                        model.generate(&ids, max_new, &sampling, &mut rng, &device)
                    };
                    tokenizer.decode_lossy(&out[ids.len().min(out.len())..])
                });
                match &outcome.prompt_decision {
                    Decision::Refuse { blocker, scope, .. } => {
                        println!(
                            "refused before any forward pass: blocker {blocker} (scope {scope})"
                        )
                    }
                    Decision::Allow { approved } if !approved.is_empty() => {
                        println!("approved scopes lifted: {}", approved.join(","))
                    }
                    Decision::Allow { .. } => println!("no blocker fired on the prompt"),
                }
                if let Some(Decision::Refuse { blocker, .. }) = &outcome.output_decision {
                    println!("output replaced: blocker {blocker} fired on what the model produced");
                }
                println!("model called: {}", outcome.model_called);
                println!("---\n{}\n---", outcome.text);
                return Ok(());
            }

            let ids = tokenizer.encode(&prompt);

            println!("{} vocab={}", config.describe(), config.vocab_size);

            let started = std::time::Instant::now();
            let out = if lookahead > 0 {
                let budget = Budget {
                    max_evaluations: budget,
                    max_depth: lookahead,
                    beam_width: beam,
                };
                let (out, stats) = model.generate_lookahead(&ids, max_new, top_k, budget, &device);
                println!(
                    "lookahead: {} tokens | {:.2} forward passes/token | mean depth {:.2} | {}",
                    stats.committed,
                    stats.calls_per_token(),
                    stats.mean_depth(),
                    if stats.budget_exhausted {
                        "budget cut at least one search short"
                    } else {
                        "every search completed inside its budget"
                    }
                );
                out
            } else {
                let mut rng = StdRng::seed_from_u64(seed);
                let sampling = Sampling::parse(&sampling, top_k, temperature)?;
                if cached {
                    model.generate_cached(&ids, max_new, &sampling, &mut rng, &device)
                } else {
                    model.generate(&ids, max_new, &sampling, &mut rng, &device)
                }
            };
            let elapsed = started.elapsed();

            println!(
                "decoder={} | {} tokens in {}",
                if lookahead > 0 {
                    "lookahead"
                } else if cached {
                    "greedy+kv-cache"
                } else {
                    "greedy"
                },
                out.len() - ids.len(),
                format_duration(elapsed)
            );
            println!("---\n{}\n---", tokenizer.decode_lossy(&out));
            if weights.is_none() {
                println!(
                    "Weights are random, so the text is noise. What this run shows is\n\
                     that the decoding paths agree and what each one costs."
                );
            }
            Ok(())
        }
    }
}

/// The configurations one `lm bench` axis compares, with the loss-free bias
/// rate each trains under (roadmap 28.6, extended per axis in Phase 25).
fn lm_bench_variants(axis: &str) -> Result<Vec<(String, LmConfig, f32)>> {
    use diffusionblocks::vit::MoeTrunkConfig;
    let tiny = LmConfig::tiny();
    let layers = tiny.num_layers;
    let moe = MoeTrunkConfig {
        num_experts: 3,
        top_k: 1,
        every_n_layers: 2,
        z_level: 1e-3,
        balance_bias: false,
    };
    let schedule = |text: &str| AttentionSchedule::parse(text, layers, 8, 4);
    Ok(match axis {
        "trunk" => vec![
            ("dense".into(), tiny.clone(), 0.0),
            (
                "moe".into(),
                LmConfig {
                    moe: Some(moe),
                    ..tiny.clone()
                },
                0.0,
            ),
            (
                "moe+bias".into(),
                LmConfig {
                    moe: Some(MoeTrunkConfig {
                        balance_bias: true,
                        ..moe
                    }),
                    ..tiny.clone()
                },
                1e-3,
            ),
        ],
        "attention" => [
            "dense",
            "3:1",
            "sliding8",
            "retrieval4",
            "linear",
            "learned",
        ]
        .iter()
        .map(|name| {
            Ok((
                name.to_string(),
                tiny.clone().with_attention(schedule(name)?),
                0.0,
            ))
        })
        .collect::<Result<Vec<_>>>()?,
        "positions" => [
            PositionKind::Learned,
            PositionKind::Rotary,
            PositionKind::None,
        ]
        .iter()
        .map(|kind| {
            (
                kind.name().to_string(),
                tiny.clone().with_positions(*kind),
                0.0,
            )
        })
        .collect(),
        "routing" => vec![
            (
                "moe".into(),
                LmConfig {
                    moe: Some(moe),
                    ..tiny.clone()
                },
                0.0,
            ),
            (
                "moe+state8".into(),
                LmConfig {
                    moe: Some(moe),
                    ..tiny.clone()
                }
                .with_routing_state(8),
                0.0,
            ),
        ],
        other => anyhow::bail!(
            "unknown bench axis '{other}' (expected trunk|attention|positions|routing)"
        ),
    })
}

fn cmd_experts(action: ExpertsAction) -> Result<()> {
    match action {
        ExpertsAction::Init {
            out,
            boxes,
            top_box,
            top_expert,
        } => {
            let parsed: Result<Vec<BoxSpec>> = boxes
                .iter()
                .map(|entry| {
                    let (name, experts) = entry.split_once(':').ok_or_else(|| {
                        anyhow::anyhow!("expected <box>:<expert>,<expert>, got '{entry}'")
                    })?;
                    let experts: Vec<ExpertSpec> = experts
                        .split(',')
                        .filter(|e| !e.is_empty())
                        .map(|e| ExpertSpec::new(format!("{name}/{e}"), e).with_tags(&[e]))
                        .collect();
                    anyhow::ensure!(
                        !experts.is_empty(),
                        "box '{name}' needs at least one expert"
                    );
                    Ok(BoxSpec::new(name, name, experts))
                })
                .collect();

            let spec = MosmeSpec {
                boxes: parsed?,
                top_box,
                top_expert,
                route_on_tokens: true,
                balance: Default::default(),
            };
            spec.write(&out)?;
            println!(
                "wrote {} ({} boxes, {} experts)",
                out.display(),
                spec.boxes.len(),
                spec.num_experts()
            );
            Ok(())
        }
        ExpertsAction::List { index } => {
            // An index is the richer document; fall back to a bare spec so the
            // command is useful before anything has been trained.
            match ExpertIndex::read(&index) {
                Ok(index) => print!("{}", index.render()),
                Err(index_err) => {
                    let spec = MosmeSpec::read(&index).with_context(|| {
                        format!(
                            "{} is neither an expert index ({index_err:#}) nor a spec",
                            index.display()
                        )
                    })?;
                    println!(
                        "spec (untrained): {} boxes, {} experts, top_box={} top_expert={}",
                        spec.boxes.len(),
                        spec.num_experts(),
                        spec.top_box,
                        spec.top_expert
                    );
                    for b in &spec.boxes {
                        println!("\n[{}] {}", b.id, b.label);
                        for e in &b.experts {
                            println!(
                                "  {:<24} {:<9} {}",
                                e.id,
                                if e.enabled { "enabled" } else { "disabled" },
                                e.tags.join(",")
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        ExpertsAction::Validate { index } => {
            let index = ExpertIndex::read(&index)?;
            println!(
                "valid: {} boxes, {} experts, site={}",
                index.num_boxes(),
                index.num_experts(),
                index.site.name()
            );
            Ok(())
        }
    }
}

// ------------------------------------------------------- coding student --

/// The training backend for `dblocks student train`: CPU autodiff, the same
/// backend the `lm` trainer defaults to.
type StudentTrain = burn::backend::Autodiff<Eval>;

/// The language flag: a shipped specialist, or `unknown` for unlabeled data
/// (the fallback SFT corpus). Parsed here so the error names the flag.
fn parse_student_language(text: &str) -> Result<Option<diffusionblocks::student::CodeLanguage>> {
    use diffusionblocks::student::CodeLanguage;
    if text.trim().eq_ignore_ascii_case("unknown") {
        return Ok(None);
    }
    CodeLanguage::parse(text).map(Some)
}

/// Read a StudentConfig sidecar, or the default config. A missing file is an
/// error naming it, not a silent fall back to defaults.
fn load_student_config(path: Option<&Path>) -> Result<diffusionblocks::student::StudentConfig> {
    use diffusionblocks::student::StudentConfig;
    let config = match path {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read student config {}", path.display()))?;
            serde_json::from_str(&text)
                .with_context(|| format!("parse student config {}", path.display()))?
        }
        None => StudentConfig::default(),
    };
    config.validate()?;
    Ok(config)
}

fn cmd_student(action: StudentAction) -> Result<()> {
    match action {
        StudentAction::Info { config } => cmd_student_info(config.as_deref()),
        StudentAction::Route {
            language_prompt,
            checkpoint,
            config,
        } => cmd_student_route(&language_prompt, checkpoint.as_deref(), config.as_deref()),
        StudentAction::Train { args } => cmd_student_train(&args),
        StudentAction::Think {
            prompt,
            depths,
            checkpoint,
            config,
            top_k,
        } => cmd_student_think(
            &prompt,
            &depths,
            checkpoint.as_deref(),
            config.as_deref(),
            top_k,
        ),
    }
}

fn cmd_student_info(config_path: Option<&Path>) -> Result<()> {
    use diffusionblocks::student::CodeLanguage;
    let config = load_student_config(config_path)?;
    let cost = config.cost()?;
    let spec = config.mosme_spec()?;
    println!(
        "student config{}",
        config_path.map_or(" (default)".to_string(), |p| format!(
            " from {}",
            p.display()
        ))
    );
    println!("{}", serde_json::to_string_pretty(&config)?);
    println!(
        "cost: {:.2}M active / {:.2}M total params ({:.1}% resident experts), {:.2} MFLOPs per token at context {}",
        cost.active_params as f64 / 1e6,
        cost.total_params as f64 / 1e6,
        100.0 * cost.resident_expert_params as f64 / cost.total_params as f64,
        cost.trunk_flops_per_token / 1e6,
        config.context,
    );
    println!(
        "  (geometric stream {:.2}M, probe router {:.2}M)",
        cost.geom_stream_params as f64 / 1e6,
        cost.probe_router_params as f64 / 1e6
    );
    println!(
        "box registry (top_box={}, top_expert={}):",
        spec.top_box, spec.top_expert
    );
    for (idx, language) in CodeLanguage::ALL.iter().enumerate() {
        let entry = &spec.boxes[idx];
        println!(
            "  [{}] {:<14} {} ({} micro experts)",
            idx,
            language.box_id(),
            entry.label,
            entry.experts.len()
        );
    }
    println!("checkpoints: {DEFAULT_STUDENT_OUT}");
    Ok(())
}

fn cmd_student_route(
    prompt: &str,
    checkpoint_path: Option<&Path>,
    config_path: Option<&Path>,
) -> Result<()> {
    use diffusionblocks::student::{CodeLanguage, GeometricStudent};
    let config = load_student_config(config_path)?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <Eval as burn::tensor::backend::Backend>::seed(&device, 0);
    let mut model = GeometricStudent::<Eval>::new(&config, &device)?;
    diffusionblocks::tensor_ext::force_initialization(&model);
    match checkpoint_path {
        Some(path) => {
            model = checkpoint::load::<Eval, _>(model, path, &device)?;
            println!("checkpoint: {}", path.display());
        }
        None => println!(
            "no --checkpoint: fresh weights, so the traffic below is the untrained baseline (chance is 1/6)"
        ),
    }
    let ids = ByteTokenizer::new().encode_document(prompt);
    let n = ids.len();
    let tokens = burn::tensor::Tensor::<Eval, 1, burn::tensor::Int>::from_ints(
        ids.iter()
            .map(|t| i64::from(*t))
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    )
    .reshape([1, n]);
    let routing = model.route_languages(tokens)?;
    let shown: String = prompt.chars().take(60).collect();
    println!("routing '{shown}' ({n} tokens):");
    let mut counts = vec![0usize; CodeLanguage::ALL.len()];
    for top in &routing.top_language {
        counts[top.index()] += 1;
    }
    let best = routing
        .traffic
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i);
    for language in CodeLanguage::ALL {
        let idx = language.index();
        let marker = if Some(idx) == best {
            "  <- top box"
        } else {
            ""
        };
        println!(
            "  {:<8} {:>6.2}%  top-1 on {}/{} tokens{marker}",
            language.name(),
            100.0 * routing.traffic[idx],
            counts[idx],
            n
        );
    }
    Ok(())
}

fn cmd_student_think(
    prompt: &str,
    depths: &[usize],
    checkpoint_path: Option<&Path>,
    config_path: Option<&Path>,
    top_k: usize,
) -> Result<()> {
    use diffusionblocks::student::GeometricStudent;
    use diffusionblocks::vecread::VecReader;
    let config = load_student_config(config_path)?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <Eval as burn::tensor::backend::Backend>::seed(&device, 0);
    let mut model = GeometricStudent::<Eval>::new(&config, &device)?;
    diffusionblocks::tensor_ext::force_initialization(&model);
    match checkpoint_path {
        Some(path) => {
            anyhow::ensure!(
                path.try_exists()
                    .with_context(|| format!("stat {}", path.display()))?,
                "student checkpoint {} does not exist; pass --checkpoint <path> from a \
                 `dblocks student train` run (default output {DEFAULT_STUDENT_OUT})",
                path.display()
            );
            model = checkpoint::load::<Eval, _>(model, path, &device)?;
            println!("checkpoint: {}", path.display());
        }
        None => println!(
            "no --checkpoint: fresh weights, so the decoded tokens below are the untrained baseline"
        ),
    }
    let ids = ByteTokenizer::new().encode_document(prompt);
    let n = ids.len();
    let tokens = burn::tensor::Tensor::<Eval, 1, burn::tensor::Int>::from_ints(
        ids.iter()
            .map(|t| i64::from(*t))
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    )
    .reshape([1, n]);
    let reader = VecReader {
        top_k,
        ..VecReader::default()
    };
    let trace = reader.read_student_thinking(&model, tokens, depths)?;
    println!("{}", trace.render());
    Ok(())
}

fn cmd_student_train(args: &StudentTrainArgs) -> Result<()> {
    use diffusionblocks::student::{
        train_language_specialist, GeometricStudent, LanguageBatch, SpecialistTrainConfig,
    };
    let language = parse_student_language(&args.language)?;
    let config = load_student_config(args.config.as_deref())?;
    let traces = &args.traces;
    let out_dir = &args.out_dir;
    anyhow::ensure!(
        traces
            .try_exists()
            .with_context(|| format!("stat {}", traces.display()))?,
        "traces file {} does not exist; pass --traces <path> (the fallback SFT corpus is {FALLBACK_SFT}, loaded with language=unknown)",
        traces.display()
    );
    let resume_path = match args.resume.as_deref() {
        None => None,
        Some("") => Some(
            checkpoint::latest_in_dir(out_dir, "student")?.ok_or_else(|| {
                anyhow::anyhow!(
                    "--resume without a value needs a checkpoint in {}; none found",
                    out_dir.display()
                )
            })?,
        ),
        Some(path) => {
            let path = PathBuf::from(path);
            anyhow::ensure!(
                path.try_exists()
                    .with_context(|| format!("stat {}", path.display()))?,
                "resume checkpoint {} does not exist",
                path.display()
            );
            Some(path)
        }
    };
    let data = LanguageBatch::load(traces, language, config.context, args.seed)?;
    println!(
        "traces: {} ({} examples kept of {} lines, {} skipped with no supervised token; filter {})",
        traces.display(),
        data.len(),
        data.lines_read(),
        data.skipped_no_target(),
        language.map_or("none".to_string(), |l| l.name().to_string())
    );
    // The gate's verdict is printed whether or not it dropped anything. A
    // silent zero is indistinguishable from a gate that did not run, and
    // "nothing was removed" is only a claim worth making when something
    // checked.
    let scan = data.cheat_scan();
    if scan.dropped == 0 {
        println!("cheat gate: {} row(s) checked, none dropped", scan.scanned);
    } else {
        let reasons = scan
            .dropped_by_class
            .iter()
            .map(|(class, n)| format!("{class}:{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "cheat gate: {} of {} row(s) dropped ({reasons})",
            scan.dropped, scan.scanned
        );
    }
    if language.is_none() {
        println!("language=unknown: the routing auxiliary is off (no box to supervise)");
    }
    let device: <StudentTrain as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <StudentTrain as burn::tensor::backend::Backend>::seed(&device, args.seed);
    let model = GeometricStudent::<StudentTrain>::new(&config, &device)?;
    diffusionblocks::tensor_ext::force_initialization(&model);
    let train_config = SpecialistTrainConfig {
        steps: args.steps,
        batch_size: args.batch_size,
        lr: args.lr,
        clip_norm: args.clip_norm,
        router_aux_weight: args.router_aux_weight,
        seed: args.seed,
        log_every: args.log_every,
        out_dir: Some(out_dir.clone()),
        checkpoint_every: args.checkpoint_every,
        resume: resume_path,
        language,
        ..SpecialistTrainConfig::default()
    };
    let (_model, report) = train_language_specialist(model, &data, &train_config, &device)?;
    println!(
        "done: {} steps ({} skipped, {} clipped) in {:.1}s | loss {:.4} -> {:.4} (ce {:.4} -> {:.4})",
        report.steps_taken,
        report.steps_skipped,
        report.steps_clipped,
        report.elapsed_secs,
        report.first_loss,
        report.last_loss,
        report.first_ce,
        report.last_ce
    );
    if report.last_router_aux.is_finite() {
        println!(
            "router-which-knows: aux {:.4} -> {:.4} | p(lang) {:.3} -> {:.3} | top1 {:.3} -> {:.3} (chance {:.3})",
            report.first_router_aux,
            report.last_router_aux,
            report.first_target_traffic,
            report.last_target_traffic,
            report.first_target_top1,
            report.last_target_top1,
            1.0 / diffusionblocks::student::CodeLanguage::ALL.len() as f32
        );
    }
    if let Some(path) = &report.checkpoint {
        println!("checkpoint: {}", path.display());
    }
    Ok(())
}

fn cmd_sigmas(num_blocks: usize, gamma: f64) -> Result<()> {
    let sampler = sigma::DblockSigmaSampler::new(num_blocks, gamma);
    println!("block boundaries (ascending):");
    for (i, s) in sampler.block_sigmas.iter().enumerate() {
        println!("  [{i}] {s:.6}");
    }
    println!("\nblock windows (block 0 is the noisiest):");
    for b in 0..num_blocks {
        let (lo, hi) = sigma::block_window(&sampler.block_sigmas, b);
        let (elo, ehi) = sampler.extended_window(b);
        println!("  block {b}: ({lo:.6}, {hi:.6}]  extended [{elo:.6}, {ehi:.6}]");
    }
    Ok(())
}

#[derive(clap::Subcommand, Clone, Debug)]
enum CheatAction {
    /// Scan a tree for suppressions, test evasion and gate tampering. Exits
    /// non-zero on any finding; findings already accepted in the repository
    /// baseline are not reported, only the delta.
    Scan {
        /// Tree to scan. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Also print findings as JSON.
        #[arg(long)]
        json: bool,
        /// Print every finding, not just the most severe.
        #[arg(long)]
        all: bool,
    },
    /// Calibrate the language-mixing rule from the scan tallies a queue log
    /// recorded, and write the profile the evidence justifies.
    Adapt {
        /// The Merkle-chained queue log. Verified before it is read.
        #[arg(long, default_value = "logs/queue.jsonl")]
        log: PathBuf,
        /// Strike a declared/scored pair once it has fired on this fraction of
        /// its declared language. 0.5 by default: a rule that fires on half of
        /// what it is shown is a constant, not a detector.
        #[arg(long, default_value_t = 0.5)]
        max_fire_rate: f64,
        /// Where to write the profile. Required.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Load a profile and report what it changes. A profile naming a language
    /// the parser does not know, or one whose thresholds are both zero, is
    /// refused rather than described.
    Policy {
        /// The profile `adapt --out` wrote.
        #[arg(long)]
        profile: Option<PathBuf>,
    },
    /// The learned second-opinion filter: score a snippet on stdin with a
    /// trained model. A pattern finding outranks the score either way.
    Ml {
        /// The trained model.
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// The outcome-grounded adaptation loop: mine the scanner's misses and
    /// false alarms from a labeled corpus, propose tightening edits, and accept
    /// one only if a rerun of the *same* corpus shows it helps.
    ///
    /// The scanner's rules are frozen; only its operating point moves. Every
    /// edit this loop can make is a tightening one — see
    /// [`diffusionblocks::cheatharness`].
    Harness {
        #[command(subcommand)]
        action: CheatHarnessAction,
    },
}

/// The adaptation loop's steps.
#[derive(Subcommand, Clone, Debug)]
enum CheatHarnessAction {
    /// Run one round: mine the packet, propose the candidate group, and score
    /// every candidate by rerun. Nothing is committed.
    Round {
        /// Labeled corpus, one `{"code","lang","cheat"}` JSON object per line.
        #[arg(long, default_value = "logs/cheat-corpus.jsonl")]
        corpus: PathBuf,
        /// The chain of accepted profiles.
        #[arg(long, default_value = "logs/cheat-harness.json")]
        registry: PathBuf,
        /// Candidates to propose per round.
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        /// Also write the round as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Take the best accepted candidate from a round and append it to the chain.
    /// Refuses when nothing was accepted, so an empty round cannot quietly
    /// advance the version.
    Commit {
        #[arg(long, default_value = "logs/cheat-corpus.jsonl")]
        corpus: PathBuf,
        #[arg(long, default_value = "logs/cheat-harness.json")]
        registry: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        /// Recorded on the entry next to the measured reward, so a profile is
        /// never in force without a statement of what it was measured against.
        #[arg(long)]
        provenance: String,
    },
    /// Withdraw the most recent accepted profile.
    Rollback {
        #[arg(long, default_value = "logs/cheat-harness.json")]
        registry: PathBuf,
    },
    /// Report the profile in force and the chain behind it.
    Status {
        #[arg(long, default_value = "logs/cheat-harness.json")]
        registry: PathBuf,
    },
}

/// Load the chain, or start one when the file does not exist yet.
fn cheat_harness_registry(path: &Path) -> Result<diffusionblocks::cheatharness::Registry> {
    if path.exists() {
        return diffusionblocks::cheatharness::Registry::load(path);
    }
    Ok(diffusionblocks::cheatharness::Registry::default_registry())
}

fn cmd_cheat_harness_round(
    corpus: &Path,
    registry_path: &Path,
    candidates: usize,
    json: bool,
) -> Result<()> {
    let items = diffusionblocks::cheatharness::load_corpus(corpus)?;
    let registry = cheat_harness_registry(registry_path)?;
    let baseline = registry.current().clone();
    let round = diffusionblocks::cheatharness::run_round(&items, &registry, candidates)?;

    if json {
        println!("{}", diffusionblocks::cheatharness::round_json(&round)?);
    }
    print!(
        "{}",
        diffusionblocks::cheatharness::render_round(&round, &baseline)
    );
    match round.best() {
        Some(c) => println!(
            "best: {} (version {} would follow)",
            c.summary(&baseline),
            registry.entries.len()
        ),
        None => {
            println!("best: none accepted — the corpus did not show a tightening edit that helps")
        }
    }
    Ok(())
}

fn cmd_cheat_harness_commit(
    corpus: &Path,
    registry_path: &Path,
    candidates: usize,
    provenance: &str,
) -> Result<()> {
    let items = diffusionblocks::cheatharness::load_corpus(corpus)?;
    let mut registry = cheat_harness_registry(registry_path)?;
    let baseline = registry.current().clone();
    let round = diffusionblocks::cheatharness::run_round(&items, &registry, candidates)?;

    let best = round.best().ok_or_else(|| {
        anyhow::anyhow!(
            "no candidate was accepted, so there is nothing to commit; the round's refusals are \
             the answer here — run `dblocks cheat harness round` to read them"
        )
    })?;
    // `commit` is the only writer, so the entry it appended is the last one.
    // The borrow it returns is dropped here: `save` needs `&registry`, and
    // holding the reference across it would not compile.
    registry.commit(&baseline, best, provenance)?;
    registry.save(registry_path)?;
    let entry = registry
        .entries
        .last()
        .ok_or_else(|| anyhow::anyhow!("committed an entry, but the chain reports empty"))?;
    // `commit` only lands a profile the acceptance test passed, and that test
    // requires a positive delta, so this prints a number with no sign.
    println!(
        "committed version {}: {} (delta +{}, caught {}/{} -> {}/{}, false-drops {} -> {})",
        entry.version,
        entry.summary,
        entry.reward.delta(),
        entry.reward.baseline.caught,
        entry.reward.baseline.cheats,
        entry.reward.candidate.caught,
        entry.reward.candidate.cheats,
        entry.reward.baseline.false_drops,
        entry.reward.candidate.false_drops,
    );
    Ok(())
}

fn cmd_cheat_harness_rollback(registry_path: &Path) -> Result<()> {
    let mut registry = cheat_harness_registry(registry_path)?;
    let entry = registry.rollback()?;
    registry.save(registry_path)?;
    // The chain's version, not `HarnessProfile::version` — that field is the
    // schema version, and printing it here would report the schema as though
    // it were the position in the chain.
    let now = registry.entries.last().map_or(0, |e| e.version);
    println!(
        "withdrew version {}: {}; version {now} is in force",
        entry.version, entry.summary
    );
    Ok(())
}

fn cmd_cheat_harness_status(registry_path: &Path) -> Result<()> {
    let registry = cheat_harness_registry(registry_path)?;
    let current = registry.current();
    println!(
        "in force: version {} — {}",
        registry.entries.last().map_or(0, |e| e.version),
        current.provenance
    );
    println!(
        "  mixing: min_score={} min_margin={} admitted_pairs={}",
        current.mixing.min_score,
        current.mixing.min_margin,
        current.mixing.admitted_pairs.len()
    );
    if current.gates.is_empty() {
        println!("  class gates: none (the scanner's own severities are in force)");
    } else {
        for (class, gate) in &current.gates {
            println!("  {class}: {:?}", gate);
        }
    }
    println!("chain: {} entries", registry.entries.len());
    for entry in &registry.entries {
        println!("  v{} {}", entry.version, entry.summary,);
    }
    Ok(())
}

fn cmd_cheat_scan(root: &Path, json: bool, all: bool) -> Result<()> {
    let candidate = cheat::load_from_dir(root)
        .with_context(|| format!("loading candidate tree {}", root.display()))?;
    let report = cheat::scan(&candidate);

    if json {
        println!("{}", report.to_json()?);
    }

    if report.is_clean() {
        println!("cheat: clean — no suppression, no evasion, no gate tampering");
        return Ok(());
    }

    let shown: Vec<&cheat::Finding> = if all {
        report.sorted()
    } else {
        report.sorted().into_iter().take(1).collect()
    };
    let hidden = report.findings.len() - shown.len();
    for f in shown {
        let loc = match (&f.path, f.line) {
            (p, Some(l)) if !p.is_empty() => format!("{p}:{l}"),
            (p, _) if !p.is_empty() => p.clone(),
            (_, Some(l)) => format!("line {l}"),
            _ => "-".to_string(),
        };
        println!(
            "{} [{}] {loc}\n    {}\n    because: {}",
            f.severity.as_str(),
            f.class.as_str(),
            f.evidence.trim(),
            f.because
        );
    }
    if hidden > 0 {
        println!("... and {hidden} more (pass --all to see them)");
    }
    let worst = report.worst().map(|s| s.as_str()).unwrap_or("unknown");
    anyhow::bail!(
        "{} finding(s), worst {worst}; the gate would be satisfied by defeating it, not by obeying it",
        report.findings.len()
    );
}

/// `dblocks cheat adapt` — read the scan tallies a queue log recorded and
/// write the language-mixing profile they justify.
///
/// The chain is verified before it is read, so the profile is derived from
/// counts that cannot have been edited afterwards, and the rule it applies can
/// only strike pairs: it cannot add a finding the filter did not already
/// make. The output says which pairs were struck and at what rate, because a
/// profile that changes what gets dropped has to be readable by whoever the
/// next drop count surprises.
fn cmd_cheat_adapt(log: &Path, max_fire_rate: f64, out: Option<&Path>) -> Result<()> {
    let Some(out) = out else {
        anyhow::bail!(
            "--out is required: calibration writes a profile that changes which rows are \
             dropped, so the old behaviour has to stay recoverable by not overwriting it"
        );
    };
    let tally = nina::mixing_tally_from_chain(log)?;
    println!(
        "calibrating from {}: {} scanned document(s), {} firing pair(s)",
        log.display(),
        tally.scanned_total(),
        tally.fired.len()
    );
    let adaptation = cheat::calibrate_mixing(&tally, max_fire_rate)?;
    for pair in &adaptation.pairs {
        println!(
            "  {:<24} {:>6}/{:<6} {:>7.2}%  {}",
            pair.pair,
            pair.fired,
            pair.scanned,
            100.0 * pair.fire_rate,
            if pair.removed { "struck" } else { "kept" }
        );
    }
    adaptation.policy.save(out)?;
    println!(
        "\nwrote {} — {} pair(s) struck at max_fire_rate {max_fire_rate}\n{}",
        out.display(),
        adaptation.policy.admitted_pairs.len(),
        adaptation.policy.provenance
    );
    Ok(())
}

/// `dblocks cheat policy` — load a profile and report what it changes.
///
/// Loading is the check: a profile naming a language the parser does not know,
/// or one whose thresholds are both zero, is refused rather than described.
fn cmd_cheat_policy(profile: Option<&Path>) -> Result<()> {
    let Some(path) = profile else {
        anyhow::bail!("--profile is required: pass the file `dblocks cheat adapt --out` wrote")
    };
    let policy = cheat::MixingPolicy::load(path)?;
    println!("{path:?}");
    println!("  min_score   {}", policy.min_score);
    println!("  min_margin  {}", policy.min_margin);
    println!("  struck      {} pair(s)", policy.admitted_pairs.len());
    for (declared, scored) in &policy.admitted_pairs {
        println!("    {declared} -> {scored}");
    }
    println!("  provenance  {}", policy.provenance);
    Ok(())
}

/// `dblocks cheat ml` — the learned second-opinion filter.
///
/// With `--model`, score one snippet read from stdin and print the model's
/// answer, so a score can be inspected against the text that produced it.
/// Without one, report what the layer is and where a model would live, rather
/// than training one from nothing: the labels are the pattern scanner's
/// verdicts, and a run that silently invented them would be the circularity
/// the module documents.
fn cmd_cheat_ml(model: Option<&Path>) -> Result<()> {
    let Some(path) = model else {
        println!("cheat ml: the learned layer generalizes cheat.rs verdicts to snippets whose");
        println!(
            "patterns do not fire. It does not invent ground truth (see mlcheat::LABEL_SOURCE)."
        );
        println!();
        println!("usage: dblocks cheat ml --model <ml-cheat-model.json> < snippet");
        println!("       the trained model is loaded and scores stdin as one snippet");
        return Ok(());
    };
    let model = mlcheat::MlCheatModel::load(path)?;
    let mut text = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)
        .with_context(|| format!("read snippet from stdin for {}", path.display()))?;
    let lang = cheat::SnippetLanguage::Other;
    let score = model.score(&text, lang);
    println!("score        {score:.6}");
    println!("declared     {}", lang.as_str());
    println!("model        {}", path.display());
    println!("labels       {}", mlcheat::LABEL_SOURCE);
    println!();
    println!(
        "A score is a second opinion, not a verdict: a pattern finding at quarantine grade or \
         above still rejects the trace whatever this number says."
    );
    Ok(())
}

fn cmd_verify(group: Option<&str>) -> Result<()> {
    let mut report = verify::run_all();
    if let Some(group) = group {
        report.certificates.retain(|c| c.group == group);
        if report.certificates.is_empty() {
            anyhow::bail!("no certificates in group '{group}'");
        }
    }
    print!("{}", report.render());
    if !report.passed() {
        anyhow::bail!("{} certificate(s) failed", report.failures().len());
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn cmd_train(command: Command) -> Result<()> {
    let Command::Train {
        dataset,
        data_dir,
        dataset_weights,
        mix,
        streaming,
        objective,
        teacher,
        teacher_weights,
        image_size,
        num_labels,
        num_blocks,
        gamma,
        batch_size,
        lr,
        weight_decay,
        steps,
        log_every,
        seed,
        out_dir,
        log_file,
        grad_checkpointing,
        async_save,
        resume,
        checkpoint_every,
        no_checks,
        no_preflight,
        verify_every,
        synthetic_negatives,
        negative_penalty,
        mosme_spec,
        mosme_every,
        index_out,
        lr_schedule,
        accumulate,
        clip_norm,
        ema_decay,
        normalize_block_loss,
        balance_schedule,
        balance_weight,
        balance_scope,
        bias_balance_rate,
        z_level,
        uncertainty,
        importance_bins,
        moe_every,
        moe_experts,
        moe_top_k,
    } = command
    else {
        anyhow::bail!("internal error: cmd_train dispatched with a command other than `train`");
    };

    let out_path = Path::new(&out_dir).to_path_buf();
    // `--resume` with no value means "the newest checkpoint in --out-dir".
    let resume = match resume {
        None => None,
        Some(value) if value.is_empty() => {
            let found = checkpoint::latest_in_dir(&out_path, "dblocks")?;
            if found.is_none() {
                println!("--resume: no checkpoint found in {out_dir}, starting fresh");
            }
            found
        }
        Some(value) => Some(PathBuf::from(value)),
    };

    let config = TrainConfig {
        image_size,
        num_labels,
        batch_size,
        num_blocks,
        gamma,
        lr,
        weight_decay,
        steps,
        log_every,
        seed,
        log_file: log_file.map(PathBuf::from),
        dataset: DatasetChoice::parse(&dataset[0], data_dir.first().cloned(), streaming)?,
        extra_datasets: dataset
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, name)| DatasetChoice::parse(name, data_dir.get(i).cloned(), streaming))
            .collect::<Result<Vec<_>>>()?,
        dataset_weights: parse_weights(&dataset_weights)?,
        mix_mode: diffusionblocks::mix::MixMode::parse(&mix)?,
        objective: Objective::parse(&objective)?,
        checks: if no_checks {
            TrainingChecks::none()
        } else {
            TrainingChecks {
                preflight: !no_preflight,
                verify_every: (verify_every > 0).then_some(verify_every),
                ..TrainingChecks::default()
            }
        },
        resume,
        teacher: teacher.first().cloned(),
        extra_teachers: teacher.iter().skip(1).cloned().collect(),
        teacher_weights: parse_weights(&teacher_weights)?,
        moe: moe_every.map(|every| MoeTrunkConfig {
            num_experts: moe_experts,
            top_k: moe_top_k,
            every_n_layers: every,
            z_level,
            balance_bias: bias_balance_rate > 0.0,
        }),
        mosme: mosme_spec
            .as_deref()
            .map(MosmeSpec::read)
            .transpose()?
            .map(|mut spec| {
                // The CLI overrides what the index file says: a stored spec
                // records how a model was trained, and a sweep over this knob
                // should not require rewriting the file each time.
                spec.balance.z_level = z_level;
                MosmeTrunkConfig::new(spec)
                    .with_every_n_layers(mosme_every)
                    .with_balance_bias(bias_balance_rate > 0.0)
            }),
        lr_schedule: LrSchedule::parse(&lr_schedule, lr, steps)?,
        accumulate,
        clip_norm,
        ema_decay,
        normalize_block_loss,
        uncertainty,
        importance_bins,
        balance_schedule: Some(diffusionblocks::schedule::BalanceSchedule::parse(
            &balance_schedule,
            balance_weight,
            steps,
        )?),
        balance_scope: diffusionblocks::schedule::BalanceScope::parse(&balance_scope)?,
        bias_balance_rate,
        // The trainer writes the checkpoint itself so the training state lands
        // beside it. `--async-save` keeps the old weights-only background save.
        out_dir: (!async_save).then(|| out_path.clone()),
        checkpoint_every,
        synthetic_negatives,
        negative_penalty,
    };

    println!(
        "training: dataset={} objective={} blocks={num_blocks} steps={steps}",
        dataset.join("+"),
        config.objective.name()
    );

    let path = if grad_checkpointing {
        let (model, summary) = train::train_generic::<
            burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
        >(&config)?;
        match summary.checkpoint {
            Some(path) => path,
            None => checkpoint::save_content_addressed_async(model, out_path, "dblocks")
                .join()
                .map_err(|payload| {
                    anyhow::anyhow!(
                        "checkpoint save thread panicked: {}",
                        panic_message(&payload)
                    )
                })??,
        }
    } else {
        let (model, summary) = train::train(&config)?;
        println!(
            "done: {} steps in {:.1}s (mean loss {:.4}, {} rejected by a quality check, {:.1}% reject rate{})",
            summary.steps_taken,
            summary.elapsed_secs,
            summary.mean_loss,
            summary.steps_skipped,
            100.0 * summary.skip_rate(),
            if summary.steps_accumulated > 0 {
                format!(", {} micro-batches accumulated", summary.steps_accumulated)
            } else {
                String::new()
            }
        );
        if let Some(reason) = &summary.aborted {
            println!("run stopped early: {reason}");
        }
        if summary.steps_clipped > 0 {
            println!(
                "{} step(s) had their gradients rescaled by --clip-norm",
                summary.steps_clipped
            );
        }
        if summary.periodic_verifications > 0 {
            println!(
                "live model re-verified {} time(s) during the run",
                summary.periodic_verifications
            );
        }
        print!("\nper-block quality:\n{}", summary.health.render());
        match summary.checkpoint {
            Some(path) => path,
            None => {
                println!("note: --async-save writes the weights only, without a training state");
                checkpoint::save_content_addressed_async(model, out_path, "dblocks")
                    .join()
                    .map_err(|payload| {
                        anyhow::anyhow!(
                            "checkpoint save thread panicked: {}",
                            panic_message(&payload)
                        )
                    })??
            }
        }
    };
    println!("checkpoint saved: {}", path.display());

    // The manifest is written next to the checkpoint and keyed by its content
    // hash, so an inference engine can tell the two belong together.
    if let Some(trunk) = &config.mosme {
        let index_path = index_out.unwrap_or_else(|| path.with_extension("index.json"));
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let index = describe_experts(&trunk.spec, &config, &model_id)?;
        index.write(&index_path)?;
        println!("expert index saved: {}", index_path.display());
    }
    Ok(())
}

/// Build the manifest for a trained MoSME trunk.
///
/// Rebuilds the model shape from the config to read the live expert modules
/// back; the index records the *first* hierarchical layer, which is the one an
/// engine routes with.
fn describe_experts(spec: &MosmeSpec, config: &TrainConfig, model_id: &str) -> Result<ExpertIndex> {
    use diffusionblocks::mosme::{MosmeConfig, MosmeFeedForward};
    let vit = config.vit_config();
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let mosme = MosmeConfig::new(vit.hidden_size, vit.cond_hidden_size, spec.clone())
        .with_intermediate_size(vit.intermediate_size);
    let layer = MosmeFeedForward::<Eval>::new(&mosme, &device);
    layer.index(spec, model_id, "vit.layers.mlp", vit.cond_hidden_size)
}

fn parse_strategy(name: &str, k: usize) -> Result<Strategy> {
    Ok(match name {
        "sequential" => Strategy::Sequential,
        "parallel" => Strategy::Parallel { k },
        "hybrid" => Strategy::Hybrid {
            k,
            warmup_frac: 0.3,
        },
        "adaptive" => Strategy::Adaptive {
            k_max: k.max(1),
            conf_threshold: 0.9,
        },
        other => anyhow::bail!(
            "unknown strategy '{other}' (expected sequential|parallel|hybrid|adaptive)"
        ),
    })
}

fn parse_gates(name: &str, num_blocks: usize) -> Result<LayerGates> {
    Ok(match name {
        "lenient" => LayerGates::uniform(QualityGateConfig::lenient()),
        "strict" => LayerGates::uniform(QualityGateConfig::strict()),
        "tightening" => LayerGates::tightening(
            num_blocks,
            QualityGateConfig::lenient(),
            QualityGateConfig::strict(),
        ),
        other => anyhow::bail!("unknown gate '{other}' (expected lenient|strict|tightening)"),
    })
}

fn cmd_sample(command: Command) -> Result<()> {
    let Command::Sample {
        model: model_args,
        num_inference_steps,
        batch_size,
        solver,
        strategy,
        k,
        precision,
        precision_switch,
        gate,
        guidance,
        guidance_rescale,
        logit_norm,
        logit_tau,
        ensemble,
        planned,
        plan_depth,
        plan_beam,
        plan_budget,
    } = command
    else {
        anyhow::bail!("internal error: cmd_sample dispatched with a command other than `sample`");
    };

    let model = model_args.build(Some(num_inference_steps))?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();

    let mut rng = StdRng::seed_from_u64(model_args.seed);
    let mut dataset = SyntheticDataset::new(
        model_args.image_size,
        model_args.num_labels,
        batch_size,
        model_args.seed,
    );
    let batch = dataset.next_batch(&mut rng, &device)?;

    let coarse = Precision::parse(&precision)?;
    let config = MultiBlockConfig {
        strategy: Gated {
            inner: parse_strategy(&strategy, k)?,
            gate: parse_gates(&gate, model_args.num_blocks)?,
        },
        solver: SolverKind::parse(&solver)?,
        num_steps: Some(num_inference_steps),
        precision: if coarse == Precision::F32 {
            PrecisionPolicy::default()
        } else {
            PrecisionPolicy::mixed(coarse, precision_switch)
        },
        guidance: Guidance::new(guidance).with_rescale(guidance_rescale),
        logit_norm: LogitNorm::parse(&logit_norm, logit_tau)?,
    };

    // Three mutually exclusive paths, most specific first. Planning replaces
    // the schedule outright, so it cannot also be ensembled here without
    // silently deciding which of the two the user meant.
    let (logits, stats) = if planned {
        let planned_config = PlannedConfig {
            budget: Budget {
                max_evaluations: plan_budget,
                max_depth: plan_depth,
                beam_width: plan_beam,
            },
            solver: config.solver,
            max_steps: num_inference_steps.max(2),
            logit_norm: config.logit_norm,
            ..PlannedConfig::default()
        };
        let (logits, stats, trace) =
            model.sample_planned(&batch.pixel_values, &planned_config, &mut rng);
        println!(
            "planned: {} steps | mean lookahead depth {:.2} | {} evaluations | {} step(s) cut short by the budget",
            trace.steps.len(),
            trace.mean_depth(),
            trace.total_evaluations(),
            trace.budget_exhausted_steps
        );
        for (i, step) in trace.steps.iter().enumerate() {
            println!(
                "  step {i}: sigma -> {:.5} with a {}-block span",
                step.sigma, step.width
            );
        }
        if trace.forced_final_step {
            println!("  (the step cap bound before sigma_min; the last step was unplanned)");
        }
        println!(
            "  planning overhead: {:.0}% of executed layers",
            100.0 * stats.planning_overhead()
        );
        (logits, stats)
    } else if !ensemble.is_empty() {
        let kind = Ensemble::parse(&ensemble)?;
        // Solver diversity is the cheapest source of disagreement available:
        // the members share every weight and differ only in how they integrate.
        let members: Vec<MultiBlockConfig> = SolverKind::deterministic()
            .iter()
            .map(|s| MultiBlockConfig {
                solver: *s,
                ..config.clone()
            })
            .collect();
        println!(
            "ensemble={} over {} solvers: {}",
            kind.name(),
            members.len(),
            members
                .iter()
                .map(|m| m.solver.name())
                .collect::<Vec<_>>()
                .join(", ")
        );
        // `sample_ensemble` returns probabilities, so the normalization has
        // already been applied to each member's logits inside the members.
        model.sample_ensemble(&batch.pixel_values, &members, kind, &mut rng)
    } else {
        model.sample_multi_block(&batch.pixel_values, &config, &mut rng)
    };

    let preds: Vec<i64> = logits
        .argmax(1)
        .squeeze_dim::<1>(1)
        .into_data()
        .convert::<i64>()
        .iter()
        .collect();
    let truth: Vec<i64> = batch.labels.into_data().convert::<i64>().iter().collect();

    println!(
        "solver={} strategy={strategy} gate={gate} precision={}",
        config.solver.name(),
        coarse.name()
    );
    println!("schedule (descending): {:?}", model.inference_sigmas());
    println!(
        "model calls: {} | layers executed: {} | mean span: {:.2} | gated samples: {} | reduced-precision windows: {}",
        stats.model_calls,
        stats.layers_executed,
        stats.mean_span_width(),
        stats.gated_samples,
        stats.reduced_precision_windows
    );
    for block in 0..stats.ledger.num_blocks() {
        if stats.ledger.rejected(block) > 0 {
            println!(
                "  block {block}: {:.1}% of updates gated",
                100.0 * stats.ledger.rejection_rate(block)
            );
        }
    }

    println!("predicted vs true:");
    for (i, (p, t)) in preds.iter().zip(&truth).enumerate() {
        println!("  sample {i}: pred={p} true={t}");
    }
    // Untrained weights make this a plumbing check, not an accuracy measurement.
    let correct = preds.iter().zip(&truth).filter(|(a, b)| a == b).count();
    println!(
        "top-1 agreement with synthetic labels: {correct}/{}",
        truth.len()
    );
    Ok(())
}

fn cmd_bench(
    model_args: ModelArgs,
    num_inference_steps: usize,
    batch_size: usize,
    repeats: usize,
    warmup: usize,
    json: Option<PathBuf>,
) -> Result<()> {
    use diffusionblocks::experiment::{Record, RunLog};
    let bench_config = serde_json::json!({
        "image_size": model_args.image_size,
        "num_labels": model_args.num_labels,
        "num_hidden_layers": model_args.num_hidden_layers,
        "num_blocks": model_args.num_blocks,
        "checkpoint": model_args.checkpoint.as_ref().map(|p| p.display().to_string()),
        "num_inference_steps": num_inference_steps,
        "batch_size": batch_size,
        "repeats": repeats,
        "warmup": warmup,
    });
    let mut records: Vec<Record> = Vec::new();
    let model = model_args.build(Some(num_inference_steps))?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let mut rng = StdRng::seed_from_u64(model_args.seed);
    let mut dataset = SyntheticDataset::new(
        model_args.image_size,
        model_args.num_labels,
        batch_size,
        model_args.seed,
    );
    let batch = dataset.next_batch(&mut rng, &device)?;

    // A reference run everything else is compared against: sequential Euler is
    // the original DiffusionBlocks inference path.
    let reference_cfg = MultiBlockConfig {
        strategy: Gated::uniform(Strategy::Sequential, QualityGateConfig::lenient()),
        solver: SolverKind::Euler,
        num_steps: Some(num_inference_steps),
        precision: PrecisionPolicy::default(),
        guidance: Guidance::none(),
        logit_norm: LogitNorm::None,
    };
    let (reference_logits, _) =
        model.sample_multi_block(&batch.pixel_values, &reference_cfg, &mut rng);
    let reference: Vec<i64> = reference_logits
        .argmax(1)
        .squeeze_dim::<1>(1)
        .into_data()
        .convert::<i64>()
        .iter()
        .collect();

    println!(
        "{:<10} {:<12} {:>10} {:>12} {:>8} {:>10}",
        "solver", "strategy", "mean ms", "model calls", "layers", "agree"
    );
    println!("{}", "-".repeat(68));

    let strategies: Vec<(&str, Strategy)> = vec![
        ("sequential", Strategy::Sequential),
        ("parallel-2", Strategy::Parallel { k: 2 }),
        (
            "hybrid-2",
            Strategy::Hybrid {
                k: 2,
                warmup_frac: 0.3,
            },
        ),
        (
            "adaptive",
            Strategy::Adaptive {
                k_max: 3,
                conf_threshold: 0.9,
            },
        ),
    ];

    let mut profiler = Profiler::new();
    for kind in SolverKind::all() {
        for (label, strategy) in &strategies {
            let config = MultiBlockConfig {
                strategy: Gated::uniform(*strategy, QualityGateConfig::lenient()),
                solver: kind,
                num_steps: Some(num_inference_steps),
                precision: PrecisionPolicy::default(),
                guidance: Guidance::none(),
                logit_norm: LogitNorm::None,
            };

            let mut last = None;
            let scope = format!("{}/{label}", kind.name());
            let mut record = Record::new(
                format!("bench/{scope}"),
                "ms",
                bench_config.clone(),
                vec![model_args.seed],
            );
            for trial in 0..warmup + repeats.max(1) {
                let is_warmup = trial < warmup;
                let start = std::time::Instant::now();
                let out = model.sample_multi_block(&batch.pixel_values, &config, &mut rng);
                let elapsed = start.elapsed();
                record.push(elapsed.as_secs_f64() * 1e3, is_warmup);
                if !is_warmup {
                    profiler.record(&scope, elapsed);
                }
                last = Some(out);
            }

            let (logits, stats) =
                last.context("no repeat ran: --repeats and --warmup left nothing to measure")?;
            let preds: Vec<i64> = logits
                .argmax(1)
                .squeeze_dim::<1>(1)
                .into_data()
                .convert::<i64>()
                .iter()
                .collect();
            let agree = preds.iter().zip(&reference).filter(|(a, b)| a == b).count();
            record.extra = serde_json::json!({
                "model_calls": stats.model_calls,
                "layers_executed": stats.layers_executed,
                "agree": agree,
                "of": reference.len(),
            });
            records.push(record);

            let timing = profiler
                .stats(&scope)
                .with_context(|| format!("no timing recorded for scope {scope:?}"))?;
            println!(
                "{:<10} {:<12} {:>10} {:>12} {:>8} {:>9}/{}  ±{}",
                kind.name(),
                label,
                format_duration(timing.mean()),
                stats.model_calls,
                stats.layers_executed,
                agree,
                reference.len(),
                format_duration(timing.ci95_half_width())
            );
        }
    }
    println!(
        "(± is the half-width of the 95% t-interval over {} measured repeat(s))",
        repeats.max(1)
    );

    println!(
        "\nAgreement is measured against sequential Euler on the SAME weights.\n\
         With random weights it reports how much the discretization changes the\n\
         answer, not which solver is better -- that needs a trained model."
    );

    // Test-time compute scaling (roadmap 22.3). "Spend more at inference and
    // get more accuracy" is a claim, not a law: it holds up to a point and then
    // flattens. Measuring it is the only way to know where, for this model.
    let mut curve = ScalingCurve::new();
    for steps in [2usize, 4, 8] {
        let config = MultiBlockConfig {
            strategy: Gated::uniform(Strategy::Sequential, QualityGateConfig::lenient()),
            solver: SolverKind::DpmPlusPlus2M,
            num_steps: Some(steps),
            precision: PrecisionPolicy::default(),
            guidance: Guidance::none(),
            logit_norm: LogitNorm::None,
        };
        let (logits, stats) = model.sample_multi_block(&batch.pixel_values, &config, &mut rng);
        let acc = diffusionblocks::accuracy::accuracy(&logits, &batch.labels);
        let mut record = Record::new(
            format!("bench/scaling/sequential/steps={steps}"),
            "accuracy",
            bench_config.clone(),
            vec![model_args.seed],
        );
        record.push(acc, false);
        record.extra = serde_json::json!({ "model_calls": stats.model_calls, "layers_executed": stats.layers_executed });
        records.push(record);
        curve.push(ScalingPoint::new(
            format!("sequential/steps={steps}"),
            stats.model_calls,
            stats.layers_executed,
            acc,
        ));
    }
    for depth in [0usize, 1, 2] {
        let config = PlannedConfig {
            budget: Budget {
                max_evaluations: 48,
                max_depth: depth,
                beam_width: 3,
            },
            solver: SolverKind::Euler,
            max_steps: 6,
            ..PlannedConfig::default()
        };
        let (logits, stats, _) = model.sample_planned(&batch.pixel_values, &config, &mut rng);
        let acc = diffusionblocks::accuracy::accuracy(&logits, &batch.labels);
        let mut record = Record::new(
            format!("bench/scaling/planned/depth={depth}"),
            "accuracy",
            bench_config.clone(),
            vec![model_args.seed],
        );
        record.push(acc, false);
        record.extra = serde_json::json!({ "model_calls": stats.model_calls, "layers_executed": stats.layers_executed });
        records.push(record);
        curve.push(ScalingPoint::new(
            format!("planned/depth={depth}"),
            stats.model_calls,
            stats.layers_executed,
            acc,
        ));
    }

    println!("\nTest-time compute scaling (* marks the Pareto frontier):");
    print!("{}", curve.render());
    if let Some(path) = &json {
        for record in &records {
            RunLog::append(path, record)?;
        }
        println!(
            "\n{} experiment record(s) appended to {}",
            records.len(),
            path.display()
        );
    }
    for (label, rate) in curve.marginal_returns() {
        println!("  {label}: {:+.5} top-1 per extra layer", rate);
    }
    println!(
        "\nOn random weights top-1 is chance, so the frontier here demonstrates the\n\
         measurement, not a result. Run it on trained weights to size a budget."
    );
    Ok(())
}

fn parse_weights(text: &str) -> Result<Vec<f64>> {
    text.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|e| anyhow::anyhow!("weight {s:?}: {e}"))
        })
        .collect()
}

fn parse_seeds(text: &str) -> Result<Vec<u64>> {
    let seeds: Vec<u64> = text
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<u64>()
                .map_err(|e| anyhow::anyhow!("seed {s:?}: {e}"))
        })
        .collect::<Result<_>>()?;
    anyhow::ensure!(!seeds.is_empty(), "at least one seed is needed");
    Ok(seeds)
}

fn cmd_sweep(
    grid: &str,
    seeds: &str,
    steps: usize,
    dataset: &str,
    data_dir: Option<PathBuf>,
    batch_size: usize,
    json: Option<PathBuf>,
) -> Result<()> {
    use diffusionblocks::sweep::{self, Grid};
    let grid = Grid::parse(grid)?;
    let seeds = parse_seeds(seeds)?;
    let base = TrainConfig {
        steps,
        batch_size,
        log_every: steps.max(1),
        dataset: DatasetChoice::parse(dataset, data_dir, false)?,
        ..TrainConfig::default()
    };
    let cells = grid.cells();
    println!(
        "sweep: {} cell(s) x {} seed(s) x {steps} steps on {dataset}{}",
        cells.len(),
        seeds.len(),
        json.as_ref()
            .map(|p| format!(" -> {}", p.display()))
            .unwrap_or_default()
    );
    let records = sweep::run_grid(&base, &grid, &seeds, json.as_deref())?;
    print!("\n{}", sweep::render(&records));
    println!(
        "\nEvery cell differs from every other only in what the grid names; the trials are the\n\
         per-seed final losses and the interval is the 95% t-interval over seeds."
    );
    Ok(())
}

fn cmd_policy(action: PolicyAction) -> Result<()> {
    use diffusionblocks::policy::{starter, Applies, Blocker, Grant, Key, Policy};
    match action {
        PolicyAction::Init { out, key } => {
            let k = Key::generate();
            k.write(&key)?;
            let policy = starter(&k)?;
            policy.write(&out)?;
            println!(
                "policy {} ({} blockers, scopes {}) and key {} (id {}) written",
                out.display(),
                policy.blockers.len(),
                policy.scopes().join(","),
                key.display(),
                k.id()
            );
            Ok(())
        }
        PolicyAction::List { policy } => {
            let p = Policy::read(&policy)?;
            println!(
                "policy {} | key id {} | {} blocker(s) | {} revoked grant(s)",
                policy.display(),
                p.key_id,
                p.blockers.len(),
                p.revoked.len()
            );
            for b in &p.blockers {
                println!(
                    "- {} [{}] {:?}: {} pattern(s); refusal: {:?}",
                    b.id,
                    b.scope,
                    b.applies_to,
                    b.patterns.len(),
                    b.refusal
                );
                for pat in &b.patterns {
                    println!("    {pat}");
                }
            }
            for g in &p.revoked {
                println!("revoked: {g}");
            }
            Ok(())
        }
        PolicyAction::AddBlocker {
            policy,
            id,
            scope,
            pattern,
            applies_to,
            refusal,
            description,
        } => {
            let mut p = Policy::read(&policy)?;
            p.add_blocker(Blocker {
                id: id.clone(),
                scope,
                description,
                patterns: pattern,
                applies_to: Applies::parse(&applies_to)?,
                refusal,
            })?;
            p.write(&policy)?;
            println!("blocker {id} added to {}", policy.display());
            Ok(())
        }
        PolicyAction::RemoveBlocker { policy, id } => {
            let mut p = Policy::read(&policy)?;
            let removed = p.remove_blocker(&id)?;
            p.write(&policy)?;
            println!(
                "blocker {} [{}] removed from {}",
                removed.id,
                removed.scope,
                policy.display()
            );
            Ok(())
        }
        PolicyAction::Check {
            policy,
            prompt,
            key,
            grant,
        } => {
            let p = Policy::read(&policy)?;
            let grants: Vec<Grant> = grant
                .iter()
                .map(|g| Grant::read(g))
                .collect::<Result<_>>()?;
            let approved = match (&key, grants.is_empty()) {
                (Some(key_path), false) => {
                    p.approved_scopes(&Key::read(key_path)?, &grants, checkpoint::unix_now())
                }
                (None, false) => anyhow::bail!("--grant needs --key to verify against"),
                _ => Vec::new(),
            };
            let hits = p.hits(&prompt, false);
            if hits.is_empty() {
                println!("no blocker fires");
            }
            for h in &hits {
                println!(
                    "blocker {} [{}] fires at bytes {}..{}",
                    h.blocker, h.scope, h.start, h.end
                );
            }
            println!("decision: {:?}", p.decide_prompt(&prompt, &approved));
            Ok(())
        }
        PolicyAction::Revoke { policy, grant_id } => {
            let mut p = Policy::read(&policy)?;
            p.revoke(&grant_id);
            p.write(&policy)?;
            println!("grant {grant_id} revoked in {}", policy.display());
            Ok(())
        }
    }
}

fn cmd_approvals(action: ApprovalsAction) -> Result<()> {
    use diffusionblocks::policy::{Approval, Grant, Key, Policy};
    match action {
        ApprovalsAction::Issue {
            key,
            policy,
            id,
            scopes,
            expires,
            note,
            out,
        } => {
            let k = Key::read(&key)?;
            let p = Policy::read(&policy)?;
            anyhow::ensure!(
                p.key_id == k.id(),
                "key {} is not the policy's key ({})",
                k.id(),
                p.key_id
            );
            let approval = Approval {
                id: id.clone(),
                scopes: scopes
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                issued_unix: checkpoint::unix_now(),
                expires_unix: expires,
                note,
            };
            let unknown: Vec<&String> = approval
                .scopes
                .iter()
                .filter(|s| *s != "*" && !p.scopes().contains(s))
                .collect();
            if !unknown.is_empty() {
                println!(
                    "note: scope(s) {:?} have no blocker in this policy",
                    unknown
                );
            }
            let grant = Grant::issue(&k, approval)?;
            grant.write(&out)?;
            println!(
                "grant {id} for {} written to {} (expires {expires})",
                scopes,
                out.display()
            );
            Ok(())
        }
        ApprovalsAction::Verify { key, policy, grant } => {
            let g = Grant::read(&grant)?;
            match g.verify(
                &Key::read(&key)?,
                &Policy::read(&policy)?,
                checkpoint::unix_now(),
            ) {
                Ok(()) => {
                    println!(
                        "grant {} is valid for {} until {}",
                        g.approval.id,
                        g.approval.scopes.join(","),
                        g.approval.expires_unix
                    );
                    Ok(())
                }
                Err(err) => anyhow::bail!("{err}"),
            }
        }
    }
}

fn cmd_merge(
    model_args: ModelArgs,
    input: Vec<PathBuf>,
    weights: &str,
    out: PathBuf,
) -> Result<()> {
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let template = ModelArgs {
        checkpoint: None,
        ..model_args.clone()
    }
    .build(None)?;
    let weights = if weights.trim().is_empty() {
        vec![1.0; input.len()]
    } else {
        parse_weights(weights)?
    };
    let (merged, parents) =
        diffusionblocks::merge::merge_checkpoints::<Eval, _>(template, &input, &weights, &device)?;
    let path = checkpoint::save_content_addressed(merged, &out, "dblocks")?;
    write_merge_state(&path, &parents, &weights, &input)?;
    println!(
        "merged {} -> {}",
        diffusionblocks::merge::describe(&parents, &weights),
        path.display()
    );
    Ok(())
}

/// A state directory for a merged model recording its parents, so the
/// provenance chain does not stop at the merge.
/// A checkpoint plus its device, loaded for evaluation-side tooling.
fn load_lm_eval(
    checkpoint: &Path,
    tiny: bool,
) -> Result<(
    LanguageModel<Eval>,
    <Eval as burn::tensor::backend::BackendTypes>::Device,
)> {
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let config = if tiny {
        LmConfig::tiny()
    } else {
        LmConfig::default()
    };
    let model = checkpoint::load::<Eval, _>(
        LanguageModel::<Eval>::new(&config, &device)?,
        checkpoint,
        &device,
    )?;
    Ok((model, device))
}

/// Training-state sidecar for a checkpoint derived from another by a
/// weight-space operation (ablation, decensoring): no optimizer, a parent.
fn write_derived_state(path: &Path, kind: &str, extras: serde_json::Value) -> Result<()> {
    use diffusionblocks::checkpoint::{self as ck, TrainState};
    let dir = TrainState::dir_for(path);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    let state = TrainState {
        format_version: ck::STATE_FORMAT_VERSION,
        kind: kind.into(),
        step: 0,
        seed: 0,
        host_rng: serde_json::Value::Null,
        config: serde_json::Value::Null,
        build: ck::BuildInfo::current(),
        datasets: Vec::new(),
        model: ck::model_entry(path)?,
        optimizer: None,
        ema: None,
        head: None,
        head_optimizer: None,
        extras,
        saved_unix_secs: ck::unix_now(),
    };
    state.write(&dir)?;
    Ok(())
}

fn write_merge_state(
    path: &Path,
    parents: &[String],
    weights: &[f64],
    inputs: &[PathBuf],
) -> Result<()> {
    use diffusionblocks::checkpoint::{self as ck, TrainState};
    let dir = TrainState::dir_for(path);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    let state = TrainState {
        format_version: ck::STATE_FORMAT_VERSION,
        kind: "merge".into(),
        step: 0,
        seed: 0,
        host_rng: serde_json::Value::Null,
        config: serde_json::json!({
            "inputs": inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "weights": weights,
        }),
        build: ck::BuildInfo::current(),
        datasets: Vec::new(),
        model: ck::model_entry(path)?,
        optimizer: None,
        ema: None,
        head: None,
        head_optimizer: None,
        extras: serde_json::json!({ "parents": parents, "weights": weights }),
        saved_unix_secs: ck::unix_now(),
    };
    state.write(&dir)?;
    Ok(())
}

fn cmd_audit(action: AuditAction) -> Result<()> {
    match action {
        AuditAction::Propagation {
            model: model_args,
            batch_size,
            epsilon,
            json,
        } => {
            let model = model_args.build(None)?;
            let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
            let mut rng = StdRng::seed_from_u64(model_args.seed);
            let mut dataset = SyntheticDataset::new(
                model_args.image_size,
                model_args.num_labels,
                batch_size,
                model_args.seed,
            );
            let batch = dataset.next_batch(&mut rng, &device)?;
            let report = diffusionblocks::audit::propagation(
                &model,
                &batch.pixel_values,
                &batch.labels,
                epsilon,
            );
            print!("{}", report.render());
            println!(
                "sensitivity is ||H(z+e) - H(z)|| / ||e|| for the block's one-step map; amplification is the\n\
                 product of the later blocks' sensitivities. On random weights these describe the\n\
                 initialization; pass --checkpoint to audit a trained model."
            );
            if let Some(path) = json {
                std::fs::write(&path, report.to_json()?)
                    .map_err(|err| anyhow::anyhow!("write {}: {err}", path.display()))?;
                println!("report written to {}", path.display());
            }
            Ok(())
        }
    }
}

fn cmd_experiment(action: ExperimentAction) -> Result<()> {
    use diffusionblocks::experiment::{compare, render_comparison, RunLog};
    match action {
        ExperimentAction::Show { path } => {
            let records = RunLog::read(&path)?;
            println!(
                "{:<40} {:>6} {:>4} {:>4} {:>14} {:>12}",
                "name", "unit", "n", "warm", "mean ± ci95", "median"
            );
            println!("{}", "-".repeat(86));
            for r in &records {
                let warm = r.trials.iter().filter(|t| t.warmup).count();
                match r.summary {
                    Some(s) => println!(
                        "{:<40} {:>6} {:>4} {:>4} {:>14} {:>12.4}",
                        r.name,
                        r.unit,
                        s.n,
                        warm,
                        format!(
                            "{:.4}±{:.4}",
                            s.mean,
                            if s.ci95_half_width.is_nan() {
                                0.0
                            } else {
                                s.ci95_half_width
                            }
                        ),
                        s.median
                    ),
                    None => println!(
                        "{:<40} {:>6} {:>4} {:>4} {:>14} {:>12}",
                        r.name, r.unit, 0, warm, "-", "-"
                    ),
                }
            }
            if let Some(first) = records.first() {
                let e = &first.environment;
                println!(
                    "\n{} record(s); first taken on {} ({} cpu(s)), {} {}, build {} ({})",
                    records.len(),
                    e.cpu_model,
                    e.logical_cpus,
                    e.os_name,
                    e.os_release,
                    e.build.git_revision,
                    e.build.profile
                );
            }
            Ok(())
        }
        ExperimentAction::Compare { a, b } => {
            let rows = compare(&RunLog::read(&a)?, &RunLog::read(&b)?);
            anyhow::ensure!(
                !rows.is_empty(),
                "no record names in common between {} and {}",
                a.display(),
                b.display()
            );
            print!("{}", render_comparison(&rows));
            println!(
                "\noverlap = the two 95% intervals overlap: not evidence of no difference, only of not enough trials to show one."
            );
            Ok(())
        }
    }
}

fn cmd_infer(
    model_args: ModelArgs,
    batch_size: usize,
    num_inference_steps: usize,
    top_k: usize,
    solver: &str,
) -> Result<()> {
    let model = model_args.build(Some(num_inference_steps))?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let mut rng = StdRng::seed_from_u64(model_args.seed);
    let mut dataset = SyntheticDataset::new(
        model_args.image_size,
        model_args.num_labels,
        batch_size,
        model_args.seed,
    );
    let batch = dataset.next_batch(&mut rng, &device)?;

    let engine = InferenceEngine::new(
        model,
        InferenceConfig {
            solver: SolverKind::parse(solver)?,
            num_steps: Some(num_inference_steps),
            batch_size,
            ..InferenceConfig::default()
        },
    );

    let mut profiler = Profiler::new();
    let preds = engine.classify_profiled(batch.pixel_values, &mut rng, &mut profiler);
    let truth: Vec<i64> = batch.labels.into_data().convert::<i64>().iter().collect();

    println!("top-{top_k} predictions:");
    for (i, row) in preds.top_k(top_k).iter().enumerate() {
        let formatted: Vec<String> = row.iter().map(|(c, p)| format!("{c}:{p:.3}")).collect();
        println!("  sample {i} (true {}): {}", truth[i], formatted.join("  "));
    }
    println!(
        "\naccuracy vs synthetic labels: {:.1}%",
        100.0 * preds.accuracy(&truth.iter().map(|&t| t as usize).collect::<Vec<_>>())
    );
    print!("\n{}", profiler.render());
    Ok(())
}

/// Field corpora: the per-field normalizer of PLAN-NINA §2.
fn cmd_corpus(action: CorpusAction) -> Result<()> {
    match action {
        CorpusAction::BuildField {
            field,
            datasets_root,
            out_root,
            config,
            force,
        } => {
            let config = match config {
                Some(path) => {
                    let text = std::fs::read_to_string(&path)
                        .with_context(|| format!("read config {}", path.display()))?;
                    serde_json::from_str::<fieldnorm::FieldnormConfig>(&text)
                        .with_context(|| format!("{} is not a FieldnormConfig", path.display()))?
                }
                None => fieldnorm::FieldnormConfig::default(),
            };
            let spec = fieldnorm::field_spec(&field)?;
            let corpus_dir = out_root.join(&spec.field);
            let manifest =
                fieldnorm::build_field_corpus(&spec, &datasets_root, &corpus_dir, &config, force)?;
            println!(
                "{}: {} documents ({} chars), {} sources, {} cheat-dropped, {} near-dup, {} oversize; manifest {}",
                manifest.field,
                manifest.total_documents,
                manifest.total_chars,
                manifest.sources.len(),
                manifest.dropped_by_class.values().sum::<usize>(),
                manifest.near_duplicates,
                manifest.oversize,
                corpus_dir.join("dataset/manifest.json").display()
            );
            Ok(())
        }
        CorpusAction::ToPairs {
            corpus,
            field,
            out,
            max_docs,
        } => {
            let config = fieldnorm::ToPairsConfig { max_docs };
            let manifest = fieldnorm::to_pairs(&corpus, &field, &out, &config)?;
            println!(
                "{}: {} docs in -> {} pairs ({} skipped under {} chars), cap {}; manifest at {}.manifest.json",
                manifest.field,
                manifest.docs_in,
                manifest.docs_out,
                manifest.skipped_short,
                fieldnorm::MIN_PAIR_DOC_CHARS,
                manifest.max_docs,
                out.display()
            );
            Ok(())
        }
    }
}

/// The NINA program: MCP server, registry reads, chain verification.
fn cmd_nina(action: NinaAction) -> Result<()> {
    match action {
        NinaAction::McpServe { root } => {
            let server = nina::McpServer::new(nina::NinaPaths::new(root));
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            server.serve(stdin.lock(), stdout.lock())
        }
        NinaAction::Registry { action, registry } => cmd_nina_registry(&action, &registry),
        NinaAction::Chat { args } => cmd_nina_chat(&args),
        NinaAction::Serve { port, root } => {
            let server = ninagui::GuiServer::bind(&root, port)?;
            println!(
                "nina dashboard: http://{} (root {})",
                server.local_addr()?,
                root.display()
            );
            server.serve()
        }
        NinaAction::VerifyLog { path } => {
            // A broken chain propagates: main returns Err, the exit code is
            // non-zero, and the message names the first bad line.
            let report = nina::ChainedLog::verify(&path)?;
            println!(
                "chain ok: {} — {} record(s), head {}",
                path.display(),
                report.lines,
                report.head_hash
            );
            Ok(())
        }
    }
}

fn cmd_nina_registry(action: &NinaRegistryAction, registry: &Path) -> Result<()> {
    let file = nina::RegistryFile::load(registry)?;
    match action {
        NinaRegistryAction::List { field } => {
            let models: Vec<&nina::ModelEntry> = file
                .models
                .iter()
                .filter(|m| field.as_deref().is_none_or(|f| m.field == f))
                .collect();
            println!("{}", serde_json::to_string_pretty(&models)?);
            println!(
                "{} entr{}",
                models.len(),
                if models.len() == 1 { "y" } else { "ies" }
            );
            Ok(())
        }
        NinaRegistryAction::Get { name } => {
            let entry = file
                .models
                .iter()
                .find(|m| &m.name == name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no model named '{name}' in {} ({} entries)",
                        registry.display(),
                        file.models.len()
                    )
                })?;
            println!("{}", serde_json::to_string_pretty(entry)?);
            Ok(())
        }
    }
}

/// How long `nina chat` waits for a subnext responder on the interop file
/// channel before falling back to no-context: short, finite and loud —
/// hanging is not an option while no responder exists.
const CHAT_CHANNEL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// How many hits `nina chat` asks the retriever for; the token budget in
/// `augment_prompt` decides how many of them actually fit.
const CHAT_RETRIEVAL_K: usize = 8;

/// Resolve a registry entry to its checkpoint on disk. The pipeline's
/// provenance sidecar (`config.json` in the model directory) names both the
/// checkpoint file and the `StudentConfig` it was trained with; without one,
/// the content-addressed `student-*.mpk` files are the candidates, and more
/// than one is disambiguated by the registry's checkpoint hash.
///
/// A sidecar that names a checkpoint which is **not on disk** is stale, not
/// authoritative: retraining writes a new content-addressed file and the
/// registry entry with it, and nothing rewrites the pipeline's sidecar. The
/// registry is what names the live checkpoint, so resolution falls through to
/// the candidates rather than refusing to serve a model that is present. A
/// sidecar naming a file that *does* exist is still obeyed — that is the case
/// where two checkpoints are on disk and only the sidecar can say which.
fn resolve_specialist(
    model_dir: &Path,
    entry: &nina::ModelEntry,
) -> Result<(PathBuf, Option<diffusionblocks::student::StudentConfig>)> {
    let sidecar = model_dir.join("config.json");
    if sidecar.is_file() {
        let text = std::fs::read_to_string(&sidecar)
            .with_context(|| format!("read {}", sidecar.display()))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parse {}", sidecar.display()))?;
        let file = value
            .get("checkpoint")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                anyhow::anyhow!("{} carries no \"checkpoint\" file name", sidecar.display())
            })?;
        anyhow::ensure!(
            !file.contains('/') && !file.contains('\\'),
            "{} names checkpoint '{file}', which escapes the model directory",
            sidecar.display()
        );
        let checkpoint = model_dir.join(file);
        if checkpoint.is_file() {
            let config = value
                .get("student_config")
                .map(|v| {
                    serde_json::from_value(v.clone()).with_context(|| {
                        format!("parse the student_config in {}", sidecar.display())
                    })
                })
                .transpose()?;
            return Ok((checkpoint, config));
        }
        // Stale sidecar: say so and let the registry decide below. The
        // student_config is dropped with it — it described the checkpoint
        // that is gone, and serving the live one under the dead one's
        // config is a shape mismatch waiting to happen.
        eprintln!(
            "note: {} names checkpoint '{file}', which is not on disk; resolving '{}' from \
             the registry's checkpoint hash instead",
            sidecar.display(),
            entry.name
        );
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    for dir_entry in std::fs::read_dir(model_dir)
        .with_context(|| format!("list model directory {}", model_dir.display()))?
    {
        let path = dir_entry
            .with_context(|| format!("read an entry of {}", model_dir.display()))?
            .path();
        if path.extension().is_some_and(|ext| ext == "mpk") {
            candidates.push(path);
        }
    }
    candidates.sort();
    anyhow::ensure!(
        !candidates.is_empty(),
        "no specialist checkpoint in {}: expected the pipeline's config.json sidecar or a \
         content-addressed student-*.mpk (registry entry '{}')",
        model_dir.display(),
        entry.name
    );
    if candidates.len() == 1 {
        return Ok((candidates.remove(0), None));
    }
    // Two or more checkpoints: the registry's `checkpoint_hash` is the sha256
    // of the *file bytes*, which is a different digest from the one in the
    // file name — `save_content_addressed` names a file after a canonical
    // hash of the module record, so the registry hash never appears in the
    // name and matching on it there can never succeed. Hash the candidates.
    //
    // This costs a pass over each file (a few hundred ms for a 200 MB
    // checkpoint), which is the right trade against refusing to serve a
    // model that is present and whose identity the registry states exactly.
    let mut matching = Vec::new();
    let mut hashed = Vec::new();
    for path in &candidates {
        let digest = checkpoint::file_sha256_hex(path)
            .with_context(|| format!("hash the candidate checkpoint {}", path.display()))?;
        hashed.push((path.clone(), digest.clone()));
        if !entry.checkpoint_hash.is_empty() && digest == entry.checkpoint_hash {
            matching.push(path.clone());
        }
    }
    let (Some(first), None) = (matching.first().cloned(), matching.get(1).cloned()) else {
        anyhow::bail!(
            "{} checkpoints in {} and the registry hash '{}' {} of them: {}",
            candidates.len(),
            model_dir.display(),
            entry.checkpoint_hash,
            if matching.is_empty() {
                "matches none"
            } else {
                "matches more than one"
            },
            hashed
                .iter()
                .map(|(p, d)| format!("{} (sha256 {d})", p.display()))
                .collect::<Vec<_>>()
                .join(", ")
        );
    };
    Ok((first, None))
}

/// Recover the `StudentConfig` a checkpoint was trained with from its
/// training-state sidecar; the default config when none is recorded (the same
/// rule the registry's router applies).
fn student_config_beside(checkpoint: &Path) -> Result<diffusionblocks::student::StudentConfig> {
    match checkpoint::TrainState::for_model(checkpoint)? {
        Some(state) => match state.config.get("student_config") {
            Some(value) => serde_json::from_value(value.clone()).with_context(|| {
                format!(
                    "parse the student config recorded beside {}",
                    checkpoint.display()
                )
            }),
            None => Ok(diffusionblocks::student::StudentConfig::default()),
        },
        None => Ok(diffusionblocks::student::StudentConfig::default()),
    }
}

/// Byte-tokenize a serve-time prompt: document-style BOS prefix, then the raw
/// bytes, and deliberately no closing EOS so the model continues the document
/// instead of being told it ended.
fn encode_prompt(text: &str) -> Vec<u16> {
    let mut ids = Vec::with_capacity(text.len() + 1);
    ids.push(diffusionblocks::tokenizer::Special::Bos.id());
    ids.extend(ByteTokenizer::new().encode(text));
    ids
}

/// Augment `prompt` with `hits` under a whitespace-token budget, then shrink
/// the budget until the byte-tokenized result plus `max_new_tokens` fits the
/// model's context window. Whitespace tokens and byte tokens are different
/// units (one word is several bytes), so the CLI's budget is only a starting
/// point; the window check here is the authoritative one. A bare prompt that
/// does not fit is a loud error, never a silent truncation.
fn fit_augmented_prompt(
    prompt: &str,
    hits: &[diffusionblocks::interop::ContextHit],
    mut budget: usize,
    context_window: usize,
    max_new_tokens: usize,
) -> Result<(String, Vec<u16>, usize)> {
    anyhow::ensure!(
        max_new_tokens < context_window,
        "--max-new-tokens {max_new_tokens} leaves no room for a prompt in the model's \
         {context_window}-token context window"
    );
    let bare = encode_prompt(prompt);
    anyhow::ensure!(
        bare.len() + max_new_tokens <= context_window,
        "the bare prompt is {} byte-tokens; with --max-new-tokens {max_new_tokens} it needs {} \
         of the model's {context_window}-token context window — shorten the prompt",
        bare.len(),
        bare.len() + max_new_tokens
    );
    loop {
        let augmented = diffusionblocks::interop::augment_prompt(prompt, hits, budget);
        let ids = encode_prompt(&augmented);
        if ids.len() + max_new_tokens <= context_window {
            return Ok((augmented, ids, budget));
        }
        anyhow::ensure!(
            budget > 0,
            "context overflow with no injectable budget left: the augmented prompt is {} \
             byte-tokens against a {context_window}-token window and {max_new_tokens} \
             requested new tokens",
            ids.len()
        );
        // Scale the budget by the observed overflow (the loop only reaches
        // here when ids.len() exceeds the room, so the ratio shrinks), and
        // always give up at least an eighth so the loop terminates.
        let room = context_window - max_new_tokens;
        budget = (budget * room / ids.len()).min(budget * 7 / 8);
    }
}

/// Greedy decode from a specialist: argmax at each step over a full-prefix
/// forward (the student keeps no KV cache), stopping at `<eos>` or a NUL
/// byte.
/// Context overflow is a loud error naming both numbers, never a silent
/// truncation.
fn greedy_generate(
    model: &diffusionblocks::student::GeometricStudent<Eval>,
    prompt_ids: &[u16],
    max_new_tokens: usize,
) -> Result<Vec<u16>> {
    use burn::tensor::{Int, Tensor};
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let context = model.config().context;
    let mut ids: Vec<u16> = prompt_ids.to_vec();
    let mut generated: Vec<u16> = Vec::new();
    for _ in 0..max_new_tokens {
        anyhow::ensure!(
            ids.len() < context,
            "context overflow during generation: the prefix grew to {} tokens against the \
             model's {context}-token context window",
            ids.len()
        );
        let flat: Vec<i64> = ids.iter().map(|t| i64::from(*t)).collect();
        let tokens =
            Tensor::<Eval, 1, Int>::from_ints(flat.as_slice(), &device).reshape([1, ids.len()]);
        let logits = model.forward(tokens)?.logits;
        let [_, n, vocab] = logits.dims();
        // argmax keeps its reduced dim; reshape before reading the scalar.
        let next: i64 = logits
            .slice([0..1, (n - 1)..n, 0..vocab])
            .reshape([vocab])
            .argmax(0)
            .reshape([1])
            .into_scalar();
        let next = u16::try_from(next)
            .map_err(|e| anyhow::anyhow!("the argmax index {next} is not a byte-token id: {e}"))?;
        if next == diffusionblocks::tokenizer::Special::Eos.id() || next == 0 {
            break;
        }
        ids.push(next);
        generated.push(next);
    }
    Ok(generated)
}

/// Retrieve context for one chat turn: the context root when given, else the
/// interop file channel with a short, loud timeout, else nothing. Zero hits
/// is the designed safe degradation — augmentation with no hits returns the
/// prompt byte-identical. The returned string says where the hits came from,
/// for the transcript and the log.
fn chat_retrieve(
    args: &NinaChatArgs,
    field: &str,
) -> Result<(Vec<diffusionblocks::interop::ContextHit>, String)> {
    use diffusionblocks::interop;
    if args.no_context {
        return Ok((Vec::new(), "none (--no-context)".to_string()));
    }
    if let Some(root) = &args.context_root {
        let retriever = interop::LocalRetriever::load_blocks(root)?;
        let note = format!("{} ({} blocks)", root.display(), retriever.doc_count());
        return Ok((retriever.retrieve(&args.prompt, CHAT_RETRIEVAL_K), note));
    }
    let channel = interop::RequestChannel::open(Path::new(interop::INTEROP_DIR))?;
    let query = interop::ContextQuery::new(&args.prompt, CHAT_RETRIEVAL_K, field);
    let id = channel.submit(&query)?;
    match channel.poll(&id, CHAT_CHANNEL_TIMEOUT) {
        Ok(hits) => {
            let collected = channel.gc()?;
            Ok((
                hits,
                format!(
                    "interop channel {} (request {id}, {collected} answered pair(s) collected)",
                    interop::INTEROP_DIR
                ),
            ))
        }
        Err(err) => {
            eprintln!(
                "note: {err}; continuing without context — the bare prompt is the designed \
                 safe degradation"
            );
            Ok((
                Vec::new(),
                "none (no context responder answered)".to_string(),
            ))
        }
    }
}

fn cmd_nina_chat(args: &NinaChatArgs) -> Result<()> {
    use diffusionblocks::student::GeometricStudent;

    let paths = nina::NinaPaths::new(&args.root);
    let registry = nina::RegistryFile::load(&paths.registry_json())?;
    let entry = registry
        .models
        .iter()
        .find(|m| m.name == args.model)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model named '{}' in {} ({} entries: {})",
                args.model,
                paths.registry_json().display(),
                registry.models.len(),
                registry
                    .models
                    .iter()
                    .map(|m| m.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let model_dir = paths.model_dir(&entry.field, &entry.subfield);
    let (checkpoint_path, sidecar_config) = resolve_specialist(&model_dir, entry)?;
    let config = match sidecar_config {
        Some(config) => config,
        None => student_config_beside(&checkpoint_path)?,
    };
    config.validate()?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <Eval as burn::tensor::backend::Backend>::seed(&device, 0);
    let model = GeometricStudent::<Eval>::new(&config, &device)?;
    diffusionblocks::tensor_ext::force_initialization(&model);
    let model = checkpoint::load::<Eval, _>(model, &checkpoint_path, &device)?;
    println!(
        "model: {} ({}, checkpoint {})",
        entry.name,
        entry.status.as_str(),
        checkpoint_path.display()
    );

    let (hits, context_note) = chat_retrieve(args, &entry.field)?;
    let (augmented, ids, budget) = fit_augmented_prompt(
        &args.prompt,
        &hits,
        args.max_context_tokens,
        config.context,
        args.max_new_tokens,
    )?;
    let injected: Vec<&diffusionblocks::interop::ContextHit> = hits
        .iter()
        .filter(|hit| {
            augmented.contains(&format!(
                "[context id={} source={}]",
                hit.id, hit.provenance.source
            ))
        })
        .collect();
    println!(
        "context: {context_note}; {} of {} hit(s) injected (budget {budget} whitespace tokens)",
        injected.len(),
        hits.len()
    );
    for hit in &injected {
        println!("  id={} score={:.4}", hit.id, hit.score);
    }

    let generated = greedy_generate(&model, &ids, args.max_new_tokens)?;
    let answer = ByteTokenizer::new().decode_lossy(&generated);
    println!("answer ({} tokens):\n{answer}", generated.len());

    let chat_log = paths.root().join("logs").join("chat.jsonl");
    let mut log = nina::ChainedLog::open(&chat_log)?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let record = log.append(&serde_json::json!({
        "ts": ts,
        "model": entry.name,
        "prompt": args.prompt,
        "context": {
            "source": context_note,
            "budget_tokens": budget,
            "injected": injected
                .iter()
                .map(|h| serde_json::json!({"id": h.id, "score": h.score}))
                .collect::<Vec<_>>(),
        },
        "answer": answer,
        "answer_tokens": generated.len(),
    }))?;
    println!(
        "turn logged: {} seq {} (verify: dblocks nina verify-log {})",
        chat_log.display(),
        record.seq,
        chat_log.display()
    );
    Ok(())
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it (see the `[lints]` table in
// `Cargo.toml` and the contract in the crate docs).
#[allow(
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
mod student_cli_tests {
    use super::*;

    fn scratch(stem: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("dblocks-student-cli-{stem}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_sidecar(dir: &Path) -> PathBuf {
        let path = dir.join("student.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "hidden_size": 32,
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
                "context": 24,
                "intermediate_size": 16,
                "cond_hidden_size": 8,
                "frequency_embedding_size": 8,
                "attention": "dense",
                "experts_per_box": 1,
                "geom_slots": 2,
                "geom_dim": 4,
                "geom_refine_steps": 1,
                "geom_cond_hidden": 8,
                "geom_frequency": 4,
            })
            .to_string(),
        )
        .unwrap();
        path
    }

    fn fixture_traces(dir: &Path) -> PathBuf {
        let path = dir.join("train.jsonl");
        let lines: Vec<String> = [
            ("rust", "fn add(", ") -> i32 { 3 }"),
            ("rust", "let x =", " 1 + 2;"),
            ("python", "def f(", "): pass"),
            ("rust", "impl A {", " fn b(&self) {} }"),
        ]
        .iter()
        .map(|(language, prompt, completion)| {
            serde_json::json!({
                "id": language,
                "language": language,
                "prompt": prompt,
                "completion": completion,
                "source": "fixture",
            })
            .to_string()
        })
        .collect();
        std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
        path
    }

    fn train_args(dir: &Path) -> StudentTrainArgs {
        StudentTrainArgs {
            language: "rust".into(),
            traces: fixture_traces(dir),
            steps: 2,
            batch_size: 2,
            lr: 1e-3,
            config: Some(tiny_sidecar(dir)),
            resume: None,
            out_dir: dir.join("ckpt"),
            seed: 5,
            router_aux_weight: 0.1,
            log_every: 0,
            checkpoint_every: 0,
            clip_norm: 0.0,
        }
    }

    #[test]
    fn student_subcommands_parse() {
        let cli = Cli::try_parse_from([
            "dblocks",
            "student",
            "train",
            "--language",
            "rust",
            "--traces",
            "t.jsonl",
            "--steps",
            "3",
        ])
        .unwrap();
        let Command::Student {
            action: StudentAction::Train { args },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(args.language, "rust");
        assert_eq!(args.steps, 3);
        assert_eq!(args.out_dir, PathBuf::from(DEFAULT_STUDENT_OUT));
        assert!(args.resume.is_none());
        assert!(Cli::try_parse_from([
            "dblocks",
            "student",
            "route",
            "--language-prompt",
            "fn main() {}"
        ])
        .is_ok());
        assert!(Cli::try_parse_from(["dblocks", "student", "info"]).is_ok());
        // `--language` is required for train.
        assert!(Cli::try_parse_from(["dblocks", "student", "train"]).is_err());

        let cli = Cli::try_parse_from([
            "dblocks",
            "student",
            "think",
            "--prompt",
            "fn main() {}",
            "--depths",
            "1,2,4",
            "--top-k",
            "3",
        ])
        .unwrap();
        let Command::Student {
            action:
                StudentAction::Think {
                    prompt,
                    depths,
                    checkpoint,
                    top_k,
                    ..
                },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(prompt, "fn main() {}");
        assert_eq!(depths, vec![1, 2, 4]);
        assert_eq!(top_k, 3);
        assert!(checkpoint.is_none());
        // The defaults: the 1,2,4,8 sweep and top-5.
        let cli = Cli::try_parse_from(["dblocks", "student", "think", "--prompt", "x"]).unwrap();
        let Command::Student {
            action: StudentAction::Think { depths, top_k, .. },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(depths, vec![1, 2, 4, 8]);
        assert_eq!(top_k, 5);
        // `--prompt` is required.
        assert!(Cli::try_parse_from(["dblocks", "student", "think"]).is_err());
    }

    #[test]
    fn student_info_runs_on_a_fresh_config() {
        cmd_student_info(None).unwrap();
        let dir = scratch("info");
        cmd_student_info(Some(&tiny_sidecar(&dir))).unwrap();
        let err = cmd_student_info(Some(&dir.join("absent.json")))
            .unwrap_err()
            .to_string();
        assert!(err.contains("absent.json"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn student_train_on_a_fixture_writes_a_checkpoint() {
        let dir = scratch("train");
        cmd_student_train(&train_args(&dir)).unwrap();
        let checkpoint = checkpoint::latest_in_dir(&dir.join("ckpt"), "student")
            .unwrap()
            .expect("the run must write a checkpoint");
        let state = checkpoint::TrainState::for_model(&checkpoint)
            .unwrap()
            .expect("training state beside the model");
        assert_eq!(state.kind, "student");
        assert_eq!(state.step, 2);
        state
            .verify_files(&checkpoint::TrainState::dir_for(&checkpoint))
            .unwrap();

        // A bad language fails before any work, naming the flag value.
        let mut bad = train_args(&dir);
        bad.language = "cobol".into();
        let err = cmd_student_train(&bad).unwrap_err().to_string();
        assert!(err.contains("cobol"), "{err}");

        // Missing traces fail loudly, naming the path.
        let mut missing = train_args(&dir);
        missing.traces = dir.join("absent.jsonl");
        let err = cmd_student_train(&missing).unwrap_err().to_string();
        assert!(err.contains("absent.jsonl"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn student_route_runs_with_fresh_weights() {
        let dir = scratch("route");
        cmd_student_route(
            "fn main() { println!(\"hi\"); }",
            None,
            Some(&tiny_sidecar(&dir)),
        )
        .unwrap();
        let err = cmd_student_route(
            "x",
            Some(&dir.join("absent.mpk")),
            Some(&tiny_sidecar(&dir)),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("absent.mpk"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn student_think_runs_with_fresh_weights() {
        let dir = scratch("think");
        cmd_student_think("fn main() {}", &[1, 2], None, Some(&tiny_sidecar(&dir)), 3).unwrap();
        // A missing checkpoint is a loud error naming the path.
        let err = cmd_student_think(
            "x",
            &[1],
            Some(&dir.join("absent.mpk")),
            Some(&tiny_sidecar(&dir)),
            3,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("absent.mpk"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it (see the `[lints]` table in
// `Cargo.toml` and the contract in the crate docs).
#[allow(
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
mod nina_cli_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SCRATCH: AtomicUsize = AtomicUsize::new(0);

    fn scratch(stem: &str) -> PathBuf {
        let n = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "dblocks-nina-cli-{stem}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn nina_subcommands_parse() {
        let cli = Cli::try_parse_from(["dblocks", "nina", "mcp-serve"]).unwrap();
        let Command::Nina {
            action: NinaAction::McpServe { root },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(root, PathBuf::from(nina::DEFAULT_ROOT));

        let cli =
            Cli::try_parse_from(["dblocks", "nina", "mcp-serve", "--root", "/tmp/nina-x"]).unwrap();
        let Command::Nina {
            action: NinaAction::McpServe { root },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(root, PathBuf::from("/tmp/nina-x"));

        let cli = Cli::try_parse_from(["dblocks", "nina", "registry", "list", "--field", "coding"])
            .unwrap();
        let Command::Nina {
            action:
                NinaAction::Registry {
                    action: NinaRegistryAction::List { field },
                    registry,
                },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(field.as_deref(), Some("coding"));
        assert_eq!(registry, PathBuf::from(nina::DEFAULT_REGISTRY));

        let cli =
            Cli::try_parse_from(["dblocks", "nina", "registry", "get", "coding/rust"]).unwrap();
        let Command::Nina {
            action:
                NinaAction::Registry {
                    action: NinaRegistryAction::Get { name },
                    ..
                },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(name, "coding/rust");

        // --registry is global: it also parses after the leaf subcommand.
        let cli = Cli::try_parse_from([
            "dblocks",
            "nina",
            "registry",
            "get",
            "coding/rust",
            "--registry",
            "/tmp/r.json",
        ])
        .unwrap();
        let Command::Nina {
            action: NinaAction::Registry { registry, .. },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(registry, PathBuf::from("/tmp/r.json"));

        let cli =
            Cli::try_parse_from(["dblocks", "nina", "verify-log", "logs/registry.jsonl"]).unwrap();
        let Command::Nina {
            action: NinaAction::VerifyLog { path },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(path, PathBuf::from("logs/registry.jsonl"));

        // `get` needs a name and `verify-log` needs a path.
        assert!(Cli::try_parse_from(["dblocks", "nina", "registry", "get"]).is_err());
        assert!(Cli::try_parse_from(["dblocks", "nina", "verify-log"]).is_err());
    }

    #[test]
    fn nina_serve_parses_with_defaults_and_overrides() {
        let cli = Cli::try_parse_from(["dblocks", "nina", "serve"]).unwrap();
        let Command::Nina {
            action: NinaAction::Serve { port, root },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(port, 7878);
        assert_eq!(root, PathBuf::from(nina::DEFAULT_ROOT));

        let cli = Cli::try_parse_from([
            "dblocks",
            "nina",
            "serve",
            "--port",
            "0",
            "--root",
            "/tmp/nina-gui-x",
        ])
        .unwrap();
        let Command::Nina {
            action: NinaAction::Serve { port, root },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(port, 0);
        assert_eq!(root, PathBuf::from("/tmp/nina-gui-x"));

        // A non-numeric port is rejected at parse time.
        assert!(Cli::try_parse_from(["dblocks", "nina", "serve", "--port", "x"]).is_err());
    }

    #[test]
    fn nina_registry_cli_and_verify_log_run() {
        let dir = scratch("registry");
        let registry_path = dir.join("registry.json");
        let log_path = dir.join("logs").join("registry.jsonl");
        let mut registry = nina::Registry::open(&registry_path, &log_path).unwrap();
        let entry = nina::ModelEntry {
            name: "coding/rust".into(),
            field: "coding".into(),
            subfield: "rust".into(),
            box_id: "coding:rust".into(),
            eval: nina::EvalEvidence {
                accuracy: Some(0.9),
                ..nina::EvalEvidence::default()
            },
            ..nina::ModelEntry::default()
        };
        registry.register(entry).unwrap();

        cmd_nina_registry(&NinaRegistryAction::List { field: None }, &registry_path).unwrap();
        cmd_nina_registry(
            &NinaRegistryAction::List {
                field: Some("coding".into()),
            },
            &registry_path,
        )
        .unwrap();
        cmd_nina_registry(
            &NinaRegistryAction::Get {
                name: "coding/rust".into(),
            },
            &registry_path,
        )
        .unwrap();
        let err = cmd_nina_registry(
            &NinaRegistryAction::Get {
                name: "bio/general".into(),
            },
            &registry_path,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("bio/general"), "{err}");
        let err = cmd_nina_registry(
            &NinaRegistryAction::List { field: None },
            &dir.join("absent.json"),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("absent.json"), "{err}");

        // verify-log: a good chain reports; a tampered one fails, naming the line.
        cmd_nina(NinaAction::VerifyLog {
            path: log_path.clone(),
        })
        .unwrap();
        let text = std::fs::read_to_string(&log_path).unwrap();
        let tampered = text.replacen("coding/rust", "coding/ruby", 1);
        assert_ne!(tampered, text);
        std::fs::write(&log_path, tampered).unwrap();
        let err = cmd_nina(NinaAction::VerifyLog {
            path: log_path.clone(),
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("line 1"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn nina_chat_parses_with_defaults_and_overrides() {
        let cli = Cli::try_parse_from([
            "dblocks",
            "nina",
            "chat",
            "--model",
            "tech/general",
            "--prompt",
            "hi",
        ])
        .unwrap();
        let Command::Nina {
            action: NinaAction::Chat { args },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(args.model, "tech/general");
        assert_eq!(args.prompt, "hi");
        assert_eq!(args.context_root, None);
        assert!(!args.no_context);
        assert_eq!(args.max_context_tokens, 768);
        assert_eq!(args.max_new_tokens, 64);
        assert_eq!(args.root, PathBuf::from(nina::DEFAULT_ROOT));

        let cli = Cli::try_parse_from([
            "dblocks",
            "nina",
            "chat",
            "--model",
            "coding/rust",
            "--prompt",
            "fn?",
            "--context-root",
            "/tmp/blocks",
            "--max-context-tokens",
            "128",
            "--max-new-tokens",
            "8",
            "--root",
            "/tmp/nina-root",
        ])
        .unwrap();
        let Command::Nina {
            action: NinaAction::Chat { args },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(args.context_root, Some(PathBuf::from("/tmp/blocks")));
        assert_eq!(args.max_context_tokens, 128);
        assert_eq!(args.max_new_tokens, 8);
        assert_eq!(args.root, PathBuf::from("/tmp/nina-root"));

        // --no-context and --context-root name the same switch twice.
        assert!(Cli::try_parse_from([
            "dblocks",
            "nina",
            "chat",
            "--model",
            "m/f",
            "--prompt",
            "x",
            "--no-context",
            "--context-root",
            "/tmp/x",
        ])
        .is_err());
        // --model and --prompt are required.
        assert!(Cli::try_parse_from(["dblocks", "nina", "chat"]).is_err());
        assert!(Cli::try_parse_from(["dblocks", "nina", "chat", "--model", "m/f"]).is_err());
    }

    #[test]
    fn nina_chat_runs_a_turn_and_logs_it() {
        use diffusionblocks::student::GeometricStudent;
        let dir = scratch("chat");

        // A registry with one tiny specialist.
        let entry = nina::ModelEntry {
            name: "test/tiny".into(),
            field: "test".into(),
            subfield: "tiny".into(),
            ..nina::ModelEntry::default()
        };
        let file = nina::RegistryFile {
            version: 1,
            models: vec![entry],
        };
        std::fs::write(
            dir.join("registry.json"),
            serde_json::to_string_pretty(&file).unwrap(),
        )
        .unwrap();

        // The specialist: a tiny student, content-addressed into the model
        // directory, with the pipeline's sidecar naming checkpoint + config.
        let config: diffusionblocks::student::StudentConfig =
            serde_json::from_value(serde_json::json!({
                "hidden_size": 32,
                "num_layers": 2,
                "num_heads": 2,
                "num_kv_heads": 1,
                "context": 512,
                "intermediate_size": 16,
                "cond_hidden_size": 8,
                "frequency_embedding_size": 8,
                "attention": "dense",
                "experts_per_box": 1,
                "geom_slots": 2,
                "geom_dim": 4,
                "geom_refine_steps": 1,
                "geom_cond_hidden": 8,
                "geom_frequency": 4,
            }))
            .unwrap();
        let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
        <Eval as burn::tensor::backend::Backend>::seed(&device, 0);
        let model = GeometricStudent::<Eval>::new(&config, &device).unwrap();
        diffusionblocks::tensor_ext::force_initialization(&model);
        let model_dir = dir.join("models").join("test").join("tiny");
        let checkpoint = checkpoint::save_content_addressed(model, &model_dir, "student").unwrap();
        let file_name = checkpoint
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap()
            .to_string();
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::json!({
                "name": "test/tiny",
                "checkpoint": file_name,
                "student_config": serde_json::to_value(&config).unwrap(),
            })
            .to_string(),
        )
        .unwrap();

        // The context root holds the answer to the prompt.
        let blocks = dir.join("blocks");
        std::fs::create_dir_all(&blocks).unwrap();
        std::fs::write(
            blocks.join("relay.txt"),
            "The Argent Fox relay listens on TCP port 7447 by default.",
        )
        .unwrap();

        let args = NinaChatArgs {
            model: "test/tiny".into(),
            prompt: "Which TCP port does the Argent Fox relay listen on?".into(),
            context_root: Some(blocks),
            no_context: false,
            max_context_tokens: 64,
            max_new_tokens: 4,
            root: dir.clone(),
        };
        cmd_nina_chat(&args).unwrap();

        // The turn is in the Merkle-chained log, the injected block is named,
        // and the chain verifies.
        let chat_log = dir.join("logs").join("chat.jsonl");
        let report = nina::ChainedLog::verify(&chat_log).unwrap();
        assert_eq!(report.lines, 1);
        let line = std::fs::read_to_string(&chat_log).unwrap();
        assert!(line.contains("test/tiny"), "{line}");
        assert!(line.contains("relay.txt"), "{line}");

        // --no-context is byte-identical to augmentation with zero hits.
        let (augmented, fitted, budget) =
            fit_augmented_prompt(&args.prompt, &[], 64, 512, 4).unwrap();
        assert_eq!(augmented, args.prompt);
        assert_eq!(fitted, encode_prompt(&args.prompt));
        assert_eq!(budget, 64);
        let args = NinaChatArgs {
            no_context: true,
            context_root: None,
            ..args
        };
        cmd_nina_chat(&args).unwrap();
        let report = nina::ChainedLog::verify(&chat_log).unwrap();
        assert_eq!(report.lines, 2);
        let line = std::fs::read_to_string(&chat_log)
            .unwrap()
            .lines()
            .next_back()
            .unwrap()
            .to_string();
        assert!(line.contains("--no-context"), "{line}");

        // An unknown model name is a loud error listing what exists.
        let args = NinaChatArgs {
            model: "bio/absent".into(),
            ..args
        };
        let err = cmd_nina_chat(&args).unwrap_err().to_string();
        assert!(err.contains("bio/absent"), "{err}");
        assert!(err.contains("test/tiny"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fit_augmented_prompt_shrinks_the_budget_to_the_window() {
        use diffusionblocks::interop::{ContextHit, Provenance};
        // One hit whose text is far wider than the window; the budget loop
        // must shrink until prompt + max-new fits, and the result stays marked.
        let hit = ContextHit::new(
            "doc-big",
            1.0,
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima",
            Provenance::new("blocks", "big.txt"),
        );
        let (augmented, ids, budget) =
            fit_augmented_prompt("q?", std::slice::from_ref(&hit), 64, 90, 4).unwrap();
        assert!(ids.len() + 4 <= 90);
        assert!(budget < 64);
        assert!(augmented.contains("[context id=doc-big source=blocks]"));
        assert!(augmented.ends_with("\nq?"));

        // A bare prompt that cannot fit even with budget zero is a loud error.
        let long = "x".repeat(64);
        let err = fit_augmented_prompt(&long, &[], 64, 40, 4)
            .unwrap_err()
            .to_string();
        assert!(err.contains("bare prompt"), "{err}");
    }

    /// Two checkpoints on disk are disambiguated by **hashing the files**,
    /// not by matching the registry hash against their names.
    ///
    /// Those are two different digests: `save_content_addressed` names a file
    /// after a canonical hash of the module record, while the registry stores
    /// the sha256 of the file's bytes, so the registry hash never appears in
    /// the name. The name-based match could therefore never succeed, and a
    /// second training run in a model directory made the model unservable —
    /// which is what happened to every coding specialist that was retrained.
    #[test]
    fn two_checkpoints_are_disambiguated_by_content_hash_not_by_name() {
        let dir = scratch("two-checkpoints");
        let model_dir = dir.join("models").join("coding").join("rust");
        std::fs::create_dir_all(&model_dir).unwrap();

        // Same stem, different bytes: the names differ, the digests differ.
        let live = model_dir.join("student-aaaaaaaaaaaaaaaa.mpk");
        let dead = model_dir.join("student-bbbbbbbbbbbbbbbb.mpk");
        std::fs::write(&live, b"the live checkpoint").unwrap();
        std::fs::write(&dead, b"the superseded one").unwrap();
        let live_hash = checkpoint::file_sha256_hex(&live).unwrap();

        let entry = nina::ModelEntry {
            name: "coding/rust".into(),
            field: "coding".into(),
            subfield: "rust".into(),
            box_id: "coding:rust".into(),
            checkpoint_hash: live_hash.clone(),
            status: nina::ModelStatus::Candidate,
            eval: Default::default(),
            datasets: Vec::new(),
            teacher: None,
            bytes: 19,
            alias_of: None,
        };

        let (resolved, _) = resolve_specialist(&model_dir, &entry).unwrap();
        assert_eq!(
            resolved, live,
            "the registry names the live checkpoint by content, and that is what is served"
        );

        // A registry hash matching neither candidate is still refused, and the
        // error names both digests so the operator can see what is there.
        let entry = nina::ModelEntry {
            checkpoint_hash: "0".repeat(64),
            ..entry
        };
        let err = resolve_specialist(&model_dir, &entry)
            .unwrap_err()
            .to_string();
        assert!(err.contains("matches none"), "{err}");
        assert!(
            err.contains(&live_hash[..16]),
            "the error names what is there: {err}"
        );
    }

    /// A model directory whose `config.json` names a checkpoint that is no
    /// longer on disk is a *stale sidecar*, not a broken model.
    ///
    /// Retraining writes a new content-addressed `.mpk` and the registry entry
    /// that names it; nothing rewrites the pipeline's provenance sidecar. Six
    /// of the deployed coding specialists were unservable for exactly this
    /// reason — the sidecar named an older checkpoint that had been replaced,
    /// and `resolve_specialist` refused rather than falling through to the
    /// registry, which is what actually names the live checkpoint.
    #[test]
    fn a_stale_sidecar_falls_through_to_the_registry_instead_of_refusing() {
        let dir = scratch("stale-sidecar");
        let model_dir = dir.join("models").join("coding").join("rust");
        std::fs::create_dir_all(&model_dir).unwrap();

        // The live checkpoint is the only .mpk present.
        let live = model_dir.join("student-aaaaaaaaaaaaaaaa.mpk");
        std::fs::write(&live, b"not really a checkpoint").unwrap();
        // The sidecar names a different one, which is gone.
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::json!({
                "name": "coding/rust",
                "checkpoint": "student-bbbbbbbbbbbbbbbb.mpk",
                "student_config": {"hidden_size": 999},
            })
            .to_string(),
        )
        .unwrap();

        let entry = nina::ModelEntry {
            name: "coding/rust".into(),
            field: "coding".into(),
            subfield: "rust".into(),
            box_id: "coding:rust".into(),
            checkpoint_hash: "abcdef0123456789".into(),
            status: nina::ModelStatus::Candidate,
            eval: Default::default(),
            datasets: Vec::new(),
            teacher: None,
            bytes: 21,
            alias_of: None,
        };

        let (resolved, config) = resolve_specialist(&model_dir, &entry).unwrap();
        assert_eq!(resolved, live, "the live checkpoint is the one served");
        assert!(
            config.is_none(),
            "the dead sidecar's student_config described a checkpoint that is gone; serving \
             the live one under it would be a shape mismatch"
        );
    }

    /// A sidecar naming a checkpoint that *is* present is still obeyed: that is
    /// the case where two checkpoints are on disk and only the sidecar knows
    /// which one the model directory is deployed with.
    #[test]
    fn a_sidecar_naming_a_present_checkpoint_still_wins() {
        let dir = scratch("fresh-sidecar");
        let model_dir = dir.join("models").join("coding").join("go");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("student-aaaaaaaaaaaaaaaa.mpk"), b"a").unwrap();
        std::fs::write(model_dir.join("student-bbbbbbbbbbbbbbbb.mpk"), b"b").unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            serde_json::json!({
                "name": "coding/go",
                "checkpoint": "student-bbbbbbbbbbbbbbbb.mpk",
            })
            .to_string(),
        )
        .unwrap();

        let entry = nina::ModelEntry {
            name: "coding/go".into(),
            field: "coding".into(),
            subfield: "go".into(),
            box_id: "coding:go".into(),
            // A hash matching neither file: the registry must not override a
            // sidecar that found its checkpoint.
            checkpoint_hash: "0000000000000000".into(),
            status: nina::ModelStatus::Candidate,
            eval: Default::default(),
            datasets: Vec::new(),
            teacher: None,
            bytes: 1,
            alias_of: None,
        };

        let (resolved, _) = resolve_specialist(&model_dir, &entry).unwrap();
        assert_eq!(
            resolved,
            model_dir.join("student-bbbbbbbbbbbbbbbb.mpk"),
            "the sidecar found its checkpoint, so it is authoritative"
        );
    }

    /// A sidecar naming nothing usable at all is still refused — the fallback
    /// is for a *stale* name, not for a broken directory.
    #[test]
    fn an_empty_model_directory_is_still_refused() {
        let dir = scratch("empty-model-dir");
        let model_dir = dir.join("models").join("coding").join("jsts");
        std::fs::create_dir_all(&model_dir).unwrap();
        let entry = nina::ModelEntry {
            name: "coding/jsts".into(),
            field: "coding".into(),
            subfield: "jsts".into(),
            box_id: "coding:jsts".into(),
            checkpoint_hash: "0000000000000000".into(),
            status: nina::ModelStatus::Candidate,
            eval: Default::default(),
            datasets: Vec::new(),
            teacher: None,
            bytes: 0,
            alias_of: None,
        };
        let err = resolve_specialist(&model_dir, &entry)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no specialist checkpoint"), "{err}");
    }
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right thing
// for a test to say. The grant is scoped to this module; production code in
// the same file is still denied it (see the `[lints]` table in `Cargo.toml`
// and the contract in the crate docs).
#[allow(
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
mod cheat_cli_tests {
    use super::*;

    /// Every `dblocks cheat` subcommand parses, with its defaults and with
    /// its flags.
    ///
    /// These subcommands exist as a shape, not just as a fix: while the four
    /// of them were one flat flag list, every option landed on the root `Cli`,
    /// which clap materialises whole on the stack of whatever parses it — and
    /// that pushed the test threads in this very binary past 2 MB. A parse
    /// test is what notices the next time the root command grows.
    /// The adaptation loop's steps must parse, and their defaults must be the
    /// ones the documentation quotes. A subcommand that cannot be reached from
    /// the command line is a module that is not wired, whatever its tests say.
    #[test]
    fn cheat_harness_subcommands_parse() {
        let cli = Cli::try_parse_from(["dblocks", "cheat", "harness", "round"]).unwrap();
        let Command::Cheat {
            action:
                CheatAction::Harness {
                    action:
                        CheatHarnessAction::Round {
                            corpus,
                            registry,
                            candidates,
                            json,
                        },
                },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(corpus, PathBuf::from("logs/cheat-corpus.jsonl"));
        assert_eq!(registry, PathBuf::from("logs/cheat-harness.json"));
        assert_eq!(candidates, 8);
        assert!(!json);

        let cli = Cli::try_parse_from([
            "dblocks",
            "cheat",
            "harness",
            "commit",
            "--candidates",
            "3",
            "--provenance",
            "a labeled rerun",
        ])
        .unwrap();
        let Command::Cheat {
            action:
                CheatAction::Harness {
                    action:
                        CheatHarnessAction::Commit {
                            candidates,
                            provenance,
                            ..
                        },
                },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(candidates, 3);
        assert_eq!(provenance, "a labeled rerun");

        // `provenance` is required on commit: an entry without a statement of
        // what it was measured against is a profile nobody can audit.
        assert!(
            Cli::try_parse_from(["dblocks", "cheat", "harness", "commit"]).is_err(),
            "commit must refuse to run without --provenance"
        );

        let cli = Cli::try_parse_from(["dblocks", "cheat", "harness", "rollback"]).unwrap();
        let Command::Cheat {
            action:
                CheatAction::Harness {
                    action: CheatHarnessAction::Rollback { registry },
                },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(registry, PathBuf::from("logs/cheat-harness.json"));

        let cli = Cli::try_parse_from(["dblocks", "cheat", "harness", "status"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Cheat {
                action: CheatAction::Harness {
                    action: CheatHarnessAction::Status { .. }
                }
            }
        ));
    }

    /// The `cheat scan` / `adapt` / `policy` / `ml` leaves all parse, with the
    /// defaults their help text quotes. A leaf that stops parsing is the kind
    /// of break a doc comment is supposed to notice.
    #[test]
    fn cheat_subcommands_parse() {
        let cli = Cli::try_parse_from(["dblocks", "cheat", "scan"]).unwrap();
        let Command::Cheat {
            action: CheatAction::Scan { root, json, all },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(root, PathBuf::from("."));
        assert!(!json && !all);

        let cli = Cli::try_parse_from([
            "dblocks", "cheat", "scan", "--root", "/tmp/t", "--json", "--all",
        ])
        .unwrap();
        let Command::Cheat {
            action: CheatAction::Scan { root, json, all },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(root, PathBuf::from("/tmp/t"));
        assert!(json && all);

        let cli = Cli::try_parse_from(["dblocks", "cheat", "adapt"]).unwrap();
        let Command::Cheat {
            action:
                CheatAction::Adapt {
                    log,
                    max_fire_rate,
                    out,
                },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(log, PathBuf::from("logs/queue.jsonl"));
        assert!((max_fire_rate - 0.5).abs() < f64::EPSILON);
        assert!(out.is_none());

        let cli = Cli::try_parse_from([
            "dblocks",
            "cheat",
            "adapt",
            "--out",
            "p.json",
            "--max-fire-rate",
            "0.9",
        ])
        .unwrap();
        let Command::Cheat {
            action: CheatAction::Adapt {
                max_fire_rate, out, ..
            },
        } = cli.command
        else {
            unreachable!()
        };
        assert!((max_fire_rate - 0.9).abs() < f64::EPSILON);
        assert_eq!(out, Some(PathBuf::from("p.json")));

        let cli =
            Cli::try_parse_from(["dblocks", "cheat", "policy", "--profile", "p.json"]).unwrap();
        let Command::Cheat {
            action: CheatAction::Policy { profile },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(profile, Some(PathBuf::from("p.json")));

        let cli = Cli::try_parse_from(["dblocks", "cheat", "ml", "--model", "m.json"]).unwrap();
        let Command::Cheat {
            action: CheatAction::Ml { model },
        } = cli.command
        else {
            unreachable!()
        };
        assert_eq!(model, Some(PathBuf::from("m.json")));
    }

    /// A bare `dblocks cheat` names no action, because "scan" is not a safe
    /// guess: the other three write files or read logs, and a mistyped
    /// subcommand that silently scanned instead would read as a clean run.
    #[test]
    fn cheat_without_a_subcommand_is_refused() {
        let err = match Cli::try_parse_from(["dblocks", "cheat"]) {
            Ok(_) => panic!("a bare `dblocks cheat` parsed; it must not"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("subcommand"), "{err}");
    }
}
