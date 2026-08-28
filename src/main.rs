//! `dblocks` CLI: train, sample, benchmark and verify DiffusionBlocks models.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

use diffusionblocks::{
    accuracy::{Ensemble, Guidance, LogitNorm, ScalingCurve, ScalingPoint},
    checkpoint,
    corpus::TokenCorpus,
    lm::{LanguageModel, LmConfig, Sampling},
    tokenizer::ByteTokenizer,
    data::{SyntheticDataset, TrainDataset},
    dblock::{DblockClassifier, DblockConfig},
    infer::{InferenceConfig, InferenceEngine},
    multi_block::{Gated, MultiBlockConfig, PlannedConfig, Strategy},
    planner::Budget,
    precision::{Precision, PrecisionPolicy},
    profile::{format_duration, Profiler},
    sigma,
    solver::SolverKind,
    expert_index::{BoxSpec, ExpertIndex, ExpertSpec, MosmeSpec},
    quality::{LayerGates, QualityGateConfig, TrainingChecks},
    schedule::LrSchedule,
    train::{self, DatasetChoice, Objective, TrainConfig},
    verify,
    vit::{MoeTrunkConfig, MosmeTrunkConfig, ViTDiTConfig},
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
        let model = DblockClassifier::<Eval>::new(&vit_cfg, &dblock_cfg, &device);
        match &self.checkpoint {
            Some(path) => checkpoint::load::<Eval, _>(model, path, &device),
            None => Ok(model),
        }
    }
}

// `Train` carries far more flags than the other subcommands, so the enum is
// sized for it. Boxing the variant would mean a second struct definition and an
// extra indirection on a type that is constructed exactly once per process.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum LmAction {
    /// Tokenize a UTF-8 text file into a pre-tokenized corpus.
    Tokenize {
        /// Input text file.
        #[arg(long)]
        input: PathBuf,
        /// Output corpus file (little-endian u16 tokens).
        #[arg(long)]
        out: PathBuf,
    },
    /// Report what a corpus contains, without loading it into memory.
    Corpus {
        #[arg(long)]
        path: PathBuf,
        /// Context length used to count training windows.
        #[arg(long, default_value_t = 256)]
        context: usize,
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
    },
}

#[derive(Subcommand)]
enum Command {
    /// Block-wise training.
    Train {
        /// Dataset: synthetic | cifar100 | tiny-imagenet.
        #[arg(long, default_value = "synthetic")]
        dataset: String,
        /// Directory holding the dataset's `.bin` splits.
        #[arg(long)]
        data_dir: Option<PathBuf>,
        /// Stream records from disk instead of loading the split into memory.
        #[arg(long, default_value_t = false)]
        streaming: bool,
        /// Objective: dblock | consistency | flow | distill.
        #[arg(long, default_value = "dblock")]
        objective: String,
        /// Frozen teacher checkpoint for `--objective distill`.
        #[arg(long)]
        teacher: Option<PathBuf>,
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
        /// when passed without a value.
        #[arg(long, num_args = 0..=1, default_missing_value = "")]
        resume: Option<String>,
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
    /// Language-model paths: tokenize a corpus, generate text, and compare
    /// the decoding strategies (roadmap Phases 19 and 21b).
    Lm {
        #[command(subcommand)]
        action: LmAction,
    },
    /// Inspect or scaffold an expert index — the manifest an inference engine
    /// routes from. Reads JSON only; never loads weights.
    Experts {
        #[command(subcommand)]
        action: ExpertsAction,
    },
    /// Print the block sigma schedule for inspection.
    Sigmas {
        #[arg(long, default_value_t = 3)]
        num_blocks: usize,
        #[arg(long, default_value_t = 0.05)]
        gamma: f64,
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

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Sigmas { num_blocks, gamma } => cmd_sigmas(num_blocks, gamma),
        Command::Lm { action } => cmd_lm(action),
        Command::Experts { action } => cmd_experts(action),
        Command::Verify { group } => cmd_verify(group.as_deref()),
        command @ Command::Train { .. } => cmd_train(command),
        command @ Command::Sample { .. } => cmd_sample(command),
        Command::Bench { model, num_inference_steps, batch_size, repeats } => {
            cmd_bench(model, num_inference_steps, batch_size, repeats)
        }
        Command::Infer { model, batch_size, num_inference_steps, top_k, solver } => {
            cmd_infer(model, batch_size, num_inference_steps, top_k, &solver)
        }
    }
}

fn cmd_lm(action: LmAction) -> Result<()> {
    match action {
        LmAction::Tokenize { input, out } => {
            let count = TokenCorpus::tokenize_file(&input, &out)?;
            println!(
                "{} -> {} | {count} tokens ({} bytes, u16 little-endian)",
                input.display(),
                out.display(),
                count * diffusionblocks::corpus::TOKEN_BYTES
            );
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
        } => {
            let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
            <Eval as burn::tensor::backend::Backend>::seed(&device, seed);

            let config = LmConfig::default();
            let model = LanguageModel::<Eval>::new(&config, &device);
            let tokenizer = ByteTokenizer::new();
            let ids = tokenizer.encode(&prompt);

            println!(
                "context={} layers={} hidden={} vocab={}",
                config.context, config.num_layers, config.hidden_size, config.vocab_size
            );

            let started = std::time::Instant::now();
            let out = if lookahead > 0 {
                let budget = Budget {
                    max_evaluations: budget,
                    max_depth: lookahead,
                    beam_width: beam,
                };
                let (out, stats) =
                    model.generate_lookahead(&ids, max_new, top_k, budget, &device);
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
            println!(
                "Weights are random, so the text is noise. What this run shows is\n\
                 that the decoding paths agree and what each one costs."
            );
            Ok(())
        }
    }
}

fn cmd_experts(action: ExpertsAction) -> Result<()> {
    match action {
        ExpertsAction::Init { out, boxes, top_box, top_expert } => {
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
                    anyhow::ensure!(!experts.is_empty(), "box '{name}' needs at least one expert");
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
                Err(_) => {
                    let spec = MosmeSpec::read(&index)?;
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
        streaming,
        objective,
        teacher,
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
        no_checks,
        no_preflight,
        verify_every,
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
        z_level,
        uncertainty,
        importance_bins,
        moe_every,
        moe_experts,
        moe_top_k,
    } = command
    else {
        unreachable!("cmd_train is only called with Command::Train")
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
        dataset: DatasetChoice::parse(&dataset, data_dir, streaming)?,
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
        teacher,
        moe: moe_every.map(|every| MoeTrunkConfig {
            num_experts: moe_experts,
            top_k: moe_top_k,
            every_n_layers: every,
            z_level,
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
                MosmeTrunkConfig::new(spec).with_every_n_layers(mosme_every)
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
    };

    println!(
        "training: dataset={dataset} objective={} blocks={num_blocks} steps={steps}",
        config.objective.name()
    );

    let path = if grad_checkpointing {
        let model = train::train_synthetic_generic::<
            burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
        >(&config)?;
        checkpoint::save_content_addressed_async(model, out_path, "dblocks")
            .join()
            .expect("save thread")?
    } else {
        let (model, summary) = train::train(&config)?;
        println!(
            "done: {} steps in {:.1}s (mean loss {:.4}, {} rejected by a quality check, {:.1}% reject rate)",
            summary.steps_taken,
            summary.elapsed_secs,
            summary.mean_loss,
            summary.steps_skipped,
            100.0 * summary.skip_rate()
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
        if async_save {
            checkpoint::save_content_addressed_async(model, out_path, "dblocks")
                .join()
                .expect("save thread")?
        } else {
            checkpoint::save_content_addressed(model, &out_path, "dblocks")?
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
fn describe_experts(
    spec: &MosmeSpec,
    config: &TrainConfig,
    model_id: &str,
) -> Result<ExpertIndex> {
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
        "hybrid" => Strategy::Hybrid { k, warmup_frac: 0.3 },
        "adaptive" => Strategy::Adaptive { k_max: k.max(1), conf_threshold: 0.9 },
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
        unreachable!("cmd_sample is only called with Command::Sample")
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
    let batch = dataset.next_batch(&mut rng, &device);

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
            println!("  step {i}: sigma -> {:.5} with a {}-block span", step.sigma, step.width);
        }
        if trace.forced_final_step {
            println!(
                "  (the step cap bound before sigma_min; the last step was unplanned)"
            );
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
            .map(|s| MultiBlockConfig { solver: *s, ..config.clone() })
            .collect();
        println!(
            "ensemble={} over {} solvers: {}",
            kind.name(),
            members.len(),
            members.iter().map(|m| m.solver.name()).collect::<Vec<_>>().join(", ")
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
    println!("top-1 agreement with synthetic labels: {correct}/{}", truth.len());
    Ok(())
}

fn cmd_bench(
    model_args: ModelArgs,
    num_inference_steps: usize,
    batch_size: usize,
    repeats: usize,
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
    let batch = dataset.next_batch(&mut rng, &device);

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
        ("hybrid-2", Strategy::Hybrid { k: 2, warmup_frac: 0.3 }),
        ("adaptive", Strategy::Adaptive { k_max: 3, conf_threshold: 0.9 }),
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
            for _ in 0..repeats.max(1) {
                let start = std::time::Instant::now();
                let out = model.sample_multi_block(&batch.pixel_values, &config, &mut rng);
                profiler.record(&scope, start.elapsed());
                last = Some(out);
            }

            let (logits, stats) = last.expect("at least one repeat");
            let preds: Vec<i64> = logits
                .argmax(1)
                .squeeze_dim::<1>(1)
                .into_data()
                .convert::<i64>()
                .iter()
                .collect();
            let agree = preds.iter().zip(&reference).filter(|(a, b)| a == b).count();

            println!(
                "{:<10} {:<12} {:>10} {:>12} {:>8} {:>9}/{}",
                kind.name(),
                label,
                format_duration(profiler.stats(&scope).unwrap().mean()),
                stats.model_calls,
                stats.layers_executed,
                agree,
                reference.len()
            );
        }
    }

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
        curve.push(ScalingPoint::new(
            format!("sequential/steps={steps}"),
            stats.model_calls,
            stats.layers_executed,
            diffusionblocks::accuracy::accuracy(&logits, &batch.labels),
        ));
    }
    for depth in [0usize, 1, 2] {
        let config = PlannedConfig {
            budget: Budget { max_evaluations: 48, max_depth: depth, beam_width: 3 },
            solver: SolverKind::Euler,
            max_steps: 6,
            ..PlannedConfig::default()
        };
        let (logits, stats, _) = model.sample_planned(&batch.pixel_values, &config, &mut rng);
        curve.push(ScalingPoint::new(
            format!("planned/depth={depth}"),
            stats.model_calls,
            stats.layers_executed,
            diffusionblocks::accuracy::accuracy(&logits, &batch.labels),
        ));
    }

    println!("\nTest-time compute scaling (* marks the Pareto frontier):");
    print!("{}", curve.render());
    for (label, rate) in curve.marginal_returns() {
        println!("  {label}: {:+.5} top-1 per extra layer", rate);
    }
    println!(
        "\nOn random weights top-1 is chance, so the frontier here demonstrates the\n\
         measurement, not a result. Run it on trained weights to size a budget."
    );
    Ok(())
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
    let batch = dataset.next_batch(&mut rng, &device);

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
