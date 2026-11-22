//! The `dblocks ncm` command tree as library types.
//!
//! The binary parses these with clap, calls [`run`], and prints what comes
//! back: progress lines through the callback, then [`NcmReport::render`].
//! Nothing here prints.
//!
//! The native trainers and generator share a backend-generic path.
//! Converted release checkpoints use the CPU runtime.

use crate::ncm::decode::{DecodeBudget, DecodeConfig, DecodeReport};
use crate::ncm::tokenizer::train::{
    evaluate_checkpoint, load_student_reader, train_tokenizer, EvalReport, IntentConfig,
    IntentLoss, IntentReader, TokenizerTrainConfig, TokenizerTrainReport,
};
use crate::ncm::tokenizer::NcmTokenizerConfig;
use anyhow::Result;
use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use clap::{Args, Subcommand};
use std::path::PathBuf;

type Cpu = burn::backend::NdArray<f32>;

/// Next-Concept Modeling: each stage is one variant.
#[derive(Subcommand, Debug, Clone)]
pub enum NcmAction {
    /// Stage 1: the multiscale residual-VQ tokenizer (train it, evaluate it).
    Tokenizer {
        #[command(subcommand)]
        action: TokenizerAction,
    },
    /// Stage 2: train the concept-field generator against a frozen
    /// tokenizer checkpoint.
    Train(Box<GeneratorTrainArgs>),
    /// Generate with a trained generator (NFE = K level calls per sample).
    Generate(Box<GenerateArgs>),
    /// Run a converted NCM release with native MoSME and concept geometry.
    Released(Box<crate::ncm::released::ReleasedArgs>),
    /// Fit the released model's final MoSME expert biases.
    ReleasedTune(Box<crate::ncm::released::TuneArgs>),
}

/// What `ncm tokenizer` does.
#[derive(Subcommand, Debug, Clone)]
pub enum TokenizerAction {
    /// Train the tokenizer on a byte-token corpus and report reconstruction
    /// quality per cutoff level on held-out data.
    Train(Box<TokenizerTrainArgs>),
    /// Evaluate a saved tokenizer on held-out corpora.
    Eval(TokenizerEvalArgs),
}

/// Flags of `ncm tokenizer train`.
#[derive(Args, Debug, Clone)]
pub struct TokenizerTrainArgs {
    /// Training corpus (`.corpus.bin`: little-endian u16 byte tokens).
    #[arg(long)]
    pub corpus: PathBuf,
    /// A held-out corpus, reported on its own; repeat once per field. With
    /// none, the corpus tail (`--val-fraction`, cut at a BOS) is held out.
    #[arg(long = "val-corpus")]
    pub val_corpus: Vec<PathBuf>,
    /// Read whole pre-shuffled response records in order rather than slicing random windows.
    #[arg(long)]
    pub ordered_records: bool,
    /// `tiny` (CPU minutes), `paper-owt`, `nina-4k` (GPU), or a JSON file of
    /// `NcmTokenizerConfig` fields.
    #[arg(long, default_value = "tiny")]
    pub config: String,
    /// Optimizer steps.
    #[arg(long, default_value_t = 300)]
    pub steps: usize,
    #[arg(long, default_value_t = 16)]
    pub batch_size: usize,
    /// Micro-batches per optimizer step.
    #[arg(long, default_value_t = 1)]
    pub accumulate: usize,
    /// Peak learning rate; 1e-3 below width 256, 3e-4 at and above it.
    #[arg(long)]
    pub lr: Option<f64>,
    #[arg(long, default_value_t = 0.0)]
    pub weight_decay: f64,
    /// Checkpoint directory (content-addressed `ncm-tokenizer-<hash>.mpk`
    /// plus its `.state` directory).
    #[arg(long)]
    pub out: PathBuf,
    /// Resume from a checkpoint written by this trainer.
    #[arg(long)]
    pub resume: Option<PathBuf>,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 10)]
    pub log_every: usize,
    /// Steps between evaluations; 0 evaluates at the start and the end only.
    #[arg(long, default_value_t = 0)]
    pub eval_every: usize,
    /// Validation windows per held-out set, at most.
    #[arg(long, default_value_t = 64)]
    pub val_windows: usize,
    /// Steps between checkpoints; 0 checkpoints at the end only.
    #[arg(long, default_value_t = 0)]
    pub checkpoint_every: usize,
    #[arg(long, default_value_t = 0.05)]
    pub val_fraction: f64,
    /// Weight λ of the optional intent-alignment term (unvalidated); 0 keeps
    /// it off: the reader is never loaded or called.
    #[arg(long, default_value_t = 0.0)]
    pub intent_weight: f64,
    /// Frozen nina student checkpoint the intent term aligns with. Required
    /// when `--intent-weight` is positive.
    #[arg(long)]
    pub intent_reader: Option<PathBuf>,
    /// The reader's StudentConfig JSON, when its training state does not
    /// record one.
    #[arg(long)]
    pub intent_reader_config: Option<PathBuf>,
    /// Coarse levels `c` the intent term reads (`ẑ^(c)`); default `⌈K/4⌉`.
    #[arg(long)]
    pub intent_levels: Option<usize>,
    /// cosine | mse.
    #[arg(long, default_value = "cosine")]
    pub intent_loss: String,
}

/// Flags of `ncm tokenizer eval`.
#[derive(Args, Debug, Clone)]
pub struct TokenizerEvalArgs {
    /// A tokenizer checkpoint (`ncm-tokenizer-<hash>.mpk`).
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Held-out corpus; repeat once per field.
    #[arg(long = "val-corpus", required = true)]
    pub val_corpus: Vec<PathBuf>,
    /// The architecture, when the checkpoint has no training state beside it.
    #[arg(long)]
    pub config: Option<String>,
    #[arg(long, default_value_t = 64)]
    pub val_windows: usize,
    #[arg(long, default_value_t = 16)]
    pub batch_size: usize,
}

/// What an `ncm` action produced.
#[derive(Debug, Clone)]
pub enum NcmReport {
    TokenizerTrain(Box<TokenizerTrainReport>),
    TokenizerEval(EvalReport),
    GeneratorTrain(Box<GeneratorTrainSummary>),
    Generate(Box<GenerateSummary>),
    Released(String),
}

impl NcmReport {
    /// Plain text for a terminal.
    pub fn render(&self) -> String {
        match self {
            Self::TokenizerTrain(report) => report.render(),
            Self::TokenizerEval(report) => report.render(),
            Self::GeneratorTrain(report) => report.render(),
            Self::Generate(report) => report.render(),
            Self::Released(report) => report.clone(),
        }
    }
}

fn train_config(args: TokenizerTrainArgs) -> Result<TokenizerTrainConfig> {
    let tokenizer = NcmTokenizerConfig::resolve(&args.config)?;
    let levels = tokenizer.schedule.levels();
    let mut config = TokenizerTrainConfig::new(tokenizer, args.corpus);
    if let Some(lr) = args.lr {
        config.lr = lr;
    }
    let intent = (args.intent_weight != 0.0 || args.intent_reader.is_some()).then(|| {
        IntentLoss::parse(&args.intent_loss).map(|loss| IntentConfig {
            weight: args.intent_weight,
            levels: args.intent_levels.unwrap_or(levels.div_ceil(4)),
            loss,
            reader: args.intent_reader.as_ref().map(|p| p.display().to_string()),
        })
    });
    Ok(TokenizerTrainConfig {
        val_corpora: args.val_corpus,
        ordered_records: args.ordered_records,
        val_fraction: args.val_fraction,
        steps: args.steps,
        batch_size: args.batch_size,
        accumulate: args.accumulate,
        weight_decay: args.weight_decay,
        seed: args.seed,
        log_every: args.log_every,
        eval_every: args.eval_every,
        val_windows: args.val_windows,
        checkpoint_every: args.checkpoint_every,
        out_dir: Some(args.out),
        resume: args.resume,
        intent: intent.transpose()?,
        ..config
    })
}

/// Run one `ncm` action on the CPU backend. Progress lines go to `progress`.
pub fn run(action: NcmAction, progress: &mut dyn FnMut(&str)) -> Result<NcmReport> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    match action {
        NcmAction::Released(args) => super::released::run(*args).map(NcmReport::Released),
        NcmAction::ReleasedTune(args) => {
            super::released::tune(*args, progress).map(NcmReport::Released)
        }
        action => run_native::<Cpu>(action, &device, progress),
    }
}

/// Run a native NCM tokenizer, trainer, or generator on the selected backend.
pub fn run_native<B: Backend<FloatElem = f32>>(
    action: NcmAction,
    device: &B::Device,
    progress: &mut dyn FnMut(&str),
) -> Result<NcmReport> {
    match action {
        NcmAction::Tokenizer { action } => match action {
            TokenizerAction::Train(args) => {
                let reader_path = args.intent_reader.clone();
                let reader_config = args.intent_reader_config.clone();
                let config = train_config(*args)?;
                let reader = match (&config.intent, reader_path) {
                    (Some(intent), Some(path)) if intent.is_active() => {
                        Some(load_student_reader::<B>(
                            &path,
                            reader_config.as_deref(),
                            config.tokenizer.seq_len(),
                            device,
                        )?)
                    }
                    _ => None,
                };
                let reader = reader.as_ref().map(|r| r as &dyn IntentReader<B>);
                let report = train_tokenizer::<Autodiff<B>>(&config, reader, device, progress)?;
                Ok(NcmReport::TokenizerTrain(Box::new(report)))
            }
            TokenizerAction::Eval(args) => {
                let config = args
                    .config
                    .as_deref()
                    .map(NcmTokenizerConfig::resolve)
                    .transpose()?;
                let report = evaluate_checkpoint::<B>(
                    &args.checkpoint,
                    config,
                    &args.val_corpus,
                    args.val_windows,
                    args.batch_size,
                    device,
                )?;
                Ok(NcmReport::TokenizerEval(report))
            }
        },
        NcmAction::Train(args) => train_generator_cli::<B>(*args, device, progress)
            .map(|r| NcmReport::GeneratorTrain(Box::new(r))),
        NcmAction::Generate(args) => {
            generate_cli::<B>(*args, device).map(|r| NcmReport::Generate(Box::new(r)))
        }
        NcmAction::Released(_) | NcmAction::ReleasedTune(_) => {
            anyhow::bail!("converted release inference and tuning require the CPU runtime")
        }
    }
}

// ------------------------------------------------- stage 2: generator --

/// Flags of `ncm train` (the concept-field generator).
#[derive(Args, Debug, Clone)]
pub struct GeneratorTrainArgs {
    /// The frozen tokenizer checkpoint (`ncm-tokenizer-<hash>.mpk` with its
    /// `.state` directory). Its sha256 is recorded with the generator.
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// Training corpus (`.corpus.bin`).
    #[arg(long)]
    pub corpus: PathBuf,
    /// Held-out corpus for per-level validation.
    #[arg(long = "val-corpus")]
    pub val_corpus: Option<PathBuf>,
    /// `tiny`, `sudoku` (8/512/8), `paper` (12/768/12), or a JSON file of
    /// `NcmGeneratorConfig` fields.
    #[arg(long, default_value = "tiny")]
    pub config: String,
    #[arg(long, default_value_t = 300)]
    pub steps: usize,
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,
    /// Peak learning rate (warmup, then cosine to `--min-lr`).
    #[arg(long, default_value_t = 3e-4)]
    pub lr: f64,
    #[arg(long, default_value_t = 3e-5)]
    pub min_lr: f64,
    #[arg(long, default_value_t = 20)]
    pub warmup: usize,
    /// Learning-rate schedule for full-backpropagation generator training.
    #[arg(long, default_value = "cosine", value_parser = ["cosine", "wsd"])]
    pub lr_schedule: String,
    /// Zero-based step where WSD begins to decay. Defaults to 80% of the run.
    #[arg(long)]
    pub wsd_decay_start: Option<usize>,
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f64,
    #[arg(long, default_value_t = 1.0)]
    pub grad_clip: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// `sampled` (one level per example, unbiased) or `all` (every level).
    #[arg(long, default_value = "sampled")]
    pub estimator: String,
    /// Level-weight exponent α of the loss.
    #[arg(long)]
    pub alpha: Option<f64>,
    /// Relaxation steps T per level call.
    #[arg(long)]
    pub thinking_steps: Option<usize>,
    /// Longest prompt prefix in tokens (0: unconditional).
    #[arg(long)]
    pub prompt_len: Option<usize>,
    /// Sample complete fixed-width prompt/target records, using the full prompt.
    #[arg(long)]
    pub aligned_windows: bool,
    /// Visit each pre-shuffled prompt/target record before starting the next pass.
    #[arg(long, requires = "aligned_windows")]
    pub ordered_records: bool,
    /// Weight of the deep-supervision term on intermediate states.
    #[arg(long)]
    pub deep_supervision: Option<f64>,
    /// Probability a corrupted conditioning code comes from the model's own
    /// prediction instead of a uniform draw.
    #[arg(long)]
    pub self_prob: Option<f64>,
    /// Turn crystal growth on with this confidence threshold.
    #[arg(long)]
    pub adaptive_confidence: Option<f64>,
    /// Crystal-growth budget as a fraction of `ℓ·T`.
    #[arg(long, default_value_t = 1.0)]
    pub adaptive_budget: f64,
    /// Crystal-growth per-position step cap (default `2T`).
    #[arg(long)]
    pub adaptive_cap: Option<usize>,
    /// Steps between validations; 0 validates at the start and the end.
    #[arg(long, default_value_t = 0)]
    pub val_every: usize,
    #[arg(long, default_value_t = 16)]
    pub val_windows: usize,
    /// Steps between checkpoints; 0 checkpoints at the end only.
    #[arg(long, default_value_t = 0)]
    pub checkpoint_every: usize,
    #[arg(long, default_value_t = 10)]
    pub log_every: usize,
    /// Checkpoint directory (`ncm-generator-<hash>.mpk` plus `.state`).
    #[arg(long)]
    pub out: PathBuf,
    /// Resume from a generator checkpoint (refused if the tokenizer differs).
    #[arg(long)]
    pub resume: Option<PathBuf>,
}

/// Flags of `ncm generate`.
#[derive(Args, Debug, Clone)]
pub struct GenerateArgs {
    /// The frozen tokenizer checkpoint the generator was trained against
    /// (refused if its sha256 differs from the one the generator recorded).
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// A generator checkpoint from `ncm train`.
    #[arg(long)]
    pub generator: PathBuf,
    /// Prompt text (byte-tokenized); omitted generates unconditionally.
    #[arg(long)]
    pub prompt: Option<String>,
    /// Read the complete UTF-8 prompt from stdin up to the trained prefix width.
    #[arg(long, conflicts_with = "prompt")]
    pub prompt_stdin: bool,
    /// Emit generated text and inference counters as one JSON object.
    #[arg(long, conflicts_with = "show_thoughts")]
    pub json: bool,
    /// Run a budgeted experimental decoder with sampling and strategy settings from JSON.
    #[arg(long, conflicts_with_all = ["temperature", "top_k", "token_temperature", "seed", "mbr", "show_thoughts"])]
    pub decode_policy: Option<PathBuf>,
    /// Code sampling temperature; 0 is greedy.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,
    /// Keep the k most likely codes; 0 keeps all.
    #[arg(long, default_value_t = 0)]
    pub top_k: usize,
    /// Final token decode temperature; 0 is greedy.
    #[arg(long, default_value_t = 0.0)]
    pub token_temperature: f64,
    /// Minimum-Bayes-risk samples (1 disables MBR).
    #[arg(long, default_value_t = 1)]
    pub mbr: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Override the relaxation depth T.
    #[arg(long)]
    pub thinking_steps: Option<usize>,
    /// Print every level's intermediate thoughts, decoded.
    #[arg(long)]
    pub show_thoughts: bool,
}

/// What `ncm train` did.
#[derive(Debug, Clone)]
pub struct GeneratorTrainSummary {
    pub report: crate::ncm::generator::NcmTrainReport,
    pub lengths: Vec<usize>,
    pub tokenizer_sha256: String,
}

impl GeneratorTrainSummary {
    pub fn render(&self) -> String {
        let r = &self.report;
        let mut out = format!(
            "concept-field generator: {} params, steps {}..{} ({} taken, {} skipped, {} clipped) in {:.1}s\n\
             loss first {:.4} last {:.4} mean {:.4}; trunk evaluations {}\n\
             tokenizer sha256 {}\n",
            r.params,
            r.start_step,
            r.start_step + r.steps_taken + r.steps_skipped,
            r.steps_taken,
            r.steps_skipped,
            r.steps_clipped,
            r.elapsed_secs,
            r.first_loss,
            r.last_loss,
            r.mean_loss,
            r.trunk_evals,
            self.tokenizer_sha256
        );
        out.push_str(&format!(
            "corruption replacements: {} uniform, {} self-conditioned\nexamples per level: {:?}\n",
            r.replaced_uniform, r.replaced_self, r.level_visits
        ));
        if let (Some(start), Some(end)) = (&r.val_start, &r.val_end) {
            out.push_str(&format!(
                "validation ({} windows, clean conditioning): weighted CE {:.4} -> {:.4} (uniform ln V = {:.4})\n\
                 level  len   CE start  CE end   acc end  unigram H\n",
                end.windows, start.weighted, end.weighted, end.uniform_baseline
            ));
            for k in 0..end.cross_entropy.len() {
                out.push_str(&format!(
                    "{:>5} {:>5} {:>9.4} {:>8.4} {:>8.3} {:>9.4}\n",
                    k,
                    self.lengths.get(k).copied().unwrap_or(0),
                    start.cross_entropy[k],
                    end.cross_entropy[k],
                    end.accuracy[k],
                    end.unigram_entropy[k]
                ));
            }
        }
        if let Some(hull) = &r.hull_end {
            out.push_str(&format!(
                "relaxed-state drift over {} positions (distance / median attractor spacing; min p10 p50 p90 max):\n\
                 to nearest attractor {:.3?}\nto soft mixture P·A   {:.3?}\n",
                hull.positions, hull.to_nearest, hull.to_mixture
            ));
        }
        if let Some(path) = &r.checkpoint {
            out.push_str(&format!("checkpoint: {}\n", path.display()));
        }
        out
    }
}

/// What `ncm generate` produced.
#[derive(Debug, Clone)]
pub struct GenerateSummary {
    pub generation: crate::ncm::generator::Generation,
    pub text: Vec<String>,
    /// `(level, step, text)` for every recorded thought.
    pub thoughts: Vec<(usize, usize, String)>,
    pub levels: usize,
    json: bool,
    prompt_bytes: usize,
    experimental: Option<ExperimentalDecodeReport>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodePolicy {
    decode: DecodeConfig,
    budget: DecodeBudget,
    /// Same generator weights read out at a shallower depth, including zero.
    draft_thinking_steps: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ExperimentalDecodeReport {
    #[serde(flatten)]
    counters: DecodeReport,
    target_trunk_evaluations: usize,
    draft_trunk_evaluations: usize,
}

impl GenerateSummary {
    pub fn render(&self) -> String {
        let g = &self.generation;
        if self.json {
            let mut report = serde_json::json!({
                "text": self.text,
                "prompt_bytes": self.prompt_bytes,
                "levels": self.levels,
                "level_calls": g.nfe_total,
                "trunk_evaluations": g.trunk_evals,
            });
            if let Some(experimental) = &self.experimental {
                report["decode"] = serde_json::json!(experimental);
            }
            return format!("{report}\n");
        }
        let mut out = format!(
            "NFE {} level calls per sample (K = {}), {} total over {} sample(s); trunk evaluations {}; \
             position-steps {} of budget {} ({} crystallised); codec reconstruct {} decode {}\n",
            g.nfe_per_sample,
            self.levels,
            g.nfe_total,
            g.samples,
            g.trunk_evals,
            g.position_steps,
            g.position_budget,
            g.crystallised,
            g.reconstruct_calls,
            g.decode_calls
        );
        if let Some(mbr) = &g.mbr {
            out.push_str(&format!(
                "MBR over {} samples (normalised Hamming medoid): chosen {:?}, mean distance {:.3?}\n",
                mbr.samples, mbr.chosen, mbr.mean_distance
            ));
        }
        if let Some(experimental) = &self.experimental {
            let counters = &experimental.counters;
            out.push_str(&format!(
                "decode used {} levels, {} particles, {} drafted calls, {} accepted levels, {} rejected levels, {} probes, budget stopped {}\n",
                counters.levels_used, counters.particles, counters.draft_calls,
                counters.accepted_drafts, counters.rejected_drafts, counters.probes,
                counters.stopped_by_budget
            ));
        }
        for (level, step, text) in &self.thoughts {
            out.push_str(&format!("thought level {level} step {step}: {text:?}\n"));
        }
        for text in &self.text {
            out.push_str("---\n");
            out.push_str(text);
            out.push_str("\n---\n");
        }
        out
    }
}

/// Load a tokenizer checkpoint with the architecture its training state
/// records, frozen.
pub fn load_tokenizer<B: burn::tensor::backend::Backend>(
    path: &std::path::Path,
    device: &B::Device,
) -> Result<crate::ncm::tokenizer::NcmTokenizer<B>> {
    use anyhow::Context as _;
    use burn::module::Module as _;
    let state = crate::checkpoint::TrainState::for_model(path)?
        .with_context(|| format!("no training state next to {}", path.display()))?;
    anyhow::ensure!(
        state.kind == crate::ncm::tokenizer::train::STATE_KIND,
        "{} was written by the '{}' trainer, not a tokenizer",
        path.display(),
        state.kind
    );
    state.verify_files(&crate::checkpoint::TrainState::dir_for(path))?;
    let config: NcmTokenizerConfig = serde_json::from_value(
        state
            .config
            .get("tokenizer")
            .cloned()
            .context("the training state records no tokenizer config")?,
    )
    .context("parse the recorded tokenizer config")?;
    let model = crate::ncm::tokenizer::NcmTokenizer::<B>::new(&config, device)?;
    crate::tensor_ext::force_initialization(&model);
    let model = crate::checkpoint::load::<B, _>(model, path, device)?;
    Ok(model.no_grad())
}

fn generator_config(args: &GeneratorTrainArgs) -> Result<crate::ncm::generator::NcmTrainConfig> {
    use crate::ncm::generator::{
        AdaptiveThinking, LossEstimator, LrShape, NcmGeneratorConfig, NcmTrainConfig,
    };
    let mut model = match NcmGeneratorConfig::preset(&args.config) {
        Ok(config) => config,
        Err(preset_err) => {
            let text = std::fs::read_to_string(&args.config).map_err(|err| {
                anyhow::anyhow!(
                    "{preset_err}; and reading {} as JSON failed: {err}",
                    args.config
                )
            })?;
            serde_json::from_str(&text)
                .map_err(|err| anyhow::anyhow!("parse {}: {err}", args.config))?
        }
    };
    if let Some(alpha) = args.alpha {
        model.alpha = alpha;
    }
    if let Some(steps) = args.thinking_steps {
        model.thinking_steps = steps;
    }
    if let Some(len) = args.prompt_len {
        model.prompt_len = len;
    }
    if let Some(weight) = args.deep_supervision {
        model.deep_supervision = weight;
    }
    if let Some(p) = args.self_prob {
        model.corruption.self_prob = p;
    }
    if let Some(confidence) = args.adaptive_confidence {
        model.adaptive = Some(AdaptiveThinking {
            confidence,
            budget_factor: args.adaptive_budget,
            max_steps_per_position: args.adaptive_cap.unwrap_or(2 * model.thinking_steps),
        });
    }
    let estimator = match args.estimator.as_str() {
        "sampled" => LossEstimator::SampledLevel,
        "all" => LossEstimator::AllLevels,
        other => anyhow::bail!("unknown estimator {other:?} (sampled, all)"),
    };
    let lr_shape = match (args.lr_schedule.as_str(), args.wsd_decay_start) {
        ("wsd", start) => LrShape::WarmupStableDecay {
            decay_start: start.unwrap_or_else(|| {
                args.steps
                    .saturating_sub((args.steps / 5).max(1))
                    .max(args.warmup)
            }),
        },
        ("cosine", None) => LrShape::WarmupCosine,
        _ => anyhow::bail!("--wsd-decay-start requires --lr-schedule wsd"),
    };
    let config = NcmTrainConfig {
        model,
        steps: args.steps,
        batch_size: args.batch_size,
        peak_lr: args.lr,
        min_lr: args.min_lr,
        warmup_steps: args.warmup,
        lr_shape,
        weight_decay: args.weight_decay,
        grad_clip: args.grad_clip,
        seed: args.seed,
        estimator,
        aligned_windows: args.aligned_windows,
        ordered_records: args.ordered_records,
        val_every: args.val_every,
        val_windows: args.val_windows,
        checkpoint_every: args.checkpoint_every,
        out_dir: Some(args.out.clone()),
    };
    config.validate()?;
    Ok(config)
}

fn train_generator_cli<B: Backend<FloatElem = f32>>(
    args: GeneratorTrainArgs,
    device: &B::Device,
    progress: &mut dyn FnMut(&str),
) -> Result<GeneratorTrainSummary> {
    use crate::ncm::generator::{
        resume_point, train_generator, CodecShape, NcmGenerator, NcmTrainData, TrainSession,
    };
    use crate::ncm::ConceptCodec as _;
    let config = generator_config(&args)?;
    let tokenizer = load_tokenizer::<Autodiff<B>>(&args.tokenizer, device)?;
    let tokenizer_sha256 = crate::checkpoint::file_sha256_hex(&args.tokenizer)?;
    let shape = CodecShape::of(&tokenizer)?;
    let (model, resume) = match &args.resume {
        Some(path) => {
            let (model, point) =
                resume_point::<Autodiff<B>>(path, &config, &shape, &tokenizer_sha256, device)?;
            (model, Some(point))
        }
        None => {
            B::seed(device, args.seed);
            (
                NcmGenerator::<Autodiff<B>>::new(&config.model, &shape, device)?,
                None,
            )
        }
    };
    progress(&format!(
        "concept-field generator {} params; tokenizer levels {:?}, codebook {}, latent {}",
        model.num_params(),
        shape.lengths,
        shape.codebook_size,
        shape.latent_dim
    ));
    let mut train = crate::corpus::TokenCorpus::streaming(&args.corpus)?;
    let mut val = args
        .val_corpus
        .as_ref()
        .map(|p| crate::corpus::TokenCorpus::streaming(p))
        .transpose()?;
    let data = NcmTrainData {
        train: &mut train,
        validation: val.as_mut(),
    };
    let session = TrainSession {
        tokenizer_sha256: tokenizer_sha256.clone(),
        resume,
    };
    let log_every = args.log_every.max(1);
    let (_, report) = train_generator(
        model,
        &tokenizer,
        data,
        &config,
        session,
        device,
        &mut |log| {
            if log.step % log_every == 0 || log.val.is_some() {
                progress(&format!(
                    "step {}: loss {:.4} lr {:.2e} grad {:.3}{} levels {:?}",
                    log.step,
                    log.loss,
                    log.lr,
                    log.grad_norm,
                    if log.clipped { " (clipped)" } else { "" },
                    log.level_rows
                ));
            }
            if let Some(val) = &log.val {
                progress(&format!(
                    "  val weighted CE {:.4}, per level {:.3?}",
                    val.weighted, val.cross_entropy
                ));
            }
        },
    )?;
    Ok(GeneratorTrainSummary {
        report,
        lengths: tokenizer.schedule().lengths().to_vec(),
        tokenizer_sha256,
    })
}

fn decode_with_policy<B: Backend<FloatElem = f32>>(
    model: &crate::ncm::generator::NcmGenerator<B>,
    tokenizer: &crate::ncm::tokenizer::NcmTokenizer<B>,
    prompt: Option<burn::tensor::Tensor<B, 2, burn::tensor::Int>>,
    relax: crate::ncm::generator::RelaxOptions,
    policy: DecodePolicy,
    device: &B::Device,
) -> Result<(crate::ncm::generator::Generation, ExperimentalDecodeReport)> {
    use crate::ncm::decode::Decoder;
    use crate::ncm::generator::{ConceptFieldLm, Generation, LevelTrace};
    use anyhow::ensure;

    ensure!(
        relax.steps > 0,
        "target generation needs at least one latent step"
    );
    ensure!(
        policy.decode.outline_cache.is_none(),
        "outline caching needs a persistent decoder and is unavailable for a single request"
    );
    ensure!(
        policy.decode.speculation.is_some() == policy.draft_thinking_steps.is_some(),
        "speculation and draft_thinking_steps must be set together"
    );
    let target = ConceptFieldLm::new(model, tokenizer, relax)?;
    let draft = policy
        .draft_thinking_steps
        .map(|depth| {
            ensure!(
                depth < relax.steps,
                "draft depth must be lower than target depth"
            );
            ConceptFieldLm::new(
                model,
                tokenizer,
                crate::ncm::generator::RelaxOptions {
                    steps: depth,
                    adaptive: None,
                    ..relax
                },
            )
        })
        .transpose()?;
    let mut decoder = Decoder::new(tokenizer, &target, device);
    if let Some(draft) = &draft {
        decoder = decoder.with_draft(draft);
    }
    let result = decoder.decode(prompt, 1, &policy.decode, &policy.budget, None)?;
    let target_traces = target.take_traces()?;
    let draft_traces = match &draft {
        Some(draft) => draft.take_traces()?,
        None => Vec::new(),
    };
    let trunk_calls =
        |traces: &[LevelTrace]| traces.iter().map(|trace| trace.usage.rounds).sum::<usize>();
    let target_trunk_evaluations = trunk_calls(&target_traces);
    let draft_trunk_evaluations = trunk_calls(&draft_traces);
    let usages = target_traces
        .iter()
        .chain(&draft_traces)
        .map(|trace| &trace.usage);
    let tokens = result
        .tokens
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(u32::try_from)
                .collect::<std::result::Result<_, _>>()
        })
        .collect::<std::result::Result<_, _>>()?;
    let codes = result
        .codes
        .into_iter()
        .map(|level| {
            level
                .into_iter()
                .flatten()
                .map(i64::try_from)
                .collect::<std::result::Result<_, _>>()
        })
        .collect::<std::result::Result<_, _>>()?;
    let report = result.report;
    Ok((
        Generation {
            tokens,
            codes,
            nfe_per_sample: report.level_calls,
            nfe_total: report.level_calls,
            samples: 1,
            trunk_evals: target_trunk_evaluations + draft_trunk_evaluations,
            position_steps: usages.clone().flat_map(|usage| &usage.position_steps).sum(),
            position_budget: usages.clone().flat_map(|usage| &usage.budget).sum(),
            crystallised: usages.flat_map(|usage| &usage.crystallised).sum(),
            reconstruct_calls: report.reconstructs,
            decode_calls: report.decodes,
            thoughts: Vec::new(),
            mbr: None,
        },
        ExperimentalDecodeReport {
            counters: report,
            target_trunk_evaluations,
            draft_trunk_evaluations,
        },
    ))
}

fn generate_cli<B: Backend<FloatElem = f32>>(
    args: GenerateArgs,
    device: &B::Device,
) -> Result<GenerateSummary> {
    use std::io::Read as _;

    use crate::ncm::generator::{
        generate, load_generator, thought_readout, CodeSampling, GenerateOptions,
    };
    use crate::ncm::ConceptCodec as _;
    let tokenizer = load_tokenizer::<B>(&args.tokenizer, device)?;
    let tokenizer_sha256 = crate::checkpoint::file_sha256_hex(&args.tokenizer)?;
    let loaded = load_generator::<B>(&args.generator, device)?;
    anyhow::ensure!(
        loaded.tokenizer_sha256 == tokenizer_sha256,
        "the generator was trained against tokenizer {}, not {}",
        loaded.tokenizer_sha256,
        tokenizer_sha256
    );
    let config = &loaded.spec.config;
    let input = if args.prompt_stdin {
        anyhow::ensure!(config.prompt_len > 0, "this generator has no prompt prefix");
        let mut text = String::new();
        std::io::stdin()
            .take(u64::try_from(config.prompt_len)?.saturating_add(1))
            .read_to_string(&mut text)?;
        anyhow::ensure!(
            text.len() <= config.prompt_len && !text.is_empty(),
            "stdin needs 1 to {} UTF-8 prompt bytes",
            config.prompt_len
        );
        Some(text)
    } else {
        args.prompt
    };
    let prompt = match &input {
        Some(text) if config.prompt_len > 0 => {
            let ids = crate::tokenizer::ByteTokenizer::new().encode(text);
            Some(crate::ncm::generator::prompt_tensor::<B>(
                &ids,
                config.prompt_len,
                config.pad_id,
                1,
                device,
            ))
        }
        Some(_) => anyhow::bail!("this generator was trained without a prompt (prompt_len 0)"),
        None => None,
    };
    let mut relax = config.relax();
    if let Some(steps) = args.thinking_steps {
        relax.steps = steps;
    }
    let options = GenerateOptions {
        codes: CodeSampling {
            temperature: args.temperature,
            top_k: args.top_k,
        },
        token_temperature: args.token_temperature,
        mbr_samples: args.mbr,
        seed: args.seed,
        relax,
        record_thoughts: args.show_thoughts,
    };
    let (generation, experimental) = if let Some(path) = args.decode_policy {
        use anyhow::Context as _;
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let policy = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse decode policy {}", path.display()))?;
        let (generation, report) =
            decode_with_policy(&loaded.model, &tokenizer, prompt, relax, policy, device)?;
        (generation, Some(report))
    } else {
        (
            generate(&loaded.model, &tokenizer, prompt, 1, &options)?,
            None,
        )
    };
    let bytes = crate::tokenizer::ByteTokenizer::new();
    let render = |ids: &[u32]| -> String {
        let ids: Vec<u16> = ids
            .iter()
            .map(|&t| u16::try_from(t).unwrap_or(u16::MAX))
            .collect();
        bytes.decode_lossy(&ids)
    };
    let mut thoughts = Vec::new();
    for level in &generation.thoughts {
        for (step, codes) in level.state_codes.iter().enumerate() {
            let text = thought_readout(
                &tokenizer,
                &generation.codes[..level.level],
                codes,
                1,
                device,
            )?;
            thoughts.push((
                level.level,
                step,
                text.first().map(|t| render(t)).unwrap_or_default(),
            ));
        }
    }
    let text = generation.tokens.iter().map(|t| render(t)).collect();
    Ok(GenerateSummary {
        generation,
        text,
        thoughts,
        levels: tokenizer.schedule().levels(),
        json: args.json,
        prompt_bytes: input
            .as_ref()
            .map_or(0, |text| text.len().min(config.prompt_len)),
        experimental,
    })
}
