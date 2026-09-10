//! `dblocks` CLI: train, sample, benchmark and verify DiffusionBlocks models.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

use diffusionblocks::{
    accuracy::{Ensemble, Guidance, LogitNorm, ScalingCurve, ScalingPoint},
    antipattern::{Labeler, RuleSet},
    checkpoint,
    corpus::{self, TokenCorpus},
    lm::{LanguageModel, LmConfig, Sampling, Unlikelihood},
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
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f64,
        /// Unlikelihood coefficient on labeled targets (roadmap Phase 24).
        /// 0 trains plainly; requires labels when positive.
        #[arg(long, default_value_t = 0.0)]
        penalty: f32,
        /// Stream windows from disk instead of loading the corpus.
        #[arg(long, default_value_t = false)]
        streaming: bool,
        /// The small configuration, for CPU smoke runs.
        #[arg(long, default_value_t = false)]
        tiny: bool,
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
        /// Append one record per variant here.
        #[arg(long)]
        json: Option<PathBuf>,
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
        /// The small configuration -- must match the inputs'.
        #[arg(long, default_value_t = false)]
        tiny: bool,
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
        /// The small configuration -- must match the checkpoint's.
        #[arg(long, default_value_t = false)]
        tiny: bool,
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
        Command::Experts { action } => cmd_experts(action),
        Command::Verify { group } => cmd_verify(group.as_deref()),
        command @ Command::Train { .. } => cmd_train(command),
        command @ Command::Sample { .. } => cmd_sample(command),
        Command::Bench { model, num_inference_steps, batch_size, repeats, warmup, json } => {
            cmd_bench(model, num_inference_steps, batch_size, repeats, warmup, json)
        }
        Command::Sweep { grid, seeds, steps, dataset, data_dir, batch_size, json } => {
            cmd_sweep(&grid, &seeds, steps, &dataset, data_dir, batch_size, json)
        }
        Command::Audit { action } => cmd_audit(action),
        Command::Merge { model, input, weights, out } => cmd_merge(model, input, &weights, out),
        Command::Experiment { action } => cmd_experiment(action),
        Command::Infer { model, batch_size, num_inference_steps, top_k, solver } => {
            cmd_infer(model, batch_size, num_inference_steps, top_k, &solver)
        }
    }
}

/// The rule set a `--rules` flag names, or the built-in one.
fn load_labeler(rules: Option<&Path>) -> Result<Labeler> {
    match rules {
        Some(path) => Labeler::new(RuleSet::read(path)?),
        None => Labeler::builtin(),
    }
}

fn cmd_lm(action: LmAction) -> Result<()> {
    match action {
        LmAction::Tokenize { input, out, label, rules } => {
            let count = TokenCorpus::tokenize_file(&input, &out)?;
            println!(
                "{} -> {} | {count} tokens ({} bytes, u16 little-endian)",
                input.display(),
                out.display(),
                count * corpus::TOKEN_BYTES
            );
            if label {
                let manifest = TokenCorpus::label_file(&out, &load_labeler(rules.as_deref())?)?;
                println!("labels -> {}\n{}", corpus::labels_path(&out).display(), manifest.render());
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
                println!("labels: none (`dblocks lm label --corpus {}` to add them)", path.display());
            }
            Ok(())
        }
        LmAction::Label { corpus: path, rules } => {
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
        LmAction::Scan { input, rules } => {
            let text = std::fs::read_to_string(&input)
                .map_err(|err| anyhow::anyhow!("read {}: {err}", input.display()))?;
            let labeler = load_labeler(rules.as_deref())?;
            let report = labeler.report(&text);
            if report.is_empty() {
                println!("{}: no findings", input.display());
            } else {
                print!("{report}");
                println!("{} finding(s) in {}", report.lines().count(), input.display());
            }
            Ok(())
        }
        LmAction::Score { input, language, lexical, structural, external, out } => {
            use diffusionblocks::codequality::{
                CodeAnalyzer, CompositeAnalyzer, ExternalAnalyzer as ExtTool,
                Language as Lang, QualityScore, StructuralAnalyzer,
            };
            let text = std::fs::read_to_string(&input)
                .map_err(|err| anyhow::anyhow!("read {}: {err}", input.display()))?;
            let lang = Lang::parse(&language);
            let lexical_labeler = if lexical { Some(load_labeler(None)?) } else { None };
            let structural_analyzer = if structural { Some(StructuralAnalyzer::default()) } else { None };
            let external_analyzer = external.as_ref().map(|tool| {
                // The default invocation is a placeholder: users who care
                // about this dimension configure their own command via the
                // analyzer API. The CLI exists so the dimension is reachable
                // without writing Rust code.
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
            steps,
            batch_size,
            lr,
            weight_decay,
            penalty,
            streaming,
            tiny,
            seed,
            log_every,
            log,
            out_dir,
            checkpoint_every,
            resume,
        } => {
            type Train = train::DefaultTrainBackend;
            let device: <Train as burn::tensor::backend::BackendTypes>::Device = Default::default();
            <Train as burn::tensor::backend::Backend>::seed(&device, seed);

            let corpus_path = corpus_paths[0].clone();
            let mut corpora: Vec<TokenCorpus> = Vec::with_capacity(corpus_paths.len());
            let mut any_labels = false;
            for path in &corpus_paths {
                let mut corpus = if streaming { TokenCorpus::streaming(path)? } else { TokenCorpus::in_memory(path)? };
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
                        if penalty > 0.0 { format!("alpha={penalty}") } else { "off (measuring only)".into() }
                    );
                    any_labels = true;
                } else {
                    println!("labels ({}): none", path.display());
                }
                corpora.push(corpus);
            }
            if penalty > 0.0 && !any_labels {
                anyhow::bail!(
                    "--penalty {penalty} needs labels; run `dblocks lm label --corpus {}` first",
                    corpus_path.display()
                );
            }
            let weights = diffusionblocks::mix::MixWeights::parse(&corpus_weights, corpora.len())?;
            let mix_mode = diffusionblocks::mix::MixMode::parse(&mix)?;
            let mut mix = diffusionblocks::mix::CorpusMix::new(corpora.iter_mut().collect(), weights, mix_mode)?;

            let model_config = if tiny { LmConfig::tiny() } else { LmConfig::default() };
            let model = LanguageModel::<Train>::new(&model_config, &device);
            println!(
                "training: {} tokens in {} corpus(es) ({}) | context={} layers={} hidden={} | steps={steps} batch={batch_size} lr={lr}",
                mix.total_tokens(),
                mix.len(),
                mix.mode().name(),
                model_config.context,
                model_config.num_layers,
                model_config.hidden_size
            );
            let teacher_weights = parse_weights(&teacher_weights)?;
            let mut inputs = train::LmTrainInputs::<Train>::default();
            for (i, path) in teacher.iter().enumerate() {
                let loaded = checkpoint::load::<Train, _>(LanguageModel::<Train>::new(&model_config, &device), path, &device)?;
                let w = teacher_weights.get(i).copied().unwrap_or(1.0);
                println!("teacher {}: {} (weight {w})", i + 1, path.display());
                inputs.teachers.push((loaded, w));
            }
            if let Some(path) = &negative_teacher {
                inputs.negative_teacher =
                    Some(checkpoint::load::<Train, _>(LanguageModel::<Train>::new(&model_config, &device), path, &device)?);
                println!("negative teacher: {} (confidence >= {negative_confidence}, penalty {negative_penalty})", path.display());
            }

            let config = train::LmTrainConfig {
                steps,
                batch_size,
                lr,
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
                distill_weight: if teacher.is_empty() { 0.0 } else { distill_weight },
                distill_temperature,
                negative_confidence,
                negative_penalty: if negative_teacher.is_some() { negative_penalty } else { 0.0 },
            };
            let (model, report) = train::train_lm_mixed(model, &mut mix, &inputs, &config, &device)?;
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
                    report.penalized_tokens, report.first_penalized_prob, report.last_penalized_prob
                );
            }
            if report.negative_teacher_tokens > 0 {
                println!(
                    "negative teacher: {} proposals charged | p {:.4} -> {:.4}",
                    report.negative_teacher_tokens, report.first_negative_teacher_prob, report.last_negative_teacher_prob
                );
            }
            if report.last_distill_loss > 0.0 {
                println!("distillation term at the last step: {:.4}", report.last_distill_loss);
            }
            let path = match report.checkpoint {
                Some(path) => path,
                None => checkpoint::save_content_addressed(model, &out_dir, "lm")?,
            };
            println!("checkpoint saved: {} (training state beside it)", path.display());
            Ok(())
        }
        LmAction::Policy { action } => cmd_policy(action),
        LmAction::Approvals { action } => cmd_approvals(action),
        LmAction::RefusalCorpus { policy, prompts, answers, out } => {
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
            let tokens: Vec<u16> = docs.iter().flat_map(|d| tokenizer.encode_document(d)).collect();
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
        LmAction::Merge { input, weights, out, tiny } => {
            let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
            let config = if tiny { LmConfig::tiny() } else { LmConfig::default() };
            let template = LanguageModel::<Eval>::new(&config, &device);
            let weights = if weights.trim().is_empty() { vec![1.0; input.len()] } else { parse_weights(&weights)? };
            let (merged, parents) = diffusionblocks::merge::merge_checkpoints::<Eval, _>(template, &input, &weights, &device)?;
            let path = checkpoint::save_content_addressed(merged, &out, "lm")?;
            write_merge_state(&path, &parents, &weights, &input)?;
            println!("merged {} -> {}", diffusionblocks::merge::describe(&parents, &weights), path.display());
            Ok(())
        }
        LmAction::Bench { corpus: corpus_path, steps, seeds, batch_size, json } => {
            use diffusionblocks::experiment::{Record, RunLog};
            use diffusionblocks::vit::MoeTrunkConfig;
            type Train = train::DefaultTrainBackend;
            let device: <Train as burn::tensor::backend::BackendTypes>::Device = Default::default();
            let seeds = parse_seeds(&seeds)?;
            let mut corpus = TokenCorpus::in_memory(&corpus_path)?;
            let moe = MoeTrunkConfig { num_experts: 3, top_k: 1, every_n_layers: 2, z_level: 1e-3, balance_bias: false };
            let variants: Vec<(&str, LmConfig, f32)> = vec![
                ("dense", LmConfig::tiny(), 0.0),
                ("moe", LmConfig { moe: Some(moe), ..LmConfig::tiny() }, 0.0),
                ("moe+bias", LmConfig { moe: Some(MoeTrunkConfig { balance_bias: true, ..moe }), ..LmConfig::tiny() }, 1e-3),
            ];
            println!("{:<10} {:>6} {:>12} {:>12} {:>10}", "variant", "seeds", "final loss", "±ci95", "ms/step");
            println!("{}", "-".repeat(54));
            let mut records = Vec::new();
            for (name, model_config, rate) in variants {
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
                    let model = LanguageModel::<Train>::new(&model_config, &device);
                    let cfg = train::LmTrainConfig { seed, ..train_config.clone() };
                    let (_, report) = train::train_lm(model, &mut corpus, &cfg, &device)?;
                    record.push(f64::from(report.last_loss), false);
                    ms_per_step.push(1e3 * report.elapsed_secs / report.steps_taken.max(1) as f64);
                }
                let mean_ms = ms_per_step.iter().sum::<f64>() / ms_per_step.len() as f64;
                record.extra = serde_json::json!({ "ms_per_step": ms_per_step, "forward_passes_per_token": 1 });
                let summary = record.summary.context("no seed produced a measurement")?;
                println!(
                    "{:<10} {:>6} {:>12.4} {:>12.4} {:>10.1}",
                    name,
                    summary.n,
                    summary.mean,
                    if summary.ci95_half_width.is_nan() { 0.0 } else { summary.ci95_half_width },
                    mean_ms
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
            tiny,
            policy,
            key,
            grant,
        } => {
            let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
            <Eval as burn::tensor::backend::Backend>::seed(&device, seed);

            let config = if tiny { LmConfig::tiny() } else { LmConfig::default() };
            let mut model = LanguageModel::<Eval>::new(&config, &device);
            if let Some(path) = &weights {
                model = checkpoint::load::<Eval, _>(model, path, &device)?;
                println!("loaded {}", path.display());
            }
            let tokenizer = ByteTokenizer::new();

            if let Some(policy_path) = &policy {
                use diffusionblocks::policy::{gated_generate, Decision, Grant, Key, Policy};
                let policy = Policy::read(policy_path)?;
                let grants: Vec<Grant> = grant.iter().map(|p| Grant::read(p)).collect::<Result<_>>()?;
                let approved = match (&key, grants.is_empty()) {
                    (Some(key_path), false) => {
                        let key = Key::read(key_path)?;
                        let approved = policy.approved_scopes(&key, &grants, checkpoint::unix_now());
                        for g in &grants {
                            match g.verify(&key, &policy, checkpoint::unix_now()) {
                                Ok(()) => println!("grant {}: valid for {}", g.approval.id, g.approval.scopes.join(",")),
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
                        println!("refused before any forward pass: blocker {blocker} (scope {scope})")
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
                Err(index_err) => {
                    let spec = MosmeSpec::read(&index).with_context(|| {
                        format!("{} is neither an expert index ({index_err:#}) nor a spec", index.display())
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
                .map_err(|payload| anyhow::anyhow!("checkpoint save thread panicked: {}", panic_message(&payload)))??,
        }
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
        match summary.checkpoint {
            Some(path) => path,
            None => {
                println!("note: --async-save writes the weights only, without a training state");
                checkpoint::save_content_addressed_async(model, out_path, "dblocks")
                    .join()
                    .map_err(|payload| anyhow::anyhow!("checkpoint save thread panicked: {}", panic_message(&payload)))??
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
            let mut record = Record::new(format!("bench/{scope}"), "ms", bench_config.clone(), vec![model_args.seed]);
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

            let (logits, stats) = last.context("no repeat ran: --repeats and --warmup left nothing to measure")?;
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

            let timing = profiler.stats(&scope).with_context(|| format!("no timing recorded for scope {scope:?}"))?;
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
    println!("(± is the half-width of the 95% t-interval over {} measured repeat(s))", repeats.max(1));

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
        let mut record = Record::new(format!("bench/scaling/sequential/steps={steps}"), "accuracy", bench_config.clone(), vec![model_args.seed]);
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
            budget: Budget { max_evaluations: 48, max_depth: depth, beam_width: 3 },
            solver: SolverKind::Euler,
            max_steps: 6,
            ..PlannedConfig::default()
        };
        let (logits, stats, _) = model.sample_planned(&batch.pixel_values, &config, &mut rng);
        let acc = diffusionblocks::accuracy::accuracy(&logits, &batch.labels);
        let mut record = Record::new(format!("bench/scaling/planned/depth={depth}"), "accuracy", bench_config.clone(), vec![model_args.seed]);
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
        println!("\n{} experiment record(s) appended to {}", records.len(), path.display());
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
        .map(|s| s.trim().parse::<f64>().map_err(|e| anyhow::anyhow!("weight {s:?}: {e}")))
        .collect()
}

fn parse_seeds(text: &str) -> Result<Vec<u64>> {
    let seeds: Vec<u64> = text
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u64>().map_err(|e| anyhow::anyhow!("seed {s:?}: {e}")))
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
        json.as_ref().map(|p| format!(" -> {}", p.display())).unwrap_or_default()
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
            println!("policy {} | key id {} | {} blocker(s) | {} revoked grant(s)", policy.display(), p.key_id, p.blockers.len(), p.revoked.len());
            for b in &p.blockers {
                println!("- {} [{}] {:?}: {} pattern(s); refusal: {:?}", b.id, b.scope, b.applies_to, b.patterns.len(), b.refusal);
                for pat in &b.patterns {
                    println!("    {pat}");
                }
            }
            for g in &p.revoked {
                println!("revoked: {g}");
            }
            Ok(())
        }
        PolicyAction::AddBlocker { policy, id, scope, pattern, applies_to, refusal, description } => {
            let mut p = Policy::read(&policy)?;
            p.add_blocker(Blocker { id: id.clone(), scope, description, patterns: pattern, applies_to: Applies::parse(&applies_to)?, refusal })?;
            p.write(&policy)?;
            println!("blocker {id} added to {}", policy.display());
            Ok(())
        }
        PolicyAction::RemoveBlocker { policy, id } => {
            let mut p = Policy::read(&policy)?;
            let removed = p.remove_blocker(&id)?;
            p.write(&policy)?;
            println!("blocker {} [{}] removed from {}", removed.id, removed.scope, policy.display());
            Ok(())
        }
        PolicyAction::Check { policy, prompt, key, grant } => {
            let p = Policy::read(&policy)?;
            let grants: Vec<Grant> = grant.iter().map(|g| Grant::read(g)).collect::<Result<_>>()?;
            let approved = match (&key, grants.is_empty()) {
                (Some(key_path), false) => p.approved_scopes(&Key::read(key_path)?, &grants, checkpoint::unix_now()),
                (None, false) => anyhow::bail!("--grant needs --key to verify against"),
                _ => Vec::new(),
            };
            let hits = p.hits(&prompt, false);
            if hits.is_empty() {
                println!("no blocker fires");
            }
            for h in &hits {
                println!("blocker {} [{}] fires at bytes {}..{}", h.blocker, h.scope, h.start, h.end);
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
        ApprovalsAction::Issue { key, policy, id, scopes, expires, note, out } => {
            let k = Key::read(&key)?;
            let p = Policy::read(&policy)?;
            anyhow::ensure!(p.key_id == k.id(), "key {} is not the policy's key ({})", k.id(), p.key_id);
            let approval = Approval {
                id: id.clone(),
                scopes: scopes.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
                issued_unix: checkpoint::unix_now(),
                expires_unix: expires,
                note,
            };
            let unknown: Vec<&String> = approval.scopes.iter().filter(|s| *s != "*" && !p.scopes().contains(s)).collect();
            if !unknown.is_empty() {
                println!("note: scope(s) {:?} have no blocker in this policy", unknown);
            }
            let grant = Grant::issue(&k, approval)?;
            grant.write(&out)?;
            println!("grant {id} for {} written to {} (expires {expires})", scopes, out.display());
            Ok(())
        }
        ApprovalsAction::Verify { key, policy, grant } => {
            let g = Grant::read(&grant)?;
            match g.verify(&Key::read(&key)?, &Policy::read(&policy)?, checkpoint::unix_now()) {
                Ok(()) => {
                    println!("grant {} is valid for {} until {}", g.approval.id, g.approval.scopes.join(","), g.approval.expires_unix);
                    Ok(())
                }
                Err(err) => anyhow::bail!("{err}"),
            }
        }
    }
}

fn cmd_merge(model_args: ModelArgs, input: Vec<PathBuf>, weights: &str, out: PathBuf) -> Result<()> {
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let template = ModelArgs { checkpoint: None, ..model_args.clone() }.build(None)?;
    let weights = if weights.trim().is_empty() { vec![1.0; input.len()] } else { parse_weights(weights)? };
    let (merged, parents) = diffusionblocks::merge::merge_checkpoints::<Eval, _>(template, &input, &weights, &device)?;
    let path = checkpoint::save_content_addressed(merged, &out, "dblocks")?;
    write_merge_state(&path, &parents, &weights, &input)?;
    println!("merged {} -> {}", diffusionblocks::merge::describe(&parents, &weights), path.display());
    Ok(())
}

/// A state directory for a merged model recording its parents, so the
/// provenance chain does not stop at the merge.
fn write_merge_state(path: &Path, parents: &[String], weights: &[f64], inputs: &[PathBuf]) -> Result<()> {
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
        AuditAction::Propagation { model: model_args, batch_size, epsilon, json } => {
            let model = model_args.build(None)?;
            let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
            let mut rng = StdRng::seed_from_u64(model_args.seed);
            let mut dataset = SyntheticDataset::new(model_args.image_size, model_args.num_labels, batch_size, model_args.seed);
            let batch = dataset.next_batch(&mut rng, &device)?;
            let report = diffusionblocks::audit::propagation(&model, &batch.pixel_values, &batch.labels, epsilon);
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
            println!("{:<40} {:>6} {:>4} {:>4} {:>14} {:>12}", "name", "unit", "n", "warm", "mean ± ci95", "median");
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
                        format!("{:.4}±{:.4}", s.mean, if s.ci95_half_width.is_nan() { 0.0 } else { s.ci95_half_width }),
                        s.median
                    ),
                    None => println!("{:<40} {:>6} {:>4} {:>4} {:>14} {:>12}", r.name, r.unit, 0, warm, "-", "-"),
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
            anyhow::ensure!(!rows.is_empty(), "no record names in common between {} and {}", a.display(), b.display());
            print!("{}", render_comparison(&rows));
            println!("\noverlap = the two 95% intervals overlap: not evidence of no difference, only of not enough trials to show one.");
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
