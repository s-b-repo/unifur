//! Training-step probe: wall time per optimizer step of the specialist
//! trainer at a scheduler file's student shape, on CPU and (with the
//! feature) on wgpu. A measurement, not a promise — the sibling of
//! `gpu_probe.rs`, which covered inference forwards.
//!
//! ```text
//! cargo run --release --features wgpu --example train_probe -- \
//!     --scheduler /srv/m-sda/unifur/scheduler-coding.json \
//!     --dataset /path/to/traces.jsonl --language rust --steps 5
//! ```
//!
//! Per backend the probe runs the real trainer
//! (`student::train_language_specialist`) three times on the same seed — a
//! discarded warm-up (shader compilation, lazy backend setup land there),
//! then `--steps 1` and `--steps N` — so the marginal per-step time
//! `(T_N - T_1) / (N - 1)` is the steady-state step, with the trainer's own
//! average printed alongside.
//! No checkpoints are written (`out_dir: None`); the dataset never leaves
//! the disk it started on. `--init <checkpoint.mpk>` starts every backend
//! from the same weights (backend RNG streams differ, so a seeded fresh
//! init is a *different* model per backend); two backends from one init are
//! the cross-backend numerics check.
//!
//! Measured on this box (RTX 3060 12 GB, burn 0.21 cubecl/wgpu over Vulkan,
//! driver 615.71.09): the production student shape (`scheduler-coding.json`:
//! hidden 512, 8 layers, ctx 4096, 10 coding boxes) does not survive on the
//! *plain* wgpu backend — batch 2 OOMs, and batch 1 pegs VRAM at 97% and
//! returns *deterministically wrong* numbers (completion CE pinned near
//! ln(vocab), cubecl's memory pool asserting on the device thread). Burn
//! 0.21's `BalancedCheckpointing` autodiff strategy (`--backend wgpu-ckpt`,
//! or any scheduler with `gpu_checkpointing: true`) recomputes the ops the
//! backend marks memory-bound during the backward and makes batch 1 *correct*
//! — step-0 loss matching NdArray to 1e-4 from one init checkpoint — but
//! batch 2 still OOMs the pool. The shipped GPU shape is therefore batch 2
//! run as 2 x 1 with gradient accumulation (`micro_batch: 1` in
//! [`nina::SchedulerConfig`]): 6.2 s/step against 93.3 s on CPU, peak 11.9
//! GB, zero pool assertions across the probe and a full scratch drain
//! (training, checkpoint, eval, registry). Every smaller shape checked
//! (dense and `3:1` attention at ctx 4096 narrow, full width at ctx 512, the
//! field shape at ctx 1024, the tiny shape) matches NdArray step for step
//! without checkpointing. Trust a wgpu run only after an `--init`
//! cross-check at the same shape; speed where correct was 6-22x per step.
//!
//! `--save-init <dir>` writes the scheduler's student built on CPU at the
//! scheduler's seed into `<dir>` and exits — the checkpoint every `--init`
//! cross-check starts from.
//!
//! Re-measured 2026-10-15 against the production coding shape
//! (`scheduler-coding-gpu.json`, `pipeline-gpu/rust`, 1812 examples x 4096),
//! from one CPU-built init checkpoint, 8 steps on each backend:
//!
//! ```text
//!   step   wgpu-ckpt    ndarray      diff
//!      0      5.9913     5.9913     0.0000
//!      1      4.8981     4.8983     0.0002
//!      3      4.1384     4.1391     0.0007
//!      4      4.2040     4.2019     0.0021   <- worst
//!      7      3.8949     3.8952     0.0003
//! ```
//!
//! Step 0 — the one that cannot be explained by accumulation — agrees to four
//! decimals, and the worst divergence over the run is 2.1e-3 on a loss of 4.2
//! (5e-4 relative). That is float32 reduction-order noise between backends and
//! nothing else, so a wgpu drain's numbers are comparable to a CPU drain's.
//!
//! Marginal step time was 14.671 s against 139.281 s: **9.5x**. The figures
//! above it in this file (6.2 s vs 93.3 s) were taken on an idle box; the ones
//! here ran alongside a full-corpus ingest holding a core, which inflates the
//! ndarray side and so makes the ratio a conservative one.

use anyhow::{Context, Result};
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use diffusionblocks::nina::SchedulerConfig;
use diffusionblocks::student::{
    CodeLanguage, GeometricStudent, LanguageBatch, SpecialistTrainConfig,
};
use std::path::{Path, PathBuf};

/// One measured run: total trainer seconds for `steps` steps.
struct RunOutcome {
    backend: &'static str,
    steps: usize,
    elapsed_secs: f64,
    first_loss: f32,
    last_loss: f32,
}

/// Train `steps` steps of the scheduler's student shape on backend `B` and
/// return what the trainer measured. The model is rebuilt and reseeded every
/// call, so two calls with different step counts are the same run truncated.
fn measured_run<B: AutodiffBackend<FloatElem = f32>>(
    backend: &'static str,
    scheduler: &SchedulerConfig,
    batch: &LanguageBatch,
    language: Option<CodeLanguage>,
    init: Option<&Path>,
    steps: usize,
) -> Result<RunOutcome> {
    let device: <B as BackendTypes>::Device = Default::default();
    <B as Backend>::seed(&device, scheduler.seed);
    let mut model = GeometricStudent::<B>::new(&scheduler.student, &device)
        .with_context(|| format!("build the student on {backend}"))?;
    diffusionblocks::tensor_ext::force_initialization(&model);
    if let Some(path) = init {
        // Identical starting weights on every backend: the divergence a run
        // reports is then the backend's arithmetic, not its init stream.
        model = diffusionblocks::checkpoint::load::<B, _>(model, path, &device)
            .with_context(|| format!("load the init checkpoint {} on {backend}", path.display()))?;
    }
    let train_config = SpecialistTrainConfig {
        steps,
        batch_size: scheduler.batch_size,
        lr: scheduler.lr,
        seed: scheduler.seed,
        log_every: 1,
        out_dir: None,
        language,
        router_aux_weight: scheduler.router_aux_weight,
        micro_batch: scheduler.micro_batch,
        ..SpecialistTrainConfig::default()
    };
    let (_model, report) =
        diffusionblocks::student::train_language_specialist(model, batch, &train_config, &device)
            .with_context(|| format!("train {steps} step(s) on {backend}"))?;
    Ok(RunOutcome {
        backend,
        steps,
        elapsed_secs: report.elapsed_secs,
        first_loss: report.first_loss,
        last_loss: report.last_loss,
    })
}

/// The wgpu run, when the feature built it in. `checkpointed` selects Burn
/// 0.21's `BalancedCheckpointing` autodiff strategy: ops the backend marks
/// memory-bound (the cheap elementwise ones — gelu, exp, scalar arithmetic)
/// are recomputed during the backward instead of their outputs staying
/// resident. Matmul and attention intermediates are compute-bound and stay;
/// this is the only checkpointing burn 0.21 ships, and it is a *backend type*,
/// so the choice lives here rather than in a serde config.
#[cfg(feature = "wgpu")]
fn run_wgpu(
    scheduler: &SchedulerConfig,
    batch: &LanguageBatch,
    language: Option<CodeLanguage>,
    init: Option<&Path>,
    steps: usize,
    checkpointed: bool,
) -> Result<RunOutcome> {
    if checkpointed {
        measured_run::<
            burn::backend::Autodiff<
                burn::backend::Wgpu<f32>,
                burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
            >,
        >("wgpu-ckpt", scheduler, batch, language, init, steps)
    } else {
        measured_run::<burn::backend::Autodiff<burn::backend::Wgpu<f32>>>(
            "wgpu", scheduler, batch, language, init, steps,
        )
    }
}

#[cfg(not(feature = "wgpu"))]
fn run_wgpu(
    _scheduler: &SchedulerConfig,
    _batch: &LanguageBatch,
    _language: Option<CodeLanguage>,
    _init: Option<&Path>,
    _steps: usize,
    _checkpointed: bool,
) -> Result<RunOutcome> {
    anyhow::bail!(
        "--backend wgpu needs the `wgpu` cargo feature, which this binary was built without; \
         rebuild with `cargo build --release --features wgpu --example train_probe`"
    )
}

fn main() -> Result<()> {
    let mut scheduler_path: Option<PathBuf> = None;
    let mut dataset_path: Option<PathBuf> = None;
    let mut language: Option<CodeLanguage> = None;
    let mut init_path: Option<PathBuf> = None;
    let mut save_init_dir: Option<PathBuf> = None;
    let mut steps = 5usize;
    let mut backends = vec!["cpu", "wgpu"];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scheduler" => {
                scheduler_path = Some(PathBuf::from(
                    args.next().context("--scheduler needs a path")?,
                ));
            }
            "--dataset" => {
                dataset_path = Some(PathBuf::from(
                    args.next().context("--dataset needs a path")?,
                ));
            }
            "--language" => {
                let raw = args.next().context("--language needs a name")?;
                language = Some(CodeLanguage::parse(&raw).with_context(|| "--language")?);
            }
            "--init" => {
                init_path = Some(PathBuf::from(
                    args.next()
                        .context("--init needs a checkpoint (.mpk) path")?,
                ));
            }
            "--save-init" => {
                save_init_dir = Some(PathBuf::from(
                    args.next()
                        .context("--save-init needs a directory to write the checkpoint into")?,
                ));
            }
            "--steps" => {
                let raw = args.next().context("--steps needs a count")?;
                steps = raw
                    .parse::<usize>()
                    .with_context(|| format!("--steps '{raw}' is not a positive integer"))?;
                anyhow::ensure!(steps >= 2, "--steps must be at least 2, got {steps}");
            }
            "--backend" => {
                let raw = args
                    .next()
                    .context("--backend needs cpu, wgpu, wgpu-ckpt or both")?;
                backends = match raw.as_str() {
                    "cpu" => vec!["cpu"],
                    "wgpu" => vec!["wgpu"],
                    "wgpu-ckpt" => vec!["wgpu-ckpt"],
                    "both" => vec!["cpu", "wgpu"],
                    other => anyhow::bail!(
                        "unknown backend '{other}' (want cpu, wgpu, wgpu-ckpt or both)"
                    ),
                };
            }
            other => anyhow::bail!(
                "unknown argument '{other}' (want --scheduler, --dataset, --language, --init, \
                 --save-init, --steps, --backend)"
            ),
        }
    }
    let scheduler_path = scheduler_path.context("--scheduler <config.json> is required")?;
    let text = std::fs::read_to_string(&scheduler_path)
        .with_context(|| format!("read scheduler config {}", scheduler_path.display()))?;
    let scheduler: SchedulerConfig = serde_json::from_str(&text)
        .with_context(|| format!("parse scheduler config {}", scheduler_path.display()))?;
    scheduler
        .validate()
        .context("the scheduler config is not valid")?;

    if let Some(dir) = save_init_dir {
        // The CPU build of the scheduler's student at the scheduler's seed,
        // saved once so `--init` can give every backend the same weights.
        type Cpu = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
        let device: <Cpu as BackendTypes>::Device = Default::default();
        <Cpu as Backend>::seed(&device, scheduler.seed);
        let model = GeometricStudent::<Cpu>::new(&scheduler.student, &device)
            .context("build the student for --save-init")?;
        diffusionblocks::tensor_ext::force_initialization(&model);
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let path = diffusionblocks::checkpoint::save_content_addressed(model, &dir, "student")
            .with_context(|| format!("save the init checkpoint into {}", dir.display()))?;
        println!("init checkpoint: {}", path.display());
        return Ok(());
    }

    let dataset_path = dataset_path.context("--dataset <traces.jsonl> is required")?;
    let batch = LanguageBatch::load(
        &dataset_path,
        language,
        scheduler.student.context,
        scheduler.seed,
    )
    .with_context(|| format!("load {}", dataset_path.display()))?;
    println!(
        "train probe: {} examples x {} tokens, student hidden {} layers {} batch {}",
        batch.len(),
        scheduler.student.context,
        scheduler.student.hidden_size,
        scheduler.student.num_layers,
        scheduler.batch_size
    );

    for name in &backends {
        // The scheduler's `gpu_checkpointing` flag selects the checkpointed
        // autodiff strategy for the wgpu arm, so the probe measures the
        // configuration the drain will actually run; `wgpu-ckpt` forces it
        // on for a scheduler that does not set it (the comparison runs that
        // motivated the flag).
        let checkpointed = scheduler.gpu_checkpointing;
        let run = |steps: usize| -> Result<RunOutcome> {
            match *name {
                "cpu" => measured_run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
                    "ndarray",
                    &scheduler,
                    &batch,
                    language,
                    init_path.as_deref(),
                    steps,
                ),
                "wgpu" => run_wgpu(
                    &scheduler,
                    &batch,
                    language,
                    init_path.as_deref(),
                    steps,
                    checkpointed,
                ),
                _ => run_wgpu(
                    &scheduler,
                    &batch,
                    language,
                    init_path.as_deref(),
                    steps,
                    true,
                ),
            }
        };
        // A discarded warm-up run pays the one-time costs (shader
        // compilation, lazy buffers) so both timed runs are warm and the
        // marginal subtraction is meaningful.
        let _warm = run(1)?;
        let one = run(1)?;
        let full = run(steps)?;
        let marginal = (full.elapsed_secs - one.elapsed_secs) / (steps as f64 - 1.0);
        println!(
            "  {:8} 1 step {:9.3}s | {} steps {:9.3}s (avg {:8.3}s/step) | marginal {:8.3}s/step | loss {:.4} -> {:.4}",
            one.backend,
            one.elapsed_secs,
            full.steps,
            full.elapsed_secs,
            full.elapsed_secs / full.steps as f64,
            marginal,
            full.first_loss,
            full.last_loss,
        );
    }
    println!(
        "note: a discarded warm-up run pays the one-time costs (shader compilation, lazy \
         buffers) before the timed runs; `marginal` = (T_N - T_1)/(N-1) is the steady-state step."
    );
    Ok(())
}
