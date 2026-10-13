//! Drain the NINA training queue with an explicit scheduler configuration.
//!
//! `dblocks nina mcp-serve` trains with `SchedulerConfig::default()`; the
//! field specialists need a config that names their routing boxes and the
//! production student shape, which lives in a JSON file on the external
//! drive. This example is the loop the MCP `nina.train.tick` tool runs,
//! repeated until the queue is empty:
//!
//! ```text
//! cargo run --release --example queue_runner -- \
//!     --root /srv/m-sda/unifur --scheduler /srv/m-sda/unifur/scheduler-field.json
//! cargo run --release --features wgpu --example queue_runner -- \
//!     --root /srv/m-sda/unifur --scheduler /srv/m-sda/unifur/scheduler-coding.json \
//!     --backend wgpu
//! ```
//!
//! `--backend cpu` (the default) runs [`nina::train_tick`], bit for bit the
//! CPU path the MCP tool runs. `--backend wgpu` runs the same queue
//! semantics through [`nina::train_tick_with`] on `Autodiff<Wgpu<f32>>`; it
//! needs the `wgpu` cargo feature and fails loudly when the binary was built
//! without it. Cross-backend runs are not bit-identical (reduction orders
//! differ), so an entry's recorded numbers belong to the backend that
//! produced them.
//!
//! Each finished entry prints one JSON line (the `QueueEntryOutcome`, ok or
//! failed), so the run doubles as a progress log. A failed entry does not
//! stop the queue — it is recorded `queue.failed` by the scheduler and the
//! loop continues, exactly like the MCP tool.

use anyhow::{Context, Result};
use diffusionblocks::nina::{self, QueueEntryOutcome, SchedulerConfig};
use std::path::{Path, PathBuf};

/// Which device family the training slices run on.
enum BackendChoice {
    /// The CPU [`nina::train_tick`] path.
    Cpu,
    /// `Autodiff<Wgpu<f32>>` through [`nina::train_tick_with`].
    Wgpu,
}

/// Drain the queue through `tick`, printing each entry's outcome as one JSON
/// line. Returns the drained count.
fn drain(
    root: &Path,
    scheduler: &SchedulerConfig,
    mut tick: impl FnMut(&Path, &SchedulerConfig) -> Result<Option<QueueEntryOutcome>>,
) -> Result<usize> {
    let mut drained = 0usize;
    while let Some(outcome) = tick(root, scheduler)? {
        println!(
            "{}",
            serde_json::to_string(&outcome).context("render the outcome as JSON")?
        );
        drained += 1;
    }
    Ok(drained)
}

/// The wgpu arm, when the feature built it in. The scheduler's
/// `gpu_checkpointing` flag selects Burn 0.21's `BalancedCheckpointing`
/// autodiff strategy (memory-bound ops recomputed in the backward) — a
/// backend *type*, so the choice is made here, at instantiation. Measured on
/// this box (RTX 3060 12 GB, cubecl/wgpu over Vulkan): at the coding shape
/// (ctx 4096, hidden 512, 8 layers) the plain wgpu backend corrupts under
/// memory pressure (cubecl pool assertion, losses pegged at ln(vocab)) while
/// the checkpointed one matches the CPU step for step.
#[cfg(feature = "wgpu")]
fn drain_wgpu(root: &Path, scheduler: &SchedulerConfig) -> Result<usize> {
    if scheduler.gpu_checkpointing {
        drain(
            root,
            scheduler,
            nina::train_tick_with::<
                burn::backend::Autodiff<
                    burn::backend::Wgpu<f32>,
                    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
                >,
            >,
        )
    } else {
        drain(
            root,
            scheduler,
            nina::train_tick_with::<burn::backend::Autodiff<burn::backend::Wgpu<f32>>>,
        )
    }
}

/// Without the feature the arm cannot run; the failure says how to build it.
#[cfg(not(feature = "wgpu"))]
fn drain_wgpu(_root: &Path, _scheduler: &SchedulerConfig) -> Result<usize> {
    anyhow::bail!(
        "--backend wgpu needs the `wgpu` cargo feature, which this binary was built without; \
         rebuild with `cargo build --release --features wgpu --example queue_runner`"
    )
}

fn main() -> Result<()> {
    let mut root = PathBuf::from(nina::DEFAULT_ROOT);
    let mut scheduler_path: Option<PathBuf> = None;
    let mut backend = BackendChoice::Cpu;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                root = PathBuf::from(args.next().context("--root needs a path")?);
            }
            "--scheduler" => {
                scheduler_path = Some(PathBuf::from(
                    args.next().context("--scheduler needs a path")?,
                ));
            }
            "--backend" => {
                let value = args.next().context("--backend needs cpu or wgpu")?;
                backend = match value.as_str() {
                    "cpu" => BackendChoice::Cpu,
                    "wgpu" => BackendChoice::Wgpu,
                    other => anyhow::bail!("unknown backend '{other}' (want cpu or wgpu)"),
                };
            }
            other => {
                anyhow::bail!("unknown argument '{other}' (want --root, --scheduler, --backend)")
            }
        }
    }
    let scheduler_path = scheduler_path.context("--scheduler <config.json> is required")?;
    let text = std::fs::read_to_string(&scheduler_path)
        .with_context(|| format!("read scheduler config {}", scheduler_path.display()))?;
    let scheduler: SchedulerConfig = serde_json::from_str(&text)
        .with_context(|| format!("parse scheduler config {}", scheduler_path.display()))?;

    let drained = match backend {
        BackendChoice::Cpu => drain(&root, &scheduler, nina::train_tick)?,
        BackendChoice::Wgpu => drain_wgpu(&root, &scheduler)?,
    };
    println!(
        "queue empty: {drained} entr{} drained",
        if drained == 1 { "y" } else { "ies" }
    );
    Ok(())
}
