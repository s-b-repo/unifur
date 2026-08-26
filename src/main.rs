//! `dblocks` CLI: train and sample with DiffusionBlocks.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::Path;

use diffusionblocks::{
    checkpoint,
    data::{SyntheticDataset, TrainDataset},
    dblock::{DblockClassifier, DblockConfig},
    sigma,
    train::{self, train_synthetic, TrainConfig},
    vit::ViTDiTConfig,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

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

#[derive(Subcommand)]
enum Command {
    /// Block-wise training on a dataset.
    Train {
        /// Dataset name (only `synthetic` for now; cifar100 coming).
        #[arg(long, default_value = "synthetic")]
        dataset: String,
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
    },
    /// Run diffusion sampling and print predictions for one batch.
    Sample {
        #[arg(long, default_value_t = 32)]
        image_size: usize,
        #[arg(long, default_value_t = 10)]
        num_labels: usize,
        #[arg(long, default_value_t = 12)]
        num_hidden_layers: usize,
        #[arg(long, default_value_t = 4)]
        num_blocks: usize,
        #[arg(long, default_value_t = 3)]
        num_inference_steps: usize,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// ODE solver: euler | heun | ddim | dpmpp2m
        #[arg(long, default_value = "euler")]
        solver: String,
        /// Multi-block strategy: sequential | parallel | hybrid | adaptive
        #[arg(long, default_value = "sequential")]
        strategy: String,
        /// Parallel span width for parallel/hybrid strategies.
        #[arg(long, default_value_t = 2)]
        k: usize,
    },
    /// Time every solver on the same random-weights model and report
    /// agreement statistics (Phase 4.7 benchmark).
    BenchSolvers {
        #[arg(long, default_value_t = 32)]
        image_size: usize,
        #[arg(long, default_value_t = 10)]
        num_labels: usize,
        #[arg(long, default_value_t = 12)]
        num_hidden_layers: usize,
        #[arg(long, default_value_t = 4)]
        num_blocks: usize,
        #[arg(long, default_value_t = 8)]
        num_inference_steps: usize,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Print the block sigma schedule for inspection.
    Sigmas {
        #[arg(long, default_value_t = 3)]
        num_blocks: usize,
        #[arg(long, default_value_t = 0.05)]
        gamma: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Sigmas { num_blocks, gamma } => {
            let sampler = sigma::DblockSigmaSampler::new(num_blocks, gamma);
            println!("block boundaries (ascending):");
            for (i, s) in sampler.block_sigmas.iter().enumerate() {
                println!("  [{i}] {s:.6}");
            }
            for b in 0..num_blocks {
                let (lo, hi) = sampler.extended_window(b);
                println!("block {b}: extended window [{lo:.6}, {hi:.6}]");
            }
            Ok(())
        }
        Command::Train {
            dataset,
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
        } => {
            if dataset != "synthetic" {
                anyhow::bail!("dataset '{dataset}' not supported yet; use 'synthetic'");
            }
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
                log_file: log_file.map(std::path::PathBuf::from),
            };
            let out_path = Path::new(&out_dir).to_path_buf();
            if grad_checkpointing {
                let model = train::train_synthetic_generic::<
                    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
                >(&config)?;
                let handle = checkpoint::save_content_addressed_async(model, out_path, "dblocks");
                println!("checkpoint saved (async): {}", handle.join().expect("save thread")?.display());
            } else {
                let model = train_synthetic(&config)?;
                let path = if async_save {
                    let handle =
                        checkpoint::save_content_addressed_async(model, out_path, "dblocks");
                    handle.join().expect("save thread")?
                } else {
                    checkpoint::save_content_addressed(model, &out_path, "dblocks")?
                };
                println!("checkpoint saved: {}", path.display());
            }
            Ok(())
        }
        Command::Sample {
            image_size,
            num_labels,
            num_hidden_layers,
            num_blocks,
            num_inference_steps,
            batch_size,
            seed,
            solver,
            strategy,
            k,
        } => {
            type B = burn::backend::NdArray<f32>;
            let device = Default::default();
            <B as burn::tensor::backend::Backend>::seed(&device, seed);
            let mut vit_cfg = ViTDiTConfig::with_image_size(image_size, num_labels);
            vit_cfg.num_hidden_layers = num_hidden_layers;
            let dblock_cfg = DblockConfig {
                num_blocks,
                num_inference_steps: Some(num_inference_steps),
                ..DblockConfig::default()
            };
            let model = DblockClassifier::<B>::new(&vit_cfg, &dblock_cfg, &device);

            // Deterministic random images + labels.
            let mut rng = StdRng::seed_from_u64(seed);
            let mut dataset = SyntheticDataset::new(image_size, num_labels, batch_size, seed);
            let batch = dataset.next_batch(&mut rng, &device);

            let solver_kind = diffusionblocks::solver::SolverKind::parse(&solver)?;
            let inner = match strategy.as_str() {
                "sequential" => diffusionblocks::multi_block::Strategy::Sequential,
                "parallel" => diffusionblocks::multi_block::Strategy::Parallel { k },
                "hybrid" => diffusionblocks::multi_block::Strategy::Hybrid { k, warmup_frac: 0.3 },
                "adaptive" => {
                    diffusionblocks::multi_block::Strategy::Adaptive { k_max: k.max(1), conf_threshold: 1.0 }
                }
                other => anyhow::bail!("unknown strategy '{other}'"),
            };
            let mb_config = diffusionblocks::multi_block::MultiBlockConfig {
                strategy: diffusionblocks::multi_block::Gated {
                    inner,
                    gate: diffusionblocks::quality::QualityGateConfig::strict(),
                },
                solver: solver_kind,
                num_steps: Some(num_inference_steps),
            };
            let (logits, stats) = model.sample_multi_block(&batch.pixel_values, &mb_config, &mut rng);
            let pred = logits.argmax(1).squeeze_dim::<1>(1);

            let preds = pred.into_data().convert::<i64>();
            let truth: Vec<i64> = batch.labels.into_data().convert::<i64>().iter().collect();

            println!(
                "solver={} strategy={} schedule (descending): {:?}",
                solver_kind.name(),
                strategy,
                model.inference_sigmas()
            );
            println!(
                "model calls: {} | gated samples: {}",
                stats.model_calls, stats.gated_samples
            );
            println!("predicted vs true:");
            let pv: Vec<i64> = preds.iter().collect();
            let correct = pv.iter().zip(&truth).filter(|(a, b)| a == b).count();
            for (i, (p, t)) in pv.iter().zip(&truth).enumerate() {
                println!("  sample {i}: pred={p} true={t}");
            }
            println!(
                "top-1 agreement with random labels: {correct}/{}",
                truth.len()
            );
            Ok(())
        }
        Command::BenchSolvers {
            image_size,
            num_labels,
            num_hidden_layers,
            num_blocks,
            num_inference_steps,
            batch_size,
            seed,
        } => {
            type B = burn::backend::NdArray<f32>;
            let device = Default::default();
            <B as burn::tensor::backend::Backend>::seed(&device, seed);
            let mut vit_cfg = ViTDiTConfig::with_image_size(image_size, num_labels);
            vit_cfg.num_hidden_layers = num_hidden_layers;
            let dblock_cfg =
                DblockConfig { num_blocks, ..DblockConfig::default() };
            let model = DblockClassifier::<B>::new(&vit_cfg, &dblock_cfg, &device);

            let mut rng = StdRng::seed_from_u64(seed);
            let mut dataset = SyntheticDataset::new(image_size, num_labels, batch_size, seed);
            let batch = dataset.next_batch(&mut rng, &device);

            println!(
                "{:<10} {:>10} {:>12} {:>14}",
                "solver", "ms", "model_calls", "argmax_sum"
            );
            for name in ["euler", "heun", "ddim", "dpmpp2m"] {
                let kind = diffusionblocks::solver::SolverKind::parse(name)?;
                let start = std::time::Instant::now();
                let logits = model.solve(
                    &batch.pixel_values,
                    kind,
                    num_inference_steps,
                    None,
                    &mut rng,
                );
                let elapsed = start.elapsed().as_millis();
                let pred = logits.argmax(1).squeeze_dim::<1>(1).into_data().convert::<i64>();
                let argmax_sum: i64 = pred.iter::<i64>().sum();
                println!("{name:<10} {elapsed:>10} {num:>12} {argmax_sum:>14}", num = num_inference_steps + 1);
            }
            Ok(())
        }
    }
}
