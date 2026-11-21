//! Compare equal per-rank gradients with a whole-batch native NCM update.
//! This CPU probe exercises synchronization in one process with clean
//! conditioning, a frozen codec, and one optimizer.

use anyhow::{anyhow, ensure, Context, Result};
use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::Backend, backend::BackendTypes, Int, Tensor};
use clap::Parser;
use diffusionblocks::{
    checkpoint,
    corpus::TokenCorpus,
    ncm::{
        cli::load_tokenizer,
        generator::{weighted_ncm_loss, CodecShape, NcmGenerator, NcmGeneratorConfig},
        tokenizer::NcmTokenizer,
        ConceptCodec,
    },
    schedule::GradientAccumulator,
};
use serde_json::json;
use std::{path::PathBuf, thread};

type Cpu = NdArray<f32>;
type Train = Autodiff<Cpu>;
type Device = <Train as BackendTypes>::Device;
// Batched and per-rank matmuls use different FP32 reduction orders. The
// geometric distance derivative includes terms that can cancel.
const GRADIENT_TOLERANCE: f64 = 1e-3;
const LOSS_TOLERANCE: f64 = 1e-5;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    codec: PathBuf,
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long, default_value_t = 29)]
    prompt_width: usize,
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u16).range(2..=40))]
    workers: u16,
    #[arg(long, default_value_t = 2)]
    steps: usize,
}

struct Objective {
    codec: NcmTokenizer<Train>,
    config: NcmGeneratorConfig,
    weights: Vec<f64>,
    device: Device,
}

impl Objective {
    fn grad(
        &self,
        model: &NcmGenerator<Train>,
        records: &[u16],
        scale: f64,
    ) -> Result<(f32, GradientsParams)> {
        let prefix = self.config.prompt_len;
        let target = self.codec.schedule().seq_len();
        let batch = records.len() / (prefix + target);
        let mut prompts = Vec::with_capacity(batch * prefix);
        let mut targets = Vec::with_capacity(batch * target);
        for record in records.chunks_exact(prefix + target) {
            let (prompt, response) = record.split_at(prefix);
            prompts.extend(prompt.iter().copied().map(i64::from));
            targets.extend(response.iter().copied().map(i64::from));
        }
        let tokens = |ids: &[i64], width| {
            Tensor::<Train, 1, Int>::from_ints(ids, &self.device).reshape([batch, width])
        };
        let prompt = tokens(&prompts, prefix);
        let codes = self.codec.encode(tokens(&targets, target))?.codes;
        let relax = self.config.relax();
        let mut logits = Vec::with_capacity(codes.len());
        for level in 0..codes.len() {
            let input = self.codec.reconstruct(&codes[..level], batch)?.detach();
            let output = model.forward_level(&self.codec, level, input, Some(&prompt), &relax)?;
            ensure!(
                output.usage.rounds == self.config.thinking_steps,
                "latent depth changed"
            );
            logits.push(output.logits);
        }
        let loss = weighted_ncm_loss(&logits, &codes, &self.weights)?;
        let value = loss.clone().into_scalar();
        ensure!(value.is_finite(), "nonfinite native loss");
        let grad = GradientsParams::from_grads(loss.mul_scalar(scale).backward(), model);
        Ok((value, grad))
    }
}

struct GradientComparison<'a> {
    whole: &'a GradientsParams,
    reduced: &'a GradientsParams,
    signal: f64,
    squared_error: f64,
    compared: usize,
    missing: usize,
}

impl ModuleVisitor<Train> for GradientComparison<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Train, D>>) {
        let whole = self.whole.get::<Cpu, D>(param.id);
        let reduced = self.reduced.get::<Cpu, D>(param.id);
        match (whole, reduced) {
            (Some(whole), Some(reduced)) => {
                self.compared += 1;
                let delta = whole.clone() - reduced;
                self.squared_error += f64::from(delta.clone().mul(delta).sum().into_scalar());
                self.signal += f64::from(whole.clone().mul(whole).sum().into_scalar());
            }
            (None, None) => {}
            _ => self.missing += 1,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let workers = usize::from(args.workers);
    ensure!(
        args.prompt_width > 0 && args.steps > 0,
        "require a prompt and at least one step"
    );
    let device = Default::default();
    let codec = load_tokenizer::<Train>(&args.codec, &device)?;
    let config = NcmGeneratorConfig {
        blocks: 1,
        hidden: 32,
        prompt_len: args.prompt_width,
        alpha: 1.0,
        ..NcmGeneratorConfig::tiny()
    };
    let span = config
        .prompt_len
        .checked_add(codec.schedule().seq_len())
        .context("record width exceeds usize")?;
    let batch_tokens = span.checked_mul(workers).context("batch is too large")?;
    let mut corpus = TokenCorpus::streaming(&args.corpus)?;
    ensure!(
        corpus.len().is_multiple_of(span),
        "corpus does not contain whole records"
    );
    ensure!(
        args.steps <= corpus.len() / batch_tokens,
        "need {} records per step",
        workers
    );
    let weights = codec.schedule().loss_weights(config.alpha)?;
    Train::seed(&device, 42);
    let mut model = NcmGenerator::<Train>::new(&config, &CodecShape::of(&codec)?, &device)?;
    let objective = Objective {
        codec,
        config,
        weights,
        device,
    };
    let mut optimizer = AdamWConfig::new().with_beta_2(0.95).init();
    let initial = checkpoint::canonical_hash_hex(&model);
    let mut reports = Vec::new();
    for step in 0..args.steps {
        let records = corpus.window(step * batch_tokens, batch_tokens)?;
        ensure!(
            records
                .iter()
                .all(|&token| usize::from(token) < objective.codec.vocab_size()),
            "corpus token exceeds the codec vocabulary"
        );
        let (whole_loss, whole) = objective.grad(&model, &records, 1.0)?;
        let mut accumulator = GradientAccumulator::new(workers);
        let scale = accumulator.loss_scale();
        let mut mean_loss = 0.0;
        let mut reduced = None;
        thread::scope(|scope| -> Result<()> {
            let mut pending = Vec::with_capacity(workers);
            for record in records.chunks_exact(span) {
                let model = &model;
                let objective = &objective;
                pending.push(scope.spawn(move || objective.grad(model, record, scale)));
            }
            for worker in pending {
                let (loss, grad) = worker.join().map_err(|_| anyhow!("local rank failed"))??;
                mean_loss += f64::from(loss) * scale;
                if let Some(grad) = accumulator.fold::<Train, _>(grad, &model).into_gradients() {
                    reduced = Some(grad);
                }
            }
            Ok(())
        })?;
        let reduced = reduced.context("gradient synchronization did not finish")?;
        let mut check = GradientComparison {
            whole: &whole,
            reduced: &reduced,
            signal: 0.0,
            squared_error: 0.0,
            compared: 0,
            missing: 0,
        };
        model.visit(&mut check);
        let residual = (check.squared_error / check.signal).sqrt();
        let loss_delta = (f64::from(whole_loss) - mean_loss).abs();
        ensure!(check.missing == 0 && check.compared > 0 && residual <= GRADIENT_TOLERANCE && loss_delta <= LOSS_TOLERANCE,
            "rank synchronization mismatch at step {step}: relative gradient error {residual}, loss delta {loss_delta}, missing gradients {}", check.missing);
        reports.push(json!({
            "step": step + 1, "loss": mean_loss, "full_batch_loss": whole_loss,
            "absolute_loss_error": loss_delta, "relative_gradient_l2_error": residual,
            "parameter_tensors_compared": check.compared,
        }));
        model = optimizer.step(3e-4, model, reduced);
    }
    let final_hash = checkpoint::canonical_hash_hex(&model);
    ensure!(
        initial != final_hash,
        "the synchronized updates did not change the weights"
    );
    println!(
        "{}",
        json!({
            "workers": workers, "records_per_step": workers,
            "optimizer_steps": args.steps, "thinking_steps_per_level": objective.config.thinking_steps,
            "levels": objective.codec.schedule().levels(),
            "mode": "same_host_cpu_thread_ranks_frozen_codec_clean_conditioning",
            "generator_parameters": model.num_params(),
        "relative_gradient_l2_tolerance": GRADIENT_TOLERANCE, "absolute_loss_tolerance": LOSS_TOLERANCE,
            "initial_parameter_sha256": initial, "final_parameter_sha256": final_hash,
            "steps": reports,
        })
    );
    Ok(())
}
