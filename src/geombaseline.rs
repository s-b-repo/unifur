//! A matched-depth transformer baseline for the geometric reasoner (plan
//! Part 2): the control arm for the open claim in `docs/Claims.md` that
//! attractor relaxation beats a matched-depth transformer on real geometric
//! questions.
//!
//! The comparison is only fair if everything except the reasoning mechanism
//! is held constant, so the baseline shares with [`crate::geometry::GeometricReasoner`]:
//!
//! - the same scene tokens and the same token embedding (same initializer),
//! - **no position table** -- the reasoner uses none ("the fixed scene layout
//!   is the position system"), so the baseline gets none either,
//! - the same `hidden_size`, and a **matched sequential depth**: the
//!   reasoner's forward answer path runs `num_blocks` refinement blocks, each
//!   applying `refine_steps` relaxation iterations, and every iteration
//!   transforms the symbolic stream once (one geodesic-attention read plus
//!   one writer update). The baseline therefore gets
//!   `num_layers = num_blocks * refine_steps` pre-norm transformer layers,
//!   each transforming the stream once (attention + MLP). At the shipped
//!   defaults that is 2 x 2 = 4 layers.
//! - a non-causal read of the whole scene -- the reasoner's geodesic
//!   attention is also non-causal,
//! - the answer readout at the last token (the `?` position) with the same
//!   declared answer-set restriction, scored by literally the same metric
//!   (the reasoner's answer cross-entropy),
//! - the same corpus, batching, optimizer, learning-rate schedule, clipping
//!   and held-out evaluation protocol ([`train_geom_baseline`] /
//!   [`evaluate_geom_baseline`] mirror `train_geom` / `evaluate_geom` on the
//!   answer objective).
//!
//! What the baseline deliberately does NOT have: the learned metric, the
//! geometric stream and its certified energy relaxation, the block-diffusion
//! machinery, and the MoSME readout. It is one linear read head over a plain
//! transformer trunk. The MLP width (4x hidden) is chosen so the total
//! parameter count lands within a few percent of the reasoner's -- both
//! counts are printed by `geom baseline` and `geom eval --baseline`, which is
//! where a mismatch would be caught and adjusted.

use burn::module::{Initializer, Module};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::checkpoint;
use crate::corpus::TokenCorpus;
use crate::geometry::{
    answer_ce, sample_aligned_batch, GeomConfig, GeomMeta, GeomStepMetrics, GeomTrainConfig,
    GeomTrainReport,
};

/// Configuration for a [`GeomBaseline`], derived from a [`GeomConfig`] so the
/// match is by construction rather than by a second, driftable flag set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeomBaselineConfig {
    pub vocab_size: usize,
    pub scene_len: usize,
    pub hidden_size: usize,
    /// `num_blocks * refine_steps` of the reasoner config -- see the module
    /// docs for the depth-matching formula.
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub initializer_range: f64,
    /// The closed answer set, as declared by the corpus metadata. Empty means
    /// "score the whole byte vocabulary", the same convention as the
    /// reasoner.
    pub answer_tokens: Vec<u16>,
}

impl GeomBaselineConfig {
    /// The baseline matched to a reasoner config: same scene width, same
    /// hidden size, `num_blocks * refine_steps` layers, four attention heads,
    /// and a 4x MLP. The 4x width is the parameter-count match: with it the
    /// baseline's total lands within a few percent of the reasoner's (whose
    /// MoSME readout heads dominate its count), well inside the 25% band the
    /// comparison requires; the CLI prints both counts so a config change
    /// that breaks the match is visible.
    pub fn matched_to(config: &GeomConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            scene_len: config.scene_len,
            hidden_size: config.hidden_size,
            num_layers: config.num_blocks * config.refine_steps,
            num_heads: 4,
            intermediate_size: 4 * config.hidden_size,
            layer_norm_eps: config.layer_norm_eps,
            initializer_range: config.initializer_range,
            answer_tokens: config.answer_tokens.clone(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.hidden_size >= self.num_heads && self.hidden_size % self.num_heads == 0,
            "hidden size {} must be a multiple of the {} heads",
            self.hidden_size,
            self.num_heads
        );
        anyhow::ensure!(
            self.num_layers >= 1,
            "the baseline needs at least one layer"
        );
        anyhow::ensure!(
            self.intermediate_size >= 1,
            "the MLP width must be positive"
        );
        Ok(())
    }

    /// One line naming the architecture, for banners and records.
    pub fn describe(&self) -> String {
        format!(
            "transformer baseline: layers={} hidden={} heads={} mlp={} scene={}",
            self.num_layers,
            self.hidden_size,
            self.num_heads,
            self.intermediate_size,
            self.scene_len
        )
    }
}

/// One pre-norm transformer layer: multi-head self-attention (non-causal)
/// and a GELU MLP, both residual.
#[derive(Module, Debug)]
struct BaselineBlock<B: Backend> {
    norm_attn: LayerNorm<B>,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    attn_out: Linear<B>,
    norm_mlp: LayerNorm<B>,
    fc_in: Linear<B>,
    fc_out: Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend<FloatElem = f32>> BaselineBlock<B> {
    fn new(config: &GeomBaselineConfig, device: &B::Device) -> Self {
        let head_dim = config.hidden_size / config.num_heads;
        Self {
            norm_attn: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            query: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            key: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            value: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            attn_out: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            norm_mlp: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            fc_in: LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(true)
                .init(device),
            fc_out: LinearConfig::new(config.intermediate_size, config.hidden_size)
                .with_bias(true)
                .init(device),
            num_heads: config.num_heads,
            head_dim,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, n, _] = x.dims();
        let hn = self.norm_attn.forward(x.clone());
        let split = |t: Tensor<B, 3>| {
            t.reshape([batch, n, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let q = split(self.query.forward(hn.clone()));
        let k = split(self.key.forward(hn.clone()));
        let v = split(self.value.forward(hn));
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((self.head_dim as f32).sqrt());
        let attn = softmax(scores, 3).matmul(v);
        let attn = attn
            .swap_dims(1, 2)
            .reshape([batch, n, self.num_heads * self.head_dim]);
        let x = x.add(self.attn_out.forward(attn));
        let hn = self.norm_mlp.forward(x.clone());
        x.add(
            self.fc_out
                .forward(crate::tensor_ext::exact_gelu(self.fc_in.forward(hn))),
        )
    }
}

/// A plain pre-norm transformer encoder over the same scenes the
/// [`crate::geometry::GeometricReasoner`] reads, matched in depth, width,
/// data and objective (see the module docs). No metric, no relaxation, no
/// routing.
#[derive(Module, Debug)]
pub struct GeomBaseline<B: Backend> {
    token_embedding: Embedding<B>,
    layers: Vec<BaselineBlock<B>>,
    final_norm: LayerNorm<B>,
    readout: Linear<B>,
    scene_len: usize,
    hidden_size: usize,
    num_layers: usize,
    vocab_size: usize,
    /// The declared answer set, empty to score the whole vocabulary. Not a
    /// parameter: it is part of the task, exactly as on the reasoner.
    answer_tokens: Vec<u16>,
}

impl<B: Backend<FloatElem = f32>> GeomBaseline<B> {
    /// Build the baseline. Fallsible for the same reason the reasoner's
    /// constructor is: an invalid config has to name the field that is out of
    /// range instead of aborting the run.
    pub fn new(config: &GeomBaselineConfig, device: &B::Device) -> anyhow::Result<Self> {
        config.validate()?;
        Ok(Self {
            // The same initializer choice the reasoner makes.
            token_embedding: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: config.initializer_range,
                })
                .init(device),
            layers: (0..config.num_layers)
                .map(|_| BaselineBlock::new(config, device))
                .collect(),
            final_norm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            readout: LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: config.initializer_range,
                })
                .with_bias(true)
                .init(device),
            scene_len: config.scene_len,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            vocab_size: config.vocab_size,
            answer_tokens: config.answer_tokens.clone(),
        })
    }

    pub fn scene_len(&self) -> usize {
        self.scene_len
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Answer logits `[batch, vocab]` at the query position (the last token
    /// the model reads: the `?` before the answer), exactly as the reasoner's
    /// `answer_logits` reads them.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [batch, n] = tokens.dims();
        let mut h = self.token_embedding.forward(tokens);
        for layer in &self.layers {
            h = layer.forward(h);
        }
        let query = self
            .final_norm
            .forward(h.narrow(1, n - 1, 1))
            .reshape([batch, self.hidden_size]);
        self.readout.forward(query)
    }

    /// Predicted answer bytes for `logits` `[batch, vocab]`, read over the
    /// declared answer set when one exists (the same restriction training
    /// and evaluation use). Mirrors `GeometricReasoner::predict_answers`.
    pub fn predict_answers(&self, logits: &Tensor<B, 2>) -> Vec<u16> {
        let device = logits.device();
        let batch = logits.dims()[0];
        let (scored, set_vec): (Tensor<B, 2>, Vec<i64>) = match self.answer_tokens.is_empty() {
            false => (
                crate::geometry::gather_answer_columns(logits, &self.answer_tokens, &device),
                self.answer_tokens.iter().map(|t| i64::from(*t)).collect(),
            ),
            true => (logits.clone(), (0..self.vocab_size as i64).collect()),
        };
        let idx: Vec<i64> = scored
            .argmax(1)
            .reshape([batch, 1])
            .into_data()
            .convert::<i64>()
            .iter::<i64>()
            .collect();
        idx.into_iter()
            .map(|i| set_vec[i as usize] as u16)
            .collect()
    }

    /// The clean answer objective, identical to the reasoner's answer
    /// objective minus the router terms the baseline does not have:
    /// cross-entropy on the answer token, read from the query position,
    /// scored by the same answer cross-entropy the reasoner uses.
    pub fn answer_step(&self, tokens_full: Tensor<B, 2, Int>) -> (Tensor<B, 1>, GeomStepMetrics) {
        let [batch, len] = tokens_full.dims();
        assert_eq!(
            len, self.scene_len,
            "a scene is {} tokens, got {}",
            self.scene_len, len
        );
        let tokens = tokens_full.clone().narrow(1, 0, self.scene_len - 1);
        let answers = tokens_full
            .narrow(1, self.scene_len - 1, 1)
            .reshape([batch]);
        let logits = self.forward(tokens);
        let (ce, ce_value, accuracy) =
            answer_ce(&logits, &answers, &self.answer_tokens, self.vocab_size);
        (
            ce,
            GeomStepMetrics {
                loss: ce_value,
                accuracy,
                energy: 0.0,
                displacement: 0.0,
                consistency: 0.0,
                balance: 0.0,
                load_entropy: 0.0,
                token_entropy: 0.0,
                scenes: batch,
            },
        )
    }
}

/// Train the baseline on an aligned corpus, mirroring [`crate::geometry::train_geom`]
/// on the answer objective: the same corpus reader, the same batch sampling,
/// the same optimizer, schedule, clipping, accumulation and EMA settings from
/// the shared [`GeomTrainConfig`], and the same content-addressed checkpoint
/// (prefix `geom-baseline`). The diffusion-only fields of the config
/// (`objective`, `gamma`, `consistency_weight`, `moe_balance_weight`) do not
/// apply to a model without sigmas or a router.
pub fn train_geom_baseline<B: AutodiffBackend<FloatElem = f32>>(
    mut model: GeomBaseline<B>,
    corpus: &mut TokenCorpus,
    meta: &GeomMeta,
    config: &GeomTrainConfig,
    device: &B::Device,
) -> anyhow::Result<(GeomBaseline<B>, GeomTrainReport)> {
    anyhow::ensure!(
        config.steps > 0 && config.batch_size > 0,
        "steps and batch_size must be positive"
    );
    anyhow::ensure!(
        meta.scene_len == model.scene_len(),
        "the metadata's scene length {} does not match the model's {}",
        meta.scene_len,
        model.scene_len()
    );
    anyhow::ensure!(
        corpus.len() % meta.scene_len == 0,
        "the corpus holds {} tokens, not a multiple of the {}-token scene length",
        corpus.len(),
        meta.scene_len
    );
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let lr_schedule = match &config.lr_schedule {
        crate::schedule::LrSchedule::Constant { .. } => {
            crate::schedule::LrSchedule::Constant { lr: config.lr }
        }
        crate::schedule::LrSchedule::WarmupCosine {
            min_lr,
            warmup_steps,
            ..
        } => crate::schedule::LrSchedule::WarmupCosine {
            peak: config.lr,
            min_lr: *min_lr,
            warmup_steps: *warmup_steps,
            total_steps: config.steps,
        },
        crate::schedule::LrSchedule::WarmupConstant { warmup_steps, .. } => {
            crate::schedule::LrSchedule::WarmupConstant {
                peak: config.lr,
                warmup_steps: *warmup_steps,
            }
        }
    };
    let mut rng = ChaCha12Rng::seed_from_u64(config.seed);
    let mut logger = config
        .log_path
        .as_ref()
        .map(|path| crate::logging::MetricsLogger::open(path))
        .transpose()?;
    let mut accumulator = crate::schedule::GradientAccumulator::new(config.accumulate);
    let mut ema = config
        .ema_decay
        .map(|d| crate::schedule::Ema::new(&model, d));
    let mut report = GeomTrainReport {
        steps_clipped: 0,
        steps_taken: 0,
        steps_skipped: 0,
        steps_accumulated: 0,
        first_loss: f32::NAN,
        last_loss: f32::NAN,
        mean_loss: 0.0,
        first_accuracy: f32::NAN,
        last_accuracy: f32::NAN,
        mean_accuracy: 0.0,
        energy_start: f32::NAN,
        energy_end: f32::NAN,
        displacement_end: f32::NAN,
        converged_steps: 0,
        last_consistency: 0.0,
        load_entropy_end: 0.0,
        token_entropy_end: 0.0,
        balance_end: 0.0,
        scenes_seen: 0,
        elapsed_secs: 0.0,
        checkpoint: None,
    };
    let mut loss_sum = 0.0f64;
    let mut accuracy_sum = 0.0f64;
    let started = std::time::Instant::now();
    for step in 0..config.steps {
        <B as burn::tensor::backend::Backend>::seed(
            device,
            crate::train::step_seed(config.seed, step),
        );
        let windows = sample_aligned_batch(corpus, meta.scene_len, config.batch_size, &mut rng)?;
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| i64::from(*t)))
            .collect();
        let tokens_full = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
            .reshape([config.batch_size, meta.scene_len]);
        let (loss, metrics) = model.answer_step(tokens_full);
        if !metrics.loss.is_finite() {
            report.steps_skipped += 1;
            accumulator.skip();
            println!(
                "step {step}: non-finite loss {}, step discarded",
                metrics.loss
            );
            continue;
        }
        let scaled = if accumulator.every() > 1 {
            loss.mul_scalar(accumulator.loss_scale() as f32)
        } else {
            loss
        };
        let grads = GradientsParams::from_grads(scaled.backward(), &model);
        let cycle = accumulator.fold(grads, &model);
        let Some(summed) = cycle.into_gradients() else {
            report.steps_accumulated += 1;
            continue;
        };
        let mut grads = summed;
        if config.clip_norm > 0.0 {
            let total_norm = crate::quality::global_grad_norm(&model, &grads);
            if total_norm > config.clip_norm as f32 {
                crate::schedule::clip_gradients(
                    &mut grads,
                    &model,
                    total_norm,
                    config.clip_norm as f32,
                );
                report.steps_clipped += 1;
            }
        }
        model = optim.step(lr_schedule.at(step), model, grads);
        if let Some(ema) = ema.as_mut() {
            ema.update::<B>(&model);
        }

        if report.steps_taken == 0 {
            report.first_loss = metrics.loss;
            report.first_accuracy = metrics.accuracy;
        }
        report.last_loss = metrics.loss;
        report.last_accuracy = metrics.accuracy;
        report.scenes_seen += metrics.scenes;
        loss_sum += f64::from(metrics.loss);
        accuracy_sum += f64::from(metrics.accuracy);
        report.steps_taken += 1;

        if config.log_every > 0 && (step % config.log_every == 0 || step + 1 == config.steps) {
            println!(
                "step {step}: loss {:.4} accuracy {:.4}",
                metrics.loss, metrics.accuracy
            );
            if let Some(logger) = logger.as_mut() {
                logger.log(
                    step,
                    &[
                        ("loss", format!("{:.6}", metrics.loss)),
                        ("accuracy", format!("{:.6}", metrics.accuracy)),
                    ],
                )?;
            }
        }
        if config.checkpoint_every > 0 && report.steps_taken % config.checkpoint_every == 0 {
            if let Some(dir) = &config.out_dir {
                let path = checkpoint::save_content_addressed(model.clone(), dir, "geom-baseline")?;
                println!("checkpointed at step {step}: {}", path.display());
            }
        }
    }
    report.elapsed_secs = started.elapsed().as_secs_f64();
    report.mean_loss = (loss_sum / report.steps_taken.max(1) as f64) as f32;
    report.mean_accuracy = (accuracy_sum / report.steps_taken.max(1) as f64) as f32;
    if let Some(e) = &ema {
        println!("returning EMA weights ({} updates)", e.updates());
        model = e.shadow().clone();
    }
    if let Some(dir) = &config.out_dir {
        report.checkpoint = Some(checkpoint::save_content_addressed(
            model.clone(),
            dir,
            "geom-baseline",
        )?);
    }
    Ok((model, report))
}

/// Answer accuracy over aligned scenes, mirroring [`crate::geometry::evaluate_geom`]:
/// the same sampling (same seed sees the same scenes) and the same
/// answer-set-restricted metric, so a baseline number and a reasoner number
/// from the same seed are directly comparable. Returns `(accuracy, scenes
/// seen)`.
pub fn evaluate_geom_baseline<B: Backend<FloatElem = f32>>(
    model: &GeomBaseline<B>,
    corpus: &mut TokenCorpus,
    meta: &GeomMeta,
    batches: usize,
    batch_size: usize,
    seed: u64,
    device: &B::Device,
) -> anyhow::Result<(f32, usize)> {
    anyhow::ensure!(
        batches > 0 && batch_size > 0,
        "batches and batch_size must be positive"
    );
    anyhow::ensure!(
        meta.scene_len == model.scene_len(),
        "the metadata's scene length {} does not match the model's {}",
        meta.scene_len,
        model.scene_len()
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut correct = 0.0f32;
    let mut seen = 0usize;
    for _ in 0..batches {
        let windows = sample_aligned_batch(corpus, meta.scene_len, batch_size, &mut rng)?;
        let flat: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|t| i64::from(*t)))
            .collect();
        let tokens_full = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device)
            .reshape([batch_size, meta.scene_len]);
        let tokens = tokens_full.clone().narrow(1, 0, meta.scene_len - 1);
        let answers = tokens_full
            .narrow(1, meta.scene_len - 1, 1)
            .reshape([batch_size]);
        let logits = model.forward(tokens);
        let (_, _, accuracy) = answer_ce(&logits, &answers, &model.answer_tokens, model.vocab_size);
        correct += accuracy * batch_size as f32;
        seen += batch_size;
    }
    Ok((correct / seen.max(1) as f32, seen))
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
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn tiny_baseline() -> (GeomBaseline<B>, GeomBaselineConfig) {
        let device = Default::default();
        let mut geom = GeomConfig::tiny();
        geom.answer_tokens = vec![
            u16::from(b'A'),
            u16::from(b'B'),
            u16::from(b'n'),
            u16::from(b'y'),
        ];
        let config = GeomBaselineConfig::matched_to(&geom);
        (GeomBaseline::<B>::new(&config, &device).unwrap(), config)
    }

    #[test]
    fn test_matched_depth_and_width_follow_the_reasoner() {
        let geom = GeomConfig {
            num_blocks: 3,
            refine_steps: 2,
            ..GeomConfig::default()
        };
        let config = GeomBaselineConfig::matched_to(&geom);
        assert_eq!(config.num_layers, 6, "layers = num_blocks * refine_steps");
        assert_eq!(config.hidden_size, geom.hidden_size);
        assert_eq!(config.scene_len, geom.scene_len);
    }

    #[test]
    fn test_forward_shapes() {
        let (model, config) = tiny_baseline();
        let device = Default::default();
        let scene: Vec<i64> = (0..2 * (config.scene_len - 1))
            .map(|i| i64::from(b'A' + (i % 4) as u8))
            .collect();
        let tokens = Tensor::<B, 1, Int>::from_ints(scene.as_slice(), &device)
            .reshape([2, config.scene_len - 1]);
        let logits = model.forward(tokens);
        assert_eq!(logits.dims(), [2, config.vocab_size]);
    }

    #[test]
    fn test_answer_tokens_restrict_the_prediction() {
        let (model, config) = tiny_baseline();
        let device = Default::default();
        let scene: Vec<i64> = (0..3 * (config.scene_len - 1))
            .map(|i| i64::from(b'A' + (i % 4) as u8))
            .collect();
        let tokens = Tensor::<B, 1, Int>::from_ints(scene.as_slice(), &device)
            .reshape([3, config.scene_len - 1]);
        let logits = model.forward(tokens);
        for pred in model.predict_answers(&logits) {
            assert!(
                config.answer_tokens.contains(&pred),
                "prediction {pred} is outside the declared answer set"
            );
        }
    }

    #[test]
    fn test_answer_step_shapes_and_metrics() {
        let (model, config) = tiny_baseline();
        let device = Default::default();
        let scene: Vec<i64> = (0..2 * config.scene_len)
            .map(|i| i64::from(b'A' + (i % 4) as u8))
            .collect();
        let tokens = Tensor::<B, 1, Int>::from_ints(scene.as_slice(), &device)
            .reshape([2, config.scene_len]);
        let (loss, metrics) = model.answer_step(tokens);
        let value = loss.into_scalar();
        assert!(value.is_finite() && value > 0.0);
        assert!((0.0..=1.0).contains(&metrics.accuracy));
        assert_eq!(metrics.scenes, 2);
    }

    #[test]
    fn test_parameter_count_is_in_the_reasoners_band() {
        let device = Default::default();
        let geom = GeomConfig::default();
        let reasoner = crate::geometry::GeometricReasoner::<B>::new(&geom, &device).unwrap();
        let baseline =
            GeomBaseline::<B>::new(&GeomBaselineConfig::matched_to(&geom), &device).unwrap();
        let (r, b) = (reasoner.num_params() as f64, baseline.num_params() as f64);
        let skew = (r - b).abs() / r;
        assert!(
            skew <= 0.25,
            "parameter counts must sit in a 25% band: reasoner {r}, baseline {b} ({:.1}% apart)",
            skew * 100.0
        );
    }
}
