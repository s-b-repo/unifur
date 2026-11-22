//! Native inference for the HazyResearch NCM XSum release.
//!
//! The released generator and residual-VQ decoder use the published tensor
//! layouts. Upcycled MLPs occupy independent shards, and the crate's MoSME
//! router selects one box and one expert for each level's concept field.
//! Prefix keys and values are reused across levels because the checkpoint's
//! attention mask never lets prefix rows attend to generated rows.

use crate::bpe::BpeTokenizer;
use crate::expert_index::{BoxSpec, ExpertSpec, MosmeSpec};
use crate::import::{open_index, read_tensor_f32, SafetensorsIndex};
use crate::mosme::{HierarchicalRouter, MosmeConfig};
use anyhow::{anyhow, ensure, Context, Result};
use burn::backend::{ndarray::NdArrayDevice, NdArray};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend as _;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Int, Tensor, TensorData};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod certified;
mod tuning;
pub use tuning::{tune, TuneArgs};

type Cpu = NdArray<f32>;
type Matrix = Tensor<Cpu, 2>;
type Heads = Tensor<Cpu, 3>;
type Vector = Tensor<Cpu, 1>;
const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;
const WIDTH: usize = 768;
const LATENT: usize = 384;
const HEAD_SIZE: usize = 64;
const BLOCKS: usize = 12;
const CONTEXT: usize = 512;
const LEVELS: [usize; 9] = [4, 6, 10, 14, 20, 26, 32, 40, 48];
const LENGTH: usize = 48;
const CODES: usize = 4096;
const VOCAB: usize = 50304;
const EOS: u32 = 50256;
const PAD: u32 = 50257;

/// Arguments for running a converted release.
#[derive(Args, Clone, Debug)]
pub struct ReleasedArgs {
    /// Directory produced by the release weight converter.
    #[arg(long)]
    pub model: PathBuf,
    /// Document to summarize. The original training context was 512 tokens.
    #[arg(long)]
    pub prompt: String,
    /// Select one named MoSME box. If omitted the box router selects it.
    #[arg(long)]
    pub field: Option<String>,
    /// Maximum weight of the geometric correction to code logits.
    #[arg(long, default_value_t = 0.12)]
    pub geometry: f64,
    /// Select expert zero without routing for dense-release parity checks.
    #[arg(long, conflicts_with_all = ["field", "level_experts"])]
    pub dense_reference: bool,
    /// Use coarse, middle and fine experts inside the routed field box.
    #[arg(long)]
    pub level_experts: bool,
    /// Domain log-priors fitted from training targets by the corpus adapter.
    #[arg(long, requires = "field")]
    pub domain_prior: Option<PathBuf>,
    /// Trained output biases for selected MoSME experts.
    #[arg(long)]
    pub expert_adapter: Option<PathBuf>,
    /// Check the first block's pooled projection and integer readout.
    #[arg(long)]
    pub attention_certificate: bool,
    /// Emit the result and routing trace as JSON.
    #[arg(long)]
    pub json: bool,
}

/// One level's externally observable computation.
#[derive(Debug, Serialize)]
struct LevelTrace {
    length: usize,
    experts: Vec<usize>,
    max_geometry_delta: f32,
}

/// Inference result and a record of which expert shards were used.
#[derive(Debug, Serialize)]
struct ReleasedOutput {
    text: String,
    generator_parameters: usize,
    active_generator_parameters: usize,
    specialized_training_steps: usize,
    domain_prior_documents: Option<usize>,
    adapted_experts: usize,
    prompt_tokens: usize,
    prompt_truncated: bool,
    expert_shards_used: usize,
    elapsed_ms: u128,
    codes: Vec<Vec<i64>>,
    tokens: Vec<u32>,
    levels: Vec<LevelTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_certificate: Option<certified::Report>,
}

#[derive(Deserialize)]
struct Manifest {
    version: u32,
    boxes: BTreeMap<String, usize>,
    experts: usize,
    generator_parameters: usize,
    shared_parameters: usize,
    expert_parameters: usize,
    specialized_training_steps: usize,
}

#[derive(Deserialize)]
struct DomainPrior {
    version: u32,
    fields: BTreeMap<String, FittedField>,
}

#[derive(Deserialize)]
struct FittedField {
    documents: usize,
    logit_offsets: Vec<Vec<f32>>,
}

fn domain_prior(args: &ReleasedArgs) -> Result<Option<(usize, Vec<Matrix>)>> {
    let Some(path) = &args.domain_prior else {
        return Ok(None);
    };
    let data =
        std::fs::read(path).with_context(|| format!("read domain prior {}", path.display()))?;
    let mut prior: DomainPrior = serde_json::from_slice(&data)?;
    ensure!(
        prior.version == 1,
        "unsupported domain prior version {}",
        prior.version
    );
    let name = args
        .field
        .as_deref()
        .context("a domain prior needs a field")?;
    let field = prior
        .fields
        .remove(name)
        .with_context(|| format!("domain prior has no field {name:?}"))?;
    ensure!(
        field.documents > 0,
        "domain prior has no training documents"
    );
    ensure!(
        field.logit_offsets.len() == LEVELS.len(),
        "domain prior level count differs from codec"
    );
    let shifts = field
        .logit_offsets
        .into_iter()
        .map(|values| {
            ensure!(
                values.len() == CODES
                    && values
                        .iter()
                        .all(|value| value.is_finite() && value.abs() <= 1.0),
                "domain prior needs one finite offset in [-1, 1] per code"
            );
            Ok(Matrix::from_data(
                TensorData::new(values, [1, CODES]),
                &DEVICE,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some((field.documents, shifts)))
}

struct Shard {
    path: PathBuf,
    index: SafetensorsIndex,
}

impl Shard {
    fn open(path: PathBuf) -> Result<Self> {
        let index = open_index(&path)?;
        Ok(Self { path, index })
    }

    fn tensor<const D: usize>(&self, key: &str, shape: [usize; D]) -> Result<Tensor<Cpu, D>> {
        let values = read_tensor_f32(&self.path, &self.index, key)?;
        ensure!(
            values.shape == shape,
            "{key} shape {:?}, expected {shape:?}",
            values.shape
        );
        Ok(Tensor::from_data(
            TensorData::new(values.data, shape),
            &DEVICE,
        ))
    }

    fn parameters(&self) -> Result<usize> {
        self.index.tensors.iter().try_fold(0usize, |sum, tensor| {
            let size = tensor.shape.iter().try_fold(1usize, |count, width| {
                count
                    .checked_mul(*width)
                    .context("checkpoint tensor is too large")
            })?;
            sum.checked_add(size).context("checkpoint is too large")
        })
    }
}

struct Linear {
    weight: Matrix,
    bias: Option<Matrix>,
}

impl Linear {
    fn load(shard: &Shard, key: &str, input: usize, output: usize, bias: bool) -> Result<Self> {
        let weight = shard
            .tensor(&format!("{key}.weight"), [output, input])?
            .transpose();
        let bias = bias
            .then(|| -> Result<_> {
                Ok(shard
                    .tensor(&format!("{key}.bias"), [output])?
                    .reshape([1, output]))
            })
            .transpose()?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: Matrix) -> Matrix {
        let x = x.matmul(self.weight.clone());
        match &self.bias {
            Some(bias) => x + bias.clone(),
            None => x,
        }
    }
}

struct AffineNorm {
    weight: Matrix,
    bias: Option<Matrix>,
}

impl AffineNorm {
    fn load(shard: &Shard, key: &str, width: usize, bias: bool) -> Result<Self> {
        let weight = shard
            .tensor(&format!("{key}.weight"), [width])?
            .reshape([1, width]);
        let bias = bias
            .then(|| -> Result<_> {
                Ok(shard
                    .tensor(&format!("{key}.bias"), [width])?
                    .reshape([1, width]))
            })
            .transpose()?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: Matrix) -> Matrix {
        let centered = x.clone() - x.mean_dim(1);
        let variance = centered.clone().square().mean_dim(1);
        let value = centered / (variance + 1e-6).sqrt() * self.weight.clone();
        match &self.bias {
            Some(bias) => value + bias.clone(),
            None => value,
        }
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn load(shard: &Shard, key: &str, width: usize) -> Result<Self> {
        Ok(Self {
            up: Linear::load(shard, &format!("{key}.c_fc"), width, 4 * width, true)?,
            down: Linear::load(shard, &format!("{key}.c_proj"), 4 * width, width, true)?,
        })
    }

    fn forward(&self, x: Matrix) -> Matrix {
        let activation = crate::tensor_ext::exact_gelu(self.up.forward(x).unsqueeze_dim::<3>(0));
        self.down.forward(activation.squeeze_dim::<2>(0))
    }
}

struct Rope {
    cos: Heads,
    sin: Heads,
}

impl Rope {
    fn new(length: usize) -> Self {
        let half = HEAD_SIZE / 2;
        let mut cos = Vec::with_capacity(length * HEAD_SIZE);
        let mut sin = Vec::with_capacity(length * HEAD_SIZE);
        for position in 0..length {
            for i in 0..HEAD_SIZE {
                let exponent = (i % half) as f32 / half as f32;
                let phase = position as f32 * 100_000.0f32.powf(-exponent);
                cos.push(phase.cos());
                sin.push(phase.sin());
            }
        }
        Self {
            cos: Tensor::from_data(TensorData::new(cos, [1, length, HEAD_SIZE]), &DEVICE),
            sin: Tensor::from_data(TensorData::new(sin, [1, length, HEAD_SIZE]), &DEVICE),
        }
    }

    fn rotate(&self, x: Heads) -> Heads {
        let half = HEAD_SIZE / 2;
        let first = x.clone().narrow(2, 0, half);
        let last = x.clone().narrow(2, half, half);
        let rotated = Tensor::cat(vec![-last, first], 2);
        x * self.cos.clone() + rotated * self.sin.clone()
    }
}

struct Attention {
    qkv: Linear,
    output: Linear,
    q_norm: Heads,
    k_norm: Heads,
    width: usize,
}

type Kv = (Heads, Heads);

impl Attention {
    fn load(shard: &Shard, key: &str, width: usize) -> Result<Self> {
        let norm = |name| -> Result<_> {
            Ok(shard
                .tensor(&format!("{key}.{name}_norm.weight"), [HEAD_SIZE])?
                .reshape([1, 1, HEAD_SIZE]))
        };
        Ok(Self {
            qkv: Linear::load(shard, &format!("{key}.c_attn"), width, 3 * width, true)?,
            output: Linear::load(shard, &format!("{key}.c_proj"), width, width, true)?,
            q_norm: norm("q")?,
            k_norm: norm("k")?,
            width,
        })
    }

    fn project(&self, x: Matrix, rope: Option<&Rope>) -> (Heads, Heads, Heads) {
        let [length, width] = x.dims();
        let qkv = self.qkv.forward(x);
        let part = |i| {
            qkv.clone()
                .narrow(1, i * self.width, self.width)
                .reshape([length, width / HEAD_SIZE, HEAD_SIZE])
                .swap_dims(0, 1)
        };
        let norm = |value: Heads, weight: &Heads| {
            let rms = (value.clone().square().mean_dim(2) + 1e-6).sqrt();
            let value = value / rms * weight.clone();
            match rope {
                Some(rope) => rope.rotate(value),
                None => value,
            }
        };
        (
            norm(part(0), &self.q_norm),
            norm(part(1), &self.k_norm),
            part(2),
        )
    }

    fn forward(&self, x: Matrix, rope: Option<&Rope>, prefix: Option<&Kv>) -> (Matrix, Kv) {
        let [length, width] = x.dims();
        let (q, mut k, mut v) = self.project(x, rope);
        if let Some((prefix_k, prefix_v)) = prefix {
            k = Tensor::cat(vec![prefix_k.clone(), k], 1);
            v = Tensor::cat(vec![prefix_v.clone(), v], 1);
        }
        let scores = q.matmul(k.clone().swap_dims(1, 2)) / (HEAD_SIZE as f64).sqrt();
        let context = softmax(scores, 2)
            .matmul(v.clone())
            .swap_dims(0, 1)
            .reshape([length, width]);
        (self.output.forward(context), (k, v))
    }
}

struct Block {
    norm_attention: AffineNorm,
    attention: Attention,
    norm_feed_forward: AffineNorm,
}

impl Block {
    fn load(shard: &Shard, key: &str, width: usize) -> Result<Self> {
        Ok(Self {
            norm_attention: AffineNorm::load(shard, &format!("{key}.ln_1"), width, true)?,
            attention: Attention::load(shard, &format!("{key}.attn"), width)?,
            norm_feed_forward: AffineNorm::load(shard, &format!("{key}.ln_2"), width, true)?,
        })
    }

    fn attend(&self, x: Matrix, rope: Option<&Rope>, prefix: Option<&Kv>) -> (Matrix, Kv) {
        let (attention, cache) =
            self.attention
                .forward(self.norm_attention.forward(x.clone()), rope, prefix);
        (x + attention, cache)
    }

    fn add_expert(&self, x: Matrix, expert: &FeedForward) -> Matrix {
        x.clone() + expert.forward(self.norm_feed_forward.forward(x))
    }
}

struct RoutedBlock {
    block: Block,
    site: String,
    router: HierarchicalRouter<Cpu>,
    active: BTreeMap<usize, FeedForward>,
}

struct Segment<'a> {
    rope: &'a Rope,
    prefix: Option<&'a Kv>,
    level: usize,
}

impl RoutedBlock {
    fn forward(
        &mut self,
        x: Matrix,
        segment: Segment<'_>,
        experts: &[Shard],
        routing: &Routing,
    ) -> Result<(Matrix, Kv, usize)> {
        let (x, cache) = self.block.attend(x, Some(segment.rope), segment.prefix);
        let expert = if routing.dense {
            0
        } else {
            let condition = self.block.norm_feed_forward.forward(x.clone()).mean_dim(0);
            let gates = self.router.route(condition);
            let box_index = routing
                .field
                .map_or_else(|| scalar_id(gates.top_box_idx), Ok)?;
            let within_box = gates
                .top_expert_idx
                .get(box_index)
                .context("router box outside checkpoint")?;
            let specialist = if routing.level_experts {
                level_group(segment.level)
            } else {
                scalar_id(within_box.clone())?
            };
            routing.offsets[box_index] + specialist
        };
        let expert_weights = match self.active.entry(expert) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let shard = experts
                    .get(expert)
                    .context("router expert outside checkpoint")?;
                entry.insert(FeedForward::load(
                    shard,
                    &format!("{}.mlp", self.site),
                    WIDTH,
                )?)
            }
        };
        Ok((self.block.add_expert(x, expert_weights), cache, expert))
    }
}

fn scalar_id(id: Tensor<Cpu, 2, Int>) -> Result<usize> {
    let id = id
        .into_data()
        .to_vec::<i64>()
        .map_err(|err| anyhow!("router index: {err:?}"))?;
    usize::try_from(*id.first().context("empty router selection")?).context("negative router index")
}

fn floats<const D: usize>(value: Tensor<Cpu, D>) -> Result<Vec<f32>> {
    value
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow!("read tensor: {err:?}"))
}

fn lookup(table: &Matrix, ids: &[i64]) -> Matrix {
    table
        .clone()
        .select(0, Tensor::<Cpu, 1, Int>::from_ints(ids, &DEVICE))
}

struct Convolution {
    weight: Tensor<Cpu, 3>,
    bias: Vector,
}

impl Convolution {
    fn load(shard: &Shard, key: &str) -> Result<Self> {
        Ok(Self {
            weight: shard.tensor(&format!("{key}.weight"), [LATENT, LATENT, 3])?,
            bias: shard.tensor(&format!("{key}.bias"), [LATENT])?,
        })
    }

    fn forward(&self, input: Matrix) -> Matrix {
        let rows = input.dims()[0];
        let input = input.transpose().reshape([1, LATENT, rows]);
        conv1d(
            input,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1], [1], [1], 1),
        )
        .reshape([LATENT, rows])
        .transpose()
    }
}

/// Transposed convolution with stride equal to its kernel size. There
/// are no overlapping windows, so one matrix product computes all outputs.
struct Upsample {
    weight: Matrix,
    bias: Matrix,
    stride: usize,
}

impl Upsample {
    fn load(shard: &Shard, key: &str, stride: usize) -> Result<Self> {
        let weight = shard.tensor(&format!("{key}.weight"), [LATENT, LATENT, stride])?;
        Ok(Self {
            weight: weight.swap_dims(1, 2).reshape([LATENT, stride * LATENT]),
            bias: shard
                .tensor(&format!("{key}.bias"), [LATENT])?
                .reshape([1, LATENT]),
            stride,
        })
    }

    fn forward(&self, x: Matrix) -> Matrix {
        let rows = x.dims()[0];
        x.matmul(self.weight.clone())
            .reshape([rows * self.stride, LATENT])
            + self.bias.clone()
    }
}

fn group_norm(x: Matrix) -> Matrix {
    let [rows, channels] = x.dims();
    let groups = x.transpose().reshape([8, channels / 8 * rows]);
    let centered = groups.clone() - groups.mean_dim(1);
    let variance = centered.clone().square().mean_dim(1);
    (centered / (variance + 1e-5).sqrt())
        .reshape([channels, rows])
        .transpose()
}

/// Half-pixel linear interpolation as used by the released codec.
fn linear_resize(x: Matrix, to: usize) -> Matrix {
    let [from, _] = x.dims();
    let mut weights = vec![0.0; to * from];
    for i in 0..to {
        let position =
            (((i as f64 + 0.5) * from as f64 / to as f64) - 0.5).clamp(0.0, (from - 1) as f64);
        let lower = position.floor() as usize; // audit-allow: position is clamped to valid rows
        let upper = (lower + 1).min(from - 1);
        let fraction = (position - lower as f64) as f32;
        weights[i * from + lower] += 1.0 - fraction;
        weights[i * from + upper] += fraction;
    }
    Matrix::from_data(TensorData::new(weights, [to, from]), &DEVICE).matmul(x)
}

/// Area interpolation uses adaptive-pooling windows with ceil end points.
fn area_resize(x: Matrix, to: usize) -> Matrix {
    let [from, _] = x.dims();
    let mut weights = vec![0.0; to * from];
    for i in 0..to {
        let start = i * from / to;
        let end = ((i + 1) * from).div_ceil(to);
        weights[i * from + start..i * from + end].fill(1.0 / (end - start) as f32);
    }
    Matrix::from_data(TensorData::new(weights, [to, from]), &DEVICE).matmul(x)
}

struct Codec {
    embedding: Matrix,
    books: Vec<Matrix>,
    residual: Vec<Convolution>,
    upsample_first: Upsample,
    upsample_second: Upsample,
    decoder: Block,
    decoder_mlp: FeedForward,
    decoder_norm: AffineNorm,
    decoder_out: Linear,
    unigram: Vec<f32>,
}

impl Codec {
    fn load(shard: &Shard, decode: &Shard) -> Result<Self> {
        Ok(Self {
            embedding: shard.tensor("encoder.wte.weight", [VOCAB, LATENT])?,
            books: (0..LEVELS.len())
                .map(|i| shard.tensor(&format!("vq.embeddings.{i}.weight"), [CODES, LATENT]))
                .collect::<Result<_>>()?,
            residual: (0..LEVELS.len())
                .map(|i| Convolution::load(shard, &format!("vq.quant_resi.{i}")))
                .collect::<Result<_>>()?,
            upsample_first: Upsample::load(shard, "vq.learned_up_l1.0", 3)?,
            upsample_second: Upsample::load(shard, "vq.learned_up_l1.2", 4)?,
            decoder: Block::load(shard, "decoder.middle_block", LATENT)?,
            decoder_mlp: FeedForward::load(shard, "decoder.middle_block.mlp", LATENT)?,
            decoder_norm: AffineNorm::load(shard, "decoder.ln_f", LATENT, false)?,
            decoder_out: Linear::load(shard, "decoder.output_layer", LATENT, VOCAB, false)?,
            unigram: read_tensor_f32(&decode.path, &decode.index, "token_unigram")?.data,
        })
    }

    fn refine_reconstruction(&self, z: Matrix, index: usize, ids: &[i64]) -> Matrix {
        let vectors = lookup(&self.books[index], ids);
        let full = match index {
            0 => self
                .upsample_second
                .forward(group_norm(self.upsample_first.forward(vectors))),
            i if i + 1 == LEVELS.len() => vectors,
            _ => linear_resize(vectors, LENGTH),
        };
        z + full.clone() * 0.5 + self.residual[index].forward(full) * 0.5
    }

    fn surface(&self, z: Matrix) -> Result<Vec<u32>> {
        let (x, _) = self.decoder.attend(z, None, None);
        let x = self.decoder.add_expert(x, &self.decoder_mlp);
        let logits = floats(self.decoder_out.forward(self.decoder_norm.forward(x)))?;
        ensure!(
            self.unigram.len() >= VOCAB,
            "token unigram is shorter than the decoder vocabulary"
        );
        let mut counts = vec![0_u32; VOCAB];
        let mut picked = Vec::with_capacity(LENGTH);
        for (position, row) in logits.chunks_exact(VOCAB).enumerate() {
            let mut best = (0, f64::NEG_INFINITY);
            for (token, &logit) in row.iter().enumerate() {
                let allowed = f64::from(self.unigram[token]) * LENGTH as f64;
                let difference = f64::from(counts[token]) - allowed;
                let mut score = f64::from(logit) - 2.0 * difference.max(0.0)
                    + (LENGTH - position) as f64 / LENGTH as f64 * (-difference).max(0.0);
                for (lag, &previous) in picked.iter().rev().take(4).enumerate() {
                    if previous == token {
                        score -= 5.0 * 0.5_f64.powf(lag as f64);
                    }
                }
                if score > best.1 {
                    best = (token, score);
                }
            }
            ensure!(
                best.1.is_finite(),
                "decoder produced no finite token at {position}"
            );
            picked.push(best.0);
            counts[best.0] += 1;
        }
        picked
            .into_iter()
            .map(|id| u32::try_from(id).context("decoder token out of range"))
            .collect()
    }
}

struct ConceptHead {
    projection: Linear,
    squared_norms: Matrix,
    scale: f64,
    metric: Matrix,
}

impl ConceptHead {
    fn load(shard: &Shard, level: usize, book: &Matrix) -> Result<Self> {
        let key = format!("lm_heads.{level}");
        let squared = book.clone().square();
        let squared_norms = squared.clone().sum_dim(1).transpose();
        let rms = f64::from(floats(squared_norms.clone().mean_dim(1))?[0]).max(1e-12);
        let inv_temperature =
            f64::from(floats(shard.tensor(&format!("{key}.log_inv_T"), [1])?)?[0]).exp();
        ensure!(
            inv_temperature.is_finite(),
            "invalid checkpoint concept temperature"
        );
        let metric = squared.mean_dim(0).clamp_min(1e-5).recip();
        let metric = metric.clone() / metric.mean_dim(1);
        Ok(Self {
            projection: Linear::load(shard, &format!("{key}.proj"), WIDTH, LATENT, false)?,
            squared_norms,
            scale: inv_temperature / rms,
            metric,
        })
    }

    fn logits(&self, hidden: Matrix, book: &Matrix) -> Matrix {
        (self
            .projection
            .forward(hidden)
            .matmul(book.clone().transpose())
            * 2.0
            - self.squared_norms.clone())
            * self.scale
    }

    /// Refine the posterior mean of the actual RVQ code vectors along the
    /// adjacent-position graph and the aligned coarser reconstruction.
    /// The diagonal metric is positive and derived from codebook moments.
    /// The geometric logit delta is the difference of two squared-distance
    /// readouts with their row constants removed.
    fn relax(
        &self,
        logits: Matrix,
        book: &Matrix,
        coarse: Option<&Matrix>,
        strength: f64,
    ) -> Result<(Matrix, f32)> {
        if strength == 0.0 {
            return Ok((logits, 0.0));
        }
        let origin = softmax(logits.clone(), 1).matmul(book.clone());
        let mut state = origin.clone();
        let mut delta = Matrix::zeros(logits.shape(), &DEVICE);
        let length = state.dims()[0];
        let left: Vec<_> = (0..length as i64).map(|i| (i - 1).max(0)).collect();
        let right: Vec<_> = (0..length as i64)
            .map(|i| (i + 1).min(length as i64 - 1))
            .collect();
        let book_scale = (book.clone().square() * self.metric.clone())
            .sum_dim(1)
            .mean_dim(0)
            .clamp_min(1e-12);
        for _ in 0..2 {
            let mut neighbors = vec![state.clone(), lookup(&state, &left), lookup(&state, &right)];
            if let Some(coarse) = coarse {
                let origin_norm = origin.clone().square().sum_dim(1).sqrt();
                let parent_norm = coarse.clone().square().sum_dim(1).sqrt().clamp_min(1e-12);
                neighbors.push(coarse.clone() / parent_norm * origin_norm);
            }
            let scores = neighbors
                .iter()
                .map(|neighbor| {
                    ((neighbor.clone() - state.clone()).square() * self.metric.clone()).sum_dim(1)
                        / book_scale.clone()
                        * -1.0
                })
                .collect();
            let gates = softmax(Tensor::cat(scores, 1), 1);
            let mut message = Matrix::zeros(state.shape(), &DEVICE);
            for (index, neighbor) in neighbors.into_iter().enumerate() {
                message = message + neighbor * gates.clone().narrow(1, index, 1);
            }
            state = (state + message) * 0.5;
            delta = (((state.clone() - origin.clone()) * self.metric.clone())
                .matmul(book.clone().transpose())
                / book_scale.clone()
                * 2.0)
                .clamp(-1.0, 1.0)
                * strength;
            state = softmax(logits.clone() + delta.clone(), 1).matmul(book.clone());
        }
        let maximum = floats(delta.clone().abs().max())?[0];
        ensure!(maximum.is_finite(), "non-finite concept geometry update");
        Ok((logits + delta, maximum))
    }
}

struct Routing {
    spec: MosmeSpec,
    offsets: Vec<usize>,
    field: Option<usize>,
    dense: bool,
    level_experts: bool,
}

fn level_group(level: usize) -> usize {
    3 * level / LEVELS.len()
}

impl Routing {
    fn load(
        manifest: &Manifest,
        name: Option<&str>,
        dense: bool,
        level_experts: bool,
    ) -> Result<Self> {
        let field = name
            .map(|name| {
                manifest
                    .boxes
                    .keys()
                    .position(|box_name| box_name == name)
                    .with_context(|| format!("unknown expert box {name:?}"))
            })
            .transpose()?;
        let mut offsets = Vec::new();
        let mut boxes = Vec::new();
        let mut offset = 0;
        for (name, count) in &manifest.boxes {
            ensure!(
                (1..=32).contains(count),
                "{name} box must have 1 to 32 experts"
            );
            ensure!(
                !level_experts || *count >= 3,
                "{name} box needs three experts for coarse, middle and fine routing"
            );
            offsets.push(offset);
            let experts = (0..*count)
                .map(|i| ExpertSpec::new(format!("{name}/{i}"), name))
                .collect();
            boxes.push(BoxSpec::new(name, name, experts));
            offset += count;
        }
        ensure!(
            offset == manifest.experts && offset <= 64,
            "invalid number of expert shards"
        );
        let spec = MosmeSpec {
            boxes,
            top_box: 1,
            top_expert: 1,
            route_on_tokens: false,
            balance: Default::default(),
        };
        spec.validate()?;
        Ok(Self {
            spec,
            offsets,
            field,
            dense,
            level_experts,
        })
    }
}

struct ReleasedModel {
    manifest: Manifest,
    codec: Codec,
    tokenizer: BpeTokenizer,
    experts: Vec<Shard>,
    blocks: Vec<RoutedBlock>,
    projection: Linear,
    final_norm: AffineNorm,
    heads: Vec<ConceptHead>,
    level_embedding: Matrix,
    initial_queries: Matrix,
}

fn read_manifest(model: &Path) -> Result<Manifest> {
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(model.join("manifest.json"))?)?;
    ensure!(
        manifest.version == 1,
        "unsupported released-checkpoint version {}",
        manifest.version
    );
    Ok(manifest)
}

impl ReleasedModel {
    fn load(model: &Path, manifest: Manifest, spec: MosmeSpec) -> Result<Self> {
        let shared = Shard::open(model.join("generator.safetensors"))?;
        let experts: Vec<_> = (0..manifest.experts)
            .map(|i| Shard::open(model.join(format!("expert-{i:02}.safetensors"))))
            .collect::<Result<_>>()?;
        let mut parameters = shared.parameters()?;
        ensure!(
            parameters == manifest.shared_parameters,
            "shared parameter count differs from manifest"
        );
        for expert in &experts {
            let count = expert.parameters()?;
            ensure!(
                count == manifest.expert_parameters,
                "expert parameter count differs from manifest"
            );
            parameters = parameters
                .checked_add(count)
                .context("checkpoint is too large")?;
        }
        ensure!(
            parameters == manifest.generator_parameters,
            "generator parameter count differs from manifest"
        );
        let codec = Codec::load(
            &Shard::open(model.join("codec.safetensors"))?,
            &Shard::open(model.join("decode.safetensors"))?,
        )?;
        let tokenizer =
            BpeTokenizer::from_json(&std::fs::read_to_string(model.join("tokenizer.json"))?)?;
        let projection = Linear::load(&shared, "transformer.wte", LATENT, WIDTH, true)?;
        let level_embedding: Matrix =
            shared.tensor("transformer.wle.weight", [LEVELS.len(), WIDTH])?;
        let final_norm = AffineNorm::load(&shared, "transformer.ln_f", WIDTH, true)?;
        let initial_queries = shared
            .tensor("l0_queries", [1, LEVELS[0], LATENT])?
            .reshape([LEVELS[0], LATENT]);
        let heads = codec
            .books
            .iter()
            .enumerate()
            .map(|(level, book)| ConceptHead::load(&shared, level, book))
            .collect::<Result<Vec<_>>>()?;
        Cpu::seed(&DEVICE, 42);
        let config = MosmeConfig::new(WIDTH, WIDTH, spec);
        let blocks = (0..BLOCKS)
            .map(|index| {
                let site = format!("transformer.blocks.{index}");
                Ok(RoutedBlock {
                    block: Block::load(&shared, &site, WIDTH)?,
                    site,
                    router: HierarchicalRouter::new(&config, &DEVICE),
                    active: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            manifest,
            codec,
            tokenizer,
            experts,
            blocks,
            projection,
            final_norm,
            heads,
            level_embedding,
            initial_queries,
        })
    }

    fn prompt(&self, text: &str) -> Result<(Vec<u32>, bool)> {
        let mut tokens = self.tokenizer.encode(text)?;
        ensure!(!tokens.is_empty(), "the document has no tokens");
        let truncated = tokens.len() > CONTEXT;
        tokens.truncate(CONTEXT);
        Ok((tokens, truncated))
    }

    fn prefix_cache(
        &mut self,
        prompt: &[u32],
        routing: &Routing,
    ) -> Result<(Vec<Kv>, BTreeSet<usize>)> {
        let mut prefix = self.embed_prompt(prompt);
        let mut prefix_cache = Vec::with_capacity(BLOCKS);
        let prefix_rope = Rope::new(prompt.len());
        let mut used = BTreeSet::new();
        for block in &mut self.blocks {
            let (next, cache, expert) = block.forward(
                prefix,
                Segment {
                    rope: &prefix_rope,
                    prefix: None,
                    level: 0,
                },
                &self.experts,
                routing,
            )?;
            prefix = next;
            prefix_cache.push(cache);
            used.insert(expert);
        }
        Ok((prefix_cache, used))
    }

    fn embed_prompt(&self, prompt: &[u32]) -> Matrix {
        let ids: Vec<_> = prompt.iter().map(|&id| i64::from(id)).collect();
        self.projection.forward(lookup(&self.codec.embedding, &ids))
    }

    fn embed_level(&self, input: Matrix, level: usize) -> Matrix {
        self.projection.forward(input) + self.level_embedding.clone().narrow(0, level, 1)
    }

    fn attention_certificate(&self, prompt: &[u32], prefix: &[Kv]) -> Result<certified::Report> {
        let block = &self
            .blocks
            .first()
            .context("release has no attention block")?
            .block;
        let (prefix_k, prefix_v) = prefix.first().context("prefix has no attention cache")?;
        let source = block.norm_attention.forward(self.embed_prompt(prompt));
        let target = block
            .norm_attention
            .forward(self.embed_level(self.initial_queries.clone(), 0));
        let (queries, keys, values) = block
            .attention
            .project(target.clone(), Some(&Rope::new(LEVELS[0])));
        certified::probe(
            &block.attention,
            Tensor::cat(vec![source, target], 0),
            queries,
            Tensor::cat(vec![prefix_k.clone(), keys], 1),
            Tensor::cat(vec![prefix_v.clone(), values], 1),
        )
    }

    fn level_state(
        &mut self,
        level: usize,
        reconstruction: &Matrix,
        prefix_cache: &[Kv],
        routing: &Routing,
    ) -> Result<LevelState> {
        let length = LEVELS[level];
        let coarse = (level > 0).then(|| area_resize(reconstruction.clone(), length));
        let input = coarse
            .clone()
            .unwrap_or_else(|| self.initial_queries.clone());
        let mut x = self.embed_level(input, level);
        let rope = Rope::new(length);
        let mut experts = Vec::with_capacity(BLOCKS);
        for (block, cache) in self.blocks.iter_mut().zip(prefix_cache) {
            let (next, _, expert) = block.forward(
                x,
                Segment {
                    rope: &rope,
                    prefix: Some(cache),
                    level,
                },
                &self.experts,
                routing,
            )?;
            x = next;
            experts.push(expert);
        }
        Ok(LevelState {
            hidden: x,
            coarse,
            experts,
        })
    }
}

struct LevelState {
    hidden: Matrix,
    coarse: Option<Matrix>,
    experts: Vec<usize>,
}

/// Generate using the checkpoint's coarse-to-fine factorization.
pub fn run(args: ReleasedArgs) -> Result<String> {
    ensure!(
        !args.prompt.trim().is_empty(),
        "a non-empty document is required"
    );
    ensure!(
        (0.0..=1.0).contains(&args.geometry),
        "geometry must be between 0 and 1"
    );
    let started = Instant::now();
    let prior = domain_prior(&args)?;
    let manifest = read_manifest(&args.model)?;
    let routing = Routing::load(
        &manifest,
        args.field.as_deref(),
        args.dense_reference,
        args.level_experts,
    )?;
    let adapter = args
        .expert_adapter
        .as_deref()
        .map(|path| tuning::load_biases(path, manifest.experts))
        .transpose()?;
    let mut model = ReleasedModel::load(&args.model, manifest, routing.spec.clone())?;
    let (prompt, prompt_truncated) = model.prompt(&args.prompt)?;
    let (prefix_cache, mut used) = model.prefix_cache(&prompt, &routing)?;
    let attention_certificate = args
        .attention_certificate
        .then(|| model.attention_certificate(&prompt, &prefix_cache))
        .transpose()?;
    let mut reconstruction = Matrix::zeros([LENGTH, LATENT], &DEVICE);
    let mut codes = Vec::with_capacity(LEVELS.len());
    let mut levels = Vec::with_capacity(LEVELS.len());
    for (level, length) in LEVELS.into_iter().enumerate() {
        let state = model.level_state(level, &reconstruction, &prefix_cache, &routing)?;
        used.extend(state.experts.iter().copied());
        let mut hidden = state.hidden;
        if let Some(bias) = state
            .experts
            .last()
            .and_then(|id| adapter.as_ref()?.get(id))
        {
            hidden = hidden + bias.clone();
        }
        let book = &model.codec.books[level];
        let head = &model.heads[level];
        let mut logits = head.logits(model.final_norm.forward(hidden), book);
        if let Some((_, shifts)) = &prior {
            logits = logits + shifts[level].clone();
        }
        let (logits, max_geometry_delta) =
            head.relax(logits, book, state.coarse.as_ref(), args.geometry)?;
        let ids = logits
            .argmax(1)
            .into_data()
            .to_vec::<i64>()
            .map_err(|err| anyhow!("concept ids: {err:?}"))?;
        reconstruction = model
            .codec
            .refine_reconstruction(reconstruction, level, &ids);
        codes.push(ids);
        levels.push(LevelTrace {
            length,
            experts: state.experts,
            max_geometry_delta,
        });
    }
    let tokens = model.codec.surface(reconstruction)?;
    let visible: Vec<_> = tokens
        .iter()
        .copied()
        .take_while(|&id| id != EOS)
        .filter(|&id| id < PAD)
        .collect();
    let text = model.tokenizer.decode(&visible)?.trim().to_owned();
    let manifest = &model.manifest;
    let output = ReleasedOutput {
        text,
        generator_parameters: manifest.generator_parameters,
        active_generator_parameters: manifest.shared_parameters + manifest.expert_parameters,
        specialized_training_steps: manifest.specialized_training_steps,
        domain_prior_documents: prior.map(|(documents, _)| documents),
        adapted_experts: adapter.as_ref().map_or(0, BTreeMap::len),
        prompt_tokens: prompt.len(),
        prompt_truncated,
        expert_shards_used: used.len(),
        elapsed_ms: started.elapsed().as_millis(),
        codes,
        tokens,
        levels,
        attention_certificate,
    };
    if args.json {
        Ok(serde_json::to_string_pretty(&output)?)
    } else {
        let mut result = format!(
            "{}\n\n{} generator parameters, {} active per field pass, {} levels, {} ms",
            output.text,
            output.generator_parameters,
            output.active_generator_parameters,
            output.levels.len(),
            output.elapsed_ms,
        );
        if let Some(report) = output.attention_certificate {
            use std::fmt::Write as _;
            write!(
                result,
                "\nAttention certificate scored {} of {} token-query pairs. {} of {} decisions agreed with the float readout.",
                report.scored_tokens, report.total_token_scores, report.float_agreements, report.queries
            )?;
        }
        Ok(result)
    }
}

pub(crate) fn certificates() -> Result<Vec<crate::verify::Certificate>> {
    let matrix = |values: &[f32], rows, cols| {
        Matrix::from_data(TensorData::new(values.to_vec(), [rows, cols]), &DEVICE)
    };
    let residual = |actual: Vec<f32>, expected: &[f32]| -> f64 {
        if actual.len() != expected.len() || actual.iter().any(|x| !x.is_finite()) {
            return f64::INFINITY;
        }
        actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max)
    };
    let pooling = residual(
        floats(area_resize(matrix(&[0.0, 6.0, 12.0], 3, 1), 2))?,
        &[3.0, 9.0],
    )
    .max(residual(
        floats(linear_resize(matrix(&[0.0, 4.0], 2, 1), 4))?,
        &[0.0, 1.0, 3.0, 4.0],
    ));
    let head = ConceptHead {
        projection: Linear {
            weight: Matrix::zeros([2, 2], &DEVICE),
            bias: None,
        },
        squared_norms: Matrix::zeros([1, 2], &DEVICE),
        scale: 1.0,
        metric: matrix(&[0.5, 1.5], 1, 2),
    };
    let book = matrix(&[1.0, 0.0, -1.0, 1.0], 2, 2);
    let base = matrix(&[2.0, 0.0, 0.0, 2.0], 2, 2);
    let (same, zero) = head.relax(base.clone(), &book, None, 0.0)?;
    let identity = zero
        .abs()
        .max(residual(floats(same)?, &floats(base.clone())?) as f32);
    let (changed, delta) = head.relax(base.clone(), &book, None, 0.1)?;
    let observed = floats((changed - base).abs().max())?[0];
    let bounded = if observed.is_finite() && observed > 0.0 {
        (f64::from(delta.max(observed)) - 0.1).max(0.0)
    } else {
        f64::INFINITY
    };
    let groups: Vec<_> = (0..LEVELS.len()).map(level_group).collect();
    let grouping = f64::from(u8::from(groups != [0, 0, 0, 1, 1, 1, 2, 2, 2]));
    let projection = Linear {
        weight: matrix(&[2.0, -1.0, 0.5, 4.0], 2, 2),
        bias: Some(matrix(&[1.5, -2.0], 1, 2)),
    };
    let values = matrix(&[-1.0, 2.0, 3.0, 4.0, 1.0, -2.0], 3, 2);
    let weights = matrix(&[0.25, 0.5, 0.25], 1, 3);
    let pooled = weights.clone().matmul(projection.forward(values.clone()));
    let projected = projection.forward(weights.matmul(values));
    let affine = residual(floats(projected)?, &floats(pooled)?);
    let counterexample = matrix(&[-1.0, 1.0], 2, 1);
    let relu = residual(
        floats(
            counterexample.clone().clamp_min(0.0).mean_dim(0)
                - counterexample.mean_dim(0).clamp_min(0.0),
        )?,
        &[0.5],
    );
    let attention = Attention {
        qkv: Linear {
            weight: Matrix::ones([HEAD_SIZE, 3 * HEAD_SIZE], &DEVICE) * 0.015625,
            bias: None,
        },
        output: Linear {
            weight: Matrix::eye(HEAD_SIZE, &DEVICE),
            bias: None,
        },
        q_norm: Heads::ones([1, 1, HEAD_SIZE], &DEVICE),
        k_norm: Heads::ones([1, 1, HEAD_SIZE], &DEVICE),
        width: HEAD_SIZE,
    };
    let row = |value| Matrix::ones([1, HEAD_SIZE], &DEVICE) * value;
    let source = Tensor::cat(vec![row(0.5), row(-0.125)], 0);
    let (_, cached) = attention.forward(source.clone(), None, None);
    let mut forks: f64 = 0.0;
    for target in [row(0.25), row(0.75)] {
        let (whole, _) = attention.forward(
            Tensor::cat(vec![source.clone(), target.clone()], 0),
            None,
            None,
        );
        let (suffix, _) = attention.forward(target, None, Some(&cached));
        forks = forks.max(residual(floats(suffix)?, &floats(whole.narrow(0, 2, 1))?));
    }
    let certificate = |name, theorem, residual, tolerance| crate::verify::Certificate {
        group: "ncm",
        name,
        theorem,
        residual,
        tolerance,
    };
    Ok(vec![
        certificate("released_resampling_matches_reference", "The released codec uses adaptive area windows and half-pixel interpolation.", pooling, 0.0),
        certificate("released_geometry_zero_is_identity", "Zero geometric correction preserves every trained code logit.", f64::from(identity), 0.0),
        certificate("released_geometry_is_bounded", "The graph can move code logits while respecting the configured correction bound.", bounded, 1e-6),
        certificate("released_level_groups_partition_schedule", "Coarse, middle and fine experts partition the nine released levels into three equal groups.", grouping, 0.0),
        certificate("released_expert_bias_gradient_matches_finite_difference", "The output bias gradient matches the finite difference of teacher-forced code cross-entropy.", tuning::gradient_residual()?, 2e-4),
        certificate("released_pair_weight_matches_loss_gradient", "The counterfactual branch weight differentiates the stable pairwise loss.", tuning::pair_gradient_residual(), 1e-9),
        certificate("released_projection_commutes_with_normalized_pooling", "A shared affine projection can follow a normalized weighted pool.", affine, 0.0),
        certificate("released_relu_swap_has_a_counterexample", "A two-row witness rejects moving ReLU after a mean.", relu, 0.0),
        certificate("released_shared_prefix_preserves_fork_attention", "Reusing source keys and values preserves suffix attention in each independent branch.", forks, 1e-6),
        certificate("released_integer_attention_certifies_full_readout", "An exact signed margin interval encloses the full readout and preserves its winner.", certified::residual()?, 0.0),
    ])
}
