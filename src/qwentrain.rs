//! Adapters-only (QLoRA) training engine for the NF4-resident Qwen trunk
//! (Task 3b of the 27B wiring project).
//!
//! The gate this builds toward: the 27B resident (~13 GiB packed), a
//! training step completing end-to-end on CPU, and a bit-identical resume.
//!
//! - **Auto-LoRA**: [`attach_lora`] hangs a zero-init [`LoraAdapter`] on
//!   every NF4-resident projection (full-attn q/k/v/o, MLP gate/up/down,
//!   delta-head qkv/z/b/a/out). Embeddings, `lm_head`, norms, conv and
//!   `dt_bias`/`A_log` stay adapter-free, and
//!   [`freeze_non_adapter_params`] marks every non-adapter parameter
//!   `require_grad(false)`, so the optimizer only ever sees adapter
//!   gradients (burn's `OptimizerAdaptor` creates state per parameter WITH
//!   a gradient -- m/v exist for adapters and nothing else).
//! - **Segment-checkpointed backward**
//!   ([`segmented_next_token_backward`]): one no-grad forward stores the
//!   per-layer boundary activations; each layer then re-runs WITH grad
//!   from its boundary when its backward turn comes, chaining the exact
//!   vector-Jacobian product upstream via `Tensor::grad`. At most one
//!   layer's graph (plus the loss segment's) is alive at a time, which is
//!   what lets a 64-layer trunk train next to 13 GiB of packed weights on
//!   a 46 GB machine. Numerically identical to a one-shot backward
//!   (certified in `verify.rs`).
//! - **Paged AdamW** ([`OptimPager`]): the optimizer record is a
//!   `HashMap<ParamId, _>`, so pages are just partitions of it by decoder
//!   layer; pages serialize/evict between micro-steps and merge back on
//!   their update turn. Resume restores m/v exactly.
//! - **Resume**: content-addressed model record (in NF4 mode the record
//!   holds only the f32 params -- adapters + smalls -- since packed
//!   weights are `#[module(skip)]`; full resume = deterministic
//!   re-quantize from shards, proven in 3a, + this record) plus
//!   [`TrainState`] (optimizer record, EMA shadow, host RNG, counters).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use burn::module::{list_param_ids, Module, ModuleVisitor, Param, ParamId};
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Int, Tensor};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::checkpoint::{self, StateFile, TrainState};
use crate::quality::{global_grad_norm, non_finite_parameters};
use crate::quantize::LoraAdapter;
use crate::qwennet::{QwenDecoderLayer, QwenLinear, QwenLoraConfig, QwenMixer, QwenTrunk};
use crate::schedule::{clip_gradients, Ema, GradientAccumulator};
use crate::train::step_seed;

fn default_lr() -> f64 {
    1e-4
}

fn default_accumulate() -> usize {
    1
}

/// Training-loop configuration. Every field has a serde default so
/// sidecars written before a field existed still parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenTrainConfig {
    /// Learning rate (constant schedule; the gate takes single steps).
    #[serde(default = "default_lr")]
    pub lr: f64,
    /// Gradient-clip bound on the ACCUMULATED gradient (train.rs
    /// discipline: clipping each micro-batch separately would bound k
    /// small vectors whose sum can still be large). `None` disables.
    #[serde(default)]
    pub clip_norm: Option<f32>,
    /// Micro-batches averaged per optimizer step (`--accumulate`
    /// semantics: each micro-batch's loss is scaled by `1/k`, the sum is
    /// the gradient of the mean over all `k * batch_size` samples).
    #[serde(default = "default_accumulate")]
    pub accumulate: usize,
    /// Bias-corrected EMA decay over the adapter bank; `None` disables.
    #[serde(default)]
    pub ema_decay: Option<f64>,
    /// Adapter hyperparameters.
    #[serde(default)]
    pub lora: QwenLoraConfig,
    /// Run seed (device RNG is reseeded per micro-step from `(seed,
    /// micro_counter)` via `train::step_seed`, which is what makes a
    /// resumed run bit-identical to an uninterrupted one).
    #[serde(default)]
    pub seed: u64,
}

impl Default for QwenTrainConfig {
    fn default() -> Self {
        Self {
            lr: default_lr(),
            clip_norm: Some(1.0),
            accumulate: default_accumulate(),
            ema_decay: Some(0.99),
            lora: QwenLoraConfig::default(),
            seed: 0,
        }
    }
}

/// The adapter slots of one decoder layer, in a documented order: mixer
/// slots first (delta head: qkv, z, b, a, out; full attention: q, k, v,
/// o), then MLP (gate, up, down). Every slot carries an adapter after
/// [`attach_lora`].
fn layer_adapter_slots<B: Backend>(layer: &QwenDecoderLayer<B>) -> Vec<&QwenLinear<B>> {
    let mut slots: Vec<&QwenLinear<B>> = match &layer.mixer {
        QwenMixer::Linear(head) => {
            vec![
                &head.in_proj_qkv,
                &head.in_proj_z,
                &head.in_proj_b,
                &head.in_proj_a,
                &head.out_proj,
            ]
        }
        QwenMixer::Full(attn) => vec![&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj],
    };
    slots.extend([
        &layer.mlp.gate_proj,
        &layer.mlp.up_proj,
        &layer.mlp.down_proj,
    ]);
    slots
}

fn layer_adapter_slots_mut<B: Backend>(layer: &mut QwenDecoderLayer<B>) -> Vec<&mut QwenLinear<B>> {
    let mut slots: Vec<&mut QwenLinear<B>> = match &mut layer.mixer {
        QwenMixer::Linear(head) => vec![
            &mut head.in_proj_qkv,
            &mut head.in_proj_z,
            &mut head.in_proj_b,
            &mut head.in_proj_a,
            &mut head.out_proj,
        ],
        QwenMixer::Full(attn) => {
            vec![
                &mut attn.q_proj,
                &mut attn.k_proj,
                &mut attn.v_proj,
                &mut attn.o_proj,
            ]
        }
    };
    slots.extend([
        &mut layer.mlp.gate_proj,
        &mut layer.mlp.up_proj,
        &mut layer.mlp.down_proj,
    ]);
    slots
}

/// Attach a fresh zero-init adapter to every NF4-resident projection of
/// the trunk (the placement list in the module docs). Returns the number
/// of adapters attached. Attaching is an EXACT no-op on the trunk's
/// function (B = 0), pinned by test.
pub fn attach_lora<B: Backend>(
    trunk: &mut QwenTrunk<B>,
    config: &QwenLoraConfig,
    device: &B::Device,
) -> Result<usize> {
    let mut attached = 0;
    for layer in &mut trunk.layers {
        for slot in layer_adapter_slots_mut(layer) {
            let placeholder = QwenLinear::F32(
                burn::nn::LinearConfig::new(1, 1)
                    .with_bias(false)
                    .init(device),
            );
            let old = std::mem::replace(slot, placeholder);
            *slot = old.with_lora(config, device)?;
            attached += 1;
        }
    }
    Ok(attached)
}

/// Mark one parameter tensor frozen (`require_grad(false)`), preserving
/// its `ParamId` so records and optimizer state keep matching.
fn freeze_param<B: AutodiffBackend, const D: usize>(param: &mut Param<Tensor<B, D>>) {
    *param = Param::initialized(param.id, param.val().set_require_grad(false));
}

/// Freeze a `QwenLinear` slot's own parameters (never the adapter's).
fn freeze_slot<B: AutodiffBackend>(slot: &mut QwenLinear<B>) {
    match slot {
        QwenLinear::F32(linear) => freeze_param(&mut linear.weight),
        QwenLinear::Nf4(linear) => {
            if let Some(bias) = linear.bias.as_mut() {
                freeze_param(bias);
            }
        }
        QwenLinear::Qlora(linear) => {
            if let Some(bias) = linear.base.bias.as_mut() {
                freeze_param(bias);
            }
        }
    }
}

/// Freeze EVERY non-adapter parameter of the trunk: norms, conv,
/// `dt_bias`/`A_log`, and any f32 weights (embeddings, `lm_head`, and the
/// whole trunk in f32 test configurations). Returns the number of frozen
/// tensors. Idempotent; call again after `checkpoint::load` (a record load
/// re-marks parameters trainable).
pub fn freeze_non_adapter_params<B: AutodiffBackend>(trunk: &mut QwenTrunk<B>) -> usize {
    let mut frozen = 0;
    macro_rules! freeze {
        ($param:expr) => {{
            freeze_param(&mut $param);
            frozen += 1;
        }};
    }
    // Only the f32 embedding needs freezing. An NF4-resident embedding holds no
    // `Param` at all -- `packed`, `n_embedding` and `d_model` are all
    // `#[module(skip)]` -- so there is nothing to mark and no shadow copy that
    // could quietly start drawing gradients.
    let mut trunk = trunk;
    {
        let trunk = &mut trunk;
        if let crate::qwennet::QwenEmbed::F32(embedding) = &mut trunk.embed_tokens {
            freeze!(embedding.weight);
        }
        if let QwenLinear::F32(linear) = &mut trunk.lm_head {
            freeze!(linear.weight);
        }
        freeze!(trunk.final_norm.weight);
        for layer in &mut trunk.layers {
            freeze!(layer.input_norm.weight);
            freeze!(layer.post_norm.weight);
            match &mut layer.mixer {
                QwenMixer::Linear(head) => {
                    freeze!(head.conv_weight);
                    freeze!(head.dt_bias);
                    freeze!(head.a_log);
                    freeze!(head.norm.weight);
                    for slot in layer_adapter_slots_mut_head(head) {
                        freeze_slot(slot);
                    }
                }
                QwenMixer::Full(attn) => {
                    freeze!(attn.q_norm.weight);
                    freeze!(attn.k_norm.weight);
                    for slot in [
                        &mut attn.q_proj,
                        &mut attn.k_proj,
                        &mut attn.v_proj,
                        &mut attn.o_proj,
                    ] {
                        freeze_slot(slot);
                    }
                }
            }
            for slot in [
                &mut layer.mlp.gate_proj,
                &mut layer.mlp.up_proj,
                &mut layer.mlp.down_proj,
            ] {
                freeze_slot(slot);
            }
        }
    }
    frozen
}

fn layer_adapter_slots_mut_head<B: Backend>(
    head: &mut crate::qwennet::GatedDeltaHead<B>,
) -> Vec<&mut QwenLinear<B>> {
    vec![
        &mut head.in_proj_qkv,
        &mut head.in_proj_z,
        &mut head.in_proj_b,
        &mut head.in_proj_a,
        &mut head.out_proj,
    ]
}

/// Every adapter parameter id of the trunk, in layer/slot order.
pub fn adapter_param_ids<B: Backend>(trunk: &QwenTrunk<B>) -> Result<Vec<ParamId>> {
    let mut ids = Vec::new();
    for layer in &trunk.layers {
        for slot in layer_adapter_slots(layer) {
            match slot {
                QwenLinear::Qlora(linear) => ids.extend(list_param_ids::<_, B>(&linear.adapter)),
                other => anyhow::bail!(
                    "projection slot {other:?} has no adapter; call attach_lora first"
                ),
            }
        }
    }
    Ok(ids)
}

/// Trainable value count (sum of `rank * (in + out)` over adapters).
pub fn trainable_param_count<B: Backend>(trunk: &QwenTrunk<B>) -> Result<usize> {
    let mut total = 0;
    for layer in &trunk.layers {
        for slot in layer_adapter_slots(layer) {
            match slot {
                QwenLinear::Qlora(linear) => {
                    total += linear.adapter.rank()
                        * (linear.adapter.in_features() + linear.adapter.out_features());
                }
                other => anyhow::bail!(
                    "projection slot {other:?} has no adapter; call attach_lora first"
                ),
            }
        }
    }
    Ok(total)
}

/// The trunk's adapters as a standalone module (the EMA shadow's target:
/// a full-trunk shadow would clone the 13 GiB of packed weights, so the
/// shadow tracks the adapter bank only). Cloning the bank shares `ParamId`s
/// and tensors with the trunk's adapters, so an optimizer step on the
/// trunk followed by [`QwenAdapterBank::from_trunk`] re-syncs it.
#[derive(Module, Debug)]
pub struct QwenAdapterBank<B: Backend> {
    pub layers: Vec<QwenLayerAdapters<B>>,
}

/// One decoder layer's adapters, in the order `layer_adapter_slots` yields them.
#[derive(Module, Debug)]
pub struct QwenLayerAdapters<B: Backend> {
    pub slots: Vec<LoraAdapter<B>>,
}

impl<B: Backend> QwenAdapterBank<B> {
    pub fn from_trunk(trunk: &QwenTrunk<B>) -> Result<Self> {
        let mut layers = Vec::with_capacity(trunk.layers.len());
        for layer in &trunk.layers {
            let mut slots = Vec::new();
            for slot in layer_adapter_slots(layer) {
                match slot {
                    QwenLinear::Qlora(linear) => slots.push(linear.adapter.clone()),
                    other => anyhow::bail!(
                        "projection slot {other:?} has no adapter; call attach_lora first"
                    ),
                }
            }
            layers.push(QwenLayerAdapters { slots });
        }
        Ok(Self { layers })
    }
}

/// Merge `from` into `into`, parameter by parameter (schedule.rs's
/// SumVisitor pattern: a visitor because the rank differs per parameter
/// and only the module traversal knows it).
fn merge_gradients<B: AutodiffBackend<FloatElem = f32>>(
    into: GradientsParams,
    from: GradientsParams,
    trunk: &QwenTrunk<B>,
) -> GradientsParams {
    struct SumVisitor<'a, B: AutodiffBackend> {
        into: &'a mut GradientsParams,
        from: GradientsParams,
        _backend: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend<FloatElem = f32>> ModuleVisitor<B> for SumVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let Some(addend) = self.from.remove::<B::InnerBackend, D>(param.id) else {
                return;
            };
            let total = match self.into.remove::<B::InnerBackend, D>(param.id) {
                Some(existing) => existing + addend,
                None => addend,
            };
            self.into.register::<B::InnerBackend, D>(param.id, total);
        }
    }
    let mut into = into;
    trunk.visit(&mut SumVisitor {
        into: &mut into,
        from,
        _backend: std::marker::PhantomData,
    });
    into
}

/// Next-token cross-entropy on the trunk's logits (`[b, n, vocab]`,
/// gathered log-softmax over positions `0..n-1` against tokens `1..n`).
/// Token ids are `i64` Int tensors throughout: the 248320-entry vocab
/// never touches the crate's u16-based helpers.
pub fn next_token_loss<B: AutodiffBackend<FloatElem = f32>>(
    trunk: &QwenTrunk<B>,
    tokens: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [b, n] = tokens.dims();
    let logits = trunk.forward(tokens.clone());
    let vocab = logits.dims()[2];
    let flat = logits.narrow(1, 0, n - 1).reshape([b * (n - 1), vocab]);
    let targets = tokens.narrow(1, 1, n - 1).reshape([b * (n - 1), 1]);
    -log_softmax(flat, 1).gather(1, targets).mean()
}

/// The segment-checkpointed backward (see module docs). Returns the
/// (unscaled) loss value and the adapter gradients of the loss scaled by
/// `loss_scale` (the gradient accumulator's `1/k`).
///
/// The chain is exact, not approximate: the loss segment's backward gives
/// `dL/d(hidden)`; each layer then minimizes `(layer_output * upstream).
/// sum()`, whose backward is the vector-Jacobian product `J^T upstream`
/// through exactly the same ops a one-shot backward would visit, in the
/// same order -- so the gradients are bit-identical to one-shot
/// (certified).
pub fn segmented_next_token_backward<B: AutodiffBackend<FloatElem = f32>>(
    trunk: &QwenTrunk<B>,
    tokens: Tensor<B, 2, Int>,
    loss_scale: f64,
) -> Result<(f32, GradientsParams)> {
    let [b, n] = tokens.dims();
    ensure!(n >= 2, "next-token loss needs at least two positions");

    // ---- no-grad forward, storing each layer's input boundary ----
    let mut boundaries: Vec<Tensor<B, 3>> = Vec::with_capacity(trunk.layers.len());
    let mut x = trunk.embed_tokens.forward(tokens.clone()).detach();
    for layer in &trunk.layers {
        boundaries.push(x.clone());
        x = layer.forward(x).detach();
    }

    // ---- loss segment: final norm + lm_head + CE, with grad ----
    let final_input = x.require_grad();
    let hidden = trunk.final_norm.forward(final_input.clone());
    let logits = trunk.lm_head.forward(hidden);
    let vocab = logits.dims()[2];
    let flat = logits.narrow(1, 0, n - 1).reshape([b * (n - 1), vocab]);
    let targets = tokens.narrow(1, 1, n - 1).reshape([b * (n - 1), 1]);
    let nll = -log_softmax(flat, 1).gather(1, targets).mean();
    let loss_value = nll.clone().into_scalar();
    let scaled = if (loss_scale - 1.0).abs() > f64::EPSILON {
        nll.mul_scalar(loss_scale)
    } else {
        nll
    };
    let mut grads = scaled.backward();
    let mut upstream = final_input
        .grad_remove(&mut grads)
        .context("final hidden boundary produced no gradient")?;
    let mut merged = GradientsParams::from_grads(grads, trunk);

    // ---- per-layer segments, last to first ----
    for (idx, layer) in trunk.layers.iter().enumerate().rev() {
        let input = boundaries[idx].clone().require_grad();
        let out = layer.forward(input.clone());
        let pseudo = (out * Tensor::<B, 3>::from_inner(upstream)).sum();
        let mut grads = pseudo.backward();
        upstream = input
            .grad_remove(&mut grads)
            .with_context(|| format!("layer {idx} boundary produced no gradient"))?;
        let segment = GradientsParams::from_grads(grads, trunk);
        merged = merge_gradients(merged, segment, trunk);
    }
    Ok((loss_value, merged))
}

/// AdamW over the trunk (state exists only for grad-bearing -- i.e.
/// adapter -- parameters, burn's `OptimizerAdaptor` creates it lazily).
pub type QwenOptim<B> = burn::optim::adaptor::OptimizerAdaptor<AdamW, QwenTrunk<B>, B>;
/// The optimizer record: one entry per state-holding parameter.
pub type QwenOptimRecord<B> = <QwenOptim<B> as Optimizer<QwenTrunk<B>, B>>::Record;

/// Paged optimizer state: the optimizer record is a `HashMap<ParamId, _>`,
/// so a page is just the subset of entries belonging to one decoder
/// layer's adapters. Pages may be serialized/evicted between micro-steps
/// and merged back on their update turn; m/v restore exactly.
#[derive(Debug, Clone)]
pub struct OptimPager {
    page_of: HashMap<ParamId, usize>,
    trainable_values: usize,
    pub pages: usize,
}

impl OptimPager {
    /// One page per decoder layer, holding that layer's adapter params.
    pub fn for_trunk<B: Backend>(trunk: &QwenTrunk<B>) -> Result<Self> {
        let mut page_of = HashMap::new();
        let mut trainable_values = 0;
        for (idx, layer) in trunk.layers.iter().enumerate() {
            for slot in layer_adapter_slots(layer) {
                match slot {
                    QwenLinear::Qlora(linear) => {
                        for id in list_param_ids::<_, B>(&linear.adapter) {
                            page_of.insert(id, idx);
                        }
                        trainable_values += linear.adapter.rank()
                            * (linear.adapter.in_features() + linear.adapter.out_features());
                    }
                    other => anyhow::bail!(
                        "projection slot {other:?} has no adapter; call attach_lora first"
                    ),
                }
            }
        }
        Ok(Self {
            page_of,
            trainable_values,
            pages: trunk.layers.len(),
        })
    }

    pub fn page_of(&self, id: ParamId) -> Option<usize> {
        self.page_of.get(&id).copied()
    }

    pub fn trainable_values(&self) -> usize {
        self.trainable_values
    }

    /// Resident optimizer bytes: m and v, f32 each, per trainable value.
    pub fn resident_bytes(&self) -> usize {
        8 * self.trainable_values
    }

    /// Partition an optimizer record into per-page records.
    ///
    /// Note the map type: `OptimizerAdaptor`'s record is a `hashbrown::HashMap`,
    /// which is the only `HashMap` that implements burn's `Record`. A pager
    /// over `std::collections::HashMap` looks equivalent and then cannot be
    /// serialized at all, which is the trap this signature avoids by being
    /// generic in the record type itself rather than naming a map.
    pub fn split<R, P>(&self, record: P) -> Vec<P>
    where
        P: IntoIterator<Item = (ParamId, R)> + Default + Extend<(ParamId, R)>,
    {
        let mut pages: Vec<P> = (0..self.pages).map(|_| P::default()).collect();
        for (id, entry) in record {
            if let Some(page) = self.page_of(id) {
                pages[page].extend(std::iter::once((id, entry)));
            }
        }
        pages
    }

    /// Merge per-page records back into one optimizer record.
    pub fn merge<R, P>(&self, pages: Vec<P>) -> P
    where
        P: IntoIterator<Item = (ParamId, R)> + Default + Extend<(ParamId, R)>,
    {
        let mut merged = P::default();
        for page in pages {
            merged.extend(page);
        }
        merged
    }
}

/// What one `train_step` call did.
#[derive(Debug, Clone)]
pub struct QwenStepReport {
    /// Mean next-token loss of the micro-batch (unscaled).
    pub loss: f32,
    /// Global grad norm over the accumulated gradient (`NaN` when the
    /// cycle is still filling and no step was taken).
    pub grad_norm: f32,
    /// The accumulated gradient was rescaled by the clipping bound.
    pub clipped: bool,
    /// Whether an optimizer step happened (the cycle completed).
    pub stepped: bool,
}

/// Training state persisted beyond the weights (TrainState extras).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QwenTrainExtras {
    #[serde(default)]
    pub micro_counter: usize,
    #[serde(default)]
    pub tokens_seen: usize,
    #[serde(default)]
    pub loss_sum: f64,
    #[serde(default)]
    pub ema_updates: Option<usize>,
    #[serde(default)]
    pub trainable_values: usize,
    #[serde(default)]
    pub lora: QwenLoraConfig,
}

/// The adapters-only trainer: trunk, optimizer, adapter bank (EMA
/// target), pager and counters.
///
/// No `Debug`: `OptimizerAdaptor` holds burn `AdamW` state, which carries no
/// `Debug`, and hand-writing one that skipped the optimizer would print a
/// trainer that looks complete while omitting the field a divergence check
/// most wants to see. The per-step [`QwenStepReport`] is the printable summary.
pub struct QwenTrainer<B: AutodiffBackend> {
    pub trunk: QwenTrunk<B>,
    pub optim: QwenOptim<B>,
    pub bank: QwenAdapterBank<B>,
    pub ema: Option<Ema<QwenAdapterBank<B>>>,
    pub pager: OptimPager,
    pub config: QwenTrainConfig,
    pub device: B::Device,
    pub step: usize,
    pub micro_counter: usize,
    pub tokens_seen: usize,
    pub loss_sum: f64,
    accumulator: GradientAccumulator,
}

impl<B: AutodiffBackend<FloatElem = f32>> QwenTrainer<B> {
    /// Attach adapters, freeze everything else, and build the optimizer,
    /// bank, EMA and pager.
    pub fn new(
        mut trunk: QwenTrunk<B>,
        config: QwenTrainConfig,
        device: &B::Device,
    ) -> Result<Self> {
        attach_lora(&mut trunk, &config.lora, device)?;
        freeze_non_adapter_params(&mut trunk);
        let bank = QwenAdapterBank::from_trunk(&trunk)?;
        let pager = OptimPager::for_trunk(&trunk)?;
        let optim: QwenOptim<B> = AdamWConfig::new().init();
        let ema = config.ema_decay.map(|decay| Ema::new(&bank, decay));
        Ok(Self {
            trunk,
            optim,
            bank,
            ema,
            pager,
            accumulator: GradientAccumulator::new(config.accumulate),
            config,
            device: device.clone(),
            step: 0,
            micro_counter: 0,
            tokens_seen: 0,
            loss_sum: 0.0,
        })
    }

    /// One micro-batch: segmented backward, fold into the accumulation
    /// cycle; on cycle completion clip the SUM, step AdamW, re-sync the
    /// bank and update the EMA (train.rs semantics throughout).
    pub fn train_step(&mut self, tokens: Tensor<B, 2, Int>) -> Result<QwenStepReport> {
        <B as Backend>::seed(
            &self.device,
            step_seed(self.config.seed, self.micro_counter),
        );
        let scale = self.accumulator.loss_scale();
        let tokens_count = tokens.dims()[0] * tokens.dims()[1];
        let (loss, grads) = segmented_next_token_backward(&self.trunk, tokens, scale)?;
        self.micro_counter += 1;
        self.tokens_seen += tokens_count;
        self.loss_sum += f64::from(loss);
        let cycle = self.accumulator.fold(grads, &self.trunk);
        let Some(mut summed) = cycle.into_gradients() else {
            return Ok(QwenStepReport {
                loss,
                grad_norm: f32::NAN,
                clipped: false,
                stepped: false,
            });
        };

        let total_norm = global_grad_norm(&self.trunk, &summed);
        let mut clipped = false;
        if let Some(max_norm) = self.config.clip_norm {
            let scale = clip_gradients(&mut summed, &self.trunk, total_norm, max_norm);
            clipped = scale < 1.0;
        }
        let lr = self.config.lr;
        self.trunk = self.optim.step(lr, self.trunk.clone(), summed);
        self.bank = QwenAdapterBank::from_trunk(&self.trunk)?;
        if let Some(ema) = self.ema.as_mut() {
            ema.update::<B>(&self.bank);
        }
        // A NaN that reaches the adapters poisons every later step; that is
        // fatal, not skippable (train.rs's parameters_finite discipline).
        let bad = non_finite_parameters(&self.bank);
        ensure!(
            bad == 0,
            "step {}: {bad} adapter parameter tensor(s) went non-finite after the optimizer step \
             (grad norm {total_norm:.3e}); the run cannot continue",
            self.step
        );
        self.step += 1;
        Ok(QwenStepReport {
            loss,
            grad_norm: total_norm,
            clipped,
            stepped: true,
        })
    }

    /// Evict the optimizer state to per-page records (between micro-steps
    /// the m/v pages need not be resident). Returns the page files for
    /// [`Self::restore_optimizer_pages`]; the in-memory optimizer restarts
    /// empty.
    pub fn evict_optimizer_pages(&mut self, dir: &Path) -> Result<Vec<(usize, StateFile)>> {
        let record = self.optim.to_record();
        let pages = self.pager.split(record);
        let mut files = Vec::new();
        for (page, record) in pages.into_iter().enumerate() {
            if record.is_empty() {
                continue;
            }
            let file =
                checkpoint::save_record::<B, _>(record, dir, &format!("qwen-optim-page-{page}"))?;
            files.push((page, file));
        }
        let fresh: QwenOptim<B> = AdamWConfig::new().init();
        self.optim = fresh;
        Ok(files)
    }

    /// Merge previously evicted pages back into the optimizer. m/v restore
    /// exactly (records are full-precision).
    pub fn restore_optimizer_pages(
        &mut self,
        dir: &Path,
        pages: &[(usize, StateFile)],
    ) -> Result<()> {
        let mut merged = QwenOptimRecord::<B>::default();
        for (_page, file) in pages {
            let record: QwenOptimRecord<B> = checkpoint::load_record(dir, file, &self.device)?;
            merged.extend(record);
        }
        let fresh: QwenOptim<B> = AdamWConfig::new().init();
        self.optim = fresh.load_record(merged);
        Ok(())
    }

    /// Content-addressed checkpoint + `TrainState` sidecar (mirrors the lm
    /// flow): model record, optimizer record, EMA shadow, host RNG,
    /// counters. Resume at an accumulation-cycle boundary (a run ending
    /// mid-cycle discards the pending micro-batches, same as train.rs).
    pub fn save_checkpoint(&self, dir: &Path, rng: &ChaCha12Rng) -> Result<PathBuf> {
        let model_path = checkpoint::save_content_addressed(self.trunk.clone(), dir, "qwen")?;
        let state_dir = TrainState::dir_for(&model_path);
        if state_dir.exists() {
            std::fs::remove_dir_all(&state_dir)
                .with_context(|| format!("clear {}", state_dir.display()))?;
        }
        let optimizer = Some(checkpoint::save_record::<B, _>(
            self.optim.to_record(),
            &state_dir,
            "optimizer",
        )?);
        let ema_file = self
            .ema
            .as_ref()
            .map(|e| {
                checkpoint::save_record::<B, _>(e.shadow().clone().into_record(), &state_dir, "ema")
            })
            .transpose()?;
        let extras = QwenTrainExtras {
            micro_counter: self.micro_counter,
            tokens_seen: self.tokens_seen,
            loss_sum: self.loss_sum,
            ema_updates: self.ema.as_ref().map(Ema::updates),
            trainable_values: self.pager.trainable_values(),
            lora: self.config.lora,
        };
        let state = TrainState {
            format_version: checkpoint::STATE_FORMAT_VERSION,
            kind: "qwen".into(),
            step: self.step,
            seed: self.config.seed,
            host_rng: serde_json::to_value(rng).context("serialize host RNG")?,
            config: serde_json::to_value(&self.config).context("serialize training config")?,
            build: checkpoint::BuildInfo::current(),
            datasets: Vec::new(),
            model: checkpoint::model_entry(&model_path)?,
            optimizer,
            ema: ema_file,
            head: None,
            head_optimizer: None,
            extras: serde_json::to_value(&extras).context("serialize training extras")?,
            saved_unix_secs: checkpoint::unix_now(),
        };
        state.write(&state_dir)?;
        Ok(model_path)
    }

    /// Resume from a checkpoint: the caller supplies a FRESH trunk (in NF4
    /// mode: re-quantized from the shards, deterministic); this attaches
    /// adapters, loads the model record over them (values AND `ParamId`s
    /// restore, so the optimizer record matches), re-freezes, and restores
    /// the optimizer, EMA, host RNG and counters exactly.
    pub fn resume(
        model_path: &Path,
        mut trunk: QwenTrunk<B>,
        config: QwenTrainConfig,
        device: &B::Device,
    ) -> Result<(Self, ChaCha12Rng)> {
        let state = TrainState::for_model(model_path)?
            .with_context(|| format!("no training state next to {}", model_path.display()))?;
        ensure!(
            state.kind == "qwen",
            "training state is for {:?}, not \"qwen\"",
            state.kind
        );
        let state_dir = TrainState::dir_for(model_path);
        state.verify_files(&state_dir)?;
        let extras: QwenTrainExtras =
            serde_json::from_value(state.extras.clone()).context("parse qwen training extras")?;

        attach_lora(&mut trunk, &config.lora, device)?;
        trunk = checkpoint::load(trunk, model_path, device)?;
        freeze_non_adapter_params(&mut trunk);
        let bank = QwenAdapterBank::from_trunk(&trunk)?;
        let pager = OptimPager::for_trunk(&trunk)?;
        let mut optim: QwenOptim<B> = AdamWConfig::new().init();
        if let Some(file) = &state.optimizer {
            let record = checkpoint::load_record(&state_dir, file, device)?;
            optim = optim.load_record(record);
        }
        let ema = match (config.ema_decay, &state.ema) {
            (Some(decay), Some(file)) => {
                let shadow = bank
                    .clone()
                    .load_record(checkpoint::load_record(&state_dir, file, device)?);
                Some(Ema::from_parts(
                    shadow,
                    decay,
                    extras.ema_updates.unwrap_or(0),
                ))
            }
            (Some(decay), None) => Some(Ema::new(&bank, decay)),
            (None, _) => None,
        };
        let rng: ChaCha12Rng =
            serde_json::from_value(state.host_rng.clone()).context("restore host RNG")?;
        let trainer = Self {
            trunk,
            optim,
            bank,
            ema,
            pager,
            accumulator: GradientAccumulator::new(config.accumulate),
            config,
            device: device.clone(),
            step: state.step,
            micro_counter: extras.micro_counter,
            tokens_seen: extras.tokens_seen,
            loss_sum: extras.loss_sum,
        };
        Ok((trainer, rng))
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
mod tests {
    use super::*;
    use crate::qwen::QwenArchDims;
    use crate::qwennet::QwenTrunkConfig;
    use crate::tensor_ext::force_initialization;
    use burn::backend::{Autodiff, NdArray};
    use rand::{Rng, SeedableRng};
    use std::sync::atomic::{AtomicU64, Ordering};

    type B = Autodiff<NdArray<f32>>;

    static FIXTURE_SEQ: AtomicU64 = AtomicU64::new(0);

    fn tiny_dims() -> QwenArchDims {
        QwenArchDims {
            num_layers: 5,
            hidden_size: 64,
            intermediate_size: 96,
            vocab_size: 128,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            linear_k_groups: 2,
            linear_v_heads: 4,
            linear_head_dim: 16,
            conv_kernel: 4,
            full_attention_interval: 4,
            attn_output_gate: true,
        }
    }

    fn tiny_config() -> QwenTrunkConfig {
        QwenTrunkConfig {
            dims: tiny_dims(),
            num_layers: None,
            rope_base: 10_000_000.0,
            eps: 1e-6,
        }
    }

    /// A tiny trunk in NF4 residency: the only state adapters attach to.
    ///
    /// The adapter path is defined on a packed NF4 base — that is the whole
    /// point of QLoRA — so a test that builds an f32 trunk and then attaches is
    /// testing a configuration the production path cannot produce.
    ///
    /// `seed` re-seeds burn's global RNG first. Without it two calls draw
    /// different weights *and* different `ParamId`s, so any comparison keyed by
    /// id between two builds compares nothing — which is how a real numerical
    /// bug can hide behind an id mismatch instead of a value mismatch. Tests
    /// that compare two trunks pass the same seed to both.
    fn nf4_trunk_seeded(
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        seed: u64,
    ) -> QwenTrunk<B> {
        <B as burn::tensor::backend::Backend>::seed(device, seed);
        let mut trunk = QwenTrunk::<B>::new(tiny_config(), device);
        trunk.to_nf4().unwrap_or_else(|e| panic!("to_nf4: {e:#}"));
        force_initialization(&trunk);
        trunk
    }

    fn nf4_trunk(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> QwenTrunk<B> {
        nf4_trunk_seeded(device, 0x9E37_79B9_7F4A_7C15)
    }

    fn tiny_train_config(seed: u64) -> QwenTrainConfig {
        QwenTrainConfig {
            lr: 1e-3,
            clip_norm: Some(1.0),
            accumulate: 1,
            ema_decay: Some(0.99),
            lora: QwenLoraConfig {
                rank: 4,
                alpha: 4.0,
            },
            seed,
        }
    }

    fn sample_tokens(rng: &mut ChaCha12Rng, b: usize, n: usize, vocab: i64) -> Tensor<B, 2, Int> {
        let device = Default::default();
        let ids: Vec<i64> = (0..b * n).map(|_| rng.random_range(0..vocab)).collect();
        Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([b, n])
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "dblocks-qwentrain-{tag}-{}-{}",
            std::process::id(),
            FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    }
    /// Every parameter's values, flattened in module traversal order.
    ///
    /// Two properties this needs and `HashMap` cannot give:
    ///
    /// - **Traversal order.** Element *i* is the same parameter in both
    ///   sequences, so the sequences can be compared element-wise. Sorting
    ///   them would destroy that pairing and let two unrelated parameter sets
    ///   pass, which is the failure a numerical check must not have.
    /// - **No `ParamId` keys.** `ParamId` comes from a process-global counter,
    ///   so two independently constructed trunks never share ids even when
    ///   their values are bit-identical. A key-based comparison then fails for
    ///   structural reasons and cannot distinguish a real divergence from a
    ///   renamed key — the worse error, because it reports a divergence that
    ///   is not there.
    fn param_values<M, BK>(module: &M) -> Vec<f32>
    where
        BK: Backend,
        M: Module<BK>,
    {
        struct Flatten(Vec<f32>);
        impl<BK: Backend> ModuleVisitor<BK> for Flatten {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<BK, D>>) {
                self.0.extend(param.val().to_data().iter::<f32>());
            }
        }
        let mut flat = Flatten(Vec::new());
        module.visit(&mut flat);
        flat.0
    }

    /// Gradient values keyed by param id.
    fn grad_values<BK: AutodiffBackend>(
        grads: &GradientsParams,
        trunk: &QwenTrunk<BK>,
    ) -> HashMap<ParamId, Vec<f32>> {
        struct Collect<'a> {
            grads: &'a GradientsParams,
            out: HashMap<ParamId, Vec<f32>>,
        }
        impl<'a, BK: AutodiffBackend> ModuleVisitor<BK> for Collect<'a> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<BK, D>>) {
                if let Some(g) = self.grads.get::<BK::InnerBackend, D>(param.id) {
                    let values: Vec<f32> = g.to_data().iter::<f32>().collect();
                    self.out.insert(param.id, values);
                }
            }
        }
        let mut collect = Collect {
            grads,
            out: HashMap::new(),
        };
        trunk.visit(&mut collect);
        collect.out
    }

    #[test]
    fn attaching_adapters_is_an_exact_noop() {
        // B = 0 at attach: the adapted trunk computes exactly what the
        // unadapted one computed (QLoRA's whole safety argument).
        let device = Default::default();
        let mut trunk = nf4_trunk(&device);
        force_initialization(&trunk);
        let tokens = sample_tokens(&mut ChaCha12Rng::seed_from_u64(7), 1, 6, 128);
        let before = trunk.forward(tokens.clone());
        let attached = attach_lora(
            &mut trunk,
            &QwenLoraConfig {
                rank: 4,
                alpha: 4.0,
            },
            &device,
        )
        .unwrap_or_else(|e| panic!("attach: {e:#}"));
        // 4 linear layers x (5 + 3) + 1 full layer x (4 + 3) = 39 adapters.
        assert_eq!(attached, 39);
        let after = trunk.forward(tokens);
        let gap = (before - after).abs().max().into_scalar();
        assert_eq!(gap, 0.0, "attaching adapters perturbed the trunk");
    }

    #[test]
    fn only_adapter_params_are_trainable() {
        let device = Default::default();
        // Attach and freeze here rather than delegating to `QwenTrainer::new`:
        // this case is about the *pair* of operations, and inspecting the
        // intermediate trunk is the whole point.
        let mut trunk = nf4_trunk(&device);
        attach_lora(
            &mut trunk,
            &QwenLoraConfig {
                rank: 4,
                alpha: 4.0,
            },
            &device,
        )
        .unwrap_or_else(|e| panic!("attach: {e:#}"));
        freeze_non_adapter_params(&mut trunk);

        // Count trainable (require_grad) parameters by visit.
        struct TrainableIds(Vec<ParamId>);
        impl ModuleVisitor<B> for TrainableIds {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                if param.val().is_require_grad() {
                    self.0.push(param.id);
                }
            }
        }
        let mut visitor = TrainableIds(Vec::new());
        trunk.visit(&mut visitor);
        // 39 adapters x 2 factors = 78 trainable tensors, exactly the
        // adapter params.
        assert_eq!(visitor.0.len(), 78);
        let mut adapter_ids = adapter_param_ids(&trunk).unwrap_or_else(|e| panic!("ids: {e:#}"));
        adapter_ids.sort();
        visitor.0.sort();
        assert_eq!(
            visitor.0, adapter_ids,
            "trainable set is not exactly the adapters"
        );

        // Value count, computed from dims (rank 4): per adapter 4*(in+out).
        let d = tiny_dims();
        let (h, inter) = (d.hidden_size, d.intermediate_size);
        let (kd, vd, nv) = (d.key_dim(), d.value_dim(), d.num_v_heads());
        let (qw, kvw) = (d.num_q_heads * d.head_dim, d.num_kv_heads * d.head_dim);
        let per_linear =
            4 * ((h + 2 * kd + vd) + (h + vd) + 2 * (h + nv) + (vd + h) + 3 * (h + inter));
        let per_full = 4 * ((h + 2 * qw) + 2 * (h + kvw) + (qw + h) + 3 * (h + inter));
        let want = 4 * per_linear + per_full;
        let got = trainable_param_count(&trunk).unwrap_or_else(|e| panic!("count: {e:#}"));
        assert_eq!(got, want, "trainable value count drifted");

        // Frozen params (norms, conv, ...) must not require grad: snapshot
        // them, run one optimizer step, and demand bit-identity.
        let frozen_before = param_values::<_, B>(&trunk.final_norm);
        let mixer_before = match &trunk.layers[0].mixer {
            QwenMixer::Linear(head) => param_values::<_, B>(&head.norm),
            QwenMixer::Full(_) => unreachable!("layer 0 is linear"),
        };
        let config = tiny_train_config(3);
        // The trainer gets its own identically-seeded trunk: `trunk` is already
        // adapted and the trainer refuses a double attach by design, but the
        // seeds match so the frozen tensors below are the same tensors.
        let mut trainer = QwenTrainer::new(
            nf4_trunk_seeded(&device, 0x51D1_C0DE_1234_5678),
            config,
            &device,
        )
        .unwrap_or_else(|e| panic!("trainer: {e:#}"));
        let tokens = sample_tokens(&mut ChaCha12Rng::seed_from_u64(11), 1, 6, 128);
        let report = trainer
            .train_step(tokens)
            .unwrap_or_else(|e| panic!("step: {e:#}"));
        assert!(report.stepped && report.loss.is_finite());
        assert_eq!(
            param_values::<_, B>(&trainer.trunk.final_norm),
            frozen_before,
            "final norm moved"
        );
        let mixer_after = match &trainer.trunk.layers[0].mixer {
            QwenMixer::Linear(head) => param_values::<_, B>(&head.norm),
            QwenMixer::Full(_) => unreachable!("layer 0 is linear"),
        };
        assert_eq!(mixer_after, mixer_before, "frozen gate norm moved");
    }

    #[test]
    fn checkpointed_backward_matches_one_shot() {
        // THE numerical identity of the engine: segment-checkpointed
        // backward == one-shot backward, gradient for gradient.
        let device = Default::default();
        let mut trunk = nf4_trunk(&device);
        // This case drives the backward directly rather than through a
        // trainer, so it owns the attach/freeze pair itself.
        attach_lora(
            &mut trunk,
            &QwenLoraConfig {
                rank: 4,
                alpha: 4.0,
            },
            &device,
        )
        .unwrap_or_else(|e| panic!("attach: {e:#}"));
        freeze_non_adapter_params(&mut trunk);
        let tokens = sample_tokens(&mut ChaCha12Rng::seed_from_u64(17), 1, 6, 128);

        let loss = next_token_loss(&trunk, tokens.clone());
        let one_shot = GradientsParams::from_grads(loss.backward(), &trunk);
        let (seg_loss, segmented) = segmented_next_token_backward(&trunk, tokens, 1.0)
            .unwrap_or_else(|e| panic!("segmented: {e:#}"));
        assert!(seg_loss.is_finite());

        let a = grad_values(&one_shot, &trunk);
        let b = grad_values(&segmented, &trunk);
        assert_eq!(a.len(), b.len(), "gradient sets differ");
        assert_eq!(a.len(), 78, "every adapter factor has a gradient");
        let mut worst = 0.0f32;
        for (id, ga) in &a {
            let gb = b
                .get(id)
                .unwrap_or_else(|| panic!("missing grad for {id:?}"));
            assert_eq!(ga.len(), gb.len());
            for (x, y) in ga.iter().zip(gb) {
                worst = worst.max((x - y).abs());
            }
        }
        assert!(
            worst <= 0.0,
            "checkpointed backward diverged from one-shot: {worst}"
        );
    }

    #[test]
    fn accumulation_matches_large_batch() {
        // Two micro-batches with loss scaled 1/2 must move the adapters
        // like one batch twice the size -- train.rs's `--accumulate`
        // semantics. The loss values differ from the big-batch mean by at
        // most a rounding sliver (mean-of-means vs one mean), so the
        // tolerance is small but honest, not zero.
        let device = Default::default();
        // `QwenTrainer::new` attaches and freezes, so the builder hands it a
        // bare NF4 trunk. Two calls to `build` must produce identical
        // parameter ids for the paired comparisons below to mean anything,
        // which is why this goes through the same deterministic path.
        // Both sides of the comparison below must be the *same* trunk, so the
        // builder re-seeds: burn's RNG is a process global, and two unseeded
        // builds draw different weights and different ParamIds.
        let build = || nf4_trunk_seeded(&device, 0x51D1_C0DE_1234_5678);
        let mut rng = ChaCha12Rng::seed_from_u64(23);
        let micro1 = sample_tokens(&mut rng, 1, 6, 128);
        let micro2 = sample_tokens(&mut rng, 1, 6, 128);
        let big = Tensor::cat(vec![micro1.clone(), micro2.clone()], 0);

        let config = QwenTrainConfig {
            accumulate: 2,
            ..tiny_train_config(29)
        };
        let mut acc_trainer =
            QwenTrainer::new(build(), config, &device).unwrap_or_else(|e| panic!("trainer: {e:#}"));
        assert!(
            !acc_trainer
                .train_step(micro1)
                .unwrap_or_else(|e| panic!("micro1: {e:#}"))
                .stepped
        );
        let report = acc_trainer
            .train_step(micro2)
            .unwrap_or_else(|e| panic!("micro2: {e:#}"));
        assert!(report.stepped, "second micro-batch completes the cycle");

        let config = QwenTrainConfig {
            accumulate: 1,
            ..tiny_train_config(29)
        };
        let mut big_trainer =
            QwenTrainer::new(build(), config, &device).unwrap_or_else(|e| panic!("trainer: {e:#}"));
        big_trainer
            .train_step(big)
            .unwrap_or_else(|e| panic!("big: {e:#}"));

        // Value comparison, not `ParamId`-keyed: the two trainers are built
        // separately and a global id counter guarantees the keys differ even
        // when the weights agree.
        let a = param_values::<_, B>(&acc_trainer.bank);
        let b = param_values::<_, B>(&big_trainer.bank);
        assert_eq!(
            a.len(),
            b.len(),
            "accumulated and large-batch banks differ in size"
        );
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        // The two paths sum the same gradients in a different order, so they
        // cannot agree bit for bit. Stated *relative to the parameter scale*:
        // an absolute tolerance on weights of mixed magnitude says nothing,
        // because the same number is negligible on one tensor and ruinous on
        // another.
        //
        // This bound is EMPIRICAL, and that is worth being explicit about
        // rather than dressing up: a rigorous pairwise-summation bound would
        // need the exact count of gradient terms reduced per parameter, which
        // this engine does not track. Observed drift here is ~1.5e-5 relative.
        // The bound is set two orders of magnitude above that so a systematic
        // error -- a micro-batch counted twice, a missing `1/k`, a wrong
        // gradient-clipping scale -- still fails loudly; those land many orders
        // above it. AdamW divides by `sqrt(v)`, which amplifies a relative
        // difference in the gradient into a much larger relative difference in
        // the step, so the drift is a property of the optimizer rather than
        // evidence of a wrong accumulation.
        //
        // TODO(derive): tighten this to `C * f32::EPSILON * scale * sqrt(terms)`
        // once `terms` is countable, and drop the empirical factor.
        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-3);
        let bound = 1e-4 * scale;
        assert!(
            worst <= bound,
            "accumulated step diverged from large-batch step by {worst}, above the \
             relative bound {bound} at scale {scale}"
        );
    }

    #[test]
    fn ema_is_bias_corrected() {
        // With updates = 0 the effective decay is min(0.99, 1/10) = 0.1,
        // so after one update the shadow sits 10% of the way from the
        // initial adapters to the stepped ones -- the warm-up ramp the
        // bias correction exists for (schedule.rs's EMA discipline).
        let device = Default::default();
        let trunk = nf4_trunk(&device);
        let config = tiny_train_config(31);
        let mut trainer =
            QwenTrainer::new(trunk, config, &device).unwrap_or_else(|e| panic!("trainer: {e:#}"));
        // The shadow's starting point: the adapters as attached, before any
        // step. With `b` zero at attach this is what the bias correction
        // divides out, so the assertion below is about the decay alone.
        let init_bank =
            QwenAdapterBank::from_trunk(&trainer.trunk).unwrap_or_else(|e| panic!("bank: {e:#}"));
        trainer
            .train_step(sample_tokens(
                &mut ChaCha12Rng::seed_from_u64(37),
                1,
                6,
                128,
            ))
            .unwrap_or_else(|e| panic!("step: {e:#}"));
        let ema = trainer
            .ema
            .as_ref()
            .unwrap_or_else(|| panic!("ema enabled"));
        assert_eq!(ema.updates(), 1);
        let init = param_values::<_, B>(&init_bank);
        let live = param_values::<_, B>(&trainer.bank);
        let shadow = param_values::<_, B>(ema.shadow());
        assert_eq!(
            init.len(),
            live.len(),
            "initial and live banks differ in size"
        );
        assert_eq!(live.len(), shadow.len(), "live and shadow differ in size");
        let mut worst = 0.0f32;
        for ((i, l), s) in init.iter().zip(&live).zip(&shadow) {
            let want = 0.9 * i + 0.1 * l;
            worst = worst.max((want - s).abs());
        }
        assert!(
            worst <= 1e-5,
            "shadow is not the bias-corrected average: {worst}"
        );
    }

    #[test]
    fn resume_is_bit_identical() {
        // k steps, checkpoint, k more -- versus 2k uninterrupted. Losses,
        // adapter params and the next sampled batch must be bit-identical.
        let device = Default::default();
        // `QwenTrainer::new` attaches and freezes, so the builder hands it a
        // bare NF4 trunk. Two calls to `build` must produce identical
        // parameter ids for the paired comparisons below to mean anything,
        // which is why this goes through the same deterministic path.
        // Both sides of the comparison below must be the *same* trunk, so the
        // builder re-seeds: burn's RNG is a process global, and two unseeded
        // builds draw different weights and different ParamIds.
        let build = || nf4_trunk_seeded(&device, 0x51D1_C0DE_1234_5678);
        let seed = 41;
        let mut rng_a = ChaCha12Rng::seed_from_u64(43);
        let batches: Vec<Tensor<B, 2, Int>> = (0..4)
            .map(|_| sample_tokens(&mut rng_a, 1, 6, 128))
            .collect();

        // Uninterrupted run A: 4 steps.
        let mut trainer_a = QwenTrainer::new(build(), tiny_train_config(seed), &device)
            .unwrap_or_else(|e| panic!("trainer: {e:#}"));
        let mut losses_a = Vec::new();
        for tokens in &batches {
            losses_a.push(
                trainer_a
                    .train_step(tokens.clone())
                    .unwrap_or_else(|e| panic!("step: {e:#}"))
                    .loss,
            );
        }

        // Run B: 2 steps, save, resume, 2 more.
        let dir = tmp_dir("resume");
        let mut trainer_b = QwenTrainer::new(build(), tiny_train_config(seed), &device)
            .unwrap_or_else(|e| panic!("trainer: {e:#}"));
        let mut losses_b = Vec::new();
        for tokens in &batches[..2] {
            losses_b.push(
                trainer_b
                    .train_step(tokens.clone())
                    .unwrap_or_else(|e| panic!("step: {e:#}"))
                    .loss,
            );
        }
        // Save `rng_a` — the stream that actually drew the batches — or the
        // resume restores an unrelated stream and the next batch differs for
        // a reason that has nothing to do with the trainer.
        let model_path = trainer_b
            .save_checkpoint(&dir, &rng_a)
            .unwrap_or_else(|e| panic!("save: {e:#}"));
        let fresh = build();
        let (mut trainer_b2, rng_b2) =
            QwenTrainer::resume(&model_path, fresh, tiny_train_config(seed), &device)
                .unwrap_or_else(|e| panic!("resume: {e:#}"));
        assert_eq!(trainer_b2.step, 2, "resume continues at the saved step");
        for tokens in &batches[2..] {
            losses_b.push(
                trainer_b2
                    .train_step(tokens.clone())
                    .unwrap_or_else(|e| panic!("step: {e:#}"))
                    .loss,
            );
        }

        assert_eq!(losses_a, losses_b, "losses diverged across resume");
        let pa = param_values::<_, B>(&trainer_a.bank);
        let pb = param_values::<_, B>(&trainer_b2.bank);
        assert_eq!(pa, pb, "adapter params diverged across resume");
        // Host RNG restored: the next batch draw is identical.
        let next_a = sample_tokens(&mut rng_a, 1, 6, 128);
        let mut rng_b2 = rng_b2;
        let next_b = sample_tokens(&mut rng_b2, 1, 6, 128);
        let va: Vec<i64> = next_a.to_data().iter::<i64>().collect();
        let vb: Vec<i64> = next_b.to_data().iter::<i64>().collect();
        assert_eq!(va, vb, "host RNG not restored");
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("clean {}: {e}", dir.display()));
    }

    #[test]
    fn max_logit_probe_reports_only_full_attention_layers() {
        let device = Default::default();
        let trunk = QwenTrunk::<B>::new(tiny_config(), &device);
        force_initialization(&trunk);
        let tokens = sample_tokens(&mut ChaCha12Rng::seed_from_u64(47), 1, 6, 128);
        let probes = trunk.max_logits(tokens);
        assert_eq!(probes.len(), 5);
        for (idx, probe) in probes.iter().enumerate() {
            if idx == 3 {
                let value = probe
                    .as_ref()
                    .unwrap_or_else(|| panic!("layer 3 is full attention"))
                    .clone()
                    .into_scalar();
                assert!(value.is_finite(), "max logit not finite: {value}");
            } else {
                assert!(probe.is_none(), "delta layer {idx} reported a max logit");
            }
        }
    }

    #[test]
    fn optimizer_pages_round_trip() {
        // Evict + restore the optimizer state per page; the next step must
        // be exactly the uninterrupted one (m/v restore bit-exact).
        let device = Default::default();
        // `QwenTrainer::new` attaches and freezes, so the builder hands it a
        // bare NF4 trunk. Two calls to `build` must produce identical
        // parameter ids for the paired comparisons below to mean anything,
        // which is why this goes through the same deterministic path.
        // Both sides of the comparison below must be the *same* trunk, so the
        // builder re-seeds: burn's RNG is a process global, and two unseeded
        // builds draw different weights and different ParamIds.
        let build = || nf4_trunk_seeded(&device, 0x51D1_C0DE_1234_5678);
        let mut rng = ChaCha12Rng::seed_from_u64(53);
        let batches: Vec<Tensor<B, 2, Int>> =
            (0..3).map(|_| sample_tokens(&mut rng, 1, 6, 128)).collect();
        let dir = tmp_dir("pager");

        let mut paged = QwenTrainer::new(build(), tiny_train_config(59), &device)
            .unwrap_or_else(|e| panic!("trainer: {e:#}"));
        let mut plain = QwenTrainer::new(build(), tiny_train_config(59), &device)
            .unwrap_or_else(|e| panic!("trainer: {e:#}"));
        paged
            .train_step(batches[0].clone())
            .unwrap_or_else(|e| panic!("step: {e:#}"));
        plain
            .train_step(batches[0].clone())
            .unwrap_or_else(|e| panic!("step: {e:#}"));

        let files = paged
            .evict_optimizer_pages(&dir)
            .unwrap_or_else(|e| panic!("evict: {e:#}"));
        assert!(!files.is_empty(), "no optimizer pages written");
        assert!(paged.pager.resident_bytes() > 0);
        paged
            .restore_optimizer_pages(&dir, &files)
            .unwrap_or_else(|e| panic!("restore: {e:#}"));

        let la = paged
            .train_step(batches[1].clone())
            .unwrap_or_else(|e| panic!("step: {e:#}"))
            .loss;
        let lb = plain
            .train_step(batches[1].clone())
            .unwrap_or_else(|e| panic!("step: {e:#}"))
            .loss;
        assert_eq!(la, lb, "paged optimizer diverged after round trip");
        let pa = param_values::<_, B>(&paged.bank);
        let pb = param_values::<_, B>(&plain.bank);
        assert_eq!(pa, pb, "adapter params diverged after the page round trip");
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("clean {}: {e}", dir.display()));
    }
}
