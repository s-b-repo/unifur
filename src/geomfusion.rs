//! The geometric-reasoning stream attached to the Qwen3.8 trunk: the model
//! reasons geometrically, not by brute-force next-token guessing.
//!
//! [`crate::geometry::GeometricReasoner`] answers synthetic point scenes by
//! relaxing a geometric state into a certified attractor. [`GeomFusion`] is
//! the same machinery attached to the language trunk's hidden states: the
//! trunk's `[b, n, 5120]` representation is projected into a geometric
//! stream, refined by the certified energy descent (the step size comes from
//! the closed-form Lipschitz bound of [`crate::geometry::lipschitz_step`], so
//! the energy decrease is watched by the `geomfusion` certificates, not
//! assumed), read out through MoSME boxes named for reasoning kinds, and
//! added back as a residual.
//!
//! Two properties make the stream more than decoration:
//!
//! - **It is identity at birth.** The output projection is zero-initialized,
//!   so a fresh fusion layer returns the trunk hidden state bit for bit
//!   (`geomfusion/fusion_is_identity_at_zero_projection`). Training turns the
//!   stream on; attaching it to a trained trunk perturbs nothing.
//! - **It can change answers.** The refinement is a real dynamics: deeper
//!   relaxation moves the representation (the read/write coupling of
//!   [`crate::geometry::GeomBlock::refine`]), the readout routes the moved
//!   state, and the residual carries the difference back into the trunk.
//!   Refinement depth is an inference-time knob ([`GeomFusion::forward`]'s
//!   `depth`), the same test-time-compute dial the standalone reasoner has.
//!
//! The routing never hand-rolls a scatter: the readout goes through
//! [`crate::mosme::HierarchicalRouter`], whose gates come from the certified
//! [`crate::moe::scatter_gates`]. The broadcast-scatter bug that once made
//! every readout row sum to `n_boxes` (see
//! `docs/Geometric-Reasoning-Flaws.md`) is exactly what
//! `geomfusion/fused_readout_gates_form_a_distribution` would catch.
//!
//! The composed model, its training and its proof live below:
//!
//! - [`FusedQwen`] composes [`crate::qwennet::QwenTrunk`] and [`GeomFusion`]:
//!   trunk hidden states -> geometric stream -> untied `lm_head`. A fresh
//!   model is the trunk bit for bit (zero output projection), certified as
//!   `geomfusion/fused_qwen_is_the_trunk_at_zero_fusion`.
//! - [`FusedQwenTrainer`] trains it adapters-only: LoRA on the trunk through
//!   the [`crate::qwentrain`] machinery (segment-checkpointed backward,
//!   accumulation, EMA, verified resume) plus a second AdamW over the
//!   geometric stream's own (tiny) parameters. The corpus is the geometric
//!   scene format of [`crate::geometry`], tokenized with the trunk's BPE;
//!   the objective is next-token cross-entropy on the answer span
//!   ([`GeomObjective::Answer`]) optionally plus the EDM denoising term on
//!   the geometric state ([`GeomObjective::Diffusion`]).
//! - [`prove`] is the proof harness: held-out accuracy against the
//!   empirical answer distribution, a refinement sweep (accuracy and energy
//!   vs depth, agree-with-depth-1 trace dependence), and every emitted
//!   answer checked through the exact rational kernel
//!   ([`crate::geomkernel`]) -- the model proposes, the kernel decides
//!   verified / falsified / indeterminate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::module::{list_param_ids, Initializer, Module, ModuleVisitor, Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::bpe::BpeTokenizer;
use crate::checkpoint::{self, StateFile, TrainState};
use crate::corpus::TokenCorpus;
use crate::expert_index::{BalanceWeights, BoxSpec, ExpertSpec, MosmeSpec};
use crate::geometry::{GeomBlock, GeomConfig, GeomMeta, GeomObjective, RefineTrace, RoutingInfo};
use crate::geomkernel::{self, Q};
use crate::mosme::{HierarchicalRouter, MosmeConfig};
use crate::qwennet::{QwenLoraConfig, QwenTrunk};
use crate::qwentrain::{self, QwenAdapterBank, QwenOptim, QwenOptimRecord};
use crate::schedule::{clip_gradients, Ema, GradientAccumulator};

/// The reasoning kinds the readout's MoSME boxes are named for. The box
/// count is the stream's: adding a kind is an architecture change, not a
/// config value.
pub const REASONING_KINDS: [&str; 4] = ["math", "logic", "proof", "code"];

/// Configuration for a [`GeomFusion`] layer.
///
/// `#[serde(default)]` at the container level: a sidecar written by an older
/// build parses, with the missing fields taking the declared defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeomFusionConfig {
    /// The trunk's hidden width (5120 on Qwen3.8-27B).
    pub hidden_size: usize,
    /// The width the trunk hidden state is projected into for the geometric
    /// stream, and routed back out of.
    pub geom_hidden: usize,
    /// Geometric slots `K` -- the concept points the tokens pool into.
    pub slots: usize,
    /// Dimension of the geometric space the slots live in.
    pub dim: usize,
    /// Refinement blocks; each redraws its landscape from the previous
    /// block's refined stream.
    pub num_blocks: usize,
    /// Relaxation iterations per block (overridable per forward pass).
    pub refine_steps: usize,
    /// Width of the sigma-conditioning embedding inside each block.
    pub cond_hidden_size: usize,
    pub frequency_embedding_size: usize,
    pub layer_norm_eps: f64,
    /// The repulsion coefficient the blocks are initialized at; each block
    /// learns its own.
    pub repulsion: f64,
    /// Experts per reasoning-kind box in the MoSME readout.
    pub moe_experts: usize,
    /// Experts selected per box, and boxes selected overall.
    pub moe_top_k: usize,
}

impl Default for GeomFusionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 5120,
            geom_hidden: 128,
            slots: 32,
            dim: 32,
            num_blocks: 2,
            refine_steps: 4,
            cond_hidden_size: 64,
            frequency_embedding_size: 16,
            layer_norm_eps: 1e-5,
            repulsion: 0.5,
            moe_experts: 2,
            moe_top_k: 1,
        }
    }
}

impl GeomFusionConfig {
    /// The small configuration, for CPU smoke runs and the certificate suite.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 64,
            geom_hidden: 32,
            slots: 8,
            dim: 16,
            cond_hidden_size: 16,
            frequency_embedding_size: 8,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.hidden_size >= 1, "hidden_size must be positive");
        anyhow::ensure!(self.geom_hidden >= 1, "geom_hidden must be positive");
        anyhow::ensure!(self.slots >= 2, "slots must be at least 2");
        anyhow::ensure!(self.dim >= 1, "dim must be positive");
        anyhow::ensure!(
            self.num_blocks >= 1 && self.refine_steps >= 1,
            "num_blocks and refine_steps must be positive"
        );
        anyhow::ensure!(
            self.cond_hidden_size >= 1,
            "cond_hidden_size must be positive"
        );
        anyhow::ensure!(
            self.frequency_embedding_size >= 2 && self.frequency_embedding_size % 2 == 0,
            "frequency_embedding_size must be even"
        );
        anyhow::ensure!(self.layer_norm_eps > 0.0, "layer_norm_eps must be positive");
        // `GeomBlock` inverts softplus on this value: at or below zero the
        // raw logit is ln(exp(r) - 1) of a non-positive number.
        anyhow::ensure!(
            self.repulsion.is_finite() && self.repulsion > 0.0,
            "repulsion must be positive and finite"
        );
        anyhow::ensure!(
            self.moe_experts >= 1
                && self.moe_top_k >= 1
                && self.moe_top_k <= self.moe_experts.min(REASONING_KINDS.len()),
            "moe_top_k must be in [1, {}], not {}",
            self.moe_experts.min(REASONING_KINDS.len()),
            self.moe_top_k
        );
        Ok(())
    }
}

/// What the geometric stream produced, before it is added to the trunk
/// hidden state. Private: [`GeomFusion::forward`] is the entry point, and the
/// tests read the parts through it.
struct GeometricPath<B: Backend> {
    /// The routed geometric readout `[b, n, geom_hidden]` -- what `out_proj`
    /// maps back into the trunk width.
    mixture: Tensor<B, 3>,
    /// The final relaxed state `[b, slots, dim]`.
    final_state: Tensor<B, 3>,
    /// One [`RefineTrace`] per block: the energy the certificate watches.
    traces: Vec<RefineTrace>,
    /// The composed routing gates `[b * n, boxes * experts]`.
    gates: Tensor<B, 2>,
    routing: RoutingInfo<B>,
}

/// What one fused forward pass returns.
#[derive(Debug, Clone)]
pub struct FusedForward<B: Backend> {
    /// `trunk hidden + out_proj(geometric readout)`, `[b, n, hidden_size]`.
    pub hidden: Tensor<B, 3>,
    /// The final relaxed geometric state `[b, slots, dim]`.
    pub final_state: Tensor<B, 3>,
    /// One [`RefineTrace`] per refinement block: per-step mean energy and
    /// displacement. The `geomfusion/relaxation_reduces_energy_monotonically`
    /// certificate reads these.
    pub traces: Vec<RefineTrace>,
    /// The composed routing gates `[b * n, boxes * experts]`: each row sums
    /// to 1 (`geomfusion/fused_readout_gates_form_a_distribution`).
    pub gates: Tensor<B, 2>,
    /// The router diagnostics (balance breakdown, entropies, box loads).
    pub routing: RoutingInfo<B>,
}

/// The geometric-reasoning stream fused onto a trunk's hidden states.
///
/// `hidden [b, n, hidden_size]` -> project to `geom_hidden` -> pool into
/// `slots` concept points of dimension `dim` -> `num_blocks` certified
/// refinement blocks ([`GeomBlock`]: geodesic attention under a learned
/// metric, then Lipschitz-certified relaxation) -> MoSME readout over the
/// [`REASONING_KINDS`] boxes -> zero-initialized projection back to
/// `hidden_size`, added as a residual.
#[derive(Module, Debug)]
pub struct GeomFusion<B: Backend> {
    in_proj: Linear<B>,
    /// Per-slot projection of the chunk-pooled stream: slot `k` starts near
    /// the region it was pooled from.
    slot_init: Linear<B>,
    blocks: Vec<GeomBlock<B>>,
    router: HierarchicalRouter<B>,
    experts: Vec<Vec<Linear<B>>>,
    /// Zero-initialized: a fresh fusion layer is the identity on the trunk,
    /// exactly (the same argument as the DiT zero-init and the LoRA `B`
    /// factor). No bias, so "zero" is the whole map, not a shifted one.
    out_proj: Linear<B>,
    balance: BalanceWeights,
    hidden_size: usize,
    geom_hidden: usize,
    slots: usize,
    refine_steps: usize,
}

impl<B: Backend<FloatElem = f32>> GeomFusion<B> {
    /// Build a fusion layer from `config`.
    ///
    /// Fallible because [`GeomFusionConfig::validate`] is: an invalid
    /// geometry or routing shape is a caller mistake that has to name itself,
    /// and a `new` that panicked on bad input would take a training run down
    /// instead of reporting which field was out of range.
    pub fn new(config: &GeomFusionConfig, device: &B::Device) -> anyhow::Result<Self> {
        config.validate()?;
        // The refinement blocks are the standalone reasoner's, unchanged: the
        // block's own `GeomConfig` spells the geometric stream's widths, and
        // the scene/corpus fields it ignores keep their defaults.
        let geom_config = GeomConfig {
            hidden_size: config.geom_hidden,
            slots: config.slots,
            dim: config.dim,
            num_blocks: config.num_blocks,
            refine_steps: config.refine_steps,
            cond_hidden_size: config.cond_hidden_size,
            frequency_embedding_size: config.frequency_embedding_size,
            layer_norm_eps: config.layer_norm_eps,
            repulsion: config.repulsion,
            ..GeomConfig::default()
        };
        let spec = MosmeSpec {
            boxes: REASONING_KINDS
                .iter()
                .map(|kind| {
                    BoxSpec::new(
                        *kind,
                        format!("{kind} reasoning"),
                        (0..config.moe_experts)
                            .map(|j| {
                                ExpertSpec::new(
                                    format!("{kind}/expert_{j}"),
                                    format!("{kind} expert {j}"),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
            top_box: config.moe_top_k,
            top_expert: config.moe_top_k,
            // The readout routes on the token state alone, as the standalone
            // readout does: the router input is the state unchanged.
            route_on_tokens: false,
            balance: BalanceWeights::default(),
        };
        let balance = spec.balance;
        let router_config = MosmeConfig::new(config.geom_hidden, config.geom_hidden, spec);
        Ok(Self {
            in_proj: LinearConfig::new(config.hidden_size, config.geom_hidden)
                .with_bias(true)
                .init(device),
            slot_init: LinearConfig::new(config.geom_hidden, config.dim)
                .with_bias(true)
                .init(device),
            blocks: (0..config.num_blocks)
                .map(|_| GeomBlock::new(&geom_config, device))
                .collect(),
            router: HierarchicalRouter::new(&router_config, device),
            experts: (0..REASONING_KINDS.len())
                .map(|_| {
                    (0..config.moe_experts)
                        .map(|_| {
                            LinearConfig::new(config.geom_hidden, config.geom_hidden)
                                .with_initializer(Initializer::Normal {
                                    mean: 0.0,
                                    std: 0.02,
                                })
                                .with_bias(true)
                                .init(device)
                        })
                        .collect()
                })
                .collect(),
            out_proj: LinearConfig::new(config.geom_hidden, config.hidden_size)
                .with_bias(false)
                .with_initializer(Initializer::Zeros)
                .init(device),
            balance,
            hidden_size: config.hidden_size,
            geom_hidden: config.geom_hidden,
            slots: config.slots,
            refine_steps: config.refine_steps,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn geom_hidden(&self) -> usize {
        self.geom_hidden
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The configured refinement depth per block (the default `depth`).
    pub fn refine_steps(&self) -> usize {
        self.refine_steps
    }

    /// The inner two-level router, for callers that need the raw gates.
    pub fn router(&self) -> &HierarchicalRouter<B> {
        &self.router
    }

    /// Mutable access to the inner router, e.g. to enable or disable an
    /// expert (`HierarchicalRouter::set_enabled`).
    pub fn router_mut(&mut self) -> &mut HierarchicalRouter<B> {
        &mut self.router
    }

    /// The initial geometric state: the stream's tokens pooled into `slots`
    /// contiguous chunks, each projected into the geometric space -- the same
    /// chunk-pooled init the standalone reasoner uses, so slot `k` starts
    /// near the region it was pooled from.
    fn init_slots(&self, h: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n, _] = h.dims();
        let mut pooled = Vec::with_capacity(self.slots);
        for k in 0..self.slots {
            let start = k * n / self.slots;
            let end = (k + 1) * n / self.slots;
            let chunk = h.clone().narrow(1, start, end - start);
            pooled.push(chunk.mean_dim(1));
        }
        let pooled = Tensor::cat(pooled, 1);
        self.slot_init.forward(pooled)
    }

    /// The geometric stream, end to end, stopping before the residual: the
    /// refined stream routed through the reasoning-kind boxes.
    fn geometric_path(
        &self,
        hidden: &Tensor<B, 3>,
        depth: Option<usize>,
    ) -> anyhow::Result<GeometricPath<B>> {
        let [batch, n, _] = hidden.dims();
        let mut h = self.in_proj.forward(hidden.clone());
        let mut z = self.init_slots(&h);
        let mut traces = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            // The clean reasoning path: noise scale zero. The FiLM
            // conditioning is what makes a block a denoiser; at sigma zero
            // the blocks are the standalone reasoner's clean path.
            let (h_next, z_next, trace) = block.refine(h, z, 0.0, depth);
            traces.push(trace);
            h = h_next;
            z = z_next;
        }

        let mut path = self.readout(&h, batch, n)?;
        path.final_state = z;
        path.traces = traces;
        Ok(path)
    }

    /// The fused forward pass: the geometric stream's readout, projected back
    /// to the trunk width and added to the trunk hidden state.
    ///
    /// `hidden` is `[b, n, hidden_size]` -- straight from
    /// [`crate::qwennet::QwenTrunk::forward_hidden`]. `depth` overrides the
    /// refinement depth (the test-time-compute knob: `Some(steps)` runs that
    /// many relaxation iterations per block, `None` uses the configured
    /// `refine_steps`). The caller needs at least `slots` tokens: the slot
    /// init pools contiguous chunks, and an empty chunk has no mean.
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        depth: Option<usize>,
    ) -> anyhow::Result<FusedForward<B>> {
        let [_, n, width] = hidden.dims();
        anyhow::ensure!(
            width == self.hidden_size,
            "hidden width {width} does not match the fusion layer's hidden_size {}",
            self.hidden_size
        );
        anyhow::ensure!(
            n >= self.slots,
            "the geometric stream pools {n} tokens into {} slots; it needs at least as many tokens as slots",
            self.slots
        );
        let path = self.geometric_path(&hidden, depth)?;
        let fused = hidden.add(self.out_proj.forward(path.mixture.clone()));
        Ok(FusedForward {
            hidden: fused,
            final_state: path.final_state,
            traces: path.traces,
            gates: path.gates,
            routing: path.routing,
        })
    }
}

// ---------------------------------------------------------------- scenes ---

/// A geometric scene parsed back out of its rendered byte form (the corpus
/// layout of [`crate::geometry`]: header, two-digit coordinates per point,
/// kind letter, four reference slots, `?`, answer). Parsing is how the proof
/// harness hands the model's answer to the exact kernel: the bytes the model
/// reads are the only scene description it gets, and the check must rebuild
/// the same scene from them.
#[derive(Debug, Clone)]
pub struct ParsedScene {
    /// Point coordinates on the integer grid, in scene order (`A` first).
    pub points: Vec<(i32, i32)>,
    /// The question kind (`nearest`, `farthest`, `direction`, `inside`,
    /// `collinear`).
    pub kind: String,
    /// The reference point indices the question asks about (`_` slots
    /// skipped).
    pub refs: Vec<usize>,
    /// The gold answer byte.
    pub answer: u8,
}

fn scene_byte(scene: &[u16], idx: usize) -> anyhow::Result<u8> {
    let token = scene.get(idx).copied().ok_or_else(|| {
        anyhow::anyhow!(
            "scene of {} tokens has no offset {idx}",
            scene.len()
        )
    })?;
    anyhow::ensure!(
        token <= 0xFF,
        "scene token {token} at offset {idx} is not a byte"
    );
    Ok(token as u8)
}

fn two_digits(scene: &[u16], idx: usize, what: &str) -> anyhow::Result<i32> {
    let hi = scene_byte(scene, idx)?;
    let lo = scene_byte(scene, idx + 1)?;
    anyhow::ensure!(
        hi.is_ascii_digit() && lo.is_ascii_digit(),
        "{what} at offset {idx} is not two digits: {:?}",
        String::from_utf8_lossy(&[hi, lo])
    );
    Ok(i32::from(hi - b'0') * 10 + i32::from(lo - b'0'))
}

/// Parse one rendered scene. Every layout violation names its offset: a
/// malformed corpus is a data bug to fix, not a scene to skip.
pub fn parse_scene(scene: &[u16]) -> anyhow::Result<ParsedScene> {
    anyhow::ensure!(
        scene.len() >= 10,
        "a scene is at least 10 tokens, got {}",
        scene.len()
    );
    anyhow::ensure!(
        scene_byte(scene, 0)? == b'P',
        "scene starts with {:?}, not 'P'",
        char::from(scene_byte(scene, 0)?)
    );
    let count = two_digits(scene, 1, "point count")? as usize;
    anyhow::ensure!(
        count >= 1 && count <= 26,
        "point count {count} is outside [1, 26]"
    );
    anyhow::ensure!(
        scene.len() == 10 + 5 * count,
        "a {count}-point scene is {} tokens, got {}",
        10 + 5 * count,
        scene.len()
    );
    let mut points = Vec::with_capacity(count);
    for i in 0..count {
        let base = 3 + 5 * i;
        let letter = scene_byte(scene, base)?;
        anyhow::ensure!(
            letter == b'A' + i as u8,
            "point {i} is named {:?}, expected {:?}",
            char::from(letter),
            char::from(b'A' + i as u8)
        );
        let x = two_digits(scene, base + 1, "x coordinate")?;
        let y = two_digits(scene, base + 3, "y coordinate")?;
        points.push((x, y));
    }
    let kind_at = 3 + 5 * count;
    let kind = match scene_byte(scene, kind_at)? {
        b'n' => "nearest",
        b'f' => "farthest",
        b'd' => "direction",
        b'i' => "inside",
        b'c' => "collinear",
        other => anyhow::bail!("unknown kind letter {:?} at offset {kind_at}", char::from(other)),
    };
    let mut refs = Vec::with_capacity(4);
    for slot in 0..4 {
        let b = scene_byte(scene, kind_at + 1 + slot)?;
        if b == b'_' {
            continue;
        }
        anyhow::ensure!(
            (b'A'..=b'Z').contains(&b),
            "reference slot {slot} holds {:?}, not a point name or '_'",
            char::from(b)
        );
        let idx = usize::from(b - b'A');
        anyhow::ensure!(idx < count, "reference {char} names a point outside the {count}-point scene", char = char::from(b));
        refs.push(idx);
    }
    anyhow::ensure!(
        scene_byte(scene, kind_at + 5)? == b'?',
        "the question mark is missing at offset {}",
        kind_at + 5
    );
    let answer = scene_byte(scene, kind_at + 6)?;
    Ok(ParsedScene {
        points,
        kind: kind.to_string(),
        refs,
        answer,
    })
}

/// The scene as the text the trunk's tokenizer sees: one char per byte.
pub fn scene_text(scene: &[u16]) -> anyhow::Result<String> {
    (0..scene.len())
        .map(|i| scene_byte(scene, i).map(char::from))
        .collect()
}

/// A scene tokenized for the trunk: the BPE encoding of the rendered scene
/// with the answer as the final token, plus the position the answer is
/// predicted from and the answer's token id.
///
/// The answer byte is encoded **on its own** and appended: tokenizing the
/// whole rendered scene at once would let the merges fuse the answer byte
/// with the `?` before it, and then the answer span would move with the
/// vocabulary instead of with the scene. Train and eval share this encoding,
/// so the model always sees the answer as one dedicated token.
#[derive(Debug, Clone)]
pub struct TokenizedScene {
    /// `encode(prefix) ++ [answer token]`, as `i64` ids (the Qwen
    /// vocabulary does not fit the crate's `u16` token type).
    pub tokens: Vec<i64>,
    /// Position of the last prefix token: the answer token at
    /// `tokens[query + 1]` is predicted from here.
    pub query: usize,
    /// The answer's token id.
    pub answer: i64,
    /// The answer byte, for accuracy and kernel checking.
    pub answer_byte: u8,
}

impl TokenizedScene {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}

/// Tokenize one scene (see the type docs for the answer-span choice).
pub fn tokenize_scene(tokenizer: &BpeTokenizer, scene: &[u16]) -> anyhow::Result<TokenizedScene> {
    let text = scene_text(scene)?;
    let answer_char = text
        .chars()
        .next_back()
        .ok_or_else(|| anyhow::anyhow!("an empty scene has no answer byte"))?;
    let prefix = &text[..text.len() - answer_char.len_utf8()];
    let prefix_ids = tokenizer
        .encode(prefix)
        .with_context(|| format!("encode scene prefix {prefix:?}"))?;
    anyhow::ensure!(
        !prefix_ids.is_empty(),
        "the scene prefix {prefix:?} tokenizes to nothing"
    );
    let answer_ids = tokenizer
        .encode(&answer_char.to_string())
        .with_context(|| format!("encode answer byte {answer_char:?}"))?;
    anyhow::ensure!(
        answer_ids.len() == 1,
        "the answer byte {answer_char:?} encodes to {} tokens, not one; the answer objective needs a single-token answer span",
        answer_ids.len()
    );
    let query = prefix_ids.len() - 1;
    let mut tokens: Vec<i64> = prefix_ids.iter().map(|t| i64::from(*t)).collect();
    tokens.push(i64::from(answer_ids[0]));
    Ok(TokenizedScene {
        tokens,
        query,
        answer: i64::from(answer_ids[0]),
        answer_byte: answer_char as u8,
    })
}

/// The token id of every answer byte the corpus declares, as `(byte, id)`
/// pairs -- the closed candidate set answer prediction is restricted to
/// (the same restriction `GeometricReasoner::answer_ce` applies, ported to
/// the trunk's vocabulary). Every answer byte must encode to exactly one
/// token, and the ids must be distinct; either failure names the byte.
pub fn answer_token_ids(
    tokenizer: &BpeTokenizer,
    meta: &GeomMeta,
) -> anyhow::Result<Vec<(u16, u32)>> {
    let mut out = Vec::with_capacity(meta.answer_tokens.len());
    for &byte in &meta.answer_tokens {
        let ch = char::from(u8::try_from(byte).with_context(|| {
            format!("answer token {byte} is not a byte")
        })?);
        let ids = tokenizer
            .encode(&ch.to_string())
            .with_context(|| format!("encode answer byte {ch:?}"))?;
        anyhow::ensure!(
            ids.len() == 1,
            "answer byte {ch:?} encodes to {} tokens, not one",
            ids.len()
        );
        anyhow::ensure!(
            !out.iter().any(|(_, id): &(u16, u32)| *id == ids[0]),
            "answer bytes share token id {}: the candidate set must be distinct",
            ids[0]
        );
        out.push((byte, ids[0]));
    }
    Ok(out)
}

// ------------------------------------------------------------ kernel gate ---

/// What the exact rational kernel says about one emitted answer. The model
/// proposes; the kernel decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVerdict {
    /// The proposed answer is what exact arithmetic establishes.
    Verified,
    /// Exact arithmetic establishes the opposite.
    Falsified,
    /// The kernel cannot decide this question: `direction` answers are
    /// absolute compass bearings, a predicate the exact kernel does not
    /// carry; ties at an extremum (nearest/farthest) and on-edge
    /// containment are degenerate by the generator's own rules; an answer
    /// outside the kind's alphabet proposes nothing checkable.
    Indeterminate,
}

impl KernelVerdict {
    pub fn name(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Falsified => "falsified",
            Self::Indeterminate => "indeterminate",
        }
    }
}

fn point_name(idx: usize) -> String {
    char::from(b'A' + idx as u8).to_string()
}

fn exact_coords(scene: &ParsedScene) -> Vec<(String, Q, Q)> {
    scene
        .points
        .iter()
        .enumerate()
        .map(|(i, (x, y))| {
            (
                point_name(i),
                Q::from_int(i64::from(*x)),
                Q::from_int(i64::from(*y)),
            )
        })
        .collect()
}

fn coord_of(coords: &[(String, Q, Q)], idx: usize) -> anyhow::Result<(Q, Q)> {
    coords
        .get(idx)
        .map(|(_, x, y)| (*x, *y))
        .ok_or_else(|| anyhow::anyhow!("point index {idx} is outside the scene"))
}

fn exact_sq_dist(coords: &[(String, Q, Q)], i: usize, j: usize) -> anyhow::Result<Q> {
    let (ax, ay) = coord_of(coords, i)?;
    let (bx, by) = coord_of(coords, j)?;
    let dx = ax.sub(&bx)?;
    let dy = ay.sub(&by)?;
    dx.mul(&dx)?.add(&dy.mul(&dy)?)
}

/// The unique extremum over squared distances from `reference`, in exact
/// arithmetic: `Ok(Some(idx))` the unique nearest (`farthest = false`) or
/// farthest other point, `Ok(None)` a tie at the extremum.
fn exact_extreme(
    coords: &[(String, Q, Q)],
    reference: usize,
    farthest: bool,
) -> anyhow::Result<Option<usize>> {
    let mut best: Option<(usize, Q)> = None;
    let mut tied = false;
    for idx in 0..coords.len() {
        if idx == reference {
            continue;
        }
        let d = exact_sq_dist(coords, reference, idx)?;
        match &best {
            None => best = Some((idx, d)),
            Some((_, best_d)) => {
                if (farthest && best_d.less(&d)) || (!farthest && d.less(best_d)) {
                    best = Some((idx, d));
                    tied = false;
                } else if !d.less(best_d) && !best_d.less(&d) {
                    tied = true;
                }
            }
        }
    }
    Ok(if tied { None } else { best.map(|(idx, _)| idx) })
}

/// Check one proposed answer byte against the scene, in the kernel's exact
/// rational arithmetic. See [`KernelVerdict`] for what is decidable.
pub fn kernel_check(scene: &ParsedScene, proposed: u8) -> anyhow::Result<KernelVerdict> {
    let coords = exact_coords(scene);
    match scene.kind.as_str() {
        "collinear" => {
            anyhow::ensure!(
                scene.refs.len() >= 3,
                "a collinear question needs 3 references, got {}",
                scene.refs.len()
            );
            let (a, b, c) = (
                point_name(scene.refs[0]),
                point_name(scene.refs[1]),
                point_name(scene.refs[2]),
            );
            let collinear = geomkernel::Constraint::Collinear {
                a: a.clone(),
                b: b.clone(),
                c: c.clone(),
            };
            let non_collinear = geomkernel::Constraint::NonCollinear { a, b, c };
            let claim = match proposed {
                b'y' => collinear.clone(),
                b'n' => non_collinear.clone(),
                _ => return Ok(KernelVerdict::Indeterminate),
            };
            let counter = match proposed {
                b'y' => non_collinear,
                _ => collinear,
            };
            if geomkernel::constraint_holds(&coords, &claim)? {
                Ok(KernelVerdict::Verified)
            } else if geomkernel::constraint_holds(&coords, &counter)? {
                Ok(KernelVerdict::Falsified)
            } else {
                Ok(KernelVerdict::Indeterminate)
            }
        }
        "inside" => {
            anyhow::ensure!(
                scene.refs.len() >= 4,
                "an inside question needs 4 references, got {}",
                scene.refs.len()
            );
            if proposed != b'y' && proposed != b'n' {
                return Ok(KernelVerdict::Indeterminate);
            }
            // Strict containment is three same-side tests; the kernel's
            // exact cross product is the side test.
            let a = coord_of(&coords, scene.refs[0])?;
            let b = coord_of(&coords, scene.refs[1])?;
            let c = coord_of(&coords, scene.refs[2])?;
            let d = coord_of(&coords, scene.refs[3])?;
            let s1 = geomkernel::cross2(a, b, d)?;
            let s2 = geomkernel::cross2(b, c, d)?;
            let s3 = geomkernel::cross2(c, a, d)?;
            if s1.is_zero() || s2.is_zero() || s3.is_zero() {
                // On an edge: the generator rejects these, and strict
                // containment is genuinely undecidable for a y/n label.
                return Ok(KernelVerdict::Indeterminate);
            }
            let neg = |s: &Q| s.less(&Q::ZERO);
            let inside = (neg(&s1) == neg(&s2)) && (neg(&s2) == neg(&s3));
            Ok(match (proposed, inside) {
                (b'y', true) | (b'n', false) => KernelVerdict::Verified,
                _ => KernelVerdict::Falsified,
            })
        }
        "nearest" | "farthest" => {
            anyhow::ensure!(
                !scene.refs.is_empty(),
                "a {} question needs a reference",
                scene.kind
            );
            if !(b'A'..b'A' + scene.points.len() as u8).contains(&proposed) {
                return Ok(KernelVerdict::Indeterminate);
            }
            let claimed = usize::from(proposed - b'A');
            if claimed == scene.refs[0] {
                // "The nearest point to A is A" proposes nothing.
                return Ok(KernelVerdict::Indeterminate);
            }
            match exact_extreme(&coords, scene.refs[0], scene.kind == "farthest")? {
                Some(best) if best == claimed => Ok(KernelVerdict::Verified),
                Some(_) => Ok(KernelVerdict::Falsified),
                None => Ok(KernelVerdict::Indeterminate),
            }
        }
        // Absolute compass bearings are not a predicate the exact kernel
        // carries; direction answers are reported, not kernel-judged.
        "direction" => Ok(KernelVerdict::Indeterminate),
        other => anyhow::bail!("unknown question kind '{other}'"),
    }
}

// ------------------------------------------------------------- FusedQwen ---

/// One fused forward pass, end to end: the logits plus everything the proof
/// harness reads (traces, gates, routing, final state).
#[derive(Debug, Clone)]
pub struct FusedQwenForward<B: Backend> {
    /// `lm_head(trunk hidden + geometric residual)`, `[b, n, vocab]`.
    pub logits: Tensor<B, 3>,
    /// The fused hidden state before `lm_head`.
    pub fused: FusedForward<B>,
}

/// The Qwen trunk with the geometric stream fused onto its hidden states:
/// `forward_hidden` -> [`GeomFusion`] -> untied `lm_head`.
///
/// A fresh model is the trunk, exactly: the fusion's output projection is
/// zero-initialized, so the residual it adds is zero bit for bit (the
/// `geomfusion/fused_qwen_is_the_trunk_at_zero_fusion` certificate).
/// Training turns the stream on.
#[derive(Module, Debug)]
pub struct FusedQwen<B: Backend> {
    pub trunk: QwenTrunk<B>,
    pub fusion: GeomFusion<B>,
}

impl<B: Backend<FloatElem = f32>> FusedQwen<B> {
    /// Compose a trunk and a fresh geometric stream. Fallible: the stream's
    /// `hidden_size` must be the trunk's, or the residual would silently add
    /// mismatched widths.
    pub fn new(
        trunk: QwenTrunk<B>,
        config: &GeomFusionConfig,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.hidden_size == trunk.config().dims.hidden_size,
            "the fusion layer's hidden_size {} does not match the trunk's {}",
            config.hidden_size,
            trunk.config().dims.hidden_size
        );
        let fusion = GeomFusion::new(config, device)?;
        Ok(Self { trunk, fusion })
    }

    /// `tokens [b, n]` -> logits `[b, n, vocab]`. `depth` overrides the
    /// refinement depth (the test-time-compute knob).
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        depth: Option<usize>,
    ) -> anyhow::Result<Tensor<B, 3>> {
        let fused = self.fusion.forward(self.trunk.forward_hidden(tokens), depth)?;
        Ok(self.trunk.lm_head.forward(fused.hidden))
    }

    /// The fused forward with the full trace: logits, refinement traces,
    /// routing gates and diagnostics.
    pub fn forward_traced(
        &self,
        tokens: Tensor<B, 2, Int>,
        depth: Option<usize>,
    ) -> anyhow::Result<FusedQwenForward<B>> {
        let fused = self.fusion.forward(self.trunk.forward_hidden(tokens), depth)?;
        let logits = self.trunk.lm_head.forward(fused.hidden.clone());
        Ok(FusedQwenForward { logits, fused })
    }

    /// Logits at one query position only: `[b, vocab]`. The answer
    /// objective and prediction both read a single position, and the
    /// untied head is the widest matrix in the model -- narrowing before it
    /// is the difference between `[b, 1, vocab]` and `[b, n, vocab]`.
    pub fn query_logits(
        &self,
        tokens: Tensor<B, 2, Int>,
        query: usize,
        depth: Option<usize>,
    ) -> anyhow::Result<(Tensor<B, 2>, FusedForward<B>)> {
        let fused = self.fusion.forward(self.trunk.forward_hidden(tokens), depth)?;
        let [b, n, _] = fused.hidden.dims();
        anyhow::ensure!(
            query < n,
            "query position {query} is outside the {n}-token sequence"
        );
        let logits = self
            .trunk
            .lm_head
            .forward(fused.hidden.clone().narrow(1, query, 1))
            .reshape([b, self.trunk.config().dims.vocab_size]);
        Ok((logits, fused))
    }

    /// Predict the answer byte for one tokenized scene: the argmax at the
    /// query position restricted to the declared candidate ids (the closed
    /// answer set, so an impossible byte can never win).
    pub fn predict_answer(
        &self,
        scene: &TokenizedScene,
        candidates: &[(u16, u32)],
        depth: Option<usize>,
        device: &B::Device,
    ) -> anyhow::Result<(u8, FusedForward<B>)> {
        let tokens = Tensor::<B, 1, Int>::from_ints(scene.tokens.as_slice(), device)
            .reshape([1, scene.len()]);
        let (logits, fused) = self.query_logits(tokens, scene.query, depth)?;
        let cand_ids: Vec<i64> = candidates.iter().map(|(_, id)| i64::from(*id)).collect();
        let cand = Tensor::<B, 1, Int>::from_ints(cand_ids.as_slice(), device);
        let restricted = logits.select(1, cand).reshape([candidates.len()]);
        let best: i64 = restricted.argmax(0).into_scalar();
        let idx = usize::try_from(best)
            .map_err(|_| anyhow::anyhow!("argmax index {best} does not fit usize"))?;
        let byte = u8::try_from(candidates[idx].0)
            .map_err(|_| anyhow::anyhow!("answer byte {} does not fit u8", candidates[idx].0))?;
        Ok((byte, fused))
    }
}

// ------------------------------------------- diffusion on the fused stream ---

/// One EDM denoising step of a single refinement block, on the fused
/// stream. Mirrors `GeometricReasoner::block_step`: the clean attractor
/// state corrupted at the block window's noise scale, EDM-preconditioned
/// back toward it.
#[derive(Debug, Clone)]
pub struct FusionBlockStep<B: Backend> {
    /// `hidden + out_proj(mixture(h_out))`: the fused hidden state after the
    /// denoising block, which the answer CE reads.
    pub hidden: Tensor<B, 3>,
    /// `mse(x0, z_star)`: the (unweighted) denoising error.
    pub mse: Tensor<B, 1>,
    pub mse_value: f32,
    pub sigma: f64,
    pub block: usize,
}

impl<B: Backend<FloatElem = f32>> GeomFusion<B> {
    /// The clean attractor state of the geometric stream over `hidden`,
    /// detached: the target every diffusion step denoises toward (the
    /// standalone reasoner's `clean_state`, on the fused path).
    pub fn clean_final_state(
        &self,
        hidden: &Tensor<B, 3>,
        depth: Option<usize>,
    ) -> anyhow::Result<Tensor<B, 3>> {
        Ok(self.geometric_path(hidden, depth)?.final_state.detach())
    }

    /// Train block `block_idx` standalone on the clean fused state corrupted
    /// at `sigma`, EDM-preconditioned; the returned fused hidden feeds the
    /// answer CE, and `mse` (weighted by the caller with
    /// `crate::sigma::edm_loss_weight`) trains the block as a denoiser.
    /// This is what the geometric corpus format supports of the diffusion
    /// objective: the scenes are identical, the noise is model-side.
    pub fn block_diffusion_step(
        &self,
        hidden: Tensor<B, 3>,
        block_idx: usize,
        sigma: f64,
        sigma_data: f64,
        z_star: &Tensor<B, 3>,
    ) -> anyhow::Result<FusionBlockStep<B>> {
        anyhow::ensure!(
            block_idx < self.blocks.len(),
            "block {block_idx} out of range ({})",
            self.blocks.len()
        );
        let [batch, n, width] = hidden.dims();
        anyhow::ensure!(
            width == self.hidden_size,
            "hidden width {width} does not match the fusion layer's hidden_size {}",
            self.hidden_size
        );
        anyhow::ensure!(
            n >= self.slots,
            "the geometric stream pools {n} tokens into {} slots; it needs at least as many tokens as slots",
            self.slots
        );
        let device = hidden.device();
        let h = self.in_proj.forward(hidden.clone());
        let noise = Tensor::<B, 3>::random(
            [batch, self.slots, z_star.dims()[2]],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z_in = z_star.clone().add(noise.mul_scalar(sigma as f32));
        let (h_out, z_out, _) = self.blocks[block_idx].refine(h, z_in.clone(), sigma, None);
        let precond = crate::sigma::EdmPreconditioning::new(sigma, sigma_data);
        let x0 = z_in
            .mul_scalar(precond.c_skip as f32)
            .add(z_out.mul_scalar(precond.c_out as f32));
        let diff = x0.sub(z_star.clone());
        let mse = diff.clone().mul(diff).mean();
        let mse_value = mse.clone().detach().into_scalar();
        // The readout over the denoised stream, so the answer CE trains the
        // same pathway the clean forward uses.
        let path = self.readout(&h_out, batch, n)?;
        let fused = hidden.add(self.out_proj.forward(path.mixture));
        Ok(FusionBlockStep {
            hidden: fused,
            mse,
            mse_value,
            sigma,
            block: block_idx,
        })
    }
}

// --------------------------------------------------------- fused losses ---

/// The objective shape for one backward: the clean answer objective, or the
/// EDM denoising variant at one block and noise scale.
#[derive(Debug, Clone, Copy)]
pub enum FusedLossSpec {
    /// Next-token cross-entropy on the answer span only.
    Answer,
    /// The EDM denoising term of block `block` at `sigma`, plus the
    /// sigma-weighted answer CE through the denoised pathway.
    Diffusion { block: usize, sigma: f64, sigma_data: f64 },
}

/// What a fused loss computation reports.
#[derive(Debug, Clone)]
pub struct FusedLoss<B: Backend> {
    /// The scalar loss (answer CE, plus the weighted denoising term and the
    /// router balance charges when configured).
    pub loss: Tensor<B, 1>,
    /// Unrestricted argmax accuracy at the answer position (the reported
    /// metric; the candidate-restricted accuracy is [`prove`]'s).
    pub accuracy: f32,
    /// Mean final refinement energy across rows and blocks.
    pub energy: f32,
    /// The denoising term's value (0 in the answer objective).
    pub mse: f32,
}

/// The geometric stream's routed readout, split out of
/// [`GeomFusion::geometric_path`] so the diffusion step can read out a
/// denoised stream through the same pathway the clean forward uses.
impl<B: Backend<FloatElem = f32>> GeomFusion<B> {
    fn readout(
        &self,
        h: &Tensor<B, 3>,
        batch: usize,
        n: usize,
    ) -> anyhow::Result<GeometricPath<B>> {
        let flat = h.clone().reshape([batch * n, self.geom_hidden]);
        let input = self.router.router_input_2d(&flat, &flat);
        let gates = self.router.route(input);

        // Dense evaluation over every head, weighted by the composed gates --
        // the same accumulation as `GeomRouter::forward`, so the mixture is a
        // convex combination of head outputs and the partition-of-unity
        // invariant is inherited from `mosme`, not re-proven here.
        let composed = gates.composed();
        let heads: Vec<_> = composed
            .iter()
            .zip(self.experts.iter())
            .flat_map(|(box_gates, experts)| {
                experts
                    .iter()
                    .enumerate()
                    .map(move |(j, head)| (box_gates, j, head))
            })
            .collect();
        anyhow::ensure!(
            !heads.is_empty(),
            "the {}x{} mixture has no heads to evaluate",
            REASONING_KINDS.len(),
            self.experts.first().map_or(0, Vec::len)
        );
        let (first_gates, first_j, first_head) = heads[0];
        let mut mixture = first_head
            .forward(flat.clone())
            .mul(first_gates.clone().narrow(1, first_j, 1));
        for (box_gates, j, head) in &heads[1..] {
            mixture = mixture.add(
                head.forward(flat.clone())
                    .mul((*box_gates).clone().narrow(1, *j, 1)),
            );
        }
        let composed_flat = gates.composed_flat();

        let balance = gates.balance_loss(self.balance);
        let n_boxes = REASONING_KINDS.len();
        let ln_boxes = (n_boxes as f32).ln().max(1e-6);
        let traffic = gates.box_traffic().reshape([n_boxes]);
        let pl = traffic.clone().clamp_min(1e-30).log();
        let load_entropy = traffic.clone().mul(pl).sum().neg().into_scalar() / ln_boxes;
        let token_entropy = crate::moe::entropy_of_rows(&gates.box_gates).into_scalar() / ln_boxes;
        let box_load: Vec<f32> = traffic.into_data().convert::<f32>().iter::<f32>().collect();

        Ok(GeometricPath {
            mixture: mixture.reshape([batch, n, self.geom_hidden]),
            final_state: h.clone(),
            traces: Vec::new(),
            gates: composed_flat,
            routing: RoutingInfo {
                balance,
                z_level: self.balance.z_level,
                load_entropy,
                token_entropy,
                box_load,
            },
        })
    }
}

/// The answer-span loss over one batch, from the trunk's (grad-carrying)
/// hidden state. Each row's fusion runs on its TRUE length: rows are padded
/// for the batched trunk pass, but the slot pooling must not see padding.
///
/// `hidden` is `[b, n_max, hidden_size]` post-`final_norm`; `scenes` carries
/// the per-row true length (`scene.len()`), the query position and the
/// answer token id.
pub fn fused_answer_loss<B: AutodiffBackend<FloatElem = f32>>(
    model: &FusedQwen<B>,
    hidden: Tensor<B, 3>,
    scenes: &[TokenizedScene],
    spec: FusedLossSpec,
    depth: Option<usize>,
    balance_weight: f64,
) -> anyhow::Result<FusedLoss<B>> {
    let [b, _, _] = hidden.dims();
    anyhow::ensure!(
        b == scenes.len(),
        "hidden batch {b} does not match {} scenes",
        scenes.len()
    );
    let vocab = model.trunk.config().dims.vocab_size;
    let mut query_rows = Vec::with_capacity(b);
    let mut energy_sum = 0.0f32;
    let mut mse_terms: Vec<Tensor<B, 1>> = Vec::new();
    let mut mse_value = 0.0f32;
    let mut balance: Option<Tensor<B, 1>> = None;
    let mut z_loss: Option<Tensor<B, 1>> = None;
    let mut z_level = 0.0f32;
    for (row, scene) in scenes.iter().enumerate() {
        anyhow::ensure!(
            scene.len() <= hidden.dims()[1],
            "row {row} has {} tokens, more than the {} the trunk saw",
            scene.len(),
            hidden.dims()[1]
        );
        anyhow::ensure!(
            scene.len() >= model.fusion.slots() + 1,
            "row {row} has {} tokens; the stream pools into {} slots and the answer needs a prefix position",
            scene.len(),
            model.fusion.slots()
        );
        let row_hidden = hidden
            .clone()
            .narrow(0, row, 1)
            .narrow(1, 0, scene.len());
        let fused_hidden = match spec {
            FusedLossSpec::Answer => {
                let fused = model.fusion.forward(row_hidden, depth)?;
                energy_sum += fused
                    .traces
                    .last()
                    .and_then(|t| t.energy.last())
                    .copied()
                    .unwrap_or(0.0);
                balance = Some(match balance {
                    None => fused.routing.balance.total.clone(),
                    Some(total) => total + fused.routing.balance.total.clone(),
                });
                z_loss = Some(match z_loss {
                    None => fused.routing.balance.z_loss.clone(),
                    Some(total) => total + fused.routing.balance.z_loss.clone(),
                });
                z_level = fused.routing.z_level;
                fused.hidden
            }
            FusedLossSpec::Diffusion {
                block,
                sigma,
                sigma_data,
            } => {
                let z_star = model.fusion.clean_final_state(&row_hidden.detach(), None)?;
                let step = model.fusion.block_diffusion_step(
                    row_hidden, block, sigma, sigma_data, &z_star,
                )?;
                mse_terms.push(step.mse);
                mse_value = step.mse_value;
                step.hidden
            }
        };
        query_rows.push(fused_hidden.narrow(1, scene.query, 1));
    }
    let query_hidden = Tensor::cat(query_rows, 0);
    let logits = model
        .trunk
        .lm_head
        .forward(query_hidden)
        .reshape([b, vocab]);
    let answer_ids: Vec<i64> = scenes.iter().map(|s| s.answer).collect();
    let targets =
        Tensor::<B, 1, Int>::from_ints(answer_ids.as_slice(), &hidden.device()).reshape([b, 1]);
    let log_probs = log_softmax(logits.clone(), 1);
    let ce = -log_probs.gather(1, targets).mean();

    let predicted: Vec<i64> = logits
        .argmax(1)
        .into_data()
        .convert::<i64>()
        .iter::<i64>()
        .collect();
    let correct = predicted
        .iter()
        .zip(answer_ids.iter())
        .filter(|(p, a)| p == a)
        .count();
    let accuracy = correct as f32 / b.max(1) as f32;

    let mut loss = match spec {
        FusedLossSpec::Answer => ce,
        FusedLossSpec::Diffusion {
            sigma, sigma_data, ..
        } => {
            let mse = mse_terms
                .into_iter()
                .reduce(|a, b| a + b)
                .ok_or_else(|| anyhow::anyhow!("the diffusion step produced no mse term"))?
                / b as f32;
            let w_ce = ((crate::sigma::SIGMA_MAX - sigma)
                / (crate::sigma::SIGMA_MAX - crate::sigma::SIGMA_MIN)) as f32;
            let w_mse = crate::sigma::edm_loss_weight(sigma, sigma_data) as f32;
            ce.mul_scalar(w_ce).add(mse.mul_scalar(w_mse))
        }
    };
    if balance_weight > 0.0 {
        if let Some(bal) = balance {
            loss = loss + bal.div_scalar(b as f32).mul_scalar(balance_weight as f32);
        }
        if let Some(zl) = z_loss {
            loss = loss + zl.div_scalar(b as f32).mul_scalar(z_level);
        }
    }
    Ok(FusedLoss {
        loss,
        accuracy,
        energy: energy_sum / b.max(1) as f32,
        mse: mse_value,
    })
}

/// The gradients of one fused micro-batch: the trunk's adapter gradients
/// and the geometric stream's, separately (they step through separate
/// optimizers).
pub struct FusedGrads {
    pub loss: f32,
    pub accuracy: f32,
    pub energy: f32,
    pub mse: f32,
    pub trunk: GradientsParams,
    pub fusion: GradientsParams,
}

/// Merge `from` into `into`, parameter by parameter (the same visitor
/// pattern as `qwentrain`'s gradient merge: only the module traversal knows
/// each parameter's rank).
fn merge_gradients_into<B: AutodiffBackend<FloatElem = f32>, M: Module<B>>(
    into: GradientsParams,
    from: GradientsParams,
    module: &M,
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
    module.visit(&mut SumVisitor {
        into: &mut into,
        from,
        _backend: std::marker::PhantomData,
    });
    into
}

/// The fused model's segment-checkpointed backward: one no-grad trunk pass
/// stores the per-layer boundaries, the loss segment (final norm + fusion +
/// lm_head + answer loss) runs with grad, then each trunk layer re-runs
/// with grad on its boundary, chaining the exact vector-Jacobian product
/// upstream -- `qwentrain::segmented_next_token_backward`'s scheme with the
/// geometric stream inside the loss segment. At most one layer's graph is
/// alive at a time, which is what keeps a 27B trunk trainable next to its
/// packed weights. Numerically identical to a one-shot backward, certified
/// as `geomfusion/fused_segmented_backward_matches_one_shot`.
pub fn segmented_fused_backward<B: AutodiffBackend<FloatElem = f32>>(
    model: &FusedQwen<B>,
    tokens: Tensor<B, 2, Int>,
    scenes: &[TokenizedScene],
    spec: FusedLossSpec,
    depth: Option<usize>,
    balance_weight: f64,
    loss_scale: f64,
) -> anyhow::Result<FusedGrads> {
    // ---- no-grad trunk forward, storing each layer's input boundary ----
    let mut boundaries: Vec<Tensor<B, 3>> = Vec::with_capacity(model.trunk.layers.len());
    let mut x = model.trunk.embed_tokens.forward(tokens).detach();
    for layer in &model.trunk.layers {
        boundaries.push(x.clone());
        x = layer.forward(x).detach();
    }

    // ---- loss segment: final norm + geometric stream + lm_head + loss ----
    let final_input = x.require_grad();
    let hidden = model.trunk.final_norm.forward(final_input.clone());
    let fused = fused_answer_loss(model, hidden, scenes, spec, depth, balance_weight)?;
    let loss_value = fused.loss.clone().into_scalar();
    let scaled = if (loss_scale - 1.0).abs() > f64::EPSILON {
        fused.loss.mul_scalar(loss_scale)
    } else {
        fused.loss
    };
    let mut grads = scaled.backward();
    let mut upstream = final_input
        .grad_remove(&mut grads)
        .ok_or_else(|| anyhow::anyhow!("final hidden boundary produced no gradient"))?;
    // The loss segment's only trainable trunk parameters would be unfrozen
    // ones -- there are none (freeze_non_adapter_params), so the trunk
    // gradients here are empty and the fusion's are complete.
    let fusion_grads = GradientsParams::from_grads(grads, &model.fusion);

    // ---- per-layer segments, last to first ----
    let mut trunk_grads = GradientsParams::new();
    for (idx, layer) in model.trunk.layers.iter().enumerate().rev() {
        let input = boundaries[idx].clone().require_grad();
        let out = layer.forward(input.clone());
        let pseudo = (out * Tensor::<B, 3>::from_inner(upstream)).sum();
        let mut grads = pseudo.backward();
        upstream = input
            .grad_remove(&mut grads)
            .with_context(|| format!("layer {idx} boundary produced no gradient"))?;
        let segment = GradientsParams::from_grads(grads, &model.trunk);
        trunk_grads = merge_gradients_into(trunk_grads, segment, &model.trunk);
    }
    Ok(FusedGrads {
        loss: loss_value,
        accuracy: fused.accuracy,
        energy: fused.energy,
        mse: fused.mse,
        trunk: trunk_grads,
        fusion: fusion_grads,
    })
}

// -------------------------------------------------------------- training ---

/// AdamW over the geometric stream (the trunk's adapters step through
/// [`crate::qwentrain::QwenOptim`]).
pub type FusionOptim<B> = burn::optim::adaptor::OptimizerAdaptor<AdamW, GeomFusion<B>, B>;
/// The fusion optimizer's record.
pub type FusionOptimRecord<B> = <FusionOptim<B> as Optimizer<GeomFusion<B>, B>>::Record;

fn default_fused_lr() -> f64 {
    1e-4
}

fn default_fused_fusion_lr() -> f64 {
    1e-3
}

fn default_fused_accumulate() -> usize {
    1
}

fn default_holdout() -> usize {
    512
}

fn default_log_every() -> usize {
    10
}

fn default_sigma_data() -> f64 {
    0.5
}

fn default_gamma() -> f64 {
    0.05
}

fn default_balance_weight() -> f64 {
    0.01
}

/// Training-loop configuration for the fused model. `#[serde(default)]` at
/// the container level, so a sidecar written by an older build still parses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FusedTrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    /// Learning rate of the trunk's LoRA adapters.
    pub lr: f64,
    /// Learning rate of the geometric stream's own parameters. Higher than
    /// the adapters': the stream starts at the identity and has to move
    /// before it can help.
    pub fusion_lr: f64,
    /// Gradient-clip bound on each accumulated gradient set (adapters and
    /// stream separately). `None` disables.
    pub clip_norm: Option<f32>,
    /// Micro-batches averaged per optimizer step.
    pub accumulate: usize,
    /// Bias-corrected EMA decay over the adapter bank and the stream;
    /// `None` disables both.
    pub ema_decay: Option<f64>,
    /// Adapter hyperparameters.
    pub lora: QwenLoraConfig,
    pub seed: u64,
    /// `answer` or `diffusion` (see [`FusedLossSpec`]).
    pub objective: GeomObjective,
    /// Sigma-window extension factor of the diffusion sampler.
    pub gamma: f64,
    /// EDM preconditioning scale (`--sigma-data` on the standalone loop).
    pub sigma_data: f64,
    /// Charge on the readout router's balance loss; `0.0` is off.
    pub moe_balance_weight: f64,
    /// Refinement depth during training (`None` = the configured depth).
    pub depth: Option<usize>,
    /// Scenes held out at the corpus tail: training samples only from the
    /// head, so [`prove`] on the tail never sees a training scene.
    pub holdout: usize,
    /// Print progress every this many steps; `0` silences.
    pub log_every: usize,
}

impl Default for FusedTrainConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch_size: 4,
            lr: default_fused_lr(),
            fusion_lr: default_fused_fusion_lr(),
            clip_norm: Some(1.0),
            accumulate: default_fused_accumulate(),
            ema_decay: Some(0.99),
            lora: QwenLoraConfig::default(),
            seed: 0,
            objective: GeomObjective::Answer,
            gamma: default_gamma(),
            sigma_data: default_sigma_data(),
            moe_balance_weight: default_balance_weight(),
            depth: None,
            holdout: default_holdout(),
            log_every: default_log_every(),
        }
    }
}

impl FusedTrainConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.steps >= 1, "steps must be positive");
        anyhow::ensure!(self.batch_size >= 1, "batch_size must be positive");
        anyhow::ensure!(self.accumulate >= 1, "accumulate must be positive");
        anyhow::ensure!(
            self.lr.is_finite() && self.lr > 0.0,
            "lr must be positive and finite, not {}",
            self.lr
        );
        anyhow::ensure!(
            self.fusion_lr.is_finite() && self.fusion_lr > 0.0,
            "fusion_lr must be positive and finite, not {}",
            self.fusion_lr
        );
        if let Some(decay) = self.ema_decay {
            anyhow::ensure!(
                decay > 0.0 && decay < 1.0,
                "ema_decay must be in (0, 1), got {decay}"
            );
        }
        anyhow::ensure!(
            self.sigma_data.is_finite() && self.sigma_data > 0.0,
            "sigma_data must be positive and finite, not {}",
            self.sigma_data
        );
        Ok(())
    }
}

/// What one fused `train_step` did.
#[derive(Debug, Clone)]
pub struct FusedStepReport {
    /// Micro-batch loss (unscaled).
    pub loss: f32,
    /// Unrestricted argmax accuracy at the answer positions.
    pub accuracy: f32,
    /// Mean final refinement energy of the batch.
    pub energy: f32,
    /// Denoising term (diffusion objective; 0 otherwise).
    pub mse: f32,
    pub grad_norm_trunk: f32,
    pub grad_norm_fusion: f32,
    pub clipped: bool,
    /// Whether an optimizer step happened (the accumulation cycle
    /// completed; a discarded non-finite micro-batch is also not a step).
    pub stepped: bool,
}

/// Training state persisted beyond the weights (`TrainState` extras).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusedTrainExtras {
    #[serde(default)]
    pub micro_counter: usize,
    #[serde(default)]
    pub tokens_seen: usize,
    #[serde(default)]
    pub loss_sum: f64,
    #[serde(default)]
    pub ema_updates_trunk: Option<usize>,
    #[serde(default)]
    pub ema_updates_fusion: Option<usize>,
    #[serde(default)]
    pub lora: QwenLoraConfig,
    #[serde(default)]
    pub fusion_optimizer: Option<StateFile>,
    #[serde(default)]
    pub fusion_ema: Option<StateFile>,
}

/// The adapters-only trainer for the fused model: LoRA on the trunk through
/// the [`crate::qwentrain`] machinery (attach, freeze, segment-checkpointed
/// backward, accumulation, EMA, content-addressed resume) plus a second
/// AdamW over the geometric stream, whose parameters are tiny next to the
/// trunk and train in full.
///
/// No `Debug`, for the same reason as `QwenTrainer`: the optimizer state
/// does not carry one, and the per-step [`FusedStepReport`] is the printable
/// summary.
pub struct FusedQwenTrainer<B: AutodiffBackend> {
    pub model: FusedQwen<B>,
    pub trunk_optim: QwenOptim<B>,
    pub fusion_optim: FusionOptim<B>,
    pub bank: QwenAdapterBank<B>,
    pub ema_trunk: Option<Ema<QwenAdapterBank<B>>>,
    pub ema_fusion: Option<Ema<GeomFusion<B>>>,
    pub config: FusedTrainConfig,
    pub device: B::Device,
    pub step: usize,
    pub micro_counter: usize,
    pub tokens_seen: usize,
    pub loss_sum: f64,
    acc_trunk: GradientAccumulator,
    acc_fusion: GradientAccumulator,
}

impl<B: AutodiffBackend<FloatElem = f32>> FusedQwenTrainer<B> {
    /// Attach adapters to the trunk, freeze everything else, and build the
    /// optimizers, the adapter bank and the EMA shadows. The geometric
    /// stream trains in full (it is the new machinery; the trunk is the
    /// capability substrate that stays resident).
    pub fn new(
        mut model: FusedQwen<B>,
        config: FusedTrainConfig,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        qwentrain::attach_lora(&mut model.trunk, &config.lora, device)?;
        qwentrain::freeze_non_adapter_params(&mut model.trunk);
        let bank = QwenAdapterBank::from_trunk(&model.trunk)?;
        let trunk_optim: QwenOptim<B> = AdamWConfig::new().init();
        let fusion_optim: FusionOptim<B> = AdamWConfig::new().init();
        let ema_trunk = config.ema_decay.map(|decay| Ema::new(&bank, decay));
        let ema_fusion = config
            .ema_decay
            .map(|decay| Ema::new(&model.fusion, decay));
        Ok(Self {
            acc_trunk: GradientAccumulator::new(config.accumulate),
            acc_fusion: GradientAccumulator::new(config.accumulate),
            model,
            trunk_optim,
            fusion_optim,
            bank,
            ema_trunk,
            ema_fusion,
            config,
            device: device.clone(),
            step: 0,
            micro_counter: 0,
            tokens_seen: 0,
            loss_sum: 0.0,
        })
    }

    /// The loss spec for this micro-batch: the configured objective, with
    /// the diffusion variant's block and sigma drawn from a host RNG seeded
    /// per micro-step (resume-deterministic, like the device reseeding).
    fn loss_spec(&self) -> FusedLossSpec {
        match self.config.objective {
            GeomObjective::Answer => FusedLossSpec::Answer,
            GeomObjective::Diffusion => {
                let mut rng = ChaCha12Rng::seed_from_u64(
                    crate::train::step_seed(self.config.seed, self.micro_counter) ^ 0xD1FF_0510,
                );
                let blocks = self.model.fusion.num_blocks();
                let sampler = crate::sigma::DblockSigmaSampler::new(blocks, self.config.gamma);
                let block = rng.random_range(0..blocks);
                let sigma = sampler.sample(&mut rng, block, 1)[0];
                FusedLossSpec::Diffusion {
                    block,
                    sigma,
                    sigma_data: self.config.sigma_data,
                }
            }
        }
    }

    /// One micro-batch: the segment-checkpointed fused backward, folded
    /// into the accumulation cycles of both parameter sets; on cycle
    /// completion, clip each summed gradient, step both optimizers, re-sync
    /// the bank and update the EMA shadows.
    pub fn train_step(&mut self, scenes: &[TokenizedScene]) -> anyhow::Result<FusedStepReport> {
        anyhow::ensure!(!scenes.is_empty(), "an empty batch has no loss");
        let max_len = scenes.iter().map(TokenizedScene::len).max().unwrap_or(0);
        let mut flat = Vec::with_capacity(scenes.len() * max_len);
        for scene in scenes {
            let mut row = scene.tokens.clone();
            // Tail padding is never read: attention is causal, and the loss
            // narrows each row's fusion to its true length. Zero is a
            // value, not a mask, and it must not matter.
            row.resize(max_len, 0);
            flat.extend(row);
        }
        let tokens = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), &self.device)
            .reshape([scenes.len(), max_len]);

        <B as Backend>::seed(
            &self.device,
            crate::train::step_seed(self.config.seed, self.micro_counter),
        );
        let spec = self.loss_spec();
        let scale = self.acc_trunk.loss_scale();
        let grads = segmented_fused_backward(
            &self.model,
            tokens,
            scenes,
            spec,
            self.config.depth,
            self.config.moe_balance_weight,
            scale,
        )?;
        self.micro_counter += 1;
        self.tokens_seen += scenes.iter().map(TokenizedScene::len).sum::<usize>();
        self.loss_sum += f64::from(grads.loss);
        let base = FusedStepReport {
            loss: grads.loss,
            accuracy: grads.accuracy,
            energy: grads.energy,
            mse: grads.mse,
            grad_norm_trunk: f32::NAN,
            grad_norm_fusion: f32::NAN,
            clipped: false,
            stepped: false,
        };
        if !grads.loss.is_finite() {
            // The crate-wide policy: a pathological step is discarded, not
            // clipped into something that looks fine.
            drop(self.acc_trunk.skip());
            drop(self.acc_fusion.skip());
            return Ok(base);
        }
        let cycle_trunk = self.acc_trunk.fold(grads.trunk, &self.model.trunk);
        let cycle_fusion = self.acc_fusion.fold(grads.fusion, &self.model.fusion);
        let (Some(mut g_trunk), Some(mut g_fusion)) = (
            cycle_trunk.into_gradients(),
            cycle_fusion.into_gradients(),
        ) else {
            return Ok(base);
        };

        let norm_trunk = crate::quality::global_grad_norm(&self.model.trunk, &g_trunk);
        let norm_fusion = crate::quality::global_grad_norm(&self.model.fusion, &g_fusion);
        let mut clipped = false;
        if let Some(max_norm) = self.config.clip_norm {
            clipped |= clip_gradients::<B, _>(&mut g_trunk, &self.model.trunk, norm_trunk, max_norm)
                < 1.0;
            clipped |=
                clip_gradients::<B, _>(&mut g_fusion, &self.model.fusion, norm_fusion, max_norm)
                    < 1.0;
        }
        self.model.trunk = self
            .trunk_optim
            .step(self.config.lr, self.model.trunk.clone(), g_trunk);
        self.model.fusion = self.fusion_optim.step(
            self.config.fusion_lr,
            self.model.fusion.clone(),
            g_fusion,
        );
        self.bank = QwenAdapterBank::from_trunk(&self.model.trunk)?;
        if let Some(ema) = self.ema_trunk.as_mut() {
            ema.update::<B>(&self.bank);
        }
        if let Some(ema) = self.ema_fusion.as_mut() {
            ema.update::<B>(&self.model.fusion);
        }
        // A NaN that reaches the weights poisons every later step; that is
        // fatal, not skippable (train.rs's parameters_finite discipline).
        let bad_trunk = crate::quality::non_finite_parameters(&self.bank);
        let bad_fusion = crate::quality::non_finite_parameters(&self.model.fusion);
        anyhow::ensure!(
            bad_trunk == 0 && bad_fusion == 0,
            "step {}: {bad_trunk} adapter + {bad_fusion} stream parameter tensor(s) went non-finite after the optimizer step \
             (grad norms {norm_trunk:.3e} / {norm_fusion:.3e}); the run cannot continue",
            self.step
        );
        self.step += 1;
        Ok(FusedStepReport {
            grad_norm_trunk: norm_trunk,
            grad_norm_fusion: norm_fusion,
            clipped,
            stepped: true,
            ..base
        })
    }

    /// Content-addressed checkpoint + `TrainState` sidecar: fused-model
    /// record (in NF4 mode only the f32 params -- adapters, smalls, and the
    /// whole stream -- are recorded, the packed weights re-quantize from the
    /// shards on resume), both optimizer records, both EMA shadows, the host
    /// RNG and the counters.
    pub fn save_checkpoint(&self, dir: &Path, rng: &ChaCha12Rng) -> anyhow::Result<PathBuf> {
        let model_path = checkpoint::save_content_addressed(self.model.clone(), dir, "geomfusion")?;
        let state_dir = TrainState::dir_for(&model_path);
        if state_dir.exists() {
            std::fs::remove_dir_all(&state_dir)
                .with_context(|| format!("clear {}", state_dir.display()))?;
        }
        let optimizer = Some(checkpoint::save_record::<B, _>(
            self.trunk_optim.to_record(),
            &state_dir,
            "optimizer",
        )?);
        let ema_file = self
            .ema_trunk
            .as_ref()
            .map(|e| {
                checkpoint::save_record::<B, _>(e.shadow().clone().into_record(), &state_dir, "ema")
            })
            .transpose()?;
        let fusion_optimizer = Some(checkpoint::save_record::<B, _>(
            self.fusion_optim.to_record(),
            &state_dir,
            "fusion-optimizer",
        )?);
        let fusion_ema = self
            .ema_fusion
            .as_ref()
            .map(|e| {
                checkpoint::save_record::<B, _>(
                    e.shadow().clone().into_record(),
                    &state_dir,
                    "fusion-ema",
                )
            })
            .transpose()?;
        let extras = FusedTrainExtras {
            micro_counter: self.micro_counter,
            tokens_seen: self.tokens_seen,
            loss_sum: self.loss_sum,
            ema_updates_trunk: self.ema_trunk.as_ref().map(Ema::updates),
            ema_updates_fusion: self.ema_fusion.as_ref().map(Ema::updates),
            lora: self.config.lora,
            fusion_optimizer,
            fusion_ema,
        };
        let state = TrainState {
            format_version: checkpoint::STATE_FORMAT_VERSION,
            kind: "geomfusion".into(),
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

    /// Resume from a checkpoint. The caller supplies a FRESH fused model (in
    /// NF4 mode: the trunk re-quantized from the shards, deterministic; the
    /// stream freshly initialized -- the record overwrites it, values and
    /// `ParamId`s alike, so the optimizer records match).
    pub fn resume(
        model_path: &Path,
        mut model: FusedQwen<B>,
        config: FusedTrainConfig,
        device: &B::Device,
    ) -> anyhow::Result<(Self, ChaCha12Rng)> {
        let state = TrainState::for_model(model_path)?
            .with_context(|| format!("no training state next to {}", model_path.display()))?;
        anyhow::ensure!(
            state.kind == "geomfusion",
            "training state is for {:?}, not \"geomfusion\"",
            state.kind
        );
        let state_dir = TrainState::dir_for(model_path);
        state.verify_files(&state_dir)?;
        let extras: FusedTrainExtras = serde_json::from_value(state.extras.clone())
            .context("parse geomfusion training extras")?;

        qwentrain::attach_lora(&mut model.trunk, &config.lora, device)?;
        model = checkpoint::load(model, model_path, device)?;
        qwentrain::freeze_non_adapter_params(&mut model.trunk);
        let bank = QwenAdapterBank::from_trunk(&model.trunk)?;
        let mut trunk_optim: QwenOptim<B> = AdamWConfig::new().init();
        if let Some(file) = &state.optimizer {
            let record: QwenOptimRecord<B> = checkpoint::load_record(&state_dir, file, device)?;
            trunk_optim = trunk_optim.load_record(record);
        }
        let mut fusion_optim: FusionOptim<B> = AdamWConfig::new().init();
        if let Some(file) = &extras.fusion_optimizer {
            let record: FusionOptimRecord<B> =
                checkpoint::load_record(&state_dir, file, device)?;
            fusion_optim = fusion_optim.load_record(record);
        }
        let ema_trunk = match (config.ema_decay, &state.ema) {
            (Some(decay), Some(file)) => {
                let shadow = bank
                    .clone()
                    .load_record(checkpoint::load_record(&state_dir, file, device)?);
                Some(Ema::from_parts(
                    shadow,
                    decay,
                    extras.ema_updates_trunk.unwrap_or(0),
                ))
            }
            (Some(decay), None) => Some(Ema::new(&bank, decay)),
            (None, _) => None,
        };
        let ema_fusion = match (config.ema_decay, &extras.fusion_ema) {
            (Some(decay), Some(file)) => {
                let shadow = model
                    .fusion
                    .clone()
                    .load_record(checkpoint::load_record(&state_dir, file, device)?);
                Some(Ema::from_parts(
                    shadow,
                    decay,
                    extras.ema_updates_fusion.unwrap_or(0),
                ))
            }
            (Some(decay), None) => Some(Ema::new(&model.fusion, decay)),
            (None, _) => None,
        };
        let rng: ChaCha12Rng =
            serde_json::from_value(state.host_rng.clone()).context("restore host RNG")?;
        let trainer = Self {
            acc_trunk: GradientAccumulator::new(config.accumulate),
            acc_fusion: GradientAccumulator::new(config.accumulate),
            model,
            trunk_optim,
            fusion_optim,
            bank,
            ema_trunk,
            ema_fusion,
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

/// What a fused training run did.
#[derive(Debug, Clone)]
pub struct FusedTrainReport {
    pub steps_taken: usize,
    /// Micro-batches discarded for a non-finite loss.
    pub steps_skipped: usize,
    pub steps_clipped: usize,
    pub first_loss: f32,
    pub last_loss: f32,
    pub mean_loss: f32,
    pub first_accuracy: f32,
    pub last_accuracy: f32,
    pub mean_accuracy: f32,
    pub scenes_seen: usize,
    pub elapsed_secs: f64,
    pub checkpoint: Option<PathBuf>,
}

/// The training driver: sample aligned scenes from the corpus HEAD (the
/// `holdout` tail is [`prove`]'s), tokenize them with the trunk's BPE, and
/// step the trainer. Mirrors `geometry::train_geom`'s loop shape
/// (per-step reseeding, skip-on-non-finite, report), with the geometric
/// corpus's own batching (`TokenCorpus::window` over aligned scenes).
pub fn train_fused_qwen<B: AutodiffBackend<FloatElem = f32>>(
    trainer: &mut FusedQwenTrainer<B>,
    corpus: &mut TokenCorpus,
    meta: &GeomMeta,
    tokenizer: &BpeTokenizer,
    out_dir: Option<&Path>,
) -> anyhow::Result<FusedTrainReport> {
    anyhow::ensure!(
        corpus.len() % meta.scene_len == 0,
        "the corpus holds {} tokens, not a multiple of the {}-token scene length",
        corpus.len(),
        meta.scene_len
    );
    let total_scenes = corpus.len() / meta.scene_len;
    anyhow::ensure!(
        total_scenes > trainer.config.holdout,
        "the corpus holds {total_scenes} scenes, not more than the {}-scene holdout",
        trainer.config.holdout
    );
    let train_scenes = total_scenes - trainer.config.holdout;
    let mut rng = ChaCha12Rng::seed_from_u64(trainer.config.seed);
    let mut report = FusedTrainReport {
        steps_taken: 0,
        steps_skipped: 0,
        steps_clipped: 0,
        first_loss: f32::NAN,
        last_loss: f32::NAN,
        mean_loss: 0.0,
        first_accuracy: f32::NAN,
        last_accuracy: f32::NAN,
        mean_accuracy: 0.0,
        scenes_seen: 0,
        elapsed_secs: 0.0,
        checkpoint: None,
    };
    let mut loss_sum = 0.0f64;
    let mut accuracy_sum = 0.0f64;
    let started = std::time::Instant::now();
    for step in 0..trainer.config.steps {
        let mut scenes = Vec::with_capacity(trainer.config.batch_size);
        for _ in 0..trainer.config.batch_size {
            let scene_idx = rng.random_range(0..train_scenes);
            let window = corpus.window(scene_idx * meta.scene_len, meta.scene_len)?;
            scenes.push(tokenize_scene(tokenizer, &window)?);
        }
        let step_report = trainer.train_step(&scenes)?;
        if !step_report.loss.is_finite() {
            report.steps_skipped += 1;
            println!("step {step}: non-finite loss {}, step discarded", step_report.loss);
            continue;
        }
        if report.steps_taken == 0 {
            report.first_loss = step_report.loss;
            report.first_accuracy = step_report.accuracy;
        }
        report.last_loss = step_report.loss;
        report.last_accuracy = step_report.accuracy;
        report.steps_clipped += usize::from(step_report.clipped);
        report.scenes_seen += scenes.len();
        loss_sum += f64::from(step_report.loss);
        accuracy_sum += f64::from(step_report.accuracy);
        report.steps_taken += 1;
        if trainer.config.log_every > 0
            && (step % trainer.config.log_every == 0 || step + 1 == trainer.config.steps)
        {
            println!(
                "step {step}: loss {:.4} accuracy {:.4} | energy {:.4} mse {:.4} | grads {:.3e}/{:.3e}{}",
                step_report.loss,
                step_report.accuracy,
                step_report.energy,
                step_report.mse,
                step_report.grad_norm_trunk,
                step_report.grad_norm_fusion,
                if step_report.clipped { " (clipped)" } else { "" }
            );
        }
    }
    report.elapsed_secs = started.elapsed().as_secs_f64();
    report.mean_loss = (loss_sum / report.steps_taken.max(1) as f64) as f32;
    report.mean_accuracy = (accuracy_sum / report.steps_taken.max(1) as f64) as f32;
    if let Some(dir) = out_dir {
        report.checkpoint = Some(trainer.save_checkpoint(dir, &rng)?);
    }
    Ok(report)
}

// ------------------------------------------------------------------ proof ---

/// What a proof run evaluates.
#[derive(Debug, Clone)]
pub struct ProofConfig {
    /// Held-out scenes, read from the corpus TAIL (the region
    /// [`train_fused_qwen`]'s `holdout` keeps out of training).
    pub scenes: usize,
    /// Refinement depth for the headline accuracy (`None` = configured).
    pub depth: Option<usize>,
    /// Also run the refine sweep.
    pub sweep: bool,
    /// Depths of the sweep; empty defaults to `1..=2 * refine_steps`.
    pub sweep_depths: Vec<usize>,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            scenes: 256,
            depth: None,
            sweep: true,
            sweep_depths: Vec::new(),
        }
    }
}

/// One row of the refinement sweep.
#[derive(Debug, Clone)]
pub struct RefinePoint {
    pub depth: usize,
    pub accuracy: f32,
    /// Mean final refinement energy at this depth.
    pub mean_energy: f32,
    /// Fraction of scenes whose prediction at this depth equals the
    /// depth-1 prediction. High agreement where the accuracy curve is flat
    /// is the attractor; high agreement where accuracy still climbs would
    /// mean the depth knob is decoration.
    pub agree_with_depth1: f32,
}

/// Per-kind breakdown: the hard kinds must not hide behind the easy ones.
#[derive(Debug, Clone)]
pub struct KindProof {
    pub kind: String,
    pub scenes: usize,
    pub correct: usize,
    pub verified: usize,
    pub falsified: usize,
    pub indeterminate: usize,
}

/// The proof harness's report.
#[derive(Debug, Clone)]
pub struct ProofReport {
    pub scenes: usize,
    pub correct: usize,
    pub accuracy: f32,
    /// The best constant-guess accuracy: the largest empirical share of any
    /// answer byte over the evaluated scenes. Computed from the answer
    /// distribution, not quoted.
    pub chance: f32,
    /// `accuracy - chance`.
    pub margin: f32,
    /// Emitted answers the exact kernel confirmed.
    pub verified: usize,
    /// Emitted answers the exact kernel refuted.
    pub falsified: usize,
    /// Emitted answers the kernel cannot decide (direction bearings,
    /// degenerate ties, out-of-alphabet proposals).
    pub indeterminate: usize,
    pub per_kind: Vec<KindProof>,
    /// The refinement sweep, empty when `sweep` was off.
    pub sweep: Vec<RefinePoint>,
}

/// Predict every scene at one depth. Returns the predicted bytes and the
/// mean final refinement energy.
fn predict_all<B: Backend<FloatElem = f32>>(
    model: &FusedQwen<B>,
    scenes: &[TokenizedScene],
    candidates: &[(u16, u32)],
    depth: Option<usize>,
    device: &B::Device,
) -> anyhow::Result<(Vec<u8>, f32)> {
    let mut predictions = Vec::with_capacity(scenes.len());
    let mut energy = 0.0f32;
    for scene in scenes {
        let (byte, fused) = model.predict_answer(scene, candidates, depth, device)?;
        energy += fused
            .traces
            .last()
            .and_then(|t| t.energy.last())
            .copied()
            .unwrap_or(0.0);
        predictions.push(byte);
    }
    Ok((predictions, energy / scenes.len().max(1) as f32))
}

/// The proof of reasoning: accuracy on held-out scenes against the
/// empirical answer distribution, the kernel's verdict on every emitted
/// answer, and (optionally) the refinement sweep with agree-with-depth-1
/// trace dependence. Reports what it measures, whatever it measures: a
/// fresh fusion at the identity sits at chance, and that is the honest
/// baseline the training run has to move.
pub fn prove<B: Backend<FloatElem = f32>>(
    model: &FusedQwen<B>,
    corpus: &mut TokenCorpus,
    meta: &GeomMeta,
    tokenizer: &BpeTokenizer,
    config: &ProofConfig,
    device: &B::Device,
) -> anyhow::Result<ProofReport> {
    anyhow::ensure!(config.scenes >= 1, "scenes must be positive");
    anyhow::ensure!(
        corpus.len() % meta.scene_len == 0,
        "the corpus holds {} tokens, not a multiple of the {}-token scene length",
        corpus.len(),
        meta.scene_len
    );
    let total_scenes = corpus.len() / meta.scene_len;
    anyhow::ensure!(
        config.scenes <= total_scenes,
        "asked for {} held-out scenes, the corpus holds {total_scenes}",
        config.scenes
    );
    let candidates = answer_token_ids(tokenizer, meta)?;
    let start = total_scenes - config.scenes;
    let mut scenes = Vec::with_capacity(config.scenes);
    let mut parsed = Vec::with_capacity(config.scenes);
    for idx in 0..config.scenes {
        let window = corpus.window((start + idx) * meta.scene_len, meta.scene_len)?;
        parsed.push(parse_scene(&window)?);
        scenes.push(tokenize_scene(tokenizer, &window)?);
    }

    // Chance from the answer distribution: the best a constant guess can
    // do on these scenes.
    let mut histogram: HashMap<u8, usize> = HashMap::new();
    for scene in &parsed {
        *histogram.entry(scene.answer).or_insert(0) += 1;
    }
    let chance = histogram
        .values()
        .copied()
        .max()
        .unwrap_or(0) as f32
        / config.scenes as f32;

    let (predictions, _) = predict_all(model, &scenes, &candidates, config.depth, device)?;
    let mut report = ProofReport {
        scenes: config.scenes,
        correct: 0,
        accuracy: 0.0,
        chance,
        margin: 0.0,
        verified: 0,
        falsified: 0,
        indeterminate: 0,
        per_kind: Vec::new(),
        sweep: Vec::new(),
    };
    let mut kinds: Vec<KindProof> = meta
        .kinds
        .iter()
        .map(|kind| KindProof {
            kind: kind.clone(),
            scenes: 0,
            correct: 0,
            verified: 0,
            falsified: 0,
            indeterminate: 0,
        })
        .collect();
    for (scene, (&predicted, parsed_scene)) in scenes.iter().zip(predictions.iter().zip(&parsed)) {
        let _ = scene;
        let correct = predicted == parsed_scene.answer;
        report.correct += usize::from(correct);
        let verdict = kernel_check(parsed_scene, predicted)?;
        match verdict {
            KernelVerdict::Verified => report.verified += 1,
            KernelVerdict::Falsified => report.falsified += 1,
            KernelVerdict::Indeterminate => report.indeterminate += 1,
        }
        if let Some(entry) = kinds.iter_mut().find(|k| k.kind == parsed_scene.kind) {
            entry.scenes += 1;
            entry.correct += usize::from(correct);
            match verdict {
                KernelVerdict::Verified => entry.verified += 1,
                KernelVerdict::Falsified => entry.falsified += 1,
                KernelVerdict::Indeterminate => entry.indeterminate += 1,
            }
        }
    }
    report.accuracy = report.correct as f32 / config.scenes as f32;
    report.margin = report.accuracy - chance;
    report.per_kind = kinds;

    if config.sweep {
        let depths: Vec<usize> = if config.sweep_depths.is_empty() {
            (1..=(2 * model.fusion.refine_steps()).max(1)).collect()
        } else {
            config.sweep_depths.clone()
        };
        let (shallow, _) = predict_all(model, &scenes, &candidates, Some(1), device)?;
        for depth in depths {
            let (predictions, mean_energy) =
                predict_all(model, &scenes, &candidates, Some(depth), device)?;
            let correct = predictions
                .iter()
                .zip(parsed.iter())
                .filter(|(p, s)| **p == s.answer)
                .count();
            let agree = predictions
                .iter()
                .zip(shallow.iter())
                .filter(|(a, b)| a == b)
                .count();
            report.sweep.push(RefinePoint {
                depth,
                accuracy: correct as f32 / config.scenes as f32,
                mean_energy,
                agree_with_depth1: agree as f32 / config.scenes as f32,
            });
        }
    }
    Ok(report)
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
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    fn tiny_model() -> GeomFusion<B> {
        let device = Default::default();
        GeomFusion::<B>::new(&GeomFusionConfig::tiny(), &device).unwrap()
    }

    fn tiny_hidden() -> Tensor<B, 3> {
        let device = Default::default();
        Tensor::<B, 3>::random([2, 12, 64], Distribution::Normal(0.0, 1.0), &device)
    }

    #[test]
    fn test_fused_forward_is_finite_and_shaped() {
        let model = tiny_model();
        let hidden = tiny_hidden();
        let out = model.forward(hidden, None).unwrap();
        assert_eq!(out.hidden.dims(), [2, 12, 64]);
        assert_eq!(out.final_state.dims(), [2, 8, 16]);
        assert_eq!(out.traces.len(), model.num_blocks());
        let finite: f32 = out.hidden.clone().abs().max().into_scalar();
        assert!(finite.is_finite(), "fused output is not finite: {finite}");
        // One gate row per token, one column per (box, expert) head.
        assert_eq!(out.gates.dims(), [2 * 12, 4 * 2]);
    }

    #[test]
    fn test_the_residual_is_added_exactly() {
        let model = tiny_model();
        let hidden = tiny_hidden();
        // The fused output is `hidden + out_proj(mixture)`: recompute the
        // right-hand side from the geometric path's own parts and demand bit
        // equality -- the same op order, so any deviation is the residual
        // wiring, not float noise.
        let path = model.geometric_path(&hidden, None).unwrap();
        let expected = hidden
            .clone()
            .add(model.out_proj.forward(path.mixture.clone()));
        let fused = model.forward(hidden, None).unwrap().hidden;
        let diff: f32 = (fused - expected).abs().max().into_scalar();
        assert_eq!(diff, 0.0, "the geometric readout is not added as-is");
    }

    #[test]
    fn test_fusion_is_the_identity_at_zero_projection() {
        let model = tiny_model();
        let hidden = tiny_hidden();
        let fused = model.forward(hidden.clone(), None).unwrap().hidden;
        let diff: f32 = (fused - hidden).abs().max().into_scalar();
        assert_eq!(
            diff, 0.0,
            "a zero-initialized out_proj must leave the trunk untouched"
        );
    }

    #[test]
    fn test_refine_depth_lowers_the_energy_monotonically() {
        let model = tiny_model();
        let hidden = tiny_hidden();
        let deep = model.forward(hidden.clone(), Some(12)).unwrap();
        // Within a block the landscape is fixed, so the certified step never
        // raises the energy it reports (the geometry.rs discipline: 1e-4).
        for (block, trace) in deep.traces.iter().enumerate() {
            assert_eq!(trace.energy.len(), 12);
            for step in trace.energy.windows(2) {
                assert!(
                    step[1] <= step[0] + 1e-4,
                    "block {block}: energy rose from {} to {}",
                    step[0],
                    step[1]
                );
            }
        }
        // Across depths: block 0's relaxation is the same trajectory sampled
        // at different lengths, so the longer run ends no higher.
        let shallow = model.forward(hidden, Some(1)).unwrap();
        let e1 = shallow.traces[0].energy[0];
        let e12 = *deep.traces[0].energy.last().unwrap();
        assert!(
            e12 <= e1 + 1e-4,
            "twelve steps ended at {e12}, above one step's {e1}"
        );
    }

    #[test]
    fn test_readout_gates_are_a_distribution_per_row() {
        let model = tiny_model();
        let hidden = tiny_hidden();
        let out = model.forward(hidden, None).unwrap();
        // The composed gates come straight out of the certified
        // `moe::scatter_gates` inside `HierarchicalRouter::route`: every row
        // is a distribution over the (box, expert) heads. This is the check
        // the old hand-rolled broadcast-scatter (rows summing to n_boxes)
        // would have failed.
        let sums: Vec<f32> = out
            .gates
            .sum_dim(1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        assert_eq!(sums.len(), 24);
        for (row, s) in sums.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-6, "gate row {row} sums to {s}, not 1");
        }
    }

    #[test]
    fn test_the_geometric_path_is_live() {
        // Turn the output projection on and drive the refinement to two
        // depths: if the fused outputs agree, the stream is decoration.
        let device = Default::default();
        let mut model = tiny_model();
        model.out_proj = LinearConfig::new(32, 64).with_bias(false).init(&device);
        let hidden = tiny_hidden();
        let shallow = model.forward(hidden.clone(), Some(1)).unwrap().hidden;
        let deep = model.forward(hidden.clone(), Some(8)).unwrap().hidden;
        let moved: f32 = (shallow.clone() - deep.clone()).abs().max().into_scalar();
        assert!(
            moved > 1e-6,
            "deeper refinement changed nothing: the stream is not live"
        );
        // And the stream does move the representation at all: a live
        // out_proj over the refined readout is not the trunk input.
        let base: f32 = (shallow - hidden).abs().max().into_scalar();
        assert!(base > 1e-6, "the residual added nothing");
    }

    #[test]
    fn test_invalid_config_names_the_field() {
        let device = Default::default();
        let cases: Vec<(GeomFusionConfig, &str)> = vec![
            (
                GeomFusionConfig {
                    hidden_size: 0,
                    ..GeomFusionConfig::tiny()
                },
                "hidden_size",
            ),
            (
                GeomFusionConfig {
                    slots: 1,
                    ..GeomFusionConfig::tiny()
                },
                "slots",
            ),
            (
                GeomFusionConfig {
                    repulsion: 0.0,
                    ..GeomFusionConfig::tiny()
                },
                "repulsion",
            ),
            (
                GeomFusionConfig {
                    moe_top_k: 3,
                    ..GeomFusionConfig::tiny()
                },
                "moe_top_k",
            ),
            (
                GeomFusionConfig {
                    frequency_embedding_size: 7,
                    ..GeomFusionConfig::tiny()
                },
                "frequency_embedding_size",
            ),
        ];
        for (config, field) in cases {
            let err = GeomFusion::<B>::new(&config, &device).unwrap_err();
            assert!(
                format!("{err:#}").contains(field),
                "error for {field} names something else: {err:#}"
            );
        }
        // A forward pass on a mismatched width or a too-short sequence names
        // its constraint too.
        let model = tiny_model();
        let wide = Tensor::<B, 3>::random([1, 12, 128], Distribution::Normal(0.0, 1.0), &device);
        let err = model.forward(wide, None).unwrap_err();
        assert!(format!("{err:#}").contains("hidden_size"));
        let short = Tensor::<B, 3>::random([1, 4, 64], Distribution::Normal(0.0, 1.0), &device);
        let err = model.forward(short, None).unwrap_err();
        assert!(format!("{err:#}").contains("slots"));
    }
}
