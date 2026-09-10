//! Hybrid attention stack (roadmap Phase 25): per-layer attention modes,
//! rotary positions, and the per-layer inference state each mode needs.
//!
//! The language trunk of Phase 19 attends densely in every layer. That is the
//! right *control*, but it is not the only way to spend attention compute:
//! GLM-5.3-Flash and Nemotron 3 interleave cheap recurrent / linear layers
//! with a few expensive precise ones, and the roadmap asks for that mixture to
//! be **configurable** so 1:1, 2:1, 3:1, 4:1 and learned schedules can be
//! compared on the same trunk. This module holds the four modes, the schedule
//! that assigns one to every layer, and the mathematics each mode needs; the
//! wiring into the layer lives in [`crate::vit`].
//!
//! | Mode | Cost per token | What it keeps at decode time |
//! |---|---|---|
//! | [`AttentionMode::Dense`] | `O(n)` keys | every key and value |
//! | [`AttentionMode::Sliding`] | `O(w)` keys | the last `w - 1` keys and values |
//! | [`AttentionMode::Retrieval`] | scores `O(n)`, values `O(k)` | every key and value; reads only the `k` best |
//! | [`AttentionMode::Linear`] | `O(d^2)` | a `[d, d]` state and a `[d]` normalizer |
//! | [`AttentionMode::Learned`] | dense + linear | both |
//!
//! # Linear attention, and why its state form is exact
//!
//! With a positive feature map `phi` (here `elu(x) + 1`), causal linear
//! attention is
//!
//! ```text
//! out_i = phi(q_i)^T S_i / (phi(q_i) . z_i),   S_i = sum_{j<=i} phi(k_j) v_j^T,   z_i = sum_{j<=i} phi(k_j)
//! ```
//!
//! which is a recurrence in `(S, z)` -- the "recurrent state" of a
//! Mamba/KDA-style layer -- **and** the masked matrix product
//! `(phi(Q) phi(K)^T ⊙ M) V` row-normalized. The two are the same numbers up to
//! summation order, so the decode-time state is not an approximation of the
//! training-time computation; certificate `hybrid/linear_state_matches_masked_form`
//! demands they agree.
//!
//! # Rotary positions
//!
//! The Phase 19 model adds a learned table to the embedding, which bounds the
//! sequence length by the table and stops a cache at that edge. Rotary
//! positions ([`PositionKind::Rotary`]) rotate each query/key pair by an angle
//! proportional to its absolute position, so a score depends only on the
//! *distance* `i - j` (certificate `hybrid/rotary_scores_depend_on_distance`),
//! any offset is valid, and a sliding-window or linear layer can continue
//! past any fixed context. What rotary positions do **not** promise is that a
//! model trained on short sequences generalizes to long ones; that is a
//! quality question for a GPU.

use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

/// How one layer attends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum AttentionMode {
    /// Standard softmax attention over every earlier position: the control.
    Dense,
    /// Softmax attention over the last `window` positions (the query included).
    Sliding { window: usize },
    /// Softmax attention over the `top_k` highest-scoring earlier positions:
    /// sparse retrieval. The scores are still computed for every key -- the
    /// saving is in what is *read*, which is what a value-bandwidth-bound
    /// decoder pays for.
    Retrieval { top_k: usize },
    /// Causal linear attention with a recurrent state (see the module docs).
    Linear,
    /// A learned per-layer mixture of dense and linear attention: training
    /// decides the schedule.
    Learned,
}

impl AttentionMode {
    /// Short name, as accepted by [`AttentionSchedule::parse`].
    pub fn name(&self) -> String {
        match self {
            Self::Dense => "dense".into(),
            Self::Sliding { window } => format!("sliding{window}"),
            Self::Retrieval { top_k } => format!("retrieval{top_k}"),
            Self::Linear => "linear".into(),
            Self::Learned => "learned".into(),
        }
    }

    /// One letter, for compact schedule strings: `D`, `S`, `R`, `L`, `M`.
    pub fn letter(&self) -> char {
        match self {
            Self::Dense => 'D',
            Self::Sliding { .. } => 'S',
            Self::Retrieval { .. } => 'R',
            Self::Linear => 'L',
            Self::Learned => 'M',
        }
    }

    /// Whether a key/value cache is what this mode keeps at decode time.
    pub fn keeps_kv(&self) -> bool {
        !matches!(self, Self::Linear)
    }

    /// Whether the linear-attention state is part of what this mode keeps.
    pub fn keeps_linear_state(&self) -> bool {
        matches!(self, Self::Linear | Self::Learned)
    }

    /// Parse one mode: `dense`, `linear`, `learned`, `sliding<w>`,
    /// `retrieval<k>`, or the single letters `D L M S R` (with the default
    /// window / top-k).
    pub fn parse(text: &str, default_window: usize, default_top_k: usize) -> anyhow::Result<Self> {
        let t = text.trim();
        let lower = t.to_ascii_lowercase();
        Ok(match lower.as_str() {
            "dense" | "d" | "full" => Self::Dense,
            "linear" | "l" | "recurrent" => Self::Linear,
            "learned" | "m" | "mixed" => Self::Learned,
            "sliding" | "s" | "window" => Self::Sliding { window: default_window },
            "retrieval" | "r" | "sparse" => Self::Retrieval { top_k: default_top_k },
            _ => {
                if let Some(rest) = lower.strip_prefix("sliding") {
                    let window: usize = rest.parse().map_err(|_| anyhow::anyhow!("bad sliding window in '{t}'"))?;
                    anyhow::ensure!(window >= 1, "a sliding window must be at least 1");
                    Self::Sliding { window }
                } else if let Some(rest) = lower.strip_prefix("retrieval") {
                    let top_k: usize = rest.parse().map_err(|_| anyhow::anyhow!("bad retrieval top-k in '{t}'"))?;
                    anyhow::ensure!(top_k >= 1, "retrieval must read at least 1 key");
                    Self::Retrieval { top_k }
                } else {
                    anyhow::bail!("unknown attention mode '{t}' (expected dense|linear|learned|sliding<w>|retrieval<k>)")
                }
            }
        })
    }
}

/// One attention mode per layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionSchedule {
    pub modes: Vec<AttentionMode>,
}

impl AttentionSchedule {
    /// Every layer dense: the pre-Phase-25 trunk.
    pub fn dense(num_layers: usize) -> Self {
        Self { modes: vec![AttentionMode::Dense; num_layers] }
    }

    /// `cheap` layers of `filler` for every `expensive` layer of `precise`,
    /// the precise one last in each group (so the trunk ends on a precise
    /// layer whenever the ratio divides the depth).
    pub fn ratio(num_layers: usize, cheap: usize, filler: AttentionMode, precise: AttentionMode) -> Self {
        let group = cheap + 1;
        let modes = (0..num_layers)
            .map(|i| if i % group == group - 1 { precise } else { filler })
            .collect();
        Self { modes }
    }

    /// Parse a schedule for `num_layers` layers.
    ///
    /// Accepted forms:
    /// - a single mode name (`dense`, `linear`, ...): every layer;
    /// - `<cheap>:1`, e.g. `3:1`: three linear layers per dense one;
    /// - `<cheap>:1@<mode>` to pick the cheap mode, e.g. `3:1@sliding64`;
    /// - a comma-separated list, one entry per layer, e.g.
    ///   `linear,linear,dense,linear`;
    /// - a letter pattern such as `LLLD`, repeated to cover the depth
    ///   (`D` dense, `L` linear, `M` learned, `S` sliding, `R` retrieval).
    pub fn parse(text: &str, num_layers: usize, default_window: usize, default_top_k: usize) -> anyhow::Result<Self> {
        let t = text.trim();
        anyhow::ensure!(num_layers > 0, "a schedule needs at least one layer");
        if t.is_empty() {
            return Ok(Self::dense(num_layers));
        }
        if let Some((cheap, rest)) = t.split_once(':') {
            let cheap: usize = cheap.trim().parse().map_err(|_| anyhow::anyhow!("bad ratio '{t}'"))?;
            let (one, filler) = match rest.split_once('@') {
                Some((one, mode)) => (one, AttentionMode::parse(mode, default_window, default_top_k)?),
                None => (rest, AttentionMode::Linear),
            };
            anyhow::ensure!(one.trim() == "1", "a ratio is written <cheap>:1, got '{t}'");
            return Ok(Self::ratio(num_layers, cheap, filler, AttentionMode::Dense));
        }
        if t.contains(',') {
            let modes: Vec<AttentionMode> = t
                .split(',')
                .map(|m| AttentionMode::parse(m, default_window, default_top_k))
                .collect::<anyhow::Result<_>>()?;
            anyhow::ensure!(
                modes.len() == num_layers,
                "schedule lists {} modes for {num_layers} layers",
                modes.len()
            );
            return Ok(Self { modes });
        }
        if let Ok(mode) = AttentionMode::parse(t, default_window, default_top_k) {
            if t.len() > 1 {
                return Ok(Self { modes: vec![mode; num_layers] });
            }
        }
        // A letter pattern, repeated across the depth.
        let pattern: Vec<AttentionMode> = t
            .chars()
            .map(|c| AttentionMode::parse(&c.to_string(), default_window, default_top_k))
            .collect::<anyhow::Result<_>>()?;
        Ok(Self { modes: (0..num_layers).map(|i| pattern[i % pattern.len()]).collect() })
    }

    pub fn num_layers(&self) -> usize {
        self.modes.len()
    }

    pub fn mode(&self, layer: usize) -> AttentionMode {
        self.modes.get(layer).copied().unwrap_or(AttentionMode::Dense)
    }

    pub fn is_all_dense(&self) -> bool {
        self.modes.iter().all(|m| *m == AttentionMode::Dense)
    }

    /// The letter pattern, e.g. `LLLDLLLD`.
    pub fn pattern(&self) -> String {
        self.modes.iter().map(AttentionMode::letter).collect()
    }

    /// Count of layers per mode letter, e.g. `6L 2D`.
    pub fn summary(&self) -> String {
        let mut counts: Vec<(char, usize)> = Vec::new();
        for m in &self.modes {
            let c = m.letter();
            match counts.iter_mut().find(|(k, _)| *k == c) {
                Some((_, n)) => *n += 1,
                None => counts.push((c, 1)),
            }
        }
        counts.iter().map(|(c, n)| format!("{n}{c}")).collect::<Vec<_>>().join(" ")
    }
}

/// How positions enter the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionKind {
    /// A learned `[context, hidden]` table added to the embedding (Phase 19).
    /// Bounds the sequence length by the table.
    #[default]
    Learned,
    /// Rotary embedding applied to queries and keys in every layer. No table,
    /// no bound: any absolute offset is valid.
    Rotary,
    /// No position signal at all (NoPE). Causal masking alone breaks the
    /// symmetry; a control for the position ablation.
    None,
}

impl PositionKind {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        Ok(match text.trim().to_ascii_lowercase().as_str() {
            "learned" | "table" | "absolute" => Self::Learned,
            "rotary" | "rope" => Self::Rotary,
            "none" | "nope" => Self::None,
            other => anyhow::bail!("unknown position kind '{other}' (expected learned|rotary|none)"),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Learned => "learned",
            Self::Rotary => "rotary",
            Self::None => "none",
        }
    }

    /// Whether the sequence length is bounded by a position table.
    pub fn is_bounded(&self) -> bool {
        matches!(self, Self::Learned)
    }
}

// ------------------------------------------------------------ masks --

/// Additive mask `[1, 1, m, total]` for `m` queries at absolute positions
/// `query_offset + j` over `total` keys whose first key sits at absolute
/// position `key_offset`. A key is visible when it is not in the future and,
/// with `window = Some(w)`, when it is among the last `w` positions up to and
/// including the query.
pub fn attention_mask<B: Backend>(
    m: usize,
    total: usize,
    query_offset: usize,
    key_offset: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut values = Vec::with_capacity(m * total);
    for query in 0..m {
        let q = query_offset + query;
        for key in 0..total {
            let k = key_offset + key;
            let visible = k <= q && window.is_none_or(|w| k + w > q);
            values.push(if visible { 0.0f32 } else { f32::NEG_INFINITY });
        }
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([1, 1, m, total])
}

/// Multiplicative `0/1` causal mask `[1, 1, m, total]` with the same geometry
/// as [`attention_mask`] and no window: what linear attention multiplies by.
pub fn causal_keep_mask<B: Backend>(m: usize, total: usize, query_offset: usize, key_offset: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut values = Vec::with_capacity(m * total);
    for query in 0..m {
        for key in 0..total {
            values.push(if key_offset + key <= query_offset + query { 1.0f32 } else { 0.0 });
        }
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([1, 1, m, total])
}

/// Keep only the `top_k` largest entries of every row of `scores`
/// `[b, h, m, total]`, replacing the rest by `-inf`. Rows with fewer than
/// `top_k` finite entries keep all of theirs; a `-inf` entry that is chosen
/// stays `-inf`, so it weighs exactly nothing after the softmax.
pub fn keep_top_k<B: Backend>(scores: Tensor<B, 4>, top_k: usize) -> Tensor<B, 4> {
    let [b, h, m, total] = scores.dims();
    if top_k >= total {
        return scores;
    }
    let (_, idx) = scores.clone().topk_with_indices(top_k, 3); // [b, h, m, k]
    let device = scores.device();
    let ones = Tensor::<B, 4>::ones([b, h, m, top_k], &device);
    let keep = Tensor::<B, 4>::zeros([b, h, m, total], &device).scatter(3, idx, ones, burn::tensor::IndexingUpdateOp::Add);
    let dropped: Tensor<B, 4, Bool> = keep.equal_elem(0.0);
    scores.mask_fill(dropped, f32::NEG_INFINITY)
}

// ------------------------------------------------------- linear attention --

/// The positive feature map `elu(x) + 1`, written as `max(x, 0) + exp(min(x, 0))`
/// so it is exact in both branches and never produces a zero -- a zero row of
/// `phi(k)` would make the normalizer vanish.
pub fn feature_map<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().clamp_min(0.0) + x.clamp_max(0.0).exp()
}

/// Recurrent state of one linear-attention layer: `S = sum phi(k) v^T`
/// `[b, heads, d, d]` and `z = sum phi(k)` `[b, heads, d]`.
#[derive(Debug, Clone)]
pub struct LinearState<B: Backend> {
    pub s: Tensor<B, 4>,
    pub z: Tensor<B, 3>,
}

/// Causal linear attention over `m` new queries, given `phi(q)`, `phi(k)`
/// and `v` for the new positions `[b, heads, m, d]` and an optional state
/// carrying every earlier position. Returns the outputs `[b, heads, m, d]`
/// and the state after absorbing the new keys.
///
/// The intra-chunk part is the masked matrix form; the inter-chunk part reads
/// the state. With no state and `m = n` this is the full training-time
/// computation, which is what makes the two forms comparable.
pub fn linear_attention<B: Backend>(
    phi_q: Tensor<B, 4>,
    phi_k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    state: Option<LinearState<B>>,
    eps: f32,
) -> (Tensor<B, 4>, LinearState<B>) {
    let [b, heads, m, d] = phi_q.dims();
    let device = phi_q.device();

    // Intra-chunk: (phi(Q) phi(K)^T ⊙ M) V, normalized by the masked row sums.
    let mask = causal_keep_mask::<B>(m, m, 0, 0, &device);
    let scores = phi_q.clone().matmul(phi_k.clone().swap_dims(2, 3)) * mask; // [b, h, m, m]
    let mut numerator = scores.clone().matmul(v.clone()); // [b, h, m, d]
    let mut denominator = scores.sum_dim(3); // [b, h, m, 1]

    // Inter-chunk: every earlier position, folded into the state.
    let (s_prev, z_prev) = match state {
        Some(LinearState { s, z }) => (s, z),
        None => (
            Tensor::<B, 4>::zeros([b, heads, d, d], &device),
            Tensor::<B, 3>::zeros([b, heads, d], &device),
        ),
    };
    numerator = numerator + phi_q.clone().matmul(s_prev.clone());
    denominator = denominator + phi_q.matmul(z_prev.clone().unsqueeze_dim::<4>(3));

    let out = numerator / denominator.clamp_min(eps);

    // Absorb the new keys.
    let s = s_prev + phi_k.clone().swap_dims(2, 3).matmul(v); // [b, h, d, d]
    let z = z_prev + phi_k.sum_dim(2).squeeze_dim::<3>(2); // [b, h, d]
    (out, LinearState { s, z })
}

// ------------------------------------------------------------- rotary --

/// Rotary tables `(cos, sin)` for `n` positions starting at `offset`, each
/// `[1, 1, n, dim]` with the half-dimension frequencies duplicated so they
/// broadcast against `[b, heads, n, dim]`. `dim` must be even.
pub fn rotary_tables<B: Backend>(offset: usize, n: usize, dim: usize, base: f64, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    assert!(dim % 2 == 0, "rotary positions need an even head dimension, got {dim}");
    let half = dim / 2;
    let mut cos = Vec::with_capacity(n * dim);
    let mut sin = Vec::with_capacity(n * dim);
    for p in 0..n {
        let position = (offset + p) as f64;
        let mut row_cos = Vec::with_capacity(dim);
        let mut row_sin = Vec::with_capacity(dim);
        for i in 0..half {
            let inv_freq = base.powf(-(2.0 * i as f64) / dim as f64);
            let angle = position * inv_freq;
            row_cos.push(angle.cos() as f32);
            row_sin.push(angle.sin() as f32);
        }
        // Duplicate so the table matches the rotate-half layout `[x1, x2]`.
        cos.extend_from_slice(&row_cos);
        cos.extend_from_slice(&row_cos);
        sin.extend_from_slice(&row_sin);
        sin.extend_from_slice(&row_sin);
    }
    (
        Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([1, 1, n, dim]),
        Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([1, 1, n, dim]),
    )
}

/// Apply a rotary embedding to `x` `[b, heads, n, dim]` at absolute positions
/// `offset..offset + n`: `x * cos + rotate_half(x) * sin`.
pub fn apply_rotary<B: Backend>(x: Tensor<B, 4>, offset: usize, base: f64) -> Tensor<B, 4> {
    let [_, _, n, dim] = x.dims();
    let device = x.device();
    let (cos, sin) = rotary_tables::<B>(offset, n, dim, base, &device);
    let half = dim / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let rotated = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos + rotated * sin
}

/// The rotary base every layer uses.
pub const ROTARY_BASE: f64 = 10_000.0;

// ------------------------------------------------------- per-layer state --

/// Everything one attention layer keeps between decode steps, whichever mode
/// it runs in.
///
/// A dense or retrieval layer keeps every key and value; a sliding layer
/// keeps the last `window - 1`; a linear layer keeps its `(S, z)` state; a
/// learned mixture keeps both. `first_key_position` records the absolute
/// position of the oldest key still held, which is what lets a sliding
/// window forget the past without losing track of where it is.
#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    pub keys: Option<Tensor<B, 4>>,
    pub values: Option<Tensor<B, 4>>,
    pub first_key_position: usize,
    pub linear: Option<LinearState<B>>,
    /// Positions this layer has absorbed.
    pub positions: usize,
}

impl<B: Backend> Default for LayerState<B> {
    fn default() -> Self {
        Self { keys: None, values: None, first_key_position: 0, linear: None, positions: 0 }
    }
}

impl<B: Backend> LayerState<B> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keys currently held.
    pub fn len(&self) -> usize {
        self.keys.as_ref().map_or(0, |k| k.dims()[2])
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0 && self.linear.is_none()
    }

    /// Forget everything.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Append new keys and values, then keep only the last `keep` positions
    /// when a window is given.
    pub fn push_kv(&mut self, k_new: Tensor<B, 4>, v_new: Tensor<B, 4>, keep: Option<usize>) {
        let (k, v) = match (self.keys.take(), self.values.take()) {
            (Some(pk), Some(pv)) => (Tensor::cat(vec![pk, k_new], 2), Tensor::cat(vec![pv, v_new], 2)),
            _ => (k_new, v_new),
        };
        let total = k.dims()[2];
        let (k, v) = match keep {
            Some(keep) if keep < total => {
                let drop = total - keep;
                self.first_key_position += drop;
                (k.narrow(2, drop, keep), v.narrow(2, drop, keep))
            }
            _ => (k, v),
        };
        self.keys = Some(k);
        self.values = Some(v);
    }

    /// The held keys and values; panics when there are none.
    pub fn kv(&self) -> (Tensor<B, 4>, Tensor<B, 4>) {
        (
            self.keys.clone().expect("keys were pushed before being read"),
            self.values.clone().expect("values were pushed before being read"),
        )
    }

    /// Keep only the newest `keep` positions, forgetting the rest and
    /// advancing `first_key_position` past them.
    pub fn trim(&mut self, keep: usize) {
        let total = self.len();
        if keep >= total {
            return;
        }
        let drop = total - keep;
        if let (Some(k), Some(v)) = (self.keys.take(), self.values.take()) {
            self.keys = Some(k.narrow(2, drop, keep));
            self.values = Some(v.narrow(2, drop, keep));
        }
        self.first_key_position += drop;
    }

    /// Floats of tensor data held, for a memory footprint.
    pub fn resident_floats(&self) -> usize {
        let kv = self.keys.as_ref().map_or(0, |k| k.shape().num_elements())
            + self.values.as_ref().map_or(0, |v| v.shape().num_elements());
        let lin = self
            .linear
            .as_ref()
            .map_or(0, |l| l.s.shape().num_elements() + l.z.shape().num_elements());
        kv + lin
    }
}

/// Which index tensors are read: a helper so the value-read cost of retrieval
/// attention can be counted.
pub fn keys_read_per_query(mode: AttentionMode, positions: usize) -> usize {
    match mode {
        AttentionMode::Dense | AttentionMode::Learned => positions,
        AttentionMode::Sliding { window } => window.min(positions),
        AttentionMode::Retrieval { top_k } => top_k.min(positions),
        AttentionMode::Linear => 0,
    }
}

/// Convert a rank-4 Bool mask to the additive form.
pub fn additive_from_keep<B: Backend>(keep: Tensor<B, 4, Bool>) -> Tensor<B, 4> {
    let device = keep.device();
    let dims = keep.dims();
    Tensor::<B, 4>::zeros(dims, &device).mask_fill(keep.bool_not(), f32::NEG_INFINITY)
}

/// Int helper: absolute positions `offset..offset + n` as a rank-1 tensor.
pub fn positions_tensor<B: Backend>(offset: usize, n: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::arange(offset as i64..(offset + n) as i64, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{activation::softmax, Distribution};

    type B = NdArray<f32>;

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().convert::<f32>().iter::<f32>().collect()
    }

    #[test]
    fn test_schedule_parsing_covers_every_form() {
        let d = AttentionSchedule::parse("", 4, 8, 4).unwrap();
        assert!(d.is_all_dense());
        assert_eq!(AttentionSchedule::parse("dense", 3, 8, 4).unwrap().pattern(), "DDD");
        assert_eq!(AttentionSchedule::parse("3:1", 8, 8, 4).unwrap().pattern(), "LLLDLLLD");
        assert_eq!(AttentionSchedule::parse("1:1", 4, 8, 4).unwrap().pattern(), "LDLD");
        let s = AttentionSchedule::parse("2:1@sliding16", 6, 8, 4).unwrap();
        assert_eq!(s.pattern(), "SSDSSD");
        assert_eq!(s.mode(0), AttentionMode::Sliding { window: 16 });
        let list = AttentionSchedule::parse("linear,retrieval3,dense,learned", 4, 8, 4).unwrap();
        assert_eq!(list.modes[1], AttentionMode::Retrieval { top_k: 3 });
        assert_eq!(list.pattern(), "LRDM");
        assert_eq!(list.summary(), "1L 1R 1D 1M");
        assert_eq!(AttentionSchedule::parse("LLD", 5, 8, 4).unwrap().pattern(), "LLDLL");
        assert!(AttentionSchedule::parse("linear,dense", 3, 8, 4).is_err(), "wrong length");
        assert!(AttentionSchedule::parse("2:2", 4, 8, 4).is_err());
        assert!(AttentionSchedule::parse("sliding0", 4, 8, 4).is_err());
        assert!(AttentionMode::parse("hyper", 8, 4).is_err());
        let json = serde_json::to_string(&s).unwrap();
        let back: AttentionSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn test_position_kind_parses_and_reports_bounds() {
        assert_eq!(PositionKind::parse("rope").unwrap(), PositionKind::Rotary);
        assert_eq!(PositionKind::parse("learned").unwrap(), PositionKind::Learned);
        assert_eq!(PositionKind::parse("nope").unwrap(), PositionKind::None);
        assert!(PositionKind::parse("alibi").is_err());
        assert!(PositionKind::Learned.is_bounded() && !PositionKind::Rotary.is_bounded());
    }

    #[test]
    fn test_attention_mask_geometry() {
        let device = Default::default();
        // 2 new queries at positions 3, 4 over 5 keys starting at 0, window 2.
        let m = to_vec(attention_mask::<B>(2, 5, 3, 0, Some(2), &device));
        let visible: Vec<bool> = m.iter().map(|v| *v == 0.0).collect();
        // query 3 sees keys 2, 3; query 4 sees keys 3, 4.
        assert_eq!(visible, vec![false, false, true, true, false, false, false, false, true, true]);
        // No window: plain causal with an offset.
        let c = to_vec(attention_mask::<B>(2, 5, 3, 0, None, &device));
        let visible: Vec<bool> = c.iter().map(|v| *v == 0.0).collect();
        assert_eq!(visible, vec![true, true, true, true, false, true, true, true, true, true]);
        // Keys that start later than 0 (a truncated sliding cache).
        let t = to_vec(attention_mask::<B>(1, 2, 5, 4, Some(2), &device));
        assert_eq!(t.iter().map(|v| *v == 0.0).collect::<Vec<_>>(), vec![true, true]);
        let keep = to_vec(causal_keep_mask::<B>(2, 2, 0, 0, &device));
        assert_eq!(keep, vec![1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_keep_top_k_keeps_exactly_the_largest() {
        let device = Default::default();
        let row = [0.1f32, 5.0, -2.0, 3.0, f32::NEG_INFINITY];
        let scores = Tensor::<B, 1>::from_floats(row.as_slice(), &device).reshape([1, 1, 1, 5]);
        let kept = to_vec(keep_top_k(scores.clone(), 2));
        assert_eq!(kept[1], 5.0);
        assert_eq!(kept[3], 3.0);
        assert!(kept[0].is_infinite() && kept[2].is_infinite() && kept[4].is_infinite());
        // k >= width is the identity.
        assert_eq!(to_vec(keep_top_k(scores.clone(), 5)), row.to_vec());
        // A softmax over the kept entries is a distribution over exactly them.
        let p = to_vec(softmax(keep_top_k(scores, 2), 3));
        assert!((p[1] + p[3] - 1.0).abs() < 1e-6 && p[0] == 0.0 && p[2] == 0.0);
    }

    #[test]
    fn test_feature_map_is_positive_and_matches_elu_plus_one() {
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats([-3.0f32, -0.5, 0.0, 0.5, 4.0].as_slice(), &device);
        let phi = to_vec(feature_map(x));
        let expected: Vec<f32> = [-3.0f32, -0.5, 0.0, 0.5, 4.0]
            .iter()
            .map(|v| if *v > 0.0 { v + 1.0 } else { v.exp() })
            .collect();
        for (a, b) in phi.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
            assert!(*a > 0.0);
        }
    }

    #[test]
    fn test_linear_attention_state_form_matches_the_masked_form() {
        // The property the decode-time state rests on: feeding the sequence in
        // chunks through the recurrence gives the same outputs as one masked
        // matrix product over the whole sequence.
        let device = Default::default();
        let (b, h, n, d) = (2usize, 2usize, 7usize, 4usize);
        let q = feature_map(Tensor::<B, 4>::random([b, h, n, d], Distribution::Uniform(-1.0, 1.0), &device));
        let k = feature_map(Tensor::<B, 4>::random([b, h, n, d], Distribution::Uniform(-1.0, 1.0), &device));
        let v = Tensor::<B, 4>::random([b, h, n, d], Distribution::Uniform(-1.0, 1.0), &device);
        let (full, _) = linear_attention(q.clone(), k.clone(), v.clone(), None, 1e-6);
        let full = to_vec(full);

        for chunks in [vec![n], vec![1; n], vec![3, 1, 3], vec![2, 5]] {
            let mut state: Option<LinearState<B>> = None;
            let mut outputs = Vec::new();
            let mut at = 0;
            for size in chunks {
                let (out, next) = linear_attention(
                    q.clone().narrow(2, at, size),
                    k.clone().narrow(2, at, size),
                    v.clone().narrow(2, at, size),
                    state.take(),
                    1e-6,
                );
                outputs.push(out);
                state = Some(next);
                at += size;
            }
            // Reassemble along the sequence axis before flattening: the
            // chunks are `[b, h, size, d]`, and flattening them one by one
            // would interleave batch and position.
            let produced = to_vec(Tensor::cat(outputs, 2));
            assert_eq!(produced.len(), full.len());
            for (i, (a, c)) in produced.iter().zip(&full).enumerate() {
                assert!((a - c).abs() <= 1e-5 * c.abs().max(1.0), "element {i}: {a} vs {c}");
            }
        }
    }

    #[test]
    fn test_linear_attention_cannot_see_the_future() {
        let device = Default::default();
        let (n, d) = (6usize, 4usize);
        let q = feature_map(Tensor::<B, 4>::random([1, 1, n, d], Distribution::Uniform(-1.0, 1.0), &device));
        let k = feature_map(Tensor::<B, 4>::random([1, 1, n, d], Distribution::Uniform(-1.0, 1.0), &device));
        let v = Tensor::<B, 4>::random([1, 1, n, d], Distribution::Uniform(-1.0, 1.0), &device);
        let (reference, _) = linear_attention(q.clone(), k.clone(), v.clone(), None, 1e-6);
        let tail = v.clone().narrow(2, n - 1, 1) + 10.0;
        let perturbed_v = Tensor::cat(vec![v.narrow(2, 0, n - 1), tail], 2);
        let (changed, _) = linear_attention(q, k, perturbed_v, None, 1e-6);
        let prefix = (reference.clone().narrow(2, 0, n - 1) - changed.clone().narrow(2, 0, n - 1)).abs().max().into_scalar();
        assert_eq!(prefix, 0.0, "a later value leaked backwards");
        let last = (reference.narrow(2, n - 1, 1) - changed.narrow(2, n - 1, 1)).abs().max().into_scalar();
        assert!(last > 1e-4);
    }

    #[test]
    fn test_rotary_scores_depend_only_on_distance_and_preserve_norms() {
        let device = Default::default();
        let dim = 8;
        let q = Tensor::<B, 4>::random([1, 1, 1, dim], Distribution::Uniform(-1.0, 1.0), &device);
        let k = Tensor::<B, 4>::random([1, 1, 1, dim], Distribution::Uniform(-1.0, 1.0), &device);
        let score = |qo: usize, ko: usize| -> f32 {
            let rq = apply_rotary(q.clone(), qo, ROTARY_BASE);
            let rk = apply_rotary(k.clone(), ko, ROTARY_BASE);
            (rq * rk).sum().into_scalar()
        };
        let base = score(7, 3);
        for shift in [0usize, 1, 10, 100] {
            let shifted = score(7 + shift, 3 + shift);
            assert!((shifted - base).abs() < 1e-4, "shift {shift}: {shifted} vs {base}");
        }
        assert!((score(3, 7) - base).abs() > 1e-4 || dim < 2, "direction of the distance must matter");
        let norm_before: f32 = q.clone().powf_scalar(2.0).sum().into_scalar();
        let norm_after: f32 = apply_rotary(q, 13, ROTARY_BASE).powf_scalar(2.0).sum().into_scalar();
        assert!((norm_before - norm_after).abs() < 1e-5);
        // Position 0 is the identity rotation.
        let x = Tensor::<B, 4>::random([1, 2, 3, dim], Distribution::Uniform(-1.0, 1.0), &device);
        let at_zero = apply_rotary(x.clone(), 0, ROTARY_BASE).narrow(2, 0, 1);
        let drift = (at_zero - x.narrow(2, 0, 1)).abs().max().into_scalar();
        assert_eq!(drift, 0.0);
    }

    #[test]
    fn test_layer_state_window_keeps_the_newest_keys() {
        let device = Default::default();
        let mut state = LayerState::<B>::new();
        let k = |v: f32| Tensor::<B, 4>::full([1, 1, 1, 2], v, &device);
        for i in 0..5 {
            state.push_kv(k(i as f32), k(-(i as f32)), Some(2));
        }
        assert_eq!(state.len(), 2);
        assert_eq!(state.first_key_position, 3);
        let kept = to_vec(state.keys.clone().unwrap());
        assert_eq!(kept, vec![3.0, 3.0, 4.0, 4.0]);
        assert!(state.resident_floats() == 8);
        state.clear();
        assert!(state.is_empty() && state.first_key_position == 0);
        assert_eq!(keys_read_per_query(AttentionMode::Sliding { window: 4 }, 10), 4);
        assert_eq!(keys_read_per_query(AttentionMode::Retrieval { top_k: 3 }, 2), 2);
        assert_eq!(keys_read_per_query(AttentionMode::Linear, 100), 0);
        assert_eq!(keys_read_per_query(AttentionMode::Dense, 100), 100);
    }
}

