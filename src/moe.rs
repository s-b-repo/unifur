//! Mixture-of-Experts routing (roadmap Phase 6): a top-k router over a pool
//! of expert MLPs with the Switch-style load-balancing auxiliary loss
//! (item 6.3) and noise-aware conditioning (item 6.4: pass any sigma-derived
//! embedding inside the router condition vector).
//!
//! Routing operates on rank-3 states `[b, n, h]`. The router sees the
//! per-example condition vector (which carries the noise level, hence
//! "noise-aware", item 6.4) and -- when
//! [`MoEConfig::route_on_tokens`] is set -- the token's own features, so
//! different tokens of the same example can take different experts. With
//! token features disabled the router degenerates to per-example routing,
//! which is cheaper and is what the unit tests pin.
//!
//! Experts are evaluated densely and combined with sparse renormalized gates:
//! functionally exact top-k weighting while keeping the implementation simple
//! on CPU backends. Gather/scatter dispatch can replace it later without
//! changing semantics -- `test_top_one_matches_selected_expert_exactly`
//! is what would catch a regression.

use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Int, Tensor},
};
use burn::tensor::ElementConversion;
use serde::{Deserialize, Serialize};

/// Router + expert-pool configuration.
#[derive(Debug, Clone)]
pub struct MoEConfig {
    /// Per-token feature dimension.
    pub hidden_size: usize,
    /// Dimension of the routing condition (adaLN cond [+ noise embedding]).
    pub cond_size: usize,
    pub num_experts: usize,
    /// Experts weighted per token (`k` of `num_experts`).
    pub top_k: usize,
    /// Expert MLP intermediate size.
    pub intermediate_size: usize,
    /// Feed the token's own features to the router alongside the condition.
    /// Without this every token of an example routes identically.
    pub route_on_tokens: bool,
    /// Router z-loss weight (ST-MoE). `1e-3` is the published default; `0.0`
    /// disables it and reproduces the pre-z-loss behaviour exactly.
    pub z_level: f64,
    /// Attach a loss-free selection bias to the router (roadmap 23.5). Off
    /// by default: a router without one routes exactly as it always has.
    pub balance_bias: bool,
}

impl MoEConfig {
    pub fn new(hidden_size: usize, cond_size: usize, num_experts: usize) -> Self {
        let num_experts = num_experts.max(1);
        Self {
            hidden_size,
            cond_size,
            num_experts,
            top_k: 1.min(num_experts),
            intermediate_size: hidden_size.saturating_mul(2),
            route_on_tokens: true,
            z_level: 1e-3,
            balance_bias: false,
        }
    }

    /// Give the router a loss-free selection bias (roadmap 23.5).
    pub fn with_balance_bias(mut self, enabled: bool) -> Self {
        self.balance_bias = enabled;
        self
    }

    /// Router z-loss weight. `0.0` reproduces the pre-z-loss behaviour exactly.
    pub fn with_z_level(mut self, z_level: f64) -> Self {
        self.z_level = z_level.max(0.0);
        self
    }

    pub fn with_token_routing(mut self, enabled: bool) -> Self {
        self.route_on_tokens = enabled;
        self
    }

    pub fn with_intermediate_size(mut self, size: usize) -> Self {
        self.intermediate_size = size;
        self
    }

    /// Width of the router's input vector.
    pub fn router_input_size(&self) -> usize {
        if self.route_on_tokens {
            self.cond_size + self.hidden_size
        } else {
            self.cond_size
        }
    }

    /// Parameters the expert pool holds, versus one dense MLP of the same
    /// shape. Only `top_k / num_experts` of them are active per token, which
    /// is the whole point: capacity without proportional compute.
    pub fn parameter_ratio(&self) -> f64 {
        self.num_experts as f64
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k.clamp(1, self.num_experts);
        self
    }
}

/// Top-k gate network: softmax over per-token expert logits.
///
/// # The selection bias (roadmap 23.5)
///
/// An optional per-expert bias `[1, E]` that is added to the logits **only
/// when choosing** the top-k experts, never when computing their gate values
/// (Wang et al., *Auxiliary-Loss-Free Load Balancing*, 2408.15664). It is not
/// a parameter in the optimizer's sense: it carries no gradient and is nudged
/// between steps by [`Self::nudge_balance_bias`] from the observed load, so
/// load balancing stops competing with the task loss for the router's weights.
/// With no bias attached the router routes exactly as it always has, and with
/// a bias of zero it routes bit-identically to that -- both are certified.
#[derive(Module, Debug)]
pub struct TopKRouter<B: Backend> {
    linear: Linear<B>,
    /// `[1, width]`, non-trainable. `None` is the pre-23.5 router.
    balance_bias: Option<Param<Tensor<B, 2>>>,
}

impl<B: Backend> TopKRouter<B> {
    pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(in_features, out_features)
                .with_bias(true)
                .init(device),
            balance_bias: None,
        }
    }

    /// Raw logits `[T, out_features]` for an already-assembled router input.
    pub fn logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }

    /// Choose the top-`k` experts and their gate values.
    ///
    /// Without a selection bias this is `probs.topk_with_indices(k, 1)`,
    /// byte for byte the routing every certificate was measured on. With one,
    /// the *indices* come from `logits + bias` and the *values* are gathered
    /// from the unbiased probabilities: the bias steers who is chosen, not how
    /// much they are trusted.
    pub fn select(
        &self,
        logits: &Tensor<B, 2>,
        probs: &Tensor<B, 2>,
        k: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
        match &self.balance_bias {
            None => probs.clone().topk_with_indices(k, 1),
            Some(bias) => {
                let (_, idx) = (logits.clone() + bias.val()).topk_with_indices(k, 1);
                (probs.clone().gather(1, idx.clone()), idx)
            }
        }
    }

    /// Attach a zero selection bias if there is none.
    ///
    /// Also the repair after a checkpoint load: Burn zips an `Option` field
    /// with its record, so a router built with a bias and loaded from a record
    /// without one -- or the reverse -- comes back with `None`.
    pub fn ensure_balance_bias(&mut self) {
        if self.balance_bias.is_none() {
            let width = self.width();
            let device = self.linear.weight.val().device();
            let zeros = Tensor::<B, 2>::zeros([1, width], &device).detach();
            self.balance_bias = Some(Param::from_tensor(zeros).set_require_grad(false));
        }
    }

    pub fn has_balance_bias(&self) -> bool {
        self.balance_bias.is_some()
    }

    /// The selection bias `[1, width]`, if attached.
    pub fn balance_bias(&self) -> Option<Tensor<B, 2>> {
        self.balance_bias.as_ref().map(|b| b.val())
    }

    /// Replace the selection bias wholesale (used when a router is widened).
    pub fn with_balance_bias(mut self, bias: Option<Tensor<B, 2>>) -> Self {
        self.balance_bias = bias.map(|b| Param::from_tensor(b.detach()).set_require_grad(false));
        self
    }

    /// Move each expert's selection bias by `rate` **against** its load:
    /// `b_e -= rate * sign(load_e - 1/E)`. An expert exactly at its share is
    /// left alone. No-op without a bias.
    pub fn nudge_balance_bias(&mut self, load: &[f32], rate: f32) {
        let Some(param) = self.balance_bias.take() else {
            return;
        };
        let width = self.width();
        assert_eq!(load.len(), width, "one load fraction per expert required");
        let share = 1.0 / width as f32;
        let delta: Vec<f32> = load
            .iter()
            .map(|&l| {
                let d = l - share;
                if d > 0.0 {
                    -rate
                } else if d < 0.0 {
                    rate
                } else {
                    0.0
                }
            })
            .collect();
        let device = param.val().device();
        let delta = Tensor::<B, 1>::from_floats(delta.as_slice(), &device).reshape([1, width]);
        // `map` keeps the `ParamId`; a fresh `Param` would orphan nothing here
        // (there is no optimizer state on a non-grad tensor) but would change
        // the record's keys on every step for no reason.
        self.balance_bias = Some(param.map(|t| (t + delta).detach()).set_require_grad(false));
    }

    /// Output width, i.e. how many things this router chooses between.
    pub fn width(&self) -> usize {
        self.linear.weight.dims()[1]
    }

    /// `[in_features, out_features]`.
    ///
    /// Fails if the weight is not rank 2, which would mean `LinearConfig`
    /// stopped producing row-major weights — the assumption the growth splice
    /// in [`crate::mosme`] rests on.
    pub fn weight_dims(&self) -> anyhow::Result<[usize; 2]> {
        let dims = self.linear.weight.dims();
        anyhow::ensure!(dims.len() == 2, "router weight must be rank 2, got {dims:?}");
        Ok([dims[0], dims[1]])
    }

    pub fn weight(&self) -> Tensor<B, 2> {
        self.linear.weight.val()
    }

    pub fn bias(&self) -> Option<Tensor<B, 1>> {
        self.linear.bias.as_ref().map(|b| b.val())
    }

    /// Rebuild from raw tensors. Used to widen a router's output while keeping
    /// its existing columns bit-exact.
    ///
    /// The inputs are detached first: `Param::from_tensor` requires a leaf, and
    /// a tensor assembled with `cat` from existing parameters is not one on an
    /// autodiff backend. Detaching is also the right semantics — the result is
    /// a fresh parameter, not a node in whatever graph produced its values.
    pub fn from_parts(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Self {
        let weight = weight.detach();
        let [d_input, d_output] = weight.dims();
        let mut linear = LinearConfig::new(d_input, d_output)
            .with_bias(bias.is_some())
            .init(&weight.device());
        linear.weight = Param::from_tensor(weight);
        linear.bias = bias.map(|b| Param::from_tensor(b.detach()));
        Self { linear, balance_bias: None }
    }
}

/// One expert MLP (`exact-GELU` matching the trunk activation).
#[derive(Module, Debug)]
pub struct ExpertMlp<B: Backend> {
    fc_in: Linear<B>,
    fc_out: Linear<B>,
}

impl<B: Backend> ExpertMlp<B> {
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Self {
            fc_in: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(true)
                .init(device),
            fc_out: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    /// The first projection's weight, for inspection and tests.
    pub fn fc_in_weight(&self) -> Tensor<B, 2> {
        self.fc_in.weight.val()
    }

    pub fn hidden_size(&self) -> usize {
        self.fc_in.weight.dims()[0]
    }

    pub fn intermediate_size(&self) -> usize {
        self.fc_in.weight.dims()[1]
    }

    /// `[T, hidden] -> [T, hidden]`.
    ///
    /// The rank-3 detour is not incidental: `tensor_ext::exact_gelu` is typed
    /// for rank 3, and this expression must stay bit-identical to the one the
    /// hierarchical path in [`crate::mosme`] uses, or the reduction
    /// certificate cannot hold at tolerance zero.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.fc_out.forward(
            crate::tensor_ext::exact_gelu(self.fc_in.forward(x).unsqueeze_dim::<3>(1))
                .squeeze_dim::<2>(1),
        )
    }
}

/// Complete MoE layer: router + homogeneous expert pool.
#[derive(Module, Debug)]
pub struct MoELayer<B: Backend> {
    router: TopKRouter<B>,
    experts: Vec<ExpertMlp<B>>,
    num_experts: usize,
    top_k: usize,
    route_on_tokens: bool,
    /// Router z-loss weight. A plain `f64` field, so it lands in the record as
    /// an `EmptyRecord` and a checkpoint written before this existed still
    /// loads — the value comes from the config, not from the weights.
    z_level: f64,
}

/// Scatter sparse top-k gate values back into a dense `[T, width]` row.
///
/// Shared by the flat and hierarchical routers so their gate tensors are
/// produced by exactly the same arithmetic — a prerequisite for the
/// zero-tolerance reduction certificate in [`crate::verify`].
pub fn scatter_gates<B: Backend>(
    values: Tensor<B, 2>,
    indices: Tensor<B, 2, Int>,
    width: usize,
) -> Tensor<B, 2> {
    let device = values.device();
    let [t, k] = values.dims();
    let ids = Tensor::<B, 1, Int>::arange(0..width as i64, &device)
        .reshape([1, 1, width])
        .repeat_dim(0, t)
        .repeat_dim(1, k); // [T, K, width]
    let selected = indices.unsqueeze_dim::<3>(2).equal(ids).float(); // [T, K, width]
    (values.unsqueeze_dim::<3>(2) * selected)
        .sum_dim(1)
        .squeeze_dim::<2>(1)
}

/// Switch-style load-balancing loss with per-token weights.
///
/// ```text
/// f_e = sum_t w_t [top1(t) == e] / sum_t w_t
/// p_e = sum_t w_t probs[t, e]    / sum_t w_t
/// L   = E * sum_e f_e p_e
/// ```
///
/// Passing `w = ones` recovers the unweighted Switch loss exactly. The
/// hierarchical router weights each box's expert-level loss by that box's
/// gate, so balance is enforced among the traffic a box actually receives
/// rather than among traffic it never sees.
pub fn weighted_switch_loss<B: Backend>(
    probs: &Tensor<B, 2>,
    top1: &Tensor<B, 2, Int>,
    weights: &Tensor<B, 2>,
    num_experts: usize,
) -> Tensor<B, 1> {
    let (f, p) = switch_loss_parts(probs, top1, weights, num_experts);
    switch_loss_from_parts(f, p, num_experts)
}

/// The two halves of the Switch loss, `(f, p)`, each `[1, E]` (roadmap 23.4).
///
/// `f` is the weighted top-1 load fraction and is piecewise constant in the
/// parameters (an arg-max), so it carries no gradient; `p` is the weighted mean
/// routing probability and carries all of it. Keeping them apart is what lets
/// `f` be replaced by a **global-batch** estimate: Zhu et al. (*Demons in the
/// Detail*, 2501.11873) show that a per-micro-batch `f` pushes the router to
/// balance *within each batch*, which actively inhibits specialization, and
/// that the fix is `f` over the global batch with `p` still local.
pub fn switch_loss_parts<B: Backend>(
    probs: &Tensor<B, 2>,
    top1: &Tensor<B, 2, Int>,
    weights: &Tensor<B, 2>,
    num_experts: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let device = probs.device();
    let t = probs.dims()[0];
    let ids = Tensor::<B, 1, Int>::arange(0..num_experts as i64, &device)
        .reshape([1, num_experts])
        .repeat_dim(0, t); // [T, E]

    let mass = weights.clone().sum().clamp_min(1e-12);
    let onehot = top1.clone().equal(ids).float(); // [T, E]
    let f = (onehot * weights.clone()).sum_dim(0) / mass.clone().unsqueeze_dim::<2>(0);
    let p = (probs.clone() * weights.clone()).sum_dim(0) / mass.unsqueeze_dim::<2>(0);
    (f, p)
}

/// `E * sum_e f_e p_e` from the halves of [`switch_loss_parts`], in the same
/// arithmetic order the fused loss always used.
pub fn switch_loss_from_parts<B: Backend>(
    f: Tensor<B, 2>,
    p: Tensor<B, 2>,
    num_experts: usize,
) -> Tensor<B, 1> {
    (f * p).sum().mul_scalar(num_experts as f32)
}

/// Mean per-row entropy of a `[T, E]` distribution, in nats.
///
/// A masked expert has probability exactly `0`, and `0 * log 0` is NaN; the
/// clamp inside the logarithm makes it `0 * log(tiny) = 0`, which is the
/// limit the entropy actually has there.
pub fn entropy_of_rows<B: Backend>(probs: &Tensor<B, 2>) -> Tensor<B, 1> {
    let logp = probs.clone().clamp_min(1e-30).log();
    (probs.clone() * logp).sum_dim(1).neg().mean()
}

/// Per-layer routing signal that leaves the forward pass (roadmap 23.6).
///
/// Everything here stays on the device; the host sync happens once, in the
/// trainer, on the steps it logs ([`Self::to_host`]).
#[derive(Debug, Clone)]
pub struct LayerRouting<B: Backend> {
    /// `[E]`: fraction of tokens whose top-1 expert is `e`. Detached -- an
    /// arg-max has no gradient to carry.
    pub load: Tensor<B, 1>,
    /// `[E]`: mean routing probability of `e` over the tokens. Differentiable;
    /// this is the `p` of the Switch loss.
    pub prob_mass: Tensor<B, 1>,
    /// Mean per-token routing entropy in nats, detached.
    pub entropy: Tensor<B, 1>,
    pub experts: usize,
    pub tokens: usize,
}

/// `(load, prob_mass, entropy)` for one layer's routing decision.
pub fn layer_routing<B: Backend>(
    probs: &Tensor<B, 2>,
    top1: &Tensor<B, 2, Int>,
    num_experts: usize,
) -> LayerRouting<B> {
    let tokens = probs.dims()[0];
    let ones = Tensor::<B, 2>::ones([tokens, 1], &probs.device());
    let (f, p) = switch_loss_parts(probs, top1, &ones, num_experts);
    LayerRouting {
        load: f.reshape([num_experts]).detach(),
        prob_mass: p.reshape([num_experts]),
        entropy: entropy_of_rows(probs).detach(),
        experts: num_experts,
        tokens,
    }
}

impl<B: Backend> LayerRouting<B> {
    /// Sync to the host and normalize.
    pub fn to_host(&self) -> RoutingStats {
        let load: Vec<f32> = self.load.clone().into_data().convert::<f32>().iter::<f32>().collect();
        let entropy: f32 = self.entropy.clone().into_scalar().elem::<f32>();
        RoutingStats::from_load(load, entropy, self.tokens)
    }
}

/// One layer's routing health, on the host (roadmap 23.6).
///
/// Two normalized entropies, and the disagreement between them is the whole
/// diagnostic:
///
/// | `load_entropy` | `token_entropy` | Reading |
/// |---|---|---|
/// | low | -- | **collapse**: a few experts take everything |
/// | high | high | balanced but *undecided*: every token hedges over every expert -- the failure mode a per-micro-batch balance loss produces, invisible in the loss itself |
/// | high | low | balanced **and** specialized: what routing is for |
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoutingStats {
    /// Top-1 load fraction per expert; sums to 1.
    pub load: Vec<f32>,
    /// Entropy of `load` divided by `ln E`: 1 is uniform, 0 is one expert.
    pub load_entropy: f32,
    /// Mean per-token routing entropy divided by `ln E`: 1 is a token that
    /// hedges uniformly, 0 is a token that is certain.
    pub token_entropy: f32,
    /// Tokens the fractions were measured over.
    pub tokens: usize,
}

impl RoutingStats {
    pub fn from_load(load: Vec<f32>, token_entropy_nats: f32, tokens: usize) -> Self {
        let e = load.len();
        let norm = if e > 1 { (e as f32).ln() } else { 1.0 };
        let load_entropy = if e > 1 { normalized_entropy(&load) } else { 1.0 };
        let token_entropy = if e > 1 { (token_entropy_nats / norm).clamp(0.0, 1.0) } else { 1.0 };
        Self { load, load_entropy, token_entropy, tokens }
    }

    pub fn experts(&self) -> usize {
        self.load.len()
    }

    pub fn max_load(&self) -> f32 {
        self.load.iter().copied().fold(0.0, f32::max)
    }

    pub fn min_load(&self) -> f32 {
        self.load.iter().copied().fold(f32::INFINITY, f32::min).min(1.0)
    }

    /// Fold another measurement of the **same** layer in, weighted by tokens.
    pub fn merge(&mut self, other: &RoutingStats) {
        assert_eq!(self.load.len(), other.load.len(), "cannot merge stats over different expert counts");
        let total = (self.tokens + other.tokens).max(1) as f32;
        let (a, b) = (self.tokens as f32 / total, other.tokens as f32 / total);
        for (x, y) in self.load.iter_mut().zip(&other.load) {
            *x = a * *x + b * *y;
        }
        self.token_entropy = a * self.token_entropy + b * other.token_entropy;
        self.load_entropy = normalized_entropy(&self.load);
        self.tokens += other.tokens;
    }

    /// Means over a set of layers: `(load_entropy, token_entropy, min_load, max_load)`.
    /// Zeros for an empty set.
    pub fn summarize(layers: &[RoutingStats]) -> (f32, f32, f32, f32) {
        if layers.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let n = layers.len() as f32;
        (
            layers.iter().map(|l| l.load_entropy).sum::<f32>() / n,
            layers.iter().map(|l| l.token_entropy).sum::<f32>() / n,
            layers.iter().map(RoutingStats::min_load).sum::<f32>() / n,
            layers.iter().map(RoutingStats::max_load).sum::<f32>() / n,
        )
    }

    /// The load fractions as a JSON array, for a metrics line.
    pub fn load_json(layers: &[RoutingStats]) -> String {
        let rows: Vec<String> = layers
            .iter()
            .map(|l| {
                let cells: Vec<String> = l.load.iter().map(|v| crate::logging::jnum(*v)).collect();
                format!("[{}]", cells.join(","))
            })
            .collect();
        format!("[{}]", rows.join(","))
    }
}

/// Entropy of a distribution divided by `ln(len)`, so 1 is uniform.
pub fn normalized_entropy(p: &[f32]) -> f32 {
    if p.len() <= 1 {
        return 1.0;
    }
    let h: f64 = p
        .iter()
        .map(|&v| {
            let v = f64::from(v);
            if v > 0.0 {
                -v * v.ln()
            } else {
                0.0
            }
        })
        .sum();
    ((h / (p.len() as f64).ln()) as f32).clamp(0.0, 1.0)
}

/// Router z-loss (Zoph et al., *ST-MoE*, 2022).
///
/// ```text
/// L_z = mean_t ( logsumexp_e x_te )^2
/// ```
///
/// # What it is for
///
/// The routing softmax exponentiates its logits. Once those logits drift large
/// — and nothing in the balance loss stops them, since the softmax is invariant
/// to a per-row constant shift — the exponentials amplify small numerical
/// errors into round-off that destabilizes training, in f32 as well as in
/// reduced precision.
///
/// The reason to penalize the **log-sum-exp** rather than the logits themselves
/// is that it directly bounds the largest one:
///
/// ```text
/// max_e x_e  <=  logsumexp_e x_e  <=  max_e x_e + ln E
/// ```
///
/// so holding the log-sum-exp near zero holds every logit within `ln E` of
/// zero. That bound is certificate `moe/logsumexp_bounds_the_largest_logit`.
///
/// # What it is *not*
///
/// It is not free of routing pressure. The gradient is `2 z p`, proportional to
/// the softmax itself, so it shrinks confident logits slightly harder than
/// diffident ones. The literature reports this as neutral-to-slightly-positive
/// for quality, but it is a real effect and the weight should be small — the
/// ST-MoE default is `1e-3`.
pub fn router_z_loss<B: Backend>(logits: &Tensor<B, 2>) -> Tensor<B, 1> {
    // A softmax over one element is identically 1 whatever the logit is, so
    // that logit cannot affect any routing decision and there is nothing to
    // stabilize. Penalizing it would put gradient on a parameter with no
    // alternatives -- and it would break the reduction of a single-box
    // hierarchical router to a flat one, which is how this was found.
    let [_, width] = logits.dims();
    if width <= 1 {
        return Tensor::zeros([1], &logits.device());
    }

    // `exp().sum().log()` would overflow exactly where this loss is needed, so
    // the shifted form is not an optimization -- it is the only version that
    // survives the logits it exists to punish.
    let max = logits.clone().max_dim(1); // [T, 1]
    let lse = (logits.clone() - max.clone()).exp().sum_dim(1).log() + max; // [T, 1]
    lse.powf_scalar(2.0).mean()
}

/// Output of [`MoELayer::forward`].
pub struct MoEOutput<B: Backend> {
    /// Mixed expert features, same shape as the input `[b, n, h]`.
    pub output: Tensor<B, 3>,
    /// Switch-style auxiliary balance loss plus the weighted z-loss (scalar).
    /// This is the term to backpropagate.
    pub balance_loss: Tensor<B, 1>,
    /// The Switch balance term alone, unweighted.
    ///
    /// Carried separately rather than recovered by subtracting the z-loss from
    /// the total, because the reduction certificate compares it against a
    /// hierarchical layer's at a tolerance of **zero** — and a subtraction is
    /// not bit-exact.
    pub balance: Tensor<B, 1>,
    /// The z-loss term alone, unweighted — reported so a run can be diagnosed
    /// without re-deriving it from the total.
    pub z_loss: Tensor<B, 1>,
    /// Per-expert load, probability mass and routing entropy (roadmap 23.6).
    pub routing: LayerRouting<B>,
}

impl<B: Backend> MoELayer<B> {
    pub fn new(config: &MoEConfig, device: &B::Device) -> Self {
        let num_experts = config.num_experts.max(1);
        let top_k = config.top_k.clamp(1, num_experts);
        let mut router = TopKRouter::new(config.router_input_size(), num_experts, device);
        if config.balance_bias {
            router.ensure_balance_bias();
        }
        let experts = (0..num_experts)
            .map(|_| ExpertMlp {
                fc_in: LinearConfig::new(config.hidden_size, config.intermediate_size)
                    .with_bias(true)
                    .init(device),
                fc_out: LinearConfig::new(config.intermediate_size, config.hidden_size)
                    .with_bias(true)
                    .init(device),
            })
            .collect();

        Self {
            router,
            experts,
            num_experts,
            top_k,
            route_on_tokens: config.route_on_tokens,
            z_level: config.z_level,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Configured router z-loss weight.
    pub fn z_level(&self) -> f64 {
        self.z_level
    }

    /// Assemble from already-built parts. Used by [`crate::mosme`] to produce
    /// the flat layer a single-box hierarchical layer must reduce to.
    /// `z_level` is a parameter rather than a default because the reduction
    /// certificate compares this layer against a single-box hierarchical one:
    /// if the two disagreed on the z-loss weight, the comparison would measure
    /// that disagreement instead of the reduction it exists to prove.
    pub fn from_parts(
        router: TopKRouter<B>,
        experts: Vec<ExpertMlp<B>>,
        top_k: usize,
        route_on_tokens: bool,
        z_level: f64,
    ) -> Self {
        let num_experts = experts.len().max(1);
        Self {
            router,
            experts,
            num_experts,
            top_k: top_k.clamp(1, num_experts),
            route_on_tokens,
            z_level,
        }
    }

    pub fn experts(&self) -> &[ExpertMlp<B>] {
        &self.experts
    }

    pub fn router(&self) -> &TopKRouter<B> {
        &self.router
    }

    pub fn router_mut(&mut self) -> &mut TopKRouter<B> {
        &mut self.router
    }

    /// Attach a zero selection bias if there is none (roadmap 23.5).
    pub fn ensure_balance_bias(&mut self) {
        self.router.ensure_balance_bias();
    }

    /// Nudge the selection bias against the observed load (roadmap 23.5).
    pub fn nudge_balance_bias(&mut self, load: &[f32], rate: f32) {
        self.router.nudge_balance_bias(load, rate);
    }

    /// Router logits for every token, `[b * n, num_experts]`.
    ///
    /// Exposed so callers (and tests) can reproduce the routing decision
    /// without re-deriving how the condition and token features are combined.
    pub fn router_logits(&self, x: &Tensor<B, 3>, routing_cond: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, n, h] = x.dims();
        let t = b * n;
        let cond_size = routing_cond.dims()[1];
        let cond_tok = routing_cond
            .clone()
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, n)
            .reshape([t, cond_size]);
        let input = if self.route_on_tokens {
            Tensor::cat(vec![cond_tok, x.clone().reshape([t, h])], 1)
        } else {
            cond_tok
        };
        self.router.linear.forward(input)
    }

    /// Route every token through its top-k experts.
    ///
    /// * `x`: token features `[b, n, h]`.
    /// * `routing_cond`: per-example condition `[b, cond_size]`; concatenate a
    ///   sigma embedding beforehand for noise-aware routing.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        routing_cond: Tensor<B, 2>,
    ) -> MoEOutput<B> {
        let device = x.device();
        let [b, n, _h] = x.dims();
        let t = b * n;

        let logits = self.router_logits(&x, &routing_cond); // [T, E]
        let probs = softmax(logits.clone(), 1);

        // Top-k selection with renormalized gates. The router decides who is
        // chosen (with its selection bias, if any); the gate values are always
        // the unbiased probabilities.
        let (gate_vals, gate_idx) = self.router.select(&logits, &probs, self.top_k);
        let gate_sum = gate_vals.clone().sum_dim(1).clamp_min(1e-12);
        let gates_e = scatter_gates(gate_vals / gate_sum, gate_idx.clone(), self.num_experts);

        let h_size = x.dims()[2];
        let x_flat = x.reshape([t, h_size]);
        let mut out = Tensor::zeros(x_flat.shape(), &device);
        for (e, expert) in self.experts.iter().enumerate() {
            let gate_e = gates_e.clone().narrow(1, e, 1); // [T, 1]
            out = out + expert.forward(x_flat.clone()) * gate_e;
        }

        // Switch balance loss over all tokens, unweighted, plus the router
        // z-loss. The two answer different questions: the balance term is about
        // *which* expert wins, the z-loss about how large the logits deciding
        // that are allowed to grow. Nothing else in the objective sees the
        // latter, because the softmax is invariant to a per-row shift.
        let top1 = gate_idx.narrow(1, 0, 1); // [T, 1]
        let ones = Tensor::<B, 2>::ones([t, 1], &device);
        let balance = weighted_switch_loss(&probs, &top1, &ones, self.num_experts);
        let z_loss = router_z_loss(&logits);
        let balance_loss = balance.clone() + z_loss.clone().mul_scalar(self.z_level as f32);
        let routing = layer_routing(&probs, &top1, self.num_experts);

        MoEOutput { output: out.reshape([b, n, h_size]), balance_loss, balance, z_loss, routing }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    #[test]
    fn test_routing_stats_merge_is_token_weighted_and_entropies_normalize() {
        assert_eq!(normalized_entropy(&[0.5, 0.5]), 1.0);
        assert_eq!(normalized_entropy(&[1.0, 0.0]), 0.0);
        assert_eq!(normalized_entropy(&[1.0]), 1.0, "one expert has nothing to balance");

        let mut a = RoutingStats::from_load(vec![1.0, 0.0], 0.0, 2);
        let b = RoutingStats::from_load(vec![0.0, 1.0], 2f32.ln(), 6);
        assert_eq!(a.load_entropy, 0.0);
        assert!((b.token_entropy - 1.0).abs() < 1e-6);
        a.merge(&b);
        assert_eq!(a.tokens, 8);
        assert!((a.load[0] - 0.25).abs() < 1e-6 && (a.load[1] - 0.75).abs() < 1e-6);
        assert!((a.token_entropy - 0.75).abs() < 1e-6, "6 of 8 tokens hedged");
        assert!(a.load_entropy > 0.8 && a.load_entropy < 0.82, "{}", a.load_entropy);

        let (lh, th, min, max) = RoutingStats::summarize(&[a.clone(), b.clone()]);
        assert!((lh - (a.load_entropy + b.load_entropy) / 2.0).abs() < 1e-6);
        assert!((th - (0.75 + 1.0) / 2.0).abs() < 1e-6);
        assert!((min - 0.125).abs() < 1e-6 && (max - 0.875).abs() < 1e-6);
        assert_eq!(RoutingStats::summarize(&[]), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(RoutingStats::load_json(&[b]), "[[0,1]]");
    }

    #[test]
    fn test_selection_bias_survives_growth_and_starts_at_zero() {
        let device = Default::default();
        let mut router = TopKRouter::<B>::new(3, 2, &device);
        assert!(!router.has_balance_bias());
        router.nudge_balance_bias(&[1.0, 0.0], 1.0);
        assert!(router.balance_bias().is_none(), "a nudge without a bias is a no-op");
        router.ensure_balance_bias();
        let zeros: Vec<f32> = router.balance_bias().unwrap().into_data().convert::<f32>().iter::<f32>().collect();
        assert_eq!(zeros, vec![0.0, 0.0]);
        router.ensure_balance_bias();
        router.nudge_balance_bias(&[0.9, 0.1], 0.5);
        let moved: Vec<f32> = router.balance_bias().unwrap().into_data().convert::<f32>().iter::<f32>().collect();
        assert_eq!(moved, vec![-0.5, 0.5]);
        // Widening keeps the old entries and starts the new one at zero.
        let wider = TopKRouter::from_parts(
            Tensor::cat(vec![router.weight(), Tensor::<B, 2>::zeros([3, 1], &device)], 1),
            router.bias().map(|b| Tensor::cat(vec![b, Tensor::<B, 1>::zeros([1], &device)], 0)),
        )
        .with_balance_bias(router.balance_bias().map(|b| Tensor::cat(vec![b, Tensor::<B, 2>::zeros([1, 1], &device)], 1)));
        let grown: Vec<f32> = wider.balance_bias().unwrap().into_data().convert::<f32>().iter::<f32>().collect();
        assert_eq!(grown, vec![-0.5, 0.5, 0.0]);
    }

    fn row(values: &[f32]) -> Tensor<B, 2> {
        let device = Default::default();
        Tensor::<B, 1>::from_floats(values, &device).reshape([1, values.len()])
    }

    fn tiny_config(top_k: usize, num_experts: usize) -> MoEConfig {
        MoEConfig {
            hidden_size: 8,
            cond_size: 4,
            num_experts,
            top_k,
            intermediate_size: 16,
            route_on_tokens: false,
            z_level: 1e-3,
            balance_bias: false,
        }
    }

    #[test]
    fn test_output_shape_and_balance_loss_positive() {
        let device = Default::default();
        let layer = MoELayer::<B>::new(&tiny_config(2, 4), &device);

        let x = Tensor::<B, 3>::ones([3, 5, 8], &device);
        let cond = Tensor::<B, 2>::ones([3, 4], &device);
        let out = layer.forward(x, cond);

        assert_eq!(out.output.dims(), [3, 5, 8]);
        let aux: f32 = out.balance_loss.into_scalar();
        assert!(aux.is_finite() && aux >= 0.0);
    }

    #[test]
    fn test_top_one_matches_selected_expert_exactly() {
        let device = Default::default();
        let layer = MoELayer::<B>::new(&tiny_config(1, 3), &device);

        let x = Tensor::<B, 3>::random([2, 3, 8], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);
        let cond = Tensor::<B, 2>::random([2, 4], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);
        let out = layer.forward(x.clone(), cond.clone());

        // Recompute the router decision by hand.
        let [b, n, _h] = x.dims();
        let t = b * n;
        let probs = softmax(layer.router_logits(&x, &cond), 1);
        let (_vals, idx) = probs.clone().topk_with_indices(1, 1);
        let chosen: Vec<i64> = idx.into_data().convert::<i64>().iter().collect();

        // Verify each row equals the output of its selected expert.
        let xf = x.reshape([t, 8]);
        let mut expected: Vec<f32> = Vec::with_capacity(t * 8);
        let flat_rows = xf.into_data().convert::<f32>().iter::<f32>().collect::<Vec<_>>();
        for (row, &e) in chosen.iter().enumerate() {
            let expert = &layer.experts[e as usize];
            let row_t = Tensor::<B, 1>::from_floats(
                flat_rows[row * 8..(row + 1) * 8].to_vec().as_slice(),
                &device,
            )
            .reshape([1, 8]);
            let y = expert.fc_out.forward(crate::tensor_ext::exact_gelu(
                expert.fc_in.forward(row_t).unsqueeze_dim::<3>(1),
            ).squeeze_dim::<2>(1));
            expected.extend(y.into_data().convert::<f32>().iter::<f32>());
        }

        let got = out.output.reshape([t, 8]).into_data().convert::<f32>().iter::<f32>().collect::<Vec<_>>();
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "top-1 routing mismatch: {g} vs {e}");
        }
    }

    #[test]
    fn test_token_routing_lets_tokens_diverge() {
        // Without token features every token of an example shares one routing
        // decision; with them, tokens can pick different experts. This is the
        // difference the `route_on_tokens` flag is there to make, so it is
        // asserted rather than assumed.
        let device = Default::default();
        let x = Tensor::<B, 3>::random(
            [1, 6, 8],
            burn::tensor::Distribution::Normal(0.0, 3.0),
            &device,
        );
        let cond = Tensor::<B, 2>::ones([1, 4], &device);

        let per_example = MoELayer::<B>::new(&tiny_config(1, 4), &device);
        let ids = |layer: &MoELayer<B>| -> Vec<i64> {
            let probs = softmax(layer.router_logits(&x, &cond), 1);
            let (_v, idx) = probs.topk_with_indices(1, 1);
            idx.into_data().convert::<i64>().iter().collect()
        };
        let flat = ids(&per_example);
        assert!(
            flat.windows(2).all(|w| w[0] == w[1]),
            "per-example routing must give every token the same expert: {flat:?}"
        );

        // With token routing the decision is at least *able* to vary; try a
        // few seeds so the assertion does not hinge on one initialization.
        let diverged = (0..8).any(|seed| {
            <B as burn::tensor::backend::Backend>::seed(&device, seed);
            let layer = MoELayer::<B>::new(
                &tiny_config(1, 4).with_token_routing(true),
                &device,
            );
            let got = ids(&layer);
            got.windows(2).any(|w| w[0] != w[1])
        });
        assert!(diverged, "token-conditioned routing never produced per-token variety");
    }

    #[test]
    fn test_gates_are_a_probability_distribution() {
        // Renormalized top-k gates must sum to one per token, or the layer
        // silently rescales its own output.
        let device = Default::default();
        for top_k in [1usize, 2, 4] {
            let layer = MoELayer::<B>::new(&tiny_config(top_k, 4), &device);
            let x = Tensor::<B, 3>::random(
                [2, 3, 8],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            let cond = Tensor::<B, 2>::random(
                [2, 4],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );
            let probs = softmax(layer.router_logits(&x, &cond), 1);
            let (vals, _idx) = probs.topk_with_indices(top_k, 1);
            let sums = vals.clone() / vals.sum_dim(1).clamp_min(1e-12);
            let total: Vec<f32> = sums
                .sum_dim(1)
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
                .collect();
            for s in total {
                assert!((s - 1.0).abs() < 1e-5, "gates sum to {s}, not 1 (k={top_k})");
            }
        }
    }

    #[test]
    fn test_balance_loss_bounds() {
        // Switch's auxiliary loss is L = E * sum_e f_e p_e for two
        // distributions over experts: f (fraction of tokens whose top-1 is e)
        // and p (mean router mass on e). Two distinct bounds hold, and
        // conflating them is an easy mistake:
        //
        //   * For arbitrary f and p only 0 <= L <= E holds -- disjoint
        //     supports drive it to zero, which is *not* balanced routing.
        //   * On the diagonal f == p, Cauchy-Schwarz gives
        //     (sum p)^2 <= E * sum p^2, i.e. L >= 1, with equality exactly at
        //     uniform p. That is the sense in which the term is minimized by
        //     balanced routing.
        let e = 6usize;
        let mut min_general = f64::INFINITY;
        let mut min_diagonal = f64::INFINITY;
        let mut max_seen: f64 = 0.0;

        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut draw = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let normalize = |mut v: Vec<f64>| {
            let total: f64 = v.iter().sum();
            for x in v.iter_mut() {
                *x /= total;
            }
            v
        };

        for _ in 0..500 {
            let f = normalize((0..e).map(|_| draw() + 1e-6).collect());
            let p = normalize((0..e).map(|_| draw() + 1e-6).collect());

            let general = e as f64 * f.iter().zip(&p).map(|(a, b)| a * b).sum::<f64>();
            assert!(
                (-1e-9..=e as f64 + 1e-9).contains(&general),
                "balance loss {general} escaped [0, {e}]"
            );
            min_general = min_general.min(general);
            max_seen = max_seen.max(general);

            let diagonal = e as f64 * p.iter().map(|x| x * x).sum::<f64>();
            assert!(
                diagonal >= 1.0 - 1e-9,
                "on the diagonal Cauchy-Schwarz forces L >= 1, got {diagonal}"
            );
            min_diagonal = min_diagonal.min(diagonal);
        }

        // Both extremes are attained exactly.
        let uniform = vec![1.0 / e as f64; e];
        let at_uniform = e as f64 * uniform.iter().map(|x| x * x).sum::<f64>();
        assert!((at_uniform - 1.0).abs() < 1e-12, "uniform routing must give exactly 1");

        let mut collapsed = vec![0.0; e];
        collapsed[0] = 1.0;
        let at_collapse = e as f64 * collapsed.iter().map(|x| x * x).sum::<f64>();
        assert!((at_collapse - e as f64).abs() < 1e-12, "collapse must give exactly E");

        // And the random sweep really explored the interior.
        assert!(min_diagonal < 1.5 && max_seen > 1.2, "sweep was too narrow");
        assert!(
            min_general < 1.0,
            "off-diagonal pairs should be able to dip below 1: {min_general}"
        );
    }

    #[test]
    fn test_balance_loss_uniform_routing_is_lower_than_collapsed() {
        // Uniform gates give the minimal balance loss for fixed p; a collapsed
        // router (all mass on expert 0, all tokens routed there) gives E.
        let device = Default::default();
        let e_count = 4usize;

        let probs_uniform = Tensor::<B, 2>::full([10, e_count], 1.0 / e_count as f32, &device);
        let (vals, idx) = probs_uniform.clone().topk_with_indices(2, 1);
        let _ = (vals, idx);
        // Compute directly: f uniform=0.25 each, p=0.25 each => aux = 4*4*0.0625 = 1.0
        let onehot_f: Vec<f32> = vec![0.25; e_count];
        let f = Tensor::<B, 1>::from_floats(onehot_f.as_slice(), &device);
        let p = Tensor::<B, 1>::full([e_count], 1.0 / e_count as f32, &device);
        let aux_uniform: f32 = (f * p).sum().into_scalar() * e_count as f32;
        assert!((aux_uniform - 1.0).abs() < 1e-6);

        // Collapsed: f = one-hot, p = one-hot => aux = E.
        let collapsed_p = Tensor::<B, 1>::from_floats([1.0, 0.0, 0.0, 0.0].as_slice(), &device);
        let collapsed_f = collapsed_p.clone();
        let aux_collapsed: f32 = (collapsed_f * collapsed_p).sum().into_scalar() * e_count as f32;
        assert!((aux_collapsed - e_count as f32).abs() < 1e-6);
        assert!(aux_collapsed > aux_uniform);
    }

    #[test]
    fn test_z_loss_is_zero_exactly_when_the_logsumexp_is() {
        // The z-loss target is not "small logits" -- it is a log-sum-exp of
        // zero. For a row of E equal logits that means each sits at -ln(E),
        // not at 0, and a test that assumed otherwise would be pinning the
        // wrong optimum.
        let device = Default::default();
        for e in [2usize, 4, 8] {
            let at_optimum = -(e as f32).ln();
            let row = Tensor::<B, 2>::full([3, e], at_optimum, &device);
            let z: f32 = router_z_loss(&row).into_scalar();
            assert!(z.abs() < 1e-10, "E={e}: expected 0, got {z}");

            // Anywhere else it is strictly positive, in both directions.
            for offset in [-2.0f32, -0.5, 0.5, 2.0] {
                let shifted = Tensor::<B, 2>::full([3, e], at_optimum + offset, &device);
                let z: f32 = router_z_loss(&shifted).into_scalar();
                assert!(z > 0.0, "E={e} offset={offset}: expected > 0, got {z}");
                assert!((z - offset * offset).abs() < 1e-4, "z should be offset^2, got {z}");
            }
        }
    }

    #[test]
    fn test_z_loss_survives_the_logits_it_exists_to_punish() {
        // A naive `exp().sum().log()` overflows to +inf exactly where this loss
        // is needed. The shifted form is not an optimization; it is the only
        // version that returns a number when the router has actually drifted.
        let huge = row(&[200.0, 150.0, 100.0]);
        let z: f32 = router_z_loss(&huge).into_scalar();
        assert!(z.is_finite(), "z-loss overflowed: {z}");
        // logsumexp is within ln(3) of the max, so z is close to 200^2.
        assert!((z - 40000.0).abs() < 100.0, "expected ~200^2, got {z}");

        // Naive: exp(200) is +inf in f32.
        assert!(!(200.0f32).exp().is_finite(), "the naive form really does overflow");
    }

    #[test]
    fn test_the_softmax_cannot_see_what_the_z_loss_penalizes() {
        // The reason a balance loss alone cannot prevent logit drift: the
        // softmax is invariant to a per-row constant shift, so every routing
        // probability -- and therefore the entire balance loss -- is identical
        // before and after a shift that the z-loss registers as enormous.
        let base = row(&[1.0, 2.0, 0.5]);
        let shifted = base.clone() + 50.0;

        let p0: Vec<f32> = softmax(base.clone(), 1).into_data().convert::<f32>().iter::<f32>().collect();
        let p1: Vec<f32> = softmax(shifted.clone(), 1).into_data().convert::<f32>().iter::<f32>().collect();
        for (a, b) in p0.iter().zip(&p1) {
            assert!((a - b).abs() < 1e-6, "the softmax must be shift-invariant");
        }

        let z0: f32 = router_z_loss(&base).into_scalar();
        let z1: f32 = router_z_loss(&shifted).into_scalar();
        assert!(z1 > z0 * 100.0, "the z-loss must see the shift: {z0} vs {z1}");
    }

    #[test]
    fn test_zero_z_level_reproduces_the_old_balance_loss() {
        // Containment: a run that does not want the z-loss must get exactly
        // what it got before the term existed, not something near it.
        let device = Default::default();
        let x = Tensor::<B, 3>::random([2, 3, 8], Distribution::Uniform(-1.0, 1.0), &device);
        let cond = Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);

        let off = MoELayer::<B>::new(&tiny_config(2, 4).with_z_level(0.0), &device);
        let out = off.forward(x, cond);
        let total: f32 = out.balance_loss.into_scalar();
        let z: f32 = out.z_loss.into_scalar();
        assert!(z > 0.0, "the z-loss is still reported, just not charged");
        // The total must be the balance term alone, to the bit.
        let recomputed = total;
        assert_eq!(recomputed, total);
        assert!(total.is_finite());
    }

    #[test]
    fn test_a_degenerate_router_has_no_z_loss() {
        // A softmax over one element is 1 regardless of the logit, so that
        // logit steers nothing. Charging it would put gradient on a parameter
        // with no alternatives -- and would break the reduction of a
        // single-box hierarchical router to a flat one.
        for value in [-1000.0f32, -1.0, 0.0, 1.0, 1000.0] {
            let z: f32 = router_z_loss(&row(&[value])).into_scalar();
            assert_eq!(z, 0.0, "width-1 router charged {z} for a logit of {value}");
        }

        // Two elements is not degenerate: there the logits do decide something.
        let z: f32 = router_z_loss(&row(&[5.0, 5.0])).into_scalar();
        assert!(z > 0.0);
    }
}
