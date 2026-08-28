//! Mixture of Specialized Micro Experts (roadmap Phase 18).
//!
//! Experts are grouped into **boxes** — a `coding` box holding a Rust expert, a
//! Python expert and a secure-code expert; a `cybersecurity` box holding its
//! own specialists — and routing is two-level: a box router picks domains, then
//! a per-box expert router picks specialists within them. The catalogue lives
//! in [`crate::expert_index`], separately from the weights, so an inference
//! engine can decide *what to load* before loading anything.
//!
//! # Why there is no single `Expert` trait
//!
//! A specialist can be three different things, and they are not variants of one
//! idea — they sit at different levels of the architecture and mix in different
//! spaces:
//!
//! | Kind | Signature | Mixed in | Valid because |
//! |---|---|---|---|
//! | MLP sub-layer | `[T, h] -> [T, h]` | feature space | the result feeds a residual add |
//! | LoRA adapter | delta on one `Linear` | weight-delta space | branches are linear in `x`, so `sum_e g_e x A_e B_e = x (sum_e g_e dW_e)` |
//! | Micro-model | `(pixels, zt, sigmas) -> logits` | probability space | a convex combination of distributions is a distribution |
//!
//! A common trait would have to accept the union of three input shapes and
//! return the union of three output shapes — a sum type pretending to be a
//! product, with an `expect()` at every call site. Burn also removes the usual
//! escape hatch: `Module` is not implemented for `Box<T>`, so `Box<dyn Trait>`
//! is unavailable.
//!
//! **What is genuinely shared is the router and the index, not the expert.** So
//! there is one [`HierarchicalRouter`], one balance loss and one manifest,
//! applied at three *sites*. A box is homogeneous in kind, and `kind` is
//! therefore a property of the site rather than of the individual expert. That
//! costs nothing — the user's own example ("coding: Rust, Python, secure-code")
//! is naturally homogeneous — and it avoids the union-type mess entirely.
//!
//! # Disabled experts are masked, not removed
//!
//! An expert can be switched off by adding `-inf` to its router logit.
//! `exp(-inf - max) == 0.0` exactly, so its gate is *exactly* zero and every
//! other gate in the box is bit-identical to what it was before. That is what
//! makes growing a model with a new expert a provable no-op until the expert is
//! deliberately enabled, and it is why the mask is a `Param<Tensor>` rather
//! than a `Vec<bool>`: Burn gives `bool` fields of a module an `EmptyRecord`,
//! so a `Vec<bool>` would silently fail to survive a checkpoint round trip.

use burn::{
    module::{Module, Param},
    nn::Linear,
    tensor::{activation::softmax, backend::Backend, Int, Tensor},
};

use crate::{
    dblock::DblockClassifier,
    quantize::{LoraAdapter, LoraConfig},
    expert_index::{
        BalanceWeights, BoxEntry, BoxLayout, ExpertEntry, ExpertIndex, ExpertKind, MosmeSpec,
        RoutingSpec, SiteKind, WeightLocator,
    },
    moe::{scatter_gates, weighted_switch_loss, ExpertMlp, MoEConfig, MoELayer, TopKRouter},
};

/// Configuration of a hierarchical expert site.
#[derive(Debug, Clone)]
pub struct MosmeConfig {
    pub hidden_size: usize,
    pub cond_size: usize,
    pub intermediate_size: usize,
    pub route_on_tokens: bool,
    pub spec: MosmeSpec,
}

impl MosmeConfig {
    pub fn new(hidden_size: usize, cond_size: usize, spec: MosmeSpec) -> Self {
        Self {
            hidden_size,
            cond_size,
            intermediate_size: hidden_size.saturating_mul(2),
            route_on_tokens: spec.route_on_tokens,
            spec,
        }
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

    pub fn total_experts(&self) -> usize {
        self.spec.num_experts()
    }

    pub fn balance(&self) -> BalanceWeights {
        self.spec.balance
    }

    /// The flat [`MoEConfig`] this reduces to when there is exactly one box.
    ///
    /// `None` for a multi-box config. Used by the reduction certificate, which
    /// asserts the hierarchical path is a strict generalization rather than a
    /// parallel implementation.
    pub fn as_flat_moe(&self) -> Option<MoEConfig> {
        if self.spec.boxes.len() != 1 {
            return None;
        }
        Some(
            MoEConfig::new(self.hidden_size, self.cond_size, self.spec.boxes[0].experts.len())
                .with_top_k(self.spec.top_expert)
                .with_intermediate_size(self.intermediate_size)
                .with_token_routing(self.route_on_tokens),
        )
    }
}

/// Everything the gates and the balance loss need, kept per level so a
/// certificate can bound each one independently rather than only the total.
#[derive(Debug, Clone)]
pub struct HierarchicalGates<B: Backend> {
    /// `[T, K]`, sparse and renormalized; each row sums to 1.
    pub box_gates: Tensor<B, 2>,
    /// `[T, K]`, the dense box softmax before top-k.
    pub box_probs: Tensor<B, 2>,
    /// `[T, K]`, the raw box-router logits.
    ///
    /// Kept alongside the probabilities because the z-loss needs the *scale*
    /// the softmax throws away: `softmax(x) == softmax(x + c)`, so a router can
    /// drift arbitrarily large while every probability stays put.
    pub box_logits: Tensor<B, 2>,
    /// `[T, 1]`, top-1 box per token.
    pub top_box_idx: Tensor<B, 2, Int>,
    /// Per box `[T, E_i]`, sparse and renormalized; each row sums to 1.
    pub expert_gates: Vec<Tensor<B, 2>>,
    /// Per box `[T, E_i]`, the dense post-mask softmax before top-k.
    pub expert_probs: Vec<Tensor<B, 2>>,
    /// Per box `[T, E_i]`, the post-mask logits.
    pub expert_logits: Vec<Tensor<B, 2>>,
    /// Per box `[T, 1]`, top-1 expert within that box.
    pub top_expert_idx: Vec<Tensor<B, 2, Int>>,
}

impl<B: Backend> HierarchicalGates<B> {
    /// `g(i, j) = box_gates[:, i] * expert_gates[i][:, j]`, per box `[T, E_i]`.
    pub fn composed(&self) -> Vec<Tensor<B, 2>> {
        self.expert_gates
            .iter()
            .enumerate()
            .map(|(i, g)| self.box_gates.clone().narrow(1, i, 1) * g.clone())
            .collect()
    }

    /// The same, flattened over the global expert index: `[T, sum_i E_i]`.
    pub fn composed_flat(&self) -> Tensor<B, 2> {
        Tensor::cat(self.composed(), 1)
    }

    /// Share of traffic each box received, `[1, K]`. Sums to 1 because each
    /// row of `box_gates` does.
    pub fn box_traffic(&self) -> Tensor<B, 2> {
        self.box_gates.clone().mean_dim(0)
    }

    /// Hierarchical load balancing.
    ///
    /// The box level is the plain Switch loss over boxes. The expert level is
    /// computed **per box, weighted by that box's gate**, then combined as a
    /// convex sum with the box traffic shares. Weighting matters: balancing
    /// experts uniformly over all tokens would push balance pressure into
    /// boxes the router never selects, which is the wrong gradient — a box
    /// that receives no traffic should contribute nothing.
    pub fn balance_loss(&self, weights: BalanceWeights) -> BalanceBreakdown<B> {
        let device = self.box_gates.device();
        let [t, num_boxes] = self.box_gates.dims();

        let ones = Tensor::<B, 2>::ones([t, 1], &device);
        let box_loss = weighted_switch_loss(&self.box_probs, &self.top_box_idx, &ones, num_boxes);

        let traffic = self.box_traffic(); // [1, K]
        let mut per_box = Vec::with_capacity(num_boxes);
        let mut expert_loss: Option<Tensor<B, 1>> = None;
        for i in 0..num_boxes {
            let gate = self.box_gates.clone().narrow(1, i, 1); // [T, 1]
            let width = self.expert_probs[i].dims()[1];
            let l = weighted_switch_loss(
                &self.expert_probs[i],
                &self.top_expert_idx[i],
                &gate,
                width,
            );
            let share = traffic.clone().narrow(1, i, 1).reshape([1]);
            let term = l.clone() * share;
            expert_loss = Some(match expert_loss {
                None => term,
                Some(acc) => acc + term,
            });
            per_box.push(l);
        }
        let expert_loss = expert_loss.unwrap_or_else(|| Tensor::zeros([1], &device));

        // Router z-loss, both levels. The expert level is weighted by box
        // traffic for the same reason the balance term is: a box the router
        // never selects should contribute no pressure, and its logits are not
        // the ones that will overflow.
        let mut z_loss = crate::moe::router_z_loss(&self.box_logits);
        for i in 0..num_boxes {
            let share = traffic.clone().narrow(1, i, 1).reshape([1]);
            z_loss = z_loss + crate::moe::router_z_loss(&self.expert_logits[i]) * share;
        }

        // `total` is the *balance* loss only. The z-loss is reported alongside
        // and weighted by the caller: it is a numerical stabilizer, not a
        // routing regularizer, and multiplying it by the balance weight (and
        // decaying it with a balance schedule) is exactly the mistake this
        // separation exists to prevent.
        let total = box_loss.clone().mul_scalar(weights.box_level as f32)
            + expert_loss.clone().mul_scalar(weights.expert_level as f32);
        BalanceBreakdown { box_loss, expert_loss, per_box, z_loss, total }
    }
}

/// Balance loss, kept per level so each can be bounded independently.
#[derive(Debug, Clone)]
pub struct BalanceBreakdown<B: Backend> {
    pub box_loss: Tensor<B, 1>,
    pub expert_loss: Tensor<B, 1>,
    pub per_box: Vec<Tensor<B, 1>>,
    /// Router z-loss, summed over both routing levels. **Unweighted** — see
    /// [`crate::vit::RouterAux`] for why it is not folded into `total`.
    pub z_loss: Tensor<B, 1>,
    /// The weighted balance loss: `w_box * box_loss + w_expert * expert_loss`.
    /// Does **not** include the z-loss.
    pub total: Tensor<B, 1>,
}

/// Two-level router: boxes, then experts within them.
#[derive(Module, Debug)]
pub struct HierarchicalRouter<B: Backend> {
    box_router: TopKRouter<B>,
    expert_routers: Vec<TopKRouter<B>>,
    /// Per box `[1, E_i]`: `0.0` for an enabled expert, `-inf` for a disabled
    /// one. Non-trainable, but a real tensor so it persists.
    masks: Vec<Param<Tensor<B, 2>>>,
    top_box: usize,
    top_expert: usize,
    route_on_tokens: bool,
}

impl<B: Backend> HierarchicalRouter<B> {
    pub fn new(config: &MosmeConfig, device: &B::Device) -> Self {
        let layout = config.spec.layout();
        let input = config.router_input_size();
        let num_boxes = layout.num_boxes();

        let expert_routers = (0..num_boxes)
            .map(|i| TopKRouter::new(input, layout.experts_in(i), device))
            .collect();
        let masks = (0..num_boxes)
            .map(|i| mask_param(&layout, i, device))
            .collect();

        Self {
            box_router: TopKRouter::new(input, num_boxes, device),
            expert_routers,
            masks,
            top_box: config.spec.top_box.max(1),
            top_expert: config.spec.top_expert.max(1),
            route_on_tokens: config.route_on_tokens,
        }
    }

    pub fn num_boxes(&self) -> usize {
        self.expert_routers.len()
    }

    pub fn experts_in(&self, box_idx: usize) -> usize {
        self.expert_routers.get(box_idx).map_or(0, TopKRouter::width)
    }

    pub fn experts_per_box(&self) -> Vec<usize> {
        self.expert_routers.iter().map(TopKRouter::width).collect()
    }

    pub fn total_experts(&self) -> usize {
        self.expert_routers.iter().map(TopKRouter::width).sum()
    }

    pub fn top_box(&self) -> usize {
        self.top_box
    }

    pub fn top_expert(&self) -> usize {
        self.top_expert
    }

    /// Read the enabled flags back out of the masks.
    pub fn layout(&self) -> BoxLayout {
        BoxLayout::new(
            self.masks
                .iter()
                .map(|m| {
                    m.val()
                        .into_data()
                        .convert::<f32>()
                        .iter::<f32>()
                        .map(|v| v == 0.0)
                        .collect()
                })
                .collect(),
        )
    }

    /// Assemble the router input from rank-3 token features, exactly as
    /// [`MoELayer::router_logits`] does.
    pub fn router_input(&self, x: &Tensor<B, 3>, cond: &Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, n, h] = x.dims();
        self.router_input_2d(&x.clone().reshape([b * n, h]), &broadcast_cond(cond, n))
    }

    /// Assemble the router input from already-flat features `[T, h]` and a
    /// matching condition `[T, cond]`.
    pub fn router_input_2d(&self, x: &Tensor<B, 2>, cond: &Tensor<B, 2>) -> Tensor<B, 2> {
        if self.route_on_tokens {
            Tensor::cat(vec![cond.clone(), x.clone()], 1)
        } else {
            cond.clone()
        }
    }

    /// Two-level routing decision for an assembled input `[T, R]`.
    ///
    /// Every box's expert router runs for every token, following the same
    /// dense-evaluation convention as [`MoELayer`]: the box gate is exactly
    /// zero for unselected boxes, so their contribution vanishes, the autodiff
    /// graph stays static, and the reduction to the flat path is exact rather
    /// than approximate.
    pub fn route(&self, input: Tensor<B, 2>) -> HierarchicalGates<B> {
        let num_boxes = self.num_boxes();

        let box_logits = self.box_router.logits(input.clone()); // [T, K]
        let box_probs = softmax(box_logits.clone(), 1);
        let kb = self.top_box.clamp(1, num_boxes);
        let (box_vals, box_idx) = box_probs.clone().topk_with_indices(kb, 1);
        let box_sum = box_vals.clone().sum_dim(1).clamp_min(1e-12);
        let box_gates = scatter_gates(box_vals / box_sum, box_idx.clone(), num_boxes);
        let top_box_idx = box_idx.narrow(1, 0, 1);

        let mut expert_gates = Vec::with_capacity(num_boxes);
        let mut expert_probs = Vec::with_capacity(num_boxes);
        let mut expert_logits = Vec::with_capacity(num_boxes);
        let mut top_expert_idx = Vec::with_capacity(num_boxes);

        for i in 0..num_boxes {
            let width = self.expert_routers[i].width();
            // Adding the mask before the softmax is what makes a disabled
            // expert's gate exactly zero rather than merely small.
            let logits = self.expert_routers[i].logits(input.clone()) + self.masks[i].val();
            let probs = softmax(logits.clone(), 1); // [T, E_i]

            // Clamping to the full width is safe even when some experts are
            // disabled: a masked entry contributes exactly 0.0 to both the
            // top-k values and their sum, so the renormalization is over the
            // enabled entries either way. That avoids a host sync to count
            // them on every forward pass.
            let ke = self.top_expert.clamp(1, width);
            let (vals, idx) = probs.clone().topk_with_indices(ke, 1);
            let sum = vals.clone().sum_dim(1).clamp_min(1e-12);

            expert_gates.push(scatter_gates(vals / sum, idx.clone(), width));
            top_expert_idx.push(idx.narrow(1, 0, 1));
            expert_probs.push(probs);
            expert_logits.push(logits);
        }

        HierarchicalGates {
            box_gates,
            box_probs,
            box_logits,
            top_box_idx,
            expert_gates,
            expert_probs,
            expert_logits,
            top_expert_idx,
        }
    }

    /// Enable or disable one expert by rewriting its mask entry.
    pub fn set_enabled(
        &mut self,
        box_idx: usize,
        expert_idx: usize,
        enabled: bool,
    ) -> anyhow::Result<()> {
        let mut layout = self.layout();
        if box_idx >= layout.num_boxes() || expert_idx >= layout.experts_in(box_idx) {
            anyhow::bail!("no expert at box {box_idx}, index {expert_idx}");
        }
        let mut flags: Vec<Vec<bool>> = layout.enabled().to_vec();
        flags[box_idx][expert_idx] = enabled;
        if flags[box_idx].iter().all(|e| !e) {
            anyhow::bail!(
                "disabling that expert would leave box {box_idx} empty; the router softmax \
                 would be NaN"
            );
        }
        layout = BoxLayout::new(flags);
        let device = self.masks[box_idx].val().device();
        self.masks[box_idx] = mask_param(&layout, box_idx, &device);
        Ok(())
    }

    /// Widen to a superset layout: new expert columns are zero-initialized and
    /// masked off, existing columns are preserved bit-exactly.
    ///
    /// The three splices below assume `LinearLayout::Row`, i.e. weights shaped
    /// `[in, out]`, which is what `LinearConfig::new` produces throughout this
    /// crate. The assertion guards against that changing underneath us.
    pub fn grown(mut self, layout: &BoxLayout, device: &B::Device) -> anyhow::Result<Self> {
        if layout.num_boxes() < self.num_boxes() {
            anyhow::bail!("cannot shrink the box count");
        }

        for i in 0..self.num_boxes() {
            let old = self.expert_routers[i].width();
            let new = layout.experts_in(i);
            if new < old {
                anyhow::bail!("cannot shrink box {i} from {old} to {new} experts");
            }
            if new == old {
                self.masks[i] = mask_param(layout, i, device);
                continue;
            }
            self.expert_routers[i] = grow_router(&self.expert_routers[i], new, device)?;
            self.masks[i] = mask_param(layout, i, device);
        }

        // Whole new boxes widen the box router too.
        if layout.num_boxes() > self.num_boxes() {
            let input = self.box_router.weight_dims()?[0];
            self.box_router = grow_router(&self.box_router, layout.num_boxes(), device)?;
            for i in self.expert_routers.len()..layout.num_boxes() {
                self.expert_routers
                    .push(TopKRouter::new(input, layout.experts_in(i), device));
                self.masks.push(mask_param(layout, i, device));
            }
        }
        Ok(self)
    }
}

/// `[1, E_i]` of `0.0` / `-inf`, non-trainable.
fn mask_param<B: Backend>(
    layout: &BoxLayout,
    box_idx: usize,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    let width = layout.experts_in(box_idx);
    let values: Vec<f32> = (0..width)
        .map(|e| {
            if layout.is_enabled(box_idx, e) {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let tensor = Tensor::<B, 1>::from_floats(values.as_slice(), device)
        .reshape([1, width])
        .detach();
    Param::from_tensor(tensor).set_require_grad(false)
}

/// Widen a router's output, preserving existing columns exactly.
fn grow_router<B: Backend>(
    router: &TopKRouter<B>,
    width: usize,
    device: &B::Device,
) -> anyhow::Result<TopKRouter<B>> {
    let dims = router.weight_dims()?;
    let (input, old) = (dims[0], dims[1]);
    anyhow::ensure!(width >= old, "cannot shrink a router");

    let weight = Tensor::cat(
        vec![router.weight(), Tensor::<B, 2>::zeros([input, width - old], device)],
        1,
    );
    let bias = router
        .bias()
        .map(|b| Tensor::cat(vec![b, Tensor::<B, 1>::zeros([width - old], device)], 0));
    Ok(TopKRouter::from_parts(weight, bias))
}

/// Broadcast a per-example condition `[b, cond]` to per-token `[b * n, cond]`.
fn broadcast_cond<B: Backend>(cond: &Tensor<B, 2>, n: usize) -> Tensor<B, 2> {
    let [b, c] = cond.dims();
    cond.clone()
        .unsqueeze_dim::<3>(1)
        .repeat_dim(1, n)
        .reshape([b * n, c])
}

// -------------------------------------------------- site (c): FFN slot --

/// Output of [`MosmeFeedForward::forward`].
#[derive(Debug, Clone)]
pub struct MosmeOutput<B: Backend> {
    pub output: Tensor<B, 3>,
    pub balance: BalanceBreakdown<B>,
    pub gates: HierarchicalGates<B>,
}

/// A transformer block's feed-forward, replaced by boxes of MLP experts.
#[derive(Module, Debug)]
pub struct MosmeFeedForward<B: Backend> {
    router: HierarchicalRouter<B>,
    /// `boxes[i]` holds `E_i` experts. Ragged box sizes are supported because
    /// `Vec<T>` is itself a Burn `Module`.
    boxes: Vec<Vec<ExpertMlp<B>>>,
    balance: BalanceWeights,
}

impl<B: Backend> MosmeFeedForward<B> {
    pub fn new(config: &MosmeConfig, device: &B::Device) -> Self {
        let layout = config.spec.layout();
        let boxes = (0..layout.num_boxes())
            .map(|i| {
                (0..layout.experts_in(i))
                    .map(|_| {
                        ExpertMlp::new(config.hidden_size, config.intermediate_size, device)
                    })
                    .collect()
            })
            .collect();
        Self {
            router: HierarchicalRouter::new(config, device),
            boxes,
            balance: config.balance(),
        }
    }

    pub fn router(&self) -> &HierarchicalRouter<B> {
        &self.router
    }

    /// Configured router z-loss weight.
    pub fn z_level(&self) -> f64 {
        self.balance.z_level
    }

    pub fn router_mut(&mut self) -> &mut HierarchicalRouter<B> {
        &mut self.router
    }

    pub fn expert(&self, box_idx: usize, expert_idx: usize) -> Option<&ExpertMlp<B>> {
        self.boxes.get(box_idx)?.get(expert_idx)
    }

    pub fn num_boxes(&self) -> usize {
        self.boxes.len()
    }

    pub fn total_experts(&self) -> usize {
        self.boxes.iter().map(Vec::len).sum()
    }

    /// Mix every expert, weighted by its composed gate.
    ///
    /// The accumulation order deliberately matches [`MoELayer::forward`]'s so
    /// the single-box case reduces to it bit-for-bit.
    pub fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 2>) -> MosmeOutput<B> {
        let device = x.device();
        let [b, n, h] = x.dims();
        let t = b * n;

        let input = self.router.router_input(&x, &cond);
        let gates = self.router.route(input);
        let composed = gates.composed();

        let x_flat = x.reshape([t, h]);
        let mut out = Tensor::zeros(x_flat.shape(), &device);
        for (i, experts) in self.boxes.iter().enumerate() {
            for (j, expert) in experts.iter().enumerate() {
                let gate = composed[i].clone().narrow(1, j, 1); // [T, 1]
                out = out + expert.forward(x_flat.clone()) * gate;
            }
        }

        let balance = gates.balance_loss(self.balance);
        MosmeOutput { output: out.reshape([b, n, h]), balance, gates }
    }

    /// The exact flat equivalent, when there is exactly one box.
    ///
    /// This is the constructive core of the reduction certificate: it hands
    /// back a real [`MoELayer`] sharing this layer's weights, so the two can be
    /// compared directly rather than by re-deriving what the flat path would
    /// have done.
    pub fn as_flat(&self) -> Option<MoELayer<B>> {
        if self.boxes.len() != 1 {
            return None;
        }
        // Burn parameters are lazily initialized, and cloning one that has not
        // been forced yet gives the clone its own deferred initializer -- it
        // would draw *different* random weights while keeping the same
        // `ParamId`. Materialize before cloning or this "equivalent" layer is a
        // different model.
        crate::tensor_ext::force_initialization(self);
        Some(MoELayer::from_parts(
            self.router.expert_routers[0].clone(),
            self.boxes[0].clone(),
            self.router.top_expert,
            self.router.route_on_tokens,
            self.balance.z_level,
        ))
    }

    /// Widen to a superset spec. Existing weights are preserved bit-exactly and
    /// new experts arrive masked off, so the layer's output is unchanged until
    /// one is deliberately enabled.
    pub fn grown(
        mut self,
        spec: &MosmeSpec,
        config: &MosmeConfig,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let layout = spec.layout();
        anyhow::ensure!(
            layout.num_boxes() >= self.boxes.len(),
            "cannot shrink the box count"
        );

        for i in 0..layout.num_boxes() {
            let want = layout.experts_in(i);
            if i < self.boxes.len() {
                anyhow::ensure!(want >= self.boxes[i].len(), "cannot shrink box {i}");
                for _ in self.boxes[i].len()..want {
                    self.boxes[i].push(ExpertMlp::new(
                        config.hidden_size,
                        config.intermediate_size,
                        device,
                    ));
                }
            } else {
                self.boxes.push(
                    (0..want)
                        .map(|_| {
                            ExpertMlp::new(config.hidden_size, config.intermediate_size, device)
                        })
                        .collect(),
                );
            }
        }
        self.router = self.router.grown(&layout, device)?;
        Ok(self)
    }
}

impl<B: Backend> MosmeFeedForward<B> {
    /// Emit the manifest describing this layer's experts.
    ///
    /// The spec supplies the names and tags — they cannot be recovered from
    /// the weights, and Burn would not persist them inside the module anyway.
    /// Everything else (shapes, parameter counts, per-expert content hashes,
    /// the enabled mask) is measured from the live module, so the index makes
    /// only claims it can substantiate.
    pub fn index(
        &self,
        spec: &MosmeSpec,
        model_id: &str,
        param_path: &str,
        cond_size: usize,
    ) -> anyhow::Result<ExpertIndex> {
        anyhow::ensure!(
            spec.experts_per_box() == self.boxes.iter().map(Vec::len).collect::<Vec<_>>(),
            "spec and module disagree on expert counts: {:?} vs {:?}",
            spec.experts_per_box(),
            self.boxes.iter().map(Vec::len).collect::<Vec<_>>()
        );

        let layout = self.router.layout();
        let hidden_size = self.hidden_size();
        let intermediate_size = self.intermediate_size();
        let mut global = 0usize;
        let mut boxes = Vec::with_capacity(spec.boxes.len());

        for (bi, box_spec) in spec.boxes.iter().enumerate() {
            let mut experts = Vec::with_capacity(box_spec.experts.len());
            for (ei, expert_spec) in box_spec.experts.iter().enumerate() {
                let module = &self.boxes[bi][ei];
                experts.push(ExpertEntry {
                    id: expert_spec.id.clone(),
                    label: expert_spec.label.clone(),
                    index: ei,
                    global_index: global,
                    kind: ExpertKind::Mlp { hidden_size, intermediate_size },
                    // Read from the mask, not the spec: the module is the
                    // authority on what is actually switched on.
                    enabled: layout.is_enabled(bi, ei),
                    num_parameters: 2 * hidden_size * intermediate_size
                        + hidden_size
                        + intermediate_size,
                    weights: WeightLocator {
                        checkpoint: None,
                        param_path: format!("{param_path}.boxes.{bi}.{ei}"),
                        sha256: crate::checkpoint::canonical_hash_hex::<B, _>(module),
                    },
                    tags: expert_spec.tags.clone(),
                });
                global += 1;
            }
            boxes.push(BoxEntry {
                id: box_spec.id.clone(),
                label: box_spec.label.clone(),
                index: bi,
                experts,
            });
        }

        let index = ExpertIndex {
            schema_version: crate::expert_index::INDEX_SCHEMA_VERSION,
            model_id: model_id.to_string(),
            site: SiteKind::Mlp,
            routing: RoutingSpec {
                cond_size,
                hidden_size,
                route_on_tokens: self.router.route_on_tokens,
                top_box: self.router.top_box,
                top_expert: self.router.top_expert,
                balance: self.balance,
            },
            boxes,
        };
        index.validate()?;
        Ok(index)
    }

    fn hidden_size(&self) -> usize {
        self.boxes
            .first()
            .and_then(|b| b.first())
            .map_or(0, ExpertMlp::hidden_size)
    }

    fn intermediate_size(&self) -> usize {
        self.boxes
            .first()
            .and_then(|b| b.first())
            .map_or(0, ExpertMlp::intermediate_size)
    }
}

// ------------------------------------------- site (a): adapter bank --

/// Output of [`MosmeAdapterBank::forward`].
#[derive(Debug, Clone)]
pub struct BankOutput<B: Backend> {
    pub output: Tensor<B, 2>,
    pub balance: BalanceBreakdown<B>,
    pub gates: HierarchicalGates<B>,
}

/// Boxes of specialists as **low-rank deltas over one shared frozen `Linear`**.
///
/// This is the granularity that makes "hundreds of specialists" affordable: the
/// base weight is paid for once, and each expert costs `rank * (in + out)`
/// parameters. Because every branch is linear in `x`,
///
/// ```text
/// sum_e g_e * (x A_e B_e s_e) == x * (sum_e g_e * dW_e)
/// ```
///
/// so a routing decision can be *merged* into a single dense weight for
/// deployment. [`Self::merged_for`] does that, and the equality is asserted
/// rather than assumed.
///
/// Every adapter's `B` factor is zero-initialized, so a freshly built bank is
/// **exactly** its base layer for any input and any routing condition — a
/// strictly stronger statement than the flat LoRA identity, which only covers
/// a single adapter.
#[derive(Module, Debug)]
pub struct MosmeAdapterBank<B: Backend> {
    router: HierarchicalRouter<B>,
    /// Shared and frozen. NF4-quantized when built through [`Self::quantized`].
    base: Linear<B>,
    adapters: Vec<Vec<LoraAdapter<B>>>,
    balance: BalanceWeights,
    scaling: f64,
}

impl<B: Backend> MosmeAdapterBank<B> {
    /// Attach a bank of adapters to an existing layer.
    ///
    /// The router is conditioned on `cond` concatenated with the layer's own
    /// input when `spec.route_on_tokens` is set, so the routing input width is
    /// `cond_size + in_features`.
    pub fn from_linear(
        base: Linear<B>,
        spec: &MosmeSpec,
        cond_size: usize,
        rank: usize,
        alpha: f64,
        device: &B::Device,
    ) -> Self {
        let dims = base.weight.dims();
        let (in_features, out_features) = (dims[0], dims[1]);
        let layout = spec.layout();

        let lora = LoraConfig::new(in_features, out_features, rank).with_alpha(alpha);
        let adapters = (0..layout.num_boxes())
            .map(|i| {
                (0..layout.experts_in(i))
                    .map(|_| LoraAdapter::new(&lora, device))
                    .collect()
            })
            .collect();

        let router_cfg = MosmeConfig::new(in_features, cond_size, spec.clone());
        Self {
            router: HierarchicalRouter::new(&router_cfg, device),
            // Frozen: the whole point is that specialists are cheap deltas on
            // a base nobody retrains.
            base: base.no_grad(),
            adapters,
            balance: spec.balance,
            scaling: lora.scaling(),
        }
    }

    pub fn router(&self) -> &HierarchicalRouter<B> {
        &self.router
    }

    pub fn router_mut(&mut self) -> &mut HierarchicalRouter<B> {
        &mut self.router
    }

    pub fn base(&self) -> &Linear<B> {
        &self.base
    }

    pub fn adapter(&self, box_idx: usize, expert_idx: usize) -> Option<&LoraAdapter<B>> {
        self.adapters.get(box_idx)?.get(expert_idx)
    }

    pub fn total_experts(&self) -> usize {
        self.adapters.iter().map(Vec::len).sum()
    }

    pub fn in_features(&self) -> usize {
        self.base.weight.dims()[0]
    }

    pub fn out_features(&self) -> usize {
        self.base.weight.dims()[1]
    }

    /// `base(x) + sum_{i,j} g(i,j) * adapter_ij(x)`.
    pub fn forward(&self, x: Tensor<B, 2>, cond: Tensor<B, 2>) -> BankOutput<B> {
        let input = self.router.router_input_2d(&x, &cond);
        let gates = self.router.route(input);
        let composed = gates.composed();

        let mut out = self.base.forward(x.clone());
        for (i, experts) in self.adapters.iter().enumerate() {
            for (j, adapter) in experts.iter().enumerate() {
                let gate = composed[i].clone().narrow(1, j, 1); // [T, 1]
                out = out + adapter.forward(x.clone()) * gate;
            }
        }

        let balance = gates.balance_loss(self.balance);
        BankOutput { output: out, balance, gates }
    }

    /// The dense weight a fixed routing decision is equivalent to:
    /// `W_base + sum_{i,j} g_ij * dW_ij`.
    ///
    /// `gates` is indexed in the flattened `(box, expert)` order. Only
    /// meaningful when every row of the batch routes identically — otherwise
    /// there is no single weight the layer is equivalent to.
    pub fn merged_for(&self, gates: &[f32]) -> anyhow::Result<Linear<B>> {
        anyhow::ensure!(
            gates.len() == self.total_experts(),
            "expected {} gates, got {}",
            self.total_experts(),
            gates.len()
        );

        let mut weight = self.base.weight.val();
        let mut flat = 0usize;
        for experts in &self.adapters {
            for adapter in experts {
                if gates[flat] != 0.0 {
                    weight = weight + adapter.delta_weight().mul_scalar(gates[flat]);
                }
                flat += 1;
            }
        }

        let mut merged = self.base.clone();
        merged.weight = burn::module::Param::from_tensor(weight.detach());
        Ok(merged)
    }

    /// Emit the manifest for this bank.
    pub fn index(
        &self,
        spec: &MosmeSpec,
        model_id: &str,
        param_path: &str,
        cond_size: usize,
    ) -> anyhow::Result<ExpertIndex> {
        anyhow::ensure!(
            spec.experts_per_box() == self.adapters.iter().map(Vec::len).collect::<Vec<_>>(),
            "spec and module disagree on expert counts"
        );

        let layout = self.router.layout();
        let (in_features, out_features) = (self.in_features(), self.out_features());
        let mut global = 0usize;
        let mut boxes = Vec::with_capacity(spec.boxes.len());

        for (bi, box_spec) in spec.boxes.iter().enumerate() {
            let mut experts = Vec::with_capacity(box_spec.experts.len());
            for (ei, expert_spec) in box_spec.experts.iter().enumerate() {
                let adapter = &self.adapters[bi][ei];
                let rank = adapter.rank();
                experts.push(ExpertEntry {
                    id: expert_spec.id.clone(),
                    label: expert_spec.label.clone(),
                    index: ei,
                    global_index: global,
                    kind: ExpertKind::Adapter {
                        rank,
                        alpha: self.scaling * rank as f64,
                        in_features,
                        out_features,
                    },
                    enabled: layout.is_enabled(bi, ei),
                    num_parameters: rank * (in_features + out_features),
                    weights: WeightLocator {
                        checkpoint: None,
                        param_path: format!("{param_path}.adapters.{bi}.{ei}"),
                        sha256: crate::checkpoint::canonical_hash_hex::<B, _>(adapter),
                    },
                    tags: expert_spec.tags.clone(),
                });
                global += 1;
            }
            boxes.push(BoxEntry {
                id: box_spec.id.clone(),
                label: box_spec.label.clone(),
                index: bi,
                experts,
            });
        }

        let index = ExpertIndex {
            schema_version: crate::expert_index::INDEX_SCHEMA_VERSION,
            model_id: model_id.to_string(),
            site: SiteKind::Adapter,
            routing: RoutingSpec {
                cond_size,
                hidden_size: in_features,
                route_on_tokens: self.router.route_on_tokens,
                top_box: self.router.top_box,
                top_expert: self.router.top_expert,
                balance: self.balance,
            },
            boxes,
        };
        index.validate()?;
        Ok(index)
    }

    /// Parameters `mode` permits an optimizer to move.
    pub fn trainable(&self, mode: &TrainingMode, spec: &MosmeSpec) -> anyhow::Result<TrainableSet> {
        use burn::module::list_param_ids;

        Ok(match mode {
            TrainingMode::Joint => TrainableSet::all(),
            TrainingMode::Router => TrainableSet::from_ids(list_param_ids::<_, B>(&self.router)),
            TrainingMode::Experts => TrainableSet::from_ids(
                self.adapters
                    .iter()
                    .flatten()
                    .flat_map(|a| list_param_ids::<_, B>(a))
                    .collect(),
            ),
            TrainingMode::Specialist { expert_id } => {
                let (bi, ei) = spec
                    .position(expert_id)
                    .ok_or_else(|| anyhow::anyhow!("no expert '{expert_id}' in this spec"))?;
                let adapter = self
                    .adapter(bi, ei)
                    .ok_or_else(|| anyhow::anyhow!("expert '{expert_id}' is not in this module"))?;
                TrainableSet::from_ids(list_param_ids::<_, B>(adapter))
            }
        })
    }
}

impl<B: Backend<FloatElem = f32>> MosmeAdapterBank<B> {
    /// [`Self::from_linear`] with the base weight NF4-quantized first — the
    /// QLoRA composition, where the frozen base costs ~4.5 bits per weight and
    /// only the tiny adapters are trained.
    pub fn quantized(
        base: Linear<B>,
        spec: &MosmeSpec,
        cond_size: usize,
        rank: usize,
        alpha: f64,
        double_quantization: bool,
        device: &B::Device,
    ) -> Self {
        let mut base = base;
        let quantized =
            crate::quantize::quantize_dequantize_tensor(base.weight.val(), double_quantization);
        base.weight = burn::module::Param::from_tensor(quantized.detach());
        Self::from_linear(base, spec, cond_size, rank, alpha, device)
    }
}

// -------------------------------------------- site (b): micro-models --

/// Output of [`MosmeEnsemble`]'s forward passes.
#[derive(Debug, Clone)]
pub struct EnsembleOutput<B: Backend> {
    /// Mixture probabilities `[b, num_labels]`; sums to 1 per row.
    pub probs: Tensor<B, 2>,
    pub balance: BalanceBreakdown<B>,
    pub gates: HierarchicalGates<B>,
    /// The `(box, expert)` pairs actually evaluated. Every pair for the dense
    /// path; only the selected ones for the sparse path.
    pub evaluated: Vec<(usize, usize)>,
}

impl<B: Backend> EnsembleOutput<B> {
    /// Log-probabilities, so existing cross-entropy code paths work unchanged.
    pub fn log_probs(&self) -> Tensor<B, 2> {
        self.probs.clone().clamp_min(1e-30).log()
    }
}

/// Boxes of **whole micro-models**, mixed in probability space.
///
/// This is the granularity that matches "a box filled with small models": each
/// expert is a complete [`DblockClassifier`] with its own weights, and a
/// `coding/rust` specialist is a model in its own right rather than a sub-layer.
///
/// # Why probability space
///
/// The mixture is `sum_{i,j} g(i,j) * softmax(logits_ij)`, not a mixture of
/// logits. A convex combination of distributions is a distribution, so the
/// result still sums to one and `x0 = probs @ W` still lies in the convex hull
/// of the label embeddings — the two `model` certificates survive verbatim.
/// Mixing logits would break both readings for no benefit.
///
/// # Why sparsity is real here
///
/// The intra-layer sites evaluate every expert densely, because the gate is a
/// tensor and skipping would fragment the graph for no gain at that scale. Here
/// each unevaluated expert is an entire forward pass, so
/// [`Self::forward_sparse`] genuinely skips them. The two paths agree
/// **bit-for-bit**: a skipped term contributes `0.0 * y`, and `x + 0.0` is
/// exact.
#[derive(Debug)]
pub struct MosmeEnsemble<B: Backend<FloatElem = f32>> {
    router: HierarchicalRouter<B>,
    specialists: Vec<Vec<DblockClassifier<B>>>,
    balance: BalanceWeights,
}

impl<B: Backend<FloatElem = f32>> MosmeEnsemble<B> {
    /// Build an ensemble from already-constructed specialists.
    pub fn new(
        router: HierarchicalRouter<B>,
        specialists: Vec<Vec<DblockClassifier<B>>>,
        balance: BalanceWeights,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            router.experts_per_box() == specialists.iter().map(Vec::len).collect::<Vec<_>>(),
            "router and specialist counts disagree: {:?} vs {:?}",
            router.experts_per_box(),
            specialists.iter().map(Vec::len).collect::<Vec<_>>()
        );
        Ok(Self { router, specialists, balance })
    }

    /// Build a fresh ensemble: one identically shaped model per expert.
    pub fn fresh(
        spec: &MosmeSpec,
        vit: &crate::vit::ViTDiTConfig,
        dblock: &crate::dblock::DblockConfig,
        cond_size: usize,
        device: &B::Device,
    ) -> Self {
        let layout = spec.layout();
        let specialists = (0..layout.num_boxes())
            .map(|i| {
                (0..layout.experts_in(i))
                    .map(|_| DblockClassifier::<B>::new(vit, dblock, device))
                    .collect()
            })
            .collect();
        // The router sees only the condition here: there is no single "token
        // feature" at model scope, and the caller supplies the domain signal.
        let mut routing = spec.clone();
        routing.route_on_tokens = false;
        let router_cfg = MosmeConfig::new(vit.hidden_size, cond_size, routing);
        Self {
            router: HierarchicalRouter::new(&router_cfg, device),
            specialists,
            balance: spec.balance,
        }
    }

    pub fn router(&self) -> &HierarchicalRouter<B> {
        &self.router
    }

    pub fn router_mut(&mut self) -> &mut HierarchicalRouter<B> {
        &mut self.router
    }

    pub fn specialist(&self, box_idx: usize, expert_idx: usize) -> Option<&DblockClassifier<B>> {
        self.specialists.get(box_idx)?.get(expert_idx)
    }

    pub fn total_experts(&self) -> usize {
        self.specialists.iter().map(Vec::len).sum()
    }

    /// Every specialist evaluated and gated. Differentiable through all of
    /// them, which is what training needs.
    pub fn forward_dense(
        &self,
        pixel_values: &Tensor<B, 4>,
        zt: &Tensor<B, 2>,
        sigmas: &[f64],
        cond: Tensor<B, 2>,
    ) -> EnsembleOutput<B> {
        self.run(pixel_values, zt, sigmas, cond, false)
    }

    /// Only the selected specialists evaluated.
    ///
    /// Routing here is per-*example*, so the top-k indices are already on the
    /// host and the skip decision costs nothing. This is the path an
    /// index-reading inference engine wants: it loads and runs only what the
    /// router asked for.
    pub fn forward_sparse(
        &self,
        pixel_values: &Tensor<B, 4>,
        zt: &Tensor<B, 2>,
        sigmas: &[f64],
        cond: Tensor<B, 2>,
    ) -> EnsembleOutput<B> {
        self.run(pixel_values, zt, sigmas, cond, true)
    }

    fn run(
        &self,
        pixel_values: &Tensor<B, 4>,
        zt: &Tensor<B, 2>,
        sigmas: &[f64],
        cond: Tensor<B, 2>,
        sparse: bool,
    ) -> EnsembleOutput<B> {
        let device = pixel_values.device();
        let batch = pixel_values.dims()[0];

        // Model-scope routing is per example, so the router input is the
        // condition alone.
        let gates = self.router.route(cond.clone());
        let composed = gates.composed();

        // Which pairs carry any weight at all, decided on the host.
        let mut live = vec![vec![true; 0]; self.specialists.len()];
        for (i, experts) in self.specialists.iter().enumerate() {
            let column: Vec<f32> = composed[i]
                .clone()
                .max_dim(0)
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
                .collect();
            live[i] = (0..experts.len()).map(|j| !sparse || column[j] > 0.0).collect();
        }

        let num_labels = self.specialists[0][0]
            .model()
            .label_embedding_weight()
            .dims()[0];
        let mut probs: Option<Tensor<B, 2>> = None;
        let mut evaluated = Vec::new();

        for (i, experts) in self.specialists.iter().enumerate() {
            for (j, model) in experts.iter().enumerate() {
                if !live[i][j] {
                    continue;
                }
                let logits = model.denoise(pixel_values.clone(), zt.clone(), sigmas, None);
                let gate = composed[i].clone().narrow(1, j, 1); // [b, 1]
                let term = softmax(logits, 1) * gate;
                probs = Some(match probs {
                    None => term,
                    Some(acc) => acc + term,
                });
                evaluated.push((i, j));
            }
        }

        let probs = probs.unwrap_or_else(|| Tensor::zeros([batch, num_labels], &device));
        let balance = gates.balance_loss(self.balance);
        EnsembleOutput { probs, balance, gates, evaluated }
    }

    /// Emit the manifest. Each specialist is a whole model, so the recorded
    /// shape is the model's rather than a layer's.
    pub fn index(
        &self,
        spec: &MosmeSpec,
        vit: &crate::vit::ViTDiTConfig,
        model_id: &str,
        cond_size: usize,
    ) -> anyhow::Result<ExpertIndex> {
        anyhow::ensure!(
            spec.experts_per_box() == self.specialists.iter().map(Vec::len).collect::<Vec<_>>(),
            "spec and module disagree on expert counts"
        );

        let layout = self.router.layout();
        let mut global = 0usize;
        let mut boxes = Vec::with_capacity(spec.boxes.len());

        for (bi, box_spec) in spec.boxes.iter().enumerate() {
            let mut experts = Vec::with_capacity(box_spec.experts.len());
            for (ei, expert_spec) in box_spec.experts.iter().enumerate() {
                let model = &self.specialists[bi][ei];
                experts.push(ExpertEntry {
                    id: expert_spec.id.clone(),
                    label: expert_spec.label.clone(),
                    index: ei,
                    global_index: global,
                    kind: ExpertKind::Model {
                        num_blocks: model.num_blocks(),
                        num_hidden_layers: vit.num_hidden_layers,
                        hidden_size: vit.hidden_size,
                        num_labels: vit.num_labels,
                    },
                    enabled: layout.is_enabled(bi, ei),
                    num_parameters: 0,
                    weights: WeightLocator {
                        // A whole model belongs in its own checkpoint: that is
                        // what lets an engine fetch one specialist without
                        // pulling the rest.
                        checkpoint: Some(format!("{}.mpk", expert_spec.id.replace('/', "-")).into()),
                        param_path: String::new(),
                        sha256: crate::checkpoint::canonical_hash_hex::<B, _>(model),
                    },
                    tags: expert_spec.tags.clone(),
                });
                global += 1;
            }
            boxes.push(BoxEntry {
                id: box_spec.id.clone(),
                label: box_spec.label.clone(),
                index: bi,
                experts,
            });
        }

        let index = ExpertIndex {
            schema_version: crate::expert_index::INDEX_SCHEMA_VERSION,
            model_id: model_id.to_string(),
            site: SiteKind::Model,
            routing: RoutingSpec {
                cond_size,
                hidden_size: vit.hidden_size,
                route_on_tokens: false,
                top_box: self.router.top_box,
                top_expert: self.router.top_expert,
                balance: self.balance,
            },
            boxes,
        };
        index.validate()?;
        Ok(index)
    }
}

// ------------------------------------------------------ training modes --

/// Which parameters a training run is allowed to move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingMode {
    /// Everything trains.
    Joint,
    /// Both routers train; every expert is frozen. Use after adding experts,
    /// to teach the router where the new capacity is.
    Router,
    /// One expert trains; routers and every other expert are frozen. This is
    /// what "add a Rust specialist without retraining the others" means
    /// operationally.
    Specialist { expert_id: String },
    /// Every expert trains; routers frozen.
    Experts,
}

impl TrainingMode {
    pub fn parse(name: &str, expert: Option<&str>) -> anyhow::Result<Self> {
        match name {
            "joint" => Ok(Self::Joint),
            "router" => Ok(Self::Router),
            "experts" => Ok(Self::Experts),
            "specialist" => Ok(Self::Specialist {
                expert_id: expert
                    .ok_or_else(|| anyhow::anyhow!("--mosme-mode specialist needs --mosme-expert"))?
                    .to_string(),
            }),
            other => anyhow::bail!(
                "unknown mosme mode '{other}' (expected joint|router|specialist|experts)"
            ),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Joint => "joint",
            Self::Router => "router",
            Self::Specialist { .. } => "specialist",
            Self::Experts => "experts",
        }
    }
}

/// An allowlist of parameters an optimizer may update.
///
/// # This must be rebuilt after any checkpoint load
///
/// Burn's `load_record` adopts the *record's* `ParamId` for every parameter, so
/// a set captured before a `--resume` refers to ids that no longer exist. The
/// failure is at least loud rather than silent — the filtered `GradientsParams`
/// comes back empty, the global gradient norm is `0.0`, and the existing
/// gradient gate rejects every step — but it is still a trap. Build the set
/// from the live model, after all loading, immediately before the first step.
/// It *does* survive `optim.step`, which preserves ids.
#[derive(Debug, Clone, Default)]
pub struct TrainableSet {
    ids: Option<Vec<burn::module::ParamId>>,
}

impl TrainableSet {
    /// Everything trainable.
    pub fn all() -> Self {
        Self { ids: None }
    }

    pub fn from_ids(ids: Vec<burn::module::ParamId>) -> Self {
        Self { ids: Some(ids) }
    }

    /// Number of allowed parameters, or `None` when unrestricted.
    pub fn len(&self) -> Option<usize> {
        self.ids.as_ref().map(Vec::len)
    }

    pub fn is_empty(&self) -> bool {
        self.ids.as_ref().is_some_and(Vec::is_empty)
    }

    pub fn ids(&self) -> Option<&[burn::module::ParamId]> {
        self.ids.as_deref()
    }

    /// Collect only the permitted gradients.
    pub fn gradients<BB, M>(
        &self,
        grads: &mut BB::Gradients,
        model: &M,
    ) -> burn::optim::GradientsParams
    where
        BB: burn::tensor::backend::AutodiffBackend,
        M: burn::module::AutodiffModule<BB>,
    {
        match &self.ids {
            // Unfiltered, so the common case pays no lookup cost.
            None => burn::optim::GradientsParams::from_module(grads, model),
            Some(ids) => burn::optim::GradientsParams::from_params(grads, model, ids),
        }
    }
}

impl<B: Backend> MosmeFeedForward<B> {
    /// The parameters `mode` permits an optimizer to move.
    ///
    /// Frozen parameters simply produce no gradient entry, so the optimizer
    /// leaves them untouched — there is no need to also flip `require_grad`,
    /// and doing so would change what the backward pass computes.
    pub fn trainable(&self, mode: &TrainingMode, spec: &MosmeSpec) -> anyhow::Result<TrainableSet> {
        use burn::module::list_param_ids;

        Ok(match mode {
            TrainingMode::Joint => TrainableSet::all(),
            TrainingMode::Router => {
                // The routing masks are non-trainable by construction, so they
                // never appear in a gradient set; listing them is harmless but
                // pointless, and excluding them keeps the intent explicit.
                TrainableSet::from_ids(list_param_ids::<_, B>(&self.router))
            }
            TrainingMode::Experts => TrainableSet::from_ids(
                self.boxes
                    .iter()
                    .flatten()
                    .flat_map(|e| list_param_ids::<_, B>(e))
                    .collect(),
            ),
            TrainingMode::Specialist { expert_id } => {
                let (bi, ei) = spec.position(expert_id).ok_or_else(|| {
                    anyhow::anyhow!("no expert '{expert_id}' in this spec")
                })?;
                let expert = self
                    .expert(bi, ei)
                    .ok_or_else(|| anyhow::anyhow!("expert '{expert_id}' is not in this module"))?;
                TrainableSet::from_ids(list_param_ids::<_, B>(expert))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_index::{BoxSpec, ExpertSpec};
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    fn ragged_spec() -> MosmeSpec {
        MosmeSpec {
            boxes: vec![
                BoxSpec::new(
                    "coding",
                    "Code",
                    vec![
                        ExpertSpec::new("coding/rust", "Rust"),
                        ExpertSpec::new("coding/python", "Python"),
                        ExpertSpec::new("coding/secure", "Secure"),
                    ],
                ),
                BoxSpec::new(
                    "cyber",
                    "Cybersecurity",
                    vec![
                        ExpertSpec::new("cyber/netsec", "Network"),
                        ExpertSpec::new("cyber/malware", "Malware"),
                    ],
                ),
            ],
            top_box: 1,
            top_expert: 1,
            route_on_tokens: true,
            balance: BalanceWeights::default(),
        }
    }

    fn config(spec: MosmeSpec) -> MosmeConfig {
        MosmeConfig::new(8, 4, spec).with_intermediate_size(16)
    }

    fn inputs(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> (Tensor<B, 3>, Tensor<B, 2>) {
        (
            Tensor::<B, 3>::random([2, 3, 8], Distribution::Uniform(-1.0, 1.0), device),
            Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), device),
        )
    }

    fn max_abs(t: Tensor<B, 2>) -> f32 {
        t.abs().max().into_scalar()
    }

    #[test]
    fn test_composed_gates_sum_to_one_over_ragged_boxes() {
        // The defining property of composing two renormalized levels. Without
        // it the layer output is an arbitrarily scaled mixture rather than a
        // convex combination of expert outputs.
        let device = Default::default();
        for top_box in [1usize, 2] {
            for top_expert in [1usize, 2, 3] {
                let mut spec = ragged_spec();
                spec.top_box = top_box;
                spec.top_expert = top_expert;
                let cfg = config(spec);
                let router = HierarchicalRouter::<B>::new(&cfg, &device);
                let (x, cond) = inputs(&device);

                let gates = router.route(router.router_input(&x, &cond));
                let mass: Vec<f32> = gates
                    .composed_flat()
                    .sum_dim(1)
                    .into_data()
                    .convert::<f32>()
                    .iter::<f32>()
                    .collect();
                for m in mass {
                    assert!(
                        (m - 1.0).abs() < 1e-6,
                        "gates sum to {m} at top_box={top_box}, top_expert={top_expert}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_top_expert_beyond_box_width_is_harmless() {
        // A spec asking for 3 experts from a 2-expert box must clamp rather
        // than panic, and must still produce a valid distribution.
        let device = Default::default();
        let mut spec = ragged_spec();
        spec.top_expert = 5;
        let router = HierarchicalRouter::<B>::new(&config(spec), &device);
        let (x, cond) = inputs(&device);
        let gates = router.route(router.router_input(&x, &cond));
        let mass: Vec<f32> = gates
            .composed_flat()
            .sum_dim(1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        assert!(mass.iter().all(|m| (m - 1.0).abs() < 1e-6), "{mass:?}");
    }

    #[test]
    fn test_disabled_expert_gets_exactly_zero_gate() {
        // Exactly zero, not merely small: this is what makes hot-swap an
        // identity rather than an approximation.
        let device = Default::default();
        let mut spec = ragged_spec();
        spec.boxes[0].experts[1].enabled = false;
        spec.top_expert = 3;
        let router = HierarchicalRouter::<B>::new(&config(spec), &device);
        let (x, cond) = inputs(&device);

        let gates = router.route(router.router_input(&x, &cond));
        let disabled = gates.composed()[0].clone().narrow(1, 1, 1);
        assert_eq!(max_abs(disabled), 0.0, "a disabled expert must get gate 0.0");

        // ...and the rest still form a distribution.
        let mass: Vec<f32> = gates
            .composed_flat()
            .sum_dim(1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        assert!(mass.iter().all(|m| (m - 1.0).abs() < 1e-6), "{mass:?}");
    }

    #[test]
    fn test_single_box_matches_flat_moe_exactly() {
        // The reduction certificate's unit-level form: with one box the
        // hierarchical path must BE the flat path, not merely approximate it.
        let device = Default::default();
        let cfg = config(MosmeSpec::flat(4));
        let layer = MosmeFeedForward::<B>::new(&cfg, &device);
        let flat = layer.as_flat().expect("one box");

        let (x, cond) = inputs(&device);
        let hierarchical = layer.forward(x.clone(), cond.clone());
        let reference = flat.forward(x, cond);

        let out_diff = (hierarchical.output - reference.output).abs().max().into_scalar();
        assert_eq!(out_diff, 0.0, "single-box output must be bit-identical");

        let loss_diff = (hierarchical.balance.expert_loss - reference.balance)
            .abs()
            .max()
            .into_scalar();
        assert_eq!(loss_diff, 0.0, "single-box balance loss must be bit-identical");

        // The box level is degenerate: softmax over one logit is exactly 1.
        let box_loss: f32 = hierarchical.balance.box_loss.into_scalar();
        assert_eq!(box_loss, 1.0, "one box gives a balance loss of exactly 1");

        // ...and contributes no z-loss either, for the same reason: a logit
        // that cannot change any routing decision has nothing to stabilize. So
        // the hierarchical z-loss is exactly the flat one.
        let z_diff = (hierarchical.balance.z_loss - reference.z_loss)
            .abs()
            .max()
            .into_scalar();
        assert_eq!(z_diff, 0.0, "single-box z-loss must be bit-identical too");
    }

    #[test]
    fn test_multi_box_has_no_flat_equivalent() {
        let device = Default::default();
        let layer = MosmeFeedForward::<B>::new(&config(ragged_spec()), &device);
        assert!(layer.as_flat().is_none());
        assert!(config(ragged_spec()).as_flat_moe().is_none());
        assert!(config(MosmeSpec::flat(3)).as_flat_moe().is_some());
    }

    #[test]
    fn test_box_traffic_is_a_distribution() {
        let device = Default::default();
        let router = HierarchicalRouter::<B>::new(&config(ragged_spec()), &device);
        let (x, cond) = inputs(&device);
        let gates = router.route(router.router_input(&x, &cond));
        let total: f32 = gates.box_traffic().sum().into_scalar();
        assert!((total - 1.0).abs() < 1e-6, "box traffic sums to {total}");
    }

    #[test]
    fn test_expert_balance_ignores_boxes_with_no_traffic() {
        // The reason the expert level is gate-weighted: a box the router never
        // selects must contribute nothing. Weighting uniformly over all tokens
        // would push balance pressure into traffic that does not exist.
        let device = Default::default();
        let router = HierarchicalRouter::<B>::new(&config(ragged_spec()), &device);
        let (x, cond) = inputs(&device);
        let mut gates = router.route(router.router_input(&x, &cond));

        // Force all traffic to box 0.
        let t = gates.box_gates.dims()[0];
        gates.box_gates = Tensor::cat(
            vec![
                Tensor::<B, 2>::ones([t, 1], &device),
                Tensor::<B, 2>::zeros([t, 1], &device),
            ],
            1,
        );

        let breakdown = gates.balance_loss(BalanceWeights::default());
        let expert: f32 = breakdown.expert_loss.into_scalar();
        let box0: f32 = breakdown.per_box[0].clone().into_scalar();
        assert!(
            (expert - box0).abs() < 1e-5,
            "with all traffic in box 0 the expert loss should be box 0's: {expert} vs {box0}"
        );
    }

    #[test]
    fn test_balance_loss_bounds() {
        // Per level the Switch loss lies in [1, N] on the diagonal; the convex
        // combination over boxes inherits that termwise.
        let device = Default::default();
        let router = HierarchicalRouter::<B>::new(&config(ragged_spec()), &device);
        let (x, cond) = inputs(&device);
        let gates = router.route(router.router_input(&x, &cond));
        let breakdown = gates.balance_loss(BalanceWeights::default());

        let box_loss: f32 = breakdown.box_loss.into_scalar();
        assert!((0.0..=2.0 + 1e-5).contains(&box_loss), "box loss {box_loss} outside [0, 2]");

        let widths = router.experts_per_box();
        for (i, l) in breakdown.per_box.iter().enumerate() {
            let v: f32 = l.clone().into_scalar();
            assert!(
                (0.0..=widths[i] as f32 + 1e-5).contains(&v),
                "box {i} expert loss {v} outside [0, {}]",
                widths[i]
            );
        }
        let total: f32 = breakdown.total.into_scalar();
        assert!(total.is_finite() && total > 0.0);
    }

    #[test]
    fn test_growing_preserves_outputs_bitwise() {
        // The hot-swap guarantee. Adding an expert must change nothing at all
        // until it is enabled, or "add a specialist without retraining the
        // others" is not a real claim.
        //
        // `top_expert` is set wide enough that every enabled expert receives a
        // gate. Under top-1 routing a newly enabled expert only changes the
        // output if it happens to win the argmax, which would make the second
        // half of this test depend on the random initialization.
        let device = Default::default();
        let mut spec = ragged_spec();
        spec.top_expert = 4;
        let cfg = config(spec.clone());
        let layer = MosmeFeedForward::<B>::new(&cfg, &device);
        let (x, cond) = inputs(&device);
        let before = layer.forward(x.clone(), cond.clone()).output;

        let grown_spec = spec
            .extended_with("coding", ExpertSpec::new("coding/go", "Go"))
            .unwrap();
        let grown = layer.grown(&grown_spec, &cfg, &device).unwrap();
        assert_eq!(grown.total_experts(), 6);
        assert_eq!(grown.router().experts_per_box(), vec![4, 2]);

        let after = grown.forward(x.clone(), cond.clone()).output;
        let diff = (before.clone() - after).abs().max().into_scalar();
        assert_eq!(diff, 0.0, "growing must be an exact identity");

        // The disabled expert really is masked, not merely unlucky.
        let gates = {
            let g = grown.router();
            g.route(g.router_input(&x, &cond))
        };
        assert_eq!(
            max_abs(gates.composed()[0].clone().narrow(1, 3, 1)),
            0.0,
            "the new expert must be gated to exactly zero while disabled"
        );

        // ...and enabling it must then actually give it a gate, so the identity
        // above is not just a dead code path.
        //
        // Asserted on the gate rather than on the output magnitude: the gate
        // going from exactly 0 to strictly positive holds *by construction*
        // (the mask stops being -inf), whereas how much the output moves
        // depends on random weights drawn from a backend RNG that is global and
        // shared across parallel tests.
        let mut enabled = grown;
        enabled.router_mut().set_enabled(0, 3, true).unwrap();
        // Checked on the *expert-level* gate, not the composed one. With
        // top_box = 1 only one box is selected per token, so the composed gate
        // is legitimately zero for every token that routed to the other box —
        // which says nothing about whether this expert was enabled. Within its
        // own box the softmax is over finite logits, so an enabled expert has a
        // positive gate for every token, unconditionally.
        let gates = {
            let g = enabled.router();
            g.route(g.router_input(&x, &cond))
        };
        let new_gate = gates.expert_gates[0]
            .clone()
            .narrow(1, 3, 1)
            .min()
            .into_scalar();
        assert!(
            new_gate > 0.0,
            "an enabled expert must receive a positive within-box gate, got {new_gate}"
        );

        // The output moves for any token that actually reached box 0.
        let box0_traffic = gates.box_gates.clone().narrow(1, 0, 1).max().into_scalar();
        if box0_traffic > 0.0 {
            let moved = (before - enabled.forward(x, cond).output)
                .abs()
                .max()
                .into_scalar();
            assert!(moved > 0.0, "a positive gate must change the output");
        }
    }

    #[test]
    fn test_set_enabled_refuses_to_empty_a_box() {
        let device = Default::default();
        let mut spec = ragged_spec();
        spec.boxes[1].experts[0].enabled = false;
        let cfg = config(spec);
        let mut router = HierarchicalRouter::<B>::new(&cfg, &device);

        // Disabling the last enabled expert in box 1 would make its softmax NaN.
        let err = router.set_enabled(1, 1, false).unwrap_err().to_string();
        assert!(err.contains("NaN"), "{err}");
        assert!(router.set_enabled(9, 0, false).is_err(), "out-of-range must error");

        // Re-enabling is always fine.
        router.set_enabled(1, 0, true).unwrap();
        assert!(router.layout().is_enabled(1, 0));
    }

    #[test]
    fn test_index_describes_the_live_module() {
        // The index must make only claims it can substantiate: shapes and
        // hashes come from the module, names come from the spec, and the
        // enabled flags come from the *mask* rather than the spec -- the
        // module is the authority on what is actually switched on.
        let device = Default::default();
        let spec = ragged_spec();
        let cfg = config(spec.clone());
        let mut layer = MosmeFeedForward::<B>::new(&cfg, &device);
        layer.router_mut().set_enabled(0, 1, false).unwrap();

        let index = layer.index(&spec, "abc123", "vit.layers.1.mlp", 4).unwrap();
        assert_eq!(index.num_boxes(), 2);
        assert_eq!(index.num_experts(), 5);
        assert_eq!(index.experts_per_box(), vec![3, 2]);
        assert_eq!(index.site, crate::expert_index::SiteKind::Mlp);

        let (bx, expert) = index.expert("coding/python").unwrap();
        assert_eq!(bx.id, "coding");
        assert!(!expert.enabled, "the mask, not the spec, decides");
        assert_eq!(expert.weights.param_path, "vit.layers.1.mlp.boxes.0.1");
        assert!(index.expert("coding/rust").unwrap().1.enabled);

        // Distinct experts must hash differently, or the index cannot tell an
        // engine whether it holds the right weights.
        let a = &index.expert("coding/rust").unwrap().1.weights.sha256;
        let b = &index.expert("coding/secure").unwrap().1.weights.sha256;
        assert_ne!(a, b, "independently initialized experts must hash differently");

        // And it survives the wire format an engine would read it through.
        assert_eq!(
            crate::expert_index::ExpertIndex::from_json(&index.to_json().unwrap()).unwrap(),
            index
        );
    }

    #[test]
    fn test_index_rejects_a_spec_that_does_not_match() {
        let device = Default::default();
        let spec = ragged_spec();
        let layer = MosmeFeedForward::<B>::new(&config(spec.clone()), &device);
        let wrong = spec
            .extended_with("coding", ExpertSpec::new("coding/go", "Go"))
            .unwrap();
        let err = layer.index(&wrong, "abc", "p", 4).unwrap_err().to_string();
        assert!(err.contains("disagree on expert counts"), "{err}");
    }

    fn ensemble_fixture() -> (MosmeEnsemble<B>, MosmeSpec, crate::vit::ViTDiTConfig) {
        use crate::dblock::DblockConfig;
        use crate::vit::ViTDiTConfig;
        let device = Default::default();
        let spec = ragged_spec();
        let vit = ViTDiTConfig::tiny(10);
        let dblock = DblockConfig { num_blocks: 2, ..DblockConfig::default() };
        let ensemble = MosmeEnsemble::<B>::fresh(&spec, &vit, &dblock, 4, &device);
        (ensemble, spec, vit)
    }

    #[test]
    fn test_ensemble_output_is_a_distribution() {
        // Mixing in probability space keeps the two `model` certificates valid:
        // a convex combination of distributions is a distribution.
        let device = Default::default();
        let (ensemble, _, _) = ensemble_fixture();
        let pixels =
            Tensor::<B, 4>::random([3, 3, 32, 32], Distribution::Uniform(-1.0, 1.0), &device);
        let zt = Tensor::<B, 2>::random([3, 32], Distribution::Normal(0.0, 1.0), &device);
        let cond = Tensor::<B, 2>::random([3, 4], Distribution::Uniform(-1.0, 1.0), &device);

        let out = ensemble.forward_dense(&pixels, &zt, &[1.0; 3], cond);
        assert_eq!(out.probs.dims(), [3, 10]);
        for m in out
            .probs
            .clone()
            .sum_dim(1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
        {
            assert!((m - 1.0).abs() < 1e-5, "mixture must be a distribution, got {m}");
        }
        assert!(out.log_probs().into_data().convert::<f32>().iter::<f32>().all(f32::is_finite));
        assert_eq!(out.evaluated.len(), 5, "the dense path runs every specialist");
    }

    #[test]
    fn test_ensemble_sparse_matches_dense_exactly() {
        // The whole justification for skipping: an unselected specialist
        // contributes `0.0 * y`, and `x + 0.0` is exact. So sparsity here is
        // free rather than approximate -- unlike the intra-layer sites, where
        // each expert is cheap and dense evaluation is simpler.
        let device = Default::default();
        let (ensemble, _, _) = ensemble_fixture();
        let pixels =
            Tensor::<B, 4>::random([2, 3, 32, 32], Distribution::Uniform(-1.0, 1.0), &device);
        let zt = Tensor::<B, 2>::random([2, 32], Distribution::Normal(0.0, 1.0), &device);
        let cond = Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);

        let dense = ensemble.forward_dense(&pixels, &zt, &[1.0; 2], cond.clone());
        let sparse = ensemble.forward_sparse(&pixels, &zt, &[1.0; 2], cond);

        let diff = (dense.probs - sparse.probs).abs().max().into_scalar();
        assert_eq!(diff, 0.0, "sparse dispatch must be bit-identical to dense");

        // ...and it must actually skip something, or the test is vacuous.
        assert!(
            sparse.evaluated.len() < dense.evaluated.len(),
            "top-1 box routing should skip at least one specialist: {} vs {}",
            sparse.evaluated.len(),
            dense.evaluated.len()
        );
    }

    #[test]
    fn test_ensemble_index_records_per_model_checkpoints() {
        let (ensemble, spec, vit) = ensemble_fixture();
        let index = ensemble.index(&spec, &vit, "ens1", 4).unwrap();
        assert_eq!(index.site, SiteKind::Model);
        assert_eq!(index.num_experts(), 5);

        let (_, expert) = index.expert("cyber/netsec").unwrap();
        assert!(matches!(expert.kind, ExpertKind::Model { hidden_size: 32, num_labels: 10, .. }));
        // A whole model belongs in its own file so an engine can fetch one
        // specialist without pulling the rest.
        assert_eq!(
            expert.weights.checkpoint.as_deref(),
            Some(std::path::Path::new("cyber-netsec.mpk"))
        );
        assert!(!expert.weights.sha256.is_empty());

        // Independently initialized models must hash differently.
        let a = &index.expert("coding/rust").unwrap().1.weights.sha256;
        let b = &index.expert("coding/python").unwrap().1.weights.sha256;
        assert_ne!(a, b);
    }

    #[test]
    fn test_ensemble_rejects_mismatched_router() {
        use crate::dblock::DblockConfig;
        use crate::vit::ViTDiTConfig;
        let device = Default::default();
        let vit = ViTDiTConfig::tiny(10);
        let dblock = DblockConfig { num_blocks: 2, ..DblockConfig::default() };

        // A router built for a different layout must not be silently accepted.
        let wrong = HierarchicalRouter::<B>::new(&config(MosmeSpec::flat(2)), &device);
        let specialists = vec![vec![DblockClassifier::<B>::new(&vit, &dblock, &device)]];
        let err = MosmeEnsemble::new(wrong, specialists, Default::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("disagree"), "{err}");
    }

    #[test]
    fn test_adapter_bank_is_exactly_its_base_at_init() {
        // Strictly stronger than the flat LoRA identity: every branch is zero,
        // so the bank equals its base for *any* input and *any* routing
        // condition, not just one adapter at one point.
        use burn::nn::LinearConfig;
        let device = Default::default();
        let base: burn::nn::Linear<B> = LinearConfig::new(8, 6).init(&device);
        crate::tensor_ext::force_initialization(&base);
        let spec = ragged_spec();
        let bank = MosmeAdapterBank::<B>::from_linear(base.clone(), &spec, 4, 4, 4.0, &device);

        for seed in 0..4u64 {
            <B as burn::tensor::backend::Backend>::seed(&device, seed);
            let x = Tensor::<B, 2>::random([3, 8], Distribution::Uniform(-2.0, 2.0), &device);
            let cond = Tensor::<B, 2>::random([3, 4], Distribution::Uniform(-2.0, 2.0), &device);
            let diff = (bank.forward(x.clone(), cond).output - base.forward(x))
                .abs()
                .max()
                .into_scalar();
            assert_eq!(diff, 0.0, "a fresh adapter bank must be exactly its base");
        }
        assert_eq!(bank.total_experts(), 5);
    }

    #[test]
    fn test_adapter_bank_matches_its_merged_weight() {
        // The linearity that makes merging legal:
        //   sum_e g_e (x A_e B_e s_e) == x (sum_e g_e dW_e).
        // Checked with non-trivial adapters, so it is not 0 == 0.
        use burn::nn::LinearConfig;
        let device = Default::default();
        let base: burn::nn::Linear<B> = LinearConfig::new(8, 6).init(&device);
        crate::tensor_ext::force_initialization(&base);
        // Routing must not depend on x here: `merged_for` collapses ONE
        // routing decision into a weight, so the batch has to route uniformly.
        // With `route_on_tokens` the router sees each row's own features and
        // every row routes differently, which is exactly the precondition the
        // method documents.
        let mut spec = MosmeSpec::flat(3);
        spec.route_on_tokens = false;
        let mut bank = MosmeAdapterBank::<B>::from_linear(base, &spec, 4, 2, 2.0, &device);

        // Give every adapter a real B factor.
        for (j, adapter) in bank.adapters[0].iter_mut().enumerate() {
            let b = Tensor::<B, 2>::random(
                [adapter.rank(), adapter.out_features()],
                Distribution::Uniform(-0.4, 0.4),
                &device,
            )
            .add_scalar(0.05 * j as f32);
            *adapter = LoraAdapter::from_parts(adapter.a(), b, adapter.scaling());
        }

        let x = Tensor::<B, 2>::random([4, 8], Distribution::Uniform(-1.0, 1.0), &device);
        let cond = Tensor::<B, 2>::ones([4, 4], &device);
        let out = bank.forward(x.clone(), cond);

        // Confirm the precondition actually holds before relying on it.
        let per_row: Vec<f32> = out
            .gates
            .composed_flat()
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        assert_eq!(&per_row[0..3], &per_row[3..6], "every row must route identically");

        let gates: Vec<f32> = out
            .gates
            .composed_flat()
            .narrow(0, 0, 1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let merged = bank.merged_for(&gates).unwrap();

        let diff = (out.output - merged.forward(x)).abs().max().into_scalar();
        assert!(diff < 1e-5, "factored and merged forms disagree: {diff}");

        assert!(bank.merged_for(&[1.0]).is_err(), "wrong gate count must error");
    }

    #[test]
    fn test_adapter_bank_index_records_rank_and_site() {
        use burn::nn::LinearConfig;
        let device = Default::default();
        let base: burn::nn::Linear<B> = LinearConfig::new(256, 128).init(&device);
        let spec = ragged_spec();
        let bank = MosmeAdapterBank::<B>::from_linear(base, &spec, 4, 8, 16.0, &device);

        let index = bank.index(&spec, "m1", "head.classifier", 4).unwrap();
        assert_eq!(index.site, SiteKind::Adapter);
        let (_, expert) = index.expert("coding/rust").unwrap();
        assert!(matches!(
            expert.kind,
            ExpertKind::Adapter { rank: 8, in_features: 256, out_features: 128, .. }
        ));
        // rank * (in + out), which is the whole point of this granularity:
        // 3072 parameters against 32768 for a dense replacement.
        assert_eq!(expert.num_parameters, 8 * (256 + 128));
        assert!(
            expert.num_parameters * 10 < 256 * 128,
            "a rank-8 adapter on 256x128 should be ~10x smaller, got {} vs {}",
            expert.num_parameters,
            256 * 128
        );
    }

    #[test]
    fn test_quantized_bank_base_lands_on_the_nf4_grid() {
        use burn::nn::LinearConfig;
        let device = Default::default();
        let base: burn::nn::Linear<B> = LinearConfig::new(64, 8).init(&device);
        crate::tensor_ext::force_initialization(&base);
        let dense = base.weight.val();
        let spec = MosmeSpec::flat(2);
        let bank =
            MosmeAdapterBank::<B>::quantized(base, &spec, 4, 4, 4.0, true, &device);

        let quantized = bank.base().weight.val();
        assert_eq!(quantized.dims(), dense.dims());
        let moved = (quantized - dense).abs().max().into_scalar();
        assert!(moved > 0.0, "quantization must actually change the weights");

        // The adapters are still an exact no-op on top of the quantized base.
        let x = Tensor::<B, 2>::random([2, 64], Distribution::Uniform(-1.0, 1.0), &device);
        let cond = Tensor::<B, 2>::ones([2, 4], &device);
        let diff = (bank.forward(x.clone(), cond).output - bank.base().forward(x))
            .abs()
            .max()
            .into_scalar();
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_adapter_bank_training_modes() {
        use crate::train::DefaultTrainBackend as A;
        use burn::module::list_param_ids;
        use burn::nn::LinearConfig;

        let device = Default::default();
        let base: burn::nn::Linear<A> = LinearConfig::new(8, 6).init(&device);
        let spec = ragged_spec();
        let bank = MosmeAdapterBank::<A>::from_linear(base, &spec, 4, 4, 4.0, &device);

        let one = bank
            .trainable(
                &TrainingMode::Specialist { expert_id: "cyber/malware".into() },
                &spec,
            )
            .unwrap();
        let expected = list_param_ids::<_, A>(bank.adapter(1, 1).unwrap());
        assert_eq!(one.ids().unwrap(), expected.as_slice());
        // A rank-4 adapter over 8x6 has two factors, so a handful of params.
        assert_eq!(one.len(), Some(2));
    }

    #[test]
    fn test_training_modes_select_the_right_parameters() {
        use crate::train::DefaultTrainBackend as A;
        use burn::module::list_param_ids;

        let device = Default::default();
        let spec = ragged_spec();
        let cfg = MosmeConfig::new(8, 4, spec.clone()).with_intermediate_size(16);
        let layer = MosmeFeedForward::<A>::new(&cfg, &device);

        let joint = layer.trainable(&TrainingMode::Joint, &spec).unwrap();
        assert!(joint.len().is_none(), "joint mode is unfiltered");

        // One specialist: exactly that expert's parameters, nothing else.
        let one = layer
            .trainable(
                &TrainingMode::Specialist { expert_id: "coding/python".into() },
                &spec,
            )
            .unwrap();
        let expected = list_param_ids::<_, A>(layer.expert(0, 1).unwrap());
        assert_eq!(one.len(), Some(expected.len()));
        assert_eq!(one.ids().unwrap(), expected.as_slice());

        // ...and it is a strict subset of the experts mode, which is itself a
        // strict subset of everything.
        let all_experts = layer.trainable(&TrainingMode::Experts, &spec).unwrap();
        assert!(all_experts.len().unwrap() > one.len().unwrap());
        for id in one.ids().unwrap() {
            assert!(all_experts.ids().unwrap().contains(id));
        }

        // Router mode must not overlap the experts at all.
        let router = layer.trainable(&TrainingMode::Router, &spec).unwrap();
        assert!(router.len().unwrap() > 0);
        for id in router.ids().unwrap() {
            assert!(
                !all_experts.ids().unwrap().contains(id),
                "router and expert parameter sets must be disjoint"
            );
        }

        assert!(layer
            .trainable(&TrainingMode::Specialist { expert_id: "nope".into() }, &spec)
            .is_err());
    }

    #[test]
    fn test_specialist_training_moves_only_its_own_expert() {
        // The operational claim behind "add a specialist without retraining the
        // others", checked by actually taking an optimizer step.
        use crate::train::DefaultTrainBackend as A;
        use burn::optim::{AdamWConfig, Optimizer};

        let device = Default::default();
        let spec = ragged_spec();
        let cfg = MosmeConfig::new(8, 4, spec.clone()).with_intermediate_size(16);
        let layer = MosmeFeedForward::<A>::new(&cfg, &device);
        crate::tensor_ext::force_initialization(&layer);

        let snapshot = |l: &MosmeFeedForward<A>, bi: usize, ei: usize| -> Vec<f32> {
            l.expert(bi, ei)
                .unwrap()
                .clone()
                .fc_in_weight()
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
                .collect()
        };
        let target_before = snapshot(&layer, 0, 1);
        let other_before = snapshot(&layer, 0, 0);
        let cyber_before = snapshot(&layer, 1, 0);

        let trainable = layer
            .trainable(
                &TrainingMode::Specialist { expert_id: "coding/python".into() },
                &spec,
            )
            .unwrap();

        let x = Tensor::<A, 3>::random([2, 3, 8], Distribution::Uniform(-1.0, 1.0), &device);
        let cond = Tensor::<A, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let out = layer.forward(x, cond);
        let loss = out.output.powf_scalar(2.0).mean();

        let mut grads = loss.backward();
        let params = trainable.gradients(&mut grads, &layer);
        assert!(!params.is_empty(), "the specialist must receive gradients");

        let mut optim = AdamWConfig::new().init();
        let updated = optim.step(1e-2, layer, params);

        assert_ne!(
            snapshot(&updated, 0, 1),
            target_before,
            "the trained specialist must move"
        );
        assert_eq!(
            snapshot(&updated, 0, 0),
            other_before,
            "a sibling expert must be untouched"
        );
        assert_eq!(
            snapshot(&updated, 1, 0),
            cyber_before,
            "an expert in another box must be untouched"
        );
    }

    #[test]
    fn test_training_mode_parsing() {
        assert_eq!(TrainingMode::parse("joint", None).unwrap(), TrainingMode::Joint);
        assert_eq!(TrainingMode::parse("router", None).unwrap().name(), "router");
        assert_eq!(
            TrainingMode::parse("specialist", Some("coding/rust")).unwrap(),
            TrainingMode::Specialist { expert_id: "coding/rust".into() }
        );
        // Specialist mode without a target is a configuration error, not a
        // silent fall-back to training everything.
        assert!(TrainingMode::parse("specialist", None).is_err());
        assert!(TrainingMode::parse("nonsense", None).is_err());
    }

    #[test]
    fn test_layout_round_trips_through_the_masks() {
        let device = Default::default();
        let mut spec = ragged_spec();
        spec.boxes[0].experts[2].enabled = false;
        let router = HierarchicalRouter::<B>::new(&config(spec.clone()), &device);

        let layout = router.layout();
        assert_eq!(layout.experts_per_box(), vec![3, 2]);
        assert!(layout.is_enabled(0, 0));
        assert!(!layout.is_enabled(0, 2));
        assert_eq!(layout, spec.layout());
    }
}
