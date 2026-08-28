//! Optimization schedules and convergence aids (roadmap Phase 20).
//!
//! Motivated by a measurement from this repository's own runs: the block owning
//! the lowest-sigma window shows a mean loss around **1900** against roughly
//! **13** for the highest-sigma block — a ~140x imbalance driven entirely by the
//! EDM weight `w(sigma) = (sigma^2 + sigma_d^2) / (sigma sigma_d)^2` exploding
//! as `sigma -> sigma_min`. That is what the objective asks for, but it means
//! the blocks receive wildly different gradient scales, and the run is
//! effectively tuned for whichever block happens to dominate.
//!
//! Four independent levers live here:
//!
//! - [`LrSchedule`] — the learning rate was previously **constant**, which is
//!   the cheapest thing to fix in the whole crate.
//! - [`GradientAccumulator`] — a larger effective batch without the memory.
//! - [`clip_gradients`] — a complement to the existing skip gate: clipping
//!   rescales a step, skipping discards it, and they answer different questions.
//! - [`Ema`] — a shadow copy of the weights that is almost always a better
//!   evaluation model than the raw ones.
//! - [`LossScales`] — per-block loss normalization, which is the direct answer
//!   to the imbalance above.

use burn::{
    module::{AutodiffModule, Module, ModuleMapper, Param},
    optim::GradientsParams,
    tensor::{backend::AutodiffBackend, backend::Backend, Tensor},
};

/// How the learning rate varies over a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LrSchedule {
    /// What the crate did before this module existed.
    Constant { lr: f64 },
    /// Linear warmup then cosine decay to `min_lr`.
    ///
    /// Warmup matters more than usual here: block-wise training visits one
    /// block per step, so early large steps land on an essentially untouched
    /// block and are pure noise.
    WarmupCosine {
        peak: f64,
        min_lr: f64,
        warmup_steps: usize,
        total_steps: usize,
    },
    /// Linear warmup then a constant plateau.
    WarmupConstant { peak: f64, warmup_steps: usize },
}

impl Default for LrSchedule {
    fn default() -> Self {
        Self::Constant { lr: 1e-3 }
    }
}

impl LrSchedule {
    /// The standard recipe: 5% of the run spent warming up, decaying to 1% of
    /// the peak.
    pub fn cosine(peak: f64, total_steps: usize) -> Self {
        Self::WarmupCosine {
            peak,
            min_lr: peak * 0.01,
            warmup_steps: (total_steps / 20).max(1),
            total_steps: total_steps.max(1),
        }
    }

    pub fn parse(name: &str, peak: f64, total_steps: usize) -> anyhow::Result<Self> {
        match name {
            "constant" => Ok(Self::Constant { lr: peak }),
            "cosine" => Ok(Self::cosine(peak, total_steps)),
            "warmup" => Ok(Self::WarmupConstant {
                peak,
                warmup_steps: (total_steps / 20).max(1),
            }),
            other => anyhow::bail!(
                "unknown lr schedule '{other}' (expected constant|cosine|warmup)"
            ),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Constant { .. } => "constant",
            Self::WarmupCosine { .. } => "cosine",
            Self::WarmupConstant { .. } => "warmup",
        }
    }

    /// Learning rate at `step`.
    pub fn at(&self, step: usize) -> f64 {
        match *self {
            Self::Constant { lr } => lr,
            Self::WarmupConstant { peak, warmup_steps } => peak * warmup_factor(step, warmup_steps),
            Self::WarmupCosine { peak, min_lr, warmup_steps, total_steps } => {
                if step < warmup_steps {
                    return peak * warmup_factor(step, warmup_steps);
                }
                let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
                let progress = ((step - warmup_steps) as f64 / decay_steps as f64).min(1.0);
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                min_lr + (peak - min_lr) * cosine
            }
        }
    }

    /// Largest rate the schedule will ever return.
    pub fn peak(&self) -> f64 {
        match *self {
            Self::Constant { lr } => lr,
            Self::WarmupCosine { peak, .. } | Self::WarmupConstant { peak, .. } => peak,
        }
    }
}

/// Warmup ramps over `[0, warmup_steps)` and reaches 1 at `warmup_steps`.
///
/// Step 0 gets a non-zero rate: a first step of exactly zero wastes a batch
/// and, more subtly, leaves the optimizer's moment estimates untouched so the
/// second step behaves like a first one anyway.
fn warmup_factor(step: usize, warmup_steps: usize) -> f64 {
    if warmup_steps == 0 {
        return 1.0;
    }
    ((step + 1) as f64 / warmup_steps as f64).min(1.0)
}

/// Accumulate gradients over several micro-batches before stepping.
///
/// Averaging rather than summing keeps the gradient magnitude independent of
/// the accumulation count, so the learning rate does not need retuning when it
/// changes.
#[derive(Debug)]
pub struct GradientAccumulator {
    every: usize,
    pending: usize,
    /// Running sum of the micro-batch gradients folded in so far.
    ///
    /// This is the whole point of the type, and its absence was a real bug:
    /// counting micro-batches and stepping on every k-th one *discards* the
    /// other k-1 gradients. That is not accumulation, it is a k-fold reduction
    /// in the data each step sees.
    buffer: Option<GradientsParams>,
    /// Micro-batches actually folded into `buffer`. Differs from `pending`
    /// when a step is rejected mid-cycle.
    folded: usize,
}

impl GradientAccumulator {
    pub fn new(every: usize) -> Self {
        Self { every: every.max(1), pending: 0, buffer: None, folded: 0 }
    }

    pub fn every(&self) -> usize {
        self.every
    }

    pub fn pending(&self) -> usize {
        self.pending
    }

    /// Micro-batches folded into the current cycle.
    pub fn folded(&self) -> usize {
        self.folded
    }

    /// Fold one micro-batch's gradients in and advance the cycle.
    ///
    /// The caller should have scaled the loss by [`Self::loss_scale`] before
    /// the backward pass, so the sum returned when the cycle completes is the
    /// gradient of the mean over all `k * batch_size` samples — exactly what
    /// one `k`-times-larger batch would produce.
    pub fn fold<B, M>(&mut self, grads: GradientsParams, module: &M) -> Cycle
    where
        B: AutodiffBackend<FloatElem = f32>,
        M: AutodiffModule<B>,
    {
        self.buffer = Some(match self.buffer.take() {
            None => grads,
            Some(mut acc) => {
                let mut visitor = SumVisitor::<B> {
                    into: &mut acc,
                    from: grads,
                    _backend: std::marker::PhantomData,
                };
                module.visit(&mut visitor);
                acc
            }
        });
        self.folded += 1;
        self.advance()
    }

    /// Advance the cycle *without* folding anything in.
    ///
    /// Used when a micro-batch is rejected by a quality gate: its gradients are
    /// not trustworthy, but the cycle should still complete rather than stall
    /// forever waiting for a batch that may never come.
    pub fn skip(&mut self) -> Cycle {
        self.advance()
    }

    fn advance(&mut self) -> Cycle {
        self.pending += 1;
        if self.pending < self.every {
            return Cycle::Filling;
        }
        self.pending = 0;
        self.folded = 0;
        Cycle::Ready(self.buffer.take())
    }

    /// Scale a micro-batch's loss so the accumulated total is the mean.
    pub fn loss_scale(&self) -> f64 {
        1.0 / self.every as f64
    }

    /// Whether anything is buffered (a run ending mid-cycle discards it).
    pub fn has_pending(&self) -> bool {
        self.pending > 0
    }
}

/// What one micro-batch did to the accumulation cycle.
///
/// The two states are kept distinct because "the cycle completed" and "there
/// are gradients to step with" are different facts: a cycle in which every
/// micro-batch was rejected by a quality gate completes with nothing to apply.
/// Collapsing them into a single `Option` made the cadence unobservable, which
/// is how the first version of the accumulation certificate ended up unable to
/// see whether the cycle fired at all.
#[derive(Debug)]
pub enum Cycle {
    /// Still filling; nothing to do.
    Filling,
    /// The cycle completed. `Some` carries the summed gradients; `None` means
    /// every micro-batch in it was rejected.
    Ready(Option<GradientsParams>),
}

impl Cycle {
    /// Whether the cycle completed, regardless of whether anything survived.
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    /// The summed gradients, if the cycle completed with any.
    pub fn into_gradients(self) -> Option<GradientsParams> {
        match self {
            Self::Filling => None,
            Self::Ready(grads) => grads,
        }
    }
}

/// Adds one `GradientsParams` into another, parameter by parameter.
///
/// A visitor rather than a loop over ids because the rank `D` differs per
/// parameter and only the module traversal knows it — the same reason
/// `ClipVisitor` and `GradNormVisitor` are written this way.
struct SumVisitor<'a, B: AutodiffBackend> {
    into: &'a mut GradientsParams,
    from: GradientsParams,
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend<FloatElem = f32>> burn::module::ModuleVisitor<B> for SumVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(addend) = self.from.remove::<B::InnerBackend, D>(param.id) else {
            return;
        };
        // A parameter absent from the accumulator has no gradient yet -- that
        // happens legitimately, since block-wise training only touches the
        // executed span -- so the addend becomes its first contribution.
        let total = match self.into.remove::<B::InnerBackend, D>(param.id) {
            Some(existing) => existing + addend,
            None => addend,
        };
        self.into.register::<B::InnerBackend, D>(param.id, total);
    }
}

/// How the auxiliary balance-loss weight evolves over a run.
///
/// # Why this is not just a constant
///
/// The Switch balance loss is computed over whatever batch the forward pass
/// saw. Zhu et al. (*Demons in the Detail*, 2025) show that computing it per
/// **micro-batch** pushes the router to distribute tokens evenly *within each
/// batch* — and since a micro-batch holds few distinct inputs, that pressure
/// lands on individual sequences and **actively inhibits expert
/// specialization**. Their fix is to compute the loss over the global batch,
/// which needs the per-expert counts to escape the forward pass; this crate
/// cannot do that yet (see the roadmap note).
///
/// Annealing the *weight* is the tractable half of the same idea. Early in a
/// run the balance loss is doing the job it is good at — preventing routing
/// collapse, where one expert wins everything and the rest never receive
/// gradient. Once every expert is alive, that pressure has nothing left to buy
/// and starts costing specialization instead. So: hold it high while collapse
/// is the risk, decay it once it is not.
///
/// [`Self::Constant`] reproduces the old behaviour exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BalanceSchedule {
    /// One weight for the whole run.
    Constant { weight: f64 },
    /// `start` until `hold_steps`, then decayed geometrically to `end` by
    /// `total_steps`, then held at `end`.
    ///
    /// Geometric rather than linear because the useful range spans orders of
    /// magnitude — a typical run wants `1e-2` early and `1e-4` late, and a
    /// linear ramp between those spends almost all of its steps near the top.
    Anneal {
        start: f64,
        end: f64,
        hold_steps: usize,
        total_steps: usize,
    },
}

impl Default for BalanceSchedule {
    fn default() -> Self {
        Self::Constant { weight: 0.01 }
    }
}

impl BalanceSchedule {
    pub fn constant(weight: f64) -> Self {
        Self::Constant { weight }
    }

    /// Hold `start` for the first tenth of the run, then decay to `end`.
    pub fn anneal(start: f64, end: f64, total_steps: usize) -> Self {
        Self::Anneal {
            start,
            end,
            hold_steps: total_steps / 10,
            total_steps: total_steps.max(1),
        }
    }

    pub fn parse(name: &str, weight: f64, total_steps: usize) -> anyhow::Result<Self> {
        match name {
            "constant" => Ok(Self::constant(weight)),
            // The end point is two decades below the start, which is the range
            // the literature reports as "enough to prevent collapse" versus
            // "small enough not to fight the task loss".
            "anneal" => Ok(Self::anneal(weight, weight * 0.01, total_steps)),
            other => anyhow::bail!("unknown balance schedule '{other}' (expected constant|anneal)"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Constant { .. } => "constant",
            Self::Anneal { .. } => "anneal",
        }
    }

    /// Weight at `step`.
    pub fn at(&self, step: usize) -> f64 {
        match *self {
            Self::Constant { weight } => weight,
            Self::Anneal { start, end, hold_steps, total_steps } => {
                if step <= hold_steps {
                    return start;
                }
                let decay_steps = total_steps.saturating_sub(hold_steps).max(1);
                let progress = ((step - hold_steps) as f64 / decay_steps as f64).min(1.0);
                // Geometric interpolation. Guarded because a zero endpoint has
                // no logarithm, and "decay all the way off" is a reasonable
                // thing to ask for.
                if start <= 0.0 || end <= 0.0 {
                    return start + (end - start) * progress;
                }
                (start.ln() + (end.ln() - start.ln()) * progress).exp()
            }
        }
    }

    /// Largest weight the schedule will ever return.
    pub fn peak(&self) -> f64 {
        match *self {
            Self::Constant { weight } => weight,
            Self::Anneal { start, end, .. } => start.max(end),
        }
    }
}

/// Rescale every gradient so their global L2 norm is at most `max_norm`.
///
/// Returns the scale that was applied (`1.0` when no clipping was needed).
///
/// Clipping and the existing skip gate are complements, not alternatives:
/// clipping keeps a step and shrinks it, skipping discards it entirely. In a
/// block-wise loop a *clipped* bad step still writes to that block's
/// parameters, so the gate remains the right tool for a pathological step and
/// clipping the right one for a merely large one.
pub fn clip_gradients<B, M>(
    grads: &mut GradientsParams,
    module: &M,
    total_norm: f32,
    max_norm: f32,
) -> f32
where
    B: AutodiffBackend<FloatElem = f32>,
    M: AutodiffModule<B>,
{
    if !total_norm.is_finite() || total_norm <= max_norm || max_norm <= 0.0 {
        return 1.0;
    }
    let scale = max_norm / total_norm;

    let mut visitor = ClipVisitor::<B> { grads, scale, _backend: std::marker::PhantomData };
    module.visit(&mut visitor);
    scale
}

struct ClipVisitor<'a, B: AutodiffBackend> {
    grads: &'a mut GradientsParams,
    scale: f32,
    _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend<FloatElem = f32>> burn::module::ModuleVisitor<B> for ClipVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
            self.grads
                .register::<B::InnerBackend, D>(param.id, grad.mul_scalar(self.scale));
        }
    }
}

/// Exponential moving average of a module's parameters.
///
/// `shadow <- decay * shadow + (1 - decay) * live`. The averaged weights are
/// usually a better evaluation model than the live ones, because they sit
/// nearer the centre of the basin the optimizer is bouncing around in — which
/// matters more here than usual, since each step moves only one block.
#[derive(Debug, Clone)]
pub struct Ema<M> {
    shadow: M,
    decay: f64,
    updates: usize,
}

impl<M: Clone> Ema<M> {
    /// Start from a copy of the current weights.
    ///
    /// `decay` is clamped to `[0, 1]`: outside that range the recursion either
    /// diverges or inverts the sign of the history.
    pub fn new(module: &M, decay: f64) -> Self {
        Self { shadow: module.clone(), decay: decay.clamp(0.0, 1.0), updates: 0 }
    }

    pub fn decay(&self) -> f64 {
        self.decay
    }

    pub fn updates(&self) -> usize {
        self.updates
    }

    pub fn shadow(&self) -> &M {
        &self.shadow
    }

    pub fn into_shadow(self) -> M {
        self.shadow
    }

    /// Effective decay at the current step, with bias correction.
    ///
    /// Early on the shadow is still mostly its initialization, so the nominal
    /// decay would leave it lagging badly. Ramping it in as
    /// `min(decay, (1 + t) / (10 + t))` is the standard correction and makes
    /// the first few hundred steps usable rather than misleading.
    pub fn effective_decay(&self) -> f64 {
        let warm = (1.0 + self.updates as f64) / (10.0 + self.updates as f64);
        self.decay.min(warm)
    }
}

impl<M: Clone> Ema<M> {
    /// Fold the live weights into the shadow.
    ///
    /// The backend is a method-level parameter rather than a field: an `Ema`
    /// is just a shadow copy plus two scalars, and threading a `PhantomData<B>`
    /// through it would make every use site spell the backend twice.
    pub fn update<B>(&mut self, live: &M)
    where
        B: Backend<FloatElem = f32>,
        M: Module<B>,
    {
        let decay = self.effective_decay();
        let live_params = LiveParams::collect(live);
        let expected = live_params.values.len();
        let mut mapper = EmaMapper::<B> {
            live: live_params,
            cursor: 0,
            decay: decay as f32,
            _backend: std::marker::PhantomData,
        };
        let shadow = std::mem::replace(&mut self.shadow, live.clone());
        self.shadow = shadow.map(&mut mapper);
        assert_eq!(
            mapper.cursor, expected,
            "EMA visited {} shadow parameters but the live module has {expected}",
            mapper.cursor
        );
        self.updates += 1;
    }
}

/// Live parameter values in traversal order.
///
/// Pairing by **order**, not by `ParamId`. Burn's `load_record` adopts the
/// record's ids, so an id-keyed pairing would silently stop matching the moment
/// either side came from a checkpoint — the EMA would quietly freeze and
/// nothing would look wrong. Traversal order is derived from field declaration
/// order and is identical for two modules of the same type, so it survives that.
/// The shape check in the mapper turns any residual mismatch into a panic
/// rather than a wrong blend.
struct LiveParams {
    values: Vec<burn::tensor::TensorData>,
}

impl LiveParams {
    fn collect<B: Backend<FloatElem = f32>, M: Module<B>>(module: &M) -> Self {
        let mut visitor = LiveCollector { values: Vec::new() };
        module.visit(&mut visitor);
        Self { values: visitor.values }
    }
}

struct LiveCollector {
    values: Vec<burn::tensor::TensorData>,
}

impl<B: Backend<FloatElem = f32>> burn::module::ModuleVisitor<B> for LiveCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.values.push(param.val().into_data());
    }
}

struct EmaMapper<B: Backend> {
    live: LiveParams,
    cursor: usize,
    decay: f32,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend<FloatElem = f32>> ModuleMapper<B> for EmaMapper<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let Some(data) = self.live.values.get(self.cursor) else {
            return param;
        };
        self.cursor += 1;

        let current = param.val();
        assert_eq!(
            current.shape(),
            data.shape.clone(),
            "EMA shadow and live module diverged in traversal order at parameter {}",
            self.cursor - 1
        );

        let live = Tensor::<B, D>::from_data(data.clone(), &current.device());
        let blended = current.mul_scalar(self.decay) + live.mul_scalar(1.0 - self.decay);
        Param::from_tensor(blended.detach()).set_require_grad(false)
    }
}

/// Per-block loss normalization (roadmap 20.7).
///
/// Tracks a running mean of each block's loss and rescales future losses by its
/// inverse, so no block dominates the gradient purely because the EDM weight is
/// large in its sigma window. The scale is bounded in both directions: an
/// unbounded reciprocal would let a block whose loss briefly collapses acquire
/// an enormous weight.
#[derive(Debug, Clone)]
pub struct LossScales {
    means: Vec<f64>,
    counts: Vec<usize>,
    momentum: f64,
    min_scale: f64,
    max_scale: f64,
}

impl LossScales {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            means: vec![0.0; num_blocks],
            counts: vec![0; num_blocks],
            momentum: 0.99,
            min_scale: 1e-3,
            max_scale: 1e3,
        }
    }

    pub fn with_bounds(mut self, min_scale: f64, max_scale: f64) -> Self {
        self.min_scale = min_scale;
        self.max_scale = max_scale;
        self
    }

    /// Fold one observation in.
    pub fn observe(&mut self, block: usize, loss: f32) {
        if block >= self.means.len() || !loss.is_finite() {
            return;
        }
        let loss = loss as f64;
        if self.counts[block] == 0 {
            self.means[block] = loss;
        } else {
            self.means[block] = self.momentum * self.means[block] + (1.0 - self.momentum) * loss;
        }
        self.counts[block] += 1;
    }

    pub fn mean(&self, block: usize) -> f64 {
        self.means.get(block).copied().unwrap_or(0.0)
    }

    /// Multiplier that brings `block`'s loss onto the geometric mean scale of
    /// all observed blocks.
    ///
    /// The *geometric* mean, not the arithmetic one: the imbalance spans orders
    /// of magnitude, and an arithmetic mean would be pinned to whichever block
    /// is largest.
    pub fn scale(&self, block: usize) -> f64 {
        let observed: Vec<f64> = self
            .means
            .iter()
            .zip(&self.counts)
            .filter(|(m, c)| **c > 0 && **m > 0.0)
            .map(|(m, _)| *m)
            .collect();
        if observed.len() < 2 || self.counts.get(block).is_none_or(|c| *c == 0) {
            return 1.0;
        }
        let own = self.means[block];
        if own <= 0.0 {
            return 1.0;
        }
        let log_mean: f64 =
            observed.iter().map(|m| m.ln()).sum::<f64>() / observed.len() as f64;
        (log_mean.exp() / own).clamp(self.min_scale, self.max_scale)
    }

    pub fn num_blocks(&self) -> usize {
        self.means.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_constant_schedule_is_constant() {
        let s = LrSchedule::Constant { lr: 3e-4 };
        for step in [0usize, 1, 100, 10_000] {
            assert_eq!(s.at(step), 3e-4);
        }
        assert_eq!(s.name(), "constant");
    }

    #[test]
    fn test_warmup_ramps_then_holds() {
        let s = LrSchedule::WarmupConstant { peak: 1.0, warmup_steps: 4 };
        // Step 0 is non-zero: a first step of exactly zero wastes a batch and
        // leaves the optimizer's moments untouched.
        assert!(s.at(0) > 0.0);
        assert!(s.at(0) < s.at(1) && s.at(1) < s.at(2));
        assert_eq!(s.at(3), 1.0);
        assert_eq!(s.at(50), 1.0, "the plateau must hold");

        // Zero warmup is immediate, not a division by zero.
        let none = LrSchedule::WarmupConstant { peak: 2.0, warmup_steps: 0 };
        assert_eq!(none.at(0), 2.0);
    }

    #[test]
    fn test_cosine_decays_monotonically_after_warmup() {
        let s = LrSchedule::cosine(1e-3, 1000);
        let LrSchedule::WarmupCosine { warmup_steps, min_lr, .. } = s else {
            panic!("expected cosine");
        };
        assert_eq!(warmup_steps, 50);

        // Peak at the end of warmup.
        assert!((s.at(warmup_steps) - 1e-3).abs() < 1e-12);

        // Strictly decreasing afterwards, ending at min_lr.
        let mut prev = f64::INFINITY;
        for step in (warmup_steps..=1000).step_by(25) {
            let lr = s.at(step);
            assert!(lr < prev, "cosine must decay: {lr} !< {prev} at {step}");
            assert!(lr >= min_lr - 1e-15, "must not undershoot min_lr");
            prev = lr;
        }
        assert!((s.at(1000) - min_lr).abs() < 1e-12);
        // Clamped past the horizon rather than turning back up.
        assert!((s.at(5000) - min_lr).abs() < 1e-12);
    }

    #[test]
    fn test_schedule_never_exceeds_its_peak() {
        // A schedule that overshoots would silently invalidate any LR tuning
        // done against its nominal peak.
        for s in [
            LrSchedule::cosine(1e-3, 500),
            LrSchedule::WarmupConstant { peak: 5e-4, warmup_steps: 10 },
            LrSchedule::Constant { lr: 2e-3 },
        ] {
            for step in 0..600 {
                let lr = s.at(step);
                assert!(lr <= s.peak() + 1e-15, "{} overshot at {step}: {lr}", s.name());
                assert!(lr >= 0.0);
            }
        }
    }

    #[test]
    fn test_schedule_parsing() {
        assert_eq!(LrSchedule::parse("constant", 1e-3, 100).unwrap().name(), "constant");
        assert_eq!(LrSchedule::parse("cosine", 1e-3, 100).unwrap().name(), "cosine");
        assert_eq!(LrSchedule::parse("warmup", 1e-3, 100).unwrap().name(), "warmup");
        assert!(LrSchedule::parse("triangular", 1e-3, 100).is_err());
    }

    #[test]
    fn test_accumulator_fires_every_n_steps() {
        let mut acc = GradientAccumulator::new(3);
        assert!(!acc.skip().is_ready());
        assert!(!acc.skip().is_ready());
        assert!(acc.skip().is_ready(), "third micro-batch must trigger a step");
        assert!(!acc.has_pending(), "the cycle resets");
        assert!((acc.loss_scale() - 1.0 / 3.0).abs() < 1e-12);

        // An accumulation count of zero is meaningless; clamp to stepping every
        // time rather than never.
        let mut every = GradientAccumulator::new(0);
        assert_eq!(every.every(), 1);
        assert!(every.skip().is_ready());
    }

    #[test]
    fn test_a_completed_cycle_with_nothing_folded_yields_no_gradients() {
        // "The cycle completed" and "there are gradients to apply" are
        // different facts. A cycle in which every micro-batch was rejected by a
        // quality gate must still complete -- otherwise one persistently bad
        // block stalls the run forever -- but it has nothing to step with.
        let mut acc = GradientAccumulator::new(2);
        assert!(!acc.skip().is_ready());
        let cycle = acc.skip();
        assert!(cycle.is_ready(), "the cycle must complete even with nothing folded");
        assert!(cycle.into_gradients().is_none(), "but there is nothing to apply");
    }

    #[test]
    fn test_folding_actually_sums_the_gradients() {
        // The bug this type was written to have and did not: counting
        // micro-batches and stepping on every k-th one silently discards the
        // other k-1 gradients. Folding two identical gradients must give twice
        // one of them, not one of them.
        use burn::backend::{Autodiff, NdArray};
        use burn::nn::LinearConfig;
        use burn::tensor::Distribution;

        type A = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let layer = LinearConfig::new(4, 3).with_bias(false).init::<A>(&device);
        let x = Tensor::<A, 2>::random([5, 4], Distribution::Uniform(-1.0, 1.0), &device);

        let grads_of = || {
            let loss = layer.forward(x.clone()).powf_scalar(2.0).mean();
            GradientsParams::from_grads(loss.backward(), &layer)
        };

        let one: Vec<f32> = grads_of()
            .get::<NdArray<f32>, 2>(layer.weight.id)
            .expect("weight gradient")
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();

        let mut acc = GradientAccumulator::new(2);
        assert!(!acc.fold(grads_of(), &layer).is_ready());
        let summed = acc
            .fold(grads_of(), &layer)
            .into_gradients()
            .expect("a completed cycle with two folds has gradients");
        let two: Vec<f32> = summed
            .get::<NdArray<f32>, 2>(layer.weight.id)
            .expect("weight gradient")
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();

        for (a, b) in two.iter().zip(&one) {
            assert!(
                (a - 2.0 * b).abs() < 1e-6,
                "folding two copies must double the gradient: {a} vs {}",
                2.0 * b
            );
        }
    }

    #[test]
    fn test_accumulated_loss_scale_preserves_magnitude() {
        // Averaging rather than summing is what keeps the learning rate valid
        // when the accumulation count changes.
        for every in [1usize, 2, 8] {
            let acc = GradientAccumulator::new(every);
            let total: f64 = (0..every).map(|_| 4.0 * acc.loss_scale()).sum();
            assert!((total - 4.0).abs() < 1e-12, "accumulating {every} micro-batches");
        }
    }

    #[test]
    fn test_loss_scales_equalize_across_blocks() {
        // The measured problem: the lowest-sigma block's loss is ~140x the
        // highest-sigma block's, purely from the EDM weight.
        let mut scales = LossScales::new(3);
        for _ in 0..200 {
            scales.observe(0, 13.4);
            scales.observe(1, 29.3);
            scales.observe(2, 1909.8);
        }

        let scaled: Vec<f64> = (0..3).map(|b| scales.scale(b) * scales.mean(b)).collect();
        // After scaling, every block sits on the same (geometric-mean) scale.
        let spread = scaled.iter().cloned().fold(0.0f64, f64::max)
            / scaled.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(spread < 1.01, "blocks should be equalized, spread {spread}");

        // The block that was too loud is turned down, the quiet one up.
        assert!(scales.scale(2) < 1.0, "the loud block must be attenuated");
        assert!(scales.scale(0) > 1.0, "the quiet block must be amplified");
    }

    #[test]
    fn test_loss_scales_are_bounded_and_safe() {
        let mut scales = LossScales::new(2).with_bounds(0.1, 10.0);
        // A single observed block has nothing to be normalized against.
        scales.observe(0, 5.0);
        assert_eq!(scales.scale(0), 1.0);
        assert_eq!(scales.scale(1), 1.0, "an unobserved block is left alone");

        // An extreme ratio is clamped rather than producing a huge multiplier.
        for _ in 0..100 {
            scales.observe(0, 1e-8);
            scales.observe(1, 1e8);
        }
        assert!((0.1..=10.0).contains(&scales.scale(0)));
        assert!((0.1..=10.0).contains(&scales.scale(1)));

        // Non-finite and out-of-range observations are ignored, not recorded.
        let before = scales.mean(0);
        scales.observe(0, f32::NAN);
        scales.observe(99, 1.0);
        assert_eq!(scales.mean(0), before);
    }

    #[test]
    fn test_clipping_rescales_to_the_bound_and_preserves_direction() {
        use crate::dblock::{DblockClassifier, DblockConfig};
        use crate::train::DefaultTrainBackend as A;
        use crate::vit::ViTDiTConfig;
        use burn::tensor::Distribution;

        let device = Default::default();
        let model = DblockClassifier::<A>::new(
            &ViTDiTConfig::tiny(10),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        let pixels =
            Tensor::<A, 4>::random([2, 3, 32, 32], Distribution::Uniform(-0.5, 0.5), &device);
        let labels = Tensor::<A, 1, burn::tensor::Int>::from_ints([1i64, 4].as_slice(), &device);
        let mut rng = rand::rngs::StdRng::from_seed([7u8; 32]);

        let (loss, _) = model.training_step(pixels, labels, 0.05, &mut rng);
        let mut grads = GradientsParams::from_grads(loss.backward(), &model);
        let before = crate::quality::global_grad_norm(&model, &grads);
        assert!(before > 0.0);

        // A bound far below the norm must rescale to exactly the bound.
        let max_norm = before / 4.0;
        let scale = clip_gradients(&mut grads, &model, before, max_norm);
        assert!((scale - 0.25).abs() < 1e-5, "scale should be max/total, got {scale}");
        let after = crate::quality::global_grad_norm(&model, &grads);
        assert!(
            (after - max_norm).abs() < max_norm * 1e-3,
            "clipped norm {after} should equal the bound {max_norm}"
        );

        // Already inside the bound: untouched, and reported as such.
        let mut grads2 = GradientsParams::from_grads(
            model
                .training_step(
                    Tensor::<A, 4>::zeros([2, 3, 32, 32], &device),
                    Tensor::<A, 1, burn::tensor::Int>::from_ints([0i64, 1].as_slice(), &device),
                    0.05,
                    &mut rng,
                )
                .0
                .backward(),
            &model,
        );
        let n = crate::quality::global_grad_norm(&model, &grads2);
        assert_eq!(clip_gradients(&mut grads2, &model, n, n * 10.0), 1.0);
        // A non-finite norm must not produce a NaN scale.
        assert_eq!(clip_gradients(&mut grads2, &model, f32::NAN, 1.0), 1.0);
    }

    #[test]
    fn test_ema_tracks_then_lags_the_live_weights() {
        use crate::dblock::{DblockClassifier, DblockConfig};
        use crate::vit::ViTDiTConfig;
        use burn::backend::NdArray;

        type N = NdArray<f32>;
        let device = Default::default();
        let model = DblockClassifier::<N>::new(
            &ViTDiTConfig::tiny(10),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        crate::tensor_ext::force_initialization(&model);

        let mut ema = Ema::new(&model, 0.99);
        let table = |m: &DblockClassifier<N>| -> Tensor<N, 2> {
            m.model().label_embedding_weight()
        };

        // A fresh shadow is exactly the model.
        assert_eq!(
            (table(ema.shadow()) - table(&model)).abs().max().into_scalar(),
            0.0
        );

        // Move the live weights, then fold them in. The shadow must land
        // strictly between where it was and where the model is -- that is what
        // an average is.
        let mut moved = model.clone().into_record();
        let shifted = moved.model.vit.embeddings.label_embeddings.weight.val() + 1.0;
        moved.model.vit.embeddings.label_embeddings.weight =
            burn::module::Param::from_tensor(shifted.detach());
        let moved = DblockClassifier::<N>::new(
            &ViTDiTConfig::tiny(10),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        )
        .load_record(moved);

        ema.update::<N>(&moved);
        assert_eq!(ema.updates(), 1);

        let gap_to_old = (table(ema.shadow()) - table(&model)).abs().max().into_scalar();
        let gap_to_new = (table(ema.shadow()) - table(&moved)).abs().max().into_scalar();
        assert!(gap_to_old > 0.0 && gap_to_new > 0.0, "must sit between the two");
        assert!(
            gap_to_old + gap_to_new <= 1.0 + 1e-4,
            "the shadow must lie on the segment: {gap_to_old} + {gap_to_new}"
        );

        // Repeated folding converges toward the live weights.
        for _ in 0..40 {
            ema.update::<N>(&moved);
        }
        let converged = (table(ema.shadow()) - table(&moved)).abs().max().into_scalar();
        assert!(converged < gap_to_new, "the shadow must approach the live weights");
    }

    #[test]
    fn test_ema_survives_a_checkpoint_round_trip() {
        // Regression guard. An earlier version paired shadow and live
        // parameters by `ParamId`, which `load_record` reassigns -- so after a
        // resume the EMA silently stopped updating and nothing looked wrong.
        // Pairing by traversal order fixes it, and this pins the behaviour.
        use crate::dblock::{DblockClassifier, DblockConfig};
        use crate::vit::ViTDiTConfig;
        use burn::backend::NdArray;

        type N = NdArray<f32>;
        let device = Default::default();
        let cfg = ViTDiTConfig::tiny(10);
        let db = DblockConfig { num_blocks: 2, ..DblockConfig::default() };

        let model = DblockClassifier::<N>::new(&cfg, &db, &device);
        crate::tensor_ext::force_initialization(&model);
        let mut ema = Ema::new(&model, 0.5);

        // Round-trip the live model through a record, which reassigns ids.
        let reloaded =
            DblockClassifier::<N>::new(&cfg, &db, &device).load_record(model.clone().into_record());

        // Perturb it so an update has something to move toward.
        let mut rec = reloaded.into_record();
        let shifted = rec.model.vit.embeddings.label_embeddings.weight.val() + 2.0;
        rec.model.vit.embeddings.label_embeddings.weight =
            burn::module::Param::from_tensor(shifted.detach());
        let reloaded = DblockClassifier::<N>::new(&cfg, &db, &device).load_record(rec);

        let before = ema.shadow().model().label_embedding_weight();
        ema.update::<N>(&reloaded);
        let after = ema.shadow().model().label_embedding_weight();

        let moved = (after - before).abs().max().into_scalar();
        assert!(
            moved > 0.0,
            "the EMA must still track a model whose ParamIds were reassigned"
        );
    }

    #[test]
    fn test_ema_decay_is_clamped_and_bias_corrected() {
        // Bare integers stand in for a module here; the arithmetic is what
        // matters and `Ema::new` only needs `Clone`.
        let ema = Ema::new(&1.0f64, 5.0);
        assert_eq!(ema.decay(), 1.0, "decay above 1 would diverge");
        assert_eq!(Ema::new(&1.0f64, -3.0).decay(), 0.0);

        // Early updates use a ramped decay so the shadow is not stuck on its
        // initialization.
        let ema = Ema::new(&1.0f64, 0.999);
        assert!(ema.effective_decay() < 0.2, "first update must track closely");
        let mut later = Ema::new(&1.0f64, 0.999);
        later.updates = 10_000;
        assert!((later.effective_decay() - 0.999).abs() < 1e-9);
    }

    #[test]
    fn test_a_constant_balance_schedule_never_moves() {
        // Containment: the default must reproduce exactly what every run did
        // before the schedule existed.
        let schedule = BalanceSchedule::constant(0.01);
        for step in [0usize, 1, 500, 100_000] {
            assert_eq!(schedule.at(step), 0.01);
        }
        assert_eq!(schedule.peak(), 0.01);
    }

    #[test]
    fn test_annealing_holds_then_decays_monotonically() {
        // The shape the design argues for: hold while routing collapse is the
        // risk, then decay once every expert is alive and the pressure is only
        // costing specialization.
        let total = 1000usize;
        let schedule = BalanceSchedule::anneal(1e-2, 1e-4, total);

        // Held over the first tenth.
        for step in [0usize, 50, 100] {
            assert!((schedule.at(step) - 1e-2).abs() < 1e-12, "step {step} should hold");
        }

        // Then strictly decreasing, and never above the start or below the end.
        let mut previous = schedule.at(100);
        for step in 101..=total {
            let w = schedule.at(step);
            assert!(w <= previous + 1e-15, "step {step}: {w} > {previous}");
            assert!((1e-4 - 1e-12..=1e-2 + 1e-12).contains(&w), "step {step}: {w} out of range");
            previous = w;
        }
        assert!((schedule.at(total) - 1e-4).abs() < 1e-9, "should land on the endpoint");

        // ...and stays there past the end rather than continuing down.
        assert!((schedule.at(total * 3) - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn test_annealing_is_geometric_not_linear() {
        // The useful range spans two decades. A linear ramp spends almost all
        // of its steps near the top, which defeats the purpose -- the midpoint
        // is the tell.
        let total = 1000usize;
        let schedule = BalanceSchedule::anneal(1e-2, 1e-4, total);
        let hold = total / 10;
        let midpoint = schedule.at(hold + (total - hold) / 2);

        let geometric = (1e-2f64 * 1e-4).sqrt(); // 1e-3
        let linear = (1e-2 + 1e-4) / 2.0; // ~5e-3
        assert!(
            (midpoint - geometric).abs() < geometric * 0.05,
            "midpoint {midpoint} should be the geometric mean {geometric}"
        );
        assert!(midpoint < linear / 2.0, "and nowhere near the arithmetic mean");
    }

    #[test]
    fn test_a_zero_endpoint_falls_back_to_linear() {
        // "Decay the balance loss all the way off" is a reasonable request and
        // zero has no logarithm, so the geometric path must not produce a NaN.
        let schedule = BalanceSchedule::anneal(1e-2, 0.0, 100);
        for step in 0..=200 {
            let w = schedule.at(step);
            assert!(w.is_finite(), "step {step} gave {w}");
            assert!((0.0..=1e-2 + 1e-12).contains(&w), "step {step} gave {w}");
        }
        assert!(schedule.at(200).abs() < 1e-12, "it should actually reach zero");
    }

    #[test]
    fn test_balance_schedule_parsing_round_trips() {
        assert_eq!(BalanceSchedule::parse("constant", 0.02, 100).unwrap().name(), "constant");
        assert_eq!(BalanceSchedule::parse("anneal", 0.02, 100).unwrap().name(), "anneal");
        assert!(BalanceSchedule::parse("nope", 0.02, 100).is_err());

        // Annealing ends two decades below where it starts.
        let annealed = BalanceSchedule::parse("anneal", 0.02, 100).unwrap();
        assert!((annealed.at(0) - 0.02).abs() < 1e-12);
        assert!((annealed.at(100) - 0.0002).abs() < 1e-9);
    }
}
