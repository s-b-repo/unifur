//! Hybrid loop-graph dynamic transformers (roadmap Phase 11).
//!
//! Instead of running every block once in a fixed order, a loop graph decides
//! *at run time* which blocks to execute, which to skip, and when to revisit
//! an earlier one, subject to a compute budget. Four behaviours, all driven by
//! per-sample signals rather than by the schedule:
//!
//! - **Skip** a block whose input already looks converged.
//! - **Loop back** to an earlier block when the current one leaves the
//!   estimate unconfident.
//! - **Exit early** once the accumulated halting probability is spent.
//! - **Stop** unconditionally at a compute budget.
//!
//! # Where the guarantees come from
//!
//! Dynamic control flow is easy to get subtly wrong -- non-terminating loops,
//! budgets that are exceeded, mixture weights that do not sum to one. The
//! decision logic is therefore kept in [`LoopPlanner`], a small host-side
//! state machine with no tensors in it, so its invariants can be stated and
//! tested directly:
//!
//! 1. it never authorizes more than `budget` block executions,
//! 2. it always terminates within `max_iterations` iterations,
//! 3. the ACT mixture weights it produces sum to exactly `1`.
//!
//! Invariant 3 is what makes the output a genuine convex combination of block
//! outputs: it is the "remainder" rule of Graves (2016), where the final step
//! is charged whatever probability mass is left rather than its own halting
//! probability.
//!
//! [`LoopGraph`] then applies those decisions to a
//! [`DblockClassifier`]'s block spans, and the result is an `x0` predictor
//! with the same signature as the fixed-depth one -- so it plugs straight into
//! [`crate::solver::integrate`].

use crate::{
    adaptive::{HaltingConfig, HaltingHead},
    dblock::DblockClassifier,
};
use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

/// Loop-graph execution policy.
#[derive(Debug, Clone, Copy)]
pub struct LoopGraphConfig {
    /// Hard cap on iterations, budget or not. Guarantees termination.
    pub max_iterations: usize,
    /// Stop once the accumulated halting probability reaches this.
    pub exit_threshold: f32,
    /// Maximum block executions; `None` means only `max_iterations` applies.
    pub budget: Option<usize>,
    /// Skip a block when the incoming confidence already exceeds this.
    pub skip_threshold: f32,
    /// Revisit an earlier block when confidence falls below this.
    pub loopback_threshold: f32,
    /// Cap on loop-backs per run, so refinement cannot livelock.
    pub max_loopbacks: usize,
}

impl Default for LoopGraphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 16,
            exit_threshold: 0.99,
            budget: None,
            skip_threshold: 0.995,
            loopback_threshold: 0.25,
            max_loopbacks: 2,
        }
    }
}

impl LoopGraphConfig {
    /// Straight-through execution: every block once, no skips or loops.
    /// Useful as the control condition in an ablation.
    pub fn feedforward(num_blocks: usize) -> Self {
        Self {
            max_iterations: num_blocks,
            exit_threshold: f32::INFINITY,
            budget: None,
            skip_threshold: f32::INFINITY,
            loopback_threshold: f32::NEG_INFINITY,
            max_loopbacks: 0,
        }
    }
}

/// What the planner decided for one iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Execute `block` and mix its output in with the returned weight scale.
    Run(usize),
    /// Skip `block` entirely.
    Skip(usize),
    /// Re-execute the earlier block `to` after `from`.
    LoopBack { from: usize, to: usize },
    /// Terminate the run.
    Stop,
}

/// Record of everything a run did, for analysis and for the block-usage
/// visualization of roadmap 8.7.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExecutionTrace {
    pub blocks_run: Vec<usize>,
    pub skipped: Vec<usize>,
    pub looped: Vec<(usize, usize)>,
    /// ACT mixture weight charged to each executed block, in run order.
    pub weights: Vec<f32>,
    pub iterations: usize,
}

impl ExecutionTrace {
    /// Block executions performed (loop-backs included).
    pub fn executions(&self) -> usize {
        self.blocks_run.len()
    }

    /// Sum of the mixture weights; `1.0` for any completed run.
    pub fn weight_mass(&self) -> f32 {
        self.weights.iter().sum()
    }

    /// How many times each block ran, indexed by block.
    pub fn usage_histogram(&self, num_blocks: usize) -> Vec<usize> {
        let mut counts = vec![0usize; num_blocks];
        for &b in &self.blocks_run {
            if b < num_blocks {
                counts[b] += 1;
            }
        }
        counts
    }
}

/// Host-side state machine deciding the execution order.
///
/// Kept free of tensors on purpose: the control flow is where the
/// correctness risk lives, so it is made directly testable.
#[derive(Debug, Clone)]
pub struct LoopPlanner {
    config: LoopGraphConfig,
    num_blocks: usize,
    /// Next block in the straight-through order.
    cursor: usize,
    executions: usize,
    iterations: usize,
    loopbacks: usize,
    /// Unspent halting probability (ACT remainder).
    remainder: f32,
    finished: bool,
}

impl LoopPlanner {
    pub fn new(config: LoopGraphConfig, num_blocks: usize) -> Self {
        Self {
            config,
            num_blocks,
            cursor: 0,
            executions: 0,
            iterations: 0,
            loopbacks: 0,
            remainder: 1.0,
            finished: num_blocks == 0,
        }
    }

    /// Whether the run has ended (budget spent, halting mass exhausted, or
    /// every block visited).
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Unspent ACT probability mass.
    pub fn remainder(&self) -> f32 {
        self.remainder
    }

    /// Next action given the confidence of the current estimate.
    ///
    /// Call once per iteration *before* running anything; the returned
    /// decision says what to do, and [`Self::charge`] then books the mixture
    /// weight for whatever was executed.
    pub fn next(&mut self, confidence: f32) -> Decision {
        if self.finished {
            return Decision::Stop;
        }
        self.iterations += 1;

        // Terminate on the hard caps first: these are the guarantees.
        if self.iterations > self.config.max_iterations
            || self.config.budget.is_some_and(|b| self.executions >= b)
        {
            self.finished = true;
            return Decision::Stop;
        }

        // Refinement: revisit the previous block when the estimate regressed.
        if self.cursor > 0
            && self.loopbacks < self.config.max_loopbacks
            && confidence < self.config.loopback_threshold
        {
            self.loopbacks += 1;
            let from = self.cursor - 1;
            let to = from.saturating_sub(1);
            return Decision::LoopBack { from, to };
        }

        if self.cursor >= self.num_blocks {
            self.finished = true;
            return Decision::Stop;
        }

        let block = self.cursor;
        self.cursor += 1;

        if confidence >= self.config.skip_threshold {
            return Decision::Skip(block);
        }
        Decision::Run(block)
    }

    /// Book the ACT weight for an executed block and report it.
    ///
    /// `halt_prob` is the block's halting probability. The weight is
    /// `p * remainder`, except on the final charge, where the *entire*
    /// remainder is spent -- which is exactly what makes the weights sum to
    /// one.
    pub fn charge(&mut self, halt_prob: f32) -> f32 {
        self.executions += 1;
        let p = halt_prob.clamp(0.0, 1.0);
        let spent_all = 1.0 - self.remainder + p * self.remainder;

        let terminal = self.finished
            || spent_all >= self.config.exit_threshold
            || self.cursor >= self.num_blocks
            || self.iterations >= self.config.max_iterations
            || self.config.budget.is_some_and(|b| self.executions >= b);

        if terminal {
            let w = self.remainder;
            self.remainder = 0.0;
            self.finished = true;
            w
        } else {
            let w = p * self.remainder;
            self.remainder -= w;
            w
        }
    }
}

/// Confidence of an estimate at noise level `sigma`, from its
/// signal-to-noise ratio: `1 - exp(-SNR)`, mapped into `[0, 1)` and monotone
/// increasing in the signal power.
///
/// The mapping is scale-free in the right way: doubling both the signal and
/// `sigma` leaves the confidence unchanged, so a threshold means the same
/// thing at every point of the trajectory.
pub fn snr_confidence(signal_power: f32, sigma: f64) -> f32 {
    let noise_power = (sigma * sigma) as f32 + 1e-8;
    let snr = (signal_power.max(0.0) / noise_power) as f64;
    (1.0 - (-snr).exp()) as f32
}

/// Learnable part of the loop graph: a halting head plus skip-connection
/// weights between blocks.
///
/// The skip weights are **zero-initialized**, so a freshly built loop graph
/// computes exactly the ACT mixture with no skip contribution -- the same
/// identity-at-init discipline the adaLN-zero trunk uses. Any skip pathway the
/// model ends up using is one training deliberately opened.
#[derive(Module, Debug)]
pub struct LoopGraph<B: Backend> {
    halting: HaltingHead<B>,
    /// `[num_blocks, num_blocks]`: `skip[from][to]` weights block `from`'s
    /// output when block `to` executes.
    skip_weights: Param<Tensor<B, 2>>,
    num_blocks: usize,
}

impl<B: Backend<FloatElem = f32>> LoopGraph<B> {
    pub fn new(num_blocks: usize, hidden_size: usize, device: &B::Device) -> Self {
        Self {
            halting: HaltingHead::new(
                &HaltingConfig { hidden_size, ..HaltingConfig::default() },
                device,
            ),
            skip_weights: Param::from_tensor(Tensor::zeros([num_blocks, num_blocks], device)),
            num_blocks,
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn halting(&self) -> &HaltingHead<B> {
        &self.halting
    }

    /// Scalar skip weight from `from` into the accumulator at `to`.
    fn skip_weight(&self, from: usize, to: usize) -> Tensor<B, 1> {
        self.skip_weights
            .val()
            .narrow(0, from.min(self.num_blocks - 1), 1)
            .narrow(1, to.min(self.num_blocks - 1), 1)
            .reshape([1])
    }

    /// Dynamic `x0` estimate at `(z, sigma)`.
    ///
    /// Returns the mixture plus the trace of what ran. The signature matches
    /// the fixed-depth predictor, so this can be handed to
    /// [`crate::solver::integrate`] unchanged.
    pub fn x0_estimate(
        &self,
        model: &DblockClassifier<B>,
        pixel_values: &Tensor<B, 4>,
        z: &Tensor<B, 2>,
        sigma: f64,
        config: &LoopGraphConfig,
    ) -> (Tensor<B, 2>, ExecutionTrace) {
        let mut planner = LoopPlanner::new(*config, self.num_blocks);
        let mut trace = ExecutionTrace::default();

        let mut mixture: Option<Tensor<B, 2>> = None;
        let mut skip_accum: Option<Tensor<B, 2>> = None;
        let mut confidence = snr_confidence(mean_square(z), sigma);
        let mut last_output: Option<(usize, Tensor<B, 2>)> = None;

        loop {
            let (block, loop_from) = match planner.next(confidence) {
                Decision::Stop => break,
                Decision::Skip(b) => {
                    trace.skipped.push(b);
                    continue;
                }
                Decision::Run(b) => (b, None),
                Decision::LoopBack { from, to } => (to, Some(from)),
            };

            let span = model.layer_range(block);
            let x0 = model.x0_estimate(pixel_values, z, sigma, Some(span));

            // Halting probability from the block's own output.
            let halt = self.halting.halt_probability(x0.clone());
            let halt_scalar = halt.clone().mean().into_scalar();
            let weight = planner.charge(halt_scalar);

            let contribution = x0.clone().mul_scalar(weight);
            mixture = Some(match mixture {
                None => contribution,
                Some(acc) => acc + contribution,
            });

            // Skip connection: the previous block's output re-enters weighted
            // by the learned (zero-initialized) skip strength.
            if let Some((prev_block, prev_out)) = &last_output {
                let w = self.skip_weight(*prev_block, block).unsqueeze_dim::<2>(0);
                let contrib = prev_out.clone() * w;
                skip_accum = Some(match skip_accum {
                    None => contrib,
                    Some(acc) => acc + contrib,
                });
            }

            confidence = snr_confidence(mean_square(&x0), sigma);
            trace.blocks_run.push(block);
            trace.weights.push(weight);
            if let Some(from) = loop_from {
                trace.looped.push((from, block));
            }
            last_output = Some((block, x0));

            if planner.finished() {
                break;
            }
        }

        trace.iterations = trace.blocks_run.len() + trace.skipped.len();

        let device = z.device();
        let mut out = mixture.unwrap_or_else(|| Tensor::zeros(z.dims(), &device));
        if let Some(skip) = skip_accum {
            out = out + skip;
        }
        (out, trace)
    }
}

/// Batch-mean of `x^2`, the signal power used by [`snr_confidence`].
fn mean_square<B: Backend<FloatElem = f32>>(x: &Tensor<B, 2>) -> f32 {
    x.clone().powf_scalar(2.0).mean().into_scalar()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    /// Drive a planner to completion with a fixed confidence and halting
    /// probability, returning the trace-relevant facts.
    fn drive(config: LoopGraphConfig, num_blocks: usize, confidence: f32, halt: f32) -> (Vec<usize>, Vec<f32>, usize) {
        let mut planner = LoopPlanner::new(config, num_blocks);
        let (mut run, mut weights) = (Vec::new(), Vec::new());
        let mut iterations = 0usize;
        loop {
            iterations += 1;
            assert!(iterations < 10_000, "planner failed to terminate");
            let block = match planner.next(confidence) {
                Decision::Stop => break,
                Decision::Skip(_) => continue,
                Decision::Run(b) => b,
                Decision::LoopBack { to, .. } => to,
            };
            run.push(block);
            weights.push(planner.charge(halt));
            if planner.finished() {
                break;
            }
        }
        (run, weights, iterations)
    }

    #[test]
    fn test_planner_always_terminates() {
        // Adversarial settings: confidence pinned below the loop-back
        // threshold, halting probability pinned at zero. Neither the ACT mass
        // nor the confidence can ever end the run, so only the hard caps can
        // -- which is precisely what they exist for.
        for max_iterations in [1usize, 3, 8, 32] {
            let config = LoopGraphConfig {
                max_iterations,
                loopback_threshold: 1.0,
                max_loopbacks: usize::MAX,
                skip_threshold: f32::INFINITY,
                exit_threshold: f32::INFINITY,
                ..LoopGraphConfig::default()
            };
            let (run, _, _) = drive(config, 4, 0.0, 0.0);
            assert!(
                run.len() <= max_iterations,
                "{} executions exceeded max_iterations {max_iterations}",
                run.len()
            );
        }
    }

    #[test]
    fn test_planner_respects_the_budget() {
        for budget in [1usize, 2, 3, 5] {
            let config = LoopGraphConfig {
                budget: Some(budget),
                max_iterations: 100,
                loopback_threshold: 1.0,
                max_loopbacks: usize::MAX,
                exit_threshold: f32::INFINITY,
                skip_threshold: f32::INFINITY,
            };
            let (run, _, _) = drive(config, 4, 0.0, 0.0);
            assert!(run.len() <= budget, "{} executions exceeded budget {budget}", run.len());
        }
    }

    #[test]
    fn test_act_weights_sum_to_one() {
        // The defining property of the ACT remainder rule: whatever the
        // halting probabilities, the mixture is a convex combination. Without
        // it the "x0 estimate" would be an arbitrarily scaled vector.
        for halt in [0.0f32, 0.05, 0.3, 0.5, 0.9, 1.0] {
            for num_blocks in [1usize, 2, 4, 7] {
                let (run, weights, _) =
                    drive(LoopGraphConfig::default(), num_blocks, 0.5, halt);
                assert!(!run.is_empty(), "at least one block must run");
                let mass: f32 = weights.iter().sum();
                assert!(
                    (mass - 1.0).abs() < 1e-5,
                    "weights sum to {mass}, not 1 (halt={halt}, blocks={num_blocks})"
                );
                assert!(weights.iter().all(|&w| (0.0..=1.0).contains(&w)), "weights must be probabilities");
            }
        }
    }

    #[test]
    fn test_high_halting_probability_exits_early() {
        // p = 1 spends the whole remainder immediately, so exactly one block
        // should run even though several are available.
        let (run, weights, _) = drive(LoopGraphConfig::default(), 6, 0.5, 1.0);
        assert_eq!(run, vec![0], "halting at p=1 must exit after one block");
        assert!((weights[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_feedforward_config_runs_every_block_once() {
        // The ablation control: no skips, no loops, no early exit.
        let (run, weights, _) = drive(LoopGraphConfig::feedforward(5), 5, 0.5, 0.0);
        assert_eq!(run, vec![0, 1, 2, 3, 4]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_confident_inputs_are_skipped() {
        // Loop-back disabled so the test isolates the skip/run decision.
        let config = LoopGraphConfig {
            skip_threshold: 0.5,
            loopback_threshold: f32::NEG_INFINITY,
            ..LoopGraphConfig::default()
        };
        let mut planner = LoopPlanner::new(config, 3);
        assert_eq!(planner.next(0.9), Decision::Skip(0), "confident input must skip");
        assert_eq!(planner.next(0.1), Decision::Run(1), "unconfident input must run");
        // Skipping does not consume budget, only executions do.
        assert_eq!(planner.remainder(), 1.0);
    }

    #[test]
    fn test_loopback_is_bounded() {
        let config = LoopGraphConfig {
            loopback_threshold: 1.0,
            max_loopbacks: 2,
            skip_threshold: f32::INFINITY,
            max_iterations: 100,
            exit_threshold: f32::INFINITY,
            ..LoopGraphConfig::default()
        };
        let mut planner = LoopPlanner::new(config, 4);
        // First call cannot loop back (cursor is still 0).
        assert_eq!(planner.next(0.0), Decision::Run(0));
        planner.charge(0.0);
        let mut loopbacks = 0;
        for _ in 0..20 {
            match planner.next(0.0) {
                Decision::LoopBack { .. } => {
                    loopbacks += 1;
                    planner.charge(0.0);
                }
                Decision::Stop => break,
                _ => {
                    planner.charge(0.0);
                }
            }
        }
        assert_eq!(loopbacks, 2, "loop-backs must be capped at max_loopbacks");
    }

    #[test]
    fn test_snr_confidence_is_monotone_and_scale_invariant() {
        // Monotone in signal power...
        let mut prev = -1.0f32;
        for i in 0..50 {
            let c = snr_confidence(i as f32 * 0.1, 1.0);
            assert!(c > prev, "confidence must increase with signal power");
            assert!((0.0..1.0).contains(&c));
            prev = c;
        }
        // ...and unchanged when signal power and sigma^2 scale together, so a
        // threshold means the same thing at every noise level.
        let base = snr_confidence(4.0, 2.0);
        for k in [0.25f64, 1.0, 9.0, 100.0] {
            let scaled = snr_confidence(4.0 * k as f32, 2.0 * k.sqrt());
            assert!((scaled - base).abs() < 1e-5, "not scale invariant at k={k}");
        }
        assert_eq!(snr_confidence(0.0, 1.0), 0.0);
    }

    #[test]
    fn test_trace_histogram_counts_repeats() {
        let trace = ExecutionTrace {
            blocks_run: vec![0, 1, 0, 2],
            weights: vec![0.25, 0.25, 0.25, 0.25],
            ..ExecutionTrace::default()
        };
        assert_eq!(trace.usage_histogram(3), vec![2, 1, 1]);
        assert_eq!(trace.executions(), 4);
        assert!((trace.weight_mass() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_loop_graph_is_a_convex_mixture_at_init() {
        use crate::dblock::DblockConfig;
        use crate::vit::ViTDiTConfig;

        let device = Default::default();
        let cfg = ViTDiTConfig::tiny(10);
        let model = DblockClassifier::<B>::new(
            &cfg,
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        let graph = LoopGraph::<B>::new(2, 32, &device);

        let pixels = Tensor::<B, 4>::zeros([2, 3, 32, 32], &device);
        let z = Tensor::<B, 2>::ones([2, 32], &device);
        let (x0, trace) = graph.x0_estimate(
            &model,
            &pixels,
            &z,
            1.0,
            &LoopGraphConfig::feedforward(2),
        );

        assert_eq!(x0.dims(), [2, 32]);
        assert!(!trace.blocks_run.is_empty());
        assert!(
            (trace.weight_mass() - 1.0).abs() < 1e-5,
            "mixture weights must sum to 1, got {}",
            trace.weight_mass()
        );

        // Zero-initialized skip weights make the output a pure ACT mixture of
        // block outputs, so it cannot exceed their magnitude range.
        let per_block: Vec<f32> = trace
            .blocks_run
            .iter()
            .map(|&b| {
                model
                    .x0_estimate(&pixels, &z, 1.0, Some(model.layer_range(b)))
                    .abs()
                    .max()
                    .into_scalar()
            })
            .collect();
        let bound = per_block.iter().cloned().fold(0.0f32, f32::max);
        let got = x0.abs().max().into_scalar();
        assert!(
            got <= bound + 1e-5,
            "a convex mixture cannot exceed its inputs: {got} > {bound}"
        );
    }
}
