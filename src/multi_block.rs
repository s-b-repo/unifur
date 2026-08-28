//! Multi-block denoising strategies (roadmap Phases 2 and 10).
//!
//! The sigma schedule is partitioned into windows; each window selects a
//! contiguous span of transformer blocks to execute:
//!
//! - [`Strategy::Sequential`]: one block per window (original DiffusionBlocks).
//! - [`Strategy::Parallel`]: `k` adjacent blocks jointly per window. During
//!   training, running only a span is simultaneously gradient routing: no
//!   gradient flows through layers outside the executed span.
//! - [`Strategy::Hybrid`]: sequential above a fraction of `sigma_max`,
//!   parallel below it (coarse structure from single blocks, fine detail
//!   from joint spans).
//! - [`Strategy::Adaptive`]: widens the span while the class confidence of the
//!   x0 estimate stays below a threshold and narrows it again once the
//!   estimate is confident.
//!
//! Every strategy runs through the shared [`crate::solver::SolverState`], so
//! the solver choice and the span policy are independent: any
//! [`SolverKind`] can be combined with any [`Strategy`].
//!
//! [`Gated`] wraps a strategy with the Phase-12 quality gates: samples whose
//! update looks degenerate keep their previous latent, and the per-block
//! rejection tally is reported back through [`SamplingStats`].

use crate::{
    accuracy::{Ensemble, Guidance, LogitNorm},
    dblock::DblockClassifier,
    planner::{Budget, Plan, TrajectoryPlanner, TrajectoryStep},
    precision::PrecisionPolicy,
    quality::{self, GateLedger, LayerGates, QualityGateConfig},
    sigma::{P_MEAN, P_STD, SIGMA_MAX, SIGMA_MIN},
    solver::{SolverKind, SolverState},
};
use burn::tensor::{backend::Backend, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;

/// How many transformer layers each window executes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Strategy {
    #[default]
    Sequential,
    Parallel {
        k: usize,
    },
    /// Sequential while `sigma > warmup_frac * sigma_max`, then Parallel{k}.
    Hybrid {
        k: usize,
        warmup_frac: f64,
    },
    /// Widen the span up to `k_max` while confidence is below
    /// `conf_threshold`, narrow it back down once the estimate is confident.
    Adaptive {
        k_max: usize,
        conf_threshold: f32,
    },
}

impl Strategy {
    /// Whether the strategy needs class probabilities from the denoiser (only
    /// [`Strategy::Adaptive`] inspects them to steer the span width).
    pub fn needs_probabilities(&self) -> bool {
        matches!(self, Strategy::Adaptive { .. })
    }

    /// Largest span width the strategy can request.
    pub fn max_width(&self) -> usize {
        match self {
            Strategy::Sequential => 1,
            Strategy::Parallel { k } | Strategy::Hybrid { k, .. } => (*k).max(1),
            Strategy::Adaptive { k_max, .. } => (*k_max).max(1),
        }
    }
}

/// Strategy + quality gates.
#[derive(Debug, Clone, Default)]
pub struct Gated {
    pub inner: Strategy,
    /// Per-block gates; [`LayerGates::uniform`] reproduces a single
    /// batch-level configuration.
    pub gate: LayerGates,
}

impl Gated {
    /// Strategy with one gate applied to every block.
    pub fn uniform(inner: Strategy, gate: QualityGateConfig) -> Self {
        Self { inner, gate: LayerGates::uniform(gate) }
    }
}

/// Configuration of a multi-block sampling run.
#[derive(Debug, Clone, Default)]
pub struct MultiBlockConfig {
    pub strategy: Gated,
    pub solver: SolverKind,
    /// Number of inference windows; defaults to the model's `num_blocks`.
    pub num_steps: Option<usize>,
    /// Per-window arithmetic precision (roadmap 10.6). The default is `f32`
    /// everywhere, in which case the rounding path is skipped entirely.
    pub precision: PrecisionPolicy,
    /// Guidance applied to every x0 estimate (roadmap 22.5). The default is the
    /// exact identity and costs nothing; anything else doubles the model calls,
    /// because an unconditional estimate has to be produced alongside each
    /// conditional one.
    pub guidance: Guidance,
    /// Normalization applied to the final logits (roadmap 22.6). Never changes
    /// the prediction — see [`LogitNorm`] for why it is worth having anyway.
    pub logit_norm: LogitNorm,
}

/// Outcome statistics of a sampling run.
#[derive(Debug, Clone, Default)]
pub struct SamplingStats {
    /// Denoiser invocations performed (including a solver's extra corrector
    /// calls and the final denoise).
    pub model_calls: usize,
    /// Samples rejected by the quality gate at least once.
    pub gated_samples: usize,
    /// Per-block gate tally.
    pub ledger: GateLedger,
    /// Layer spans executed, in window order.
    pub spans: Vec<std::ops::Range<usize>>,
    /// Total transformer layers executed across the run; the honest cost
    /// measure when strategies use different span widths. Includes a solver's
    /// corrector evaluations and any planning work.
    pub layers_executed: usize,
    /// Windows that ran below `f32`.
    pub reduced_precision_windows: usize,
    /// Denoiser invocations spent *planning* rather than advancing the
    /// trajectory. Zero for every strategy that follows a fixed schedule.
    pub planning_calls: usize,
    /// Layers executed while planning. Counted inside `layers_executed` too —
    /// it is real compute — but reported separately so the overhead of
    /// lookahead is visible rather than blended into the sampling cost.
    pub planning_layers: usize,
}

impl SamplingStats {
    /// Mean span width across the executed windows.
    ///
    /// Averages the *recorded spans*, not `layers_executed`. Those differ:
    /// `layers_executed` also carries a solver's corrector evaluations and any
    /// planning work, so dividing it by the window count would report Heun's
    /// spans as twice their width and a planned run's as many times it.
    pub fn mean_span_width(&self) -> f32 {
        if self.spans.is_empty() {
            return 0.0;
        }
        let total: usize = self.spans.iter().map(std::ops::Range::len).sum();
        total as f32 / self.spans.len() as f32
    }

    /// Fraction of executed layers spent planning rather than sampling.
    pub fn planning_overhead(&self) -> f32 {
        if self.layers_executed == 0 {
            return 0.0;
        }
        self.planning_layers as f32 / self.layers_executed as f32
    }
}

impl<B: Backend<FloatElem = f32>> DblockClassifier<B> {
    /// Multi-block sampling returning final logits plus statistics.
    ///
    /// Generalizes [`Self::diffusion_step`] with selectable block spans,
    /// solvers and quality gating.
    pub fn sample_multi_block(
        &self,
        pixel_values: &Tensor<B, 4>,
        config: &MultiBlockConfig,
        rng: &mut impl Rng,
    ) -> (Tensor<B, 2>, SamplingStats) {
        let num_blocks = self.num_blocks();
        let steps = config.num_steps.unwrap_or(num_blocks).max(2);
        let schedule = crate::sigma::discrete_sigmas_dblock(steps, SIGMA_MIN, SIGMA_MAX, P_MEAN, P_STD);
        crate::solver::assert_descending(&schedule);

        let b = pixel_values.dims()[0];
        let h_dim = self.model().label_embedding_weight().dims()[1];

        // Initial latent N(0, I) scaled to sqrt(1 + sigma_0^2), matching
        // diffusion_step's convention.
        let s0 = schedule[0];
        let mut z = crate::solver::randn_like(
            &Tensor::<B, 2>::zeros([b, h_dim], &pixel_values.device()),
            rng,
        )
        .mul_scalar((1.0 + s0 * s0).sqrt() as f32);

        let mut stats = SamplingStats {
            ledger: GateLedger::new(num_blocks),
            ..SamplingStats::default()
        };
        let mut ever_gated = vec![false; b];
        let mut prev_x0: Option<Tensor<B, 2>> = None;
        let mut current_k: usize = 1;
        let mut solver = SolverState::new(config.solver);
        let bounds = self.block_bounds();

        for window in schedule.windows(2) {
            let (sigma, s_next) = (window[0], window[1]);

            let base_block = crate::sigma::estimate_target_layer(&bounds, &[sigma]);
            let span = self.select_span(&config.strategy.inner, base_block, current_k, sigma);

            // Mixed precision: the latent entering a window and the estimate
            // leaving it are rounded to the window's format, so the
            // representation error is injected exactly where a native
            // low-precision kernel would inject it.
            let window_precision = config.precision.for_sigma(sigma);
            if !config.precision.is_full_precision() {
                z = window_precision.round(z);
            }

            let (x0, probs) = self.x0_estimate_probs(
                pixel_values,
                &z,
                sigma,
                Some(span.clone()),
                config.strategy.inner.needs_probabilities(),
            );
            let x0 = window_precision.round(x0);
            stats.model_calls += 1;

            // Guidance needs the same estimate without the image conditioning,
            // so it is a second call per window -- charged here rather than
            // hidden, since a guided run is genuinely twice the compute.
            let x0 = if config.guidance.is_identity() {
                x0
            } else {
                let uncond = self.x0_estimate_unconditional(pixel_values, &z, sigma, Some(span.clone()));
                stats.model_calls += 1;
                stats.layers_executed += span.len();
                config.guidance.apply(x0, window_precision.round(uncond))
            };
            stats.layers_executed += span.len();
            stats.spans.push(span.clone());
            if window_precision != crate::precision::Precision::F32 {
                stats.reduced_precision_windows += 1;
            }

            // Adaptive depth: widen while the estimate looks unconfident and
            // narrow again once it is confident, so the extra depth is spent
            // only where it is needed.
            if let (Strategy::Adaptive { conf_threshold, k_max }, Some(p)) =
                (&config.strategy.inner, &probs)
            {
                let mean_conf = mean_max_probability(p);
                current_k = if mean_conf < *conf_threshold {
                    (current_k + 1).min((*k_max).max(1))
                } else {
                    current_k.saturating_sub(1).max(1)
                };
            }

            // The solver may need extra evaluations (Heun's corrector); they
            // reuse the same span policy and the same precision, so both the
            // cost accounting and the emulated arithmetic stay consistent with
            // the primary call.
            let mut predictor = |sig: f64, zz: &Tensor<B, 2>| {
                stats.model_calls += 1;
                stats.layers_executed += span.len();
                let estimate = self.x0_estimate(pixel_values, zz, sig, Some(span.clone()));
                config.precision.for_sigma(sig).round(estimate)
            };
            let new_z = solver.step(sigma, s_next, z.clone(), &x0, &mut predictor, rng);

            match &prev_x0 {
                None => z = new_z,
                Some(prev) => {
                    let gate = config.strategy.gate.for_block(base_block);
                    let report = quality::evaluate(gate, prev, &x0, probs.as_ref());
                    stats.ledger.record(base_block, &report);
                    for (i, ok) in report.passed.iter().enumerate() {
                        if !ok {
                            ever_gated[i] = true;
                        }
                    }
                    z = merge_kept(&new_z, &z, &report.passed);
                }
            }
            prev_x0 = Some(x0);
        }

        stats.gated_samples = ever_gated.iter().filter(|&&g| g).count();
        let min_sigma = *schedule.last().expect("non-empty schedule");
        let logits = self.denoise(pixel_values.clone(), z, &vec![min_sigma; b], None);
        stats.model_calls += 1;
        (config.logit_norm.apply(logits), stats)
    }

    /// Sample under several configurations and combine their answers
    /// (roadmap 22.4).
    ///
    /// The members should *disagree* — different solvers, different span
    /// strategies, different seeds. An ensemble of identical members is exactly
    /// one member at N times the cost, which is the certificate
    /// `accuracy/identical_members_are_identity`.
    ///
    /// Returns **probabilities**, not logits: a plurality vote has no logit
    /// scale, and a mean of distributions is not the softmax of anything. The
    /// reported statistics are the sum across members, so the cost of the
    /// ensemble is visible rather than amortized away.
    ///
    /// # Panics
    ///
    /// If `members` is empty.
    pub fn sample_ensemble(
        &self,
        pixel_values: &Tensor<B, 4>,
        members: &[MultiBlockConfig],
        ensemble: Ensemble,
        rng: &mut impl Rng,
    ) -> (Tensor<B, 2>, SamplingStats) {
        assert!(!members.is_empty(), "an ensemble needs at least one member");

        let mut logits: Vec<Tensor<B, 2>> = Vec::with_capacity(members.len());
        let mut total = SamplingStats {
            ledger: GateLedger::new(self.num_blocks()),
            ..SamplingStats::default()
        };

        for config in members {
            let (member_logits, stats) = self.sample_multi_block(pixel_values, config, rng);
            logits.push(member_logits);

            total.model_calls += stats.model_calls;
            total.layers_executed += stats.layers_executed;
            total.reduced_precision_windows += stats.reduced_precision_windows;
            total.planning_calls += stats.planning_calls;
            total.planning_layers += stats.planning_layers;
            total.gated_samples += stats.gated_samples;
            total.spans.extend(stats.spans);
            total.ledger.merge(&stats.ledger);
        }

        (ensemble.combine(&logits), total)
    }

    /// Contiguous layer span for one window under the given strategy.
    fn select_span(
        &self,
        strategy: &Strategy,
        base_block: usize,
        current_k: usize,
        sigma: f64,
    ) -> std::ops::Range<usize> {
        let n = self.num_blocks();
        let k = match strategy {
            Strategy::Sequential => 1,
            Strategy::Parallel { k } => *k,
            Strategy::Hybrid { k, warmup_frac } => {
                if sigma > warmup_frac * SIGMA_MAX {
                    1
                } else {
                    *k
                }
            }
            Strategy::Adaptive { .. } => current_k,
        };
        let start = base_block.min(n.saturating_sub(1));
        let end_block = (start + k.max(1)).clamp(start + 1, n);
        self.layer_range(start).start..self.layer_range(end_block - 1).end
    }
}

/// Configuration for planned sampling (roadmap Phase 21a).
///
/// Every other strategy follows a schedule fixed before the first model call.
/// This one decides each step from evidence: candidate `(sigma, span)` pairs
/// are scored, optionally by rolling them forward, and only the winner's first
/// step is committed.
#[derive(Debug, Clone)]
pub struct PlannedConfig {
    /// Ceiling on planning work per committed step.
    pub budget: Budget,
    /// Multiplicative sigma steps to consider.
    pub sigma_ratios: Vec<f64>,
    /// Span widths, in blocks, to consider.
    pub widths: Vec<usize>,
    /// Score charged per block executed. Trades accuracy against compute.
    pub cost_per_block: f64,
    /// Score credited per nat of log-sigma descended.
    ///
    /// Without it the planner would take the smallest jump available every
    /// time — the most accurate single step is always the shortest one — and
    /// never reach `sigma_min`. Progress is measured in log-sigma because that
    /// is the coordinate the solvers integrate in (`lambda = -log sigma`).
    pub progress_weight: f64,
    pub solver: SolverKind,
    /// Hard cap on committed steps, independent of the planning budget.
    pub max_steps: usize,
    /// Normalization applied to the final logits (roadmap 22.6).
    ///
    /// Guidance has no counterpart here on purpose: scoring every candidate
    /// under guidance would double each evaluation and change what the budget
    /// means, while applying it only to the committed step would make the
    /// planner choose against a trajectory it is not actually following. Use
    /// [`MultiBlockConfig::guidance`] on the scheduled sampler for that.
    pub logit_norm: LogitNorm,
}

impl Default for PlannedConfig {
    fn default() -> Self {
        Self {
            budget: Budget::default(),
            sigma_ratios: vec![0.3, 0.5, 0.75],
            widths: vec![1, 2],
            cost_per_block: 0.02,
            progress_weight: 0.05,
            solver: SolverKind::Euler,
            max_steps: 32,
            logit_norm: LogitNorm::None,
        }
    }
}

/// What the planner decided, step by step.
#[derive(Debug, Clone, Default)]
pub struct PlanTrace {
    /// Committed `(sigma, width)` steps, in order.
    pub steps: Vec<TrajectoryStep>,
    /// Candidate evaluations charged, per committed step.
    pub evaluations: Vec<usize>,
    /// Lookahead depth actually achieved, per committed step.
    pub depths: Vec<usize>,
    /// Committed steps whose search the budget cut short.
    pub budget_exhausted_steps: usize,
    /// Whether `max_steps` bound before the trajectory reached `sigma_min`, so
    /// the remaining distance was closed in one unplanned step.
    pub forced_final_step: bool,
    /// Noise level the trajectory actually ended at. Always `sigma_min` — the
    /// field exists so a caller can check rather than trust.
    pub final_sigma: f64,
}

impl PlanTrace {
    /// Mean lookahead depth across the **planned** steps.
    ///
    /// A forced final step (see [`Self::forced_final_step`]) is excluded: it
    /// was never planned, and averaging its depth of zero in would understate
    /// how far the planner actually looked.
    pub fn mean_depth(&self) -> f64 {
        let planned = self.planned_steps();
        if planned == 0 {
            return 0.0;
        }
        self.depths[..planned].iter().sum::<usize>() as f64 / planned as f64
    }

    /// Steps the planner chose, excluding any forced final step.
    pub fn planned_steps(&self) -> usize {
        self.depths.len() - usize::from(self.forced_final_step)
    }

    /// Total candidate evaluations across the run.
    pub fn total_evaluations(&self) -> usize {
        self.evaluations.iter().sum()
    }
}

/// Identifies a hypothesized path, for looking up the state it leads to.
///
/// Sigmas are keyed by their bit pattern rather than by value: they are
/// reproduced by the identical arithmetic on every visit, so equality is exact
/// and a tolerance would only invite two distinct paths to collide.
type PathKey = Vec<(u64, usize)>;

fn path_key(path: &crate::planner::Path<TrajectoryStep>) -> PathKey {
    path.steps.iter().map(|s| (s.sigma.to_bits(), s.width)).collect()
}

impl<B: Backend<FloatElem = f32>> DblockClassifier<B> {
    /// Sample by planning each step rather than following a schedule.
    ///
    /// At every step the planner scores candidate `(sigma, span)` pairs and,
    /// when `budget.max_depth > 0`, rolls the promising ones forward to see
    /// where they lead. Only the winner's first step is committed; the rest is
    /// re-planned once its consequences are observed.
    ///
    /// # Where lookahead earns its keep
    ///
    /// A candidate is scored by the confidence of the x0 estimate at the sigma
    /// it departs from, so at depth 0 the choice of *how far to jump* is
    /// decided by the progress-versus-cost trade alone — the planner cannot
    /// yet tell a good jump from an over-long one. Depth 1 is exactly what
    /// reveals it: an over-long jump lands somewhere the next estimate is less
    /// confident, and that shows up in the path score. Lookahead here is not
    /// a refinement of the greedy signal; it supplies information the greedy
    /// signal does not contain.
    ///
    /// # Cost
    ///
    /// One model call per distinct span width per expanded node, so
    /// `model_calls <= evaluations` always. [`PlannedConfig::budget`] bounds
    /// evaluations per committed step, and [`PlannedConfig::max_steps`] bounds
    /// the steps — together a hard ceiling on the whole run.
    pub fn sample_planned(
        &self,
        pixel_values: &Tensor<B, 4>,
        config: &PlannedConfig,
        rng: &mut impl Rng,
    ) -> (Tensor<B, 2>, SamplingStats, PlanTrace) {
        let num_blocks = self.num_blocks();
        let schedule = crate::sigma::discrete_sigmas_dblock(
            config.max_steps.max(2),
            SIGMA_MIN,
            SIGMA_MAX,
            P_MEAN,
            P_STD,
        );
        let sigma_start = schedule[0];
        let sigma_floor = *schedule.last().expect("non-empty schedule");

        let b = pixel_values.dims()[0];
        let h_dim = self.model().label_embedding_weight().dims()[1];
        let mut z = crate::solver::randn_like(
            &Tensor::<B, 2>::zeros([b, h_dim], &pixel_values.device()),
            rng,
        )
        .mul_scalar((1.0 + sigma_start * sigma_start).sqrt() as f32);

        let mut stats = SamplingStats {
            ledger: GateLedger::new(num_blocks),
            ..SamplingStats::default()
        };
        let mut trace = PlanTrace::default();
        let bounds = self.block_bounds();
        let mut solver = SolverState::new(config.solver);

        let planner = TrajectoryPlanner::new(config.budget)
            .with_ratios(config.sigma_ratios.clone())
            .with_widths(config.widths.clone())
            .with_cost_per_block(config.cost_per_block);

        let mut sigma = sigma_start;
        while sigma > sigma_floor && trace.steps.len() < config.max_steps {
            // Rolled-forward latents, addressed by the path that produced them.
            // Cleared each committed step: once a step is taken the hypotheses
            // that assumed otherwise are worthless.
            //
            // Each entry carries a **clone of the solver state**, not just the
            // latent. A multistep method (DPM++ 2M/3M) integrates from its
            // history of past x0 predictions, so a rollout that advanced with
            // plain Euler would score candidates under dynamics the sampler is
            // not going to follow -- and one that shared the live solver would
            // corrupt the trajectory actually being taken.
            let mut states: HashMap<PathKey, (Tensor<B, 2>, SolverState<B>)> = HashMap::new();
            states.insert(Vec::new(), (z.clone(), solver.clone()));

            // Rollouts draw their own noise. Using the caller's stream would
            // make the committed trajectory depend on how much *speculation*
            // happened, so the same seed would stop reproducing the same run.
            let mut plan_rng = StdRng::seed_from_u64(0x5EED ^ trace.steps.len() as u64);
            // x0 estimates and their confidences, one per (node, span width).
            // Every sigma candidate departing from the same node with the same
            // width shares them, so the sigma choice costs no extra model call.
            let mut evaluated: HashMap<(PathKey, usize), (Tensor<B, 2>, f64)> = HashMap::new();
            let mut calls = 0usize;
            let mut layers = 0usize;

            let plan: Plan<TrajectoryStep> = planner.plan_with(
                sigma,
                sigma_floor,
                |path, next_sigma, width, remaining| {
                    let key = path_key(path);
                    let (z_here, solver_here) = states.get(&key)?.clone();
                    let here = path.steps.last().map_or(sigma, |s| s.sigma);

                    let (x0, score) = match evaluated.get(&(key.clone(), width)) {
                        Some((x0, conf)) => (x0.clone(), *conf),
                        None => {
                            // A fresh evaluation costs a model call, so decline
                            // rather than overrun the allowance. Truncating
                            // afterwards would bound the bookkeeping, not the
                            // compute.
                            if remaining == 0 {
                                return None;
                            }
                            let base = crate::sigma::estimate_target_layer(&bounds, &[here]);
                            let span =
                                self.select_span(&Strategy::Parallel { k: width }, base, 1, here);
                            let (x0, probs) = self.x0_estimate_probs(
                                pixel_values,
                                &z_here,
                                here,
                                Some(span.clone()),
                                true,
                            );
                            calls += 1;
                            layers += span.len();
                            let conf = f64::from(probs.as_ref().map_or(0.0, mean_max_probability));
                            evaluated.insert((key.clone(), width), (x0.clone(), conf));
                            (x0, conf)
                        }
                    };

                    // Roll the hypothesis forward with the *real* solver so a
                    // deeper level is scored from where this step actually
                    // lands. Heun's corrector is a genuine extra model call and
                    // is charged as one.
                    let mut child_solver = solver_here;
                    let base = crate::sigma::estimate_target_layer(&bounds, &[here]);
                    let span = self.select_span(&Strategy::Parallel { k: width }, base, 1, here);
                    let mut predictor = |sig: f64, zz: &Tensor<B, 2>| {
                        calls += 1;
                        layers += span.len();
                        self.x0_estimate(pixel_values, zz, sig, Some(span.clone()))
                    };
                    let z_child = child_solver.step(
                        here,
                        next_sigma,
                        z_here,
                        &x0,
                        &mut predictor,
                        &mut plan_rng,
                    );

                    let mut child = key;
                    child.push((next_sigma.to_bits(), width));
                    states.insert(child, (z_child, child_solver));

                    let progress = config.progress_weight * (here.ln() - next_sigma.ln());
                    Some(score + progress)
                },
            );

            stats.model_calls += calls;
            stats.layers_executed += layers;
            stats.planning_calls += calls;
            stats.planning_layers += layers;

            let Some(step) = plan.commit().copied() else { break };

            let base = crate::sigma::estimate_target_layer(&bounds, &[sigma]);
            let span = self.select_span(&Strategy::Parallel { k: step.width }, base, 1, sigma);
            stats.spans.push(span.clone());

            // The committed width was scored at this node, so its estimate is
            // already in hand; recomputing it would be paying twice for the
            // same answer.
            let x0 = match evaluated.remove(&(Vec::new(), step.width)).map(|(x0, _)| x0) {
                Some(x0) => x0,
                None => {
                    stats.model_calls += 1;
                    stats.layers_executed += span.len();
                    self.x0_estimate(pixel_values, &z, sigma, Some(span.clone()))
                }
            };

            let mut predictor = |sig: f64, zz: &Tensor<B, 2>| {
                stats.model_calls += 1;
                stats.layers_executed += span.len();
                self.x0_estimate(pixel_values, zz, sig, Some(span.clone()))
            };
            z = solver.step(sigma, step.sigma, z.clone(), &x0, &mut predictor, rng);

            trace.evaluations.push(plan.evaluations);
            trace.depths.push(plan.depth());
            if plan.budget_exhausted {
                trace.budget_exhausted_steps += 1;
            }
            trace.steps.push(step);
            sigma = step.sigma;
        }

        // `max_steps` can bind before the trajectory reaches the floor. Handing
        // a latent that is still at sigma=0.6 to a denoise at sigma_min would
        // be a silent discontinuity -- the estimate would be conditioned on a
        // noise level the latent does not have. Close the remaining distance in
        // one step, and record that it happened.
        if sigma > sigma_floor {
            let base = crate::sigma::estimate_target_layer(&bounds, &[sigma]);
            let width = trace.steps.last().map_or(1, |s| s.width);
            let span = self.select_span(&Strategy::Parallel { k: width }, base, 1, sigma);
            let x0 = self.x0_estimate(pixel_values, &z, sigma, Some(span.clone()));
            stats.model_calls += 1;
            stats.layers_executed += span.len();
            stats.spans.push(span.clone());

            let mut predictor = |sig: f64, zz: &Tensor<B, 2>| {
                stats.model_calls += 1;
                stats.layers_executed += span.len();
                self.x0_estimate(pixel_values, zz, sig, Some(span.clone()))
            };
            z = solver.step(sigma, sigma_floor, z.clone(), &x0, &mut predictor, rng);

            trace.steps.push(TrajectoryStep { sigma: sigma_floor, width });
            trace.evaluations.push(0);
            trace.depths.push(0);
            trace.forced_final_step = true;
            sigma = sigma_floor;
        }
        trace.final_sigma = sigma;

        let logits = self.denoise(pixel_values.clone(), z, &vec![sigma_floor; b], None);
        stats.model_calls += 1;
        (config.logit_norm.apply(logits), stats, trace)
    }

}

/// Batch-mean of the per-sample max class probability.
fn mean_max_probability<B: Backend<FloatElem = f32>>(probs: &Tensor<B, 2>) -> f32 {
    let per_sample: Vec<f32> = probs
        .clone()
        .max_dim(1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    if per_sample.is_empty() {
        return 0.0;
    }
    per_sample.iter().sum::<f32>() / per_sample.len() as f32
}

/// Single Euler step of dz/dsigma = (z - x0)/sigma toward `s_next`.
pub(crate) fn euler_step<B: Backend>(
    s: f64,
    s_next: f64,
    z: &Tensor<B, 2>,
    x0: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let d = (z.clone() - x0.clone()) / s;
    z.clone() + (s_next - s) * d
}

/// Keep old latent rows where `keep[i] == false`.
fn merge_kept<B: Backend>(new: &Tensor<B, 2>, old: &Tensor<B, 2>, keep: &[bool]) -> Tensor<B, 2> {
    let device = new.device();
    let mask_f: Vec<f32> = keep.iter().map(|&k| if k { 1.0 } else { 0.0 }).collect();
    let mask = Tensor::<B, 1>::from_floats(mask_f.as_slice(), &device).unsqueeze_dim::<2>(1);
    new.clone() * mask.clone() + old.clone() * (mask.neg() + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_merge_kept_selects_rows() {
        let device = Default::default();
        let new = Tensor::<B, 2>::ones([3, 2], &device);
        let old = Tensor::<B, 2>::zeros([3, 2], &device);
        let merged = merge_kept(&new, &old, &[true, false, true]);
        let got: Vec<f32> = merged.into_data().convert::<f32>().iter::<f32>().collect();
        assert_eq!(got, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_max_probability() {
        let device = Default::default();
        let probs = Tensor::<B, 1>::from_floats([0.7f32, 0.3, 0.25, 0.75].as_slice(), &device)
            .reshape([2, 2]);
        assert!((mean_max_probability(&probs) - 0.725).abs() < 1e-6);
    }

    #[test]
    fn test_strategy_metadata() {
        assert!(!Strategy::Sequential.needs_probabilities());
        assert!(Strategy::Adaptive { k_max: 3, conf_threshold: 0.5 }.needs_probabilities());
        assert_eq!(Strategy::Sequential.max_width(), 1);
        assert_eq!(Strategy::Parallel { k: 4 }.max_width(), 4);
        // A zero width is meaningless; the API clamps rather than emitting an
        // empty span.
        assert_eq!(Strategy::Parallel { k: 0 }.max_width(), 1);
    }
}
