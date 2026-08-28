//! Next-step and path prediction (roadmap Phase 21).
//!
//! Instead of always taking the greedy next step, score several candidates —
//! and, where it pays, short rollouts of what follows each one — and take the
//! best. Two planners live here because the two state spaces genuinely differ:
//!
//! - [`TrajectoryPlanner`] chooses the next `(sigma, span)` of a diffusion
//!   trajectory. It generalizes `Strategy::Adaptive` and `LoopGraph`, which
//!   already do a one-step, no-lookahead version of this.
//! - [`LookaheadDecoder`] chooses the next token of a language model by
//!   scoring candidate continuations rather than committing to the arg-max.
//!
//! They are **not** forced into one abstraction. What they share is the search:
//! [`Beam`] and [`Budget`] are generic over the candidate type, so the search
//! policy is written and verified once while each planner keeps the state and
//! the scoring function that actually fit it.
//!
//! # Why a budget is a type
//!
//! Lookahead multiplies work: `beam x depth x candidates` model calls per
//! committed step. A planner without an enforced ceiling is a planner that
//! occasionally takes minutes to emit one token. [`Budget`] is therefore
//! checked before every expansion and is the thing the certificates pin — the
//! same discipline `LoopGraph`'s planner uses.

use std::cmp::Ordering;

/// A hard ceiling on planning work.
///
/// Both limits are enforced: `max_evaluations` bounds total cost, and
/// `max_depth` bounds how far ahead any single path may look. Depth alone is
/// not enough — a wide beam at shallow depth is just as expensive.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Total candidate evaluations permitted.
    pub max_evaluations: usize,
    /// Longest rollout, in steps.
    pub max_depth: usize,
    /// Paths kept between expansions.
    pub beam_width: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self { max_evaluations: 64, max_depth: 2, beam_width: 4 }
    }
}

impl Budget {
    /// No lookahead: score the immediate candidates and commit. Reduces to the
    /// greedy policy the crate already had.
    pub fn greedy() -> Self {
        Self { max_evaluations: usize::MAX, max_depth: 0, beam_width: 1 }
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn with_beam(mut self, width: usize) -> Self {
        self.beam_width = width.max(1);
        self
    }

    /// Worst-case evaluations for `candidates` options per expansion.
    ///
    /// Reported rather than merely bounded, so a caller can see what it is
    /// about to spend before spending it.
    pub fn worst_case(&self, candidates: usize) -> usize {
        let per_level = self.beam_width.max(1) * candidates;
        let levels = self.max_depth + 1;
        per_level.saturating_mul(levels).min(self.max_evaluations)
    }
}

/// Tracks spend against a [`Budget`].
#[derive(Debug, Clone)]
pub struct Spend {
    budget: Budget,
    used: usize,
}

impl Spend {
    pub fn new(budget: Budget) -> Self {
        Self { budget, used: 0 }
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Whether one more evaluation is permitted.
    pub fn can_spend(&self) -> bool {
        self.used < self.budget.max_evaluations
    }

    /// Evaluations still affordable.
    pub fn remaining(&self) -> usize {
        self.budget.max_evaluations.saturating_sub(self.used)
    }

    /// Charge one evaluation; returns false when the budget is exhausted.
    pub fn spend(&mut self) -> bool {
        if !self.can_spend() {
            return false;
        }
        self.used += 1;
        true
    }
}

/// One scored path through the search.
#[derive(Debug, Clone, PartialEq)]
pub struct Path<T> {
    /// Steps taken, in order.
    pub steps: Vec<T>,
    /// Accumulated score; higher is better.
    pub score: f64,
}

impl<T: Clone> Path<T> {
    pub fn root() -> Self {
        Self { steps: Vec::new(), score: 0.0 }
    }

    pub fn extended(&self, step: T, delta: f64) -> Self {
        let mut steps = self.steps.clone();
        steps.push(step);
        Self { steps, score: self.score + delta }
    }

    /// The step this path commits to, i.e. its first.
    pub fn head(&self) -> Option<&T> {
        self.steps.first()
    }

    pub fn depth(&self) -> usize {
        self.steps.len()
    }
}

/// Beam search over candidate steps.
///
/// Generic over the step type so the diffusion and language planners share one
/// search implementation — and one set of guarantees — while keeping their own
/// state and scoring.
#[derive(Debug, Clone)]
pub struct Beam {
    budget: Budget,
}

impl Beam {
    pub fn new(budget: Budget) -> Self {
        Self { budget }
    }

    /// Search for the best next step.
    ///
    /// `expand(path, remaining)` returns the `(step, score_delta)` options
    /// available after `path`; an empty return marks that path **complete** —
    /// it is carried forward and still competes, but is never extended again.
    ///
    /// `remaining` is how many evaluations the budget can still afford. A cheap
    /// scoring function may ignore it: any excess options are truncated, so the
    /// ceiling holds either way. An `expand` that calls a model must consult it
    /// **before doing the work**, because truncation after the fact bounds the
    /// bookkeeping, not the compute.
    ///
    /// Ties are broken by the order `expand` returns candidates, so a planner
    /// with a deterministic `expand` is itself deterministic.
    ///
    /// # Comparing paths
    ///
    /// Only paths at the **same** expansion level are ever compared. That
    /// matters: score deltas accumulate, so a partially expanded level would
    /// otherwise always look better (log-probabilities) or always worse
    /// (costs) than a complete one purely because it is shorter. The winner is
    /// the top of the last fully expanded level.
    ///
    /// A path that completes early keeps its accumulated score and is compared
    /// against longer paths as-is. For the trajectory planner that is the
    /// intended reading — reaching `sigma_min` in fewer steps is a success, not
    /// a truncation. A caller that wants an end-of-path bonus or penalty should
    /// fold it into that path's final score delta.
    pub fn search<T, F>(&self, mut expand: F) -> Plan<T>
    where
        T: Clone,
        F: FnMut(&Path<T>, usize) -> Vec<(T, f64)>,
    {
        let mut spend = Spend::new(self.budget);
        let mut frontier = vec![Path::<T>::root()];
        // Top of the last *fully* expanded level. Only this is ever committed,
        // so a level cut short by the budget never decides the outcome.
        let mut settled: Option<Path<T>> = None;

        for _ in 0..=self.budget.max_depth {
            let mut next: Vec<Path<T>> = Vec::new();
            let mut extended_any = false;
            let mut cut_short = false;

            for path in &frontier {
                let remaining = spend.remaining();
                if remaining == 0 {
                    cut_short = true;
                    break;
                }

                let mut options = expand(path, remaining);
                if options.is_empty() {
                    // Complete: carry it forward so it keeps competing.
                    if path.depth() > 0 {
                        next.push(path.clone());
                    }
                    continue;
                }
                if options.len() > remaining {
                    // The caller ignored the allowance. Honour the ceiling
                    // anyway; the level is then incomplete.
                    options.truncate(remaining);
                    cut_short = true;
                }

                for (step, delta) in options {
                    debug_assert!(spend.can_spend(), "truncation should have prevented this");
                    spend.spend();
                    next.push(path.extended(step, delta));
                    extended_any = true;
                }
            }

            if cut_short {
                // This level is incomplete, so its members are not comparable
                // with each other. Fall back to the last settled level -- or,
                // if none exists yet, to the best of what this level did
                // produce, so even a budget of one yields a usable step.
                let partial = best_of(next);
                return Plan {
                    best: settled.or(partial),
                    evaluations: spend.used(),
                    budget_exhausted: true,
                };
            }

            if next.is_empty() {
                break;
            }

            // A stable sort means equal scores retain expansion order, so ties
            // resolve deterministically.
            next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            next.truncate(self.budget.beam_width.max(1));

            settled = next.first().cloned();
            frontier = next;

            if !extended_any {
                // Every surviving path is complete; further levels are a no-op.
                break;
            }
        }

        Plan {
            best: settled,
            evaluations: spend.used(),
            budget_exhausted: false,
        }
    }
}

/// Highest-scoring path, or `None` when there are none.
fn best_of<T: Clone>(paths: Vec<Path<T>>) -> Option<Path<T>> {
    paths
        .into_iter()
        .filter(|p| p.depth() > 0)
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
}

/// Outcome of a search.
#[derive(Debug, Clone)]
pub struct Plan<T> {
    /// The best path found, if any candidate was available at all.
    pub best: Option<Path<T>>,
    /// Evaluations actually spent.
    pub evaluations: usize,
    /// Whether the budget stopped the search early.
    pub budget_exhausted: bool,
}

impl<T: Clone> Plan<T> {
    /// The step to actually take now.
    ///
    /// Lookahead informs the choice but only the *first* step is committed —
    /// the rest of the path is re-planned once its consequences are observed,
    /// which is what makes the search worth repeating rather than running once.
    pub fn commit(&self) -> Option<&T> {
        self.best.as_ref().and_then(Path::head)
    }

    pub fn depth(&self) -> usize {
        self.best.as_ref().map_or(0, Path::depth)
    }

    pub fn score(&self) -> f64 {
        self.best.as_ref().map_or(f64::NEG_INFINITY, |p| p.score)
    }
}

// ------------------------------------------------- diffusion trajectory --

/// One candidate move along a diffusion trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryStep {
    /// Noise level to step to.
    pub sigma: f64,
    /// How many blocks to execute for this step.
    pub width: usize,
}

/// Plans the next `(sigma, span width)` of a diffusion trajectory.
///
/// Generalizes `Strategy::Adaptive`, which widens or narrows the span by one
/// based on the current confidence and no lookahead. Here several
/// `(sigma, width)` pairs are scored, optionally with a short rollout, and the
/// best first move is committed.
#[derive(Debug, Clone)]
pub struct TrajectoryPlanner {
    budget: Budget,
    /// Multiplicative sigma steps to consider, e.g. `[0.5, 0.7, 0.9]`.
    sigma_ratios: Vec<f64>,
    /// Span widths to consider.
    widths: Vec<usize>,
    /// Penalty per executed block, in score units.
    cost_per_block: f64,
}

impl Default for TrajectoryPlanner {
    fn default() -> Self {
        Self {
            budget: Budget::default(),
            sigma_ratios: vec![0.4, 0.6, 0.8],
            widths: vec![1, 2],
            cost_per_block: 0.01,
        }
    }
}

impl TrajectoryPlanner {
    pub fn new(budget: Budget) -> Self {
        Self { budget, ..Self::default() }
    }

    pub fn with_ratios(mut self, ratios: Vec<f64>) -> Self {
        self.sigma_ratios = ratios;
        self
    }

    pub fn with_widths(mut self, widths: Vec<usize>) -> Self {
        self.widths = widths;
        self
    }

    pub fn with_cost_per_block(mut self, cost: f64) -> Self {
        self.cost_per_block = cost;
        self
    }

    pub fn candidates_per_expansion(&self) -> usize {
        self.sigma_ratios.len() * self.widths.len()
    }

    /// Plan from `sigma`, stopping at `sigma_min`.
    ///
    /// `quality(sigma, width)` estimates how good arriving at `sigma` having
    /// executed `width` blocks is; higher is better. The planner subtracts the
    /// compute cost itself, so `quality` is purely about the result.
    ///
    /// This is the cheap form, for a scoring function with no state and no
    /// per-call cost worth budgeting. A rollout that actually runs the model
    /// needs [`Self::plan_with`].
    pub fn plan<F>(&self, sigma: f64, sigma_min: f64, mut quality: F) -> Plan<TrajectoryStep>
    where
        F: FnMut(f64, usize) -> f64,
    {
        self.plan_with(sigma, sigma_min, |_path, next, width, _remaining| {
            Some(quality(next, width))
        })
    }

    /// Plan from `sigma`, with the rollout state and the compute budget in hand.
    ///
    /// `score(prefix, sigma, width, remaining)` is called once per candidate.
    /// `prefix` is the hypothesized path leading to this candidate, which is
    /// what lets a caller look up the rolled-forward state it belongs to;
    /// `remaining` is the evaluation allowance still unspent. Returning `None`
    /// declines the candidate — the way to stop before spending compute the
    /// budget cannot cover.
    ///
    /// Unlike [`Self::plan`], the returned score is used as given except for
    /// the block-cost subtraction, which stays the planner's job so that the
    /// cost model is stated in exactly one place.
    pub fn plan_with<F>(&self, sigma: f64, sigma_min: f64, mut score: F) -> Plan<TrajectoryStep>
    where
        F: FnMut(&Path<TrajectoryStep>, f64, usize, usize) -> Option<f64>,
    {
        let beam = Beam::new(self.budget);
        beam.search(|path: &Path<TrajectoryStep>, remaining: usize| {
            let current = path.steps.last().map_or(sigma, |s| s.sigma);
            if current <= sigma_min {
                return Vec::new();
            }
            let mut options = Vec::with_capacity(self.candidates_per_expansion());
            for ratio in &self.sigma_ratios {
                // Never step past sigma_min, and never fail to make progress:
                // a ratio of 1.0 would let the search stall forever inside its
                // depth budget without advancing the trajectory.
                let next = (current * ratio).max(sigma_min);
                if next >= current {
                    continue;
                }
                for width in &self.widths {
                    let left = remaining.saturating_sub(options.len());
                    if left == 0 {
                        return options;
                    }
                    let Some(value) = score(path, next, *width, left) else {
                        continue;
                    };
                    let adjusted = value - self.cost_per_block * *width as f64;
                    options.push((TrajectoryStep { sigma: next, width: *width }, adjusted));
                }
            }
            options
        })
    }
}

// ------------------------------------------------------- LM lookahead --

/// A candidate token together with the score of committing to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenStep {
    pub token: u16,
    /// Log-probability of this token given the prefix.
    pub logprob: f64,
}

/// Chooses the next token by scoring candidate *continuations* rather than
/// committing to the immediate arg-max.
///
/// Greedy decoding is myopic: a token that looks best now can lead into a
/// continuation the model itself considers unlikely. Looking ahead a few tokens
/// and scoring the whole path catches that, at a cost the [`Budget`] bounds.
#[derive(Debug, Clone)]
pub struct LookaheadDecoder {
    budget: Budget,
    /// Tokens considered at each position.
    top_k: usize,
    /// Penalty per token of lookahead, discouraging length-driven scores.
    length_penalty: f64,
}

impl Default for LookaheadDecoder {
    fn default() -> Self {
        Self { budget: Budget::default(), top_k: 4, length_penalty: 0.0 }
    }
}

impl LookaheadDecoder {
    pub fn new(budget: Budget, top_k: usize) -> Self {
        Self { budget, top_k: top_k.max(1), length_penalty: 0.0 }
    }

    pub fn with_length_penalty(mut self, penalty: f64) -> Self {
        self.length_penalty = penalty;
        self
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Plan the next token.
    ///
    /// `next_logprobs(prefix)` returns candidate `(token, log-probability)`
    /// pairs given the tokens committed so far *plus* the ones the current path
    /// has hypothesized. Returning fewer than `top_k` is fine; returning an
    /// empty vec ends the path.
    pub fn plan<F>(&self, prefix: &[u16], mut next_logprobs: F) -> Plan<TokenStep>
    where
        F: FnMut(&[u16]) -> Vec<(u16, f64)>,
    {
        let beam = Beam::new(self.budget);
        beam.search(|path: &Path<TokenStep>, remaining: usize| {
            let mut context: Vec<u16> = prefix.to_vec();
            context.extend(path.steps.iter().map(|s| s.token));

            let mut options = next_logprobs(&context);
            options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            options.truncate(self.top_k.min(remaining));

            options
                .into_iter()
                .map(|(token, logprob)| {
                    (TokenStep { token, logprob }, logprob - self.length_penalty)
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_is_never_exceeded() {
        // The guarantee the whole design rests on: lookahead multiplies work,
        // so a planner without an enforced ceiling occasionally takes minutes
        // to emit one step. Driven adversarially -- every expansion offers many
        // options and no path ever terminates on its own.
        for max_evaluations in [1usize, 5, 17, 64] {
            let beam = Beam::new(Budget {
                max_evaluations,
                max_depth: 10,
                beam_width: 8,
            });
            let plan = beam.search(|_path: &Path<u32>, _budget: usize| {
                (0..6).map(|i| (i, i as f64)).collect::<Vec<_>>()
            });
            assert!(
                plan.evaluations <= max_evaluations,
                "spent {} against a budget of {max_evaluations}",
                plan.evaluations
            );
        }
    }

    #[test]
    fn test_depth_zero_reduces_to_greedy() {
        // A rollout of depth 0 must be exactly "score the immediate options and
        // take the best", i.e. the policy the crate already had. If it is not,
        // the planner is not a generalization but a different algorithm.
        let beam = Beam::new(Budget::greedy());
        let plan = beam.search(|path: &Path<u32>, _budget: usize| {
            if path.depth() > 0 {
                return Vec::new();
            }
            vec![(10, 0.5), (20, 0.9), (30, 0.1)]
        });
        assert_eq!(plan.commit(), Some(&20), "greedy must take the best option");
        assert_eq!(plan.depth(), 1, "depth 0 commits exactly one step");
        assert_eq!(plan.evaluations, 3);
    }

    #[test]
    fn test_lookahead_can_beat_greedy() {
        // The reason to look ahead at all: an option that scores best now can
        // lead somewhere worse. Here `1` looks better immediately but is a dead
        // end, while `2` pays off one step later.
        let expand = |path: &Path<u32>, _budget: usize| -> Vec<(u32, f64)> {
            match path.steps.as_slice() {
                [] => vec![(1, 1.0), (2, 0.9)],
                [1] => vec![(11, 0.0)],
                [2] => vec![(22, 5.0)],
                _ => Vec::new(),
            }
        };

        let greedy = Beam::new(Budget::greedy()).search(expand);
        assert_eq!(greedy.commit(), Some(&1), "greedy takes the locally best step");

        let looked = Beam::new(Budget { max_evaluations: 64, max_depth: 1, beam_width: 4 })
            .search(expand);
        assert_eq!(
            looked.commit(),
            Some(&2),
            "one step of lookahead should prefer the better continuation"
        );
        assert!(looked.score() > greedy.score());
    }

    #[test]
    fn test_search_terminates_when_no_candidates_remain() {
        let beam = Beam::new(Budget::default());
        let plan = beam.search(|_: &Path<u32>, _budget: usize| Vec::new());
        assert!(plan.best.is_none(), "no candidates means no plan");
        assert_eq!(plan.commit(), None);
        assert_eq!(plan.evaluations, 0);
    }

    #[test]
    fn test_ties_resolve_deterministically() {
        // A planner with a deterministic expansion must itself be
        // deterministic, or a run cannot be reproduced from its seed.
        let expand = |path: &Path<u32>, _budget: usize| -> Vec<(u32, f64)> {
            if path.depth() > 0 {
                return Vec::new();
            }
            vec![(7, 1.0), (8, 1.0), (9, 1.0)]
        };
        let beam = Beam::new(Budget::greedy());
        let first = beam.search(expand).commit().copied();
        for _ in 0..5 {
            assert_eq!(beam.search(expand).commit().copied(), first);
        }
        assert_eq!(first, Some(7), "the earliest candidate wins a tie");
    }

    #[test]
    fn test_worst_case_is_reported_honestly() {
        let b = Budget { max_evaluations: 1000, max_depth: 2, beam_width: 3 };
        // 3 paths x 4 candidates x 3 levels.
        assert_eq!(b.worst_case(4), 36);
        // ...and it never claims more than the hard cap.
        let capped = Budget { max_evaluations: 10, max_depth: 5, beam_width: 5 };
        assert_eq!(capped.worst_case(10), 10);
    }

    #[test]
    fn test_trajectory_planner_descends_toward_sigma_min() {
        // Every planned step must reduce sigma and never undershoot the floor,
        // or the "trajectory" is not one.
        let planner = TrajectoryPlanner::new(Budget { max_evaluations: 200, max_depth: 2, beam_width: 3 });
        let plan = planner.plan(80.0, 0.002, |sigma, _width| -sigma);

        let path = plan.best.as_ref().expect("a plan should exist");
        let mut previous = 80.0;
        for step in &path.steps {
            assert!(step.sigma < previous, "sigma must decrease: {} !< {previous}", step.sigma);
            assert!(step.sigma >= 0.002, "must not undershoot sigma_min");
            previous = step.sigma;
        }
        assert!(plan.commit().is_some());
    }

    #[test]
    fn test_trajectory_planner_stops_at_sigma_min() {
        let planner = TrajectoryPlanner::new(Budget::default());
        // Already at the floor: nothing left to plan.
        let plan = planner.plan(0.002, 0.002, |_, _| 1.0);
        assert!(plan.best.is_none());
        assert_eq!(plan.evaluations, 0);
    }

    #[test]
    fn test_trajectory_cost_discourages_wide_spans() {
        // With a quality function indifferent to width, the cost term must
        // break the tie toward the cheaper span -- otherwise "adaptive depth"
        // would always widen.
        let planner = TrajectoryPlanner::new(Budget::greedy())
            .with_widths(vec![1, 4])
            .with_cost_per_block(0.1);
        let plan = planner.plan(1.0, 0.002, |_sigma, _width| 1.0);
        assert_eq!(plan.commit().map(|s| s.width), Some(1), "cheaper span should win");

        // ...and a quality function that genuinely favours width overrides it.
        let planner = planner.with_cost_per_block(0.001);
        let plan = planner.plan(1.0, 0.002, |_s, width| width as f64);
        assert_eq!(plan.commit().map(|s| s.width), Some(4));
    }

    #[test]
    fn test_lookahead_decoder_prefers_the_better_continuation() {
        // The language-side counterpart: token 1 is likelier now, but leads
        // into a continuation the model itself considers unlikely.
        let decoder = LookaheadDecoder::new(
            Budget { max_evaluations: 64, max_depth: 1, beam_width: 4 },
            2,
        );
        let plan = decoder.plan(&[65], |context: &[u16]| match context.len() {
            1 => vec![(1, -0.1), (2, -0.5)],
            2 if context[1] == 1 => vec![(11, -6.0)],
            2 => vec![(22, -0.2)],
            _ => Vec::new(),
        });
        assert_eq!(
            plan.commit().map(|s| s.token),
            Some(2),
            "lookahead should avoid the dead end"
        );

        // Greedy, by contrast, takes the locally likelier token.
        let greedy = LookaheadDecoder::new(Budget::greedy(), 2)
            .plan(&[65], |c: &[u16]| {
                if c.len() == 1 { vec![(1, -0.1), (2, -0.5)] } else { Vec::new() }
            });
        assert_eq!(greedy.commit().map(|s| s.token), Some(1));
    }

    #[test]
    fn test_lookahead_respects_top_k() {
        // Only the `k` most likely tokens are ever expanded, which is what
        // keeps the branching factor bounded independently of vocabulary size.
        let decoder = LookaheadDecoder::new(Budget::greedy(), 2);
        let mut seen = Vec::new();
        let plan = decoder.plan(&[0], |_| {
            vec![(1, -0.1), (2, -0.2), (3, -0.3), (4, -9.0)]
        });
        if let Some(path) = &plan.best {
            seen.extend(path.steps.iter().map(|s| s.token));
        }
        assert_eq!(plan.evaluations, 2, "only top-2 should be expanded");
        assert!(seen.iter().all(|t| *t == 1 || *t == 2));
    }

    #[test]
    fn test_only_the_first_step_is_committed() {
        // Lookahead informs the choice, but the rest of the path is a
        // hypothesis to be re-planned once its consequences are observed.
        let beam = Beam::new(Budget { max_evaluations: 64, max_depth: 3, beam_width: 2 });
        let plan = beam.search(|path: &Path<u32>, _budget: usize| {
            if path.depth() >= 3 {
                return Vec::new();
            }
            vec![(path.depth() as u32, 1.0)]
        });
        assert_eq!(plan.depth(), 3, "the search explored three steps");
        assert_eq!(plan.commit(), Some(&0), "but only the first is committed");
    }

    #[test]
    fn test_a_path_that_ends_early_still_competes() {
        // Reaching the goal in fewer steps is a success, not a truncation. If
        // completed paths were dropped from the frontier, the planner would
        // systematically prefer whichever branch happens to keep going.
        let beam = Beam::new(Budget { max_evaluations: 64, max_depth: 3, beam_width: 4 });
        let plan = beam.search(|path: &Path<u32>, _budget: usize| match path.steps.as_slice() {
            [] => vec![(1, 10.0), (2, 1.0)],
            // `1` is done immediately with a score nothing else can reach...
            [1] => Vec::new(),
            // ...while `2` keeps accumulating small gains.
            [2, ..] if path.depth() < 3 => vec![(2, 1.0)],
            _ => Vec::new(),
        });
        assert_eq!(plan.commit(), Some(&1), "the completed path should win");
        assert_eq!(plan.depth(), 1);
    }

    #[test]
    fn test_an_incomplete_level_never_decides_the_outcome() {
        // Score deltas accumulate, so a half-expanded level is not comparable
        // with a full one. When the budget cuts a level short, the planner must
        // fall back to the last level it fully evaluated.
        //
        // Level 0 costs 2 evaluations and settles on `1`. Level 1 would offer
        // `2` a large gain, but the budget stops after seeing only `1`'s
        // continuation -- which must not be mistaken for the best of level 1.
        let beam = Beam::new(Budget { max_evaluations: 3, max_depth: 2, beam_width: 4 });
        let plan = beam.search(|path: &Path<u32>, _budget: usize| match path.steps.as_slice() {
            [] => vec![(1, 1.0), (2, 0.9)],
            [1] => vec![(11, -50.0)],
            [2] => vec![(22, 100.0)],
            _ => Vec::new(),
        });
        assert!(plan.budget_exhausted, "the budget should have bitten");
        assert_eq!(plan.evaluations, 3);
        assert_eq!(plan.commit(), Some(&1), "fall back to the settled level");
        assert_eq!(plan.depth(), 1, "not the partially expanded one");
    }

    #[test]
    fn test_a_single_evaluation_still_yields_a_step() {
        // Degenerate budget: no level ever completes. Returning nothing would
        // leave the caller with no move at all, so the best of what was seen is
        // used -- the one case where a partial level is allowed to decide.
        let beam = Beam::new(Budget { max_evaluations: 1, max_depth: 5, beam_width: 4 });
        let plan = beam.search(|_: &Path<u32>, _budget: usize| vec![(7, 1.0), (8, 2.0)]);
        assert!(plan.budget_exhausted);
        assert_eq!(plan.evaluations, 1);
        assert_eq!(plan.commit(), Some(&7));
    }

    #[test]
    fn test_beam_width_bounds_the_frontier() {
        // Without truncation the frontier grows as candidates^depth. The width
        // is what keeps lookahead affordable, so it must actually bind.
        let width = 2usize;
        let beam = Beam::new(Budget { max_evaluations: usize::MAX, max_depth: 3, beam_width: width });
        let mut widest = 0usize;
        let mut level_calls = 0usize;
        beam.search(|path: &Path<u32>, _budget: usize| {
            if path.depth() == 1 {
                level_calls += 1;
            }
            widest = widest.max(level_calls);
            (0..5).map(|i| (i, i as f64)).collect::<Vec<_>>()
        });
        assert_eq!(widest, width, "at most `beam_width` paths are expanded per level");
    }

    #[test]
    fn test_greedy_is_contained_in_beam_one() {
        // beam(1) with no lookahead must reproduce greedy exactly -- the
        // containment that makes the planner a generalization rather than a
        // replacement.
        let expand = |path: &Path<u32>, _budget: usize| -> Vec<(u32, f64)> {
            if path.depth() > 0 {
                return Vec::new();
            }
            vec![(3, 0.2), (5, 0.7), (9, 0.4)]
        };
        let greedy = Beam::new(Budget::greedy()).search(expand);
        let beam_one = Beam::new(Budget { max_evaluations: usize::MAX, max_depth: 0, beam_width: 1 })
            .search(expand);
        assert_eq!(greedy.commit(), beam_one.commit());
        assert_eq!(greedy.score(), beam_one.score());
    }

    #[test]
    fn test_expand_is_told_what_it_can_afford_before_it_works() {
        // Truncating options after the fact bounds the bookkeeping, not the
        // compute. An expand that runs a model must be able to stop *before*
        // spending, so the allowance it is handed has to be accurate.
        let beam = Beam::new(Budget { max_evaluations: 5, max_depth: 3, beam_width: 2 });
        let mut allowances = Vec::new();
        let mut work = 0usize;
        let plan = beam.search(|_path: &Path<u32>, remaining: usize| {
            allowances.push(remaining);
            // Do only what is affordable, exactly as a model-calling planner
            // would.
            let n = remaining.min(2);
            work += n;
            (0..n as u32).map(|i| (i, i as f64)).collect::<Vec<_>>()
        });
        assert!(work <= 5, "did {work} units of work against a budget of 5");
        assert_eq!(plan.evaluations, work, "every unit of work is charged once");
        assert!(
            allowances.windows(2).all(|w| w[1] <= w[0]),
            "the allowance must shrink monotonically: {allowances:?}"
        );
        assert_eq!(allowances[0], 5);
    }

    #[test]
    fn test_plan_with_can_decline_a_candidate() {
        // `None` is how a rollout says "I cannot afford this one". A declined
        // candidate must vanish from the search rather than score zero, which
        // would silently make it a contender.
        let planner = TrajectoryPlanner::new(Budget::greedy())
            .with_ratios(vec![0.5])
            .with_widths(vec![1, 2, 3])
            .with_cost_per_block(0.0);
        let mut offered = Vec::new();
        let plan = planner.plan_with(1.0, 0.002, |_path, _sigma, width, _left| {
            offered.push(width);
            // Refuse the width that would otherwise win.
            if width == 3 { None } else { Some(width as f64) }
        });
        assert_eq!(offered, vec![1, 2, 3], "every candidate is still offered");
        assert_eq!(
            plan.commit().map(|s| s.width),
            Some(2),
            "the declined candidate must not be selectable"
        );
        assert_eq!(plan.evaluations, 2, "and it is not charged for");
    }

    #[test]
    fn test_trajectory_rollout_sees_its_own_prefix() {
        // Depth is only useful if a rollout can locate the state a hypothesized
        // path leads to. The prefix handed to the scorer is that address.
        let planner = TrajectoryPlanner::new(Budget { max_evaluations: 64, max_depth: 2, beam_width: 1 })
            .with_ratios(vec![0.5])
            .with_widths(vec![1]);
        let mut depths = Vec::new();
        planner.plan_with(1.0, 0.1, |path, sigma, _w, _left| {
            depths.push((path.depth(), sigma));
            Some(0.0)
        });
        assert_eq!(depths.len(), 3, "1.0 -> 0.5 -> 0.25 -> 0.125");
        assert_eq!(depths[0].0, 0);
        assert_eq!(depths[1].0, 1);
        assert_eq!(depths[2].0, 2);
        assert!((depths[1].1 - 0.25).abs() < 1e-12, "the prefix determines the sigma reached");
    }
}
