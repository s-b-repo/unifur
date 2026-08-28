# Next-Step Planning

Choose the next step by scoring candidates — and, where it pays, short rollouts
of what follows each one — instead of always taking the greedy one. Two
planners: one over diffusion trajectories, one over tokens.

> **In this repository.** `planner.rs` (`Beam`, `Budget`, `Spend`, `Path`,
> `Plan`, `TrajectoryPlanner`, `TrajectoryStep`, `LookaheadDecoder`,
> `TokenStep`), `multi_block.rs` (`PlannedConfig`, `PlanTrace`,
> `DblockClassifier::sample_planned`), `lm.rs`
> (`LanguageModel::generate_lookahead`, `LookaheadStats`). CLI:
> `dblocks sample --planned`, `dblocks lm generate --lookahead N`.
> Certificates: the `planner` group.

---

## Two planners, one search

The two state spaces genuinely differ, so they are **not** forced into one
abstraction:

| Planner | State | Candidates | Score |
|---|---|---|---|
| `TrajectoryPlanner` | latent + noise level | `(sigma, span width)` | confidence of the x0 estimate, less compute cost, plus progress |
| `LookaheadDecoder` | token prefix | next token | the model's own log-probability |

What they *do* share is the search. `Beam` is generic over the step type, so the
search policy — and the guarantees that go with it — is written and certified
once, while each planner keeps the state and scoring that actually fit it.

---

## The budget is a type

Lookahead multiplies work: `beam × depth × candidates` model calls per committed
step. A planner without an enforced ceiling is a planner that occasionally takes
minutes to emit one token.

```rust
pub struct Budget {
    pub max_evaluations: usize,  // total candidate evaluations
    pub max_depth: usize,        // longest rollout
    pub beam_width: usize,       // paths kept between expansions
}
```

Both limits bind. Depth alone is not enough — a wide beam at shallow depth is
just as expensive.

Crucially, **`expand` is told its allowance before it does the work**:

```rust
beam.search(|path, remaining| {
    if remaining == 0 { return Vec::new(); }
    // ... call the model at most `remaining` times
})
```

Truncating the returned options *after the fact* would bound the bookkeeping,
not the compute — the model calls would already have happened. The search still
truncates as a backstop, so the ceiling holds even against an `expand` that
ignores the allowance, and `Budget::worst_case(candidates)` reports the spend
before a caller commits to it.

---

## Only fully expanded levels are compared

This is the subtlety that a first implementation gets wrong.

Score deltas accumulate. Under log-probabilities every extra step makes a path
score *worse*; under costs, better. So comparing a depth-1 path against a
depth-2 path on raw score does not measure quality — it measures length. A
planner that did so would always commit the shallowest path and lookahead would
be decorative.

`Beam::search` therefore compares only paths at the same expansion level, and
commits the top of the **last fully expanded** level. When the budget cuts a
level short, the search falls back to the previous settled level rather than
letting a half-evaluated one decide. (A budget so small that no level ever
completes still returns the best candidate seen — otherwise the caller would
have no move at all.)

A path that runs out of candidates is **complete**: it is carried forward and
keeps competing at its own score, never extended again. For the trajectory
planner that is the intended reading — reaching `sigma_min` in fewer steps is a
success, not a truncation.

---

## Planned diffusion sampling

```bash
dblocks sample --planned --plan-depth 2 --plan-beam 3 --plan-budget 32
```

```rust
let (logits, stats, trace) = model.sample_planned(&pixels, &PlannedConfig {
    budget: Budget { max_evaluations: 48, max_depth: 2, beam_width: 3 },
    ..PlannedConfig::default()
}, &mut rng);
println!("mean lookahead depth {:.2}", trace.mean_depth());
```

Every other strategy follows a schedule fixed before the first model call. This
one decides each step from evidence.

**The progress term is load-bearing.** A candidate's score is the confidence of
the x0 estimate less the compute its span costs — and without a reward for
descending, the planner takes the smallest jump available every time, because
the most accurate single step is always the shortest one, and never reaches
`sigma_min`. Progress is credited in **log-sigma**, the coordinate the solvers
integrate in (`lambda = -log sigma`).

**Where lookahead earns its keep.** A candidate is scored at the sigma it
*departs from*, so at depth 0 the planner cannot tell a good jump from an
over-long one — that choice falls to the progress-versus-cost trade alone. Depth
1 is what reveals it: an over-long jump lands somewhere the next estimate is
less confident, and that shows up in the path score. Lookahead here is not a
refinement of the greedy signal; it supplies information the greedy signal does
not contain.

Cost: one model call per distinct span width per expanded node, so
`model_calls ≤ evaluations` always. Rolled-forward latents are addressed by the
path that produced them, so a deeper level is scored from where its hypothesis
actually lands rather than from a stand-in.

---

## Lookahead decoding

```bash
dblocks lm generate --prompt "Hello" --lookahead 2 --beam 3 --budget 32
```

```rust
let (tokens, stats) = model.generate_lookahead(&prompt, 64, 4, budget, &device);
println!("{:.2} forward passes per token", stats.calls_per_token());
```

Greedy decoding is myopic: the likeliest token now can open onto a continuation
the model itself rates poorly. Lookahead scores whole continuations and commits
only their first token, then re-plans.

**The model is its own verifier** — a continuation's score is the sum of its own
log-probabilities. No second network is involved, which keeps this a decoding
change rather than a training one.

Beam paths share prefixes heavily, so continuations are cached per committed
token; `LookaheadStats` reports both `model_calls` and `evaluations`, and their
difference is exactly the cache hits.

---

## Containment

The planners are generalizations, not replacements, and that is certified rather
than asserted:

- `planner/depth_zero_is_greedy` — a rollout of depth 0 selects the
  highest-scoring immediate candidate. Exactly the policy the crate already had.
- `planner/greedy_within_beam_one` — `beam(1)` at depth 0 reproduces greedy
  step for step *and* score for score.
- `lm::test_lookahead_with_no_depth_reproduces_greedy_exactly` — the same claim
  end to end through the real model, at the same one-call-per-token cost.

So a pipeline can leave the planner permanently in place with depth as the only
knob, and depth 0 costs nothing.

The rest of the group: the budget is never exceeded under any
`(budget, depth, width)`; only a plan's first step is ever committed; every
planned step lowers sigma and none undershoots `sigma_min`; and lookahead
actually defeats a myopic trap, in the beam and in the language decoder alike.

---

## What is heuristic

The step-value model is hand-written — confidence, cost, progress — not learned.
A learned value head is the natural next step (roadmap 22.7) and needs training
data this repository does not have. Everything about the *search* is exact; what
it searches over is a heuristic, and the two should not be confused.

---

See also: [Multi-Block Denoising](Multi-Block-Denoising.md) ·
[Adaptive Depth](Adaptive-Depth.md) · [Hybrid Loop Graph](Hybrid-Loop-Graph.md) ·
[Language Modeling](Language-Modeling.md) · [Quality Gate](Quality-Gate.md)
