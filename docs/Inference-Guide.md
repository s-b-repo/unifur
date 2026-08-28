# Inference Guide

Sampling, benchmarking, and the library API.

---

## In this repository

| | |
|---|---|
| **Modules** | `src/multi_block.rs`, `src/infer.rs`, `src/solver.rs` |
| **Key types** | `MultiBlockConfig`, `Strategy`, `SolverKind`, `InferenceEngine` |
| **CLI** | `dblocks sample`, `dblocks infer`, `dblocks bench` |

---

## Sampling

```bash
./target/release/dblocks sample \
    --checkpoint checkpoints/dblocks-<hash>.mpk \
    --num-blocks 4 --num-inference-steps 6 \
    --solver dpmpp3m --strategy adaptive --k 3 \
    --gate tightening --precision bf16 --precision-switch 1.0
```

Without `--checkpoint` a randomly initialized model is used. That is useful for
checking plumbing and measuring cost, but a DiT-initialized model outputs
exactly zero logits, so every class comes back at probability `1/num_labels` —
which is the `dit_zero_init` certificate doing its job, not a bug.

### What it reports

```text
solver=dpmpp3m strategy=adaptive gate=tightening precision=bf16
schedule (descending): [80.0, 0.827, 0.408, 0.222, 0.110, 0.002]
model calls: 6 | layers executed: 27 | mean span: 5.40 | gated samples: 0 | reduced-precision windows: 1
```

`layers executed` is the honest cost measure. `model calls` alone understates a
wide-span strategy: `parallel-2` makes the same number of calls as `sequential`
but runs nearly twice the layers per call.

---

## Solvers

All five integrate `dz/dσ = (z - x0)/σ`, equivalently `dz/dλ = -z + x0(λ)` with
`λ = -log σ`.

| `--solver` | Order | Calls/step | Notes |
|---|---|---|---|
| `euler` | 1 | 1 | The original DiffusionBlocks path |
| `heun` | 2 | **2** | Predictor–corrector; costs an extra model call |
| `ddim` | 1 | 1 | Ancestral noise with `eta`; `eta = 0` is exactly Euler |
| `dpmpp2m` | 2 | 1 | Published DPM-Solver++(2M) coefficients |
| `dpmpp3m` | 3 | 1 | Exact-kernel exponential integrator |

Two implementation notes worth knowing:

- **`dpmpp2m` is deliberately not an exact exponential integrator.** It weights
  the slope by `(1 - e^-h)h/2` rather than the exact `h - 1 + e^-h`; the two
  differ at order `h³`. This is kept bit-faithful to the published algorithm so
  results reproduce, and pinned by a test so nobody "fixes" it into a different
  method by accident.
- **`ddim` splits each step** into a deterministic descent to `sigma_down` and
  an injection of `sigma_up` noise, with
  `sigma_down² + sigma_up² = sigma_next²`. Noise is *redistributed*, not added
  — certified by `ancestral_variance_preserved`. The noise comes from the
  caller's RNG, so a seeded trajectory replays exactly.

Convergence order is measured, not assumed: `verify.rs` fits the log-log slope
of error against step count for each solver and requires it to reach the
classical order.

---

## Strategies

How many transformer blocks each sampling window executes.

| `--strategy` | Behaviour |
|---|---|
| `sequential` | One block per window (original) |
| `parallel --k N` | `N` adjacent blocks jointly |
| `hybrid --k N` | Sequential above `0.3 × sigma_max`, parallel below |
| `adaptive --k N` | Widen while unconfident, narrow again once confident |

Strategy and solver are **independent axes** — any combination works, because
both drive the same step-wise `SolverState`. The integration test exercises the
full 4 × 5 matrix.

Parallel sampling is only legitimate if a joint span reproduces the sequential
composition of the same blocks. That is not assumed: the cross-fork consistency
term trains it directly (see [Consistency Training](Consistency-Training.md)).

### Planned sampling

`--planned` replaces the schedule entirely: each step's `(sigma, span)` is
chosen by scoring candidates and rolling the promising ones forward.

```bash
dblocks sample --planned --plan-depth 2 --plan-beam 3 --plan-budget 32
```

```text
planned: 6 steps | mean lookahead depth 3.00 | 108 evaluations | 0 step(s) cut short by the budget
  step 0: sigma -> 24.00000 with a 1-block span
  ...
  planning overhead: 78% of executed layers
```

`--plan-depth 0` is certified to be exactly the greedy policy, so the flag can
stay on with depth as the only knob. The planning overhead is reported
separately from the sampling cost rather than blended into it, because it is
substantial. Full detail: [Next-Step Planning](Next-Step-Planning.md).

### Guidance, normalization and ensembling

```bash
dblocks sample --guidance 3.0 --guidance-rescale 0.7
dblocks sample --logit-norm standardize
dblocks sample --ensemble probability
```

Each has an exact identity setting (`--guidance 1.0`, `--logit-norm none`, no
`--ensemble`), and each identity is certified — so these can be left in a
pipeline with the value as the only knob. See
[Accuracy Improvements](Accuracy-Improvements.md).

---

## Quality gates

```bash
--gate lenient      # reject only degenerate transitions (default)
--gate strict       # conservative thresholds on every block
--gate tightening   # per-block, monotonically stricter at low sigma
```

Samples failing a gate keep their previous latent instead of taking the bad
update. Per-block rejection rates are reported. See
[Quality Gate](Quality-Gate.md).

---

## Mixed precision

```bash
--precision bf16 --precision-switch 1.0
```

Runs windows at `σ ≥ 1.0` in emulated bf16 and the rest in f32. At high sigma
the latent is dominated by noise of magnitude `σ`, so a relative representation
error of `2⁻⁸` is far below the noise floor; near `sigma_min` the estimate is
nearly converged and the same error no longer is.

**This is emulation, not acceleration.** Values are rounded onto the target
format's grid (correct round-to-nearest-even, subnormals and overflow) while
arithmetic stays in f32, so it models representation error faithfully and is
*slower* than plain f32. It answers "how much accuracy would bf16 cost here?",
which is the question worth answering before a backend supports the format
natively.

---

## Benchmarking

```bash
./target/release/dblocks bench --num-blocks 4 --num-inference-steps 6 --repeats 3
```

Sweeps every solver against every strategy:

```text
solver     strategy        mean ms  model calls   layers      agree
--------------------------------------------------------------------
euler      sequential      343.2ms            6       15         8/8
euler      parallel-2      563.4ms            6       27         8/8
heun       sequential      602.7ms           11       30         8/8
dpmpp3m    adaptive        572.2ms            6       27         8/8
```

`agree` compares against sequential Euler on the **same weights**. With random
weights it reports how much the discretization changes the answer, not which
solver is better — that needs a trained model, and the command says so in its
own output rather than inviting the wrong reading.

---

## The library API

`InferenceEngine` bundles the model with its policy and handles batching,
ranking and profiling:

```rust
use diffusionblocks::infer::{InferenceConfig, InferenceEngine};
use diffusionblocks::solver::SolverKind;

let engine = InferenceEngine::new(model, InferenceConfig {
    solver: SolverKind::DpmPlusPlus3M,
    num_steps: Some(8),
    batch_size: 64,
    ..InferenceConfig::default()
});

let preds = engine.classify(pixels, &mut rng);
preds.labels;            // arg-max per input
preds.confidence;        // max class probability
preds.top_k(3);          // [(label, probability); 3] per input, descending
preds.accuracy(&truth);
preds.stats.ledger;      // per-block gate tally, merged across batches
```

Inputs larger than `batch_size` are split automatically. Splitting is not only
a memory convenience: the sampler draws its initial latent per batch, so
batching also bounds how much one unlucky noise draw can affect.

`classify_profiled` records per-batch timings into a `Profiler`, which reports
count, total, mean, median, p95 and share — percentiles rather than means
alone, because a step that is usually fast but occasionally stalls has a
healthy mean and a terrible p95, and only the second is visible to a user.

### Why there is no HTTP server

Serving is a deployment concern with its own dependency tree — HTTP stack,
batching queue, metrics endpoint — and this crate deliberately has neither
network nor async dependencies. Everything a server would need to call is here,
behind one type.

---

## See also

- [Inference Solvers](Inference-Solvers.md) — solver derivations
- [Multi-Block Denoising](Multi-Block-Denoising.md) — strategy details
- [Quality Gate](Quality-Gate.md) — gates and certificates
- [Precision & I/O](Precision-IO.md) — the precision emulation in detail
- [Next-Step Planning](Next-Step-Planning.md) — planned trajectories and lookahead decoding
- [Accuracy Improvements](Accuracy-Improvements.md) — guidance, normalization, ensembling, compute scaling
