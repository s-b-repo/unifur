# Quality Gate

Quality control in DiffusionBlocks++ happens at three levels, each answering a
different question:

| Level | Question | Where |
|---|---|---|
| **Certificates** | Is the *implementation* mathematically correct? | `src/verify.rs`, `dblocks verify` |
| **Training checks** | Is *this run* healthy, phase by phase? | `src/quality.rs` + `src/train.rs` |
| **Sampling gates** | Is *this denoising step* usable? | `src/quality.rs` + `src/multi_block.rs` |

They are independent. A correct implementation can still produce a diverging
run, and a healthy run can still emit a bad individual step.

---

## 1. Certificates — is the implementation correct?

### What "mathematically proven" means here

Nothing in `verify.rs` is a proof in the formal sense. Rust is not a proof
assistant, and a floating-point implementation of a real-valued identity can at
best be **correct to a stated tolerance**. What the suite does instead is make
every load-bearing mathematical claim in the crate falsifiable and continuously
checked:

- each claim is written down as a theorem statement, in prose, next to the code
  that checks it;
- the check produces a **residual**: a single number that is zero exactly when
  the claim holds;
- the residual is compared against a tolerance chosen from the arithmetic, not
  from whatever the code currently happens to produce.

A claim that cannot be reduced to a residual is not listed.

### Running it

```bash
dblocks verify                 # all 73 certificates; exits non-zero on failure
dblocks verify --group solver  # one group
```

```text
group          certificate                               residual   tolerance  margin  status
----------------------------------------------------------------------------------------------------
schedule       boundary_endpoints                       1.038e-11    1.000e-8   0.00x  ok
               cdf_uniform_spacing                      1.110e-16   1.000e-12   0.00x  ok
               window_tiling                              0.000e0     0.000e0       -  ok
               block_routing_involution                   0.000e0     0.000e0       -  ok
...
73 / 73 certificates passed
```

The **margin** column is residual ÷ tolerance. A certificate drifting toward
its bound is visible there long before it fails.

### Coverage

| Group | Certified |
|---|---|
| `schedule` | Endpoints, CDF-uniform spacing, window tiling, and that the training-time sigma sampler and the inference-time block router are **mutual inverses** |
| `preconditioning` | The EDM identity `D(z) = x` for an exact denoiser; `c_in²(σ²+σ_d²) = 1`; `w(σ)·c_out² = 1` |
| `stats` | `erf + erfc = 1`; `norm_ppf ∘ norm_cdf = id`; scipy reference values |
| `solver` | Closed-form exactness; kernel moments vs quadrature; interpolation nodes; observed order of convergence; ancestral variance preservation |
| `precision` | Relative error ≤ 2⁻ᵖ; rounding idempotence |
| `quantize` | NF4 error ≤ half the widest level gap; exact zero; LoRA identity at init |
| `loopgraph` | ACT weights are a partition of unity; the budget is hard under adversarial signals |
| `moe` | Gates are a distribution; the balance loss lies in `[1, E]` on the diagonal; `max_e x_e ≤ logsumexp ≤ max_e x_e + ln E`; the z-loss is the squared distance of the log-sum-exp from zero; a per-row shift moves the z-loss but no routing probability |
| `mosme` | Two-level gates compose into a distribution; one box reduces *exactly* to flat MoE; adding a disabled expert is bit-identical |
| `lm` | Tokenization is lossless; causal attention leaks *exactly* nothing backwards; an untrained tied head starts at `ln(vocab)`; top-1 sampling is greedy decoding; the KV cache matches full recompute under every chunking, and cached decoding emits identical tokens |
| `planner` | The budget is never exceeded, even against an `expand` that ignores its allowance; depth 0 *is* the greedy policy; `beam(1)` reproduces greedy exactly; only a plan's first step is committed; planned sigmas fall monotonically without undershooting; lookahead defeats a myopic trap |
| `optim` | Accumulation over `k` steps equals one `k`x batch and fires on that cadence; the EMA is a convex combination that never extrapolates; the LR schedule is bounded and its ramp monotone; the uncertainty optimum is `ln L` with gradient scale 1 at every loss magnitude; importance sampling is unbiased with weights bounded by the smoothing floor |
| `accuracy` | Guidance at scale 1 is bitwise identity and affine in the estimates elsewhere; every logit normalization preserves the arg-max; ensembles emit distributions and N copies of one member are that member; the scaling frontier is exactly the undominated set |
| `model` | Softmax partition; `x0` inside the label-embedding convex hull; unit-norm embeddings; DiT zero-init |
| `autodiff` | Finite-difference gradient check on the distillation loss; Gibbs' inequality |

### Why this is worth the code

Two of the bugs fixed in this repository were found *by* the suite rather than
by tests written to look for them:

- **`erf` / `erfc` crossover.** The identity `erf(x) + erfc(x) = 1` held only
  to 2.7e-14, because `erf` used its alternating Maclaurin series up to
  |x| = 3, where the largest term is of order `e^{x²}` and roughly four digits
  are lost to cancellation. Both functions now cross over at |x| = 2 and the
  residual is 1.1e-16. Fixed at the source, not by widening the tolerance.
- **Block-index convention.** `block_routing_involution` is the certificate
  that would have caught the most damaging bug in the codebase: sigmas drawn
  for block *b* at training time being routed to a different block at inference
  time. It now asserts the round trip for every block count.

### Adding a certificate

New mathematical claims belong here rather than in an ad-hoc test:

```rust
out.push(cert(
    "schedule",
    "window_tiling",
    "Block windows tile [sigma_min, sigma_max]: block b's lower edge is block b+1's upper edge.",
    tiling_err,   // residual: zero exactly when the claim holds
    0.0,          // tolerance: exact arithmetic, so exact equality
));
```

Pick the tolerance from the arithmetic. `0.0` for exact integer or comparison
logic, `1e-12`–`1e-15` for f64 identities, `1e-6` for anything that passes
through f32 tensors, and something explicitly justified for iterative methods.

---

## 2. Training checks — is this run healthy?

Block-wise training fails **quietly**. A step trains one block on one noise
window; if that block receives no gradient, or the optimizer pushes a parameter
to infinity, the loss of the *other* blocks keeps the average looking healthy
for a long time. So verification is applied at every phase of a step, and each
phase reports separately — "loss was NaN" and "a parameter went non-finite
after the update" call for different responses.

### The five phases

| Phase | Checked | On failure |
|---|---|---|
| `preflight` | `verify::preflight()` — schedule, preconditioning, stats | **Abort before step 0** |
| `loss` | The loss is finite | Reject the step |
| `gradients` | Global gradient norm within `[min, max]` | Reject the step |
| `parameters` | Every parameter still finite after the optimizer step | **Abort the run** |
| `periodic` | `verify::model_health()` against the *live* weights | Log and record |

Two of those are fatal rather than skippable, on purpose:

- **Preflight** verifies what the run silently depends on for its whole
  duration. If the block-index convention or the EDM identity is broken, every
  subsequent step is wasted; the check costs milliseconds against hours.
- **A non-finite parameter** poisons every later step. There is nothing left to
  train, so the run stops where it happened instead of producing a flat loss
  curve nobody can explain afterwards.

The other three reject the step and continue, because intermittent rejection is
normal in block-wise training — a badly conditioned sigma draw is not a broken
run. A **streak** is, so `max_consecutive_rejections` (default 50) aborts a run
that is stuck.

### Why skip rather than clip

The gradient gate skips a bad step instead of clipping it. In a block-wise
loop each step trains a different block: a clipped bad step still writes to
that block's parameters, while a skipped one leaves them for a healthier draw
of the same block.

### Configuration

```bash
dblocks train --steps 2000                    # all checks on (default)
dblocks train --verify-every 100              # + re-verify live weights every 100 steps
dblocks train --no-preflight                  # skip the pre-run certificates
dblocks train --no-checks                     # everything off
```

```rust
use diffusionblocks::quality::{TrainingChecks, GradNormGate};

TrainingChecks::default()      // preflight, loss, gradients, parameters
TrainingChecks::thorough(100)  // + periodic model health every 100 steps
TrainingChecks::none()         // reproduce a run from before the checks existed

TrainingChecks {
    grad_gate: Some(GradNormGate { min_norm: 1e-8, max_norm: 1e4 }),
    max_consecutive_rejections: 50,
    ..TrainingChecks::default()
}
```

### The per-block health report

Every run ends with a table, because the aggregate loss curve is exactly the
thing that hides a per-block failure:

```text
per-block quality:
block     steps   rejected    mean loss     mean |g|      max |g|   reject%
--------------------------------------------------------------------------
0            67          0      14.2251    1.443e+01    2.331e+01      0.0%
1            65          0      29.6249    1.611e+01    4.605e+01      0.0%
2            68          2    3418.1597    2.008e+03    9.912e+03      2.9%
```

Block 2 owns the lowest-sigma window, where the EDM weight
`w(σ) = (σ²+σ_d²)/(σσ_d)²` grows very large — so a much higher mean loss there
is expected, not a bug. What the table is for is spotting what *is* a bug:

- **`max |g| == 0`** for a block that ran: it is dead. Reported explicitly as
  `dead_blocks()`, with a warning naming the likely cause (a `num_blocks` /
  `num_hidden_layers` mismatch, or sigma windows that never select it).
- **One block with a much higher `reject%`** than the others: that block's
  window is badly conditioned, not the run as a whole.

`TrainingSummary::health` exposes the same data programmatically, and the
JSONL log records `accepted`, `grad_norm` and `skipped` per logged step.

Retained failure details are capped at 64 entries — a run that rejects every
step must not exhaust memory recording that fact — while the *counts* stay
exact.

---

## 3. Sampling gates — is this denoising step usable?

At inference a gate inspects the transition `x0_prev → x0_new` and marks
samples that fail. Failing samples keep their previous latent rather than
taking the bad update.

| Check | Rejects when |
|---|---|
| Cosine | The embedding estimate's direction changed too abruptly |
| MSE | The step moved too far |
| Confidence | The max class probability is too low |

### Per-layer gates

A single batch-level threshold treats every block identically, but blocks do
not face identical problems. A high-sigma block legitimately makes large,
direction-changing updates; a low-sigma block that does so is diverging.
`LayerGates` therefore holds one default plus per-block overrides:

```rust
use diffusionblocks::quality::{LayerGates, QualityGateConfig};

// One gate everywhere.
LayerGates::uniform(QualityGateConfig::strict());

// Monotonically stricter as sigma falls: block 0 is allowed large moves,
// the final block should only be making small corrections.
LayerGates::tightening(
    num_blocks,
    QualityGateConfig::lenient(),
    QualityGateConfig::strict(),
);
```

`min_cosine` interpolates linearly and `max_mse` geometrically, so both are
monotone in the block index **by construction** — asserted by
`test_tightening_gates_are_monotone` for every block count rather than checked
at the endpoints.

```bash
dblocks sample --gate lenient      # reject only degenerate transitions
dblocks sample --gate strict       # conservative thresholds everywhere
dblocks sample --gate tightening   # per-block, stricter at low sigma
```

### The gate ledger

`SamplingStats::ledger` tallies decisions per block, so a gate that is firing
constantly on one block is visible:

```text
model calls: 6 | layers executed: 27 | mean span: 5.40 | gated samples: 3
  block 2: 12.5% of updates gated
```

Ledgers from separate batches are combined with `GateLedger::merge`, which
carries **both** the evaluated and the rejected counts. Adding only the
rejections would drive every rate that has any rejection to 100% — a bug this
repository shipped briefly and now has a regression test for.

---

## Putting it together

The development loop that exercises all three levels:

```bash
cargo test --all              # certificates + unit + integration
cargo clippy --all-targets
dblocks verify                # the gate, standalone
dblocks train --verify-every 100 --steps 2000
```

### One thing a certificate cannot catch

A test that pins a *value* rather than a *property* can pass for years and then
fail when something unrelated runs beside it. That happened here:
`integration_mosme_trunk_trains_end_to_end` asserted `balance_loss >= 2.0`,
treating the Switch bound `L >= 1` as unconditional. It holds only on the
diagonal `f == p`, which hard top-k routing does not give — so the assertion was
really pinning one draw from a **global** backend RNG, and adding a concurrent
test that builds a model broke it.

The fix was not to reseed harder. It was to assert the structural upper bound,
which is a theorem for arbitrary traffic. The lesson generalizes to every
certificate in the suite: *state the theorem, measure the residual*. A number
that happens to be true is not a certificate.

## See also

- [Mathematical Foundation](Mathematical-Foundation.md) — the theory the
  certificates check
- [Training Guide](Training-Guide.md) — running and interpreting a training run
- [Multi-Block Denoising](Multi-Block-Denoising.md) — the sampling strategies
  the gates wrap
