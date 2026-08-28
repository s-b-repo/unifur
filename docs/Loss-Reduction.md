# Loss Reduction & Convergence

Everything that makes the optimizer's job easier without changing the model:
learning-rate schedules, gradient accumulation and clipping, weight averaging,
per-block normalization, learned per-sigma uncertainty weighting, and
importance sampling over noise levels.

> **In this repository.** `schedule.rs` (`LrSchedule`, `GradientAccumulator`,
> `clip_gradients`, `Ema`, `LossScales`) and `reweight.rs`
> (`UncertaintyWeighting`, `LogVarianceHead`, `SigmaImportanceSampler`), wired
> through `TrainConfig` and `DblockClassifier::training_step_on`. CLI:
> `--lr-schedule --accumulate --clip-norm --ema-decay --normalize-block-loss
> --uncertainty --importance-bins`. Certificates: the `optim` group.

---

## The measurement this phase exists for

From this repository's own runs:

| Block | Mean loss |
|---|---|
| 0 | 13.4 |
| 1 | ~90 |
| 2 | **1909.8** |

A ~140× spread. It is not a bug. The EDM weight

```text
w(sigma) = (sigma^2 + sigma_d^2) / (sigma * sigma_d)^2
```

diverges as `sigma → 0`, and block 2 owns the low-noise end. But a **shared
trunk** trained on a sum of terms differing by two orders of magnitude is, in
effect, trained almost entirely on the largest one. The other blocks contribute
gradient noise.

Two independent remedies address it, and they compose.

---

## Uncertainty weighting

Replace each sample's loss `L` with

```text
exp(-l) * L + l,        l = logvar(sigma)
```

where `logvar` is a small MLP over the sinusoidal embedding of **log** sigma —
log, because sigma spans four decades and a basis over sigma itself would
resolve the noisy end and collapse the entire low-noise range, which is exactly
the range whose scale needs explaining.

```bash
dblocks train --uncertainty 1.0
```

### Why the `+ l` term is the whole design

Without it the head drives `l → +∞` and every loss vanishes. With it, the
objective in `l` alone is minimized at

```text
l* = ln L,      value = 1 + ln L
```

so the head is *forced* to report the true loss scale. It cannot buy a smaller
number by claiming more uncertainty. `optim/uncertainty_optimum_is_the_log_loss`
checks stationarity, minimality, and the value at the optimum.

### Why this fixes the imbalance rather than papering over it

At `l = l*`, the gradient with respect to the network is

```text
exp(-l*) · ∂L/∂θ  =  (1/L) · ∂L/∂θ  =  ∂(ln L)/∂θ
```

which is **invariant under `L → c·L` for any per-sigma constant `c`**. That is
the precise statement: the imbalance is a property of the *scale* of `w(sigma)`,
and the gradient no longer sees the scale at all. No choice of loss-weighting
convention can reintroduce it.

`optim/uncertainty_gradient_is_scale_free` checks it directly — the effective
gradient scale is 1 for a loss of 13.4, for 1909.8, and for either multiplied by
anything from `1e-6` to `1e6`.

### Two safety properties

- **Zero-initialized head.** `exp(-0) = 1`, so the reweighted loss starts
  identical to the unweighted one and enabling the feature cannot make step 0
  worse. The same zero-init argument the DiT blocks and the LoRA adapters use.
- **Clamped log-variance.** `exp(-l)` is unbounded below; an unclamped head that
  overshoots early can multiply a loss by `e^20` and destroy the run in one
  step. The bound is on the log, so it is symmetric in scale.

### Where it lives

**Not in the model.** Putting the head inside `vit.rs` would add a parameter to
the module record and every existing checkpoint would stop loading. It is owned
by the training loop, with its own optimizer state, and shares the trunk's
backward pass via `GradientsParams::from_module` — which borrows the gradients
rather than consuming them, so two modules can draw from one `backward()`.

Its gradients are deliberately kept out of the model's global norm: it is one
scalar per noise level against a whole trunk, and folding it in would let it
move the clipping threshold and the gradient gate for reasons that have nothing
to do with the trunk.

---

## Importance sampling over sigma

Training draws sigma uniformly in lognormal-CDF space across a block's window.
That spends the same number of samples where the loss is flat as where it
varies, and the variance of the gradient estimate is dominated by the latter.

```bash
dblocks train --importance-bins 16
```

The sampler keeps a running loss estimate per equal-probability CDF bin,
proposes bins in proportion to it, and returns an **importance weight** `p/q`
with each sample:

```text
E_q[(p/q)·f] = Σ_b q_b · (p_b/q_b) · f_b = Σ_b p_b · f_b = E_p[f]
```

Bins are equal in *probability* rather than in sigma, which is what makes
`p_b = 1/bins` exactly and the weight a ratio of two simple numbers instead of
an integral. `optim/importance_sampling_is_unbiased` checks the identity
exactly, not by convergence.

### The smoothing floor is not optional

The proposal is mixed with the uniform one. Without that floor, a bin whose loss
estimate happens to start near zero is never sampled again, its estimate never
updates, and — worse — if the true loss there later grows, `p/q` for that bin is
enormous. **A starved bin is not merely unvisited; it is a variance bomb.**

`optim/importance_weights_are_bounded_by_the_smoothing_floor` drives the sampler
adversarially (all the mass in one bin) and checks the worst weight stays under
`1/smoothing`. It also checks that a **cold** sampler is *exactly* plain
sampling, with every weight equal to 1 — so the feature can be switched on
before it has learned anything.

---

## The rest of the phase

| Flag | What it does | Certified |
|---|---|---|
| `--lr-schedule warmup-cosine` | Warmup then cosine decay | Never exceeds its peak; ramp non-decreasing, decay non-increasing |
| `--accumulate k` | `k` micro-batches per optimizer step, **averaged** not summed | Equals one `k`× batch exactly, and fires on exactly that cadence |
| `--clip-norm` | Rescales a merely-large step | — (complements the gradient *gate*, which discards a pathological one) |
| `--ema-decay` | Bias-corrected weight averaging | Coefficients sum to 1; the blend never leaves the interval its inputs span |
| `--normalize-block-loss` | Equalizes blocks on the **geometric** mean | — (geometric because the imbalance spans orders of magnitude and an arithmetic mean would be pinned to the largest block) |

Averaging rather than summing in the accumulator is what keeps the learning rate
meaningful: if it summed, the effective step size would scale with the
accumulation count and every hyperparameter would silently change with it.

---

## What it actually does, measured

400 steps on the synthetic dataset, 3 blocks, same seed. Both runs visited the
blocks the same number of times (138 / 145 / 117), so the comparison is like for
like:

| Block | Baseline loss | mean \|g\| | rejected | `--uncertainty 1.0` loss | mean \|g\| | rejected |
|---|---|---|---|---|---|---|
| 0 | 25.7 | 1.32e2 | 0 | 7.7 | 3.66e0 | 0 |
| 1 | 38.2 | 3.33e2 | 0 | 10.6 | 4.54e0 | 0 |
| 2 | 1585.7 | 2.39e3 | **5 (4.3%)** | 397.1 | 1.51e2 | **0** |

Read it carefully, because two of the three effects are real and one is not yet.

**Real.** Absolute gradient magnitudes fell by roughly 16x across every block,
and block 2's five gradient-gate rejections went to zero — the gate was firing
on steps that were merely large, not pathological, and after reweighting they
are no longer large.

**Real.** Blocks 0 and 1, whose raw losses are within 1.5x of each other,
converged from a 2.5x gradient ratio to **1.24x**. That is the mechanism
working: comparable losses become comparable gradients.

**Not yet.** Block 2 did *not* close the gap — its gradient ratio against block 0
went from 18x to 41x. 400 CPU steps is not enough for the head to track a loss
40x larger than its neighbours while that loss is itself still moving. The
theory says the ratio goes to 1 at the head's optimum; this run does not reach
the optimum, and reporting it as if it did would be dishonest.

## What is not claimed

That either reweighting improves final **accuracy** on this crate's datasets.
That needs GPU runs long enough for convergence to mean something, and it is
listed as blocked in `TODO.md`. What *is* claimed and checked: both are exact
identities when off, the uncertainty optimum is scale-free, and the importance
estimator is unbiased.

---

See also: [Training Guide](Training-Guide.md) · [Quality Gate](Quality-Gate.md) ·
[Mathematical Foundation](Mathematical-Foundation.md) ·
[Configuration](Configuration.md) · [Accuracy Improvements](Accuracy-Improvements.md)
