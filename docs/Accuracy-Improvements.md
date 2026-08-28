# Accuracy Improvements

Four techniques that raise accuracy *after* training, without touching the
weights. Each has an exact identity setting, and each identity is certified —
so a mis-set knob degrades to the un-improved baseline rather than to garbage.

> **In this repository.** `accuracy.rs` (`Guidance`, `LogitNorm`, `Ensemble`,
> `ScalingCurve`, `ScalingPoint`, `accuracy`), wired through
> `MultiBlockConfig::{guidance, logit_norm}`, `InferenceConfig`, and
> `DblockClassifier::{x0_estimate_unconditional, sample_ensemble}`. CLI:
> `dblocks sample --guidance --guidance-rescale --logit-norm --ensemble`,
> `dblocks bench` for the scaling curve. Certificates: the `accuracy` group.

---

## Guidance

```text
x0_guided = x0_uncond + scale * (x0_cond - x0_uncond)
```

`scale = 1` is the conditional estimate, `0` the unconditional one, and `> 1`
sharpens conditioning at the cost of diversity and — past some point — fidelity.

```bash
dblocks sample --guidance 3.0 --guidance-rescale 0.7
```

"Unconditional" here means a **zero image**, not a learned null embedding: this
model has no null token to fall back on, and zeros are the input the patch
embedding maps to its own bias — the closest thing to "no evidence" available
without retraining. A learned null embedding would be stronger and is the
natural upgrade if guidance proves worth training for.

Two details that are easy to get wrong:

**Scale 1 returns the conditional estimate untouched**, rather than computing
`u + 1.0 * (c - u)`. Those are equal in exact arithmetic and not in floating
point: the round trip through `c - u` and back loses low bits whenever `|u|` is
much larger than `|c - u|`. Here the result feeds an ODE step, where the lost
bits compound over the trajectory. `accuracy/guidance_identity_is_exact` checks
it **bitwise**, at a tolerance of zero.

**Large scales inflate the estimate**, and again the inflation compounds through
the solver. `--guidance-rescale` interpolates the guided estimate back toward
the conditional estimate's per-sample standard deviation; `0.0` disables it,
`1.0` restores the spread exactly.

Guidance costs a second model call per window, and `SamplingStats::model_calls`
reports it rather than hiding it — a guided run is genuinely twice the compute.

---

## Logit normalization

```bash
dblocks sample --logit-norm standardize --logit-tau 1.0
```

Every variant — temperature, L2, standardization — is a strictly increasing
per-row affine map, so **the arg-max never moves**. Top-1 accuracy is identical
before and after, and `accuracy/logit_normalization_preserves_argmax` holds that
to a residual of exactly zero.

Which raises the obvious question: what is it for?

**The gates.** `Strategy::Adaptive` widens its span when confidence is low, and
`quality.rs` rejects updates on confidence thresholds. Both read a number whose
scale is an artifact of how large the trained logits happen to be — a threshold
tuned on one checkpoint means something else on the next. Normalizing first
makes those thresholds portable. It is an accuracy improvement by way of the
gates, not by way of the classifier.

```rust
// Two checkpoints differing only in logit scale.
let small = logits(&[1.0, 2.0, 0.5]);
let large = logits(&[10.0, 20.0, 5.0]);
// Raw confidences disagree by more than 0.3; standardized they agree to 1e-5.
```

Label smoothing, which the roadmap listed alongside this, is a *training*
change and is deliberately not here.

---

## Ensembling

```bash
dblocks sample --ensemble probability
```

The crate already has more than one way to reach an answer — four solvers, four
span strategies, a planner — and they make different errors. Averaging them is
the cheapest accuracy gain available, at a cost exactly linear in the member
count.

| Rule | Combines | Character |
|---|---|---|
| `probability` | mean of the member distributions | weights a member by its confidence; the default |
| `logit` | softmax of the mean logits | sharper, but dominated by the member with the largest logit scale — pair it with `--logit-norm` |
| `vote` | mean of the one-hot predictions | plurality; resists one confidently wrong member |

`combine` returns **probabilities**, not logits: a vote has no logit scale, and
a mean of distributions is not the softmax of anything.

An empty ensemble panics rather than returning a uniform distribution — a
uniform answer would hide a caller's bug behind plausible-looking numbers. An
ensemble of N copies of one member *is* that member
(`accuracy/identical_members_are_identity`), so a pipeline can keep the ensemble
permanently in place with the member count as the only knob.

---

## Test-time compute scaling

"Spend more compute at inference and get more accuracy" is a claim, not a law.
It holds up to a point and then flattens or reverses, and where that happens
depends on the model, the schedule and the solver.

```bash
dblocks bench
```

```text
Test-time compute scaling (* marks the Pareto frontier):
  configuration                 calls   layers   top-1   frontier
  sequential/steps=2                2        2   0.125   *
  sequential/steps=4                4        4   0.188   *
  planned/depth=0                  13       13   0.125
  planned/depth=2                  57       57   0.250   *
  planned/depth=2: +0.00234 top-1 per extra layer
```

The **Pareto frontier** is the whole of the useful answer: a dominated point —
one that costs more and delivers no more — is a configuration nobody should ever
choose, and reporting it alongside the rest makes a scaling study look like a
menu of trade-offs when some entries are simply worse.
`accuracy/pareto_frontier_is_exactly_the_undominated_set` checks the frontier by
brute force against the definition.

**Marginal accuracy per extra layer** is the number a budget is actually set
from. Where it falls to near zero is where extra test-time compute stops paying.

Layers, not calls, is the cost axis: strategies use different span widths, so a
wide span costs more per call and counting calls alone would flatter it.

---

## EMA evaluation weights

```bash
dblocks train --ema-decay 0.999
```

`train` returns the EMA shadow when the flag is set, so evaluation uses averaged
weights rather than whichever point the last step happened to land on. The
averaging is bias-corrected, so the early shadow is usable rather than stuck on
its initialization — and it pairs parameters by traversal order rather than by
`ParamId`, because `load_record` reassigns ids and a resumed run would otherwise
freeze the shadow silently.

---

## The one omission

**Self-conditioning** is deliberately absent. Feeding the previous x0 estimate
back as conditioning only helps a model *trained* with it; bolted on at sampling
time it degrades results. Doing it properly needs a new projection inside
`vit.rs`, which adds a parameter to the module record and therefore cannot load
an existing checkpoint. That cost is written down in `TODO.md` rather than paid
by surprise.

---

See also: [Inference Guide](Inference-Guide.md) ·
[Next-Step Planning](Next-Step-Planning.md) ·
[Multi-Block Denoising](Multi-Block-Denoising.md) ·
[Quality Gate](Quality-Gate.md) · [Configuration](Configuration.md)
