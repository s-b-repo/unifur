# Direction Ablation

A behaviour lives, in part, along a direction in the residual stream. Find
it as the difference of mean activations between a target and a baseline
prompt set, then either **remove** it from the weights ("abliteration"),
**project it out** of the activations at inference, or **penalize** it during
training — the last being negative supervision for any model whose hidden
states can be read.

> **In this repository.** `ablation.rs` (`Direction`, `extract`, `best`,
> `orthogonalize`, `residual_projection`, `project_out`, `projection_penalty`,
> `RESIDUAL_WRITERS`), `lm.rs` (`LanguageModel::hidden_states`,
> `forward_ablated`, the `direction` term of `next_token_step_full`),
> `dblock.rs` (`DblockClassifier::training_step_negative`, CLS hidden states).
> CLI: `dblocks lm direction`, `dblocks lm ablate`, `dblocks lm direction-score`,
> `dblocks lm train --direction --direction-weight`, `dblocks train
> --synthetic-negatives`. Certificates: the `ablation` group. After mlabonne,
> *Uncensor any LLM with abliteration* (Hugging Face blog).

---

## Finding a direction

```bash
dblocks lm direction --checkpoint lm.mpk --target refusals.txt --baseline plain.txt \
    --layer auto --out dir.json
```

Both files hold one prompt per line. For every layer the model's output at
the last position is averaged over each set, and the direction is
`normalize(mean_target − mean_baseline)`. `--layer auto` keeps the layer with
the largest **separation** (the gap between the two sets' mean projections
in units of their pooled spread); the file records every candidate's
separation so the choice is inspectable.

## Three ways to use it

| Mechanism | What changes | When |
|---|---|---|
| **Ablate** (`dblocks lm ablate`) | Every weight that *writes* the residual stream — the attention output projection, the MLP output projection (each expert's, in a sparse layer), the token and position embeddings — is orthogonalized: each row `w` becomes `w − (w·d) d`. Readers of the stream (queries, keys, values, `fc_in`, norms, routers) are untouched. A new checkpoint is written whose state records the parent and the direction | permanently, once |
| **Project out** (`forward_ablated`) | `h ← h − (h·d) d` after every layer | at inference, without touching the weights |
| **Penalize** (`--direction --direction-weight λ`) | `λ · mean((h_L · d)²)` at the direction's layer joins the loss | during training, alongside the unlikelihood negatives of [Negative Supervision](Negative-Supervision.md) and the negative teacher of [Multi-Source Training](Multi-Source-Training.md) |

`dblocks lm direction-score --checkpoint --direction --prompts` reports the
mean projection of a prompt set before and after: the number the blog reads
as a refusal count, here as a continuous score.

## Negative supervision for the image trunk

The same idea, without a direction: `DblockClassifier::training_step_negative`
takes per-sample **negative labels** ("not class k") and charges
`−log(1 − p_k)` — the bounded term of Phase 24 on class probabilities —
mixable with ordinary positives in one batch. `--synthetic-negatives p` marks
a fraction of the synthetic data negative to exercise it; a real dataset
supplies its own.

## Gated residual adds

This trunk's residual adds are gated by adaLN: `h += g ⊙ branch(h)`. A
writer whose rows are orthogonal to `d` can still put `d` into the stream
once each coordinate is scaled by `g`, so `orthogonalize` projects a writer
inside layer `l` off `normalize(g_l ⊙ d)` instead, where `g_l` is the gate
that layer applies under the language conditioning
(`LanguageModel::layer_gates`). Then `(g ⊙ W x)·d = (W x)·(g ⊙ d) = 0`. A
plain-residual model has `g = 1` and the two coincide; a zero gate silences
the branch and its writer is left alone. The embeddings are not gated and
are projected off `d` itself.

## Heretic mode

[Heretic](https://github.com/p-e-w/heretic) (p-e-w) turns abliteration into
a search: instead of one direction removed everywhere at full strength, each
component (attention `dense`, MLP `fc_out`) gets a **trapezoid kernel** over
layers — `max_weight` at `max_weight_position`, falling linearly to
`min_weight` over `min_weight_distance` layers, `min_weight` beyond — and a
**fractional direction index** interpolating between neighbouring layers'
directions. Weighted orthogonalization `W ← W − α_l (W d) dᵀ` is applied
per layer, and the parameters are optimized against two objectives at once:
the **refusal rate** on the target prompts (short greedy generations scored
by a phrase-list `RefusalDetector`, extensible) and the **KL divergence** of
the first output token from the original model on the baseline prompts,
scalarized as `refusals + kl_weight · kl` with the Pareto front kept. The
sampler is a seeded TPE-like sequential one: uniform start-up trials, then
Gaussians around the best quartile with shrinking width. Every trial is
recorded.

`heretic.rs` implements this over the same `Direction`s and the same
gate-aware orthogonalizer. `dblocks lm heretic --checkpoint --target
--baseline --trials --kl-weight --out [--json]` decensors a saved model and
writes the best trial as a new checkpoint whose state records the parent
and the parameters; `dblocks lm train --heretic-target --heretic-baseline
...` runs the same search after training, so the final checkpoint of the
run is the decensored one — training when the filters are not wanted. The
[policy gate](Cyber-Policy.md) still applies to whatever the weights do.

## What is certified

| Certificate (`ablation` group) | Claim |
|---|---|
| `extracted_direction_separates_its_sets` | the best direction is a unit vector along which the target prompts lie above the baseline prompts |
| `orthogonalized_writers_have_no_component_along_the_direction` | after ablation of the real model, `max |W d'|` over every residual writer is zero up to the arithmetic of two `h`-term dot products |
| `orthogonalizing_twice_is_the_identity` | on a coordinate axis the projection is exact, so a second pass changes no bit |
| `inference_ablation_removes_the_direction_from_every_layer` | with `d` projected out after every layer, no layer's output has a component along it |
| `zero_direction_weight_is_the_plain_loss` | `λ = 0` reproduces the plain loss bit for bit and reports no projection |
| `penalized_step_ends_below_plain_step` | from one initialization and batch, a penalized step leaves the layer projecting less onto `d` than a plain step |
| `zero_negative_charge_costs_nothing` | negative labels at charge 0 contribute exactly zero cross-entropy while still being counted |
| `negative_charge_is_bounded` | the charge is clamped at `−ln ε` per sample |
| `negative_step_ends_below_rewarded_step` | from one initialization and batch, charging the labels leaves them less probable on the clean latent than rewarding them |
| `heretic_kernel_is_a_trapezoid` | `max_weight` at its position, `min_weight` beyond its distance, linear between, within `[min, max]` everywhere |
| `heretic_identity_parameters_touch_nothing` | the identity parameters ablate nothing, change no bit, and a model's KL from itself is zero |
| `integer_direction_index_is_that_layers_direction` | an integer index interpolates to exactly that layer's direction |
| `heretic_best_is_never_worse_than_any_trial` | the reported best has the lowest score of every trial run |
| `refusal_detector_matches_its_phrases` | a refusal phrase is flagged, a compliant answer is not |

## What is not claimed

That removing a direction removes a behaviour, or that Heretic's search
finds the trade-off it finds on large models. The evidence for both is on
large instruction-tuned models; on this crate's hardware the mechanism is
what is shipped, and its effect on a real model is UNKNOWN in
[Claims](Claims.md). The [policy gate](Cyber-Policy.md) is the complementary
control that holds regardless of the weights.

---

See also: [Negative Supervision](Negative-Supervision.md) · [Cyber Policy](Cyber-Policy.md) ·
[Multi-Source Training](Multi-Source-Training.md) · [Configuration](Configuration.md) · [Home](Home.md)
