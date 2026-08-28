# Mixture of Specialized Micro Experts

Boxes of small specialized experts — a `coding` box holding a Rust expert, a
Python expert and a secure-code expert; a `cybersecurity` box holding its own
specialists — with an index above every box so an inference engine can decide
what to load before loading anything.

> **In this repository.** `src/mosme.rs` (routing and the layer) and
> `src/expert_index.rs` (the manifest, with **no Burn dependency** so an external
> routing service can depend on it alone).
>
> | Piece | Type |
> |---|---|
> | Authoring document | `MosmeSpec` / `BoxSpec` / `ExpertSpec` |
> | Emitted manifest | `ExpertIndex` / `BoxEntry` / `ExpertEntry` |
> | Two-level router | `HierarchicalRouter`, `HierarchicalGates` |
> | Balance loss | `BalanceBreakdown` |
> | Layer | `MosmeFeedForward`, wired as `vit::FeedForward::Hierarchical` |
> | Trunk placement | `ViTDiTConfig::mosme`, `MosmeTrunkConfig` |
>
> CLI: `dblocks experts init | list | validate`, and
> `dblocks train --mosme-spec boxes.json --mosme-every 2`.

---

## What it is

Routing happens twice. A **box router** picks domains; a per-box **expert
router** picks specialists within them. The composed gate is

```text
g(box i, expert j) = g_box(i) · g_expert(j | i)
```

Both levels are renormalized after top-k, so their product is a distribution
over `(box, expert)` pairs by construction:

```text
Σ_{i,j} g(i,j) = Σ_i g_box(i) · (Σ_j g_expert(j|i)) = Σ_i g_box(i) · 1 = 1
```

That is not a nicety. The layer output is a gate-weighted sum of expert
outputs, so if the gates did not sum to one the result would be an arbitrarily
scaled mixture rather than a convex combination — and every downstream bound,
including the convex-hull certificate on `x0`, would stop holding. It is
certified as `mosme::composed_gates_partition_of_unity`.

Boxes may be **ragged**: three coding experts and two cybersecurity experts is
fine. Each box gets its own `Linear` router rather than one padded matrix, which
avoids padding masks entirely at identical parameter cost.

---

## Why there is no single `Expert` trait

A specialist can be three different things, and they are not variants of one
idea — they sit at different levels and mix in different spaces:

| Kind | Signature | Mixed in | Valid because |
|---|---|---|---|
| MLP sub-layer | `[T,h] → [T,h]` | feature space | the result feeds a residual add |
| LoRA adapter | delta on one `Linear` | weight-delta space | branches are linear in `x`, so `Σ gₑ·x AₑBₑ = x·(Σ gₑ ΔWₑ)` |
| Micro-model | `(pixels, zt, σ) → logits` | probability space | a convex combination of distributions is a distribution |

A common trait would have to accept the union of three input shapes and return
the union of three output shapes — a sum type pretending to be a product, with
an `expect()` at every call site. Burn also removes the usual escape hatch:
`Module` is not implemented for `Box<T>`, so `Box<dyn Trait>` is unavailable.

**What is genuinely shared is the router and the index, not the expert.** So
there is one router, one balance loss and one manifest, applied at three
*sites*. A box is homogeneous in kind, and `kind` is a property of the site
rather than of the individual expert — which costs nothing, since "coding:
Rust, Python, secure-code" is naturally homogeneous.

Currently shipped: the **MLP sub-layer** site. The adapter and model sites are
specified in [`TODO.md`](../TODO.md) items 18.13 and 18.14.

---

## The index

Names, domains and tags cannot live inside the model. Burn gives `String`,
`bool` and `usize` fields of a `#[derive(Module)]` struct an `EmptyRecord`, so
**metadata stored in a module does not survive a checkpoint round trip**. That
is not a limitation to route around but a reason the manifest should be separate
anyway: an engine deciding *which* expert to load must read the catalogue before
it has loaded anything.

Two documents, deliberately distinct:

- **`MosmeSpec`** is what a human writes: boxes, experts, labels, tags. Input.
- **`ExpertIndex`** is what training emits: the spec plus measured shapes,
  parameter counts, per-expert content hashes, the live enabled mask, and the id
  of the checkpoint the weights sit in. Output.

Keeping them apart matters because the index makes claims about weights that
only the trainer can substantiate. The enabled flags come from the **mask**, not
from the spec — the module is the authority on what is actually switched on.

```json
{
  "schema_version": 1,
  "model_id": "dblocks-384d2b2c25dabf51",
  "site": "mlp",
  "routing": { "top_box": 1, "top_expert": 1, "hidden_size": 128, ... },
  "boxes": [
    { "id": "coding", "label": "Code", "index": 0,
      "experts": [
        { "id": "coding/rust", "label": "Rust", "index": 0, "global_index": 0,
          "kind": "mlp", "hidden_size": 128, "intermediate_size": 512,
          "enabled": true, "num_parameters": 131712,
          "weights": { "param_path": "vit.layers.mlp.boxes.0.0", "sha256": "..." },
          "tags": ["rust"] } ] } ]
}
```

`ExpertKind` is serialized with an explicit `"kind"` tag rather than serde's
default externally-tagged form, so an engine in another language does not have
to know how Rust encodes enums. `test_wire_format_is_externally_readable` pins
that.

`validate()` enforces what every consumer is entitled to assume: unique ids,
dense indices matching vec positions, a consistent flattened order, one site
kind, and **at least one enabled expert per box**. That last one is not
cosmetic: masking every logit in a box to `-inf` makes the softmax denominator
zero and yields NaN, so an index permitting it would poison a run rather than
misroute it.

---

## Adding an expert without retraining the others

An expert is switched off by adding `-inf` to its router logit.
`exp(-inf - max)` is exactly `0.0`, so:

1. the disabled expert's gate is **exactly** zero, not merely small;
2. the softmax denominator gains `+0.0`, which is exact, so every other gate is
   bit-identical to what it was;
3. `y + 0.0 · expert(x)` is exactly `y`.

Therefore growing a model is a **bit-exact identity** until the new expert is
deliberately enabled:

```rust
let grown_spec = spec.extended_with("coding", ExpertSpec::new("coding/go", "Go"))?;
let model = model.grown(&grown_spec, &config, &device)?;   // outputs unchanged
// ...train only coding/go...
model.router_mut().set_enabled(box_idx, new_idx, true)?;
```

Certified as `mosme::hot_swap_is_an_exact_identity` and
`mosme::disabled_expert_gate_is_exactly_zero`, both at tolerance **0.0**. The
mask is a `Param<Tensor>` rather than a `Vec<bool>` precisely so it persists.

`grown()` splices three tensors per widened box — the router weight, its bias,
and the mask — assuming row-major `Linear` weights, which is what
`LinearConfig::new` produces throughout this crate. `weight_dims()` asserts the
rank so a layout change cannot corrupt it silently.

---

## Balance loss

Per level, the Switch auxiliary `L = N · Σₑ fₑ pₑ`. The box level is the plain
unweighted form. The expert level is computed **per box, weighted by that box's
gate**, then combined as a convex sum over box traffic shares
`mᵢ = mean_t g_box(t, i)`:

```text
L_expert = Σᵢ mᵢ · L_i        with  Σᵢ mᵢ = 1
```

The weighting is the point. Balancing experts uniformly over all tokens would
push balance pressure into boxes the router never selects — a gradient
decoupled from actual use. A box receiving no traffic must contribute nothing,
which `test_expert_balance_ignores_boxes_with_no_traffic` checks directly.

Because `Σᵢ mᵢ = 1` exactly, `L_expert` inherits the per-level bounds termwise,
which is what `mosme::hierarchical_balance_bounds` measures.

---

## Reduction to flat MoE

With exactly one box the hierarchical path must **be** the flat
[`MoE`](MoE-Routing.md) path — same output, same balance loss, bit-for-bit — or
it is a second subtly different implementation rather than a generalization.

Softmax over a single logit is exactly `1.0`, so the degenerate box gate is
exactly `1.0`, `mᵢ = 1.0`, and the box-level loss is exactly `1`. Multiplying
by `1.0f32` and adding `0.0f32` are exact operations, and both paths share the
same `scatter_gates` and `weighted_switch_loss` kernels and the same
accumulation order.

`mosme::single_box_reduces_to_flat_moe` therefore runs at tolerance **0.0**, and
`MosmeFeedForward::as_flat()` hands back a real `MoELayer` sharing this layer's
weights so the two are compared directly rather than by re-deriving what the
flat path would have done.

> **A Burn footgun this certificate caught.** `Param` initialization is
> **lazy**, and `Param::clone` on a parameter that has not been forced yet gives
> the clone *its own* deferred initializer — Burn's own comment says
> "initializing one does not affect the other". So two clones draw **different
> random weights while keeping the same `ParamId`**. `as_flat()` looked correct
> and produced a different model. `tensor_ext::force_initialization` fixes it,
> and `test_cloning_an_unforced_module_draws_different_weights` pins the
> behaviour so nobody rediscovers it the hard way.

---

## Training modes

Which parameters a run is allowed to move:

```rust
use diffusionblocks::mosme::{TrainingMode, TrainableSet};

TrainingMode::Joint                                          // everything
TrainingMode::Router                                         // routers only, experts frozen
TrainingMode::Experts                                        // experts only, routers frozen
TrainingMode::Specialist { expert_id: "coding/rust".into() } // one expert only
```

`layer.trainable(&mode, &spec)` returns a `TrainableSet` — an allowlist of
`ParamId`s — and `set.gradients(&mut grads, &model)` collects only those. Frozen
parameters simply produce no gradient entry, so the optimizer leaves them
untouched; there is no need to also flip `require_grad`, and doing so would
change what the backward pass computes.

`test_specialist_training_moves_only_its_own_expert` takes a real optimizer step
and asserts the target expert's weights change while a sibling in the same box
and an expert in another box stay **bit-identical**.

> **Rebuild the set after any checkpoint load.** Burn's `load_record` adopts the
> *record's* `ParamId` for every parameter, so a set captured before a
> `--resume` refers to ids that no longer exist. The failure is at least loud —
> the filtered gradients come back empty, the global gradient norm is `0.0`, and
> the existing gradient gate rejects every step — but build the set from the live
> model, after all loading, and the question does not arise. It *does* survive
> `optim.step`, which preserves ids.

The full workflow for adding a specialist to a trained model:

```rust
let grown_spec = spec.extended_with("coding", ExpertSpec::new("coding/go", "Go"))?;
let model = model.grown(&grown_spec, &config, &device)?;      // bit-exact identity
let trainable = model.trainable(
    &TrainingMode::Specialist { expert_id: "coding/go".into() }, &grown_spec)?;
// ...train...
model.router_mut().set_enabled(box_idx, new_idx, true)?;      // switch it on
```

---

## Usage

```bash
# Scaffold a spec.
dblocks experts init --out boxes.json \
    --box coding:rust,python,secure \
    --box cyber:netsec,malware

dblocks experts list --index boxes.json

# Train with expert boxes on every second trunk layer. The manifest is written
# next to the content-addressed checkpoint and keyed by its hash.
dblocks train --mosme-spec boxes.json --mosme-every 2 --steps 2000

dblocks experts validate --index checkpoints/dblocks-<hash>.index.json
dblocks verify --group mosme
```

`--mosme-spec` and `--moe-every` are mutually exclusive: they contend for the
same feed-forward slot, and the hierarchical path already generalizes the flat
one.

---

## What is not built yet

- **Selective freezing** (18.12) — specialist and router training modes.
  `GradientsParams::from_params` plus `burn::module::list_param_ids` are the
  primitives. The hazard to design around: `ParamId` does **not** survive
  `load_record`, so a trainable set captured before a `--resume` is stale.
- **Adapter site** (18.13) — experts as LoRA deltas over one shared frozen
  `Linear`, reusing `quantize::LoraAdapter`, whose zero-initialized `B` makes a
  fresh adapter an exact no-op for *any* routing condition.
- **Model site** (18.14) — whole micro-models mixed in probability space, with
  real sparse dispatch and per-expert checkpoints. This is the one where dense
  evaluation is genuinely expensive, and the one an index-reading engine most
  wants.

---

See also: [Quality Gate](Quality-Gate.md) · [MoE Routing](MoE-Routing.md) · [QLoRA](QLoRA.md) · [Language Modeling](Language-Modeling.md) · [Training Guide](Training-Guide.md) · [Home](Home.md)
