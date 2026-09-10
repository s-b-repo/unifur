# Hybrid Attention, Positions and Routing State

Per-layer attention modes for the language trunk, rotary positions that
remove the context bound, the decode-time state each mode keeps, a
per-token routing state carried through the layers, and the locality
diagnostics a multi-axis router needs (roadmap Phase 25).

> **In this repository.** `hybrid.rs` (`AttentionMode`, `AttentionSchedule`,
> `PositionKind`, `LayerState`, `LinearState`, `attention_mask`,
> `keep_top_k`, `feature_map`, `linear_attention`, `apply_rotary`),
> `routing.rs` (`RoutingState`, `RouterKind`, `token_stability`,
> `layer_agreement`), `cost.rs` (`LayerCost`, `TrunkCost`, `attention_cost`,
> `moe_cost`, `ComputeLedger`). Wiring: `vit.rs` (`ViTDiTConfig::attention |
> rotary | routing_state`, `LayerCarry`), `lm.rs` (`LmConfig::attention |
> positions | routing_state`, `LmConfig::cost`, `KvCache::resident_floats`),
> `moe.rs` / `mosme.rs` (`state_size`, `forward_with_state`,
> `RoutingStats::stability | agreement | kind`). CLI: `dblocks lm train |
> generate | merge --attention --positions --routing-state --window
> --retrieval-k`, `dblocks lm bench --axis attention | positions | routing`.
> Certificates: the `hybrid` group (9).

---

## Why a schedule, not a switch

GLM-5.3-Flash and Nemotron 3 spend most layers on cheap recurrent or linear
attention and a few on precise attention. Whether that trade pays, and at
what ratio, is a measurement -- so the trunk takes **one mode per layer**
and the CLI accepts every form the comparison needs:

| `--attention` | Meaning |
|---|---|
| `dense` (default, empty) | every layer dense: the Phase 19 trunk, bit for bit |
| `linear`, `learned`, `sliding64`, `retrieval32` | every layer that mode |
| `3:1` | three linear layers per dense one, the dense one last in each group |
| `2:1@sliding64` | two sliding layers per dense one |
| `LLLD`, `SSRD` | a letter pattern repeated to the depth (`D` dense, `L` linear, `S` sliding, `R` retrieval, `M` learned) |
| `linear,dense,learned,dense` | one mode per layer |

`dblocks lm bench --axis attention` trains the tiny model under `dense`,
`3:1`, `sliding8`, `retrieval4`, `linear` and `learned` and records the
loss per seed, the milliseconds per step and the **counted** cost per token.

## The modes

| Mode | Attends to | Cost per token | Keeps at decode time |
|---|---|---|---|
| Dense | every earlier position | `O(n)` keys read | all keys and values |
| Sliding `w` | the last `w` positions | `O(w)` | the last `w - 1` keys and values |
| Retrieval `k` | the `k` highest-scoring earlier positions | scores `O(n)`, values `O(k)` | all keys and values |
| Linear | every earlier position through `phi(q)^T S` | `O(d^2)` per head, independent of `n` | `S [d, d]` and `z [d]` per head |
| Learned | a softmax mixture of dense and linear, one `[2]` logit per layer | dense + linear | both |

Every mode is causal; the image trunk keeps dense bidirectional attention
and refuses any other mode at construction.

**Retrieval** computes every score and reads only the chosen values. On a
CPU backend that saves nothing measurable -- the point of the mode is the
value bandwidth it does not spend, and that is what `cost.rs` counts
(`keys_read`). The claim that retrieval attention *is* as good as dense
attention at `k` keys is a GPU question and is listed as such in
`docs/Claims.md`.

**Linear** attention uses the feature map `elu(x) + 1`, so the normalizer
never vanishes. Its recurrent form (the state `S = sum phi(k) v^T`,
`z = sum phi(k)`) and its masked matrix form are the same numbers up to
summation order, which is certificate `hybrid/linear_state_matches_masked_form`.
There is no attention dropout on a linear layer: there are no probabilities
to drop.

**Learned** mixes dense and linear attention with a per-layer softmax over a
zero-initialized `[2]` logit, so an untrained layer is an even mixture and
training decides the schedule. It is the only mode that adds a parameter.

## Positions

| `--positions` | Sequence length | How |
|---|---|---|
| `learned` (default) | bounded by `context` | the Phase 19 `[context, hidden]` table |
| `rotary` | unbounded | queries and keys rotated by their absolute position; a score depends only on the distance between them |
| `none` | unbounded | no position signal; the causal mask alone breaks the symmetry (a control) |

Under a learned table, cached decoding still stops at the table's edge, as
Phase 19 documented. Under rotary or no positions it keeps going, and with
sliding or linear layers the per-layer state stays bounded however far it
goes: `hybrid/rotary_decoding_continues_past_the_table` decodes past the
context and checks the cached logits against a full recompute over the whole
longer sequence. Whether a model *trained* at one length is any good at
another is not claimed.

## Decode-time state

`LayerState` replaces the Phase 19 key/value cache per layer: keys and
values for dense and retrieval layers, the last `window - 1` of them for a
sliding layer (with the absolute position of the oldest key kept, so the
window mask stays correct after forgetting), the `(S, z)` pair for a linear
layer, and both for a learned one. `KvCache::resident_floats` reports the
footprint, and `hybrid/every_mode_decodes_from_its_state_exactly` checks
every mode under every position kind and several chunkings against the
full recompute.

## Routing state

Every router in the trunk decides from what it can see. With
`--routing-state s` each layer carries a per-token state

```text
r_l = tanh(W_h LN(h_l) + W_r r_{l-1} + b)
```

updated from the layer's *input*, and every router built with the state --
the FFN expert router, the MoSME box and expert routers, the value-expert
router of Phase 26 -- takes it appended to its input. The state is bounded
by the `tanh` and lives for one forward pass (it is a function of the
token's own hidden states, so there is nothing to cache across steps). With
`s = 0` nothing is built: the router inputs and the parameter count are
exactly what they were (`hybrid/routing_state_is_bounded_and_carries_history`).

## Router report

`RoutingStats` is the one report every router emits, whatever axis it
decides (`RouterKind`: FFN, value, attention mode, depth, branch). Phase 25
adds two locality numbers to the load and the two entropies of Phase 23:

- **stability** -- the fraction of adjacent positions, within a sequence,
  that chose the same top-1 expert. High stability over long trajectories
  is what would justify expert residency or paging (issue #4, B7); low
  stability says not to build it.
- **agreement** -- the fraction of tokens whose top-1 expert matches the
  previous routed layer's, for layers of equal width. The number the
  routing state is meant to raise.

Both reach the JSONL log and the end-of-run table through the existing
`RouterAux` path.

## Cost model

`cost.rs` counts active parameters, FLOPs (multiply-adds times two), keys
read and decode-state floats per token from the shapes alone, per layer and
per trunk (`LmConfig::cost`). `dblocks lm train` prints the total at the
model's context; `dblocks lm bench` records it beside the timing, because on
a CPU backend every mode is executed densely and wall-clock measures the
backend, not the architecture. The quality-per-active-FLOP axis of the
roadmap is read from these counts.

## What is and is not claimed

Certified (the `hybrid` group): an all-dense schedule is the Phase 19 trunk
bit for bit; a window or a retrieval set covering the sequence is dense
attention bit for bit; the linear recurrence equals the masked form; rotary
scores depend only on distance; every mode decodes from its state exactly
as it recomputes; rotary decoding continues past the table; the routing
state is bounded, carries history and costs nothing when off; the locality
diagnostics read as specified.

Not claimed: that any schedule, position kind or routing state improves
quality per FLOP. `dblocks lm bench --axis attention | positions | routing`
is the harness; the numbers it produces on a CPU in a few steps say what
the mechanisms cost, not what they buy.

See also: [Language Modeling](Language-Modeling.md) ·
[MoE Routing](MoE-Routing.md) · [Configuration](Configuration.md) ·
[Claims](Claims.md)
