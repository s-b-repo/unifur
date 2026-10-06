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

## Qwen-style trunk knobs (all default-off, all certified)

Portable pieces of the current open frontier (Qwen3-Next, Kimi Linear,
GLM-4.5), each an exact identity when off:

| Flag | Meaning | Off state |
|---|---|---|
| `--kv-heads k` | Grouped-query attention: `k` KV heads per layer | Full MHA, bit for bit (`gqa_with_full_heads_is_dense…`) |
| `--rotary-fraction f` | Rotary cover as a fraction of the head dim (Qwen3-Next uses 0.25, GLM-4.5 uses 0.5) | Full rotation, bit for bit |
| `--gated-attention` | Merged heads scaled by `1 + tanh(g(x))`, zero-init gate | Ungated trunk to 1e-6 (measured 0.0) |
| `--norm rmsnorm` | RMSNorm instead of LayerNorm on trunk layers + final norm | LayerNorm, checkpoints unchanged |
| `--mtp-steps k --mtp-weight w` | Multi-token prediction: auxiliary CE on `k` future offsets (GLM uses λ 0.3 → 0.1) | Plain next-token loss, bit for bit |

Two findings from those labs are already this crate's defaults: the `3:1`
linear:dense schedule is their measured-best hybrid ratio (Kimi Linear §7.2:
7:1 collapses on val, 1:1 only costs more), and `--bias-balance-rate` is
DeepSeek's loss-free balancing. Two of their postmortems are now runtime
behavior: retrieval on a CPU backend prints a warning (it computes every
score, so wall-clock measures the backend, not the architecture), and the
readout-gate scatter that once broadcast rows to sum `n` instead of 1 is
pinned by `geom/readout_gates_form_a_distribution`.

Training levers ported from the image loop to `lm train` (roadmap Phase 20):
`--lr-schedule cosine|warmup`, `--accumulate k` (averaged, clip applies to
the sum), `--clip-norm`, `--ema-decay` (bias-corrected shadow returned for
evaluation and saved beside the checkpoint for resume). `geom train` gains
`--accumulate` and `--ema-decay` the same way. With every lever off, both
loops are numerically what they were.

Every `lm train` log line also carries `maxlogit`: the largest causally-valid
attention score over the trunk on a truncated probe sample (Kimi K2's
early-warning metric — loss and grad-norm miss logit blowup until it spikes
the run; past 100 with `--qk-norm` off is blowup territory and the line says
so). Linear layers keep no scores and report nothing. Logged to JSONL as
`max_logit` beside the loss.

## What is and is not claimed

Certified (the `hybrid` group): an all-dense schedule is the Phase 19 trunk
bit for bit; a window or a retrieval set covering the sequence is dense
attention bit for bit; the linear recurrence equals the masked form; rotary
scores depend only on distance; every mode decodes from its state exactly
as it recomputes; rotary decoding continues past the table; the routing
state is bounded, carries history and costs nothing when off; the locality
diagnostics read as specified; partial rotary is full rotary at fraction one
and leaves the suffix; GQA with full heads is dense and narrow KV shrinks
the decode state; gated attention is identity at init; QK-normalized trunks
decode exactly and move the output; every causal mode reads its past (a
mask deaf to everything would still decode exactly, so only perturbation
sees it); RMSNorm starts at unit RMS; MTP at weight zero is the plain loss.

Not claimed: that any schedule, position kind, routing state, GQA factor,
gate, norm or MTP setting improves quality per FLOP.
`dblocks lm bench --axis attention | positions | routing` is the harness;
the numbers it produces on a CPU in a few steps say what the mechanisms
cost, not what they buy. The labs' numbers (Kimi 3:1, GLM 0.5-RoPE) are
their measurements on their scale, cited as priors, not transferred facts.

## Field notes from other labs (what ports, what does not)

- **QK-Norm** (`--qk-norm`): Qwen3 (`use_qk_norm`), GLM-4.5 (large variant),
  LLaMA-4 Scout (QK-RMSNorm, no affine) all normalize queries/keys; Kimi K2's
  postmortem is attention logits past 100 with soft-cap/QK-Norm alone judged
  inadequate next to their MuonClip rescale. Under AdamW (this crate's
  optimizer) QK-Norm is the standard stabilizer: scores bounded by the head
  dim whatever the projections learn. Not yet here: max-logit-per-layer
  logging (K2's actual early-warning metric) — the next cheap addition.
- **Keep a dense layer first.** DeepSeekMoE, OLMoE, Kimi K2 and GLM all keep
  layer 0 (or the first blocks) dense: balance converges slowest there. The
  default `--moe-every 2` / `--mosme-every 2` already leaves layer 0 dense
  (`applies_to` fires on odd indices); setting either to 1 opts out
  deliberately.
- **NoPE/partial-RoPE must be consistent.** Kimi Linear ships NoPE everywhere
  because RoPE-on-dense plus nothing-on-linear over-emphasized short range;
  GLM uses partial RoPE (0.5) on all layers. Here rotary applies inside every
  mode (including linear) when enabled, so there is no mixed regime to fall
  into; `--rotary-fraction` covers the Qwen3-Next (0.25) and GLM (0.5) points.
- **No linear/SWA for long-range reasoning.** MiniMax M2 abandoned its
  lightning/SWA hybrid after 100B–1T ablations: matched short-context scores,
  collapsed past 32k retrieval/multi-hop. Sliding and retrieval stay opt-in
  here for exactly this reason; the default is dense.
- **More heads can help reasoning without helping loss** (GLM-4.5: 96 heads
  on 5120 dim, no train-loss gain, consistent MMLU/BBH gain). Do not prune
  `--num-heads` on loss alone.

See also: [Language Modeling](Language-Modeling.md) ·
[MoE Routing](MoE-Routing.md) · [Configuration](Configuration.md) ·
[Claims](Claims.md)
