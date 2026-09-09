# MoE Routing

Mixture-of-experts routing for DiffusionBlocks++. Each block can have
multiple expert sub-networks that specialize in different noise regimes.

> **In this repository.** `src/moe.rs` (`MoELayer`, `TopKRouter`, `MoEConfig`)
> plus trunk integration through `ViTDiTConfig::moe` / `MoeTrunkConfig`.
>
> `MoeTrunkConfig::every_n_layers` replaces every *n*-th layer's dense MLP with
> a mixture of experts — alternating dense and sparse layers is the standard
> Switch/GLaM placement and keeps a dense path for the features every expert
> needs.
>
> **Noise-aware routing (6.4)** falls out of the conditioning: the router sees
> the adaLN vector, which is a pure function of sigma. With
> `MoEConfig::route_on_tokens` it also sees the token's own features, so tokens
> of the same example can diverge; without it every token in an example routes
> identically.
>
> **Router z-loss** (`moe::router_z_loss`, `--z-level`) and **balance-weight
> annealing** (`--balance-schedule anneal`) address the two routing pathologies
> the literature is clearest about — see [Routing Quality](#routing-quality).
> Routing diagnostics (`moe::RoutingStats`, logged on every sparse run),
> global-batch load (`--balance-scope global`) and loss-free bias balancing
> (`--bias-balance-rate`) close roadmap 23.4–23.6 — see
> [What shipped](#what-shipped-and-what-it-reports).
>
> The Switch load-balancing loss of the executed span is summed and added to
> the objective automatically (`DblockConfig::moe_aux_weight`, default 0.01),
> and logged as `balance_loss`.
>
> Two bounds are certified: gates form a probability distribution per token,
> and on the diagonal `f == p` the balance loss lies in `[1, E]` by
> Cauchy-Schwarz, attaining 1 exactly at uniform routing. (For *arbitrary*
> `f` and `p` only `0 <= L <= E` holds — conflating the two is an easy mistake.)
>
> CLI: `--moe-every N --moe-experts E --moe-top-k K`.
>
> This layer is **flat**: one router over a homogeneous expert vector, with
> experts addressed by integer index only. For named specialists grouped into
> domains, with a manifest an inference engine can route from, see
> [Mixture of Specialized Micro Experts](Mixture-of-Specialized-Micro-Experts.md)
> — a single-box hierarchical layer reduces to this one bit-for-bit.


## Overview

Standard DiffusionBlocks uses the same network for all noise levels. But
different noise regimes may benefit from different computation:

- **High noise (σ large)**: Needs broad, global patterns
- **Low noise (σ small)**: Needs fine, local refinement
- **Boundary noise**: Needs to agree with adjacent blocks

**MoE routing** allows each block to have N expert sub-networks, with a
lightweight router selecting which expert to use per token.

## Architecture

```
Input (batch, seq_len, hidden)
    │
    ├──► Router (Linear → num_experts)
    │       └──► Top-K selection (weights, indices)
    │
    ├──► Expert 0 ──┐
    ├──► Expert 1 ──┤
    ├──► Expert 2 ──┼──► Weighted sum ──► Output
    └──► Expert 3 ──┘
```

## Top-K Router

The router computes routing probabilities:

```python
logits = self.router_nn(x)           # (batch, seq, num_experts)
weights = softmax(logits, dim=-1)
topk_weights, topk_indices = topk(weights, K, dim=-1)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
```

**Key points**:
- Top-K: only K experts are active per token (sparse)
- Renormalize: weights sum to 1 over the K active experts
- Load balancing: auxiliary loss encourages uniform expert usage

## Load Balancing Loss

Without regularization, the router may collapse to using only a few experts.
The load balancing loss encourages uniform distribution:

```python
# Count tokens per expert
expert_load = one_hot(indices).sum(dim=(0, 1))  # (num_experts,)
expert_fraction = expert_load / expert_load.sum()

# Average routing probability
avg_weights = weights.mean(dim=(0, 1))  # (num_experts,)

# Target: uniform distribution
target = 1.0 / num_experts
loss = ((expert_fraction - target) ** 2).sum() + ((avg_weights - target) ** 2).sum()
```

## Noise-Aware Router

For DiffusionBlocks++, the router can also condition on the noise level:

```python
class NoiseAwareRouter:
    def forward(self, x, sigma_emb):
        # sigma_emb: noise level embedding (batch, cond_hidden)
        cond = self.cond_proj(sigma_emb)  # (batch, hidden)
        x_cond = x + cond.unsqueeze(1)     # Broadcast to sequence
        return super().forward(x_cond)
```

This allows experts to specialize in different noise regimes:
- Expert 0: High noise (σ > 10)
- Expert 1: Medium noise (1 < σ < 10)
- Expert 2: Low noise (σ < 1)
- Expert 3: Boundary noise (σ ≈ boundary)

## MoE Block

Each block in DiffusionBlocks++ can be an MoE block:

```python
class MoEBlock(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        self.attention = Attention(hidden_size)  # Shared
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size) for _ in range(num_experts)
        ])
        self.router = Router(hidden_size, num_experts, top_k)
    
    def forward(self, x, training=True):
        # Shared attention
        x = x + self.attention(x)
        
        # MoE MLP
        weights, indices, aux_loss, metrics = self.router(x, training)
        moe_output = self.dispatch_to_experts(x, weights, indices)
        return x + moe_output, aux_loss, metrics
```

## Benefits

| Benefit | Description |
|---|---|
| **Specialization** | Different experts handle different noise regimes |
| **Capacity** | More parameters without more compute per token |
| **Scalability** | Experts can be distributed across devices |
| **Efficiency** | Only K of N experts are active per token |

## Configuration

```bash
# Enable MoE with 4 experts, top-2 routing
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --moe --num_experts 4 --top_k 2

# With noise-aware routing
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --moe --num_experts 4 --top_k 2 \
    --noise_aware_router
```

## References

- Outrageously Large Neural Networks (Shazeer et al., 2017)
- Mixtral of Experts (Jiang et al., 2024)
- Switch Transformer (Fedus et al., 2021)
- ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al., 2022) — router z-loss
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized MoE Models (Zhu et al., 2025) — global-batch LBL
- Auxiliary-Loss-Free Load Balancing for Mixture-of-Experts (Wang et al., 2024) — bias-based balancing

---

## Routing Quality

Two failure modes dominate the MoE training literature, and they pull in
opposite directions.

### 1. Logit drift — and why the balance loss cannot see it

The routing softmax exponentiates its logits, and **the softmax is invariant to
a per-row constant shift**. So a router can drift arbitrarily large while every
routing probability — and therefore the entire balance loss — stays exactly
where it was. Once the logits are large, the exponentials amplify small
numerical errors into round-off that destabilizes training, in f32 as well as in
reduced precision.

The router z-loss (Zoph et al., ST-MoE) is the term that *can* see it:

```text
L_z = mean_t ( logsumexp_e x_te )^2
```

```bash
dblocks train --z-level 1e-3     # the ST-MoE default; 0.0 disables it exactly
```

Three implementation points that are easy to get wrong:

- **It penalizes the log-sum-exp, not the logits.** That is what bounds the
  largest one: `max_e x_e <= logsumexp <= max_e x_e + ln E`, so holding the
  log-sum-exp near zero holds *every* logit within `ln E` of zero. Certificate
  `moe/logsumexp_bounds_the_largest_logit`.
- **Its optimum is not a logit of zero.** For a row of `E` equal logits it is
  `-ln E`, and an offset `d` from there costs exactly `d²`. Certificate
  `moe/z_loss_optimum_is_a_zero_logsumexp`.
- **A width-1 router is charged nothing.** A softmax over one element is 1
  whatever the logit is, so that logit steers nothing — charging it would put
  gradient on a parameter with no alternatives, and it would break the
  single-box reduction. This crate found that the second way.

It is not free of routing pressure: the gradient is `2 z p`, proportional to the
softmax, so it shrinks confident logits slightly harder than diffident ones.
Keep the weight small.

**`z_level` is a coefficient on the total loss**, matching ST-MoE's convention.
That took a fix: the z-loss was originally summed into the balance term, which
is itself scaled by `moe_aux_weight` (0.01), so a configured `1e-3` reached the
objective at `1e-5`. Worse, `--balance-schedule anneal` decayed it — attenuating
a numerical *stabilizer* exactly as a run gets long enough for logit drift to
matter. The two are now carried separately end to end (`vit::RouterAux`): the
balance term is scaled and may be annealed, the z-loss is not.

### 2. The balance loss fights specialization

Zhu et al. (*Demons in the Detail*, 2025) show that computing the load-balancing
loss over a **micro-batch** pushes the router to spread tokens evenly *within
each batch*. Since a micro-batch holds few distinct inputs, that pressure lands
on individual sequences and forces even clearly domain-specific inputs to route
uniformly — it **actively inhibits the specialization the experts exist for**.

Their fix computes the loss over the global batch. That needs the per-expert
counts to escape the forward pass, which this crate cannot do yet (roadmap
23.4).

What ships is the tractable half of the same idea: anneal the *weight*.

```bash
dblocks train --balance-schedule anneal --balance-weight 0.01
```

Early in a run the balance loss is doing the job it is good at — preventing
routing collapse, where one expert wins everything and the rest never receive
gradient. Once every expert is alive, that pressure has nothing left to buy and
starts costing specialization instead. So the schedule holds it high while
collapse is the risk and decays it once it is not.

The decay is **geometric**, because the useful range spans two decades (`1e-2`
early, `1e-4` late) and a linear ramp would spend almost all of its steps near
the top.

### Measured on MoSME

400 steps, 3 blocks, seed 42, MoSME with 2 boxes / 5 experts on every second
trunk layer. Identical block visit counts across every run (138 / 145 / 117):

| Configuration | mean loss | rejected | block 2 mean \|g\| | block 2 peak \|g\| |
|---|---|---|---|---|
| MoSME baseline (`--z-level 0`) | 482.90 | 5 | 2.44e3 | 8.19e4 |
| `--z-level 1e-3` | 494.76 | 5 | 2.36e3 | **7.03e4** |
| `--z-level 1e-3 --uncertainty 1.0` | **136.34** | **0** | **1.90e2** | **9.53e3** |

The z-loss does what it is for: block 2's **peak** gradient falls 14%. It is
just not the binding constraint here — the five gate rejections survive it,
because the gate is firing on the EDM weight `w(sigma)` diverging in block 2,
not on router logit drift. [Loss Reduction](Loss-Reduction.md)'s uncertainty
weighting is what addresses that, and does.

Read the mean-loss column carefully: with `z_level > 0` the z term is part of
the reported loss, so some of the 482.90 → 494.76 rise is the term being counted
rather than the model being worse. The gradient columns are the fair comparison.

**Practical recommendation for a MoSME run:** start with `--uncertainty 1.0`.
Leave `--z-level` at its default — it is cheap insurance against a slow failure
mode that 400 CPU steps cannot produce — and reach for `--balance-schedule
anneal` only once a routing diagnostic exists to tell you whether it helped.

### What shipped, and what it reports

The three items the survey above named are now in the trunk (roadmap 23.4–23.6).

**Diagnostics (23.6).** A balance loss tells you whether the *load* is even; it
says nothing about whether routing is *confident*. Every sparse layer now
reports, per step, its top-1 load fraction per expert and two normalized
entropies (`moe::RoutingStats`):

| Load entropy | Per-token entropy | Reading |
|---|---|---|
| low | — | routing collapse: one expert wins everything |
| high | high | **balanced but unspecialized** — the failure mode above |
| high | low | balanced *and* specialized — what you want |

The middle row is what a per-micro-batch balance loss produces and what its own
value cannot show. The numbers reach the JSONL log (`load_entropy`,
`token_entropy`, `min_load`, `max_load`, `routing_load`), the language-model
step (`LmMetrics::routing_entropy` / `routing_max_load`), and the end-of-run
per-block table as `load H` / `token H`. Both entropies are certified to read
as specified — including the disagreement case — and the reported values are
certified against a host recomputation from the router's own probabilities.

**Global-batch load (23.4).** `--balance-scope global` replaces each layer's
micro-batch load `f` with its mean over the last `--accumulate` micro-batches,
recombined with *this* micro-batch's differentiable `p`
(`moe::switch_loss_parts`, `schedule::GlobalLoad`). Zhu et al.'s point is
exactly that `f` is an arg-max with no gradient of its own, so where it is
measured is a free choice — and measuring it per micro-batch pushes the router
to balance every batch on its own. Two certificates pin it: a window of one is
the fused Switch loss bit for bit, and two micro-batches routed entirely to
different experts cost `E` each alone but `E/2` over the window. Flat MoE only
for now; a MoSME trunk weights each box's term by its own gate, which one
window per layer cannot express yet.

**Loss-free bias balancing (23.5).** `--bias-balance-rate u` attaches a
non-trainable per-expert bias to every router that is added to the logits
**only when choosing** the top-k — the gate values are always the unbiased
probabilities — and moves it by `±u` per step against the observed load
(`TopKRouter::nudge_balance_bias`, DeepSeek's rule with `u = 1e-3`). Load
balancing then stops competing with the task loss for the router's weights.
Certified: a zero or uniform bias is a bitwise identity, a bias steers selection
without touching a selected expert's gate, one nudge moves each bias by exactly
the rate, and growth widens the bias with zeros. One hazard is documented in
the code: Burn zips an `Option` field with its record, so a bias missing on
either side of a checkpoint load comes back as `None` — the trainer re-attaches
zeros after every load.

---

See also: [Mixture of Specialized Micro Experts](Mixture-of-Specialized-Micro-Experts.md) · [Loss Reduction](Loss-Reduction.md) · [Quality Gate](Quality-Gate.md) · [Training Guide](Training-Guide.md) · [Home](Home.md)
