# MoE Routing

Mixture-of-experts routing for DiffusionBlocks++. Each block can have
multiple expert sub-networks that specialize in different noise regimes.

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
