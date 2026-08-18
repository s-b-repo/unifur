# Adaptive Depth / Block Skipping

Learned halting probabilities so the model can dynamically skip blocks
when confident enough, enabling dynamic compute per sample at inference.

## Overview

In standard DiffusionBlocks, all B blocks are evaluated for every sample.
But some samples are "easy" and don't need all blocks, while "hard" samples
may benefit from the full depth.

**Adaptive depth** learns a halting probability at each block. If the model
is confident enough, subsequent blocks are skipped.

## Halting Probability

Each block has a halting module that predicts a halting probability:

```python
class HaltingModule(nn.Module):
    def __init__(self, hidden_size):
        self.halting_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)  # (batch, hidden)
        logit = self.halting_proj(pooled).squeeze(-1)
        return torch.sigmoid(logit)
```

## Adaptive Depth Controller

Manages halting probabilities across all blocks:

```python
class AdaptiveDepthController(nn.Module):
    def __init__(self, num_blocks, hidden_size):
        self.halting_modules = nn.ModuleList([
            HaltingModule(hidden_size) for _ in range(num_blocks)
        ])
    
    def forward(self, block_outputs, training=True):
        halting_probs = [module(h) for module, h in 
                         zip(self.halting_modules, block_outputs)]
        
        if training:
            # Sample halting decisions (straight-through estimator)
            halting_decisions = torch.bernoulli(halting_probs)
            weights = halting_decisions - halting_probs.detach() + halting_probs
        else:
            # Use threshold at inference
            weights = (halting_probs > 0.5).float()
        
        return weights
```

## Training

During training, we use the straight-through estimator:

```python
# Forward: sample binary decisions
halting_decisions = torch.bernoulli(halting_probs)

# Backward: use continuous probabilities
halting_weights = halting_decisions - halting_probs.detach() + halting_probs
```

This allows gradients to flow through the halting probabilities while still
using discrete decisions in the forward pass.

## Depth Regularization

We regularize the expected depth toward a target:

```python
# Expected depth = sum_t P(halt at t)
cumulative_continue = torch.cumprod(1 - halting_probs, dim=1)
expected_depth = (halting_probs * cumulative_continue).sum(dim=1).mean()

# L2 regularization toward target
reg_loss = (expected_depth - target_depth) ** 2
```

## Inference

At inference, we can skip blocks based on halting decisions:

```python
for i, block in enumerate(blocks):
    if not should_skip[i]:
        hidden = block(hidden)
    
    # Check halting
    halting_prob = halting_module(hidden)
    if halting_prob > threshold:
        break  # Skip remaining blocks
```

## Benefits

| Benefit | Description |
|---|---|
| **Dynamic compute** | Easy samples use fewer blocks |
| **Efficiency** | Average depth can be much less than max depth |
| **Regularization** | Depth regularization prevents overfitting |
| **Interpretability** | Block usage reveals sample difficulty |

## Configuration

```bash
# Enable adaptive depth
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --adaptive_depth --target_depth 2.0

# With custom halting threshold
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --adaptive_depth --halting_threshold 0.9
```

## References

- PonderNet (Banino et al., 2021)
- Looped Transformers (Fan et al., 2025)
- Universal Transformers (Dehghani et al., 2019)
