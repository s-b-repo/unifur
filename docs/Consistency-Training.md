# Consistency Training

Self-consistency losses for DiffusionBlocks++ that encourage adjacent blocks
to produce consistent predictions at their boundaries, enabling faster
convergence and fewer inference steps.

## Overview

In standard DiffusionBlocks, each block is trained independently. This means
Block_i and Block_{i+1} may produce very different outputs at their shared
boundary sigma, leading to:

1. Discontinuities in the denoising trajectory
2. Slower convergence (blocks work against each other)
3. Need for more inference steps to smooth out inconsistencies

**Consistency training** adds a loss term that penalizes disagreement between
adjacent blocks at their boundary sigma.

## Types of Consistency Loss

### 1. Boundary Consistency

At the boundary sigma σ_b between Block_i and Block_{i+1}:

```
L_boundary = E[ || denoise(x, σ_b, Block_i) - denoise(x, σ_b, Block_{i+1}) ||² ]
```

This is the simplest form: both blocks should produce the same output when
given the same noisy input at their shared boundary.

### 2. Self-Consistency (Multi-Scale)

The model should produce consistent outputs across different noise levels:

```
L_self = E[ || denoise(x, σ_low, Block_i) - denoise(x, σ_high, Block_i) ||² ]
```

This encourages each block to be self-consistent across its own sigma range.

### 3. Trajectory Consistency

The full denoising trajectory should be smooth:

```
L_traj = E[ || denoise(x, σ_t, Block_i) - denoise(x, σ_{t+1}, Block_i) ||² ]
```

This penalizes large jumps between adjacent timesteps.

## Implementation

### Boundary Consistency Implementation

```python
def compute_boundary_consistency(model_i, model_j, z, boundary_sigma, pixel_values):
    """
    Compute consistency loss between two adjacent blocks at their boundary.
    
    Args:
        model_i: Block i's denoising function
        model_j: Block j's denoising function  
        z: Clean target embeddings
        boundary_sigma: The sigma value at the boundary
        pixel_values: Input conditioning
    """
    noise = torch.randn_like(z)
    z_noisy = z + boundary_sigma[:, None] * noise
    
    denoised_i = model_i(pixel_values, z_noisy, boundary_sigma)
    denoised_j = model_j(pixel_values, z_noisy, boundary_sigma)
    
    return F.mse_loss(denoised_i, denoised_j, reduction="mean")
```

### Integration with Parallel Training

In parallel depth denoising, consistency loss is computed between all active
block pairs:

```python
# For K active blocks, compute consistency at K-1 boundaries
consistency_losses = []
for i in range(len(active_blocks) - 1):
    block_i = active_blocks[i]
    block_j = active_blocks[i + 1]
    boundary_sigma = compute_boundary_sigma(block_i, block_j)
    
    loss = compute_boundary_consistency(
        model.blocks[block_i], model.blocks[block_j],
        z, boundary_sigma, pixel_values
    )
    consistency_losses.append(loss)

total_consistency = sum(consistency_losses) / len(consistency_losses)
```

## Consistency Weight Scheduling

The consistency weight λ controls the tradeoff between independent training
and cooperative training.

### Fixed Weight

```python
# Simple: fixed weight throughout training
lambda_consistency = 0.1
```

### Linear Ramp

```python
# Start low, increase linearly
lambda_consistency = min(0.1, step / warmup_steps * 0.1)
```

### Adaptive Weight

```python
# Increase when blocks disagree, decrease when they agree
disagreement = compute_block_disagreement()
lambda_consistency = base_lambda * (1 + disagreement)
```

## Benefits

| Benefit | Description |
|---|---|
| **Faster convergence** | Blocks cooperate instead of competing |
| **Smoother trajectory** | No discontinuities at block boundaries |
| **Fewer inference steps** | Consistent trajectory needs fewer integration steps |
| **Better generalization** | Consistency acts as regularization |

## Ablation Studies

Recommended ablation experiments:

1. **Weight sweep**: λ ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0}
2. **Boundary vs self-consistency**: Which helps more?
3. **Schedule comparison**: Fixed vs ramp vs adaptive
4. **With/without parallel denoising**: Does consistency help more with K>1?

## References

- Consistency Models (Song et al., 2023)
- Original DiffusionBlocks paper (Shing et al., 2026)
