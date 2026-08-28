# Flow Matching Objective

Alternative to EDM's score-matching objective based on rectified flow
(straight paths from noise to data). Enables ODE-based inference without
the complex noise schedule.

> **In this repository.** `src/flow.rs`, via `flow::flow_matching_loss` and
> `dblocks train --objective flow`.
>
> The label embedding is `x0`, Gaussian noise is `x1`, the conditional OT path
> is the straight line `x_t = (1-t)x_0 + t x_1` with `t ~ U(0,1)`, and the
> velocity target is `v* = x_1 - x_0`. `flow::flow_sample` integrates
> `dz/dt = v(z, t)` backwards from `t = 1` with Euler steps.
>
> Classification is by **cosine similarity** against the label table, with both
> sides L2-normalized. The raw dot product would rank labels partly by
> embedding norm, and the table is not norm-uniform — a bug this repository
> shipped and fixed.


## Overview

EDM (Elucidating Diffusion Models) uses score matching: the model learns
to predict the score function ∇_x log p(x). This requires a specific noise
schedule and preconditioning.

**Flow Matching** takes a different approach: the model learns to predict
the velocity field v(x,t) that transports noise to data along straight paths.

## Rectified Flow

### Definition

Rectified flow defines a straight-line path between noise and data:

```
z_t = (1 - t) * z + t * noise
```

where:
- z is the clean data embedding
- noise is Gaussian noise
- t ∈ [0, 1] is the timestep

### Velocity Field

The velocity field is the derivative of the path:

```
v(z_t, t) = d/dt z_t = noise - z
```

### Training Loss

The model predicts the velocity field v̂(z_t, t):

```
L_flow = E[ || v̂(z_t, t) - (noise - z) ||² ]
```

### Inference

At inference, we integrate the ODE from t=0 (noise) to t=1 (data):

```
z_{t+dt} = z_t + dt * v̂(z_t, t)
```

## Comparison with EDM

| Aspect | EDM (Score Matching) | Flow Matching |
|---|---|---|
| Target | Score ∇_x log p(x) | Velocity v(x,t) |
| Path | Curved (diffusion) | Straight line |
| Noise schedule | Log-normal CDF | Linear |
| Preconditioning | c_skip, c_out, c_in | None needed |
| Inference | Euler/ODE solver | Euler/ODE solver |
| Steps needed | 20-50 | 10-30 |

## Integration with DiffusionBlocks

### Block-wise Flow Matching

Each block learns a velocity field for its noise range:

```
Block 0: v̂(z_t, t) for t ∈ [0, 1/3]
Block 1: v̂(z_t, t) for t ∈ [1/3, 2/3]
Block 2: v̂(z_t, t) for t ∈ [2/3, 1]
```

### Training Loop

```python
for batch in dataloader:
    # Sample timestep
    t = torch.rand(batch_size)
    
    # Interpolate
    noise = torch.randn_like(z)
    z_t = (1 - t) * z + t * noise
    
    # Predict velocity
    v_pred = model(pixel_values, z_t, t)
    
    # Compute loss
    loss = F.mse_loss(v_pred, noise - z)
```

### Inference Loop

```python
# Start from noise
z = torch.randn(batch_size, hidden_size)

# Integrate ODE
dt = 1.0 / num_steps
for i in range(num_steps):
    t = torch.full((batch_size,), i * dt)
    v = model(pixel_values, z, t)
    z = z + dt * v
```

## Optimal Transport (OT) Flow

An alternative to rectified flow that uses constant-velocity paths:

```
v_ot = sign(noise - z)
```

This can be more stable in some cases.

## Benefits

| Benefit | Description |
|---|---|
| **Simpler schedule** | Linear instead of log-normal |
| **Straight paths** | Fewer integration steps needed |
| **No preconditioning** | No c_skip/c_out/cin needed |
| **Better for some tasks** | Especially classification |

## References

- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Rectified Flow: Straight-Line Probability Transport (Liu et al., 2022)
- Original DiffusionBlocks paper (Shing et al., 2026)

---

See also: [Quality Gate](Quality-Gate.md) · [Training Guide](Training-Guide.md) · [Inference Guide](Inference-Guide.md) · [Home](Home.md)
