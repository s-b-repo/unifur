# Parallel Depth Denoising Trajectory

The core innovation of DiffusionBlocks++: training **multiple blocks simultaneously**
on overlapping noise windows, with a fork-reconverge protocol that enables
inter-block cooperation while maintaining independent trainability.

> **In this repository.** Span execution is
> `DblockClassifier::denoise_span`; running only a span *is* gradient routing,
> since gradients flow exclusively through executed layers. Fork-reconverge
> happens through the shared latent between windows.
>
> Parallel sampling is only legitimate if a joint span reproduces the
> sequential composition of the same blocks. That is **trained, not assumed**:
> `ConsistencyWeights::cross_fork` samples a non-adjacent pair `(i, j)` with
> `j >= i+2` and penalizes the difference between the forked and sequential
> paths. See [Consistency Training](Consistency-Training.md).
>
> Overlapping windows come from `gamma` (`DblockSigmaSampler::extended_window`).
>
> **Block indexing:** boundaries ascend but block indices descend in noise
> level — block 0 owns the noisiest window. The `block_routing_involution`
> certificate asserts the training sampler and the inference router are mutual
> inverses; getting this backwards trains every block on one noise range and
> evaluates it on another, which is a bug this repository shipped and fixed.


## Motivation

Original DiffusionBlocks trains **one block per step**. With B blocks, you need
B× more steps than end-to-end training. Each block sees only 1/B of the
gradient signal and has no mechanism to cooperate with other blocks.

**Parallel depth denoising** addresses this by training K adjacent blocks at once
on overlapping sigma windows. The overlapping regions act as a soft consensus
mechanism.

## The Fork-Reconverge Protocol

### Step 1: Fork

The denoising trajectory splits into K parallel branches, one per active block.
Each branch receives the input at the appropriate noise level.

```
Input z
  │
  ├── noise(σ_max) ──► zt_max  (for Block 0)
  │
  ├── noise(σ_a) ────► zt_a    (for Block 1)
  │
  └── noise(σ_b) ────► zt_b    (for Block 2, if K=3)
```

### Step 2: Denoise (Parallel)

Each block independently minimizes its score-matching loss on its branch.

```
Block 0: zt_max → denoise(σ_max → σ_a) → output_0
Block 1: zt_a   → denoise(σ_a → σ_b)   → output_1
Block 2: zt_b   → denoise(σ_b → σ_min) → output_2  (if K=3)
```

### Step 3: Reconverge

At the boundary sigma between adjacent blocks, a consistency loss encourages
the blocks to agree on the denoised output.

```
L_total = Σ L_block_i + λ · Σ L_consistency(boundary_i,j)
```

## Overlapping Sigma Windows

The key design choice is how much overlap to give adjacent blocks.

```
No overlap (K=1, original):
  Block 0: [σ_max, σ_a]
  Block 1: [σ_a, σ_b]
  Block 2: [σ_b, σ_min]

Partial overlap (K=2, γ=0.1):
  Block 0: [σ_max, σ_a + δ]
  Block 1: [σ_a - δ, σ_b + δ]
  Block 2: [σ_b - δ, σ_min]

Full overlap (K=2, γ=0.5):
  Block 0: [σ_max, σ_b]
  Block 1: [σ_a, σ_min]
```

**Tradeoff**: More overlap → more consensus signal, but each block's effective
noise range is smaller.

## Gradient Routing

Gradients flow **only** through the active block pair. Inactive blocks are
frozen during the step.

```
Step t (Block 0 + Block 1 active):
  ∂L/∂θ_0 ≠ 0
  ∂L/∂θ_1 ≠ 0
  ∂L/∂θ_2 = 0  (frozen)

Step t+1 (Block 1 + Block 2 active):
  ∂L/∂θ_0 = 0  (frozen)
  ∂L/∂θ_1 ≠ 0
  ∂L/∂θ_2 ≠ 0
```

This maintains the memory benefit: only K blocks' activations are stored.

## Cross-Fork Training

Beyond adjacent blocks, we can also train non-adjacent blocks together:

```
Standard fork:     Block 0 + Block 1 (adjacent)
Cross fork:        Block 0 + Block 2 (skip Block 1)
```

Cross-fork training provides a form of skip connection in the training signal,
allowing blocks to learn from distant parts of the network.

## Consistency Loss

The consistency loss at block boundaries is:

```
L_consistency = E[ || denoise(x, σ_b, Block_i) - denoise(x, σ_b, Block_{i+1}) ||² ]
```

This encourages adjacent blocks to produce similar denoised outputs when given
the same noisy input at the boundary sigma.

**Implementation**: For each boundary sigma, we:
1. Sample a batch of (x, z) pairs
2. Add noise at the boundary sigma: zt = z + σ_b * ε
3. Run both blocks on (x, zt, σ_b)
4. Compute MSE between their outputs

## Training Schedule

With B blocks and K parallel blocks, the training schedule cycles through
all C(B, K) combinations:

```
B=3, K=2:
  Step 0: Block 0 + Block 1
  Step 1: Block 1 + Block 2
  Step 2: Block 0 + Block 2  (cross-fork)
  Step 3: Block 0 + Block 1  (repeat)
  ...
```

Each block appears in C(B-1, K-1) combinations, so it gets updated
C(B-1, K-1) / C(B, K) = K / B fraction of steps.

## Comparison with Original

| Metric | Original (K=1) | Parallel (K=2) | Parallel (K=3) |
|---|---|---|---|
| Steps to cover all blocks | B | ⌈B/(K-1)⌉ | ⌈B/(K-1)⌉ |
| Gradient signal per step | 1/B | K/B | K/B |
| Inter-block communication | None | Via overlap + consistency | Via overlap + consistency |
| Memory | O(1) | O(K) | O(K) |
| Consistency signal | None | Strong | Very strong |

## Implementation Notes

### Sigma Schedule

The base sigma schedule uses log-normal CDF spacing:

```python
def get_block_sigmas(num_layers, sigma_min, sigma_max, p_mean, p_std):
    cdf_min = norm.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max) - p_mean) / p_std)
    block_sigmas = []
    for i in range(num_layers + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / num_layers)
        sigma = np.exp(p_mean + p_std * norm.ppf(p))
        block_sigmas.append(sigma)
    return block_sigmas
```

### Overlap Extension

The overlap parameter γ extends each block's sigma range:

```python
def extend_sigma_range(sigma_min, sigma_max, gamma):
    log_min = np.log(sigma_min)
    log_max = np.log(sigma_max)
    log_range = log_max - log_min
    new_min = np.exp(log_min - gamma * log_range)
    new_max = np.exp(log_max + gamma * log_range)
    return new_min, new_max
```

### Gradient Masking

Only active blocks receive gradients:

```python
for i, block in enumerate(blocks):
    if i in active_indices:
        block.requires_grad_(True)
    else:
        block.requires_grad_(False)
```

## Open Questions

1. **Optimal overlap**: How much overlap gives the best quality/speed tradeoff?
   - Too little: blocks don't cooperate
   - Too much: each block's effective range is too small

2. **Adaptive K**: Should K vary during training?
   - Start with K=1 (stable), increase to K=B (full cooperation)

3. **Cross-fork frequency**: How often to use cross-fork vs adjacent pairs?
   - More cross-fork: better long-range signal, but less local coherence

4. **Consistency weight scheduling**: Should λ vary during training?
   - Start low (blocks learn independently), increase (enforce cooperation)

## References

- Original DiffusionBlocks paper (Shing et al., 2026)
- Universal Transformers (Dehghani et al., 2019)
- Looped Transformers (Fan et al., 2025)
- PonderNet (Banino et al., 2021)

---

See also: [Quality Gate](Quality-Gate.md) · [Training Guide](Training-Guide.md) · [Inference Guide](Inference-Guide.md) · [Home](Home.md)
