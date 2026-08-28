# Architecture

Detailed architecture description for DiffusionBlocks++.

> **In this repository.** `src/vit.rs` (`ViTDiTModel`,
> `ViTDiTForImageClassification`, `DbLayer`, `DbOutputHead`) and `src/dblock.rs`
> (`DblockClassifier`).
>
> The noisy class embedding takes the CLS slot; every layer is conditioned by
> adaLN-zero on a sinusoidal timestep embedding of `c_noise = 0.25 log sigma`.
> The `1/sqrt(head_dim)` attention scale is folded into the Q projection at
> initialization, so the forward pass needs no extra elementwise multiply.
>
> A layer's feed-forward is `FeedForward::{Dense, Sparse}` — an enum rather
> than a boxed trait, so a checkpoint carries its own dense/sparse layout
> through Burn's serialization.
>
> `forward_block(range, ..)` runs only a contiguous layer window and applies
> the final LayerNorm; that primitive is what block-wise training, parallel
> spans and the loop graph are all built on.
>
> DiT initialization zeroes every adaLN modulation linear and the classifier,
> so a fresh model outputs **exactly** zero logits — certified as
> `model::dit_zero_init`.


## Overview

DiffusionBlocks++ partitions a transformer into independently trainable blocks.
Each block learns to denoise from a high noise level to a lower one, following
the EDM (Elucidating Diffusion Models) framework.

## Base Architecture: Vision Transformer (ViT)

The base model is a standard Vision Transformer:

```
Input Image (3×32×32)
    │
    ▼
Patch Embedding (16 patches of 4×4×3 = 48 dims → 128 dims)
    │
    ▼
Position Embedding (learnable, 16+1 positions)
    │
    ▼
Transformer Layer × 12
    ├── Multi-Head Self-Attention (4 heads)
    ├── LayerNorm
    ├── MLP (128 → 512 → 128)
    └── Residual Connection
    │
    ▼
Classification Head (128 → 100 classes)
```

## DiffusionBlocks Architecture

The 12 transformer layers are partitioned into B blocks (default B=3):

```
Input Image (3×32×32)
    │
    ▼
Patch Embedding
    │
    ▼
Block 0 (Layers 0-3)
    ├── AdaLN conditioning on noise level σ
    ├── Multi-Head Self-Attention
    ├── MLP
    └── Residual Connection
    │
    ▼
Block 1 (Layers 4-7)
    ├── AdaLN conditioning on noise level σ
    ├── Multi-Head Self-Attention
    ├── MLP
    └── Residual Connection
    │
    ▼
Block 2 (Layers 8-11)
    ├── AdaLN conditioning on noise level σ
    ├── Multi-Head Self-Attention
    ├── MLP
    └── Residual Connection
    │
    ▼
Classification Head
```

## AdaLN Conditioning

Each block uses Adaptive Layer Normalization (AdaLN) to condition on the
noise level:

```python
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

# In each layer:
shift, scale = adaLN(sigma_embedding).chunk(2)
x = modulate(layer_norm(x), shift, scale)
```

The sigma embedding is computed by a TimestepEmbedder (2-layer MLP with sinusoidal
embeddings).

## EDM Preconditioning

Following EDM, the denoising function uses preconditioning:

```
c_skip = σ_data² / (σ² + σ_data²)
c_out  = σ * σ_data / √(σ² + σ_data²)
c_in   = 1 / √(σ² + σ_data²)
c_noise = 0.25 * log(σ)

denoise(x, z_t, σ):
    return hidden_states * c_out + z_t * c_skip
```

## Parallel Depth Architecture

With parallel depth denoising (K=2), two blocks are trained simultaneously:

```
                    ┌─────────────────────────────┐
                    │     Input (pixel_values)     │
                    └──────────────┬──────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │  Patch Embedding    │
                         └─────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
     │   Block 0       │  │   Block 1      │  │   Block 2       │
     │   (σ_max → σ_a) │  │   (σ_a → σ_b)  │  │   (σ_b → σ_min) │
     │                 │  │                │  │                 │
     │  ┌───────────┐  │  │  ┌───────────┐ │  │  ┌───────────┐  │
     │  │ AdaLN(σ)  │  │  │  │ AdaLN(σ)  │ │  │  │ AdaLN(σ)  │  │
     │  └─────┬─────┘  │  │  └─────┬─────┘ │  │  └─────┬─────┘  │
     │  ┌─────▼─────┐  │  │  ┌─────▼─────┐ │  │  ┌─────▼─────┐  │
     │  │ Attention │  │  │  │ Attention │ │  │  │ Attention │  │
     │  └─────┬─────┘  │  │  └─────┬─────┘ │  │  └─────┬─────┘  │
     │  ┌─────▼─────┐  │  │  ┌─────▼─────┐ │  │  ┌─────▼─────┐  │
     │  │   MLP     │  │  │  │   MLP     │ │  │  │   MLP     │  │
     │  └─────┬─────┘  │  │  └─────┬─────┘ │  │  └─────┬─────┘  │
     └────────┼────────┘  └────────┼────────┘  └────────┼────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │   Output Head      │
                         └───────────────────┘
```

## MoE Block Architecture

With MoE routing enabled, each block has multiple expert MLPs:

```
Input
  │
  ├──► Router (Linear → num_experts)
  │       └──► Top-K selection
  │
  ├──► Expert 0 ──┐
  ├──► Expert 1 ──┼──► Weighted sum
  ├──► Expert 2 ──┤
  └──► Expert 3 ──┘
  │
  ▼
Output
```

## Adaptive Depth Architecture

With adaptive depth, each block has a halting module:

```
Block 0 ──► Halting Module 0 ──► p₀
  │
  ├── if p₀ > threshold: STOP
  │
  ▼
Block 1 ──► Halting Module 1 ──► p₁
  │
  ├── if p₁ > threshold: STOP
  │
  ▼
Block 2 ──► Halting Module 2 ──► p₂
  │
  ▼
Output
```

## QLoRA Block Architecture

With QLoRA, block weights are quantized and LoRA adapters are added:

```
Input
  │
  ├──► Quantized Attention (4-bit) + LoRA Adapter (16-bit)
  │
  ├──► Quantized MLP (4-bit) + LoRA Adapter (16-bit)
  │
  ▼
Output
```

## Model Sizes

| Model | Layers | Hidden | Heads | Params | Memory (FP16) | Memory (QLoRA) |
|---|---|---|---|---|---|---|
| ViT-S (CIFAR) | 12 | 128 | 4 | 1.2M | 2.4 MB | 0.6 MB |
| ViT-B (TinyImgNet) | 12 | 768 | 12 | 26M | 52 MB | 13 MB |
| DiT-S (ImageNet) | 28 | 384 | 8 | 33M | 66 MB | 16.5 MB |
| DiT-B (ImageNet) | 28 | 768 | 16 | 130M | 260 MB | 65 MB |

---

See also: [Quality Gate](Quality-Gate.md) · [Training Guide](Training-Guide.md) · [Inference Guide](Inference-Guide.md) · [Home](Home.md)
