# QLoRA / 4-bit Block Training

Memory-efficient block training using 4-bit NormalFloat quantization with
LoRA adapters.

## Overview

Training many blocks simultaneously can be memory-intensive. QLoRA reduces
the memory footprint by:

1. Storing block weights in 4-bit NormalFloat (NF4)
2. Using LoRA adapters for trainable parameters
3. Only the adapters are stored in full precision (16-bit)

This enables training 3× more blocks per GPU.

## Quantization

### 4-bit NormalFloat

NF4 is a quantization scheme optimized for normally distributed weights:

1. Normalize weights to [-1, 1]
2. Quantize to 16 levels using non-uniform spacing (denser near 0)
3. Store as 4-bit integers

```python
# Pseudo-code for NF4 quantization
def quantize_nf4(weight):
    # Normalize
    weight_norm = weight / weight.abs().max()
    
    # NF4 lookup table (16 levels, optimized for normal distribution)
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    
    # Quantize to nearest level
    indices = (weight_norm.unsqueeze(-1) - nf4_levels).abs().argmin(dim=-1)
    return indices

def dequantize_nf4(indices):
    nf4_levels = ...  # Same lookup table
    return nf4_levels[indices]
```

### Double Quantization

To further save memory, the quantization constants themselves can be quantized:

```python
# First quantization: weights → 4-bit
quantized_weights, quantize_stats = quantize_nf4(weights)

# Second quantization: quantize_stats → 8-bit
quantized_stats = quantize_int8(quantize_stats)
```

## LoRA Adapters

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to the frozen
quantized weights:

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
    
    def forward(self, x):
        # x @ (W_quantized + lora_A @ lora_B * scaling)
        return x @ self.lora_A @ self.lora_B * self.scaling
```

## Integration with DiffusionBlocks

Each block can be wrapped with QLoRA:

```python
class QLoRABlock(nn.Module):
    def __init__(self, block, lora_rank=16, lora_alpha=32):
        super().__init__()
        # Quantize block weights to 4-bit
        self.quantized_block = quantize_block(block)
        
        # Add LoRA adapters to attention and MLP
        self.lora_attention = LoRAAdapter(block.attention, lora_rank, lora_alpha)
        self.lora_mlp = LoRAAdapter(block.mlp, lora_rank, lora_alpha)
    
    def forward(self, x):
        # Use quantized weights + LoRA adapters
        return self.quantized_block(x) + self.lora_attention(x) + self.lora_mlp(x)
```

## Memory Savings

| Component | Full Precision (FP16) | QLoRA (NF4 + LoRA16) | Savings |
|---|---|---|---|
| Block weights | 2 bytes/param | 0.5 bytes/param | 4× |
| LoRA adapters | — | 2 bytes/param | — |
| Optimizer states | 8 bytes/param | 8 bytes/param (LoRA only) | ~4× |
| Activations | 2 bytes/param | 2 bytes/param | 1× |
| **Total per block** | ~10 bytes/param | ~3 bytes/param | **~3×** |

## Benefits

| Benefit | Description |
|---|---|
| **Memory** | 3-4× reduction in block memory |
| **More blocks** | Train more blocks on same hardware |
| **Quality** | Minimal quality loss vs full precision |
| **Flexibility** | Can enable/disable LoRA per block |

## Configuration

```bash
# Enable QLoRA
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --use_qlora --lora_rank 16 --lora_alpha 32

# With 8-bit LoRA
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --use_qlora --lora_rank 8 --lora_alpha 16
```

## References

- QLoRA (Dettmers et al., 2023)
- LoRA (Hu et al., 2021)
- bitsandbytes (Dettmers et al., 2022)
