# Training Guide

Step-by-step guide for training DiffusionBlocks++ models.

## Prerequisites

```bash
# Install dependencies
uv sync

# Login to HuggingFace and W&B
huggingface-cli login
wandb login
```

## Quick Start

### Train Baseline ViT

```bash
uv run python -m diffusionblocks.main train cifar100 --model_type vit
```

### Train DiffusionBlocks

```bash
uv run python -m diffusionblocks.main train cifar100 --model_type dblock
```

### Train with Parallel Denoising

```bash
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --parallel_blocks 2 --consistency_loss
```

## Data Preparation

### Supported Datasets

| Dataset | Image Size | Classes | Auto-download |
|---|---|---|---|
| CIFAR-100 | 32×32 | 100 | Yes |
| Tiny ImageNet | 64×64 | 200 | Yes |
| ImageNet-1K | 256×256 | 1000 | No (manual) |
| text8 | - | 256 | Yes |
| OpenWebText | - | 50257 | Yes |

### Custom Dataset

```python
# src/diffusionblocks/data.py
class CustomDataModule(ImageDataModule):
    data_name = "your-dataset-name"
    image_size = 64
    num_labels = 10
```

## Training Configurations

### Base DiffusionBlocks

```yaml
# configs/base.yaml
model_type: dblock
num_blocks: 3
batch_size: 128
num_epochs: 1500  # 500 * num_blocks
lr: 0.001
scheduler: constant_with_warmup
```

### Parallel Denoising

```yaml
# configs/parallel_denoise.yaml
model_type: dblock
num_blocks: 3
parallel_blocks: 2
consistency_loss: true
consistency_weight: 0.1
gamma: 0.05
overlap: 0.5
```

### Flow Matching

```yaml
# configs/flow_matching.yaml
model_type: dblock
denoising_objective: flow
flow_type: rectified
num_inference_steps: 50
```

### MoE

```yaml
# configs/moe.yaml
model_type: dblock
moe: true
num_experts: 4
top_k: 2
noise_std: 0.1
```

## Hyperparameter Tuning

### Learning Rate

| Model | Recommended LR | Notes |
|---|---|---|
| ViT baseline | 1e-3 | Standard for ViT-S |
| DiffusionBlocks | 1e-3 | Same as baseline |
| Parallel (K=2) | 5e-4 | Slightly lower for stability |
| Parallel (K=3) | 3e-4 | Even lower for stability |

### Batch Size

| GPU Memory | Batch Size | Gradient Accumulation |
|---|---|---|
| 8 GB | 64 | 2 |
| 16 GB | 128 | 1 |
| 24 GB | 256 | 1 |
| 40 GB+ | 512 | 1 |

### Number of Blocks

| Dataset | Recommended B | Notes |
|---|---|---|
| CIFAR-100 | 3 | Good balance |
| Tiny ImageNet | 3-4 | More capacity needed |
| ImageNet-1K | 6-8 | For DiT |

## Distributed Training

### DDP (Data Parallel)

```bash
# Single node, multiple GPUs
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --devices 4
```

### DeepSpeed

```bash
# With DeepSpeed ZeRO-2
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --devices 4 --deepspeed
```

### Multi-Node

```bash
# 2 nodes, 4 GPUs each
srun uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --devices 8
```

## Monitoring

### W&B Dashboard

Training logs are automatically sent to Weights & Biases:

- Training loss per block
- Validation accuracy
- Learning rate schedule
- Sigma distributions
- Gradient norms

### TensorBoard

```bash
# Alternative: use TensorBoard
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --logger tensorboard
```

## Checkpointing

Checkpoints are saved in `logs/<experiment-name>/`:

```
logs/
└── 2025-01-15T10-30-00-dblock/
    ├── last.ckpt              # Latest checkpoint
    ├── epoch=499-step=19500.ckpt  # Best checkpoint
    └── hparams.yaml           # Saved hyperparameters
```

### Resume Training

```bash
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock \
    --ckpt_path logs/<run>/last.ckpt
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Enable gradient checkpointing: `--gradient_checkpointing`
- Use QLoRA: `--use_qlora`
- Reduce number of parallel blocks

### Poor Convergence

- Increase warmup steps
- Try cosine scheduler: `--scheduler_type cosine_with_min_lr`
- Increase consistency weight
- Check sigma schedule visualization

### Slow Training

- Increase batch size
- Use more workers: `--num_workers 16`
- Enable mixed precision: `--precision bf16-mixed`
- Use DPM-Solver++ for inference
