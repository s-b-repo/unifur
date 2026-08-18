# Configuration Reference

Complete reference for all DiffusionBlocks++ configuration options.

## Config File Format

```yaml
# configs/base.yaml
stage: train
data_name: cifar100
model_type: dblock

# Training
devices: 1
num_epochs: 1500
batch_size: 128
lr: 0.001
weight_decay: 0.01
optimizer: adamw
scheduler_type: cosine_with_min_lr
num_warmup_steps: 3900
seed: 42
save_top_k: 1
save_every_n_epochs: 5
accumulate_grad_batches: 1

# DiffusionBlocks
num_blocks: 3
parallel_blocks: 2
gamma: 0.05
overlap: 0.5

# Denoising objective
denoising_objective: edm  # "edm" or "flow"
flow_type: rectified      # "rectified" or "ot"

# Consistency training
consistency_loss: true
consistency_weight: 0.1

# Inference
solver: dpmpp              # "euler", "heun", "ddim", "dpmpp"
num_inference_steps: 20
cfg_scale: 0.0
class_dropout_prob: 0.0

# MoE
moe: false
num_experts: 4
top_k: 2

# Adaptive depth
adaptive_depth: false
target_depth: 2.0
max_halting_prob: 0.5

# Distillation
distill: false
teacher_path: null

# QLoRA
use_qlora: false
lora_rank: 16
lora_alpha: 32

# Debug
debug: false
postfix: ""
```

## CLI Flags

```bash
# All CLI flags (see --help for full list)
uv run python -m diffusionblocks.main --help

# Train
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock \
    --num_blocks 3 --parallel_blocks 2 \
    --consistency_loss --consistency_weight 0.1 \
    --solver dpmpp --num_inference_steps 20

# Test
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --adaptive_depth --solver dpmpp
```

## Parameter Reference

### Training Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stage` | str | `train` | `train` or `test` |
| `data_name` | str | `cifar100` | Dataset name |
| `model_type` | str | `vit` | `vit` (baseline) or `dblock` (DiffusionBlocks) |
| `devices` | int | `1` | Number of GPUs |
| `num_epochs` | int | `500` | Base epoch count (× num_blocks for dblock) |
| `batch_size` | int | `128` | Training batch size |
| `eval_batch_size` | int | `None` | Evaluation batch size (defaults to batch_size) |
| `lr` | float | `0.001` | Learning rate |
| `weight_decay` | float | `0.01` | Weight decay |
| `optimizer` | str | `adamw` | Optimizer (only adamw supported) |
| `scheduler_type` | str | `constant_with_warmup` | LR scheduler |
| `num_warmup_steps` | int | `0` | Warmup steps |
| `accumulate_grad_batches` | int | `1` | Gradient accumulation steps |
| `gradient_checkpointing` | bool | `False` | Enable gradient checkpointing |
| `seed` | int | `42` | Random seed |
| `save_top_k` | int | `1` | Number of best checkpoints to save |
| `save_every_n_epochs` | int | `5` | Save checkpoint every N epochs |
| `num_workers` | int | `8` | DataLoader workers |
| `debug` | bool | `False` | Enable debug mode (offline W&B) |
| `postfix` | str | `""` | Postfix for experiment name |

### DiffusionBlocks Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_blocks` | int | `3` | Number of blocks to partition the network into |
| `parallel_blocks` | int | `1` | Number of blocks to train simultaneously (1 = original) |
| `gamma` | float | `0.05` | Sigma range extension factor |
| `overlap` | float | `0.5` | Overlap fraction for parallel sigma windows |
| `denoising_objective` | str | `edm` | `edm` or `flow` |
| `flow_type` | str | `rectified` | `rectified` or `ot` |

### Consistency Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `consistency_loss` | bool | `False` | Enable consistency training |
| `consistency_weight` | float | `0.1` | Weight for consistency loss |

### Inference Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver` | str | `euler` | ODE solver: `euler`, `heun`, `ddim`, `dpmpp` |
| `num_inference_steps` | int | `None` | Inference steps (defaults to num_blocks) |
| `cfg_scale` | float | `0.0` | Classifier-free guidance scale |
| `class_dropout_prob` | float | `0.0` | Class dropout probability for CFG |

### MoE Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `moe` | bool | `False` | Enable mixture-of-experts routing |
| `num_experts` | int | `4` | Number of experts per block |
| `top_k` | int | `2` | Number of experts to route to |
| `noise_aware_router` | bool | `False` | Enable noise-aware routing |

### Adaptive Depth Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adaptive_depth` | bool | `False` | Enable adaptive depth / block skipping |
| `target_depth` | float | `2.0` | Target expected depth for regularization |
| `max_halting_prob` | float | `0.5` | Maximum halting probability |
| `halting_threshold` | float | `0.5` | Halting threshold at inference |

### Distillation Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `distill` | bool | `False` | Enable block distillation |
| `teacher_path` | str | `None` | Path to teacher checkpoint |
| `distill_weight` | float | `0.5` | Weight for distillation loss |
| `task_weight` | float | `0.5` | Weight for task loss |

### QLoRA Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_qlora` | bool | `False` | Enable QLoRA / 4-bit training |
| `lora_rank` | int | `16` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA alpha (scaling factor) |
