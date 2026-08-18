# DiffusionBlocks++  —  Documentation & Roadmap

<div align="center">

```
  ____  _  __  _          _               _      _           _
 |  _ \(_)/ _|(_)___     / \   _ __   ___| |__ (_)_ __  ___(_)___
 | | | | | |_ | / __|   / _ \ | '_ \ / __| '_ \| | '_ \/ __| / __|
 | |_| | |  _|| \__ \  / ___ \| | | | (__| | | | | |_) \__ \ \__ \
 |____/|_|_|(_)|_|___/ /_/   \_\_| |_|\___|_| |_|_| .__/|___/_|___/
                                                   |_|
```

</div>

> **DiffusionBlocks++**: Block-wise neural network training with parallel depth
> denoising trajectories, consistency regularization, flow-matching objectives,
> mixture-of-experts routing, and block distillation — matching end-to-end
> backpropagation while training multiple blocks simultaneously.

Based on **[DiffusionBlocks: Block-wise Neural Network Training via Diffusion
Interpretation](https://arxiv.org/abs/2506.14202)** (Shing, Koyama, Akiba — ICLR 2026).
Original code: [SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Idea](#core-idea)
3. [Architecture](#architecture)
4. [Parallel Depth Denoising Trajectory](#parallel-depth-denoising-trajectory)
33|5. [Feature Ports & Extensions](#feature-ports--extensions)
34|   - [Multi-Block Denoising](#multi-block-denoising)
35|   - [Parallel Depth Denoising](#parallel-depth-denoising)
36|   - [Hybrid Denoising](#hybrid-denoising)
37|   - [Adaptive Denoising](#adaptive-denoising)
38|   - [Quality-Gated Denoising](#quality-gated-denoising)
39|   - [Precision Denoising](#precision-denoising)
40|   - [Hybrid Loop Graph](#hybrid-loop-graph-dynamic-transformers)
41|   - [Consistency Training](#consistency-training)
42|   - [Flow Matching Objective](#flow-matching-objective)
43|   - [DPM-Solver++ / DDIM Inference](#dpm-solver--ddim-inference)
44|   - [Mixture-of-Experts Routing](#mixture-of-experts-routing)
45|   - [Block Distillation](#block-distillation)
46|   - [Adaptive Depth / Block Skipping](#adaptive-depth--block-skipping)
47|   - [QLoRA / 4-bit Block Training](#qlora--4-bit-block-training)
48|   - [Multi-Task Denoising Heads](#multi-task-denoising-heads)
49|   - [I/O Uring Support](#io-uring-support)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [Training Guide](#training-guide)
10. [Inference Guide](#inference-guide)
11. [Configuration Reference](#configuration-reference)
12. [Roadmap & TODO](#roadmap--todo)
13. [Contributing](#contributing)
14. [Citation](#citation)
15. [License](#license)

---

## Project Overview

DiffusionBlocks++ extends the original DiffusionBlocks framework with parallel
training of multiple blocks, advanced denoising objectives, and production-ready
features ported from other state-of-the-art projects.

**Why?** End-to-end backpropagation requires storing activations for all layers.
DiffusionBlocks partitions a transformer into independently trainable blocks.
DiffusionBlocks++ makes this practical by:

- Training **multiple blocks at once** via parallel depth trajectories
- **Consistency losses** so blocks agree at their boundaries
- **High-order ODE solvers** (DDIM, DPM-Solver++) for fast inference
- **Flow matching** as an alternative to EDM score matching
- **MoE routing** so blocks specialize per noise regime
- **Block distillation** to compress N blocks → fewer at inference
- **Adaptive depth** to skip blocks when confident
- **QLoRA** for memory-efficient block training

---

## Core Idea

A residual network computes:

```
h_{l+1} = h_l + f_θ_l(h_l)
```

This is a discretized ODE: `dh/dt = f_θ(h)`. DiffusionBlocks observes that
the reverse process of a diffusion model is also an ODE, so each block can
learn its own denoising objective independently:

```
Block i:  learns to denoise from σ_i → σ_{i+1}
```

Blocks are trained independently → memory is O(1) instead of O(L).

**DiffusionBlocks++ extends this**: instead of training one block per step,
train K blocks simultaneously on overlapping noise windows. The overlapping
regions act as a soft consensus mechanism.

---

## Architecture

```
                         ┌─────────────────────────────┐
                         │     Input (pixel_values)     │
                         └──────────────┬──────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  Patch Embedding    │
                              │  (+ Position Emb)   │
                              └─────────┬──────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
     ┌────────▼────────┐       ┌────────▼────────┐       ┌────────▼────────┐
     │   Block 0       │       │   Block 1       │       │   Block 2       │
     │   (σ_max → σ_a) │       │   (σ_a → σ_b)   │       │   (σ_b → σ_min) │
     │                 │       │                 │       │                 │
     │  ┌───────────┐  │       │  ┌───────────┐  │       │  ┌───────────┐  │
     │  │ AdaLN(zt) │  │       │  │ AdaLN(zt) │  │       │  │ AdaLN(zt) │  │
     │  └─────┬─────┘  │       │  └─────┬─────┘  │       │  └─────┬─────┘  │
     │  ┌─────▼─────┐  │       │  ┌─────▼─────┐  │       │  ┌─────▼─────┐  │
     │  │ Attention │  │       │  │ Attention │  │       │  │ Attention │  │
     │  └─────┬─────┘  │       │  └─────┬─────┘  │       │  └─────┬─────┘  │
     │  ┌─────▼─────┐  │       │  ┌─────▼─────┐  │       │  ┌─────▼─────┐  │
     │  │   MLP     │  │       │  │   MLP     │  │       │  │   MLP     │  │
     │  └─────┬─────┘  │       │  └─────┬─────┘  │       │  └─────┬─────┘  │
     └────────┼────────┘       └────────┼────────┘       └────────┼────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   Output Head      │
                              │  (classifier)      │
                              └───────────────────┘
```

---

## Parallel Depth Denoising Trajectory

### The Problem with Single-Block Training

Original DiffusionBlocks trains **one block per step**. With B blocks, you
need B× more steps than end-to-end training. Each block sees only 1/B of
the gradient signal.

### The Parallel Solution

Train **K adjacent blocks simultaneously** on overlapping noise windows.
The denoising trajectory forks into parallel branches that reconverge.

```
Standard (B=3 blocks, K=1 at a time):
  t=0   Block0 (σ_max → σ_a)
  t=1   Block1 (σ_a → σ_b)
  t=2   Block2 (σ_b → σ_min)

Parallel (K=2 blocks at a time):
  t=0   Block0 + Block1   (σ_max → σ_a, σ_a → σ_b, overlap on σ_a)
  t=1   Block1 + Block2   (σ_a → σ_b, σ_b → σ_min, overlap on σ_b)
  t=2   Block0 + Block2   (σ_max → σ_b, σ_b → σ_min, cross-fork)
```

### Fork-Reconverge Protocol

```
FORK:
  z → noise(σ_max) → zt_max
                       ├─► Block0 branch: zt_max at σ_max
                       └─► Block1 branch: noise(σ_a) → zt_a

DENOISE (parallel):
  Block0: zt_max → denoise(σ_max → σ_a) → output_0
  Block1: zt_a   → denoise(σ_a → σ_b)   → output_1

RECONVERGE:
  At boundary σ_a: consistency_loss(output_0, output_1)
  Combined: L_total = L_block0 + L_block1 + λ · L_consistency
```

### Gradient Routing

- Gradients flow **only** through the active block pair
- Inactive blocks are frozen during the step
- Overlapping sigma regions create a soft consensus signal
- Each block gets K× more gradient signal than single-block training

### Benefits

| Metric | Original | Parallel (K=2) | Parallel (K=3) |
|---|---|---|---|
| Steps to cover all blocks | B | ⌈B/(K-1)⌉ | ⌈B/(K-1)⌉ |
| Gradient signal per block | 1/B | K/B | K/B |
| Inter-block communication | None | Via overlap | Via overlap |
| Memory | O(1) | O(K) | O(K) |

---

## Feature Ports & Extensions

### Mathematical Foundation

Rigorous mathematical treatment of multi-micro-block denoising:

- **Theorem 1**: DiffusionBlocks ODE Equivalence — block-wise dynamics as reverse ODE
- **Theorem 2**: Tweedie's Formula — optimal denoiser via score function
- **Theorem 3**: Convergence to Lossless — score matching → optimal denoiser
- **Theorem 4**: Compositional Losslessness — micro-block losslessness adds up
- **Theorem 5**: Block-wise Convergence — error propagation bounded
- **Theorem 6**: Parallel Training Convergence — K× speedup with K blocks
- **Theorem 7**: Consistency Loss Bound — adjacent block agreement

See [docs/Mathematical-Foundation.md](docs/Mathematical-Foundation.md) for full proofs.

DiffusionBlocks++ supports multiple denoising strategies that can be combined:

| Strategy | Description | Config |
|---|---|---|
| **Sequential** | One block at a time (original) | `--strategy sequential` |
| **Parallel** | K blocks simultaneously | `--strategy parallel --parallel_k 2` |
| **Hybrid** | Mix based on noise regime/phase | `--strategy hybrid` |
| **Adaptive** | Dynamic block selection | `--strategy adaptive` |
| **Quality-Gated** | Per-step quality checks | `--quality_gate` |

```bash
# Sequential (original)
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --strategy sequential

# Parallel (K=2 blocks at a time)
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --strategy parallel --parallel_k 2

# Hybrid (sequential early, parallel late)
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --strategy hybrid --parallel_k 2

# Adaptive (dynamic K based on loss)
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --strategy adaptive --max_parallel_k 3

# With quality gating
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --strategy parallel --parallel_k 2 \
    --quality_gate --mse_threshold 0.1 --cos_threshold 0.9
```

### Parallel Depth Denoising

Train K adjacent blocks simultaneously on overlapping noise windows. The
denoising trajectory forks into parallel branches that reconverge.

```
Standard (K=1):     Block0 → Block1 → Block2
Parallel (K=2):     Block0 + Block1 → Block1 + Block2
Parallel (K=3):     Block0 + Block1 + Block2
```

See [docs/Parallel-Denoising.md](docs/Parallel-Denoising.md) for details.

### Hybrid Denoising

Combines sequential and parallel strategies based on training phase or
noise regime:

```python
# Phase-based: sequential during warmup, parallel after
if epoch < warmup_epochs:
    train_sequential()
else:
    train_parallel(K=2)

# Noise-regime based: parallel at high noise, sequential at low noise
if sigma > 10.0:
    train_parallel(K=3)
else:
    train_sequential()
```

### Adaptive Denoising

Dynamically adjusts the number of blocks and training strategy based on
model confidence:

```python
# Increase K when loss plateaus
if loss_plateau:
    K = min(max_K, K + 1)
# Decrease K when loss is unstable
elif loss_unstable:
    K = max(1, K - 1)
```

### Quality-Gated Denoising

Per-step quality checks prevent bad denoising from corrupting training:

| Check | Threshold | Action |
|---|---|---|
| MSE | < 0.1 × σ | Reject if exceeded |
| Cosine Similarity | > 0.9 | Reject if below |
| Confidence | > 0.5 | Reject if below |
| Gradient Norm | < 10.0 | Clip if exceeded |

```python
# Bad denoising detected → use previous output or skip update
if not quality_gate.check(output, target, sigma):
    output = previous_output  # Fallback
```

### Precision Denoising

Use different numerical precision for different blocks or noise regimes:

| Noise Regime | Precision | Reason |
|---|---|---|
| High noise (σ > 10) | FP32 | Large gradients need precision |
| Medium noise (1 < σ < 10) | BF16 | Balanced precision/memory |
| Low noise (σ < 1) | FP16 | Fine refinement, less precision needed |
| Inference | FP16/INT8 | Maximum speed |

```bash
# Enable precision denoising
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --precision_strategy dynamic
```

### Hybrid Loop Graph Dynamic Transformers

Dynamic computation graph that adapts to what needs to be done:

- **Adaptive Block Selection**: Only run blocks that need training
- **Skip Connections**: Loop graph with skip connections for gradient flow
- **Confidence-Based Early Exit**: Skip remaining blocks when confident
- **Dynamic K**: Adjust parallelism per layer/sample

```bash
# Enable hybrid loop graph
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --hybrid_loop_graph --max_iterations 10
```

### I/O Uring Support

Linux io_uring for high-performance async I/O:

- **Async Data Loading**: Non-blocking file reads
- **Async Checkpointing**: Non-blocking checkpoint saves
- **Batched Syscalls**: Reduce syscall overhead
- **Zero-Copy**: Direct data transfer

```bash
# Enable io_uring
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --io_uring --queue_depth 128
```

See [docs/Multi-Block-Denoising.md](docs/Multi-Block-Denoising.md) for full details
on all multi-block denoising strategies, quality gates, precision denoising,
and hybrid loop graph dynamic transformers.

### Consistency Training

**Source**: [Consistency Models](https://arxiv.org/abs/2303.01469) (Song et al., 2023)

**Idea**: Adjacent blocks should produce consistent predictions at their
boundary sigma. Add a self-consistency loss:

```
L_consistency = E[ || denoise(x, σ_b, Block_i) - denoise(x, σ_b, Block_{i+1}) ||² ]
```

**Benefits**:
- Faster convergence (blocks cooperate instead of competing)
- Fewer inference steps needed
- Smoother denoising trajectory

**Config**: `--consistency_loss --consistency_weight 0.1`

---

### Flow Matching Objective

**Source**: [Flow Matching](https://arxiv.org/abs/2210.02747) (Lipman et al., 2022) / [Rectified Flow](https://arxiv.org/abs/2209.03003) (Liu et al., 2022)

**Idea**: Replace EDM's score-matching objective with rectified flow
(straight paths from noise to data). Useful when you want ODE-based
inference without the complex noise schedule.

```
Rectified flow:
  zt = (1 - t) * z + t * noise
  target_velocity = noise - z
  loss = E[ || v(zt, t) - (noise - z) ||² ]
```

**Benefits**:
- Simpler noise schedule (linear instead of log-normal)
- Straight-line paths → fewer integration steps
- Better suited for some architectures

**Config**: `--denoising_objective flow`

---

### DPM-Solver++ / DDIM Inference

**Source**: [DPM-Solver++](https://arxiv.org/abs/2211.01095) (Lu et al., 2022) / [DDIM](https://arxiv.org/abs/2010.02502) (Song et al., 2020)

**Idea**: High-order ODE solvers for accelerating inference beyond basic Euler.

| Solver | Order | Steps for good quality | Notes |
|---|---|---|---|
| Euler | 1 | 50-100 | Simplest, most evaluations |
| Heun | 2 | 20-50 | Predictor-corrector |
| DDIM | 1 (implicit) | 10-50 | Non-Markovian, fast |
| DPM-Solver++ | 2 | 5-20 | Best accuracy per step |

**Config**: `--solver dpmpp --num_inference_steps 20`

---

### Mixture-of-Experts Routing

**Source**: [MoE](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017) / [Mixtral](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024)

**Idea**: Each block has N expert sub-networks. A lightweight router selects
which expert performs the denoising for a given (σ, x) pair.

```
Input → Router → Top-K experts → Weighted sum → Output
```

**Benefits**:
- Specialization: different experts handle different noise regimes
- Capacity: more parameters without more compute per token
- Scalability: experts can be distributed across devices

**Config**: `--moe --num_experts 4 --top_k 2`

---

### Block Distillation

**Source**: [Knowledge Distillation](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015) + [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)

**Idea**: Train a large DiffusionBlocks model, then distill it into a
smaller one by minimizing the KL divergence between teacher and student
denoising trajectories.

```
Teacher (many blocks) → Student (fewer blocks)
Loss: KL(teacher_denoise(z, σ) || student_denoise(z, σ))
```

**Benefits**:
- Compress N blocks → M < N blocks for inference
- Teacher provides richer training signal than hard labels
- Can combine with QLoRA for efficient student training

**Config**: `--distill --teacher_path logs/teacher/last.ckpt`

---

### Adaptive Depth / Block Skipping

**Source**: [PonderNet](https://arxiv.org/abs/2107.09082) (Banino et al., 2021) / [Looped Transformers](https://arxiv.org/abs/2502.05171) (Geiping et al., 2025)

**Idea**: Each block learns a halting probability. If the model is
confident enough, subsequent blocks are skipped at inference.

```
Block 0 → halting_prob p_0
  if p_0 > threshold: STOP
  else: Block 1 → halting_prob p_1
    if p_1 > threshold: STOP
    else: Block 2 → ...
```

**Benefits**:
- Dynamic compute per sample (easy samples use fewer blocks)
- Regularization toward target depth
- Natural early-exit mechanism

**Config**: `--adaptive_depth --target_depth 2.0 --max_halting_prob 0.5`

---

### QLoRA / 4-bit Block Training

**Source**: [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)

**Idea**: Store block weights in 4-bit NormalFloat. Only LoRA adapters
are full-precision. Train 3× more blocks per GPU.

```
Block weight (4-bit) + LoRA adapter (16-bit) = trainable block
```

**Benefits**:
- 4× memory reduction for block weights
- Train more blocks on same hardware
- Minimal quality loss

**Config**: `--use_qlora --lora_rank 16 --lora_alpha 32`

---

### Multi-Task Denoising Heads

**Source**: Multi-task diffusion models

**Idea**: One backbone, multiple task-specific denoising objectives.
Each block can contribute to multiple tasks.

```
Backbone blocks → Task head A (classification)
                → Task head B (segmentation)
                → Task head C (captioning)
```

**Benefits**:
- Shared representation across tasks
- Each task gets its own denoising trajectory
- Efficient multi-task training

**Config**: `--multi_task --tasks classification,segmentation`

---

## Project Structure

```
diffusionblocks/
├── pyproject.toml              # Project config (uv / pip)
├── LICENSE                     # Apache 2.0
├── README.md                   # This file (main documentation)
├── .gitignore
├── .python-version
├── configs/                    # YAML training configs
│   ├── base.yaml               # Default DiffusionBlocks config
│   ├── parallel_denoise.yaml   # Parallel depth denoising
│   ├── flow_matching.yaml      # Flow matching objective
│   ├── consistency.yaml        # Consistency training
│   ├── moe.yaml                # Mixture-of-experts
│   ├── distillation.yaml       # Block distillation
│   └── adaptive_depth.yaml     # Adaptive depth / block skipping
├── docs/                       # GitHub Wiki pages
│   ├── Home.md                 # Wiki home
│   ├── Parallel-Denoising.md   # Deep dive on parallel trajectories
│   ├── Consistency-Training.md
│   ├── Flow-Matching.md
│   ├── MoE-Routing.md
│   ├── Block-Distillation.md
│   ├── Adaptive-Depth.md
│   ├── QLoRA.md
│   ├── Inference-Solvers.md
│   ├── Configuration.md        # Full config reference
│   ├── Training-Guide.md
│   ├── Inference-Guide.md
│   ├── Architecture.md         # Architecture details
│   └── FAQ.md
├── src/diffusionblocks/        # Source code (when implemented)
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── model.py                # Lightning modules
│   ├── vit.py                  # ViT backbone
│   ├── dblock_modules.py       # Sigma schedules
│   ├── parallel_trajectory.py  # ★ Parallel depth denoising
│   ├── consistency.py          # ★ Consistency training loss
│   ├── solvers.py              # ★ DDIM, DPM-Solver++, Heun, Euler
│   ├── flow_matching.py        # ★ Rectified flow objective
│   ├── adaptive_depth.py       # ★ Learned block skipping
│   ├── moe_routing.py          # ★ Mixture-of-experts router
│   ├── distillation.py         # ★ Block distillation
│   ├── quantization.py         # ★ QLoRA / 4-bit training
│   └── data.py                 # LightningDataModules
├── tests/                      # Test suite
│   ├── test_model.py
│   ├── test_parallel.py
│   ├── test_solvers.py
│   ├── test_flow.py
│   ├── test_moe.py
│   ├── test_adaptive_depth.py
│   └── test_distillation.py
├── examples/                   # Example scripts
│   ├── train_cifar100.py
│   ├── train_parallel.py
│   ├── train_flow_matching.py
│   ├── train_consistency.py
│   ├── train_moe.py
│   ├── distill_blocks.py
│   └── run_inference.py
└── scripts/                    # Utility scripts
    ├── setup_env.sh
    ├── download_data.sh
    └── benchmark_solvers.py
```

---

## Installation

### Requirements

- Python 3.12+
- CUDA 12.2+ (for GPU training)
- uv (recommended) or pip

### Quick Install

```bash
# Clone
git clone https://github.com/unifr/diffusionblocks-plusplus.git && cd diffusionblocks-plusplus

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

# Login to HuggingFace and W&B
huggingface-cli login
wandb login
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check imports
python -c "import diffusionblocks; print(diffusionblocks.__version__)"
```

---

## Quick Start

### Train DiffusionBlocks Baseline

```bash
# Standard ViT (end-to-end)
uv run python -m diffusionblocks.main train cifar100 --model_type vit

# DiffusionBlocks (block-wise)
uv run python -m diffusionblocks.main train cifar100 --model_type dblock
```

### Train with Parallel Depth Denoising

```bash
# Train 2 blocks at a time
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --parallel_blocks 2

# Train 3 blocks at a time with consistency loss
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --parallel_blocks 3 \
    --consistency_loss --consistency_weight 0.1
```

### Train with Flow Matching

```bash
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --denoising_objective flow
```

### Train with MoE Routing

```bash
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --moe --num_experts 4 --top_k 2
```

### Train with QLoRA

```bash
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --use_qlora --lora_rank 16
```

### Evaluate

```bash
# Standard evaluation
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt

# With DPM-Solver++ (faster inference)
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --solver dpmpp --num_inference_steps 20

# With adaptive depth
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --adaptive_depth
```

---

## Training Guide

### Data Supported

| Dataset | Image Size | Classes | Config Name |
|---|---|---|---|
| CIFAR-100 | 32×32 | 100 | `cifar100` |
| Tiny ImageNet | 64×64 | 200 | `tiny-imagenet` |
| ImageNet-1K | 256×256 | 1000 | `imagenet` (DiT) |
| text8 | - | 256 (vocab) | `text8` (Masked Diffusion) |
| OpenWebText | - | 50257 (vocab) | `owt` (AR Transformer) |

### Training Configs

**Base DiffusionBlocks**:
```yaml
# configs/base.yaml
model_type: dblock
num_blocks: 3
batch_size: 128
num_epochs: 1500  # 500 * num_blocks
lr: 0.001
scheduler: constant_with_warmup
```

**Parallel Denoising**:
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

**Flow Matching**:
```yaml
# configs/flow_matching.yaml
model_type: dblock
denoising_objective: flow
flow_type: rectified
num_inference_steps: 50
```

**MoE**:
```yaml
# configs/moe.yaml
model_type: dblock
moe: true
num_experts: 4
top_k: 2
noise_std: 0.1
```

### Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--num_blocks` | 3 | Number of blocks to partition the network into |
| `--parallel_blocks` | 1 | Number of blocks to train simultaneously (1 = original) |
| `--gamma` | 0.05 | Sigma range extension factor |
| `--lr` | 0.001 | Learning rate |
| `--batch_size` | 128 | Training batch size |
| `--num_epochs` | 500 | Base epoch count (× num_blocks for dblock) |
| `--weight_decay` | 0.01 | Weight decay |
| `--scheduler_type` | constant_with_warmup | LR scheduler |
| `--cfg_scale` | 0.0 | Classifier-free guidance scale |
| `--class_dropout_prob` | 0.0 | Class dropout for CFG |

### Distributed Training

```bash
# Multi-GPU with DDP
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --devices 4

# With DeepSpeed
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --devices 4 --deepspeed
```

---

## Inference Guide

### Solver Selection

| Use Case | Recommended Solver | Steps |
|---|---|---|
| Maximum quality | DPM-Solver++ | 20-30 |
| Balanced | Heun / DDIM | 10-20 |
| Fastest | DDIM | 5-10 |
| Debugging | Euler | 50-100 |

### Classifier-Free Guidance

```bash
# Enable CFG (improves quality at cost of diversity)
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --cfg_scale 3.0 --class_dropout_prob 0.1
```

### Adaptive Depth at Inference

```bash
# Enable learned block skipping
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --adaptive_depth --halting_threshold 0.9
```

---

## Configuration Reference

### Full Config Example

```yaml
# configs/full_example.yaml
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

# Multi-block denoising
strategy: hybrid           # sequential, parallel, hybrid, adaptive
max_parallel_k: 3
cross_fork: true

# Quality gate
quality_gate: true
mse_threshold: 0.1
cos_threshold: 0.9
min_confidence: 0.5
max_grad_norm: 10.0

# Precision denoising
precision_strategy: dynamic  # fp32, fp16, bf16, dynamic
high_noise_dtype: fp32
low_noise_dtype: bf16

# Hybrid loop graph
hybrid_loop_graph: true
max_iterations: 10
confidence_exit_threshold: 0.9

# I/O uring
io_uring: true
queue_depth: 128
async_checkpoint: true

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

### CLI Flags

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

---

## Roadmap & TODO

### Phase 1: Core DiffusionBlocks (Original Paper)
- [ ] Port ViT baseline (image classification)
- [ ] Port DiffusionBlocks training loop
- [ ] Port EDM noise schedule and loss
- [ ] Reproduce CIFAR-100 results
- [ ] Reproduce Tiny ImageNet results

### Phase 2: Parallel Depth Denoising
- [ ] Implement parallel trajectory sampling (K blocks at a time)
- [ ] Implement fork-reconverge protocol
- [ ] Implement gradient routing (only active blocks)
- [ ] Overlapping sigma windows with consensus
- [ ] Cross-fork training (non-adjacent block pairs)
- [ ] Benchmark: K=1 vs K=2 vs K=3 on CIFAR-100

### Phase 3: Consistency Training
- [ ] Implement boundary consistency loss
- [ ] Implement self-consistency loss (different noise levels)
- [ ] Integrate with parallel denoising
- [ ] Ablation: consistency weight sweep

### Phase 4: Inference Solvers
- [ ] Euler solver
- [ ] Heun solver (2nd order)
- [ ] DDIM solver
- [ ] DPM-Solver++ (2nd order)
- [ ] DPM-Solver++ (3rd order, stretch goal)
- [ ] Solver benchmark: quality vs steps vs time

### Phase 5: Flow Matching
- [ ] Rectified flow objective
- [ ] OT flow objective
- [ ] Linear sigma schedule
- [ ] Compare with EDM objective across tasks

### Phase 6: MoE Routing
- [ ] Top-K router
- [ ] Expert MLP pool
- [ ] Load balancing loss
- [ ] Noise-aware router (conditions on sigma)
- [ ] Benchmark: 1 vs 2 vs 4 vs 8 experts

### Phase 7: Block Distillation
- [ ] Teacher-student training loop
- [ ] KL divergence on denoising trajectory
- [ ] MSE on denoising trajectory
- [ ] Distill N blocks → M blocks
- [ ] Combine with QLoRA

### Phase 8: Adaptive Depth
- [ ] Halting probability module
- [ ] Adaptive depth controller
- [ ] Depth regularization
- [ ] Block skipping policy
- [ ] Benchmark: quality vs expected depth

### Phase 9: QLoRA / Quantization
- [ ] 4-bit NormalFloat block weights
- [ ] LoRA adapters for each block
- [ ] Integration with block-wise training
- [ ] Memory benchmark

### Phase 10: Architecture Ports
- [ ] DiT (image generation, ImageNet 256)
- [ ] Masked Diffusion (text, text8)
- [ ] Autoregressive Transformer (OWT)
- [ ] Recurrent-depth Transformer (single-pass training)

### Phase 11: Production
- [ ] Mixed precision (bf16) training
- [ ] Gradient checkpointing
- [ ] Distributed training (DDP, DeepSpeed, FSDP)
- [ ] W&B logging integration
- [ ] Model checkpointing
- [ ] Inference API / serving

### Phase 12: Applications & Demos
- [ ] Gradio demo for interactive exploration
- [ ] Colab notebook for quick start
- [ ] Blog post explaining parallel denoising
- [ ] Video tutorial

---

## Contributing

### How to Contribute

1. **Pick an item** from the [Roadmap & TODO](#roadmap--todo) above
2. **Create an issue** describing your planned implementation
3. **Fork the repo** and create a feature branch
4. **Implement** with tests
5. **Submit a PR** with:
   - Clear description of what was implemented
   - Test results
   - Updated documentation

### Development Setup

```bash
# Install dev dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run linter
ruff check src/

# Run type checking
mypy src/diffusionblocks/
```

### Code Style

- Python 3.12+ type hints
- Ruff for formatting and linting
- Max line length: 100
- Docstrings: Google style

---

## Citation

If you use DiffusionBlocks++, please cite the original paper:

```bibtex
@inproceedings{shing2026diffusionblocks,
  title     = {DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation},
  author    = {Makoto Shing and Masanori Koyama and Takuya Akiba},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=pwVSmK71cS}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## See Also

- [docs/Parallel-Denoising.md](docs/Parallel-Denoising.md) — Deep dive on parallel depth denoising
- [docs/Consistency-Training.md](docs/Consistency-Training.md) — Consistency training details
- [docs/Flow-Matching.md](docs/Flow-Matching.md) — Flow matching objective
- [docs/MoE-Routing.md](docs/MoE-Routing.md) — Mixture-of-experts routing
- [docs/Block-Distillation.md](docs/Block-Distillation.md) — Block distillation
- [docs/Adaptive-Depth.md](docs/Adaptive-Depth.md) — Adaptive depth / block skipping
- [docs/QLoRA.md](docs/QLoRA.md) — QLoRA / 4-bit training
- [docs/Inference-Solvers.md](docs/Inference-Solvers.md) — Inference solvers reference
- [docs/Configuration.md](docs/Configuration.md) — Full configuration reference
- [docs/FAQ.md](docs/FAQ.md) — Frequently asked questions
