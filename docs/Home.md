# DiffusionBlocks++ Wiki

Welcome to the DiffusionBlocks++ documentation. This project extends the original
[DiffusionBlocks](https://arxiv.org/abs/2506.14202) framework with parallel depth
denoising, consistency training, flow matching, MoE routing, and more.

## Navigation

### Core Documentation
- [Multi-Block Denoising](Multi-Block-Denoising.md) — Sequential, parallel, hybrid, adaptive, quality-gated strategies
- [Parallel Denoising](Parallel-Denoising.md) — The core innovation: training multiple blocks at once
- [Consistency Training](Consistency-Training.md) — Self-consistency losses for block cooperation
- [Flow Matching](Flow-Matching.md) — Alternative to EDM score matching
- [Inference Solvers](Inference-Solvers.md) — DDIM, DPM-Solver++, Heun, Euler

### Features
- [Multi-Block Denoising](Multi-Block-Denoising.md) — Sequential, parallel, hybrid, adaptive, quality-gated strategies
- [Hybrid Loop Graph](Hybrid-Loop-Graph.md) — Dynamic computation graph and quality gates
- [MoE Routing](MoE-Routing.md) — Mixture-of-experts for noise-regime specialization
- [Block Distillation](Block-Distillation.md) — Compress N blocks → fewer blocks
- [Adaptive Depth](Adaptive-Depth.md) — Learned block skipping at inference
- [QLoRA](QLoRA.md) — 4-bit block training with LoRA adapters

### Guides
- [Mathematical-Foundation](Mathematical-Foundation.md) — Rigorous mathematical theory
- [Precision-IO](Precision-IO.md) — Precision denoising and I/O uring
- [Configuration](Configuration.md) — Full config reference
- [Training Guide](Training-Guide.md) — Step-by-step training instructions
- [Inference Guide](Inference-Guide.md) — Inference and evaluation
- [Architecture](Architecture.md) — Architecture details and diagrams

### Reference
- [FAQ](FAQ.md) — Frequently asked questions
- [Roadmap](/../../issues) — See GitHub Issues for current TODO
- [Contributing](/../../blob/main/README.md#contributing) — How to contribute

## Quick Links

| Resource | Link |
|---|---|
| Paper | [arxiv.org/abs/2506.14202](https://arxiv.org/abs/2506.14202) |
| Original Code | [SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks) |
| Project Page | [pub.sakana.ai/diffusionblocks](https://pub.sakana.ai/diffusionblocks/) |
| OpenReview | [openreview.net/forum?id=pwVSmK71cS](https://openreview.net/forum?id=pwVSmK71cS) |

## Key Features

| Feature | Description |
|---|---|
| **Mathematical Foundation** | Rigorous theory: multi-micro-block decomposition, lossless conditions, convergence proofs |
| **Multi-Block Denoising** | Sequential, parallel, hybrid, adaptive, quality-gated strategies |
| **Parallel Depth Denoising** | Train K blocks simultaneously on overlapping noise windows |
| **Hybrid Loop Graph** | Dynamic computation graph that adapts to what needs to be done |
| **Quality Gate** | Per-layer per-step quality checks that prevent bad denoising |
| **Precision Denoising** | Mixed precision per noise regime (FP32/BF16/FP16) |
| **Consistency Training** | Boundary consistency loss for inter-block agreement |
| **Flow Matching** | Rectified flow / OT flow as alternative objectives |
| **DPM-Solver++** | High-order ODE solver for fast inference |
| **MoE Routing** | Experts specialize in different noise regimes |
| **Block Distillation** | Teacher-student compression across blocks |
| **Adaptive Depth** | Dynamic block skipping based on confidence |
| **QLoRA** | 4-bit quantized block weights + LoRA adapters |
| **Multi-Task** | One backbone, multiple denoising heads |
| **I/O Uring** | Linux async I/O for high-performance data loading and checkpointing |

## Current Status

See [Roadmap & Todo](/../../blob/main/TODO.md) for the full breakdown.

**In Progress**:
- Parallel depth denoising (K=2)
- Consistency training loss

**Planned**:
- Flow matching objective
- DPM-Solver++ inference
- MoE routing
- Block distillation
- Adaptive depth
- QLoRA
- DiT / Masked Diffusion / AR Transformer ports
