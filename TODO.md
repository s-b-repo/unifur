# TODO & Roadmap

This file tracks the implementation progress for DiffusionBlocks++.

## Phase 1: Core DiffusionBlocks (Original Paper Port)

- [ ] **1.1** Port ViT baseline (image classification)
- [ ] **1.2** Port DiffusionBlocks training loop
- [ ] **1.3** Port EDM noise schedule and loss
- [ ] **1.4** Port data loading (CIFAR-100, Tiny ImageNet)
- [ ] **1.5** Port checkpointing and logging
- [ ] **1.6** Reproduce CIFAR-100 results (baseline ViT)
- [ ] **1.7** Reproduce CIFAR-100 results (DiffusionBlocks)
- [ ] **1.8** Reproduce Tiny ImageNet results

**Status**: Not started

## Phase 2: Parallel Depth Denoising

- [ ] **2.1** Implement `ParallelTrajectoryConfig` class
- [ ] **2.2** Implement `ParallelDenoisingTrajectory` class
- [ ] **2.3** Implement parallel trajectory sampling (K blocks at a time)
- [ ] **2.4** Implement fork-reconverge protocol
- [ ] **2.5** Implement gradient routing (only active blocks)
- [ ] **2.6** Implement overlapping sigma windows
- [ ] **2.7** Implement cross-fork training (non-adjacent block pairs)
- [ ] **2.8** Integrate with Lightning training loop
- [ ] **2.9** Benchmark: K=1 vs K=2 vs K=3 on CIFAR-100
- [ ] **2.10** Document optimal hyperparameters

**Status**: Not started

## Phase 3: Consistency Training

- [ ] **3.1** Implement boundary consistency loss
- [ ] **3.2** Implement self-consistency loss (different noise levels)
- [ ] **3.3** Implement trajectory consistency loss
- [ ] **3.4** Integrate with parallel denoising
- [ ] **3.5** Implement consistency weight scheduling
- [ ] **3.6** Ablation: consistency weight sweep
- [ ] **3.7** Ablation: boundary vs self-consistency

**Status**: Not started

## Phase 4: Inference Solvers

- [ ] **4.1** Implement Euler solver
- [ ] **4.2** Implement Heun solver (2nd order)
- [ ] **4.3** Implement DDIM solver
- [ ] **4.4** Implement DPM-Solver++ (2nd order)
- [ ] **4.5** Implement DPM-Solver++ (3rd order, stretch goal)
- [ ] **4.6** Implement solver factory/selection
- [ ] **4.7** Solver benchmark: quality vs steps vs time
- [ ] **4.8** Integrate solvers into inference pipeline

**Status**: Not started

## Phase 5: Flow Matching

- [ ] **5.1** Implement rectified flow objective
- [ ] **5.2** Implement OT flow objective
- [ ] **5.3** Implement linear sigma schedule
- [ ] **5.4** Integrate with DiffusionBlocks training loop
- [ ] **5.5** Compare with EDM objective across tasks

**Status**: Not started

## Phase 6: MoE Routing

- [ ] **6.1** Implement Top-K router
- [ ] **6.2** Implement expert MLP pool
- [ ] **6.3** Implement load balancing loss
- [ ] **6.4** Implement noise-aware router (conditions on sigma)
- [ ] **6.5** Integrate MoE blocks into DiffusionBlocks
- [ ] **6.6** Benchmark: 1 vs 2 vs 4 vs 8 experts

**Status**: Not started

## Phase 7: Block Distillation

- [ ] **7.1** Implement teacher-student training loop
- [ ] **7.2** Implement KL divergence on denoising trajectory
- [ ] **7.3** Implement MSE on denoising trajectory
- [ ] **7.4** Implement block-to-block distillation
- [ ] **7.5** Implement distill N blocks → M blocks
- [ ] **7.6** Combine with QLoRA for efficient student training
- [ ] **7.7** Benchmark: teacher vs student quality

**Status**: Not started

## Phase 8: Adaptive Depth

- [ ] **8.1** Implement halting probability module
- [ ] **8.2** Implement adaptive depth controller
- [ ] **8.3** Implement depth regularization
- [ ] **8.4** Implement block skipping policy
- [ ] **8.5** Integrate with inference pipeline
- [ ] **8.6** Benchmark: quality vs expected depth
- [ ] **8.7** Visualize block usage patterns

**Status**: Not started

## Phase 9: QLoRA / Quantization

- [ ] **9.1** Implement 4-bit NormalFloat block weights
- [ ] **9.2** Implement LoRA adapters for each block
- [ ] **9.3** Integrate with block-wise training
- [ ] **9.4** Memory benchmark: full precision vs QLoRA
- [ ] **9.5** Quality benchmark: full precision vs QLoRA

**Status**: Not started

## Phase 10: Multi-Block Denoising

- [ ] **10.1** Implement sequential denoising (original)
- [ ] **10.2** Implement parallel denoising (K blocks)
- [ ] **10.3** Implement hybrid denoising (mix strategies)
- [ ] **10.4** Implement adaptive denoise (dynamic K)
- [ ] **10.5** Implement quality-gated denoising
- [ ] **10.6** Implement precision denoising (mixed precision)
- [ ] **10.7** Benchmark all strategies

**Status**: Not started

## Phase 11: Hybrid Loop Graph Dynamic Transformers

- [ ] **11.1** Implement dynamic computation graph
- [ ] **11.2** Implement adaptive block selection
- [ ] **11.3** Implement loop graph with skip connections
- [ ] **11.4** Implement confidence-based early exit
- [ ] **11.5** Benchmark dynamic vs static graphs

**Status**: Not started

## Phase 12: Quality Gate

- [ ] **12.1** Implement MSE quality check
- [ ] **12.2** Implement cosine similarity check
- [ ] **12.3** Implement confidence threshold
- [ ] **12.4** Implement gradient norm check
- [ ] **12.5** Implement per-layer quality gates
- [ ] **12.6** Implement batch filtering for bad samples
- [ ] **12.7** Ablation: quality gate thresholds

**Status**: Not started

## Phase 13: I/O Uring & Performance

- [ ] **13.1** Implement async data loading with io_uring
- [ ] **13.2** Implement async checkpoint saving
- [ ] **13.3** Implement batched syscall optimization
- [ ] **13.4** Implement zero-copy data transfer
- [ ] **13.5** Implement performance profiler
- [ ] **13.6** Benchmark io_uring vs standard I/O

**Status**: Not started

## Phase 14: Mathematical Foundation

- [ ] **14.1** Formalize multi-micro-block decomposition
- [ ] **14.2** Prove lossless denoising conditions
- [ ] **14.3** Prove convergence bounds for parallel training
- [ ] **14.4** Derive consistency loss bounds
- [ ] **14.5** Formalize quality gate guarantees
- [ ] **14.6** Verify all proofs with numerical experiments

**Status**: Not started

- [ ] **14.1** Port DiT (image generation, ImageNet 256)
- [ ] **14.2** Port Masked Diffusion (text, text8)
- [ ] **14.3** Port Autoregressive Transformer (OWT)
- [ ] **14.4** Port Recurrent-depth Transformer (single-pass training)
- [ ] **14.5** Benchmark all architectures

**Status**: Not started

## Phase 15: Production Features

- [ ] **15.1** Mixed precision (bf16) training
- [ ] **15.2** Gradient checkpointing
- [ ] **15.3** Distributed training (DDP)
- [ ] **15.4** Distributed training (DeepSpeed)
- [ ] **15.5** Distributed training (FSDP)
- [ ] **15.6** W&B logging integration
- [ ] **15.7** Model checkpointing (save/resume)
- [ ] **15.8** Inference API / serving

**Status**: Not started

## Phase 16: Tests & Documentation

- [ ] **16.1** Unit tests for model components
- [ ] **16.2** Unit tests for parallel trajectory
- [ ] **16.3** Unit tests for solvers
- [ ] **16.4** Unit tests for flow matching
- [ ] **16.5** Unit tests for MoE
- [ ] **16.6** Unit tests for adaptive depth
- [ ] **16.7** Unit tests for multi-block denoising
- [ ] **16.8** Unit tests for quality gate
- [ ] **16.9** Unit tests for hybrid loop graph
- [ ] **16.10** Integration tests for training
- [ ] **16.11** Integration tests for inference
- [ ] **16.12** Write all wiki documentation
- [ ] **16.13** Create example scripts

**Status**: Not started

## Phase 17: Applications & Demos

- [ ] **17.1** Gradio demo for interactive exploration
- [ ] **17.2** Colab notebook for quick start
- [ ] **17.3** Blog post explaining parallel denoising
- [ ] **17.4** Video tutorial

**Status**: Not started

## Completed

None yet.

## Progress Summary

| Phase | Progress |
|---|---|
| 1. Core DiffusionBlocks | 0/8 (0%) |
| 2. Parallel Depth Denoising | 0/10 (0%) |
| 3. Consistency Training | 0/7 (0%) |
| 4. Inference Solvers | 0/8 (0%) |
| 5. Flow Matching | 0/5 (0%) |
| 6. MoE Routing | 0/6 (0%) |
| 7. Block Distillation | 0/7 (0%) |
| 8. Adaptive Depth | 0/7 (0%) |
| 9. QLoRA / Quantization | 0/5 (0%) |
| 10. Multi-Block Denoising | 0/7 (0%) |
| 11. Hybrid Loop Graph | 0/5 (0%) |
| 12. Quality Gate | 0/7 (0%) |
| 13. I/O Uring & Performance | 0/6 (0%) |
| 14. Mathematical Foundation | 0/6 (0%) |
| 15. Architecture Ports | 0/5 (0%) |
| 16. Production Features | 0/8 (0%) |
| 17. Tests & Documentation | 0/13 (0%) |
| 18. Applications & Demos | 0/4 (0%) |
| **Total** | **0/126 (0%)** |

## How to Contribute

1. Pick an unchecked item from above
2. Create an issue referencing the item number (e.g., "Implement 2.1: ParallelTrajectoryConfig")
3. Fork the repo, implement, and submit a PR
4. Update this file when your PR is merged

## Priority Order

If you're unsure where to start, here's the recommended priority:

1. **Phase 1** (Core) — Everything depends on this
2. **Phase 2** (Parallel Denoising) — The main new feature
3. **Phase 4** (Solvers) — Needed for practical inference
4. **Phase 3** (Consistency) — Complements parallel denoising
5. **Phase 10** (Multi-Block) — Core new features
6. **Phase 11-12** (Loop Graph, Quality Gate) — Advanced features
7. **Phase 13** (I/O Uring) — Performance
8. **Phase 5-9, 14** (Features) — In any order of interest
9. **Phase 15-17** (Production, Tests, Demos) — Last, after everything works
