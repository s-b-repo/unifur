# TODO & Roadmap

This file tracks the implementation progress for DiffusionBlocks++.

Status legend: `[x]` implemented & tested · `[#]` partial (notes inline) ·
`[ ]` not started / blocked on external resources (noted inline).

## Phase 1: Core DiffusionBlocks (Original Paper Port)

- [x] **1.1** Port ViT baseline (image classification)
- [x] **1.2** Port DiffusionBlocks training loop
- [x] **1.3** Port EDM noise schedule and loss
- [#] **1.4** Port data loading — CIFAR-100 binary loader done (`cifar.rs`,
      network-free, fixture-tested); Tiny ImageNet pending
- [#] **1.5** Port checkpointing and logging — content-addressed checkpoints
      (`checkpoint.rs`) + JSONL metrics logger (`logging.rs`) done;
      W&B export adapter pending
- [ ] **1.6** Reproduce CIFAR-100 results (baseline ViT) — *blocked: needs GPU hours*
- [ ] **1.7** Reproduce CIFAR-100 results (DiffusionBlocks) — *blocked: needs GPU hours*
- [ ] **1.8** Reproduce Tiny ImageNet results — *blocked: needs dataset + GPU*

**Status**: Core port complete; reproduction runs pending hardware

## Phase 2: Parallel Depth Denoising

- [x] **2.1** Parallel trajectory configuration — `MultiBlockConfig` / `Strategy`
- [x] **2.2** Parallel denoising trajectory — span execution via
      `DblockClassifier::denoise_span`
- [x] **2.3** K-block joint sampling — `Strategy::Parallel{k}` per window
- [x] **2.4** Fork-reconverge protocol — windows re-converge through the shared
      latent between spans
- [x] **2.5** Gradient routing (only active blocks) — inherent: gradients only
      flow through executed span layers
- [x] **2.6** Overlapping sigma windows — extended windows via `gamma`
- [#] **2.7** Cross-fork training (non-adjacent block pairs) — trajectory
      consistency loss trains non-adjacent blocks against chain endpoints
      (`consistency.rs`); dedicated pair sampling pending
- [x] **2.8** Integration with training loop — `consistency_step` drops into
      the standard Burn loop
- [ ] **2.9** Benchmark K=1 vs K=2 vs K=3 on CIFAR-100 — *blocked: needs GPU*
- [ ] **2.10** Document optimal hyperparameters — after 2.9

**Status**: Core mechanisms implemented; benchmarks pending hardware

## Phase 3: Consistency Training

- [x] **3.1** Boundary consistency loss (`consistency.rs`)
- [x] **3.2** Self-consistency loss (different noise levels)
- [x] **3.3** Trajectory consistency loss
- [x] **3.4** Integrated with multi-block inference (`multi_block.rs`)
- [x] **3.5** Weight scheduling (constant / linear / cosine)
- [ ] **3.6** Ablation: consistency weight sweep — *blocked: needs GPU*
- [ ] **3.7** Ablation: boundary vs self-consistency — *blocked: needs GPU*

**Status**: Implemented; ablations pending hardware

## Phase 4: Inference Solvers

- [x] **4.1** Euler solver (`solver.rs`, reference path)
- [x] **4.2** Heun solver (2nd order)
- [x] **4.3** DDIM solver (ancestral `eta`; `eta=0` ≡ Euler here, documented)
- [x] **4.4** DPM-Solver++ (2nd order multistep)
- [ ] **4.5** DPM-Solver++ (3rd order) — deferred
- [x] **4.6** Solver factory/selection (`SolverKind::parse`)
- [#] **4.7** Solver benchmark — `dblocks bench-solvers` reports time +
    prediction stats on random weights; quality-vs-steps study needs a
    trained model (GPU)
- [x] **4.8** Integrated into inference pipeline (`sample --solver`, `solve()`)

**Status**: 4 solvers shipped + benchmark command

## Phase 5: Flow Matching

- [x] **5.1** Rectified flow objective (`flow::flow_matching_loss`)
- [x] **5.2** OT path (straight-line conditional path)
- [x] **5.3** Linear sigma schedule (`t ~ U(0,1)`, `xt = (1-t)x0 + t·x1`)
- [x] **5.4** Training loop integration (`flow::train_flow_synthetic`)
- [ ] **5.5** Compare with EDM objective across tasks — *blocked: needs GPU*

**Status**: Objective + sampler + trainer shipped

## Phase 6: MoE Routing

- [x] **6.1** Top-K router (`moe::TopKRouter`, renormalized gates)
- [x] **6.2** Expert MLP pool (`ExpertPool` / `MoELayer`)
- [x] **6.3** Load balancing loss (Switch-style auxiliary)
- [x] **6.4** Noise-aware router (condition accepts concatenated noise embedding)
- [#] **6.5** Integrate MoE blocks into DiffusionBlocks trunk — standalone
      layer shipped; automatic replacement of DbLayer MLPs pending
- [ ] **6.6** Benchmark: 1/2/4/8 experts — *blocked: needs GPU*

**Status**: Layer + losses unit-tested; trunk integration pending

## Phase 7: Block Distillation

- [ ] **7.1–7.7** Not started (teacher/student loops, KL/MSE distillation,
  QLoRA students)

## Phase 8: Adaptive Depth

- [x] **8.1** Halting probability module (`adaptive::HaltingHead`)
- [x] **8.2** Adaptive depth controller — confidence-driven span widening in
      `Strategy::Adaptive`
- [x] **8.3** Depth regularization (expected-depth / ponder cost)
- [x] **8.4** Block skipping policy — cumulative-probability early exit
      (`early_exit_step`)
- [#] **8.5** Inference-pipeline integration — confidence-based widening wired
      into `sample_multi_block`; halting-head-driven exits need a trained head
- [ ] **8.6** Benchmark quality vs expected depth — *blocked: needs GPU*
- [ ] **8.7** Visualize block usage patterns — needs trained runs

## Phase 9: QLoRA / Quantization

- [ ] **9.1–9.5** Not started (NF4 quantization, LoRA adapters)

## Phase 10: Multi-Block Denoising

- [x] **10.1** Sequential denoising (`Strategy::Sequential`)
- [x] **10.2** Parallel denoising K blocks (`Strategy::Parallel`)
- [x] **10.3** Hybrid strategies (`Strategy::Hybrid`)
- [x] **10.4** Adaptive dynamic K (`Strategy::Adaptive`)
- [x] **10.5** Quality-gated denoising (`Gated` + `quality.rs`)
- [ ] **10.6** Precision denoising (mixed precision) — *blocked: ndarray CPU
      backend computes in f32 only*
- [#] **10.7** Benchmark all strategies — smoke-level stats emitted by
      `sample_multi_block`; full benchmark needs trained models

## Phase 11: Hybrid Loop Graph Dynamic Transformers

- [ ] **11.1–11.5** Not started

## Phase 12: Quality Gate

- [x] **12.1** MSE check (`quality::evaluate`)
- [x] **12.2** Cosine similarity check
- [x] **12.3** Confidence threshold
- [x] **12.4** Gradient norm check (`grad_norm_ok`)
- [#] **12.5** Per-layer quality gates — batch-level gates shipped; per-layer
      granularity pending
- [x] **12.6** Batch filtering for bad samples (`filter_indices`, gate masks
      keep previous latents during sampling)
- [ ] **12.7** Ablation: thresholds — *blocked: needs GPU*

## Phase 13: I/O Uring & Performance

- [ ] **13.1** io_uring data loading — N/A until real dataloaders exist
      (synthetic data is generated on-device); revisit with 1.4 Tiny ImageNet
- [x] **13.2** Async checkpoint saving
      (`checkpoint::save_content_addressed_async`, background thread)
- [ ] **13.3** Batched syscall optimization — no evidence of syscall-bound
      paths; profile first
- [ ] **13.4** Zero-copy data transfer — pending real dataloader
- [#] **13.5** Performance profiler — op-level timing harness used during
      development (removed); `bench-solvers` covers inference timing
- [ ] **13.6** io_uring benchmark — depends on 13.1

## Phase 14: Mathematical Foundation

- [#] **14.1–14.5** Formalized where load-bearing: solver discretizations and
  their consistency are documented/tested against closed-form solutions
  (`solver.rs` tests), preconditioning identity verified in `sigma.rs` tests,
  consistency-loss semantics documented in `consistency.rs`. Full formal
  proofs remain documentation work.
- [x] **14.6** Numerical verification — scipy-matched schedule values,
  constant-oracle ODE convergence, roundtrip checkpoint equality

## Phase 15: Production Features

- [ ] **15.1** Mixed precision (bf16) — *blocked: ndarray CPU backend is f32*
- [x] **15.2** Gradient checkpointing
      (`CheckpointedTrainBackend` / `--grad-checkpointing`)
- [ ] **15.3** Distributed training (DDP) — *needs cluster hardware*
- [ ] **15.4** DeepSpeed — *N/A outside PyTorch; equivalent unimplemented*
- [ ] **15.5** FSDP — *needs cluster hardware*
- [#] **15.6** W&B logging — local JSONL schema is W&B-compatible
      (`logging.rs`); hosted-W&B client intentionally excluded
      (network-free crate)
- [x] **15.7** Model checkpointing save (`--out-dir` content-addressed);
      resume/load CLI flag still open
- [ ] **15.8** Inference API / serving — CLI covers offline inference

## Phase 16: Tests & Documentation

- [x] **16.1** Unit tests for model components (vit, dblock, sigma, stats)
- [#] **16.2** Unit tests for parallel trajectory (strategy spans covered in
  integration tests)
- [x] **16.3** Unit tests for solvers (closed-form oracle convergence)
- [x] **16.4** Unit/integration tests for flow matching
- [x] **16.5** Unit tests for MoE (exactness of top-1 routing, balance loss)
- [x] **16.6** Unit tests for adaptive depth (exit boundaries, ponder range)
- [#] **16.7** Multi-block denoising — strategy matrix covered in integration
  tests; property tests pending
- [x] **16.8** Unit tests for quality gate
- [ ] **16.9** Unit tests for hybrid loop graph — phase not started
- [x] **16.10** Integration test for training (`tests/integration.rs`)
- [x] **16.11** Integration test for inference (same file)
- [#] **16.12** Wiki docs — module-level rustdoc comprehensive; wiki sync open
- [x] **16.13** Example scripts — `dblocks` subcommands double as examples

## Phase 17: Applications & Demos

- [ ] **17.1–17.4** Gradio/Colab/blog/video — out of scope for this Rust
  repository

## Completed Summary

- Phase 1.1–1.5: ViT-DiT backbone, block-wise training loop, EDM schedules &
  loss, CIFAR-100 binary loader, content-addressed checkpointing, JSONL metrics.
- Phase 2/10: sequential / parallel / hybrid / adaptive / gated multi-block
  sampling with span execution (= gradient routing).
- Phase 3: three consistency objectives + weight scheduling.
- Phase 4: Euler / Heun / DDIM / DPM++2M solvers + factory + benchmark command.
- Phase 5: rectified-flow objective, OT path, trainer, velocity-field sampler.
- Phase 6: top-k MoE layer, expert pool, Switch balance loss.
- Phase 8: halting head, expected-depth regularizer, early-exit logic.
- Phase 12: MSE / cosine / confidence gates, grad-norm check, batch filtering.
- Phase 13.2: async checkpoint saves. Phase 15.2: gradient checkpointing.
  Phase 15.6: W&B-compatible JSONL logging.

Test inventory: 41 unit + 4 integration tests, all passing; clippy clean.

## Remaining (requires external resources)

| Item | Blocker |
|---|---|
| 1.6–1.8 result reproduction | GPU compute |
| 2.9–2.10, 3.6–3.7, 5.5, 6.6, 8.6–8.7, 10.7, 12.7 ablations/benchmarks | Trained models (GPU) |
| 4.5 DPM++3M | Straightforward once 4.4 validated on quality runs |
| 6.5 MoE-in-trunk wiring | Design decision (which MLPs become experts) |
| 7.x distillation, 9.x QLoRA, 11.x loop graphs | Feature work, no blocker besides priority |
| 13.1/13.4/13.6 io_uring | Real dataloader dependency |
| 15.1 bf16 | Backend support (ndarray = f32) |
| 15.3/15.5 distributed | Cluster hardware |
| 15.8 serving | Product decision |
| 17.x demos | Different artifact type (web/notebook/video) |

## How to Contribute

1. Pick an unchecked item from above
2. Create an issue referencing the item number (e.g., "Implement 2.1")
3. Fork the repo, implement, and submit a PR
4. Update this file when your PR is merged
