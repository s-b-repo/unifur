# DiffusionBlocks++ Wiki

Block-wise neural network training via diffusion interpretation, extended with
parallel depth denoising, consistency training, flow matching, MoE routing,
block distillation, QLoRA, dynamic loop graphs, boxes of specialized micro
experts, a causal language-model path, and planners that choose the next step
instead of always taking the greedy one.

Based on [DiffusionBlocks](https://arxiv.org/abs/2506.14202) (Shing, Koyama,
Akiba).

---

## Read this first

**The implementation is a Rust crate built on [Burn](https://burn.dev).** The
Python snippets throughout this wiki are the original design specification —
they state the intent of each feature clearly and are kept for that reason, but
they do not describe files that exist. Every page now opens with an
**"In this repository"** block giving the actual module, types and CLI flags.

Two documents are authoritative about what is and is not implemented:

- [`TODO.md`](../TODO.md) — per-item status, the bugs found and fixed while
  completing the roadmap, and the specific blocker on each remaining item.
- [Quality Gate](Quality-Gate.md) — how correctness is verified, at the
  implementation, run and step level.

```bash
cargo build --release
./target/release/dblocks verify        # 73 certificates, non-zero exit on failure
./target/release/dblocks train --steps 200
./target/release/dblocks sample --planned --plan-depth 2   # plan the trajectory
./target/release/dblocks lm generate --lookahead 2         # plan the tokens
./target/release/dblocks --help
```

---

## Navigation

### Start here
- [Quality Gate](Quality-Gate.md) — certificates, training-phase checks, sampling gates
- [Training Guide](Training-Guide.md) — every objective, dataset and flag
- [Inference Guide](Inference-Guide.md) — sampling, benchmarking, the inference API
- [Configuration](Configuration.md) — full CLI and config reference

### Core mechanics
- [Multi-Block Denoising](Multi-Block-Denoising.md) — sequential, parallel, hybrid, adaptive, gated
- [Parallel Denoising](Parallel-Denoising.md) — training multiple blocks at once
- [Consistency Training](Consistency-Training.md) — boundary, self, trajectory, cross-fork
- [Inference Solvers](Inference-Solvers.md) — Euler, Heun, DDIM, DPM-Solver++ 2M & 3M
- [Flow Matching](Flow-Matching.md) — rectified flow as an alternative objective
- [Loss Reduction](Loss-Reduction.md) — schedules, EMA, uncertainty weighting, importance sampling
- [Language Modeling](Language-Modeling.md) — byte tokenizer, causal trunk, KV cache, corpora
- [Next-Step Planning](Next-Step-Planning.md) — beam search over trajectories and tokens
- [Accuracy Improvements](Accuracy-Improvements.md) — guidance, normalization, ensembling, compute scaling

### Capacity and compression
- [Mixture of Specialized Micro Experts](Mixture-of-Specialized-Micro-Experts.md) — boxes of specialists with a routable index
- [MoE Routing](MoE-Routing.md) — flat mixture-of-experts, router z-loss, balance annealing
- [Block Distillation](Block-Distillation.md) — fewer steps, same trajectory
- [QLoRA](QLoRA.md) — NF4 quantization plus low-rank adapters
- [Adaptive Depth](Adaptive-Depth.md) — halting, early exit, span widening
- [Hybrid Loop Graph](Hybrid-Loop-Graph.md) — skip, loop back, budget

### Reference
- [Mathematical Foundation](Mathematical-Foundation.md) — the theory, and which parts are certified
- [Architecture](Architecture.md) — ViT-DiT backbone and block partitioning
- [Precision & I/O](Precision-IO.md) — mixed precision, streaming reads, profiling
- [FAQ](FAQ.md)

---

## Module map

| Concern | Module |
|---|---|
| ViT-DiT backbone, adaLN-zero, MoE placement | `vit.rs` |
| Block-wise denoiser, EDM preconditioning | `dblock.rs`, `sigma.rs` |
| ODE solvers | `solver.rs` |
| Sampling strategies | `multi_block.rs` |
| Next-step and path prediction | `planner.rs` |
| Post-training accuracy techniques | `accuracy.rs` |
| Causal language model, tokenizer, corpora | `lm.rs`, `tokenizer.rs`, `corpus.rs` |
| LR schedules, EMA, accumulation, clipping | `schedule.rs` |
| Per-sigma uncertainty weighting, importance sampling | `reweight.rs` |
| Consistency and cross-fork objectives | `consistency.rs` |
| Flow matching | `flow.rs` |
| Flat mixture-of-experts | `moe.rs` |
| Boxes of specialized micro experts | `mosme.rs`, `expert_index.rs` |
| Block distillation | `distill.rs` |
| NF4 quantization + LoRA | `quantize.rs` |
| Adaptive depth, loop graph | `adaptive.rs`, `loopgraph.rs` |
| Quality gates — sampling *and* training | `quality.rs` |
| Mixed-precision emulation | `precision.rs` |
| Datasets and streaming I/O | `data.rs`, `rawdata.rs`, `cifar.rs`, `tinyimagenet.rs` |
| Training loop, checkpoints, logging | `train.rs`, `checkpoint.rs`, `logging.rs` |
| Inference API, profiler | `infer.rs`, `profile.rs` |
| Numerical certificate suite | `verify.rs` |

## Deliberate omissions

Four things this wiki asks for are intentionally **not** implemented, each for
a stated reason rather than for lack of time:

- **Hosted W&B and HTTP serving** would pull in a network dependency tree. The
  JSONL metrics schema is W&B-compatible and `infer::InferenceEngine` exposes
  everything a server would call.
- **Native `io_uring`** would require an external crate. The measurable goal —
  fewer syscalls per batch — is delivered by positional reads and run
  coalescing in `rawdata.rs`, with the syscall count exposed for measurement.
- **Native bf16 arithmetic** needs backend support the `ndarray` backend does
  not have. `precision.rs` emulates the format exactly, so the accuracy
  question can be studied today; it is not a speedup, and says so.
- **Self-conditioning** (roadmap 22.2) only helps a model *trained* with it;
  added at sampling time it degrades results. Doing it properly needs a new
  projection in `vit.rs`, which adds a parameter to the module record and so
  cannot load an existing checkpoint. That cost is why it is not shipped
  half-done — see the Phase 22 note in `TODO.md`.

One dependency was added deliberately, against the crate's dependency-light
policy: **`serde_json`**, for the expert index. It was already in `Cargo.lock`
transitively, and the index has to be *parsed* by an inference engine, not
merely emitted — hand-rolled parsing would be the worse trade.
