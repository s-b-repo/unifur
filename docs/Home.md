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
./target/release/dblocks verify        # 227 certificates, non-zero exit on failure
./target/release/dblocks cheat --root .  # attempts to defeat the gate, not satisfy it
./target/release/dblocks train --steps 200
./target/release/dblocks sample --planned --plan-depth 2   # plan the trajectory
./target/release/dblocks lm generate --lookahead 2         # plan the tokens
./target/release/dblocks lm train --corpus repo.bin --penalty 1   # unlearn labeled anti-patterns
./target/release/dblocks sweep --grid "lr=1e-4,1e-3 num_blocks=2,3" --json sweep.jsonl
./target/release/dblocks audit propagation --checkpoint checkpoints/dblocks-<hash>.mpk
./target/release/dblocks --help
```

---

## Navigation

### Start here
- [Quality Gate](Quality-Gate.md) — certificates, training-phase checks, sampling gates
- [Reward Integrity](#reward-integrity-cheat--rewards-integrity) — what happens when the training signal is the gate, and how that gets exploited
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
- [Hybrid Attention](Hybrid-Attention.md) — per-layer attention modes, rotary positions, decode state, routing state, router diagnostics, the cost model
- [Negative Supervision](Negative-Supervision.md) — anti-pattern rules, labeled corpora, the unlikelihood charge
- [Multi-Source Training](Multi-Source-Training.md) — dataset mixtures and composites, several teachers, a negative teacher, merged checkpoints
- [Cyber Policy](Cyber-Policy.md) — blockers, refusals and signed approvals gating a model's capabilities
- [Direction Ablation](Direction-Ablation.md) — find a behaviour direction, remove it from the weights, or penalize it while training
- [Next-Step Planning](Next-Step-Planning.md) — beam search over trajectories and tokens
- [Geometric Reasoning](Geometric-Reasoning.md) — a dual-stream reasoner over a learned Riemannian metric, geodesic attention, certified attractor relaxation, block diffusion, and the exact rational kernel beside it
- [Geometric Reasoning Flaws](Geometric-Reasoning-Flaws.md) — every known flaw of the reasoner, its status, and the artifact that covers it
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
- [Claims](Claims.md) — every claim classified VERIFIED / PLAUSIBLE / REJECTED / UNKNOWN, with the harness that settles each
- [Architecture](Architecture.md) — ViT-DiT backbone and block partitioning
- [Precision & I/O](Precision-IO.md) — mixed precision, streaming reads, profiling
- [Parallel I/O & Block Execution](Parallel-IO-and-Block-Execution.md) — mirror-striped io_uring reads, and the sync/mt/par block execution selector
- [Quality Coder](Quality-Coder.md) — design document for the agentic code refiner (scaffolding only)
## Reward Integrity

A model trained to clear `cargo test` + `cargo clippy` +
`audit-bad-patterns.sh` has two ways to score: obey the gate, or change it. The
second is cheaper, survives one run, and is invisible to any metric that only
counts passing tests — so a training signal built on test outcomes alone rewards
exactly the wrong behaviour.

> **In this repository.** `src/cheat.rs` (`CheatReport`, `Finding`, `Severity`,
> `CheatClass`, `GateReport`, `scan`, `load_from_dir`, `gate`), wired as
> `dblocks cheat`. Certificates: the `cheat` group (26). Tests: 38 in-module.
> The paired half is `src/codegen_eval.rs` (`CorpusIndex`, `decontaminate`,
> `admit`, `split`, `TaskOutcome`, `Scorecard`), certificates in the
> `codegen_eval` group (22), and the task generator
> `examples/evalgen.rs` writing `repo-native-eval.jsonl`.

```bash
dblocks cheat --root .            # exits non-zero on any finding
dblocks cheat --root . --all      # every finding, not just the worst
dblocks cheat --root . --json     # machine-readable, for a training log
```

| Severity | Class | Example |
|---|---|---|
| **Fraud** | `gate-tampering` | deleting `audit-bad-patterns.sh` or `src/verify.rs`, `--cap-lints`, a manifest `[lints]` table, a clippy threshold raised out of reach, `|| true` in the gate script |
| **Suppression** | `lint-suppression` | `#[allow(…)]`, `#![allow(…)]`, `#[expect(…)]`, and the whitespace variants `#[ allow(…)` |
| **Evasion** | `test-evasion` | `#[ignore]` on a test, `assert!(true)`, a field-free `self == self` |

Fraud outranks suppression because suppressing one lint degrades quality while
editing the checker *manufactures* an appearance of quality.

**Scoring is all-or-nothing, and that is structural.** `GateReport::score()`
returns `1.0` or `0.0` and the type carries no partial-credit field, so a caller
cannot reintroduce weighting by accident. A run with green tests plus one new
`#[allow]` scores zero, not "mostly right" — any partial-credit scheme leaves a
cheap strategy (suppress the noisiest lint, bank the rest) which is the incentive
this exists to remove.

**Pre-existing suppressions are baselined, by count.** The crate ships 10
`#[allow]`s and 3 GPU-gated `#[ignore]`s that the audit already accepts. The
baseline is `(path, evidence, budget)` — deliberately *not* line-pinned, because
an unrelated edit above a baselined attribute would shift it and produce a false
positive, and the cheapest fix for a false positive is deleting the suppression
being audited. The count budget stops a new suppression hiding behind an
existing one in the same file.

**Known limits.** Detection is syntactic. `#[cfg(any())]` around a suppression,
a `build.rs` that rewrites sources before `cargo` sees them, or a suppression
assembled from concatenated string literals will pass. This raises the cost of
cheating; it is a gate on reward, not a sandbox. Two mitigations sit outside it:
keep the gate script and `src/verify.rs` out of the model's writable tree, and
re-hash them after every evaluation.

## Cheat: rewards integrity

The eval half answers a different question. Every public coding benchmark sits
inside the pretraining data of every frontier model, so a score on one measures
recall — and it *rises* as you train, which reads as improvement while being its
opposite. `src/codegen_eval.rs` therefore treats the contamination check as the
eval: a task is admitted only after it is shown absent from the training corpus
by 13-gram overlap, and a set that cannot be shown clean is reported rather than
quietly scored.

Three leak paths, three independent checks:

| Path | Check | Where |
|---|---|---|
| The task is in the training corpus | 13-gram overlap, threshold 0.10 | `decontaminate` |
| The teacher has memorised it | named benchmarks, advisory | `contamination_risk` |
| The task's *tests* are in training | tests are part of the checked text | `admit` |

Overlap is a fraction of the **task**, not of the corpus, so a large corpus cannot
launder a verbatim-contaminated task into looking clean — certified as
`a_large_corpus_cannot_launder_a_contaminated_task`. A task too short to check is
*refused*, not scored clean. The train/eval split is a hash of the task id rather
than a counter, so inserting one row cannot reshuffle the split and invalidate
every score recorded before the insertion.

The task set itself is repo-native (`examples/evalgen.rs`, 25 tasks, 93 test
functions): each states one rule this repository enforces and asks for an
implementation against the crate's real API. A model that has memorised
HumanEval gains nothing from them, because no amount of recall tells it that
*this* repo wants an error naming the offending value, or that composition
preserves the box load rather than summing to 1.

Every test was checked in **both** directions: the correct implementation passes,
and a plausible lazy implementation fails. That discipline earned its keep — it
caught three of the specs asserting falsehoods of their own (including
`1/3 + 1/6 != 0.5`, which is exactly `0.5` in f64), and it caught a monotonicity
test that could not see a wrong repulsion coefficient, which is the bug
`Geometric-Reasoning-Flaws.md` documents. A test a lazy solution also passes
measures nothing.

See also: [Quality Gate](Quality-Gate.md) · [Claims](Claims.md) ·
[Geometric Reasoning](Geometric-Reasoning.md) · [Home](Home.md)

---

## More

- [Model Parallelism](Model-Parallelism.md) — why block-wise training is not model parallelism, and what is
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
| Attention modes, rotary positions, per-mode decode state | `hybrid.rs` |
| Routing state, router kinds, routing-locality diagnostics | `routing.rs` |
| Active parameters, FLOPs and decode state, counted from shapes | `cost.rs` |
| Anti-pattern rules and labels for negative supervision | `antipattern.rs` |
| Code-quality signals, window filter, quality regularizer | `codequality/` |
| Quality-coder scaffolding: data adapters, eval harness | `quality_coder/` |
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
| Training loop, checkpoints and training state, logging | `train.rs`, `checkpoint.rs`, `logging.rs` |
| Experiment records, sweeps, propagation audit | `experiment.rs`, `sweep.rs`, `audit.rs` |
| Dataset and corpus mixing, checkpoint merging | `mix.rs`, `merge.rs` |
| Blockers, refusals, signed approvals | `policy.rs` |
| Behaviour directions: extraction, orthogonalization, penalty | `ablation.rs` |
| Heretic: kernel-weighted ablation searched against refusals and KL | `heretic.rs` |
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
