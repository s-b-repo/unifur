# TODO & Roadmap

This file tracks the implementation progress for DiffusionBlocks++.

Status legend: `[x]` implemented & tested · `[#]` partial (notes inline) ·
`[ ]` not started / blocked on external resources (noted inline).

**Everything implementable without a GPU, a cluster, or a new dependency is
done.** What remains is listed under
[Remaining](#remaining-requires-external-resources) with the specific blocker.

## Phase 1: Core DiffusionBlocks (Original Paper Port)

- [x] **1.1** Port ViT baseline (image classification)
- [x] **1.2** Port DiffusionBlocks training loop
- [x] **1.3** Port EDM noise schedule and loss
- [x] **1.4** Port data loading — CIFAR-100 binary (`cifar.rs`) and Tiny
      ImageNet (`tinyimagenet.rs`) both on the shared fixed-record reader
      (`rawdata.rs`), in-memory or streaming, fixture-tested. Tiny ImageNet
      needs a one-off JPEG→raw conversion (`tinyimagenet::CONVERTER`) since the
      crate carries no image codec.
- [x] **1.5** Port checkpointing and logging — content-addressed checkpoints
      (`checkpoint.rs`, save + load + `latest_in_dir`) and append-mode JSONL
      metrics (`logging.rs`). The JSONL schema is W&B-compatible; a hosted-W&B
      client stays out (network-free crate).
- [ ] **1.6** Reproduce CIFAR-100 results (baseline ViT) — *blocked: GPU hours*
- [ ] **1.7** Reproduce CIFAR-100 results (DiffusionBlocks) — *blocked: GPU hours*
- [ ] **1.8** Reproduce Tiny ImageNet results — *blocked: GPU hours*

**Status**: Core port complete, both datasets loadable; reproduction runs
pending hardware.

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
- [x] **2.7** Cross-fork training (non-adjacent block pairs) —
      `ConsistencyWeights::cross_fork` samples a pair `(i, j)` with `j >= i+2`
      and trains the joint span against the sequential composition of the same
      blocks, which is exactly the property `Strategy::Parallel` relies on
- [x] **2.8** Integration with training loop — `--objective consistency`
- [ ] **2.9** Benchmark K=1 vs K=2 vs K=3 on CIFAR-100 — harness shipped
      (`dblocks bench` reports time, model calls and layers executed per
      strategy); *quality comparison blocked: needs a trained model*
- [ ] **2.10** Document optimal hyperparameters — after 2.9

**Status**: Mechanisms and benchmark harness complete; quality numbers pending
hardware.

## Phase 3: Consistency Training

- [x] **3.1** Boundary consistency loss (`consistency.rs`)
- [x] **3.2** Self-consistency loss (different noise levels)
- [x] **3.3** Trajectory consistency loss
- [x] **3.4** Integrated with multi-block inference (`multi_block.rs`)
- [x] **3.5** Weight scheduling (constant / linear / cosine)
- [ ] **3.6** Ablation: consistency weight sweep — *blocked: GPU*
- [ ] **3.7** Ablation: boundary vs self-consistency — *blocked: GPU*
      (`ConsistencyWeights::boundary_only` / `none` exist for the sweep)

**Status**: Implemented; ablations pending hardware.

## Phase 4: Inference Solvers

- [x] **4.1** Euler solver (`solver.rs`)
- [x] **4.2** Heun solver (2nd order)
- [x] **4.3** DDIM solver (ancestral `eta`, variance-preserving split)
- [x] **4.4** DPM-Solver++ (2nd order multistep, published coefficients)
- [x] **4.5** DPM-Solver++ (3rd order) — exact-kernel exponential integrator
- [x] **4.6** Solver factory/selection (`SolverKind::parse` / `all`)
- [x] **4.7** Solver benchmark — `dblocks bench` sweeps every solver against
      every strategy; convergence order is separately certified against
      closed-form solutions in `verify.rs`
- [x] **4.8** Integrated into inference pipeline — a single `SolverState`
      drives `solve()`, `sample_multi_block` and the distillation teacher

**Status**: 5 solvers, order-certified, benchmarked.

## Phase 5: Flow Matching

- [x] **5.1** Rectified flow objective (`flow::flow_matching_loss`)
- [x] **5.2** OT path (straight-line conditional path)
- [x] **5.3** Linear sigma schedule (`t ~ U(0,1)`, `xt = (1-t)x0 + t·x1`)
- [x] **5.4** Training loop integration (`--objective flow`)
- [ ] **5.5** Compare with EDM objective across tasks — *blocked: GPU*

**Status**: Objective, sampler and trainer shipped.

## Phase 6: MoE Routing

- [x] **6.1** Top-K router (`moe::TopKRouter`, renormalized gates)
- [x] **6.2** Expert MLP pool (`ExpertPool` / `MoELayer`)
- [x] **6.3** Load balancing loss (Switch-style auxiliary, bounds certified)
- [x] **6.4** Noise-aware router — the router is conditioned on the adaLN
      vector, which is a pure function of sigma
- [x] **6.5** Integrate MoE blocks into DiffusionBlocks trunk —
      `ViTDiTConfig::moe` / `MoeTrunkConfig` replaces every n-th layer's MLP;
      the balance loss is summed over the executed span and added to the
      objective (`--moe-every`)
- [ ] **6.6** Benchmark: 1/2/4/8 experts — *blocked: GPU*

**Status**: Layer, trunk integration and losses tested.

## Phase 7: Block Distillation

- [x] **7.1** Teacher/student setup (`distill.rs`, teacher detached)
- [x] **7.2** Trajectory distillation — teacher takes N substeps, student one
- [x] **7.3** KL distillation with temperature and the `T²` correction
- [x] **7.4** Hard-label term
- [x] **7.5** Training-loop integration (`--objective distill --teacher ...`)
- [x] **7.6** QLoRA students — `quantize_module` produces a quantized student
      that distils against the full-precision teacher
- [ ] **7.7** Distillation quality study — *blocked: GPU*

**Status**: Implemented; the self-distillation identity is unit-tested.

## Phase 8: Adaptive Depth

- [x] **8.1** Halting probability module (`adaptive::HaltingHead`)
- [x] **8.2** Adaptive depth controller — confidence-driven span widening *and*
      narrowing in `Strategy::Adaptive`
- [x] **8.3** Depth regularization (expected-depth / ponder cost)
- [x] **8.4** Block skipping policy — cumulative-probability early exit
- [x] **8.5** Inference-pipeline integration — `Strategy::Adaptive` plus the
      loop graph's halting-driven ACT mixture
- [ ] **8.6** Benchmark quality vs expected depth — harness shipped
      (`SamplingStats::layers_executed` / `mean_span_width`); *quality blocked: GPU*
- [x] **8.7** Visualize block usage patterns — `ExecutionTrace::usage_histogram`
      and `SamplingStats::spans` record what actually ran

## Phase 9: QLoRA / Quantization

- [x] **9.1** NF4 quantization (`quantize::Nf4Tensor`, blockwise absmax)
- [x] **9.2** Double quantization of the per-block scales
- [x] **9.3** LoRA adapters (`LoraAdapter`, identity at init)
- [x] **9.4** QLoRA linear layer + adapter merging (`QLoraLinear`)
- [x] **9.5** Whole-model quantization (`quantize_module`, skip lists)

**Status**: Error bound, exact-zero and identity-at-init all certified.

## Phase 10: Multi-Block Denoising

- [x] **10.1** Sequential denoising (`Strategy::Sequential`)
- [x] **10.2** Parallel denoising K blocks (`Strategy::Parallel`)
- [x] **10.3** Hybrid strategies (`Strategy::Hybrid`)
- [x] **10.4** Adaptive dynamic K (`Strategy::Adaptive`)
- [x] **10.5** Quality-gated denoising (`Gated` + `quality.rs`)
- [x] **10.6** Precision denoising (mixed precision) — `precision.rs` emulates
      bf16/f16 exactly (RNE, subnormals, overflow) and `PrecisionPolicy` runs
      high-sigma windows coarse and low-sigma windows in f32. Emulation models
      representation error faithfully but is *not* a speedup; a native cast
      swaps in at `Precision::round_scalar` once a backend offers one.
- [x] **10.7** Benchmark all strategies — `dblocks bench`

## Phase 11: Hybrid Loop Graph Dynamic Transformers

- [x] **11.1** Dynamic graph construction (`loopgraph::LoopPlanner`)
- [x] **11.2** Skip / loop-back / early-exit / budget decisions
- [x] **11.3** Learned skip connections (zero-initialized, identity at init)
- [x] **11.4** Confidence-based early exit (`snr_confidence`, scale-invariant)
- [x] **11.5** Integration — `LoopGraph::x0_estimate` has the same signature as
      the fixed-depth predictor, so it plugs into `solver::integrate`

**Status**: Termination, budget and partition-of-unity are certified
invariants, not assumptions.

## Phase 12: Quality Gate

- [x] **12.1** MSE check (`quality::evaluate`)
- [x] **12.2** Cosine similarity check
- [x] **12.3** Confidence threshold
- [x] **12.4** Gradient norm check — wired into the training loop, which skips
      (rather than clips) a step whose global gradient norm is pathological
- [x] **12.5** Per-layer quality gates — `LayerGates` with per-block overrides,
      `LayerGates::tightening` for monotonically stricter low-sigma blocks, and
      a per-block `GateLedger`
- [x] **12.6** Batch filtering for bad samples (`filter_indices`, gate masks)
- [x] **12.8** Quality verification at **every phase of training**
      (`TrainingChecks` / `StepVerdict` / `TrainingHealth`): preflight
      certificates before step 0, finite-loss check, gradient-norm gate,
      finite-parameter check after the optimizer step, and periodic
      re-verification of the live model. Per-block health (mean loss, gradient
      range, rejection rate, dead-block detection) is reported at the end of
      every run.
- [ ] **12.7** Ablation: thresholds — *blocked: GPU*

## Phase 13: I/O & Performance

- [#] **13.1** io_uring data loading — the measurable goal (fewer syscalls per
      batch) is delivered by positional reads + run coalescing in
      `rawdata::StreamingSplit`. True `io_uring` submission needs an external
      crate and is deliberately not vendored.
- [x] **13.2** Async checkpoint saving (`save_content_addressed_async`)
- [x] **13.3** Batched syscall optimization — sampled indices are sorted and
      contiguous records coalesce into one `pread`; a 16-record contiguous span
      costs one syscall, asserted by test
- [x] **13.4** Zero-copy data transfer — records land in reusable buffers and
      convert straight to `f32`; steady-state batching allocates nothing
- [x] **13.5** Performance profiler (`profile.rs`, exact percentiles)
- [#] **13.6** io_uring benchmark — `StreamingSplit::reads_issued` exposes the
      syscall count the optimization targets; a native io_uring comparison
      depends on 13.1.

## Phase 14: Mathematical Foundation

- [x] **14.1–14.5** Every load-bearing identity is stated as a theorem and
      checked as a residual against a tolerance in `verify.rs`
- [x] **14.6** Numerical verification — 73 certificates across 15 groups, run by
      `dblocks verify` (non-zero exit on failure) and by the test suite

**Status**: See [Quality gate](#quality-gate) below.

## Phase 15: Production Features

- [x] **15.1** Mixed precision (bf16) — see 10.6; emulated exactly, native cast
      pending backend support
- [x] **15.2** Gradient checkpointing (`--grad-checkpointing`)
- [ ] **15.3** Distributed training (DDP) — *blocked: cluster hardware*
- [ ] **15.4** DeepSpeed — *N/A outside PyTorch*
- [ ] **15.5** FSDP — *blocked: cluster hardware*
- [x] **15.6** W&B logging — append-mode JSONL in a W&B-compatible schema;
      hosted client intentionally excluded (network-free crate)
- [x] **15.7** Model checkpointing — save, load, and `--resume` (with or
      without a path; bare `--resume` picks the newest in `--out-dir`)
- [x] **15.8** Inference API — `infer::InferenceEngine` (batching, top-k,
      accuracy, per-batch profiling) and `dblocks infer`. An HTTP server is out
      of scope: it would add a network dependency tree this crate avoids.

## Phase 16: Tests & Documentation

- [x] **16.1** Unit tests for model components (vit, dblock, sigma, stats)
- [x] **16.2** Unit tests for parallel trajectory (span selection, cost)
- [x] **16.3** Unit tests for solvers (order, exactness, published formulas)
- [x] **16.4** Unit/integration tests for flow matching
- [x] **16.5** Unit tests for MoE (top-1 exactness, gate partition, bounds)
- [x] **16.6** Unit tests for adaptive depth (exit boundaries, ponder range)
- [x] **16.7** Multi-block denoising — full strategy x solver matrix
- [x] **16.8** Unit tests for quality gate (including per-layer monotonicity)
- [x] **16.9** Unit tests for hybrid loop graph (termination, budget, ACT mass)
- [x] **16.10** Integration test for training
- [x] **16.11** Integration test for inference
- [x] **16.12** Wiki docs — `docs/Home.md`, `Training-Guide.md`,
      `Inference-Guide.md` and `Configuration.md` rewritten against the real
      CLI; new `docs/Quality-Gate.md`; every feature page now opens with an
      accurate "In this repository" block naming the module, types, flags and
      any deliberate deviation from the design spec
- [x] **16.13** Example scripts — `dblocks` subcommands double as examples

## Phase 17: Applications & Demos

- [ ] **17.1–17.4** Gradio/Colab/blog/video — out of scope for this Rust
  repository (different artifact types).

## Phase 18: Mixture of Specialized Micro Experts

Boxes of small specialized experts — a `coding` box holding a Rust expert, a
Python expert and a secure-code expert; a `cybersecurity` box holding its own
specialists — with a queryable index above every box so an inference engine can
route **without loading weights**.

- [x] **18.1** Expert identity and metadata (`expert_index.rs`: `ExpertSpec`,
      `BoxSpec`, `ExpertEntry`, `BoxEntry`, `ExpertKind`, `WeightLocator`)
- [x] **18.2** The index itself — `ExpertIndex`, JSON round trip, structural
      validation, `render()`. **No Burn dependency**, so an external routing
      service can depend on it alone
- [x] **18.3** Two-level router (`mosme::HierarchicalRouter`): box router plus
      one expert router per box, ragged box sizes supported
- [x] **18.4** Gate composition `g(i, j) = g_box(i) * g_expert(j | i)`, both
      levels renormalized so the product is a distribution by construction
- [x] **18.5** Hierarchical balance loss — box-level Switch loss plus a
      *gate-weighted* expert level, aggregated as a convex combination over box
      traffic shares. Weighting matters: a box the router never selects must
      contribute nothing
- [x] **18.6** Disabled experts via a `-inf` routing mask. `exp(-inf - max)` is
      exactly `0.0`, which is what makes adding an expert a bit-exact identity.
      The mask is a `Param<Tensor>` rather than a `Vec<bool>` because Burn gives
      `bool` module fields an `EmptyRecord` and they would not persist
- [x] **18.7** Trunk wiring — `FeedForward::Hierarchical`,
      `ViTDiTConfig::mosme`, `MosmeTrunkConfig`, `--mosme-spec/--mosme-every`
- [x] **18.8** Hot swap — `grown()` widens routers and boxes while preserving
      existing weights bit-exactly; new experts arrive disabled
- [x] **18.9** Index emission from a live module, written next to the
      content-addressed checkpoint and keyed by its hash
- [x] **18.10** CLI — `dblocks experts init | list | validate`
- [x] **18.11** Certificates (`mosme` group, 11 of them)
- [x] **18.12** Selective freezing and training modes (`TrainingMode`,
      `TrainableSet`): specialist (one expert trainable), router (experts
      frozen), experts (routers frozen), joint. Built on
      `GradientsParams::from_params` plus `burn::module::list_param_ids`.
      `test_specialist_training_moves_only_its_own_expert` takes a real
      optimizer step and asserts the siblings stay bit-identical. Documented
      hazard: `ParamId` does **not** survive `load_record`, so the set must be
      rebuilt after any resume
- [x] **18.13** Adapter site (`MosmeAdapterBank`) — experts as LoRA deltas over
      one shared frozen `Linear`, reusing `quantize::LoraAdapter`. A rank-8
      adapter on 256x128 costs 3072 parameters against 32768 dense.
      `MosmeAdapterBank::quantized` composes it with NF4 for full QLoRA.
      `merged_for` folds a routing decision into one dense weight, asserted
      equal to the factored form
- [x] **18.14** Model site (`MosmeEnsemble`) — whole micro-models mixed in
      *probability* space, so the `model` convex-hull certificate keeps holding.
      Real sparse dispatch: `forward_sparse` never evaluates an unselected
      specialist, and matches `forward_dense` bit-for-bit. Each specialist is
      indexed with its own `WeightLocator::checkpoint`
- [ ] **18.15** Benchmark box/expert counts — *blocked: GPU*

**Status**: All three expert granularities ship — MLP sub-layer, LoRA adapter,
and whole micro-model — sharing one router, one balance loss and one manifest.

## Phase 19: Language Modeling

The crate is an image classifier today: attention is fully bidirectional with no
mask parameter, sequence length is fixed at `num_patches + 1`, the only
`Embedding` is indexed by a rank-1 *class label* tensor, and the output is
`[batch, num_labels]` with no sequence axis. An LM path is new work, not a flag.

- [x] **19.1** Byte-level tokenizer (`tokenizer.rs`) — 256 byte tokens plus
      `<bos>/<eos>/<pad>`, no new dependency. Specials sit *above* the byte
      range so a token id is exactly its byte and a corpus stays readable with
      `xxd`. Lossless by construction on any input, binary included
- [x] **19.2** Pre-tokenized corpora (`corpus.rs`) — a flat little-endian `u16`
      file with no header, the same fixed-stride shape `rawdata.rs` uses, so any
      window is one seek and a corpus larger than memory costs no more per batch
      than one that fits. In-memory and streaming readers are certified to
      return identical windows. A truncated file is rejected rather than read
      with every token after the damage shifted
- [x] **19.3** Causal mask (`vit::causal_mask`, `ViTDiTConfig::causal`), added
      to the *scores* before the softmax so a masked position contributes
      exactly zero — masking probabilities afterwards would leave the
      denominator polluted by the future
- [x] **19.4** Token embedding and a **structurally** tied head: logits are
      `h @ E^T` against the embedding table itself, so there is no output
      projection to drift and the vocabulary is paid for once
- [x] **19.5** Next-token cross-entropy (`LanguageModel::next_token_loss`) with
      padding excluded from the denominator, not merely zeroed
- [x] **19.6** Generation (`generate`, greedy and top-k, reproducible from the
      caller's seed) **and the KV cache** (`lm::KvCache`, `forward_cached`,
      `generate_cached`): `O(n)` per token instead of `O(n^2)`. Not a
      speed-for-accuracy trade — a causal model's keys and values at position
      `i` depend only on tokens up to `i`, so the cache can only reproduce what
      a recompute would. Certified against full recompute under several
      chunkings, and end to end on the emitted tokens. Past the context window
      it stops rather than sliding: dropping the oldest cached positions would
      invalidate every position embedding after them
- [x] **19.7** Block-wise spans carry over — `forward_span` mirrors
      `denoise_span`, so gradient routing works on the LM path unchanged
- [x] **19.8** MoSME on the LM trunk — the same `DbLayer` is reused, so expert
      boxes, balance losses and spans all apply without a second implementation
- [x] **19.9** Certificates (`lm` group): lossless tokenizer round trip, causal
      attention leaking *exactly* nothing backwards, an untrained tied head
      starting at `ln(vocab)`, and top-1 sampling equalling greedy decoding
- [ ] **19.10** Benchmarks — *blocked: GPU*

**Status**: Done except the GPU benchmarks. A causal LM trains (exercised end to
end by `integration_a_language_model_trains_on_a_corpus`: tokenize, stream
windows, take AdamW steps, loss falls), streams a corpus, generates with a KV
cache, and decodes with lookahead — reachable from `dblocks lm`. Loading pretrained Llama/Mistral-class weights stays out of scope — it needs safetensors/GGUF
parsing and a real tokenizer, i.e. dependencies this crate has deliberately
avoided.

## Phase 20: Loss Reduction and Convergence

Motivated by a measurement from this repository's own runs: block 2 shows a mean
loss of **1909.8** against block 0's **13.4**, a ~140x imbalance driven entirely
by the EDM weight `w(sigma) = (sigma^2 + sigma_d^2)/(sigma sigma_d)^2` exploding
near `sigma_min`. That is expected from the objective, but it means the blocks
are not receiving comparable gradient scales.

- [x] **20.1** Learning-rate schedule (`schedule::LrSchedule`): constant,
      warmup+cosine, warmup+plateau. `--lr-schedule`. Certified never to exceed
      its peak and to decay monotonically after warmup
- [x] **20.2** Gradient accumulation (`GradientAccumulator`, `--accumulate`),
      averaging rather than summing so the learning rate survives a change in
      the accumulation count
- [x] **20.3** Gradient clipping (`clip_gradients`, `--clip-norm`). A complement
      to the skip gate, not a replacement: clipping rescales a merely-large
      step, the gate discards a pathological one
- [x] **20.4** EMA of weights (`schedule::Ema`, `--ema-decay`), with bias
      correction so the early shadow is usable rather than stuck on its
      initialization
- [x] **20.5** Uncertainty-weighted per-sigma loss (`reweight::UncertaintyWeighting`,
      `LogVarianceHead`, `--uncertainty`). The loss becomes
      `exp(-l)*L + l` with `l = logvar(sigma)`; the `+ l` term is what stops the
      head buying a smaller number by claiming more uncertainty, since the
      objective in `l` alone is minimized at `l = ln L`. **At that optimum the
      gradient is that of `log` loss, which is invariant under `L -> c*L` for
      any per-sigma constant** — so the 140x imbalance cannot reappear under a
      different weighting convention. The head is zero-initialized (an exact
      no-op at step 0) and lives *outside* the model record, so enabling it
      still writes checkpoints an unmodified build can load
- [x] **20.6** Importance sampling of sigmas (`reweight::SigmaImportanceSampler`,
      `--importance-bins`): equal-probability CDF bins proposed in proportion to
      the observed loss, each sample carrying its `p/q` weight so the estimator
      stays unbiased. The uniform-mixture floor is not optional — a starved bin
      is not merely unvisited, its importance weight explodes, so the floor
      bounds the worst weight at `1/smoothing`
- [x] **20.7** Per-block loss normalization (`LossScales`,
      `--normalize-block-loss`), equalizing on the *geometric* mean because the
      imbalance spans orders of magnitude and an arithmetic mean would be pinned
      to the largest block
- [x] **20.8** Certificates (`optim` group): accumulation over `k` steps equals
      one `k`x batch step *and* fires on exactly that cadence; the EMA
      coefficients sum to 1 and the blend never leaves the interval its inputs
      span; no schedule exceeds its peak, the ramp is non-decreasing and the
      decay non-increasing; the uncertainty objective is stationary and minimal
      at `ln L`; its gradient scale is 1 for every loss magnitude; the
      importance estimator is unbiased and its worst weight is bounded by the
      smoothing floor

**Status**: Done. Every item ships and the properties are certified rather than
asserted.

Measured on 400 CPU steps, 3 blocks, same seed, identical block visit counts:
`--uncertainty 1.0` cut absolute gradient magnitudes ~16x across every block and
took block 2's gradient-gate rejections from 5 (4.3%) to 0. Blocks 0 and 1 —
whose raw losses are within 1.5x — closed from a 2.5x gradient ratio to 1.24x.
Block 2 did **not** close (18x to 41x): 400 steps is not enough for the head to
track a loss 40x larger than its neighbours while that loss is still moving.
The theory says the ratio goes to 1 at the head's optimum; this run does not
reach it, and it is recorded that way rather than rounded up.

What is *not* claimed: that either reweighting improves final accuracy on this
crate's datasets. That needs GPU runs long enough for convergence to mean
something.

## Phase 23: MoE Routing Quality

Motivated by a literature review of MoE training pathologies and by this
repository's own MoSME runs, which showed 5 gradient-gate rejections in 400
steps with `max |g|` reaching **8.2e4**.

- [x] **23.1** Router z-loss (`moe::router_z_loss`, `--z-level`, ST-MoE default
      `1e-3`), applied at both levels of the hierarchy. It penalizes the
      **log-sum-exp**, not the logits, because that is what bounds the largest
      one: `max_e x_e <= logsumexp <= max_e x_e + ln E`. Computed in the shifted
      form — the naive `exp().sum().log()` overflows to `+inf` exactly where the
      loss is needed. A width-1 router is charged **nothing**: a softmax over one
      element is 1 whatever the logit is, so that logit steers nothing
- [x] **23.2** Balance-weight annealing (`schedule::BalanceSchedule`,
      `--balance-schedule anneal`), geometric because the useful range spans two
      decades and a linear ramp spends almost all its steps near the top. Holds
      the weight high while routing *collapse* is the risk, then decays it once
      every expert is alive and the pressure only costs specialization
- [x] **23.3** Certificates (`moe` group): the log-sum-exp sandwich bound; the
      z-loss is exactly the squared distance of the log-sum-exp from zero (so a
      row of `E` equal logits is optimal at `-ln E`, not at 0); and a per-row
      constant shift leaves every routing probability unchanged while the z-loss
      registers it — which is *why* logit drift needs its own term
- [ ] **23.4** Global-batch load balancing — Zhu et al., *Demons in the Detail*
      (2501.11873): computing the balance loss per **micro-batch** pushes the
      router to spread tokens evenly *within each batch*, which actively
      **inhibits expert specialization**. Their fix computes it over the global
      batch. Doing it here needs the per-expert counts to escape the forward
      pass — `FeedForward::forward` currently returns only a scalar — so it is a
      signature change through `vit.rs`, `dblock.rs` and `train.rs`. 23.2 is the
      tractable half of the same idea
- [ ] **23.5** Loss-free bias balancing — DeepSeek (2408.15664): a per-expert
      bias added to the routing scores *before* top-k, nudged by recent load with
      **no gradient**, so load balancing stops competing with the task loss. Needs
      state that mutates between steps inside a Burn module; the crate's
      `ModuleMapper` pattern (`quantize::Nf4Quantizer`, `schedule::EmaMapper`)
      is the way in
- [ ] **23.6** Routing entropy diagnostics — two normalized entropies per level:
      **load entropy** over the marginal expert usage (low = collapse) and
      **per-token routing entropy** (low = confident, specialized routing). They
      can disagree, and the disagreement is the whole diagnostic: balanced load
      with *high* per-token entropy is precisely the 23.4 failure mode, and it is
      invisible in the balance loss. Same plumbing blocker as 23.4

**Status**: The two items that fit the current plumbing ship — **and neither
helped**, which is worth recording as plainly as a success.

400 steps, 3 blocks, seed 42, MoSME with 2 boxes / 5 experts on every second
trunk layer. All runs visited the blocks identically (138 / 145 / 117):

| Configuration | mean loss | rejected | block 2 mean \\|g\\| | block 2 peak \\|g\\| |
|---|---|---|---|---|
| MoSME baseline (`--z-level 0`) | 482.90 | 5 | 2.44e3 | 8.19e4 |
| `--z-level 1e-3` | 494.76 | 5 | 2.36e3 | **7.03e4** |
| `--z-level 1e-3 --uncertainty 1.0` | **136.34** | **0** | **1.90e2** | **9.53e3** |

**This table replaces an earlier one that was measured with the z-loss reaching
the objective at 1% of its setting** (see the `mosme.rs`/`vit.rs` row in the bug
table). At full strength the conclusion is narrower than "moves nothing":

The z-loss **does** do its job: block 2's peak gradient falls 14% (8.19e4 to
7.03e4), which is precisely what a stabilizer is for. It is simply not the
binding constraint here — the five gate rejections survive it, because the gate
is firing on the EDM weight `w(sigma)` diverging in block 2, not on router logit
drift. Phase 20's uncertainty weighting is what addresses that, and does:
3.5x lower loss, 13x smaller gradients, every step accepted.

The mean-loss column is not a clean comparison across the first two rows: with
`z_level > 0` the z term is itself part of the reported loss, so some of the
482.90 -> 494.76 rise is the term being counted rather than the model being
worse. The gradient columns are the fair comparison.

So: a real but small effect on the thing it targets, and no effect on the thing
that was actually failing. Router logit drift is a slow failure mode, and 400
CPU steps from random initialization is nowhere near enough for it to bite; the
certificates show the term behaves as specified. It is insurance, correctly
priced at `1e-3`, against something that has not happened yet.

The remaining three items are designed and their specific blockers named; 23.6
is the one to build first, because it is what would let 23.4 and 23.5 be
*evaluated* rather than assumed — and, on this evidence, what would tell us
whether the experts are specializing at all.

## Phase 21: Next-Step and Path Prediction

Predict the best next step and evaluate candidate future paths, rather than
always taking the greedy one. Two planners sharing scoring primitives — not
forced into one abstraction, because the state spaces genuinely differ.

- [x] **21.1** Step-value model scoring a candidate next step. On the diffusion
      side the value is the confidence of the x0 estimate at the sigma the
      candidate departs from, less the compute the span costs; on the language
      side it is the model's own log-probability, which makes the model its own
      verifier and keeps this a decoding change rather than a training one
- [x] **21.2** Candidate generation over (sigma, span, step size) —
      `TrajectoryPlanner`, with the sigma ratios and span widths as the
      candidate set and a **progress term in log-sigma**. Without it the planner
      takes the smallest jump available every time (the most accurate single
      step is always the shortest) and never reaches `sigma_min`. Log-sigma
      because that is the coordinate the solvers integrate in
- [x] **21.3** Short-horizon rollouts under a compute budget
      (`DblockClassifier::sample_planned`). Rolled-forward latents are addressed
      by the path that produced them, so a deeper level is scored from where its
      hypothesis actually lands. `Budget` is checked *before* each model call,
      not after: truncating afterwards bounds the bookkeeping, not the compute
- [x] **21.4** Beam search over trajectories (`planner::Beam`), generic over the
      step type so the two planners share one search and one set of guarantees.
      Only paths at the **same** expansion level are ever compared — score
      deltas accumulate, so a partially expanded level would otherwise always
      look better (log-probabilities) or always worse (costs) purely because it
      is shorter. A level the budget cuts short never decides the outcome
- [x] **21.5** LM lookahead decoding (`generate_lookahead`), with prefix caching
      across the beam's shared continuations and a `LookaheadStats` report of
      forward passes per emitted token
- [x] **21.6** Certificates (`planner` group): the budget is never exceeded,
      under any (budget, depth, width) and against an `expand` that ignores its
      allowance; depth 0 is exactly the greedy policy; `beam(1)` reproduces
      greedy step for step and score for score; only the first step of a plan is
      committed; every planned step lowers sigma and none undershoots
      `sigma_min`; and lookahead actually defeats a myopic trap in both planners

**Status**: Done. `dblocks sample --planned` plans the trajectory;
`dblocks lm generate --lookahead N` plans the tokens. The step-value model is
heuristic rather than learned — a *learned* value head is the natural next step
and would need training data this repository does not have.

## Phase 22: Accuracy Improvements

Techniques that raise accuracy *after* training, without touching the weights.
Each has an exact identity setting, and each identity is certified — so a
mis-set knob degrades to the un-improved baseline rather than to garbage.

- [x] **22.1** EMA evaluation weights — `train` returns the shadow when
      `--ema-decay` is set, so evaluation uses the averaged weights rather than
      whichever point the last step happened to land on
- [ ] **22.2** Self-conditioning — **deliberately not built.** Feeding the
      previous x0 estimate back as conditioning only helps a model *trained*
      with it; bolted on at sampling time it degrades results. Doing it properly
      needs a new projection inside `vit.rs`, which adds a parameter to the
      module record and therefore cannot load an existing checkpoint. That cost
      is the reason it is still here rather than shipped half-done
- [x] **22.3** Test-time compute scaling (`accuracy::ScalingCurve`), reported by
      `dblocks bench` over both step counts and planner depths. The Pareto
      frontier is certified to be exactly the undominated set, and the marginal
      accuracy per extra layer is printed — the number a budget is actually set
      from. "More compute is more accuracy" holds up to a point; measuring is
      the only way to know where
- [x] **22.4** Solver and strategy ensembling (`accuracy::Ensemble`,
      `sample_ensemble`, `dblocks sample --ensemble`): probability mean, logit
      mean, and plurality vote. Returns probabilities, because a vote has no
      logit scale and a mean of distributions is not the softmax of anything
- [x] **22.5** Guidance-scale analogue (`accuracy::Guidance`,
      `--guidance`/`--guidance-rescale`). Scale 1 returns the conditional
      estimate **bitwise** rather than computing `u + 1*(c - u)`, which is equal
      only in exact arithmetic — and the result feeds an ODE step, where the
      lost bits would compound
- [x] **22.6** Logit normalization (`accuracy::LogitNorm`): temperature, L2, and
      standardization. Certified never to move the arg-max, which is exactly why
      it is worth having — `Strategy::Adaptive` and the quality gates threshold
      on confidences whose scale is an artifact of how large the trained logits
      happen to be, so a threshold tuned on one checkpoint means something else
      on the next. Label smoothing is a *training* change and is not here
- [ ] **22.7** Learned step-value head for the planner *(follows 21.1)* —
      *blocked: needs trained models to generate the value targets*

**Status**: Five of six shipped, plus a new item. 22.2 is the one deliberate
omission and its cost is written down above.

---

## Quality gate

`dblocks verify` runs a suite of numerical certificates. Each one states a
theorem, measures a **residual** that is zero exactly when the theorem holds,
and compares it against a tolerance derived from the arithmetic rather than
from current behaviour. The command exits non-zero on any failure, and
`cargo test` runs the same suite.

| Group | Certified |
|---|---|
| `schedule` | Endpoints, CDF-uniform spacing, window tiling, and that the training-time sigma sampler and the inference-time block router are mutual inverses |
| `preconditioning` | The EDM identity `D(z) = x` for an exact denoiser; `c_in²(σ²+σ_d²) = 1`; `w(σ)·c_out² = 1` |
| `stats` | `erf + erfc = 1`; `norm_ppf ∘ norm_cdf = id`; scipy reference values |
| `solver` | Closed-form exactness; kernel moments vs quadrature; interpolation nodes; observed order of convergence; ancestral variance preservation |
| `precision` | Relative error ≤ 2⁻ᵖ; rounding idempotence |
| `quantize` | NF4 error ≤ half the widest level gap; exact zero; LoRA identity at init |
| `loopgraph` | ACT weights are a partition of unity; the budget is hard under adversarial signals |
| `moe` | Gates are a distribution; the balance loss lies in `[1, E]` on the diagonal; `max_e x_e <= logsumexp <= max_e x_e + ln E`; the z-loss is the squared distance of the log-sum-exp from zero; a per-row shift moves the z-loss but no routing probability |
| `mosme` | Two-level gates compose into a distribution; one box reduces *exactly* to flat MoE; adding a disabled expert is bit-identical; a `-inf` mask gives an exactly zero gate |
| `lm` | Tokenization is lossless; causal attention leaks *exactly* nothing backwards; an untrained tied head starts at `ln(vocab)`; top-1 sampling is greedy decoding; the KV cache matches full recompute under every chunking, and cached decoding emits identical tokens |
| `planner` | The budget is never exceeded, even against an `expand` that ignores its allowance; depth 0 *is* the greedy policy; `beam(1)` reproduces greedy exactly; only a plan's first step is committed; planned sigmas fall monotonically without undershooting; lookahead defeats a myopic trap |
| `optim` | Accumulation over `k` steps equals one `k`x batch and fires on that cadence; the EMA is a convex combination that never extrapolates; the LR schedule is bounded, its ramp monotone; the uncertainty optimum is `ln L` and its gradient scale is 1 for every loss magnitude; importance sampling is unbiased with weights bounded by the smoothing floor |
| `accuracy` | Guidance at scale 1 is bitwise identity and affine in the estimates elsewhere; every logit normalization preserves the arg-max; ensembles emit distributions and N copies of one member are that member; the scaling frontier is exactly the undominated set |
| `model` | Softmax partition; `x0` inside the label-embedding convex hull; unit-norm embeddings; DiT zero-init |
| `autodiff` | Finite-difference gradient check on the distillation loss; Gibbs' inequality |

Two caveats stated plainly: these are numerical checks to a tolerance, not
machine-checked proofs, and a certificate only covers what it measures. The
suite is designed so that a regression in any identity the implementation
rests on surfaces as a *named* failure rather than as slightly worse accuracy
months later — which is how the `erf`/`erfc` crossover bug in this repository
was found.

### Verification during training

The same discipline is applied inside a run, because block-wise training fails
quietly: a step trains one block, so a dead block or a poisoned parameter stays
hidden in the aggregate loss for a long time. Each phase is checked separately
and reports separately.

| Phase | Checked | On failure |
|---|---|---|
| `preflight` | Schedule, preconditioning and statistics certificates | **Abort before step 0** |
| `loss` | The loss is finite | Reject the step |
| `gradients` | Global gradient norm within `[1e-8, 1e4]` | Reject the step |
| `parameters` | Every parameter still finite after the optimizer step | **Abort the run** |
| `periodic` | `verify::model_health` against the live weights | Log and record |

Preflight and the parameter check are fatal on purpose: training against a
broken schedule wastes the whole run, and a NaN in the weights poisons every
step after it. The others reject and continue — intermittent rejection is
normal in block-wise training, a *streak* is not, so 50 consecutive rejections
abort.

Every run ends with a per-block table (steps, rejections, mean loss, gradient
range) and an explicit warning for any block that ran without ever receiving a
non-zero gradient. Flags: `--verify-every N`, `--no-preflight`, `--no-checks`.
See [`docs/Quality-Gate.md`](docs/Quality-Gate.md).

## Bugs fixed while completing the roadmap

| Where | Bug |
|---|---|
| `sigma.rs` | `extended_window` indexed block windows in the opposite order to `estimate_target_layer`, so every block was **trained on one noise range and evaluated on another**. Now one documented convention (`block_window`), certified by a round-trip test. |
| `solver.rs` | DPM++2M used `r = (t_prev - t)/h`, which is negative in λ-space, so the multistep extrapolated backwards and was *worse than Euler*. |
| `solver.rs` | DDIM added ancestral noise *before* the drift step and computed the drift from the noised latent; the k-diffusion order is step-then-noise. `eta > 1` could also request more noise than the target level holds. |
| `solver.rs` | `integrate` ignored the caller's RNG, so seeded stochastic trajectories were not reproducible. |
| `consistency.rs` | Each residual was scaled by *its own value*, silently optimizing `L²` — a gradient that vanishes exactly where the residual is already small. |
| `consistency.rs` | The trajectory rollout multiplied sigma by a factor > 1, so the "chain" ran *up* the noise schedule. |
| `multi_block.rs` | `config.solver` was stored and never used: `sample --solver` was silently ignored and everything ran Euler. |
| `stats.rs` | `erf` used its Maclaurin series up to \|x\|=3, losing ~4 digits to cancellation; the crossover now matches `erfc` at \|x\|=2. Found by the `erf + erfc = 1` certificate. |
| `flow.rs` | `flow_sample` documented cosine similarity but computed an unnormalized dot product, ranking labels partly by embedding norm. |
| `logging.rs` | `File::create` truncated an "append-only" log, erasing history on resume. |
| `checkpoint.rs` | A fixed temp filename let concurrent saves rename each other's half-written files. |
| `quantize.rs` | `QLoraLinear::merged` built a `Param` from a tracked non-leaf tensor, panicking on the autodiff backend. |
| `quality.rs` | Merging per-batch gate ledgers added only the rejection counts, so any block that ever rejected anything reported a 100% rejection rate. |
| `lm.rs` | Burn's default embedding initializer is `N(0, 1)`, which with a *tied* output head gives logits of scale `sqrt(hidden)` — an untrained model that is confidently wrong. The initial loss was 28.6 against the `ln(259) = 5.56` a uniform model should show, so training would have begun by undoing a spurious prior. Caught by the certificate asserting the untrained loss. |
| `schedule.rs` | The EMA paired shadow and live parameters by `ParamId`, which `load_record` reassigns — so after a `--resume` the average would silently stop updating with nothing looking wrong. Now paired by traversal order, with a shape assertion, and `test_ema_survives_a_checkpoint_round_trip` pins it. |
| `expert_index.rs` | `ExpertKind` serialized *nested* as `"kind": {"kind": "mlp", ...}`, forcing an external engine to descend a level it has no reason to know about. The wire-format test passed because it only checked for the inner substring; the CLI smoke test exposed it. Now `#[serde(flatten)]`, and the test asserts the *absence* of nesting. |
| `moe.rs` | `TopKRouter::from_parts` built a `Param` from a tensor assembled with `cat`, which is a tracked non-leaf on an autodiff backend — so growing a model panicked there while passing every unit test on the plain `NdArray` backend. Caught by the integration test, which runs on autodiff. |
| `tensor_ext.rs` | Burn initializes `Linear` weights **lazily**, and `Param::clone` on an un-forced parameter gives the clone its *own* deferred initializer — so two clones draw **different random weights while keeping the same `ParamId`**. Found by the MoSME reduction certificate. `force_initialization` fixes it; `test_cloning_an_unforced_module_draws_different_weights` pins the behaviour. |
| `moe.rs` | Documented as per-token routing but the router only saw the per-example condition, so every token routed identically. |
| `multi_block.rs` | `SamplingStats::mean_span_width` divided `layers_executed` by the window count, but `layers_executed` also carries a solver's corrector evaluations and any planning work. Heun's spans were therefore reported at **twice** their real width, which inverts a strategy comparison. It now averages the recorded `spans`. |
| `multi_block.rs` | Planned sampling could stop above `sigma_min` when `max_steps` bound first, and then denoise the latent at `sigma_min` anyway — an estimate conditioned on a noise level the latent did not have. The remaining distance is now closed explicitly and `PlanTrace::forced_final_step` records that it happened. |
| `planner.rs` | The first beam search compared paths of *different* depths on raw accumulated score. Because deltas accumulate, a shorter path always wins under log-probabilities and always loses under costs — so lookahead could never beat greedy. Only fully expanded levels are compared now, and `test_lookahead_can_beat_greedy` is the regression. |
| `planner.rs` | The budget was charged *after* `expand` returned, so a scoring function that calls a model had already spent the compute by the time it was truncated. The allowance is now passed into `expand`, with truncation kept only as a backstop. |
| `schedule.rs`, `train.rs` | **`--accumulate` did not accumulate.** `GradientAccumulator` held only a counter: the loop scaled each micro-batch loss by `1/k`, backpropagated, and then stepped with *only the k-th* micro-batch's gradients — the other `k-1` were computed and dropped. `--accumulate 4` meant "run 4 passes, discard 3, take a quarter-sized step". Both existing checks passed anyway: the unit test asserted only the optimizer-step *count*, and the certificate validated the averaging *formula* in f64 without ever touching the code path. Now the accumulator sums real gradients through a `ModuleVisitor`, clipping applies to the accumulated step, and the certificate folds real gradients through a real module and compares against one `k`x batch. |
| `dblock.rs`, `train.rs` | The importance sampler was estimating its proposal from **its own reweighted output**: `per_sample` already carried the `p/q` factor, so a favoured bin reported a smaller loss, which lowered its own `q`, which raised `w` again. It oscillated instead of converging on the loss profile. |
| `dblock.rs`, `train.rs` | Importance and uncertainty weighting composed wrongly: `exp(-l)(wL) + l` instead of `w(exp(-l)L + l)`. The `+ l` term went unweighted, moving the head's optimum from `ln L` to `ln(wL)` — it would have learned to absorb the *proposal* rather than the loss scale, silently violating a certified property in the one configuration where both features are on. Weights are now carried out of `StepParts` and applied at aggregation, after any transform. |
| `mosme.rs`, `vit.rs`, `dblock.rs` | **The router z-loss reached the objective at 1% of its setting.** It was summed into the balance term, which is then scaled by `moe_aux_weight` (0.01), so `z_level = 1e-3` arrived as `1e-5` — and `--balance-schedule anneal` decayed the *stabilizer* along with the regularizer, weakening it exactly as a run gets long enough for logit drift to matter. Balance and z-loss are now carried separately end to end (`vit::RouterAux`), and `z_level` means what ST-MoE means by it. This invalidated a null result already reported and required re-measuring. |
| `train.rs` | `logvar_head.take()` sat inside a tuple pattern, so a failed match on any *later* element would move the head out and drop it — silently disabling uncertainty weighting for the rest of the run while the startup banner still announced it. Latent rather than live, but one refactor away from firing. |
| `multi_block.rs` | The trajectory planner rolled candidates forward with plain `euler_step` while the committed step used `SolverState`. For DPM++ 2M/3M — which integrate from a history of past x0 predictions — the planner was scoring candidates under dynamics the sampler would not follow. Rollouts now clone the real solver state per path, and draw noise from their own seeded RNG so speculation cannot perturb the committed trajectory's reproducibility. |
| `tests/integration.rs` | `integration_mosme_trunk_trains_end_to_end` asserted `balance_loss >= 2.0`, treating the Switch bound `L >= 1` as unconditional. It holds only on the diagonal `f == p`, which hard top-k routing does not give — so the test was really pinning one draw from a *global* backend RNG, and adding any concurrent test that builds a model broke it. It now asserts the structural upper bound, which is a theorem. |

## Remaining (requires external resources)

| Item | Blocker |
|---|---|
| 1.6–1.8 result reproduction | GPU compute |
| 2.9–2.10, 3.6–3.7, 5.5, 6.6, 7.7, 8.6, 12.7 ablations | Trained models (GPU). Harnesses exist; only the runs are missing. |
| 13.1 / 13.6 native io_uring | Would require an external crate; the syscall-count win is already delivered |
| 15.1 native bf16 | Backend support (ndarray = f32); emulation ships today |
| 15.3 / 15.5 distributed | Cluster hardware |
| 15.8 HTTP serving | Would require a network dependency tree; the library API ships today |
| 17.x demos | Different artifact type (web/notebook/video) |
| 18.15 MoSME benchmarks | GPU compute |
| 19.10 language-model benchmarks | GPU compute. The LM path itself ships; pretrained-LLM *loading* additionally needs safetensors/GGUF parsing and a real tokenizer |
| 20.x convergence *results* | GPU compute. The mechanisms ship and are certified; whether they improve final accuracy needs runs long enough for convergence to mean something |
| 22.2 self-conditioning | Needs a new module parameter, which breaks existing checkpoint records — see the phase note |
| 22.7 learned step-value head | Trained models, to generate the value targets |

## Test inventory

310 unit + 22 integration tests, all passing; `cargo clippy --all-targets`
clean; `cargo doc` warning-free. 73 numerical certificates in 15 groups, plus
five-phase verification inside every training run.

## How to Contribute

1. Pick an unchecked item from above
2. Create an issue referencing the item number (e.g., "Implement 2.9")
3. Fork the repo, implement, and submit a PR
4. Run `cargo test && cargo clippy --all-targets && dblocks verify` before
   opening it — new mathematical claims belong in `verify.rs` as certificates
5. Update this file when your PR is merged
