# Handoff prompt: 27B Burn wiring + NF4/offload engine (Path 2, Phases C–E)

> Paste everything below the line into Kimi K3 as the task prompt. It is
> written to be self-sufficient: repo state, ground truths, tasks,
> verification gates, and hard-won API lessons are all inline.

---

You are continuing Path 2 of `docs/Frontier-27B-Plan.md` in `/home/cortix/Downloads/inc/unifur` (Rust, Burn 0.21, crate `diffusionblocks`). The deliverable is a **geometric reasoning model**: the Qwen3.8-27B weights (DavidAU TURBO variant, BF16 safetensors source) wired in as the capability substrate, fused with this repo's geometric machinery — dual symbolic+geometric streams, learned Riemannian metric, geodesic attention, certified attractor relaxation, MoSME expert boxes for reasoning kinds, and the exact rational kernel as verifier — trained with block-diffusion on geometric corpora on consumer hardware (RTX 3060 12 GB + 46 GB RAM). The 27B wiring below is the foundation, not the finish line: a plain fine-tuned 27B without the geometric fusion does NOT satisfy this task. Success = kernel-verified geometric reasoning that beats the base model on geometric benchmarks (Geometry3K/GeoQA first), with side-by-side task behavior as the secondary gate.

## Step 0 — confirm the baseline before touching anything

1. `cargo test --lib` must show **659 passed, 0 failed** (includes the shard-audit tests that read `/srv`; they skip cleanly if files are absent).
2. `cargo clippy --all-targets` must show no errors.
3. `./target/release/dblocks verify` → 167/167 certificates (rebuild release first if stale: `cargo build --release`).
4. Weights must be present: `/srv/m-sdd/unifur/weights/qwen38-bf16/` should hold `model-00001..00012-of-00012.safetensors` + `model-mtp-restored.safetensors` (~55.7 GB total). If shards are missing, the background retry loop is `/srv/m-sdd/unifur/dl-loop.sh` — check `download-qwen38.log` beside it, do NOT start a second downloader.
5. Tokenizer + data already on disk: `/srv/m-sdd/unifur/tokenizers/qwen38-source/tokenizer.json`, `/srv/m-sdd/unifur/datasets/sft-12k.jsonl` (+ `subset-report.txt`, `build_subset.py`).

## The geometric target (read before designing anything)

- `src/geometry.rs` — `GeometricReasoner`: dual-stream reasoner (symbolic byte tokens + continuous slots in a learned Riemannian metric), geodesic attention, certified Lipschitz energy relaxation to an attractor, MoSME readout. Train via `dblocks geom train --objective answer|diffusion`; corpora via `dblocks geom data --kind-weights` (stratify hard kinds); eval via `dblocks geom eval` (accuracy, `--refine-sweep`, agree-with-depth-1 trace-dependence).
- `docs/Mixture-of-Specialized-Micro-Experts.md` — MoSME boxes (coding:rust,python,secure; reasoning:math,logic,proof). The fused model routes geometric queries through these boxes with the geometric stream attached.
- `src/geomkernel.rs` — exact rational kernel (propose/verify split: model proposes constructions/answers, kernel proves). Every geometric answer the fused model emits must be kernel-checkable.
- `docs/Geometric-Reasoning.md` + `docs/Geometric-Reasoning-Flaws.md` — mechanism docs and the known-flaw ledger (read both; the scatter-sums-to-n bug class in particular must never reappear in new routing code).
- Research mapping already in-repo: `docs/Hybrid-Attention.md` (Qwen/Kimi/GLM lab notes), `docs/Training-Guide.md` (diffusion-lab notes), `docs/Geometric-Reasoning.md` (geometric-models notes + eval order: Geometry3K/GeoQA → MathVista-GPS → miniF2F → ProofNet).

## What is already done (do not redo, do reuse)

- `src/bpe.rs` — HF `tokenizer.json` loader (vocab/merges/added, 3 GPT-2 split dialects incl. Qwen's exact pattern, byte alphabet, NFC, `u32` ids). Parity-tested against the real Qwen3.8 file.
- `src/import.rs` — safetensors header index + ranged `f32` decode (never loads whole shards), exact F16/BF16 conversion, corrupt-header refusal.
- `src/deltanet.rs` — Gated DeltaNet **reference core** (recurrent math only): `gated_delta_step`, `gated_delta_recurrent`, causal depthwise conv, L2 norm, Qwen3.8 grouping preset (16 groups × 1Q/1K + 3V, dim 128). 6 unit tests + 5 `deltanet` gate certs.
- `src/qwen.rs` — architecture dims + **tensor-name remap validated against all 1,199 real names** (851 text = 48×14 + 16×11 + 3; 15 MTP + 333 vision allow-listed). `QwenPart::expected_shape` gives file-side shapes; `audit_tensors` validates a shard index.
- `examples/tokcount.rs` — token counter used for the 6.53M-token measurement.
- Full spec: `docs/Frontier-27B-Plan.md` (read it first).

## Ground truths (violating any of these is a bug — all verified against real artifacts)

1. Tensor root is `model.language_model.` (NOT `model.`). MTP lives under `mtp.*` (15 tensors), vision under `model.visual.*` (333). Both are allow-listed skips, never absorbed.
2. This generation uses **separate** projections: `in_proj_qkv` + `in_proj_z`, `in_proj_b` + `in_proj_a` (Qwen3-Next fuses them — do NOT copy Next layouts).
3. `conv1d` is **bias-free** here (no `conv1d.bias` in the 1,199 names).
4. File layout is PyTorch `[out, in]`. Verify Burn 0.21's `LinearConfig` weight layout with a unit test before filling any module; transpose on fill as needed.
5. Full-attention `q_proj` is **double-width** (`attn_output_gate`): Q concatenated with its sigmoid gate, split downstream. `q_norm`/`k_norm` are plain head-dim RMSNorms applied post-projection.
6. Gated-delta math: `S_t = α·S(I−βkkᵀ) + βvkᵀ`, `o = Sq`; `β = sigmoid(b)`; `g = −exp(A_log)·softplus(a + dt_bias)`, `α = exp(g)`; q/k L2-normalized; Q and K repeated per V-group (48/16 = 3); output gated by SiLU(z) through RMSNormGated then `out_proj`.
7. The 27B is dense (no MoE); `lm_head` is **untied** (the repo's LM head is tied — that gap is part of your work).

## Task 1 — `GatedDeltaHead` Burn module + per-layer loader (foundation for Task 4)

- Module holding: `in_proj_qkv/z/b/a` linears, depthwise `conv1d` (+ measured state), `dt_bias`/`A_log` params, RMSNormGated, `out_proj`; decode state `S` per (group, V-head) + conv state.
- `forward` (prefill: conv → split per group → L2 q/k → recurrent rollout via `deltanet.rs` core → gate → norm → out_proj) and `step` (single-token decode reusing state).
- Loader: ranged-read each tensor via `import.rs`, transpose per (4), split qkv/z/ba per the vLLM-verified group layout, fill Params; **assert every loaded value finite**.
- Tests: forward-finite on random weights; decode-step equals prefill-prefix outputs (same discipline as the repo's KV-cache certs); one layer filled from **real shard weights** runs finite (skip-if-absent when shards missing).

## Task 2 — 64-layer trunk load

- Assemble embed → 48× linear layers + 16× full-attention layers (every 4th, 1-based) + SwiGLU MLPs + RMSNorms + final norm + **untied** `lm_head`. Reuse the repo's existing attention (GQA/partial-RoPE/gating/QK-norm flags already exist) wherever the math matches; add only what is missing.
- Loader walks all shards through `audit_tensors` first (refuse on any shape/name mismatch), then fills.
- Tests: full-model forward on a short prompt is finite with sensible logit scale; checkpoint round-trips through the repo's content-addressed format.

## Task 3 — NF4/offload engine (memory budget: ~14 GB RAM + ≤11 GB VRAM)

- NF4-*resident* weights (today `quantize.rs` only simulates in f32) + CPU paging (one block on GPU, prefetch next; see `blockexec.rs`) + paged AdamW over adapters only.
- Then the adapters-only blockwise loop (auto-LoRA on q/k/v/o/gate/up/down; never full-model steps), carrying over accumulation, EMA, max-logit monitoring, and verified resume.
- Gate: 27B resident across RAM/VRAM, a training step completes end-to-end, resume is bit-identical.

## Task 4 — geometric fusion, training, and proof (the actual deliverable)

Tasks 1–3 exist to serve this. Do not stop at a wired 27B.

- **Fusion architecture**: attach the geometric stream to the wired trunk — project Qwen hidden states into slots, run the metric/geodesic-attention/relaxation path of `geometry.rs` over them, read out through MoSME boxes sized for reasoning kinds (math, logic, proof, code). Query/key routing must reuse the certified `moe::scatter_gates` helper (the hand-rolled broadcast-scatter bug that once summed rows to `n` instead of 1 is documented in `docs/Geometric-Reasoning.md` — read it first).
- **Training**: block-diffusion (`--objective diffusion`) plus answer-objective runs on stratified geometric corpora (`--kind-weights` oversampling hard kinds), with teacher traces from the 27B where they help. Reuse the adapters-only loop, accumulation, EMA, max-logit monitoring, and verified resume from Task 3.
- **Proof of reasoning** (all required): `geom eval` accuracy above chance with margin on held-out scenes; `--refine-sweep` showing the depth curve plus high agree-with-depth-1 only where the curve is flat (deep refinement must *change* answers while accuracy is still climbing); every emitted answer kernel-checked via `geomkernel`; a `readout_gates_form_a_distribution`-style certificate for any new routing code you add.
- **Benchmark order**: synthetic held-out first, then Geometry3K/GeoQA. Not ARC-AGI, not MATH-Vision/PutnamBench until tier 1 clears 60%.

## Repo law (the audit script enforces it — run `./audit-bad-patterns.sh`)

- `anyhow::Result` + `ensure!`/`context`; **no `unwrap`/`expect`/`panic!` outside `#[cfg(test)]`**; loud errors naming the offending value, never silent fallbacks.
- Every numerical identity gets a `verify.rs` certificate (new `deltanet` certs for the chunkwise form when it lands; tolerance 0.0 for exact identities).
- `cargo test --lib` green + `cargo clippy --all-targets` clean before you stop. Temp files get unique names (atomic counter — parallel tests share `/tmp`).
- Burn 0.21 API lessons already paid for: `unsqueeze_dim::<2>(1)` (bare `unsqueeze` hits dim 0), `squeeze::<1>()` takes no arg, `from_floats` rank must match the literal nesting (rank-1 then `reshape`), `repeat_dim` tiles (document any grouping assumption), `argmax` keeps its dim.
- `serde(default)` on every new config field (old checkpoints sidecars must still parse); don't break existing tests — extend them.
- Do NOT touch: `src/geombaseline.rs`, `src/grade.rs`, `docs/Geometric-Reasoning-Flaws.md`, `geom*/` corpora (a concurrent workstream owns them).

## Explicit non-goals

Vision tower, MTP heads (frozen/dropped), 256K context (start 8–32K), Muon optimizer, speculative MTP decode, GGUF export (Phase F, separate task), any new `.md` docs (contract lives in rustdoc + certs).
