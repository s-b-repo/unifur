# Frontier-27B Plan (Path 2): fine-tune Qwen3.8-27B in-repo on consumer hardware

Goal (user-defined success): the 27B behaves better on the user's tasks.
Benchmarks secondary; side-by-side task behavior vs base is the gate.
Honest ceiling: a fine-tuned 27B specialist, not a frontier foundation model.

Target weights: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF`
(reference) + BF16 safetensors source (`…-NM-DAU` collection) for training.
Downloading now (background retry loop `/srv/m-sdd/unifur/dl-loop.sh`, 25
attempts): 12 shards, 55.6 GB total, to
`/srv/m-sdd/unifur/weights/qwen38-bf16/`. Arch confirmed from its
`config.json`: 64 layers, 24Q/4KV, hidden 5120, inter 17408, head-dim 256,
vocab 248320, 3:1 DeltaNet/full hybrid, MTP 1 layer, untied head, BF16,
text-only scope (vision encoder out). NOTE: unauthenticated pulls are
throttled and one xet transfer already died mid-shard (partials wiped,
completed files intact); a `HF_TOKEN` would raise limits and speed this up
several-fold.

## 0. Measured constraints (verified, not assumed)

| Fact | Number | Consequence |
|---|---|---|
| Free disk (`/`, 906 GB total) | **7.3 GB** | Blocks every download (Q4_K_M = 18.5 GB). Need ≥70 GB on the user's other drive before anything starts |
| Free RAM / swap | 14 GB free / swap 100% full | 16 GB quant file needs app-closing + free disk first; close desktop apps or OOM |
| Free VRAM (RTX 3060) | 11.2 of 12 GB | Q4_K_M never fits whole; partial GPU+CPU offload only |
| QLoRA 27B floor | ~22 GB (Unsloth / QLoRA-paper math) | Weights borrow RAM; expect pageround throughput |
| Block diffusion saves | Activations O(1), **not parameters** | 27B params = ~13.5 GB at 4-bit before one gradient exists |
| 3 days at ~10 tok/s | ~2.6M tokens ≈ 5K samples × 500 tok | Unsloth-scale LoRA fine-tune (1–10K samples, 1–3 epochs). Qwen3 trained on ~36T: ratio ≈ 1:10,000,000 |
| Repo today | 259-token byte LM, no BPE, no safetensors/GGUF loader, f32 CPU training, largest runs = tiny CPU demos | 13 audited gap items (see Phase A–F) |

Retired claim: nobody trains frontier-scale models on consumer hardware.
DavidAU's ARC numbers are fine-tunes/merges of Alibaba's cluster-trained
Qwen; Unsloth runs target 24 GB+ cards. Memory tricks change what fits;
only data × compute change how good it gets.

## 1. Datasets (public; Polar-STRICT / F451-STRICT are private 401s)

Landed on `/srv/m-sdd/unifur/datasets/` (all drives writable; `m-sdd` chosen, 484 GB free):

| Pick | Dataset | Landed | Role |
|---|---|---|---|
| 1 (core code SFT) | `ise-uiuc/Magicoder-OSS-Instruct-75K` (75,197 rows, card: MIT) + `ise-uiuc/Magicoder-Evol-Instruct-110K` (111,183 rows, card: Apache-2.0) | 194 MB + 244 MB JSONL, decontaminated splits | Re-verify licenses before any release |
| 1a (train subset) | `sft-12k.jsonl` via `datasets/build_subset.py` (seed 7, deterministic) | 12,000 rows (4,800 OSS + 7,200 Evol), 26.3M chars | **6,525,976 real Qwen tokens** (measured with `examples/tokcount.rs`, 0 failures, 0 over-8K, mean 544/row) |
| 2 (reasoning traces) | `nvidia/OpenMathInstruct-1` (1.8M, permissive) or `-2` 1M-split | Not yet downloaded | Qwen's own recipe: math inclusion doesn't hurt code |
| 3 (harder reasoning) | `AI-MO/NuminaMath-1.5` (~900K CoT, Apache-2.0) | Not yet downloaded | Hard slice |
| 4 (reserve) | `OpenCodeInstruct` (5M) / `OpenCoder-LLM` SFT (435K) | Not yet downloaded | Later runs only |

Run math (measured, not estimated): 6.53M tokens ≈ **7.5 days at 10 tok/s per epoch**.
Phase G config decision: ~1 epoch on 12K in a week, or trim to 8K for 2 epochs.
Qwen `tokenizer.json` (20 MB, 248K vocab) landed at
`/srv/m-sdd/unifur/tokenizers/qwen38-source/`; the ignored parity gate now
runs green against it in-suite.

## 2. Engineering phases

### Phase 0 — prerequisites (user, ~1 hr, blocks everything)
- [x] Writable workspace: `/srv/m-sdd/unifur/` (`datasets/`, `tokenizers/`,
      `.hf-cache/` with `HF_HOME` pointed there so `~/.cache` stays empty).
      `m-sdd` verified 484 GB free; all `/srv/m-sd*` writable.
- [ ] Confirm dataset picks (§1 landed per this file) or adjust
- [ ] Close desktop apps / free RAM before any 16 GB+ file lands (14 GB free
      now, swap full — required before Phase B's BF16 source download)

### Phase A — BPE tokenizer (est. 2–4 d) → `src/bpe.rs` — DONE
- `tokenizer.json` loader (Qwen 248K BPE); `u32` ids (current `Vec<u16>` overflows at 65K — `src/tokenizer.rs:63,83`); encode/decode round-trip cert; chat templates.
- Gate: `dblocks lm tokenize` reproduces HF token ids on samples.
- Status: complete. GPT-2-family pipeline fully implemented: regex splitter
  (original + GLM dialects, hand scanner, no regex crate), byte alphabet
  (exact mapping, bijective), file-derived defaults (strategy/dialect/
  byte-level/prefix-space), feasible normalizers (Lowercase/Strip/Prepend/
  literal-Replace/Sequence; NFC-family + regex-Replace stay loud errors —
  decomposition tables are infeasible `std`-only), `"a b"` + `["a","b"]`
  merge forms, `#version` skip, added ids past vocab range, merge-rank
  fidelity. 12 tests green incl. real-file validation (154K vocab / 321K
  merges / 36 added tokens load; diverse round-trips; merges fire; added
  bypass), clippy-clean, audit-clean (remaining hits: in-test asserts per
  repo culture, one intentional `#[ignore]` gate, guarded indexing).
  Real-file parity gate is the ignored test
  `qwen_tokenizer_parity_against_reference` (`QWEN_TOKENIZER_JSON` env).

### Phase C — Qwen3.8 arch, DeltaNet core DONE → `src/deltanet.rs`
- Gated delta rule (Yang et al. Eq. 10, verified against the paper text +
  vLLM's `QwenGatedDeltaNetAttention` splits): per-pair recurrence,
  L2-normalized q/k, causal depthwise conv (kernel 4), Qwen3.8 grouping
  (16 groups × 1Q/1K dim-128 + 3V dim-128, α/β per V head), config +
  cost helpers. 6 unit tests + 5 gate certs (`deltanet` group) green.
- REMAINING: chunkwise/WY training form (paper §3.3), `GatedDeltaHead`
  module with projections + decode state, full `src/qwen.rs` 64-layer
  wiring (awaits Phase B remap against real shard names).

### Phase B — weight loader → `src/import.rs` + `src/qwen.rs` — READER + REMAP DONE
- Reader: header index, ranged reads, exact F16/BF16, 5 tests green.
- Remap (`src/qwen.rs`): SOURCED from HF `modeling_qwen3_5.py`, not vLLM
  (which fuses/renames). This generation uses separate `in_proj_qkv` +
  `in_proj_z`, `in_proj_b` + `in_proj_a`, and a **bias-free** `conv1d` —
  confirmed against the real `model.safetensors.index.json` (1,199 names:
  851 text = 48×14 + 16×11 + 3, 15 MTP, 333 vision; prefix
  `model.language_model.`; test `test_remap_covers_the_real_shard_map`
  pins the whole inventory). MTP/vision are allow-listed skips.
- REMAINING: per-shard shape audit against landed weight files
  ([`audit_tensors`] is written; needs the bytes), then Burn module fill
  (Phase C `qwen.rs` wiring + Phase D NF4/offload).
- Status: ALL 12 shards + MTP shard landed (55.7 GB) and audited
  (`test_all_landed_shards_audit_clean`: every tensor name parsed, every
  shape checked, per-shard spot decodes finite). Two real findings from
  real weights, both fixed in `src/qwen.rs`: PyTorch stores linears as
  `[out, in]` (transpose on Burn fill), and full-attention `q_proj` is
  double-width (output gate concatenated, `attn_output_gate`). Also fixed:
  parallel-test temp-file collision in `src/import.rs` fixtures (unique
  names via atomic counter).
- safetensors parser (check Burn 0.21 `burn-store` support first; fallback: minimal parser — JSON header + raw tensors) + Qwen3.8 remap (fused qkv split, gate/up/down, norms, untied `lm_head`; freeze/drop MTP heads; skip vision).
- Tolerant loader: partial load + shape-mismatch *report*, never abort (`src/checkpoint.rs:34,90-99`).
- Gate: BF16 source loads, every tensor accounted, hash-manifested.
- Status: reader DONE (`src/import.rs`: header index, ranged reads — never
  loads whole shards, exact F16 (incl. subnormals) / BF16 conversions, 5
  tests green, clippy + audit clean). Burn's `burn-store` exists upstream
  but the remap needs raw named tensors, so self-owned reader stands.
  REMAINING: Qwen name→module remap, implemented after inspecting real
  shard tensor names (not guessed) once the download lands.

### Phase C — Qwen3.8 arch in Burn (est. 1–2 wk, the hard one) → `src/qwen.rs`
- 64 layers of 3×Gated-DeltaNet + 1×Gated-Attention (DeltaNet per spec — repo's ELU-linear is different math: `src/hybrid.rs:339,359`).
- GQA 24Q/4KV head-dim 256, partial RoPE-64, SwiGLU-17408, RMSNorm, 248K embedding; start ctx 8–32K (not 256K).
- Reuse: gated-attention, QK-norm, RMSNorm, rotary-fraction.
- Gate: random-init forward finite + `verify` group for DeltaNet recurrence vs masked form (same discipline as the linear cert).

### Phase D — memory engine (est. ~1 wk) → `quantize.rs`, `blockexec.rs`, `schedule.rs`
- NF4-*resident* (today f32-simulated: `src/quantize.rs:26-28`), CPU offload paging (one block on GPU, prefetch next), paged AdamW for adapters only.
- Budget target: base ~14 GB RAM + active block + adapters + activations ≤ 11 GB VRAM.
- Gate: 27B resident across RAM/VRAM, step completes, resume bit-identical.

### Phase E — LoRA-blockwise loop (est. 3–5 d) → `quantize.rs`, `mosme.rs`, `main.rs`, `train.rs`
- Auto-LoRA on q/k/v/o/gate/up/down (rank/alpha CLI); adapters-only `TrainingMode` (never `Joint=all` on 27B: `src/train.rs:1906,1956-1960`); LM gradient checkpointing (exists only for image path: `src/train.rs:66-78`); carry over blockwise + accumulation + EMA + maxlogit monitoring.
- Gate: loss falls on 100-step shakedown; adapters-only delta verified (frozen base bit-identical).

### Phase F — GGUF export (est. 2–3 d)
- Merge adapters → BF16 → GGUF via `gguf-rs` (read/write+mmap exist). Train against the BF16 source; quantize only at export (K-quant *dequant* for training-time reads is the alternative — porting dequant tables costs more, do it only if source is unavailable).
- Gate: exported file loads in llama.cpp; perplexity within noise of in-repo eval.

### Phase G — the run (days, RTX 3060 at 100%)
- 5–15K code + 2–5K reasoning traces → tokenize → SFT, cosine schedule, small LR, batch via accumulation, checkpoint every N steps with full verified state, watch `maxlogit`/balance.
- Eval per §0 success metric: side-by-side task behavior vs base 27B.

## 3. Risks
- **Throughput, not memory, is the risk**: every PCIe round-trip taxes each step; plan sample counts from single-digit tok/s, not hope.
- **Long-run stability** (driver timeouts, OOM creep, power loss): checkpoint-every-N + verified resume (already repo discipline).
- **Scope guardrails**: vision encoder out (text-only coder); 256K context out (start 8–32K); Muon/speculative-MTP out (separate projects).

## 4. Open inputs (blocking Phase 0)
- [ ] Other-drive mount path + `df -h` showing ≥70 GB free
- [ ] Dataset confirm (default §1 table) or replacements
- [ ] Go-ahead to start Phase A
