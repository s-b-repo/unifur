# Plan: Frontier-Quality Geometric Reasoning Model (Vector Stream + MoSME + Anti-Cheat)

## Goal

Train a geometric reasoning model that beats the matched-depth transformer baseline on held-out geometric scenes and Geometry3K/GeoQA, using the repo's existing fused stack (`FusedQwen` = Qwen trunk + `GeomFusion` vector stream + MoSME readout), with every emitted answer kernel-checked by `geomkernel`, anti-cheat gates green, and the 6-stage research pipeline mapped onto existing machinery. Target hardware: RTX 3060 12 GB + ~46 GB RAM (consumer). Success gate is side-by-side task behavior + `geom eval`/`prove` margins, not benchmark chasing.

## Current state (do not redo)

- `src/geomfusion.rs`: `GeomFusion`, `FusedQwen`, `FusedQwenTrainer`, `train_fused_qwen`, `prove` already exist; certs in `geomfusion` group (gates pass).
- `src/geometry.rs`: standalone reasoner + certs (`geom` group, 18).
- `src/mosme.rs`, `src/moe.rs`: certified routing; readout reuses `HierarchicalRouter` (no hand-rolled scatter).
- `src/qwentrain.rs`: adapters-only loop, segmented backward, paged AdamW, verified resume.
- `src/cheat.rs`, `src/mlcheat.rs`, `src/antipattern.rs`, `src/codequality/`: anti-cheat + decontamination + quality signals.
- `src/vecread.rs`: decodes raw-vector thinking (logit lens + tuned lens) into traces.
- `docs/research-pipeline.md`: 6-stage pipeline reference.
- CLI: `geom` subcommands exist; **no fused training/eval CLI** (gap).
- `docs/Geometric-Reasoning.md` measured: reasoner 0.292 vs baseline 0.312 at 3000 steps (REJECTED at that scale); fused run unmeasured.

## Constraints

- Hardware: 12 GB VRAM, ~46 GB RAM, single 3060. Adapters-only training (never `Joint=all` on 27B). NF4-resident weights + CPU paging.
- Repo law: no `unwrap`/`expect`/`panic!` outside `#[cfg(test)]`; every numerical identity is a `verify.rs` certificate; `make gate` green before stop; unique temp file names; `serde(default)` on new config fields.
- Anti-cheat: decontaminated corpora, cheat harness registry, nina fleet, policy gate. No training on unfiltered autoformalizations.
- Do not touch: `src/geombaseline.rs`, `src/grade.rs`, `docs/Geometric-Reasoning-Flaws.md`, `geom*/` corpora (concurrent workstream).
- No new `.md` docs beyond plan; contract lives in rustdoc + certs. Doc *updates* to existing docs are allowed and listed below.

## Decisions (resolved)

1. Base substrate: Qwen3.8-27B BF16 (existing Phase B/C wiring) as trunk; geometric stream attached via `GeomFusion` (identity at birth).
2. Readout: MoSME boxes named for reasoning kinds (math, logic, proof, code) reusing certified `HierarchicalRouter`; no second routing impl.
3. Verifier: `geomkernel` (exact rationals) decides verified/falsified/indeterminate on every emitted answer; denoising loss never replaces verifier reward.
4. Training objective: answer CE on answer span + optional EDM denoising on geometric state (block-diffusion), adapters-only, accumulation + EMA + maxlogit monitoring + verified resume.
5. Corpus: synthetic geometric scenes stratified (`--kind-weights 1,1,2,2,2`) + augmentation, then Geometry3K/GeoQA. Decontaminated; anti-cheat scan before training.
6. Inference-time knobs: refine depth (test-time compute) + diffusion denoising; vecread traces exposed for auditability.
7. Anti-cheat gates are hard: decontamination report, cheat harness commit, policy gate, audit propagation all must pass before any training run is trusted.

## Research pipeline mapping (docs/research-pipeline.md)

| Stage | Research concept | Repo machinery |
|---|---|---|
| 0 Shared vector space | VL-JEPA/CLIP + topological alignment | trunk hidden states -> geometric slots; metric `G=L Lᵀ` |
| 1 Language -> scene graph | LINC/Logic-LM pattern | `geomkernel::SceneGraph`, `parse_scene`, `ParsedScene` |
| 2 Generation | CommonScenes two-branch | `GeomFusion` structure branch + diffusion content branch |
| 3 Training efficiency | CoT distillation + Diffusion of Thought | `distill.rs`, block-diffusion objective, conditional ODE solver (8-64 steps) |
| 4 Post-training RL | GDPO | optional; deferred until base stabilizes |
| 5 Output translation | vector reasoning -> plain language | `vecread.rs` thought traces + kernel verdicts |

Eviction/durability/reasoning-recovery items from the Subnext prompt are **out of scope** for this model (they concern KV-cache runtime, not training); noted as not applicable.

## Phased tasks

### Phase 0 — Baseline & gates (no code changes)
- Run `cargo test --lib` (expect 1176 passed), `cargo clippy --all-targets`, `dblocks verify` (255/255).
- Confirm weights/tokenizer/data present per `docs/Frontier-27B-Plan.md` §0.
- Anti-cheat: run cheat scan on intended corpora; commit harness registry entry.

### Phase 1 — Wire fused CLI (code) — DONE
- `dblocks geom fused train|eval|prove` dispatches to `train_fused_qwen`/`prove` with `LmRuntimeArgs`.
- Add config validation (`FusedTrainConfig::validate`, `ProofConfig`) returning `Result` naming fields.
- `serde(default)` on any new config fields.
- Cert: add `fused_cli_args_round_trip`-style certificate if new parsing lands.

### Phase 2 — Anti-cheat + data prep (code + data)
- Decontaminate geometric corpus via `codegen_eval` n-gram decon; `mlcheat` scan; record report.
- Stratify corpus (`--kind-weights 1,1,2,2,2`) + augmentation; generate held-out independent corpus for eval (avoid augmented-copy inflation).
- `policy` gate: refusal/authorization checks on tool calls during data build; full audit logging.
- Add certificate: `fused_corpus_decontaminated_and_stratified`.

### Phase 3 — Training run (execution)
- Adapters-only fused training on stratified corpus; monitor maxlogit/balance; checkpoint every N; verified resume check.
- Hardware constraint on a 12 GB card: use the rotating-cohort MoE strategy (shared expert, rotating 2–3 expert groups, dual-method fingerprinting with agreement gate, early gatekeeping, cross-cohort replay, continuously resident router, load balancing, light calibration, cheap pilot) so only a cohort is resident at once while the router learns the full population over time. See "Hardware solution: rotating-cohort MoE training for budget GPUs".
- Run `prove` on held-out; record accuracy, margin vs chance, kernel verified/falsified/indeterminate, refine sweep.
- Refinement-depth curve must show answers change while accuracy climbs (not flat decoration).

### Phase 4 — Vector reasoning exposure (code)
- Wire `vecread` to decode fused hidden/slot states at each refine depth into thought traces; expose assumptions/evidence before final answer.
- Certify trace decode matches trunk readout (logit lens parity).

### Phase 5 — Proof & benchmark ladder (execution)
- Synthetic held-out first; then Geometry3K/GeoQA; then MathVista-GPS; then miniF2F-v2.
- Every emitted answer kernel-checked; report per-kind accuracy.
- Compare vs matched baseline (`geom eval --baseline`) and record in `docs/Claims.md`.

### Phase 6 — Optional RL (deferred)
- GDPO only after base stabilizes; requires reward tied to kernel verdicts.
- Certify variance decomposition / group sampling.

### Phase 7 — Docs updates (existing docs only)
- Update `docs/Geometric-Reasoning.md` Measured section with fused run numbers.
- Update `docs/Claims.md` rows for fused quality (move from UNKNOWN to measured).
- Update `docs/Geometric-Reasoning-Flaws.md` only if new flaws found (do not edit existing entries per concurrent owner; append new rows only).
- Update `docs/Frontier-27B-Plan.md` Phase 4 status.
- Update `docs/Home.md` feature table if new CLI lands.
- Update `TODO.md` status paragraphs.

## Anti-cheat gates (must pass, in order)

1. `./audit-bad-patterns.sh --strict` clean.
2. `cargo clippy --all-targets` no warnings-as-errors.
3. `cargo test --lib` + integration green.
4. `dblocks verify` 255+/255+ (new certs added).
5. Cheat harness registry commit + nina fleet scan green.
6. Decontamination report attached to corpus sidecar.
7. Policy gate: every tool call re-authorized with args; audit log present.
8. `audit propagation` sensitivity/amplification columns recorded.

## Risks

- Fused run unmeasured; identity-at-birth means fresh model sits at chance — training must move it; no quality claim without measurement.
- Augmented corpus inflates own held-out split; eval on independent corpus.
- Depth curve may stay flat (attractor) — that's a finding, not a bug; record it.
- Hardware throughput: plan sample counts from measured tok/s, not hope.
- New routing code must reuse certified scatter; hand-rolled scatter is a known bug class.

## Validation plan

- `make gate` green at every phase boundary.
- Certs: existing `geomfusion` group + new certs for CLI parse, corpus decontamination, fused trace decode.
- Measurements recorded in `docs/Geometric-Reasoning.md` Measured section with checkpoint hashes.
- Claims updated in `docs/Claims.md` with harness commands.

## Hardware solution: rotating-cohort MoE training for budget GPUs

A training strategy for MoE specialists on a single consumer GPU (12 GB VRAM, ~46 GB RAM). It complements the existing MoSME readout and the fused stack rather than replacing them. Nine stages:

1. **Shared expert pretraining.** Train one dense 1–3B shared expert on general data to convergence, then freeze it; it becomes the always-active backbone so routed experts do not relearn basics.
2. **Rotating expert groups.** Load 2–3 experts + the frozen shared expert + the router into VRAM, train that cohort for N steps with active routing so they differentiate, then swap the next cohort in, until every expert has trained.
3. **Fingerprinting.** Before swapping out a trained expert, extract two fingerprints: mean activation statistics over a fixed probe set (cheap gross-overlap check) and output embeddings on fixed probe prompts (deeper behavioral check). Store both in VRAM (tiny vs. weights). Flag redundancy only when *both* methods agree.
4. **Active early-training gatekeeping.** After a handful of steps on a fresh expert, fingerprint it and compare to stored fingerprints; if too similar, kill and reinitialize (stronger diversity seed, different data subset, different init) before it fully forms.
5. **Cross-cohort replay.** Every few rotations, reload one previously-trained expert alongside the current cohort for a short joint-training burst; rotate which old expert is replayed so every expert is refreshed against neighbors. Counters the amnesia pure fingerprinting cannot catch.
6. **Continuously resident router.** Keep the router resident and trained across every rotation from the start; combined with replay it observes the full population instead of a 2–3 expert slice.
7. **Load-balancing loss throughout.** Apply the load-balancing auxiliary at cohort level (Stage 2) and during calibration, penalizing uneven usage so the router does not collapse onto favorites.
8. **Light final calibration.** With all experts frozen, fine-tune only the router on held-out data; cross-expert awareness was already built during training.
9. **Cheap pilot validation.** Run the whole pipeline on 3–4 experts + a small held-out task and measure whether they show genuinely distinct specialization vs. training the same 3–4 fully resident together; catch failure cheaply before the full run.

Mapping to existing machinery: Stage 7 reuses the existing MoE load-balancing / z-loss (`moe::balance_loss`, `HierarchicalGates::balance_loss`); Stage 6 reuses the resident router; fingerprint storage is small (activation stats + probe embeddings) and fits the existing checkpoint sidecar pattern. Stage 9's pilot is the gate before committing compute. Status: mechanism implemented in `src/cohort.rs` (pure logic + `cohort` cert group, 7 certs); GPU training loops that consume it remain blocked on weights/compute.

- Refine depth default for fused (standalone uses trained depth; fused unmeasured).
- MoSME box count for reasoning kinds (start 1 box × N experts, identity at birth).
- Whether to enable diffusion objective in first fused run (answer-only first, then diffusion).

## Out of scope

- Vision tower, MTP heads, 256K context, Muon, GGUF export (separate phases).
- KV-cache eviction/durability (Subnext runtime concerns, not model training).
- Distributed training, hosted serving.
- New `.md` docs (only updates to existing).
</content>