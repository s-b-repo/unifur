# Geometric Reasoning: Flaws and Their Dispositions

Every known flaw, failure mode and open weakness of the geometric reasoner,
each with its current status and the artifact — certificate, test, flag or
measurement — that covers it. Companion to [Geometric Reasoning](Geometric-Reasoning.md).

> **In this repository.** Certificates: the `geom` group in `src/verify.rs`
> (18). Unit tests: `src/geometry.rs`. Baseline harness: `src/geombaseline.rs`,
> `dblocks geom baseline`, `dblocks geom eval --baseline`. Bug history: the
> `TODO.md` bug table.

---

## Fixed, and pinned so they stay fixed

| Flaw | What happened | Pinned by |
|---|---|---|
| The repulsion coefficient `λ` vs `2λ` | `energy_grad_step` descended a *different* quadratic than the `energy_value` it reported: the ordered-pair sum counts each unordered pair twice, so the gradient carries `2λ` where the step applied `λ`. The monotone-descent certificate passed or failed by the draw (the group went red ~1 run in 3); a small `η` hid the mismatch | `test_the_step_is_the_gradient_of_the_reported_energy` (finite-differences the reported energy against the step's own direction; fails at rel. error 1.90 if the coefficient drifts), `geom/energy_descent_is_monotone_under_the_certified_step` |
| The gate scatter that summed to `n_boxes` | The readout's hand-rolled `[b,k]`-vs-`[b,n]` comparison broadcast the single top-1 gate across all columns before summing: every row summed to `n_boxes` instead of 1, logits ran hot, entropies read −3.0, learning stalled at chance | The readout now routes through `mosme::HierarchicalRouter` — no hand-rolled scatter remains — and `geom/readout_gates_form_a_distribution` (load sums to 1, entropies in [0,1]) plus `geom/readout_inherits_mosme_invariants` (a disabled expert's composed gate is exactly 0.0, untouched boxes bit-identical, rows still sum to 1) |
| A second routing implementation | The readout's router was a hand-rolled duplicate of MoSME's two-level routing, free to drift from the certified original (and it had: the scatter bug) | Rewired onto `mosme::HierarchicalRouter`; partition of unity is now *inherited* from `mosme::composed_gates_partition_of_unity` rather than re-proven per copy. The expert heads stay local — per the MoSME design, what is shared is the router, not the expert |
| inside/collinear scenes labeled points they did not show | The generator rendered scene tokens *before* the inside/collinear constructions overwrote `coords[0..]`: every such scene stored an answer about different points than the tokens described (decode check: a scene claiming `(10,8)` inside triangle `(6,0)(2,4)(0,2)`). Found by the augmentation work in September 2026. Every measurement taken before the fix trained partly on these mislabeled scenes, including the RTX 3060 demo numbers in [Geometric Reasoning](Geometric-Reasoning.md) | Rendering now happens after the coordinates are final (`render_scene`); construction-failure fallbacks take the recomputed label. Pinned by the end-to-end decode pass inside `geom/augmentation_preserves_labels` (regenerate a corpus, decode, recompute every stored answer) |

## Bounded, with the bound certified

| Flaw | Disposition | Pinned by |
|---|---|---|
| Metric temperature drift | Jointly learned metrics have no large-scale stabilization report in the literature; the known failure is silent drift (collapse or blowup) with vanishing gradients. `bounded_temperature` clamps to `[1e-3, 2·dim]`: the floor keeps `-d²/τ` off a zero divisor, the cap keeps runaway drift from flattening attention into uniformity. Clamp, not remap, so checkpoints stay bit-identical | `geom/attention_temperature_is_bounded` |
| `σ_data` mismatch across commands | EDM's 0.5 default is an image number; geom checkpoints carry no training state, so `--sigma-data` must be set consistently across `geom train`, `eval` and `generate` by hand | The flag exists on all three subcommands and the docs say so; there is no mechanism enforcing agreement — this stays a discipline, not a certificate |
| Truncation faithfulness of extra depth | Accuracy that stays flat with depth while answers stay identical to depth 1 means extra relaxation is post-hoc decoration (the Anthropic CoT lesson) | `geom eval --refine-sweep` prints **agree-with-depth-1** per depth on the same seeded scenes; the measured curve rises to the trained depth and flattens with agreement near 1, which is the attractor showing itself rather than a bug |

## Open, with the harness or the reason named

| Flaw | Status |
|---|---|
| Corpus starvation on hard kinds | AlphaGeometry generated 1B diagrams but only 9% of the kept 100M needed auxiliary constructions; uniform sampling starves the hard cases. `--kind-weights` oversamples them and `geom data` prints the per-kind balance into the sidecar. The default stays uniform round-robin deliberately (TongGeometry's value-filtering lesson needs a prover loop that is not built) |
| `direction` caps augmentation | Its answer is an *absolute* compass bearing (8 sectors clockwise from north), so rotation rotates the label. `direction` scenes get translation + uniform scale only; every other kind gets the full similarity. Enforced in `augment_coords` and asserted by `geom/augmentation_preserves_labels` |
| An augmented corpus inflates its own held-out split | Augmented copies are emitted beside their originals, so `geom eval` on the same corpus draws near-duplicates of training scenes: the stratified+augmented run scored 0.3457 on its own corpus but 0.2949 cross-evaluated on the independent uniform corpus — statistically identical to the uniform-trained 0.2920. At the 3000-step scale, stratification + augmentation bought no generalization (measured; see the Measured section of [Geometric Reasoning](Geometric-Reasoning.md)). Evaluate augmented training runs on an independent corpus |
| Learned metric vs flow matching | Riemannian flow-matching assumes a *fixed* metric; geodesics and kernels drift with a learned one. The block-diffusion objective sidesteps this by working in the latent directly, but a learned-metric flow objective is unbuilt and unmotivated so far |
| No auxiliary constructions, synthetic ceiling | The corpus is fixed-width point scenes; there is no prover loop, no auxiliary-point construction, no real benchmark. Eval order when leaving synthetic data: Geometry3K/GeoQA first (small models score 22–64%), then MathVista-GPS, then miniF2F-v2, then ProofNet. Not ARC-AGI (spatial abstraction, not proof) |
| Attractor relaxation vs a matched-depth transformer | The quality claim this module is for. Harness: `src/geombaseline.rs` — a pre-norm transformer encoder at `num_blocks × refine_steps` layers, same embedding, same hidden size, no position table, trained on the same corpus with the same batching, optimizer and held-out split, scored by literally the same `answer_ce` code. `dblocks geom baseline` trains it, `dblocks geom eval --baseline` prints the side-by-side. Result: recorded in [Claims](Claims.md) and the Measured section of [Geometric Reasoning](Geometric-Reasoning.md) |
| Checkpoint fragility | Geom checkpoints carry no training state, so any record-layout change (e.g. the MoSME rewire) invalidates them silently until load. Accepted: retraining is minutes on a GPU; the pre-rewire `checkpoints/geom-*.mpk` no longer load and are kept for provenance only |
| Parallel-suite test flake | The process-global NdArray RNG is seeded per test, so concurrently running tests can draw between the seed and the model init: `integration_geometric_block_diffusion_trains_and_denoises` can fail `last < first` under the full parallel suite while passing standalone (pre-existing, reproduced against the pre-rewire router). Fix direction: serialize that test or make its assertion robust to the draw |

---

See also: [Geometric Reasoning](Geometric-Reasoning.md) · [Claims](Claims.md) ·
[Mixture of Specialized Micro Experts](Mixture-of-Specialized-Micro-Experts.md) ·
[Quality Gate](Quality-Gate.md) · [Home](Home.md)
