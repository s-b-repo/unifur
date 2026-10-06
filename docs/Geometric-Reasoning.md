# Geometric Reasoning over a Learned Riemannian Geometry

A dual-stream reasoner that answers geometric questions by relaxing an
explicit energy to a certified attractor, the block-diffusion objective that
trains the relaxation, and the exact rational kernel that stands beside the
model as proof (roadmap Phase 33).

> **In this repository.** `geometry.rs` (`GeometricReasoner`, `MetricField`,
> `GeomBlock`, `GeomRouter`, `geodesic_attention`, `energy_value`,
> `energy_grad_step`, `lipschitz_step`, `generate_corpus`,
> `sample_aligned_batch`, `train_geom`, `evaluate_geom`, `GeomObjective`,
> `GeomTrainConfig`, `GeomTrainReport`), `geomkernel.rs` (the exact
> rational kernel: `Q`, `Constraint`, `SceneGraph`, `graph_from_points`,
> `construct`, `rule_saturate`, `falsify`), `geombaseline.rs` (the
> matched-depth transformer baseline). Wiring: `lib.rs` (`geometry`,
> `geomkernel`, `geombaseline`), `main.rs` (`dblocks geom …` with wgpu
> dispatch). Certificates: the `geom` group (18). Tests: `tests/integration.rs`
> (`integration_a_geometric_reasoner_trains_and_answers`,
> `integration_geometric_block_diffusion_trains_and_denoises`).
> Every known flaw and its disposition: [Geometric Reasoning
> Flaws](Geometric-Reasoning-Flaws.md).

---

## The policy: answers from geometry, not from scale

A conventional transformer earns its answers from scale — more layers, more
data, more compute per token. This module is the other policy. The model
builds an explicit geometric description of a scene (entities as points in a
learned Riemannian space, relations as metric distances) and answers by
*relaxing* that description into an attractor basin: a fixed point of an
explicit energy, reached by a certified descent.

The learned parameters shape the landscape — the metric, the attention, the
writers. The reasoning itself is fixed geometric dynamics whose depth and
precision are **inference-time knobs**, so the answers come from the geometry
rather than from brute-force training.

Four ideas this phase implements, all from Michels' papers:

- **Attractor State.** The answer state is a fixed point of an explicit energy
  `E`; relaxation descends `E` with a step size derived from a closed-form
  Lipschitz bound, so monotone decrease and convergence are *certificates*
  (the `geom` group), not assumptions.
- **Rule by Technocratic Mind Control.** The model is its own verifier. No
  external reward model grades the answer; the readout consumes the state the
  geometry certifies (final energy, displacement, convergence), and evaluation
  reports those diagnostics beside accuracy.
- **The Dissolution of a False Divide.** The scene is carried as two
  complementary descriptions of one state — a symbolic stream (discrete
  tokens) and a geometric stream (continuous points). Each refines the other
  through geodesic attention under the shared metric; the landscape is fixed
  within a block and redrawn from the refined symbolic stream by the next, so
  the two descriptions stay two views of one recursive trajectory.
- **Principia Cybernetica II.** Information geometry. Distances are computed
  under a learned metric `G = L Lᵀ` with `L` parameterized so `G` is positive
  definite by construction; attention is inverse distance under `G`
  (geodesic attention); and the block-diffusion machinery runs on that metric
  space.

### The learned Riemannian metric

`MetricField` holds `G = L Lᵀ` with `L` lower-triangular and a positive
diagonal **by construction**: the strictly lower part and the log of the
diagonal are the free parameters, so `G` is positive definite for *any*
parameter values the optimizer can reach. A fresh metric is the identity.
Every distance in the module — the attention logits and the energy — is
computed under this one `G`.

### Geodesic attention

`geodesic_attention` softmaxes the *negative* metric distance, so attention
weights are relations read from the learned geometry rather than arbitrary
learned logits. The temperature is `softplus(raw) + 1e-3`, learned per block.

### Certified attractor relaxation

The energy over the geometric stream, against a fixed context:

```
E(z) = Σ_k ‖z_k − c_k‖²_G  +  λ Σ_{k≠l} ‖z_k − z_l‖²_G
```

Attachment draws each slot toward its context; repulsion (`λ`, learned per
block) keeps slots from collapsing onto one point. Relaxation is gradient
descent in the metric, `z ← z − 2η G((z − c) + λ(Kz − Σz))`, with the step
size from a **closed-form Lipschitz bound** `L = 2(1 + 2λK)·σ_max(G)²` and a
`σ_max` estimated by power iteration and inflated 10% so the step stays
strictly inside the descent lemma's stability range. `E` is a
positive-definite quadratic, so under that step the energy never increases —
that monotone decrease is the `energy_descent_is_monotone_under_the_certified_step`
certificate.

### The MoSME readout

The query state is read out through boxes of specialized expert heads (one
box per question kind) with two-level sparse routing — the crate's real
`mosme::HierarchicalRouter` (`route_on_tokens = false`, so the query state is
the router input), not a local copy of it. The composed gates are a partition
of unity by inheritance (`mosme::composed_gates_partition_of_unity`), the
balance loss is `HierarchicalGates::balance_loss` with the z-loss carried
separately at its own level (never scaled by the balance weight), and the
expert heads themselves stay local: per the MoSME design, what is shared is
the router, not the expert. `--moe-boxes 0` uses the plain linear readout —
the certified identity setting.

## Block diffusion

The reasoner is a block-diffusion model of the attractor state, conditioned on
the scene. The sigma trajectory is partitioned into `num_blocks` blocks (block
0 noisiest, the `sigma::block_sigmas` convention). In the `diffusion`
objective, block `b` is trained standalone on the clean state corrupted at its
window's noise scale, EDM-preconditioned (`sigma::EdmPreconditioning`), with
boundary consistency at the shared sigma between adjacent blocks — the exact
training scheme of the DiffusionBlocks core, applied to a geometric latent.
Inference runs the same trajectory at any noise scale: the clean path (scale
zero) is the fast reasoning path, and `generate --diffusion` starts from pure
noise and denoises into the scene's basin with the number of refinement steps
a runtime knob.

## The exact kernel beside it

`geomkernel.rs` is a deterministic kernel over **exact rationals** (`Q`). It
saturates a scene graph's facts through deduction rules to a fixed point, and
`falsify` hunts randomized counterexamples to a geometric claim. The kernel is
the source of mathematical truth: the model *proposes*, the kernel *proves*.
`dblocks geom solve` and `dblocks geom counterexample` never load a neural
model. Rationals are the point: `1/3 + 1/6` is exactly `1/2`, an identity a
float kernel fails.

## CLI

```bash
# Write the synthetic geometric-question corpus + metadata sidecar.
# --kind-weights oversamples hard kinds (default: legacy round-robin).
# --augment M adds M similarity copies per scene (rotation + translation +
# uniform scale, labels recomputed on the grid; direction scenes get
# translation + scale only, their answer being an absolute compass bearing).
dblocks geom data --points 6 --count 4096 --seed 7 --out-dir geom
dblocks geom data --points 6 --kinds nearest,farthest,direction,inside,collinear \
  --kind-weights 1,1,2,2,2 --count 8192 --seed 7 --out-dir geom-hard

# Train the clean reasoning path (answer objective) on a GPU.
dblocks geom train --corpus geom/corpus.bin --meta geom/meta.json \
  --steps 2000 --objective answer --backend wgpu --device discrete:0

# Train the block-diffusion objective (sigma windows, EDM, boundary consistency).
dblocks geom train --corpus geom/corpus.bin --meta geom/meta.json \
  --steps 2000 --objective diffusion --backend wgpu

# Train the matched-depth transformer baseline (num_blocks * refine_steps
# layers, same hidden/data/optimizer; prints both parameter counts), then
# compare on the same held-out scenes.
dblocks geom baseline --corpus geom/corpus.bin --meta geom/meta.json --steps 2000
dblocks geom eval --checkpoint checkpoints/geom-XXX.mpk \
  --baseline checkpoints/geom-baseline-YYY.mpk --corpus geom/corpus.bin --meta geom/meta.json

# Evaluate held-out accuracy; --refine-sweep draws the depth scaling curve.
dblocks geom eval --checkpoint checkpoints/geom-final.mpk \
  --corpus geom/corpus.bin --meta geom/meta.json --refine-sweep

# Answer one scene; --diffusion denoises from pure noise (needs a diffusion ckpt).
dblocks geom generate --scene "..." --checkpoint checkpoints/geom-final.mpk

# The exact kernel: no model loaded.
dblocks geom solve --graph figure.json
dblocks geom counterexample --claim claim.json --trials 4096
```

Both backends dispatch through the same `LmRuntimeArgs` as the language
trunk, so `--backend cpu|wgpu` and `--device discrete:N` work identically.

## The two streams

| Stream | Carries | Refined by |
|---|---|---|
| Symbolic | `scene_len - 1` byte tokens (header, points, query `?`) | The geometric stream writes back each step (`writer_in` → `writer_out`) |
| Geometric | `slots` concept points in a learned `dim`-space | A certified energy relaxation; slot `k` starts near the chunk it was pooled from |

There is **no position table**. The scenes are fixed width, the fixed layout
*is* the position system, and the geometry carries the structure — so the
readout is a single query at the last token the model reads (the `?`).

## Certificates (`geom` group, 18)

| Certificate | Claim |
|---|---|
| `kernel_rationals_are_exact_where_floats_are_not` | `1/3 + 1/6 = 1/2` exactly, where floats are not |
| `kernel_intersection_is_exact` | Segment intersection is exact in rationals |
| `kernel_refuses_degenerate_configurations` | Degenerate figures are refused, not silently accepted |
| `kernel_rules_derive_with_certificates_and_saturate` | Derived facts carry their rule and reach a fixed point |
| `kernel_falsification_rejects_and_survives_correctly` | Falsify finds a real counterexample and survives a true claim |
| `metric_is_positive_definite_by_construction` | `G = L Lᵀ` is PD for any parameters |
| `geodesic_attention_rows_are_distributions` | Inverse-distance attention rows sum to 1 |
| `attention_temperature_is_bounded` | The learned temperature stays in `[1e-3, 2·dim]` |
| `energy_descent_is_monotone_under_the_certified_step` | The Lipschitz step never increases `E` |
| `relaxation_converges_to_the_exact_fixed_point` | 96 certified steps land within 1% of the exact fixed point |
| `with_repulsion_off_the_attractor_is_the_context` | With `λ → 0` every slot lands on its own context |
| `block_span_covering_everything_is_the_full_path` | A span covering every block reproduces the forward pass bit for bit |
| `denoiser_is_identity_at_sigma_zero` | At `σ = 0` the diffusion path returns its input |
| `a_single_box_single_expert_mixture_is_that_expert` | A 1×1 MoSME mixture is exactly that expert (the routing identity) |
| `readout_gates_form_a_distribution` | Readout load sums to 1, entropies in [0,1] |
| `readout_inherits_mosme_invariants` | A disabled expert's composed gate is exactly 0, untouched boxes bit-identical |
| `augmentation_preserves_labels` | Similarity copies keep their labels (direction: translation + scale only) |
| `constraint_evaluation_matches_the_scene_graph` | The kernel's constraint evaluation agrees with the scene graph |

### The `2·λ` that had to be right

`energy_grad_step` originally applied the repulsion term with coefficient `λ`
where the energy's is `2λ`: the ordered-pair sum `Σ_{k≠l}‖z_k − z_l‖²_G`
counts each unordered pair twice, so it differentiates to `4G(K z_k − Σz)`.
The step was therefore descending a *different* quadratic than the
`energy_value` it reported, and the monotone-descent certificate passed or
failed depending on the draw — over 400 random landscapes the step raised the
energy it claimed to lower by up to **0.34**, and the group went red about one
run in three. The monotone test had passed by luck, because a small `η` hides
the mismatch.

So the step is now `z ← z − 2η G((z − c) + 2λ(K z − Σz))`, the fixed point moved
with it to `(c + 2λ·Σc)/(1 + 2λK)`, and
`test_the_step_is_the_gradient_of_the_reported_energy` finite-differences the
reported energy against the step's own direction — it fails at relative error
1.90 if the coefficient ever drifts again. This is the same discipline the
`TODO.md` bug table applies to the rest of the crate: a certificate that
merely observes a quantity is weaker than one that ties two quantities
together.

### The scatter that summed to `n` instead of 1

The MoSME readout hand-rolled its gate scatter as a `[b,k]`-vs-`[b,n]`
comparison followed by a sum. With top-1 routing over `n` boxes that
broadcasts the single gate across all `n` columns and then adds them, so
every row summed to `n_boxes` (3.0 in the probe) instead of 1: the mixture
logits ran 6× too hot, the entropies read −3.0 (impossible for a
distribution), the balance read ~15, and the saturated readout stalled
learning at chance. The trunk's `moe::scatter_gates` does the same job
correctly (unsqueeze, compare, sum over K). The readout has since been rewired
onto the crate's real two-level router, `mosme::HierarchicalRouter`, whose
scatter, balance breakdown and z-loss are the certified ones, so there is no
longer a second routing implementation to drift. Pinned by
`geom/readout_gates_form_a_distribution` (load sums to 1, entropies in
[0,1]), by `geom/readout_inherits_mosme_invariants` (the router's mask
invariant holds through the readout), and by the integration test, which
trains through `train_geom` to above chance on nearest-4.

## Measured: RTX 3060 runs

Backend `wgpu` / Vulkan on an NVIDIA GeForce RTX 3060 (driver 615.71.09),
`--backend wgpu --device discrete:0`. All scenes are 40 tokens, 6 points, all
five question kinds (16 answer bytes, so chance is 6.25%). Held-out numbers
are the seed-42 protocol (1024 scenes, `geom eval`'s default).

> The pre-rewire demo numbers that used to live here (0.3203 answer objective,
> 0.2969 diffusion objective) were trained on the corpus **before** the
> inside/collinear label fix (see
> [Flaws](Geometric-Reasoning-Flaws.md)) and with the hand-rolled router; their
> checkpoints no longer load. They are superseded by the runs below.

**Answer objective, rewired MoSME readout**, 3000 steps, batch 128, lr 2e-3,
uniform 8192-scene corpus: loss 2.7884 → 1.3823, train-batch accuracy 0.0547 →
0.2812 (mean 0.2974), router healthy throughout (load H ~0.99, balance ~2.0).
Held-out accuracy **0.2920** (4.7× chance). Checkpoint
`checkpoints/geom-87da2ead3c55a751.mpk`.

**Matched-depth baseline** (`geombaseline.rs`, 4 layers = blocks × refine
steps, same embedding/hidden/corpus/optimizer, 233,475 params vs the
reasoner's 242,630), same 3000-step budget: held-out **0.3115**. The baseline
**beats the reasoner at this scale** — the "attractor relaxation beats a
matched-depth transformer" claim is REJECTED at this budget in
[Claims](Claims.md), and the harness (`geom baseline`, `geom eval --baseline`)
is the rerun path for any larger budget.

**Refinement-depth scaling curve** — accuracy against relaxation depth on the
*same* weights:

| refine depth | accuracy | agree-with-depth-1 |
|---|---|---|
| 1 | 0.2969 | 1.0000 |
| 2 | 0.2920 | 0.2891 |
| 3 | 0.2998 | 0.0967 |
| 4 | 0.2891 | 0.0967 |

The curve is flat past the trained depth — extra relaxation buys nothing,
which is the attractor showing itself. This time agreement is **low** (0.29 at
the trained depth): deeper refinement answers *differently* on most scenes
while getting no more of them right, so the depth reshuffles errors rather
than decorating a fixed answer.

**Stratified + augmented corpus** (`--kind-weights 1,1,2,2,2 --augment 2`,
24,576 scenes), same 3000-step budget: 0.3457 evaluated on its own corpus —
but that split draws augmented copies of training scenes, so it inflates.
Cross-evaluated on the independent uniform corpus: **0.2949**, statistically
indistinguishable from the uniform-trained 0.2920. At this scale the
stratification + augmentation bought no generalization — measured, not
assumed, and the own-corpus inflation pitfall is now in the flaws doc.

**Block-diffusion objective** (pre-rewire measurement, kept for the
convergence counter): 1500 steps, loss 7.7088 → 3.0640, accuracy 0.0000 →
0.2969, **482 / 1500 steps converged** (final displacement under tolerance) —
the counter the "model is its own verifier" claim rests on. Its checkpoint
predates the rewire and no longer loads; rerun with `--objective diffusion`
for current weights.

## Field notes: trained geometric models (what ports, what does not)

- **Stratify the corpus, don't just scale it.** AlphaGeometry generated 1B
  diagrams but only 9% of the kept 100M needed auxiliary constructions — the
  model starves on hard cases under uniform sampling. `--kind-weights` exists
  for exactly this (oversample `direction`/`inside`/`collinear`), and `geom
  data` prints the per-kind balance into the sidecar. TongGeometry's further
  lesson (symmetry + value filtering of generated theorems) needs a prover
  loop and is not built; uniform-round-robin stays the default.
- **The temperature is bounded on both sides.** Metric-learning literature
  has no large-scale stabilization report for jointly learned metrics; the
  known failure is drift (collapse or blowup) with gradients that vanish
  instead of erroring. `bounded_temperature` clamps to `[1e-3, 2·dim]`:
  the floor keeps `-d²/tau` off a zero divisor, the cap keeps runaway drift
  from flattening attention into uniformity. Clamp, not remap, so existing
  checkpoints are bit-identical. Certified as
  `geom/attention_temperature_is_bounded`.
- **Match `sigma_data` to the latent scale** (EDM hygiene from the diffusion
  labs: the 0.5 default is an image number). `--sigma-data` on
  `geom train|eval|generate` sets the preconditioning scale; it must agree
  across the three because geom checkpoints carry no training state.
- **Keep the kernel beside the model.** AlphaGeometry, DeepSeek-Prover
  (Lean binary reward + RMaxTS/GRPO) and LeanWorkbook all converge on the
  same split: the neural net *proposes*, a sound verifier *rewards/filters*.
  Never replace the verifier reward with the denoising loss alone; never
  train on unfiltered autoformalizations.
- **Supervise the trajectory, not the endpoint.** DEQ practice (Jacobian
  regularization, iteration caps, residual logging) and the consistency-model
  warning (CT loss values are meaningless for selection) are already this
  module's shape: energy/displacement traces, `converged_steps`, tolerance
  stops. Do not early-stop on the loss value.
- **What does not port (yet):** learned-coordinate scenes have no precedent
  (AlphaGeometry feeds its LM symbolic DSL, never raw coordinates);
  Riemannian flow-matching assumes a *fixed* metric (geodesics/kernels drift
  with a learned one); strict SE(3)-equivariance was dropped at scale
  (AlphaFold3) for augmentation — augmentation is now built (`--augment`) and
  was measured at the 3000-step scale: no generalization gain (Measured
  above), so treat it as an open question, not a solved one.
- **Eval order when leaving synthetic data:** Geometry3K/GeoQA first (small
  models score 22–64% there today), then MathVista-GPS, then miniF2F-v2,
  then ProofNet. Not ARC-AGI (spatial abstraction, not proof), not
  MATH-Vision/PutnamBench until tier 1 clears 60%.

### What this does and does not show

It shows the mechanism trains end to end on real hardware and that the
attractor behaves as the certificates say — depth changes the answers up to
convergence and then buys nothing. It does **not** show that attractor
relaxation beats a matched-depth transformer: at the measured budget (3000
steps, 8192 scenes) the matched baseline wins, 0.312 to 0.292, recorded as
REJECTED-at-this-scale in [Claims](Claims.md) with `geom baseline` /
`geom eval --baseline` as the rerun harness for any larger budget.
