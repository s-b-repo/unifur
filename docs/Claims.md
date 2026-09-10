# Claims

Every load-bearing claim in this repository, classified the way issue #1 asks:

| Status | Meaning |
|---|---|
| **VERIFIED** | Reproduced: a certificate in `dblocks verify`, a test, or a measurement recorded in `TODO.md` |
| **PLAUSIBLE** | Mathematically or empirically motivated, not yet demonstrated here |
| **REJECTED** | Tested here and shown not to do what was hoped |
| **UNKNOWN** | Insufficient evidence; the harness that would settle it is named |

A mechanism being VERIFIED means it does what its documentation says, to a
stated tolerance. It does **not** mean it improves accuracy: on this crate's
hardware nothing has been trained long enough for a quality claim, and every
quality claim below is UNKNOWN with its harness named. The governing rule is
the one the issue states: an optimization hypothesis does not become a fact
without a measurement.

> **In this repository.** Certificates: `src/verify.rs` (`dblocks verify`,
> 19 groups). Measurements: the `Status` paragraphs of `TODO.md`. Harnesses:
> `dblocks sweep`, `dblocks bench --json`, `dblocks audit propagation`,
> `dblocks lm bench`, `dblocks experiment compare`.

---

## Mathematical results

| Claim | Status | Evidence / assumption |
|---|---|---|
| Block-wise dynamics are a reverse-ODE discretization (Thm 1) | PLAUSIBLE | Interpretation; the schedule/preconditioning identities it relies on are VERIFIED (`schedule`, `preconditioning` groups) |
| Tweedie's formula (Thm 2) | VERIFIED | Classical; the EDM denoiser identity `D(z) = x` is certified (`preconditioning::edm_denoiser_identity`) |
| Score-matching loss → optimal denoiser in `L²` (Thm 3) | PLAUSIBLE | Standard for the idealized objective; nothing here measures the trained trunk against `D*` |
| Compositional losslessness (Prop 4) | PLAUSIBLE | **Corrected**: bound is `(Σ√δ)² ≤ BM·Σδ`, not `Σδ`, and needs 1-Lipschitz blocks. `dblocks audit propagation` measures the sensitivity proxy that assumption needs |
| Block-wise error propagation (Prop 5) | PLAUSIBLE | Holds *given* per-block Lipschitz constants; `L ≈ 1` is a heuristic. Same harness |
| Parallel training convergence rate (Heur 6) | UNKNOWN | **Relabeled**: strong-convexity rate does not apply to a neural loss. What survives is the `K/B` update-fraction bookkeeping. Harness: `dblocks sweep --grid num_blocks=2,3,4` (GPU) |
| Consistency loss bound (Prop 7) | VERIFIED | Elementary inequality; bounds agreement by accuracy, not accuracy by agreement |
| `w(σ)·c_out² = 1`, `c_in²(σ²+σ_d²) = 1` | VERIFIED | `preconditioning` group |
| Sigma windows tile `[σ_min, σ_max]`, CDF-uniform, train/inference agree on ownership | VERIFIED | `schedule` group; the ownership bug this found is in the `TODO.md` bug table |
| Solvers achieve their classical order; kernel moments; DDIM preserves variance | VERIFIED | `solver` group |
| `erf + erfc = 1`, `ppf ∘ cdf = id` | VERIFIED | `stats` group (found the `erf` crossover bug) |

## Mechanisms (do what they say)

| Claim | Status | Evidence |
|---|---|---|
| Uncertainty weighting is minimized at `l = ln L` and its gradient is scale-free | VERIFIED | `optim` group, measured through `apply` |
| Importance sampling is unbiased with weights bounded by the smoothing floor | VERIFIED | `optim` group, measured through `sample` |
| Gradient accumulation over `k` micro-batches equals one `k×` batch | VERIFIED | `optim::accumulation_equals_one_large_batch` (real gradients, after the `--accumulate` bug) |
| EMA is a convex combination that never extrapolates | VERIFIED | `optim` group |
| MoE gates are a distribution; Switch loss in `[1, E]`; z-loss bounds the largest logit | VERIFIED | `moe` group |
| Routing entropies read as specified, incl. the balanced-but-hedging case | VERIFIED | `moe::routing_entropies_read_as_specified` (Phase 23.6) |
| Global-batch load with a window of one is the fused loss bit for bit; a two-batch window allows cross-batch specialization | VERIFIED | `moe` group (23.4) |
| A zero or uniform selection bias is a bitwise identity; bias steers selection, not gates; one nudge moves exactly the rate | VERIFIED | `moe` group (23.5) |
| One box reduces MoSME to flat MoE exactly; hot swap is bit-identical; disabled experts contribute exactly 0 | VERIFIED | `mosme` group |
| Tokenizer is lossless; causal attention leaks exactly nothing; KV cache equals recompute | VERIFIED | `lm` group |
| Unlikelihood objective: zero charge reproduces the plain loss bit for bit; each target in exactly one sum; penalized step ends below plain step | VERIFIED | `antipattern` group |
| Planner budget is never exceeded; depth 0 is greedy; only the first step is committed | VERIFIED | `planner` group |
| Guidance at scale 1 is bitwise identity; logit normalization preserves the arg-max; ensembles emit distributions | VERIFIED | `accuracy` group |
| NF4 error ≤ half the widest level gap; LoRA is the identity at init | VERIFIED | `quantize` group |
| bf16/f16 emulation: relative error ≤ 2⁻ᵖ **and** rounding actually rounds | VERIFIED | `precision` group (the one-sided bound was a mutation-testing finding) |
| ACT weights are a partition of unity; the loop-graph budget is hard | VERIFIED | `loopgraph` group |
| Coalesced positional reads: a 16-record contiguous span costs one syscall | VERIFIED | `rawdata` test |
| A resumed run is bit-identical to an uninterrupted one (weights, EMA shadow, logged losses) | VERIFIED | `integration_resume_is_bit_identical_for_the_*_trainer` (Phase 28) |
| A corrupted checkpoint file is refused by name | VERIFIED | same test |
| Experiment intervals are the t interval on the mean; more trials narrow it | VERIFIED | `experiment` group |
| Dataset/corpus mixtures follow their weights and composites slice exactly; a one-source mix is the plain source | VERIFIED | `multisource` group (Phase 29) |
| Multi-teacher distillation reduces to single-teacher distillation for one teacher; the mixture target is a distribution | VERIFIED | `multisource` group |
| A negative teacher below its confidence is the plain loss bit for bit and never contradicts the corpus; its step lowers its proposals relative to a plain step | VERIFIED | `multisource` group |
| Checkpoint merging is the identity on equal inputs and linear in its weights | VERIFIED | `multisource` group |
| The certificate suite catches plausible defects | VERIFIED, with limits | Mutation sweep: 5 of 9 mutants survived until the certificates were rewritten to call the code; see `TODO.md` |

## Measurements made here (CPU, ≤ 400 steps)

| Claim | Status | Where |
|---|---|---|
| Uncertainty weighting cuts gradient magnitudes ~16× and the gate rejections to 0 on a 3-block run | VERIFIED (this setting) | `TODO.md` Phase 20 |
| Uncertainty weighting closes the block-2 gradient imbalance | REJECTED at 400 steps (18–41× remains) | `TODO.md` Phase 20 |
| Router z-loss lowers block-2 peak gradient (14%) | VERIFIED (small) | `TODO.md` Phase 23 |
| Router z-loss removes the gate rejections | REJECTED | `TODO.md` Phase 23: they come from `w(σ)`, not logit drift |
| Balance-weight annealing changes anything at 400 steps | REJECTED (no effect observed) | `TODO.md` Phase 23 |
| Negative supervision drives a labeled idiom >20× down while the clean tokens still learn | VERIFIED (tiny model) | `integration_negative_supervision_unlearns_error_swallowing`, `TODO.md` Phase 24 |
| Plain training *learns* `except: pass` | VERIFIED | same |
| Loss-free bias balancing / global-batch load on the 400-step MoSME run | see `TODO.md` Phase 23 status | measured with `dblocks train`; entropies reported |

## Quality claims (all GPU-blocked)

Every row below is UNKNOWN. The harness column is a single command once
hardware exists; the records it writes carry environment, config, seeds and
raw trials.

| Claim | Harness |
|---|---|
| Parallel `K = 2, 3` reaches equal quality with less compute or wall-clock than `K = 1` | `dblocks sweep --grid num_blocks=1,2,3 --seeds 1,2,3 --dataset cifar100`, then `dblocks bench --json` for inference cost |
| Any recommended learning rate, batch size, block count, `gamma`, consistency weight, balance weight, `z_level` | `dblocks sweep --grid lr=1e-4,3e-4,1e-3 ...` — every default in `TrainConfig` is **untuned**; none has been swept |
| Consistency training improves end-to-end quality (vs merely making blocks agree) | `dblocks sweep --grid consistency=0,0.01,0.1,1` |
| Overlap `gamma` buys enough boundary consistency to justify its compute | `dblocks sweep --grid gamma=0,0.05,0.1` + `dblocks audit propagation` for the boundary column |
| Flow matching vs EDM objective at equal compute | `dblocks sweep --grid objective=dblock,flow` |
| MoE expert counts improve quality per FLOP; global-batch load / loss-free bias improve specialization | `dblocks sweep --grid moe_experts=2,4,8 balance_scope=micro,global bias_balance_rate=0,1e-3`; routing entropies are in every record's log |
| Adaptive depth / early exit is a Pareto improvement | `dblocks bench --json` (scaling curve with the Pareto frontier marked) on a trained checkpoint |
| Precision policy (bf16 high-σ, f32 low-σ) preserves accuracy | `dblocks sample --precision` on a trained checkpoint; emulation only, no speed claim |
| Block distillation compensates for fewer blocks | `dblocks train --objective distill --teacher` on a trained teacher, then `dblocks bench` |
| Solver choice changes accuracy (not just agreement with Euler) | `dblocks bench --json` on a trained checkpoint |
| A locally good block cannot damage end-to-end quality | `dblocks audit propagation --checkpoint` — the sensitivity and amplification columns |
| Every Phase 25–27 mechanism (routing state, attention modes, MoVA, adaptive MTP, token-level exits) improves quality per active FLOP | `dblocks lm bench --json` per axis, on a real corpus and a GPU |

## Out of scope, and why

Distributed training (needs a cluster), native bf16 (backend), hosted W&B and
HTTP serving (network dependency tree), pretrained-weight loading (safetensors /
GGUF and a real tokenizer), self-conditioning (breaks existing checkpoint
records). Each is listed with its blocker in `TODO.md`.

---

See also: [Mathematical Foundation](Mathematical-Foundation.md) · [Quality Gate](Quality-Gate.md) ·
[Training Guide](Training-Guide.md) · [Configuration](Configuration.md) · [Home](Home.md)
