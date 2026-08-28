# Configuration Reference

Every CLI flag and the config types behind them.

---

## In this repository

Configuration is passed as command-line flags, not YAML. The types are plain
Rust structs (`TrainConfig`, `MultiBlockConfig`, `InferenceConfig`,
`TrainingChecks`, …) with `Default` implementations, so the library API and the
CLI cannot drift apart.

```bash
dblocks --help
dblocks train --help
```

---

## Commands

| Command | Purpose |
|---|---|
| `train` | Block-wise training |
| `sample` | One batch through the sampler, with statistics |
| `infer` | Batched classification with top-k output |
| `bench` | Sweep solvers × strategies, plus a compute-scaling curve |
| `lm` | Tokenize a corpus, describe it, and generate text |
| `experts` | Inspect or scaffold an expert index |
| `verify` | Run the numerical certificate suite |
| `sigmas` | Print the block schedule and windows |

---

## `dblocks train`

### Data

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `synthetic` | `synthetic` \| `cifar100` \| `tiny-imagenet` |
| `--data-dir` | — | Directory holding the `.bin` splits; required by the real datasets |
| `--streaming` | `false` | Fetch records per batch instead of loading the split |
| `--image-size` | `32` | Ignored when the dataset dictates it |
| `--num-labels` | `100` | Ignored when the dataset dictates it |
| `--batch-size` | `128` | |

### Objective

| Flag | Default | Meaning |
|---|---|---|
| `--objective` | `dblock` | `dblock` \| `consistency` \| `flow` \| `distill` |
| `--teacher` | — | Frozen teacher checkpoint for `distill` |
| `--num-blocks` | `3` | Must divide `num_hidden_layers` |
| `--gamma` | `0.05` | Sigma-window extension, in log space |

### Optimization

| Flag | Default | Meaning |
|---|---|---|
| `--lr` | `0.001` | AdamW learning rate |
| `--weight-decay` | `0.01` | |
| `--steps` | `200` | Optimizer steps, not epochs |
| `--seed` | `42` | Seeds both host and device RNGs |
| `--grad-checkpointing` | `false` | Recompute activations during backward |

### Loss reduction and convergence (Phase 20)

| Flag | Default | Meaning |
|---|---|---|
| `--lr-schedule` | `constant` | `constant` \| `warmup-cosine` \| `warmup` |
| `--accumulate` | `1` | Micro-batches per optimizer step, **averaged** so the learning rate does not scale with the count |
| `--clip-norm` | — | Rescale gradients above this global norm. Complements the gradient *gate*, which discards rather than rescales |
| `--ema-decay` | — | Bias-corrected weight averaging; the shadow is returned in place of the live weights |
| `--normalize-block-loss` | `false` | Equalize blocks on the **geometric** mean |
| `--uncertainty` | `0.0` | Learned per-sigma uncertainty weighting, in `[0, 1]`. `0.0` is the exact identity; at its optimum the gradient becomes that of log-loss |
| `--importance-bins` | `0` | Equal-probability CDF bins for sigma importance sampling. `0` disables it; a cold sampler is exactly plain sampling |
| `--z-level` | `0.001` | Router z-loss weight (ST-MoE). Penalizes large routing logits, which the balance loss cannot see. `0.0` disables it exactly |
| `--balance-schedule` | `constant` | `constant` \| `anneal`. Annealing decays the balance weight geometrically once routing collapse is no longer the risk, so it stops fighting specialization |
| `--balance-weight` | `0.01` | Starting weight for `--balance-schedule` |

See [Loss Reduction](Loss-Reduction.md) for why the `+ l` term and the smoothing
floor are the load-bearing parts.

### Quality verification

| Flag | Default | Meaning |
|---|---|---|
| `--no-checks` | `false` | Disable all training-time verification |
| `--no-preflight` | `false` | Skip the pre-run certificate check |
| `--verify-every` | `0` | Re-verify the live model every *n* steps (0 = never) |

See [Quality Gate](Quality-Gate.md).

### Mixture-of-experts

| Flag | Default | Meaning |
|---|---|---|
| `--moe-every` | — | Replace every *n*-th layer's MLP with an MoE layer |
| `--moe-experts` | `4` | Experts per MoE layer |
| `--moe-top-k` | `1` | Experts weighted per token |

### Output

| Flag | Default | Meaning |
|---|---|---|
| `--out-dir` | `checkpoints` | Content-addressed checkpoint directory |
| `--async-save` | `false` | Serialize on a background thread |
| `--resume [PATH]` | — | Bare: newest in `--out-dir`; with a path: that file |
| `--log-file` | — | Append-mode JSONL metrics |
| `--log-every` | `20` | Steps between log lines |

---

## `dblocks sample`

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | — | Model to load; random weights when omitted |
| `--num-blocks` | `4` | |
| `--num-hidden-layers` | `12` | |
| `--num-inference-steps` | `3` | Sampling windows |
| `--batch-size` | `8` | |
| `--solver` | `euler` | `euler` \| `heun` \| `ddim` \| `dpmpp2m` \| `dpmpp3m` |
| `--strategy` | `sequential` | `sequential` \| `parallel` \| `hybrid` \| `adaptive` |
| `--k` | `2` | Span width for parallel/hybrid/adaptive |
| `--gate` | `lenient` | `lenient` \| `strict` \| `tightening` |
| `--precision` | `f32` | `f32` \| `bf16` \| `f16` (emulated) |
| `--precision-switch` | `0.0` | Sigma below which sampling reverts to f32 |

### Accuracy (Phase 22)

| Flag | Default | Meaning |
|---|---|---|
| `--guidance` | `1.0` | Guidance scale. `1.0` returns the conditional estimate **bitwise**; anything else doubles the model calls |
| `--guidance-rescale` | `0.0` | Fraction of the conditional estimate's spread to restore after guidance, in `[0, 1]` |
| `--logit-norm` | `none` | `none` \| `temperature` \| `l2` \| `standardize`. Never changes the prediction — only the reported confidence |
| `--logit-tau` | `1.0` | Temperature for `--logit-norm` |
| `--ensemble` | — | `probability` \| `logit` \| `vote`, over the deterministic solvers. Empty runs a single solver |

### Planning (Phase 21a)

| Flag | Default | Meaning |
|---|---|---|
| `--planned` | `false` | Plan each step instead of following the schedule |
| `--plan-depth` | `1` | Rollout depth. `0` is greedy planning and is exactly the old policy |
| `--plan-beam` | `3` | Paths kept between expansions |
| `--plan-budget` | `32` | Candidate evaluations per committed step |

`--planned` and `--ensemble` are mutually exclusive; planning replaces the
schedule outright, so combining them would silently pick one.

## `dblocks bench`

Adds `--repeats` (default 3) to the model flags above. Also reports a
**test-time compute scaling curve** over step counts and planner depths, with
the Pareto frontier marked and the marginal accuracy per extra layer — see
[Accuracy Improvements](Accuracy-Improvements.md).

## `dblocks lm`

Language-model paths (Phases 19 and 21b).

### `dblocks lm tokenize`

| Flag | Default | Meaning |
|---|---|---|
| `--input` | — | UTF-8 text file |
| `--out` | — | Corpus file: little-endian `u16` tokens, no header |

### `dblocks lm corpus`

| Flag | Default | Meaning |
|---|---|---|
| `--path` | — | Corpus to describe. Opened streaming, so reporting never needs to hold it in memory |
| `--context` | `256` | Context length used to count training windows |

### `dblocks lm generate`

| Flag | Default | Meaning |
|---|---|---|
| `--prompt` | `Hello` | |
| `--max-new` | `32` | |
| `--sampling` | `greedy` | `greedy` \| `topk` |
| `--top-k` | `8` | Also the branching factor under `--lookahead` |
| `--temperature` | `1.0` | For `--sampling topk` |
| `--cached` | `false` | Decode with a KV cache: `O(n)` per token instead of `O(n²)` |
| `--lookahead` | `0` | Score continuations this many tokens deep. `0` is ordinary greedy decoding |
| `--beam` | `3` | Beam width for `--lookahead` |
| `--budget` | `32` | Candidate evaluations per committed token |
| `--seed` | `1337` | |

Weights are random unless a checkpoint is loaded, so the text is noise. What the
command demonstrates is that the decoding paths agree and what each one costs.

## `dblocks infer`

Adds `--top-k` (default 3) to the model flags above.

## `dblocks verify`

| Flag | Meaning |
|---|---|
| `--group NAME` | Run only one group: `schedule`, `preconditioning`, `stats`, `solver`, `precision`, `quantize`, `loopgraph`, `moe`, `model`, `autodiff` |

Exits non-zero if any certificate fails.

## `dblocks sigmas`

`--num-blocks` (default 3) and `--gamma` (default 0.05). Prints the ascending
boundary grid and each block's window, bare and gamma-extended.

---

## Library configuration

### `ViTDiTConfig`

```rust
ViTDiTConfig::with_image_size(32, 100)   // CIFAR preset: patch 4, 12 layers, hidden 128
ViTDiTConfig::tiny_imagenet(200)         // 64x64: hidden 768, 12 heads
ViTDiTConfig::tiny(10)                   // small preset for tests and smoke runs
ViTDiTConfig::tiny(10).with_moe(MoeTrunkConfig::default())
```

### `DblockConfig`

| Field | Default | Meaning |
|---|---|---|
| `num_blocks` | `3` | |
| `gamma` | `0.05` | Sigma-window extension |
| `sigma_data` | `0.5` | EDM data scale |
| `num_inference_steps` | `None` | Defaults to `num_blocks` |
| `moe_aux_weight` | `0.01` | Switch balance-loss weight |

### Noise schedule constants

Fixed to the reference values in `sigma.rs`: `SIGMA_MIN = 0.002`,
`SIGMA_MAX = 80.0`, `P_MEAN = -1.2`, `P_STD = 1.2`, `RHO = 7.0`.

### Block indexing

Block boundaries ascend, but **block indices descend in noise level**: the
composition `y = H_{B-1} ∘ … ∘ H_0(x)` integrates the reverse ODE, so block 0
runs first and owns the noisiest window:

```text
block b covers (block_sigmas[B - b - 1], block_sigmas[B - b]]
```

Everything downstream must agree on this — the training-sigma sampler, boundary
consistency, span selection. The `block_routing_involution` certificate asserts
the round trip, because getting it backwards means every block is trained on
one noise range and evaluated on another.

Inspect it with:

```bash
dblocks sigmas --num-blocks 3
```

```text
block windows (block 0 is the noisiest):
  block 0: (0.505041, 80.000000]  extended [0.392048, 80.000000]
  block 1: (0.179632, 0.505041]   extended [0.170584, 0.531832]
  block 2: (0.002000, 0.179632]   extended [0.002000, 0.224933]
```

---

## See also

- [Training Guide](Training-Guide.md)
- [Inference Guide](Inference-Guide.md)
- [Loss Reduction](Loss-Reduction.md)
- [Language Modeling](Language-Modeling.md)
- [Next-Step Planning](Next-Step-Planning.md)
- [Accuracy Improvements](Accuracy-Improvements.md)
- [Quality Gate](Quality-Gate.md)
