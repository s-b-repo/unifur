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
| `--dataset` | `synthetic` | `synthetic` \| `cifar100` \| `tiny-imagenet`. Repeatable (Phase 29): several sources in one run |
| `--data-dir` | — | Directory of a dataset's `.bin` splits; repeat once per `--dataset` that needs one, in order |
| `--dataset-weights` | uniform | Weights over the datasets, e.g. `0.7,0.3` |
| `--mix` | `mixture` | `mixture` (each batch from one source, drawn by weight) \| `composite` (every batch sliced from every source) |
| `--streaming` | `false` | Stream records from disk instead of loading a split into memory |
| `--image-size` / `--num-labels` | `32` / `100` | For synthetic data; real datasets dictate their own |
| `--batch-size` | `128` | |

### Objective

| Flag | Default | Meaning |
|---|---|---|
| `--objective` | `dblock` | `dblock` \| `consistency` \| `flow` \| `distill` |
| `--teacher` | — | Frozen teacher checkpoint for `distill` |
| `--num-blocks` | `3` | Must divide `num_hidden_layers` |
| `--gamma` | `0.05` | Sigma-window extension, in log space |
| `--teacher` | — | Repeatable (Phase 29): several teachers distil at once, weighted by `--teacher-weights` |

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
| `--mosme-spec` | — | Boxes-of-experts JSON; enables MoSME (conflicts with `--moe-every`) |
| `--mosme-every` | `2` | Which trunk layers get expert boxes |
| `--z-level` | `1e-3` | Router z-loss weight (ST-MoE); `0` disables it exactly |
| `--balance-schedule` | `constant` | `constant` \| `anneal` — how the balance-loss weight evolves |
| `--balance-weight` | `0.01` | Starting balance-loss weight |
| `--balance-scope` | `micro` | `micro` \| `global` (Phase 23.4). `global` measures the expert load over the `--accumulate` window instead of each micro-batch alone, so the router is not pushed to balance every micro-batch by itself. Flat MoE only for now |
| `--bias-balance-rate` | `0` | Loss-free bias balancing (Phase 23.5): move each router's selection bias by this much per step against its observed load. `0` is off and attaches no bias. DeepSeek-V3 uses `1e-3` |

Routing diagnostics (Phase 23.6) need no flag: every MoE or MoSME run logs
`load_entropy`, `token_entropy`, `min_load`, `max_load` and the per-layer
`routing_load` array to the JSONL file, and the end-of-run per-block table
gains `load H` and `token H` columns. See [MoE Routing](MoE-Routing.md).

### Output

| Flag | Default | Meaning |
|---|---|---|
| `--out-dir` | `checkpoints` | Content-addressed checkpoint directory |
| `--async-save` | `false` | Serialize on a background thread |
| `--resume [PATH]` | — | Bare: newest in `--out-dir`; with a path: that file |
| `--log-file` | — | Append-mode JSONL metrics |
| `--log-every` | `20` | Steps between log lines |
| `--checkpoint-every` | `0` | Also write a checkpoint (model + training state) every n steps (Phase 28). `--resume` restores and verifies that state |

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

Adds to the model flags above:

| Flag | Default | Meaning |
|---|---|---|
| `--repeats` | `3` | Measured repetitions per configuration; the table shows the mean and the 95% t-interval half-width |
| `--warmup` | `1` | Untimed repetitions first; recorded and flagged, left out of the summary (Phase 28) |
| `--json` | — | Append one experiment record per configuration (environment, config, seed, every raw trial, summary) to this JSONL file; never truncates |

Also reports a **test-time compute scaling curve** over step counts and
planner depths, with the Pareto frontier marked and the marginal accuracy per
extra layer — see [Accuracy Improvements](Accuracy-Improvements.md).

## `dblocks sweep` (Phase 28)

Train a grid of configurations, one run per seed, into experiment records.

| Flag | Default | Meaning |
|---|---|---|
| `--grid` | — | `lr=1e-4,3e-4 num_blocks=2,3 consistency=0,0.1`: axes separated by whitespace, values by commas. Keys: `lr steps batch_size num_blocks gamma weight_decay accumulate clip_norm ema_decay uncertainty importance_bins normalize_block_loss objective consistency lr_schedule moe_experts moe_top_k balance_scope bias_balance_rate` |
| `--seeds` | `1,2,3` | Every cell runs once per seed; the trials are the per-seed final losses |
| `--steps` | `50` | |
| `--dataset` / `--data-dir` | `synthetic` | As for `train` |
| `--batch-size` | `16` | |
| `--json` | — | Append each cell's record as soon as it finishes |

## `dblocks merge` (Phase 29)

Average same-architecture checkpoints into a new one: `--input` (repeatable),
`--weights` (uniform when empty), `--out` (default `checkpoints`), plus the
model flags that describe the architecture. The state directory beside the
result records the parents' hashes and weights. `dblocks lm merge` does the
same for `lm train` checkpoints (`--tiny` must match the inputs).

## `dblocks audit propagation` (Phase 28)

Per block, on one batch: local loss at the window midpoint, boundary mismatch
with the next block's `x0` estimate, the finite-difference sensitivity
`‖H(z+ε) − H(z)‖/‖ε‖` of the block's one-step map, and the downstream
amplification (product of the later blocks' sensitivities); plus the model's
own end-to-end cross-entropy and accuracy. Takes the model flags, `--batch-size`
(16), `--epsilon` (1e-2) and `--json`. On random weights it describes the
initialization; with `--checkpoint` it describes the model.

## `dblocks experiment` (Phase 28)

| Command | Meaning |
|---|---|
| `show --path log.jsonl` | Every record: name, unit, trials, warm-ups, mean ± 95% interval, median; the environment of the first |
| `compare --a x.jsonl --b y.jsonl` | Match records by name and print both summaries, the ratio of means and whether the intervals overlap |

## `dblocks lm`

Language-model paths (Phases 19, 21b and 24).

### `dblocks lm tokenize`

| Flag | Default | Meaning |
|---|---|---|
| `--input` | — | UTF-8 text file |
| `--out` | — | Corpus file: little-endian `u16` tokens, no header |
| `--label` | `false` | Also label anti-patterns, writing `<out>.labels` and `<out>.labels.json` (Phase 24) |
| `--rules` | built-in | Rule set JSON for `--label` |

### `dblocks lm corpus`

| Flag | Default | Meaning |
|---|---|---|
| `--path` | — | Corpus to describe. Opened streaming, so reporting never needs to hold it in memory. Reports the label manifest when one exists |
| `--context` | `256` | Context length used to count training windows |

### `dblocks lm label` (Phase 24)

| Flag | Default | Meaning |
|---|---|---|
| `--corpus` | — | Existing corpus; writes `<corpus>.labels` and `<corpus>.labels.json` next to it |
| `--rules` | built-in | Rule set JSON |

### `dblocks lm scan` (Phase 24)

| Flag | Default | Meaning |
|---|---|---|
| `--input` | — | Source file; findings are printed with line numbers and categories |
| `--rules` | built-in | Rule set JSON |

### `dblocks lm rules` (Phase 24)

| Flag | Default | Meaning |
|---|---|---|
| `--out` | stdout | Write the built-in rule set as JSON, to extend it |
| `--check` | — | Validate a rule file: every rule must match its examples and none of its counterexamples |

### `dblocks lm score`

| Flag | Default | Meaning |
|---|---|---|
| `--input` | — | Source file to score |
| `--language` | `rust` | Analyzer language |
| `--lexical` | `true` | Include the anti-pattern dimension |
| `--structural` | `true` | Include the structural heuristics dimension |
| `--external` | — | `clippy` \| `ruff` \| `eslint` \| a command; needs the `codequality-external` Cargo feature, else silently skipped |
| `--out` | stdout | Write the JSON report here |

### `dblocks lm train` (Phase 24)

| Flag | Default | Meaning |
|---|---|---|
| `--corpus` | — | Corpus from `lm tokenize`; a `.labels` sidecar next to it is opened automatically. Repeatable (Phase 29) |
| `--corpus-weights` | uniform | Weights over the corpora |
| `--mix` | `mixture` | `mixture` \| `composite` |
| `--teacher` | — | `lm train` checkpoints to distil from, repeatable (Phase 29) |
| `--teacher-weights` | uniform | |
| `--distill-weight` | `0` | Weight on `KL(teacher mixture \|\| student)`; `0` is off |
| `--distill-temperature` | `2.0` | |
| `--negative-teacher` | — | A checkpoint whose confident next-token choices are charged (Phase 29): an open-weight source of bad patterns |
| `--negative-confidence` | `0.5` | Charge a proposal only where the negative teacher is at least this sure |
| `--negative-penalty` | `1.0` | Coefficient on that charge |
| `--steps` | `200` | |
| `--batch-size` | `8` | |
| `--lr` | `3e-4` | AdamW |
| `--weight-decay` | `0.01` | |
| `--penalty` | `0` | Unlikelihood coefficient on labeled targets. `0` trains plainly and only *measures* `p(bad)`; a positive value charges for them and requires labels |
| `--streaming` | `false` | Sample windows from disk instead of loading the corpus |
| `--tiny` | `false` | The small configuration, for CPU smoke runs |
| `--seed` | `42` | |
| `--log-every` | `10` | |
| `--log` | — | Append-mode JSONL metrics (`loss`, `perplexity`, `penalized_tokens`, `penalized_prob`, `penalty`) |
| `--out-dir` | `checkpoints` | Content-addressed checkpoint, stem `lm`, with the training state beside it |
| `--checkpoint-every` | `0` | Also checkpoint every n steps (Phase 28) |
| `--resume` | — | A model from `lm train`; its training state is restored and verified |

### `dblocks lm bench` (Phase 28)

| Flag | Default | Meaning |
|---|---|---|
| `--corpus` | — | Corpus from `lm tokenize` |
| `--steps` | `30` | Steps per variant per seed |
| `--seeds` | `1,2` | |
| `--batch-size` | `4` | |
| `--json` | — | Append one record per variant |

Trains the tiny model under each trunk variant (`dense`, `moe`, `moe+bias`) and
reports the final loss per seed with its interval and the milliseconds per
step. On CPU this measures what the mechanisms cost, not what they buy.

### `dblocks lm policy` / `dblocks lm approvals` / `dblocks lm refusal-corpus` (Phase 30)

| Command | Flags | Meaning |
|---|---|---|
| `lm policy init` | `--out policy.json --key key.hex` | Write the starter policy (three cyber scopes) and a fresh signing key |
| `lm policy list` | `--policy` | Blockers, scopes, patterns and revocations |
| `lm policy add-blocker` | `--policy --id --scope --pattern (repeatable) --applies-to prompt\|output\|both --refusal [--description]` | Add a blocker; every pattern must parse and consume something |
| `lm policy remove-blocker` | `--policy --id` | Remove a blocker |
| `lm policy check` | `--policy --prompt [--key --grant ...]` | Show which blockers fire and the decision |
| `lm policy revoke` | `--policy --grant-id` | Refuse a grant from now on |
| `lm approvals issue` | `--key --policy --id --scopes a,b --expires <unix> [--note] --out grant.json` | Sign a grant |
| `lm approvals verify` | `--key --policy --grant` | Check signature, key, expiry and revocation, in that order |
| `lm refusal-corpus` | `--policy --prompts file [--answers file] --out corpus.bin` | Build the refusal/approval training documents and tokenize them |

`dblocks lm generate` takes `--policy policy.json` (with `--key key.hex` and any
number of `--grant grant.json`): a blocked prompt is refused before any
forward pass, a lifted one is sent with the approval marker, and the output
is checked against the output blockers. See [Cyber Policy](Cyber-Policy.md).

### `dblocks lm direction` / `ablate` / `direction-score` (Phase 31)

| Command | Flags | Meaning |
|---|---|---|
| `lm direction` | `--checkpoint --target a.txt --baseline b.txt [--layer auto\|n] --out dir.json [--tiny]` | Extract the behaviour direction (one prompt per line per file); `auto` keeps the layer with the largest separation |
| `lm ablate` | `--checkpoint --direction dir.json --out dir [--tiny]` | Orthogonalize every residual-writing weight against the direction; writes a new checkpoint whose state records the parent and the direction |
| `lm direction-score` | `--checkpoint --direction dir.json --prompts p.txt [--prompts q.txt] [--ablated] [--tiny]` | Mean projection of each prompt file onto the direction at its layer; `--ablated` also reports it with the direction projected out at inference |
| `lm heretic` | `--checkpoint --target a.txt --baseline b.txt [--trials 12] [--kl-weight 1] [--max-new 24] [--seed 0] --out dir [--json report.json] [--tiny]` | Heretic search over per-component weighted ablations against refusals and first-token KL; saves the best trial as a new checkpoint (Phase 31.6) |

`dblocks lm train` takes `--direction dir.json --direction-weight λ` to
penalize the squared projection during training; `dblocks train` takes
`--synthetic-negatives p` to mark a fraction of synthetic samples as negative
labels charged with `−log(1 − p_k)`. See [Direction Ablation](Direction-Ablation.md).

`dblocks lm train --heretic-target a.txt --heretic-baseline b.txt
[--heretic-trials 12] [--heretic-kl-weight 1] [--heretic-max-new 24]` runs the
Heretic search after training and saves the decensored model as the run's
final checkpoint (`heretic` in the report). `dblocks train --synthetic-negatives p
[--negative-penalty α]` relabels a fraction `p` of every batch with a wrong
class and charges the model `α · −log(1 − p_label)` for it (`negative_samples`,
`negative_prob` in the JSONL); the sweep key is `negatives`.

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
| `--checkpoint` | — | Weights from `dblocks lm train`; random when omitted |
| `--tiny` | `false` | Must match the checkpoint's configuration |
| `--policy` / `--key` / `--grant` | — | Gate the request through a policy (Phase 30); `--grant` is repeatable |

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
