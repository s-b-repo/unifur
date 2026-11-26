# Training Guide

Everything `dblocks train` can do, and how to read what it reports.

## Native Nina workers

Nina trains a multiscale residual-VQ codec with `dblocks ncm tokenizer train`.
The concept-field generator uses the frozen codec and combines geometric
scoring with latent vector updates at each NCM level. The 4B worker recipe
uses two latent updates per level. At inference, Omni passes the current
question and a bounded Subnext working set to `dblocks ncm generate
--prompt-stdin --json`. Subnext stores durable context, and the generator
reports its level calls and trunk evaluations. [Omni contract](https://github.com/s-b-repo/omni/blob/main/docs/CONTRACT.md#the-native-nina-route)

### Training resources

| Resource | Verified work and limits |
|---|---|
| Laptop, i5-1235U, 12 threads, 23 GiB RAM | CPU pilots, corpus preparation and checkpoint replay. Release-build samples measured 3.9 GFLOP/s for the tiny codec and 2.7–15.3 GFLOP/s for generator widths 64–512. |
| Standard public GitHub Ubuntu runner | 4 vCPUs, 16 GB RAM and 14 GB SSD per job. The capacity run completed 40 independent Geometry3K fits of 384 updates each, peaked at 20 active fits, and verified 80 checkpoints. |
| Single discrete GPU worker | The self-hosted workflow uses one WGPU device. Profile peak memory, scratch disk, sustained non-padding tokens per second and resume on the actual device. A 24 GiB card can run smaller pilots. |
| Multi-GPU worker group | The native gradient probe compares synchronized CPU ranks with a full-batch update on one machine. Production multi-host collectives and sharded resume still require implementation and profiling. |

The 40-job run validated independent fits. One hosted checkpoint resumed on
the laptop at step 192 and agreed on all 120 development predictions through
step 384. The two final weight hashes differ. The local gradient probe with a
trained codec passed at two and 40 ranks. The four-step codec exceeded the
gradient tolerance at 40 ranks, so the hosted preflight checks two ranks.
[Hosted capacity run](https://github.com/s-b-repo/unifur/actions/runs/44946149797)
· [Gradient probe](../examples/ncm_grid_probe.rs)

Standard hosted jobs have a six-hour cap. GitHub lists 20, 40 and 60 concurrent
standard jobs for Free, Pro and Team respectively. Concurrent jobs each get
their own VM and memory. GitHub lists larger CPU runners up to 96 vCPUs and
384 GB RAM and a larger GPU option with a 16 GB T4. Access and billing depend
on the repository's account and runner group.
[Hosted runner specs](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
· [Actions limits](https://docs.github.com/en/actions/reference/limits)
· [Larger runner specs](https://docs.github.com/en/actions/reference/runners/larger-runners)

The proposed **3.98 billion generator parameters per text model** use 35
blocks, width 3,072, 48 heads and aligned 1,024-byte prompt and target
records. FP32 weights, gradients and two AdamW moments alone take about
64 GB (59 GiB). A checkpoint retaining weights and both moments needs
roughly 48 GB (45 GiB) of uncompressed tensor data before codec files,
staging copies and retained checkpoints. The full fit and its wall time
are unverified. Qualify one full replica per device or verify the sharded
backend, then estimate duration from measured throughput at the chosen
batch and sequence size.

### Dispatch a worker

Run the public four-step codec and generator preflight with the [Nina worker workflow](../.github/workflows/nina-worker.yml).

```bash
gh workflow run nina-worker.yml --repo s-b-repo/unifur --ref main -f mode=preflight
```

For a full fit, set the repository variable `NINA_WORKER_ROOT` to durable
storage on the authorized runner labeled `nina-gpu`. Place the corpus under
`$NINA_WORKER_ROOT/corpora/$corpus_id` with `manifest.json` and each of
`{train,validation}.{aligned,codec}.bin`. The [worker script](../experiments/model_matrix/nina_worker.py)
validates the hashes, record widths, split counts and disjoint packed IDs.
The current combined instruction pack has 145,310 training and 7,662
validation records, with about 33.0 million non-padding target tokens per
training pass. Its inputs include an expanded pool of 120,807 training and
6,711 validation pairs and 110,178 packed longer-response records. The
367 million byte-token field archive is a separate corpus. Track unique
target tokens and repeated exposures explicitly if the campaign also targets
four billion **training tokens**.

```bash
records=$(python3 -B experiments/model_matrix/nina_worker.py \
  validate "$NINA_WORKER_ROOT/corpora/$corpus_id" | jq -er .records.train)
codec_steps=$(( (records + codec_batch - 1) / codec_batch ))
recipe=(-f stage=codec -f corpus="$corpus_id" -f run_key="$run_key"
  -F steps="$codec_steps" -F batch_size="$codec_batch" -F checkpoint_every="$save_every")
gh workflow run nina-worker.yml --repo s-b-repo/unifur --ref "$training_ref" \
  -f mode=gpu-qualify "${recipe[@]}"
gh workflow run nina-worker.yml --repo s-b-repo/unifur --ref "$training_ref" \
  -f mode=gpu-fit -F stop_at="$next_step" "${recipe[@]}"
```

Pin `training_ref` for both stages and any retries. Wait for qualification
before the first fit. Qualification performs four updates of the full
schedule and reloads a verified model and optimizer after the first step.
It records end-to-end non-padding target throughput, peak host RSS, sampled
GPU framebuffer use and the runtime, corpus and recipe fingerprints. Use
a discrete WGPU accelerator with NVIDIA or AMD Linux framebuffer telemetry.
Generator qualification checks the FP32 parameter-state lower bound and
room for two checkpoint writes before allocating the model.

Set `next_step` to an absolute optimizer step above the saved step. Choose
slices that fit the workflow's 300-minute runtime and retain the planned
total `steps` across retries. Training uses that total for the learning-rate
schedule and visits prepared records in order. The codec must finish
before generator qualification. Use the same run key and corpus with
`stage=generator` and the generator batch and step counts. Set
`validation_windows` in both dispatches if the chosen GPU needs a smaller
validation batch. Keep the runtime, matching codec and generator, and both
`.state` sidecars together. Cosine is the default. WSD remains available for
measured comparisons.

The standard generation path reuses prompt embeddings, masks, the frozen
concept metric and fixed codebook statistics. On the local CPU parity set, 64 greedy and eight
seeded sampling requests reproduced the previous reports. For public
release, verify clean-restart reproducibility and evaluate long-answer
reconstruction, held-out answer quality, geometry and external-context
behavior.

### Reproduce the compact grounding pilot

The [grounding scripts](../experiments/nina_grounding/) build paired questions
with two different supported answers, related distractors and evidence-free
controls. The codec defaults to 16-byte targets with balanced frequencies,
disjoint synthetic values and 160 ordered updates. Evaluation uses different
source identities and paraphrased questions.

```bash
cargo build --locked --release --bin dblocks
pilot=experiments/nina_grounding
run=$(mktemp -d)
python3 -B "$pilot/grounding.py" prepare --out "$run/corpus"
python3 -B "$pilot/pilot.py" --data "$run/corpus" --out "$run/codec" \
  --codec-only --threads 3 --val-windows 24
tokenizer=$(jq -er '.tokenizer.path' "$run/codec/codec.json")
python3 -B "$pilot/grounding.py" oracle --cases "$run/corpus/test.jsonl" \
  --tokenizer "$tokenizer" --out "$run/codec-test.json"
```

The codec reproduced 60 of 72 targets through EOS, including 24 of 36
supported answers. The generator trained with this codec scored 0 of 36
on the supported holdout. `--no-balance-codec`, `--no-augment-codec` and
`--no-ordered-codec` provide corpus and training controls. Use
`grounding.py evaluate` to score a generator with the same cases.

`grounding_stress.py` uses the pinned Subnext reader and answerability
models to measure the trade-off between abstention and exact supported
answers. Set `NINA_SUBNEXT_SCRIPTS` if Subnext is outside the sibling
checkout layout. Pass the public SQuAD 2.0 development set with
`--squad-dev`, the prepared corpus with `--corpus`, the Subnext model
directory with `--model`, and a report path with `--out`.
[Subnext semantic model setup](https://github.com/s-b-repo/subnext/blob/main/README.md#learned-retrieval-for-compact-evidence) · [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) · [Selective QA under domain shift](https://aclanthology.org/2020.acl-main.503/)

GLM-5.2 uses index reuse to reduce repeated attention selection. The
Nina runtime applies the related idea to tensors fixed within one request.
Subnext's hybrid shortlist and reranking follow the same broad two-stage
pattern as the context selection studied by DeepSeek and MiniMax.
[GLM-5.2](https://huggingface.co/blog/zai-org/glm-52-blog) · [DeepSeek-V4](https://arxiv.org/abs/2606.19348) · [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)

### Gate a trained candidate

Stage `cases.jsonl` and `policy.json` under
`$NINA_WORKER_ROOT/evaluations/$run_key` and dispatch `mode=gpu-release-eval`
with the corpus, run key and training revision. Set the worker variables
`NINA_OMNI_BIN` and `NINA_DCR_BIN` for the external-context cases.

Each case carries `id`, `field`, `source`, `prompt` and `answers`
(accepted answer strings). Context cases also carry `context.relevant`
and `context.distractor` source text. The policy gives
`min_codec_token_accuracy`, a `fields` map with `min_cases` and
`min_accuracy` for every trained field, and a `context` section with
`min_cases`, `min_relevant_accuracy` and `min_distractor_accuracy`.
Use reviewed held-out questions with answers that normalized exact match
can score. The evaluator checks case IDs and normalized prompts against both
packed splits, verifies the checkpoint digests, and compares one and two
latent updates. It uses isolated Subnext stores for the context controls
and writes the pass or fail report and fingerprints to `report.json`.
[Evaluator source](../experiments/model_matrix/nina_release.py)

### Native research controls

Use `dblocks nina research check-grader` to check the seven pinned numeric
fixtures. `compare` scores a single greedy output per case under a shared
limit. The report keeps every planned case in its denominator and records
strict correctness, semantic correctness, output status, paired gains and
losses, and joint correctness on reworded and answer-changing pairs. The
`runtime` command crosses two checkpoints with two runtimes using the same
case plan. Use its per-case output comparison to detect changes that leave
aggregate accuracy unchanged.

`first-draws` joins the first prediction for every scheduled case, with
coverage broken out by skill or length slice. Its report records pending
draws, timeouts and incomplete outputs. Wait for its readiness flag before
spending a retry budget. The archived capped pass decoded 58 of 96 prompts,
including ten multiplication prompts and zero division prompts. `early-rank`
requires both rungs to cover every candidate and seed on the same cohort.
The historical science ranking reversed all three pairs from 60 to 120
updates, so use the early screen for integrity and minimum-quality checks.

```bash
RAYON_NUM_THREADS=2 MATMUL_NUM_THREADS=2 dblocks nina research arithmetic \
  --cases experiments/nina_research/arithmetic-cases.json \
  --out-dir runs/arithmetic-96
```

The CPU pilot checks arithmetic independently and rejects any additive or
multiplicative family shared by training and development. It fits two seeds
with verified labels and two matched scrambled controls. The controls keep
the same answer histogram, byte lengths and supervised positions. The run
directory records the protocol before fitting, every attempt's elapsed time,
the checkpoint digests, and the two CPU readouts for each fit. Use
`--prepare-only` to inspect the protocol and byte masks before scheduling fits.

Advancement requires both solver seeds to beat the train-only majority and
their matched controls on canonical cases, improve on alternate phrasing,
and answer both members of at least one answer-changing pair. In the original
research the 384-step solver arms got 4/96 and 3/96, with zero jointly correct
answer-changing pairs. Both fit all ten repeated training mappings. The
result closed that recipe at development.

`science-screen --development PATH --out-dir DIR` preserves unabridged
science prompts, supplies the first- and longest-answer reference scores,
and prepares four-choice rotations for `positions`. The historical 256-byte
screen included 131 of 297 cases. The full plan retains the over-width slice
for a longer-context evaluation. `long-record-validation --source PACK
--out-dir DIR` verifies the existing byte pack, excludes normalized stems
seen in raw training, and emits aligned and codec validation views together.
The historical pack yielded 7,531 clean rows and a 384-row stratified screen.
Treat both fast screens as development evidence. The retained research plan
defines the separate family-disjoint confirmation gate.

During a dense masked `lm train --accumulate N` run, `--accumulation-weighting
auto` pools the loss and gradients over the accepted supervised targets.
`tokens` requires a poolable objective, and `microbatches` selects the
equal-window weighting. Resuming uses the reduction recorded in the sidecar.
Keep the optimizer and state JSON with the weight file. LM and student
snapshots retain distinct training boundaries when canonical weights repeat.
The archived control measured the difference using windows with six and ten
targets. Compare effective target exposure before interpreting a throughput or
accuracy change. [Accumulation study](https://huggingface.co/blog/gradient_accumulation)

### Converted XSum checkpoints

The native Rust CLI also loads the converted Hazy Research XSum release with
MoSME routing and geometric scoring. The bundle includes the trained codec,
shared generator and 17 initially byte-identical expert shards with
`specialized_training_steps: 0`. The optional adapter fits expert output
biases. Full expert specialization and public instruction quality need
their own training and behavioral evaluations. [Upstream XSum weights](https://huggingface.co/hazyresearch/ncm-checkpoints/tree/main/xsum)

| Native XSum probe | Result |
|---|---|
| Shared-prefix adapter training | Source passes fell from 4 to 1 and concept-level passes from 36 to 33. The original release probe returned identical adapter biases and held-out losses. |
| Pooling before the value projection | The isolated diagnostic moved from 360 to 48 value projections with maximum FP32 difference `8.94e-8`. |
| Certified integer attention | All 48 decisions matched the full integer reference, and 32 of 1,440 token scores were skipped. Agreement with the floating-point readout was 46/48. |
| Generation and certificates | The reference comparison matched 200 concept codes and 48 output tokens. The original XSum build passed all 34 NCM certificates with LTO disabled. The integrated build matched generation again and passed all 36 NCM and eight NCM-decode certificates. |

Shared-prefix reuse is the adapter training default. The pooled projection
comparison and integer readout run on demand with
`--attention-certificate`. Generation uses its floating-point path.
An integrated two-thread debug replay reproduced the training pass counts
with maximum differences of `3.63e-7` in adapter biases and `1.06e-5` nats
in held-out losses. Record the build and thread settings with parity results.

```bash
CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  cargo build --locked --release --bin dblocks
target/release/dblocks ncm released --model weights/nina-1b \
  --prompt 'The library approved its revised opening schedule.' \
  --attention-certificate --json
```

Prepare `weights/nina-1b` with
[the release converter](../experiments/nina_1b/convert_release.py). Run
`ncm released-tune` with disjoint labeled training and validation JSONL
files, or dispatch the [CPU adapter workflow](../.github/workflows/nina-adapter.yml)
with the run ID of a labeled corpus artifact. Use `--no-share-prefix` for a
reference run with the same records and optimizer settings.

---

## In this repository

| | |
|---|---|
| **Module** | `src/train.rs` |
| **Key types** | `TrainConfig`, `Objective`, `DatasetChoice`, `TrainSummary` |
| **Quality** | `quality::TrainingChecks`, `quality::TrainingHealth` |
| **Convergence** | `schedule.rs`, `reweight.rs` — see [Loss Reduction](Loss-Reduction.md) |
| **CLI** | `dblocks train` |

---

## The shortest useful run

```bash
cargo build --release
./target/release/dblocks train --steps 200 --num-blocks 3
```

Synthetic data needs no download, so this exercises the whole loop — model
construction, block-wise loss, gradient gating, checkpointing — in seconds. It
will not learn anything: the images are noise. Use it to check plumbing and to
measure step throughput, not accuracy.

---

## Datasets

| Flag | Image size | Classes | Notes |
|---|---|---|---|
| `--dataset synthetic` | 32 | as `--num-labels` | Default; random images |
| `--dataset cifar100 --data-dir DIR` | 32 | 100 | Needs `train.bin` / `test.bin` |
| `--dataset tiny-imagenet --data-dir DIR` | 64 | 200 | Needs a one-off conversion |

A real dataset **dictates** the image size and class count — they are taken
from the dataset rather than from `--image-size` / `--num-labels`. Silently
training a 32×32 / 100-class model on 64×64 / 200-class data would otherwise be
an easy mistake to make.

### CIFAR-100

```bash
wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar xzf cifar-100-binary.tar.gz          # -> cifar-100/train.bin, test.bin

./target/release/dblocks train \
    --dataset cifar100 --data-dir cifar-100 \
    --steps 20000 --batch-size 128 --num-blocks 3
```

The **fine** label (byte 1 of each record) is the 100-way target; byte 0 is the
20-way coarse label and is ignored.

### Tiny ImageNet

Tiny ImageNet ships as JPEGs, and this crate carries no image codec on purpose,
so convert once to the raw fixed-record layout
(`[label_u16_le, R(4096), G(4096), B(4096)]`, planar CHW):

```python
import numpy as np, pathlib
from PIL import Image
root = pathlib.Path('tiny-imagenet-200/train')
wnids = sorted(p.name for p in root.iterdir() if p.is_dir())
with open('tiny-imagenet/train.bin', 'wb') as out:
    for label, wnid in enumerate(wnids):
        for img in sorted((root / wnid / 'images').glob('*.JPEG')):
            a = np.asarray(Image.open(img).convert('RGB'), dtype=np.uint8)
            out.write(np.uint16(label).tobytes())
            out.write(a.transpose(2, 0, 1).tobytes())   # HWC -> CHW
```

The same script is available as `tinyimagenet::CONVERTER`. Then:

```bash
./target/release/dblocks train \
    --dataset tiny-imagenet --data-dir tiny-imagenet --streaming \
    --steps 50000 --batch-size 64
```

### `--streaming`

Without it the whole split is loaded into memory; with it records are fetched
per batch. Streaming is worth it for Tiny ImageNet (~1.2 GB in this layout) and
usually not for CIFAR-100 (~150 MB).

The two paths are **bit-identical** — same seed, same batches, same checkpoint
hash — which `test_streaming_matches_in_memory_exactly` asserts. Switching to
streaming for memory reasons must not silently change what the model sees.

---

## Objectives

```bash
--objective dblock        # EDM-weighted cross-entropy on one random block (default)
--objective consistency   # + boundary / self / trajectory / cross-fork residuals
--objective flow          # rectified-flow velocity regression
--objective distill       # block distillation against a frozen teacher
```

### `dblock`

The original paper's loss. Each step picks a random block, samples per-example
sigmas inside that block's (gamma-extended) window, noises the label
embeddings, and trains the block to recover them — cross-entropy weighted by
`w(σ) = (σ² + σ_d²)/(σσ_d)²`.

That weight grows very large near `sigma_min`, so **the block owning the lowest
sigma window legitimately shows a much higher loss** than the others. This is
expected, not a bug; the per-block table exists partly so it is visible rather
than confusing.

### `consistency`

Adds four residuals, each a plain MSE scaled by `weight × λ(step)`:

- **boundary** — blocks `b` and `b+1` must agree at their shared sigma
- **self** — one block at two noise levels of the same clean state
- **trajectory** — a full chain rollout must land where direct noising lands
- **cross-fork** — a joint span `i..=j` must reproduce the sequential
  composition of those blocks, which is exactly what `Strategy::Parallel`
  relies on

It costs roughly 12 forward passes per step instead of one, so expect it to be
an order of magnitude slower. Disable terms you are not studying with
`ConsistencyWeights::boundary_only()` or by zeroing individual weights.

With every weight zero it reduces **exactly** to the `dblock` loss —
`integration_disabling_every_consistency_term_reduces_to_the_plain_loss` pins
that, so `consistency` is a strict superset rather than a parallel
implementation.

### `flow`

Samples `t ~ U(0,1)`, builds the straight-line OT path
`x_t = (1-t)x_0 + t x_1`, and regresses the pooled hidden onto `v* = x_1 - x_0`.

### `distill`

```bash
# Against a separately trained teacher.
./target/release/dblocks train --objective distill \
    --teacher checkpoints/dblocks-<hash>.mpk --steps 5000

# Without --teacher: the initial model is frozen as the teacher. This is
# self-distillation, and still meaningful -- the student must cover several
# teacher substeps in one.
./target/release/dblocks train --objective distill --steps 5000
```

The teacher takes `teacher_substeps` solver steps across a window; the student
must reach the same latent in **one**. Teacher outputs are detached, so no
gradient reaches it even though both share a backend.

---

## Quality verification during training

Block-wise training fails quietly, so every phase of a step is verified
separately. See [Quality Gate](Quality-Gate.md) for the full design.

```bash
--verify-every 100     # re-verify the live model every 100 steps
--no-preflight         # skip the pre-run certificate check
--no-checks            # disable all training-time verification
```

Default behaviour:

1. **Before step 0** the schedule, preconditioning and statistics certificates
   run. A failure aborts — training against a broken schedule is wasted.
2. **Each step** the loss must be finite and the global gradient norm must lie
   in `[1e-8, 1e4]`. A failure rejects that step (skip, not clip) and continues.
3. **After each optimizer step** every parameter must still be finite. A
   failure aborts: a NaN in the weights poisons everything after it.
4. **50 consecutive rejections** abort the run.

### Reading the report

```text
preflight: 12 certificates passed
step      0 | loss 25.5619 | grad 2.321e1 | avg 25.5619 | skipped 0 | 4.0 steps/s
...
done: 200 steps in 51.2s (mean loss 18.3, 0 rejected by a quality check, 0.0% reject rate)

per-block quality:
block     steps   rejected    mean loss     mean |g|      max |g|   reject%
--------------------------------------------------------------------------
0            67          0      14.2251    1.443e+01    2.331e+01      0.0%
1            65          0      29.6249    1.611e+01    4.605e+01      0.0%
2            68          2    3418.1597    2.008e+03    9.912e+03      2.9%
```

What to look for:

| Symptom | Meaning |
|---|---|
| `max \|g\| == 0` for a block that ran | The block is **dead**; also reported explicitly with a warning |
| One block's `reject%` far above the others | That block's sigma window is badly conditioned |
| High mean loss on the last block | Expected — the EDM weight explodes near `sigma_min` |
| Run aborted at `parameters` | A NaN reached the weights; lower the learning rate |
| Run aborted after 50 rejections | Something systematic; check `num_blocks` divides `num_hidden_layers` |

---

## Mixture-of-experts

```bash
./target/release/dblocks train --moe-every 2 --moe-experts 4 --moe-top-k 2
```

Replaces every *n*-th layer's dense MLP with a mixture of experts. The router
is conditioned on the adaLN vector, which is a pure function of sigma — that is
what makes the routing noise-aware without extra plumbing.

The Switch load-balancing loss of the executed span is added to the objective
automatically (weight `DblockConfig::moe_aux_weight`, default 0.01) and logged
as `balance_loss`. Without it the router is free to collapse onto one expert.
Expect it near `1.0` per sparse layer — the uniform-routing minimum.

---

## Checkpoints and resuming

Checkpoints are **content-addressed**: the filename embeds a canonical hash of
every parameter tensor, so identical weights always map to the same filename
and saving the same model twice writes one file.

```bash
--out-dir checkpoints        # where to write
--checkpoint-every 500       # also checkpoint every n steps
--async-save                 # serialize the weights on a background thread (weights only)
--resume                     # newest checkpoint in --out-dir
--resume path/to/ckpt.mpk    # a specific one
```

Bare `--resume` on an empty directory starts fresh and says so, rather than
failing — a first run has nothing to resume from.

### The training state beside the model (Phase 28)

A model file alone cannot continue a run: the optimizer's moments, the
schedules' positions, the RNG, the EMA shadow and every host-side estimator
would restart, and the "resumed" run would quietly be a different one. The
trainer therefore writes a directory next to every model file it saves:

```text
checkpoints/dblocks-3f9c…​.mpk          the weights (content-addressed, as before)
checkpoints/dblocks-3f9c…​.state/
    state.json      step, seed, host RNG, config, build (git revision, rustc,
                    Burn version), dataset identity (sha256 of the data),
                    sha256 of every file below, host-side estimators
    optimizer.mpk   AdamW moments
    ema.mpk         the EMA shadow, when --ema-decay is on
    head.mpk, head-optimizer.mpk   the uncertainty head and its optimizer
```

`--resume` restores all of it and **verifies every file's hash first**: a
truncated or edited file is refused by name rather than loaded. It also
refuses a checkpoint whose dataset hash differs from the data it was asked to
continue on, and warns when the configuration differs in anything but the
step count and paths. A model file with no state directory (one written by an
earlier version, or with `--async-save`) still resumes — weights only, and it
says so.

The guarantee is bit-identity: six steps in one run and three steps plus a
resumed three produce the same weights, the same EMA shadow and the same
logged losses. `integration_resume_is_bit_identical_for_the_image_trainer`
holds that line with EMA, the uncertainty head, importance sampling, loss
normalization and dropout all on. What makes it possible is that the
**device** RNG is reseeded at the top of every step from `(seed, step)`: the
backend's random stream is a process-wide global, and snapshotting the host
RNG alone would have left it at a different position.

---

## Logging

```bash
--log-file runs/metrics.jsonl
```

One flat JSON object per logged step, in a W&B-compatible schema:

```json
{"step":0,"loss":25.56,"ce_loss":2.30,"balance_loss":0,"block":1,
 "grad_norm":23.21,"steps_per_s":4.04,"skipped":0,"accepted":true}
```

The file is opened in **append** mode, so a resumed run extends its own history
rather than erasing what it is resuming from.

---

## Memory and speed

```bash
--grad-checkpointing   # recompute activations during backward
--batch-size 64        # the other lever
```

The `ndarray` CPU backend computes in f32 only. `precision.rs` can *emulate*
bf16/f16 for accuracy studies at inference, but it is not a speedup and does
not apply to training — see [Precision & I/O](Precision-IO.md).

---

## Reproducibility

`--seed` seeds both the host RNG (block choice, sigma draws, data sampling) and
the on-device RNG (initialization, dropout). Two runs with the same config and
seed produce bit-identical checkpoints — the same content hash, which is a
convenient way to check — and a resumed run is bit-identical to an
uninterrupted one (above).

What a run was is recorded, not remembered: the state file carries the git
revision the binary was built from (`-dirty` if the tree had changes), the
compiler, the Burn version, and a hash of the dataset. `dblocks bench --json`,
`dblocks sweep --json` and `dblocks lm bench --json` write the same
provenance into every experiment record, with every raw trial, so a number
can be traced to the machine, build, data and seed that produced it. None of
the defaults in `TrainConfig` has been tuned; [Claims](Claims.md) lists them
as UNKNOWN with the sweep that would settle each.

Stochastic solvers take their noise from the caller's RNG rather than the
backend's, so a seeded DDIM trajectory replays exactly.

---

## Field notes: diffusion labs (what ports, what does not)

- **Absorbing diffusion for text, never uniform** (D3PM, Austin et al.:
  uniform schedules waste capacity on text). The byte LM path has no MASK
  token and no masked-diffusion objective yet — that is the prerequisite
  project, not a flag.
- **Weight by surprisal, not uniform time** (DiffusionBERT spindle beat
  `(T-t+1)⁻¹` 77.5 vs 125 PPL): the sigma-importance sampler
  (`--importance-bins`) is this idea on the continuous side; keep it on for
  text-adjacent data.
- **Train the denoiser target, normalize per noise level** (EDM/EDM2): the
  `F`-target with unit weight plus `--uncertainty` per-sigma normalization
  is already the loop's shape; the EDM2 follow-up (post-hoc power-EMA
  snapshots instead of one fixed decay) is not built.
- **Masked-CE only, `t ~ U`, semi-AR blocks** (MDLM/LLaDA/BD3-LM recipe:
  loss on masked tokens, block-diffusion sampling, low-confidence remask
  with top-k margin re-scored on current context). Applies the day the MASK
  objective lands; the block machinery (`L' = 4–8`) already matches BD3.
- **One trunk, two losses** (Transfusion/Show-o: causal AR head +
  diffusion head, separate masks, summed losses) is the shape a future
  AR+denoise byte trunk takes — not built.
- **Inference: Heun for quality, DPM++2M for budgets, annealed guidance
  with thresholding** (high-order solvers without data-prediction +
  multistep + thresholding fail under large guidance; high CFG
  oversaturates). Already the sampler set; annealed `--guidance` is not.
- **Data beats denoiser size** (Imagen: frozen T5-XXL + fused dense
  captions; hard negatives fix attribute binding): recaption code comments
  densely and mine swapped-attribute blocks rather than adding depth.
- **Watch diversity vs steps** (SEDD follow-up: entropy *falls* with steps
  under FP bugs): if samples get more stereotyped with more steps, check
  precision before blaming the schedule.

## See also

- [Quality Gate](Quality-Gate.md) — what is verified and why
- [Loss Reduction](Loss-Reduction.md) — schedules, EMA, uncertainty weighting, importance sampling
- [Configuration](Configuration.md) — every flag
- [Consistency Training](Consistency-Training.md) — the four residuals in detail
- [Block Distillation](Block-Distillation.md) — teacher/student setup
