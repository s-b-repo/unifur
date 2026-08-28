# Training Guide

Everything `dblocks train` can do, and how to read what it reports.

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
--async-save                 # serialize on a background thread
--resume                     # newest checkpoint in --out-dir
--resume path/to/ckpt.mpk    # a specific one
```

Bare `--resume` on an empty directory starts fresh and says so, rather than
failing — a first run has nothing to resume from.

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
convenient way to check.

Stochastic solvers take their noise from the caller's RNG rather than the
backend's, so a seeded DDIM trajectory replays exactly.

---

## See also

- [Quality Gate](Quality-Gate.md) — what is verified and why
- [Loss Reduction](Loss-Reduction.md) — schedules, EMA, uncertainty weighting, importance sampling
- [Configuration](Configuration.md) — every flag
- [Consistency Training](Consistency-Training.md) — the four residuals in detail
- [Block Distillation](Block-Distillation.md) — teacher/student setup
