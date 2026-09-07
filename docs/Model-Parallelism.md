# Model parallelism, block-wise training, and why 233×3B ≠ 700B

Status: explanatory. This document answers a recurring question —
*"can we train a 700B-parameter model by training 3B blocks
independently and combining them?"* — with what is actually true
about model parallelism rather than what would be reassuring. The
short answer is **no**, and the long answer explains why, what
*can* be done, and where this codebase's existing block-wise
training actually sits in the landscape.

## The question

The idea is appealing: a 700B-parameter model is too big to train
on one machine. A 3B-parameter model fits comfortably. If we could
train 234 independent 3B blocks and stitch them together, we would
get 702B parameters trained with 234× the compute of a single 3B
run, distributed across commodity hardware. No special
interconnect, no model-parallel runtime, just `for i in 0..234:
train_block(i)`.

This section explains why that does not produce a 700B model. The
remaining sections describe what does.

## Why independent blocks don't compose

A 700B-parameter model is a function
`f_θ : tokens → logits` with θ ∈ ℝ^(7×10¹¹). The parameters θ are
not 234 disjoint groups; they are coupled by every gradient step of
training. A concrete example makes this concrete.

### The loss surface is one function of all parameters

Consider a network with parameters (W₁, W₂, W₃), each a 1024×1024
matrix. The loss L depends on **every** entry of W₁, W₂, W₃.
Gradient descent updates them jointly:

```
W₁ ← W₁ - η ∂L/∂W₁    (depends on W₂, W₃ through the forward pass)
W₂ ← W₂ - η ∂L/∂W₂    (depends on W₁, W₃ through the forward pass)
W₃ ← W₃ - η ∂L/∂W₃    (depends on W₁, W₂ through the forward pass)
```

If we freeze W₂ and W₃ and train only W₁ to convergence, we find
the W₁ that minimizes L *given* the random W₂, W₃. Then we freeze
W₁ and the new W₂' and train only W₃. The W₃' we find minimizes
L given W₁ and W₂'. This is **not** the same W₃ the joint training
would have found — the joint training would have moved all three
parameters in coordinated directions that none of the individual
convergence points reaches.

A 700B-parameter model's parameters are not "234 chunks of 3B
each." They are a 700B-dimensional vector that gradient descent
walks. Walking each 3B-dimensional projection independently to its
optimum and concatenating produces a vector that is, in general, far
from the joint optimum.

### An explicit numerical example

For a 2-block model with blocks (A, B), the joint optimum is
(A\*, B\*) such that ∂L/∂A = 0 and ∂L/∂B = 0 simultaneously. The
"train A, then train B" procedure finds:

- A₁ = argmin_A L(A, B₀)         for some random init B₀
- B₁ = argmin_B L(A₁, B)

This is a coordinate-descent on L. It converges under convexity but
to a *different point* than joint gradient descent. In deep
networks, where L is non-convex, it does not even converge — the
two steps chase each other indefinitely. The block-wise training in
this codebase, which is the closest analog, is described below; it
is not coordinate descent.

### What block-wise training actually buys you

This repository has a block-wise training scheme in
[`src/dblock.rs`] and [`src/lm.rs`]. It does **not** train
independent blocks and concatenate them. It trains one block at a
time on the full loss, with the other blocks frozen — for *one
gradient step*, not to convergence. Then it rotates to the next
block. After K rotations, every block has been updated K times
against the live values of the others.

This is **stochastic block coordinate descent**, not independent
block training. The crucial difference: at every step, the block
being trained sees the *current* values of the other blocks. The
end result approximates joint gradient descent for memory-constrained
training (only one block's activations are kept in memory at a
time).

A 700B block-wise model would still be 700B parameters; the scheme
just lets you train it on a machine that cannot hold all 700B
parameters' activations in memory simultaneously. The parameters
themselves, and their joint optimization, are unchanged.

### What real model parallelism does

The four production-grade model-parallel schemes are:

| Scheme | What is split | Coupling | Hardware requirement |
|---|---|---|---|
| **Data Parallel (DDP)** | batch | parameters identical, gradients all-reduced | one full model replica per GPU |
| **Pipeline Parallel** | layers | activations passed stage-to-stage; bubbles in the pipeline | fast interconnect between stages |
| **Tensor Parallel** | individual matrix multiplies | all-reduce on every layer's activations | very fast interconnect (NVLink/IB) |
| **Expert Parallel (MoE)** | expert modules | all-to-all on routing decisions | moderate interconnect |

None of them is "train independent chunks and concatenate." They all
require synchronized communication at every gradient step. The
compute and memory savings come from distributing the work, not
from decoupling the optimization.

A 700B model trained with Tensor Parallel on 8 GPUs each holding
~90B of parameters is one model with 700B parameters that have
been jointly optimized. The "blocks" are slices of one tensor that
get sliced and re-stitched at every forward and backward pass; they
are not independent training runs.

## What *can* be done with limited hardware

For a researcher with commodity hardware who wants to do meaningful
work on large-model questions, the practical options are:

1. **Fine-tune a small open model.** Qwen2.5-Coder-1.5B,
   DeepSeek-Coder-1.3B, StarCoder2-3B. Fits on a single GPU with
   LoRA. This is the path the
   [`Quality-Coder.md`](Quality-Coder.md) document takes. It
   produces a useful tool without claiming to scale to AGI.

2. **Train a small model from scratch on a narrow task.** The
   existing [`codequality`] module is one example: a classifier,
   not a generator, that fits the codebase's design and gets the
   job done in 700 lines.

4. **Use a hosted 700B model via API.** Don't train; just *use* a
   700B model. Anthropic, OpenAI, Together, and others expose
   large models at API-call cost. Useful for evaluation, less
   useful for producing a deployable artifact.

5. **Borrow a 700B model and fine-tune via LoRA on a single large
   GPU.** With QLoRA, a 70B model fits in 24GB for inference and
   fine-tuning. A 700B model needs ~1.5TB of GPU RAM at full
   precision; ~150GB at 4-bit; still out of reach for most
   researchers but achievable on multi-node clusters.

What does **not** work, and is worth saying so clearly:

> "Train 233 independent 3B models, concatenate the weights,
> call it 700B."

This produces 233 unrelated 3B models whose weights happen to be
arranged in the same order as a 700B model's weights would be. The
result has the parameter count of a 700B model and the capability
of a 3B model. Worse, it is dishonest: it claims a capability that
does not exist and consumes compute that could have produced a
real result.

## Where this codebase's block-wise training sits

The crate's [`src/dblock.rs`] and [`src/lm.rs`] implement
**block-wise training in the original DiffusionBlocks sense**: each
step trains one block on one noise window, with full joint
optimization across all blocks over the course of a run. The
parameters are coupled. The block structure is a memory-and
-scheduling choice, not a parameter-independence claim.

The block sigma schedule in [`src/sigma.rs`] makes this concrete:
block `b` is responsible for noise levels in `(σ_b, σ_{b+1}]`, and
the loss weighting, preconditioning, and gradient flow all assume
every block is being optimized against the others' *current*
values. Removing any block from the optimization would invalidate
the schedule.

This is exactly the right structure for the codebase's goals
(memory-efficient training of a single model) and exactly the
wrong structure for "combine independent training runs into a
larger model." They are not different points on the same spectrum;
they are different problems.

## TL;DR

- **233 × 3B ≠ 700B.** Joint optimization is the whole point of
  large-model training; decoupling it removes the property that
  makes large models capable.
- **Block-wise training in this crate is joint optimization with a
  memory-saving scheduler.** It is not a recipe for distributing
  training across machines.
- **Real model parallelism requires synchronized communication at
  every step.** It is well-understood and well-engineered
  (DeepSpeed, Megatron, FSDP, etc.). It does not look like
  "train independent blocks."
- **For a single researcher on commodity hardware, fine-tuning a
  small open model is the high-leverage path.** That is what
  [`Quality-Coder.md`](Quality-Coder.md) does.