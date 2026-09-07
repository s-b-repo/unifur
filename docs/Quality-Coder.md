# Quality Coder — design document

Status: **design + scaffolding only.** No model is trained, no weights are
produced, no compute is spent beyond compiling the crate. This document is the
contract; code lands behind it; you read it and decide whether to proceed.

## What the model is for

Given a source file that contains one or more quality defects — error
swallowing, dead code, broken memory, removed functionality, suppressed
diagnostics, hardcoded secrets, broad catches, the patterns the existing
[`crate::antipattern`] module already names — output a **minimal patch**
that fixes the defect *without* deleting real functionality.

The key word is **repair**, not **rewrite**. A common failure mode in
code-generation systems is to encounter a defect and produce a new file
that lacks the broken behavior entirely. That is *not* a fix; it is
deletion of the feature the code was trying to provide. The model is
explicitly trained to refuse that: when given `try: ... except:
return None`, it is not allowed to emit `try: ... except: return None`
with the `try` body deleted, and it is not allowed to emit `return None`
unconditionally. It must keep the failing path live and route the error
somewhere it can be handled.

This is the task the user described: *"agentic coding with no error
swallowing, no dead code, no bad memory, or any other bad pattern like
removing code instead of wiring if it's broken."* The last clause is the
part most code-generation systems fail at, and the part the training
data and eval are built around.

## What the model is *not*

- **Not a code generator from scratch.** No "write me a function that
  does X" prompts. The model takes existing broken code and produces a
  patch.
- **Not a security auditor.** It is not trained to find *new* defects;
  it is trained to fix *known* ones. The input includes a defect
  description (see [Inputs](#inputs)).
- **Not autonomous.** It does not run tests, does not invoke linters,
  does not iterate on its own output. It is a single-shot refiner. The
  agent loop is the caller's responsibility.
- **Not a substitute for code review.** It produces patches; a human
  reviews them. The eval harness measures how often the patch is
  correct, not whether it should be merged.

## Inputs

A single inference call takes three pieces:

1. **Source file.** The full text of the file to repair. The model sees
   the whole file, not a snippet, so it can reason about imports,
   helper functions, and the surrounding context that the defect
   touches.
2. **Defect list.** One or more named defects from the antipattern
   catalog: `error-swallowing`, `broad-catch`, `dead-code`,
   `uninitialized-memory`, `use-after-free`, `memory-leak`,
   `suppressed-diagnostics`, `hardcoded-secret`, etc. Each entry
   carries the line range or token span where the defect was found.
3. **Test command (optional).** When provided, the prompt asks the
   model to produce a patch that the test command would still pass.
   When omitted, the model uses the defect description alone.

The defect list is what makes the task tractable. The model is not
asked to *find* defects; it is asked to *fix named ones*. A defect
classifier (which could be the [`crate::codequality`] module I just
built, or a separate linter) produces the list; the refiner consumes
it.

## Outputs

A unified diff. Specifically, the unified-diff format that `git apply`
and `patch -p1` accept. The output is *not* free-form text and not
"here's the fixed file": the unified-diff format is the contract
because (a) it is unambiguous about what changed, (b) it forces the
model to express the *minimal* edit (a full-file rewrite is harder to
express in a diff than a few `+`/`-` lines), and (c) the eval harness
can apply the diff and re-run the tests automatically.

When the model believes the only correct response is to refuse the
fix (e.g., the defect is intentional behavior the user should not
silently undo), it emits an empty diff and a one-line reason. The
reason is read by the agent, not by the test suite.

## Training data

Three public sources, combined:

### CodeReviewer (Microsoft, 2022)

- **What it is.** ~14k GitHub pull requests with human-written review
  comments. Each comment names a defect and proposes a fix.
- **How it is used.** Pairs are constructed as `(pre-PR file, review
  comment → proposed fix)`. The pre-PR file is the broken input; the
  comment names the defect; the fix is the target.
- **Coverage.** Strong on Python and Java, weak on Rust, JS, and
  systems languages.
- **License.** Each PR inherits its repo's license; the dataset is a
  redistribution. Use requires respecting the per-PR licenses, which
  means filtering by license at load time and refusing to train on
  ambiguous ones.

### SWE-bench (Lite)

- **What it is.** 300 real GitHub issues, each paired with the patch
  that resolved them, drawn from popular Python repositories
  (django, flask, requests, etc.).
- **How it is used.** The pre-patch file is the broken input; the
  issue text names the defect in prose; the gold patch is the target.
  SWE-bench Lite is small (300 pairs) but high quality: every patch
  is one a maintainer accepted.
- **Coverage.** Python only. Excellent for the "don't remove code,
  wire it" evaluation.
- **License.** MIT/BSD, permissive.

### Code-Feedback (m-a-p)

- **What it is.** 156k multi-turn examples where a model generated
  code, executed it, observed a failure, and was told what went
  wrong. Each turn is `(code, execution result, feedback, corrected
  code)`.
- **How it is used.** The first turn's `(code, feedback)` is the
  broken input; the corrected code is the target. The "feedback"
  field is rephrased into a defect name where possible.
- **Coverage.** Mixed languages, mixed quality. Needs aggressive
  filtering: the "corrected" examples are not all correct.
- **License.** Apache-2.0.

### Mixing strategy

The three sources are mixed 1:1:1 by default. CodeReviewer and
SWE-bench are higher quality per-example; Code-Feedback is much
larger and noisier. A held-out validation split is taken from each
source separately so eval is per-source and not dominated by the
largest source.

A small **synthetic** slice is added: take clean code from
[`crate::antipattern`]'s negative examples, inject a known defect
from the rule catalog, and train on the (defective code, defect
name, fixed code) triple. This slice is reproducible, has zero
license ambiguity, and is the slice the unit tests exercise.

## Model

Base model: **Qwen2.5-Coder-1.5B-Instruct**. Chosen because:

- Permissive license (Apache-2.0 for the base, custom for the
  Instruct variant).
- ~1.5B params fits on a 24GB GPU for fine-tuning with LoRA, and on a
  16GB GPU for inference with 4-bit quantization.
- Strong baseline on HumanEval and MBPP out of the box, so we are
  fine-tuning a competent coder rather than building one from
  scratch.
- Architectural family is well-supported by the standard
  HuggingFace + PEFT + TRL stack; nothing exotic.

The model is fine-tuned with **LoRA** (rank 16, alpha 32, target
modules: q_proj, k_proj, v_proj, o_proj) rather than full
fine-tuning. The reasoning is that a 1.5B model trained on ~14k +
300 + 156k examples has ~170k gradient steps available, which is
enough for LoRA to specialize without overfitting but not enough to
rewrite the base. Full fine-tuning on this scale tends to produce
catastrophic forgetting of the base model's general competence.

## Evaluation

Three metrics, each measured on a held-out split of each source:

1. **`apply_patch_success`.** The model's unified diff is applied to
   the original file with `git apply --check`. A patch that fails to
   apply is wrong by construction. *Target: > 99%.*

2. **`lint_clean_rate`.** After the patch is applied, the result is
   scored by the [`crate::codequality`] module. The patch "improved
   quality" iff the overall score is higher than the original file's
   score and no new dimension regresses below 0.5. *Target: > 80%
   of patches improve or hold every dimension.*

3. **`test_preserved`.** When the SWE-bench-style "test command"
   input is provided, the patch is applied and the test command is
   run. A patch that breaks previously-passing tests has removed
   functionality, which is the failure mode the task exists to
   prevent. *Target: > 95% of patches preserve previously-passing
   tests.*

A composite score `quality_coder_v1 = 0.1 * apply + 0.5 * lint + 0.4
* test` is the headline number reported on each eval run. Numbers
below the targets trigger a model-quality investigation before the
checkpoint is published.

A **human review sample** of 100 randomly-selected patches is read
by the author once per release. This is not a metric; it is a sanity
check that the automated metrics are not gaming themselves.

## Success criteria for the first release

A release is "v1.0" when:

- The scaffolding compiles cleanly with `cargo build --release`.
- The dataset adapters load all three sources and produce the
  expected number of `(input, target)` pairs, reported in the eval
  harness.
- The base model + LoRA fine-tuning pipeline trains for at least one
  epoch without divergence on a single GPU.
- The eval harness reports all three metrics on the held-out split.
- The composite score on the held-out split is above the targets
  listed above.
- A patch from the held-out set can be applied by `git apply` to a
  clean checkout of the corresponding repo and produces a file that
  passes the original test command.

Until all six are checked, the model is not released.

## Honest limits

- **1.5B is small.** It will produce patches that look right but
  introduce subtle bugs. The human review sample exists because
  automated metrics cannot catch everything.
- **The training data is biased toward Python and Java.** Rust,
  Go, and JS coverage is thin. The first release will explicitly
  refuse inputs in languages where the eval is below target rather
  than shipping a broken model.
- **The defect list is the model's whole world.** A patch is only as
  good as the defect description it was given. A defect classifier
  that misses the actual problem produces a patch that fixes the
  wrong thing. The pipeline assumes the upstream classifier is
  trustworthy.
- **The training data contains GPL-licensed code.** CodeReviewer
  pulls from arbitrary GitHub repos. A license filter is applied at
  load time, but it is imperfect, and any released model carries the
  risk of having seen GPL code during training. This is a known
  unresolved problem in code-LLM training; it is documented here so
  the limitation is visible.
- **The eval is not the deployment.** Real refiner usage includes
  agent loops, multi-file edits, tool calls, and human feedback.
  The first version is single-shot, single-file, no tools. Expanding
  any of these is a separate design exercise.

## What this document is *not*

- Not a justification for training a 700B-parameter model by
  combining 3B blocks. See [`Model-Parallelism.md`](Model-Parallelism.md)
  for the honest answer to that question.
- Not a commitment to train anything yet. The scaffolding compiles,
  the data adapters are tested with fixtures, the eval harness
  produces numbers from synthetic data — but no weights are
  produced until you sign off on this document.