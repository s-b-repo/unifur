# Multi-Source Training

Several datasets or corpora in one run, several teachers at once, an
open-weight source of *negative* examples, and averaged checkpoints as a
starting point. Everything a run learns from, and how much, is a weight.

> **In this repository.** `mix.rs` (`MixMode`, `MixWeights`, `BatchOrigin`,
> `SourceStats`, `CorpusMix`, `concat_batches`), `merge.rs` (`merge_into`,
> `merge_checkpoints`, `ParamSnapshot`), `distill.rs`
> (`DblockClassifier::distill_step_multi`, `teacher_mixture`,
> `soft_target_kl_probs`), `lm.rs` (`LanguageModel::negative_proposals`,
> `next_token_step_full`, `ExtraNegatives`, `Distillation`), `train.rs`
> (`TrainConfig::extra_datasets` / `dataset_weights` / `mix_mode` /
> `extra_teachers`, `train_lm_mixed`, `LmTrainInputs`). CLI: `dblocks train
> --dataset a --dataset b --dataset-weights 0.7,0.3 --mix composite --teacher
> t1 --teacher t2`, `dblocks lm train --corpus a --corpus b --teacher t
> --negative-teacher n`, `dblocks merge`, `dblocks lm merge`. Certificates:
> the `multisource` group.

---

## Several sources

Two ways to share a run, both driven by the same weights:

| `--mix` | Every step | Reporting |
|---|---|---|
| `mixture` (default) | draws its batch from **one** source, chosen by weight from the host RNG — so the choice is part of the resumable state | `source` in the JSONL is the index that supplied the batch |
| `composite` | slices its batch from **every** source, sized by weight, exactly (largest-remainder apportionment: the slices always sum to the batch) | `source` is `-1`; the per-source table credits the loss in proportion |

Image sources must share the image size and label count; the run refuses a
mismatch at open. Corpora need nothing in common: each keeps its own labels,
and a window from an unlabeled corpus carries zero penalty weight while a
labeled one carries its manifest's weights.

Every source's bytes are hashed into the training state, so a resume checks
the whole list, and the end of a run prints a per-source table (batches,
samples, mean loss) — a source whose loss never falls is invisible in the
aggregate and obvious here.

```bash
dblocks train --dataset cifar100 --data-dir cifar --dataset synthetic \
              --dataset-weights 0.9,0.1 --mix mixture
dblocks lm train --corpus code.bin --corpus prose.bin --corpus-weights 2,1 --mix composite
```

## Several teachers

`--teacher` is repeatable on both trainers. The image trunk distils toward the
weighted **mean of the teachers' substep latents** and the weighted **mixture
of their softened distributions**; the language model adds
`KL(mixture || student)` at the distillation temperature, times its weight,
to the corpus loss. One teacher at any weight takes the single-teacher path
unchanged, so the earlier objective is a special case bit for bit.

```bash
dblocks train --objective distill --teacher a.mpk --teacher b.mpk --teacher-weights 2,1
dblocks lm train --corpus code.bin --teacher a.mpk --teacher b.mpk --distill-weight 0.5
```

## A negative teacher

Phase 24's negatives come from a labeled corpus. A **negative teacher** is the
open-weight source of the same signal: a frozen model whose confident
next-token choices are what the student is charged for. At every position the
negative model's arg-max token, where it is at least `--negative-confidence`
sure, becomes a charged token — **unless it equals the corpus target**, which
is never contradicted. The charge is the same bounded unlikelihood term
`-log(1 - p)` the labeled path uses; the token is not a target, so nothing is
removed from the likelihood.

```bash
dblocks lm train --corpus code.bin --negative-teacher bad-habits.mpk \
                 --negative-confidence 0.6 --negative-penalty 1.0
```

Only checkpoints written by `dblocks lm train` can be loaded; a
Llama-class negative teacher would need the weight loading the roadmap leaves
out of scope. The mechanism is what is shipped; on this crate's hardware the
claim that it improves a real model is UNKNOWN, like every other quality
claim in [Claims](Claims.md).

## Merging checkpoints

`dblocks merge --input a.mpk --input b.mpk --weights 1,1 --out dir` averages
same-architecture checkpoints parameter by parameter (pairing by traversal
order, as the EMA does, with a shape check at every parameter) and writes a
new content-addressed model whose state directory records the parents'
hashes and weights. Merging a checkpoint with itself is the identity to the
bit; merging is linear in the weights.

## What is certified

| Certificate | Claim |
|---|---|
| `mixture_draws_follow_the_weights` | 4000 draws of a 3:1 mix put 0.75 of the batches on the heavy source, within 3.6 binomial σ |
| `composite_slices_are_exact_apportionment` | slices sum to the batch for every batch size, each within one item of its share |
| `single_source_mix_is_the_plain_corpus` | one corpus in a mix draws exactly what it draws alone |
| `teacher_mixture_of_one_is_its_own_softened_distribution`, `teacher_mixture_is_a_distribution`, `probability_target_kl_agrees_with_logit_target_kl` | the mixture target is what a single teacher was, and a distribution |
| `negative_teacher_below_its_confidence_is_the_plain_loss`, `negative_teacher_never_contradicts_the_corpus`, `negative_teacher_step_ends_below_plain_step` | the charge is off exactly when nothing is proposed, never fights the corpus, and moves the proposals down relative to a plain step |
| `self_distillation_is_zero` | a model distilled toward itself costs nothing |
| `merge_of_identical_checkpoints_is_the_identity`, `merge_is_linear_in_its_weights` | merging is the identity on equal inputs and linear otherwise |

---

See also: [Negative Supervision](Negative-Supervision.md) · [Block Distillation](Block-Distillation.md) ·
[Training Guide](Training-Guide.md) · [Configuration](Configuration.md) · [Home](Home.md)
