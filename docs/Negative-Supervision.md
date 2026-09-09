# Negative Supervision

Anti-pattern rules name the idioms a code corpus should **not** teach, label
every token that belongs to one, and the language-model loss charges the model
for the probability it assigns to those tokens instead of rewarding it.

> **In this repository.** `antipattern.rs` (`RuleSet`, `Rule`, `Category`,
> `Pattern`, `Labeler`, `LabelManifest`, `CLEAN`), `corpus.rs` (`labels_path`,
> `manifest_path`, `TokenCorpus::label_file`, `open_labels`, `window_labels`,
> `sample_batch_labeled`), `lm.rs` (`Unlikelihood`, `unlikelihood`,
> `label_weights`, `LanguageModel::next_token_loss_penalized`, the
> `penalized_*` fields of `LmMetrics`), `train.rs` (`train_lm`, `LmTrainConfig`,
> `LmTrainReport`). CLI: `dblocks lm tokenize --label`, `dblocks lm label | scan
> | rules`, `dblocks lm train --penalty`. Certificates: the `antipattern` group.
> The `codequality` module (window filter, quality regularizer) composes with
> this page's objective; see [Quality Coder](Quality-Coder.md) for the design
> those pieces serve.

---

## Why a second signal

Ordinary next-token training has exactly one signal: *make the corpus more
likely*. A corpus scraped from real repositories contains `except: pass`,
`catch (e) {}`, `# noqa`, `@ts-ignore` and `password = "hunter2"` in
quantity, and every one of them is a lesson in writing it again. Filtering
them out of the corpus loses the surrounding code, which is usually fine;
leaving them in teaches them.

Negative supervision keeps the code and flips the sign on the idiom. The plain
objective's behaviour is not a guess: `integration_negative_supervision_unlearns_error_swallowing`
trains on a corpus in which every handler swallows its error and watches
`p(pass | except:)` climb above 0.25. The same initialization trained with the
charge ends more than 20x lower, still fits the clean tokens, and no longer
continues `except:` with `pass`.

---

## Rules: context and body

A rule has two patterns. The **context** is matched but not penalized; the
**body** is matched *and* penalized.

```text
except:
    pass
```

Here the context is `except\b[^:\n]*:\s*` and the body is `pass\b`. Nothing is
wrong with `except:`; the failure is choosing `pass` after it. Labeling the
whole match would push the model away from writing `except` at all — the
opposite of teaching it to handle errors.

Every rule carries `examples` it must match and `counterexamples` it must not,
and `RuleSet::validate` checks both. `dblocks lm rules --check my-rules.json`
runs the same check on a file you extend, and the built-in set is certified
against its own examples on every `dblocks verify`.

### The pattern language

No regex crate is used — the crate is deliberately dependency-light — so
patterns are matched by a small backtracking engine over a subset of the usual
syntax:

| Syntax | Meaning |
|---|---|
| `a`, `\(`, `\\` | a literal byte (escape the metacharacters `.[]()\|{}*+?^$`) |
| `.` | any byte except newline |
| `\s` `\w` `\d` | whitespace, `[A-Za-z0-9_]`, `[0-9]`; upper case negates |
| `[abc]`, `[a-z]`, `[^)]` | a class, a range, a negated class |
| `\b`, `$` | a word boundary, the end of a line or of the text |
| `*`, `+`, `?`, `{n}`, `{n,m}` | greedy quantifiers on the preceding atom |

Groups and alternation are **not** supported. An unescaped `(`, `)` or `|` is a
parse error rather than a silent literal: two rules are cheaper than one wrong
one.

Matching runs over **token ids**, not text. With the byte-level tokenizer a
byte's id is its value, so a pattern written against text applies unchanged —
and a special token (`<bos>`, `<eos>`, `<pad>`) matches nothing, so no rule can
span two documents.

### The built-in rules

| Category | Weight | Rules | Languages |
|---|---|---|---|
| `error-swallowing` | 1.0 | 11 | Python, JS/TS, Java, C-family, Rust, Go |
| `broad-catch` | 0.5 | 3 | Python, Java |
| `suppressed-diagnostics` | 0.5 | 6 | TypeScript, Python, JS/TS, Rust |
| `hardcoded-secret` | 1.0 | 4 | any |

The weight is a multiplier on a category's charge: `0.0` labels the tokens but
charges nothing, which is how a category can be measured before it is
penalized. `dblocks lm rules` prints the whole set as JSON to extend;
`dblocks lm scan --input file.py` lists findings with line numbers, which is
the fastest way to see what a rule actually catches.

---

## Labels on disk

```bash
dblocks lm tokenize --input repo.txt --out repo.bin --label
# or, for an existing corpus:
dblocks lm label --corpus repo.bin
dblocks lm corpus --path repo.bin          # reports the label manifest too
```

`repo.labels` holds one `u8` per token in the same index space as `repo.bin`
(`0` is `CLEAN`; `k + 1` is category `k`), read with the same one-seek window as
the tokens, so a labeled window costs two reads rather than one.
`repo.labels.json` is the [`LabelManifest`]: the category names, weights and
counts, and per-rule hit counts — enough to train on the corpus without the
rule set that produced it.

Two decisions worth naming:

- **Labels are computed over the whole corpus, not per window.** A rule whose
  match would straddle a window boundary is still found, because windows are cut
  *after* labeling. `label_file` loads the corpus into memory for this; labeling
  is a one-off preprocessing step, like tokenizing, and the text was in memory
  for that too.
- **A sidecar of the wrong length is refused.** Right length is the only thing
  that keeps label `i` on token `i`; a mismatch means the sidecar was made from
  a different corpus, and every label after the first divergence would land on
  the wrong token. Likewise a run that asks for a penalty on a corpus without
  labels is an error, not a silent plain run.

`in_memory` and `streaming` corpora return identical labels for identical
windows (`antipattern/labels_follow_tokens_through_both_readers`).

---

## The objective

A labeled target is a **negative** example. The per-token term is Welleck et
al.'s unlikelihood,

```text
charge(p) = -log(1 - p)          floored at -log(eps), eps = 1e-6
```

not a negative weight on the cross-entropy. The distinction is the whole
design:

- `-w · (-log p) = w · log p` is **unbounded below**. The model can drive the
  loss to `-inf` by making one bad token impossible, and that single term then
  dominates every real target in the batch.
- `-log(1 - p)` is `0` when the bad token is impossible and grows only as
  `p -> 1`. It rewards nothing; it stops charging once the pattern is gone.

The batch objective is

```text
loss = ( Σ_clean -log p  +  alpha · Σ_flagged w · charge(p) ) / counted
```

with three properties the certificates pin:

- **A charged target leaves the likelihood term.** Rewarding and charging the
  same token would leave the gradient at whichever term is currently larger —
  a tug of war, not a signal. Each non-padding target is in exactly one sum
  (`a_flagged_target_is_charged_and_leaves_the_likelihood`, recomputed by hand
  from the raw logits).
- **`alpha = 0` is the plain loss, bit for bit**, with the flagged targets
  still counted and their mean probability still reported
  (`zero_weights_or_zero_alpha_reproduce_the_plain_loss`). That is how a plain
  run is shown to *learn* an anti-pattern rather than merely tolerate it. The
  first version removed flagged targets from the likelihood while charging
  nothing, and a plain run's `p(bad)` *fell* — the metric was measuring tokens
  the model was never shown.
- **Padding is never a negative target**, whatever the label file says about
  it; it is outside both sums.

Both terms share the denominator, so a batch with few negatives is not
dominated by them and a batch with none reduces to the plain loss exactly.

---

## Training

```bash
dblocks lm train --corpus repo.bin --steps 2000 --penalty 1.0 --log run.jsonl
```

With a `.labels` sidecar present the run opens it and reports, at every logged
step, how many targets were flagged, the mean probability the model gives them
(`p(bad)`), and the mean charge. With `--penalty 0` it trains plainly and only
measures; with a positive penalty it charges. `LmTrainReport` carries
`first_penalized_prob` and `last_penalized_prob`, which is what the integration
test compares between the two runs.

One measured detail: the unlikelihood gradient scales with `p`, so for the
first few dozen steps generalization from `try:` raises `p(bad)` faster than
the charge lowers it (0.004 -> 0.018 at step 30 in the Phase 24 measurement)
before the charge takes over. That is why the certificate about optimizer
steps is **comparative**: from the same weights and batch, one penalized step
ends with the flagged tokens strictly less probable than one plain step does
(`penalized_step_ends_below_plain_step`), which is true to first order for any
initialization — whereas "one penalized step lowers `p(bad)`" is not, and was
retired when it failed at residuals that moved with the RNG.

---

## What is not built

- **Learned detectors.** The rules are lexical, reviewable and extensible; they
  are a floor, not a parser. An AST-aware or learned labeler would catch more
  and needs labeled data this repository does not have.
- **A claim about code quality.** Unlearning 24 idioms does not make a model
  write good code. What is shown is narrower and measured: the objective drives
  the labeled idioms out while the clean tokens are still learned.

---

See also: [Language Modeling](Language-Modeling.md) · [Quality Coder](Quality-Coder.md) ·
[Configuration](Configuration.md) · [Quality Gate](Quality-Gate.md) · [Home](Home.md)
