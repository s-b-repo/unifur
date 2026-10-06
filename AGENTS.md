# AGENTS.md — how to work in this repository

Machine-readable rules first. Everything in the "The law" section below is
enforced by a tool that fails the build. If a rule here and a tool disagree, the
tool is right and this file is stale — fix this file in the same change.

**Run `make gate` before you stop.** It is the whole contract: format, clippy
with warnings as errors, the test suite, the pattern audit, the docs, and the
numerical certificate suite. It takes several minutes. CI runs exactly these
targets, so a green `make gate` and a green CI are the same command.

Current state of the tree: 719 unit tests, 38 integration tests, 208/208
certificates. If your change makes a number smaller, that is the finding.

---

## The law

### 1. Errors are returned, not raised

```rust
// No. Neither of these is accepted anywhere outside a test.
let weights = self.experts[0].unwrap();
anyhow::ensure!(count > 0, "need at least one expert");

// Yes: return the error, and name the value that made it fail.
let weights = self.experts.first().ok_or_else(|| anyhow::anyhow!(
    "a mixture needs at least one expert, this one has {}",
    self.experts.len()
))?;
```

`unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!` and `unreachable!` are
**denied** in `Cargo.toml`'s `[lints]` table. That is a compiler error, not a
review comment.

The exemptions are narrow and scoped:

- **A `#[cfg(test)]` module** carries a file-scoped `#[allow]` for exactly these
  lints, with a comment saying why. Production code in the same file is still
  denied. There is no crate-root exemption, so a new module cannot inherit one.
- **`tests/*.rs`** is a test binary and carries the same grant for the same
  reason.
- **`assert!` is allowed.** It states an invariant the caller established, and
  there is no error channel at the point it fires. Prefer it to `unwrap` for
  "this cannot happen, and if it does the message should say why".

### 2. Errors are not swallowed

```rust
let _ = f();                        // denied
if let Ok(v) = f() { /* ... */ }    // denied
```

A `Result` that is genuinely not needed has to say so: `drop(f())` with a comment
naming why the value does not matter. Silence is how a failure becomes a number
somebody trusts.

`std::process::exit` is denied for the same reason the crate returns
`Result<()>` from `main`.

### 3. Numerical claims are certificates

Every load-bearing mathematical identity goes in `src/verify.rs` as a theorem with
a residual, not as an ad-hoc test. An exact identity gets tolerance `0.0`. A new
solver, schedule, preconditioning constant or routing rule is not done until
`dblocks verify` states and checks it.

```rust
out.push(cert(
    "solver",                       // group
    "euler_matches_the_closed_form", // stable name
    "Euler on y' = -y reproduces the closed form to the tolerance.", // the claim
    residual,                        // 0.0 means exactly satisfied
    1e-6,                            // tolerance; 0.0 for an exact identity
));
```

A certificate that cannot be set up is a **failing** certificate, not a skipped
one — hence `checks_or_failed`, which turns a group that returns `Err` into a
certificate with an infinite residual. Certificate groups return
`anyhow::Result<Vec<Certificate>>` for that reason.

### 4. Suppressions are visible and cost something

`clippy::allow_attributes` is deliberately **not** enabled — it would flag the
test grants above, and a rule that makes you exempt your own exemptions is a rule
you delete. `./audit-bad-patterns.sh --strict` does that job instead, where a
grant inside a test module is distinguishable from one in production code.

`#[allow(clippy::all)]`, `#[allow(unused)]`, `#[allow(dead_code)]` and a
crate-level `#![allow(...)]` in `src/` are all failures. Fix the code the
attribute was covering.

One finding may be waived for one line, and the waiver must carry a reason:

```rust
let scale = (v / max * 255.0).round() as u8; // audit-allow: mirrors Nf4Tensor::decode exactly
```

A bare `// audit-allow:` waives nothing, because the reason is the part worth
reading.

### 5. Config is validated, not assumed

A constructor that can be handed a bad config returns `anyhow::Result` and names
the field at fault. `LanguageModel::new`, `DblockClassifier::new`,
`ViTDiTModel::new`, `GeometricReasoner::new` and `LmConfig::cost` are all
fallible for this reason. If you add a constructor that validates, it returns
`Result` too.

New `serde` config fields get `#[serde(default)]`, so a checkpoint sidecar
written by an older build still parses.

---

## Things that are easy to get wrong here

**Burn 0.21 API.** These cost time to learn once:

- `unsqueeze_dim::<2>(1)` — a bare `unsqueeze` hits dim 0.
- `squeeze::<1>()` takes no argument.
- `from_floats` needs rank-nesting to match the literal; rank 1 then `reshape`.
- `repeat_dim` *tiles*; document any grouping assumption it implies.
- `argmax` keeps its reduced dim. `reshape` before reading a scalar.

**Weights must be forced before they are cloned.** `force_initialization` exists
because Burn initialises lazily: `Param::clone` on an unforced parameter draws a
*different* value while keeping the same id. Clone-then-compare silently
produces two different models. Call it first.

**Integration tests run single-threaded, and that is load-bearing.** Burn's
backend RNG is a process-wide global, so two tests drawing from it at once makes
a seeded run unreproducible. `tests/resume.rs` asserts bit-identical resume and
can only do so because its binary is serialized. Do not "fix" the slowness by
raising the thread count.

**Temp files need unique names.** Parallel tests share `/tmp`; the crate's test
helpers include the process id and a counter.

**`pyproject.toml` is inert.** This is a Rust crate. The Python snippets in the
README are the original design specification, not files that exist here. Do not
try to `pip install -e .` or add Python tests — nothing will run them.

**Two modules are not `mod`-declared and never compile:** `src/geomnonclid.rs`
and `src/qwentrain.rs`. "Clippy is clean" says nothing about them. If you wire
one in, it enters the build and the gate applies to it.

---

## Where things are

| Path | What lives there |
|---|---|
| `src/verify.rs` | the certificate suite — the quality gate's actual subject |
| `src/main.rs` | the `dblocks` CLI; every subcommand returns `Result` |
| `Cargo.toml` `[lints]` | the enforced rule set, with the reasoning inline |
| `audit-bad-patterns.sh` | the checks the compiler cannot make |
| `Makefile` | `make gate` and the targets it is made of |
| `.github/workflows/ci.yml` | the same targets, on a clean machine |
| `.githooks/pre-commit` | the fast subset; `make install-hooks` to enable |
| `src/lib.rs` | the crate overview and the error-handling contract |

---

## Working with someone else in the tree

Another agent or a teammate may be editing at the same time. Before you start:

```bash
git status --short          # what is already modified, and by whom
git log --oneline -3
```

Re-read any file you are about to edit — it may have changed since you last
looked. Do not revert a modification you did not make; ask, or work around it.

## Scope discipline

Do not refactor code the task did not ask you to touch. This repository's gate
turns an unrelated style change into a whole-file diff, and a reviewer has to
read both. If something nearby is genuinely wrong, say so rather than fixing it
in the same change.

Do not add `.md` files. The contract lives in rustdoc and in certificates;
`docs/TEAM.md` says the same.

If you find yourself wanting `#[allow]`, `unwrap`, or `as` without a guard: the
answer is usually a type change, not an attribute. `push_kv` returning what it
just stored, and a zero-width "no state" tensor instead of an `Option`, each
removed an `expect` from the crate without making a caller harder to write.