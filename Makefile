# The quality gate, as make targets.
#
# `make gate` is the whole contract: if it passes, the change is submittable. CI
# runs exactly these targets, so "it passed locally" and "it passed in CI" are the
# same command -- there is no second, weaker list for a laptop.
#
# Every target here is also runnable on its own, which is the point: a failure
# three hours into `gate` should have been findable in three minutes.

CARGO ?= cargo
# Integration tests run single-threaded, and that is not a performance setting:
# Burn's backend RNG is a process-wide global, so two tests drawing from it at
# once makes a seeded run non-reproducible. `tests/resume.rs` asserts exactly
# that property, which is only assertable because the binary is serialized.
TEST_THREADS ?= 1

.DEFAULT_GOAL := help
.PHONY: help gate fmt fmt-check lint lint-strict test test-lib test-integration \
        audit doc verify clean all-features hooks install-hooks

help: ## List the targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sed 's/:.*## /\t/' \
	  | awk -F'\t' '{printf "  %-20s %s\n", $$1, $$2}'

# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

gate: fmt-check lint-strict test audit doc verify ## Everything CI runs, in order
	@echo
	@echo "gate: green."

all-features: ## Check every optional feature compiles clean, not just the default set
	$(CARGO) clippy --all-targets --all-features -- -D warnings

# --------------------------------------------------------------------------
# Formatting
#
# rustfmt only. `cargo fmt` also formats generated code if any target has a
# rustfmt.toml pointing elsewhere; there is none, so this is `cargo fmt` spelled
# out for the two halves it would otherwise do at once.
# --------------------------------------------------------------------------

fmt: ## Format in place
	$(CARGO) fmt --all

fmt-check: ## Fail if anything is unformatted
	$(CARGO) fmt --all -- --check

# --------------------------------------------------------------------------
# Lints
#
# Two levels on purpose. `lint` is what you run while working -- it surfaces
# problems without failing your build over a lint you have not seen yet.
# `lint-strict` is what CI runs and what `gate` runs: every warning is an error,
# including ones this repository has never seen. Without the `-D warnings`, a
# newly-introduced lint is a warning nobody reads.
#
# The error-handling rules themselves are `deny` in Cargo.toml's `[lints]` table,
# so they apply at both levels. `-D warnings` is what stops *other* lints.
# --------------------------------------------------------------------------

lint: ## Clippy over every target, warnings shown
	$(CARGO) clippy --all-targets

lint-strict: ## Clippy with every warning promoted to an error (the CI level)
	$(CARGO) clippy --all-targets -- -D warnings

# --------------------------------------------------------------------------
# Tests
#
# `test` is the full suite. The `--test-threads=$(TEST_THREADS)` is load-bearing,
# not a speed trade: see the note at the top of this file.
# --------------------------------------------------------------------------

test: test-lib test-integration ## Every test target

test-lib: ## Unit tests, including the numerical certificate suite
	$(CARGO) test --lib -- --test-threads=$(TEST_THREADS)

test-integration: ## Integration and resume tests
	$(CARGO) test --tests -- --test-threads=$(TEST_THREADS)

# --------------------------------------------------------------------------
# Static audit
#
# The one check clippy cannot do: is there a suppression, a hard-coded secret,
# or an injected command somewhere the compiler is happy about? Reports
# production findings as failures; test findings are reported and not fatal.
# --------------------------------------------------------------------------

audit: ## Pattern audit over the tree (production findings fail)
	./audit-bad-patterns.sh --strict

# --------------------------------------------------------------------------
# Docs and certificates
# --------------------------------------------------------------------------

doc: ## Build the docs with warnings as errors
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --no-deps --all-features

verify: ## Run the numerical certificate suite through the built binary
	$(CARGO) run --release --quiet --bin dblocks -- verify

# --------------------------------------------------------------------------
# Git hooks
#
# Opt-in, because a hook that rewrites someone's working tree without warning is
# worse than no hook. `install-hooks` is what you run once after cloning.
# --------------------------------------------------------------------------

install-hooks: ## Point git at .githooks/ (run once after cloning)
	git config core.hooksPath .githooks
	@echo "hooks installed. 'make hooks' runs the fast subset by hand."

hooks: ## The fast checks, as the pre-commit hook runs them
	@$(MAKE) --no-print-directory fmt-check lint-strict audit

clean: ## Remove build artefacts
	$(CARGO) clean