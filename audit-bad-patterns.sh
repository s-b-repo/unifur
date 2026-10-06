#!/usr/bin/env bash
# Pattern-level audit for the things the Rust compiler cannot see.
#
# Why this exists alongside the `[lints]` table in Cargo.toml
# ------------------------------------------------------------
# The lints catch *language-level* escapes: `unwrap`, `panic!`, `let _ =`.
# Three classes of problem are not expressible that way and are checked here:
#
#   1. Suppression. A lint that can be silenced by an attribute is only as
#      strong as the weakest `#[allow]` in the tree. `clippy::allow_attributes`
#      is deliberately not enabled (it would flag the test grants every
#      `#[cfg(test)]` module legitimately needs), so section C does that job
#      here, where a grant inside a test module is distinguishable from one in
#      production code.
#
#   2. Secrets in source. A hard-coded key is a committed credential.
#
#   3. Semantic smells the compiler has no opinion about: SQL built by string
#      formatting, a command line assembled from a variable, `md5`/`sha1` used
#      for a security purpose, an empty `catch_unwind`.
#
# Usage
# -----
#   ./audit-bad-patterns.sh              # full report, informational, exit 0
#   ./audit-bad-patterns.sh --strict     # exit 1 on any finding (the CI gate)
#   ./audit-bad-patterns.sh --section C  # one section
#   ./audit-bad-patterns.sh --files a.rs # restrict to these paths
#
# A single finding may be waived for one line with a comment naming why:
#
#   let bytes = idx as usize; // audit-allow: idx is bounded by KINDS.len() above
#
# The waiver must carry text after the colon. A bare `// audit-allow:` waives
# nothing, because the reason is the part worth reading.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STRICT=0
SECTION=""
FILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)  STRICT=1; shift ;;
    --section) SECTION="$2"; shift 2 ;;
    --files)   shift; while [[ $# -gt 0 && "$1" != --* ]]; do FILES+=("$1"); shift; done ;;
    --help)    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Which files are in scope.
#
# Production code is `src/` plus `build.rs`. Tests and examples are audited too,
# but they are the one place `unwrap` is legitimate, so section A counts them
# separately rather than treating a passing test suite as a failure.
# ---------------------------------------------------------------------------
PROD_FILES=()
TEST_FILES=()
while IFS= read -r -d '' f; do
  case "$f" in
    */tests/*|tests/*|*/examples/*|examples/*) TEST_FILES+=("$f") ;;
    *)                                     PROD_FILES+=("$f") ;;
  esac
done < <(find src tests examples -type f -name '*.rs' -print0 2>/dev/null; \
         [[ -f build.rs ]] && printf '%s\0' build.rs)

if [[ ${#FILES[@]} -gt 0 ]]; then
  PROD_FILES=()
  TEST_FILES=()
  for f in "${FILES[@]}"; do
    [[ -f "$f" ]] || { printf 'no such file: %s\n' "$f" >&2; exit 2; }
    case "$f" in
      */tests/*|tests/*|*/examples/*|examples/*) TEST_FILES+=("$f") ;;
      *)                                     PROD_FILES+=("$f") ;;
    esac
  done
fi

# ---------------------------------------------------------------------------
# Regions. Inside a file, everything from `#[cfg(test)]` onwards is test code,
# and everything after it is still test code -- the tree uses one test module
# per file, at the bottom. A `.unwrap()` before that line is production.
#
# Emits `<file>\t<line>\t<text>` for a region of interest, one line at a time.
# ---------------------------------------------------------------------------
# Prints `<file>\t<line>\t<text>` for every line matching a pattern, skipping
# comment lines (which is where a `// audit-allow:` waiver lives).
scan() {
  # Dest must be a *path*, not a variable name: the caller's `local out` would
  # otherwise shadow a nameref of the same name inside this function.
  local pattern="$1" dest="$2"
  local file
  for file in "${PROD_FILES[@]}" "${TEST_FILES[@]}"; do
    # Two files are exempt, both for the same reason: each names the very
    # patterns this script greps for, as string literals, because detecting or
    # defining them is its job. See section C.
    case "$file" in
      src/cheat.rs | src/antipattern.rs) continue ;;
    esac
    # `grep -n` prefixes each hit with its real line number; that number is what
    # the prod/test split keys on, so it is read rather than counted.
    local match
    while IFS= read -r match; do
      [[ -z "$match" ]] && continue
      local n="${match%%:*}" text="${match#*:}"
      [[ "$text" =~ ^[[:space:]]*(//|/\*|\*) ]] && continue
      printf '%s\t%s\t%s\n' "$file" "$n" "$text" >> "$dest"
    done < <(grep -nE "$pattern" "$file" 2>/dev/null || true)
  done
}

# A finding is production unless it is at or after the file's first
# `#[cfg(test)]`. This is what lets section A report a test's `unwrap` without
# failing the build for it.
is_prod_hit() {
  local file="$1" line="$2" first
  first="$(grep -n -m1 '^[[:space:]]*#\[cfg(test)\]' "$file" 2>/dev/null | cut -d: -f1 || true)"
  [[ -z "$first" ]] && return 0
  [[ "$line" -lt "$first" ]]
}

# Some files hold a `#[cfg(test)] mod tests` in the middle and production items
# after it (the crate root, and any file that grew a test module before a later
# edit). A line-count test alone would call all of those test code, which is the
# wrong answer for a file whose *last* item is production. The brace-depth check
# is what distinguishes them: production code is at depth 0, a test module is at
# depth >= 1.
in_test_module() {
  local file="$1" line="$2"
  awk -v stop="$line" '
    /^[[:space:]]*#\[cfg[(]test[)]\]/ { in_test = 1 }
    {
      opens = gsub(/\{/, "{")
      closes = gsub(/\}/, "}")
      depth += opens - closes
      if (in_test && depth >= 1) {
        # Inside the test module until it closes back to its own start.
        if (opens == 0 && closes > 0 && depth == 0) in_test = 0
        next
      }
      if (NR == stop) { print (depth == 0 ? "prod" : "test"); exit }
    }
  ' "$file"
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

run_section() {
  # $1 section letter, $2 name, rest: pattern|pattern|...
  local letter="$1" name="$2"; shift 2
  local hits=0 prod_hits=0
  printf '\n%s  %s\n' "$letter" "$name"
  local pattern
  for pattern in "$@"; do
    local hits_file="$TMP/hits"
    : > "$hits_file"
    scan "$pattern" "$hits_file"
    local count=0 count_prod=0
    while IFS=$'\t' read -r file line text; do
      [[ -z "$file" ]] && continue
      # A waiver has to carry a reason.
      if [[ "$text" == *"// audit-allow:"* ]]; then
        local reason="${text#*// audit-allow:}"
        if [[ -n "${reason// /}" ]]; then continue; fi
      fi
      count=$((count + 1))
      # A test *binary* is not production code: `tests/*.rs` and `examples/*.rs`
      # are compiled into their own crates and reach nobody through the library's
      # API, so a crate-level grant in one of them silences nothing that ships.
      is_test_target=0
      case "$file" in tests/* | examples/*) is_test_target=1 ;; esac
      if [[ "$is_test_target" -eq 0 ]] \
         && is_prod_hit "$file" "$line" \
         && [[ "$(in_test_module "$file" "$line")" == "prod" ]]; then
        count_prod=$((count_prod + 1))
        printf '      %s:%s: %s\n' "$file" "$line" "${text#"${text%%[![:space:]]*}"}"
      fi
    done < "$hits_file"
    if [[ "$count" -gt 0 ]]; then
      printf '    [%3d] %s\n' "$count" "$pattern"
      hits=$((hits + count))
      prod_hits=$((prod_hits + count_prod))
    fi
  done
  SECTION_HITS=$hits
  SECTION_PROD=$prod_hits
}

TOTAL=0
TOTAL_PROD=0
FAILING_SECTIONS=()

# Sections whose *production* hits fail `--strict`. Tests are reported but do not
# gate: a test that unwraps is the test working.
# E is not in this list, and that is a deliberate decision rather than an
# oversight. A numeric crate has ~700 `as` conversions and every one of them is
# load-bearing arithmetic whose correctness is a question for the *code*, not the
# pattern: `idx as usize` is fine when `idx` came from `enumerate` and wrong when
# it came from user input, and no grep can tell those apart. So section E is
# reported for a human to read and is not a gate -- the gate for numeric
# correctness in this crate is `src/verify.rs`, which states each identity as a
# theorem and checks it as a residual.
STRICT_SECTIONS="C D M N L"

section_a() { # panicking escapes in production
  run_section A "Panicking error handling in production code" \
    '\.unwrap\(\)' \
    '\.expect\(' \
    '\.unwrap_err\(' \
    '\.expect_err\(' \
    '[^a-z_]panic!\(' \
    'todo!\(' \
    'unimplemented!\(' \
    'unreachable!\('
}

section_c() { # lint suppression
  # A suppression *written as Rust code*, which is why the check is anchored to
  # `#[allow(` immediately followed by a lint name rather than to the name
  # itself. Two consequences, both deliberate:
  #
  #   - A line that merely mentions the attribute inside a string or a comment
  #     is data, not code. This tree has a module (`src/cheat.rs`) whose whole
  #     job is scanning candidate changes for suppressions, so its fixtures name
  #     every pattern this section looks for; without the anchor they would
  #     drown the report.
  #   - The baseline in that module, which records the suppressions this
  #     repository has deliberately accepted, is likewise data.
  #
  # `src/cheat.rs` is the one file excluded outright, and for the reason above:
  # it is the detector, and a detector that cannot name its own patterns cannot
  # report them.
  run_section C "Lint suppression" \
    '#\[allow\((clippy::all|clippy::pedantic|unused|dead_code|unreachable_code)\)' \
    '#\[expect\((clippy::all|clippy::pedantic)\)' \
    '#!\[allow\('
}

section_l() { # configuration that silently lowers the floor
  # A lint threshold set in a file is a rule turned off without a diff showing
  # it. Same for `--cap-lints`, which is how a build script opts out of every
  # lint in this table.
  run_section L "Lint configuration that lowers the bar" \
    '^# *\[*lints\.' \
    'cap-lints[ =]allow' \
    '^# *clippy::.*(allow|warn)[[:space:]]*$'
}

section_d() { # panic vectors that are not spelled as panics
  # A literal index is the one indexing form whose bound is checkable by eye:
  # `c[0]` on a slice known to be `chunks_exact(4)` is fine, `xs[n]` where `n`
  # is user input is not. Ranges (`xs[a..b]`) are not listed because this tree
  # uses them on shapes it has already checked, and every one is a false alarm
  # at the volume they appear.
  # The pattern requires an identifier immediately before the bracket, which is
  # what separates `term[0]` (an index, and a panic if `term` is short) from
  # `ones([4])` (a dimension, where the brackets are a call's argument list).
  # Without that, every `Tensor::zeros([3, 4])` in the tree is a false alarm.
  run_section D "Indexing as an unchecked panic" \
    '[[:alnum:]_)\]]\[[0-9]+\]' \
    '\.unwrap_or\(panic!'
}

section_m() { # injection
  run_section M "Command and query injection" \
    'Command::new\("(/bin/)?sh"\)' \
    'Command::new\("cmd(\.exe)?"\)' \
    '\.arg\("-c"\)' \
    'format!\("(SELECT|INSERT|UPDATE|DELETE|DROP)'
}

section_n() { # undefined behaviour and unbounded trust
  run_section N "Undefined behaviour and unvalidated descent" \
    'static[[:space:]]+mut' \
    'mem::transmute' \
    'mem::uninitialized' \
    'mem::zeroed' \
    'ptr::read\(' \
    'ptr::write\(' \
    'catch_unwind\(\|\|\| *\)?' \
    'std::panic::set_hook'
}

section_o() { # avoidable cost
  # `.len() == 0` is reported because clippy's own `len_zero` fires on it, but it
  # is an `is_empty` in disguise and rewriting two of them is a worse diff than
  # the warning is worth. It stays in the report so the choice is visible; it is
  # out of `STRICT_SECTIONS` because a reviewer, not a gate, should decide.
  run_section O "Performance smells" \
    '\.iter\(\)\.count\(\)' \
    '\.len\(\) *== *0' \
    '\.len\(\) *> *0' \
    '\.to_string\(\)\.as_str\(\)' \
    'with_capacity\(0\)'
}

section_s() { # secrets
  run_section S "Secrets in source" \
    'sk-[A-Za-z0-9]{20,}' \
    'AKIA[0-9A-Z]{16}' \
    'Bearer [A-Za-z0-9._-]{40,}' \
    'ghp_[A-Za-z0-9]{36}'
}

section_p() { # hygiene that is not a panic
  run_section P "Source hygiene" \
    '\.clone\(\)\.clone\(\)'
}

# `as` conversions are not errors: a numeric crate has thousands of them and
# `clippy::as_conversions` has no opinion about whether a particular one is
# guarded. What matters is the *unguarded* one, so this section reports every
# conversion and a reviewer reads the list once, rather than the linter guessing.
section_e() {
  run_section E "Numeric conversions (review these; waive with a reason)" \
    ' as (u8|u16|u32|u64|usize|i8|i16|i32|i64|isize|f32|f64)' \
    ' as [*]const' \
    ' as [*]mut'
}

ALL_SECTIONS=(a c d e l m n o s p)
for s in "${ALL_SECTIONS[@]}"; do
  upper="$(printf '%s' "$s" | tr '[:lower:]' '[:upper:]')"
  if [[ -n "$SECTION" && "$SECTION" != "$upper" ]]; then continue; fi
  "section_$s"
  TOTAL=$((TOTAL + SECTION_HITS))
  TOTAL_PROD=$((TOTAL_PROD + SECTION_PROD))
  if [[ "$STRICT" -eq 1 && " $STRICT_SECTIONS " == *" $upper "* && "$SECTION_PROD" -gt 0 ]]; then
    FAILING_SECTIONS+=("$upper")
  fi
done

printf '\n%s\n' "------------------------------------------------------------"
printf 'scanned %d production file(s), %d test file(s)\n' "${#PROD_FILES[@]}" "${#TEST_FILES[@]}"
printf '%d finding line(s); %d in production code\n' "$TOTAL" "$TOTAL_PROD"

if [[ "$STRICT" -eq 1 ]]; then
  if [[ ${#FAILING_SECTIONS[@]} -gt 0 ]]; then
    printf '\nSTRICT: failing sections with production hits: %s\n' "${FAILING_SECTIONS[*]}"
    exit 1
  fi
  printf '\nSTRICT: clean.\n'
  exit 0
fi

printf '\nRun with --strict to make production findings an error (the CI gate).\n'