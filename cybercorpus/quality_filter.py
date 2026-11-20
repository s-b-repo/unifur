#!/usr/bin/env python3
"""Corpus-level quality pass: deduplicate, scrub secrets, drop low-quality docs.

This is the stage between `build_dataset.py` (which normalizes raw sources into
one document per record) and `dblocks lm tokenize` (which turns kept text into a
`u16` token corpus). It operates on whole *documents*, which is deliberately a
different job from the two filters already in the Rust crate:

  * `src/codequality/`  — scores a *training window* of code (anti-pattern
                          density + structural heuristics + external linters)
                          and keeps/downweights/drops it at training time.
  * `src/codegen_eval.rs` — decontaminates an *eval set* against the corpus by
                          13-gram overlap.

Neither deduplicates the corpus, removes credentials, or rejects whole
boilerplate/gibberish documents before they are ever tokenized. That is what
this stage does, and why it lives here rather than in either of those.

Stages, each independently toggleable and each reported honestly in the output
manifest (nothing is dropped silently):

  1. exact dedup     — identical documents (after whitespace normalization),
                       keyed by SHA-256; the first occurrence is kept.
  2. near-dup        — MinHash + LSH over 5-word shingles; a document whose
                       estimated Jaccard similarity to an already-kept document
                       is >= --near-threshold is dropped as a near-duplicate.
  3. secret scrub    — high-confidence structured credentials (private-key
                       blocks, AWS/GitHub/Google/Slack tokens, JWTs) are
                       replaced with a typed placeholder so the model never
                       memorizes a live secret. Educational payload text is
                       kept; only credentials that match a specific shape are
                       touched. `--scrub-generic` additionally redacts
                       `key = "<long-token>"` assignments (off by default, as it
                       bites example payloads).
  4. quality drop    — per-document heuristics: too short / too long, too much
                       line repetition, too high a non-text (control-char)
                       ratio. Each threshold is a flag; the defaults are
                       conservative.

Input is `dataset/all.jsonl` (records: id, source, path, format, text) by
default, or any JSONL passed with --input, or a directory of raw files passed
with --paths for quick ad-hoc testing. Output is a cleaned JSONL plus a
`quality-report.json` next to it.

Standard library only: no numpy, no huggingface, no network. Run it with
`python3 -I` on an untrusted corpus directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# --------------------------------------------------------------------------- #
# Document source
# --------------------------------------------------------------------------- #


@dataclass
class Doc:
    id: str
    source: str
    path: str
    format: str
    text: str


def read_jsonl(path: Path) -> Iterator[Doc]:
    """Yield documents from a build_dataset.py-style JSONL file."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: {path}:{lineno}: bad JSON ({exc}); skipped",
                      file=sys.stderr)
                continue
            yield Doc(
                id=str(rec.get("id", f"{path.stem}:{lineno}")),
                source=str(rec.get("source", path.stem)),
                path=str(rec.get("path", "")),
                format=str(rec.get("format", "text")),
                text=str(rec.get("text", "")),
            )


def read_paths(paths: Iterable[Path]) -> Iterator[Doc]:
    """Yield one document per readable file under the given paths (for testing)."""
    for root in paths:
        files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*") if p.is_file()
        )
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"warning: {p}: {exc}; skipped", file=sys.stderr)
                continue
            yield Doc(id=str(p), source=root.name, path=str(p),
                      format=p.suffix.lstrip(".") or "text", text=text)


# --------------------------------------------------------------------------- #
# 1 + 2. Deduplication
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def exact_key(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


# Deterministic MinHash. hash() is not stable across runs (PYTHONHASHSEED), so
# shingles are hashed with blake2b and permuted by 64 fixed (a, b) pairs over a
# Mersenne prime. Signature[i] = min over shingles of (a_i*h + b_i) mod P.
_MINHASH_P = (1 << 61) - 1
_MINHASH_K = 64


def _permutations(k: int) -> list[tuple[int, int]]:
    out = []
    for i in range(k):
        seed = hashlib.blake2b(i.to_bytes(4, "big"), digest_size=16).digest()
        a = int.from_bytes(seed[:8], "big") | 1          # odd, non-zero
        b = int.from_bytes(seed[8:], "big") % _MINHASH_P
        out.append((a % _MINHASH_P, b))
    return out


_PERMS = _permutations(_MINHASH_K)


def shingles(text: str, width: int = 5) -> set[int]:
    words = normalize(text).split()
    if len(words) < width:
        # Short docs: one shingle of whatever words exist, so they still hash.
        joined = " ".join(words)
        return {int.from_bytes(hashlib.blake2b(joined.encode(), digest_size=8).digest(), "big")} if joined else set()
    out: set[int] = set()
    for i in range(len(words) - width + 1):
        gram = " ".join(words[i:i + width]).encode("utf-8")
        out.add(int.from_bytes(hashlib.blake2b(gram, digest_size=8).digest(), "big"))
    return out


def minhash(sig_shingles: set[int]) -> tuple[int, ...]:
    if not sig_shingles:
        return tuple([0] * _MINHASH_K)
    sig = []
    for a, b in _PERMS:
        sig.append(min(((a * h + b) % _MINHASH_P) for h in sig_shingles))
    return tuple(sig)


def estimated_jaccard(x: tuple[int, ...], y: tuple[int, ...]) -> float:
    equal = sum(1 for xi, yi in zip(x, y) if xi == yi)
    return equal / len(x) if x else 0.0


@dataclass
class NearDupIndex:
    """LSH over MinHash signatures. bands * rows == _MINHASH_K."""
    threshold: float
    bands: int = 16
    rows: int = _MINHASH_K // 16
    buckets: dict[tuple[int, bytes], list[int]] = field(default_factory=dict)
    sigs: list[tuple[int, ...]] = field(default_factory=list)

    def query_and_add(self, sig: tuple[int, ...]) -> int | None:
        """Return the index of a kept near-duplicate, or None after inserting."""
        candidate_ids: set[int] = set()
        band_keys = []
        for band in range(self.bands):
            chunk = sig[band * self.rows:(band + 1) * self.rows]
            key = (band, hashlib.blake2b(
                b"".join(c.to_bytes(8, "big") for c in chunk), digest_size=8).digest())
            band_keys.append(key)
            candidate_ids.update(self.buckets.get(key, ()))
        for cid in candidate_ids:
            if estimated_jaccard(sig, self.sigs[cid]) >= self.threshold:
                return cid
        idx = len(self.sigs)
        self.sigs.append(sig)
        for key in band_keys:
            self.buckets.setdefault(key, []).append(idx)
        return None


# --------------------------------------------------------------------------- #
# 3. Secret scrubbing
# --------------------------------------------------------------------------- #

# High-confidence, structured credentials only. Each pattern has a type name so
# the redaction is typed (and so the report can say what was found where).
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
        re.DOTALL)),
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[0-9A-Za-z]{36,}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("SLACK_WEBHOOK", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+")),
]

# Opt-in: `secret = "longtoken"` style assignments. Off by default because it
# fires on example payloads in security docs, which are signal, not secrets.
GENERIC_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|bearer|authorization)\b"
    r"(\s*[:=]\s*)(['\"]?)([A-Za-z0-9/+=_\-]{16,})\3")


def scrub_secrets(text: str, scrub_generic: bool) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pat in SECRET_PATTERNS:
        text, n = pat.subn(f"<REDACTED:{name}>", text)
        if n:
            counts[name] = counts.get(name, 0) + n
    if scrub_generic:
        def _repl(m: re.Match[str]) -> str:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}<REDACTED:GENERIC_SECRET>{m.group(3)}"
        text, n = GENERIC_SECRET.subn(_repl, text)
        if n:
            counts["GENERIC_SECRET"] = counts.get("GENERIC_SECRET", 0) + n
    return text, counts


# --------------------------------------------------------------------------- #
# 4. Document-quality heuristics
# --------------------------------------------------------------------------- #


def line_repetition_ratio(text: str) -> float:
    """Fraction of non-blank lines that are exact duplicates of an earlier line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for ln in lines:
        if ln in seen:
            repeats += 1
        else:
            seen.add(ln)
    return repeats / len(lines)


def nontext_ratio(text: str) -> float:
    """Fraction of characters that are control/non-printable (excl. tab/newline)."""
    if not text:
        return 0.0
    bad = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r")
    return bad / len(text)


@dataclass
class QualityLimits:
    min_chars: int = 80
    max_chars: int = 2_000_000
    max_line_repeat: float = 0.80
    max_nontext: float = 0.02

    def reject_reason(self, text: str) -> str | None:
        n = len(text)
        if n < self.min_chars:
            return "too_short"
        if n > self.max_chars:
            return "too_long"
        if line_repetition_ratio(text) > self.max_line_repeat:
            return "repetitive"
        if nontext_ratio(text) > self.max_nontext:
            return "nontext"
        return None


# --------------------------------------------------------------------------- #
# Smuggle detection (TRISKELE chain #5). A document with reject-grade hidden
# content is QUARANTINED (dropped + reported), never silently stripped — a
# stripped doc would hide that the source tried to smuggle. Semantics are
# aligned with the Rust corpus.rs stage:
#   REJECT  tag chars (ASCII smuggling), bidi override/isolate controls
#           (Trojan Source), zero-width chars inside text/identifiers,
#           variation-selector payloads off emoji, Latin↔Cyrillic homoglyph
#           words, and base64/hex that decodes to prompt-injection.
#   ALLOW   a single leading BOM, ZWJ inside an emoji sequence, ZWNJ in
#           Arabic/Indic script, and bidi MARKS (LRM/RLM/ALM) in RTL prose.
# --------------------------------------------------------------------------- #

_TAG_RANGE = range(0xE0000, 0xE0080)
_VARSEL = set(range(0xFE00, 0xFE10)) | set(range(0xE0100, 0xE01F0))
_BIDI_CONTROL = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}
_ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x2060, 0x180E, 0xFEFF}
_INJECTION = re.compile(
    r"(?i)ignore (all |the )?(previous|prior|above)|disregard (all|the|previous)"
    r"|system prompt|you are now|new instructions?|do anything now|jailbreak")
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_HEXRUN = re.compile(r"(?:[0-9a-fA-F]{2}){16,}")


def _script(cp: int) -> str:
    if (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF
            or 0x1F1E6 <= cp <= 0x1F1FF or cp in (0x2764, 0x2122, 0x203C, 0x2705)):
        return "emoji"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFEFF:
        return "arabic"
    if 0x0900 <= cp <= 0x0DFF:
        return "indic"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A or 0x00C0 <= cp <= 0x024F:
        return "latin"
    return "other"


def _try_b64(s: str) -> str:
    import base64
    import binascii
    try:
        return base64.b64decode(s + "=" * (-len(s) % 4), validate=True).decode("utf-8", "replace")
    except (binascii.Error, ValueError):
        return ""


def _try_hex(s: str) -> str:
    try:
        return bytes.fromhex(s).decode("utf-8", "replace")
    except ValueError:
        return ""


def scan_smuggle(text: str) -> list[tuple[str, str]]:
    """Return (category, sample) for each reject-grade hidden-content finding.
    An empty list means the document is clean of smuggling."""
    findings: list[tuple[str, str]] = []
    body = text[1:] if text[:1] == "﻿" else text   # allow one leading BOM
    cps = [ord(c) for c in body]
    n = len(cps)

    def word_neighbour(i: int) -> bool:
        for j in (i - 1, i + 1):
            if 0 <= j < n and _script(cps[j]) in ("latin", "cyrillic") or (0 <= j < n and 0x30 <= cps[j] <= 0x39):
                return True
        return False

    def emoji_before(i: int) -> bool:
        return i > 0 and _script(cps[i - 1]) == "emoji"

    def near_script(i: int, want: tuple[str, ...]) -> bool:
        return any(0 <= j < n and _script(cps[j]) in want for j in (i - 1, i + 1))

    for i, cp in enumerate(cps):
        if cp in _TAG_RANGE:
            findings.append(("tag_char", f"U+{cp:05X}"))
        elif cp in _BIDI_CONTROL:
            findings.append(("bidi_override", f"U+{cp:04X}"))
        elif cp in _VARSEL and not emoji_before(i):
            findings.append(("variation_selector", f"U+{cp:04X}"))
        elif cp in _ZERO_WIDTH:
            if cp == 0x200D and near_script(i, ("emoji",)):
                continue                                  # emoji ZWJ sequence
            if cp == 0x200C and near_script(i, ("arabic", "indic")):
                continue                                  # ZWNJ in Arabic/Indic
            if word_neighbour(i):
                findings.append(("zero_width_in_text", f"U+{cp:04X}"))

    for m in re.finditer(r"\w{2,}", body, re.UNICODE):
        scripts = {_script(ord(c)) for c in m.group()}
        if "latin" in scripts and "cyrillic" in scripts:   # classic homoglyph
            findings.append(("homoglyph", m.group()[:24]))
            break

    for pat, dec in ((_B64, _try_b64), (_HEXRUN, _try_hex)):
        for m in pat.finditer(body):
            decoded = dec(m.group())
            if decoded and _INJECTION.search(decoded):
                findings.append(("encoded_injection", decoded[:48].replace("\n", " ")))
                break

    return findings


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    seen: int = 0
    kept: int = 0
    dropped_exact: int = 0
    dropped_near: int = 0
    dropped_quality: dict[str, int] = field(default_factory=dict)
    secrets_redacted: dict[str, int] = field(default_factory=dict)
    docs_with_secrets: int = 0
    quarantined_smuggle: int = 0
    smuggle_by_type: dict[str, int] = field(default_factory=dict)
    per_source_kept: dict[str, int] = field(default_factory=dict)
    per_source_seen: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "dropped": {
                "exact_duplicate": self.dropped_exact,
                "near_duplicate": self.dropped_near,
                "quality": self.dropped_quality,
            },
            "secrets": {
                "docs_with_secrets": self.docs_with_secrets,
                "redactions_by_type": self.secrets_redacted,
            },
            "smuggle": {
                "quarantined_docs": self.quarantined_smuggle,
                "by_type": self.smuggle_by_type,
            },
            "per_source": {
                src: {"seen": self.per_source_seen.get(src, 0),
                      "kept": self.per_source_kept.get(src, 0)}
                for src in sorted(self.per_source_seen)
            },
        }


# Must match src/tokenizer.rs: 256 byte values, then specials.
BOS_ID = 256
EOS_ID = 257


def frame_document(text: str) -> bytes:
    """Encode one document as the crate's `encode_document` does: BOS, each
    UTF-8 byte as a little-endian u16, EOS. The corpus is headerless (see
    src/corpus.rs), so documents concatenate directly."""
    out = bytearray()
    out += struct.pack("<H", BOS_ID)
    for b in text.encode("utf-8"):
        out += struct.pack("<H", b)
    out += struct.pack("<H", EOS_ID)
    return bytes(out)


def run(docs: Iterator[Doc], out: Path, limits: QualityLimits,
        near_threshold: float, do_exact: bool, do_near: bool,
        do_secrets: bool, scrub_generic: bool,
        corpus_out: Path | None = None, do_smuggle: bool = True) -> Report:
    report = Report()
    seen_exact: set[str] = set()
    near = NearDupIndex(threshold=near_threshold) if do_near else None
    corpus_fh = corpus_out.open("wb") if corpus_out is not None else None

    with out.open("w", encoding="utf-8") as fh:
        for doc in docs:
            report.seen += 1
            report.per_source_seen[doc.source] = report.per_source_seen.get(doc.source, 0) + 1

            if do_smuggle:
                smug = scan_smuggle(doc.text)
                if smug:
                    report.quarantined_smuggle += 1
                    for cat, _ in smug:
                        report.smuggle_by_type[cat] = report.smuggle_by_type.get(cat, 0) + 1
                    continue

            reason = limits.reject_reason(doc.text)
            if reason is not None:
                report.dropped_quality[reason] = report.dropped_quality.get(reason, 0) + 1
                continue

            if do_exact:
                key = exact_key(doc.text)
                if key in seen_exact:
                    report.dropped_exact += 1
                    continue
                seen_exact.add(key)

            if near is not None:
                sig = minhash(shingles(doc.text))
                if near.query_and_add(sig) is not None:
                    report.dropped_near += 1
                    continue

            text = doc.text
            if do_secrets:
                text, counts = scrub_secrets(text, scrub_generic)
                if counts:
                    report.docs_with_secrets += 1
                    for k, v in counts.items():
                        report.secrets_redacted[k] = report.secrets_redacted.get(k, 0) + v

            fh.write(json.dumps({
                "id": doc.id, "source": doc.source, "path": doc.path,
                "format": doc.format, "text": text,
            }, ensure_ascii=False) + "\n")
            if corpus_fh is not None:
                corpus_fh.write(frame_document(text))
            report.kept += 1
            report.per_source_kept[doc.source] = report.per_source_kept.get(doc.source, 0) + 1

    if corpus_fh is not None:
        corpus_fh.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input", type=Path, help="input JSONL (default: dataset/all.jsonl)")
    src.add_argument("--paths", type=Path, nargs="+",
                     help="raw files/dirs to treat as documents (for ad-hoc testing)")
    ap.add_argument("--out", type=Path, help="output JSONL (default: alongside input as clean.jsonl)")
    ap.add_argument("--corpus-out", type=Path,
                    help="also write a training-ready headerless LE-u16 corpus "
                         "(BOS/EOS per doc) that `dblocks lm train --corpus` reads")
    ap.add_argument("--near-threshold", type=float, default=0.80,
                    help="estimated-Jaccard cutoff for near-dup drop (default 0.80)")
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=2_000_000)
    ap.add_argument("--max-line-repeat", type=float, default=0.80)
    ap.add_argument("--max-nontext", type=float, default=0.02)
    ap.add_argument("--no-exact", action="store_true", help="skip exact dedup")
    ap.add_argument("--no-near", action="store_true", help="skip near-dup detection")
    ap.add_argument("--no-secrets", action="store_true", help="skip secret scrubbing")
    ap.add_argument("--no-smuggle", action="store_true",
                    help="skip hidden-content (smuggle) quarantine")
    ap.add_argument("--scrub-generic", action="store_true",
                    help="also redact key=<token> assignments (bites example payloads)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    if args.paths:
        docs = read_paths(args.paths)
        default_out = here / "dataset" / "clean.jsonl"
    else:
        inp = args.input or (here / "dataset" / "all.jsonl")
        if not inp.is_file():
            print(f"error: {inp} not found; run build_dataset.py first "
                  f"or pass --input/--paths", file=sys.stderr)
            return 1
        docs = read_jsonl(inp)
        default_out = inp.with_name("clean.jsonl")

    out = args.out or default_out
    out.parent.mkdir(parents=True, exist_ok=True)

    limits = QualityLimits(
        min_chars=args.min_chars, max_chars=args.max_chars,
        max_line_repeat=args.max_line_repeat, max_nontext=args.max_nontext,
    )
    report = run(
        docs, out, limits, near_threshold=args.near_threshold,
        do_exact=not args.no_exact, do_near=not args.no_near,
        do_secrets=not args.no_secrets, scrub_generic=args.scrub_generic,
        corpus_out=args.corpus_out, do_smuggle=not args.no_smuggle,
    )

    report_path = out.with_name("quality-report.json")
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")

    d = report.as_dict()
    print(f"seen {d['seen']}  kept {d['kept']}  "
          f"(exact -{d['dropped']['exact_duplicate']}, "
          f"near -{d['dropped']['near_duplicate']}, "
          f"quality -{sum(d['dropped']['quality'].values())})")
    if d["secrets"]["redactions_by_type"]:
        print(f"secrets: {d['secrets']['docs_with_secrets']} docs, "
              f"{d['secrets']['redactions_by_type']}")
    if d["smuggle"]["quarantined_docs"]:
        print(f"smuggle: quarantined {d['smuggle']['quarantined_docs']} docs, "
              f"{d['smuggle']['by_type']}")
    print(f"clean -> {out}")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
