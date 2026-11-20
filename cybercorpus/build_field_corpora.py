#!/usr/bin/env python3
"""Group cleaned documents by taxonomy field and write one training-ready corpus
per field, in the exact framing `dblocks lm train --corpus` consumes.

This is the per-field counterpart of `quality_filter.py --corpus-out`: instead of
one corpus for the whole dataset, it partitions documents by their **field** and
emits `dataset/<field>.corpus.bin` for each -- the per-field corpora the §13
per-field specialists train on. The u16 framing (headerless, BOS=256 / EOS=257
per document, each UTF-8 byte as a little-endian u16) is NOT re-implemented here:
`frame_document` is imported from `quality_filter.py` so the two stages can never
drift apart.

Field of a document (first that is present and non-empty):
  1. an explicit ``"field"`` key on the record,
  2. its ``"source"`` (documents ingested by `ingest_hf.py` live under
     `raw/<field>/...`, so `build_dataset.py` records ``source == <field>``),
  3. ``"unknown"``.
A ``--map`` JSON of ``{source_or_field: field}`` can remap/merge sources into a
field (e.g. fold several wordlist sources into one "cyber" corpus).

Input is a cleaned JSONL (default `dataset/clean.jsonl`, else `dataset/all.jsonl`)
or, with ``--paths``, a directory whose top-level subdirectories are the fields.

For each field it writes, beside the corpus:
  dataset/<field>.corpus.bin       headerless LE-u16 token stream
  dataset/<field>.manifest.json    {field, documents, tokens, bytes, ...}

Standard library only; run with ``python3 -I``. Idempotent: each field's corpus
and manifest are overwritten in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

# `python3 -I` does NOT put the script's own directory on sys.path (-I implies
# -P), so make the sibling import explicit. This adds only our own code dir --
# never a directory that holds downloaded/untrusted data.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from quality_filter import BOS_ID, EOS_ID, frame_document  # noqa: E402


# --------------------------------------------------------------------------- #
# Field resolution
# --------------------------------------------------------------------------- #


def field_of(rec: dict, source_map: dict[str, str]) -> str:
    """The field a record belongs to (see module docstring for the order)."""
    explicit = rec.get("field")
    if isinstance(explicit, str) and explicit.strip():
        key = explicit.strip()
        return source_map.get(key, key)
    source = rec.get("source")
    if isinstance(source, str) and source.strip():
        key = source.strip()
        return source_map.get(key, key)
    return "unknown"


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def field_slug(field: str) -> str:
    """Filesystem-safe stem for a field name (e.g. 'pentest/recon' ->
    'pentest__recon'), so corpora stay flat in dataset/ with no path escapes."""
    slug = field.replace("/", "__")
    slug = _SLUG_RE.sub("-", slug).strip("-._")
    return slug or "unknown"


# --------------------------------------------------------------------------- #
# Document sources
# --------------------------------------------------------------------------- #


def read_jsonl_records(path: Path) -> Iterator[dict]:
    """Yield raw record dicts from a JSONL file (keeps any `field` key)."""
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
            if isinstance(rec, dict):
                yield rec


def read_dir_records(root: Path) -> Iterator[dict]:
    """Yield a record per readable file under `root`; the field is the first
    path component below `root` (matching the `raw/<field>/...` layout)."""
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        field = rel.parts[0] if len(rel.parts) > 1 else root.name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"warning: {path}: {exc}; skipped", file=sys.stderr)
            continue
        yield {"id": rel.as_posix(), "source": field, "field": field,
               "path": rel.as_posix(), "format": path.suffix.lstrip(".") or "text",
               "text": text}


# --------------------------------------------------------------------------- #
# Per-field writer
# --------------------------------------------------------------------------- #


class FieldWriter:
    """Streams framed documents into one field's corpus, tracking counts."""

    def __init__(self, field: str, out_dir: Path):
        self.field = field
        self.slug = field_slug(field)
        self.corpus_path = out_dir / f"{self.slug}.corpus.bin"
        self.manifest_path = out_dir / f"{self.slug}.manifest.json"
        self.fh = self.corpus_path.open("wb")
        self.documents = 0
        self.tokens = 0  # number of u16 tokens (BOS + bytes + EOS)
        self.byte_len = 0  # number of bytes in the corpus file
        self.source_counts: dict[str, int] = {}

    def add(self, text: str, source: str) -> None:
        frame = frame_document(text)
        self.fh.write(frame)
        self.documents += 1
        self.tokens += len(frame) // 2  # 2 bytes per u16 token
        self.byte_len += len(frame)
        self.source_counts[source] = self.source_counts.get(source, 0) + 1

    def close(self) -> dict:
        self.fh.close()
        manifest = {
            "field": self.field,
            "corpus": self.corpus_path.name,
            "documents": self.documents,
            "tokens": self.tokens,
            "bytes": self.byte_len,
            "token_width_bytes": 2,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "sources": dict(sorted(self.source_counts.items())),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    srcg = ap.add_mutually_exclusive_group()
    srcg.add_argument("--input", type=Path,
                      help="input JSONL (default: dataset/clean.jsonl, "
                           "else dataset/all.jsonl)")
    srcg.add_argument("--paths", type=Path,
                      help="a directory whose top-level subdirs are fields")
    ap.add_argument("--out-dir", type=Path,
                    help="where to write <field>.corpus.bin (default: dataset/)")
    ap.add_argument("--map", type=Path,
                    help="JSON {source_or_field: field} to remap/merge sources")
    ap.add_argument("--min-chars", type=int, default=1,
                    help="skip documents shorter than this many chars (default 1)")
    ap.add_argument("--only-field", action="append", default=[],
                    help="restrict output to this field (repeatable)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = args.out_dir or (here / "dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    source_map: dict[str, str] = {}
    if args.map:
        loaded = json.loads(args.map.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            print(f"error: {args.map}: expected a JSON object", file=sys.stderr)
            return 2
        source_map = {str(k): str(v) for k, v in loaded.items()}

    # Resolve the record stream.
    if args.paths is not None:
        if not args.paths.is_dir():
            print(f"error: {args.paths} is not a directory", file=sys.stderr)
            return 1
        records = read_dir_records(args.paths)
        origin = args.paths
    else:
        inp = args.input
        if inp is None:
            clean = here / "dataset" / "clean.jsonl"
            allj = here / "dataset" / "all.jsonl"
            inp = clean if clean.is_file() else allj
        if not inp.is_file():
            print(f"error: {inp} not found; run quality_filter.py/build_dataset.py "
                  f"first, or pass --input/--paths", file=sys.stderr)
            return 1
        records = read_jsonl_records(inp)
        origin = inp

    only = set(args.only_field)
    writers: dict[str, FieldWriter] = {}
    skipped_short = 0
    skipped_field = 0

    for rec in records:
        field = field_of(rec, source_map)
        if only and field not in only:
            skipped_field += 1
            continue
        text = rec.get("text")
        if not isinstance(text, str) or len(text) < args.min_chars:
            skipped_short += 1
            continue
        writer = writers.get(field)
        if writer is None:
            writer = writers[field] = FieldWriter(field, out_dir)
        writer.add(text, str(rec.get("source", field)))

    if not writers:
        print(f"no documents grouped from {origin}; nothing written",
              file=sys.stderr)
        return 1

    print(f"source: {origin}")
    manifests = []
    total_docs = total_tokens = 0
    for field in sorted(writers):
        m = writers[field].close()
        manifests.append(m)
        total_docs += m["documents"]
        total_tokens += m["tokens"]
        print(f"  {field:20s} {m['documents']:7d} docs  "
              f"{m['tokens']:10d} tokens  -> {writers[field].corpus_path.name}")

    # A small index tying the per-field manifests together.
    index_path = out_dir / "field-corpora.index.json"
    index_path.write_text(json.dumps({
        "source": str(origin),
        "fields": {m["field"]: m for m in manifests},
        "totals": {"documents": total_docs, "tokens": total_tokens,
                   "fields": len(manifests)},
    }, indent=2) + "\n", encoding="utf-8")

    extra = []
    if skipped_short:
        extra.append(f"{skipped_short} too-short/no-text")
    if skipped_field:
        extra.append(f"{skipped_field} outside --only-field")
    tail = f"  (skipped {', '.join(extra)})" if extra else ""
    print(f"\nTOTAL {total_docs} docs, {total_tokens} tokens across "
          f"{len(manifests)} fields{tail}")
    print(f"index -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
