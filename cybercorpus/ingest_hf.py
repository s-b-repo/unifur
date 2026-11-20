#!/usr/bin/env python3
"""Stream small samples of per-field HuggingFace datasets into the cybercorpus
`raw/` tree so the existing `build_dataset.py` picks them up unchanged.

Each taxonomy field (TRISKELE-PLAN.md §13: coding, math, cyber, ...) maps to one
or more HuggingFace dataset ids. For every dataset we stream a bounded sample
(``--limit`` rows), pull out a text field, and write one plain-text document per
row to::

    cybercorpus/raw/<field>/<dataset-slug>/NNN.txt

`build_dataset.py` walks `raw/<source>/`, using the *top-level* directory as the
record ``source``; writing under `raw/<field>/...` therefore makes ``source ==
<field>``, which is exactly the grouping key `build_field_corpora.py` keys on.

Everything downloaded is UNTRUSTED. It is only ever written under the output
`raw/` tree (never executed, never imported); filenames are generated integers,
never taken from the data. Run with ``python3 -I``.

`datasets` / `huggingface_hub` are OPTIONAL and imported lazily. If the library
is missing or there is no network, the tool prints a clear message and falls
back to a local fixture directory (``--fixture``), so the `raw/<field>/` layout
is still produced for a pipeline smoke-test.

Examples::

    # Stream 200 rows per dataset for every field in the default map:
    python3 -I cybercorpus/ingest_hf.py --limit 200

    # Only the math field:
    python3 -I cybercorpus/ingest_hf.py --field math --limit 50

    # No lib / no network: ingest from a local fixture dir of <field>.jsonl files
    python3 -I cybercorpus/ingest_hf.py --fixture ./fixtures --out cybercorpus/raw
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

# --------------------------------------------------------------------------- #
# The field -> dataset map. EDIT ME. Each field maps to a list of datasets;
# each dataset is {id, split?, config?, text_fields?}:
#   id          HuggingFace dataset id (namespace/name).
#   split       which split to stream (default "train").
#   config      dataset config / subset name (optional).
#   text_fields ordered field names to concatenate into the document text.
#               If none are present in a row, a generic extractor is used.
#
# The ids below are real, well-known public datasets chosen to line up with the
# taxonomy (§13) and with the Rust field normalizer (src/fieldnorm.rs, which
# already builds math/science/tech/cyber from these same corpora). Confidence
# and any caveats are noted inline; swap freely -- nothing here is load-bearing.
# --------------------------------------------------------------------------- #
DEFAULT_FIELD_DATASETS: dict[str, list[dict]] = {
    # Coding -- OPEN code datasets (the-stack-smol is gated/auth-only). mbpp is
    # small Python problems+solutions; tiny-codes is multi-language.
    "coding": [
        {"id": "google-research-datasets/mbpp", "config": "full", "split": "train",
         "text_fields": ["text", "code"]},
        {"id": "nampdn-ai/tiny-codes", "split": "train",
         "text_fields": ["prompt", "response"]},
    ],
    # Math -- step-by-step reasoning corpora (both used by fieldnorm.rs "math").
    "math": [
        {"id": "meta-math/MetaMathQA", "split": "train",
         "text_fields": ["query", "response"]},
        {"id": "AI-MO/NuminaMath-CoT", "split": "train",
         "text_fields": ["problem", "solution"]},
    ],
    # Science QA -- matches fieldnorm.rs "science" (allenai/ai2_arc).
    "science": [
        {"id": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "train",
         "text_fields": ["question", "choices"]},
    ],
    # Tech / data -- synthetic text-to-SQL (fieldnorm.rs "tech" uses gretelai's).
    "tech": [
        {"id": "gretelai/synthetic_text_to_sql", "split": "train",
         "text_fields": ["sql_prompt", "sql_context", "sql", "sql_explanation"]},
    ],
    # Cyber -- instruction-tuning security corpus. The exact id/casing is the
    # least certain in this map (fieldnorm.rs ingests the same corpus from a
    # local parquet export as "trendyol-cyber"); confirm before a real pull.
    "cyber": [
        {"id": "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-dataset",
         "split": "train",
         "text_fields": ["instruction", "input", "output"]},
    ],
}

# Generic text-field candidates, tried (in order) when a row has none of the
# dataset's declared text_fields. Keeps the tool usable on an unknown schema.
_GENERIC_TEXT_FIELDS = [
    "text", "content", "code", "body", "document", "markdown", "raw",
    "response", "output", "answer", "solution", "completion", "chosen",
    "instruction", "input", "prompt", "question", "query", "problem", "title",
]

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(dataset_id: str) -> str:
    """A filesystem-safe directory name for a dataset id (no path escapes)."""
    slug = dataset_id.replace("/", "__")
    slug = _SLUG_RE.sub("-", slug).strip("-._") or "dataset"
    return slug


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def extract_text(row: dict, text_fields: Iterable[str] | None) -> str:
    """Assemble a document from a dataset row.

    Prefer the dataset's declared ``text_fields``; if none are present, fall
    back to a generic candidate list; failing that, take the longest string
    value. Non-string values (lists of choices, nested dicts) are JSON-encoded.
    """
    parts: list[str] = []
    for name in (text_fields or []):
        if name in row and row[name] not in (None, ""):
            s = _stringify(row[name]).strip()
            if s:
                parts.append(s)
    if parts:
        return "\n\n".join(parts)

    for name in _GENERIC_TEXT_FIELDS:
        if name in row and row[name] not in (None, ""):
            s = _stringify(row[name]).strip()
            if s:
                parts.append(s)
    if parts:
        return "\n\n".join(parts)

    best = ""
    for value in row.values():
        if isinstance(value, str) and len(value) > len(best):
            best = value
    return best.strip()


# --------------------------------------------------------------------------- #
# Row sources
# --------------------------------------------------------------------------- #


def iter_hf_rows(spec: dict, limit: int) -> Iterator[dict]:
    """Stream up to ``limit`` rows from a HuggingFace dataset. Lazy import; the
    caller handles ImportError / network errors and falls back to a fixture."""
    import datasets  # noqa: PLC0415  (intentionally lazy/optional)

    load_kwargs = {"split": spec.get("split", "train"), "streaming": True}
    if spec.get("config"):
        load_kwargs["name"] = spec["config"]
    # trust_remote_code defaults to False: we refuse to execute dataset scripts.
    ds = datasets.load_dataset(spec["id"], **load_kwargs)
    for i, row in enumerate(ds):
        if i >= limit:
            break
        if isinstance(row, dict):
            yield row


def iter_fixture_rows(path: Path, limit: int) -> Iterator[dict]:
    """Yield up to ``limit`` JSON-object rows from a local JSONL fixture file."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        count = 0
        for lineno, line in enumerate(fh, 1):
            if count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: {path}:{lineno}: bad JSON ({exc}); skipped",
                      file=sys.stderr)
                continue
            if isinstance(row, dict):
                count += 1
                yield row
            elif isinstance(row, str):
                count += 1
                yield {"text": row}


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

_NNN_RE = re.compile(r"^\d+\.txt$")


def write_documents(rows: Iterator[dict], field: str, slug: str,
                    out_root: Path, text_fields: Iterable[str] | None,
                    limit: int) -> int:
    """Write one `NNN.txt` per non-empty row under `out_root/<field>/<slug>/`.

    Idempotent: the target slug directory is cleared of prior `NNN.txt` files
    first, so a rerun (and a smaller --limit) never leaves stale documents.
    """
    dest = out_root / field.replace("/", "__") / slug
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.iterdir():
        if old.is_file() and _NNN_RE.match(old.name):
            old.unlink()

    width = max(3, len(str(max(limit - 1, 0))))
    written = 0
    for row in rows:
        text = extract_text(row, text_fields)
        if not text.strip():
            continue
        (dest / f"{written:0{width}d}.txt").write_text(text, encoding="utf-8")
        written += 1
    if written == 0:
        # Leave no empty dir behind so the layout reflects reality.
        try:
            dest.rmdir()
        except OSError:
            pass
    print(f"  {field:16s} {slug:40s} {written:5d} docs -> {dest}")
    return written


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def load_map(override: Path | None) -> dict[str, list[dict]]:
    if override is None:
        return DEFAULT_FIELD_DATASETS
    data = json.loads(override.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"error: {override}: expected a JSON object field->list")
    # Normalize: allow a bare string or list-of-strings as shorthand.
    norm: dict[str, list[dict]] = {}
    for field, specs in data.items():
        if isinstance(specs, str):
            specs = [{"id": specs}]
        out = []
        for spec in specs:
            out.append({"id": spec} if isinstance(spec, str) else dict(spec))
        norm[str(field)] = out
    return norm


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", help="only ingest this taxonomy field (default: all)")
    ap.add_argument("--limit", type=int, default=200,
                    help="max rows streamed per dataset (default 200)")
    ap.add_argument("--out", type=Path,
                    help="raw/ root to write into (default: cybercorpus/raw)")
    ap.add_argument("--map", type=Path,
                    help="JSON file overriding the field->datasets map")
    ap.add_argument("--fixture", type=Path,
                    help="local dir of <field>.jsonl files; used when datasets/"
                         "network is unavailable (or always, to skip the network)")
    ap.add_argument("--list", action="store_true",
                    help="print the effective field->datasets map and exit")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out_root = args.out or (here / "raw")
    field_map = load_map(args.map)

    if args.field:
        if args.field not in field_map:
            print(f"error: unknown field {args.field!r}; known: "
                  f"{', '.join(sorted(field_map))}", file=sys.stderr)
            return 2
        field_map = {args.field: field_map[args.field]}

    if args.list:
        print(json.dumps(field_map, indent=2))
        return 0

    # Decide the row source once: HF if importable, else fixture.
    have_datasets = False
    try:
        import datasets  # noqa: F401  (probe only)
        have_datasets = True
    except Exception as exc:  # ImportError or a broken install
        print(f"note: `datasets` not available ({exc.__class__.__name__}: {exc}); "
              f"falling back to local fixture", file=sys.stderr)

    if not have_datasets and args.fixture is None:
        print("error: `datasets`/`huggingface_hub` is not installed and no "
              "--fixture directory was given, so there is nothing to ingest.\n"
              "  Install with:  pip install datasets huggingface_hub\n"
              "  Or pass:       --fixture <dir-of-<field>.jsonl-files>",
              file=sys.stderr)
        return 1

    out_root.mkdir(parents=True, exist_ok=True)
    total = 0
    print(f"ingest -> {out_root}  (limit {args.limit}/dataset, "
          f"{'HF streaming' if have_datasets and args.fixture is None else 'fixture'})")

    for field in sorted(field_map):
        specs = field_map[field]
        fixture_file = None
        if args.fixture is not None:
            cand = args.fixture / f"{field.replace('/', '__')}.jsonl"
            fixture_file = cand if cand.is_file() else None

        # Fixture mode (explicit --fixture, or HF unavailable).
        if args.fixture is not None or not have_datasets:
            if fixture_file is None:
                print(f"  {field:16s} (no fixture file "
                      f"{args.fixture}/{field.replace('/', '__')}.jsonl; skipped)")
                continue
            text_fields = specs[0].get("text_fields") if specs else None
            total += write_documents(
                iter_fixture_rows(fixture_file, args.limit),
                field, "fixture", out_root, text_fields, args.limit)
            continue

        # HF streaming mode.
        for spec in specs:
            slug = slugify(spec["id"])
            # A spec may carry its own "limit" to balance sources within a field
            # (e.g. few rows of a 1MB/row corpus, many of a small-row one);
            # falls back to the global --limit.
            lim = int(spec.get("limit", args.limit))
            try:
                rows = iter_hf_rows(spec, lim)
                total += write_documents(rows, field, slug, out_root,
                                         spec.get("text_fields"), lim)
            except Exception as exc:  # network / missing dataset / schema
                print(f"  {field:16s} {slug:40s}  SKIPPED "
                      f"({exc.__class__.__name__}: {exc})", file=sys.stderr)

    print(f"\nTOTAL {total} documents written under {out_root}")
    print("next: python3 -I cybercorpus/build_dataset.py   # normalizes raw/ -> dataset/all.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
