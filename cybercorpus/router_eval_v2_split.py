#!/usr/bin/env python3
"""Split router-eval-v2.jsonl into a calibration half and a test half.

Thresholds (OOS "unsure" cutoff, combo weight floor, etc.) must be tuned on the
calibration half and reported only on the test half, so they are never fit to
the same prompts they are scored on. The split is DETERMINISTIC (hash of the
prompt text, not the RNG) and STRATIFIED by (style, primary category): within
each stratum the rows are ordered by hash and dealt alternately, so both halves
keep near-equal per-category and per-style coverage and re-running is stable.
The full file is left in place. Writes router-eval-v2.cal.jsonl / .test.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def h(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()


def main() -> int:
    base = Path(__file__).resolve().parent.parent.parent / "nina-data" / "field-corpora"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", type=Path, default=base / "router-eval-v2.jsonl")
    ap.add_argument("--cal", type=Path, default=base / "router-eval-v2.cal.jsonl")
    ap.add_argument("--test", type=Path, default=base / "router-eval-v2.test.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.full.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Short prompts stratify by primary category (keep per-category balance);
    # combos and OOS are each pooled into one stratum so the small counts split
    # ~evenly instead of skewing when a category has only one or two of them.
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        prim = r["categories"][0]["id"] if (r["style"] == "short" and r["categories"]) else "*"
        strata[(r["style"], prim)].append(r)

    cal, test = [], []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: h(r["text"]))
        for i, r in enumerate(group):
            (cal if i % 2 == 0 else test).append(r)

    for part, path in ((cal, args.cal), (test, args.test)):
        part.sort(key=lambda r: r["id"])
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n", encoding="utf-8")

    def profile(part):
        st = Counter(r["style"] for r in part)
        cc = Counter(c["id"] for r in part if r["style"] == "short" for c in r["categories"])
        return st, cc

    cs, ccc = profile(cal)
    ts, tcc = profile(test)
    cal_ids = {r["id"] for r in cal}
    test_ids = {r["id"] for r in test}
    print(f"full {len(rows)} -> cal {len(cal)} / test {len(test)}  (disjoint: {not (cal_ids & test_ids)}, "
          f"covers all: {len(cal) + len(test) == len(rows)})")
    print(f"cal : short {cs['short']} combo {cs['combo']} oos {cs['oos']}")
    print(f"test: short {ts['short']} combo {ts['combo']} oos {ts['oos']}")
    print(f"min short/cat  cal {min(ccc.values())} test {min(tcc.values())} "
          f"(both halves cover all 16 cats: {len(ccc) == 16 and len(tcc) == 16})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
