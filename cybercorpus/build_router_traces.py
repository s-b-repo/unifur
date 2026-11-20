#!/usr/bin/env python3
"""Generate (task text -> category combo) traces for the meta-router (§13 L3).

The 3-layer router's top layer (meta-router) decides WHICH category combination a
task needs. This emits its training traces in the schema cybersec-ai-30 asked for:

    {"id": "...", "text": "<task text>",
     "categories": [{"id": "cyber-offensive", "weight": 0.6}, ...]}

The primary category is the document's field; secondary categories are added when
the text shows another domain (so a cyber task that is really about SQL gets both
cyber-offensive AND data) — that is exactly the combo signal the meta-router learns.
Weights are normalised to sum to 1.0. Deterministic; stdlib only; `python3 -I`.

Category ids are the §13 taxonomy ids. Field -> primary category is `FIELD_PRIMARY`;
edit it as the taxonomy grows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Field (dataset source) -> primary §13 category id.
FIELD_PRIMARY = {
    "coding": "coding",
    "cyber": "cyber-offensive",
    "math": "math",
    "science": "science",
    "tech": "data",          # the tech sample is text-to-SQL -> data
    "systems": "systems",
    "networking": "networking",
    "cloud": "cloud",
}

# Secondary category signals: category id -> regex over the task text.
SECONDARY = {
    "coding": re.compile(r"(?i)\b(def |class |function|import |return|const |let |fn |public |#include)\b"),
    "data": re.compile(r"(?i)\b(sql|select |database|query|table|schema|join|dataframe)\b"),
    "networking": re.compile(r"(?i)\b(tcp|udp|http|dns|socket|port \d|packet|firewall|proxy)\b"),
    "cyber-netsec-crypto": re.compile(r"(?i)\b(encrypt|decrypt|cipher|aes|rsa|sha-?\d|hash|tls|certificate)\b"),
    "cyber-offensive": re.compile(r"(?i)\b(exploit|payload|injection|xss|csrf|rce|privilege|reverse shell|malware|cve-)\b"),
    "cyber-defensive": re.compile(r"(?i)\b(detect|defen|siem|forensic|incident|hardening|firewall rule|ids|ips)\b"),
    "cyber-appsec": re.compile(r"(?i)\b(fuzz|sast|dast|code review|owasp|sanitiz)\b"),
    "cloud": re.compile(r"(?i)\b(aws|azure|gcp|kubernetes|docker|terraform|s3 bucket|lambda)\b"),
    "ai-ml": re.compile(r"(?i)\b(neural|model training|dataset|gradient|transformer|embedding|classifier)\b"),
    "math": re.compile(r"(?i)\b(theorem|proof|integral|derivative|equation|polynomial|matrix|probability)\b"),
    "coding-review": re.compile(r"(?i)\b(unwrap|panic|lint|clippy|refactor)\b"),
}


def task_text(text: str, max_chars: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def categories_for(field: str, text: str) -> list[dict]:
    primary = FIELD_PRIMARY.get(field, field)
    weights: dict[str, float] = {primary: 1.0}
    for cat, pat in SECONDARY.items():
        if cat != primary and pat.search(text):
            weights[cat] = weights.get(cat, 0.0) + 0.5
    total = sum(weights.values())
    return [{"id": c, "weight": round(w / total, 3)}
            for c, w in sorted(weights.items(), key=lambda kv: -kv[1])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="clean.jsonl (records with source + text)")
    ap.add_argument("--out", type=Path, required=True, help="output traces JSONL")
    ap.add_argument("--val-out", type=Path, help="also write a document-level val split here")
    ap.add_argument("--val-frac", type=int, default=15, help="percent held out by document (default 15)")
    ap.add_argument("--max-chars", type=int, default=600)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 1
    tr = args.out.open("w", encoding="utf-8")
    va = args.val_out.open("w", encoding="utf-8") if args.val_out else None
    n = nv = 0
    counts: dict[str, int] = {}
    for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        field = str(rec.get("source", "unknown"))
        text = task_text(str(rec.get("text", "")), args.max_chars)
        if len(text) < 40:
            continue
        cats = categories_for(field, text)
        for c in cats:
            counts[c["id"]] = counts.get(c["id"], 0) + 1
        trace = {"id": str(rec.get("id", f"t{n}")), "text": text, "categories": cats}
        h = int(hashlib.sha1(trace["id"].encode()).hexdigest(), 16) % 100
        if va is not None and h < args.val_frac:
            va.write(json.dumps(trace, ensure_ascii=False) + "\n")
            nv += 1
        else:
            tr.write(json.dumps(trace, ensure_ascii=False) + "\n")
            n += 1
    tr.close()
    if va is not None:
        va.close()
    print(f"router traces: {n} train" + (f", {nv} val" if va is not None else ""))
    print(f"category coverage: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
    print(f"-> {args.out}" + (f"  (+ {args.val_out})" if va is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
