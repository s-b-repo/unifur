#!/usr/bin/env python3
"""Document-level held-out VAL splits for the scaled corpora, leak-free.

Each kept document (from <field>.clean.jsonl) goes to train or val by the hash
of its doc id, so a given document is ALWAYS in the same split — reproducible
and disjoint. ~1.5% to val. Rewrites <field>-scale.corpus.bin to TRAIN ONLY
(val docs removed, so train and val never overlap) and writes
<field>-scale.val.corpus.bin, plus the all-fields train and val mixes. Uses the
pipeline's own frame_document (BOS + UTF-8 bytes as LE u16 + EOS), so the splits
are byte-compatible with `dblocks lm train --corpus`. Prints sha256 of every
output for the manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_filter import frame_document  # noqa: E402

BASE = Path("/home/kali/Music/cybersec ai")
RAW = BASE / "nina-data/raw-scale"
OUT = BASE / "nina-data/field-corpora"
FIELDS = ["cyber", "coding", "math", "science", "tech"]
VAL_PERMILLE = 15  # 1.5% held out


def is_val(doc_id: str) -> bool:
    h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 1000) < VAL_PERMILLE


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    all_train = OUT / "all-fields-scale.corpus.bin"
    all_val = OUT / "all-fields-scale.val.corpus.bin"
    atf = all_train.open("wb")
    avf = all_val.open("wb")
    rows = []
    for field in FIELDS:
        clean = RAW / f"{field}.clean.jsonl"
        if not clean.is_file():
            print(f"{field}: SKIP (no {clean.name})")
            continue
        tp = OUT / f"{field}-scale.corpus.bin"
        vp = OUT / f"{field}-scale.val.corpus.bin"
        tf = tp.open("wb")
        vf = vp.open("wb")
        nt = nv = tt = tv = 0
        with clean.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                frame = frame_document(r["text"])
                if is_val(r["id"]):
                    vf.write(frame)
                    avf.write(frame)
                    nv += 1
                    tv += len(frame) // 2
                else:
                    tf.write(frame)
                    atf.write(frame)
                    nt += 1
                    tt += len(frame) // 2
        tf.close()
        vf.close()
        rows.append((field, nt, tt, nv, tv, sha256_file(tp), sha256_file(vp)))
    atf.close()
    avf.close()

    print(f"{'field':9s} {'train_docs':>10} {'train_tok':>11} {'val_docs':>8} {'val_tok':>9} {'val%':>5}")
    for field, nt, tt, nv, tv, _ts, _vs in rows:
        pct = 100 * nv / (nt + nv) if (nt + nv) else 0
        print(f"{field:9s} {nt:>10} {tt:>11} {nv:>8} {tv:>9} {pct:>5.2f}")
    print("\nsha256:")
    for field, _nt, _tt, _nv, _tv, ts, vs in rows:
        print(f"  {field}-scale.corpus.bin      {ts}")
        print(f"  {field}-scale.val.corpus.bin  {vs}")
    print(f"  all-fields-scale.corpus.bin   {sha256_file(all_train)}")
    print(f"  all-fields-scale.val.corpus.bin {sha256_file(all_val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
