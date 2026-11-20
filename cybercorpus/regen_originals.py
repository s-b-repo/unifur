#!/usr/bin/env python3
"""Regenerate the byte-exact PRE-SPLIT original corpora from the intact
clean.jsonl files, into a read-only location.

The val-split rewrote <field>-scale.corpus.bin in place (train-only), which
invalidated slice offsets/sha256 that trial agents had recorded against the
originals. quality_filter.run() writes the same scrubbed text to clean.jsonl and
frames it into the corpus in one loop, so reframing clean.jsonl in order
reproduces the original corpus byte-for-byte. This writes those originals under
new names, verifies the token counts, prints sha256, and marks them read-only.
Does NOT touch any existing corpus file.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_filter import frame_document  # noqa: E402

BASE = Path("/home/kali/Music/cybersec ai")
RAW = BASE / "nina-data/raw-scale"
ORIG = BASE / "nina-data/corpora-scaled/originals"
FIELDS = ["cyber", "coding", "math", "science", "tech"]
# Known original token counts (from the run, pre-split) to verify against.
EXPECT = {"cyber": 90450742, "coding": 52683770, "math": 106460401,
          "science": 77982380, "tech": 39448467}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ORIG.mkdir(parents=True, exist_ok=True)
    allf_path = ORIG / "all-fields.full.corpus.bin"
    allf = allf_path.open("wb")
    results = []
    ok = True
    for field in FIELDS:
        clean = RAW / f"{field}.clean.jsonl"
        out = ORIG / f"{field}.full.corpus.bin"
        toks = 0
        with out.open("wb") as f, clean.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                frame = frame_document(json.loads(line)["text"])
                f.write(frame)
                allf.write(frame)
                toks += len(frame) // 2
        match = (toks == EXPECT[field])
        ok = ok and match
        results.append((field, out, toks, match))
    allf.close()

    # Mark everything read-only (0444).
    for _f, out, _t, _m in results:
        os.chmod(out, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(allf_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    total = sum(t for _f, _o, t, _m in results)
    print(f"regenerated originals -> {ORIG} (read-only)")
    for field, out, toks, match in results:
        print(f"  {field}.full.corpus.bin  {toks} tokens  verify={'OK' if match else 'MISMATCH'}  sha256={sha256_file(out)}")
    print(f"  all-fields.full.corpus.bin  {total} tokens  sha256={sha256_file(allf_path)}")
    print(f"all per-field token counts match the known originals: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
