#!/usr/bin/env python3
"""Create fixed-width spatial queries over annotated Geometry3K points."""

import argparse
import hashlib
import io
import itertools
import json
import math
import random
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

SOURCE = "https://raw.githubusercontent.com/lupantech/InterGPS/main/data/geometry3k/logic_forms.zip"
SHA256 = "264cd7316056d7ee57171b57397e3555c97e40d648a4e901db7baec59423665d"
KINDS = {"nearest": "n", "farthest": "f", "direction": "d", "inside": "i", "collinear": "c"}
ANSWERS = "01234567ABCDny"
POINT_NAMES = "ABCD"
SCENE_BYTES = 29
NCM_TARGET = 8
SEED = 42


def write_tokens(path, rows):
    with path.open("wb") as output:
        for row in rows:
            output.write(struct.pack(f"<{len(row)}H", *row))


def quantize(points):
    low_x = min(x for x, _ in points)
    high_y = max(y for _, y in points)
    span = max(max(x for x, _ in points) - low_x, high_y - min(y for _, y in points))
    if span == 0:
        return []
    # Image coordinates run down. The reasoner measures north on positive y.
    return [(round(15 * (x - low_x) / span), round(15 * (high_y - y) / span)) for x, y in points]


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def encode(points, kind, refs):
    coords = "".join(f"{name}{x:02}{y:02}" for name, (x, y) in zip(POINT_NAMES, points))
    names = "".join(POINT_NAMES[index] for index in refs).ljust(4, "_")
    return f"P04{coords}{KINDS[kind]}{names}?"


def answer(points, kind, refs):
    a = points[refs[0]]
    if kind in ("nearest", "farthest"):
        distances = {i: (x - a[0]) ** 2 + (y - a[1]) ** 2 for i, (x, y) in enumerate(points) if i != refs[0]}
        value = (min if kind == "nearest" else max)(distances.values())
        matches = [index for index, distance in distances.items() if distance == value]
        return POINT_NAMES[matches[0]] if len(matches) == 1 else None
    b, c, d = (points[i] for i in (*refs[1:], 0, 0)[:3])
    if kind == "direction":
        angle = (90 - math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 360
        return str(math.floor(angle / 45 + 0.5) % 8)
    if kind == "collinear":
        return "y" if cross(a, b, c) == 0 else "n"
    if cross(a, b, c) == 0:
        return None
    sides = [cross(a, b, d), cross(b, c, d), cross(c, a, d)]
    return "y" if min(sides) > 0 or max(sides) < 0 else "n"


def make_cases(diagrams):
    spaces = {name: [] for name in ("train", "dev", "test")}
    for identifier, diagram in sorted(diagrams.items(), key=lambda item: int(item[0])):
        raw = [tuple(pair) for _, pair in sorted(diagram["point_positions"].items())
               if isinstance(pair, (list, tuple)) and len(pair) == 2
               and all(isinstance(value, (int, float)) and math.isfinite(value) for value in pair)]
        points = list(dict.fromkeys(quantize(raw))) if raw else []
        if len(points) < 4:
            continue
        signature = tuple(sorted(points))
        group = hashlib.sha256(repr(signature).encode()).hexdigest()
        bucket = int(group[:8], 16) % 100
        split = "test" if bucket < 15 else "dev" if bucket < 25 else "train"
        spaces[split].append((identifier, points, signature))

    cases = {split: [] for split in spaces}
    seen_groups = set()
    seen_prompts = set()
    # Keep held-out input sequences first in the event of an identical query.
    for split in ("test", "dev", "train"):
        for identifier, points, signature in spaces[split]:
            if signature in seen_groups:
                continue
            seen_groups.add(signature)
            rng = random.Random(f"{SEED}/{identifier}")
            selections = {tuple(rng.sample(points, 4)) for _ in range(5)}
            for selected in sorted(selections):
                indices = tuple(range(4))
                references = {
                    "nearest": [(i,) for i in indices],
                    "farthest": [(i,) for i in indices],
                    "direction": rng.sample(list(itertools.permutations(indices, 2)), 6),
                    "inside": [(tuple(i for i in indices if i != d) + (d,)) for d in indices],
                    "collinear": list(itertools.combinations(indices, 3)),
                }
                for kind, queries in references.items():
                    for refs in queries:
                        label = answer(selected, kind, refs)
                        prompt = encode(selected, kind, refs)
                        if label is None or prompt in seen_prompts:
                            continue
                        seen_prompts.add(prompt)
                        cases[split].append({"diagram": identifier, "prompt": prompt, "answer": label, "kind": kind})
    return cases


def select(rows, limit):
    groups = defaultdict(list)
    for row in rows:
        groups[row["kind"], row["answer"]].append(row)
    rng = random.Random(SEED)
    selected = []
    for kind in KINDS:
        lists = [rows for (category, _), rows in sorted(groups.items()) if category == kind]
        if not lists:
            raise ValueError(f"No scenes for {kind}")
        cap = min(limit // len(lists), *(len(group) for group in lists))
        for group in lists:
            selected.extend(rng.sample(group, cap))
    rng.shuffle(selected)
    return selected


def write_split(directory, split, rows):
    geometry = directory / f"{split}.geom.bin"
    write_tokens(geometry, ((row["prompt"] + row["answer"]).encode() for row in rows))
    write_tokens(directory / f"{split}.ncm.bin",
                 (row["prompt"].encode() + row["answer"].encode() * NCM_TARGET for row in rows))
    write_tokens(directory / f"{split}.codec.bin",
                 (row["answer"].encode() * NCM_TARGET for row in rows))
    base = directory / f"{split}.scene-lm.bin"
    write_tokens(base, ((256, *row["prompt"].encode(), ord(row["answer"]), 257) for row in rows))
    with base.with_suffix(".mask").open("wb") as mask:
        mask.write((bytes(SCENE_BYTES + 1) + b"\x01\x00") * len(rows))
    kinds = Counter(row["kind"] for row in rows)
    meta = {"scene_len": SCENE_BYTES + 1, "points": 4, "kinds": list(KINDS),
            "kind_counts": [kinds[kind] for kind in KINDS], "scenes": len(rows),
            "answer_tokens": list(map(ord, ANSWERS)), "seed": SEED}
    (directory / f"{split}.meta.json").write_text(json.dumps(meta) + "\n")
    return dict(kinds)


def prepare(directory, archive):
    if archive:
        blob = archive.read_bytes()
    else:
        with urlopen(SOURCE, timeout=60) as stream:
            blob = stream.read()
    if hashlib.sha256(blob).hexdigest() != SHA256:
        raise ValueError("Unexpected Geometry3K annotation archive checksum")
    with zipfile.ZipFile(io.BytesIO(blob)) as source:
        diagrams = json.loads(source.read("logic_forms/diagram_logic_forms_annot.json"))
    cases = make_cases(diagrams)
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, limit in [("train", 10000), ("dev", 300), ("test", 40)]:
        cases[split] = select(cases[split], limit)
        counts[split] = write_split(directory, split, cases[split])
    with (directory / "train.traces.jsonl").open("w") as output:
        for index, row in enumerate(cases["train"]):
            record = {"id": str(index), "language": "unknown", "field": "geometry", "prompt": row["prompt"],
                      "completion": row["answer"], "source": "geometry3k"}
            output.write(json.dumps(record) + "\n")
    for split in ("dev", "test"):
        with (directory / f"{split}.jsonl").open("w") as output:
            for row in cases[split]:
                output.write(json.dumps(row) + "\n")
    report = {"source": SOURCE, "sha256": SHA256, "seed": SEED, "scenes_by_split_and_kind": counts,
              "label_space": ANSWERS, "feature_bytes": SCENE_BYTES, "ncm_target_length": NCM_TARGET,
              "total_source_diagrams": len(diagrams)}
    (directory / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    prepare(args.directory, args.archive)
