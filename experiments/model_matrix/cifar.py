#!/usr/bin/env python3
"""Extract and package the canonical CIFAR-100 binary splits."""

import argparse
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

SOURCE = "https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz"
MD5 = "03b5dce01913d631647c71ecec9e9cb8"
RECORD_BYTES = 3074


def extract(archive, out):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != MD5:
            raise ValueError("The archive does not match the published CIFAR-100 checksum")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = Path(member.name).name
            if member.isfile() and name in {"train.bin", "test.bin", "fine_label_names.txt"}:
                with stream.extractfile(member) as source, (out / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
    for split, count in [("train", 50000), ("test", 10000)]:
        if (out / f"{split}.bin").stat().st_size != count * RECORD_BYTES:
            raise ValueError(f"Unexpected {split} size in the official archive")
    if len((out / "fine_label_names.txt").read_text().splitlines()) != 100:
        raise ValueError("The official archive should have 100 fine labels")


def bundle(dataset, out):
    weights, = (out / "weights").glob("*.mpk")
    train_card = {"variant": "mosme-image", "checkpoint": weights.relative_to(out).as_posix(),
                  "dataset": SOURCE, "archive_md5": MD5, "training_images": 50000,
                  "input": "32x32 planar RGB", "record_bytes": RECORD_BYTES}
    with weights.open("rb") as stream:
        train_card["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (out / "model.json").write_text(json.dumps(train_card, indent=2) + "\n")
    shutil.copy2(dataset / "fine_label_names.txt", out / "fine_label_names.txt")
    counts = Counter()
    with (dataset / "test.bin").open("rb") as source, (out / "heldout.bin").open("wb") as test:
        for record in iter(lambda: source.read(RECORD_BYTES), b""):
            label = record[1]
            if counts[label] < 5:
                if not counts:
                    (out / "demo.rgb").write_bytes(record[2:])
                    (out / "demo-expected.json").write_text(json.dumps({"label": label}) + "\n")
                counts[label] += 1
                test.write(record)
    if len(counts) != 100 or any(count != 5 for count in counts.values()):
        raise ValueError("Expected five official test records per fine label")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    read = commands.add_parser("extract")
    read.add_argument("archive", type=Path)
    read.add_argument("dataset", type=Path)
    save = commands.add_parser("bundle")
    save.add_argument("dataset", type=Path)
    save.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.action == "extract":
        extract(args.archive, args.dataset)
    else:
        bundle(args.dataset, args.out)
