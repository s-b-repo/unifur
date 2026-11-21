#!/usr/bin/env python3
"""Extract and package the canonical CIFAR-100 binary splits."""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

SOURCE = "https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz"
MD5 = "03b5dce01913d631647c71ecec9e9cb8"
RECORD_BYTES = 3074
MIRROR = "https://huggingface.co/datasets/uoft-cs/cifar100"
REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
SPLITS = {"train": 50000, "test": 10000}


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
    for split, count in SPLITS.items():
        if (out / f"{split}.bin").stat().st_size != count * RECORD_BYTES:
            raise ValueError(f"Unexpected {split} size in the official archive")
    if len((out / "fine_label_names.txt").read_text().splitlines()) != 100:
        raise ValueError("The official archive should have 100 fine labels")
    (out / "source.json").write_text(json.dumps({"dataset": SOURCE, "archive_md5": MD5}) + "\n")


def convert_parquet(path, destination, count):
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.ParquetFile(path)
    if table.metadata.num_rows != count:
        raise ValueError(f"Unexpected row count for {path.name}")
    metadata = json.loads(table.schema_arrow.metadata[b"huggingface"])
    names = metadata["info"]["features"]["fine_label"]["names"]
    if len(names) != 100 or len(set(names)) != 100:
        raise ValueError(f"Unexpected fine classes for {path.name}")
    with destination.open("wb") as output:
        for batch in table.iter_batches(batch_size=512, columns=["img", "fine_label", "coarse_label"]):
            for row in batch.to_pylist():
                fine, coarse = row["fine_label"], row["coarse_label"]
                if not 0 <= fine < 100 or not 0 <= coarse < 20:
                    raise ValueError(f"Out-of-range class in {path.name}")
                with Image.open(io.BytesIO(row["img"]["bytes"])) as image:
                    if image.size != (32, 32):
                        raise ValueError(f"Unexpected image dimensions in {path.name}")
                    rgb = image.convert("RGB").tobytes()
                output.write(bytes((coarse, fine)) + rgb[0::3] + rgb[1::3] + rgb[2::3])
    return names


def fetch_mirror(out):
    out.mkdir(parents=True, exist_ok=True)
    classes = None
    shards = {}
    for split, count in SPLITS.items():
        filename = f"{split}-00000-of-00001.parquet"
        location = f"{MIRROR}/resolve/{REVISION}/cifar100/{filename}"
        local = out.parent / filename
        with urlopen(location, timeout=60) as source, local.open("wb") as target:
            shutil.copyfileobj(source, target)
        with local.open("rb") as source:
            shards[filename] = hashlib.file_digest(source, "sha256").hexdigest()
        names = convert_parquet(local, out / f"{split}.bin", count)
        if classes is not None and names != classes:
            raise ValueError("The train and test splits use different class IDs")
        classes = names
    (out / "fine_label_names.txt").write_text("\n".join(classes) + "\n")
    (out / "source.json").write_text(json.dumps({"dataset": MIRROR, "revision": REVISION,
                                                "shard_sha256": shards}, indent=2) + "\n")


def bundle(dataset, out):
    weights, = (out / "weights").glob("*.mpk")
    train_card = {"variant": "mosme-image", "checkpoint": weights.relative_to(out).as_posix(),
                  **json.loads((dataset / "source.json").read_text()), "training_images": 50000,
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
    fetch = commands.add_parser("mirror")
    fetch.add_argument("dataset", type=Path)
    save = commands.add_parser("bundle")
    save.add_argument("dataset", type=Path)
    save.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.action == "extract":
        extract(args.archive, args.dataset)
    elif args.action == "mirror":
        fetch_mirror(args.dataset)
    else:
        bundle(args.dataset, args.out)
