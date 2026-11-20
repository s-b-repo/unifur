#!/usr/bin/env python3
"""Frame the original MATH dataset for the NCM geometry experiment."""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pyarrow.parquet as parquet

EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT.parents[1] / "cybercorpus"))
from quality_filter import frame_document  # noqa: E402

DATASET = "EleutherAI/hendrycks_math"
REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def download(path):
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{path}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as stream:
                return stream.read()
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def problem_key(problem):
    text = " ".join(unicodedata.normalize("NFKC", problem).split()).casefold()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def final_box(solution):
    _, marker, suffix = solution.rpartition(r"\boxed{")
    if not marker:
        return None
    depth = 1
    for index, character in enumerate(suffix):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return suffix[:index].strip()
    return None


def answer_prompt(problem):
    return f"Problem:\n{problem}\nAnswer:"


def document(row):
    problem = row["problem"]
    solution = row["solution"]
    answer = final_box(solution)
    prefix = f"{answer_prompt(problem)} {answer}" if answer else f"Problem:\n{problem}"
    return frame_document(f"{prefix}\nSolution:\n{solution}")


def read_dataset():
    splits = {"train": {}, "test": {}}
    shards = {}
    for split in splits:
        for subject in SUBJECTS:
            path = f"{subject}/{split}-00000-of-00001.parquet"
            blob = download(path)
            shards[path] = hashlib.sha256(blob).hexdigest()
            table = parquet.read_table(BytesIO(blob), columns=["problem", "solution", "level"])
            records = []
            for row in table.to_pylist():
                problem = row["problem"]
                solution = row["solution"]
                if not problem or not solution or not problem.strip() or not solution.strip():
                    raise ValueError(f"Missing problem or solution in {path}")
                records.append({**row, "problem": problem.strip(), "solution": solution.strip()})
            splits[split][subject] = records
    return splits, shards


def prepare(out):
    splits, shards = read_dataset()
    official_test = {
        problem_key(row["problem"]) for records in splits["test"].values() for row in records
    }
    length_limit = json.loads((EXPERIMENT / "generator.json").read_text())["prompt_len"]
    out.mkdir(parents=True, exist_ok=True)
    exclusions = Counter()
    counts = Counter()
    seen = set()
    # Assign every copy of a problem consistently using its normalized text key.
    with (out / "math.train.corpus.bin").open("wb") as math, \
         (out / "geometry.train.corpus.bin").open("wb") as geometry, \
         (out / "geometry.dev.corpus.bin").open("wb") as dev:
        for subject, rows in splits["train"].items():
            for row in rows:
                key = problem_key(row["problem"])
                if key in official_test:
                    exclusions["official_test_overlap"] += 1
                    continue
                if key in seen:
                    exclusions["repeated_training_problem"] += 1
                    continue
                seen.add(key)
                frame = document(row)
                if subject == "geometry" and int(key[:8], 16) % 10 == 0:
                    dev.write(frame)
                    counts["geometry.dev"] += 1
                    continue
                math.write(frame)
                counts[f"{subject}.train"] += 1
                if subject == "geometry":
                    geometry.write(frame)

    seen.clear()
    with (out / "geometry.test.corpus.bin").open("wb") as test, \
         (out / "geometry.integer-test.jsonl").open("w", encoding="utf-8") as answer_test:
        for row in splits["test"]["geometry"]:
            key = problem_key(row["problem"])
            if key in seen:
                continue
            seen.add(key)
            test.write(document(row))
            counts["geometry.test"] += 1
            prompt = answer_prompt(row["problem"])
            answer = final_box(row["solution"])
            if answer and re.fullmatch(r"[+-]?[0-9]+", answer) and len(prompt.encode()) <= length_limit:
                case = {"id": key, "level": row["level"], "prompt": prompt, "answer": int(answer)}
                answer_test.write(json.dumps(case) + "\n")
                counts["geometry.integer_test"] += 1

    if not all(counts[name] for name in ("geometry.train", "geometry.dev", "geometry.integer_test")):
        raise ValueError("Empty geometry train, development, or integer-answer test split")
    files = {}
    for name in ("math.train", "geometry.train", "geometry.dev", "geometry.test"):
        path = out / f"{name}.corpus.bin"
        with path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        files[path.name] = {"tokens": path.stat().st_size // 2, "sha256": sha256}
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "license": "MIT (dataset card)",
        "shards_sha256": shards,
        "documents": counts,
        "excluded_from_train": exclusions,
        "corpora": files,
        "geometry_test_prompt_limit_bytes": length_limit,
    }
    (out / "corpus.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    prepare(parser.parse_args().out)
