#!/usr/bin/env python3
"""Run one paired NCM depth trial on the public Geometry3K fixture."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from worker import checkpoint, report, run, write_json


SEEDS = range(42, 62)
DEPTHS = (1, 2)
MIDPOINT = 192
FINAL_STEP = 2 * MIDPOINT


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit(runtime, bundle, seed, depth, resume):
    binary = runtime / "dblocks"
    codec = checkpoint(runtime / "codec/weights")
    train = runtime / "scenes/train.ncm.bin"
    validation = runtime / "scenes/dev.ncm.bin"
    queries = runtime / "scenes/dev.jsonl"
    inputs = {path.relative_to(runtime).as_posix(): digest(path)
              for path in (binary, codec, train, validation, queries)}

    def state(path):
        return read_json(path.with_suffix(".state") / "state.json")

    if resume is not None:
        previous = read_json(resume.parent.parent / "trial.json")
        if (previous["inputs"], previous["seed"], previous["depth"], state(resume)["step"]) != (
            inputs, seed, depth, MIDPOINT
        ):
            raise ValueError("Resume requires the same runtime, inputs, seed, depth, and midpoint")

    bundle.mkdir(parents=True)
    config = bundle / "generator.json"
    write_json(config, {"blocks": 2, "hidden": 64, "heads": 4, "prompt_len": 29,
                        "attractor": True, "thinking_steps": depth})
    weights = bundle / "weights"
    resume_args = [] if resume is None else ["--resume", resume]
    started = perf_counter()
    run(binary, "ncm", "train", "--aligned-windows", "--config", config, "--tokenizer", codec,
        "--corpus", train, "--val-corpus", validation, "--out", weights,
        "--estimator", "all", "--seed", seed, "--batch-size", 8, "--warmup", 24,
        "--val-windows", 32, "--log-every", 48, "--steps", FINAL_STEP,
        "--checkpoint-every", MIDPOINT, *resume_args, log=bundle / "train.log")
    train_seconds = perf_counter() - started
    checkpoints = {state(path)["step"]: path.relative_to(bundle).as_posix()
                   for path in sorted(weights.glob("*.mpk"))}
    expected = {MIDPOINT, FINAL_STEP} if resume is None else {FINAL_STEP}
    if set(checkpoints) != expected:
        raise ValueError(f"Missing training steps in checkpoint sidecars: {checkpoints}")
    manifest = {"variant": f"ncm-{depth}", "checkpoint": checkpoints[FINAL_STEP],
                "tokenizer": str(codec)}
    cases = [json.loads(line) for line in queries.read_text(encoding="utf-8").splitlines()][:120]
    score = report(runtime, bundle, manifest, cases)
    write_json(bundle / "development.json", score)
    trial = {"seed": seed, "depth": depth, "checkpoints": checkpoints, "inputs": inputs,
             "parameters_sha256_prefix": Path(manifest["checkpoint"]).stem.rsplit("-", 1)[1],
             "resumed_from": None if resume is None else resume.name, "train_seconds": train_seconds,
             **{key: score[key] for key in ("cases", "correct", "per_kind_majority_correct")}}
    write_json(bundle / "trial.json", trial)


def summarize(trials, output):
    reports = [read_json(path) for path in sorted(trials.glob("*/trial.json"))]
    expected = {(seed, depth) for seed in SEEDS for depth in DEPTHS}
    observed = {(row["seed"], row["depth"]) for row in reports}
    if len(reports) != len(expected) or observed != expected:
        raise ValueError(f"Incomplete or duplicate trials. Missing {sorted(expected - observed)}")
    first = reports[0]
    if any(row["inputs"] != first["inputs"] or row["cases"] != first["cases"] for row in reports):
        raise ValueError("Trials used different runtime inputs or development cases")
    by_depth = {depth: [row["correct"] for row in reports if row["depth"] == depth] for depth in DEPTHS}
    write_json(output, {"jobs": len(reports), "development_cases": first["cases"],
                        "majority_correct": first["per_kind_majority_correct"],
                        "mean_correct_by_depth": {depth: mean(scores) for depth, scores in by_depth.items()},
                        "trials": reports})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trial = commands.add_parser("fit")
    trial.add_argument("--runtime", type=Path, required=True)
    trial.add_argument("--out", type=Path, required=True)
    trial.add_argument("--seed", type=int, choices=SEEDS, required=True)
    trial.add_argument("--depth", type=int, choices=DEPTHS, required=True)
    trial.add_argument("--resume", type=Path)
    collect = commands.add_parser("summarize")
    collect.add_argument("trials", type=Path)
    collect.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "fit":
        fit(args.runtime.resolve(), args.out.resolve(), args.seed, args.depth, args.resume)
    else:
        summarize(args.trials, args.output)


if __name__ == "__main__":
    main()
