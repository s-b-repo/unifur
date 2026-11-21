#!/usr/bin/env python3
"""Train, score, and use saved models on four-point spatial queries."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

LABELS = "01234567ABCDny"
GEOM_OPTIONS = ["--slots", "4", "--hidden-size", "64", "--num-blocks", "2", "--refine-steps", "2"]
STEPS = {
    "ncm-1": 6500, "ncm-2": 6500, "mosme-text": 6500,
    "geometry-answer": 6500, "geometry-diffusion": 4500,
    "geometry-mosme-answer": 6500, "geometry-mosme-diffusion": 4500,
    "geometry-baseline": 6500, "vector-student": 5000,
}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(*args, log=None):
    command = list(map(str, args))
    if log:
        with log.open("a", encoding="utf-8") as output:
            output.write("\n" + subprocess.list2cmdline(command) + "\n")
            output.flush()
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command)
        return None
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{command[:4]} exited {result.returncode}: {result.stderr or result.stdout}")
    return result.stdout


def checkpoint(folder):
    files = sorted(folder.glob("*.mpk"))
    if len(files) != 1:
        raise ValueError(f"Expected one final checkpoint in {folder}, got {len(files)}")
    return files[0]


def copy_checkpoint(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    weights = dest / source.name
    shutil.copy2(source, weights)
    state = source.with_suffix(".state")
    if state.is_dir():
        shutil.copytree(state, weights.with_suffix(".state"))
    return weights


def train_codec(runtime, directory, steps):
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "vocab": 259, "latent_dim": 32, "codebook_size": 32,
        "schedule": {"lengths": [2, 4, 8]}, "encoder_blocks": 0, "decoder_blocks": 1,
        "heads": 4, "ema_decay": 0.95, "commitment_beta": 0.25,
        "nested_dropout": 0.5, "dead_code_threshold": 0.03, "restart_every": 40,
    }
    path = directory / "codec.json"
    write_json(path, config)
    scenes = runtime / "scenes"
    run(runtime / "dblocks", "ncm", "tokenizer", "train", "--config", path,
        "--corpus", scenes / "train.codec.bin", "--val-corpus", scenes / "dev.codec.bin",
        "--steps", steps, "--batch-size", 32, "--val-windows", 128, "--log-every", 100,
        "--out", directory / "weights", log=directory / "train.log")
    model = checkpoint(directory / "weights")
    run(runtime / "dblocks", "ncm", "tokenizer", "eval", "--checkpoint", model,
        "--val-corpus", scenes / "test.codec.bin", "--val-windows", 128, log=directory / "test.log")


def marked_text(text):
    _, start, remainder = text.partition("\n---\n")
    body, end, _ = remainder.partition("\n---\n")
    if not start or not end:
        raise ValueError("Generation output is missing its delimiters")
    return body


def prediction(runtime, bundle, manifest, prompt, diffuse=False):
    variant = manifest["variant"]
    dblocks = runtime / "dblocks"
    weights = bundle / manifest["checkpoint"]
    if variant.startswith("ncm-"):
        result = run(dblocks, "ncm", "generate", "--generator", weights,
                     "--tokenizer", bundle / manifest["tokenizer"], "--top-k", 1, "--prompt", prompt)
        answer = marked_text(result)[:1]
    elif variant == "mosme-text":
        result = run(dblocks, "lm", "generate", "--checkpoint", weights, "--max-new", 1, "--prompt", prompt)
        answer = marked_text(result).removeprefix(prompt)[:1]
    elif variant in ("geometry-baseline", "vector-student"):
        mode = "baseline" if variant == "geometry-baseline" else "student"
        result = run(runtime / "scene_predict", mode, "--checkpoint", weights, "--scene", prompt,
                     "--config", bundle / manifest["configuration"])
        token = int(result.removeprefix("answer token: ").strip())
        answer = chr(token) if 0 <= token < 256 else ""
    else:
        flags = ["--diffusion"] if diffuse else []
        result = run(dblocks, "geom", "generate", "--checkpoint", weights, "--scene", prompt,
                     *GEOM_OPTIONS, "--moe-boxes", manifest["boxes"], *flags)
        match = re.search(r"^answer: (.*?) \| energy", result, re.M)
        if not match:
            raise ValueError("Missing geometric answer in inference output")
        answer = match[1]
    return answer if len(answer) == 1 and answer in LABELS else None


def report(runtime, bundle, manifest, cases, diffuse=False):
    predictions = [{**row, "prediction": prediction(runtime, bundle, manifest, row["prompt"], diffuse)}
                   for row in cases]
    by_kind = {}
    for kind in sorted({row["kind"] for row in predictions}):
        matches = [row for row in predictions if row["kind"] == kind]
        by_kind[kind] = {"cases": len(matches),
                         "correct": sum(row["prediction"] == row["answer"] for row in matches)}
    total = len(predictions)
    correct = sum(row["prediction"] == row["answer"] for row in predictions)
    baseline = sum(max(Counter(row["answer"] for row in predictions if row["kind"] == kind).values())
                   for kind in by_kind)
    print(f"{manifest['variant']} {'denoised' if diffuse else 'clean'}: {correct}/{total} exact; "
          f"per-kind majority {baseline}/{total}", flush=True)
    return {"cases": total, "correct": correct, "accuracy": correct / total,
            "per_kind_majority_correct": baseline, "by_kind": by_kind, "predictions": predictions}


def fit(runtime, variant, bundle, steps, test_limit):
    bundle.mkdir(parents=True, exist_ok=True)
    dblocks = runtime / "dblocks"
    data = runtime / "scenes"
    out = bundle / "weights"
    log = bundle / "train.log"
    manifest = {"variant": variant, "steps_requested": steps, "input": "four-point geometry scene"}
    common = ["--steps", steps, "--log-every", 100]
    if variant.startswith("ncm-"):
        codec = copy_checkpoint(checkpoint(runtime / "codec/weights"), bundle / "tokenizer")
        manifest["tokenizer"] = codec.relative_to(bundle).as_posix()
        config = bundle / "generator.json"
        write_json(config, {"blocks": 3, "hidden": 96, "heads": 4, "prompt_len": 29,
                            "thinking_steps": int(variant[-1]), "attractor": True})
        run(dblocks, "ncm", "train", "--aligned-windows", "--config", config, "--tokenizer", codec,
            "--corpus", data / "train.ncm.bin", "--val-corpus", data / "dev.ncm.bin",
            "--batch-size", 16, "--warmup", min(40, steps), "--estimator", "all", "--val-windows", 200,
            "--out", out, *common, log=log)
    elif variant == "mosme-text":
        spec = bundle / "experts.json"
        run(dblocks, "experts", "init", "--out", spec, "--box", "measure:distance,bearing",
            "--box", "shape:incidence,containment", log=log)
        run(dblocks, "lm", "train", "--tiny", "--context", 64, "--hidden-size", 64,
            "--num-layers", 3, "--num-heads", 4, "--intermediate-size", 128,
            "--mosme-spec", spec, "--mosme-every", 2, "--corpus", data / "train.scene-lm.bin",
            "--batch-size", 16, "--out-dir", out, *common, log=log)
    elif variant == "vector-student":
        config = bundle / "student.json"
        write_json(config, {"hidden_size": 64, "num_layers": 2, "num_heads": 4, "num_kv_heads": 2,
                            "context": 64, "intermediate_size": 96, "attention": "dense",
                            "cond_hidden_size": 32, "frequency_embedding_size": 16,
                            "geom_slots": 4, "geom_dim": 8, "geom_refine_steps": 2,
                            "geom_cond_hidden": 32, "geom_frequency": 16, "experts_per_box": 2,
                            "coding_boxes": ["python"], "field_boxes": ["geometry"]})
        manifest["configuration"] = config.name
        run(dblocks, "student", "train", "--language", "unknown", "--traces", data / "train.traces.jsonl",
            "--config", config, "--batch-size", 16, "--lr", 0.001, "--out-dir", out, *common, log=log)
    else:
        baseline = variant == "geometry-baseline"
        boxes = 5 if "mosme" in variant or baseline else 0
        manifest["boxes"] = boxes
        if baseline:
            shutil.copy2(data / "train.meta.json", bundle / "geometry.json")
            manifest["configuration"] = "geometry.json"
        objective = variant.rsplit("-", 1)[-1]
        option = [] if baseline else ["--objective", objective]
        run(dblocks, "geom", "baseline" if baseline else "train", "--corpus", data / "train.geom.bin",
            "--meta", data / "train.meta.json", "--batch-size", 24, "--moe-boxes", boxes,
            *GEOM_OPTIONS, *option, "--out-dir", out, *common, log=log)
    model = checkpoint(out)
    manifest["checkpoint"] = model.relative_to(bundle).as_posix()
    with model.open("rb") as stream:
        manifest["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(bundle / "model.json", manifest)
    shutil.copy2(data / "dataset.json", bundle / "dataset.json")
    shutil.copy2(data / "test.jsonl", bundle / "test.jsonl")
    shutil.copy2(Path(__file__), bundle / "use_model.py")
    cases = [json.loads(line) for line in (bundle / "test.jsonl").read_text().splitlines()][:test_limit]
    clean = report(runtime, bundle, manifest, cases)
    write_json(bundle / "heldout.json", clean)
    row = cases[0]
    write_json(bundle / "demo.json", {**row, "prediction": clean["predictions"][0]["prediction"]})
    if variant.endswith("diffusion"):
        write_json(bundle / "heldout-diffusion.json", report(runtime, bundle, manifest, cases, diffuse=True))
    if variant == "vector-student":
        trace = run(dblocks, "student", "think", "--config", bundle / manifest["configuration"],
                    "--checkpoint", model, "--prompt", row["prompt"], "--depths", "1,2")
        (bundle / "demo-thought-trace.txt").write_text(trace, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="directory containing the release executables")
    commands = parser.add_subparsers(dest="action", required=True)
    codec = commands.add_parser("codec")
    codec.add_argument("--steps", type=int, default=2200)
    train = commands.add_parser("train")
    train.add_argument("variant", choices=STEPS)
    train.add_argument("bundle", type=Path)
    train.add_argument("--steps", type=int)
    train.add_argument("--test-limit", type=int, default=200)
    use = commands.add_parser("use")
    use.add_argument("bundle", type=Path)
    use.add_argument("--scene", help="use the saved demo scene when omitted")
    use.add_argument("--test", action="store_true", help="score the complete packaged held-out set")
    use.add_argument("--diffusion", action="store_true", help="denoise from noise with a geometric diffusion model")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if args.action == "codec":
        train_codec(runtime, runtime / "codec", args.steps)
    elif args.action == "train":
        if args.test_limit < 1 or (args.steps is not None and args.steps < 1):
            parser.error("steps and test limit must be positive")
        fit(runtime, args.variant, args.bundle, args.steps or STEPS[args.variant], args.test_limit)
    else:
        manifest = json.loads((args.bundle / "model.json").read_text())
        if args.diffusion and not manifest["variant"].endswith("diffusion"):
            parser.error("--diffusion needs a geometric diffusion checkpoint")
        if args.test:
            cases = [json.loads(line) for line in (args.bundle / "test.jsonl").read_text().splitlines()]
            result = report(runtime, args.bundle, manifest, cases, args.diffusion)
            print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))
        else:
            scene = args.scene or json.loads((args.bundle / "demo.json").read_text())["prompt"]
            if len(scene) != 29 or not scene.isascii() or not scene.endswith("?"):
                parser.error("a four-point scene must have 29 ASCII bytes and end in '?'")
            print(json.dumps({"scene": scene, "answer": prediction(runtime, args.bundle, manifest, scene, args.diffusion)}))


if __name__ == "__main__":
    main()
