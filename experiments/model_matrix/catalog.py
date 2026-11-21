#!/usr/bin/env python3
"""List, test, and run trained models downloaded from the worker matrix."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare import KINDS, POINT_NAMES, encode

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "data/matrix-results"
REFERENCE_COUNTS = {"nearest": 1, "farthest": 1, "direction": 2, "inside": 4, "collinear": 3}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scene_query(path):
    obj = read_json(path)
    points = obj["points"]
    kind = obj["kind"]
    refs = obj["refs"]
    if not isinstance(points, list) or len(points) != 4 or any(
        not isinstance(pair, list) or len(pair) != 2
        or any(type(x) is not int or not 0 <= x <= 15 for x in pair)
        for pair in points
    ):
        raise ValueError("points must contain four integer coordinate pairs on the 0–15 grid")
    if len(set(map(tuple, points))) != 4:
        raise ValueError("points must be distinct")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not isinstance(refs, list) or len(refs) != REFERENCE_COUNTS[kind] or any(
        not isinstance(ref, str) or len(ref) != 1 or ref not in POINT_NAMES for ref in refs
    ) or len(set(refs)) != len(refs):
        raise ValueError(f"{kind} needs {REFERENCE_COUNTS[kind]} distinct references from A, B, C, D")
    return encode(points, kind, [POINT_NAMES.index(ref) for ref in refs])


def command(root, name, action, scene, image, diffuse, trace):
    bundle = root / f"model-{name}"
    test = action == "test"
    spatial = root / "spatial-runtime"
    extended = root / "fusion-image-runtime"

    if name == "ncm-tokenizer":
        if scene or image or diffuse or trace:
            raise ValueError("the tokenizer demo and test use its packaged codec split")
        weights, = (spatial / "codec/weights").glob("*.mpk")
        return [spatial / "dblocks", "ncm", "tokenizer", "eval", "--checkpoint", weights,
                "--val-corpus", spatial / "scenes/test.codec.bin", "--val-windows", 128 if test else 4]
    if name == "mosme-image":
        if scene or diffuse or trace:
            raise ValueError("mosme-image accepts --image for a 3072-byte planar RGB image")
        flags = ["--test"] if test else ["--image", image] if image else []
        return [extended / "mosme_image", "--bundle", bundle, *flags]
    if image:
        raise ValueError("--image applies to mosme-image")
    if name.startswith("fusion-"):
        if diffuse or trace:
            raise ValueError("fused checkpoints expose the kernel-checked answer path")
        flags = ["--cases", bundle / "test.jsonl"] if test else ["--scene", scene or read_json(bundle / "demo.json")["scene"]]
        return [extended / "fused_scene", "use", "--bundle", bundle, *flags]
    if trace:
        if name != "vector-student" or test:
            raise ValueError("--trace applies to a vector-student prediction")
        card = read_json(bundle / "model.json")
        return [spatial / "dblocks", "student", "think", "--checkpoint", bundle / card["checkpoint"],
                "--config", bundle / card["configuration"], "--depths", "1,2",
                "--prompt", scene or read_json(bundle / "demo.json")["prompt"]]
    flags = ["--test"] if test else ["--scene", scene] if scene else []
    if diffuse:
        flags.append("--diffusion")
    return [sys.executable, "-B", bundle / "use_model.py", "--runtime", spatial, "use", bundle, *flags]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("action", choices=("list", "demo", "test"))
    parser.add_argument("model", nargs="?", default="all")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--scene", help="fixed-width four-point scene")
    inputs.add_argument("--query", type=Path, help='JSON with points, kind and refs, e.g. refs ["A"] for nearest')
    inputs.add_argument("--image", type=Path, help="32×32 RGB image in planar byte format")
    parser.add_argument("--diffusion", action="store_true", help="sample the standalone geometry diffusion path")
    parser.add_argument("--trace", action="store_true", help="print the trained vector student's thought trace")
    args = parser.parse_args()
    root = args.artifacts.resolve()
    models = sorted(path.parent.name.removeprefix("model-") for path in root.glob("model-*/heldout.json"))
    if (root / "spatial-runtime/codec/test.log").is_file():
        models.append("ncm-tokenizer")
    if args.action == "list":
        for name in models:
            if name == "ncm-tokenizer":
                print(f"{name:25} reconstruction (see codec/test.log)")
            else:
                result = read_json(root / f"model-{name}/heldout.json")
                print(f"{name:25} {result['correct']:>3}/{result['cases']:<3} exact")
        if not models:
            parser.error(f"no evaluated model bundles in {root}")
        return
    selected = models if args.model == "all" else [args.model]
    if not selected or any(name not in models for name in selected):
        parser.error(f"available models: {', '.join(models) or 'none'}")
    if args.model == "all" and (args.scene or args.query or args.image or args.diffusion or args.trace):
        parser.error("choose one model when supplying custom inputs or inference options")
    if args.action == "test" and (args.scene or args.query or args.image):
        parser.error("test uses the packaged held-out set")
    try:
        scene = scene_query(args.query) if args.query else args.scene
        for name in selected:
            print(f"[{name}]", flush=True)
            argv = command(root, name, args.action, scene, args.image, args.diffusion, args.trace)
            env = {**os.environ, "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "2")}
            summarize = args.action == "test" and (name.startswith("fusion-") or name == "mosme-image")
            result = subprocess.run(list(map(str, argv)), check=True, env=env, text=True,
                                    capture_output=summarize)
            if summarize:
                report = json.loads(result.stdout)
                report.pop("predictions", None)
                print(json.dumps(report, indent=2))
    except subprocess.CalledProcessError as error:
        parser.error((error.stderr or str(error)).strip())
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
