"""Validate packed Nina data and run one resumable native NCM training stage."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess


FILES = ("train.aligned.bin", "train.codec.bin", "validation.aligned.bin", "validation.codec.bin")
FOUR_BILLION = {"blocks": 35, "hidden": 3072, "heads": 48, "mlp_ratio": 4,
                "thinking_steps": 2, "attractor": True, "alpha": 1.0}


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def geometry_fixture(scenes, output):
    """Prepare the public Geometry3K data in the same format as Nina instruction data."""
    from shutil import copyfile

    source = read_json(scenes / "dataset.json")
    output.mkdir(parents=True, exist_ok=False)
    for prefix, split in (("train", "train"), ("validation", "dev")):
        copyfile(scenes / f"{split}.ncm.bin", output / f"{prefix}.aligned.bin")
        copyfile(scenes / f"{split}.codec.bin", output / f"{prefix}.codec.bin")
    write_json(output / "manifest.json", {
        "prompt_width_bytes": source["feature_bytes"],
        "response_width_bytes_including_eos": source["ncm_target_length"],
        "packed_by_split_and_field": {
            prefix: source["scenes_by_split_and_kind"][split]
            for prefix, split in (("train", "train"), ("validation", "dev"))
        },
        "corpus_hashes": {name: digest(output / name) for name in FILES},
        "dataset": source,
    })


def validate(bundle):
    manifest = read_json(bundle / "manifest.json")
    prompt = manifest["prompt_width_bytes"]
    target = manifest["response_width_bytes_including_eos"]
    if type(prompt) is not int or type(target) is not int or min(prompt, target) < 1:
        raise ValueError("Both record widths must be positive integers")
    counts = {}
    for split in ("train", "validation"):
        count = sum(manifest["packed_by_split_and_field"][split].values())
        if count < 1:
            raise ValueError(f"Empty {split} split")
        counts[split] = count
        for kind, width in (("aligned", prompt + target), ("codec", target)):
            name = f"{split}.{kind}.bin"
            path = bundle / name
            if path.stat().st_size != 2 * width * count or digest(path) != manifest["corpus_hashes"][name]:
                raise ValueError(f"{name} disagrees with the declared record count or digest")
    if "packed_ids" in manifest:
        train = set(manifest["packed_ids"]["train"])
        validation = set(manifest["packed_ids"]["validation"])
        if len(train) != counts["train"] or len(validation) != counts["validation"] or train & validation:
            raise ValueError("Packed IDs must be unique and disjoint across splits")
    return {"manifest_sha256": digest(bundle / "manifest.json"), "records": counts,
            "prompt_width": prompt, "target_width": target}


def checkpoint(directory, kind):
    matches = []
    for weight in directory.glob(f"{kind}-*.mpk"):
        path = weight.with_suffix(".state") / "state.json"
        if path.is_file():
            state = read_json(path)
            if state["kind"] == kind:
                matches.append((state["step"], weight))
    return max(matches, default=(0, None))


def fit(args):
    data = validate(args.input)
    width = data["target_width"]
    small = args.profile == "preflight"
    if small and width != 8:
        raise ValueError("The preflight expects the Geometry3K eight-byte target")
    if not small and (width != 1024 or data["prompt_width"] != 1024):
        raise ValueError("The 4B recipe expects 1024-byte prompt and target records")
    if not small and args.backend != "wgpu":
        raise ValueError("The 4B recipe requires a discrete WGPU training worker")
    if min(args.steps, args.batch_size, args.checkpoint_every) < 1:
        raise ValueError("Training steps, batch size and checkpoint interval must be positive")
    if args.generator_lr_schedule != "cosine" and args.stage != "generator":
        raise ValueError("The generator learning-rate schedule only applies to the generator stage")
    planned_records = args.steps * args.batch_size
    if not small and planned_records < data["records"]["train"]:
        raise ValueError("The 4B run must schedule at least one pass over every packed training record")

    args.out.mkdir(parents=True, exist_ok=True)
    directory = args.out / args.stage
    options = [args.binary, "ncm"]
    if args.backend == "wgpu":
        options += ["--backend", "wgpu", "--device", args.device]
    codec = args.stage == "codec"
    kind = "ncm-tokenizer" if codec else "ncm-generator"
    previous_step, resume = checkpoint(directory, kind)
    if previous_step > args.steps:
        raise ValueError("Existing checkpoint is past the requested final step")
    immutable = {
        "data": data, "profile": args.profile, "backend": args.backend, "steps": args.steps,
        "batch_size": args.batch_size, "checkpoint_every": args.checkpoint_every, "device": args.device,
        "stage": args.stage, "binary_sha256": digest(args.binary),
    }
    if args.generator_lr_schedule != "cosine":
        immutable["generator_lr_schedule"] = args.generator_lr_schedule
    if args.seed != 42:
        immutable["seed"] = args.seed
    recipe = args.out / f"{args.stage}-recipe.json"
    if recipe.exists() and read_json(recipe) != immutable:
        raise ValueError("The run directory belongs to a different recipe")
    write_json(recipe, immutable)
    if previous_step == args.steps:
        print(json.dumps({"stage": args.stage, "step": previous_step, "checkpoint": str(resume)}))
        return
    if codec:
        config = "paper-owt"
        if small:
            config = args.out / "codec.json"
            write_json(config, {"vocab": 259, "latent_dim": 32, "codebook_size": 32,
                                "schedule": {"lengths": [2, 4, 8]}, "encoder_blocks": 0,
                                "decoder_blocks": 1, "heads": 4, "ema_decay": 0.95,
                                "commitment_beta": 0.25, "nested_dropout": 0.5,
                                "dead_code_threshold": 0.03})
        options += ["tokenizer", "train", "--config", config, "--ordered-records"]
        stream = "codec"
    else:
        _, tokenizer = checkpoint(args.out / "codec", "ncm-tokenizer")
        if tokenizer is None:
            raise ValueError("Train the codec before dispatching the generator")
        config = args.out / "generator.json"
        shape = {"blocks": 2, "hidden": 64, "heads": 4, "thinking_steps": 2, "attractor": True} if small else FOUR_BILLION
        write_json(config, {**shape, "prompt_len": data["prompt_width"]})
        warmup = min(100, args.steps)
        options += ["train"]
        if args.generator_lr_schedule == "wsd":
            warmup = min(100, args.steps // 20)
            options += ["--lr-schedule", "wsd"]
        options += ["--config", config, "--tokenizer", tokenizer, "--aligned-windows",
                    "--ordered-records", "--warmup", warmup]
        stream = "aligned"
    options += ["--corpus", args.input / f"train.{stream}.bin",
                "--val-corpus", args.input / f"validation.{stream}.bin",
                "--steps", args.steps, "--batch-size", args.batch_size, "--out", directory,
                "--seed", args.seed,
                "--checkpoint-every", args.checkpoint_every, "--val-windows", 8 if small else 128]
    if resume is not None:
        options += ["--resume", resume]
    subprocess.run([str(arg) for arg in options], check=True)
    step, final = checkpoint(directory, kind)
    if step != args.steps:
        raise RuntimeError(f"Expected a reloadable checkpoint at step {args.steps}, got {step}")
    full_passes, records_in_next_pass = divmod(planned_records, data["records"]["train"])
    print(json.dumps({"stage": args.stage, "step": step, "checkpoint": str(final),
                      "full_passes": full_passes, "records_in_next_pass": records_in_next_pass}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    geometry = commands.add_parser("geometry-fixture")
    geometry.add_argument("scenes", type=Path)
    geometry.add_argument("out", type=Path)
    checking = commands.add_parser("validate")
    checking.add_argument("input", type=Path)
    job = commands.add_parser("fit")
    job.add_argument("--input", type=Path, required=True)
    job.add_argument("--out", type=Path, required=True)
    job.add_argument("--binary", type=Path, required=True)
    job.add_argument("--stage", choices=("codec", "generator"), required=True)
    job.add_argument("--profile", choices=("preflight", "4b"), required=True)
    job.add_argument("--backend", choices=("cpu", "wgpu"), default="cpu")
    job.add_argument("--device", default="discrete:0")
    job.add_argument("--steps", type=int, required=True)
    job.add_argument("--seed", type=int, default=42)
    job.add_argument("--generator-lr-schedule", choices=("cosine", "wsd"), default="cosine")
    job.add_argument("--batch-size", type=int, default=1)
    job.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    try:
        if args.action == "geometry-fixture":
            geometry_fixture(args.scenes, args.out)
        elif args.action == "validate":
            print(json.dumps(validate(args.input), indent=2))
        else:
            fit(args)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
