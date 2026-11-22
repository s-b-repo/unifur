"""Validate packed Nina data and run one resumable native NCM training stage."""

import argparse
import fcntl
import json
import os
import resource
import shutil
import subprocess
import sys
from array import array
from contextlib import contextmanager
from hashlib import file_digest
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

FILES = ("train.aligned.bin", "train.codec.bin", "validation.aligned.bin", "validation.codec.bin")
FOUR_BILLION = {
    "blocks": 35,
    "hidden": 3072,
    "heads": 48,
    "mlp_ratio": 4,
    "thinking_steps": 2,
    "attractor": True,
    "alpha": 1.0,
}
GENERATOR_PARAMETERS = 3_977_714_688


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another Nina worker owns {directory}") from error
        yield


def geometry_fixture(scenes, output):
    """Prepare the public Geometry3K data in the same format as Nina instruction data."""
    from shutil import copyfile

    source = read_json(scenes / "dataset.json")
    output.mkdir(parents=True, exist_ok=False)
    for prefix, split in (("train", "train"), ("validation", "dev")):
        copyfile(scenes / f"{split}.ncm.bin", output / f"{prefix}.aligned.bin")
        copyfile(scenes / f"{split}.codec.bin", output / f"{prefix}.codec.bin")
    write_json(
        output / "manifest.json",
        {
            "prompt_width_bytes": source["feature_bytes"],
            "response_width_bytes_including_eos": source["ncm_target_length"],
            "packed_by_split_and_field": {
                prefix: source["scenes_by_split_and_kind"][split]
                for prefix, split in (("train", "train"), ("validation", "dev"))
            },
            "corpus_hashes": {name: digest(output / name) for name in FILES},
            "dataset": source,
        },
    )


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
            if (
                path.stat().st_size != 2 * width * count
                or digest(path) != manifest["corpus_hashes"][name]
            ):
                raise ValueError(f"{name} disagrees with the declared record count or digest")
    if "packed_ids" in manifest:
        train = set(manifest["packed_ids"]["train"])
        validation = set(manifest["packed_ids"]["validation"])
        if (
            len(train) != counts["train"]
            or len(validation) != counts["validation"]
            or train & validation
        ):
            raise ValueError("Packed IDs must be unique and disjoint across splits")
    return {
        "manifest_sha256": digest(bundle / "manifest.json"),
        "records": counts,
        "prompt_width": prompt,
        "target_width": target,
    }


def checkpoint(directory, kind):
    matches = []
    for weight in directory.glob(f"{kind}-*.mpk"):
        path = weight.with_suffix(".state") / "state.json"
        if path.is_file():
            state = read_json(path)
            if state["kind"] == kind:
                matches.append((state["step"], weight))
    return max(matches, default=(0, None))


def verify_checkpoint(weight, kind, step, *, optimizer=True):
    directory = weight.with_suffix(".state")
    state = read_json(directory / "state.json")
    if (state["format_version"], state["kind"], state["step"]) != (2, kind, step):
        raise ValueError(f"{weight} has incompatible checkpoint state")
    if digest(weight) != state["model"]["sha256"]:
        raise ValueError(f"{weight} has a corrupt model record")
    if optimizer and not state["optimizer"]:
        raise ValueError(f"{weight} has no optimizer record")
    for name in ("optimizer", "ema", "head", "head_optimizer") if optimizer else ():
        entry = state.get(name)
        if entry:
            filename = Path(entry["path"])
            if filename.name != str(filename) or digest(directory / filename) != entry["sha256"]:
                raise ValueError(f"{weight} has an invalid {name} record")
    return state


def gpu_memory():
    """Return framebuffer samples for the visible NVIDIA or AMD adapters."""
    try:
        driver = shutil.which("nvidia-smi")
        if driver:
            result = subprocess.run(
                [
                    driver,
                    "--query-gpu=uuid,driver_version,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                f"{uuid}@{version}": (int(total) * 2**20, int(used) * 2**20)
                for uuid, version, total, used in (
                    tuple(value.strip() for value in line.split(","))
                    for line in result.stdout.splitlines()
                )
            }
    except (OSError, ValueError, subprocess.SubprocessError):
        # The runner may have an AMD GPU with NVIDIA tooling still installed.
        pass
    devices = Path("/sys/class/drm").glob("card[0-9]*/device")
    return {
        adapter.resolve().name: (
            int(total.read_text()),
            int((adapter / "mem_info_vram_used").read_text()),
        )
        for adapter in devices
        if (total := adapter / "mem_info_vram_total").is_file()
    }


def hardware():
    host = os.uname()
    return {
        "host": host.nodename,
        "kernel": host.release,
        "gpu_bytes": {name: total for name, (total, _) in gpu_memory().items()},
    }


def prove_resources(args, device):
    if args.profile != "4b":
        return
    if not args.device.startswith("discrete:") or not device["gpu_bytes"]:
        raise ValueError(
            "4B qualification needs a discrete WGPU adapter with readable VRAM telemetry"
        )
    if args.stage == "generator":
        state_bytes = GENERATOR_PARAMETERS * 16
        if max(device["gpu_bytes"].values()) < state_bytes:
            raise ValueError(
                f"4B FP32 optimizer needs at least {state_bytes} bytes of VRAM before activations"
            )
        checkpoint_bytes = GENERATOR_PARAMETERS * 12
        if shutil.disk_usage(args.out).free < 2 * checkpoint_bytes:
            raise ValueError(
                f"4B qualification needs room for two checkpoint writes ({2 * checkpoint_bytes} tensor bytes)"
            )


def fit(args, *, data=None, qualifying=False):
    data = data or validate(args.input)
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
    stop_at = args.stop_at if args.stop_at is not None else args.steps
    if not 1 <= stop_at <= args.steps:
        raise ValueError("stop-at must be within the planned schedule")
    default_windows = 8 if small else 128
    validation_windows = (
        default_windows if args.validation_windows is None else args.validation_windows
    )
    if validation_windows < 1:
        raise ValueError("Validation windows must be positive")
    if args.generator_lr_schedule != "cosine" and args.stage != "generator":
        raise ValueError("The generator learning-rate schedule only applies to the generator stage")
    planned_records = args.steps * args.batch_size
    if not small and planned_records < data["records"]["train"]:
        raise ValueError(
            "The 4B run must schedule at least one pass over every packed training record"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    directory = args.out / args.stage
    options = [args.binary, "ncm"]
    if args.backend == "wgpu":
        options += ["--backend", "wgpu", "--device", args.device]
    codec = args.stage == "codec"
    kind = "ncm-tokenizer" if codec else "ncm-generator"
    previous_step, resume = checkpoint(directory, kind)
    if previous_step > stop_at:
        raise ValueError("Existing checkpoint is past the requested stop step")
    immutable = {
        "data": data,
        "profile": args.profile,
        "backend": args.backend,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "checkpoint_every": args.checkpoint_every,
        "device": args.device,
        "stage": args.stage,
        "binary_sha256": digest(args.binary),
    }
    if args.generator_lr_schedule != "cosine":
        immutable["generator_lr_schedule"] = args.generator_lr_schedule
    if args.seed != 42:
        immutable["seed"] = args.seed
    if validation_windows != default_windows:
        immutable["validation_windows"] = validation_windows
    codec_checkpoint = None
    if not codec:
        codec_step, codec_checkpoint = checkpoint(args.out / "codec", "ncm-tokenizer")
        codec_recipe = args.out / "codec-recipe.json"
        if (
            codec_checkpoint is None
            or not codec_recipe.is_file()
            or codec_step != read_json(codec_recipe)["steps"]
        ):
            raise ValueError("Finish the matching codec stage before dispatching the generator")
        if not small:
            immutable["codec_sha256"] = verify_checkpoint(
                codec_checkpoint,
                "ncm-tokenizer",
                codec_step,
                optimizer=False,
            )["model"]["sha256"]
    if not small and not qualifying:
        proof = args.out / f"{args.stage}-qualification.json"
        if not proof.is_file():
            raise ValueError(f"Qualify the {args.stage} on this worker before starting a 4B fit")
        qualification = read_json(proof)
        if (
            qualification.get("recipe") != immutable
            or qualification.get("hardware") != hardware()
            or qualification.get("resume_verified") is not True
        ):
            raise ValueError("The qualification belongs to a different recipe or worker")
    recipe = args.out / f"{args.stage}-recipe.json"
    if recipe.exists() and read_json(recipe) != immutable:
        raise ValueError("The run directory belongs to a different recipe")
    write_json(recipe, immutable)
    if previous_step == stop_at:
        verify_checkpoint(resume, kind, stop_at)
        return {"stage": args.stage, "step": previous_step, "checkpoint": str(resume)}
    if codec:
        config = "paper-owt"
        if small:
            config = args.out / "codec.json"
            write_json(
                config,
                {
                    "vocab": 259,
                    "latent_dim": 32,
                    "codebook_size": 32,
                    "schedule": {"lengths": [2, 4, 8]},
                    "encoder_blocks": 0,
                    "decoder_blocks": 1,
                    "heads": 4,
                    "ema_decay": 0.95,
                    "commitment_beta": 0.25,
                    "nested_dropout": 0.5,
                    "dead_code_threshold": 0.03,
                },
            )
        options += ["tokenizer", "train", "--config", config, "--ordered-records"]
        stream = "codec"
    else:
        config = args.out / "generator.json"
        shape = (
            {"blocks": 2, "hidden": 64, "heads": 4, "thinking_steps": 2, "attractor": True}
            if small
            else FOUR_BILLION
        )
        write_json(config, {**shape, "prompt_len": data["prompt_width"]})
        warmup = min(100, args.steps)
        options += ["train"]
        if args.generator_lr_schedule == "wsd":
            warmup = min(100, args.steps // 20)
            options += ["--lr-schedule", "wsd"]
        options += [
            "--config",
            config,
            "--tokenizer",
            codec_checkpoint,
            "--aligned-windows",
            "--ordered-records",
            "--warmup",
            warmup,
        ]
        stream = "aligned"
    options += [
        "--corpus",
        args.input / f"train.{stream}.bin",
        "--val-corpus",
        args.input / f"validation.{stream}.bin",
        "--steps",
        args.steps,
        "--batch-size",
        args.batch_size,
        "--out",
        directory,
        "--seed",
        args.seed,
        "--checkpoint-every",
        args.checkpoint_every,
        "--val-windows",
        validation_windows,
    ]
    if stop_at != args.steps:
        options += ["--stop-at", stop_at]
    if resume is not None:
        options += ["--resume", resume]
    started = perf_counter()
    subprocess.run([str(arg) for arg in options], check=True)
    seconds = perf_counter() - started
    step, final = checkpoint(directory, kind)
    if step != stop_at:
        raise RuntimeError(f"Expected a reloadable checkpoint at step {stop_at}, got {step}")
    verify_checkpoint(final, kind, step)
    full_passes, records_in_next_pass = divmod(step * args.batch_size, data["records"]["train"])
    return {
        "stage": args.stage,
        "step": step,
        "resumed_from": previous_step,
        "checkpoint": str(final),
        "wall_seconds": seconds,
        "full_passes": full_passes,
        "records_in_next_pass": records_in_next_pass,
    }


def target_tokens(bundle, data, first, last):
    records = data["records"]["train"]
    width = data["target_width"]
    count = 0
    with (bundle / "train.codec.bin").open("rb") as stream:
        for record in range(first, last):
            stream.seek((record % records) * width * 2)
            tokens = array("H")
            tokens.fromfile(stream, width)
            if sys.byteorder != "little":
                tokens.byteswap()
            count += width - tokens.count(258)
    return count


def qualify(args):
    data = validate(args.input)
    device = (
        hardware() if args.backend == "wgpu" else {"host": os.uname().nodename, "gpu_bytes": {}}
    )
    prove_resources(args, device)
    kind = "ncm-tokenizer" if args.stage == "codec" else "ncm-generator"
    start, _ = checkpoint(args.out / args.stage, kind)
    end = start + args.probe_steps
    if args.probe_steps < 2 or end > args.steps:
        raise ValueError(
            "Qualification needs at least two remaining optimizer steps in the planned schedule"
        )
    high_water = {}
    telemetry_errors = []
    stopped = Event()

    def sample():
        try:
            while not stopped.is_set():
                readings = gpu_memory()
                if set(readings) != set(device["gpu_bytes"]):
                    raise ValueError("the visible GPU inventory changed during qualification")
                for name, (_, used) in readings.items():
                    high_water[name] = max(used, high_water.get(name, 0))
                stopped.wait(1.0)
        except (OSError, ValueError) as error:
            telemetry_errors.append(str(error))

    sampler = Thread(target=sample) if args.backend == "wgpu" else None
    if sampler:
        sampler.start()
    try:
        first = fit(
            argparse.Namespace(**{**vars(args), "stop_at": start + 1}), data=data, qualifying=True
        )
        second = fit(
            argparse.Namespace(**{**vars(args), "stop_at": end}), data=data, qualifying=True
        )
    finally:
        stopped.set()
        if sampler:
            sampler.join()
    if second.get("resumed_from") != first["step"]:
        raise RuntimeError("Qualification did not reload the first optimizer checkpoint")
    if args.backend == "wgpu" and (telemetry_errors or not high_water):
        raise RuntimeError(
            f"GPU framebuffer sampling incomplete: {telemetry_errors or 'no sample'}"
        )
    records = (end - start) * args.batch_size
    tokens = target_tokens(args.input, data, start * args.batch_size, end * args.batch_size)
    wall = first["wall_seconds"] + second["wall_seconds"]
    report = {
        "recipe": read_json(args.out / f"{args.stage}-recipe.json"),
        "hardware": device,
        "resume_verified": True,
        "invocations": [first, second],
        "non_padding_target_tokens": tokens,
        "records": records,
        "wall_seconds": wall,
        "end_to_end_target_tokens_per_second": tokens / wall,
        "end_to_end_records_per_second": records / wall,
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
        "gpu_peak_sampled_framebuffer_bytes": high_water,
    }
    write_json(args.out / f"{args.stage}-qualification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    geometry = commands.add_parser("geometry-fixture")
    geometry.add_argument("scenes", type=Path)
    geometry.add_argument("out", type=Path)
    checking = commands.add_parser("validate")
    checking.add_argument("input", type=Path)
    training_args = argparse.ArgumentParser(add_help=False)
    job = training_args
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
    job.add_argument("--validation-windows", type=int)
    fit_command = commands.add_parser("fit", parents=[training_args])
    fit_command.add_argument("--stop-at", type=int)
    qualification = commands.add_parser("qualify", parents=[training_args])
    qualification.add_argument("--probe-steps", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.action == "geometry-fixture":
            geometry_fixture(args.scenes, args.out)
        elif args.action == "validate":
            print(json.dumps(validate(args.input), indent=2))
        else:
            with run_lock(args.out):
                result = qualify(args) if args.action == "qualify" else fit(args)
            print(json.dumps(result))
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
