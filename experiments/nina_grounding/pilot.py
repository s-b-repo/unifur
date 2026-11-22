"""Train, reload, and evaluate a small prompt-conditioned native Nina model."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
from time import perf_counter


HERE = Path(__file__).resolve().parent
DEFAULT_BINARY = HERE.parents[1] / "target/release/dblocks"
CHECKPOINT = re.compile(r"^(?:checkpoint|saved)\s*[: ]\s*(.*\.mpk)\s*$", re.IGNORECASE | re.MULTILINE)
TOKENIZER_WIDTHS = {"tiny": 64, "paper-owt": 1024, "nina-4k": 4096}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(row):
    prompt = f"Question:\n{row['prompt']}\nAnswer:\n"
    evidence = [text.strip() for text in row.get("evidence", ()) if text.strip()]
    if evidence:
        lines = "".join(json.dumps(text, ensure_ascii=False) + "\n" for text in evidence)
        return f"Evidence:\n{lines}{prompt}"
    return prompt


def run_train(command, name, output, env):
    log = output / f"{name}.log"
    print(f"Starting {name}. Log {log}", flush=True)
    started = perf_counter()
    with log.open("w") as stream:
        subprocess.run(command, check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)
    transcript = log.read_text()
    written = [Path(path) for path in CHECKPOINT.findall(transcript)]
    if not written:
        written = sorted((output / name).glob("*.mpk"), key=lambda path: path.stat().st_mtime_ns)
    if not written or not written[-1].is_file():
        raise RuntimeError(f"No checkpoint recorded in {log}")
    checkpoint = written[-1]
    print(f"Finished {name} after {perf_counter() - started:.1f}s. {checkpoint.name}", flush=True)
    return checkpoint


def prediction(prefix, tokenizer, generator, question, env, evidence=()):
    prompt = canonical({"prompt": question, "evidence": evidence})
    argv = [
        *prefix, "generate", "--tokenizer", tokenizer, "--generator", generator,
        f"--prompt={prompt}", "--temperature", "0", "--token-temperature", "0",
    ]
    start = perf_counter()
    response = subprocess.run(argv, env=env, check=True, capture_output=True, text=True, timeout=120)
    match = re.search(r"^---\n(.*)\n---\s*\Z", response.stdout, re.DOTALL | re.MULTILINE)
    if match is None:
        raise RuntimeError("Native NCM output is missing the answer block")
    return match[1].strip(), perf_counter() - start


def evaluate(prefix, tokenizer, generator, data, out, env, jobs):
    cases = [json.loads(line) for line in (data / "pilot-validation.jsonl").open()]

    def score(case):
        answer, seconds = prediction(prefix, tokenizer, generator, case["prompt"], env, case.get("evidence", ()))
        normalized = lambda text: " ".join(text.casefold().split())
        return {
            "id": case["id"], "field": case["field"], "source": case["source"],
            "prompt": case["prompt"], "reference": case["answer"], "output": answer,
            "exact": normalized(answer) == normalized(case["answer"]),
            "wall_seconds": seconds,
        }

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        rows = list(executor.map(score, cases))
    total = Counter(row["field"] for row in rows)
    correct = Counter(row["field"] for row in rows if row["exact"])
    report = {
        "stage": "native_instruction_pilot",
        "binary_sha256": sha256(prefix[0]),
        "runtime_arguments": [str(arg) for arg in prefix[1:]],
        "tokenizer": {"path": str(tokenizer), "sha256": sha256(tokenizer)},
        "generator": {"path": str(generator), "sha256": sha256(generator)},
        "corpus_manifest_sha256": sha256(data / "manifest.json"),
        "rows": len(rows),
        "correct": sum(correct.values()),
        "nonempty": sum(bool(row["output"]) for row in rows),
        "unique_outputs": len({row["output"] for row in rows}),
        "by_field": {name: {"correct": correct[name], "total": count} for name, count in sorted(total.items())},
        "median_wall_seconds": median(row["wall_seconds"] for row in rows),
        "predictions": rows,
    }
    write_json(out / "held-out.json", report)
    print(f"Held-out normalized exact match {report['correct']}/{report['rows']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--backend", choices=("cpu", "wgpu"), default="cpu")
    parser.add_argument("--device", help="WGPU adapter, for example discrete:0")
    parser.add_argument("--tokenizer-steps", type=int, help="Optimizer steps, following the corpus manifest by default")
    parser.add_argument("--codec-only", action="store_true", help="Stop after training and recording the codec")
    parser.add_argument("--ordered-codec", action=argparse.BooleanOptionalAction,
                        help="Train on complete response records, following the corpus manifest by default")
    parser.add_argument("--generator-steps", type=int, default=220)
    parser.add_argument("--tokenizer-config", help="Native NCM tokenizer preset or JSON matching the packed response width")
    parser.add_argument("--generator-config", default="tiny", help="Native generator preset or JSON path")
    parser.add_argument("--tokenizer-batch-size", type=int, default=8)
    parser.add_argument("--generator-batch-size", type=int, default=6)
    parser.add_argument("--val-windows", type=int, default=36)
    parser.add_argument("--alpha", type=float, help="Override the generator's concept loss weighting exponent")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    data, out, binary = args.data.resolve(), args.out.resolve(), args.binary.resolve()
    corpus = read_json(data / "manifest.json")
    codec_steps = corpus.get("tokenizer_steps", 320) if args.tokenizer_steps is None else args.tokenizer_steps
    ordered_codec = corpus.get("ordered_codec_records", False) if args.ordered_codec is None else args.ordered_codec
    if min(codec_steps, args.generator_steps, args.threads, args.tokenizer_batch_size,
           args.generator_batch_size, args.val_windows) < 1:
        parser.error("steps, threads, batch sizes and validation windows must be positive")
    if (args.alpha is not None and not 0.0 <= args.alpha <= 1.0) or args.checkpoint_every < 0:
        parser.error("alpha must be between zero and one and checkpoint interval must be nonnegative")
    if args.device and args.backend != "wgpu":
        parser.error("--device requires --backend wgpu")
    prefix = [binary, "ncm"]
    if args.backend == "wgpu":
        prefix += ["--backend", "wgpu", "--device", args.device or "discrete:0"]
    for filename, expected in corpus["corpus_hashes"].items():
        if sha256(data / filename) != expected:
            raise ValueError(f"Unexpected {filename} digest")
    codec = {
        "vocab": 259, "latent_dim": 48, "codebook_size": 192,
        "schedule": {"lengths": [2, 4, 8, 16, 32, 64, 128]},
        "encoder_blocks": 0, "decoder_blocks": 1, "heads": 4,
        "ema_decay": 0.95, "commitment_beta": 0.25, "nested_dropout": 0.4,
        "dead_code_threshold": 0.03, "restart_every": 50,
    }
    response_width = corpus["response_width_bytes_including_eos"]
    if not args.tokenizer_config and response_width in codec["schedule"]["lengths"]:
        codec["schedule"]["lengths"] = [length for length in codec["schedule"]["lengths"] if length <= response_width]
    width = TOKENIZER_WIDTHS.get(args.tokenizer_config)
    if width is None:
        if args.tokenizer_config:
            codec = read_json(Path(args.tokenizer_config))
        width = codec["schedule"]["lengths"][-1]
    if width != response_width:
        parser.error("tokenizer output width must match the packed response width")
    out.mkdir(mode=0o700, parents=True)
    env = {**os.environ, "RAYON_NUM_THREADS": str(args.threads), "MATMUL_NUM_THREADS": str(args.threads)}
    if args.tokenizer_config in TOKENIZER_WIDTHS:
        codec_config = args.tokenizer_config
    else:
        codec_config = out / "tokenizer.json"
        write_json(codec_config, codec)
    tokenizer = run_train([
        *prefix, "tokenizer", "train", "--config", codec_config,
        "--corpus", data / "train.codec.bin", "--val-corpus", data / "validation.codec.bin",
        "--steps", str(codec_steps), "--batch-size", str(args.tokenizer_batch_size),
        "--val-windows", str(args.val_windows), "--log-every", "20", "--out", out / "tokenizer",
        *(["--ordered-records"] if ordered_codec else []),
        *(["--lr", "0.0015"] if not args.tokenizer_config else []),
    ], "tokenizer", out, env)
    if args.codec_only:
        write_json(out / "codec.json", {
            "binary_sha256": sha256(binary),
            "tokenizer": {"path": str(tokenizer), "sha256": sha256(tokenizer)},
            "corpus_manifest_sha256": sha256(data / "manifest.json"),
        })
        return
    trainer_steps = args.generator_steps
    alpha = ["--alpha", str(args.alpha)] if args.alpha is not None else []
    generator = run_train([
        *prefix, "train", "--config", args.generator_config, *alpha,
        "--tokenizer", tokenizer, "--corpus", data / "train.aligned.bin",
        "--val-corpus", data / "validation.aligned.bin", "--aligned-windows",
        "--prompt-len", str(corpus["prompt_width_bytes"]), "--steps", str(trainer_steps),
        "--batch-size", str(args.generator_batch_size), "--lr", "0.0008", "--warmup", str(min(20, trainer_steps - 1)),
        "--val-windows", str(args.val_windows), "--log-every", "20", "--checkpoint-every", str(args.checkpoint_every),
        "--out", out / "generator",
    ], "generator", out, env)
    evaluate(prefix, tokenizer, generator, data, out, env, jobs=1 if args.backend == "wgpu" else 2)


if __name__ == "__main__":
    main()
