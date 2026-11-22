"""Score a native Nina checkpoint against a held-out release suite and policy."""

import argparse
import json
import math
import struct
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from nina_worker import checkpoint, digest, read_json, validate, verify_checkpoint, write_json


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def native_prompt(question):
    return f"Question:\n{question}\nAnswer:\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_cases(path):
    cases, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            require(isinstance(case, dict), f"{path.name}:{number} needs a JSON object")
            required = ("id", "field", "prompt", "source")
            require(
                all(isinstance(case.get(key), str) and case[key].strip() for key in required),
                f"{path.name}:{number} needs {', '.join(required)}",
            )
            require(case["id"] not in seen, f"Repeated release case ID {case['id']}")
            seen.add(case["id"])
            answers = case.get("answers", [case.get("answer")])
            require(
                isinstance(answers, list)
                and answers
                and all(isinstance(answer, str) and text_key(answer) for answer in answers),
                f"Case {case['id']} needs one or more nonempty accepted answers",
            )
            case["answers"] = answers
            if "context" in case:
                context = case["context"]
                require(
                    isinstance(context, dict)
                    and all(
                        isinstance(context.get(name), str) and context[name].strip()
                        for name in ("relevant", "distractor")
                    ),
                    f"Case {case['id']} needs both relevant and distractor context documents",
                )
            cases.append(case)
    require(cases, "The release suite has no cases")
    return cases


def check_held_out(cases, corpus, data):
    manifest = read_json(corpus / "manifest.json")
    case_ids = {case["id"] for case in cases}
    prompts = {text_key(native_prompt(case["prompt"])): case["id"] for case in cases}
    width = data["prompt_width"]
    frame = struct.Struct(f"<{width}H")
    target_bytes = data["target_width"] * 2
    for split, count in data["records"].items():
        overlap = case_ids.intersection(manifest.get("packed_ids", {}).get(split, []))
        require(not overlap, f"Release case IDs appear in packed {split}: {sorted(overlap)[:5]}")
        with (corpus / f"{split}.aligned.bin").open("rb") as stream:
            for _ in range(count):
                tokens = frame.unpack(stream.read(frame.size))
                stream.seek(target_bytes, 1)
                content = bytes(token for token in tokens if token < 256)
                # Some diagnostic corpora encode non-text features. Match only
                # complete UTF-8 instructions against the held-out text suite.
                try:
                    prompt = text_key(content.decode("utf-8"))
                except UnicodeDecodeError:
                    continue
                require(
                    prompt not in prompts,
                    f"Release prompt {prompts.get(prompt)} appears in packed {split}",
                )


def completed_stage(run, data, binary_hash, stage):
    kind = "ncm-tokenizer" if stage == "codec" else "ncm-generator"
    recipe = read_json(run / f"{stage}-recipe.json")
    proof = read_json(run / f"{stage}-qualification.json")
    require(
        recipe["data"] == data
        and recipe["binary_sha256"] == binary_hash
        and recipe["profile"] == "4b"
        and recipe["backend"] == "wgpu",
        f"The {stage} recipe does not match this 4B corpus and runtime",
    )
    require(
        proof["recipe"] == recipe
        and proof["resume_verified"] is True
        and proof["gpu_peak_sampled_framebuffer_bytes"],
        f"The {stage} GPU qualification does not match its recipe",
    )
    step, path = checkpoint(run / stage, kind)
    require(path is not None and step == recipe["steps"], f"The {stage} stage is incomplete")
    return path, verify_checkpoint(path, kind, step)


def candidate(args, data, binary_hash):
    if args.run:
        tokenizer, codec = completed_stage(args.run, data, binary_hash, "codec")
        generator, model = completed_stage(args.run, data, binary_hash, "generator")
    else:
        require(args.tokenizer is not None, "--generator requires --tokenizer")
        tokenizer, generator = args.tokenizer, args.generator
        codec = read_json(tokenizer.with_suffix(".state") / "state.json")
        model = read_json(generator.with_suffix(".state") / "state.json")
        codec = verify_checkpoint(tokenizer, "ncm-tokenizer", codec["step"], optimizer=False)
        model = verify_checkpoint(generator, "ncm-generator", model["step"], optimizer=False)
    require(
        model["extras"]["tokenizer_sha256"] == codec["model"]["sha256"],
        "Generator was trained against a different codec",
    )
    spec = model["config"]["model"]
    require(
        spec["prompt_len"] == data["prompt_width"] and spec["thinking_steps"] == 2,
        "Candidate must use the evaluated prefix width and two latent updates",
    )
    latest = [row for row in codec["extras"]["evals"] if row["step"] == codec["step"]]
    require(
        latest
        and latest[-1]["sets"]
        and all(row["windows"] > 0 and row["levels"] for row in latest[-1]["sets"]),
        "Codec checkpoint lacks a held-out reconstruction evaluation",
    )
    return tokenizer, generator, codec, model, latest[-1]


def valid_fraction(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1


def check_policy(policy, cases):
    domains = policy["fields"]
    require(isinstance(domains, dict) and domains, "Policy must name the evaluated fields")
    require(
        valid_fraction(policy["min_codec_token_accuracy"]), "Invalid codec reconstruction threshold"
    )
    counts = defaultdict(int)
    for case in cases:
        counts[case["field"]] += 1
    require(set(counts).issubset(domains), "A release field is missing from the policy")
    for field, rule in domains.items():
        require(
            type(rule["min_cases"]) is int
            and rule["min_cases"] > 0
            and valid_fraction(rule["min_accuracy"]),
            f"Invalid release policy for {field}",
        )
        require(counts[field] >= rule["min_cases"], f"Too few release cases for {field}")
    if "context" in policy:
        context = policy["context"]
        require(
            type(context["min_cases"]) is int
            and context["min_cases"] > 0
            and all(
                valid_fraction(context[f"min_{name}_accuracy"])
                for name in ("relevant", "distractor")
            ),
            "Invalid external-context policy",
        )
        require(
            sum("context" in case for case in cases) >= context["min_cases"],
            "Too few cases for the external-context policy",
        )


def command(argv, timeout, prompt=None):
    process = subprocess.run(argv, input=prompt, text=True, capture_output=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(
            f"{Path(argv[0]).name} exited {process.returncode}: {process.stderr.strip()[-900:]}"
        )
    return process.stdout


def score_response(report, answers, seconds, *, expected_depth):
    text = report["text"]
    levels, trunk = report["level_calls"], report["trunk_evaluations"]
    require(
        isinstance(text, list) and len(text) == 1 and isinstance(text[0], str),
        "Invalid generation text",
    )
    require(
        type(levels) is int and levels > 0 and trunk == levels * expected_depth,
        "Generation reported an unexpected number of latent updates",
    )
    answer = text[0]
    return {
        "answer": answer,
        "correct": text_key(answer) in answers,
        "nonempty": bool(text_key(answer)),
        "wall_seconds": seconds,
        "level_calls": levels,
        "trunk_evaluations": trunk,
    }


def evaluate(args, cases, tokenizer, generator, width):
    paths = ["--tokenizer", tokenizer, "--generator", generator]
    selector = ["--backend", args.backend]
    if args.backend == "wgpu":
        selector += ["--device", args.device]
    native = [
        args.binary,
        "ncm",
        *selector,
        "generate",
        *paths,
        "--prompt-stdin",
        "--json",
        "--temperature",
        "0",
        "--token-temperature",
        "0",
    ]
    has_context = any("context" in case for case in cases)
    require(not has_context or (args.omni and args.dcr), "Context cases need both --omni and --dcr")
    for case in cases:
        prompt = native_prompt(case["prompt"])
        require(
            len(prompt.encode()) <= width, f"Release case {case['id']} exceeds the native prefix"
        )
        answers = {text_key(answer) for answer in case["answers"]}
        result = {"id": case["id"], "field": case["field"], "source": case["source"]}
        for depth in (1, 2):
            started = perf_counter()
            report = json.loads(
                command([*native, "--thinking-steps", str(depth)], args.timeout, prompt)
            )
            require(
                report["prompt_bytes"] == len(prompt.encode()),
                "Native runtime consumed a different prompt length",
            )
            result[f"depth_{depth}"] = score_response(
                report, answers, perf_counter() - started, expected_depth=depth
            )
        if "context" in case:
            for name in ("relevant", "distractor"):
                with TemporaryDirectory(prefix="nina-release-") as temporary:
                    document, store = Path(temporary) / "evidence.txt", Path(temporary) / "store"
                    document.write_text(case["context"][name] + "\n", encoding="utf-8")
                    command([args.dcr, "ingest", "--store", store, document], args.timeout)
                    harness = [
                        args.omni,
                        "model",
                        "nina",
                        "--binary",
                        args.binary,
                        *selector,
                        *paths,
                        "--dcr",
                        args.dcr,
                        "--context-store",
                        store,
                        "--context-budget",
                        "120",
                    ]
                    started = perf_counter()
                    report = json.loads(command(harness, args.timeout, case["prompt"]))
                    used = report["context"]["used_ids"]
                    require(used, f"Case {case['id']} could not fit its {name} context")
                    scores = score_response(
                        report["generation"], answers, perf_counter() - started, expected_depth=2
                    )
                    result[name] = {**scores, "evidence_ids": used}
        yield result


def score_summary(rows, key):
    scores = [row[key] for row in rows if key in row]
    total = len(scores)
    correct = sum(row["correct"] for row in scores)
    return {
        "correct": correct,
        "cases": total,
        "accuracy": correct / total if total else None,
        "nonempty": sum(row["nonempty"] for row in scores),
    }


def report_results(rows, validation, policy):
    failures, fields = [], {}
    accuracy = min(row["levels"][-1]["token_accuracy"] for row in validation["sets"])
    if not math.isfinite(accuracy) or accuracy < policy["min_codec_token_accuracy"]:
        failures.append(f"Codec reconstruction accuracy {accuracy:.5f} missed policy")
    for field, rule in policy["fields"].items():
        selected = [row for row in rows if row["field"] == field]
        summary = {key: score_summary(selected, key) for key in ("depth_1", "depth_2")}
        fields[field] = summary
        if summary["depth_2"]["accuracy"] < rule["min_accuracy"]:
            failures.append(f"{field} two-update answer accuracy missed policy")
    context = {name: score_summary(rows, name) for name in ("relevant", "distractor")}
    if "context" in policy:
        for name, stats in context.items():
            if stats["accuracy"] < policy["context"][f"min_{name}_accuracy"]:
                failures.append(f"{name} external-context answer accuracy missed policy")
    return fields, context, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--run", type=Path, help="Completed and qualified 4B worker run")
    model.add_argument("--generator", type=Path, help="Standalone checkpoint for an evaluation")
    parser.add_argument("--tokenizer", type=Path)
    for name in ("binary", "corpus", "cases", "policy", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "wgpu"), default="cpu")
    parser.add_argument("--device", default="discrete:0")
    parser.add_argument("--omni", type=Path)
    parser.add_argument("--dcr", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    try:
        require(
            args.timeout > 0 and not (args.run and args.tokenizer),
            "Invalid timeout or checkpoint selection",
        )
        inputs = (args.binary, args.cases, args.policy, args.generator, args.tokenizer)
        require(
            all(path is None or args.out.resolve() != path.resolve() for path in inputs),
            "Release report must use a separate output path",
        )
        data = validate(args.corpus)
        cases, policy = load_cases(args.cases), read_json(args.policy)
        if args.run:
            trained_fields = read_json(args.corpus / "manifest.json")["packed_by_split_and_field"][
                "train"
            ]
            require(
                set(trained_fields).issubset(policy["fields"]),
                "The release policy must cover each trained field",
            )
            require("context" in policy, "The release policy must cover external context")
        check_policy(policy, cases)
        check_held_out(cases, args.corpus, data)
        runtime_hash = digest(args.binary)
        tokenizer, generator, codec, model_state, validation = candidate(args, data, runtime_hash)
        rows = list(evaluate(args, cases, tokenizer, generator, data["prompt_width"]))
        fields, context, failures = report_results(rows, validation, policy)
        report = {
            "version": 1,
            "policy_passed": not failures,
            "release_eligible": bool(args.run) and not failures,
            "runtime_sha256": runtime_hash,
            "corpus": data,
            "backend": args.backend,
            "codec_sha256": codec["model"]["sha256"],
            "generator_sha256": model_state["model"]["sha256"],
            "cases_sha256": digest(args.cases),
            "policy_sha256": digest(args.policy),
            "codec_reconstruction": validation,
            "fields": fields,
            "context": context,
            "failures": failures,
            "predictions": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
        print(
            json.dumps(
                {key: report[key] for key in ("policy_passed", "release_eligible", "failures")}
            )
        )
        return int(bool(failures))
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
