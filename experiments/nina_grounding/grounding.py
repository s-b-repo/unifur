"""Build and score paired evidence-following examples for the native Nina pilot."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from random import Random
import shutil
from string import ascii_uppercase
import subprocess
import tempfile

from pilot import DEFAULT_BINARY, canonical, prediction, run_train, sha256, write_json
from records import EOS, PAD, bounded, digest, pack, read_jsonl, write_jsonl


WIDTH = 256
GROUNDING_ANSWER_WIDTH = 16
UNKNOWN = "Unknown."
SEED = 20261122
FAMILIES = (
    {
        "kind": "pickup",
        "question": "Where can {name} collect the repaired shoes?",
        "paraphrase": "Which place should {name} visit for the mended footwear?",
        "source": "The cobbler will hand {name} the boots at {answer}.",
        "decoy": "The repair invoice for {name} was paid at Market Square.",
        "known": ("Pine Lane", "Oak Road", "Elm Court", "Hill Street"),
        "novel": ("Rówan Street", "Wren Gate", "Maple Walk", "Briar Row"),
    },
    {
        "kind": "parcel",
        "question": "What is the parcel code for {name}?",
        "paraphrase": "Which identifier should {name} quote to claim the package?",
        "source": 'Dispatch assigned "{answer}" to the shipment for {name}.',
        "decoy": 'The customer reference for {name} is "CX-08".\nThe package is ready.',
        "known": ("LX-31", "MF-24", "NB-57", "RD-19"),
        "novel": ("KP-62", "ZH-41", "VT-75", "SG-26"),
    },
    {
        "kind": "hours",
        "question": "When does the pool at {name} stop admitting swimmers?",
        "paraphrase": "What is the last entry time for swimming at {name}?",
        "source": "The aquatics centre at {name} closes its admission gate at {answer}.",
        "decoy": "The desk at {name} opens at 06:30. The pool is cleaned each morning.",
        "known": ("17:00", "18:30", "19:00", "20:30"),
        "novel": ("16:45", "17:15", "19:30", "20:45"),
    },
)


def rows_for(split, groups, answer_width):
    rng = Random(SEED + groups)
    rows = []
    for number in range(groups):
        family = FAMILIES[number % len(FAMILIES)]
        name = f"{split.title()}-{number:03}"
        group = f"{family['kind']}/{name}"
        question = family["paraphrase" if split == "test" else "question"].format(name=name)
        decoy = family["decoy"].format(name=name)
        pool = "novel" if split == "test" and number % 2 else "known"
        first, second = rng.sample(family[pool], 2)
        decoy_first = rng.choice((False, True))
        for variant, answer in (("evidence", first), ("swapped", second), ("distractor", UNKNOWN), ("absent", UNKNOWN)):
            if variant == "absent":
                evidence = []
            elif variant == "distractor":
                evidence = [decoy]
            else:
                source = family["source"].format(name=name, answer=answer)
                evidence = [decoy, source] if decoy_first else [source, decoy]
            row = {
                "group": group, "variant": variant, "domain": family["kind"],
                "field": "general", "source": "synthetic_grounding", "prompt": question,
                "answer": answer, "evidence": evidence, "novel_value": pool == "novel",
            }
            row["id"] = digest(canonical(row))
            if not bounded(row, WIDTH, answer_width):
                raise ValueError(f"Prompt or response exceeds the native width for {group}/{variant}")
            rows.append(row)
    rng.shuffle(rows)
    return rows


def codec_targets(rows, answer_width, augment):
    rng = Random(SEED)
    unique = list({row["answer"]: row for row in rows}.values())
    augmented = []
    if augment:
        reserved = {value for family in FAMILIES for key in ("known", "novel") for value in family[key]}
        streets = ("Cédar", "Lóren", "Ash", "Viola", "Aster", "Grain", "Élan", "Thorn", "North", "Alba")
        suffixes = ("Road", "Way", "Court", "Lane", "Row", "Street", "Gate", "Walk")
        values = [f"{street} {suffix}" for street in streets for suffix in suffixes]
        values.extend(f"{hour:02}:{minute:02}" for hour in range(8, 23) for minute in range(0, 60, 5))
        values.extend(f"{left}{right}-{number:02}" for left in ascii_uppercase for right in ascii_uppercase
                      for number in (12, 28, 35, 49, 53, 67, 70, 86, 94))
        # Keep the synthetic shapes similarly represented even when their
        # possible vocabularies differ in size.
        types = ([value for value in values if separator in value and value not in reserved
                  and len(value.encode()) < answer_width] for separator in (" ", ":", "-"))
        for pool in types:
            if not pool:
                raise ValueError(f"Cannot augment codec targets at width {answer_width}")
            augmented.extend({"answer": value} for value in rng.sample(pool, min(len(pool), len(rows) // 6)))
    count = len(rows) - len(augmented)
    packed = [unique[index % len(unique)] for index in range(count)] + augmented
    rng.shuffle(packed)
    return packed


def prepare(out, answer_width, balance_codec, augment_codec):
    out.mkdir(mode=0o700, parents=True)
    sizes = {"train": 96, "validation": 12, "test": 18}
    manifest = {
        "seed": SEED, "prompt_width_bytes": WIDTH, "response_width_bytes_including_eos": answer_width,
        "groups_by_split": sizes, "packed_ids": {}, "corpus_hashes": {},
        "balance_codec": balance_codec, "augment_codec": augment_codec,
        "ordered_codec_records": True, "tokenizer_steps": 160,
    }
    for split, groups in sizes.items():
        rows = rows_for(split, groups, answer_width)
        manifest["packed_ids"][split] = [row["id"] for row in rows]
        write_jsonl(out / f"{split}.jsonl", rows)
        if split == "validation":
            write_jsonl(out / "pilot-validation.jsonl", rows)
        if split != "test":
            aligned = out / f"{split}.aligned.bin"
            codec = out / f"{split}.codec.bin"
            codec_rows = None
            if split == "train" and balance_codec:
                codec_rows = codec_targets(rows, answer_width, augment_codec)
                manifest["codec_train_counts"] = dict(Counter(row["answer"] for row in codec_rows))
            pack(aligned, codec, rows, WIDTH, answer_width, codec_rows)
            for path in (aligned, codec):
                manifest["corpus_hashes"][path.name] = sha256(path)
        manifest["corpus_hashes"][f"{split}.jsonl"] = sha256(out / f"{split}.jsonl")
    write_json(out / "manifest.json", manifest)


def roundtrip(binary, tokenizer, cases, env, answer_width):
    tokens = []
    for case in cases:
        target = list(case["answer"].encode()) + [EOS]
        if len(target) > answer_width:
            raise ValueError(f"Response {case['id']} needs {len(target)} bytes including EOS")
        tokens.append(target + [PAD] * (answer_width - len(target)))
    with tempfile.TemporaryDirectory(prefix="nina-codec-") as directory:
        source = Path(directory) / "targets.json"
        dest = Path(directory) / "reconstructed.json"
        write_json(source, tokens)
        subprocess.run(
            [binary, "ncm", "tokenizer", "reconstruct", "--checkpoint", tokenizer, "--token-ids", source, "--out", dest],
            env=env, check=True, timeout=180,
        )
        reconstructed = json.loads(dest.read_text())
    if len(reconstructed) != len(tokens) or any(len(row) != answer_width for row in reconstructed):
        raise ValueError("Native codec reconstruction changed the number or width of responses")
    result = {}
    for case, expected, actual in zip(cases, tokens, reconstructed):
        # A correct answer ends at EOS. Padding after that boundary is irrelevant.
        stop = expected.index(EOS) + 1
        result[case["id"]] = actual[:stop] == expected[:stop]
    return result


def evaluate(cases_path, binary, tokenizer, generator, codec_binary, output, answer_width):
    cases = list(read_jsonl(cases_path))
    if not cases:
        raise ValueError("The held-out case file is empty")
    env = {**os.environ, "RAYON_NUM_THREADS": "2", "MATMUL_NUM_THREADS": "2"}
    oracle = roundtrip(codec_binary, tokenizer, cases, env, answer_width) if codec_binary else {}

    def measure(case):
        predicted, _ = prediction([binary, "ncm"], tokenizer, generator, case["prompt"], env, case["evidence"])
        return {**case, "output": predicted, "exact": predicted == case["answer"], "codec_exact": oracle.get(case["id"])}

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(measure, cases))
    report = {
        "cases_sha256": sha256(cases_path), "binary_sha256": sha256(binary),
        "tokenizer_sha256": sha256(tokenizer), "generator_sha256": sha256(generator),
        "codec_binary_sha256": sha256(codec_binary) if codec_binary else None,
        **summarize(rows), "predictions": rows,
    }
    write_json(output, report)
    print(json.dumps({key: report[key] for key in ("matched_pairs", "by_condition")}, indent=2))
    return report


def evaluate_codec(cases_path, binary, tokenizer, output, answer_width):
    cases = list(read_jsonl(cases_path))
    env = {**os.environ, "RAYON_NUM_THREADS": "2", "MATMUL_NUM_THREADS": "2"}
    exact = roundtrip(binary, tokenizer, cases, env, answer_width)
    rows = [
        {**row, "output": row["answer"] if exact[row["id"]] else "",
         "exact": exact[row["id"]], "codec_exact": exact[row["id"]]}
        for row in cases
    ]
    report = {
        "cases_sha256": sha256(cases_path), "codec_binary_sha256": sha256(binary),
        "tokenizer_sha256": sha256(tokenizer), "answer_width": answer_width,
        "exact": sum(exact.values()), "total": len(rows), **summarize(rows),
    }
    write_json(output, report)
    print(f"{report['exact']}/{report['total']} exact codec reconstructions")


def fit_check(corpus, binary, tokenizer, out, adapter_rank):
    manifest = json.loads((corpus / "manifest.json").read_text())
    train = corpus / "train.jsonl"
    if sha256(train) != manifest["corpus_hashes"][train.name]:
        raise ValueError("The training cases differ from the manifest")
    answer_width = manifest["response_width_bytes_including_eos"]
    prompt_width = manifest["prompt_width_bytes"]
    groups = defaultdict(list)
    for row in read_jsonl(train):
        if row["domain"] == "parcel":
            groups[row["group"]].append(row)
    examples = [rows for _, rows in sorted(groups.items()) if len(rows) == 4][:4]
    if not examples:
        raise ValueError("No complete contrast groups in the training corpus")
    env = {**os.environ, "RAYON_NUM_THREADS": "2", "MATMUL_NUM_THREADS": "2"}
    exact = roundtrip(binary, tokenizer, [row for group in examples for row in group], env, answer_width)
    rows = next((group for group in examples if all(exact[row["id"]] for row in group)), None)
    if rows is None:
        raise ValueError("No sampled group survived the codec round trip")
    out.mkdir(mode=0o700, parents=True)
    cases = out / "training-replay.jsonl"
    aligned = out / "train.bin"
    replay = out / "replay.bin"
    write_jsonl(cases, rows)
    pack(aligned, out / "codec.bin", rows, prompt_width, answer_width)
    # The trainer requires different files. This positive control scores
    # training rows intentionally.
    shutil.copyfile(aligned, replay)
    command = [
        binary, "ncm", "train", "--config", "tiny", "--tokenizer", tokenizer,
        "--corpus", aligned, "--val-corpus", replay, "--aligned-records",
        "--prompt-len", str(prompt_width), "--trim-prompt-padding", "--cache-codec-windows", "4",
        "--steps", "320", "--batch-size", "4", "--lr", "0.002", "--warmup", "12",
        "--val-windows", "4", "--log-every", "40", "--out", out / "generator",
    ]
    if adapter_rank is not None:
        command.extend(("--adapter-rank", str(adapter_rank)))
    generator = run_train(command, "generator", out, env)
    report = evaluate(cases, binary, tokenizer, generator, binary, out / "training-replay.json", answer_width)
    if not all(row["exact"] for row in report["predictions"]):
        raise ValueError("The generator failed to fit all four training control cases")


def summarize(rows):
    summary = defaultdict(Counter)
    pairs = defaultdict(list)
    for row in rows:
        result = {"total": 1, "exact": int(row["exact"]), "abstained": int(row["output"] == UNKNOWN)}
        if row.get("codec_exact") is not None:
            result["codec_exact"] = int(row["codec_exact"])
        summary[row["variant"]].update(result)
        if row["variant"] in ("evidence", "swapped"):
            pairs[row["group"]].append(row["exact"])
            key = "novel_answers" if row["novel_value"] else "known_answers"
            summary[key].update(result)
    return {
        "matched_pairs": {"both_correct": sum(len(pair) == 2 and all(pair) for pair in pairs.values()), "total": len(pairs)},
        "by_condition": dict(summary),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--answer-width", type=int, default=GROUNDING_ANSWER_WIDTH)
    build.add_argument("--balance-codec", action=argparse.BooleanOptionalAction, default=True,
                       help="Give each distinct codec target equal exposure (default)")
    build.add_argument("--augment-codec", action=argparse.BooleanOptionalAction,
                       help="Add disjoint synthetic codec values when balancing (default)")
    score = commands.add_parser("evaluate")
    codec = commands.add_parser("oracle")
    for evaluator in (score, codec):
        for flag in ("cases", "tokenizer", "out"):
            evaluator.add_argument(f"--{flag}", type=Path, required=True)
        evaluator.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
        evaluator.add_argument("--answer-width", type=int, help="Override the saved codec response width")
    score.add_argument("--generator", type=Path, required=True)
    score.add_argument("--codec-binary", type=Path, help="Native binary with tokenizer reconstruct")
    fit = commands.add_parser("fit-check", help="Replay a four-case training control with an exact codec")
    for flag in ("corpus", "binary", "tokenizer", "out"):
        fit.add_argument(f"--{flag}", type=Path, required=True)
    fit.add_argument("--adapter-rank", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.augment_codec and not args.balance_codec:
            parser.error("--augment-codec requires --balance-codec")
        augment = args.balance_codec if args.augment_codec is None else args.augment_codec
        prepare(args.out, args.answer_width, args.balance_codec, augment)
    elif args.command == "fit-check":
        fit_check(args.corpus.resolve(), args.binary.resolve(), args.tokenizer.resolve(),
                  args.out.resolve(), args.adapter_rank)
    else:
        answer_width = args.answer_width
        if answer_width is None and (args.command == "oracle" or args.codec_binary):
            state = json.loads((args.tokenizer.with_suffix(".state") / "state.json").read_text())
            answer_width = state["config"]["tokenizer"]["schedule"]["lengths"][-1]
        if args.command == "oracle":
            evaluate_codec(args.cases, args.binary, args.tokenizer, args.out, answer_width)
        else:
            evaluate(args.cases, args.binary, args.tokenizer, args.generator, args.codec_binary, args.out, answer_width)


if __name__ == "__main__":
    main()
