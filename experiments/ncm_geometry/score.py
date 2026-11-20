#!/usr/bin/env python3
"""Score greedy NCM continuations on complete held-out integer-answer prompts."""

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


def score(args):
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines()]
    levels = Counter()
    correct = Counter()
    parsed = 0
    predictions = []
    for case in cases:
        result = subprocess.run(
            [str(args.dblocks), "ncm", "generate", "--tokenizer", str(args.tokenizer),
             "--generator", str(args.generator), "--top-k", "1", "--prompt", case["prompt"]],
            capture_output=True, check=True, text=True, timeout=60,
        )
        _, marker, rest = result.stdout.partition("\n---\n")
        if not marker:
            raise ValueError("NCM generation output is missing its completion delimiter")
        first_line = rest.split("\n", 1)[0].strip()
        prediction = int(first_line) if re.fullmatch(r"[+-]?[0-9]+", first_line) else None
        level = case["level"]
        levels[level] += 1
        parsed += prediction is not None
        correct[level] += prediction == case["answer"]
        predictions.append({"id": case["id"], "expected": case["answer"], "prediction": prediction})
    return {
        "questions": len(cases),
        "parsed_integer_answers": parsed,
        "correct": sum(correct.values()),
        "by_level": {level: {"questions": count, "correct": correct[level]} for level, count in levels.items()},
        "predictions": predictions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dblocks", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parsed_args = parser.parse_args()
    report = score(parsed_args)
    parsed_args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Integer answer accuracy {report['correct']}/{report['questions']}; "
          f"{report['parsed_integer_answers']} parseable answers.")
