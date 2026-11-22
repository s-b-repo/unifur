"""Compare untuned and tuned native NCM output on short factual questions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess


CASES = [
    ("coding", "In Python, which list method returns a shallow copy of the list?", r"\bcopy\b"),
    ("mathematics", "What is 17 plus 25? Answer with the number.", r"\b42\b"),
    ("science", "Which planet is known as the Red Planet?", r"\bmars\b"),
    ("technology", "In SQL, which clause filters groups after GROUP BY?", r"\bhaving\b"),
    ("technology", "What security principle grants only the permissions necessary for the task?", r"\bleast privilege\b"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dblocks", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def evaluate(case):
        field, prompt, expected = case
        command = [
            str(args.dblocks.resolve()), "ncm", "released", "--model", str(args.model),
            "--field", field, "--level-experts", "--prompt", prompt, "--json",
        ]
        result = {"field": field, "question": prompt, "expected_expression": expected}
        tuned = ["--expert-adapter", str(args.adapter)]
        if args.prior:
            tuned.extend(["--domain-prior", str(args.prior)])
        for stage, flags in (
            ("base", ["--geometry", "0"]),
            ("fused_tuned", tuned),
        ):
            response = json.loads(subprocess.check_output(
                command + flags, env=dict(os.environ, RAYON_NUM_THREADS="2"), text=True, timeout=180,
            ))
            result[stage] = {
                "text": response["text"],
                "expected_answer_mentioned": bool(re.search(expected, response["text"], re.I)),
                "elapsed_ms": response["elapsed_ms"],
                "adapted_experts": response["adapted_experts"],
            }
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(evaluate, CASES))
    counts = {stage: sum(row[stage]["expected_answer_mentioned"] for row in results)
              for stage in ("base", "fused_tuned")}
    with args.out.open("x", encoding="utf-8") as output:
        json.dump({"cases": results, "exact_answer_mentions": counts}, output, indent=2)
        output.write("\n")
    print(json.dumps({"exact_answer_mentions": counts, "cases": len(results)}))


if __name__ == "__main__":
    main()
