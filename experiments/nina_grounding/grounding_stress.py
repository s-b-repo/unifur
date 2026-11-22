"""Compare reader abstention on paraphrased evidence and adversarial SQuAD questions."""

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

from grounding import UNKNOWN
from grounding_reader import Encoder, candidate_answer, load_splits
from pilot import sha256, write_json


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def squad_splits(path, count):
    data = json.loads(path.read_text())
    if data.get("version") != "v2.0":
        raise ValueError("Expected the official SQuAD 2.0 development file")
    by_title = defaultdict(list)
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for case in paragraph["qas"]:
                answers = tuple(normalized(answer["text"]) for answer in case["answers"])
                if case["is_impossible"] or any(answers):
                    by_title[article["title"]].append((case["id"], case["question"], (paragraph["context"],), answers))
    titles = sorted(by_title, key=lambda title: hashlib.sha256(title.encode()).digest())
    border = len(titles) // 3
    for part in (titles[:border], titles[border:]):
        strata = defaultdict(list)
        for title in part:
            for row in by_title[title]:
                strata[bool(row[3])].append(row)
        if any(len(strata[answerable]) < count for answerable in (False, True)):
            raise ValueError("SQuAD document split has too few answerable or unanswerable questions")
        yield [row for rows in strata.values() for row in sorted(
            rows, key=lambda row: hashlib.sha256(row[0].encode()).digest(),
        )[:count]]


def synthetic_rows(rows):
    return [
        (row["id"], row["prompt"], tuple(row["evidence"]),
         (normalized(row["answer"]),) if row["answer"] != UNKNOWN else ())
        for row in rows
    ]


def propose(reader, verifier, rows):
    pairs = list(dict.fromkeys((question, source) for _, question, evidence, _ in rows for source in evidence))
    offers = {}
    for start in range(0, len(pairs), 8):
        batch = pairs[start:start + 8]
        attributed = [("According to the passage, " + q[:1].lower() + q[1:], source) for q, source in batch]
        (first, last), tokens, _ = reader.forward(batch)
        (attr_first, attr_last), attr_tokens, _ = reader.forward(attributed)
        support = verifier.forward(attributed)[0][0].reshape(-1)
        direct = (candidate_answer(token, a, b) for token, a, b in zip(tokens, first, last, strict=True))
        reasked = (candidate_answer(token, a, b) for token, a, b in zip(attr_tokens, attr_first, attr_last, strict=True))
        for pair, original, repeated, score in zip(batch, direct, reasked, support, strict=True):
            if original is None or repeated is None:
                continue
            margin, begin, end = original
            attr_margin, attr_begin, attr_end = repeated
            source = pair[1]
            answer = normalized(source[begin:end])
            agrees = answer == normalized(source[attr_begin:attr_end])
            scores = (margin, float(score), attr_margin, agrees)
            if not all(math.isfinite(value) for value in scores):
                raise ValueError("Reader produced a nonfinite confidence score")
            if answer:
                offers[pair] = (answer, scores)
    return [tuple(offers[question, source] for source in evidence if (question, source) in offers)
            for _, question, evidence, _ in rows]


def count_answers(rows, candidates, thresholds):
    result = dict.fromkeys(("supported", "unsupported", "exact", "wrong_supported", "false_unsupported"), 0)
    for (_, _, _, references), offers in zip(rows, candidates, strict=True):
        result["supported" if references else "unsupported"] += 1
        accepted = [offer for offer in offers if all(
            value >= floor for value, floor in zip(offer[1], thresholds, strict=True)
        )]
        if not accepted:
            continue
        answer, confidence = max(accepted, key=lambda offer: offer[1][0])
        if any(other != answer and scores[0] > confidence[0] - 2 for other, scores in accepted):
            continue
        outcome = "exact" if answer in references else "wrong_supported" if references else "false_unsupported"
        result[outcome] += 1
    return result


def calibrate(synthetic, public):
    unbounded = -math.inf
    grid = itertools.product((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12),
                             (unbounded, -2, 0, 1, 2, 3, 4), (unbounded, 0, 2, 4, 6, 8), (0, 1))
    best = (-1.0, 0, ())
    for floors in grid:
        a = count_answers(*synthetic, floors)
        b = count_answers(*public, floors)
        if (a["wrong_supported"] or a["false_unsupported"] or
                b["false_unsupported"] * 20 > b["unsupported"] or b["wrong_supported"] * 10 > b["supported"]):
            continue
        coverage = a["exact"] / a["supported"] + b["exact"] / b["supported"]
        best = max(best, (coverage, -sum(cut not in (unbounded, 0) for cut in floors), floors))
    return best[2] or (math.inf, unbounded, unbounded, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("corpus", "squad-dev", "model", "out"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--cases-per-class", type=int, default=120)
    args = parser.parse_args()
    if args.cases_per_class < 1:
        parser.error("--cases-per-class must be positive")
    synthetic = load_splits(args.corpus)
    squad_validation, squad_test = squad_splits(args.squad_dev, args.cases_per_class)
    reader = Encoder(args.model, "reader.onnx", "reader-tokenizer.json")
    verifier = Encoder(args.model, "answerability.onnx", "answerability-tokenizer.json")
    splits = {
        "synthetic_validation": synthetic_rows(synthetic["validation"]),
        "squad_validation": squad_validation,
        "synthetic_test": synthetic_rows(synthetic["test"]),
        "squad_test": squad_test,
    }
    fitted = {}
    for name, rows in splits.items():
        fitted[name] = rows, propose(reader, verifier, rows)
        print(f"Scored {len(rows)} {name.replace('_', ' ')} cases", flush=True)
    floors = calibrate(fitted["synthetic_validation"], fitted["squad_validation"])
    baseline = (0, -math.inf, -math.inf, 0)
    report = {
        "squad_dev_sha256": sha256(args.squad_dev),
        "synthetic_split_sha256": {split: sha256(args.corpus / f"{split}.jsonl") for split in ("validation", "test")},
        "model_sha256": {file: sha256(args.model / file) for file in ("reader.onnx", "answerability.onnx")},
        "reader_floors": dict(zip(("margin", "answerability", "attributed_margin", "agreement"),
                                  (cut if math.isfinite(cut) else str(cut) for cut in floors), strict=True)),
        "sets": {name: {"reader_margin": count_answers(*data, baseline),
                        "calibrated": count_answers(*data, floors)} for name, data in fitted.items()},
    }
    write_json(args.out, report)
    print(f"Saved scores to {args.out}")


if __name__ == "__main__":
    main()
