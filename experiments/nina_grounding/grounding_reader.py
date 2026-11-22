"""Fit answer-or-abstain calibration for the local extractive reader."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from grounding import UNKNOWN, summarize
from pilot import sha256, write_json
from records import read_jsonl


SCRIPTS = Path(os.environ.get("NINA_SUBNEXT_SCRIPTS", Path(__file__).resolve().parents[3] / "subnext/scripts"))
sys.path.insert(0, str(SCRIPTS))
from semantic_encoder import Encoder, candidate_answer  # noqa: E402


def proposals(rows, model_dir):
    reader = Encoder(model_dir, "reader.onnx", "reader-tokenizer.json")
    entailment = Encoder(model_dir, "answerability.onnx", "answerability-tokenizer.json")
    encode = Encoder(model_dir)
    texts = [(row["prompt"], source) for row in rows for source in row["evidence"]]
    unique = list(dict.fromkeys(text for pair in texts for text in pair))
    vectors = dict(zip(unique, encode.encode(unique)))
    candidates = {}
    for offset in range(0, len(texts), 8):
        batch = texts[offset:offset + 8]
        (starts, ends), tokens, _ = reader.forward(batch)
        relevance, _, _ = entailment.forward(batch)
        for pair, token, first, last, support in zip(batch, tokens, starts, ends, relevance[0].reshape(-1)):
            answer = candidate_answer(token, first, last)
            if answer is None:
                continue
            margin, start, end = answer
            question, source = pair
            text = source[start:end].strip(' \r\n\t"“”.,;!?')
            if not text:
                continue
            start = source.index(text, start, end)
            features = (margin, float(support), float(vectors[question] @ vectors[source]))
            candidates[pair] = (text, start, start + len(text), features)
    return [
        [(index, *candidates[pair]) for index, pair in enumerate((row["prompt"], text) for text in row["evidence"])
         if pair in candidates]
        for row in rows
    ]


def fit(rows, candidates):
    examples = [
        (features, row["answer"] != UNKNOWN and answer == row["answer"])
        for row, spans in zip(rows, candidates)
        for _, answer, _, _, features in spans
    ]
    features, labels = (np.asarray(values, dtype=float) for values in zip(*examples))
    if len(set(labels)) != 2 or not np.isfinite(features).all():
        raise ValueError("Calibration needs finite features with both supported and unsupported examples")
    mean = features.mean(axis=0)
    scale = features.std(axis=0).clip(0.01)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    weights = np.zeros(design.shape[1])
    # Regularized logistic loss. The intercept is independent of weight decay.
    for _ in range(400):
        probability = 1 / (1 + np.exp(-(design @ weights).clip(-30, 30)))
        gradient = design.T @ (probability - labels) / len(labels)
        gradient[1:] += 0.05 * weights[1:]
        weights -= 0.3 * gradient
    return weights, mean, scale


def score(candidates, fit_result=None):
    if fit_result is None:
        confidence = lambda features: features[0]
    else:
        weights, mean, scale = fit_result
        confidence = lambda features: float(weights[0] + weights[1:] @ ((features - mean) / scale))
    return [
        max(((confidence(features), index, text, first, end) for index, text, first, end, features in row),
            key=lambda item: item[0], default=None)
        for row in candidates
    ]


def render(rows, ranked, threshold):
    result = []
    for row, hit in zip(rows, ranked):
        if hit is not None and hit[0] >= threshold:
            confidence, index, text, start, end = hit
            source = row["evidence"][index]
            assert source[start:end] == text
            provenance = {
                "passage": index, "from_byte": len(source[:start].encode()),
                "to_byte": len(source[:end].encode()), "score": confidence,
            }
        else:
            text, provenance = UNKNOWN, None
        result.append({**row, "output": text, "exact": text == row["answer"], "provenance": provenance})
    return result


def choose_threshold(rows, ranked):
    trial = sorted({hit[0] for hit in ranked if hit is not None})
    trial.extend((float("-inf"), float("inf")))

    def value(threshold):
        return sum(
            1 if row["exact"] else 0 if row["output"] == UNKNOWN else -2
            for row in render(rows, ranked, threshold)
        )

    return max(trial, key=lambda threshold: (value(threshold), threshold))


def load_splits(corpus):
    manifest = json.loads((corpus / "manifest.json").read_text())
    splits = {}
    seen = set()
    for split in ("train", "validation", "test"):
        file = corpus / f"{split}.jsonl"
        if sha256(file) != manifest["corpus_hashes"][file.name]:
            raise ValueError(f"Changed corpus split {file}")
        rows = list(read_jsonl(file))
        group = {row["group"] for row in rows}
        if not group or group & seen:
            raise ValueError(f"Missing or overlapping groups in {split}")
        seen.update(group)
        splits[split] = rows
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("corpus", "model", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    splits = load_splits(args.corpus)
    rows = splits["train"] + splits["validation"] + splits["test"]
    train_end = len(splits["train"])
    val_end = train_end + len(splits["validation"])
    candidates = proposals(rows, args.model)
    trained = fit(rows[:train_end], candidates[:train_end])
    threshold = choose_threshold(rows[train_end:val_end], score(candidates[train_end:val_end], trained))
    frozen = render(rows[val_end:], score(candidates[val_end:]), 0)
    calibrated = render(rows[val_end:], score(candidates[val_end:], trained), threshold)
    report = {
        "corpus_manifest_sha256": sha256(args.corpus / "manifest.json"),
        "model_sha256": {path.name: sha256(path) for path in sorted(args.model.glob("*.onnx"))},
        "features": ["reader_margin", "entailment_logit", "embedding_cosine"],
        "calibration": {
            "weights": trained[0].tolist(), "feature_mean": trained[1].tolist(), "feature_scale": trained[2].tolist(),
            "threshold": threshold if np.isfinite(threshold) else ("always abstain" if threshold > 0 else "always answer"),
        },
        "frozen_reader": {**summarize(frozen), "predictions": frozen},
        "calibrated_reader": {**summarize(calibrated), "predictions": calibrated},
    }
    write_json(args.out, report)
    print(json.dumps({name: report[name]["matched_pairs"] for name in ("frozen_reader", "calibrated_reader")}))


if __name__ == "__main__":
    main()
