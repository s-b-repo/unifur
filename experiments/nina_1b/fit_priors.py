"""Fit small domain code priors for the native NCM checkpoint."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from tokenizers import Tokenizer
import torch
from torch.nn import functional as F

from convert_release import HASHES, LOCAL_CHECKPOINTS, read_weights


ROOT = LOCAL_CHECKPOINTS.parent
LEVELS = (4, 6, 10, 14, 20, 26, 32, 40, 48)
CODES = 4096
EOS, PAD = 50256, 50257
TOPICS = re.compile(
    r"\b(?:exploit\w*|shellcode|phish\w*|pentest\w*|backdoor|malware|ransomware|"
    r"intrusion|injection|payload|bypass\w*|password crack\w*|"
    r"pathogen\w*|virolog\w*|transmissib\w*|virulence|bioweapon\w*|"
    r"bomb\w*|improvised weapon)\b",
    re.I,
)
KEY = re.compile(r"-----BEGIN .*PRIVATE KEY-----|\b(?:ghp_|sk-ant-|github_pat_)[A-Za-z0-9_]{20,}")
SOURCES = {
    "mathematics": ("math", "AI-MO__NuminaMath-CoT"),
    "coding": ("coding", "google-research-datasets__mbpp"),
    "science": ("science", "allenai__sciq"),
    "technology": ("tech", "gretelai__synthetic_text_to_sql"),
}
SECURITY_TOPICS = [
    "Multifactor authentication combines distinct factor types, such as a password and a security key.",
    "Grant a service identity only the permissions needed for its documented operations.",
    "Record administrative changes with the time, actor, target resource, action, and outcome.",
    "Store backups separately from primary systems and test that the backups can be restored.",
    "Review authentication logs for new devices, unusual locations, and unexpected access times.",
    "Rotate an exposed credential and invalidate existing sessions before investigating its use.",
    "Use a separate privileged account for administrative work and review its access periodically.",
    "Keep an inventory of production dependencies and track whether security updates have been applied.",
    "Use network segmentation to limit unnecessary communication between service groups.",
    "A threat model identifies assets, trust boundaries, likely failures, and appropriate controls.",
    "Approval workflows can require independent review of changes to access policies.",
    "Incident response records should preserve a timeline and the provenance of collected evidence.",
]
SECURITY_QUESTIONS = [
    "What makes an authentication method multifactor?",
    "How should a service identity's permissions be scoped?",
    "What belongs in a record of an administrative change?",
    "How should an organization verify its backups?",
    "What anomalies are useful to review in authentication logs?",
    "What should an administrator do after discovering an exposed credential?",
    "How should privileged accounts be managed?",
    "How can a team track security updates for its dependencies?",
    "What is the purpose of network segmentation?",
    "What does a threat model identify?",
    "How can an organization review changes to access policies?",
    "What should an incident response record preserve?",
]


def candidates(data, export):
    for box, (field, source) in SOURCES.items():
        path = data / f"{field}.clean.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if Path(record["path"]).parent.name != source:
                    continue
                text = record["text"].strip()
                parts = [part.strip() for part in text.split("\n\n") if part.strip()]
                if len(parts) < 2 or TOPICS.search(text) or KEY.search(text):
                    continue
                if box == "technology" and not parts[-1].upper().startswith("SELECT "):
                    continue
                prompt = parts[0] if box in {"coding", "mathematics"} else "\n\n".join(parts[:-1])
                yield box, prompt, parts[-1], record["id"], source
    with (export / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            user, assistant = json.loads(line)["messages"]
            prompt, answer = user["content"].strip(), assistant["content"].strip()
            text = prompt + "\n" + answer
            if not TOPICS.search(text) and not KEY.search(text):
                yield "general", prompt, answer, sha256(answer.encode()).hexdigest(), "claude-chat-export"


def encode_targets(ids, weights):
    def conv(x, key, stride=1, padding=0, transpose=False):
        operation = F.conv_transpose1d if transpose else F.conv1d
        return operation(x, weights[f"{key}.weight"], weights[f"{key}.bias"],
                         stride=stride, padding=padding)

    embedded = F.embedding(ids, weights["encoder.wte.weight"])
    embedded = embedded + weights["encoder.wpe.weight"][:ids.shape[1]]
    residual = F.group_norm(embedded.transpose(1, 2), 8)
    result = []
    for i, length in enumerate(LEVELS):
        if i == 0:
            coarse = conv(residual, "vq.learned_down_l1.0", stride=4)
            coarse = conv(F.group_norm(coarse, 8), "vq.learned_down_l1.2", stride=3)
            coarse = F.group_norm(coarse, 8)
        else:
            coarse = F.adaptive_avg_pool1d(residual, length)
        book = weights[f"vq.embeddings.{i}.weight"]
        partial = coarse.transpose(1, 2)
        code = (2 * partial @ book.T - book.square().sum(-1)).argmax(-1)
        result.append(code)
        vectors = F.embedding(code, book).transpose(1, 2)
        if i == 0:
            vectors = conv(vectors, "vq.learned_up_l1.0", stride=3, transpose=True)
            vectors = conv(F.group_norm(vectors, 8), "vq.learned_up_l1.2", stride=4, transpose=True)
        elif length != LEVELS[-1]:
            vectors = F.interpolate(vectors, size=LEVELS[-1], mode="linear", align_corners=False)
        residual = residual - 0.5 * (vectors + conv(vectors, f"vq.quant_resi.{i}", padding=1))
    return result


def collect(args, tokenizer):
    selected = defaultdict(lambda: {"train": [], "validation": []})
    seen = set()
    seen_prompts = set()
    for field, prompt, text, identity, source in candidates(args.data, args.export):
        digest = sha256(" ".join(text.split()).casefold().encode()).hexdigest()
        prompt_digest = sha256(" ".join(prompt.split()).casefold().encode()).hexdigest()
        if digest in seen or prompt_digest in seen_prompts:
            continue
        split = "validation" if int(digest[:8], 16) % 5 == 0 else "train"
        limit = args.documents if split == "train" else max(4, args.documents // 4)
        if len(selected[field][split]) >= limit:
            continue
        tokens = tokenizer.encode(text).ids
        if len(tokens) < 3 or len(tokens) > 2 * LEVELS[-1]:
            continue
        tokens = tokens[: LEVELS[-1] - 1] + [EOS]
        tokens += [PAD] * (LEVELS[-1] - len(tokens))
        selected[field][split].append((tokens, identity, source, prompt))
        seen.add(digest)
        seen_prompts.add(prompt_digest)
        if all(len(selected[field][split]) >= limit for field in [*SOURCES, "general"]):
            if all(len(parts["validation"]) >= max(4, args.documents // 4)
                   and len(parts["train"]) >= args.documents for parts in selected.values()):
                break
    for i, text in enumerate(SECURITY_TOPICS):
        ids = tokenizer.encode(text).ids + [EOS]
        ids += [PAD] * (LEVELS[-1] - len(ids))
        split = "validation" if i % 4 == 0 else "train"
        selected["technology"][split].append((ids, f"security-{i}", "general-security-notes", SECURITY_QUESTIONS[i]))
    return selected


def counts(records, weights):
    histogram = torch.zeros(len(LEVELS), CODES, dtype=torch.float64)
    for start in range(0, len(records), 12):
        ids = torch.tensor([row[0] for row in records[start:start + 12]], dtype=torch.long)
        for i, codes in enumerate(encode_targets(ids, weights)):
            histogram[i] += torch.bincount(codes.reshape(-1), minlength=CODES)
    return histogram


def write_labels(out, records, weights, limit):
    out.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation"):
        with (out / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for field, parts in sorted(records.items()):
                # Include security questions in the technology pilot.
                rows = sorted(parts[split], key=lambda row: row[2] != "general-security-notes")[:limit]
                if field == "technology" and not any(row[2] != "general-security-notes" for row in rows):
                    rows[-1] = next(row for row in parts[split] if row[2] != "general-security-notes")
                codes = encode_targets(torch.tensor([row[0] for row in rows]), weights)
                for i, (_, identity, source, prompt) in enumerate(rows):
                    record = {"id": identity, "field": field, "source": source, "prompt": prompt,
                              "codes": [level[i].tolist() for level in codes]}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {"version": 1, "tokenizer_sha256": HASHES["tokenizer.pt"], "levels": LEVELS}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parent / "nina-data/raw-scale")
    parser.add_argument("--export", type=Path, default=ROOT.parent / "claude-training-export/dataset")
    parser.add_argument("--checkpoints", type=Path, default=LOCAL_CHECKPOINTS)
    parser.add_argument("--documents", type=int, default=48)
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints/domain-priors.json")
    parser.add_argument("--labeled-out", type=Path)
    parser.add_argument("--labeled-per-field", type=int, default=4)
    args = parser.parse_args()
    if args.documents < 4 or args.out.exists():
        parser.error("--documents must be at least 4 and --out must be a new file")
    torch.set_num_threads(3)
    tokenizer = Tokenizer.from_file(str(args.checkpoints / "gpt2/tokenizer.json"))
    records = collect(args, tokenizer)
    weights = read_weights("tokenizer.pt", args.checkpoints)["model"]
    observed = {}
    for name, splits in records.items():
        if not splits["train"] or not splits["validation"]:
            raise ValueError(f"Need training and held-out targets for {name}")
        observed[name] = {split: counts(rows, weights) for split, rows in splits.items()}
        print(f"{name}: {len(splits['train'])} train, {len(splits['validation'])} held out", flush=True)
    if args.labeled_out:
        if args.labeled_per_field < 2:
            parser.error("--labeled-per-field must be at least 2")
        write_labels(args.labeled_out, records, weights, args.labeled_per_field)

    pooled = sum((x["train"] for x in observed.values()), torch.zeros(len(LEVELS), CODES))
    pooled = (pooled + 1) / (pooled + 1).sum(-1, keepdim=True)
    fields, metrics = {}, {}
    for name, distribution in observed.items():
        train, validation = distribution["train"], distribution["validation"]
        regularized = train + 64 * pooled
        fitted = regularized / regularized.sum(-1, keepdim=True)
        loss = lambda p: float(-(validation * p.log()).sum() / validation.sum())
        base, held_out = loss(pooled), loss(fitted)
        shifts = (fitted / pooled).log().clamp(-1, 1) * (0.15 if held_out < base else 0.0)
        fields[name] = {"documents": len(records[name]["train"]), "logit_offsets": shifts.tolist()}
        metrics[name] = {
            "pooled_heldout_code_ce": base,
            "field_heldout_code_ce": held_out,
            "heldout_documents": len(records[name]["validation"]),
            "applied": held_out < base,
            "sources": sorted({row[2] for part in records[name].values() for row in part}),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump({"version": 1, "fields": fields, "heldout_code_prior_metrics": metrics}, output)
        output.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
