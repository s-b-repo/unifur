#!/usr/bin/env python3
"""Held-out eval set for the 3-layer router (meta-router L3 + category L2).

Addresses the overfitting the router owner (unifur hrouter) flagged: the training
traces come from six HF datasets, so 22/22 val accuracy mostly measures
*recognising the source dataset*, not understanding a request. This set is
deliberately DIFFERENT in kind:

  * SHORT, one-sentence, user-style prompts (not instruction-dataset rows),
    hand-authored so they share no source with the training traces;
  * labelled with the taxonomy CATEGORY ids (meta-router target) AND leaf
    SUBFIELD ids from unifur/src/taxonomy.rs (L2 category-router target);
  * ~15% OUT-OF-SCOPE prompts with `"categories": []`, so the router's unsure/
    refuse path finally has signal (the training traces had none);
  * a few deliberate CROSS-DOMAIN combos (e.g. a Rust port scanner) so the
    meta-router's multi-category path is exercised on held-out text.

Schema per line: {"id", "text", "categories": [{"id","weight"}...], "subfields": [...]}.
Weights over categories sum to 1.0. Out-of-scope lines carry categories [] and no subfields.
Stdlib only; `python3 -I`. Edit PROMPTS as the taxonomy grows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (text, [(category_id, weight)...], [subfield_leaf_id...])
PROMPTS: list[tuple[str, list[tuple[str, float]], list[str]]] = [
    # coding
    ("Why does the Rust borrow checker reject this closure capturing self?", [("coding", 1.0)], ["coding/rust"]),
    ("Write a Python function that merges two sorted lists.", [("coding", 1.0)], ["coding/python"]),
    ("How do I fix a segfault from a dangling pointer in C?", [("coding", 1.0)], ["coding/c"]),
    ("What's the idiomatic way to handle errors in Go?", [("coding", 1.0)], ["coding/go"]),
    ("Explain what a closure is in JavaScript.", [("coding", 1.0)], ["coding/js-ts"]),
    # math
    ("What's the derivative of x squared?", [("math", 1.0)], ["math/calculus"]),
    ("Prove there are infinitely many prime numbers.", [("math", 1.0)], ["math/number-theory"]),
    ("Solve the quadratic equation x^2 - 5x + 6 = 0.", [("math", 1.0)], ["math/algebra"]),
    ("What's the probability of rolling two sixes in a row?", [("math", 1.0)], ["math/probability"]),
    # science
    ("Which gas do plants absorb during photosynthesis?", [("science", 1.0)], ["science/general-science"]),
    ("Why is the sky blue?", [("science", 1.0)], ["science/general-science"]),
    ("What's the difference between an acid and a base?", [("science", 1.0)], ["science/general-science"]),
    # bio
    ("What is CRISPR used for?", [("science", 1.0)], ["science/biology"]),
    ("How does an mRNA vaccine work?", [("science", 1.0)], ["science/biology"]),
    # data
    ("Write a SQL query to find duplicate rows in a table.", [("data", 1.0)], ["data/databases"]),
    ("How do I add an index to speed up this join?", [("data", 1.0)], ["data/databases"]),
    ("What's the difference between a left and inner join?", [("data", 1.0)], ["data/databases"]),
    # cloud
    ("How do I set up a Kubernetes ingress?", [("cloud", 1.0)], ["cloud/kubernetes"]),
    ("What's the cheapest way to host a static site on AWS?", [("cloud", 1.0)], ["cloud/aws"]),
    # networking
    ("What port does DNS use?", [("networking", 1.0)], ["networking/dns"]),
    ("How does a TCP three-way handshake work?", [("networking", 1.0)], ["networking/tcp-ip"]),
    # systems
    ("How do I write a simple Linux kernel module?", [("systems", 1.0)], ["systems/kernel"]),
    ("What's the difference between a process and a thread?", [("systems", 1.0)], ["systems/concurrency"]),
    # web-mobile
    ("How do I center a div with flexbox?", [("web-mobile", 1.0)], ["web-mobile/frontend"]),
    ("Why is my Android app crashing on startup?", [("web-mobile", 1.0)], ["web-mobile/android"]),
    # ai-ml
    ("How does backpropagation work?", [("ai-ml", 1.0)], ["ai-ml/deep-learning"]),
    ("What's the difference between precision and recall?", [("ai-ml", 1.0)], ["ai-ml/ml"]),
    # cs-theory
    ("Is P equal to NP?", [("cs-theory", 1.0)], ["cs-theory/complexity"]),
    ("What's the time complexity of quicksort?", [("cs-theory", 1.0)], ["cs-theory/algorithms"]),
    # cyber-offensive
    ("How do I enumerate subdomains during recon?", [("cyber-offensive", 1.0)], ["cyber-offensive/recon"]),
    ("What's a payload to exploit a buffer overflow?", [("cyber-offensive", 1.0)], ["cyber-offensive/binary-exploitation"]),
    # cyber-defensive
    ("How do I detect lateral movement in Windows event logs?", [("cyber-defensive", 1.0)], ["cyber-defensive/detection"]),
    ("Write a Sigma rule for suspicious PowerShell.", [("cyber-defensive", 1.0)], ["cyber-defensive/detection"]),
    # cyber-appsec
    ("How do I fuzz a C JSON parser?", [("cyber-appsec", 1.0)], ["cyber-appsec/bug-finding"]),
    ("Review this login function for vulnerabilities.", [("cyber-appsec", 1.0)], ["cyber-appsec/code-audit"]),
    # cyber-netsec-crypto
    ("Explain the TLS 1.3 handshake.", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    ("Is it safe to use AES-ECB for encrypting files?", [("cyber-netsec-crypto", 1.0)], ["cyber-netsec-crypto/cryptography"]),
    # cyber-specialized
    ("How do I analyze an Android APK for malware?", [("cyber-specialized", 1.0)], ["cyber-specialized/mobile-sec"]),
    ("What's OSINT and how do I start?", [("cyber-specialized", 1.0)], ["cyber-specialized/osint"]),
    # cyber-data
    ("How do I build an anomaly detector for network logs?", [("cyber-data", 1.0)], ["cyber-data/anomaly-detection"]),
    # cross-domain combos (exercise the meta-router's multi-category path)
    ("Write a Rust program that scans a host for open TCP ports.",
     [("coding", 0.5), ("cyber-offensive", 0.3), ("networking", 0.2)], ["coding/rust", "cyber-offensive/recon"]),
    ("Write a SQL injection payload that bypasses a login form.",
     [("cyber-offensive", 0.55), ("data", 0.25), ("cyber-appsec", 0.2)], ["cyber-offensive/recon", "data/databases"]),
    ("Train a small neural net in Python to classify malware samples.",
     [("ai-ml", 0.5), ("coding", 0.3), ("cyber-data", 0.2)], ["ai-ml/deep-learning", "coding/python"]),
    ("Encrypt a file in Go using AES-GCM.",
     [("coding", 0.5), ("cyber-netsec-crypto", 0.5)], ["coding/go", "cyber-netsec-crypto/cryptography"]),
]

# Out-of-scope: the unsure/refuse path. `categories: []`, no subfields.
OUT_OF_SCOPE: list[str] = [
    "What's a good recipe for banana bread?",
    "Who won the World Cup in 2018?",
    "How are you doing today?",
    "What's the weather like this weekend?",
    "Recommend a good movie to watch tonight.",
    "How do I get a red wine stain out of a carpet?",
    "Tell me a joke.",
    "What's a fun place to visit in Italy?",
    "How do I train for a marathon?",
    "What should I get my mum for her birthday?",
    "Which is better, cats or dogs?",
    "Write me a haiku about autumn.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent.parent
                    / "nina-data" / "field-corpora" / "router-eval.jsonl")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lines, in_scope, oos = [], 0, 0
    for i, (text, cats, subs) in enumerate(PROMPTS):
        total = sum(w for _, w in cats)
        categories = [{"id": c, "weight": round(w / total, 3)} for c, w in cats]
        lines.append(json.dumps({"id": f"eval-{i:03d}", "text": text,
                                 "categories": categories, "subfields": subs}, ensure_ascii=False))
        in_scope += 1
    for j, text in enumerate(OUT_OF_SCOPE):
        lines.append(json.dumps({"id": f"oos-{j:03d}", "text": text,
                                 "categories": [], "subfields": []}, ensure_ascii=False))
        oos += 1

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = in_scope + oos
    print(f"router eval set: {total} prompts ({in_scope} in-scope, {oos} out-of-scope "
          f"= {100 * oos / total:.0f}%) -> {args.out}")
    cats = sorted({c["id"] for ln in lines for c in json.loads(ln)["categories"]})
    print(f"categories covered ({len(cats)}): {cats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
