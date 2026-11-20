#!/usr/bin/env python3
"""TRAINING prompts for the 3-layer router — short, user-style, leaf-labelled.

Companion to router_eval_set.py. The router scored 0.125 on the held-out eval
set because its only training data (router-traces.jsonl, from six HF datasets)
taught it to recognise the SOURCE, not to understand a short request, and had
zero out-of-scope examples so the unsure path never fired. This builds the
missing training signal: many short prompts per category, out-of-scope prompts,
and cross-category combos — a DIFFERENT style/source than both the HF traces and
the eval set (it is deduped against router-eval.jsonl so the held-out set stays
clean).

Each line: {"id","text","categories":[{"id","weight"}...],"subfields":[...],"style": "short"|"combo"|"oos"}.
Categories and leaf subfields are the canonical ids from the taxonomy JSON
(`dblocks experts taxonomy --json` → nina-data/taxonomy.json). Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_terms import CATEGORY_TARGET, TERM_BANK  # noqa: E402
from router_messy_prompts import MESSY_COMBOS, MESSY_OOS, MESSY_SHORT  # noqa: E402

# Phrasings for a domain TERM (richer vocabulary than leaf labels).
TERM_TEMPLATES = [
    "What is {x}?", "Explain {x}.", "How does {x} work?", "Give a simple example of {x}.",
    "When would I use {x}?", "I keep hearing about {x} — what is it?", "Can you break down {x}?",
    "What's the idea behind {x}?", "Walk me through {x}.", "Why does {x} matter?",
]

# Topic templates that read naturally with a leaf LABEL substituted for {t}.
GENERIC = [
    "How does {t} work?",
    "Explain {t} in simple terms.",
    "What are common mistakes with {t}?",
    "When should I use {t}?",
    "Give me a quick intro to {t}.",
    "I'm new to {t}, where do I start?",
    "What's a good practice for {t}?",
    "Walk me through a basic {t} example.",
    "How do I debug a {t} problem?",
    "What should I watch out for with {t}?",
    "Can you summarise {t} for me?",
    "What's the point of {t}?",
]
# Category-specific flavour (keyed by category id), appended to GENERIC.
FLAVOUR = {
    "coding": [
        "How do I handle errors idiomatically in {t}?",
        "Write a {t} function that reverses a string.",
        "Why won't my {t} code compile?",
        "How do I run unit tests in {t}?",
    ],
    "math": [
        "Help me solve a {t} problem.",
        "What's a classic result in {t}?",
        "I'm stuck on this {t} question.",
    ],
    "science": ["What's a simple {t} experiment I can try?", "Explain a {t} concept to a 10-year-old."],
    "cyber-offensive": ["How do I approach {t} on an engagement?", "What tools help with {t}?"],
    "cyber-defensive": ["How do I set up {t} for my team?", "Write a detection idea for {t}."],
    "cyber-appsec": ["How do I get started with {t}?", "What's a quick {t} checklist?"],
    "cyber-netsec-crypto": ["Is my {t} setup secure?", "What breaks {t} in practice?"],
    "cyber-specialized": ["Where do I begin with {t}?", "What's the workflow for {t}?"],
    "cyber-data": ["How do I build a {t} pipeline?", "What features matter for {t}?"],
    "cloud": ["How do I deploy with {t}?", "What's the cost gotcha with {t}?"],
    "networking": ["How do I troubleshoot {t}?", "What port or protocol does {t} use?"],
    "systems": ["How do I profile {t}?", "What's a {t} pitfall?"],
    "data": ["How do I optimise a {t} workload?", "Design a {t} schema for me."],
    "web-mobile": ["How do I build a {t} feature?", "Why is my {t} slow?"],
    "ai-ml": ["How do I train a small {t} model?", "What dataset suits {t}?"],
    "cs-theory": ["Prove a basic fact about {t}.", "What's the intuition behind {t}?"],
}

COMBOS: list[tuple[str, list[tuple[str, float]], list[str]]] = [
    ("Write a Go service that stores events in Postgres and exposes a REST API.",
     [("coding", 0.4), ("data", 0.3), ("web-mobile", 0.3)], ["coding/go", "data/databases", "web-mobile/api-design"]),
    ("Build a Python ML pipeline that flags anomalous firewall logs.",
     [("ai-ml", 0.4), ("cyber-data", 0.35), ("coding", 0.25)], ["ai-ml/ml", "cyber-data/anomaly-detection", "coding/python"]),
    ("Set up a Kubernetes cluster with TLS and an intrusion detection sidecar.",
     [("cloud", 0.4), ("cyber-netsec-crypto", 0.3), ("cyber-defensive", 0.3)],
     ["cloud/kubernetes", "cyber-netsec-crypto/cryptography", "cyber-defensive/detection"]),
    ("Fuzz a Rust parser and triage the crashes you find.",
     [("cyber-appsec", 0.5), ("coding", 0.3), ("cyber-offensive", 0.2)], ["cyber-appsec/fuzzing", "coding/rust"]),
    ("Reverse-engineer an Android app and script the analysis in Python.",
     [("cyber-specialized", 0.4), ("cyber-defensive", 0.3), ("coding", 0.3)],
     ["cyber-specialized/mobile-sec", "cyber-defensive/reverse-engineering", "coding/python"]),
    ("Prove the time complexity of this sorting algorithm and implement it in C.",
     [("cs-theory", 0.5), ("coding", 0.5)], ["cs-theory/complexity", "coding/c"]),
    ("Capture packets on the wire and decrypt the TLS session for analysis.",
     [("cyber-netsec-crypto", 0.55), ("networking", 0.45)],
     ["cyber-netsec-crypto/packet-analysis", "networking/tls"]),
    ("Scan a subnet, enumerate services, and write the findings to a database.",
     [("cyber-offensive", 0.5), ("networking", 0.25), ("data", 0.25)],
     ["cyber-offensive/scanning", "networking/tcp-ip", "data/databases"]),
    ("Fine-tune a small language model in Python and export it to ONNX.",
     [("ai-ml", 0.6), ("coding", 0.4)], ["ai-ml/nlp", "coding/python"]),
    ("Containerise a Linux daemon and deploy it to Kubernetes.",
     [("systems", 0.35), ("cloud", 0.4), ("coding", 0.25)], ["systems/linux", "cloud/kubernetes"]),
    ("Write a Suricata rule to detect DNS tunnelling.",
     [("cyber-defensive", 0.6), ("networking", 0.4)], ["cyber-defensive/detection", "networking/dns"]),
    ("Build a feature pipeline from a Postgres warehouse for a recommender.",
     [("data", 0.5), ("ai-ml", 0.5)], ["data/data-engineering", "ai-ml/ml"]),
    ("Audit a React login form for XSS and CSRF.",
     [("cyber-appsec", 0.55), ("web-mobile", 0.45)], ["cyber-appsec/code-audit", "web-mobile/frontend"]),
    ("Explain the linear algebra behind backpropagation.",
     [("math", 0.5), ("ai-ml", 0.5)], ["math/linear-algebra", "ai-ml/deep-learning"]),
    ("Write a Rust kernel module that logs system calls.",
     [("systems", 0.55), ("coding", 0.45)], ["systems/kernel", "coding/rust"]),
    ("Harden an AWS account against cloud misconfiguration attacks.",
     [("cloud", 0.5), ("cyber-specialized", 0.5)], ["cloud/aws", "cyber-specialized/cloud-sec"]),
    ("Query firewall logs in SQL to find beaconing hosts.",
     [("data", 0.4), ("cyber-data", 0.6)], ["data/databases", "cyber-data/log-analysis"]),
    ("Analyse the time complexity of training a transformer.",
     [("cs-theory", 0.5), ("ai-ml", 0.5)], ["cs-theory/complexity", "ai-ml/deep-learning"]),
    ("Write a Python script that sniffs and parses HTTP traffic.",
     [("coding", 0.45), ("networking", 0.55)], ["coding/python", "networking/http"]),
    ("Find an SSRF vulnerability in a Node.js API.",
     [("cyber-offensive", 0.55), ("web-mobile", 0.45)], ["cyber-offensive/web-exploitation", "web-mobile/backend"]),
    ("Build a SIEM dashboard that flags anomalous logins.",
     [("cyber-defensive", 0.5), ("cyber-data", 0.5)], ["cyber-defensive/siem", "cyber-data/anomaly-detection"]),
    ("Set up a VPC with private subnets and a NAT gateway on AWS.",
     [("cloud", 0.6), ("networking", 0.4)], ["cloud/aws", "networking/tcp-ip"]),
    ("Implement fast modular exponentiation in C.",
     [("math", 0.5), ("coding", 0.5)], ["math/number-theory", "coding/c"]),
    ("Train a model to classify galaxy images.",
     [("science", 0.4), ("ai-ml", 0.6)], ["science/earth-space", "ai-ml/cv"]),
    ("Write a fuzzer harness in Go for a protobuf parser.",
     [("cyber-appsec", 0.55), ("coding", 0.45)], ["cyber-appsec/fuzzing", "coding/go"]),
    ("Implement AES-GCM file encryption in Rust.",
     [("cyber-netsec-crypto", 0.55), ("coding", 0.45)], ["cyber-netsec-crypto/cryptography", "coding/rust"]),
    ("Detect a rootkit by diffing the kernel symbol table.",
     [("systems", 0.4), ("cyber-defensive", 0.6)], ["systems/kernel", "cyber-defensive/forensics"]),
    ("Add full-text search to a web app backed by Postgres.",
     [("data", 0.5), ("web-mobile", 0.5)], ["data/databases", "web-mobile/backend"]),
    ("Serve a PyTorch model behind a FastAPI endpoint reading from Redis.",
     [("ai-ml", 0.4), ("coding", 0.3), ("data", 0.3)], ["ai-ml/mlops", "coding/python", "data/databases"]),
    ("Configure a firewall to allow only TLS 1.3.",
     [("cyber-netsec-crypto", 0.55), ("networking", 0.45)], ["cyber-netsec-crypto/firewalls", "networking/tls"]),
    ("Train a model on a GPU instance and store checkpoints in S3.",
     [("cloud", 0.5), ("ai-ml", 0.5)], ["cloud/aws", "ai-ml/ml"]),
    ("Implement Dijkstra's shortest-path algorithm in Java.",
     [("cs-theory", 0.55), ("coding", 0.45)], ["cs-theory/algorithms", "coding/java"]),
    ("Train an isolation forest on netflow data to spot intrusions.",
     [("cyber-data", 0.55), ("ai-ml", 0.45)], ["cyber-data/ml-for-security", "ai-ml/ml"]),
    ("Crack a captured WPA2 handshake and report the weakness.",
     [("cyber-specialized", 0.5), ("cyber-netsec-crypto", 0.5)], ["cyber-specialized/wireless-sec", "cyber-netsec-crypto/cryptanalysis"]),
    ("Reverse-engineer a firmware image and script the unpacking in Python.",
     [("cyber-specialized", 0.4), ("cyber-defensive", 0.3), ("coding", 0.3)],
     ["cyber-specialized/iot-ics-scada", "cyber-defensive/reverse-engineering", "coding/python"]),
    ("Clone a login page to phish credentials for a red-team test.",
     [("cyber-specialized", 0.5), ("cyber-offensive", 0.5)], ["cyber-specialized/social-engineering", "cyber-offensive/web-exploitation"]),
    ("Write a CI pipeline that runs tests and a security scan on every push.",
     [("cloud", 0.5), ("cyber-appsec", 0.3), ("coding", 0.2)], ["cloud/ci-cd", "cyber-appsec/sast"]),
    ("Build a distributed rate limiter in Go backed by Redis.",
     [("coding", 0.4), ("data", 0.3), ("systems", 0.3)], ["coding/go", "data/distributed-systems", "systems/concurrency"]),
    # harder NEAR-MISS combos: two categories that share surface vocabulary and
    # are easy to confuse, so the label needs both (for combo-F1 stress).
    ("Implement a red-black tree from scratch in Rust.",
     [("cs-theory", 0.5), ("coding", 0.5)], ["cs-theory/data-structures", "coding/rust"]),
    ("Find and exploit a use-after-free in this C program.",
     [("cyber-appsec", 0.5), ("cyber-offensive", 0.5)], ["cyber-appsec/bug-finding", "cyber-offensive/binary-exploitation"]),
    ("Clean and featurise raw training data for a model.",
     [("data", 0.5), ("ai-ml", 0.5)], ["data/data-engineering", "ai-ml/ml"]),
    ("Set up mutual TLS between two microservices.",
     [("networking", 0.45), ("cyber-netsec-crypto", 0.55)], ["networking/tls", "cyber-netsec-crypto/pki"]),
    ("Write a memory allocator in C.",
     [("systems", 0.5), ("coding", 0.5)], ["systems/linux", "coding/c"]),
    ("Correlate auth logs to find brute-force attempts.",
     [("cyber-defensive", 0.5), ("cyber-data", 0.5)], ["cyber-defensive/detection", "cyber-data/log-analysis"]),
    ("Debug why my Kubernetes container keeps getting OOM-killed.",
     [("cloud", 0.5), ("systems", 0.5)], ["cloud/kubernetes", "systems/linux"]),
    ("Add CSRF protection to a Django backend.",
     [("web-mobile", 0.5), ("cyber-appsec", 0.5)], ["web-mobile/backend", "cyber-appsec/code-audit"]),
    ("Extract secrets baked into an iOS app.",
     [("cyber-specialized", 0.5), ("cyber-offensive", 0.5)], ["cyber-specialized/mobile-sec", "cyber-offensive/web-exploitation"]),
    ("Detect payment fraud with a gradient-boosted model.",
     [("ai-ml", 0.5), ("cyber-data", 0.5)], ["ai-ml/ml", "cyber-data/ml-for-security"]),
    ("Design a REST API with per-user rate limiting.",
     [("coding", 0.45), ("web-mobile", 0.55)], ["coding/python", "web-mobile/api-design"]),
    ("Tune TCP buffer sizes on a busy Linux server.",
     [("networking", 0.5), ("systems", 0.5)], ["networking/tcp-ip", "systems/linux"]),
    ("Write a small regular-expression engine from scratch.",
     [("cs-theory", 0.5), ("coding", 0.5)], ["cs-theory/compilers", "coding/python"]),
    ("Break a custom XOR-based cipher.",
     [("cyber-netsec-crypto", 0.55), ("cyber-offensive", 0.45)], ["cyber-netsec-crypto/cryptanalysis", "cyber-offensive/exploit-dev"]),
    ("Prove gradient descent converges for a convex loss.",
     [("math", 0.55), ("ai-ml", 0.45)], ["math/analysis", "ai-ml/deep-learning"]),
]

# Out-of-scope (the unsure/refuse path). Each template is paired with fills that
# make a NATURAL sentence, so no absurd "recipe for the World Cup" combinations.
OOS_PAIRS = [
    ("What's a good recipe for {x}?", ["banana bread", "lasagne", "pancakes", "beef stew", "apple pie", "risotto", "tacos", "curry", "sourdough", "brownies", "paella"]),
    ("Who won {x}?", ["the 2018 World Cup", "the last Super Bowl", "Wimbledon this year", "the Tour de France", "the Oscars best picture", "the Masters", "the Grand National", "Eurovision"]),
    ("Recommend a {x} to watch tonight.", ["comedy film", "documentary", "thriller", "nature series", "romance movie", "sci-fi show", "cooking show", "mystery series"]),
    ("What's the weather like in {x} this weekend?", ["Lisbon", "Rome", "Tokyo", "Berlin", "Cape Town", "Oslo", "Seoul", "Dublin"]),
    ("How do I get {x} stains out of a shirt?", ["red wine", "coffee", "grass", "ink", "grease", "chocolate", "tomato sauce", "mud"]),
    ("What should I get my {x} for their birthday?", ["sister", "dad", "best friend", "mum", "nephew", "colleague", "grandma", "flatmate"]),
    ("Plan a weekend trip to {x} for me.", ["Rome", "Barcelona", "Kyoto", "Amsterdam", "Cape Town", "Prague", "Edinburgh", "Porto", "Montreal"]),
    ("Tell me a joke about {x}.", ["cats", "Mondays", "coffee", "the weather", "pirates", "robots", "cheese", "dinosaurs", "aliens"]),
    ("Write a short poem about {x}.", ["autumn", "the sea", "sunsets", "rain", "the moon", "snow", "a river", "the stars", "a garden"]),
    ("How do I train for a {x}?", ["5k run", "marathon", "triathlon", "half marathon", "first hike", "cycling race", "swim meet", "obstacle course"]),
    ("What's a fun fact about {x}?", ["octopuses", "honey", "the brain", "volcanoes", "penguins", "bees", "the ocean", "ancient Rome", "sloths"]),
    ("Is {x} healthier than coffee?", ["green tea", "matcha", "chicory", "decaf", "herbal tea", "yerba mate", "kombucha", "hot chocolate"]),
    ("What's the best way to {x}?", ["learn guitar", "save money", "sleep better", "make friends", "stay motivated", "declutter a room", "meal prep", "start journaling", "wake up early", "reduce stress"]),
    ("Can you suggest a good {x}?", ["board game", "podcast", "novel", "hobby", "houseplant", "weekend activity", "dinner idea", "card game", "audiobook", "picnic spot"]),
    ("What does it mean when {x}?", ["my cat purrs", "you dream about falling", "leaves change colour", "a dog wags its tail", "it rains while sunny", "you get goosebumps", "cats knead", "the moon looks red"]),
    ("Help me write a {x}.", ["birthday toast", "thank-you note", "wedding speech", "grocery list", "packing list for a trip", "holiday card message", "get-well card", "retirement toast"]),
    ("How long does it take to {x}?", ["roast a chicken", "brew kombucha", "grow tomatoes", "learn to knit", "train a puppy", "boil an egg", "ripen an avocado", "bake sourdough"]),
    ("What's a thoughtful gift for a {x}?", ["new neighbour", "teacher", "coffee lover", "gardener", "book club", "new parent", "pet owner", "traveller"]),
    ("How do I keep {x} alive?", ["my basil", "a succulent", "an orchid", "cut flowers", "a fern", "my herb garden", "a bonsai", "a peace lily"]),
    ("What's a good {x} for a road trip?", ["playlist", "podcast", "snack", "audiobook", "game for kids", "stop halfway", "singalong song", "car tidy routine"]),
    ("How do I fix a {x}?", ["squeaky hinge", "leaky tap", "wobbly table", "stuck zipper", "running toilet", "loose doorknob", "jammed drawer", "flat bike tyre"]),
    ("What's the difference between {x}?", ["espresso and drip coffee", "jam and marmalade", "a crocodile and an alligator", "baking soda and baking powder", "a latte and a cappuccino", "weather and climate", "a turtle and a tortoise", "cement and concrete"]),
    ("Can you recommend a {x}?", ["mystery novel", "feel-good movie", "beginner houseplant", "easy weeknight dinner", "relaxing playlist", "cosy cafe", "day hike", "family board game"]),
    ("How do I get my {x}?", ["toddler to eat veggies", "dog to stop barking", "cat to use the scratch post", "kid to sleep earlier", "orchid to flower", "bread to rise", "lawn to green up", "cat to stop meowing"]),
    ("What's a quick {x} I can make?", ["healthy breakfast", "weeknight dinner", "packed lunch", "no-bake dessert", "smoothie", "one-pot meal", "snack for kids", "pasta sauce"]),
    ("Any tips for {x}?", ["sleeping better", "staying awake on a drive", "keeping houseplants alive", "learning to swim as an adult", "packing light", "meal prepping", "saving on groceries", "beating jet lag"]),
    ("How do I {x}?", ["clean cast iron without ruining it", "fold a fitted sheet neatly", "peel garlic without the smell", "remove gum from hair", "whiten laundry without bleach", "sharpen a knife without a stone", "ripen bananas faster", "descale a kettle naturally"]),
    ("What's a fun indoor activity for {x}?", ["a rainy day", "a kids' birthday", "a quiet evening", "two bored teens", "a snow day", "a date night in", "a toddler", "a group of adults"]),
]


def slug(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()[:12]


def load_eval_texts(path: Path) -> set[str]:
    out: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.add(slug(json.loads(line)["text"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent.parent.parent / "nina-data"
    ap.add_argument("--taxonomy", type=Path, default=base / "taxonomy.json")
    ap.add_argument("--eval", type=Path, default=base / "field-corpora" / "router-eval.jsonl")
    ap.add_argument("--eval2", type=Path, default=base / "field-corpora" / "router-eval-v2.jsonl")
    ap.add_argument("--out", type=Path, default=base / "field-corpora" / "router-train-prompts.jsonl")
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--oos-frac", type=float, default=0.17)
    ap.add_argument("--no-messy", action="store_true",
                    help="omit the messy-style subset (the clean baseline for the style-gap A/B)")
    args = ap.parse_args()

    taxo = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    # never overlap EITHER held-out eval (v1 or the messy v2 and its cal/test halves)
    used = load_eval_texts(args.eval) | load_eval_texts(args.eval2)
    seen: set[str] = set(used)
    lines: list[str] = []
    counts: dict[str, int] = {}

    def add(text: str, cats: list[tuple[str, float]], subs: list[str], style: str,
            messy: bool = False) -> None:
        s = slug(text)
        if s in seen:
            return
        seen.add(s)
        total = sum(w for _, w in cats) or 1.0
        categories = [{"id": c, "weight": round(w / total, 3)} for c, w in cats]
        # messy-style rows carry a "messy-" id prefix and messy=true so the A/B
        # "without the messy subset" is a one-field filter, no re-generation needed.
        rec = {"id": f"{'messy-' if messy else ''}{style}-{len(lines):04d}", "text": text,
               "categories": categories, "subfields": subs, "style": style}
        if messy:
            rec["messy"] = True
        lines.append(json.dumps(rec, ensure_ascii=False))
        for c, _ in cats:
            counts[c] = counts.get(c, 0) + 1

    for cat in taxo["categories"]:
        cid = cat["id"]
        target = CATEGORY_TARGET.get(cid, args.per_category)  # weak categories get more
        # 1) term-based prompts: real domain VOCABULARY, grouped by canonical leaf
        #    (what fixes the n-gram router's vocabulary gap).
        pairs = [(leaf_id, term) for leaf_id, terms in TERM_BANK.get(cid, {}).items() for term in terms]
        ti = 0
        for _pass in range(12):
            for leaf_id, term in pairs:
                if counts.get(cid, 0) >= target:
                    break
                add(TERM_TEMPLATES[ti % len(TERM_TEMPLATES)].format(x=term),
                    [(cid, 1.0)], [f"{cid}/{leaf_id}"], "short")
                ti += 1
            if counts.get(cid, 0) >= target:
                break
        # 2) top up with leaf-label phrasings if the term bank was thin for a category
        pool = GENERIC + FLAVOUR.get(cid, [])
        leaves = cat["leaves"]
        for rounds in range(target * 2):
            if counts.get(cid, 0) >= target:
                break
            leaf = leaves[rounds % len(leaves)]
            tmpl = pool[(rounds // len(leaves)) % len(pool)]
            add(tmpl.format(t=leaf["label"]), [(cid, 1.0)], [f"{cid}/{leaf['id']}"], "short")

    for text, cats, subs in COMBOS:
        add(text, cats, subs, "combo")

    # messy-STYLE prompts (typos, fragments, "halp") so train matches the v2 eval
    # surface form, not just its vocabulary. Deduped against train, v1 and v2 by add().
    # --no-messy omits them: the clean baseline arm of the style-gap A/B.
    if not args.no_messy:
        for text, cats, subs in MESSY_SHORT:
            add(text, cats, subs, "short", messy=True)
        for text, cats, subs in MESSY_COMBOS:
            add(text, cats, subs, "combo", messy=True)

    # out-of-scope to ~oos-frac of the final set, natural template+fill pairs
    target_oos = int(len(lines) * args.oos_frac / (1 - args.oos_frac))
    i = 0
    if not args.no_messy:
        for text in MESSY_OOS:  # messy OOS first, then templated fills top up to target
            if i >= target_oos:
                break
            before = len(lines)
            add(text, [], [], "oos", messy=True)
            i += len(lines) - before
    # interleave fills across templates so the set is varied, not template-blocked
    max_fills = max(len(f) for _, f in OOS_PAIRS)
    for k in range(max_fills):
        for tmpl, fills in OOS_PAIRS:
            if i >= target_oos:
                break
            if k < len(fills):
                add(tmpl.format(x=fills[k]), [], [], "oos")
                i += 1
        if i >= target_oos:
            break

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    oos = sum(1 for ln in lines if json.loads(ln)["style"] == "oos")
    combo = sum(1 for ln in lines if json.loads(ln)["style"] == "combo")
    print(f"router train prompts: {len(lines)} total "
          f"({len(lines)-oos-combo} short, {combo} combo, {oos} oos = {100*oos/len(lines):.0f}%) -> {args.out}")
    print(f"per-category (incl. combos): {dict(sorted(counts.items()))}")
    print(f"deduped against eval set ({len(used)} eval texts): no overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
