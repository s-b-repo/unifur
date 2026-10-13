#!/usr/bin/env python3
"""Convert CVE JSON 5.x records (CVEProject/cvelistV5) into readable documents.

Reads cybercorpus/raw/cvelist/cves/<year>/**/CVE-*.json and writes one markdown
document per published CVE into cybercorpus/raw/cve-records/<year>/. Rejected
records are skipped.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "raw" / "cvelist" / "cves"
OUT = ROOT / "raw" / "cve-records"


def render(record: dict) -> str | None:
    meta = record.get("cveMetadata", {})
    if meta.get("state") != "PUBLISHED":
        return None
    cve_id = meta.get("cveId", "?")
    cna = record.get("containers", {}).get("cna", {})
    lines = [f"# {cve_id}", ""]
    title = cna.get("title")
    if title:
        lines.append(f"Title: {title}")
    if meta.get("datePublished"):
        lines.append(f"Published: {meta['datePublished']}")
    assigner = cna.get("assignerShortName")
    if assigner:
        lines.append(f"Assigner: {assigner}")
    descs = [d.get("value", "") for d in cna.get("descriptions", []) if d.get("lang") == "en"]
    if not descs:
        descs = [d.get("value", "") for d in cna.get("descriptions", [])]
    if descs:
        lines += ["", "## Description", "", descs[0].strip()]
    cwes = []
    for pt in cna.get("problemTypes", []):
        for d in pt.get("descriptions", []):
            if d.get("cweId"):
                cwes.append(f"{d['cweId']} {d.get('description', '')}".strip())
            elif d.get("description"):
                cwes.append(d["description"])
    if cwes:
        lines += ["", "## Weakness", ""] + [f"- {c}" for c in cwes]
    metrics = []
    for m in cna.get("metrics", []):
        for key in ("cvssV4_0", "cvssV3_1", "cvssV3_0", "cvssV2_0"):
            if key in m:
                v = m[key]
                metrics.append(
                    f"{key} {v.get('baseScore', '?')} ({v.get('baseSeverity', 'n/a')}): {v.get('vectorString', '')}"
                )
        if m.get("other") and m["other"].get("type") == "ssvc":
            pass  # SSVC blobs are decision trees, not readable prose
    if metrics:
        lines += ["", "## Severity", ""] + [f"- {x}" for x in metrics]
    affected = []
    for a in cna.get("affected", []):
        vendor = a.get("vendor", "")
        product = a.get("product", "")
        versions = ", ".join(
            f"{v.get('version', '?')} ({v.get('status', '')})" for v in a.get("versions", [])[:8]
        )
        entry = f"{vendor} {product}".strip()
        if versions:
            entry += f": {versions}"
        if entry:
            affected.append(entry)
    if affected:
        lines += ["", "## Affected", ""] + [f"- {a}" for a in affected[:20]]
    refs = [r.get("url") for r in cna.get("references", []) if r.get("url")]
    if refs:
        lines += ["", "## References", ""] + [f"- {u}" for u in refs[:10]]
    return "\n".join(lines) + "\n"


def main() -> int:
    if not SRC.is_dir():
        print(f"error: {SRC} not found (sparse-clone cvelistV5 first)", file=sys.stderr)
        return 1
    written = skipped = 0
    for path in sorted(SRC.rglob("CVE-*.json")):
        year = path.relative_to(SRC).parts[0]
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        doc = render(record)
        if doc is None or len(doc) < 80:
            skipped += 1
            continue
        out_path = OUT / year / f"{path.stem}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc, encoding="utf-8")
        written += 1
    print(f"{written} CVE documents written, {skipped} skipped (rejected/unparseable) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
