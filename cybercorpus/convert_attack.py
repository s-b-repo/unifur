#!/usr/bin/env python3
"""Convert MITRE ATT&CK STIX bundles (mitre/cti) into readable documents.

Reads every */<domain>.json bundle under cybercorpus/raw/mitre-cti/ and writes
one markdown document per object into cybercorpus/raw/mitre-attack/, which the
normal dataset builder then picks up like any other source.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CTI = ROOT / "raw" / "mitre-cti"
OUT = ROOT / "raw" / "mitre-attack"

CITATION = re.compile(r"\(Citation: [^)]+\)")
TAG = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    import html as html_mod

    return re.sub(r"\s{2,}", " ", html_mod.unescape(TAG.sub(" ", text))).strip()


def clean(text: str) -> str:
    text = CITATION.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)[:80].strip("-") or "unnamed"


def ext_id(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        source = ref.get("source_name", "")
        if source.startswith("mitre") or source == "capec":
            return ref.get("external_id", "")
    return ""


def render(value) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "; ".join(
                "; ".join(f"{k}: {v}" for k, v in item.items()) for item in value
            )
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {render(v)}" for k, v in value.items())
    return str(value)


def build_index(objects: list[dict]):
    by_id = {o["id"]: o for o in objects if "id" in o}
    rels = [o for o in objects if o.get("type") == "relationship"]
    return by_id, rels


def related(by_id, rels, target_id, rel_type, direction="target"):
    out = []
    for r in rels:
        if r.get("relationship_type") != rel_type:
            continue
        if direction == "target" and r.get("target_ref") == target_id:
            src = by_id.get(r.get("source_ref"))
            if src is not None:
                out.append(src)
        elif direction == "source" and r.get("source_ref") == target_id:
            tgt = by_id.get(r.get("target_ref"))
            if tgt is not None:
                out.append(tgt)
    return out


def technique_doc(obj, by_id, rels, domain):
    tid = ext_id(obj)
    kind = "sub-technique" if obj.get("x_mitre_is_subtechnique") else "technique"
    framework = "CAPEC attack pattern" if domain == "capec" else f"ATT&CK {kind}"
    lines = [f"# {obj.get('name', '?')} ({tid})", ""]
    lines.append(f"{framework} in the {domain} domain.")
    tactics = [p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])]
    if tactics:
        lines.append(f"Tactics: {', '.join(tactics)}")
    for field, label in (
        ("x_mitre_platforms", "Platforms"),
        ("x_mitre_permissions_required", "Permissions required"),
        ("x_mitre_effective_permissions", "Effective permissions"),
        ("x_mitre_data_sources", "Data sources"),
        ("x_mitre_system_requirements", "System requirements"),
        ("x_mitre_network_requirements", "Network requirements"),
        ("x_mitre_remote_support", "Remote support"),
        ("x_mitre_defense_bypassed", "Defenses bypassed"),
        ("x_mitre_impact_type", "Impact type"),
        ("x_capec_abstraction", "Abstraction"),
        ("x_capec_status", "Status"),
        ("x_capec_likelihood_of_attack", "Likelihood of attack"),
        ("x_capec_typical_severity", "Typical severity"),
        ("x_capec_prerequisites", "Prerequisites"),
        ("x_capec_skills_required", "Skills required"),
        ("x_capec_resources_required", "Resources required"),
        ("x_capec_consequences", "Consequences"),
        ("x_capec_execution_flow", "Execution flow"),
        ("x_capec_alternate_terms", "Alternate terms"),
    ):
        val = obj.get(field)
        if val:
            lines.append(f"{label}: {clean(strip_html(render(val)))}")
    examples = obj.get("x_capec_example_instances")
    if examples:
        lines += ["", "## Examples", ""]
        lines += [f"- {clean(e)}" for e in examples]
    version = obj.get("x_mitre_version")
    if version:
        lines.append(f"Version: {version}")
    desc = clean(obj.get("description", ""))
    if desc:
        lines += ["", "## Description", "", desc]
    detection = clean(obj.get("x_mitre_detection", ""))
    if detection:
        lines += ["", "## Detection", "", detection]
    mitigations = related(by_id, rels, obj["id"], "mitigates")
    if mitigations:
        lines += ["", "## Mitigations", ""]
        for m in mitigations:
            lines.append(f"- {m.get('name', '?')}: {clean(m.get('description', ''))[:400]}")
    detectors = [
        d for d in related(by_id, rels, obj["id"], "detects")
        if d.get("type") == "x-mitre-data-component"
    ]
    if detectors:
        lines += ["", "## Data components that detect this", ""]
        for d in detectors:
            lines.append(f"- {d.get('name', '?')}")
    users = [
        u
        for u in related(by_id, rels, obj["id"], "uses")
        if u.get("type") in ("intrusion-set", "tool", "malware", "campaign")
    ]
    if users:
        lines += ["", "## Known uses", ""]
        for u in users:
            lines.append(f"- {u.get('name', '?')} ({u.get('type')})")
    refs = [
        r.get("url")
        for r in obj.get("external_references", [])
        if r.get("url") and not r.get("source_name", "").startswith("mitre")
    ]
    if refs:
        lines += ["", "## References", ""]
        lines += [f"- {u}" for u in refs]
    return tid, "\n".join(lines) + "\n"


def generic_doc(obj, by_id, rels, domain):
    lines = [f"# {obj.get('name', '?')} ({ext_id(obj)})", ""]
    labels = {
        "intrusion-set": "threat group",
        "malware": "malware",
        "tool": "tool",
        "campaign": "campaign",
        "course-of-action": "mitigation",
        "x-mitre-tactic": "tactic",
        "x-mitre-data-component": "data component",
        "x-mitre-data-source": "data source",
        "identity": "identity",
        "marking-definition": "marking",
    }
    lines.append(f"ATT&CK {labels.get(obj.get('type'), obj.get('type'))} in the {domain} domain.")
    aliases = obj.get("aliases") or obj.get("x_mitre_aliases") or []
    if aliases:
        lines.append(f"Aliases: {', '.join(a for a in aliases if a != obj.get('name'))}")
    desc = clean(obj.get("description", ""))
    if desc:
        lines += ["", "## Description", "", desc]
    techniques = [
        t for t in related(by_id, rels, obj["id"], "uses", direction="source")
        if t.get("type") == "attack-pattern"
    ]
    if techniques:
        lines += ["", "## Techniques used", ""]
        for t in techniques:
            lines.append(f"- {ext_id(t)} {t.get('name', '?')}")
    refs = [
        r.get("url")
        for r in obj.get("external_references", [])
        if r.get("url") and not r.get("source_name", "").startswith("mitre")
    ]
    if refs:
        lines += ["", "## References", ""]
        lines += [f"- {u}" for u in refs]
    return "\n".join(lines) + "\n"


def convert_bundle(bundle_path: Path, domain: str) -> int:
    with bundle_path.open(encoding="utf-8") as f:
        bundle = json.load(f)
    objects = bundle.get("objects", [])
    by_id, rels = build_index(objects)
    written = 0
    for obj in objects:
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        otype = obj.get("type")
        if otype == "attack-pattern":
            tid, doc = technique_doc(obj, by_id, rels, domain)
            name = f"{tid or slug(obj.get('name', '?'))}-{slug(obj.get('name', '?'))}.md"
            sub = "techniques"
        elif otype in (
            "intrusion-set", "malware", "tool", "campaign", "course-of-action",
            "x-mitre-tactic", "x-mitre-data-component", "x-mitre-data-source",
        ):
            doc = generic_doc(obj, by_id, rels, domain)
            name = f"{slug(obj.get('name', '?'))}.md"
            sub = {
                "intrusion-set": "groups",
                "malware": "software",
                "tool": "software",
                "campaign": "campaigns",
                "course-of-action": "mitigations",
                "x-mitre-tactic": "tactics",
                "x-mitre-data-component": "data-components",
                "x-mitre-data-source": "data-sources",
            }[otype]
        else:
            continue
        if len(doc) < 80:
            continue
        out_path = OUT / domain / sub / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc, encoding="utf-8")
        written += 1
    return written


def main() -> int:
    if not CTI.is_dir():
        print(f"error: {CTI} not found (clone mitre/cti first)", file=sys.stderr)
        return 1
    total = 0
    bundles = [
        ("enterprise-attack", CTI / "enterprise-attack" / "enterprise-attack.json"),
        ("ics-attack", CTI / "ics-attack" / "ics-attack.json"),
        ("mobile-attack", CTI / "mobile-attack" / "mobile-attack.json"),
        ("pre-attack", CTI / "pre-attack" / "pre-attack.json"),
        ("capec", CTI / "capec" / "2.1" / "stix-capec.json"),
    ]
    for domain, bundle in bundles:
        if not bundle.is_file():
            print(f"{domain:20s} skipped (no bundle at {bundle})")
            continue
        n = convert_bundle(bundle, domain)
        total += n
        print(f"{domain:20s} {n:5d} documents")
    print(f"TOTAL {total} ATT&CK/CAPEC documents -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
