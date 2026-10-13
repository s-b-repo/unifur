#!/usr/bin/env python3
"""Normalize the raw security sources into per-document UTF-8 text and JSONL.

Walks cybercorpus/raw/<source>/, converts each content file to plain text
(markdown kept as markdown minus front matter, YAML kept raw, HTML stripped,
PDFs via pdftotext, payload scripts as-is), and writes:

  cybercorpus/text/<source>/<relpath>.txt   one file per document
  cybercorpus/dataset/<source>.jsonl        {"id", "source", "path", "format", "text"}
  cybercorpus/dataset/all.jsonl             every document
  cybercorpus/dataset/manifest.json         per-source document/char counts
"""

import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
TEXT_DIR = ROOT / "text"
DATASET_DIR = ROOT / "dataset"

MAX_CHARS = 2_000_000  # cap giant wordlists so they do not dominate
MIN_CHARS = 80         # drop stubs and boilerplate-only pages

TEXT_EXTS = {
    ".md", ".markdown", ".txt", ".yml", ".yaml", ".json", ".csv", ".lst",
    ".sh", ".ps1", ".psm1", ".bat", ".cmd", ".py", ".rb", ".pl", ".php",
    ".c", ".h", ".cpp", ".asm", ".s", ".sql", ".go", ".rs", ".js", ".ts",
    ".vbs", ".hta", ".reg", ".inf", ".ini", ".conf", ".cfg", ".xml",
}
SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".woff",
    ".woff2", ".ttf", ".eot", ".zip", ".gz", ".tar", ".7z", ".rar", ".jar",
    ".exe", ".dll", ".sys", ".bin", ".dat", ".mp3", ".mp4", ".webm", ".epub",
    ".sqlite", ".lock", ".pack", ".idx",
}
SKIP_NAMES = {".env", ".ds_store", "license", "license.md", "license.txt"}
SKIP_DIRS = {".git", ".github", "node_modules", "__pycache__", "_site", ".bundle"}
# Raw bundles that only enter the dataset through their converted form.
SKIP_SOURCES = {"mitre-cti", "cvelist"}


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        out = "".join(self.parts)
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", out)).strip()


def strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            nl = text.find("\n", end + 1)
            if nl != -1:
                return text[nl + 1:]
    return text


def looks_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def pdf_to_text(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, timeout=120, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode("utf-8", errors="replace")


def convert(path: Path) -> tuple[str, str] | None:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return pdf_to_text(path), "pdf"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if looks_binary(raw[:8192]):
        return None
    text = raw.decode("utf-8", errors="replace")
    if ext in (".html", ".htm"):
        parser = HTMLText()
        try:
            parser.feed(text)
        except Exception:
            return None
        return parser.text(), "html"
    if ext in (".md", ".markdown", ""):
        return text.strip(), "markdown"
    return text.strip(), ext.lstrip(".") or "txt"


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    all_path = DATASET_DIR / "all.jsonl"
    total_docs = 0
    with all_path.open("w", encoding="utf-8") as all_out:
        for source_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
            source = source_dir.name
            if source in SKIP_SOURCES:
                continue
            source_jsonl = DATASET_DIR / f"{source}.jsonl"
            docs = chars = 0
            with source_jsonl.open("w", encoding="utf-8") as src_out:
                for path in sorted(source_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(source_dir)
                    if any(part in SKIP_DIRS for part in rel.parts):
                        continue
                    if path.name.lower() in SKIP_NAMES:
                        continue
                    ext = path.suffix.lower()
                    if ext in SKIP_EXTS:
                        continue
                    if ext not in TEXT_EXTS and ext not in (".html", ".htm", ".pdf", ""):
                        continue
                    result = convert(path)
                    if result is None:
                        continue
                    text, fmt = result
                    if len(text) > MAX_CHARS:
                        text = text[:MAX_CHARS] + "\n[truncated]"
                    if len(text) < MIN_CHARS:
                        continue
                    record = {
                        "id": f"{source}:{rel.as_posix()}",
                        "source": source,
                        "path": rel.as_posix(),
                        "format": fmt,
                        "text": text,
                    }
                    line = json.dumps(record, ensure_ascii=False)
                    src_out.write(line + "\n")
                    all_out.write(line + "\n")
                    out_txt = TEXT_DIR / source / (rel.as_posix() + ".txt")
                    out_txt.parent.mkdir(parents=True, exist_ok=True)
                    out_txt.write_text(text, encoding="utf-8")
                    docs += 1
                    chars += len(text)
            manifest[source] = {"documents": docs, "chars": chars}
            total_docs += docs
            print(f"{source:35s} {docs:6d} docs  {chars/1e6:8.1f} MB")
    (DATASET_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nTOTAL {total_docs} documents -> {all_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
