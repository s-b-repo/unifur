#!/usr/bin/env python3
"""Upload the prepared HF dataset folder to the Hugging Face Hub.

Usage:
    hf auth login                       # or: export HF_TOKEN=hf_...
    python3 cybercorpus/upload_hf.py <namespace>/<repo-name> [--private]

Example:
    python3 cybercorpus/upload_hf.py myname/cybersec-knowledge-corpus --private
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

FOLDER = Path(__file__).resolve().parent / "hf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="namespace/repo-name, e.g. myname/cybersec-knowledge-corpus")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    args = parser.parse_args()

    if not FOLDER.is_dir():
        print(f"error: {FOLDER} not found", file=sys.stderr)
        return 1

    api = HfApi()
    who = api.whoami()
    print(f"authenticated as: {who['name']}")

    url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"repo: {url}")

    api.upload_folder(
        folder_path=str(FOLDER),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="CyberSec Knowledge Corpus: 208k documents from 50+ security sources",
    )
    print(f"uploaded -> https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
