"""Convert the public NCM XSum tensors to Unifur's native weight shards."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from safetensors.torch import save_file
import torch


LOCAL_CHECKPOINTS = Path(__file__).resolve().parents[3] / "checkpoints"
EXPERTS = 17
BOXES = {"general": 4, "coding": 4, "mathematics": 3, "science": 3, "technology": 3}
HASHES = {
    "generator.pt": "f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4",
    "tokenizer.pt": "1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7",
}
GPT2_SHA256 = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"


def verify(path, expected):
    with path.open("rb") as file:
        actual = file_digest(file, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Unrecognized release artifact {path.name}")


def read_weights(name, checkpoints=LOCAL_CHECKPOINTS):
    path = checkpoints / "xsum" / name
    verify(path, HASHES[name])
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def download(checkpoints):
    model = "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/main/xsum"
    gpt2 = "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json"
    files = [(f"{model}/{name}", checkpoints / "xsum" / name) for name in HASHES]
    files.append((gpt2, checkpoints / "gpt2/tokenizer.json"))
    for url, path in files:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with urlopen(url, timeout=60) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        temporary.replace(path)


def save(path, tensors):
    # Scalar tensors are one-element vectors for Unifur's tensor reader.
    save_file({name: value.contiguous().reshape(value.shape or (1,)) for name, value in tensors.items()}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=LOCAL_CHECKPOINTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    output = (args.output or args.checkpoints / "nina-1b").resolve()
    if output.exists():
        raise ValueError(f"Output already exists at {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.download:
        download(args.checkpoints)
    tokenizer = args.checkpoints / "gpt2/tokenizer.json"
    verify(tokenizer, GPT2_SHA256)
    generation = read_weights("generator.pt", args.checkpoints)
    codec = read_weights("tokenizer.pt", args.checkpoints)["model"]
    shared, expert = {}, {}
    for name, value in generation["model"].items():
        if name in {"lvl_1L", "attn_mask_for_masking"}:
            continue
        (expert if ".mlp." in name else shared)[name] = value
    codec = {
        name: value for name, value in codec.items()
        if name == "encoder.wte.weight" or name.startswith((
            "decoder.", "vq.embeddings.", "vq.quant_resi.", "vq.learned_up_l1.",
        ))
    }
    generator_params = sum(value.numel() for value in shared.values())
    expert_params = sum(value.numel() for value in expert.values())
    with TemporaryDirectory(prefix=".nina-1b-", dir=output.parent) as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        save(bundle / "generator.safetensors", shared)
        save(bundle / "codec.safetensors", codec)
        save(bundle / "decode.safetensors", generation["inference"])
        for index in range(EXPERTS):
            save(bundle / f"expert-{index:02}.safetensors", expert)
        shutil.copyfile(tokenizer, bundle / "tokenizer.json")
        stats = {
            "version": 1,
            "upstream": "hazyresearch/ncm-checkpoints/xsum",
            "checkpoint_sha256": HASHES,
            "boxes": BOXES,
            "experts": EXPERTS,
            "generator_parameters": generator_params + EXPERTS * expert_params,
            "shared_parameters": generator_params,
            "expert_parameters": expert_params,
            "specialized_training_steps": 0,
        }
        (bundle / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
        bundle.rename(output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
