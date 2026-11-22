"""Encode byte-token records for the native Nina grounding pilot."""

import hashlib
import json
import struct

from pilot import canonical


EOS, PAD = 257, 258


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def bounded(row, prompt_width, answer_width):
    return len(canonical(row).encode()) <= prompt_width and len(row["answer"].encode()) < answer_width


def pack(path, codec_path, rows, prompt_width, answer_width, codec_rows=None):
    span = struct.Struct(f"<{prompt_width + answer_width}H")
    codec_frame = struct.Struct(f"<{answer_width}H")

    def response(row):
        answer = list(row["answer"].encode()) + [EOS]
        return answer + [PAD] * (answer_width - len(answer))

    with path.open("wb") as output, codec_path.open("wb") as codec:
        for row in rows:
            prompt = list(canonical(row).encode())
            target = response(row)
            output.write(span.pack(*([PAD] * (prompt_width - len(prompt))), *prompt, *target))
            if codec_rows is None:
                codec.write(codec_frame.pack(*target))
        if codec_rows is not None:
            for row in codec_rows:
                codec.write(codec_frame.pack(*response(row)))


def write_jsonl(path, rows):
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
