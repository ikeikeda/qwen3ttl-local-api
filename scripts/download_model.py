#!/usr/bin/env python3
"""Pre-download CustomVoice weights into Hugging Face cache (or --local-dir)."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


DEFAULT_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS CustomVoice models")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Hugging Face model id (repeatable). Default: 1.7B + 0.6B CustomVoice",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="If set, download into this directory (single model only)",
    )
    args = parser.parse_args()
    models = args.models or [DEFAULT_MODELS[0]]

    if args.local_dir and len(models) != 1:
        raise SystemExit("--local-dir requires exactly one --model")

    for mid in models:
        print(f"Downloading {mid} ...")
        path = snapshot_download(mid, local_dir=args.local_dir)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
