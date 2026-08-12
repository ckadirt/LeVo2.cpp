#!/usr/bin/env python3
"""Export official LeVo v2-medium greedy generation tokens for C++ parity.

The program intentionally delegates generation to the pinned upstream LeVo
implementation.  It is a diagnostic/oracle producer, not an alternative
runtime: the generated NPY is stream-major int32 ``[mixed, vocal, bgm]`` and
is suitable for direct comparison with ``levo-cli --greedy`` artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from levo_reference import (
    LEVO_SOURCE_REVISION,
    MODEL_REVISION,
    MODEL_SHA256,
    RUNTIME_REVISION,
    build_reference_model,
    make_no_prompt_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levo-source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg-scale", default=1.5, type=float)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output and manifest"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("--duration must be finite and positive")
    if not math.isfinite(args.cfg_scale):
        raise ValueError("--cfg-scale must be finite")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    frames = math.floor(args.duration * 25)
    if frames < 1:
        raise ValueError("--duration must produce at least one 25 Hz frame")
    if args.output.suffix != ".npy":
        raise ValueError("--output must use the .npy extension")
    manifest = args.output.with_suffix(".json")
    if not args.overwrite and (args.output.exists() or manifest.exists()):
        raise FileExistsError("output exists; use --overwrite to replace it")

    started = time.monotonic()
    model, _ = build_reference_model(
        levo_source=args.levo_source,
        model_dir=args.model_dir,
        runtime_dir=args.runtime_dir,
        device=args.device,
    )
    load_seconds = time.monotonic() - started
    prompt = make_no_prompt_tokens(args.device)
    with torch.inference_mode():
        # Upstream uses `use_sampling=False` as its greedy branch. Temperature
        # zero documents the equivalent no-softmax path; top-k remains part of
        # provenance even though argmax makes it inactive.
        tokens = model.generate(
            texts=[args.lyrics],
            descriptions=[args.description],
            audio_qt_embs=prompt,
            max_gen_len=frames,
            use_sampling=False,
            temp=0.0,
            top_k=args.top_k,
            top_p=0.0,
            cfg_coef=args.cfg_scale,
            record_tokens=True,
            # CodecLM's released generation parameters select this window; it
            # must agree with levo.cpp's repetition implementation.
            record_window=50,
        )
    generation_seconds = time.monotonic() - started - load_seconds
    values = tokens.squeeze(0).to(dtype=torch.int32, device="cpu").numpy()
    if values.ndim != 2 or values.shape[0] != 3:
        raise RuntimeError(f"official generator returned unexpected shape {values.shape}")
    if values.shape[1] > frames:
        raise RuntimeError("official generator returned more frames than requested")
    if not np.logical_and(values >= 0, values <= 16384).all():
        raise RuntimeError("official generator emitted an invalid token ID")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, values, allow_pickle=False)
    metadata = {
        "format": "levo2-official-generation-oracle-v1",
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "frames_requested": frames,
        "duration_seconds": args.duration,
        "levo_source_revision": LEVO_SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "runtime_revision": RUNTIME_REVISION,
        "torch_version": torch.__version__,
        "device": str(args.device),
        "sampling": {
            "use_sampling": False,
            "temperature": 0.0,
            "top_k": args.top_k,
            "top_p": 0.0,
            "cfg_scale": args.cfg_scale,
            "record_tokens": True,
            "record_window": 50,
        },
        "lyrics_sha256": sha256_text(args.lyrics),
        "description_sha256": sha256_text(args.description),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "output_sha256": sha256_file(args.output),
    }
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} {list(values.shape)} in {generation_seconds:.3f}s "
        f"(model load {load_seconds:.3f}s)"
    )


if __name__ == "__main__":
    main()
