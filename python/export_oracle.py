#!/usr/bin/env python3
"""Export compact full-model LeLM parity data from the official Python model."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--levo-source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    model, _ = build_reference_model(
        levo_source=args.levo_source,
        model_dir=args.model_dir,
        runtime_dir=args.runtime_dir,
        device=args.device,
    )

    captured: dict[str, np.ndarray] = {}
    hooks = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor[:, -1].detach().float().cpu().numpy()

        return hook

    for index, layer in enumerate(model.transformer.model.layers):
        hooks.append(layer.register_forward_hook(capture(f"main_layer_{index:02d}")))
    hooks.append(model.transformer.model.norm.register_forward_hook(capture("main_norm")))
    hooks.append(model.mlp.register_forward_hook(capture("bridge")))
    for index, layer in enumerate(model.transformer2.model.layers):
        hooks.append(layer.register_forward_hook(capture(f"detail_layer_{index:02d}")))
    hooks.append(model.transformer2.model.norm.register_forward_hook(capture("detail_norm")))

    prompt = make_no_prompt_tokens(args.device)
    with torch.inference_mode():
        condition = model.prepare_condition_tensors(
            batch_size=1,
            text=[args.lyrics],
            descriptions=[args.description],
            audio_qt_emb=prompt,
            prepare_null_condition=True,
        )
        sequence = torch.full(
            (2, 3, 1),
            model.special_token_id,
            dtype=torch.long,
            device=args.device,
        )
        all_logits = model(sequence, condition_tensors=condition)
        cond_logits, uncond_logits = all_logits.chunk(2, dim=0)
        cfg_logits = uncond_logits + 1.5 * (cond_logits - uncond_logits)

    for hook in hooks:
        hook.remove()

    captured["conditional_logits"] = cond_logits[:, :, -1].float().cpu().numpy()
    captured["unconditional_logits"] = uncond_logits[:, :, -1].float().cpu().numpy()
    captured["cfg_logits"] = cfg_logits[:, :, -1].float().cpu().numpy()
    captured["greedy_tokens"] = (
        cfg_logits[:, :, -1].argmax(dim=-1).to(torch.int32).cpu().numpy()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **captured)
    metadata = {
        "format": "levo2-python-oracle-v1",
        "levo_source_revision": LEVO_SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "runtime_revision": RUNTIME_REVISION,
        "torch_version": torch.__version__,
        "device": str(args.device),
        "lyrics_sha256": text_hash(args.lyrics),
        "description_sha256": text_hash(args.description),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in captured.items()
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
