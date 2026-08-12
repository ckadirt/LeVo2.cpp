#!/usr/bin/env python3
"""Export official conditional/null dense prefixes for C++ parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from levo_reference import build_reference_model, make_no_prompt_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levo-source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, _ = build_reference_model(
        levo_source=args.levo_source, model_dir=args.model_dir,
        runtime_dir=args.runtime_dir, device=args.device,
    )
    with torch.inference_mode():
        tensors = model.prepare_condition_tensors(
            batch_size=1, text=[args.lyrics], descriptions=[args.description],
            audio_qt_emb=make_no_prompt_tokens(args.device), prepare_null_condition=True,
        )
    # Fuser order is description, prompt_audio, type_info. Conditioner outputs
    # are [batch,time,width]; concatenate and store [branch,tower,time,width].
    main = torch.cat([tensors[name][0] for name in ("description", "prompt_audio", "type_info")], dim=1)
    detail = torch.cat([tensors[name][1] for name in ("description", "prompt_audio", "type_info")], dim=1)
    arrays = {
        "conditional_main": main[0].float().cpu().numpy(),
        "conditional_detail": detail[0].float().cpu().numpy(),
        "null_main": main[1].float().cpu().numpy(),
        "null_detail": detail[1].float().cpu().numpy(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps({k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
