#!/usr/bin/env python3
"""Export a no-prefix LeLM forward for direct C++ graph parity debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from levo_reference import build_reference_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levo-source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    model, _ = build_reference_model(
        levo_source=args.levo_source,
        model_dir=args.model_dir,
        runtime_dir=args.runtime_dir,
        device=args.device,
    )
    if args.compute_dtype == "float32":
        # GGUF F16 values are retained, then promoted for operations. This
        # diagnostic distinguishes graph/layout errors from activation-rounding
        # differences against the released all-F16 PyTorch execution.
        model.float()
    sequence = torch.full(
        (1, 3, 1), model.special_token_id, dtype=torch.long, device=args.device
    )
    with torch.inference_mode():
        main_input = model.emb[0](sequence[:, 0])
        main = model.transformer(inputs_embeds=main_input, use_cache=False)
        mixed_logits = main.logits
        detail_embedding = model.layer2_emb[1](sequence[:, 1]) + model.layer2_emb[2](sequence[:, 2])
        detail_input = model.mlp(torch.cat([detail_embedding, main.hidden_states], dim=-1))
        detail = model.transformer2(inputs_embeds=detail_input, use_cache=False)
        logits = torch.stack(
            [mixed_logits, model.linears[0](detail.hidden_states), model.linears[1](detail.hidden_states)],
            dim=1,
        )
    logits = logits[0, :, 0].float().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, logits)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "levo2-unconditioned-oracle-v1",
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
                "greedy": logits.argmax(axis=-1).tolist(),
                "finite": bool(np.isfinite(logits).all()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
