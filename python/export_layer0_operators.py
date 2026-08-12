#!/usr/bin/env python3
"""Capture the official main transformer layer-0 operator boundaries.

This is an investigation helper: it calls the pinned upstream modules and
writes only the final conditioned BOS position.  It does not duplicate model
inference for production use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from levo_reference import build_reference_model, make_no_prompt_tokens


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levo-source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def last_width(value: torch.Tensor) -> np.ndarray:
    return value[0, -1].float().cpu().numpy()


def main() -> None:
    args = arguments()
    model, _ = build_reference_model(
        levo_source=args.levo_source,
        model_dir=args.model_dir,
        runtime_dir=args.runtime_dir,
        device="cuda",
    )
    with torch.inference_mode():
        prompt = make_no_prompt_tokens("cuda")
        condition = model.prepare_condition_tensors(
            batch_size=1,
            text=[args.lyrics],
            descriptions=[args.description],
            audio_qt_emb=prompt,
            prepare_null_condition=True,
        )
        sequence = torch.full((2, 3, 1), model.special_token_id, dtype=torch.long, device="cuda")
        first = model.emb[0](sequence[:, 0])
        second = sum(model.layer2_emb[index](sequence[:, index]) for index in range(1, 3))
        fused, _ = model.fuser(first, second, condition)
        x = fused[:1]
        layer = model.transformer.model.layers[0]
        attention = layer.self_attn
        norm = layer.input_layernorm(x)
        q_raw = attention.q_proj(norm)
        k_raw = attention.k_proj(norm)
        v_raw = attention.v_proj(norm)
        batch, steps, _ = q_raw.shape
        q = q_raw.view(batch, steps, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = k_raw.view(batch, steps, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        v = v_raw.view(batch, steps, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        positions = torch.arange(steps, device="cuda", dtype=torch.long).unsqueeze(0)
        cos, sin = attention.rotary_emb(v, seq_len=steps)
        from codeclm.models.llama.modeling_llama import apply_rotary_pos_emb
        q_rotary, k_rotary = apply_rotary_pos_emb(q, k, cos, sin, positions)
        scores = torch.matmul(q_rotary, k_rotary.transpose(2, 3)) / (attention.head_dim ** 0.5)
        mask = model.transformer.model._prepare_decoder_attention_mask(
            torch.ones((1, steps), dtype=torch.bool, device="cuda"), (1, steps), x, 0
        )
        scores_masked = scores + mask
        probs = torch.nn.functional.softmax(scores_masked, dim=-1, dtype=torch.float32).to(q.dtype)
        attended = torch.matmul(probs, v)
        attended_flat = attended.transpose(1, 2).contiguous().reshape(1, steps, attention.hidden_size)
        projected = attention.o_proj(attended_flat)
        post_attention = x + projected
        ffn_norm = layer.post_attention_layernorm(post_attention)
        gate = layer.mlp.gate_proj(ffn_norm)
        gate_silu = layer.mlp.act_fn(gate)
        up = layer.mlp.up_proj(ffn_norm)
        product = gate_silu * up
        down = layer.mlp.down_proj(product)
        output = post_attention + down

        # Tensor dimensions are explicitly adjusted to mirror the C++ raw
        # stream: vector values are width-major and attention is [head,key].
        values = {
            "input": last_width(x),
            "attn_norm": last_width(norm),
            "q": last_width(q_raw),
            "k": last_width(k_raw),
            "v": last_width(v_raw),
            "q_rope": q_rotary[0, :, -1, :].reshape(-1).float().cpu().numpy(),
            "k_rope": k_rotary[0, :, -1, :].reshape(-1).float().cpu().numpy(),
            "attention_scores": scores[0, :, -1, :].reshape(-1).float().cpu().numpy(),
            "attention_probs": probs[0, :, -1, :].reshape(-1).float().cpu().numpy(),
            "attention_output": attended_flat[0, -1].float().cpu().numpy(),
            "o_proj": last_width(projected),
            "post_attention_residual": last_width(post_attention),
            "ffn_norm": last_width(ffn_norm),
            "gate": last_width(gate),
            "gate_silu": last_width(gate_silu),
            "up": last_width(up),
            "ffn_product": last_width(product),
            "down": last_width(down),
            "output": last_width(output),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **values)


if __name__ == "__main__":
    main()
