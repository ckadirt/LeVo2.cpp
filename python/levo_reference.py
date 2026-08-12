"""Pinned Python reference helpers for LeVo2.cpp parity tests.

This module deliberately imports the official LeVo implementation rather than
copying model math. It also contains the one compatibility repair required by
the released v2-medium checkpoint/runtime pair: the checkpoint's type-info
embedding has 151652 rows while the shipped Qwen2 tokenizer exposes 151646 IDs.
The six tail rows are unreachable, but PyTorch still requires the module shape
to match before loading the state dict.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator

import torch
from omegaconf import OmegaConf


LEVO_SOURCE_REVISION = "653cbcf4716101834900c75b7d5da43b07e15d5b"
MODEL_REVISION = "7d91660ebfa041e29bace194f5631e775796f600"
MODEL_SHA256 = "4ef2be41f6d838824f5432491408f68d9ffbeda3b1349e1208f9cdfcc64445b1"
RUNTIME_REVISION = "cc258cc694a63114c61684cc26d0583b8ad777d0"


def _register_resolvers(levo_source: Path) -> None:
    resolvers = {
        "eval": lambda expression: eval(expression, {"__builtins__": {}}, {}),
        "concat": lambda *values: [item for value in values for item in value],
        "get_fname": lambda: "levo2-reference",
        "load_yaml": lambda value: list(
            OmegaConf.load(str(levo_source / value))
        ),
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


@contextlib.contextmanager
def _source_import_path(levo_source: Path) -> Iterator[None]:
    value = str(levo_source)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)


def build_reference_model(
    *,
    levo_source: Path,
    model_dir: Path,
    runtime_dir: Path,
    device: str = "cuda",
) -> tuple[torch.nn.Module, object]:
    """Build and load the exact released v2-medium LeLM reference model."""

    levo_source = levo_source.resolve()
    model_dir = model_dir.resolve()
    runtime_dir = runtime_dir.resolve()
    checkpoint_path = model_dir / "model.pt"
    config_path = model_dir / "config.yaml"
    tokenizer_path = runtime_dir / "third_party" / "Qwen2-7B"
    for required in (checkpoint_path, config_path, tokenizer_path / "tokenizer.json"):
        if not required.exists():
            raise FileNotFoundError(f"required reference input is missing: {required}")

    _register_resolvers(levo_source)
    previous_cwd = Path.cwd()
    with _source_import_path(levo_source):
        os.chdir(levo_source)
        try:
            from codeclm.models import builders

            cfg = OmegaConf.load(str(config_path))
            cfg.mode = "inference"
            cfg.lm.use_flash_attn_2 = False
            cfg.conditioners.description.QwTokenizer.token_path = str(tokenizer_path)
            cfg.conditioners.type_info.QwTextTokenizer.token_path = str(tokenizer_path)

            model = builders.get_lm_model(cfg)
        finally:
            os.chdir(previous_cwd)

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = {
        key.removeprefix("audiolm."): tensor
        for key, tensor in state.items()
        if key.startswith("audiolm.")
    }

    type_key = "condition_provider.conditioners.type_info.output_proj.weight"
    type_rows, type_dim = state[type_key].shape
    type_conditioner = model.condition_provider.conditioners.type_info
    if type_conditioner.output_proj.weight.shape != state[type_key].shape:
        if type_conditioner.output_proj.num_embeddings != 151646 or type_rows != 151652:
            raise RuntimeError(
                "unexpected type-info embedding mismatch: "
                f"runtime={tuple(type_conditioner.output_proj.weight.shape)}, "
                f"checkpoint={tuple(state[type_key].shape)}"
            )
        type_conditioner.output_proj = torch.nn.Embedding(
            type_rows,
            type_dim,
            padding_idx=151643,
        )

    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected state-dict result: {incompatible}")

    model.eval()
    model.to(device=device, dtype=torch.float16)
    return model, cfg


def make_no_prompt_tokens(device: str = "cuda") -> torch.Tensor:
    """Return the released model's no-audio-prompt input, shaped [1,3,250]."""

    return torch.full((1, 3, 250), 16385, dtype=torch.long, device=device)
