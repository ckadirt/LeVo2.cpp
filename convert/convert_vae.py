#!/usr/bin/env python3
"""Strictly convert the released LeVo Oobleck VAE *decoder* to GGUF.

The released checkpoint also contains an encoder, but LeVo rendering only
consumes Flow-produced VAE latents.  This converter deliberately emits the
reachable decoder alone.  It is intentionally an allow-list converter: every
one of the 365 source tensors must be classified and every source shape must
match the pinned Oobleck topology before an artifact can be written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RUNTIME_REPO = "lglg666/SongGeneration-Runtime"
RUNTIME_REVISION = "cc258cc694a63114c61684cc26d0583b8ad777d0"
LEVO_REPO = "https://github.com/levo-demo/LeVo"
LEVO_REVISION = "653cbcf4716101834900c75b7d5da43b07e15d5b"
GGML_REPO = "https://github.com/ggml-org/ggml"
GGML_REVISION = "8846b79e66747bb9f68597420e95114c177315ce"
CONVERTER_VERSION = "1"
SCHEMA_VERSION = 1

SAMPLE_RATE = 48_000
AUDIO_CHANNELS = 2
LATENT_DIM = 64
CHANNELS = 128
C_MULTS = (1, 2, 4, 8, 16)
STRIDES = (2, 4, 4, 6, 10)
DOWNSAMPLING_RATIO = 1920


@dataclass(frozen=True)
class ConvSpec:
    """Raw legacy PyTorch weight-norm source and folded GGUF destinations."""

    source: str
    weight: str
    bias: str | None
    shape: tuple[int, int, int]
    transpose: bool = False


def _conv_source_shapes(spec: ConvSpec) -> dict[str, tuple[int, ...]]:
    # torch.nn.utils.weight_norm uses dim=0 by default for both Conv1d and
    # ConvTranspose1d.  Thus the scalar dimension is source shape[0] even for
    # ConvTranspose1d, whose native layout is [Cin, Cout, kernel].
    return {
        f"{spec.source}.weight_v": spec.shape,
        f"{spec.source}.weight_g": (spec.shape[0], 1, 1),
        **({f"{spec.source}.bias": (spec.shape[1] if spec.transpose else spec.shape[0],)} if spec.bias else {}),
    }


def _decoder_specs() -> tuple[list[ConvSpec], dict[str, str], dict[str, tuple[int, ...]]]:
    """Return folded convolution specs, Snake mappings, and raw source shapes."""
    convs: list[ConvSpec] = []
    snakes: dict[str, str] = {}
    source_shapes: dict[str, tuple[int, ...]] = {}

    def add_conv(source: str, weight: str, bias: str | None, shape: tuple[int, int, int], *, transpose: bool = False) -> None:
        spec = ConvSpec(source, weight, bias, shape, transpose)
        convs.append(spec)
        source_shapes.update(_conv_source_shapes(spec))

    def add_snake(source: str, target: str, channels: int) -> None:
        snakes[f"{source}.alpha"] = f"{target}.alpha_log"
        snakes[f"{source}.beta"] = f"{target}.beta_log"
        source_shapes[f"{source}.alpha"] = (channels,)
        source_shapes[f"{source}.beta"] = (channels,)

    add_conv("decoder.layers.0", "vae.decoder.input.weight", "vae.decoder.input.bias", (2048, LATENT_DIM, 7))
    widths = (2048, 1024, 512, 256, 128)
    # Decoder construction reverses config strides: 10, 6, 4, 4, 2.
    for stage, (cin, cout, stride) in enumerate(zip(widths, widths[1:] + (128,), reversed(STRIDES), strict=True)):
        base = f"decoder.layers.{stage + 1}.layers"
        target = f"vae.decoder.stage.{stage}"
        add_snake(f"{base}.0", f"{target}.upsample.activation", cin)
        add_conv(f"{base}.1", f"{target}.upsample.weight", f"{target}.upsample.bias", (cin, cout, 2 * stride), transpose=True)
        for residual, dilation in enumerate((1, 3, 9)):
            rbase = f"{base}.{residual + 2}.layers"
            rtarget = f"{target}.residual.{residual}"
            add_snake(f"{rbase}.0", f"{rtarget}.pre", cout)
            add_conv(f"{rbase}.1", f"{rtarget}.conv.0.weight", f"{rtarget}.conv.0.bias", (cout, cout, 7))
            add_snake(f"{rbase}.2", f"{rtarget}.post", cout)
            add_conv(f"{rbase}.3", f"{rtarget}.conv.1.weight", f"{rtarget}.conv.1.bias", (cout, cout, 1))
    add_snake("decoder.layers.6", "vae.decoder.output", 128)
    add_conv("decoder.layers.7", "vae.decoder.output.weight", None, (AUDIO_CHANNELS, 128, 7))
    return convs, snakes, source_shapes


DECODER_CONVS, DECODER_SNAKES, DECODER_SOURCE_SHAPES = _decoder_specs()


def _encoder_source_shapes() -> dict[str, tuple[int, ...]]:
    """Exact source schema for unreachable encoder tensors, for classification."""
    out: dict[str, tuple[int, ...]] = {}

    def add_conv(source: str, shape: tuple[int, int, int]) -> None:
        out.update({
            f"{source}.weight_v": shape,
            f"{source}.weight_g": (shape[0], 1, 1),
            f"{source}.bias": (shape[0],),
        })

    def add_snake(source: str, width: int) -> None:
        out[f"{source}.alpha"] = (width,)
        out[f"{source}.beta"] = (width,)

    add_conv("encoder.layers.0", (128, AUDIO_CHANNELS, 7))
    widths = (128, 128, 256, 512, 1024, 2048)
    for block, (cin, cout, stride) in enumerate(zip(widths[:-1], widths[1:], STRIDES, strict=True)):
        base = f"encoder.layers.{block + 1}.layers"
        for residual in range(3):
            rbase = f"{base}.{residual}.layers"
            add_snake(f"{rbase}.0", cin)
            add_conv(f"{rbase}.1", (cin, cin, 7))
            add_snake(f"{rbase}.2", cin)
            add_conv(f"{rbase}.3", (cin, cin, 1))
        add_snake(f"{base}.3", cin)
        add_conv(f"{base}.4", (cout, cin, 2 * stride))
    add_snake("encoder.layers.6", 2048)
    add_conv("encoder.layers.7", (128, 2048, 3))
    return out


ENCODER_SOURCE_SHAPES = _encoder_source_shapes()
EXPECTED_SOURCE_SHAPES = {**DECODER_SOURCE_SHAPES, **ENCODER_SOURCE_SHAPES}
assert len(DECODER_SOURCE_SHAPES) == 182
assert len(ENCODER_SOURCE_SHAPES) == 183
assert len(EXPECTED_SOURCE_SHAPES) == 365


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CLI diagnostic
        raise RuntimeError("PyTorch is required to load the VAE checkpoint") from exc
    return torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict(path: Path) -> dict[str, Any]:
    """Safely load the sole tensor mapping accepted by this converter."""
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("VAE checkpoint must be a mapping containing tensor state_dict")
    state: dict[str, Any] = {}
    for key, value in payload["state_dict"].items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint contains non-tensor state entry {key!r}")
        if key in state:
            raise ValueError(f"duplicate checkpoint key {key!r}")
        state[key] = value
    return state


def _as_f32(value: Any, source: str) -> np.ndarray:
    if not hasattr(value, "detach"):
        raise ValueError(f"checkpoint value {source!r} is not a tensor")
    tensor = value.detach().cpu()
    if str(tensor.dtype) != "torch.float32":
        raise ValueError(f"dtype mismatch for {source}: got {tensor.dtype}, expected torch.float32")
    array = np.ascontiguousarray(tensor.numpy())
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values in {source!r}")
    return array


def _fold_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray, spec: ConvSpec) -> np.ndarray:
    if tuple(weight_v.shape) != spec.shape:
        raise ValueError(f"shape mismatch for {spec.source}.weight_v: got {tuple(weight_v.shape)}, expected {spec.shape}")
    expected_g = (spec.shape[0], 1, 1)
    if tuple(weight_g.shape) != expected_g:
        raise ValueError(f"shape mismatch for {spec.source}.weight_g: got {tuple(weight_g.shape)}, expected {expected_g}")
    # Legacy torch.nn.utils.weight_norm defaults to dim=0.  For ConvTranspose1d
    # this deliberately means per-input-channel normalization because raw
    # PyTorch weight layout is [Cin, Cout, K], not [Cout, Cin, K].
    norm = np.sqrt(np.sum(np.square(weight_v, dtype=np.float32), axis=(1, 2), keepdims=True, dtype=np.float32))
    if np.any(norm == 0):
        raise ValueError(f"zero weight-norm magnitude in {spec.source}.weight_v")
    return np.ascontiguousarray(weight_v * (weight_g / norm), dtype=np.float32)


def inspect_state_dict(state: Mapping[str, Any], *, strict_shapes: bool = True) -> tuple[dict[str, np.ndarray], list[dict[str, str]], list[str]]:
    """Classify, validate, fold, and map a state dict without writing GGUF."""
    normalized: dict[str, Any] = {}
    for source, value in state.items():
        if not isinstance(source, str):
            raise ValueError("checkpoint tensor names must be strings")
        if source in normalized:
            raise ValueError(f"duplicate checkpoint key {source!r}")
        normalized[source] = value

    unknown = sorted(set(normalized) - set(EXPECTED_SOURCE_SHAPES))
    if unknown:
        raise ValueError(f"unclassified checkpoint tensor {unknown[0]!r}; update the explicit VAE allowlist")
    if strict_shapes:
        missing = sorted(set(EXPECTED_SOURCE_SHAPES) - set(normalized))
        if missing:
            raise ValueError(f"missing required checkpoint tensor {missing[0]!r}")

    arrays: dict[str, np.ndarray] = {}
    for source in sorted(normalized):
        array = _as_f32(normalized[source], source)
        expected = EXPECTED_SOURCE_SHAPES[source]
        if strict_shapes and tuple(array.shape) != expected:
            raise ValueError(f"shape mismatch for {source}: got {tuple(array.shape)}, expected {expected}")
        arrays[source] = array

    emitted: dict[str, np.ndarray] = {}
    for spec in DECODER_CONVS:
        keys = _conv_source_shapes(spec)
        if not set(keys).issubset(arrays):
            if strict_shapes:
                raise AssertionError("missing source should already have been rejected")
            continue
        emitted[spec.weight] = _fold_weight_norm(arrays[f"{spec.source}.weight_v"], arrays[f"{spec.source}.weight_g"], spec)
        if spec.bias is not None:
            emitted[spec.bias] = np.ascontiguousarray(arrays[f"{spec.source}.bias"], dtype=np.float32)
    for source, target in DECODER_SNAKES.items():
        if source in arrays:
            emitted[target] = np.ascontiguousarray(arrays[source], dtype=np.float32)

    omissions = [
        {"source": source, "reason": "encoder tensor is unreachable in decoder-only LeVo rendering"}
        for source in sorted(ENCODER_SOURCE_SHAPES) if source in arrays
    ]
    if strict_shapes and len(emitted) != 145:
        raise ValueError(f"decoder mapping emitted {len(emitted)} tensors, expected 145")
    return emitted, omissions, sorted(normalized)


def _read_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    decoder = config.get("model", {}).get("decoder", {})
    expected = {
        "out_channels": AUDIO_CHANNELS, "channels": CHANNELS, "c_mults": list(C_MULTS),
        "strides": list(STRIDES), "latent_dim": LATENT_DIM, "use_snake": True,
        "final_tanh": False,
    }
    actual = decoder.get("config")
    if not isinstance(actual, Mapping):
        raise ValueError("VAE config is missing model.decoder.config")
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"VAE config mismatch for decoder.{key}: got {actual.get(key)!r}, expected {value!r}")
    if config.get("sample_rate") != SAMPLE_RATE or config.get("model", {}).get("downsampling_ratio") != DOWNSAMPLING_RATIO:
        raise ValueError("VAE config sample rate or downsampling ratio differs from released decoder contract")
    return config


def _add_metadata(writer: Any, checkpoint_sha256: str, config_sha256: str, file_type: str) -> None:
    writer.add_name("LeVo2 Oobleck VAE decoder")
    writer.add_file_type(1 if file_type == "F16" else 0)
    writer.add_string("general.license", "Tencent AI Lab academic/research/education only")
    writer.add_string("general.source.repo_url", "https://huggingface.co/" + RUNTIME_REPO)
    writer.add_string("levo2.converter", "convert_vae.py")
    writer.add_string("levo2.converter.version", CONVERTER_VERSION)
    writer.add_uint32("levo2.schema_version", SCHEMA_VERSION)
    writer.add_string("levo2.source.runtime_repository", RUNTIME_REPO)
    writer.add_string("levo2.source.runtime_revision", RUNTIME_REVISION)
    writer.add_string("levo2.source.levo_repository", LEVO_REPO)
    writer.add_string("levo2.source.levo_revision", LEVO_REVISION)
    writer.add_string("levo2.source.ggml_repository", GGML_REPO)
    writer.add_string("levo2.source.ggml_revision", GGML_REVISION)
    writer.add_string("levo2.source.checkpoint_sha256", checkpoint_sha256)
    writer.add_string("levo2.source.config_sha256", config_sha256)
    ints = {
        "sample_rate": SAMPLE_RATE, "audio_channels": AUDIO_CHANNELS,
        "latent_dim": LATENT_DIM, "channels": CHANNELS,
        "downsampling_ratio": DOWNSAMPLING_RATIO, "decoder.stage_count": 5,
        "decoder.residual_units_per_stage": 3,
    }
    for key, value in ints.items():
        writer.add_uint32(f"levo2.vae.{key}", value)
    writer.add_array("levo2.vae.c_mults", list(C_MULTS))
    writer.add_array("levo2.vae.strides", list(STRIDES))
    writer.add_bool("levo2.vae.use_snake_beta", True)
    writer.add_bool("levo2.vae.final_tanh", False)
    writer.add_bool("levo2.vae.soft_clip", False)
    writer.add_bool("levo2.vae.weight_norm_folded", True)
    writer.add_string("levo2.vae.parameter_dtype", file_type)
    writer.add_uint32("levo2.vae.source_tensor_count", 365)
    writer.add_uint32("levo2.vae.decoder_source_tensor_count", 182)
    writer.add_uint32("levo2.vae.tensor_count", 145)


def convert(checkpoint: Path, config: Path, output: Path, *, dtype: str = "F32", dry_run: bool = False, strict_shapes: bool = True, gguf_python: Path | None = None) -> dict[str, Any]:
    if dtype not in ("F16", "F32"):
        raise ValueError("dtype must be F16 or F32")
    _read_and_validate_config(config)
    source = load_state_dict(checkpoint)
    tensors, omissions, source_keys = inspect_state_dict(source, strict_shapes=strict_shapes)
    manifest: dict[str, Any] = {
        "converter": "convert_vae.py", "converter_version": CONVERTER_VERSION,
        "schema_version": SCHEMA_VERSION, "dtype": dtype,
        "source": {
            "repository": RUNTIME_REPO, "revision": RUNTIME_REVISION,
            "checkpoint": {"filename": checkpoint.name, "bytes": checkpoint.stat().st_size, "sha256": _sha256(checkpoint)},
            "config": {"filename": config.name, "bytes": config.stat().st_size, "sha256": _sha256(config)},
        },
        "source_tensor_count": len(source_keys), "emitted_tensor_count": len(tensors),
        "omitted_tensor_count": len(omissions), "omissions": omissions,
        "parameter_count": int(sum(tensor.size for tensor in tensors.values())),
        "tensors": [{"name": name, "shape": list(tensor.shape), "dtype": dtype} for name, tensor in sorted(tensors.items())],
    }
    if dry_run:
        return manifest
    if gguf_python is not None:
        sys.path.insert(0, str(gguf_python))
    from gguf import GGUFWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output), "levo2_vae")
    _add_metadata(writer, manifest["source"]["checkpoint"]["sha256"], manifest["source"]["config"]["sha256"], dtype)
    for name, tensor in sorted(tensors.items()):
        writer.add_tensor(name, np.ascontiguousarray(tensor.astype(np.float16 if dtype == "F16" else np.float32, copy=False)))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    manifest["artifact"] = {"filename": output.name, "bytes": output.stat().st_size, "sha256": _sha256(output)}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{manifest['artifact']['sha256']}  {output.name}\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="autoencoder_music_1320k.ckpt")
    parser.add_argument("--config", required=True, type=Path, help="stable_audio_1920_vae.json")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--dtype", choices=("F16", "F32"), default="F32")
    parser.add_argument("--dry-run", "--list", action="store_true", help="validate and print inventory without writing GGUF")
    parser.add_argument("--gguf-python", type=Path)
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    manifest = convert(args.checkpoint, args.config, args.output or Path("unused.gguf"), dtype=args.dtype, dry_run=args.dry_run, gguf_python=args.gguf_python)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
