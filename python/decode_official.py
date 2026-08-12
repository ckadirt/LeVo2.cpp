#!/usr/bin/env python3
"""Render a canonical LeVo2.cpp token artifact with the official LeVo v2 decoder.

This is deliberately a thin adapter: it validates the ``[3, T]`` int32 NumPy
artifact produced by ``levo-cli``, selects its vocal and BGM streams, and calls
the released ``Flow1dVAESeparate`` renderer.  The Flow model and 48 kHz VAE
remain the upstream Python implementation; this script does not port or
reimplement any renderer math.

The current upstream renderer has a 40-second internal Flow window, even for
short token inputs.  Small inputs therefore still exercise the complete
renderer but produce only their requested output duration (25 tokens/sec).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = Path(os.environ.get("LEVO_OFFICIAL_SOURCE", "/workspace/reference/LeVo"))
DEFAULT_RUNTIME_DIR = Path(os.environ.get("LEVO_RUNTIME_DIR", "/workspace/models/SongGeneration-Runtime"))
RUNTIME_REVISION = "cc258cc694a63114c61684cc26d0583b8ad777d0"
SAMPLE_RATE = 48_000
FRAME_RATE = 25
CODEBOOK_SIZE = 16_384


def required_paths(source_dir: Path, runtime_dir: Path) -> dict[str, Path]:
    """Return the small, decode-only subset of the pinned official runtime."""

    return {
        "official Flow renderer": source_dir / "codeclm/tokenizer/Flow1dVAE/generate_septoken.py",
        "official VAE loader": source_dir / "codeclm/tokenizer/Flow1dVAE/tools/get_1dvae_large.py",
        "dual-stream Flow checkpoint": runtime_dir / "ckpt/model_septoken/model_2.safetensors",
        "VAE checkpoint": runtime_dir / "ckpt/vae/autoencoder_music_1320k.ckpt",
        "VAE config": runtime_dir / "ckpt/vae/stable_audio_1920_vae.json",
        "vendored stable-audio-tools": runtime_dir / "third_party/stable_audio_tools/stable_audio_tools/models/autoencoders.py",
        # Imported by generate_septoken.py even though decoding tokens does not
        # instantiate the separator or require its checkpoint.
        "Demucs import shim": runtime_dir / "third_party/demucs/models/pretrained.py",
    }


def missing_assets(source_dir: Path, runtime_dir: Path) -> list[str]:
    return [f"{label}: {path}" for label, path in required_paths(source_dir, runtime_dir).items() if not path.is_file()]


def _prepend_python_paths(paths: Iterable[Path]) -> None:
    for path in reversed([str(path.resolve()) for path in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_token_artifact(path: Path) -> tuple[np.ndarray, dict]:
    """Strictly validate the C++ NPY plus its required JSON companion."""

    try:
        from validate_token_artifact import validate
    except ModuleNotFoundError:
        _prepend_python_paths([REPO_ROOT / "python"])
        from validate_token_artifact import validate

    tokens = validate(path)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    duration = manifest.get("duration", {})
    if config.get("sample_rate") != SAMPLE_RATE or config.get("frame_rate") != FRAME_RATE:
        raise ValueError("token artifact is not configured for the released 48 kHz / 25 Hz LeVo v2 decoder")
    if duration.get("frames") != int(tokens.shape[1]) or duration.get("frame_rate") != FRAME_RATE:
        raise ValueError("token artifact duration metadata does not match its 25 Hz token tensor")
    if np.any(tokens >= CODEBOOK_SIZE):
        raise ValueError(
            "renderer input contains EOS/special tokens; trim generation at EOS before decoding "
            "so vocal and BGM token IDs are in [0, 16383]"
        )
    return tokens, manifest


def _load_official_renderer(source_dir: Path, runtime_dir: Path, device: str):
    missing = missing_assets(source_dir, runtime_dir)
    if missing:
        raise FileNotFoundError("official LeVo decoder assets are missing:\n  " + "\n  ".join(missing))

    flow_dir = source_dir / "codeclm/tokenizer/Flow1dVAE"
    stable_audio_dir = runtime_dir / "third_party/stable_audio_tools"
    # The released sources use a few top-level imports (model_septoken and
    # tools) and relative configuration paths. Preserve that import layout.
    _prepend_python_paths([runtime_dir, stable_audio_dir, source_dir, flow_dir])
    os.chdir(flow_dir)

    module = importlib.import_module("codeclm.tokenizer.audio_tokenizer")
    flow_path = runtime_dir / "ckpt/model_septoken/model_2.safetensors"
    vae_config = runtime_dir / "ckpt/vae/stable_audio_1920_vae.json"
    vae_model = runtime_dir / "ckpt/vae/autoencoder_music_1320k.ckpt"
    return module.AudioTokenizer.get_pretrained(
        f"Flow1dVAESeparate_{flow_path}",
        str(vae_config),
        str(vae_model),
        device=device,
        tango_device=device,
    )


def decode_tokens(
    tokens: np.ndarray,
    *,
    source_dir: Path,
    runtime_dir: Path,
    device: str,
    steps: int,
    chunked: bool,
    chunk_size: int,
    seed: int | None,
):
    """Run the released dual-stream Flow + VAE decoder for ``tokens``."""

    if steps <= 0:
        raise ValueError("--steps must be positive")
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA renderer requested but torch.cuda.is_available() is false")
    if seed is not None:
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

    renderer = _load_official_renderer(source_dir, runtime_dir, device)
    # LeVo's separate tokenizer decodes only streams 1 and 2. Stream 0 is the
    # mixed autoregressive control stream and is intentionally not rendered.
    vocal = torch.from_numpy(np.ascontiguousarray(tokens[1], dtype=np.int64)).unsqueeze(0).unsqueeze(0)
    bgm = torch.from_numpy(np.ascontiguousarray(tokens[2], dtype=np.int64)).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        # This is the released Tango.code2sound implementation that
        # Flow1dVAESeparate.decode invokes. The production default is 50.
        audio = renderer.model.code2sound(
            [vocal, bgm],
            guidance_scale=1.5,
            num_steps=steps,
            disable_progress=True,
            chunked=chunked,
            chunk_size=chunk_size,
        )
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise RuntimeError(f"official renderer returned {tuple(audio.shape)}, expected [2, samples]")
    if not torch.isfinite(audio).all():
        raise RuntimeError("official renderer produced non-finite samples")
    return audio.detach().cpu().float()


def write_wav(output: Path, audio) -> None:
    import torchaudio

    if output.suffix.lower() != ".wav":
        raise ValueError("--output must use the .wav extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), audio, SAMPLE_RATE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", type=Path, help="canonical levo-cli int32 [3,T] .npy artifact")
    parser.add_argument("--output", required=True, type=Path, help="destination stereo 48 kHz .wav")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="pinned official LeVo source checkout")
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR, help="pinned SongGeneration-Runtime directory")
    parser.add_argument("--device", default="cuda", help="renderer device (default: cuda)")
    parser.add_argument("--steps", type=int, default=50, help="official Flow Euler steps (default: 50)")
    parser.add_argument("--chunked", action="store_true", help="enable the upstream VAE chunked decoder")
    parser.add_argument("--chunk-size", type=int, default=128, help="upstream VAE chunk size when --chunked is set")
    parser.add_argument("--seed", type=int, help="optional torch RNG seed for reproducible Flow noise")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokens_path = args.tokens.resolve()
    output = args.output.resolve()
    source_dir = args.source_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    tokens, manifest = load_token_artifact(tokens_path)
    audio = decode_tokens(
        tokens,
        source_dir=source_dir,
        runtime_dir=runtime_dir,
        device=args.device,
        steps=args.steps,
        chunked=args.chunked,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    write_wav(output, audio)
    duration = audio.shape[1] / SAMPLE_RATE
    print(
        f"rendered {output}: channels=2 sample_rate={SAMPLE_RATE} samples={audio.shape[1]} "
        f"duration={duration:.3f}s frames={tokens.shape[1]} runtime_revision={manifest.get('provenance', {}).get('runtime_revision', '')}"
    )


if __name__ == "__main__":
    main()
