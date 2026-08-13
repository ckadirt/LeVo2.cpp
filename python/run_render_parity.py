#!/usr/bin/env python3
"""Drive one end-to-end renderer parity case.

This is the single implementation of the release-gate comparison: it draws the
shared initial noise, runs the official Flow/VAE oracle from
``export_renderer_oracles.py render``, converts the captured tensors into the
raw little-endian F32 files ``levo-render-parity`` consumes, and runs the native
renderer against them.

The stored noise tensor, not a seed, is the reproducibility boundary between
PyTorch and C++ (docs/renderer-parity.md), so both runtimes read the same bytes.
Latents are exported channel-major ``[W,64,1000]`` by the oracle and frame-major
``[W,1000,64]`` by the renderer; the transpose is applied here, once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_renderer_oracles import (  # noqa: E402
    FLOW_LATENT_DIM,
    FLOW_WINDOW_FRAMES,
    VAE_DOWNSAMPLING,
    _render_window_plan,
)

ORACLE = Path(__file__).resolve().parent / "export_renderer_oracles.py"
METRIC_RE = re.compile(
    r"^(?P<label>\S+) max_abs=(?P<max_abs>\S+) rmse=(?P<rmse>\S+) "
    r"rel_rms=(?P<rel_rms>\S+) cosine=(?P<cosine>\S+)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_memory_mib() -> int | None:
    """Return total device memory in use without adding a Python dependency."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return sum(values) if values else None
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def _run_measured(command: list[str]) -> tuple[subprocess.CompletedProcess[str], float, int | None]:
    """Run a release-gate subprocess and poll whole-device peak memory."""
    started = time.monotonic()
    peak = _gpu_memory_mib()
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.poll() is None:
        used = _gpu_memory_mib()
        if used is not None:
            peak = used if peak is None else max(peak, used)
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    used = _gpu_memory_mib()
    if used is not None:
        peak = used if peak is None else max(peak, used)
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        time.monotonic() - started,
        peak,
    )


def _parse_metrics(output: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        match = METRIC_RE.fullmatch(line.strip())
        if match is None:
            continue
        metrics[match.group("label")] = {
            name: float(match.group(name))
            for name in ("max_abs", "rmse", "rel_rms", "cosine")
        }
    return metrics


def _load_frames(tokens: Path) -> int:
    array = np.load(tokens, allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != 3 or array.shape[1] == 0:
        raise ValueError(f"{tokens} is not a canonical [3,T] token artifact")
    return int(array.shape[1])


def _write_noise(destination: Path, windows: int, seed: int) -> tuple[Path, Path]:
    """Write one noise tensor as both the oracle .npy and the native .f32."""
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal((windows, FLOW_WINDOW_FRAMES, FLOW_LATENT_DIM)).astype("<f4")
    npy_path = destination / "noise.npy"
    raw_path = destination / "noise.f32"
    np.save(npy_path, noise, allow_pickle=False)
    noise.tofile(raw_path)
    return npy_path, raw_path


def _write_references(captured: np.lib.npyio.NpzFile, destination: Path, windows: int) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    # The oracle keeps the Euler solution frame-major and only the
    # denormalization step permutes to the VAE's channel-major layout, so
    # exactly one of the two needs transposing to the renderer's [1000,64].
    frame_major = (windows, FLOW_WINDOW_FRAMES, FLOW_LATENT_DIM)
    channel_major = (windows, FLOW_LATENT_DIM, FLOW_WINDOW_FRAMES)
    for key, name, expected in (("final_latent_normalized", "normalized", frame_major),
                                ("final_latent_denormalized", "denormalized", channel_major)):
        latent = np.asarray(captured[key])
        if latent.shape != expected:
            raise ValueError(f"oracle {key} has shape {latent.shape}, expected {expected}")
        if expected == channel_major:
            latent = latent.transpose(0, 2, 1)
        path = destination / f"{name}.f32"
        np.ascontiguousarray(latent, dtype="<f4").tofile(path)
        paths[name] = path

    audio = np.asarray(captured["audio"])
    if audio.ndim == 3:  # [1, 2, samples]
        audio = audio[0]
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"oracle audio has shape {audio.shape}, expected channel-major stereo")
    paths["audio"] = destination / "audio.f32"
    np.ascontiguousarray(audio, dtype="<f4").tofile(paths["audio"])

    decoded_keys = sorted(key for key in captured.files if key.startswith("decoded_window_"))
    if len(decoded_keys) == windows:
        stacked = []
        for key in decoded_keys:
            window = np.asarray(captured[key])
            if window.ndim == 3:
                window = window[0]
            if window.shape != (2, FLOW_WINDOW_FRAMES * VAE_DOWNSAMPLING):
                raise ValueError(f"oracle {key} has shape {window.shape}")
            stacked.append(window)
        paths["decoded"] = destination / "decoded.f32"
        np.ascontiguousarray(np.stack(stacked), dtype="<f4").tofile(paths["decoded"])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=Path, required=True, help="canonical [3,T] int32 token .npy")
    parser.add_argument("--tool", type=Path, required=True, help="path to levo-render-parity")
    parser.add_argument("--flow-model", type=Path, required=True)
    parser.add_argument("--vae-model", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True, help="directory for noise, oracle, and references")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--reuse-oracle", action="store_true", help="skip the oracle run when its npz already exists")
    parser.add_argument("--wav-output", type=Path, default=None, help="also write the oracle WAV")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON summary of the run")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(args.tokens)
    padded, starts = _render_window_plan(frames)
    windows = len(starts)
    noise_npy, noise_raw = _write_noise(args.workdir, windows, args.seed)

    oracle_npz = args.workdir / "oracle.npz"
    oracle_seconds = 0.0
    if not (args.reuse_oracle and oracle_npz.is_file()):
        command = [sys.executable, str(ORACLE), "render", "--tokens", str(args.tokens), "--noise", str(noise_npy),
                   "--steps", str(args.steps), "--guidance", str(args.guidance), "--device", args.device,
                   "--output", str(oracle_npz)]
        if args.source_dir is not None:
            command.extend(["--source-dir", str(args.source_dir)])
        if args.runtime_dir is not None:
            command.extend(["--runtime-dir", str(args.runtime_dir)])
        if args.wav_output is not None:
            command.extend(["--wav-output", str(args.wav_output)])
        oracle_run, oracle_seconds, oracle_peak_gpu_mib = _run_measured(command)
        sys.stdout.write(oracle_run.stdout)
        sys.stderr.write(oracle_run.stderr)
        if oracle_run.returncode != 0:
            raise subprocess.CalledProcessError(oracle_run.returncode, command)
    else:
        oracle_peak_gpu_mib = None

    with np.load(oracle_npz, allow_pickle=False) as captured:
        references = _write_references(captured, args.workdir, windows)

    command = [str(args.tool), "--tokens", str(args.tokens), "--noise", str(noise_raw),
               "--flow-model", str(args.flow_model), "--vae-model", str(args.vae_model),
               "--backend", args.backend, "--steps", str(args.steps), "--cfg", str(args.guidance),
               "--latent-normalized", str(references["normalized"]),
               "--latent-denormalized", str(references["denormalized"]),
               "--audio", str(references["audio"])]
    if "decoded" in references:
        command.extend(["--decoded-windows", str(references["decoded"])])
    completed, native_seconds, native_peak_gpu_mib = _run_measured(command)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "format": "levo2-render-parity-report-v1",
            "tokens": {"filename": args.tokens.name, "sha256": _sha256(args.tokens)},
            "flow_model": {"filename": args.flow_model.name, "sha256": _sha256(args.flow_model)},
            "vae_model": {"filename": args.vae_model.name, "sha256": _sha256(args.vae_model)},
            "frames": frames,
            "padded_frames": int(padded),
            "windows": windows,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "backend": args.backend,
            "returncode": completed.returncode,
            "oracle_seconds": round(oracle_seconds, 3),
            "oracle_peak_gpu_mib": oracle_peak_gpu_mib,
            "native_seconds": round(native_seconds, 3),
            "native_peak_gpu_mib": native_peak_gpu_mib,
            "metrics": _parse_metrics(completed.stdout),
        }, indent=2, sort_keys=True) + "\n")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
