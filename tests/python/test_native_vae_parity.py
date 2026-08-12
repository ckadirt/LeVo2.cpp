"""Opt-in native decoder parity against official captured VAE stages."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[2]
SOURCE = Path(os.environ.get("LEVO_OFFICIAL_SOURCE", "/workspace/reference/LeVo"))
RUNTIME = Path(os.environ.get("LEVO_RUNTIME_DIR", "/workspace/models/SongGeneration-Runtime"))


@pytest.mark.parametrize("frames", [1, 2])
def test_native_vae_stage_parity(tmp_path: Path, frames: int) -> None:
    if os.environ.get("LEVO_RUN_NATIVE_VAE_PARITY") != "1":
        pytest.skip("set LEVO_RUN_NATIVE_VAE_PARITY=1 to run native VAE parity")
    tool = Path(os.environ.get("LEVO_VAE_PARITY_TOOL", ""))
    model = Path(os.environ.get("LEVO_VAE_F32_GGUF", "/tmp/LeVo2-v2-vae-F32.gguf"))
    if not tool.is_file() or not model.is_file():
        pytest.fail("set LEVO_VAE_PARITY_TOOL and provide LEVO_VAE_F32_GGUF")
    oracle = tmp_path / "oracle.npz"
    subprocess.run(
        [sys.executable, str(ROOT / "python/export_renderer_oracles.py"), "vae", "--frames", str(frames),
         "--device", "cuda", "--dtype", "float32", "--seed", "1234", "--source-dir", str(SOURCE),
         "--runtime-dir", str(RUNTIME), "--output", str(oracle)],
        check=True,
    )
    captured = np.load(oracle, allow_pickle=False)
    paths: dict[str, Path] = {}
    for name in ["latent_input", "audio", *[f"decoder_layer_{index:02d}" for index in range(1, 6)]]:
        path = tmp_path / f"{name}.f32"
        np.ascontiguousarray(captured[name], dtype="<f4").tofile(path)
        paths[name] = path
    command = [str(tool), "--model", str(model), "--frames", str(frames), "--backend", "cuda",
               "--latent", str(paths["latent_input"]), "--audio", str(paths["audio"])]
    for index in range(5):
        command.extend([f"--stage{index}", str(paths[f"decoder_layer_{index + 1:02d}"])])
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    assert "stage0 max_abs=" in completed.stdout
    assert "stage4 max_abs=" in completed.stdout
    assert "audio max_abs=" in completed.stdout
