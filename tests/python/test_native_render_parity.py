"""Opt-in end-to-end token-to-WAV parity against the official renderer oracle.

This is the release gate of docs/renderer-plan.md. It is CUDA-only and needs the
pinned checkpoints plus both renderer GGUFs, so it is disabled unless
LEVO_RUN_NATIVE_RENDER_PARITY=1.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SOURCE = Path(os.environ.get("LEVO_OFFICIAL_SOURCE", "/workspace/reference/LeVo"))
RUNTIME = Path(os.environ.get("LEVO_RUNTIME_DIR", "/workspace/models/SongGeneration-Runtime"))
FIXTURES = Path(os.environ.get("LEVO_RENDER_FIXTURE_DIR", str(ROOT / "artifacts/render-parity")))

# The four release cases of docs/renderer-parity.md: one full window from a
# minimal input, the 10.08-second delay boundary, the complete 30-second
# fixture, and an input long enough for a second hop plus overlap.
CASES = [
    ("tokens-2s.npy", 50),
    ("tokens-10.08s.npy", 252),
    ("tokens-30s.npy", 750),
    ("tokens-50s.npy", 1250),
]


@pytest.mark.parametrize("steps", [1, 50])
@pytest.mark.parametrize("fixture,frames", CASES)
def test_native_render_parity(tmp_path: Path, fixture: str, frames: int, steps: int) -> None:
    if os.environ.get("LEVO_RUN_NATIVE_RENDER_PARITY") != "1":
        pytest.skip("set LEVO_RUN_NATIVE_RENDER_PARITY=1 to run end-to-end render parity")
    tool = Path(os.environ.get("LEVO_RENDER_PARITY_TOOL", ""))
    flow = Path(os.environ.get("LEVO_FLOW_F32_GGUF", "/tmp/LeVo2-v2-flow-F32.gguf"))
    vae = Path(os.environ.get("LEVO_VAE_F32_GGUF", "/tmp/LeVo2-v2-vae-F32.gguf"))
    tokens = FIXTURES / fixture
    if not tool.is_file() or not flow.is_file() or not vae.is_file():
        pytest.fail("set LEVO_RENDER_PARITY_TOOL and provide LEVO_FLOW_F32_GGUF and LEVO_VAE_F32_GGUF")
    if not tokens.is_file():
        pytest.fail(f"missing token fixture {tokens}; set LEVO_RENDER_FIXTURE_DIR")

    completed = subprocess.run(
        [sys.executable, str(ROOT / "python/run_render_parity.py"), "--tokens", str(tokens),
         "--tool", str(tool), "--flow-model", str(flow), "--vae-model", str(vae),
         "--workdir", str(tmp_path), "--steps", str(steps), "--guidance", "1.5",
         "--seed", "1234", "--backend", "cuda", "--device", "cuda",
         "--source-dir", str(SOURCE), "--runtime-dir", str(RUNTIME),
         "--report", str(_report_path(tmp_path, fixture, steps))],
        text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "normalized_latent[0] max_abs=" in completed.stdout
    assert "audio_left max_abs=" in completed.stdout
    assert f"source_frames={frames}" in completed.stdout
    assert f"samples_per_channel={frames * 1920}" in completed.stdout


def _report_path(tmp_path: Path, fixture: str, steps: int) -> Path:
    configured = os.environ.get("LEVO_RENDER_REPORT_DIR")
    directory = Path(configured) if configured else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{Path(fixture).stem}-steps-{steps}.json"
