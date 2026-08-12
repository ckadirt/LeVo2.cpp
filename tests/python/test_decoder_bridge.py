"""Focused integration test for the official LeVo v2 Flow/VAE bridge.

The decode-only runtime is about 5.5 GB and its upstream renderer always runs
a full 40-second internal Flow window. Set LEVO_RUN_DECODER_TEST=1 to opt into
this heavyweight test after the official assets have been installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
from decode_official import DEFAULT_RUNTIME_DIR, DEFAULT_SOURCE_DIR, missing_assets  # noqa: E402


def _write_artifact(path: Path) -> None:
    # The released renderer accepts arbitrary codebook IDs. One short, valid
    # frame is enough to exercise its internal 40-second Flow/VAE window while
    # keeping the written WAV at the requested 40 ms output duration.
    tokens = np.asarray([[12794], [7883], [12301]], dtype="<i4")
    np.save(path, tokens, allow_pickle=False)
    digest = hashlib.sha256(tokens.tobytes(order="C")).hexdigest()
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "levo2-token-artifact",
                "schema_version": 1,
                "artifact": {"type": "levo2-tokens", "format": "numpy-npy", "npy_version": "1.0"},
                "tensor": {"shape": [3, 1], "dtype": "int32", "order": "C", "sha256": digest},
                "duration": {"seconds": 1 / 25, "frames": 1, "frame_rate": 25},
                "config": {"sample_rate": 48000, "frame_rate": 25, "eos_token_id": 16384, "special_token_id": 16385},
                "provenance": {"runtime_revision": "cc258cc694a63114c61684cc26d0583b8ad777d0"},
            }
        ),
        encoding="utf-8",
    )


def test_official_dual_stream_decoder_bridge(tmp_path: Path) -> None:
    missing = missing_assets(DEFAULT_SOURCE_DIR, DEFAULT_RUNTIME_DIR)
    if missing:
        pytest.skip("official decoder assets unavailable: " + "; ".join(missing))
    if os.environ.get("LEVO_RUN_DECODER_TEST") != "1":
        pytest.skip("heavy official Flow/VAE test; set LEVO_RUN_DECODER_TEST=1 to run it")

    tokens = tmp_path / "tokens.npy"
    output = tmp_path / "render.wav"
    _write_artifact(tokens)
    steps = os.environ.get("LEVO_DECODER_TEST_STEPS", "1")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python/decode_official.py"),
            str(tokens),
            "--output",
            str(output),
            "--source-dir",
            str(DEFAULT_SOURCE_DIR),
            "--runtime-dir",
            str(DEFAULT_RUNTIME_DIR),
            "--steps",
            steps,
            "--seed",
            "1234",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "channels=2 sample_rate=48000" in result.stdout
    audio, sample_rate = sf.read(output, always_2d=True)
    assert sample_rate == 48_000
    assert audio.shape[1] == 2
    assert audio.shape[0] > 0
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 1e-6
    assert audio.shape[0] / sample_rate == pytest.approx(1 / 25, abs=2 / sample_rate)
