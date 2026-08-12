"""Focused tests for the small official Flow/VAE renderer oracle fixtures.

Flow inventory is metadata-only and runs by default.  The VAE capture loads
the 644 MB official VAE checkpoint and is opt-in with
``LEVO_RUN_RENDERER_ORACLE_TEST=1``.  No oracle arrays are written into the
repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[2]
SOURCE_DIR = Path(os.environ.get("LEVO_OFFICIAL_SOURCE", "/workspace/reference/LeVo"))
RUNTIME_DIR = Path(os.environ.get("LEVO_RUNTIME_DIR", "/workspace/models/SongGeneration-Runtime"))
SCRIPT = REPO_ROOT / "python/export_renderer_oracles.py"
FLOW_CHECKPOINT = RUNTIME_DIR / "ckpt/model_septoken/model_2.safetensors"
VAE_CHECKPOINT = RUNTIME_DIR / "ckpt/vae/autoencoder_music_1320k.ckpt"


def _oracle_module():
    spec = importlib.util.spec_from_file_location("export_renderer_oracles", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*arguments: str, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *arguments,
            "--source-dir",
            str(SOURCE_DIR),
            "--runtime-dir",
            str(RUNTIME_DIR),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )


def test_flow_inventory_is_metadata_only(tmp_path: Path) -> None:
    if not FLOW_CHECKPOINT.is_file():
        pytest.skip(f"official Flow checkpoint unavailable: {FLOW_CHECKPOINT}")
    output = tmp_path / "flow-inventory.json"
    _run("flow", "--mode", "inventory", output=output)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["kind"] == "flow_inventory"
    assert manifest["checkpoint"]["tensor_count"] > 900
    assert manifest["checkpoint"]["sha256"] == "430b7c1c245722fbe3893cd621b3d4a90076404596e9fb1ce987a4a0f2a4fc6f"
    assert manifest["provenance"]["runtime_revision"]
    assert any(name.startswith("cfm_wrapper.estimator.") for name in manifest["tensors"])


def test_npz_writer_is_deterministic_and_records_array_statistics(tmp_path: Path) -> None:
    oracle = _oracle_module()
    arrays = {"b": np.asarray([[1, 2]], dtype=np.int32), "a": np.asarray([0.5, -0.5], dtype=np.float32)}
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    oracle._save_npz(first, arrays, {"kind": "unit"})
    oracle._save_npz(second, arrays, {"kind": "unit"})
    assert first.read_bytes() == second.read_bytes()
    manifest = json.loads(first.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["arrays"]["a"]["rms"] == pytest.approx(0.5)
    assert len(manifest["arrays"]["a"]["sha256"]) == 64


def test_renderer_dry_contract_validates_canonical_tokens_and_window_noise(tmp_path: Path) -> None:
    tokens = np.vstack([
        np.arange(1250, dtype=np.int32) % 16_384,
        (np.arange(1250, dtype=np.int32) + 1) % 16_384,
        (np.arange(1250, dtype=np.int32) + 2) % 16_384,
    ])
    noise = np.zeros((2, 1000, 64), dtype=np.float32)
    tokens_path = tmp_path / "tokens.npy"
    noise_path = tmp_path / "noise.npy"
    output = tmp_path / "renderer-contract.npz"
    np.save(tokens_path, tokens, allow_pickle=False)
    np.save(noise_path, noise, allow_pickle=False)
    _run(
        "render", "--dry-run", "--tokens", str(tokens_path), "--noise", str(noise_path), "--steps", "1",
        output=output,
    )
    values = np.load(output, allow_pickle=False)
    assert values["tokens"].shape == (3, 1250)
    assert values["window_major_noise"].shape == (2, 1000, 64)
    assert values["window_starts"].tolist() == [0, 750]
    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "renderer_contract"
    assert manifest["padded_frames"] == 1750
    assert manifest["window_count"] == 2


def test_renderer_dry_contract_rejects_noncanonical_noise(tmp_path: Path) -> None:
    tokens_path = tmp_path / "tokens.npy"
    noise_path = tmp_path / "noise.npy"
    output = tmp_path / "renderer-contract.npz"
    np.save(tokens_path, np.zeros((3, 1000), dtype=np.int32), allow_pickle=False)
    np.save(noise_path, np.zeros((1, 1000, 64), dtype=np.float64), allow_pickle=False)
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "render", "--dry-run",
            "--source-dir", str(SOURCE_DIR), "--runtime-dir", str(RUNTIME_DIR), "--device", "cpu",
            "--tokens", str(tokens_path), "--noise", str(noise_path), "--output", str(output),
        ], text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "must be F32" in (result.stderr + result.stdout)


def test_vae_tiny_capture_shape_and_finiteness(tmp_path: Path) -> None:
    if not VAE_CHECKPOINT.is_file():
        pytest.skip(f"official VAE checkpoint unavailable: {VAE_CHECKPOINT}")
    if os.environ.get("LEVO_RUN_RENDERER_ORACLE_TEST") != "1":
        pytest.skip("heavy official VAE oracle test; set LEVO_RUN_RENDERER_ORACLE_TEST=1 to run it")

    output = tmp_path / "vae-t1.npz"
    _run("vae", "--frames", "1", "--seed", "19", output=output)
    values = np.load(output, allow_pickle=False)
    assert values["latent_input"].shape == (1, 64, 1)
    assert values["audio"].shape == (1, 2, 1920)
    assert np.isfinite(values["latent_input"]).all()
    assert np.isfinite(values["audio"]).all()
    assert np.max(np.abs(values["audio"])) > 1e-6

    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "vae_stages"
    assert manifest["latent_frames"] == 1
    assert manifest["arrays"]["decoder_layer_01"]["shape"] == [1, 1024, 10]
    assert manifest["arrays"]["decoder_layer_05"]["shape"] == [1, 128, 1920]
