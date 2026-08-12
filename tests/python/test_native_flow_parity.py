"""Opt-in F32 native Flow estimator parity at the frozen two-frame boundary."""
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


def test_native_flow_estimator_velocity_t2(tmp_path: Path) -> None:
    if os.environ.get("LEVO_RUN_NATIVE_FLOW_PARITY") != "1":
        pytest.skip("set LEVO_RUN_NATIVE_FLOW_PARITY=1 to run the native Flow estimator gate")
    tool = Path(os.environ.get("LEVO_FLOW_PARITY_TOOL", ""))
    model = Path(os.environ.get("LEVO_FLOW_F32_GGUF", "/tmp/LeVo2-v2-flow-F32.gguf"))
    if not tool.is_file() or not model.is_file():
        pytest.fail("set LEVO_FLOW_PARITY_TOOL and provide LEVO_FLOW_F32_GGUF")
    work = tmp_path
    oracle = work / "flow.npz"
    subprocess.run([sys.executable, str(ROOT / "python/export_renderer_oracles.py"), "flow", "--mode", "velocity", "--frames", "2", "--time", "0.5", "--guidance", "1.5", "--seed", "1234", "--device", "cuda", "--source-dir", str(SOURCE), "--runtime-dir", str(RUNTIME), "--output", str(oracle)], check=True)
    values = np.load(oracle, allow_pickle=False)
    paths: dict[str, Path] = {}
    for name in ["model_input", "timestep_embedding", "timestep_modulation", "block0_input", "block0_output", "full_output", "velocity"]:
        path = work / f"{name}.f32"; np.ascontiguousarray(values[name], dtype="<f4").tofile(path); paths[name] = path
    time_path = work / "timesteps.f32"; np.asarray([0.5, 0.5], dtype="<f4").tofile(time_path)
    command = [str(tool), "--model", str(model), "--frames", "2", "--backend", "cuda", "--guidance", "1.5", "--model-input", str(paths["model_input"]), "--timesteps", str(time_path), "--timestep-embedding", str(paths["timestep_embedding"]), "--timestep-modulation", str(paths["timestep_modulation"]), "--block0-input", str(paths["block0_input"]), "--block0-output", str(paths["block0_output"]), "--full-output", str(paths["full_output"]), "--velocity", str(paths["velocity"])]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert "timestep_embedding max_abs=" in completed.stdout
    assert "block0_output max_abs=" in completed.stdout
    assert "velocity max_abs=" in completed.stdout
    assert completed.returncode == 0, completed.stdout + completed.stderr
