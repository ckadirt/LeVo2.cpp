"""Asset-free tests for renderer release-report parsing."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "python/run_render_parity.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_render_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_metrics_returns_structured_float_values() -> None:
    runner = _module()
    parsed = runner._parse_metrics(
        "normalized_latent[0] max_abs=0.001 rmse=2.5e-4 "
        "rel_rms=3e-4 cosine=0.999999\n"
        "windows=1 source_frames=50 samples_per_channel=96000 backend=CUDA0\n"
    )
    assert parsed == {
        "normalized_latent[0]": {
            "max_abs": 0.001,
            "rmse": 2.5e-4,
            "rel_rms": 3e-4,
            "cosine": 0.999999,
        }
    }
