"""Guard the committed v0.2 release evidence against its frozen policy."""
from __future__ import annotations

import json
from pathlib import Path


MATRIX = Path(__file__).parents[2] / "docs/renderer-release-matrix.json"


def test_release_matrix_is_complete_and_within_frozen_gates() -> None:
    report = json.loads(MATRIX.read_text(encoding="utf-8"))
    cases = report["cases"]
    assert {(case["frames"], case["steps"]) for case in cases} == {
        (frames, steps) for frames in (50, 252, 750, 1250) for steps in (1, 50)
    }
    thresholds = report["thresholds"]
    for case in cases:
        assert case["latent_max_abs"] <= thresholds["latent_max_abs"]
        assert case["latent_rel_rms"] <= thresholds["latent_rel_rms"]
        assert case["audio_max_abs"] <= thresholds["audio_max_abs"]
        assert case["audio_rel_rms"] <= thresholds["audio_rel_rms"]
        assert case["native_seconds"] > 0
        assert case["native_peak_gpu_mib"] > 0

    smoke = report["production_smoke"]
    assert smoke["finite"] is True
    assert smoke["sample_rate"] == 48000
    assert smoke["samples_per_channel"] == 750 * 1920
    assert smoke["channels"] == 2
    assert smoke["subtype"] == "FLOAT"
    assert smoke["rms"] > 0
