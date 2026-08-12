"""NumPy compatibility test for the canonical token artifact manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from python.validate_token_artifact import validate  # noqa: E402


def test_numpy_round_trip_and_manifest_validation(tmp_path: Path) -> None:
    values = np.array([[1, 2, 3], [4, 5, 6], [16384, 16385, 0]], dtype="<i4")
    path = tmp_path / "tokens.npy"
    np.save(path, values, allow_pickle=False)
    digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    (tmp_path / "tokens.json").write_text(
        json.dumps(
            {
                "format": "levo2-token-artifact",
                "schema_version": 1,
                "artifact": {"type": "levo2-tokens", "format": "numpy-npy", "npy_version": "1.0"},
                "tensor": {"shape": [3, 3], "dtype": "int32", "order": "C", "sha256": digest},
                "duration": {"seconds": 0.12, "frames": 3, "frame_rate": 25},
            }
        ),
        encoding="utf-8",
    )
    loaded = validate(path)
    assert np.array_equal(loaded, values)
    round_trip = tmp_path / "roundtrip.npy"
    np.save(round_trip, loaded, allow_pickle=False)
    assert np.array_equal(np.load(round_trip, allow_pickle=False), values)
