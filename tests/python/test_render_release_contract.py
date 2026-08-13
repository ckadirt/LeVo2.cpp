"""Validate ignored renderer release fixtures against their tracked contract."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[2]
CONTRACT = Path(__file__).with_name("fixtures") / "render_release_contract.json"
DEFAULT_FIXTURES = ROOT / "artifacts/render-parity"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_contract_is_internally_consistent() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["format"] == "levo2-render-release-fixtures-v1"
    assert [fixture["frames"] for fixture in contract["fixtures"]] == [50, 252, 750, 1250]
    assert len({fixture["filename"] for fixture in contract["fixtures"]}) == 4
    for fixture in contract["fixtures"]:
        assert len(fixture["npy_sha256"]) == 64
        assert len(fixture["tensor_sha256"]) == 64


def test_local_release_fixtures_match_contract_when_available() -> None:
    directory = Path(os.environ.get("LEVO_RENDER_FIXTURE_DIR", DEFAULT_FIXTURES))
    if not directory.is_dir():
        pytest.skip("ignored renderer release fixtures are not present")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for fixture in contract["fixtures"]:
        path = directory / fixture["filename"]
        if not path.is_file():
            pytest.fail(f"release fixture is missing: {path}")
        array = np.load(path, allow_pickle=False)
        assert array.shape == (3, fixture["frames"])
        assert array.dtype == np.dtype("int32")
        assert _sha256(path) == fixture["npy_sha256"]
        assert hashlib.sha256(np.ascontiguousarray(array, dtype="<i4").tobytes()).hexdigest() == fixture["tensor_sha256"]
        arrays[fixture["filename"]] = array
    expected_long = np.tile(arrays["tokens-30s.npy"], (1, 2))[:, :1250]
    assert np.array_equal(arrays["tokens-50s.npy"], expected_long)
