"""The local R2 publisher lock must reject a concurrent mutable publication."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
UPLOADER = ROOT / "tools/upload_ckpts.py"


def _uploader_module():
    spec = importlib.util.spec_from_file_location("levo2_r2_lock_uploader", UPLOADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_publication_lock_is_non_blocking_and_exclusive(tmp_path: Path) -> None:
    uploader = _uploader_module()
    manifest = tmp_path / "manifest.json"
    with uploader.publication_lock(manifest):
        with pytest.raises(uploader.PublicationError, match="already using"):
            with uploader.publication_lock(manifest):
                pass
