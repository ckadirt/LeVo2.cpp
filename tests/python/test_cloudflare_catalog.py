"""Lock the generated LeVo2 Cantor catalog to its frozen release contract."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
CATALOG_PATH = ROOT / "docs/cloudflare-catalog-v1.json"
UPLOADER_PATH = ROOT / "tools/upload_ckpts.py"


def _uploader_module():
    spec = importlib.util.spec_from_file_location("levo2_r2_uploader", UPLOADER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_committed_catalog_is_the_uploader_output() -> None:
    uploader = _uploader_module()
    assert _catalog() == uploader.catalog()


def test_catalog_has_exact_engine_roles_and_digest_format() -> None:
    catalog = _catalog()
    assert catalog["schema"] == 1
    assert len(catalog["models"]) == 1
    model = catalog["models"][0]
    assert model["name"] == model["engine"] == "levo2"
    assert [variant["tag"] for variant in model["variants"]] == [
        "1.0-fast",
        "1.0-balanced",
        "1.0-quality",
    ]
    digest = re.compile(r"sha256:[0-9a-f]{64}\Z")
    for variant in model["variants"]:
        assert [component["role"] for component in variant["components"]] == ["lm", "dit", "vae"]
        assert variant["needs"]["backends"] == ["cuda12", "vulkan", "cpu"]
        for component in variant["components"]:
            assert digest.fullmatch(component["blob"])
            assert component["url"].startswith("https://cantor-ckpts.ckadirt.xyz/levo2-1.0/")
            assert component["bytes"] > 0


def test_only_lm_changes_between_starter_variants() -> None:
    variants = _catalog()["models"][0]["variants"]
    shared_by_role = {
        role: {variant["components"][index]["blob"] for variant in variants}
        for index, role in enumerate(("lm", "dit", "vae"))
    }
    assert len(shared_by_role["lm"]) == 3
    assert len(shared_by_role["dit"]) == 1
    assert len(shared_by_role["vae"]) == 1
