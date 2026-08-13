"""Static release gates for Cantor's self-contained engine archive contract."""
from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).parents[2] / ".github/workflows/engine-release.yml"


def test_engine_workflow_carries_all_required_backend_contracts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "backend: cpu",
        "backend: cuda12",
        "backend: vulkan",
        "humbletim/install-vulkan-sdk@v1.2",
        "-DGGML_CPU_ALL_VARIANTS=ON",
        "-DGGML_BACKEND_DL=ON",
        "libcantor_engine.so",
        "libggml-cpu*.so*",
        "libggml-cuda.so*",
        "libggml-vulkan.so*",
        "patchelf --set-rpath '$ORIGIN'",
        "env -u LD_LIBRARY_PATH python3",
        "grep -c '^cantor_engine_'",
        "levo2-backends-v1.fragment.json",
        "AWS_REQUEST_CHECKSUM_CALCULATION: when_required",
        "public, max-age=31536000, immutable",
    ):
        assert required in text


def test_engine_workflow_preserves_immutable_backend_identity() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'key="backends/${GITHUB_SHA::12}/${ARCHIVE}"' in text
    assert "refusing to overwrite immutable R2 engine archive" in text
    assert "commit[:12]" in text
    assert '"sha256": sha256' in text
    assert '"name": "levo2", "abi": 1' in text
