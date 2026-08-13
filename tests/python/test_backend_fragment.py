"""Keep the manual Cantor backend handoff accurate while CUDA is pending."""
from __future__ import annotations

import json
from pathlib import Path


FRAGMENT = Path(__file__).parents[2] / "docs/levo2-backends-v1.fragment.json"


def test_cpu_and_vulkan_handoff_values_are_frozen() -> None:
    fragment = json.loads(FRAGMENT.read_text(encoding="utf-8"))
    assert fragment["name"] == "levo2"
    assert fragment["abi"] == 1
    backends = {item["backend"]: item for item in fragment["backends"]}
    assert backends["cpu"] == {
        "backend": "cpu",
        "arch": "x86_64",
        "url": "https://cantor-ckpts.ckadirt.xyz/backends/b406beb25d12/levo2-engine-cpu-x86_64.tar.gz",
        "sha256": "868c6429a4d232139149a038f6ddfb834aadf4b8ffd4152d7f45e6b8ed06afba",
        "bytes": 6356609,
    }
    assert backends["vulkan"] == {
        "backend": "vulkan",
        "arch": "x86_64",
        "url": "https://cantor-ckpts.ckadirt.xyz/backends/b406beb25d12/levo2-engine-vulkan-x86_64.tar.gz",
        "sha256": "1d6e5fae44873421e6f4c15270e1826184738eda68b42a357bee148790b9e192",
        "bytes": 22315403,
    }


def test_cuda_is_an_explicit_placeholder_not_a_fake_release() -> None:
    fragment = json.loads(FRAGMENT.read_text(encoding="utf-8"))
    cuda = next(item for item in fragment["backends"] if item["backend"] == "cuda12")
    assert cuda["arch"] == "x86_64"
    assert cuda["url"].endswith("/levo2-engine-cuda12-x86_64.tar.gz")
    assert cuda["sha256"] == "REPLACE_WITH_CUDA12_TARBALL_SHA256"
    assert cuda["bytes"] == 0
