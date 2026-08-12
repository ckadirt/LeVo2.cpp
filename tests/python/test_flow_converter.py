"""Strict Flow renderer converter tests; no multi-gigabyte GGUF is written."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "convert"))
import convert_flow as flow  # noqa: E402


FIXTURE = Path(__file__).with_name("fixtures") / "flow_converter_contract.json"


def _minimal_infos() -> dict[str, flow.TensorInfo]:
    infos = {rule.source: flow.TensorInfo(rule.shape, "F32") for rule in flow.RULES}
    infos.update({rule.source: flow.TensorInfo(rule.shape, "F32") for rule in flow.RVQ_RULES})
    return infos


def test_rules_are_unique_and_full_required_inventory_is_explicit() -> None:
    infos = _minimal_infos()
    inventory = flow.inspect_inventory(infos)
    assert inventory["emitted_source_tensor_count"] == len(flow.RULES) + len(flow.RVQ_RULES)
    assert inventory["emitted_tensor_count"] == len(flow.RULES) + 6  # two g/v pairs materialize to two tensors
    assert inventory["emitted_parameter_count"] == 663_310_785
    assert {x["target"] for x in inventory["emitted"] if x["transform"].startswith("weight_norm")} == {
        "flow.rvq.vocal.out_proj.weight", "flow.rvq.bgm.out_proj.weight"
    }


def test_unknown_and_shape_mismatch_fail_closed() -> None:
    infos = _minimal_infos()
    infos["unexpected.weight"] = flow.TensorInfo((1,), "F32")
    with pytest.raises(ValueError, match="unclassified Flow checkpoint tensor"):
        flow.inspect_inventory(infos)

    infos = _minimal_infos()
    infos["mask_emb.weight"] = flow.TensorInfo((4, flow.MASK_DIM), "F32")
    with pytest.raises(ValueError, match="shape mismatch for mask_emb.weight"):
        flow.inspect_inventory(infos)

    infos = _minimal_infos()
    infos["mask_emb.weight"] = flow.TensorInfo((3, flow.MASK_DIM), "F16")
    with pytest.raises(ValueError, match="dtype mismatch for mask_emb.weight"):
        flow.inspect_inventory(infos)


def test_weight_norm_materialization_matches_definition() -> None:
    g = np.full((flow.CONDITION_DIM, 1, 1), 2.0, dtype=np.float32)
    v = np.zeros((flow.CONDITION_DIM, flow.CODEBOOK_DIM, 1), dtype=np.float32)
    v[:, 0, 0] = 3.0
    weight = flow._materialize_weight_norm(g, v)
    assert weight.shape == (flow.CONDITION_DIM, flow.CODEBOOK_DIM)
    assert np.all(weight[:, 0] == pytest.approx(2.0))
    assert np.count_nonzero(weight[:, 1:]) == 0


def test_pinned_checkpoint_header_inventory_when_available() -> None:
    checkpoint = os.environ.get("LEVO_FLOW_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set LEVO_FLOW_CHECKPOINT to inspect the pinned runtime checkpoint")
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manifest = flow.convert(Path(checkpoint), dry_run=True, hash_source=False)
    inventory = manifest["inventory"]
    expected_keys = {
        "source_tensor_count": "checkpoint_tensor_count",
        "emitted_source_tensor_count": "emitted_source_tensor_count",
        "emitted_tensor_count": "emitted_tensor_count",
        "omitted_source_tensor_count": "omitted_source_tensor_count",
    }
    for actual, fixture in expected_keys.items():
        assert inventory[actual] == expected[fixture]
    assert manifest["source"]["sha256"] is None
    assert manifest["source"]["header_sha256"] == expected["header_sha256"]
    assert inventory["emitted_parameter_count"] == expected["emitted_parameter_count"]
    assert any(x["target"] == "flow.position_embedding.weight" for x in inventory["emitted"])
    assert any(x["source"] == "cfm_wrapper.estimator.wte.weight" for x in inventory["omissions"])
