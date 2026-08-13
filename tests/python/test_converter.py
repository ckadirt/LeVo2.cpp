"""Synthetic converter tests; no SongGeneration checkpoint is needed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[2] / "convert"))
import levo2_to_gguf as conv  # noqa: E402


def _tokenizer(path: Path) -> None:
    path.mkdir()
    (path / "vocab.json").write_text(json.dumps({"a": 0, "b": 1, "c": 2}), encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (path / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 3, "content": "<|endoftext|>"}]}), encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_allowlist_maps_and_classifies_unused_tensor() -> None:
    state = {
        "emb.0.weight": torch.zeros(3, 4),
        "transformer.lm_head.weight": torch.ones(2, 4),
        "layer2_emb.0.weight": torch.zeros(3, 4),
    }
    mapped, omitted, keys = conv.inspect_state_dict(state, strict_shapes=False)
    assert set(mapped) == {"token_embd.mixed", "output.mixed"}
    assert omitted[0]["source"] == "audiolm.layer2_emb.0.weight"
    assert keys == sorted(keys)


def test_unknown_audiolm_tensor_is_fatal() -> None:
    with pytest.raises(ValueError, match="unclassified checkpoint tensor"):
        conv.inspect_state_dict({"audiolm.new_parameter.weight": torch.zeros(1)}, strict_shapes=False)


def test_released_v2_model_specs_have_distinct_complete_inventories() -> None:
    assert set(conv.MODEL_SPECS) == {"v2-medium", "v2-large"}
    assert conv.V2_MEDIUM.tensor_count == 380
    assert conv.V2_LARGE.tensor_count == 452
    assert conv.V2_LARGE.width == 2048
    assert conv.V2_LARGE.ffn == 11008
    assert conv.V2_LARGE.main_blocks == 36
    assert len(conv._rules(conv.V2_MEDIUM)) == conv.V2_MEDIUM.tensor_count
    assert len(conv._rules(conv.V2_LARGE)) == conv.V2_LARGE.tensor_count


def test_synthetic_f16_artifact_is_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("gguf")
    model = tmp_path / "model.pt"
    torch.save({"emb.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4), "transformer.lm_head.weight": torch.ones(2, 4)}, model)
    tok = tmp_path / "qwen"
    _tokenizer(tok)
    first = tmp_path / "first.gguf"
    second = tmp_path / "second.gguf"
    conv.convert(model, tok, first, dtype="F16", dry_run=False, strict_shapes=False)
    conv.convert(model, tok, second, dtype="F16", dry_run=False, strict_shapes=False)
    assert first.read_bytes() == second.read_bytes()
    assert first.with_suffix(".gguf.sha256").read_text().split()[0] == conv._sha256(first)
    manifest = json.loads(first.with_suffix(".gguf.manifest.json").read_text())
    assert manifest["runtime"] == {
        "repository": "lglg666/SongGeneration-Runtime",
        "revision": conv.RUNTIME_REVISION,
    }
    assert manifest["tokenizer"]["revision"] == conv.RUNTIME_REVISION
    assert manifest["tokenizer"]["primary_sha256"] == manifest["tokenizer"]["assets"]["tokenizer.json"]

    from gguf import GGUFReader

    reader = GGUFReader(str(first))
    assert reader.fields["levo2.source.runtime_revision"].parts[-1].tobytes().decode() == conv.RUNTIME_REVISION
