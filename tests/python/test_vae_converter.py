"""Focused tests for the strict decoder-only Oobleck VAE converter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[2] / "convert"))
import convert_vae as conv  # noqa: E402


def _valid_config(path: Path) -> None:
    path.write_text(json.dumps({"sample_rate": 48000, "model": {"downsampling_ratio": 1920, "decoder": {"config": {
        "out_channels": 2, "channels": 128, "c_mults": [1, 2, 4, 8, 16], "strides": [2, 4, 4, 6, 10],
        "latent_dim": 64, "use_snake": True, "final_tanh": False,
    }}}}), encoding="utf-8")


def _full_state() -> dict[str, object]:
    return {name: torch.ones(shape, dtype=torch.float32) for name, shape in conv.EXPECTED_SOURCE_SHAPES.items()}


def test_schema_classifies_all_pinned_source_tensors() -> None:
    assert len(conv.DECODER_SOURCE_SHAPES) == 182
    assert len(conv.ENCODER_SOURCE_SHAPES) == 183
    assert len(conv.EXPECTED_SOURCE_SHAPES) == 365
    state = _full_state()
    emitted, omitted, keys = conv.inspect_state_dict(state)
    assert len(emitted) == 145
    assert len(omitted) == 183
    assert len(keys) == 365
    assert all(item["source"].startswith("encoder.") for item in omitted)
    assert "vae.decoder.input.weight" in emitted
    assert "vae.decoder.stage.0.upsample.weight" in emitted
    assert "vae.decoder.stage.4.residual.2.conv.1.weight" in emitted
    assert "vae.decoder.output.alpha_log" in emitted
    assert "vae.decoder.output.weight" in emitted


def test_folds_transpose_weight_norm_per_source_dim_zero() -> None:
    # ConvTranspose1d has [Cin, Cout, K] source layout.  Weight norm's default
    # dim=0 must therefore scale independently across its first/source axis.
    spec = conv.ConvSpec("example", "weight", None, (2, 3, 2), transpose=True)
    weight_v = np.asarray([[[3, 4], [0, 0], [0, 0]], [[0, 0], [5, 12], [0, 0]]], dtype=np.float32)
    weight_g = np.asarray([[[10]], [[26]]], dtype=np.float32)
    folded = conv._fold_weight_norm(weight_v, weight_g, spec)
    assert np.allclose(folded[0], weight_v[0] * 2.0)  # norm 5 -> scale 2
    assert np.allclose(folded[1], weight_v[1] * 2.0)  # norm 13 -> scale 2


def test_unknown_missing_and_shape_errors_are_fatal() -> None:
    with pytest.raises(ValueError, match="unclassified checkpoint tensor"):
        conv.inspect_state_dict({**_full_state(), "decoder.unexpected": torch.zeros(1)})
    state = _full_state()
    del state["decoder.layers.0.weight_v"]
    with pytest.raises(ValueError, match="missing required checkpoint tensor"):
        conv.inspect_state_dict(state)
    state = _full_state()
    state["decoder.layers.0.weight_v"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        conv.inspect_state_dict(state)
    state = _full_state()
    state["decoder.layers.0.weight_v"] = state["decoder.layers.0.weight_v"].half()
    with pytest.raises(ValueError, match="dtype mismatch"):
        conv.inspect_state_dict(state)


def test_dry_run_inventory_and_gguf_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vae.ckpt"
    config = tmp_path / "vae.json"
    _valid_config(config)
    torch.save({"state_dict": _full_state()}, checkpoint)
    manifest = conv.convert(checkpoint, config, tmp_path / "unused.gguf", dry_run=True)
    assert manifest["source_tensor_count"] == 365
    assert manifest["emitted_tensor_count"] == 145
    assert manifest["omitted_tensor_count"] == 183

    pytest.importorskip("gguf")
    output = tmp_path / "vae.gguf"
    written = conv.convert(checkpoint, config, output, dtype="F32")
    assert written["artifact"]["sha256"] == conv._sha256(output)
    from gguf import GGUFReader
    reader = GGUFReader(str(output))
    assert reader.fields["general.architecture"].parts[-1].tobytes().decode() == "levo2_vae"
    assert bool(reader.fields["levo2.vae.weight_norm_folded"].parts[-1][0])
    assert len(reader.tensors) == 145
