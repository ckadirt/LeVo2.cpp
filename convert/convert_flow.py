#!/usr/bin/env python3
"""Strict converter for LeVo v2's inference-only Flow renderer checkpoint.

The released ``model_2.safetensors`` contains training-only audio encoders and
loss heads alongside the Flow denoiser.  This converter intentionally accepts
only the reviewed, token-to-latent subset and makes every other source tensor
an explicit omission.  ``--dry-run`` reads SafeTensors headers only, so it is
safe to use for CI inventory checks without allocating or writing the roughly
2.5 GiB Flow GGUF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPO = "lglg666/SongGeneration-Runtime"
RUNTIME_REVISION = "cc258cc694a63114c61684cc26d0583b8ad777d0"
LEVO_REPO = "https://github.com/levo-demo/LeVo"
LEVO_REVISION = "653cbcf4716101834900c75b7d5da43b07e15d5b"
GGML_REVISION = "8846b79e66747bb9f68597420e95114c177315ce"
CONVERTER_VERSION = "1"
SCHEMA_VERSION = 1

HIDDEN = 2200
LAYERS = 16
HEADS = 20
HEAD_DIM = 110
INTERMEDIATE = 4400
MAX_FRAMES = 1000
CODEBOOK_SIZE = 16384
CODEBOOK_DIM = 32
CONDITION_DIM = 1024
LATENT_DIM = 64
MASK_DIM = 24
TIME_DIM = 512
PINNED_HEADER_SHA256 = "0bd1b4f0a843babde4f2d2d114e318f169389ca90a1d82f85550b362981ddcfa"


@dataclass(frozen=True)
class Rule:
    source: str
    target: str
    shape: tuple[int, ...]
    transform: str = "identity"


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str


def _estimator_rules() -> list[Rule]:
    out = [
        Rule("mask_emb.weight", "flow.mask_embedding.weight", (3, MASK_DIM)),
        Rule("zero_cond_embedding1", "flow.null_condition.weight", (CONDITION_DIM,)),
        Rule("normfeat.counts", "flow.norm.counts", (1,)),
        Rule("normfeat.sum_x", "flow.norm.sum_x", (LATENT_DIM,)),
        Rule("normfeat.sum_x2", "flow.norm.sum_x2", (LATENT_DIM,)),
        Rule("cfm_wrapper.estimator.wpe.weight", "flow.position_embedding.weight", (MAX_FRAMES, HIDDEN)),
        Rule("cfm_wrapper.estimator.adaln_single.emb.timestep_embedder.linear_1.weight", "flow.time_embedding.linear_1.weight", (HIDDEN, TIME_DIM)),
        Rule("cfm_wrapper.estimator.adaln_single.emb.timestep_embedder.linear_1.bias", "flow.time_embedding.linear_1.bias", (HIDDEN,)),
        Rule("cfm_wrapper.estimator.adaln_single.emb.timestep_embedder.linear_2.weight", "flow.time_embedding.linear_2.weight", (HIDDEN, HIDDEN)),
        Rule("cfm_wrapper.estimator.adaln_single.emb.timestep_embedder.linear_2.bias", "flow.time_embedding.linear_2.bias", (HIDDEN,)),
        Rule("cfm_wrapper.estimator.adaln_single.linear.weight", "flow.time_modulation.weight", (6 * HIDDEN, HIDDEN)),
        Rule("cfm_wrapper.estimator.adaln_single.linear.bias", "flow.time_modulation.bias", (6 * HIDDEN,)),
        Rule("cfm_wrapper.estimator.scale_shift_table", "flow.final_modulation.weight", (2, HIDDEN)),
        Rule("cfm_wrapper.estimator.ln_f.weight", "flow.final_norm.weight", (HIDDEN,)),
        Rule("cfm_wrapper.estimator.ln_f.bias", "flow.final_norm.bias", (HIDDEN,)),
        Rule("cfm_wrapper.estimator.proj_out.weight", "flow.output.weight", (HIDDEN, HIDDEN)),
        Rule("cfm_wrapper.estimator.proj_out.bias", "flow.output.bias", (HIDDEN,)),
    ]
    for i in range(LAYERS):
        s, t = f"cfm_wrapper.estimator.h.{i}", f"flow.block.{i}"
        out.extend((
            # transformers.Conv1D stores [in, out]; GGUF uses [out, in].
            Rule(f"{s}.attn.c_attn.weight", f"{t}.attn.qkv.weight", (HIDDEN, 3 * HIDDEN), "transpose_2d"),
            Rule(f"{s}.attn.c_attn.bias", f"{t}.attn.qkv.bias", (3 * HIDDEN,)),
            Rule(f"{s}.attn.c_proj.weight", f"{t}.attn.out.weight", (HIDDEN, HIDDEN), "transpose_2d"),
            Rule(f"{s}.attn.c_proj.bias", f"{t}.attn.out.bias", (HIDDEN,)),
            Rule(f"{s}.ln_1.weight", f"{t}.norm_1.weight", (HIDDEN,)),
            Rule(f"{s}.ln_1.bias", f"{t}.norm_1.bias", (HIDDEN,)),
            Rule(f"{s}.ln_2.weight", f"{t}.norm_2.weight", (HIDDEN,)),
            Rule(f"{s}.ln_2.bias", f"{t}.norm_2.bias", (HIDDEN,)),
            Rule(f"{s}.mlp.c_fc.weight", f"{t}.ffn.in.weight", (HIDDEN, INTERMEDIATE), "transpose_2d"),
            Rule(f"{s}.mlp.c_fc.bias", f"{t}.ffn.in.bias", (INTERMEDIATE,)),
            Rule(f"{s}.mlp.c_proj.weight", f"{t}.ffn.out.weight", (INTERMEDIATE, HIDDEN), "transpose_2d"),
            Rule(f"{s}.mlp.c_proj.bias", f"{t}.ffn.out.bias", (HIDDEN,)),
            Rule(f"{s}.scale_shift_table", f"{t}.modulation.weight", (6, HIDDEN)),
        ))
    return out


RULES = tuple(_estimator_rules())
RULE_BY_SOURCE = {rule.source: rule for rule in RULES}
assert len(RULE_BY_SOURCE) == len(RULES)


RVQ_SPECS = {
    "rvq_bestrq_emb": "vocal",
    "rvq_bestrq_bgm_emb": "bgm",
}


def _rvq_rules() -> list[Rule]:
    out: list[Rule] = []
    for source_prefix, target_stream in RVQ_SPECS.items():
        source = f"{source_prefix}.quantizers.0"
        target = f"flow.rvq.{target_stream}"
        out.extend((
            Rule(f"{source}.codebook.weight", f"{target}.codebook.weight", (CODEBOOK_SIZE, CODEBOOK_DIM)),
            # WeightNorm's g and v are intentionally materialized together.
            Rule(f"{source}.out_proj.weight_g", f"{target}.out_proj.weight", (CONDITION_DIM, 1, 1), "weight_norm_out_proj"),
            Rule(f"{source}.out_proj.weight_v", f"{target}.out_proj.weight", (CONDITION_DIM, CODEBOOK_DIM, 1), "weight_norm_out_proj_v"),
            Rule(f"{source}.out_proj.bias", f"{target}.out_proj.bias", (CONDITION_DIM,)),
        ))
    return out


RVQ_RULES = tuple(_rvq_rules())

OMITTED_EXACT: dict[str, str] = {
    "cfm_wrapper.estimator.wte.weight": "unused because renderer supplies inputs_embeds",
    "normfeat.sum_target_x2": "training-only statistic; inference uses counts, sum_x, and sum_x2",
    "rsp48toclap.kernel": "training-only audio feature resampler",
    "rsq48tobestrq.kernel": "audio-to-token encoder resampler",
    "rsq48tohubert.kernel": "training-only audio feature resampler",
    "rsq48towav2vec.kernel": "training-only audio feature resampler",
}


def _omission_reason(source: str) -> str | None:
    if source in OMITTED_EXACT:
        return OMITTED_EXACT[source]
    if source.startswith("bestrq."):
        return "audio-to-token MusicFM encoder is unreachable from token rendering"
    if source.startswith("hubert."):
        return "training-only Hubert projection-loss encoder"
    if source.startswith("cfm_wrapper.mlp."):
        return "training-only projection-loss MLP"
    for prefix in RVQ_SPECS:
        q = f"{prefix}.quantizers.0."
        if source.startswith(q):
            suffix = source[len(q):]
            if suffix.startswith("in_proj."):
                return "RVQ input projection is only used by audio-to-token encoding"
            if suffix == "stale_counter":
                return "training-only codebook maintenance state"
    return None


def _classify(source: str, info: TensorInfo) -> tuple[str, Rule | None, str | None]:
    """Return ``(emitted|omitted, rule, reason)`` or fail closed."""
    if source in RULE_BY_SOURCE:
        rule = RULE_BY_SOURCE[source]
        if info.dtype != "F32":
            raise ValueError(f"dtype mismatch for {source}: got {info.dtype}, expected F32")
        if info.shape != rule.shape:
            raise ValueError(f"shape mismatch for {source}: got {info.shape}, expected {rule.shape}")
        return "emitted", rule, None
    for rule in RVQ_RULES:
        if source == rule.source:
            if info.dtype != "F32":
                raise ValueError(f"dtype mismatch for {source}: got {info.dtype}, expected F32")
            if info.shape != rule.shape:
                raise ValueError(f"shape mismatch for {source}: got {info.shape}, expected {rule.shape}")
            return "emitted", rule, None
    reason = _omission_reason(source)
    if reason is not None:
        return "omitted", None, reason
    raise ValueError(f"unclassified Flow checkpoint tensor {source!r}; update explicit conversion rules")


def inspect_inventory(infos: Mapping[str, TensorInfo]) -> dict[str, Any]:
    """Validate the complete checkpoint key set without loading tensor values."""
    emitted: list[dict[str, Any]] = []
    omitted: list[dict[str, str]] = []
    targets: set[str] = set()
    for source in sorted(infos):
        disposition, rule, reason = _classify(source, infos[source])
        if disposition == "omitted":
            omitted.append({"source": source, "reason": str(reason)})
            continue
        assert rule is not None
        # The g/v pair intentionally maps to one materialized target.
        if rule.target in targets and rule.transform not in {"weight_norm_out_proj", "weight_norm_out_proj_v"}:
            raise ValueError(f"duplicate output tensor {rule.target!r}")
        targets.add(rule.target)
        target_shape = list(reversed(rule.shape)) if rule.transform == "transpose_2d" else list(rule.shape)
        if rule.transform.startswith("weight_norm_out_proj"):
            target_shape = [CONDITION_DIM, CODEBOOK_DIM]
        emitted.append({"source": source, "target": rule.target, "source_shape": list(rule.shape), "shape": target_shape, "transform": rule.transform})
    missing = sorted(set(RULE_BY_SOURCE) - set(infos))
    for rule in RVQ_RULES:
        if rule.source not in infos:
            missing.append(rule.source)
    if missing:
        raise ValueError("missing required Flow checkpoint tensors: " + ", ".join(missing))
    unique_shapes = {item["target"]: item["shape"] for item in emitted}
    return {
        "source_tensor_count": len(infos),
        "emitted_source_tensor_count": len(emitted),
        "emitted_tensor_count": len(targets),
        "emitted_parameter_count": sum(math.prod(shape) for shape in unique_shapes.values()),
        "omitted_source_tensor_count": len(omitted),
        "emitted": emitted,
        "omissions": omitted,
    }


def _safe_infos(path: Path) -> dict[str, TensorInfo]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required; install it in the converter environment") from exc
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: TensorInfo(tuple(handle.get_slice(name).get_shape()), str(handle.get_slice(name).get_dtype())) for name in handle.keys()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _header_sha256(infos: Mapping[str, TensorInfo]) -> str:
    payload = [(name, list(info.shape), info.dtype) for name, info in sorted(infos.items())]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata(writer: Any, *, source_sha256: str, dtype: str) -> None:
    writer.add_name("LeVo v2 Flow renderer")
    writer.add_file_type(1 if dtype == "F16" else 0)
    writer.add_string("general.license", "Tencent AI Lab academic/research/education only")
    writer.add_string("general.source.repo_url", "https://huggingface.co/" + RUNTIME_REPO)
    writer.add_string("levo2.converter", "convert_flow.py")
    writer.add_string("levo2.converter.version", CONVERTER_VERSION)
    writer.add_uint32("levo2.schema_version", SCHEMA_VERSION)
    writer.add_string("levo2.source.runtime_repository", RUNTIME_REPO)
    writer.add_string("levo2.source.runtime_revision", RUNTIME_REVISION)
    writer.add_string("levo2.source.levo_repository", LEVO_REPO)
    writer.add_string("levo2.source.levo_revision", LEVO_REVISION)
    writer.add_string("levo2.source.ggml_repository", "https://github.com/ggml-org/ggml")
    writer.add_string("levo2.source.ggml_revision", GGML_REVISION)
    writer.add_string("levo2.source.model_sha256", source_sha256)
    for key, value in {
        "hidden_size": HIDDEN, "n_layer": LAYERS, "n_head": HEADS, "head_dim": HEAD_DIM,
        "intermediate_size": INTERMEDIATE, "max_frames": MAX_FRAMES, "codebook_size": CODEBOOK_SIZE,
        "codebook_dim": CODEBOOK_DIM, "condition_dim": CONDITION_DIM, "latent_dim": LATENT_DIM,
        "mask_dim": MASK_DIM, "time_embedding_dim": TIME_DIM, "euler_steps_default": 50,
        "sample_rate": 48000, "frame_rate": 25, "window_frames": 1000,
        "hop_frames": 750, "overlap_frames": 250,
    }.items():
        writer.add_uint32("levo2.flow." + key, value)
    writer.add_float32("levo2.flow.cfg_default", 1.5)
    writer.add_float32("levo2.flow.rope_theta", 10000.0)
    writer.add_float32("levo2.flow.time_embedding_scale", 1000.0)
    writer.add_float32("levo2.flow.sigma_min", 1.0e-4)
    writer.add_bool("levo2.flow.rvq_weight_norm_folded", True)
    writer.add_string("levo2.flow.parameter_dtype", dtype)


def _to_numpy(tensor: Any) -> np.ndarray:
    # SafeTensors always returns a CPU torch tensor in this converter.
    result = tensor.detach().cpu()
    if str(result.dtype) == "torch.bfloat16":
        result = result.float()
    arr = result.numpy()
    if arr.dtype not in (np.float16, np.float32):
        raise ValueError(f"unsupported emitted tensor dtype {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite emitted tensor")
    return np.ascontiguousarray(arr)


def _materialize_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    if g.shape != (CONDITION_DIM, 1, 1) or v.shape != (CONDITION_DIM, CODEBOOK_DIM, 1):
        raise ValueError(f"unexpected RVQ weight-norm shapes g={g.shape}, v={v.shape}")
    norm = np.linalg.norm(v, axis=(1, 2), keepdims=True)
    if np.any(norm == 0):
        raise ValueError("RVQ weight_norm contains a zero-norm vector")
    return np.ascontiguousarray((v * (g / norm)).reshape(CONDITION_DIM, CODEBOOK_DIM))


def _load_emitted_tensors(path: Path, inventory: dict[str, Any]) -> dict[str, np.ndarray]:
    from safetensors.torch import load_file
    state = load_file(str(path), device="cpu")
    out: dict[str, np.ndarray] = {}
    rvq_parts: dict[str, dict[str, np.ndarray]] = {stream: {} for stream in RVQ_SPECS.values()}
    for item in inventory["emitted"]:
        source, target, transform = item["source"], item["target"], item["transform"]
        arr = _to_numpy(state[source])
        if transform == "transpose_2d":
            out[target] = np.ascontiguousarray(arr.T)
        elif transform == "weight_norm_out_proj":
            stream = target.split(".")[2]
            rvq_parts[stream]["g"] = arr
        elif transform == "weight_norm_out_proj_v":
            stream = target.split(".")[2]
            rvq_parts[stream]["v"] = arr
        else:
            out[target] = arr
    for stream, parts in rvq_parts.items():
        if set(parts) != {"g", "v"}:
            raise ValueError(f"missing RVQ weight-norm component for {stream}")
        out[f"flow.rvq.{stream}.out_proj.weight"] = _materialize_weight_norm(parts["g"], parts["v"])
    return out


def convert(model: Path, output: Path | None = None, *, dtype: str = "F32", dry_run: bool = False, hash_source: bool = True, gguf_python: Path | None = None) -> dict[str, Any]:
    """Convert a Flow checkpoint or return its strict header-only inventory."""
    if dtype not in {"F16", "F32"}:
        raise ValueError("dtype must be F16 or F32")
    if not model.is_file():
        raise FileNotFoundError(model)
    infos = _safe_infos(model)
    header_sha256 = _header_sha256(infos)
    if header_sha256 != PINNED_HEADER_SHA256:
        raise ValueError(
            "Flow checkpoint header does not match the pinned 993-tensor contract: "
            f"got {header_sha256}, expected {PINNED_HEADER_SHA256}"
        )
    inventory = inspect_inventory(infos)
    manifest: dict[str, Any] = {
        "converter": "convert_flow.py", "converter_version": CONVERTER_VERSION, "schema_version": SCHEMA_VERSION,
        "dtype": dtype, "source": {"repository": RUNTIME_REPO, "revision": RUNTIME_REVISION,
        "filename": model.name, "bytes": model.stat().st_size,
        "header_sha256": header_sha256,
        "sha256": _sha256(model) if hash_source else None},
        "inventory": inventory,
    }
    if dry_run:
        return manifest
    if output is None:
        raise ValueError("output is required unless dry_run=True")
    if manifest["source"]["sha256"] is None:
        raise ValueError("source hashing may only be skipped for --dry-run inventory checks")
    tensors = _load_emitted_tensors(model, inventory)
    expected_count = inventory["emitted_tensor_count"]
    if len(tensors) != expected_count:
        raise ValueError(f"internal mapping count mismatch: {len(tensors)} != {expected_count}")
    if gguf_python is not None:
        sys.path.insert(0, str(gguf_python))
    from gguf import GGUFWriter
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output), "levo2_flow")
    _metadata(writer, source_sha256=manifest["source"]["sha256"] or "", dtype=dtype)
    for name in sorted(tensors):
        writer.add_tensor(name, tensors[name].astype(np.float16 if dtype == "F16" else np.float32, copy=False))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    manifest["tensor_count"] = len(tensors)
    manifest["parameter_count"] = int(sum(t.size for t in tensors.values()))
    if manifest["parameter_count"] != inventory["emitted_parameter_count"]:
        raise ValueError("materialized Flow parameter count does not match the header inventory")
    manifest["artifact"] = {"filename": output.name, "bytes": output.stat().st_size, "sha256": _sha256(output)}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{manifest['artifact']['sha256']}  {output.name}\n", encoding="utf-8")
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="model_septoken/model_2.safetensors")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--dtype", choices=("F16", "F32"), default="F32")
    parser.add_argument("--dry-run", action="store_true", help="validate SafeTensors headers without loading/writing weights")
    parser.add_argument("--no-hash-source", action="store_true", help="skip the optional full checkpoint SHA-256 pass")
    parser.add_argument("--gguf-python", type=Path)
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is supplied")
    print(json.dumps(convert(args.model, args.output, dtype=args.dtype, dry_run=args.dry_run, hash_source=not args.no_hash_source, gguf_python=args.gguf_python), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
