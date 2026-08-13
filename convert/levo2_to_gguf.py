#!/usr/bin/env python3
"""Convert a trusted SongGeneration v2 LeLM checkpoint to GGUF.

This converter deliberately does not instantiate the upstream model.  A model
checkpoint is an untrusted input, so it is loaded with ``weights_only=True``
and only tensor values from a fixed, reviewed allow-list are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RUNTIME_REPO = "lglg666/SongGeneration-Runtime"
RUNTIME_REVISION = "cc258cc694a63114c61684cc26d0583b8ad777d0"
LEVO_REVISION = "653cbcf4716101834900c75b7d5da43b07e15d5b"
GGML_REVISION = "8846b79e66747bb9f68597420e95114c177315ce"
CONVERTER_VERSION = "2"
SCHEMA_VERSION = 1

CODE_SIZE = 16384
CODE_EOS = 16384
CODE_SPECIAL = 16385
CODE_INPUT_VOCAB = CODE_SIZE + 2
CODE_OUTPUT_VOCAB = CODE_SIZE + 1
STRUCTURE_TOKENS = ("[verse]", "[chorus]", "[bridge]", "[intro-short]", "[intro-medium]", "[intro-long]", "[outro-short]", "[outro-medium]", "[outro-long]", "[inst-short]", "[inst-medium]", "[inst-long]", "[silence]")


@dataclass(frozen=True)
class ModelSpec:
    """Immutable architecture and provenance contract for one released LeLM."""

    variant: str
    name: str
    repository: str
    revision: str
    model_bytes: int
    model_sha256: str
    config_bytes: int
    config_sha256: str
    width: int
    ffn: int
    main_blocks: int
    detail_blocks: int
    attention_heads: int

    @property
    def tensor_count(self) -> int:
        # Twenty non-block tensors plus nine tensors per transformer block.
        return 20 + 9 * (self.main_blocks + self.detail_blocks)


V2_MEDIUM = ModelSpec(
    variant="v2-medium",
    name="LeVo2 v2-medium",
    repository="lglg666/SongGeneration-v2-medium",
    revision="7d91660ebfa041e29bace194f5631e775796f600",
    model_bytes=7_343_951_582,
    model_sha256="4ef2be41f6d838824f5432491408f68d9ffbeda3b1349e1208f9cdfcc64445b1",
    config_bytes=3_352,
    config_sha256="9b1e1cb79b9824816e9f119e1d3e1c3dbb91265121fdd4219667f8e0158a563f",
    width=1536,
    ffn=8960,
    main_blocks=28,
    detail_blocks=12,
    attention_heads=12,
)

V2_LARGE = ModelSpec(
    variant="v2-large",
    name="LeVo2 v2-large",
    repository="lglg666/SongGeneration-v2-large",
    revision="115805364ad74479fb3764fe65970c92faeb1a5a",
    model_bytes=12_899_965_446,
    model_sha256="dc763aa9a76a22a87597c2faf9a51c24d13349ac754699b37e9068b483639def",
    config_bytes=3_352,
    config_sha256="14a991bd7342b9dde348e6324afd44b5c6ecb1db8d0ed4d2dbe666b220b04c59",
    width=2048,
    ffn=11008,
    main_blocks=36,
    detail_blocks=12,
    attention_heads=16,
)

MODEL_SPECS = {spec.variant: spec for spec in (V2_MEDIUM, V2_LARGE)}

# Compatibility aliases for external synthetic tests which import the original
# v2-medium constants. Production conversion always uses an explicit spec.
MODEL_REPO = V2_MEDIUM.repository
MODEL_REVISION = V2_MEDIUM.revision
WIDTH = V2_MEDIUM.width
FFN = V2_MEDIUM.ffn
MAIN_BLOCKS = V2_MEDIUM.main_blocks
DETAIL_BLOCKS = V2_MEDIUM.detail_blocks


@dataclass(frozen=True)
class TensorRule:
    source: str
    target: str
    shape: tuple[int, ...] | None = None
    omission: str | None = None


def _rules(spec: ModelSpec = V2_MEDIUM) -> tuple[TensorRule, ...]:
    out: list[TensorRule] = [
        TensorRule("audiolm.emb.0.weight", "token_embd.mixed", (CODE_INPUT_VOCAB, spec.width)),
        TensorRule("audiolm.transformer.lm_head.weight", "output.mixed", (CODE_OUTPUT_VOCAB, spec.width)),
        TensorRule("audiolm.mlp.0.weight", "bridge.0.weight", (spec.width, spec.width * 2)),
        TensorRule("audiolm.mlp.0.bias", "bridge.0.bias", (spec.width,)),
        TensorRule("audiolm.mlp.2.weight", "bridge.2.weight", (spec.width, spec.width)),
        TensorRule("audiolm.mlp.2.bias", "bridge.2.bias", (spec.width,)),
        TensorRule("audiolm.layer2_emb.1.weight", "token_embd.vocal", (CODE_INPUT_VOCAB, spec.width)),
        TensorRule("audiolm.layer2_emb.2.weight", "token_embd.bgm", (CODE_INPUT_VOCAB, spec.width)),
        TensorRule("audiolm.condition_provider.conditioners.description.output_proj.weight", "cond.lyrics.weight"),
        TensorRule("audiolm.condition_provider.conditioners.description.structure_emb.weight", "cond.structure.weight", (200, spec.width)),
        TensorRule("audiolm.condition_provider.conditioners.type_info.output_proj.weight", "cond.style.weight"),
        TensorRule("audiolm.condition_provider.conditioners.prompt_audio.emb.0.weight", "cond.prompt_embd.mixed.weight", (CODE_INPUT_VOCAB, spec.width)),
        TensorRule("audiolm.condition_provider.conditioners.prompt_audio.emb.1.weight", "cond.prompt_embd.vocal.weight", (CODE_INPUT_VOCAB, spec.width)),
        TensorRule("audiolm.condition_provider.conditioners.prompt_audio.emb.2.weight", "cond.prompt_embd.bgm.weight", (CODE_INPUT_VOCAB, spec.width)),
        TensorRule("audiolm.condition_provider.conditioners.prompt_audio.EOT_emb", "cond.prompt_eot.mixed", (1, spec.width)),
        TensorRule("audiolm.condition_provider.conditioners.prompt_audio.layer2_EOT_emb", "cond.prompt_eot.detail", (1, spec.width)),
    ]
    # Hugging Face Llama names are kept in this table rather than transformed
    # heuristically, so a newly-added checkpoint tensor cannot silently enter
    # the artifact.
    for tower, nblocks, prefix in (("transformer", spec.main_blocks, "main"), ("transformer2", spec.detail_blocks, "detail")):
        for i in range(nblocks):
            s = f"audiolm.{tower}.model.layers.{i}"
            d = f"{prefix}.blk.{i}"
            out.extend([
                TensorRule(f"{s}.input_layernorm.weight", f"{d}.attn_norm.weight", (spec.width,)),
                TensorRule(f"{s}.self_attn.q_proj.weight", f"{d}.attn_q.weight", (spec.width, spec.width)),
                TensorRule(f"{s}.self_attn.k_proj.weight", f"{d}.attn_k.weight", (spec.width, spec.width)),
                TensorRule(f"{s}.self_attn.v_proj.weight", f"{d}.attn_v.weight", (spec.width, spec.width)),
                TensorRule(f"{s}.self_attn.o_proj.weight", f"{d}.attn_output.weight", (spec.width, spec.width)),
                TensorRule(f"{s}.post_attention_layernorm.weight", f"{d}.ffn_norm", (spec.width,)),
                TensorRule(f"{s}.mlp.gate_proj.weight", f"{d}.ffn_gate.weight", (spec.ffn, spec.width)),
                TensorRule(f"{s}.mlp.up_proj.weight", f"{d}.ffn_up.weight", (spec.ffn, spec.width)),
                TensorRule(f"{s}.mlp.down_proj.weight", f"{d}.ffn_down.weight", (spec.width, spec.ffn)),
            ])
        out.append(TensorRule(f"audiolm.{tower}.model.norm.weight", f"{prefix}.output_norm", (spec.width,)))
    out.extend([
        TensorRule("audiolm.linears.0.weight", "output.vocal", (CODE_OUTPUT_VOCAB, spec.width)),
        TensorRule("audiolm.linears.1.weight", "output.bgm", (CODE_OUTPUT_VOCAB, spec.width)),
    ])
    return tuple(out)


RULES = _rules(V2_MEDIUM)
RULE_BY_SOURCE = {r.source: r for r in RULES}

# Keys initialized by the Hugging Face modules but not reached by released
# inference.  Keeping these names explicit is part of the converter contract.
OMISSIONS: dict[str, str] = {
    "audiolm.layer2_emb.0.weight": "detail input embedding 0 is never summed (forward sums streams 1 and 2)",
    "audiolm.transformer.model.embed_tokens.weight": "input is passed as inputs_embeds",
    "audiolm.transformer2.model.embed_tokens.weight": "input is passed as inputs_embeds",
    "audiolm.transformer2.lm_head.weight": "released inference uses linears.0 and linears.1",
    "audiolm.out_norm.weight": "out_norm path is disabled in released inference",
    "audiolm.out_norm.bias": "out_norm path is disabled in released inference",
}


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - useful diagnostic for CLI users
        raise RuntimeError("PyTorch is required to load model.pt; use a CPU PyTorch environment") from exc
    return torch


def load_state_dict(path: Path) -> dict[str, Any]:
    """Load only a tensor state dictionary, on CPU, with safe torch loading."""
    torch = _torch()
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(obj, Mapping) and "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping) or not obj:
        raise ValueError("model.pt does not contain a non-empty tensor state_dict")
    result: dict[str, Any] = {}
    for key, value in obj.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint contains non-tensor state entry {key!r}")
        if value.device.type != "cpu":
            raise ValueError(f"checkpoint tensor {key!r} is not on CPU")
        # Lightning's checkpoint has audiolm.*; synthetic tests may use the
        # model-local names.  No arbitrary prefixes are accepted.
        key = key.removeprefix("module.")
        if not key.startswith("audiolm."):
            key = "audiolm." + key
        if key in result:
            raise ValueError(f"duplicate normalized checkpoint key {key!r}")
        result[key] = value
    return result


def inspect_state_dict(state: Mapping[str, Any], *, spec: ModelSpec = V2_MEDIUM,
                       strict_shapes: bool = True) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    """Validate and map a state dictionary without writing a file.

    Returns mapped numpy arrays, explicit omissions, and source keys in stable
    order.  This function is intentionally usable by tiny synthetic tests.
    """
    rule_by_source = {rule.source: rule for rule in _rules(spec)}
    mapped: dict[str, np.ndarray] = {}
    omitted: list[dict[str, Any]] = []
    seen: set[str] = set()
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if not key.startswith("audiolm."):
            key = "audiolm." + key
        if key in normalized:
            raise ValueError(f"duplicate normalized checkpoint key {key!r}")
        normalized[key] = value

    for source in sorted(normalized):
        value = normalized[source]
        if source in OMISSIONS:
            omitted.append({"source": source, "reason": OMISSIONS[source]})
            continue
        rule = rule_by_source.get(source)
        if rule is None:
            if source.startswith("audiolm."):
                raise ValueError(f"unclassified checkpoint tensor {source!r}; update the explicit allowlist")
            raise ValueError(f"unexpected checkpoint key {source!r}")
        if rule.target in mapped:
            raise ValueError(f"duplicate output tensor {rule.target!r}")
        if not hasattr(value, "detach"):
            raise ValueError(f"checkpoint value {source!r} is not a tensor")
        # The released checkpoint is bfloat16 for most transformer tensors.
        # NumPy has no portable bfloat16 dtype, so convert it through torch
        # float32 before handing data to gguf-py.  All other accepted source
        # dtypes are likewise normalized without changing values in F32 mode.
        tensor = value.detach().cpu()
        if str(tensor.dtype) == "torch.bfloat16":
            arr = tensor.float().numpy()
        else:
            arr = tensor.numpy()
        if arr.dtype not in (np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64):
            raise ValueError(f"unsupported dtype for {source}: {arr.dtype}")
        arr = np.asarray(arr, dtype=np.float32 if arr.dtype == np.float64 else arr.dtype)
        if not np.isfinite(arr).all() and np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"non-finite values in {source!r}")
        if strict_shapes and rule.shape is not None and tuple(arr.shape) != rule.shape:
            raise ValueError(f"shape mismatch for {source}: got {tuple(arr.shape)}, expected {rule.shape}")
        mapped[rule.target] = np.ascontiguousarray(arr)
        seen.add(source)
    if strict_shapes:
        expected_sources = set(rule_by_source) | set(OMISSIONS)
        missing = sorted(expected_sources - set(normalized))
        if missing:
            raise ValueError("checkpoint is missing required tensor(s): " + ", ".join(missing))
        if len(mapped) != spec.tensor_count:
            raise ValueError(f"checkpoint maps {len(mapped)} runtime tensors; {spec.variant} requires {spec.tensor_count}")
    return mapped, omitted, sorted(normalized)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokenizer_assets(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Qwen2 tokenizer directory does not exist: {directory}")
    def read_json(name: str, default: Any) -> Any:
        p = directory / name
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))
    vocab = read_json("vocab.json", {})
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Qwen2 directory must contain a non-empty vocab.json")
    # Qwen's vocab.json does not contain its three tokenizer.json added
    # specials.  Reconstruct the exact table that Qwen2Tokenizer exposes,
    # then append the thirteen LeVo structure tokens in conf/vocab.yaml (the
    # upstream conditioner calls add_tokens(..., special_tokens=True)).
    tok_json_path = directory / "tokenizer.json"
    tok_json = json.loads(tok_json_path.read_text(encoding="utf-8")) if tok_json_path.exists() else {}
    added = list(tok_json.get("added_tokens", []))
    max_id = max([int(v) for v in vocab.values()] + [int(x["id"]) for x in added])
    token_by_id = {int(v): token for token, v in vocab.items()}
    for x in added:
        token_by_id[int(x["id"])] = x["content"]
    structure_path = ROOT / "conf" / "vocab.yaml"
    structures: list[str] = []
    if structure_path.exists():
        structures = [line.strip()[2:-1] for line in structure_path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("- '")]
    if not structures:
        structures = list(STRUCTURE_TOKENS)
    next_id = max_id + 1
    added_by_content = {x["content"] for x in added}
    for content in structures:
        if content not in added_by_content:
            added.append({"id": next_id, "content": content, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True})
            token_by_id[next_id] = content
            next_id += 1
    tokens = [token_by_id[i] for i in range(max(token_by_id) + 1)]
    if any(x is None for x in tokens):
        raise ValueError("vocab.json has non-contiguous token IDs")
    merges_path = directory / "merges.txt"
    merges = [line.strip() for line in merges_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#version:")] if merges_path.exists() else []
    # Preserve externally supplied added_tokens.json entries too, while the
    # canonical tokenizer.json table above remains the source of token IDs.
    extra_added = read_json("added_tokens.json", [])
    if isinstance(extra_added, dict):
        extra_added = [{"content": k, **v} for k, v in extra_added.items()]
    added.extend(x for x in extra_added if x not in added)
    special = read_json("special_tokens_map.json", {})
    config = read_json("tokenizer_config.json", {})
    return {"tokens": tokens, "merges": merges, "tokenizer_json": tok_json, "added_tokens": added, "special_tokens": special, "tokenizer_config": config,
            "hashes": {p.name: _sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}}


def _add_metadata(writer: Any, tok: dict[str, Any], spec: ModelSpec, source_hash: str,
                  config_hash: str | None, file_type: str) -> None:
    writer.add_name(spec.name)
    writer.add_file_type(1 if file_type == "F16" else 0)
    writer.add_string("general.license", "Tencent AI Lab academic/research/education only")
    writer.add_string("general.license.link", "https://github.com/levo-demo/LeVo/blob/653cbcf4716101834900c75b7d5da43b07e15d5b/LICENSE")
    writer.add_string("general.source.repo_url", "https://huggingface.co/" + spec.repository)
    writer.add_string("levo2.converter", "levo2_to_gguf.py")
    writer.add_string("levo2.converter.version", CONVERTER_VERSION)
    writer.add_uint32("levo2.schema_version", SCHEMA_VERSION)
    writer.add_string("levo2.source.model_repository", spec.repository)
    writer.add_string("levo2.source.model_revision", spec.revision)
    writer.add_string("levo2.source.runtime_repository", RUNTIME_REPO)
    writer.add_string("levo2.source.runtime_revision", RUNTIME_REVISION)
    writer.add_string("levo2.source.levo_repository", "https://github.com/levo-demo/LeVo")
    writer.add_string("levo2.source.levo_revision", LEVO_REVISION)
    writer.add_string("levo2.source.ggml_repository", "https://github.com/ggml-org/ggml")
    writer.add_string("levo2.source.ggml_revision", GGML_REVISION)
    ints = {"main.block_count": spec.main_blocks, "detail.block_count": spec.detail_blocks, "embedding_length": spec.width, "feed_forward_length": spec.ffn,
            "attention.head_count": spec.attention_heads, "attention.kv_head_count": spec.attention_heads, "context_length": 10000, "codebook.count": 3,
            "codebook.size": CODE_SIZE, "token.eos_id": CODE_EOS, "token.special_id": CODE_SPECIAL, "audio.frame_rate": 25,
            "audio.sample_rate": 48000, "condition.lyrics_prefix_length": 600, "condition.prompt_prefix_length": 252, "condition.style_prefix_length": 100}
    for key, value in ints.items(): writer.add_uint32("levo2." + key, value)
    writer.add_float32("levo2.rms_norm_epsilon", 1e-5)
    writer.add_float32("levo2.main.rope_theta", 500000.0)
    writer.add_float32("levo2.detail.rope_theta", 500000.0)
    writer.add_array("levo2.pattern.delays", [0, 250, 250])
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tok["tokens"])
    writer.add_token_merges(tok["merges"])
    writer.add_string("levo2.tokenizer.json", json.dumps(tok["tokenizer_json"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    writer.add_string("levo2.tokenizer.added_tokens.json", json.dumps(tok["added_tokens"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    writer.add_string("levo2.tokenizer.special_tokens.json", json.dumps(tok["special_tokens"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    writer.add_string("levo2.tokenizer.config.json", json.dumps(tok["tokenizer_config"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    writer.add_uint32("levo2.tokenizer.vocab_size", len(tok["tokens"]))
    writer.add_string("levo2.tokenizer.revision", RUNTIME_REVISION)
    writer.add_string("levo2.tokenizer.sha256", tok["hashes"]["tokenizer.json"])
    writer.add_string("levo2.source.model_sha256", source_hash)
    writer.add_string("levo2.source.config_sha256", config_hash or "")
    writer.add_string("levo2.tokenizer.assets_sha256.json", json.dumps(tok["hashes"], sort_keys=True, separators=(",", ":")))


def _verify_source(model_path: Path, config_path: Path | None, spec: ModelSpec) -> tuple[str, str]:
    """Refuse a mis-pinned source before expensive tensor conversion begins."""
    if config_path is None:
        raise ValueError(f"--config is required to verify the pinned {spec.variant} source")
    if model_path.stat().st_size != spec.model_bytes:
        raise ValueError(f"model.pt byte size {model_path.stat().st_size} does not match pinned {spec.variant} source ({spec.model_bytes})")
    source_hash = _sha256(model_path)
    if source_hash != spec.model_sha256:
        raise ValueError(f"model.pt SHA-256 does not match pinned {spec.variant} source")
    if config_path.stat().st_size != spec.config_bytes:
        raise ValueError(f"config.yaml byte size {config_path.stat().st_size} does not match pinned {spec.variant} source ({spec.config_bytes})")
    config_hash = _sha256(config_path)
    if config_hash != spec.config_sha256:
        raise ValueError(f"config.yaml SHA-256 does not match pinned {spec.variant} source")
    return source_hash, config_hash


def convert(model_path: Path, tokenizer_dir: Path, output: Path, *, dtype: str = "F16", config_path: Path | None = None,
            spec: ModelSpec = V2_MEDIUM, verify_source: bool = False, dry_run: bool = False,
            strict_shapes: bool = True, gguf_python: Path | None = None) -> dict[str, Any]:
    if dtype not in ("F16", "F32"):
        raise ValueError("dtype must be F16 or F32")
    source_hash = _sha256(model_path)
    config_hash = _sha256(config_path) if config_path else None
    if verify_source:
        source_hash, config_hash = _verify_source(model_path, config_path, spec)
    state = load_state_dict(model_path)
    tensors, omitted, source_keys = inspect_state_dict(state, spec=spec, strict_shapes=strict_shapes)
    tok = _tokenizer_assets(tokenizer_dir)
    shape_notes: list[str] = []
    if strict_shapes:
        if "cond.lyrics.weight" in tensors and tensors["cond.lyrics.weight"].shape[0] != len(tok["tokens"]):
            raise ValueError(f"cond.lyrics.weight vocabulary shape {tensors['cond.lyrics.weight'].shape[0]} does not match tokenizer assets ({len(tok['tokens'])})")
        if "cond.style.weight" in tensors:
            # The pinned checkpoint contains 151652 rows even though the base
            # Qwen2 tokenizer addresses IDs 0..151645. Preserve all source
            # rows; the six unreachable tail rows are a documented upstream
            # packaging inconsistency, not a conversion error.
            style_rows = tensors["cond.style.weight"].shape[0]
            if style_rows != 151652:
                raise ValueError(f"cond.style.weight has unexpected shape {tensors['cond.style.weight'].shape}")
            shape_notes.append("cond.style.weight retains six unreachable source rows beyond the 151646 base Qwen2 IDs; no truncation")
    mapping = []
    for source in sorted(source_keys):
        rule = {item.source: item for item in _rules(spec)}.get(source)
        if rule is not None and rule.target in tensors:
            mapping.append({"source": source, "target": rule.target, "shape": list(tensors[rule.target].shape)})
    manifest: dict[str, Any] = {"converter": "levo2_to_gguf.py", "converter_version": CONVERTER_VERSION, "schema_version": SCHEMA_VERSION, "dtype": dtype,
        "model_profile": spec.variant,
        "source": {"repository": spec.repository, "revision": spec.revision, "filename": model_path.name, "bytes": model_path.stat().st_size, "sha256": source_hash, "config_sha256": config_hash},
        "levo_source_revision": LEVO_REVISION, "ggml_revision": GGML_REVISION,
        "runtime": {"repository": RUNTIME_REPO, "revision": RUNTIME_REVISION},
        "tokenizer": {"revision": RUNTIME_REVISION, "primary_sha256": tok["hashes"]["tokenizer.json"],
                      "assets": tok["hashes"]}, "tensor_count": len(tensors),
        "parameter_count": int(sum(int(a.size) for a in tensors.values())), "source_tensor_count": len(source_keys), "omissions": omitted,
        "tensors": sorted(({"name": n, "shape": list(a.shape), "dtype": dtype} for n, a in tensors.items()), key=lambda x: x["name"]), "mapping": mapping, "shape_notes": shape_notes}
    if dry_run:
        return manifest
    if gguf_python is not None:
        sys.path.insert(0, str(gguf_python))
    from gguf import GGUFWriter
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output), "levo2")
    _add_metadata(writer, tok, spec, manifest["source"]["sha256"], manifest["source"]["config_sha256"], dtype)
    for name in sorted(tensors):
        arr = tensors[name].astype(np.float16 if dtype == "F16" else np.float32, copy=False)
        writer.add_tensor(name, np.ascontiguousarray(arr))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    manifest["artifact"] = {"filename": output.name, "bytes": output.stat().st_size, "sha256": _sha256(output)}
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{manifest['artifact']['sha256']}  {output.name}\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--tokenizer-dir", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--dtype", choices=("F16", "F32"), default="F16")
    p.add_argument("--variant", choices=tuple(MODEL_SPECS), default="v2-medium",
                   help="immutable released LeLM source profile")
    p.add_argument("--config", type=Path)
    p.add_argument("--gguf-python", type=Path, help="path to ggml/gguf-py (otherwise import the installed gguf package)")
    p.add_argument("--dry-run", "--list", action="store_true", help="validate and print deterministic manifest without writing GGUF")
    p.add_argument("--allow-shape-mismatch", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--allow-unverified-source", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.tokenizer_dir is None:
        p.error("--tokenizer-dir is required for both conversion and --dry-run")
    if not args.dry_run and args.output is None:
        p.error("--output is required unless --dry-run is used")
    manifest = convert(args.model, args.tokenizer_dir, args.output or Path("unused.gguf"), dtype=args.dtype,
                       config_path=args.config, spec=MODEL_SPECS[args.variant],
                       verify_source=not args.allow_unverified_source, dry_run=args.dry_run,
                       strict_shapes=not args.allow_shape_mismatch, gguf_python=args.gguf_python)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
