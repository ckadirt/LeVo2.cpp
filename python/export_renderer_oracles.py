#!/usr/bin/env python3
"""Export small, deterministic oracle fixtures for the official Flow/VAE renderer.

This is a diagnostic tool, not a second renderer.  It imports the pinned
official implementation and records intermediate tensors in NumPy ``.npz``
files, together with a JSON manifest.  The Flow checkpoint is large, so the
default command is metadata-only inventory; ``inputs`` captures the small RVQ
and conditioning boundary, while ``velocity`` and ``trace`` opt into a full
Flow model load.  VAE captures are intentionally small (``--frames 1`` or
``--frames 2`` are useful smoke fixtures).

All random inputs are exported as F32 arrays.  A seed is merely a convenience;
the saved noise/latent is the reproducibility boundary for another runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_SOURCE_DIR = Path(os.environ.get("LEVO_OFFICIAL_SOURCE", "/workspace/reference/LeVo"))
DEFAULT_RUNTIME_DIR = Path(os.environ.get("LEVO_RUNTIME_DIR", "/workspace/models/SongGeneration-Runtime"))
SOURCE_REVISION = "653cbcf4716101834900c75b7d5da43b07e15d5b"
RUNTIME_REVISION = "cc258cc694a63114c61684cc26d0583b8ad777d0"
FLOW_CHECKPOINT = "ckpt/model_septoken/model_2.safetensors"
VAE_CHECKPOINT = "ckpt/vae/autoencoder_music_1320k.ckpt"
VAE_CONFIG = "ckpt/vae/stable_audio_1920_vae.json"
CODEBOOK_SIZE = 16_384
CODEBOOK_DIM = 32
FLOW_CONDITION_DIM = 1024
FLOW_HIDDEN_DIM = 2200
FLOW_LATENT_DIM = 64
FLOW_MASK_DIM = 24
FLOW_DEFAULT_FRAMES = 1000
FLOW_DEFAULT_STEPS = 50
VAE_DOWNSAMPLING = 1920
FLOW_WINDOW_FRAMES = 1000
FLOW_HOP_FRAMES = 750
FLOW_OVERLAP_FRAMES = 250
MAX_RENDER_FRAMES = 10_000
MAX_RENDER_WINDOWS = 64
FLOW_CHECKPOINT_SHA256 = "430b7c1c245722fbe3893cd621b3d4a90076404596e9fb1ce987a4a0f2a4fc6f"
VAE_CHECKPOINT_SHA256 = "10ccb6c83613781ad32e998a90597ba7eb9292911a224598da1fd53728eb4cd3"
VAE_CONFIG_SHA256 = "5cd2859efe00bc2b0f6f9bdac738ad11822a36473d6d810427b60efd057c538b"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _prepend_python_paths(paths: Iterable[Path]) -> None:
    for path in reversed([str(path.resolve()) for path in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)


def _runtime_paths(source_dir: Path, runtime_dir: Path) -> dict[str, Path]:
    return {
        "flow_source": source_dir / "codeclm/tokenizer/Flow1dVAE",
        "flow_checkpoint": runtime_dir / FLOW_CHECKPOINT,
        "vae_checkpoint": runtime_dir / VAE_CHECKPOINT,
        "vae_config": runtime_dir / VAE_CONFIG,
        "stable_audio": runtime_dir / "third_party/stable_audio_tools",
        "demucs_shim": runtime_dir / "third_party/demucs/models/pretrained.py",
    }


def _prepare_official_imports(source_dir: Path, runtime_dir: Path) -> Path:
    paths = _runtime_paths(source_dir, runtime_dir)
    _require_file(paths["flow_source"] / "model_septoken.py", "official Flow source")
    _require_file(paths["stable_audio"] / "stable_audio_tools/models/autoencoders.py", "stable-audio-tools")
    _require_file(paths["demucs_shim"], "Demucs import shim")
    _prepend_python_paths(
        [runtime_dir, paths["stable_audio"], source_dir, paths["flow_source"]]
    )
    return paths["flow_source"]


def _torch_device(device: str):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA renderer requested but torch.cuda.is_available() is false")
    if device.startswith("cuda"):
        # torch.backends.cudnn.allow_tf32 defaults to True, so the decoder's
        # convolutions would silently run in TF32 with a 10-bit mantissa while
        # this exporter advertises float32. A TF32 reference is neither the F32
        # correctness mode nor the FP16-autocast mode of docs/renderer-parity.md,
        # and at a full 1000-frame window it dominates the comparison. Autocast
        # captures still opt into FP16 explicitly and are unaffected.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device(device)


def _provenance(
    *,
    source_dir: Path,
    runtime_dir: Path,
    device: str,
    dtype: str,
    autocast: bool,
    seed: int | None,
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        # The dry/input contract path is intentionally usable from a minimal
        # Python environment without importing the GPU renderer stack.
        return {
            "source_revision": SOURCE_REVISION,
            "runtime_revision": RUNTIME_REVISION,
            "source_dir": str(source_dir),
            "runtime_dir": str(runtime_dir),
            "torch_version": None,
            "device": device,
            "dtype": dtype,
            "autocast": bool(autocast),
            "seed": seed,
        }

    result: dict[str, Any] = {
        "source_revision": SOURCE_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "source_dir": str(source_dir),
        "runtime_dir": str(runtime_dir),
        "torch_version": torch.__version__,
        "device": device,
        "dtype": dtype,
        "autocast": bool(autocast),
        "seed": seed,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        result["cuda_device_name"] = torch.cuda.get_device_name(torch.device(device))
        result["cuda_device_capability"] = list(torch.cuda.get_device_capability(torch.device(device)))
    return result


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(value)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _save_npz(output: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    output = output.resolve()
    if output.suffix != ".npz":
        raise ValueError(f"NPZ output must end in .npz: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    checked: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        checked[name] = _finite(name, np.asarray(array))
    # NumPy's convenience writer embeds current ZIP timestamps. Fixed member
    # metadata makes identical oracle arrays byte-identical across exports.
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(checked.items()):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, value, allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, payload.getvalue())
    metadata = dict(metadata)
    metadata["arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.astype(np.float64).mean()),
            "rms": float(np.sqrt(np.square(value.astype(np.float64)).mean())),
        }
        for name, value in checked.items()
    }
    metadata["npz_sha256"] = _sha256_file(output)
    output.with_suffix(".json").write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _save_json(output: Path, metadata: dict[str, Any]) -> None:
    output = output.resolve()
    if output.suffix != ".json":
        raise ValueError(f"JSON output must end in .json: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_npy(path: Path, *, name: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} is not a NumPy array: {path}")
    if not np.isfinite(value).all() and np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} contains non-finite values: {path}")
    return np.ascontiguousarray(value)


def _load_codes(path: Path, *, name: str, frames: int) -> np.ndarray:
    value = _load_npy(path, name=name)
    if value.ndim == 3 and value.shape[0] == 1 and value.shape[1] == 1:
        value = value[0, 0]
    elif value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim != 1:
        raise ValueError(f"{name} must have shape [T], [1,T], or [1,1,T], got {value.shape}")
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must contain integer token IDs, got {value.dtype}")
    if value.size == 0:
        raise ValueError(f"{name} cannot be empty")
    value = value.astype("<i8", copy=False)
    if np.any(value < 0) or np.any(value >= CODEBOOK_SIZE):
        raise ValueError(f"{name} contains an ID outside [0,{CODEBOOK_SIZE - 1}]")
    repeats = (frames + value.size - 1) // value.size
    return np.tile(value, repeats)[:frames]


def _random_f32(shape: tuple[int, ...], *, device: str, seed: int) -> np.ndarray:
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    value = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    return value.cpu().numpy().astype("<f4", copy=False)


def _load_flow_noise(path: Path | None, *, frames: int, device: str, seed: int) -> tuple[np.ndarray, int | None]:
    if path is None:
        return _random_f32((1, frames, FLOW_LATENT_DIM), device=device, seed=seed), seed
    value = _load_npy(path, name="Flow initial noise")
    if value.ndim == 2:
        if value.shape == (FLOW_LATENT_DIM, frames):
            value = value.T
        value = value[None]
    expected = (1, frames, FLOW_LATENT_DIM)
    if value.shape != expected:
        raise ValueError(f"Flow initial noise must have shape [64,T] or [1,T,64] with T={frames}, got {value.shape}")
    return value.astype("<f4", copy=False), None


def _load_render_tokens(path: Path) -> np.ndarray:
    """Load the canonical renderer token boundary ``[3,T]``."""
    value = _load_npy(path, name="renderer tokens")
    if value.ndim != 2 or value.shape[0] != 3 or value.shape[1] <= 0:
        raise ValueError(f"renderer tokens must have canonical shape [3,T], got {value.shape}")
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"renderer tokens must contain integer IDs, got {value.dtype}")
    if value.shape[1] > MAX_RENDER_FRAMES:
        raise ValueError(f"renderer tokens exceed the strict {MAX_RENDER_FRAMES}-frame limit")
    value = value.astype("<i8", copy=False)
    if np.any(value < 0) or np.any(value >= CODEBOOK_SIZE):
        raise ValueError(f"renderer tokens contain an ID outside [0,{CODEBOOK_SIZE - 1}]")
    return value


def _render_window_plan(frames: int) -> tuple[int, np.ndarray]:
    if frames <= 0 or frames > MAX_RENDER_FRAMES:
        raise ValueError(f"renderer frame count must be in [1,{MAX_RENDER_FRAMES}]")
    if frames <= FLOW_WINDOW_FRAMES:
        padded = FLOW_WINDOW_FRAMES
    else:
        padded = ((frames - FLOW_OVERLAP_FRAMES + FLOW_HOP_FRAMES - 1) // FLOW_HOP_FRAMES) * FLOW_HOP_FRAMES + FLOW_OVERLAP_FRAMES
    starts = np.arange(0, padded - FLOW_OVERLAP_FRAMES, FLOW_HOP_FRAMES, dtype=np.int64)
    if starts.size == 0 or starts[-1] + FLOW_WINDOW_FRAMES > padded:
        raise AssertionError("invalid Flow window plan")
    if starts.size > MAX_RENDER_WINDOWS:
        raise ValueError(f"renderer requires {starts.size} windows; strict limit is {MAX_RENDER_WINDOWS}")
    return padded, starts


def _load_render_noise(path: Path, *, windows: int) -> np.ndarray:
    value = _load_npy(path, name="window-major renderer noise")
    expected = (windows, FLOW_WINDOW_FRAMES, FLOW_LATENT_DIM)
    if value.shape != expected:
        raise ValueError(f"window-major renderer noise must have shape [windows,1000,64]={expected}, got {value.shape}")
    if value.dtype.kind != "f" or value.dtype.itemsize != 4:
        raise ValueError(f"window-major renderer noise must be F32, got {value.dtype}")
    return value.astype("<f4", copy=False)


def _load_flow_model(source_dir: Path, runtime_dir: Path, device: str, *, dtype=None):
    import torch
    from safetensors.torch import load_file

    flow_dir = _prepare_official_imports(source_dir, runtime_dir)
    previous_cwd = Path.cwd()
    os.chdir(flow_dir)
    try:
        from model_septoken import PromptCondAudioDiffusion

        model = PromptCondAudioDiffusion(
            num_channels=32,
            unet_model_name=None,
            unet_model_config_path="configs/models/transformer2D_wocross_inch112_1x4_multi_large.json",
            snr_gamma=None,
        )
    finally:
        os.chdir(previous_cwd)
    device_obj = _torch_device(device)
    model = model.to(device_obj)
    checkpoint = _require_file(runtime_dir / FLOW_CHECKPOINT, "Flow checkpoint")
    state = load_file(str(checkpoint), device=str(device_obj))
    model.load_state_dict(state, strict=False)
    model.eval()
    # Keep Flow weights in F32 for stable reference execution.  ``dtype``
    # controls the VAE/output side; optional CUDA autocast is the explicit
    # mechanism for a reduced-precision Flow run.
    model.init_device_dtype(device_obj, torch.float32)
    return model


def _flow_inventory(source_dir: Path, runtime_dir: Path, output: Path, device: str, seed: int | None) -> None:
    checkpoint = _require_file(runtime_dir / FLOW_CHECKPOINT, "Flow checkpoint")
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise RuntimeError("safetensors is required for Flow inventory") from exc

    tensors: dict[str, Any] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            tensors[key] = {
                "shape": list(handle.get_slice(key).get_shape()),
                "dtype": str(handle.get_slice(key).get_dtype()),
            }
    checkpoint_sha256 = _sha256_file(checkpoint)
    if checkpoint_sha256 != FLOW_CHECKPOINT_SHA256:
        raise ValueError(
            f"Flow checkpoint SHA-256 mismatch: got {checkpoint_sha256}, "
            f"expected {FLOW_CHECKPOINT_SHA256}"
        )
    _save_json(
        output,
        {
            "format": "levo2-renderer-oracle-v1",
            "kind": "flow_inventory",
            "schema_version": 1,
            "checkpoint": {
                "path": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": checkpoint_sha256,
                "tensor_count": len(tensors),
            },
            "tensors": tensors,
            "provenance": _provenance(
                source_dir=source_dir,
                runtime_dir=runtime_dir,
                device=device,
                dtype="metadata",
                autocast=False,
                seed=seed,
            ),
        },
    )


def _flow_inputs(args: argparse.Namespace) -> None:
    import torch

    frames = args.frames
    if frames <= 0 or frames > 1000:
        raise ValueError("--frames must be in [1,1000]")
    model = _load_flow_model(args.source_dir, args.runtime_dir, args.device)
    device = _torch_device(args.device)

    if args.vocal_codes:
        vocal_codes = _load_codes(args.vocal_codes, name="vocal codes", frames=frames)
    else:
        vocal_codes = np.zeros(frames, dtype="<i8")
    if args.bgm_codes:
        bgm_codes = _load_codes(args.bgm_codes, name="BGM codes", frames=frames)
    else:
        bgm_codes = np.zeros(frames, dtype="<i8")

    codes_vocal = torch.from_numpy(vocal_codes).to(device=device, dtype=torch.long).view(1, 1, frames)
    codes_bgm = torch.from_numpy(bgm_codes).to(device=device, dtype=torch.long).view(1, 1, frames)
    with torch.inference_mode():
        vocal_q, vocal_lookup, _ = model.rvq_bestrq_emb.from_codes(codes_vocal)
        bgm_q, bgm_lookup, _ = model.rvq_bestrq_bgm_emb.from_codes(codes_bgm)
        vocal_projected = vocal_q.permute(0, 2, 1).contiguous()
        bgm_projected = bgm_q.permute(0, 2, 1).contiguous()
        mask_ids = torch.full((1, frames), args.mask_value, device=device, dtype=torch.long)
        mask_embedding = model.mask_emb(mask_ids)
        conditioning = torch.cat([vocal_projected, bgm_projected], dim=-1)
        noise, generated_seed = _load_flow_noise(args.noise, frames=frames, device=args.device, seed=args.seed)
        initial_noise = torch.from_numpy(noise).to(device=device)
        incontext = torch.zeros_like(initial_noise)
        positional = model.cfm_wrapper.estimator.wpe(
            torch.arange(frames, device=device, dtype=torch.long).unsqueeze(0)
        )

    arrays = {
        "vocal_codes": vocal_codes.reshape(1, -1).astype("<i4"),
        "bgm_codes": bgm_codes.reshape(1, -1).astype("<i4"),
        "vocal_codebook_lookup": vocal_lookup.permute(0, 2, 1).float().cpu().numpy().astype("<f4"),
        "bgm_codebook_lookup": bgm_lookup.permute(0, 2, 1).float().cpu().numpy().astype("<f4"),
        "vocal_projected": vocal_projected.float().cpu().numpy().astype("<f4"),
        "bgm_projected": bgm_projected.float().cpu().numpy().astype("<f4"),
        "mask_ids": mask_ids.cpu().numpy().astype("<i4"),
        "mask_embedding": mask_embedding.float().cpu().numpy().astype("<f4"),
        "conditioning": conditioning.float().cpu().numpy().astype("<f4"),
        "incontext_latent": incontext.cpu().numpy().astype("<f4"),
        "initial_noise": initial_noise.cpu().numpy().astype("<f4"),
        "positional_embedding": positional.float().cpu().numpy().astype("<f4"),
        "zero_condition_embedding": model.zero_cond_embedding1.float().detach().cpu().numpy().reshape(1, -1).astype("<f4"),
    }
    _save_npz(
        args.output,
        arrays,
        {
            "format": "levo2-renderer-oracle-v1",
            "kind": "flow_inputs",
            "schema_version": 1,
            "frames": frames,
            "mask_value": args.mask_value,
            "provenance": _provenance(
                source_dir=args.source_dir,
                runtime_dir=args.runtime_dir,
                device=args.device,
                dtype="float32",
                autocast=False,
                seed=generated_seed,
            ),
        },
    )


def _flow_model_input(
    mask_embedding,
    incontext,
    conditioning,
    latent,
    *,
    guidance: bool,
):
    import torch

    if guidance:
        return torch.cat(
            [
                torch.cat([mask_embedding, mask_embedding], dim=0),
                torch.cat([incontext, incontext], dim=0),
                torch.cat([torch.zeros_like(conditioning), conditioning], dim=0),
                torch.cat([latent, latent], dim=0),
            ],
            dim=-1,
        )
    return torch.cat([mask_embedding, incontext, conditioning, latent], dim=-1)


def _flow_velocity(args: argparse.Namespace, trace: bool) -> None:
    import torch

    if args.autocast and not args.device.startswith("cuda"):
        raise ValueError("--autocast requires a CUDA device")
    frames = args.frames
    if frames < 2:
        raise ValueError("Flow velocity/trace requires at least 2 frames because the official block squeezes a singleton sequence")
    model = _load_flow_model(args.source_dir, args.runtime_dir, args.device)
    device = _torch_device(args.device)
    if args.vocal_codes:
        vocal_codes = _load_codes(args.vocal_codes, name="vocal codes", frames=frames)
    else:
        vocal_codes = np.zeros(frames, dtype="<i8")
    if args.bgm_codes:
        bgm_codes = _load_codes(args.bgm_codes, name="BGM codes", frames=frames)
    else:
        bgm_codes = np.zeros(frames, dtype="<i8")
    cv = torch.from_numpy(vocal_codes).to(device=device, dtype=torch.long).view(1, 1, frames)
    cb = torch.from_numpy(bgm_codes).to(device=device, dtype=torch.long).view(1, 1, frames)
    with torch.inference_mode():
        vq, _, _ = model.rvq_bestrq_emb.from_codes(cv)
        bq, _, _ = model.rvq_bestrq_bgm_emb.from_codes(cb)
        conditioning = torch.cat([vq.permute(0, 2, 1), bq.permute(0, 2, 1)], dim=-1)
        mask_ids = torch.full((1, frames), args.mask_value, device=device, dtype=torch.long)
        mask = model.mask_emb(mask_ids)
        incontext = torch.zeros((1, frames, FLOW_LATENT_DIM), device=device, dtype=torch.float32)
        noise_array, generated_seed = _load_flow_noise(args.noise, frames=frames, device=args.device, seed=args.seed)
        noise = torch.from_numpy(noise_array).to(device)
        latent = noise.clone()

        def run_velocity(state, timestep: float, *, capture_operators: bool = False):
            model_input = _flow_model_input(mask, incontext, conditioning, state, guidance=True)
            time = torch.full((2,), timestep, device=device, dtype=torch.float32)
            operator_captures: dict[str, Any] = {}
            hooks = []

            def first_tensor(value):
                if isinstance(value, torch.Tensor):
                    return value
                if isinstance(value, (tuple, list)):
                    for item in value:
                        found = first_tensor(item)
                        if found is not None:
                            return found
                return None

            def capture_output(name: str):
                def hook(_module, _inputs, output):
                    value = first_tensor(output)
                    if value is not None:
                        operator_captures[name] = value.detach().float().cpu().numpy().astype("<f4")
                return hook

            def capture_input(name: str):
                def hook(_module, inputs):
                    value = first_tensor(inputs)
                    if value is not None:
                        operator_captures[name] = value.detach().float().cpu().numpy().astype("<f4")
                return hook

            if capture_operators:
                estimator = model.cfm_wrapper.estimator
                hooks.extend((
                    estimator.adaln_single.emb.register_forward_hook(capture_output("timestep_embedding")),
                    estimator.adaln_single.linear.register_forward_hook(capture_output("timestep_modulation")),
                    estimator.h[0].register_forward_pre_hook(capture_input("block0_input")),
                    estimator.h[0].register_forward_hook(capture_output("block0_output")),
                ))
            context = torch.autocast("cuda", dtype=torch.float16) if args.autocast else nullcontext()
            try:
                with context:
                    full = model.cfm_wrapper.estimator(
                        inputs_embeds=model_input,
                        attention_mask=torch.ones((2, 1, 1, frames), device=device, dtype=torch.bool),
                        time_step=time,
                    ).last_hidden_state
            finally:
                for hook in hooks:
                    hook.remove()
            raw = full[..., -FLOW_LATENT_DIM:]
            uncond, cond = raw.chunk(2, dim=0)
            return full, uncond + args.guidance * (cond - uncond), operator_captures

        if trace:
            states = [latent.clone()]
            velocities = []
            for step in range(args.steps):
                timestep = float(step) / args.steps
                _, velocity, _ = run_velocity(latent, timestep)
                latent = latent + (1.0 / args.steps) * velocity
                velocities.append(velocity.clone())
                states.append(latent.clone())
            arrays = {
                "initial_noise": noise.float().cpu().numpy().astype("<f4"),
                "states": torch.stack(states).float().cpu().numpy().astype("<f4"),
                "velocities": torch.stack(velocities).float().cpu().numpy().astype("<f4"),
                "final_latent": latent.float().cpu().numpy().astype("<f4"),
                "final_latent_denormalized": model.normfeat.return_sample(
                    latent.permute(0, 2, 1).contiguous()
                ).float().cpu().numpy().astype("<f4"),
            }
            kind = "flow_trace"
        else:
            full, velocity, operators = run_velocity(latent, args.time, capture_operators=True)
            arrays = {
                "model_input": _flow_model_input(mask, incontext, conditioning, latent, guidance=True).float().cpu().numpy().astype("<f4"),
                "initial_noise": noise.float().cpu().numpy().astype("<f4"),
                "full_output": full.float().cpu().numpy().astype("<f4"),
                "velocity": velocity.float().cpu().numpy().astype("<f4"),
                **operators,
            }
            kind = "flow_velocity"
    _save_npz(
        args.output,
        arrays,
        {
            "format": "levo2-renderer-oracle-v1",
            "kind": kind,
            "schema_version": 1,
            "frames": frames,
            "steps": args.steps,
            "time": args.time if not trace else None,
            "guidance_scale": args.guidance,
            "provenance": _provenance(
                source_dir=args.source_dir,
                runtime_dir=args.runtime_dir,
                device=args.device,
                dtype="float16-autocast" if args.autocast else "float32",
                autocast=args.autocast,
                seed=generated_seed,
            ),
        },
    )


def _load_vae_renderer(args: argparse.Namespace):
    """Load the pinned decoder used by the official renderer."""
    flow_dir = _prepare_official_imports(args.source_dir, args.runtime_dir)
    previous_cwd = Path.cwd()
    os.chdir(flow_dir)
    try:
        from tools.get_1dvae_large import get_model
    finally:
        os.chdir(previous_cwd)
    config = _require_file(args.runtime_dir / VAE_CONFIG, "VAE config")
    checkpoint = _require_file(args.runtime_dir / VAE_CHECKPOINT, "VAE checkpoint")
    config_sha256 = _sha256_file(config)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if config_sha256 != VAE_CONFIG_SHA256 or checkpoint_sha256 != VAE_CHECKPOINT_SHA256:
        raise ValueError("VAE checkpoint/config SHA-256 does not match the pinned renderer contract")
    import torch

    device = _torch_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if dtype == torch.float16 and device.type != "cuda":
        raise ValueError("renderer float16 capture requires a CUDA device")
    model = get_model(str(config), str(checkpoint)).to(device=device, dtype=dtype).eval()
    return model, checkpoint_sha256, config_sha256


def _render_flow_window(model, vocal_codes, bgm_codes, noise, *, prior, steps, guidance, device, autocast):
    """Run one official estimator continuation window from explicit noise."""
    import torch

    frames = vocal_codes.shape[-1]
    cv = torch.from_numpy(vocal_codes).to(device=device, dtype=torch.long).view(1, 1, frames)
    cb = torch.from_numpy(bgm_codes).to(device=device, dtype=torch.long).view(1, 1, frames)
    with torch.inference_mode():
        activation_dtype = next(model.cfm_wrapper.estimator.parameters()).dtype
        vocal_q, _, _ = model.rvq_bestrq_emb.from_codes(cv)
        bgm_q, _, _ = model.rvq_bestrq_bgm_emb.from_codes(cb)
        conditioning = torch.cat([vocal_q.permute(0, 2, 1), bgm_q.permute(0, 2, 1)], dim=-1)
        mask_ids = torch.full((1, frames), 2, device=device, dtype=torch.long)
        incontext = torch.zeros((1, frames, FLOW_LATENT_DIM), device=device, dtype=activation_dtype)
        incontext_length = 0
        if prior is not None:
            incontext_length = FLOW_OVERLAP_FRAMES
            mask_ids[:, :incontext_length] = 1
            incontext[:, :incontext_length] = prior[:, -incontext_length:]
        mask = model.mask_emb(mask_ids)
        attention = (mask_ids > 0).view(1, 1, frames)
        attention = (attention & attention.transpose(-1, -2)).unsqueeze(1)
        latent = torch.from_numpy(noise).to(device=device, dtype=activation_dtype).clone()
        original_noise = latent.clone()
        t_span = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
        for step in range(steps):
            t = t_span[step]
            if incontext_length:
                latent[:, :incontext_length] = (
                    (1.0 - (1.0 - model.cfm_wrapper.sigma_min) * t) * original_noise[:, :incontext_length]
                    + t * incontext[:, :incontext_length]
                )
            model_input = _flow_model_input(mask, incontext, conditioning, latent, guidance=True)
            timestep = t.repeat(2)
            use_autocast = autocast or activation_dtype == torch.float16
            context = torch.autocast("cuda", dtype=torch.float16) if use_autocast else nullcontext()
            with context:
                full = model.cfm_wrapper.estimator(
                    inputs_embeds=model_input,
                    attention_mask=attention.repeat(2, 1, 1, 1),
                    time_step=timestep,
                ).last_hidden_state
            raw = full[..., -FLOW_LATENT_DIM:]
            uncond, cond = raw.chunk(2, dim=0)
            latent = latent + (t_span[step + 1] - t) * (uncond + guidance * (cond - uncond))
        if incontext_length:
            latent[:, :incontext_length] = incontext[:, :incontext_length]
        normalized = latent.float()
        denormalized = model.normfeat.return_sample(normalized.permute(0, 2, 1).contiguous()).float()
    return normalized, denormalized


def _crossfade_audio(windows, *, target_samples: int):
    import torch

    if not windows:
        raise ValueError("at least one decoded audio window is required")
    overlap = FLOW_OVERLAP_FRAMES * VAE_DOWNSAMPLING
    output = windows[0]
    blend = torch.linspace(0.0, 1.0, overlap, device=output.device, dtype=output.dtype).view(1, -1)
    for current in windows[1:]:
        if output.shape[-1] < overlap or current.shape[-1] < overlap:
            raise ValueError("decoded VAE window is shorter than the required overlap")
        output = output.clone()
        output[:, -overlap:] = output[:, -overlap:] * (1.0 - blend) + current[:, :overlap] * blend
        output = torch.cat([output, current[:, overlap:]], dim=-1)
    return output[:, :target_samples]


def _render_contract(args: argparse.Namespace) -> None:
    tokens = _load_render_tokens(args.tokens)
    padded, starts = _render_window_plan(tokens.shape[1])
    noise = _load_render_noise(args.noise, windows=len(starts))
    _save_npz(
        args.output,
        {
            "tokens": tokens.astype("<i4"),
            "window_starts": starts,
            "window_major_noise": noise,
        },
        {
            "format": "levo2-renderer-oracle-v1",
            "kind": "renderer_contract",
            "schema_version": 1,
            "frames": int(tokens.shape[1]),
            "padded_frames": int(padded),
            "window_count": int(len(starts)),
            "window_frames": FLOW_WINDOW_FRAMES,
            "hop_frames": FLOW_HOP_FRAMES,
            "overlap_frames": FLOW_OVERLAP_FRAMES,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "dry_run": True,
            "input_sha256": {"tokens": _sha256_file(args.tokens), "noise": _sha256_file(args.noise)},
            "provenance": _provenance(
                source_dir=args.source_dir, runtime_dir=args.runtime_dir,
                device=args.device, dtype=args.dtype, autocast=args.autocast, seed=None,
            ),
        },
    )


def _render_end_to_end(args: argparse.Namespace) -> None:
    import torch

    if args.autocast and not args.device.startswith("cuda"):
        raise ValueError("--autocast requires a CUDA device")
    if args.steps not in (1, FLOW_DEFAULT_STEPS):
        raise ValueError("end-to-end renderer oracle --steps must be 1 or 50")
    tokens = _load_render_tokens(args.tokens)
    padded, starts = _render_window_plan(tokens.shape[1])
    noise = _load_render_noise(args.noise, windows=len(starts))
    device = _torch_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    flow = _load_flow_model(args.source_dir, args.runtime_dir, args.device, dtype=dtype)
    normalized_windows = []
    denormalized_windows = []
    prior = None
    padded_tokens = np.tile(tokens, (1, (padded + tokens.shape[1] - 1) // tokens.shape[1]))[:, :padded]
    with torch.inference_mode():
        for index, start in enumerate(starts.tolist()):
            vocal = padded_tokens[1, start:start + FLOW_WINDOW_FRAMES]
            bgm = padded_tokens[2, start:start + FLOW_WINDOW_FRAMES]
            normalized, denormalized = _render_flow_window(
                flow, vocal, bgm, noise[index:index + 1], prior=prior,
                steps=args.steps, guidance=args.guidance, device=device,
                autocast=args.autocast or dtype == torch.float16,
            )
            normalized_windows.append(normalized)
            denormalized_windows.append(denormalized)
            prior = normalized

    vae, vae_checkpoint_sha256, vae_config_sha256 = _load_vae_renderer(args)
    decoded_windows = []
    vae_dtype = next(vae.parameters()).dtype
    with torch.inference_mode():
        for latent in denormalized_windows:
            decoded_windows.append(vae.decode_audio(latent.to(dtype=vae_dtype), chunked=False)[0].float())
    audio = _crossfade_audio(decoded_windows, target_samples=tokens.shape[1] * VAE_DOWNSAMPLING)
    arrays: dict[str, np.ndarray] = {
        "tokens": tokens.astype("<i4"),
        "window_starts": starts,
        "window_major_noise": noise,
        "final_latent_normalized": torch.cat(normalized_windows, dim=0).cpu().numpy().astype("<f4"),
        "final_latent_denormalized": torch.cat(denormalized_windows, dim=0).cpu().numpy().astype("<f4"),
        "audio": audio.cpu().numpy().astype("<f4"),
    }
    for index, decoded in enumerate(decoded_windows):
        arrays[f"decoded_window_{index:03d}"] = decoded.cpu().numpy().astype("<f4")
    _save_npz(
        args.output, arrays,
        {
            "format": "levo2-renderer-oracle-v1",
            "kind": "renderer_end_to_end",
            "schema_version": 1,
            "frames": int(tokens.shape[1]),
            "padded_frames": int(padded),
            "window_count": int(len(starts)),
            "window_frames": FLOW_WINDOW_FRAMES,
            "hop_frames": FLOW_HOP_FRAMES,
            "overlap_frames": FLOW_OVERLAP_FRAMES,
            "sample_rate": 48_000,
            "downsampling_ratio": VAE_DOWNSAMPLING,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "input_sha256": {"tokens": _sha256_file(args.tokens), "noise": _sha256_file(args.noise)},
            "flow_checkpoint_sha256": FLOW_CHECKPOINT_SHA256,
            "vae_checkpoint_sha256": vae_checkpoint_sha256,
            "vae_config_sha256": vae_config_sha256,
            "provenance": _provenance(
                source_dir=args.source_dir, runtime_dir=args.runtime_dir,
                device=args.device, dtype=args.dtype, autocast=args.autocast, seed=None,
            ),
        },
    )
    if args.wav_output:
        import soundfile as sf

        args.wav_output.parent.mkdir(parents=True, exist_ok=True)
        # The assembled renderer waveform is channel-major [2, samples]; the
        # per-stage VAE capture keeps its batch axis, so only that one is
        # indexed before transposing.
        sf.write(str(args.wav_output), arrays["audio"].T, 48_000, subtype="FLOAT")


def _load_latent(path: Path | None, *, frames: int, device: str, seed: int) -> tuple[Any, int | None]:
    import torch

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"latent is missing: {path}")
        latent = _load_npy(path, name="latent")
        if latent.ndim == 2:
            latent = latent[None]
        if latent.shape != (1, FLOW_LATENT_DIM, frames):
            raise ValueError(f"latent must have shape [64,T] or [1,64,T] with T={frames}, got {latent.shape}")
        return torch.from_numpy(latent.astype("<f4", copy=False)).to(device=device), None
    value = _random_f32((1, FLOW_LATENT_DIM, frames), device=device, seed=seed)
    return torch.from_numpy(value).to(device=device), seed


def _vae_capture(args: argparse.Namespace) -> None:
    import torch

    if args.frames <= 0 or args.frames > 32:
        raise ValueError("VAE oracle --frames must be in [1,32] unless a future large-fixture mode is added")
    flow_dir = _prepare_official_imports(args.source_dir, args.runtime_dir)
    previous_cwd = Path.cwd()
    os.chdir(flow_dir)
    try:
        from tools.get_1dvae_large import get_model
    finally:
        os.chdir(previous_cwd)
    config = _require_file(args.runtime_dir / VAE_CONFIG, "VAE config")
    checkpoint = _require_file(args.runtime_dir / VAE_CHECKPOINT, "VAE checkpoint")
    config_sha256 = _sha256_file(config)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if config_sha256 != VAE_CONFIG_SHA256 or checkpoint_sha256 != VAE_CHECKPOINT_SHA256:
        raise ValueError("VAE checkpoint/config SHA-256 does not match the pinned renderer contract")
    model = get_model(str(config), str(checkpoint))
    device = _torch_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if dtype == torch.float16 and device.type != "cuda":
        raise ValueError("VAE float16 capture requires a CUDA device")
    model = model.to(device=device, dtype=dtype).eval()
    latent, generated_seed = _load_latent(args.latent, frames=args.frames, device=args.device, seed=args.seed)
    latent = latent.to(dtype=dtype)
    captured: dict[str, Any] = {}
    hooks = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().float().cpu().numpy()

        return hook

    for index, layer in enumerate(model.decoder.layers):
        hooks.append(layer.register_forward_hook(capture(f"decoder_layer_{index:02d}")))
    with torch.inference_mode():
        audio = model.decode_audio(latent, chunked=False)
    for hook in hooks:
        hook.remove()
    captured["latent_input"] = latent.float().cpu().numpy()
    captured["audio"] = audio.detach().float().cpu().numpy()
    _save_npz(
        args.output,
        captured,
        {
            "format": "levo2-renderer-oracle-v1",
            "kind": "vae_stages",
            "schema_version": 1,
            "latent_frames": args.frames,
            "sample_rate": 48_000,
            "downsampling_ratio": VAE_DOWNSAMPLING,
            "generated_latent_seed": generated_seed,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "provenance": _provenance(
                source_dir=args.source_dir,
                runtime_dir=args.runtime_dir,
                device=args.device,
                dtype=args.dtype,
                autocast=False,
                seed=args.seed,
            ),
        },
    )
    if args.wav_output:
        import soundfile as sf

        wav = captured["audio"][0].T
        args.wav_output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.wav_output), wav, 48_000, subtype="FLOAT")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    flow = subparsers.add_parser("flow", help="inventory or capture official Flow tensors")
    _add_common(flow)
    flow.add_argument("--mode", choices=("inventory", "inputs", "velocity", "trace"), default="inventory")
    flow.add_argument("--frames", type=int, default=FLOW_DEFAULT_FRAMES)
    flow.add_argument("--vocal-codes", type=Path)
    flow.add_argument("--bgm-codes", type=Path)
    flow.add_argument("--noise", type=Path, help="optional F32 .npy initial latent noise [T,64] or [1,T,64]")
    flow.add_argument("--mask-value", type=int, default=2)
    flow.add_argument("--time", type=float, default=0.0)
    flow.add_argument("--steps", type=int, default=FLOW_DEFAULT_STEPS)
    flow.add_argument("--guidance", type=float, default=1.5)
    flow.add_argument("--autocast", action="store_true")

    vae = subparsers.add_parser("vae", help="capture official Oobleck VAE decoder stages")
    _add_common(vae)
    vae.add_argument("--frames", type=int, default=1)
    vae.add_argument("--latent", type=Path)
    vae.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    vae.add_argument("--wav-output", type=Path)

    render = subparsers.add_parser("render", help="run the deterministic official Flow/VAE renderer oracle")
    _add_common(render)
    render.add_argument("--tokens", type=Path, required=True, help="canonical integer token array [3,T] (mixed,vocal,bgm)")
    render.add_argument("--noise", type=Path, required=True, help="explicit F32 window-major noise [windows,1000,64]")
    render.add_argument("--steps", type=int, choices=(1, FLOW_DEFAULT_STEPS), default=FLOW_DEFAULT_STEPS)
    render.add_argument("--guidance", type=float, default=1.5)
    render.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    render.add_argument("--autocast", action="store_true")
    render.add_argument("--dry-run", action="store_true", help="validate inputs and export the window plan without loading checkpoints")
    render.add_argument("--wav-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.source_dir = args.source_dir.resolve()
    args.runtime_dir = args.runtime_dir.resolve()
    args.output = args.output.resolve()
    if args.command == "render":
        args.tokens = args.tokens.resolve()
        args.noise = args.noise.resolve()
        if args.wav_output:
            args.wav_output = args.wav_output.resolve()
    if args.command == "flow":
        if args.mode == "inventory":
            _flow_inventory(args.source_dir, args.runtime_dir, args.output, args.device, args.seed)
        elif args.mode == "inputs":
            _flow_inputs(args)
        else:
            if args.steps <= 0:
                raise ValueError("--steps must be positive")
            if not 0.0 <= args.time <= 1.0:
                raise ValueError("--time must be in [0,1]")
            _flow_velocity(args, trace=args.mode == "trace")
    elif args.command == "vae":
        _vae_capture(args)
    elif args.dry_run:
        _render_contract(args)
    else:
        _render_end_to_end(args)


if __name__ == "__main__":
    main()
