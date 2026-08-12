#!/usr/bin/env python3
"""Validate a LeVo2.cpp token artifact and exercise NumPy's round trip.

The C++ writer stores the tensor payload as little-endian int32 values in
stream-major C order. This helper intentionally uses NumPy's public loader,
with pickle disabled, so it is also useful as a decoder-side smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MAX_TOKEN = 16385


def validate(path: Path) -> np.ndarray:
    path = path.resolve()
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError("token artifact is not a NumPy array")
    if array.ndim != 2 or array.shape[0] != 3:
        raise ValueError(f"expected shape [3,T], got {array.shape}")
    if array.dtype != np.dtype("<i4"):
        raise ValueError(f"expected little-endian int32, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("token artifact must be C-contiguous")
    if np.any(array < 0) or np.any(array > MAX_TOKEN):
        raise ValueError("token ID is outside [0, 16385]")

    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["artifact"]
    tensor = manifest["tensor"]
    if artifact["format"] != "numpy-npy" or artifact["npy_version"] not in {"1.0", "2.0"}:
        raise ValueError("manifest artifact format is not canonical NumPy NPY")
    if tensor["dtype"] != "int32" or tensor["order"] != "C":
        raise ValueError("manifest tensor dtype/order does not match the canonical format")
    if tensor["shape"] != [3, int(array.shape[1])]:
        raise ValueError("manifest tensor shape does not match the NumPy tensor")
    digest = hashlib.sha256(array.astype("<i4", copy=False).tobytes(order="C")).hexdigest()
    if tensor["sha256"] != digest:
        raise ValueError("manifest tensor SHA-256 does not match the NumPy tensor")
    if manifest["duration"]["frames"] != int(array.shape[1]):
        raise ValueError("manifest duration frame count does not match the tensor")
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    array = validate(args.artifact)
    # Save/reload through NumPy to catch accidental non-standard headers or
    # byte ordering changes in the interchange contract.
    round_trip = args.artifact.with_suffix(".roundtrip.npy")
    np.save(round_trip, array, allow_pickle=False)
    try:
        reloaded = np.load(round_trip, allow_pickle=False)
        if not np.array_equal(array, reloaded):
            raise ValueError("NumPy round trip changed token values")
    finally:
        round_trip.unlink(missing_ok=True)
    print(f"valid LeVo token artifact: shape={tuple(array.shape)} dtype={array.dtype}")


if __name__ == "__main__":
    main()
