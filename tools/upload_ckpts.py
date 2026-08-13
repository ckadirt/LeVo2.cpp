#!/usr/bin/env python3
"""Publish the frozen LeVo2 starter catalog to immutable Cloudflare R2 keys.

The script is intentionally data-first: the tracked component identities below
are the authority, not filenames discovered from a local model directory. It
downloads, hashes, uploads, and deletes one source file at a time so its peak
workspace use is the largest component rather than the whole catalog.

``--catalog-only`` needs no cloud credentials and regenerates the tracked
public catalog. ``--publish`` is deliberately explicit; it refuses to replace
an existing object and always verifies the public CDN contract afterwards.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HF_REPO = "ckadirt/LeVo2-GGUF"
HF_REVISION = "9d7b5746fdc74fdc80f85295e7b6c783be3703da"
PUBLIC_BASE_URL = "https://cantor-ckpts.ckadirt.xyz"
R2_PREFIX = "levo2-1.0"
MODEL_NAME = "levo2"
LICENSE = (
    "Tencent SongGeneration License — academic, research, and education use only; "
    "commercial and production use prohibited."
)
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"
BACKENDS = ["cuda12", "vulkan", "cpu"]


@dataclass(frozen=True)
class Component:
    filename: str
    role: str
    quant: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Variant:
    tag: str
    lm: str
    dit: str
    vae: str
    vram_bytes: int


COMPONENTS = (
    Component(
        filename="LeVo2-v2-medium-Q4_K_M.gguf",
        role="lm",
        quant="Q4_K_M",
        bytes=1_916_558_688,
        sha256="9412bb0ef5373fd0b9085fd24e4b5ffa0d341efece3829067563816d44d4aeca",
    ),
    Component(
        filename="LeVo2-v2-large-Q6_K.gguf",
        role="lm",
        quant="Q6_K",
        bytes=4_102_556_992,
        sha256="5f46280c137a5ee425bd86d852165a1b6065684e9e401efc4b58889155a8a5c6",
    ),
    Component(
        filename="LeVo2-v2-large-F16.gguf",
        role="lm",
        quant="F16",
        bytes=9_986_305_440,
        sha256="368cba66fabbdca3d208d4c15a9c9c2059ea5d8ca2a822ee3a3f1d9f854134d5",
    ),
    Component(
        filename="LeVo2-v2-flow-Q6_K.gguf",
        role="dit",
        quant="Q6_K",
        bytes=707_404_288,
        sha256="a3579de6915c5ea060072b83af4e158e729834a8a13f62f1b370ef82e7317c00",
    ),
    Component(
        filename="LeVo2-v2-vae-F16.gguf",
        role="vae",
        quant="F16",
        bytes=168_805_120,
        sha256="23e5b11558ae332fbe216d9a06775884469fcbf32236c26ab52defa18c5c8398",
    ),
)
COMPONENT_BY_FILENAME = {component.filename: component for component in COMPONENTS}

VARIANTS = (
    Variant(
        tag="1.0-fast",
        lm="LeVo2-v2-medium-Q4_K_M.gguf",
        dit="LeVo2-v2-flow-Q6_K.gguf",
        vae="LeVo2-v2-vae-F16.gguf",
        vram_bytes=4 * 1024**3,
    ),
    Variant(
        tag="1.0-balanced",
        lm="LeVo2-v2-large-Q6_K.gguf",
        dit="LeVo2-v2-flow-Q6_K.gguf",
        vae="LeVo2-v2-vae-F16.gguf",
        vram_bytes=8 * 1024**3,
    ),
    Variant(
        tag="1.0-quality",
        lm="LeVo2-v2-large-F16.gguf",
        dit="LeVo2-v2-flow-Q6_K.gguf",
        vae="LeVo2-v2-vae-F16.gguf",
        vram_bytes=20 * 1024**3,
    ),
)


class PublicationError(RuntimeError):
    """An immutable-object or public-CDN invariant was violated."""


def object_key(component: Component) -> str:
    return f"{R2_PREFIX}/{component.filename}"


def public_url(component: Component) -> str:
    return f"{PUBLIC_BASE_URL}/{object_key(component)}"


def component_catalog_entry(component: Component) -> dict[str, Any]:
    return {
        "role": component.role,
        "blob": f"sha256:{component.sha256}",
        "url": public_url(component),
        "bytes": component.bytes,
        "quant": component.quant,
    }


def catalog() -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    for variant in VARIANTS:
        components = [
            component_catalog_entry(COMPONENT_BY_FILENAME[variant.lm]),
            component_catalog_entry(COMPONENT_BY_FILENAME[variant.dit]),
            component_catalog_entry(COMPONENT_BY_FILENAME[variant.vae]),
        ]
        variants.append(
            {
                "tag": variant.tag,
                "components": components,
                "needs": {"vram_bytes": variant.vram_bytes, "backends": BACKENDS},
            }
        )
    return {
        "schema": 1,
        "models": [
            {
                "name": MODEL_NAME,
                "licence": LICENSE,
                "engine": MODEL_NAME,
                "variants": variants,
            }
        ],
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_download(path: Path, component: Component) -> None:
    actual_size = path.stat().st_size
    if actual_size != component.bytes:
        raise PublicationError(
            f"{component.filename}: expected {component.bytes} bytes, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != component.sha256:
        raise PublicationError(
            f"{component.filename}: expected sha256 {component.sha256}, got {actual_sha256}"
        )


def manifest_record(component: Component) -> dict[str, Any]:
    return {
        "sha256": component.sha256,
        "bytes": component.bytes,
        "key": object_key(component),
        "role": component.role,
        "quant": component.quant,
    }


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PublicationError(f"invalid publication manifest: {path}") from exc
    if not isinstance(value, dict) or not all(
        isinstance(filename, str) and isinstance(record, dict)
        for filename, record in value.items()
    ):
        raise PublicationError(f"invalid publication manifest shape: {path}")
    return value


def update_manifest(
    path: Path, manifest: dict[str, dict[str, Any]], component: Component
) -> None:
    expected = manifest_record(component)
    current = manifest.get(component.filename)
    if current is not None and current != expected:
        raise PublicationError(
            f"{component.filename}: private manifest identity differs from the frozen release"
        )
    manifest[component.filename] = expected
    write_json(path, manifest)


class publication_lock:
    """A non-blocking local lock preventing concurrent mutable publish state."""

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
        self._file: Any | None = None

    def __enter__(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.close()
            self._file = None
            raise PublicationError(
                f"another LeVo2 R2 publish is already using {self._path}"
            ) from exc
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None


def s3_client_from_environment() -> tuple[Any, str]:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise PublicationError(
            "missing uploader dependencies; install tools/requirements-cloudflare.txt"
        ) from exc

    required = (
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise PublicationError("missing required R2 environment variables: " + ", ".join(missing))
    # R2 rejects automatic CRC32 request checksums from newer botocore multipart
    # uploads. Keep this explicit even if a future botocore changes its default.
    config = Config(request_checksum_calculation="when_required")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=config,
    )
    return client, os.environ["R2_BUCKET"]


def head_object(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # ClientError is intentionally lazy-imported with boto3.
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def verify_existing_object(head: dict[str, Any], component: Component) -> None:
    if head.get("ContentLength") != component.bytes:
        raise PublicationError(
            f"R2 object already exists with a different length: {object_key(component)}"
        )
    metadata = {str(key).lower(): str(value) for key, value in head.get("Metadata", {}).items()}
    if metadata.get("sha256") != component.sha256:
        raise PublicationError(
            f"R2 object already exists without the expected sha256 metadata: {object_key(component)}"
        )


def download_from_hugging_face(component: Component, destination: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PublicationError(
            "missing uploader dependencies; install tools/requirements-cloudflare.txt"
        ) from exc
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="model",
        filename=component.filename,
        revision=HF_REVISION,
        local_dir=destination,
    )
    return Path(local_path)


def upload_component(client: Any, bucket: str, component: Component) -> None:
    key = object_key(component)
    existing = head_object(client, bucket, key)
    if existing is not None:
        verify_existing_object(existing, component)
        print(f"already immutable: {key}")
        return

    with tempfile.TemporaryDirectory(prefix="levo2-r2-") as directory:
        local_path = download_from_hugging_face(component, Path(directory))
        verify_download(local_path, component)
        # This preflight deliberately makes overwrite a hard error. The R2 key
        # is content-addressed by this frozen manifest and must never be reused.
        if head_object(client, bucket, key) is not None:
            raise PublicationError(f"refusing to overwrite R2 object: {key}")
        client.upload_file(
            str(local_path),
            bucket,
            key,
            ExtraArgs={
                "ContentType": "application/octet-stream",
                "CacheControl": IMMUTABLE_CACHE_CONTROL,
                "Metadata": {
                    "sha256": component.sha256,
                    "source": HF_REPO,
                    "source-revision": HF_REVISION,
                },
            },
        )
    uploaded = head_object(client, bucket, key)
    if uploaded is None:
        raise PublicationError(f"R2 object disappeared after upload: {key}")
    verify_existing_object(uploaded, component)
    print(f"uploaded immutable: {key}")


def _parse_curl_headers(raw_headers: str, url: str) -> tuple[int, dict[str, str]]:
    blocks = [block for block in raw_headers.replace("\r\n", "\n").split("\n\n") if block.strip()]
    if not blocks:
        raise PublicationError(f"curl returned no response headers: {url}")
    lines = blocks[-1].splitlines()
    try:
        status = int(lines[0].split()[1])
    except (IndexError, ValueError) as exc:
        raise PublicationError(f"curl returned malformed response headers: {url}") from exc
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.lower()] = value.strip()
    return status, headers


def curl_public_request(url: str, extra_args: list[str]) -> tuple[int, dict[str, str], bytes]:
    curl = shutil.which("curl")
    if curl is None:
        raise PublicationError("public R2 verification requires curl on PATH")
    with tempfile.TemporaryDirectory(prefix="levo2-r2-verify-") as directory:
        root = Path(directory)
        headers_path = root / "headers.txt"
        body_path = root / "body.bin"
        command = [
            curl,
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "60",
            "--user-agent",
            "LeVo2-R2-Publisher/1.0",
            "--dump-header",
            str(headers_path),
            "--output",
            str(body_path),
            *extra_args,
            url,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or f"curl exit {result.returncode}"
            raise PublicationError(f"public object request failed: {url}: {detail}")
        status, headers = _parse_curl_headers(headers_path.read_text(encoding="iso-8859-1"), url)
        return status, headers, body_path.read_bytes()


def verify_public(component: Component) -> None:
    url = public_url(component)
    # Cloudflare serves the endpoint correctly to curl but rejects or drops
    # Python's stdlib HTTP client. Use curl for the same HEAD/range semantics
    # the node's downloader relies on, with an explicit release user agent.
    status, headers, _ = curl_public_request(url, ["--head"])
    if status != 200:
        raise PublicationError(f"public HEAD returned {status}: {url}")
    if headers.get("content-length") != str(component.bytes):
        raise PublicationError(f"public Content-Length mismatch: {url}")
    cache_control = headers.get("cache-control", "")
    if not all(token in cache_control for token in ("public", "max-age=31536000", "immutable")):
        raise PublicationError(f"public immutable Cache-Control missing: {url}")
    if "bytes" not in headers.get("accept-ranges", "").lower():
        raise PublicationError(f"public byte ranges unavailable: {url}")
    status, headers, body = curl_public_request(url, ["--range", "0-0"])
    if status != 206:
        raise PublicationError(f"public range request did not return 206: {url}")
    if headers.get("content-length") != "1":
        raise PublicationError(f"public range length mismatch: {url}")
    expected_range = f"bytes 0-0/{component.bytes}"
    if headers.get("content-range") != expected_range:
        raise PublicationError(f"public Content-Range mismatch: {url}")
    if len(body) != 1:
        raise PublicationError(f"public range body mismatch: {url}")
    print(f"verified public: {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--catalog-only",
        action="store_true",
        help="regenerate the public catalog only; no network or credentials are used",
    )
    action.add_argument(
        "--publish",
        action="store_true",
        help="upload frozen objects to R2 and verify all public URLs",
    )
    action.add_argument(
        "--verify-public",
        action="store_true",
        help="verify all previously published public URLs without cloud credentials",
    )
    parser.add_argument(
        "--catalog-out",
        type=Path,
        default=Path("docs/cloudflare-catalog-v1.json"),
        help="catalog destination (default: docs/cloudflare-catalog-v1.json)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release-state/levo2-r2-manifest.json"),
        help="private local idempotency manifest used by --publish",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_json(args.catalog_out, catalog())
    print(f"wrote catalog: {args.catalog_out}")
    if args.catalog_only:
        return 0
    if args.verify_public:
        for component in COMPONENTS:
            verify_public(component)
        return 0

    with publication_lock(args.manifest):
        client, bucket = s3_client_from_environment()
        manifest = load_manifest(args.manifest)
        for component in COMPONENTS:
            upload_component(client, bucket, component)
            update_manifest(args.manifest, manifest, component)
        for component in COMPONENTS:
            verify_public(component)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublicationError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
