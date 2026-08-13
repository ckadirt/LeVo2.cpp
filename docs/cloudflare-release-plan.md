# Cloudflare R2 / Cantor release plan

## Status

Approved for implementation on 2026-08-13. This document covers immutable R2
publication and generated Cantor relay fragments for the first LeVo2 catalog.
It does **not** authorize modifying or deploying the Cantor relay repository:
the generated backend fragment is committed here for the maintainer to append
manually.

## Shared renderer contract

SongGeneration v2-large changes only the LeLM checkpoint and its architecture
(width, FFN, main-block count, and attention-head count). It retains exactly
the v2-medium token contract, separate-token Flow checkpoint, and Stable Audio
1920 VAE. Catalog component roles are therefore:

| Role | Meaning | Shared between starter variants? |
| --- | --- | --- |
| `lm` | LeLM lyrics/style-to-code model | No: selected per variant |
| `dit` | self-contained Flow GGUF | Yes: Flow Q6_K is identical in every starter variant |
| `vae` | Oobleck VAE GGUF | Yes: F16-storage VAE in every variant |

The engine reports `cantor_engine_model() == "levo2"`, ABI `1`, and accepts
only those three role strings. The public model name, engine name, and backend
manifest name must remain `levo2` exactly.

## Initial catalog

The first release deliberately exposes three materially different user choices
rather than three adjacent precision tiers. Object bytes and SHA-256 values
below are frozen local/Hugging Face artifact identities.

| Tag | `lm` | `dit` | `vae` | Transfer bytes | Intent |
| --- | --- | --- | --- | ---: | --- |
| `1.0-fast` | v2-medium Q4_K_M | Flow Q6_K | F16 storage | 2,792,768,096 | Smallest complete pipeline; CPU/small-GPU oriented. |
| `1.0-balanced` | v2-large Q6_K | Flow Q6_K | F16 storage | 4,978,766,400 | Default large-model tier; Q6 is the quality-favored Flow tier. |
| `1.0-quality` | v2-large F16 | Flow Q6_K | F16 storage | 10,862,514,848 | Fidelity-first LM tier with the same high-quality shared renderer. |

The Flow and VAE are intentionally byte-identical across all three tags, so
only the LeLM changes when a user changes tier. Flow Q6_K is the strongest
shared renderer tier validated for this first menu; an F32 Flow pairing may be
published later as a deliberately non-shared specialist variant.

`needs.vram_bytes` is not an automatic variant selector in the current Cantor
node. It is published as an eviction/residency budget and documentation only.
Before R2 publication, fresh CUDA peak-memory probes set the conservative
initial budgets to 4 GiB (`fast`), 8 GiB (`balanced`), and 20 GiB (`quality`).
The 20 GiB value is intentionally above the observed 17.6 GiB v2-large-F16
LeLM load on the RTX 4090. The engine currently reloads models per stage and
reports zero persistent residency, so these values must not be advertised as a
hard admission-control mechanism.

Every variant documents `cuda12`, `vulkan`, and `cpu` backends. Runtime backend
selection remains the node/engine's CUDA → other GPU → CPU decision, after the
user has selected a model tag.

## Immutable R2 objects

Model blobs use `levo2-1.0/<filename>.gguf`. The uploader must:

1. Download one pinned file at a time from
   `ckadirt/LeVo2-GGUF@9d7b5746fdc74fdc80f85295e7b6c783be3703da`.
2. Verify its exact expected bytes and SHA-256 before upload.
3. Refuse to overwrite an existing R2 key. A pre-existing key is accepted only
   when its size and stored SHA-256 metadata match the frozen identity.
4. Set `Cache-Control: public, max-age=31536000, immutable` and persist a
   private idempotency manifest with filename, digest, byte count, role,
   quantization, and R2 key.
5. Verify the public URL's content length, immutable cache policy,
   `Accept-Ranges: bytes`, and a `Range: bytes=0-0` response with status 206.

The catalog has `sha256:`-prefixed component digests. The separate backend
fragment has bare 64-hex digests: those formats must never be mixed.

## Engine tarballs

The initial architecture matrix is Linux `x86_64` for `cpu`, `cuda12`, and
`vulkan`. Each output is an immutable object at:

```text
backends/<source-commit-12>/levo2-engine-<backend>-x86_64.tar.gz
```

Each archive has one root directory and contains the exact node-facing
`libcantor_engine.so`, `libggml.so*`, `libggml-base.so*`, **all**
`libggml-cpu*.so*` variants, and its selected GPU backend library where
applicable. The workflow builds with `GGML_CPU_ALL_VARIANTS=ON` and
`GGML_BACKEND_DL=ON`, preserves SONAME symlinks with `cp -a`, rewrites the
current build-tree RUNPATH to `$ORIGIN`, and proves an extracted engine can be
`dlopen`ed without `LD_LIBRARY_PATH`.

It emits a complete, hash-bearing `levo2-backends-v1.fragment.json` GitHub
Actions artifact. That fragment is committed as release evidence and is ready
to paste manually into Cantor's `relay/public/backends/v1.json`; this project
does not edit or deploy that relay file.

## Relay handoff

The generated `docs/cloudflare-catalog-v1.json` is ready to copy into Cantor's
`relay/public/catalog/v1.json`. The Cantor verifier must regard `dit` as a
shared role in addition to `vae`/`embed`, otherwise it will not catch a
mistaken multi-gigabyte Flow digest drift. The relay maintainer performs the
final merge, `npm run check`, and Wrangler deployment.

## Licensing

The catalog licence string is the real SongGeneration restriction:
`Tencent SongGeneration License — academic, research, and education use only;
commercial and production use prohibited.` No catalog or R2 object weakens the
upstream terms in [`LICENSE`](../LICENSE).
