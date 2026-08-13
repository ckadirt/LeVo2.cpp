# Cloudflare R2 / Cantor release implementation report

## Status

In progress. This is the live evidence record for the immutable R2 model and
engine publication work. It supplements the existing quantization and v2-large
reports.

## Scope boundary

LeVo2 generates and commits the R2 catalog and a ready-to-paste backend
fragment. The maintainer, not this repository, appends the backend fragment to
Cantor relay and deploys the Worker.

## Milestone log

### 2026-08-13 — release contract frozen

- Confirmed that v2-large changes the LeLM only; the v2 Flow and VAE renderer
  artifacts remain shared component identities.
- Refined the three starter tags so the renderer is exactly shared: `1.0-fast`
  (medium Q4 LM), `1.0-balanced` (large Q6 LM), and `1.0-quality` (large F16
  LM) all use Flow Q6_K and VAE F16. This makes LeLM the only component that
  changes between tags and enables relay verification of both shared renderer
  blobs.
- Confirmed the C ABI identifiers: engine model `levo2`, ABI `1`, and catalog
  roles `lm`, `dit`, and `vae`.
- Identified the release-specific packaging repair: the build produces
  `liblevo-cantor-engine.so` with a build-directory RUNPATH, while Cantor
  requires a packaged `libcantor_engine.so`. The engine workflow will rename
  only the packaged file and rewrite its runtime path to `$ORIGIN`.
- Per maintainer direction, no Cantor relay manifest is edited or deployed by
  this implementation. The project will instead generate the exact manifest
  fragment and document the manual handoff.

### 2026-08-13 — uploader contract implementation

- Added `tools/upload_ckpts.py` with five unique pinned source identities:
  three LeLM files plus the shared Flow Q6_K and VAE F16 files. It verifies the
  expected byte length and SHA-256 before upload, rejects R2 overwrite attempts,
  writes a local idempotency manifest, and verifies the public immutable/range
  response contract after every publication.
- Added the deterministic `docs/cloudflare-catalog-v1.json` handoff and tests
  that prove it is generated from the publisher, uses the exact ABI role names,
  uses `sha256:`-prefixed model digests, and changes only `lm` across tiers.
- Added a non-blocking local publisher lock after detecting that detached shell
  monitoring could accidentally start a second publication. A second local
  invocation now fails before it can race an immutable R2 key.
- **Deviation resolved:** Cloudflare serves the custom domain to ordinary
  clients but rejects or drops Python's stdlib HTTP client. The public verifier
  now invokes `curl` with the explicit `LeVo2-R2-Publisher/1.0` identifier;
  direct validation then returned HTTP 200, the immutable cache policy, and a
  one-byte HTTP 206 range response for the first uploaded blob.
- **Deviation recorded:** the referenced canonical `upload_ckpts.py` was not
  present in this workspace or the available Cantor/ACE repositories. A local,
  compatible implementation was added instead, including the required
  botocore `request_checksum_calculation="when_required"` workaround for R2.

### 2026-08-13 — engine release automation

- Added `.github/workflows/engine-release.yml`. It builds Linux x86_64 CPU,
  CUDA 12, and Vulkan cells with dynamic GGML backends and every CPU dispatch
  variant, then publishes immutable commit-addressed R2 tarballs only after
  packaging gates pass.
- The archive deliberately contains only a single engine root with
  `libcantor_engine.so`, GGML base/runtime libraries, all CPU plugins, and the
  selected GPU plugin. Its real shared objects are patched to `$ORIGIN` and an
  extracted archive is loaded through `ctypes` with `LD_LIBRARY_PATH` removed.
- The workflow emits, but does not deploy, a hash-bearing
  `levo2-backends-v1.fragment.json` artifact. This is the sole backend-manifest
  handoff for the maintainer to append manually.
- Locally built the portable CPU engine configuration (14 CPU plugins) and
  proved that the extracted tarball reports Cantor ABI `1` and engine `levo2`
  with no `LD_LIBRARY_PATH`.

### 2026-08-13 — manual backend-fragment handoff

- Published engine archives from source commit `b406beb25d12`: CPU
  (`6,356,609` bytes,
  `868c6429a4d232139149a038f6ddfb834aadf4b8ffd4152d7f45e6b8ed06afba`)
  and Vulkan (`22,315,403` bytes,
  `1d6e5fae44873421e6f4c15270e1826184738eda68b42a357bee148790b9e192`).
  Both have immutable cache policy and public byte-range support.
- Added `docs/levo2-backends-v1.fragment.json` for the maintainer's manual
  Cantor handoff. CUDA 12 intentionally remains a non-schema-valid,
  conspicuous placeholder (`sha256` replacement text and byte count `0`) until
  its lengthy CI build finishes; it must be filled with that artifact's bare
  SHA-256 and byte size before relay validation/deployment.
