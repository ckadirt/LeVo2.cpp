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
