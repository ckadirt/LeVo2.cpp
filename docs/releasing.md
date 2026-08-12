# Release procedure

## Licensing

LeVo inference-enabling code and weights are restricted by Tencent's
SongGeneration license to academic, research, and education use and may not be
used commercially or in production. Converted GGUF weights retain that
restriction. Model cards, release notes, and repository documentation must state
it prominently and must not substitute a permissive SPDX label.

GGML is MIT-licensed. Qwen2 tokenizer assets carry their upstream Apache-2.0
terms. Required notices and source links ship with source and artifact releases.

## Pre-release checklist

1. All milestone tests in `docs/parity.md` pass from a clean checkout.
2. CPU and CUDA release builds pass with the pinned submodule.
3. The real F16 GGUF loads, generates tokens, and completes official decoder
   integration.
4. `docs/implementation_report.md` contains final commands, metrics, peak GPU
   memory, commits, limitations, and deviations.
5. The worktree is clean and no credential, checkpoint, generated audio, or
   unlicensed fixture is tracked.
6. Artifact SHA-256 is recomputed after staging and matches the manifest.
7. A second model-load smoke test uses the staged artifact rather than the
   converter's working file.

## Renderer checklist (v0.2 and later)

The renderer ships two additional artifacts and adds its own gates:

1. Every release case in `docs/renderer-parity.md` passes at one and fifty Euler
   steps against the pinned Python oracle, from the same stored initial noise.
2. `cuda-precision` passes, so no F32 GEMM silently runs in TF32.
3. Rendered audio is stereo, 48 kHz, finite, non-silent, and exactly
   `frames * 1920` samples per channel.
4. `docs/renderer-implementation-report.md` records the measured matrix, wall
   time, and peak GPU memory.
5. The parity matrix is re-run against the **staged** Flow and VAE artifacts, not
   the converter's working files, so the published bytes are the gated bytes.

## Hugging Face

Create public model repository `ckadirt/LeVo2-GGUF` only after the gates pass.
Upload, each with its `.sha256` and `.manifest.json` sidecar:

- `LeVo2-v2-medium-F16.gguf` (LeLM)
- `LeVo2-v2-flow-F32.gguf` (Flow transformer)
- `LeVo2-v2-vae-F32.gguf` (Oobleck decoder)
- Model card with provenance, build/use commands, limitations, and the exact
  research-only restriction
- Required license and third-party notices

The renderer artifacts are F32 because that is the only precision with a passing
gate. No F16 renderer artifact is published until it has one of its own.

Verify the remotely downloaded artifact hash before announcing it.

## GitHub

Development milestones are pushed to the private `origin/main`. At release:

1. Run the secret and license audit one final time.
2. Make `ckadirt/LeVo2.cpp` public.
3. Push the annotated milestone tag (`v0.1.0` for the LeLM token generator,
   `v0.2.0` for the native renderer).
4. Create a source release linking the exact Hugging Face revision and checksum.
5. Confirm a public anonymous clone can initialize submodules and follow the
   documented CPU and CUDA build commands.

No quantized model is included. A quantized artifact receives its own parity
report and manifest in a later release.
