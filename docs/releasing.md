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

## Hugging Face

Create public model repository `ckadirt/LeVo2-GGUF` only after the gates pass.
Upload:

- `LeVo2-v2-medium-F16.gguf`
- `LeVo2-v2-medium-F16.gguf.sha256`
- `manifest.json`
- Model card with provenance, build/use commands, limitations, and the exact
  research-only restriction
- Required license and third-party notices

Verify the remotely downloaded artifact hash before announcing it.

## GitHub

Development milestones are pushed to the private `origin/main`. At release:

1. Run the secret and license audit one final time.
2. Make `ckadirt/LeVo2.cpp` public.
3. Push annotated tag `v0.1.0`.
4. Create a source release linking the exact Hugging Face revision and checksum.
5. Confirm a public anonymous clone can initialize submodules and follow the
   documented CPU and CUDA build commands.

No quantized model is included in v0.1. A quantized artifact receives its own
parity report and manifest in a later release.
