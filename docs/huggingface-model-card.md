---
license: other
library_name: ggml
tags:
  - audio-generation
  - music-generation
  - gguf
  - research
  - levo
  - songgeneration
---

# LeVo2 v2-medium GGUF

This repository contains the three GGUF components for LeVo2.cpp v0.2:

| File | Precision | Purpose | SHA-256 |
| --- | --- | --- | --- |
| `LeVo2-v2-medium-F16.gguf` | F16 | Lyrics/style to three LeVo token streams | `b765d0e79f17cf05c0acdc6cef8bfcd072104adfd8357bb0470f5b9ae91d9e64` |
| `LeVo2-v2-flow-F32.gguf` | F32 | Vocal/BGM tokens to 64-channel audio latents | `a8cf50dbecef243501b9b345109b1d2f283b3e22f4e4856715197e4b22129d10` |
| `LeVo2-v2-vae-F32.gguf` | F32 | Audio latents to stereo 48 kHz waveform | `26f9ea955f586ed3d7668fe345a851ba222b8db95b406e3eea3c9565f4a0b515` |

Each GGUF has a standard `.sha256` sidecar and a deterministic
`.gguf.manifest.json` conversion inventory. Verify every downloaded component
before loading it.

## License and permitted use

**Research, academic, and education use only. Commercial and production use
are prohibited.** The converted weights and inference-enabling implementation
remain subject to the upstream Tencent SongGeneration terms. Read `LICENSE`
and `THIRD_PARTY_NOTICES.md`; this card grants no additional rights.

## Native lyrics-to-WAV use

Build the matching source repository with CUDA:

```bash
git clone --recurse-submodules https://github.com/ckadirt/LeVo2.cpp.git
cd LeVo2.cpp
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j
```

Generate tokens and render them without Python:

```bash
./build-cuda/bin/levo-cli \
  --model LeVo2-v2-medium-F16.gguf \
  --lyrics lyrics.txt --prompt "female, pop" --duration 30 \
  --output tokens.npy --backend cuda --seed 1235

./build-cuda/bin/levo-render tokens.npy \
  --flow-model LeVo2-v2-flow-F32.gguf \
  --vae-model LeVo2-v2-vae-F32.gguf \
  --output song.wav --backend cuda --steps 50 --cfg 1.5 --seed 1234
```

The renderer writes stereo 48 kHz IEEE-F32 WAV and crops output to exactly
`token_frames * 1920` samples per channel. The v0.2 baseline Flow/VAE artifacts
remain F32. The post-v0.2 catalog adds explicitly selected Q8_0, Q6_K, Q5_K_M,
and Q4_K_M Flow variants after their individual validation gates; VAE remains
F32 until its separate F16 waveform gate passes. See the source project's
quantization implementation report for the current publication state.

## Provenance

| Input | Pinned revision |
| --- | --- |
| LeVo source | `levo-demo/LeVo@653cbcf4716101834900c75b7d5da43b07e15d5b` |
| LeLM checkpoint | `lglg666/SongGeneration-v2-medium@7d91660ebfa041e29bace194f5631e775796f600` |
| Flow/VAE runtime | `lglg666/SongGeneration-Runtime@cc258cc694a63114c61684cc26d0583b8ad777d0` |
| GGML | `ggml-org/ggml@8846b79e66747bb9f68597420e95114c177315ce` |

The Flow conversion emits 231 runtime tensors / 663,310,785 parameters. The
decoder-only VAE conversion folds weight normalization and emits 145 tensors /
84,395,776 parameters. Training-only encoders, discriminators, and maintenance
state are classified and omitted in their manifests.

## Validation

LeVo2.cpp v0.2 passes strict loaders, CPU/CUDA tests, stage-level Flow/VAE
parity, and an eight-case token-to-WAV matrix against the pinned official
Python oracle. Cases cover 50, 252, 750, and 1,250 token frames at one and fifty
Euler steps, including the second 750-frame hop and 250-frame overlap.

The worst measured release values are:

| Boundary | Measured worst | Frozen maximum |
| --- | ---: | ---: |
| Latent maximum error | `2.013e-3` | `2e-2` |
| Latent relative RMS | `3.283e-5` | `5e-3` |
| Audio maximum error | `2.718e-3` | `3e-3` |
| Audio relative RMS | `7.676e-5` | `1e-3` |

A public-path 30-second smoke generated `[3,750]` tokens and a finite,
non-silent stereo WAV with exactly 1,440,000 samples per channel. Full commands,
fixture hashes, timings, GPU memory, deviations, and limitations are recorded
in the source repository's `docs/renderer-implementation-report.md` and
`docs/renderer-release-matrix.json`.

## Limitations

- Flow and VAE are F32 only; there are no renderer quantizations.
- Input is lyrics plus an optional style description; audio-prompt encoding and
  source separation are not implemented natively.
- PyTorch/CUDA and GGML can make different near-tied token choices for prompts
  outside the frozen LeLM parity fixture.
- Generated music can contain artifacts or inappropriate content. Respect
  copyright, privacy, and applicable law, and do not use it for high-stakes
  decisions.
