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

This repository contains the v0.2 baseline GGUF components and the native
post-v0.2 low-bit catalog. Select the LeLM and Flow files independently; VAE
stays F32 for every combination.

| File | Precision | Purpose | SHA-256 |
| --- | --- | --- | --- |
| `LeVo2-v2-medium-F16.gguf` | F16 | Lyrics/style to three LeVo token streams | `b765d0e79f17cf05c0acdc6cef8bfcd072104adfd8357bb0470f5b9ae91d9e64` |
| `LeVo2-v2-medium-Q8_0.gguf` | Q8_0 | LeLM, largest low-bit tier | `706cd0b3fcb84c7d4522745331e679114a288deab5dfcad37d546ebb6d002291` |
| `LeVo2-v2-medium-Q6_K.gguf` | Q6_K | LeLM, balanced low-bit tier | `06304f3f49ee6d1ed9265dde0651af199017cb6ca743d5f5011f6900a63a3c90` |
| `LeVo2-v2-medium-Q5_K_M.gguf` | Q5_K_M + Q6_K | LeLM, compact mixed tier | `fc8616097d264d8b5437ab01f453d9ac1bfaba9ef834f3a47d728f0fd20724de` |
| `LeVo2-v2-medium-Q4_K_M.gguf` | Q4_K_M + Q6_K | LeLM, smallest mixed tier | `9412bb0ef5373fd0b9085fd24e4b5ffa0d341efece3829067563816d44d4aeca` |
| `LeVo2-v2-flow-F32.gguf` | F32 | Vocal/BGM tokens to 64-channel audio latents | `a8cf50dbecef243501b9b345109b1d2f283b3e22f4e4856715197e4b22129d10` |
| `LeVo2-v2-flow-Q8_0.gguf` | Q8_0 + F32 controls | Flow, largest low-bit tier | `9e27b8d060edd8b57b8c3033b14260de5cfa08cde8e3c58532a3ce439bfced3a` |
| `LeVo2-v2-flow-Q6_K.gguf` | Q6_K + F32 controls | Flow, balanced low-bit tier | `a3579de6915c5ea060072b83af4e158e729834a8a13f62f1b370ef82e7317c00` |
| `LeVo2-v2-flow-Q5_K_M.gguf` | Q5_K_M/Q6_K + F32 controls | Flow, compact mixed tier | `0c36bdda1148068aeca4e32abedeeaa9445b672e20f841d0fef84d250b447dcb` |
| `LeVo2-v2-flow-Q4_K_M.gguf` | Q4_K_M/Q6_K + F32 controls | Flow, smallest mixed tier | `08cc21590702f1c2e9bc62b164d4ae82b9b95d1b1985e13329c2f9d3fae89edc` |
| `LeVo2-v2-vae-F32.gguf` | F32 | Audio latents to stereo 48 kHz waveform | `26f9ea955f586ed3d7668fe345a851ba222b8db95b406e3eea3c9565f4a0b515` |
| `LeVo2-v2-vae-F16.gguf` | F16 storage, F32 graph | Smaller decoder-weight artifact | `23e5b11558ae332fbe216d9a06775884469fcbf32236c26ab52defa18c5c8398` |

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

Generate tokens and render them without Python. This example uses the balanced
Q6_K pairing; replace either selected file with a different listed tier when
you want a different memory/quality trade-off:

```bash
./build-cuda/bin/levo-cli \
  --model LeVo2-v2-medium-Q6_K.gguf \
  --lyrics lyrics.txt --prompt "female, pop" --duration 30 \
  --output tokens.npy --backend cuda --seed 1235

./build-cuda/bin/levo-render tokens.npy \
  --flow-model LeVo2-v2-flow-Q6_K.gguf \
  --vae-model LeVo2-v2-vae-F16.gguf \
  --output song.wav --backend cuda --steps 50 --cfg 1.5 --seed 1234
```

The renderer writes stereo 48 kHz IEEE-F32 WAV and crops output to exactly
`token_frames * 1920` samples per channel. The native loader verifies every
quantized tensor type, policy revision, and Flow physical padded layout before
running it. The F16 VAE has its own source-artifact lineage and is loaded only
when its precision tags are present; decoder operators promote its stored F16
weights to F32 for execution.

## Quantization policy and selection

The Q5_K_M and Q4_K_M files deliberately retain sensitive matrices at Q6_K:
LeLM retains embeddings, output heads, conditioner tables, bridge weights,
attention V/output, and FFN-down projections; Flow retains attention/FFN output
projections. Flow controls, embeddings, normalizers, modulation tables, biases,
and final output remain F32. This is a fixed, strict policy revision, not
runtime dequantization or a name-only precision hint.

Use the F16 LeLM and F32 Flow files when fidelity is the priority. Q6_K is the
best deterministic Flow match in the frozen comparison below. Q8_0, Q5_K_M,
and Q4_K_M are available as explicit capacity/quality trade-offs; Q4_K_M is
the smallest and has the largest measured deviation. The source repository's
`docs/quantization-implementation-report.md` and
`docs/quantization-validation-matrix.json` contain the complete policy and
reproducible identities.

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

Each low-bit file passed its exact sidecar checksum, strict native loader, and
real CUDA smoke. Every low-bit Flow file also rendered a finite two-second,
48 kHz stereo WAV through the native quantized graph on CUDA; Q6_K/Q5_K_M/
Q4_K_M additionally passed CPU renders (Q8_0 had already done so). Against
the same F32 Flow output with fixed tokens, noise, one Euler step, and seed,
the measured waveform SNR/correlation was:

| Flow tier | SNR vs F32 | Correlation vs F32 |
| --- | ---: | ---: |
| Q8_0 | 13.505 dB | 0.978 |
| Q6_K | 17.754 dB | 0.992 |
| Q5_K_M | 10.842 dB | 0.958 |
| Q4_K_M | 8.098 dB | 0.919 |

The LeLM tiers all completed the same CUDA greedy `[3,50]` token request.
Quantization changes near-tied token choices, so neither token IDs nor audio
are claimed bit-identical to F16/F32. Verify file bytes with the supplied
sidecars before loading.

The F16 VAE passed its independent 1,000-frame full-window storage comparison
against a native F32 VAE reference on CPU and CUDA. CUDA measured `2.65128e-3`
maximum error, `6.56119e-5` RMSE, and `0.999999914` correlation; CPU measured
`2.65826e-3`, `6.56703e-5`, and `0.999999914`. The decoder graph remains F32,
so this artifact reduces stored-weight size rather than claiming an F16
activation/accumulation path.

## Limitations

- The VAE F16 artifact is a storage optimization: its decoder graph is F32.
  Low-bit LeLM/Flow files are approximate inference tiers, not F16/F32 parity
  claims; quality depends on the lyric, prompt, seed, and sampler settings.
- Input is lyrics plus an optional style description; audio-prompt encoding and
  source separation are not implemented natively.
- PyTorch/CUDA and GGML can make different near-tied token choices for prompts
  outside the frozen LeLM parity fixture.
- Generated music can contain artifacts or inappropriate content. Respect
  copyright, privacy, and applicable law, and do not use it for high-stakes
  decisions.
