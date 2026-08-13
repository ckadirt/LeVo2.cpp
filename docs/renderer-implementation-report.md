# Native renderer implementation report

## Status

The native renderer is complete and release-gated. The eight-case full-window
matrix and the public lyrics-to-WAV smoke pass with measured evidence below.
Token-to-WAV rendering runs entirely in C++/GGML: dual RVQ conditioning, the
16-block Flow velocity transformer,
classifier-free guidance, uniform Euler integration, 1000/750/250 window
continuation, the five-stage Oobleck decoder, crossfade assembly, and the F32
WAV writer. The earlier WIP assertion that all release cases passed was not
backed by committed matrix reports and is superseded by this trace.

The official Python decoder remains in the repository as the reference oracle,
not as the production path. F16 renderer execution is still deferred until it
passes its own precision gate; F16 GGUFs are rejected rather than downcast.

## Baseline

- Starting source release: `v0.1.0` at commit
  `c750fcbad144a67421fe92cba7431c364f8451d6`.
- Test machine: NVIDIA GeForce RTX 4090, CUDA 12.4.
- Pinned Flow checkpoint: 4,808,167,708 bytes, SHA-256
  `430b7c1c245722fbe3893cd621b3d4a90076404596e9fb1ce987a4a0f2a4fc6f`.
- Pinned VAE checkpoint: 674,920,616 bytes, SHA-256
  `10ccb6c83613781ad32e998a90597ba7eb9292911a224598da1fd53728eb4cd3`.
- Pinned VAE config SHA-256:
  `5cd2859efe00bc2b0f6f9bdac738ad11822a36473d6d810427b60efd057c538b`.

The upstream checkpoint includes large training/encoding subgraphs that the
token-to-WAV path never calls. The converter will classify every source tensor
before runtime code is accepted, following the same strict-inventory discipline
as the LeLM converter.

## Source audit

The initial read-only source audit fixed the native boundary before converter
work:

- Flow: 993 source tensors; approximately 2.471 GiB is reachable from token
  rendering after training/audio-encoder omissions.
- VAE: 365 checkpoint tensors, of which 182 belong to the decoder; folding 37
  weight-normalized convolutions produces 145 runtime tensors.
- VAE convolutions are zero padded. CUDA transpose-convolution padding will be
  represented as a zero-padding operation followed by a symmetric crop.
- Flow F32 correctness and official CUDA FP16-autocast compatibility are
  explicitly distinct parity modes.

## Landed checkpoints

| Commit | Gate | Result |
| --- | --- | --- |
| `8bfedea` | Renderer plan | Architecture, GGUF, parity, and execution plan |
| `7e04f44` | Source contract | Exact Flow/VAE topology and inference inventory |
| `f70f2af` | Flow converter | 993 source tensors classified; 231 emitted |
| `78436bb` | VAE converter | 365 source tensors classified; 145 emitted |
| `0b0b237` | Python oracles | Deterministic Flow/VAE intermediate capture |
| `8900054` | GGML primitives | F32 Conv1d, padded ConvTranspose1d, SnakeBeta |
| `d6b4b3a` | WAV output | Strict interleaved IEEE-F32 RIFF/WAVE writer |
| `37ba221` | Renderer schedule | Repeat/crop, windows, Euler, CFG, crossfade |
| `a27bca5` | VAE loader | Strict 145-tensor F32/F16 GGUF loader |
| `90f3df1` | VAE decoder | Complete native five-stage Oobleck graph |
| `6cb1308` | Flow loader | Strict 231-tensor F32/F16 GGUF loader |
| `573a539` | VAE parity | Stage-level CPU/CUDA official-oracle gate |
| `a08c4a5` | Flow conditioning | RVQ projection, masks, positions, and latent normalization |
| `251017c` | Flow estimator | Complete F32 16-block velocity transformer |
| `a1de25f` | Flow parity | Two-frame official operator/velocity gate |
| `e66026d` | Flow renderer | CFG, Euler integration, and continuation windows |
| `43016ad` | Public renderer | Native token-to-WAV API and CLI |
| `b49acd2` | Renderer oracle | Deterministic full-window Python exporter |
| `b5a571a` | End-to-end gate | Production-path native parity tool |
| `5c865ff` | True F32 gate | Disabled implicit TF32 on both oracle and native paths |
| `aa46deb` | Release harness | v0.2 version and structured timing/memory reports |
| `5e1f47c` | Release fixtures | Reproducible four-case input contract |
| `9538dbd` | Release matrix | Eight passing cases and native 30-second WAV smoke |

The strict Flow converter produced `LeVo2-v2-flow-F32.gguf` with 231 tensors,
663,310,785 parameters, 2,653,259,456 bytes, and SHA-256
`a8cf50dbecef243501b9b345109b1d2f283b3e22f4e4856715197e4b22129d10`.

The strict VAE converter produced `LeVo2-v2-vae-F32.gguf` with 145 tensors,
84,395,776 parameters, 337,596,448 bytes, and SHA-256
`26f9ea955f586ed3d7668fe345a851ba222b8db95b406e3eea3c9565f4a0b515`.
These hashes identify the gated v0.2 F32 release artifacts.

Validated commands include:

```bash
LEVO_FLOW_CHECKPOINT=/workspace/models/SongGeneration-Runtime/ckpt/model_septoken/model_2.safetensors \
  /venv/main/bin/python -m pytest -q tests/python/test_flow_converter.py
/venv/main/bin/python -m pytest -q tests/python/test_vae_converter.py
LEVO_RUN_RENDERER_ORACLE_TEST=1 \
  /venv/main/bin/python -m pytest -q tests/python/test_renderer_oracles.py
ctest --test-dir build-cpu -R '^audio-ops$' --output-on-failure
ctest --test-dir build-cuda -R 'audio-ops' --output-on-failure
```

At this checkpoint the focused results are Flow converter 4/4, VAE converter
4/4, renderer oracles 3/3 (including the official T=1 VAE), and native audio
operators 1/1 CPU plus 2/2 in the CUDA build.

The complete native T=1 Oobleck decoder now passes the same fixed official
PyTorch oracle on both tested backends:

| Backend | Maximum error | Relative RMS | Cosine |
| --- | ---: | ---: | ---: |
| CPU | `2.02447e-4` | `3.89340e-4` | `1.0` |
| RTX 4090 CUDA | `1.81034e-4` | `4.04892e-4` | `1.0` |

This is within the frozen F32 waveform gates (`3e-3` maximum error and `1e-3`
relative RMS) without threshold changes. The local strict Flow and VAE loaders
also loaded their complete 231- and 145-tensor F32 artifacts successfully.

The diagnostic parity runner additionally checks all five VAE decoder stages.
The CPU and CUDA T=1/T=2 cases pass the frozen stage gates of maximum error
`<= 2e-3`, RMSE `<= 2.5e-4`, and cosine `>= 0.9999995`; the worst observed
stage maximum error was `1.45030e-3` and the worst waveform maximum error was
`3.22908e-4`.

The native Flow conditioning boundary was compared with the official two-frame
F32 input oracle. Codebook lookups, mask embeddings, position embeddings, and
the null condition are bit-exact. The projected vocal and BGM conditions have
maximum errors `8.94070e-8` and `5.96046e-8`, respectively, far inside the
frozen `5e-5` conditioning gate.

The complete native Flow velocity graph now passes the official CUDA-F32
two-frame oracle at `t=0.5`. Timestep embeddings and modulation differ by at
most `4.76837e-7`; block 0 by `8.70228e-6`; the complete 2200-wide output by
`9.17912e-6`; and CFG-combined 64-wide velocity by `1.74046e-5`. Full-output
and velocity cosine similarity exceed `0.99999999999`. The two semantic fixes
identified by the staged capture were interleaved Q/K/V slice offsets and the
AdaLN attention gate; both are now independently covered by the parity tool.

The Flow renderer layer implements the exact uniform Euler schedule,
classifier-free branches, sigma-min in-context interpolation and hard restore,
1000-frame windows, 750-frame hops, and 250-frame continuation context. Its
asset-free solver/window tests and the full one-step/fifty-step release matrix
pass.

## v0.2 release finalization trace

Finalization resumed from clean `origin/main` commit `c45e78b` on 2026-08-13.
The workspace is not volume-backed, so every accepted gate is committed and
pushed before the next long-running step.

- Restored GGML `8846b79e66747bb9f68597420e95114c177315ce` and LeVo
  `653cbcf4716101834900c75b7d5da43b07e15d5b` from their public repositories.
- Baseline hardware is an RTX 4090 (24 GiB), driver 595.84, and CUDA toolkit
  13.2. The version differs from the original CUDA 12.4 implementation host and
  is recorded as release-matrix coverage rather than silently substituted.
- A clean CPU release build with parity tools passed 21/21 CTests.
- The CUDA 13.2 release build passed 25/25 CTests, including the RTX 4090
  backend, F32/TF32 precision guard, CUDA audio operators, and CUDA VAE graph.
- The asset-free Python suite passed 16 tests with 15 credential/heavy tests
  skipped by their explicit opt-in gates.
- The downloaded Flow, VAE, config, and public LeLM inputs match every pinned
  byte count and SHA-256 recorded in this report and the v0.1 report.
- The public Hugging Face repository still contained only the v0.1 LeLM
  artifact when finalization resumed. The README statement that renderer GGUFs
  were already published was premature; publication remains a release gate.
- The release runner now emits structured per-case timing, metrics, artifact
  hashes, and polled peak GPU memory so the final status can be evidence-backed.
- Fresh F32 conversions reproduced both earlier artifacts byte-for-byte. Flow
  is 2,653,259,456 bytes with SHA-256
  `a8cf50dbecef243501b9b345109b1d2f283b3e22f4e4856715197e4b22129d10`;
  VAE is 337,596,448 bytes with SHA-256
  `26f9ea955f586ed3d7668fe345a851ba222b8db95b406e3eea3c9565f4a0b515`.
  Their checksum sidecars, manifests, parameter/tensor counts, and strict
  CPU/CUDA loaders pass. The focused converter suite is 8/8.
- The renderer fixture contract is now tracked while raw token arrays remain
  ignored. The exact prompt has no trailing newline. The 50-frame greedy
  tensor reproduces its historical SHA-256 exactly. On the CUDA 13.2 host the
  freshly generated 252-frame greedy tensor differs from the v0.1 hash after
  the short boundary; it is frozen under its measured hash for renderer input
  coverage rather than represented as renewed LeLM cross-version parity.
- Sampled seed 1234 reached model EOS at 683 frames. Following the release
  fixture policy, the first subsequent seed, 1235, completed all 750 frames and
  is the frozen sampled case. The 1,250-frame overlap case repeats this tensor
  and crops it to the documented length.
- The native two-frame Flow estimator gate passes 1/1. The first VAE parity
  invocation failed before model execution because the recreated Python 3.12
  environment lacked the legacy `pkg_resources` import used by `clip`.
  Pinning `setuptools<81` restored that upstream compatibility; the unchanged
  CUDA F32 VAE stage gate then passed 2/2. No numerical threshold or model code
  changed.

### Final release matrix

All four required inputs pass at both one and fifty Euler steps. Values below
are the worst boundary value for each run across normalized/denormalized
latents, decoded windows, and assembled channel audio. The machine-readable
record is `docs/renderer-release-matrix.json`.

| Frames | Windows | Steps | Latent max | Latent rel. RMS | Audio max | Audio rel. RMS | Oracle / native seconds | Oracle / native peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 1 | 1 | `4.816e-5` | `4.572e-6` | `1.770e-4` | `3.050e-5` | 16.399 / 3.780 | 16,742 / 10,140 |
| 50 | 1 | 50 | `1.105e-3` | `2.473e-5` | `3.105e-4` | `4.588e-5` | 21.715 / 10.053 | 16,742 / 10,140 |
| 252 | 1 | 1 | `2.432e-5` | `3.283e-6` | `2.699e-6` | `2.569e-5` | 16.390 / 3.659 | 16,742 / 10,140 |
| 252 | 1 | 50 | `8.414e-4` | `6.397e-6` | `4.916e-5` | `2.084e-5` | 21.700 / 10.066 | 16,742 / 10,140 |
| 750 | 1 | 1 | `5.054e-5` | `4.687e-6` | `4.334e-4` | `2.329e-5` | 16.447 / 3.661 | 16,742 / 10,140 |
| 750 | 1 | 50 | `6.176e-4` | `1.297e-5` | `4.830e-4` | `2.690e-5` | 21.698 / 10.056 | 16,742 / 10,140 |
| 1,250 | 2 | 1 | `2.040e-4` | `7.667e-6` | `4.336e-4` | `3.188e-5` | 16.952 / 6.542 | 17,682 / 10,140 |
| 1,250 | 2 | 50 | `2.013e-3` | `3.282e-5` | `2.718e-3` | `7.676e-5` | 27.632 / 19.476 | 17,682 / 10,140 |

The frozen latent limits are `2e-2` maximum and `5e-3` relative RMS; waveform
limits are `3e-3` maximum and `1e-3` relative RMS. No threshold changed.

The public two-command production smoke regenerated the sampled `[3,750]`
artifact in 47.401 s with 10,514 MiB peak GPU memory, then rendered it with the
native seed 1234 in 10.177 s with 10,140 MiB peak GPU memory. The resulting
IEEE-F32 WAV is stereo 48 kHz with 1,440,000 samples per channel, finite,
non-silent (RMS `0.390319`, absolute peak `1.852677`), 11,520,044 bytes, and
SHA-256 `b6b7a3d60ce01b3a2644c4ef9adf2b0306021ec2d22b0cdf1719b25a948a83ed`.
