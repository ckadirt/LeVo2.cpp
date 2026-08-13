# Quantization implementation report

## Scope and operating method

This is the live implementation record for the post-v0.2 GGUF quantization
milestone. It supplements the immutable historical sections in
`implementation_report.md`; that report links here rather than being rewritten
as implementation proceeds.

Every accepted slice follows the same order:

1. document the intended contract and any new evidence;
2. implement the smallest independently testable change;
3. build and run the relevant tests on CPU (and CUDA where graph execution is
   affected);
4. commit and push the source to `origin/main`;
5. publish a Hugging Face artifact only after its strict loader and declared
   validation gates have passed.

Artifact publication is intentionally later than source publication. A model
file is not a progress snapshot: it must be self-describing, checksummed, and
usable by the matching released runtime.

## Approved artifact catalog

| Component | Baseline | New profiles | Publication gate |
| --- | --- | --- | --- |
| LeLM | F16 | Q8_0, Q6_K, Q5_K_M, Q4_K_M | strict load, token smoke, frozen quality matrix |
| Flow | F32 | F16 after separate gate; Q8_0, Q6_K, Q5_K_M, Q4_K_M | strict load, padded-matrix graph smoke, render matrix |
| VAE | F32 | F16 after waveform gate only | decoder/waveform gate; no low-bit VAE |

Cantor keeps its existing explicit `lm`, `dit`, and `vae` paths in this
milestone. It will not silently choose a quality tier. Measured combinations
will inform a later, separately approved quality/balanced/fast policy.

## Frozen policy revision 1

All produced quantized artifacts carry:

```text
levo2.quantization.profile
levo2.quantization.policy_revision = "1"
levo2.quantization.source_artifact_sha256
```

The source digest is the complete input GGUF SHA-256, not the upstream
checkpoint digest. This makes the artifact lineage verifiable after a process
restart and prevents an accidental resume across different quantizations.

### LeLM

- Every rank-0/1 tensor remains F32.
- Q8_0 and Q6_K quantize every rank-2 matrix to the profile type.
- Q5_K_M and Q4_K_M promote these matrices to Q6_K: token embeddings, output
  heads, all `cond.*.weight`, bridge weights, attention V/output projections,
  and FFN down projections. Other rank-2 matrices use Q5_K or Q4_K.

LeLM's 1536- and 8960-wide input axes are divisible by the GGML K block size,
so no layout padding is required.

### Flow

- Controls, embeddings, normalizers, modulation tables, biases, final output,
  and every non-transformer tensor remain F32.
- Only `flow.block.N.attn.{qkv,out}.weight` and
  `flow.block.N.ffn.{in,out}.weight` are quantized.
- Q5_K_M/Q4_K_M promote `attn.out` and `ffn.out` to Q6_K; `qkv` and `ffn.in`
  use the profile base type.

Flow's logical hidden/intermediate widths (2200/4400) are not divisible by
GGML's Q8 or K block widths. Quantized Flow matrices therefore store padded
input axes: Q8_0 uses 2200 -> 2208 and 4400 -> 4416 (32-value blocks); K
profiles use 2200 -> 2304 and 4400 -> 4608 (256-value blocks). The runtime
zero-pads only the corresponding activation input immediately before
`mul_mat`; logical output dimensions and Flow metadata remain 2200/4400. A
loader rejects a quantized Flow tensor without exactly this layout.

## Resumability and artifact identity

The resume checkpoint format will be bumped and old blobs explicitly refused.
LeLM, Flow, and VAE content SHA-256 values will become part of the relevant
stage stamps. Source provenance alone is insufficient: two quantized files can
share that provenance but produce different audio. The runtime will compute
the digest from the GGUF bytes rather than trusting a sidecar.

## Milestone log

### 2026-08-13 — contract and shared policy scaffold

- Added the versioned native policy definition in `src/levo-quantization.h`.
- Confirmed the Flow 2200/4400 input-axis divisibility issue from GGML's
  `ggml_row_size` contract. Q8_0 uses a 32-value block and K profiles use a
  256-value block; the physical padding is profile-specific. This is handled
  by artifact storage plus runtime padding, not by rounding Flow's logical
  architecture constants.
- No quantized artifact has been written or published yet.
- Pending validation: build the scaffold after loader/requantizer integration,
  then commit/push this initial documented foundation.

### 2026-08-13 — native writer and runtime loading

- Added `levo-quantize`, a C++17 streaming GGUF requantizer. It accepts only
  F32/F16 LeLM and Flow input, uses `ggml_quantize_chunk`, preserves every
  metadata field, tags the output with policy revision 1 and the complete
  source-artifact SHA-256, and writes deterministic checksum/manifest
  sidecars. It refuses VAE, in-place output, existing artifact names, and
  already-quantized input.
- Strict LeLM loading now verifies the profile's complete tensor-type routing;
  strict Flow loading additionally verifies `MIXED` dtype and exact
  profile-specific physical input widths. Both loaders calculate the complete
  GGUF SHA-256 for provenance rather than trusting a sidecar.
- Added a direct CPU GGML Q8_0 padded-matmul test. It quantizes a 32-column
  storage row, zero-pads a six-value logical activation, and verifies the
  logical output remains within Q8 tolerance.
- Built the portable runtime and passed the targeted `quantization`,
  `quantizer-cli-help`, `model`, `lm`, `kv`, and `flow-model` tests.
- Real Flow Q8_0 smoke artifact (not published):
  `LeVo2-v2-flow-Q8_0.gguf`, 835,813,888 bytes,
  SHA-256 `9e27b8d060edd8b57b8c3033b14260de5cfa08cde8e3c58532a3ce439bfced3a`.
  Its strict Flow loader test and checksum sidecar passed; inventory is 167
  F32 tensors plus 64 Q8_0 transformer matrices, with the declared
  `hidden=2200->2208;intermediate=4400->4416` layout.
- The same local artifact rendered the frozen 2-second token fixture on CPU at
  one Euler step through native quantized Flow plus F32 VAE. The result is a
  finite stereo IEEE-F32 WAV at 48 kHz with exactly 2.0 seconds of audio;
  SHA-256 `b0475936c0498564800448d3d91acc6993f90b34f31096e7919a3ecb40d3f322`.
- Full portable CTest after the change: 28/28 passed.

### 2026-08-13 — first real LeLM/CUDA pass and checkpoint identity

- The first real LeLM Q8_0 CUDA smoke initially exposed a missed boundary:
  the conditioner copied whole F32/F16 embeddings directly from the backend.
  That cannot dequantize a native GGML embedding. The conditioner now performs
  a small `ggml_get_rows` graph for quantized weights and retains the exact
  direct F32/F16 path. It gathers only requested rows, so it also eliminates
  unnecessary host copies of the 151k-row conditioner tables.
- Local, unpublished LeLM Q8_0 artifact:
  `LeVo2-v2-medium-Q8_0.gguf`, 2,909,949,120 bytes,
  SHA-256 `706cd0b3fcb84c7d4522745331e679114a288deab5dfcad37d546ebb6d002291`.
  It contains 84 F32 vectors and 296 Q8_0 matrices. Its sidecar checksum
  passed.
- On RTX 4090/CUDA, the LeLM Q8_0 artifact completed a 2-second greedy lyrics
  generation as a valid `[3,50]` token artifact. The token payload SHA-256 is
  `cdf7c375e7f3cf236dc7fcefa2f1e5cee57b62ce9406a045c2f08b655f57bb7c`;
  the artifact manifest records the loaded GGUF SHA-256.
- The local Flow Q8_0 artifact also rendered the frozen 2-second fixture on
  RTX 4090/CUDA through F32 VAE, yielding finite 48 kHz stereo PCM-F32 audio
  with SHA-256 `44dbf4c4c02074dbc249462b1f72a5eacb80b6e74451061408c762d3c35580b7`.
- Cantor checkpoint magics advance to `LEVOLM02`, `LEVOFL02`, and `LEVOLT02`.
  Each new LeLM/Flow stamp includes the complete GGUF digest; Flow additionally
  stamps the selected VAE artifact so DECODE refuses a different waveform
  decoder. `01` blobs are rejected explicitly because their source provenance
  cannot distinguish quantization artifacts.
- Pending before publication: complete CPU/CUDA render smoke, then the frozen
  render matrix and actual quality measurements. The artifact remains local
  until those gates pass.

## Deviations

None so far.
