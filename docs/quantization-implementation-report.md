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

The resume checkpoint format is bumped and old blobs are explicitly refused.
LeLM, Flow, and VAE content SHA-256 values are part of the relevant stage
stamps. Source provenance alone is insufficient: two quantized files can share
that provenance but produce different audio. The runtime computes the digest
from the GGUF bytes rather than trusting a sidecar. DECODE verifies both the
selected Flow and VAE digests against its completed `LEVOLT02` boundary before
it allocates the decoder, so a component substitution after a pause fails
loudly rather than silently changing a waveform.

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

### 2026-08-13 — LeLM K-profile smoke expansion (local only)

- Produced native, self-describing Q6_K and Q5_K_M LeLM artifacts from the
  frozen F16 source. Both sidecar checksums pass and both retain policy
  revision 1 plus the source-artifact digest.
- Local Q6_K: `LeVo2-v2-medium-Q6_K.gguf`, 2,249,071,968 bytes,
  SHA-256 `06304f3f49ee6d1ed9265dde0651af199017cb6ca743d5f5011f6900a63a3c90`.
  CUDA greedy generation completed as a valid `[3,50]` token artifact;
  token payload SHA-256
  `37c15a3a0b2ea4d65cc8447d31ddb6f9e391ec4a2d50442b6ba598ae6a761fa3`.
- Local Q5_K_M: `LeVo2-v2-medium-Q5_K_M.gguf`, 2,077,777,248 bytes,
  SHA-256 `fc8616097d264d8b5437ab01f453d9ac1bfaba9ef834f3a47d728f0fd20724de`.
  CUDA greedy generation completed as a valid `[3,50]` token artifact;
  token payload SHA-256
  `d240ddc453d50a80704398110530de7050c7fe00e57f732f2e337f4379b10eff`.
- At this point Q4_K_M was the remaining LeLM conversion. No K-profile is
  published yet: the full LeLM and Flow profile matrix, frozen comparison, and
  release-side manifest review remain publication gates.

### 2026-08-13 — LeLM Q4_K_M smoke completed (local only)

- Local Q4_K_M: `LeVo2-v2-medium-Q4_K_M.gguf`, 1,916,558,688 bytes,
  SHA-256 `9412bb0ef5373fd0b9085fd24e4b5ffa0d341efece3829067563816d44d4aeca`.
  The checksum sidecar passed. Its inventory is 84 F32 vectors, 160 Q4_K
  matrices, and 136 Q6_K promoted matrices, exactly as policy revision 1
  requires.
- The strict CUDA loader completed the same 2-second greedy request and wrote
  a valid `[3,50]` token artifact; payload SHA-256
  `e5438d77fad3f72d0432f1099a1491205084447a1b63a6cc1ad9df38e0a693b6`.
- The complete local LeLM profile set (Q8_0/Q6_K/Q5_K_M/Q4_K_M) has now passed
  its checksum and CUDA token-smoke gate. Frozen token and audio comparisons
  remain pending before Hugging Face publication.

### 2026-08-13 — complete profile smoke and deterministic comparison

- All Flow profiles now pass strict metadata/type/layout loading and produce a
  finite 2-second, 48 kHz stereo IEEE-F32 WAV from the frozen `[3,50]` token
  fixture on CUDA. Q8_0 had previously passed the equivalent CPU smoke; the
  Q6_K, Q5_K_M, and Q4_K_M profiles also passed it. This exercises the actual
  Q8/K padded matrix activation path, not a dequantized fallback.
- Flow artifacts and CUDA output hashes: Q6_K is 707,404,288 bytes,
  `a3579de6915c5ea060072b83af4e158e729834a8a13f62f1b370ef82e7317c00`,
  WAV `67628465847efb24a04a289361a359ab6cafdbe36f02f95c75c37423db021cc0`;
  Q5_K_M is 653,548,032 bytes,
  `0c36bdda1148068aeca4e32abedeeaa9445b672e20f841d0fef84d250b447dcb`,
  WAV `4239189ae3d66f05112a37f174d02d779641bd9ec90efe7d58ff89ef734549f5`;
  Q4_K_M is 602,860,032 bytes,
  `08cc21590702f1c2e9bc62b164d4ae82b9b95d1b1985e13329c2f9d3fae89edc`,
  WAV `df0c108b24aaf0804a9ee1c6f5c581337e145d4137e08b41429db9147c90ab07`.
- Against an F32 CUDA render of exactly the same fixture/noise/one-Euler-step
  request, Flow Q8_0/Q6_K/Q5_K_M/Q4_K_M respectively measured
  `13.505/17.754/10.842/8.098` dB SNR and
  `0.978/0.992/0.958/0.919` waveform correlation. The full reproducible
  fixture, artifact identities, output hashes, and LeLM greedy-token match
  counts are frozen in
  [`quantization-validation-matrix.json`](quantization-validation-matrix.json).
  Q6_K is the closest profile on this cell; all profiles are published as
  explicit user-selected trade-offs, not as identical-quality replacements.
- The F16 LeLM reference and every low-bit LeLM profile generated a valid
  `[3,50]` token tensor on CUDA. Greedy IDs are not bit-identical under
  quantization (Q8_0 25/150, Q6_K 97/150, Q5_K_M 68/150, Q4_K_M 42/150
  matching IDs), so the artifacts are accurately described as approximate
  inference tiers rather than a parity claim.
- Tightened the completed-boundary check: DECODE now verifies the selected Flow
  GGUF digest as well as the VAE digest before decoding a `LEVOLT02` payload.
  CUDA build plus `ctest --label-exclude cuda` passed 28/28 after this change.
- All local artifact files, sidecars, and validation evidence are now ready for
  the first Hugging Face publication transaction. The model-card update and
  anonymous remote checksum verification are the remaining gates.

## Deviations

None so far.
