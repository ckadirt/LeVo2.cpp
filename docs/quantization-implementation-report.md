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

## Deviations

None so far.
