# LeVo 2 v2-large implementation plan

## Status

Approved for implementation on 2026-08-13. This is a v2-large LeLM extension
only. The v2 renderer remains a shared codec-to-waveform component, not a
second model family.

## Pinned source contract

| Property | v2-large value |
| --- | --- |
| Model repository / revision | `lglg666/SongGeneration-v2-large@115805364ad74479fb3764fe65970c92faeb1a5a` |
| `model.pt` bytes / SHA-256 | `12,899,965,446` / `dc763aa9a76a22a87597c2faf9a51c24d13349ac754699b37e9068b483639def` |
| `config.yaml` bytes / SHA-256 | `3,352` / `14a991bd7342b9dde348e6324afd44b5c6ecb1db8d0ed4d2dbe666b220b04c59` |
| LeLM width / FFN width | `2048` / `11008` |
| Main / detail blocks | `36` / `12` |
| Attention / KV heads | `16` / `16` |
| Context / main & detail RoPE | `10000` / `500000` & `500000` |
| Streams / codebook / delay | `3` / `16384` / `[0,250,250]` |
| Conditioning prefix | `600 + 252 + 100 = 952` positions |

The large configuration retains v2-medium's frame rate (25 Hz), sample rate
(48 kHz), token layout, conditioner topology, separate-token Flow checkpoint,
and Stable Audio 1920 VAE. Therefore the existing Flow GGUFs and F16/F32 VAE
GGUFs are the shared renderer artifacts for both LeLM scales.

## Deliverables

1. Add a named, strict `v2-large` LeLM profile alongside—not in place of—the
   current v2-medium profile.
2. Generalize the converter's reviewed tensor allow-list by a declarative
   model specification. It must validate the exact source revision, input file
   digest, config digest, tensor names, count, and shapes before writing a
   `LeVo2-v2-large-F16.gguf` artifact.
3. Generalize the C++ strict loader to dispatch by a recognized source
   profile. It must retain exact provenance, hparameter, tokenizer, and tensor
   inventory checks; it must not use the existing loose test-only load path.
   With the same released tensor names, v2-large's 36 main blocks yield 452
   runtime tensors, versus medium's 380.
4. Produce independent LeLM artifacts:

   ```text
   LeVo2-v2-large-F16.gguf
   LeVo2-v2-large-{Q8_0,Q6_K,Q5_K_M,Q4_K_M}.gguf
   ```

   Each receives its own checksum and deterministic manifest. Existing Flow
   and VAE names remain shared because their source bytes and architecture are
   unchanged; duplicating them under a `v2-large` label would falsely suggest
   different weights.
5. Run real **CUDA-only** large-model gates: strict loading, deterministic
   token generation from F16 and every low-bit LeLM profile, and at least one
   native token-to-WAV render through the shared Flow/VAE artifacts. No
   large-model CPU generation or render is in scope.
6. Publish the validated five large LeLM files plus sidecars to
   `ckadirt/LeVo2-GGUF`, update the model card and a dedicated validation
   matrix, then verify public LFS hashes anonymously.
7. Create `artifacts/v2-large-english-120s/` containing a CUDA-generated
   English two-minute WAV, its token/WAV provenance sidecars, English lyrics,
   and a short README. The final song uses the F16 large LeLM and the shared
   F32 Flow / F16-storage VAE at the normal 50 Euler steps unless a recorded
   CUDA safety gate establishes a necessary deviation.

## Non-goals and safety gates

- Legacy v1 base/large checkpoints, v2-fast, audio prompting, and a different
  renderer are out of scope.
- The existing quantization routing applies by tensor role/rank and is expected
  to work for wider large matrices. Acceptance still requires real CUDA
  artifacts; no size-only assumption is sufficient.
- A converter or loader mismatch fails closed. `require_v2_medium = false` is
  test-only plumbing, not a production compatibility solution.
- The large checkpoint is downloaded and checksummed before conversion. A
  public artifact is uploaded only after strict CUDA evidence is committed and
  pushed to GitHub.

## Evidence and reporting

`v2-large-implementation-report.md` is the live chronology for deviations,
commands, generated artifact identities, CUDA timings, public verification,
and the final English song path. The existing quantization report records only
the shared policy; it will link to the large-specific record rather than mix
the two evidence sets.
