# LeVo 2 v2-large implementation report

## Status

In progress. This report is the live evidence record for the v2-large LeLM
extension. It supplements the existing v2-medium and quantization reports; it
does not rewrite their historical validation.

## Operating method

1. Commit and push source/documentation milestones before model publication.
2. Treat v2-large as a distinct, strict LeLM source profile.
3. Use CUDA for every real large-model generation and render gate. No
   large-model CPU generation or render is claimed.
4. Publish a GGUF only after its matching source commit, strict loader,
   checksum/manifest, CUDA evidence, and model-card entry exist.
5. Record every deviation instead of silently broadening a contract.

## Milestone log

### 2026-08-13 — contract established

- Pinned `lglg666/SongGeneration-v2-large` revision
  `115805364ad74479fb3764fe65970c92faeb1a5a`, its 12,899,965,446-byte
  `model.pt` SHA-256
  `dc763aa9a76a22a87597c2faf9a51c24d13349ac754699b37e9068b483639def`,
  and the 3,352-byte `config.yaml` SHA-256
  `14a991bd7342b9dde348e6324afd44b5c6ecb1db8d0ed4d2dbe666b220b04c59`.
- Verified the v2-large configuration changes LeLM dimensions only
  (`2048/11008`, 36 main blocks, 16 heads) while retaining v2-medium's token
  contract, Flow source, Stable Audio 1920 VAE, 25 Hz frame rate, 48 kHz
  output, and `[0,250,250]` hierarchy delay.
- Recorded the full approved scope and explicit no-duplicate-renderer rule in
  [`v2-large-plan.md`](v2-large-plan.md). Pending: profile-based converter and
  strict loader implementation, followed by CUDA-only validation.

### 2026-08-13 — profile-aware conversion and loading implemented

- Replaced the converter's v2-medium-only architecture constants with reviewed
  immutable `v2-medium` and `v2-large` specifications. The `v2-large`
  specification declares 2048 hidden width, 11008 FFN width, 36 main blocks,
  12 detail blocks, 16 attention/KV heads, and exactly 452 runtime tensors.
- The production CLI selects a named `--variant` (`v2-medium` is the
  compatibility default; this release explicitly passes `v2-large`) and
  verifies the selected `model.pt` byte count/SHA-256 and `config.yaml` byte
  count/SHA-256 before conversion. Its hidden unverified mode exists solely
  for the tiny deterministic unit fixture; it is not used for release work.
- Generalized the strict C++ loader from a medium-only branch to a recognized
  v2 source-profile dispatcher. It checks the profile name, repository,
  revision, source/config digests, tokenizer/runtime identity, every
  architecture field, exact tensor shapes/types, and the profile-specific
  inventory count before allocating weights.
- Rebuilt the CUDA configuration and ran the converter test suite with the
  project environment: **4 passed**. This is source-level validation only;
  no large checkpoint inference or render has been run on CPU.

### 2026-08-13 — conversion gate deviation: output-head profile leak

- The first real, hash-verified v2-large conversion stopped before writing a
  GGUF because its `output.vocal` tensor was `[16385, 2048]` while two residual
  converter rules still expected the medium width `[16385, 1536]`.
- This was a converter-specification defect, not an upstream mismatch. The
  strict shape gate did exactly what it should: no artifact was emitted. Both
  output-head rules now use the selected profile width, with a regression
  assertion for the v2-large shapes. Conversion is being retried from the
  verified local source only after that source/test fix.

### 2026-08-13 — verified F16 conversion and five-profile CUDA matrix

- Downloaded the pinned source into the local Hugging Face cache and verified
  both exact byte counts and SHA-256 values before conversion. The corrected
  strict converter emitted **452** runtime tensors / **4,988,100,608**
  parameters in `LeVo2-v2-large-F16.gguf` (9,986,305,440 bytes,
  `368cba66fabbdca3d208d4c15a9c9c2059ea5d8ca2a822ee3a3f1d9f854134d5`).
- Generated the complete native LeLM catalog from that F16 source with policy
  revision 1: Q8_0 (5,310,562,912 bytes), Q6_K (4,102,556,992 bytes), Q5_K_M
  (3,761,638,720 bytes), and Q4_K_M (3,440,774,464 bytes). Every checksum
  sidecar was recomputed and verified before validation.
- Strict CUDA loading and deterministic greedy generation passed for F16 and
  all four quantized artifacts on an RTX 4090 (driver 595.84, 24,564 MiB).
  The compact fixture requested two seconds; some profiles emitted an earlier
  valid EOS and therefore have different final frame counts. This is expected
  model behavior, not a partial artifact. Exact token hashes and timings are
  stored in [`v2-large-validation-matrix.json`](v2-large-validation-matrix.json).
- A complete native CUDA pipeline gate used the Q6_K LeLM, shared F32 Flow,
  and shared F16-storage VAE at 50 Euler steps. It produced 50 frames of
  finite, non-silent, stereo 48 kHz audio in 17.34 s. The renderer is shared
  by design; no fake `v2-large` Flow/VAE copies were created.
- The CUDA build's full CTest suite passed **32/32**. No v2-large CPU
  generation or render was performed; CPU was used only for offline GGUF
  requantization.

### 2026-08-13 — requested two-minute English CUDA song

- Generated original English lyrics and sampled exactly 3,000 F16 v2-large
  frames (120.0 s) on CUDA with seed `2026081301`. Token generation took
  314.656 s, including 26.967 s model load; token SHA-256 is
  `1bfa15e30cdc927b0b319ebaae152f6457886a9b75ad6ef51adada68e8dc67a6`.
- Rendered those tokens with the shared F32 Flow and F16-storage VAE at the
  normal 50 Euler steps. The output is a finite, non-silent 48 kHz stereo
  IEEE-F32 WAV with 5,760,000 samples/channel (120.0 s), SHA-256
  `f3473bd200df88ec33820f867c523cf8bc1df292bd0e4759f0f555a59c38edee`.
  Rendering took 45.501 s over four Flow windows.
- The ignored working artifact directory is
  `artifacts/v2-large-english-120s/`; it contains the lyrics, canonical token
  artifact plus provenance, WAV, and WAV provenance. The validation matrix
  records its exact model identities and audio statistics.

### 2026-08-13 — Hugging Face publication and anonymous verification

- Published the F16 artifact and sidecars, then the four low-bit artifacts and
  sidecars, to [`ckadirt/LeVo2-GGUF`](https://huggingface.co/ckadirt/LeVo2-GGUF).
  The completed catalog/card revision is
  [`9d7b5746fdc74fdc80f85295e7b6c783be3703da`](https://huggingface.co/ckadirt/LeVo2-GGUF/commit/9d7b5746fdc74fdc80f85295e7b6c783be3703da).
- Queried the public repository without credentials and verified every LFS
  object's exact byte count and SHA-256 against the local strict manifests.
  Downloaded all ten small checksum/manifest sidecars with `token=False` and
  compared them byte-for-byte to the local files: all passed.
- Published the matching model card, artifact naming document, and
  [`V2-LARGE-VALIDATION-MATRIX.json`](https://huggingface.co/ckadirt/LeVo2-GGUF/blob/9d7b5746fdc74fdc80f85295e7b6c783be3703da/V2-LARGE-VALIDATION-MATRIX.json)
  in that final Hub revision. The public catalog, source documentation, and
  local generated-song provenance now carry the same five artifact identities.
