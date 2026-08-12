# Implementation report

## Status

The v0.1 LeLM implementation is complete: strict GGUF conversion/loading,
embedded Qwen2 conditioning, both hierarchical LeLM towers, persistent KV
caches, delayed generation, canonical NPY/JSON artifacts, and the official
Python Flow/VAE decoder bridge are implemented. The native Flow/VAE renderer is
not ported in v0.1. The real-F16 numerical and frozen exact-greedy parity gates
pass with the parity-safe CUDA compute mode described below.

## Environment baseline

- Repository: `/workspace/LeVo2.cpp`
- Branch: `main`
- Initial commit: `50ae8c8`
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24 GB VRAM
- CUDA toolkit: 12.4
- CPU threads: 20
- System RAM: 251 GiB
- Workspace capacity at start: 150 GB, approximately 148 GB available
- Workspace is not volume-backed; pushed commits and remote model artifacts are
  the durability boundary.
- `ckadirt/ggml-fork` is synchronized with official GGML at
  `8846b79e66747bb9f68597420e95114c177315ce`.

## Pinned upstream inputs

- LeVo source: `levo-demo/LeVo@653cbcf4716101834900c75b7d5da43b07e15d5b`
- v2-medium model repository:
  `lglg666/SongGeneration-v2-medium@7d91660ebfa041e29bace194f5631e775796f600`
- Source `model.pt` advertised size: 7,343,951,582 bytes
- Runtime repository:
  `lglg666/SongGeneration-Runtime@cc258cc694a63114c61684cc26d0583b8ad777d0`
- GGML: `ggml-org/ggml@8846b79e66747bb9f68597420e95114c177315ce`

## Milestone log

### Documentation baseline

- Commit: `32b1ca3` (`docs: define LeLM v0.1 execution plan`)
- Pushed to `origin/main`.
- Added the plan, architecture, GGUF, parity, release, and reporting contracts.

### Build foundation

- Added official GGML as a git submodule pinned at `8846b79e`.
- Added C++17 `levo-core`, `levo-cli`, CPU/CUDA backend enumeration, and a
  deterministic GGML vector-add diagnostic.
- CPU release configuration and build passed with GCC 11.4 and GGML 0.19.0.
- CPU CTest: 1/1 passed; diagnostic result was `5.0 5.0 5.0 5.0`.
- CUDA release configuration selected CUDA 12.4 and architecture 89.
- CUDA CTest: 2/2 passed, including execution on NVIDIA GeForce RTX 4090.
- CUDA diagnostic reported compute capability 8.9, VMM enabled, and 24,118 MiB
  total GGML-visible VRAM; result was `5.0 5.0 5.0 5.0`.
- The CUDA compiler emitted an upstream GGML optimizer warning while compiling
  `ggml-cuda.cu`; it did not fail compilation or either runtime test. No GGML
  source was modified.

### GGUF conversion and runtime primitives

- Loaded the pinned 7,343,951,582-byte checkpoint safely on CPU with
  `torch.load(weights_only=True, mmap=True)`; its SHA-256 is
  `4ef2be41f6d838824f5432491408f68d9ffbeda3b1349e1208f9cdfcc64445b1`.
- Classified all 386 checkpoint tensors: 380 runtime-reachable tensors are
  emitted and six explicitly documented unused tensors are omitted.
- The frozen F16 conversion produced 2,728,912,896 parameters in a
  5,467,925,728-byte GGUF. Strict CUDA loading and its SHA-256 sidecar passed;
  SHA-256 is
  `b765d0e79f17cf05c0acdc6cef8bfcd072104adfd8357bb0470f5b9ae91d9e64`.
  The staged artifact passes the strict loader and release parity gates.
- Added deterministic JSON manifests, source/tokenizer provenance and hashes,
  F16/F32 output, shape/type validation, and synthetic converter tests.
- Added a strict C++ schema-1 loader that rejects wrong metadata, unknown or
  missing tensors, wrong shapes/types, invalid offsets, and truncated files.
- Added an uncached two-tower GGML graph with exact RMSNorm, NeoX RoPE, causal
  attention, SwiGLU, hierarchical fusion, exact-erf bridge GELU, and three
  output heads.
- Added Qwen2 byte-level BPE, delayed-pattern build/revert, CFG, repetition,
  mixed/detail top-k sampling, and EOS primitives.
- Integrated six CPU CTests; all passed. Converter PyTest: 3/3 passed.
- Captured a compact real-Python oracle with all 28 main layers, bridge, all 12
  detail layers, conditional/unconditional/CFG logits, and initial greedy IDs
  `[12794, 7883, 12301]`. Large oracle arrays remain ignored.

### Conditioning, caching, generation, and decoder

- Embedded the full Qwen2 byte-level BPE inventory and JSON configuration in
  GGUF; generation has no external tokenizer path.
- Vendored the llama.cpp Unicode regex splitter and utf8proc NFC normalization;
  ten representative tokenizer conformance cases match the pinned
  Transformers tokenizer exactly.
- Reproduced all four fixed `[952,1536]` conditional/null main/detail prefix
  tensors exactly against official Python (`max_abs_error = 0`).
- Added persistent backend-resident K/V caches for both towers and separate
  conditional/null sessions, including prefix prefill, one-token decode, reset,
  context validation, and optional FP16 activation-boundary diagnostics.
- Added the public C++ generator and `levo-cli` interface, exact `[0,250,250]`
  delayed scatter/revert behavior, CFG, upstream repetition logic, sampling,
  per-stream EOS state, earliest-EOS trim, progress reporting, and self-validating
  int32 `[3,T]` NPY/JSON output.
- CPU CTest is 10/10; the CUDA suite adds a real RTX 4090 backend test and is
  11/11. Python lightweight tests are 4 passed with two real-asset tests
  skipped by default; the tokenizer conformance and heavyweight decoder tests
  both pass when explicitly enabled.
- A real two-frame greedy C++ CUDA smoke completed in 15.44 seconds and wrote
  valid tokens. The 250-frame delay makes even short requests execute 252
  delayed steps.
- The same C++ artifact passed the official released `Flow1dVAESeparate` and
  48 kHz VAE renderer at both the one-step test setting and the production
  50-step default. Output was stereo, finite, non-silent, and exactly 0.08 s.
- The decode-only pinned runtime subset is about 5.2 GiB: the 4.81 GB
  `model_septoken/model_2.safetensors`, 675 MB VAE checkpoint/config, and the
  required upstream `stable_audio_tools`/Demucs import sources. No encoder,
  ContentVec, mixed renderer, or Demucs weights are required.
- A final complete sampled 30-second generation using seed 1234 and a two-section
  verse/chorus fixture produced a valid `[3,750]` tensor with no EOS/special
  IDs. Its NPY file SHA-256 is
  `0bb58d470184788369b02d816d68136c83ca733af848e4710b9e73b3617b08ed`.
  The production 50-step official decoder finished in 29.944 s. Its WAV is
  exactly `(1,440,000,2)` at 48 kHz, finite, non-silent (RMS `0.32711`), and
  30.0 s, with SHA-256
  `62f6255a4fee435439db7bfbfb8f7a46f1de47f6ae59f506523b4cd54b827660`.
  Full-window polling observed peaks of 10,332 MiB for C++ generation and
  13,918 MiB for the official decoder on the RTX 4090.

### Real-F16 parity gate

- Direct F16 GGML/PyTorch conditioning input is exact, and an unconditioned
  promoted-F32 graph check has cosine similarity above `0.999998` with exact
  greedy IDs, ruling out tensor-map, tower-order, RoPE-layout, and head-layout
  errors.
- Operator-level tracing localized the original first-token mismatch to the
  first F16 Q/K/V cuBLAS GEMM: the conditioned input and RMSNorm were exact,
  while GGML's default F16 accumulation produced Q/K/V maximum errors of
  `0.00977`, `0.02344`, and `0.02344`. RoPE and attention masking were therefore
  ruled out as the first cause.
- The official PyTorch path uses FP32 accumulation for these F16 GEMMs. GGML's
  upstream `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` path matches that behavior; the
  public generator now selects it by default on CUDA unless the caller has
  explicitly set the variable.
- With FP32 accumulation, both the combined condition+BOS path and the split
  prefix/KV-decode path produce the exact official greedy IDs
  `[12794, 7883, 12301]`. Combined-path CFG logit maximum errors are `0.02539`,
  `0.06641`, and `0.07422`; cosine similarities are `0.99999784`, `0.99999699`,
  and `0.99999630`. The streaming path also passes, with maximum errors
  `0.02539`, `0.06836`, and `0.07031` and cosine similarities at least
  `0.99999583`.
- This is an official GGML runtime switch, not a local GGML patch. The frozen
  thresholds were not changed, and the final greedy equality gate passes.

The frozen generation fixture is lyrics `[verse] Hello world.` with style
`female, pop`. C++ and the official Python oracle are exactly equal at 2.0 s
(`[3,50]`), 10.08 s (`[3,252]`), and 30 s (`[3,750]`). The raw tensor SHA-256
values are, respectively,
`f57268812d4befe556a2b8ee54afae70b657c997616f3b959d7fc5add7ef737a`,
`6f83f15e1ad5815ff215e00ef7c016bf858432a681127f0b9ee28178848dfc98`, and
`95adbd38aeee3188f9a2bd1a770eba3587af7af2ef7fbc7103644a53dda466b0`.
The split prefix/KV-decode strategy is used for this parity run because the
upstream combines prefix and BOS in one first call; the split preserves the
same sequence while avoiding a kernel-boundary numerical difference. Python
generation for 30 s took 75.054 s to load and 30.738 s to generate (108.095 s
wall time); the C++ run took 52.408 s.

An exploratory multilingual 10.08-second input had exactly matching
conditioning tensors and tokenizer IDs but diverged at near-tied greedy logits
(mixed frame 64, vocal frame 0, accompaniment frame 3). This known
cross-library floating-point limitation does not weaken the frozen release
thresholds.

## Deviations from plan

The released checkpoint has a pre-existing style-conditioner packaging
inconsistency: its embedding has 151652 rows while Qwen2 addresses 151646 base
IDs. The converter preserves the six unreachable tail rows and the Python
oracle repairs the released module shape before strict loading. This is an
upstream input defect, not a C++ behavior change.

## Known limitations

- The planned v0.1 target is the single-sample, lyrics/style, v2-medium F16 LeLM
  path only.
- Audio prompt encoding, native audio rendering, quantization, stable C ABI, and
  mobile integration are intentionally deferred.
- The official Python decoder remains required for WAV output in v0.1.
