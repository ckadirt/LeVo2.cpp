# Implementation report

## Status

The build foundation, strict GGUF converter/loader, uncached hierarchical LeLM
graph, tokenizer, delayed pattern, and sampling primitives are implemented.
KV-cached conditioning and generation are the next milestone.

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
- A real F16 conversion produced 2,728,912,896 parameters in a 5.1 GiB GGUF.
  Reader validation and the artifact SHA-256 sidecar passed. This pre-release
  artifact will be regenerated after the final metadata/runtime contract is
  frozen; it has not been uploaded.
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

## Deviations from plan

None. The released checkpoint has a pre-existing style-conditioner packaging
inconsistency: its embedding has 151652 rows while Qwen2 addresses 151646 base
IDs. The converter preserves the six unreachable tail rows and the Python
oracle repairs the released module shape before strict loading. This is an
upstream input defect, not a C++ behavior change.

## Known limitations

- The planned v0.1 target is the single-sample, lyrics/style, v2-medium F16 LeLM
  path only.
- Audio prompt encoding, native audio rendering, quantization, stable C ABI, and
  mobile integration are intentionally deferred.
