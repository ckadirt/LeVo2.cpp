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

### Release publication

- Final implementation commit: `1f3df56` (`fix: close LeLM parity and
  tokenizer gates`), pushed to `origin/main`.
- The public F16 artifact is hosted at `ckadirt/LeVo2-GGUF`, immutable revision
  `3df87c4dd2b32e7c8f89caa6af534585cffda894`.
- The public repository reports the exact 5,467,925,728-byte LFS object and
  SHA-256 `b765d0e79f17cf05c0acdc6cef8bfcd072104adfd8357bb0470f5b9ae91d9e64`.

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

## Post-v0.1 native-renderer addendum

This report is the immutable historical record of the `v0.1.0` token-generator
release. The statement above remains correct for that tag, but not for current
`main`: v0.2 adds a fully native C++/GGML Flow and Oobleck VAE renderer.

The renderer implementation history, release deviations, full eight-case
Python/C++ parity matrix, performance, and WAV evidence are maintained in
`docs/renderer-implementation-report.md` and
`docs/renderer-release-matrix.json`. The v0.1 LeLM F16 GGUF is unchanged and is
the token-generation half of the v0.2 end-to-end pipeline. The public Flow and
VAE artifacts are pinned by the model-repository tag `v0.2.0` and immutable
Hugging Face revision `04b6819a185fb33fc5e35669688694d820bacb26`; their
anonymous download, hash, strict-loader, and execution checks are recorded in
the renderer report.

## Post-v0.2 observability milestone

The released v0.2 callbacks identify token-generation steps and broad renderer
stages, but they do not expose Flow Euler progress, elapsed time, ETA,
cancellation, structured logs, or durable WAV provenance. This became material
during a real 20-second CPU run: LeLM generation took 4m36s and the 50-step
single-window Flow/VAE render took 15m10s, with approximately 14.5 minutes of
silence inside the Flow stage.

The approved remediation contract is frozen in
`docs/observability-plan.md`. Implementation begins from clean pushed source
commit `49b053b0c676630142434ac2d43f511d60d9825a`; generated songs and model
artifacts remain ignored.

### Observability implementation trace

- `20eea39` (`docs: define native observability contract`) froze event stages,
  CLI modes, cancellation behavior, the WAV sidecar schema, and verification
  gates before implementation.
- `f4c3663` (`feat: expose native pipeline progress telemetry`) added public
  generation/render stage and timing fields, per-window/per-Euler Flow events,
  optional cancellation predicates, and the distinct `operation_cancelled`
  exception. The existing `void` callback type remains source-compatible.
- `46e0553` (`feat: add structured progress and render provenance`) added
  plain and NDJSON CLI reporters, configurable one-second heartbeats, ETA and
  throughput, quiet mode, SIGINT/SIGTERM handling, generation timing metadata,
  and staged `song.wav` plus `song.wav.json` output with streaming SHA-256.
- The first incremental compile exposed a missing direct `levo.h` include in
  the internal Flow renderer header. Adding the dependency fixed the compile;
  no graph or numerical code changed.
- GGML emits CUDA device diagnostics directly to stderr by default. Successful
  NDJSON streams would therefore not be parseable line-by-line. JSON and quiet
  modes install a no-op GGML callback; plain mode retains the useful upstream
  diagnostics. This affects logging only.

Asset-free focused tests cover monotonic generation progress, exact Euler event
counts, generation and Flow cancellation boundaries, progress-mode/interval
validation, plain and JSON formatting, heartbeat repetition, quiet output,
token timing metadata, render-sidecar content/hash, and cleanup when WAV
validation fails. The complete CPU suite passes 22/22 and the complete CUDA
suite passes 26/26.

A real one-window CUDA smoke with five Euler steps printed every step with live
rate and ETA, then the VAE and assembly stages. A separate one-step structured
smoke emitted 12/12 JSON-parseable stderr events covering all nine renderer
stages. Its schema-1 sidecar reports a 768,044-byte, two-second stereo F32 WAV
with SHA-256
`958d41ffa51ecb355d848537849780af686d1bb0fc83b3e330121a7192f346c3`;
the independently hashed file agrees. Measured render time was 3.364 s and
artifact completion 3.369 s.

A real two-second CUDA LeLM smoke emitted 307 parseable generation events in
the six required stages, completed 300/300 delayed positions, and persisted
backend/model/conditioning/prefill/generation/total timings. Total generation
time was 18.428 s. A quiet one-step render produced no stderr bytes. Sending
SIGINT immediately after Flow Euler step 1/50 exited with status 130 and left
neither a WAV nor metadata sidecar.

The frozen one-window renderer parity smoke remains numerically unchanged:
normalized-latent maximum error `4.8160553e-5`, decoded-window maximum error
`1.77033246e-4`, and audio maximum error `2.87592411e-5` against the same
oracle. No threshold, model input, noise value, tensor layout, or GGML graph
changed.

### Observability limitations and deviations

GGUF loading and each VAE decode are monolithic lower-layer calls. Their exact
start, completion, and elapsed time are reported, while a CLI heartbeat proves
liveness between those boundaries; byte-level loader progress and per-operator
VAE progress are not fabricated. Cooperative cancellation likewise takes
effect at the next safe boundary and cannot interrupt a GGML graph already in
flight. These are the explicit granularity limits of this milestone.

## Post-observability resumability planning audit

On 2026-08-13, the current cancellation paths were audited against the local
ACE-Step staged-engine implementation (`acestep.cpp` commit `79994ed`, staged
engine merge `7985032`). LeVo2.cpp was at clean pushed commit `12a1253` when the
audit began.

The audit confirms that current `main` is not resumable. LeLM cancellation
destroys the partial delayed sequence, sampler/EOS state, and paired K/V caches;
Flow cancellation destroys the current Euler tensor and completed latent
windows; VAE cancellation destroys decoded windows. The CLIs correctly avoid
partial artifacts, but no cross-process checkpoint exists.

The proposed contract and implementation order are frozen in
`docs/resumability-plan.md`. The principal design findings are:

- LeLM should checkpoint its small delayed token stream, resolved seed, and RNG
  draw cursor, then rebuild K/V with the exact sequential graph shape. The
  existing combined prefill is deliberately not used because prior parity work
  proved graph-boundary numerical differences can flip near-tied logits.
- Flow should checkpoint exact initial noise, completed denormalized windows,
  the current normalized Euler state, and `(window,step)`. Fixed-step Euler and
  LeVo's direct CFG expression have no hidden cross-step history.
- The durable Flow-to-VAE boundary must retain overlapping 1000-frame latent
  windows; an assembled raw `[T,64]` tensor cannot reproduce the official
  decode-then-crossfade path.
- VAE can initially pause with no new blob and rerun from the completed Flow
  boundary, but the measured CPU graph duration requires abort-callback and/or
  chunking work before the 20-second host shutdown guarantee can be claimed.
- A Cantor-compatible C ABI, self-identifying checked blobs, strict
  backend/model/numerical stamps, thread-local errors, exception guards, and
  context re-entry tests are part of the milestone rather than follow-up work.

Intentional differences from ACE-Step are recorded in the plan: LeVo has no
`PLAN` stage; its stateless Flow CFG remains resumable above 1.0; completed Flow
state is windowed; exact Flow noise is persisted to avoid Box-Muller/libm drift;
and the local GGML CPU abort callback will be evaluated for long graph latency.

This planning change does not alter inference code, formats, tests, or release
artifacts. Implementation and validation have not started.

### Resumability implementation trace

- Foundation slice: added a C-compatible `cantor_engine.h` and a hidden-symbol
  `liblevo-cantor-engine` target. All ABI symbols are present and pure-C loading,
  error, ownership, and discovery paths are tested. Until the LeLM stage is
  wired it honestly reports an empty stage mask and rejects execution rather
  than advertising a partial feature.
- Added the common self-identifying binary blob envelope. It uses explicit
  little-endian fields, a checked section directory, exact length/overflow/
  overlap validation, a 2 GiB ceiling, and standard SHA-256 with the digest
  field zeroed while hashing. Asset-free tests cover successful round trip,
  known SHA-256 output, wrong magic, truncation, and corruption.
- `Sampler` now records its exact raw `mt19937_64` draw cursor and restores from
  seed plus `discard(cursor)`. Sampling, greedy no-draw, and mixed raw draws are
  covered by continuation tests.
- Flow's fixed-step Euler primitive now exposes a completed-step/normalized-
  state boundary. Fresh behavior is unchanged; restarting at every synthetic
  step is exact and malformed resume states are rejected.

The first slice was intentionally infrastructure only. The following CODES
slice now supersedes its "no request parser/serialized LeLM state" limitation;
Flow window checkpointing, VAE pause, and CLI flags remain pending.

### Resumability implementation trace: CODES stage

- The shared engine now advertises only `CANTOR_STAGE_CODES`. It accepts fresh
  bounded request JSON or sniffs a `LEVOLM01` blob, and returns
  `CANTOR_PAUSED` (status `1`, thread-local `CANTOR_ERR_CANCEL`) at a delayed
  token boundary rather than treating stop as an error.
- Fresh requests resolve a missing seed once, serialize it canonically into the
  blob, and never require the caller to supply it again. The blob contains
  `[3,S]` delayed int32 IDs, exact `mt19937_64` raw-draw cursor, EOS validation
  state, a SHA-256 digest of the rebuilt next logits, and backend/model/runtime/
  tokenizer stamps. K/V is never serialized.
- Resume recreates the ordinary split prefill and replay-decodes each saved
  delayed position before comparing the logit digest and continuing. A stop
  while replaying returns the original logical checkpoint, so the durable
  boundary is never weakened by a partial K/V rebuild.
- The request decoder deliberately differs from the original `yyjson` plan: a
  small strict decoder was implemented for the fixed schema, with duplicate and
  unknown-field rejection, UTF-8 validation, JSON-number grammar validation,
  1 MiB input limit, and exact `uint64` seed parsing. This avoids a new third-
  party dependency; it must be replaced with a pinned parser if the schema
  grows materially.
- Asset-free verification passed on the CPU build: `cantor-abi`,
  `engine-request`, `generation`, and `sampling`. These cover ABI discovery and
  error ownership, strict request rejection, pause/resume controller equality,
  and sampler continuation. Real-model CPU/CUDA cross-process equivalence is
  still a required later validation gate.

### Resumability implementation trace: DIFFUSE stage

- The engine now also advertises `CANTOR_STAGE_DIFFUSE`. Its fresh input is the
  self-contained CODES completion (canonical request plus stream-major token
  IDs and SHA-256); request JSON carries Flow seed, Euler-step, and CFG controls
  through CODES unchanged.
- A pause returns `LEVOFL01` with exact initial F32 noise, only completed
  denormalized `[1000,64]` windows, and (when inside a window) the normalized
  post-update Euler state plus its step cursor. The blob also stamps the token
  SHA-256, resolved Flow parameters, schedule, backend, Flow model SHA-256,
  and runtime revision. Direct CFG and Euler have no unpersisted history.
- Resume re-derives every conditioning tensor, restores the active Euler tensor
  at its recorded step, and derives the next-window continuation from the last
  stored raw window. It refuses schedule, token, backend, model, or runtime
  mismatches. A completed solve returns `LEVOLT01`, retaining only the request,
  codes, Flow stamp/schedule, and raw overlapping windows needed by VAE.
- Asset-free tests still cover the underlying fixed-step cursor at every Euler
  boundary and the stage discovery/strict JSON paths. A real-model DIFFUSE
  pause/resume parity run remains pending; it is deliberately not claimed by
  this source-level milestone.
