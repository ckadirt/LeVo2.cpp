# LeVo2.cpp

Portable C++17/GGML inference for the LeVo 2 hierarchical audio language model.

The v0.1 milestone generates three streams of LeVo audio tokens from lyrics and
a style description; the pinned v2-medium F16 LeLM path is implemented and
parity-checked. The v0.2 milestone adds the native Flow/VAE renderer, so tokens
become a 48 kHz stereo waveform without Python. The official Python decoder
remains available as the reference oracle.

> **License:** academic, research, and education use only. Commercial and
> production use are prohibited by the upstream SongGeneration terms.

## Build and test

```bash
git clone --recurse-submodules https://github.com/ckadirt/LeVo2.cpp.git
cd LeVo2.cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA build for an RTX 4090:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

The CLI can enumerate backends, run a deterministic GGML operation, or generate
the canonical three-stream token artifact using only assets embedded in GGUF.
Download `LeVo2-v2-medium-F16.gguf` and its checksum/manifest from
[`ckadirt/LeVo2-GGUF`](https://huggingface.co/ckadirt/LeVo2-GGUF), then verify
it with `sha256sum -c LeVo2-v2-medium-F16.gguf.sha256`.

Example commands:

```bash
./build/bin/levo-cli --list-backends
./build/bin/levo-cli --smoke cpu
./build-cuda/bin/levo-cli --smoke cuda

./build-cuda/bin/levo-cli \
  --model LeVo2-v2-medium-F16.gguf \
  --lyrics lyrics.txt \
  --prompt "female, pop" \
  --duration 30 \
  --output tokens.npy \
  --backend cuda --seed 1234
```

Use `--greedy` for deterministic argmax generation. The output is an int32
NumPy tensor with shape `[3,T]` plus a JSON provenance sidecar.

Both production CLIs report timestamped progress once per second by default.
Generation identifies backend initialization, model loading, conditioning,
prefix prefill, and delayed-token progress with throughput and ETA. Control the
stream with:

```bash
# Machine-readable newline-delimited JSON on stderr
./build-cuda/bin/levo-cli ... --progress json 2>generation.ndjson

# Change the heartbeat interval, or suppress progress and GGML diagnostics
./build-cuda/bin/levo-cli ... --progress-interval 2
./build-cuda/bin/levo-cli ... --quiet
```

`--progress-interval 0` emits every computational event. `--progress none` is
equivalent to `--quiet`. SIGINT/SIGTERM is checked cooperatively between token
positions and exits with status 130 without writing a partial artifact.

The frozen parity fixture is lyrics `[verse] Hello world.` with style
`female, pop`. C++ and the official Python oracle produce byte-identical greedy
arrays at 2.0 seconds (`[3,50]`), 10.08 seconds (`[3,252]`), and 30 seconds
(`[3,750]`). The raw tensor SHA-256 values are
`f57268812d4befe556a2b8ee54afae70b657c997616f3b959d7fc5add7ef737a`,
`6f83f15e1ad5815ff215e00ef7c016bf858432a681127f0b9ee28178848dfc98`, and
`95adbd38aeee3188f9a2bd1a770eba3587af7af2ef7fbc7103644a53dda466b0`.

For CUDA parity, the generator defaults GGML's cuBLAS GEMMs to FP32
accumulation while keeping the model and activation boundaries F16. An explicit
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` environment value overrides that default.

The Qwen2 tokenizer uses the exact vendored llama.cpp Unicode regex splitter
and utf8proc NFC normalization, with the pinned tokenizer configuration
embedded in GGUF. Its 10-case English, punctuation, whitespace, and Unicode
conformance check passes against the official Transformers tokenizer.

## Render tokens to audio natively

`levo-render` turns the canonical token artifact into a 48 kHz stereo WAV using
the native Flow transformer and Oobleck VAE decoder:

```bash
./build-cuda/bin/levo-render tokens.npy \
  --flow-model LeVo2-v2-flow-F32.gguf \
  --vae-model LeVo2-v2-vae-F16.gguf \
  --output song.wav \
  --backend cuda --steps 50 --cfg 1.5 --seed 1234
```

Flow progress includes the current renderer window and every completed Euler
step, for example `window 2/3, Euler 27/50`, plus elapsed time, throughput, and
ETA. VAE progress is reported per decoded window. The same `--progress
plain|json|none`, `--progress-interval`, and `--quiet` controls are available;
SIGINT/SIGTERM is checked at Euler-step, stage, and VAE-window boundaries.

Alongside `song.wav`, the production CLI writes `song.wav.json`. This schema-1
sidecar records the WAV SHA-256 and format, token tensor SHA-256, model/runtime
provenance, backend, resolved Euler/CFG settings, seed, window count, and
per-stage timings. The library retains `write_render_wav` for WAV-only callers
and provides `write_render_artifact` for the paired artifact.

Both renderer GGUFs are published alongside the LeLM artifact in
[`ckadirt/LeVo2-GGUF`](https://huggingface.co/ckadirt/LeVo2-GGUF). The renderer
accepts the separately tagged F16 VAE storage artifact. Its weights, biases,
and SnakeBeta parameters are promoted to F32 in the native correctness graph;
the VAE input, operators, accumulation, and WAV stay F32. Flow remains F32 or
an explicitly selected low-bit profile, never an implicit precision change.

Verify the renderer artifacts before loading them:

```bash
sha256sum -c LeVo2-v2-flow-F32.gguf.sha256
sha256sum -c LeVo2-v2-vae-F16.gguf.sha256
```

The expected hashes are `a8cf50dbecef243501b9b345109b1d2f283b3e22f4e4856715197e4b22129d10`
for Flow and `26f9ea955f586ed3d7668fe345a851ba222b8db95b406e3eea3c9565f4a0b515`
for the F32 VAE baseline. The F16 VAE SHA-256 is
`23e5b11558ae332fbe216d9a06775884469fcbf32236c26ab52defa18c5c8398`.

For measured 10.08-second CPU (eight physical cores) and RTX 4090 CUDA timing
of every LeLM/Flow quantization tier plus F16/F32 VAE storage, see
[`docs/quantization-benchmark-report.md`](docs/quantization-benchmark-report.md).

`--steps 0` and `--cfg 0` select the checkpoint defaults (50 Euler steps,
guidance 1.5). `--noise-f32` replaces the internal Gaussian draw with an
explicit window-major `[windows, 1000, 64]` F32 blob, which is the exact
reproducibility boundary shared with the Python oracle. Output is cropped to
`frames * 1920` samples.

On CUDA the renderer sets `NVIDIA_TF32_OVERRIDE=0` before initializing the
backend, because GGML creates its cuBLAS handles with TF32 tensor-op math and a
1000-frame Flow window would otherwise be computed with a 10-bit mantissa. An
explicit environment value is never overwritten.

## Render tokens with official Python LeVo

The bridge invokes the released dual-stream Flow model and 48 kHz VAE; it
does not reimplement the renderer, and it is the oracle the native renderer is
gated against:

```bash
python python/decode_official.py tokens.npy \
  --output song.wav \
  --source-dir /path/to/pinned/LeVo \
  --runtime-dir /path/to/SongGeneration-Runtime \
  --device cuda
```

The runtime directory needs the pinned `model_septoken` checkpoint, VAE
checkpoint/config, and upstream Python dependencies described in
[the implementation report](docs/implementation_report.md). The production
renderer default is 50 Euler steps.

## Convert the pinned checkpoint

```bash
python -m pip install -r convert/requirements.txt
python convert/levo2_to_gguf.py /path/to/model.pt \
  --tokenizer-dir /path/to/Qwen2-7B \
  --config /path/to/config.yaml \
  --dtype F16 \
  --output LeVo2-v2-medium-F16.gguf
```

Conversion emits the GGUF, a deterministic `.manifest.json`, and a `.sha256`
sidecar. The converter accepts only the pinned v2-medium tensor inventory and
does not instantiate upstream Python model code.

See [the execution plan](docs/plan.md), [architecture contract](docs/architecture.md),
[parity policy](docs/parity.md), and [implementation report](docs/implementation_report.md).

## Resumable staged generation

The shared Cantor engine supports cross-process pause/resume for LeLM `CODES`
and Flow `DIFFUSE`: it writes self-contained token or latent checkpoints rather
than model/KV state. Its VAE `DECODE` stage retries from the durable Flow
boundary if stopped. `levo-cantor` is the corresponding command-line adapter;
it accepts a strict fresh request JSON and atomically fsyncs each checkpoint.

```json
{
  "lyrics": "Come back to the light",
  "description": "warm indie pop, female vocal",
  "duration_seconds": 20,
  "seed": 1234,
  "flow": { "seed": 5678, "euler_steps": 50, "cfg_scale": 1.5 }
}
```

```bash
./build-cuda/bin/levo-cantor \
  --input request.json --checkpoint song.resume --output song.wav \
  --lm LeVo2-v2-medium-F16.gguf \
  --dit LeVo2-v2-flow-F32.gguf --vae LeVo2-v2-vae-F32.gguf

# After a successful SIGINT/SIGTERM pause (exit 130), resume from only the blob.
./build-cuda/bin/levo-cantor \
  --resume song.resume --checkpoint song.resume --output song.wav \
  --lm LeVo2-v2-medium-F16.gguf \
  --dit LeVo2-v2-flow-F32.gguf --vae LeVo2-v2-vae-F32.gguf
```

`levo-cantor` keeps the last durable checkpoint on successful decode as well;
delete it explicitly when it is no longer wanted. The full staged contract,
determinism rules, and remaining validation gates are in the
[resumability plan](docs/resumability-plan.md).

The native renderer has its own
[execution plan](docs/renderer-plan.md),
[architecture contract](docs/renderer-architecture.md),
[GGUF contract](docs/renderer-gguf.md),
[parity policy](docs/renderer-parity.md), and
[implementation report](docs/renderer-implementation-report.md).

Every release case is gated against the pinned Python oracle from the same
stored initial noise, comparing per-window latents, per-window decoded audio,
and the assembled waveform. Reproduce it with the parity tools enabled:

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DLEVO_BUILD_PARITY_TOOLS=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j

LEVO_RUN_NATIVE_RENDER_PARITY=1 \
  LEVO_RENDER_PARITY_TOOL=build-cuda/levo-render-parity \
  LEVO_FLOW_F32_GGUF=LeVo2-v2-flow-F32.gguf \
  LEVO_VAE_F32_GGUF=LeVo2-v2-vae-F32.gguf \
  python -m pytest -q tests/python/test_native_render_parity.py
```

The committed [v0.2 matrix](docs/renderer-release-matrix.json) covers 50, 252,
750, and 1,250 frames at both one and fifty Euler steps. It also records a real
30-second `levo-cli` → `levo-render` smoke: stereo IEEE-F32, 48 kHz, finite,
non-silent, and exactly 1,440,000 samples per channel.
