# LeVo2.cpp

Portable C++17/GGML inference for the LeVo 2 hierarchical audio language model.

The v0.1 milestone generates three streams of LeVo audio tokens from lyrics and
a style description. Audio rendering remains in the official Python decoder
until a later milestone.

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
the canonical three-stream token artifact using only assets embedded in GGUF:

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

For CUDA parity, the generator defaults GGML's cuBLAS GEMMs to FP32
accumulation while keeping the model and activation boundaries F16. An explicit
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` environment value overrides that default.

## Render tokens with official Python LeVo

The v0.1 bridge invokes the released dual-stream Flow model and 48 kHz VAE; it
does not reimplement the renderer:

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
