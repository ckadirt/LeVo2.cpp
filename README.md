# LeVo2.cpp

Portable C++17/GGML inference for the LeVo 2 hierarchical audio language model.

The v0.1 milestone generates three streams of LeVo audio tokens from lyrics and
a style description. Audio rendering remains in the official Python decoder
until a later milestone.

> **License:** academic, research, and education use only. Commercial and
> production use are prohibited by the upstream SongGeneration terms.

## Foundation build

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

The current foundation CLI can enumerate backends and run a deterministic GGML
operation:

```bash
./build/bin/levo-cli --list-backends
./build/bin/levo-cli --smoke cpu
./build-cuda/bin/levo-cli --smoke cuda
```

See [the execution plan](docs/plan.md), [architecture contract](docs/architecture.md),
and [implementation report](docs/implementation_report.md).
