# 10.08-second native quantization benchmark

## Status

Complete. This report records a reproducible performance comparison of the
published LeLM, Flow, and VAE precision variants. It is deliberately separate
from the correctness/quality matrix: a lower wall time is not a quality claim.

## Benchmark contract

- **Date:** 2026-08-13
- **Audio duration:** 10.08 seconds / 252 token frames / 483,840 samples per
  channel after native crop.
- **LeLM input:** the frozen greedy parity prompt, lyrics `[verse] Hello
  world.` and description `female, pop`, duration `10.08`, seed `1234`.
- **Renderer input:** the existing canonical
  `artifacts/render-parity/tokens-10.08s.npy` fixture. Supplying the same
  token artifact isolates Flow and VAE performance from LeLM token changes.
- **Flow settings:** one Euler step, CFG 1.5, seed 1234. One step makes the
  complete CPU/CUDA/profile matrix tractable while exercising the production
  Flow graph and a full 1,000-frame VAE decode. Results must not be presented
  as 50-step production-song latency.
- **CPU isolation:** `taskset --cpu-list 0-7,16-23`, selecting both hardware
  threads of physical cores 0--7: eight of the host's 16 physical cores and
  16 of its 32 logical CPUs. This is an enforcement boundary, not merely a
  thread-count hint. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `MKL_NUM_THREADS` are also set to 16 for any linked library that honors
  them.
- **CUDA isolation:** CUDA device 0, with the renderer's existing F32
  accumulation / TF32-disabled policy.
- **Measurement:** one fresh-process run per cell, including model load and
  WAV/sidecar output. POSIX `time -p` supplies process wall/user/system time;
  generated sidecars supply internal stage timing. Model-file I/O caching is
  not flushed, so values include normal OS page-cache state and must be read
  as a realistic single-run comparison, not a statistically rigorous
  cold-storage benchmark.

## Matrix

| Component | Variants | Backends |
| --- | --- | --- |
| LeLM token generation | F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M | CPU (half host cores), CUDA |
| Flow render | F32, Q8_0, Q6_K, Q5_K_M, Q4_K_M with VAE F16 fixed | CPU (half host cores), CUDA |
| VAE decode | F32 and F16 storage with Flow F32 fixed | CPU (half host cores), CUDA |

The Flow and VAE rows are separate control experiments. They do not add the
latencies of two different Flow runs to one end-to-end render.

## Results

### Platform and provenance

| Item | Value |
| --- | --- |
| Source revision at benchmark start | `062bd6ea38642d3b6295f3826931a3c487b69436` |
| CPU | AMD Ryzen 9 5950X, 16 physical cores / 32 logical CPUs |
| CPU benchmark affinity | logical CPUs `0-7,16-23` (eight physical cores; 16 logical CPUs) |
| GPU | NVIDIA GeForce RTX 4090, CUDA compute capability 8.9, 24,082 MiB reported VRAM |
| Token fixture used for Flow/VAE | `artifacts/render-parity/tokens-10.08s.npy`, SHA-256 `98c1b74c79e629fcf0b7ddda74d13f8d2da19cadbb326fd6273196fb9846d7aa` |
| Completed evidence | 10 LeLM token artifacts and 12 renderer artifacts; every token artifact is `[3,252]`, every renderer artifact is finite stereo PCM-F32 WAV at 48 kHz with 483,840 samples/channel (10.08 s) |

The figures below are one completed process per cell. **Wall** is POSIX
`time -p real`, including process startup, backend initialization, model load,
and artifact writes. Stage values are the production sidecar's internal
timings and do not include WAV/JSON writing. All values are seconds, rounded
to two decimals.

### LeLM: lyrics/style to 10.08-second token artifact

Greedy sampling, seed 1234, and the same lyrics/style input were used for all
five profiles. `load / prefill / generate` is shown to make the sequential
generation bottleneck explicit.

| LeLM profile | GGUF size (GiB) | CPU wall | CPU load / prefill / generate | CUDA wall | CUDA load / prefill / generate |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 | 5.09 | 201.74 | 16.28 / 38.47 / 146.53 | 35.91 | 15.31 / 0.33 / 19.84 |
| Q8_0 | 2.71 | 151.52 | 8.88 / 33.33 / 108.96 | 16.97 | 8.38 / 0.18 / 8.04 |
| Q6_K | 2.09 | 137.31 | 6.88 / 32.46 / 97.65 | 15.14 | 7.08 / 0.20 / 7.50 |
| Q5_K_M | 1.94 | 136.91 | 6.32 / 36.22 / 94.06 | 14.88 | 6.57 / 0.20 / 7.71 |
| Q4_K_M | 1.79 | 125.68 | 5.84 / 29.62 / 89.92 | 12.99 | 5.45 / 0.19 / 6.98 |

Relative to F16, Q4_K_M reduces end-to-end LeLM wall time by 1.61x on the
half-host CPU and 2.76x on CUDA. Its delayed-token generation phase is 1.63x
and 2.84x faster respectively. Q6_K is the quality-favored Flow tier from the
separate waveform matrix; these timing values do not override that quality
evidence.

### Flow: fixed 10.08-second tokens to latents, VAE F16 held constant

Every Flow row uses the F16 VAE storage artifact so the comparison changes
only Flow. The VAE decode is still present in the total because this is the
production renderer, but it is displayed separately and is expected to remain
almost constant. `load / Flow / VAE decode` refers to internal stage timing.

| Flow profile | GGUF size (GiB) | CPU wall | CPU load / Flow / VAE | CUDA wall | CUDA load / Flow / VAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| F32 | 2.47 | 72.13 | 8.04 / 17.17 / 46.16 | 11.76 | 7.99 / 0.22 / 2.68 |
| Q8_0 | 0.78 | 62.42 | 2.55 / 12.93 / 46.28 | 6.31 | 2.59 / 0.18 / 2.69 |
| Q6_K | 0.66 | 61.83 | 2.17 / 12.99 / 46.03 | 5.96 | 2.22 / 0.18 / 2.68 |
| Q5_K_M | 0.61 | 62.97 | 1.99 / 14.29 / 46.05 | 5.76 | 2.06 / 0.19 / 2.68 |
| Q4_K_M | 0.56 | 60.77 | 1.84 / 12.09 / 46.20 | 5.61 | 1.90 / 0.19 / 2.68 |

On CPU, the fixed VAE decode dominates this one-step render (~46 seconds), so
the Q4_K_M Flow artifact produces a modest 1.19x end-to-end renderer speedup
despite a 1.42x faster Flow phase. On CUDA, quantization's largest effect is
model load: F32 Flow load is 7.99 seconds versus 1.90 seconds for Q4_K_M;
the Flow compute phase itself changes only from 0.224 to 0.186 seconds. Thus
the CUDA end-to-end renderer improves 2.10x here, primarily from the smaller
artifact rather than a 2x kernel-throughput claim.

### VAE: F16 storage versus F32 storage, Flow F32 held constant

The F16 rows are the Flow-F32 controls above; the F32 rows repeat the same
render with only the VAE artifact changed. The decoder graph deliberately
promotes F16 stored tensors to F32, so a nearly unchanged decode time is the
intended result.

| VAE storage | GGUF size (GiB) | CPU renderer wall | CPU VAE load / decode | CUDA renderer wall | CUDA VAE load / decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 storage, F32 graph | 0.16 | 72.13 | 0.52 / 46.16 | 11.76 | 0.51 / 2.68 |
| F32 | 0.31 | 72.94 | 1.03 / 46.47 | 12.32 | 1.01 / 2.68 |

F16 VAE storage halves the decoder artifact and its load time (~2x on both
backends), while this single-run decode time is effectively unchanged. That
matches the documented design: it is a transfer and storage optimization, not
an F16 activation/accumulation implementation.

### Indicative composed lyrics-to-WAV latency

The following sums pair a LeLM cell with the independently measured renderer
cell of the same profile, always using VAE F16. They are useful capacity
estimates, but not a separate monolithic process measurement: Flow/VAE use the
fixed token fixture to isolate their component timings, and quantized LeLM
profiles may emit different valid token IDs.

| Pairing (VAE F16) | CPU wall sum | CUDA wall sum |
| --- | ---: | ---: |
| F16 LeLM + F32 Flow | 273.87 s | 47.67 s |
| Q8_0 LeLM + Q8_0 Flow | 213.94 s | 23.28 s |
| Q6_K LeLM + Q6_K Flow | 199.14 s | 21.10 s |
| Q5_K_M LeLM + Q5_K_M Flow | 199.88 s | 20.64 s |
| Q4_K_M LeLM + Q4_K_M Flow | 186.45 s | 18.60 s |

These one-Euler figures should not be multiplied blindly for a 50-step song:
loading occurs once, while Flow work scales with Euler steps and the VAE decode
does not. The stage columns above are the appropriate basis for a workload
model.

## Reproduction and evidence

The non-versioned raw benchmark output (commands, `time -p` logs, token/WAV
artifacts, and their normal sidecars) is retained at:

```text
/workspace/benchmarks/levo2-quantization-10s-20260813/
```

The final CPU F16 and Q4_K_M figures come from `lelm/cpu-clean/`, rerun after
all CUDA work completed. Earlier overlapping probes remain in `lelm/cpu/` for
audit but are deliberately not used in the table.

Representative exact commands (substitute a profile/model path and output
directory for every matrix cell) are:

```bash
# LeLM on exactly half the host's physical cores.
time -p taskset --cpu-list 0-7,16-23 \
  env OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  ./build-cpu/bin/levo-cli \
  --model /workspace/models/quantization-staging/LeVo2-v2-medium-Q6_K.gguf \
  --lyrics /workspace/benchmarks/levo2-quantization-10s-20260813/inputs/lyrics-hello-world.txt \
  --prompt 'female, pop' --duration 10.08 --output tokens.npy \
  --backend cpu --seed 1234 --greedy --progress none

# Flow profile control: identical canonical tokens and F16 VAE storage.
time -p ./build-cuda/bin/levo-render artifacts/render-parity/tokens-10.08s.npy \
  --flow-model /workspace/models/quantization-staging/LeVo2-v2-flow-Q6_K.gguf \
  --vae-model /workspace/models/quantization-staging/LeVo2-v2-vae-F16.gguf \
  --output song.wav --backend cuda --steps 1 --cfg 1.5 --seed 1234 --progress none
```

## Deviations and limits

- Two initial CPU LeLM probes were started while the CUDA queue was still
  draining. They are retained as raw evidence but excluded; CPU F16 and
  Q4_K_M were rerun after CUDA was idle and those clean results are reported.
  Q8_0/Q6_K/Q5_K_M started after CUDA completion and did not overlap it.
- No attempt was made to flush the kernel page cache, lock CPU frequency, or
  collect repeated samples. This is a transparent single-run engineering
  benchmark, not a confidence-interval study or a cold-storage I/O claim.
- POSIX `time` is available on this host, but GNU `/usr/bin/time` is not, so
  peak RSS is intentionally not reported rather than inferred.
- CPU affinity limits the process to half the physical cores; it does not
  guarantee that every GGML graph reaches all 16 allowed logical CPUs. The
  measured user times demonstrate multi-core work during the relevant graph
  phases, while CPU utilization is implementation- and operation-dependent.
- The Flow/VAE controls use only one Euler step. Quality remains governed by
  the frozen validation matrix, and real production renders normally use the
  checkpoint default of 50 steps.
