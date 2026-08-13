# 10.08-second native quantization benchmark

## Status

In progress. This report records a reproducible performance comparison of the
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
- **Measurement:** one cold-process run per cell, including model load and
  WAV/sidecar output. `/usr/bin/time` supplies process wall time and peak RSS;
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

Pending execution.
