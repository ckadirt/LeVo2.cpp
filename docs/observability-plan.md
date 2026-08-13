# Native pipeline observability plan

## Motivation

The v0.2 API exposes callbacks, but the renderer reports only coarse stage
boundaries. A 20-second, 50-step CPU render remained at
`generating Flow latents 0/1` for approximately 14.5 minutes even though it was
healthy. Multi-window Flow execution likewise jumps from `0/N` to `N/N`, and
the CLI discards renderer provenance after writing the WAV.

This milestone adds status, timing, cancellation, and durable provenance. It
must not alter model inputs, tensor layouts, random draws, GGML graphs,
thresholds, or output samples.

## Event contract

Generation reports these stages:

1. backend initialization;
2. LeLM model loading;
3. tokenizer and conditioning preparation;
4. conditional/null prefix prefill;
5. delayed autoregressive generation;
6. completion.

Rendering reports backend initialization, token loading, Flow loading, Flow
generation, Flow release, VAE loading, VAE decoding, audio assembly, and
completion. Flow events identify the one-based current window, completed
windows, completed Euler steps, and total Euler steps. Every event carries
total and current-stage elapsed seconds.

Callbacks remain `void` so existing clients continue to compile. Cooperative
cancellation is a separate optional predicate in each configuration. It is
checked at safe boundaries: generation positions, Flow Euler steps, renderer
stages, and VAE windows. Cancellation throws the distinct public
`operation_cancelled` exception; CLIs map it to exit status 130.

## CLI contract

Both production tools accept:

- `--progress plain` (default): timestamped stage messages, percentages,
  throughput, and a smoothed ETA when measurable;
- `--progress json`: newline-delimited JSON events on stderr for supervisors
  and applications;
- `--progress none` or `--quiet`: no progress events;
- `--progress-interval SECONDS`: heartbeat/throttle interval, default one
  second; zero emits every computational event.

Long monolithic operations such as GGUF loading and a VAE graph cannot expose
internal byte/operator progress without changing lower layers. The CLI repeats
the most recent stage as a heartbeat, with increasing elapsed time, so these
operations never appear stalled. Normal result text remains on stdout and
progress remains on stderr.

SIGINT and SIGTERM set the cooperative cancellation predicate. Partially
computed tensors are discarded, and output WAV/metadata files are not written.

## WAV artifact contract

`levo-render` writes `song.wav.json` after successfully writing `song.wav`.
Schema 1 records:

- WAV filename, byte count, SHA-256, F32 stereo format, sample rate, samples
  per channel, and duration;
- token filename, tensor SHA-256, and source frame count;
- selected backend and LeVo2.cpp version;
- Flow/VAE provenance hashes and pinned source/runtime revisions;
- resolved Euler steps, CFG, seed, external-noise flag, and window count;
- backend, token/model loading, Flow, VAE, assembly, render-total, WAV-write,
  and artifact-total timings.

The existing `write_render_wav` API remains available. A new artifact writer
owns the WAV plus sidecar transaction for production clients.

## Verification gates

1. Asset-free tests prove stage ordering, monotonic counters/timing, exact
   Euler callback counts, and cancellation at a step boundary.
2. CLI-format tests parse every JSON line and cover plain/none selection and
   interval validation.
3. Artifact tests validate sidecar paths, required schema fields, WAV size and
   SHA-256, and failure cleanup.
4. Existing CPU and CUDA suites pass unchanged.
5. A short real CUDA smoke demonstrates live Euler progress and a valid
   sidecar. Existing frozen renderer parity remains unchanged.

## Persistence

Accepted gates are committed and pushed independently. Commands, timings,
results, and deviations are appended to `docs/implementation_report.md`.
