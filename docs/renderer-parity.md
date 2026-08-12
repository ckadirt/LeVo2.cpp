# Native renderer parity policy

## Reproducibility boundary

Flow sampling begins with Gaussian noise. Oracle fixtures therefore store the
actual F32 initial noise tensor and C++ consumes that tensor directly. A common
integer seed alone is insufficient because PyTorch and the C++ runtime do not
promise the same normal-distribution algorithm.

Production C++ may expose a seed for its own repeatability. It does not claim
seed-identical audio with Python.

## Staged fixtures

Flow fixtures freeze:

1. Vocal/BGM codebook lookup and projected 2048-wide conditioning.
2. Mask/null input construction for conditional and CFG-null batches.
3. Timestep embedding and modulation at `t = 0`, `0.5`, and `0.98`.
4. Block 0 outputs, full 16-block velocity, and the final 64-channel slice.
5. One Euler update, the complete 50-step latent, and a second overlapped
   window.

VAE fixtures freeze:

1. Weight-normalized convolution kernels after folding.
2. Decoder input projection and every upsampling-stage output.
3. Selected residual/SnakeBeta outputs.
4. Full 1-frame, short, 10.08-second, and 30-second stereo waveforms.

Fixtures record shape, dtype, SHA-256, min/max/mean/RMS, and finite status. Raw
arrays and audio stay ignored; small deterministic synthetic fixtures may be
tracked.

## Threshold discipline

- Shape, tensor order, padding/cropping, token lookup, folded weights, window
  schedule, Euler timesteps, and output sample counts are exact gates.
- F32 operator thresholds are frozen in this document after the Python oracle
  exporter lands and before the corresponding C++ graph implementation.
- F16 thresholds are separate and are frozen before F16 conversion/runtime
  work. They may include latent max/RMSE/cosine and waveform SI-SDR, spectral
  convergence, log-magnitude error, correlation, peak, and RMS ratios.
- Thresholds are never relaxed silently. Any change includes before/after
  evidence, an explanation of the first divergent operator, and its own
  reviewable commit.
- Final stochastic WAV files are not required to be byte-identical across
  runtimes. Given identical stored initial noise, latent and waveform metrics
  must satisfy the frozen numerical gates.

## Required release cases

- A minimal valid token input that exercises one full 1000-frame Flow window.
- The existing 10.08-second delay-boundary token fixture.
- The existing complete sampled 30-second fixture.
- A synthetic input long enough to exercise a second 750-frame hop and
  250-frame overlap without requiring a long LeLM generation.

Each case runs with one Euler step for diagnostics and the official 50 steps for
release. Output must be stereo 48 kHz, finite, non-silent, correctly cropped to
`T * 1920` samples, and validated from a clean CUDA build. CPU tests cover
converter/loader behavior and tractable primitive graphs; the full production
Flow gate is CUDA because the 2200-wide 1000-position model is not a practical
CPU smoke test.
