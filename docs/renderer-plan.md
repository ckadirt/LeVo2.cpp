# Native renderer execution plan

## Objective

The next milestone removes Python from the decode half of the v0.1 pipeline:

```text
int32 [3,T] LeVo tokens
        |
        | vocal + accompaniment streams
        v
dual RVQ code lookup and projection
        |
        v
conditional Flow transformer + 50 Euler steps
        |
        v
float latent [1,64,T]
        |
        v
Oobleck VAE decoder
        |
        v
stereo 48 kHz WAV
```

The source and runtime revisions remain the v0.1 pins:

- LeVo source `653cbcf4716101834900c75b7d5da43b07e15d5b`.
- SongGeneration Runtime `cc258cc694a63114c61684cc26d0583b8ad777d0`.
- GGML `8846b79e66747bb9f68597420e95114c177315ce`.

The native renderer consumes the existing canonical NPY/JSON token artifact.
It does not add audio-prompt encoding, source separation, or token extraction.

## Locked decisions

- Keep Flow and VAE as separate GGUF artifacts and C++ components. Their
  architectures, precision sensitivity, memory lifetime, and future
  quantization policies are independent.
- The initial correctness path is F32. F16 artifacts are published only after
  their own latent/audio quality gates pass; precision is never inferred from
  the LeLM release.
- Add a dedicated `levo-render` tool and renderer API. The existing `levo-cli`
  token generator remains backward compatible.
- Export initial Flow noise from Python and feed the same values to C++ parity
  tests. Matching a seed across PyTorch and C++ RNG implementations is not a
  correctness requirement.
- Port only inference-reachable tensors. Best-RQ/Hubert audio encoders,
  resamplers, training-only projection heads, discriminators, and the VAE
  encoder are excluded and listed explicitly in manifests.
- Preserve the official 40-second / 1000-frame Flow window, 75% hop, 25%
  overlap, CFG 1.5, sigma-min `1e-4`, 50 uniform Euler steps, 25 Hz token rate,
  64-channel latent, 1920x VAE expansion, and earliest-token-artifact duration.
- Use official upstream GGML. A GGML fork still requires a minimal failing
  operator test and a documented upstream gap.
- Every numbered gate below gets its own commit and immediate push. A gate is
  not combined with the next one merely because both are locally complete.

## Commit gates

1. **Renderer contracts**
   - Commit this plan, the architecture contract, GGUF schema, and parity
     policy before renderer code.
   - Push `docs: define native renderer execution plan`.

2. **Oracle and inventory tooling**
   - Add deterministic Python exporters for Flow conditioning/velocity/Euler
     states and VAE decoder stages.
   - Record exact checkpoint/config hashes and reachable/omitted inventories.
   - Freeze numerical thresholds before implementing the corresponding C++
     graph.
   - Push `test: add native renderer parity oracles`.

3. **Flow conversion and strict loading**
   - Convert only dual codebooks/projections, mask/null/stat tensors, timestep
     conditioning, positional embeddings, 16 Flow blocks, final norm, and
     output projection.
   - Validate names, shapes, dtypes, dimensions, provenance, and truncation.
   - Load the staged GGUF on CPU and CUDA.
   - Push `feat: add Flow GGUF conversion and loading`.

4. **Flow conditioning**
   - Reproduce RVQ lookup plus weight-normalized output projection, zero/null
     conditioning, mask embeddings, and input concatenation.
   - Pass exact or bounded tensor parity before proceeding.
   - Push `feat: port Flow token conditioning`.

5. **Flow velocity transformer**
   - First pass one block, then all 16 noncausal GPT2-style blocks, interleaved
     RoPE, AdaLN-single modulation, final modulation, and velocity slice.
   - Commit the one-block pass separately from the full-transformer pass if
     either needs nontrivial numerical work.
   - Push no later than `feat: port Flow velocity transformer`.

6. **Euler solver and segmentation**
   - Implement CFG branches, in-context overwrite, uniform Euler integration,
     official repetition/padding of short token sequences, 1000-frame windows,
     750-frame hops, and 250-frame latent overlap.
   - Pass one-step, 50-step, and multi-window latent gates.
   - Push `feat: add native Flow renderer`.

7. **VAE conversion and strict loading**
   - Convert the decoder only, reconstructing inference weights from PyTorch
     weight-normalization parameters deterministically.
   - Validate the exact Oobleck decoder topology and the 1920x ratio.
   - Push `feat: add VAE GGUF conversion and loading`.

8. **VAE decoder graph**
   - Implement zero padding, Conv1d, ConvTranspose1d, residual units,
     SnakeBeta, stage cropping, and final stereo projection.
   - Pass per-stage and full-waveform gates in F32 before evaluating F16.
   - Push `feat: port native Oobleck VAE decoder`.

9. **Public API, WAV writer, and CLI**
   - Add strict token-artifact input, renderer configuration/progress,
     deterministic external-noise testing, float-to-PCM WAV output, and
     `levo-render`.
   - Run token-to-WAV tests on CPU where practical and CUDA for the production
     50-step path.
   - Push `feat: add native token-to-WAV rendering`.

10. **Release gate**
    - Render frozen short, 10.08-second boundary, and 30-second token fixtures.
    - Compare latent tensors and audio against the pinned Python oracle, verify
      finite/non-silent stereo 48 kHz output, audit licenses/secrets, and test
      an anonymous clean clone.
    - Publish immutable Flow/VAE artifacts with hashes and manifests, then tag
      the renderer milestone only after remote hash verification.

## Persistence and reporting

`/workspace` is still not volume-backed. Each accepted gate is committed and
pushed immediately. Model weights, parity arrays, initial noise, and WAV files
stay ignored. `docs/renderer-implementation-report.md` records the commit,
commands, metrics, hashes, memory use, and deviations for every gate.
