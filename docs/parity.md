# Parity and acceptance policy

## Oracle

The Python oracle uses the pinned official LeVo source, v2-medium checkpoint,
PyTorch 2.6, and Transformers 4.37.2. Oracle scripts run with evaluation mode,
no gradient tracking, fixed inputs, and explicitly selected dtypes/devices.

Large oracle tensors and checkpoints are not committed. The repository contains
small deterministic synthetic fixtures. Real-model integration outputs live in
ignored artifact directories; compact JSON metric summaries and their input
hashes are committed to the implementation report.

## Numerical gates

- F32 CPU block outputs: `atol=1e-4`, `rtol=1e-4`.
- F16 CUDA hidden states and logits: cosine similarity at least `0.9999` and
  normalized RMSE at most `1e-2`.
- Final F16 logit maximum absolute error: at most `0.1`.
- Greedy argmax: exact token equality for every tested stream and generation
  step.

Failures are investigated at the earliest divergent layer. Tolerances are not
silently widened. Any justified change is recorded as a plan deviation with the
before/after metrics.

## Required test layers

1. Deterministic GGUF conversion, metadata round-trip, mmap load, and malformed
   input rejection.
2. Qwen2 token IDs, special tokens, padding, truncation, structure coverage,
   and conditional/null embeddings.
3. Main and detail transformer block outputs after every layer.
4. Final norms, mixed logits, bridge MLP, vocal logits, and accompaniment logits.
5. Cached versus uncached execution for prefill and incremental steps.
6. Conditional and null branches before and after CFG.
7. Pattern build/revert behavior for toy delays and the real `[0,250,250]`
   boundary.
8. Repetition penalty, top-k, greedy sampling, stable seeded sampling, EOS, and
   invalid-token rejection.
9. NPY/JSON writing and Python round-trip.

## End-to-end gates

- Exact Python/C++ greedy token equality for a short input, a case crossing the
  250-frame delay, and a 30-second input.
- Sampled pre-sampling probability agreement at selected steps.
- One complete sampled 30-second token generation on the RTX 4090.
- Official Python decoder output is stereo, 48 kHz, finite, non-silent, and has
  the expected approximate duration.
- Maximum-duration execution remains within the model context and fails cleanly
  if memory is insufficient; it must never emit a corrupt token file.

CI runs CPU builds and synthetic fixtures without model credentials. Real-model
and CUDA gates run locally on the provisioned RTX 4090 before release.

The frozen real-model fixture is lyrics `[verse] Hello world.` and style
`female, pop`. It is exact at all three release lengths: 2.0 seconds (`[3,50]`),
10.08 seconds (`[3,252]`, crossing the delayed-stream boundary), and 30 seconds
(`[3,750]`). The raw tensor SHA-256 values are, respectively:

```text
f57268812d4befe556a2b8ee54afae70b657c997616f3b959d7fc5add7ef737a
6f83f15e1ad5815ff215e00ef7c016bf858432a681127f0b9ee28178848dfc98
95adbd38aeee3188f9a2bd1a770eba3587af7af2ef7fbc7103644a53dda466b0
```

For numerical parity, C++ pre-fills the condition prefix into the KV cache and
then decodes the initial BOS slot separately. This is a deliberate numerical
strategy: upstream combines prefix and BOS in its first transformer call, but
the split path preserves the same mathematical sequence while matching the
frozen PyTorch greedy oracle through the long generation. CUDA uses FP32 cuBLAS
accumulation for F16 GEMMs; F16 storage and activation boundaries are retained.

An exploratory multilingual 10.08-second prompt had exactly matching
conditioning tensors and tokenizer IDs but diverged at near-tied greedy logits
(mixed frame 64, vocal frame 0, accompaniment frame 3). This is recorded as a
known cross-library floating-point limitation; release thresholds and the frozen
fixture above are unchanged.

## CUDA accumulation contract

The F16 checkpoint retains F16 storage and activation boundaries, but its
matrix multiplications must accumulate in FP32 to reproduce the pinned PyTorch
oracle. The public generator sets GGML's upstream
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` mode on CUDA when the caller has not already
set it. An explicit environment value remains an advanced override and may not
satisfy the release parity gates.

The tokenizer parity gate also covers the vendored llama.cpp Qwen2 regex splitter
and utf8proc NFC normalization. Ten representative cases match the pinned
Transformers tokenizer exactly.
