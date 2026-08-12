# LeVo2.cpp LeLM v0.1 execution plan

## Objective

Implement and publish the first LeVo2.cpp milestone:

```text
lyrics + style description
          |
          v
      LeLM in C++/GGML
          |
          v
mixed + vocal + accompaniment tokens
          |
          v
tokens.npy + tokens.json
          |
          v
official Python Flow1dVAESeparate decoder
          |
          v
48 kHz stereo audio
```

The implementation root is `/workspace/LeVo2.cpp`. The compatibility target is
`lglg666/SongGeneration-v2-medium` only. The source oracle is
`levo-demo/LeVo` at commit
`653cbcf4716101834900c75b7d5da43b07e15d5b`. GGML starts from official
upstream commit `8846b79e66747bb9f68597420e95114c177315ce`.

## Locked decisions

- Release F16 first. F32 is supported for local parity work; quantization is
  deferred until after v0.1.
- Require bounded numerical logit parity and exact greedy-token parity.
  Sampled mode must be internally reproducible and distributionally correct,
  but does not promise PyTorch-identical random samples.
- Publish verified model artifacts to public `ckadirt/LeVo2-GGUF`.
- Push every validated source milestone to `origin/main`.
- Keep the source repository private during development and make it public only
  after the v0.1 release gates pass.
- Preserve the upstream academic/research/education-only restriction for LeVo
  inference code and weights. Do not describe the project or converted weights
  as commercially licensed.
- v0.1 accepts lyrics and an optional comma-separated style description. Audio
  prompt encoding, multi-sample batching, native flow/VAE rendering,
  quantization, a stable C ABI, Android, JNI, and Cantor are out of scope.

## Milestones and commit gates

1. **Documentation baseline**
   - Commit the plan, architecture, GGUF contract, parity policy, release
     procedure, and implementation report before code.
   - Push `docs: define LeLM v0.1 execution plan`.

2. **Build foundation**
   - Add official GGML as a pinned submodule.
   - Add C++17 `levo-core`, `levo-cli`, and test targets.
   - Prove CPU and CUDA backend execution; use CUDA architecture 89 locally.
   - Push only after clean release builds and smoke tests pass.

3. **Converter and loader**
   - Implement deterministic F32/F16 conversion of the trusted v2-medium
     checkpoint.
   - Embed the tokenizer, all inference metadata, and only runtime-reachable
     LeLM tensors.
   - Add strict metadata/tensor validation and mmap-capable GGUF loading.
   - Record source hashes, converter revision, tensor inventory, and output
     SHA-256 in a manifest.

4. **Transformer parity**
   - Reproduce the 28-layer mixed tower, 12-layer detail tower, bridge MLP, and
     three output heads.
   - Validate every block, the normalized hidden states, bridge output, and all
     logits against deterministic Python oracle fixtures.
   - Add separate KV caches for both towers and both CFG batch members; cached
     and uncached forward paths must agree.

5. **Conditioning and generation**
   - Port the exact Qwen2 tokenizer and learned fixed-length conditioners.
   - Reproduce the `[0, 250, 250]` delayed pattern, CFG, repetition behavior,
     EOS handling, and stream-specific sampling.
   - Expose the C++ API and CLI and write the canonical NPY/JSON interchange.

6. **Decoder integration and release**
   - Obtain exact greedy token parity for short, delay-boundary, and 30-second
     cases.
   - Render a sampled 30-second result with the official Python decoder and
     validate finite, non-silent 48 kHz stereo output.
   - Verify clean-clone builds, licenses, secrets, hashes, and documentation.
   - Publish the F16 GGUF and manifest to Hugging Face, make the source public,
     tag `v0.1.0`, and create the source release.

## Work and persistence policy

`/workspace` is not backed by a persistent volume. Every accepted milestone is
therefore committed and pushed immediately. Downloaded checkpoints and build
outputs are reproducible and remain ignored. Credentials are loaded from
`/workspace/.env` only for the command that needs them; their values must never
appear in files, process arguments, logs, commits, or reports.

`docs/implementation_report.md` is updated in every milestone commit with
commands, results, commit IDs, artifact hashes, current limitations, and all
deviations from this plan. A deviation records the evidence and its
compatibility consequences, not merely the changed implementation.

## GGML fork fallback

Official GGML is the default. `/workspace/ggml-fork` is currently synchronized
to the pinned official commit. A fork change is allowed only after a minimal
test or benchmark demonstrates a missing or materially inadequate operation.
The change is developed and tested on a dedicated fork branch, pushed to
`ckadirt/ggml-fork`, documented in the implementation report, and only then
pinned by LeVo2.cpp. Model-specific graph construction alone is not a reason to
fork GGML.

## Post-v0.1 roadmap

After v0.1, evaluate Q8_0, Q6_K, Q5_K_M, and Q4_K_M separately on both towers.
Native flow rendering, the VAE, stable C ABI, JNI, and Android/Cantor integration
are independent future design and implementation milestones.
