# Native renderer GGUF contract

## Artifacts

The renderer uses two independently checksummed files:

- `LeVo2-v2-flow-F32.gguf` initially, then `LeVo2-v2-flow-F16.gguf` after its
  precision gate.
- `LeVo2-v2-vae-F32.gguf` initially, then `LeVo2-v2-vae-F16.gguf` after its
  waveform gate.

Each artifact has its own `.manifest.json` and `.sha256` companion. Conversion
is deterministic and rejects any source checkpoint outside the pinned tensor
inventory.

## Flow metadata and names

`general.architecture` is `levo2_flow`. Required metadata uses the
`levo2.flow.*` namespace and records architecture
dimensions, CFG, sigma minimum, Euler steps, token/latent frame rates, window,
hop, overlap, codebook size, source/runtime revisions, every input hash,
converter version, tensor count, parameter count, and conversion dtype.

Canonical tensor families are:

```text
flow.rvq.{vocal,bgm}.codebook.weight
flow.rvq.{vocal,bgm}.out_proj.{weight,bias}
flow.mask_embedding.weight
flow.null_condition.weight
flow.norm.{counts,sum_x,sum_x2}
flow.position_embedding.weight
flow.time_embedding.linear_{1,2}.{weight,bias}
flow.time_modulation.{weight,bias}
flow.block.N.norm_{1,2}.{weight,bias}
flow.block.N.attn.{qkv,out}.{weight,bias}
flow.block.N.ffn.{in,out}.{weight,bias}
flow.block.N.modulation.weight
flow.final_norm.{weight,bias}
flow.final_modulation.weight
flow.output.{weight,bias}
```

The manifest explicitly omits the Best-RQ and Hubert encoders, resampling
kernels, input token embedding, training-only hidden projection MLP, codebook
encoder projections/stale counters, and any unused checkpoint tensors.

## VAE metadata and names

`general.architecture` is `levo2_vae`. Required metadata uses `levo2.vae.*`
and records the decoder channel plan,
strides, latent width, stereo output width, 1920x ratio, sample rate, SnakeBeta
mode, source/runtime revisions, config/checkpoint hashes, tensor inventory, and
conversion dtype.

Canonical tensor families are:

```text
vae.decoder.input.{weight,bias}
vae.decoder.stage.N.residual.M.{snake1,conv1,snake2,conv2}.*
vae.decoder.stage.N.upsample.{weight,bias}
vae.decoder.output.{snake,conv}.*
```

Exact numeric indices are resolved from the instantiated pinned decoder and
frozen in the converter tests. Converted convolution weights are already
weight-normalized. Snake parameters end in `.alpha_log` and `.beta_log` to make
their stored domain explicit. The C++ loader rejects leftover `weight_g` or
`weight_v` tensors.

## Quantized Flow layout

The native quantizer may produce Q8_0, Q6_K, Q5_K_M, and Q4_K_M Flow files.
Only transformer block dense matrices are quantized; controls and output remain
F32. Because the logical 2200/4400 input dimensions do not align with GGML's
quantization blocks, those matrix input axes are physically padded: Q8_0 to
2208/4416 and K profiles to 2304/4608. The estimator zero-pads activation
inputs only at those matrices, then returns the original logical dimensions.
This layout is tagged and strictly validated; consumers must not infer it from
a filename.

## Loader policy

Both loaders are schema-strict: unknown/missing metadata, unknown/missing
tensors, wrong rank/shape/type, incompatible F32/F16 mixes, invalid offsets,
and truncated data are errors. The manifest checksum is provenance; the loader
still validates the GGUF itself and never trusts the sidecar for memory safety.
