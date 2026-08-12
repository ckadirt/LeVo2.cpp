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

Required metadata uses the `levo2.flow.*` namespace and records architecture
dimensions, CFG, sigma minimum, Euler steps, token/latent frame rates, window,
hop, overlap, codebook size, source/runtime revisions, every input hash,
converter version, tensor count, parameter count, and conversion dtype.

Canonical tensor families are:

```text
flow.codebook.{vocal,bgm}.weight
flow.codebook.{vocal,bgm}.output.{weight,bias}
flow.condition.null
flow.condition.mask.weight
flow.latent_stats.{count,sum,sum_sq}
flow.position.weight
flow.time.in.{weight,bias}
flow.time.out.{weight,bias}
flow.time.modulation.{weight,bias}
flow.blk.N.{attn_norm,attn_qkv,attn_output,ffn_norm,ffn_up,ffn_down}.*
flow.blk.N.modulation
flow.output_norm.{weight,bias}
flow.output_modulation
flow.output.{weight,bias}
```

The manifest explicitly omits the Best-RQ and Hubert encoders, resampling
kernels, input token embedding, training-only hidden projection MLP, codebook
encoder projections/stale counters, and any unused checkpoint tensors.

## VAE metadata and names

Required metadata uses `levo2.vae.*` and records the decoder channel plan,
strides, latent width, stereo output width, 1920x ratio, sample rate, SnakeBeta
mode, source/runtime revisions, config/checkpoint hashes, tensor inventory, and
conversion dtype.

Canonical tensor families are:

```text
vae.decoder.input.{weight,bias}
vae.decoder.stage.N.residual.M.{snake1,conv1,snake2,conv2}.*
vae.decoder.stage.N.upsample.{weight,bias}
vae.decoder.output.snake.*
vae.decoder.output.conv.{weight,bias}
```

Exact numeric indices are resolved from the instantiated pinned decoder and
frozen in the converter tests. Converted convolution weights are already
weight-normalized. The C++ loader rejects leftover `weight_g` or `weight_v`
tensors.

## Loader policy

Both loaders are schema-strict: unknown/missing metadata, unknown/missing
tensors, wrong rank/shape/type, incompatible F32/F16 mixes, invalid offsets,
and truncated data are errors. The manifest checksum is provenance; the loader
still validates the GGUF itself and never trusts the sidecar for memory safety.
