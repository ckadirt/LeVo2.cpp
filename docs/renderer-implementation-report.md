# Native renderer implementation report

## Status

Planning and source audit are in progress. Native Flow and VAE inference are
not yet implemented. The v0.1 official Python decoder remains the working WAV
path until every release gate in `renderer-plan.md` passes.

## Baseline

- Starting source release: `v0.1.0` at commit
  `c750fcbad144a67421fe92cba7431c364f8451d6`.
- Test machine: NVIDIA GeForce RTX 4090, CUDA 12.4.
- Pinned Flow checkpoint: 4,808,167,708 bytes, SHA-256
  `430b7c1c245722fbe3893cd621b3d4a90076404596e9fb1ce987a4a0f2a4fc6f`.
- Pinned VAE checkpoint: 674,920,616 bytes, SHA-256
  `10ccb6c83613781ad32e998a90597ba7eb9292911a224598da1fd53728eb4cd3`.
- Pinned VAE config SHA-256:
  `5cd2859efe00bc2b0f6f9bdac738ad11822a36473d6d810427b60efd057c538b`.

The upstream checkpoint includes large training/encoding subgraphs that the
token-to-WAV path never calls. The converter will classify every source tensor
before runtime code is accepted, following the same strict-inventory discipline
as the LeLM converter.

## Source audit

The initial read-only source audit fixed the native boundary before converter
work:

- Flow: 993 source tensors; approximately 2.471 GiB is reachable from token
  rendering after training/audio-encoder omissions.
- VAE: 365 checkpoint tensors, of which 182 belong to the decoder; folding 37
  weight-normalized convolutions produces 145 runtime tensors.
- VAE convolutions are zero padded. CUDA transpose-convolution padding will be
  represented as a zero-padding operation followed by a symmetric crop.
- Flow F32 correctness and official CUDA FP16-autocast compatibility are
  explicitly distinct parity modes.

## Landed checkpoints

| Commit | Gate | Result |
| --- | --- | --- |
| `8bfedea` | Renderer plan | Architecture, GGUF, parity, and execution plan |
| `7e04f44` | Source contract | Exact Flow/VAE topology and inference inventory |
| `f70f2af` | Flow converter | 993 source tensors classified; 231 emitted |
| `78436bb` | VAE converter | 365 source tensors classified; 145 emitted |
| `0b0b237` | Python oracles | Deterministic Flow/VAE intermediate capture |
| `8900054` | GGML primitives | F32 Conv1d, padded ConvTranspose1d, SnakeBeta |

The strict Flow converter produced `LeVo2-v2-flow-F32.gguf` with 231 tensors,
663,310,785 parameters, 2,653,259,456 bytes, and SHA-256
`a8cf50dbecef243501b9b345109b1d2f283b3e22f4e4856715197e4b22129d10`.

The strict VAE converter produced `LeVo2-v2-vae-F32.gguf` with 145 tensors,
84,395,776 parameters, 337,596,448 bytes, and SHA-256
`26f9ea955f586ed3d7668fe345a851ba222b8db95b406e3eea3c9565f4a0b515`.
These hashes identify the current local F32 correctness artifacts; public
publication remains gated on the native graph parity tests.

Validated commands include:

```bash
LEVO_FLOW_CHECKPOINT=/workspace/models/SongGeneration-Runtime/ckpt/model_septoken/model_2.safetensors \
  /venv/main/bin/python -m pytest -q tests/python/test_flow_converter.py
/venv/main/bin/python -m pytest -q tests/python/test_vae_converter.py
LEVO_RUN_RENDERER_ORACLE_TEST=1 \
  /venv/main/bin/python -m pytest -q tests/python/test_renderer_oracles.py
ctest --test-dir build-cpu -R '^audio-ops$' --output-on-failure
ctest --test-dir build-cuda -R 'audio-ops' --output-on-failure
```

At this checkpoint the focused results are Flow converter 4/4, VAE converter
4/4, renderer oracles 3/3 (including the official T=1 VAE), and native audio
operators 1/1 CPU plus 2/2 in the CUDA build.
