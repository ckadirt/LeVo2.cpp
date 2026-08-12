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
