# Artifact and sidecar naming

The public `ckadirt/LeVo2-GGUF` repository keeps each GGUF beside a checksum and
deterministic conversion manifest:

```text
LeVo2-v2-medium-F16.gguf
LeVo2-v2-medium-F16.gguf.sha256
LeVo2-v2-medium-F16.gguf.manifest.json
LeVo2-v2-medium-Q8_0.gguf
LeVo2-v2-medium-Q8_0.gguf.sha256
LeVo2-v2-medium-Q8_0.gguf.manifest.json
LeVo2-v2-medium-Q6_K.gguf
LeVo2-v2-medium-Q6_K.gguf.sha256
LeVo2-v2-medium-Q6_K.gguf.manifest.json
LeVo2-v2-medium-Q5_K_M.gguf
LeVo2-v2-medium-Q5_K_M.gguf.sha256
LeVo2-v2-medium-Q5_K_M.gguf.manifest.json
LeVo2-v2-medium-Q4_K_M.gguf
LeVo2-v2-medium-Q4_K_M.gguf.sha256
LeVo2-v2-medium-Q4_K_M.gguf.manifest.json
LeVo2-v2-flow-F32.gguf
LeVo2-v2-flow-F32.gguf.sha256
LeVo2-v2-flow-F32.gguf.manifest.json
LeVo2-v2-flow-Q8_0.gguf
LeVo2-v2-flow-Q8_0.gguf.sha256
LeVo2-v2-flow-Q8_0.gguf.manifest.json
LeVo2-v2-flow-Q6_K.gguf
LeVo2-v2-flow-Q6_K.gguf.sha256
LeVo2-v2-flow-Q6_K.gguf.manifest.json
LeVo2-v2-flow-Q5_K_M.gguf
LeVo2-v2-flow-Q5_K_M.gguf.sha256
LeVo2-v2-flow-Q5_K_M.gguf.manifest.json
LeVo2-v2-flow-Q4_K_M.gguf
LeVo2-v2-flow-Q4_K_M.gguf.sha256
LeVo2-v2-flow-Q4_K_M.gguf.manifest.json
LeVo2-v2-vae-F32.gguf
LeVo2-v2-vae-F32.gguf.sha256
LeVo2-v2-vae-F32.gguf.manifest.json
LeVo2-v2-vae-F16.gguf
LeVo2-v2-vae-F16.gguf.sha256
LeVo2-v2-vae-F16.gguf.manifest.json
```

The full GGUF filename is retained as the manifest prefix so multiple
components and future precision variants cannot overwrite one another.
Publication requires exact agreement among the file bytes, checksum sidecar,
and manifest `artifact` object. After upload, download the public sidecar and
manifest without credentials and compare them with the public Hub LFS object's
reported SHA-256 and byte count; all four identities must match. Existing
artifacts are never replaced under a new meaning; future precision or
quantized variants receive distinct filenames and their own parity evidence.

## GitHub binary archives

GitHub Release attachments use the release tag and an explicit platform/backend
suffix. The initial supported matrix is Linux x86_64 CPU and CUDA 12:

```text
LeVo2-v0.2.0-linux-x86_64-cpu.tar.gz
LeVo2-v0.2.0-linux-x86_64-cpu.tar.gz.sha256
LeVo2-v0.2.0-linux-x86_64-cuda12.tar.gz
LeVo2-v0.2.0-linux-x86_64-cuda12.tar.gz.sha256
```

Each archive expands to a single same-named directory containing:

```text
bin/levo-cli
bin/levo-render
bin/levo-cantor
bin/levo-quantize
bin/libggml-*.so*             # dynamically discovered backend modules
lib/liblevo-cantor-engine.so*
lib/libggml.so*
lib/libggml-base.so*
LICENSE
THIRD_PARTY_NOTICES.md
README.md
```

The executables and shared objects have relative rpaths. Backend modules stay
beside the executables because GGML's dynamic-loader discovery scans that
directory; no `LD_LIBRARY_PATH` setting is required. Archives never contain
model weights: users fetch the separately licensed GGUF components from the
model release named in `docs/releasing.md`.

The workflow attaches each archive and its SHA-256 sidecar to an existing
GitHub Release. It fails rather than replacing either name, including on a
rerun. A new byte sequence therefore requires a new tag and archive name.
