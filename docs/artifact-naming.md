# Artifact and sidecar naming

The public `ckadirt/LeVo2-GGUF` repository keeps each GGUF beside a checksum and
deterministic conversion manifest:

```text
LeVo2-v2-medium-F16.gguf
LeVo2-v2-medium-F16.gguf.sha256
LeVo2-v2-medium-F16.gguf.manifest.json
LeVo2-v2-flow-F32.gguf
LeVo2-v2-flow-F32.gguf.sha256
LeVo2-v2-flow-F32.gguf.manifest.json
LeVo2-v2-vae-F32.gguf
LeVo2-v2-vae-F32.gguf.sha256
LeVo2-v2-vae-F32.gguf.manifest.json
```

The full GGUF filename is retained as the manifest prefix so multiple
components and future precision variants cannot overwrite one another.
Publication requires exact agreement among the file bytes, checksum sidecar,
and manifest `artifact` object, followed by anonymous remote download and
verification. Existing artifacts are never replaced under a new meaning;
future precision or quantized variants receive distinct filenames and their own
parity evidence.
