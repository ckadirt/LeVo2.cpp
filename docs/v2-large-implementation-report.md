# LeVo 2 v2-large implementation report

## Status

In progress. This report is the live evidence record for the v2-large LeLM
extension. It supplements the existing v2-medium and quantization reports; it
does not rewrite their historical validation.

## Operating method

1. Commit and push source/documentation milestones before model publication.
2. Treat v2-large as a distinct, strict LeLM source profile.
3. Use CUDA for every real large-model generation and render gate. No
   large-model CPU generation or render is claimed.
4. Publish a GGUF only after its matching source commit, strict loader,
   checksum/manifest, CUDA evidence, and model-card entry exist.
5. Record every deviation instead of silently broadening a contract.

## Milestone log

### 2026-08-13 — contract established

- Pinned `lglg666/SongGeneration-v2-large` revision
  `115805364ad74479fb3764fe65970c92faeb1a5a`, its 12,899,965,446-byte
  `model.pt` SHA-256
  `dc763aa9a76a22a87597c2faf9a51c24d13349ac754699b37e9068b483639def`,
  and the 3,352-byte `config.yaml` SHA-256
  `14a991bd7342b9dde348e6324afd44b5c6ecb1db8d0ed4d2dbe666b220b04c59`.
- Verified the v2-large configuration changes LeLM dimensions only
  (`2048/11008`, 36 main blocks, 16 heads) while retaining v2-medium's token
  contract, Flow source, Stable Audio 1920 VAE, 25 Hz frame rate, 48 kHz
  output, and `[0,250,250]` hierarchy delay.
- Recorded the full approved scope and explicit no-duplicate-renderer rule in
  [`v2-large-plan.md`](v2-large-plan.md). Pending: profile-based converter and
  strict loader implementation, followed by CUDA-only validation.
