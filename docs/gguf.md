# LeVo2 GGUF contract

## Goals

The GGUF must be a self-contained, mmap-friendly inference artifact. It carries
all runtime-reachable LeLM weights, Qwen2 tokenizer data, added lyric tokens,
architecture constants, source provenance, and the conversion version. External
files are required only by the separate Python audio decoder.

GGUF format behavior follows the pinned official GGML documentation and APIs.
The architecture identifier is `levo2` and the initial schema version is `1`.

## Required metadata

Standard metadata includes `general.architecture`, `general.name`,
`general.file_type`, source repository/revision information, license link, and
converter identity. LeVo-specific metadata includes:

- Main/detail block counts, width, feed-forward width, head counts, context
  length, RMS epsilon, and both RoPE bases.
- Codebook count/size, EOS and special IDs, audio frame rate, sample rate, and
  delay array.
- Lyrics, prompt-audio, and style prefix lengths.
- Tokenizer type, tokens, merges, added tokens, special IDs, tokenizer JSON, and
  tokenizer configuration needed to reproduce Qwen2 byte-level BPE.
- Source checkpoint object hash and config hash.

The exact key spelling and scalar types are frozen in this document when the
converter lands. The loader rejects unsupported schema versions rather than
guessing defaults.

## Tensor groups

The conversion map covers:

- Mixed input embedding and mixed LM head.
- Main tower Q/K/V/O projections, both RMSNorms, gate/up/down projections, and
  final norm.
- Vocal/BGM detail embeddings.
- Bridge MLP weights and biases.
- Detail tower projections, norms, MLPs, and final norm.
- Vocal and accompaniment output heads.
- Lyrics token embedding and structure embedding.
- Style/type token embedding.
- Null prompt-audio stream embeddings and learned main/detail EOT embeddings.

The converter keeps an explicit allowlist of source keys. Runtime-inaccessible
tensors, including the unused detail `lm_head`, are listed in the manifest and
omitted. A newly observed `audiolm.*` key is an error until classified.

## Dtypes and layout

- F32 is a local parity/debug output.
- F16 is the only v0.1 release output and mirrors the upstream inference-time
  cast of all LeLM parameters.
- Tensor dimensions are written in GGML order and validated against the source
  shape map.
- No quantized tensor is emitted before v0.1.

The loader checks metadata, tensor names, dimensions, element counts, type,
alignment, offsets, and file bounds before creating backend allocations. Invalid,
truncated, duplicate, unsupported, or mismatched inputs fail with a diagnostic.

## Artifact manifest

Every real conversion emits a JSON manifest and `.sha256` sidecar containing:

- Source model repository, revision, filename, byte size, and SHA-256.
- LeVo source/config/tokenizer revisions and hashes.
- Converter and GGML commit IDs.
- GGUF schema/file type, tensor count, total parameter count, and exclusions.
- Artifact filename, byte size, and SHA-256.
- Conversion command and dependency versions without credentials or local
  secrets.
