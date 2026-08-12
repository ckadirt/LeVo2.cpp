# LeVo2 GGUF contract

## Goals

The GGUF must be a self-contained, mmap-friendly inference artifact. It carries
all runtime-reachable LeLM weights, Qwen2 tokenizer data, added lyric tokens,
architecture constants, source provenance, and the conversion version. External
files are required only by the separate Python audio decoder.

GGUF format behavior follows the pinned official GGML documentation and APIs.
The architecture identifier is `levo2` and the initial schema version is `1`.

## Required metadata

The schema-1 spellings and types are frozen as follows:

| Key | Type | Meaning |
| --- | --- | --- |
| `general.architecture` | string | `levo2` |
| `general.name` | string | Model display name |
| `general.file_type` | uint32 | `0` F32 or `1` F16 |
| `general.license`, `general.license.link` | string | Upstream use terms and URL |
| `general.source.repo_url` | string | Source model URL |
| `levo2.schema_version` | uint32 | `1` |
| `levo2.converter`, `levo2.converter.version` | string | Converter identity/version |
| `levo2.source.model_repository`, `levo2.source.model_revision` | string | Pinned checkpoint source |
| `levo2.source.runtime_repository`, `levo2.source.runtime_revision` | string | Pinned Python decoder/runtime source |
| `levo2.source.levo_repository`, `levo2.source.levo_revision` | string | Pinned Python implementation |
| `levo2.source.ggml_repository`, `levo2.source.ggml_revision` | string | Pinned GGML implementation |
| `levo2.source.model_sha256`, `levo2.source.config_sha256` | string | Source object hashes |
| `levo2.tokenizer.assets_sha256.json` | string | Canonical JSON asset/hash map |
| `levo2.tokenizer.revision`, `levo2.tokenizer.sha256` | string | Pinned tokenizer revision and primary `tokenizer.json` hash |

Architecture metadata uses the following `levo2.*` keys:

- Main/detail block counts, width, feed-forward width, head counts, context
  length, RMS epsilon, and both RoPE bases: `main.block_count`,
  `detail.block_count`, `embedding_length`, `feed_forward_length`,
  `attention.head_count`, `attention.kv_head_count`, `context_length`,
  `rms_norm_epsilon`, `main.rope_theta`, and `detail.rope_theta`.
- Codebook count/size, EOS and special IDs, audio frame rate, sample rate, and
  delay array: `codebook.count`, `codebook.size`, `token.eos_id`,
  `token.special_id`, `audio.frame_rate`, `audio.sample_rate`, and
  `pattern.delays`.
- Lyrics, prompt-audio, and style prefix lengths.
  These are `condition.lyrics_prefix_length`,
  `condition.prompt_prefix_length`, and `condition.style_prefix_length`.

Qwen2 byte-level BPE is embedded through the standard GGUF tokenizer model,
token, and merge fields. Exact upstream JSON is additionally stored in
`levo2.tokenizer.json`, `levo2.tokenizer.added_tokens.json`,
`levo2.tokenizer.special_tokens.json`, and `levo2.tokenizer.config.json`; the
token count is `levo2.tokenizer.vocab_size`. The loader rejects unsupported
schema versions rather than guessing defaults.

## Tensor groups

The conversion map contains exactly 380 tensors and covers:

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

Layer tensors are named `main.blk.N.*` and `detail.blk.N.*`, with suffixes
`attn_norm.weight`, `attn_q.weight`, `attn_k.weight`, `attn_v.weight`,
`attn_output.weight`, `ffn_norm`, `ffn_gate.weight`, `ffn_up.weight`, and
`ffn_down.weight`. Final norms are `main.output_norm` and
`detail.output_norm`.

The converter keeps an explicit allowlist of source keys. Six
runtime-inaccessible tensors are listed in the manifest and omitted:
`layer2_emb.0.weight`, both Hugging Face input embeddings, detail `lm_head`,
and `out_norm.{weight,bias}`. A newly observed `audiolm.*` key is an error until
classified. The released style embedding's six unreachable tail rows are
preserved and documented instead of silently truncated.

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
