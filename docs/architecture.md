# LeLM v2-medium architecture contract

## Source of truth

Behavior is pinned to these upstream files at LeVo commit
`653cbcf4716101834900c75b7d5da43b07e15d5b`:

- `codeclm/models/lm_levo.py`
- `codeclm/models/levo.py`
- `codeclm/modules/conditioners.py`
- `codeclm/modules/pattern.py`
- `codeclm/models/codeclm.py`
- `generate.py`

The model checkpoint/config is `lglg666/SongGeneration-v2-medium` revision
`7d91660ebfa041e29bace194f5631e775796f600`.

## Fixed dimensions and IDs

| Property | Value |
| --- | ---: |
| Model width | 1536 |
| Main blocks | 28 |
| Detail blocks | 12 |
| Attention heads / KV heads | 12 / 12 |
| Head dimension | 128 |
| Feed-forward width | 8960 |
| RMSNorm epsilon | 1e-5 |
| Main/detail RoPE theta | 500000 / 500000 |
| Context length | 10000 |
| Audio streams | 3 |
| Codec vocabulary | 16384 |
| EOS ID | 16384 |
| Special/EOP/pad ID | 16385 |
| Audio frame rate | 25 Hz |
| Target sample rate | 48000 Hz |
| Delay pattern | `[0, 250, 250]` |

The transformer MLPs are SwiGLU even though the YAML contains an `activation`
field: that field is not forwarded into the upstream `LlamaConfig`. Only the
bridge MLP uses GELU. RoPE matches Hugging Face's half-rotation layout, i.e. the
GGML NeoX RoPE mode.

## Hierarchical forward pass

```text
mixed code IDs --> mixed embedding --> main Llama x 28 --> RMSNorm
                                             |                |
                                             |                +--> mixed LM head
                                             v
vocal embedding + BGM embedding --> concatenate normalized main hidden
                                             |
                         Linear(3072,1536) -> GELU -> Linear(1536,1536)
                                             |
                                      detail Llama x 12
                                             |
                                          RMSNorm
                                      /               \
                              vocal head           BGM head
```

`transformer2.lm_head` exists in the checkpoint but is not called by released
inference. Vocal and accompaniment logits come from `linears.0` and
`linears.1`.

## Conditioning prefix

The first streaming call prepends exactly 952 positions, in this order:

1. Lyrics: 600 positions.
2. Prompt-audio conditioner: 252 positions.
3. Style/type description: 100 positions.

Lyrics and style are tokenized as the literal string `<|im_start|>` followed by
the supplied text. Lyrics use the 13 added structure tokens from
`conf/vocab.yaml`. A learned structure embedding is added from each structure
tag through the next structure boundary. Style uses a different learned token
embedding and no structure embedding.

v0.1 has no audio-prompt input, but it must still construct the learned null
prompt condition. Its three 250-frame streams are filled with 16385. Upstream
temporarily prepends EOS, then its all-special first-frame mask overwrites that
EOS and the complete stream with 16385; the learned EOT embeddings are prepended
separately. The CFG-null branch still has the same fixed 952 positions, formed
from empty text, all-special prompt audio, and empty style text.

Condition padding masks are calculated upstream but not passed to either Llama
tower. All fixed prefix positions therefore participate in causal attention and
must not be removed or masked in C++.

## Streaming and KV caches

Classifier-free guidance conceptually evaluates two branches:

1. Fully conditional prefix.
2. Fully null-conditioned prefix.

C++ uses one KV session per branch. Each session has independent main/detail
per-layer K/V caches; together they are equivalent to upstream's batch of two.
Positions are continuous across the 952 condition positions and the delayed
audio pattern. The first model call processes the full condition prefix plus
the initial all-special sequence slot. Later calls process one delayed sequence
position at a time.

For `T = floor(duration_seconds * 25)`, the delayed pattern contains `T + 251`
sequence positions including the initial empty slot. At the maximum 270 seconds,
the total transformer length is 7953 and remains below the 10000-position model
limit.

## Delayed stream semantics

At sequence slot `s`:

- Slot 0 contains only special IDs.
- Mixed token `M[t]` is generated at `s = t + 1`.
- Vocal and accompaniment tokens `V[t]` and `B[t]` are generated together at
  `s = t + 251`.
- All unavailable stream positions contain special ID 16385.

The returned tensor is stream-major `[3,T]`: mixed, vocal, accompaniment.
Generation tracks EOS independently for each stream and then trims the returned
tensor before the earliest EOS, matching the Python `CodecLM` wrapper.

## Sampling

Default logits use `uncond + 1.5 * (cond - uncond)`. Mixed tokens use
temperature 0.9 and top-k 50. Vocal and accompaniment use top-k 1. The previous
50 sampled delayed steps receive the upstream unique-token repetition divisor
of 1.1. With no audio prompt, the mixed ignore-token list is empty.
