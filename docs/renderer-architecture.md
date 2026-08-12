# Native Flow/VAE renderer architecture contract

## Source of truth

Behavior is pinned to these files at the revisions in `renderer-plan.md`:

- `codeclm/tokenizer/Flow1dVAE/generate_septoken.py`
- `codeclm/tokenizer/Flow1dVAE/model_septoken.py`
- `codeclm/tokenizer/Flow1dVAE/models_gpt/models/`
  `gpt2_rope2_time_new_correct_mask_noncasual_reflow.py`
- `codeclm/tokenizer/Flow1dVAE/tools/get_1dvae_large.py`
- the pinned `stable_audio_tools` Oobleck decoder implementation
- `ckpt/vae/stable_audio_1920_vae.json`

Pinned runtime inputs:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `model_septoken/model_2.safetensors` | 4,808,167,708 | `430b7c1c245722fbe3893cd621b3d4a90076404596e9fb1ce987a4a0f2a4fc6f` |
| `vae/autoencoder_music_1320k.ckpt` | 674,920,616 | `10ccb6c83613781ad32e998a90597ba7eb9292911a224598da1fd53728eb4cd3` |
| `vae/stable_audio_1920_vae.json` | 3,613 | `5cd2859efe00bc2b0f6f9bdac738ad11822a36473d6d810427b60efd057c538b` |

## Token-to-latent Flow

Only streams 1 and 2 of `[mixed, vocal, accompaniment]` are rendered. Each
stream indexes its own 16,384-entry, 32-wide codebook. A weight-normalized 1x1
projection maps each lookup to 1024 dimensions. The two results form 2048
conditioning channels per frame.

The official no-prompt path uses a 1000-frame (40-second) window. Short inputs
are repeated and cropped to 1000 frames for Flow inference; output is cropped
back to `T * 1920` waveform samples. Longer inputs use 750-frame hops with 250
frames of latent in-context overlap.

For each window, the Flow transformer input width is 2200:

```text
mask embedding                 24
in-context VAE latent          64
vocal RVQ conditioning       1024
accompaniment RVQ condition  1024
current noisy latent           64
                              ----
                              2200
```

The estimator is a noncausal GPT2-style transformer:

| Property | Value |
| --- | ---: |
| Width | 2200 |
| Blocks | 16 |
| Attention heads | 20 |
| Head width | 110 |
| MLP width | 4400 |
| Maximum positions | 1000 |
| LayerNorm epsilon | 1e-5 |
| Activation | GPT2 `gelu_new` |
| RoPE | adjacent-pair, theta 10000, Q and K |
| Attention | bidirectional/noncausal with supplied in-context mask |

Every block uses timestep-driven AdaLN-single shift, scale, and gate values for
both attention and MLP. A final timestep shift/scale is applied after the final
LayerNorm and before the 2200-wide output projection. Only the final 64
channels are the velocity prediction.

The solver starts from supplied Gaussian noise and uses 50 uniform Euler steps
over `[0,1]`. At each step, in-context positions follow the official noisy
interpolation, conditional and null batches are evaluated together, and
velocity uses `uncond + 1.5 * (cond - uncond)`. The update is `x += dt * v`.

## Latent-to-audio VAE

The VAE consumes `[batch,64,frames]` and emits `[batch,2,samples]`. Its decoder
is the Oobleck architecture configured with base width 128, channel multipliers
`[1,2,4,8,16]`, strides `[2,4,4,6,10]`, SnakeBeta activations, and no final
tanh. The stride product is 1920, so one 25 Hz latent frame produces 1920
stereo samples at 48 kHz.

Only the decoder is reachable from generated tokens. The encoder, stochastic
VAE encode path, discriminators, loss modules, and optimizer state are not part
of native rendering. PyTorch weight-normalization `weight_g`/`weight_v` values
are folded into ordinary convolution weights during deterministic conversion;
C++ does not implement mutable weight-normalization state.

The port must preserve the upstream reflection padding and output-length
cropping around odd kernels and transposed convolutions. Full output length is
an invariant checked at every VAE stage and at the final WAV boundary.

## Memory lifetime

The renderer API loads Flow and VAE separately. After a Flow window is reduced
to its 64-channel latent, Flow graph scratch can be released before the VAE
decode graph runs. Multi-window rendering retains only the overlap latent and
assembled audio needed by the official overlap policy. The implementation must
not require the LeLM model to remain loaded while rendering an existing token
artifact.
