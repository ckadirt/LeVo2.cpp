#pragma once

#include "ggml.h"

namespace levo::detail {

// F32 convolution lowering used by the renderer correctness path. GGML's
// convenience conv_1d helper intentionally lowers most weights through an F16
// im2col buffer, which is not precise enough for the frozen F32 VAE oracle.
ggml_tensor * conv_1d_f32(ggml_context * context,
                          ggml_tensor * weight,
                          ggml_tensor * input,
                          ggml_tensor * bias,
                          int stride,
                          int padding,
                          int dilation);

// PyTorch ConvTranspose1d padding is represented as GGML's native p=0
// operation followed by a symmetric view/copy crop. The current renderer uses
// batch size one for VAE decode, matching the native GGML operator contract.
ggml_tensor * conv_transpose_1d_f32(ggml_context * context,
                                    ggml_tensor * weight,
                                    ggml_tensor * input,
                                    ggml_tensor * bias,
                                    int stride,
                                    int padding);

// Oobleck SnakeBeta. alpha_log and beta_log have shape [1, channels, 1]
// in GGML order and remain in the log domain in the artifact.
ggml_tensor * snake_beta(ggml_context * context,
                         ggml_tensor * input,
                         ggml_tensor * alpha_log,
                         ggml_tensor * beta_log);

} // namespace levo::detail
