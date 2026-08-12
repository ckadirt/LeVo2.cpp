#include "levo-audio-ops.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace levo::detail {
namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::invalid_argument("LeVo audio graph: " + message);
}

void require_f32(const ggml_tensor * tensor, const char * name) {
    if (tensor == nullptr) {
        fail(std::string(name) + " is null");
    }
    if (tensor->type != GGML_TYPE_F32) {
        fail(std::string(name) + " must be F32");
    }
}

ggml_tensor * add_channel_bias(ggml_context * context,
                               ggml_tensor * input,
                               ggml_tensor * bias) {
    if (bias == nullptr) {
        return input;
    }
    require_f32(bias, "bias");
    if (bias->ne[0] != input->ne[1] || bias->ne[1] != 1 ||
        bias->ne[2] != 1 || bias->ne[3] != 1) {
        fail("bias must contain exactly one value per output channel");
    }
    ggml_tensor * shaped = ggml_reshape_3d(context, bias, 1, input->ne[1], 1);
    return ggml_add(context, input, shaped);
}

} // namespace

ggml_tensor * conv_1d_f32(ggml_context * context,
                          ggml_tensor * weight,
                          ggml_tensor * input,
                          ggml_tensor * bias,
                          int stride,
                          int padding,
                          int dilation) {
    if (context == nullptr) {
        fail("context is null");
    }
    require_f32(weight, "convolution weight");
    require_f32(input, "convolution input");
    if (stride <= 0 || padding < 0 || dilation <= 0) {
        fail("invalid Conv1d stride, padding, or dilation");
    }
    if (weight->ne[0] <= 0 || weight->ne[1] != input->ne[1] ||
        weight->ne[2] <= 0 || weight->ne[3] != 1 || input->ne[3] != 1) {
        fail("incompatible Conv1d weight/input shapes");
    }

    ggml_tensor * columns = ggml_im2col(context, weight, input,
                                        stride, 0, padding, 0,
                                        dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * columns_2d = ggml_reshape_2d(
        context, columns, columns->ne[0], columns->ne[1] * columns->ne[2]);
    ggml_tensor * weight_2d = ggml_reshape_2d(
        context, weight, weight->ne[0] * weight->ne[1], weight->ne[2]);
    ggml_tensor * output = ggml_mul_mat(context, columns_2d, weight_2d);
    output = ggml_reshape_3d(context, output,
                             columns->ne[1], weight->ne[2], columns->ne[2]);
    return add_channel_bias(context, output, bias);
}

ggml_tensor * conv_transpose_1d_f32(ggml_context * context,
                                    ggml_tensor * weight,
                                    ggml_tensor * input,
                                    ggml_tensor * bias,
                                    int stride,
                                    int padding) {
    if (context == nullptr) {
        fail("context is null");
    }
    require_f32(weight, "transpose-convolution weight");
    require_f32(input, "transpose-convolution input");
    if (stride <= 0 || padding < 0) {
        fail("invalid ConvTranspose1d stride or padding");
    }
    if (weight->ne[0] <= 0 || weight->ne[1] <= 0 ||
        weight->ne[2] != input->ne[1] || weight->ne[3] != 1 ||
        input->ne[2] != 1 || input->ne[3] != 1) {
        fail("incompatible ConvTranspose1d weight/input shapes");
    }

    ggml_tensor * expanded = ggml_conv_transpose_1d(
        context, weight, input, stride, 0, 1);
    if (padding > 0) {
        const int64_t cropped_length = expanded->ne[0] - 2LL * padding;
        if (cropped_length <= 0) {
            fail("ConvTranspose1d padding removes the entire output");
        }
        const std::size_t offset = static_cast<std::size_t>(padding) * expanded->nb[0];
        expanded = ggml_view_3d(context, expanded,
                                cropped_length, expanded->ne[1], expanded->ne[2],
                                expanded->nb[1], expanded->nb[2], offset);
        expanded = ggml_cont(context, expanded);
    }
    return add_channel_bias(context, expanded, bias);
}

ggml_tensor * snake_beta(ggml_context * context,
                         ggml_tensor * input,
                         ggml_tensor * alpha_log,
                         ggml_tensor * beta_log) {
    if (context == nullptr) {
        fail("context is null");
    }
    require_f32(input, "SnakeBeta input");
    require_f32(alpha_log, "SnakeBeta alpha_log");
    require_f32(beta_log, "SnakeBeta beta_log");
    const auto valid_parameter = [input](const ggml_tensor * parameter) {
        return parameter->ne[0] == 1 && parameter->ne[1] == input->ne[1] &&
               parameter->ne[2] == 1 && parameter->ne[3] == 1;
    };
    if (!valid_parameter(alpha_log) || !valid_parameter(beta_log)) {
        fail("SnakeBeta parameters must have shape [1, channels, 1]");
    }

    ggml_tensor * alpha = ggml_exp(context, alpha_log);
    ggml_tensor * beta = ggml_scale_bias(context, ggml_exp(context, beta_log),
                                         1.0F, 1.0e-9F);
    ggml_tensor * periodic = ggml_sqr(context,
                                     ggml_sin(context, ggml_mul(context, input, alpha)));
    return ggml_add(context, input, ggml_div(context, periodic, beta));
}

} // namespace levo::detail
