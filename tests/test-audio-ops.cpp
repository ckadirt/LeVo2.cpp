#include "levo-audio-ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if LEVO_HAS_CUDA
#include "ggml-cuda.h"
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct backend_deleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend != nullptr) ggml_backend_free(backend);
    }
};

struct context_deleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) ggml_free(context);
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
    }
};

using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

void expect_close(const std::vector<float> & actual,
                  const std::vector<float> & expected,
                  float tolerance,
                  const char * label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(label) + " output size mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > tolerance) {
            throw std::runtime_error(std::string(label) + " mismatch at element " +
                                     std::to_string(index) + ": got " +
                                     std::to_string(actual[index]) + ", expected " +
                                     std::to_string(expected[index]));
        }
    }
}

std::vector<float> read_tensor(ggml_tensor * tensor) {
    std::vector<float> result(static_cast<std::size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        bool use_cuda = argc == 2 && std::string(argv[1]) == "cuda";
        backend_ptr backend;
#if LEVO_HAS_CUDA
        if (use_cuda) backend.reset(ggml_backend_cuda_init(0));
#else
        if (use_cuda) throw std::runtime_error("CUDA backend is not compiled");
#endif
        if (!backend) backend.reset(ggml_backend_cpu_init());
        if (!backend) throw std::runtime_error("cannot initialize GGML backend");

        constexpr std::size_t graph_nodes = 128;
        const std::size_t context_size = 64 * ggml_tensor_overhead() +
                                         ggml_graph_overhead_custom(graph_nodes, false);
        std::vector<std::byte> storage(context_size);
        context_ptr context(ggml_init({context_size, storage.data(), true}));
        if (!context) throw std::runtime_error("cannot initialize GGML context");

        ggml_tensor * conv_input = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, 4, 1, 1);
        ggml_tensor * conv_weight = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, 3, 1, 1);
        ggml_tensor * conv_bias = ggml_new_tensor_1d(
            context.get(), GGML_TYPE_F32, 1);
        ggml_tensor * conv_output = levo::detail::conv_1d_f32(
            context.get(), conv_weight, conv_input, conv_bias, 1, 1, 1);

        ggml_tensor * transpose_input = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, 2, 1, 1);
        ggml_tensor * transpose_weight = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, 4, 1, 1);
        ggml_tensor * transpose_bias = ggml_new_tensor_1d(
            context.get(), GGML_TYPE_F32, 1);
        ggml_tensor * transpose_output = levo::detail::conv_transpose_1d_f32(
            context.get(), transpose_weight, transpose_input, transpose_bias, 2, 1);

        ggml_tensor * snake_input = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, 3, 2, 1);
        ggml_tensor * alpha_log = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, 1, 2, 1);
        ggml_tensor * beta_log = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, 1, 2, 1);
        ggml_tensor * snake_output = levo::detail::snake_beta(
            context.get(), snake_input, alpha_log, beta_log);

        ggml_tensor * graph_output = ggml_add(
            context.get(), ggml_sum(context.get(), conv_output),
            ggml_add(context.get(), ggml_sum(context.get(), transpose_output),
                     ggml_sum(context.get(), snake_output)));
        ggml_cgraph * graph = ggml_new_graph_custom(context.get(), graph_nodes, false);
        ggml_build_forward_expand(graph, graph_output);

        buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
        if (!buffer) throw std::runtime_error("cannot allocate backend tensors");

        const std::array<float, 4> conv_input_data = {1.0F, 2.0F, 3.0F, 4.0F};
        const std::array<float, 3> conv_weight_data = {1.0F, 2.0F, 1.0F};
        const std::array<float, 1> conv_bias_data = {0.5F};
        const std::array<float, 2> transpose_input_data = {1.0F, 2.0F};
        const std::array<float, 4> transpose_weight_data = {1.0F, 1.0F, 1.0F, 1.0F};
        const std::array<float, 1> transpose_bias_data = {0.25F};
        const std::array<float, 6> snake_input_data = {
            -1.0F, 0.0F, 1.0F,
            -0.5F, 0.5F, 1.5F,
        };
        const std::array<float, 2> alpha_log_data = {
            std::log(2.0F), std::log(0.5F),
        };
        const std::array<float, 2> beta_log_data = {
            std::log(1.5F), std::log(0.75F),
        };
        ggml_backend_tensor_set(conv_input, conv_input_data.data(), 0, sizeof(conv_input_data));
        ggml_backend_tensor_set(conv_weight, conv_weight_data.data(), 0, sizeof(conv_weight_data));
        ggml_backend_tensor_set(conv_bias, conv_bias_data.data(), 0, sizeof(conv_bias_data));
        ggml_backend_tensor_set(transpose_input, transpose_input_data.data(), 0,
                                sizeof(transpose_input_data));
        ggml_backend_tensor_set(transpose_weight, transpose_weight_data.data(), 0,
                                sizeof(transpose_weight_data));
        ggml_backend_tensor_set(transpose_bias, transpose_bias_data.data(), 0,
                                sizeof(transpose_bias_data));
        ggml_backend_tensor_set(snake_input, snake_input_data.data(), 0, sizeof(snake_input_data));
        ggml_backend_tensor_set(alpha_log, alpha_log_data.data(), 0, sizeof(alpha_log_data));
        ggml_backend_tensor_set(beta_log, beta_log_data.data(), 0, sizeof(beta_log_data));

        const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("audio graph failed: ") +
                                     ggml_status_to_string(status));
        }

        expect_close(read_tensor(conv_output), {4.5F, 8.5F, 12.5F, 11.5F},
                     1.0e-5F, "Conv1d");
        expect_close(read_tensor(transpose_output), {1.25F, 3.25F, 3.25F, 2.25F},
                     1.0e-5F, "ConvTranspose1d");

        std::vector<float> snake_expected;
        snake_expected.reserve(snake_input_data.size());
        for (std::size_t channel = 0; channel < 2; ++channel) {
            const float alpha = std::exp(alpha_log_data[channel]);
            const float beta = std::exp(beta_log_data[channel]) + 1.0e-9F;
            for (std::size_t time = 0; time < 3; ++time) {
                const float value = snake_input_data[channel * 3 + time];
                const float sine = std::sin(value * alpha);
                snake_expected.push_back(value + sine * sine / beta);
            }
        }
        expect_close(read_tensor(snake_output), snake_expected,
                     use_cuda ? 2.0e-5F : 1.0e-6F, "SnakeBeta");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
