#include "levo-quantization.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

struct context_deleter {
    void operator()(ggml_context * context) const noexcept { if (context) ggml_free(context); }
};
struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept { if (buffer) ggml_backend_buffer_free(buffer); }
};
struct allocator_deleter {
    void operator()(ggml_gallocr_t allocator) const noexcept { if (allocator) ggml_gallocr_free(allocator); }
};

void test_padded_quantized_matmul() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) throw std::runtime_error("cannot initialize CPU backend");
    try {
        std::unique_ptr<ggml_context, context_deleter> weights(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        std::unique_ptr<ggml_context, context_deleter> inputs(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        if (!weights || !inputs) throw std::runtime_error("cannot create padded quantization contexts");
        ggml_tensor * matrix = ggml_new_tensor_2d(weights.get(), GGML_TYPE_Q8_0, 32, 4);
        ggml_tensor * activation = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_F32, 6);
        if (!matrix || !activation) throw std::runtime_error("cannot create padded quantization tensors");
        std::unique_ptr<ggml_backend_buffer, buffer_deleter> weight_buffer(ggml_backend_alloc_ctx_tensors(weights.get(), backend));
        std::unique_ptr<ggml_backend_buffer, buffer_deleter> input_buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), backend));
        if (!weight_buffer || !input_buffer) throw std::runtime_error("cannot allocate padded quantization tensors");

        std::array<float, 32 * 4> original{};
        for (int row = 0; row < 4; ++row) for (int column = 0; column < 6; ++column) {
            original[static_cast<std::size_t>(row * 32 + column)] = 0.03125F * static_cast<float>((row + 1) * (column + 1));
        }
        std::vector<std::uint8_t> quantized(ggml_row_size(GGML_TYPE_Q8_0, 32) * 4U);
        require(ggml_quantize_chunk(GGML_TYPE_Q8_0, original.data(), quantized.data(), 0, 4, 32, nullptr) == quantized.size(),
                "Q8 padded matrix quantization produced an unexpected size");
        const std::array<float, 6> input{{0.25F, -0.5F, 0.75F, -1.0F, 1.25F, -1.5F}};
        ggml_backend_tensor_set(matrix, quantized.data(), 0, quantized.size());
        ggml_backend_tensor_set(activation, input.data(), 0, input.size() * sizeof(float));

        std::unique_ptr<ggml_context, context_deleter> graph_context(ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true}));
        if (!graph_context) throw std::runtime_error("cannot create padded quantization graph context");
        ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 32, false);
        if (!graph) throw std::runtime_error("cannot create padded quantization graph");
        ggml_tensor * padded = ggml_pad(graph_context.get(), activation, 26, 0, 0, 0);
        padded = ggml_reshape_2d(graph_context.get(), padded, 32, 1);
        ggml_tensor * output = ggml_mul_mat(graph_context.get(), matrix, padded);
        ggml_build_forward_expand(graph, output);
        std::unique_ptr<ggml_gallocr, allocator_deleter> allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) throw std::runtime_error("cannot allocate padded quantization graph");
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("padded quantized matrix graph failed");
        std::array<float, 4> actual{};
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (int row = 0; row < 4; ++row) {
            float expected = 0.0F;
            for (int column = 0; column < 6; ++column) expected += original[static_cast<std::size_t>(row * 32 + column)] * input[static_cast<std::size_t>(column)];
            require(std::fabs(actual[static_cast<std::size_t>(row)] - expected) < 0.015F,
                    "zero-padded Q8 matrix product changed logical Flow output");
        }
        ggml_backend_free(backend);
    } catch (...) {
        ggml_backend_free(backend);
        throw;
    }
}

} // namespace

int main() {
    try {
        using levo::quantization::profile;
        using levo::quantization::lelm_tensor_type;
        using levo::quantization::flow_tensor_type;

        require(levo::quantization::parse("Q8_0") == profile::q8_0, "Q8_0 profile parse failed");
        require(levo::quantization::parse("Q6_K") == profile::q6_k, "Q6_K profile parse failed");
        require(levo::quantization::parse("Q5_K_M") == profile::q5_k_m, "Q5_K_M profile parse failed");
        require(levo::quantization::parse("Q4_K_M") == profile::q4_k_m, "Q4_K_M profile parse failed");
        require(!levo::quantization::parse("q4"), "unsupported profile was accepted");

        require(lelm_tensor_type("main.output_norm", 1, profile::q4_k_m) == GGML_TYPE_F32,
                "LeLM vector must remain F32");
        require(lelm_tensor_type("main.blk.0.attn_q.weight", 2, profile::q4_k_m) == GGML_TYPE_Q4_K,
                "LeLM non-critical Q4 matrix type is wrong");
        require(lelm_tensor_type("main.blk.0.attn_v.weight", 2, profile::q4_k_m) == GGML_TYPE_Q6_K,
                "LeLM attention V promotion is wrong");
        require(lelm_tensor_type("cond.lyrics.weight", 2, profile::q5_k_m) == GGML_TYPE_Q6_K,
                "LeLM conditioner promotion is wrong");
        require(lelm_tensor_type("output.mixed", 2, profile::q6_k) == GGML_TYPE_Q6_K,
                "LeLM Q6 routing is wrong");

        require(flow_tensor_type("flow.time_modulation.weight", 2, profile::q4_k_m) == GGML_TYPE_F32,
                "Flow control must remain F32");
        require(flow_tensor_type("flow.block.0.attn.qkv.weight", 2, profile::q4_k_m) == GGML_TYPE_Q4_K,
                "Flow base Q4 routing is wrong");
        require(flow_tensor_type("flow.block.0.attn.out.weight", 2, profile::q4_k_m) == GGML_TYPE_Q6_K,
                "Flow attention output promotion is wrong");
        require(flow_tensor_type("flow.block.0.ffn.out.weight", 2, profile::q5_k_m) == GGML_TYPE_Q6_K,
                "Flow FFN output promotion is wrong");
        require(levo::quantization::padded_input_columns(2200, GGML_TYPE_Q8_0) == 2208,
                "Q8 Flow padding is wrong");
        require(levo::quantization::padded_input_columns(2200, GGML_TYPE_Q4_K) == 2304,
                "K Flow hidden padding is wrong");
        require(levo::quantization::padded_input_columns(4400, GGML_TYPE_Q6_K) == 4608,
                "K Flow intermediate padding is wrong");
        require(levo::quantization::padded_input_columns(2200, GGML_TYPE_F32) == 2200,
                "F32 must not be padded");
        require(levo::quantization::is_hex_sha256(std::string(64, 'a')),
                "valid SHA-256 was rejected");
        require(!levo::quantization::is_hex_sha256("ABC"), "invalid SHA-256 was accepted");
        test_padded_quantized_matmul();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
