#include "levo-lm.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace levo::detail {
namespace {

constexpr std::size_t graph_capacity = 4096;

struct context_deleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

struct allocator_deleter {
    void operator()(ggml_gallocr_t allocator) const noexcept {
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using allocator_ptr = std::unique_ptr<ggml_gallocr, allocator_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo LM: " + message);
}

std::string layer_tensor_name(bool detail, unsigned layer, const char * suffix) {
    return detail ? tensor_names::detail_layer(layer, suffix) : tensor_names::main_layer(layer, suffix);
}

void require_shape(const ggml_tensor * tensor, const std::string & name,
                   std::initializer_list<int64_t> expected) {
    if (tensor == nullptr) {
        fail("missing required tensor '" + name + "'");
    }
    if (ggml_n_dims(tensor) != static_cast<int>(expected.size())) {
        fail("tensor '" + name + "' has an unexpected rank");
    }
    std::size_t dimension = 0;
    for (const int64_t value : expected) {
        if (tensor->ne[dimension++] != value) {
            std::ostringstream message;
            message << "tensor '" << name << "' has an unexpected shape";
            fail(message.str());
        }
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        fail("tensor '" + name + "' has an unsupported type");
    }
}

ggml_tensor * require_tensor(const model & model, const std::string & name,
                              std::initializer_list<int64_t> expected) {
    ggml_tensor * tensor = model.tensor(name);
    require_shape(tensor, name, expected);
    return tensor;
}

struct tower_weights {
    struct layer {
        ggml_tensor * attention_norm = nullptr;
        ggml_tensor * q = nullptr;
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
        ggml_tensor * output = nullptr;
        ggml_tensor * ffn_norm = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
    };
    std::vector<layer> layers;
    ggml_tensor * final_norm = nullptr;
};

tower_weights load_tower_weights(const model & model, bool detail) {
    const model_hparams & hparams = model.hparams();
    const int32_t layer_count = detail ? hparams.detail_layers : hparams.main_layers;
    tower_weights result;
    result.layers.resize(static_cast<std::size_t>(layer_count));
    for (int32_t index = 0; index < layer_count; ++index) {
        auto & layer = result.layers[static_cast<std::size_t>(index)];
        const auto name = [detail, index](const char * suffix) {
            return layer_tensor_name(detail, static_cast<unsigned>(index), suffix);
        };
        layer.attention_norm = require_tensor(model, name("attn_norm.weight"), {hparams.embedding_length});
        layer.q = require_tensor(model, name("attn_q.weight"), {hparams.embedding_length, hparams.embedding_length});
        layer.k = require_tensor(model, name("attn_k.weight"), {hparams.embedding_length,
                                                           hparams.kv_attention_heads * hparams.head_dimension()});
        layer.v = require_tensor(model, name("attn_v.weight"), {hparams.embedding_length,
                                                           hparams.kv_attention_heads * hparams.head_dimension()});
        layer.output = require_tensor(model, name("attn_output.weight"), {hparams.embedding_length, hparams.embedding_length});
        layer.ffn_norm = require_tensor(model, name("ffn_norm"), {hparams.embedding_length});
        layer.gate = require_tensor(model, name("ffn_gate.weight"), {hparams.embedding_length, hparams.feed_forward_length});
        layer.up = require_tensor(model, name("ffn_up.weight"), {hparams.embedding_length, hparams.feed_forward_length});
        layer.down = require_tensor(model, name("ffn_down.weight"), {hparams.feed_forward_length, hparams.embedding_length});
    }
    result.final_norm = require_tensor(model, detail ? tensor_names::detail_norm : tensor_names::main_norm,
                                       {hparams.embedding_length});
    return result;
}

ggml_tensor * rms_norm_weighted(ggml_context * context, ggml_tensor * input,
                                ggml_tensor * weight, float epsilon) {
    ggml_tensor * normalized = ggml_rms_norm(context, input, epsilon);
    // GGUF F16 mirrors the released weights, while GGML's activation path is
    // F32. CUDA broadcast kernels require matching element widths.
    if (weight->type != normalized->type) weight = ggml_cast(context, weight, normalized->type);
    return ggml_mul(context, normalized, weight);
}

ggml_tensor * add_bias(ggml_context * context, ggml_tensor * input, ggml_tensor * bias) {
    if (bias->type != input->type) bias = ggml_cast(context, bias, input->type);
    return ggml_add(context, input, bias);
}

ggml_tensor * round_f16_if_model(ggml_context * context, ggml_tensor * input,
                                 const ggml_tensor * model_tensor) {
    if (model_tensor->type != GGML_TYPE_F16) return input;
    return ggml_cast(context, ggml_cast(context, input, GGML_TYPE_F16), GGML_TYPE_F32);
}

ggml_tensor * llama_tower(ggml_context * context, ggml_tensor * input, ggml_tensor * positions,
                          const tower_weights & weights, const model_hparams & hparams,
                          float rope_theta) {
    const int32_t token_count = static_cast<int32_t>(input->ne[1]);
    const int32_t head_dimension = hparams.head_dimension();
    const int32_t kv_width = hparams.kv_attention_heads * head_dimension;
    ggml_tensor * current = input;

    for (const auto & layer : weights.layers) {
        const ggml_tensor * residual = current;
        ggml_tensor * normalized = rms_norm_weighted(context, current, layer.attention_norm,
                                                      hparams.rms_norm_epsilon);
        ggml_tensor * q = ggml_mul_mat(context, layer.q, normalized);
        ggml_tensor * k = ggml_mul_mat(context, layer.k, normalized);
        ggml_tensor * v = ggml_mul_mat(context, layer.v, normalized);

        q = ggml_rope_ext(context,
                           ggml_reshape_3d(context, q, head_dimension, hparams.attention_heads, token_count),
                           positions, nullptr, head_dimension, GGML_ROPE_TYPE_NEOX, hparams.context_length,
                           rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
        k = ggml_rope_ext(context,
                           ggml_reshape_3d(context, k, head_dimension, hparams.kv_attention_heads, token_count),
                           positions, nullptr, head_dimension, GGML_ROPE_TYPE_NEOX, hparams.context_length,
                           rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);

        // This v2-medium model has the same number of query and KV heads.  The
        // explicit check keeps an accidental GQA artifact from producing an
        // incorrectly broadcast graph.
        if (hparams.attention_heads != hparams.kv_attention_heads || kv_width != hparams.embedding_length) {
            fail("the uncached Llama graph currently requires attention_heads == kv_attention_heads");
        }
        q = ggml_permute(context, q, 0, 2, 1, 3); // [head_dim, tokens, heads]
        k = ggml_permute(context, k, 0, 2, 1, 3); // [head_dim, tokens, heads]
        ggml_tensor * attention = ggml_mul_mat(context, k, q); // [keys, queries, heads]
        attention = ggml_diag_mask_inf(context, attention, 0);
        attention = ggml_soft_max_ext(context, attention, nullptr,
                                      1.0F / std::sqrt(static_cast<float>(head_dimension)), 0.0F);

        ggml_tensor * value = ggml_cont_3d(
            context,
            ggml_permute(context,
                         ggml_reshape_3d(context, v, head_dimension, hparams.kv_attention_heads, token_count),
                         1, 2, 0, 3),
            token_count, head_dimension, hparams.kv_attention_heads);
        ggml_tensor * attended = ggml_mul_mat(context, value, attention);
        attended = ggml_permute(context, attended, 0, 2, 1, 3);
        attended = ggml_cont_2d(context, attended, hparams.embedding_length, token_count);
        current = ggml_add(context, ggml_mul_mat(context, layer.output, attended),
                           const_cast<ggml_tensor *>(residual));

        residual = current;
        normalized = rms_norm_weighted(context, current, layer.ffn_norm, hparams.rms_norm_epsilon);
        ggml_tensor * gate = ggml_silu(context, ggml_mul_mat(context, layer.gate, normalized));
        ggml_tensor * up = ggml_mul_mat(context, layer.up, normalized);
        current = ggml_add(context, ggml_mul_mat(context, layer.down, ggml_mul(context, gate, up)),
                           const_cast<ggml_tensor *>(residual));
    }
    return rms_norm_weighted(context, current, weights.final_norm, hparams.rms_norm_epsilon);
}

void validate_input(const lm_input & input, const model_hparams & hparams) {
    const std::size_t count = input.mixed_tokens.size();
    if (count == 0) {
        fail("at least one token is required");
    }
    if (count > static_cast<std::size_t>(hparams.context_length)) {
        fail("uncached input exceeds the model context length");
    }
    if (input.vocal_tokens.size() != count || input.bgm_tokens.size() != count) {
        fail("mixed, vocal, and BGM token sequences must have identical lengths");
    }
    if (!input.positions.empty() && input.positions.size() != count) {
        fail("positions must be omitted or have one entry per token");
    }
    const auto valid_token = [&hparams](int32_t token) {
        return token >= 0 && token <= hparams.special_token_id;
    };
    if (!std::all_of(input.mixed_tokens.begin(), input.mixed_tokens.end(), valid_token) ||
        !std::all_of(input.vocal_tokens.begin(), input.vocal_tokens.end(), valid_token) ||
        !std::all_of(input.bgm_tokens.begin(), input.bgm_tokens.end(), valid_token)) {
        fail("token IDs must be within the model input vocabulary");
    }
    const auto valid_position = [&hparams](int32_t position) {
        return position >= 0 && position < hparams.context_length;
    };
    if (!input.positions.empty() && !std::all_of(input.positions.begin(), input.positions.end(), valid_position)) {
        fail("positions must be within the model context length");
    }
}

} // namespace

struct lm::impl {
    std::shared_ptr<const model> weights;
    tower_weights main;
    tower_weights detail;
    ggml_tensor * mixed_embedding = nullptr;
    ggml_tensor * vocal_embedding = nullptr;
    ggml_tensor * bgm_embedding = nullptr;
    ggml_tensor * mixed_output = nullptr;
    ggml_tensor * vocal_output = nullptr;
    ggml_tensor * bgm_output = nullptr;
    ggml_tensor * bridge_0_weight = nullptr;
    ggml_tensor * bridge_0_bias = nullptr;
    ggml_tensor * bridge_2_weight = nullptr;
    ggml_tensor * bridge_2_bias = nullptr;
};

lm::lm(std::shared_ptr<const model> model) : impl_(std::make_unique<impl>()) {
    if (!model) {
        fail("a model is required");
    }
    const model_hparams & hparams = model->hparams();
    impl_->weights = std::move(model);
    impl_->mixed_embedding = require_tensor(*impl_->weights, tensor_names::mixed_embedding,
                                             {hparams.embedding_length, hparams.token_input_size()});
    impl_->vocal_embedding = require_tensor(*impl_->weights, tensor_names::detail_vocal_embedding,
                                             {hparams.embedding_length, hparams.token_input_size()});
    impl_->bgm_embedding = require_tensor(*impl_->weights, tensor_names::detail_bgm_embedding,
                                           {hparams.embedding_length, hparams.token_input_size()});
    impl_->mixed_output = require_tensor(*impl_->weights, tensor_names::mixed_output,
                                          {hparams.embedding_length, hparams.token_output_size()});
    impl_->vocal_output = require_tensor(*impl_->weights, tensor_names::vocal_output,
                                          {hparams.embedding_length, hparams.token_output_size()});
    impl_->bgm_output = require_tensor(*impl_->weights, tensor_names::bgm_output,
                                        {hparams.embedding_length, hparams.token_output_size()});
    impl_->bridge_0_weight = require_tensor(*impl_->weights, tensor_names::bridge_0_weight,
                                             {hparams.embedding_length * 2, hparams.embedding_length});
    impl_->bridge_0_bias = require_tensor(*impl_->weights, tensor_names::bridge_0_bias,
                                           {hparams.embedding_length});
    impl_->bridge_2_weight = require_tensor(*impl_->weights, tensor_names::bridge_2_weight,
                                             {hparams.embedding_length, hparams.embedding_length});
    impl_->bridge_2_bias = require_tensor(*impl_->weights, tensor_names::bridge_2_bias,
                                           {hparams.embedding_length});
    impl_->main = load_tower_weights(*impl_->weights, false);
    impl_->detail = load_tower_weights(*impl_->weights, true);
}

lm::~lm() = default;
lm::lm(lm &&) noexcept = default;
lm & lm::operator=(lm &&) noexcept = default;

std::unique_ptr<lm> lm::create(std::shared_ptr<const model> model) {
    return std::unique_ptr<lm>(new lm(std::move(model)));
}

lm_output lm::forward(const lm_input & input) const {
    const model_hparams & hparams = impl_->weights->hparams();
    validate_input(input, hparams);
    const int32_t token_count = static_cast<int32_t>(input.mixed_tokens.size());
    const std::size_t context_size = graph_capacity * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(graph_capacity, false);
    const ggml_init_params params = {context_size, nullptr, true};
    context_ptr context(ggml_init(params));
    if (!context) {
        fail("cannot create GGML graph context");
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    if (!graph) {
        fail("cannot create GGML computation graph");
    }

    // Inputs live in a separate static context.  This is the current official
    // GGML pattern for graph-allocator inputs: it prevents the allocator from
    // reusing their storage before the host writes them.
    const ggml_init_params input_params = {4 * ggml_tensor_overhead(), nullptr, true};
    context_ptr input_context(ggml_init(input_params));
    if (!input_context) {
        fail("cannot create GGML input context");
    }
    ggml_tensor * mixed_ids = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, token_count);
    ggml_tensor * vocal_ids = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, token_count);
    ggml_tensor * bgm_ids = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, token_count);
    ggml_tensor * positions = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, token_count);
    if (!mixed_ids || !vocal_ids || !bgm_ids || !positions) {
        fail("cannot create GGML input tensors");
    }
    ggml_set_name(mixed_ids, "levo.mixed_ids");
    ggml_set_name(vocal_ids, "levo.vocal_ids");
    ggml_set_name(bgm_ids, "levo.bgm_ids");
    ggml_set_name(positions, "levo.positions");
    ggml_set_input(mixed_ids);
    ggml_set_input(vocal_ids);
    ggml_set_input(bgm_ids);
    ggml_set_input(positions);
    buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(input_context.get(), impl_->weights->backend()));
    if (!input_buffer) {
        fail("cannot allocate GGML input tensors");
    }

    ggml_tensor * main_input = ggml_get_rows(context.get(), impl_->mixed_embedding, mixed_ids);
    ggml_tensor * main_hidden = llama_tower(context.get(), main_input, positions, impl_->main, hparams,
                                            hparams.main_rope_theta);
    ggml_tensor * mixed_logits = ggml_mul_mat(context.get(), impl_->mixed_output, main_hidden);

    ggml_tensor * detail_embedding = ggml_add(
        context.get(), ggml_get_rows(context.get(), impl_->vocal_embedding, vocal_ids),
        ggml_get_rows(context.get(), impl_->bgm_embedding, bgm_ids));
    detail_embedding = round_f16_if_model(context.get(), detail_embedding, impl_->vocal_embedding);
    ggml_tensor * detail_input = ggml_concat(context.get(), detail_embedding, main_hidden, 0);
    detail_input = add_bias(context.get(),
                            ggml_mul_mat(context.get(), impl_->bridge_0_weight, detail_input),
                            impl_->bridge_0_bias);
    // torch.nn.GELU defaults to the exact erf formulation; ggml_gelu is the
    // tanh approximation and would create a persistent bridge-parity error.
    detail_input = ggml_gelu_erf(context.get(), detail_input);
    detail_input = add_bias(context.get(),
                            ggml_mul_mat(context.get(), impl_->bridge_2_weight, detail_input),
                            impl_->bridge_2_bias);
    ggml_tensor * detail_hidden = llama_tower(context.get(), detail_input, positions, impl_->detail, hparams,
                                              hparams.detail_rope_theta);
    ggml_tensor * vocal_logits = ggml_mul_mat(context.get(), impl_->vocal_output, detail_hidden);
    ggml_tensor * bgm_logits = ggml_mul_mat(context.get(), impl_->bgm_output, detail_hidden);
    ggml_set_name(mixed_logits, "levo.mixed_logits");
    ggml_set_name(vocal_logits, "levo.vocal_logits");
    ggml_set_name(bgm_logits, "levo.bgm_logits");
    ggml_set_output(mixed_logits);
    ggml_set_output(vocal_logits);
    ggml_set_output(bgm_logits);
    ggml_build_forward_expand(graph, mixed_logits);
    ggml_build_forward_expand(graph, vocal_logits);
    ggml_build_forward_expand(graph, bgm_logits);

    allocator_ptr allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->weights->backend())));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        fail("cannot allocate the GGML computation graph");
    }
    std::vector<int32_t> default_positions;
    const std::vector<int32_t> * position_values = &input.positions;
    if (position_values->empty()) {
        default_positions.resize(static_cast<std::size_t>(token_count));
        for (int32_t index = 0; index < token_count; ++index) {
            default_positions[static_cast<std::size_t>(index)] = index;
        }
        position_values = &default_positions;
    }
    const std::size_t input_bytes = static_cast<std::size_t>(token_count) * sizeof(int32_t);
    ggml_backend_tensor_set(mixed_ids, input.mixed_tokens.data(), 0, input_bytes);
    ggml_backend_tensor_set(vocal_ids, input.vocal_tokens.data(), 0, input_bytes);
    ggml_backend_tensor_set(bgm_ids, input.bgm_tokens.data(), 0, input_bytes);
    ggml_backend_tensor_set(positions, position_values->data(), 0, input_bytes);
    const ggml_status status = ggml_backend_graph_compute(impl_->weights->backend(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        fail(std::string("GGML graph execution failed: ") + ggml_status_to_string(status));
    }

    const std::size_t output_elements = static_cast<std::size_t>(hparams.token_output_size()) * token_count;
    lm_output result;
    result.token_count = token_count;
    result.vocabulary_size = hparams.token_output_size();
    result.mixed_logits.resize(output_elements);
    result.vocal_logits.resize(output_elements);
    result.bgm_logits.resize(output_elements);
    const std::size_t output_bytes = output_elements * sizeof(float);
    ggml_backend_tensor_get(mixed_logits, result.mixed_logits.data(), 0, output_bytes);
    ggml_backend_tensor_get(vocal_logits, result.vocal_logits.data(), 0, output_bytes);
    ggml_backend_tensor_get(bgm_logits, result.bgm_logits.data(), 0, output_bytes);
    return result;
}

} // namespace levo::detail
