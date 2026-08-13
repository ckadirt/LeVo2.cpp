#include "levo-kv.h"

#include "levo-quantization.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace levo::detail {
namespace {

// This is deliberately independent of graph_capacity in levo-lm.cpp.  The
// cached graph has two write nodes per transformer layer in addition to the
// ordinary forward nodes.
constexpr std::size_t graph_capacity = 4096;
constexpr int32_t maximum_context_length = 10000;

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
    throw std::runtime_error("LeVo KV: " + message);
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
            fail("tensor '" + name + "' has an unexpected shape");
        }
    }
    const bool matrix = expected.size() >= 2;
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 &&
        !(matrix && quantization::is_quantized(tensor->type))) {
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
        layer.q = require_tensor(model, name("attn_q.weight"),
                                 {hparams.embedding_length, hparams.embedding_length});
        layer.k = require_tensor(model, name("attn_k.weight"),
                                 {hparams.embedding_length, hparams.kv_attention_heads * hparams.head_dimension()});
        layer.v = require_tensor(model, name("attn_v.weight"),
                                 {hparams.embedding_length, hparams.kv_attention_heads * hparams.head_dimension()});
        layer.output = require_tensor(model, name("attn_output.weight"),
                                      {hparams.embedding_length, hparams.embedding_length});
        layer.ffn_norm = require_tensor(model, name("ffn_norm"), {hparams.embedding_length});
        layer.gate = require_tensor(model, name("ffn_gate.weight"),
                                    {hparams.embedding_length, hparams.feed_forward_length});
        layer.up = require_tensor(model, name("ffn_up.weight"),
                                  {hparams.embedding_length, hparams.feed_forward_length});
        layer.down = require_tensor(model, name("ffn_down.weight"),
                                    {hparams.feed_forward_length, hparams.embedding_length});
    }
    result.final_norm = require_tensor(model, detail ? tensor_names::detail_norm : tensor_names::main_norm,
                                       {hparams.embedding_length});
    return result;
}

ggml_tensor * cast_f16(ggml_context * context, ggml_tensor * input) {
    return input->type == GGML_TYPE_F16 ? input : ggml_cast(context, input, GGML_TYPE_F16);
}

ggml_tensor * cast_f32(ggml_context * context, ggml_tensor * input) {
    return input->type == GGML_TYPE_F32 ? input : ggml_cast(context, input, GGML_TYPE_F32);
}

// The PyTorch reference casts the RMSNorm input to F32 for its reduction,
// casts the normalized values back to the activation dtype, and then applies
// the FP16 RMSNorm weight.  Keep that behavior behind an explicit flag: the
// normal GGML path intentionally retains its historical F32 activations.
ggml_tensor * rms_norm_weighted(ggml_context * context, ggml_tensor * input,
                                ggml_tensor * weight, float epsilon,
                                bool emulate_fp16_activations) {
    if (emulate_fp16_activations) {
        ggml_tensor * normalized = ggml_rms_norm(context, cast_f32(context, input), epsilon);
        normalized = cast_f16(context, normalized);
        return ggml_mul(context, normalized, weight);
    }

    ggml_tensor * normalized = ggml_rms_norm(context, input, epsilon);
    // GGML's RMSNorm result is F32 even for an F16 GGUF tensor.  CUDA's
    // broadcast kernel requires matching element widths, so promotion must be
    // explicit rather than left to an unsupported mixed-type broadcast.
    if (weight->type != normalized->type) {
        weight = ggml_cast(context, weight, normalized->type);
    }
    return ggml_mul(context, normalized, weight);
}

ggml_tensor * add_bias(ggml_context * context, ggml_tensor * input, ggml_tensor * bias) {
    if (bias->type != input->type) {
        bias = ggml_cast(context, bias, input->type);
    }
    return ggml_add(context, input, bias);
}

ggml_tensor * round_f16(ggml_context * context, ggml_tensor * input,
                        bool emulate_fp16_activations) {
    return emulate_fp16_activations ? cast_f16(context, input) : input;
}

// An output boundary must stay FP16 inside the graph, but callers retrieving
// logits or traces require host F32 values.  This conversion is deliberately
// after the FP16 round, not a replacement for it.
ggml_tensor * output_f32(ggml_context * context, ggml_tensor * input,
                         bool emulate_fp16_activations) {
    return emulate_fp16_activations ? cast_f32(context, input) : input;
}

bool is_cuda_backend(ggml_backend_t backend) {
#if LEVO_HAS_CUDA
    const char * const name = ggml_backend_name(backend);
    return name != nullptr && std::strncmp(name, "CUDA", 4) == 0;
#else
    (void) backend;
    return false;
#endif
}

// Flat per-tower buffers match GGML's long-standing GPT cache representation:
// [layer][position][kv_head][head_dim].  Views are shaped only in the graph,
// so all cache storage remains backend-resident and is never graph-allocated.
struct cache_storage {
    int32_t layer_count = 0;
    int32_t capacity = 0;
    int32_t kv_width = 0;
    ggml_type type = GGML_TYPE_F32;
    context_ptr context;
    buffer_ptr buffer;
    ggml_tensor * keys = nullptr;
    ggml_tensor * values = nullptr;

    void initialize(ggml_backend_t backend, int32_t requested_layers, int32_t requested_capacity,
                    int32_t requested_kv_width, ggml_type requested_type, const char * name) {
        layer_count = requested_layers;
        capacity = requested_capacity;
        kv_width = requested_kv_width;
        type = requested_type;
        if (layer_count == 0) {
            return;
        }
        const std::size_t layers = static_cast<std::size_t>(layer_count);
        const std::size_t positions = static_cast<std::size_t>(capacity);
        const std::size_t width = static_cast<std::size_t>(kv_width);
        if (layers > std::numeric_limits<std::size_t>::max() / positions ||
            layers * positions > std::numeric_limits<std::size_t>::max() / width) {
            fail("KV cache element count overflows size_t");
        }
        const std::size_t elements = layers * positions * width;
        if (elements > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
            fail("KV cache element count exceeds GGML limits");
        }
        const ggml_init_params params = {2 * ggml_tensor_overhead(), nullptr, true};
        context.reset(ggml_init(params));
        if (!context) {
            fail("cannot create persistent KV cache context");
        }
        keys = ggml_new_tensor_1d(context.get(), type, static_cast<int64_t>(elements));
        values = ggml_new_tensor_1d(context.get(), type, static_cast<int64_t>(elements));
        if (!keys || !values) {
            fail("cannot create persistent KV cache tensors");
        }
        ggml_set_name(keys, (std::string(name) + ".k").c_str());
        ggml_set_name(values, (std::string(name) + ".v").c_str());
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) {
            fail("cannot allocate persistent KV cache buffer");
        }
    }

    [[nodiscard]] size_t layer_offset_bytes(int32_t layer, int32_t position) const {
        if (layer < 0 || layer >= layer_count || position < 0 || position > capacity) {
            fail("invalid persistent KV cache view");
        }
        const std::size_t element =
            (static_cast<std::size_t>(layer) * static_cast<std::size_t>(capacity) +
             static_cast<std::size_t>(position)) * static_cast<std::size_t>(kv_width);
        return element * ggml_element_size(keys);
    }

    ggml_tensor * view(ggml_context * graph_context, bool key, int32_t layer,
                       int32_t position, int32_t count) const {
        if (count <= 0 || position > capacity - count) {
            fail("persistent KV cache view exceeds its capacity");
        }
        ggml_tensor * const source = key ? keys : values;
        return ggml_view_1d(graph_context, source,
                            static_cast<int64_t>(count) * kv_width,
                            layer_offset_bytes(layer, position));
    }
};

// A small, deliberately opt-in trace of the first main-tower block.  Capturing
// graph values rather than reimplementing the math in a diagnostic executable
// keeps the comparison tied to the production GGML expression tree.
struct operator_trace {
    std::vector<std::pair<std::string, ggml_tensor *>> tensors;
};

void capture_last_column(ggml_context * context, operator_trace * trace,
                         const char * name, ggml_tensor * tensor,
                         int32_t token_count, bool emulate_fp16_activations) {
    if (trace == nullptr) return;
    if (tensor->ne[1] != token_count) {
        fail(std::string("unexpected trace token dimension for ") + name);
    }
    ggml_tensor * value = ggml_view_1d(
        context, tensor, tensor->ne[0],
        static_cast<std::size_t>(token_count - 1) * tensor->nb[1]);
    value = output_f32(context, value, emulate_fp16_activations);
    ggml_set_output(value);
    trace->tensors.emplace_back(name, value);
}

void capture_last_heads(ggml_context * context, operator_trace * trace,
                        const char * name, ggml_tensor * tensor,
                        int32_t token_count, bool emulate_fp16_activations) {
    if (trace == nullptr) return;
    // `tensor` has GGML [key, query, head] layout.  Selecting the final
    // query then retaining [key, head] preserves PyTorch's flattened
    // [head, key] memory order (keys are the innermost dimension).
    if (tensor->ne[1] != token_count || ggml_n_dims(tensor) < 3) {
        fail(std::string("unexpected attention trace shape for ") + name);
    }
    ggml_tensor * value = ggml_view_2d(
        context, tensor, tensor->ne[0], tensor->ne[2], tensor->nb[2],
        static_cast<std::size_t>(token_count - 1) * tensor->nb[1]);
    value = ggml_cont(context, value);
    value = output_f32(context, value, emulate_fp16_activations);
    ggml_set_output(value);
    trace->tensors.emplace_back(name, value);
}

ggml_tensor * repeat_kv_heads(ggml_context * context, ggml_tensor * tensor,
                               int32_t query_heads) {
    if (tensor->ne[2] == query_heads) {
        return tensor;
    }
    ggml_tensor * target = ggml_new_tensor_3d(context, tensor->type,
                                                tensor->ne[0], tensor->ne[1], query_heads);
    if (!target) {
        fail("cannot create grouped-query attention repeat target");
    }
    return ggml_repeat(context, tensor, target);
}

// Creates a standard causal Llama block over the session's cached prefix plus
// the current append.  The explicit write nodes are intentionally expanded
// before the output graph; this is the official GGML persistent-cache pattern
// and guarantees the current append is visible to its own attention queries.
ggml_tensor * cached_llama_tower(ggml_context * context, ggml_cgraph * graph,
                                 ggml_tensor * input, ggml_tensor * positions,
                                 const tower_weights & weights, cache_storage & cache,
                                 const model_hparams & hparams, float rope_theta,
                                 int32_t past,
                                 bool emulate_fp16_activations,
                                 std::vector<ggml_tensor *> * trace = nullptr,
                                 operator_trace * first_layer_ops = nullptr) {
    const int32_t token_count = static_cast<int32_t>(input->ne[1]);
    const int32_t head_dimension = hparams.head_dimension();
    const int32_t kv_width = hparams.kv_attention_heads * head_dimension;
    const int32_t total_tokens = past + token_count;
    ggml_tensor * current = input;

    if (cache.layer_count != static_cast<int32_t>(weights.layers.size()) ||
        cache.kv_width != kv_width || total_tokens > cache.capacity) {
        fail("KV cache shape does not match the Llama tower");
    }

    for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const auto & layer = weights.layers[layer_index];
        const ggml_tensor * residual = current;
        ggml_tensor * normalized = rms_norm_weighted(context, current, layer.attention_norm,
                                                      hparams.rms_norm_epsilon,
                                                      emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "input", current, token_count,
                                emulate_fp16_activations);
            capture_last_column(context, first_layer_ops, "attn_norm", normalized, token_count,
                                emulate_fp16_activations);
        }
        ggml_tensor * q = round_f16(context, ggml_mul_mat(context, layer.q, normalized),
                                    emulate_fp16_activations);
        ggml_tensor * k = round_f16(context, ggml_mul_mat(context, layer.k, normalized),
                                    emulate_fp16_activations);
        ggml_tensor * v = round_f16(context, ggml_mul_mat(context, layer.v, normalized),
                                    emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "q", q, token_count,
                                emulate_fp16_activations);
            capture_last_column(context, first_layer_ops, "k", k, token_count,
                                emulate_fp16_activations);
            capture_last_column(context, first_layer_ops, "v", v, token_count,
                                emulate_fp16_activations);
        }

        q = round_f16(context,
                      ggml_rope_ext(context,
                                      ggml_reshape_3d(context, q, head_dimension, hparams.attention_heads, token_count),
                                      positions, nullptr, head_dimension, GGML_ROPE_TYPE_NEOX, hparams.context_length,
                                      rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F),
                      emulate_fp16_activations);
        k = round_f16(context,
                      ggml_rope_ext(context,
                                      ggml_reshape_3d(context, k, head_dimension, hparams.kv_attention_heads, token_count),
                                      positions, nullptr, head_dimension, GGML_ROPE_TYPE_NEOX, hparams.context_length,
                                      rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F),
                      emulate_fp16_activations);
        if (layer_index == 0) {
            // RoPE tensors are [head_dim, heads, tokens], whose final-token
            // head blocks are contiguous and flatten identically to PyTorch.
            ggml_tensor * q_flat = ggml_reshape_2d(context, q, hparams.embedding_length, token_count);
            ggml_tensor * k_flat = ggml_reshape_2d(context, k, hparams.embedding_length, token_count);
            capture_last_column(context, first_layer_ops, "q_rope", q_flat, token_count,
                                emulate_fp16_activations);
            capture_last_column(context, first_layer_ops, "k_rope", k_flat, token_count,
                                emulate_fp16_activations);
        }

        const int32_t index = static_cast<int32_t>(layer_index);
        ggml_tensor * const key_write = cache.view(context, true, index, past, token_count);
        ggml_tensor * const value_write = cache.view(context, false, index, past, token_count);
        ggml_build_forward_expand(graph, ggml_cpy(context, k, key_write));
        ggml_build_forward_expand(graph, ggml_cpy(context, v, value_write));

        q = ggml_permute(context, q, 0, 2, 1, 3); // [head_dim, queries, query_heads]
        ggml_tensor * cached_k = cache.view(context, true, index, 0, total_tokens);
        cached_k = ggml_permute(context,
                                ggml_reshape_3d(context, cached_k, head_dimension,
                                                hparams.kv_attention_heads, total_tokens),
                                0, 2, 1, 3); // [head_dim, keys, kv_heads]
        cached_k = repeat_kv_heads(context, cached_k, hparams.attention_heads);

        ggml_tensor * attention = ggml_mul_mat(context, cached_k, q); // [keys, queries, heads]
        if (emulate_fp16_activations) {
            // PyTorch performs the half-precision QK product and scale before
            // it upcasts for masking/softmax.
            attention = ggml_scale(context, attention,
                                   1.0F / std::sqrt(static_cast<float>(head_dimension)));
            attention = cast_f32(context, cast_f16(context, attention));
        }
        if (layer_index == 0) {
            capture_last_heads(context, first_layer_ops, "attention_scores", attention, token_count,
                               emulate_fp16_activations);
        }
        attention = ggml_diag_mask_inf(context, attention, past);
        attention = ggml_soft_max_ext(context, attention, nullptr,
                                      emulate_fp16_activations ? 1.0F
                                                               : 1.0F / std::sqrt(static_cast<float>(head_dimension)),
                                      0.0F);
        attention = round_f16(context, attention, emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_heads(context, first_layer_ops, "attention_probs", attention, token_count,
                               emulate_fp16_activations);
        }

        ggml_tensor * cached_v = cache.view(context, false, index, 0, total_tokens);
        cached_v = ggml_cont_3d(
            context,
            ggml_permute(context,
                         ggml_reshape_3d(context, cached_v, head_dimension,
                                         hparams.kv_attention_heads, total_tokens),
                         1, 2, 0, 3),
            total_tokens, head_dimension, hparams.kv_attention_heads); // [keys, head_dim, kv_heads]
        cached_v = repeat_kv_heads(context, cached_v, hparams.attention_heads);
        ggml_tensor * attended = round_f16(context, ggml_mul_mat(context, cached_v, attention),
                                            emulate_fp16_activations); // [head_dim, queries, heads]
        attended = ggml_permute(context, attended, 0, 2, 1, 3);
        attended = ggml_cont_2d(context, attended, hparams.embedding_length, token_count);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "attention_output", attended, token_count,
                                emulate_fp16_activations);
        }
        ggml_tensor * attention_output = round_f16(context, ggml_mul_mat(context, layer.output, attended),
                                                    emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "o_proj", attention_output, token_count,
                                emulate_fp16_activations);
        }
        current = round_f16(context, ggml_add(context, attention_output,
                                               const_cast<ggml_tensor *>(residual)),
                            emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "post_attention_residual", current, token_count,
                                emulate_fp16_activations);
        }

        residual = current;
        normalized = rms_norm_weighted(context, current, layer.ffn_norm, hparams.rms_norm_epsilon,
                                       emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "ffn_norm", normalized, token_count,
                                emulate_fp16_activations);
        }
        ggml_tensor * gate_linear = round_f16(context, ggml_mul_mat(context, layer.gate, normalized),
                                              emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "gate", gate_linear, token_count,
                                emulate_fp16_activations);
        }
        ggml_tensor * gate = round_f16(context, ggml_silu(context, gate_linear),
                                       emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "gate_silu", gate, token_count,
                                emulate_fp16_activations);
        }
        ggml_tensor * up = round_f16(context, ggml_mul_mat(context, layer.up, normalized),
                                     emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "up", up, token_count,
                                emulate_fp16_activations);
        }
        ggml_tensor * ffn_product = round_f16(context, ggml_mul(context, gate, up),
                                               emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "ffn_product", ffn_product, token_count,
                                emulate_fp16_activations);
        }
        ggml_tensor * ffn_output = round_f16(context, ggml_mul_mat(context, layer.down, ffn_product),
                                             emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "down", ffn_output, token_count,
                                emulate_fp16_activations);
        }
        current = round_f16(context, ggml_add(context, ffn_output,
                                               const_cast<ggml_tensor *>(residual)),
                            emulate_fp16_activations);
        if (layer_index == 0) {
            capture_last_column(context, first_layer_ops, "output", current, token_count,
                                emulate_fp16_activations);
        }
        if (trace != nullptr) {
            ggml_tensor * last = ggml_view_1d(
                context, current, hparams.embedding_length,
                static_cast<std::size_t>(token_count - 1) * current->nb[1]);
            last = output_f32(context, last, emulate_fp16_activations);
            ggml_set_output(last);
            ggml_build_forward_expand(graph, last);
            trace->push_back(last);
        }
    }
    return rms_norm_weighted(context, current, weights.final_norm, hparams.rms_norm_epsilon,
                             emulate_fp16_activations);
}

void validate_tokens(const lm_input & input, const model_hparams & hparams,
                     int32_t cache_tokens, int32_t cache_capacity,
                     int32_t next_position, bool first_append) {
    const std::size_t count = input.mixed_tokens.size();
    if (count == 0) {
        fail("at least one token is required");
    }
    if (count > static_cast<std::size_t>(cache_capacity - cache_tokens)) {
        fail("append exceeds the KV cache context length");
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

    if (input.positions.empty()) {
        if (next_position > hparams.context_length - static_cast<int32_t>(count)) {
            fail("append positions exceed the model context length");
        }
        return;
    }
    const int32_t start = input.positions.front();
    if (start < 0 || start >= hparams.context_length) {
        fail("positions must be within the model context length");
    }
    if (!first_append && start != next_position) {
        fail("positions must continue from the session's absolute position");
    }
    for (std::size_t index = 0; index < count; ++index) {
        const int64_t expected = static_cast<int64_t>(start) + static_cast<int64_t>(index);
        if (expected >= hparams.context_length || input.positions[index] != expected) {
            fail("positions must be contiguous and within the model context length");
        }
    }
}

void validate_condition(const condition_pair & condition, const model_hparams & hparams) {
    const auto validate_tensor = [&hparams](const condition_tensor & tensor, const char * name) {
        if (tensor.width != hparams.embedding_length || tensor.prefix < 0) {
            fail(std::string("condition ") + name + " has an incompatible shape");
        }
        const std::size_t width = static_cast<std::size_t>(tensor.width);
        const std::size_t prefix = static_cast<std::size_t>(tensor.prefix);
        if (prefix > std::numeric_limits<std::size_t>::max() / width ||
            tensor.values.size() != width * prefix) {
            fail(std::string("condition ") + name + " has an invalid dense payload");
        }
    };
    validate_tensor(condition.main, "main");
    validate_tensor(condition.detail, "detail");
    if (condition.main.prefix != condition.detail.prefix) {
        fail("main and detail conditions must have equal prefix lengths");
    }
    if (condition.main.prefix > hparams.context_length) {
        fail("condition prefix exceeds the model context length");
    }
}

ggml_tensor * logits_tail(ggml_context * context, ggml_tensor * logits,
                           int32_t prefix, int32_t audio_tokens) {
    if (logits == nullptr || logits->ne[0] <= 0 || logits->ne[1] < prefix + audio_tokens) {
        fail("cannot select conditioned audio logits");
    }
    return ggml_view_2d(context, logits, logits->ne[0], audio_tokens, logits->nb[1],
                        static_cast<size_t>(prefix) * logits->nb[1]);
}

} // namespace

struct kv_session::impl {
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
    cache_storage main_cache;
    cache_storage detail_cache;
    bool emulate_fp16_activations = false;
    int32_t cache_tokens = 0;
    int32_t next_absolute_position = 0;
};

kv_session::kv_session(std::shared_ptr<const model> model, bool emulate_fp16_activations)
    : impl_(std::make_unique<impl>()) {
    if (!model) {
        fail("a model is required");
    }
    const model_hparams & hparams = model->hparams();
    if (hparams.context_length > maximum_context_length) {
        fail("KV cache supports at most 10000 context positions");
    }
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

    // Production F16 GGUFs use the emulation path.  Do not turn it on for a
    // synthetic/F32 model: this keeps both existing tests and callers on their
    // original F32-activation semantics even when they pass true explicitly.
    impl_->emulate_fp16_activations = emulate_fp16_activations &&
                                        impl_->mixed_embedding->type == GGML_TYPE_F16;
    const ggml_type cache_type = (is_cuda_backend(impl_->weights->backend()) ||
                                  impl_->emulate_fp16_activations)
        ? GGML_TYPE_F16
        : GGML_TYPE_F32;
    const int32_t kv_width = hparams.kv_attention_heads * hparams.head_dimension();
    impl_->main_cache.initialize(impl_->weights->backend(), hparams.main_layers, hparams.context_length,
                                 kv_width, cache_type, "levo.main_kv");
    impl_->detail_cache.initialize(impl_->weights->backend(), hparams.detail_layers, hparams.context_length,
                                   kv_width, cache_type, "levo.detail_kv");
}

kv_session::~kv_session() = default;
kv_session::kv_session(kv_session &&) noexcept = default;
kv_session & kv_session::operator=(kv_session &&) noexcept = default;

std::unique_ptr<kv_session> kv_session::create(std::shared_ptr<const model> model,
                                                bool emulate_fp16_activations) {
    return std::unique_ptr<kv_session>(new kv_session(std::move(model), emulate_fp16_activations));
}

lm_output kv_session::prefill(const lm_input & input) {
    const model_hparams & hparams = impl_->weights->hparams();
    validate_tokens(input, hparams, impl_->cache_tokens, impl_->main_cache.capacity,
                    impl_->next_absolute_position, impl_->cache_tokens == 0);
    const int32_t token_count = static_cast<int32_t>(input.mixed_tokens.size());
    const std::size_t context_size = graph_capacity * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(graph_capacity, false);
    context_ptr context(ggml_init({context_size, nullptr, true}));
    if (!context) {
        fail("cannot create GGML graph context");
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    if (!graph) {
        fail("cannot create GGML computation graph");
    }

    // Inputs live in a separate static context so graph allocation cannot
    // recycle them before their host values are uploaded.
    context_ptr input_context(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
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
    ggml_set_name(mixed_ids, "levo.kv.mixed_ids");
    ggml_set_name(vocal_ids, "levo.kv.vocal_ids");
    ggml_set_name(bgm_ids, "levo.kv.bgm_ids");
    ggml_set_name(positions, "levo.kv.positions");
    ggml_set_input(mixed_ids);
    ggml_set_input(vocal_ids);
    ggml_set_input(bgm_ids);
    ggml_set_input(positions);
    buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(input_context.get(), impl_->weights->backend()));
    if (!input_buffer) {
        fail("cannot allocate GGML input tensors");
    }

    ggml_tensor * main_input = round_f16(context.get(),
                                         ggml_get_rows(context.get(), impl_->mixed_embedding, mixed_ids),
                                         impl_->emulate_fp16_activations);
    ggml_tensor * main_hidden = cached_llama_tower(context.get(), graph, main_input, positions,
                                                   impl_->main, impl_->main_cache, hparams,
                                                   hparams.main_rope_theta, impl_->cache_tokens,
                                                   impl_->emulate_fp16_activations);
    ggml_tensor * mixed_logits = output_f32(
        context.get(), round_f16(context.get(), ggml_mul_mat(context.get(), impl_->mixed_output, main_hidden),
                                 impl_->emulate_fp16_activations),
        impl_->emulate_fp16_activations);

    ggml_tensor * detail_embedding = ggml_add(
        context.get(), ggml_get_rows(context.get(), impl_->vocal_embedding, vocal_ids),
        ggml_get_rows(context.get(), impl_->bgm_embedding, bgm_ids));
    detail_embedding = round_f16(context.get(), detail_embedding, impl_->emulate_fp16_activations);
    ggml_tensor * detail_input = ggml_concat(context.get(), detail_embedding, main_hidden, 0);
    detail_input = round_f16(context.get(), add_bias(context.get(),
        round_f16(context.get(), ggml_mul_mat(context.get(), impl_->bridge_0_weight, detail_input),
                  impl_->emulate_fp16_activations), impl_->bridge_0_bias),
        impl_->emulate_fp16_activations);
    detail_input = round_f16(context.get(), ggml_gelu_erf(context.get(), detail_input),
                             impl_->emulate_fp16_activations);
    detail_input = round_f16(context.get(), add_bias(context.get(),
        round_f16(context.get(), ggml_mul_mat(context.get(), impl_->bridge_2_weight, detail_input),
                  impl_->emulate_fp16_activations), impl_->bridge_2_bias),
        impl_->emulate_fp16_activations);
    ggml_tensor * detail_hidden = cached_llama_tower(context.get(), graph, detail_input, positions,
                                                     impl_->detail, impl_->detail_cache, hparams,
                                                     hparams.detail_rope_theta, impl_->cache_tokens,
                                                     impl_->emulate_fp16_activations);
    ggml_tensor * vocal_logits = output_f32(
        context.get(), round_f16(context.get(), ggml_mul_mat(context.get(), impl_->vocal_output, detail_hidden),
                                 impl_->emulate_fp16_activations),
        impl_->emulate_fp16_activations);
    ggml_tensor * bgm_logits = output_f32(
        context.get(), round_f16(context.get(), ggml_mul_mat(context.get(), impl_->bgm_output, detail_hidden),
                                 impl_->emulate_fp16_activations),
        impl_->emulate_fp16_activations);
    ggml_set_name(mixed_logits, "levo.kv.mixed_logits");
    ggml_set_name(vocal_logits, "levo.kv.vocal_logits");
    ggml_set_name(bgm_logits, "levo.kv.bgm_logits");
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

    std::vector<int32_t> generated_positions;
    const std::vector<int32_t> * position_values = &input.positions;
    if (position_values->empty()) {
        generated_positions.resize(static_cast<std::size_t>(token_count));
        for (int32_t index = 0; index < token_count; ++index) {
            generated_positions[static_cast<std::size_t>(index)] = impl_->next_absolute_position + index;
        }
        position_values = &generated_positions;
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

    impl_->cache_tokens += token_count;
    impl_->next_absolute_position = position_values->back() + 1;
    return result;
}

lm_output kv_session::prefill_conditioned(const condition_pair & condition, const lm_input & audio,
                                          bool capture_trace) {
    if (impl_->cache_tokens != 0) {
        fail("conditioning can only be prefetched into an empty KV session");
    }
    const model_hparams & hparams = impl_->weights->hparams();
    validate_condition(condition, hparams);
    const int32_t prefix_length = condition.main.prefix;
    // Validate the externally visible audio positions against the positions
    // immediately after the dense prefix.  The graph itself receives the full
    // [condition, audio] position sequence, beginning at absolute position 0.
    validate_tokens(audio, hparams, prefix_length, impl_->main_cache.capacity,
                    prefix_length, false);
    const int32_t audio_tokens = static_cast<int32_t>(audio.mixed_tokens.size());
    const int32_t total_tokens = prefix_length + audio_tokens;
    const std::size_t context_size = graph_capacity * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(graph_capacity, false);
    context_ptr context(ggml_init({context_size, nullptr, true}));
    if (!context) {
        fail("cannot create GGML graph context");
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    if (!graph) {
        fail("cannot create GGML computation graph");
    }

    const std::size_t dense_inputs = prefix_length == 0 ? 0U : 2U;
    context_ptr input_context(ggml_init({(4U + dense_inputs) * ggml_tensor_overhead(), nullptr, true}));
    if (!input_context) {
        fail("cannot create GGML input context");
    }
    ggml_tensor * mixed_ids = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, audio_tokens);
    ggml_tensor * vocal_ids = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, audio_tokens);
    ggml_tensor * bgm_ids = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, audio_tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, total_tokens);
    ggml_tensor * main_condition = nullptr;
    ggml_tensor * detail_condition = nullptr;
    if (prefix_length != 0) {
        main_condition = ggml_new_tensor_2d(input_context.get(), GGML_TYPE_F32,
                                             hparams.embedding_length, prefix_length);
        detail_condition = ggml_new_tensor_2d(input_context.get(), GGML_TYPE_F32,
                                               hparams.embedding_length, prefix_length);
    }
    if (!mixed_ids || !vocal_ids || !bgm_ids || !positions ||
        (prefix_length != 0 && (!main_condition || !detail_condition))) {
        fail("cannot create conditioned GGML input tensors");
    }
    ggml_set_name(mixed_ids, "levo.kv.conditioned_mixed_ids");
    ggml_set_name(vocal_ids, "levo.kv.conditioned_vocal_ids");
    ggml_set_name(bgm_ids, "levo.kv.conditioned_bgm_ids");
    ggml_set_name(positions, "levo.kv.conditioned_positions");
    ggml_set_input(mixed_ids);
    ggml_set_input(vocal_ids);
    ggml_set_input(bgm_ids);
    ggml_set_input(positions);
    if (main_condition) {
        ggml_set_name(main_condition, "levo.kv.main_condition");
        ggml_set_name(detail_condition, "levo.kv.detail_condition");
        ggml_set_input(main_condition);
        ggml_set_input(detail_condition);
    }
    buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(input_context.get(), impl_->weights->backend()));
    if (!input_buffer) {
        fail("cannot allocate conditioned GGML input tensors");
    }

    // Keep the backend input handles separate from the graph values.  FP16
    // emulation replaces the latter with cast nodes, which are graph outputs
    // and cannot accept the original F32 condition payload directly.
    ggml_tensor * const main_condition_input = main_condition;
    ggml_tensor * const detail_condition_input = detail_condition;

    ggml_tensor * main_audio = round_f16(context.get(),
                                         ggml_get_rows(context.get(), impl_->mixed_embedding, mixed_ids),
                                         impl_->emulate_fp16_activations);
    if (main_condition != nullptr) {
        main_condition = round_f16(context.get(), main_condition, impl_->emulate_fp16_activations);
    }
    ggml_tensor * main_input = main_condition
        ? ggml_concat(context.get(), main_condition, main_audio, 1)
        : main_audio;
    std::vector<ggml_tensor *> main_trace;
    std::vector<ggml_tensor *> detail_trace;
    operator_trace main_layer0_ops;
    ggml_tensor * main_hidden = cached_llama_tower(context.get(), graph, main_input, positions,
                                                   impl_->main, impl_->main_cache, hparams,
                                                   hparams.main_rope_theta, 0,
                                                   impl_->emulate_fp16_activations,
                                                   capture_trace ? &main_trace : nullptr,
                                                   capture_trace ? &main_layer0_ops : nullptr);
    if (capture_trace) {
        for (const auto & entry : main_layer0_ops.tensors) {
            ggml_build_forward_expand(graph, entry.second);
        }
    }
    ggml_tensor * main_norm_trace = nullptr;
    if (capture_trace) {
        main_norm_trace = ggml_view_1d(context.get(), main_hidden, hparams.embedding_length,
            static_cast<std::size_t>(total_tokens - 1) * main_hidden->nb[1]);
        main_norm_trace = output_f32(context.get(), main_norm_trace, impl_->emulate_fp16_activations);
        ggml_set_output(main_norm_trace);
        ggml_build_forward_expand(graph, main_norm_trace);
    }
    ggml_tensor * mixed_logits = output_f32(
        context.get(), round_f16(context.get(), ggml_mul_mat(context.get(), impl_->mixed_output, main_hidden),
                                 impl_->emulate_fp16_activations),
        impl_->emulate_fp16_activations);

    ggml_tensor * detail_audio = ggml_add(
        context.get(), ggml_get_rows(context.get(), impl_->vocal_embedding, vocal_ids),
        ggml_get_rows(context.get(), impl_->bgm_embedding, bgm_ids));
    detail_audio = round_f16(context.get(), detail_audio, impl_->emulate_fp16_activations);
    // Detail conditioner values are time-concatenated with the detail audio
    // embedding, not joined to main hidden until the bridge's width concat.
    if (detail_condition != nullptr) {
        detail_condition = round_f16(context.get(), detail_condition, impl_->emulate_fp16_activations);
    }
    ggml_tensor * detail_embedding = detail_condition
        ? ggml_concat(context.get(), detail_condition, detail_audio, 1)
        : detail_audio;
    ggml_tensor * detail_input = ggml_concat(context.get(), detail_embedding, main_hidden, 0);
    detail_input = round_f16(context.get(), add_bias(context.get(),
        round_f16(context.get(), ggml_mul_mat(context.get(), impl_->bridge_0_weight, detail_input),
                  impl_->emulate_fp16_activations), impl_->bridge_0_bias),
        impl_->emulate_fp16_activations);
    detail_input = round_f16(context.get(), ggml_gelu_erf(context.get(), detail_input),
                             impl_->emulate_fp16_activations);
    detail_input = round_f16(context.get(), add_bias(context.get(),
        round_f16(context.get(), ggml_mul_mat(context.get(), impl_->bridge_2_weight, detail_input),
                  impl_->emulate_fp16_activations), impl_->bridge_2_bias),
        impl_->emulate_fp16_activations);
    ggml_tensor * bridge_trace = nullptr;
    if (capture_trace) {
        bridge_trace = ggml_view_1d(context.get(), detail_input, hparams.embedding_length,
            static_cast<std::size_t>(total_tokens - 1) * detail_input->nb[1]);
        bridge_trace = output_f32(context.get(), bridge_trace, impl_->emulate_fp16_activations);
        ggml_set_output(bridge_trace);
        ggml_build_forward_expand(graph, bridge_trace);
    }
    ggml_tensor * detail_hidden = cached_llama_tower(context.get(), graph, detail_input, positions,
                                                     impl_->detail, impl_->detail_cache, hparams,
                                                     hparams.detail_rope_theta, 0,
                                                     impl_->emulate_fp16_activations,
                                                     capture_trace ? &detail_trace : nullptr);
    ggml_tensor * detail_norm_trace = nullptr;
    if (capture_trace) {
        detail_norm_trace = ggml_view_1d(context.get(), detail_hidden, hparams.embedding_length,
            static_cast<std::size_t>(total_tokens - 1) * detail_hidden->nb[1]);
        detail_norm_trace = output_f32(context.get(), detail_norm_trace, impl_->emulate_fp16_activations);
        ggml_set_output(detail_norm_trace);
        ggml_build_forward_expand(graph, detail_norm_trace);
    }
    ggml_tensor * vocal_logits = output_f32(
        context.get(), round_f16(context.get(), ggml_mul_mat(context.get(), impl_->vocal_output, detail_hidden),
                                 impl_->emulate_fp16_activations),
        impl_->emulate_fp16_activations);
    ggml_tensor * bgm_logits = output_f32(
        context.get(), round_f16(context.get(), ggml_mul_mat(context.get(), impl_->bgm_output, detail_hidden),
                                 impl_->emulate_fp16_activations),
        impl_->emulate_fp16_activations);

    ggml_tensor * mixed_audio_logits = logits_tail(context.get(), mixed_logits, prefix_length, audio_tokens);
    ggml_tensor * vocal_audio_logits = logits_tail(context.get(), vocal_logits, prefix_length, audio_tokens);
    ggml_tensor * bgm_audio_logits = logits_tail(context.get(), bgm_logits, prefix_length, audio_tokens);
    ggml_set_name(mixed_audio_logits, "levo.kv.conditioned_mixed_logits");
    ggml_set_name(vocal_audio_logits, "levo.kv.conditioned_vocal_logits");
    ggml_set_name(bgm_audio_logits, "levo.kv.conditioned_bgm_logits");
    ggml_set_output(mixed_audio_logits);
    ggml_set_output(vocal_audio_logits);
    ggml_set_output(bgm_audio_logits);
    ggml_build_forward_expand(graph, mixed_audio_logits);
    ggml_build_forward_expand(graph, vocal_audio_logits);
    ggml_build_forward_expand(graph, bgm_audio_logits);

    allocator_ptr allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->weights->backend())));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        fail("cannot allocate the conditioned GGML computation graph");
    }

    std::vector<int32_t> position_values(static_cast<std::size_t>(total_tokens));
    for (int32_t index = 0; index < total_tokens; ++index) {
        position_values[static_cast<std::size_t>(index)] = index;
    }
    const std::size_t audio_bytes = static_cast<std::size_t>(audio_tokens) * sizeof(int32_t);
    ggml_backend_tensor_set(mixed_ids, audio.mixed_tokens.data(), 0, audio_bytes);
    ggml_backend_tensor_set(vocal_ids, audio.vocal_tokens.data(), 0, audio_bytes);
    ggml_backend_tensor_set(bgm_ids, audio.bgm_tokens.data(), 0, audio_bytes);
    ggml_backend_tensor_set(positions, position_values.data(), 0,
                            position_values.size() * sizeof(int32_t));
    if (main_condition_input) {
        const std::size_t condition_bytes = condition.main.values.size() * sizeof(float);
        ggml_backend_tensor_set(main_condition_input, condition.main.data(), 0, condition_bytes);
        ggml_backend_tensor_set(detail_condition_input, condition.detail.data(), 0, condition_bytes);
    }
    const ggml_status status = ggml_backend_graph_compute(impl_->weights->backend(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        fail(std::string("GGML graph execution failed: ") + ggml_status_to_string(status));
    }

    const std::size_t output_elements = static_cast<std::size_t>(hparams.token_output_size()) * audio_tokens;
    lm_output result;
    result.token_count = audio_tokens;
    result.vocabulary_size = hparams.token_output_size();
    result.mixed_logits.resize(output_elements);
    result.vocal_logits.resize(output_elements);
    result.bgm_logits.resize(output_elements);
    const std::size_t output_bytes = output_elements * sizeof(float);
    ggml_backend_tensor_get(mixed_audio_logits, result.mixed_logits.data(), 0, output_bytes);
    ggml_backend_tensor_get(vocal_audio_logits, result.vocal_logits.data(), 0, output_bytes);
    ggml_backend_tensor_get(bgm_audio_logits, result.bgm_logits.data(), 0, output_bytes);
    if (capture_trace) {
        const auto read_tensor = [this](ggml_tensor * tensor) {
            std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            return values;
        };
        result.main_layers.reserve(main_trace.size());
        for (ggml_tensor * tensor : main_trace) result.main_layers.push_back(read_tensor(tensor));
        result.main_layer0_ops.reserve(main_layer0_ops.tensors.size());
        for (const auto & entry : main_layer0_ops.tensors) {
            result.main_layer0_ops.push_back({entry.first, read_tensor(entry.second)});
        }
        result.main_norm = read_tensor(main_norm_trace);
        result.bridge = read_tensor(bridge_trace);
        result.detail_layers.reserve(detail_trace.size());
        for (ggml_tensor * tensor : detail_trace) result.detail_layers.push_back(read_tensor(tensor));
        result.detail_norm = read_tensor(detail_norm_trace);
    }

    impl_->cache_tokens = total_tokens;
    impl_->next_absolute_position = total_tokens;
    return result;
}

void kv_session::prefill_conditioned_prefix(const condition_pair & condition) {
    if (impl_->cache_tokens != 0) {
        fail("conditioning can only be prefetched into an empty KV session");
    }
    const model_hparams & hparams = impl_->weights->hparams();
    validate_condition(condition, hparams);
    const int32_t prefix_length = condition.main.prefix;
    // A prefix-only graph has no token-ID inputs; reject an empty condition
    // rather than constructing any zero-sized GGML tensors.
    if (prefix_length == 0) {
        fail("prefix-only conditioning requires at least one condition position");
    }
    if (prefix_length > impl_->main_cache.capacity) {
        fail("condition prefix exceeds the KV cache context length");
    }

    const std::size_t context_size = graph_capacity * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(graph_capacity, false);
    context_ptr context(ggml_init({context_size, nullptr, true}));
    if (!context) {
        fail("cannot create conditioned-prefix GGML graph context");
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    if (!graph) {
        fail("cannot create conditioned-prefix GGML computation graph");
    }

    // Dense main/detail conditions and their absolute positions are the only
    // inputs.  In particular, this path must not create empty audio-ID or
    // audio-embedding tensors.
    context_ptr input_context(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
    if (!input_context) {
        fail("cannot create conditioned-prefix GGML input context");
    }
    ggml_tensor * main_condition = ggml_new_tensor_2d(input_context.get(), GGML_TYPE_F32,
                                                       hparams.embedding_length, prefix_length);
    ggml_tensor * detail_condition = ggml_new_tensor_2d(input_context.get(), GGML_TYPE_F32,
                                                         hparams.embedding_length, prefix_length);
    ggml_tensor * positions = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, prefix_length);
    if (!main_condition || !detail_condition || !positions) {
        fail("cannot create conditioned-prefix GGML input tensors");
    }
    ggml_set_name(main_condition, "levo.kv.prefix_main_condition");
    ggml_set_name(detail_condition, "levo.kv.prefix_detail_condition");
    ggml_set_name(positions, "levo.kv.prefix_positions");
    ggml_set_input(main_condition);
    ggml_set_input(detail_condition);
    ggml_set_input(positions);
    buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(input_context.get(), impl_->weights->backend()));
    if (!input_buffer) {
        fail("cannot allocate conditioned-prefix GGML input tensors");
    }

    // Retain the input handles for upload: FP16 emulation may replace the
    // graph values below with cast nodes.
    ggml_tensor * const main_condition_input = main_condition;
    ggml_tensor * const detail_condition_input = detail_condition;
    main_condition = round_f16(context.get(), main_condition, impl_->emulate_fp16_activations);
    detail_condition = round_f16(context.get(), detail_condition, impl_->emulate_fp16_activations);
    ggml_tensor * main_hidden = cached_llama_tower(context.get(), graph, main_condition, positions,
                                                   impl_->main, impl_->main_cache, hparams,
                                                   hparams.main_rope_theta, 0,
                                                   impl_->emulate_fp16_activations);

    // The two dense streams are time-aligned.  Concatenate detail condition
    // and normalized main hidden in width only after the main tower, exactly
    // as in the combined condition+audio path.
    ggml_tensor * detail_input = ggml_concat(context.get(), detail_condition, main_hidden, 0);
    detail_input = round_f16(context.get(), add_bias(context.get(),
        round_f16(context.get(), ggml_mul_mat(context.get(), impl_->bridge_0_weight, detail_input),
                  impl_->emulate_fp16_activations), impl_->bridge_0_bias),
        impl_->emulate_fp16_activations);
    detail_input = round_f16(context.get(), ggml_gelu_erf(context.get(), detail_input),
                             impl_->emulate_fp16_activations);
    detail_input = round_f16(context.get(), add_bias(context.get(),
        round_f16(context.get(), ggml_mul_mat(context.get(), impl_->bridge_2_weight, detail_input),
                  impl_->emulate_fp16_activations), impl_->bridge_2_bias),
        impl_->emulate_fp16_activations);
    ggml_tensor * detail_hidden = cached_llama_tower(context.get(), graph, detail_input, positions,
                                                     impl_->detail, impl_->detail_cache, hparams,
                                                     hparams.detail_rope_theta, 0,
                                                     impl_->emulate_fp16_activations);

    // A final hidden-state graph root is sufficient to make all cache writes
    // and both towers executable.  No LM head is built or returned here.
    ggml_set_name(detail_hidden, "levo.kv.conditioned_prefix_hidden");
    ggml_set_output(detail_hidden);
    ggml_build_forward_expand(graph, detail_hidden);
    allocator_ptr allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->weights->backend())));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        fail("cannot allocate the conditioned-prefix GGML computation graph");
    }

    std::vector<int32_t> position_values(static_cast<std::size_t>(prefix_length));
    for (int32_t index = 0; index < prefix_length; ++index) {
        position_values[static_cast<std::size_t>(index)] = index;
    }
    const std::size_t condition_bytes = condition.main.values.size() * sizeof(float);
    ggml_backend_tensor_set(main_condition_input, condition.main.data(), 0, condition_bytes);
    ggml_backend_tensor_set(detail_condition_input, condition.detail.data(), 0, condition_bytes);
    ggml_backend_tensor_set(positions, position_values.data(), 0,
                            position_values.size() * sizeof(int32_t));
    const ggml_status status = ggml_backend_graph_compute(impl_->weights->backend(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        fail(std::string("GGML graph execution failed: ") + ggml_status_to_string(status));
    }

    impl_->cache_tokens = prefix_length;
    impl_->next_absolute_position = prefix_length;
}

lm_output kv_session::decode(int32_t mixed_token, int32_t vocal_token, int32_t bgm_token) {
    lm_input input;
    input.mixed_tokens = {mixed_token};
    input.vocal_tokens = {vocal_token};
    input.bgm_tokens = {bgm_token};
    return prefill(input);
}

void kv_session::reset() noexcept {
    impl_->cache_tokens = 0;
    impl_->next_absolute_position = 0;
}

int32_t kv_session::token_count() const noexcept {
    return impl_->cache_tokens;
}

int32_t kv_session::next_position() const noexcept {
    return impl_->next_absolute_position;
}

int32_t kv_session::capacity() const noexcept {
    return impl_->main_cache.capacity;
}

} // namespace levo::detail
