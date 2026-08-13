#include "levo-flow-estimator.h"

#include "levo-quantization.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace levo::flow {
namespace {

constexpr int kGraphCapacity = 32768;
constexpr float kLayerNormEpsilon = 1.0e-5F;

struct context_deleter {
    void operator()(ggml_context * context) const noexcept { if (context) ggml_free(context); }
};
struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept { if (buffer) ggml_backend_buffer_free(buffer); }
};
struct allocator_deleter {
    void operator()(ggml_gallocr_t allocator) const noexcept { if (allocator) ggml_gallocr_free(allocator); }
};
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using allocator_ptr = std::unique_ptr<ggml_gallocr, allocator_deleter>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo Flow estimator: " + message);
}

std::size_t checked_product(std::initializer_list<std::size_t> values, const char * what) {
    std::size_t result = 1;
    for (const std::size_t value : values) {
        if (value && result > std::numeric_limits<std::size_t>::max() / value) {
            fail(std::string(what) + " size overflow");
        }
        result *= value;
    }
    return result;
}

void require_finite(const std::vector<float> & values, const char * name) {
    for (const float value : values) {
        if (!std::isfinite(value)) fail(std::string(name) + " contains a non-finite value");
    }
}

ggml_tensor * weight(const std::shared_ptr<const model> & model, const std::string & name) {
    ggml_tensor * tensor = model->tensor(name);
    if (!tensor) fail("missing model tensor '" + name + "'");
    if (tensor->type == GGML_TYPE_F32) return tensor;
    if (quantization::is_quantized(tensor->type) && quantization::flow_block_matrix(name)) return tensor;
    if (tensor->type != GGML_TYPE_F32) {
        fail("tensor '" + name + "' must be F32 or a declared quantized Flow block matrix");
    }
    return tensor;
}

ggml_tensor * add_bias(ggml_context * context, ggml_tensor * value, ggml_tensor * bias) {
    return ggml_add(context, value, bias);
}

ggml_tensor * affine(ggml_context * context, ggml_tensor * value,
                     ggml_tensor * matrix, ggml_tensor * bias) {
    if (matrix->ne[0] < value->ne[0]) fail("matrix input width is smaller than its activation");
    if (matrix->ne[0] == value->ne[0]) return add_bias(context, ggml_mul_mat(context, matrix, value), bias);
    if (!quantization::is_quantized(matrix->type)) fail("only quantized Flow matrices may have padded input axes");
    const int64_t padding = matrix->ne[0] - value->ne[0];
    ggml_tensor * padded = ggml_pad(context, value, static_cast<int>(padding), 0, 0, 0);
    padded = ggml_reshape_3d(context, padded, matrix->ne[0], value->ne[1], value->ne[2]);
    return add_bias(context, ggml_mul_mat(context, matrix, padded), bias);
}

// Broadcast a [hidden] vector to an activation shaped [hidden, frames, batch].
ggml_tensor * broadcast_hidden(ggml_context * context, ggml_tensor * vector,
                               ggml_tensor * activation) {
    ggml_tensor * one = ggml_reshape_3d(context, ggml_cont(context, vector), vector->ne[0], 1, vector->ne[1]);
    return ggml_repeat(context, one, activation);
}

ggml_tensor * add_scale_shift(ggml_context * context, ggml_tensor * normalized,
                              ggml_tensor * scale, ggml_tensor * shift) {
    ggml_tensor * one_plus_scale = ggml_scale_bias(context, scale, 1.0F, 1.0F);
    return ggml_add(context, ggml_mul(context, normalized, one_plus_scale), shift);
}

ggml_tensor * modulation_chunk(ggml_context * context, ggml_tensor * time_modulation,
                               ggml_tensor * table, int chunk, ggml_tensor * activation) {
    const int64_t hidden = table->ne[0];
    ggml_tensor * dynamic = ggml_view_2d(context, time_modulation, hidden,
                                         time_modulation->ne[1], time_modulation->nb[1],
                                         static_cast<std::size_t>(chunk) * static_cast<std::size_t>(hidden) * sizeof(float));
    ggml_tensor * learned = ggml_view_1d(context, table, hidden,
                                         static_cast<std::size_t>(chunk) * static_cast<std::size_t>(hidden) * sizeof(float));
    return broadcast_hidden(context, ggml_add(context, dynamic, learned), activation);
}

ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * value,
                         ggml_tensor * gamma, ggml_tensor * beta) {
    return add_bias(context, ggml_mul(context, ggml_norm(context, value, kLayerNormEpsilon), gamma), beta);
}

ggml_tensor * attention(ggml_context * context, ggml_tensor * qkv,
                        ggml_tensor * positions, const flow_hparams & hp,
                        ggml_tensor * mask) {
    const int64_t hidden = hp.hidden_size;
    const int64_t frames = qkv->ne[1];
    const int64_t batch = qkv->ne[2];
    const int64_t heads = hp.n_head;
    const int64_t head_dim = hp.head_dim;
    ggml_tensor * q = ggml_view_3d(context, qkv, hidden, frames, batch, qkv->nb[1], qkv->nb[2], 0);
    ggml_tensor * k = ggml_view_3d(context, qkv, hidden, frames, batch, qkv->nb[1], qkv->nb[2], static_cast<std::size_t>(hidden) * sizeof(float));
    ggml_tensor * v = ggml_view_3d(context, qkv, hidden, frames, batch, qkv->nb[1], qkv->nb[2], 2 * static_cast<std::size_t>(hidden) * sizeof(float));

    q = ggml_rope_ext(context, ggml_reshape_4d(context, ggml_cont(context, q), head_dim, heads, frames, batch),
                      positions, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, hp.max_frames,
                      hp.rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
    k = ggml_rope_ext(context, ggml_reshape_4d(context, ggml_cont(context, k), head_dim, heads, frames, batch),
                      positions, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, hp.max_frames,
                      hp.rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);

    q = ggml_permute(context, q, 0, 2, 1, 3); // [head_dim, frames, heads, batch]
    k = ggml_permute(context, k, 0, 2, 1, 3);
    ggml_tensor * scores = ggml_mul_mat(context, k, q); // [frames, frames, heads, batch]
    scores = ggml_add(context, scores, mask);
    scores = ggml_soft_max_ext(context, scores, nullptr,
                               1.0F / std::sqrt(static_cast<float>(head_dim)), 0.0F);

    ggml_tensor * value = ggml_permute(context,
                                       ggml_reshape_4d(context, ggml_cont(context, v), head_dim, heads, frames, batch),
                                       1, 2, 0, 3); // [heads, frames, head_dim, batch]
    value = ggml_cont_4d(context, value, frames, head_dim, heads, batch);
    ggml_tensor * attended = ggml_mul_mat(context, value, scores); // [head_dim, frames, heads, batch]
    attended = ggml_permute(context, attended, 0, 2, 1, 3);
    return ggml_cont_3d(context, attended, hidden, frames, batch);
}

struct graph_result {
    ggml_tensor * frequencies = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * timestep_embedding = nullptr;
    ggml_tensor * timestep_modulation = nullptr;
    ggml_tensor * block0_input = nullptr;
    ggml_tensor * block0_output = nullptr;
};

graph_result build_graph(ggml_context * context, ggml_tensor * input,
                         ggml_tensor * timesteps, ggml_tensor * positions,
                         ggml_tensor * mask, const std::shared_ptr<const model> & model) {
    const flow_hparams & hp = model->hparams();
    const int64_t frames = input->ne[1];
    const int64_t batch = input->ne[2];
    const int64_t hidden = hp.hidden_size;
    const int64_t half = hp.time_embedding_dim / 2;

    ggml_tensor * position_embedding = weight(model, "flow.position_embedding.weight");
    ggml_tensor * position_ids = positions;
    ggml_tensor * current = ggml_add(context, input,
                                     ggml_repeat(context,
                                                 ggml_reshape_3d(context,
                                                                 ggml_get_rows(context, position_embedding, position_ids),
                                                                 hidden, frames, 1),
                                                 input));

    // Official Flow timestep embedding: sinusoidal(t * 1000), two linear
    // layers with SiLU between them, and a second SiLU before modulation.
    ggml_tensor * frequencies = ggml_new_tensor_1d(context, GGML_TYPE_F32, half);
    std::vector<float> frequency_values(static_cast<std::size_t>(half));
    for (int64_t i = 0; i < half; ++i) {
        frequency_values[static_cast<std::size_t>(i)] = std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(half));
    }
    ggml_set_input(frequencies);
    ggml_tensor * sinus_target = ggml_new_tensor_2d(context, GGML_TYPE_F32, half, batch);
    ggml_tensor * scaled_t = ggml_scale(context,
                                        ggml_reshape_2d(context, timesteps, 1, batch),
                                        hp.time_embedding_scale);
    ggml_tensor * arguments = ggml_mul(context,
                                       ggml_repeat(context, ggml_reshape_2d(context, frequencies, half, 1), sinus_target),
                                       ggml_repeat(context, scaled_t, sinus_target));
    ggml_tensor * sinusoid = ggml_concat(context, ggml_cos(context, arguments), ggml_sin(context, arguments), 0);
    ggml_tensor * timestep_embedding = affine(
        context,
        ggml_silu(context, affine(context, sinusoid,
                                  weight(model, "flow.time_embedding.linear_1.weight"),
                                  weight(model, "flow.time_embedding.linear_1.bias"))),
        weight(model, "flow.time_embedding.linear_2.weight"),
        weight(model, "flow.time_embedding.linear_2.bias"));
    ggml_tensor * timestep_modulation = affine(
        context, ggml_silu(context, timestep_embedding),
        weight(model, "flow.time_modulation.weight"),
        weight(model, "flow.time_modulation.bias"));

    graph_result result;
    result.frequencies = frequencies;
    result.timestep_embedding = timestep_embedding;
    result.timestep_modulation = timestep_modulation;

    for (int layer_index = 0; layer_index < hp.n_layer; ++layer_index) {
        const std::string root = "flow.block." + std::to_string(layer_index);
        if (layer_index == 0) result.block0_input = current;
        ggml_tensor * table = weight(model, root + ".modulation.weight");
        ggml_tensor * residual = current;
        ggml_tensor * normalized = layer_norm(context, current,
                                              weight(model, root + ".norm_1.weight"),
                                              weight(model, root + ".norm_1.bias"));
        normalized = add_scale_shift(context, normalized,
                                     modulation_chunk(context, timestep_modulation, table, 1, current),
                                     modulation_chunk(context, timestep_modulation, table, 0, current));
        ggml_tensor * qkv = affine(context, normalized,
                                   weight(model, root + ".attn.qkv.weight"),
                                   weight(model, root + ".attn.qkv.bias"));
        ggml_tensor * attended = attention(context, qkv, positions, hp, mask);
        current = ggml_add(context,
                           ggml_mul(context,
                                    affine(context, attended,
                                           weight(model, root + ".attn.out.weight"),
                                           weight(model, root + ".attn.out.bias")),
                                    modulation_chunk(context, timestep_modulation, table, 2, current)),
                           residual);

        residual = current;
        normalized = layer_norm(context, current,
                                weight(model, root + ".norm_2.weight"),
                                weight(model, root + ".norm_2.bias"));
        normalized = add_scale_shift(context, normalized,
                                     modulation_chunk(context, timestep_modulation, table, 4, current),
                                     modulation_chunk(context, timestep_modulation, table, 3, current));
        ggml_tensor * feed_forward = affine(context, normalized,
                                            weight(model, root + ".ffn.in.weight"),
                                            weight(model, root + ".ffn.in.bias"));
        feed_forward = ggml_gelu(context, feed_forward);
        feed_forward = affine(context, feed_forward,
                              weight(model, root + ".ffn.out.weight"),
                              weight(model, root + ".ffn.out.bias"));
        current = ggml_add(context,
                           ggml_mul(context, feed_forward,
                                    modulation_chunk(context, timestep_modulation, table, 5, current)),
                           residual);
        if (layer_index == 0) result.block0_output = current;
    }

    ggml_tensor * final_table = weight(model, "flow.final_modulation.weight");
    ggml_tensor * final_target = ggml_new_tensor_3d(context, GGML_TYPE_F32, hidden, 2, batch);
    ggml_tensor * final_modulation = ggml_add(
        context,
        ggml_repeat(context, ggml_reshape_3d(context, final_table, hidden, 2, 1), final_target),
        ggml_repeat(context, ggml_reshape_3d(context, timestep_embedding, hidden, 1, batch), final_target));
    ggml_tensor * final_shift = ggml_view_2d(context, final_modulation, hidden, batch,
                                             final_modulation->nb[2], 0);
    ggml_tensor * final_scale = ggml_view_2d(context, final_modulation, hidden, batch,
                                             final_modulation->nb[2], static_cast<std::size_t>(hidden) * sizeof(float));
    current = add_scale_shift(context,
                              layer_norm(context, current,
                                         weight(model, "flow.final_norm.weight"),
                                         weight(model, "flow.final_norm.bias")),
                              broadcast_hidden(context, final_scale, current),
                              broadcast_hidden(context, final_shift, current));
    result.output = affine(context, current,
                           weight(model, "flow.output.weight"),
                           weight(model, "flow.output.bias"));
    return result;
}

void copy_tensor(ggml_tensor * tensor, std::vector<float> & output) {
    const std::size_t bytes = ggml_nbytes(tensor);
    output.resize(bytes / sizeof(float));
    ggml_backend_tensor_get(tensor, output.data(), 0, bytes);
}

} // namespace

struct estimator::impl {
    std::shared_ptr<const model> weights;
};

estimator::estimator(std::shared_ptr<const model> model) : impl_(std::make_unique<impl>()) {
    if (!model) fail("cannot create an estimator without a model");
    impl_->weights = std::move(model);
    const flow_hparams & hp = impl_->weights->hparams();
    std::string reason;
    if (!hp.valid(&reason)) fail("invalid Flow hparams: " + reason);
    if (hp.time_embedding_dim <= 0 || hp.time_embedding_dim % 2 != 0) fail("time_embedding_dim must be positive and even");
    if (hp.head_dim % 2 != 0) fail("head_dim must be even for adjacent-pair RoPE");
    // Force the complete inventory/type check at construction time, so a
    // partially converted GGUF cannot fail after graph allocation.
    const char * required[] = {
        "flow.position_embedding.weight", "flow.time_embedding.linear_1.weight",
        "flow.time_embedding.linear_1.bias", "flow.time_embedding.linear_2.weight",
        "flow.time_embedding.linear_2.bias", "flow.time_modulation.weight",
        "flow.time_modulation.bias", "flow.final_norm.weight", "flow.final_norm.bias",
        "flow.final_modulation.weight", "flow.output.weight", "flow.output.bias",
    };
    for (const char * name : required) (void) weight(impl_->weights, name);
    for (int i = 0; i < hp.n_layer; ++i) {
        const std::string root = "flow.block." + std::to_string(i);
        for (const char * suffix : {"attn.qkv.weight", "attn.qkv.bias", "attn.out.weight", "attn.out.bias",
                                    "norm_1.weight", "norm_1.bias", "norm_2.weight", "norm_2.bias",
                                    "ffn.in.weight", "ffn.in.bias", "ffn.out.weight", "ffn.out.bias",
                                    "modulation.weight"}) {
            (void) weight(impl_->weights, root + "." + suffix);
        }
    }
}

estimator::~estimator() = default;
estimator::estimator(estimator &&) noexcept = default;
estimator & estimator::operator=(estimator &&) noexcept = default;

std::unique_ptr<estimator> estimator::create(std::shared_ptr<const model> model) {
    return std::unique_ptr<estimator>(new estimator(std::move(model)));
}

std::vector<float> estimator::velocity(const estimator_input & input, estimator_capture * capture) const {
    if (!impl_ || !impl_->weights) fail("estimator is not initialized");
    const flow_hparams & hp = impl_->weights->hparams();
    if (input.batch == 0 || input.frames == 0) fail("batch and frames must be positive");
    if (input.frames > static_cast<std::size_t>(hp.max_frames)) fail("frames exceed Flow max_frames");
    const std::size_t input_elements = checked_product({input.batch, input.frames, static_cast<std::size_t>(hp.hidden_size)}, "model_input");
    const std::size_t mask_elements = checked_product({input.batch, input.frames, input.frames}, "attention_mask");
    if (input.model_input.size() != input_elements) fail("model_input must have shape [batch, frames, hidden_size]");
    if (input.timesteps.size() != input.batch) fail("timesteps must have shape [batch]");
    if (!input.attention_mask.empty() && input.attention_mask.size() != mask_elements) fail("attention_mask must have shape [batch, frames, frames]");
    require_finite(input.model_input, "model_input");
    require_finite(input.timesteps, "timesteps");
    if (!input.attention_mask.empty()) require_finite(input.attention_mask, "attention_mask");
    if (input.batch > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) || input.frames > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) fail("input dimensions exceed GGML limits");

    const std::size_t graph_context_size = static_cast<std::size_t>(kGraphCapacity) * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphCapacity, false);
    context_ptr context(ggml_init({graph_context_size, nullptr, true}));
    if (!context) fail("cannot create GGML graph context");
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), kGraphCapacity, false);
    if (!graph) fail("cannot create GGML computation graph");

    context_ptr input_context(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
    if (!input_context) fail("cannot create GGML input context");
    ggml_tensor * model_input = ggml_new_tensor_3d(input_context.get(), GGML_TYPE_F32, hp.hidden_size, input.frames, input.batch);
    ggml_tensor * timesteps = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_F32, input.batch);
    ggml_tensor * positions = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, input.frames);
    ggml_tensor * mask = ggml_new_tensor_4d(input_context.get(), GGML_TYPE_F32, input.frames, input.frames, 1, input.batch);
    if (!model_input || !timesteps || !positions || !mask) fail("cannot create GGML estimator inputs");
    ggml_set_input(model_input); ggml_set_input(timesteps); ggml_set_input(positions); ggml_set_input(mask);
    buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(input_context.get(), impl_->weights->backend()));
    if (!input_buffer) fail("cannot allocate GGML estimator inputs");

    graph_result roots = build_graph(context.get(), model_input, timesteps, positions, mask, impl_->weights);
    ggml_set_output(roots.output);
    if (capture) {
        ggml_set_output(roots.timestep_embedding);
        ggml_set_output(roots.timestep_modulation);
        ggml_set_output(roots.block0_input);
        ggml_set_output(roots.block0_output);
    }
    ggml_build_forward_expand(graph, roots.output);
    allocator_ptr allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->weights->backend())));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) fail("cannot allocate GGML estimator graph");

    std::vector<int32_t> position_values(input.frames);
    for (std::size_t i = 0; i < input.frames; ++i) position_values[i] = static_cast<int32_t>(i);
    std::vector<float> zero_mask(mask_elements, 0.0F);
    ggml_backend_tensor_set(model_input, input.model_input.data(), 0, input.model_input.size() * sizeof(float));
    ggml_backend_tensor_set(timesteps, input.timesteps.data(), 0, input.timesteps.size() * sizeof(float));
    std::vector<float> frequency_values(static_cast<std::size_t>(hp.time_embedding_dim / 2));
    for (std::size_t i = 0; i < frequency_values.size(); ++i) {
        frequency_values[i] = std::exp(-std::log(10000.0F) * static_cast<float>(i) /
                                       static_cast<float>(frequency_values.size()));
    }
    ggml_backend_tensor_set(roots.frequencies, frequency_values.data(), 0, frequency_values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask, input.attention_mask.empty() ? zero_mask.data() : input.attention_mask.data(), 0, mask_elements * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(impl_->weights->backend(), graph);
    if (status != GGML_STATUS_SUCCESS) fail(std::string("GGML graph execution failed: ") + ggml_status_to_string(status));

    std::vector<float> output;
    copy_tensor(roots.output, output);
    if (capture) {
        copy_tensor(roots.timestep_embedding, capture->timestep_embedding);
        copy_tensor(roots.timestep_modulation, capture->timestep_modulation);
        copy_tensor(roots.block0_input, capture->block0_input);
        copy_tensor(roots.block0_output, capture->block0_output);
        capture->full_output = output;
    }
    const std::size_t rows = input.batch * input.frames;
    const std::size_t hidden = static_cast<std::size_t>(hp.hidden_size);
    const std::size_t latent = static_cast<std::size_t>(hp.latent_dim);
    std::vector<float> velocity(rows * latent);
    for (std::size_t row = 0; row < rows; ++row) {
        std::copy_n(output.begin() + static_cast<std::ptrdiff_t>(row * hidden + hidden - latent),
                    latent,
                    velocity.begin() + static_cast<std::ptrdiff_t>(row * latent));
    }
    return velocity;
}

} // namespace levo::flow
