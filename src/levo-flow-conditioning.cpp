#include "levo-flow-conditioning.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace levo::flow {
namespace {

constexpr std::size_t k_graph_capacity = 128;

struct context_deleter { void operator()(ggml_context * value) const noexcept { if (value) ggml_free(value); } };
struct allocator_deleter { void operator()(ggml_gallocr_t value) const noexcept { if (value) ggml_gallocr_free(value); } };
struct buffer_deleter { void operator()(ggml_backend_buffer_t value) const noexcept { if (value) ggml_backend_buffer_free(value); } };
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using allocator_ptr = std::unique_ptr<ggml_gallocr, allocator_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

[[noreturn]] void fail(const std::string & message) { throw std::runtime_error("LeVo Flow conditioning: " + message); }

void require_shape(const ggml_tensor * tensor, const std::string & name, std::initializer_list<int64_t> shape) {
    if (!tensor) fail("missing tensor '" + name + "'");
    if (tensor->type != GGML_TYPE_F32) fail("tensor '" + name + "' must be F32; F16 Flow conditioning is not yet enabled");
    if (ggml_n_dims(tensor) != static_cast<int>(shape.size())) fail("tensor '" + name + "' has an unexpected rank");
    std::size_t i = 0;
    for (const int64_t dimension : shape) if (tensor->ne[i++] != dimension) fail("tensor '" + name + "' has an unexpected shape");
}

ggml_tensor * require_tensor(const model & model, const std::string & name, std::initializer_list<int64_t> shape) {
    ggml_tensor * result = model.tensor(name);
    require_shape(result, name, shape);
    return result;
}

void require_finite(const std::vector<float> & values, const char * label) {
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) fail(std::string(label) + " contains a non-finite value");
}

std::size_t checked_elements(int32_t frames, int32_t width, const char * label) {
    if (frames <= 0 || width <= 0) fail(std::string(label) + " has non-positive dimensions");
    return static_cast<std::size_t>(frames) * static_cast<std::size_t>(width);
}

} // namespace

struct conditioning::impl {
    std::shared_ptr<const model> weights;
    ggml_tensor * vocal_codebook = nullptr;
    ggml_tensor * vocal_projection = nullptr;
    ggml_tensor * vocal_bias = nullptr;
    ggml_tensor * bgm_codebook = nullptr;
    ggml_tensor * bgm_projection = nullptr;
    ggml_tensor * bgm_bias = nullptr;
    ggml_tensor * mask_embedding = nullptr;
    ggml_tensor * position_embedding = nullptr;
    ggml_tensor * null_condition = nullptr;
    ggml_tensor * norm_counts = nullptr;
    ggml_tensor * norm_sum_x = nullptr;
    ggml_tensor * norm_sum_x2 = nullptr;
    std::vector<float> mean;
    std::vector<float> stddev;
};

conditioning::conditioning(std::shared_ptr<const model> weights) : impl_(std::make_unique<impl>()) {
    if (!weights) fail("a Flow model is required");
    const flow_hparams & hp = weights->hparams();
    impl_->weights = std::move(weights);
    const auto tensor = [this, &hp](const std::string & name, std::initializer_list<int64_t> shape) {
        return require_tensor(*impl_->weights, name, shape);
    };
    impl_->vocal_codebook = tensor("flow.rvq.vocal.codebook.weight", {hp.codebook_dim, hp.codebook_size});
    impl_->vocal_projection = tensor("flow.rvq.vocal.out_proj.weight", {hp.codebook_dim, hp.condition_dim});
    impl_->vocal_bias = tensor("flow.rvq.vocal.out_proj.bias", {hp.condition_dim});
    impl_->bgm_codebook = tensor("flow.rvq.bgm.codebook.weight", {hp.codebook_dim, hp.codebook_size});
    impl_->bgm_projection = tensor("flow.rvq.bgm.out_proj.weight", {hp.codebook_dim, hp.condition_dim});
    impl_->bgm_bias = tensor("flow.rvq.bgm.out_proj.bias", {hp.condition_dim});
    impl_->mask_embedding = tensor("flow.mask_embedding.weight", {hp.mask_dim, 3});
    impl_->position_embedding = tensor("flow.position_embedding.weight", {hp.hidden_size, hp.max_frames});
    impl_->null_condition = tensor("flow.null_condition.weight", {hp.condition_dim});
    impl_->norm_counts = tensor("flow.norm.counts", {1});
    impl_->norm_sum_x = tensor("flow.norm.sum_x", {hp.latent_dim});
    impl_->norm_sum_x2 = tensor("flow.norm.sum_x2", {hp.latent_dim});

    float count = 0.0F;
    ggml_backend_tensor_get(impl_->norm_counts, &count, 0, sizeof(count));
    if (!std::isfinite(count) || count < 10.0F) fail("Flow norm counts must be finite and at least 10");
    impl_->mean.resize(static_cast<std::size_t>(hp.latent_dim));
    std::vector<float> sum_x2(static_cast<std::size_t>(hp.latent_dim));
    ggml_backend_tensor_get(impl_->norm_sum_x, impl_->mean.data(), 0, impl_->mean.size() * sizeof(float));
    ggml_backend_tensor_get(impl_->norm_sum_x2, sum_x2.data(), 0, sum_x2.size() * sizeof(float));
    impl_->stddev.resize(impl_->mean.size());
    for (std::size_t i = 0; i < impl_->mean.size(); ++i) {
        impl_->mean[i] /= count;
        const float variance = std::max(0.0F, sum_x2[i] / count - impl_->mean[i] * impl_->mean[i]);
        impl_->stddev[i] = std::sqrt(variance);
        if (!std::isfinite(impl_->mean[i]) || !std::isfinite(impl_->stddev[i]) || impl_->stddev[i] < 1.0e-12F) {
            fail("Flow normalization statistics are invalid");
        }
    }
}

conditioning::~conditioning() = default;
conditioning::conditioning(conditioning &&) noexcept = default;
conditioning & conditioning::operator=(conditioning &&) noexcept = default;

std::vector<float> conditioning::normalize_latents(const std::vector<float> & raw, int32_t frames) const {
    const flow_hparams & hp = impl_->weights->hparams();
    if (raw.size() != checked_elements(frames, hp.latent_dim, "raw latent")) fail("raw latent has an unexpected size");
    require_finite(raw, "raw latent");
    std::vector<float> result(raw.size());
    for (int32_t frame = 0; frame < frames; ++frame) for (int32_t channel = 0; channel < hp.latent_dim; ++channel) {
        const std::size_t index = static_cast<std::size_t>(frame) * hp.latent_dim + channel;
        result[index] = (raw[index] - impl_->mean[static_cast<std::size_t>(channel)]) / impl_->stddev[static_cast<std::size_t>(channel)];
    }
    return result;
}

std::vector<float> conditioning::denormalize_latents(const std::vector<float> & normalized, int32_t frames) const {
    const flow_hparams & hp = impl_->weights->hparams();
    if (normalized.size() != checked_elements(frames, hp.latent_dim, "normalized latent")) fail("normalized latent has an unexpected size");
    require_finite(normalized, "normalized latent");
    std::vector<float> result(normalized.size());
    for (int32_t frame = 0; frame < frames; ++frame) for (int32_t channel = 0; channel < hp.latent_dim; ++channel) {
        const std::size_t index = static_cast<std::size_t>(frame) * hp.latent_dim + channel;
        result[index] = normalized[index] * impl_->stddev[static_cast<std::size_t>(channel)] + impl_->mean[static_cast<std::size_t>(channel)];
    }
    return result;
}

conditioning_output conditioning::prepare(const conditioning_input & input) const {
    const flow_hparams & hp = impl_->weights->hparams();
    if (input.vocal_codes.empty() || input.vocal_codes.size() != input.bgm_codes.size()) fail("vocal and BGM code streams must be non-empty and equally sized");
    if (input.vocal_codes.size() > static_cast<std::size_t>(hp.max_frames)) fail("Flow conditioning exceeds max_frames");
    const int32_t frames = static_cast<int32_t>(input.vocal_codes.size());
    const auto valid_code = [&hp](int32_t code) { return code >= 0 && code < hp.codebook_size; };
    if (!std::all_of(input.vocal_codes.begin(), input.vocal_codes.end(), valid_code) || !std::all_of(input.bgm_codes.begin(), input.bgm_codes.end(), valid_code)) fail("audio code IDs must be in [0, codebook_size)");
    std::vector<int32_t> masks = input.mask_ids;
    if (masks.empty()) masks.assign(static_cast<std::size_t>(frames), 2);
    if (masks.size() != static_cast<std::size_t>(frames) || !std::all_of(masks.begin(), masks.end(), [](int32_t id) { return id >= 0 && id <= 2; })) fail("mask IDs must have one value per frame in [0,2]");
    std::vector<float> raw_context = input.raw_incontext_latent;
    if (raw_context.empty()) raw_context.assign(checked_elements(frames, hp.latent_dim, "raw latent"), 0.0F);
    std::vector<float> noise = input.initial_noise;
    if (noise.empty()) noise.assign(checked_elements(frames, hp.latent_dim, "initial noise"), 0.0F);
    if (noise.size() != checked_elements(frames, hp.latent_dim, "initial noise")) fail("initial noise has an unexpected size");
    require_finite(noise, "initial noise");
    std::vector<float> normalized = normalize_latents(raw_context, frames);
    for (int32_t frame = 0; frame < frames; ++frame) if (masks[static_cast<std::size_t>(frame)] != 1) {
        std::fill_n(normalized.begin() + static_cast<std::size_t>(frame) * hp.latent_dim, hp.latent_dim, 0.0F);
    }

    const std::size_t graph_size = k_graph_capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(k_graph_capacity, false);
    context_ptr context(ggml_init({graph_size, nullptr, true}));
    if (!context) fail("cannot create GGML graph context");
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), k_graph_capacity, false);
    if (!graph) fail("cannot create GGML graph");
    context_ptr inputs(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
    if (!inputs) fail("cannot create GGML input context");
    ggml_tensor * vocal_ids = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_I32, frames);
    ggml_tensor * bgm_ids = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_I32, frames);
    ggml_tensor * mask_ids = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_I32, frames);
    if (!vocal_ids || !bgm_ids || !mask_ids) fail("cannot allocate GGML IDs");
    ggml_set_input(vocal_ids); ggml_set_input(bgm_ids); ggml_set_input(mask_ids);
    buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), impl_->weights->backend()));
    if (!input_buffer) fail("cannot allocate GGML ID inputs");

    ggml_tensor * vocal_lookup = ggml_get_rows(context.get(), impl_->vocal_codebook, vocal_ids);
    ggml_tensor * bgm_lookup = ggml_get_rows(context.get(), impl_->bgm_codebook, bgm_ids);
    ggml_tensor * vocal_projected = ggml_add(context.get(), ggml_mul_mat(context.get(), impl_->vocal_projection, vocal_lookup), impl_->vocal_bias);
    ggml_tensor * bgm_projected = ggml_add(context.get(), ggml_mul_mat(context.get(), impl_->bgm_projection, bgm_lookup), impl_->bgm_bias);
    ggml_tensor * mask_embedding = ggml_get_rows(context.get(), impl_->mask_embedding, mask_ids);
    // Position IDs are graph-local so the graph allocator owns their storage.
    std::vector<int32_t> position_values(static_cast<std::size_t>(frames));
    for (int32_t i = 0; i < frames; ++i) position_values[static_cast<std::size_t>(i)] = i;
    ggml_tensor * position_ids = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, frames);
    ggml_set_input(position_ids);
    ggml_tensor * positional = ggml_get_rows(context.get(), impl_->position_embedding, position_ids);
    ggml_set_output(vocal_lookup); ggml_set_output(bgm_lookup); ggml_set_output(vocal_projected); ggml_set_output(bgm_projected); ggml_set_output(mask_embedding); ggml_set_output(positional);
    ggml_build_forward_expand(graph, vocal_lookup); ggml_build_forward_expand(graph, bgm_lookup); ggml_build_forward_expand(graph, vocal_projected); ggml_build_forward_expand(graph, bgm_projected); ggml_build_forward_expand(graph, mask_embedding); ggml_build_forward_expand(graph, positional);
    allocator_ptr allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->weights->backend())));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) fail("cannot allocate Flow conditioning graph");
    const std::size_t id_bytes = static_cast<std::size_t>(frames) * sizeof(int32_t);
    ggml_backend_tensor_set(vocal_ids, input.vocal_codes.data(), 0, id_bytes);
    ggml_backend_tensor_set(bgm_ids, input.bgm_codes.data(), 0, id_bytes);
    ggml_backend_tensor_set(mask_ids, masks.data(), 0, id_bytes);
    ggml_backend_tensor_set(position_ids, position_values.data(), 0, id_bytes);
    if (ggml_backend_graph_compute(impl_->weights->backend(), graph) != GGML_STATUS_SUCCESS) fail("GGML Flow conditioning graph execution failed");

    conditioning_output output; output.frames = frames; output.normalized_incontext = std::move(normalized); output.initial_noise = std::move(noise);
    output.vocal_codebook_lookup.resize(checked_elements(frames, hp.codebook_dim, "vocal lookup"));
    output.bgm_codebook_lookup.resize(checked_elements(frames, hp.codebook_dim, "BGM lookup"));
    output.vocal_projected.resize(checked_elements(frames, hp.condition_dim, "vocal condition"));
    output.bgm_projected.resize(checked_elements(frames, hp.condition_dim, "BGM condition"));
    output.mask_embedding.resize(checked_elements(frames, hp.mask_dim, "mask embedding"));
    output.positional_embedding.resize(checked_elements(frames, hp.hidden_size, "position embedding"));
    output.null_condition.resize(static_cast<std::size_t>(hp.condition_dim));
    ggml_backend_tensor_get(vocal_lookup, output.vocal_codebook_lookup.data(), 0, output.vocal_codebook_lookup.size() * sizeof(float));
    ggml_backend_tensor_get(bgm_lookup, output.bgm_codebook_lookup.data(), 0, output.bgm_codebook_lookup.size() * sizeof(float));
    ggml_backend_tensor_get(vocal_projected, output.vocal_projected.data(), 0, output.vocal_projected.size() * sizeof(float));
    ggml_backend_tensor_get(bgm_projected, output.bgm_projected.data(), 0, output.bgm_projected.size() * sizeof(float));
    ggml_backend_tensor_get(mask_embedding, output.mask_embedding.data(), 0, output.mask_embedding.size() * sizeof(float));
    ggml_backend_tensor_get(positional, output.positional_embedding.data(), 0, output.positional_embedding.size() * sizeof(float));
    ggml_backend_tensor_get(impl_->null_condition, output.null_condition.data(), 0, output.null_condition.size() * sizeof(float));
    output.conditioning.resize(checked_elements(frames, 2 * hp.condition_dim, "conditioning"));
    for (int32_t frame = 0; frame < frames; ++frame) {
        const std::size_t destination = static_cast<std::size_t>(frame) * 2 * hp.condition_dim;
        const std::size_t source = static_cast<std::size_t>(frame) * hp.condition_dim;
        const float * vocal = masks[static_cast<std::size_t>(frame)] == 0 ? output.null_condition.data() : output.vocal_projected.data() + source;
        const float * bgm = masks[static_cast<std::size_t>(frame)] == 0 ? output.null_condition.data() : output.bgm_projected.data() + source;
        std::copy_n(vocal, hp.condition_dim, output.conditioning.begin() + destination);
        std::copy_n(bgm, hp.condition_dim, output.conditioning.begin() + destination + hp.condition_dim);
    }
    return output;
}

} // namespace levo::flow
