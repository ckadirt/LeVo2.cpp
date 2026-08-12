#include "levo-vae.h"

#include "levo-audio-ops.h"

#include "ggml-alloc.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace levo::detail {
namespace {

constexpr std::size_t graph_capacity = 2048;

struct context_deleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) ggml_free(context);
    }
};

struct allocator_deleter {
    void operator()(ggml_gallocr_t allocator) const noexcept {
        if (allocator != nullptr) ggml_gallocr_free(allocator);
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
    }
};

using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using allocator_ptr = std::unique_ptr<ggml_gallocr, allocator_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo VAE: " + message);
}

void require_f32(const ggml_tensor * tensor, const char * label) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        fail(std::string(label) + " must be present and F32 for the correctness graph");
    }
}

void validate_weights(const vae_decoder_weights & weights) {
    const auto snake = [](const vae_snake_weights & value) {
        require_f32(value.alpha_log, "SnakeBeta alpha");
        require_f32(value.beta_log, "SnakeBeta beta");
    };
    const auto conv = [](const vae_conv_weights & value, bool require_bias) {
        require_f32(value.weight, "convolution weight");
        if (require_bias) require_f32(value.bias, "convolution bias");
    };
    conv(weights.input, true);
    for (const auto & stage : weights.stage) {
        snake(stage.upsample.activation);
        conv(stage.upsample.conv, true);
        for (const auto & residual : stage.residual) {
            snake(residual.pre);
            conv(residual.conv_0, true);
            snake(residual.post);
            conv(residual.conv_1, true);
        }
    }
    snake(weights.output_activation);
    require_f32(weights.output_weight, "output convolution weight");
}

ggml_tensor * run_snake(ggml_context * context,
                        ggml_tensor * input,
                        const vae_snake_weights & weights) {
    ggml_tensor * alpha = ggml_reshape_3d(
        context, weights.alpha_log, 1, weights.alpha_log->ne[0], 1);
    ggml_tensor * beta = ggml_reshape_3d(
        context, weights.beta_log, 1, weights.beta_log->ne[0], 1);
    return snake_beta(context, input, alpha, beta);
}

ggml_tensor * run_residual(ggml_context * context,
                           ggml_tensor * input,
                           const vae_residual_weights & weights,
                           int dilation) {
    ggml_tensor * hidden = run_snake(context, input, weights.pre);
    hidden = conv_1d_f32(context, weights.conv_0.weight, hidden,
                         weights.conv_0.bias, 1, 3 * dilation, dilation);
    hidden = run_snake(context, hidden, weights.post);
    hidden = conv_1d_f32(context, weights.conv_1.weight, hidden,
                         weights.conv_1.bias, 1, 0, 1);
    return ggml_add(context, input, hidden);
}

std::vector<float> read_f32(ggml_tensor * tensor) {
    std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

} // namespace

struct vae_decoder::impl {
    std::shared_ptr<const vae_model> weights;
};

vae_decoder::vae_decoder(std::shared_ptr<const vae_model> model)
    : impl_(std::make_unique<impl>()) {
    if (!model) fail("a VAE model is required");
    validate_weights(model->decoder());
    impl_->weights = std::move(model);
}

vae_decoder::~vae_decoder() = default;
vae_decoder::vae_decoder(vae_decoder &&) noexcept = default;
vae_decoder & vae_decoder::operator=(vae_decoder &&) noexcept = default;

std::unique_ptr<vae_decoder> vae_decoder::create(std::shared_ptr<const vae_model> model) {
    return std::unique_ptr<vae_decoder>(new vae_decoder(std::move(model)));
}

vae_decode_result vae_decoder::decode(const std::vector<float> & latent,
                                      std::size_t frames,
                                      bool capture_stages) const {
    const vae_hparams & hparams = impl_->weights->hparams();
    if (frames == 0 || frames > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
        fail("latent frame count must be positive and representable");
    }
    if (frames > std::numeric_limits<std::size_t>::max() /
                     static_cast<std::size_t>(hparams.latent_dim) ||
        latent.size() != frames * static_cast<std::size_t>(hparams.latent_dim)) {
        fail("latent payload does not have shape [64,frames]");
    }
    if (frames > std::numeric_limits<std::size_t>::max() /
                     static_cast<std::size_t>(hparams.downsampling_ratio)) {
        fail("decoded sample count overflows size_t");
    }
    for (const float value : latent) {
        if (!std::isfinite(value)) fail("latent payload contains a non-finite value");
    }

    const std::size_t context_size = graph_capacity * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(graph_capacity, false);
    context_ptr context(ggml_init({context_size, nullptr, true}));
    if (!context) fail("cannot create GGML graph context");
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    if (!graph) fail("cannot create GGML computation graph");

    context_ptr input_context(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    if (!input_context) fail("cannot create GGML input context");
    ggml_tensor * input = ggml_new_tensor_3d(
        input_context.get(), GGML_TYPE_F32, static_cast<int64_t>(frames),
        hparams.latent_dim, 1);
    if (!input) fail("cannot create VAE latent input tensor");
    ggml_set_input(input);
    buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(
        input_context.get(), impl_->weights->backend()));
    if (!input_buffer) fail("cannot allocate VAE input tensor");

    const vae_decoder_weights & weights = impl_->weights->decoder();
    ggml_tensor * hidden = conv_1d_f32(context.get(), weights.input.weight,
                                       input, weights.input.bias, 1, 3, 1);
    std::array<ggml_tensor *, 5> stages{};
    const std::array<int, 5> strides{{10, 6, 4, 4, 2}};
    const std::array<int, 3> dilations{{1, 3, 9}};
    for (std::size_t stage = 0; stage < weights.stage.size(); ++stage) {
        hidden = run_snake(context.get(), hidden,
                           weights.stage[stage].upsample.activation);
        hidden = conv_transpose_1d_f32(
            context.get(), weights.stage[stage].upsample.conv.weight,
            hidden, weights.stage[stage].upsample.conv.bias,
            strides[stage], (strides[stage] + 1) / 2);
        for (std::size_t residual = 0;
             residual < weights.stage[stage].residual.size(); ++residual) {
            hidden = run_residual(context.get(), hidden,
                                  weights.stage[stage].residual[residual],
                                  dilations[residual]);
        }
        stages[stage] = hidden;
        if (capture_stages) ggml_set_output(hidden);
    }
    hidden = run_snake(context.get(), hidden, weights.output_activation);
    ggml_tensor * output = conv_1d_f32(context.get(), weights.output_weight,
                                       hidden, nullptr, 1, 3, 1);
    ggml_set_output(output);
    if (capture_stages) {
        for (ggml_tensor * stage : stages) ggml_build_forward_expand(graph, stage);
    }
    ggml_build_forward_expand(graph, output);

    allocator_ptr allocator(ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(impl_->weights->backend())));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        fail("cannot allocate VAE computation graph");
    }
    ggml_backend_tensor_set(input, latent.data(), 0, latent.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(impl_->weights->backend(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        fail(std::string("GGML graph execution failed: ") + ggml_status_to_string(status));
    }

    const std::size_t sample_count = frames *
                                     static_cast<std::size_t>(hparams.downsampling_ratio);
    if (output->ne[0] != static_cast<int64_t>(sample_count) ||
        output->ne[1] != hparams.audio_channels || output->ne[2] != 1 ||
        output->type != GGML_TYPE_F32) {
        fail("VAE output shape does not match [2,frames*1920]");
    }
    vae_decode_result result;
    result.frames = frames;
    result.samples_per_channel = sample_count;
    result.audio = read_f32(output);
    if (capture_stages) {
        result.stage_outputs.reserve(stages.size());
        for (ggml_tensor * stage : stages) result.stage_outputs.push_back(read_f32(stage));
    }
    return result;
}

} // namespace levo::detail
