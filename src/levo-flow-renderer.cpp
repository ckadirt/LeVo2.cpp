#include "levo-flow-renderer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace levo::flow {
namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo Flow renderer: " + message);
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char * label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        fail(std::string(label) + " size overflow");
    }
    return lhs * rhs;
}

void require_finite(const std::vector<float> & values, const char * label) {
    for (const float value : values) {
        if (!std::isfinite(value)) fail(std::string(label) + " contains a non-finite value");
    }
}

std::size_t element_count(std::size_t frames, std::size_t width, const char * label) {
    if (frames == 0 || width == 0) fail(std::string(label) + " dimensions must be positive");
    return checked_product(frames, width, label);
}

void append_model_row(std::vector<float> & destination, const conditioning_output & condition,
                      const std::vector<float> & state, std::size_t frame,
                      bool unconditional, const flow_hparams & hp) {
    const std::size_t mask_offset = frame * static_cast<std::size_t>(hp.mask_dim);
    const std::size_t context_offset = frame * static_cast<std::size_t>(hp.latent_dim);
    const std::size_t condition_offset = frame * 2U * static_cast<std::size_t>(hp.condition_dim);
    destination.insert(destination.end(), condition.mask_embedding.begin() + mask_offset,
                       condition.mask_embedding.begin() + mask_offset + hp.mask_dim);
    destination.insert(destination.end(), condition.normalized_incontext.begin() + context_offset,
                       condition.normalized_incontext.begin() + context_offset + hp.latent_dim);
    if (unconditional) {
        destination.insert(destination.end(), 2U * static_cast<std::size_t>(hp.condition_dim), 0.0F);
    } else {
        destination.insert(destination.end(), condition.conditioning.begin() + condition_offset,
                           condition.conditioning.begin() + condition_offset + 2U * hp.condition_dim);
    }
    destination.insert(destination.end(), state.begin() + context_offset,
                       state.begin() + context_offset + hp.latent_dim);
}

} // namespace

std::vector<float> solve_flow_euler(const euler_input & input, std::size_t steps,
                                    const velocity_callback & velocity) {
    if (steps == 0) fail("Euler step count must be positive");
    if (!velocity) fail("Euler velocity callback is required");
    const std::size_t elements = element_count(input.frames, input.latent_dim, "Euler state");
    if (input.initial_noise.size() != elements || input.normalized_incontext.size() != elements) {
        fail("Euler tensors must have shape [frames, latent_dim]");
    }
    if (input.incontext_frames > input.frames) fail("Euler in-context length exceeds frame count");
    if (!std::isfinite(input.sigma_min) || input.sigma_min <= 0.0F) fail("Euler sigma_min must be finite and positive");
    require_finite(input.initial_noise, "Euler noise");
    require_finite(input.normalized_incontext, "Euler in-context latent");

    const float dt = 1.0F / static_cast<float>(steps);
    std::vector<float> state = input.initial_noise;
    for (std::size_t step = 0; step < steps; ++step) {
        const float time = static_cast<float>(step) * dt;
        const float noise_scale = 1.0F - (1.0F - input.sigma_min) * time;
        for (std::size_t frame = 0; frame < input.incontext_frames; ++frame) {
            const std::size_t offset = frame * input.latent_dim;
            for (std::size_t channel = 0; channel < input.latent_dim; ++channel) {
                state[offset + channel] = noise_scale * input.initial_noise[offset + channel] +
                                          time * input.normalized_incontext[offset + channel];
            }
        }
        const std::vector<float> current_velocity = velocity(state, time);
        if (current_velocity.size() != elements) fail("Euler velocity callback returned an unexpected shape");
        require_finite(current_velocity, "Euler velocity");
        for (std::size_t element = 0; element < elements; ++element) state[element] += dt * current_velocity[element];
        require_finite(state, "Euler state");
    }
    const std::size_t context_elements = input.incontext_frames * input.latent_dim;
    std::copy_n(input.normalized_incontext.begin(), context_elements, state.begin());
    return state;
}

std::vector<float> assemble_flow_latent_windows(const std::vector<latent_window> & windows,
                                                 std::size_t source_frames,
                                                 std::size_t window_frames,
                                                 std::size_t hop_frames,
                                                 std::size_t latent_dim) {
    if (windows.empty() || source_frames == 0 || window_frames == 0 || hop_frames == 0 ||
        hop_frames > window_frames || latent_dim == 0) {
        fail("invalid latent window assembly dimensions");
    }
    const std::size_t per_window = element_count(window_frames, latent_dim, "latent window");
    const std::size_t overlap = window_frames - hop_frames;
    const std::size_t continuation = element_count(hop_frames, latent_dim, "latent continuation");
    std::vector<float> assembled;
    assembled.reserve(checked_product(window_frames + (windows.size() - 1U) * hop_frames,
                                      latent_dim, "assembled latent"));
    for (std::size_t index = 0; index < windows.size(); ++index) {
        const latent_window & window = windows[index];
        if (window.input_offset_frames != index * hop_frames) fail("latent window offsets are not contiguous");
        if (window.denormalized_latents.size() != per_window) fail("latent window has an unexpected shape");
        require_finite(window.denormalized_latents, "denormalized latent window");
        const std::size_t start = index == 0 ? 0 : overlap * latent_dim;
        const std::size_t count = index == 0 ? per_window : continuation;
        assembled.insert(assembled.end(), window.denormalized_latents.begin() + start,
                         window.denormalized_latents.begin() + start + count);
    }
    const std::size_t target = element_count(source_frames, latent_dim, "source latent");
    if (assembled.size() < target) fail("latent windows do not cover the source sequence");
    assembled.resize(target);
    return assembled;
}

renderer::renderer(std::shared_ptr<const model> weights)
        : weights_(std::move(weights)), conditioning_(weights_), estimator_(estimator::create(weights_)) {
    if (!weights_) fail("a Flow model is required");
    std::string reason;
    if (!weights_->hparams().valid(&reason)) fail("invalid Flow hparams: " + reason);
}

renderer::~renderer() = default;
renderer::renderer(renderer &&) noexcept = default;
renderer & renderer::operator=(renderer &&) noexcept = default;

render_output renderer::render(const render_input & input, const render_options & options) const {
    if (input.vocal_codes.empty() || input.vocal_codes.size() != input.bgm_codes.size()) {
        fail("vocal and BGM token streams must be non-empty and equally sized");
    }
    if (!weights_ || !estimator_) fail("renderer is not initialized");
    const flow_hparams & hp = weights_->hparams();
    const std::size_t source_frames = input.vocal_codes.size();
    const std::size_t window_frames = static_cast<std::size_t>(hp.window_frames);
    const std::size_t hop_frames = static_cast<std::size_t>(hp.hop_frames);
    const std::size_t overlap_frames = static_cast<std::size_t>(hp.overlap_frames);
    const std::size_t latent_dim = static_cast<std::size_t>(hp.latent_dim);
    if (window_frames != static_cast<std::size_t>(hp.max_frames) || window_frames != hop_frames + overlap_frames) {
        fail("Flow window metadata is inconsistent");
    }
    const std::size_t steps = options.euler_steps == 0 ? static_cast<std::size_t>(hp.euler_steps_default) : options.euler_steps;
    const float guidance = options.guidance_scale == 0.0F ? hp.cfg_default : options.guidance_scale;
    if (steps == 0 || !std::isfinite(guidance) || guidance <= 0.0F) fail("Euler steps and CFG guidance must be positive and finite");
    if (input.initial_incontext_frames > window_frames ||
        input.raw_initial_incontext.size() != checked_product(input.initial_incontext_frames, latent_dim, "initial context")) {
        fail("initial raw in-context latent has an unexpected shape");
    }
    require_finite(input.raw_initial_incontext, "initial raw in-context latent");

    std::size_t padded_frames = source_frames <= window_frames ? window_frames : 0;
    if (source_frames > window_frames) {
        const std::size_t base = source_frames - overlap_frames;
        const std::size_t groups = base / hop_frames + (base % hop_frames != 0 ? 1U : 0U);
        padded_frames = groups * hop_frames + overlap_frames;
    }
    const std::size_t windows_count = 1U + (padded_frames - window_frames) / hop_frames;
    const std::size_t per_window = element_count(window_frames, latent_dim, "Flow noise window");
    const std::size_t total_noise = checked_product(windows_count, per_window, "Flow noise");
    if (input.initial_noise.size() != total_noise) fail("initial_noise must be window-major [scheduled_windows, window_frames, latent_dim]");
    require_finite(input.initial_noise, "initial noise");

    render_output result;
    result.source_frames = source_frames;
    result.padded_frames = padded_frames;
    result.windows.reserve(windows_count);
    std::vector<float> prior_raw;
    for (std::size_t window_index = 0; window_index < windows_count; ++window_index) {
        const std::size_t token_offset = window_index * hop_frames;
        const std::size_t incontext_frames = window_index == 0 ? input.initial_incontext_frames : overlap_frames;
        std::vector<int32_t> vocal(window_frames), bgm(window_frames), masks(window_frames, 2);
        std::vector<float> raw_context(per_window, 0.0F);
        for (std::size_t frame = 0; frame < window_frames; ++frame) {
            const std::size_t source_index = (token_offset + frame) % source_frames;
            vocal[frame] = input.vocal_codes[source_index];
            bgm[frame] = input.bgm_codes[source_index];
        }
        std::fill_n(masks.begin(), incontext_frames, 1);
        if (window_index == 0) {
            std::copy(input.raw_initial_incontext.begin(), input.raw_initial_incontext.end(), raw_context.begin());
        } else {
            if (prior_raw.size() != per_window) fail("prior Flow window is unavailable for continuation");
            const std::size_t tail = (window_frames - overlap_frames) * latent_dim;
            std::copy_n(prior_raw.begin() + tail, overlap_frames * latent_dim, raw_context.begin());
        }
        conditioning_input condition_input;
        condition_input.vocal_codes = std::move(vocal);
        condition_input.bgm_codes = std::move(bgm);
        condition_input.mask_ids = std::move(masks);
        condition_input.raw_incontext_latent = std::move(raw_context);
        condition_input.initial_noise.assign(input.initial_noise.begin() + window_index * per_window,
                                             input.initial_noise.begin() + (window_index + 1U) * per_window);
        const conditioning_output condition = conditioning_.prepare(condition_input);
        if (condition.frames != hp.window_frames) fail("conditioning returned an unexpected frame count");

        const bool cfg = guidance > 1.0F;
        const std::size_t batch = cfg ? 2U : 1U;
        const auto evaluate_velocity = [this, &condition, &hp, batch, cfg, guidance, window_frames, latent_dim](
                                               const std::vector<float> & state, float time) {
            estimator_input estimator_input;
            estimator_input.batch = batch;
            estimator_input.frames = window_frames;
            estimator_input.timesteps.assign(batch, time);
            estimator_input.model_input.reserve(batch * window_frames * static_cast<std::size_t>(hp.hidden_size));
            for (std::size_t row_batch = 0; row_batch < batch; ++row_batch) {
                for (std::size_t frame = 0; frame < window_frames; ++frame) {
                    append_model_row(estimator_input.model_input, condition, state, frame,
                                     cfg && row_batch == 0U, hp);
                }
            }
            const std::vector<float> output = estimator_->velocity(estimator_input);
            const std::size_t complete = checked_product(window_frames, latent_dim, "CFG velocity");
            if (output.size() != checked_product(batch, complete, "estimator velocity")) {
                fail("estimator returned an unexpected velocity shape");
            }
            if (!cfg) return output;
            const std::vector<float> unconditional(output.begin(), output.begin() + complete);
            std::vector<float> conditional(complete);
            std::copy_n(output.begin() + complete, complete, conditional.begin());
            std::vector<float> guided(complete);
            for (std::size_t index = 0; index < complete; ++index) {
                guided[index] = unconditional[index] + guidance * (conditional[index] - unconditional[index]);
            }
            return guided;
        };
        const euler_input solver_input{condition.initial_noise, condition.normalized_incontext,
                                       window_frames, latent_dim, incontext_frames, hp.sigma_min};
        const std::vector<float> normalized = solve_flow_euler(solver_input, steps, evaluate_velocity);
        latent_window window;
        window.input_offset_frames = token_offset;
        window.normalized_latents = normalized;
        window.denormalized_latents = conditioning_.denormalize_latents(normalized, hp.window_frames);
        prior_raw = window.denormalized_latents;
        result.windows.push_back(std::move(window));
    }
    result.denormalized_latents = assemble_flow_latent_windows(result.windows, source_frames,
                                                                 window_frames, hop_frames, latent_dim);
    return result;
}

} // namespace levo::flow
