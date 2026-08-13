#include "levo.h"

#include "levo-cuda-precision.h"
#include "levo-flow-model.h"
#include "levo-flow-renderer.h"
#include "levo-renderer-pattern.h"
#include "levo-token-io.h"
#include "levo-vae.h"
#include "levo-wav.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace levo {
namespace {

struct backend_deleter {
    void operator()(ggml_backend_t value) const noexcept { if (value) ggml_backend_free(value); }
};
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo native renderer: " + message);
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char * label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        fail(std::string(label) + " size overflow");
    }
    return lhs * rhs;
}

bool is_gpu(ggml_backend_dev_t device) {
    const auto kind = ggml_backend_dev_type(device);
    return kind == GGML_BACKEND_DEVICE_TYPE_GPU || kind == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

bool is_cuda(ggml_backend_dev_t device) {
    return is_gpu(device) && std::string(ggml_backend_dev_name(device)).rfind("CUDA", 0) == 0;
}

ggml_backend_dev_t select_device(backend_kind request, int index) {
    if (index < 0) fail("device index must be non-negative");
    ggml_backend_load_all();
    const auto nth = [index](const auto & predicate) -> ggml_backend_dev_t {
        int found = 0;
        for (std::size_t item = 0; item < ggml_backend_dev_count(); ++item) {
            ggml_backend_dev_t device = ggml_backend_dev_get(item);
            if (predicate(device) && found++ == index) return device;
        }
        return nullptr;
    };
    if (request == backend_kind::auto_select) {
        if (auto device = nth([](ggml_backend_dev_t value) { return is_cuda(value); })) return device;
        if (auto device = nth([](ggml_backend_dev_t value) { return is_gpu(value); })) return device;
        if (auto device = nth([](ggml_backend_dev_t value) {
                return ggml_backend_dev_type(value) == GGML_BACKEND_DEVICE_TYPE_CPU;
            })) return device;
    } else {
        const auto matches = [request](ggml_backend_dev_t value) {
            if (request == backend_kind::cuda) return is_cuda(value);
            if (request == backend_kind::gpu) return is_gpu(value);
            return ggml_backend_dev_type(value) == GGML_BACKEND_DEVICE_TYPE_CPU;
        };
        if (auto device = nth(matches)) return device;
    }
    fail("requested backend device is unavailable");
}

// SplitMix64 plus Box-Muller is intentionally implemented here instead of
// std::normal_distribution, whose sequence is not standardized by C++.
uint64_t splitmix64(uint64_t & state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t value = state;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

double uniform_open(uint64_t & state) {
    // (0,1) using 53 random bits; neither endpoint is representable.
    constexpr double scale = 1.0 / 9007199254740992.0;
    return (static_cast<double>(splitmix64(state) >> 11U) + 0.5) * scale;
}

std::vector<float> native_gaussian_noise(std::size_t count, uint64_t seed) {
    std::vector<float> result(count);
    uint64_t state = seed;
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (std::size_t offset = 0; offset < count; offset += 2) {
        const double radius = std::sqrt(-2.0 * std::log(uniform_open(state)));
        const double angle = two_pi * uniform_open(state);
        result[offset] = static_cast<float>(radius * std::cos(angle));
        if (offset + 1U < count) result[offset + 1U] = static_cast<float>(radius * std::sin(angle));
    }
    return result;
}

void require_finite(const std::vector<float> & values, const char * label) {
    for (float value : values) if (!std::isfinite(value)) fail(std::string(label) + " contains a non-finite value");
}

using progress_clock = std::chrono::steady_clock;

double elapsed_seconds(progress_clock::time_point begin, progress_clock::time_point end = progress_clock::now()) {
    return std::chrono::duration<double>(end - begin).count();
}

class render_reporter final {
public:
    render_reporter(render_progress_callback callback, cancellation_callback cancelled)
        : callback_(std::move(callback)), cancelled_(std::move(cancelled)),
          started_(progress_clock::now()), stage_started_(started_) {}

    void enter(render_stage stage, std::size_t completed_windows = 0,
               std::size_t total_windows = 0, std::size_t current_window = 0,
               std::size_t completed_steps = 0, std::size_t total_steps = 0) {
        stage_ = stage;
        stage_started_ = progress_clock::now();
        update(completed_windows, total_windows, current_window, completed_steps, total_steps);
    }

    void update(std::size_t completed_windows = 0, std::size_t total_windows = 0,
                std::size_t current_window = 0, std::size_t completed_steps = 0,
                std::size_t total_steps = 0) const {
        const auto now = progress_clock::now();
        render_progress value;
        value.stage = stage_;
        value.completed_windows = completed_windows;
        value.total_windows = total_windows;
        value.current_window = current_window;
        value.completed_steps = completed_steps;
        value.total_steps = total_steps;
        value.elapsed_seconds = elapsed_seconds(started_, now);
        value.stage_elapsed_seconds = elapsed_seconds(stage_started_, now);
        if (callback_) callback_(value);
        check_cancelled();
    }

    void check_cancelled() const {
        if (cancelled_ && cancelled_()) throw operation_cancelled();
    }

    double elapsed() const { return elapsed_seconds(started_); }

private:
    render_progress_callback callback_;
    cancellation_callback cancelled_;
    progress_clock::time_point started_;
    progress_clock::time_point stage_started_;
    render_stage stage_ = render_stage::loading_tokens;
};

std::vector<std::vector<int32_t>> stream_major(const token_io::artifact & tokens) {
    if (tokens.frame_count == 0 || tokens.tokens.size() != 3U * tokens.frame_count) {
        fail("tokens.npy does not have canonical shape [3,T]");
    }
    std::vector<std::vector<int32_t>> result(3, std::vector<int32_t>(tokens.frame_count));
    for (std::size_t stream = 0; stream < result.size(); ++stream) {
        std::copy_n(tokens.tokens.begin() + stream * tokens.frame_count, tokens.frame_count,
                    result[stream].begin());
    }
    return result;
}

std::vector<float> channel_major_latent(const std::vector<float> & frame_major,
                                        std::size_t frames, std::size_t latent_dim) {
    if (frame_major.size() != checked_product(frames, latent_dim, "Flow latent")) {
        fail("Flow latent window has an unexpected shape");
    }
    std::vector<float> result(frame_major.size());
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t channel = 0; channel < latent_dim; ++channel) {
            result[channel * frames + frame] = frame_major[frame * latent_dim + channel];
        }
    }
    return result;
}

renderer_stereo_audio stereo_from_vae(const detail::vae_decode_result & decoded) {
    if (decoded.samples_per_channel == 0 || decoded.audio.size() != 2U * decoded.samples_per_channel) {
        fail("VAE decoder returned an unexpected stereo payload");
    }
    renderer_stereo_audio result;
    result.left.assign(decoded.audio.begin(), decoded.audio.begin() + decoded.samples_per_channel);
    result.right.assign(decoded.audio.begin() + decoded.samples_per_channel, decoded.audio.end());
    return result;
}

} // namespace

render_result render_tokens_to_audio(const render_config & config,
                                     render_progress_callback progress) {
    if (config.tokens_path.empty() || config.flow_model_path.empty() || config.vae_model_path.empty()) {
        fail("tokens, Flow model, and VAE model paths are required");
    }
    if (config.tokens_path.extension() != ".npy") fail("tokens path must use the .npy extension");
    if (!std::isfinite(config.cfg_scale) || config.cfg_scale < 0.0F) fail("CFG scale must be finite and non-negative");
    render_reporter reporter(std::move(progress), config.cancelled);
    render_timings timings;

    reporter.enter(render_stage::loading_tokens);
    const auto token_started = progress_clock::now();
    const token_io::artifact artifact = token_io::read(config.tokens_path);
    const renderer_token_streams streams = validate_renderer_tokens(stream_major(artifact));
    const renderer_schedule schedule = make_renderer_schedule(streams.frames());
    const std::size_t per_window_noise = checked_product(renderer_window_frames, std::size_t(64), "Flow noise window");
    const std::size_t expected_noise = checked_product(schedule.windows.size(), per_window_noise, "Flow noise");
    std::vector<float> noise = config.external_noise.empty()
        ? native_gaussian_noise(expected_noise, config.seed) : config.external_noise;
    if (noise.size() != expected_noise) fail("external noise must have shape [window_count, 1000, 64]");
    require_finite(noise, "external noise");
    timings.token_load_seconds = elapsed_seconds(token_started);

    reporter.enter(render_stage::initializing_backend, 0, schedule.windows.size());
    const auto backend_started = progress_clock::now();
    ggml_backend_dev_t device = select_device(config.backend, config.device_index);
    // Must precede backend initialization: this is the difference between a
    // true F32 renderer and a TF32 one on Ampere and later.
    detail::configure_cuda_gemm_f32_accumulation(device);
    detail::configure_cuda_disable_tf32(device);
    backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) fail("cannot initialize selected GGML backend");
    timings.backend_seconds = elapsed_seconds(backend_started);

    flow::render_output latents;
    render_provenance provenance;
    provenance.backend_name = ggml_backend_name(backend.get());
    provenance.token_tensor_sha256 = token_io::tensor_sha256(artifact.tokens);
    provenance.seed = config.seed;
    provenance.external_noise = !config.external_noise.empty();
    std::size_t resolved_steps = config.euler_steps;
    float resolved_cfg = config.cfg_scale;
    reporter.enter(render_stage::loading_flow, 0, schedule.windows.size());
    const auto flow_load_started = progress_clock::now();
    progress_clock::time_point flow_release_started;
    {
        const std::shared_ptr<flow::model> model = flow::model::load_gguf(
            config.flow_model_path.string(), {backend.get(), true, true, false});
        const flow::flow_provenance & source = model->provenance();
        provenance.flow_model_name = source.name;
        provenance.flow_levo_revision = source.levo_revision;
        provenance.flow_model_sha256 = source.model_sha256;
        provenance.flow_runtime_revision = source.runtime_revision;
        resolved_steps = resolved_steps == 0 ? static_cast<std::size_t>(model->hparams().euler_steps_default) : resolved_steps;
        resolved_cfg = resolved_cfg == 0.0F ? model->hparams().cfg_default : resolved_cfg;
        flow::renderer flow_renderer(model);
        flow::render_input flow_input;
        flow_input.vocal_codes = streams.vocal;
        flow_input.bgm_codes = streams.bgm;
        flow_input.initial_noise = std::move(noise);
        flow::render_options options;
        options.euler_steps = config.euler_steps;
        options.guidance_scale = config.cfg_scale;
        options.cancelled = config.cancelled;
        options.progress = [&reporter](std::size_t window, std::size_t windows,
                                       std::size_t completed_steps, std::size_t total_steps) {
            const std::size_t completed_windows = completed_steps == total_steps ? window : window - 1U;
            reporter.update(completed_windows, windows, window, completed_steps, total_steps);
        };
        timings.flow_load_seconds = elapsed_seconds(flow_load_started);
        reporter.enter(render_stage::generating_latents, 0, schedule.windows.size());
        const auto flow_started = progress_clock::now();
        latents = flow_renderer.render(flow_input, options);
        timings.flow_seconds = elapsed_seconds(flow_started);
        reporter.enter(render_stage::releasing_flow, schedule.windows.size(), schedule.windows.size());
        flow_release_started = progress_clock::now();
    }
    timings.flow_release_seconds = elapsed_seconds(flow_release_started);
    if (latents.windows.size() != schedule.windows.size() || latents.source_frames != streams.frames()) {
        fail("Flow renderer returned an inconsistent window schedule");
    }

    reporter.enter(render_stage::loading_vae, 0, latents.windows.size());
    const auto vae_load_started = progress_clock::now();
    std::vector<renderer_stereo_audio> decoded_windows;
    decoded_windows.reserve(latents.windows.size());
    {
        const std::shared_ptr<detail::vae_model> model = detail::vae_model::load_gguf(
            config.vae_model_path.string(), {backend.get(), true, false, true});
        const detail::vae_provenance & source = model->provenance();
        provenance.vae_model_name = source.name;
        provenance.vae_checkpoint_sha256 = source.checkpoint_sha256;
        provenance.vae_config_sha256 = source.config_sha256;
        provenance.vae_levo_revision = source.levo_revision;
        provenance.vae_runtime_revision = source.runtime_revision;
        const detail::vae_hparams & hp = model->hparams();
        if (hp.sample_rate != static_cast<int32_t>(renderer_sample_rate) || hp.audio_channels != 2 ||
            hp.latent_dim != 64 || hp.downsampling_ratio != static_cast<int32_t>(renderer_samples_per_frame)) {
            fail("VAE GGUF does not match the native Flow renderer contract");
        }
        std::unique_ptr<detail::vae_decoder> decoder = detail::vae_decoder::create(model);
        timings.vae_load_seconds = elapsed_seconds(vae_load_started);
        reporter.enter(render_stage::decoding_window, 0, latents.windows.size(), 1U);
        const auto vae_started = progress_clock::now();
        for (std::size_t index = 0; index < latents.windows.size(); ++index) {
            reporter.check_cancelled();
            const flow::latent_window & window = latents.windows[index];
            const std::vector<float> input = channel_major_latent(window.denormalized_latents,
                renderer_window_frames, static_cast<std::size_t>(hp.latent_dim));
            decoded_windows.push_back(stereo_from_vae(decoder->decode(input, renderer_window_frames)));
            reporter.update(index + 1U, latents.windows.size(), index + 1U);
        }
        timings.vae_seconds = elapsed_seconds(vae_started);
    }
    reporter.enter(render_stage::assembling_audio, latents.windows.size(), latents.windows.size());
    const auto assembly_started = progress_clock::now();
    const renderer_stereo_audio stereo = assemble_renderer_audio(schedule, decoded_windows);
    if (stereo.left.size() != stereo.right.size()) fail("audio assembler returned uneven stereo channels");
    render_result result;
    if (config.capture_windows) {
        result.windows.reserve(latents.windows.size());
        for (std::size_t index = 0; index < latents.windows.size(); ++index) {
            render_window_capture capture;
            capture.input_offset_frames = latents.windows[index].input_offset_frames;
            capture.normalized_latents = latents.windows[index].normalized_latents;
            capture.denormalized_latents = latents.windows[index].denormalized_latents;
            capture.decoded_left = std::move(decoded_windows[index].left);
            capture.decoded_right = std::move(decoded_windows[index].right);
            result.windows.push_back(std::move(capture));
        }
    }
    result.samples_per_channel = stereo.samples();
    result.source_frames = streams.frames();
    result.rendered_windows = latents.windows.size();
    result.sample_rate = renderer_sample_rate;
    result.euler_steps = resolved_steps;
    result.cfg_scale = resolved_cfg;
    result.provenance = std::move(provenance);
    result.interleaved_stereo.reserve(checked_product(result.samples_per_channel, std::size_t(2), "interleaved audio"));
    for (std::size_t sample = 0; sample < result.samples_per_channel; ++sample) {
        result.interleaved_stereo.push_back(stereo.left[sample]);
        result.interleaved_stereo.push_back(stereo.right[sample]);
    }
    timings.assembly_seconds = elapsed_seconds(assembly_started);
    timings.total_seconds = reporter.elapsed();
    result.timings = timings;
    reporter.enter(render_stage::complete, latents.windows.size(), latents.windows.size());
    return result;
}

void write_render_wav(const std::filesystem::path & output_path,
                      const render_result & result) {
    if (result.sample_rate != renderer_sample_rate || result.samples_per_channel == 0 ||
        result.interleaved_stereo.size() != 2U * result.samples_per_channel) {
        fail("render result is not valid 48 kHz stereo audio");
    }
    detail::write_wav_f32(output_path, result.interleaved_stereo, result.sample_rate, 2);
}

} // namespace levo
