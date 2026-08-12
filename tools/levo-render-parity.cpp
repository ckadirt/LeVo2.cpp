// Diagnostic-only end-to-end renderer parity runner. It drives the production
// `levo::render_tokens_to_audio` path with explicit external noise and compares
// every intermediate boundary against raw little-endian F32 files exported from
// python/export_renderer_oracles.py `render`.
//
// The gates are the frozen F32 thresholds in docs/renderer-parity.md: latents
// are checked at maximum absolute error and relative RMS, and so is the final
// waveform. Cosine similarity is reported as a diagnostic and only gates when
// --min-cosine is supplied.
#include "levo.h"

#include "levo-renderer-pattern.h"
#include "levo-token-io.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Frozen from docs/renderer-parity.md. The latent gate is the Euler step-50
// boundary; the audio gate is the final VAE waveform.
constexpr float k_frozen_latent_max_abs = 2.0e-2F;
constexpr double k_frozen_latent_rel_rms = 5.0e-3;
constexpr float k_frozen_audio_max_abs = 3.0e-3F;
constexpr double k_frozen_audio_rel_rms = 1.0e-3;

struct metrics {
    float max_abs = 0.0F;
    double rmse = 0.0;
    double rel_rms = 0.0;
    double cosine = 1.0;
};

struct arguments {
    std::string tokens;
    std::string noise;
    std::string flow_model;
    std::string vae_model;
    std::string backend;
    std::string latent_normalized;
    std::string latent_denormalized;
    std::string decoded_windows;
    std::string audio;
    std::size_t steps = 0;
    float cfg = 0.0F;
    float latent_max_abs = k_frozen_latent_max_abs;
    double latent_rel_rms = k_frozen_latent_rel_rms;
    float audio_max_abs = k_frozen_audio_max_abs;
    double audio_rel_rms = k_frozen_audio_rel_rms;
    // Negative disables the optional cosine gate; cosine is always reported.
    double min_cosine = -2.0;
};

[[noreturn]] void usage(const char * program, const std::string & error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << '\n';
    std::cerr << "usage: " << program << " --tokens tokens.npy --noise noise.f32"
              << " --flow-model flow-F32.gguf --vae-model vae-F32.gguf --backend cpu|cuda"
              << " --steps N --cfg X --latent-normalized normalized.f32"
              << " --latent-denormalized denormalized.f32 --audio audio.f32"
              << " [--decoded-windows windows.f32]"
              << " [--latent-max-abs X --latent-rel-rms X --audio-max-abs X --audio-rel-rms X"
              << " --min-cosine X]\n";
    std::exit(error.empty() ? 0 : 2);
}

float parse_float(const char * text, const char * option) {
    char * end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) throw std::invalid_argument(std::string(option) + " must be finite");
    return value;
}

std::size_t parse_size(const char * text, const char * option) {
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0 || value > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument(std::string(option) + " must be a positive integer");
    return static_cast<std::size_t>(value);
}

arguments parse_arguments(int argc, char ** argv) {
    arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") usage(argv[0]);
        if (++i == argc) usage(argv[0], "missing value for " + option);
        const char * value = argv[i];
        if (option == "--tokens") result.tokens = value;
        else if (option == "--noise") result.noise = value;
        else if (option == "--flow-model") result.flow_model = value;
        else if (option == "--vae-model") result.vae_model = value;
        else if (option == "--backend") result.backend = value;
        else if (option == "--latent-normalized") result.latent_normalized = value;
        else if (option == "--latent-denormalized") result.latent_denormalized = value;
        else if (option == "--decoded-windows") result.decoded_windows = value;
        else if (option == "--audio") result.audio = value;
        else if (option == "--steps") result.steps = parse_size(value, "--steps");
        else if (option == "--cfg") result.cfg = parse_float(value, "--cfg");
        else if (option == "--latent-max-abs") result.latent_max_abs = parse_float(value, "--latent-max-abs");
        else if (option == "--latent-rel-rms") result.latent_rel_rms = parse_float(value, "--latent-rel-rms");
        else if (option == "--audio-max-abs") result.audio_max_abs = parse_float(value, "--audio-max-abs");
        else if (option == "--audio-rel-rms") result.audio_rel_rms = parse_float(value, "--audio-rel-rms");
        else if (option == "--min-cosine") result.min_cosine = parse_float(value, "--min-cosine");
        else usage(argv[0], "unknown option " + option);
    }
    if (result.tokens.empty() || result.noise.empty() || result.flow_model.empty() ||
        result.vae_model.empty() || result.backend.empty() || result.latent_normalized.empty() ||
        result.latent_denormalized.empty() || result.audio.empty() || result.steps == 0) {
        usage(argv[0], "token, noise, model, backend, step, latent, and audio options are required");
    }
    if (result.backend != "cpu" && result.backend != "cuda") usage(argv[0], "--backend must be cpu or cuda");
    if (result.latent_max_abs < 0.0F || result.latent_rel_rms < 0.0 || result.audio_max_abs < 0.0F ||
        result.audio_rel_rms < 0.0 || result.min_cosine > 1.0) {
        usage(argv[0], "invalid parity thresholds");
    }
    return result;
}

std::vector<float> read_raw_f32(const std::string & path, std::size_t expected_count, const std::string & label) {
    if (expected_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) throw std::runtime_error(label + " expected size overflows");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + label + ": " + path);
    const std::streamoff bytes = input.tellg();
    const std::size_t expected_bytes = expected_count * sizeof(float);
    if (bytes < 0 || static_cast<uintmax_t>(bytes) != expected_bytes) {
        throw std::runtime_error(label + " has " + std::to_string(bytes) + " bytes, expected " + std::to_string(expected_bytes));
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> values(expected_count);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(expected_bytes));
    if (input.gcount() != static_cast<std::streamsize>(expected_bytes)) throw std::runtime_error("truncated " + label);
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) throw std::runtime_error(label + " contains non-finite values");
    return values;
}

metrics compare(const float * actual, const float * expected, std::size_t count, const std::string & label) {
    if (count == 0) throw std::runtime_error(label + " comparison shape is invalid");
    double squared = 0.0, reference = 0.0, dot = 0.0, actual_norm = 0.0, expected_norm = 0.0;
    float maximum = 0.0F;
    for (std::size_t i = 0; i < count; ++i) {
        const double a = actual[i], b = expected[i], delta = a - b;
        maximum = std::max(maximum, static_cast<float>(std::abs(delta)));
        squared += delta * delta;
        reference += b * b;
        dot += a * b;
        actual_norm += a * a;
        expected_norm += b * b;
    }
    const double elements = static_cast<double>(count);
    metrics result;
    result.max_abs = maximum;
    result.rmse = std::sqrt(squared / elements);
    const double reference_rms = std::sqrt(reference / elements);
    // A zero reference cannot be scaled; report the absolute error instead of
    // dividing, so an all-zero oracle can never silently pass.
    result.rel_rms = reference_rms > 0.0 ? result.rmse / reference_rms : result.rmse;
    result.cosine = actual_norm == 0.0 || expected_norm == 0.0 ? (actual_norm == expected_norm ? 1.0 : 0.0) : dot / std::sqrt(actual_norm * expected_norm);
    return result;
}

void print_metric(const std::string & label, const metrics & value) {
    std::cout << label << " max_abs=" << std::setprecision(9) << value.max_abs
              << " rmse=" << std::setprecision(12) << value.rmse
              << " rel_rms=" << std::setprecision(12) << value.rel_rms
              << " cosine=" << std::setprecision(12) << value.cosine << '\n';
}

bool check(const std::string & label, const metrics & value, float max_abs, double rel_rms, double min_cosine) {
    print_metric(label, value);
    bool ok = value.max_abs <= max_abs && value.rel_rms <= rel_rms;
    if (min_cosine >= -1.0) ok = ok && value.cosine >= min_cosine;
    if (!ok) {
        std::cerr << "parity threshold failure at " << label << ": max_abs<=" << max_abs
                  << " rel_rms<=" << rel_rms;
        if (min_cosine >= -1.0) std::cerr << " cosine>=" << min_cosine;
        std::cerr << '\n';
    }
    return ok;
}

levo::backend_kind backend_kind_from(const std::string & requested) {
    return requested == "cuda" ? levo::backend_kind::cuda : levo::backend_kind::cpu;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const arguments args = parse_arguments(argc, argv);

        // The window count is a property of the token artifact, so the noise
        // and reference files are validated against the schedule rather than
        // trusted for their own length.
        const levo::token_io::artifact tokens = levo::token_io::read(args.tokens);
        const levo::renderer_schedule schedule = levo::make_renderer_schedule(tokens.frame_count);
        const std::size_t windows = schedule.windows.size();
        const std::size_t latent_dim = 64U;
        const std::size_t per_window_latent = levo::renderer_window_frames * latent_dim;
        const std::size_t per_window_samples = levo::renderer_window_frames * levo::renderer_samples_per_frame;
        const std::size_t target_samples = schedule.target_samples;

        levo::render_config config;
        config.tokens_path = args.tokens;
        config.flow_model_path = args.flow_model;
        config.vae_model_path = args.vae_model;
        config.backend = backend_kind_from(args.backend);
        config.euler_steps = args.steps;
        config.cfg_scale = args.cfg;
        config.capture_windows = true;
        config.external_noise = read_raw_f32(args.noise, windows * per_window_latent, "noise");

        const std::vector<float> expected_normalized =
                read_raw_f32(args.latent_normalized, windows * per_window_latent, "normalized latent");
        const std::vector<float> expected_denormalized =
                read_raw_f32(args.latent_denormalized, windows * per_window_latent, "denormalized latent");
        const std::vector<float> expected_audio = read_raw_f32(args.audio, 2U * target_samples, "audio");
        std::vector<float> expected_decoded;
        if (!args.decoded_windows.empty()) {
            expected_decoded = read_raw_f32(args.decoded_windows, windows * 2U * per_window_samples, "decoded windows");
        }

        const levo::render_result result = levo::render_tokens_to_audio(config);
        if (result.windows.size() != windows) throw std::runtime_error("renderer returned an unexpected window inventory");
        if (result.samples_per_channel != target_samples) {
            throw std::runtime_error("renderer produced " + std::to_string(result.samples_per_channel) +
                                     " samples per channel, expected " + std::to_string(target_samples));
        }
        if (result.interleaved_stereo.size() != 2U * target_samples) throw std::runtime_error("renderer returned an uneven stereo payload");

        bool success = true;
        for (std::size_t window = 0; window < windows; ++window) {
            const levo::render_window_capture & capture = result.windows[window];
            if (capture.normalized_latents.size() != per_window_latent ||
                capture.denormalized_latents.size() != per_window_latent) {
                throw std::runtime_error("captured latent window has an unexpected shape");
            }
            const std::string suffix = "[" + std::to_string(window) + "]";
            success = check("normalized_latent" + suffix,
                            compare(capture.normalized_latents.data(),
                                    expected_normalized.data() + window * per_window_latent,
                                    per_window_latent, "normalized latent"),
                            args.latent_max_abs, args.latent_rel_rms, args.min_cosine) && success;
            success = check("denormalized_latent" + suffix,
                            compare(capture.denormalized_latents.data(),
                                    expected_denormalized.data() + window * per_window_latent,
                                    per_window_latent, "denormalized latent"),
                            args.latent_max_abs, args.latent_rel_rms, args.min_cosine) && success;
            if (expected_decoded.empty()) continue;
            if (capture.decoded_left.size() != per_window_samples || capture.decoded_right.size() != per_window_samples) {
                throw std::runtime_error("captured decoded window has an unexpected shape");
            }
            const std::size_t base = window * 2U * per_window_samples;
            success = check("decoded_left" + suffix,
                            compare(capture.decoded_left.data(), expected_decoded.data() + base,
                                    per_window_samples, "decoded left"),
                            args.audio_max_abs, args.audio_rel_rms, args.min_cosine) && success;
            success = check("decoded_right" + suffix,
                            compare(capture.decoded_right.data(),
                                    expected_decoded.data() + base + per_window_samples,
                                    per_window_samples, "decoded right"),
                            args.audio_max_abs, args.audio_rel_rms, args.min_cosine) && success;
        }

        // The oracle stores the assembled waveform channel-major; the public
        // result is interleaved because that is what the WAV writer consumes.
        std::vector<float> left(target_samples), right(target_samples);
        for (std::size_t sample = 0; sample < target_samples; ++sample) {
            left[sample] = result.interleaved_stereo[2U * sample];
            right[sample] = result.interleaved_stereo[2U * sample + 1U];
        }
        success = check("audio_left", compare(left.data(), expected_audio.data(), target_samples, "audio left"),
                        args.audio_max_abs, args.audio_rel_rms, args.min_cosine) && success;
        success = check("audio_right",
                        compare(right.data(), expected_audio.data() + target_samples, target_samples, "audio right"),
                        args.audio_max_abs, args.audio_rel_rms, args.min_cosine) && success;

        std::cout << "windows=" << windows << " source_frames=" << result.source_frames
                  << " samples_per_channel=" << result.samples_per_channel
                  << " backend=" << result.provenance.backend_name << '\n';
        return success ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
