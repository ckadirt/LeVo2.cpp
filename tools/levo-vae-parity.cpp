// Diagnostic-only native Oobleck VAE parity runner.  It consumes simple raw
// little-endian F32 files exported from python/export_renderer_oracles.py and
// compares every DecoderBlock output plus final stereo waveform.
#include "levo-vae-model.h"
#include "levo-vae.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Frozen from the pinned official CUDA-F32 captures at T=1/T=2, validated on
// this RTX 4090 for native CPU and CUDA backends.  These are stage-level,
// deliberately tighter than a final-waveform-only threshold.
constexpr float k_frozen_max_abs = 2.0e-3F;
constexpr float k_frozen_max_rmse = 2.5e-4F;
constexpr double k_frozen_min_cosine = 0.9999995;

struct backend_deleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend != nullptr) ggml_backend_free(backend);
    }
};
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

struct metrics {
    float max_abs = 0.0F;
    double rmse = 0.0;
    double cosine = 1.0;
};

struct arguments {
    std::string model;
    std::string latent;
    std::array<std::string, 5> stage;
    std::string audio;
    std::size_t frames = 0;
    std::string backend;
    float max_abs = k_frozen_max_abs;
    float max_rmse = k_frozen_max_rmse;
    double min_cosine = k_frozen_min_cosine;
};

[[noreturn]] void usage(const char * program, const std::string & error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << '\n';
    std::cerr << "usage: " << program << " --model VAE-F32.gguf --frames N --backend cpu|cuda"
              << " --latent latent.f32 --stage0 block0.f32 --stage1 block1.f32"
              << " --stage2 block2.f32 --stage3 block3.f32 --stage4 block4.f32"
              << " --audio audio.f32 [--max-abs X --max-rmse X --min-cosine X]\n";
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
        if (option == "--model") result.model = value;
        else if (option == "--latent") result.latent = value;
        else if (option == "--stage0") result.stage[0] = value;
        else if (option == "--stage1") result.stage[1] = value;
        else if (option == "--stage2") result.stage[2] = value;
        else if (option == "--stage3") result.stage[3] = value;
        else if (option == "--stage4") result.stage[4] = value;
        else if (option == "--audio") result.audio = value;
        else if (option == "--frames") result.frames = parse_size(value, "--frames");
        else if (option == "--backend") result.backend = value;
        else if (option == "--max-abs") result.max_abs = parse_float(value, "--max-abs");
        else if (option == "--max-rmse") result.max_rmse = parse_float(value, "--max-rmse");
        else if (option == "--min-cosine") result.min_cosine = parse_float(value, "--min-cosine");
        else usage(argv[0], "unknown option " + option);
    }
    if (result.model.empty() || result.latent.empty() || result.audio.empty() || result.frames == 0 ||
        result.backend.empty() || std::any_of(result.stage.begin(), result.stage.end(), [](const std::string & path) { return path.empty(); })) {
        usage(argv[0], "all model, input, stage, audio, frame, and backend options are required");
    }
    if (result.backend != "cpu" && result.backend != "cuda") usage(argv[0], "--backend must be cpu or cuda");
    if (result.max_abs < 0.0F || result.max_rmse < 0.0F || result.min_cosine < -1.0 || result.min_cosine > 1.0) usage(argv[0], "invalid parity thresholds");
    return result;
}

ggml_backend_dev_t select_device(const std::string & requested) {
    ggml_backend_load_all();
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const auto type = ggml_backend_dev_type(device);
        if (requested == "cpu" && type == GGML_BACKEND_DEVICE_TYPE_CPU) return device;
        if (requested == "cuda" && (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            std::string(ggml_backend_dev_name(device)).rfind("CUDA", 0) == 0) return device;
    }
    throw std::runtime_error("requested backend is unavailable");
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

metrics compare(const std::vector<float> & actual, const std::vector<float> & expected, const std::string & label) {
    if (actual.size() != expected.size() || actual.empty()) throw std::runtime_error(label + " comparison shape is invalid");
    double squared = 0.0, dot = 0.0, actual_norm = 0.0, expected_norm = 0.0;
    float maximum = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i], delta = a - b;
        maximum = std::max(maximum, static_cast<float>(std::abs(delta)));
        squared += delta * delta; dot += a * b; actual_norm += a * a; expected_norm += b * b;
    }
    metrics result;
    result.max_abs = maximum;
    result.rmse = std::sqrt(squared / static_cast<double>(actual.size()));
    result.cosine = actual_norm == 0.0 || expected_norm == 0.0 ? (actual_norm == expected_norm ? 1.0 : 0.0) : dot / std::sqrt(actual_norm * expected_norm);
    return result;
}

bool passes(const metrics & value, const arguments & args) {
    return value.max_abs <= args.max_abs && value.rmse <= args.max_rmse && value.cosine >= args.min_cosine;
}

void print_metric(const char * label, const metrics & value) {
    std::cout << label << " max_abs=" << std::setprecision(9) << value.max_abs
              << " rmse=" << std::setprecision(12) << value.rmse
              << " cosine=" << std::setprecision(12) << value.cosine << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const arguments args = parse_arguments(argc, argv);
        backend_ptr backend(ggml_backend_dev_init(select_device(args.backend), nullptr));
        if (!backend) throw std::runtime_error("failed to initialize backend");
        levo::detail::vae_model_load_options options;
        options.backend = backend.get();
        // The native VAE correctness graph is intentionally F32 only.
        options.allow_f32 = true;
        options.allow_f16 = false;
        const auto weights = levo::detail::vae_model::load_gguf(args.model, options);
        const auto decoder = levo::detail::vae_decoder::create(weights);

        if (args.frames > std::numeric_limits<std::size_t>::max() / 1920U) throw std::runtime_error("frame count overflows sample count");
        const std::size_t samples = args.frames * 1920U;
        const std::array<std::size_t, 5> stage_values{{1024U * 10U * args.frames, 512U * 60U * args.frames,
                                                        256U * 240U * args.frames, 128U * 960U * args.frames,
                                                        128U * 1920U * args.frames}};
        const std::vector<float> latent = read_raw_f32(args.latent, 64U * args.frames, "latent");
        std::array<std::vector<float>, 5> expected_stages;
        for (std::size_t stage = 0; stage < expected_stages.size(); ++stage) expected_stages[stage] = read_raw_f32(args.stage[stage], stage_values[stage], "stage" + std::to_string(stage));
        const std::vector<float> expected_audio = read_raw_f32(args.audio, 2U * samples, "audio");
        const auto result = decoder->decode(latent, args.frames, true);
        if (result.stage_outputs.size() != expected_stages.size() || result.audio.size() != expected_audio.size()) throw std::runtime_error("native decoder returned unexpected capture inventory");

        bool success = true;
        for (std::size_t stage = 0; stage < expected_stages.size(); ++stage) {
            const metrics value = compare(result.stage_outputs[stage], expected_stages[stage], "stage" + std::to_string(stage));
            print_metric(("stage" + std::to_string(stage)).c_str(), value);
            success = success && passes(value, args);
        }
        const metrics audio = compare(result.audio, expected_audio, "audio");
        print_metric("audio", audio);
        success = success && passes(audio, args);
        if (!success) {
            std::cerr << "parity threshold failure: max_abs<=" << args.max_abs << " rmse<=" << args.max_rmse << " cosine>=" << args.min_cosine << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
