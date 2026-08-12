// Diagnostic-only F32 Flow estimator parity runner. Inputs and references are
// raw little-endian F32 tensors exported from export_renderer_oracles.py.
#include "levo-flow-estimator.h"
#include "levo-flow-model.h"

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

// docs/renderer-parity.md frozen F32 targets.  The timestep/modulation
// captures are checked with the block-0 numerical boundary until they receive
// their own independently frozen document row.
constexpr float k_block_max_abs = 2.0e-3F;
constexpr double k_block_min_cosine = 0.99999;
constexpr float k_full_max_abs = 5.0e-3F;
constexpr double k_full_max_relative_rmse = 3.0e-3;
constexpr double k_full_min_cosine = 0.99999;
constexpr std::size_t k_batch = 2;
constexpr std::size_t k_hidden = 2200;
constexpr std::size_t k_latent = 64;
constexpr std::size_t k_modulation = 6 * k_hidden;

struct backend_deleter { void operator()(ggml_backend_t value) const noexcept { if (value) ggml_backend_free(value); } };
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

struct metrics {
    float max_abs = 0.0F;
    double rmse = 0.0;
    double reference_rms = 0.0;
    double relative_rmse = 0.0;
    double cosine = 1.0;
};

struct arguments {
    std::string model, model_input, timesteps, timestep_embedding, timestep_modulation, block0_input, block0_output, full_output, velocity, backend;
    std::size_t frames = 0;
    float guidance = 1.5F;
};

[[noreturn]] void usage(const char * program, const std::string & error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << '\n';
    std::cerr << "usage: " << program << " --model FLOW-F32.gguf --frames 2 --backend cpu|cuda"
              << " --model-input model_input.f32 --timesteps timesteps.f32"
              << " --timestep-embedding timestep_embedding.f32 --timestep-modulation timestep_modulation.f32"
              << " --block0-input block0_input.f32 --block0-output block0_output.f32"
              << " --full-output full_output.f32 --velocity velocity.f32 [--guidance 1.5]\n";
    std::exit(error.empty() ? 0 : 2);
}

std::size_t parse_size(const char * text, const char * option) {
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0 || value > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument(std::string(option) + " must be a positive integer");
    return static_cast<std::size_t>(value);
}
float parse_float(const char * text, const char * option) {
    char * end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) throw std::invalid_argument(std::string(option) + " must be finite");
    return value;
}

arguments parse_arguments(int argc, char ** argv) {
    arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") usage(argv[0]);
        if (++i == argc) usage(argv[0], "missing value for " + key);
        const char * value = argv[i];
        if (key == "--model") result.model = value;
        else if (key == "--frames") result.frames = parse_size(value, "--frames");
        else if (key == "--backend") result.backend = value;
        else if (key == "--model-input") result.model_input = value;
        else if (key == "--timesteps") result.timesteps = value;
        else if (key == "--timestep-embedding") result.timestep_embedding = value;
        else if (key == "--timestep-modulation") result.timestep_modulation = value;
        else if (key == "--block0-input") result.block0_input = value;
        else if (key == "--block0-output") result.block0_output = value;
        else if (key == "--full-output") result.full_output = value;
        else if (key == "--velocity") result.velocity = value;
        else if (key == "--guidance") result.guidance = parse_float(value, "--guidance");
        else usage(argv[0], "unknown option " + key);
    }
    const std::array<const std::string *, 10> required{{&result.model, &result.model_input, &result.timesteps, &result.timestep_embedding,
        &result.timestep_modulation, &result.block0_input, &result.block0_output, &result.full_output, &result.velocity, &result.backend}};
    if (result.frames == 0 || std::any_of(required.begin(), required.end(), [](const std::string * value) { return value->empty(); })) usage(argv[0], "all required options must be supplied");
    if (result.backend != "cpu" && result.backend != "cuda") usage(argv[0], "--backend must be cpu or cuda");
    return result;
}

ggml_backend_dev_t select_device(const std::string & wanted) {
    ggml_backend_load_all();
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (wanted == "cpu" && type == GGML_BACKEND_DEVICE_TYPE_CPU) return device;
        if (wanted == "cuda" && (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) && std::string(ggml_backend_dev_name(device)).rfind("CUDA", 0) == 0) return device;
    }
    throw std::runtime_error("requested backend is unavailable");
}

std::vector<float> read_raw(const std::string & path, std::size_t expected, const char * label) {
    if (expected > std::numeric_limits<std::size_t>::max() / sizeof(float)) throw std::runtime_error(std::string(label) + " expected size overflows");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error(std::string("cannot open ") + label + ": " + path);
    const std::streamoff bytes = input.tellg();
    const std::size_t expected_bytes = expected * sizeof(float);
    if (bytes < 0 || static_cast<uintmax_t>(bytes) != expected_bytes) throw std::runtime_error(std::string(label) + " byte size does not match the declared tensor shape");
    input.seekg(0, std::ios::beg);
    std::vector<float> value(expected);
    input.read(reinterpret_cast<char *>(value.data()), static_cast<std::streamsize>(expected_bytes));
    if (input.gcount() != static_cast<std::streamsize>(expected_bytes) || !std::all_of(value.begin(), value.end(), [](float x) { return std::isfinite(x); })) throw std::runtime_error(std::string(label) + " is truncated or non-finite");
    return value;
}

metrics compare(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    if (actual.size() != expected.size() || actual.empty()) throw std::runtime_error(std::string(label) + " capture shape mismatch");
    metrics value;
    double squared = 0.0, dot = 0.0, actual_sq = 0.0, expected_sq = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i], d = a - b;
        value.max_abs = std::max(value.max_abs, static_cast<float>(std::abs(d)));
        squared += d * d; dot += a * b; actual_sq += a * a; expected_sq += b * b;
    }
    value.rmse = std::sqrt(squared / actual.size());
    value.reference_rms = std::sqrt(expected_sq / expected.size());
    value.relative_rmse = value.reference_rms == 0.0 ? (value.rmse == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : value.rmse / value.reference_rms;
    value.cosine = actual_sq == 0.0 || expected_sq == 0.0 ? (actual_sq == expected_sq ? 1.0 : 0.0) : dot / std::sqrt(actual_sq * expected_sq);
    return value;
}

void report(const char * name, const metrics & value) {
    std::cout << name << " max_abs=" << std::setprecision(9) << value.max_abs
              << " rmse=" << std::setprecision(12) << value.rmse
              << " relative_rmse=" << std::setprecision(12) << value.relative_rmse
              << " cosine=" << std::setprecision(12) << value.cosine << '\n';
}

bool block_passes(const metrics & value) { return value.max_abs <= k_block_max_abs && value.cosine >= k_block_min_cosine; }
bool full_passes(const metrics & value) { return value.max_abs <= k_full_max_abs && value.relative_rmse <= k_full_max_relative_rmse && value.cosine >= k_full_min_cosine; }

std::vector<float> cfg_velocity(const std::vector<float> & raw, std::size_t frames, float guidance) {
    const std::size_t batch_width = frames * k_latent;
    if (raw.size() != 2U * batch_width) throw std::runtime_error("raw velocity has invalid fixed batch shape");
    std::vector<float> result(frames * k_latent);
    for (std::size_t frame = 0; frame < frames; ++frame) for (std::size_t channel = 0; channel < k_latent; ++channel) {
        const std::size_t offset = frame * k_latent + channel;
        const float uncond = raw[offset];
        const float cond = raw[batch_width + offset];
        result[frame * k_latent + channel] = uncond + guidance * (cond - uncond);
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const arguments args = parse_arguments(argc, argv);
        if (args.frames > std::numeric_limits<std::size_t>::max() / (k_batch * k_hidden)) throw std::runtime_error("frame count overflows tensor shape");
        backend_ptr backend(ggml_backend_dev_init(select_device(args.backend), nullptr));
        if (!backend) throw std::runtime_error("failed to initialize backend");
        levo::flow::load_options options;
        options.backend = backend.get(); options.allow_f32 = true; options.allow_f16 = false;
        const auto model = levo::flow::model::load_gguf(args.model, options);
        const auto estimator = levo::flow::estimator::create(model);
        if (model->hparams().hidden_size != static_cast<int32_t>(k_hidden) || model->hparams().latent_dim != static_cast<int32_t>(k_latent)) throw std::runtime_error("Flow GGUF does not match the fixed estimator parity contract");

        const std::size_t sequence = k_batch * args.frames * k_hidden;
        levo::flow::estimator_input input;
        input.batch = k_batch; input.frames = args.frames;
        input.model_input = read_raw(args.model_input, sequence, "model_input");
        input.timesteps = read_raw(args.timesteps, k_batch, "timesteps");
        const auto expected_timestep = read_raw(args.timestep_embedding, k_batch * k_hidden, "timestep_embedding");
        const auto expected_modulation = read_raw(args.timestep_modulation, k_batch * k_modulation, "timestep_modulation");
        const auto expected_block0_input = read_raw(args.block0_input, sequence, "block0_input");
        const auto expected_block0_output = read_raw(args.block0_output, sequence, "block0_output");
        const auto expected_full = read_raw(args.full_output, sequence, "full_output");
        const auto expected_velocity = read_raw(args.velocity, args.frames * k_latent, "velocity");
        levo::flow::estimator_capture capture;
        const std::vector<float> raw_velocity = estimator->velocity(input, &capture);
        const std::vector<float> velocity = cfg_velocity(raw_velocity, args.frames, args.guidance);

        const std::array<std::pair<const char *, metrics>, 6> measurements{{
            {"timestep_embedding", compare(capture.timestep_embedding, expected_timestep, "timestep_embedding")},
            {"timestep_modulation", compare(capture.timestep_modulation, expected_modulation, "timestep_modulation")},
            {"block0_input", compare(capture.block0_input, expected_block0_input, "block0_input")},
            {"block0_output", compare(capture.block0_output, expected_block0_output, "block0_output")},
            {"full_output", compare(capture.full_output, expected_full, "full_output")},
            {"velocity", compare(velocity, expected_velocity, "velocity")},
        }};
        bool success = true;
        for (const auto & item : measurements) { report(item.first, item.second); success = success && (std::string(item.first) == "full_output" || std::string(item.first) == "velocity" ? full_passes(item.second) : block_passes(item.second)); }
        if (!success) {
            std::cerr << "parity threshold failure: block/timestep max_abs<=" << k_block_max_abs << " cosine>=" << k_block_min_cosine
                      << "; full/velocity max_abs<=" << k_full_max_abs << " relative_rmse<=" << k_full_max_relative_rmse << " cosine>=" << k_full_min_cosine << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
