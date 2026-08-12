#include "levo.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

levo::backend_kind parse_backend(const std::string & value) {
    if (value == "auto") return levo::backend_kind::auto_select;
    if (value == "cpu") return levo::backend_kind::cpu;
    if (value == "cuda") return levo::backend_kind::cuda;
    if (value == "gpu") return levo::backend_kind::gpu;
    throw std::invalid_argument("unknown backend '" + value + "'");
}

uint64_t parse_u64(const std::string & value, const char * label) {
    if (value.empty() || value.front() == '-') throw std::invalid_argument(std::string("invalid ") + label);
    std::size_t used = 0;
    try {
        const auto parsed = std::stoull(value, &used, 10);
        if (used != value.size() || parsed > std::numeric_limits<uint64_t>::max()) throw std::out_of_range(label);
        return static_cast<uint64_t>(parsed);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + label + " '" + value + "'");
    }
}

int parse_device(const std::string & value) {
    const uint64_t parsed = parse_u64(value, "device index");
    if (parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("device index is too large");
    return static_cast<int>(parsed);
}

float parse_float(const std::string & value, const char * label) {
    std::size_t used = 0;
    try {
        const float parsed = std::stof(value, &used);
        if (used != value.size() || !std::isfinite(parsed)) throw std::invalid_argument(label);
        return parsed;
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + label + " '" + value + "'");
    }
}

const char * stage_name(levo::render_stage stage) {
    switch (stage) {
        case levo::render_stage::loading_tokens: return "loading tokens";
        case levo::render_stage::loading_flow: return "loading Flow";
        case levo::render_stage::generating_latents: return "generating Flow latents";
        case levo::render_stage::releasing_flow: return "releasing Flow";
        case levo::render_stage::loading_vae: return "loading VAE";
        case levo::render_stage::decoding_window: return "decoding VAE window";
        case levo::render_stage::assembling_audio: return "assembling audio";
        case levo::render_stage::complete: return "complete";
    }
    return "unknown";
}

std::vector<float> read_noise_f32(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open F32 noise file: " + path);
    const std::streamoff bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::invalid_argument("F32 noise file size must be a positive multiple of four bytes");
    }
    const auto count = static_cast<std::size_t>(bytes / static_cast<std::streamoff>(sizeof(float)));
    input.seekg(0, std::ios::beg);
    std::vector<float> values(count);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!input) throw std::runtime_error("cannot read complete F32 noise file: " + path);
    return values;
}

void usage(const char * program) {
    std::cout << "Usage:\n  " << program
              << " tokens.npy --flow-model FLOW.gguf --vae-model VAE.gguf --output song.wav"
              << " [--backend auto|cpu|cuda|gpu --device N --steps N --cfg X --seed N]"
              << " [--noise-f32 window-major-noise.f32]\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
            usage(argv[0]);
            return 0;
        }
        levo::render_config config;
        config.tokens_path = argv[1];
        std::string output;
        bool flow = false, vae = false, output_present = false;
        for (int index = 2; index < argc; ++index) {
            const std::string option = argv[index];
            const auto value = [&]() -> std::string {
                if (++index >= argc) throw std::invalid_argument("missing value for " + option);
                return argv[index];
            };
            if (option == "--flow-model") { config.flow_model_path = value(); flow = true; }
            else if (option == "--vae-model") { config.vae_model_path = value(); vae = true; }
            else if (option == "--output") { output = value(); output_present = true; }
            else if (option == "--backend") config.backend = parse_backend(value());
            else if (option == "--device") config.device_index = parse_device(value());
            else if (option == "--steps") config.euler_steps = static_cast<std::size_t>(parse_u64(value(), "Euler steps"));
            else if (option == "--cfg") config.cfg_scale = parse_float(value(), "CFG scale");
            else if (option == "--seed") config.seed = parse_u64(value(), "seed");
            else if (option == "--noise-f32") config.external_noise = read_noise_f32(value());
            else throw std::invalid_argument("unknown option '" + option + "'");
        }
        if (!flow || !vae || !output_present) throw std::invalid_argument("--flow-model, --vae-model, and --output are required");
        if (std::filesystem::path(output).extension() != ".wav") {
            throw std::invalid_argument("--output must use the .wav extension");
        }
        const auto progress = [](const levo::render_progress & value) {
            std::cerr << stage_name(value.stage);
            if (value.total_windows != 0) std::cerr << " " << value.completed_windows << "/" << value.total_windows;
            std::cerr << '\n';
        };
        const levo::render_result result = levo::render_tokens_to_audio(config, progress);
        levo::write_render_wav(output, result);
        std::cout << "wrote " << output << " (" << result.samples_per_channel << " stereo samples, "
                  << result.provenance.backend_name << ")\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
