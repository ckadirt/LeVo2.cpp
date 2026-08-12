#include "levo-vae.h"

#include "ggml-cpu.h"
#if LEVO_HAS_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct backend_deleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend != nullptr) ggml_backend_free(backend);
    }
};
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

std::vector<float> read_f32(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open F32 fixture " + path);
    const std::streamoff bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid F32 fixture size");
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!input) throw std::runtime_error("cannot read F32 fixture");
    return values;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        // Real-model parity is opt-in because the decoder GGUF is 322 MiB.
        const char * gguf = std::getenv("LEVO_VAE_GGUF_FIXTURE");
        const char * latent_path = std::getenv("LEVO_VAE_LATENT_FIXTURE");
        const char * audio_path = std::getenv("LEVO_VAE_AUDIO_FIXTURE");
        if (gguf == nullptr || latent_path == nullptr || audio_path == nullptr) return 0;

        const bool cuda = argc == 2 && std::string(argv[1]) == "cuda";
        backend_ptr backend;
#if LEVO_HAS_CUDA
        if (cuda) backend.reset(ggml_backend_cuda_init(0));
#else
        if (cuda) throw std::runtime_error("CUDA backend is not compiled");
#endif
        if (!backend) backend.reset(ggml_backend_cpu_init());
        if (!backend) throw std::runtime_error("cannot initialize GGML backend");

        const auto model = levo::detail::vae_model::load_gguf(
            gguf, {backend.get(), true, false, true});
        const auto decoder = levo::detail::vae_decoder::create(model);
        const std::vector<float> latent = read_f32(latent_path);
        if (latent.size() % 64 != 0) throw std::runtime_error("latent fixture is not [64,T]");
        const auto actual = decoder->decode(latent, latent.size() / 64, true);
        const std::vector<float> expected = read_f32(audio_path);
        if (actual.audio.size() != expected.size() || actual.stage_outputs.size() != 5) {
            throw std::runtime_error("VAE output/stage count mismatch");
        }
        double squared_error = 0.0;
        double squared_reference = 0.0;
        double dot = 0.0;
        double squared_actual = 0.0;
        float maximum_error = 0.0F;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            if (!std::isfinite(actual.audio[index])) {
                throw std::runtime_error("native VAE produced non-finite audio");
            }
            const double error = static_cast<double>(actual.audio[index]) - expected[index];
            squared_error += error * error;
            squared_reference += static_cast<double>(expected[index]) * expected[index];
            squared_actual += static_cast<double>(actual.audio[index]) * actual.audio[index];
            dot += static_cast<double>(actual.audio[index]) * expected[index];
            maximum_error = std::max(maximum_error, static_cast<float>(std::fabs(error)));
        }
        const double relative_rms = std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
        const double cosine = dot / std::sqrt(std::max(squared_reference * squared_actual, 1.0e-30));
        std::cout << "vae max_error=" << maximum_error
                  << " relative_rms=" << relative_rms
                  << " cosine=" << cosine << '\n';
        if (maximum_error > 3.0e-3F || relative_rms > 1.0e-3 || cosine < 0.99999) {
            throw std::runtime_error("native VAE exceeds the frozen F32 waveform threshold");
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
