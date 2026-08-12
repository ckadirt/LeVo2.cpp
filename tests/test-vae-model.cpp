#include "levo-vae-model.h"

#include "ggml-cpu.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

int main() {
    try {
        using namespace levo::detail;
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend == nullptr) throw std::runtime_error("cannot initialize CPU backend");
        {
            if (!vae_hparams{}.valid()) throw std::runtime_error("default VAE hparams must be valid");
            vae_hparams wrong;
            wrong.final_tanh = true;
            if (wrong.valid()) throw std::runtime_error("strict VAE hparams accepted final tanh");
            const auto tiny = vae_model::make_test_model(vae_hparams{}, {{"tiny", {{1}, {42.0F}}}}, backend);
            if (tiny->tensor_count() != 1U || !tiny->contains("tiny")) throw std::runtime_error("tiny VAE helper failed");

            const char * fixture = std::getenv("LEVO_VAE_GGUF_FIXTURE");
            if (fixture != nullptr && *fixture != '\0') {
                const auto loaded = vae_model::load_gguf(fixture, {backend, true, true, true});
                if (loaded->tensor_count() != 145U || loaded->decoder().input.weight == nullptr ||
                    loaded->decoder().stage[0].upsample.conv.weight == nullptr ||
                    loaded->decoder().stage[4].residual[2].conv_1.weight == nullptr ||
                    loaded->decoder().output_activation.alpha_log == nullptr ||
                    loaded->decoder().output_weight == nullptr) {
                    throw std::runtime_error("loaded VAE decoder bindings are incomplete");
                }
            }
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
