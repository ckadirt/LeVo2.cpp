#include "levo-flow-conditioning.h"

#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("cannot write " + path);
    file.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) throw std::runtime_error("cannot finish " + path);
}

} // namespace

int main() {
    try {
        const char * model_path = std::getenv("LEVO_FLOW_GGUF");
        if (model_path == nullptr || *model_path == '\0') {
            // The strict model is multi-gigabyte, so the regular CTest suite
            // remains asset-free. CI that provisions it gets the real gate.
            return 0;
        }
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("cannot initialize CPU backend");
        {
            const auto weights = levo::flow::model::load_gguf(model_path, {backend, true, true, false});
            levo::flow::conditioning boundary(weights);
            levo::flow::conditioning_input input;
            input.vocal_codes = {0, 0};
            input.bgm_codes = {0, 0};
            input.mask_ids = {2, 2};
            input.raw_incontext_latent.assign(2 * 64, 0.0F);
            input.initial_noise.assign(2 * 64, 0.0F);
            const levo::flow::conditioning_output output = boundary.prepare(input);
            if (output.frames != 2 || output.vocal_codebook_lookup.size() != 64 ||
                output.vocal_projected.size() != 2048 || output.conditioning.size() != 4096 ||
                output.mask_embedding.size() != 48 || output.positional_embedding.size() != 4400) {
                throw std::runtime_error("unexpected Flow conditioning output dimensions");
            }
            const std::vector<float> restored = boundary.denormalize_latents(boundary.normalize_latents(input.raw_incontext_latent, 2), 2);
            for (float value : restored) if (std::abs(value) > 1.0e-2F) throw std::runtime_error("Flow latent normalization inverse is inaccurate");
            if (const char * dump = std::getenv("LEVO_FLOW_CONDITIONING_DUMP")) {
                write_f32(std::string(dump) + ".vocal_lookup.f32", output.vocal_codebook_lookup);
                write_f32(std::string(dump) + ".bgm_lookup.f32", output.bgm_codebook_lookup);
                write_f32(std::string(dump) + ".vocal_projected.f32", output.vocal_projected);
                write_f32(std::string(dump) + ".bgm_projected.f32", output.bgm_projected);
                write_f32(std::string(dump) + ".mask.f32", output.mask_embedding);
                write_f32(std::string(dump) + ".positional.f32", output.positional_embedding);
                write_f32(std::string(dump) + ".null.f32", output.null_condition);
            }
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
