#include "levo-lm.h"
#include "levo-model.h"

#include "ggml-backend.h"

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct backend_deleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend != nullptr) ggml_backend_free(backend);
    }
};
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

ggml_backend_dev_t select_device(const std::string & requested) {
    ggml_backend_load_all();
    if (requested == "cuda") {
        if (auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) return device;
        throw std::runtime_error("CUDA backend is unavailable");
    }
    if (requested == "cpu") {
        if (auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) return device;
        throw std::runtime_error("CPU backend is unavailable");
    }
    throw std::invalid_argument("backend must be cpu or cuda");
}

void write_values(std::ofstream & output, const std::vector<float> & values) {
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) throw std::runtime_error("failed writing parity output");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: " << argv[0] << " MODEL.gguf OUTPUT.f32 cpu|cuda\n";
            return 2;
        }
        backend_ptr backend(ggml_backend_dev_init(select_device(argv[3]), nullptr));
        if (!backend) throw std::runtime_error("failed to initialize backend");
        levo::detail::model_load_options options;
        options.backend = backend.get();
        auto weights = levo::detail::model::load_gguf(argv[1], options);
        auto runtime = levo::detail::lm::create(weights);
        const int32_t special = weights->hparams().special_token_id;
        levo::detail::lm_input input{{special}, {special}, {special}, {0}};
        const auto result = runtime->forward(input);
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open parity output");
        write_values(output, result.mixed_logits);
        write_values(output, result.vocal_logits);
        write_values(output, result.bgm_logits);
        std::cout << result.vocabulary_size << " " << result.token_count << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
