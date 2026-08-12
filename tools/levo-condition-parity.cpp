#include "levo-conditioner.h"
#include "levo-model.h"
#include "levo-tokenizer.h"

#include "ggml-backend.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
struct backend_deleter { void operator()(ggml_backend_t value) const noexcept { if (value) ggml_backend_free(value); } };
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

void write(std::ofstream & out, const levo::detail::condition_tensor & tensor) {
    out.write(reinterpret_cast<const char *>(tensor.data()),
              static_cast<std::streamsize>(tensor.size() * sizeof(float)));
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 7) {
            std::cerr << "usage: " << argv[0] << " MODEL.gguf TOKENIZER_DIR LYRICS DESCRIPTION OUTPUT.f32 cpu|cuda\n";
            return 2;
        }
        ggml_backend_load_all();
        const auto type = std::string(argv[6]) == "cuda" ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
        auto * device = ggml_backend_dev_by_type(type);
        if (!device) throw std::runtime_error("requested backend unavailable");
        backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        levo::detail::model_load_options options;
        options.backend = backend.get();
        auto model = levo::detail::model::load_gguf(argv[1], options);
        const std::string root = argv[2];
        auto tokenizer = levo::ByteLevelBPETokenizer::load(
            root + "/vocab.json", root + "/merges.txt", root + "/tokenizer_config.json");
        const char * structures[] = {"[verse]", "[chorus]", "[bridge]", "[intro-short]", "[intro-medium]", "[intro-long]", "[outro-short]", "[outro-medium]", "[outro-long]", "[inst-short]", "[inst-medium]", "[inst-long]", "[silence]"};
        for (int index = 0; index < 13; ++index) tokenizer.add_special_token(structures[index], 151646 + index);
        const auto result = levo::detail::prepare_conditioning(*model, tokenizer, argv[3], argv[4]);
        std::ofstream output(argv[5], std::ios::binary | std::ios::trunc);
        write(output, result.conditional.main);
        write(output, result.conditional.detail);
        write(output, result.null_condition.main);
        write(output, result.null_condition.detail);
        if (!output) throw std::runtime_error("failed writing output");
        std::cout << result.conditional.main.prefix << " " << result.conditional.main.width << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
