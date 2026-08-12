#include "levo-conditioner.h"
#include "levo-kv.h"
#include "levo-model.h"
#include "levo-sampling.h"
#include "levo-tokenizer.h"

#include "ggml-backend.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct backend_deleter { void operator()(ggml_backend_t value) const noexcept { if (value) ggml_backend_free(value); } };
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

std::vector<float> last(const std::vector<float> & values, int32_t vocabulary) {
    if (values.size() < static_cast<std::size_t>(vocabulary)) throw std::runtime_error("short logits");
    return {values.end() - vocabulary, values.end()};
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 6 || argc > 8) {
            std::cerr << "usage: " << argv[0]
                      << " MODEL.gguf TOKENIZER_DIR LYRICS DESCRIPTION OUTPUT.f32"
                         " [--emulate-fp16] [--streaming-first-token]\n";
            return 2;
        }
        bool emulate_fp16 = false;
        bool streaming_first_token = false;
        for (int index = 6; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--emulate-fp16") emulate_fp16 = true;
            else if (option == "--streaming-first-token") streaming_first_token = true;
            else throw std::runtime_error("unknown option: " + option);
        }
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) throw std::runtime_error("CUDA backend unavailable");
        backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        levo::detail::model_load_options options;
        options.backend = backend.get();
        auto model = levo::detail::model::load_gguf(argv[1], options);
        const std::string root = argv[2];
        auto tokenizer = levo::ByteLevelBPETokenizer::load(root + "/vocab.json", root + "/merges.txt", root + "/tokenizer_config.json");
        const char * structures[] = {"[verse]", "[chorus]", "[bridge]", "[intro-short]", "[intro-medium]", "[intro-long]", "[outro-short]", "[outro-medium]", "[outro-long]", "[inst-short]", "[inst-medium]", "[inst-long]", "[silence]"};
        for (int i = 0; i < 13; ++i) tokenizer.add_special_token(structures[i], 151646 + i);
        const auto conditions = levo::detail::prepare_conditioning(*model, tokenizer, argv[3], argv[4]);
        auto conditional = levo::detail::kv_session::create(model, emulate_fp16);
        auto null_session = levo::detail::kv_session::create(model, emulate_fp16);
        const int32_t special = model->hparams().special_token_id;
        levo::detail::lm_input initial{{special}, {special}, {special}, {952}};
        levo::detail::lm_output cond;
        levo::detail::lm_output uncond;
        if (streaming_first_token) {
            conditional->prefill_conditioned_prefix(conditions.conditional);
            null_session->prefill_conditioned_prefix(conditions.null_condition);
            cond = conditional->decode(special, special, special);
            uncond = null_session->decode(special, special, special);
        } else {
            cond = conditional->prefill_conditioned(conditions.conditional, initial, true);
            uncond = null_session->prefill_conditioned(conditions.null_condition, initial);
        }
        const int32_t vocab = cond.vocabulary_size;
        const std::vector<std::vector<float>> cond_streams = {
            last(cond.mixed_logits, vocab), last(cond.vocal_logits, vocab), last(cond.bgm_logits, vocab)};
        const std::vector<std::vector<float>> null_streams = {
            last(uncond.mixed_logits, vocab), last(uncond.vocal_logits, vocab), last(uncond.bgm_logits, vocab)};
        const auto cfg = levo::apply_cfg(cond_streams, null_streams, 1.5F);
        std::ofstream out(argv[5], std::ios::binary | std::ios::trunc);
        for (const auto & stream : cfg) out.write(reinterpret_cast<const char *>(stream.data()), stream.size() * sizeof(float));
        if (!out) throw std::runtime_error("failed writing CFG logits");
        std::ofstream cond_out(std::string(argv[5]) + ".conditional", std::ios::binary | std::ios::trunc);
        std::ofstream null_out(std::string(argv[5]) + ".unconditional", std::ios::binary | std::ios::trunc);
        for (const auto & stream : cond_streams) {
            cond_out.write(reinterpret_cast<const char *>(stream.data()), stream.size() * sizeof(float));
        }
        for (const auto & stream : null_streams) {
            null_out.write(reinterpret_cast<const char *>(stream.data()), stream.size() * sizeof(float));
        }
        if (!cond_out || !null_out) throw std::runtime_error("failed writing branch logits");
        std::cout << vocab;
        for (const auto & stream : cfg) std::cout << " " << std::distance(stream.begin(), std::max_element(stream.begin(), stream.end()));
        std::cout << "\n";
        if (streaming_first_token) return 0;
        std::ofstream trace(std::string(argv[5]) + ".trace", std::ios::binary | std::ios::trunc);
        const auto write_vector = [&trace](const std::vector<float> & values) {
            trace.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        };
        for (const auto & values : cond.main_layers) write_vector(values);
        write_vector(cond.main_norm);
        write_vector(cond.bridge);
        for (const auto & values : cond.detail_layers) write_vector(values);
        write_vector(cond.detail_norm);
        if (!trace) throw std::runtime_error("failed writing trace");
        for (const auto & entry : cond.main_layer0_ops) {
            const std::filesystem::path path = std::string(argv[5]) + ".ops." + entry.name + ".f32";
            std::ofstream op(path, std::ios::binary | std::ios::trunc);
            op.write(reinterpret_cast<const char *>(entry.values.data()),
                     static_cast<std::streamsize>(entry.values.size() * sizeof(float)));
            if (!op) throw std::runtime_error("failed writing operator trace: " + entry.name);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
