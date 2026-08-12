#include "levo-flow-model.h"

#include "ggml-cpu.h"
#include "gguf.h"

#include <exception>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

template <typename Function>
bool throws(const Function & function) {
    try {
        function();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

struct context_deleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) ggml_free(context);
    }
};
struct gguf_deleter {
    void operator()(gguf_context * context) const noexcept {
        if (context != nullptr) gguf_free(context);
    }
};

void flow_metadata(gguf_context * gguf, const char * architecture) {
    gguf_set_val_str(gguf, "general.architecture", architecture);
    gguf_set_val_str(gguf, "general.name", "Flow test fixture");
    gguf_set_val_u32(gguf, "general.file_type", 0);
    gguf_set_val_u32(gguf, "levo2.schema_version", 1);
    gguf_set_val_str(gguf, "levo2.converter", "test-flow-model.cpp");
    gguf_set_val_str(gguf, "levo2.source.runtime_repository", "example");
    gguf_set_val_str(gguf, "levo2.source.runtime_revision", "example");
    gguf_set_val_str(gguf, "levo2.source.levo_repository", "example");
    gguf_set_val_str(gguf, "levo2.source.levo_revision", "example");
    gguf_set_val_str(gguf, "levo2.source.ggml_revision", "example");
    gguf_set_val_str(gguf, "levo2.source.model_sha256", "example");
    gguf_set_val_str(gguf, "levo2.flow.parameter_dtype", "F32");
    const std::pair<const char *, uint32_t> integers[] = {
        {"hidden_size", 4}, {"n_layer", 1}, {"n_head", 1}, {"head_dim", 4}, {"intermediate_size", 8},
        {"max_frames", 8}, {"codebook_size", 4}, {"codebook_dim", 2}, {"condition_dim", 3}, {"latent_dim", 2},
        {"mask_dim", 1}, {"time_embedding_dim", 2}, {"euler_steps_default", 1}, {"sample_rate", 48000}, {"frame_rate", 25},
    };
    for (const auto & item : integers) gguf_set_val_u32(gguf, ("levo2.flow." + std::string(item.first)).c_str(), item.second);
    gguf_set_val_f32(gguf, "levo2.flow.cfg_default", 1.5F);
    gguf_set_val_f32(gguf, "levo2.flow.rope_theta", 10000.0F);
    gguf_set_val_f32(gguf, "levo2.flow.time_embedding_scale", 1000.0F);
}

void write_fixture(const std::filesystem::path & path, const char * architecture) {
    std::unique_ptr<gguf_context, gguf_deleter> gguf(gguf_init_empty());
    if (!gguf) throw std::runtime_error("cannot create GGUF fixture");
    flow_metadata(gguf.get(), architecture);
    std::unique_ptr<ggml_context, context_deleter> context(ggml_init({2 * ggml_tensor_overhead(), nullptr, false}));
    if (!context) throw std::runtime_error("cannot create tensor fixture");
    ggml_tensor * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 1);
    ggml_set_name(tensor, "fixture.scalar");
    const float value = 42.0F;
    gguf_add_tensor(gguf.get(), tensor);
    gguf_set_tensor_data(gguf.get(), "fixture.scalar", &value);
    if (!gguf_write_to_file(gguf.get(), path.string().c_str(), false)) throw std::runtime_error("cannot write GGUF fixture");
}

} // namespace

int main() {
    try {
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend == nullptr) throw std::runtime_error("cannot initialize CPU backend");
        const levo::flow::flow_hparams hparams;
        const std::unordered_map<std::string, levo::flow::tensor_data> tensors = {
            {"flow.test.weight", {{2, 3}, {1, 2, 3, 4, 5, 6}}},
        };
        const auto model = levo::flow::model::make_test_model(hparams, tensors, backend);
        if (model->tensor_count() != 1 || !model->contains("flow.test.weight") || model->tensor("missing") != nullptr) {
            throw std::runtime_error("test Flow model inventory is wrong");
        }
        float values[6] = {};
        ggml_backend_tensor_get(model->tensor("flow.test.weight"), values, 0, sizeof(values));
        if (values[0] != 1.0F || values[5] != 6.0F || model->hparams().hidden_size != 2200) {
            throw std::runtime_error("test Flow model payload is wrong");
        }
        levo::flow::flow_hparams invalid = hparams;
        invalid.head_dim = 109;
        if (!throws([&] { (void) levo::flow::model::make_test_model(invalid, tensors, backend); })) {
            throw std::runtime_error("invalid Flow hparams were accepted");
        }
        const auto root = std::filesystem::temp_directory_path() / "levo-flow-model-test";
        const auto bad = root.string() + "-bad.gguf";
        write_fixture(bad, "wrong");
        const levo::flow::load_options options = {backend, false, true, true};
        if (!throws([&] { (void) levo::flow::model::load_gguf(bad, options); })) throw std::runtime_error("wrong Flow architecture accepted");
        std::filesystem::remove(bad);
        const char * real = std::getenv("LEVO_FLOW_GGUF_FIXTURE");
        if (real != nullptr && *real != '\0') {
            const auto loaded = levo::flow::model::load_gguf(
                real, {backend, true, true, true});
            if (loaded->tensor_count() != 231 ||
                !loaded->contains("flow.block.15.ffn.out.weight") ||
                !loaded->contains("flow.rvq.vocal.codebook.weight")) {
                throw std::runtime_error("real Flow GGUF inventory is incomplete");
            }
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
