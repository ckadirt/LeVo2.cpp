#include "levo-model.h"

#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using levo::detail::gguf_keys::architecture;

struct context_deleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

struct gguf_deleter {
    void operator()(gguf_context * context) const noexcept {
        if (context != nullptr) {
            gguf_free(context);
        }
    }
};

void set_metadata(gguf_context * context, const std::string & architecture_value) {
    using namespace levo::detail;
    gguf_set_val_str(context, gguf_keys::architecture, architecture_value.c_str());
    gguf_set_val_str(context, gguf_keys::name, "tiny test model");
    gguf_set_val_str(context, gguf_keys::source_repository, "https://example.invalid/levo");
    gguf_set_val_str(context, gguf_keys::converter, "test-model.cpp");
    gguf_set_val_str(context, gguf_keys::source_model_sha256,
                     "0000000000000000000000000000000000000000000000000000000000000000");
    gguf_set_val_str(context, gguf_keys::source_config_sha256, "");
    gguf_set_val_u32(context, gguf_keys::schema_version, 1);
    gguf_set_val_u32(context, gguf_keys::file_type, 0);
    gguf_set_val_u32(context, gguf_keys::main_layers, 1);
    gguf_set_val_u32(context, gguf_keys::detail_layers, 1);
    gguf_set_val_u32(context, gguf_keys::embedding_length, 4);
    gguf_set_val_u32(context, gguf_keys::feed_forward_length, 8);
    gguf_set_val_u32(context, gguf_keys::attention_heads, 1);
    gguf_set_val_u32(context, gguf_keys::kv_attention_heads, 1);
    gguf_set_val_u32(context, gguf_keys::context_length, 16);
    gguf_set_val_f32(context, gguf_keys::rms_norm_epsilon, 1.0e-5F);
    gguf_set_val_f32(context, gguf_keys::main_rope_theta, 500000.0F);
    gguf_set_val_f32(context, gguf_keys::detail_rope_theta, 500000.0F);
    gguf_set_val_u32(context, gguf_keys::codebook_count, 3);
    gguf_set_val_u32(context, gguf_keys::codebook_size, 2);
    gguf_set_val_u32(context, gguf_keys::eos_token_id, 2);
    gguf_set_val_u32(context, gguf_keys::special_token_id, 3);
    gguf_set_val_u32(context, gguf_keys::frame_rate, 25);
    gguf_set_val_u32(context, gguf_keys::sample_rate, 48000);
    const int32_t delays[] = {0, 1, 1};
    gguf_set_arr_data(context, gguf_keys::delays, GGUF_TYPE_INT32, delays, 3);
    gguf_set_val_u32(context, gguf_keys::lyrics_prefix_length, 0);
    gguf_set_val_u32(context, gguf_keys::prompt_prefix_length, 0);
    gguf_set_val_u32(context, gguf_keys::style_prefix_length, 0);
}

void write_fixture(const std::filesystem::path & path, const std::string & architecture_value) {
    std::unique_ptr<gguf_context, gguf_deleter> gguf(gguf_init_empty());
    if (!gguf) {
        throw std::runtime_error("cannot make GGUF fixture");
    }
    set_metadata(gguf.get(), architecture_value);
    const ggml_init_params params = {2 * ggml_tensor_overhead(), nullptr, false};
    std::unique_ptr<ggml_context, context_deleter> context(ggml_init(params));
    if (!context) {
        throw std::runtime_error("cannot make tensor fixture");
    }
    ggml_tensor * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 1);
    ggml_set_name(tensor, "fixture.scalar");
    const float value = 42.0F;
    ggml_set_f32_1d(tensor, 0, value);
    gguf_add_tensor(gguf.get(), tensor);
    gguf_set_tensor_data(gguf.get(), "fixture.scalar", &value);
    if (!gguf_write_to_file(gguf.get(), path.string().c_str(), false)) {
        throw std::runtime_error("cannot write GGUF fixture");
    }
}

bool throws(const std::function<void()> & function) {
    try {
        function();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    try {
        using namespace levo::detail;
        const std::filesystem::path root = std::filesystem::temp_directory_path() / "levo-model-test";
        const std::filesystem::path good = root.string() + ".gguf";
        const std::filesystem::path bad = root.string() + "-bad.gguf";
        std::filesystem::remove(good);
        std::filesystem::remove(bad);
        write_fixture(good, "levo2");
        write_fixture(bad, "not-levo2");

        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            throw std::runtime_error("cannot initialize CPU backend");
        }
        const model_load_options options = {backend, false, true, true};
        {
            const auto loaded = model::load_gguf(good.string(), options);
            if (!loaded->contains("fixture.scalar") || loaded->tensor_count() != 1) {
                throw std::runtime_error("loaded tensor inventory is wrong");
            }
            float value = 0.0F;
            ggml_backend_tensor_get(loaded->tensor("fixture.scalar"), &value, 0, sizeof(value));
            if (value != 42.0F) {
                throw std::runtime_error("loaded tensor payload is wrong");
            }
            if (!throws([&] { (void) model::load_gguf(bad.string(), options); })) {
                throw std::runtime_error("loader accepted a mismatched architecture");
            }
        }
        ggml_backend_free(backend);
        std::filesystem::remove(good);
        std::filesystem::remove(bad);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
