#include "levo-lm.h"

#include "ggml-cpu.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using levo::detail::model_hparams;
using levo::detail::tensor_data;

tensor_data filled(std::vector<int64_t> shape, float value) {
    std::size_t count = 1;
    for (const int64_t dimension : shape) {
        count *= static_cast<std::size_t>(dimension);
    }
    return {std::move(shape), std::vector<float>(count, value)};
}

tensor_data identity(int32_t dimension) {
    tensor_data result = filled({dimension, dimension}, 0.0F);
    for (int32_t row = 0; row < dimension; ++row) {
        result.values[static_cast<std::size_t>(row) * dimension + row] = 1.0F;
    }
    return result;
}

} // namespace

int main() {
    try {
        using namespace levo::detail;
        model_hparams hparams;
        hparams.main_layers = 0;
        hparams.detail_layers = 0;
        hparams.embedding_length = 2;
        hparams.feed_forward_length = 2;
        hparams.attention_heads = 1;
        hparams.kv_attention_heads = 1;
        hparams.context_length = 8;
        hparams.codebook_size = 2;
        hparams.eos_token_id = 2;
        hparams.special_token_id = 3;
        hparams.delays = {0, 1, 1};
        hparams.lyrics_prefix_length = 0;
        hparams.prompt_prefix_length = 0;
        hparams.style_prefix_length = 0;

        std::unordered_map<std::string, tensor_data> tensors;
        tensors.emplace(tensor_names::mixed_embedding,
                        tensor_data{{2, 4}, {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F}});
        tensors.emplace(tensor_names::detail_vocal_embedding,
                        tensor_data{{2, 4}, {1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F}});
        tensors.emplace(tensor_names::detail_bgm_embedding,
                        tensor_data{{2, 4}, {0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F}});
        tensors.emplace(tensor_names::main_norm, filled({2}, 1.0F));
        tensors.emplace(tensor_names::detail_norm, filled({2}, 1.0F));
        tensors.emplace(tensor_names::mixed_output, filled({2, 3}, 1.0F));
        tensors.emplace(tensor_names::vocal_output, filled({2, 3}, 1.0F));
        tensors.emplace(tensor_names::bgm_output, filled({2, 3}, 1.0F));
        tensors.emplace(tensor_names::bridge_0_weight,
                        tensor_data{{4, 2}, {1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F}});
        tensors.emplace(tensor_names::bridge_0_bias, filled({2}, 0.0F));
        tensors.emplace(tensor_names::bridge_2_weight, identity(2));
        tensors.emplace(tensor_names::bridge_2_bias, filled({2}, 0.0F));

        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            throw std::runtime_error("cannot initialize CPU backend");
        }
        {
            const auto model = model::make_test_model(hparams, tensors, backend);
            const auto runtime = lm::create(model);
            lm_input input;
            input.mixed_tokens = {0, 1};
            input.vocal_tokens = {0, 1};
            input.bgm_tokens = {1, 0};
            const lm_output output = runtime->forward(input);
            if (output.token_count != 2 || output.vocabulary_size != 3 ||
                output.mixed_logits.size() != 6 || output.vocal_logits.size() != 6 ||
                output.bgm_logits.size() != 6) {
                throw std::runtime_error("unexpected output shape");
            }
            for (const float value : output.mixed_logits) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite mixed output");
                }
            }
            for (const float value : output.vocal_logits) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite vocal output");
                }
            }
            for (const float value : output.bgm_logits) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite BGM output");
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
