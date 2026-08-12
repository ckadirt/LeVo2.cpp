#include "levo-kv.h"

#include "ggml-cpu.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

tensor_data ramp(std::vector<int64_t> shape, float scale, float bias = 0.0F) {
    std::size_t count = 1;
    for (const int64_t dimension : shape) {
        count *= static_cast<std::size_t>(dimension);
    }
    tensor_data result;
    result.dimensions = std::move(shape);
    result.values.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.values[index] = bias + scale * static_cast<float>(static_cast<int>(index % 11) - 5);
    }
    return result;
}

tensor_data identity(int32_t dimension, float scale = 1.0F) {
    tensor_data result = filled({dimension, dimension}, 0.0F);
    for (int32_t row = 0; row < dimension; ++row) {
        result.values[static_cast<std::size_t>(row) * dimension + row] = scale;
    }
    return result;
}

void expect_close(const std::vector<float> & expected, const std::vector<float> & actual,
                  const char * name) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error(std::string(name) + " output shape differs");
    }
    // CUDA F16 cache writes are separately rounded between prefix prefill and
    // BOS decode, while the combined graph rounds a larger append at once.
    // The two legal reduction paths differ by a few FP16 ulps.
    constexpr float tolerance = 1.0e-4F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (std::fabs(expected[index] - actual[index]) > tolerance) {
            throw std::runtime_error(std::string(name) + " cached result differs at element " +
                                     std::to_string(index) + " (expected " +
                                     std::to_string(expected[index]) + ", got " +
                                     std::to_string(actual[index]) + ")");
        }
    }
}

void expect_last_close(const std::vector<float> & full, const std::vector<float> & step,
                       int32_t vocabulary_size, const char * name) {
    if (step.size() != static_cast<std::size_t>(vocabulary_size) ||
        full.size() < static_cast<std::size_t>(vocabulary_size)) {
        throw std::runtime_error(std::string(name) + " output shape is invalid");
    }
    const std::size_t begin = full.size() - static_cast<std::size_t>(vocabulary_size);
    constexpr float tolerance = 2.0e-5F;
    for (int32_t index = 0; index < vocabulary_size; ++index) {
        if (std::fabs(full[begin + static_cast<std::size_t>(index)] -
                      step[static_cast<std::size_t>(index)]) > tolerance) {
            throw std::runtime_error(std::string(name) + " decode result differs at element " +
                                     std::to_string(index));
        }
    }
}

void expect_tail_close(const std::vector<float> & full, const std::vector<float> & tail,
                       int32_t vocabulary_size, int32_t tail_tokens, const char * name) {
    const std::size_t tail_elements = static_cast<std::size_t>(vocabulary_size) * tail_tokens;
    if (tail.size() != tail_elements || full.size() < tail_elements) {
        throw std::runtime_error(std::string(name) + " output shape is invalid");
    }
    const std::size_t begin = full.size() - tail_elements;
    constexpr float tolerance = 2.0e-5F;
    for (std::size_t index = 0; index < tail_elements; ++index) {
        if (std::fabs(full[begin + index] - tail[index]) > tolerance) {
            throw std::runtime_error(std::string(name) + " conditioned result differs at element " +
                                     std::to_string(index));
        }
    }
}

std::vector<float> embedding_column(const tensor_data & table, int32_t token, int32_t width) {
    const std::size_t begin = static_cast<std::size_t>(token) * width;
    return {table.values.begin() + static_cast<std::ptrdiff_t>(begin),
            table.values.begin() + static_cast<std::ptrdiff_t>(begin + width)};
}

void add_tower_tensors(std::unordered_map<std::string, tensor_data> & tensors,
                       bool detail, const model_hparams & hparams) {
    const auto layer_name = [detail](unsigned layer, const char * suffix) {
        return detail ? levo::detail::tensor_names::detail_layer(layer, suffix)
                      : levo::detail::tensor_names::main_layer(layer, suffix);
    };
    const int32_t layers = detail ? hparams.detail_layers : hparams.main_layers;
    const int32_t width = hparams.embedding_length;
    const int32_t kv_width = hparams.kv_attention_heads * hparams.head_dimension();
    for (int32_t layer = 0; layer < layers; ++layer) {
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "attn_norm.weight"), filled({width}, 1.0F));
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "attn_q.weight"), identity(width, 0.7F));
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "attn_k.weight"), identity(kv_width, 0.5F));
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "attn_v.weight"), identity(kv_width, 0.6F));
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "attn_output.weight"), identity(width, 0.8F));
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "ffn_norm"), filled({width}, 1.0F));
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "ffn_gate.weight"), ramp({width, hparams.feed_forward_length}, 0.03F));
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "ffn_up.weight"), ramp({width, hparams.feed_forward_length}, 0.02F, 0.01F));
        tensors.emplace(layer_name(static_cast<unsigned>(layer), "ffn_down.weight"), ramp({hparams.feed_forward_length, width}, 0.02F));
    }
    tensors.emplace(detail ? levo::detail::tensor_names::detail_norm : levo::detail::tensor_names::main_norm,
                    filled({width}, 1.0F));
}

void test_conditioned_prefix_then_bos(const model_hparams & hparams,
                                      const std::unordered_map<std::string, tensor_data> & tensors,
                                      ggml_backend_t backend, const char * backend_name) {
    using namespace levo::detail;
    const auto weights = model::make_test_model(hparams, tensors, backend);
    condition_pair condition;
    condition.main.width = hparams.embedding_length;
    condition.main.prefix = 1;
    condition.main.values = embedding_column(tensors.at(tensor_names::mixed_embedding), 2,
                                             hparams.embedding_length);
    condition.detail.width = hparams.embedding_length;
    condition.detail.prefix = 1;
    const std::vector<float> conditioned_vocal = embedding_column(
        tensors.at(tensor_names::detail_vocal_embedding), 0, hparams.embedding_length);
    const std::vector<float> conditioned_bgm = embedding_column(
        tensors.at(tensor_names::detail_bgm_embedding), 1, hparams.embedding_length);
    condition.detail.values.resize(static_cast<std::size_t>(hparams.embedding_length));
    for (int32_t index = 0; index < hparams.embedding_length; ++index) {
        condition.detail.values[static_cast<std::size_t>(index)] =
            conditioned_vocal[static_cast<std::size_t>(index)] +
            conditioned_bgm[static_cast<std::size_t>(index)];
    }

    lm_input bos;
    bos.mixed_tokens = {hparams.special_token_id};
    bos.vocal_tokens = {hparams.special_token_id};
    bos.bgm_tokens = {hparams.special_token_id};
    const auto combined = kv_session::create(weights);
    const lm_output combined_bos = combined->prefill_conditioned(condition, bos);

    const auto split = kv_session::create(weights);
    split->prefill_conditioned_prefix(condition);
    if (split->token_count() != 1 || split->next_position() != 1) {
        throw std::runtime_error(std::string(backend_name) + " prefix-only prefill did not advance cache state");
    }
    const lm_output split_bos = split->decode(hparams.special_token_id, hparams.special_token_id,
                                              hparams.special_token_id);
    const std::string mixed_name = std::string(backend_name) + " mixed prefix-only BOS";
    const std::string vocal_name = std::string(backend_name) + " vocal prefix-only BOS";
    const std::string bgm_name = std::string(backend_name) + " BGM prefix-only BOS";
    expect_close(combined_bos.mixed_logits, split_bos.mixed_logits, mixed_name.c_str());
    expect_close(combined_bos.vocal_logits, split_bos.vocal_logits, vocal_name.c_str());
    expect_close(combined_bos.bgm_logits, split_bos.bgm_logits, bgm_name.c_str());
    if (split->token_count() != 2 || split->next_position() != 2) {
        throw std::runtime_error(std::string(backend_name) + " BOS decode did not advance cache state");
    }

    split->reset();
    if (split->token_count() != 0 || split->next_position() != 0) {
        throw std::runtime_error(std::string(backend_name) + " reset did not clear cache state");
    }
    split->prefill_conditioned_prefix(condition);
    const lm_output reset_bos = split->decode(hparams.special_token_id, hparams.special_token_id,
                                              hparams.special_token_id);
    const std::string reset_name = std::string(backend_name) + " reset prefix-only BOS";
    expect_close(combined_bos.mixed_logits, reset_bos.mixed_logits, (reset_name + " mixed").c_str());
    expect_close(combined_bos.vocal_logits, reset_bos.vocal_logits, (reset_name + " vocal").c_str());
    expect_close(combined_bos.bgm_logits, reset_bos.bgm_logits, (reset_name + " BGM").c_str());
}

} // namespace

int main() {
    try {
        using namespace levo::detail;
        model_hparams hparams;
        hparams.main_layers = 1;
        hparams.detail_layers = 1;
        hparams.embedding_length = 4;
        hparams.feed_forward_length = 4;
        hparams.attention_heads = 1;
        hparams.kv_attention_heads = 1;
        hparams.context_length = 8;
        hparams.codebook_size = 3;
        hparams.eos_token_id = 3;
        hparams.special_token_id = 4;
        hparams.delays = {0, 1, 1};
        hparams.lyrics_prefix_length = 0;
        hparams.prompt_prefix_length = 0;
        hparams.style_prefix_length = 0;

        std::unordered_map<std::string, tensor_data> tensors;
        tensors.emplace(tensor_names::mixed_embedding, ramp({4, 5}, 0.08F, 0.04F));
        tensors.emplace(tensor_names::detail_vocal_embedding, ramp({4, 5}, 0.06F, 0.02F));
        tensors.emplace(tensor_names::detail_bgm_embedding, ramp({4, 5}, 0.05F, -0.01F));
        tensors.emplace(tensor_names::mixed_output, ramp({4, 4}, 0.04F));
        tensors.emplace(tensor_names::vocal_output, ramp({4, 4}, 0.03F, 0.01F));
        tensors.emplace(tensor_names::bgm_output, ramp({4, 4}, 0.02F, -0.01F));
        tensors.emplace(tensor_names::bridge_0_weight, ramp({8, 4}, 0.025F, 0.01F));
        tensors.emplace(tensor_names::bridge_0_bias, filled({4}, 0.01F));
        tensors.emplace(tensor_names::bridge_2_weight, identity(4, 0.9F));
        tensors.emplace(tensor_names::bridge_2_bias, filled({4}, 0.005F));
        add_tower_tensors(tensors, false, hparams);
        add_tower_tensors(tensors, true, hparams);

        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            throw std::runtime_error("cannot initialize CPU backend");
        }
        try {
            const auto weights = model::make_test_model(hparams, tensors, backend);
            const auto uncached = lm::create(weights);
            const auto cached = kv_session::create(weights);

            lm_input prefix;
            prefix.mixed_tokens = {0, 1, 2};
            prefix.vocal_tokens = {1, 2, 0};
            prefix.bgm_tokens = {2, 0, 1};
            // A nonzero first RoPE position catches implementations that use
            // the cache index as the position during decode.
            prefix.positions = {2, 3, 4};
            const lm_output expected_prefix = uncached->forward(prefix);
            const lm_output actual_prefix = cached->prefill(prefix);
            expect_close(expected_prefix.mixed_logits, actual_prefix.mixed_logits, "mixed prefill");
            expect_close(expected_prefix.vocal_logits, actual_prefix.vocal_logits, "vocal prefill");
            expect_close(expected_prefix.bgm_logits, actual_prefix.bgm_logits, "BGM prefill");

            // The opt-in emulation mode is intentionally inert for a
            // synthetic F32 model.  This protects callers that use the flag
            // generically while retaining the established F32 graph.
            const auto emulated_f32 = kv_session::create(weights, true);
            const lm_output actual_emulated_f32 = emulated_f32->prefill(prefix);
            expect_close(expected_prefix.mixed_logits, actual_emulated_f32.mixed_logits,
                         "mixed F32 emulation prefill");
            expect_close(expected_prefix.vocal_logits, actual_emulated_f32.vocal_logits,
                         "vocal F32 emulation prefill");
            expect_close(expected_prefix.bgm_logits, actual_emulated_f32.bgm_logits,
                         "BGM F32 emulation prefill");

            lm_input full = prefix;
            full.mixed_tokens.push_back(1);
            full.vocal_tokens.push_back(0);
            full.bgm_tokens.push_back(2);
            full.positions.push_back(5);
            const lm_output expected_full = uncached->forward(full);
            const lm_output actual_step = cached->decode(1, 0, 2);
            expect_last_close(expected_full.mixed_logits, actual_step.mixed_logits,
                              expected_full.vocabulary_size, "mixed decode");
            expect_last_close(expected_full.vocal_logits, actual_step.vocal_logits,
                              expected_full.vocabulary_size, "vocal decode");
            expect_last_close(expected_full.bgm_logits, actual_step.bgm_logits,
                              expected_full.vocabulary_size, "BGM decode");
            if (cached->token_count() != 4 || cached->next_position() != 6 || cached->capacity() != 8) {
                throw std::runtime_error("KV session did not advance its continuous absolute position");
            }

            // A dense one-position condition constructed from the same three
            // embedding rows as an ordinary first token must be exactly
            // equivalent to uncached execution over that token plus audio.
            // This exercises the required independent main/detail temporal
            // concatenations before the bridge width concatenation.
            condition_pair condition;
            condition.main.width = hparams.embedding_length;
            condition.main.prefix = 1;
            condition.main.values = embedding_column(tensors.at(tensor_names::mixed_embedding), 2,
                                                     hparams.embedding_length);
            condition.detail.width = hparams.embedding_length;
            condition.detail.prefix = 1;
            const std::vector<float> conditioned_vocal = embedding_column(
                tensors.at(tensor_names::detail_vocal_embedding), 0, hparams.embedding_length);
            const std::vector<float> conditioned_bgm = embedding_column(
                tensors.at(tensor_names::detail_bgm_embedding), 1, hparams.embedding_length);
            condition.detail.values.resize(static_cast<std::size_t>(hparams.embedding_length));
            for (int32_t index = 0; index < hparams.embedding_length; ++index) {
                condition.detail.values[static_cast<std::size_t>(index)] =
                    conditioned_vocal[static_cast<std::size_t>(index)] +
                    conditioned_bgm[static_cast<std::size_t>(index)];
            }
            lm_input conditioned_audio;
            conditioned_audio.mixed_tokens = {1, 0};
            conditioned_audio.vocal_tokens = {2, 1};
            conditioned_audio.bgm_tokens = {0, 2};
            lm_input equivalent_full;
            equivalent_full.mixed_tokens = {2, 1, 0};
            equivalent_full.vocal_tokens = {0, 2, 1};
            equivalent_full.bgm_tokens = {1, 0, 2};
            const lm_output expected_conditioned = uncached->forward(equivalent_full);
            const auto conditioned = kv_session::create(weights);
            const lm_output actual_conditioned = conditioned->prefill_conditioned(condition, conditioned_audio);
            expect_tail_close(expected_conditioned.mixed_logits, actual_conditioned.mixed_logits,
                              expected_conditioned.vocabulary_size, 2, "mixed conditioned prefill");
            expect_tail_close(expected_conditioned.vocal_logits, actual_conditioned.vocal_logits,
                              expected_conditioned.vocabulary_size, 2, "vocal conditioned prefill");
            expect_tail_close(expected_conditioned.bgm_logits, actual_conditioned.bgm_logits,
                              expected_conditioned.vocabulary_size, 2, "BGM conditioned prefill");
            if (conditioned->token_count() != 3 || conditioned->next_position() != 3) {
                throw std::runtime_error("conditioned prefill did not populate both KV caches");
            }
            test_conditioned_prefix_then_bos(hparams, tensors, backend, "CPU");
        } catch (...) {
            ggml_backend_free(backend);
            throw;
        }
        ggml_backend_free(backend);
#if LEVO_HAS_CUDA
        ggml_backend_load_all();
        ggml_backend_dev_t cuda_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (cuda_device == nullptr) {
            throw std::runtime_error("CUDA build did not register a GPU backend");
        }
        ggml_backend_t cuda_backend = ggml_backend_dev_init(cuda_device, nullptr);
        if (cuda_backend == nullptr) {
            throw std::runtime_error("cannot initialize CUDA backend");
        }
        try {
            test_conditioned_prefix_then_bos(hparams, tensors, cuda_backend, "CUDA");
        } catch (...) {
            ggml_backend_free(cuda_backend);
            throw;
        }
        ggml_backend_free(cuda_backend);
#endif
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
