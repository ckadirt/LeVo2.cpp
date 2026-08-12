#include "levo-conditioner.h"

#include "ggml-cpu.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifndef LEVO_TEST_FIXTURES_DIR
#define LEVO_TEST_FIXTURES_DIR "tests/fixtures"
#endif

namespace {

std::vector<float> table(int64_t width, int64_t rows, float scale) {
    std::vector<float> values(static_cast<std::size_t>(width * rows));
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < width; ++col) {
            values[static_cast<std::size_t>(row * width + col)] = scale * row + static_cast<float>(col);
        }
    }
    return values;
}

void expect_close(float actual, float expected) {
    if (std::fabs(actual - expected) > 1.0e-6F) {
        throw std::runtime_error("conditioner value mismatch: " + std::to_string(actual) + " != " + std::to_string(expected));
    }
}

} // namespace

int main() {
    try {
        using namespace levo;
        using namespace levo::detail;
        const std::string fixture_dir = LEVO_TEST_FIXTURES_DIR;
        ByteLevelBPETokenizer tokenizer = ByteLevelBPETokenizer::load(
            fixture_dir + "/tiny-vocab.json", fixture_dir + "/tiny-merges.txt",
            fixture_dir + "/tiny-tokenizer-config.json");
        const char * structures[] = {
            "[verse]", "[chorus]", "[bridge]", "[intro-short]", "[intro-medium]",
            "[intro-long]", "[outro-short]", "[outro-medium]", "[outro-long]",
            "[inst-short]", "[inst-medium]", "[inst-long]", "[silence]",
        };
        for (int i = 0; i < 13; ++i) {
            tokenizer.add_special_token(structures[i], 151646 + i);
        }

        model_hparams hp;
        hp.embedding_length = 2;
        hp.feed_forward_length = 2;
        hp.attention_heads = 1;
        hp.kv_attention_heads = 1;
        hp.codebook_size = 3;
        hp.eos_token_id = 3;
        hp.special_token_id = 4;
        std::unordered_map<std::string, tensor_data> tensors;
        tensors.emplace("cond.lyrics.weight", tensor_data{{2, 151659}, table(2, 151659, 10.0F)});
        tensors.emplace("cond.structure.weight", tensor_data{{2, 200}, table(2, 200, 100.0F)});
        tensors.emplace("cond.style.weight", tensor_data{{2, 151652}, table(2, 151652, 1000.0F)});
        tensors.emplace("cond.prompt_embd.mixed.weight", tensor_data{{2, hp.token_input_size()}, table(2, hp.token_input_size(), 2.0F)});
        tensors.emplace("cond.prompt_embd.vocal.weight", tensor_data{{2, hp.token_input_size()}, table(2, hp.token_input_size(), 3.0F)});
        tensors.emplace("cond.prompt_embd.bgm.weight", tensor_data{{2, hp.token_input_size()}, table(2, hp.token_input_size(), 5.0F)});
        tensors.emplace("cond.prompt_eot.mixed", tensor_data{{2, 1}, {70.0F, 71.0F}});
        tensors.emplace("cond.prompt_eot.detail", tensor_data{{2, 1}, {80.0F, 81.0F}});

        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            throw std::runtime_error("cannot initialize CPU backend");
        }
        const auto weights = model::make_test_model(hp, tensors, backend);
        const conditioning_result result = prepare_conditioning(
            *weights, tokenizer, "[verse]a[chorus]b", "a");
        if (result.conditional.main.width != 2 || result.conditional.main.prefix != 952 ||
            result.conditional.main.values.size() != 1904 ||
            result.conditional.detail.values.size() != 1904) {
            throw std::runtime_error("condition shape mismatch");
        }

        // `<|im_start|>` is column zero.  The verse row (151646) carries
        // structure row one, and the chorus row (151647) carries row two.
        expect_close(result.conditional.main.values[0], 190.0F);
        expect_close(result.conditional.main.values[2], 1516460.0F + 100.0F);
        // The first lyric column after the verse tag remains in verse until
        // the next structure boundary.
        expect_close(result.conditional.main.values[4], 10.0F + 100.0F);
        // Coverage ends with the actual text. Padding after the last section
        // must use structure row zero, not inherit the chorus embedding.
        expect_close(result.conditional.main.values[10], 1516430.0F);
        // Prompt starts after 600 lyric columns: EOT, then 251 special frames.
        const std::size_t prompt = 600U * 2U;
        expect_close(result.conditional.main.values[prompt], 70.0F);
        expect_close(result.conditional.main.values[prompt + 2], 8.0F);
        expect_close(result.conditional.main.values[prompt + 4], 8.0F);
        expect_close(result.conditional.detail.values[prompt], 80.0F);
        expect_close(result.conditional.detail.values[prompt + 2], 32.0F);
        expect_close(result.conditional.detail.values[prompt + 4], 32.0F);

        // The null branch has only the literal im_start token in text and is
        // deterministic.  Its prompt remains EOT + special-frame embeddings.
        expect_close(result.null_condition.main.values[0], 190.0F);
        expect_close(result.null_condition.main.values[prompt], 70.0F);
        expect_close(result.null_condition.main.values[prompt + 2], 8.0F);
        ggml_backend_free(backend);
        std::cout << "conditioner ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
