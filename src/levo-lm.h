#pragma once

#include "levo-model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace levo::detail {

struct lm_input {
    // All streams have exactly token_count values.  Positions default to a
    // contiguous sequence [0, token_count) when omitted.
    std::vector<int32_t> mixed_tokens;
    std::vector<int32_t> vocal_tokens;
    std::vector<int32_t> bgm_tokens;
    std::vector<int32_t> positions;
};

struct lm_output {
    int32_t token_count = 0;
    int32_t vocabulary_size = 0;
    // GGML order [vocabulary_size, token_count].
    std::vector<float> mixed_logits;
    std::vector<float> vocal_logits;
    std::vector<float> bgm_logits;
    // Optional parity trace. Production calls leave these empty.
    std::vector<std::vector<float>> main_layers;
    std::vector<float> main_norm;
    std::vector<float> bridge;
    std::vector<std::vector<float>> detail_layers;
    std::vector<float> detail_norm;
    // Fine-grained first-main-layer trace used only by the parity diagnostic
    // executable.  Each entry is a flattened F32 tensor with its stable
    // operator name; normal inference never populates this vector.
    struct named_trace {
        std::string name;
        std::vector<float> values;
    };
    std::vector<named_trace> main_layer0_ops;
};

// Uncached, reusable Llama forward runner for both LeLM towers.  It accepts
// arbitrary valid model_hparams through model::make_test_model(), while GGUF
// loading pins production input to v2-medium.
class lm final {
public:
    lm(const lm &) = delete;
    lm & operator=(const lm &) = delete;
    lm(lm &&) noexcept;
    lm & operator=(lm &&) noexcept;
    ~lm();

    static std::unique_ptr<lm> create(std::shared_ptr<const model> model);
    [[nodiscard]] lm_output forward(const lm_input & input) const;

private:
    explicit lm(std::shared_ptr<const model> model);
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace levo::detail
