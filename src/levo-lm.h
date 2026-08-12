#pragma once

#include "levo-model.h"

#include <cstdint>
#include <memory>
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
