#pragma once

#include "levo-model.h"
#include "levo-tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace levo::detail {

// A dense condition in GGML's [width, prefix] order.  `values` is laid out
// with the width (embedding) dimension contiguous, so column t starts at
// t * width.
struct condition_tensor {
    int32_t width = 0;
    int32_t prefix = 0;
    std::vector<float> values;

    [[nodiscard]] const float * data() const noexcept { return values.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return values.size(); }
};

struct condition_pair {
    condition_tensor main;
    condition_tensor detail;
};

// The first pair is the requested condition and the second is the CFG-null
// condition.  Both pairs have exactly 952 positions for v2-medium.
struct conditioning_result {
    condition_pair conditional;
    condition_pair null_condition;
};

class conditioner final {
public:
    conditioner(std::shared_ptr<const model> model,
                const ByteLevelBPETokenizer & tokenizer);

    [[nodiscard]] conditioning_result prepare(const std::string & lyrics,
                                              const std::string & description) const;

private:
    std::shared_ptr<const model> model_;
    const ByteLevelBPETokenizer * tokenizer_ = nullptr;
};

// Convenience entry point for callers that do not need to retain a
// conditioner object.
[[nodiscard]] conditioning_result prepare_conditioning(
    const model & model,
    const ByteLevelBPETokenizer & tokenizer,
    const std::string & lyrics,
    const std::string & description);

} // namespace levo::detail
