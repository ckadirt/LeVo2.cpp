#pragma once

#include "levo-flow-model.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace levo::flow {

// All rank-two values below are stored row-major as [frame, feature].  Tokens
// and mask IDs have exactly one entry per frame.  Raw in-context latents are
// VAE-space values and are normalized by the checkpoint statistics here.
struct conditioning_input {
    std::vector<int32_t> vocal_codes;
    std::vector<int32_t> bgm_codes;
    std::vector<int32_t> mask_ids;              // optional: defaults to 2.
    std::vector<float> raw_incontext_latent;    // optional: defaults to zero.
    std::vector<float> initial_noise;           // optional passthrough, [T,64].
};

struct conditioning_output {
    int32_t frames = 0;
    std::vector<float> vocal_codebook_lookup;   // [T, 32]
    std::vector<float> bgm_codebook_lookup;     // [T, 32]
    std::vector<float> vocal_projected;         // [T, 1024], before null masking
    std::vector<float> bgm_projected;           // [T, 1024], before null masking
    std::vector<float> conditioning;            // [T, 2048], mask 0 replaced with null
    std::vector<float> mask_embedding;          // [T, 24]
    std::vector<float> normalized_incontext;    // [T, 64], retained only for mask 1
    std::vector<float> initial_noise;           // [T, 64], provided input or zero
    std::vector<float> positional_embedding;    // [T, 2200]
    std::vector<float> null_condition;          // [1024]
};

class conditioning final {
public:
    explicit conditioning(std::shared_ptr<const model> weights);
    ~conditioning();
    conditioning(const conditioning &) = delete;
    conditioning & operator=(const conditioning &) = delete;
    conditioning(conditioning &&) noexcept;
    conditioning & operator=(conditioning &&) noexcept;

    [[nodiscard]] conditioning_output prepare(const conditioning_input & input) const;
    // These are exposed for Euler/prompt callers that need the inverse Flow
    // normalizer while preserving the checkpoint's exact F32 statistics.
    [[nodiscard]] std::vector<float> normalize_latents(const std::vector<float> & raw, int32_t frames) const;
    [[nodiscard]] std::vector<float> denormalize_latents(const std::vector<float> & normalized, int32_t frames) const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace levo::flow
