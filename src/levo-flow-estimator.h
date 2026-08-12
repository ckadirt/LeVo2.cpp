#pragma once

#include "levo-flow-model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace levo::flow {

struct estimator_input {
    // C-order F32 [batch, frames, 2200]. The estimator adds learned WPE.
    std::vector<float> model_input;
    // F32 [batch], in the same [0, 1] coordinate system as PyTorch.
    std::vector<float> timesteps;
    // Optional additive attention mask, C-order [batch, frames, frames].
    // Empty means an all-zero (fully visible, non-causal) mask.
    std::vector<float> attention_mask;
    std::size_t batch = 0;
    std::size_t frames = 0;
};

struct estimator_capture {
    // All arrays use C-order [batch, frames, width] unless noted otherwise.
    std::vector<float> full_output;
    std::vector<float> timestep_embedding;  // [batch, 2200]
    std::vector<float> timestep_modulation; // [batch, 13200]
    std::vector<float> block0_input;
    std::vector<float> block0_output;
};

class estimator final {
public:
    estimator(const estimator &) = delete;
    estimator & operator=(const estimator &) = delete;
    estimator(estimator &&) noexcept;
    estimator & operator=(estimator &&) noexcept;
    ~estimator();

    static std::unique_ptr<estimator> create(std::shared_ptr<const model> model);

    // Evaluate exactly one official Flow estimator graph and return its final
    // latent-width slice in C order [batch, frames, 64]. The optional capture
    // retains the complete [batch, frames, 2200] projection. CFG, Euler
    // integration, and renderer windowing intentionally live elsewhere. The
    // initial implementation is strict F32 for weights and activations.
    [[nodiscard]] std::vector<float> velocity(
            const estimator_input & input,
            estimator_capture * capture = nullptr) const;

private:
    explicit estimator(std::shared_ptr<const model> model);
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace levo::flow
