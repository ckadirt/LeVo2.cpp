#pragma once

#include "levo.h"
#include "levo-vae-model.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace levo::detail {

struct vae_decode_result {
    std::size_t frames = 0;
    std::size_t samples_per_channel = 0;
    // Channel-major F32 `[left, right]`, shape [2, samples_per_channel].
    std::vector<float> audio;
    // Optional channel-major stage tensors used by oracle diagnostics.
    std::vector<std::vector<float>> stage_outputs;
};

class vae_decoder final {
public:
    vae_decoder(const vae_decoder &) = delete;
    vae_decoder & operator=(const vae_decoder &) = delete;
    vae_decoder(vae_decoder &&) noexcept;
    vae_decoder & operator=(vae_decoder &&) noexcept;
    ~vae_decoder();

    static std::unique_ptr<vae_decoder> create(std::shared_ptr<const vae_model> model);

    // Input is channel-major F32 `[64,frames]`, matching NumPy C-order
    // `[1,64,frames]`. Production accepts the strict F32 correctness artifact;
    // F16 execution is introduced only after its independent parity gate.
    [[nodiscard]] vae_decode_result decode(const std::vector<float> & latent,
                                           std::size_t frames,
                                           bool capture_stages = false,
                                           const cancellation_callback & cancelled = {}) const;

private:
    explicit vae_decoder(std::shared_ptr<const vae_model> model);
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace levo::detail
