#pragma once

#include "levo.h"
#include "levo-flow-conditioning.h"
#include "levo-flow-estimator.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace levo::flow {

// Euler state is normalized Flow latent space and is always C-order
// [frames, latent_dim].  The prefix [0, incontext_frames) follows the exact
// CFM interpolation used by the official renderer.
struct euler_input {
    std::vector<float> initial_noise;
    std::vector<float> normalized_incontext;
    std::size_t frames = 0;
    std::size_t latent_dim = 0;
    std::size_t incontext_frames = 0;
    float sigma_min = 1.0e-4F;
};

using velocity_callback = std::function<std::vector<float>(
        const std::vector<float> & state, float timestep)>;
using euler_progress_callback = std::function<void(
        std::size_t completed_steps, std::size_t total_steps)>;

// Fixed-step Euler with the official in-context interpolation and final hard
// context restore.  `velocity` receives the state after interpolation at each
// t in {0, 1/steps, ..., (steps-1)/steps}.
[[nodiscard]] std::vector<float> solve_flow_euler(
        const euler_input & input,
        std::size_t steps,
        const velocity_callback & velocity,
        const euler_progress_callback & progress = {},
        const cancellation_callback & cancelled = {});

// One denormalized [window_frames, latent_dim] Flow result. The input frame
// offset identifies the corresponding repeated/padded token window. The
// normalized Euler solution is retained alongside it because the official
// oracle exports both, and the continuation context is defined in normalized
// space.
struct latent_window {
    std::size_t input_offset_frames = 0;
    std::vector<float> normalized_latents;
    std::vector<float> denormalized_latents;
};

// Concatenate Flow windows by retaining the first complete window and dropping
// `window_frames - hop_frames` continuation frames from every later window,
// then crop to source_frames. This is a latent convenience view; VAE/audio
// consumers should retain `latent_window`s so they can perform their native
// overlap handling.
[[nodiscard]] std::vector<float> assemble_flow_latent_windows(
        const std::vector<latent_window> & windows,
        std::size_t source_frames,
        std::size_t window_frames,
        std::size_t hop_frames,
        std::size_t latent_dim);

struct render_options {
    // Zero selects the model metadata default.  The official checkpoint uses
    // 50 steps and cfg=1.5.
    std::size_t euler_steps = 0;
    float guidance_scale = 0.0F;
    // Window is one-based. A zero-step event is emitted when the window starts,
    // followed by one event after each completed Euler update.
    std::function<void(std::size_t window, std::size_t total_windows,
                       std::size_t completed_steps, std::size_t total_steps)> progress;
    cancellation_callback cancelled;
};

struct render_input {
    // Source-length audio token streams. The renderer repeats the complete
    // source sequence to fill each 1000-frame Flow window, matching upstream.
    std::vector<int32_t> vocal_codes;
    std::vector<int32_t> bgm_codes;

    // Required deterministic normalized-F32 Euler noise, window-major:
    // [scheduled_windows, window_frames, latent_dim]. This explicit payload is
    // the reproducibility boundary; the renderer never draws random numbers.
    std::vector<float> initial_noise;

    // Optional raw (VAE-space) prefix for a prompt/context in the first
    // window. Later windows automatically continue from the preceding raw
    // Flow output overlap. Must have exactly initial_incontext_frames * 64
    // values and cannot exceed a window.
    std::vector<float> raw_initial_incontext;
    std::size_t initial_incontext_frames = 0;
};

struct render_output {
    std::size_t source_frames = 0;
    std::size_t padded_frames = 0;
    std::vector<latent_window> windows;
    // Denormalized C-order [source_frames, latent_dim], assembled without
    // latent-space crossfading and cropped to the original token duration.
    std::vector<float> denormalized_latents;
};

// Native Flow scheduler/ODE boundary. Estimator math remains isolated in
// `estimator`; this layer owns only conditioning assembly, CFG, Euler updates,
// explicit noise, and 1000/750/250 window continuation.
class renderer final {
public:
    explicit renderer(std::shared_ptr<const model> weights);
    renderer(const renderer &) = delete;
    renderer & operator=(const renderer &) = delete;
    renderer(renderer &&) noexcept;
    renderer & operator=(renderer &&) noexcept;
    ~renderer();

    [[nodiscard]] render_output render(
            const render_input & input,
            const render_options & options = {}) const;

private:
    std::shared_ptr<const model> weights_;
    conditioning conditioning_;
    std::unique_ptr<estimator> estimator_;
};

} // namespace levo::flow
