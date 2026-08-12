#include "levo-flow-renderer.h"

#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename Function>
void expect_throw(Function && function) {
    bool thrown = false;
    try {
        function();
    } catch (const std::exception &) {
        thrown = true;
    }
    assert(thrown);
}

bool close(float lhs, float rhs) { return std::fabs(lhs - rhs) < 1.0e-6F; }

} // namespace

int main() {
    using namespace levo::flow;

    // This test is asset-free and exercises the exact CFM context path with a
    // deterministic synthetic velocity. The final context must be restored
    // after the last Euler step, just as the official Python solver does.
    euler_input input;
    input.frames = 3;
    input.latent_dim = 2;
    input.incontext_frames = 1;
    input.sigma_min = 0.1F;
    input.initial_noise = {1.0F, 2.0F, 10.0F, 20.0F, 30.0F, 40.0F};
    input.normalized_incontext = {4.0F, 8.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::vector<float> times;
    const std::vector<float> solved = solve_flow_euler(
        input, 2,
        [&times](const std::vector<float> & state, float time) {
            times.push_back(time);
            if (times.size() == 1) {
                assert(close(state[0], 1.0F));
                assert(close(state[1], 2.0F));
            } else {
                // (1 - 0.9 * 0.5) * noise + 0.5 * incontext,
                // followed by the first +0.5 Euler update.
                assert(close(state[0], 2.55F));
                assert(close(state[1], 5.10F));
            }
            return std::vector<float>(state.size(), 1.0F);
        });
    assert(times.size() == 2 && close(times[0], 0.0F) && close(times[1], 0.5F));
    assert(close(solved[0], 4.0F) && close(solved[1], 8.0F));
    assert(close(solved[2], 11.0F) && close(solved[3], 21.0F));
    assert(close(solved[4], 31.0F) && close(solved[5], 41.0F));
    expect_throw([&] { (void) solve_flow_euler(input, 0, [](const auto &, float) { return std::vector<float>{}; }); });

    // Two 4-frame windows with a 3-frame hop retain the first full window and
    // discard the one-frame continuation prefix, then crop to source length.
    std::vector<latent_window> windows{
        {.input_offset_frames = 0, .denormalized_latents = {0.0F, 1.0F, 2.0F, 3.0F}},
        {.input_offset_frames = 3, .denormalized_latents = {30.0F, 31.0F, 32.0F, 33.0F}},
    };
    const std::vector<float> assembled = assemble_flow_latent_windows(windows, 6, 4, 3, 1);
    assert((assembled == std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, 31.0F, 32.0F}));
    expect_throw([&] { (void) assemble_flow_latent_windows(windows, 8, 4, 3, 1); });
    windows[1].input_offset_frames = 2;
    expect_throw([&] { (void) assemble_flow_latent_windows(windows, 6, 4, 3, 1); });

    std::cout << "flow renderer ok\n";
    return 0;
}
