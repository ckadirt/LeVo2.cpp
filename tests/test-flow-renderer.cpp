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
    std::vector<std::size_t> completed_steps;
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
        },
        [&completed_steps](std::size_t completed, std::size_t total) {
            assert(total == 2);
            completed_steps.push_back(completed);
        });
    assert(times.size() == 2 && close(times[0], 0.0F) && close(times[1], 0.5F));
    assert((completed_steps == std::vector<std::size_t>{1, 2}));
    assert(close(solved[0], 4.0F) && close(solved[1], 8.0F));
    assert(close(solved[2], 11.0F) && close(solved[3], 21.0F));
    assert(close(solved[4], 31.0F) && close(solved[5], 41.0F));

    // Every completed Euler step is a durable restart boundary.  Pausing at
    // each boundary and rebuilding from the saved state must be bit-exact
    // with the uninterrupted solve.
    for (std::size_t pause_at = 0; pause_at < 2; ++pause_at) {
        euler_resume_state checkpoint;
        std::size_t polls = 0;
        bool paused = false;
        try {
            (void) solve_flow_euler(
                input, 2,
                [](const std::vector<float> & state, float) {
                    return std::vector<float>(state.size(), 1.0F);
                }, {},
                [&polls, pause_at] { return polls++ >= pause_at; },
                nullptr, &checkpoint);
        } catch (const levo::operation_cancelled &) {
            paused = true;
        }
        assert(paused && checkpoint.completed_steps == pause_at);
        const std::vector<float> resumed = solve_flow_euler(
            input, 2,
            [](const std::vector<float> & state, float) {
                return std::vector<float>(state.size(), 1.0F);
            }, {}, {}, &checkpoint);
        assert(resumed == solved);
        assert(checkpoint.completed_steps == 2 && checkpoint.normalized_state == solved);
    }

    euler_resume_state invalid{3, std::vector<float>(input.initial_noise.size(), 0.0F)};
    expect_throw([&] {
        (void) solve_flow_euler(input, 2,
                                [](const auto & state, float) {
                                    return std::vector<float>(state.size(), 1.0F);
                                }, {}, {}, &invalid);
    });
    invalid = euler_resume_state{1, {}};
    expect_throw([&] {
        (void) solve_flow_euler(input, 2,
                                [](const auto & state, float) {
                                    return std::vector<float>(state.size(), 1.0F);
                                }, {}, {}, &invalid);
    });
    expect_throw([&] { (void) solve_flow_euler(input, 0, [](const auto &, float) { return std::vector<float>{}; }); });

    // Cancellation is checked before every expensive velocity evaluation and
    // preserves the completed-step boundary already reported to the caller.
    std::size_t cancel_polls = 0;
    std::size_t velocity_calls = 0;
    bool cancelled = false;
    try {
        (void) solve_flow_euler(
            input, 2,
            [&velocity_calls](const std::vector<float> & state, float) {
                ++velocity_calls;
                return std::vector<float>(state.size(), 1.0F);
            }, {},
            [&cancel_polls] { return cancel_polls++ != 0; });
    } catch (const levo::operation_cancelled &) {
        cancelled = true;
    }
    assert(cancelled && velocity_calls == 1 && cancel_polls == 2);

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
