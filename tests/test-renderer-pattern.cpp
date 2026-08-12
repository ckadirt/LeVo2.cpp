#include "../src/levo-renderer-pattern.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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

} // namespace

int main() {
    using namespace levo;

    const std::vector<std::vector<std::int32_t>> tokens{
        {10, 11, 12},
        {20, 21, 22},
        {30, 31, 32},
    };
    const auto streams = validate_renderer_tokens(tokens);
    assert(streams.frames() == 3);
    assert((streams.vocal == std::vector<std::int32_t>{20, 21, 22}));
    assert((streams.bgm == std::vector<std::int32_t>{30, 31, 32}));

    expect_throw([] { validate_renderer_tokens({{1}, {2}}); });
    expect_throw([] { validate_renderer_tokens({{1, 2}, {3}, {4, 5}}); });
    expect_throw([] { validate_renderer_tokens({{}, {}, {}}); });
    expect_throw([] { validate_renderer_tokens({{1}, {-1}, {2}}); });
    expect_throw([] { validate_renderer_tokens({{1}, {renderer_codebook_size}, {2}}); });

    for (const std::size_t frames : {std::size_t(1), std::size_t(252),
                                     std::size_t(750), std::size_t(1000)}) {
        const auto schedule = make_renderer_schedule(frames);
        assert(schedule.source_frames == frames);
        assert(schedule.padded_frames == renderer_window_frames);
        assert(schedule.windows.size() == 1);
        assert(schedule.target_samples == frames * renderer_samples_per_frame);
        assert(schedule.padded_source_indices.size() == renderer_window_frames);
        if (frames < renderer_window_frames)
            assert(schedule.padded_source_indices[frames] == 0);
        assert(schedule.windows[0].input_offset_frames == 0);
        assert(schedule.windows[0].overlap_left_samples == 0);
        assert(schedule.windows[0].overlap_right_samples == 0);
    }

    const auto two_windows = make_renderer_schedule(1001);
    assert(two_windows.padded_frames == 1750);
    assert(two_windows.windows.size() == 2);
    assert(two_windows.windows[0].input_offset_frames == 0);
    assert(two_windows.windows[1].input_offset_frames == renderer_hop_frames);
    assert(two_windows.windows[1].output_offset_samples == renderer_hop_frames * renderer_samples_per_frame);
    assert(two_windows.windows[1].overlap_left_samples == renderer_overlap_frames * renderer_samples_per_frame);
    assert(two_windows.windows[1].overlap_right_samples == 0);
    assert(two_windows.padded_source_indices[1000] == 1000);
    assert(two_windows.padded_source_indices[1001] == 0);

    const auto three_windows = make_renderer_schedule(1751);
    assert(three_windows.padded_frames == 2500);
    assert(three_windows.windows.size() == 3);
    assert(three_windows.windows[2].input_offset_frames == 1500);

    expect_throw([] { make_renderer_schedule(0); });
    expect_throw([] { renderer_sample_count(std::numeric_limits<std::size_t>::max()); });
    expect_throw([] { renderer_sample_count(1, 0); });

    const auto euler = make_renderer_euler_schedule(3);
    assert(euler.times.size() == 4);
    assert(euler.times.front() == 0.0f);
    assert(euler.times.back() == 1.0f);
    assert(std::fabs(euler.dt - (1.0f / 3.0f)) < 1e-7f);
    assert(std::fabs(euler.times[2] - 2.0f / 3.0f) < 1e-6f);
    expect_throw([] { make_renderer_euler_schedule(0); });

    const auto cfg = combine_renderer_cfg({1.0f, 2.0f}, {3.0f, 6.0f}, 1.5f);
    assert(std::fabs(cfg[0] - 4.0f) < 1e-6f);
    assert(std::fabs(cfg[1] - 8.0f) < 1e-6f);
    expect_throw([] { combine_renderer_cfg({1.0f}, {1.0f, 2.0f}, 1.0f); });

    // Verify the official two-window linear crossfade and exact target crop.
    std::vector<renderer_stereo_audio> decoded;
    for (std::size_t index = 0; index < two_windows.windows.size(); ++index) {
        const std::size_t samples = two_windows.windows[index].decoded_samples;
        decoded.push_back({std::vector<float>(samples, index == 0 ? 1.0f : 3.0f),
                           std::vector<float>(samples, index == 0 ? 2.0f : 4.0f)});
    }
    const auto assembled = assemble_renderer_audio(two_windows, decoded);
    assert(assembled.samples() == 1001 * renderer_samples_per_frame);
    const std::size_t overlap = renderer_overlap_frames * renderer_samples_per_frame;
    const std::size_t boundary = renderer_hop_frames * renderer_samples_per_frame;
    assert(std::fabs(assembled.left[boundary] - 1.0f) < 1e-6f);
    assert(std::fabs(assembled.left[boundary + overlap - 1] - 3.0f) < 1e-6f);
    assert(std::fabs(assembled.left[boundary + overlap] - 3.0f) < 1e-6f);
    assert(std::fabs(assembled.right.front() - 2.0f) < 1e-6f);
    assert(std::fabs(assembled.right.back() - 4.0f) < 1e-6f);
    expect_throw([&] { assemble_renderer_audio(two_windows, {decoded[0]}); });

    std::cout << "renderer pattern ok\n";
}
