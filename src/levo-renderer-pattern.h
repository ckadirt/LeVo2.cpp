#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace levo {

// The official Flow/VAE renderer operates on 40 seconds internally:
// 1000 LeLM frames at 25 Hz, with 750-frame hops and 250-frame overlaps.
constexpr std::size_t renderer_window_frames = 1000;
constexpr std::size_t renderer_hop_frames = 750;
constexpr std::size_t renderer_overlap_frames = 250;
constexpr std::size_t renderer_samples_per_frame = 1920;
constexpr std::size_t renderer_sample_rate = 48000;
constexpr std::size_t renderer_frame_rate = 25;
constexpr std::size_t renderer_codebook_size = 16384;

struct renderer_token_streams {
    std::vector<std::int32_t> vocal;
    std::vector<std::int32_t> bgm;

    std::size_t frames() const noexcept { return vocal.size(); }
};

// Validate stream-major [3,T] LeLM tokens and select the two streams consumed
// by Flow1dVAESeparate. EOS/special tokens are deliberately rejected here.
renderer_token_streams validate_renderer_tokens(
        const std::vector<std::vector<std::int32_t>> & tokens,
        std::int32_t codebook_size = static_cast<std::int32_t>(renderer_codebook_size));

struct renderer_window {
    // Offset into the repeated/padded token tensor.
    std::size_t input_offset_frames = 0;
    std::size_t input_frames = renderer_window_frames;
    // Placement of the decoded window in the assembled latent/audio stream.
    std::size_t output_offset_samples = 0;
    std::size_t decoded_samples = 0;
    std::size_t overlap_left_samples = 0;
    std::size_t overlap_right_samples = 0;
};

struct renderer_schedule {
    std::size_t source_frames = 0;
    std::size_t padded_frames = 0;
    std::size_t target_samples = 0;
    // For every padded frame, the source token frame to repeat. This mirrors
    // the upstream repeat-and-crop behavior and avoids implicit modulo logic
    // in callers that prepare model inputs.
    std::vector<std::size_t> padded_source_indices;
    std::vector<renderer_window> windows;
};

renderer_schedule make_renderer_schedule(std::size_t source_frames);

std::size_t renderer_sample_count(
        std::size_t frames,
        std::size_t samples_per_frame = renderer_samples_per_frame);

struct renderer_euler_schedule {
    std::vector<float> times;
    float dt = 0.0f;
};

renderer_euler_schedule make_renderer_euler_schedule(std::size_t steps);

// Elementwise classifier-free guidance: unconditional + scale * (conditional
// - unconditional). The vectors must have equal non-empty length.
std::vector<float> combine_renderer_cfg(
        const std::vector<float> & unconditional,
        const std::vector<float> & conditional,
        float guidance_scale);

struct renderer_stereo_audio {
    std::vector<float> left;
    std::vector<float> right;

    std::size_t samples() const noexcept { return left.size(); }
};

// Assemble decoded [left,right] windows using the official linear crossfade,
// then crop to source_frames * 1920 samples. Every decoded window must contain
// exactly 1000 * 1920 samples per channel.
renderer_stereo_audio assemble_renderer_audio(
        const renderer_schedule & schedule,
        const std::vector<renderer_stereo_audio> & decoded_windows);

} // namespace levo
