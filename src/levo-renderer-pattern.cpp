#include "levo-renderer-pattern.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace levo {

namespace {

std::size_t checked_add(std::size_t a, std::size_t b, const char * what) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error(what);
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char * what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error(what);
    return a * b;
}

void require_finite_audio(const std::vector<float> & values, const char * name) {
    for (const float value : values)
        if (!std::isfinite(value)) throw std::invalid_argument(std::string(name) + " contains non-finite samples");
}

} // namespace

renderer_token_streams validate_renderer_tokens(
        const std::vector<std::vector<std::int32_t>> & tokens,
        std::int32_t codebook_size) {
    if (tokens.size() != 3)
        throw std::invalid_argument("renderer tokens must have shape [3,T]");
    if (tokens[0].empty())
        throw std::invalid_argument("renderer token sequence must not be empty");
    const std::size_t frames = tokens[0].size();
    for (std::size_t stream = 0; stream < tokens.size(); ++stream) {
        if (tokens[stream].size() != frames)
            throw std::invalid_argument("renderer token streams have different lengths");
        for (const std::int32_t token : tokens[stream]) {
            if (token < 0 || token >= codebook_size)
                throw std::invalid_argument("renderer token ID is outside the codebook range");
        }
    }
    return {tokens[1], tokens[2]};
}

std::size_t renderer_sample_count(std::size_t frames, std::size_t samples_per_frame) {
    if (samples_per_frame == 0)
        throw std::invalid_argument("samples per frame must be positive");
    return checked_mul(frames, samples_per_frame, "renderer sample count overflows size_t");
}

renderer_schedule make_renderer_schedule(std::size_t source_frames) {
    if (source_frames == 0)
        throw std::invalid_argument("renderer source frame count must be positive");

    renderer_schedule schedule;
    schedule.source_frames = source_frames;
    schedule.target_samples = renderer_sample_count(source_frames);

    if (source_frames <= renderer_window_frames) {
        schedule.padded_frames = renderer_window_frames;
    } else {
        // Upstream:
        // ceil((codes_len - overlap) / hop) * hop + overlap.
        const std::size_t base = source_frames - renderer_overlap_frames;
        std::size_t groups = base / renderer_hop_frames;
        if (base % renderer_hop_frames != 0) ++groups;
        if (groups > (std::numeric_limits<std::size_t>::max() - renderer_overlap_frames) /
                    renderer_hop_frames)
            throw std::overflow_error("renderer padded frame count overflows size_t");
        schedule.padded_frames = checked_add(
                checked_mul(groups, renderer_hop_frames, "renderer padded frame count overflows size_t"),
                renderer_overlap_frames,
                "renderer padded frame count overflows size_t");
    }

    // The Python implementation repeats the whole code tensor whenever it
    // needs more frames, then crops it. This index map makes that behavior
    // explicit for native callers.
    schedule.padded_source_indices.resize(schedule.padded_frames);
    for (std::size_t index = 0; index < schedule.padded_frames; ++index)
        schedule.padded_source_indices[index] = index % source_frames;

    const std::size_t decoded_samples = renderer_sample_count(renderer_window_frames);
    const std::size_t overlap_samples = renderer_sample_count(renderer_overlap_frames);
    for (std::size_t offset = 0; offset <= schedule.padded_frames - renderer_window_frames;
         offset += renderer_hop_frames) {
        renderer_window window;
        window.input_offset_frames = offset;
        window.output_offset_samples = renderer_sample_count(offset);
        window.decoded_samples = decoded_samples;
        window.overlap_left_samples = offset == 0 ? 0 : overlap_samples;
        window.overlap_right_samples =
                offset < schedule.padded_frames - renderer_window_frames ? overlap_samples : 0;
        schedule.windows.push_back(window);
        if (offset > std::numeric_limits<std::size_t>::max() - renderer_hop_frames)
            break;
    }
    if (schedule.windows.empty())
        throw std::logic_error("renderer schedule produced no windows");
    return schedule;
}

renderer_euler_schedule make_renderer_euler_schedule(std::size_t steps) {
    if (steps == 0)
        throw std::invalid_argument("Euler step count must be positive");
    renderer_euler_schedule schedule;
    schedule.dt = 1.0f / static_cast<float>(steps);
    schedule.times.resize(steps + 1);
    for (std::size_t index = 0; index <= steps; ++index)
        schedule.times[index] = static_cast<float>(index) * schedule.dt;
    // torch.linspace(0, 1, steps + 1) guarantees exact endpoints.
    schedule.times.front() = 0.0f;
    schedule.times.back() = 1.0f;
    return schedule;
}

std::vector<float> combine_renderer_cfg(
        const std::vector<float> & unconditional,
        const std::vector<float> & conditional,
        float guidance_scale) {
    if (unconditional.empty() || unconditional.size() != conditional.size())
        throw std::invalid_argument("CFG vectors must have equal non-empty lengths");
    if (!std::isfinite(guidance_scale))
        throw std::invalid_argument("CFG guidance scale must be finite");
    std::vector<float> result(unconditional.size());
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = unconditional[index] + guidance_scale * (conditional[index] - unconditional[index]);
    return result;
}

renderer_stereo_audio assemble_renderer_audio(
        const renderer_schedule & schedule,
        const std::vector<renderer_stereo_audio> & decoded_windows) {
    if (schedule.source_frames == 0 || schedule.padded_frames == 0 || schedule.windows.empty())
        throw std::invalid_argument("renderer schedule is empty");
    if (decoded_windows.size() != schedule.windows.size())
        throw std::invalid_argument("decoded window count does not match renderer schedule");

    const std::size_t assembled_samples = renderer_sample_count(schedule.padded_frames);
    const std::size_t overlap_samples = renderer_sample_count(renderer_overlap_frames);
    renderer_stereo_audio output;
    output.left.assign(assembled_samples, 0.0f);
    output.right.assign(assembled_samples, 0.0f);

    for (std::size_t window_index = 0; window_index < schedule.windows.size(); ++window_index) {
        const renderer_window & descriptor = schedule.windows[window_index];
        const renderer_stereo_audio & decoded = decoded_windows[window_index];
        if (decoded.left.size() != descriptor.decoded_samples ||
            decoded.right.size() != descriptor.decoded_samples)
            throw std::invalid_argument("decoded window has an unexpected sample count");
        if (decoded.left.size() != decoded.right.size())
            throw std::invalid_argument("decoded stereo channels have different lengths");
        require_finite_audio(decoded.left, "decoded left channel");
        require_finite_audio(decoded.right, "decoded right channel");

        const std::size_t offset = descriptor.output_offset_samples;
        const std::size_t end = checked_add(offset, descriptor.decoded_samples,
                                            "renderer window end overflows size_t");
        if (end > assembled_samples)
            throw std::invalid_argument("renderer window exceeds assembled sample buffer");
        if (window_index == 0) {
            std::copy(decoded.left.begin(), decoded.left.end(), output.left.begin() + offset);
            std::copy(decoded.right.begin(), decoded.right.end(), output.right.begin() + offset);
            continue;
        }

        if (descriptor.overlap_left_samples != overlap_samples ||
            descriptor.overlap_left_samples >= descriptor.decoded_samples)
            throw std::invalid_argument("renderer overlap descriptor is invalid");
        const std::size_t overlap_end = checked_add(offset, overlap_samples,
                                                    "renderer overlap end overflows size_t");
        if (overlap_end > assembled_samples)
            throw std::invalid_argument("renderer overlap exceeds assembled sample buffer");

        // numpy.linspace(0, 1, overlap_samples) with its default endpoint is
        // the upstream crossfade. Previous audio receives (1-alpha), current
        // audio receives alpha.
        const float denominator = static_cast<float>(overlap_samples - 1);
        for (std::size_t sample = 0; sample < overlap_samples; ++sample) {
            const float alpha = static_cast<float>(sample) / denominator;
            output.left[offset + sample] =
                    output.left[offset + sample] * (1.0f - alpha) + decoded.left[sample] * alpha;
            output.right[offset + sample] =
                    output.right[offset + sample] * (1.0f - alpha) + decoded.right[sample] * alpha;
        }
        std::copy(decoded.left.begin() + overlap_samples, decoded.left.end(),
                  output.left.begin() + overlap_end);
        std::copy(decoded.right.begin() + overlap_samples, decoded.right.end(),
                  output.right.begin() + overlap_end);
    }

    if (schedule.target_samples > output.left.size())
        throw std::logic_error("renderer target crop exceeds assembled output");
    output.left.resize(schedule.target_samples);
    output.right.resize(schedule.target_samples);
    return output;
}

} // namespace levo
