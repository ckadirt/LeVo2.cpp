#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace levo::detail {

// Writes an interleaved IEEE-float RIFF/WAVE file. The native renderer keeps
// waveform values in F32 and deliberately does not clamp or quantize them.
void write_wav_f32(const std::filesystem::path & path,
                   const std::vector<float> & interleaved_samples,
                   uint32_t sample_rate = 48000,
                   uint16_t channels = 2);

} // namespace levo::detail
