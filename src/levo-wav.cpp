#include "levo-wav.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace levo::detail {
namespace {

void write_u16(std::ostream & output, uint16_t value) {
    const char bytes[2] = {
        static_cast<char>(value & 0xffU),
        static_cast<char>((value >> 8U) & 0xffU),
    };
    output.write(bytes, sizeof(bytes));
}

void write_u32(std::ostream & output, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xffU),
        static_cast<char>((value >> 8U) & 0xffU),
        static_cast<char>((value >> 16U) & 0xffU),
        static_cast<char>((value >> 24U) & 0xffU),
    };
    output.write(bytes, sizeof(bytes));
}

void write_f32(std::ostream & output, float value) {
    static_assert(sizeof(float) == sizeof(uint32_t), "WAV writer requires 32-bit float");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(output, bits);
}

} // namespace

void write_wav_f32(const std::filesystem::path & path,
                   const std::vector<float> & interleaved_samples,
                   uint32_t sample_rate,
                   uint16_t channels) {
    if (path.filename().empty() || path.extension() != ".wav") {
        throw std::invalid_argument("native renderer output path must use the .wav extension");
    }
    if (channels == 0 || sample_rate == 0) {
        throw std::invalid_argument("WAV sample rate and channel count must be positive");
    }
    if (interleaved_samples.empty() || interleaved_samples.size() % channels != 0) {
        throw std::invalid_argument("WAV samples must contain complete non-empty interleaved frames");
    }
    for (const float value : interleaved_samples) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("WAV samples must all be finite");
        }
    }

    constexpr uint32_t format_chunk_bytes = 16;
    constexpr uint16_t ieee_float_format = 3;
    constexpr uint16_t bits_per_sample = 32;
    constexpr uint32_t header_after_riff_bytes = 36;
    const uint64_t data_bytes_u64 = interleaved_samples.size() * sizeof(float);
    if (data_bytes_u64 > std::numeric_limits<uint32_t>::max() - header_after_riff_bytes) {
        throw std::length_error("waveform is too large for a standard RIFF/WAVE file");
    }
    const uint32_t data_bytes = static_cast<uint32_t>(data_bytes_u64);
    const uint32_t block_align_u32 = static_cast<uint32_t>(channels) * sizeof(float);
    const uint64_t byte_rate_u64 = static_cast<uint64_t>(sample_rate) * block_align_u32;
    if (block_align_u32 > std::numeric_limits<uint16_t>::max() ||
        byte_rate_u64 > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("WAV sample rate/channel combination is not representable");
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open WAV output: " + path.string());
    }
    output.write("RIFF", 4);
    write_u32(output, header_after_riff_bytes + data_bytes);
    output.write("WAVE", 4);
    output.write("fmt ", 4);
    write_u32(output, format_chunk_bytes);
    write_u16(output, ieee_float_format);
    write_u16(output, channels);
    write_u32(output, sample_rate);
    write_u32(output, static_cast<uint32_t>(byte_rate_u64));
    write_u16(output, static_cast<uint16_t>(block_align_u32));
    write_u16(output, bits_per_sample);
    output.write("data", 4);
    write_u32(output, data_bytes);
    for (const float value : interleaved_samples) {
        write_f32(output, value);
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("failed while writing WAV output: " + path.string());
    }
}

} // namespace levo::detail
