#include "levo-wav.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint16_t read_u16(const std::vector<uint8_t> & bytes, std::size_t offset) {
    return static_cast<uint16_t>(bytes.at(offset)) |
           static_cast<uint16_t>(bytes.at(offset + 1) << 8U);
}

uint32_t read_u32(const std::vector<uint8_t> & bytes, std::size_t offset) {
    return static_cast<uint32_t>(bytes.at(offset)) |
           (static_cast<uint32_t>(bytes.at(offset + 1)) << 8U) |
           (static_cast<uint32_t>(bytes.at(offset + 2)) << 16U) |
           (static_cast<uint32_t>(bytes.at(offset + 3)) << 24U);
}

template <typename Function>
void expect_invalid(Function && function, const char * label) {
    try {
        function();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error(std::string(label) + " did not reject invalid input");
}

} // namespace

int main() {
    try {
        const auto output = std::filesystem::temp_directory_path() /
                            "levo2-native-wav-test.wav";
        const std::vector<float> samples = {
            0.0F, 0.25F,
            -0.5F, 1.0F,
            0.75F, -1.25F,
        };
        levo::detail::write_wav_f32(output, samples);
        std::ifstream input(output, std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                         std::istreambuf_iterator<char>());
        std::filesystem::remove(output);
        if (bytes.size() != 44 + samples.size() * sizeof(float) ||
            std::string(bytes.begin(), bytes.begin() + 4) != "RIFF" ||
            std::string(bytes.begin() + 8, bytes.begin() + 12) != "WAVE" ||
            std::string(bytes.begin() + 12, bytes.begin() + 16) != "fmt " ||
            std::string(bytes.begin() + 36, bytes.begin() + 40) != "data") {
            throw std::runtime_error("WAV chunk layout is incorrect");
        }
        if (read_u32(bytes, 4) != bytes.size() - 8 ||
            read_u32(bytes, 16) != 16 || read_u16(bytes, 20) != 3 ||
            read_u16(bytes, 22) != 2 || read_u32(bytes, 24) != 48000 ||
            read_u32(bytes, 28) != 384000 || read_u16(bytes, 32) != 8 ||
            read_u16(bytes, 34) != 32 ||
            read_u32(bytes, 40) != samples.size() * sizeof(float)) {
            throw std::runtime_error("WAV metadata is incorrect");
        }
        for (std::size_t index = 0; index < samples.size(); ++index) {
            uint32_t expected = 0;
            std::memcpy(&expected, &samples[index], sizeof(expected));
            if (read_u32(bytes, 44 + index * sizeof(float)) != expected) {
                throw std::runtime_error("WAV F32 payload changed a sample");
            }
        }

        expect_invalid([&] { levo::detail::write_wav_f32(output, {}); }, "empty samples");
        expect_invalid([&] { levo::detail::write_wav_f32(output, {0.0F}, 48000, 2); },
                       "partial stereo frame");
        expect_invalid([&] {
            levo::detail::write_wav_f32(output,
                                        {std::numeric_limits<float>::quiet_NaN(), 0.0F});
        }, "non-finite samples");
        expect_invalid([&] {
            auto wrong_extension = output;
            wrong_extension.replace_extension(".raw");
            levo::detail::write_wav_f32(wrong_extension, samples);
        }, "wrong extension");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
