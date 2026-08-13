#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace levo::checkpoint {

// The node rejects blobs above this limit. Keep the engine's parser at the
// same boundary so hostile or truncated files cannot allocate arbitrarily.
inline constexpr std::uint64_t max_blob_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::uint32_t format_version = 1;

enum class section_kind : std::uint32_t {
    request_json = 1,
    delayed_tokens_i32 = 2,
    flow_noise_f32 = 3,
    completed_latents_f32 = 4,
    euler_state_f32 = 5,
    metadata = 6,
};

struct section {
    section_kind kind{};
    std::vector<std::uint8_t> bytes;
};

struct decoded_blob {
    std::array<char, 8> magic{};
    std::uint32_t version = 0;
    std::uint32_t stage = 0;
    std::vector<section> sections;
};

// Encode a private, self-identifying resume blob. Integers are explicitly
// little endian and the SHA-256 digest covers all bytes with its own 32-byte
// field zeroed. No native struct layout is serialized.
std::vector<std::uint8_t> encode(const std::array<char, 8> & magic,
                                 std::uint32_t stage,
                                 const std::vector<section> & sections);

// Strictly decode a blob. `expected_magic` is checked before section payloads
// are trusted. The caller supplies stage-specific shape and semantic checks
// after this common envelope validation.
decoded_blob decode(const std::uint8_t * bytes, std::size_t size,
                    const std::array<char, 8> & expected_magic);

inline decoded_blob decode(const std::vector<std::uint8_t> & bytes,
                           const std::array<char, 8> & expected_magic) {
    return decode(bytes.data(), bytes.size(), expected_magic);
}

std::array<std::uint8_t, 32> sha256(const std::uint8_t * bytes, std::size_t size);
std::string hex_digest(const std::array<std::uint8_t, 32> & digest);

} // namespace levo::checkpoint
