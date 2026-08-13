#include "levo-checkpoint.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace levo::checkpoint {
namespace {

constexpr std::size_t fixed_header_bytes = 72;
constexpr std::size_t section_entry_bytes = 24;
constexpr std::size_t digest_offset = 40;
constexpr std::size_t digest_bytes = 32;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo checkpoint: " + message);
}

bool checked_add(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t * out) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

bool checked_mul(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t * out) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) return false;
    *out = lhs * rhs;
    return true;
}

void put_u32(std::vector<std::uint8_t> & out, std::size_t offset, std::uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        out[offset++] = static_cast<std::uint8_t>(value >> shift);
    }
}

void put_u64(std::vector<std::uint8_t> & out, std::size_t offset, std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) {
        out[offset++] = static_cast<std::uint8_t>(value >> shift);
    }
}

std::uint32_t get_u32(const std::uint8_t * bytes, std::size_t offset) {
    std::uint32_t result = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) {
        result |= static_cast<std::uint32_t>(bytes[offset++]) << shift;
    }
    return result;
}

std::uint64_t get_u64(const std::uint8_t * bytes, std::size_t offset) {
    std::uint64_t result = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) {
        result |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
    }
    return result;
}

class sha256_state final {
public:
    sha256_state() : state_{{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}} {}

    void update(const std::uint8_t * data, std::size_t size) {
        bit_count_ += static_cast<std::uint64_t>(size) * 8U;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - buffered_);
            std::copy(data, data + take, block_.begin() + static_cast<std::ptrdiff_t>(buffered_));
            buffered_ += take;
            data += take;
            size -= take;
            if (buffered_ == block_.size()) {
                transform(block_.data());
                buffered_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t original_bits = bit_count_;
        const std::uint8_t one = 0x80;
        const std::uint8_t zero = 0;
        update(&one, 1);
        while (buffered_ != 56) update(&zero, 1);
        std::array<std::uint8_t, 8> length{};
        for (unsigned index = 0; index != length.size(); ++index) {
            length[length.size() - 1U - index] = static_cast<std::uint8_t>(original_bits >> (index * 8U));
        }
        update(length.data(), length.size());

        std::array<std::uint8_t, 32> result{};
        for (std::size_t word = 0; word != state_.size(); ++word) {
            for (unsigned byte = 0; byte != 4; ++byte) {
                result[word * 4U + byte] = static_cast<std::uint8_t>(state_[word] >> (24U - byte * 8U));
            }
        }
        return result;
    }

private:
    static constexpr std::array<std::uint32_t, 64> constants{{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U}};

    static std::uint32_t ror(std::uint32_t value, unsigned amount) {
        return (value >> amount) | (value << (32U - amount));
    }

    void transform(const std::uint8_t * bytes) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index != 16; ++index) {
            words[index] = (static_cast<std::uint32_t>(bytes[index * 4U]) << 24U) |
                           (static_cast<std::uint32_t>(bytes[index * 4U + 1U]) << 16U) |
                           (static_cast<std::uint32_t>(bytes[index * 4U + 2U]) << 8U) |
                           static_cast<std::uint32_t>(bytes[index * 4U + 3U]);
        }
        for (std::size_t index = 16; index != words.size(); ++index) {
            const auto s0 = ror(words[index - 15U], 7) ^ ror(words[index - 15U], 18) ^ (words[index - 15U] >> 3U);
            const auto s1 = ror(words[index - 2U], 17) ^ ror(words[index - 2U], 19) ^ (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
        }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (std::size_t index = 0; index != words.size(); ++index) {
            const auto s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto temp1 = h + s1 + choice + constants[index] + words[index];
            const auto s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> block_{};
    std::size_t buffered_ = 0;
    std::uint64_t bit_count_ = 0;
};

std::array<std::uint8_t, 32> digest_with_zeroed_field(const std::uint8_t * bytes, std::size_t size) {
    if (size < fixed_header_bytes) fail("blob is shorter than the fixed header");
    sha256_state hash;
    hash.update(bytes, digest_offset);
    const std::array<std::uint8_t, digest_bytes> zeros{};
    hash.update(zeros.data(), zeros.size());
    hash.update(bytes + digest_offset + digest_bytes, size - digest_offset - digest_bytes);
    return hash.finish();
}

} // namespace

std::array<std::uint8_t, 32> sha256(const std::uint8_t * bytes, std::size_t size) {
    if (size != 0 && bytes == nullptr) throw std::invalid_argument("cannot hash a null non-empty payload");
    sha256_state hash;
    if (size != 0) hash.update(bytes, size);
    return hash.finish();
}

std::string hex_digest(const std::array<std::uint8_t, 32> & digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

std::vector<std::uint8_t> encode(const std::array<char, 8> & magic, std::uint32_t stage,
                                 const std::vector<section> & sections) {
    if (stage == 0) throw std::invalid_argument("checkpoint stage must be nonzero");
    if (sections.size() > std::numeric_limits<std::uint32_t>::max()) fail("too many checkpoint sections");
    std::uint64_t directory_bytes = 0;
    if (!checked_mul(static_cast<std::uint64_t>(sections.size()), section_entry_bytes, &directory_bytes)) {
        fail("checkpoint section directory overflows");
    }
    std::uint64_t header_bytes = 0;
    if (!checked_add(fixed_header_bytes, directory_bytes, &header_bytes) || header_bytes > max_blob_bytes) {
        fail("checkpoint header exceeds size limit");
    }
    std::uint64_t total = header_bytes;
    for (const section & entry : sections) {
        if (entry.kind == static_cast<section_kind>(0)) fail("checkpoint section kind must be nonzero");
        if (!checked_add(total, entry.bytes.size(), &total) || total > max_blob_bytes) {
            fail("checkpoint exceeds 2 GiB size limit");
        }
    }
    if (total > std::numeric_limits<std::size_t>::max()) fail("checkpoint cannot fit in address space");

    std::vector<std::uint8_t> result(static_cast<std::size_t>(total), 0);
    std::copy(magic.begin(), magic.end(), reinterpret_cast<char *>(result.data()));
    put_u32(result, 8, format_version);
    put_u32(result, 12, stage);
    put_u32(result, 16, static_cast<std::uint32_t>(sections.size()));
    put_u32(result, 20, static_cast<std::uint32_t>(header_bytes));
    put_u64(result, 24, total);

    std::uint64_t payload_offset = header_bytes;
    for (std::size_t index = 0; index != sections.size(); ++index) {
        const section & entry = sections[index];
        const std::size_t directory_offset = fixed_header_bytes + index * section_entry_bytes;
        put_u32(result, directory_offset, static_cast<std::uint32_t>(entry.kind));
        put_u64(result, directory_offset + 8, payload_offset);
        put_u64(result, directory_offset + 16, entry.bytes.size());
        if (!entry.bytes.empty()) {
            std::copy(entry.bytes.begin(), entry.bytes.end(), result.begin() + static_cast<std::ptrdiff_t>(payload_offset));
        }
        payload_offset += entry.bytes.size();
    }
    const auto digest = digest_with_zeroed_field(result.data(), result.size());
    std::copy(digest.begin(), digest.end(), result.begin() + static_cast<std::ptrdiff_t>(digest_offset));
    return result;
}

decoded_blob decode(const std::uint8_t * bytes, std::size_t size, const std::array<char, 8> & expected_magic) {
    if (bytes == nullptr) throw std::invalid_argument("checkpoint bytes cannot be null");
    if (size < fixed_header_bytes) fail("blob is shorter than the fixed header");
    if (size > max_blob_bytes) fail("blob exceeds 2 GiB size limit");
    if (!std::equal(expected_magic.begin(), expected_magic.end(), reinterpret_cast<const char *>(bytes))) {
        fail("blob magic does not identify the expected stage");
    }
    const std::uint32_t version = get_u32(bytes, 8);
    if (version != format_version) fail("unsupported blob format version " + std::to_string(version));
    const std::uint32_t stage = get_u32(bytes, 12);
    if (stage == 0) fail("blob stage is zero");
    const std::uint32_t section_count = get_u32(bytes, 16);
    const std::uint32_t header_bytes = get_u32(bytes, 20);
    const std::uint64_t total_bytes = get_u64(bytes, 24);
    std::uint64_t directory_bytes = 0;
    std::uint64_t expected_header = 0;
    if (!checked_mul(section_count, section_entry_bytes, &directory_bytes) ||
        !checked_add(fixed_header_bytes, directory_bytes, &expected_header) ||
        expected_header > max_blob_bytes || header_bytes != expected_header) {
        fail("blob section directory has an invalid size");
    }
    if (total_bytes != size || total_bytes < header_bytes) fail("blob total size does not match supplied bytes");
    const auto calculated = digest_with_zeroed_field(bytes, size);
    if (!std::equal(calculated.begin(), calculated.end(), bytes + digest_offset)) fail("blob SHA-256 does not match");

    struct range { std::uint64_t begin; std::uint64_t end; };
    std::vector<range> ranges;
    ranges.reserve(section_count);
    decoded_blob result;
    std::copy_n(reinterpret_cast<const char *>(bytes), result.magic.size(), result.magic.begin());
    result.version = version;
    result.stage = stage;
    result.sections.reserve(section_count);
    for (std::size_t index = 0; index != section_count; ++index) {
        const std::size_t directory_offset = fixed_header_bytes + index * section_entry_bytes;
        const std::uint32_t raw_kind = get_u32(bytes, directory_offset);
        const std::uint64_t offset = get_u64(bytes, directory_offset + 8);
        const std::uint64_t count = get_u64(bytes, directory_offset + 16);
        std::uint64_t end = 0;
        if (raw_kind == 0 || offset < header_bytes || !checked_add(offset, count, &end) || end > total_bytes) {
            fail("blob section has an invalid range");
        }
        ranges.push_back({offset, end});
        section entry;
        entry.kind = static_cast<section_kind>(raw_kind);
        entry.bytes.assign(bytes + offset, bytes + end);
        result.sections.push_back(std::move(entry));
    }
    std::sort(ranges.begin(), ranges.end(), [](const range & lhs, const range & rhs) { return lhs.begin < rhs.begin; });
    for (std::size_t index = 1; index != ranges.size(); ++index) {
        if (ranges[index - 1U].end > ranges[index].begin) fail("blob sections overlap");
    }
    return result;
}

} // namespace levo::checkpoint
