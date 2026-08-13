#include "levo-checkpoint.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::array<char, 8> magic{{'L', 'E', 'V', 'O', 'L', 'M', '0', '1'}};

void expect_throw(const std::vector<std::uint8_t> & blob) {
    bool rejected = false;
    try {
        (void) levo::checkpoint::decode(blob, magic);
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main() {
    try {
        using namespace levo::checkpoint;
        const std::vector<section> sections{
            {section_kind::request_json, {'{', '}', '\n'}},
            {section_kind::delayed_tokens_i32, {1, 0, 0, 0, 2, 0, 0, 0}},
        };
        const std::vector<std::uint8_t> blob = encode(magic, 2, sections);
        const decoded_blob decoded = decode(blob, magic);
        assert(decoded.version == format_version);
        assert(decoded.stage == 2);
        assert(decoded.sections.size() == sections.size());
        assert(decoded.sections[0].kind == section_kind::request_json);
        assert(decoded.sections[0].bytes == sections[0].bytes);
        assert(decoded.sections[1].bytes == sections[1].bytes);
        assert(hex_digest(sha256(blob.data(), blob.size())).size() == 64);
        constexpr std::array<std::uint8_t, 3> abc{{'a', 'b', 'c'}};
        assert(hex_digest(sha256(abc.data(), abc.size())) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

        auto corrupted = blob;
        corrupted.back() ^= 1U;
        expect_throw(corrupted);
        corrupted = blob;
        corrupted[0] ^= 1U;
        expect_throw(corrupted);
        corrupted = blob;
        corrupted.pop_back();
        expect_throw(corrupted);
        std::cout << "checkpoint ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
