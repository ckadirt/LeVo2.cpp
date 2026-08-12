#include "../src/levo-token-io.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool throws(const std::function<void()> & function) {
    try {
        function();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

std::string read_text(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

} // namespace

int main() {
    using namespace levo::token_io;
    const auto root = std::filesystem::temp_directory_path() / "levo2-token-io-test";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    std::filesystem::create_directories(root);
    const auto npy = root / "tokens.npy";

    const std::vector<int32_t> expected{1, 2, 3, 4, 5, 6, 16384, 16385, 0, 7, 8, 9};
    artifact_metadata metadata;
    metadata.model_name = "tiny-model";
    metadata.model_revision = "test-revision";
    metadata.model_sha256 = "abc";
    metadata.generator = "test-token-io";
    metadata.backend_name = "CPU";
    metadata.lyrics = "line\nwith \"quotes\"";
    metadata.description = "style";
    metadata.duration_seconds = 0.48;
    metadata.seed_present = true;
    metadata.seed = 1234;
    metadata.cfg_scale = 1.25F;

    write(npy, expected, metadata);
    assert(read_tokens_npy(npy) == expected);
    const artifact loaded = read(npy);
    assert(loaded.frame_count == 4);
    assert(loaded.tokens == expected);
    assert(metadata_path(npy) == root / "tokens.json");
    const std::string json = read_text(metadata_path(npy));
    assert(json.find("\"shape\": [3, 4]") != std::string::npos);
    assert(json.find("\"dtype\": \"int32\"") != std::string::npos);
    assert(json.find("lyrics_sha256") != std::string::npos);
    assert(json.find("\"backend\": \"CPU\"") != std::string::npos);
    assert(json.find("\"cfg_scale\": 1.25") != std::string::npos);
    assert(json.find("\\nwith \\\"quotes\\\"") != std::string::npos);
    assert(tensor_sha256(expected) == "7508f1b33c15dd460825c9203fe6178ca8949e298cb7f657f7371cbc2c73cd1b");

    assert(throws([&] { write_tokens_npy(root / "bad-dim.npy", {1, 2}); }));
    assert(throws([&] { write_tokens_npy(root / "bad-token.npy", {0, 1, -1}); }));
    assert(throws([&] { write_tokens_npy(root / "bad-token-2.npy", {0, 1, 16386}); }));
    artifact_metadata bad_metadata;
    bad_metadata.duration_seconds = -1.0;
    assert(throws([&] { write(npy, expected, bad_metadata); }));

    std::filesystem::remove_all(root, ignored);
    std::cout << "token io ok\n";
    return 0;
}
