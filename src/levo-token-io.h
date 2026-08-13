#pragma once

#include "levo-sampling.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace levo {
namespace token_io {

// The LeVo delayed-code representation is stream-major: mixed, vocal, BGM.
inline constexpr std::size_t stream_count = 3;
inline constexpr int32_t first_token_id = 0;
inline constexpr int32_t eos_token_id = 16384;
inline constexpr int32_t special_token_id = 16385;
inline constexpr std::size_t default_frame_rate = 25;
inline constexpr std::size_t default_sample_rate = 48000;

struct artifact_metadata {
    // Provenance. Empty values are allowed for locally generated fixtures, but
    // are still emitted so the manifest has a stable schema.
    std::string model_name;
    std::string model_revision;
    std::string model_sha256;
    std::string model_artifact_sha256;
    std::string generator;
    std::string generator_revision;
    std::string runtime_revision;
    std::string tokenizer_revision;
    std::string tokenizer_sha256;
    std::string backend_name;

    // User inputs. The complete UTF-8 strings are retained, together with
    // their deterministic SHA-256 values in the JSON manifest.
    std::string lyrics;
    std::string description;

    // Duration is recorded in seconds as supplied by the caller. frame_count
    // is always taken from the tensor and is written by the implementation.
    double duration_seconds = 0.0;
    std::size_t frame_rate = default_frame_rate;
    std::size_t sample_rate = default_sample_rate;

    int32_t eos_id = eos_token_id;
    int32_t special_id = special_token_id;
    std::array<int32_t, stream_count> delays{{0, 250, 250}};

    sampling_config sampling;
    float cfg_scale = 1.5F;
    bool seed_present = false;
    uint64_t seed = 0;

    double backend_seconds = 0.0;
    double model_load_seconds = 0.0;
    double conditioning_seconds = 0.0;
    double prefill_seconds = 0.0;
    double generation_seconds = 0.0;
    double total_seconds = 0.0;
};

struct artifact {
    // Exactly stream_count * frame_count values, in C order ([3,T]).
    std::vector<int32_t> tokens;
    std::size_t frame_count = 0;
    artifact_metadata metadata;
};

// Returns the canonical companion path (tokens.npy -> tokens.json).
std::filesystem::path metadata_path(const std::filesystem::path & npy_path);

// SHA-256 of the canonical little-endian int32 C-order tensor payload (not
// including the NumPy header).
std::string tensor_sha256(const std::vector<int32_t> & tokens);

// Streaming SHA-256 of a complete file. Shared by token and WAV provenance
// writers; it does not load the file into memory.
std::string file_sha256(const std::filesystem::path & path);

// Validate and write a NumPy 1.0 (or, for an unusually large header, 2.0)
// int32 C-order array with shape [3,T]. The JSON companion is written beside
// it. Both files are staged in the destination directory and renamed into
// place after their contents have been flushed.
void write(const std::filesystem::path & npy_path,
           const std::vector<int32_t> & tokens,
           const artifact_metadata & metadata = {});

// Explicit names are provided for callers that do not want to use the short
// write/read spellings above.
void write_tokens_npy(const std::filesystem::path & npy_path,
                      const std::vector<int32_t> & tokens);
std::vector<int32_t> read_tokens_npy(const std::filesystem::path & npy_path);

// Reads and strictly validates the NumPy tensor. Metadata is intentionally not
// parsed here: the JSON manifest is provenance, while the NPY is the portable
// data interchange. Use metadata_path(path) to locate the companion.
artifact read(const std::filesystem::path & npy_path);

inline void write_token_artifact(const std::filesystem::path & path,
                                 const std::vector<int32_t> & tokens,
                                 const artifact_metadata & metadata = {}) {
    write(path, tokens, metadata);
}
inline artifact read_token_artifact(const std::filesystem::path & path) {
    return read(path);
}

} // namespace token_io

// Convenience aliases follow the rest of the C++ API, which places public
// runtime helpers directly in levo. The implementation remains grouped under
// token_io to keep artifact-specific names together.
using token_artifact_metadata = token_io::artifact_metadata;
using token_artifact = token_io::artifact;
inline void write_token_artifact(const std::filesystem::path & path,
                                 const std::vector<int32_t> & tokens,
                                 const token_artifact_metadata & metadata = {}) {
    token_io::write(path, tokens, metadata);
}
inline token_artifact read_token_artifact(const std::filesystem::path & path) {
    return token_io::read(path);
}

} // namespace levo
