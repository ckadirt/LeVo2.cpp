#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace levo {

enum class backend_kind {
    auto_select,
    cpu,
    cuda,
};

struct backend_info {
    std::string name;
    std::string description;
    backend_kind kind = backend_kind::cpu;
    std::size_t memory_free = 0;
    std::size_t memory_total = 0;
};

const char * version() noexcept;

// Enumerates backends registered in the linked GGML build.
std::vector<backend_info> available_backends();

// Runs a deterministic GGML vector addition and returns its four output values.
// This is the foundation smoke test and is retained as a backend diagnostic.
std::vector<float> backend_smoke(backend_kind kind = backend_kind::auto_select,
                                 int device_index = 0);

// Sampling controls for the autoregressive LeLM.  `use_sampling = false`
// selects deterministic greedy decoding; a caller may also use temperature
// zero for the same per-stream choice.
struct generation_sampling_config {
    bool use_sampling = true;
    float temperature = 0.9F;
    std::size_t top_k_mixed = 50;
    std::size_t top_k_detail = 1;
    std::size_t repetition_window = 50;
    float repetition_penalty = 1.1F;
    std::vector<int64_t> ignore_tokens;
};

// Inputs for the v0.1 token generator.  The GGUF is self-contained: lyrics
// and description are tokenized exclusively from the embedded Qwen2 assets;
// no tokenizer path or external Python dependency is accepted.
struct generation_config {
    std::filesystem::path model_path;
    std::string lyrics;
    std::string description;
    double duration_seconds = 0.0;
    backend_kind backend = backend_kind::auto_select;
    int device_index = 0;

    // The sampled path is reproducible when this is true.  In greedy mode the
    // seed is retained in the artifact only for provenance.
    bool seed_present = false;
    uint64_t seed = 0;
    float cfg_scale = 1.5F;
    generation_sampling_config sampling;

    // Optional provenance fields copied into the JSON token manifest.  They
    // are intentionally not used for model selection or validation.
    std::string model_revision;
    std::string model_sha256;
    std::string runtime_revision;
    std::string tokenizer_revision;
    std::string tokenizer_sha256;
};

struct generation_progress {
    // Delayed-pattern positions completed after the initial all-special
    // position. `total_steps` excludes that initial position.
    std::size_t completed_steps = 0;
    std::size_t total_steps = 0;
    std::size_t requested_frames = 0;
    std::array<bool, 3> ended{{false, false, false}};
};

using generation_progress_callback = std::function<void(const generation_progress &)>;

struct generation_result {
    // Canonical stream-major int32 tensor payload, C-order [mixed, vocal,
    // BGM] / [3, frame_count]. The earliest EOS is omitted.
    std::vector<int32_t> tokens;
    std::size_t frame_count = 0;
    std::size_t requested_frames = 0;
    std::size_t sequence_steps = 0;
    std::array<bool, 3> ended{{false, false, false}};
    std::string backend_name;
    std::string model_name;
    std::string model_revision;
    std::string model_sha256;
    std::string runtime_revision;
    std::string tokenizer_revision;
    std::string tokenizer_sha256;
};

// Loads a v2-medium LeVo GGUF, forms conditional and null CFG prefixes from
// its embedded tokenizer, and produces a canonical [3,T] audio-token tensor.
// Valid durations are 0 < seconds <= 270, with T = floor(seconds * 25).
generation_result generate_tokens(const generation_config & config,
                                  generation_progress_callback progress = {});

// Writes `tokens.npy` and its required `tokens.json` companion. The JSON
// captures the supplied generation configuration and is compatible with the
// bundled Python LeVo decoder bridge.
void write_generation_artifact(const std::filesystem::path & output_path,
                               const generation_result & result,
                               const generation_config & config);

} // namespace levo
