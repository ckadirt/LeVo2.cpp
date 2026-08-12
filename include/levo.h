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
    gpu,
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

// Native renderer controls. Both model paths must point to the strict F32
// correctness GGUF artifacts. `external_noise`, when supplied, is C-order
// F32 [window_count, 1000, 64] and is the exact Flow reproducibility boundary.
// Otherwise the renderer derives the same shape from `seed` using its stable
// native Gaussian generator.
struct render_config {
    std::filesystem::path tokens_path;
    std::filesystem::path flow_model_path;
    std::filesystem::path vae_model_path;
    backend_kind backend = backend_kind::auto_select;
    int device_index = 0;
    std::size_t euler_steps = 0;
    float cfg_scale = 0.0F;
    uint64_t seed = 0;
    std::vector<float> external_noise;
    // Retain every per-window Flow latent and decoded audio window in the
    // result. Parity tooling needs those intermediate boundaries, and capturing
    // them here keeps the tooling on the production path instead of a parallel
    // reimplementation. Nothing extra is allocated while this is false.
    bool capture_windows = false;
};

enum class render_stage {
    loading_tokens,
    loading_flow,
    generating_latents,
    releasing_flow,
    loading_vae,
    decoding_window,
    assembling_audio,
    complete,
};

struct render_progress {
    render_stage stage = render_stage::loading_tokens;
    std::size_t completed_windows = 0;
    std::size_t total_windows = 0;
};

using render_progress_callback = std::function<void(const render_progress &)>;

struct render_provenance {
    std::string backend_name;
    std::string flow_model_name;
    std::string flow_levo_revision;
    std::string flow_model_sha256;
    std::string flow_runtime_revision;
    std::string vae_model_name;
    std::string vae_checkpoint_sha256;
    std::string vae_config_sha256;
    std::string vae_levo_revision;
    std::string vae_runtime_revision;
    uint64_t seed = 0;
    bool external_noise = false;
};

// Per-window renderer intermediates, populated only when `capture_windows` is
// set. Latents are C-order [1000, 64] in the renderer's own frame-major layout;
// decoded audio is channel-major [2, 1000 * 1920] before crossfade and crop.
struct render_window_capture {
    std::size_t input_offset_frames = 0;
    std::vector<float> normalized_latents;
    std::vector<float> denormalized_latents;
    std::vector<float> decoded_left;
    std::vector<float> decoded_right;
};

struct render_result {
    // IEEE-F32-ready, frame-interleaved [left, right] samples at 48 kHz.
    std::vector<float> interleaved_stereo;
    std::size_t samples_per_channel = 0;
    std::size_t source_frames = 0;
    std::size_t rendered_windows = 0;
    uint32_t sample_rate = 48000;
    render_provenance provenance;
    std::vector<render_window_capture> windows;
};

// Read canonical [3,T] int32 `tokens.npy`, run Flow then VAE on the selected
// backend, and return the cropped stereo waveform. Flow weights are scoped and
// released before VAE weights are loaded, so one backend need not hold both
// checkpoints at once. F16 GGUFs are deliberately rejected in this parity
// correctness path.
render_result render_tokens_to_audio(const render_config & config,
                                     render_progress_callback progress = {});

// Write a render result as a 48 kHz two-channel IEEE-F32 RIFF/WAVE file.
void write_render_wav(const std::filesystem::path & output_path,
                      const render_result & result);

} // namespace levo
