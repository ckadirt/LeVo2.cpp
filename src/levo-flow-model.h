#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace levo::flow {

struct flow_hparams {
    int32_t hidden_size = 2200;
    int32_t n_layer = 16;
    int32_t n_head = 20;
    int32_t head_dim = 110;
    int32_t intermediate_size = 4400;
    int32_t max_frames = 1000;
    int32_t codebook_size = 16384;
    int32_t codebook_dim = 32;
    int32_t condition_dim = 1024;
    int32_t latent_dim = 64;
    int32_t mask_dim = 24;
    int32_t time_embedding_dim = 512;
    int32_t euler_steps_default = 50;
    int32_t sample_rate = 48000;
    int32_t frame_rate = 25;
    int32_t window_frames = 1000;
    int32_t hop_frames = 750;
    int32_t overlap_frames = 250;
    float cfg_default = 1.5F;
    float rope_theta = 10000.0F;
    float time_embedding_scale = 1000.0F;
    float sigma_min = 1.0e-4F;
    bool rvq_weight_norm_folded = true;

    [[nodiscard]] bool valid(std::string * reason = nullptr) const;
};

struct flow_provenance {
    std::string name;
    std::string runtime_repository;
    std::string runtime_revision;
    std::string levo_repository;
    std::string levo_revision;
    std::string ggml_repository;
    std::string ggml_revision;
    std::string model_sha256;
    std::string parameter_dtype;
};

struct tensor_data {
    std::vector<int64_t> dimensions; // GGML order.
    std::vector<float> values;       // F32 test helper payload.
};

struct load_options {
    ggml_backend_t backend = nullptr; // Borrowed; must outlive this model.
    bool require_pinned_runtime = true;
    bool allow_f32 = true;
    bool allow_f16 = true;
};

// The Flow model owns only backend-resident tensors.  It has no graph or
// sampling policy; that is intentionally implemented in a later layer.
class model final {
public:
    model(const model &) = delete;
    model & operator=(const model &) = delete;
    model(model &&) noexcept;
    model & operator=(model &&) noexcept;
    ~model();

    static std::shared_ptr<model> load_gguf(const std::string & path,
                                             const load_options & options);

    // Test-only constructor. It supports tiny tensors but never changes the
    // strict production GGUF contract enforced by load_gguf().
    static std::shared_ptr<model> make_test_model(
        const flow_hparams & hparams,
        const std::unordered_map<std::string, tensor_data> & tensors,
        ggml_backend_t backend);

    [[nodiscard]] const flow_hparams & hparams() const noexcept;
    [[nodiscard]] const flow_provenance & provenance() const noexcept;
    [[nodiscard]] ggml_backend_t backend() const noexcept;
    [[nodiscard]] ggml_tensor * tensor(const std::string & name) const;
    [[nodiscard]] bool contains(const std::string & name) const noexcept;
    [[nodiscard]] std::size_t tensor_count() const noexcept;

private:
    model();
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace levo::flow
