#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace levo::detail {

// Exact decoder contract emitted by convert/convert_vae.py.  The tensors are
// stored in GGML dimension order (kernel, input/output channels) but retain
// the original PyTorch ConvTranspose raw layout after weight-norm folding.
struct vae_hparams {
    int32_t sample_rate = 48000;
    int32_t audio_channels = 2;
    int32_t latent_dim = 64;
    int32_t channels = 128;
    int32_t downsampling_ratio = 1920;
    std::array<int32_t, 5> c_mults{{1, 2, 4, 8, 16}};
    std::array<int32_t, 5> strides{{2, 4, 4, 6, 10}};
    int32_t stage_count = 5;
    int32_t residual_units_per_stage = 3;
    bool use_snake_beta = true;
    bool final_tanh = false;
    bool soft_clip = false;
    bool weight_norm_folded = true;

    [[nodiscard]] bool valid(std::string * reason = nullptr) const;
};

struct vae_snake_weights {
    ggml_tensor * alpha_log = nullptr;
    ggml_tensor * beta_log = nullptr;
};

struct vae_conv_weights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr; // nullptr only for vae.decoder.output.
};

struct vae_residual_weights {
    vae_snake_weights pre;
    vae_conv_weights conv_0;
    vae_snake_weights post;
    vae_conv_weights conv_1;
};

struct vae_upsample_weights {
    vae_snake_weights activation;
    vae_conv_weights conv;
};

struct vae_stage_weights {
    vae_upsample_weights upsample;
    std::array<vae_residual_weights, 3> residual;
};

struct vae_decoder_weights {
    vae_conv_weights input;
    std::array<vae_stage_weights, 5> stage;
    vae_snake_weights output_activation;
    ggml_tensor * output_weight = nullptr;
};

struct vae_provenance {
    std::string name;
    std::string runtime_repository;
    std::string runtime_revision;
    std::string levo_repository;
    std::string levo_revision;
    std::string ggml_repository;
    std::string ggml_revision;
    std::string checkpoint_sha256;
    std::string config_sha256;
};

struct vae_model_load_options {
    ggml_backend_t backend = nullptr; // borrowed; must outlive the model
    bool allow_f32 = true;
    bool allow_f16 = true;
    bool require_pinned_provenance = true;
};

struct vae_test_tensor_data {
    std::vector<int64_t> dimensions;
    std::vector<float> values;
};

// Owns a backend-resident, decoder-only Oobleck weight set.  It does not own
// the supplied backend.  The model uses the same no-allocation GGUF parse plus
// backend-buffer initialization pattern as the LeLM model loader.
class vae_model final {
public:
    vae_model(const vae_model &) = delete;
    vae_model & operator=(const vae_model &) = delete;
    vae_model(vae_model &&) noexcept;
    vae_model & operator=(vae_model &&) noexcept;
    ~vae_model();

    static std::shared_ptr<vae_model> load_gguf(const std::string & path,
                                                 const vae_model_load_options & options);

    // Test-only helper; unlike load_gguf it does not apply the release GGUF
    // schema.  It lets graph tests construct tiny F32 payloads without a
    // 322 MiB decoder artifact.
    static std::shared_ptr<vae_model> make_test_model(
        const vae_hparams & hparams,
        const std::unordered_map<std::string, vae_test_tensor_data> & tensors,
        ggml_backend_t backend);

    [[nodiscard]] const vae_hparams & hparams() const noexcept;
    [[nodiscard]] const vae_decoder_weights & decoder() const noexcept;
    [[nodiscard]] const vae_provenance & provenance() const noexcept;
    [[nodiscard]] ggml_backend_t backend() const noexcept;
    [[nodiscard]] ggml_tensor * tensor(const std::string & name) const;
    [[nodiscard]] bool contains(const std::string & name) const noexcept;
    [[nodiscard]] std::size_t tensor_count() const noexcept;

private:
    vae_model();
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace levo::detail
