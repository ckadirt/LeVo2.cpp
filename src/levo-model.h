#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// This header intentionally contains the runtime-facing GGUF contract, rather
// than the public application API.  The public API is introduced after the
// tokenizer and generation layers are ready.
namespace levo::detail {

namespace gguf_keys {
inline constexpr const char architecture[] = "general.architecture";
inline constexpr const char name[] = "general.name";
inline constexpr const char schema_version[] = "levo2.schema_version";
inline constexpr const char file_type[] = "general.file_type";
inline constexpr const char source_repository[] = "general.source.repo_url";
inline constexpr const char converter[] = "levo2.converter";
inline constexpr const char source_model_sha256[] = "levo2.source.model_sha256";
inline constexpr const char source_config_sha256[] = "levo2.source.config_sha256";
inline constexpr const char source_model_repository[] = "levo2.source.model_repository";
inline constexpr const char source_model_revision[] = "levo2.source.model_revision";
inline constexpr const char source_runtime_repository[] = "levo2.source.runtime_repository";
inline constexpr const char source_runtime_revision[] = "levo2.source.runtime_revision";
inline constexpr const char main_layers[] = "levo2.main.block_count";
inline constexpr const char detail_layers[] = "levo2.detail.block_count";
inline constexpr const char embedding_length[] = "levo2.embedding_length";
inline constexpr const char feed_forward_length[] = "levo2.feed_forward_length";
inline constexpr const char attention_heads[] = "levo2.attention.head_count";
inline constexpr const char kv_attention_heads[] = "levo2.attention.kv_head_count";
inline constexpr const char context_length[] = "levo2.context_length";
inline constexpr const char rms_norm_epsilon[] = "levo2.rms_norm_epsilon";
inline constexpr const char main_rope_theta[] = "levo2.main.rope_theta";
inline constexpr const char detail_rope_theta[] = "levo2.detail.rope_theta";
inline constexpr const char codebook_count[] = "levo2.codebook.count";
inline constexpr const char codebook_size[] = "levo2.codebook.size";
inline constexpr const char eos_token_id[] = "levo2.token.eos_id";
inline constexpr const char special_token_id[] = "levo2.token.special_id";
inline constexpr const char frame_rate[] = "levo2.audio.frame_rate";
inline constexpr const char sample_rate[] = "levo2.audio.sample_rate";
inline constexpr const char delays[] = "levo2.pattern.delays";
inline constexpr const char lyrics_prefix_length[] = "levo2.condition.lyrics_prefix_length";
inline constexpr const char prompt_prefix_length[] = "levo2.condition.prompt_prefix_length";
inline constexpr const char style_prefix_length[] = "levo2.condition.style_prefix_length";
inline constexpr const char tokenizer_model[] = "tokenizer.ggml.model";
inline constexpr const char tokenizer_tokens[] = "tokenizer.ggml.tokens";
inline constexpr const char tokenizer_merges[] = "tokenizer.ggml.merges";
inline constexpr const char tokenizer_added_json[] = "levo2.tokenizer.added_tokens.json";
inline constexpr const char tokenizer_config_json[] = "levo2.tokenizer.config.json";
inline constexpr const char tokenizer_revision[] = "levo2.tokenizer.revision";
inline constexpr const char tokenizer_sha256[] = "levo2.tokenizer.sha256";
} // namespace gguf_keys

namespace tensor_names {
inline constexpr const char mixed_embedding[] = "token_embd.mixed";
inline constexpr const char mixed_output[] = "output.mixed";
inline constexpr const char main_norm[] = "main.output_norm";
inline constexpr const char detail_vocal_embedding[] = "token_embd.vocal";
inline constexpr const char detail_bgm_embedding[] = "token_embd.bgm";
inline constexpr const char detail_norm[] = "detail.output_norm";
inline constexpr const char vocal_output[] = "output.vocal";
inline constexpr const char bgm_output[] = "output.bgm";
inline constexpr const char bridge_0_weight[] = "bridge.0.weight";
inline constexpr const char bridge_0_bias[] = "bridge.0.bias";
inline constexpr const char bridge_2_weight[] = "bridge.2.weight";
inline constexpr const char bridge_2_bias[] = "bridge.2.bias";

std::string main_layer(unsigned layer, const char * suffix);
std::string detail_layer(unsigned layer, const char * suffix);
} // namespace tensor_names

struct tokenizer_assets {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::string added_tokens_json;
    std::string config_json;
};

struct model_provenance {
    std::string name;
    std::string model_repository;
    std::string model_revision;
    std::string model_sha256;
    std::string runtime_repository;
    std::string runtime_revision;
    std::string tokenizer_revision;
    std::string tokenizer_sha256;
    // SHA-256 of the GGUF bytes actually loaded. Source model provenance is
    // intentionally separate: F16 and quantized artifacts share it.
    std::string artifact_sha256;
};

struct model_hparams {
    int32_t main_layers = 28;
    int32_t detail_layers = 12;
    int32_t embedding_length = 1536;
    int32_t feed_forward_length = 8960;
    int32_t attention_heads = 12;
    int32_t kv_attention_heads = 12;
    int32_t context_length = 10000;
    float rms_norm_epsilon = 1.0e-5F;
    float main_rope_theta = 500000.0F;
    float detail_rope_theta = 500000.0F;
    int32_t codebook_count = 3;
    int32_t codebook_size = 16384;
    int32_t eos_token_id = 16384;
    int32_t special_token_id = 16385;
    int32_t frame_rate = 25;
    int32_t sample_rate = 48000;
    std::vector<int32_t> delays = {0, 250, 250};
    int32_t lyrics_prefix_length = 600;
    int32_t prompt_prefix_length = 252;
    int32_t style_prefix_length = 100;

    [[nodiscard]] int32_t token_input_size() const noexcept { return codebook_size + 2; }
    [[nodiscard]] int32_t token_output_size() const noexcept { return codebook_size + 1; }
    [[nodiscard]] int32_t head_dimension() const noexcept {
        return attention_heads == 0 ? 0 : embedding_length / attention_heads;
    }
    [[nodiscard]] bool valid(std::string * reason = nullptr) const;
};

struct tensor_data {
    std::vector<int64_t> dimensions; // GGML order, i.e. input dimension first.
    std::vector<float> values;       // F32 test helper payload in GGML order.
};

struct model_load_options {
    ggml_backend_t backend = nullptr; // Borrowed; it must outlive the loaded model.
    // Production artifacts must match one of the reviewed v2 source profiles
    // (currently medium or large). False is reserved for tiny unit fixtures.
    bool require_supported_v2_profile = true;
    bool allow_f32 = true;
    bool allow_f16 = true;
    bool allow_quantized = true;
};

// Owns the backend-resident weight context/buffer.  The backend itself remains
// owned by its caller so a model can share an application-level backend.
class model final {
public:
    model(const model &) = delete;
    model & operator=(const model &) = delete;
    model(model &&) noexcept;
    model & operator=(model &&) noexcept;
    ~model();

    static std::shared_ptr<model> load_gguf(const std::string & path,
                                             const model_load_options & options);

    // Test-only constructor.  It deliberately permits tiny dimensions and F32
    // tensors so graph tests never need the 7+ GiB checkpoint.
    static std::shared_ptr<model> make_test_model(
        const model_hparams & hparams,
        const std::unordered_map<std::string, tensor_data> & tensors,
        ggml_backend_t backend);

    [[nodiscard]] const model_hparams & hparams() const noexcept;
    [[nodiscard]] ggml_backend_t backend() const noexcept;
    [[nodiscard]] ggml_tensor * tensor(const std::string & name) const;
    [[nodiscard]] bool contains(const std::string & name) const noexcept;
    [[nodiscard]] std::size_t tensor_count() const noexcept;
    [[nodiscard]] const tokenizer_assets & tokenizer() const noexcept;
    [[nodiscard]] const model_provenance & provenance() const noexcept;

private:
    model();
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace levo::detail
