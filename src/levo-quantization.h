#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

// Quantization policy is deliberately shared by the writer and the strict
// loaders.  A GGUF is therefore accepted only when every tensor has the type
// its declared profile prescribes; accepting an arbitrary mixed GGUF would
// make model identity and numerical expectations ambiguous.
namespace levo::quantization {

inline constexpr const char profile_key[] = "levo2.quantization.profile";
inline constexpr const char policy_revision_key[] = "levo2.quantization.policy_revision";
inline constexpr const char source_artifact_sha256_key[] = "levo2.quantization.source_artifact_sha256";
inline constexpr const char flow_padded_input_key[] = "levo2.quantization.flow_padded_input";
inline constexpr const char policy_revision[] = "1";

enum class profile {
    q8_0,
    q6_k,
    q5_k_m,
    q4_k_m,
};

inline const char * name(profile value) noexcept {
    switch (value) {
        case profile::q8_0: return "Q8_0";
        case profile::q6_k: return "Q6_K";
        case profile::q5_k_m: return "Q5_K_M";
        case profile::q4_k_m: return "Q4_K_M";
    }
    return "unknown";
}

inline std::optional<profile> parse(const std::string & value) {
    if (value == "Q8_0") return profile::q8_0;
    if (value == "Q6_K") return profile::q6_k;
    if (value == "Q5_K_M") return profile::q5_k_m;
    if (value == "Q4_K_M") return profile::q4_k_m;
    return std::nullopt;
}

inline ggml_type base_type(profile value) noexcept {
    switch (value) {
        case profile::q8_0: return GGML_TYPE_Q8_0;
        case profile::q6_k: return GGML_TYPE_Q6_K;
        case profile::q5_k_m: return GGML_TYPE_Q5_K;
        case profile::q4_k_m: return GGML_TYPE_Q4_K;
    }
    return GGML_TYPE_COUNT;
}

inline uint32_t gguf_file_type(profile value) noexcept {
    // GGML's standard file-type identifiers.  K_M is a tensor-routing
    // profile, not a separate GGML storage type.
    switch (value) {
        case profile::q8_0: return 7;
        case profile::q4_k_m: return 12;
        case profile::q5_k_m: return 13;
        case profile::q6_k: return 14;
    }
    return 0;
}

inline bool is_quantized(ggml_type type) noexcept {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

inline bool uses_promoted_matrices(profile value) noexcept {
    return value == profile::q5_k_m || value == profile::q4_k_m;
}

inline bool lelm_critical_matrix(const std::string & tensor_name) {
    if (tensor_name.rfind("token_embd.", 0) == 0 ||
        tensor_name.rfind("output.", 0) == 0 ||
        tensor_name.rfind("cond.", 0) == 0 ||
        tensor_name == "bridge.0.weight" || tensor_name == "bridge.2.weight") {
        return true;
    }
    return tensor_name.find(".attn_v.weight") != std::string::npos ||
           tensor_name.find(".attn_output.weight") != std::string::npos ||
           tensor_name.find(".ffn_down.weight") != std::string::npos;
}

inline bool ends_with(const std::string & value, const char * suffix) {
    const std::size_t suffix_size = std::char_traits<char>::length(suffix);
    return value.size() >= suffix_size &&
           value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

inline ggml_type lelm_tensor_type(const std::string & tensor_name, int rank, profile value) {
    if (rank < 2) return GGML_TYPE_F32;
    if (uses_promoted_matrices(value) && lelm_critical_matrix(tensor_name)) return GGML_TYPE_Q6_K;
    return base_type(value);
}

inline bool flow_block_matrix(const std::string & tensor_name) {
    if (tensor_name.rfind("flow.block.", 0) != 0) return false;
    return ends_with(tensor_name, ".attn.qkv.weight") ||
           ends_with(tensor_name, ".attn.out.weight") ||
           ends_with(tensor_name, ".ffn.in.weight") ||
           ends_with(tensor_name, ".ffn.out.weight");
}

inline bool flow_critical_matrix(const std::string & tensor_name) {
    return ends_with(tensor_name, ".attn.out.weight") || ends_with(tensor_name, ".ffn.out.weight");
}

inline ggml_type flow_tensor_type(const std::string & tensor_name, int rank, profile value) {
    if (rank < 2 || !flow_block_matrix(tensor_name)) return GGML_TYPE_F32;
    if (uses_promoted_matrices(value) && flow_critical_matrix(tensor_name)) return GGML_TYPE_Q6_K;
    return base_type(value);
}

inline int64_t padded_input_columns(int64_t columns, ggml_type type) {
    if (!is_quantized(type)) return columns;
    const int64_t block = ggml_blck_size(type);
    if (columns <= 0 || block <= 0) return columns;
    return ((columns + block - 1) / block) * block;
}

inline std::string flow_padded_input_layout(profile value) {
    return "hidden=2200->" + std::to_string(padded_input_columns(2200, base_type(value))) +
           ";intermediate=4400->" + std::to_string(padded_input_columns(4400, base_type(value)));
}

inline bool is_hex_sha256(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
}

} // namespace levo::quantization
