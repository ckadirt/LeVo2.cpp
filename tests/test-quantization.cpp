#include "levo-quantization.h"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        using levo::quantization::profile;
        using levo::quantization::lelm_tensor_type;
        using levo::quantization::flow_tensor_type;

        require(levo::quantization::parse("Q8_0") == profile::q8_0, "Q8_0 profile parse failed");
        require(levo::quantization::parse("Q6_K") == profile::q6_k, "Q6_K profile parse failed");
        require(levo::quantization::parse("Q5_K_M") == profile::q5_k_m, "Q5_K_M profile parse failed");
        require(levo::quantization::parse("Q4_K_M") == profile::q4_k_m, "Q4_K_M profile parse failed");
        require(!levo::quantization::parse("q4"), "unsupported profile was accepted");

        require(lelm_tensor_type("main.output_norm", 1, profile::q4_k_m) == GGML_TYPE_F32,
                "LeLM vector must remain F32");
        require(lelm_tensor_type("main.blk.0.attn_q.weight", 2, profile::q4_k_m) == GGML_TYPE_Q4_K,
                "LeLM non-critical Q4 matrix type is wrong");
        require(lelm_tensor_type("main.blk.0.attn_v.weight", 2, profile::q4_k_m) == GGML_TYPE_Q6_K,
                "LeLM attention V promotion is wrong");
        require(lelm_tensor_type("cond.lyrics.weight", 2, profile::q5_k_m) == GGML_TYPE_Q6_K,
                "LeLM conditioner promotion is wrong");
        require(lelm_tensor_type("output.mixed", 2, profile::q6_k) == GGML_TYPE_Q6_K,
                "LeLM Q6 routing is wrong");

        require(flow_tensor_type("flow.time_modulation.weight", 2, profile::q4_k_m) == GGML_TYPE_F32,
                "Flow control must remain F32");
        require(flow_tensor_type("flow.block.0.attn.qkv.weight", 2, profile::q4_k_m) == GGML_TYPE_Q4_K,
                "Flow base Q4 routing is wrong");
        require(flow_tensor_type("flow.block.0.attn.out.weight", 2, profile::q4_k_m) == GGML_TYPE_Q6_K,
                "Flow attention output promotion is wrong");
        require(flow_tensor_type("flow.block.0.ffn.out.weight", 2, profile::q5_k_m) == GGML_TYPE_Q6_K,
                "Flow FFN output promotion is wrong");
        require(levo::quantization::padded_input_columns(2200, GGML_TYPE_Q8_0) == 2208,
                "Q8 Flow padding is wrong");
        require(levo::quantization::padded_input_columns(2200, GGML_TYPE_Q4_K) == 2304,
                "K Flow hidden padding is wrong");
        require(levo::quantization::padded_input_columns(4400, GGML_TYPE_Q6_K) == 4608,
                "K Flow intermediate padding is wrong");
        require(levo::quantization::padded_input_columns(2200, GGML_TYPE_F32) == 2200,
                "F32 must not be padded");
        require(levo::quantization::is_hex_sha256(std::string(64, 'a')),
                "valid SHA-256 was rejected");
        require(!levo::quantization::is_hex_sha256("ABC"), "invalid SHA-256 was accepted");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
