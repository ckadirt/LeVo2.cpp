#include "levo-flow-estimator.h"

#include "ggml-cpu.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using levo::flow::flow_hparams;
using levo::flow::tensor_data;

tensor_data zeros(std::initializer_list<int64_t> shape) {
    std::size_t count = 1;
    std::vector<int64_t> dimensions(shape);
    for (const int64_t dimension : dimensions) count *= static_cast<std::size_t>(dimension);
    return {std::move(dimensions), std::vector<float>(count, 0.0F)};
}

std::unordered_map<std::string, tensor_data> zero_estimator_tensors(const flow_hparams & hp) {
    std::unordered_map<std::string, tensor_data> tensors;
    tensors.emplace("flow.position_embedding.weight", zeros({hp.hidden_size, hp.max_frames}));
    tensors.emplace("flow.time_embedding.linear_1.weight", zeros({hp.time_embedding_dim, hp.hidden_size}));
    tensors.emplace("flow.time_embedding.linear_1.bias", zeros({hp.hidden_size}));
    tensors.emplace("flow.time_embedding.linear_2.weight", zeros({hp.hidden_size, hp.hidden_size}));
    tensors.emplace("flow.time_embedding.linear_2.bias", zeros({hp.hidden_size}));
    tensors.emplace("flow.time_modulation.weight", zeros({hp.hidden_size, 6 * hp.hidden_size}));
    tensors.emplace("flow.time_modulation.bias", zeros({6 * hp.hidden_size}));
    for (int layer = 0; layer < hp.n_layer; ++layer) {
        const std::string root = "flow.block." + std::to_string(layer);
        tensors.emplace(root + ".attn.qkv.weight", zeros({hp.hidden_size, 3 * hp.hidden_size}));
        tensors.emplace(root + ".attn.qkv.bias", zeros({3 * hp.hidden_size}));
        tensors.emplace(root + ".attn.out.weight", zeros({hp.hidden_size, hp.hidden_size}));
        tensors.emplace(root + ".attn.out.bias", zeros({hp.hidden_size}));
        tensors.emplace(root + ".norm_1.weight", zeros({hp.hidden_size}));
        tensors.emplace(root + ".norm_1.bias", zeros({hp.hidden_size}));
        tensors.emplace(root + ".norm_2.weight", zeros({hp.hidden_size}));
        tensors.emplace(root + ".norm_2.bias", zeros({hp.hidden_size}));
        tensors.emplace(root + ".ffn.in.weight", zeros({hp.hidden_size, hp.intermediate_size}));
        tensors.emplace(root + ".ffn.in.bias", zeros({hp.intermediate_size}));
        tensors.emplace(root + ".ffn.out.weight", zeros({hp.intermediate_size, hp.hidden_size}));
        tensors.emplace(root + ".ffn.out.bias", zeros({hp.hidden_size}));
        tensors.emplace(root + ".modulation.weight", zeros({hp.hidden_size, 6}));
    }
    tensors.emplace("flow.final_norm.weight", zeros({hp.hidden_size}));
    tensors.emplace("flow.final_norm.bias", zeros({hp.hidden_size}));
    tensors.emplace("flow.final_modulation.weight", zeros({hp.hidden_size, 2}));
    tensors.emplace("flow.output.weight", zeros({hp.hidden_size, hp.hidden_size}));
    tensors.emplace("flow.output.bias", zeros({hp.hidden_size}));
    return tensors;
}

template <typename Function>
bool throws(Function && function) {
    try {
        function();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    try {
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("cannot initialize CPU backend");

        flow_hparams hp;
        hp.hidden_size = 4;
        hp.n_layer = 1;
        hp.n_head = 1;
        hp.head_dim = 4;
        hp.intermediate_size = 8;
        hp.max_frames = 8;
        hp.time_embedding_dim = 2;
        hp.latent_dim = 2;
        {
            const auto model = levo::flow::model::make_test_model(hp, zero_estimator_tensors(hp), backend);
            const auto runtime = levo::flow::estimator::create(model);

            levo::flow::estimator_input input;
            input.batch = 2;
            input.frames = 3;
            input.model_input.resize(input.batch * input.frames * hp.hidden_size);
            for (std::size_t i = 0; i < input.model_input.size(); ++i) input.model_input[i] = static_cast<float>(i + 1);
            input.timesteps = {0.5F, 0.25F};
            levo::flow::estimator_capture capture;
            const std::vector<float> output = runtime->velocity(input, &capture);
            const std::size_t expected = input.batch * input.frames * static_cast<std::size_t>(hp.hidden_size);
            const std::size_t expected_velocity = input.batch * input.frames * static_cast<std::size_t>(hp.latent_dim);
            if (output.size() != expected_velocity || capture.full_output.size() != expected ||
                capture.block0_input.size() != expected || capture.block0_output.size() != expected ||
                capture.timestep_embedding.size() != input.batch * static_cast<std::size_t>(hp.hidden_size) ||
                capture.timestep_modulation.size() != input.batch * 6U * static_cast<std::size_t>(hp.hidden_size)) {
                throw std::runtime_error("unexpected Flow estimator capture shape");
            }
            for (const float value : output) {
                if (value != 0.0F || !std::isfinite(value)) throw std::runtime_error("zero Flow estimator output is incorrect");
            }
            if (!throws([&] {
                    levo::flow::estimator_input bad = input;
                    bad.model_input.pop_back();
                    (void) runtime->velocity(bad);
                })) throw std::runtime_error("invalid model_input shape was accepted");
            if (!throws([&] {
                    levo::flow::estimator_input bad = input;
                    bad.timesteps[0] = std::numeric_limits<float>::quiet_NaN();
                    (void) runtime->velocity(bad);
                })) throw std::runtime_error("non-finite timestep was accepted");
        }
        ggml_backend_free(backend);
        std::cout << "flow estimator ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
