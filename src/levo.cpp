#include "levo.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace levo {
namespace {

struct backend_deleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct context_deleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

backend_kind classify_device(ggml_backend_dev_t device) {
    switch (ggml_backend_dev_type(device)) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return backend_kind::cuda;
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
        default:
            return backend_kind::cpu;
    }
}

ggml_backend_dev_t select_device(backend_kind requested, int device_index) {
    if (device_index < 0) {
        throw std::invalid_argument("device index cannot be negative");
    }

    ggml_backend_load_all();

    const auto wanted = requested == backend_kind::cuda
        ? GGML_BACKEND_DEVICE_TYPE_GPU
        : GGML_BACKEND_DEVICE_TYPE_CPU;

    if (requested == backend_kind::auto_select) {
        if (auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
            return gpu;
        }
        if (auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            return cpu;
        }
        throw std::runtime_error("GGML did not register a usable CPU or GPU backend");
    }

    int match = 0;
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) == wanted) {
            if (match == device_index) {
                return device;
            }
            ++match;
        }
    }

    const char * label = requested == backend_kind::cuda ? "CUDA/GPU" : "CPU";
    throw std::runtime_error(std::string("requested ") + label +
                             " backend device is unavailable");
}

} // namespace

const char * version() noexcept {
    return LEVO_VERSION;
}

std::vector<backend_info> available_backends() {
    ggml_backend_load_all();

    std::vector<backend_info> result;
    result.reserve(ggml_backend_dev_count());
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        backend_info info;
        info.name = ggml_backend_dev_name(device);
        info.description = ggml_backend_dev_description(device);
        info.kind = classify_device(device);
        ggml_backend_dev_memory(device, &info.memory_free, &info.memory_total);
        result.push_back(std::move(info));
    }
    return result;
}

std::vector<float> backend_smoke(backend_kind kind, int device_index) {
    auto * device = select_device(kind, device_index);
    backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) {
        throw std::runtime_error("failed to initialize requested GGML backend");
    }

    constexpr std::size_t tensor_count = 3;
    const std::size_t context_size = tensor_count * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(8, false);
    std::vector<std::byte> context_storage(context_size);
    ggml_init_params params = {
        context_size,
        context_storage.data(),
        true,
    };
    context_ptr context(ggml_init(params));
    if (!context) {
        throw std::runtime_error("failed to create GGML context");
    }

    auto * lhs = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * rhs = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * sum = ggml_add(context.get(), lhs, rhs);
    auto * graph = ggml_new_graph_custom(context.get(), 8, false);
    ggml_build_forward_expand(graph, sum);

    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
    if (!buffer) {
        throw std::runtime_error("failed to allocate GGML backend tensor buffer");
    }

    const std::array<float, 4> lhs_data = {1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> rhs_data = {4.0F, 3.0F, 2.0F, 1.0F};
    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, sizeof(lhs_data));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, sizeof(rhs_data));

    const auto status = ggml_backend_graph_compute(backend.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("GGML smoke graph failed: ") +
                                 ggml_status_to_string(status));
    }

    std::vector<float> output(4);
    ggml_backend_tensor_get(sum, output.data(), 0, output.size() * sizeof(float));
    return output;
}

} // namespace levo
