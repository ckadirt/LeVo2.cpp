// Asset-free guard for the renderer's F32 GEMM precision on CUDA.
//
// Upstream GGML creates every cuBLAS handle with CUBLAS_TF32_TENSOR_OP_MATH, so
// an F32 matrix multiply large enough to reach cuBLAS silently runs in TF32
// (10-bit mantissa). That is invisible at the two-frame parity boundary and
// costs three orders of magnitude at a full 1000-frame Flow window, so the
// renderer requests NVIDIA's TF32 override before initializing the backend.
//
// This test multiplies matrices whose reduction is long enough to reach cuBLAS
// and fails if the result carries TF32-sized error. TF32 lands near 1e-3
// relative; true F32 lands near 1e-7.
#include "levo-cuda-precision.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int k_reduction = 2200; // one Flow hidden dimension
constexpr int k_rows = 256;
constexpr int k_cols = 256;
// TF32 on this shape lands around 1e-3; true F32 is near 1e-7. The gate sits
// well below any plausible TF32 result and well above F32 rounding.
constexpr double k_max_relative = 1.0e-5;

ggml_backend_dev_t find_device(const std::string & requested) {
    ggml_backend_load_all();
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const auto type = ggml_backend_dev_type(device);
        if (requested == "cpu" && type == GGML_BACKEND_DEVICE_TYPE_CPU) return device;
        if (requested == "cuda" && (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            std::string(ggml_backend_dev_name(device)).rfind("CUDA", 0) == 0) {
            return device;
        }
    }
    return nullptr;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string backend_name = argc > 1 ? argv[1] : "cpu";
    ggml_backend_dev_t device = find_device(backend_name);
    if (device == nullptr) {
        std::printf("cuda precision skipped: %s backend unavailable\n", backend_name.c_str());
        return 0;
    }
    levo::detail::configure_cuda_gemm_f32_accumulation(device);
    levo::detail::configure_cuda_disable_tf32(device);
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "cannot initialize %s backend\n", backend_name.c_str());
        return 1;
    }

    std::mt19937 engine(1234);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> lhs(static_cast<std::size_t>(k_reduction) * k_rows);
    std::vector<float> rhs(static_cast<std::size_t>(k_reduction) * k_cols);
    for (float & value : lhs) value = distribution(engine);
    for (float & value : rhs) value = distribution(engine);

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_2d(context, GGML_TYPE_F32, k_reduction, k_rows);
    ggml_tensor * b = ggml_new_tensor_2d(context, GGML_TYPE_F32, k_reduction, k_cols);
    ggml_tensor * product = ggml_mul_mat(context, a, b);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, product);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "cannot allocate tensors\n");
        return 1;
    }
    ggml_backend_tensor_set(a, lhs.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, rhs.data(), 0, ggml_nbytes(b));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        return 1;
    }
    std::vector<float> result(static_cast<std::size_t>(k_rows) * k_cols);
    ggml_backend_tensor_get(product, result.data(), 0, ggml_nbytes(product));

    // Double-precision reference on the host.
    double worst = 0.0;
    for (int col = 0; col < k_cols; ++col) {
        for (int row = 0; row < k_rows; ++row) {
            double reference = 0.0;
            for (int index = 0; index < k_reduction; ++index) {
                reference += static_cast<double>(lhs[static_cast<std::size_t>(row) * k_reduction + index]) *
                             static_cast<double>(rhs[static_cast<std::size_t>(col) * k_reduction + index]);
            }
            const double actual = result[static_cast<std::size_t>(col) * k_rows + row];
            // Normalize by the reduction scale rather than by the individual
            // element, whose cancellation would inflate a relative error.
            const double scale = std::sqrt(static_cast<double>(k_reduction) / 3.0);
            worst = std::max(worst, std::abs(actual - reference) / scale);
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);

    std::printf("cuda precision %s worst_scaled_error=%.3e gate=%.3e\n", backend_name.c_str(), worst, k_max_relative);
    if (!(worst <= k_max_relative)) {
        std::fprintf(stderr, "F32 matmul carries reduced-precision error; TF32 is likely enabled\n");
        return 1;
    }
    std::printf("cuda precision ok\n");
    return 0;
}
