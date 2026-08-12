#include "levo-cuda-precision.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace levo::detail {
namespace {

bool is_cuda_device(ggml_backend_dev_t device) {
    if (device == nullptr) return false;
    const auto type = ggml_backend_dev_type(device);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) return false;
    return std::string(ggml_backend_dev_name(device)).rfind("CUDA", 0) == 0;
}

void set_if_unset(const char * name, const char * value) {
    if (std::getenv(name) != nullptr) return;
#if defined(_WIN32)
    if (_putenv_s(name, value) != 0) {
        throw std::runtime_error(std::string("LeVo CUDA precision: could not set ") + name);
    }
#else
    if (::setenv(name, value, 0) != 0) {
        throw std::runtime_error(std::string("LeVo CUDA precision: could not set ") + name);
    }
#endif
}

} // namespace

void configure_cuda_gemm_f32_accumulation(ggml_backend_dev_t device) {
    if (!is_cuda_device(device)) return;
    set_if_unset("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32");
}

void configure_cuda_disable_tf32(ggml_backend_dev_t device) {
    if (!is_cuda_device(device)) return;
    set_if_unset("NVIDIA_TF32_OVERRIDE", "0");
}

} // namespace levo::detail
