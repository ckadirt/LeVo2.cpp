#pragma once

#include "ggml-backend.h"

namespace levo::detail {

// Upstream GGML creates every cuBLAS handle with CUBLAS_TF32_TENSOR_OP_MATH
// (ggml/src/ggml-cuda/common.cuh), so an F32 GEMM that reaches cuBLAS is
// computed in TF32 with a 10-bit mantissa. Small matrices never reach cuBLAS,
// which is why short-sequence parity looks exact while a 1000-frame Flow window
// diverges by ~1e-1. Requesting NVIDIA's documented TF32 override restores true
// F32 accumulation for the renderer's strict correctness path.
//
// Both helpers are no-ops on non-CUDA devices and never overwrite a value the
// caller already set in the environment, so an explicit override still wins.
// They must run before the backend is initialized.
void configure_cuda_gemm_f32_accumulation(ggml_backend_dev_t device);
void configure_cuda_disable_tf32(ggml_backend_dev_t device);

} // namespace levo::detail
