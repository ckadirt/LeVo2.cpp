#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace levo {

enum class backend_kind {
    auto_select,
    cpu,
    cuda,
};

struct backend_info {
    std::string name;
    std::string description;
    backend_kind kind = backend_kind::cpu;
    std::size_t memory_free = 0;
    std::size_t memory_total = 0;
};

const char * version() noexcept;

// Enumerates backends registered in the linked GGML build.
std::vector<backend_info> available_backends();

// Runs a deterministic GGML vector addition and returns its four output values.
// This is the foundation smoke test and is retained as a backend diagnostic.
std::vector<float> backend_smoke(backend_kind kind = backend_kind::auto_select,
                                 int device_index = 0);

} // namespace levo
