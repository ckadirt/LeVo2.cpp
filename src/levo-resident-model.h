#pragma once

#include "ggml-backend.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace levo::detail {

struct resident_backend_deleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend) ggml_backend_free(backend);
    }
};

// Member order guarantees that weights die before their borrowed backend.
// Request graphs, KV sessions and samplers deliberately stay outside this owner.
template <class Model>
struct resident_model {
    std::string path;
    ggml_backend_dev_t device = nullptr;
    std::unique_ptr<ggml_backend, resident_backend_deleter> backend;
    std::shared_ptr<Model> weights;
};

template <class Model, class Loader>
std::shared_ptr<resident_model<Model>> load_resident_model(
    std::shared_ptr<resident_model<Model>> * cache,
    const std::string & path, ggml_backend_dev_t device, Loader && loader) {
    if (cache && *cache && (*cache)->path == path && (*cache)->device == device) return *cache;
    auto loaded = std::make_shared<resident_model<Model>>();
    loaded->path = path;
    loaded->device = device;
    loaded->backend.reset(ggml_backend_dev_init(device, nullptr));
    if (!loaded->backend) throw std::runtime_error("cannot initialize model backend");
    loaded->weights = loader(loaded->backend.get());
    // A failed load never publishes a half-constructed cache entry.
    if (cache) *cache = loaded;
    return loaded;
}

} // namespace levo::detail
