#include "levo-resident-model.h"
#include "levo-flow-model.h"

#include <iostream>
#include <stdexcept>

void check(bool value) { if (!value) throw std::runtime_error("residency assertion failed"); }

int main() {
    try {
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        check(device != nullptr);
        using Model = levo::flow::model;
        using Resident = levo::detail::resident_model<Model>;
        std::shared_ptr<Resident> cache;
        int loads = 0;
        const auto loader = [&](ggml_backend_t backend) {
            ++loads;
            return Model::make_test_model({}, {{"test", {{2}, {1, 2}}}}, backend);
        };
        std::weak_ptr<Resident> previous;
        {
            auto first = levo::detail::load_resident_model(&cache, "a", device, loader);
            previous = first;
            check(first->weights->resident_bytes() > 0);
        }
        check(!previous.expired());
        {
            auto second = levo::detail::load_resident_model(&cache, "a", device, loader);
            check(second == previous.lock() && loads == 1);
            float data[2]{};
            ggml_backend_tensor_get(second->weights->tensor("test"), data, 0, sizeof(data));
            check(data[0] == 1 && data[1] == 2);
        }
        // Failed loads leave the old entry intact; changed models replace it.
        try {
            levo::detail::load_resident_model(&cache, "bad", device,
                [](ggml_backend_t) -> std::shared_ptr<Model> { throw std::runtime_error("load"); });
        } catch (const std::runtime_error &) {}
        check(cache == previous.lock());
        levo::detail::load_resident_model(&cache, "b", device, loader);
        check(previous.expired() && loads == 2);
        previous = cache;
        cache.reset();
        check(previous.expired());
        // With retention off, stage exit destroys weights and then backend.
        for (int i = 0; i < 2; ++i) {
            {
                auto temporary = levo::detail::load_resident_model<Model>(nullptr, "a", device, loader);
                previous = temporary;
            }
            check(previous.expired());
        }
        check(loads == 4);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
