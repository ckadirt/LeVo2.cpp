#pragma once

#include "levo.h"

#include <cstdint>
#include <string>
#include <vector>

namespace levo::engine_request {

struct request {
    std::string lyrics;
    std::string description;
    double duration_seconds = 0.0;
    bool seed_present = false;
    std::uint64_t seed = 0;
    float cfg_scale = 1.5F;
    generation_sampling_config sampling;
};

// Strict request parsing for the C engine. The accepted object is deliberately
// small: lyrics, description, duration_seconds (or duration), optional seed,
// cfg_scale, and an optional sampling object. Unknown fields are rejected so a
// caller cannot believe that an ignored setting affected a checkpoint.
request parse(const std::string & json);

// Canonical, stable request JSON. A resolved seed is always emitted.
std::string serialize(const request & value);

generation_config generation_config_for(const request & value,
                                        const std::string & model_path);

// Completed CODES output consumed by the later DIFFUSE stage. It repeats the
// resolved request and provides canonical stream-major int32 audio codes.
std::string serialize_codes(const request & value, const generation_result & result);

} // namespace levo::engine_request
