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
    // Retained while CODES runs so a completed CODES value is self-contained
    // input to DIFFUSE.
    std::uint64_t flow_seed = 0;
    std::size_t flow_euler_steps = 0; // zero selects the Flow GGUF default
    float flow_cfg_scale = 0.0F;      // zero selects the Flow GGUF default
};

struct codes {
    request generation_request;
    std::size_t frame_count = 0;
    std::vector<std::int32_t> tokens; // stream-major [3,T]
    std::string token_sha256;
};

// Strict request parsing for the C engine. The accepted object is deliberately
// small: lyrics, description, duration_seconds (or duration), optional seed,
// cfg_scale, and an optional sampling object. Unknown fields are rejected so a
// caller cannot believe that an ignored setting affected a checkpoint.
request parse(const std::string & json);

// Strict parser for a CODES completion emitted by this engine.
codes parse_codes(const std::string & json);

// Canonical, stable request JSON. A resolved seed is always emitted.
std::string serialize(const request & value);

generation_config generation_config_for(const request & value,
                                        const std::string & model_path);

// Completed CODES output consumed by the later DIFFUSE stage. It repeats the
// resolved request and provides canonical stream-major int32 audio codes.
std::string serialize_codes(const request & value, const generation_result & result);

} // namespace levo::engine_request
