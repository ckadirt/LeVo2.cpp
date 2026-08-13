#include "cantor_engine.h"

#include "levo-checkpoint.h"
#include "levo-cuda-precision.h"
#include "levo-engine-request.h"
#include "levo-flow-model.h"
#include "levo-flow-renderer.h"
#include "levo-generator.h"
#include "levo-renderer-pattern.h"
#include "levo-token-io.h"
#include "levo-vae.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct cantor_ctx {
    std::string lm_path;
    std::string dit_path;
    std::string vae_path;
    cantor_load_opts options{};
    std::vector<float> audio;
    int audio_samples = 0;
};

namespace {

constexpr std::array<char, 8> lm_magic{{'L', 'E', 'V', 'O', 'L', 'M', '0', '2'}};
constexpr std::array<char, 8> flow_magic{{'L', 'E', 'V', 'O', 'F', 'L', '0', '2'}};
constexpr std::array<char, 8> latent_magic{{'L', 'E', 'V', 'O', 'L', 'T', '0', '2'}};
constexpr std::array<char, 8> lm_magic_v1{{'L', 'E', 'V', 'O', 'L', 'M', '0', '1'}};
constexpr std::array<char, 8> flow_magic_v1{{'L', 'E', 'V', 'O', 'F', 'L', '0', '1'}};
constexpr std::array<char, 8> latent_magic_v1{{'L', 'E', 'V', 'O', 'L', 'T', '0', '1'}};

struct backend_deleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend != nullptr) ggml_backend_free(backend);
    }
};
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

thread_local cantor_error g_last_error_code = CANTOR_OK;
thread_local std::string g_last_error;

void clear_error() {
    g_last_error_code = CANTOR_OK;
    g_last_error.clear();
}

void set_error(cantor_error code, const std::string & message) {
    g_last_error_code = code;
    g_last_error = message;
}

template <typename F>
cantor_status guard(F && operation) {
    try {
        return operation();
    } catch (const std::bad_alloc &) {
        set_error(CANTOR_ERR_OOM, "[LeVo ABI] allocation failed");
    } catch (const std::invalid_argument & error) {
        set_error(CANTOR_ERR_OTHER, std::string("[LeVo ABI] invalid argument: ") + error.what());
    } catch (const std::exception & error) {
        set_error(CANTOR_ERR_OTHER, std::string("[LeVo ABI] ") + error.what());
    } catch (...) {
        set_error(CANTOR_ERR_OTHER, "[LeVo ABI] unknown exception");
    }
    return CANTOR_ERR;
}

bool cancelled(cantor_cancel_fn callback, void * userdata) {
    return callback != nullptr && callback(userdata) != 0;
}

bool is_gpu(ggml_backend_dev_t device) {
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

bool is_cuda(ggml_backend_dev_t device) {
    return is_gpu(device) && std::string(ggml_backend_dev_name(device)).rfind("CUDA", 0) == 0;
}

ggml_backend_dev_t select_engine_device() {
    ggml_backend_load_all();
    const auto first = [](const auto & predicate) -> ggml_backend_dev_t {
        for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            ggml_backend_dev_t device = ggml_backend_dev_get(index);
            if (predicate(device)) return device;
        }
        return nullptr;
    };
    if (ggml_backend_dev_t device = first([](ggml_backend_dev_t value) { return is_cuda(value); })) return device;
    if (ggml_backend_dev_t device = first([](ggml_backend_dev_t value) { return is_gpu(value); })) return device;
    if (ggml_backend_dev_t device = first([](ggml_backend_dev_t value) {
            return ggml_backend_dev_type(value) == GGML_BACKEND_DEVICE_TYPE_CPU;
        })) return device;
    throw std::runtime_error("no usable GGML backend is registered");
}

std::vector<float> native_gaussian_noise(std::size_t count, std::uint64_t seed) {
    auto splitmix64 = [](std::uint64_t & state) {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t value = state;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    };
    const auto uniform_open = [&splitmix64](std::uint64_t & state) {
        constexpr double scale = 1.0 / 9007199254740992.0;
        return (static_cast<double>(splitmix64(state) >> 11U) + 0.5) * scale;
    };
    std::vector<float> result(count);
    std::uint64_t state = seed;
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (std::size_t offset = 0; offset < count; offset += 2U) {
        const double radius = std::sqrt(-2.0 * std::log(uniform_open(state)));
        const double angle = two_pi * uniform_open(state);
        result[offset] = static_cast<float>(radius * std::cos(angle));
        if (offset + 1U < count) result[offset + 1U] = static_cast<float>(radius * std::sin(angle));
    }
    return result;
}

void emit_progress(cantor_progress_fn callback, void * userdata, cantor_stage stage,
                   std::size_t completed, std::size_t total) {
    if (callback == nullptr) return;
    if (completed > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("progress counter exceeds Cantor ABI range");
    }
    callback(stage, static_cast<int>(completed), static_cast<int>(total), userdata);
}

bool blob_alloc(const std::vector<std::uint8_t> & source, std::uint8_t ** out, std::size_t * out_len) {
    std::uint8_t * result = static_cast<std::uint8_t *>(std::malloc(source.empty() ? 1U : source.size()));
    if (result == nullptr) {
        set_error(CANTOR_ERR_OOM, "[LeVo ABI] cannot allocate stage output blob");
        return false;
    }
    if (!source.empty()) std::memcpy(result, source.data(), source.size());
    *out = result;
    *out_len = source.size();
    return true;
}

bool blob_alloc_string(const std::string & source, std::uint8_t ** out, std::size_t * out_len) {
    return blob_alloc(std::vector<std::uint8_t>(source.begin(), source.end()), out, out_len);
}

void append_u64(std::vector<std::uint8_t> & out, std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u32(std::vector<std::uint8_t> & out, std::uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_f32(std::vector<std::uint8_t> & out, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "LeVo checkpoint requires IEEE binary32 floats");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(out, bits);
}

std::uint64_t take_u64(const std::vector<std::uint8_t> & source, std::size_t * offset) {
    if (*offset > source.size() || source.size() - *offset < 8) throw std::runtime_error("truncated LeLM checkpoint metadata");
    std::uint64_t result = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) result |= static_cast<std::uint64_t>(source[(*offset)++]) << shift;
    return result;
}

std::uint32_t take_u32(const std::vector<std::uint8_t> & source, std::size_t * offset) {
    if (*offset > source.size() || source.size() - *offset < 4) throw std::runtime_error("truncated LeLM checkpoint metadata");
    std::uint32_t result = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) result |= static_cast<std::uint32_t>(source[(*offset)++]) << shift;
    return result;
}

float take_f32(const std::vector<std::uint8_t> & source, std::size_t * offset) {
    const std::uint32_t bits = take_u32(source, offset);
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    if (!std::isfinite(result)) throw std::runtime_error("checkpoint contains a non-finite float");
    return result;
}

std::vector<std::uint8_t> encode_f32(const std::vector<float> & source, const char * label) {
    if (source.size() > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error(std::string(label) + " is too large");
    }
    std::vector<std::uint8_t> result;
    result.reserve(source.size() * sizeof(float));
    for (const float value : source) {
        if (!std::isfinite(value)) throw std::runtime_error(std::string(label) + " contains a non-finite float");
        append_f32(result, value);
    }
    return result;
}

std::vector<float> decode_f32(const std::vector<std::uint8_t> & source, const char * label) {
    if (source.size() % sizeof(float) != 0) throw std::runtime_error(std::string(label) + " byte size is not float32-aligned");
    std::vector<float> result;
    result.reserve(source.size() / sizeof(float));
    std::size_t offset = 0;
    while (offset != source.size()) result.push_back(take_f32(source, &offset));
    return result;
}

void append_string(std::vector<std::uint8_t> & out, const std::string & value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("checkpoint stamp is too long");
    append_u32(out, static_cast<std::uint32_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

std::string take_string(const std::vector<std::uint8_t> & source, std::size_t * offset) {
    const std::uint32_t size = take_u32(source, offset);
    if (size > source.size() - *offset) throw std::runtime_error("truncated LeLM checkpoint stamp");
    std::string result(reinterpret_cast<const char *>(source.data() + *offset), size);
    *offset += size;
    return result;
}

struct lm_stamp {
    std::string backend;
    std::string model_sha256;
    std::string artifact_sha256;
    std::string runtime_revision;
    std::string tokenizer_sha256;
};

std::vector<std::uint8_t> encode_lm_metadata(const levo::detail::generation_resume_state & state,
                                             const lm_stamp & stamp) {
    std::vector<std::uint8_t> result;
    result.reserve(3U * 8U + 3U + state.next_logits_digest.size() + 4U * 4U +
                   stamp.backend.size() + stamp.model_sha256.size() + stamp.artifact_sha256.size() + stamp.runtime_revision.size() + stamp.tokenizer_sha256.size());
    append_u64(result, state.sampler_seed);
    append_u64(result, state.sampler_draws);
    append_u64(result, state.earliest_eos);
    for (const bool ended : state.ended) result.push_back(ended ? 1U : 0U);
    result.insert(result.end(), state.next_logits_digest.begin(), state.next_logits_digest.end());
    append_string(result, stamp.backend);
    append_string(result, stamp.model_sha256);
    append_string(result, stamp.artifact_sha256);
    append_string(result, stamp.runtime_revision);
    append_string(result, stamp.tokenizer_sha256);
    return result;
}

std::pair<levo::detail::generation_resume_state, lm_stamp> decode_lm_metadata(
    const std::vector<std::uint8_t> & source) {
    levo::detail::generation_resume_state state;
    lm_stamp stamp;
    std::size_t offset = 0;
    state.sampler_seed = take_u64(source, &offset);
    state.sampler_draws = take_u64(source, &offset);
    const std::uint64_t earliest_eos = take_u64(source, &offset);
    if (earliest_eos > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("LeLM checkpoint EOS index exceeds this platform");
    }
    state.earliest_eos = static_cast<std::size_t>(earliest_eos);
    if (source.size() - offset < state.ended.size() + state.next_logits_digest.size()) {
        throw std::runtime_error("truncated LeLM checkpoint metadata state");
    }
    for (std::size_t index = 0; index != state.ended.size(); ++index) {
        if (source[offset] > 1U) throw std::runtime_error("invalid LeLM checkpoint EOS state");
        state.ended[index] = source[offset++] != 0;
    }
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(offset), state.next_logits_digest.size(), state.next_logits_digest.begin());
    offset += state.next_logits_digest.size();
    stamp.backend = take_string(source, &offset);
    stamp.model_sha256 = take_string(source, &offset);
    stamp.artifact_sha256 = take_string(source, &offset);
    stamp.runtime_revision = take_string(source, &offset);
    stamp.tokenizer_sha256 = take_string(source, &offset);
    if (offset != source.size()) throw std::runtime_error("trailing LeLM checkpoint metadata bytes");
    return {std::move(state), std::move(stamp)};
}

std::vector<std::uint8_t> encode_delayed(const std::vector<std::vector<std::int64_t>> & sequence) {
    if (sequence.size() != 3 || sequence.front().empty()) throw std::runtime_error("cannot serialize invalid delayed sequence");
    const std::size_t steps = sequence.front().size();
    if (steps > std::numeric_limits<std::size_t>::max() / (3U * sizeof(std::int32_t))) throw std::runtime_error("delayed sequence is too large");
    std::vector<std::uint8_t> result;
    result.reserve(steps * 3U * sizeof(std::int32_t));
    for (const auto & stream : sequence) {
        if (stream.size() != steps) throw std::runtime_error("delayed sequence rows differ");
        for (const std::int64_t value : stream) {
            if (value < 0 || value > std::numeric_limits<std::int32_t>::max()) throw std::runtime_error("delayed ID cannot fit int32");
            append_u32(result, static_cast<std::uint32_t>(value));
        }
    }
    return result;
}

std::vector<std::vector<std::int64_t>> decode_delayed(const std::vector<std::uint8_t> & source) {
    if (source.empty() || source.size() % (3U * sizeof(std::uint32_t)) != 0) {
        throw std::runtime_error("delayed checkpoint payload is not [3,S] int32");
    }
    const std::size_t steps = source.size() / (3U * sizeof(std::uint32_t));
    std::vector<std::vector<std::int64_t>> result(3, std::vector<std::int64_t>(steps));
    std::size_t offset = 0;
    for (auto & stream : result) for (auto & value : stream) value = static_cast<std::int32_t>(take_u32(source, &offset));
    return result;
}

const levo::checkpoint::section & required_section(const levo::checkpoint::decoded_blob & blob,
                                                    levo::checkpoint::section_kind kind) {
    const levo::checkpoint::section * result = nullptr;
    for (const auto & entry : blob.sections) if (entry.kind == kind) {
        if (result != nullptr) throw std::runtime_error("duplicate checkpoint section");
        result = &entry;
    }
    if (result == nullptr) throw std::runtime_error("checkpoint is missing required section");
    return *result;
}

std::uint64_t resolve_seed() {
    std::random_device random;
    return (static_cast<std::uint64_t>(random()) << 32U) ^ static_cast<std::uint64_t>(random());
}

lm_stamp stamp_from_result(const levo::generation_result & result) {
    return {result.backend_name, result.model_sha256, result.model_artifact_sha256, result.runtime_revision, result.tokenizer_sha256};
}

void validate_stamp(const lm_stamp & expected, const levo::generation_result & actual) {
    const lm_stamp have = stamp_from_result(actual);
    if (expected.backend != have.backend) {
        throw std::runtime_error("cannot resume: blob was paused on backend '" + expected.backend +
                                 "', this engine runs on '" + have.backend + "'");
    }
    if (expected.model_sha256 != have.model_sha256 || expected.artifact_sha256 != have.artifact_sha256 || expected.runtime_revision != have.runtime_revision ||
        expected.tokenizer_sha256 != have.tokenizer_sha256) {
        throw std::runtime_error("cannot resume: LeLM artifact/model/runtime/tokenizer stamp differs from the paused run");
    }
}

void append_size(std::vector<std::uint8_t> & out, std::size_t value, const char * label) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error(std::string(label) + " exceeds checkpoint range");
    }
    append_u64(out, static_cast<std::uint64_t>(value));
}

std::size_t take_size(const std::vector<std::uint8_t> & source, std::size_t * offset, const char * label) {
    const std::uint64_t value = take_u64(source, offset);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(label) + " exceeds this platform");
    }
    return static_cast<std::size_t>(value);
}

struct flow_stamp {
    std::string backend;
    std::string token_sha256;
    std::string model_sha256;
    std::string artifact_sha256;
    std::string vae_artifact_sha256;
    std::string runtime_revision;
};

struct flow_metadata {
    std::size_t source_frames = 0;
    std::size_t padded_frames = 0;
    std::size_t window_count = 0;
    std::size_t euler_steps = 0;
    std::uint64_t seed = 0;
    float cfg_scale = 0.0F;
    std::size_t completed_windows = 0;
    bool active_window = false;
    std::size_t active_window_index = 0;
    std::size_t active_euler_steps = 0;
    flow_stamp stamp;
};

std::vector<std::uint8_t> encode_flow_metadata(const flow_metadata & value) {
    std::vector<std::uint8_t> result;
    result.reserve(8U * 8U + 5U + value.stamp.backend.size() + value.stamp.token_sha256.size() + value.stamp.model_sha256.size() +
                   value.stamp.artifact_sha256.size() + value.stamp.vae_artifact_sha256.size() + value.stamp.runtime_revision.size());
    append_size(result, value.source_frames, "Flow source frame count");
    append_size(result, value.padded_frames, "Flow padded frame count");
    append_size(result, value.window_count, "Flow window count");
    append_size(result, value.euler_steps, "Flow Euler step count");
    append_u64(result, value.seed);
    append_f32(result, value.cfg_scale);
    append_size(result, value.completed_windows, "Flow completed window count");
    result.push_back(value.active_window ? 1U : 0U);
    append_size(result, value.active_window_index, "Flow active window index");
    append_size(result, value.active_euler_steps, "Flow active Euler step count");
    append_string(result, value.stamp.backend);
    append_string(result, value.stamp.token_sha256);
    append_string(result, value.stamp.model_sha256);
    append_string(result, value.stamp.artifact_sha256);
    append_string(result, value.stamp.vae_artifact_sha256);
    append_string(result, value.stamp.runtime_revision);
    return result;
}

flow_metadata decode_flow_metadata(const std::vector<std::uint8_t> & source) {
    flow_metadata result;
    std::size_t offset = 0;
    result.source_frames = take_size(source, &offset, "Flow source frame count");
    result.padded_frames = take_size(source, &offset, "Flow padded frame count");
    result.window_count = take_size(source, &offset, "Flow window count");
    result.euler_steps = take_size(source, &offset, "Flow Euler step count");
    result.seed = take_u64(source, &offset);
    result.cfg_scale = take_f32(source, &offset);
    result.completed_windows = take_size(source, &offset, "Flow completed window count");
    if (offset >= source.size() || source[offset] > 1U) throw std::runtime_error("invalid Flow active-window flag");
    result.active_window = source[offset++] != 0;
    result.active_window_index = take_size(source, &offset, "Flow active window index");
    result.active_euler_steps = take_size(source, &offset, "Flow active Euler step count");
    result.stamp.backend = take_string(source, &offset);
    result.stamp.token_sha256 = take_string(source, &offset);
    result.stamp.model_sha256 = take_string(source, &offset);
    result.stamp.artifact_sha256 = take_string(source, &offset);
    result.stamp.vae_artifact_sha256 = take_string(source, &offset);
    result.stamp.runtime_revision = take_string(source, &offset);
    if (offset != source.size()) throw std::runtime_error("trailing Flow checkpoint metadata bytes");
    if (result.source_frames == 0 || result.window_count == 0 || result.euler_steps == 0 ||
        !std::isfinite(result.cfg_scale) || result.cfg_scale <= 0.0F ||
        result.completed_windows > result.window_count ||
        (result.active_window && result.active_window_index != result.completed_windows) ||
        (!result.active_window && result.active_window_index != 0) ||
        (!result.active_window && result.active_euler_steps != 0)) {
        throw std::runtime_error("Flow checkpoint metadata has invalid schedule state");
    }
    return result;
}

flow_stamp flow_stamp_from(ggml_backend_t backend, const levo::flow::flow_provenance & provenance,
                           const std::string & token_sha256, const std::string & vae_artifact_sha256) {
    return {ggml_backend_name(backend), token_sha256, provenance.model_sha256, provenance.artifact_sha256,
            vae_artifact_sha256, provenance.runtime_revision};
}

void validate_flow_stamp(const flow_stamp & expected, ggml_backend_t backend,
                         const levo::flow::flow_provenance & provenance,
                         const std::string & token_sha256, const std::string & vae_artifact_sha256) {
    const flow_stamp actual = flow_stamp_from(backend, provenance, token_sha256, vae_artifact_sha256);
    if (expected.backend != actual.backend) {
        throw std::runtime_error("cannot resume: blob was paused on backend '" + expected.backend +
                                 "', this engine runs on '" + actual.backend + "'");
    }
    if (expected.token_sha256 != actual.token_sha256 || expected.model_sha256 != actual.model_sha256 ||
        expected.artifact_sha256 != actual.artifact_sha256 ||
        expected.vae_artifact_sha256 != actual.vae_artifact_sha256 ||
        expected.runtime_revision != actual.runtime_revision) {
        throw std::runtime_error("cannot resume: Flow artifact/model/runtime stamp differs from the paused run");
    }
}

cantor_status run_codes(cantor_ctx * context, const std::uint8_t * input, std::size_t input_size,
                        std::uint8_t ** output, std::size_t * output_size,
                        cantor_progress_fn progress, cantor_cancel_fn should_cancel, void * userdata) {
    if (context->lm_path.empty()) {
        set_error(CANTOR_ERR_MODEL, "[LeVo ABI] CODES requires an lm component");
        return CANTOR_ERR;
    }
    if (input_size >= lm_magic_v1.size() &&
        std::equal(lm_magic_v1.begin(), lm_magic_v1.end(), reinterpret_cast<const char *>(input))) {
        throw std::runtime_error("LEVOLM01 checkpoints predate artifact identity; restart CODES from the original request");
    }
    bool resuming = input_size > lm_magic.size() &&
                    std::equal(lm_magic.begin(), lm_magic.end(), reinterpret_cast<const char *>(input));
    levo::engine_request::request request;
    levo::detail::generation_resume_state resume_state;
    lm_stamp expected_stamp;
    std::string canonical_request;
    if (resuming) {
        const auto blob = levo::checkpoint::decode(input, input_size, lm_magic);
        if (blob.stage != CANTOR_STAGE_CODES) throw std::runtime_error("LeLM checkpoint has the wrong Cantor stage");
        if (blob.sections.size() != 3U) throw std::runtime_error("LeLM checkpoint has an unexpected section set");
        const auto & request_section = required_section(blob, levo::checkpoint::section_kind::request_json);
        canonical_request.assign(reinterpret_cast<const char *>(request_section.bytes.data()), request_section.bytes.size());
        request = levo::engine_request::parse(canonical_request);
        if (!request.seed_present) throw std::runtime_error("paused LeLM request does not contain a resolved seed");
        resume_state.delayed_sequence = decode_delayed(required_section(blob, levo::checkpoint::section_kind::delayed_tokens_i32).bytes);
        auto metadata = decode_lm_metadata(required_section(blob, levo::checkpoint::section_kind::metadata).bytes);
        resume_state.sampler_seed = metadata.first.sampler_seed;
        resume_state.sampler_draws = metadata.first.sampler_draws;
        resume_state.ended = metadata.first.ended;
        resume_state.earliest_eos = metadata.first.earliest_eos;
        resume_state.next_logits_digest = metadata.first.next_logits_digest;
        expected_stamp = std::move(metadata.second);
    } else {
        canonical_request.assign(reinterpret_cast<const char *>(input), input_size);
        request = levo::engine_request::parse(canonical_request);
        if (!request.seed_present) {
            request.seed = resolve_seed();
            request.seed_present = true;
        }
        canonical_request = levo::engine_request::serialize(request);
    }

    levo::generation_config config = levo::engine_request::generation_config_for(request, context->lm_path);
    config.cancelled = [should_cancel, userdata] { return cancelled(should_cancel, userdata); };
    const auto report = [progress, userdata](const levo::generation_progress & value) {
        emit_progress(progress, userdata, CANTOR_STAGE_CODES, value.completed_steps, value.total_steps);
    };
    const levo::generation_model_ready_callback verify_resume_stamp = [resuming, &expected_stamp](
        const levo::generation_result & model_stamp) {
        if (resuming) validate_stamp(expected_stamp, model_stamp);
    };
    levo::detail::resumable_generation_result result = levo::generate_tokens_resumable(
        config, resuming ? &resume_state : nullptr, report, verify_resume_stamp);

    if (result.paused) {
        const lm_stamp stamp = stamp_from_result(result.result);
        if (stamp.backend.empty() || stamp.model_sha256.empty() || stamp.artifact_sha256.empty() ||
            stamp.runtime_revision.empty() || stamp.tokenizer_sha256.empty()) {
            throw std::runtime_error("cannot create a LeLM checkpoint without full model provenance");
        }
        const std::vector<levo::checkpoint::section> sections{
            {levo::checkpoint::section_kind::request_json,
             std::vector<std::uint8_t>(canonical_request.begin(), canonical_request.end())},
            {levo::checkpoint::section_kind::delayed_tokens_i32, encode_delayed(result.resume.delayed_sequence)},
            {levo::checkpoint::section_kind::metadata, encode_lm_metadata(result.resume, stamp)},
        };
        const auto blob = levo::checkpoint::encode(lm_magic, CANTOR_STAGE_CODES, sections);
        if (!blob_alloc(blob, output, output_size)) return CANTOR_ERR;
        set_error(CANTOR_ERR_CANCEL, "[LeVo ABI] CODES paused at a delayed-token boundary");
        return CANTOR_PAUSED;
    }

    const std::string complete = levo::engine_request::serialize_codes(request, result.result);
    if (!blob_alloc_string(complete, output, output_size)) return CANTOR_ERR;
    return CANTOR_DONE;
}

std::vector<std::uint8_t> encode_completed_flow_windows(const std::vector<levo::flow::latent_window> & windows,
                                                        std::size_t per_window) {
    if (windows.size() > std::numeric_limits<std::size_t>::max() / per_window) {
        throw std::runtime_error("Flow completed windows exceed checkpoint range");
    }
    std::vector<float> flattened;
    flattened.reserve(windows.size() * per_window);
    for (std::size_t index = 0; index < windows.size(); ++index) {
        const levo::flow::latent_window & window = windows[index];
        if (window.denormalized_latents.size() != per_window) {
            throw std::runtime_error("Flow completed window has an invalid latent shape");
        }
        flattened.insert(flattened.end(), window.denormalized_latents.begin(), window.denormalized_latents.end());
    }
    return encode_f32(flattened, "Flow completed latent windows");
}

std::vector<levo::flow::latent_window> decode_completed_flow_windows(const std::vector<std::uint8_t> & bytes,
                                                                       std::size_t completed_windows,
                                                                       std::size_t per_window,
                                                                       std::size_t hop_frames) {
    if (completed_windows > std::numeric_limits<std::size_t>::max() / per_window) {
        throw std::runtime_error("Flow completed-window count overflows this platform");
    }
    const std::vector<float> flattened = decode_f32(bytes, "Flow completed latent windows");
    if (flattened.size() != completed_windows * per_window) {
        throw std::runtime_error("Flow completed latent payload does not match its window count");
    }
    std::vector<levo::flow::latent_window> result;
    result.reserve(completed_windows);
    for (std::size_t index = 0; index < completed_windows; ++index) {
        levo::flow::latent_window window;
        window.input_offset_frames = index * hop_frames;
        window.denormalized_latents.assign(flattened.begin() + static_cast<std::ptrdiff_t>(index * per_window),
                                           flattened.begin() + static_cast<std::ptrdiff_t>((index + 1U) * per_window));
        result.push_back(std::move(window));
    }
    return result;
}

std::vector<std::vector<std::int64_t>> tokens_as_delayed(const std::vector<std::int32_t> & tokens,
                                                          std::size_t frame_count) {
    if (frame_count == 0 || tokens.size() != 3U * frame_count) {
        throw std::runtime_error("CODES token payload is not [3,T]");
    }
    std::vector<std::vector<std::int64_t>> result(3, std::vector<std::int64_t>(frame_count));
    for (std::size_t stream = 0; stream != 3; ++stream) {
        for (std::size_t frame = 0; frame != frame_count; ++frame) {
            result[stream][frame] = tokens[stream * frame_count + frame];
        }
    }
    return result;
}

std::vector<std::int32_t> delayed_as_tokens(const std::vector<std::vector<std::int64_t>> & delayed) {
    if (delayed.size() != 3 || delayed.front().empty()) throw std::runtime_error("Flow token checkpoint is not [3,T]");
    const std::size_t frames = delayed.front().size();
    std::vector<std::int32_t> result;
    result.reserve(3U * frames);
    for (const auto & stream : delayed) {
        if (stream.size() != frames) throw std::runtime_error("Flow token checkpoint streams differ in length");
        for (const std::int64_t value : stream) {
            if (value < 0 || value > std::numeric_limits<std::int32_t>::max()) {
                throw std::runtime_error("Flow token checkpoint contains an invalid ID");
            }
            result.push_back(static_cast<std::int32_t>(value));
        }
    }
    return result;
}

cantor_status run_flow(cantor_ctx * context, const std::uint8_t * input, std::size_t input_size,
                       std::uint8_t ** output, std::size_t * output_size,
                       cantor_progress_fn progress, cantor_cancel_fn should_cancel, void * userdata) {
    if (context->dit_path.empty()) {
        set_error(CANTOR_ERR_MODEL, "[LeVo ABI] DIFFUSE requires a dit component");
        return CANTOR_ERR;
    }
    if (context->vae_path.empty()) {
        set_error(CANTOR_ERR_MODEL, "[LeVo ABI] DIFFUSE requires a vae component to stamp its durable decode boundary");
        return CANTOR_ERR;
    }
    if (input_size >= flow_magic_v1.size() &&
        std::equal(flow_magic_v1.begin(), flow_magic_v1.end(), reinterpret_cast<const char *>(input))) {
        throw std::runtime_error("LEVOFL01 checkpoints predate artifact identity; restart DIFFUSE from the CODES boundary");
    }
    const bool resuming = input_size > flow_magic.size() &&
        std::equal(flow_magic.begin(), flow_magic.end(), reinterpret_cast<const char *>(input));
    levo::engine_request::request request;
    std::string canonical_request;
    std::vector<std::int32_t> tokens;
    std::vector<float> noise;
    flow_metadata saved_metadata;
    levo::flow::renderer_resume_state saved_resume;
    if (resuming) {
        const auto blob = levo::checkpoint::decode(input, input_size, flow_magic);
        if (blob.stage != CANTOR_STAGE_DIFFUSE || (blob.sections.size() != 5U && blob.sections.size() != 6U)) {
            throw std::runtime_error("Flow checkpoint has an unexpected stage or section set");
        }
        const auto & request_section = required_section(blob, levo::checkpoint::section_kind::request_json);
        canonical_request.assign(reinterpret_cast<const char *>(request_section.bytes.data()), request_section.bytes.size());
        request = levo::engine_request::parse(canonical_request);
        if (!request.seed_present) throw std::runtime_error("paused Flow request does not contain a resolved LeLM seed");
        tokens = delayed_as_tokens(decode_delayed(required_section(blob, levo::checkpoint::section_kind::delayed_tokens_i32).bytes));
        saved_metadata = decode_flow_metadata(required_section(blob, levo::checkpoint::section_kind::metadata).bytes);
        noise = decode_f32(required_section(blob, levo::checkpoint::section_kind::flow_noise_f32).bytes, "Flow checkpoint noise");
        const auto & completed = required_section(blob, levo::checkpoint::section_kind::completed_latents_f32);
        // Split the stored raw windows after Flow hparams have verified their
        // exact window dimensions below. Keeping bytes opaque until then makes
        // malformed claims fail before model execution.
        saved_resume.active_window = saved_metadata.active_window;
        saved_resume.active_window_index = saved_metadata.active_window_index;
        saved_resume.active_euler.completed_steps = saved_metadata.active_euler_steps;
        if (saved_metadata.active_window) {
            if (blob.sections.size() != 6U) throw std::runtime_error("Flow checkpoint is missing active Euler state");
            saved_resume.active_euler.normalized_state = decode_f32(
                required_section(blob, levo::checkpoint::section_kind::euler_state_f32).bytes, "Flow checkpoint Euler state");
        } else if (blob.sections.size() != 5U) {
            throw std::runtime_error("Flow checkpoint has an unexpected Euler state section");
        }
        // Temporarily retain the payload in normalized_state only for the
        // pre-model semantic validation; it is replaced with a real Euler
        // cursor below if this is an active-window checkpoint.
        saved_resume.completed_windows = decode_completed_flow_windows(
            completed.bytes, saved_metadata.completed_windows,
            static_cast<std::size_t>(levo::renderer_window_frames) * 64U,
            static_cast<std::size_t>(levo::renderer_hop_frames));
    } else {
        const std::string codes_json(reinterpret_cast<const char *>(input), input_size);
        const levo::engine_request::codes codes = levo::engine_request::parse_codes(codes_json);
        request = codes.generation_request;
        canonical_request = levo::engine_request::serialize(request);
        tokens = codes.tokens;
    }
    if (tokens.size() % 3U != 0 || tokens.empty()) throw std::runtime_error("Flow input tokens are not [3,T]");
    const std::size_t frame_count = tokens.size() / 3U;
    const std::vector<std::vector<std::int32_t>> streams_by_codebook{
        std::vector<std::int32_t>(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(frame_count)),
        std::vector<std::int32_t>(tokens.begin() + static_cast<std::ptrdiff_t>(frame_count),
                                  tokens.begin() + static_cast<std::ptrdiff_t>(2U * frame_count)),
        std::vector<std::int32_t>(tokens.begin() + static_cast<std::ptrdiff_t>(2U * frame_count), tokens.end()),
    };
    const levo::renderer_token_streams streams = levo::validate_renderer_tokens(streams_by_codebook);
    const levo::renderer_schedule schedule = levo::make_renderer_schedule(frame_count);
    const std::size_t per_window = static_cast<std::size_t>(levo::renderer_window_frames) * 64U;
    if (schedule.windows.size() > std::numeric_limits<std::size_t>::max() / per_window) {
        throw std::runtime_error("Flow schedule noise shape overflows this platform");
    }
    const std::size_t noise_count = schedule.windows.size() * per_window;
    if (!resuming) noise = native_gaussian_noise(noise_count, request.flow_seed);
    if (noise.size() != noise_count) throw std::runtime_error("Flow checkpoint noise has an unexpected shape");

    ggml_backend_dev_t device = select_engine_device();
    levo::detail::configure_cuda_gemm_f32_accumulation(device);
    levo::detail::configure_cuda_disable_tf32(device);
    backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) throw std::runtime_error("cannot initialize Flow backend");
    const std::shared_ptr<levo::flow::model> model = levo::flow::model::load_gguf(
        context->dit_path, {backend.get(), true, true, false});
    const levo::flow::flow_hparams & hp = model->hparams();
    if (hp.window_frames != static_cast<int32_t>(levo::renderer_window_frames) ||
        hp.hop_frames != static_cast<int32_t>(levo::renderer_hop_frames) ||
        hp.overlap_frames != static_cast<int32_t>(levo::renderer_overlap_frames) || hp.latent_dim != 64) {
        throw std::runtime_error("Flow GGUF does not match the native window/latent contract");
    }
    const std::size_t resolved_steps = resuming ? saved_metadata.euler_steps :
        (request.flow_euler_steps == 0 ? static_cast<std::size_t>(hp.euler_steps_default) : request.flow_euler_steps);
    const float resolved_cfg = resuming ? saved_metadata.cfg_scale :
        (request.flow_cfg_scale == 0.0F ? hp.cfg_default : request.flow_cfg_scale);
    if (resolved_steps == 0 || !std::isfinite(resolved_cfg) || resolved_cfg <= 0.0F) {
        throw std::runtime_error("Flow request resolves to invalid Euler settings");
    }
    const std::string token_sha256 = levo::token_io::tensor_sha256(tokens);
    const std::string vae_artifact_sha256 = levo::token_io::file_sha256(context->vae_path);
    if (vae_artifact_sha256.empty()) throw std::runtime_error("cannot calculate VAE artifact digest");
    if (resuming) {
        if (saved_metadata.source_frames != frame_count || saved_metadata.padded_frames != schedule.padded_frames ||
            saved_metadata.window_count != schedule.windows.size() || saved_metadata.completed_windows != saved_resume.completed_windows.size()) {
            throw std::runtime_error("Flow checkpoint schedule does not match its embedded CODES input");
        }
        if (saved_resume.active_window && saved_resume.active_euler.normalized_state.size() != per_window) {
            throw std::runtime_error("Flow checkpoint Euler tensor has an unexpected shape");
        }
        validate_flow_stamp(saved_metadata.stamp, backend.get(), model->provenance(), token_sha256, vae_artifact_sha256);
    }

    levo::flow::render_input flow_input;
    flow_input.vocal_codes = streams.vocal;
    flow_input.bgm_codes = streams.bgm;
    flow_input.initial_noise = noise;
    levo::flow::render_options options;
    options.euler_steps = resolved_steps;
    options.guidance_scale = resolved_cfg;
    options.cancelled = [should_cancel, userdata] { return cancelled(should_cancel, userdata); };
    options.progress = [progress, userdata](std::size_t window, std::size_t windows,
                                            std::size_t completed_steps, std::size_t total_steps) {
        if (window == 0 || total_steps == 0 || windows > std::numeric_limits<std::size_t>::max() / total_steps ||
            window - 1U > std::numeric_limits<std::size_t>::max() / total_steps) {
            throw std::runtime_error("Flow progress counter overflows");
        }
        emit_progress(progress, userdata, CANTOR_STAGE_DIFFUSE,
                      (window - 1U) * total_steps + completed_steps, windows * total_steps);
    };
    levo::flow::renderer renderer(model);
    levo::flow::resumable_render_output result = renderer.render_resumable(
        flow_input, options, resuming ? &saved_resume : nullptr);

    const flow_stamp stamp = flow_stamp_from(backend.get(), model->provenance(), token_sha256, vae_artifact_sha256);
    const auto metadata_for = [&](const std::vector<levo::flow::latent_window> & completed,
                                  bool active, std::size_t active_index, std::size_t active_steps) {
        flow_metadata metadata;
        metadata.source_frames = frame_count;
        metadata.padded_frames = schedule.padded_frames;
        metadata.window_count = schedule.windows.size();
        metadata.euler_steps = resolved_steps;
        metadata.seed = request.flow_seed;
        metadata.cfg_scale = resolved_cfg;
        metadata.completed_windows = completed.size();
        metadata.active_window = active;
        metadata.active_window_index = active ? active_index : 0;
        metadata.active_euler_steps = active ? active_steps : 0;
        metadata.stamp = stamp;
        return metadata;
    };
    const std::vector<std::vector<std::int64_t>> delayed_tokens = tokens_as_delayed(tokens, frame_count);
    if (result.paused) {
        const flow_metadata metadata = metadata_for(result.resume.completed_windows, result.resume.active_window,
            result.resume.active_window_index, result.resume.active_euler.completed_steps);
        std::vector<levo::checkpoint::section> sections{
            {levo::checkpoint::section_kind::request_json, std::vector<std::uint8_t>(canonical_request.begin(), canonical_request.end())},
            {levo::checkpoint::section_kind::delayed_tokens_i32, encode_delayed(delayed_tokens)},
            {levo::checkpoint::section_kind::flow_noise_f32, encode_f32(noise, "Flow initial noise")},
            {levo::checkpoint::section_kind::completed_latents_f32, encode_completed_flow_windows(result.resume.completed_windows, per_window)},
            {levo::checkpoint::section_kind::metadata, encode_flow_metadata(metadata)},
        };
        if (result.resume.active_window) {
            sections.push_back({levo::checkpoint::section_kind::euler_state_f32,
                                encode_f32(result.resume.active_euler.normalized_state, "Flow Euler state")});
        }
        const auto blob = levo::checkpoint::encode(flow_magic, CANTOR_STAGE_DIFFUSE, sections);
        if (!blob_alloc(blob, output, output_size)) return CANTOR_ERR;
        set_error(CANTOR_ERR_CANCEL, "[LeVo ABI] DIFFUSE paused at an Euler/window boundary");
        return CANTOR_PAUSED;
    }
    const flow_metadata metadata = metadata_for(result.output.windows, false, 0, 0);
    const std::vector<levo::checkpoint::section> sections{
        {levo::checkpoint::section_kind::request_json, std::vector<std::uint8_t>(canonical_request.begin(), canonical_request.end())},
        {levo::checkpoint::section_kind::delayed_tokens_i32, encode_delayed(delayed_tokens)},
        {levo::checkpoint::section_kind::completed_latents_f32, encode_completed_flow_windows(result.output.windows, per_window)},
        {levo::checkpoint::section_kind::metadata, encode_flow_metadata(metadata)},
    };
    const auto blob = levo::checkpoint::encode(latent_magic, CANTOR_STAGE_DIFFUSE, sections);
    if (!blob_alloc(blob, output, output_size)) return CANTOR_ERR;
    return CANTOR_DONE;
}

std::vector<float> channel_major_latent(const std::vector<float> & frame_major,
                                        std::size_t frames, std::size_t latent_dim) {
    if (frame_major.size() != frames * latent_dim) throw std::runtime_error("VAE latent window has an unexpected shape");
    std::vector<float> result(frame_major.size());
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t channel = 0; channel < latent_dim; ++channel) {
            result[channel * frames + frame] = frame_major[frame * latent_dim + channel];
        }
    }
    return result;
}

levo::renderer_stereo_audio stereo_from_vae(const levo::detail::vae_decode_result & decoded) {
    if (decoded.samples_per_channel == 0 || decoded.audio.size() != 2U * decoded.samples_per_channel) {
        throw std::runtime_error("VAE decoder returned an unexpected stereo payload");
    }
    levo::renderer_stereo_audio result;
    result.left.assign(decoded.audio.begin(), decoded.audio.begin() + static_cast<std::ptrdiff_t>(decoded.samples_per_channel));
    result.right.assign(decoded.audio.begin() + static_cast<std::ptrdiff_t>(decoded.samples_per_channel), decoded.audio.end());
    return result;
}

cantor_status pause_decode_boundary() {
    set_error(CANTOR_ERR_CANCEL, "[LeVo ABI] DECODE paused; retrying from the durable Flow boundary");
    return CANTOR_PAUSED;
}

cantor_status run_decode(cantor_ctx * context, const std::uint8_t * input, std::size_t input_size,
                         cantor_progress_fn progress, cantor_cancel_fn should_cancel, void * userdata) {
    if (context->dit_path.empty()) {
        set_error(CANTOR_ERR_MODEL, "[LeVo ABI] DECODE requires a dit component to verify its Flow boundary");
        return CANTOR_ERR;
    }
    if (context->vae_path.empty()) {
        set_error(CANTOR_ERR_MODEL, "[LeVo ABI] DECODE requires a vae component");
        return CANTOR_ERR;
    }
    if (input_size >= latent_magic_v1.size() &&
        std::equal(latent_magic_v1.begin(), latent_magic_v1.end(), reinterpret_cast<const char *>(input))) {
        throw std::runtime_error("LEVOLT01 checkpoints predate artifact identity; restart DIFFUSE from the CODES boundary");
    }
    if (input_size <= latent_magic.size() ||
        !std::equal(latent_magic.begin(), latent_magic.end(), reinterpret_cast<const char *>(input))) {
        throw std::runtime_error("DECODE expects a completed LEVOLT02 Flow boundary");
    }
    const auto blob = levo::checkpoint::decode(input, input_size, latent_magic);
    if (blob.stage != CANTOR_STAGE_DIFFUSE || blob.sections.size() != 4U) {
        throw std::runtime_error("completed Flow boundary has an unexpected stage or section set");
    }
    const auto & request_section = required_section(blob, levo::checkpoint::section_kind::request_json);
    const std::string canonical_request(reinterpret_cast<const char *>(request_section.bytes.data()), request_section.bytes.size());
    const levo::engine_request::request request = levo::engine_request::parse(canonical_request);
    if (!request.seed_present) throw std::runtime_error("completed Flow request does not contain a resolved LeLM seed");
    const std::vector<std::int32_t> tokens = delayed_as_tokens(
        decode_delayed(required_section(blob, levo::checkpoint::section_kind::delayed_tokens_i32).bytes));
    if (tokens.empty() || tokens.size() % 3U != 0) throw std::runtime_error("completed Flow tokens are not [3,T]");
    const std::size_t frame_count = tokens.size() / 3U;
    const levo::renderer_schedule schedule = levo::make_renderer_schedule(frame_count);
    const flow_metadata metadata = decode_flow_metadata(required_section(blob, levo::checkpoint::section_kind::metadata).bytes);
    if (metadata.active_window || metadata.completed_windows != metadata.window_count ||
        metadata.source_frames != frame_count || metadata.padded_frames != schedule.padded_frames ||
        metadata.window_count != schedule.windows.size() ||
        metadata.stamp.token_sha256 != levo::token_io::tensor_sha256(tokens)) {
        throw std::runtime_error("completed Flow boundary does not match its embedded CODES input");
    }
    if (cancelled(should_cancel, userdata)) return pause_decode_boundary();

    const std::string flow_artifact_sha256 = levo::token_io::file_sha256(context->dit_path);
    if (flow_artifact_sha256.empty()) throw std::runtime_error("cannot calculate Flow artifact digest");
    if (metadata.stamp.artifact_sha256 != flow_artifact_sha256) {
        throw std::runtime_error("cannot DECODE: Flow artifact stamp differs from the completed Flow boundary");
    }

    ggml_backend_dev_t device = select_engine_device();
    levo::detail::configure_cuda_gemm_f32_accumulation(device);
    levo::detail::configure_cuda_disable_tf32(device);
    backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) throw std::runtime_error("cannot initialize VAE backend");
    const std::shared_ptr<levo::detail::vae_model> model = levo::detail::vae_model::load_gguf(
        context->vae_path, {backend.get(), true, false, true});
    if (model->provenance().artifact_sha256.empty()) {
        throw std::runtime_error("VAE artifact digest is unavailable");
    }
    if (metadata.stamp.vae_artifact_sha256 != model->provenance().artifact_sha256) {
        throw std::runtime_error("cannot DECODE: VAE artifact stamp differs from the completed Flow boundary");
    }
    const levo::detail::vae_hparams & hp = model->hparams();
    if (hp.sample_rate != static_cast<int32_t>(levo::renderer_sample_rate) || hp.audio_channels != 2 ||
        hp.latent_dim != 64 || hp.downsampling_ratio != static_cast<int32_t>(levo::renderer_samples_per_frame)) {
        throw std::runtime_error("VAE GGUF does not match the native Flow renderer contract");
    }
    const std::size_t per_window = static_cast<std::size_t>(levo::renderer_window_frames) *
                                   static_cast<std::size_t>(hp.latent_dim);
    const std::vector<levo::flow::latent_window> windows = decode_completed_flow_windows(
        required_section(blob, levo::checkpoint::section_kind::completed_latents_f32).bytes,
        metadata.completed_windows, per_window, static_cast<std::size_t>(levo::renderer_hop_frames));
    if (cancelled(should_cancel, userdata)) return pause_decode_boundary();
    std::unique_ptr<levo::detail::vae_decoder> decoder = levo::detail::vae_decoder::create(model);
    std::vector<levo::renderer_stereo_audio> decoded;
    decoded.reserve(windows.size());
    for (std::size_t index = 0; index < windows.size(); ++index) {
        if (cancelled(should_cancel, userdata)) return pause_decode_boundary();
        const std::vector<float> latent = channel_major_latent(windows[index].denormalized_latents,
            static_cast<std::size_t>(levo::renderer_window_frames), static_cast<std::size_t>(hp.latent_dim));
        try {
            decoded.push_back(stereo_from_vae(decoder->decode(
                latent, levo::renderer_window_frames, false,
                [should_cancel, userdata] { return cancelled(should_cancel, userdata); })));
        } catch (const levo::operation_cancelled &) {
            return pause_decode_boundary();
        }
        emit_progress(progress, userdata, CANTOR_STAGE_DECODE, index + 1U, windows.size());
    }
    if (cancelled(should_cancel, userdata)) return pause_decode_boundary();
    const levo::renderer_stereo_audio stereo = levo::assemble_renderer_audio(schedule, decoded);
    if (stereo.left.size() != stereo.right.size() || stereo.left.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("decoded audio has an invalid sample count");
    }
    context->audio.clear();
    context->audio.reserve(stereo.left.size() * 2U);
    context->audio.insert(context->audio.end(), stereo.left.begin(), stereo.left.end());
    context->audio.insert(context->audio.end(), stereo.right.begin(), stereo.right.end());
    context->audio_samples = static_cast<int>(stereo.left.size());
    return CANTOR_DONE;
}

} // namespace

extern "C" {

uint32_t cantor_engine_abi_version(void) {
    return CANTOR_ENGINE_ABI;
}

const char * cantor_engine_model(void) {
    return "levo2";
}

const char * cantor_engine_version(void) {
#ifdef LEVO_VERSION
    return LEVO_VERSION;
#else
    return "unknown";
#endif
}

uint32_t cantor_engine_stages(void) {
    return (1U << CANTOR_STAGE_CODES) | (1U << CANTOR_STAGE_DIFFUSE) | (1U << CANTOR_STAGE_DECODE);
}

cantor_error cantor_engine_last_error_code(void) {
    return g_last_error_code;
}

const char * cantor_engine_last_error(void) {
    return g_last_error.c_str();
}

void cantor_engine_free_blob(uint8_t * blob) {
    std::free(blob);
}

cantor_ctx * cantor_engine_load(const cantor_component * components, size_t count, const cantor_load_opts * options) {
    clear_error();
    try {
        if (components == nullptr || count == 0) {
            set_error(CANTOR_ERR_MODEL, "[LeVo ABI] at least one model component is required");
            return nullptr;
        }
        std::unique_ptr<cantor_ctx> context(new cantor_ctx());
        if (options) context->options = *options;
        for (size_t index = 0; index < count; ++index) {
            if (components[index].role == nullptr || components[index].path == nullptr) {
                set_error(CANTOR_ERR_MODEL, "[LeVo ABI] model component role and path are required");
                return nullptr;
            }
            const std::string role = components[index].role;
            const std::string path = components[index].path;
            if (role.empty() || path.empty()) {
                set_error(CANTOR_ERR_MODEL, "[LeVo ABI] model component role and path cannot be empty");
                return nullptr;
            }
            std::string * destination = role == "lm" ? &context->lm_path : role == "dit" ? &context->dit_path :
                                       role == "vae" ? &context->vae_path : nullptr;
            if (destination == nullptr) {
                set_error(CANTOR_ERR_MODEL, "[LeVo ABI] unknown component role '" + role + "'");
                return nullptr;
            }
            if (!destination->empty()) {
                set_error(CANTOR_ERR_MODEL, "[LeVo ABI] duplicate component role '" + role + "'");
                return nullptr;
            }
            *destination = path;
        }
        return context.release();
    } catch (const std::bad_alloc &) {
        set_error(CANTOR_ERR_OOM, "[LeVo ABI] allocation failed while loading engine context");
    } catch (const std::exception & error) {
        set_error(CANTOR_ERR_OTHER, std::string("[LeVo ABI] ") + error.what());
    } catch (...) {
        set_error(CANTOR_ERR_OTHER, "[LeVo ABI] unknown exception while loading engine context");
    }
    return nullptr;
}

void cantor_engine_free(cantor_ctx * context) {
    delete context;
}

cantor_status cantor_engine_run_stage(cantor_ctx * context, cantor_stage stage,
                                      const uint8_t * state_in, size_t input_size,
                                      uint8_t ** state_out, size_t * output_size,
                                      cantor_progress_fn progress, cantor_cancel_fn should_cancel, void * userdata) {
    clear_error();
    if (context == nullptr || state_in == nullptr || input_size == 0 || state_out == nullptr || output_size == nullptr) {
        set_error(CANTOR_ERR_OTHER, "[LeVo ABI] run_stage called with a null or empty argument");
        return CANTOR_ERR;
    }
    *state_out = nullptr;
    *output_size = 0;
    context->audio.clear();
    context->audio_samples = 0;
    if (stage != CANTOR_STAGE_CODES && stage != CANTOR_STAGE_DIFFUSE && stage != CANTOR_STAGE_DECODE) {
        set_error(CANTOR_ERR_OTHER, "[LeVo ABI] requested stage is not implemented");
        return CANTOR_ERR;
    }
    return guard([&] {
        if (stage == CANTOR_STAGE_CODES) {
            return run_codes(context, state_in, input_size, state_out, output_size, progress, should_cancel, userdata);
        }
        if (stage == CANTOR_STAGE_DIFFUSE) {
            return run_flow(context, state_in, input_size, state_out, output_size, progress, should_cancel, userdata);
        }
        return run_decode(context, state_in, input_size, progress, should_cancel, userdata);
    });
}

const float * cantor_engine_audio(cantor_ctx * context, int * n_samples, int * sample_rate) {
    if (n_samples) *n_samples = context ? context->audio_samples : 0;
    if (sample_rate) *sample_rate = context && !context->audio.empty() ? 48000 : 0;
    return context && !context->audio.empty() ? context->audio.data() : nullptr;
}

uint64_t cantor_engine_resident_bytes(cantor_ctx *) {
    return 0;
}

int cantor_engine_resident_modules(cantor_ctx *) {
    return 0;
}

} // extern "C"
