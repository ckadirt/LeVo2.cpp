#include "cantor_engine.h"

#include "levo-checkpoint.h"
#include "levo-engine-request.h"
#include "levo-generator.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
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

constexpr std::array<char, 8> lm_magic{{'L', 'E', 'V', 'O', 'L', 'M', '0', '1'}};

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

std::uint64_t take_u64(const std::vector<std::uint8_t> & source, std::size_t * offset) {
    if (source.size() - *offset < 8) throw std::runtime_error("truncated LeLM checkpoint metadata");
    std::uint64_t result = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) result |= static_cast<std::uint64_t>(source[(*offset)++]) << shift;
    return result;
}

std::uint32_t take_u32(const std::vector<std::uint8_t> & source, std::size_t * offset) {
    if (source.size() - *offset < 4) throw std::runtime_error("truncated LeLM checkpoint metadata");
    std::uint32_t result = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) result |= static_cast<std::uint32_t>(source[(*offset)++]) << shift;
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
    std::string runtime_revision;
    std::string tokenizer_sha256;
};

std::vector<std::uint8_t> encode_lm_metadata(const levo::detail::generation_resume_state & state,
                                             const lm_stamp & stamp) {
    std::vector<std::uint8_t> result;
    result.reserve(3U * 8U + 3U + state.next_logits_digest.size() + 4U * 4U +
                   stamp.backend.size() + stamp.model_sha256.size() + stamp.runtime_revision.size() + stamp.tokenizer_sha256.size());
    append_u64(result, state.sampler_seed);
    append_u64(result, state.sampler_draws);
    append_u64(result, state.earliest_eos);
    for (const bool ended : state.ended) result.push_back(ended ? 1U : 0U);
    result.insert(result.end(), state.next_logits_digest.begin(), state.next_logits_digest.end());
    append_string(result, stamp.backend);
    append_string(result, stamp.model_sha256);
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
    return {result.backend_name, result.model_sha256, result.runtime_revision, result.tokenizer_sha256};
}

void validate_stamp(const lm_stamp & expected, const levo::generation_result & actual) {
    const lm_stamp have = stamp_from_result(actual);
    if (expected.backend != have.backend) {
        throw std::runtime_error("cannot resume: blob was paused on backend '" + expected.backend +
                                 "', this engine runs on '" + have.backend + "'");
    }
    if (expected.model_sha256 != have.model_sha256 || expected.runtime_revision != have.runtime_revision ||
        expected.tokenizer_sha256 != have.tokenizer_sha256) {
        throw std::runtime_error("cannot resume: LeLM model/runtime/tokenizer stamp differs from the paused run");
    }
}

cantor_status run_codes(cantor_ctx * context, const std::uint8_t * input, std::size_t input_size,
                        std::uint8_t ** output, std::size_t * output_size,
                        cantor_progress_fn progress, cantor_cancel_fn should_cancel, void * userdata) {
    if (context->lm_path.empty()) {
        set_error(CANTOR_ERR_MODEL, "[LeVo ABI] CODES requires an lm component");
        return CANTOR_ERR;
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
    levo::detail::resumable_generation_result result = levo::generate_tokens_resumable(
        config, resuming ? &resume_state : nullptr, report);
    if (resuming) validate_stamp(expected_stamp, result.result);

    if (result.paused) {
        const lm_stamp stamp = stamp_from_result(result.result);
        if (stamp.backend.empty() || stamp.model_sha256.empty() || stamp.runtime_revision.empty() || stamp.tokenizer_sha256.empty()) {
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
    return 1U << CANTOR_STAGE_CODES;
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
    if (stage != CANTOR_STAGE_CODES) {
        set_error(CANTOR_ERR_OTHER, "[LeVo ABI] requested stage is not implemented");
        return CANTOR_ERR;
    }
    return guard([&] { return run_codes(context, state_in, input_size, state_out, output_size, progress, should_cancel, userdata); });
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
