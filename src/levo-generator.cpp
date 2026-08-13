#include "levo-generator.h"

#include "levo-conditioner.h"
#include "levo-kv.h"
#include "levo-pattern.h"
#include "levo-token-io.h"
#include "levo-tokenizer.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace levo {
namespace {

struct backend_deleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo generation: " + message);
}

bool is_gpu_device(ggml_backend_dev_t device) {
    const auto type = ggml_backend_dev_type(device);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU ||
           type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

bool is_cuda_device(ggml_backend_dev_t device) {
    const std::string name = ggml_backend_dev_name(device);
    return is_gpu_device(device) && name.rfind("CUDA", 0) == 0;
}

void configure_cuda_compute_type(ggml_backend_dev_t device) {
    if (!is_cuda_device(device) || std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") != nullptr) {
        return;
    }
#if defined(_WIN32)
    if (_putenv_s("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32") != 0) {
        fail("could not select FP32 CUDA GEMM accumulation");
    }
#else
    if (::setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32", 0) != 0) {
        fail("could not select FP32 CUDA GEMM accumulation");
    }
#endif
}

ggml_backend_dev_t select_device(backend_kind requested, int device_index) {
    if (device_index < 0) {
        throw std::invalid_argument("device index cannot be negative");
    }
    ggml_backend_load_all();

    const auto nth = [device_index](const auto & predicate) -> ggml_backend_dev_t {
        int match = 0;
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (predicate(device) && match++ == device_index) {
                return device;
            }
        }
        return nullptr;
    };
    if (requested == backend_kind::auto_select) {
        if (ggml_backend_dev_t device = nth([](ggml_backend_dev_t value) { return is_cuda_device(value); })) {
            return device;
        }
        if (ggml_backend_dev_t device = nth([](ggml_backend_dev_t value) { return is_gpu_device(value); })) {
            return device;
        }
        if (ggml_backend_dev_t device = nth([](ggml_backend_dev_t value) {
                return ggml_backend_dev_type(value) == GGML_BACKEND_DEVICE_TYPE_CPU;
            })) {
            return device;
        }
        fail("GGML did not register a usable CPU or GPU backend at the requested device index");
    }

    const auto matches_requested = [requested](ggml_backend_dev_t device) {
        if (requested == backend_kind::cuda) return is_cuda_device(device);
        if (requested == backend_kind::gpu) return is_gpu_device(device);
        return ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
    };
    if (ggml_backend_dev_t device = nth(matches_requested)) {
        return device;
    }
    const char * const wanted = requested == backend_kind::cuda ? "CUDA" :
                                requested == backend_kind::gpu ? "GPU" : "CPU";
    fail(std::string("requested ") + wanted + " backend device is unavailable");
}

sampling_config to_sampling_config(const generation_sampling_config & config) {
    sampling_config result;
    result.use_sampling = config.use_sampling;
    result.temperature = config.temperature;
    result.top_k_mixed = config.top_k_mixed;
    result.top_k_detail = config.top_k_detail;
    result.repetition_window = config.repetition_window;
    result.repetition_penalty = config.repetition_penalty;
    result.ignore_tokens = config.ignore_tokens;
    return result;
}

std::size_t requested_frames(const detail::model_hparams & hparams,
                             const generation_config & config) {
    if (!std::isfinite(config.duration_seconds) || config.duration_seconds <= 0.0 ||
        config.duration_seconds > 270.0) {
        throw std::invalid_argument("duration must be finite and satisfy 0 < seconds <= 270");
    }
    if (hparams.frame_rate <= 0) {
        fail("model has a non-positive audio frame rate");
    }
    const double raw = std::floor(config.duration_seconds * static_cast<double>(hparams.frame_rate));
    if (raw < 1.0 || raw > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("duration does not produce a positive representable audio-token frame count");
    }
    return static_cast<std::size_t>(raw);
}

void validate_duration_bounds(const generation_config & config) {
    if (!std::isfinite(config.duration_seconds) || config.duration_seconds <= 0.0 ||
        config.duration_seconds > 270.0) {
        throw std::invalid_argument("duration must be finite and satisfy 0 < seconds <= 270");
    }
}

detail::generation_logits cfg_logits_from_outputs(const detail::lm_output & conditional,
                                                   const detail::lm_output & unconditional,
                                                   float scale) {
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("CFG scale must be finite");
    }
    if (conditional.token_count <= 0 || conditional.token_count != unconditional.token_count ||
        conditional.vocabulary_size <= 0 || conditional.vocabulary_size != unconditional.vocabulary_size) {
        fail("conditional and null CFG outputs have incompatible shapes");
    }
    const std::size_t vocabulary = static_cast<std::size_t>(conditional.vocabulary_size);
    const std::size_t offset = static_cast<std::size_t>(conditional.token_count - 1) * vocabulary;
    const auto extract = [vocabulary, offset](const std::vector<float> & values) {
        if (values.size() < offset + vocabulary) {
            fail("LM output has an invalid logit payload");
        }
        return logits(values.begin() + static_cast<std::ptrdiff_t>(offset),
                      values.begin() + static_cast<std::ptrdiff_t>(offset + vocabulary));
    };
    return {
        apply_cfg(extract(conditional.mixed_logits), extract(unconditional.mixed_logits), scale),
        apply_cfg(extract(conditional.vocal_logits), extract(unconditional.vocal_logits), scale),
        apply_cfg(extract(conditional.bgm_logits), extract(unconditional.bgm_logits), scale),
    };
}

std::array<bool, 3> ended_states(const eos_tracker & tracker) {
    return {{tracker.ended(0), tracker.ended(1), tracker.ended(2)}};
}

using progress_clock = std::chrono::steady_clock;

double elapsed_seconds(progress_clock::time_point begin, progress_clock::time_point end = progress_clock::now()) {
    return std::chrono::duration<double>(end - begin).count();
}

void check_cancelled(const cancellation_callback & cancelled) {
    if (cancelled && cancelled()) throw operation_cancelled();
}

} // namespace

namespace detail {

generation_result run_generation_controller(
    const model_hparams & hparams,
    const generation_config & config,
    generation_logits current_logits,
    generation_advance_callback advance,
    generation_progress_callback progress) {
    if (hparams.codebook_count != 3 || hparams.delays.size() != 3) {
        fail("v0.1 generation requires exactly three delayed codebook streams");
    }
    if (hparams.context_length <= 0 ||
        std::any_of(hparams.delays.begin(), hparams.delays.end(), [](int32_t delay) { return delay < 0; })) {
        fail("model has invalid context or delayed-pattern metadata");
    }
    if (hparams.special_token_id != hparams.codebook_size + 1 ||
        hparams.eos_token_id != hparams.codebook_size) {
        fail("model token IDs do not have the required [codebook, EOS, special] layout");
    }
    if (!std::isfinite(config.cfg_scale)) {
        throw std::invalid_argument("CFG scale must be finite");
    }
    const std::size_t frames = requested_frames(hparams, config);
    const std::vector<std::size_t> delays = {
        static_cast<std::size_t>(hparams.delays[0]),
        static_cast<std::size_t>(hparams.delays[1]),
        static_cast<std::size_t>(hparams.delays[2]),
    };
    const pattern delayed = make_delayed_pattern(3, frames, delays);
    const std::size_t prefix = static_cast<std::size_t>(hparams.lyrics_prefix_length) +
                               static_cast<std::size_t>(hparams.prompt_prefix_length) +
                               static_cast<std::size_t>(hparams.style_prefix_length);
    if (prefix > static_cast<std::size_t>(hparams.context_length) ||
        delayed.sequence_steps() > static_cast<std::size_t>(hparams.context_length) - prefix) {
        throw std::invalid_argument("requested duration and conditioning prefix exceed the model context length");
    }
    for (const logits & values : current_logits) {
        if (values.size() != static_cast<std::size_t>(hparams.token_output_size())) {
            fail("initial logit vocabulary does not match the model");
        }
    }
    if (!advance) {
        throw std::invalid_argument("generation controller requires an advance callback");
    }

    // Build supplies the upstream scatter mask for each delayed sequence
    // position. The placeholder payload is never read.
    const std::vector<std::vector<int64_t>> placeholder(
        3, std::vector<int64_t>(frames, hparams.special_token_id));
    const pattern_result delayed_mask = delayed.build(placeholder, hparams.special_token_id);
    std::vector<std::vector<int64_t>> sequence(
        3, std::vector<int64_t>(1, hparams.special_token_id));
    std::random_device entropy;
    const uint64_t sampler_seed = config.seed_present
        ? config.seed
        : (static_cast<uint64_t>(entropy()) << 32U) ^ static_cast<uint64_t>(entropy());
    Sampler sampler(sampler_seed);
    const sampling_config sampler_config = to_sampling_config(config.sampling);
    eos_tracker eos(3, hparams.eos_token_id);
    const std::size_t total_steps = delayed.sequence_steps() - 1;

    if (progress) {
        generation_progress value;
        value.stage = generation_stage::generating;
        value.total_steps = total_steps;
        value.requested_frames = frames;
        progress(value);
    }

    for (std::size_t step = 1; step < delayed.sequence_steps(); ++step) {
        check_cancelled(config.cancelled);
        // The upstream order matters: sample every head from the same CFG
        // logits first, then scatter-mask invalid delayed positions. History
        // records those masked special values for the next repetition window.
        std::vector<int64_t> next = sampler.sample_streams(
            {current_logits[0], current_logits[1], current_logits[2]}, sampler_config, sequence);
        for (std::size_t stream = 0; stream < 3; ++stream) {
            if (delayed_mask.mask[stream][step] == 0 || eos.ended(stream)) {
                next[stream] = hparams.special_token_id;
            }
        }
        eos.update(next);
        std::array<int32_t, 3> delayed_input{};
        for (std::size_t stream = 0; stream < 3; ++stream) {
            sequence[stream].push_back(next[stream]);
            delayed_input[stream] = static_cast<int32_t>(next[stream]);
        }
        if (progress) {
            generation_progress value;
            value.stage = generation_stage::generating;
            value.completed_steps = step;
            value.total_steps = total_steps;
            value.requested_frames = frames;
            value.ended = ended_states(eos);
            progress(value);
        }

        // Once every stream has an EOS, no later value can affect the final
        // tensor (which is trimmed before the earliest EOS). Avoid needless
        // transformer steps while preserving the exact emitted prefix.
        if (eos.all_ended() || step + 1 == delayed.sequence_steps()) {
            break;
        }
        current_logits = advance(delayed_input);
        for (const logits & values : current_logits) {
            if (values.size() != static_cast<std::size_t>(hparams.token_output_size())) {
                fail("advance callback returned a logit vocabulary incompatible with the model");
            }
        }
    }

    pattern_result reverted = delayed.revert(sequence, hparams.special_token_id);
    const std::size_t trim = trim_length_at_eos(reverted.values, hparams.eos_token_id);
    generation_result result;
    result.requested_frames = frames;
    result.frame_count = trim;
    result.sequence_steps = sequence.front().size();
    result.ended = ended_states(eos);
    result.tokens.reserve(3 * trim);
    for (std::size_t stream = 0; stream < 3; ++stream) {
        result.tokens.insert(result.tokens.end(), reverted.values[stream].begin(),
                             reverted.values[stream].begin() + static_cast<std::ptrdiff_t>(trim));
    }
    return result;
}

} // namespace detail

generation_result generate_tokens(const generation_config & config,
                                  generation_progress_callback progress) {
    if (config.model_path.empty()) {
        throw std::invalid_argument("a GGUF model path is required");
    }
    validate_duration_bounds(config);
    const auto total_started = progress_clock::now();
    auto stage_started = total_started;
    generation_stage current_stage = generation_stage::initializing_backend;
    const auto emit = [&](generation_progress value = {}) {
        const auto now = progress_clock::now();
        value.stage = current_stage;
        value.elapsed_seconds = elapsed_seconds(total_started, now);
        value.stage_elapsed_seconds = elapsed_seconds(stage_started, now);
        if (progress) progress(value);
        check_cancelled(config.cancelled);
    };
    const auto begin_stage = [&](generation_stage stage) {
        current_stage = stage;
        stage_started = progress_clock::now();
        emit();
    };

    generation_timings timings;
    begin_stage(generation_stage::initializing_backend);
    ggml_backend_dev_t device = select_device(config.backend, config.device_index);
    // The released PyTorch model accumulates F16 GEMMs in FP32. GGML exposes
    // the matching upstream cuBLAS path through this environment switch. Its
    // default F16 accumulation is materially different for this near-tied
    // first-token logit, so use the parity-safe mode unless the caller has
    // explicitly selected another GGML compute type.
    configure_cuda_compute_type(device);
    backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) {
        fail("failed to initialize the requested GGML backend");
    }
    timings.backend_seconds = elapsed_seconds(stage_started);

    begin_stage(generation_stage::loading_model);
    detail::model_load_options load_options;
    load_options.backend = backend.get();
    const std::shared_ptr<detail::model> model = detail::model::load_gguf(
        config.model_path.string(), load_options);
    timings.model_load_seconds = elapsed_seconds(stage_started);

    begin_stage(generation_stage::preparing_conditioning);
    const ByteLevelBPETokenizer tokenizer = ByteLevelBPETokenizer::load_embedded(
        model->tokenizer().tokens, model->tokenizer().merges,
        model->tokenizer().added_tokens_json, model->tokenizer().config_json);
    const detail::conditioning_result conditioning = detail::prepare_conditioning(
        *model, tokenizer, config.lyrics, config.description);
    timings.conditioning_seconds = elapsed_seconds(stage_started);

    begin_stage(generation_stage::prefilling);
    std::unique_ptr<detail::kv_session> conditional = detail::kv_session::create(model, true);
    std::unique_ptr<detail::kv_session> null_branch = detail::kv_session::create(model, true);
    // Populate each branch's dense prefix, then decode delayed slot zero (the
    // all-special BOS) to predict slot one. Upstream fuses both into its first
    // streaming graph; the split is mathematically equivalent and matches its
    // cache state numerically, while a combined GGML graph uses different
    // CUDA kernels and perturbs near-tied detail logits after cached steps.
    conditional->prefill_conditioned_prefix(conditioning.conditional);
    null_branch->prefill_conditioned_prefix(conditioning.null_condition);
    const int32_t special = model->hparams().special_token_id;
    const detail::lm_output conditional_initial = conditional->decode(special, special, special);
    const detail::lm_output null_initial = null_branch->decode(special, special, special);
    const detail::generation_logits initial_logits = cfg_logits_from_outputs(
        conditional_initial, null_initial, config.cfg_scale);
    timings.prefill_seconds = elapsed_seconds(stage_started);

    begin_stage(generation_stage::generating);
    const auto generation_started = stage_started;
    generation_progress latest_generation;
    const generation_progress_callback timed_progress = [&](const generation_progress & source) {
        generation_progress value = source;
        const auto now = progress_clock::now();
        value.stage = generation_stage::generating;
        value.elapsed_seconds = elapsed_seconds(total_started, now);
        value.stage_elapsed_seconds = elapsed_seconds(generation_started, now);
        latest_generation = value;
        if (progress) progress(value);
        check_cancelled(config.cancelled);
    };
    generation_result result = detail::run_generation_controller(
        model->hparams(), config, initial_logits,
        [&conditional, &null_branch, scale = config.cfg_scale](const std::array<int32_t, 3> & input) {
            return cfg_logits_from_outputs(conditional->decode(input[0], input[1], input[2]),
                                           null_branch->decode(input[0], input[1], input[2]), scale);
        },
        timed_progress);
    timings.generation_seconds = elapsed_seconds(generation_started);
    result.backend_name = ggml_backend_name(backend.get());
    result.model_name = model->provenance().name;
    result.model_revision = model->provenance().model_revision;
    result.model_sha256 = model->provenance().model_sha256;
    result.runtime_revision = model->provenance().runtime_revision;
    result.tokenizer_revision = model->provenance().tokenizer_revision;
    result.tokenizer_sha256 = model->provenance().tokenizer_sha256;
    timings.total_seconds = elapsed_seconds(total_started);
    result.timings = timings;

    current_stage = generation_stage::complete;
    stage_started = progress_clock::now();
    generation_progress complete = latest_generation;
    complete.requested_frames = result.requested_frames;
    complete.ended = result.ended;
    emit(complete);
    return result;
}

void write_generation_artifact(const std::filesystem::path & output_path,
                               const generation_result & result,
                               const generation_config & config) {
    if (result.tokens.size() != 3 * result.frame_count) {
        throw std::invalid_argument("generation result does not contain a [3,T] token tensor");
    }
    token_io::artifact_metadata metadata;
    metadata.model_name = result.model_name.empty() ? config.model_path.filename().string() : result.model_name;
    metadata.model_revision = result.model_revision;
    metadata.model_sha256 = result.model_sha256;
    metadata.generator = "levo.cpp";
    metadata.generator_revision = version();
    metadata.runtime_revision = result.runtime_revision;
    metadata.tokenizer_revision = result.tokenizer_revision;
    metadata.tokenizer_sha256 = result.tokenizer_sha256;
    metadata.backend_name = result.backend_name;
    metadata.lyrics = config.lyrics;
    metadata.description = config.description;
    metadata.duration_seconds = config.duration_seconds;
    metadata.sampling = to_sampling_config(config.sampling);
    metadata.cfg_scale = config.cfg_scale;
    metadata.seed_present = config.seed_present;
    metadata.seed = config.seed;
    metadata.backend_seconds = result.timings.backend_seconds;
    metadata.model_load_seconds = result.timings.model_load_seconds;
    metadata.conditioning_seconds = result.timings.conditioning_seconds;
    metadata.prefill_seconds = result.timings.prefill_seconds;
    metadata.generation_seconds = result.timings.generation_seconds;
    metadata.total_seconds = result.timings.total_seconds;
    token_io::write(output_path, result.tokens, metadata);
}

} // namespace levo
