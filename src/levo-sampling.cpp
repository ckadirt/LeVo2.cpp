#include "levo-sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace levo {
namespace {

double manual_uniform(std::mt19937_64 & rng) {
    // 53 random bits gives the same precision as a conventional double. The
    // half-unit offset avoids ever returning exactly zero or one.
    constexpr double denom = 9007199254740992.0; // 2^53
    return (static_cast<double>(rng() >> 11) + 0.5) / denom;
}

} // namespace

logits apply_cfg(const logits & conditional, const logits & unconditional, float scale) {
    if (conditional.size() != unconditional.size()) throw std::invalid_argument("CFG logits have different sizes");
    logits out(conditional.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = unconditional[i] + scale * (conditional[i] - unconditional[i]);
    return out;
}

std::vector<logits> apply_cfg(const std::vector<logits> & conditional,
                              const std::vector<logits> & unconditional, float scale) {
    if (conditional.size() != unconditional.size()) throw std::invalid_argument("CFG batch sizes differ");
    std::vector<logits> out;
    out.reserve(conditional.size());
    for (std::size_t i = 0; i < conditional.size(); ++i) out.push_back(apply_cfg(conditional[i], unconditional[i], scale));
    return out;
}

void apply_unique_token_repetition_penalty(logits & stream_logits,
                                           const std::vector<int64_t> & recent_tokens,
                                           float penalty, std::size_t vocabulary_limit) {
    if (!(penalty > 0.0F) || !std::isfinite(penalty)) throw std::invalid_argument("repetition penalty must be finite and positive");
    const std::size_t limit = vocabulary_limit == 0 ? stream_logits.size() : std::min(vocabulary_limit, stream_logits.size());
    std::unordered_set<int64_t> unique;
    for (const auto token : recent_tokens) if (token >= 0 && static_cast<std::size_t>(token) < limit) unique.insert(token);
    for (const auto token : unique) stream_logits[static_cast<std::size_t>(token)] /= penalty;
}

logits apply_unique_token_repetition_penalty(const logits & stream_logits,
                                             const std::vector<int64_t> & recent_tokens,
                                             float penalty, std::size_t vocabulary_limit) {
    logits out = stream_logits;
    apply_unique_token_repetition_penalty(out, recent_tokens, penalty, vocabulary_limit);
    return out;
}

Sampler::Sampler(uint64_t seed) : engine_(seed) {}
void Sampler::seed(uint64_t seed) { engine_.seed(seed); }
uint64_t Sampler::uniform_bits() { return engine_(); }
double Sampler::uniform01() { return manual_uniform(engine_); }

int64_t sample_top_k(const logits & input, std::size_t top_k, float temperature,
                     bool use_sampling, std::mt19937_64 & rng) {
    if (input.empty()) throw std::invalid_argument("cannot sample an empty vocabulary");
    if (!std::isfinite(temperature) || temperature < 0.0F) throw std::invalid_argument("temperature must be finite and non-negative");
    std::vector<std::size_t> candidates(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (std::isnan(input[i])) throw std::invalid_argument("logits contain NaN");
        candidates[i] = i;
    }
    if (top_k > 0 && top_k < candidates.size()) {
        std::stable_sort(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) {
            return input[a] > input[b];
        });
        candidates.resize(top_k);
    }
    auto best = *std::max_element(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) {
        return input[a] < input[b];
    });
    if (!use_sampling || temperature == 0.0F) return static_cast<int64_t>(best);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (auto i : candidates) max_logit = std::max(max_logit, input[i]);
    std::vector<double> weights(candidates.size());
    double total = 0.0;
    for (std::size_t n = 0; n < candidates.size(); ++n) {
        const float value = input[candidates[n]];
        if (std::isinf(value) && value < 0) weights[n] = 0.0;
        else weights[n] = std::exp(static_cast<double>((value - max_logit) / temperature));
        total += weights[n];
    }
    if (!(total > 0.0) || !std::isfinite(total)) return static_cast<int64_t>(best);
    const double target = manual_uniform(rng) * total;
    double cumulative = 0.0;
    for (std::size_t n = 0; n < weights.size(); ++n) {
        cumulative += weights[n];
        if (target < cumulative) return static_cast<int64_t>(candidates[n]);
    }
    return static_cast<int64_t>(candidates.back());
}

int64_t Sampler::sample(const logits & input, std::size_t top_k, float temperature, bool use_sampling) {
    return sample_top_k(input, top_k, temperature, use_sampling, engine_);
}

std::vector<int64_t> Sampler::sample_streams(const std::vector<logits> & streams,
                                             const sampling_config & config,
                                             const std::vector<std::vector<int64_t>> & recent) {
    if (!recent.empty() && recent.size() != streams.size()) throw std::invalid_argument("recent-token stream count differs");
    std::vector<int64_t> out;
    out.reserve(streams.size());
    for (std::size_t q = 0; q < streams.size(); ++q) {
        logits adjusted = streams[q];
        if (!recent.empty()) {
            const std::size_t begin = recent[q].size() > config.repetition_window ? recent[q].size() - config.repetition_window : 0;
            apply_unique_token_repetition_penalty(adjusted,
                std::vector<int64_t>(recent[q].begin() + static_cast<std::ptrdiff_t>(begin), recent[q].end()),
                config.repetition_penalty);
        }
        // Upstream applies prompt-token exclusion only to the mixed stream;
        // detail streams use top-k=1 without that ignore list.
        if (q == 0) for (const auto token : config.ignore_tokens)
            if (token >= 0 && static_cast<std::size_t>(token) < adjusted.size()) adjusted[static_cast<std::size_t>(token)] = -std::numeric_limits<float>::infinity();
        const std::size_t top_k = q == 0 ? config.top_k_mixed : config.top_k_detail;
        out.push_back(sample(adjusted, top_k, config.temperature, config.use_sampling));
    }
    return out;
}

eos_tracker::eos_tracker(std::size_t streams, int64_t eos_id) : eos_id_(eos_id), ended_(streams, 0) {
    if (streams == 0) throw std::invalid_argument("EOS tracker needs at least one stream");
}
void eos_tracker::update(const std::vector<int64_t> & next) {
    if (next.size() != ended_.size()) throw std::invalid_argument("EOS update stream count differs");
    ++steps_;
    for (std::size_t q = 0; q < next.size(); ++q) if (!ended_[q] && next[q] == eos_id_) {
        ended_[q] = 1;
        if (first_end_ == SIZE_MAX) first_end_ = steps_ - 1;
    }
}
bool eos_tracker::ended(std::size_t stream) const {
    if (stream >= ended_.size()) throw std::out_of_range("EOS stream index out of range");
    return ended_[stream] != 0;
}
bool eos_tracker::all_ended() const noexcept {
    return std::all_of(ended_.begin(), ended_.end(), [](uint8_t x) { return x != 0; });
}
std::size_t eos_tracker::first_ended_length() const noexcept { return first_end_ == SIZE_MAX ? steps_ : first_end_; }

std::size_t trim_length_at_eos(const std::vector<std::vector<int64_t>> & streams, int64_t eos_id) {
    if (streams.empty()) return 0;
    const std::size_t generated = streams.front().size();
    std::size_t length = generated;
    for (const auto & stream : streams) {
        if (stream.size() != generated) throw std::invalid_argument("EOS streams have different lengths");
        const auto it = std::find(stream.begin(), stream.end(), eos_id);
        if (it != stream.end()) length = std::min(length, static_cast<std::size_t>(it - stream.begin()));
    }
    return length;
}

} // namespace levo
