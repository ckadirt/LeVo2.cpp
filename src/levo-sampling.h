#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace levo {

using logits = std::vector<float>;

// Classifier-free guidance as used by LeVo: uncond + scale * (cond-uncond).
logits apply_cfg(const logits & conditional, const logits & unconditional, float scale);
std::vector<logits> apply_cfg(const std::vector<logits> & conditional,
                              const std::vector<logits> & unconditional, float scale);

struct sampling_config {
    bool use_sampling = true;
    float temperature = 0.9F;
    std::size_t top_k_mixed = 50;
    std::size_t top_k_detail = 1;
    std::size_t repetition_window = 50;
    float repetition_penalty = 1.1F;
    std::vector<int64_t> ignore_tokens;
};

// Applies the exact upstream CodecLM behavior: each distinct token in the
// trailing window is divided once, regardless of how many times it occurred.
void apply_unique_token_repetition_penalty(logits & stream_logits,
                                           const std::vector<int64_t> & recent_tokens,
                                           float penalty = 1.1F,
                                           std::size_t vocabulary_limit = 0);
logits apply_unique_token_repetition_penalty(const logits & stream_logits,
                                             const std::vector<int64_t> & recent_tokens,
                                             float penalty = 1.1F,
                                             std::size_t vocabulary_limit = 0);

class Sampler {
public:
    explicit Sampler(uint64_t seed = 0);
    void seed(uint64_t seed);
    uint64_t uniform_bits();
    double uniform01();

    int64_t sample(const logits & input, std::size_t top_k = 0,
                   float temperature = 1.0F, bool use_sampling = true);
    std::vector<int64_t> sample_streams(const std::vector<logits> & streams,
                                         const sampling_config & config = {},
                                         const std::vector<std::vector<int64_t>> & recent = {});

private:
    std::mt19937_64 engine_;
};

// Stateless convenience entry point with explicit RNG, useful for tests.
int64_t sample_top_k(const logits & input, std::size_t top_k, float temperature,
                     bool use_sampling, std::mt19937_64 & rng);

struct eos_tracker {
    explicit eos_tracker(std::size_t streams = 3, int64_t eos_id = 16384);
    void update(const std::vector<int64_t> & next);
    bool ended(std::size_t stream) const;
    bool all_ended() const noexcept;
    std::size_t first_ended_length() const noexcept;
    const std::vector<uint8_t> & states() const noexcept { return ended_; }
    int64_t eos_id() const noexcept { return eos_id_; }

private:
    int64_t eos_id_;
    std::vector<uint8_t> ended_;
    std::size_t steps_ = 0;
    std::size_t first_end_ = SIZE_MAX;
};

// Returns the exclusive output length at the earliest EOS. This is the index
// of EOS (the EOS frame itself is omitted), matching CodecLM's `[..., :length]`.
std::size_t trim_length_at_eos(const std::vector<std::vector<int64_t>> & streams,
                               int64_t eos_id);

} // namespace levo
