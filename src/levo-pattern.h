#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace levo {

struct pattern_coord {
    std::size_t timestep = 0;
    std::size_t codebook = 0;
    bool operator==(const pattern_coord & other) const noexcept {
        return timestep == other.timestep && codebook == other.codebook;
    }
};

struct pattern_result {
    // Stream-major [codebook][sequence step] values and validity mask.
    std::vector<std::vector<int64_t>> values;
    std::vector<std::vector<uint8_t>> mask;
    // Flattened source indexes, [codebook][sequence step].  The sentinel is
    // code_depth * timesteps for build and code_depth * sequence_steps for
    // revert, matching the upstream scatter implementation.
    std::vector<std::vector<std::size_t>> indexes;
};

class pattern {
public:
    pattern() = default;
    pattern(std::size_t code_depth, std::size_t timesteps,
            std::vector<std::vector<pattern_coord>> layout);

    std::size_t code_depth() const noexcept { return code_depth_; }
    std::size_t timesteps() const noexcept { return timesteps_; }
    std::size_t sequence_steps() const noexcept { return layout_.size(); }
    std::size_t max_delay() const noexcept;
    const std::vector<std::vector<pattern_coord>> & layout() const noexcept { return layout_; }

    // The layout up to the last fully defined timestep (upstream valid_layout).
    std::vector<std::vector<pattern_coord>> valid_layout() const;
    std::vector<std::size_t> steps_with_timestep(std::size_t timestep,
                                                  std::size_t codebook = SIZE_MAX) const;
    std::size_t first_step_with_timestep(std::size_t timestep,
                                         std::size_t codebook = SIZE_MAX) const;

    pattern_result build(const std::vector<std::vector<int64_t>> & input,
                         int64_t special_token,
                         bool keep_only_valid_steps = false) const;
    pattern_result revert(const std::vector<std::vector<int64_t>> & sequence,
                          int64_t special_token,
                          bool keep_only_valid_steps = false) const;

private:
    std::size_t code_depth_ = 0;
    std::size_t timesteps_ = 0;
    std::vector<std::vector<pattern_coord>> layout_;
    void validate() const;
};

class DelayedPatternProvider {
public:
    explicit DelayedPatternProvider(std::size_t code_depth,
                                     std::vector<std::size_t> delays = {},
                                     std::size_t flatten_first = 0,
                                     std::size_t empty_initial = 0);
    pattern get_pattern(std::size_t timesteps) const;
    const std::vector<std::size_t> & delays() const noexcept { return delays_; }
    std::size_t code_depth() const noexcept { return code_depth_; }

private:
    std::size_t code_depth_;
    std::vector<std::size_t> delays_;
    std::size_t flatten_first_;
    std::size_t empty_initial_;
};

// Convenience free functions useful to callers that do not need a provider.
pattern make_delayed_pattern(std::size_t code_depth, std::size_t timesteps,
                             const std::vector<std::size_t> & delays = {},
                             std::size_t flatten_first = 0,
                             std::size_t empty_initial = 0);
pattern_result build_delayed_pattern(const std::vector<std::vector<int64_t>> & input,
                                     int64_t special_token,
                                     const std::vector<std::size_t> & delays = {},
                                     std::size_t flatten_first = 0,
                                     std::size_t empty_initial = 0,
                                     bool keep_only_valid_steps = false);
pattern_result revert_delayed_pattern(const std::vector<std::vector<int64_t>> & sequence,
                                      std::size_t timesteps, int64_t special_token,
                                      const std::vector<std::size_t> & delays = {},
                                      std::size_t flatten_first = 0,
                                      std::size_t empty_initial = 0,
                                      bool keep_only_valid_steps = false);

} // namespace levo
