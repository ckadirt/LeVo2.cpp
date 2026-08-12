#include "levo-pattern.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace levo {

pattern::pattern(std::size_t code_depth, std::size_t timesteps,
                 std::vector<std::vector<pattern_coord>> layout)
    : code_depth_(code_depth), timesteps_(timesteps), layout_(std::move(layout)) {
    validate();
}

void pattern::validate() const {
    if (code_depth_ == 0) throw std::invalid_argument("pattern code depth must be positive");
    if (layout_.empty() || !layout_.front().empty())
        throw std::invalid_argument("pattern must begin with an empty sequence step");
    std::vector<std::size_t> last(code_depth_, 0);
    std::vector<uint8_t> seen(code_depth_, 0);
    for (std::size_t s = 0; s < layout_.size(); ++s) {
        std::vector<uint8_t> in_step(code_depth_, 0);
        for (const auto & c : layout_[s]) {
            if (c.codebook >= code_depth_)
                throw std::invalid_argument("pattern codebook index out of range");
            // Delayed tails intentionally retain future coordinates (for
            // example q0 at T+249 for [0,250,250]); scatter masks them.
            if (in_step[c.codebook])
                throw std::invalid_argument("pattern has duplicate codebook in one step");
            // Upstream currently leaves this check disabled, but accepting a
            // future-to-past jump would make revert ambiguous. Keep the
            // documented invariant explicit for hand-built patterns.
            if (seen[c.codebook] && c.timestep < last[c.codebook])
                throw std::invalid_argument("pattern timesteps are not monotonic");
            in_step[c.codebook] = 1;
            seen[c.codebook] = 1;
            last[c.codebook] = c.timestep;
        }
    }
}

std::size_t pattern::max_delay() const noexcept {
    std::size_t max_t = 0;
    bool any = false;
    for (const auto & step : layout_) for (const auto & c : step) {
        max_t = std::max(max_t, c.timestep + 1);
        any = true;
    }
    if (!any || max_t < timesteps_) return 0;
    return max_t - timesteps_;
}

std::vector<std::vector<pattern_coord>> pattern::valid_layout() const {
    const std::size_t n = sequence_steps() - std::min(sequence_steps(), max_delay());
    return {layout_.begin(), layout_.begin() + static_cast<std::ptrdiff_t>(n)};
}

std::vector<std::size_t> pattern::steps_with_timestep(std::size_t timestep,
                                                       std::size_t codebook) const {
    if (timestep > timesteps_) throw std::invalid_argument("timestep exceeds pattern length");
    if (codebook != SIZE_MAX && codebook >= code_depth_)
        throw std::invalid_argument("codebook exceeds pattern depth");
    std::vector<std::size_t> out;
    for (std::size_t s = 0; s < layout_.size(); ++s)
        for (const auto & c : layout_[s])
            if (c.timestep == timestep && (codebook == SIZE_MAX || c.codebook == codebook)) out.push_back(s);
    return out;
}

std::size_t pattern::first_step_with_timestep(std::size_t timestep, std::size_t codebook) const {
    auto steps = steps_with_timestep(timestep, codebook);
    if (steps.empty()) throw std::out_of_range("timestep is absent from pattern");
    return steps.front();
}

pattern_result pattern::build(const std::vector<std::vector<int64_t>> & input,
                              int64_t special_token, bool keep_only_valid_steps) const {
    if (input.size() != code_depth_) throw std::invalid_argument("input code depth does not match pattern");
    for (const auto & row : input) if (row.size() != timesteps_)
        throw std::invalid_argument("input timestep count does not match pattern");
    const auto ref = keep_only_valid_steps ? valid_layout() : layout_;
    const std::size_t sentinel = code_depth_ * timesteps_;
    pattern_result out;
    out.values.assign(code_depth_, std::vector<int64_t>(ref.size(), special_token));
    out.mask.assign(code_depth_, std::vector<uint8_t>(ref.size(), 0));
    out.indexes.assign(code_depth_, std::vector<std::size_t>(ref.size(), sentinel));
    for (std::size_t s = 0; s < ref.size(); ++s) for (const auto & c : ref[s]) {
        if (c.timestep >= timesteps_) continue;
        out.values[c.codebook][s] = input[c.codebook][c.timestep];
        out.mask[c.codebook][s] = 1;
        out.indexes[c.codebook][s] = c.timestep + c.codebook * timesteps_;
    }
    return out;
}

pattern_result pattern::revert(const std::vector<std::vector<int64_t>> & sequence,
                               int64_t special_token, bool keep_only_valid_steps) const {
    if (sequence.size() != code_depth_) throw std::invalid_argument("sequence code depth does not match pattern");
    const auto ref = keep_only_valid_steps ? valid_layout() : layout_;
    for (const auto & row : sequence) {
        if (row.size() > ref.size()) throw std::invalid_argument("sequence is longer than pattern");
        if (row.size() != sequence.front().size()) throw std::invalid_argument("sequence rows have different lengths");
    }
    const std::size_t S = sequence.front().size();
    const std::size_t sentinel = code_depth_ * S;
    pattern_result out;
    out.values.assign(code_depth_, std::vector<int64_t>(timesteps_, special_token));
    out.mask.assign(code_depth_, std::vector<uint8_t>(timesteps_, 0));
    out.indexes.assign(code_depth_, std::vector<std::size_t>(timesteps_, sentinel));
    for (std::size_t s = 0; s < S; ++s) for (const auto & c : ref[s]) {
        if (c.timestep >= timesteps_) continue;
        out.values[c.codebook][c.timestep] = sequence[c.codebook][s];
        out.mask[c.codebook][c.timestep] = 1;
        out.indexes[c.codebook][c.timestep] = s + c.codebook * S;
    }
    return out;
}

DelayedPatternProvider::DelayedPatternProvider(std::size_t code_depth,
                                               std::vector<std::size_t> delays,
                                               std::size_t flatten_first,
                                               std::size_t empty_initial)
    : code_depth_(code_depth), delays_(std::move(delays)), flatten_first_(flatten_first), empty_initial_(empty_initial) {
    if (code_depth_ == 0) throw std::invalid_argument("pattern code depth must be positive");
    if (delays_.empty()) { delays_.resize(code_depth_); for (std::size_t q = 0; q < code_depth_; ++q) delays_[q] = q; }
    if (delays_.size() != code_depth_) throw std::invalid_argument("delay count does not match code depth");
    if (!std::is_sorted(delays_.begin(), delays_.end())) throw std::invalid_argument("delays must be sorted");
}

pattern DelayedPatternProvider::get_pattern(std::size_t timesteps) const {
    std::vector<std::vector<pattern_coord>> layout(1);
    layout.insert(layout.end(), empty_initial_, {});
    const std::size_t max_delay = delays_.empty() ? 0 : delays_.back();
    const std::size_t first = std::min(flatten_first_, timesteps);
    for (std::size_t t = 0; t < first; ++t)
        for (std::size_t q = 0; q < code_depth_; ++q) layout.push_back({{t, q}});
    for (std::size_t t = first; t < timesteps + max_delay; ++t) {
        std::vector<pattern_coord> step;
        for (std::size_t q = 0; q < code_depth_; ++q)
            if (t >= delays_[q] && t - delays_[q] >= first)
                step.push_back({t - delays_[q], q});
        layout.push_back(std::move(step));
    }
    return pattern(code_depth_, timesteps, std::move(layout));
}

pattern make_delayed_pattern(std::size_t code_depth, std::size_t timesteps,
                             const std::vector<std::size_t> & delays,
                             std::size_t flatten_first, std::size_t empty_initial) {
    return DelayedPatternProvider(code_depth, delays, flatten_first, empty_initial).get_pattern(timesteps);
}

pattern_result build_delayed_pattern(const std::vector<std::vector<int64_t>> & input,
                                     int64_t special_token,
                                     const std::vector<std::size_t> & delays,
                                     std::size_t flatten_first, std::size_t empty_initial,
                                     bool keep_only_valid_steps) {
    if (input.empty()) throw std::invalid_argument("input must have at least one codebook");
    return make_delayed_pattern(input.size(), input.front().size(), delays, flatten_first, empty_initial)
        .build(input, special_token, keep_only_valid_steps);
}

pattern_result revert_delayed_pattern(const std::vector<std::vector<int64_t>> & sequence,
                                      std::size_t timesteps, int64_t special_token,
                                      const std::vector<std::size_t> & delays,
                                      std::size_t flatten_first, std::size_t empty_initial,
                                      bool keep_only_valid_steps) {
    if (sequence.empty()) throw std::invalid_argument("sequence must have at least one codebook");
    return make_delayed_pattern(sequence.size(), timesteps, delays, flatten_first, empty_initial)
        .revert(sequence, special_token, keep_only_valid_steps);
}

} // namespace levo
