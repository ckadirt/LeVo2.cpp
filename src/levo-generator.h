#pragma once

#include "levo.h"
#include "levo-model.h"
#include "levo-sampling.h"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace levo::detail {

// A single CFG-combined set of logits in mixed, vocal, BGM order.  This
// small controller seam is deliberately independent from GGML so delayed
// pattern and EOS behavior can be regression-tested with synthetic logits.
using generation_logits = std::array<logits, 3>;
using generation_advance_callback = std::function<generation_logits(
    const std::array<int32_t, 3> & delayed_input)>;

// The mutable host-side portion of a paused LeLM run. K/V is deliberately not
// present: it is backend-resident and is rebuilt with the same one-token graph
// sequence before this controller resumes.
struct generation_resume_state {
    std::vector<std::vector<int64_t>> delayed_sequence;
    uint64_t sampler_seed = 0;
    uint64_t sampler_draws = 0;
    std::array<bool, 3> ended{{false, false, false}};
    std::size_t earliest_eos = 0;
    std::array<std::uint8_t, 32> next_logits_digest{};
};

struct resumable_generation_result {
    bool paused = false;
    generation_result result;
    generation_resume_state resume;
};

// Runs the delayed token controller after an initial all-special position has
// already been prefetched. `advance` receives each sampled/masked position
// and returns the CFG-combined logits for the following position.
generation_result run_generation_controller(
    const model_hparams & hparams,
    const generation_config & config,
    generation_logits initial_logits,
    generation_advance_callback advance,
    generation_progress_callback progress = {});

// Runs from an initial BOS prediction or a restored delayed sequence. A
// cancellation callback produces a checkpointable paused result; the blocking
// compatibility wrapper above preserves its historical exception behavior.
resumable_generation_result run_generation_controller_resumable(
    const model_hparams & hparams,
    const generation_config & config,
    generation_logits initial_logits,
    generation_advance_callback advance,
    const generation_resume_state * resume = nullptr,
    generation_progress_callback progress = {});

} // namespace levo::detail

namespace levo {

using generation_model_ready_callback = std::function<void(const generation_result &)>;

// Internal engine seam: the normal public generator remains blocking, while
// the Cantor ABI uses this result to turn a cooperative cancellation into a
// serializable delayed-token checkpoint.
detail::resumable_generation_result generate_tokens_resumable(
    const generation_config & config,
    const detail::generation_resume_state * resume = nullptr,
    generation_progress_callback progress = {},
    generation_model_ready_callback model_ready = {});

} // namespace levo
