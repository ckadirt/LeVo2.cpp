#pragma once

#include "levo.h"
#include "levo-model.h"
#include "levo-sampling.h"

#include <array>
#include <functional>

namespace levo::detail {

// A single CFG-combined set of logits in mixed, vocal, BGM order.  This
// small controller seam is deliberately independent from GGML so delayed
// pattern and EOS behavior can be regression-tested with synthetic logits.
using generation_logits = std::array<logits, 3>;
using generation_advance_callback = std::function<generation_logits(
    const std::array<int32_t, 3> & delayed_input)>;

// Runs the delayed token controller after an initial all-special position has
// already been prefetched. `advance` receives each sampled/masked position
// and returns the CFG-combined logits for the following position.
generation_result run_generation_controller(
    const model_hparams & hparams,
    const generation_config & config,
    generation_logits initial_logits,
    generation_advance_callback advance,
    generation_progress_callback progress = {});

} // namespace levo::detail
