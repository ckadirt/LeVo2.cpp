#include "levo-generator.h"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

levo::detail::model_hparams tiny_hparams() {
    levo::detail::model_hparams hparams;
    hparams.main_layers = 1;
    hparams.detail_layers = 1;
    hparams.embedding_length = 4;
    hparams.feed_forward_length = 4;
    hparams.attention_heads = 1;
    hparams.kv_attention_heads = 1;
    hparams.context_length = 16;
    hparams.codebook_count = 3;
    hparams.codebook_size = 3;
    hparams.eos_token_id = 3;
    hparams.special_token_id = 4;
    hparams.frame_rate = 1;
    hparams.delays = {0, 1, 1};
    hparams.lyrics_prefix_length = 0;
    hparams.prompt_prefix_length = 0;
    hparams.style_prefix_length = 0;
    return hparams;
}

levo::detail::generation_logits logits_for(int mixed, int vocal, int bgm) {
    const auto row = [](int best) {
        levo::logits result(4, -10.0F);
        result[static_cast<std::size_t>(best)] = 10.0F;
        return result;
    };
    return {{row(mixed), row(vocal), row(bgm)}};
}

levo::generation_config test_config() {
    levo::generation_config config;
    config.duration_seconds = 2.0;
    config.sampling.use_sampling = false;
    config.sampling.top_k_mixed = 1;
    config.sampling.top_k_detail = 1;
    config.sampling.repetition_penalty = 1.0F;
    return config;
}

void expect(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        using namespace levo;
        using namespace levo::detail;
        const model_hparams hparams = tiny_hparams();
        const generation_config config = test_config();
        std::vector<std::array<int32_t, 3>> inputs;
        std::vector<generation_progress> progress_events;
        const generation_result normal = run_generation_controller(
            hparams, config, logits_for(1, 2, 2),
            [&inputs](const std::array<int32_t, 3> & input) {
                inputs.push_back(input);
                return logits_for(1, 2, 2);
            },
            [&progress_events](const generation_progress & value) { progress_events.push_back(value); });
        expect(normal.requested_frames == 2 && normal.frame_count == 2,
               "controller did not retain the requested two frames");
        expect(normal.sequence_steps == 4, "controller sequence length is not initial + delayed positions");
        expect((normal.tokens == std::vector<int32_t>{1, 1, 2, 2, 2, 2}),
               "delayed controller did not revert stream-major values correctly");
        expect(inputs.size() == 2, "controller did not advance exactly once per nonterminal position");
        expect(inputs[0] == std::array<int32_t, 3>{{1, 4, 4}},
               "invalid delayed detail positions were not forced to special before decode");
        expect(inputs[1] == std::array<int32_t, 3>{{1, 2, 2}},
               "first fully valid delayed position was not decoded");
        expect(progress_events.size() == 4 && progress_events.front().completed_steps == 0 &&
                   progress_events.back().completed_steps == progress_events.back().total_steps,
               "generation progress did not cover the complete monotonic step range");
        for (std::size_t index = 1; index < progress_events.size(); ++index) {
            expect(progress_events[index].stage == generation_stage::generating &&
                       progress_events[index].completed_steps == progress_events[index - 1].completed_steps + 1,
                   "generation progress steps are not monotonic");
        }

        generation_config cancellable = config;
        std::size_t cancel_polls = 0;
        cancellable.cancelled = [&cancel_polls] { return cancel_polls++ != 0; };
        bool was_cancelled = false;
        try {
            (void) run_generation_controller(
                hparams, cancellable, logits_for(1, 2, 2),
                [](const std::array<int32_t, 3> &) { return logits_for(1, 2, 2); });
        } catch (const operation_cancelled &) {
            was_cancelled = true;
        }
        expect(was_cancelled && cancel_polls == 2,
               "generation cancellation was not observed at a delayed-step boundary");

        // An EOS drawn at a valid mixed position is retained through revert
        // and then removed, together with every frame after the earliest EOS.
        const generation_result eos = run_generation_controller(
            hparams, config, logits_for(3, 3, 3),
            [](const std::array<int32_t, 3> &) { return logits_for(3, 3, 3); });
        expect(eos.frame_count == 0 && eos.tokens.empty(),
               "earliest EOS was not trimmed before artifact construction");
        expect(eos.ended[0] && eos.ended[1] && eos.ended[2],
               "EOS tracker did not retain per-stream endings");

        generation_config invalid = config;
        invalid.duration_seconds = 0.0;
        bool rejected = false;
        try {
            (void) run_generation_controller(hparams, invalid, logits_for(1, 1, 1),
                                              [](const std::array<int32_t, 3> &) { return logits_for(1, 1, 1); });
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        expect(rejected, "zero duration was not rejected before generation");
        std::cout << "generation ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
