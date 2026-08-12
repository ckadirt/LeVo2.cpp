#include "../src/levo-sampling.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace levo;
    assert(apply_cfg({2,4},{1,1},1.5F) == logits({2.5F,5.5F}));
    logits x{10, 8, 6};
    apply_unique_token_repetition_penalty(x, {0,0,1}, 2.0F);
    assert(x[0] == 5 && x[1] == 4 && x[2] == 6); // unique IDs, not frequencies
    sampling_config config; config.use_sampling=false; config.repetition_penalty=2.0F;
    Sampler upstream(0);
    const auto sampled = upstream.sample_streams({{9,8,10,7}}, config, {{2,3}});
    assert(sampled[0] == 0); // token 2 is penalized; EOS (last logit) is excluded
    Sampler a(1234), b(1234);
    for (int i=0;i<20;++i) assert(a.sample({0,1,2}, 3, 1, true) == b.sample({0,1,2}, 3, 1, true));
    assert(Sampler(1).sample({-1,3,-2}, 1, 1, true) == 1);
    // Upstream keeps every candidate tied with the kth probability. With
    // k=1, both equal maxima must therefore remain sampleable.
    std::mt19937_64 tie_rng(7);
    bool saw_first = false, saw_second = false;
    for (int i = 0; i < 64; ++i) {
        const auto token = sample_top_k({5,5,0}, 1, 1, true, tie_rng);
        saw_first |= token == 0;
        saw_second |= token == 1;
    }
    assert(saw_first && saw_second);
    eos_tracker e(3, 9); e.update({1,9,2}); assert(!e.all_ended() && e.ended(1)); e.update({9,9,2}); e.update({9,9,9}); assert(e.all_ended() && e.first_ended_length()==0);
    assert(trim_length_at_eos({{1,2,9,4},{1,9,3,4}},9)==1);
    std::cout << "sampling ok\n";
}
