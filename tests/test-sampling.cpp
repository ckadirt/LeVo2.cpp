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
    const auto sampled = upstream.sample_streams({{9,8,10}}, config, {{2}});
    assert(sampled[0] == 2); // EOS (last logit) is not repetition-penalized
    Sampler a(1234), b(1234);
    for (int i=0;i<20;++i) assert(a.sample({0,1,2}, 3, 1, true) == b.sample({0,1,2}, 3, 1, true));
    assert(Sampler(1).sample({-1,3,-2}, 1, 1, true) == 1);
    eos_tracker e(3, 9); e.update({1,9,2}); assert(!e.all_ended() && e.ended(1)); e.update({9,9,2}); e.update({9,9,9}); assert(e.all_ended() && e.first_ended_length()==0);
    assert(trim_length_at_eos({{1,2,9,4},{1,9,3,4}},9)==1);
    std::cout << "sampling ok\n";
}
