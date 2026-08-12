#include "../src/levo-pattern.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace levo;
    const std::vector<std::vector<int64_t>> in{{10,11,12,13},{20,21,22,23},{30,31,32,33}};
    auto p = make_delayed_pattern(3, 4, {0, 1, 1});
    assert(p.sequence_steps() == 6);
    auto b = p.build(in, 99);
    assert((b.values[0] == std::vector<int64_t>{99,10,11,12,13,99}));
    assert((b.values[1] == std::vector<int64_t>{99,99,20,21,22,23}));
    assert((b.values[2] == std::vector<int64_t>{99,99,30,31,32,33}));
    auto r = p.revert(b.values, -1);
    assert(r.values == in);
    auto real = make_delayed_pattern(3, 251, {0,250,250});
    assert(real.sequence_steps() == 502);
    const pattern_coord c00{0,0}, c01{0,1}, c02{0,2};
    assert(real.layout()[1].size() == 1 && real.layout()[1][0] == c00);
    assert(real.layout()[251].size() == 3);
    assert(real.layout()[251][1] == c01);
    assert(real.layout()[251][2] == c02);
    std::cout << "pattern ok\n";
}
