#include "levo-engine-request.h"

#include <cassert>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

void expect_throw(const std::string & text) {
    bool rejected = false;
    try {
        (void) levo::engine_request::parse(text);
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main() {
    try {
        using namespace levo::engine_request;
        request value = parse("{\"description\":\"pop\",\"duration_seconds\":2,\"lyrics\":\"hello \\uD83C\\uDFB5\",\"sampling\":{\"top_k_mixed\":7},\"seed\":42}");
        assert(value.seed_present && value.seed == 42 && value.sampling.top_k_mixed == 7);
        const std::string stable = serialize(value);
        const request reparsed = parse(stable);
        assert(reparsed.lyrics == value.lyrics && reparsed.seed == 42);
        const request max_seed = parse("{\"lyrics\":\"a\",\"description\":\"b\",\"duration\":1,\"seed\":18446744073709551615}");
        assert(max_seed.seed == UINT64_MAX);
        expect_throw("{\"lyrics\":\"a\",\"description\":\"b\",\"duration\":1,\"unexpected\":true}");
        expect_throw("{\"lyrics\":\"a\",\"lyrics\":\"b\",\"description\":\"b\",\"duration\":1}");
        expect_throw("{\"lyrics\":\"a\",\"description\":\"b\",\"duration\":1,\"seed\":-1}");
        expect_throw("{\"lyrics\":\"a\",\"description\":\"b\",\"duration\":01}");
        expect_throw("{\"lyrics\":\"a\",\"description\":\"b\",\"duration\":1.}");
        expect_throw("{\"lyrics\":\"a\",\"description\":\"b\",\"duration\":1e}");
        std::cout << "engine request ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
