#include "levo-engine-request.h"
#include "levo-token-io.h"

#include <cassert>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

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

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

} // namespace

int main() {
    try {
        using namespace levo::engine_request;
        request value = parse("{\"caption\":\"pop\",\"duration_seconds\":2,\"flow\":{\"cfg_scale\":1.25,\"euler_steps\":8,\"seed\":9},\"lyrics\":\"hello \\uD83C\\uDFB5\",\"sampling\":{\"top_k_mixed\":7},\"seed\":42}");
        assert(value.seed_present && value.seed == 42 && value.sampling.top_k_mixed == 7);
        assert(value.flow_seed == 9 && value.flow_euler_steps == 8 && value.flow_cfg_scale == 1.25F);
        const std::string stable = serialize(value);
        const request reparsed = parse(stable);
        assert(reparsed.lyrics == value.lyrics && reparsed.seed == 42);
        assert(reparsed.description == "pop");
        levo::generation_result generated;
        generated.frame_count = 2;
        generated.tokens = {1, 2, 3, 4, 5, 6};
        const std::string completed_json = serialize_codes(value, generated);
        const codes completed = parse_codes(completed_json);
        assert(completed.frame_count == 2 && completed.tokens == generated.tokens);
        assert(completed.generation_request.flow_seed == 9);
        const request max_seed = parse("{\"lyrics\":\"a\",\"caption\":\"b\",\"duration\":1,\"seed\":18446744073709551615}");
        assert(max_seed.seed == UINT64_MAX);

        assert(parse("{\"description\":\"legacy\",\"duration_seconds\":2,\"lyrics\":\"a\"}").description == "legacy");
        const auto flat = parse("{\"caption\":\"pop\",\"temperature\":0.7,\"top_k_mixed\":20,\"top_k_detail\":30}");
        assert(flat.sampling.top_k_mixed == 20 && flat.sampling.top_k_detail == 30);
        auto legacy_codes = completed_json;
        const auto promoted = legacy_codes.find(",\"duration\":2");
        assert(promoted != std::string::npos);
        legacy_codes.erase(promoted, std::string(",\"duration\":2").size());
        assert(parse_codes(legacy_codes).frame_count == 2);

        // The host speaks one request dialect to every engine family: a
        // caption rather than a description, and engine-neutral names for the
        // two knobs that land on the Flow renderer here.
        const request host = parse("{\"caption\":\"a warm nocturnal bolero\",\"guidance_scale\":2.5,\"inference_steps\":12}");
        assert(host.description == "a warm nocturnal bolero");
        assert(host.flow_euler_steps == 12 && host.flow_cfg_scale == 2.5F);
        // An explicit flow block is the engine-native form and stays authoritative.
        const request both = parse("{\"caption\":\"a\",\"inference_steps\":12,\"flow\":{\"euler_steps\":30}}");
        assert(both.flow_euler_steps == 30);

        // An instrumental request carries no lyrics, and an unspecified length
        // takes the model's own default rather than failing the job.
        const request sparse = parse("{\"caption\":\"solo piano\"}");
        assert(sparse.lyrics.empty());
        assert(sparse.duration_seconds == 120.0);
        assert(sparse.flow_euler_steps == 0 && sparse.flow_cfg_scale == 0.0F);

        // The host reads `duration` off the CODES blob to prove the engine did
        // not grow the length it was asked for, so it has to be there, and it
        // has to agree with the request the blob carries.
        assert(contains(completed_json, "\"duration\":2"));
        bool doctored_rejected = false;
        try {
            (void) parse_codes("{\"audio_codes\":[1,2,3,4,5,6],\"duration\":9,\"frame_count\":2,\"request\":" +
                               stable + ",\"token_sha256\":\"" +
                               levo::token_io::tensor_sha256(generated.tokens) + "\"}");
        } catch (const std::exception &) {
            doctored_rejected = true;
        }
        assert(doctored_rejected);

        expect_throw("{\"caption\":\"b\",\"description\":\"b\"}");
        expect_throw("{\"lyrics\":\"a\",\"caption\":\"b\",\"duration\":1,\"unexpected\":true}");
        expect_throw("{\"lyrics\":\"a\",\"lyrics\":\"b\",\"caption\":\"b\",\"duration\":1}");
        expect_throw("{\"lyrics\":\"a\",\"caption\":\"b\",\"duration\":1,\"seed\":-1}");
        expect_throw("{\"lyrics\":\"a\",\"caption\":\"b\",\"duration\":1,\"flow\":{\"unknown\":1}}");
        expect_throw("{\"lyrics\":\"a\",\"caption\":\"b\",\"duration\":1,\"guidance_scale\":-1}");
        expect_throw("{\"lyrics\":\"a\",\"caption\":\"b\",\"duration\":01}");
        expect_throw("{\"lyrics\":\"a\",\"caption\":\"b\",\"duration\":1.}");
        expect_throw("{\"lyrics\":\"a\",\"caption\":\"b\",\"duration\":1e}");
        std::cout << "engine request ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
