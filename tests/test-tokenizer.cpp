#include "levo-tokenizer.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#ifndef LEVO_TEST_FIXTURES_DIR
#define LEVO_TEST_FIXTURES_DIR "tests/fixtures"
#endif

int main() {
    using namespace levo;
    const std::string root = LEVO_TEST_FIXTURES_DIR;
    auto tokenizer = ByteLevelBPETokenizer::load(
        root + "/tiny-vocab.json", root + "/tiny-merges.txt",
        root + "/tiny-tokenizer-config.json");
    tokenizer.add_special_token("[verse]", 20);

    assert(tokenizer.encode("hello world") == std::vector<int64_t>({15, 16}));
    assert(tokenizer.encode("a  b!") == std::vector<int64_t>({1, 13, 13, 2, 0}));
    assert(tokenizer.encode("<|im_start|>[verse]hello") ==
           std::vector<int64_t>({19, 20, 15}));
    assert(tokenizer.decode(tokenizer.encode("hello world")) == "hello world");
    assert(tokenizer.decode({19, 20, 15}, true) == "hello");

    std::vector<std::string> embedded_tokens;
    for (int64_t id = 0; id < 20; ++id) embedded_tokens.push_back(tokenizer.token_string(id));
    const std::vector<std::string> embedded_merges = {
        "h e", "he l", "hel l", "hell o", "w o", "wo r", "wor l", "worl d",
        "Ġ h", "Ġhello", "Ġ w", "Ġworld"};
    auto embedded = ByteLevelBPETokenizer::load_embedded(
        embedded_tokens, embedded_merges,
        R"([{"id":18,"content":"<|endoftext|>","special":true},{"id":19,"content":"<|im_start|>","special":true}])");
    assert(embedded.encode("<|im_start|>hello world") == std::vector<int64_t>({19, 15, 16}));
    assert(embedded.decode({19, 15, 16}, true) == "hello world");
    std::cout << "tokenizer ok\n";
}
