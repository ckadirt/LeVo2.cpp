#include "../src/levo-tokenizer.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace levo;
    const std::string p = "/workspace/models/SongGeneration-Runtime/third_party/Qwen2-7B/";
    auto t = ByteLevelBPETokenizer::load(p+"vocab.json", p+"merges.txt", p+"tokenizer_config.json");
    t.add_special_token("[verse]", 151646);
    assert(t.encode("hello world") == std::vector<int64_t>({14990, 1879}));
    assert(t.encode("中文歌词") == std::vector<int64_t>({104811,114355}));
    assert(t.encode("a  b!") == std::vector<int64_t>({64,220,293,0}));
    assert(t.encode("it's 2024") == std::vector<int64_t>({275,594,220,17,15,17,19}));
    assert(t.encode("<|im_start|>[verse]hello") == std::vector<int64_t>({151644,151646,14990}));
    assert(t.decode(t.encode("hello world")) == "hello world");
    std::cout << "tokenizer ok\n";
}
