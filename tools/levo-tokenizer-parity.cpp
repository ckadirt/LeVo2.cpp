#include "levo-tokenizer.h"

#include <exception>
#include <iostream>
#include <iterator>
#include <string>

int main(int argc, char ** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: " << argv[0] << " TOKENIZER_DIR TEXT\n";
            return 2;
        }
        const std::string root = argv[1];
        auto tokenizer = levo::ByteLevelBPETokenizer::load_tokenizer_json(
            root + "/tokenizer.json", root + "/tokenizer_config.json");
        const char * structures[] = {
            "[verse]", "[chorus]", "[bridge]", "[intro-short]", "[intro-medium]",
            "[intro-long]", "[outro-short]", "[outro-medium]", "[outro-long]",
            "[inst-short]", "[inst-medium]", "[inst-long]", "[silence]",
        };
        for (int64_t index = 0; index < 13; ++index) {
            tokenizer.add_special_token(structures[index], 151646 + index);
        }
        const auto ids = tokenizer.encode(argv[2]);
        for (std::size_t index = 0; index < ids.size(); ++index) {
            if (index != 0) std::cout << ' ';
            std::cout << ids[index];
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
