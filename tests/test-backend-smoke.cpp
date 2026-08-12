#include "levo.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    try {
        levo::backend_kind kind = levo::backend_kind::cpu;
        if (argc == 2 && std::string(argv[1]) == "cuda") {
            kind = levo::backend_kind::cuda;
        }

        const auto result = levo::backend_smoke(kind);
        if (result.size() != 4) {
            std::cerr << "unexpected result size\n";
            return 1;
        }
        for (const float value : result) {
            if (std::fabs(value - 5.0F) > 1.0e-6F) {
                std::cerr << "unexpected result value: " << value << '\n';
                return 1;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
