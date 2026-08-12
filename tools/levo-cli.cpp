#include "levo.h"

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

levo::backend_kind parse_backend(const std::string & value) {
    if (value == "auto") {
        return levo::backend_kind::auto_select;
    }
    if (value == "cpu") {
        return levo::backend_kind::cpu;
    }
    if (value == "cuda") {
        return levo::backend_kind::cuda;
    }
    throw std::invalid_argument("unknown backend '" + value + "'");
}

void usage(const char * program) {
    std::cout
        << "LeVo2.cpp " << levo::version() << "\n\n"
        << "Foundation diagnostics:\n"
        << "  " << program << " --list-backends\n"
        << "  " << program << " --smoke [auto|cpu|cuda] [device-index]\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 1 || std::string(argv[1]) == "--help") {
            usage(argv[0]);
            return 0;
        }

        const std::string command = argv[1];
        if (command == "--list-backends") {
            for (const auto & backend : levo::available_backends()) {
                std::cout << backend.name << '\t' << backend.description << '\t'
                          << backend.memory_free << '/' << backend.memory_total << " bytes\n";
            }
            return 0;
        }

        if (command == "--smoke") {
            const auto kind = argc >= 3 ? parse_backend(argv[2])
                                        : levo::backend_kind::auto_select;
            const int device = argc >= 4 ? std::stoi(argv[3]) : 0;
            const auto result = levo::backend_smoke(kind, device);
            for (std::size_t i = 0; i < result.size(); ++i) {
                if (i != 0) {
                    std::cout << ' ';
                }
                std::cout << std::fixed << std::setprecision(1) << result[i];
            }
            std::cout << '\n';
            return 0;
        }

        usage(argv[0]);
        throw std::invalid_argument("unknown command '" + command + "'");
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
