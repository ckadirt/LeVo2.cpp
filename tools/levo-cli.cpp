#include "levo.h"

#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
    if (value == "gpu") {
        return levo::backend_kind::gpu;
    }
    throw std::invalid_argument("unknown backend '" + value + "'");
}

void usage(const char * program) {
    std::cout
        << "LeVo2.cpp " << levo::version() << "\n\n"
        << "Foundation diagnostics:\n"
        << "  " << program << " --list-backends\n"
        << "  " << program << " --smoke [auto|cpu|cuda|gpu] [device-index]\n\n"
        << "Autoregressive token generation:\n"
        << "  " << program << " --model MODEL.gguf --lyrics lyrics.txt [--prompt TEXT]\n"
        << "      --duration SECONDS --output tokens.npy\n"
        << "      [--backend auto|cpu|cuda|gpu] [--device N] [--greedy] [--seed N]\n";
}

std::string read_text_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open lyrics file: " + path);
    }
    std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("failed while reading lyrics file: " + path);
    }
    return contents;
}

double parse_duration(const std::string & value) {
    std::size_t used = 0;
    double result = 0.0;
    try {
        result = std::stod(value, &used);
    } catch (const std::exception &) {
        throw std::invalid_argument("invalid duration '" + value + "'");
    }
    if (used != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument("invalid duration '" + value + "'");
    }
    return result;
}

uint64_t parse_seed(const std::string & value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument("invalid seed '" + value + "'");
    }
    std::size_t used = 0;
    try {
        const unsigned long long result = std::stoull(value, &used, 10);
        if (used != value.size() || result > std::numeric_limits<uint64_t>::max()) {
            throw std::invalid_argument("seed is outside uint64 range");
        }
        return static_cast<uint64_t>(result);
    } catch (const std::invalid_argument &) {
        throw;
    } catch (const std::exception &) {
        throw std::invalid_argument("invalid seed '" + value + "'");
    }
}

int parse_device(const std::string & value) {
    std::size_t used = 0;
    try {
        const int result = std::stoi(value, &used, 10);
        if (used != value.size() || result < 0) {
            throw std::invalid_argument("device index must be non-negative");
        }
        return result;
    } catch (const std::invalid_argument &) {
        throw;
    } catch (const std::exception &) {
        throw std::invalid_argument("invalid device index '" + value + "'");
    }
}

struct cli_generation_request {
    levo::generation_config config;
    std::filesystem::path output_path;
};

cli_generation_request parse_generation(int argc, char ** argv) {
    cli_generation_request request;
    levo::generation_config & config = request.config;
    bool model = false;
    bool lyrics = false;
    bool duration = false;
    bool output = false;
    std::string lyrics_path;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            return argv[index];
        };
        if (argument == "--model") {
            config.model_path = value();
            model = true;
        } else if (argument == "--lyrics") {
            lyrics_path = value();
            lyrics = true;
        } else if (argument == "--prompt") {
            config.description = value();
        } else if (argument == "--duration") {
            config.duration_seconds = parse_duration(value());
            duration = true;
        } else if (argument == "--output") {
            request.output_path = value();
            output = true;
        } else if (argument == "--backend") {
            config.backend = parse_backend(value());
        } else if (argument == "--device") {
            config.device_index = parse_device(value());
        } else if (argument == "--seed") {
            config.seed = parse_seed(value());
            config.seed_present = true;
        } else if (argument == "--greedy") {
            config.sampling.use_sampling = false;
        } else if (argument == "--help") {
            throw std::invalid_argument("--help must be used by itself");
        } else {
            throw std::invalid_argument("unknown option '" + argument + "'");
        }
    }
    if (!model || !lyrics || !duration || !output) {
        throw std::invalid_argument("generation requires --model, --lyrics, --duration, and --output");
    }
    config.lyrics = read_text_file(lyrics_path);
    if (request.output_path.extension() != ".npy") {
        throw std::invalid_argument("--output must use the .npy extension");
    }
    return request;
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

        if (command.rfind("--", 0) == 0) {
            const cli_generation_request request = parse_generation(argc, argv);
            const auto progress = [](const levo::generation_progress & value) {
                if (value.completed_steps == value.total_steps || value.completed_steps % 25 == 0) {
                    std::cerr << "generation " << value.completed_steps << '/' << value.total_steps << " delayed steps\n";
                }
            };
            const levo::generation_result result = levo::generate_tokens(request.config, progress);
            levo::write_generation_artifact(request.output_path, result, request.config);
            std::cout << "wrote " << request.output_path << " [3, " << result.frame_count << "]"
                      << " using " << result.backend_name << '\n';
            return 0;
        }

        usage(argv[0]);
        throw std::invalid_argument("unknown command '" + command + "'");
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
