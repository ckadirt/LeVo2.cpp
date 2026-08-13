#include "cantor_engine.h"
#include "levo.h"
#include "levo-progress.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

constexpr std::array<char, 8> lm_magic{{'L', 'E', 'V', 'O', 'L', 'M', '0', '2'}};
constexpr std::array<char, 8> flow_magic{{'L', 'E', 'V', 'O', 'F', 'L', '0', '2'}};
constexpr std::array<char, 8> latent_magic{{'L', 'E', 'V', 'O', 'L', 'T', '0', '2'}};
constexpr std::array<char, 8> lm_magic_v1{{'L', 'E', 'V', 'O', 'L', 'M', '0', '1'}};
constexpr std::array<char, 8> flow_magic_v1{{'L', 'E', 'V', 'O', 'F', 'L', '0', '1'}};
constexpr std::array<char, 8> latent_magic_v1{{'L', 'E', 'V', 'O', 'L', 'T', '0', '1'}};

struct options {
    std::filesystem::path input;
    std::filesystem::path checkpoint;
    std::filesystem::path output;
    std::string lm;
    std::string dit;
    std::string vae;
    bool resuming = false;
};

[[noreturn]] void fail(const std::string & message) { throw std::runtime_error("levo-cantor: " + message); }

void usage(const char * program) {
    std::cout << "Usage:\n  " << program
              << " --input request.json --checkpoint state.bin --output song.wav"
                 " --lm LM.gguf --dit FLOW.gguf --vae VAE.gguf\n  "
              << program
              << " --resume state.bin --checkpoint state.bin --output song.wav"
                 " --lm LM.gguf --dit FLOW.gguf --vae VAE.gguf\n\n"
                 "The checkpoint is atomically rewritten on CODES/DIFFUSE pauses and before "
                 "DECODE starts. SIGINT/SIGTERM returns 130 after a durable pause.\n";
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) fail("cannot open " + path.string());
    const std::streamoff size = input.tellg();
    if (size <= 0) fail("input is empty: " + path.string());
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char *>(result.data()), size);
    if (!input) fail("cannot read " + path.string());
    return result;
}

void sync_file(const std::filesystem::path & path) {
#if defined(__unix__) || defined(__APPLE__)
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) fail("cannot open checkpoint for fsync: " + path.string());
    const int result = ::fsync(descriptor);
    const int close_result = ::close(descriptor);
    if (result != 0 || close_result != 0) fail("cannot fsync checkpoint: " + path.string());
#else
    (void) path;
#endif
}

void sync_directory(const std::filesystem::path & path) {
#if defined(__unix__) || defined(__APPLE__)
    const std::filesystem::path directory = path.parent_path().empty() ? "." : path.parent_path();
    const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) fail("cannot open checkpoint directory for fsync: " + directory.string());
    const int result = ::fsync(descriptor);
    const int close_result = ::close(descriptor);
    if (result != 0 || close_result != 0) fail("cannot fsync checkpoint directory: " + directory.string());
#else
    (void) path;
#endif
}

void write_checkpoint(const std::filesystem::path & target, const std::vector<std::uint8_t> & bytes) {
    if (bytes.empty()) fail("refusing to overwrite checkpoint with an empty blob");
    const std::filesystem::path temporary = target.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) fail("cannot create temporary checkpoint: " + temporary.string());
        output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output) fail("cannot write temporary checkpoint: " + temporary.string());
    }
    sync_file(temporary);
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) fail("cannot atomically install checkpoint: " + error.message());
    sync_directory(target);
}

bool has_magic(const std::vector<std::uint8_t> & bytes, const std::array<char, 8> & magic) {
    return bytes.size() > magic.size() && std::equal(magic.begin(), magic.end(), reinterpret_cast<const char *>(bytes.data()));
}

cantor_stage stage_for(const std::vector<std::uint8_t> & bytes) {
    if (has_magic(bytes, lm_magic)) return CANTOR_STAGE_CODES;
    if (has_magic(bytes, flow_magic)) return CANTOR_STAGE_DIFFUSE;
    if (has_magic(bytes, latent_magic)) return CANTOR_STAGE_DECODE;
    if (has_magic(bytes, lm_magic_v1) || has_magic(bytes, flow_magic_v1) || has_magic(bytes, latent_magic_v1)) {
        fail("checkpoint format 01 predates artifact identity; restart from the original request or durable CODES boundary");
    }
    return CANTOR_STAGE_CODES;
}

const char * stage_name(cantor_stage stage) {
    switch (stage) {
        case CANTOR_STAGE_CODES: return "CODES";
        case CANTOR_STAGE_DIFFUSE: return "DIFFUSE";
        case CANTOR_STAGE_DECODE: return "DECODE";
        default: return "unknown";
    }
}

struct progress_state {
    std::array<std::chrono::steady_clock::time_point, 5> last{};
};

void on_progress(cantor_stage stage, int completed, int total, void * userdata) {
    auto * state = static_cast<progress_state *>(userdata);
    const auto now = std::chrono::steady_clock::now();
    const std::size_t index = static_cast<std::size_t>(stage);
    if (completed != total && state != nullptr && state->last[index].time_since_epoch().count() != 0 &&
        std::chrono::duration<double>(now - state->last[index]).count() < 1.0) return;
    if (state != nullptr) state->last[index] = now;
    std::cerr << '[' << stage_name(stage) << "] " << completed << '/' << total << '\n';
}

int should_cancel(void *) { return levo_cli::cancellation_requested() ? 1 : 0; }

options parse(int argc, char ** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        const auto value = [&]() -> std::string {
            if (++index >= argc) fail("missing value for " + key);
            return argv[index];
        };
        if (key == "--input") result.input = value();
        else if (key == "--resume") { result.input = value(); result.resuming = true; }
        else if (key == "--checkpoint") result.checkpoint = value();
        else if (key == "--output") result.output = value();
        else if (key == "--lm") result.lm = value();
        else if (key == "--dit") result.dit = value();
        else if (key == "--vae") result.vae = value();
        else if (key == "--help") fail("--help must be used by itself");
        else fail("unknown option " + key);
    }
    if (result.input.empty() || result.checkpoint.empty() || result.output.empty() || result.lm.empty() ||
        result.dit.empty() || result.vae.empty()) {
        fail("--input/--resume, --checkpoint, --output, --lm, --dit, and --vae are required");
    }
    if (result.output.extension() != ".wav") fail("--output must end in .wav");
    return result;
}

std::vector<std::uint8_t> copy_engine_blob(std::uint8_t * blob, std::size_t size) {
    if (blob == nullptr || size == 0) fail("engine returned an empty stage blob");
    std::vector<std::uint8_t> result(blob, blob + size);
    cantor_engine_free_blob(blob);
    return result;
}

void write_audio(const std::filesystem::path & output, cantor_ctx * context) {
    int samples = 0;
    int sample_rate = 0;
    const float * planar = cantor_engine_audio(context, &samples, &sample_rate);
    if (planar == nullptr || samples <= 0 || sample_rate != 48000) fail("engine did not return valid 48 kHz audio");
    levo::render_result result;
    result.sample_rate = static_cast<std::uint32_t>(sample_rate);
    result.samples_per_channel = static_cast<std::size_t>(samples);
    result.interleaved_stereo.reserve(static_cast<std::size_t>(samples) * 2U);
    for (int index = 0; index < samples; ++index) {
        result.interleaved_stereo.push_back(planar[index]);
        result.interleaved_stereo.push_back(planar[samples + index]);
    }
    levo::write_render_wav(output, result);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
            usage(argv[0]);
            return 0;
        }
        const options args = parse(argc, argv);
        levo_cli::install_signal_handlers();
        const std::array<cantor_component, 3> components{{
            {"lm", args.lm.c_str()}, {"dit", args.dit.c_str()}, {"vae", args.vae.c_str()},
        }};
        cantor_ctx * context = cantor_engine_load(components.data(), components.size(), nullptr);
        if (context == nullptr) fail(cantor_engine_last_error());
        std::vector<std::uint8_t> state = read_bytes(args.input);
        cantor_stage stage = stage_for(state);
        progress_state progress;
        while (true) {
            std::cerr << '[' << stage_name(stage) << "] starting\n";
            std::uint8_t * output = nullptr;
            std::size_t output_size = 0;
            const cantor_status status = cantor_engine_run_stage(
                context, stage, state.data(), state.size(), &output, &output_size,
                on_progress, should_cancel, &progress);
            if (status == CANTOR_ERR) {
                const std::string message = cantor_engine_last_error();
                cantor_engine_free(context);
                fail(message);
            }
            if (stage == CANTOR_STAGE_DECODE) {
                if (status == CANTOR_PAUSED) {
                    if (output != nullptr || output_size != 0) {
                        cantor_engine_free_blob(output);
                        cantor_engine_free(context);
                        fail("DECODE pause unexpectedly returned a blob");
                    }
                    std::cerr << "paused; durable checkpoint remains " << args.checkpoint << '\n';
                    cantor_engine_free(context);
                    return 130;
                }
                if (output != nullptr) cantor_engine_free_blob(output);
                write_audio(args.output, context);
                std::cout << "wrote " << args.output << '\n';
                cantor_engine_free(context);
                return 0;
            }
            state = copy_engine_blob(output, output_size);
            if (status == CANTOR_PAUSED) {
                write_checkpoint(args.checkpoint, state);
                std::cerr << "paused; wrote " << args.checkpoint << '\n';
                cantor_engine_free(context);
                return 130;
            }
            if (stage == CANTOR_STAGE_DIFFUSE) {
                // This is the precise durable input DECODE needs if it later
                // pauses with its permitted NULL blob.
                write_checkpoint(args.checkpoint, state);
            }
            stage = stage == CANTOR_STAGE_CODES ? CANTOR_STAGE_DIFFUSE : CANTOR_STAGE_DECODE;
        }
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
