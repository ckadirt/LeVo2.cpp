#include "levo.h"

#include <cassert>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename Function>
void expect_throw(Function && function) {
    bool thrown = false;
    try { function(); } catch (const std::exception &) { thrown = true; }
    assert(thrown);
}

} // namespace

int main() {
    using namespace levo;
    const auto root = std::filesystem::temp_directory_path() / "levo2-render-public-test";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    std::filesystem::create_directories(root);

    // The public WAV hand-off stays usable without any model asset.
    render_result wave;
    wave.samples_per_channel = 2;
    wave.interleaved_stereo = {0.0F, 0.25F, -0.5F, 1.0F};
    const auto wav = root / "result.wav";
    write_render_wav(wav, wave);
    assert(std::filesystem::file_size(wav) == 44U + wave.interleaved_stereo.size() * sizeof(float));
    wave.interleaved_stereo.pop_back();
    expect_throw([&] { write_render_wav(root / "bad.wav", wave); });

    render_config missing;
    expect_throw([&] { (void) render_tokens_to_audio(missing); });
    missing.tokens_path = root / "tokens.raw";
    missing.flow_model_path = root / "flow.gguf";
    missing.vae_model_path = root / "vae.gguf";
    expect_throw([&] { (void) render_tokens_to_audio(missing); });
    std::filesystem::remove_all(root, ignored);
    std::cout << "render public api ok\n";
    return 0;
}
