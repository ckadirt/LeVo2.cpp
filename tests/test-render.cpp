#include "levo.h"

#include <cassert>
#include <exception>
#include <filesystem>
#include <fstream>
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

    render_config artifact_config;
    artifact_config.tokens_path = root / "source.tokens.npy";
    wave.source_frames = 2;
    wave.rendered_windows = 1;
    wave.euler_steps = 50;
    wave.cfg_scale = 1.5F;
    wave.provenance.backend_name = "CPU";
    wave.provenance.token_tensor_sha256 = "token-hash";
    wave.provenance.seed = 42;
    const auto artifact_wav = root / "artifact.wav";
    const render_artifact_info info = write_render_artifact(artifact_wav, wave, artifact_config);
    assert(info.wav_path == artifact_wav);
    assert(info.metadata_path == root / "artifact.wav.json");
    assert(info.wav_bytes == 44U + wave.interleaved_stereo.size() * sizeof(float));
    assert(info.wav_sha256.size() == 64);
    assert(std::filesystem::exists(artifact_wav) && std::filesystem::exists(info.metadata_path));
    std::ifstream metadata(info.metadata_path);
    const std::string json{std::istreambuf_iterator<char>(metadata), std::istreambuf_iterator<char>()};
    assert(json.find("\"format\": \"levo2-render-artifact\"") != std::string::npos);
    assert(json.find("\"sha256\": \"" + info.wav_sha256 + "\"") != std::string::npos);
    assert(json.find("\"backend\": \"CPU\"") != std::string::npos);
    assert(json.find("\"euler_steps\": 50") != std::string::npos);

    wave.interleaved_stereo.pop_back();
    expect_throw([&] { write_render_wav(root / "bad.wav", wave); });
    const auto failed_artifact = root / "failed.wav";
    expect_throw([&] { (void) write_render_artifact(failed_artifact, wave, artifact_config); });
    assert(!std::filesystem::exists(failed_artifact));
    assert(!std::filesystem::exists(render_metadata_path(failed_artifact)));

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
