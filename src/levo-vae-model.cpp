#include "levo-vae-model.h"

#include "levo-token-io.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace levo::detail {
namespace {

constexpr const char * k_architecture = "general.architecture";
constexpr const char * k_name = "general.name";
constexpr const char * k_file_type = "general.file_type";
constexpr const char * k_converter = "levo2.converter";
constexpr const char * k_schema_version = "levo2.schema_version";
constexpr const char * k_runtime_repository = "levo2.source.runtime_repository";
constexpr const char * k_runtime_revision = "levo2.source.runtime_revision";
constexpr const char * k_levo_repository = "levo2.source.levo_repository";
constexpr const char * k_levo_revision = "levo2.source.levo_revision";
constexpr const char * k_ggml_repository = "levo2.source.ggml_repository";
constexpr const char * k_ggml_revision = "levo2.source.ggml_revision";
constexpr const char * k_checkpoint_sha256 = "levo2.source.checkpoint_sha256";
constexpr const char * k_config_sha256 = "levo2.source.config_sha256";

struct context_deleter { void operator()(ggml_context * p) const noexcept { if (p != nullptr) ggml_free(p); } };
struct buffer_deleter { void operator()(ggml_backend_buffer_t p) const noexcept { if (p != nullptr) ggml_backend_buffer_free(p); } };
struct gguf_deleter { void operator()(gguf_context * p) const noexcept { if (p != nullptr) gguf_free(p); } };
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;

[[noreturn]] void fail(const std::string & message) { throw std::runtime_error("LeVo VAE GGUF: " + message); }
std::string quote(const std::string & text) { return "'" + text + "'"; }

int64_t required_key(const gguf_context * context, const char * key, gguf_type type) {
    const int64_t index = gguf_find_key(context, key);
    if (index < 0) fail("missing required metadata key " + quote(key));
    if (gguf_get_kv_type(context, index) != type) {
        fail("metadata key " + quote(key) + " has incorrect type");
    }
    return index;
}

uint32_t required_u32(const gguf_context * context, const char * key) {
    return gguf_get_val_u32(context, required_key(context, key, GGUF_TYPE_UINT32));
}
bool required_bool(const gguf_context * context, const char * key) {
    return gguf_get_val_bool(context, required_key(context, key, GGUF_TYPE_BOOL));
}
std::string required_string(const gguf_context * context, const char * key) {
    return gguf_get_val_str(context, required_key(context, key, GGUF_TYPE_STRING));
}

std::array<int32_t, 5> required_i32_array_5(const gguf_context * context, const char * key) {
    const int64_t index = required_key(context, key, GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(context, index) != GGUF_TYPE_INT32 || gguf_get_arr_n(context, index) != 5U) {
        fail("metadata array " + quote(key) + " must contain exactly five int32 values");
    }
    const auto * values = static_cast<const int32_t *>(gguf_get_arr_data(context, index));
    std::array<int32_t, 5> result{};
    std::copy(values, values + result.size(), result.begin());
    return result;
}

void expect_equal(const char * key, int32_t actual, int32_t expected) {
    if (actual != expected) {
        std::ostringstream out; out << "metadata " << quote(key) << " is " << actual << ", expected " << expected; fail(out.str());
    }
}
void require_sha256(const std::string & value, const char * key) {
    if (value.size() != 64U || !std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c) != 0; })) {
        fail("metadata " + quote(key) + " must be a SHA-256 hex string");
    }
}

vae_hparams parse_hparams(const gguf_context * context) {
    if (required_string(context, k_architecture) != "levo2_vae") fail("general.architecture must be 'levo2_vae'");
    if (required_string(context, k_name).empty()) fail("general.name must not be empty");
    if (required_string(context, k_converter) != "convert_vae.py") fail("levo2.converter must be 'convert_vae.py'");
    expect_equal(k_schema_version, static_cast<int32_t>(required_u32(context, k_schema_version)), 1);

    vae_hparams result;
    result.sample_rate = static_cast<int32_t>(required_u32(context, "levo2.vae.sample_rate"));
    result.audio_channels = static_cast<int32_t>(required_u32(context, "levo2.vae.audio_channels"));
    result.latent_dim = static_cast<int32_t>(required_u32(context, "levo2.vae.latent_dim"));
    result.channels = static_cast<int32_t>(required_u32(context, "levo2.vae.channels"));
    result.downsampling_ratio = static_cast<int32_t>(required_u32(context, "levo2.vae.downsampling_ratio"));
    result.stage_count = static_cast<int32_t>(required_u32(context, "levo2.vae.decoder.stage_count"));
    result.residual_units_per_stage = static_cast<int32_t>(required_u32(context, "levo2.vae.decoder.residual_units_per_stage"));
    result.c_mults = required_i32_array_5(context, "levo2.vae.c_mults");
    result.strides = required_i32_array_5(context, "levo2.vae.strides");
    result.use_snake_beta = required_bool(context, "levo2.vae.use_snake_beta");
    result.final_tanh = required_bool(context, "levo2.vae.final_tanh");
    result.soft_clip = required_bool(context, "levo2.vae.soft_clip");
    result.weight_norm_folded = required_bool(context, "levo2.vae.weight_norm_folded");
    std::string reason;
    if (!result.valid(&reason)) fail("invalid VAE metadata: " + reason);
    return result;
}

vae_provenance parse_provenance(const gguf_context * context, bool require_pinned) {
    vae_provenance result;
    result.name = required_string(context, k_name);
    result.runtime_repository = required_string(context, k_runtime_repository);
    result.runtime_revision = required_string(context, k_runtime_revision);
    result.levo_repository = required_string(context, k_levo_repository);
    result.levo_revision = required_string(context, k_levo_revision);
    result.ggml_repository = required_string(context, k_ggml_repository);
    result.ggml_revision = required_string(context, k_ggml_revision);
    result.checkpoint_sha256 = required_string(context, k_checkpoint_sha256);
    result.config_sha256 = required_string(context, k_config_sha256);
    require_sha256(result.checkpoint_sha256, k_checkpoint_sha256);
    require_sha256(result.config_sha256, k_config_sha256);
    if (require_pinned) {
        if (result.runtime_repository != "lglg666/SongGeneration-Runtime" ||
            result.runtime_revision != "cc258cc694a63114c61684cc26d0583b8ad777d0" ||
            result.levo_repository != "https://github.com/levo-demo/LeVo" ||
            result.levo_revision != "653cbcf4716101834900c75b7d5da43b07e15d5b" ||
            result.ggml_repository != "https://github.com/ggml-org/ggml" ||
            result.ggml_revision != "8846b79e66747bb9f68597420e95114c177315ce" ||
            result.checkpoint_sha256 != "10ccb6c83613781ad32e998a90597ba7eb9292911a224598da1fd53728eb4cd3" ||
            result.config_sha256 != "5cd2859efe00bc2b0f6f9bdac738ad11822a36473d6d810427b60efd057c538b") {
            fail("source provenance does not match the pinned released VAE");
        }
    }
    return result;
}

std::string stage_name(unsigned stage, const char * suffix) {
    return "vae.decoder.stage." + std::to_string(stage) + "." + suffix;
}

void validate_tensor(const gguf_context * context, const std::string & name, const std::vector<int64_t> & expected, ggml_type type) {
    const int64_t id = gguf_find_tensor(context, name.c_str());
    if (id < 0) fail("missing required tensor " + quote(name));
    if (gguf_get_tensor_type(context, id) != type) fail("tensor " + quote(name) + " has type inconsistent with general.file_type");
    const int64_t * dimensions = gguf_get_tensor_ne(context, id);
    for (std::size_t i = 0; i < expected.size(); ++i) if (dimensions[i] != expected[i]) fail("tensor " + quote(name) + " has an unexpected shape");
    for (std::size_t i = expected.size(); i < GGML_MAX_DIMS; ++i) if (dimensions[i] != 1) fail("tensor " + quote(name) + " has an unexpected rank");
}

std::unordered_set<std::string> validate_inventory(const gguf_context * context, const vae_hparams & hparams) {
    const uint32_t file_type = required_u32(context, k_file_type);
    if (file_type != 0U && file_type != 1U) fail("general.file_type must be F32 (0) or F16 (1)");
    const ggml_type type = file_type == 0U ? GGML_TYPE_F32 : GGML_TYPE_F16;
    std::unordered_set<std::string> expected;
    const auto add = [&](const std::string & name, std::vector<int64_t> shape) {
        if (!expected.emplace(name).second) fail("internal duplicate expected tensor " + quote(name));
        validate_tensor(context, name, shape, type);
    };
    const auto snake = [&](const std::string & name, int32_t width) {
        add(name + ".alpha_log", {width}); add(name + ".beta_log", {width});
    };
    const auto conv = [&](const std::string & name, int32_t kernel, int32_t second, int32_t first, int32_t bias_width) {
        // GGUF stores NumPy [first, second, kernel] as GGML [kernel, second, first].
        add(name + ".weight", {kernel, second, first}); if (bias_width >= 0) add(name + ".bias", {bias_width});
    };

    conv("vae.decoder.input", 7, hparams.latent_dim, 2048, 2048);
    const std::array<int32_t, 5> input_widths{{2048, 1024, 512, 256, 128}};
    const std::array<int32_t, 5> output_widths{{1024, 512, 256, 128, 128}};
    const std::array<int32_t, 5> decoder_strides{{10, 6, 4, 4, 2}};
    for (unsigned stage = 0; stage < 5U; ++stage) {
        const std::string base = stage_name(stage, "");
        snake(base + "upsample.activation", input_widths[stage]);
        // ConvTranspose source layout [Cin, Cout, K].  Its folded weights are
        // intentionally not transposed by the converter.
        conv(base + "upsample", 2 * decoder_strides[stage], output_widths[stage], input_widths[stage], output_widths[stage]);
        for (unsigned residual = 0; residual < 3U; ++residual) {
            const std::string r = base + "residual." + std::to_string(residual) + ".";
            snake(r + "pre", output_widths[stage]);
            conv(r + "conv.0", 7, output_widths[stage], output_widths[stage], output_widths[stage]);
            snake(r + "post", output_widths[stage]);
            conv(r + "conv.1", 1, output_widths[stage], output_widths[stage], output_widths[stage]);
        }
    }
    snake("vae.decoder.output", 128);
    conv("vae.decoder.output", 7, 128, hparams.audio_channels, -1);

    if (expected.size() != 145U) fail("internal VAE tensor inventory does not contain 145 tensors");
    if (static_cast<std::size_t>(gguf_get_n_tensors(context)) != expected.size()) {
        std::ostringstream out; out << "GGUF contains " << gguf_get_n_tensors(context) << " tensors; decoder requires exactly 145"; fail(out.str());
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(context); ++i) {
        const std::string name = gguf_get_tensor_name(context, i);
        if (expected.find(name) == expected.end()) fail("GGUF contains unrecognized tensor " + quote(name));
    }
    return expected;
}

std::size_t file_size_or_fail(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) fail("cannot open " + quote(path));
    const std::streamoff size = file.tellg();
    if (size < 0 || static_cast<uintmax_t>(size) > std::numeric_limits<std::size_t>::max()) fail("cannot determine file size");
    return static_cast<std::size_t>(size);
}

void validate_file_bounds(const gguf_context * context, std::size_t file_size) {
    const std::size_t alignment = gguf_get_alignment(context);
    const std::size_t data_offset = gguf_get_data_offset(context);
    if (alignment == 0 || (alignment & (alignment - 1U)) != 0U || data_offset > file_size || data_offset % alignment != 0U) fail("invalid GGUF data alignment or offset");
    std::size_t expected_offset = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(context); ++i) {
        const std::size_t offset = gguf_get_tensor_offset(context, i);
        const std::size_t size = gguf_get_tensor_size(context, i);
        if (offset != expected_offset || offset % alignment != 0U || offset > file_size - data_offset || size > file_size - data_offset - offset) fail("invalid tensor data bounds");
        if (size > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) fail("tensor size overflow");
        expected_offset += (size + alignment - 1U) & ~(alignment - 1U);
    }
    if (expected_offset > file_size - data_offset) fail("truncated GGUF tensor data");
}

void read_payloads(const std::string & path, const gguf_context * gguf, const std::unordered_map<std::string, ggml_tensor *> & tensors) {
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("cannot reopen " + quote(path));
    std::vector<char> buffer(8U * 1024U * 1024U);
    const std::size_t data_offset = gguf_get_data_offset(gguf);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const std::string name = gguf_get_tensor_name(gguf, i);
        const auto destination = tensors.find(name);
        if (destination == tensors.end()) fail("internal destination tensor is missing");
        const std::size_t offset = data_offset + gguf_get_tensor_offset(gguf, i);
        const std::size_t size = gguf_get_tensor_size(gguf, i);
        file.clear(); file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!file) fail("cannot seek to tensor " + quote(name));
        for (std::size_t copied = 0; copied < size;) {
            const std::size_t count = std::min(buffer.size(), size - copied);
            file.read(buffer.data(), static_cast<std::streamsize>(count));
            if (file.gcount() != static_cast<std::streamsize>(count)) fail("truncated tensor payload " + quote(name));
            ggml_backend_tensor_set(destination->second, buffer.data(), copied, count);
            copied += count;
        }
    }
}

std::size_t checked_elements(const vae_test_tensor_data & data, const std::string & name) {
    if (data.dimensions.empty() || data.dimensions.size() > GGML_MAX_DIMS) fail("test tensor " + quote(name) + " has invalid rank");
    std::size_t count = 1;
    for (int64_t dimension : data.dimensions) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<std::size_t>::max() / count) fail("test tensor " + quote(name) + " has invalid shape");
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

template <typename T> void bind_snake(T & snake, const std::unordered_map<std::string, ggml_tensor *> & tensors, const std::string & base) {
    snake.alpha_log = tensors.at(base + ".alpha_log"); snake.beta_log = tensors.at(base + ".beta_log");
}
void bind_conv(vae_conv_weights & conv, const std::unordered_map<std::string, ggml_tensor *> & tensors, const std::string & base, bool bias) {
    conv.weight = tensors.at(base + ".weight"); conv.bias = bias ? tensors.at(base + ".bias") : nullptr;
}
void bind_decoder(vae_decoder_weights & decoder, const std::unordered_map<std::string, ggml_tensor *> & tensors) {
    bind_conv(decoder.input, tensors, "vae.decoder.input", true);
    for (unsigned stage = 0; stage < 5U; ++stage) {
        const std::string base = stage_name(stage, "");
        bind_snake(decoder.stage[stage].upsample.activation, tensors, base + "upsample.activation");
        bind_conv(decoder.stage[stage].upsample.conv, tensors, base + "upsample", true);
        for (unsigned residual = 0; residual < 3U; ++residual) {
            const std::string r = base + "residual." + std::to_string(residual) + ".";
            bind_snake(decoder.stage[stage].residual[residual].pre, tensors, r + "pre");
            bind_conv(decoder.stage[stage].residual[residual].conv_0, tensors, r + "conv.0", true);
            bind_snake(decoder.stage[stage].residual[residual].post, tensors, r + "post");
            bind_conv(decoder.stage[stage].residual[residual].conv_1, tensors, r + "conv.1", true);
        }
    }
    bind_snake(decoder.output_activation, tensors, "vae.decoder.output");
    decoder.output_weight = tensors.at("vae.decoder.output.weight");
}

} // namespace

struct vae_model::impl {
    vae_hparams hparams;
    vae_decoder_weights decoder;
    vae_provenance provenance;
    ggml_backend_t backend = nullptr;
    context_ptr context;
    buffer_ptr buffer;
    std::unordered_map<std::string, ggml_tensor *> tensors;
};

bool vae_hparams::valid(std::string * reason) const {
    const auto invalid = [reason](const char * message) { if (reason != nullptr) *reason = message; return false; };
    if (sample_rate != 48000 || audio_channels != 2 || latent_dim != 64 || channels != 128 || downsampling_ratio != 1920) return invalid("base dimensions do not match stable_audio_1920");
    if (c_mults != std::array<int32_t, 5>{{1, 2, 4, 8, 16}} || strides != std::array<int32_t, 5>{{2, 4, 4, 6, 10}}) return invalid("channel multipliers or strides do not match stable_audio_1920");
    if (stage_count != 5 || residual_units_per_stage != 3 || !use_snake_beta || final_tanh || soft_clip || !weight_norm_folded) return invalid("decoder topology flags do not match the strict release contract");
    return true;
}

vae_model::vae_model() : impl_(std::make_unique<impl>()) {}
vae_model::vae_model(vae_model &&) noexcept = default;
vae_model & vae_model::operator=(vae_model &&) noexcept = default;
vae_model::~vae_model() = default;

std::shared_ptr<vae_model> vae_model::load_gguf(const std::string & path, const vae_model_load_options & options) {
    if (options.backend == nullptr) fail("a GGML backend is required to load weights");
    const std::size_t file_size = file_size_or_fail(path);
    ggml_context * source_raw = nullptr;
    const gguf_init_params params = {true, &source_raw};
    gguf_ptr gguf(gguf_init_from_file(path.c_str(), params));
    context_ptr source(source_raw);
    if (!gguf || !source) fail("cannot parse GGUF " + quote(path));
    if (gguf_get_version(gguf.get()) != GGUF_VERSION || gguf_get_n_tensors(gguf.get()) <= 0) fail("unsupported or empty GGUF");
    validate_file_bounds(gguf.get(), file_size);
    const vae_hparams hparams = parse_hparams(gguf.get());
    vae_provenance provenance = parse_provenance(gguf.get(), options.require_pinned_provenance);
    provenance.artifact_sha256 = token_io::file_sha256(path);
    (void) validate_inventory(gguf.get(), hparams);

    const auto result = std::shared_ptr<vae_model>(new vae_model());
    result->impl_->hparams = hparams; result->impl_->provenance = provenance; result->impl_->backend = options.backend;
    const int64_t tensor_count = gguf_get_n_tensors(gguf.get());
    if (static_cast<uint64_t>(tensor_count) > std::numeric_limits<std::size_t>::max() / ggml_tensor_overhead()) fail("tensor context size overflows");
    result->impl_->context.reset(ggml_init({static_cast<std::size_t>(tensor_count) * ggml_tensor_overhead(), nullptr, true}));
    if (!result->impl_->context) fail("cannot create GGML weight context");
    for (int64_t i = 0; i < tensor_count; ++i) {
        const std::string name = gguf_get_tensor_name(gguf.get(), i);
        const ggml_type type = gguf_get_tensor_type(gguf.get(), i);
        if ((type == GGML_TYPE_F32 && !options.allow_f32) || (type == GGML_TYPE_F16 && !options.allow_f16) || (type != GGML_TYPE_F32 && type != GGML_TYPE_F16)) fail("unsupported tensor type for " + quote(name));
        const ggml_tensor * source_tensor = ggml_get_tensor(source.get(), name.c_str());
        if (source_tensor == nullptr || source_tensor->type != type) fail("GGML tensor metadata is inconsistent for " + quote(name));
        ggml_tensor * destination = ggml_dup_tensor(result->impl_->context.get(), source_tensor);
        if (destination == nullptr || !result->impl_->tensors.emplace(name, destination).second) fail("cannot create unique destination tensor " + quote(name));
        ggml_set_name(destination, name.c_str());
    }
    result->impl_->buffer.reset(ggml_backend_alloc_ctx_tensors(result->impl_->context.get(), options.backend));
    if (!result->impl_->buffer) fail("backend cannot allocate VAE GGUF weights");
    read_payloads(path, gguf.get(), result->impl_->tensors);
    ggml_backend_synchronize(options.backend);
    bind_decoder(result->impl_->decoder, result->impl_->tensors);
    return result;
}

std::shared_ptr<vae_model> vae_model::make_test_model(const vae_hparams & hparams, const std::unordered_map<std::string, vae_test_tensor_data> & tensors, ggml_backend_t backend) {
    if (backend == nullptr) fail("a GGML backend is required to create a test VAE model");
    std::string reason; if (!hparams.valid(&reason)) fail("invalid test VAE hyperparameters: " + reason);
    if (tensors.empty() || tensors.size() > std::numeric_limits<std::size_t>::max() / ggml_tensor_overhead()) fail("invalid test VAE tensor inventory");
    const auto result = std::shared_ptr<vae_model>(new vae_model());
    result->impl_->hparams = hparams; result->impl_->backend = backend;
    result->impl_->context.reset(ggml_init({tensors.size() * ggml_tensor_overhead(), nullptr, true}));
    if (!result->impl_->context) fail("cannot create test VAE context");
    for (const auto & item : tensors) {
        const std::size_t count = checked_elements(item.second, item.first);
        if (item.second.values.size() != count) fail("test tensor payload size does not match shape");
        ggml_tensor * tensor = ggml_new_tensor(result->impl_->context.get(), GGML_TYPE_F32, static_cast<int>(item.second.dimensions.size()), item.second.dimensions.data());
        if (tensor == nullptr || !result->impl_->tensors.emplace(item.first, tensor).second) fail("cannot create unique test tensor " + quote(item.first));
        ggml_set_name(tensor, item.first.c_str());
    }
    result->impl_->buffer.reset(ggml_backend_alloc_ctx_tensors(result->impl_->context.get(), backend));
    if (!result->impl_->buffer) fail("backend cannot allocate test VAE weights");
    for (const auto & item : tensors) ggml_backend_tensor_set(result->impl_->tensors.at(item.first), item.second.values.data(), 0, item.second.values.size() * sizeof(float));
    ggml_backend_synchronize(backend);
    return result;
}

const vae_hparams & vae_model::hparams() const noexcept { return impl_->hparams; }
const vae_decoder_weights & vae_model::decoder() const noexcept { return impl_->decoder; }
const vae_provenance & vae_model::provenance() const noexcept { return impl_->provenance; }
ggml_backend_t vae_model::backend() const noexcept { return impl_->backend; }
ggml_tensor * vae_model::tensor(const std::string & name) const { const auto found = impl_->tensors.find(name); return found == impl_->tensors.end() ? nullptr : found->second; }
bool vae_model::contains(const std::string & name) const noexcept { return impl_->tensors.find(name) != impl_->tensors.end(); }
std::size_t vae_model::tensor_count() const noexcept { return impl_->tensors.size(); }

} // namespace levo::detail
