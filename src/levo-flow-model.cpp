#include "levo-flow-model.h"

#include "levo-quantization.h"
#include "levo-token-io.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace levo::flow {
namespace {

constexpr const char * k_architecture = "general.architecture";
constexpr const char * k_name = "general.name";
constexpr const char * k_file_type = "general.file_type";
constexpr const char * k_schema = "levo2.schema_version";
constexpr const char * k_converter = "levo2.converter";
constexpr const char * k_runtime_repo = "levo2.source.runtime_repository";
constexpr const char * k_runtime_revision = "levo2.source.runtime_revision";
constexpr const char * k_levo_repo = "levo2.source.levo_repository";
constexpr const char * k_levo_revision = "levo2.source.levo_revision";
constexpr const char * k_ggml_repo = "levo2.source.ggml_repository";
constexpr const char * k_ggml_revision = "levo2.source.ggml_revision";
constexpr const char * k_model_sha256 = "levo2.source.model_sha256";
constexpr const char * k_parameter_dtype = "levo2.flow.parameter_dtype";

struct context_deleter { void operator()(ggml_context * p) const noexcept { if (p) ggml_free(p); } };
struct buffer_deleter { void operator()(ggml_backend_buffer_t p) const noexcept { if (p) ggml_backend_buffer_free(p); } };
struct gguf_deleter { void operator()(gguf_context * p) const noexcept { if (p) gguf_free(p); } };
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;

[[noreturn]] void fail(const std::string & value) { throw std::runtime_error("LeVo Flow GGUF: " + value); }
std::string quote(const std::string & value) { return "'" + value + "'"; }

int64_t key(const gguf_context * ctx, const char * name, gguf_type type) {
    const int64_t index = gguf_find_key(ctx, name);
    if (index < 0) fail("missing required metadata key " + quote(name));
    if (gguf_get_kv_type(ctx, index) != type) {
        fail("metadata key " + quote(name) + " has type " + gguf_type_name(gguf_get_kv_type(ctx, index)) +
             ", expected " + gguf_type_name(type));
    }
    return index;
}
uint32_t u32(const gguf_context * ctx, const char * name) { return gguf_get_val_u32(ctx, key(ctx, name, GGUF_TYPE_UINT32)); }
float f32(const gguf_context * ctx, const char * name) { return gguf_get_val_f32(ctx, key(ctx, name, GGUF_TYPE_FLOAT32)); }
bool boolean(const gguf_context * ctx, const char * name) { return gguf_get_val_bool(ctx, key(ctx, name, GGUF_TYPE_BOOL)); }
std::string str(const gguf_context * ctx, const char * name) { return gguf_get_val_str(ctx, key(ctx, name, GGUF_TYPE_STRING)); }

void equal(const char * name, int32_t got, int32_t wanted) {
    if (got != wanted) fail("metadata " + quote(name) + " is " + std::to_string(got) + ", expected " + std::to_string(wanted));
}
void equal(const char * name, float got, float wanted) {
    uint32_t a = 0, b = 0;
    std::memcpy(&a, &got, sizeof(a)); std::memcpy(&b, &wanted, sizeof(b));
    if (a != b) fail("metadata " + quote(name) + " does not match the Flow schema");
}
std::string flow_key(const char * suffix) { return "levo2.flow." + std::string(suffix); }

std::optional<quantization::profile> parse_quantization(const gguf_context * ctx) {
    const int64_t profile = gguf_find_key(ctx, quantization::profile_key);
    const int64_t policy = gguf_find_key(ctx, quantization::policy_revision_key);
    const int64_t source = gguf_find_key(ctx, quantization::source_artifact_sha256_key);
    const int64_t padding = gguf_find_key(ctx, quantization::flow_padded_input_key);
    const bool any = profile >= 0 || policy >= 0 || source >= 0 || padding >= 0;
    if (!any) return std::nullopt;
    if (profile < 0 || policy < 0 || source < 0 || padding < 0) fail("quantized Flow metadata is incomplete");
    const auto parsed = quantization::parse(str(ctx, quantization::profile_key));
    if (!parsed) fail("quantized Flow profile is unsupported");
    if (str(ctx, quantization::policy_revision_key) != quantization::policy_revision) fail("quantized Flow policy revision is unsupported");
    if (!quantization::is_hex_sha256(str(ctx, quantization::source_artifact_sha256_key))) fail("quantized Flow source artifact digest is invalid");
    if (u32(ctx, k_file_type) != quantization::gguf_file_type(*parsed)) fail("quantized Flow general.file_type disagrees with profile");
    if (str(ctx, quantization::flow_padded_input_key) != quantization::flow_padded_input_layout(*parsed)) {
        fail("quantized Flow padded-input layout is unsupported");
    }
    return parsed;
}

flow_hparams parse_hparams(const gguf_context * ctx) {
    if (str(ctx, k_architecture) != "levo2_flow") fail("general.architecture must be 'levo2_flow'");
    if (str(ctx, k_name).empty() || str(ctx, k_converter) != "convert_flow.py") fail("general.name must not be empty and levo2.converter must be 'convert_flow.py'");
    equal(k_schema, static_cast<int32_t>(u32(ctx, k_schema)), 1);
    flow_hparams hp;
    hp.hidden_size = static_cast<int32_t>(u32(ctx, flow_key("hidden_size").c_str()));
    hp.n_layer = static_cast<int32_t>(u32(ctx, flow_key("n_layer").c_str()));
    hp.n_head = static_cast<int32_t>(u32(ctx, flow_key("n_head").c_str()));
    hp.head_dim = static_cast<int32_t>(u32(ctx, flow_key("head_dim").c_str()));
    hp.intermediate_size = static_cast<int32_t>(u32(ctx, flow_key("intermediate_size").c_str()));
    hp.max_frames = static_cast<int32_t>(u32(ctx, flow_key("max_frames").c_str()));
    hp.codebook_size = static_cast<int32_t>(u32(ctx, flow_key("codebook_size").c_str()));
    hp.codebook_dim = static_cast<int32_t>(u32(ctx, flow_key("codebook_dim").c_str()));
    hp.condition_dim = static_cast<int32_t>(u32(ctx, flow_key("condition_dim").c_str()));
    hp.latent_dim = static_cast<int32_t>(u32(ctx, flow_key("latent_dim").c_str()));
    hp.mask_dim = static_cast<int32_t>(u32(ctx, flow_key("mask_dim").c_str()));
    hp.time_embedding_dim = static_cast<int32_t>(u32(ctx, flow_key("time_embedding_dim").c_str()));
    hp.euler_steps_default = static_cast<int32_t>(u32(ctx, flow_key("euler_steps_default").c_str()));
    hp.sample_rate = static_cast<int32_t>(u32(ctx, flow_key("sample_rate").c_str()));
    hp.frame_rate = static_cast<int32_t>(u32(ctx, flow_key("frame_rate").c_str()));
    hp.window_frames = static_cast<int32_t>(u32(ctx, flow_key("window_frames").c_str()));
    hp.hop_frames = static_cast<int32_t>(u32(ctx, flow_key("hop_frames").c_str()));
    hp.overlap_frames = static_cast<int32_t>(u32(ctx, flow_key("overlap_frames").c_str()));
    hp.cfg_default = f32(ctx, flow_key("cfg_default").c_str());
    hp.rope_theta = f32(ctx, flow_key("rope_theta").c_str());
    hp.time_embedding_scale = f32(ctx, flow_key("time_embedding_scale").c_str());
    hp.sigma_min = f32(ctx, flow_key("sigma_min").c_str());
    hp.rvq_weight_norm_folded = boolean(ctx, flow_key("rvq_weight_norm_folded").c_str());
    std::string reason;
    if (!hp.valid(&reason)) fail("invalid Flow metadata: " + reason);
    return hp;
}

void validate_pinned(const gguf_context * ctx, const flow_hparams & hp,
                     const std::optional<quantization::profile> & profile) {
    const auto expect = [ctx](const char * name, const char * value) {
        if (str(ctx, name) != value) fail("metadata " + quote(name) + " does not match pinned Flow source");
    };
    expect(k_runtime_repo, "lglg666/SongGeneration-Runtime");
    expect(k_runtime_revision, "cc258cc694a63114c61684cc26d0583b8ad777d0");
    expect(k_levo_repo, "https://github.com/levo-demo/LeVo");
    expect(k_levo_revision, "653cbcf4716101834900c75b7d5da43b07e15d5b");
    expect(k_ggml_repo, "https://github.com/ggml-org/ggml");
    expect(k_ggml_revision, "8846b79e66747bb9f68597420e95114c177315ce");
    expect(k_model_sha256, "430b7c1c245722fbe3893cd621b3d4a90076404596e9fb1ce987a4a0f2a4fc6f");
    const flow_hparams expected;
    equal("levo2.flow.hidden_size", hp.hidden_size, expected.hidden_size);
    equal("levo2.flow.n_layer", hp.n_layer, expected.n_layer);
    equal("levo2.flow.n_head", hp.n_head, expected.n_head);
    equal("levo2.flow.head_dim", hp.head_dim, expected.head_dim);
    equal("levo2.flow.intermediate_size", hp.intermediate_size, expected.intermediate_size);
    equal("levo2.flow.max_frames", hp.max_frames, expected.max_frames);
    equal("levo2.flow.codebook_size", hp.codebook_size, expected.codebook_size);
    equal("levo2.flow.codebook_dim", hp.codebook_dim, expected.codebook_dim);
    equal("levo2.flow.condition_dim", hp.condition_dim, expected.condition_dim);
    equal("levo2.flow.latent_dim", hp.latent_dim, expected.latent_dim);
    equal("levo2.flow.mask_dim", hp.mask_dim, expected.mask_dim);
    equal("levo2.flow.time_embedding_dim", hp.time_embedding_dim, expected.time_embedding_dim);
    equal("levo2.flow.euler_steps_default", hp.euler_steps_default, expected.euler_steps_default);
    equal("levo2.flow.sample_rate", hp.sample_rate, expected.sample_rate);
    equal("levo2.flow.frame_rate", hp.frame_rate, expected.frame_rate);
    equal("levo2.flow.window_frames", hp.window_frames, expected.window_frames);
    equal("levo2.flow.hop_frames", hp.hop_frames, expected.hop_frames);
    equal("levo2.flow.overlap_frames", hp.overlap_frames, expected.overlap_frames);
    equal("levo2.flow.cfg_default", hp.cfg_default, expected.cfg_default);
    equal("levo2.flow.rope_theta", hp.rope_theta, expected.rope_theta);
    equal("levo2.flow.time_embedding_scale", hp.time_embedding_scale, expected.time_embedding_scale);
    equal("levo2.flow.sigma_min", hp.sigma_min, expected.sigma_min);
    if (!hp.rvq_weight_norm_folded) fail("levo2.flow.rvq_weight_norm_folded must be true");
    const uint32_t file_type = u32(ctx, k_file_type);
    const std::string dtype = str(ctx, k_parameter_dtype);
    if (profile) {
        if (file_type != quantization::gguf_file_type(*profile) || dtype != "MIXED") {
            fail("quantized Flow general.file_type and parameter dtype disagree");
        }
    } else if ((file_type == 0 && dtype != "F32") || (file_type == 1 && dtype != "F16") || (file_type != 0 && file_type != 1)) {
        fail("general.file_type and levo2.flow.parameter_dtype disagree");
    }
}

void tensor(const gguf_context * ctx, const std::string & name, const std::vector<int64_t> & numpy_shape,
            ggml_type type,
            std::unordered_set<std::string> & expected) {
    if (!expected.emplace(name).second) fail("internal duplicate expected tensor " + quote(name));
    const int64_t index = gguf_find_tensor(ctx, name.c_str());
    if (index < 0) fail("missing required tensor " + quote(name));
    if (gguf_get_tensor_type(ctx, index) != type) fail("tensor " + quote(name) + " has an inconsistent type");
    const int64_t * shape = gguf_get_tensor_ne(ctx, index);
    for (std::size_t i = 0; i < numpy_shape.size(); ++i) {
        int64_t dimension = numpy_shape[numpy_shape.size() - 1 - i];
        if (i == 0 && quantization::flow_block_matrix(name)) {
            dimension = quantization::padded_input_columns(dimension, type);
        }
        if (shape[i] != dimension) fail("tensor " + quote(name) + " has an unexpected shape");
    }
    for (std::size_t i = numpy_shape.size(); i < GGML_MAX_DIMS; ++i) if (shape[i] != 1) fail("tensor " + quote(name) + " has an unexpected rank");
}

void validate_tensors(const gguf_context * ctx, const flow_hparams & hp,
                      const std::optional<quantization::profile> & profile) {
    const uint32_t file_type = u32(ctx, k_file_type);
    const ggml_type type = file_type == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    std::unordered_set<std::string> expected;
    const auto add = [&ctx, &expected, type, &profile](const std::string & name, std::vector<int64_t> shape) {
        const ggml_type expected_type = profile
            ? quantization::flow_tensor_type(name, static_cast<int>(shape.size()), *profile)
            : type;
        tensor(ctx, name, shape, expected_type, expected);
    };
    for (const char * stream : {"vocal", "bgm"}) {
        const std::string root = "flow.rvq." + std::string(stream);
        add(root + ".codebook.weight", {hp.codebook_size, hp.codebook_dim});
        add(root + ".out_proj.weight", {hp.condition_dim, hp.codebook_dim});
        add(root + ".out_proj.bias", {hp.condition_dim});
    }
    add("flow.mask_embedding.weight", {3, hp.mask_dim});
    add("flow.null_condition.weight", {hp.condition_dim});
    add("flow.norm.counts", {1}); add("flow.norm.sum_x", {hp.latent_dim}); add("flow.norm.sum_x2", {hp.latent_dim});
    add("flow.position_embedding.weight", {hp.max_frames, hp.hidden_size});
    add("flow.time_embedding.linear_1.weight", {hp.hidden_size, hp.time_embedding_dim});
    add("flow.time_embedding.linear_1.bias", {hp.hidden_size});
    add("flow.time_embedding.linear_2.weight", {hp.hidden_size, hp.hidden_size});
    add("flow.time_embedding.linear_2.bias", {hp.hidden_size});
    add("flow.time_modulation.weight", {6 * hp.hidden_size, hp.hidden_size});
    add("flow.time_modulation.bias", {6 * hp.hidden_size});
    for (int32_t i = 0; i < hp.n_layer; ++i) {
        const std::string root = "flow.block." + std::to_string(i);
        add(root + ".attn.qkv.weight", {3 * hp.hidden_size, hp.hidden_size}); add(root + ".attn.qkv.bias", {3 * hp.hidden_size});
        add(root + ".attn.out.weight", {hp.hidden_size, hp.hidden_size}); add(root + ".attn.out.bias", {hp.hidden_size});
        add(root + ".norm_1.weight", {hp.hidden_size}); add(root + ".norm_1.bias", {hp.hidden_size});
        add(root + ".norm_2.weight", {hp.hidden_size}); add(root + ".norm_2.bias", {hp.hidden_size});
        add(root + ".ffn.in.weight", {hp.intermediate_size, hp.hidden_size}); add(root + ".ffn.in.bias", {hp.intermediate_size});
        add(root + ".ffn.out.weight", {hp.hidden_size, hp.intermediate_size}); add(root + ".ffn.out.bias", {hp.hidden_size});
        add(root + ".modulation.weight", {6, hp.hidden_size});
    }
    add("flow.final_norm.weight", {hp.hidden_size}); add("flow.final_norm.bias", {hp.hidden_size});
    add("flow.final_modulation.weight", {2, hp.hidden_size});
    add("flow.output.weight", {hp.hidden_size, hp.hidden_size}); add("flow.output.bias", {hp.hidden_size});
    if (expected.size() != 231U) fail("internal Flow tensor inventory must contain exactly 231 tensors");
    if (static_cast<std::size_t>(gguf_get_n_tensors(ctx)) != expected.size()) fail("GGUF does not contain exactly 231 Flow tensors");
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const std::string name = gguf_get_tensor_name(ctx, i);
        if (expected.find(name) == expected.end()) fail("GGUF contains unrecognized tensor " + quote(name));
    }
}

std::size_t file_size(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) fail("cannot open " + quote(path));
    const auto size = f.tellg();
    if (size < 0 || static_cast<uintmax_t>(size) > std::numeric_limits<std::size_t>::max()) fail("invalid file size");
    return static_cast<std::size_t>(size);
}
void file_bounds(const gguf_context * ctx, std::size_t size) {
    const std::size_t alignment = gguf_get_alignment(ctx), data = gguf_get_data_offset(ctx);
    if (!alignment || (alignment & (alignment - 1)) || data > size || data % alignment) fail("invalid GGUF alignment/data offset");
    std::size_t expected = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const std::size_t offset = gguf_get_tensor_offset(ctx, i), bytes = gguf_get_tensor_size(ctx, i);
        if (offset != expected || offset % alignment || offset > size - data || bytes > size - data - offset) fail("invalid tensor data range");
        if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) fail("tensor size overflow");
        expected += (bytes + alignment - 1) & ~(alignment - 1);
    }
    if (expected > size - data) fail("GGUF data section is truncated");
}
void copy_payloads(const std::string & path, const gguf_context * ctx, const std::unordered_map<std::string, ggml_tensor *> & tensors) {
    std::ifstream f(path, std::ios::binary); if (!f) fail("cannot reopen GGUF");
    std::vector<char> buffer(8U * 1024U * 1024U); const std::size_t data = gguf_get_data_offset(ctx);
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const std::string name = gguf_get_tensor_name(ctx, i); const auto it = tensors.find(name);
        if (it == tensors.end()) fail("internal missing destination tensor");
        const std::size_t offset = data + gguf_get_tensor_offset(ctx, i), bytes = gguf_get_tensor_size(ctx, i);
        f.clear(); f.seekg(static_cast<std::streamoff>(offset)); if (!f) fail("cannot seek tensor data");
        for (std::size_t copied = 0; copied < bytes;) {
            const std::size_t count = std::min(buffer.size(), bytes - copied); f.read(buffer.data(), static_cast<std::streamsize>(count));
            if (f.gcount() != static_cast<std::streamsize>(count)) fail("truncated tensor payload");
            ggml_backend_tensor_set(it->second, buffer.data(), copied, count); copied += count;
        }
    }
}
std::size_t elements(const tensor_data & data, const std::string & name) {
    if (data.dimensions.empty() || data.dimensions.size() > GGML_MAX_DIMS) fail("test tensor " + quote(name) + " has invalid rank");
    std::size_t n = 1; for (int64_t d : data.dimensions) { if (d <= 0 || static_cast<uint64_t>(d) > std::numeric_limits<std::size_t>::max() / n) fail("test tensor has invalid shape"); n *= static_cast<std::size_t>(d); } return n;
}

} // namespace

struct model::impl { flow_hparams hp; flow_provenance prov; ggml_backend_t backend = nullptr; context_ptr context; buffer_ptr buffer; std::unordered_map<std::string, ggml_tensor *> tensors; };

bool flow_hparams::valid(std::string * reason) const {
    const auto bad = [reason](const char * text) { if (reason) *reason = text; return false; };
    if (hidden_size <= 0 || n_layer <= 0 || n_head <= 0 || head_dim <= 0 || intermediate_size <= 0 || max_frames <= 0 || codebook_size <= 0 || codebook_dim <= 0 || condition_dim <= 0 || latent_dim <= 0 || mask_dim <= 0 || time_embedding_dim <= 0 || euler_steps_default <= 0 || sample_rate <= 0 || frame_rate <= 0 || window_frames <= 0 || hop_frames <= 0 || overlap_frames <= 0) return bad("dimensions must be positive");
    if (hidden_size != n_head * head_dim) return bad("hidden_size must equal n_head * head_dim");
    if (window_frames != hop_frames + overlap_frames) return bad("window must equal hop plus overlap");
    if (!std::isfinite(cfg_default) || !std::isfinite(rope_theta) || !std::isfinite(time_embedding_scale) || !std::isfinite(sigma_min) || cfg_default <= 0 || rope_theta <= 0 || time_embedding_scale <= 0 || sigma_min <= 0 || !rvq_weight_norm_folded) return bad("floating parameters and weight-norm state are invalid");
    return true;
}
model::model() : impl_(std::make_unique<impl>()) {}
model::model(model &&) noexcept = default; model & model::operator=(model &&) noexcept = default; model::~model() = default;

std::shared_ptr<model> model::load_gguf(const std::string & path, const load_options & options) {
    if (!options.backend) fail("a GGML backend is required");
    ggml_context * raw = nullptr; const gguf_init_params params = {true, &raw}; gguf_ptr gguf(gguf_init_from_file(path.c_str(), params)); context_ptr source(raw);
    if (!gguf || !source) fail("cannot parse GGUF " + quote(path));
    if (gguf_get_version(gguf.get()) != GGUF_VERSION || gguf_get_n_tensors(gguf.get()) <= 0) fail("unsupported or empty GGUF");
    file_bounds(gguf.get(), file_size(path)); const flow_hparams hp = parse_hparams(gguf.get());
    const std::optional<quantization::profile> profile = parse_quantization(gguf.get());
    if (options.require_pinned_runtime) validate_pinned(gguf.get(), hp, profile);
    const uint32_t ft = u32(gguf.get(), k_file_type); const ggml_type type = ft == 0 ? GGML_TYPE_F32 : ft == 1 ? GGML_TYPE_F16 : GGML_TYPE_COUNT;
    const std::string dtype = str(gguf.get(), k_parameter_dtype);
    if (!profile && ((type == GGML_TYPE_F32 && dtype != "F32") || (type == GGML_TYPE_F16 && dtype != "F16"))) fail("general.file_type and Flow parameter dtype disagree");
    if (profile) {
        if (!options.allow_quantized || dtype != "MIXED") fail("quantized Flow tensors are disabled or malformed");
    } else if ((type == GGML_TYPE_F32 && !options.allow_f32) || (type == GGML_TYPE_F16 && !options.allow_f16) || type == GGML_TYPE_COUNT) {
        fail("unsupported Flow tensor type");
    }
    validate_tensors(gguf.get(), hp, profile);
    const auto result = std::shared_ptr<model>(new model()); result->impl_->hp = hp; result->impl_->backend = options.backend;
    result->impl_->prov = {str(gguf.get(), k_name), str(gguf.get(), k_runtime_repo), str(gguf.get(), k_runtime_revision), str(gguf.get(), k_levo_repo), str(gguf.get(), k_levo_revision), str(gguf.get(), k_ggml_repo), str(gguf.get(), k_ggml_revision), str(gguf.get(), k_model_sha256), str(gguf.get(), k_parameter_dtype), token_io::file_sha256(path)};
    const int64_t count = gguf_get_n_tensors(gguf.get()); if (static_cast<uint64_t>(count) > std::numeric_limits<std::size_t>::max() / ggml_tensor_overhead()) fail("tensor count overflow");
    result->impl_->context.reset(ggml_init({static_cast<std::size_t>(count) * ggml_tensor_overhead(), nullptr, true})); if (!result->impl_->context) fail("cannot create weight context");
    for (int64_t i = 0; i < count; ++i) {
        const std::string name = gguf_get_tensor_name(gguf.get(), i); const ggml_tensor * src = ggml_get_tensor(source.get(), name.c_str());
        if (!src || src->type != gguf_get_tensor_type(gguf.get(), i)) fail("inconsistent GGUF tensor metadata");
        ggml_tensor * dst = ggml_dup_tensor(result->impl_->context.get(), src);
        if (!dst) fail("cannot allocate tensor object");
        ggml_set_name(dst, name.c_str());
        if (!result->impl_->tensors.emplace(name, dst).second) fail("duplicate tensor");
    }
    result->impl_->buffer.reset(ggml_backend_alloc_ctx_tensors(result->impl_->context.get(), options.backend)); if (!result->impl_->buffer) fail("backend cannot allocate Flow weights");
    copy_payloads(path, gguf.get(), result->impl_->tensors); ggml_backend_synchronize(options.backend); return result;
}

std::shared_ptr<model> model::make_test_model(const flow_hparams & hp, const std::unordered_map<std::string, tensor_data> & tensors, ggml_backend_t backend) {
    if (!backend) fail("a GGML backend is required");
    std::string reason;
    if (!hp.valid(&reason)) fail("invalid test Flow hparams: " + reason);
    if (tensors.empty()) fail("test model has no tensors");
    const auto result = std::shared_ptr<model>(new model()); result->impl_->hp = hp; result->impl_->backend = backend;
    result->impl_->context.reset(ggml_init({tensors.size() * ggml_tensor_overhead(), nullptr, true})); if (!result->impl_->context) fail("cannot create test context");
    for (const auto & item : tensors) { const std::size_t n = elements(item.second, item.first); if (n != item.second.values.size()) fail("test tensor payload mismatch"); ggml_tensor * t = ggml_new_tensor(result->impl_->context.get(), GGML_TYPE_F32, static_cast<int>(item.second.dimensions.size()), item.second.dimensions.data()); if (!t) fail("cannot create test tensor"); ggml_set_name(t, item.first.c_str()); result->impl_->tensors.emplace(item.first, t); }
    result->impl_->buffer.reset(ggml_backend_alloc_ctx_tensors(result->impl_->context.get(), backend)); if (!result->impl_->buffer) fail("backend cannot allocate test tensors");
    for (const auto & item : tensors) {
        ggml_backend_tensor_set(result->impl_->tensors.at(item.first), item.second.values.data(), 0,
                                item.second.values.size() * sizeof(float));
    }
    ggml_backend_synchronize(backend);
    return result;
}
const flow_hparams & model::hparams() const noexcept { return impl_->hp; } const flow_provenance & model::provenance() const noexcept { return impl_->prov; } ggml_backend_t model::backend() const noexcept { return impl_->backend; }
ggml_tensor * model::tensor(const std::string & name) const { const auto it = impl_->tensors.find(name); return it == impl_->tensors.end() ? nullptr : it->second; }
bool model::contains(const std::string & name) const noexcept { return impl_->tensors.find(name) != impl_->tensors.end(); }
std::size_t model::tensor_count() const noexcept { return impl_->tensors.size(); }

} // namespace levo::flow
