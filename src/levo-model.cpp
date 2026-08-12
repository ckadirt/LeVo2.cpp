#include "levo-model.h"

#include "gguf.h"

#include <algorithm>
#include <array>
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

struct context_deleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

struct gguf_deleter {
    void operator()(gguf_context * context) const noexcept {
        if (context != nullptr) {
            gguf_free(context);
        }
    }
};

using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo GGUF: " + message);
}

std::string quote(const std::string & value) {
    return "'" + value + "'";
}

int64_t required_key(const gguf_context * context, const char * key, gguf_type type) {
    const int64_t index = gguf_find_key(context, key);
    if (index < 0) {
        fail("missing required metadata key " + quote(key));
    }
    if (gguf_get_kv_type(context, index) != type) {
        fail("metadata key " + quote(key) + " has type " +
             gguf_type_name(gguf_get_kv_type(context, index)) + ", expected " +
             gguf_type_name(type));
    }
    return index;
}

uint32_t required_u32(const gguf_context * context, const char * key) {
    return gguf_get_val_u32(context, required_key(context, key, GGUF_TYPE_UINT32));
}

float required_f32(const gguf_context * context, const char * key) {
    return gguf_get_val_f32(context, required_key(context, key, GGUF_TYPE_FLOAT32));
}

std::string required_string(const gguf_context * context, const char * key) {
    return gguf_get_val_str(context, required_key(context, key, GGUF_TYPE_STRING));
}

std::vector<int32_t> required_i32_array(const gguf_context * context, const char * key) {
    const int64_t index = required_key(context, key, GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(context, index) != GGUF_TYPE_INT32) {
        fail("metadata array " + quote(key) + " has element type " +
             gguf_type_name(gguf_get_arr_type(context, index)) + ", expected int32");
    }
    const std::size_t count = gguf_get_arr_n(context, index);
    if (count == 0 || count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        fail("metadata array " + quote(key) + " has an invalid element count");
    }
    const auto * values = static_cast<const int32_t *>(gguf_get_arr_data(context, index));
    return {values, values + count};
}

void expect_equal(const char * key, int32_t actual, int32_t expected) {
    if (actual != expected) {
        std::ostringstream message;
        message << "metadata " << quote(key) << " is " << actual << ", expected " << expected;
        fail(message.str());
    }
}

void expect_equal(const char * key, float actual, float expected) {
    // Metadata values are written as F32, so bitwise equality is the strict and
    // portable way to reject an artifact for a different architecture.
    uint32_t a = 0;
    uint32_t e = 0;
    std::memcpy(&a, &actual, sizeof(a));
    std::memcpy(&e, &expected, sizeof(e));
    if (a != e) {
        std::ostringstream message;
        message << "metadata " << quote(key) << " is " << actual << ", expected " << expected;
        fail(message.str());
    }
}

model_hparams parse_hparams(const gguf_context * context) {
    if (required_string(context, gguf_keys::architecture) != "levo2") {
        fail("general.architecture must be 'levo2'");
    }
    if (required_string(context, gguf_keys::name).empty()) {
        fail("general.name must not be empty");
    }
    if (required_string(context, gguf_keys::source_repository).empty()) {
        fail("general.source.repo_url must not be empty");
    }
    if (required_string(context, gguf_keys::converter).empty()) {
        fail("levo2.converter must not be empty");
    }
    if (required_string(context, gguf_keys::source_model_sha256).size() != 64) {
        fail("levo2.source.model_sha256 must be a SHA-256 hex string");
    }
    // The optional upstream config is absent in some historical checkpoints;
    // an empty string is therefore meaningful and accepted.
    (void) required_string(context, gguf_keys::source_config_sha256);
    expect_equal(gguf_keys::schema_version,
                 static_cast<int32_t>(required_u32(context, gguf_keys::schema_version)), 1);

    model_hparams hparams;
    hparams.main_layers = static_cast<int32_t>(required_u32(context, gguf_keys::main_layers));
    hparams.detail_layers = static_cast<int32_t>(required_u32(context, gguf_keys::detail_layers));
    hparams.embedding_length = static_cast<int32_t>(required_u32(context, gguf_keys::embedding_length));
    hparams.feed_forward_length = static_cast<int32_t>(required_u32(context, gguf_keys::feed_forward_length));
    hparams.attention_heads = static_cast<int32_t>(required_u32(context, gguf_keys::attention_heads));
    hparams.kv_attention_heads = static_cast<int32_t>(required_u32(context, gguf_keys::kv_attention_heads));
    hparams.context_length = static_cast<int32_t>(required_u32(context, gguf_keys::context_length));
    hparams.rms_norm_epsilon = required_f32(context, gguf_keys::rms_norm_epsilon);
    hparams.main_rope_theta = required_f32(context, gguf_keys::main_rope_theta);
    hparams.detail_rope_theta = required_f32(context, gguf_keys::detail_rope_theta);
    hparams.codebook_count = static_cast<int32_t>(required_u32(context, gguf_keys::codebook_count));
    hparams.codebook_size = static_cast<int32_t>(required_u32(context, gguf_keys::codebook_size));
    hparams.eos_token_id = static_cast<int32_t>(required_u32(context, gguf_keys::eos_token_id));
    hparams.special_token_id = static_cast<int32_t>(required_u32(context, gguf_keys::special_token_id));
    hparams.frame_rate = static_cast<int32_t>(required_u32(context, gguf_keys::frame_rate));
    hparams.sample_rate = static_cast<int32_t>(required_u32(context, gguf_keys::sample_rate));
    hparams.delays = required_i32_array(context, gguf_keys::delays);
    hparams.lyrics_prefix_length = static_cast<int32_t>(required_u32(context, gguf_keys::lyrics_prefix_length));
    hparams.prompt_prefix_length = static_cast<int32_t>(required_u32(context, gguf_keys::prompt_prefix_length));
    hparams.style_prefix_length = static_cast<int32_t>(required_u32(context, gguf_keys::style_prefix_length));

    std::string reason;
    if (!hparams.valid(&reason)) {
        fail("invalid LeVo metadata: " + reason);
    }
    return hparams;
}

void validate_v2_medium(const gguf_context * context, const model_hparams & hparams) {
    const model_hparams expected;
    expect_equal(gguf_keys::main_layers, hparams.main_layers, expected.main_layers);
    expect_equal(gguf_keys::detail_layers, hparams.detail_layers, expected.detail_layers);
    expect_equal(gguf_keys::embedding_length, hparams.embedding_length, expected.embedding_length);
    expect_equal(gguf_keys::feed_forward_length, hparams.feed_forward_length, expected.feed_forward_length);
    expect_equal(gguf_keys::attention_heads, hparams.attention_heads, expected.attention_heads);
    expect_equal(gguf_keys::kv_attention_heads, hparams.kv_attention_heads, expected.kv_attention_heads);
    expect_equal(gguf_keys::context_length, hparams.context_length, expected.context_length);
    expect_equal(gguf_keys::rms_norm_epsilon, hparams.rms_norm_epsilon, expected.rms_norm_epsilon);
    expect_equal(gguf_keys::main_rope_theta, hparams.main_rope_theta, expected.main_rope_theta);
    expect_equal(gguf_keys::detail_rope_theta, hparams.detail_rope_theta, expected.detail_rope_theta);
    expect_equal(gguf_keys::codebook_count, hparams.codebook_count, expected.codebook_count);
    expect_equal(gguf_keys::codebook_size, hparams.codebook_size, expected.codebook_size);
    expect_equal(gguf_keys::eos_token_id, hparams.eos_token_id, expected.eos_token_id);
    expect_equal(gguf_keys::special_token_id, hparams.special_token_id, expected.special_token_id);
    expect_equal(gguf_keys::frame_rate, hparams.frame_rate, expected.frame_rate);
    expect_equal(gguf_keys::sample_rate, hparams.sample_rate, expected.sample_rate);
    expect_equal(gguf_keys::lyrics_prefix_length, hparams.lyrics_prefix_length, expected.lyrics_prefix_length);
    expect_equal(gguf_keys::prompt_prefix_length, hparams.prompt_prefix_length, expected.prompt_prefix_length);
    expect_equal(gguf_keys::style_prefix_length, hparams.style_prefix_length, expected.style_prefix_length);
    if (hparams.delays != expected.delays) {
        fail("metadata " + quote(gguf_keys::delays) + " does not match [0, 250, 250]");
    }

    const uint32_t file_type = required_u32(context, gguf_keys::file_type);
    if (file_type != 0 && file_type != 1) {
        fail("general.file_type must be GGML F32 (0) or F16 (1)");
    }
}

void validate_tensor(const gguf_context * context, const std::string & name,
                     std::initializer_list<int64_t> expected_shape, ggml_type expected_type) {
    const int64_t id = gguf_find_tensor(context, name.c_str());
    if (id < 0) {
        fail("missing required tensor " + quote(name));
    }
    if (gguf_get_tensor_type(context, id) != expected_type) {
        fail("tensor " + quote(name) + " has type " + ggml_type_name(gguf_get_tensor_type(context, id)) +
             ", inconsistent with general.file_type");
    }
    const int64_t * shape = gguf_get_tensor_ne(context, id);
    std::size_t dimension = 0;
    for (const int64_t value : expected_shape) {
        if (shape[dimension++] != value) {
            fail("tensor " + quote(name) + " has an unexpected shape");
        }
    }
    for (; dimension < GGML_MAX_DIMS; ++dimension) {
        if (shape[dimension] != 1) {
            fail("tensor " + quote(name) + " has an unexpected rank");
        }
    }
}

void validate_tensor_inventory(const gguf_context * context, const model_hparams & hparams) {
    const uint32_t file_type = required_u32(context, gguf_keys::file_type);
    const ggml_type type = file_type == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    const int64_t width = hparams.embedding_length;
    const int64_t ffn = hparams.feed_forward_length;
    const int64_t input_vocab = hparams.token_input_size();
    const int64_t output_vocab = hparams.token_output_size();
    std::unordered_set<std::string> expected;
    const auto add = [&expected, context, type](const std::string & name,
                                                 std::initializer_list<int64_t> shape) {
        if (!expected.emplace(name).second) {
            fail("internal duplicate expected tensor " + quote(name));
        }
        validate_tensor(context, name, shape, type);
    };

    add(tensor_names::mixed_embedding, {width, input_vocab});
    add(tensor_names::detail_vocal_embedding, {width, input_vocab});
    add(tensor_names::detail_bgm_embedding, {width, input_vocab});
    add(tensor_names::mixed_output, {width, output_vocab});
    add(tensor_names::vocal_output, {width, output_vocab});
    add(tensor_names::bgm_output, {width, output_vocab});
    add(tensor_names::main_norm, {width});
    add(tensor_names::detail_norm, {width});
    add(tensor_names::bridge_0_weight, {width * 2, width});
    add(tensor_names::bridge_0_bias, {width});
    add(tensor_names::bridge_2_weight, {width, width});
    add(tensor_names::bridge_2_bias, {width});
    // Qwen2's 151646 IDs plus the thirteen LeVo lyric structure tokens.
    add("cond.lyrics.weight", {width, 151659});
    add("cond.structure.weight", {width, 200});
    // The checkpoint intentionally retains six unreachable style rows.
    add("cond.style.weight", {width, 151652});
    add("cond.prompt_embd.mixed.weight", {width, input_vocab});
    add("cond.prompt_embd.vocal.weight", {width, input_vocab});
    add("cond.prompt_embd.bgm.weight", {width, input_vocab});
    add("cond.prompt_eot.mixed", {width, 1});
    add("cond.prompt_eot.detail", {width, 1});

    const auto add_tower = [&add, width, ffn](int32_t count, bool detail) {
        for (int32_t layer = 0; layer < count; ++layer) {
            const auto name = [detail, layer](const char * suffix) {
                return detail ? tensor_names::detail_layer(static_cast<unsigned>(layer), suffix)
                              : tensor_names::main_layer(static_cast<unsigned>(layer), suffix);
            };
            add(name("attn_norm.weight"), {width});
            add(name("attn_q.weight"), {width, width});
            add(name("attn_k.weight"), {width, width});
            add(name("attn_v.weight"), {width, width});
            add(name("attn_output.weight"), {width, width});
            add(name("ffn_norm"), {width});
            add(name("ffn_gate.weight"), {width, ffn});
            add(name("ffn_up.weight"), {width, ffn});
            add(name("ffn_down.weight"), {ffn, width});
        }
    };
    add_tower(hparams.main_layers, false);
    add_tower(hparams.detail_layers, true);

    if (expected.size() != 380U) {
        fail("internal v2-medium tensor inventory does not contain 380 tensors");
    }
    if (static_cast<std::size_t>(gguf_get_n_tensors(context)) != expected.size()) {
        std::ostringstream message;
        message << "GGUF contains " << gguf_get_n_tensors(context) << " tensors; v2-medium requires exactly "
                << expected.size();
        fail(message.str());
    }
    for (int64_t index = 0; index < gguf_get_n_tensors(context); ++index) {
        const std::string name = gguf_get_tensor_name(context, index);
        if (expected.find(name) == expected.end()) {
            fail("GGUF contains unrecognized tensor " + quote(name));
        }
    }
}

std::size_t file_size_or_fail(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fail("cannot open " + quote(path));
    }
    const std::streamoff size = file.tellg();
    if (size < 0 || static_cast<uintmax_t>(size) > std::numeric_limits<std::size_t>::max()) {
        fail("cannot determine a representable size for " + quote(path));
    }
    return static_cast<std::size_t>(size);
}

void validate_tensor_file_bounds(const gguf_context * context, std::size_t file_size) {
    const std::size_t alignment = gguf_get_alignment(context);
    const std::size_t data_offset = gguf_get_data_offset(context);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || data_offset % alignment != 0) {
        fail("GGUF alignment or tensor data offset is invalid");
    }
    if (data_offset > file_size) {
        fail("GGUF tensor data offset is beyond end of file");
    }

    std::size_t expected_offset = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(context); ++i) {
        const char * name = gguf_get_tensor_name(context, i);
        const std::size_t offset = gguf_get_tensor_offset(context, i);
        const std::size_t size = gguf_get_tensor_size(context, i);
        if (offset != expected_offset || offset % alignment != 0) {
            fail("tensor " + quote(name) + " has an invalid or non-contiguous data offset");
        }
        if (size > file_size - data_offset || offset > file_size - data_offset - size) {
            fail("tensor " + quote(name) + " exceeds GGUF file bounds");
        }
        if (size > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
            fail("tensor " + quote(name) + " size cannot be aligned safely");
        }
        expected_offset += (size + alignment - 1) & ~(alignment - 1);
    }
    if (expected_offset > file_size - data_offset) {
        fail("GGUF tensor data section is truncated");
    }
}

void validate_tensor_type(const std::string & name, ggml_type type, const model_load_options & options) {
    if (type == GGML_TYPE_F32 && options.allow_f32) {
        return;
    }
    if (type == GGML_TYPE_F16 && options.allow_f16) {
        return;
    }
    fail("tensor " + quote(name) + " uses unsupported type " + ggml_type_name(type) +
         "; only F32/F16 model weights are accepted");
}

void read_tensor_payloads(const std::string & path, const gguf_context * gguf,
                          const std::unordered_map<std::string, ggml_tensor *> & tensors) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fail("cannot reopen " + quote(path) + " to read tensor data");
    }

    constexpr std::size_t chunk_size = 8U * 1024U * 1024U;
    std::vector<char> chunk(chunk_size);
    const std::size_t data_offset = gguf_get_data_offset(gguf);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const std::string name = gguf_get_tensor_name(gguf, i);
        const auto destination = tensors.find(name);
        if (destination == tensors.end()) {
            fail("internal error: destination tensor is missing for " + quote(name));
        }
        const std::size_t offset = data_offset + gguf_get_tensor_offset(gguf, i);
        const std::size_t size = gguf_get_tensor_size(gguf, i);
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!file) {
            fail("cannot seek to tensor " + quote(name));
        }
        for (std::size_t copied = 0; copied < size;) {
            const std::size_t count = std::min(chunk.size(), size - copied);
            file.read(chunk.data(), static_cast<std::streamsize>(count));
            if (file.gcount() != static_cast<std::streamsize>(count)) {
                fail("truncated data while reading tensor " + quote(name));
            }
            ggml_backend_tensor_set(destination->second, chunk.data(), copied, count);
            copied += count;
        }
    }
}

std::size_t checked_element_count(const std::vector<int64_t> & dimensions, const std::string & name) {
    if (dimensions.empty() || dimensions.size() > GGML_MAX_DIMS) {
        fail("test tensor " + quote(name) + " has an invalid rank");
    }
    std::size_t elements = 1;
    for (const int64_t dimension : dimensions) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<std::size_t>::max() / elements) {
            fail("test tensor " + quote(name) + " has an invalid shape");
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    return elements;
}

} // namespace

struct model::impl {
    model_hparams hparams;
    ggml_backend_t backend = nullptr;
    context_ptr context;
    buffer_ptr buffer;
    std::unordered_map<std::string, ggml_tensor *> tensors;
};

namespace tensor_names {

std::string main_layer(unsigned layer, const char * suffix) {
    return "main.blk." + std::to_string(layer) + "." + suffix;
}

std::string detail_layer(unsigned layer, const char * suffix) {
    return "detail.blk." + std::to_string(layer) + "." + suffix;
}

} // namespace tensor_names

bool model_hparams::valid(std::string * reason) const {
    const auto invalid = [reason](const char * message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (main_layers < 0 || detail_layers < 0) {
        return invalid("layer count must not be negative");
    }
    if (embedding_length <= 0 || feed_forward_length <= 0 || attention_heads <= 0 ||
        kv_attention_heads <= 0 || context_length <= 0) {
        return invalid("transformer dimensions must be positive");
    }
    if (embedding_length % attention_heads != 0 || attention_heads % kv_attention_heads != 0) {
        return invalid("embedding/head dimensions are incompatible");
    }
    if (!std::isfinite(rms_norm_epsilon) || rms_norm_epsilon <= 0.0F ||
        !std::isfinite(main_rope_theta) || main_rope_theta <= 0.0F ||
        !std::isfinite(detail_rope_theta) || detail_rope_theta <= 0.0F) {
        return invalid("normalization epsilon and RoPE bases must be finite and positive");
    }
    if (codebook_count != 3 || codebook_size <= 0 || eos_token_id != codebook_size ||
        special_token_id != codebook_size + 1) {
        return invalid("only three codebooks with contiguous EOS/special token IDs are supported");
    }
    if (delays.size() != static_cast<std::size_t>(codebook_count) || delays.empty() || delays.front() != 0 ||
        !std::is_sorted(delays.begin(), delays.end()) || delays.back() < 0) {
        return invalid("delays must be a sorted three-element vector beginning at zero");
    }
    if (frame_rate <= 0 || sample_rate <= 0 || lyrics_prefix_length < 0 || prompt_prefix_length < 0 ||
        style_prefix_length < 0) {
        return invalid("audio and prefix lengths are invalid");
    }
    return true;
}

model::model() : impl_(std::make_unique<impl>()) {}
model::model(model &&) noexcept = default;
model & model::operator=(model &&) noexcept = default;
model::~model() = default;

std::shared_ptr<model> model::load_gguf(const std::string & path, const model_load_options & options) {
    if (options.backend == nullptr) {
        fail("a GGML backend is required to load weights");
    }
    const std::size_t input_size = file_size_or_fail(path);
    ggml_context * source_context_raw = nullptr;
    const gguf_init_params params = {
        /* no_alloc = */ true,
        /* ctx      = */ &source_context_raw,
    };
    gguf_ptr gguf(gguf_init_from_file(path.c_str(), params));
    context_ptr source_context(source_context_raw);
    if (!gguf || !source_context) {
        fail("cannot parse GGUF " + quote(path));
    }
    if (gguf_get_version(gguf.get()) != GGUF_VERSION) {
        fail("unsupported GGUF version " + std::to_string(gguf_get_version(gguf.get())) +
             "; this runtime requires version " + std::to_string(GGUF_VERSION));
    }
    if (gguf_get_n_tensors(gguf.get()) <= 0) {
        fail("GGUF has no tensors");
    }
    validate_tensor_file_bounds(gguf.get(), input_size);
    const model_hparams hparams = parse_hparams(gguf.get());
    if (options.require_v2_medium) {
        validate_v2_medium(gguf.get(), hparams);
        validate_tensor_inventory(gguf.get(), hparams);
    }

    const auto result = std::shared_ptr<model>(new model());
    result->impl_->hparams = hparams;
    result->impl_->backend = options.backend;
    const int64_t tensor_count = gguf_get_n_tensors(gguf.get());
    if (static_cast<uint64_t>(tensor_count) > std::numeric_limits<std::size_t>::max() / ggml_tensor_overhead()) {
        fail("GGUF tensor count overflows GGML context allocation");
    }
    const std::size_t context_size = static_cast<std::size_t>(tensor_count) * ggml_tensor_overhead();
    const ggml_init_params weight_params = {context_size, nullptr, true};
    result->impl_->context.reset(ggml_init(weight_params));
    if (!result->impl_->context) {
        fail("cannot create GGML weight context");
    }

    for (int64_t i = 0; i < tensor_count; ++i) {
        const std::string name = gguf_get_tensor_name(gguf.get(), i);
        const ggml_type type = gguf_get_tensor_type(gguf.get(), i);
        validate_tensor_type(name, type, options);
        const ggml_tensor * source = ggml_get_tensor(source_context.get(), name.c_str());
        if (source == nullptr || source->type != type) {
            fail("GGML tensor metadata is inconsistent for " + quote(name));
        }
        ggml_tensor * destination = ggml_dup_tensor(result->impl_->context.get(), source);
        if (destination == nullptr) {
            fail("cannot create destination tensor " + quote(name));
        }
        ggml_set_name(destination, name.c_str());
        if (!result->impl_->tensors.emplace(name, destination).second) {
            fail("duplicate tensor " + quote(name));
        }
    }
    result->impl_->buffer.reset(ggml_backend_alloc_ctx_tensors(result->impl_->context.get(), options.backend));
    if (!result->impl_->buffer) {
        fail("backend cannot allocate GGUF weights");
    }
    read_tensor_payloads(path, gguf.get(), result->impl_->tensors);
    ggml_backend_synchronize(options.backend);
    return result;
}

std::shared_ptr<model> model::make_test_model(
    const model_hparams & hparams,
    const std::unordered_map<std::string, tensor_data> & tensors,
    ggml_backend_t backend) {
    if (backend == nullptr) {
        fail("a GGML backend is required to create a test model");
    }
    std::string reason;
    if (!hparams.valid(&reason)) {
        fail("invalid test model hyperparameters: " + reason);
    }
    if (tensors.empty()) {
        fail("test model must contain at least one tensor");
    }
    if (tensors.size() > std::numeric_limits<std::size_t>::max() / ggml_tensor_overhead()) {
        fail("test model has too many tensors");
    }
    const auto result = std::shared_ptr<model>(new model());
    result->impl_->hparams = hparams;
    result->impl_->backend = backend;
    const ggml_init_params params = {tensors.size() * ggml_tensor_overhead(), nullptr, true};
    result->impl_->context.reset(ggml_init(params));
    if (!result->impl_->context) {
        fail("cannot create test weight context");
    }
    for (const auto & item : tensors) {
        const std::size_t element_count = checked_element_count(item.second.dimensions, item.first);
        if (item.second.values.size() != element_count) {
            fail("test tensor " + quote(item.first) + " payload size does not match its shape");
        }
        ggml_tensor * tensor = ggml_new_tensor(result->impl_->context.get(), GGML_TYPE_F32,
                                                 static_cast<int>(item.second.dimensions.size()),
                                                 item.second.dimensions.data());
        if (!tensor) {
            fail("cannot create test tensor " + quote(item.first));
        }
        ggml_set_name(tensor, item.first.c_str());
        result->impl_->tensors.emplace(item.first, tensor);
    }
    result->impl_->buffer.reset(ggml_backend_alloc_ctx_tensors(result->impl_->context.get(), backend));
    if (!result->impl_->buffer) {
        fail("backend cannot allocate test weights");
    }
    for (const auto & item : tensors) {
        ggml_backend_tensor_set(result->impl_->tensors.at(item.first), item.second.values.data(), 0,
                                item.second.values.size() * sizeof(float));
    }
    ggml_backend_synchronize(backend);
    return result;
}

const model_hparams & model::hparams() const noexcept { return impl_->hparams; }
ggml_backend_t model::backend() const noexcept { return impl_->backend; }

ggml_tensor * model::tensor(const std::string & name) const {
    const auto item = impl_->tensors.find(name);
    return item == impl_->tensors.end() ? nullptr : item->second;
}

bool model::contains(const std::string & name) const noexcept {
    return impl_->tensors.find(name) != impl_->tensors.end();
}

std::size_t model::tensor_count() const noexcept { return impl_->tensors.size(); }

} // namespace levo::detail
