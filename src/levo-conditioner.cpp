#include "levo-conditioner.h"

#include "levo-quantization.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace levo::detail {
namespace {

constexpr int64_t kQwenPadId = 151643;
constexpr int64_t kStructureBaseId = 151645;
constexpr const char * kLyricsWeights = "cond.lyrics.weight";
constexpr const char * kStructureWeights = "cond.structure.weight";
constexpr const char * kStyleWeights = "cond.style.weight";
constexpr const char * kPromptMixedWeights = "cond.prompt_embd.mixed.weight";
constexpr const char * kPromptVocalWeights = "cond.prompt_embd.vocal.weight";
constexpr const char * kPromptBgmWeights = "cond.prompt_embd.bgm.weight";
constexpr const char * kPromptEotMixed = "cond.prompt_eot.mixed";
constexpr const char * kPromptEotDetail = "cond.prompt_eot.detail";

const char * structure_names[] = {
    "[verse]", "[chorus]", "[bridge]", "[intro-short]", "[intro-medium]",
    "[intro-long]", "[outro-short]", "[outro-medium]", "[outro-long]",
    "[inst-short]", "[inst-medium]", "[inst-long]", "[silence]",
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo conditioner: " + message);
}

struct context_deleter {
    void operator()(ggml_context * context) const noexcept { if (context) ggml_free(context); }
};
struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept { if (buffer) ggml_backend_buffer_free(buffer); }
};
struct allocator_deleter {
    void operator()(ggml_gallocr_t allocator) const noexcept { if (allocator) ggml_gallocr_free(allocator); }
};
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using allocator_ptr = std::unique_ptr<ggml_gallocr, allocator_deleter>;

ggml_tensor * checked_embedding(const model & weights, const char * name, int64_t width, int64_t rows) {
    ggml_tensor * tensor = weights.tensor(name);
    if (tensor == nullptr) {
        fail(std::string("missing tensor '") + name + "'");
    }
    if (tensor->ne[0] != width || tensor->ne[1] != rows) {
        std::ostringstream message;
        message << "tensor '" << name << "' has shape [" << tensor->ne[0] << ", "
                << tensor->ne[1] << "], expected [" << width << ", " << rows << "]";
        fail(message.str());
    }
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] != 1) {
            fail(std::string("tensor '") + name + "' must be rank two");
        }
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && !quantization::is_quantized(tensor->type)) {
        fail(std::string("tensor '") + name + "' has an unsupported embedding type");
    }
    return tensor;
}

std::vector<float> read_rows_f32(const model & weights, const char * name,
                                 int64_t width, int64_t rows,
                                 const std::vector<int64_t> & ids) {
    if (ids.empty()) return {};
    ggml_tensor * tensor = checked_embedding(weights, name, width, rows);
    for (const int64_t id : ids) if (id < 0 || id >= rows) fail(std::string("row index is outside tensor '") + name + "'");
    const std::size_t count = static_cast<std::size_t>(width) * ids.size();
    std::vector<float> result(count);
    if (tensor->type == GGML_TYPE_F32) {
        for (std::size_t index = 0; index < ids.size(); ++index) {
            ggml_backend_tensor_get(tensor, result.data() + index * static_cast<std::size_t>(width),
                                    static_cast<std::size_t>(ids[index]) * static_cast<std::size_t>(width) * sizeof(float),
                                    static_cast<std::size_t>(width) * sizeof(float));
        }
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> values(static_cast<std::size_t>(width));
        for (std::size_t index = 0; index < ids.size(); ++index) {
            ggml_backend_tensor_get(tensor, values.data(),
                                    static_cast<std::size_t>(ids[index]) * static_cast<std::size_t>(width) * sizeof(ggml_fp16_t),
                                    values.size() * sizeof(ggml_fp16_t));
            for (std::size_t column = 0; column < values.size(); ++column) {
                result[index * values.size() + column] = ggml_fp16_to_fp32(values[column]);
            }
        }
    } else {
        context_ptr graph_context(ggml_init({16U * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true}));
        context_ptr input_context(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        if (!graph_context || !input_context) fail("cannot create quantized conditioner graph context");
        ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 16, false);
        if (!graph) fail("cannot create quantized conditioner graph");
        ggml_tensor * row_ids = ggml_new_tensor_1d(input_context.get(), GGML_TYPE_I32, static_cast<int64_t>(ids.size()));
        if (!row_ids) fail("cannot create quantized conditioner row IDs");
        ggml_set_input(row_ids);
        buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(input_context.get(), weights.backend()));
        if (!input_buffer) fail("cannot allocate quantized conditioner row IDs");
        std::vector<int32_t> ids_i32(ids.begin(), ids.end());
        ggml_backend_tensor_set(row_ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
        ggml_tensor * gathered = ggml_get_rows(graph_context.get(), tensor, row_ids);
        if (!gathered || gathered->type != GGML_TYPE_F32 || gathered->ne[0] != width || gathered->ne[1] != static_cast<int64_t>(ids.size())) {
            fail(std::string("GGML cannot gather F32 rows from quantized tensor '") + name + "'");
        }
        ggml_set_output(gathered);
        ggml_build_forward_expand(graph, gathered);
        allocator_ptr allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) fail("cannot allocate quantized conditioner graph");
        if (ggml_backend_graph_compute(weights.backend(), graph) != GGML_STATUS_SUCCESS) {
            fail(std::string("GGML quantized conditioner graph failed for '") + name + "'");
        }
        ggml_backend_tensor_get(gathered, result.data(), 0, result.size() * sizeof(float));
    }
    return result;
}

float f16_add(float lhs, float rhs) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(lhs + rhs));
}

struct padded_text {
    std::vector<int64_t> ids;
    std::size_t content_length = 0;
};

padded_text padded_tokens(const ByteLevelBPETokenizer & tokenizer,
                          const std::string & text, int32_t length) {
    if (length < 0) {
        fail("negative text prefix length");
    }
    padded_text result;
    result.ids = tokenizer.encode(std::string("<|im_start|>") + text);
    result.content_length = std::min(result.ids.size(), static_cast<std::size_t>(length));
    if (result.ids.size() > static_cast<std::size_t>(length)) {
        result.ids.resize(static_cast<std::size_t>(length));
    } else {
        result.ids.resize(static_cast<std::size_t>(length), kQwenPadId);
    }
    return result;
}

std::vector<int64_t> lyric_structure_ids(const ByteLevelBPETokenizer & tokenizer) {
    std::vector<int64_t> ids;
    ids.reserve(sizeof(structure_names) / sizeof(structure_names[0]));
    for (const char * name : structure_names) {
        if (!tokenizer.has_token(name)) {
            fail(std::string("tokenizer is missing structure token '") + name + "'");
        }
        ids.push_back(tokenizer.token_id(name));
    }
    return ids;
}

condition_tensor text_condition(const model & weights, const ByteLevelBPETokenizer & tokenizer,
                                const std::string & text, int32_t prefix, const char * embedding,
                                bool structures) {
    const int64_t width = weights.hparams().embedding_length;
    const int64_t vocab = structures ? 151659 : 151652;
    padded_text padded = padded_tokens(tokenizer, text, prefix);
    const std::vector<int64_t> & ids = padded.ids;
    std::vector<int64_t> range(ids.size(), 0);
    if (structures) {
        const std::vector<int64_t> struct_ids = lyric_structure_ids(tokenizer);
        std::vector<std::size_t> boundaries;
        for (std::size_t i = 0; i < padded.content_length; ++i) {
            if (std::find(struct_ids.begin(), struct_ids.end(), ids[i]) != struct_ids.end()) {
                boundaries.push_back(i);
            }
        }
        for (std::size_t i = 0; i < boundaries.size(); ++i) {
            const std::size_t end = i + 1 < boundaries.size() ? boundaries[i + 1]
                                                               : padded.content_length;
            const int64_t row = ids[boundaries[i]] - kStructureBaseId;
            if (row < 0 || row >= 200) {
                fail("structure token ID is outside cond.structure.weight");
            }
            std::fill(range.begin() + static_cast<std::ptrdiff_t>(boundaries[i]),
                      range.begin() + static_cast<std::ptrdiff_t>(end), row);
        }
    }
    const bool source_f16 = weights.tensor(embedding)->type == GGML_TYPE_F16;
    const std::vector<float> content = read_rows_f32(weights, embedding, width, vocab, ids);
    const std::vector<float> structure = structures
        ? read_rows_f32(weights, kStructureWeights, width, 200, range) : std::vector<float>{};
    condition_tensor result;
    result.width = static_cast<int32_t>(width);
    result.prefix = prefix;
    result.values.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(prefix));
    for (int32_t t = 0; t < prefix; ++t) {
        const int64_t token = ids[static_cast<std::size_t>(t)];
        if (token < 0 || token >= vocab) {
            fail("text token ID is outside conditioner embedding");
        }
        for (int32_t d = 0; d < width; ++d) {
            float value = content[static_cast<std::size_t>(t) * static_cast<std::size_t>(width) + d];
            if (structures && range[static_cast<std::size_t>(t)] != 0) {
                const float addition = structure[static_cast<std::size_t>(t) * static_cast<std::size_t>(width) + d];
                value = source_f16 ? f16_add(value, addition) : value + addition;
            }
            result.values[static_cast<std::size_t>(t) * static_cast<std::size_t>(width) + d] = value;
        }
    }
    return result;
}

condition_pair make_branch(const model & weights, const ByteLevelBPETokenizer & tokenizer,
                           const std::string & lyrics, const std::string & description) {
    const model_hparams & hp = weights.hparams();
    condition_pair result;
    const condition_tensor lyric = text_condition(weights, tokenizer, lyrics, hp.lyrics_prefix_length,
                                                  kLyricsWeights, true);
    const condition_tensor style = text_condition(weights, tokenizer, description, hp.style_prefix_length,
                                                  kStyleWeights, false);
    const int64_t width = hp.embedding_length;
    const int32_t prompt_length = hp.prompt_prefix_length;
    if (prompt_length < 2) {
        fail("prompt prefix must contain EOT and an EOS frame");
    }
    const int32_t frames = prompt_length - 2;
    const int64_t input_vocab = hp.token_input_size();
    const int64_t special = hp.special_token_id;
    const std::vector<int64_t> special_row{special};
    const std::vector<int64_t> eot_row{0};
    const std::vector<float> mixed = read_rows_f32(weights, kPromptMixedWeights, width, input_vocab, special_row);
    const std::vector<float> vocal = read_rows_f32(weights, kPromptVocalWeights, width, input_vocab, special_row);
    const std::vector<float> bgm = read_rows_f32(weights, kPromptBgmWeights, width, input_vocab, special_row);
    const std::vector<float> eot_mixed = read_rows_f32(weights, kPromptEotMixed, width, 1, eot_row);
    const std::vector<float> eot_detail = read_rows_f32(weights, kPromptEotDetail, width, 1, eot_row);
    const bool prompt_f16 = weights.tensor(kPromptVocalWeights)->type == GGML_TYPE_F16;
    condition_tensor prompt_main{static_cast<int32_t>(width), prompt_length,
                                 std::vector<float>(static_cast<std::size_t>(width) * prompt_length)};
    condition_tensor prompt_detail{static_cast<int32_t>(width), prompt_length,
                                   std::vector<float>(static_cast<std::size_t>(width) * prompt_length)};
    for (int32_t d = 0; d < width; ++d) {
        prompt_main.values[d] = eot_mixed[d];
        prompt_detail.values[d] = eot_detail[d];
    }
    // v0.1 has no audio-prompt input. Upstream first prepends EOS, then masks
    // the entire stream when frame zero is special; for an all-special null
    // prompt this overwrites that EOS too. Both conditional and CFG-null
    // branches therefore contain EOT followed by 251 special-token frames.
    for (int32_t t = 0; t < frames + 1; ++t) {
        for (int32_t d = 0; d < width; ++d) {
            const std::size_t at = static_cast<std::size_t>(t + 1) * width + d;
            prompt_main.values[at] = mixed[static_cast<std::size_t>(d)];
            const float vocal_value = vocal[static_cast<std::size_t>(d)];
            const float bgm_value = bgm[static_cast<std::size_t>(d)];
            prompt_detail.values[at] = prompt_f16 ? f16_add(vocal_value, bgm_value)
                                                   : vocal_value + bgm_value;
        }
    }
    result.main.width = static_cast<int32_t>(width);
    result.main.prefix = hp.lyrics_prefix_length + hp.prompt_prefix_length + hp.style_prefix_length;
    result.main.values.reserve(lyric.values.size() + prompt_main.values.size() + style.values.size());
    result.main.values.insert(result.main.values.end(), lyric.values.begin(), lyric.values.end());
    result.main.values.insert(result.main.values.end(), prompt_main.values.begin(), prompt_main.values.end());
    result.main.values.insert(result.main.values.end(), style.values.begin(), style.values.end());
    result.detail = result.main;
    result.detail.values.clear();
    result.detail.values.reserve(lyric.values.size() + prompt_detail.values.size() + style.values.size());
    result.detail.values.insert(result.detail.values.end(), lyric.values.begin(), lyric.values.end());
    result.detail.values.insert(result.detail.values.end(), prompt_detail.values.begin(), prompt_detail.values.end());
    result.detail.values.insert(result.detail.values.end(), style.values.begin(), style.values.end());
    return result;
}

} // namespace

conditioner::conditioner(std::shared_ptr<const model> model,
                         const ByteLevelBPETokenizer & tokenizer)
    : model_(std::move(model)), tokenizer_(&tokenizer) {
    if (!model_) {
        fail("a model is required");
    }
    if (tokenizer_ == nullptr) {
        fail("a tokenizer is required");
    }
}

conditioning_result conditioner::prepare(const std::string & lyrics,
                                         const std::string & description) const {
    if (!model_ || tokenizer_ == nullptr) {
        fail("conditioner is not initialized");
    }
    conditioning_result result;
    result.conditional = make_branch(*model_, *tokenizer_, lyrics, description);
    result.null_condition = make_branch(*model_, *tokenizer_, {}, {});
    return result;
}

conditioning_result prepare_conditioning(const model & weights,
                                         const ByteLevelBPETokenizer & tokenizer,
                                         const std::string & lyrics,
                                         const std::string & description) {
    conditioning_result result;
    result.conditional = make_branch(weights, tokenizer, lyrics, description);
    result.null_condition = make_branch(weights, tokenizer, {}, {});
    return result;
}

} // namespace levo::detail
