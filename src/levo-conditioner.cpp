#include "levo-conditioner.h"

#include "ggml-backend.h"

#include <algorithm>
#include <limits>
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

std::size_t element_count(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        fail("null tensor");
    }
    std::size_t count = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0 ||
            static_cast<uint64_t>(tensor->ne[i]) > std::numeric_limits<std::size_t>::max() / count) {
            fail("tensor has invalid dimensions");
        }
        count *= static_cast<std::size_t>(tensor->ne[i]);
    }
    return count;
}

std::vector<float> read_f32(const model & weights, const char * name,
                            int64_t width, int64_t rows) {
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
    const std::size_t count = element_count(tensor);
    std::vector<float> result(count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, result.data(), 0, count * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> values(count);
        ggml_backend_tensor_get(tensor, values.data(), 0, count * sizeof(ggml_fp16_t));
        for (std::size_t i = 0; i < count; ++i) {
            result[i] = ggml_fp16_to_fp32(values[i]);
        }
    } else {
        fail(std::string("tensor '") + name + "' is not F32/F16");
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
    const bool source_f16 = weights.tensor(embedding)->type == GGML_TYPE_F16;
    const std::vector<float> content = read_f32(weights, embedding, width, vocab);
    std::vector<float> structure;
    if (structures) {
        structure = read_f32(weights, kStructureWeights, width, 200);
    }
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
            float value = content[static_cast<std::size_t>(token) * static_cast<std::size_t>(width) + d];
            if (structures && range[static_cast<std::size_t>(t)] != 0) {
                const int64_t row = range[static_cast<std::size_t>(t)];
                const float addition = structure[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + d];
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
    const std::vector<float> mixed = read_f32(weights, kPromptMixedWeights, width, input_vocab);
    const std::vector<float> vocal = read_f32(weights, kPromptVocalWeights, width, input_vocab);
    const std::vector<float> bgm = read_f32(weights, kPromptBgmWeights, width, input_vocab);
    const std::vector<float> eot_mixed = read_f32(weights, kPromptEotMixed, width, 1);
    const std::vector<float> eot_detail = read_f32(weights, kPromptEotDetail, width, 1);
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
    const int64_t special = hp.special_token_id;
    for (int32_t t = 0; t < frames + 1; ++t) {
        const int64_t token = special;
        for (int32_t d = 0; d < width; ++d) {
            const std::size_t at = static_cast<std::size_t>(t + 1) * width + d;
            prompt_main.values[at] = mixed[static_cast<std::size_t>(token) * width + d];
            const float vocal_value = vocal[static_cast<std::size_t>(token) * width + d];
            const float bgm_value = bgm[static_cast<std::size_t>(token) * width + d];
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
