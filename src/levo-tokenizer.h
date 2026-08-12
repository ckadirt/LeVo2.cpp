#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace levo {

class ByteLevelBPETokenizer {
public:
    ByteLevelBPETokenizer() = default;

    // Loads the GPT-2/Qwen2 vocab and merge-rank files. tokenizer_config_json
    // is optional, but when present it supplies added/special token IDs.
    static ByteLevelBPETokenizer load(const std::string & vocab_json,
                                      const std::string & merges_txt,
                                      const std::string & tokenizer_config_json = {});
    // tokenizer.json contains model.vocab/model.merges and added_tokens.
    static ByteLevelBPETokenizer load_tokenizer_json(const std::string & tokenizer_json,
                                                     const std::string & tokenizer_config_json = {});
    // Constructs directly from the self-contained GGUF tokenizer metadata.
    static ByteLevelBPETokenizer load_embedded(
        const std::vector<std::string> & tokens,
        const std::vector<std::string> & merges,
        const std::string & added_tokens_json,
        const std::string & tokenizer_config_json = {});

    void add_special_token(const std::string & token, int64_t id = -1);
    std::vector<int64_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int64_t> & ids, bool skip_special_tokens = false) const;

    std::size_t vocab_size() const noexcept { return id_to_token_.size(); }
    bool has_token(const std::string & token) const noexcept { return token_to_id_.find(token) != token_to_id_.end(); }
    int64_t token_id(const std::string & token) const;
    const std::string & token_string(int64_t id) const;
    const std::unordered_map<std::string, int64_t> & vocabulary() const noexcept { return token_to_id_; }
    const std::unordered_map<std::string, int64_t> & special_tokens() const noexcept { return special_tokens_; }

private:
    std::unordered_map<std::string, int64_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::size_t> merge_rank_;
    std::unordered_map<std::string, int64_t> special_tokens_;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    void load_vocab_object(const std::string & json);
    void load_merges_text(const std::string & text);
    void ensure_id(std::size_t id);
    std::vector<std::string> bpe(const std::string & piece) const;
};

using Qwen2Tokenizer = ByteLevelBPETokenizer;
using GPT2Tokenizer = ByteLevelBPETokenizer;

} // namespace levo
