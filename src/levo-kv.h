#pragma once

#include "levo-conditioner.h"
#include "levo-lm.h"

#include <cstdint>
#include <memory>

namespace levo::detail {

// Stateful, append-only LeLM execution for one logical sequence.  A session
// owns independent K/V buffers for both transformer towers, so CFG's
// conditional and unconditional sequences use two sessions rather than a
// coupled batch cache.  A session is intentionally not concurrently callable.
class kv_session final {
public:
    kv_session(const kv_session &) = delete;
    kv_session & operator=(const kv_session &) = delete;
    kv_session(kv_session &&) noexcept;
    kv_session & operator=(kv_session &&) noexcept;
    ~kv_session();

    // The default path keeps GGML's F32 activation graph.  Opt in to
    // emulate_fp16_activations only for numerical parity against the pinned
    // PyTorch reference, whose model and activation boundaries are FP16.
    static std::unique_ptr<kv_session> create(std::shared_ptr<const model> model,
                                              bool emulate_fp16_activations = false);

    // Appends every input position to the cache and returns logits for only
    // those appended positions.  Explicit positions must be contiguous with
    // the session's current absolute position.  When positions are omitted,
    // the current absolute position is used.
    [[nodiscard]] lm_output prefill(const lm_input & input);

    // First-call path for a prepared condition branch.  `condition.main` and
    // `condition.detail` are dense [width, prefix] streams; they are prepended
    // to their matching audio embeddings before their respective towers.  The
    // returned logits cover audio positions only.  With explicit audio
    // positions, the first position must equal condition.main.prefix.
    // Conditioning is only valid on an empty (or reset) session.
    [[nodiscard]] lm_output prefill_conditioned(const condition_pair & condition,
                                                const lm_input & audio,
                                                bool capture_trace = false);

    // First-call prefix-only path used by the streaming generator: append the
    // dense conditioning positions to both tower caches, produce no logits,
    // then call decode() with the all-special BOS position.  The condition
    // must contain at least one position and the session must be empty/reset.
    void prefill_conditioned_prefix(const condition_pair & condition);

    // Appends one delayed-pattern position.  This is the steady-state decode
    // operation; its result has token_count == 1.
    [[nodiscard]] lm_output decode(int32_t mixed_token, int32_t vocal_token,
                                   int32_t bgm_token);

    void reset() noexcept;
    [[nodiscard]] int32_t token_count() const noexcept;
    [[nodiscard]] int32_t next_position() const noexcept;
    [[nodiscard]] int32_t capacity() const noexcept;

private:
    explicit kv_session(std::shared_ptr<const model> model, bool emulate_fp16_activations);
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace levo::detail
