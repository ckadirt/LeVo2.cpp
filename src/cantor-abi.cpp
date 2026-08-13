#include "cantor_engine.h"

#include <cstdlib>
#include <new>
#include <string>

struct cantor_ctx {
    // The first implementation slice deliberately owns no backend/model state.
    // Keeping a concrete context and complete ABI surface lets host discovery
    // and error paths be tested before resumable stages are wired.
    int reserved = 0;
};

namespace {

thread_local cantor_error g_last_error_code = CANTOR_OK;
thread_local std::string g_last_error;

void clear_error() {
    g_last_error_code = CANTOR_OK;
    g_last_error.clear();
}

void set_error(cantor_error code, const char * message) {
    g_last_error_code = code;
    g_last_error = message;
}

} // namespace

extern "C" {

uint32_t cantor_engine_abi_version(void) {
    return CANTOR_ENGINE_ABI;
}

const char * cantor_engine_model(void) {
    return "levo2";
}

const char * cantor_engine_version(void) {
#ifdef LEVO_VERSION
    return LEVO_VERSION;
#else
    return "unknown";
#endif
}

uint32_t cantor_engine_stages(void) {
    // Stage execution is intentionally not advertised until the resumable
    // implementation is wired behind this ABI.
    return 0;
}

cantor_error cantor_engine_last_error_code(void) {
    return g_last_error_code;
}

const char * cantor_engine_last_error(void) {
    return g_last_error.c_str();
}

void cantor_engine_free_blob(uint8_t * blob) {
    std::free(blob);
}

cantor_ctx * cantor_engine_load(const cantor_component * components, size_t count, const cantor_load_opts *) {
    clear_error();
    if (components == nullptr || count == 0) {
        set_error(CANTOR_ERR_MODEL, "[LeVo ABI] at least one model component is required");
        return nullptr;
    }
    for (size_t index = 0; index < count; ++index) {
        if (components[index].role == nullptr || components[index].path == nullptr) {
            set_error(CANTOR_ERR_MODEL, "[LeVo ABI] model component role and path are required");
            return nullptr;
        }
    }
    cantor_ctx * context = new (std::nothrow) cantor_ctx();
    if (context == nullptr) {
        set_error(CANTOR_ERR_OOM, "[LeVo ABI] cannot allocate engine context");
    }
    return context;
}

void cantor_engine_free(cantor_ctx * context) {
    delete context;
}

cantor_status cantor_engine_run_stage(cantor_ctx * context, cantor_stage,
                                      const uint8_t * state_in, size_t,
                                      uint8_t ** state_out, size_t * out_len,
                                      cantor_progress_fn, cantor_cancel_fn, void *) {
    clear_error();
    if (context == nullptr || state_in == nullptr || state_out == nullptr || out_len == nullptr) {
        set_error(CANTOR_ERR_OTHER, "[LeVo ABI] run_stage called with a null argument");
        return CANTOR_ERR;
    }
    *state_out = nullptr;
    *out_len = 0;
    set_error(CANTOR_ERR_OTHER, "[LeVo ABI] no stages are wired in this engine slice");
    return CANTOR_ERR;
}

const float * cantor_engine_audio(cantor_ctx *, int * n_samples, int * sample_rate) {
    if (n_samples) *n_samples = 0;
    if (sample_rate) *sample_rate = 0;
    return nullptr;
}

uint64_t cantor_engine_resident_bytes(cantor_ctx *) {
    return 0;
}

int cantor_engine_resident_modules(cantor_ctx *) {
    return 0;
}

} // extern "C"
