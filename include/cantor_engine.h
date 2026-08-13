// cantor_engine.h — the only surface the node depends on.
//
// Everything else in this repository is free to change without the node
// caring. Bump CANTOR_ENGINE_ABI on any breaking change to what is declared
// here; the node refuses a version it does not understand, which is what
// stops "update the fork, cut a release, point at it" from turning into a
// silent segfault on someone's machine.

#ifndef CANTOR_ENGINE_H
#define CANTOR_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(CANTOR_ENGINE_BUILD)
#    define CANTOR_API __declspec(dllexport)
#  else
#    define CANTOR_API __declspec(dllimport)
#  endif
#else
#  define CANTOR_API __attribute__((visibility("default")))
#endif

#define CANTOR_ENGINE_ABI 1

CANTOR_API uint32_t cantor_engine_abi_version(void);
CANTOR_API const char * cantor_engine_model(void);
CANTOR_API const char * cantor_engine_version(void);

typedef enum {
    CANTOR_STAGE_PLAN    = 1,
    CANTOR_STAGE_CODES   = 2,
    CANTOR_STAGE_DIFFUSE = 3,
    CANTOR_STAGE_DECODE  = 4,
} cantor_stage;

CANTOR_API uint32_t cantor_engine_stages(void);

typedef enum {
    CANTOR_DONE   = 0,
    CANTOR_PAUSED = 1,
    CANTOR_ERR    = -1,
} cantor_status;

typedef enum {
    CANTOR_OK          = 0,
    CANTOR_ERR_OOM     = 1,
    CANTOR_ERR_MODEL   = 2,
    CANTOR_ERR_BACKEND = 3,
    CANTOR_ERR_CANCEL  = 4,
    CANTOR_ERR_OTHER   = 5,
} cantor_error;

CANTOR_API cantor_error cantor_engine_last_error_code(void);
CANTOR_API const char * cantor_engine_last_error(void);

typedef struct cantor_ctx cantor_ctx;

typedef struct {
    const char * role;
    const char * path;
} cantor_component;

typedef struct {
    uint64_t vram_budget_bytes;
    int      keep_loaded;
    int      vae_chunk;
    int      vae_overlap;
    int      n_threads;
    int      disable_flash_attn;
    int      disable_batch_cfg;
} cantor_load_opts;

CANTOR_API cantor_ctx * cantor_engine_load(const cantor_component * components, size_t n, const cantor_load_opts * opts);
CANTOR_API void cantor_engine_free(cantor_ctx * ctx);

typedef void (*cantor_progress_fn)(cantor_stage stage, int i, int n, void * userdata);
typedef int (*cantor_cancel_fn)(void * userdata);

CANTOR_API cantor_status cantor_engine_run_stage(cantor_ctx *       ctx,
                                      cantor_stage       stage,
                                      const uint8_t *    state_in,
                                      size_t             in_len,
                                      uint8_t **         state_out,
                                      size_t *           out_len,
                                      cantor_progress_fn on_progress,
                                      cantor_cancel_fn   should_cancel,
                                      void *             userdata);

CANTOR_API void cantor_engine_free_blob(uint8_t * blob);
CANTOR_API const float * cantor_engine_audio(cantor_ctx * ctx, int * n_samples, int * sample_rate);
CANTOR_API uint64_t cantor_engine_resident_bytes(cantor_ctx * ctx);
CANTOR_API int cantor_engine_resident_modules(cantor_ctx * ctx);

#ifdef __cplusplus
}
#endif

#endif
