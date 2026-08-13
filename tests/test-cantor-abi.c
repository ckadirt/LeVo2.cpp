#include "cantor_engine.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    assert(cantor_engine_abi_version() == CANTOR_ENGINE_ABI);
    assert(strcmp(cantor_engine_model(), "levo2") == 0);
    assert(cantor_engine_stages() == 0);

    cantor_component component = { "lm", "placeholder.gguf" };
    cantor_ctx * context = cantor_engine_load(&component, 1, NULL);
    assert(context != NULL);
    uint8_t state[] = { '{', '}' };
    uint8_t * output = (uint8_t *) (uintptr_t) 1;
    size_t output_length = 99;
    assert(cantor_engine_run_stage(context, CANTOR_STAGE_CODES, state, sizeof(state),
                                   &output, &output_length, NULL, NULL, NULL) == CANTOR_ERR);
    assert(output == NULL && output_length == 0);
    assert(cantor_engine_last_error_code() == CANTOR_ERR_OTHER);
    assert(strstr(cantor_engine_last_error(), "no stages") != NULL);
    cantor_engine_free(context);
    return 0;
}
