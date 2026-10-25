#include <stddef.h>
#include <stdio.h>
#include "llama.h"

int main(void) {
    printf("{\"model_size\":%zu,\"context_size\":%zu,"
           "\"moe_cache_size\":%zu,\"abort_callback\":%zu,\"ctx_other\":%zu}\n",
           sizeof(struct llama_model_params), sizeof(struct llama_context_params),
           offsetof(struct llama_context_params, moe_cache_size),
           offsetof(struct llama_context_params, abort_callback),
           offsetof(struct llama_context_params, ctx_other));
    return 0;
}
