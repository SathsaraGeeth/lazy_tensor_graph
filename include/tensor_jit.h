/*
 * user/include/tensor/tensor_jit.h
 *
 * Copyright (C) 2026 Sathsara Geeth
 *
 */

/*
 * Version 1.0
 *
 * Version History
 *
 * Version | Description
 * --------+-----------------------------------------
 * 1.0     | Initial implementation
 */

/*
 * Comments:
 * 1. FROZEN
 */

#ifndef TENSOR_JIT_H
#define TENSOR_JIT_H

#include "device.h"
#include "tensor_core.h"

typedef void (*jit_ker_t)(void);

typedef enum {
    CACHE_COMPILING,
    CACHE_READY,
    CACHE_FAILED
} jit_cache_state_t;

typedef struct {
    dtype_t  dtype;
    extent   rank;
    extent  *shape;
} jit_tensor_sig_t;


typedef struct {
    uint32           op;
    device           dev;
    uint64           compatibility_hash;
    jit_tensor_sig_t output;
    jit_tensor_sig_t *inputs;
    extent           num_inputs;
    void            *storage;
} jit_cache_key_t;

typedef struct jit_cache *jit_cache_t;

#ifdef __cplusplus
extern "C" {
#endif

jit_cache_t     tensor_jit_cache_create(void);
void            tensor_jit_cache_destroy(jit_cache_t cache);
jit_cache_key_t tensor_jit_key_create(uint32 op, const tensor *output,
                const tensor *const *inputs, extent num_inputs, device dev);
jit_ker_t       tensor_jit_get(jit_cache_key_t key, jit_cache_t cache);
boolean         tensor_jit_prepare(jit_cache_key_t key, jit_cache_t cache);

/*
 * The runtime cache init
 */
jit_cache_t     tensor_get_jit_cache(void);


/* persistent jit cache extension */

#ifndef PRESISTENT_JIT_CACHE
#define PRESISTENT_JIT_CACHE 1
#define PRESISTENT_JIT_CACHE_PATH "/tmp/tensor-jit-"
#endif

boolean         tensor_jit_cache_store(jit_cache_t cache, const char *path);
boolean         tensor_jit_cache_load (const char *path, jit_cache_t cache);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_JIT_H */
