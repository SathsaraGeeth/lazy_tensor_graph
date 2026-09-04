#include "tensor_graph.h"
#include "tensor_jit.h"

#include <stdlib.h>

static jit_cache_t cache;

static const char *persistent_cache_path(void) {
    const char *override = getenv("TENSOR_JIT_CACHE_PATH");
    if (override && *override) return override;
    if (get_device() == CPU) return "/tmp/tensor-jit-cpu.cache";
    if (get_device() == CUDA) return "/tmp/tensor-jit-cuda.cache";
    return "/tmp/tensor-jit-generic.cache";
}

jit_cache_t tensor_get_jit_cache(void) {
    if (!cache) {
        cache = tensor_jit_cache_create();
        if (cache) tensor_jit_cache_load(persistent_cache_path(), cache);
    }
    return cache;
}

static tensor metadata_from(const vtensor *source) {
    tensor metadata = {0};
    metadata.shape = source->shape;
    metadata.rank = source->rank;
    metadata.dtype = source->dtype;
    metadata.size = 1;
    metadata.is_contiguous = true;
    for (extent i = 0; i < source->rank; ++i) metadata.size *= source->shape[i];
    return metadata;
}

boolean graph_jit_prepare(vtensor *node) {
    if (!node || !node->edge || node->edge->kind != IR_NODE) return true;
    if (node->edge->kernel) return false;

    extent count = node->num_parents;
    tensor output = metadata_from(node);
    tensor *input_metadata = count ? malloc(count * sizeof(*input_metadata)) : NULL;
    const tensor **inputs = count ? malloc(count * sizeof(*inputs)) : NULL;
    if (count && (!input_metadata || !inputs)) goto failed;

    for (extent i = 0; i < count; ++i) {
        if (!node->parents[i] || !node->parents[i]->shape) goto failed;
        input_metadata[i] = metadata_from(node->parents[i]);
        inputs[i] = &input_metadata[i];
    }

    jit_cache_key_t key = tensor_jit_key_create(node->edge->op.op, &output, inputs, count, get_device());
    node->edge->kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    free(inputs);
    free(input_metadata);
    return false;

failed:
    free(inputs);
    free(input_metadata);
    return true;
}

void graph_jit_shutdown(void) {
    if (cache) tensor_jit_cache_store(cache, persistent_cache_path());
    tensor_jit_cache_destroy(cache);
    cache = NULL;
}
