#include "tensor_ops.h"
#include "tensor_jit.h"
#include "../../common/dev_tools/profiler/internal_trace.h"

typedef struct {
    real32 alpha, beta;
    uint32 white_level;
} blend_params;

vtensor *tensor_blend_rgb2yuv(const vtensor *first, const vtensor *second,
                              real32 alpha, real32 beta, uint32 white_level) {
    if (!first || !second || first->rank != 3 || second->rank != 3 ||
        first->shape[2] != 3 || second->shape[2] != 3 ||
        first->dtype != UINT16 || first->dtype != second->dtype ||
        first->shape[0] != second->shape[0] ||
        first->shape[1] != second->shape[1])
        return NULL;
    vtensor *output = tensor_lazy_alloc(3, first->shape, UINT16,
                                        0, false, false, NULL);
    const vtensor *inputs[] = {first, second};
    blend_params params = {alpha, beta, white_level};
    if (!output || tensor_lazy_op_dispatch(blend_rgb2yuv_ker, output, inputs,
                                            &params)) {
        tensor_lazy_free(output);
        return NULL;
    }
    return output;
}

boolean blend_rgb2yuv_lower(tensor *output, const tensor *const *inputs,
                            extent input_count, const void *parameters,
                            extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("BLEND_RGB2YUV", blend_rgb2yuv_ker.op,
                        output, inputs, input_count);
    if (!output || !inputs || input_count != 2 || !inputs[0] || !inputs[1] ||
        output->dtype != UINT16 || inputs[0]->dtype != UINT16 ||
        inputs[1]->dtype != UINT16 || output->rank != 3 ||
        inputs[0]->rank != 3 || inputs[1]->rank != 3 ||
        output->size != inputs[0]->size || output->size != inputs[1]->size ||
        output->shape[2] != 3 || parameter_bytes != sizeof(blend_params))
        return true;
    jit_cache_key_t key = tensor_jit_key_create(blend_rgb2yuv_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel || tensor_ker_dispatch(kernel, output, inputs, input_count,
                                           parameters, parameter_bytes);
}
