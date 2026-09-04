#include "tensor_ops.h"
#include "tensor_jit.h"
#include "../../common/dev_tools/profiler/internal_trace.h"

typedef struct {
    real32 r_gain, g_gain, b_gain;
    uint32 bayer, white_level;
} awb_params;

vtensor *tensor_awb(const vtensor *input, real32 r_gain, real32 g_gain,
                    real32 b_gain, uint32 bayer, uint32 white_level) {
    if (!input || input->rank != 2 || input->dtype != UINT16) return NULL;
    vtensor *output = tensor_lazy_alloc(2, input->shape, UINT16,
                                        0, false, false, NULL);
    const vtensor *inputs[] = {input};
    awb_params params = {r_gain, g_gain, b_gain, bayer, white_level};
    if (!output || tensor_lazy_op_dispatch(awb_ker, output, inputs, &params)) {
        tensor_lazy_free(output);
        return NULL;
    }
    return output;
}

boolean awb_lower(tensor *output, const tensor *const *inputs,
                  extent input_count, const void *parameters,
                  extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("AWB", awb_ker.op, output, inputs, input_count);
    if (!output || !inputs || input_count != 1 || !inputs[0] ||
        output->dtype != UINT16 || inputs[0]->dtype != UINT16 ||
        output->rank != 2 || inputs[0]->rank != 2 ||
        output->size != inputs[0]->size || parameter_bytes != sizeof(awb_params))
        return true;
    jit_cache_key_t key = tensor_jit_key_create(awb_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel || tensor_ker_dispatch(kernel, output, inputs, input_count,
                                           parameters, parameter_bytes);
}
