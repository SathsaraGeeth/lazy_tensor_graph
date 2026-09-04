#include "tensor_ops.h"
#include "tensor_jit.h"
#include "../../common/dev_tools/profiler/internal_trace.h"

vtensor *tensor_blur(const vtensor *input) {
    if (!input || input->rank != 3 || input->shape[2] != 3 ||
        input->dtype != UINT16)
        return NULL;
    vtensor *output = tensor_lazy_alloc(3, input->shape, UINT16,
                                        0, false, false, NULL);
    const vtensor *inputs[] = {input};
    if (!output || tensor_lazy_op_dispatch(blur_ker, output, inputs, NULL)) {
        tensor_lazy_free(output);
        return NULL;
    }
    return output;
}

boolean blur_lower(tensor *output, const tensor *const *inputs,
                   extent input_count, const void *parameters,
                   extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("BLUR", blur_ker.op, output, inputs, input_count);
    (void)parameters;
    if (!output || !inputs || input_count != 1 || !inputs[0] ||
        output->dtype != UINT16 || inputs[0]->dtype != UINT16 ||
        output->rank != 3 || inputs[0]->rank != 3 ||
        output->size != inputs[0]->size || output->shape[2] != 3 ||
        parameter_bytes)
        return true;
    jit_cache_key_t key = tensor_jit_key_create(blur_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel || tensor_ker_dispatch(kernel, output, inputs, input_count,
                                           NULL, 0);
}
