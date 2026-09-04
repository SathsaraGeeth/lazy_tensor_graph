#include "../../common/dev_tools/profiler/internal_trace.h"
#include "tensor_ops.h"
#include "tensor_jit.h"

vtensor *tensor_brightness(const vtensor *input, real32 parameter) {
    if (!input || !input->shape || !input->shape ||
        !input->rank || !dtype_size(input->dtype)) return NULL;
    vtensor *output = tensor_lazy_alloc(input->rank,
                                        (const extent *)input->shape,
                                        input->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {input};
    if (!output || tensor_lazy_op_dispatch(brightness_ker, output, inputs,
                                            &parameter)) {
        tensor_lazy_free(output);
        return NULL;
    }
    return output;
}

boolean brightness_lower(tensor *output, const tensor *const *inputs,
                         extent input_count, const void *parameters,
                         extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("BRIGHTNESS", brightness_ker.op,
                        output, inputs, input_count);
    (void)parameter_bytes;
    jit_cache_key_t key = tensor_jit_key_create(brightness_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    if (!kernel) return true;
    return tensor_ker_dispatch(kernel, output, inputs, input_count, parameters,
                               parameter_bytes);
}
