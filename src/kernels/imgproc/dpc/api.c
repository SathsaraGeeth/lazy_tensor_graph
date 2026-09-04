#include "tensor_ops.h"
#include "tensor_jit.h"
#include "../../common/dev_tools/profiler/internal_trace.h"

vtensor *tensor_dpc(const vtensor *input, uint32 threshold) {
    if (!input || input->rank != 2 || input->dtype != UINT16) return NULL;
    vtensor *output = tensor_lazy_alloc(2, input->shape, UINT16,
                                        0, false, false, NULL);
    const vtensor *inputs[] = {input};
    if (!output || tensor_lazy_op_dispatch(dpc_ker, output, inputs, &threshold)) {
        tensor_lazy_free(output);
        return NULL;
    }
    return output;
}

boolean dpc_lower(tensor *output, const tensor *const *inputs,
                  extent input_count, const void *parameters,
                  extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("DPC", dpc_ker.op, output, inputs, input_count);
    if (!output || !inputs || input_count != 1 || !inputs[0] ||
        output->dtype != UINT16 || inputs[0]->dtype != UINT16 ||
        output->rank != 2 || inputs[0]->rank != 2 ||
        output->size != inputs[0]->size || parameter_bytes != sizeof(uint32))
        return true;
    jit_cache_key_t key = tensor_jit_key_create(dpc_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel || tensor_ker_dispatch(kernel, output, inputs, input_count,
                                           parameters, parameter_bytes);
}
