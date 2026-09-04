#include "tensor_ops.h"
#include "tensor_jit.h"
#include "../../common/dev_tools/profiler/internal_trace.h"

vtensor *tensor_ccm(const vtensor *input, const vtensor *matrix,
                    uint32 white_level) {
    if (!input || !matrix || input->rank != 3 || input->shape[2] != 3 ||
        input->dtype != UINT16 || matrix->rank != 2 ||
        matrix->shape[0] != 3 || matrix->shape[1] != 3 ||
        matrix->dtype != REAL32)
        return NULL;
    vtensor *output = tensor_lazy_alloc(3, input->shape, UINT16,
                                        0, false, false, NULL);
    const vtensor *inputs[] = {input, matrix};
    if (!output || tensor_lazy_op_dispatch(ccm_ker, output, inputs,
                                            &white_level)) {
        tensor_lazy_free(output);
        return NULL;
    }
    return output;
}

boolean ccm_lower(tensor *output, const tensor *const *inputs,
                  extent input_count, const void *parameters,
                  extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("CCM", ccm_ker.op, output, inputs, input_count);
    if (!output || !inputs || input_count != 2 || !inputs[0] || !inputs[1] ||
        output->dtype != UINT16 || inputs[0]->dtype != UINT16 ||
        inputs[1]->dtype != REAL32 || output->rank != 3 ||
        inputs[0]->rank != 3 || inputs[1]->rank != 2 ||
        output->size != inputs[0]->size || output->shape[2] != 3 ||
        inputs[1]->size != 9 || parameter_bytes != sizeof(uint32))
        return true;
    jit_cache_key_t key = tensor_jit_key_create(ccm_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel || tensor_ker_dispatch(kernel, output, inputs, input_count,
                                           parameters, parameter_bytes);
}
