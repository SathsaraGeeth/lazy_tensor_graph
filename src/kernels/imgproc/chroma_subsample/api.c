#include "tensor_ops.h"
#include "tensor_jit.h"
#include "../../common/dev_tools/profiler/internal_trace.h"

vtensor *tensor_chroma_subsample(const vtensor *input) {
    if (!input || input->rank != 3 || input->shape[2] != 3 ||
        input->dtype != UINT16 || input->shape[0] % 2 ||
        input->shape[1] % 2)
        return NULL;
    extent shape[] = {input->shape[0] + input->shape[0] / 2, input->shape[1]};
    vtensor *output = tensor_lazy_alloc(2, shape, UINT16,
                                        0, false, false, NULL);
    const vtensor *inputs[] = {input};
    if (!output ||
        tensor_lazy_op_dispatch(chroma_subsample_ker, output, inputs, NULL)) {
        tensor_lazy_free(output);
        return NULL;
    }
    return output;
}

boolean chroma_subsample_lower(tensor *output, const tensor *const *inputs,
                               extent input_count, const void *parameters,
                               extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("CHROMA_SUBSAMPLE", chroma_subsample_ker.op,
                        output, inputs, input_count);
    (void)parameters;
    if (!output || !inputs || input_count != 1 || !inputs[0] ||
        output->dtype != UINT16 || inputs[0]->dtype != UINT16 ||
        output->rank != 2 || inputs[0]->rank != 3 ||
        inputs[0]->shape[2] != 3 ||
        output->shape[0] != inputs[0]->shape[0] * 3 / 2 ||
        output->shape[1] != inputs[0]->shape[1] || parameter_bytes)
        return true;
    jit_cache_key_t key = tensor_jit_key_create(chroma_subsample_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel || tensor_ker_dispatch(kernel, output, inputs, input_count,
                                           NULL, 0);
}
