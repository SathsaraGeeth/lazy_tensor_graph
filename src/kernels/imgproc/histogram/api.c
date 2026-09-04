#include "../../common/dev_tools/profiler/internal_trace.h"
#include "tensor_ops.h"
#include "tensor_jit.h"
#include <string.h>

vtensor *tensor_histogram(const vtensor *input, extent bins) {
    if (!input || !input->shape || !input->shape || !input->rank ||
        !dtype_size(input->dtype) || !bins) return NULL;
    extent shape[] = {bins};
    vtensor *output = tensor_lazy_alloc(1, shape, UINT32, 0, false, false, NULL);
    const vtensor *inputs[] = {input};
    if (!output || tensor_lazy_op_dispatch(histogram_ker, output, inputs, NULL)) {
        tensor_lazy_free(output); return NULL;
    }
    return output;
}

vtensor *tensor_histogram_equalization(const vtensor *input, extent bins) {
    if (!input || !bins) return NULL;
    vtensor *histogram = tensor_histogram(input, bins);
    if (!histogram) return NULL;
    extent shape[] = {bins};
    vtensor *table = tensor_lazy_alloc(1, shape, REAL32, 0, false, false, NULL);
    const vtensor *inputs[] = {histogram};
    if (!table || tensor_lazy_op_dispatch(histogram_equalization_ker, table,
                                           inputs, NULL)) {
        tensor_lazy_free(table); return NULL;
    }
    vtensor *output = tensor_lut(input, table);
    if (!output) tensor_lazy_free(table);
    return output;
}

boolean histogram_lower(tensor *output, const tensor *const *inputs,
                        extent input_count, const void *parameters,
                        extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("HISTOGRAM", histogram_ker.op,
                        output, inputs, input_count);
    (void)parameters;
    if (input_count != 1 || parameter_bytes || !output || !inputs ||
        !inputs[0] || !output->data || !inputs[0]->data ||
        !output->data->ptr || !inputs[0]->data->ptr || output->rank != 1 ||
        output->dtype != UINT32 || !output->size)
        return true;
    memset(output->data->ptr, 0, output->size * sizeof(uint32));
    jit_cache_key_t key = tensor_jit_key_create(histogram_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    if (!kernel) return true;
    return tensor_ker_dispatch(kernel, output, inputs, input_count, NULL,
                               parameter_bytes);
}

boolean histogram_equalization_lower(
    tensor *output, const tensor *const *inputs, extent input_count,
    const void *parameters, extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("HISTOGRAM_EQUALIZATION",
                        histogram_equalization_ker.op,
                        output, inputs, input_count);
    (void)parameters;
    if (input_count != 1 || parameter_bytes || !output || !inputs ||
        !inputs[0] || !output->data || !inputs[0]->data ||
        !output->data->ptr || !inputs[0]->data->ptr || output->rank != 1 ||
        inputs[0]->rank != 1 || output->dtype != REAL32 ||
        inputs[0]->dtype != UINT32 || output->size != inputs[0]->size ||
        !output->size)
        return true;
    jit_cache_key_t key = tensor_jit_key_create(histogram_equalization_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel ||
           tensor_ker_dispatch(kernel, output, inputs, input_count, NULL,
                               parameter_bytes);
}
