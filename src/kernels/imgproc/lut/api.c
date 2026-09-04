#include "../../common/dev_tools/profiler/internal_trace.h"
#include "tensor_ops.h"
#include "tensor_jit.h"
#include <math.h>
#include <stdlib.h>

vtensor *tensor_lut(const vtensor *input, const vtensor *table) {
    if (!input || !table || !input->shape || !input->shape ||
        !table->shape || !table->shape || !input->rank ||
        !dtype_size(input->dtype) || table->rank != 1 ||
        table->dtype != REAL32 ||
        ((const extent *)table->shape)[0] < 2) return NULL;
    vtensor *output = tensor_lazy_alloc(input->rank, (const extent *)input->shape,
                                        input->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {input, table};
    if (!output || tensor_lazy_op_dispatch(lut_ker, output, inputs, NULL)) {
        tensor_lazy_free(output); return NULL;
    }
    return output;
}

vtensor *tensor_gamma_correction(const vtensor *input, real32 gamma, extent entries) {
    if (!input || !input->shape || !input->shape || !input->rank ||
        !dtype_size(input->dtype) || entries < 2) return NULL;
    real32 *table = malloc(entries * sizeof(*table));
    for (extent index = 0; index < entries; ++index)
        table[index] = powf((real32)index / (real32)(entries - 1), gamma);
    extent shape[] = {entries};
    vtensor *lookup = tensor_lazy_alloc(1, shape, REAL32, 0, true, false, (dptr *)table);
    vtensor *output = tensor_lut(input, lookup);
    if (!lookup || !output) tensor_lazy_free(lookup);
    free(table);
    return output;
}

boolean lut_lower(tensor *output, const tensor *const *inputs,
                  extent input_count, const void *parameters,
                  extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("LUT", lut_ker.op, output, inputs, input_count);
    (void)parameters;
    if (input_count != 2 || parameter_bytes || !output || !inputs ||
        !inputs[0] || !inputs[1] || !output->data || !inputs[0]->data ||
        !inputs[1]->data || !output->data->ptr || !inputs[0]->data->ptr ||
        !inputs[1]->data->ptr || output->dtype != inputs[0]->dtype ||
        output->size != inputs[0]->size || inputs[1]->rank != 1 ||
        inputs[1]->dtype != REAL32 || inputs[1]->size < 2)
        return true;
    jit_cache_key_t key = tensor_jit_key_create(lut_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    if (!kernel) return true;
    return tensor_ker_dispatch(kernel, output, inputs, input_count, NULL,
                               parameter_bytes);
}
