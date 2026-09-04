#include "../../common/dev_tools/profiler/internal_trace.h"
#include "tensor_ops.h"
#include "tensor_jit.h"

vtensor *tensor_matrix_inv_square(const vtensor *input) {
    const extent *shape = input ? (const extent *)input->shape : NULL;

    if (!input || input->rank != 2 ||
        (input->dtype != REAL32 && input->dtype != REAL64) || shape[0] != shape[1])
        return NULL;

    extent output_shape[] = {shape[0], shape[1]};
    vtensor *output = tensor_lazy_alloc(2, output_shape, input->dtype, 0, false, false, NULL);

    const vtensor *inputs[] = {input};
    return tensor_lazy_op_dispatch(matrix_inv_square_ker, output, inputs, NULL) ? NULL : output;
}

boolean matrix_inv_square_lower(tensor *output, const tensor *const *inputs,
                                extent input_count, const void *parameters,
                                extent parameter_bytes) {

    TENSOR_KERNEL_TRACE("MATRIX_INV_SQUARE", matrix_inv_square_ker.op, output, inputs, input_count);

    (void)parameters;

    if (input_count != 1 || parameter_bytes || !output || !inputs ||
        !inputs[0] || !output->data || !inputs[0]->data ||
        !output->data->ptr || !inputs[0]->data->ptr ||
        output->rank != 2 || inputs[0]->rank != 2 ||
        (output->dtype != REAL32 && output->dtype != REAL64) ||
        output->dtype != inputs[0]->dtype)

        return true;

    const extent *out_shape = (const extent *)output->shape;
    const extent *in_shape  = (const extent *)inputs[0]->shape;

    if (!out_shape || !in_shape || !in_shape[0] ||
        in_shape[0] != in_shape[1] || out_shape[0] != in_shape[0] ||
        out_shape[1] != in_shape[1]) 
        
        return true;

    jit_cache_key_t key = tensor_jit_key_create(matrix_inv_square_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());

    return !kernel || tensor_ker_dispatch(kernel, output, inputs, input_count, NULL, parameter_bytes);
}
