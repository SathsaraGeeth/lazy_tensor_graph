#include "../../common/dev_tools/profiler/internal_trace.h"
#include "tensor_ops.h"
#include "tensor_jit.h"
#include <stdint.h>
#include "device.h"

vtensor *tensor_matrix_mul(const vtensor *left, const vtensor *right) {
    const extent *left_shape  = left  ? (const extent *)left->shape  : NULL;
    const extent *right_shape = right ? (const extent *)right->shape : NULL;

    if (!left || !right || left->rank != 2 || right->rank != 2 ||
        left->dtype != right->dtype || left_shape[1] != right_shape[0]) {
        return NULL;
    }

    extent shape[]  = {left_shape[0], right_shape[1]};
    vtensor *output = tensor_lazy_alloc(2, shape, left->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {left, right};

    return tensor_lazy_op_dispatch(matrix_mul_ker, output, inputs, NULL) ? NULL : output;
}

boolean matrix_mul_lower(tensor *output, const tensor *const *inputs,
                         extent input_count, const void *parameters,
                         extent parameter_bytes) {

    TENSOR_KERNEL_TRACE("MATRIX_MUL", matrix_mul_ker.op,
                        output, inputs, input_count);

    (void)parameters;

    if (input_count != 2 || parameter_bytes || !output || !inputs ||
        !inputs[0] || !inputs[1] || !output->data || !inputs[0]->data ||
        !inputs[1]->data || output->rank != 2 || inputs[0]->rank != 2 ||
        inputs[1]->rank != 2 || output->dtype != inputs[0]->dtype ||
        output->dtype != inputs[1]->dtype) {

        return true;
    }

    const extent *out_shape = (const extent *)output->shape;
    const extent *a_shape   = (const extent *)inputs[0]->shape;
    const extent *b_shape   = (const extent *)inputs[1]->shape;

    if (a_shape[1] != b_shape[0] || out_shape[0] != a_shape[0] ||
        out_shape[1] != b_shape[1])
        return true;

    jit_cache_key_t key = tensor_jit_key_create(matrix_mul_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    if (!kernel) return true;

    return tensor_ker_dispatch(kernel, output, inputs, input_count, NULL, parameter_bytes);
}
