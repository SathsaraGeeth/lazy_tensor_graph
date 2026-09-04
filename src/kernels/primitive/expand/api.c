#include "tensor_ops.h"

vtensor *tensor_expand(const vtensor *input, extent rank, const extent *shape) {
    if (!input || rank != input->rank) return NULL;
    const extent *input_shape = (const extent *)input->shape;
    for (extent i = 0; i < rank; ++i)
        if (shape[i] != input_shape[i] && input_shape[i] != 1) return NULL;
    vtensor *output = tensor_lazy_alloc(rank, shape, input->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {input};
    tensor_lazy_op_dispatch(expand_ker, output, inputs, NULL);
    return output;
}
