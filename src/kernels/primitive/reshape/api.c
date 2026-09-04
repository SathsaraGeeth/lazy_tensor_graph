#include "tensor_ops.h"

vtensor *tensor_reshape(const vtensor *input, extent rank, const extent *shape) {
    if (!input || !rank) return NULL;
    extent size = 1;
    for (extent i = 0; i < rank; ++i) size *= shape[i];
    extent input_size = 1;
    const extent *input_shape = (const extent *)input->shape;
    for (extent i = 0; i < input->rank; ++i) input_size *= input_shape[i];
    if (size != input_size) return NULL;
    vtensor *output = tensor_lazy_alloc(rank, shape, input->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {input};
    tensor_lazy_op_dispatch(reshape_ker, output, inputs, NULL);
    return output;
}
