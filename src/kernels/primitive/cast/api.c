#include "tensor_ops.h"

vtensor *tensor_cast(const vtensor *input, dtype_t dtype) {
    vtensor *output = tensor_lazy_alloc(input->rank,
                                        (const extent *)input->shape,
                                        dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {input};
    tensor_lazy_op_dispatch(cast_ker, output, inputs, NULL);
    return output;
}
