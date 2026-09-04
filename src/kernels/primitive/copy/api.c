#include "tensor_ops.h"

vtensor *tensor_copy(const vtensor *input) {
    vtensor *output = tensor_lazy_alloc(input->rank,
                                        (const extent *)input->shape,
                                        input->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {input};
    tensor_lazy_op_dispatch(copy_ker, output, inputs, NULL);
    return output;
}
