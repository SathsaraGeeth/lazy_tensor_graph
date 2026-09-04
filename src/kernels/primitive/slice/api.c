#include "tensor_ops.h"

vtensor *tensor_slice(const vtensor *input, extent rank, const extent *starts, const extent *ends) {
    if (!input || rank != input->rank || rank > 64) return NULL;
    extent shape[64];
    const extent *input_shape = (const extent *)input->shape;
    for (extent i = 0; i < rank; ++i) {
        if (starts[i] > ends[i] || ends[i] > input_shape[i]) return NULL;
        shape[i] = ends[i] - starts[i];
    }
    
    vtensor *output = tensor_lazy_alloc(rank, shape, input->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {input};
    
    typedef struct { extent starts[64]; } slice_params_t;
    slice_params_t params;
    for (extent i = 0; i < rank; ++i)
        params.starts[i] = starts[i];
    
    tensor_lazy_op_dispatch(slice_ker, output, inputs, &params);
    return output;
}
