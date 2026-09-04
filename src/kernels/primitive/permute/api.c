#include "tensor_ops.h"

vtensor *tensor_permute(const vtensor *input, extent rank, const extent *permutation) {
    if (!input || rank != input->rank || rank > 8) return NULL;
    const extent *input_shape = (const extent *)input->shape;
    extent shape[64];
    boolean seen[64] = {false};
    for (extent i = 0; i < rank; ++i) {
        if (permutation[i] >= rank || seen[permutation[i]]) return NULL;
        seen[permutation[i]] = true;
        shape[i] = input_shape[permutation[i]];
    }
    vtensor *output = tensor_lazy_alloc(rank, shape, input->dtype, 0, false, false, NULL);
    extent parameters[8] = {0};
    for (extent i = 0; i < rank; ++i) parameters[i] = permutation[i];
    const vtensor *inputs[] = {input};
    tensor_lazy_op_dispatch(permute_ker, output, inputs, parameters);
    return output;
}
