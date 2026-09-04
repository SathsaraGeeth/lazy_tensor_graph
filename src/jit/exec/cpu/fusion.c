#include "internal.h"
#include "../../gen/cpu/fusion/fusion.h"

#include <stdlib.h>

static tensor metadata_from(const vtensor *node) {
    tensor value = {0};
    value.shape = node->shape;
    value.rank = node->rank;
    value.dtype = node->dtype;
    value.size = 1;
    value.is_contiguous = true;
    for (extent i = 0; i < node->rank; ++i) value.size *= node->shape[i];
    return value;
}

static jit_ker_t prepare(vtensor *node) {
    if (!node || !node->edge || !node->edge->parameters || !node->edge->op.parameter_bytes) return NULL;
    tensor output = metadata_from(node);
    tensor *external = node->num_parents ? malloc(node->num_parents * sizeof(*external)) : NULL;
    const tensor **inputs = node->num_parents ? malloc(node->num_parents * sizeof(*inputs)) : NULL;
    if (node->num_parents && (!external || !inputs)) {
        free(external);
        free(inputs);
        return NULL;
    }
    for (extent i = 0; i < node->num_parents; ++i) {
        external[i] = metadata_from(node->parents[i]);
        inputs[i] = &external[i];
    }
    jit_ker_t kernel = tensor_cpu_fused_get(node->edge->parameters, node->edge->op.parameter_bytes, &output,
                                            inputs, node->num_parents);
    free(external);
    free(inputs);
    return kernel;
}

boolean exec_prepare_fused(vtensor *node) {
    return !prepare(node);
}

boolean exec_execute_fused(vtensor *node, const tensor *const *inputs) {
    ker_t kernel = (ker_t)tensor_cpu_fused_get(node->edge->parameters, node->edge->op.parameter_bytes,
                                               node->phy_tensor, inputs, node->num_parents);
    return !kernel || tensor_ker_dispatch(kernel, node->phy_tensor, inputs, node->num_parents,
                                          node->edge->parameters, node->edge->op.parameter_bytes);
}
