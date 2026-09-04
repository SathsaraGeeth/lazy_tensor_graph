#include "internal.h"
#include "kernels.h"
#include "dev_tools/profiler/trace.h"

#include <stdint.h>
#include <stdlib.h>

const char *graph_operation_name(uint32 operation);
boolean graph_jit_prepare(vtensor *node);

static boolean is_fused(uint32 operation) {
    return (operation & UINT32_C(0xf0000000)) == UINT32_C(0xf0000000) &&
           (operation & UINT32_C(0x0fffffff)) >= 2;
}

boolean exec_dispatch_operation(uint32 operation, tensor *output, const tensor *const *inputs, extent input_count,
                                const void *parameters, extent parameter_bytes) {
    boolean failed = true;
    switch (operation) {
#define TENSOR_OPERATION(operation_name, code, descriptor, lower, input_arity, parameters_size) \
        case code: \
            tensor_profile_count_begin(code); \
            failed = tensor_ker_dispatch(lower, output, inputs, input_count, parameters, parameter_bytes); \
            tensor_profile_count_end(code); \
            break;
#include "../../../kernels/operations.def"
#undef TENSOR_OPERATION
        default:
            break;
    }
    return failed;
}

boolean exec_prepare_kernel(vtensor *node) {
    if (!node || !node->edge) return true;
    return is_fused(node->edge->op.op) ? exec_prepare_fused(node)
                                       : graph_jit_prepare(node);
}

boolean exec_execute_kernel(vtensor *node) {
    if (!node || !node->edge || node->edge->kind != IR_NODE || !node->phy_tensor) return true;
    const tensor *local_inputs[8];
    const tensor **inputs = node->num_parents <= 8
                          ? local_inputs
                          : malloc(node->num_parents * sizeof(*inputs));
    if (node->num_parents && !inputs) return true;
    for (extent i = 0; i < node->num_parents; ++i)
        inputs[i] = node->parents[i]->phy_tensor;
    uint32 operation = node->edge->op.op;
    boolean fused = is_fused(operation);
    boolean failed;
    if (fused) {
        failed = exec_execute_fused(node, inputs);
    } else if (node->edge->kernel) {
        tensor_profile_count_begin(operation);
        failed = tensor_ker_dispatch(node->edge->kernel, node->phy_tensor,
                                     inputs, node->num_parents,
                                     node->edge->parameters,
                                     node->edge->op.parameter_bytes);
        tensor_profile_count_end(operation);
    } else {
        failed = exec_dispatch_operation(operation, node->phy_tensor, inputs,
                                         node->num_parents,
                                         node->edge->parameters,
                                         node->edge->op.parameter_bytes);
    }
    if (inputs != local_inputs) free(inputs);
    if (!failed) node->state = MAT;
    return failed;
}
