#include "internal.h"
#include "kernels.h"
#include "dev_tools/profiler/trace.h"
#include "../../../dev_tools/profiler/internal.h"

#include <stdint.h>
#include <stdlib.h>

const char *graph_operation_name(uint32 operation);
uint64 graph_node_id(const vtensor *node);
boolean graph_jit_prepare(vtensor *node);

static boolean is_fused(uint32 operation) {
    return (operation & UINT32_C(0xf0000000)) == UINT32_C(0xf0000000) &&
           (operation & UINT32_C(0x0fffffff)) >= 2;
}

static uint64 tensor_bytes(const tensor *value) {
    return value ? value->size * dtype_size(value->dtype) : 0;
}

typedef struct {
    uint32 magic;
    uint32 count;
    extent parameter_bytes;
} profile_fusion_header;

typedef struct {
    uint32 operation;
    uint32 reserved;
    extent input_count;
    extent parameter_bytes;
} profile_fusion_stage;

static uint64 fused_operation_count(const vtensor *node, const tensor *output,
                                    const tensor *const *inputs,
                                    boolean *known) {
    *known = false;
    if (!node->edge->parameters ||
        node->edge->op.parameter_bytes < sizeof(profile_fusion_header))
        return 0;
    const profile_fusion_header *header = node->edge->parameters;
    if (header->magic != UINT32_C(0x46555331)) return 0;
    const profile_fusion_stage *stages =
        (const profile_fusion_stage *)(header + 1);
    const uint8 *parameters = (const uint8 *)(stages + header->count);
    extent external = 0;
    uint64 total = 0;
    *known = true;
    for (uint32 i = 0; i < header->count; ++i) {
        if (stages[i].input_count > 8 ||
            !tensor_profile_has_operation_count(stages[i].operation)) {
            *known = false;
            return 0;
        }
        const tensor *stage_inputs[8];
        for (extent j = 0; j < stages[i].input_count; ++j) {
            if (i && !j) stage_inputs[j] = output;
            else if (external < node->num_parents)
                stage_inputs[j] = inputs[external++];
            else {
                *known = false;
                return 0;
            }
        }
        total += tensor_profile_operation_count(
            stages[i].operation, output, stage_inputs,
            stages[i].input_count, parameters);
        parameters += stages[i].parameter_bytes;
    }
    return total;
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
    uint64 bytes = tensor_bytes(node->phy_tensor);
    for (extent i = 0; i < node->num_parents; ++i) {
        inputs[i] = node->parents[i]->phy_tensor;
        bytes += tensor_bytes(inputs[i]);
    }
    uint32 operation = node->edge->op.op;
    boolean fused = is_fused(operation);
    const char *name = fused ? "FUSED" : graph_operation_name(operation);
    uint64 profile_id = (uint64)(uintptr_t)node->phy_tensor;
    tensor_profile_record("NODE_BEGIN", name, graph_node_id(node), 0,
                          operation, node->dtype, bytes, 0);
    if (!fused)
        tensor_profile_record(
            "OP_METADATA", name, profile_id,
            tensor_profile_has_operation_count(operation), operation,
            node->dtype, bytes,
            tensor_profile_operation_count(operation, node->phy_tensor, inputs,
                                           node->num_parents,
                                           node->edge->parameters));
    else {
        boolean known;
        uint64 operations = fused_operation_count(node, node->phy_tensor,
                                                  inputs, &known);
        tensor_profile_record("OP_METADATA", name, profile_id, known,
                              operation, node->dtype, bytes, operations);
    }
    boolean dnn_host = fused ||
        (operation & UINT32_C(0xffff0000)) == UINT32_C(0x00020000);
    if (dnn_host)
        tensor_profile_record("KERNEL_BEGIN", name, profile_id, 0, operation,
                              node->dtype, bytes, 0);
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
    if (dnn_host)
        tensor_profile_record("KERNEL_END", name, profile_id, 0, operation,
                              node->dtype, bytes, 0);
    tensor_profile_record("NODE_END", name, graph_node_id(node), 0, operation,
                          node->dtype, bytes, 0);
    if (inputs != local_inputs) free(inputs);
    if (!failed) node->state = MAT;
    return failed;
}
