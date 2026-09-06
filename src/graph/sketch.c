#include "tensor_graph.h"
#include "dev_tools/profiler/trace.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void graph_register(vtensor *node);
void graph_edge_free(edge_t *edge);
boolean graph_fusion_rewrite(vtensor *output);
const char *graph_operation_name(uint32 operation);
uint64 graph_node_id(const vtensor *node);

static extent storage_bytes(extent rank, const extent *shape, dtype_t dtype) {
    extent bytes = dtype_size(dtype);
    for (extent i = 0; bytes && shape && i < rank; ++i) bytes *= shape[i];
    return rank && shape ? bytes : 0;
}

vtensor *graph_sketch_alloc(extent rank, const extent *shape, dtype_t dtype, extent static_refs, boolean with_data,
                            boolean static_data, dptr *data) {
    extent bytes = storage_bytes(rank, shape, dtype);
    if (!bytes || (with_data && !data)) return NULL;

    vtensor *node = calloc(1, sizeof(*node));
    edge_t *edge = calloc(1, sizeof(*edge));
    tensor_init_t *init = calloc(1, sizeof(*init));
    extent *saved_shape = malloc(rank * sizeof(*saved_shape));
    if (!node || !edge || !init || !saved_shape) goto failed;

    memcpy(saved_shape, shape, rank * sizeof(*saved_shape));
    init->shape = saved_shape;
    init->rank = rank;
    init->dtype = dtype;
    init->init_with_data = with_data;
    init->init_static_data = static_data;
    if (with_data) {
        init->data_block = static_data ? mem_view_from(bytes, data)
                                       : mem_alloc(bytes, false, NULL);
        if (!init->data_block) goto failed;
        if (!static_data && mem_view_copy(init->data_block, data)) goto failed;
    }

    edge->kind = ALLOC;
    edge->init = init;
    node->edge = edge;
    node->rank = rank;
    node->shape = saved_shape;
    node->dtype = dtype;
    node->num_live_static_ref = static_refs;
    graph_register(node);
    tensor_profile_record("NODE_ALLOC", "ALLOC", graph_node_id(node), 0, 0,
                          dtype, bytes, 0);
    return node;

failed:
    if (init && init->data_block) mem_free(init->data_block);
    free(saved_shape);
    free(init);
    free(edge);
    free(node);
    return NULL;
}

boolean graph_sketch_operation(op_t operation, vtensor *output, const vtensor **inputs, const void *parameters) {
    if (!output || (operation.input_count && !inputs) || (operation.parameter_bytes && !parameters)) return true;
    for (extent i = 0; i < operation.input_count; ++i)
        if (!inputs[i]) return true;

    vtensor **parents = operation.input_count ? malloc(operation.input_count * sizeof(*parents)) : NULL;
    edge_t *edge = calloc(1, sizeof(*edge));
    if ((operation.input_count && !parents) || !edge) goto failed;

    edge->kind = IR_NODE;
    edge->op = operation;
    if (operation.parameter_bytes) {
        edge->parameters = malloc(operation.parameter_bytes);
        if (!edge->parameters) goto failed;
        memcpy(edge->parameters, parameters, operation.parameter_bytes);
    }
    for (extent i = 0; i < operation.input_count; ++i) {
        parents[i] = (vtensor *)inputs[i];
        parents[i]->num_live_children++;
    }

    graph_edge_free(output->edge);
    free(output->parents);
    output->edge = edge;
    output->parents = parents;
    output->num_parents = operation.input_count;
    output->state = UNMAT;
    if (graph_fusion_rewrite(output)) return true;
    uint32 sketched_operation = output->edge->op.op;
    tensor_profile_record("NODE_OP", graph_operation_name(sketched_operation),
                          graph_node_id(output), 0, sketched_operation,
                          output->dtype, 0, 0);
    for (extent i = 0; i < output->num_parents; ++i)
        tensor_profile_record("EDGE", "dependency", graph_node_id(output),
                              graph_node_id(output->parents[i]),
                              sketched_operation, output->dtype, 0, 0);
    return false;

failed:
    if (edge) free(edge->parameters);
    free(edge);
    free(parents);
    return true;
}
