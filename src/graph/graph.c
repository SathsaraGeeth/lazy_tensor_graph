#include "tensor_graph.h"
#include "kernels.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static uint64_t tensor_probe_time_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;
}

vtensor *graph_sketch_alloc(extent rank, const extent *shape, dtype_t dtype, extent static_refs, boolean with_data,
                            boolean static_data, dptr *data);
boolean graph_sketch_operation(op_t operation, vtensor *output, const vtensor **inputs, const void *parameters);
boolean graph_walk(vtensor *output);
void graph_jit_shutdown(void);

static vtensor *registry;

uint64 graph_node_id(const vtensor *node) {
    return (uint64)(uintptr_t)node;
}

const char *graph_operation_name(uint32 operation) {
    switch (operation) {
#define TENSOR_OPERATION(name, code, descriptor, lower, input_arity, parameters_size) \
        case code: return #name;
#include "../kernels/operations.def"
#undef TENSOR_OPERATION
        default: return operation ? "UNKNOWN_OP" : "ALLOC";
    }
}

void graph_edge_free(edge_t *edge) {
    if (!edge) return;
    if (edge->kind == ALLOC && edge->init) {
        if (edge->init->data_block) mem_free(edge->init->data_block);
        free(edge->init);
    } else if (edge->kind == IR_NODE) {
        free(edge->parameters);
    }
    free(edge);
}

void graph_shutdown(void) {
    while (registry) {
        vtensor *node = registry;
        registry = node->registry_next;
        free(node->parents);
        if (node->phy_tensor) tensor_free(node->phy_tensor);
        graph_edge_free(node->edge);
        free(node->shape);
        free(node);
    }
    graph_jit_shutdown();
}

void graph_register(vtensor *node) {
    node->registry_next = registry;
    registry = node;
}

static void graph_unregister(vtensor *node) {
    vtensor **cursor = &registry;
    while (*cursor && *cursor != node) cursor = &(*cursor)->registry_next;
    if (*cursor) *cursor = node->registry_next;
}

static void graph_invalidate(void) {
    for (vtensor *node = registry; node; node = node->registry_next)
        if (node->edge && node->edge->kind == IR_NODE) node->state = UNMAT;
}

static boolean graph_release(vtensor *node, boolean release_static_ref) {
    if (!node) return true;
    if (release_static_ref && node->num_live_static_ref) node->num_live_static_ref--;
    if (node->num_live_children || node->num_live_static_ref) return false;

    for (extent i = 0; i < node->num_parents; ++i) {
        vtensor *parent = node->parents[i];
        if (parent->num_live_children) parent->num_live_children--;
        if (!parent->num_live_children && !parent->num_live_static_ref) graph_release(parent, false);
    }
    graph_unregister(node);
    free(node->parents);
    if (node->phy_tensor) tensor_free(node->phy_tensor);
    graph_edge_free(node->edge);
    free(node->shape);
    free(node);
    return false;
}

vtensor *tensor_lazy_alloc(extent rank, const extent *shape, dtype_t dtype, extent num_static_ref, boolean init_with_data,
                           boolean init_data_static, dptr *data) {
    return graph_sketch_alloc(rank, shape, dtype, num_static_ref, init_with_data, init_data_static, data);
}

boolean tensor_lazy_op_dispatch(op_t operation, vtensor *output, const vtensor **inputs, const void *parameters) {
    return graph_sketch_operation(operation, output, inputs, parameters);
}

dptr *tensor_lazy_view_to(const vtensor *tensor) {
    uint64_t started = tensor_probe_time_ns();
    if (!tensor || graph_walk((vtensor *)tensor)) return NULL;
    uint64_t walked = tensor_probe_time_ns();
    if (!tensor->num_live_static_ref) ((vtensor *)tensor)->num_live_static_ref = 1;
    dptr *result = tensor_view_to(tensor->phy_tensor);
    uint64_t finished = tensor_probe_time_ns();
    static unsigned samples;
    if (getenv("TENSOR_TIMING_PROBES") && samples++ < 32)
        fprintf(stderr,
                "[tensor probe] op=%s graph_walk=%.3f us view=%.3f us total=%.3f us\n",
                graph_operation_name(tensor->edge ? tensor->edge->op.op : 0),
                (walked - started) / 1000.0,
                (finished - walked) / 1000.0,
                (finished - started) / 1000.0);
    return result;
}

boolean tensor_lazy_view_from(vtensor *tensor, dptr *data) {
    if (!tensor || !data || !tensor->edge || tensor->edge->kind != ALLOC || !tensor->edge->init ||
        !tensor->edge->init->init_static_data || !tensor->edge->init->data_block) {
        return true;
    }
    if (mem_view(tensor->edge->init->data_block, data)) return true;
    if (tensor->phy_tensor && mem_view(tensor->phy_tensor->data, data)) return true;
    tensor->state = MAT;
    graph_invalidate();
    return false;
}

boolean tensor_lazy_free(vtensor *tensor) {
    return graph_release(tensor, true);
}
