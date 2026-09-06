#include "tensor_graph.h"
#include "dev_tools/profiler/trace.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static uint64_t tensor_probe_time_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;
}

enum {
    GRAPH_WORK_ALLOC,
    GRAPH_WORK_INIT,
    GRAPH_WORK_KERNEL
};

void *graph_do_create(void);
boolean graph_do_push(void *queue, int kind, vtensor *node);
boolean graph_do_submit(void *queue);
void graph_do_destroy(void *queue);
boolean graph_jit_prepare(vtensor *node);
boolean graph_do_enqueue_memory(void *queue, vtensor *node);
const char *graph_operation_name(uint32 operation);
uint64 graph_node_id(const vtensor *node);

typedef struct {
    vtensor **nodes;
    extent count;
    extent capacity;
} walk_state;

static boolean mark_seen(walk_state *state, vtensor *node) {
    for (extent i = 0; i < state->count; ++i)
        if (state->nodes[i] == node) return false;
    if (state->count == state->capacity) {
        extent capacity = state->capacity ? state->capacity * 2 : 32;
        vtensor **nodes = realloc(state->nodes, capacity * sizeof(*nodes));
        if (!nodes) return true;
        state->nodes = nodes;
        state->capacity = capacity;
    }
    state->nodes[state->count++] = node;
    return false;
}

static boolean was_seen(const walk_state *state, const vtensor *node) {
    for (extent i = 0; i < state->count; ++i)
        if (state->nodes[i] == node) return true;
    return false;
}

static boolean discover(vtensor *node, void *queue, walk_state *state) {
    if (!node || !node->edge) return true;
    if (node->state == MAT && node->phy_tensor && node->phy_tensor->data) return false;
    if (was_seen(state, node)) return false;
    if (mark_seen(state, node)) return true;

    uint32 operation = node->edge->kind == IR_NODE ? node->edge->op.op : 0;
    tensor_profile_record("NODE_REQUEST", graph_operation_name(operation),
                          graph_node_id(node), 0, operation, node->dtype, 0, 0);

    for (extent i = 0; i < node->num_parents; ++i)
        if (discover(node->parents[i], queue, state)) return true;

    if (graph_do_enqueue_memory(queue, node)) return true;
    return node->edge->kind == IR_NODE ? graph_do_push(queue, GRAPH_WORK_KERNEL, node) : false;
}

boolean graph_walk(vtensor *output) {
    uint64_t started = tensor_probe_time_ns();
    void *queue = graph_do_create();
    if (!queue) return true;
    walk_state state = {0};
    uint64_t discover_started = tensor_probe_time_ns();
    boolean failed = discover(output, queue, &state);
    uint64_t discover_finished = tensor_probe_time_ns();
    uint64_t submit_started = discover_finished;
    if (!failed) failed = graph_do_submit(queue);
    uint64_t submit_finished = tensor_probe_time_ns();
    free(state.nodes);
    graph_do_destroy(queue);
    static unsigned samples;
    if (getenv("TENSOR_TIMING_PROBES") && samples++ < 32)
        fprintf(stderr,
                "[tensor probe] graph discover=%.3f us submit=%.3f us total=%.3f us\n",
                (discover_finished - discover_started) / 1000.0,
                (submit_finished - submit_started) / 1000.0,
                (submit_finished - started) / 1000.0);
    return failed;
}
