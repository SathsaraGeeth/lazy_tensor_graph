#include "tensor_graph.h"

#include <stdlib.h>

typedef struct {
    int kind;
    vtensor *node;
} graph_work;

typedef struct {
    graph_work *items;
    extent count;
    extent capacity;
} graph_queue;

enum {
    GRAPH_WORK_ALLOC,
    GRAPH_WORK_INIT
};

boolean tensor_exec_submit(const void *items, extent count);

void *graph_do_create(void) {
    return calloc(1, sizeof(graph_queue));
}

boolean graph_do_push(void *opaque_queue, int kind, vtensor *node) {
    graph_queue *queue = opaque_queue;
    if (!queue || !node) return true;
    if (queue->count == queue->capacity) {
        extent capacity = queue->capacity ? queue->capacity * 2 : 16;
        graph_work *items = realloc(queue->items, capacity * sizeof(*items));
        if (!items) return true;
        queue->items = items;
        queue->capacity = capacity;
    }
    queue->items[queue->count++] = (graph_work){.kind = kind, .node = node};
    return false;
}

boolean graph_do_enqueue_memory(void *queue, vtensor *node) {
    if (!node) return true;
    if (!node->phy_tensor && graph_do_push(queue, GRAPH_WORK_ALLOC, node)) return true;
    return node->edge && node->edge->kind == ALLOC ? graph_do_push(queue, GRAPH_WORK_INIT, node) : false;
}

boolean graph_do_submit(void *opaque_queue) {
    graph_queue *queue = opaque_queue;
    return !queue || tensor_exec_submit(queue->items, queue->count);
}

void graph_do_destroy(void *opaque_queue) {
    graph_queue *queue = opaque_queue;
    if (!queue) return;
    free(queue->items);
    free(queue);
}
