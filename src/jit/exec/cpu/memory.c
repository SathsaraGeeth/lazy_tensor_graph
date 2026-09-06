#include "internal.h"
#include "dev_tools/profiler/trace.h"

#include <stdint.h>
#include <stdlib.h>

static mem_pool *pool;
static free_block *available;

void graph_shutdown(void);
uint64 graph_node_id(const vtensor *node);

static void free_available(void) {
    while (available) {
        free_block *entry = available;
        available = entry->next;
        mem_free(entry->block);
        free(entry);
    }
}

static void shutdown(void) {
    free_available();
    pool = NULL;
    graph_shutdown();
}

static mem_block *take_block(exec_context *context, extent bytes) {
    free_block **cursor = &context->free_blocks;
    while (*cursor && (*cursor)->block->size < bytes) cursor = &(*cursor)->next;
    if (!*cursor) return NULL;
    free_block *entry = *cursor;
    mem_block *block = entry->block;
    *cursor = entry->next;
    free(entry);
    return block;
}

static mem_block *alias_block(const mem_block *source) {
    if (!source || !source->ptr || !source->size) return NULL;
    mem_block *block = malloc(sizeof(*block));
    if (!block) return NULL;
    *block = (mem_block){.ptr = source->ptr, .size = source->size};
    return block;
}

static tensor *tensor_from_block(extent rank, const extent *shape, dtype_t dtype, mem_block *block) {
    if (!rank || !shape || !dtype_size(dtype) || !block || !block->ptr) return NULL;
    tensor *value = calloc(1, sizeof(*value));
    if (!value) return NULL;
    value->shape = malloc(rank * sizeof(*value->shape));
    value->strides = malloc(rank * sizeof(*value->strides));
    if (!value->shape || !value->strides) {
        free(value->shape);
        free(value->strides);
        free(value);
        return NULL;
    }
    value->rank = rank;
    value->dtype = dtype;
    value->size = 1;
    value->is_contiguous = true;
    value->data = block;
    for (extent i = 0; i < rank; ++i) {
        value->shape[i] = shape[i];
        value->size *= shape[i];
    }
    ptrdiff_t stride = 1;
    for (extent i = rank; i; --i) {
        value->strides[i - 1] = stride;
        stride *= (ptrdiff_t)value->shape[i - 1];
    }
    if (value->size * dtype_size(dtype) <= block->size) return value;
    value->data = NULL;
    tensor_free(value);
    return NULL;
}

boolean exec_context_begin(exec_context *context) {
    if (!context) return true;
    if (!pool) {
        pool = mem_pool_init();
        if (!pool) return true;
        mem_pool_set_cleanup(pool, shutdown);
    }
    context->free_blocks = available;
    available = NULL;
    return false;
}

void exec_context_end(exec_context *context) {
    if (!context || !context->free_blocks) return;
    free_block *tail = context->free_blocks;
    while (tail->next) tail = tail->next;
    tail->next = available;
    available = context->free_blocks;
    context->free_blocks = NULL;
}

boolean exec_allocate(exec_context *context, vtensor *node) {
    mem_block *block = NULL;
    tensor_init_t *init = node->edge && node->edge->kind == ALLOC ? node->edge->init : NULL;
    if (init && init->init_with_data && init->init_static_data) block = alias_block(init->data_block);
    if (!block) {
        extent bytes = dtype_size(node->dtype);
        for (extent i = 0; i < node->rank; ++i) bytes *= node->shape[i];
        block = take_block(context, bytes);
    }
    node->phy_tensor = block ? tensor_from_block(node->rank, node->shape, node->dtype, block)
                             : tensor_alloc(node->rank, node->shape, node->dtype, true);
    if (!node->phy_tensor || node->phy_tensor->error) {
        if (block) mem_free(block);
        node->phy_tensor = NULL;
        return true;
    }
    tensor_profile_record("NODE_PHYSICAL", "tensor",
                          (uint64)(uintptr_t)node->phy_tensor,
                          graph_node_id(node), 0, node->dtype,
                          node->phy_tensor->data->size, 0);
    return false;
}

boolean exec_initialize(vtensor *node) {
    tensor_init_t *init = node && node->edge ? node->edge->init : NULL;
    if (!init || !node->phy_tensor) return true;
    if (init->init_with_data && init->data_block) {
        boolean failed = init->init_static_data ? mem_view(node->phy_tensor->data, init->data_block->ptr)
                                                : mem_copy(node->phy_tensor->data, init->data_block,
                                                           init->data_block->size);
        if (failed) return true;
    }
    node->state = MAT;
    return false;
}

void exec_recycle(exec_context *context, vtensor *node) {
    if (!context || !node || node->num_live_static_ref || !node->phy_tensor) return;
    mem_block *block = node->phy_tensor->data;
    node->phy_tensor->data = NULL;
    tensor_free(node->phy_tensor);
    node->phy_tensor = NULL;
    node->state = UNMAT;
    if (!block->is_owner || !block->ptr) {
        mem_free(block);
        return;
    }
    free_block *entry = malloc(sizeof(*entry));
    if (!entry) {
        mem_free(block);
        return;
    }
    *entry = (free_block){.block = block, .next = context->free_blocks};
    context->free_blocks = entry;
}
