#include "internal.h"

#include <stdlib.h>

static tensor *tensor_from_block(const vtensor *node, mem_block *block) {
    tensor *value = calloc(1, sizeof(*value));
    if (!value) return NULL;
    value->shape = malloc(node->rank * sizeof(*value->shape));
    value->strides = malloc(node->rank * sizeof(*value->strides));
    if (!value->shape || !value->strides) {
        free(value->shape);
        free(value->strides);
        free(value);
        return NULL;
    }
    value->rank = node->rank;
    value->dtype = node->dtype;
    value->size = 1;
    value->is_contiguous = true;
    value->data = block;
    for (extent i = 0; i < node->rank; ++i) {
        value->shape[i] = node->shape[i];
        value->size *= node->shape[i];
    }
    stride step = 1;
    for (extent i = node->rank; i; --i) {
        value->strides[i - 1] = step;
        step *= (stride)value->shape[i - 1];
    }
    return value;
}

boolean cuda_allocate(vtensor *node) {
    if (!node || !node->rank || !node->shape) return true;
    extent bytes = dtype_size(node->dtype);
    for (extent i = 0; i < node->rank; ++i) bytes *= node->shape[i];
    mem_block *block = mem_alloc(bytes, false, NULL);
    node->phy_tensor = block ? tensor_from_block(node, block) : NULL;
    if (node->phy_tensor) return false;
    mem_free(block);
    return true;
}

boolean cuda_initialize(vtensor *node) {
    tensor_init_t *init = node && node->edge ? node->edge->init : NULL;
    if (!init || !node->phy_tensor || !node->phy_tensor->data) return true;
    if (init->init_with_data && init->data_block &&
        mem_view_copy(node->phy_tensor->data, init->data_block->ptr)) return true;
    node->state = MAT;
    return false;
}
