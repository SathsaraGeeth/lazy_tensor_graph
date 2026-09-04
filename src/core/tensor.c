/* 
 * user/src/tensor/src/core/tensor.c
 *
 * Copyright (C) 2026 Sathsara Geeth
 *
 */

/*
 * Version 1.0
 *
 * Version History
 *
 * Version | Description
 * --------+-----------------------------------------
 * 1.0     | Initial implementation
 */

#include "tensor_core.h"
#include "memory.h"
#include <stdlib.h>

tensor ERROR_TENSOR = { .error = 1 };

/* Helpers--- */
static tensor* init_tensor_meta(extent rank, const extent* shape, dtype_t dtype) {
    tensor* t = (tensor*)malloc(sizeof(tensor));
    if (!t) return NULL;

    t->rank          = rank;
    t->dtype         = dtype;
    t->is_contiguous = true;
    t->error         = false;
    t->data          = NULL;
    t->shape         = malloc(rank * sizeof(*t->shape));
    t->strides       = malloc(rank * sizeof(*t->strides));

    if (!t->shape || !t->strides) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    t->size = 1;
    for (extent i = 0; i < rank; i++) {
        t->shape[i] = shape[i];
        t->size     *= shape[i];
    }

    ptrdiff_t stride = 1;
    for (int i = (int)rank - 1; i >= 0; i--) {
        t->strides[i] = stride;
        stride       *= (ptrdiff_t)t->shape[i];
    }

    return t;
}

static extent tensor_storage_size(extent rank, const extent *shape, dtype_t dtype) {
    if (!rank || !shape || !dtype_size(dtype)) return 0;
    extent bytes = dtype_size(dtype);
    for (extent i = 0; i < rank; ++i) bytes *= shape[i];
    return bytes;
}
/*  ---Helpers */



tensor* tensor_alloc(extent rank, const extent* shape, dtype_t dtype, boolean is_pooled) {
    extent bytes = tensor_storage_size(rank, shape, dtype);

    tensor *result = bytes ? init_tensor_meta(rank, shape, dtype) : NULL;

    if (result) {
        result->data = mem_alloc(bytes, is_pooled, NULL);
        if (!result->data) {
            free(result->shape);
            free(result->strides);
            free(result);
            result = NULL;
        }
    }

    return result ? result : &ERROR_TENSOR;
}

tensor* tensor_view_from(tensor* t, const dptr* data) {
    return (!t || t->error || mem_view(t->data, data)) ? &ERROR_TENSOR : t;
}


boolean tensor_ker_dispatch(ker_t ker, tensor* output, const tensor* const* inputs, 
                            extent input_count, const void* parameters, extent parameter_bytes) {
    if (!ker || !output || output->error || !output->data || !output->data->ptr || (input_count && !inputs)) {
            return true;
    } 

    for (extent i = 0; i < input_count; ++i) {
        if (!inputs[i] || inputs[i]->error || !inputs[i]->data || !inputs[i]->data->ptr) {
            return true;
        }
    }
    return ker(output, inputs, input_count, parameters, parameter_bytes);
}

boolean tensor_free(tensor* t) {
    boolean failed = !t || t == &ERROR_TENSOR;
    if (!failed) {
        if (t->data) mem_free(t->data);
        free(t->shape);
        free(t->strides);
        free(t);
    }

    return failed;
}

dptr* tensor_view_to(const tensor* t) {
    return (!t || t->error || !t->data || !t->is_contiguous)
         ? NULL
         : mem_view_to(t->data);
}
