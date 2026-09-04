/* 
 * user/include/tensor/tensor_core.h
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

/*
 * Comments:
 * 1. Eager
 * 2. FROZEN
 */

#ifndef TENSOR_CORE_H
#define TENSOR_CORE_H

#include "dtype.h"
#include "memory.h"
#include <stddef.h>
#include <stdint.h>


typedef struct tensor tensor;

typedef boolean (*ker_t)(tensor *output, const tensor *const *inputs, extent input_count,
                         const void *parameters, extent parameter_bytes);

struct tensor {
    mem_block*   data;
    extent*      shape;
    stride*      strides;
    extent       rank;
    dtype_t      dtype;
    extent       size;
    boolean      is_contiguous;
    boolean      error;
};

extern  tensor ERROR_TENSOR;

tensor* tensor_alloc       (extent rank, const extent* shape, dtype_t dtype, boolean is_pooled);   // error - error tensor
tensor* tensor_view_from   (tensor* t, const dptr* data);                                          // error - error tensor
boolean tensor_ker_dispatch(ker_t ker, tensor* output, const tensor* const* inputs,
                            extent input_count, const void* parameters, extent parameter_bytes);   // error - 1
boolean tensor_free        (tensor* t);                                                            // error - 1
dptr*   tensor_view_to     (const tensor* t);                                                      // error - null

#endif /* TENSOR_CORE_H */
