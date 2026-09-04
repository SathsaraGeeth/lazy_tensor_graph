/*
 * user/include/tensor/tensor_ops.h
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
 * 1. FROZEN
 */

#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "kernels.h"
#include "tensor_graph.h"

#define TENSOR_API(return_type, name, args) return_type name args;
#define TENSOR_OPERATION(op, code, ker, lower, inputs, parameters)
#include "../src/kernels/operations.def"
#undef TENSOR_OPERATION
#undef TENSOR_API

#define TENSOR_OPERATION(op, code, ker, lower, inputs, parameters) \
    static const op_t ker = { code, inputs, parameters };
#include "../src/kernels/operations.def"
#undef TENSOR_OPERATION

#endif /* TENSOR_OPS_H */
