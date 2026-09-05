/*
 * user/src/tensor/src/jit/gen/cpu/fusion/fusion.h
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
 * 1. Generates JIT kernels for fused operations.
 *    The .op->parser-->SSA scalar(as .bc) is per .op/kernel; they are
 *    not aware of fusion, as it is a runtime decision made by
 *    graph/graph/fusion.c. It performs fusion transparently to the whole
 *    library and user. It rewrites node/vtensor metadata so the graph
 *    is fused and still safe. It changes the op (and other metadata,
 *    but that is not the concern of this file). This file parses that
 *    new op and figures out how to generate a fused kernel with one
 *    outer loop, so all data remains in CPU registers with no round
 *    trips to the stack.
 */

#ifndef TENSOR_CPU_FUSED_JIT_H
#define TENSOR_CPU_FUSED_JIT_H

#include "tensor_jit.h"

#ifdef __cplusplus
extern "C" {
#endif

jit_ker_t tensor_cpu_fused_get(const void *program, extent program_bytes, const tensor *output,
                               const tensor *const *inputs, extent input_count);

#ifdef __cplusplus
}
#endif

#endif
