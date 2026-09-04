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
 * 1. Generates JIT kernels for fused ones
 *    the .op->parser-->SSA scalar(as .bc) is
 *    per .op/kernel they don't aware about fusion
 *    as it is runtime decision made by the graph/
 *    and graph/fusion.c do the fusion thingy
 *    transparenlty to whole library and to the user - it 
 *    rewrites the metadata of the nodes/vtensors so
 *    the graph now is fused and still safe
 *    it chnages the op (and other metas but not the concern of this
 *    file) this file basically parse that new op and
 *    figure out how to generate a fused kernel with
 *    just one outer loop so all the data remains in cpu
 *    regs no round trips to stack
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
