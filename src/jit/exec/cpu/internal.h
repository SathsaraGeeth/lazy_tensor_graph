#ifndef TENSOR_JIT_EXEC_CPU_INTERNAL_H
#define TENSOR_JIT_EXEC_CPU_INTERNAL_H

#include "tensor_graph.h"

typedef struct free_block {
    mem_block *block;
    struct free_block *next;
} free_block;

typedef struct {
    free_block *free_blocks;
} exec_context;

typedef struct {
    int kind;
    vtensor *node;
} exec_work;

enum {
    EXEC_ALLOC,
    EXEC_INIT,
    EXEC_KERNEL
};

boolean exec_context_begin(exec_context *context);
void exec_context_end(exec_context *context);
boolean exec_allocate(exec_context *context, vtensor *node);
boolean exec_initialize(vtensor *node);
void exec_recycle(exec_context *context, vtensor *node);
boolean exec_prepare_fused(vtensor *node);
boolean exec_execute_fused(vtensor *node, const tensor *const *inputs);
boolean exec_dispatch_operation(uint32 operation, tensor *output, const tensor *const *inputs, extent input_count,
                                const void *parameters, extent parameter_bytes);
boolean exec_prepare_kernel(vtensor *node);
boolean exec_execute_kernel(vtensor *node);

#endif
