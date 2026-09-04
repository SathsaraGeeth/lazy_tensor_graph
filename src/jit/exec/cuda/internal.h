#ifndef TENSOR_JIT_EXEC_CUDA_INTERNAL_H
#define TENSOR_JIT_EXEC_CUDA_INTERNAL_H

#include "tensor_graph.h"

typedef struct {
    int kind;
    vtensor *node;
} exec_work;

enum {
    EXEC_ALLOC,
    EXEC_INIT,
    EXEC_KERNEL
};

#ifdef __cplusplus
extern "C" {
#endif

boolean cuda_allocate(vtensor *node);
boolean cuda_initialize(vtensor *node);

#ifdef __cplusplus
}
#endif

#endif
