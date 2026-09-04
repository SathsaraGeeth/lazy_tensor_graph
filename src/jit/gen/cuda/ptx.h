#ifndef TENSOR_CUDA_PTX_H
#define TENSOR_CUDA_PTX_H

#include "tensor_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

boolean tensor_cuda_ptx(const vtensor *const *nodes, extent count, char **ptx, char **kernel_name);
void tensor_cuda_ptx_free(char *text);

#ifdef __cplusplus
}
#endif

#endif
