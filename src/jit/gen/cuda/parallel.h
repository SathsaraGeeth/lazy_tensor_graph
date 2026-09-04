#ifndef TENSOR_JIT_GEN_CUDA_PARALLEL_H
#define TENSOR_JIT_GEN_CUDA_PARALLEL_H

#include <llvm/IR/IRBuilder.h>

namespace llvm {
class Function;
}

bool tensor_cuda_parallelize(llvm::Function &function);
llvm::Value *tensor_cuda_global_index(llvm::IRBuilder<> &builder);
void tensor_cuda_grid_sync(llvm::IRBuilder<> &builder);

#endif
