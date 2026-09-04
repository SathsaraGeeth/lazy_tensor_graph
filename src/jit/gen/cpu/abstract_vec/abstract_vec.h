/*
 * user/src/tensor/src/jit/gen/cpu/abstract_vec/abstract_vec.h
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
 * 1. Reference: https://llvm.org/docs/Vectorizers.html
 * 2. TODO: look at here https://llvm.org/docs/SandboxVectorizer.html
 *          it can evaluate the profitability then either commit or
 *          rollback the vecotrization
 * 3. Here currently the llvm production autovectorization is used
 *    - loop_vecotrizer: widen the loop
 *    - SLP vectorizer:  combine similar independent scalar instructions
 * 4. The kernel generator is expected to emit portable scalar SSA
 *    provided that llvm choose the simd_width and instructions
 * 5. .op--parser-->SSA scalar(as .bc)->
 *     graph.c prefethcer or serilized file->parrallizer->this_file->orc_jit.cpp->jit_cache
 * 6.  The order matter parrallize->vectorize
 */

#ifndef TENSOR_JIT_CPU_ABSTRACT_VEC_H
#define TENSOR_JIT_CPU_ABSTRACT_VEC_H

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>


bool tensor_cpu_abstract_vectorize(
    llvm::orc::ThreadSafeModule &module,
    llvm::orc::JITTargetMachineBuilder &target);


#endif /* TENSOR_JIT_CPU_ABSTRACT_VEC_H */
