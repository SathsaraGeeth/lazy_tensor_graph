/*
 * user/src/tensor/src/jit/gen/cpu/abstract_parallel/abstract_parallel.h
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
 * 1. Reference: https://llvm.org/doxygen/classllvm_1_1Loop.html
 * 2. TODO: look at here 
 *    https://llvm.org/devmtg/2018-04/slides/Finkel-Representing%20Parallelism%20Within%20LLVM.pdf
 * 3. Here currently the llvm production autovectorization is used
 *    - loop_vecotrizer: widen the loop
 *    - SLP vectorizer:  combine similar independent scalar instructions
 * 4. The kernel generator is expected to has one outer loop whose 
 *    iterations are independent
 *    then this partition that loop across the thread pool and
 *    modify the original kernel into a worker function (and rename)
 *    then use the original name for the dispatcher of the worker dispatcher
 *    e.g.
 *           bool matmul(...) {
 *              for (i = 0; i < M; ++i)
 *                   ...
 *           }
 *           into->
 *           bool matmul_worker(...) {
 *               start = M * worker_id / worker_count;
 *               end   = M * (worker_id + 1) / worker_count;
 *
 *               for (i = start; i < end; ++i)
 *                    ...
 *            }
 *           bool matmul(...) {
 *              return parallel_run(matmul_worker, ...);
 *           }
 * 
 * 5. .op--parser-->SSA scalar(as .bc)->
 *     graph.c prefethcer or serilized file->this_file->vectorizer->orc_jit.cpp->jit_cache
 * 6. The order matter parrallize->vectorize
 * 7. This make a thread pool one subgraph's work queue
 *    so thread init dont happen per kernel calls
 */

#ifndef TENSOR_JIT_CPU_ABSTRACT_PARALLEL_H
#define TENSOR_JIT_CPU_ABSTRACT_PARALLEL_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
#include <cstdint>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <string>
extern "C" {
#endif

void tensor_cpu_pool_begin(void);

typedef bool (*tensor_cpu_parallel_kernel_t)(
    void *, const void *const *, uint64_t, const void *, uint64_t);

bool tensor_cpu_parallel_run(
    tensor_cpu_parallel_kernel_t kernel,
    void *output,
    const void *const *inputs,
    uint64_t input_count,
    const void *parameters,
    uint64_t parameter_bytes,
    uint64_t operations);

uint64_t tensor_cpu_parallel_index(void);
uint64_t tensor_cpu_parallel_count(void);

#ifdef __cplusplus
}

bool tensor_cpu_abstract_parallelize(
    llvm::orc::ThreadSafeModule &module,
    const std::string &kernel_name,
    std::uint32_t operation);
#endif

#endif /* TENSOR_JIT_CPU_ABSTRACT_PARALLEL_H */
