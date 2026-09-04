#ifndef TENSOR_KERNEL_INTERNAL_TRACE_H
#define TENSOR_KERNEL_INTERNAL_TRACE_H

#include "dev_tools/profiler/trace.h"

#define TENSOR_KERNEL_TRACE(name, operation, output, inputs, input_count) do { \
    (void)(name);                                                             \
    (void)(operation);                                                        \
    (void)(output);                                                           \
    (void)(inputs);                                                           \
    (void)(input_count);                                                      \
} while (0)

#endif
