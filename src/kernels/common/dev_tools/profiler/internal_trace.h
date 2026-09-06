#ifndef TENSOR_KERNEL_INTERNAL_TRACE_H
#define TENSOR_KERNEL_INTERNAL_TRACE_H

#include "tensor_core.h"
#include "dev_tools/profiler/trace.h"

#include <stdint.h>

typedef struct {
    const char *name;
    uint64_t id;
    uint32 operation;
    dtype_t dtype;
    uint64_t bytes;
} kernel_profile_scope;

static inline kernel_profile_scope kernel_profile_open(
    const char *name, uint32 operation, tensor *output,
    const tensor *const *inputs, extent input_count) {
    kernel_profile_scope scope = {
        .name = name,
        .id = (uint64_t)(uintptr_t)output,
        .operation = operation,
        .dtype = output ? output->dtype : DTYPE_UNKNOWN,
        .bytes = output ? output->size * dtype_size(output->dtype) : 0
    };
    for (extent i = 0; inputs && i < input_count; ++i)
        if (inputs[i])
            scope.bytes += inputs[i]->size * dtype_size(inputs[i]->dtype);
    tensor_profile_record("KERNEL_BEGIN", scope.name, scope.id, 0,
                          scope.operation, scope.dtype, scope.bytes, 0);
    return scope;
}

static inline void kernel_profile_close(kernel_profile_scope *scope) {
    tensor_profile_record("KERNEL_END", scope->name, scope->id, 0,
                          scope->operation, scope->dtype, scope->bytes, 0);
}

#define TENSOR_KERNEL_TRACE(name, operation, output, inputs, input_count) \
    kernel_profile_scope tensor_kernel_profile_scope \
        __attribute__((cleanup(kernel_profile_close))) = \
            kernel_profile_open(name, operation, output, inputs, input_count)

#endif
