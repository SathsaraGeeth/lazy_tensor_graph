#include "../common/dev_tools/profiler/internal_trace.h"
#include "tensor_jit.h"
#include "tensor_ops.h"

static boolean primitive_jit(
    const char *name, uint32 operation, extent expected_inputs,
    tensor *output, const tensor *const *inputs, extent input_count,
    const void *parameters, extent parameter_bytes) {
    TENSOR_KERNEL_TRACE(name, operation, output, inputs, input_count);
    if (!output || !output->data || input_count != expected_inputs ||
        (input_count && !inputs))
        return true;
    for (extent i = 0; i < input_count; ++i)
        if (!inputs[i] || !inputs[i]->data) return true;
    jit_cache_key_t key = tensor_jit_key_create(operation, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel || tensor_ker_dispatch(
        kernel, output, inputs, input_count, parameters, parameter_bytes);
}

#define PRIMITIVE_LOWER(NAME, DESCRIPTOR, INPUTS)                         \
boolean NAME##_lower(                                                     \
    tensor *output, const tensor *const *inputs, extent input_count,       \
    const void *parameters, extent parameter_bytes) {                      \
    return primitive_jit(#NAME, DESCRIPTOR.op, INPUTS, output, inputs,     \
                         input_count, parameters, parameter_bytes);        \
}

PRIMITIVE_LOWER(add, add_ker, 2)
PRIMITIVE_LOWER(copy, copy_ker, 1)
PRIMITIVE_LOWER(cast, cast_ker, 1)
PRIMITIVE_LOWER(reshape, reshape_ker, 1)
PRIMITIVE_LOWER(permute, permute_ker, 1)
PRIMITIVE_LOWER(slice, slice_ker, 1)
PRIMITIVE_LOWER(expand, expand_ker, 1)
