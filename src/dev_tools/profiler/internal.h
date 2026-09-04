#ifndef TENSOR_PROFILER_INTERNAL_H
#define TENSOR_PROFILER_INTERNAL_H

#include "tensor_core.h"

boolean tensor_profile_has_operation_count(uint32 operation);
uint64 tensor_profile_operation_count(
    uint32 operation, const tensor *output, const tensor *const *inputs,
    extent input_count, const void *parameters);

#endif
