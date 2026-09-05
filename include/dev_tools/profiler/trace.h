/* 
 * user/src/tensor/include/dev_tools/profiler/shared.h
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
 * 1. Shared memory region definition for profiler.
 */

#ifndef TENSOR_PROFILER_TRACE_H
#define TENSOR_PROFILER_TRACE_H

#include "dtype.h"
#include "dev_tools/profiler/shared.h"

extern  tensor_profile_region *tensor_profile_shared_region;

#define tensor_profile_count_operation(operation) do { \
    tensor_profile_region *_region = tensor_profile_shared_region; \
    if (_region) \
        _region->counters[TENSOR_PROFILE_OPERATION_SLOT(operation)]++; \
} while (0)

#define tensor_profile_count_begin(operation) do { \
    tensor_profile_region *_region = tensor_profile_shared_region; \
    if (_region) \
        _region->counters[TENSOR_PROFILE_BEGIN_SLOT(operation)]++; \
} while (0)

#define tensor_profile_count_end(operation) do { \
    tensor_profile_region *_region = tensor_profile_shared_region; \
    if (_region) \
        _region->counters[TENSOR_PROFILE_END_SLOT(operation)]++; \
} while (0)

#define tensor_profile_count_utility(identifier) do { \
    tensor_profile_region *_region = tensor_profile_shared_region; \
    if (_region) \
        _region->counters[TENSOR_PROFILE_OPERATION_SLOTS + \
                          ((uint32_t)(identifier) & \
                           (TENSOR_PROFILE_UTILITY_SLOTS - 1))]++; \
} while (0)

#endif
