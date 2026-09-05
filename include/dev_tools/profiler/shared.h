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
 * 2. Lock free to reduce syncing overhead.
 * 3. An external process polls this region; profiler data can alias
 *    if the polling process is slow.
 */

#ifndef TENSOR_PROFILER_SHARED_H
#define TENSOR_PROFILER_SHARED_H

#include <stdint.h>

#define TENSOR_PROFILE_MAGIC            UINT64_C(0x5450524f46494c45)
#define TENSOR_PROFILE_VERSION          3u
#define TENSOR_PROFILE_OPERATION_SLOTS  2048u
#define TENSOR_PROFILE_UTILITY_SLOTS    256u
#define TENSOR_PROFILE_PHASE_SLOTS      TENSOR_PROFILE_OPERATION_SLOTS
#define TENSOR_PROFILE_SLOT_COUNT       (TENSOR_PROFILE_OPERATION_SLOTS \
                                         + TENSOR_PROFILE_PHASE_SLOTS \
                                         + TENSOR_PROFILE_UTILITY_SLOTS)

#define TENSOR_PROFILE_OPERATION_SLOT(operation) \
    (((((uint32_t)(operation) >> 16) & 7u) * 256u) + \
     ((uint32_t)(operation) & 255u))

#define TENSOR_PROFILE_BEGIN_SLOT(operation) TENSOR_PROFILE_OPERATION_SLOT(operation)
#define TENSOR_PROFILE_END_SLOT(operation) \
    (TENSOR_PROFILE_OPERATION_SLOTS + TENSOR_PROFILE_OPERATION_SLOT(operation))

typedef struct {
    uint64_t            magic;
    uint32_t            version;
    uint32_t            slot_count;
    uint64_t            process_id;
    volatile uint64_t   counters    [TENSOR_PROFILE_SLOT_COUNT];
} tensor_profile_region;


#if !defined(__riscv)
#include <sys/mman.h>
#endif

#endif
