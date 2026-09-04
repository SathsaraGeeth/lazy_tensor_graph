/* 
 * user/include/tensor/dtype.h
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
 * 1. Defines the data types
 * 2. FROZEN
 */

#ifndef DTYPE_H
#define DTYPE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef int8_t      int8;
typedef int16_t     int16;
typedef int32_t     int32;
typedef int64_t     int64;
typedef uint8_t     uint8;
typedef uint16_t    uint16;
typedef uint32_t    uint32;
typedef uint64_t    uint64;
typedef float       real32;
typedef double      real64;

typedef bool        boolean;
typedef size_t      extent;
typedef uint64_t    offset;
typedef ptrdiff_t   stride;
typedef void        dptr;

#define DTYPE_INT8_MIN   ((int8)-128)
#define DTYPE_INT8_MAX   ((int8)127)
#define DTYPE_INT16_MIN  ((int16)-32768)
#define DTYPE_INT16_MAX  ((int16)32767)
#define DTYPE_INT32_MIN  ((int32)-2147483647 - 1)
#define DTYPE_INT32_MAX  ((int32)2147483647)
#define DTYPE_INT64_MIN  ((int64)(-9223372036854775807LL - 1))
#define DTYPE_INT64_MAX  ((int64)9223372036854775807LL)
#define DTYPE_UINT8_MAX  ((uint8)255)
#define DTYPE_UINT16_MAX ((uint16)65535)
#define DTYPE_UINT32_MAX ((uint32)4294967295u)
#define DTYPE_UINT64_MAX ((uint64)18446744073709551615ULL)
#define DTYPE_REAL32_MIN ((real32)-__FLT_MAX__)
#define DTYPE_REAL32_MAX ((real32)__FLT_MAX__)
#define DTYPE_REAL64_MIN ((real64)-__DBL_MAX__)
#define DTYPE_REAL64_MAX ((real64)__DBL_MAX__)

typedef enum {
    DTYPE_UNKNOWN = -1,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    REAL32,
    REAL64
} dtype_t;

static inline extent dtype_size(dtype_t type) {
    switch (type) {
        case INT8:   return sizeof(int8);
        case INT16:  return sizeof(int16);
        case INT32:  return sizeof(int32);
        case INT64:  return sizeof(int64);
        case UINT8:  return sizeof(uint8);
        case UINT16: return sizeof(uint16);
        case UINT32: return sizeof(uint32);
        case UINT64: return sizeof(uint64);
        case REAL32: return sizeof(real32);
        case REAL64: return sizeof(real64);
        default:     return 0;
    }
}

static inline const char* dtype_name(dtype_t type) {
    switch (type) {
        case INT8:   return "INT8";
        case INT16:  return "INT16";
        case INT32:  return "INT32";
        case INT64:  return "INT64";
        case UINT8:  return "UINT8";
        case UINT16: return "UINT16";
        case UINT32: return "UINT32";
        case UINT64: return "UINT64";
        case REAL32: return "REAL32";
        case REAL64: return "REAL64";
        default:     return "UNKNOWN";
    }
}

#endif /* DTYPE_H */
