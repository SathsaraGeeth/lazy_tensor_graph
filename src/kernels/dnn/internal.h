#ifndef TENSOR_KERNEL_DISPATCH_H
#define TENSOR_KERNEL_DISPATCH_H

#include "tensor_core.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

static inline boolean tensor_valid(const tensor *value) {
    return value && !value->error && value->data && value->data->ptr;
}

/* ─── Utility ────────────────────────────────────────────────────────────── */

static inline boolean tensors_shape_match(const tensor* a, const tensor* b) {
    if (a->rank != b->rank) return false;
    const extent* sa = (const extent*)a->shape;
    const extent* sb = (const extent*)b->shape;
    for (extent i = 0; i < a->rank; i++)
        if (sa[i] != sb[i]) return false;
    return true;
}

static inline boolean real_only(dtype_t t) { return t == REAL32 || t == REAL64; }
static inline boolean int_only(dtype_t t)  { return t >= INT8 && t <= UINT64; }
static inline boolean any_type(dtype_t t)  { (void)t; return true; }

/* Resize output buffer if needed. */
static inline boolean tensor_prepare_output(tensor* t, extent req_elems) {
    if (!t || !t->data) return false;
    extent bytes = req_elems * dtype_size(t->dtype);
    if (t->data->size != bytes) {
        if (!t->data->is_owner) return false;
        mem_free(t->data);
        t->data = mem_alloc(bytes, false, NULL);
        if (!t->data) { t->error = true; return false; }
        t->size             = req_elems;
        t->is_contiguous    = true;
    }
    return true;
}

/* ─── Dtype Dispatch Macros ──────────────────────────────────────────────── */

/* DTYPE_SWITCH — expands BODY in a typed scope where `T` is the element type.
 * Inside BODY use:  T* _o, const T* _a, const T* _b, const T* _c, extent _n */

#define DTYPE_SWITCH_ANY(dtype, T, BODY) \
    switch (dtype) { \
        case INT8:   { typedef int8   T; BODY; break; } \
        case INT16:  { typedef int16  T; BODY; break; } \
        case INT32:  { typedef int32  T; BODY; break; } \
        case INT64:  { typedef int64  T; BODY; break; } \
        case UINT8:  { typedef uint8  T; BODY; break; } \
        case UINT16: { typedef uint16 T; BODY; break; } \
        case UINT32: { typedef uint32 T; BODY; break; } \
        case UINT64: { typedef uint64 T; BODY; break; } \
        case REAL32: { typedef real32 T; BODY; break; } \
        case REAL64: { typedef real64 T; BODY; break; } \
        default: return &ERROR_TENSOR; \
    }

#define DTYPE_SWITCH_INT(dtype, T, BODY) \
    switch (dtype) { \
        case INT8:   { typedef int8   T; BODY; break; } \
        case INT16:  { typedef int16  T; BODY; break; } \
        case INT32:  { typedef int32  T; BODY; break; } \
        case INT64:  { typedef int64  T; BODY; break; } \
        case UINT8:  { typedef uint8  T; BODY; break; } \
        case UINT16: { typedef uint16 T; BODY; break; } \
        case UINT32: { typedef uint32 T; BODY; break; } \
        case UINT64: { typedef uint64 T; BODY; break; } \
        default: return &ERROR_TENSOR; \
    }

#define DTYPE_SWITCH_UINT(dtype, T, BODY) \
    switch (dtype) { \
        case UINT8:  { typedef uint8  T; BODY; break; } \
        case UINT16: { typedef uint16 T; BODY; break; } \
        case UINT32: { typedef uint32 T; BODY; break; } \
        case UINT64: { typedef uint64 T; BODY; break; } \
        default: return &ERROR_TENSOR; \
    }

#define DTYPE_SWITCH_REAL(dtype, T, BODY) \
    switch (dtype) { \
        case REAL32: { typedef real32 T; BODY; break; } \
        case REAL64: { typedef real64 T; BODY; break; } \
        default: return &ERROR_TENSOR; \
    }

/* ─── Kernel Macro Templates ─────────────────────────────────────────────── */

/* Unary: output[i] = EXPR(input[i])
 * DTYPE_SW must be one of DTYPE_SWITCH_{ANY,INT,REAL} */
#define KERNEL_UNARY(name, DTYPE_SW, EXPR) \
static void name##_pw(void* _out_e, const void* const* _in_e, \
                      extent _idx, const void* _p, extent _np, dtype_t _dtype, \
                      const extent* _out_shape, extent _out_rank) { \
    (void)_p; (void)_np; (void)_idx; (void)_out_shape; (void)_out_rank; \
    DTYPE_SW(_dtype, _T, { \
        *(_T*)_out_e = (EXPR); \
    }) \
} \
static tensor* name##_impl(tensor* _out, const tensor** _ins, \
                            extent _nin, const void* _p, extent _np) { \
    (void)_p; (void)_np; \
    if (_nin != 1 || !tensor_valid(_ins[0]) || !_ins[0]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(_out) || !_out->data->is_owner || !_out->is_contiguous) return &ERROR_TENSOR; \
    if (_ins[0]->dtype != _out->dtype) return &ERROR_TENSOR; \
    if (!tensors_shape_match(_ins[0], _out)) return &ERROR_TENSOR; \
    extent _n = _out->size; \
    extent _esz = dtype_size(_out->dtype); \
    for (extent _i = 0; _i < _n; _i++) { \
        const void* _ie[1] = { (const uint8_t*)_ins[0]->data->ptr + _i * _esz }; \
        name##_pw((uint8_t*)_out->data->ptr + _i * _esz, _ie, _i, _p, _np, _out->dtype, (extent*)_out->shape, _out->rank); \
    } \
    return _out; \
} \
boolean name##_constr(tensor* o, const tensor** in, extent nin, \
                      const void* p, extent np) { \
    (void)p; (void)np; \
    return nin == 1 && tensor_valid(in[0]) && tensor_valid(o) \
        && in[0]->dtype == o->dtype && tensors_shape_match(in[0], o); \
}

/* Binary: output[i] = EXPR(a[i], b[i]) */
#define KERNEL_BINARY(name, DTYPE_SW, EXPR) \
static void name##_pw(void* _out_e, const void* const* _in_e, \
                      extent _idx, const void* _p, extent _np, dtype_t _dtype, \
                      const extent* _out_shape, extent _out_rank) { \
    (void)_p; (void)_np; (void)_idx; (void)_out_shape; (void)_out_rank; \
    DTYPE_SW(_dtype, _T, { \
        const _T* _a = (const _T*)_in_e[0]; \
        const _T* _b = (const _T*)_in_e[1]; \
        *(_T*)_out_e = (EXPR); \
    }) \
} \
static tensor* name##_impl(tensor* _out, const tensor** _ins, \
                             extent _nin, const void* _p, extent _np) { \
    (void)_p; (void)_np; \
    if (_nin != 2) return &ERROR_TENSOR; \
    if (!tensor_valid(_ins[0]) || !_ins[0]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(_ins[1]) || !_ins[1]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(_out) || !_out->data->is_owner || !_out->is_contiguous) return &ERROR_TENSOR; \
    if (_ins[0]->dtype != _ins[1]->dtype || _ins[0]->dtype != _out->dtype) return &ERROR_TENSOR; \
    if (!tensors_shape_match(_ins[0], _ins[1]) || !tensors_shape_match(_ins[0], _out)) return &ERROR_TENSOR; \
    extent _n = _out->size; \
    extent _esz = dtype_size(_out->dtype); \
    for (extent _i = 0; _i < _n; _i++) { \
        const void* _ie[2] = { (const uint8_t*)_ins[0]->data->ptr + _i * _esz, (const uint8_t*)_ins[1]->data->ptr + _i * _esz }; \
        name##_pw((uint8_t*)_out->data->ptr + _i * _esz, _ie, _i, _p, _np, _out->dtype, (extent*)_out->shape, _out->rank); \
    } \
    return _out; \
} \
boolean name##_constr(tensor* o, const tensor** in, extent nin, \
                      const void* p, extent np) { \
    (void)p; (void)np; \
    return nin == 2 && tensor_valid(in[0]) && tensor_valid(in[1]) \
        && tensor_valid(o) && in[0]->dtype == in[1]->dtype \
        && in[0]->dtype == o->dtype && tensors_shape_match(in[0], o); \
}

/* Ternary real: output[i] = EXPR(a[i], b[i], c[i]), REAL32/REAL64 only */
#define KERNEL_TERNARY_REAL(name, EXPR) \
static void name##_pw(void* _out_e, const void* const* _in_e, \
                      extent _idx, const void* _p, extent _np, dtype_t _dtype, \
                      const extent* _out_shape, extent _out_rank) { \
    (void)_p; (void)_np; (void)_idx; (void)_out_shape; (void)_out_rank; \
    DTYPE_SWITCH_REAL(_dtype, _T, { \
        const _T* _a = (const _T*)_in_e[0]; \
        const _T* _b = (const _T*)_in_e[1]; \
        const _T* _c = (const _T*)_in_e[2]; \
        *(_T*)_out_e = (EXPR); \
    }) \
} \
static tensor* name##_impl(tensor* _out, const tensor** _ins, \
                             extent _nin, const void* _p, extent _np) { \
    (void)_p; (void)_np; \
    if (_nin != 3) return &ERROR_TENSOR; \
    if (!real_only(_out->dtype)) return &ERROR_TENSOR; \
    if (!tensor_valid(_ins[0]) || !_ins[0]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(_ins[1]) || !_ins[1]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(_ins[2]) || !_ins[2]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(_out) || !_out->data->is_owner || !_out->is_contiguous) return &ERROR_TENSOR; \
    if (_ins[0]->dtype != _out->dtype || _ins[1]->dtype != _out->dtype \
        || _ins[2]->dtype != _out->dtype) return &ERROR_TENSOR; \
    if (!tensors_shape_match(_ins[0], _out) || !tensors_shape_match(_ins[1], _out) \
        || !tensors_shape_match(_ins[2], _out)) return &ERROR_TENSOR; \
    extent _n = _out->size; \
    extent _esz = dtype_size(_out->dtype); \
    for (extent _i = 0; _i < _n; _i++) { \
        const void* _ie[3] = { \
            (const uint8_t*)_ins[0]->data->ptr + _i * _esz, \
            (const uint8_t*)_ins[1]->data->ptr + _i * _esz, \
            (const uint8_t*)_ins[2]->data->ptr + _i * _esz \
        }; \
        name##_pw((uint8_t*)_out->data->ptr + _i * _esz, _ie, _i, _p, _np, _out->dtype, (extent*)_out->shape, _out->rank); \
    } \
    return _out; \
} \
boolean name##_constr(tensor* o, const tensor** in, extent nin, \
                      const void* p, extent np) { \
    (void)p; (void)np; \
    return nin == 3 && real_only(o->dtype) \
        && tensor_valid(in[0]) && tensor_valid(in[1]) && tensor_valid(in[2]) \
        && tensor_valid(o) && in[0]->dtype == o->dtype \
        && in[1]->dtype == o->dtype && in[2]->dtype == o->dtype \
        && tensors_shape_match(in[0], o) && tensors_shape_match(in[1], o) \
        && tensors_shape_match(in[2], o); \
}

/* Compare: UINT8 output[i] = (EXPR), inputs any matching dtype */
#define KERNEL_COMPARE(name, EXPR) \
static void name##_pw(void* _out_e, const void* const* _in_e, \
                      extent _idx, const void* _p, extent _np, dtype_t _dtype, \
                      const extent* _out_shape, extent _out_rank) { \
    (void)_p; (void)_np; (void)_idx; (void)_out_shape; (void)_out_rank; \
    DTYPE_SWITCH_ANY(_dtype, _T, { \
        const _T* _a = (const _T*)_in_e[0]; \
        const _T* _b = (const _T*)_in_e[1]; \
        *(uint8*)_out_e = (uint8)(EXPR); \
    }) \
} \
static tensor* name##_impl(tensor* _out, const tensor** _ins, \
                             extent _nin, const void* _p, extent _np) { \
    (void)_p; (void)_np; \
    if (_nin != 2) return &ERROR_TENSOR; \
    if (!tensor_valid(_ins[0]) || !_ins[0]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(_ins[1]) || !_ins[1]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(_out) || !_out->data->is_owner || !_out->is_contiguous) return &ERROR_TENSOR; \
    if (_out->dtype != UINT8) return &ERROR_TENSOR; \
    if (_ins[0]->dtype != _ins[1]->dtype) return &ERROR_TENSOR; \
    if (!tensors_shape_match(_ins[0], _ins[1]) || !tensors_shape_match(_ins[0], _out)) return &ERROR_TENSOR; \
    extent _n = _out->size; \
    extent _esz = dtype_size(_ins[0]->dtype); \
    for (extent _i = 0; _i < _n; _i++) { \
        const void* _ie[2] = { (const uint8_t*)_ins[0]->data->ptr + _i * _esz, (const uint8_t*)_ins[1]->data->ptr + _i * _esz }; \
        name##_pw((uint8_t*)_out->data->ptr + _i, _ie, _i, _p, _np, _ins[0]->dtype, (extent*)_out->shape, _out->rank); \
    } \
    return _out; \
} \
boolean name##_constr(tensor* o, const tensor** in, extent nin, \
                      const void* p, extent np) { \
    (void)p; (void)np; \
    return nin == 2 && tensor_valid(in[0]) && tensor_valid(in[1]) \
        && tensor_valid(o) && o->dtype == UINT8 \
        && in[0]->dtype == in[1]->dtype && tensors_shape_match(in[0], o); \
}

static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

#endif /* TENSOR_KERNEL_DISPATCH_H */
