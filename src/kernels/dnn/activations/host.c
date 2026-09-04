#include "../internal.h"

static tensor* silu_impl(tensor* _out, const tensor** _ins,
                         extent _nin, const void* _p, extent _np) {
    (void)_p; (void)_np;
    if (_nin != 1) return &ERROR_TENSOR;
    extent sz = _ins[0]->size;
    DTYPE_SWITCH_REAL(_ins[0]->dtype, _T, {
        const _T* in_data = (const _T*)_ins[0]->data->ptr;
        _T* out_data = (_T*)_out->data->ptr;
        _Pragma("omp parallel for schedule(static)")
        for (extent i = 0; i < sz; i++) out_data[i] = (_T)fast_silu((float)in_data[i]);
    })
    return _out;
}

boolean silu_lower(tensor *output, const tensor *const *inputs,
                         extent input_count, const void *parameters,
                         extent parameter_bytes) {
    return silu_impl(output, inputs, input_count, parameters,
                       parameter_bytes / sizeof(extent)) == &ERROR_TENSOR;
}

