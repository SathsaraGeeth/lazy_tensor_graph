#include "../internal.h"

static tensor* concat_impl(tensor* _out, const tensor** _ins,
                           extent _nin, const void* _p, extent _np) {
    if (_nin < 2 || _np < 1) return &ERROR_TENSOR;
    const extent* p = (const extent*)_p;
    extent dim = p[0];
    if (dim != 1) return &ERROR_TENSOR;
    extent out_H = ((const extent*)_out->shape)[2];
    extent out_W = ((const extent*)_out->shape)[3];
    extent spatial_size = out_H * out_W;
    DTYPE_SWITCH_ANY(_ins[0]->dtype, _T, {
        _T* out_data = (_T*)_out->data->ptr;
        extent out_offset = 0;
        for (extent n = 0; n < _nin; n++) {
            const _T* in_data = (const _T*)_ins[n]->data->ptr;
            extent C = ((const extent*)_ins[n]->shape)[1];
            for (extent i = 0; i < C * spatial_size; i++) out_data[out_offset + i] = in_data[i];
            out_offset += C * spatial_size;
        }
    })
    return _out;
}

boolean concat_lower(tensor *output, const tensor *const *inputs,
                         extent input_count, const void *parameters,
                         extent parameter_bytes) {
    return concat_impl(output, inputs, input_count, parameters,
                       parameter_bytes / sizeof(extent)) == &ERROR_TENSOR;
}

