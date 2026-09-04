#include "../internal.h"

static tensor* upsample2d_impl(tensor* _out, const tensor** _ins,
                               extent _nin, const void* _p, extent _np) {
    if (_nin != 1 || _np < 3) return &ERROR_TENSOR;
    const extent* p = (const extent*)_p;
    extent scale_h = p[0], scale_w = p[1], mode = p[2];
    if (mode != 0) return &ERROR_TENSOR;
    const extent* in_sh = (const extent*)_ins[0]->shape;
    const extent* out_sh = (const extent*)_out->shape;
    extent C = in_sh[1], H = in_sh[2], W = in_sh[3];
    extent out_H = out_sh[2], out_W = out_sh[3];
    DTYPE_SWITCH_ANY(_ins[0]->dtype, _T, {
        const _T* in_data = (const _T*)_ins[0]->data->ptr;
        _T* out_data = (_T*)_out->data->ptr;
        _Pragma("omp parallel for schedule(static)")
        for (extent c = 0; c < C; c++) {
            for (extent oh = 0; oh < out_H; oh++) {
                extent ih = oh / scale_h; if (ih >= H) ih = H - 1;
                for (extent ow = 0; ow < out_W; ow++) {
                    extent iw = ow / scale_w; if (iw >= W) iw = W - 1;
                    out_data[(c * out_H + oh) * out_W + ow] = in_data[(c * H + ih) * W + iw];
                }
            }
        }
    })
    return _out;
}

boolean upsample2d_lower(tensor *output, const tensor *const *inputs,
                         extent input_count, const void *parameters,
                         extent parameter_bytes) {
    return upsample2d_impl(output, inputs, input_count, parameters,
                       parameter_bytes / sizeof(extent)) == &ERROR_TENSOR;
}

