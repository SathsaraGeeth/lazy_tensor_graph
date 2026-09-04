#include "../internal.h"

static tensor* max_pool2d_impl(tensor* _out, const tensor** _ins,
                               extent _nin, const void* _p, extent _np) {
    if (_nin != 1 || _np < 9) return &ERROR_TENSOR;
    const extent* p = (const extent*)_p;
    extent k_h = p[0], k_w = p[1], s_h = p[2], s_w = p[3];
    extent p_h = p[4], p_w = p[5], d_h = p[6], d_w = p[7];
    (void)p[8];
    const extent* in_sh = (const extent*)_ins[0]->shape;
    const extent* out_sh = (const extent*)_out->shape;
    extent C = in_sh[1], H = in_sh[2], W = in_sh[3];
    extent out_H = out_sh[2], out_W = out_sh[3];
    DTYPE_SWITCH_REAL(_ins[0]->dtype, _T, {
        const _T* in_data = (const _T*)_ins[0]->data->ptr;
        _T* out_data = (_T*)_out->data->ptr;
        _T min_val = -INFINITY;
        _Pragma("omp parallel for schedule(static)")
        for (extent c = 0; c < C; c++) {
            for (extent oh = 0; oh < out_H; oh++) {
                for (extent ow = 0; ow < out_W; ow++) {
                    _T max_v = min_val;
                    for (extent kh = 0; kh < k_h; kh++) {
                        extent ih = oh * s_h - p_h + kh * d_h;
                        if (ih >= H) continue;
                        for (extent kw = 0; kw < k_w; kw++) {
                            extent iw = ow * s_w - p_w + kw * d_w;
                            if (iw >= W) continue;
                            _T val = in_data[(c * H + ih) * W + iw];
                            if (val > max_v) max_v = val;
                        }
                    }
                    out_data[(c * out_H + oh) * out_W + ow] = max_v;
                }
            }
        }
    })
    return _out;
}

boolean max_pool2d_lower(tensor *output, const tensor *const *inputs,
                         extent input_count, const void *parameters,
                         extent parameter_bytes) {
    return max_pool2d_impl(output, inputs, input_count, parameters,
                       parameter_bytes / sizeof(extent)) == &ERROR_TENSOR;
}

