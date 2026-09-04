#include "../internal.h"

/* Shared workspace for im2col buffer — avoids per-call malloc/free.
 * One buffer per thread; grows on demand, never shrinks during inference. */
static _Thread_local float* tl_col_buf = NULL;
static _Thread_local extent tl_col_buf_cap = 0;

static float* get_col_buf(extent need_bytes) {
    if (need_bytes > tl_col_buf_cap) {
        free(tl_col_buf);
        tl_col_buf = (float*)malloc(need_bytes);
        tl_col_buf_cap = tl_col_buf ? need_bytes : 0;
    }
    return tl_col_buf;
}

/* Winograd F(2,3): shared weights-transform U, tile-level M on thread stack. */
static tensor* winograd_conv2d_impl_ex(tensor* _out, const tensor** _ins,
                                       extent _nin, const void* _p, extent _np,
                                       int apply_silu) {
    const extent* p = (const extent*)_p;
    extent p_h = p[2], p_w = p[3];
    const tensor* x = _ins[0];
    const tensor* w = _ins[1];
    const tensor* b = (_nin > 2) ? _ins[2] : NULL;

    const extent* x_sh = (const extent*)x->shape;
    const extent* w_sh = (const extent*)w->shape;
    const extent* o_sh = (const extent*)_out->shape;

    extent C_in = x_sh[1], H = x_sh[2], W = x_sh[3];
    extent C_out = w_sh[0];
    extent out_H = o_sh[2], out_W = o_sh[3];
    extent tiles_h = (out_H + 1) / 2;
    extent tiles_w = (out_W + 1) / 2;

    DTYPE_SWITCH_ANY(x->dtype, _T, {
        const _T* x_data = (const _T*)x->data->ptr;
        const _T* w_data = (const _T*)w->data->ptr;
        const _T* b_data = b ? (const _T*)b->data->ptr : NULL;
        _T* o_data = (_T*)_out->data->ptr;
        _T* U = (_T*)malloc(C_in * C_out * 16 * sizeof(_T));
        if (!U) return &ERROR_TENSOR;

        _Pragma("omp parallel for collapse(2) schedule(static)")
        for (extent ci = 0; ci < C_in; ci++) {
            for (extent co = 0; co < C_out; co++) {
                const _T* g = w_data + (co * C_in + ci) * 9;
                _T g0 = g[0]; _T g1 = g[1]; _T g2 = g[2];
                _T g3 = g[3]; _T g4 = g[4]; _T g5 = g[5];
                _T g6 = g[6]; _T g7 = g[7]; _T g8 = g[8];
                _T R[4][3];
                R[0][0]=g0;               R[0][1]=g1;               R[0][2]=g2;
                R[1][0]=0.5f*(g0+g3+g6);  R[1][1]=0.5f*(g1+g4+g7);  R[1][2]=0.5f*(g2+g5+g8);
                R[2][0]=0.5f*(g0-g3+g6);  R[2][1]=0.5f*(g1-g4+g7);  R[2][2]=0.5f*(g2-g5+g8);
                R[3][0]=g6;               R[3][1]=g7;               R[3][2]=g8;
                _T* u = U + (ci * C_out + co) * 16;
                for (int i = 0; i < 4; i++) {
                    _T a = R[i][0]; _T b2 = R[i][1]; _T c2 = R[i][2];
                    u[i*4+0] = a;
                    u[i*4+1] = 0.5f*(a+b2+c2);
                    u[i*4+2] = 0.5f*(a-b2+c2);
                    u[i*4+3] = c2;
                }
            }
        }

        _Pragma("omp parallel")
        {
            _T* M = (_T*)malloc(C_out * 16 * sizeof(_T));
            if (M) {
                _Pragma("omp for collapse(2) schedule(static)")
                for (extent th = 0; th < tiles_h; th++) {
                    for (extent tw = 0; tw < tiles_w; tw++) {
                        memset(M, 0, C_out * 16 * sizeof(_T));
                        for (extent ci = 0; ci < C_in; ci++) {
                            _T d[4][4];
                            for (int i = 0; i < 4; i++) {
                                long long ih = (long long)th * 2 - (long long)p_h + i;
                                for (int j = 0; j < 4; j++) {
                                    long long iw = (long long)tw * 2 - (long long)p_w + j;
                                    d[i][j] = (ih >= 0 && ih < (long long)H &&
                                               iw >= 0 && iw < (long long)W)
                                              ? x_data[(ci * H + (extent)ih) * W + (extent)iw]
                                              : (_T)0;
                                }
                            }
                            _T dt[4][4];
                            for (int j = 0; j < 4; j++) {
                                dt[0][j] = d[0][j] - d[2][j];
                                dt[1][j] = d[1][j] + d[2][j];
                                dt[2][j] = -d[1][j] + d[2][j];
                                dt[3][j] = d[1][j] - d[3][j];
                            }
                            _T V[16];
                            for (int i = 0; i < 4; i++) {
                                _T a = dt[i][0]; _T b2 = dt[i][1];
                                _T c2 = dt[i][2]; _T d2 = dt[i][3];
                                V[i*4+0] = a  - c2;
                                V[i*4+1] = b2 + c2;
                                V[i*4+2] = -b2 + c2;
                                V[i*4+3] = b2 - d2;
                            }
                            const _T* u_base = U + ci * C_out * 16;
                            for (extent co = 0; co < C_out; co++) {
                                const _T* u = u_base + co * 16;
                                _T* m = M + co * 16;
                                for (int k = 0; k < 16; k++) m[k] += V[k] * u[k];
                            }
                        }

                        for (extent co = 0; co < C_out; co++) {
                            _T* m = M + co * 16;
                            _T mt0 = m[0]+m[4]+m[8];   _T mt1 = m[1]+m[5]+m[9];
                            _T mt2 = m[2]+m[6]+m[10];  _T mt3 = m[3]+m[7]+m[11];
                            _T mt4 = m[4]-m[8]-m[12];  _T mt5 = m[5]-m[9]-m[13];
                            _T mt6 = m[6]-m[10]-m[14]; _T mt7 = m[7]-m[11]-m[15];
                            _T Y[2][2];
                            Y[0][0] = mt0+mt1+mt2; Y[0][1] = mt1-mt2-mt3;
                            Y[1][0] = mt4+mt5+mt6; Y[1][1] = mt5-mt6-mt7;
                            _T bias = b_data ? b_data[co] : (_T)0;
                            for (int i = 0; i < 2; i++) {
                                extent oh = th * 2 + (extent)i;
                                if (oh >= out_H) continue;
                                for (int j = 0; j < 2; j++) {
                                    extent ow = tw * 2 + (extent)j;
                                    if (ow >= out_W) continue;
                                    _T val = Y[i][j] + bias;
                                    if (apply_silu) val = (_T)fast_silu((float)val);
                                    o_data[(co * out_H + oh) * out_W + ow] = val;
                                }
                            }
                        }
                    }
                }
                free(M);
            }
        }
        free(U);
    })
    return _out;
}

static tensor* conv2d_impl_ex(tensor* _out, const tensor** _ins,
                              extent _nin, const void* _p, extent _np,
                              int apply_silu) {
    if (_nin < 2 || _np < 4) return &ERROR_TENSOR;
    const extent* p = (const extent*)_p;
    extent s_h = p[0], s_w = p[1], p_h = p[2], p_w = p[3];

    const tensor* x = _ins[0];
    const tensor* w = _ins[1];
    const tensor* b = (_nin > 2) ? _ins[2] : NULL;

    const extent* x_sh = (const extent*)x->shape;
    const extent* w_sh = (const extent*)w->shape;
    const extent* o_sh = (const extent*)_out->shape;

    extent C_in = x_sh[1], H = x_sh[2], W = x_sh[3];
    extent C_out = w_sh[0], k_h = w_sh[2], k_w = w_sh[3];
    extent out_H = o_sh[2], out_W = o_sh[3];

    if (k_h == 3 && k_w == 3 && s_h == 1 && s_w == 1) {
        return winograd_conv2d_impl_ex(_out, _ins, _nin, _p, _np, apply_silu);
    }

    extent col_rows = C_in * k_h * k_w;
    extent col_cols = out_H * out_W;
    extent col_bytes = col_rows * col_cols * sizeof(float);

    if (x->dtype != REAL32) {
        DTYPE_SWITCH_ANY(x->dtype, _T, {
            _T* col_buf = (_T*)malloc(col_rows * col_cols * sizeof(_T));
            if (!col_buf) return &ERROR_TENSOR;
            const _T* x_data = (const _T*)x->data->ptr;
            const _T* w_data = (const _T*)w->data->ptr;
            const _T* b_data = b ? (const _T*)b->data->ptr : NULL;
            _T* o_data = (_T*)_out->data->ptr;
            _Pragma("omp parallel for collapse(2) schedule(static)")
            for (extent c = 0; c < C_in; c++) {
                for (extent kh = 0; kh < k_h; kh++) {
                    for (extent kw = 0; kw < k_w; kw++) {
                        extent out_row = c * k_h * k_w + kh * k_w + kw;
                        _T* row_ptr = col_buf + out_row * col_cols;
                        for (extent oh = 0; oh < out_H; oh++) {
                            extent ih = oh * s_h - p_h + kh;
                            for (extent ow = 0; ow < out_W; ow++) {
                                extent iw = ow * s_w - p_w + kw;
                                row_ptr[oh * out_W + ow] =
                                    (ih < H && iw < W) ? x_data[(c*H+ih)*W+iw] : (_T)0;
                            }
                        }
                    }
                }
            }
            _Pragma("omp parallel for schedule(static)")
            for (extent i = 0; i < C_out; i++) {
                _T bias = b_data ? b_data[i] : (_T)0;
                for (extent j = 0; j < col_cols; j++) o_data[i*col_cols+j] = bias;
                for (extent k = 0; k < col_rows; k++) {
                    _T w_ik = w_data[i * col_rows + k];
                    for (extent j = 0; j < col_cols; j++)
                        o_data[i*col_cols+j] += w_ik * col_buf[k*col_cols+j];
                }
                if (apply_silu)
                    for (extent j = 0; j < col_cols; j++)
                        o_data[i*col_cols+j] = (_T)fast_silu((float)o_data[i*col_cols+j]);
            }
            free(col_buf);
        })
        return _out;
    }

    {
        const float* x_data = (const float*)x->data->ptr;
        const float* w_data = (const float*)w->data->ptr;
        const float* b_data = b ? (const float*)b->data->ptr : NULL;
        float* o_data = (float*)_out->data->ptr;
        float* col_buf = (float*)get_col_buf(col_bytes);
        if (!col_buf) return &ERROR_TENSOR;

        _Pragma("omp parallel for collapse(2) schedule(static)")
        for (extent c = 0; c < C_in; c++) {
            for (extent kh = 0; kh < k_h; kh++) {
                for (extent kw = 0; kw < k_w; kw++) {
                    extent out_row = c * k_h * k_w + kh * k_w + kw;
                    float* row_ptr = col_buf + out_row * col_cols;
                    for (extent oh = 0; oh < out_H; oh++) {
                        extent ih = oh * s_h - p_h + kh;
                        if (ih >= H) {
                            memset(row_ptr + oh * out_W, 0, out_W * sizeof(float));
                            continue;
                        }
                        for (extent ow = 0; ow < out_W; ow++) {
                            extent iw = ow * s_w - p_w + kw;
                            row_ptr[oh * out_W + ow] =
                                (iw < W) ? x_data[(c * H + ih) * W + iw] : 0.0f;
                        }
                    }
                }
            }
        }

        _Pragma("omp parallel for schedule(static)")
        for (extent i = 0; i < C_out; i++) {
            float bias = b_data ? b_data[i] : 0.0f;
            float* out_row = o_data + i * col_cols;
            for (extent j = 0; j < col_cols; j++) out_row[j] = bias;
            for (extent k = 0; k < col_rows; k++) {
                float w_ik = w_data[i * col_rows + k];
                const float* col_row = col_buf + k * col_cols;
                for (extent j = 0; j < col_cols; j++) out_row[j] += w_ik * col_row[j];
            }
            if (apply_silu)
                for (extent j = 0; j < col_cols; j++)
                    out_row[j] = fast_silu(out_row[j]);
        }
    }
    return _out;
}

static tensor* conv2d_impl(tensor* _out, const tensor** _ins,
                           extent _nin, const void* _p, extent _np) {
    tensor* result = conv2d_impl_ex(_out, _ins, _nin, _p, _np, 0);
    return result;
}

static tensor* conv2d_silu_impl(tensor* _out, const tensor** _ins,
                                extent _nin, const void* _p, extent _np) {
    tensor* result = conv2d_impl_ex(_out, _ins, _nin, _p, _np, 1);
    return result;
}

boolean conv2d_lower(tensor *output, const tensor *const *inputs,
                     extent input_count, const void *parameters,
                     extent parameter_bytes) {
    return conv2d_impl(output, inputs, input_count, parameters,
                       parameter_bytes / sizeof(extent)) == &ERROR_TENSOR;
}

boolean conv2d_silu_lower(tensor *output, const tensor *const *inputs,
                          extent input_count, const void *parameters,
                          extent parameter_bytes) {
    return conv2d_silu_impl(output, inputs, input_count, parameters,
                            parameter_bytes / sizeof(extent)) == &ERROR_TENSOR;
}

