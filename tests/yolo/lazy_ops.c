#include <stdlib.h>
#include "yolo.h"

Node lazy_conv2d(Node x, Weight w, Weight* b, int s_h, int s_w, int p_h, int p_w) {
    int in_h = x.rank == 4 ? x.shape[2] : x.shape[1];
    int in_w = x.rank == 4 ? x.shape[3] : x.shape[2];
    
    int out_c = w.shape[0];
    int k_h = w.shape[2];
    int k_w = w.shape[3];
    
    int out_h = (in_h + 2 * p_h - k_h) / s_h + 1;
    int out_w = (in_w + 2 * p_w - k_w) / s_w + 1;
    
    extent w_sh[4] = {w.shape[0], w.shape[1], w.shape[2], w.shape[3]};
    vtensor* w_vt = tensor_lazy_alloc(4, w_sh, REAL32, 1, true, true, (dptr*)w.data);
    vtensor* b_vt = NULL;
    int num_ins = 2;
    if (b) {
        extent b_sh[4] = {0};
        for (int i = 0; i < b->rank; i++) b_sh[i] = b->shape[i];
        b_vt = tensor_lazy_alloc(b->rank, b_sh, REAL32, 1, true, true, (dptr*)b->data);
        num_ins = 3;
    }
    
    Node res = x;
    if (x.rank == 3) {
        res.shape[0] = out_c; res.shape[1] = out_h; res.shape[2] = out_w;
    } else {
        res.shape[1] = out_c; res.shape[2] = out_h; res.shape[3] = out_w;
    }
    
    extent res_sh[4] = {0};
    for (int i = 0; i < res.rank; i++) res_sh[i] = res.shape[i];
    vtensor* out_vt = tensor_lazy_alloc(res.rank, res_sh, REAL32, 100, false, false, NULL);
    const vtensor* ins[3] = { x.vt, w_vt, b_vt };
    extent params[] = {s_h, s_w, p_h, p_w};
    
    op_t operation = conv2d_ker;
    operation.input_count = num_ins;
    tensor_lazy_op_dispatch(operation, out_vt, ins, params);
    res.vt = out_vt;
    return res;
}

Node lazy_conv2d_silu(Node x, Weight w, Weight* b, int s_h, int s_w, int p_h, int p_w) {
    int in_h = x.rank == 4 ? x.shape[2] : x.shape[1];
    int in_w = x.rank == 4 ? x.shape[3] : x.shape[2];

    int out_c = w.shape[0];
    int k_h   = w.shape[2];
    int k_w   = w.shape[3];

    int out_h = (in_h + 2 * p_h - k_h) / s_h + 1;
    int out_w = (in_w + 2 * p_w - k_w) / s_w + 1;

    extent w_sh[4] = {w.shape[0], w.shape[1], w.shape[2], w.shape[3]};
    vtensor* w_vt = tensor_lazy_alloc(4, w_sh, REAL32, 1, true, true, (dptr*)w.data);
    vtensor* b_vt = NULL;
    int num_ins = 2;
    if (b) {
        extent b_sh[4] = {0};
        for (int i = 0; i < b->rank; i++) b_sh[i] = b->shape[i];
        b_vt = tensor_lazy_alloc(b->rank, b_sh, REAL32, 1, true, true, (dptr*)b->data);
        num_ins = 3;
    }

    Node res = x;
    if (x.rank == 3) {
        res.shape[0] = out_c; res.shape[1] = out_h; res.shape[2] = out_w;
    } else {
        res.shape[1] = out_c; res.shape[2] = out_h; res.shape[3] = out_w;
    }

    extent res_sh[4] = {0};
    for (int i = 0; i < res.rank; i++) res_sh[i] = res.shape[i];
    vtensor* out_vt = tensor_lazy_alloc(res.rank, res_sh, REAL32, 100, false, false, NULL);
    const vtensor* ins[3] = { x.vt, w_vt, b_vt };
    extent params[] = {s_h, s_w, p_h, p_w};

    op_t operation = conv2d_silu_ker;
    operation.input_count = num_ins;
    tensor_lazy_op_dispatch(operation, out_vt, ins, params);
    res.vt = out_vt;
    return res;
}


Node lazy_silu(Node x) {
    extent res_sh[4] = {0};
    for (int i = 0; i < x.rank; i++) res_sh[i] = x.shape[i];
    vtensor* out_vt = tensor_lazy_alloc(x.rank, res_sh, REAL32, 100, false, false, NULL);
    const vtensor* ins[] = { x.vt };
    tensor_lazy_op_dispatch(silu_ker, out_vt, ins, NULL);
    Node res = x;
    res.vt = out_vt;
    return res;
}

Node lazy_add(Node a, Node b) {
    extent res_sh[4] = {0};
    for (int i = 0; i < a.rank; i++) res_sh[i] = a.shape[i];
    vtensor* out_vt = tensor_lazy_alloc(a.rank, res_sh, REAL32, 100, false, false, NULL);
    const vtensor* ins[] = { a.vt, b.vt };
    tensor_lazy_op_dispatch(add_ker, out_vt, ins, NULL);
    Node res = a;
    res.vt = out_vt;
    return res;
}

Node lazy_concat(Node** ins_arr, int nin, int dim) {
    int actual_dim = ins_arr[0]->rank == 4 ? dim : (dim > 0 ? dim - 1 : 0);
    extent total_c = 0;
    for (int i = 0; i < nin; i++) {
        total_c += ins_arr[i]->shape[actual_dim];
    }
    Node res = *(ins_arr[0]);
    res.shape[actual_dim] = total_c;
    extent res_sh[4] = {0};
    for (int i = 0; i < res.rank; i++) res_sh[i] = res.shape[i];
    vtensor* out_vt = tensor_lazy_alloc(res.rank, res_sh, REAL32, 100, false, false, NULL);
    const vtensor** vt_ins = malloc(nin * sizeof(vtensor*));
    for (int i = 0; i < nin; i++) vt_ins[i] = ins_arr[i]->vt;
    
    extent params[] = {actual_dim};
    op_t operation = concat_ker;
    operation.input_count = nin;
    tensor_lazy_op_dispatch(operation, out_vt, vt_ins, params);
    free((void*)vt_ins);
    
    res.vt = out_vt;
    return res;
}

Node lazy_maxpool2d(Node x, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int d_h, int d_w) {
    int in_h = x.rank == 4 ? x.shape[2] : x.shape[1];
    int in_w = x.rank == 4 ? x.shape[3] : x.shape[2];
    
    int out_h = (in_h + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1;
    int out_w = (in_w + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;
    
    Node res = x;
    if (x.rank == 3) {
        res.shape[1] = out_h; res.shape[2] = out_w;
    } else {
        res.shape[2] = out_h; res.shape[3] = out_w;
    }
    
    extent res_sh[4] = {0};
    for (int i = 0; i < res.rank; i++) res_sh[i] = res.shape[i];
    vtensor* out_vt = tensor_lazy_alloc(res.rank, res_sh, REAL32, 100, false, false, NULL);
    const vtensor* ins[] = { x.vt };
    extent params[] = {k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w, 0};
    tensor_lazy_op_dispatch(max_pool2d_ker, out_vt, ins, params);
    
    res.vt = out_vt;
    return res;
}

Node lazy_upsample2d(Node x, float scale_h_f, float scale_w_f) {
    int out_h = (x.rank == 4 ? x.shape[2] : x.shape[1]) * scale_h_f;
    int out_w = (x.rank == 4 ? x.shape[3] : x.shape[2]) * scale_w_f;
    Node res = x;
    if (x.rank == 3) {
        res.shape[1] = out_h; res.shape[2] = out_w;
    } else {
        res.shape[2] = out_h; res.shape[3] = out_w;
    }
    
    extent res_sh[4] = {0};
    for (int i = 0; i < res.rank; i++) res_sh[i] = res.shape[i];
    vtensor* out_vt = tensor_lazy_alloc(res.rank, res_sh, REAL32, 100, false, false, NULL);
    const vtensor* ins[] = { x.vt };
    extent params[] = {(extent)scale_h_f, (extent)scale_w_f, 0};
    tensor_lazy_op_dispatch(upsample2d_ker, out_vt, ins, params);
    
    res.vt = out_vt;
    return res;
}

Node lazy_detect_head(Node x, Weight w, Weight* b, int s_h, int s_w, int p_h, int p_w) {
    return lazy_conv2d(x, w, b, s_h, s_w, p_h, p_w);
}
