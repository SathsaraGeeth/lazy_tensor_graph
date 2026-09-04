#ifndef YOLO_H
#define YOLO_H

#include "tensor/tensor.h"
#include "tensor_ops.h"

typedef struct {
    extent rank;
    extent shape[4];
    float* data;
} Weight;

typedef struct {
    Weight weight;
    Weight bias;
} YoloWeights;

typedef struct {
    vtensor* vt;
    extent rank;
    extent shape[4];
} Node;

Node lazy_conv2d(Node x, Weight w, Weight* b, int s_h, int s_w, int p_h, int p_w);
Node lazy_conv2d_silu(Node x, Weight w, Weight* b, int s_h, int s_w, int p_h, int p_w);
Node lazy_silu(Node x);
Node lazy_maxpool2d(Node x, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, int d_h, int d_w);
Node lazy_upsample2d(Node x, float scale_h_f, float scale_w_f);
Node lazy_concat(Node** ins_arr, int nin, int dim);
Node lazy_add(Node a, Node b);
Node lazy_detect_head(Node x, Weight w, Weight* b, int s_h, int s_w, int p_h, int p_w);

void Yolov5n_forward(Node img_in, YoloWeights* weights, Node* detect_outputs);

#endif
