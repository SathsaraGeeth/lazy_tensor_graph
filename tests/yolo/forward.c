#include <stdbool.h>
#include "yolo.h"

static Node yolo_conv(Node x, YoloWeights* w, int idx, int s, int p) {
    return lazy_conv2d_silu(x, w[idx].weight, &w[idx].bias, s, s, p, p);
}

static Node yolo_c3(Node x, YoloWeights* w, int idx, int n, bool shortcut) {
    Node cv1 = yolo_conv(x, w, idx, 1, 0);
    
    int block_size = shortcut ? 5 : 4;
    Node cv2 = yolo_conv(x, w, idx + 2 + n * block_size, 1, 0);
    
    Node main_branch = cv1;
    for (int i = 0; i < n; i++) {
        int bt_idx = idx + 2 + i * block_size;
        Node bt_cv1 = yolo_conv(main_branch, w, bt_idx, 1, 0);
        Node bt_cv2 = yolo_conv(bt_cv1, w, bt_idx + 2, 1, 1);
        if (shortcut) {
            main_branch = lazy_add(bt_cv2, main_branch);
        } else {
            main_branch = bt_cv2;
        }
    }
    
    Node* ins[] = {&main_branch, &cv2};
    Node cat = lazy_concat(ins, 2, 1);
    
    int cv3_idx = idx + 5 + n * block_size;
    return yolo_conv(cat, w, cv3_idx, 1, 0);
}

static Node yolo_sppf(Node x, YoloWeights* w, int idx) {
    Node cv1 = yolo_conv(x, w, idx, 1, 0);
    Node m1 = lazy_maxpool2d(cv1, 5, 5, 1, 1, 2, 2, 1, 1);
    Node m2 = lazy_maxpool2d(m1, 5, 5, 1, 1, 2, 2, 1, 1);
    Node m3 = lazy_maxpool2d(m2, 5, 5, 1, 1, 2, 2, 1, 1);
    Node* ins[] = {&cv1, &m1, &m2, &m3};
    Node cat = lazy_concat(ins, 4, 1);
    return yolo_conv(cat, w, idx + 6, 1, 0);
}

void Yolov5n_forward(Node img_in, YoloWeights* weights, Node* detect_outputs) {
    // Backbone
    Node p1 = yolo_conv(img_in, weights, 0, 2, 2); // 0-P1/2
    Node p2 = yolo_conv(p1, weights, 2, 2, 1); // 1-P2/4
    Node c3_1 = yolo_c3(p2, weights, 4, 1, true); // 2
    Node p3 = yolo_conv(c3_1, weights, 16, 2, 1); // 3-P3/8
    Node c3_2 = yolo_c3(p3, weights, 18, 2, true); // 4
    Node p4 = yolo_conv(c3_2, weights, 35, 2, 1); // 5-P4/16
    Node c3_3 = yolo_c3(p4, weights, 37, 3, true); // 6
    Node p5 = yolo_conv(c3_3, weights, 59, 2, 1); // 7-P5/32
    Node c3_4 = yolo_c3(p5, weights, 61, 1, true); // 8
    Node sppf = yolo_sppf(c3_4, weights, 73); // 9

    // Head
    Node h1 = yolo_conv(sppf, weights, 81, 1, 0); // 10
    Node h1_up = lazy_upsample2d(h1, 2.0, 2.0); // 11
    Node* ins_h2[] = {&h1_up, &c3_3}; 
    Node h2_cat = lazy_concat(ins_h2, 2, 1); // 12
    Node h2_c3 = yolo_c3(h2_cat, weights, 85, 1, false); // 13

    Node h3 = yolo_conv(h2_c3, weights, 96, 1, 0); // 14
    Node h3_up = lazy_upsample2d(h3, 2.0, 2.0); // 15
    Node* ins_h4[] = {&h3_up, &c3_2}; 
    Node h4_cat = lazy_concat(ins_h4, 2, 1); // 16
    Node h4_c3 = yolo_c3(h4_cat, weights, 100, 1, false); // 17 (P3/8-small detection input)

    Node h5 = yolo_conv(h4_c3, weights, 111, 2, 1); // 18
    Node* ins_h6[] = {&h5, &h3}; 
    Node h6_cat = lazy_concat(ins_h6, 2, 1); // 19
    Node h6_c3 = yolo_c3(h6_cat, weights, 114, 1, false); // 20 (P4/16-medium detection input)

    Node h7 = yolo_conv(h6_c3, weights, 125, 2, 1); // 21
    Node* ins_h8[] = {&h7, &h1}; 
    Node h8_cat = lazy_concat(ins_h8, 2, 1); // 22
    Node h8_c3 = yolo_c3(h8_cat, weights, 128, 1, false); // 23 (P5/32-large detection input)

    // detect
    detect_outputs[2] = lazy_detect_head(h4_c3, weights[139].weight, &weights[139].bias, 1, 1, 0, 0);
    detect_outputs[1] = lazy_detect_head(h6_c3, weights[140].weight, &weights[140].bias, 1, 1, 0, 0);
    detect_outputs[0] = lazy_detect_head(h8_c3, weights[141].weight, &weights[141].bias, 1, 1, 0, 0);
}
