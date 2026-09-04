#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "postprocess.h"
#include "util/font.h"

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void add_bbox(BBoxList* list, BBox b) {
    if (list->count >= list->capacity) {
        list->capacity = list->capacity == 0 ? 256 : list->capacity * 2;
        list->boxes = realloc(list->boxes, list->capacity * sizeof(BBox));
    }
    list->boxes[list->count++] = b;
}

BBoxList extract_bboxes(Node* detect_outputs, float conf_threshold) {
    BBoxList list = {NULL, 0, 0};

    int strides[3] = {32, 16, 8};
    float anchors[3][3][2] = {
        {{116, 90}, {156, 198}, {373, 326}}, // P5/32 (Output 0)
        {{30, 61}, {62, 45}, {59, 119}},     // P4/16 (Output 1)
        {{10, 13}, {16, 30}, {33, 23}}       // P3/8  (Output 2)
    };

    for (int i = 0; i < 3; i++) {
        Node out = detect_outputs[i];
        float* data = (float*)tensor_lazy_view_to(out.vt);
        if (!data) continue;

        int ny = out.shape[2]; // Height
        int nx = out.shape[3]; // Width
        
        int stride = strides[i];
        int num_anchors = 3;
        int num_classes = 80;
        int num_attrs = 5 + num_classes; // 85

        for (int a = 0; a < num_anchors; a++) {
            float aw = anchors[i][a][0];
            float ah = anchors[i][a][1];

            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    // data layout: [255][ny][nx] -> channel = a * 85 + attr
                    // Offset calculation: Index = c * ny * nx + y * nx + x
                    
                    int obj_c = a * num_attrs + 4;
                    float raw_obj = data[obj_c * ny * nx + y * nx + x];
                    float obj = sigmoid(raw_obj);
                    
                    if (obj < conf_threshold) continue;

                    // Find max class
                    float max_cls = 0.0f;
                    int best_class = 0;
                    for (int c = 0; c < num_classes; c++) {
                        int cls_c = a * num_attrs + 5 + c;
                        float cls_val = sigmoid(data[cls_c * ny * nx + y * nx + x]);
                        if (cls_val > max_cls) {
                            max_cls = cls_val;
                            best_class = c;
                        }
                    }

                    float score = obj * max_cls;
                    if (score < conf_threshold) continue;

                    // Decode coordinates
                    int tx_c = a * num_attrs + 0;
                    int ty_c = a * num_attrs + 1;
                    int tw_c = a * num_attrs + 2;
                    int th_c = a * num_attrs + 3;

                    float bx = (2.0f * sigmoid(data[tx_c * ny * nx + y * nx + x]) - 0.5f + x) * (float)stride;
                    float by = (2.0f * sigmoid(data[ty_c * ny * nx + y * nx + x]) - 0.5f + y) * (float)stride;
                    
                    float bw_val = 2.0f * sigmoid(data[tw_c * ny * nx + y * nx + x]);
                    float bw = bw_val * bw_val * aw;
                    
                    float bh_val = 2.0f * sigmoid(data[th_c * ny * nx + y * nx + x]);
                    float bh = bh_val * bh_val * ah;

                    BBox box;
                    box.class_id = best_class;
                    box.score = score;
                    box.x1 = bx - bw / 2.0f;
                    box.y1 = by - bh / 2.0f;
                    box.x2 = bx + bw / 2.0f;
                    box.y2 = by + bh / 2.0f;

                    add_bbox(&list, box);
                }
            }
        }
    }
    return list;
}

static int compare_bboxes(const void* a, const void* b) {
    float sA = ((BBox*)a)->score;
    float sB = ((BBox*)b)->score;
    if (sA < sB) return 1;
    if (sA > sB) return -1;
    return 0;
}

static float iou(BBox a, BBox b) {
    float x1 = fmaxf(a.x1, b.x1);
    float y1 = fmaxf(a.y1, b.y1);
    float x2 = fminf(a.x2, b.x2);
    float y2 = fminf(a.y2, b.y2);

    float w = fmaxf(0.0f, x2 - x1);
    float h = fmaxf(0.0f, y2 - y1);
    float inter = w * h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    float un = area_a + area_b - inter;
    if (un <= 0.000001f) return 0.0f;
    return inter / un;
}

BBoxList non_max_suppression(BBoxList input, float iou_threshold, int max_detections) {
    BBoxList result = {NULL, 0, 0};
    if (input.count == 0) return result;

    qsort(input.boxes, input.count, sizeof(BBox), compare_bboxes);

    int* removed = calloc(input.count, sizeof(int));
    
    for (int i = 0; i < input.count; i++) {
        if (removed[i]) continue;

        add_bbox(&result, input.boxes[i]);
        if (result.count >= max_detections) break;

        for (int j = i + 1; j < input.count; j++) {
            if (removed[j]) continue;
            // Per-class NMS
            if (input.boxes[i].class_id == input.boxes[j].class_id) {
                if (iou(input.boxes[i], input.boxes[j]) > iou_threshold) {
                    removed[j] = 1;
                }
            }
        }
    }
    
    free(removed);
    return result;
}

static uint8_t colors[20][3] = {
    {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255},
    {0, 255, 255}, {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0},
    {128, 0, 128}, {0, 128, 128}, {64, 0, 0}, {0, 64, 0}, {0, 0, 64},
    {64, 64, 0}, {64, 0, 64}, {0, 64, 64}, {192, 0, 0}, {0, 192, 0}
};

static void draw_rect(uint8_t* img, int w, int h, int c, int x1, int y1, int x2, int y2, uint8_t* color) {
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 >= w) x2 = w - 1;
    if (y2 >= h) y2 = h - 1;

    int thickness = 2;

    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            if (y < y1 + thickness || y > y2 - thickness || x < x1 + thickness || x > x2 - thickness) {
                int idx = (y * w + x) * c;
                img[idx] = color[0];
                img[idx + 1] = color[1];
                img[idx + 2] = color[2];
            }
        }
    }
}

void scale_bboxes(BBoxList* list, int orig_w, int orig_h, int target_size) {
    float r = fminf((float)target_size / orig_h, (float)target_size / orig_w);
    float unpad_w = orig_w * r;
    float unpad_h = orig_h * r;
    float dw = (target_size - unpad_w) / 2.0f;
    float dh = (target_size - unpad_h) / 2.0f;

    for (int i = 0; i < list->count; i++) {
        list->boxes[i].x1 = (list->boxes[i].x1 - dw) / r;
        list->boxes[i].x2 = (list->boxes[i].x2 - dw) / r;
        list->boxes[i].y1 = (list->boxes[i].y1 - dh) / r;
        list->boxes[i].y2 = (list->boxes[i].y2 - dh) / r;
    }
}

static void draw_text(uint8_t* img, int w, int h, int c, int x, int y, const char* str, uint8_t* color, int scale) {
    int cur_x = x;
    while (*str) {
        char ch = *str;
        if (ch >= 32 && ch < 127) {
            const unsigned char* glyph = font5x7 + (ch - 32) * 5;
            for (int cx = 0; cx < 5; cx++) {
                for (int cy = 0; cy < 7; cy++) {
                    if ((glyph[cx] >> cy) & 1) {
                        for (int sx = 0; sx < scale; sx++) {
                            for (int sy = 0; sy < scale; sy++) {
                                int px = cur_x + cx * scale + sx;
                                int py = y + cy * scale + sy;
                                if (px >= 0 && px < w && py >= 0 && py < h) {
                                    int idx = (py * w + px) * c;
                                    img[idx] = color[0];
                                    img[idx + 1] = color[1];
                                    img[idx + 2] = color[2];
                                }
                            }
                        }
                    }
                }
            }
        }
        cur_x += 6 * scale;
        str++;
    }
}

void draw_bboxes(uint8_t* img_data, int width, int height, int channels, BBoxList boxes) {
    static const char* COCO_CLASSES[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    for (int i = 0; i < boxes.count; i++) {
        BBox b = boxes.boxes[i];
        int color_idx = b.class_id % 20;
        draw_rect(img_data, width, height, channels, (int)b.x1, (int)b.y1, (int)b.x2, (int)b.y2, colors[color_idx]);
        const char* label = (b.class_id >= 0 && b.class_id < 80) ? COCO_CLASSES[b.class_id] : "unknown";
        printf("Detected %s (class %d) with score %.2f at [%.1f, %.1f, %.1f, %.1f]\n", 
            label, b.class_id, b.score, b.x1, b.y1, b.x2, b.y2);
            
        // Draw the text slightly above the bounding box (scale=2)
        int text_y = (int)b.y1 - 16;
        if (text_y < 0) text_y = 0;
        draw_text(img_data, width, height, channels, (int)b.x1, text_y, label, colors[color_idx], 2);
    }
}
