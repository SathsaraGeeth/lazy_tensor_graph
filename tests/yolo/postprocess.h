#ifndef YOLO_POSTPROCESS_H
#define YOLO_POSTPROCESS_H

#include <stdint.h>
#include "yolo.h" // For Node struct

typedef struct {
    int class_id;
    float score;
    float x1, y1, x2, y2;
} BBox;

typedef struct {
    BBox* boxes;
    int count;
    int capacity;
} BBoxList;

BBoxList extract_bboxes(Node* detect_outputs, float conf_threshold);
BBoxList non_max_suppression(BBoxList input_boxes, float iou_threshold, int max_detections);
void scale_bboxes(BBoxList* list, int orig_w, int orig_h, int target_size);
void draw_bboxes(uint8_t* img_data, int width, int height, int channels, BBoxList boxes);

#endif
