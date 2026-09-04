#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "tensor/tensor.h"

#define STB_IMAGE_IMPLEMENTATION
#include "util/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/stb_image_write.h"

#include "yolo.h"
#include "postprocess.h"

YoloWeights* load_weights(const char* path, uint32_t* num_weights_out) {
    FILE* wf = fopen(path, "rb");
    if (!wf) {
        printf("Failed to open weights file.\n");
        return NULL;
    }
    
    uint32_t num_weights;
    fread(&num_weights, sizeof(uint32_t), 1, wf);
    if (num_weights_out) *num_weights_out = num_weights;
    YoloWeights* weights = calloc(num_weights, sizeof(YoloWeights));
    
    for (uint32_t i = 0; i < num_weights; i++) {
        uint32_t has_w;
        fread(&has_w, sizeof(uint32_t), 1, wf);
        if (has_w) {
            fread(&weights[i].weight.rank, sizeof(uint32_t), 1, wf);
            for (int r = 0; r < weights[i].weight.rank; r++) {
                fread(&weights[i].weight.shape[r], sizeof(uint32_t), 1, wf);
            }
            uint32_t num_el;
            fread(&num_el, sizeof(uint32_t), 1, wf);
            weights[i].weight.data = malloc(num_el * sizeof(float));
            fread(weights[i].weight.data, sizeof(float), num_el, wf);
        }
        
        uint32_t has_b;
        fread(&has_b, sizeof(uint32_t), 1, wf);
        if (has_b) {
            fread(&weights[i].bias.rank, sizeof(uint32_t), 1, wf);
            for (int r = 0; r < weights[i].bias.rank; r++) {
                fread(&weights[i].bias.shape[r], sizeof(uint32_t), 1, wf);
            }
            uint32_t num_el;
            fread(&num_el, sizeof(uint32_t), 1, wf);
            weights[i].bias.data = malloc(num_el * sizeof(float));
            fread(weights[i].bias.data, sizeof(float), num_el, wf);
        }
    }
    fclose(wf);
    return weights;
}

void free_weights(YoloWeights* weights, uint32_t num_weights) {
    if (!weights) return;
    for (uint32_t i = 0; i < num_weights; i++) {
        if (weights[i].weight.data) free(weights[i].weight.data);
        if (weights[i].bias.data) free(weights[i].bias.data);
    }
    free(weights);
}

int main() {

    int img_w, img_h, img_c;
    uint8_t* img_data = stbi_load("/mnt/fileserver/prj/dmctp/user/src/yolo/imgs/test.jpg", &img_w, &img_h, &img_c, 3);
    if (!img_data) {
        printf("Failed to load image.\n");
        return 1;
    }

    extent in_sh[] = {img_h, img_w, 3};
    vtensor* img_vt = tensor_lazy_alloc(3, in_sh, UINT8, 1, true, true, (dptr*)img_data);
    
    extent out_sh[] = {3, 640, 640};
    vtensor* pp_vt = tensor_lazy_alloc(3, out_sh, REAL32, 1, false, false, NULL);
    
    const vtensor* pp_ins[] = { img_vt };
    tensor_lazy_op_dispatch(resize_pad_norm_ker, pp_vt, pp_ins, NULL);

    float* pp_data = (float*)tensor_lazy_view_to(pp_vt);
    
    if (pp_data) {
        uint32_t num_weights = 0;
        YoloWeights* weights = load_weights("/mnt/fileserver/prj/dmctp/user/src/yolo/model/yolo_weights.bin", &num_weights);
        
        if (weights) {
            extent yolo_in_sh[] = {1, 3, 640, 640};
            vtensor* yolo_in_vt = tensor_lazy_alloc(4, yolo_in_sh, REAL32, 100, true, true, (dptr*)pp_data);
            Node img_in = { yolo_in_vt, 4, {1, 3, 640, 640} };
            Node detect_outputs[3] = {0};


            Yolov5n_forward(img_in, weights, detect_outputs);
            BBoxList all_boxes = extract_bboxes(detect_outputs, 0.25f);


            BBoxList nms_boxes = non_max_suppression(all_boxes, 0.45f, 300);

            scale_bboxes(&nms_boxes, img_w, img_h, 640);
            
            draw_bboxes(img_data, img_w, img_h, img_c, nms_boxes);
            stbi_write_jpg("/mnt/fileserver/prj/dmctp/user/src/yolo/imgs/output_detected.jpg", img_w, img_h, img_c, img_data, 100);
	            
            if (all_boxes.boxes) free(all_boxes.boxes);
            if (nms_boxes.boxes) free(nms_boxes.boxes);

            free_weights(weights, num_weights);
        } else {
            printf("Failed to load weights.\n");
        }
    } else {
        printf("Preprocessing failed.\n");
    }

    stbi_image_free(img_data);

    
    mem_pool_shutdown();
    return 0;
}
