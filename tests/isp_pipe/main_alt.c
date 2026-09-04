#include "util/util.h"

#define WIDTH       4208
#define HEIGHT      3120
#define WHITE_LEVEL 1023
#define RAW_INPUT   "/mnt/fileserver/prj/dmctp/user/src/isp_pipe/data/payload_N000.raw"
#define OUTPUT_DIR  "/mnt/fileserver/prj/dmctp/user/src/isp_pipe/processed"

int main(void) {
    real32 camera_values[] = {
        0.7228f, -0.0893f, -0.0975f,
       -0.4792f,  1.3481f,  0.1381f,
       -0.1137f,  0.2680f,  0.5604f
    };
    real32 srgb_values[] = {
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f
    };
    extent matrix_shape[] = {3, 3};
    extent raw_shape[]    = {HEIGHT, WIDTH};

    uint16 *raw = isp_load_raw10(RAW_INPUT, WIDTH, HEIGHT);
    if (!raw) {
        mem_pool_shutdown();
        return 1;
    }

    vtensor *camera       = tensor_lazy_alloc(2, matrix_shape, REAL32, 0, true, false, (dptr *)camera_values);
    vtensor *srgb         = tensor_lazy_alloc(2, matrix_shape, REAL32, 0, true, false, (dptr *)srgb_values);
    vtensor *ccm          = isp_normalized_ccm(tensor_matrix_mul(srgb, tensor_matrix_inv_square(camera)));
    vtensor *input        = tensor_lazy_alloc(2, raw_shape, UINT16, 0, true, true, (dptr *)raw);
    vtensor *black        = tensor_blc(input, 64.0f, 64.0f, 64.0f, 64.0f, 1, WHITE_LEVEL);
    vtensor *corrected    = tensor_dpc(black, 30);
    vtensor *balanced     = tensor_awb(corrected, 1.0f / 0.4862299963f, 1.0f,1.0f / 0.7241870004f, 1, WHITE_LEVEL);
    vtensor *rgb          = tensor_demosaic(balanced, 1);
    vtensor *gamma        = isp_gamma(tensor_ccm(rgb, ccm, WHITE_LEVEL), WHITE_LEVEL, 1.0f / 2.2f);
    vtensor *blurred      = tensor_blur(gamma);
    vtensor *yuv          = tensor_blend_rgb2yuv(gamma, blurred, 0.8f, 0.2f, UINT16_MAX);
    vtensor *nv12         = tensor_chroma_subsample(yuv);
    const uint16 *preview = (const uint16 *)tensor_lazy_view_to(gamma);
    const void *nv12_data = tensor_lazy_view_to(nv12);


    int failed = !preview || !nv12_data;
    if (!failed) {
        isp_save_rgb_png(
            OUTPUT_DIR "/frame_alt_preview.png", preview, WIDTH, HEIGHT, 4);
        isp_save_raw(
            OUTPUT_DIR "/frame_alt_nv12.raw", nv12_data, (extent)WIDTH * HEIGHT * 3 / 2 * sizeof(uint16));
    }

    mem_pool_shutdown();
    free(raw);
    return failed;
}
