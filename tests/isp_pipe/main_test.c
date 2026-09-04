#define STB_IMAGE_IMPLEMENTATION
#include "util/stb_image.h"

#include "util/util.h"

#ifndef INPUT_PATH
#define INPUT_PATH      "tests/isp_pipe/data/0005.png"
#endif
#ifndef CROP_PATH
#define CROP_PATH       "tests/isp_pipe/processed/test_crop.png"
#endif
#ifndef CHANNEL_PATH
#define CHANNEL_PATH    "tests/isp_pipe/processed/test_green.png"
#endif

int main(void) {
    int width    = 0;
    int height   = 0;
    int channels = 0;

    extent input_shape[3];

    uint8 *pixels = stbi_load(INPUT_PATH, &width, &height, &channels, 3);

    if (!pixels || width <= 0 || height <= 0) {
        fprintf(stderr, "cannot load %s\n", INPUT_PATH);
        stbi_image_free(pixels);
        return 1;
    }

    input_shape[0] = (extent)height;
    input_shape[1] = (extent)width;
    input_shape[2] = 3;

    
    vtensor *input       = tensor_lazy_alloc(3, input_shape, UINT8, 0, true, true, (dptr *)pixels);
    vtensor *bright      = tensor_brightness(input, 0.10f);
    vtensor *gamma       = tensor_gamma_correction(bright, 0.85f, 256);
    extent crop_start[]  = {80, 120, 0};
    extent crop_end[]    = {400, 520, 3};
    vtensor *crop        = tensor_slice(gamma, 3, crop_start, crop_end);
    extent permutation[] = {1, 0, 2};
    vtensor *transposed  = tensor_permute(crop, 3, permutation);
    extent green_start[] = {0, 0, 1};
    extent green_end[]   = {400, 320, 2};
    vtensor *green_3d    = tensor_slice(transposed, 3, green_start, green_end);
    extent green_shape[] = {400, 320};
    vtensor *green       = tensor_reshape(green_3d, 2, green_shape);
    vtensor *green_u16   = tensor_cast(green, UINT16);
    vtensor *green_u8    = tensor_cast(green_u16, UINT8);

    const uint8 *crop_pixels = (const uint8 *)tensor_lazy_view_to(crop);
    const uint8 *green_pixels = (const uint8 *)tensor_lazy_view_to(green_u8);


    int failed = !input || !bright || !gamma || !crop || !transposed ||
                 !green_3d || !green || !green_u16 || !green_u8 ||
                 !crop_pixels || !green_pixels;
    if (!failed) {
        failed |= !stbi_write_png(CROP_PATH, 400, 320, 3, crop_pixels, 400 * 3);
        failed |= !stbi_write_png(CHANNEL_PATH, 320, 400, 1, green_pixels, 320);
    }

    mem_pool_shutdown();
    stbi_image_free(pixels);
    if (failed)
        fprintf(stderr, "image operation test failed\n");
    else
        printf("wrote %s and %s\n", CROP_PATH, CHANNEL_PATH);
    return failed;
}
