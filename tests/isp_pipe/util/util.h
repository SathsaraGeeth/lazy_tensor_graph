#ifndef ISP_PIPE_UTIL_H
#define ISP_PIPE_UTIL_H

#include "tensor/tensor.h"
#include "tensor_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static uint16 isp_reverse_10_bits(uint32 value) {
    uint16 reversed = 0;
    for (extent bit = 0; bit < 10; ++bit)
        reversed = (uint16)((reversed << 1) | ((value >> bit) & 1u));
    return reversed;
}

static uint16 *isp_load_raw10(const char *path, extent width, extent height) {
    extent pixels = width * height;
    extent packed_bytes = pixels * 10 / 8;
    uint8 *packed = malloc(packed_bytes);
    uint16 *raw = malloc(pixels * sizeof(*raw));
    FILE *file = fopen(path, "rb");
    extent bytes_read = fread(packed, 1, packed_bytes, file);
    (void)bytes_read;
    fclose(file);

    for (extent group = 0; group < pixels / 4; ++group) {
        const uint8 *source = packed + group * 5;
        uint64 bits = (uint64)source[0] |
                      (uint64)source[1] << 8 |
                      (uint64)source[2] << 16 |
                      (uint64)source[3] << 24 |
                      (uint64)source[4] << 32;
        for (extent pixel = 0; pixel < 4; ++pixel)
            raw[group * 4 + pixel] =
                isp_reverse_10_bits((uint32)(bits >> (pixel * 10)) & 0x3ffu);
    }
    free(packed);
    return raw;
}

static vtensor *isp_gamma(const vtensor *input, uint32 white_level,
                          real32 gamma) {
    const extent entries = (extent)UINT16_MAX + 1;
    real32 *values = malloc(entries * sizeof(*values));
    for (extent index = 0; index < entries; ++index) {
        real32 normalized = (real32)index / white_level;
        if (normalized > 1.0f) normalized = 1.0f;
        values[index] = powf(normalized, gamma);
    }
    extent shape[] = {entries};
    vtensor *table = tensor_lazy_alloc(1, shape, REAL32, 0,
                                        true, false, (dptr *)values);
    vtensor *output = tensor_lut(input, table);
    free(values);
    return output;
}

static vtensor *isp_normalized_ccm(vtensor *matrix) {
    const real32 *source = (const real32 *)tensor_lazy_view_to(matrix);
    real32 normalized[9];
    for (extent row = 0; row < 3; ++row) {
        real32 sum = source[row * 3] + source[row * 3 + 1] +
                     source[row * 3 + 2];
        for (extent column = 0; column < 3; ++column)
            normalized[row * 3 + column] =
                source[row * 3 + column] / sum;
    }
    extent shape[] = {3, 3};
    return tensor_lazy_alloc(2, shape, REAL32, 0,
                             true, false, (dptr *)normalized);
}

static void isp_save_rgb_png(const char *path, const uint16 *rgb,
                             extent width, extent height, extent scale) {
    extent output_width = width / scale;
    extent output_height = height / scale;
    uint8 *image = malloc(output_width * output_height * 3);
    for (extent y = 0; y < height; y += scale)
        for (extent x = 0; x < width; x += scale) {
            const uint16 *pixel = rgb + (y * width + x) * 3;
            extent output =
                ((y / scale) * output_width + x / scale) * 3;
            image[output] = (uint8)(pixel[0] >> 8);
            image[output + 1] = (uint8)(pixel[1] >> 8);
            image[output + 2] = (uint8)(pixel[2] >> 8);
        }
    stbi_write_png(path, (int)output_width, (int)output_height, 3,
                   image, (int)(output_width * 3));
    free(image);
}

static void isp_save_raw(const char *path, const void *data, extent bytes) {
    FILE *file = fopen(path, "wb");
    fwrite(data, 1, bytes, file);
    fclose(file);
}

#endif
