#include "../internal.h"

boolean resize_pad_norm_lower(tensor *output, const tensor *const *inputs,
                              extent input_count, const void *parameters,
                              extent parameter_bytes) {
    (void)parameters;
    if (!output || !inputs || input_count != 1 || parameter_bytes ||
        !inputs[0] || !inputs[0]->data || !inputs[0]->data->ptr ||
        !output->data || !output->data->ptr ||
        inputs[0]->dtype != UINT8 || output->dtype != REAL32 ||
        inputs[0]->rank != 3 || output->rank != 3)
        return true;

    const extent *input_shape = inputs[0]->shape;
    const extent *output_shape = output->shape;
    extent height = input_shape[0], width = input_shape[1];
    extent channels = input_shape[2];
    extent output_channels = output_shape[0];
    extent target_height = output_shape[1];
    extent target_width = output_shape[2];
    if (channels != 3 || output_channels != 3 ||
        target_height != target_width)
        return true;

    const uint8 *input = (const uint8 *)inputs[0]->data->ptr;
    real32 *result = (real32 *)output->data->ptr;
    extent target = target_height;
    real32 scale = (real32)target / (real32)height;
    real32 width_scale = (real32)target / (real32)width;
    if (width_scale < scale) scale = width_scale;

    extent resized_width = (extent)((real32)width * scale);
    extent resized_height = (extent)((real32)height * scale);
    extent pad_x = (target - resized_width) / 2;
    extent pad_y = (target - resized_height) / 2;
    real32 padding = 114.0f / 255.0f;

    #pragma omp parallel for schedule(static)
    for (extent index = 0; index < output->size; ++index)
        result[index] = padding;

    #pragma omp parallel for collapse(2) schedule(static)
    for (extent y = 0; y < resized_height; ++y) {
        for (extent x = 0; x < resized_width; ++x) {
            extent source_y = (extent)((real32)y / scale);
            extent source_x = (extent)((real32)x / scale);
            if (source_y >= height) source_y = height - 1;
            if (source_x >= width) source_x = width - 1;
            extent output_y = pad_y + y;
            extent output_x = pad_x + x;
            const uint8 *pixel =
                input + (source_y * width + source_x) * channels;
            result[output_y * target_width + output_x] =
                (real32)pixel[0] / 255.0f;
            result[(target_height + output_y) * target_width + output_x] =
                (real32)pixel[1] / 255.0f;
            result[(2 * target_height + output_y) * target_width + output_x] =
                (real32)pixel[2] / 255.0f;
        }
    }
    return false;
}
