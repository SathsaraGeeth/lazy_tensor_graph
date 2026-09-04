#include "../../common/dev_tools/profiler/internal_trace.h"
#include "tensor_ops.h"
#include "tensor_jit.h"

#include <stdint.h>

typedef struct {
    uint64 stride_h;
    uint64 stride_w;
    uint64 pad_h;
    uint64 pad_w;
} conv2d_parameters;

vtensor *tensor_conv2d(const vtensor *input, const vtensor *weight,
                       extent stride_h, extent stride_w,
                       extent pad_h, extent pad_w) {

    if (!input || !weight || input->rank != 4 || weight->rank != 4 ||
        input->dtype != weight->dtype || !stride_h || !stride_w) {
        return NULL;
    }

    const extent *input_shape  = (const extent *)input->shape;
    const extent *weight_shape = (const extent *)weight->shape;

    if (input_shape[1] != weight_shape[1] ||
        input_shape[2] + 2 * pad_h < weight_shape[2] ||
        input_shape[3] + 2 * pad_w < weight_shape[3])
        return NULL;

    extent output_shape[] = {input_shape[0], weight_shape[0],
                            (input_shape[2] + 2 * pad_h - weight_shape[2]) / stride_h + 1,
                            (input_shape[3] + 2 * pad_w - weight_shape[3]) / stride_w + 1};
                            
    conv2d_parameters parameters = {(uint64)stride_h, (uint64)stride_w, (uint64)pad_h, (uint64)pad_w};

    vtensor *output = tensor_lazy_alloc(4, output_shape, input->dtype, 0,false, false, NULL);
    const vtensor *inputs[] = {input, weight};

    if (!output || tensor_lazy_op_dispatch(linalg_conv2d_ker, output, inputs, &parameters))
        return NULL;

    return output;
}

boolean linalg_conv2d_lower(tensor *output, const tensor *const *inputs,
                            extent input_count, const void *parameters,
                            extent parameter_bytes) {

    TENSOR_KERNEL_TRACE("LINALG_CONV2D", linalg_conv2d_ker.op, output, inputs, input_count);

    if (input_count != 2 || parameter_bytes != sizeof(conv2d_parameters) ||
        !parameters || !output || !inputs || !inputs[0] || !inputs[1] ||
        !output->data || !inputs[0]->data || !inputs[1]->data ||
        !output->data->ptr || !inputs[0]->data->ptr || !inputs[1]->data->ptr ||
        output->rank != 4 || inputs[0]->rank != 4 || inputs[1]->rank != 4 ||
        output->dtype != inputs[0]->dtype || output->dtype != inputs[1]->dtype)

        return true;

    const conv2d_parameters *p = (const conv2d_parameters *)parameters;
    const extent *x = (const extent *)inputs[0]->shape;
    const extent *w = (const extent *)inputs[1]->shape;
    const extent *o = (const extent *)output->shape;

    if (!p->stride_h || !p->stride_w || x[1] != w[1] ||
        x[2] + 2 * p->pad_h < w[2] || x[3] + 2 * p->pad_w < w[3] ||
        o[0] != x[0] || o[1] != w[0] ||
        o[2] != (x[2] + 2 * p->pad_h - w[2]) / p->stride_h + 1 ||
        o[3] != (x[3] + 2 * p->pad_w - w[3]) / p->stride_w + 1)
        return true;

    jit_cache_key_t key = tensor_jit_key_create(linalg_conv2d_ker.op, output, inputs, input_count, get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());

    return !kernel || tensor_ker_dispatch(kernel, output, inputs, input_count, parameters, parameter_bytes);
}
