#include "../../common/dev_tools/profiler/internal_trace.h"
#include "tensor_ops.h"
#include "tensor_jit.h"

static boolean valid_book_input(const vtensor *value) {
    return value && value->rank == 1 &&
           (value->dtype == REAL32 || value->dtype == REAL64);
}

vtensor *tensor_slippage(const vtensor *price,
                         const vtensor *available_quantity,
                         const vtensor *order_quantity,
                         const vtensor *reference_price) {
    if (!valid_book_input(price) ||
        !valid_book_input(available_quantity) ||
        !valid_book_input(order_quantity) ||
        !valid_book_input(reference_price) ||
        price->dtype != available_quantity->dtype ||
        price->dtype != order_quantity->dtype ||
        price->dtype != reference_price->dtype) {
        return NULL;
    }

    const extent levels = ((const extent *)price->shape)[0];
    if (!levels ||
        ((const extent *)available_quantity->shape)[0] != levels ||
        ((const extent *)order_quantity->shape)[0] != 1 ||
        ((const extent *)reference_price->shape)[0] != 1) {
        return NULL;
    }

    extent output_shape[] = {1};
    vtensor *output = tensor_lazy_alloc(
        1, output_shape, price->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {
        price, available_quantity, order_quantity, reference_price
    };

    if (!output ||
        tensor_lazy_op_dispatch(slippage_ker, output, inputs, NULL)) {
        return NULL;
    }

    return output;
}

boolean slippage_lower(tensor *output, const tensor *const *inputs,
                       extent input_count, const void *parameters,
                       extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("SLIPPAGE", slippage_ker.op, output, inputs,
                        input_count);
    (void)parameters;

    if (input_count != 4 || parameter_bytes || !output || !inputs ||
        !inputs[0] || !inputs[1] || !inputs[2] || !inputs[3] ||
        !output->data || !inputs[0]->data || !inputs[1]->data ||
        !inputs[2]->data || !inputs[3]->data ||
        !output->data->ptr || !inputs[0]->data->ptr ||
        !inputs[1]->data->ptr || !inputs[2]->data->ptr ||
        !inputs[3]->data->ptr || output->rank != 1 ||
        inputs[0]->rank != 1 || inputs[1]->rank != 1 ||
        inputs[2]->rank != 1 || inputs[3]->rank != 1 ||
        (output->dtype != REAL32 && output->dtype != REAL64) ||
        output->dtype != inputs[0]->dtype ||
        output->dtype != inputs[1]->dtype ||
        output->dtype != inputs[2]->dtype ||
        output->dtype != inputs[3]->dtype) {
        return true;
    }

    const extent levels = ((const extent *)inputs[0]->shape)[0];
    if (!levels || ((const extent *)output->shape)[0] != 1 ||
        ((const extent *)inputs[1]->shape)[0] != levels ||
        ((const extent *)inputs[2]->shape)[0] != 1 ||
        ((const extent *)inputs[3]->shape)[0] != 1) {
        return true;
    }

    jit_cache_key_t key = tensor_jit_key_create(slippage_ker.op, output,
                                                inputs, input_count,
                                                get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel ||
           tensor_ker_dispatch(kernel, output, inputs, input_count, NULL,
                               parameter_bytes);
}
