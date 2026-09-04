#include "../../common/dev_tools/profiler/internal_trace.h"
#include "tensor_ops.h"
#include "tensor_jit.h"

vtensor *tensor_order_book_imbalance(const vtensor *bid_quantity,
                                     const vtensor *ask_quantity) {
    if (!bid_quantity || !ask_quantity ||
        bid_quantity->rank != 1 || ask_quantity->rank != 1 ||
        bid_quantity->dtype != ask_quantity->dtype ||
        (bid_quantity->dtype != REAL32 && bid_quantity->dtype != REAL64)) {
        return NULL;
    }

    const extent levels = ((const extent *)bid_quantity->shape)[0];
    if (!levels || ((const extent *)ask_quantity->shape)[0] != levels) {
        return NULL;
    }

    extent output_shape[] = {1};
    vtensor *output = tensor_lazy_alloc(
        1, output_shape, bid_quantity->dtype, 0, false, false, NULL);
    const vtensor *inputs[] = {bid_quantity, ask_quantity};

    if (!output ||
        tensor_lazy_op_dispatch(order_book_imbalance_ker, output, inputs,
                                NULL)) {
        return NULL;
    }

    return output;
}

boolean order_book_imbalance_lower(tensor *output,
                                   const tensor *const *inputs,
                                   extent input_count, const void *parameters,
                                   extent parameter_bytes) {
    TENSOR_KERNEL_TRACE("ORDER_BOOK_IMBALANCE", order_book_imbalance_ker.op,
                        output, inputs, input_count);
    (void)parameters;

    if (input_count != 2 || parameter_bytes || !output || !inputs ||
        !inputs[0] || !inputs[1] || !output->data ||
        !inputs[0]->data || !inputs[1]->data ||
        !output->data->ptr || !inputs[0]->data->ptr ||
        !inputs[1]->data->ptr || output->rank != 1 ||
        inputs[0]->rank != 1 || inputs[1]->rank != 1 ||
        (output->dtype != REAL32 && output->dtype != REAL64) ||
        output->dtype != inputs[0]->dtype ||
        output->dtype != inputs[1]->dtype) {
        return true;
    }

    if (((const extent *)output->shape)[0] != 1 ||
        !((const extent *)inputs[0]->shape)[0] ||
        ((const extent *)inputs[0]->shape)[0] !=
            ((const extent *)inputs[1]->shape)[0]) {
        return true;
    }

    jit_cache_key_t key = tensor_jit_key_create(order_book_imbalance_ker.op,
                                                output, inputs, input_count,
                                                get_device());
    ker_t kernel = (ker_t)tensor_jit_get(key, tensor_get_jit_cache());
    return !kernel ||
           tensor_ker_dispatch(kernel, output, inputs, input_count, NULL,
                               parameter_bytes);
}
