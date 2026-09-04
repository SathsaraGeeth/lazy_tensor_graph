#include "tensor_ops.h"

#include <stdio.h>

int main(void) {
    extent shape[] = {4};
    real32 values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    vtensor *input = tensor_lazy_alloc(1, shape, REAL32, 0, true, true,
                                       (dptr *)values);
    const real32 *result = input ? (const real32 *)tensor_lazy_view_to(input)
                                 : NULL;
    int failed = !result;

    for (extent index = 0; !failed && index < 4; ++index)
        failed = result[index] != values[index];

    tensor_lazy_free(input);
    mem_pool_shutdown();
    if (failed) {
        fputs("lazy tensor smoke test failed\n", stderr);
        return 1;
    }
    puts("lazy tensor smoke test passed");
    return 0;
}
