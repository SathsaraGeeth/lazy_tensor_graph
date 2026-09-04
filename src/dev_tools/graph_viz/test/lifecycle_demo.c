#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

static void phase(const char *name) {
    printf("%s\n", name);
    fflush(stdout);
}

int main(void) {
    extent matrix_shape[] = {512, 512};
    extent matrix_elements = matrix_shape[0] * matrix_shape[1];
    real32 *a_data = malloc(matrix_elements * sizeof(*a_data));
    real32 *b_data = malloc(matrix_elements * sizeof(*b_data));
    if (!a_data || !b_data) return 1;
    for (extent i = 0; i < matrix_elements; ++i) {
        a_data[i] = (real32)(i % 17) / 17000.0f;
        b_data[i] = (real32)(i % 13) / 13000.0f;
    }

    /* A starts with two static references, releasing one does not destroy it. */
    vtensor *A = tensor_lazy_alloc(2, matrix_shape, REAL32, 2, true, false, (dptr *)a_data);
    vtensor *B = tensor_lazy_alloc(2, matrix_shape, REAL32, 1, true, false, (dptr *)b_data);
    if (!A || !B) return 1;
    tensor_lazy_free(A);
    phase("A keeps one static reference");

    vtensor *C = tensor_matrix_mul(A, B);
    vtensor *D = tensor_brightness(C, 0.10f);
    vtensor *E = tensor_brightness(D, 0.20f);
    vtensor *F = tensor_brightness(E, 0.30f);
    if (!C || !D || !E || !F) return 1;
    phase("matrix graph built lazily");
    const real32 *f_data = (const real32 *)tensor_lazy_view_to(F);
    phase("matrix graph materialized");
    tensor_lazy_free(F);

    vtensor *G = tensor_matrix_mul(A, B);
    vtensor *H = tensor_brightness(G, 0.40f);
    vtensor *I = tensor_brightness(H, 0.50f);
    if (!G || !H || !I) return 1;
    phase("second matrix graph built lazily");
    tensor_lazy_free(I);
    phase("unmaterialized graph released");

    extent x_shape[] = {1, 3, 128, 128};
    extent w_shape[] = {16, 3, 3, 3};
    extent x_elements = x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3];
    extent w_elements = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3];
    real32 *x_data = malloc(x_elements * sizeof(*x_data));
    real32 *w_data = malloc(w_elements * sizeof(*w_data));
    if (!x_data || !w_data) return 1;
    for (extent i = 0; i < x_elements; ++i) x_data[i] = (real32)(i % 23) / 2300.0f;
    for (extent i = 0; i < w_elements; ++i) w_data[i] = (real32)(i % 7) / 6300.0f;

    vtensor *J = tensor_lazy_alloc(4, x_shape, REAL32, 1, true, false, (dptr *)x_data);
    vtensor *K = tensor_lazy_alloc(4, w_shape, REAL32, 1, true, false, (dptr *)w_data);
    vtensor *L = tensor_conv2d(J, K, 1, 1, 1, 1);
    vtensor *M = tensor_brightness(L, 0.15f);
    vtensor *N = tensor_brightness(M, 0.25f);
    vtensor *O = tensor_brightness(N, 0.35f);
    if (!J || !K || !L || !M || !N || !O) return 1;
    phase("convolution graph built lazily");
    const real32 *o_data = (const real32 *)tensor_lazy_view_to(O);
    phase("convolution graph materialized");

    printf("materialized outputs: %s\n", f_data && o_data ? "yes" : "no");

    mem_pool_shutdown();
    free(a_data);
    free(b_data);
    free(x_data);
    free(w_data);
    phase("complete");
    return f_data && o_data ? 0 : 1;
}
