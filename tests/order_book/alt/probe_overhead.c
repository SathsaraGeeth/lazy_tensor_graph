#include "tensor.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef LEVELS
#define LEVELS 50
#endif

#ifndef NUM_INSTRUMENTS
#define NUM_INSTRUMENTS 489
#endif

#ifndef PROBE_ITERATIONS
#define PROBE_ITERATIONS 500
#endif

typedef struct {
    uint64 alloc;
    uint64 graph;
    uint64 imbalance;
    uint64 microprice;
    uint64 slippage;
    uint64 release;
    uint64 direct_imbalance;
    uint64 direct_microprice;
    uint64 direct_slippage;
} ProbeSample;

static volatile real64 probe_sink;
static uint64 probe_time_ns(void);

static void run_direct(const real64 *bid_price, const real64 *bid_quantity,
                       const real64 *ask_price, const real64 *ask_quantity,
                       extent count, real64 order_quantity,
                       real64 reference_price, real64 *microprice,
                       ProbeSample *sample) {
    uint64 started = probe_time_ns();
    real64 bid = 0.0;
    real64 ask = 0.0;
    for (extent i = 0; i < count; ++i) {
        bid += bid_quantity[i];
        ask += ask_quantity[i];
    }
    probe_sink = (bid - ask) / (bid + ask);
    uint64 imbalance_done = probe_time_ns();

    for (extent i = 0; i < count; ++i) {
        real64 quantity = bid_quantity[i] + ask_quantity[i];
        microprice[i] = quantity == 0.0
                      ? (bid_price[i] + ask_price[i]) / 2.0
                      : (ask_price[i] * bid_quantity[i] +
                         bid_price[i] * ask_quantity[i]) / quantity;
    }
    uint64 microprice_done = probe_time_ns();

    real64 checksum = 0.0;
    for (extent i = 0; i < count; ++i) checksum += microprice[i];
    probe_sink = checksum;
    uint64 slippage_started = probe_time_ns();

    real64 remaining = order_quantity;
    real64 filled = 0.0;
    real64 cost = 0.0;
    for (extent i = 0; i < count && remaining > 0.0; ++i) {
        real64 quantity = ask_quantity[i] < remaining
                        ? ask_quantity[i] : remaining;
        cost += ask_price[i] * quantity;
        filled += quantity;
        remaining -= quantity;
    }
    probe_sink = ((cost / filled) - reference_price) / reference_price;
    uint64 slippage_done = probe_time_ns();

    sample->direct_imbalance = imbalance_done - started;
    sample->direct_microprice = microprice_done - imbalance_done;
    sample->direct_slippage = slippage_done - slippage_started;
}

static uint64 probe_time_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return (uint64)now.tv_sec * UINT64_C(1000000000) + (uint64)now.tv_nsec;
}

static int compare_u64(const void *left, const void *right) {
    uint64 a = *(const uint64 *)left;
    uint64 b = *(const uint64 *)right;
    return (a > b) - (a < b);
}

static void report(const char *name, const uint64 *values, extent count) {
    uint64 *ordered = malloc(count * sizeof(*ordered));
    long double total = 0.0;

    if (!ordered) return;
    for (extent i = 0; i < count; ++i) {
        ordered[i] = values[i];
        total += values[i];
    }
    qsort(ordered, count, sizeof(*ordered), compare_u64);

    printf("%-16s %10.3Lf %10.3f %10.3f %10.3f\n",
           name,
           total / count / 1000.0L,
           ordered[count / 2] / 1000.0,
           ordered[(count * 95) / 100] / 1000.0,
           ordered[count - 1] / 1000.0);
    free(ordered);
}

static void initialize_books(real64 *bid_price, real64 *bid_quantity,
                             real64 *ask_price, real64 *ask_quantity,
                             extent count) {
    for (extent i = 0; i < count; ++i) {
        extent level = i % LEVELS;
        bid_price[i] = 100.0 - 0.01 * level;
        bid_quantity[i] = 1.0 + 0.001 * level;
        ask_price[i] = 100.01 + 0.01 * level;
        ask_quantity[i] = 1.0 + 0.001 * level;
    }
}

int main(void) {
    const extent book_size = LEVELS * NUM_INSTRUMENTS;
    const extent book_shape[] = {book_size};
    const extent scalar_shape[] = {1};
    const real64 order_quantity = 4.0;
    const real64 reference_price = 100.01;
    real64 *storage = malloc(4 * book_size * sizeof(*storage));
    real64 *direct_microprice = malloc(book_size * sizeof(*direct_microprice));
    ProbeSample *samples = calloc(PROBE_ITERATIONS, sizeof(*samples));

    if (!storage || !direct_microprice || !samples) return EXIT_FAILURE;
    real64 *bid_price_data = storage;
    real64 *bid_quantity_data = storage + book_size;
    real64 *ask_price_data = storage + 2 * book_size;
    real64 *ask_quantity_data = storage + 3 * book_size;
    initialize_books(bid_price_data, bid_quantity_data,
                     ask_price_data, ask_quantity_data, book_size);

#ifdef REUSE_MICROPRICE
    vtensor *bid_price = tensor_lazy_alloc(1, book_shape, REAL64, 0, true, true,
                                           (dptr *)bid_price_data);
    vtensor *bid_quantity = tensor_lazy_alloc(1, book_shape, REAL64, 0, true, true,
                                              (dptr *)bid_quantity_data);
    vtensor *ask_price = tensor_lazy_alloc(1, book_shape, REAL64, 0, true, true,
                                           (dptr *)ask_price_data);
    vtensor *ask_quantity = tensor_lazy_alloc(1, book_shape, REAL64, 0, true, true,
                                              (dptr *)ask_quantity_data);
    vtensor *microprice = tensor_microprice(bid_price, bid_quantity,
                                            ask_price, ask_quantity);
    (void)tensor_lazy_view_to(microprice);

    for (extent i = 0; i < PROBE_ITERATIONS; ++i) {
        tensor_lazy_view_from(bid_price, (dptr *)bid_price_data);
        tensor_lazy_view_from(bid_quantity, (dptr *)bid_quantity_data);
        tensor_lazy_view_from(ask_price, (dptr *)ask_price_data);
        tensor_lazy_view_from(ask_quantity, (dptr *)ask_quantity_data);

        uint64 started = probe_time_ns();
        (void)tensor_lazy_view_to(microprice);
        samples[i].microprice = probe_time_ns() - started;
        run_direct(bid_price_data, bid_quantity_data,
                   ask_price_data, ask_quantity_data, book_size,
                   order_quantity, reference_price, direct_microprice,
                   &samples[i]);
    }

    uint64 tensor_values[PROBE_ITERATIONS];
    uint64 direct_values[PROBE_ITERATIONS];
    for (extent i = 0; i < PROBE_ITERATIONS; ++i) {
        tensor_values[i] = samples[i].microprice;
        direct_values[i] = samples[i].direct_microprice;
    }
    printf("shape=[%u] reused_graph=true iterations=%u\n",
           (unsigned)book_size, PROBE_ITERATIONS);
    printf("phase               mean_us  median_us     p95_us     max_us\n");
    printf("---------------- ---------- ---------- ---------- ----------\n");
    report("tensor microprice", tensor_values, PROBE_ITERATIONS);
    report("direct microprice", direct_values, PROBE_ITERATIONS);

    tensor_lazy_free(microprice);
    mem_pool_shutdown();
    free(samples);
    free(direct_microprice);
    free(storage);
    return EXIT_SUCCESS;
#endif

    for (extent i = 0; i < PROBE_ITERATIONS; ++i) {
        uint64 started = probe_time_ns();
        vtensor *bid_price = tensor_lazy_alloc(1, book_shape, REAL64, 0, true, true,
                                               (dptr *)bid_price_data);
        vtensor *bid_quantity = tensor_lazy_alloc(1, book_shape, REAL64, 0, true, true,
                                                  (dptr *)bid_quantity_data);
        vtensor *ask_price = tensor_lazy_alloc(1, book_shape, REAL64, 0, true, true,
                                               (dptr *)ask_price_data);
        vtensor *ask_quantity = tensor_lazy_alloc(1, book_shape, REAL64, 0, true, true,
                                                  (dptr *)ask_quantity_data);
        vtensor *quantity = tensor_lazy_alloc(1, scalar_shape, REAL64, 0, true, true,
                                              (dptr *)&order_quantity);
        vtensor *reference = tensor_lazy_alloc(1, scalar_shape, REAL64, 0, true, true,
                                               (dptr *)&reference_price);
        uint64 allocated = probe_time_ns();

        vtensor *imbalance = tensor_order_book_imbalance(bid_quantity, ask_quantity);
        vtensor *microprice = tensor_microprice(bid_price, bid_quantity,
                                                ask_price, ask_quantity);
        vtensor *slippage = tensor_slippage(ask_price, ask_quantity,
                                            quantity, reference);
        uint64 graphed = probe_time_ns();

#ifdef MATERIALIZE_RESULTS
        (void)tensor_lazy_view_to(imbalance);
#endif
        uint64 imbalance_done = probe_time_ns();
#ifdef MATERIALIZE_RESULTS
        (void)tensor_lazy_view_to(microprice);
#endif
        uint64 microprice_done = probe_time_ns();
#ifdef MATERIALIZE_RESULTS
        (void)tensor_lazy_view_to(slippage);
#endif
        uint64 slippage_done = probe_time_ns();

        tensor_lazy_free(imbalance);
        tensor_lazy_free(microprice);
        tensor_lazy_free(slippage);
        uint64 released = probe_time_ns();

        samples[i] = (ProbeSample){
            .alloc = allocated - started,
            .graph = graphed - allocated,
            .imbalance = imbalance_done - graphed,
            .microprice = microprice_done - imbalance_done,
            .slippage = slippage_done - microprice_done,
            .release = released - slippage_done
        };
        run_direct(bid_price_data, bid_quantity_data,
                   ask_price_data, ask_quantity_data, book_size,
                   order_quantity, reference_price, direct_microprice,
                   &samples[i]);
    }

    printf("shape=[%u] levels=%u instruments=%u iterations=%u\n",
           (unsigned)book_size, LEVELS, NUM_INSTRUMENTS, PROBE_ITERATIONS);
    printf("phase               mean_us  median_us     p95_us     max_us\n");
    printf("---------------- ---------- ---------- ---------- ----------\n");

#define REPORT_FIELD(label, field) do {                                      \
        uint64 values[PROBE_ITERATIONS];                                      \
        for (extent i = 0; i < PROBE_ITERATIONS; ++i) values[i] = samples[i].field; \
        report(label, values, PROBE_ITERATIONS);                              \
    } while (0)

    REPORT_FIELD("input allocation", alloc);
    REPORT_FIELD("API graph build", graph);
    REPORT_FIELD("imbalance view", imbalance);
    REPORT_FIELD("microprice view", microprice);
    REPORT_FIELD("slippage view", slippage);
    REPORT_FIELD("graph release", release);
    REPORT_FIELD("direct imbalance", direct_imbalance);
    REPORT_FIELD("direct microprice", direct_microprice);
    REPORT_FIELD("direct slippage", direct_slippage);
#undef REPORT_FIELD

    mem_pool_shutdown();
    free(samples);
    free(direct_microprice);
    free(storage);
    return EXIT_SUCCESS;
}
