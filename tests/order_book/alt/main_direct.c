#include "util/util.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>


#define SAMPLE_INTERVAL_NS 100000000ULL


static volatile sig_atomic_t stop_requested;
static volatile real64 result_sink;


static void request_stop(int signal_number) {
    (void)signal_number;
    stop_requested = 1;
}


static real64 order_book_imbalance(const real64 *bid_quantity,
                                   const real64 *ask_quantity,
                                   extent count) {
    real64 bid = 0.0;
    real64 ask = 0.0;

    for (extent i = 0; i < count; ++i) {
        bid += bid_quantity[i];
        ask += ask_quantity[i];
    }

    return (bid - ask) / (bid + ask);
}


static void microprice(const BookSnapshot *book, real64 *output,
                       extent count) {
    for (extent i = 0; i < count; ++i) {
        real64 total = book->bid_quantity[i] + book->ask_quantity[i];

        output[i] = total == 0.0
                  ? (book->bid_price[i] + book->ask_price[i]) / 2.0
                  : (book->ask_price[i] * book->bid_quantity[i] +
                     book->bid_price[i] * book->ask_quantity[i]) / total;
    }
}


static real64 slippage(const real64 *ask_price, const real64 *ask_quantity,
                       extent count, real64 quantity, real64 reference) {
    real64 remaining = quantity;
    real64 filled = 0.0;
    real64 cost = 0.0;

    for (extent i = 0; i < count && remaining > 0.0; ++i) {
        real64 fill = ask_quantity[i] < remaining
                    ? ask_quantity[i] : remaining;

        cost += ask_price[i] * fill;
        filled += fill;
        remaining -= fill;
    }

    return ((cost / filled) - reference) / reference;
}


int main(void) {
    signal(SIGINT, request_stop);
    signal(SIGTERM, request_stop);

    const extent instruments = NUM_INSTRUMENTS;
    const extent batch_size  = BATCH_SIZE * LEVELS * instruments;

    BookStream *stream = order_book_stream_open(instruments);
    if (!stream) {
        fprintf(stderr, "failed to open order book stream\n");
        return EXIT_FAILURE;
    }

    real64 *microprice_output = malloc(batch_size * sizeof(*microprice_output));
    if (!microprice_output) {
        order_book_stream_close(stream);
        return EXIT_FAILURE;
    }

    BookSnapshot *book = order_book_stream_next(stream);
    real64 previous_best_bid = 0.0;
    real64 previous_best_ask = 0.0;
    uint64 next_sample = time_ns() + SAMPLE_INTERVAL_NS;

    PipelineStats queue_pipeline   = {.min = UINT64_MAX};
    PipelineStats compute_pipeline = {.min = UINT64_MAX};

    while (!stop_requested && book) {
        uint64 pipeline_enter = time_ns();
        uint64 now = time_ns();
        bool sample_event = now >= next_sample;
        bool market_event = book->bid_price[0] != previous_best_bid ||
                            book->ask_price[0] != previous_best_ask;

        previous_best_bid = book->bid_price[0];
        previous_best_ask = book->ask_price[0];
        if (sample_event)
            next_sample = now + SAMPLE_INTERVAL_NS;

        real64 order_quantity = 0.0;
        for (extent i = 0; i < LEVELS; ++i)
            order_quantity += book->ask_quantity[i];

        order_quantity /= 4.0;
        real64 reference_price = book->ask_price[0];

        if (sample_event || market_event) {
            real64 imbalance = order_book_imbalance(
                book->bid_quantity, book->ask_quantity, batch_size);
            microprice(book, microprice_output, batch_size);
            real64 price_slippage = slippage(
                book->ask_price, book->ask_quantity, batch_size,
                order_quantity, reference_price);

            result_sink = imbalance + microprice_output[0] + price_slippage;
            pipeline_stats_record_pair(&queue_pipeline, &compute_pipeline,
                                       book->received_ns, pipeline_enter,
                                       time_ns());
        }

        book = order_book_stream_next(stream);
    }

    free(microprice_output);
    order_book_stream_close(stream);
    return EXIT_SUCCESS;
}
