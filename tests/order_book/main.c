#include "tensor.h"
#include "util/util.h"

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>


#define SAMPLE_INTERVAL_NS 100000000ULL


static volatile sig_atomic_t stop_requested;


static void request_stop(int signal_number) {
    (void)signal_number;
    stop_requested = 1;
}


int main(void) {
    signal(SIGINT, request_stop);
    signal(SIGTERM, request_stop);

    const extent instruments    = NUM_INSTRUMENTS;
    const extent book_size      = LEVELS * instruments;
    const extent batch_size     = BATCH_SIZE * book_size;
    const extent book_shape[]   = {batch_size};
    const extent scalar_shape[] = {1};

    real64 order_quantity_value  = 0.0;
    real64 reference_price_value = 0.0;


    /*
     * Opens a stream of snapshots over an OS socket
     */
    BookStream *stream = order_book_stream_open(instruments);
    if (!stream) {
        fprintf(stderr, "failed to open order book stream\n");
        return EXIT_FAILURE;
    }
    BookSnapshot *book = order_book_stream_next(stream);
    if (!book) {
        order_book_stream_close(stream);
        return EXIT_FAILURE;
    }


    /*
     * Build the graph once and rebind its inputs for every snapshot
     */
    vtensor *bid_price       = tensor_lazy_alloc(
                                    1                       /* rank */,
                                    book_shape              /* shape */,
                                    REAL64                  /* dtype */,
                                    1                       /* num_static_refs */,
                                    true                    /* init_with_data */,
                                    true                    /* init_with_static_data */,
                                    (dptr *)book->bid_price /* data */
                                );
    vtensor *bid_quantity    = tensor_lazy_alloc(1, book_shape, REAL64, 1,
                                                 true, true, (dptr *)book->bid_quantity);
    vtensor *ask_price       = tensor_lazy_alloc(1, book_shape, REAL64, 1,
                                                 true, true, (dptr *)book->ask_price);
    vtensor *ask_quantity    = tensor_lazy_alloc(1, book_shape, REAL64, 1,
                                                 true, true, (dptr *)book->ask_quantity);
    vtensor *order_quantity  = tensor_lazy_alloc(1, scalar_shape, REAL64, 1,
                                                 true, true, (dptr *)&order_quantity_value);
    vtensor *reference_price = tensor_lazy_alloc(1, scalar_shape, REAL64, 1,
                                                 true, true, (dptr *)&reference_price_value);

    vtensor *imbalance       = tensor_order_book_imbalance(bid_quantity, ask_quantity);
    vtensor *microprice      = tensor_microprice          (bid_price, bid_quantity,
                                                           ask_price, ask_quantity);
    vtensor *slippage        = tensor_slippage            (ask_price, ask_quantity,
                                                           order_quantity, reference_price);


    /*
     * Previous sample
     */
    real64 previous_best_bid = 0.0;
    real64 previous_best_ask = 0.0;
    uint64 next_sample       = time_ns() + SAMPLE_INTERVAL_NS;

    PipelineStats queue_pipeline   = {.min = UINT64_MAX};
    PipelineStats compute_pipeline = {.min = UINT64_MAX};


    while (!stop_requested && book) {
        uint64 pipeline_enter = time_ns();

        /*
         * Event-driven or periodic sampling
         */
        uint64 now = time_ns();
        bool sample_event = now >= next_sample;
        bool market_event = book->bid_price[0] != previous_best_bid ||
                            book->ask_price[0] != previous_best_ask;

        previous_best_bid = book->bid_price[0];
        previous_best_ask = book->ask_price[0];

        if (sample_event)
            next_sample = now + SAMPLE_INTERVAL_NS;

        order_quantity_value = 0.0;
        for (uint64 i = 0; i < LEVELS; ++i)
            order_quantity_value += book->ask_quantity[i];

        order_quantity_value /= 4.0;
        reference_price_value = book->ask_price[0];


        /*
         * Rebind to the new data
         */
        tensor_lazy_view_from(bid_price,       (dptr *)book->bid_price);
        tensor_lazy_view_from(bid_quantity,    (dptr *)book->bid_quantity);
        tensor_lazy_view_from(ask_price,       (dptr *)book->ask_price);
        tensor_lazy_view_from(ask_quantity,    (dptr *)book->ask_quantity);
        tensor_lazy_view_from(order_quantity,  (dptr *)&order_quantity_value);
        tensor_lazy_view_from(reference_price, (dptr *)&reference_price_value);


        /*
         * Materialize only when the output is interesting
         */
        if (/* true */ sample_event || market_event) {
            real64 *imbalance_result  = tensor_lazy_view_to(imbalance);
            real64 *microprice_result = tensor_lazy_view_to(microprice);
            real64 *slippage_result   = tensor_lazy_view_to(slippage);

            pipeline_stats_record_pair(&queue_pipeline, &compute_pipeline,
                                       book->received_ns, pipeline_enter,
                                       time_ns());
        }

        book = order_book_stream_next(stream);
    }


    order_book_stream_close(stream);
    mem_pool_shutdown();

    return EXIT_SUCCESS;
}
