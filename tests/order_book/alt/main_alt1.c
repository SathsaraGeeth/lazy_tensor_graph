#include "tensor.h"
#include "util/util.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>


#define SAMPLE_INTERVAL_NS 10000000ULL   /* 10 ms */


int main(int argc, char **argv) {
    const extent book_shape[]   = {LEVELS};
    const extent scalar_shape[] = {1};

    real64 order_quantity_value;
    real64 reference_price_value;


    /*
     * Opens a stream of snapshots over an OS socket
     */
    BookStream *stream = order_book_stream_open("BTCUSDT");
    if (!stream) {
        fprintf(stderr, "failed to open order book stream\n");
        return EXIT_FAILURE;
    }
    BookSnapshot book;


    /*
     * Previous sample
     */

    real64 previous_best_bid = 0.0;
    real64 previous_best_ask = 0.0;
    uint64 next_sample       = time_ns() + SAMPLE_INTERVAL_NS;
    uint64 snapshots         = 0;
    uint64 snapshot_limit    = argc > 1 ? strtoull(argv[1], NULL, 10) : 30;
    
    while (order_book_stream_next(stream, &book)) {
        ++snapshots;
        /*
         * Events driven sampling or periodic sampling
         */
        uint64 now = time_ns();
        bool sample_event = now >= next_sample;
        bool market_event = book.bid_price[0] != previous_best_bid ||
                            book.ask_price[0] != previous_best_ask;

        previous_best_bid = book.bid_price[0];
        previous_best_ask = book.ask_price[0];


        if (sample_event)
            next_sample = now + SAMPLE_INTERVAL_NS;

        order_quantity_value = 0.0;

        for (uint64 i = 0; i < LEVELS; ++i)
            order_quantity_value += book.ask_quantity[i];

        order_quantity_value /= 4.0;

        reference_price_value = book.ask_price[0];


        /*
         * Process the snapshots
         */
        vtensor *bid_price       = tensor_lazy_alloc(
                                        1                       /* rank */,
                                        book_shape              /* shape */,
                                        REAL64                  /* dtype */,
                                        0                       /* num_static_refs */,
                                        true                    /* init_with_data */,
                                        true                    /* init_with_static_data */,
                                        (dptr *)book.bid_price  /* data */
                                    );
        vtensor *bid_quantity    = tensor_lazy_alloc(1, book_shape, REAL64, 0,
                                                     true, true, (dptr *)book.bid_quantity);
        vtensor *ask_price       = tensor_lazy_alloc(1, book_shape, REAL64, 0,
                                                     true, true, (dptr *)book.ask_price);
        vtensor *ask_quantity    = tensor_lazy_alloc(1, book_shape, REAL64, 0,
                                                     true, true, (dptr *)book.ask_quantity);
        vtensor *order_quantity  = tensor_lazy_alloc(1, scalar_shape, REAL64, 0,
                                                     true, true, (dptr *)&order_quantity_value);
        vtensor *reference_price = tensor_lazy_alloc(1, scalar_shape, REAL64, 0,
                                                     true, true, (dptr *)&reference_price_value);

        vtensor *imbalance       = tensor_order_book_imbalance(bid_quantity, ask_quantity);
        vtensor *microprice      = tensor_microprice          (bid_price, bid_quantity,
                                                               ask_price, ask_quantity);
        vtensor *slippage        = tensor_slippage            (ask_price, ask_quantity,
                                                               order_quantity, reference_price);




        /*
         * Look at the output if it is intresting
         */
        if (sample_event || market_event) {
            real64 *imbalance_result  = tensor_lazy_view_to(imbalance);
            real64 *microprice_result = tensor_lazy_view_to(microprice);
            real64 *slippage_result   = tensor_lazy_view_to(slippage);

            printf("[%s] timestamp: %" PRIu64 "  bid: %.8f  ask: %.8f  imbalance: %.8f  microprice: %.8f  slippage: %.8f\n",
                    market_event ? "MARKET" : "SAMPLE", book.timestamp[0], book.bid_price[0], book.ask_price[0],
                    imbalance_result[0], microprice_result[0], slippage_result[0]);
        }
        
        /*
         * Release before processing the next one
         */
        tensor_lazy_free(imbalance);
        tensor_lazy_free(microprice);
        tensor_lazy_free(slippage);

        if (snapshots >= snapshot_limit) break;
    }

    order_book_stream_close(stream);
    mem_pool_shutdown();

    return EXIT_SUCCESS;
}
