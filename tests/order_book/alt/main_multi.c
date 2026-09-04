#include "tensor.h"
#include "util/util.h"

#include <stdio.h>
#include <stdlib.h>

#define INSTRUMENTS 5

int main(int argc, char **argv) {
    const char *symbols[INSTRUMENTS] = {
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"
    };
    const extent book_shape[] = {INSTRUMENTS * LEVELS};
    const extent scalar_shape[] = {1};
    const uint64 limit = argc > 1 ? strtoull(argv[1], NULL, 10) : 30;

    CombinedBookStream *stream = order_book_combined_open(symbols, INSTRUMENTS);
    if (!stream)
        return EXIT_FAILURE;

    BookSnapshot *book = calloc(1, sizeof(*book));

    for (uint64 snapshot = 0;
         snapshot < limit && order_book_combined_next(stream, book);
         ++snapshot) {
        real64 order_quantity = 0.0;
        for (extent i = 0; i < INSTRUMENTS * LEVELS; ++i)
            order_quantity += book->ask_quantity[i];
        order_quantity /= 4.0;
        real64 reference_price = book->ask_price[0];

        vtensor *bid_price = tensor_lazy_alloc(1, book_shape, REAL64, 0,
                                                true, true, (dptr *)book->bid_price);
        vtensor *bid_quantity = tensor_lazy_alloc(1, book_shape, REAL64, 0,
                                                   true, true, (dptr *)book->bid_quantity);
        vtensor *ask_price = tensor_lazy_alloc(1, book_shape, REAL64, 0,
                                                true, true, (dptr *)book->ask_price);
        vtensor *ask_quantity = tensor_lazy_alloc(1, book_shape, REAL64, 0,
                                                   true, true, (dptr *)book->ask_quantity);
        vtensor *quantity = tensor_lazy_alloc(1, scalar_shape, REAL64, 0,
                                               true, true, (dptr *)&order_quantity);
        vtensor *reference = tensor_lazy_alloc(1, scalar_shape, REAL64, 0,
                                                true, true, (dptr *)&reference_price);

        vtensor *imbalance = tensor_order_book_imbalance(bid_quantity, ask_quantity);
        vtensor *microprice = tensor_microprice(bid_price, bid_quantity,
                                                ask_price, ask_quantity);
        vtensor *slippage = tensor_slippage(ask_price, ask_quantity,
                                            quantity, reference);

        tensor_lazy_view_to(imbalance);
        tensor_lazy_view_to(microprice);
        tensor_lazy_view_to(slippage);
    }

    order_book_combined_close(stream);
    free(book);
    mem_pool_shutdown();
    return EXIT_SUCCESS;
}
