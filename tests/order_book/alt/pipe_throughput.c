#include "util/util.h"
#include "alt/producer_simulated.h"

#include <inttypes.h>
#include <stdio.h>

#ifndef PIPE_SNAPSHOT_LIMIT
#define PIPE_SNAPSHOT_LIMIT 200
#endif

int main(void) {
    BookStream *stream = order_book_stream_open(NUM_INSTRUMENTS);
    BookSnapshot snapshot;
    uint64 started = time_ns();
    uint64 count = 0;

    if (!stream) return 1;
    while (count < PIPE_SNAPSHOT_LIMIT && order_book_stream_next(stream, &snapshot))
        ++count;

    uint64 elapsed = time_ns() - started;
    order_book_stream_close(stream);
    printf("pipe snapshots=%" PRIu64 " elapsed_s=%.6f throughput_s=%.3f\n",
           count, elapsed / 1e9,
           elapsed ? count * 1e9 / (double)elapsed : 0.0);
    return 0;
}
