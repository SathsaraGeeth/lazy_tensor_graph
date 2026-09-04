#ifndef ORDER_BOOK_PRODUCER_SIMULATED_H
#define ORDER_BOOK_PRODUCER_SIMULATED_H

#include "util.h"

typedef struct SimulatedBookStream SimulatedBookStream;

SimulatedBookStream *simulated_stream_open(const char *path);
bool simulated_stream_next(SimulatedBookStream *stream, BookSnapshot *snapshot);
void simulated_stream_close(SimulatedBookStream *stream);

#endif
