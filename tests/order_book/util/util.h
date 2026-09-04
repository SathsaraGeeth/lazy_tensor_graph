#ifndef UTIL_H
#define UTIL_H

#include "tensor.h"

#include <stdbool.h>
#include <time.h>


/* 
 * deeper buffers amortize network overhead
 * but will increase the latency
 * (and memory usage - but it is less 
 * of a concern than latency)
 */
#ifndef LEVELS
#define LEVELS          25
#endif
#ifndef NUM_INSTRUMENTS
#define NUM_INSTRUMENTS 498
#endif
#ifndef BUFFER_DEPTH
#define BUFFER_DEPTH    256
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE      4
#endif
#ifndef WARM_UP
#define WARM_UP         224
#endif

uint64 time_ns(void);

/*
 * Use SoA ofc not AoS
 */
typedef struct {
    uint64 received_ns;
    uint64 timestamp   [BATCH_SIZE*NUM_INSTRUMENTS];
    real64 bid_price   [BATCH_SIZE*LEVELS*NUM_INSTRUMENTS];
    real64 bid_quantity[BATCH_SIZE*LEVELS*NUM_INSTRUMENTS];
    real64 ask_price   [BATCH_SIZE*LEVELS*NUM_INSTRUMENTS];
    real64 ask_quantity[BATCH_SIZE*LEVELS*NUM_INSTRUMENTS];
} BookSnapshot;

typedef struct BookStream         BookStream;

BookStream   *order_book_stream_open (extent count);             // 0 - failed

/*
 * A Zero copy api
 * Directly let consume from the 
 * queue
 */
BookSnapshot *order_book_stream_next (BookStream *stream);       // 0 - closed
void          order_book_stream_close(BookStream *stream);                                     


typedef struct {
    uint64 count, sum, min, max;
} PipelineStats;

void  pipeline_stats_record     (PipelineStats *stats, uint64 started_ns);
void  pipeline_stats_record_pair(PipelineStats *queue_stats,
                                 PipelineStats *compute_stats,
                                 uint64 queued_ns,
                                 uint64 compute_start_ns,
                                 uint64 completed_ns);
#endif /* UTIL_H */
