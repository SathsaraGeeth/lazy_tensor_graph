#ifndef ORDER_BOOK_QUEUE_H
#define ORDER_BOOK_QUEUE_H

#include "util.h"

#include <stdalign.h>
#include <stdatomic.h>

#define QUEUE_CAPACITY (WARM_UP + BUFFER_DEPTH)

/*
 * Lock free SPSC
 * A thread safe queue to absorb the
 * the jitter of the nework and OS latency
 * decouples the consumers and produers
 *
     * a contigous array (@ &data)
 *
 * batch * inst * levels
 * [||||][||||][||||][||||] d
 * [||||][||||][||||][||||] e
 * [||||][||||][||||][||||] p
 * [||||][||||][||||][||||] t
 * [||||][||||][||||][||||] h
 */

typedef struct {
    BookSnapshot                     *items;
    alignas(64) atomic_uint_least64_t head;
    alignas(64) atomic_uint_least64_t tail;
    atomic_bool                       closed;
    bool                              held;
} BookQueue;

bool          book_queue_open  (BookQueue *queue);
void          book_queue_push  (BookQueue *queue,
                                const BookSnapshot *snapshot);
BookSnapshot *book_queue_pop    (BookQueue *queue);
void          book_queue_close  (BookQueue *queue);
void          book_queue_destroy(BookQueue *queue);

#endif
