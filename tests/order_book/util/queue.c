#include "queue.h"

#include <stdlib.h>
#if defined(__APPLE__)
#include <sched.h>
#define TENSOR_THREAD_YIELD() sched_yield()
#else
#include <threads.h>
#define TENSOR_THREAD_YIELD() thrd_yield()
#endif


bool book_queue_open(BookQueue *queue) {
    *queue = (BookQueue){0};
    queue->items = calloc(QUEUE_CAPACITY, sizeof(*queue->items));
    return queue->items != NULL;
}


void book_queue_push(BookQueue *queue, const BookSnapshot *snapshot) {
    uint64 tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);
    uint64 head = atomic_load_explicit(&queue->head, memory_order_acquire);

    if (tail - head == QUEUE_CAPACITY)
        return;

    queue->items[tail % QUEUE_CAPACITY] = *snapshot;
    atomic_store_explicit(&queue->tail, tail + 1, memory_order_release);
}


BookSnapshot *book_queue_pop(BookQueue *queue) {
    if (queue->held) {
        uint64 head = atomic_load_explicit(&queue->head,
                                           memory_order_relaxed);

        atomic_store_explicit(&queue->head, head + 1, memory_order_release);
        queue->held = false;
    }

    for (;;) {
        uint64 head = atomic_load_explicit(&queue->head,
                                           memory_order_relaxed);
        uint64 tail = atomic_load_explicit(&queue->tail,
                                           memory_order_acquire);

        if (head != tail) {
            queue->held = true;
            return &queue->items[head % QUEUE_CAPACITY];
        }

        if (atomic_load_explicit(&queue->closed, memory_order_acquire))
            return NULL;

        TENSOR_THREAD_YIELD();
    }
}


void book_queue_close(BookQueue *queue) {
    if (queue->items)
        atomic_store_explicit(&queue->closed, true, memory_order_release);
}


void book_queue_destroy(BookQueue *queue) {
    free(queue->items);
    queue->items = NULL;
}
