#include "producer_simulated.h"
#include "queue.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>

#ifdef SIMULATED_PRODUCER
#define STREAM_OPEN order_book_stream_open
#define STREAM_NEXT order_book_stream_next
#define STREAM_CLOSE order_book_stream_close
#else
#define STREAM_OPEN simulated_stream_open
#define STREAM_NEXT simulated_stream_next
#define STREAM_CLOSE simulated_stream_close
#endif

typedef struct {
    uint64       delay_ns;
    BookSnapshot snapshot;
} RecordedBatch;

struct BookStream {
    void        *data;
    size_t       size;
    size_t       offset;
    BookQueue    queue;
    pthread_t    reader;
    atomic_bool  stop;
};

static void *replay_reader(void *opaque) {
	BookStream *stream = opaque;
    RecordedBatch batch;
    while (!atomic_load(&stream->stop) &&
           stream->offset + sizeof(batch) <= stream->size) {
        memcpy(&batch, (char *)stream->data + stream->offset, sizeof(batch));
        stream->offset += sizeof(batch);
        if (atomic_load(&stream->stop)) break;
        batch.snapshot.received_ns = time_ns();
        book_queue_push(&stream->queue, &batch.snapshot);
    }
    book_queue_close(&stream->queue);
    return NULL;
}

BookStream *STREAM_OPEN(extent count) {
	const char *path = getenv("ORDER_BOOK_CAPTURE");
	(void)count;
	BookStream *stream = calloc(1, sizeof(*stream));
	if (!stream || !path) {
		free(stream);
		return NULL;
	}
    int fd = open(path, O_RDONLY);
    struct stat info;
    if (fd < 0 || fstat(fd, &info) || !info.st_size) {
        if (fd >= 0) close(fd);
        free(stream);
        return NULL;
    }
    stream->size = (size_t)info.st_size;
    stream->data = mmap(NULL, stream->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (stream->data == MAP_FAILED || !book_queue_open(&stream->queue)) {
        if (stream->data != MAP_FAILED) munmap(stream->data, stream->size);
        free(stream);
        return NULL;
    }
    if (pthread_create(&stream->reader, NULL, replay_reader, stream)) {
		STREAM_CLOSE(stream);
        return NULL;
    }
    return stream;
}

bool STREAM_NEXT(BookStream *stream, BookSnapshot *snapshot) {
    return book_queue_pop(&stream->queue, snapshot);
}

void STREAM_CLOSE(BookStream *stream) {
    if (!stream) return;
    atomic_store(&stream->stop, true);
    book_queue_close(&stream->queue);
    if (stream->reader) pthread_join(stream->reader, NULL);
    if (stream->data && stream->data != MAP_FAILED)
        munmap(stream->data, stream->size);
    book_queue_destroy(&stream->queue);
    free(stream);
}
