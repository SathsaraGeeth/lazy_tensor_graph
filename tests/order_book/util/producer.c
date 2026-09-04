#include "queue.h"
#include "../alt/cjson/cJSON.h"

#include <ctype.h>
#include <curl/curl.h>
#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <sched.h>
#define TENSOR_THREAD_YIELD() sched_yield()
#else
#include <threads.h>
#define TENSOR_THREAD_YIELD() thrd_yield()
#endif

#define BOOK_CAPACITY 1000
#define INITIAL_DEPTH_LIMIT 100
#define BOOK_INIT_WORKERS 8
#define BOOK_INIT_RETRIES 5
#define MESSAGE_CAPACITY (1 << 20)
#define NETWORK_WORKERS 8
#define EVENT_QUEUE_DEPTH 1024

typedef struct BookStream BookStream;

typedef struct {
    real64 price;
    real64 quantity;
} PriceLevel;

typedef struct {
    atomic_uint_least32_t sequence;
    PriceLevel bids[BOOK_CAPACITY];
    PriceLevel asks[BOOK_CAPACITY];
    extent bid_count;
    extent ask_count;
} Book;

typedef struct {
    extent instrument;
    uint64 timestamp;
    uint64 received_ns;
} BookEvent;

typedef struct {
    BookStream *stream;
    CURL *ws;
    extent first;
    extent count;
    BookEvent events[EVENT_QUEUE_DEPTH];
    atomic_uint_least64_t head;
    atomic_uint_least64_t tail;
    pthread_t thread;
    bool started;
} NetworkWorker;

struct BookStream {
    Book *books;
    char **symbols;
    extent count;
    BookQueue queue;
    NetworkWorker workers[NETWORK_WORKERS];
    extent worker_count;
    atomic_uint_least32_t active_workers;
    atomic_bool stop;
    atomic_bool ready;
    atomic_bool reader_done;
    bool assembler_started;
    pthread_t assembler;
};

static void *network_reader(void *opaque);
static void *stream_assembler(void *opaque);

typedef struct {
    char *data;
    size_t size;
} Response;

static size_t write_response(void *data, size_t size, size_t count, void *opaque) {
    Response *response = opaque;
    size_t bytes = size * count;
    char *next = realloc(response->data, response->size + bytes + 1);
    if (!next)
        return 0;

    response->data = next;
    memcpy(next + response->size, data, bytes);
    response->size += bytes;
    next[response->size] = '\0';
    return bytes;
}

static int find_level(PriceLevel *levels, extent count, real64 price) {
    for (extent i = 0; i < count; ++i) {
        if (levels[i].price == price) {
            return (int)i;
        }
    }

    return -1;
}

static void update_level(PriceLevel *levels, extent *count, real64 price, real64 quantity) {
    int index = find_level(levels, *count, price);

    if (!quantity) {
        if (index >= 0) {
            memmove(&levels[index], &levels[index + 1],
                    (*count - (extent)index - 1) * sizeof(*levels));
            --*count;
        }
        return;
    }

    if (index >= 0) {
        levels[index].quantity = quantity;
        return;
    }

    if (*count < BOOK_CAPACITY)
        levels[(*count)++] = (PriceLevel){price, quantity};
}

static int bid_order(const void *a, const void *b) {
    real64 delta = ((const PriceLevel *)b)->price - ((const PriceLevel *)a)->price;
    return (delta > 0) - (delta < 0);
}

static int ask_order(const void *a, const void *b) {
    real64 delta = ((const PriceLevel *)a)->price - ((const PriceLevel *)b)->price;
    return (delta > 0) - (delta < 0);
}

static bool apply_levels(cJSON *array, PriceLevel *levels, extent *count) {
    cJSON *entry;

    cJSON_ArrayForEach(entry, array) {
        cJSON *price = cJSON_GetArrayItem(entry, 0);
        cJSON *quantity = cJSON_GetArrayItem(entry, 1);

        if (!cJSON_IsString(price) || !cJSON_IsString(quantity))
            return false;

        update_level(levels, count, strtod(price->valuestring, NULL),
                     strtod(quantity->valuestring, NULL));
    }
    return true;
}

static bool load_book_once(Book *book, const char *symbol) {
    char url[256];

    snprintf(url, sizeof(url),
             "https://api.binance.com/api/v3/depth?symbol=%s&limit=%d",
             symbol, INITIAL_DEPTH_LIMIT);
    Response response = {0};
    CURL *curl = curl_easy_init();

    if (!curl)
        return false;

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_response);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);

    CURLcode result = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    bool ok = result == CURLE_OK && status == 200;

    curl_easy_cleanup(curl);
    cJSON *root = ok ? cJSON_Parse(response.data) : NULL;
    free(response.data);
    if (!root)
        return false;

    cJSON *last = cJSON_GetObjectItemCaseSensitive(root, "lastUpdateId");
    cJSON *bids = cJSON_GetObjectItemCaseSensitive(root, "bids");
    cJSON *asks = cJSON_GetObjectItemCaseSensitive(root, "asks");
    book->bid_count = book->ask_count = 0;
    ok = cJSON_IsNumber(last) && cJSON_IsArray(bids) && cJSON_IsArray(asks) &&
         apply_levels(bids, book->bids, &book->bid_count) &&
         apply_levels(asks, book->asks, &book->ask_count);
    if (ok) {
        qsort(book->bids, book->bid_count, sizeof(*book->bids), bid_order);
        qsort(book->asks, book->ask_count, sizeof(*book->asks), ask_order);
    }
    cJSON_Delete(root);
    return ok;
}

static bool load_book(Book *book, const char *symbol) {
    for (int attempt = 0; attempt < BOOK_INIT_RETRIES; ++attempt) {
        if (load_book_once(book, symbol))
            return true;

        struct timespec delay = {
            .tv_sec = 0,
            .tv_nsec = (long)(attempt + 1) * 150000000L
        };
        nanosleep(&delay, NULL);
    }

    return false;
}

typedef struct {
    Book *books;
    char **symbols;
    extent count;
    extent next;
    bool failed;
    pthread_mutex_t lock;
} BookInit;

static void *book_init_worker(void *opaque) {
    BookInit *init = opaque;

    for (;;) {
        pthread_mutex_lock(&init->lock);
        extent index = init->next++;
        pthread_mutex_unlock(&init->lock);

        if (index >= init->count)
            break;

        if (!load_book(&init->books[index], init->symbols[index])) {
            fprintf(stderr, "failed to initialize %s after %d attempts\n",
                    init->symbols[index], BOOK_INIT_RETRIES);
            pthread_mutex_lock(&init->lock);
            init->failed = true;
            pthread_mutex_unlock(&init->lock);
        }
    }

    return NULL;
}

static bool initialize_books(Book *books, char **symbols, extent count) {
    BookInit init = {
        .books = books,
        .symbols = symbols,
        .count = count
    };
    extent workers = count < BOOK_INIT_WORKERS ? count : BOOK_INIT_WORKERS;
    pthread_t threads[BOOK_INIT_WORKERS];

    pthread_mutex_init(&init.lock, NULL);
    fprintf(stderr, "initializing %" PRIu64 " order books with %" PRIu64
            " workers...\n", (uint64)count, (uint64)workers);

    for (extent i = 0; i < workers; ++i)
        pthread_create(&threads[i], NULL, book_init_worker, &init);

    for (extent i = 0; i < workers; ++i)
        pthread_join(threads[i], NULL);

    pthread_mutex_destroy(&init.lock);
    fprintf(stderr, "order-book initialization %s\n",
            init.failed ? "failed" : "complete");
    return !init.failed;
}

static bool receive_message(CURL *ws, char *message, size_t capacity,
                            const atomic_bool *stop) {
    size_t total = 0;
    for (;;) {
        size_t received = 0;
        const struct curl_ws_frame *meta = NULL;
        CURLcode code = curl_ws_recv(ws, message + total, capacity - total - 1,
                                     &received, &meta);
        if (code == CURLE_AGAIN) {
            if (atomic_load_explicit(stop, memory_order_relaxed))
                return false;

            curl_socket_t socket;
            struct pollfd fd;

            if (curl_easy_getinfo(ws, CURLINFO_ACTIVESOCKET, &socket) != CURLE_OK)
                return false;

            fd = (struct pollfd){.fd = socket, .events = POLLIN};
            int ready = poll(&fd, 1, 250);

            if (ready < 0 && errno == EINTR)
                return false;
            if (ready <= 0)
                continue;

            continue;
        }

        if (code != CURLE_OK || total + received >= capacity - 1)
            return false;

        total += received;
        if (!meta || !meta->bytesleft)
            break;
    }
    message[total] = '\0';
    return true;
}

static char **discover_symbols(extent requested, extent *count) {
    Response response = {0};
    CURL *curl = curl_easy_init();

    curl_easy_setopt(curl, CURLOPT_URL,
                     "https://api.binance.com/api/v3/exchangeInfo");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_response);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    if (curl_easy_perform(curl) != CURLE_OK) {
        curl_easy_cleanup(curl);
        free(response.data);
        return NULL;
    }

    curl_easy_cleanup(curl);

    cJSON *root = cJSON_Parse(response.data);
    cJSON *symbols = root
               ? cJSON_GetObjectItemCaseSensitive(root, "symbols")
               : NULL;
    char **selected = calloc(requested, sizeof(*selected));

    free(response.data);

    cJSON *entry;
    cJSON_ArrayForEach(entry, symbols) {
        cJSON *symbol = cJSON_GetObjectItemCaseSensitive(entry, "symbol");
        cJSON *status = cJSON_GetObjectItemCaseSensitive(entry, "status");
        cJSON *quote = cJSON_GetObjectItemCaseSensitive(entry, "quoteAsset");
        cJSON *spot = cJSON_GetObjectItemCaseSensitive(entry,
                                                       "isSpotTradingAllowed");

        if (!cJSON_IsString(symbol) || !cJSON_IsString(status) ||
            !cJSON_IsString(quote) || strcmp(status->valuestring, "TRADING") ||
            strcmp(quote->valuestring, "USDT") ||
            !spot || (spot->type & 0xff) != cJSON_True)
            continue;

        size_t length = strlen(symbol->valuestring);
        selected[*count] = malloc(length + 1);
        memcpy(selected[*count], symbol->valuestring, length + 1);

        if (++*count == requested)
            break;
    }

    cJSON_Delete(root);
    return selected;
}

static bool worker_push(NetworkWorker *worker, BookEvent event) {
    uint64 tail = atomic_load_explicit(&worker->tail, memory_order_relaxed);
    while (tail - atomic_load_explicit(&worker->head, memory_order_acquire) ==
           EVENT_QUEUE_DEPTH) {
        if (atomic_load_explicit(&worker->stream->stop, memory_order_relaxed))
            return false;
        TENSOR_THREAD_YIELD();
    }
    worker->events[tail % EVENT_QUEUE_DEPTH] = event;
    atomic_store_explicit(&worker->tail, tail + 1, memory_order_release);
    return true;
}

static bool worker_pop(NetworkWorker *worker, BookEvent *event) {
    uint64 head = atomic_load_explicit(&worker->head, memory_order_relaxed);
    if (head == atomic_load_explicit(&worker->tail, memory_order_acquire))
        return false;

    *event = worker->events[head % EVENT_QUEUE_DEPTH];
    atomic_store_explicit(&worker->head, head + 1, memory_order_release);
    return true;
}

static extent find_worker_symbol(const NetworkWorker *worker,
                                 const char *name) {
    BookStream *stream = worker->stream;

    for (extent i = worker->first; i < worker->first + worker->count; ++i) {
        if (!strncmp(name, stream->symbols[i], strlen(stream->symbols[i]))) {
            return i;
        }
    }

    return stream->count;
}

static void *network_reader(void *opaque) {
    NetworkWorker *worker = opaque;
    BookStream *stream = worker->stream;
    char message[MESSAGE_CAPACITY];

    while (!atomic_load_explicit(&stream->stop, memory_order_relaxed) &&
           receive_message(worker->ws, message, sizeof(message), &stream->stop)) {
        cJSON *root = cJSON_Parse(message);
        cJSON *name = root ? cJSON_GetObjectItemCaseSensitive(root, "stream") : NULL;
        cJSON *data = root ? cJSON_GetObjectItemCaseSensitive(root, "data") : NULL;
        cJSON *time = data ? cJSON_GetObjectItemCaseSensitive(data, "E") : NULL;
        cJSON *bids = data ? cJSON_GetObjectItemCaseSensitive(data, "b") : NULL;
        cJSON *asks = data ? cJSON_GetObjectItemCaseSensitive(data, "a") : NULL;
        extent instrument = cJSON_IsString(name)
                          ? find_worker_symbol(worker, name->valuestring)
                          : stream->count;
        bool ok = instrument < stream->count && cJSON_IsArray(bids) &&
                  cJSON_IsArray(asks);
        if (ok) {
            Book *book = &stream->books[instrument];
            atomic_fetch_add_explicit(&book->sequence, 1, memory_order_acq_rel);
            ok = apply_levels(bids, book->bids, &book->bid_count) &&
                 apply_levels(asks, book->asks, &book->ask_count);
            if (ok) {
                qsort(book->bids, book->bid_count, sizeof(*book->bids), bid_order);
                qsort(book->asks, book->ask_count, sizeof(*book->asks), ask_order);
            }
            atomic_fetch_add_explicit(&book->sequence, 1, memory_order_release);
            if (ok)
                ok = worker_push(worker, (BookEvent){
                    .instrument = instrument,
                    .timestamp = cJSON_IsNumber(time) ? (uint64)time->valuedouble : 0,
                    .received_ns = time_ns()
                });
        }
        cJSON_Delete(root);
        if (!ok && atomic_load_explicit(&stream->stop, memory_order_relaxed))
            break;
    }

    atomic_fetch_sub_explicit(&stream->active_workers, 1, memory_order_release);
    return NULL;
}

static bool next_event(BookStream *stream, BookEvent *event, extent *cursor) {
    for (;;) {
        for (extent checked = 0; checked < stream->worker_count; ++checked) {
            extent index = (*cursor + checked) % stream->worker_count;
            if (worker_pop(&stream->workers[index], event)) {
                *cursor = (index + 1) % stream->worker_count;
                return true;
            }
        }
        if (!atomic_load_explicit(&stream->active_workers, memory_order_acquire))
            return false;
        if (atomic_load_explicit(&stream->stop, memory_order_relaxed))
            return false;
        TENSOR_THREAD_YIELD();
    }
}

static void copy_book(const Book *book, BookSnapshot *snapshot,
                      extent base) {
    uint32_t before, after;
    do {
        before = atomic_load_explicit(&book->sequence, memory_order_acquire);
        if (before & 1) {
            TENSOR_THREAD_YIELD();
            after = before + 1;
            continue;
        }
        for (extent level = 0; level < LEVELS; ++level) {
            if (level < book->bid_count) {
                snapshot->bid_price[base + level] = book->bids[level].price;
                snapshot->bid_quantity[base + level] = book->bids[level].quantity;
            }
            if (level < book->ask_count) {
                snapshot->ask_price[base + level] = book->asks[level].price;
                snapshot->ask_quantity[base + level] = book->asks[level].quantity;
            }
        }
        after = atomic_load_explicit(&book->sequence, memory_order_acquire);
    } while (before != after || (after & 1));
}

static void *stream_assembler(void *opaque) {
    BookStream *stream = opaque;
    extent cursor = 0;
    extent warmed = 0;

    for (;;) {
        BookSnapshot *snapshot = calloc(1, sizeof(*snapshot));
        if (!snapshot)
            break;
        bool complete = true;
        for (extent batch = 0; batch < BATCH_SIZE; ++batch) {
            BookEvent event;
            if (!next_event(stream, &event, &cursor)) {
                complete = false;
                break;
            }
            if (!batch)
                snapshot->received_ns = event.received_ns;
            for (extent instrument = 0; instrument < stream->count; ++instrument)
                copy_book(&stream->books[instrument], snapshot,
                          (batch * stream->count + instrument) * LEVELS);
            snapshot->timestamp[batch * stream->count + event.instrument] =
                event.timestamp;
        }
        if (!complete) {
            free(snapshot);
            break;
        }
        book_queue_push(&stream->queue, snapshot);
        free(snapshot);
        if (warmed < WARM_UP && ++warmed == WARM_UP)
            atomic_store_explicit(&stream->ready, true, memory_order_release);
    }

    atomic_store_explicit(&stream->reader_done, true, memory_order_release);
    book_queue_close(&stream->queue);
    return NULL;
}

static bool connect_worker(NetworkWorker *worker) {
    BookStream *stream = worker->stream;
    size_t capacity = 64 + worker->count * 40;
    char *url = calloc(capacity, 1);
    if (!url)
        return false;
    strcpy(url, "wss://stream.binance.com:9443/stream?streams=");
    size_t used = strlen(url);
    for (extent offset = 0; offset < worker->count; ++offset) {
        extent index = worker->first + offset;
        used += snprintf(url + used, capacity - used, "%s%s@depth@100ms",
                         offset ? "/" : "", stream->symbols[index]);
    }
    worker->ws = curl_easy_init();
    if (worker->ws) {
        curl_easy_setopt(worker->ws, CURLOPT_URL, url);
        curl_easy_setopt(worker->ws, CURLOPT_CONNECT_ONLY, 2L);
    }
    bool connected = worker->ws && curl_easy_perform(worker->ws) == CURLE_OK;
    free(url);
    return connected;
}

BookStream *order_book_stream_open(extent count) {
    if (!count || count > NUM_INSTRUMENTS ||
        curl_global_init(CURL_GLOBAL_DEFAULT))
        return NULL;
    extent discovered = 0;
    char **symbols = NULL;
    for (int attempt = 0; attempt < BOOK_INIT_RETRIES && !symbols; ++attempt) {
        symbols = discover_symbols(count, &discovered);
        if (!symbols) {
            struct timespec delay = {.tv_sec = 0,
                .tv_nsec = (long)(200000000ULL * (attempt + 1))};
            nanosleep(&delay, NULL);
        }
    }
    if (!symbols || !discovered) {
        free(symbols);
        curl_global_cleanup();
        return NULL;
    }
    if (discovered < count) {
        fprintf(stderr, "using %" PRIu64 " discovered active symbols (requested %" PRIu64 ")\n",
                (uint64)discovered, (uint64)count);
        count = discovered;
    }

    BookStream *stream = calloc(1, sizeof(*stream));
    if (!stream) {
        curl_global_cleanup();
        return NULL;
    }
    stream->books = calloc(count, sizeof(*stream->books));
    stream->symbols = calloc(count, sizeof(*stream->symbols));
    stream->count = count;
    if (!stream->books || !stream->symbols || !book_queue_open(&stream->queue)) {
        order_book_stream_close(stream);
        return NULL;
    }
    for (extent i = 0; i < count; ++i) {
        size_t length = strlen(symbols[i]);
        stream->symbols[i] = malloc(length + 1);
        for (size_t j = 0; j < length; ++j)
            stream->symbols[i][j] = (char)tolower((unsigned char)symbols[i][j]);
        stream->symbols[i][length] = '\0';
    }
    if (!initialize_books(stream->books, symbols, count)) {
        for (extent i = 0; i < count; ++i)
            free(symbols[i]);
        free(symbols);
        order_book_stream_close(stream);
        return NULL;
    }
    for (extent i = 0; i < count; ++i)
        free(symbols[i]);
    free(symbols);

    stream->worker_count = count < NETWORK_WORKERS ? count : NETWORK_WORKERS;
    extent first = 0;
    for (extent i = 0; i < stream->worker_count; ++i) {
        NetworkWorker *worker = &stream->workers[i];
        worker->stream = stream;
        worker->first = first;
        worker->count = (count - first) / (stream->worker_count - i);
        first += worker->count;
        if (!connect_worker(worker)) {
            order_book_stream_close(stream);
            return NULL;
        }
    }
    atomic_store(&stream->active_workers, stream->worker_count);
    for (extent i = 0; i < stream->worker_count; ++i) {
        NetworkWorker *worker = &stream->workers[i];
        if (pthread_create(&worker->thread, NULL, network_reader, worker)) {
            atomic_store(&stream->stop, true);
            order_book_stream_close(stream);
            return NULL;
        }
        worker->started = true;
    }
    if (pthread_create(&stream->assembler, NULL, stream_assembler, stream)) {
        atomic_store(&stream->stop, true);
        order_book_stream_close(stream);
        return NULL;
    }
    stream->assembler_started = true;
    fprintf(stderr, "warming up %u order-book batches over %" PRIu64
            " WebSocket workers...\n", WARM_UP, (uint64)stream->worker_count);
    while (!atomic_load_explicit(&stream->ready, memory_order_acquire) &&
           !atomic_load_explicit(&stream->reader_done, memory_order_acquire)) {
        struct timespec delay = {.tv_sec = 0, .tv_nsec = 1000000};
        nanosleep(&delay, NULL);
    }
    if (!atomic_load_explicit(&stream->ready, memory_order_acquire)) {
        order_book_stream_close(stream);
        return NULL;
    }
    fprintf(stderr, "order-book stream ready\n");
    return stream;
}

BookSnapshot *order_book_stream_next(BookStream *stream) {
    return book_queue_pop(&stream->queue);
}

void order_book_stream_close(BookStream *stream) {
    if (!stream)
        return;

    atomic_store(&stream->stop, true);

    for (extent i = 0; i < stream->worker_count; ++i) {
        if (stream->workers[i].started) {
            pthread_join(stream->workers[i].thread, NULL);
        }
    }

    if (stream->assembler_started) {
        pthread_join(stream->assembler, NULL);
    }

    book_queue_close(&stream->queue);
    book_queue_destroy(&stream->queue);

    for (extent i = 0; i < stream->worker_count; ++i) {
        if (stream->workers[i].ws) {
            curl_easy_cleanup(stream->workers[i].ws);
        }
    }

    if (stream->symbols) {
        for (extent i = 0; i < stream->count; ++i)
            free(stream->symbols[i]);
    }

    free(stream->symbols);
    free(stream->books);
    free(stream);
    curl_global_cleanup();
}
