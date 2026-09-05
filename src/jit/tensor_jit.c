/*
 * user/src/tensor/src/jit/tensor_jit.c
 *
 * Copyright (C) 2026 Sathsara Geeth
 *
 */

/*
 * Version 1.0
 *
 * Version History
 *
 * Version | Description
 * --------+-----------------------------------------
 * 1.0     | Initial implementation
 */

/*
 * Comments:
 * 1. JIT cache is thread safe, so the environment/OS
 *    supports some sort of lock/atomics/condition
 *    - written using pthread_mutex, pthread_cond
 * 2. This file is platform dependent
 *    - TODO: make it platform independent
 * 3. The key is serialized
 *    - so JIT cache can be used between processes
 *    - it looks like this in the file
 *     +----------------+
 *     | magic          |
 *     | version        |
 *     | #entry         |
 *     +----------------+
 *     | key A    bits  |
 *     +----------------+
 *     | key B    bits  |
 *     +----------------+
 * 4. Each backend should implement these
 *    tensor_device_jit_compile(
 *    device dev, const jit_cache_key_t *key);
 *    tensor_device_jit_compatibility_hash(device dev)
 * 5. graph.c or the user needs to call tensor_jit_cache_init()
 *    before calling tensor_get_jit_cache(), or use the persistent cache
 * 6. Compatibility field does not carry redundant
 *    information alongside the device embedded in the key.
 *    It identifies whether the cached runtime is compatible with the
 *    current device runtime settings. For example, a runtime compiled
 *    with ffast_math enabled is not compatible with one that suppresses it.
 */

#include "tensor_jit.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64 tensor_probe_time_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return (uint64)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;
}

/*
 * How to compile a kernel for a given key?
 * backend specific implementation private to each
 */

extern jit_ker_t tensor_device_jit_compile            (device dev, const jit_cache_key_t *key,
                                                       uint8 **object, extent *object_size);
extern jit_ker_t tensor_device_jit_load               (device dev, const jit_cache_key_t *key,
                                                       const uint8 *object, extent object_size);
extern uint64    tensor_device_jit_compatibility_hash (device dev);
extern boolean   tensor_device_jit_supported          (device dev, const jit_cache_key_t *key);

typedef struct cache_entry {
    jit_cache_key_t      key;
    jit_ker_t            kernel;
    uint8               *object;
    extent               object_size;
    struct cache_entry  *next;
    struct cache_entry  *hash_next;
    uint64               hash;
    jit_cache_state_t    state;
    pthread_cond_t       ready;
} cache_entry_t;


/*
 * We keep a little bucket
 * so it keeps similar hashes (as a linked list)
 * so it can look up in there rather than
 * the full linked list of all entries
 */
#define JIT_CACHE_BUCKETS 64
struct jit_cache {
    cache_entry_t    *entries;
    cache_entry_t    *buckets[JIT_CACHE_BUCKETS];
    pthread_mutex_t   lock;
    pthread_cond_t    idle;
    extent            compiling;
};

/* 
 * Cache main
 */

static void tensor_sig_clear(jit_tensor_sig_t *sig) {
    if (!sig) return;
    free(sig->shape);
    sig->shape = NULL;
    sig->rank = 0;
}

static boolean tensor_sig_create(jit_tensor_sig_t *sig,
                                 const tensor *tensor,
                                 extent **shape_storage) {

    if (!sig || !tensor) return false;

    sig->dtype = tensor->dtype;
    sig->rank = tensor->rank;
    sig->shape = NULL;

    if (!sig->rank)     return true;
    if (!tensor->shape) return false;

    sig->shape = *shape_storage;
    memcpy(sig->shape, tensor->shape, sig->rank * sizeof(*sig->shape));
    *shape_storage += sig->rank;

    return true;
}


static boolean tensor_sig_equal(const jit_tensor_sig_t *a,
                                const jit_tensor_sig_t *b) {

    if (a->dtype != b->dtype)
        return false;

    if (a->rank != b->rank)
        return false;

    if (!a->rank)
        return true;

    return !memcmp(a->shape, b->shape, a->rank * sizeof(extent));
}


static void key_clear(jit_cache_key_t *key) {
    if (!key) return;

    if (key->storage) {
        free(key->storage);
    } else {
        tensor_sig_clear(&key->output);
        for (extent i = 0; i < key->num_inputs; ++i)
            tensor_sig_clear(&key->inputs[i]);
        free(key->inputs);
    }

    memset(key, 0, sizeof(*key));
}


jit_cache_key_t tensor_jit_key_create(uint32 op, const tensor *output, const tensor *const *inputs,
                                      extent num_inputs, device dev) {
    jit_cache_key_t key = {0};

    if (!output) return key;

    if (num_inputs && !inputs) return key;

    key.op = op;
    key.dev = dev;
    key.compatibility_hash = tensor_device_jit_compatibility_hash(dev);
    key.num_inputs = num_inputs;

    extent shape_count = output->rank;
    for (extent i = 0; i < num_inputs; ++i) {
        if (!inputs[i] || shape_count > (extent)-1 - inputs[i]->rank) goto fail;
        shape_count += inputs[i]->rank;
    }
    if (num_inputs > (extent)-1 / sizeof(*key.inputs)) goto fail;
    if (shape_count > ((extent)-1 - num_inputs * sizeof(*key.inputs)) / sizeof(extent)) goto fail;
    extent storage_size = num_inputs * sizeof(*key.inputs) + shape_count * sizeof(extent);
    key.storage = malloc(storage_size ? storage_size : 1);
    if (!key.storage) goto fail;
    key.inputs = num_inputs ? key.storage : NULL;
    extent *shape_storage = (extent *)((uint8 *)key.storage + num_inputs * sizeof(*key.inputs));

    if (!tensor_sig_create(&key.output, output, &shape_storage)) goto fail;

    for (extent i = 0; i < num_inputs; ++i) {

        if (!tensor_sig_create(&key.inputs[i], inputs[i], &shape_storage)) goto fail;
    }

    return key;

fail:
    key_clear(&key);
    return (jit_cache_key_t){0};
}

static boolean key_equal(const jit_cache_key_t *a, 
                         const jit_cache_key_t *b) {
    if (a->op != b->op)
        return false;

    if (memcmp(&a->dev, &b->dev, sizeof(device)))
        return false;

    if (a->num_inputs != b->num_inputs)
        return false;

    if (a->compatibility_hash != b->compatibility_hash)
        return false;

    if (!tensor_sig_equal(&a->output, &b->output))
        return false;

    for (extent i = 0; i < a->num_inputs; ++i) {

        if (!tensor_sig_equal(&a->inputs[i], &b->inputs[i]))
            return false;
    }

    return true;
}

static uint64 key_hash_bytes(uint64 hash, const void *data, size_t size) {
    const uint8 *bytes = data;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64 key_hash_signature(uint64 hash, const jit_tensor_sig_t *sig) {
    hash = key_hash_bytes(hash, &sig->dtype, sizeof(sig->dtype));
    hash = key_hash_bytes(hash, &sig->rank, sizeof(sig->rank));
    return key_hash_bytes(hash, sig->shape, sig->rank * sizeof(extent));
}

static uint64 key_hash(const jit_cache_key_t *key) {
    uint64 hash = UINT64_C(1469598103934665603);
    hash = key_hash_bytes(hash, &key->op, sizeof(key->op));
    hash = key_hash_bytes(hash, &key->dev, sizeof(key->dev));
    hash = key_hash_bytes(hash, &key->compatibility_hash, sizeof(key->compatibility_hash));
    hash = key_hash_signature(hash, &key->output);
    hash = key_hash_bytes(hash, &key->num_inputs, sizeof(key->num_inputs));
    for (extent i = 0; i < key->num_inputs; ++i)
        hash = key_hash_signature(hash, &key->inputs[i]);
    return hash;
}

static cache_entry_t *entry_find(jit_cache_t cache,
                                 const jit_cache_key_t *key) {
    uint64 hash = key_hash(key);
    for (cache_entry_t *entry = cache->buckets[hash % JIT_CACHE_BUCKETS];
         entry; entry = entry->hash_next) {

        if (entry->hash == hash && key_equal(&entry->key, key))
            return entry;
    }

    return NULL;
}


static cache_entry_t *entry_create(jit_cache_key_t key) {
    cache_entry_t *entry = calloc(1, sizeof(*entry));

    if (!entry) return NULL;

    if (pthread_cond_init(&entry->ready, NULL)) {
        free(entry);
        return NULL;
    }

    entry->key   = key;
    entry->hash  = key_hash(&key);
    entry->state = CACHE_COMPILING;

    return entry;
}


static void entry_destroy(cache_entry_t *entry) {
    if (!entry) return;

    pthread_cond_destroy(&entry->ready);

    key_clear(&entry->key);
    free(entry->object);

    free(entry);
}

static void entry_insert(jit_cache_t cache, cache_entry_t *entry) {
    extent bucket = entry->hash % JIT_CACHE_BUCKETS;
    entry->hash_next = cache->buckets[bucket];
    cache->buckets[bucket] = entry;
    entry->next = cache->entries;
    cache->entries = entry;
}

jit_cache_t tensor_jit_cache_create(void) {
    jit_cache_t cache = calloc(1, sizeof(*cache));

    if (!cache) return NULL;

    if (pthread_mutex_init(&cache->lock, NULL)) {
        free(cache);
        return NULL;
    }

    if (pthread_cond_init(&cache->idle, NULL)) {
        pthread_mutex_destroy(&cache->lock);
        free(cache);
        return NULL;
    }

    return cache;
}


void tensor_jit_cache_destroy(jit_cache_t cache) {
    if (!cache) return;

    pthread_mutex_lock(&cache->lock);

    while (cache->compiling)
        pthread_cond_wait(&cache->idle, &cache->lock);

    cache_entry_t *entry = cache->entries;

    cache->entries = NULL;

    pthread_mutex_unlock(&cache->lock);

    while (entry) {
        cache_entry_t *next = entry->next;

        entry_destroy(entry);

        entry = next;
    }

    pthread_cond_destroy(&cache->idle);
    pthread_mutex_destroy(&cache->lock);

    free(cache);
}

jit_ker_t tensor_jit_get(jit_cache_key_t key, jit_cache_t cache) {
    if (!cache) {
        key_clear(&key);
        return NULL;
    }

    uint64 probe_started = tensor_probe_time_ns();
    pthread_mutex_lock(&cache->lock);

    cache_entry_t *entry = entry_find(cache, &key);
    uint64 probe_found = tensor_probe_time_ns();

    if (entry) {
        key_clear(&key);

        while (entry->state == CACHE_COMPILING)
            pthread_cond_wait(&entry->ready, &cache->lock);

        jit_ker_t kernel = entry->state == CACHE_READY ? entry->kernel : NULL;

        pthread_mutex_unlock(&cache->lock);

        if (getenv("TENSOR_TIMING_PROBES"))
            fprintf(stderr, "[tensor probe] jit_cache hit_lookup=%.3f us wait_return=%.3f us\n",
                    (probe_found - probe_started) / 1000.0,
                    (tensor_probe_time_ns() - probe_found) / 1000.0);

        return kernel;
    }


    entry = entry_create(key);

    if (!entry) {
        pthread_mutex_unlock(&cache->lock);
        key_clear(&key);
        return NULL;
    }

    entry_insert(cache, entry);

    ++cache->compiling;

    pthread_mutex_unlock(&cache->lock);

    jit_ker_t kernel = tensor_device_jit_compile(entry->key.dev, &entry->key,
                                                 &entry->object, &entry->object_size);

    pthread_mutex_lock(&cache->lock);

    entry->kernel = kernel;

    entry->state = kernel ? CACHE_READY : CACHE_FAILED;

    --cache->compiling;

    pthread_cond_broadcast(
        &entry->ready);

    if (!cache->compiling)
        pthread_cond_broadcast(&cache->idle);

    pthread_mutex_unlock(&cache->lock);

    return kernel;
}


boolean tensor_jit_prepare(jit_cache_key_t key, jit_cache_t cache) {
    if (key.compatibility_hash != tensor_device_jit_compatibility_hash(key.dev)) {
        key_clear(&key);
        return false;
    }
    if (!tensor_device_jit_supported(key.dev, &key)) {
        key_clear(&key);
        return false;
    }
    return tensor_jit_get(key, cache) == NULL;
}

/* 
 * Persistent cache 
 */

#if PRESISTENT_JIT_CACHE

#define JIT_CACHE_MAGIC   UINT64_C(0x544A495443414348)
#define JIT_CACHE_VERSION UINT32_C(2)


static const char *cache_path(const char *path) {
    return path ? path : PRESISTENT_JIT_CACHE_PATH "cache.bin";
}

static boolean file_write(FILE *file, const void *data, size_t size) {
    return fwrite(data, 1, size, file) == size;
}

static boolean file_read(FILE *file, void *data, size_t size) {
    return fread(data, 1, size, file) == size;
}

static boolean tensor_sig_store(FILE *file, const jit_tensor_sig_t *sig) {
    if (!file_write(file, &sig->dtype, sizeof(sig->dtype)))
        return false;

    if (!file_write(file, &sig->rank, sizeof(sig->rank)))
        return false;

    if (!sig->rank)
        return true;

    return file_write(file, sig->shape, sig->rank * sizeof(extent));
}

static boolean tensor_sig_load(FILE *file, jit_tensor_sig_t *sig) {
    memset(sig, 0, sizeof(*sig));

    if (!file_read(file, &sig->dtype, sizeof(sig->dtype)))
        return false;

    if (!file_read(file, &sig->rank, sizeof(sig->rank)))
        return false;

    if (!sig->rank)
        return true;

    sig->shape = malloc(sig->rank * sizeof(extent));

    if (!sig->shape)
        return false;

    if (!file_read(file, sig->shape, sig->rank * sizeof(extent))) {
        tensor_sig_clear(sig);
        return false;
    }

    return true;
}

static boolean key_store(FILE *file, const jit_cache_key_t *key) {
    if (!file_write(file, &key->op, sizeof(key->op)))
        return false;

    if (!file_write(file, &key->dev, sizeof(key->dev)))
        return false;

    if (!file_write(file, &key->compatibility_hash, sizeof(key->compatibility_hash)))
        return false;

    if (!tensor_sig_store(file, &key->output))
        return false;

    if (!file_write(file, &key->num_inputs, sizeof(key->num_inputs)))
        return false;

    for (extent i = 0; i < key->num_inputs; ++i) {
        if (!tensor_sig_store(file, &key->inputs[i]))
            return false;
    }

    return true;
}


static boolean key_load(FILE *file, jit_cache_key_t *key) {
    memset(key, 0, sizeof(*key));

    if (!file_read(file, &key->op, sizeof(key->op)))
        goto fail;

    if (!file_read(file, &key->dev, sizeof(key->dev)))
        goto fail;

    if (!file_read(file, &key->compatibility_hash, sizeof(key->compatibility_hash)))
        goto fail;

    if (!tensor_sig_load(file, &key->output))
        goto fail;

    if (!file_read(file, &key->num_inputs, sizeof(key->num_inputs)))
        goto fail;

    if (!key->num_inputs)
        return true;

    key->inputs = calloc(key->num_inputs, sizeof(*key->inputs));

    if (!key->inputs)
        goto fail;

    for (extent i = 0; i < key->num_inputs; ++i) {
        if (!tensor_sig_load(file, &key->inputs[i]))
            goto fail;
    }

    return true;

fail:
    key_clear(key);
    return false;
}


static extent cache_ready_count(jit_cache_t cache) {
    extent count = 0;

    for (cache_entry_t *entry = cache->entries; entry; entry = entry->next) {
        if (entry->state == CACHE_READY && entry->object_size)
            ++count;
    }

    return count;
}


static boolean header_store(FILE *file, extent count) {
    uint64 magic = JIT_CACHE_MAGIC;
    uint32 version = JIT_CACHE_VERSION;

    return file_write(file, &magic, sizeof(magic))     &&
           file_write(file, &version, sizeof(version)) &&
           file_write(file, &count, sizeof(count));
}


static boolean header_load(FILE *file, extent *count) {
    uint64 magic;
    uint32 version;

    if (!file_read(file, &magic, sizeof(magic)))
        return false;

    if (magic != JIT_CACHE_MAGIC)
        return false;

    if (!file_read(file, &version, sizeof(version)))
        return false;

    if (version != JIT_CACHE_VERSION)
        return false;

    return file_read(file, count, sizeof(*count));
}

#endif


boolean tensor_jit_cache_store(jit_cache_t cache, const char *path) {
#if !PRESISTENT_JIT_CACHE
    (void)cache;
    (void)path;
    return false;
#else

    if (!cache) return false;

    FILE *file = fopen(cache_path(path), "wb");

    if (!file) return false;


    pthread_mutex_lock(&cache->lock);

    while (cache->compiling)
        pthread_cond_wait(&cache->idle, &cache->lock);


    extent count = cache_ready_count(cache);

    boolean ok = header_store(file, count);


    for (cache_entry_t *entry = cache->entries; ok && entry; entry = entry->next) {
        if (entry->state != CACHE_READY || !entry->object_size)
            continue;

        ok = key_store(file, &entry->key) &&
             file_write(file, &entry->object_size, sizeof(entry->object_size)) &&
             file_write(file, entry->object, entry->object_size);
    }

    pthread_mutex_unlock(&cache->lock);

    if (fclose(file))
        ok = false;

    return ok;

#endif
}


boolean tensor_jit_cache_load(const char *path, jit_cache_t cache) {
#if !PRESISTENT_JIT_CACHE
    (void)path;
    (void)cache;
    return false;
#else

    if (!cache) return false;

    FILE *file = fopen(cache_path(path), "rb");

    if (!file)
        return false;


    extent count;

    if (!header_load(file, &count)) {
        fclose(file);
        return false;
    }


    boolean ok = true;

    for (extent i = 0; i < count; ++i) {

        jit_cache_key_t key;

        if (!key_load(file, &key)) {
            ok = false;
            break;
        }

        cache_entry_t *entry = entry_create(key);

        if (!entry || entry->key.compatibility_hash != tensor_device_jit_compatibility_hash(entry->key.dev) ||
            !file_read(file, &entry->object_size, sizeof(entry->object_size)) ||
            !entry->object_size || !(entry->object = malloc(entry->object_size)) ||
            !file_read(file, entry->object, entry->object_size) ||
            !(entry->kernel = tensor_device_jit_load(entry->key.dev, &entry->key,
                                                     entry->object, entry->object_size))) {
            entry_destroy(entry);
            ok = false;
            break;
        }

        entry->state = CACHE_READY;
        pthread_mutex_lock(&cache->lock);
        entry_insert(cache, entry);
        pthread_mutex_unlock(&cache->lock);
    }

    if (fclose(file))
        ok = false;

    return ok;

#endif
}
