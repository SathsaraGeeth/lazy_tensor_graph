/*
 * user/src/tensor/src/memory/cpu/memory.c
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
 * 1. allocator is bump allocator
 * 2. chunks are aligned to pages (CPU specific details)
 * 3. blocks are aligned to cache lines (CPU specific details)
 * 4. there can be multiple chunks in the pool, as graph
 *    evaluation is demand based
 */

#include "memory.h"
#include "dev_tools/profiler/trace.h"

#include <stdlib.h>
#include <string.h>

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#define LINE_SIZE 64
#endif

typedef struct pool_chunk {
    uint8             *data;
    extent             size;
    extent             used;
    struct pool_chunk *next;
} pool_chunk;

struct mem_pool {
    pool_chunk  *first;
    pool_chunk  *last;
    extent      planned;
    extent      capacity;
    extent      allocated;
    void (*cleanup)(void);
};

static mem_pool *active_pool;

static uint64_t profile_begin(const char *name, uint32_t operation,
                              extent bytes, const void *scope) {
    uint64_t id = (uint64_t)(uintptr_t)scope;
    tensor_profile_record("LIB_BEGIN", name, id, 0, operation,
                          DTYPE_UNKNOWN, bytes, 0);
    return id;
}

static void profile_end(const char *name, uint64_t id, uint32_t operation,
                        extent bytes) {
    tensor_profile_record("LIB_END", name, id, 0, operation,
                          DTYPE_UNKNOWN, bytes, 0);
}

static extent align_up(extent size, extent alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

static boolean add_chunk(mem_pool *pool, extent minimum_size) {
    pool_chunk *chunk = malloc(sizeof(*chunk));
    if (!chunk) return true;

    chunk->size = align_up(minimum_size, PAGE_SIZE);
    chunk->data = aligned_alloc(PAGE_SIZE, chunk->size);
    if (!chunk->data) {
        free(chunk);
        return true;
    }

    chunk->used = 0;
    chunk->next = NULL;
    if (pool->last) pool->last->next = chunk;
    else pool->first = chunk;
    pool->last = chunk;
    pool->capacity += chunk->size;
    return false;
}

static mem_block *alloc_from_pool(mem_pool *pool, extent size) {
    extent aligned_size = align_up(size, LINE_SIZE);

    if (!pool->last) {
        extent initial_size = pool->planned > aligned_size ? pool->planned : aligned_size;
        if (add_chunk(pool, initial_size)) return NULL;
    }

    if (pool->last->size - pool->last->used < aligned_size) {
        extent remaining_plan = pool->planned > pool->capacity ? pool->planned - pool->capacity : 0;
        extent chunk_size = remaining_plan > aligned_size ? remaining_plan : aligned_size;
        if (add_chunk(pool, chunk_size)) return NULL;
    }

    mem_block *block = malloc(sizeof(*block));
    if (!block) return NULL;

    block->ptr        = (dptr *)(pool->last->data + pool->last->used);
    block->size       = size;
    block->is_owner   = true;
    block->is_pooled  = true;
    pool->last->used += aligned_size;
    pool->allocated  += aligned_size;

    tensor_profile_record("POOL_ALLOC", "lazy_data",
                          (uint64_t)(uintptr_t)block, pool->planned, 0,
                          DTYPE_UNKNOWN, pool->capacity, pool->allocated);

    return block;
}

mem_pool *mem_pool_init(void) {
    if (!active_pool) active_pool = calloc(1, sizeof(*active_pool));
    return active_pool;
}

void mem_pool_plan(mem_pool *pool, extent size) {
    if (pool) {
        pool->planned += align_up(size, LINE_SIZE);
        tensor_profile_record("POOL_PLAN", "lazy_data", 0, 0, 0,
                              DTYPE_UNKNOWN, size, pool->planned);
    }
}

void mem_pool_set_cleanup(mem_pool *pool, void (*cleanup)(void)) {
    if (pool) pool->cleanup = cleanup;
}

mem_block *mem_alloc(extent size, boolean is_pooled, mem_pool *pool) {
    uint8 trace_scope;
    uint64_t trace_id = profile_begin("mem_alloc", is_pooled, size,
                                      &trace_scope);
    mem_block *block = NULL;

    if (!size) goto done;

    if (is_pooled) {
        if (!pool) pool = active_pool;
        if (pool) block = alloc_from_pool(pool, size);
        goto done;
    }

    block = malloc(sizeof(*block));
    if (!block) goto done;

    extent aligned_size = align_up(size, PAGE_SIZE);
    block->ptr = aligned_alloc(PAGE_SIZE, aligned_size);
    if (!block->ptr) {
        free(block);
        block = NULL;
        goto done;
    }

    block->size      = size;
    block->is_owner  = true;
    block->is_pooled = false;

done:
    profile_end("mem_alloc", trace_id, is_pooled, size);
    return block;
}

mem_block *mem_view_from(extent size, const dptr *data) {
    if (!size || !data) return NULL;

    mem_block *block = malloc(sizeof(*block));
    if (!block) return NULL;

    *block = (mem_block){
        .ptr = (dptr *)data,
        .size = size,
        .is_owner = false,
        .is_pooled = false
    };
    return block;
}

boolean mem_free(mem_block *block) {
    extent bytes = block ? block->size : 0;
    uint32_t storage_kind = !block ? 0 : block->is_pooled ? 1u
                                  : block->is_owner ? 0u : 2u;
    uint8 trace_scope;
    uint64_t trace_id = profile_begin("mem_free", storage_kind, bytes,
                                      &trace_scope);
    boolean failed = !block;
    if (!failed) {
        if (block->is_owner && !block->is_pooled) {
            free(block->ptr);
            tensor_profile_record("HEAP_RELEASE", "owned_storage", 0, 0, 0,
                                  DTYPE_UNKNOWN, bytes, 0);
        }
        free(block);
    }
    profile_end("mem_free", trace_id, storage_kind, bytes);
    return failed;
}

boolean mem_copy(mem_block *dst, const mem_block *src, extent size) {
    uint8 trace_scope;
    uint64_t trace_id = profile_begin("mem_copy", 0, size, &trace_scope);
    boolean failed = !dst || !src || !dst->ptr ||
                     !src->ptr || size > dst->size || size > src->size;

    if (!failed) memcpy(dst->ptr, src->ptr, size);

    profile_end("mem_copy", trace_id, 0, size);
    return failed;
}

boolean mem_view(mem_block *dst, const dptr *data) {
    boolean failed = !dst || !data || !dst->size;
    if (!failed) {
        if (dst->is_owner && !dst->is_pooled) free(dst->ptr);
        dst->ptr = (dptr *)data;
        dst->is_owner = false;
        dst->is_pooled = false;
    }

    return failed;
}

boolean mem_view_copy(mem_block *dst, const dptr *data) {
    boolean failed = !dst || !data || !dst->ptr || !dst->size;
    if (!failed) memcpy(dst->ptr, data, dst->size);

    return failed;
}

dptr *mem_view_to(mem_block *block) {
    return block ? block->ptr : NULL;
}

void mem_pool_shutdown(void) {
    mem_pool *pool = active_pool;

    if (pool) {
        tensor_profile_record("POOL_SHUTDOWN", "lazy_session", 0, 0, 0,
                              DTYPE_UNKNOWN, pool->capacity, 0);
        void (*cleanup)(void) = pool->cleanup;
        pool->cleanup = NULL;
        if (cleanup) cleanup();

        pool_chunk *chunk = pool->first;
        while (chunk) {
            pool_chunk *next = chunk->next;
            free(chunk->data);
            free(chunk);
            chunk = next;
        }

        free(pool);
        active_pool = NULL;
    }

}
