/*
 * user/src/tensor/src/memory/cuda/memory.c
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
 * 1. CUDA specific memory implementation
 */

#include "memory.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#define LINE_SIZE 32
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
static CUcontext cuda_context;

static boolean cuda_failed(CUresult result, const char *operation) {
    if (result == CUDA_SUCCESS) return false;
    const char *name = "CUDA_ERROR_UNKNOWN";
    const char *description = "unknown CUDA driver error";
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &description);
    fprintf(stderr, "tensor CUDA: %s failed: %s (%s)\n",
            operation, name ? name : "CUDA_ERROR_UNKNOWN",
            description ? description : "unknown CUDA driver error");
    return true;
}

static boolean cuda_ready(void) {
    if (cuda_context) return false;
    CUdevice device;
    if (cuda_failed(cuInit(0), "cuInit") ||
        cuda_failed(cuDeviceGet(&device, 0), "cuDeviceGet"))
        return true;
    CUcontext context = NULL;
    CUresult result = cuCtxCreate(&context, NULL, 0, device);
    if (cuda_failed(result, "cuCtxCreate")) {
        if (context) (void)cuCtxDestroy(context);
        return true;
    }
    cuda_context = context;
    return false;
}

static extent align_up(extent size, extent alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

static boolean add_chunk(mem_pool *pool, extent minimum_size) {
    pool_chunk *chunk = malloc(sizeof(*chunk));
    if (!chunk) return true;

    chunk->size = align_up(minimum_size, PAGE_SIZE);
    CUdeviceptr pointer = 0;
    if (cuda_ready() ||
        cuda_failed(cuMemAllocManaged(&pointer, chunk->size,
                                     CU_MEM_ATTACH_GLOBAL),
                    "cuMemAllocManaged(pool)")) pointer = 0;
    chunk->data = (uint8 *)(uintptr_t)pointer;
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

    return block;
}

mem_pool *mem_pool_init(void) {
    if (!active_pool) active_pool = calloc(1, sizeof(*active_pool));
    return active_pool;
}

void mem_pool_plan(mem_pool *pool, extent size) {
    if (pool) pool->planned += align_up(size, LINE_SIZE);
}

void mem_pool_set_cleanup(mem_pool *pool, void (*cleanup)(void)) {
    if (pool) pool->cleanup = cleanup;
}

mem_block *mem_alloc(extent size, boolean is_pooled, mem_pool *pool) {
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
    CUdeviceptr pointer = 0;
    if (cuda_ready() ||
        cuda_failed(cuMemAllocManaged(&pointer, aligned_size,
                                     CU_MEM_ATTACH_GLOBAL),
                    "cuMemAllocManaged")) pointer = 0;
    block->ptr = (dptr *)(uintptr_t)pointer;
    if (!block->ptr) {
        free(block);
        block = NULL;
        goto done;
    }

    block->size      = size;
    block->is_owner  = true;
    block->is_pooled = false;

done:
    return block;
}

mem_block *mem_view_from(extent size, const dptr *data) {
    mem_block *block = mem_alloc(size, false, NULL);
    if (!block) return NULL;
    if (!mem_view_copy(block, data)) return block;
    mem_free(block);
    return NULL;
}

boolean mem_free(mem_block *block) {
    boolean failed = !block;
    if (!failed) {
        if (block->is_owner && !block->is_pooled)
            cuMemFree((CUdeviceptr)(uintptr_t)block->ptr);
        free(block);
    }
    return failed;
}

boolean mem_copy(mem_block *dst, const mem_block *src, extent size) {
    boolean failed = !dst || !src || !dst->ptr ||
                     !src->ptr || size > dst->size || size > src->size;

    if (!failed && (cuda_ready() ||
        cuda_failed(cuMemcpy((CUdeviceptr)(uintptr_t)dst->ptr,
                             (CUdeviceptr)(uintptr_t)src->ptr, size),
                    "cuMemcpy"))) failed = true;

    return failed;
}

boolean mem_view(mem_block *dst, const dptr *data) {
    extent bytes = dst ? dst->size : 0;

    boolean failed = !dst || !data || !dst->size;
    if (!failed && (cuda_ready() ||
        cuda_failed(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst->ptr, data,
                                 bytes),
                    "cuMemcpyHtoD"))) failed = true;

    return failed;
}

boolean mem_view_copy(mem_block *dst, const dptr *data) {
    boolean failed = !dst || !data || !dst->ptr || !dst->size;
    if (!failed && (cuda_ready() ||
        cuda_failed(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst->ptr, data,
                                 dst->size),
                    "cuMemcpyHtoD(copy)"))) failed = true;

    return failed;
}

dptr *mem_view_to(mem_block *block) {
    if (!block || cuda_ready() || cuCtxSynchronize() != CUDA_SUCCESS) return NULL;
    return block->ptr;
}

void mem_pool_shutdown(void) {
    mem_pool *pool = active_pool;

    if (pool) {
        void (*cleanup)(void) = pool->cleanup;
        pool->cleanup = NULL;
        if (cleanup) cleanup();

        pool_chunk *chunk = pool->first;
        while (chunk) {
            pool_chunk *next = chunk->next;
            cuMemFree((CUdeviceptr)(uintptr_t)chunk->data);
            free(chunk);
            chunk = next;
        }

        free(pool);
        active_pool = NULL;
        if (cuda_context) {
            cuCtxDestroy(cuda_context);
            cuda_context = NULL;
        }
    }

}
