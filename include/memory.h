/*
 * user/src/tensor/include/memory.h
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
 * 1. Abstracts memory from backend implementations
 * 2. FROZEN
 */

#ifndef MEMORY_H
#define MEMORY_H

#include "dtype.h"

typedef struct {
    dptr*       ptr;
    extent      size;
    boolean     is_owner;
    boolean     is_pooled;
} mem_block;

typedef struct mem_pool mem_pool;

mem_pool  *mem_pool_init(void);                                          /* init a pool what is a pool is not a concern */
void       mem_pool_plan(mem_pool *pool, extent size);                   /* plan the pool on demand */     
void       mem_pool_set_cleanup(mem_pool *pool, void (*cleanup)(void));  /* a function to destroy the pool*/
void       mem_pool_shutdown(void);                                      /* destroy the pool */

mem_block *mem_alloc(extent size, boolean is_pooled, mem_pool *pool);
mem_block *mem_view_from(extent size, const dptr *data);
boolean    mem_free(mem_block *block);                                   /* can behave differently for pooled/non pooled or not dont care */
boolean    mem_copy(mem_block *dst, const mem_block *src, extent size);
boolean    mem_view(mem_block *dst, const dptr *data);
boolean    mem_view_copy(mem_block *dst, const dptr *data);
dptr      *mem_view_to(mem_block *block);                                /* only constraint is host should be able to get hold of this */

#endif /* MEMORY_H */
