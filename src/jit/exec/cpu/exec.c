#include "internal.h"

#ifdef BACKEND_CPU
#include "../../gen/cpu/abstract_parallel/abstract_parallel.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static uint64_t tensor_probe_time_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;
}

typedef struct {
    vtensor *node;
    extent remaining;
} exec_use;

static exec_use *find_use(exec_use *uses, extent count, vtensor *node) {
    for (extent i = 0; i < count; ++i)
        if (uses[i].node == node) return &uses[i];
    return NULL;
}

static exec_use *collect_uses(const exec_work *items, extent item_count, extent *count_out) {
    extent maximum = 0;
    for (extent i = 0; i < item_count; ++i)
        if (items[i].kind == EXEC_KERNEL) maximum += items[i].node->num_parents;
    exec_use *uses = maximum ? calloc(maximum, sizeof(*uses)) : NULL;
    if (maximum && !uses) {
        *count_out = (extent)-1;
        return NULL;
    }
    extent count = 0;
    for (extent i = 0; i < item_count; ++i) {
        const exec_work *work = &items[i];
        if (work->kind != EXEC_KERNEL) continue;
        for (extent j = 0; j < work->node->num_parents; ++j) {
            vtensor *input = work->node->parents[j];
            exec_use *use = find_use(uses, count, input);
            if (!use) {
                use = &uses[count++];
                *use = (exec_use){.node = input};
            }
            use->remaining++;
        }
    }
    *count_out = count;
    return uses;
}

static void release_inputs(exec_context *context, exec_use *uses, extent use_count, const exec_work *work) {
    for (extent i = 0; i < work->node->num_parents; ++i) {
        exec_use *use = find_use(uses, use_count, work->node->parents[i]);
        if (use && use->remaining && !--use->remaining) exec_recycle(context, use->node);
    }
}

static boolean execute_work(exec_context *context, const exec_work *work) {
    if (work->kind == EXEC_ALLOC) return exec_allocate(context, work->node);
    if (work->kind == EXEC_INIT) return exec_initialize(work->node);
    if (work->kind == EXEC_KERNEL) return exec_execute_kernel(work->node);
    return true;
}

boolean tensor_exec_submit(const void *opaque_items, extent item_count) {
    uint64_t started = tensor_probe_time_ns();
    const exec_work *items = opaque_items;
    if (item_count && !items) return true;
#ifdef BACKEND_CPU
    tensor_cpu_pool_begin();
#endif
    extent use_count = 0;
    exec_use *uses = collect_uses(items, item_count, &use_count);
    if (use_count == (extent)-1) return true;
    boolean failed = false;
    for (extent i = 0; i < item_count && !failed; ++i)
        if (items[i].kind == EXEC_KERNEL) failed = exec_prepare_kernel(items[i].node);
    uint64_t prepared = tensor_probe_time_ns();
    exec_context context = {0};
    if (!failed) failed = exec_context_begin(&context);
    uint64_t context_started = tensor_probe_time_ns();
    for (extent i = 0; i < item_count && !failed; ++i) {
        failed = execute_work(&context, &items[i]);
        if (!failed && items[i].kind == EXEC_KERNEL) release_inputs(&context, uses, use_count, &items[i]);
    }
    exec_context_end(&context);
    uint64_t finished = tensor_probe_time_ns();
    free(uses);
    static unsigned samples;
    if (getenv("TENSOR_TIMING_PROBES") && samples++ < 32)
        fprintf(stderr,
                "[tensor probe] exec prepare=%.3f us context=%.3f us execute=%.3f us total=%.3f us\n",
                (prepared - started) / 1000.0,
                (context_started - prepared) / 1000.0,
                (finished - context_started) / 1000.0,
                (finished - started) / 1000.0);
    return failed;
}
