#define _POSIX_C_SOURCE 200809L

#include "dev_tools/profiler/shared.h"
#include "dev_tools/profiler/trace.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

tensor_profile_region *tensor_profile_shared_region;
static int profile_initialized;

static uint64_t timestamp_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + value.tv_nsec;
}

static tensor_profile_region *region_get(void) {
    if (profile_initialized) return tensor_profile_shared_region;
    profile_initialized = 1;
    if (!getenv("TENSOR_PROFILE_SHM")) return NULL;

    char name[64];
    snprintf(name, sizeof(name), "/tensor-profiler-%ld", (long)getpid());
    int descriptor = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (descriptor < 0 || ftruncate(descriptor, sizeof(tensor_profile_region))) {
        if (descriptor >= 0) close(descriptor);
        return NULL;
    }
    void *mapping = mmap(NULL, sizeof(tensor_profile_region),
                         PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    close(descriptor);
    if (mapping == MAP_FAILED) return NULL;
    tensor_profile_shared_region = mapping;
    if (tensor_profile_shared_region->magic != TENSOR_PROFILE_MAGIC) {
        memset(tensor_profile_shared_region, 0,
               sizeof(*tensor_profile_shared_region));
        tensor_profile_shared_region->magic = TENSOR_PROFILE_MAGIC;
        tensor_profile_shared_region->version = TENSOR_PROFILE_VERSION;
        tensor_profile_shared_region->slot_count = TENSOR_PROFILE_SLOT_COUNT;
        tensor_profile_shared_region->process_id = (uint64_t)getpid();
    }
    return tensor_profile_shared_region;
}


__attribute__((constructor))
static void tensor_profile_initialize(void) {
    (void)region_get();
}

void tensor_profile_record(const char *type, const char *name,
                           uint64 id, uint64 parent, uint32 operation,
                           dtype_t dtype, uint64 bytes, uint64 operations) {
    tensor_profile_region *region = region_get();
    if (!region) return;
    uint64_t index = __atomic_fetch_add(&region->event_count, 1,
                                        __ATOMIC_RELAXED);
    if (index >= TENSOR_PROFILE_EVENT_SLOTS) {
        __atomic_fetch_add(&region->dropped_events, 1, __ATOMIC_RELAXED);
        return;
    }
    tensor_profile_event *event = &region->events[index];
    event->timestamp_ns = timestamp_ns();
    event->thread_id = (uint64_t)(uintptr_t)pthread_self();
    event->id = id;
    event->parent = parent;
    event->bytes = bytes;
    event->operations = operations;
    event->operation = operation;
    event->dtype = (uint32_t)dtype;
    snprintf(event->type, sizeof(event->type), "%s", type ? type : "");
    snprintf(event->name, sizeof(event->name), "%s", name ? name : "");
    __atomic_store_n(&event->ready, 1, __ATOMIC_RELEASE);
}
