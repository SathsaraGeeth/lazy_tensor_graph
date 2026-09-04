#define _POSIX_C_SOURCE 200809L

#include "dev_tools/profiler/shared.h"
#include "dev_tools/profiler/trace.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

tensor_profile_region *tensor_profile_shared_region;
static int profile_initialized;

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
