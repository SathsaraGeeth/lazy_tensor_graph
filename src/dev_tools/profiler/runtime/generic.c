#include "dev_tools/profiler/shared.h"
#include "dev_tools/profiler/trace.h"

tensor_profile_region *tensor_profile_shared_region;

void tensor_profile_record(const char *type, const char *name,
                           uint64 id, uint64 parent, uint32 operation,
                           dtype_t dtype, uint64 bytes, uint64 operations) {
    (void)type;
    (void)name;
    (void)id;
    (void)parent;
    (void)operation;
    (void)dtype;
    (void)bytes;
    (void)operations;
}
