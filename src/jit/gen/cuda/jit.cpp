#include "tensor_jit.h"

extern "C" jit_ker_t tensor_device_jit_compile(device dev, const jit_cache_key_t *key,
                                                uint8 **object, extent *object_size) {
    (void)dev;
    (void)key;
    (void)object;
    (void)object_size;
    return nullptr;
}

extern "C" jit_ker_t tensor_device_jit_load(device dev, const jit_cache_key_t *key,
                                             const uint8 *object, extent object_size) {
    (void)dev; (void)key; (void)object; (void)object_size;
    return nullptr;
}

extern "C" boolean tensor_device_jit_supported(device dev, const jit_cache_key_t *key) {
    (void)dev;
    (void)key;
    return false;
}

extern "C" uint64 tensor_device_jit_compatibility_hash(device dev) {
    return dev == CUDA ? UINT64_C(0x4355444150545831) : 0;
}
