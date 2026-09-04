/* 
 * user/include/tensor/device.h
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
 * 1. Define devices
 * 2. FROZEN
 */

#ifndef DEVICE_H
#define DEVICE_H

typedef enum device {
    GENERIC,
    CPU,
    CUDA
} device;

#ifdef BACKEND_CPU
#define DEVICE CPU
#elif defined(BACKEND_CUDA)
#define DEVICE CUDA
#else
#define DEVICE GENERIC
#endif

static inline device get_device(void) {
    return DEVICE;
}

#endif /* DEVICE_H */
