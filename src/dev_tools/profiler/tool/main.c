#define _POSIX_C_SOURCE 200809L

#include "dev_tools/profiler/shared.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define TENSOR_PROFILE_MAX_SAMPLES 512u

typedef struct {
    tensor_profile_region *region;
    uint64_t polls;
    uint64_t started_ns;
    uint64_t finished_ns;
    uint64_t last_begin[TENSOR_PROFILE_OPERATION_SLOTS];
    uint64_t last_end[TENSOR_PROFILE_OPERATION_SLOTS];
    uint64_t begin_ns[TENSOR_PROFILE_OPERATION_SLOTS];
    uint64_t total_ns[TENSOR_PROFILE_OPERATION_SLOTS];
    uint64_t minimum_ns[TENSOR_PROFILE_OPERATION_SLOTS];
    uint64_t maximum_ns[TENSOR_PROFILE_OPERATION_SLOTS];
    uint64_t samples[TENSOR_PROFILE_OPERATION_SLOTS];
    uint64_t missed[TENSOR_PROFILE_OPERATION_SLOTS];
    uint64_t observations[TENSOR_PROFILE_OPERATION_SLOTS][TENSOR_PROFILE_MAX_SAMPLES];
} collector;

static const uint32_t operation_slots[] = {
#define TENSOR_OPERATION(name, code, descriptor, lower, input_arity, parameters_size) \
    TENSOR_PROFILE_OPERATION_SLOT(code),
#include "../../../kernels/operations.def"
#undef TENSOR_OPERATION
};

static uint64_t time_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;
}

static int compare_u64(const void *left, const void *right) {
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static const char *operation_name(uint32_t operation) {
    switch (operation) {
#define TENSOR_OPERATION(name, code, descriptor, lower, input_arity, parameters_size) \
        case code: return #name;
#include "../../../kernels/operations.def"
#undef TENSOR_OPERATION
        default: return "UNKNOWN";
    }
}

static uint32_t operation_for_slot(uint32_t slot) {
#define TENSOR_OPERATION(name, code, descriptor, lower, input_arity, parameters_size) \
    if (TENSOR_PROFILE_OPERATION_SLOT(code) == slot) return code;
#include "../../../kernels/operations.def"
#undef TENSOR_OPERATION
    return 0;
}

static uint32_t operation_for_phase_slot(uint32_t slot) {
    return operation_for_slot(slot < TENSOR_PROFILE_OPERATION_SLOTS
                            ? slot
                            : slot - TENSOR_PROFILE_OPERATION_SLOTS);
}

static void shared_name(char *name, size_t size, pid_t process_id) {
    snprintf(name, size, "/tensor-profiler-%ld", (long)process_id);
}

static tensor_profile_region *map_shared(pid_t process_id) {
    char name[64];
    shared_name(name, sizeof(name), process_id);
    int descriptor = shm_open(name, O_RDONLY, 0);
    if (descriptor < 0) return NULL;
    tensor_profile_region *region = mmap(
        NULL, sizeof(*region), PROT_READ, MAP_SHARED, descriptor, 0);
    close(descriptor);
    if (region == MAP_FAILED || region->magic != TENSOR_PROFILE_MAGIC ||
        region->version != TENSOR_PROFILE_VERSION) {
        if (region != MAP_FAILED) munmap(region, sizeof(*region));
        return NULL;
    }
    return region;
}

static void poll_counters(collector *state) {
    uint64_t now = time_ns();
    state->finished_ns = now;
    if (!state->started_ns) state->started_ns = now;

    for (size_t i = 0; i < sizeof(operation_slots) / sizeof(operation_slots[0]); ++i) {
        uint32_t slot = operation_slots[i];
        uint64_t begins = state->region->counters[slot];
        uint64_t ends = state->region->counters[
            TENSOR_PROFILE_OPERATION_SLOTS + slot];
        uint64_t new_begins = begins - state->last_begin[slot];
        uint64_t new_ends = ends - state->last_end[slot];

        if (new_begins && new_ends) {
            state->missed[slot] += new_begins < new_ends
                                 ? new_begins : new_ends;
            state->begin_ns[slot] = new_begins > new_ends ? now : 0;
        } else if (new_begins) {
            if (new_begins > 1) state->missed[slot] += new_begins - 1;
            state->begin_ns[slot] = now;
        } else if (new_ends) {
            if (new_ends > 1) state->missed[slot] += new_ends - 1;
            if (state->begin_ns[slot]) {
                uint64_t duration = now - state->begin_ns[slot];
                state->total_ns[slot] += duration;
                if (!state->minimum_ns[slot] ||
                    duration < state->minimum_ns[slot])
                    state->minimum_ns[slot] = duration;
                if (duration > state->maximum_ns[slot])
                    state->maximum_ns[slot] = duration;
                if (state->samples[slot] < TENSOR_PROFILE_MAX_SAMPLES)
                    state->observations[slot][state->samples[slot]] = duration;
                state->samples[slot]++;
                state->begin_ns[slot] = 0;
            } else {
                state->missed[slot]++;
            }
        }

        state->last_begin[slot] = begins;
        state->last_end[slot] = ends;
    }
    state->polls++;
}

static void report(pid_t process_id, collector *state) {
    uint64_t elapsed = state->finished_ns >= state->started_ns
                     ? state->finished_ns - state->started_ns : 0;
    printf("Tensor profiler: process %ld, %llu polls, %.3f ms\n",
           (long)process_id, (unsigned long long)state->polls, elapsed / 1e6);
    printf("%-8s %-32s %14s %16s\n",
           "slot", "operation", "calls", "calls/s");
    printf("-------- -------------------------------- -------------- ----------------\n");
    for (uint32_t slot = 0; slot < state->region->slot_count; ++slot) {
        uint64_t count = state->region->counters[slot];
        if (!count) continue;
        uint32_t operation = slot < 2 * TENSOR_PROFILE_OPERATION_SLOTS
                           ? operation_for_phase_slot(slot) : 0;
        char utility[32];
        if (!operation) {
            snprintf(utility, sizeof(utility), "UTILITY_%u",
                     slot - TENSOR_PROFILE_OPERATION_SLOTS);
        } else if (slot >= TENSOR_PROFILE_OPERATION_SLOTS) {
            snprintf(utility, sizeof(utility), "%s.end",
                     operation_name(operation));
        }
        printf("%8u %-32.32s %14llu %16.3f\n",
               slot, operation ? (slot < TENSOR_PROFILE_OPERATION_SLOTS
                                ? operation_name(operation) : utility) : utility,
               (unsigned long long)count,
               elapsed ? count * 1e9 / elapsed : 0.0);
    }

    printf("\n%-32s %10s %12s %12s %12s %12s %10s\n",
           "kernel", "samples", "min_us", "median_us", "mean_us", "max_us", "missed");
    printf("-------------------------------- ---------- ------------ ------------ ------------ ------------ ----------\n");
    for (uint32_t slot = 0; slot < TENSOR_PROFILE_OPERATION_SLOTS; ++slot) {
        uint32_t operation = operation_for_slot(slot);
        if (!operation || (!state->samples[slot] && !state->missed[slot])) continue;
        double mean = state->samples[slot]
                    ? (double)state->total_ns[slot] / state->samples[slot] / 1000.0
                    : 0.0;
        size_t stored = state->samples[slot] < TENSOR_PROFILE_MAX_SAMPLES
                      ? state->samples[slot] : TENSOR_PROFILE_MAX_SAMPLES;
        qsort(state->observations[slot], stored, sizeof(uint64_t), compare_u64);
        double median = stored
                      ? state->observations[slot][stored / 2] / 1000.0 : 0.0;
        printf("%-32.32s %10llu %12.3f %12.3f %12.3f %12.3f %10llu\n",
               operation_name(operation),
               (unsigned long long)state->samples[slot],
               state->minimum_ns[slot] / 1000.0, median, mean,
               state->maximum_ns[slot] / 1000.0,
               (unsigned long long)state->missed[slot]);
    }
}

static int collect_child(pid_t child) {
    static collector state;
    int status = 0;
    for (;;) {
        pid_t result = waitpid(child, &status, WNOHANG);
        if (result < 0) return 1;
        if (!state.region) state.region = map_shared(child);
        if (state.region) poll_counters(&state);
        if (result == child) break;
    }
    if (!state.region) state.region = map_shared(child);
    if (!state.region) {
        fprintf(stderr, "tensor-profiler: shared counter region was not created\n");
        return (!WIFEXITED(status) || WEXITSTATUS(status))
             ? (WIFEXITED(status) ? WEXITSTATUS(status) : 128) : 1;
    }
    report(child, &state);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) {
        fprintf(stderr, "tensor-profiler: child status %s\n",
                WIFEXITED(status) ? "failed" : "interrupted");
    }
    munmap(state.region, sizeof(*state.region));
    char name[64];
    shared_name(name, sizeof(name), child);
    shm_unlink(name);
    return 0;
}

static int run_program(int argc, char **argv) {
    int index = 2;
    while (index < argc && strcmp(argv[index], "--")) {
        if (!strcmp(argv[index], "--trace") && index + 1 < argc) index += 2;
        else index++;
    }
    if (index < argc && !strcmp(argv[index], "--")) index++;
    if (index >= argc) return 1;
    pid_t child = fork();
    if (child == 0) {
        setenv("TENSOR_PROFILE_SHM", "1", 1);
        execvp(argv[index], &argv[index]);
        _exit(127);
    }
    return child < 0 ? 1 : collect_child(child);
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s run [OPTIONS] -- PROGRAM [ARGS...]\n", program);
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "run")) return run_program(argc, argv);
    usage(argv[0]);
    return 1;
}
