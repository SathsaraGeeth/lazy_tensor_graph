#define _POSIX_C_SOURCE 200809L

#include "dev_tools/profiler/shared.h"

#include <errno.h>
#include <math.h>
#include <signal.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_EVENTS 65536
#define MAX_NODES 8192
#define MAX_EDGES 32768
#define MAX_STATS 256

enum {
    REPORT_TIME     = 1u << 0,
    REPORT_JIT      = 1u << 1,
    REPORT_MEMORY   = 1u << 2,
    REPORT_GRAPH    = 1u << 3,
    REPORT_COMPUTE  = 1u << 4,
    REPORT_ROOFLINE = 1u << 5,
    REPORT_FLAME    = 1u << 6,
    REPORT_ALL      = (1u << 7) - 1
};

typedef struct {
    uint64_t timestamp;
    uint64_t thread;
    uint64_t id;
    uint64_t parent;
    uint64_t bytes;
    uint64_t ops;
    uint32_t operation;
    uint32_t dtype;
    char type[32];
    char name[96];
} event_t;

typedef struct {
    uint64_t id;
    uint32_t operation;
    uint32_t dtype;
    uint64_t executions;
    uint64_t kernel_calls;
    uint64_t kernel_ns;
    char name[96];
} node_t;

typedef struct {
    uint64_t child;
    uint64_t parent;
} edge_t;

typedef struct {
    char name[96];
    uint64_t count;
    uint64_t duration_ns;
    uint64_t bytes;
    uint64_t ops;
    uint32_t dtype;
    int is_kernel;
    int is_library;
    int is_jit_compile;
    int is_jit_lookup;
    int operation_count_known;
} stat_t;

static event_t events[MAX_EVENTS];
static size_t event_count;
static node_t nodes[MAX_NODES];
static size_t node_count;
static edge_t edges[MAX_EDGES];
static size_t edge_count;
static stat_t stats[MAX_STATS];
static size_t stat_count;
static char tool_directory[512] = ".";
static unsigned report_mask = REPORT_ALL;
static int report_flags_selected;
static size_t malformed_rows;
static size_t unmatched_scopes;
static int trace_truncated;

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

static int export_shared_trace(pid_t process_id, const char *path,
                               uint64_t first_event) {
    tensor_profile_region *region = map_shared(process_id);
    if (!region) {
        fprintf(stderr, "tensor-profiler: shared event region was not created\n");
        return 1;
    }
    FILE *output = fopen(path, "w");
    if (!output) {
        fprintf(stderr, "tensor-profiler: cannot write %s: %s\n",
                path, strerror(errno));
        munmap(region, sizeof(*region));
        return 1;
    }
    fputs("timestamp_ns,pid,thread,type,id,parent,operation,dtype,bytes,ops,name\n",
          output);
    uint64_t count = __atomic_load_n(&region->event_count, __ATOMIC_ACQUIRE);
    if (count > TENSOR_PROFILE_EVENT_SLOTS) count = TENSOR_PROFILE_EVENT_SLOTS;
    for (uint64_t index = first_event; index < count; ++index) {
        const tensor_profile_event *event = &region->events[index];
        if (!__atomic_load_n(&event->ready, __ATOMIC_ACQUIRE)) continue;
        fprintf(output, "%llu,%ld,%llu,%s,%llu,%llu,%u,%u,%llu,%llu,%s\n",
                (unsigned long long)event->timestamp_ns, (long)process_id,
                (unsigned long long)event->thread_id, event->type,
                (unsigned long long)event->id,
                (unsigned long long)event->parent, event->operation,
                event->dtype, (unsigned long long)event->bytes,
                (unsigned long long)event->operations, event->name);
    }
    if (region->dropped_events)
        fprintf(stderr, "tensor-profiler: %llu shared events were dropped\n",
                (unsigned long long)region->dropped_events);
    fclose(output);
    munmap(region, sizeof(*region));
    return 0;
}

static void unlink_shared(pid_t process_id) {
    char name[64];
    shared_name(name, sizeof(name), process_id);
    shm_unlink(name);
}

static char *next_field(char **cursor) {
    if (!cursor || !*cursor) return NULL;
    char *field = *cursor;
    char *separator = strchr(field, ',');
    if (separator) {
        *separator = '\0';
        *cursor = separator + 1;
    } else {
        *cursor = NULL;
    }
    return field;
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage:\n"
            "  %s run [REPORT_FLAGS] [--trace FILE] -- PROGRAM [ARGS...]\n"
            "  %s report [REPORT_FLAGS] TRACE_FILE\n"
            "  %s attach [REPORT_FLAGS] PID [SECONDS]\n"
            "  %s arch linux-cpu stat -- PROGRAM [ARGS...]\n"
            "  %s arch nvidia nsys|ncu -- PROGRAM [ARGS...]\n"
            "\nreport flags:\n"
            "  --time --jit --memory --graph --compute --roofline --flame --all\n",
            program, program, program, program, program);
}

static int select_report_flag(const char *argument) {
    unsigned flag = 0;
    if (!strcmp(argument, "--time")) flag = REPORT_TIME;
    else if (!strcmp(argument, "--jit")) flag = REPORT_JIT;
    else if (!strcmp(argument, "--memory")) flag = REPORT_MEMORY;
    else if (!strcmp(argument, "--graph")) flag = REPORT_GRAPH;
    else if (!strcmp(argument, "--compute")) flag = REPORT_COMPUTE;
    else if (!strcmp(argument, "--roofline")) flag = REPORT_ROOFLINE;
    else if (!strcmp(argument, "--flame")) flag = REPORT_FLAME;
    else if (!strcmp(argument, "--all")) flag = REPORT_ALL;
    else return 0;
    if (!report_flags_selected) {
        report_mask = 0;
        report_flags_selected = 1;
    }
    report_mask |= flag;
    return 1;
}

static int load_trace(const char *path) {
    FILE *input = fopen(path, "r");
    if (!input) {
        fprintf(stderr, "tensor-profiler: cannot open %s: %s\n",
                path, strerror(errno));
        return 1;
    }
    char line[1024];
    if (!fgets(line, sizeof(line), input)) {
        fclose(input);
        return 1;
    }
    while (fgets(line, sizeof(line), input)) {
        if (event_count == MAX_EVENTS) {
            trace_truncated = 1;
            continue;
        }
        char *fields[11] = {0};
        char *cursor = line;
        for (size_t i = 0; i < 11; ++i) {
            fields[i] = next_field(&cursor);
            if (!fields[i]) break;
        }
        if (!fields[10]) {
            malformed_rows++;
            continue;
        }
        fields[10][strcspn(fields[10], "\r\n")] = '\0';
        event_t *event = &events[event_count++];
        event->timestamp = strtoull(fields[0], NULL, 10);
        event->thread = strtoull(fields[2], NULL, 10);
        snprintf(event->type, sizeof(event->type), "%s", fields[3]);
        event->id = strtoull(fields[4], NULL, 10);
        event->parent = strtoull(fields[5], NULL, 10);
        event->operation = (uint32_t)strtoul(fields[6], NULL, 10);
        event->dtype = (uint32_t)strtoul(fields[7], NULL, 10);
        event->bytes = strtoull(fields[8], NULL, 10);
        event->ops = strtoull(fields[9], NULL, 10);
        snprintf(event->name, sizeof(event->name), "%s", fields[10]);
    }
    fclose(input);
    return 0;
}

static node_t *find_node(uint64_t id) {
    for (size_t i = 0; i < node_count; ++i)
        if (nodes[i].id == id) return &nodes[i];
    return NULL;
}

static node_t *find_kernel_node(size_t event_index, uint64_t tensor_id) {
    node_t *node = find_node(tensor_id);
    if (node) return node;
    for (size_t i = event_index; i-- > 0;)
        if (!strcmp(events[i].type, "NODE_PHYSICAL") &&
            events[i].id == tensor_id)
            return find_node(events[i].parent);
    return NULL;
}

static stat_t *find_stat(const char *name, uint32_t dtype,
                         int is_kernel, int is_library,
                         int is_jit_compile, int is_jit_lookup) {
    for (size_t i = 0; i < stat_count; ++i)
        if (stats[i].dtype == dtype && !strcmp(stats[i].name, name) &&
            stats[i].is_kernel == is_kernel &&
            stats[i].is_library == is_library &&
            stats[i].is_jit_compile == is_jit_compile &&
            stats[i].is_jit_lookup == is_jit_lookup)
            return &stats[i];
    if (stat_count == MAX_STATS) return NULL;
    stat_t *stat = &stats[stat_count++];
    memset(stat, 0, sizeof(*stat));
    snprintf(stat->name, sizeof(stat->name), "%s", name);
    stat->dtype = dtype;
    stat->is_kernel = is_kernel;
    stat->is_library = is_library;
    stat->is_jit_compile = is_jit_compile;
    stat->is_jit_lookup = is_jit_lookup;
    return stat;
}

static const char *dtype_name(uint32_t dtype) {
    static const char *names[] = {
        "INT8", "INT16", "INT32", "INT64", "UINT8",
        "UINT16", "UINT32", "UINT64", "REAL32", "REAL64"
    };
    return dtype < sizeof(names) / sizeof(names[0]) ? names[dtype] : "UNKNOWN";
}

static const char *operation_class(uint32_t dtype) {
    return dtype == 8 || dtype == 9 ? "floating-point" : "integer";
}

static const event_t *matching_begin(size_t end_index, const char *begin_type) {
    const event_t *end = &events[end_index];
    for (size_t i = end_index; i-- > 0;) {
        const event_t *begin = &events[i];
        if (!strcmp(begin->type, begin_type) &&
            begin->thread == end->thread && begin->id == end->id &&
            begin->operation == end->operation &&
            begin->dtype == end->dtype &&
            !strcmp(begin->name, end->name))
            return begin;
    }
    return NULL;
}

static uint64_t matching_operation_count(size_t end_index, int *known) {
    const event_t *end = &events[end_index];
    *known = 0;
    for (size_t i = end_index; i-- > 0;) {
        const event_t *event = &events[i];
        if (!strcmp(event->type, "OP_METADATA") &&
            event->id == end->id &&
            event->operation == end->operation) {
            *known = event->parent != 0;
            return event->ops;
        }
    }
    return 0;
}

static void analyze(void) {
    for (size_t i = 0; i < event_count; ++i) {
        event_t *event = &events[i];
        if (!strcmp(event->type, "NODE_ALLOC") || !strcmp(event->type, "NODE_OP")) {
            node_t *node = find_node(event->id);
            if (!node && node_count < MAX_NODES) {
                node = &nodes[node_count++];
                node->id = event->id;
            }
            if (node) {
                node->operation = event->operation;
                node->dtype = event->dtype;
                snprintf(node->name, sizeof(node->name), "%s", event->name);
            }
        } else if (!strcmp(event->type, "EDGE") &&
                   edge_count < MAX_EDGES) {
            edges[edge_count++] =
                (edge_t){.child = event->id, .parent = event->parent};
        } else if (!strcmp(event->type, "KERNEL_BEGIN")) {
            node_t *node = find_kernel_node(i, event->id);
            if (node) {
                node->operation = event->operation;
                node->dtype = event->dtype;
                snprintf(node->name, sizeof(node->name), "%s", event->name);
            }
        }

        const char *begin_type = NULL;
        if (!strcmp(event->type, "KERNEL_END"))
            begin_type = "KERNEL_BEGIN";
        else if (!strcmp(event->type, "JIT_COMPILE_END"))
            begin_type = "JIT_COMPILE_BEGIN";
        else if (!strcmp(event->type, "JIT_MATERIALIZE_END"))
            begin_type = "JIT_MATERIALIZE_BEGIN";
        else if (!strcmp(event->type, "JIT_LOOKUP_END"))
            begin_type = "JIT_LOOKUP_BEGIN";
        else if (!strcmp(event->type, "LIB_END"))
            begin_type = "LIB_BEGIN";
        if (begin_type) {
            const event_t *begin = matching_begin(i, begin_type);
            if (begin && event->timestamp >= begin->timestamp) {
                int is_kernel = !strcmp(event->type, "KERNEL_END");
                int is_library = !strcmp(event->type, "LIB_END");
                int is_jit_compile =
                    !strcmp(event->type, "JIT_COMPILE_END") ||
                    !strcmp(event->type, "JIT_MATERIALIZE_END");
                int is_jit_lookup =
                    !strcmp(event->type, "JIT_LOOKUP_END");
                stat_t *stat = find_stat(
                    event->name, event->dtype, is_kernel, is_library,
                    is_jit_compile, is_jit_lookup);
                if (stat) {
                    stat->count++;
                    stat->duration_ns += event->timestamp - begin->timestamp;
                    stat->bytes += event->bytes;
                    if (is_kernel && !event->ops) {
                        int known = 0;
                        stat->ops += matching_operation_count(i, &known);
                        stat->operation_count_known |= known;
                    } else {
                        stat->ops += event->ops;
                        stat->operation_count_known |=
                            is_kernel && event->ops != 0;
                    }
                }
                if (is_kernel) {
                    node_t *node = find_kernel_node(i, event->id);
                    if (node) {
                        node->kernel_calls++;
                        node->kernel_ns += event->timestamp - begin->timestamp;
                    }
                }
            } else unmatched_scopes++;
        } else if (!strcmp(event->type, "NODE_END")) {
            node_t *node = find_node(event->id);
            if (node) node->executions++;
        }
    }
}

static void print_timing(void) {
    puts("\nKernel execution");
    puts("operation                        dtype       calls"
         "     total_ms       avg_ms");
    for (size_t i = 0; i < stat_count; ++i) {
        stat_t *stat = &stats[i];
        if (stat->is_library || stat->is_jit_compile ||
            stat->is_jit_lookup) continue;
        double total_ms = stat->duration_ns / 1e6;
        printf("%-32s %-8s %8llu %12.3f %12.3f\n", stat->name,
               dtype_name(stat->dtype), (unsigned long long)stat->count, total_ms,
               stat->count ? total_ms / stat->count : 0.0);
    }
}

static uint64_t accumulated_time(int category) {
    uint64_t total = 0;
    for (size_t i = 0; i < stat_count; ++i) {
        stat_t *stat = &stats[i];
        if ((category == 0 && stat->is_kernel) ||
            (category == 1 && stat->is_library) ||
            (category == 2 && stat->is_jit_compile))
            total += stat->duration_ns;
    }
    return total;
}

static void print_summary(void) {
    uint64_t capture = event_count > 1
        ? events[event_count - 1].timestamp - events[0].timestamp : 0;
    uint64_t execution_start = 0, execution_end = 0;
    for (size_t i = 0; i < event_count; ++i) {
        if (!strcmp(events[i].type, "KERNEL_BEGIN") && !execution_start)
            execution_start = events[i].timestamp;
        if (!strcmp(events[i].type, "KERNEL_END"))
            execution_end = events[i].timestamp;
    }
    uint64_t execution = execution_end >= execution_start
        ? execution_end - execution_start : 0;
    uint64_t kernel = accumulated_time(0);
    uint64_t library = accumulated_time(1);
    uint64_t jit = accumulated_time(2);
    printf("\nSummary\n"
           "capture span                         %12.3f ms\n"
           "first-to-last kernel span             %12.3f ms\n"
           "kernel work (sum of kernel calls)     %12.3f ms\n"
           "library work (inclusive, overlaps)    %12.3f ms\n"
           "JIT compile work                      %12.3f ms\n",
           capture / 1e6, execution / 1e6, kernel / 1e6,
           library / 1e6, jit / 1e6);
}

static void print_jit(void) {
    uint64_t hits = 0;
    uint64_t misses = 0;
    for (size_t i = 0; i < event_count; ++i) {
        hits += !strcmp(events[i].type, "JIT_CACHE_HIT");
        misses += !strcmp(events[i].type, "JIT_CACHE_MISS");
    }
    puts("\nJIT");
    printf("cache hits: %llu  cache misses: %llu  hit rate: %.1f%%\n",
           (unsigned long long)hits, (unsigned long long)misses,
           hits + misses ? 100.0 * hits / (hits + misses) : 0.0);
    puts("task                             dtype       calls"
         "     total_ms       avg_ms      bitcode_bytes");
    for (size_t i = 0; i < stat_count; ++i) {
        stat_t *stat = &stats[i];
        if (!stat->is_jit_compile && !stat->is_jit_lookup) continue;
        double total_ms = stat->duration_ns / 1e6;
        printf("%-32s %-8s %8llu %12.3f %12.3f %14llu\n",
               stat->name, dtype_name(stat->dtype),
               (unsigned long long)stat->count, total_ms,
               stat->count ? total_ms / stat->count : 0.0,
               (unsigned long long)stat->bytes);
    }
}

static void print_memory(void) {
    uint64_t heap_alloc_bytes = 0, heap_release_bytes = 0, copied_bytes = 0;
    uint64_t heap_allocs = 0, heap_releases = 0, copies = 0;
    uint64_t handle_releases = 0, handle_release_bytes = 0;
    uint64_t pool_planned = 0, pool_allocated = 0, pool_capacity = 0;
    uint64_t pool_requests = 0;
    for (size_t i = 0; i < event_count; ++i) {
        event_t *event = &events[i];
        if (!strcmp(event->type, "LIB_END") &&
            !strcmp(event->name, "mem_alloc") && !event->operation) {
            heap_allocs++; heap_alloc_bytes += event->bytes;
        } else if (!strcmp(event->type, "HEAP_RELEASE")) {
            heap_releases++; heap_release_bytes += event->bytes;
        } else if (!strcmp(event->type, "LIB_END") &&
                   !strcmp(event->name, "mem_free")) {
            handle_releases++; handle_release_bytes += event->bytes;
        } else if (!strcmp(event->type, "LIB_END") &&
                   (!strcmp(event->name, "mem_copy") ||
                    !strcmp(event->name, "mem_view_copy"))) {
            copies++; copied_bytes += event->bytes;
        } else if (!strcmp(event->type, "POOL_PLAN")) {
            if (event->ops > pool_planned) pool_planned = event->ops;
        } else if (!strcmp(event->type, "POOL_ALLOC")) {
            pool_requests++;
            if (event->bytes > pool_capacity) pool_capacity = event->bytes;
            if (event->ops > pool_allocated) pool_allocated = event->ops;
        } else if (!strcmp(event->type, "POOL_SHUTDOWN") &&
                   event->bytes > pool_capacity) {
            pool_capacity = event->bytes;
        }
    }
    puts("\nMemory traffic and lazy pool");
    printf("external heap allocations       %10llu  %12llu bytes\n"
           "external heap storage releases  %10llu  %12llu bytes\n"
           "memory-block handle releases    %10llu  %12llu logical bytes\n"
           "copy operations                 %10llu  %12llu bytes moved\n"
           "pool block allocations          %10llu\n"
           "pool planned bytes                          %12llu bytes\n"
           "pool committed bytes                        %12llu bytes\n"
           "pool bump high-water                        %12llu bytes\n"
           "pool reserved but unused                    %12llu bytes\n"
           "committed-space utilization                 %11.1f%%\n"
           "plan realization                            %11.1f%%\n",
           (unsigned long long)heap_allocs,
           (unsigned long long)heap_alloc_bytes,
           (unsigned long long)heap_releases,
           (unsigned long long)heap_release_bytes,
           (unsigned long long)handle_releases,
           (unsigned long long)handle_release_bytes,
           (unsigned long long)copies,
           (unsigned long long)copied_bytes,
           (unsigned long long)pool_requests,
           (unsigned long long)pool_planned,
           (unsigned long long)pool_capacity,
           (unsigned long long)pool_allocated,
           (unsigned long long)(pool_capacity >= pool_allocated
                                    ? pool_capacity - pool_allocated : 0),
           pool_capacity ? 100.0 * pool_allocated / pool_capacity : 0.0,
           pool_planned ? 100.0 * pool_allocated / pool_planned : 0.0);
}

static void print_library_overhead(void) {
    puts("\nLibrary tasks (inclusive time)");
    puts("task                             dtype       calls"
         "     total_ms       avg_us        bytes");
    for (size_t i = 0; i < stat_count; ++i) {
        stat_t *stat = &stats[i];
        if (!stat->is_library) continue;
        double total_ms = stat->duration_ns / 1e6;
        double average_us =
            stat->count ? stat->duration_ns / (double)stat->count / 1e3 : 0.0;
        printf("%-32s %-8s %8llu %12.3f %12.3f %12llu\n",
               stat->name, dtype_name(stat->dtype),
               (unsigned long long)stat->count, total_ms, average_us,
               (unsigned long long)stat->bytes);
    }
}

static void print_graph(void) {
    puts("\nOperation graph");
    puts("node   operation                 op_code     dtype    runs kernel_ms parents");
    for (size_t i = 0; i < node_count; ++i) {
        node_t *node = &nodes[i];
        printf("N%-5zu %-25s 0x%08x %-8s %4llu %9.3f [",
               i, node->name, node->operation, dtype_name(node->dtype),
               (unsigned long long)node->executions, node->kernel_ns / 1e6);
        int first = 1;
        for (size_t j = 0; j < edge_count; ++j) {
            if (edges[j].child == node->id) {
                node_t *parent = find_node(edges[j].parent);
                if (parent)
                    printf("%sN%zu", first ? "" : ",",
                           (size_t)(parent - nodes));
                else
                    printf("%s?", first ? "" : ",");
                first = 0;
            }
        }
        puts("]");
    }
    puts("kernel sequence (first execution of each node):");
    unsigned char printed[MAX_NODES] = {0};
    int any = 0;
    for (size_t i = 0; i < event_count; ++i) {
        if (!strcmp(events[i].type, "KERNEL_BEGIN")) {
            node_t *node = find_kernel_node(i, events[i].id);
            if (!node) continue;
            size_t index = (size_t)(node - nodes);
            if (printed[index]) continue;
            printf("%sN%zu:%s", any ? " -> " : "  ", index, node->name);
            printed[index] = 1;
            any = 1;
        }
    }
    puts(any ? "" : "  (no kernels executed)");
}

static void print_compute_memory(void) {
    int has_operation_counts = 0;
    puts("\nAlgorithmic throughput and tensor traffic");
    puts("kernel                       dtype    class                 GOP/s"
         "  tensor_GB/s       OP/byte");
    for (size_t i = 0; i < stat_count; ++i) {
        stat_t *stat = &stats[i];
        if (!stat->is_kernel || !stat->bytes || !stat->duration_ns) continue;
        double seconds = stat->duration_ns / 1e9;
        if (!stat->operation_count_known) {
            printf("%-28s %-8s %-18s %10s %12.3f %13s\n",
                   stat->name, dtype_name(stat->dtype),
                   operation_class(stat->dtype), "unknown",
                   stat->bytes / seconds / 1e9, "not reported");
            continue;
        }
        has_operation_counts = 1;
        printf("%-28s %-8s %-18s %10.3f %12.3f %13.4f\n",
               stat->name, dtype_name(stat->dtype),
               operation_class(stat->dtype),
               stat->ops / seconds / 1e9,
               stat->bytes / seconds / 1e9,
               (double)stat->ops / stat->bytes);
    }
    puts("OP rates use the selected .op algorithm's static operations hint.");
    puts("tensor_GB/s is input+output tensor traffic, not hardware DRAM bandwidth.");
    if (!has_operation_counts)
        puts("No executed kernel reported a nonzero algorithmic operation count.");
}

static void write_artifacts(const char *trace_path, int write_roofline,
                            int write_flame) {
    char path[1024];
    if (write_roofline) {
        size_t roofline_points = 0;
        for (size_t i = 0; i < stat_count; ++i) {
            stat_t *s = &stats[i];
            if (s->is_kernel && s->bytes && s->ops && s->duration_ns)
                roofline_points++;
        }
        if (!roofline_points) {
            puts("\nroofline: not generated (no kernel operation counts)");
        } else {
            snprintf(path, sizeof(path), "%s.roofline.csv", trace_path);
            FILE *roofline = fopen(path, "w");
            if (roofline) {
                fputs("kernel,dtype,operation_class,count,time_ns,bytes,ops,"
                      "operational_intensity,gops_per_second,"
                      "gb_per_second\n", roofline);
                for (size_t i = 0; i < stat_count; ++i) {
                    stat_t *s = &stats[i];
                    if (!s->is_kernel || !s->bytes || !s->ops ||
                        !s->duration_ns) continue;
                    double seconds = s->duration_ns / 1e9;
                    fprintf(roofline, "%s,%s,%s,%llu,%llu,%llu,%llu,"
                            "%.9f,%.9f,%.9f\n",
                            s->name, dtype_name(s->dtype),
                            operation_class(s->dtype),
                            (unsigned long long)s->count,
                            (unsigned long long)s->duration_ns,
                            (unsigned long long)s->bytes,
                            (unsigned long long)s->ops,
                            (double)s->ops / s->bytes,
                            s->ops / seconds / 1e9,
                            s->bytes / seconds / 1e9);
                }
                fclose(roofline);
                printf("\nroofline data: %s\n", path);
            }
            double min_x = 1e300, max_x = -1e300;
            double min_y = 1e300, max_y = -1e300;
            for (size_t i = 0; i < stat_count; ++i) {
                stat_t *s = &stats[i];
                if (!s->is_kernel || !s->bytes || !s->ops ||
                    !s->duration_ns) continue;
                double x = log10((double)s->ops / s->bytes);
                double y = log10(s->ops / (s->duration_ns / 1e9) / 1e9);
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
            if (max_x - min_x < 0.5) {
                min_x -= 0.25; max_x += 0.25;
            }
            if (max_y - min_y < 0.5) {
                min_y -= 0.25; max_y += 0.25;
            }
            snprintf(path, sizeof(path), "%s.roofline.svg", trace_path);
            FILE *svg = fopen(path, "w");
            if (svg) {
                fputs("<svg xmlns=\"http://www.w3.org/2000/svg\" "
                      "width=\"1000\" height=\"620\" viewBox=\"0 0 1000 620\">"
                      "<rect width=\"1000\" height=\"620\" fill=\"#fff\"/>"
                      "<text x=\"500\" y=\"30\" text-anchor=\"middle\" "
                      "font-family=\"sans-serif\" font-size=\"20\">"
                      "Algorithmic operational intensity</text>"
                      "<line x1=\"90\" y1=\"540\" x2=\"950\" y2=\"540\" "
                      "stroke=\"#111\"/><line x1=\"90\" y1=\"540\" "
                      "x2=\"90\" y2=\"60\" stroke=\"#111\"/>"
                      "<text x=\"520\" y=\"595\" text-anchor=\"middle\" "
                      "font-family=\"sans-serif\">OP / tensor byte "
                      "(log scale)</text><text x=\"22\" y=\"300\" "
                      "text-anchor=\"middle\" font-family=\"sans-serif\" "
                      "transform=\"rotate(-90 22 300)\">GOP/s "
                      "(log scale)</text>", svg);
                fprintf(svg,
                        "<text x=\"90\" y=\"560\" font-size=\"11\">%.3g</text>"
                        "<text x=\"950\" y=\"560\" text-anchor=\"end\" "
                        "font-size=\"11\">%.3g</text>"
                        "<text x=\"80\" y=\"540\" text-anchor=\"end\" "
                        "font-size=\"11\">%.3g</text>"
                        "<text x=\"80\" y=\"70\" text-anchor=\"end\" "
                        "font-size=\"11\">%.3g</text>",
                        pow(10.0, min_x), pow(10.0, max_x),
                        pow(10.0, min_y), pow(10.0, max_y));
                for (size_t i = 0; i < stat_count; ++i) {
                    stat_t *s = &stats[i];
                    if (!s->is_kernel || !s->bytes || !s->ops ||
                        !s->duration_ns) continue;
                    double x_value = log10((double)s->ops / s->bytes);
                    double y_value =
                        log10(s->ops / (s->duration_ns / 1e9) / 1e9);
                    double x = 90.0 +
                        860.0 * (x_value - min_x) / (max_x - min_x);
                    double y = 540.0 -
                        470.0 * (y_value - min_y) / (max_y - min_y);
                    fprintf(svg,
                            "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"5\" "
                            "fill=\"#2563eb\"/><text x=\"%.1f\" y=\"%.1f\" "
                            "font-family=\"sans-serif\" font-size=\"11\">"
                            "%s</text>",
                            x, y, x + 7, y - 7, s->name);
                }
                fputs("<text x=\"950\" y=\"610\" text-anchor=\"end\" "
                      "font-family=\"sans-serif\" font-size=\"10\" "
                      "fill=\"#555\">Static .op operation counts; traced "
                      "tensor traffic</text></svg>", svg);
                fclose(svg);
                printf("roofline graph: %s\n", path);
            }
        }
    }
    if (write_flame) {
        snprintf(path, sizeof(path), "%s.flame.folded", trace_path);
        FILE *flame = fopen(path, "w");
        if (flame) {
            for (size_t i = 0; i < stat_count; ++i)
                if (stats[i].is_kernel)
                    fprintf(flame, "tensor;kernels;%s %llu\n",
                            stats[i].name,
                            (unsigned long long)
                                (stats[i].duration_ns / 1000));
            fclose(flame);
            printf("kernel flame data: %s\n", path);
        }
    }
}

static int report(const char *path) {
    if (load_trace(path)) return 1;
    analyze();
    printf("Tensor profiler: %zu events, %zu graph nodes\n",
           event_count, node_count);
    if (trace_truncated)
        printf("warning: trace exceeds %u events; report is truncated\n",
               MAX_EVENTS);
    if (malformed_rows || unmatched_scopes)
        printf("trace quality: %zu malformed rows, %zu unmatched scopes\n",
               malformed_rows, unmatched_scopes);
    print_summary();
    if (report_mask & REPORT_TIME) {
        print_timing();
        print_library_overhead();
    }
    if (report_mask & REPORT_JIT) print_jit();
    if (report_mask & REPORT_MEMORY) print_memory();
    if (report_mask & REPORT_GRAPH) print_graph();
    if (report_mask & REPORT_COMPUTE) print_compute_memory();
    write_artifacts(path, report_mask & REPORT_ROOFLINE,
                    report_mask & REPORT_FLAME);
    return 0;
}

static int run_program(int argc, char **argv) {
    const char *trace = "tensor-trace.csv";
    int index = 2;
    while (index < argc) {
        if (index + 1 < argc && !strcmp(argv[index], "--trace")) {
            trace = argv[index + 1];
            index += 2;
        } else if (select_report_flag(argv[index])) {
            index++;
        } else {
            break;
        }
    }
    if (index < argc && !strcmp(argv[index], "--")) index++;
    if (index >= argc) return 1;
    pid_t child = fork();
    if (child == 0) {
        setenv("TENSOR_PROFILE_SHM", "1", 1);
        execvp(argv[index], &argv[index]);
        _exit(127);
    }
    if (child < 0) return 1;
    int status;
    if (waitpid(child, &status, 0) < 0) return 1;
    if (!WIFEXITED(status) || WEXITSTATUS(status)) {
        fprintf(stderr, "tensor-profiler: program failed; report skipped\n");
        return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
    }
    int result = export_shared_trace(child, trace, 0);
    unlink_shared(child);
    if (result) return result;
    result = report(trace);
    if (result) return result;
    return 0;
}

static int attach_process(const char *pid, const char *seconds) {
    char *end = NULL;
    long process_id = strtol(pid, &end, 10);
    if (!pid[0] || (end && *end) || process_id <= 0) return 1;
    unsigned duration = seconds ? (unsigned)strtoul(seconds, NULL, 10) : 10;
    if (!duration) duration = 1;
    tensor_profile_region *region = NULL;
    const struct timespec startup_delay = {.tv_sec = 0, .tv_nsec = 20000000L};
    for (unsigned attempt = 0; attempt < 100 && !region; ++attempt) {
        region = map_shared((pid_t)process_id);
        if (region || (kill((pid_t)process_id, 0) && errno == ESRCH)) break;
        nanosleep(&startup_delay, NULL);
    }
    if (!region) {
        fprintf(stderr,
                "tensor-profiler: process %ld has no shared profiler region; "
                "start it with TENSOR_PROFILE_SHM=1\n",
                process_id);
        return 1;
    }
    uint64_t first_event = __atomic_load_n(&region->event_count,
                                            __ATOMIC_ACQUIRE);
    munmap(region, sizeof(*region));
    sleep(duration);
    char trace[128];
    snprintf(trace, sizeof(trace), "/tmp/tensor-trace-%ld.csv", process_id);
    if (export_shared_trace((pid_t)process_id, trace, first_event)) return 1;
    return report(trace);
}

static int run_arch_adapter(int argc, char **argv) {
    if (argc < 5) return 1;
    char adapter[1024];
    if (!strcmp(argv[2], "linux-cpu") && !strcmp(argv[3], "stat"))
        snprintf(adapter, sizeof(adapter), "%s/arch/linux_cpu/perf_stat.sh",
                 tool_directory);
    else if (!strcmp(argv[2], "nvidia") && !strcmp(argv[3], "nsys"))
        snprintf(adapter, sizeof(adapter), "%s/arch/nvidia/nsys.sh",
                 tool_directory);
    else if (!strcmp(argv[2], "nvidia") && !strcmp(argv[3], "ncu"))
        snprintf(adapter, sizeof(adapter), "%s/arch/nvidia/ncu.sh",
                 tool_directory);
    else
        return 1;
    int index = 4;
    if (!strcmp(argv[index], "--")) index++;
    if (index >= argc) return 1;
    size_t count = (size_t)(argc - index);
    char **arguments = calloc(count + 3, sizeof(*arguments));
    if (!arguments) return 1;
    arguments[0] = (char *)"sh";
    arguments[1] = adapter;
    for (size_t i = 0; i < count; ++i) arguments[i + 2] = argv[index + i];
    execv("/bin/sh", arguments);
    fprintf(stderr, "tensor-profiler: failed to run %s: %s\n",
            adapter, strerror(errno));
    free(arguments);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    const char *configured_directory = getenv("TENSOR_PROFILER_DIR");
    const char *slash = strrchr(argv[0], '/');
    if (configured_directory && *configured_directory) {
        snprintf(tool_directory, sizeof(tool_directory), "%s",
                 configured_directory);
    } else if (slash) {
        size_t length = (size_t)(slash - argv[0]);
        if (length >= sizeof(tool_directory)) length = sizeof(tool_directory) - 1;
        memcpy(tool_directory, argv[0], length);
        tool_directory[length] = '\0';
    }
    char adapter_directory[sizeof(tool_directory) + 16];
    snprintf(adapter_directory, sizeof(adapter_directory), "%s/arch",
             tool_directory);
    if (access(adapter_directory, F_OK) &&
        !access("src/dev_tools/profiler/arch", F_OK))
        snprintf(tool_directory, sizeof(tool_directory),
                 "src/dev_tools/profiler");
    if (!strcmp(argv[1], "run")) return run_program(argc, argv);
    if (!strcmp(argv[1], "report")) {
        int index = 2;
        while (index < argc && select_report_flag(argv[index])) index++;
        if (index + 1 == argc) return report(argv[index]);
        usage(argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "attach")) {
        int index = 2;
        while (index < argc && select_report_flag(argv[index])) index++;
        if (index < argc && index + 2 >= argc)
            return attach_process(argv[index],
                                  index + 1 < argc ? argv[index + 1] : NULL);
        usage(argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "arch")) return run_arch_adapter(argc, argv);
    usage(argv[0]);
    return 1;
}
