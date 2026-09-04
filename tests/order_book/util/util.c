#include "util.h"

#include <inttypes.h>
#include <stdio.h>


uint64 time_ns(void) {
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64)now.tv_sec * 1000000000ULL + (uint64)now.tv_nsec;
}


static void record(PipelineStats *stats, uint64 latency) {
    ++stats->count;
    stats->sum += latency;

    if (latency < stats->min)
        stats->min = latency;
    if (latency > stats->max)
        stats->max = latency;
}


void pipeline_stats_record(PipelineStats *stats, uint64 started_ns) {
    record(stats, time_ns() - started_ns);
}


void pipeline_stats_record_pair(PipelineStats *queue_stats,
                                PipelineStats *compute_stats,
                                uint64 queued_ns,
                                uint64 compute_start_ns,
                                uint64 completed_ns) {
    static uint64 started_ns;
    struct timespec wall;

    if (!started_ns)
        started_ns = completed_ns;

    uint64 queue_latency = completed_ns - queued_ns;
    uint64 compute_latency = completed_ns - compute_start_ns;
    double elapsed_s = (completed_ns - started_ns) / 1e9;

    clock_gettime(CLOCK_REALTIME, &wall);
    record(queue_stats, queue_latency);
    record(compute_stats, compute_latency);

    if (queue_stats->count == 1) {
        fprintf(stderr,
                "\n%17s %8s %12s %12s %12s %12s\n"
                "%17s %8s %12s %12s %12s %12s\n",
                "wall_time", "time_s", "end_to_end_us", "end_mean_us",
                "compute_us", "compute_mean",
                "-----------------", "--------", "------------", "-----------",
                "----------", "------------");
    }

    fprintf(stderr,
            "%10" PRIu64 ".%06ld %8.6f %12.3f %12.3f %12.3f %12.3f\n",
            (uint64)wall.tv_sec, wall.tv_nsec / 1000, elapsed_s,
            queue_latency / 1000.0,
            queue_stats->sum / queue_stats->count / 1000.0,
            compute_latency / 1000.0,
            compute_stats->sum / compute_stats->count / 1000.0);
}
