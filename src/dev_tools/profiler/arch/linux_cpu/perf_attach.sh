#!/bin/sh
set -eu

pid=$1
seconds=${2:-10}
output=${TENSOR_PROFILER_PERF_DATA:-tensor-perf.data}

perf record -F 199 -g -o "$output" -p "$pid" -- sleep "$seconds"
perf script -i "$output" > "$output.script"
echo "perf data: $output"
echo "flamegraph input: $output.script"
