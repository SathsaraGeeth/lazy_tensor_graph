#!/bin/sh
set -eu

perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses "$@"
