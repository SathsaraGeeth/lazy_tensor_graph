#!/bin/sh
set -eu

exec nsys profile --trace=cuda,nvtx,osrt "$@"
