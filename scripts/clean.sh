#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${1:-generic}"

case "$backend" in
  generic|cpu) ;;
  *) echo "Unsupported backend: $backend (use generic or cpu)" >&2; exit 2 ;;
esac

exec make -C "$root_dir" TENSOR_BACKEND="$backend" clean
