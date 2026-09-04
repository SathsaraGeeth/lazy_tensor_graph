#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${1:-generic}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$backend" in
  generic|cpu) ;;
  *) echo "Unsupported backend: $backend (use generic or cpu)" >&2; exit 2 ;;
esac

if [[ -z "${LLVM_CONFIG:-}" ]]; then
  for candidate in llvm-config-18 llvm-config "$(brew --prefix llvm@18 2>/dev/null || true)/bin/llvm-config"; do
    if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1; then
      LLVM_CONFIG="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "${LLVM_CONFIG:-}" ]]; then
  echo "LLVM 18 is required. Install it with: brew install llvm@18" >&2
  exit 1
fi

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v make >/dev/null || { echo "make is required" >&2; exit 1; }
llvm_bin="$(dirname "$LLVM_CONFIG")"
cc="${CC:-$llvm_bin/clang}"
cxx="${CXX:-$llvm_bin/clang++}"
if ! command -v "$cc" >/dev/null || ! command -v "$cxx" >/dev/null; then
  echo "A C and C++ compiler are required." >&2
  exit 1
fi

openmp_flags="${OPENMP_FLAGS:--fopenmp}"
if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null; then
  omp_prefix="$(brew --prefix libomp 2>/dev/null || true)"
  if [[ -n "$omp_prefix" ]]; then
    openmp_flags="-fopenmp -I$omp_prefix/include -L$omp_prefix/lib -Wl,-rpath,$omp_prefix/lib"
  fi
fi

exec make -C "$root_dir" TARGET=host TENSOR_BACKEND="$backend" LLVM_CONFIG="$LLVM_CONFIG" CLANG="$cc" CC="$cc" CXX="$cxx" OPENMP_FLAGS="$openmp_flags" "$@"
