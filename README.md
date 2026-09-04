# Lazy Tensor Graph

This repository contains a graph-based lazy tensor runtime. It provides a DSL
for defining kernels, generates a kernel registry, and JIT-compiles operations
for generic CPU, optimized CPU, and CUDA execution backends.

## Quick start

The host build requires Python 3, GNU Make, a C/C++ compiler with OpenMP
support, and LLVM 18. On macOS, install the required toolchain with
`brew install llvm@18 libomp`.

Build the default generic CPU backend:

```sh
./scripts/build.sh
```

Build and run the smoke test:

```sh
./scripts/run.sh
```

Select another backend explicitly:

```sh
./scripts/build.sh cpu
./scripts/run.sh avx2
```

The scripts locate Homebrew LLVM 18 automatically. To use a different LLVM
installation, set `LLVM_CONFIG` to its `llvm-config` executable; the matching
Clang compiler is then used both for the runtime and generated kernels.

Build products are kept inside the repository in `build/` and `lib/`. The
build produces the shared library `lib/libtensor.dylib`. Remove
the selected backend's build products with:

```sh
./scripts/clean.sh generic
```

## Backend status

- `generic`: portable host CPU backend; the default and recommended starting point.
- `cpu`: optimized CPU JIT backend.
- `avx2`: CPU backend compiled with AVX2 instructions; use only on supported CPUs.
- `cuda`: experimental; it requires `nvcc` and a CUDA toolkit, and currently has known issues.

The checked-in smoke test covers allocation and materialization of a lazy input
tensor. It provides a stable sanity check for a clean build; broader JIT
materialization coverage is still in progress.

Set `TENSOR_DISABLE_JIT=1` to execute supported handwritten host kernels
directly. The YOLO example uses this mode.
The larger application examples under `tests/` are retained as development
workloads and do not yet have a unified runner.
