# Lazy Tensor Graph

This repository contains a graph-based lazy tensor runtime.

For architecture details, features, and additional project information, see
the [Tensor Library project page](https://geethsathsara.com/projects/tensor_library/).

## Quick start

Prerequisites:

- Python 3
- GNU Make
- A C/C++ compiler with OpenMP support
- LLVM 18

Build the shared library.

```sh
./scripts/build.sh generic  # generic
./scripts/build.sh cpu      # optimized CPU (default)
```


## Backend status

- `generic`: Portable host CPU backend; the baseline.
- `cpu`: Optimized CPU JIT backend.

Note: A CUDA JIT backend exists, but it has known issues, so do not use it.

Note: Set `TENSOR_DISABLE_JIT=1` to support older examples written before JIT.
Note: There is a cuda JIT backend, it has known issues so dont use it.

## Examples

Build the shared library once with `./scripts/build.sh`.


Then run any example from the repository root.

### ISP pipeline

Processes the test image and writes a crop plus green-channel image into
`tests/isp_pipe/processed/`.

```sh
make -C tests/isp_pipe run
```

### YOLOv5n inference

Runs YOLOv5n and writes the annotated image to `tests/yolo/imgs/output_detected.jpg`.

```sh
make -C tests/yolo run
```

### Live order book pipeline

Requires `cjson`.

Connects to Binance, initializes the active order books, and runs the
order-book tensor graph.

```sh
make -C tests/order_book run
```
