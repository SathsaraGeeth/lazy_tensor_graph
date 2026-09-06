# Lazy Tensor Graph

This repository contains a graph-based lazy tensor runtime.

For architecture details, features, and additional project information, see
the [Tensor Library project page](https://geethsathsara.com/projects/tensor_library/).

## Quick start

Prerequisites:

- Computer with Linux
- Python 3
- GNU Make
- A C/C++ compiler with OpenMP support
- LLVM 18
- cjson


Note: This runtime currently targets Linux. Support for other operating systems is planned; this is a current limitation.


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

## Examples

Build the shared library once with `./scripts/build.sh`.


Then run any example from the repository root.

### Built-in profiler

Build the profiler once

```sh
make -C src/dev_tools/profiler OUT_DIR="$PWD/build/profiler"
```

### ISP pipeline

Processes the test image and writes a crop plus green-channel image into
`tests/isp_pipe/processed/`.

```sh
make -C tests/isp_pipe run
```

Run the profile and save its events:

```sh
./build/profiler/tensor-profiler run --all --trace build/isp-profile.csv -- \
  ./build/isp_pipe/isp_pipe_test
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

For a quick local run, reduce both the symbol count and warm-up period:

```sh
make -C tests/order_book clean
make -C tests/order_book NUM_INSTRUMENTS=1 WARM_UP=1
./build/order_book/order_book
```


### More profiler options

Generate selected reports later from the saved events:

```sh
./build/profiler/tensor-profiler report --time --jit --memory build/isp-profile.csv
./build/profiler/tensor-profiler report --graph --compute build/isp-profile.csv
./build/profiler/tensor-profiler report --roofline --flame build/isp-profile.csv
```

Profile Linux CPU hardware counters:

```sh
./build/profiler/tensor-profiler arch linux-cpu stat -- \
  ./build/isp_pipe/isp_pipe_test
```

Attach for 10 seconds to a repeated local ISP workload. Build it first, then
start it with shared profiling enabled:

```sh
make -C tests/isp_pipe
TENSOR_PROFILE_SHM=1 ./build/isp_pipe/isp_pipe_test 1000 &
isp_pid=$!
./build/profiler/tensor-profiler attach --all "$isp_pid" 10
```

Stop it afterward with `kill "$isp_pid"`.
