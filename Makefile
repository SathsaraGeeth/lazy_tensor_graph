# Makefile for Tensor Library
.DEFAULT_GOAL := all

TARGET ?= host
TENSOR_BACKEND ?= generic
PYTHON ?= python3
LLVM_CONFIG ?= llvm-config-18
CLANG ?= clang-18
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LIB_DIR ?= $(ROOT_DIR)/lib
OPENMP_FLAGS ?= -fopenmp
SHARED_LIBRARY ?= $(LIB_DIR)/libtensor.dylib

SRCS = src/core/tensor.c \
       src/graph/graph.c \
       src/graph/sketch.c \
       src/graph/walker.c \
       src/graph/fusion.c \
       src/graph/jit.c \
       src/graph/do.c \
       src/jit/tensor_jit.c \
       src/kernels/imgproc/brightness/api.c \
       src/kernels/imgproc/lut/api.c \
       src/kernels/imgproc/histogram/api.c \
       src/kernels/imgproc/blc/api.c src/kernels/imgproc/dpc/api.c \
       src/kernels/imgproc/awb/api.c src/kernels/imgproc/demosaicing/api.c \
       src/kernels/imgproc/ccm/api.c \
       src/kernels/imgproc/blur/api.c src/kernels/imgproc/blend_rgb2yuv/api.c \
       src/kernels/imgproc/chroma_subsample/api.c \
       src/kernels/primitive/copy/api.c \
       src/kernels/primitive/cast/api.c \
       src/kernels/primitive/reshape/api.c \
       src/kernels/primitive/permute/api.c \
       src/kernels/primitive/slice/api.c \
       src/kernels/primitive/expand/api.c \
       src/kernels/linalg/matrix_mul/api.c \
       src/kernels/linalg/matrix_inv_square/api.c \
       src/kernels/linalg/conv2D/api.c \
       src/kernels/finance/microprice/api.c \
       src/kernels/finance/orderbook_imbalance/api.c \
       src/kernels/finance/slippage/api.c \
       src/kernels/dnn/conv/host.c src/kernels/dnn/activations/host.c \
       src/kernels/dnn/concat/host.c src/kernels/dnn/pooling/host.c \
       src/kernels/dnn/upsample/host.c src/kernels/dnn/resize_pad_norm/host.c

BUILD_DIR ?= $(ROOT_DIR)/build/tensor/$(TENSOR_BACKEND)
GENERATED_DIR := $(BUILD_DIR)/generated
OP_FILES := $(shell find src/kernels -name '*.op' -type f | sort)
GENERATED_BC := $(GENERATED_DIR)/operations.bc
GENERATED_REGISTRY := $(GENERATED_DIR)/registry.c
GENERATED_STAMP := $(GENERATED_DIR)/.ops-generated
JIT_CXX_SRCS :=

ifeq ($(TENSOR_BACKEND),generic)
SRCS += src/memory/cpu/memory.c
SRCS += src/jit/exec/generic/exec.c src/jit/exec/generic/memory.c src/jit/exec/generic/kernel.c \
        src/jit/exec/generic/fusion.c
JIT_CXX_SRCS += src/jit/gen/generic/orc_jit.cpp
OP_DEVICE := generic
else ifeq ($(TENSOR_BACKEND),cpu)
SRCS += src/memory/cpu/memory.c
SRCS += src/jit/exec/cpu/exec.c src/jit/exec/cpu/memory.c src/jit/exec/cpu/kernel.c src/jit/exec/cpu/fusion.c
JIT_CXX_SRCS += src/jit/gen/cpu/orc_jit.cpp
JIT_CXX_SRCS += src/jit/gen/cpu/fusion/fusion.cpp
JIT_CXX_SRCS += src/jit/gen/cpu/abstract_parallel/abstract_parallel.cpp
JIT_CXX_SRCS += src/jit/gen/cpu/abstract_vec/abstract_vec.cpp
OP_DEVICE := cpu
BACKEND_CFLAGS += -DTENSOR_JIT_CPU_BACKEND=1 -DBACKEND_CPU=1
BACKEND_CXXFLAGS += -DBACKEND_CPU=1
else
$(error Unsupported TENSOR_BACKEND '$(TENSOR_BACKEND)'; use generic or cpu)
endif

SRCS += src/kernels/primitive/jit.c

OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS)) \
       $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(JIT_CXX_SRCS)) \
       $(GENERATED_DIR)/registry.o
DEPS = $(OBJS:.o=.d)

# Objects created before dependency-file support may not have a matching .d
# file. Keep ABI-bearing headers and backend selection as explicit fallback
# prerequisites so an existing build directory cannot silently mix layouts.
$(OBJS): Makefile include/tensor_jit.h

ifeq ($(TARGET),host)
CC ?= cc
CXX ?= c++
SRCS += src/dev_tools/profiler/runtime/host.c
CFLAGS = -Wall -Wextra -O3 -march=native -fPIC $(OPENMP_FLAGS) -I$(ROOT_DIR)/include -I. -Iinclude
CXXFLAGS = -O3 -march=native -iquote $(ROOT_DIR)/include -I. -Iinclude \
           $(shell $(LLVM_CONFIG) --cxxflags)
else
SRCS += src/dev_tools/profiler/runtime/generic.c
LLVM_BIN = /mnt/fileserver/prj/dmctp/compiler/llvm-project/build/bin
CC = $(LLVM_BIN)/clang --target=riscv32
LLC = $(LLVM_BIN)/llc
LD = $(LLVM_BIN)/ld.lld
AR = $(LLVM_BIN)/llvm-ar
CFLAGS = -march=rv32im_zalrsc -mabi=ilp32 -ffreestanding -nostdlib -fno-builtin -O1 -Wall -I$(ROOT_DIR)/include -I. -Iinclude
endif

CFLAGS += $(BACKEND_CFLAGS)
CFLAGS += -MMD -MP
CXXFLAGS += -MMD -MP
CXXFLAGS += $(BACKEND_CXXFLAGS)

-include $(DEPS)

.PHONY: FORCE

all: $(SHARED_LIBRARY)

$(SHARED_LIBRARY): $(OBJS) FORCE
	mkdir -p $(LIB_DIR)
	rm -f $@
	$(CXX) -dynamiclib -o $@ $(OBJS) $(shell $(LLVM_CONFIG) --ldflags --libs orcjit native core) $(OPENMP_FLAGS)

$(GENERATED_STAMP): $(OP_FILES) $(wildcard src/op_parser/*.py)
	@mkdir -p $(dir $@)
	CLANG="$(CLANG)" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) src/op_parser/op_parser.py \
		--device $(OP_DEVICE) src/kernels $(GENERATED_DIR)
	@touch $@

$(GENERATED_BC) $(GENERATED_REGISTRY): $(GENERATED_STAMP)
	@test -f $@

$(GENERATED_DIR)/%.o: $(GENERATED_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
ifeq ($(TARGET),host)
	$(CC) $(CFLAGS) -c $< -o $@
else
	$(CC) $(CFLAGS) -S -emit-llvm $< -o $(BUILD_DIR)/$*.ll
	$(LLC) -march=dmctp32 -mattr=+m -filetype=obj $(BUILD_DIR)/$*.ll -o $@
endif

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(ROOT_DIR)/bin/test_tensor $(SHARED_LIBRARY)
