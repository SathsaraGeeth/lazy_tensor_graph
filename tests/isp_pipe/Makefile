ROOT := ../..
LLVM_CONFIG ?= llvm-config-18
CC ?= clang-18
CXX ?= clang++-18
OPENMP_FLAGS ?= -fopenmp
BUILD := $(ROOT)/build/isp_pipe
BIN := $(BUILD)/isp_pipe_test

.PHONY: run clean

$(BIN): main_test.c $(ROOT)/lib/libtensor.so
	@mkdir -p $(BUILD)
	$(CC) -O2 -std=c11 -I$(ROOT)/include $(OPENMP_FLAGS) $< -o $@ \
		-L$(ROOT)/lib -ltensor -Wl,-rpath,'$$ORIGIN/../../lib' -lm

$(ROOT)/lib/libtensor.so:
	$(MAKE) -C $(ROOT) LLVM_CONFIG=$(LLVM_CONFIG) CC=$(CC) CXX=$(CXX) CLANG=$(CC) OPENMP_FLAGS='$(OPENMP_FLAGS)'

run: $(BIN)
	cd $(ROOT) && TENSOR_DISABLE_JIT=1 ./build/isp_pipe/isp_pipe_test

clean:
	rm -rf $(BUILD)
