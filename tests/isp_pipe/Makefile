ROOT := ../..
LLVM_CONFIG ?= $(shell brew --prefix llvm@18)/bin/llvm-config
CC := $(shell brew --prefix llvm@18)/bin/clang
OPENMP_FLAGS ?= -fopenmp -I$(shell brew --prefix libomp)/include -L$(shell brew --prefix libomp)/lib -Wl,-rpath,$(shell brew --prefix libomp)/lib
BUILD := $(ROOT)/build/isp_pipe
BIN := $(BUILD)/isp_pipe_test

.PHONY: run clean

$(BIN): main_test.c $(ROOT)/lib/libtensor.dylib
	@mkdir -p $(BUILD)
	$(CC) -O2 -std=c11 -I$(ROOT)/include $(OPENMP_FLAGS) $< -o $@ \
		-L$(ROOT)/lib -ltensor -Wl,-rpath,$(ROOT)/lib -lm

$(ROOT)/lib/libtensor.dylib:
	$(MAKE) -C $(ROOT) LLVM_CONFIG=$(LLVM_CONFIG) CC=$(CC) CXX=$(shell dirname $(CC))/clang++ CLANG=$(CC) OPENMP_FLAGS='$(OPENMP_FLAGS)'

run: $(BIN)
	cd $(ROOT) && TENSOR_DISABLE_JIT=1 ./build/isp_pipe/isp_pipe_test

clean:
	rm -rf $(BUILD)
