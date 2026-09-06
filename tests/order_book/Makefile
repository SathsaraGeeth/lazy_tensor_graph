ROOT := ../..
CC ?= clang-18
CXX ?= clang++-18
LLVM_CONFIG ?= llvm-config-18
CJSON_LIB ?= -l:libcjson.so.1
NUM_INSTRUMENTS ?= 498
WARM_UP ?= 224
BUILD := $(ROOT)/build/order_book
BIN := $(BUILD)/order_book
CFLAGS := -O2 -std=c11 -I$(ROOT)/include -I$(ROOT)/include/tensor -Iutil -Ialt/cjson -fopenmp \
	-DNUM_INSTRUMENTS=$(NUM_INSTRUMENTS) -DWARM_UP=$(WARM_UP)
LDFLAGS := -L$(ROOT)/lib -ltensor -Wl,-rpath,'$$ORIGIN/../../lib' -lcurl $(CJSON_LIB) -fopenmp

.PHONY: run clean

$(BIN): main.c util/producer.c util/queue.c util/util.c $(ROOT)/lib/libtensor.so
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) main.c util/producer.c util/queue.c util/util.c -o $@ $(LDFLAGS)

$(ROOT)/lib/libtensor.so:
	$(MAKE) -C $(ROOT) LLVM_CONFIG=$(LLVM_CONFIG) CC=$(CC) CXX=$(CXX) CLANG=$(CC)

run: $(BIN)
	cd $(ROOT) && TENSOR_DISABLE_JIT=1 ./build/order_book/order_book

clean:
	rm -rf $(BUILD)
