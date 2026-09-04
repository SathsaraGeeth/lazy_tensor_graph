ROOT := ../..
CC := $(shell brew --prefix llvm@18)/bin/clang
LLVM_CONFIG := $(shell brew --prefix llvm@18)/bin/llvm-config
CURL_PREFIX := $(shell brew --prefix curl)
CJSON_PREFIX := $(shell brew --prefix cjson)
OMP_PREFIX := $(shell brew --prefix libomp)
BUILD := $(ROOT)/build/order_book
BIN := $(BUILD)/order_book
CFLAGS := -O2 -std=c11 -I$(ROOT)/include -I$(ROOT)/include/tensor -Iutil -Ialt/cjson \
	-I$(CURL_PREFIX)/include -fopenmp -I$(OMP_PREFIX)/include
LDFLAGS := -L$(ROOT)/lib -ltensor -Wl,-rpath,$(ROOT)/lib -L$(CURL_PREFIX)/lib -lcurl \
	-L$(CJSON_PREFIX)/lib -lcjson -L$(OMP_PREFIX)/lib -lomp -Wl,-rpath,$(OMP_PREFIX)/lib

.PHONY: run clean

$(BIN): main.c util/producer.c util/queue.c util/util.c $(ROOT)/lib/libtensor.dylib
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(ROOT)/lib/libtensor.dylib:
	$(MAKE) -C $(ROOT) LLVM_CONFIG=$(LLVM_CONFIG) CC=$(CC) CXX=$(shell dirname $(CC))/clang++ CLANG=$(CC)

run: $(BIN)
	cd $(ROOT) && TENSOR_DISABLE_JIT=1 ./build/order_book/order_book

clean:
	rm -rf $(BUILD)
