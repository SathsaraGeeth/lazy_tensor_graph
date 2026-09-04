# ─── Config ───────────────────────────────────────────────────────────────────

CC      ?= cc
OPENMP_FLAGS ?= -fopenmp
CFLAGS  := -Wall -Wextra -g -O0 -std=c11
CFLAGS  += -I../../include
CFLAGS  += $(OPENMP_FLAGS)
CFLAGS  += -DENABLE_GRAPH_TRACE

ROOT    := ../..
LIB     := $(ROOT)/lib/libtensor.dylib
BUILD   := ../../build/yolo
BIN     := $(BUILD)/yolo_test

# ─── Sources ──────────────────────────────────────────────────────────────────

SRCS    := main.c forward.c lazy_ops.c postprocess.c
OBJS    := $(patsubst %.c, $(BUILD)/%.o, $(SRCS))

# ─── Rules ────────────────────────────────────────────────────────────────────

.PHONY: all run clean cleanall lib

all: lib $(BIN)

lib:
	$(MAKE) -C $(ROOT)

$(BIN): $(OBJS) $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ -o $@ -L$(ROOT)/lib -ltensor -Wl,-rpath,$(ROOT)/lib -lm $(OPENMP_FLAGS)
	@echo ">>> Built: $@"

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "  CC $<"

run: all
	@echo ">>> Running..."
	cd $(ROOT) && ./build/yolo/yolo_test

clean:
	rm -rf $(BUILD)

cleanall: clean
	$(MAKE) -C $(ROOT) clean
