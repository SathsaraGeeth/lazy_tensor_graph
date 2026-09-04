# ─── Config ───────────────────────────────────────────────────────────────────

CC      := gcc
CFLAGS  := -Wall -Wextra -g -O0 -std=c11
CFLAGS  += -I../tensor
CFLAGS  += -I../tensor/include
CFLAGS  += -I../../include
CFLAGS  += -fopenmp
CFLAGS  += -DENABLE_GRAPH_TRACE

ROOT    := ../tensor
LIB     := $(ROOT)/../../lib/libtensor.a
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
	$(CC) $(CFLAGS) $^ -o $@ -lm -fopenmp
	@echo ">>> Built: $@"

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "  CC $<"

run: all
	@echo ">>> Running..."
	./$(BIN)

clean:
	rm -rf $(BUILD)

cleanall: clean
	$(MAKE) -C $(ROOT) clean
