CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDLIBS ?= -lm
OUT_DIR ?= $(abspath $(CURDIR)/../../../../../venv/build/profiler)
TARGET := $(OUT_DIR)/tensor-profiler
PROFILER_CFLAGS := $(filter-out -MMD -MP,$(CFLAGS)) -I../../../include

.PHONY: all clean

all: $(TARGET)

$(TARGET): tool/main.c
	mkdir -p $(OUT_DIR)
	$(CC) $(PROFILER_CFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET)
