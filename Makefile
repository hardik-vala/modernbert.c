.PHONY: install-deps export-tokenizer export-model compile compiledebug compilefast clean

# Python virtualenv directory
VENV_DIR = venv

# Use venv Python/pip if it exists, otherwise use global
ifeq ($(wildcard $(VENV_DIR)/bin/python),)
	PYTHON = python3
	PIP = pip3
else
	PYTHON = $(VENV_DIR)/bin/python
	PIP = $(VENV_DIR)/bin/pip
endif

# Install Python dependencies

INSTALL_FLAGS = -qqq

install-deps:
	$(PIP) install $(INSTALL_FLAGS) --upgrade pip
	$(PIP) install $(INSTALL_FLAGS) -r requirements.txt

# Export tokenizer to binary format
export-tokenizer: install-deps
	$(PYTHON) tokenizer/export.py -t tokenizer/tokenizer.json

# Export model weights to binary format
export-model-base: install-deps
	$(PYTHON) model/export.py answerdotai/ModernBERT-base --output model/base.bin

export-model-tokclf: install-deps
	$(PYTHON) model/export.py ai4privacy/llama-ai4privacy-english-anonymiser-openpii --output model/tokclf.bin --tokclf --n_labels 3

# Compile run.c

# C compiler, e.g. gcc/clang
CC = gcc
CFLAGS = -Wall -Wextra -Isrc
LIBS = -lm -lpcre2-8 -lopenblas

# Source directory
SRC_DIR = src

# Object files
OBJS = run.o tokenizer.o

# Release build
compile: $(SRC_DIR)/tokenizer.c $(SRC_DIR)/tokenizer.h $(SRC_DIR)/run.c
	$(CC) $(CFLAGS) -O3 -c $(SRC_DIR)/tokenizer.c
	$(CC) $(CFLAGS) -O3 -c $(SRC_DIR)/run.c
	$(CC) -O3 -o run $(OBJS) $(LIBS)

# Debug build
compiledebug: $(SRC_DIR)/tokenizer.c $(SRC_DIR)/tokenizer.h $(SRC_DIR)/run.c
	$(CC) $(CFLAGS) -O0 -pg -c $(SRC_DIR)/tokenizer.c
	$(CC) $(CFLAGS) -O0 -pg -c $(SRC_DIR)/run.c
	$(CC) -O0 -pg -o run $(OBJS) $(LIBS)

# Fast build
compilefast: $(SRC_DIR)/tokenizer.c $(SRC_DIR)/tokenizer.h $(SRC_DIR)/run.c
	$(CC) $(CFLAGS) -Ofast -c $(SRC_DIR)/tokenizer.c
	$(CC) $(CFLAGS) -Ofast -c $(SRC_DIR)/run.c
	$(CC) -Ofast -o run $(OBJS) $(LIBS)

# Clean build artifacts
clean:
	rm -f run *.o
