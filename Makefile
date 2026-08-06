# *********************************************************************
# Learning-CUDA Makefile
# Targets:
#   make               		: Build + run tests (default, non-verbose)
#   make build         		: Only compile (no run)
#   make run           		: Run tests (after build, non-verbose)
#   make run VERBOSE=true 	: Run tests with verbose output
#   make clean         		: Delete temporary files
# *********************************************************************

# -------------------------------
# Configuration
# -------------------------------
PLATFORM        ?= nvidia
PLATFORM_DEFINE ?= -DPLATFORM_NVIDIA
STUDENT_SUFFIX  := cu
CFLAGS          := -std=c++17 -O0
# 编译时把 student kernel 也编到本机 GPU 的 native arch，否则 nvcc 默认
# 走 sm_52 (Maxwell) → 在 RTX 30/40 上要靠 driver JIT 解释 sm_52 SASS，
# 性能极差 (~10x) 且 dot-product 类算子的精度也会被 JIT 改写，测试 #6/#14
# float 差出容差带就来自这里。SM_ARCH 默认与 GPU 一致，可用 make SM_ARCH=... 覆盖。
SM_ARCH         ?=
EXTRA_LIBS     	:=
# RUN_ENV is prepended at run time; empty by default (non-NVIDIA platforms).
RUN_ENV         :=

# The prebuilt tester/tester_nv.o embeds PTX 8.5 / sm_52 SASS and references
# CUDA 12.x-only runtime symbols (cudaGetDeviceProperties_v2).  Pin the
# toolchain to CUDA 12.2 (the one matching the installed nvidia driver 535.x)
# and let the runtime resolve the PTX issue.  Override with
# `make NVCC_DIR=/path/to/cuda-X.Y` if your driver is different.
#
# NOTE: we deliberately do NOT honor the shell's $CUDA_HOME here — many
# users (and CI images) have CUDA_HOME pointing elsewhere for other work.
NVCC_DIR        ?= /usr/local/cuda-12.2

# Compiler & Tester object selection based on PLATFORM
ifeq ($(PLATFORM),nvidia)
    ifneq ($(shell test -x $(NVCC_DIR)/bin/nvcc && echo yes),yes)
        $(error CUDA toolkit not found at $(NVCC_DIR)/bin/nvcc; install CUDA 12.x or override with `make NVCC_DIR=/path/to/cuda-X.Y`)
    endif
    CC              := $(NVCC_DIR)/bin/nvcc
    TEST_OBJ    	:= tester/tester_nv.o
	PLATFORM_DEFINE := -DPLATFORM_NVIDIA
	EXTRA_LIBS		:= -L$(NVCC_DIR)/lib64 -lcudart
	# Use the same cudart for run-time; without this the dynamic loader may
	# pick a different CUDA's libcudart and break ABI/version checks.
	RUN_ENV		:= LD_LIBRARY_PATH=$(NVCC_DIR)/lib64:$$LD_LIBRARY_PATH
	# Default to native arch (matches the tester's sm_86 fatbin).  Override
	# with `make SM_ARCH=sm_89` (RTX 4090) etc.  Empty SM_ARCH is treated as
	# "compile for sm_86" so the student kernel runs natively instead of being
	# JIT'd from the nvcc default sm_52 (which makes dot-product float tests
	# #6/#14 fail by ~1.5x and runs ~10x slower).
	SM_ARCH_FLAG := $(if $(SM_ARCH),-arch=$(SM_ARCH),-arch=sm_86)
	# Append to CFLAGS (must be after SM_ARCH_FLAG is defined).
	CFLAGS          += $(SM_ARCH_FLAG)
else ifeq ($(PLATFORM),iluvatar)
    CC          	:= clang++
	CFLAGS          := -std=c++17 -O3
    TEST_OBJ    	:= tester/tester_iluvatar.o
	PLATFORM_DEFINE := -DPLATFORM_ILUVATAR
	EXTRA_LIBS		:= -lcudart -I/usr/local/corex/include -L/usr/local/corex/lib64 -fPIC 
else ifeq ($(PLATFORM),moore)
    CC          	:= mcc
	CFLAGS          := -std=c++11 -O3
    TEST_OBJ    	:= tester/tester_moore.o
	STUDENT_SUFFIX  := mu
	PLATFORM_DEFINE := -DPLATFORM_MOORE
	EXTRA_LIBS		:= -I/usr/local/musa/include -L/usr/lib/gcc/x86_64-linux-gnu/11/ -L/usr/local/musa/lib -lmusart
else ifeq ($(PLATFORM),metax)
    CC          	:= mxcc
    TEST_OBJ    	:= tester/tester_metax.o
	STUDENT_SUFFIX  := maca
	PLATFORM_DEFINE := -DPLATFORM_METAX
else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, iluvatar, moore, metax))
endif

# Executable name
TARGET          	:= test_kernels
# Kernel implementation
STUDENT_SRC     	:= src/kernels.$(STUDENT_SUFFIX) 
# Compiled student object (auto-generated)
STUDENT_OBJ  		:= $(addsuffix .o,$(basename $(STUDENT_SRC)))
# Tester's actual verbose argument (e.g., --verbose, -v)
TEST_VERBOSE_FLAG 	:= --verbose
# User-provided verbose mode (true/false; default: false)
VERBOSE         	:=  

# -------------------------------
# Process User Input (VERBOSE → Tester Flag)
# -------------------------------
# Translates `VERBOSE=true` (case-insensitive) to the tester's verbose flag.
# If VERBOSE is not "true" (or empty), no flag is passed.
VERBOSE_ARG := $(if $(filter true True TRUE, $(VERBOSE)), $(TEST_VERBOSE_FLAG), )

# -------------------------------
# Phony Targets
# -------------------------------
.PHONY: all build run clean

# Default target: Build + run tests (non-verbose)
all: build run

# Build target: Compile student code + link with test logic
build: $(TARGET)

# Run target: Execute tests (supports `VERBOSE=true` for verbose output)
# RUN_ENV is set per-platform (e.g. LD_LIBRARY_PATH for NVIDIA) so the
# executable finds a cudart compatible with the prebuilt tester.
run: $(TARGET)
	@echo "=== Running tests (output from $(STUDENT_OBJ)) ==="
	@# Show verbose mode status (friendly for users)
	@if [ -n "$(VERBOSE_ARG)" ]; then \
	    echo "=== Verbose mode: Enabled (using '$(TEST_VERBOSE_FLAG)') ==="; \
	else \
	    echo "=== Verbose mode: Disabled ==="; \
	fi
	$(RUN_ENV) ./$(TARGET) $(VERBOSE_ARG)

# Clean target: Delete temporary files (executable + src object)
clean:
	@echo "=== Cleaning temporary files ==="
	rm -f $(TARGET) $(STUDENT_OBJ)

# -------------------------------
# Dependency Rules (Core Logic)
# -------------------------------
# Generate executable: Link kernel code (kernels.o) with test logic (tester.o)
$(TARGET): $(STUDENT_OBJ) $(TEST_OBJ)
	@echo "=== Linking executable (student code + test logic) ==="
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -o $@ $^ $(EXTRA_LIBS)

# Generate src object: Compile kernels.cu (triggers template instantiation)
$(STUDENT_OBJ): $(STUDENT_SRC)
	@echo "=== Compiling student code ($(STUDENT_SRC)) ==="
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -c $< -o $@
