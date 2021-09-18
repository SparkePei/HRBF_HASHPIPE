#NVCC compiler and flags
CUDA_DIR ?= /usr/local/cuda
NVCC = nvcc
CC = g++
NVCCFLAGS   = -O3 --compiler-options '-fPIC' --compiler-bindir=/usr/bin/gcc --shared -Xcompiler -Wall -arch=sm_86
# linker options
CC_FLAGS     = -g -O3 -fPIC -shared -lstdc++ -mavx -msse4 \
                     -I. -I$(CUDA_DIR)/include -I/usr/local/include \
                     -L. -L/usr/local/lib \
                     -lhashpipe -lrt -lm -lhiredis
CUDA_LDFLAGS  = -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 -lcudart -lstdc++ -lcublas
HP_LDFLAGS = -L/usr/local/lib -lhashpipe -lhashpipestatus -lrt -lm -lpthread -lhiredis

NVCC_FLAGS = $(NVCCFLAGS) $(CUDA_LDFLAGS)

# HASHPIPE
HP_LIB_TARGET   = hrbf_hashpipe.o
HP_LIB_SOURCES  = hrbf_net_thread.c \
		      hrbf_gpu_thread.c \
		      hrbf_output_thread.c \
		      hrbf_databuf.c
HP_LIB_OBJECTS = $(patsubst %.c,%.o,$(HP_LIB_SOURCES))
HP_LIB_INCLUDES = hrbf_databuf.h gpu_beamformer.h
HP_TARGET = hrbf_hashpipe.so
# GPU
GPU_LIB_TARGET = gpu_beamformer.o
GPU_LIB_SOURCES = gpu_beamformer.cu
GPU_LIB_INCLUDES =  gpu_beamformer.h

all: $(GPU_LIB_TARGET) $(HP_LIB_TARGET) $(HP_TARGET)

$(GPU_LIB_TARGET): $(GPU_LIB_SOURCES)
	$(NVCC) -c $^ $(NVCC_FLAGS)

$(HP_LIB_TARGET): $(HP_LIB_SOURCES)
	$(CC) -c $^ $(CC_FLAGS) $(CUDA_LDFLAGS)

# Link HP_OBJECTS together to make plug-in .so file
$(HP_TARGET): $(GPU_LIB_TARGET)
	$(NVCC) *.o -o $@ $(NVCC_FLAGS) $(HP_LDFLAGS)
tags:
	ctags -R .
clean:
	rm -f $(HP_LIB_TARGET) $(GPU_LIB_TARGET) $(HP_TARGET) *.o tags 

prefix=/home/peix/local
LIBDIR=$(prefix)/lib
BINDIR=$(prefix)/bin
install-lib: $(HP_TARGET)
	mkdir -p "$(LIBDIR)"
	install -p $^ "$(LIBDIR)"
install: install-lib

.PHONY: all tags clean install install-lib
# vi: set ts=8 noet :
