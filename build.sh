#!/bin/bash

# always add cuda to PATH first
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# now verify
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found even after adding /usr/local/cuda/bin to PATH"
    exit 1
fi

NVCC_PATH=$(which nvcc)
echo "Using nvcc: $NVCC_PATH"
echo "nvcc version: $(nvcc --version | grep release)"

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CUDA_ARCHITECTURES=86;89" \
    -DCMAKE_CUDA_COMPILER=$NVCC_PATH
cmake --build . --parallel $(nproc)