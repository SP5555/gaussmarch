#!/bin/bash

# always add cuda to PATH first
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# verify nvcc
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found even after adding /usr/local/cuda/bin to PATH"
    exit 1
fi

# check OptiX
if [ -z "$OPTIX_INSTALL_DIR" ]; then
    echo "ERROR: OPTIX_INSTALL_DIR is not set."
    echo "  export OPTIX_INSTALL_DIR=/path/to/OptiX"
    echo "  Or pass it directly: cmake .. -DOptiX_INSTALL_DIR=/path/to/OptiX"
    exit 1
fi

NVCC_PATH=$(which nvcc)
echo "Using nvcc: $NVCC_PATH"
echo "nvcc version: $(nvcc --version | grep release)"
echo "OptiX: $OPTIX_INSTALL_DIR"

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=$NVCC_PATH \
    -DOptiX_INSTALL_DIR="$OPTIX_INSTALL_DIR"
cmake --build . --parallel $(nproc)
