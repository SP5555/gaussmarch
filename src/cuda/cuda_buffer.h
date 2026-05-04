#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "cuda_check.h"

template <typename T>
struct CudaBuffer
{
    T *ptr = nullptr;
    size_t size = 0;

    CudaBuffer() = default;

    explicit CudaBuffer(size_t size_) { allocate(size_); }

    ~CudaBuffer() { free(); }

    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    CudaBuffer(CudaBuffer &&other) noexcept
        : ptr(other.ptr), size(other.size)
    {
        other.ptr = nullptr;
        other.size = 0;
    }
    CudaBuffer &operator=(CudaBuffer &&other) noexcept
    {
        if (this != &other)
        {
            free();
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void allocate(size_t size_)
    {
        free();
        size = size_;
        cudaError_t err = cudaMalloc(&ptr, size * sizeof(T));
        if (err != cudaSuccess)
        {
            size = 0;
            throw std::runtime_error(
                std::string("[CUDA] cudaMalloc failed (")
                + std::to_string(size_ * sizeof(T) / (1024 * 1024)) + " MB): "
                + cudaGetErrorString(err));
        }
    }

    void free()
    {
        if (ptr)
        {
            CUDA_WARN(cudaFree(ptr));
            ptr = nullptr;
            size = 0;
        }
    }

    void zero()
    {
        if (ptr)
        {
            CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(T)));
        }
    }

    operator T *() { return ptr; }
    operator const T *() const { return ptr; }
};
