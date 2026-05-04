#pragma once
#include <vector>
#include "../cuda/cuda_buffer.h"

/**
 * @brief Base for GPU struct-of-arrays types.
 *
 * Subclasses implement fields() to list their CudaBuffers.
 * allocate(n) and zero() are then provided automatically.
 *
 * Convention for new types:
 * ```
 *  struct MyParams : GpuSoA {
 *      CudaBuffer<float> a, b, c;
 *      std::vector<CudaBuffer<float>*> fields() override {
 *          return {&a, &b, &c};
 *      }
 *  };
 * ```
 */
struct GpuSoA {
    int count = 0;

    virtual std::vector<CudaBuffer<float>*> fields() = 0;

    virtual void allocate(int n) {
        for (auto* f : fields()) f->allocate(n);
        count = n;
    }
    void zero() {
        for (auto* f : fields()) f->zero();
    }

    virtual ~GpuSoA() = default;
};
