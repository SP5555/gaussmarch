#pragma once
#include <vector>
#include "../cuda/cuda_buffer.h"

#define EPSILON 1e-8f

struct AdamParams {
    float beta1   = 0.9f;
    float beta2   = 0.999f;
    float epsilon = EPSILON;
};

/**
 * @brief One parameter group: a (param, grad) pair with its own lr and moment buffers.
 *
 * The raw GPU pointers are borrowed from the caller's CudaBuffers and must
 * remain valid for the optimizer's lifetime. Moment buffers are owned here.
 */
struct AdamGroup {
    float*            param;
    const float*      grad;
    int               n;
    float             lr;
    CudaBuffer<float> m, v;

    AdamGroup(float* p, const float* g, int n_, float lr_)
        : param(p), grad(g), n(n_), lr(lr_)
    {
        m.allocate(n_); m.zero();
        v.allocate(n_); v.zero();
    }
    AdamGroup(AdamGroup&&)            = default;
    AdamGroup& operator=(AdamGroup&&) = default;
    AdamGroup(const AdamGroup&)            = delete;
    AdamGroup& operator=(const AdamGroup&) = delete;
};

/**
 * @brief Generic Adam optimizer that operates over a list of parameter groups.
 *
 * Knows nothing about what the parameters represent -- callers register groups
 * via addGroup() before the first step. Each group carries its own learning rate.
 *
 * Usage:
 * ```
 *  Adam adam;
 *  adam.init({.beta1=0.9f, .beta2=0.999f});
 *  adam.addGroup(param_buf, grad_buf, n, lr);   // repeat for each param
 *  adam.step();  // call every iteration
 * ```
 */
class Adam {
public:
    void init(const AdamParams& p = {}) { hp = p; }

    void addGroup(CudaBuffer<float>& param, const CudaBuffer<float>& grad, int n, float lr)
    {
        groups.emplace_back(param.ptr, grad.ptr, n, lr);
    }

    void setLr(int idx, float lr) { groups[idx].lr = lr; }

    void step();

    int               getStepCount() const { return step_count; }
    const AdamParams& getParams()    const { return hp; }

private:
    AdamParams             hp;
    std::vector<AdamGroup> groups;
    int                    step_count = 0;
};
