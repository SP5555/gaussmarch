#pragma once
#include <cuda_runtime.h>

#include "gpu_soa.h"

/**
 * @brief GPU-side SoA storage for 2D Gaussian splats in NDC space.
 *
 * The 2D covariance is symmetric, stored as upper triangle:
 *   [[cov_xx, cov_xy],
 *    [cov_xy, cov_yy]]
 */
struct Splat2DParams : GpuSoA
{
    // NDC space
    CudaBuffer<float> pos_x, pos_y, pos_z;
    CudaBuffer<float> cov_xx, cov_xy, cov_yy;
    CudaBuffer<float> color_r, color_g, color_b;
    CudaBuffer<float> opacity;

    std::vector<CudaBuffer<float>*> fields() override {
        return {&pos_x,   &pos_y,   &pos_z,
                &cov_xx,  &cov_xy,  &cov_yy,
                &color_r, &color_g, &color_b,
                &opacity};
    }
};

/**
 * @brief Gradients w.r.t. Splat2DParams.
 */
struct Splat2DGrads : GpuSoA
{
    // NDC space
    CudaBuffer<float> grad_pos_x, grad_pos_y, grad_pos_z;
    CudaBuffer<float> grad_cov_xx, grad_cov_xy, grad_cov_yy;
    CudaBuffer<float> grad_color_r, grad_color_g, grad_color_b;
    CudaBuffer<float> grad_opacity;

    std::vector<CudaBuffer<float>*> fields() override {
        return {&grad_pos_x,   &grad_pos_y,   &grad_pos_z,
                &grad_cov_xx,  &grad_cov_xy,  &grad_cov_yy,
                &grad_color_r, &grad_color_g, &grad_color_b,
                &grad_opacity};
    }
};
