#pragma once
#include <vector>

#include "../cuda/cuda_buffer.h"
#include "../utils/sh_consts.h"
#include "gpu_soa.h"

/* ===== ===== Gaussian3D (CPU) ===== ===== */

/**
 * @brief CPU-side structure for a single 3D Gaussian splat.
 */
struct Gaussian3D
{
    float pos_x,   pos_y,   pos_z;
    float scale_x, scale_y, scale_z;        // log-scale
    float rot_w,   rot_x,   rot_y,   rot_z; // unit quaternion
    float sh_dc_r, sh_dc_g, sh_dc_b;        // DC SH coefficients (degree 0)
    float sh_rest[45] = {};                 // higher-order SH, bands 0-14 x 3 channels (channel-first: R then G then B)
    float logit_opacity;
};

/**
 * @brief GPU-side Gaussian parameters, stored as struct-of-arrays
 * for coalesced memory access.
 *
 * Inherits GpuSoA: fields() lists all buffers; zero() and allocate(n) are provided.
 * allocate(n) defaults to sh_degree=0; use allocate(n, sh_degree) to include SH bands.
 * sh_rest buffers are sized n * sh_num_bands and are safe to include in fields() since
 * CudaBuffer::zero() is a no-op on unallocated buffers.
 *
 * CPU <-> GPU transfer: see src/utils/gaussian3d_io.h
 */
struct Gaussian3DParams : GpuSoA
{
    CudaBuffer<float> pos_x,   pos_y,   pos_z;
    CudaBuffer<float> scale_x, scale_y, scale_z;
    CudaBuffer<float> rot_w,   rot_x,   rot_y,   rot_z;
    CudaBuffer<float> sh_dc_r, sh_dc_g, sh_dc_b;
    // Higher-order SH: flat buffers of size n * sh_num_bands each.
    // Layout within each buffer: [band0_splat0..splat(n-1) | band1_splat0.. | ...]
    // Access in kernels: sh_rest_r[band * count + i]
    CudaBuffer<float> sh_rest_r, sh_rest_g, sh_rest_b;
    CudaBuffer<float> logit_opacity;
    int sh_num_bands = 0;

    std::vector<CudaBuffer<float>*> fields() override {
        return {&pos_x,   &pos_y,   &pos_z,
                &scale_x, &scale_y, &scale_z,
                &rot_w,   &rot_x,   &rot_y,   &rot_z,
                &sh_dc_r, &sh_dc_g, &sh_dc_b,
                &sh_rest_r,  &sh_rest_g,  &sh_rest_b,
                &logit_opacity};
    }

    void allocate(int n) override { allocate(n, 0); }
    void allocate(int n, int sh_degree)
    {
        pos_x.allocate(n);   pos_y.allocate(n);   pos_z.allocate(n);
        scale_x.allocate(n); scale_y.allocate(n); scale_z.allocate(n);
        rot_w.allocate(n);   rot_x.allocate(n);   rot_y.allocate(n);   rot_z.allocate(n);
        sh_dc_r.allocate(n); sh_dc_g.allocate(n); sh_dc_b.allocate(n);

        sh_num_bands = sh_degree_to_bands(sh_degree);
        if (sh_num_bands > 0) {
            sh_rest_r.allocate(n * sh_num_bands);
            sh_rest_g.allocate(n * sh_num_bands);
            sh_rest_b.allocate(n * sh_num_bands);
        }
        logit_opacity.allocate(n);

        count = n;
    }
};

/**
 * @brief Gradients w.r.t. Gaussian3DParams.
 *
 * Same GpuSoA pattern as Gaussian3DParams: fields() lists all grad buffers,
 * allocate(n) defaults to sh_degree=0, allocate(n, sh_degree) includes SH bands.
 */
struct Gaussian3DGrads : GpuSoA
{
    CudaBuffer<float> grad_pos_x,   grad_pos_y,   grad_pos_z;
    CudaBuffer<float> grad_scale_x, grad_scale_y, grad_scale_z;
    CudaBuffer<float> grad_rot_w,   grad_rot_x,   grad_rot_y,   grad_rot_z;
    CudaBuffer<float> grad_sh_dc_r, grad_sh_dc_g, grad_sh_dc_b;
    CudaBuffer<float> grad_sh_rest_r, grad_sh_rest_g, grad_sh_rest_b;
    CudaBuffer<float> grad_logit_opacity;
    int sh_num_bands = 0;

    std::vector<CudaBuffer<float>*> fields() override {
        return {&grad_pos_x,   &grad_pos_y,   &grad_pos_z,
                &grad_scale_x, &grad_scale_y, &grad_scale_z,
                &grad_rot_w,   &grad_rot_x,   &grad_rot_y,   &grad_rot_z,
                &grad_sh_dc_r, &grad_sh_dc_g, &grad_sh_dc_b,
                &grad_sh_rest_r,  &grad_sh_rest_g,  &grad_sh_rest_b,
                &grad_logit_opacity};
    }

    // Proper override so the base virtual is satisfied.
    // sh_degree defaults to 0 (no SH); use allocate(n, sh_degree) to include SH bands.
    void allocate(int n) override { allocate(n, 0); }
    void allocate(int n, int sh_degree)
    {
        grad_pos_x.allocate(n);   grad_pos_y.allocate(n);   grad_pos_z.allocate(n);
        grad_scale_x.allocate(n); grad_scale_y.allocate(n); grad_scale_z.allocate(n);
        grad_rot_w.allocate(n);   grad_rot_x.allocate(n);   grad_rot_y.allocate(n);   grad_rot_z.allocate(n);
        grad_sh_dc_r.allocate(n); grad_sh_dc_g.allocate(n); grad_sh_dc_b.allocate(n);

        sh_num_bands = sh_degree_to_bands(sh_degree);
        if (sh_num_bands > 0) {
            grad_sh_rest_r.allocate(n * sh_num_bands);
            grad_sh_rest_g.allocate(n * sh_num_bands);
            grad_sh_rest_b.allocate(n * sh_num_bands);
        }
        grad_logit_opacity.allocate(n);
        
        count = n;
    }
};