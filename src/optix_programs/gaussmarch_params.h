#pragma once

// Shared between host C++ and the OptiX device shader (gaussmarch.cu).
// No host-only includes.

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ===== Constants =====
constexpr float PT_PI = 3.14159265358979323846f;

// ===== Tuning =====
constexpr uint32_t GAUSSIANS_PER_LEAF = 1;

// ===== Payload register layout (3 registers) =====
//   0 : count      (uint32)
//   1 : density    (float bits)  = sum(w_i)
//   2 : scalar_num (float bits)  = sum(w_i * scalar_i)

// ===== Attribute register layout (2 registers) =====
//   0 : scalar  (float bits, gaussian scalar value)
//   1 : weight  (float bits, opacity * exp(-0.5 * d2_mahal))

// Per-Gaussian GPU record.  12 floats = 48 bytes (16-byte aligned).
struct GpuGaussian {
    float mu_x, mu_y, mu_z;
    float opacity;
    // Sigma^-1 upper triangle: Sigma^-1 = R * diag(1/s^2) * R^T
    float s00, s01, s02, s11, s12, s22;
    float scalar;
    float pad;
};

struct GaussianLaunchParams {
    // Camera
    float3 cam_pos;
    float3 dir_00;   // unnormalized direction to top-left corner
    float3 dir_du;   // full horizontal span
    float3 dir_dv;   // full vertical span

    // BVH
    OptixTraversableHandle traversable;

    // Gaussian data (device pointer)
    GpuGaussian *gaussians;
    uint32_t     num_gaussians;

    // Marching
    float step_size;
    float shadow_step_size;
    int   max_depth;

    // Transfer function (scalar -> RGBA)
    cudaTextureObject_t colormap;

    // Scene bounding box (union of all Gaussian AABBs)
    float3 scene_min;
    float3 scene_max;

    // Lighting
    float3 light_dir;     // unit vector toward light
    float  light_ambient; // [0,1]; 1.0 = no shadows

    // Output
    float  *output;   // RGB float, width * height * 3
    float4 *accum;    // RGBA accumulation buffer, width * height

    int width;
    int height;
    int accum_id;
    int frame_id;

    // Blue noise (STBN)
    cudaTextureObject_t stbn_tex;
    int use_blue_noise;
};
