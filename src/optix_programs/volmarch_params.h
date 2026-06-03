#pragma once

// Shared between host C++ and the OptiX device shader (volmarch.cu).
// No host-only includes.

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

struct VolumeLaunchParams {
    // Camera
    float3 cam_pos;
    float3 dir_00;   // unnormalized direction to top-left corner
    float3 dir_du;   // full horizontal span
    float3 dir_dv;   // full vertical span

    // 3D volume texture (float, normalized coords, trilinear)
    cudaTextureObject_t vol_tex;

    // Marching
    float step_size;
    float shadow_step_size;
    int   max_depth;
    float density_scale;  // multiplier on raw [0,1] volume sample

    // Transfer function (scalar -> RGBA)
    cudaTextureObject_t colormap;

    // Scene bounding box (volume mapped to this box in world space)
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
