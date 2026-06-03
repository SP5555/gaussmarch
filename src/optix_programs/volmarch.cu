// OptiX device programs for raw volume rendering.
//
// Replaces Gaussian BVH point queries with direct 3D texture lookups.
// Everything else (delta tracking, Beer-Lambert shadow, TF, accumulation)
// is identical to gaussmarch.cu.
//
// Primary ray: delta tracking -- stochastic scatter on first event.
// Shadow ray:  deterministic march -- multiplicative Beer-Lambert transmittance.

#include <optix.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "volmarch_params.h"

extern "C" __constant__ VolumeLaunchParams params;

// ===== RNG: Murmur Hash 3 seed + PCG32 generator =====

__device__ __forceinline__ uint32_t murmur_mix(uint32_t hash, uint32_t k)
{
    k *= 0xcc9e2d51u;
    k = (k << 15u) | (k >> 17u);
    k *= 0x1b873593u;
    hash ^= k;
    hash = ((hash << 13u) | (hash >> 19u)) * 5u + 0xe6546b64u;
    return hash;
}

__device__ __forceinline__ uint32_t murmur_finalize(uint32_t hash)
{
    hash ^= hash >> 16u;
    hash *= 0x85ebca6bu;
    hash ^= hash >> 13u;
    hash *= 0xc2b2ae35u;
    hash ^= hash >> 16u;
    return hash;
}

__device__ __forceinline__ uint32_t rng_seed(int frame_id, uint2 pixel, uint2 dims)
{
    uint32_t s = murmur_mix(0u, pixel.x + pixel.y * dims.x);
    s = murmur_mix(s, (uint32_t)frame_id);
    return murmur_finalize(s);
}

__device__ __forceinline__ uint32_t pcg_step(uint32_t &state)
{
    uint32_t prev = state;
    state = prev * 747796405u + 2891336453u;
    uint32_t word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float pcg_float(uint32_t &state)
{
    return (float)(pcg_step(state) >> 8u) * (1.f / (float)(1u << 24u));
}

// ===== Bit-cast helpers =====

__device__ __forceinline__ uint32_t f2u(float f) { return __float_as_uint(f); }
__device__ __forceinline__ float    u2f(uint32_t u) { return __uint_as_float(u); }

// ===== Ray-AABB intersection =====

__device__ __forceinline__ bool aabb_intersect(
    float3 origin, float3 direction,
    float3 aabb_min, float3 aabb_max,
    float &tenter, float &texit)
{
    float3 inv_dir = {1.f / direction.x, 1.f / direction.y, 1.f / direction.z};
    float t1x = (aabb_min.x - origin.x) * inv_dir.x;
    float t2x = (aabb_max.x - origin.x) * inv_dir.x;
    float t1y = (aabb_min.y - origin.y) * inv_dir.y;
    float t2y = (aabb_max.y - origin.y) * inv_dir.y;
    float t1z = (aabb_min.z - origin.z) * inv_dir.z;
    float t2z = (aabb_max.z - origin.z) * inv_dir.z;
    tenter = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
    texit  = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
    return tenter < texit;
}

// ===== Volume sample =====
// Maps world position to normalized [0,1] texture coords and samples the 3D texture.
// Returns density (scaled) and scalar (raw [0,1] value for TF lookup).

__device__ __forceinline__ void vol_sample(float3 x, float &density, float &scalar)
{
    float u = (x.x - params.scene_min.x) / (params.scene_max.x - params.scene_min.x);
    float v = (x.y - params.scene_min.y) / (params.scene_max.y - params.scene_min.y);
    float w = (x.z - params.scene_min.z) / (params.scene_max.z - params.scene_min.z);
    float val = tex3D<float>(params.vol_tex, u, v, w);
    scalar  = val;
    density = val * params.density_scale;
}

// ===== Raygen =====

extern "C" __global__ void __raygen__volume()
{
    const uint3 launch_idx  = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();
    const int px = (int)launch_idx.x;
    const int py = (int)launch_idx.y;
    const int fb_ofs = py * params.width + px;

    uint32_t rng = rng_seed(params.frame_id,
                            make_uint2(launch_idx.x, launch_idx.y),
                            make_uint2(launch_dims.x, launch_dims.y));

    float sx = ((float)px + 0.5f) / (float)params.width;
    float sy = ((float)py + 0.5f) / (float)params.height;

    float3 dir = {
        params.dir_00.x + sx * params.dir_du.x + sy * params.dir_dv.x,
        params.dir_00.y + sx * params.dir_du.y + sy * params.dir_dv.y,
        params.dir_00.z + sx * params.dir_du.z + sy * params.dir_dv.z,
    };
    float inv_len = 1.f / sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir.x *= inv_len; dir.y *= inv_len; dir.z *= inv_len;

    float tenter, texit;
    bool hit = aabb_intersect(params.cam_pos, dir,
                               params.scene_min, params.scene_max,
                               tenter, texit);
    tenter = fmaxf(tenter, 0.f);

    float4 final_color = {0.f, 0.f, 0.f, 0.f};

    if (hit && tenter < texit) {
        // Delta tracking threshold
        float xi;
        if (params.use_blue_noise) {
            int tile_x  = params.frame_id % 8;
            int tile_y  = (params.frame_id / 8) % 16;
            int coord_x = (px % 256) + tile_x * 256;
            int coord_y = (py % 256) + tile_y * 256;
            xi = tex2D<float>(params.stbn_tex, coord_x, coord_y);
        } else {
            xi = pcg_float(rng);
        }
        float rhs = -logf(1.f - xi * 0.99f);
        float lhs = 0.f;

        float t = tenter + params.step_size * pcg_float(rng);

        for (int i = 0; i < params.max_depth; ++i) {
            t += params.step_size;
            if (t >= texit) break;

            float3 x = {
                params.cam_pos.x + t * dir.x,
                params.cam_pos.y + t * dir.y,
                params.cam_pos.z + t * dir.z,
            };

            float density, scalar;
            vol_sample(x, density, scalar);

            if (density <= 0.f) continue;

            float4 tf         = tex1D<float4>(params.colormap, scalar);
            float  eff_density = density * tf.w;

            lhs += eff_density;

            if (lhs > rhs) {
                // Scatter event -- compute lit color
                float3 col = {
                    powf(tf.x, 1.f / 2.2f),
                    powf(tf.y, 1.f / 2.2f),
                    powf(tf.z, 1.f / 2.2f),
                };

                float visibility = 1.f;

                if (params.light_ambient < 1.f) {
                    float3 scatter = {
                        params.cam_pos.x + t * dir.x,
                        params.cam_pos.y + t * dir.y,
                        params.cam_pos.z + t * dir.z,
                    };
                    float3 ld = params.light_dir;

                    float shadow_tenter, shadow_texit;
                    bool shadow_hit = aabb_intersect(scatter, ld,
                                                     params.scene_min, params.scene_max,
                                                     shadow_tenter, shadow_texit);
                    if (shadow_hit && shadow_texit > 0.f) {
                        float transmittance = 1.f;
                        // Bias by 2x step to avoid shadow acne from self-intersection
                        float ts = params.shadow_step_size * (2.f + pcg_float(rng));

                        int shadow_depth = params.max_depth / 4;
                        for (int j = 0; j < shadow_depth && ts < shadow_texit && transmittance > 1e-4f; ++j) {
                            ts += params.shadow_step_size;
                            float3 xs = {
                                scatter.x + ts * ld.x,
                                scatter.y + ts * ld.y,
                                scatter.z + ts * ld.z,
                            };

                            float sd, ss;
                            vol_sample(xs, sd, ss);
                            if (sd <= 0.f) continue;

                            float4 s_tf  = tex1D<float4>(params.colormap, ss);
                            float  s_eff = sd * s_tf.w;
                            transmittance *= expf(-s_eff);
                        }
                        visibility = transmittance;
                    }
                }

                float lit = visibility * (1.f - params.light_ambient) + params.light_ambient;
                final_color = {col.x * lit, col.y * lit, col.z * lit, 1.f};
                break;
            }
        }
    }

    // Temporal accumulation
    if (params.accum_id > 1) {
        float4 prev = params.accum[fb_ofs];
        float  w    = 1.f / (float)params.accum_id;
        final_color = {
            w * final_color.x + (1.f - w) * prev.x,
            w * final_color.y + (1.f - w) * prev.y,
            w * final_color.z + (1.f - w) * prev.z,
            1.f,
        };
    }
    params.accum[fb_ofs] = final_color;

    params.output[fb_ofs * 3 + 0] = final_color.x;
    params.output[fb_ofs * 3 + 1] = final_color.y;
    params.output[fb_ofs * 3 + 2] = final_color.z;
}

// ===== Miss (required by OptiX pipeline, never triggered) =====

extern "C" __global__ void __miss__volume() {}
