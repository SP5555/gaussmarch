// OptiX device programs for anisotropic Gaussian volume rendering.
//
// Kernel: Mahalanobis Gaussian  w_i = opacity_i * exp(-0.5 * d2_M)
//         d2_M = (x-mu)^T * Sigma^-1 * (x-mu)
//
// Primary ray: delta tracking -- stochastic scatter on first event.
// Shadow ray:  deterministic march -- accumulates 1-exp(-eff_density).

#include <optix.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "gaussmarch_params.h"

extern "C" __constant__ GaussianLaunchParams params;

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

// ===== Payload layout =====
// p0 (uint32) : hit count   -- number of Gaussians that passed the 3-sigma test
// p1 (float)  : density     -- sum of weights:          sum(opacity_i * exp(-0.5 * d2_M_i))
// p2 (float)  : scalar_num  -- density-weighted scalar: sum(scalar_i * weight_i)
// scalar_avg = p2 / p1  gives the local mean scalar value at the query point.

__device__ __forceinline__ void payload_init(uint32_t &p0, uint32_t &p1, uint32_t &p2)
{
    p0 = 0; p1 = f2u(0.f); p2 = f2u(0.f);
}

// ===== Point query =====
// Fires a zero-length ray (tmin=tmax=0) at world position x.
// OptiX traverses the BVH, calling __intersection__ for every Gaussian whose AABB
// contains x.  __anyhit__ accumulates each passing Gaussian into the payload.
// After the call, p0/p1/p2 hold the aggregated count/density/scalar_num.

__device__ __forceinline__ void point_query(float3 x,
                                            uint32_t &p0, uint32_t &p1, uint32_t &p2)
{
    payload_init(p0, p1, p2);
    optixTrace(
        params.traversable,
        x,
        make_float3(1.f, 0.f, 0.f),
        0.f, 0.f, 0.f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1, p2
    );
}

// ===== Raygen =====

extern "C" __global__ void __raygen__gaussian()
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

            uint32_t p0, p1, p2;
            point_query(x, p0, p1, p2);

            uint32_t count   = p0;          // number of Gaussians that contributed
            float    density = u2f(p1);     // sum(weight_i)
            float    snum    = u2f(p2);     // sum(scalar_i * weight_i)

            if (count == 0 || density <= 0.f) continue;

            float scalar_avg   = fminf(fmaxf(snum / density, 0.f), 1.f); // weighted mean scalar -> TF input
            float4 tf          = tex1D<float4>(params.colormap, scalar_avg);
            float  eff_density = density * tf.w; // modulate density by TF opacity

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
                        float shadow_texit_clamped = shadow_texit;
                        float shadow = 0.f;
                        // Start half a step in to avoid self-intersection
                        float ts = params.step_size * pcg_float(rng);

                        int shadow_depth = params.max_depth / 4;
                        for (int j = 0; j < shadow_depth && ts < shadow_texit_clamped && shadow < 1.f; ++j) {
                            ts += params.step_size;
                            float3 xs = {
                                scatter.x + ts * ld.x,
                                scatter.y + ts * ld.y,
                                scatter.z + ts * ld.z,
                            };

                            uint32_t sp0, sp1, sp2;
                            point_query(xs, sp0, sp1, sp2);

                            uint32_t scount   = sp0;
                            float    sdensity = u2f(sp1);
                            float    ssnum    = u2f(sp2);

                            if (scount == 0 || sdensity <= 0.f) continue;

                            float s_scalar = fminf(fmaxf(ssnum / sdensity, 0.f), 1.f);
                            float4 s_tf    = tex1D<float4>(params.colormap, s_scalar);
                            float  s_eff   = sdensity * s_tf.w;

                            shadow += 1.f - expf(-s_eff);
                            if (shadow > 1.f) shadow = 1.f;
                        }
                        visibility = 1.f - shadow;
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

// ===== Miss =====

extern "C" __global__ void __miss__gaussian() {}

// ===== Intersection =====
// Evaluates the Mahalanobis kernel for each Gaussian in the leaf AABB.

extern "C" __global__ void __intersection__gaussian()
{
    uint32_t cluster_id = optixGetPrimitiveIndex();
    float3   origin     = optixGetWorldRayOrigin();

    for (uint32_t i = 0; i < GAUSSIANS_PER_LEAF; ++i) {
        uint32_t gid = cluster_id * GAUSSIANS_PER_LEAF + i;
        if (gid >= params.num_gaussians) break;

        GpuGaussian g = params.gaussians[gid];

        float dx = origin.x - g.mu_x;
        float dy = origin.y - g.mu_y;
        float dz = origin.z - g.mu_z;

        // d2_M = d^T * Sigma^-1 * d  (Sigma^-1 stored as upper triangle)
        float d2m = g.s00 * dx*dx
                  + g.s11 * dy*dy
                  + g.s22 * dz*dz
                  + 2.f * g.s01 * dx*dy
                  + 2.f * g.s02 * dx*dz
                  + 2.f * g.s12 * dy*dz;

        if (d2m > 9.f) continue;  // beyond 3-sigma

        float weight = g.opacity * expf(-0.5f * d2m);
        optixReportIntersection(0.f, 0, f2u(g.scalar), f2u(weight));
    }
}

// ===== Any-hit =====
// Called once per Gaussian that passed __intersection__ (i.e. d2_M <= 9).
// Reads the two attributes reported by __intersection__, adds them into the
// shared payload, then calls optixIgnoreIntersection() so traversal continues
// to the next overlapping Gaussian instead of stopping.

extern "C" __global__ void __anyhit__gaussian()
{
    float scalar = u2f(optixGetAttribute_0()); // this Gaussian's scalar value
    float weight = u2f(optixGetAttribute_1()); // opacity * exp(-0.5 * d2_M)

    // read current payload accumulator
    uint32_t count   = optixGetPayload_0();
    float    density = u2f(optixGetPayload_1());
    float    snum    = u2f(optixGetPayload_2());

    count++;
    density += weight;
    snum    += scalar * weight;

    // write back
    optixSetPayload_0(count);
    optixSetPayload_1(f2u(density));
    optixSetPayload_2(f2u(snum));

    // don't terminate -- let OptiX keep visiting other Gaussians at this point
    optixIgnoreIntersection();
}
