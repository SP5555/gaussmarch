#include "rasterize_layer.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"

#define T_THRESHOLD     0.0001f
#define ALPHA_THRESHOLD (1.0f / 255.0f)

// Tile size matches the gsplat convention: 16x16 pixels per tile.
// num_tiles_x/y are always ceil(W/TILE_SIZE) x ceil(H/TILE_SIZE), computed
// internally from the resolution so tile boundaries align with pixel blocks.
static constexpr int TILE_SIZE = 16;

/* ===== ===== Tile Assign ===== ===== */

__device__ inline uint64_t makeKey(uint32_t tile_id, float depth)
{
    uint32_t u = __float_as_uint(depth);
    if (u >> 31)
        u = ~u;
    else
        u ^= 0x80000000u;
    return ((uint64_t)tile_id << 32) | u;
}

__global__ void tileAssignKernel(
    const float *__restrict__ ndc_x,
    const float *__restrict__ ndc_y,
    const float *__restrict__ ndc_z,
    const float *__restrict__ ndc_cxx,
    const float *__restrict__ ndc_cxy,
    const float *__restrict__ ndc_cyy,
    int splat_count,
    uint64_t *keys,
    uint32_t *values,
    uint32_t *pair_count,
    uint32_t *visible_count,
    int max_pairs,
    int num_tiles_x,
    int num_tiles_y,
    int screen_width,
    int screen_height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= splat_count) return;

    float x = ndc_x[i];
    float y = ndc_y[i];
    float z = ndc_z[i];

    if (fabsf(x) > 1.0f || fabsf(y) > 1.0f || fabsf(z) > 1.0f)
        return;

    float cxx = ndc_cxx[i];
    float cxy = ndc_cxy[i];
    float cyy = ndc_cyy[i];

    float det = cxx * cyy - cxy * cxy;
    if (!(det > 0.f)) return;  // !(x > 0) rejects both negative and NaN

    float trace   = cxx + cyy;
    float temp    = fmaxf(0.f, trace * trace - 4.f * det);
    float lambda2 = 0.5f * (trace + sqrtf(temp));

    float pixel_ndc = fmaxf(2.f / screen_width, 2.f / screen_height);
    if (3.f * sqrtf(lambda2 * 2.f) < pixel_ndc) return;

    float lambda1   = 0.5f * (trace - sqrtf(temp));
    float max_radius = 3.f * sqrtf(fmaxf(lambda1, lambda2));

    float min_x = x - max_radius, max_x = x + max_radius;
    float min_y = y - max_radius, max_y = y + max_radius;

    // Convert NDC -> pixel -> tile. Using screen_width/TILE_SIZE (float) rather
    // than num_tiles_x ensures tile boundaries align with the pixel blocks used
    // by the rasterize kernel even when the resolution isn't a multiple of TILE_SIZE.
    const float fx = (float)screen_width  / TILE_SIZE;
    const float fy = (float)screen_height / TILE_SIZE;
    auto ndcToTileX = [&](float v) { return (int)floorf((v + 1.f) * 0.5f * fx); };
    // Y-axis: ndc_y=+1 is the top of the screen (tile_y=0), ndc_y=-1 is the bottom.
    auto ndcToTileY = [&](float v) { return (int)floorf((1.f - v) * 0.5f * fy); };

    int tx0 = max(ndcToTileX(min_x), 0), tx1 = min(ndcToTileX(max_x), num_tiles_x - 1);
    // Because ndcToTileY is decreasing in v, max_y maps to the lower tile index (top).
    int ty0 = max(ndcToTileY(max_y), 0), ty1 = min(ndcToTileY(min_y), num_tiles_y - 1);
    if (tx0 > tx1 || ty0 > ty1) return;

    atomicAdd(visible_count, 1u);
    for (int ty = ty0; ty <= ty1; ty++)
    for (int tx = tx0; tx <= tx1; tx++)
    {
        uint32_t tile_id = (uint32_t)(ty * num_tiles_x + tx);
        uint64_t key     = makeKey(tile_id, z);
        uint32_t slot    = atomicAdd(pair_count, 1u);
        if (slot >= (uint32_t)max_pairs) { atomicSub(pair_count, 1u); return; }
        keys[slot]   = key;
        values[slot] = (uint32_t)i;
    }
}

/* ===== ===== Build Tile Ranges ===== ===== */

__global__ void buildTileRangesKernel(
    const uint64_t *keys_sorted,
    int2 *tile_ranges,
    uint32_t pair_count,
    int num_tiles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint32_t)i >= pair_count) return;
    int tile_id = (int)(keys_sorted[i] >> 32);
    if (tile_id < 0 || tile_id >= num_tiles) return;
    if (i == 0 || (int)(keys_sorted[i - 1] >> 32) != tile_id)
        tile_ranges[tile_id].x = i;
    if (i == (int)pair_count - 1 || (int)(keys_sorted[i + 1] >> 32) != tile_id)
        tile_ranges[tile_id].y = i + 1;
}

/* ===== ===== Forward Kernel ===== ===== */

/**
 * Tile-cooperative forward rasterizer.
 * 
 * One CUDA block per tile, BLOCK_THREADS threads per block.
 * 
 * For each batch of BLOCK_THREADS pixels in the tile, threads cooperate to:
 *   1. Load CHUNK_SIZE splats into shared memory together (coalesced reads),
 *      replacing per-thread scattered global reads with a single collaborative load.
 *   2. Each thread alpha-composites its assigned pixel against the cached chunk.
 * 
 * Tiles larger than BLOCK_THREADS pixels are handled by processing pixels in
 * sequential batches, replaying the full splat stream once per batch.
 * This trades register pressure for simplicity and is efficient when
 * tiles are not much larger than BLOCK_THREADS pixels.
 */
__global__ void rasterizeKernel(
    const float *__restrict__ ndc_x,
    const float *__restrict__ ndc_y,
    const float *__restrict__ ndc_cxx,
    const float *__restrict__ ndc_cxy,
    const float *__restrict__ ndc_cyy,
    const float *__restrict__ ndc_R,
    const float *__restrict__ ndc_G,
    const float *__restrict__ ndc_B,
    const float *__restrict__ ndc_A,
    const uint32_t *__restrict__ values_sorted,
    const int2     *__restrict__ tile_ranges,
    float *pixels,
    float *T_final,
    int   *n_contrib,
    int num_tiles_x, int num_tiles_y,
    int screen_width, int screen_height)
{
    // shared memory splat cache, loaded collaboratively each chunk
    __shared__ float sh_x      [CHUNK_SIZE];
    __shared__ float sh_y      [CHUNK_SIZE];
    __shared__ float sh_inv_cxx[CHUNK_SIZE]; // precomputed: cyy / det
    __shared__ float sh_inv_cxy[CHUNK_SIZE]; // precomputed: -cxy / det
    __shared__ float sh_inv_cyy[CHUNK_SIZE]; // precomputed: cxx / det
    __shared__ float sh_R      [CHUNK_SIZE];
    __shared__ float sh_G      [CHUNK_SIZE];
    __shared__ float sh_B      [CHUNK_SIZE];
    __shared__ float sh_A      [CHUNK_SIZE];

    int tile_id = blockIdx.x;
    int tid     = threadIdx.x;

    int tile_ix = tile_id % num_tiles_x;
    int tile_iy = tile_id / num_tiles_x;

    // pixel bounds for this tile (clamped so the last tile handles partial coverage)
    int px0 = tile_ix * TILE_SIZE;
    int px1 = min((tile_ix + 1) * TILE_SIZE, screen_width);
    int py0 = tile_iy * TILE_SIZE;
    int py1 = min((tile_iy + 1) * TILE_SIZE, screen_height);
    int tile_pixels = (px1 - px0) * (py1 - py0);

    int2 range = tile_ranges[tile_id];

    for (int px_batch_start = 0; px_batch_start < tile_pixels; px_batch_start += BLOCK_THREADS)
    {
        int local_px_idx = px_batch_start + tid;
        int lx = local_px_idx % (px1 - px0);
        int ly = local_px_idx / (px1 - px0);
        int pixel_x = px0 + lx;
        int pixel_y = py0 + ly;
        bool valid_pixel = (local_px_idx < tile_pixels);

        float x_ndc = valid_pixel ? (2.f * (pixel_x + 0.5f) / screen_width)  - 1.f : 0.f;
        float y_ndc = valid_pixel ? 1.f - (2.f * (pixel_y + 0.5f) / screen_height) : 0.f;

        float C_R = 0.f, C_G = 0.f, C_B = 0.f;
        float T   = 1.f;
        int contrib = 0;
        bool done = !valid_pixel;

        // --- stream splats in chunks ---
        for (int chunk_start = range.x; chunk_start < range.y; chunk_start += CHUNK_SIZE)
        {
            int chunk_end  = min(chunk_start + CHUNK_SIZE, range.y);
            int chunk_size = chunk_end - chunk_start;

            // --- collaborative load ---
            if (tid < chunk_size)
            {
                uint32_t sid    = values_sorted[chunk_start + tid];
                float cxx       = ndc_cxx[sid];
                float cxy       = ndc_cxy[sid];
                float cyy       = ndc_cyy[sid];
                // all values_sorted entries have det > 0 (guaranteed by tileAssignKernel)
                float inv_det   = 1.f / (cxx * cyy - cxy * cxy);
                sh_x      [tid] = ndc_x  [sid];
                sh_y      [tid] = ndc_y  [sid];
                sh_inv_cxx[tid] =  cyy * inv_det;
                sh_inv_cxy[tid] = -cxy * inv_det;
                sh_inv_cyy[tid] =  cxx * inv_det;
                sh_R      [tid] = ndc_R  [sid];
                sh_G      [tid] = ndc_G  [sid];
                sh_B      [tid] = ndc_B  [sid];
                sh_A      [tid] = ndc_A  [sid];
            }
            __syncthreads();

            // --- each thread composites its pixel against the chunk ---
            if (!done)
            {
                for (int j = 0; j < chunk_size; j++)
                {
                    float dx      = x_ndc - sh_x[j];
                    float dy      = y_ndc - sh_y[j];
                    float inv_cxx = sh_inv_cxx[j];
                    float inv_cxy = sh_inv_cxy[j];
                    float inv_cyy = sh_inv_cyy[j];

                    float dist2 = dx*dx*inv_cxx + 2.f*dx*dy*inv_cxy + dy*dy*inv_cyy;
                    if (dist2 > 9.f) continue;

                    float alpha = fminf(0.99f, sh_A[j] * expf(-0.5f * dist2));
                    if (alpha < ALPHA_THRESHOLD) continue;

                    C_R += sh_R[j] * alpha * T;
                    C_G += sh_G[j] * alpha * T;
                    C_B += sh_B[j] * alpha * T;
                    T   *= (1.f - alpha);
                    contrib++;

                    if (T < T_THRESHOLD) { done = true; break; }
                }
            }
            __syncthreads(); // done with shared mem before next chunk load
        }

        // write output
        if (valid_pixel)
        {
            int pidx = pixel_y * screen_width + pixel_x;
            pixels   [pidx * 3 + 0] = C_R;
            pixels   [pidx * 3 + 1] = C_G;
            pixels   [pidx * 3 + 2] = C_B;
            T_final  [pidx]         = T;
            n_contrib[pidx]         = contrib;
        }
    }
}

/* ===== ===== Backward Kernel ===== ===== */

/**
 * Tile-cooperative backward rasterizer (hybrid variant).
 *
 * One CUDA block per tile, BLOCK_THREADS threads per block.
 *
 * 1. COOPERATIVE LOADING (forward sweep):
 *    Splats are streamed through shared memory in chunks of CHUNK_SIZE.
 *    All threads collaborate to load each chunk (coalesced reads), then each
 *    thread scans the chunk for splats that contribute to its pixel.
 *    Contributing splat positions are recorded in contrib_pos[], storing the
 *    absolute index into values_sorted, so the backward pass can directly
 *    index into shared memory without any scattered global reads.
 *
 * 2. contrib_pos BACKWARD WALK:
 *    The backward pass streams chunks in reverse order. For each chunk, each
 *    thread walks its contrib_pos[] cursor backward, skipping non-contributing
 *    splats entirely via a single boundary check. Splat data is read from
 *    shared memory (loaded collaboratively). The cursor naturally terminates
 *    early once all contributors have been processed, making sparse pixels
 *    essentially free.
 *    Because each thread accesses a different contrib_pos[] index, atomicAdds
 *    to sh_grad_* scatter across different slots - avoiding serialization.
 *    (A j-loop visiting the same index for all threads would serialize all
 *    warp threads onto the same shared memory slot.)
 *
 * 3. SHARED GRADIENT ACCUMULATION + FLUSH:
 *    Gradients accumulate into shared memory atomics (sh_grad_*) per chunk,
 *    then flush to global with one atomicAdd per splat per chunk. All threads
 *    in the block contribute to the same shared gradient slots regardless of
 *    which pixel they own, amortizing global atomic pressure across the tile.
 *
 * 4. REFORMULATED COVARIANCE GRADIENTS (no sh_cxx/sh_cxy/sh_cyy):
 *    ddist2/dcxx, dcyy, dcxy are expressed using ddist2_dx, ddist2_dy,
 *    inv_cxy, inv_det, and dist2 - all already in registers. Eliminates
 *    sh_cxx/sh_cxy/sh_cyy (3KB shared memory), improving occupancy.
 *    Identities: ddist2_dcxx = -ddist2_dx^2 / 4,
 *                ddist2_dcyy = -ddist2_dy^2 / 4,
 *                ddist2_dcxy = -2*inv_cxy*dist2 - 2*dx*dy*inv_det.
 */
__global__ void backwardKernel(
    const float *__restrict__ ndc_x,
    const float *__restrict__ ndc_y,
    const float *__restrict__ ndc_cxx,
    const float *__restrict__ ndc_cxy,
    const float *__restrict__ ndc_cyy,
    const float *__restrict__ ndc_R,
    const float *__restrict__ ndc_G,
    const float *__restrict__ ndc_B,
    const float *__restrict__ ndc_A,
    const uint32_t *__restrict__ values_sorted,
    const int2     *__restrict__ tile_ranges,
    const float    *__restrict__ pixels,
    const float    *__restrict__ T_final,
    const int      *__restrict__ n_contrib,
    const float    *__restrict__ grad_o,
    float *grad_i_ndc_x,
    float *grad_i_ndc_y,
    float *grad_i_ndc_cxx,
    float *grad_i_ndc_cxy,
    float *grad_i_ndc_cyy,
    float *grad_i_R,
    float *grad_i_G,
    float *grad_i_B,
    float *grad_i_A,
    int num_tiles_x, int num_tiles_y,
    int screen_width, int screen_height)
{
    // splat data cache (loaded collaboratively)
    __shared__ float    sh_x      [CHUNK_SIZE];
    __shared__ float    sh_y      [CHUNK_SIZE];
    __shared__ float    sh_inv_cxx[CHUNK_SIZE]; // precomputed: cyy / det
    __shared__ float    sh_inv_cxy[CHUNK_SIZE]; // precomputed: -cxy / det
    __shared__ float    sh_inv_cyy[CHUNK_SIZE]; // precomputed: cxx / det
    __shared__ float    sh_inv_det[CHUNK_SIZE]; // precomputed: 1 / det
    __shared__ float    sh_R      [CHUNK_SIZE];
    __shared__ float    sh_G      [CHUNK_SIZE];
    __shared__ float    sh_B      [CHUNK_SIZE];
    __shared__ float    sh_A      [CHUNK_SIZE];
    __shared__ uint32_t sh_sid    [CHUNK_SIZE];

    // per-chunk gradient accumulators, flushed to global after each chunk
    __shared__ float sh_grad_x  [CHUNK_SIZE];
    __shared__ float sh_grad_y  [CHUNK_SIZE];
    __shared__ float sh_grad_cxx[CHUNK_SIZE];
    __shared__ float sh_grad_cxy[CHUNK_SIZE];
    __shared__ float sh_grad_cyy[CHUNK_SIZE];
    __shared__ float sh_grad_R  [CHUNK_SIZE];
    __shared__ float sh_grad_G  [CHUNK_SIZE];
    __shared__ float sh_grad_B  [CHUNK_SIZE];
    __shared__ float sh_grad_A  [CHUNK_SIZE];

    int tile_id = blockIdx.x;
    int tid     = threadIdx.x;

    int tile_ix = tile_id % num_tiles_x;
    int tile_iy = tile_id / num_tiles_x;

    // pixel bounds for this tile (clamped so the last tile handles partial coverage)
    int px0 = tile_ix * TILE_SIZE;
    int px1 = min((tile_ix + 1) * TILE_SIZE, screen_width);
    int py0 = tile_iy * TILE_SIZE;
    int py1 = min((tile_iy + 1) * TILE_SIZE, screen_height);
    int tile_pixels = (px1 - px0) * (py1 - py0);

    int2 range = tile_ranges[tile_id];

    for (int px_batch_start = 0; px_batch_start < tile_pixels; px_batch_start += BLOCK_THREADS)
    {
        int local_px_idx = px_batch_start + tid;
        int lx = local_px_idx % (px1 - px0);
        int ly = local_px_idx / (px1 - px0);
        int pixel_x = px0 + lx;
        int pixel_y = py0 + ly;
        bool valid_pixel = (local_px_idx < tile_pixels);

        float x_ndc = valid_pixel ? (2.f * (pixel_x + 0.5f) / screen_width)  - 1.f : 0.f;
        float y_ndc = valid_pixel ? 1.f - (2.f * (pixel_y + 0.5f) / screen_height) : 0.f;

        int pixel_idx = pixel_y * screen_width + pixel_x;
        int n_contr   = valid_pixel ? n_contrib[pixel_idx] : 0;

        // --- forward sweep: collect contributing splats in forward order ---
        // Worst-case bound: log(T_THRESHOLD) / log(1 - ALPHA_THRESHOLD) ~ 2300.
        // 512 covers scenes where mean alpha >= ~0.015; deeper contributors have
        // very low T_before and contribute negligible gradient anyway.
        const int MAX_CONTRIB = 512;
        uint32_t contrib_pos[MAX_CONTRIB];
        int contributed_splats = 0;
        for (int chunk_start = range.x; chunk_start < range.y; chunk_start += CHUNK_SIZE)
        {
            int chunk_end  = min(chunk_start + CHUNK_SIZE, range.y);
            int chunk_size = chunk_end - chunk_start;

            // --- cooperative load ---
            if (tid < chunk_size)
            {
                uint32_t sid  = values_sorted[chunk_start + tid];
                float cxx     = ndc_cxx[sid];
                float cxy     = ndc_cxy[sid];
                float cyy     = ndc_cyy[sid];
                float inv_det = 1.f / (cxx * cyy - cxy * cxy);
                sh_sid    [tid] = sid;
                sh_x      [tid] = ndc_x[sid];
                sh_y      [tid] = ndc_y[sid];
                sh_inv_cxx[tid] =  cyy * inv_det;
                sh_inv_cxy[tid] = -cxy * inv_det;
                sh_inv_cyy[tid] =  cxx * inv_det;
                sh_inv_det[tid] = inv_det;
                sh_R      [tid] = ndc_R[sid];
                sh_G      [tid] = ndc_G[sid];
                sh_B      [tid] = ndc_B[sid];
                sh_A      [tid] = ndc_A[sid];
            }
            __syncthreads();

            if (valid_pixel)
            {
                for (int i = 0;
                    i < chunk_size &&
                    contributed_splats < n_contr &&
                    contributed_splats < MAX_CONTRIB;
                    i++)
                {
                    float dx      = x_ndc - sh_x[i];
                    float dy      = y_ndc - sh_y[i];
                    float inv_cxx = sh_inv_cxx[i];
                    float inv_cxy = sh_inv_cxy[i];
                    float inv_cyy = sh_inv_cyy[i];
                    float dist2   = dx*dx*inv_cxx + 2.f*dx*dy*inv_cxy + dy*dy*inv_cyy;
                    if (dist2 > 9.f) continue;

                    float alpha = fminf(0.99f, sh_A[i] * expf(-0.5f * dist2));
                    if (alpha < ALPHA_THRESHOLD) continue;

                    contrib_pos[contributed_splats] = chunk_start + i;
                    contributed_splats++;
                }
            }
            __syncthreads(); // done with shared mem before next chunk load
        }

        float dL_dCr = valid_pixel ? grad_o[3 * pixel_idx + 0] : 0.f;
        float dL_dCg = valid_pixel ? grad_o[3 * pixel_idx + 1] : 0.f;
        float dL_dCb = valid_pixel ? grad_o[3 * pixel_idx + 2] : 0.f;

        // T_after starts at T_final (transmittance after all splats)
        float T_after  = valid_pixel ? T_final[pixel_idx] : 1.f;

        // C_back accumulates color of splats AFTER current one (for dL/dalpha)
        float C_back_r = 0.f, C_back_g = 0.f, C_back_b = 0.f;

        int total_splats = range.y - range.x;
        if (total_splats <= 0) continue;

        int num_chunks   = (total_splats + CHUNK_SIZE - 1) / CHUNK_SIZE;
        int contrib_cursor = contributed_splats - 1;

        for (int chunk_idx = num_chunks - 1; chunk_idx >= 0; chunk_idx--)
        {
            int chunk_start = range.x + chunk_idx * CHUNK_SIZE;
            int chunk_end   = min(chunk_start + CHUNK_SIZE, range.y);
            int chunk_size  = chunk_end - chunk_start;

            // --- collaborative load + zero gradient accumulators ---
            if (tid < chunk_size)
            {
                uint32_t sid  = values_sorted[chunk_start + tid];
                float cxx     = ndc_cxx[sid];
                float cxy     = ndc_cxy[sid];
                float cyy     = ndc_cyy[sid];
                float inv_det = 1.f / (cxx * cyy - cxy * cxy);
                sh_sid    [tid] = sid;
                sh_x      [tid] = ndc_x[sid];
                sh_y      [tid] = ndc_y[sid];
                sh_inv_cxx[tid] =  cyy * inv_det;
                sh_inv_cxy[tid] = -cxy * inv_det;
                sh_inv_cyy[tid] =  cxx * inv_det;
                sh_inv_det[tid] = inv_det;
                sh_R      [tid] = ndc_R[sid];
                sh_G      [tid] = ndc_G[sid];
                sh_B      [tid] = ndc_B[sid];
                sh_A      [tid] = ndc_A[sid];
                sh_grad_x  [tid] = 0.f;
                sh_grad_y  [tid] = 0.f;
                sh_grad_cxx[tid] = 0.f;
                sh_grad_cxy[tid] = 0.f;
                sh_grad_cyy[tid] = 0.f;
                sh_grad_R  [tid] = 0.f;
                sh_grad_G  [tid] = 0.f;
                sh_grad_B  [tid] = 0.f;
                sh_grad_A  [tid] = 0.f;
            }
            __syncthreads();

            // --- backward pass over this chunk in reverse order ---
            if (valid_pixel)
            {
                while (contrib_cursor >= 0 &&
                       contrib_pos[contrib_cursor] >= (uint32_t)chunk_start)
                {
                    int i = (int)(contrib_pos[contrib_cursor] - chunk_start);

                    float dx      = x_ndc - sh_x[i];
                    float dy      = y_ndc - sh_y[i];
                    float inv_cxx = sh_inv_cxx[i];
                    float inv_cxy = sh_inv_cxy[i];
                    float inv_cyy = sh_inv_cyy[i];
                    float inv_det = sh_inv_det[i];
                    float dist2   = dx*dx*inv_cxx + 2.f*dx*dy*inv_cxy + dy*dy*inv_cyy;
                    float g       = expf(-0.5f * dist2);
                    float alpha   = fminf(0.99f, sh_A[i] * g);

                    float T_before = T_after / fmaxf(1.f - alpha, 1e-6f);
                    float cR = sh_R[i], cG = sh_G[i], cB = sh_B[i];

                    // dL/dcolor
                    atomicAdd(&sh_grad_R[i], alpha * T_before * dL_dCr);
                    atomicAdd(&sh_grad_G[i], alpha * T_before * dL_dCg);
                    atomicAdd(&sh_grad_B[i], alpha * T_before * dL_dCb);

                    // dL/dalpha
                    float dL_dalpha = T_before * (
                        (cR - C_back_r) * dL_dCr +
                        (cG - C_back_g) * dL_dCg +
                        (cB - C_back_b) * dL_dCb);

                    float raw_alpha = sh_A[i] * g;
                    if (raw_alpha < 0.99f)
                        atomicAdd(&sh_grad_A[i], dL_dalpha * g);

                    float dL_ddist2 = (raw_alpha < 0.99f) ? dL_dalpha * (-0.5f) * alpha : 0.f;

                    float ddist2_dx = -2.f * (dx * inv_cxx + dy * inv_cxy);
                    float ddist2_dy = -2.f * (dx * inv_cxy + dy * inv_cyy);
                    atomicAdd(&sh_grad_x[i], dL_ddist2 * ddist2_dx);
                    atomicAdd(&sh_grad_y[i], dL_ddist2 * ddist2_dy);

                    // ddist2_dcxx = -(dx*cyy - dy*cxy)^2 / det^2 = -ddist2_dx^2 / 4
                    // ddist2_dcyy = -(dy*cxx - dx*cxy)^2 / det^2 = -ddist2_dy^2 / 4
                    // ddist2_dcxy = -2*inv_cxy*dist2 - 2*dx*dy*inv_det
                    float ddist2_dcxx = -0.25f * ddist2_dx * ddist2_dx;
                    float ddist2_dcyy = -0.25f * ddist2_dy * ddist2_dy;
                    float ddist2_dcxy = -2.f * inv_cxy * dist2 - 2.f * dx * dy * inv_det;
                    atomicAdd(&sh_grad_cxx[i], dL_ddist2 * ddist2_dcxx);
                    atomicAdd(&sh_grad_cxy[i], dL_ddist2 * ddist2_dcxy);
                    atomicAdd(&sh_grad_cyy[i], dL_ddist2 * ddist2_dcyy);

                    C_back_r += cR * alpha * T_before;
                    C_back_g += cG * alpha * T_before;
                    C_back_b += cB * alpha * T_before;
                    T_after   = T_before;

                    contrib_cursor--;
                }
            }
            __syncthreads(); // all threads done accumulating into shared mem

            // --- flush shared gradients to global memory ---
            if (tid < chunk_size)
            {
                uint32_t sid = sh_sid[tid];
                atomicAdd(&grad_i_ndc_x  [sid], sh_grad_x  [tid]);
                atomicAdd(&grad_i_ndc_y  [sid], sh_grad_y  [tid]);
                atomicAdd(&grad_i_ndc_cxx[sid], sh_grad_cxx[tid]);
                atomicAdd(&grad_i_ndc_cxy[sid], sh_grad_cxy[tid]);
                atomicAdd(&grad_i_ndc_cyy[sid], sh_grad_cyy[tid]);
                atomicAdd(&grad_i_R      [sid], sh_grad_R  [tid]);
                atomicAdd(&grad_i_G      [sid], sh_grad_G  [tid]);
                atomicAdd(&grad_i_B      [sid], sh_grad_B  [tid]);
                atomicAdd(&grad_i_A      [sid], sh_grad_A  [tid]);
            }
            __syncthreads(); // before next chunk reuses sh_grad_*
        }
    }
}

/* ===== ===== Utils ===== ===== */

uint32_t RasterizeLayer::getVisibleCount()
{
    uint32_t c = 0;
    CUDA_CHECK(cudaMemcpy(&c, d_visible_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return c;
}

/* ===== ===== Lifecycle ===== ===== */

void RasterizeLayer::allocate(int width, int height, int count)
{
    screen_width  = width;
    screen_height = height;
    num_pixels    = width * height;
    num_tiles_x   = (width  + TILE_SIZE - 1) / TILE_SIZE;
    num_tiles_y   = (height + TILE_SIZE - 1) / TILE_SIZE;
    // The tile-assign kernel degrades gracefully (drops excess pairs) if exceeded.
    max_pairs     = (1 << 25);
    int numTiles  = num_tiles_x * num_tiles_y;

    d_out_pixels.allocate   (num_pixels * 3);
    d_T_final.allocate      (num_pixels);
    d_n_contrib.allocate    (num_pixels);

    d_keys.allocate         (max_pairs);
    d_values.allocate       (max_pairs);
    d_keys_sorted.allocate  (max_pairs);
    d_values_sorted.allocate(max_pairs);
    d_pair_count.allocate   (1);
    d_visible_count.allocate(1);
    d_tile_ranges.allocate  (numTiles);

    // Pre-reserve CUB sort temp at max_pairs depth so forward() never allocates.
    {
        size_t required = 0;
        uint64_t *dummy_keys = nullptr;
        uint32_t *dummy_vals = nullptr;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, required,
            dummy_keys, dummy_keys,
            dummy_vals, dummy_vals,
            max_pairs));
        d_sort_temp.allocate(required);
        sort_temp_bytes = required;
    }

}

void RasterizeLayer::allocateGrad(int count)
{
    grad_in.allocate(count);
}

void RasterizeLayer::zero_grad()
{
    grad_in.zero_grad();
}

void RasterizeLayer::resize(int new_width, int new_height)
{
    if (new_width == screen_width && new_height == screen_height)
        return;
    screen_width  = new_width;
    screen_height = new_height;
    num_pixels    = screen_width * screen_height;
    num_tiles_x   = (screen_width  + TILE_SIZE - 1) / TILE_SIZE;
    num_tiles_y   = (screen_height + TILE_SIZE - 1) / TILE_SIZE;
    d_out_pixels.allocate(num_pixels * 3);
    d_T_final.allocate   (num_pixels);
    d_n_contrib.allocate (num_pixels);
    d_tile_ranges.allocate(num_tiles_x * num_tiles_y);
}

/* ===== ===== Forward / Backward ===== ===== */

void RasterizeLayer::forward()
{
    int numTiles = num_tiles_x * num_tiles_y;

    CUDA_CHECK(cudaMemset(d_out_pixels,    0, num_pixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pair_count,    0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_visible_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_tile_ranges,   0, numTiles * sizeof(int2)));

    // tile assign
    {
        int blocks = (in->count + BLOCK_THREADS - 1) / BLOCK_THREADS;
        tileAssignKernel<<<blocks, BLOCK_THREADS>>>(
            in->pos_x, in->pos_y, in->pos_z,
            in->cov_xx, in->cov_xy, in->cov_yy,
            in->count,
            d_keys, d_values, d_pair_count,
            d_visible_count,
            max_pairs, num_tiles_x, num_tiles_y,
            screen_width, screen_height
        );
        CUDA_SYNC_CHECK();
    }

    uint32_t pair_count = 0;
    CUDA_CHECK(cudaMemcpy(&pair_count, d_pair_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (pair_count == 0) return;

    // sort
    {
        size_t required = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, required,
            d_keys.ptr, d_keys_sorted.ptr,
            d_values.ptr, d_values_sorted.ptr,
            (int)pair_count));

        if (required > sort_temp_bytes)
        {
            d_sort_temp.allocate(required);
            sort_temp_bytes = required;
        }

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            (void *)d_sort_temp.ptr, required,
            d_keys.ptr, d_keys_sorted.ptr,
            d_values.ptr, d_values_sorted.ptr,
            (int)pair_count));
    }

    // build tile ranges
    {
        int blocks = ((int)pair_count + BLOCK_THREADS - 1) / BLOCK_THREADS;
        buildTileRangesKernel<<<blocks, BLOCK_THREADS>>>(
            d_keys_sorted, d_tile_ranges, pair_count, numTiles);
        CUDA_SYNC_CHECK();
    }

    // forward rasterize, one block per tile
    {
        rasterizeKernel<<<numTiles, BLOCK_THREADS>>>(
            in->pos_x, in->pos_y,
            in->cov_xx, in->cov_xy, in->cov_yy,
            in->color_r, in->color_g, in->color_b,
            in->opacity,
            d_values_sorted, d_tile_ranges,
            d_out_pixels, d_T_final, d_n_contrib,
            num_tiles_x, num_tiles_y,
            screen_width, screen_height
        );
        CUDA_SYNC_CHECK();
    }
}

void RasterizeLayer::backward()
{
    int numTiles = num_tiles_x * num_tiles_y;
    backwardKernel<<<numTiles, BLOCK_THREADS>>>(
        in->pos_x,   in->pos_y,
        in->cov_xx,  in->cov_xy,  in->cov_yy,
        in->color_r, in->color_g, in->color_b,
        in->opacity,
        d_values_sorted, d_tile_ranges,
        d_out_pixels, d_T_final, d_n_contrib,
        grad_pixels,
        grad_in.grad_pos_x,   grad_in.grad_pos_y,
        grad_in.grad_cov_xx,  grad_in.grad_cov_xy,  grad_in.grad_cov_yy,
        grad_in.grad_color_r, grad_in.grad_color_g, grad_in.grad_color_b,
        grad_in.grad_opacity,
        num_tiles_x, num_tiles_y,
        screen_width, screen_height
    );
    CUDA_SYNC_CHECK();
}
