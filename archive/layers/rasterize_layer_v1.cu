#include "rasterize_layer.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <math.h>
#include <stdio.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"
#define T_THRESHOLD     0.0001f
#define ALPHA_THRESHOLD (1.0f / 255.0f)

/* ===== ===== Tile Assign ===== ===== */

/**
 * Creates a 64-bit sort key from tile_id and depth.
 * Dudes from IEEE guarantee that, reinterpreting the bits of a positive float
 * as an integer preserves the sort order. Negative depths have all bits flipped
 * to preserve order across sign boundary.
 */
__device__ inline uint64_t makeKey(uint32_t tile_id, float depth)
{
    uint32_t u = __float_as_uint(depth);
    if (u >> 31)
        u = ~u;
    else
        u ^= 0x80000000u;
    return ((uint64_t)tile_id << 32) | u;
}

/**
 * Assigns each NDC-space splat to all tiles it overlaps.
 * Emits one (key=tile_id|depth, value=splat_id) pair per overlapping tile.
 * 
 * Culls splats that are completely off-screen or smaller than half a pixel.
 * 
 * One thread is launched per splat.
 * 
 * @param[in] ndc_x         X positions of splats in NDC [-1, 1]
 * @param[in] ndc_y         Y positions of splats in NDC [-1, 1]
 * @param[in] ndc_z         Z positions of splats in NDC [-1, 1]
 * @param[in] ndc_cxx       Covariance matrix element xx
 * @param[in] ndc_cxy       Covariance matrix element xy
 * @param[in] ndc_cyy       Covariance matrix element yy
 * @param[in] splat_count   Total number of splats in the scene
 * @param[out] keys         Output buffer for emitted keys (tile_id + depth) [max_pairs]
 * @param[out] values       Output buffer for emitted values (splat_id) [max_pairs]
 * @param[out] pair_count   Atomic counter for emitted pairs
 *                          Holds the total number of pairs emitted by all threads
 * @param[out] visible_count Atomic counter for visible splats (for stats)
 * @param[in] max_pairs     Capacity of the d_keys and d_values buffers
 * @param[in] num_tiles_x   Number of tiles in X direction on the screen
 * @param[in] num_tiles_y   Number of tiles in Y direction on the screen
 * @param[in] screen_width  Screen width in pixels
 * @param[in] screen_height Screen height in pixels
 */
__global__ void tileAssignKernel(
    // input splat data
    const float *__restrict__ ndc_x,
    const float *__restrict__ ndc_y,
    const float *__restrict__ ndc_z,
    const float *__restrict__ ndc_cxx,
    const float *__restrict__ ndc_cxy,
    const float *__restrict__ ndc_cyy,
    int splat_count,
    // output
    uint64_t *keys,
    uint32_t *values,
    uint32_t *pair_count,
    uint32_t *visible_count,
    int max_pairs,
    // screen config
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

    // NDC space covariance
    float cxx = ndc_cxx[i];
    float cxy = ndc_cxy[i];
    float cyy = ndc_cyy[i];

    float det = cxx * cyy - cxy * cxy;
    if (det <= 0.f) return;

    float trace   = cxx + cyy;
    float temp    = fmaxf(0.f, trace * trace - 4.f * det);
    float lambda1 = 0.5f * (trace - sqrtf(temp));
    float lambda2 = 0.5f * (trace + sqrtf(temp));
    
    float pixel_ndc = fmaxf(2.f / screen_width, 2.f / screen_height);
    // cull splats smaller than a pixel in pixel space
    if (3.f * sqrtf(lambda2 * 2.f) < pixel_ndc) return;
    // cull splats thinner than a pixel in pixel space
    // if (3.f * sqrtf(lambda1 * 2.f) < pixel_ndc) return;
    float max_radius = 3.f * sqrtf(max(lambda1, lambda2));

    // bounding box in NDC
    float min_x = x - max_radius;
    float max_x = x + max_radius;
    float min_y = y - max_radius;
    float max_y = y + max_radius;

    auto ndcToTileX = [&](float v) -> int {
        return (int)floorf((v + 1.f) * 0.5f * num_tiles_x);
    };
    auto ndcToTileY = [&](float v) -> int {
        return (int)floorf((v + 1.f) * 0.5f * num_tiles_y);
    };

    int tx0 = max(ndcToTileX(min_x), 0);
    int tx1 = min(ndcToTileX(max_x), num_tiles_x - 1);
    int ty0 = max(ndcToTileY(min_y), 0);
    int ty1 = min(ndcToTileY(max_y), num_tiles_y - 1);

    if ((tx0 > tx1) || (ty0 > ty1)) return; // off-screen
    atomicAdd(visible_count, 1u);
    // emit one pair per overlapping tile
    for (int ty = ty0; ty <= ty1; ty++)
    {
        for (int tx = tx0; tx <= tx1; tx++)
        {
            uint32_t tile_id = (uint32_t)(ty * num_tiles_x + tx);
            uint64_t key     = makeKey(tile_id, z);

            // claim a slot
            uint32_t slot = atomicAdd(pair_count, 1u);
            if (slot >= (uint32_t)max_pairs)
            {
                // overflow, back off (shouldn't happen with generous max_pairs)
                atomicSub(pair_count, 1u);
                return;
            }
            keys[slot]   = key;
            values[slot] = (uint32_t)i;
        }
    }
}

/* ===== ===== Build Tile Ranges ===== ===== */

/**
 * Scans sorted keys to find start/end index of each tile's splat run.
 * 
 * One thread is launched per key-value pair.
 * 
 * @param[in] keys_sorted   Input keys on the device, sorted by tile_id and depth
 * @param[out] tile_ranges  pair of uint32 per tile
 *                          {start, end} indices into sorted arrays.
 *                          int2.x = start, int2.y = end (exclusive)
 * @param[in] pair_count    Number of valid pairs in the sorted arrays
 * @param[in] num_tiles     Total number of tiles on the screen
 */
__global__ void buildTileRangesKernel(
    const uint64_t *keys_sorted,
    int2 *tile_ranges,
    uint32_t pair_count,
    int num_tiles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint32_t)i >= pair_count) return;

    int tile_id = (int)(keys_sorted[i] >> 32);
    // invalid tile_id (should never happen if tile assignment is correct)
    if (tile_id < 0 || tile_id >= num_tiles) return;

    // first pair in this tile
    if (i == 0 || (int)(keys_sorted[i - 1] >> 32) != tile_id)
        tile_ranges[tile_id].x = i; // start (inclusive)

    // last pair in this tile
    if (i == (int)pair_count - 1 || (int)(keys_sorted[i + 1] >> 32) != tile_id)
        tile_ranges[tile_id].y = i + 1; // end (exclusive)
}

/* ===== ===== Forward Kernel ===== ===== */

/**
 * Iterates over sorted splats in each tile, alpha compositing front-to-back.
 * Outputs final color, final transmittance, and number of contributing splats
 * for each pixel.
 * 
 * One thread is launched per pixel.
 * 
 * @param[in] ndc_x         X positions of splats in NDC [-1, 1]
 * @param[in] ndc_y         Y positions of splats in NDC [-1, 1]
 * @param[in] ndc_z         Z positions of splats in NDC [-1, 1]
 * @param[in] ndc_cxx       Covariance matrix element xx
 * @param[in] ndc_cxy       Covariance matrix element xy
 * @param[in] ndc_cyy       Covariance matrix element yy
 * @param[in] ndc_R         Splat color R
 * @param[in] ndc_G         Splat color G
 * @param[in] ndc_B         Splat color B
 * @param[in] ndc_A         Splat opacity
 * @param[in] values_sorted Sorted splat IDs on the device, sorted by tile_id and depth
 * @param[in] tile_ranges   Tile ranges for each tile
 * @param[out] pixels       Output pixel colors (RGB)
 * @param[out] T_final      Final T values for backward pass
 * @param[out] n_contrib    Number of contributing splats per pixel
 * @param[in] num_tiles_x   Number of tiles in x direction
 * @param[in] num_tiles_y   Number of tiles in y direction
 * @param[in] screen_width  Screen width in pixels
 * @param[in] screen_height Screen height in pixels
 */
__global__ void rasterizeKernel(
    // input splat data
    const float *__restrict__ ndc_x,
    const float *__restrict__ ndc_y,
    const float *__restrict__ ndc_cxx,
    const float *__restrict__ ndc_cxy,
    const float *__restrict__ ndc_cyy,
    const float *__restrict__ ndc_R,
    const float *__restrict__ ndc_G,
    const float *__restrict__ ndc_B,
    const float *__restrict__ ndc_A,
    // input tile data
    const uint32_t *__restrict__ values_sorted,
    const int2     *__restrict__ tile_ranges,
    // output
    float *pixels,
    float *T_final,
    int   *n_contrib,
    // screen config
    int num_tiles_x, int num_tiles_y,
    int screen_width, int screen_height)
{
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= screen_width || pixel_y >= screen_height) return;

    int tile_x   = (pixel_x * num_tiles_x) / screen_width;
    int tile_y   = (pixel_y * num_tiles_y) / screen_height;
    int tile_idx = tile_y * num_tiles_x + tile_x;
    int2 range   = tile_ranges[tile_idx];

    float x_ndc = (2.f * (pixel_x + 0.5f) / screen_width)  - 1.f;
    float y_ndc = (2.f * (pixel_y + 0.5f) / screen_height) - 1.f;

    float C_R = 0.f, C_G = 0.f, C_B = 0.f;
    float T   = 1.f;
    int contrib = 0;
    for (int idx = range.x; idx < range.y; idx++)
    {
        uint32_t sid = values_sorted[idx];

        float dx  = x_ndc - ndc_x[sid];
        float dy  = y_ndc - ndc_y[sid];
        float cxx = ndc_cxx[sid];
        float cxy = ndc_cxy[sid];
        float cyy = ndc_cyy[sid];

        float det = cxx * cyy - cxy * cxy;
        if (det <= 0.f) continue;

        float inv_det = 1.f / det;
        float inv_cxx =  cyy * inv_det;
        float inv_cxy = -cxy * inv_det;
        float inv_cyy =  cxx * inv_det;

        // mahalanobis distance squared
        float dist2 = dx*dx*inv_cxx + 2.f*dx*dy*inv_cxy + dy*dy*inv_cyy;
        if (dist2 > 9.f) continue; // 3-sigma cutoff

        float alpha = fminf(0.99f, ndc_A[sid] * expf(-0.5f * dist2));
        if (alpha < ALPHA_THRESHOLD) continue;

        C_R += ndc_R[sid] * alpha * T;
        C_G += ndc_G[sid] * alpha * T;
        C_B += ndc_B[sid] * alpha * T;
        T   *= (1.f - alpha);
        contrib++;

        if (T < T_THRESHOLD) break;
    }

    int pidx = pixel_y * screen_width + pixel_x;
    pixels   [pidx * 3 + 0] = C_R;
    pixels   [pidx * 3 + 1] = C_G;
    pixels   [pidx * 3 + 2] = C_B;
    T_final  [pidx]         = T;
    n_contrib[pidx]         = contrib;
}

/* ===== ===== Backward Kernel ===== ===== */

/**
 * Backward rasterization kernel. One thread per pixel.
 *
 * Receives loss from the output (dL/d_pixels). Walks compositing
 * in reverse using `T_final` division trick. Accumulates gradients
 * into Splat2DGrads via `atomicAdd`.
 * 
 * One thread is launched per pixel.
 * 
 * - N = number of splats
 * 
 * - M = number of [Key, Value] pairs.
 * 
 * Key is [tile ID | depth] and Value is splat ID.
 * M is approximately N but can be larger due to same splat in multiple tiles.
 * 

 * @param[in] ndc_x             X positions of splats in NDC [N]
 * @param[in] ndc_y             Y positions of splats in NDC [N]
 * @param[in] ndc_z             Z positions of splats in NDC [N]
 * @param[in] ndc_cxx           Covariance matrix element xx [N]
 * @param[in] ndc_cxy           Covariance matrix element xy [N]
 * @param[in] ndc_cyy           Covariance matrix element yy [N]
 * @param[in] ndc_R             Splat color R [N]
 * @param[in] ndc_G             Splat color G [N]
 * @param[in] ndc_B             Splat color B [N]
 * @param[in] ndc_A             Splat opacity [N]
 * @param[in] grad_o            dL/d_pixels from Loss Layer [H*W*3]
 * @param[in] values_sorted   Sorted contributing splat indices
 *                              for each tile [M]
 * @param[in] tile_ranges     Tile start/end indices
 *                              in d_values_sorted [num_tiles]
 * @param[in] pixels          Rendered pixel colors [H*W*3]
 * @param[in] T_final         Final transmittance values [H*W]
 * @param[in] n_contrib       Number of contributing splats
 *                              for each pixel [H*W]
 * @param[out] grad_i_ndc_x     [N]
 * @param[out] grad_i_ndc_y     [N]
 * @param[out] grad_i_ndc_cxx   [N]
 * @param[out] grad_i_ndc_cxy   [N]
 * @param[out] grad_i_ndc_cyy   [N]
 * @param[out] grad_i_R         [N]
 * @param[out] grad_i_G         [N]
 * @param[out] grad_i_B         [N]
 * @param[out] grad_i_A         [N]
 * @param[in] num_tiles_x       Number of tiles in x direction
 * @param[in] num_tiles_y       Number of tiles in y direction
 * @param[in] screen_width      Screen width in pixels
 * @param[in] screen_height     Screen height in pixels
 */
__global__ void backwardKernel(
    // read-only splat data
    const float *__restrict__ ndc_x,
    const float *__restrict__ ndc_y,
    const float *__restrict__ ndc_cxx,
    const float *__restrict__ ndc_cxy,
    const float *__restrict__ ndc_cyy,
    const float *__restrict__ ndc_R,
    const float *__restrict__ ndc_G,
    const float *__restrict__ ndc_B,
    const float *__restrict__ ndc_A,
    // read-only tile data
    const uint32_t *__restrict__ values_sorted,
    const int2     *__restrict__ tile_ranges,
    const float    *__restrict__ pixels,
    const float    *__restrict__ T_final,
    const int      *__restrict__ n_contrib,
    // gradient output
    const float *__restrict__ grad_o,
    // gradient input
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
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= screen_width || pixel_y >= screen_height) return;

    int pixel_idx = pixel_y * screen_width + pixel_x;

    int tile_x   = (pixel_x * num_tiles_x) / screen_width;
    int tile_y   = (pixel_y * num_tiles_y) / screen_height;
    int tile_idx = tile_y * num_tiles_x + tile_x;
    int2 tile_range = tile_ranges[tile_idx];
    int  n_contr  = n_contrib[pixel_idx];

    float x_ndc = (2.f * pixel_x + 1.f) / screen_width  - 1.f;
    float y_ndc = (2.f * pixel_y + 1.f) / screen_height - 1.f;

    // --- forward sweep: collect contributing splat indices in forward order ---
    // you better not have 256+ splats stacked on a single pixel.
    const int MAX_CONTRIB = 256;
    uint32_t contrib_indices[MAX_CONTRIB];
    int contributed_splats = 0;
    for (int idx = tile_range.x;
         idx < tile_range.y &&
         contributed_splats < n_contr &&
         contributed_splats < MAX_CONTRIB;
         idx++)
    {
        uint32_t splat_id = values_sorted[idx];

        float dx  = x_ndc - ndc_x[splat_id];
        float dy  = y_ndc - ndc_y[splat_id];
        float cxx = ndc_cxx[splat_id];
        float cxy = ndc_cxy[splat_id];
        float cyy = ndc_cyy[splat_id];
        float det = cxx * cyy - cxy * cxy;
        if (det <= 0.f) continue;

        float inv_det = 1.f / det;
        float inv_cxx =  cyy * inv_det;
        float inv_cxy = -cxy * inv_det;
        float inv_cyy =  cxx * inv_det;
        float dist2 = dx*dx*inv_cxx + 2.f*dx*dy*inv_cxy + dy*dy*inv_cyy;
        if (dist2 > 9.f) continue;

        float alpha = fminf(0.99f, ndc_A[splat_id] * expf(-0.5f * dist2));
        if (alpha < ALPHA_THRESHOLD) continue;

        contrib_indices[contributed_splats++] = splat_id;
    }

    // receive dL/d_pixels from loss layer
    float dL_dCr = grad_o[3 * pixel_idx + 0];
    float dL_dCg = grad_o[3 * pixel_idx + 1];
    float dL_dCb = grad_o[3 * pixel_idx + 2];

    // --- backward pass: walk in reverse using T_final division trick ---
    // start from T_final and recover T_i by dividing out (1 - alpha_i)
    // in reverse order, going from last splat to the first.
    float T_after  = T_final[pixel_idx];
    float C_back_r = 0.f, C_back_g = 0.f, C_back_b = 0.f;

    for (int i = contributed_splats - 1; i >= 0; i--)
    {
        uint32_t splat_id = contrib_indices[i];

        // recompute geometry for this splat
        float dx  = x_ndc - ndc_x[splat_id];
        float dy  = y_ndc - ndc_y[splat_id];
        float cxx = ndc_cxx[splat_id];
        float cxy = ndc_cxy[splat_id];
        float cyy = ndc_cyy[splat_id];
        float det = cxx * cyy - cxy * cxy;
        float inv_det = 1.f / det;
        float inv_cxx =  cyy * inv_det;
        float inv_cxy = -cxy * inv_det;
        float inv_cyy =  cxx * inv_det;
        float dist2 = dx*dx*inv_cxx + 2.f*dx*dy*inv_cxy + dy*dy*inv_cyy;
        float g     = expf(-0.5f * dist2);
        float alpha = fminf(0.99f, ndc_A[splat_id] * g);

        // recover: T_before = T_after / (1 - alpha)
        float T_before = T_after / fmaxf(1.f - alpha, 1e-6f);

        float cR = ndc_R[splat_id];
        float cG = ndc_G[splat_id];
        float cB = ndc_B[splat_id];

        // dL/dC
        atomicAdd(&grad_i_R[splat_id], alpha * T_before * dL_dCr);
        atomicAdd(&grad_i_G[splat_id], alpha * T_before * dL_dCg);
        atomicAdd(&grad_i_B[splat_id], alpha * T_before * dL_dCb);

        // dL/dalpha
        float dL_dalpha = T_before * (
            (cR - C_back_r) * dL_dCr +
            (cG - C_back_g) * dL_dCg +
            (cB - C_back_b) * dL_dCb
        );

        // dL/dopacity
        atomicAdd(&grad_i_A[splat_id], dL_dalpha * g);

        // dL/ddist2
        float raw_alpha  = ndc_A[splat_id] * g;
        float dL_ddist2 = (raw_alpha < 0.99f) ? dL_dalpha * (-0.5f) * alpha : 0.f;

        // dL/dpos
        float ddist2_dpos_x = -2.f * (dx * inv_cxx + dy * inv_cxy);
        float ddist2_dpos_y = -2.f * (dx * inv_cxy + dy * inv_cyy);
        atomicAdd(&grad_i_ndc_x[splat_id], dL_ddist2 * ddist2_dpos_x);
        atomicAdd(&grad_i_ndc_y[splat_id], dL_ddist2 * ddist2_dpos_y);

        // dL/dcovariance
        float dx2   = dx * dx;
        float dxdy  = dx * dy;
        float dy2   = dy * dy;
        float inv_det2 = inv_det * inv_det;

        float ddist2_dcxx = -(dx*cyy - dy*cxy) * (dx*cyy - dy*cxy) * inv_det2;
        float ddist2_dcyy = -(dy*cxx - dx*cxy) * (dy*cxx - dx*cxy) * inv_det2;
        float ddist2_dcxy =  2.f * (cxy*(dx2*cyy + dy2*cxx) - dxdy*(cxx*cyy + cxy*cxy)) * inv_det2;
        atomicAdd(&grad_i_ndc_cxx[splat_id], dL_ddist2 * ddist2_dcxx);
        atomicAdd(&grad_i_ndc_cxy[splat_id], dL_ddist2 * ddist2_dcxy);
        atomicAdd(&grad_i_ndc_cyy[splat_id], dL_ddist2 * ddist2_dcyy);

        // update C_back and T_after for next iteration
        C_back_r += cR * alpha * T_before;
        C_back_g += cG * alpha * T_before;
        C_back_b += cB * alpha * T_before;
        T_after   = T_before;
    }
}

/* ===== ===== Utils ===== ===== */
uint32_t RasterizeLayer::getVisibleCount()
{
    uint32_t c = 0;
    cudaMemcpy(&c, d_visible_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return c;
}

/* ===== ===== Lifecycle ===== ===== */

void RasterizeLayer::allocate(int width, int height, int _num_tiles_x, int _num_tiles_y,
                               int _max_pairs, int count)
{
    screen_width = width;
    screen_height = height;
    num_pixels   = width * height;
    num_tiles_x = _num_tiles_x;
    num_tiles_y = _num_tiles_y;
    max_pairs   = _max_pairs;
    int numTiles = num_tiles_x * num_tiles_y;

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

    screen_width = new_width;
    screen_height = new_height;
    num_pixels = screen_width * screen_height;

    d_out_pixels.allocate(num_pixels * 3);
    d_T_final.allocate   (num_pixels);
    d_n_contrib.allocate (num_pixels);
}

/* ===== ===== Forward / Backward ===== ===== */

void RasterizeLayer::forward()
{
    int numTiles = num_tiles_x * num_tiles_y;

    // clear per-frame buffers
    cudaMemset(d_out_pixels,    0, num_pixels * 3 * sizeof(float));
    cudaMemset(d_pair_count,    0, sizeof(uint32_t));
    cudaMemset(d_visible_count, 0, sizeof(uint32_t));
    cudaMemset(d_tile_ranges,   0, numTiles  * sizeof(int2));

    // tile assign
    {
        int blocks  = (in->count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        tileAssignKernel<<<blocks, BLOCK_SIZE>>>(
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

    // readback pair count (needed for sort)
    // one big uint32_t copy back,
    // I'm genuinely concerned about the overhead of this
    uint32_t pair_count = 0;
    cudaMemcpy(&pair_count, d_pair_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (pair_count == 0) return;

    // sort pairs by key (tile_id | depth)
    {
        size_t required = 0;
        // query required temp storage size
        // this call does no sorting
        cub::DeviceRadixSort::SortPairs(
            nullptr, required,
            d_keys.ptr,   d_keys_sorted.ptr,
            d_values.ptr, d_values_sorted.ptr,
            (int)pair_count);

        // reallocate temp buffer if needed
        if (required > sort_temp_bytes)
        {
            d_sort_temp.allocate(required);
            sort_temp_bytes = required;
        }

        // actual sort
        cub::DeviceRadixSort::SortPairs(
            (void *)d_sort_temp.ptr, required,
            d_keys.ptr,   d_keys_sorted.ptr,
            d_values.ptr, d_values_sorted.ptr,
            (int)pair_count);
    }

    // build tile ranges
    {
        int blocks  = ((int)pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        buildTileRangesKernel<<<blocks, BLOCK_SIZE>>>(
            d_keys_sorted, d_tile_ranges, pair_count, numTiles);
        CUDA_SYNC_CHECK();
    }

    // forward rasterize
    {
        dim3 threads(16, 16);
        dim3 blocks(
            (screen_width  + threads.x - 1) / threads.x,
            (screen_height + threads.y - 1) / threads.y
        );
        rasterizeKernel<<<blocks, threads>>>(
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
    dim3 threads(16, 16);
    dim3 blocks(
        (screen_width  + threads.x - 1) / threads.x,
        (screen_height + threads.y - 1) / threads.y
    );
    backwardKernel<<<blocks, threads>>>(
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
