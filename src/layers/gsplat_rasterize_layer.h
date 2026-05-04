#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "layer.h"
#include "../cuda/cuda_buffer.h"
#include "../types/splat2d.h"

/**
 * Drop-in replacement for RasterizeLayer using gsplat's rasterize algorithm.
 *
 * Algorithm differences from RasterizeLayer:
 *  - 2D thread blocks (TILE_SIZE x TILE_SIZE = 16x16), one thread per pixel.
 *    No pixel batching - avoids replaying the splat stream for large tiles.
 *  - Backward uses warp-level gradient reduction (via shuffle) before global
 *    atomicAdd, reducing atomic contention by ~32x vs per-thread atomics.
 *  - Backward uses last_ids (index of last contributing Gaussian per pixel)
 *    instead of n_contrib, eliminating the forward-sweep contrib_pos[] stack
 *    and its MAX_CONTRIB limit.
 *
 * Interface is identical to RasterizeLayer - swap in pipelines by changing
 * the declared type. num_tiles_x/y passed to allocate() are used for buffer
 * sizing; internally tiles are always TILE_SIZE-aligned.
 */
class GsplatRasterizeLayer
    : public TypedLayer<Splat2DParams, CudaBuffer<float>, Splat2DGrads, CudaBuffer<float>>
{
public:
    ~GsplatRasterizeLayer() {}

    using TypedLayer::allocate;
    void allocate(int width, int height, int count);
    void forward()  override;
    void backward() override;

    void resize(int new_width, int new_height);
    uint32_t getVisibleCount();

private:
    /* ---- forward state saved for backward ---- */
    CudaBuffer<float>   d_render_alphas; // accumulated alpha per pixel [H*W]
    CudaBuffer<int32_t> d_last_ids;      // sorted-list index of last contributing Gaussian [H*W]

    /* ---- tile sorting internals (owned) ---- */
    CudaBuffer<uint64_t> d_isect_ids;
    CudaBuffer<uint32_t> d_gauss_ids;
    CudaBuffer<uint64_t> d_isect_ids_sorted;
    CudaBuffer<uint32_t> d_flatten_ids;
    CudaBuffer<uint32_t> d_n_isects;
    CudaBuffer<uint32_t> d_visible_count;
    CudaBuffer<int2>     d_tile_offsets;
    CudaBuffer<uint8_t>  d_sort_temp;
    size_t sort_temp_bytes = 0;

    /* ---- config ---- */
    int screen_width  = 0;
    int screen_height = 0;
    int num_pixels    = 0;
    int num_tiles_x   = 0;
    int num_tiles_y   = 0;
    int max_isects    = 0;
};
