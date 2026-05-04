#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "layer.h"
#include "../cuda/cuda_buffer.h"
#include "../types/splat2d.h"

/**
 * @brief Rasterizes 2D Gaussian splats into a pixel buffer,
 * using a tiled forward rendering approach.
 * 
 * Forward pass: Each splat is assigned to all tiles it overlaps.
 * The resulting (tile_id | depth, splat_id) pairs are radix sorted
 * so splats are processed front-to-back per tile. Each pixel then
 * alpha-composites its tile's splats in depth order. Final transmittance
 * `T_final` and contributing splat count n_contrib are saved for the
 * backward pass.
 * 
 * Backward pass: Since the sort is not differentiable, gradients can't
 * flow through the sorting step traditionally. Instead, each pixel re-computes
 * its forward compositing in reverse using the `T_final` division trick
 * (`T_before` is recovered by dividing `T_after` by `(1-alpha)`), avoiding
 * the need to store per-splat intermediate transmittance values.
 * Gradients w.r.t. splat parameters are computed on the fly and
 * accumulated via `atomicAdd` since multiple pixels may contribute
 * back to the same splat.
 */
class RasterizeLayer : public Layer
{
public:
    ~RasterizeLayer() {}

    void allocate(int width, int height, int count);
    void allocateGrad(int count);
    void forward()      override;
    void backward()     override;
    void zero_grad()    override;

    void resize(int new_width, int new_height);

    // wiring
    void setInput(const Splat2DParams *params) { in = params; }
    float        *getOutput()                  { return d_out_pixels; }
    void setGradOutput(const float *grad)      { grad_pixels = grad; }
    Splat2DGrads &getGradInput()               { return grad_in; }

    // debug utils
    uint32_t getVisibleCount();

private:
    /* ---- forward input (not owned) ---- */
    const Splat2DParams *in = nullptr;

    /* ---- forward output (owned) ---- */
    CudaBuffer<float> d_out_pixels; // rendered RGB image [H*W*3]

    /* ---- backward input (not owned) ---- */
    const float *grad_pixels = nullptr; // dL/d_pixels [H*W*3]

    /* ---- backward output (owned) ---- */
    Splat2DGrads grad_in; // dL/d_splat2d

    /* ---- internals (owned) ---- */
    CudaBuffer<float> d_T_final;   // final transmittance per pixel [H*W]
    CudaBuffer<int>   d_n_contrib; // contributing splat count      [H*W]

    CudaBuffer<uint64_t> d_keys;
    CudaBuffer<uint32_t> d_values;
    CudaBuffer<uint64_t> d_keys_sorted;
    CudaBuffer<uint32_t> d_values_sorted;
    CudaBuffer<uint32_t> d_pair_count;
    CudaBuffer<uint32_t> d_visible_count; // how many splats are on the screen
    CudaBuffer<int2>     d_tile_ranges;
    CudaBuffer<uint8_t>  d_sort_temp;
    size_t    sort_temp_bytes = 0;

    /* ---- config ---- */
    int screen_width  = 0;
    int screen_height = 0;
    int num_pixels    = 0;
    int num_tiles_x   = 0;
    int num_tiles_y   = 0;
    int max_pairs     = 0;
};
