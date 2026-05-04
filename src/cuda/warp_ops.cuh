#pragma once
#include <cuda_runtime.h>

/**
 * @brief Warp-level sum reduction via shuffle-down.
 * @param val Per-lane input value.
 * @return Sum across all 32 lanes (correct only on lane 0).
 */
__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

/**
 * @brief Warp-level max all-reduce via shuffle-xor.
 *
 * Every lane receives the same maximum value across the warp.
 * Used to compute warp_bin_final for the batch-skip optimization in the backward pass.
 *
 * @param val Per-lane input value.
 * @return Maximum value across all 32 lanes (correct on every lane).
 */
__device__ __forceinline__ int warpAllReduceMax(int val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffffu, val, mask));
    return val;
}

/**
 * @brief Block-level sum reduction using warp shuffles.
 *
 * Two-level reduction: each warp reduces to lane 0 via warpReduceSum,
 * warp sums are written to shared memory, then warp 0 reduces those.
 *
 * @tparam BLOCK_SZ Block size; must be a multiple of 32 and <= 1024.
 * @param  val      Per-thread input value.
 * @return Sum across all threads in the block (correct only on thread 0).
 */
template <int BLOCK_SZ>
__device__ __forceinline__ float blockReduceSum(float val)
{
    constexpr int NUM_WARPS = BLOCK_SZ / 32;
    __shared__ float warp_sums[NUM_WARPS];

    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    val = (threadIdx.x < NUM_WARPS) ? warp_sums[lane] : 0.f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
