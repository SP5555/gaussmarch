#include "mse_loss_layer.h"
#include <cuda_runtime.h>
#include <stdexcept>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"
#include "../cuda/warp_ops.cuh"

/* ===== ===== Kernels ===== ===== */

/**
 * Computes per-pixel MSE and accumulates into a scalar.
 * Only called by forward() which is optional (logging only, not every frame).
 * 
 * L = (1 / num_pixels) * sum_i( (pixel_i - target_i)^2 )
 * 
 * @param[in]  pixels       Rendered pixel colors [H*W*3]
 * @param[in]  target       Target pixel colors   [H*W*3]
 * @param[out] loss         Scalar loss output    [1]
 * @param[in]  num_pixels   H * W
 */
__global__ void mseLossKernel(
    const float *__restrict__ pixels,
    const float *__restrict__ target,
    float *loss,
    size_t num_pixels)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.f;
    float scale = 1.f / (float)num_pixels;
    if (i < num_pixels) {
        float dR = pixels[i * 3 + 0] - target[i * 3 + 0];
        float dG = pixels[i * 3 + 1] - target[i * 3 + 1];
        float dB = pixels[i * 3 + 2] - target[i * 3 + 2];
        sum = (dR*dR + dG*dG + dB*dB) * scale;
    }

    sum = blockReduceSum<BLOCK_SIZE>(sum);
    if (threadIdx.x == 0)
        atomicAdd(loss, sum);
}

/**
 * Computes per-pixel MSE gradient
 * 
 * dL/d_pixel_i = (2 / num_pixels) * (pixel_i - target_i)
 *
 * @param[in]  pixels       Rendered pixel colors [H*W*3]
 * @param[in]  target       Target pixel colors   [H*W*3]
 * @param[out] grad_pixels  dL/d_pixels           [H*W*3]
 * @param[in]  num_pixels   H * W
 */
__global__ void mseGradKernel(
    const float *__restrict__ pixels,
    const float *__restrict__ target,
    float *grad_pixels,
    size_t num_pixels)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pixels) return;

    float scale = 2.f / (float)num_pixels;
    grad_pixels[i * 3 + 0] = scale * (pixels[i * 3 + 0] - target[i * 3 + 0]);
    grad_pixels[i * 3 + 1] = scale * (pixels[i * 3 + 1] - target[i * 3 + 1]);
    grad_pixels[i * 3 + 2] = scale * (pixels[i * 3 + 2] - target[i * 3 + 2]);
}

/* ===== ===== Lifecycle ===== ===== */

void MSELossLayer::allocate(int width, int height)
{
    num_pixels = width * height;
    d_grad_pixels.allocate(num_pixels * 3);
    d_loss.allocate(1);
}

void MSELossLayer::zeroGrad()
{
    d_grad_pixels.zero();
}

/* ===== ===== Forward / Backward ===== ===== */

void MSELossLayer::forward()
{
    int blocks  = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    mseLossKernel<<<blocks, BLOCK_SIZE>>>(d_in_pixels, d_target_pixels, d_loss, num_pixels);
    CUDA_SYNC_CHECK();
}

void MSELossLayer::backward()
{
    int blocks  = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mseGradKernel<<<blocks, BLOCK_SIZE>>>(d_in_pixels, d_target_pixels, d_grad_pixels, num_pixels);
    CUDA_SYNC_CHECK();
}

float MSELossLayer::getLoss() const
{
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss;
}