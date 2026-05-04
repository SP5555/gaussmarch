#include "persp_project_layer.h"
#include <cuda_runtime.h>
#include <float.h>
#include <glm/gtc/type_ptr.hpp>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"

__constant__ float d_pv[16]; // device copy of PV matrix (column-major, GLM layout)

/* ===== ===== Kernels ===== ===== */

/**
 * @brief Forward pass: world-space Splat3D -> NDC-space Splat2D via perspective projection.
 *
 * PV matrix is column-major (GLM layout), stored in constant memory as d_pv[col*4+row].
 *
 * Position (perspective divide):
 *   clip    = PV * [x y z 1]^T
 *   ndc.xyz = clip.xyz / clip.w
 *
 * 2D covariance via 2x3 Jacobian of NDC w.r.t. world position:
 *   J[0,k] = (pv[k*4+0]*c_w - pv[3*4+0]*c_x) / c_w^2   (d(ndc_x)/d(world_k))
 *   J[1,k] = (pv[k*4+1]*c_w - pv[3*4+1]*c_y) / c_w^2   (d(ndc_y)/d(world_k))
 *   Cov_2D = J * Cov_3D * J^T
 *
 * @note The Jacobian is 2x3, not 3x3: ndc_z is used only for depth-sorting in the
 *       rasterizer and is not differentiated through the covariance projection.
 * @note Splats with c_w <= 0 (behind the camera) are culled: ndc_z is set to
 *       FLT_MAX and the rasterizer skips them in tile assignment.
 *
 * @param[in]  i_px/y/z                 World-space position (SoA).
 * @param[in]  i_cxx/cxy/cxz/cyy/cyz/czz    Upper-triangle 3D covariance.
 * @param[in]  i_colr/g/b/a             Color and opacity (passed through unchanged).
 * @param[out] o_px/y/z                 NDC position.
 * @param[out] o_cxx/cxy/cyy            Upper-triangle 2D projected covariance.
 * @param[out] o_colr/g/b/a             Pass-through color and opacity.
 * @param[in]  count                    Number of splats; one thread per splat.
 */
__global__ void perspProjectForwardKernel(
    // inputs (world space)
    const float *__restrict__ i_px,
    const float *__restrict__ i_py,
    const float *__restrict__ i_pz,
    const float *__restrict__ i_cxx,
    const float *__restrict__ i_cxy,
    const float *__restrict__ i_cxz,
    const float *__restrict__ i_cyy,
    const float *__restrict__ i_cyz,
    const float *__restrict__ i_czz,
    const float *__restrict__ i_colr,
    const float *__restrict__ i_colg,
    const float *__restrict__ i_colb,
    const float *__restrict__ i_cola,
    // outputs (NDC space)
    float *o_px,
    float *o_py,
    float *o_pz,
    float *o_cxx,
    float *o_cxy,
    float *o_cyy,
    float *o_colr,
    float *o_colg,
    float *o_colb,
    float *o_cola,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float x = i_px[i], y = i_py[i], z = i_pz[i];

    /*
        PV is column major
        PV = [ pv[0]  pv[4]  pv[8]  pv[12] ]
             [ pv[1]  pv[5]  pv[9]  pv[13] ]
             [ pv[2]  pv[6]  pv[10] pv[14] ]
             [ pv[3]  pv[7]  pv[11] pv[15] ]

        clip_pos = PV * [x y z 1]^T
                 = [ clip_x  clip_y  clip_z  clip_w ]^T
    */
    float clip_x = d_pv[0]*x + d_pv[4]*y + d_pv[8]*z  + d_pv[12];
    float clip_y = d_pv[1]*x + d_pv[5]*y + d_pv[9]*z  + d_pv[13];
    float clip_z = d_pv[2]*x + d_pv[6]*y + d_pv[10]*z + d_pv[14];
    float clip_w = d_pv[3]*x + d_pv[7]*y + d_pv[11]*z + d_pv[15];

    // cull splats behind camera
    if (clip_w <= 0.f)
    {
        o_pz[i] = FLT_MAX; // rasterizer will skip this
        return;
    }
    float inv_w  = 1.f / clip_w;
    float inv_w2 = inv_w * inv_w;

    /*
        NDC position (perspective divide)

        ndc_pos = [ clip_x/clip_w  clip_y/clip_w  clip_z/clip_w ]^T
    */
    o_px[i] = clip_x * inv_w;
    o_py[i] = clip_y * inv_w;
    o_pz[i] = clip_z * inv_w;

    /*
        Jacobian of perspective projection (d(ndc)/d(world)) is 2x3:

        J = [ d(ndc_x)/dx  d(ndc_x)/dy  d(ndc_x)/dz ]
            [ d(ndc_y)/dx  d(ndc_y)/dy  d(ndc_y)/dz ]
          = [ d(clip_x/clip_w)/dx  d(clip_x/clip_w)/dy  d(clip_x/clip_w)/dz ]
            [ d(clip_y/clip_w)/dx  d(clip_y/clip_w)/dy  d(clip_y/clip_w)/dz ]
          = [ (pv[0]*clip_w-clip_x*pv[3])/clip_w^2  (pv[4]*clip_w-clip_x*pv[7])/clip_w^2  (pv[8]*clip_w-clip_x*pv[11])/clip_w^2 ]
            [ (pv[1]*clip_w-clip_y*pv[3])/clip_w^2  (pv[5]*clip_w-clip_y*pv[7])/clip_w^2  (pv[9]*clip_w-clip_y*pv[11])/clip_w^2 ]

    */
    float j00 = (d_pv[0] * clip_w - clip_x * d_pv[ 3]) * inv_w2;
    float j01 = (d_pv[4] * clip_w - clip_x * d_pv[ 7]) * inv_w2;
    float j02 = (d_pv[8] * clip_w - clip_x * d_pv[11]) * inv_w2;

    float j10 = (d_pv[1] * clip_w - clip_y * d_pv[ 3]) * inv_w2;
    float j11 = (d_pv[5] * clip_w - clip_y * d_pv[ 7]) * inv_w2;
    float j12 = (d_pv[9] * clip_w - clip_y * d_pv[11]) * inv_w2;

    // 3D covariance (symmetric, upper triangle)
    float cxx = i_cxx[i], cxy = i_cxy[i], cxz = i_cxz[i];
    float                 cyy = i_cyy[i], cyz = i_cyz[i];
    float                                 czz = i_czz[i];

    // J * Cov3D
    float js00 = j00*cxx + j01*cxy + j02*cxz;
    float js01 = j00*cxy + j01*cyy + j02*cyz;
    float js02 = j00*cxz + j01*cyz + j02*czz;

    float js10 = j10*cxx + j11*cxy + j12*cxz;
    float js11 = j10*cxy + j11*cyy + j12*cyz;
    float js12 = j10*cxz + j11*cyz + j12*czz;

    // Cov2D = J * Cov3D * J^T  (symmetric, upper triangle; cyy = J[1]*JS[1])
    o_cxx[i] = j00*js00 + j01*js01 + j02*js02;
    o_cxy[i] = j00*js10 + j01*js11 + j02*js12;
    o_cyy[i] = j10*js10 + j11*js11 + j12*js12;

    // pass-throughs
    o_colr[i] = i_colr[i];
    o_colg[i] = i_colg[i];
    o_colb[i] = i_colb[i];
    o_cola[i] = i_cola[i];
}

/**
 * @brief Backward pass: dL/dSplat2D -> dL/dSplat3D via the perspective Jacobian.
 *
 * Recomputes the 2x3 Jacobian J from the stored world positions rather than saving
 * it from the forward pass; trades a small extra compute cost for avoided memory traffic.
 *
 * Position gradient:    dL/d(world) = J^T * dL/d(ndc)
 * Covariance gradient:  dL/dCov_3D = J^T * dL/dCov_2D_sym * J
 *   entry (p,q): 2*v_xx*(J[0,p]*JS[0,q]) + v_xy*(J[0,p]*JS[1,q] + J[1,p]*JS[0,q])
 *              + 2*v_yy*(J[1,p]*JS[1,q]),  where JS = J * Cov_3D.
 *
 * @note No gradient flows through ndc_z - depth is used only for sorting, not compositing.
 * @note Camera matrices (PV) are treated as constants; no grad flows to the camera.
 * @note Culled splats (clip_w <= 0) receive zero on all gradient outputs.
 *
 * @param[in]  i_px/y/z                 World-space position (recomputed; not saved from forward).
 * @param[in]  i_cxx/cxy/cxz/cyy/cyz/czz  Upper-triangle 3D covariance.
 * @param[in]  grad_o_*                 Upstream gradients (from rasterizer).
 * @param[out] grad_i_*                 Downstream gradients to GaussActivLayer.
 * @param[in]  count                    Number of splats; one thread per splat.
 */
__global__ void perspProjectBackwardKernel(
    // splat3d inputs (recomputed, not saved)
    const float *__restrict__ i_px,
    const float *__restrict__ i_py,
    const float *__restrict__ i_pz,
    const float *__restrict__ i_cxx,
    const float *__restrict__ i_cxy,
    const float *__restrict__ i_cxz,
    const float *__restrict__ i_cyy,
    const float *__restrict__ i_cyz,
    const float *__restrict__ i_czz,
    // gradient output
    const float *__restrict__ grad_o_px,
    const float *__restrict__ grad_o_py,
    const float *__restrict__ grad_o_cxx,
    const float *__restrict__ grad_o_cxy,
    const float *__restrict__ grad_o_cyy,
    const float *__restrict__ grad_o_colr,
    const float *__restrict__ grad_o_colg,
    const float *__restrict__ grad_o_colb,
    const float *__restrict__ grad_o_cola,
    // gradient input
    float *grad_i_px,
    float *grad_i_py,
    float *grad_i_pz,
    float *grad_i_cxx,
    float *grad_i_cxy,
    float *grad_i_cxz,
    float *grad_i_cyy,
    float *grad_i_cyz,
    float *grad_i_czz,
    float *grad_i_colr,
    float *grad_i_colg,
    float *grad_i_colb,
    float *grad_i_cola,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float x = i_px[i], y = i_py[i], z = i_pz[i];

    // recompute clip coords
    float clip_x = d_pv[0]*x + d_pv[4]*y + d_pv[8]*z  + d_pv[12];
    float clip_y = d_pv[1]*x + d_pv[5]*y + d_pv[9]*z  + d_pv[13];
    float clip_w = d_pv[3]*x + d_pv[7]*y + d_pv[11]*z + d_pv[15];

    // skip culled splats
    if (clip_w <= 0.f)
    {
        grad_i_px[i] = 0.f;   grad_i_py[i] = 0.f;   grad_i_pz[i] = 0.f;
        grad_i_cxx[i] = 0.f;  grad_i_cxy[i] = 0.f;  grad_i_cxz[i] = 0.f;
        grad_i_cyy[i] = 0.f;  grad_i_cyz[i] = 0.f;  grad_i_czz[i] = 0.f;
        grad_i_colr[i] = 0.f; grad_i_colg[i] = 0.f;
        grad_i_colb[i] = 0.f; grad_i_cola[i] = 0.f;
        return;
    }

    float inv_w  = 1.f / clip_w;
    float inv_w2 = inv_w * inv_w;

    // recompute Jacobian
    float j00 = (d_pv[0] * clip_w - clip_x * d_pv[ 3]) * inv_w2;
    float j01 = (d_pv[4] * clip_w - clip_x * d_pv[ 7]) * inv_w2;
    float j02 = (d_pv[8] * clip_w - clip_x * d_pv[11]) * inv_w2;

    float j10 = (d_pv[1] * clip_w - clip_y * d_pv[ 3]) * inv_w2;
    float j11 = (d_pv[5] * clip_w - clip_y * d_pv[ 7]) * inv_w2;
    float j12 = (d_pv[9] * clip_w - clip_y * d_pv[11]) * inv_w2;

    float cxx = i_cxx[i], cxy = i_cxy[i], cxz = i_cxz[i];
    float                  cyy = i_cyy[i], cyz = i_cyz[i];
    float                                  czz = i_czz[i];

    // J * Cov3D
    float js00 = j00*cxx + j01*cxy + j02*cxz;
    float js01 = j00*cxy + j01*cyy + j02*cyz;
    float js02 = j00*cxz + j01*cyz + j02*czz;

    float js10 = j10*cxx + j11*cxy + j12*cxz;
    float js11 = j10*cxy + j11*cyy + j12*cyz;
    float js12 = j10*cxz + j11*cyz + j12*czz;

    float dcxx_2D = grad_o_cxx[i];
    float dcxy_2D = grad_o_cxy[i];
    float dcyy_2D = grad_o_cyy[i];

    // dL/dCov_3D = J^T * dL/dCov_2D_sym * J
    // entry (p,q): 2*dcxx_2D*(J[0,p]*js[0,q]) + dcxy_2D*(J[0,p]*js[1,q] + J[1,p]*js[0,q]) + 2*dcyy_2D*(J[1,p]*js[1,q])
    // where js = J * Cov_3D (rows js0x and js1x precomputed above)
    grad_i_cxx[i] = 2.f*dcxx_2D*j00*js00 + dcxy_2D*(j00*js10 + j10*js00) + 2.f*dcyy_2D*j10*js10;
    grad_i_cxy[i] = 2.f*dcxx_2D*j00*js01 + dcxy_2D*(j00*js11 + j10*js01) + 2.f*dcyy_2D*j10*js11;
    grad_i_cxz[i] = 2.f*dcxx_2D*j00*js02 + dcxy_2D*(j00*js12 + j10*js02) + 2.f*dcyy_2D*j10*js12;
    grad_i_cyy[i] = 2.f*dcxx_2D*j01*js01 + dcxy_2D*(j01*js11 + j11*js01) + 2.f*dcyy_2D*j11*js11;
    grad_i_cyz[i] = 2.f*dcxx_2D*j01*js02 + dcxy_2D*(j01*js12 + j11*js02) + 2.f*dcyy_2D*j11*js12;
    grad_i_czz[i] = 2.f*dcxx_2D*j02*js02 + dcxy_2D*(j02*js12 + j12*js02) + 2.f*dcyy_2D*j12*js12;

    // ===== position backward =====
    // ndc = clip / c_w, so d(ndc)/d(world) = J (already computed above)
    // dL/d(world) = J^T * dL/d(ndc)
    float ndc_x = grad_o_px[i];
    float ndc_y = grad_o_py[i];

    grad_i_px[i] = ndc_x * j00 + ndc_y * j10;
    grad_i_py[i] = ndc_x * j01 + ndc_y * j11;
    grad_i_pz[i] = ndc_x * j02 + ndc_y * j12;

    // pass-through gradients
    grad_i_colr[i] = grad_o_colr[i];
    grad_i_colg[i] = grad_o_colg[i];
    grad_i_colb[i] = grad_o_colb[i];
    grad_i_cola[i] = grad_o_cola[i];
}

void PerspProjectLayer::setCamera(const glm::mat4 &view, const glm::mat4 &proj)
{
    glm::mat4 pv = proj * view;
    memcpy(h_pv, glm::value_ptr(pv), 16 * sizeof(float));
    CUDA_CHECK(cudaMemcpyToSymbol(d_pv, h_pv, 16 * sizeof(float)));
}

/* ===== ===== Forward / Backward ===== ===== */

void PerspProjectLayer::forward()
{
    int count   = input->count;
    int blocks  = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    perspProjectForwardKernel<<<blocks, BLOCK_SIZE>>>(
        input->pos_x,   input->pos_y,   input->pos_z,
        input->cov_xx,  input->cov_xy,  input->cov_xz,
        input->cov_yy,  input->cov_yz,  input->cov_zz,
        input->color_r, input->color_g, input->color_b,
        input->opacity,
        output.pos_x,   output.pos_y,   output.pos_z,
        output.cov_xx,  output.cov_xy,  output.cov_yy,
        output.color_r, output.color_g, output.color_b,
        output.opacity,
        count
    );
    CUDA_SYNC_CHECK();
}

void PerspProjectLayer::backward()
{
    int count   = input->count;
    int blocks  = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    perspProjectBackwardKernel<<<blocks, BLOCK_SIZE>>>(
        input->pos_x,  input->pos_y,  input->pos_z,
        input->cov_xx, input->cov_xy, input->cov_xz,
        input->cov_yy, input->cov_yz, input->cov_zz,
        grad_output->grad_pos_x,   grad_output->grad_pos_y,
        grad_output->grad_cov_xx,  grad_output->grad_cov_xy,  grad_output->grad_cov_yy,
        grad_output->grad_color_r, grad_output->grad_color_g, grad_output->grad_color_b,
        grad_output->grad_opacity,
        grad_input.grad_pos_x,   grad_input.grad_pos_y,   grad_input.grad_pos_z,
        grad_input.grad_cov_xx,  grad_input.grad_cov_xy,  grad_input.grad_cov_xz,
        grad_input.grad_cov_yy,  grad_input.grad_cov_yz,  grad_input.grad_cov_zz,
        grad_input.grad_color_r, grad_input.grad_color_g, grad_input.grad_color_b,
        grad_input.grad_opacity,
        count
    );
    CUDA_SYNC_CHECK();
}
