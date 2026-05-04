#include "gauss_activ_layer.h"
#include <cuda_runtime.h>
#include <math.h>

#include "../cuda/cuda_check.h"
#include "../cuda/cuda_defs.h"
#include "../cuda/math_ops.cuh"
#include "../utils/sh_consts.h"

/* ===== ===== Kernels ===== ===== */

/**
 * @brief Forward pass: 3D covariance Cov = M*M^T (M = R*S) and SH color evaluation.
 *
 * Scale is exponentiated (s_actual = exp(s_log)); opacity through sigmoid.
 * Cov = R * S * S^T * R^T  (symmetric, stored as upper triangle)
 *
 * SH color evaluation (degrees 0-3, Condon-Shortley convention):
 *   view dir = normalize(cam_pos - splat_pos)
 *   color = sum_over_bands(SH_coeff * Y(dir)) + 0.5, clamped to [0,1]
 *
 * @param[in]  i_px/y/z             World-space position (SoA).
 * @param[in]  i_sx/y/z             Log-scale; exponentiated in kernel.
 * @param[in]  i_rw/x/y/z           Raw quaternion; normalized in kernel.
 * @param[in]  i_sh_dc_r/g/b        Degree-0 SH coefficients per channel.
 * @param[in]  i_sh_rest_r/g/b      Higher-order SH coefficients, layout [band*count+i];
 *                                  may be nullptr when sh_degree == 0.
 * @param[in]  i_logit_a            Pre-sigmoid opacity logit.
 * @param[in]  sh_degree            Active SH degree (0-3); controls which bands are evaluated.
 * @param[in]  cam_x/y/z            Camera world position for view-direction computation.
 * @param[out] o_x/y/z              Pass-through world position.
 * @param[out] o_cxx/cxy/cxz/cyy/cyz/czz  Upper-triangle 3D covariance.
 * @param[out] o_lin_r/g/b          Linear-space RGB, clamped to [0, 1].
 * @param[out] o_a                  Sigmoid opacity.
 * @param[in]  count                Number of splats; one thread per splat.
 */
__global__ void covForwardKernel(
    // inputs: pos, log-scale, raw quaternion,
    //         SH coefficient RGB, logit-opacity
    const float *__restrict__ i_px,
    const float *__restrict__ i_py,
    const float *__restrict__ i_pz,
    const float *__restrict__ i_sx,
    const float *__restrict__ i_sy,
    const float *__restrict__ i_sz,
    const float *__restrict__ i_rw,
    const float *__restrict__ i_rx,
    const float *__restrict__ i_ry,
    const float *__restrict__ i_rz,
    const float *__restrict__ i_sh_dc_r,
    const float *__restrict__ i_sh_dc_g,
    const float *__restrict__ i_sh_dc_b,
    // higher-order SH (may be nullptr if sh_degree == 0)
    // layout: i_sh_rest_r[band * count + i]
    const float *__restrict__ i_sh_rest_r,
    const float *__restrict__ i_sh_rest_g,
    const float *__restrict__ i_sh_rest_b,
    const float *__restrict__ i_logit_a,
    const int sh_degree,
    // camera position for view direction
    const float cam_x, const float cam_y, const float cam_z,
    // outputs: pos + 3D cov + RGBA
    float *o_x,   float *o_y,   float *o_z,
    float *o_cxx, float *o_cxy, float *o_cxz,
    float *o_cyy, float *o_cyz, float *o_czz,
    float *o_lin_r,
    float *o_lin_g,
    float *o_lin_b,
    float *o_a,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // pass-through position
    o_x[i] = i_px[i];
    o_y[i] = i_py[i];
    o_z[i] = i_pz[i];

    // normalize quaternion and build rotation matrix
    NormQuat nq = normalizeQuat(i_rw[i], i_rx[i], i_ry[i], i_rz[i]);
    RotMat3  R  = quatToRotMat(nq.w, nq.x, nq.y, nq.z);

    // actual scale
    float sx = expf(i_sx[i]);
    float sy = expf(i_sy[i]);
    float sz = expf(i_sz[i]);

    // M = R * S
    float m00 = R.r00*sx, m01 = R.r01*sy, m02 = R.r02*sz;
    float m10 = R.r10*sx, m11 = R.r11*sy, m12 = R.r12*sz;
    float m20 = R.r20*sx, m21 = R.r21*sy, m22 = R.r22*sz;

    // Cov = M * M^T
    o_cxx[i] = m00*m00 + m01*m01 + m02*m02;
    o_cxy[i] = m00*m10 + m01*m11 + m02*m12;
    o_cxz[i] = m00*m20 + m01*m21 + m02*m22;
    o_cyy[i] = m10*m10 + m11*m11 + m12*m12;
    o_cyz[i] = m10*m20 + m11*m21 + m12*m22;
    o_czz[i] = m20*m20 + m21*m21 + m22*m22;

    // ===== SH color evaluation =====
    // view direction: splat -> camera, normalized (standard 3DGS convention)
    float3 v = viewDir(cam_x, cam_y, cam_z, i_px[i], i_py[i], i_pz[i]);

    float cr = i_sh_dc_r[i] * SH_C0;
    float cg = i_sh_dc_g[i] * SH_C0;
    float cb = i_sh_dc_b[i] * SH_C0;

    // branches are uniform across the warp (same sh_degree for all threads)
    if (sh_degree >= 1)
    {
        cr += SH_C1 * (-v.y * i_sh_rest_r[0*count+i] + v.z * i_sh_rest_r[1*count+i] - v.x * i_sh_rest_r[2*count+i]);
        cg += SH_C1 * (-v.y * i_sh_rest_g[0*count+i] + v.z * i_sh_rest_g[1*count+i] - v.x * i_sh_rest_g[2*count+i]);
        cb += SH_C1 * (-v.y * i_sh_rest_b[0*count+i] + v.z * i_sh_rest_b[1*count+i] - v.x * i_sh_rest_b[2*count+i]);
    }
    if (sh_degree >= 2)
    {
        float xx = v.x*v.x, yy = v.y*v.y, zz = v.z*v.z;
        float xy = v.x*v.y, yz = v.y*v.z, xz = v.x*v.z;
        cr += SH_C2_0 * xy           * i_sh_rest_r[3*count+i]
            + SH_C2_1 * yz           * i_sh_rest_r[4*count+i]
            + SH_C2_2 * (2*zz-xx-yy) * i_sh_rest_r[5*count+i]
            + SH_C2_3 * xz           * i_sh_rest_r[6*count+i]
            + SH_C2_4 * (xx-yy)      * i_sh_rest_r[7*count+i];
        cg += SH_C2_0 * xy           * i_sh_rest_g[3*count+i]
            + SH_C2_1 * yz           * i_sh_rest_g[4*count+i]
            + SH_C2_2 * (2*zz-xx-yy) * i_sh_rest_g[5*count+i]
            + SH_C2_3 * xz           * i_sh_rest_g[6*count+i]
            + SH_C2_4 * (xx-yy)      * i_sh_rest_g[7*count+i];
        cb += SH_C2_0 * xy           * i_sh_rest_b[3*count+i]
            + SH_C2_1 * yz           * i_sh_rest_b[4*count+i]
            + SH_C2_2 * (2*zz-xx-yy) * i_sh_rest_b[5*count+i]
            + SH_C2_3 * xz           * i_sh_rest_b[6*count+i]
            + SH_C2_4 * (xx-yy)      * i_sh_rest_b[7*count+i];
    }
    if (sh_degree >= 3)
    {
        float xx = v.x*v.x, yy = v.y*v.y, zz = v.z*v.z;
        float xy = v.x*v.y;
        cr += SH_C3_0 * v.y*(3*xx-yy)        * i_sh_rest_r[ 8*count+i]
            + SH_C3_1 * xy*v.z               * i_sh_rest_r[ 9*count+i]
            + SH_C3_2 * v.y*(4*zz-xx-yy)     * i_sh_rest_r[10*count+i]
            + SH_C3_3 * v.z*(2*zz-3*xx-3*yy) * i_sh_rest_r[11*count+i]
            + SH_C3_4 * v.x*(4*zz-xx-yy)     * i_sh_rest_r[12*count+i]
            + SH_C3_5 * v.z*(xx-yy)          * i_sh_rest_r[13*count+i]
            + SH_C3_6 * v.x*(xx-3*yy)        * i_sh_rest_r[14*count+i];
        cg += SH_C3_0 * v.y*(3*xx-yy)        * i_sh_rest_g[ 8*count+i]
            + SH_C3_1 * xy*v.z               * i_sh_rest_g[ 9*count+i]
            + SH_C3_2 * v.y*(4*zz-xx-yy)     * i_sh_rest_g[10*count+i]
            + SH_C3_3 * v.z*(2*zz-3*xx-3*yy) * i_sh_rest_g[11*count+i]
            + SH_C3_4 * v.x*(4*zz-xx-yy)     * i_sh_rest_g[12*count+i]
            + SH_C3_5 * v.z*(xx-yy)          * i_sh_rest_g[13*count+i]
            + SH_C3_6 * v.x*(xx-3*yy)        * i_sh_rest_g[14*count+i];
        cb += SH_C3_0 * v.y*(3*xx-yy)        * i_sh_rest_b[ 8*count+i]
            + SH_C3_1 * xy*v.z               * i_sh_rest_b[ 9*count+i]
            + SH_C3_2 * v.y*(4*zz-xx-yy)     * i_sh_rest_b[10*count+i]
            + SH_C3_3 * v.z*(2*zz-3*xx-3*yy) * i_sh_rest_b[11*count+i]
            + SH_C3_4 * v.x*(4*zz-xx-yy)     * i_sh_rest_b[12*count+i]
            + SH_C3_5 * v.z*(xx-yy)          * i_sh_rest_b[13*count+i]
            + SH_C3_6 * v.x*(xx-3*yy)        * i_sh_rest_b[14*count+i];
    }

    o_lin_r[i] = fminf(fmaxf(cr + 0.5f, 0.f), 1.f);
    o_lin_g[i] = fminf(fmaxf(cg + 0.5f, 0.f), 1.f);
    o_lin_b[i] = fminf(fmaxf(cb + 0.5f, 0.f), 1.f);

    // sigmoid activation on opacity
    o_a[i] = 1.f / (1.f + expf(-i_logit_a[i]));
}

/**
 * @brief Backward pass: gradients from Splat3DParams through covariance, SH, and sigmoid.
 *
 * Recomputes all forward intermediates rather than storing them;
 * trades extra compute for saved memory bandwidth.
 *
 * Gradient through clamp [0,1]: zero when the forward output was at the boundary
 * (read from the saved output color to detect saturation).
 *
 * Normalization Jacobian for q_norm = q_raw / ||q_raw||:
 *   dL/dq_raw = inv_norm * (dL/dq_norm - dot(dL/dq_norm, q_norm) * q_norm)
 *
 * @note View direction is treated as constant (no grad w.r.t. splat position).
 *       This is the standard 3DGS approximation.
 *
 * @param[in]  pos_x/y/z        Forward input world positions.
 * @param[in]  scale_x/y/z      Forward input log-scales.
 * @param[in]  rot_w/x/y/z      Forward input raw quaternions.
 * @param[in]  color_r/g/b      Forward output colors (for clamp gating).
 * @param[in]  opacity          Forward output opacity (for sigmoid backward).
 * @param[in]  sh_degree        Active SH degree (0-3).
 * @param[in]  cam_x/y/z        Camera world position.
 * @param[in]  grad_o_*         Upstream gradients (dL/d outputs of this layer).
 * @param[out] grad_i_*         Downstream gradients (dL/d inputs to optimizer).
 * @param[in]  count            Number of splats; one thread per splat.
 */
__global__ void covBackwardKernel(
    // saved forward inputs (needed to recompute intermediates)
    const float *__restrict__ pos_x,
    const float *__restrict__ pos_y,
    const float *__restrict__ pos_z,
    const float *__restrict__ scale_x,
    const float *__restrict__ scale_y,
    const float *__restrict__ scale_z,
    const float *__restrict__ rot_w,
    const float *__restrict__ rot_x,
    const float *__restrict__ rot_y,
    const float *__restrict__ rot_z,
    // saved forward outputs (for gradient gating)
    const float *__restrict__ color_r,
    const float *__restrict__ color_g,
    const float *__restrict__ color_b,
    const float *__restrict__ opacity,
    const int sh_degree,
    const float cam_x, const float cam_y, const float cam_z,
    // gradient output (from downstream layer)
    const float *__restrict__ grad_o_px,
    const float *__restrict__ grad_o_py,
    const float *__restrict__ grad_o_pz,
    const float *__restrict__ grad_o_cxx,
    const float *__restrict__ grad_o_cxy,
    const float *__restrict__ grad_o_cxz,
    const float *__restrict__ grad_o_cyy,
    const float *__restrict__ grad_o_cyz,
    const float *__restrict__ grad_o_czz,
    const float *__restrict__ grad_o_lin_r,
    const float *__restrict__ grad_o_lin_g,
    const float *__restrict__ grad_o_lin_b,
    const float *__restrict__ grad_o_a,
    // gradient input (to upstream parameters)
    float *grad_i_px,
    float *grad_i_py,
    float *grad_i_pz,
    float *grad_i_sx,
    float *grad_i_sy,
    float *grad_i_sz,
    float *grad_i_rw,
    float *grad_i_rx,
    float *grad_i_ry,
    float *grad_i_rz,
    float *grad_i_sh_dc_r,
    float *grad_i_sh_dc_g,
    float *grad_i_sh_dc_b,
    float *grad_i_sh_rest_r,
    float *grad_i_sh_rest_g,
    float *grad_i_sh_rest_b,
    float *grad_i_logit_a,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // pass-through position gradients
    grad_i_px[i] = grad_o_px[i];
    grad_i_py[i] = grad_o_py[i];
    grad_i_pz[i] = grad_o_pz[i];

    // --- recompute forward values ---
    NormQuat nq = normalizeQuat(rot_w[i], rot_x[i], rot_y[i], rot_z[i]);
    RotMat3  R  = quatToRotMat(nq.w, nq.x, nq.y, nq.z);

    float sx = expf(scale_x[i]);
    float sy = expf(scale_y[i]);
    float sz = expf(scale_z[i]);

    // M = R * S
    float m00 = R.r00*sx, m01 = R.r01*sy, m02 = R.r02*sz;
    float m10 = R.r10*sx, m11 = R.r11*sy, m12 = R.r12*sz;
    float m20 = R.r20*sx, m21 = R.r21*sy, m22 = R.r22*sz;

    // --- dL/dCov ---
    float dxx = grad_o_cxx[i];
    float dxy = grad_o_cxy[i];
    float dxz = grad_o_cxz[i];
    float dyy = grad_o_cyy[i];
    float dyz = grad_o_cyz[i];
    float dzz = grad_o_czz[i];

    // --- dL/dM = 2 * dL/dCov_sym * M ---
    float dm00 = 2.f*(dxx*m00 + dxy*m10 + dxz*m20);
    float dm01 = 2.f*(dxx*m01 + dxy*m11 + dxz*m21);
    float dm02 = 2.f*(dxx*m02 + dxy*m12 + dxz*m22);
    float dm10 = 2.f*(dxy*m00 + dyy*m10 + dyz*m20);
    float dm11 = 2.f*(dxy*m01 + dyy*m11 + dyz*m21);
    float dm12 = 2.f*(dxy*m02 + dyy*m12 + dyz*m22);
    float dm20 = 2.f*(dxz*m00 + dyz*m10 + dzz*m20);
    float dm21 = 2.f*(dxz*m01 + dyz*m11 + dzz*m21);
    float dm22 = 2.f*(dxz*m02 + dyz*m12 + dzz*m22);

    // --- dL/ds_log ---
    grad_i_sx[i] = (dm00*R.r00 + dm10*R.r10 + dm20*R.r20) * sx;
    grad_i_sy[i] = (dm01*R.r01 + dm11*R.r11 + dm21*R.r21) * sy;
    grad_i_sz[i] = (dm02*R.r02 + dm12*R.r12 + dm22*R.r22) * sz;

    // --- dL/dR = dL/dM * S ---
    float dr00 = dm00*sx, dr10 = dm10*sx, dr20 = dm20*sx;
    float dr01 = dm01*sy, dr11 = dm11*sy, dr21 = dm21*sy;
    float dr02 = dm02*sz, dr12 = dm12*sz, dr22 = dm22*sz;

    // --- dL/dq_norm: chain through R(q) ---
    float dqw = 2.f*(
        dr10*nq.z  - dr20*nq.y +
        dr01*(-nq.z) + dr21*nq.x +
        dr02*nq.y  + dr12*(-nq.x));

    float dqx = 2.f*(
        dr10*nq.y  + dr20*nq.z +
        dr01*nq.y  + dr11*(-2.f*nq.x) + dr21*nq.w +
        dr02*nq.z  + dr12*(-nq.w)     + dr22*(-2.f*nq.x));

    float dqy = 2.f*(
        dr00*(-2.f*nq.y) + dr10*nq.x + dr20*(-nq.w) +
        dr01*nq.x +
        dr12*nq.z  + dr21*nq.z + dr22*(-2.f*nq.y) +
        dr02*nq.w);

    float dqz = 2.f*(
        dr00*(-2.f*nq.z) + dr10*nq.w + dr20*nq.x +
        dr01*(-nq.w) + dr11*(-2.f*nq.z) +
        dr12*nq.y    + dr21*nq.y +
        dr02*nq.x);

    // --- project through normalization Jacobian ---
    float dot = dqw*nq.w + dqx*nq.x + dqy*nq.y + dqz*nq.z;
    grad_i_rw[i] = nq.inv_norm * (dqw - dot*nq.w);
    grad_i_rx[i] = nq.inv_norm * (dqx - dot*nq.x);
    grad_i_ry[i] = nq.inv_norm * (dqy - dot*nq.y);
    grad_i_rz[i] = nq.inv_norm * (dqz - dot*nq.z);

    // --- SH color gradients ---
    // Gradient through clamp: zero if output was clamped (at 0 or 1 boundary)
    float dR = (color_r[i] > 0.f && color_r[i] < 1.f) ? grad_o_lin_r[i] : 0.f;
    float dG = (color_g[i] > 0.f && color_g[i] < 1.f) ? grad_o_lin_g[i] : 0.f;
    float dB = (color_b[i] > 0.f && color_b[i] < 1.f) ? grad_o_lin_b[i] : 0.f;

    // DC
    grad_i_sh_dc_r[i] = dR * SH_C0;
    grad_i_sh_dc_g[i] = dG * SH_C0;
    grad_i_sh_dc_b[i] = dB * SH_C0;

    if (sh_degree >= 1)
    {
        // recompute view direction
        float3 v = viewDir(cam_x, cam_y, cam_z, pos_x[i], pos_y[i], pos_z[i]);

        // dL/dSH_lm = dL/dcolor * Y_lm(dir)
        // degree 1
        float y0 = SH_C1 * (-v.y);
        float y1 = SH_C1 * ( v.z);
        float y2 = SH_C1 * (-v.x);
        grad_i_sh_rest_r[0*count+i] = dR * y0;
        grad_i_sh_rest_r[1*count+i] = dR * y1;
        grad_i_sh_rest_r[2*count+i] = dR * y2;
        grad_i_sh_rest_g[0*count+i] = dG * y0;
        grad_i_sh_rest_g[1*count+i] = dG * y1;
        grad_i_sh_rest_g[2*count+i] = dG * y2;
        grad_i_sh_rest_b[0*count+i] = dB * y0;
        grad_i_sh_rest_b[1*count+i] = dB * y1;
        grad_i_sh_rest_b[2*count+i] = dB * y2;

        if (sh_degree >= 2)
        {
            float xx = v.x*v.x, yy = v.y*v.y, zz = v.z*v.z;
            float xy = v.x*v.y, yz = v.y*v.z, xz = v.x*v.z;
            float b3 = SH_C2_0*xy,       b4 = SH_C2_1*yz;
            float b5 = SH_C2_2*(2*zz-xx-yy);
            float b6 = SH_C2_3*xz,       b7 = SH_C2_4*(xx-yy);
            grad_i_sh_rest_r[3*count+i] = dR*b3; grad_i_sh_rest_r[4*count+i] = dR*b4;
            grad_i_sh_rest_r[5*count+i] = dR*b5; grad_i_sh_rest_r[6*count+i] = dR*b6;
            grad_i_sh_rest_r[7*count+i] = dR*b7;
            grad_i_sh_rest_g[3*count+i] = dG*b3; grad_i_sh_rest_g[4*count+i] = dG*b4;
            grad_i_sh_rest_g[5*count+i] = dG*b5; grad_i_sh_rest_g[6*count+i] = dG*b6;
            grad_i_sh_rest_g[7*count+i] = dG*b7;
            grad_i_sh_rest_b[3*count+i] = dB*b3; grad_i_sh_rest_b[4*count+i] = dB*b4;
            grad_i_sh_rest_b[5*count+i] = dB*b5; grad_i_sh_rest_b[6*count+i] = dB*b6;
            grad_i_sh_rest_b[7*count+i] = dB*b7;

            if (sh_degree >= 3)
            {
                float c8  = SH_C3_0*v.y*(3*xx-yy);
                float c9  = SH_C3_1*xy*v.z;
                float c10 = SH_C3_2*v.y*(4*zz-xx-yy);
                float c11 = SH_C3_3*v.z*(2*zz-3*xx-3*yy);
                float c12 = SH_C3_4*v.x*(4*zz-xx-yy);
                float c13 = SH_C3_5*v.z*(xx-yy);
                float c14 = SH_C3_6*v.x*(xx-3*yy);
                grad_i_sh_rest_r[ 8*count+i] = dR*c8;  grad_i_sh_rest_r[ 9*count+i] = dR*c9;
                grad_i_sh_rest_r[10*count+i] = dR*c10; grad_i_sh_rest_r[11*count+i] = dR*c11;
                grad_i_sh_rest_r[12*count+i] = dR*c12; grad_i_sh_rest_r[13*count+i] = dR*c13;
                grad_i_sh_rest_r[14*count+i] = dR*c14;
                grad_i_sh_rest_g[ 8*count+i] = dG*c8;  grad_i_sh_rest_g[ 9*count+i] = dG*c9;
                grad_i_sh_rest_g[10*count+i] = dG*c10; grad_i_sh_rest_g[11*count+i] = dG*c11;
                grad_i_sh_rest_g[12*count+i] = dG*c12; grad_i_sh_rest_g[13*count+i] = dG*c13;
                grad_i_sh_rest_g[14*count+i] = dG*c14;
                grad_i_sh_rest_b[ 8*count+i] = dB*c8;  grad_i_sh_rest_b[ 9*count+i] = dB*c9;
                grad_i_sh_rest_b[10*count+i] = dB*c10; grad_i_sh_rest_b[11*count+i] = dB*c11;
                grad_i_sh_rest_b[12*count+i] = dB*c12; grad_i_sh_rest_b[13*count+i] = dB*c13;
                grad_i_sh_rest_b[14*count+i] = dB*c14;
            }
        }
    }

    // --- sigmoid backward: dL/dlogit = dL/dopacity * s * (1 - s) ---
    float s = opacity[i];
    grad_i_logit_a[i] = grad_o_a[i] * s * (1.f - s);
}

/* ===== ===== Forward / Backward ===== ===== */

void GaussActivLayer::forward()
{
    int count = input->count;
    output.count = count;

    int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    covForwardKernel<<<blocks, BLOCK_SIZE>>>(
        input->pos_x,   input->pos_y,   input->pos_z,
        input->scale_x, input->scale_y, input->scale_z,
        input->rot_w,   input->rot_x,   input->rot_y,   input->rot_z,
        input->sh_dc_r,
        input->sh_dc_g,
        input->sh_dc_b,
        input->sh_num_bands > 0 ? (const float *)input->sh_rest_r : nullptr,
        input->sh_num_bands > 0 ? (const float *)input->sh_rest_g : nullptr,
        input->sh_num_bands > 0 ? (const float *)input->sh_rest_b : nullptr,
        input->logit_opacity,
        sh_degree,
        cam_x, cam_y, cam_z,
        output.pos_x,  output.pos_y,  output.pos_z,
        output.cov_xx, output.cov_xy, output.cov_xz,
        output.cov_yy, output.cov_yz, output.cov_zz,
        output.color_r,
        output.color_g,
        output.color_b,
        output.opacity,
        count
    );
    CUDA_SYNC_CHECK();
}

void GaussActivLayer::backward()
{
    int count = input->count;

    int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    covBackwardKernel<<<blocks, BLOCK_SIZE>>>(
        input->pos_x,   input->pos_y,   input->pos_z,
        input->scale_x, input->scale_y, input->scale_z,
        input->rot_w,   input->rot_x,   input->rot_y,   input->rot_z,
        output.color_r, output.color_g, output.color_b,
        output.opacity,
        sh_degree,
        cam_x, cam_y, cam_z,
        grad_output->grad_pos_x,  grad_output->grad_pos_y,  grad_output->grad_pos_z,
        grad_output->grad_cov_xx, grad_output->grad_cov_xy, grad_output->grad_cov_xz,
        grad_output->grad_cov_yy, grad_output->grad_cov_yz, grad_output->grad_cov_zz,
        grad_output->grad_color_r,
        grad_output->grad_color_g,
        grad_output->grad_color_b,
        grad_output->grad_opacity,
        grad_input.grad_pos_x,   grad_input.grad_pos_y,   grad_input.grad_pos_z,
        grad_input.grad_scale_x, grad_input.grad_scale_y, grad_input.grad_scale_z,
        grad_input.grad_rot_w,   grad_input.grad_rot_x,
        grad_input.grad_rot_y,   grad_input.grad_rot_z,
        grad_input.grad_sh_dc_r,
        grad_input.grad_sh_dc_g,
        grad_input.grad_sh_dc_b,
        grad_input.sh_num_bands > 0 ? (float *)grad_input.grad_sh_rest_r : nullptr,
        grad_input.sh_num_bands > 0 ? (float *)grad_input.grad_sh_rest_g : nullptr,
        grad_input.sh_num_bands > 0 ? (float *)grad_input.grad_sh_rest_b : nullptr,
        grad_input.grad_logit_opacity,
        count
    );
    CUDA_SYNC_CHECK();
}
