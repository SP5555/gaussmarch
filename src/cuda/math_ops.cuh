#pragma once
#include <cuda_runtime.h>

/**
 * @file math_ops.cuh
 * @brief Device-side math helpers shared across CUDA kernels.
 *
 * All functions are __device__ __forceinline__ and carry zero call overhead.
 */

/* ===== ===== Quaternion ===== ===== */

/**
 * @brief Normalized quaternion together with its reciprocal original norm.
 *
 * Bundling @c inv_norm avoids recomputing it in the backward Jacobian:
 * @f$ \nabla q_\text{raw} = \text{inv\_norm} \cdot
 *     (\nabla q_\text{norm} - (\nabla q_\text{norm} \cdot q_\text{norm})\,q_\text{norm}) @f$.
 */
struct NormQuat {
    float w, x, y, z; ///< Unit quaternion components.
    float inv_norm;   ///< Reciprocal of the input's L2-norm: @f$ 1 / \|q_\text{raw}\| @f$.
};

/**
 * @brief Normalizes a raw quaternion to unit length.
 *
 * Guards against zero-norm input: @c fmaxf(dot, 1e-12f) before @c rsqrtf
 * prevents @c rsqrtf(0) = inf when the optimizer produces an all-zero quaternion.
 *
 * @param w,x,y,z  Raw (possibly un-normalized) quaternion components.
 * @return         Unit quaternion and its original reciprocal norm.
 */
__device__ __forceinline__ NormQuat normalizeQuat(float w, float x, float y, float z)
{
    float inv = rsqrtf(fmaxf(w*w + x*x + y*y + z*z, 1e-12f));
    return { w*inv, x*inv, y*inv, z*inv, inv };
}

/* ===== ===== Rotation Matrix ===== ===== */

/**
 * @brief 3x3 rotation matrix in row-major named fields.
 *
 * Field naming: r<row><col>, so the matrix is:
 * @code
 *   | r00  r01  r02 |
 *   | r10  r11  r12 |
 *   | r20  r21  r22 |
 * @endcode
 */
struct RotMat3 {
    float r00, r01, r02;
    float r10, r11, r12;
    float r20, r21, r22;
};

/**
 * @brief Builds a 3x3 rotation matrix from a unit quaternion.
 *
 * The input @b must be unit-length; call @ref normalizeQuat first.
 * Uses the standard Rodrigues formula:
 * @code
 *   R = | 1-2(yy+zz)   2(xy-wz)   2(xz+wy) |
 *       |   2(xy+wz) 1-2(xx+zz)   2(yz-wx) |
 *       |   2(xz-wy)   2(yz+wx) 1-2(xx+yy) |
 * @endcode
 *
 * @param qw,qx,qy,qz  Unit quaternion components (w, x, y, z).
 * @return             Corresponding 3x3 rotation matrix.
 */
__device__ __forceinline__ RotMat3 quatToRotMat(float qw, float qx, float qy, float qz)
{
    return {
        1.f - 2.f*(qy*qy + qz*qz),       2.f*(qx*qy - qw*qz),       2.f*(qx*qz + qw*qy),
              2.f*(qx*qy + qw*qz), 1.f - 2.f*(qx*qx + qz*qz),       2.f*(qy*qz - qw*qx),
              2.f*(qx*qz - qw*qy),       2.f*(qy*qz + qw*qx), 1.f - 2.f*(qx*qx + qy*qy)
    };
}

/* ===== ===== View Direction ===== ===== */

/**
 * @brief Computes the normalized direction from a position toward the camera.
 *
 * Returns @c normalize(cam - pos). Guards against zero-length vectors with
 * @c fmaxf(., 1e-12f), consistent with the 3DGS convention of treating the
 * view direction as constant in the backward pass (no grad flows through @c pos here).
 *
 * @param cam_x,cam_y,cam_z  Camera world-space position.
 * @param pos_x,pos_y,pos_z  Splat world-space position.
 * @return                   Unit direction vector (splat -> camera).
 */
__device__ __forceinline__ float3 viewDir(
    float cam_x, float cam_y, float cam_z,
    float pos_x, float pos_y, float pos_z)
{
    float dx = cam_x - pos_x, dy = cam_y - pos_y, dz = cam_z - pos_z;
    float inv = rsqrtf(fmaxf(dx*dx + dy*dy + dz*dz, 1e-12f));
    return make_float3(dx*inv, dy*inv, dz*inv);
}
