#pragma once

#include "layer.h"
#include "../types/gaussian3d.h"
#include "../types/splat3d.h"

#include <glm/glm.hpp>

/**
 * @brief Computes 3D Gaussian covariance matrices from scale and rotation,
 * and evaluates spherical harmonic color (up to degree 3) for a given view direction.
 *
 * Forward pass: Given log-scale s and quaternion q = (w, x, y, z) (normalized in kernel):
 *
 *   S = diag(exp(s_x), exp(s_y), exp(s_z))
 *   R = rotation matrix from normalized q
 *   Cov = R * S * S^T * R^T
 *
 *   Color is evaluated from SH coefficients using a per-splat view direction:
 *     dir = normalize(camera_pos - splat_pos)
 *   For sh_degree == 0, only the DC term is used (view-independent).
 *
 * Backward pass: Gradients flow back through covariance, SH color, and opacity.
 *   View direction is treated as constant (not differentiated).
 */
class GaussActivLayer
    : public TypedLayer<Gaussian3DParams, Splat3DParams, Gaussian3DGrads, Splat3DGrads>
{
public:
    ~GaussActivLayer() {}

    void forward()  override;
    void backward() override;

    // SH degree: 0 (DC only, default) through 3. Call once after loading.
    void setSHDegree(int degree) { sh_degree = degree; }

    // Camera world position for view-dependent SH. Call each frame before forward().
    void setCameraPosition(const glm::vec3 &pos) { cam_x = pos.x; cam_y = pos.y; cam_z = pos.z; }

    // grad_input needs (n, sh_degree) so we override the default
    void allocateGrad(int n) override { grad_input.allocate(n, sh_degree); }

private:
    int   sh_degree = 0;
    float cam_x = 0.f, cam_y = 0.f, cam_z = 0.f;
};
