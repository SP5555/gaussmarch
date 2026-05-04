#pragma once
#include <glm/glm.hpp>

#include "layer.h"
#include "../types/splat3d.h"
#include "../types/splat2d.h"

/**
 * @brief Projects Splat3D in world space to Splat2D in NDC/screen space
 * using a full perspective camera transform.
 *
 * Position:
 *   clip = PV * [x y z 1]^T
 *   NDC  = clip.xyz / clip.w
 *
 * Covariance (2x3 Jacobian of NDC w.r.t. world pos, derived from PV):
 *   J[0] = (PV[col0]*c.w - PV[col3]*c.x) / c.w^2
 *   J[1] = (PV[col1]*c.w - PV[col3]*c.y) / c.w^2
 *   Cov_2D = J * Cov_3D * J^T
 *
 * Call setCamera() every frame before forward() if the camera changes.
 * Backward pass is differentiable w.r.t. Splat3D params (not camera).
 */
class PerspProjectLayer
    : public TypedLayer<Splat3DParams, Splat2DParams, Splat3DGrads, Splat2DGrads>
{
public:
    ~PerspProjectLayer() {}

    void forward()  override;
    void backward() override;

    // call every frame with updated camera matrices
    void setCamera(const glm::mat4 &view, const glm::mat4 &proj);

private:
    float h_pv[16]; // PV = P * V, row-major on device
};
